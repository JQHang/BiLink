import os
import glob
import logging
import pyarrow.parquet as pq
import torch.utils.data
from typing import Dict, Any, Iterator, List
import pandas as pd

from .base import LoaderBatch

logger = logging.getLogger(__name__)


class SequentialLoader:
    """
    Sequential parquet loader with file-by-file processing.

    Designed for inference datasets where file count may be very large.
    Files are distributed across workers using simple modulo assignment
    and processed sequentially.

    Characteristics:
    - No shuffling (sequential processing)
    - No epoch_size concept (processes all files)
    - File-level modulo distribution across workers
    - Supports marking files as done for resume via lock directory
    """

    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        """
        Initialize sequential loader.

        Args:
            config: Dataset configuration with keys:
                - sample_path: Path to parquet files directory
                - batch_size: Batch size
                - lock_dir: (optional) Directory for lock files
            rank: Current process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.batch_size = config['batch_size']

        sample_path = config['sample_path']

        # Lock directory for tracking completed files
        self.lock_dir = config.get('lock_dir', os.path.join(sample_path, '.locks'))
        os.makedirs(self.lock_dir, exist_ok=True)

        # Get all parquet files
        sample_file_paths = glob.glob(os.path.join(sample_path, "*.parquet"))

        # Filter out already-completed files
        done_files = set()
        if os.path.exists(self.lock_dir):
            for lock_file in os.listdir(self.lock_dir):
                if lock_file.endswith('.done'):
                    done_files.add(lock_file[:-5])  # Remove .done suffix

        sample_file_paths = [
            path for path in sample_file_paths
            if os.path.basename(path) not in done_files
        ]

        # Store sorted file paths
        self.sample_file_paths = sorted(sample_file_paths)

    @property
    def rank_batch_count(self) -> int:
        """Not applicable for sequential loader (unknown total)."""
        raise NotImplementedError("Sequential loader does not have a fixed batch count")

    def mark_files_done(self, file_paths: List[str]) -> None:
        """Mark files as done by creating lock files.

        Args:
            file_paths: List of source file paths to mark as done
        """
        for file_path in file_paths:
            lock_path = os.path.join(self.lock_dir, os.path.basename(file_path) + '.done')
            with open(lock_path, 'w') as f:
                pass  # Create empty marker file

    def get_worker_dataset(self) -> Dict[str, Any]:
        """Distribute files across global workers using modulo assignment."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Calculate global worker ID
        global_worker_id = self.world_size * worker_id + self.rank
        total_workers = self.world_size * num_workers

        # Distribute files: each global worker gets every Nth file
        worker_file_paths = self.sample_file_paths[global_worker_id::total_workers]

        return {
            'sample_file_paths': worker_file_paths
        }

    def iterate_batches(
        self,
        worker_dataset: Dict[str, Any]
    ) -> Iterator[LoaderBatch]:
        """Iterate files and batches sequentially.

        Sets completed_file on the last batch of each file for resume tracking.
        """
        sample_file_paths = worker_dataset['sample_file_paths']

        for file_path in sample_file_paths:
            parquet_file = pq.ParquetFile(file_path)
            prev_batch_df = None

            for batch_table in parquet_file.iter_batches(batch_size=self.batch_size):
                if prev_batch_df is not None:
                    yield LoaderBatch(data=prev_batch_df)
                prev_batch_df = batch_table.to_pandas()

            # Last batch of this file gets completed_file marker
            if prev_batch_df is not None:
                yield LoaderBatch(data=prev_batch_df, completed_file=file_path)
