import os
import glob
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Iterator, Tuple

import torch
import torch.utils.data

from .base import LoaderBatch

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    global_worker_id: int
    total_workers: int
    target_batches: int
    file_paths: List[str]
    total_rows: int


class ShuffledLoader:
    """
    Shuffled parquet loader with greedy load-balanced file distribution.

    Designed for training datasets where file count is moderate.
    Files are distributed across workers using a greedy algorithm that
    balances total rows per worker, then workers are re-sorted to
    balance load across ranks.

    Supports:
    - Greedy load-balanced file distribution
    - File-order shuffling per iteration
    - Chunk-level row shuffling
    - fill_last for consistent batch counts across ranks
    """

    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        """
        Initialize shuffled loader with load-balanced file distribution.

        Args:
            config: Dataset configuration with keys:
                - sample_path: Path to parquet files directory
                - batch_size: Batch size
                - num_workers: Number of DataLoader workers
                - shuffle: Whether to shuffle file order and rows
                - fill_last: Whether to pad last batch
            rank: Current process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.shuffle = config.get('shuffle', True)
        self.fill_last = config.get('fill_last', False)

        # Scan and initialize file info
        self._init_file_info(config['sample_path'])
        # Initialize worker info with load-balanced distribution
        self._init_worker_info()

    def _init_file_info(self, sample_path: str):
        """Scan parquet files and get row counts."""
        file_paths = sorted(glob.glob(os.path.join(sample_path, "*.parquet")))
        if not file_paths:
            raise ValueError(f"No parquet files found in {sample_path}")

        # Parallel file size discovery
        self.file_sizes = self._parallel_get_file_sizes(file_paths)
        self.total_rows = sum(size for _, size in self.file_sizes)

    def _init_worker_info(self):
        """
        Distribute files to workers using greedy load balancing.

        Algorithm:
        1. Sort files by size descending
        2. Assign each file to the least-loaded worker
        3. Re-sort workers by load for balanced rank distribution
        """
        self.total_workers = self.world_size * self.num_workers

        # Pre-allocate arrays
        worker_files = [[] for _ in range(self.total_workers)]
        worker_rows = np.zeros(self.total_workers, dtype=np.int64)

        # Greedy assignment: assign each file to least-loaded worker
        for file_path, size in self.file_sizes:
            min_rows_worker = np.argmin(worker_rows)
            worker_files[min_rows_worker].append(file_path)
            worker_rows[min_rows_worker] += size

        # Re-sort workers by load (descending) for balanced rank distribution
        sorted_indices = np.argsort(-worker_rows)
        worker_files = [worker_files[i] for i in sorted_indices]
        worker_rows = worker_rows[sorted_indices]

        # Build worker info and compute rank-level totals
        self.workers_info = []
        self._rank_total_rows = np.zeros(self.world_size, dtype=np.int64)
        self._rank_total_batches = np.zeros(self.world_size, dtype=np.int64)

        for global_worker_id in range(self.total_workers):
            worker_total_rows = worker_rows[global_worker_id]
            target_batches = (worker_total_rows + self.batch_size - 1) // self.batch_size

            # Worker belongs to rank: rank_id = global_worker_id % world_size
            rank_id = global_worker_id % self.world_size
            self._rank_total_rows[rank_id] += worker_total_rows
            self._rank_total_batches[rank_id] += target_batches

            self.workers_info.append(WorkerInfo(
                global_worker_id=global_worker_id,
                total_workers=self.total_workers,
                target_batches=target_batches,
                file_paths=worker_files[global_worker_id],
                total_rows=worker_total_rows
            ))

    def _parallel_get_file_sizes(self, file_paths: List[str]) -> List[Tuple[str, int]]:
        """Get file row counts in parallel."""
        def get_file_size(file_path: str) -> Tuple[str, int]:
            with pq.ParquetFile(file_path) as pf:
                return file_path, pf.metadata.num_rows

        with ThreadPoolExecutor() as executor:
            file_sizes = list(executor.map(get_file_size, file_paths))

        # Sort by size descending for greedy assignment
        return sorted(file_sizes, key=lambda x: x[1], reverse=True)

    @property
    def rank_batch_count(self) -> int:
        """Total batches assigned to current rank."""
        return int(self._rank_total_batches[self.rank])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling.

        Should be called before each epoch in training loop.
        All workers will use this as shuffle seed.

        Args:
            epoch: Current epoch number
        """
        self._epoch = epoch

    def get_worker_dataset(self) -> Dict[str, Any]:
        """Get worker-specific file list."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # Compute global worker ID
        global_worker_id = self.world_size * worker_id + self.rank

        # Get pre-assigned worker info
        info = self.workers_info[global_worker_id]
        file_paths = info.file_paths.copy()

        # Shuffle file order if needed (deterministic per epoch)
        if self.shuffle and file_paths:
            seed = getattr(self, '_epoch', 0) * self.total_workers + global_worker_id
            rng = np.random.RandomState(seed)
            rng.shuffle(file_paths)

        return {
            'file_paths': file_paths,
            'worker_info': info
        }

    def iterate_batches(
        self,
        worker_dataset: Dict[str, Any]
    ) -> Iterator[LoaderBatch]:
        """
        Iterate over batches with optional shuffling.

        Reads files chunk by chunk, shuffles rows within chunks,
        buffers remaining rows, and yields complete batches.
        """
        file_paths = worker_dataset['file_paths']
        remaining_df = None

        for file_path in file_paths:
            parquet_file = pq.ParquetFile(file_path)

            for chunk in parquet_file.iter_batches(batch_size=self.batch_size):
                chunk = chunk.to_pandas()
                if self.shuffle:
                    chunk = chunk.sample(frac=1)

                # Prepend remaining data from previous chunk
                if remaining_df is not None and len(remaining_df) > 0:
                    chunk = pd.concat([remaining_df, chunk], ignore_index=True)
                    remaining_df = None

                # Yield complete batches
                while len(chunk) >= self.batch_size:
                    batch = chunk.iloc[:self.batch_size]
                    chunk = chunk.iloc[self.batch_size:]
                    yield LoaderBatch(data=batch)

                # Save remaining data
                if len(chunk) > 0:
                    remaining_df = chunk

        # Handle last remaining data
        if remaining_df is not None and len(remaining_df) > 0:
            if not self.fill_last:
                yield LoaderBatch(data=remaining_df)
            else:
                # Re-read files to fill the last batch
                for file_path in file_paths:
                    parquet_file = pq.ParquetFile(file_path)
                    for chunk in parquet_file.iter_batches(batch_size=self.batch_size):
                        chunk = chunk.to_pandas()
                        if self.shuffle:
                            chunk = chunk.sample(frac=1)
                        remaining_df = pd.concat([remaining_df, chunk], ignore_index=True)

                        if len(remaining_df) >= self.batch_size:
                            yield LoaderBatch(data=remaining_df.iloc[:self.batch_size])
                            return
