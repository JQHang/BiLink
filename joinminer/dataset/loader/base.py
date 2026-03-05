from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, List
import pandas as pd


@dataclass
class LoaderBatch:
    """Unified batch result from all loaders.

    Attributes:
        data: Batch DataFrame
        completed_file: Set only on last batch of a file (for checkpoint support).
                       None for most batches.
    """
    data: pd.DataFrame
    completed_file: Optional[str] = None


class BatchLoader:
    """
    Unified batch loader for parquet datasets.

    Delegates to strategy-specific implementations based on config['loading_strategy'].

    Responsibilities:
    - Loading strategy selection
    - Worker dataset allocation
    - Batch iteration logic

    Different strategies:
    - Shuffled: Greedy load-balanced file distribution with row shuffling (for training)
    - Sequential: Simple modulo file distribution, sequential processing (for inference)
    """

    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        """
        Initialize handler by loading appropriate implementation.

        Args:
            config: Dataset configuration with 'loading_strategy' key
            rank: Current process rank
            world_size: Total number of processes

        Raises:
            ValueError: If loading_strategy is unknown
        """
        loading_strategy = config['loading_strategy']

        if loading_strategy == 'shuffled':
            from .shuffled import ShuffledLoader
            self._impl = ShuffledLoader(config, rank, world_size)
        elif loading_strategy == 'sequential':
            from .sequential import SequentialLoader
            self._impl = SequentialLoader(config, rank, world_size)
        else:
            raise ValueError(f"Unknown loading_strategy: {loading_strategy}")

    @property
    def rank_batch_count(self) -> int:
        """Total batches assigned to current rank."""
        return self._impl.rank_batch_count

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling (proxy to shuffled loader)."""
        if hasattr(self._impl, 'set_epoch'):
            self._impl.set_epoch(epoch)

    def mark_files_done(self, file_paths: List[str]) -> None:
        """Mark files as done for resume support (proxy to sequential loader)."""
        if hasattr(self._impl, 'mark_files_done'):
            self._impl.mark_files_done(file_paths)

    def iterate_batches(self) -> Iterator[LoaderBatch]:
        """
        Iterate over batches for this worker.

        Automatically performs worker-specific setup before iteration.

        Yields:
            LoaderBatch with data DataFrame and optional completed_file
        """
        # Get worker-specific dataset info
        worker_dataset = self._impl.get_worker_dataset()

        # Iterate batches
        yield from self._impl.iterate_batches(worker_dataset)
