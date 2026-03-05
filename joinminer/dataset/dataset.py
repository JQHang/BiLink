import logging
from typing import Dict, Any, Iterator

from torch.utils.data import IterableDataset

from .loader import BatchLoader
from .converter import BatchConverter

logger = logging.getLogger(__name__)


class Dataset(IterableDataset):
    """
    Base class for datasets in JoinMiner.

    Responsibilities:
    - PyTorch IterableDataset integration
    - Distributed training coordination (rank/worker management)
    - Configuration validation and initialization
    - Delegates format-specific logic to BatchLoader
    - Delegates batch conversion to BatchConverter

    Not responsible for:
    - Loading strategy details (Shuffled/Sequential) → BatchLoader
    - Data structure conversion → BatchConverter
    """

    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """
        Initialize dataset.

        Args:
            config: Dataset configuration dict with keys:
                - format: 'bipaths' (data structure type)
                - loading_strategy: 'shuffled' or 'sequential'
                - sample_path: Path to samples
                - batch_size: Batch size
                - (format-specific fields)
                - (structure-specific fields)
            rank: Current process rank (default: 0 for single-process)
            world_size: Total number of processes (default: 1 for single-process)
        """
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Direct initialization (no registry)
        self.batch_loader = BatchLoader(
            config=config,
            rank=self.rank,
            world_size=self.world_size
        )
        self.batch_converter = BatchConverter(config)

    def __iter__(self) -> Iterator[Dict]:
        """
        Implement data iteration.

        Process:
        1. Iterate batches from batch loader (handles worker setup internally)
        2. Convert each batch using batch converter
        3. Yield converted batch
        """
        # Iterate batches (format-specific, worker setup handled internally)
        for loader_batch in self.batch_loader.iterate_batches():
            # Convert batch (structure-specific)
            converted_batch = self.batch_converter.convert(loader_batch)
            yield converted_batch
