from typing import Dict, Any

from ..loader.base import LoaderBatch


class BatchConverter:
    """
    Unified batch converter for all data structures.

    Delegates to structure-specific implementations based on config['format'].

    Responsibilities:
    - Convert LoaderBatch into model input format
    - Structure-specific feature extraction
    - Tensor conversion

    Different structures have different conversions:
    - Bilink: Complex path-based structure with grouped tokens
    - Future: Single paths, subgraphs, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize converter by loading appropriate implementation.

        Args:
            config: Dataset configuration with 'format' key

        Raises:
            ValueError: If format is unknown
        """
        self.config = config
        structure_format = config['format']

        if structure_format == 'bipaths':
            from .bilink import BilinkConverter
            self._impl = BilinkConverter(config)
        else:
            raise ValueError(f"Unknown format: {structure_format}")

    def convert(self, loader_batch: LoaderBatch):
        """
        Convert LoaderBatch into model input.

        Delegates to implementation.

        Args:
            loader_batch: LoaderBatch from batch loader

        Returns:
            Batch object with to_device() method
        """
        return self._impl.convert(loader_batch)
