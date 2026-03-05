"""
Element table generation module.

This module provides tools for building element tables from context tables.
Element tables are the fundamental building blocks for graph construction, containing:
- Entity identifiers
- Time-based partitions
- Feature vectors with complete provenance metadata

Architecture:
- ElementBuilder: Orchestrates element table construction
- SourceProcessor: Processes individual sources (context_mapping + aggregation)
- context_mapping/: Handles all ID mapping strategies (static, time_window)
- aggregation_pipeline: Multi-stage aggregation functions
- feature_metadata: Feature metadata generation utilities
"""

from .builder import ElementBuilder
from .source import SourceProcessor
from .feature_metadata import FeatureMetadataGenerator
from .context_mapping import (
    ContextMapper,
    StaticContextMapper,
    TimeWindowContextMapper,
)

__all__ = [
    # Core
    'ElementBuilder',
    'SourceProcessor',

    # Feature metadata generation
    'FeatureMetadataGenerator',

    # Context mapping
    'ContextMapper',
    'StaticContextMapper',
    'TimeWindowContextMapper',
]
