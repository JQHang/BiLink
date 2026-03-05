"""
Auxiliary managers for Spark operations.

This module contains helper classes for managing various aspects of Spark operations:
- PersistManager: Manages DataFrame persistence and unpersistence
- TableStateManager: Manages table/partition completion state tracking

Both managers are automatically initialized as attributes of SparkRunner:
- spark_runner.persist_manager
- spark_runner.table_state

Users typically access these managers through SparkRunner rather than
instantiating them directly.
"""

from joinminer.spark.managers.persist import PersistManager
from joinminer.spark.managers.table_state import TableStateManager

__all__ = [
    'PersistManager',
    'TableStateManager',
]
