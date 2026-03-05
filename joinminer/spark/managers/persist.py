"""
PersistManager: Automatic management of PySpark DataFrame persistence lifecycle.

This module provides a simple but effective way to manage persisted DataFrames,
tracking which save points depend on each persisted DataFrame and automatically
releasing them when no longer needed.
"""

from typing import List, Union, Optional, Set, Dict
from collections import defaultdict
import logging
from pyspark.sql import DataFrame

# Module-level logger
logger = logging.getLogger(__name__)


class PersistManager:
    """
    Manages the lifecycle of persisted PySpark DataFrames.

    Tracks dependencies between persisted DataFrames and release points,
    automatically unpersisting DataFrames when their associated release point is marked.

    Key Design:
    - Each DataFrame is associated with exactly one release point (the last place it's used)
    - When that release point is marked as released, the DataFrame is automatically unpersisted
    - DataFrame names are optional; unique IDs are auto-generated for tracking

    Example:
        >>> pm = PersistManager()
        >>>
        >>> # Persist DataFrames for different release points
        >>> df1 = pm.persist(author_df, release_point='seed_node', name='authors')
        >>> df2 = pm.persist(paper_df, release_point='exist_edge')  # name is optional
        >>>
        >>> # Use df1 to compute and save seed_node
        >>> result1 = compute_seed_node(df1)
        >>> save(result1)
        >>> pm.mark_released('seed_node')  # df1 is automatically unpersisted
        >>>
        >>> # Use df2 to compute and save exist_edge
        >>> result2 = compute_exist_edge(df2)
        >>> save(result2)
        >>> pm.mark_released('exist_edge')  # df2 is automatically unpersisted
    """

    def __init__(self):
        """Initialize the PersistManager."""
        # Auto-increment ID for DataFrames
        self._next_id = 0

        # Store persisted DataFrames with their metadata
        # Structure: {df_id: {'df': DataFrame, 'name': str, 'save_point': str}}
        self._persisted_dfs = {}

        # Map from release point name to DataFrame IDs that support it
        # Structure: {release_point_name: set(df_ids)}
        self._release_point_to_dfs = defaultdict(set)

        logger.info("PersistManager initialized")

    def persist(self, df: DataFrame, release_point: str, name: Optional[str] = None) -> DataFrame:
        """
        Persist a DataFrame and track which release point it depends on.

        Args:
            df: The DataFrame to persist
            release_point: The release point name where this DataFrame will be released.
                          When this release point is marked, the DataFrame will be unpersisted.
            name: Optional name for the DataFrame (for logging and debugging).
                 If not provided, an auto-generated name will be used.

        Returns:
            The persisted DataFrame

        Example:
            >>> # With explicit name
            >>> df_persisted = pm.persist(author_work_df, release_point='exist_edge', name='author_work')
            >>>
            >>> # Without name (auto-generated)
            >>> df_persisted = pm.persist(processed_df, release_point='final_result')
        """
        if not release_point:
            raise ValueError("release_point must be specified")

        # Persist the DataFrame
        df.persist()

        # Generate unique ID and name
        df_id = self._next_id
        self._next_id += 1
        df_name = name if name else f"df_{df_id}"

        # Store DataFrame metadata
        self._persisted_dfs[df_id] = {
            'df': df,
            'name': df_name,
            'release_point': release_point
        }

        # Update release point mapping
        self._release_point_to_dfs[release_point].add(df_id)

        # Log the operation
        logger.info(f"Persisted '{df_name}' (id={df_id}) for release point: {release_point}")
        self._log_status()

        return df

    def mark_released(self, release_point: str) -> int:
        """
        Mark a release point as completed and unpersist all DataFrames associated with it.

        Args:
            release_point: The name of the release point that has been completed

        Returns:
            Number of DataFrames that were unpersisted

        Example:
            >>> # After completing the 'exist_edge' processing
            >>> pm.mark_released('exist_edge')
        """
        if release_point not in self._release_point_to_dfs:
            logger.warning(f"Release point '{release_point}' was not found in dependencies")
            return 0

        released_count = 0

        # Get all DataFrames associated with this release point
        df_ids = self._release_point_to_dfs[release_point].copy()

        for df_id in df_ids:
            if df_id in self._persisted_dfs:
                df_info = self._persisted_dfs[df_id]
                df_name = df_info['name']

                try:
                    df_info['df'].unpersist()
                    logger.info(f"Released '{df_name}' (id={df_id})")
                    released_count += 1
                except Exception as e:
                    logger.error(f"Error releasing '{df_name}' (id={df_id}): {e}")

                # Remove from tracking structures
                del self._persisted_dfs[df_id]

        # Clean up the release point mapping
        del self._release_point_to_dfs[release_point]

        logger.info(f"Completed release point '{release_point}', released {released_count} DataFrame(s)")
        self._log_status()

        return released_count

    def release_all(self) -> int:
        """
        Release all managed persisted DataFrames.

        Returns:
            Number of DataFrames that were released

        Example:
            >>> # At the end of processing or in case of error
            >>> pm.release_all()
        """
        released_count = 0

        # Copy the dict as we'll be modifying it
        df_ids = list(self._persisted_dfs.keys())

        for df_id in df_ids:
            if df_id in self._persisted_dfs:
                df_info = self._persisted_dfs[df_id]
                df_name = df_info['name']

                try:
                    df_info['df'].unpersist()
                    released_count += 1
                    logger.debug(f"Released '{df_name}' (id={df_id})")
                except Exception as e:
                    logger.error(f"Error releasing '{df_name}' (id={df_id}): {e}")

        # Clear all tracking structures
        self._persisted_dfs.clear()
        self._release_point_to_dfs.clear()

        logger.info(f"Released all {released_count} persisted DataFrames")

        return released_count

    def clear_tracking(self):
        """
        Clear internal tracking without calling unpersist() on DataFrames.

        This is useful when Spark session is already dead and unpersist() would fail.
        Only clears internal state to allow fresh start after restart.

        Example:
            >>> # When Spark is already shut down by system
            >>> pm.clear_tracking()  # Just reset state, don't try to unpersist
        """
        num_dfs = len(self._persisted_dfs)

        # Clear all tracking structures without calling unpersist()
        self._persisted_dfs.clear()
        self._release_point_to_dfs.clear()

        logger.info(f"Cleared tracking for {num_dfs} persisted DataFrames (without unpersist)")

    def _log_status(self):
        """Log the current status of persisted DataFrames and their dependencies."""
        num_dfs = len(self._persisted_dfs)
        num_release_points = len(self._release_point_to_dfs)

        if num_dfs == 0:
            logger.info("No persisted DataFrames")
            return

        logger.info(f"Current status: {num_dfs} persisted DataFrames, {num_release_points} pending release points")

        # Log details of release points and their dependencies
        for release_point, df_ids in self._release_point_to_dfs.items():
            df_names = [self._persisted_dfs[df_id]['name'] for df_id in df_ids if df_id in self._persisted_dfs]
            logger.debug(f"  Release point '{release_point}': {len(df_ids)} DataFrames - {df_names}")

    def get_status(self) -> Dict:
        """
        Get current status of the PersistManager.

        Returns:
            Dictionary containing:
            - active_dataframes: Number of currently persisted DataFrames
            - pending_release_points: Number of release points waiting to be completed
            - dataframe_details: Dict mapping DataFrame IDs to their info (name, release_point)
            - release_point_details: Dict mapping release point names to the number of DataFrames they depend on
        """
        return {
            'active_dataframes': len(self._persisted_dfs),
            'pending_release_points': len(self._release_point_to_dfs),
            'dataframe_details': {
                df_id: {'name': df_info['name'], 'release_point': df_info['release_point']}
                for df_id, df_info in self._persisted_dfs.items()
            },
            'release_point_details': {
                release_point: len(df_ids)
                for release_point, df_ids in self._release_point_to_dfs.items()
            }
        }

    def __repr__(self):
        """String representation of the PersistManager."""
        return (f"PersistManager(active={len(self._persisted_dfs)}, "
                f"release_points={len(self._release_point_to_dfs)})")