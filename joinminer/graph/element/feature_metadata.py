"""
Feature metadata generation for element tables.

This module provides stateless functions to generate feature metadata based on
transformation configurations and mapping information from context_mapping.
"""

from typing import List, Dict, Any, Optional
from copy import deepcopy


class FeatureMetadataGenerator:
    """
    Stateless generator for feature metadata.

    This class provides static methods to generate feature metadata at each
    stage of the element building pipeline. The metadata structure reflects
    the actual data flow: context_mapping creates the initial data, then
    aggregation stages progressively transform it.
    """

    @staticmethod
    def initialize_from_context(
        original_features: List[str],
        context_table_name: str,
        context_mapping_config: Dict[str, Any],
        context_feature_descriptions: Dict[str, str],
        context_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Initialize feature metadata from context mapping stage.

        Creates one metadata entry per original feature, with complete
        context_mapping configuration. Does NOT create separate entries
        for each window - that happens during pivot aggregation.

        Args:
            original_features: List of feature column names from context table
            context_table_name: Name of the source context table
            context_mapping_config: Complete context_mapping configuration dict:
                Static: {'type': 'static'}
                Time window: {'type': 'time_window', 'windows': {...}}
            context_feature_descriptions: Dict of feature_name -> description
            context_dir: Full path to context table directory

        Returns:
            List of feature metadata dicts, each containing:
            - index: Position in feature list (0-indexed)
            - context_mapping: Complete mapping configuration with feature source info
            - aggregations: List of aggregation stages (initially empty)
        """
        feature_metadata = []

        for idx, feature_name in enumerate(original_features):
            # Build complete context_mapping with feature source info
            mapping_config = deepcopy(context_mapping_config)
            mapping_config['context_table'] = context_table_name
            mapping_config['context_dir'] = context_dir
            mapping_config['feature_column'] = feature_name
            mapping_config['feature_description'] = context_feature_descriptions.get(feature_name, '')

            feature_metadata.append({
                'index': idx,
                'context_mapping': mapping_config,
                'aggregations': []
            })

        return feature_metadata

    @staticmethod
    def apply_aggregation_stage(
        feature_metadata: List[Dict[str, Any]],
        aggregation_config: Dict[str, Any],
        mapping_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate feature metadata after an aggregation stage.

        Handles three mapping behaviors:
        - drop: Collapse across all mappings, keep single feature per base feature
        - keep: Preserve mapping separation, keep single feature per base feature
        - pivot: Expand each base feature into multiple features (one per mapping)

        Args:
            feature_metadata: Current feature metadata list
            aggregation_config: Aggregation stage configuration:
                - group_by: List of grouping columns
                - functions: List of aggregation functions
                - mapping: 'drop' | 'keep' | 'pivot'
                - group_count: Optional boolean
            mapping_info: Mapping information from context_mapping

        Returns:
            New feature metadata list after aggregation
        """
        group_by = aggregation_config['group_by']
        functions = aggregation_config.get('functions', [])
        mapping_behavior = aggregation_config.get('mapping', 'drop')
        group_count = aggregation_config.get('group_count', False)

        if mapping_behavior == 'pivot':
            return FeatureMetadataGenerator._apply_pivot_aggregation(
                feature_metadata, group_by, functions, mapping_info, group_count
            )
        else:
            return FeatureMetadataGenerator._apply_normal_aggregation(
                feature_metadata, group_by, functions, mapping_behavior, group_count
            )

    @staticmethod
    def _apply_normal_aggregation(
        feature_metadata: List[Dict[str, Any]],
        group_by: List[str],
        functions: List[str],
        mapping_behavior: str,
        group_count: bool
    ) -> List[Dict[str, Any]]:
        """
        Apply normal aggregation (drop or keep).

        For each function, create a new feature based on each original feature.
        """
        new_metadata = []

        for func in functions:
            for orig_meta in feature_metadata:
                new_meta = deepcopy(orig_meta)

                # Add aggregation record
                agg_record = {
                    'stage': len(orig_meta['aggregations']) + 1,
                    'function': func,
                    'group_by': group_by,
                    'mapping': mapping_behavior
                }
                new_meta['aggregations'].append(agg_record)

                new_metadata.append(new_meta)

        # Add group_count feature if requested
        if group_count:
            # Group count inherits context_mapping but removes feature_column
            if feature_metadata:
                count_meta = deepcopy(feature_metadata[0])
                # Remove feature_column and feature_description to mark as synthetic
                if 'feature_column' in count_meta['context_mapping']:
                    del count_meta['context_mapping']['feature_column']
                if 'feature_description' in count_meta['context_mapping']:
                    del count_meta['context_mapping']['feature_description']

                # Add aggregation with group_count marker
                count_meta['aggregations'].append({
                    'stage': len(feature_metadata[0]['aggregations']) + 1,
                    'group_count': True,  # Mark as count(*) operation
                    'group_by': group_by,
                    'mapping': mapping_behavior
                })
                new_metadata.append(count_meta)

        # Reindex
        for i, meta in enumerate(new_metadata):
            meta['index'] = i

        return new_metadata

    @staticmethod
    def _apply_pivot_aggregation(
        feature_metadata: List[Dict[str, Any]],
        group_by: List[str],
        functions: List[str],
        mapping_info: List[Dict[str, Any]],
        group_count: bool
    ) -> List[Dict[str, Any]]:
        """
        Apply pivot aggregation.

        THIS IS WHERE window expansion happens: each base feature gets
        expanded into multiple features (one per mapping/window).

        For each (function, feature, window) combination, create a new
        metadata entry with the specific window config in the aggregation record.
        """
        new_metadata = []

        # Extract mapping IDs
        mapping_values = [m['id'] for m in mapping_info]

        # For each function
        for func in functions:
            # For each original feature
            for orig_meta in feature_metadata:
                # For each window/mapping
                for mapping in mapping_info:
                    new_meta = deepcopy(orig_meta)

                    # Add aggregation record with pivot and window info
                    agg_record = {
                        'stage': len(orig_meta['aggregations']) + 1,
                        'function': func,
                        'group_by': group_by,
                        'mapping': 'pivot'
                    }

                    # Add window config only if not static
                    if mapping['id'] != 'static':
                        agg_record['window'] = mapping['config']

                    new_meta['aggregations'].append(agg_record)
                    new_metadata.append(new_meta)

        # Add group_count features (one per mapping)
        if group_count:
            if feature_metadata:
                base_meta = feature_metadata[0]
                for mapping in mapping_info:
                    count_meta = deepcopy(base_meta)
                    # Remove feature_column and feature_description to mark as synthetic
                    if 'feature_column' in count_meta['context_mapping']:
                        del count_meta['context_mapping']['feature_column']
                    if 'feature_description' in count_meta['context_mapping']:
                        del count_meta['context_mapping']['feature_description']

                    agg_record = {
                        'stage': len(base_meta['aggregations']) + 1,
                        'group_count': True,  # Mark as count(*) operation
                        'group_by': group_by,
                        'mapping': 'pivot'
                    }

                    if mapping['id'] != 'static':
                        agg_record['window'] = mapping['config']

                    count_meta['aggregations'].append(agg_record)
                    new_metadata.append(count_meta)

        # Reindex
        for i, meta in enumerate(new_metadata):
            meta['index'] = i

        return new_metadata

    @staticmethod
    def generate_element_info(
        feature_metadata: List[Dict[str, Any]],
        partition_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Generate final _element_info.json content.

        Args:
            feature_metadata: Final feature metadata after all transformations
            partition_columns: Element table's partition columns

        Returns:
            Dictionary in format for _element_info.json:
            {
                'partition_columns': [...],
                'features': [...]
            }
        """
        return {
            'partition_columns': partition_columns,
            'features': feature_metadata
        }
