#!/usr/bin/env python3
"""
Prepare Graph Data for Link Discovery

This script prepares all graph-related data including element tables (node and edge features)
and join edges for graph neural network training.

Features:
- YAML-driven configuration for all element definitions
- Supports multiple target dates/partitions
- Builds element tables and join edges in one step
- Error handling with detailed logging
- Completion markers to skip already-built tables

Element Types:
- Node Elements: work, author, institution, source, topic
- Edge Elements: work-work, work-author, work-institution, work-source, work-topic, author-institution

Usage:
    python 0_prepare_graph_data.py \
        --elements-config main/config/element/openalex_local.yaml \
        --graph-config main/config/graph/openalex_local.yaml \
        --bilink-config main/config/bilink/author_new_reference/21_to_24_local.yaml \
        --target-dates 2021-01-01 2022-01-01 2023-01-01 2024-01-01 \
        [--force] \
        [--verbose]
"""

import argparse
import sys
import logging
from datetime import datetime

from joinminer import PROJECT_ROOT
from joinminer.graph import Graph
from joinminer.graph.element import ElementBuilder
from joinminer.graph.join_edges import generate_join_edges
from joinminer.spark import SparkRunner
from joinminer.fileio import FileIO
from joinminer.utils import setup_logger

# Module-level logger
logger = logging.getLogger("joinminer")


# ==================== Main Function ====================

def build_element_tables(spark_ctx, elements_config, partition_spec_dict):
    """
    Build element tables from base tables.

    This function creates a dedicated ElementBuilder for each element and builds it.
    Each ElementBuilder instance is initialized with all configuration and then only
    needs the partition values to build.

    Args:
        spark_ctx: SparkContext instance (passed by SparkRunner.run())
        elements_config: Elements configuration dictionary loaded from YAML with structure:
            - context_table: Global context table config (dir, format, partition_columns)
            - element_table: Global element table config (dir, format, partition_columns)
            - elements: Dict of element configurations
        partition_spec_dict: Dictionary mapping partition spec names to actual values
            Example: {'date_partitions': ['2021-01-01', '2022-01-01']}
    """

    # Extract global configurations
    context_table_config = elements_config.get('context_table')
    if not context_table_config:
        raise ValueError("Configuration must include 'context_table' section")

    element_table_config = elements_config.get('element_table')
    if not element_table_config:
        raise ValueError("Configuration must include 'element_table' section")

    # Build all elements defined in configuration
    elements_to_build = list(elements_config['elements'].keys())

    logger.info(f"Will build {len(elements_to_build)} element tables: {elements_to_build}")
    logger.info(f"Partition specs provided: {list(partition_spec_dict.keys())}")

    # Build element tables
    for element_name in elements_to_build:
        logger.info("\n" + "=" * 60)
        logger.info(f"Building element table: {element_name}")
        logger.info("=" * 60)

        element_config = elements_config['elements'][element_name]

        # Create dedicated builder for this element
        builder = ElementBuilder(
            spark_ctx=spark_ctx,
            element_name=element_name,
            element_config=element_config,
            context_table_config=context_table_config,
            element_table_config=element_table_config
        )

        # Get partition specification for this element
        partition_spec_name = element_config.get('partition_spec')
        partition_spec = (
            partition_spec_dict.get(partition_spec_name)
            if partition_spec_name else None
        )

        # Build element table
        builder.build(partition_spec)

        logger.info(f"✓ Successfully built element table: {element_name}")


def build_join_edges(spark_ctx, fileio, graph_config, bilink_config, target_dates):
    """
    Build join edges from element tables.

    Join edges represent connections between nodes in the graph at different hops,
    used for multi-hop path exploration in link prediction tasks.

    Args:
        spark_ctx: SparkContext instance (passed by SparkRunner.run())
        fileio: FileIO instance for file operations
        graph_config: Graph structure configuration dictionary (already contains
                      element_table_dir, partition_columns, nodes, and edges)
        bilink_config: BiLink task configuration dictionary
        target_dates: List of target dates for partitions
    """
    # Extract and validate bipath_discovery configuration
    bipath_config = bilink_config.get('bipath_discovery')
    if not bipath_config:
        raise ValueError("'bipath_discovery' section not found in bilink config")

    join_edge_path = bipath_config.get('join_edge_path')
    if not join_edge_path:
        raise ValueError("'join_edge_path' not found in bipath_discovery config")

    max_neighbor = bipath_config['max_instance']

    logger.info(f"Join edge path: {join_edge_path}")
    logger.info(f"Max neighbor: {max_neighbor}")

    # Initialize Graph with loaded config (Graph will display its own summary)
    graph = Graph(graph_config, fileio)

    # Prepare partition instances
    partition_instances = [[date] for date in target_dates]
    logger.info(f"Generating join edges for {len(partition_instances)} partitions: {partition_instances}")

    # Generate join edges (automatically handles COMPLETE status internally)
    generate_join_edges(
        spark_ctx=spark_ctx,
        graph=graph,
        join_edge_path=join_edge_path,
        partition_instances=partition_instances,
        max_neighbor=max_neighbor
    )

    logger.info("✓ Join edge generation completed!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare graph data (element tables and join edges) for graph neural network training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all elements and join edges for multiple dates
  python 0_prepare_graph_data.py \\
      --elements-config main/config/element/openalex_local.yaml \\
      --graph-config main/config/graph/openalex_local.yaml \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --target-dates 2021-01-01 2022-01-01 2023-01-01 2024-01-01

  # Force rebuild all elements and join edges
  python 0_prepare_graph_data.py \\
      --elements-config main/config/element/openalex_local.yaml \\
      --graph-config main/config/graph/openalex_local.yaml \\
      --bilink-config main/config/bilink/<task>/21_to_24_example.yaml \\
      --target-dates 2021-01-01 \\
      --force
        """
    )

    # Required arguments
    parser.add_argument('--elements-config', required=True,
                        help='Path to elements configuration YAML file')

    # Optional arguments
    parser.add_argument('--target-dates', nargs='+',
                        default=['2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
                        help='Target dates for element tables (default: 2021-2024, one per year)')
    parser.add_argument('--spark-mode', default='local', choices=['local', 'cluster'],
                        help='Spark execution mode (default: local)')
    parser.add_argument('--spark-platform', default='localhost', choices=['localhost', 'example'],
                        help='Spark platform (default: localhost)')
    parser.add_argument('--backends', nargs='+',
                        default=['local'],
                        choices=['local', 'hdfs', 's3'],
                        help='File backends to enable (default: local)')

    # Required arguments for join edge generation
    parser.add_argument('--graph-config', required=True,
                        help='Path to graph configuration YAML file')
    parser.add_argument('--bilink-config', required=True,
                        help='Path to bilink configuration YAML file')

    # Flags
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild tables even if they already exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    task_name = "build_element_tables"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{PROJECT_ROOT}/data/logs/{task_name}/{timestamp}.log"
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(log_file, level=log_level)

    # Initialize FileIO
    backend_configs = {backend: {} for backend in args.backends}
    fileio = FileIO(backend_configs)
    logger.info(f"FileIO initialized with schemes: {list(fileio.backends.keys())}")

    # Initialize SparkRunner
    spark_runner = SparkRunner(
        mode=args.spark_mode,
        platform=args.spark_platform,
        fileio=fileio,
        config_dict={},
        ignore_complete=args.force  # Pass --force flag to ignore completion checks
    )

    # Load elements configuration
    config_path = f"file://{PROJECT_ROOT}/{args.elements_config}"
    logger.info(f"Loading elements configuration from: {config_path}")
    elements_config = fileio.read_yaml(config_path)
    logger.info(f"Elements configuration loaded. Found {len(elements_config['elements'])} elements.")

    # Create partition spec dict
    # Map partition spec name -> list of partition values
    partition_spec_dict = {
        'date_partitions': args.target_dates
    }
    logger.info(f"Partition spec dict created: {partition_spec_dict}")

    # Run element table building
    spark_runner.run(
        build_element_tables,
        elements_config,
        partition_spec_dict
    )

    # Load additional configurations for join edge generation
    graph_config_path = f"file://{PROJECT_ROOT}/{args.graph_config}"
    bilink_config_path = f"file://{PROJECT_ROOT}/{args.bilink_config}"

    logger.info(f"Loading graph config from: {graph_config_path}")
    graph_config = fileio.read_yaml(graph_config_path)

    logger.info(f"Loading bilink config from: {bilink_config_path}")
    bilink_config = fileio.read_yaml(bilink_config_path)

    # Run join edge generation
    logger.info("\n" + "=" * 60)
    logger.info("Starting join edge generation...")
    logger.info("=" * 60)

    spark_runner.run(
        build_join_edges,
        fileio,
        graph_config,
        bilink_config,
        args.target_dates
    )

    logger.info("\n" + "=" * 60)
    logger.info("Script completed successfully")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
