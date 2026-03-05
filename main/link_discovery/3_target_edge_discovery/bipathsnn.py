#!/usr/bin/env python3
"""
BiPathsNN Inference Script (Torchrun Version)

Run inference using trained BiPathsNN model for link prediction.
Designed for torchrun launcher with automatic distributed inference.

Usage:
    # Single GPU
    python bipathsnn.py \
        --bilink-config main/config/bilink/author_new_reference/21_to_24_example.yaml \
        --model-config main/config/model/bipathsnn/default.yaml \
        --checkpoint-name <checkpoint_name> \
        --sample-path data/openalex/.../bipaths_collection \
        --pred-path data/openalex/.../predictions

    # Multi-GPU
    torchrun --nproc_per_node=4 bipathsnn.py \
        --bilink-config main/config/bilink/author_new_reference/21_to_24_example.yaml \
        --model-config main/config/model/bipathsnn/default.yaml \
        --checkpoint-name <checkpoint_name> \
        --sample-path data/openalex/.../bipaths_collection \
        --pred-path data/openalex/.../predictions

Arguments:
    --bilink-config     Path to bilink config YAML (dataset paths, graph config)
    --model-config      Path to model config YAML (architecture, hyperparameters)
    --checkpoint-name   Checkpoint name (required)
    --sample-path       Relative path to inference dataset (from project root)
    --pred-path         Relative path for prediction output (from project root)
    --device-type       Device type: cuda, npu, or cpu (default: cuda)
"""

import os
import argparse
import pandas as pd
from types import SimpleNamespace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BiPathsNN Inference')
    parser.add_argument('--bilink-config', type=str, required=True,
                        help='Path to bilink config YAML (for dataset_dir)')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--device-type', type=str, choices=['cuda', 'npu', 'cpu'],
                        default='cuda', help='Device type (cuda, npu, or cpu)')
    parser.add_argument('--checkpoint-name', type=str, required=True,
                        help='Checkpoint name (required)')
    parser.add_argument('--sample-path', type=str, required=True,
                        help='Relative path to inference dataset (from project root)')
    parser.add_argument('--pred-path', type=str, required=True,
                        help='Relative path for prediction output (from project root)')
    return parser.parse_args()


def extract_path_types(bipath_file: str, top_k: int, max_hop: int = 6) -> tuple:
    """Extract path node and edge types from top_k paths in readable_matched_bipath.

    Args:
        bipath_file: Path to readable_matched_bipaths.csv
        top_k: Number of top paths to select by tgt_pair_count
        max_hop: Maximum hop count in paths

    Returns:
        (path_node_types, path_edge_types): Lists of unique type names
    """
    df = pd.read_csv(bipath_file)

    # Select top_k paths by tgt_pair_count
    df = df.nlargest(top_k, 'tgt_pair_count')

    path_node_types = set()
    path_edge_types = set()

    # Extract node types from node_*_type_name columns
    for i in range(max_hop + 1):
        col = f'node_{i}_type_name'
        if col in df.columns:
            path_node_types.update(df[col].dropna().unique())

    # Extract edge types from edge_*_type_name columns
    for i in range(max_hop):
        col = f'edge_{i}_type_name'
        if col in df.columns:
            path_edge_types.update(df[col].dropna().unique())

    return sorted(path_node_types), sorted(path_edge_types)


def build_config(
    args,
    bilink_config: dict,
    model_config: dict,
    fileio,
    project_root: str
) -> dict:
    """Build unified config for inference() function.

    Args:
        args: Parsed command line arguments
        bilink_config: Bilink configuration dict
        model_config: Model configuration dict
        fileio: FileIO instance
        project_root: Project root path

    Returns:
        Unified config dict with all sections
    """
    from joinminer.graph import Graph

    # 1. Load graph
    graph_config_path = bilink_config['graph_config']
    graph_yaml = fileio.read_yaml(f'file://{project_root}/{graph_config_path}')
    graph = Graph(graph_yaml, fileio)

    # 2. Extract path types from readable_matched_bipath
    matched_bipath_file = f"{project_root}/{bilink_config['bipath_discovery']['readable_matched_bipath']}"
    max_hop = bilink_config['bipath_discovery']['exploration']['max_hop']
    top_k = bilink_config['bipath_discovery']['collection']['top_k']
    path_node_types, path_edge_types = extract_path_types(matched_bipath_file, top_k, max_hop)

    # 3. Build feat_len from graph (for model.feat_proj)
    feat_len = {'node': {}, 'edge': {}}
    for node_type in path_node_types:
        feat_len['node'][node_type] = graph.nodes[node_type]['feature_count']
    for edge_type in path_edge_types:
        feat_len['edge'][edge_type] = graph.edges[edge_type]['feature_count']

    # 4. Build pair config from target_edge
    target_edge = bilink_config['target_edge']
    pair = {
        'u_node_type': target_edge['u_node_type'],
        'v_node_type': target_edge['v_node_type'],
    }

    # 5. Build dataset section (inference only, sequential loading)
    path_group_len = model_config['dataloader']['path_group_len']
    max_path_group = model_config['dataloader']['max_path_group']

    infer_config = {
        'loading_strategy': 'sequential',
        'format': 'bipaths',
        'require_ids': True,
        'require_labels': False,
        'sample_path': f'{project_root}/{args.sample_path}',
        'batch_size': model_config['dataloader']['infer']['batch_size'],
        'shuffle': False,
        'fill_last': False,
        'graph': SimpleNamespace(
            nodes={
                node_type: {
                    'feature_count': info['feature_count'],
                    'type_index': info['type_index'],
                }
                for node_type, info in graph.nodes.items()
            },
            edges={
                edge_type: {
                    'feature_count': info['feature_count'],
                    'type_index': info['type_index'],
                }
                for edge_type, info in graph.edges.items()
            },
            partition_columns=graph.partition_columns,
        ),
        'pair': pair,
        'path': {'node': path_node_types, 'edge': path_edge_types},
        'path_group_len': path_group_len,
        'max_path_group': max_path_group,
    }

    return {
        'device': {'type': args.device_type},
        'dataset': infer_config,
        'dataloader': {
            'batch_size': None,
            'num_workers': model_config['dataloader']['infer']['num_workers'],
            'prefetch_factor': model_config['dataloader']['infer']['prefetch_factor'],
            'pin_memory': True,
            'persistent_workers': False,
            'multiprocessing_context': 'spawn',
        },
        'model': {
            'name': 'bipathsnn',
            'dropout': model_config['model']['dropout'],
            'feat_proj': {
                'feat_len': feat_len,
                **model_config['model']['feat_proj'],
            },
            'path_group_encoder': model_config['model']['path_group_encoder'],
            'pair_encoder': model_config['model']['pair_encoder'],
        },
        'checkpoint': f"{project_root}/data/runs/bipathsnn/{args.checkpoint_name}",
        'pred_path': f'{project_root}/{args.pred_path}',
        'pred_batch': model_config['dataloader']['infer']['pred_batch'],
    }


def main():
    args = parse_args()

    # 1. Get rank/world_size from torchrun environment
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 2. NPU requires torch_npu import before other torch operations
    if args.device_type == "npu":
        import torch_npu  # type: ignore

    # 3. Now import torch and other modules
    import json
    import logging
    import resource

    from joinminer import PROJECT_ROOT
    from joinminer.fileio import FileIO
    from joinminer.utils import setup_logger
    from joinminer.engine import inference, find_free_port

    # 4. Setup logger with rank-aware format and level
    log_files_dir = f"{PROJECT_ROOT}/data/logs/bipathsnn_infer/{args.checkpoint_name}"
    log_file = f"{log_files_dir}/rank_{rank}.log"
    formatter = logging.Formatter(
        f'%(asctime)s - [Rank {rank}] - %(name)s - %(levelname)s - %(message)s'
    )
    level = 'DEBUG' if rank == 0 else 'INFO'
    logger = setup_logger(log_file, level=level, formatter=formatter)

    # 5. Load configs
    fileio = FileIO({'local': {}})
    bilink_config = fileio.read_yaml(f'file://{PROJECT_ROOT}/{args.bilink_config}')
    model_config = fileio.read_yaml(f'file://{PROJECT_ROOT}/{args.model_config}')

    # 6. Build unified config
    config = build_config(args, bilink_config, model_config, fileio, PROJECT_ROOT)
    config['log'] = log_files_dir

    logger.info(f"Rank: {rank}, World size: {world_size}")
    logger.debug(f"Config: {json.dumps(config, indent=2, default=str)}")

    # 7. Disable core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # 8. Launch inference
    port = find_free_port()
    inference(rank, world_size, port, config)


if __name__ == "__main__":
    main()
