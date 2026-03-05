#!/usr/bin/env python3
"""
BiPathsNN Training Script (Torchrun Version)

Train BiPathsNN model for link prediction using bipath features.
Designed for torchrun launcher with automatic distributed training.

Usage:
    # Single GPU
    python 1_bipathsnn.py \
        --bilink-config main/config/bilink/author_new_reference/21_to_24_example.yaml \
        --model-config main/config/model/bipathsnn/default.yaml \
        --train-neg-ratio 10 \
        --eval-neg-ratio 10

    # Multi-GPU
    torchrun --nproc_per_node=4 1_bipathsnn.py \
        --bilink-config main/config/bilink/author_new_reference/21_to_24_example.yaml \
        --model-config main/config/model/bipathsnn/default.yaml \
        --train-neg-ratio 10 \
        --eval-neg-ratio 10

Arguments:
    --bilink-config     Path to bilink config YAML (dataset paths, graph config)
    --model-config      Path to model config YAML (architecture, hyperparameters)
    --train-neg-ratio   Negative sampling ratio for training data
    --eval-neg-ratio    Negative sampling ratio for validation/test data
    --device-type       Device type: cuda, npu, or cpu (default: cuda)
    --checkpoint-name   Checkpoint name (default: current timestamp)
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from types import SimpleNamespace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BiPathsNN Training')
    parser.add_argument('--bilink-config', type=str, required=True,
                        help='Path to bilink config YAML (for dataset_dir)')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--device-type', type=str, choices=['cuda', 'npu', 'cpu'],
                        default='cuda', help='Device type (cuda, npu, or cpu)')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                        help='Checkpoint name (default: current timestamp)')
    parser.add_argument('--train-neg-ratio', type=int, required=True,
                        help='Negative ratio for training data')
    parser.add_argument('--eval-neg-ratio', type=int, required=True,
                        help='Negative ratio for evaluation data')
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
    checkpoint_dir: str,
    fileio,
    project_root: str
) -> dict:
    """Build unified config for train() function.

    Args:
        args: Parsed command line arguments
        bilink_config: Bilink configuration dict
        model_config: Model configuration dict
        checkpoint_dir: Checkpoint directory path
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

    # 5. Build dataset section
    dataset_dir = f"{project_root}/{bilink_config['dataloader']['dataset_dir']}"
    path_group_len = model_config['dataloader']['path_group_len']
    max_path_group = model_config['dataloader']['max_path_group']

    base_dataset_config = {
        'loading_strategy': 'shuffled',
        'format': 'bipaths',
        'require_ids': False,
        'require_labels': True,
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

    train_config = {
        **base_dataset_config,
        'sample_path': f'{dataset_dir}/train/neg_ratio={args.train_neg_ratio}',
        'batch_size': model_config['dataloader']['train']['batch_size'],
        'epoch_size': model_config['dataloader']['train']['epoch_size'],
        'num_workers': model_config['dataloader']['train']['num_workers'],
        'shuffle': True,
        'fill_last': True,
    }

    eval_config_base = {
        **base_dataset_config,
        'batch_size': model_config['dataloader']['eval']['batch_size'],
        'sample_limit': model_config['dataloader']['eval']['sample_limit'],
        'num_workers': model_config['dataloader']['eval']['num_workers'],
        'shuffle': False,
        'fill_last': False,
    }

    val_config = {
        **eval_config_base,
        'sample_path': f'{dataset_dir}/valid/neg_ratio={args.eval_neg_ratio}',
    }

    test_config = {
        **eval_config_base,
        'sample_path': f'{dataset_dir}/test/neg_ratio={args.eval_neg_ratio}',
    }

    # 6. Build complete config
    return {
        'device': {'type': args.device_type},
        'dataset': {
            'train': train_config,
            'val': val_config,
            'test': test_config,
        },
        'dataloader': {
            'train': {
                'batch_size': None,
                'num_workers': model_config['dataloader']['train']['num_workers'],
                'prefetch_factor': model_config['dataloader']['train']['prefetch_factor'],
                'pin_memory': True,
                'persistent_workers': False,
                'multiprocessing_context': 'spawn',
            },
            'eval': {
                'batch_size': None,
                'num_workers': model_config['dataloader']['eval']['num_workers'],
                'prefetch_factor': model_config['dataloader']['eval']['prefetch_factor'],
                'pin_memory': True,
                'persistent_workers': False,
                'multiprocessing_context': 'spawn',
            },
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
        'metrics': [
            'pr_auc', 'roc_auc', 'bce_logits',
            'P@1000', 'P@5000', 'P@10000',
            'P@100', 'P@500', 'P@3000'
        ],
        'training': {
            'epochs': model_config['training']['epochs'],
        },
        'optimizer': {
            'learning_rate': model_config['optimization']['learning_rate'],
            'weight_decay': model_config['optimization']['weight_decay'],
        },
        'criterion': 'bce_with_logits',
        'scheduler': {
            'type': 'cosine_warmup',
            'warmup_ratio': 0.1,
        },
        'checkpoint': checkpoint_dir,
        'early_stopping': {
            'patience': model_config['training']['patience'],
            'monitor': 'pr_auc',
            'mode': 'max',
        },
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
    from joinminer.engine import train, find_free_port

    # 4. Build checkpoint name and directories
    checkpoint_name = args.checkpoint_name or datetime.now().strftime("%Y-%m-%d-%H:%M")
    checkpoint_dir = f"{PROJECT_ROOT}/data/runs/bipathsnn/{checkpoint_name}"

    # 5. Setup logger with rank-aware format and level
    log_files_dir = f"{PROJECT_ROOT}/data/logs/bipathsnn_train/{checkpoint_name}"
    log_file = f"{log_files_dir}/rank_{rank}.log"
    formatter = logging.Formatter(
        f'%(asctime)s - [Rank {rank}] - %(name)s - %(levelname)s - %(message)s'
    )
    level = 'DEBUG' if rank == 0 else 'INFO'
    logger = setup_logger(log_file, level=level, formatter=formatter)

    # 6. Load configs
    fileio = FileIO({'local': {}})
    bilink_config = fileio.read_yaml(f'file://{PROJECT_ROOT}/{args.bilink_config}')
    model_config = fileio.read_yaml(f'file://{PROJECT_ROOT}/{args.model_config}')

    # 7. Create checkpoint directory (only rank 0)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 8. Build unified config
    config = build_config(args, bilink_config, model_config, checkpoint_dir, fileio, PROJECT_ROOT)
    config['log'] = log_files_dir

    logger.info(f"Rank: {rank}, World size: {world_size}")
    logger.debug(f"Config: {json.dumps(config, indent=2, default=str)}")

    # 9. Disable core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # 10. Launch training
    port = find_free_port()
    train(rank, world_size, port, config)


if __name__ == "__main__":
    main()
