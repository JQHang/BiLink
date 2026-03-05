"""Inference worker function for distributed inference."""

import os
import glob
import logging
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .distributed import get_device
from .checkpoint import CheckpointManager
from ..model import MODEL_REGISTRY
from ..dataset import Dataset

logger = logging.getLogger(__name__)


def _set_npu_eval_mode(model: nn.Module) -> None:
    """Set model to eval mode with NPU compatibility.

    NPU doesn't support fused TransformerEncoderLayer inference op,
    so keep TransformerEncoderLayer in train mode but disable dropout.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.train()
            for sub_module in module.modules():
                if isinstance(sub_module, nn.Dropout):
                    sub_module.eval()


def inference(rank: int, world_size: int, port: int, config: Dict[str, Any]) -> None:
    """Inference worker function.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
        port: DDP communication port.
        config: Unified config dict with sections:
            - model: {name, ...model-specific params...}
            - dataset: {infer: {...}}
            - device: {type}
            - dataloader: {num_workers, prefetch_factor}
            - checkpoint: Checkpoint directory path
            - pred_path: Output directory for predictions
            - pred_batch: Number of predictions to accumulate before saving
            - log: Log directory
    """
    device_type = config["device"]["type"]

    # 1. Setup device for this rank
    if device_type == "npu":
        import torch_npu  # type: ignore
        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)

    device = get_device(rank, device_type)

    # 2. Ensure pred_path exists
    os.makedirs(config["pred_path"], exist_ok=True)

    # 3. Create dataset (infer only)
    infer_dataset = Dataset(config["dataset"], rank=rank, world_size=world_size)

    # 4. Create dataloader
    dataloader = DataLoader(infer_dataset, **config["dataloader"])

    # 5. Create model
    model_config = config["model"]
    model = MODEL_REGISTRY[model_config["name"]](model_config).to(device)

    # 6. Load best checkpoint
    checkpoint_manager = CheckpointManager(config["checkpoint"])
    checkpoint = checkpoint_manager.load_best()
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 7. Set eval mode (with NPU compatibility)
    if device.startswith('npu'):
        _set_npu_eval_mode(model)
    else:
        model.eval()

    # 8. Run inference loop
    _inference_loop(
        model=model,
        dataloader=dataloader,
        batch_loader=infer_dataset.batch_loader,
        config=config,
        rank=rank,
        device=device,
    )


def _inference_loop(
    model: nn.Module,
    dataloader: DataLoader,
    batch_loader,
    config: Dict[str, Any],
    rank: int,
    device: str,
) -> None:
    """Process batches, accumulate predictions, save in chunks.

    Key logic:
    - Add predictions as column to pair_id_df
    - Track completed_file from each batch
    - When accumulated predictions > pred_batch:
        1. Save predictions to parquet
        2. Mark files as done via batch_loader.mark_files_done()
        3. Clear accumulators
    """
    pred_path = config["pred_path"]
    pred_batch = config["pred_batch"]

    # Count existing prediction files for this rank to avoid overwriting
    existing_files = glob.glob(os.path.join(pred_path, f"predictions_rank_{rank}_*.parquet"))
    file_index = len(existing_files)

    # Accumulators
    result_dfs: List[pd.DataFrame] = []
    completed_files: List[str] = []
    accumulated_count = 0
    total_rows = 0
    total_completed_files = 0

    pbar = tqdm(dataloader, desc="Inference") if rank == 0 else dataloader

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to_device(device)

            # Model inference
            outputs = model(batch)

            # Add prediction to pair_id_df
            batch.pair_id_df['prediction'] = outputs.cpu().numpy().flatten()
            result_dfs.append(batch.pair_id_df)

            # Track completed file
            if batch.completed_file is not None:
                completed_files.append(batch.completed_file)
                total_completed_files += 1
                if rank == 0:
                    pbar.set_postfix(files=total_completed_files, rows=total_rows)

            accumulated_count += len(outputs)

            # Check if we should save
            if accumulated_count >= pred_batch:
                result_df = pd.concat(result_dfs, ignore_index=True)
                output_file = os.path.join(pred_path, f"predictions_rank_{rank}_{file_index}.parquet")
                result_df.to_parquet(output_file, index=False)
                batch_loader.mark_files_done(completed_files)

                total_rows += len(result_df)
                file_index += 1
                if rank == 0:
                    pbar.set_postfix(files=total_completed_files, rows=total_rows)

                # Reset accumulators
                result_dfs = []
                completed_files = []
                accumulated_count = 0

    # Save remaining predictions (if any)
    if result_dfs:
        result_df = pd.concat(result_dfs, ignore_index=True)
        output_file = os.path.join(pred_path, f"predictions_rank_{rank}_{file_index}.parquet")
        result_df.to_parquet(output_file, index=False)
        batch_loader.mark_files_done(completed_files)
        total_rows += len(result_df)
        file_index += 1

    logger.info(f"Inference complete. Total {file_index} files, {total_rows} new rows.")
