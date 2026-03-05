"""Training worker function for distributed training."""

import json
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .distributed import setup_ddp, get_device
from .checkpoint import CheckpointManager
from .scheduler import create_scheduler
from .metrics import compute_metrics
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


CRITERION_REGISTRY = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "huber": nn.HuberLoss,
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
}


def train(rank: int, world_size: int, port: int, config: Dict[str, Any]) -> None:
    """Training worker function.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
        port: DDP communication port.
        config: Unified config dict with sections:
            - model: {name, ...model-specific params...}
            - dataset: {train, val, test}
            - device: {type}
            - training: {epochs}
            - optimizer: {learning_rate, weight_decay}
            - scheduler: {type, warmup_ratio}
            - early_stopping: {patience, monitor, mode}
            - checkpoint
            - criterion
            - metrics: ['pr_auc', 'P@10000', ...]
            - dataloader: {train: {...}, eval: {...}}
    """
    device_type = config["device"]["type"]

    # Setup device for this rank
    if device_type == "npu":
        import torch_npu  # type: ignore
        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)

    # Setup distributed environment
    if world_size > 1:
        setup_ddp(rank, world_size, port, device_type)

    device = get_device(rank, device_type)

    # Create dataset
    dataset_config = config["dataset"]
    train_dataset = Dataset(dataset_config["train"], rank=rank, world_size=world_size)
    val_dataset = Dataset(dataset_config["val"], rank=rank, world_size=world_size)
    test_dataset = Dataset(dataset_config["test"], rank=rank, world_size=world_size)

    # Get batch counts
    train_num_batches = train_dataset.batch_loader.rank_batch_count
    val_num_batches = val_dataset.batch_loader.rank_batch_count
    test_num_batches = test_dataset.batch_loader.rank_batch_count

    # Create dataloader
    train_dataloader = DataLoader(train_dataset, **config["dataloader"]["train"])
    val_dataloader = DataLoader(val_dataset, **config["dataloader"]["eval"])
    test_dataloader = DataLoader(test_dataset, **config["dataloader"]["eval"])

    # Create model
    model_config = config["model"]
    model = MODEL_REGISTRY[model_config["name"]](model_config).to(device)

    # Create optimizer
    optimizer_config = config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )

    # Create criterion
    criterion = CRITERION_REGISTRY[config["criterion"]]()

    # Create scheduler
    total_steps = config["training"]["epochs"] * train_num_batches
    scheduler = create_scheduler(optimizer, config["scheduler"], total_steps)

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(config["checkpoint"])

    # Initialize tracking variables
    start_epoch = 0
    best_metric = None
    patience_counter = 0

    # Load checkpoint if exists
    checkpoint = checkpoint_manager.load_latest()
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        patience_counter = checkpoint['metrics']['patience_counter']
        logger.debug(f"Resumed from epoch {checkpoint['epoch']}")

    # Early stopping config
    early_stopping_config = config["early_stopping"]
    patience = early_stopping_config["patience"]
    monitor = early_stopping_config["monitor"]
    mode = early_stopping_config["mode"]

    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.debug(f"Processing epoch {epoch}")

        # Check early stopping
        if patience_counter >= patience:
            logger.debug("Early stopping triggered")
            break

        # Set epoch for shuffle seed
        train_dataset.batch_loader.set_epoch(epoch)

        # Train epoch
        train_loss = _train_epoch(
            model, train_dataloader, optimizer, scheduler,
            criterion, train_num_batches, epoch, rank, world_size, device
        )

        # Validation
        val_metrics = _eval_epoch(
            model, val_dataloader, val_num_batches, epoch, "Validation",
            rank, world_size, device, config["metrics"]
        )

        # Test
        test_metrics = _eval_epoch(
            model, test_dataloader, test_num_batches, epoch, "Test",
            rank, world_size, device, config["metrics"]
        )

        # Check improvement
        is_best = False
        if val_metrics:
            current_val = val_metrics[monitor]

            if best_metric is None:
                is_best = True
            elif mode == "max":
                is_best = current_val > best_metric
            else:
                is_best = current_val < best_metric

            if is_best:
                best_metric = current_val
                patience_counter = 0
            else:
                patience_counter += 1

        # Build epoch metrics
        epoch_metrics = {
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "patience_counter": patience_counter
        }

        # Save checkpoint (only rank 0)
        if rank == 0:
            # Use unwrapped model to avoid 'module.' prefix in state_dict keys
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_manager.save(
                epoch=epoch,
                model=model_to_save,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics,
                best_metric=best_metric,
                model_config=model_config,
                is_best=is_best,
            )
        logger.debug(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"patience={patience_counter}/{patience}"
        )

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterion,
    num_batches: int,
    epoch: int,
    rank: int,
    world_size: int,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, total=num_batches, desc=f"Train epoch {epoch}") if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to_device(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if rank == 0 and batch_idx % 10 == 0:
            pbar.set_postfix(loss=loss.item())

    # Aggregate loss across ranks
    total_loss_tensor = torch.tensor(total_loss).to(device)

    if world_size > 1:
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss_tensor / (num_batches * world_size)).cpu().item()
    return avg_loss


def _eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    num_batches: int,
    epoch: int,
    desc: str,
    rank: int,
    world_size: int,
    device: str,
    metric_names: list,
) -> Dict[str, Any]:
    """Evaluate for one epoch."""
    if device.startswith('npu'):
        _set_npu_eval_mode(model)
    else:
        model.eval()

    all_outputs = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"{desc} epoch {epoch}", total=num_batches) if rank == 0 else dataloader

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to_device(device)

            outputs = model(batch)
            all_outputs.append(outputs)
            all_labels.append(batch.label)

    # Concatenate local results
    outputs_tensor = torch.cat(all_outputs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Gather across ranks
    if world_size > 1:
        gathered_outputs = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_outputs, outputs_tensor.cpu().numpy())
        dist.all_gather_object(gathered_labels, labels_tensor.cpu().numpy())
        all_outputs = torch.cat([torch.from_numpy(arr) for arr in gathered_outputs], dim=0)
        all_labels = torch.cat([torch.from_numpy(arr) for arr in gathered_labels], dim=0)
    else:
        all_outputs = outputs_tensor
        all_labels = labels_tensor

    # Convert to numpy and compute metrics
    outputs_np = all_outputs.cpu().numpy().flatten()
    labels_np = all_labels.cpu().numpy().flatten()

    metrics = compute_metrics(outputs_np, labels_np, metric_names)

    logger.debug(f"{desc}: evaluated {len(labels_np)} samples")
    logger.debug(f"{desc} metrics: {json.dumps(metrics, indent=2)}")

    return metrics
