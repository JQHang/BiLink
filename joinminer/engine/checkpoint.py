"""Checkpoint management for distributed training."""

import os
import json
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading for distributed training.

    Uses standard checkpoint structure:
    checkpoint_dir/
    ├── latest.pt      # Overwritten each epoch
    ├── best.pt        # Overwritten when metric improves
    └── history.json   # All epochs' metrics
    """

    def __init__(self, checkpoint_dir: str):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save/load checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.latest_path = os.path.join(checkpoint_dir, 'latest.pt')
        self.best_path = os.path.join(checkpoint_dir, 'best.pt')
        self.history_path = os.path.join(checkpoint_dir, 'history.json')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        metrics: Dict[str, Any],
        best_metric: Optional[float],
        model_config: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch number.
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Learning rate scheduler to save.
            metrics: Current epoch metrics.
            best_metric: Best metric value so far.
            model_config: Model configuration for reproducibility.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'best_metric': best_metric,
            'model_config': model_config,
        }

        # Save latest.pt (always)
        torch.save(checkpoint, self.latest_path)
        logger.info(f"Saved latest checkpoint at epoch {epoch}")

        # Save best.pt (if is_best)
        if is_best:
            torch.save(checkpoint, self.best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")

        # Append to history.json
        self._append_history(epoch, metrics)

    def _append_history(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Append epoch metrics to history file."""
        history = []
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                history = json.load(f)

        history.append({'epoch': epoch, **metrics})

        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.

        Returns:
            Checkpoint dict or None if no checkpoint found.
        """
        if not os.path.exists(self.latest_path):
            return None

        checkpoint = torch.load(self.latest_path, weights_only=False, map_location='cpu')
        logger.info(f"Loaded latest checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.

        Returns:
            Checkpoint dict or None if no checkpoint found.
        """
        if not os.path.exists(self.best_path):
            return None

        checkpoint = torch.load(self.best_path, weights_only=False, map_location='cpu')
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
