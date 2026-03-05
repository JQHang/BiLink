"""Learning rate schedulers for distributed training."""

import math
from typing import Dict, Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LRScheduler:
    """Create a cosine learning rate schedule with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 = half cycle).

    Returns:
        LRScheduler: The learning rate scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> LRScheduler:
    """Factory function to create scheduler from config.

    Args:
        optimizer: The optimizer to schedule.
        config: Scheduler configuration dict with keys:
            - type: "constant" or "cosine_warmup"
            - warmup_ratio: Fraction of steps for warmup (default 0.1)
        num_training_steps: Total number of training steps.

    Returns:
        LRScheduler: The learning rate scheduler.

    Raises:
        ValueError: If scheduler type is unknown.
    """
    scheduler_type = config["type"]

    if scheduler_type == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)
    elif scheduler_type == "cosine_warmup":
        warmup_ratio = config.get("warmup_ratio", 0.1)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
