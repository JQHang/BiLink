"""Metrics utilities."""

import re
from typing import Dict, List

import numpy as np
from scipy.special import expit  # sigmoid function
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
)


def _precision_at_k(labels: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """Precision at top K predictions."""
    top_k_indices = np.argsort(predictions)[::-1][:k]
    return np.mean(labels[top_k_indices])


def _bce_with_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """Binary cross entropy loss for logits (applies sigmoid internally).

    Equivalent to PyTorch's F.binary_cross_entropy_with_logits.

    Args:
        labels: Ground truth binary labels (0 or 1).
        logits: Raw model outputs (before sigmoid).

    Returns:
        Mean BCE loss value.
    """
    probs = expit(logits)  # sigmoid: 1 / (1 + exp(-x))
    return log_loss(labels, probs)


def _to_native(value):
    """Convert numpy types to Python native types."""
    if hasattr(value, 'item'):  # numpy scalar (float64, int64, etc.)
        return value.item()
    return value


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric_names: List[str],
) -> Dict[str, float]:
    """Compute specified metrics.

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        metric_names: List of metric names, e.g., ['pr_auc', 'P@10000'].

    Returns:
        Dict of metric name -> value (Python native types).
    """
    metrics = {}

    for name in metric_names:
        if name == 'pr_auc':
            metrics[name] = average_precision_score(labels, predictions)
        elif name == 'roc_auc':
            metrics[name] = roc_auc_score(labels, predictions)
        elif name == 'log_loss':
            metrics[name] = log_loss(labels, predictions)
        elif name == 'bce_logits':
            metrics[name] = _bce_with_logits(labels, predictions)
        elif name == 'mse':
            metrics[name] = mean_squared_error(labels, predictions)
        elif name == 'mae':
            metrics[name] = mean_absolute_error(labels, predictions)
        elif name == 'rmse':
            metrics[name] = np.sqrt(mean_squared_error(labels, predictions))
        elif match := re.match(r'P@(\d+)', name):
            k = int(match.group(1))
            metrics[name] = _precision_at_k(labels, predictions, k)
        else:
            raise ValueError(f"Unknown metric: {name}")

    return {k: _to_native(v) for k, v in metrics.items()}
