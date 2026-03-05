"""Model module."""

from .bipathsnn import BiPathsNN

MODEL_REGISTRY = {
    "bipathsnn": BiPathsNN,
}

__all__ = ["MODEL_REGISTRY"]
