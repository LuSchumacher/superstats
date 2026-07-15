"""Data-augmentation processes for generative models."""

from .missing_process import MissingProcess
from .random_missing import RandomMissing

__all__ = [
    "MissingProcess",
    "RandomMissing",
]
