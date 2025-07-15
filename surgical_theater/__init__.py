"""
SurgicalTheater: Zero-copy model validation during training.

A lightweight library for memory-efficient temporary model modifications,
enabling safe validation and experimentation without GPU memory overhead.
"""

__version__ = "0.1.0"
__author__ = "SurgicalTheater Contributors"

from .core import SurgicalTheater, surgical_theater

__all__ = [
    "SurgicalTheater",
    "surgical_theater",
]