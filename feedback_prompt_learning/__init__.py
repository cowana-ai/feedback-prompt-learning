"""
Feedback-Prompt-Learning: An evaluation-driven framework for automatic prompt optimization.

This package provides tools for optimizing prompts through iterative feedback,
enabling systematic improvement based on evaluation criteria rather than trial-and-error.
"""

__version__ = "0.1.0"
__author__ = "Chingis Oinar"
__email__ = "your.email@example.com"

import os
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from feedback_prompt_learning.search_algo import *  # noqa: F401, F403


def load_config(verbose=False):
    """Load configuration using Hydra"""
    # Use relative path to configs directory (relative to package root)
    with initialize(version_base="1.3.2", config_path="../configs"):
        cfg = compose(config_name="config", overrides=["optimizer=mcts"])
        if verbose:
            print(OmegaConf.to_yaml(cfg))
        return cfg


cfg = load_config()

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "cfg",
    "load_config",
]

