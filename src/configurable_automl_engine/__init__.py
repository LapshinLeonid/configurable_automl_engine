"""
Configurable AutoML Engine
--------------------------
A library for automated model training with flexible configuration.
"""

# Import from the subpackage where the Lazy proxy is configured
from .trainer import ModelTrainer
from .training_engine import train_best_model
from .training_engine.config_parser import Config

__all__ = [
    "Config",
    "ModelTrainer",
    "train_best_model",
]
