"""
Configurable AutoML Engine
--------------------------
Библиотека для автоматизированного обучения моделей с гибкой конфигурацией.
"""
# Импортируем из подпакета, где настроен Lazy-proxy
from .training_engine import train_best_model
from .trainer import ModelTrainer
from .training_engine.config_parser import Config
__all__ = [
    "train_best_model",
    "ModelTrainer",
    "Config",
]
