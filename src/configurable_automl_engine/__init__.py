"""
Configurable AutoML Engine
--------------------------
Библиотека для автоматизированного обучения моделей с гибкой конфигурацией.
"""
# Импорт основных компонентов для экспорта в публичный API
from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.config_parser import Config
# Определение списка экспортируемых объектов (PEP 8)
__all__ = [
    "train_best_model",
    "ModelTrainer",
    "Config",
]
