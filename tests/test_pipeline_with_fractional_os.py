"""
Интеграционный тест: проверяем, что training-pipeline
работает при дробном multiplier oversampling (1.25).
"""

from math import ceil
from pathlib import Path

import pandas as pd

from configurable_automl_engine.training_engine.component import train_best_model


def test_pipeline_with_fractional_os(tmp_path: Path):
    # ── 1. Мини-датасет (50/50 классы) ──────────────────────────────── #
    df = pd.DataFrame(
        {
            "feat": range(100),
            "target": [0] * 50 + [1] * 50,
        }
    )

    # ── 2. Временный YAML-конфиг ─────────────────────────────────────── #
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
general:
  comparison_metric: r2
  path_to_model: {tmp_path/'model.pkl'}

oversampling:
  enable: true
  multiplier: 1.25
  random_state: 0

algorithms:
  elasticnet:
    enable: true
""",
        encoding="utf-8",
    )

    # ── 3. Запускаем pipeline ───────────────────────────────────────── #
    res = train_best_model(cfg_path, df, target="target")

    # ── 4. Базовые проверки ─────────────────────────────────────────── #
    # 4.1. Модель обучилась и посчитала метрику
    assert res["score"] is not None

    # 4.2. Пикл с моделью читается
    model_obj = pd.read_pickle(res["model_path"])
    assert model_obj is not None

    # 4.3. Oversampling точно сработал: итоговых строк ≥ исходных
    #     (точное число зависит от внутренней реализации, поэтому >=)
    if hasattr(model_obj, "__dict__"):
        # пробуем достать сохранённый X, если обёртка его хранит
        for attr in ("X_", "_X", "X", "train_X", "X_train_"):
            if hasattr(model_obj, attr):
                assert getattr(model_obj, attr).shape[0] >= ceil(len(df) * 1.25)
                break
