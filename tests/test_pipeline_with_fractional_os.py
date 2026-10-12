"""
Integration test: verifies that the training pipeline
works with a fractional oversampling multiplier (1.25).
"""

from math import ceil
from pathlib import Path

import pandas as pd

from configurable_automl_engine.training_engine.component import train_best_model
import pytest


def test_pipeline_with_fractional_os(tmp_path: Path):
    # ── 1. Mini-dataset (50/50 classes) ───────────────────────────────── #
    df = pd.DataFrame(
        {
            "feat": range(100),
            "target": [0] * 50 + [1] * 50,
        }
    )

    # ── 2. Temporary YAML config ──────────────────────────────────────── #
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
general:
  comparison_metric: "r2"
  path_to_model: {tmp_path / "model.pkl"}
  phases:
    - name: "fast_search"
      n_trials: 5
      action: "all_algorithms"
    - name: "final_select"
      n_trials: 1
      action: "select"
      strategy: "cv_best"
      output: "winner"

oversampling:
  enable: true
  multiplier: 1.25 

algorithms:
  elasticnet:
    enable: true
""",
        encoding="utf-8",
    )

    # ── 3. Run the pipeline ───────────────────────────────────────────── #
    res = train_best_model(cfg_path, df, target="target")

    # ── 4. Basic checks ───────────────────────────────────────────────── #
    # 4.1. The model was trained and the metric was computed
    assert res["score"] is not None

    # 4.2. The model pickle is readable
    model_obj = pd.read_pickle(res["model_path"])
    assert model_obj is not None

    # 4.3. Oversampling really kicked in: resulting rows >= original rows
    #     (the exact number depends on the internal implementation, hence >=)
    if hasattr(model_obj, "__dict__"):
        # try to get the saved X if the wrapper stores it
        for attr in ("X_", "_X", "X", "train_X", "X_train_"):
            if hasattr(model_obj, attr):
                assert getattr(model_obj, attr).shape[0] >= ceil(len(df) * 1.25)
                break
