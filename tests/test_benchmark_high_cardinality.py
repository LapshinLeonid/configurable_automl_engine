"""Regression performance test for the high-cardinality encoding fix.

Scenario: a frame with a unique-per-row string column ("ID
scenario"). Before the fix the default one-hot expanded the column into ``n``
binary columns and produced a dense ``n × n`` float64 matrix per fold (~3 GB at
20k rows, 14x wall time). After the fix:

* the default config routes columns with cardinality > 50 to the
  ``frequency`` encoding (1 column), and
* sparse one-hot keeps even the explicit ``one_hot`` encoding of an ID column
  in CSR storage instead of a dense ``n × n`` matrix.

Acceptance bounds (must hold on the CI runner with healthy margins):

* wall time of the ID scenario <= 3x the control (numerical-only) wall time;
* peak memory (tracemalloc) <= 2x the control peak memory.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.preprocessing import build_preprocessor
from configurable_automl_engine.trainer import ModelTrainer

#: Rows of the performance-test frame. 3000 rows reproduce the original
#: n×n blow-up (a dense 3000×3001 float64 matrix ≈ 72 MB per fit) while
#: keeping the test fast enough for CI.
_N_ROWS = 3000

#: Wall-time bound: ID scenario <= 3x the control (same frame without the
#: ID column).
_WALL_TIME_RATIO = 3.0

#: Peak-memory bound: ID scenario <= 2x the control peak.
_MEMORY_RATIO = 2.0


def _control_frame() -> pd.DataFrame:
    """A purely numerical frame that mimics the ID scenario without the ID."""
    rng = np.random.default_rng(0)
    n = _N_ROWS
    return pd.DataFrame({"num1": rng.normal(size=n), "num2": rng.normal(size=n)})


def _id_frame() -> pd.DataFrame:
    """A frame with a unique-per-row string column (the ID scenario)."""
    rng = np.random.default_rng(1)
    n = _N_ROWS
    return pd.DataFrame({"id": [f"id_{i}" for i in range(n)], "num": rng.normal(size=n)})


def test_high_cardinality_preprocessing_within_bounds_on_control():
    """Wall time of the ID preprocessing <= 3x the numerical-only control.

    The regression made a 3000-row ID frame 14x slower than the
    numeric control; with the automatic high-cardinality routing (threshold 50
    + frequency) and sparse one-hot the ratio must stay well under 3x.
    """
    rng = np.random.default_rng(0)
    control = _control_frame()
    y_control = control["num1"] * 2.0 + control["num2"] + rng.normal(0, 0.1, _N_ROWS)

    tracemalloc.start()
    t0 = time.monotonic()
    pre_control = build_preprocessor(
        list(control.columns),
        categorical_features=[],
        numerical_features=list(control.columns),
    )
    pre_control.fit_transform(control)
    control_wall = time.monotonic() - t0
    _, control_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The ID frame through the DEFAULT engine settings: high-cardinality
    # (3000 > 50) -> frequency (1 column), so the matrix is 3000 x 2.
    df_id = _id_frame()
    y_id = df_id["num"] * 2.0 + rng.normal(0, 0.1, _N_ROWS)

    tracemalloc.start()
    t0 = time.monotonic()
    pre_id = build_preprocessor(
        list(df_id.columns),
        categorical_features=["id"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=50,
        high_cardinality_encoding="frequency",
    )
    out_id = pre_id.fit_transform(df_id)
    id_wall = time.monotonic() - t0
    _, id_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Sanity: the frequency-encoded matrix has exactly 2 columns (not n+1).
    assert getattr(out_id, "shape", None) == (_N_ROWS, 2)

    assert id_wall <= _WALL_TIME_RATIO * control_wall + 0.5, (
        f"ID preprocessing wall time {id_wall:.3f}s exceeds "
        f"{_WALL_TIME_RATIO}x the control {control_wall:.3f}s"
    )
    assert id_peak <= _MEMORY_RATIO * max(control_peak, 1) + 1e6, (
        f"ID preprocessing peak memory {id_peak} exceeds "
        f"{_MEMORY_RATIO}x the control {control_peak}"
    )


def test_sparse_one_hot_id_no_dense_blow_up():
    """Explicit one-hot of an ID column stays sparse (no n×n dense matrix).

    Even without the automatic high-cardinality routing, the sparse one-hot
    must not allocate the dense n×n float64 matrix that caused the OOM/14x
    regression.
    """
    df = _id_frame()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["id"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = pre.fit_transform(df)
    assert hasattr(out, "nnz"), "expected a sparse output for the high-cardinality one-hot"
    # nnz = n one-hot entries + n numeric entries — not n*n.
    assert out.nnz <= 2 * _N_ROWS


def test_trainer_high_cardinality_id_scenario_within_bounds():
    """End-to-end ModelTrainer fit on the ID scenario stays within the
    time/memory bounds vs the numerical-only control."""
    rng = np.random.default_rng(2)
    control = _control_frame()
    y_control = control["num1"] * 2.0 + control["num2"] + rng.normal(0, 0.1, _N_ROWS)

    tracemalloc.start()
    t0 = time.monotonic()
    ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        categorical_features=[],
        numerical_features=list(control.columns),
    ).fit(control, y_control)
    control_wall = time.monotonic() - t0
    _, control_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_id = _id_frame()
    y_id = df_id["num"] * 2.0 + rng.normal(0, 0.1, _N_ROWS)

    tracemalloc.start()
    t0 = time.monotonic()
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        categorical_features=["id"],
        numerical_features=["num"],
        high_cardinality_threshold=50,
        high_cardinality_encoding="frequency",
    ).fit(df_id, y_id)
    id_wall = time.monotonic() - t0
    _, id_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert trainer.val_score is not None
    assert id_wall <= _WALL_TIME_RATIO * control_wall + 2.0, (
        f"ID trainer wall time {id_wall:.3f}s exceeds {_WALL_TIME_RATIO}x "
        f"the control {control_wall:.3f}s"
    )
    assert id_peak <= _MEMORY_RATIO * max(control_peak, 1) + 2e6, (
        f"ID trainer peak memory {id_peak} exceeds {_MEMORY_RATIO}x "
        f"the control {control_peak}"
    )
