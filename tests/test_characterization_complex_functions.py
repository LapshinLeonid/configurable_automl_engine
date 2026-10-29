"""Characterization snapshots of the six functions decomposed below C901 15.

The functions ``DataOversampler._fit_resample``, ``validate_features``,
``ModelSanityGate.check`` / ``_permutation_impl``, ``iter_splits`` and
``create_model`` used to carry ``# noqa: C901``. Before they were split into
small steps, their observable behavior was pinned here on the unmodified
code: every case records a compact digest — shapes, dtypes, hashes and
rounded sums of the arrays, the returned frame's columns / dtypes / NaN
count, the log records (logger, level, text) and the exception type and
message. A refactor that keeps the behavior keeps every digest identical,
including the order of the random draws. For ``create_model`` only the
parameters that differ from the estimator's own defaults are recorded, so the
snapshots do not depend on the installed scikit-learn / XGBoost version.

Snapshots live in ``tests/snapshots/characterization_<group>.json``.
Regenerate them only for an intended behavior change with
``UPDATE_SNAPSHOTS=1 pytest tests/test_characterization_complex_functions.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine import oversampling as oversampling_module
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.validation_utils import validate_features
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    GPR_KERNEL_TYPES,
    create_model,
    get_allowed_hyperparameters,
)
from configurable_automl_engine.oversampling import (
    DataOversampler,
    reset_experimental_warnings,
)
from configurable_automl_engine.training_engine.sanity_gate import ModelSanityGate
from configurable_automl_engine.validation import iter_splits

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_SNAPSHOTS") == "1"
PACKAGE = "configurable_automl_engine"


# ──────────────────────────────────────────────────────────────────────────────
#  Digest helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def _array_digest(values: Any) -> dict[str, Any]:
    """Shape, dtype, hash and rounded sum of an array (dense or sparse)."""
    if sparse.issparse(values):
        csr = values.tocsr()
        dense = _array_digest(csr.toarray())
        dense["sparse"] = type(values).__name__
        dense["nnz"] = int(csr.nnz)
        return dense
    arr = np.asarray(values)
    digest: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.dtype.kind in "biuf":
        as_float = np.round(arr.astype(np.float64), 10) + 0.0
        digest["sha"] = _sha(np.ascontiguousarray(as_float).tobytes())
        digest["sum"] = repr(round(float(np.nansum(as_float)), 6))
        digest["nan"] = int(np.isnan(as_float).sum())
    else:
        digest["sha"] = _sha(repr(arr.tolist()).encode())
    return digest


def _frame_digest(df: pd.DataFrame) -> dict[str, Any]:
    """Columns, dtypes, NaN count and per-column hashes of a DataFrame."""
    return {
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "shape": list(df.shape),
        "nan": int(df.isna().sum().sum()),
        "values": {str(c): _array_digest(df[c].to_numpy()) for c in df.columns},
    }


def _error_digest(err: BaseException) -> dict[str, str]:
    return {"type": type(err).__name__, "message": str(err)}


def _float(value: float) -> str:
    return repr(round(float(value), 10))


class _LogCapture(logging.Handler):
    """Collect the package log records (logger, level, rendered message)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(PACKAGE):
            self.records.append([record.name, record.levelname, record.getMessage()])


def _run(fn: Callable[[], Any]) -> dict[str, Any]:
    """Run ``fn`` with log capture; record its digest or its exception."""
    reset_experimental_warnings()
    handler = _LogCapture()
    root = logging.getLogger(PACKAGE)
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        try:
            outcome: dict[str, Any] = {"result": fn()}
        except Exception as err:  # noqa: BLE001 — the exception is the outcome
            outcome = {"error": _error_digest(err)}
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    outcome["logs"] = handler.records
    return outcome


def _check_snapshot(group: str, case_id: str, actual: dict[str, Any]) -> None:
    """Compare a case with the committed group snapshot (or write it)."""
    path = SNAPSHOT_DIR / f"characterization_{group}.json"
    data = json.loads(json.dumps(actual))
    stored = json.loads(path.read_text()) if path.exists() else {}
    if UPDATE_SNAPSHOTS:
        stored[case_id] = data
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
        return
    assert case_id in stored, f"Missing case {case_id}; run with UPDATE_SNAPSHOTS=1"
    assert data == stored[case_id]


# ──────────────────────────────────────────────────────────────────────────────
#  1. DataOversampler._fit_resample
# ──────────────────────────────────────────────────────────────────────────────


def _os_numeric(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.integers(0, 10, size=n).astype(float),
            "flag": (rng.random(n) > 0.5).astype(float),
        }
    )


def _os_categorical(n: int = 40) -> pd.DataFrame:
    df = _os_numeric(n)
    df["cat"] = np.array(["x", "y", "z", "x"] * (n // 4), dtype=object)
    return df


def _os_sparse(n: int = 40) -> Any:
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(n, 4))
    dense[rng.random((n, 4)) < 0.6] = 0.0
    dense[:, 3] = (rng.random(n) > 0.7).astype(float)
    return sparse.csr_matrix(dense)


def _os_target(task: str, n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(2)
    if task == "classification":
        return np.array([0] * (n - n // 4) + [1] * (n // 4))
    return rng.normal(loc=10.0, scale=3.0, size=n)


_OS_INPUTS: dict[str, Callable[[], Any]] = {
    "numeric": _os_numeric,
    "categorical": _os_categorical,
    "sparse": _os_sparse,
}


def _resample(sampler: DataOversampler, X: Any, y: Any) -> dict[str, Any]:
    X_res, y_res = sampler.fit_resample(X, y)
    return {
        "X_type": type(X_res).__name__,
        "X": _array_digest(X_res),
        "y": _array_digest(y_res),
    }


def _os_case(
    algorithm: str, data: str, task: str, add_noise: bool, balance: bool
) -> Callable[[], Any]:
    def run() -> Any:
        sampler = DataOversampler(
            multiplier=2.0,
            algorithm=algorithm,
            add_noise=add_noise,
            balance=balance,
            random_state=7,
            task=task,
        )
        return _resample(sampler, _OS_INPUTS[data](), _os_target(task))

    return run


def _os_missing_target() -> Any:
    y = _os_target("regression")
    y[[3, 11, 25]] = np.nan
    return _resample(
        DataOversampler(multiplier=1.5, algorithm="smote", random_state=3),
        _os_numeric(),
        y,
    )


def _os_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> Any:
    class _Broken:
        def __init__(self, **_: Any) -> None:
            pass

        def fit_resample(self, *_: Any) -> Any:
            raise LookupError("broken sampler")

    monkeypatch.setattr(oversampling_module, "RandomOverSampler", _Broken)
    return _resample(
        DataOversampler(multiplier=2.0), _os_numeric(), _os_target("regression")
    )


_OS_CASES: dict[str, Callable[..., Any]] = {
    f"{algo}-{data}-{task}-noise{int(noise)}-balance{int(balance)}": _os_case(
        algo, data, task, noise, balance
    )
    for algo in ("random", "smote", "adasyn")
    for data in ("numeric", "categorical", "sparse")
    for task in ("regression", "classification")
    for noise in (False, True)
    for balance in (False, True)
}
_OS_CASES.update(
    {
        "error-multiplier-below-one": lambda: _resample(
            DataOversampler(multiplier=0.5), _os_numeric(), _os_target("regression")
        ),
        "error-target-bins": lambda: _resample(
            DataOversampler(multiplier=2.0, target_bins=1),
            _os_numeric(),
            _os_target("regression"),
        ),
        "error-unknown-algorithm": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="Fancy Algo"),
            _os_numeric(),
            _os_target("regression"),
        ),
        "error-smote-all-categorical": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote"),
            _os_categorical()[["cat"]],
            _os_target("regression"),
        ),
        "error-smote-all-target-missing": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote"),
            _os_numeric(),
            np.full(40, np.nan),
        ),
        "error-classification-singleton-class": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote", task="classification"),
            _os_numeric(),
            np.array([0] * 39 + [1]),
        ),
        "multiplier-one-no-balance": lambda: _resample(
            DataOversampler(multiplier=1.0, algorithm="smote"),
            _os_numeric(),
            _os_target("regression"),
        ),
        "smote-missing-target": _os_missing_target,
        "smote-empty-input": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote"),
            _os_numeric().iloc[:0],
            np.empty(0),
        ),
        "smote-empty-balance-all-categorical": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote", balance=True),
            _os_categorical()[["cat"]].iloc[:0],
            np.empty(0),
        ),
        "random-empty-balance": lambda: _resample(
            DataOversampler(multiplier=2.0, balance=True),
            _os_numeric().iloc[:0],
            np.empty(0),
        ),
        "sparse-empty-balance": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm="smote", balance=True),
            sparse.csr_matrix((0, 3)),
            np.empty(0),
        ),
        "random-empty-input": lambda: _resample(
            DataOversampler(multiplier=2.0), _os_numeric().iloc[:0], np.empty(0)
        ),
        "algorithm-name-normalized": lambda: _resample(
            DataOversampler(multiplier=2.0, algorithm=" ADASYN"),
            _os_numeric(),
            _os_target("regression"),
        ),
        "ndarray-input-with-noise": lambda: _resample(
            DataOversampler(multiplier=3.0, algorithm="smote", add_noise=True),
            _os_numeric().to_numpy(),
            _os_target("regression"),
        ),
        "unexpected-error": _os_unexpected_error,
    }
)


@pytest.mark.parametrize("case_id", sorted(_OS_CASES))
def test_fit_resample_characterization(
    case_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fn = _OS_CASES[case_id]
    call = (lambda: fn(monkeypatch)) if case_id == "unexpected-error" else fn
    _check_snapshot("fit_resample", case_id, _run(call))


# ──────────────────────────────────────────────────────────────────────────────
#  2. validate_features
# ──────────────────────────────────────────────────────────────────────────────


def _vf_base(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.integers(0, 3, size=n).astype(float)}
    )


def _vf_with_inf() -> pd.DataFrame:
    df = _vf_base()
    df.loc[[1, 4], "x1"] = np.inf
    df.loc[7, "x2"] = -np.inf
    return df


def _vf_with_datetime(with_hour: bool = True) -> pd.DataFrame:
    df = _vf_base()
    freq = "7h" if with_hour else "D"
    df["ts"] = pd.date_range("2024-01-01", periods=len(df), freq=freq)
    df.loc[3, "ts"] = pd.NaT
    df["dur"] = pd.to_timedelta(np.arange(len(df)), unit="h")
    return df


def _vf_with_ids() -> pd.DataFrame:
    df = _vf_base()
    df["obj_id"] = [f"id-{i}" for i in range(len(df))]
    df["int_id"] = np.arange(100, 100 + len(df))
    df["int_desc"] = np.arange(len(df))[::-1]
    df["str_id"] = pd.array([f"s{i}" for i in range(len(df))], dtype="string")
    df["low_card"] = ["a", "b"] * (len(df) // 2)
    df["int_not_monotonic"] = np.random.default_rng(1).permutation(len(df))
    return df


def _vf(frame: Callable[[], pd.DataFrame], **kwargs: Any) -> Callable[[], Any]:
    def run() -> Any:
        source = frame()
        before = source.copy()
        out, generated = validate_features(source, **kwargs)
        return {
            "frame": _frame_digest(out),
            "generated": generated,
            "input_unchanged": bool(source.equals(before)),
        }

    return run


def _vf_duplicates() -> pd.DataFrame:
    df = _vf_base()
    df.columns = ["x1", "x1"]
    df["x3"] = 1.0
    return df


def _vf_collision() -> pd.DataFrame:
    df = _vf_with_datetime()
    df["ts_year"] = 1.0
    return df


_VF_CASES: dict[str, Callable[[], Any]] = {
    "clean": _vf(_vf_base),
    "empty-rows": _vf(lambda: _vf_with_ids().iloc[:0]),
    "duplicate-columns": _vf(_vf_duplicates),
    "unknown-inf-policy": _vf(_vf_base, inf_policy="drop"),
    "unknown-datetime-policy": _vf(_vf_base, datetime_policy="keep"),
    "inf-as-missing": _vf(_vf_with_inf),
    "inf-error": _vf(_vf_with_inf, inf_policy="error"),
    "datetime-expand": _vf(_vf_with_datetime),
    "datetime-expand-no-hour": _vf(lambda: _vf_with_datetime(with_hour=False)),
    "datetime-drop": _vf(_vf_with_datetime, datetime_policy="drop"),
    "datetime-error": _vf(_vf_with_datetime, datetime_policy="error"),
    "datetime-collision-expand": _vf(_vf_collision),
    "datetime-collision-drop": _vf(_vf_collision, datetime_policy="drop"),
    "id-like-warn": _vf(_vf_with_ids),
    "id-like-drop": _vf(_vf_with_ids, drop_id_like_columns=True),
    "id-like-drop-silent": _vf(
        _vf_with_ids, drop_id_like_columns=True, warn_id_like=False
    ),
    "id-like-silent": _vf(_vf_with_ids, warn_id_like=False),
    "all-checks": _vf(
        lambda: pd.concat(
            [_vf_with_inf(), _vf_with_datetime()[["ts"]], _vf_with_ids()[["obj_id"]]],
            axis=1,
        ),
        drop_id_like_columns=True,
    ),
}


@pytest.mark.parametrize("case_id", sorted(_VF_CASES))
def test_validate_features_characterization(case_id: str) -> None:
    _check_snapshot("validate_features", case_id, _run(_VF_CASES[case_id]))


# ──────────────────────────────────────────────────────────────────────────────
#  3. ModelSanityGate.check / _permutation_impl
# ──────────────────────────────────────────────────────────────────────────────


def _sg_data(n: int = 120, p: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(n, p))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * rng.normal(size=n)
    return X, y


class _ScriptedModel:
    """A model whose ``predict`` follows a script (failures, bad shapes)."""

    def __init__(self, base: Any, mode: str) -> None:
        self.base = base
        self.mode = mode
        self.calls = 0

    def predict(self, X: Any) -> np.ndarray:
        self.calls += 1
        if self.mode == "fail-always":
            raise RuntimeError("predict is broken")
        if self.mode == "fail-on-permuted" and self.calls > 1:
            raise RuntimeError("permuted predict is broken")
        pred = np.asarray(self.base.predict(X), dtype=float)
        if self.mode == "wrong-shape":
            return pred[:-1]
        if self.mode == "nan-base":
            return np.full_like(pred, np.nan)
        if self.mode == "nan-on-permuted" and self.calls > 1:
            pred = pred.copy()
            pred[: self.calls % 3] = np.nan
        if self.mode == "perm-wrong-shape" and self.calls % 2 == 0:
            return pred[:-1]
        return pred


class _BadIndexX:
    """A matrix-like object whose row indexing fails."""

    shape = (120, 5)
    ndim = 2

    def __getitem__(self, _: Any) -> Any:
        raise IndexError("cannot index rows")


def _sg_case(
    model_kind: str,
    *,
    with_model: bool = True,
    dead: bool = False,
    frame: bool = False,
    oof_kind: str = "noisy",
    full: str = "fit",
    x_kind: str = "matrix",
    n: int = 120,
    candidate: str | None = "cand",
    **gate_kwargs: Any,
) -> Callable[[], Any]:
    def run() -> Any:
        X, y = _sg_data(n=n)
        if dead:
            X = X.copy()
            X[:, 2:] = 0.0
        if model_kind == "linear":
            model: Any = LinearRegression().fit(X, y)
        elif model_kind == "lasso":
            model = Lasso(alpha=0.5).fit(X, y)
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
            if model_kind.startswith("tree-"):
                model = _ScriptedModel(model, model_kind.removeprefix("tree-"))
        rng = np.random.default_rng(3)
        plain = model.base if isinstance(model, _ScriptedModel) else model
        full_pred = np.asarray(plain.predict(X), dtype=float)
        if oof_kind == "noisy":
            oof = full_pred + rng.normal(scale=0.8, size=n)
        elif oof_kind == "constant":
            oof = np.full(n, float(np.mean(y)))
        elif oof_kind == "exact":
            oof = y.copy()
        else:  # partial NaN
            oof = full_pred + rng.normal(scale=0.8, size=n)
            oof[:7] = np.nan
        y_full: Any = {"fit": full_pred, "none": None, "exact": y.copy()}[full]
        X_in: Any = {
            "matrix": lambda: X,
            "frame": lambda: pd.DataFrame(
                X, columns=[f"f{i}" for i in range(X.shape[1])]
            ),
            "none": lambda: None,
            "bad-index": _BadIndexX,
            "1d": lambda: X[:, 0],
            "no-shape": X.tolist,
            "short": lambda: X[:-3],
            "sparse": lambda: sparse.csr_matrix(X),
        }["frame" if frame else x_kind]()
        gate = ModelSanityGate(**gate_kwargs)
        result = gate.check(
            y=y,
            y_pred_oof=oof,
            y_pred_full=y_full,
            X=X_in,
            model=model if with_model else None,
            candidate_name=candidate,
        )
        data = asdict(result)
        for key in ("diversity_ratio", "unique_ratio", "generalization_gap"):
            data[key] = _float(data[key])
        return data

    return run


def _sg_length_mismatch() -> Any:
    _, y = _sg_data()
    return ModelSanityGate().check(y=y, y_pred_oof=y[:-1])


def _sg_full_length_mismatch() -> Any:
    _, y = _sg_data()
    return ModelSanityGate(check_dead_features=False).check(
        y=y, y_pred_oof=y + 0.1, y_pred_full=y[:-1]
    )


_SG_CASES: dict[str, Callable[[], Any]] = {
    "linear-default": _sg_case("linear"),
    "linear-dead": _sg_case("lasso", dead=True),
    "linear-dead-low-diversity": _sg_case("lasso", dead=True, oof_kind="constant"),
    "linear-dead-no-diversity-check": _sg_case(
        "lasso", dead=True, check_diversity=False
    ),
    "linear-dead-combination-off": _sg_case(
        "lasso",
        dead=True,
        oof_kind="constant",
        dead_features_require_low_diversity=False,
    ),
    "tree-default": _sg_case("tree"),
    "tree-frame": _sg_case("tree", frame=True),
    "tree-sparse": _sg_case("tree", x_kind="sparse"),
    "tree-dead": _sg_case("tree", dead=True),
    "tree-dead-low-diversity": _sg_case("tree", dead=True, oof_kind="constant"),
    "tree-dead-no-diversity-check": _sg_case("tree", dead=True, check_diversity=False),
    "tree-dead-frame-many-rows": _sg_case(
        "tree", dead=True, frame=True, permutation_max_rows=50
    ),
    "tree-many-rows": _sg_case("tree", permutation_max_rows=50),
    "tree-many-columns": _sg_case("tree", permutation_max_features=2),
    "tree-rows-and-columns": _sg_case(
        "tree",
        permutation_max_rows=40,
        permutation_max_features=3,
        permutation_repeats=2,
    ),
    "tree-predict-fails": _sg_case("tree-fail-always"),
    "tree-permuted-predict-fails": _sg_case("tree-fail-on-permuted"),
    "tree-wrong-shape": _sg_case("tree-wrong-shape"),
    "tree-nan-base": _sg_case("tree-nan-base"),
    "tree-nan-on-permuted": _sg_case("tree-nan-on-permuted"),
    "tree-perm-wrong-shape": _sg_case("tree-perm-wrong-shape"),
    "tree-bad-index": _sg_case("tree", x_kind="bad-index"),
    "tree-1d-x": _sg_case("tree", x_kind="1d"),
    "tree-no-x": _sg_case("tree", x_kind="none"),
    "tree-no-model": _sg_case("tree", with_model=False, candidate=None),
    "tree-permutation-off": _sg_case("tree", check_permutation_sensitivity=False),
    "x-no-shape": _sg_case("tree", x_kind="no-shape"),
    "x-short": _sg_case("tree", x_kind="short"),
    "gap-warn": _sg_case("tree", full="exact", generalization_gap_mode="warn"),
    "gap-reject": _sg_case("tree", full="exact", generalization_gap_mode="reject"),
    "gap-finite-reject": _sg_case(
        "linear",
        oof_kind="noisy",
        max_generalization_gap=1.0,
        generalization_gap_mode="reject",
    ),
    "leakage-warn": _sg_case("tree", oof_kind="exact"),
    "leakage-reject": _sg_case(
        "tree", oof_kind="exact", generalization_gap_mode="reject"
    ),
    "no-full-pred": _sg_case("linear", full="none"),
    "partial-nan-oof": _sg_case("linear", oof_kind="partial-nan"),
    "constant-oof": _sg_case("tree", oof_kind="constant"),
    "constant-oof-warn-only": _sg_case("tree", oof_kind="constant", warn_only=True),
    "warn-only-reject": _sg_case(
        "tree", full="exact", generalization_gap_mode="reject", warn_only=True
    ),
    "non-adaptive-diversity": _sg_case(
        "linear", adaptive_diversity=False, min_prediction_diversity=0.99
    ),
    "all-circuits-off": _sg_case(
        "tree",
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    ),
    "length-mismatch": _sg_length_mismatch,
    "full-length-mismatch": _sg_full_length_mismatch,
}


@pytest.mark.parametrize("case_id", sorted(_SG_CASES))
def test_sanity_gate_characterization(case_id: str) -> None:
    _check_snapshot("sanity_gate", case_id, _run(_SG_CASES[case_id]))


# ──────────────────────────────────────────────────────────────────────────────
#  4. iter_splits
# ──────────────────────────────────────────────────────────────────────────────


def _is_data(n: int, p: int, kind: str) -> tuple[Any, Any]:
    X = np.column_stack([np.arange(n, dtype=float)] + [np.ones(n)] * max(p - 1, 0))
    X = X[:, :p]
    y = np.arange(n, dtype=float) * 10.0
    if kind == "pandas":
        index = pd.Index([f"r{i}" for i in range(n)])
        return pd.DataFrame(X, index=index), pd.Series(y, index=index)
    return X, y


def _split_digest(part: Any, listed: bool) -> Any:
    """Row identities of a split part: listed (test parts) or hashed (train)."""
    if part is None:
        return None
    if isinstance(part, (pd.DataFrame, pd.Series)):
        rows: Any = [str(i) for i in part.index]
    else:
        arr = np.asarray(part)
        if arr.ndim == 2 and arr.shape[1] == 0:
            return {"type": "ndarray", "shape": list(arr.shape)}
        rows = (arr[:, 0] if arr.ndim == 2 else arr / 10.0).astype(int).tolist()
    digest = {"type": type(part).__name__, "len": len(rows)}
    if listed and len(rows) <= 1000:
        return digest | {"rows": rows}
    return digest | {"sha": _sha(repr(rows).encode())}


def _is_case(
    method: str,
    kind: str,
    with_y: bool,
    include_indices: bool,
    *,
    n: int = 30,
    p: int = 2,
    **kwargs: Any,
) -> Callable[[], Any]:
    def run() -> Any:
        X, y = _is_data(n, p, kind)
        folds = []
        for split in iter_splits(
            X,
            y if with_y else None,
            method=method,
            include_indices=include_indices,
            **kwargs,
        ):
            fold = [_split_digest(part, i % 2 == 1) for i, part in enumerate(split[:4])]
            if include_indices:
                idx = split[4]
                fold.append(
                    {"type": type(idx).__name__, "dtype": str(idx.dtype)}
                    | (
                        {"idx": _array_digest(idx)}
                        if n > 1000
                        else {"idx": idx.tolist()}
                    )
                )
            folds.append(fold)
        return {"n_folds": len(folds), "folds": folds}

    return run


def _is_lazy_error() -> Any:
    gen = iter_splits(np.empty((0, 2)))
    return {"generator_created": type(gen).__name__, "first": next(gen)}


_IS_CASES: dict[str, Callable[[], Any]] = {
    f"{method}-{kind}-y{int(with_y)}-idx{int(idx)}": _is_case(method, kind, with_y, idx)
    for method in ("k_fold", "loo", "train_test_split", "auto")
    for kind in ("numpy", "pandas")
    for with_y in (True, False)
    for idx in (False, True)
}
_IS_CASES.update(
    {
        "auto-loo-small": _is_case("auto", "numpy", True, True, n=10, p=2),
        "auto-kfold-heuristic": _is_case("auto", "pandas", True, True, n=120, p=3),
        "auto-holdout-large": _is_case("auto", "numpy", True, True, n=50000, p=2),
        "auto-no-features": _is_case("auto", "numpy", True, True, n=30, p=0),
        "auto-no-features-tiny": _is_case("auto", "numpy", True, True, n=5, p=0),
        "k_fold-3-seed-1": _is_case(
            "k_fold", "numpy", True, True, n_folds=3, random_state=1
        ),
        "k_fold-fallback-small": _is_case("k_fold", "numpy", True, True, n=6),
        "holdout-test-size": _is_case(
            "train_test_split", "pandas", True, True, test_size=0.4, random_state=5
        ),
        "upper-case-method": _is_case("K_FOLD", "numpy", True, False),
        "enum-method": _is_case(ValidationStrategy.loo, "numpy", True, False, n=8),
        "unknown-method": _is_case("bootstrap", "numpy", True, False),
        "auto-invalid": _is_case("auto", "numpy", True, False, n=1, p=2),
        "empty-array": _is_case("k_fold", "numpy", True, False, n=0),
        "empty-frame": _is_case("k_fold", "pandas", True, False, n=0),
        "length-mismatch": lambda: list(
            iter_splits(np.zeros((10, 2)), np.zeros(9), method="k_fold")
        ),
        "lazy-error": _is_lazy_error,
    }
)


@pytest.mark.parametrize("case_id", sorted(_IS_CASES))
def test_iter_splits_characterization(case_id: str) -> None:
    _check_snapshot("iter_splits", case_id, _run(_IS_CASES[case_id]))


# ──────────────────────────────────────────────────────────────────────────────
#  5. create_model
# ──────────────────────────────────────────────────────────────────────────────


def _engine_params(model: Any) -> dict[str, str]:
    """The parameters ``create_model`` set: those that differ from the defaults.

    The full ``get_params()`` depends on the installed scikit-learn / XGBoost
    version (new constructor parameters, changed defaults); the difference
    with a default-constructed estimator of the same class does not.
    """
    params = model.get_params(deep=False)
    defaults = type(model)().get_params(deep=False)
    return {
        k: repr(v)
        for k, v in sorted(params.items())
        if k not in defaults or repr(v) != repr(defaults[k])
    }


def _cm(algorithm: Any, **hyperparams: Any) -> Callable[[], Any]:
    def run() -> Any:
        try:
            model = create_model(algorithm, **hyperparams)
        except ValueError as err:
            # The "Allowed parameters" list mirrors the constructor signature
            # of the installed library version: pin that it does, not the list.
            message, sep, allowed = str(err).partition(" Allowed parameters: ")
            if not sep:
                raise
            expected = sorted(get_allowed_hyperparameters(algorithm.lower()) or ())
            raise ValueError(
                f"{message} Allowed parameters: <signature of the estimator: "
                f"{'matches' if allowed == str(expected) else 'MISMATCH'}>"
            ) from err
        return {"class": type(model).__name__, "params": _engine_params(model)}

    return run


_CM_CASES: dict[str, Callable[[], Any]] = {}
for _algo in AVAILABLE_ALGORITHMS:
    _CM_CASES[f"{_algo}-default"] = _cm(_algo)
    _CM_CASES[f"{_algo}-random-state-7"] = _cm(_algo, random_state=7)
    _CM_CASES[f"{_algo}-random-state-none"] = _cm(_algo, random_state=None)
    _CM_CASES[f"{_algo}-unknown-param"] = _cm(_algo, not_a_param=1)
for _kernel in GPR_KERNEL_TYPES:
    _CM_CASES[f"gpr-kernel-{_kernel}"] = _cm(
        "gaussian_process_regression", kernel_type=_kernel
    )
_CM_CASES.update(
    {
        "gpr-kernel-unknown": _cm("gaussian_process_regression", kernel_type="cubic"),
        "gpr-explicit-kernel": _cm(
            "gaussian_process_regression",
            kernel_type="matern",
            kernel=None,
            normalize_y=False,
            n_restarts_optimizer=0,
        ),
        "isotonic-bad-out-of-bounds": _cm("isotonic_regression", out_of_bounds="wrap"),
        "isotonic-raise": _cm("isotonic_regression", out_of_bounds="raise"),
        "isotonic-feature-index": _cm("isotonic_regression", feature_index=1),
        "svr-max-iter": _cm("svr", max_iter=5),
        "case-insensitive": _cm("Random_Forest", n_estimators=3),
        "unknown-algorithm": _cm("rf"),
        "non-string-algorithm": _cm(42),
    }
)


@pytest.mark.parametrize("case_id", sorted(_CM_CASES))
def test_create_model_characterization(case_id: str) -> None:
    if case_id.startswith("xgboosting") and not is_installed("xgboost"):
        pytest.skip("xgboost is not installed")
    _check_snapshot("create_model", case_id, _run(_CM_CASES[case_id]))
