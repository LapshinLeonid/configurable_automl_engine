"""Tests for artifact metadata and format auto-detection on load.

Covers the issue #159 scope:
- ``ModelTrainer.save`` populates ``artifact_metadata_`` (engine version,
  dependency versions, UTC timestamp, algorithm, params, config hash).
- ``ModelTrainer.load(path, fmt=None)`` auto-detects the format from the
  extension, falling back to trying joblib then pickle.
- A WARNING is emitted when the major/minor version of the engine or
  scikit-learn differs from the current ones.
- ``__version__`` is exported from the package ``__init__``.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import (
    resolve_load_formats,
)
from configurable_automl_engine.trainer import ModelTrainer, TrainingError

_LOGGER_NAME = "configurable_automl_engine.trainer"


@pytest.fixture
def trained_trainer() -> ModelTrainer:
    """A cheap, fitted ModelTrainer (elasticnet, tiny synthetic data)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {"a": rng.random(40), "b": rng.random(40), "c": rng.choice(["x", "y"], 40)}
    )
    y = pd.Series(X["a"] * 2 - X["b"] + rng.normal(0, 0.01, 40))
    trainer = ModelTrainer(
        algorithm="elasticnet",
        hyperparams={"alpha": 0.1, "l1_ratio": 0.5},
        metric="r2",
        random_state=42,
    )
    trainer.fit(X, y)
    return trainer


def _records_from_trainer_logger(caplog: pytest.LogCaptureFixture) -> list[Any]:
    """Log records only from the trainer module."""
    return [r for r in caplog.records if r.name == _LOGGER_NAME]


# ══════════════════════════════════════════════════════════════════════════
# Package __version__ export
# ══════════════════════════════════════════════════════════════════════════


def test_package_exports_version() -> None:
    """``configurable_automl_engine.__version__`` is exported."""
    assert hasattr(configurable_automl_engine, "__version__")
    assert isinstance(configurable_automl_engine.__version__, str)
    assert configurable_automl_engine.__version__


# ══════════════════════════════════════════════════════════════════════════
# Metadata population on save
# ══════════════════════════════════════════════════════════════════════════


def test_save_populates_artifact_metadata(trained_trainer: ModelTrainer, tmp_path: Path) -> None:
    """save() fills artifact_metadata_ with the documented fields."""
    assert trained_trainer.artifact_metadata_ is None
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)

    meta = trained_trainer.artifact_metadata_
    assert meta is not None
    assert meta["engine_version"] == configurable_automl_engine.__version__
    assert "python_version" in meta
    assert meta["algorithm"] == "elasticnet"
    assert meta["params"] == {"alpha": 0.1, "l1_ratio": 0.5}

    versions = meta["versions"]
    for key in (
        "python",
        "numpy",
        "pandas",
        "scikit-learn",
        "optuna",
        "imbalanced-learn",
        "xgboost",
    ):
        assert key in versions

    # The timestamp is an ISO-8601 UTC datetime string.
    parsed = dt.datetime.fromisoformat(meta["created_at_utc"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None

    # The config hash is a stable hex digest.
    assert isinstance(meta["config_hash"], str)
    assert len(meta["config_hash"]) == 64


def test_save_load_roundtrip_metadata_preserved(
    trained_trainer: ModelTrainer, tmp_path: Path
) -> None:
    """The metadata written by save() survives a load()."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)
    loaded = ModelTrainer.load(path)
    assert loaded.artifact_metadata_ == trained_trainer.artifact_metadata_


def test_save_untrained_model_does_not_touch_metadata(tmp_path: Path) -> None:
    """save() on an untrained trainer raises and does not set metadata."""
    trainer = ModelTrainer()
    assert trainer.artifact_metadata_ is None
    with pytest.raises(TrainingError, match="Nothing to save"):
        trainer.save(tmp_path / "model.pkl")
    assert trainer.artifact_metadata_ is None


def test_metadata_is_not_set_before_save(trained_trainer: ModelTrainer) -> None:
    """artifact_metadata_ is None before the first save()."""
    assert trained_trainer.artifact_metadata_ is None


# ══════════════════════════════════════════════════════════════════════════
# Format auto-detection on load
# ══════════════════════════════════════════════════════════════════════════


def test_load_joblib_without_fmt(trained_trainer: ModelTrainer, tmp_path: Path) -> None:
    """A .joblib artifact loads with fmt=None (acceptance criterion)."""
    trained_trainer.serialization_format = SerializationFormat.joblib
    path = tmp_path / "model.joblib"
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path)  # fmt not passed
    assert isinstance(loaded, ModelTrainer)
    # Predictions must match exactly (no retraining on load).
    preds_orig = trained_trainer.predict(_df_for_predict())
    loaded_preds = loaded.predict(_df_for_predict())
    assert np.allclose(loaded_preds, preds_orig)


def _df_for_predict() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"a": rng.random(5), "b": rng.random(5), "c": rng.choice(["x", "y"], 5)}
    )


def test_load_pickle_without_fmt(trained_trainer: ModelTrainer, tmp_path: Path) -> None:
    """A .pkl artifact loads with fmt=None."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert isinstance(loaded, ModelTrainer)
    assert np.allclose(
        loaded.predict(_df_for_predict()), trained_trainer.predict(_df_for_predict())
    )


def test_load_joblib_wrong_extension_falls_back(
    trained_trainer: ModelTrainer, tmp_path: Path
) -> None:
    """A joblib artifact with a .pkl extension still loads via the fallback.

    The extension suggests pickle first; pickle fails, then joblib is tried.
    """
    trained_trainer.serialization_format = SerializationFormat.joblib
    path = tmp_path / "model.pkl"  # misnamed
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert isinstance(loaded, ModelTrainer)
    assert np.allclose(
        loaded.predict(_df_for_predict()), trained_trainer.predict(_df_for_predict())
    )


def test_load_pickle_wrong_extension_falls_back(
    trained_trainer: ModelTrainer, tmp_path: Path
) -> None:
    """A pickle artifact with a .joblib extension still loads via the fallback."""
    path = tmp_path / "model.joblib"  # misnamed: pickle content
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert isinstance(loaded, ModelTrainer)
    assert np.allclose(
        loaded.predict(_df_for_predict()), trained_trainer.predict(_df_for_predict())
    )


def test_load_unknown_extension(trained_trainer: ModelTrainer, tmp_path: Path) -> None:
    """An unknown extension tries joblib first, then pickle."""
    trained_trainer.serialization_format = SerializationFormat.joblib
    path = tmp_path / "model.art"
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert isinstance(loaded, ModelTrainer)

    # pickle content with unknown extension also works (joblib fails, pickle wins).
    trained_trainer.serialization_format = SerializationFormat.pickle
    path2 = tmp_path / "model2.art"
    trained_trainer.save(path2)
    loaded2 = ModelTrainer.load(path2)
    assert isinstance(loaded2, ModelTrainer)


def test_load_explicit_fmt_still_works(
    trained_trainer: ModelTrainer, tmp_path: Path
) -> None:
    """Passing an explicit fmt restricts loading to that format (backward compat)."""
    trained_trainer.serialization_format = SerializationFormat.joblib
    path = tmp_path / "model.joblib"
    trained_trainer.save(path)

    loaded = ModelTrainer.load(path, fmt=SerializationFormat.joblib)
    assert isinstance(loaded, ModelTrainer)


def test_load_explicit_wrong_fmt_fails(
    trained_trainer: ModelTrainer, tmp_path: Path
) -> None:
    """Passing a wrong explicit fmt raises instead of falling back."""
    trained_trainer.serialization_format = SerializationFormat.joblib
    path = tmp_path / "model.joblib"
    trained_trainer.save(path)

    with pytest.raises(TrainingError, match="Error loading artifact"):
        ModelTrainer.load(path, fmt=SerializationFormat.pickle)


# ══════════════════════════════════════════════════════════════════════════
# Version mismatch warnings
# ══════════════════════════════════════════════════════════════════════════


def test_load_warns_on_sklearn_major_minor_mismatch(
    trained_trainer: ModelTrainer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A different scikit-learn major/minor triggers a WARNING."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)
    # Simulate an artifact created with another scikit-learn version.
    with open(path, "rb") as f:
        obj = pickle.load(f)
    obj.artifact_metadata_["versions"]["scikit-learn"] = "1.0.0"
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        ModelTrainer.load(path)

    records = _records_from_trainer_logger(caplog)
    assert any(
        "scikit-learn" in r.message and "1.0.0" in r.message and "current" in r.message
        for r in records
    )


def test_load_warns_on_engine_major_minor_mismatch(
    trained_trainer: ModelTrainer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A different engine major/minor triggers a WARNING."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    obj.artifact_metadata_["versions"]["configurable_automl_engine"] = "9.9.9"
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        ModelTrainer.load(path)

    records = _records_from_trainer_logger(caplog)
    assert any(
        "configurable_automl_engine" in r.message
        and "9.9.9" in r.message
        and "current" in r.message
        for r in records
    )


def test_load_no_warning_on_matching_versions(
    trained_trainer: ModelTrainer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Matching major/minor versions produce no version-mismatch warning."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        ModelTrainer.load(path)

    records = _records_from_trainer_logger(caplog)
    assert not any(
        "Artifact was created with" in r.message for r in records
    )


def test_load_patch_level_difference_no_warning(
    trained_trainer: ModelTrainer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A patch-level difference (same major.minor) does not warn."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    current_sklearn = obj.artifact_metadata_["versions"]["scikit-learn"]
    major_minor = current_sklearn.rsplit(".", 1)[0]
    obj.artifact_metadata_["versions"]["scikit-learn"] = f"{major_minor}.0"
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        ModelTrainer.load(path)

    records = _records_from_trainer_logger(caplog)
    assert not any(
        "Artifact was created with" in r.message for r in records
    )


def test_load_legacy_artifact_without_metadata_no_warning(
    trained_trainer: ModelTrainer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Artifacts saved before metadata existed (no attribute) load without warnings."""
    path = tmp_path / "model.pkl"
    trained_trainer.save(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    del obj.artifact_metadata_
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        loaded = ModelTrainer.load(path)

    assert isinstance(loaded, ModelTrainer)
    records = _records_from_trainer_logger(caplog)
    assert not any(
        "Artifact was created with" in r.message for r in records
    )


# ══════════════════════════════════════════════════════════════════════════
# Load error handling (backward compatibility)
# ══════════════════════════════════════════════════════════════════════════


def test_load_file_not_found(tmp_path: Path) -> None:
    """Loading a missing file raises TrainingError with a clear message."""
    with pytest.raises(TrainingError, match="File not found"):
        ModelTrainer.load(tmp_path / "missing.pkl")


def test_load_not_a_trainer(tmp_path: Path) -> None:
    """Loading a non-ModelTrainer object raises TrainingError."""
    path = tmp_path / "dummy.pkl"
    with open(path, "wb") as f:
        pickle.dump({"not": "a trainer"}, f)
    with pytest.raises(TrainingError, match="Loaded object is not a ModelTrainer"):
        ModelTrainer.load(path)


def test_load_corrupted_file_raises(tmp_path: Path) -> None:
    """A corrupted artifact raises TrainingError (not a raw exception)."""
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"this is not a valid pickle stream")
    with pytest.raises(TrainingError, match="Error loading artifact"):
        ModelTrainer.load(path)


# ══════════════════════════════════════════════════════════════════════════
# resolve_load_formats unit tests
# ══════════════════════════════════════════════════════════════════════════


def test_resolve_load_formats_explicit() -> None:
    """An explicit fmt is returned as the single candidate."""
    assert resolve_load_formats("m.pkl", SerializationFormat.joblib) == [
        SerializationFormat.joblib
    ]


def test_resolve_load_formats_pkl_extension() -> None:
    """.pkl prefers pickle, with joblib as fallback."""
    assert resolve_load_formats("m.pkl", None) == [
        SerializationFormat.pickle,
        SerializationFormat.joblib,
    ]


def test_resolve_load_formats_pickle_extension() -> None:
    """.pickle prefers pickle, with joblib as fallback."""
    assert resolve_load_formats("m.pickle", None) == [
        SerializationFormat.pickle,
        SerializationFormat.joblib,
    ]


def test_resolve_load_formats_joblib_extension() -> None:
    """.joblib prefers joblib, with pickle as fallback."""
    assert resolve_load_formats("m.joblib", None) == [
        SerializationFormat.joblib,
        SerializationFormat.pickle,
    ]


def test_resolve_load_formats_unknown_extension() -> None:
    """An unknown extension tries joblib first, then pickle."""
    assert resolve_load_formats("m.art", None) == [
        SerializationFormat.joblib,
        SerializationFormat.pickle,
    ]


def test_resolve_load_formats_case_insensitive() -> None:
    """Extension detection is case-insensitive."""
    assert resolve_load_formats("m.JOBLIB", None) == [
        SerializationFormat.joblib,
        SerializationFormat.pickle,
    ]
    assert resolve_load_formats("m.PKL", None) == [
        SerializationFormat.pickle,
        SerializationFormat.joblib,
    ]


# ══════════════════════════════════════════════════════════════════════════
# Internal version helpers (branch coverage)
# ══════════════════════════════════════════════════════════════════════════

from configurable_automl_engine.trainer import (  # noqa: E402
    _installed_module_version,
    _major_minor,
    _version_tuple,
    _warn_on_version_mismatch,
)


def test_version_tuple_with_non_numeric_component() -> None:
    """Non-numeric components are preserved as strings."""
    assert _version_tuple("1.2rc1") == (1, "2rc1")
    assert _version_tuple("1.2.3.post1") == (1, 2, 3, "post1")
    assert _version_tuple("dev0") == ("dev0",)


def test_major_minor_unparseable_returns_none() -> None:
    """Versions with fewer than two numeric components yield None."""
    assert _major_minor("one") is None
    assert _major_minor("1") is None
    assert _major_minor("x.y") is None


def test_installed_module_version_missing_module() -> None:
    """A missing module yields None instead of raising."""
    assert _installed_module_version("definitely_not_a_module_xyz") is None


def test_warn_on_version_mismatch_non_dict_metadata(caplog: pytest.LogCaptureFixture) -> None:
    """Non-dict metadata is ignored without raising."""
    logger = logging.getLogger(_LOGGER_NAME)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_on_version_mismatch("not a dict", logger)
        _warn_on_version_mismatch({"versions": "not a dict"}, logger)
        _warn_on_version_mismatch({"versions": {}}, logger)
    records = _records_from_trainer_logger(caplog)
    assert not any(
        "Artifact was created with" in r.message for r in records
    )


def test_warn_on_version_mismatch_unparseable_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unparseable stored version does not trigger a warning."""
    logger = logging.getLogger(_LOGGER_NAME)
    metadata = {
        "versions": {
            "configurable_automl_engine": "unknown",
            "scikit-learn": "unknown",
        }
    }
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_on_version_mismatch(metadata, logger)
    records = _records_from_trainer_logger(caplog)
    assert not any(
        "Artifact was created with" in r.message for r in records
    )
