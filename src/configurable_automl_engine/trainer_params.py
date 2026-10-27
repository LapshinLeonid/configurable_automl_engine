"""Validation of the ``ModelTrainer`` constructor arguments.

Every helper validates (and, where documented, normalizes) one group of the
constructor arguments and raises ``TrainingError`` with the same messages as
before, so invalid values are rejected at initialization time, before
training starts. ``ModelTrainer.__init__`` calls the helpers in the original
validation order and assigns the attributes itself (attribute names are
part of the pickle contract). The module must not import
``configurable_automl_engine.trainer``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride
from configurable_automl_engine.validation import (
    SUPPORTED_VALIDATION_METHODS,
    norm_val_method,
)

from .common.definitions import ValidationStrategy
from .trainer_errors import TrainingError

_ENCODING_STRATEGIES = ("one_hot", "ordinal", "target", "frequency", "hashing")


def _validate_algorithm(algorithm: Any) -> str:
    """Validate the algorithm name and return it lowercased."""
    if not isinstance(algorithm, str):
        raise TrainingError(f"Invalid algorithm: {algorithm!r}")
    return algorithm.lower()


def _validate_hyperparams(hyperparams: Any) -> dict[str, Any]:
    """Validate the hyperparameters (``None`` → an empty dict)."""
    if hyperparams is not None:
        if not isinstance(hyperparams, dict):
            raise TrainingError("hyperparams must be a dictionary")
        return hyperparams
    return {}


def _validate_input_policies(
    *,
    inf_policy: str,
    datetime_policy: str,
    drop_id_like_columns: Any,
    warn_id_like: Any,
) -> None:
    """Validate the input-data diagnostics policies."""
    if inf_policy not in ("as_missing", "error"):
        raise TrainingError(
            f"Unknown inf_policy: {inf_policy!r}. Expected 'as_missing' or 'error'."
        )
    if datetime_policy not in ("expand", "drop", "error"):
        raise TrainingError(
            f"Unknown datetime_policy: {datetime_policy!r}. "
            "Expected 'expand', 'drop' or 'error'."
        )
    if not isinstance(drop_id_like_columns, bool):
        raise TrainingError(
            "drop_id_like_columns must be a bool, got "
            f"{type(drop_id_like_columns).__name__}"
        )
    if not isinstance(warn_id_like, bool):
        raise TrainingError(
            f"warn_id_like must be a bool, got {type(warn_id_like).__name__}"
        )


def _validate_encoding_strategy(encoding_strategy: str) -> EncodingStrategy:
    """Validate the categorical encoding strategy."""
    if encoding_strategy not in _ENCODING_STRATEGIES:
        raise TrainingError(
            f"Unknown encoding_strategy: {encoding_strategy!r}. "
            "Expected one of ('one_hot', 'ordinal', 'target', 'frequency', "
            "'hashing')."
        )
    return cast(EncodingStrategy, encoding_strategy)


def _validate_high_cardinality(
    *,
    high_cardinality_threshold: int | None,
    high_cardinality_encoding: str | None,
    hashing_n_components: int,
    target_encoding_smoothing: float,
) -> None:
    """Validate the high-cardinality encoding parameters."""
    if (high_cardinality_threshold is None) != (high_cardinality_encoding is None):
        raise TrainingError(
            "high_cardinality_threshold and high_cardinality_encoding must "
            "be set together (both provided or both None)."
        )
    if high_cardinality_threshold is not None and high_cardinality_threshold < 0:
        raise TrainingError(
            "high_cardinality_threshold must be >= 0, got "
            f"{high_cardinality_threshold}."
        )
    if (
        high_cardinality_encoding is not None
        and high_cardinality_encoding not in _ENCODING_STRATEGIES
    ):
        raise TrainingError(
            f"Unknown high_cardinality_encoding: {high_cardinality_encoding!r}."
        )
    if hashing_n_components < 1:
        raise TrainingError(
            f"hashing_n_components must be >= 1, got {hashing_n_components}."
        )
    if target_encoding_smoothing < 0:
        raise TrainingError(
            f"target_encoding_smoothing must be >= 0, got {target_encoding_smoothing}."
        )


def _normalize_additional_metrics(
    additional_metrics: Iterable[str] | None,
) -> list[str]:
    """Validate the additional metrics and return the lowercased names.

    Any iterable sequence of strings is accepted (list, tuple, set,
    generator, etc.). Scalar strings and mappings (dict) are rejected: a
    string cannot be iterated character-by-character into metric names, and
    iterating a dict over keys would be implicit and surprising behavior. The
    iterable is materialized exactly once so that generators and other
    one-shot iterators are validated correctly.
    """
    if additional_metrics is not None and isinstance(
        additional_metrics, (str, bytes, Mapping)
    ):
        raise TrainingError("additional_metrics must be an iterable of strings")
    normalized_metrics = list(additional_metrics or [])
    if not all(isinstance(m, str) for m in normalized_metrics):
        raise TrainingError("additional_metrics must be an iterable of strings")
    return [m.lower() for m in normalized_metrics]


def _validate_preprocessing_override(
    override: PreprocessingOverride | dict[str, Any] | None,
) -> PreprocessingOverride | None:
    """Validate and normalize a user-provided preprocessing preset override.

    Args:
        override: A dict or a :class:`PreprocessingOverride` model.

    Returns:
        The normalized override model, or ``None``.

    Raises:
        TrainingError: If an object of an unsupported type or a dict with
            invalid values is passed.
    """
    if override is None:
        return None
    if isinstance(override, PreprocessingOverride):
        return override
    try:
        if isinstance(override, dict):
            return PreprocessingOverride.model_validate(override)
    except Exception as e:
        raise TrainingError(f"Invalid preprocessing_override: {e}") from e
    raise TrainingError(
        "preprocessing_override must be a dict or PreprocessingOverride, "
        f"got {type(override).__name__}"
    )


def _validate_feature_selection_active(value: Any) -> bool | None:
    """Validate the explicit feature selection flag.

    The type is strictly bool | None: the string "false" from the
    configuration/YAML must not silently become True via bool("false") in
    _resolve_feature_selection_active, so non-boolean values are rejected at
    initialization time, like the other constructor parameters.
    """
    if value is not None and not isinstance(value, bool):
        raise TrainingError(
            "feature_selection_active must be a bool or None, got "
            f"{type(value).__name__}"
        )
    return value


def _validate_validation_strategy(
    validation_strategy: ValidationStrategy | str, n_folds: Any
) -> str:
    """Validate the validation strategy and the fold count.

    The list of allowed methods is taken from
    validation.SUPPORTED_VALIDATION_METHODS — the single extension point of
    the strategy interface: adding a new strategy (e.g., TimeSeriesSplit)
    does not require changes here.

    Returns:
        The normalized strategy name.
    """
    normalized_strategy = norm_val_method(validation_strategy)
    if normalized_strategy not in SUPPORTED_VALIDATION_METHODS:
        raise TrainingError(
            f"Unknown validation_strategy: {validation_strategy!r}. "
            f"Expected one of {SUPPORTED_VALIDATION_METHODS}."
        )
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 1:
        raise TrainingError(f"n_folds must be a positive integer, got {n_folds!r}")
    if normalized_strategy == "k_fold" and n_folds < 2:
        raise TrainingError(
            f"n_folds must be >= 2 for k_fold validation, got {n_folds}"
        )
    return normalized_strategy


def _validate_test_size(test_size: Any) -> Any:
    """Validate the hold-out size.

    test_size accepts a fraction in (0, 1) or an integer number of rows (as
    in sklearn train_test_split): an integer test_size comes from the engine
    path after 'auto' resolution (choose_validation_method returns a row
    count).
    """
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TrainingError(f"test_size must be a number, got {test_size!r}")
    if isinstance(test_size, float) and not (0.0 < test_size < 1.0):
        raise TrainingError(
            f"test_size must be a fraction in (0, 1), got {test_size!r}"
        )
    if isinstance(test_size, int) and test_size < 1:
        raise TrainingError(
            f"test_size must be a positive integer (row count), got {test_size!r}"
        )
    return test_size


def _validate_oversampling(
    *, multiplier: float, algorithm: str, target_bins: int, noise_level: float
) -> None:
    """Validate the oversampling settings."""
    if multiplier < 1:
        raise TrainingError("data_oversampling_multiplier must be ≥ 1")
    if algorithm not in {"random", "smote", "adasyn"}:
        raise TrainingError("Unknown data_oversampling_algorithm")
    if target_bins < 2:
        raise TrainingError("data_oversampling_target_bins must be ≥ 2")
    if noise_level <= 0:
        raise TrainingError("data_oversampling_noise_level must be > 0")


def _validate_feature_lists(categorical_features: Any, numerical_features: Any) -> None:
    """Validate the explicit categorical/numerical column lists."""
    for attr_name, features in [
        ("categorical_features", categorical_features),
        ("numerical_features", numerical_features),
    ]:
        if features is not None and (
            not isinstance(features, list)
            or not all(isinstance(f, str) for f in features)
        ):
            raise TrainingError(
                f"Parameter {attr_name} must be a list of strings (column names)"
            )
