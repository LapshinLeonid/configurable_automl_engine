"""Setting groups shared by the HPO run and the single-model trainer.

``EncodingSettings`` (the categorical encoding of the preprocessor),
``OversamplingSettings`` (the ``DataOversampler`` step) and ``InputPolicies``
(the input-data diagnostics) are frozen value objects: ``HpoRequest`` and the
``ModelTrainer`` settings hold the same groups, so HPO and the final fit are
configured by one definition. The module holds definitions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from configurable_automl_engine.common.definitions import EncodingStrategy


@dataclass(frozen=True, slots=True)
class OversamplingSettings:
    """Oversampling of the training folds (the ``DataOversampler`` step).

    Attributes:
        enable: Whether the oversampler step is part of the trial pipeline.
        multiplier: The sample size multiplier.
        algorithm: The oversampling algorithm name.
        target_bins: The number of quantile bins of the regression strategy.
        add_noise: Whether Gaussian noise is added to the oversampled numeric
            features.
        noise_level: The noise intensity relative to the column standard
            deviation.
    """

    enable: bool = False
    multiplier: float = 1.0
    algorithm: str = "random"
    target_bins: int = 8
    add_noise: bool = False
    noise_level: float = 0.01

    def sampler_params(self) -> dict[str, Any]:
        """Return the ``DataOversampler`` constructor arguments.

        Returns:
            dict[str, Any]: The keyword arguments of ``DataOversampler``.
        """
        return {
            "multiplier": self.multiplier,
            "algorithm": self.algorithm,
            "target_bins": self.target_bins,
            "add_noise": self.add_noise,
            "noise_level": self.noise_level,
        }


@dataclass(frozen=True, slots=True)
class EncodingSettings:
    """Categorical encoding of the trial preprocessor.

    Attributes:
        encoding: The main encoding strategy (``None`` — ``'one_hot'``).
        high_cardinality_threshold: The cardinality threshold of the
            automatic high-cardinality mode (``None`` — disabled together
            with ``high_cardinality_encoding``).
        high_cardinality_encoding: The strategy of high-cardinality columns.
        hashing_n_components: The number of columns per hashed column.
        target_encoding_smoothing: The target-encoding smoothing.
        target_encoding_fallback: The target-encoding value of unknown
            categories (``None`` — the global mean).
    """

    encoding: EncodingStrategy | None = None
    high_cardinality_threshold: int | None = 50
    high_cardinality_encoding: EncodingStrategy | None = "frequency"
    hashing_n_components: int = 16
    target_encoding_smoothing: float = 20.0
    target_encoding_fallback: float | None = None


@dataclass(frozen=True, slots=True)
class InputPolicies:
    """Input-data diagnostics policies applied once on entry.

    Attributes:
        inf_policy: ``"as_missing"`` or ``"error"`` for ±inf in numeric
            features.
        datetime_policy: ``"expand"``, ``"drop"`` or ``"error"`` for
            datetime/timedelta columns.
        drop_id_like_columns: Whether ID-like columns are dropped.
    """

    inf_policy: str = "as_missing"
    datetime_policy: str = "expand"
    drop_id_like_columns: bool = False
