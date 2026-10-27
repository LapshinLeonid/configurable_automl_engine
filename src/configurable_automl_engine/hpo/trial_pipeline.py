"""Trial pipeline assembly of the HPO runner.

Builds the preprocessor of an HPO phase and assembles the trial estimator in
the order preprocessor → [isotonic adapter] → [feature selector] →
[oversampler] → model, consistent with the final ``ModelTrainer``. It may
import ``hpo.types`` only and must not import ``hpo.runner`` or
``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib
import logging as _logging

# ──────────────────────────── project
from configurable_automl_engine.preprocessing_presets import (
    requires_missing_indicator,
)

# The records keep the logger name of the pre-split ``tuner`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
log = _logging.getLogger("configurable_automl_engine.tuner")


def _missing_indicator_enabled(algorithm: str | None) -> bool:
    """Safely determine whether missing-value indicators are needed.

    Strictly univariate algorithms (``isotonic_regression``) are incompatible
    with missing-indicator columns (their input contract is exactly one
    feature), so ``False`` is returned for them. An empty algorithm name or an
    invalid name (unknown algorithm) must not crash the HPO phase — in that
    case the safe default ``True`` is returned.

    Args:
        algorithm: Regression algorithm name; it may be
            ``None``/an empty string before name validation.

    Returns:
        ``True`` — missing indicators are enabled; ``False`` — the algorithm
        is strictly univariate.
    """
    if not algorithm:
        return True
    try:
        return requires_missing_indicator(algorithm)
    except ValueError:
        # Unknown algorithm name: do not mask the config error, but do not
        # let it surface as an unhandled exception during preprocessor
        # assembly — name validation is a separate step.
        return True
