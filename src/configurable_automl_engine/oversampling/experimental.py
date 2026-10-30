"""Experimental status of oversampling and its warn-once WARNING.

Holds the ``OVERSAMPLING_EXPERIMENTAL`` flag (driven by the nightly evidence
benchmark), the WARNING text and its once-per-(algorithm, multiplier)
deduplication. Both the sampler and the orchestration read the flag from
this module, so a test patches it here. The module must not import the other
oversampling modules.
"""

import logging
import threading

#: Experimental status of the oversampling feature, driven by the nightly
#: evidence benchmark (``benchmarks/run_oversampling_benchmark.py``).
#: When any oversampling algorithm lowers the median held-out R² by
#: more than 0.01 relative to the no-oversampling baseline, the benchmark
#: reports the violation and this flag is set to ``True``: README and
#: API_REFERENCE then mark the feature experimental, and a WARNING is logged
#: once per process per (algorithm, multiplier) when the oversampler actually
#: runs, plus once per ``train_best_model`` call with oversampling enabled.
#
#: Current status: the nightly benchmark observed a median held-out R²
#: degradation of ADASYN on the count dataset (Δ ≈ +0.019 > 0.01), so the
#: feature is experimental until the evidence gate passes.
OVERSAMPLING_EXPERIMENTAL: bool = True


logger = logging.getLogger(__name__)

#: Text of the experimental-status WARNING (evidence gate).
EXPERIMENTAL_WARNING_MESSAGE = (
    "Oversampling is EXPERIMENTAL: the nightly evidence benchmark "
    "observed a median held-out R² degradation exceeding 0.01. "
    "Disable oversampling unless you have verified the effect on "
    "your own data."
)

# Experimental-warning deduplication: the oversampler runs once per fold per
# trial, so an unconditional WARNING floods the log (hundreds of identical
# lines in a realistic run). The WARNING is emitted once per process per
# (algorithm, multiplier) configuration — added to the set on first emission.
# The set is guarded by a lock: samplers run concurrently in the threaded
# parallel mode.
_experimental_lock = threading.Lock()
_experimental_warned: set[tuple[str, float]] = set()


def warn_experimental_once(*, algorithm: str, multiplier: float) -> None:
    """Log the experimental-status WARNING once per (algorithm, multiplier).

    Args:
        algorithm (str): The normalized oversampling algorithm name.
        multiplier (float): The sample size multiplier.

    Returns:
        None
    """
    key = (algorithm, float(multiplier))
    with _experimental_lock:
        if key in _experimental_warned:
            return
        _experimental_warned.add(key)
    logger.warning(EXPERIMENTAL_WARNING_MESSAGE)


def reset_experimental_warnings() -> None:
    """Clear the warn-once deduplication state (test helper).

    ``warn_experimental_once`` logs at most once per (algorithm, multiplier)
    pair for the whole process lifetime; tests call this helper to reset the
    recorded pairs between tests.
    """
    with _experimental_lock:
        _experimental_warned.clear()
