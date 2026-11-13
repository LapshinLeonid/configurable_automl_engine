"""The multi-split (repeated) validation core and config parameters.

Covers M1 of the repeated-validation epic:

- ``make_cv``/``iter_splits`` with ``n_repeats > 1`` yield R deterministic
  repeat sets with the seeds ``random_state + r``; with ``n_repeats = 1``
  the output is byte-in-byte the classic single-split one (the mandatory
  invariant);
- the repeat index is an attribute tag on the yielded tuple (the tuple
  shape and the ``include_indices=True`` OOF channel are unchanged);
- ``general.n_repeats`` / ``general.split_aggregation`` config fields and
  the cross-rules (invalid values, 'loo' + repeats, computational budget
  guard, splits-per-trial sanity limit);
- the 'mean'/'median' aggregation of per-repeat scores;
- edge cases: tiny samples (the repeated hold-out identical-split WARNING),
  R > n_folds, 'auto' resolution, pandas inputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from configurable_automl_engine.validation import (
    SKLEARN_SEED_MODULUS,
    RepeatedSplit,
    aggregate_split_scores,
    iter_splits,
    make_cv,
    norm_val_method,
    repeat_index_of,
    repeated_seed,
    resolved_n_repeats,
    with_repeat_index,
)


def _data(n: int = 24, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=n)
    return X, y


def _test_indices(
    X: np.ndarray,
    y: np.ndarray | None,
    **kwargs: object,
) -> list[list[int]]:
    """The validation-row positions of every yielded split, as plain lists."""
    return [
        np.asarray(split[-1]).tolist()
        for split in iter_splits(X, y, include_indices=True, **kwargs)  # type: ignore[arg-type]
    ]


# ───────────────────── deterministic repeat seeds ─────────────────────── #


@pytest.mark.parametrize("base", [None, 0, 7, 42, 123])
def test_repeated_seed_formula(base: int | None) -> None:
    """The seed of repeat r is exactly ``random_state + r`` (None stays None)."""
    assert repeated_seed(None, 0) is None
    assert repeated_seed(None, 5) is None
    if base is not None:
        for r in (0, 1, 4, 9):
            assert repeated_seed(base, r) == base + r


def test_resolved_n_repeats_basic() -> None:
    assert resolved_n_repeats(None, "k_fold") == 1
    assert resolved_n_repeats(1, "k_fold") == 1
    assert resolved_n_repeats(5, "train_test_split") == 5
    assert resolved_n_repeats(5, "auto") == 5


def test_resolved_n_repeats_loo_ignored_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """'loo' + n_repeats > 1: a WARNING via the module logger, repeats = 1."""
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        assert resolved_n_repeats(5, "loo") == 1
    assert any("IGNORED" in record.message for record in caplog.records)


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_resolved_n_repeats_rejects_non_positive(bad: int) -> None:
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        resolved_n_repeats(bad, "k_fold")


def test_repeat_index_tag_roundtrip() -> None:
    split = (np.zeros(2), np.zeros(1), None, None)
    tagged = with_repeat_index(split, 3)
    assert isinstance(tagged, RepeatedSplit)
    assert repeat_index_of(tagged) == 3
    assert repeat_index_of(split) == 0
    # The tuple shape/elements are unchanged: unpacking keeps working.
    a, b, c, d = tagged
    assert len(tagged) == 4


# ───────────────────── iter_splits: n_repeats > 1 ─────────────────────── #


def test_k_fold_repeats_count_and_tags() -> None:
    X, y = _data(24)
    n_folds, repeats = 3, 4
    splits = list(
        iter_splits(X, y, method="k_fold", n_folds=n_folds, n_repeats=repeats)
    )
    assert len(splits) == n_folds * repeats
    assert [repeat_index_of(s) for s in splits] == [
        r for r in range(repeats) for _ in range(n_folds)
    ]
    assert all(isinstance(s, RepeatedSplit) for s in splits)


def test_repeats_deterministic_two_calls_identical() -> None:
    """Two runs with the same random_state give identical repeated splits."""
    X, y = _data(24)
    first = _test_indices(
        X, y, method="k_fold", n_folds=3, n_repeats=5, random_state=42
    )
    second = _test_indices(
        X, y, method="k_fold", n_folds=3, n_repeats=5, random_state=42
    )
    assert first == second


def test_repeat_seeds_are_base_plus_r() -> None:
    """Repeat r of k_fold equals a single run seeded with random_state + r."""
    X, y = _data(24)
    repeats = 4
    repeated = _test_indices(
        X, y, method="k_fold", n_folds=3, n_repeats=repeats, random_state=42
    )
    per_repeat = [repeated[r * 3 : (r + 1) * 3] for r in range(repeats)]
    for r, expected in enumerate(per_repeat):
        single = _test_indices(X, y, method="k_fold", n_folds=3, random_state=42 + r)
        assert expected == single


def test_holdout_repeats_deterministic_and_distinct() -> None:
    X, y = _data(24)
    repeats = 4
    first = _test_indices(
        X,
        y,
        method="train_test_split",
        test_size=0.25,
        n_repeats=repeats,
        random_state=42,
    )
    second = _test_indices(
        X,
        y,
        method="train_test_split",
        test_size=0.25,
        n_repeats=repeats,
        random_state=42,
    )
    assert first == second
    # The repeats actually re-randomize the hold-out: distinct test parts.
    assert len({tuple(part) for part in first}) == repeats


def test_holdout_repeat_zero_equals_single_run() -> None:
    X, y = _data(24)
    repeated = _test_indices(
        X, y, method="train_test_split", test_size=0.25, n_repeats=3, random_state=42
    )
    single = _test_indices(
        X, y, method="train_test_split", test_size=0.25, random_state=42
    )
    assert repeated[0] == single[0]


def test_k_fold_repeat_zero_equals_single_run() -> None:
    X, y = _data(24)
    repeated = _test_indices(
        X, y, method="k_fold", n_folds=4, n_repeats=3, random_state=7
    )
    single = _test_indices(X, y, method="k_fold", n_folds=4, random_state=7)
    assert repeated[:4] == single


def test_auto_strategy_repeats_work() -> None:
    X, y = _data(60)
    splits = list(iter_splits(X, y, method="auto", n_repeats=3, random_state=42))
    n_cycles = 3
    assert len(splits) >= n_cycles
    tags = [repeat_index_of(s) for s in splits]
    assert tags == [r for r in range(n_cycles) for _ in range(len(splits) // n_cycles)]


def test_repeats_with_pandas_inputs() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.normal(size=30))
    splits = list(
        iter_splits(X, y, method="k_fold", n_folds=3, n_repeats=2, include_indices=True)
    )
    assert len(splits) == 6
    # The indices are positional (iloc-aligned) and the frames are preserved.
    assert all(isinstance(s[0], pd.DataFrame) for s in splits)
    assert all(np.asarray(s[4]).max(initial=-1) < 30 for s in splits)


def test_holdout_repeats_identical_splits_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tiny sample where the repeated hold-out cannot vary: a WARNING."""
    X, _ = _data(2)
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        splits = list(
            iter_splits(X, method="train_test_split", test_size=0.5, n_repeats=8)
        )
    assert len(splits) == 8
    assert any("identical splits" in record.message for record in caplog.records)


# ──────────────────── the n_repeats = 1 invariant ─────────────────────── #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "k_fold", "n_folds": 3},
        {"method": "train_test_split", "test_size": 0.25},
        {"method": "loo"},
        {"method": "auto"},
    ],
)
def test_n_repeats_one_is_byte_in_byte_classic(kwargs: dict) -> None:
    """``n_repeats=1`` (and None) reproduce the classic single-split output."""
    X, y = _data(24)
    classic = [
        (
            np.asarray(s[0]).tolist(),
            np.asarray(s[1]).tolist(),
            np.asarray(s[3]).tolist(),
        )
        for s in iter_splits(X, y, include_indices=True, **kwargs)  # type: ignore[arg-type]
    ]
    for repeats in (None, 1):
        with_repeats = [
            (
                np.asarray(s[0]).tolist(),
                np.asarray(s[1]).tolist(),
                np.asarray(s[3]).tolist(),
            )
            for s in iter_splits(
                X,
                y,
                include_indices=True,
                n_repeats=repeats,
                **kwargs,  # type: ignore[arg-type]
            )
        ]
        assert classic == with_repeats
    # And no repeat tags leak into the single-split mode.
    untagged = list(iter_splits(X, y, **kwargs))  # type: ignore[arg-type]
    assert all(not isinstance(s, RepeatedSplit) for s in untagged)


def test_n_repeats_one_invariant_loo_counts() -> None:
    X, y = _data(10)
    assert len(list(iter_splits(X, y, method="loo", n_repeats=1))) == 10


def test_n_repeats_zero_is_an_error_not_runtime_crash() -> None:
    X, y = _data(24)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            list(iter_splits(X, y, n_repeats=bad))


# ─────────────────────────── make_cv surface ──────────────────────────── #


def test_make_cv_accepts_n_repeats_and_normalizes() -> None:
    method, cv, _ = make_cv(
        100, val_method="k_fold", n_folds=5, random_state=42, test_size=0.2, n_repeats=3
    )
    assert method == "k_fold"
    assert cv is not None


def test_make_cv_n_repeats_negative_rejected() -> None:
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        make_cv(
            100,
            val_method="k_fold",
            n_folds=5,
            random_state=42,
            test_size=0.2,
            n_repeats=0,
        )


# ───────────────────── score aggregation (pure) ───────────────────────── #


def test_aggregate_split_scores_mean_and_median() -> None:
    scores = [0.1, 0.5, 0.9]
    assert aggregate_split_scores(scores, "mean") == pytest.approx(0.5)
    assert aggregate_split_scores(scores, "median") == pytest.approx(0.5)
    skewed = [0.05, 0.06, 0.07, 0.08, 0.9]
    assert aggregate_split_scores(skewed, "mean") > aggregate_split_scores(
        skewed, "median"
    )
    assert aggregate_split_scores(skewed, "median") == pytest.approx(0.07)


def test_aggregate_split_scores_drops_nan() -> None:
    """A failed repeat (NaN) cannot poison the combined estimate."""
    assert aggregate_split_scores(
        [float("nan"), 0.3, float("inf")], "median"
    ) == pytest.approx(0.3)
    assert np.isnan(aggregate_split_scores([float("nan")]))


def test_aggregate_split_scores_unknown_policy_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown split aggregation"):
        aggregate_split_scores([0.1], "sum")


def test_aggregate_split_scores_default_is_median() -> None:
    assert aggregate_split_scores([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ───────────────────────── config: GeneralCfg ──────────────────────────── #


def _phases() -> list[dict]:
    return [
        {"name": "search", "n_trials": 2, "action": "all_algorithms"},
        {
            "name": "final_select",
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
    ]


def test_general_cfg_multi_split_defaults() -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    cfg = GeneralCfg(phases=[])
    assert cfg.n_repeats == 1
    assert cfg.split_aggregation == "median"


def test_general_cfg_multi_split_valid_values() -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    cfg = GeneralCfg(phases=[], n_repeats=5, split_aggregation="median")
    assert cfg.n_repeats == 5
    assert cfg.split_aggregation == "median"
    cfg_mean = GeneralCfg(phases=[], n_repeats=2, split_aggregation="mean")
    assert cfg_mean.split_aggregation == "mean"


@pytest.mark.parametrize("bad", [0, -1])
def test_general_cfg_n_repeats_non_positive_rejected(bad: int) -> None:
    """``n_repeats: 0`` and negatives are config-validation errors."""
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    with pytest.raises(ValidationError, match="n_repeats"):
        GeneralCfg(phases=[], n_repeats=bad)


def test_general_cfg_unknown_split_aggregation_rejected() -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    with pytest.raises(ValidationError, match="split_aggregation"):
        GeneralCfg(phases=[], split_aggregation="sum")


def test_general_cfg_loo_with_repeats_warns_not_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    with caplog.at_level(
        logging.WARNING,
        logger="configurable_automl_engine.training_engine.config.general_rules",
    ):
        cfg = GeneralCfg(phases=[], n_repeats=3, validation_strategy="loo")
    assert cfg.n_repeats == 3
    assert any("IGNORED" in record.message for record in caplog.records)


def test_general_cfg_budget_guard_error() -> None:
    """A prohibitive trial budget is a config error (not a slow run)."""
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    phases = [
        {"name": "search", "n_trials": 3000, "action": "all_algorithms"},
        {
            "name": "final_select",
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
    ]
    with pytest.raises(ValidationError, match="prohibitive"):
        GeneralCfg(phases=phases, n_repeats=5, n_folds=10)  # 3000*10*5


def test_general_cfg_budget_guard_warn_zone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    phases = [
        {"name": "search", "n_trials": 50, "action": "all_algorithms"},
        {
            "name": "final_select",
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
    ]
    with caplog.at_level(
        logging.WARNING,
        logger="configurable_automl_engine.training_engine.config.general_rules",
    ):
        cfg = GeneralCfg(phases=phases, n_repeats=10, n_folds=5)  # 50*5*10 = 2500
    assert cfg.n_repeats == 10
    assert any("budget is high" in record.message for record in caplog.records)


def test_general_cfg_splits_per_trial_sanity_limit() -> None:
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    with pytest.raises(ValidationError, match="sanity limit"):
        GeneralCfg(phases=_phases(), n_repeats=10, n_folds=10)


def test_general_cfg_budget_guard_ignored_for_n_repeats_one() -> None:
    """The guard is a multi-split rule: n_repeats=1 is never checked."""
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    phases = [
        {"name": "search", "n_trials": 3000, "action": "all_algorithms"},
        {
            "name": "final_select",
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
    ]
    cfg = GeneralCfg(phases=phases, n_folds=10)
    assert cfg.n_repeats == 1


def test_config_schema_contains_multi_split_fields() -> None:
    """The regenerated config.schema.json carries the new fields."""
    import json
    from pathlib import Path

    schema_path = Path(__file__).resolve().parents[1] / "config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    general = schema["$defs"]["GeneralCfg"]["properties"]
    assert general["n_repeats"]["default"] == 1
    assert general["n_repeats"]["minimum"] == 1
    assert general["split_aggregation"]["default"] == "median"
    assert general["split_aggregation"]["enum"] == ["mean", "median"]


def test_config_json_multi_split_roundtrip() -> None:
    """A JSON config with n_repeats/split_aggregation parses end-to-end."""
    from configurable_automl_engine.training_engine.config_parser import Config

    cfg = {
        "general": {
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "n_repeats": 5,
            "split_aggregation": "median",
            "phases": _phases(),
        },
        "algorithms": {"elasticnet": {"enable": True}},
    }
    parsed = Config.model_validate(cfg)
    assert parsed.general.n_repeats == 5
    assert parsed.general.split_aggregation == "median"


# ─────────────────────────── edge cases ────────────────────────────────── #


def test_k_fold_repeats_with_r_greater_than_n_folds() -> None:
    """R > n_folds is valid: the repeats do not depend on the fold count."""
    X, y = _data(30)
    splits = list(iter_splits(X, y, method="k_fold", n_folds=2, n_repeats=7))
    assert len(splits) == 14
    assert max(repeat_index_of(s) for s in splits) == 6


def test_k_fold_fallback_to_holdout_repeats_really_randomize() -> None:
    """The k_fold → train_test_split fallback repeats re-randomize (not identical)."""
    X, _ = _data(5)  # n < max(4, 2*n_folds) → the hold-out fallback
    splits = list(
        iter_splits(
            X, method="k_fold", n_folds=3, test_size=0.34, n_repeats=6, random_state=42
        )
    )
    assert len(splits) == 6
    test_parts = {tuple(np.asarray(s[1]).ravel().tolist()) for s in splits}
    assert len(test_parts) > 1


def test_loo_repeats_effectively_single(caplog: pytest.LogCaptureFixture) -> None:
    """'loo' + repeats: a WARNING and exactly the classic LOO split count."""
    X, y = _data(8)
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        splits = list(iter_splits(X, y, method="loo", n_repeats=5))
    assert len(splits) == 8
    assert any("IGNORED" in record.message for record in caplog.records)


def test_norm_val_method_unaffected_by_multi_split() -> None:
    assert norm_val_method("k_fold") == "k_fold"
    assert (
        norm_val_method("repeated") == "repeated"
    )  # not a strategy: make_cv rejects it
    with pytest.raises(ValueError, match="Unknown validation method"):
        make_cv(100, val_method="repeated", n_folds=5, random_state=42, test_size=0.2)


# ─────────────── review fixes: resolved-method guard & seeds ───────────── #


def test_auto_resolved_to_loo_ignores_repeats_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """'auto' resolving to LOO (micro-sample) must hit the same guard as an
    explicit 'loo': repeats ignored with a WARNING, no duplicate work."""
    X, y = _data(8)
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        splits = list(iter_splits(X, y, method="auto", n_repeats=5))
    assert len(splits) == 8  # exactly one LOO cycle, not 5
    assert all(repeat_index_of(s) == 0 for s in splits)
    assert any("IGNORED" in record.message for record in caplog.records)


def test_resolved_n_repeats_rejects_non_integer() -> None:
    """``2.7`` is not silently coerced to 2 by ``int()``; bool is rejected."""
    with pytest.raises(ValueError, match="n_repeats must be a positive integer"):
        resolved_n_repeats(2.7, "k_fold")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_repeats must be a positive integer"):
        resolved_n_repeats(True, "k_fold")


@pytest.mark.parametrize("r", [0, 1, 2, 10])
def test_repeated_seed_stays_in_sklearn_range(r: int) -> None:
    """``random_state + r`` never leaves [0, 2**32 - 1] (sklearn rejects
    out-of-range seeds with ValueError at split construction)."""
    base = SKLEARN_SEED_MODULUS - 1  # the maximum valid sklearn seed
    seed = repeated_seed(base, r)
    assert seed is not None and 0 <= seed < SKLEARN_SEED_MODULUS


def test_max_base_seed_repeats_do_not_crash() -> None:
    """A config valid on main (``random_state = 2**32 - 1``) with
    ``n_repeats >= 2`` must not fail at split construction."""
    X, y = _data(24)
    splits = list(
        iter_splits(
            X, y, method="k_fold", n_folds=3, n_repeats=3, random_state=2**32 - 1
        )
    )
    assert len(splits) == 9
    splits_h = list(
        iter_splits(
            X,
            y,
            method="train_test_split",
            test_size=0.25,
            n_repeats=3,
            random_state=2**32 - 1,
        )
    )
    assert len(splits_h) == 3


def test_max_base_seed_repeats_stay_deterministic() -> None:
    """The seed wrap keeps the determinism invariant for wrapped seeds."""
    X, y = _data(24)
    first = _test_indices(
        X, y, method="k_fold", n_folds=3, n_repeats=3, random_state=2**32 - 1
    )
    second = _test_indices(
        X, y, method="k_fold", n_folds=3, n_repeats=3, random_state=2**32 - 1
    )
    assert first == second


def test_k_fold_repeats_seed_wrap_is_transparent_for_small_seeds() -> None:
    """For in-range seeds the wrap is a no-op (ordinary configs unchanged)."""
    assert repeated_seed(42, 0) == 42
    assert repeated_seed(42, 5) == 47


def test_aggregate_split_scores_warns_on_non_finite(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """±inf/NaN repeat scores are excluded with a WARNING, not silently."""
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        result = aggregate_split_scores([float("inf"), 0.3, 0.5], "median")
    assert result == pytest.approx(0.4)  # median of [0.3, 0.5]
    assert any("non-finite" in record.message for record in caplog.records)


def test_aggregate_split_scores_all_non_finite_warns_and_returns_nan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        result = aggregate_split_scores([float("nan"), float("inf")], "mean")
    assert np.isnan(result)
    assert any("non-finite" in record.message for record in caplog.records)


def test_general_cfg_budget_guard_holdout_not_multiplied_by_n_folds() -> None:
    """A hold-out repeat is ONE split: `train_test_split + n_repeats=60`
    must not be falsely rejected by the fold-based budget estimate."""
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    cfg = GeneralCfg(phases=[], n_repeats=60, validation_strategy="train_test_split")
    assert cfg.n_repeats == 60


def test_general_cfg_budget_guard_holdout_still_counts_trials() -> None:
    """The hold-out budget guard still applies to the trial count."""
    from configurable_automl_engine.training_engine.config_parser import GeneralCfg

    phases = [
        {"name": "search", "n_trials": 20000, "action": "all_algorithms"},
        {
            "name": "final_select",
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
    ]
    with pytest.raises(ValidationError, match="prohibitive"):
        GeneralCfg(
            phases=phases, n_repeats=60, validation_strategy="train_test_split"
        )  # 20000 * 60


def test_k_fold_repeats_n_samples_below_folds_times_repeats_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The n_samples check of issue-211: a tiny sample cannot give R*n_folds
    independent fold evaluations — a WARNING at the split point."""
    X, y = _data(6)
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        splits = list(
            iter_splits(X, y, method="k_fold", n_folds=3, n_repeats=5, random_state=42)
        )
    assert len(splits) == 15  # the splits are still produced
    assert any(
        "smaller than n_folds * n_repeats" in record.message
        for record in caplog.records
    )


def test_k_fold_repeats_n_samples_sufficient_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    X, y = _data(60)
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.validation"
    ):
        list(
            iter_splits(X, y, method="k_fold", n_folds=5, n_repeats=5, random_state=42)
        )
    assert not any(
        "smaller than n_folds * n_repeats" in record.message
        for record in caplog.records
    )
