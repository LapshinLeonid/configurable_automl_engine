"""
Юнит-тесты формирования пула финалистов Top-K (эпик #61, задача T3, issue #65).

Покрывают функцию ``select_finalists`` из ``training_engine/component.py``:

• коридор δ в пользовательской семантике (направление из ``user_direction``,
  issue #54): multiplicative для ошибок, additive для score-метрик со знаком;
• жёсткий кап ``top_k_candidates`` и явная комбинация «коридор ∩ top_k»;
• опциональный мягкий добор семейств (``enforce_family_diversity``);
• исключение дисквалифицированных алгоритмов (circuit breaker, issue #12);
• детерминизм: порядок по «сырому» скору, ничьи — порядок конфигурации.

Формат ``results`` соответствует контракту ``phase_results``: «алгоритм →
(сырой скор, параметры)», где сырой скор — значение в семантике оптимизатора
(для метрик-ошибок — инвертированное, ``to_user_value`` возвращает
естественное).
"""

from __future__ import annotations

from typing import Any

import pytest

from configurable_automl_engine.training_engine.component import (
    FAMILY_DIVERSITY_MULTIPLIER_DEFAULT,
    select_finalists,
)

R = dict[str, tuple[float, dict[str, Any]]]

FAMILIES = {
    "elasticnet": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "svr": "kernel",
    "gaussian_process_regression": "kernel",
    "random_forest": "ensemble",
    "gradient_boosting": "ensemble",
    "extra_trees": "ensemble",
}


# ──────────────────────────────────────────────────────────────────────────────
# Positive: пример из постановки и базовые коридоры
# ──────────────────────────────────────────────────────────────────────────────
def test_positive_example_from_issue_statement():
    """ElasticNet (0.081), SVR (0.085), Random Forest (0.089) → пул при δ=0.15.

    Метрика rmse (min-better), auto-режим → multiplicative:
    порог = 0.081 × 1.15 = 0.09315, все три кандидата внутри коридора.
    """
    results: R = {
        "elasticnet": (-0.081, {"alpha": 0.1}),
        "svr": (-0.085, {"C": 1.0}),
        "random_forest": (-0.089, {"n_estimators": 100}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]


def test_min_better_multiplicative_corridor_includes_within_delta():
    """min-better: кандидаты с ошибкой ≤ best×(1+δ) попадают в пул."""
    results: R = {
        "ridge": (-0.05, {}),
        "svr": (-0.055, {}),
        "random_forest": (-0.06, {}),
    }
    # Порог = 0.05 × 1.2 = 0.06: SVR (0.055) и RF (0.06, ровно граница) внутри.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.2, top_k_candidates=3
    )
    assert pool == ["ridge", "svr", "random_forest"]


def test_max_better_multiplicative_corridor_explicit_mode():
    """max-better: порог = best×(1−δ); явный multiplicative для r2."""
    results: R = {
        "ridge": (0.90, {}),
        "svr": (0.85, {}),
        "random_forest": (0.80, {}),
    }
    # Порог = 0.90 × 0.85 = 0.765: 0.85 и 0.80 внутри, 0.70 — вне.
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.15,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["ridge", "svr", "random_forest"]


def test_auto_mode_selects_multiplicative_for_errors():
    """auto для ошибки (rmse) → multiplicative: кандидат за порогом вне пула."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "xgboosting": (-0.09, {}),
    }
    # Порог = 0.081 × 1.05 = 0.08505: 0.09 > порога → xgboosting вне пула.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.05, corridor_mode="auto"
    )
    assert pool == ["elasticnet"]


def test_auto_mode_selects_additive_for_score_metric():
    """auto для r2 (score-метрика со знаком) → additive, а не multiplicative."""
    results: R = {
        "a": (-0.5, {}),
        "b": (-0.55, {}),
    }
    # additive: range = 0.05, порог = 0.15×0.05 = 0.0075 → b вне пула.
    pool = select_finalists(results, metric_user="r2", corridor_delta=0.15)
    assert pool == ["a"]


# ──────────────────────────────────────────────────────────────────────────────
# Negative: кандидаты вне коридора и слабые семейства
# ──────────────────────────────────────────────────────────────────────────────
def test_negative_model_outside_corridor_excluded():
    """Модель с CV-скоро вне коридора не попадает в пул (min-better)."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "svr": (-0.085, {}),
        "random_forest": (-0.089, {}),
        "xgboosting": (-0.20, {}),  # 0.20 > 0.09315 → вне коридора
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]


def test_negative_model_outside_corridor_excluded_max_better():
    """max-better: модель с r2 ниже порога best×(1−δ) не попадает в пул."""
    results: R = {
        "ridge": (0.90, {}),
        "svr": (0.85, {}),
        "random_forest": (0.70, {}),  # 0.70 < 0.765 → вне коридора
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.15,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["ridge", "svr"]


def test_negative_family_diversity_disabled_no_topup():
    """enforce_family_diversity=false: слабая модель другого семейства не добирается.

    В базовом пуле только линейные (одно семейство); слабый SVR другого
    семейства вне базового коридора не добавляется.
    """
    results: R = {
        "elasticnet": (-0.081, {}),
        "ridge": (-0.082, {}),
        "svr": (-0.095, {}),  # вне базового коридора (0.095 > 0.09315)
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=False,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge"]


# ──────────────────────────────────────────────────────────────────────────────
# Boundary: top_k, отрицательные скоры, один кандидат, дисквалификация
# ──────────────────────────────────────────────────────────────────────────────
def test_boundary_top_k_equals_one():
    """top_k=1: пул из одного лидера — эквивалент текущего поведения."""
    results: R = {
        "ridge": (0.90, {}),
        "svr": (0.85, {}),
        "random_forest": (0.80, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=1,
    )
    assert pool == ["ridge"]


def test_boundary_pool_capped_at_top_k():
    """Пул больше top_k: жёсткий кап обрезает до top_k_candidates."""
    results: R = {
        "ridge": (0.90, {}),
        "svr": (0.85, {}),
        "random_forest": (0.80, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="multiplicative",
        top_k_candidates=2,
    )
    assert pool == ["ridge", "svr"]


def test_boundary_single_valid_algorithm():
    """Один валидный алгоритм — пул из одного кандидата."""
    pool = select_finalists(
        {"ridge": (0.7, {})}, metric_user="r2", top_k_candidates=3
    )
    assert pool == ["ridge"]


def test_boundary_corridor_gives_less_than_top_k():
    """Коридор даёт меньше кандидатов, чем top_k: пул из того, что есть.

    Добор за пределы коридора не производится (без опционального правила).
    """
    results: R = {
        "elasticnet": (-0.081, {}),
        "xgboosting": (-0.20, {}),  # вне коридора
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet"]


def test_boundary_negative_r2_additive_form():
    """Отрицательные r2-скоры: additive-форма корректна, граница включается.

    range = 0.1, порог = 0.5×0.1 = 0.05: |−0.55 − (−0.5)| = 0.05 ровно на
    границе → включается (задокументированный tie-break), |−0.6+0.5| = 0.1 —
    вне.
    """
    results: R = {
        "a": (-0.5, {}),
        "b": (-0.55, {}),
        "c": (-0.6, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="additive",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_multiplicative_empty_with_negative_best_falls_back_to_additive():
    """Коридор пуст при max-better с отрицательным лидером → additive.

    Явный multiplicative для r2: порог = −0.5×0.85 = −0.425, лидер −0.5 не
    проходит (коридор пуст) → выбирается аддитивная форма; при δ=1.0 оба
    кандидата попадают в пул.
    """
    results: R = {
        "a": (-0.5, {}),
        "b": (-0.6, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=1.0,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_mae_near_zero_additive_form():
    """MAE≈0: явная additive-форма (мультипликативная некорректна).

    user-скоры 0.001/0.0015/0.002; range = 0.001, порог = 0.5×0.001 = 0.0005.
    """
    results: R = {
        "a": (-0.001, {}),
        "b": (-0.0015, {}),
        "c": (-0.002, {}),
    }
    pool = select_finalists(
        results,
        metric_user="mae",
        corridor_delta=0.5,
        corridor_mode="additive",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_disqualified_leader_excluded():
    """Дисквалифицированный алгоритм исключён из пула (даже лидер)."""
    results: R = {
        "svr": (0.95, {}),
        "ridge": (0.90, {}),
        "random_forest": (0.88, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
        disqualified={"svr"},
    )
    assert pool == ["ridge", "random_forest"]


def test_boundary_tie_breaks_by_configuration_order():
    """Равные скоры → порядок конфигурации (как в select_winner)."""
    results: R = {
        "random_forest": (0.5, {}),
        "extra_trees": (0.5, {}),
        "elasticnet": (0.5, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["random_forest", "extra_trees", "elasticnet"]


def test_boundary_equal_scores_on_corridor_edge_all_included():
    """Скоры ровно на границе коридора включаются все (tie-break включения)."""
    results: R = {
        "ridge": (-0.05, {}),
        "svr": (-0.06, {}),
        "random_forest": (-0.06, {}),
    }
    # Порог = 0.05 × 1.2 = 0.06: оба кандидата ровно на границе.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.2, top_k_candidates=3
    )
    assert pool == ["ridge", "svr", "random_forest"]


# ──────────────────────────────────────────────────────────────────────────────
# Семейства алгоритмов: опциональный мягкий добор
# ──────────────────────────────────────────────────────────────────────────────
def test_family_diversity_topup_from_extended_corridor():
    """Добор лучшего представителя недостающего семейства из расширенного коридора.

    Базовый пул: [elasticnet, ridge] (линейные). SVR (kernel, 0.095) вне
    базового коридора (0.095 > 0.09315), но внутри расширенного
    (0.081 × 1.225 = 0.0992) → добавляется, пул достигает top_k=3.
    """
    results: R = {
        "elasticnet": (-0.081, {}),
        "ridge": (-0.082, {}),
        "svr": (-0.095, {}),
        "random_forest": (-0.11, {}),  # вне даже расширенного коридора
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge", "svr"]


def test_family_diversity_respects_top_k_cap():
    """Добор семейств не превышает top_k_candidates."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "ridge": (-0.082, {}),
        "svr": (-0.095, {}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=2,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge"]


def test_family_diversity_allowed_families_restriction():
    """allowed_families ограничивает, какие семейства можно добрать."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "ridge": (-0.082, {}),
        "svr": (-0.095, {}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"ensemble"},  # kernel запрещён → добора нет
    )
    assert pool == ["elasticnet", "ridge"]


def test_family_diversity_no_topup_when_two_families_in_pool():
    """В пуле уже два семейства — правило добора не применяется."""
    results: R = {
        "elasticnet": (-0.081, {}),  # linear
        "svr": (-0.085, {}),  # kernel
        "ridge": (-0.095, {}),  # linear, вне базового коридора
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "svr"]


def test_family_diversity_unmapped_algorithms_own_family():
    """Алгоритмы без классификации — собственное семейство (детерминизм)."""
    results: R = {
        "elasticnet": (-0.081, {}),  # linear
        "ridge": (-0.082, {}),  # linear
        "custom_algo": (-0.095, {}),  # своё семейство
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families={"elasticnet": "linear", "ridge": "linear"},
    )
    assert pool == ["elasticnet", "ridge", "custom_algo"]


def test_family_diversity_without_families_map():
    """Без карты семейств каждый алгоритм — собственное семейство.

    Базовый пул [a] содержит одно «семейство»; добор добавляет лучшего
    кандидата из расширенного коридора (b, 0.095 ≤ 0.081×1.225 = 0.0992).
    """
    results: R = {
        "a": (-0.081, {}),
        "b": (-0.095, {}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=2,
        enforce_family_diversity=True,
        algorithm_families=None,
    )
    assert pool == ["a", "b"]


def test_family_diversity_default_multiplier_constant():
    """Дефолтный множитель расширенного коридора — 1.5 (из постановки)."""
    assert FAMILY_DIVERSITY_MULTIPLIER_DEFAULT == 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases и failure modes
# ──────────────────────────────────────────────────────────────────────────────
def test_edge_empty_results_raises_runtime_error():
    """Пустой phase_results → RuntimeError (по постановке T3).

    Примечание: ``select_winner`` для пустого словаря бросает ``ValueError``,
    но для пула финалистов постановка требует ``RuntimeError`` (ревью PR #33).
    """
    with pytest.raises(RuntimeError, match="empty results"):
        select_finalists({}, metric_user="r2")


def test_edge_all_disqualified_raises_runtime_error():
    """Все алгоритмы дисквалифицированы → RuntimeError."""
    results: R = {"svr": (0.95, {}), "ridge": (0.90, {})}
    with pytest.raises(RuntimeError, match="disqualified"):
        select_finalists(
            results,
            metric_user="r2",
            disqualified={"svr", "ridge"},
        )


def test_edge_invalid_top_k_candidates():
    with pytest.raises(ValueError, match="top_k_candidates"):
        select_finalists({"a": (0.5, {})}, metric_user="r2", top_k_candidates=0)


def test_edge_invalid_corridor_delta():
    with pytest.raises(ValueError, match="corridor_delta"):
        select_finalists({"a": (0.5, {})}, metric_user="r2", corridor_delta=-0.1)


def test_edge_invalid_corridor_mode():
    with pytest.raises(ValueError, match="corridor_mode"):
        select_finalists(
            {"a": (0.5, {})},
            metric_user="r2",
            corridor_mode="weird",  # type: ignore[arg-type]
        )


def test_edge_invalid_family_diversity_multiplier():
    with pytest.raises(ValueError, match="family_diversity_multiplier"):
        select_finalists(
            {"a": (0.5, {})}, metric_user="r2", family_diversity_multiplier=0.0
        )


def test_edge_unknown_metric_raises():
    with pytest.raises(ValueError, match="not implemented"):
        select_finalists({"a": (0.5, {})}, metric_user="unknown_custom_metric")


def test_edge_determinism_repeated_calls():
    """Детерминизм: повторные вызовы дают идентичный пул и порядок."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "ridge": (-0.082, {}),
        "svr": (-0.095, {}),
        "random_forest": (-0.11, {}),
    }
    kwargs = {
        "metric_user": "rmse",
        "corridor_delta": 0.15,
        "top_k_candidates": 3,
        "enforce_family_diversity": True,
        "algorithm_families": FAMILIES,
    }
    first = select_finalists(results, **kwargs)
    for _ in range(5):
        assert select_finalists(results, **kwargs) == first


def test_edge_zero_delta_only_leader():
    """δ=0: коридор схлопывается до точного равенства — только лидер."""
    results: R = {
        "ridge": (0.90, {}),
        "svr": (0.90, {}),  # ровно равен лидеру
        "random_forest": (0.89, {}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        corridor_delta=0.0,
        top_k_candidates=3,
    )
    # Лидер + кандидат с равным скором (граница включается), 0.89 — вне.
    assert pool == ["ridge", "svr"]


def test_edge_min_better_and_max_better_metrics_parametrized():
    """Направление берётся из user_direction: min- и max-better метрики."""
    # min-better (rmse): лидер с минимальной ошибкой
    min_results: R = {"good": (-0.05, {}), "bad": (-0.10, {})}
    # Порог = 0.05 × 1.5 = 0.075: bad (0.10) вне коридора.
    assert select_finalists(
        min_results, metric_user="rmse", corridor_delta=0.5, top_k_candidates=3
    ) == ["good"]
    # max-better (r2): лидер с максимальным скором
    max_results: R = {"good": (0.90, {}), "bad": (0.80, {})}
    # Порог = 0.90 × 0.5 = 0.45: bad (0.80) внутри, порог широкий.
    assert select_finalists(
        max_results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    ) == ["good", "bad"]


# ──────────────────────────────────────────────────────────────────────────────
# Регрессии ревью PR #33
# ──────────────────────────────────────────────────────────────────────────────
def test_regression_disqualified_generator_not_consumed():
    """Генератор в ``disqualified`` не «съедается» dict-comprehension.

    Блокирующее замечание ревью: раньше ``set(disqualified)`` вычислялся
    внутри comprehension, генератор потреблялся на первой итерации, и
    дисквалифицированные имена «протекали» в пул (воспроизведение: ['c']
    вместо ['a', 'b']). Теперь iterable материализуется до comprehension.
    """
    results: R = {
        "a": (0.90, {}),
        "b": (0.85, {}),
        "c": (0.95, {}),  # лучший по CV, но дисквалифицирован
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
        disqualified=(name for name in ("c",)),
    )
    assert pool == ["a", "b"]


def test_regression_pool_family_not_allowed_no_topup():
    """Единственное семейство пула не разрешено → добор не выполняется.

    Блокирующее замечание ревью: поведение приведено к документации — если
    единственное семейство пула отсутствует в ``allowed_families``, мягкий
    добор не применяется (раньше было ['elasticnet', 'svr'] вместо
    ['elasticnet']).
    """
    results: R = {
        "elasticnet": (-0.081, {}),  # linear
        "svr": (-0.095, {}),  # kernel, в расширенном коридоре
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"ensemble"},  # линейное семейство пула запрещено
    )
    assert pool == ["elasticnet"]


def test_regression_pool_family_allowed_and_missing_allowed():
    """Семейство пула и добавляемое семейство разрешены → добор выполняется.

    Позитивный контроль к ``test_regression_pool_family_not_allowed_no_topup``:
    добор происходит, когда единственное семейство пула разрешено
    ``allowed_families`` и недостающее семейство тоже разрешено.
    """
    results: R = {
        "elasticnet": (-0.081, {}),  # linear
        "ridge": (-0.082, {}),  # linear
        "svr": (-0.095, {}),  # kernel, в расширенном коридоре
        "random_forest": (-0.11, {}),  # ensemble, вне расширенного коридора
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"linear", "kernel"},
    )
    assert pool == ["elasticnet", "ridge", "svr"]


def test_regression_auto_mode_best_near_zero_switches_to_additive():
    """auto при best≈0 (MAE≈0) переключается на additive (ревью PR #33).

    Мультипликативная форма при best=0 даёт порог 0 и схлопывает пул до
    лидера; аддитивная включает кандидатов в пределах δ×размах.
    """
    results: R = {
        "a": (0.0, {}),  # MAE ≈ 0 (raw -0.0 после инверсии)
        "b": (-1e-4, {}),
        "c": (-2e-4, {}),
    }
    # additive: range = 2e-4, порог = 0.5×2e-4 = 1e-4 → b на границе включается.
    pool = select_finalists(
        results, metric_user="mae", corridor_delta=0.5, top_k_candidates=3
    )
    assert pool == ["a", "b"]