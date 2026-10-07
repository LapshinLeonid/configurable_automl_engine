"""
Юнит-тесты чистой логики выбора победителя и инварианта winner-скора.

Покрывают функции ``select_winner`` и ``is_valid_winner_score`` из
``training_engine/component.py`` (issue #32):

• select_winner — детерминированный tie-break при равных скорах
  (побеждает первый в порядке конфигурации);
• is_valid_winner_score — порог класса worst-score сентинела
  (score <= float32 minimum → невалиден), включая «сырой» float32 minimum,
  который старый фильтр через ``math.isclose(rel_tol=1e-9)`` пропускал.

Интеграция двухстадийного выбора победителя (эпик #61, задача T7, issue #69):

• пул финалистов (``select_finalists``, T3) + ``select_robust_winner`` (T4):
  три режима ``off`` / ``warn_only`` / ``active`` и fallback при тотальном
  провале пула;
• кейс из постановки: ElasticNet CV=0.081, SVR=0.085, RF=0.089, δ=0.15 →
  пул из трёх; победитель по RMSE_oof — SVR или RF, но не ElasticNet;
• «золотые» сценарии на синтетических данных N=10–30: победитель —
  честная модель, вырожденные (a)–(d) дисквалифицируются при ``mode=active``.

Все тесты используют фиксированные seed (стандарт репозитория).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.component import (
    FinalistCandidate,
    is_valid_winner_score,
    select_finalists,
    select_robust_winner,
    select_winner,
)
from configurable_automl_engine.training_engine.metrics import oof_rmse
from configurable_automl_engine.training_engine.sanity_gate import ModelSanityGate
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# select_winner: детерминированный выбор победителя
# ──────────────────────────────────────────────────────────────────────────────
def test_select_winner_single_algorithm():
    """Единственный алгоритм — он и есть победитель (возвращается имя-строка)."""
    results = {"ridge": (0.7, {"alpha": 1.0})}
    winner = select_winner(results)
    assert isinstance(winner, str)
    assert winner == "ridge"


def test_select_winner_max_score_wins():
    """Побеждает алгоритм с максимальным «сырым» скором."""
    results = {
        "ridge": (0.4, {"alpha": 1.0}),
        "random_forest": (0.9, {"n_estimators": 10}),
        "extra_trees": (0.6, {"n_estimators": 20}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_breaks_by_configuration_order():
    """B6 (issue #32): равные скоры → первый в порядке конфигурации.

    ``max`` стабилен, а dict сохраняет порядок вставки, поэтому ничья
    разрешается порядком итерации (порядок конфигурации для
    последовательного пути).
    """
    results = {
        "random_forest": (0.5, {"n_estimators": 10}),
        "extra_trees": (0.5, {"n_estimators": 20}),
        "elasticnet": (0.5, {"alpha": 0.5}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_reorders_with_negative_scores():
    """Ничья корректна и для отрицательных (инвертированных) скоров."""
    results = {
        "extra_trees": (-0.5, {}),
        "random_forest": (-0.5, {}),
    }
    assert select_winner(results) == "extra_trees"


def test_select_winner_empty_raises_value_error():
    """Пустой словарь результатов → ValueError (победителя нет)."""
    with pytest.raises(ValueError, match="empty results"):
        select_winner({})


# ──────────────────────────────────────────────────────────────────────────────
# is_valid_winner_score: инвариант winner-скора
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "score",
    [
        0.42,
        1.0,
        -0.9,
        -3.4e38,  # очень маленький, но валидный скор (выше float32 min)
        -3.39e38,
        0.0,
        5,  # целочисленный скор
        np.float32(0.5),
        np.float64(-0.75),
    ],
)
def test_is_valid_winner_score_accepts_finite_above_threshold(score: Any):
    """Валидные скоры: числа, конечные и строго выше float32-min порога."""
    assert is_valid_winner_score(score) is True


@pytest.mark.parametrize(
    "score",
    [
        None,
        "not-a-number",
        "0.5",  # числовая строка — всё равно мусор (issue #32, ревью PR #22)
        b"0.5",
        (0.5,),
        [0.5],
        {"value": 0.5},
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32("nan"),
        np.float32("inf"),
        np.float32("-inf"),
        HPO_WORST_SCORE,
        float(np.finfo(np.float32).min),  # B4: «сырой» float32 min
        np.float32(np.finfo(np.float32).min),
        -1e39,  # ниже float32 min — класс сентинела
        float("-inf"),
    ],
)
def test_is_valid_winner_score_rejects_sentinel_and_nonfinite(score: Any):
    """Невалидные скоры: не-числа, мусор, NaN/±inf и класс worst-score сентинела."""
    assert is_valid_winner_score(score) is False


def test_threshold_covers_hpo_worst_score_and_raw_f32_min():
    """Порог класса сентинела покрывает обе известные сигнатуры.

    HPO_WORST_SCORE (-3.4028235e38) и «сырой» float(np.finfo(np.float32).min)
    отличаются на ~9.88e-9 относительно — старый isclose(rel_tol=1e-9) второй
    не ловил. Оба обязаны попадать в класс сентинела (issue #32).
    """
    raw_f32_min = float(np.finfo(np.float32).min)
    assert HPO_WORST_SCORE != raw_f32_min  # разные double-значения
    assert not math.isclose(raw_f32_min, HPO_WORST_SCORE, rel_tol=1e-9)
    assert raw_f32_min <= WORST_SCORE_THRESHOLD
    assert HPO_WORST_SCORE <= WORST_SCORE_THRESHOLD
    # Валидный скор обязан быть строго выше порога
    assert -3.4e38 > WORST_SCORE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
#  Интеграция двухстадийного выбора победителя (T3 + T4, issue #69)
# ──────────────────────────────────────────────────────────────────────────────

R = dict[str, tuple[float, dict[str, Any]]]


def _golden_dataset(
    n: int = 30, seed: int = 7, noise: float = 0.5, p: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Синтетический датасет: X0/X1 информативны, остальное — шум.

    N=10–30 как в реальных кейсах постановки T7; фиксированный seed.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def _fit_oof_full(
    factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Честный OOF-вектор (модель фолда не видела строку) + full-fit.

    Возвращает (y_pred_oof, y_pred_full, full_model). Число фолдов 5 для
    маленьких датасетов N≈30 — каждая строка попадает в тест ровно один раз.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        model = factory()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    full_model = factory().fit(X, y)
    return oof, full_model.predict(X), full_model


def _candidate(
    name: str,
    cv_score: float,
    rmse_oof: float,
    y_oof: np.ndarray,
    y_full: np.ndarray,
    model: Any,
) -> FinalistCandidate:
    """Собрать ``FinalistCandidate`` из обученной модели и её векторов."""
    return FinalistCandidate(
        algo_name=name,
        model=model,
        params={},
        cv_score=cv_score,
        rmse_oof=rmse_oof,
        y_pred_oof=y_oof,
        y_pred_full=y_full,
    )


def _golden_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """Обучить «золотой» пул (a)–(e) на синтетических данных N=10–30.

    (a) ElasticNet с занулёнными коэффициентами («полка») → плоские OOF;
    (b) SVR-sigmoid, упёршийся в плато tanh → почти константные предсказания;
    (c) дерево глубины 20 (RMSE_train≈0) → контур Г;
    (d) 1-NN → контур Г;
    (e) честная модель (градиентный бустинг малой глубины) → проходит аудит.

    Порядок словаря — детерминированный порядок пула из T3 (по CV-скору
    убыванию), как его строит ``select_finalists``.
    """
    factories: dict[str, tuple[Callable[[], Any], float]] = {
        # CV-скор в «сырой» семантике оптимизатора (neg-RMSE): лучший — самый
        # близкий к нулю. Вырожденные модели «выигрывают» по CV (постановка).
        "elasticnet_zero": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr_sigmoid": (
            lambda: SVR(kernel="sigmoid", gamma=0.001, coef0=0.0),
            -0.085,
        ),
        "deep_tree": (lambda: DecisionTreeRegressor(max_depth=20, random_state=0), -0.089),
        "knn_1": (lambda: KNeighborsRegressor(n_neighbors=1), -0.093),
        "honest_gbr": (
            lambda: GradientBoostingRegressor(
                max_depth=1, n_estimators=20, learning_rate=0.05, random_state=0
            ),
            -0.11,
        ),
    }
    candidates: dict[str, FinalistCandidate] = {}
    for name, (factory, cv_score) in factories.items():
        oof, full, model = _fit_oof_full(factory, X, y)
        candidates[name] = _candidate(
            name,
            cv_score=cv_score,
            rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof,
            y_full=full,
            model=model,
        )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
#  Три режима off / warn_only / active (интеграция пула T3 + robust T4)
# ──────────────────────────────────────────────────────────────────────────────


def test_integration_three_modes_pool_to_robust_winner():
    """Пул финалистов (T3) → select_robust_winner (T4): семантика трёх режимов.

    CV-лидер — вырожденная модель (плоский ElasticNet): в ``off``/``warn_only``
    он побеждает как раньше, в ``active`` — дисквалифицируется, и побеждает
    честная модель с лучшим RMSE_oof.
    """
    X, y = _golden_dataset(n=80, seed=11)
    results: R = {
        "elasticnet_zero": (-0.081, {"alpha": 500.0}),
        "svr": (-0.085, {"C": 1.0}),
        "random_forest": (-0.089, {"n_estimators": 50}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet_zero", "svr", "random_forest"]

    # Обучаем финалистов в порядке пула (детерминированный tie-break T3).
    factories = {
        "elasticnet_zero": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr": (lambda: SVR(kernel="rbf", C=1.0), -0.085),
        "random_forest": (
            lambda: RandomForestRegressor(max_depth=3, n_estimators=50, random_state=0),
            -0.089,
        ),
    }
    cands: dict[str, FinalistCandidate] = {}
    for name in pool:
        factory, cv_score = factories[name]
        oof, full, model = _fit_oof_full(factory, X, y)
        cands[name] = _candidate(
            name, cv_score=cv_score, rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof, y_full=full, model=model,
        )

    gate = ModelSanityGate()
    X_df, y_s = pd.DataFrame(X), pd.Series(y)

    off = select_robust_winner(cands, X_df, y_s, gate, mode="off")
    assert off.winner_algo == "elasticnet_zero"  # старое поведение: лучший CV
    assert off.audit == {}
    assert off.disqualified == {}

    warn = select_robust_winner(cands, X_df, y_s, gate, mode="warn_only")
    # Победитель как при off, дисквалификации гипотетические (T6).
    assert warn.winner_algo == "elasticnet_zero"
    assert "elasticnet_zero" in warn.disqualified
    assert set(warn.audit) == set(pool)
    # Гипотетический победитель активного режима — честная модель (SVR/RF).
    assert warn.winner_if_active in {"svr", "random_forest"}
    assert warn.winner_if_active != "elasticnet_zero"

    active = select_robust_winner(cands, X_df, y_s, gate, mode="active")
    assert active.winner_algo in {"svr", "random_forest"}
    assert active.winner_algo != "elasticnet_zero"
    assert active.winner_rmse_oof == pytest.approx(cands[active.winner_algo].rmse_oof)
    assert "elasticnet_zero" in active.disqualified
    assert active.fallback_used is False


def test_boundary_warn_only_winner_matches_off_and_reports_disqualifications():
    """Boundary: mode=warn_only — победитель ровно как при mode=off.

    Дисквалификации фиксируются для отчёта (T6), но выбор не меняются.
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    gate = ModelSanityGate()
    X_df, y_s = pd.DataFrame(X), pd.Series(y)

    off = select_robust_winner(cands, X_df, y_s, gate, mode="off")
    warn = select_robust_winner(cands, X_df, y_s, gate, mode="warn_only")
    assert warn.winner_algo == off.winner_algo
    # Гипотетические дисквалификации собраны для отчёта.
    assert set(warn.disqualified) == {"elasticnet_zero", "svr_sigmoid", "deep_tree", "knn_1"}
    assert warn.winner_if_active == "honest_gbr"


def test_boundary_pool_exactly_top_k_candidates():
    """Boundary: пул ровно из top_k кандидатов (все внутри коридора δ)."""
    results: R = {
        "elasticnet": (-0.081, {}),
        "svr": (-0.085, {}),
        "random_forest": (-0.089, {}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]
    assert len(pool) == 3


# ──────────────────────────────────────────────────────────────────────────────
#  Кейс из постановки (T7, п. 6): ElasticNet 0.081 / SVR 0.085 / RF 0.089, δ=0.15
# ──────────────────────────────────────────────────────────────────────────────


def test_issue_case_elasticnet_svr_rf_delta_015_winner_not_elasticnet():
    """ElasticNet CV=0.081, SVR=0.085, RF=0.089, δ=0.15 → пул из трёх.

    ElasticNet с занулёнными коэффициентами дисквалифицируется; победитель
    по RMSE_oof — SVR или RF (но не ElasticNet).
    """
    X, y = _golden_dataset(n=80, seed=11, noise=0.5, p=5)
    results: R = {
        "elasticnet": (-0.081, {"alpha": 500.0}),
        "svr": (-0.085, {"C": 1.0}),
        "random_forest": (-0.089, {"n_estimators": 50}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]  # пул из трёх

    factories = {
        "elasticnet": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr": (lambda: SVR(kernel="rbf", C=1.0), -0.085),
        "random_forest": (
            lambda: RandomForestRegressor(max_depth=3, n_estimators=50, random_state=0),
            -0.089,
        ),
    }
    cands: dict[str, FinalistCandidate] = {}
    for name in pool:
        factory, cv_score = factories[name]
        oof, full, model = _fit_oof_full(factory, X, y)
        cands[name] = _candidate(
            name, cv_score=cv_score, rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof, y_full=full, model=model,
        )

    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
    )
    assert res.winner_algo in {"svr", "random_forest"}
    assert res.winner_algo != "elasticnet"
    assert "elasticnet" in res.disqualified
    assert res.fallback_used is False


# ──────────────────────────────────────────────────────────────────────────────
#  Золотые сценарии (N=10–30): (a)–(d) дисквалифицируются, победитель — (e)
# ──────────────────────────────────────────────────────────────────────────────


def test_golden_small_n_winner_is_honest_model_mode_active():
    """Golden (T7, п. 5): при mode=active победитель — честная модель (e).

    Вырожденные (a) «полка» ElasticNet, (b) SVR-sigmoid плато tanh,
    (c) дерево глубины 20, (d) 1-NN — дисквалифицируются; честный
    градиентный бустинг (e) проходит аудит и побеждает по RMSE_oof.
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
    )
    assert res.winner_algo == "honest_gbr"
    assert res.winner_rmse_oof == pytest.approx(cands["honest_gbr"].rmse_oof)
    assert set(res.disqualified) == {
        "elasticnet_zero",
        "svr_sigmoid",
        "deep_tree",
        "knn_1",
    }
    assert res.audit["honest_gbr"].is_valid is True
    assert res.fallback_used is False


def test_golden_small_n_mode_off_old_behavior():
    """Golden (T7, п. 5): mode=off — старое поведение select_winner.

    Победитель — лучший CV-скор (вырожденная модель), аудит не выполняется.
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="off"
    )
    assert res.winner_algo == "elasticnet_zero"  # лучший «сырой» CV-скор
    assert res.audit == {}
    assert res.disqualified == {}
    assert res.fallback_used is False


def test_golden_small_n_determinism_repeated_runs():
    """Golden (T7, п. 5): воспроизводимость — повторные запуски идентичны."""
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    first = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
    )
    for _ in range(3):
        again = select_robust_winner(
            cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
        )
        assert again.winner_algo == first.winner_algo
        assert again.disqualified == first.disqualified
        assert again.pool == first.pool
        assert again.fallback_used == first.fallback_used


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback: все кандидаты забракованы
# ──────────────────────────────────────────────────────────────────────────────


def test_boundary_all_candidates_rejected_fallback():
    """Boundary: все кандидаты пула забракованы → fallback-критерий (T4).

    Формальный критерий: минимум числа нарушенных контуров → минимум
    тяжести (Г > А > Б > В) → лучший RMSE_oof. Запуск не падает.
    """
    from unittest.mock import MagicMock

    from configurable_automl_engine.training_engine.sanity_gate import SanityCheckResult

    rng = np.random.RandomState(0)
    n = 60
    y = rng.randn(n) * 5.0
    y_oof = y + rng.randn(n) * 0.5
    x_df = pd.DataFrame(np.zeros((n, 2)))

    def _cand(name: str, cv: float, rmse: float) -> FinalistCandidate:
        return FinalistCandidate(
            algo_name=name,
            model=object(),
            params={},
            cv_score=cv,
            rmse_oof=rmse,
            y_pred_oof=y_oof,
            y_pred_full=y_oof,
        )

    cands = {
        "one_violation": _cand("one_violation", -0.08, 0.3),
        "two_violations": _cand("two_violations", -0.09, 0.2),
    }
    gate = MagicMock()
    gate.check.side_effect = [
        SanityCheckResult(
            is_valid=False, reasons=["Circuit D (generalization gap): gap=2.0 > 1.5"]
        ),
        SanityCheckResult(
            is_valid=False,
            reasons=[
                "Circuit A (diversity): ratio=0.05 < 0.15",
                "Circuit B (unique): nunique=2 < 5",
            ],
        ),
    ]
    res = select_robust_winner(cands, x_df, pd.Series(y), gate, mode="active")
    assert res.fallback_used is True
    # Меньше нарушенных контуров → «наименее проблемная» модель.
    assert res.winner_algo == "one_violation"
    assert res.winner_rmse_oof == pytest.approx(0.3)