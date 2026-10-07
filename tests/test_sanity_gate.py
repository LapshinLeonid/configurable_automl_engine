"""Тесты ModelSanityGate — модуля аудита вырождения моделей (эпик #61, T2).

Покрытие:
    1. Контур А (Diversity Ratio): честная модель проходит; константный прогноз
       проваливается; граница «равенство порогу» проходит; константный y —
       пропуск с soft_warning; адаптивный режим относительно константной модели.
    2. Контур Б (Unique Ratio): масштаб округления от std(y); нижняя граница
       nunique; граница «nunique == min_unique_count» проходит.
    3. Контур В (Dead Feature Check): Lasso с ~60% нулей — сигнал, не
       дисквалификация; комбинация «много нулей + низкий diversity»
       дисквалифицирует; пермутационный путь для нелинейных моделей с
       контролем стоимости и детерминизмом; исключения predict → reason.
    4. Контур Г (Generalization Gap): переобученные модели (дерево глубины 20,
       1-NN) проваливаются; RMSE_oof≈0 / RMSE_full≈0 — деление на ноль;
       граница gap == max проходит; пример из issue (0.1 vs 0.19).
    5. Режимы и edge cases: warn_only; отключение контуров; NaN-маскирование;
       severity; валидация параметров; различающиеся reasons/soft_warnings.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.sanity_gate import (
    ModelSanityGate,
    SanityCheckResult,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Хелперы
# ──────────────────────────────────────────────────────────────────────────────


def _dataset(seed: int = 7, n: int = 150, p: int = 6, noise: float = 1.0):
    """Чистый датасет: X0/X1 информативны, остальное — шум."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def _honest_oof_full(
    factory: Callable[[], Any], X: np.ndarray, y: np.ndarray, folds: int = 4
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Честный OOF-вектор (модель фолда не видела строку) + full-fit.

    Возвращает (y_pred_oof, y_pred_full, full_model).
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        model = factory()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    full_model = factory().fit(X, y)
    return oof, full_model.predict(X), full_model


def _with_rmse(base: np.ndarray, target_rmse: float, rng: np.random.RandomState) -> np.ndarray:
    """Вектор с заданным RMSE относительно ``base`` (см. issue, контур Г)."""
    noise = rng.randn(len(base))
    noise = noise / np.sqrt(np.mean(noise**2)) * target_rmse
    return base + noise


class _CoefModel:
    """Суб-модель только с ``coef_`` — для линейного пути контура В."""

    def __init__(self, coef: Any) -> None:
        self.coef_ = np.asarray(coef, dtype=float)


class _BrokenPredictModel:
    """Модель, падающая в ``predict`` — проверка перехвата исключений."""

    def predict(self, X: Any) -> Any:
        raise RuntimeError("boom")


# ──────────────────────────────────────────────────────────────────────────────
#  Структура результата
# ──────────────────────────────────────────────────────────────────────────────


def test_sanity_check_result_defaults_and_fields():
    result = SanityCheckResult(is_valid=True)
    assert result.is_valid is True
    assert result.reasons == []
    assert result.soft_warnings == []
    assert np.isnan(result.diversity_ratio)
    assert np.isnan(result.unique_ratio)
    assert result.dead_features_count == 0
    assert np.isnan(result.generalization_gap)
    assert result.severity == 0


def test_reasons_and_soft_warnings_are_distinct_channels():
    gate = ModelSanityGate(check_dead_features=False)
    y = np.arange(50, dtype=float)
    result = gate.check(y=y, y_pred_oof=np.full(50, 1.0), y_pred_full=y)
    assert any("Circuit A" in r for r in result.reasons)
    assert any("Circuit B" in r for r in result.reasons)
    # Сигналы и дисквалификации не смешиваются.
    assert not any("signal" in r for r in result.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Контур А — Diversity Ratio
# ──────────────────────────────────────────────────────────────────────────────


def test_diversity_honest_model_passes_circuit_a():
    X, y = _dataset()
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=10.0), X, y)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.diversity_ratio > 0.15
    assert result.reasons == []


def test_diversity_constant_predictions_fail():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert result.diversity_ratio == 0.0


def test_diversity_boundary_equality_passes():
    # y = arange → дисперсия точная; pred = c*y → ratio = c² ровно.
    y = np.arange(200, dtype=float)
    c = np.sqrt(0.15)
    gate = ModelSanityGate(
        min_prediction_diversity=0.15,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=c * y)
    assert result.is_valid
    assert result.diversity_ratio == pytest.approx(0.15, abs=1e-9)

    below = ModelSanityGate(
        min_prediction_diversity=0.15,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    res_below = below.check(y=y, y_pred_oof=np.sqrt(0.149) * y)
    assert not res_below.is_valid
    assert any("Circuit A" in r for r in res_below.reasons)


def test_diversity_constant_target_skips_with_warning():
    y = np.full(50, 5.0)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(50, 5.0))
    assert result.is_valid  # контур А пропущен, оценивает контур Б
    assert any("target variance is zero" in w for w in result.soft_warnings)
    assert not any("Circuit A" in r for r in result.reasons)
    # Отношение не определено (деление на ноль) — NaN, а не 0.0 (ревью PR #32).
    assert np.isnan(result.diversity_ratio)


def test_diversity_adaptive_mode_relative_to_constant_model():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        adaptive_diversity=True,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    # Любой положительный разброс лучше константной модели.
    result = gate.check(y=y, y_pred_oof=0.01 * y)
    assert result.is_valid
    assert result.diversity_ratio == pytest.approx(0.0001, rel=1e-6)
    # Нулевой разброс — ровно как у константной модели → провал.
    const = gate.check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert not const.is_valid
    assert any("Circuit A" in r for r in const.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Контур Б — Unique Ratio
# ──────────────────────────────────────────────────────────────────────────────


def test_unique_honest_predictions_pass():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.linspace(0, 100, 100))
    assert result.is_valid
    assert result.unique_ratio == pytest.approx(1.0)
    assert result.reasons == []


def test_unique_constant_predictions_fail():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(100, 42.0))
    assert not result.is_valid
    assert any("Circuit B" in r for r in result.reasons)
    assert result.unique_ratio == pytest.approx(1 / 100)


def test_unique_boundary_equal_to_min_unique_count_passes():
    y = np.arange(100, dtype=float)
    pred = np.tile(np.arange(5, dtype=float), 20)  # ровно 5 уникальных
    gate = ModelSanityGate(
        min_unique_count=5,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=pred)
    assert result.is_valid

    gate_strict = ModelSanityGate(
        min_unique_count=6,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    res_strict = gate_strict.check(y=y, y_pred_oof=pred)
    assert not res_strict.is_valid


def test_unique_ratio_rounding_scaled_by_target_std():
    # Крупномасштабный y: одинаковый «джиттер» 0.001 в предсказаниях
    # неразличим после округления (granularity = 10) → провал.
    y_big = np.arange(100, dtype=float) * 100.0  # std ≈ 2887 → decimals ≤ 0
    pred_big = 50000.0 + np.linspace(0.0, 0.099, 100)  # шаг 0.001
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    res_big = gate.check(y=y_big, y_pred_oof=pred_big)
    assert not res_big.is_valid
    assert any("Circuit B" in r for r in res_big.reasons)

    # Мелкомасштабный y (std ≈ 0.29, granularity = 0.001): тот же джиттер
    # значим → все 100 значений различны → проход.
    y_small = np.arange(100, dtype=float) * 0.01
    pred_small = 5.0 + np.linspace(0.0, 0.099, 100)  # шаг 0.001
    res_small = gate.check(y=y_small, y_pred_oof=pred_small)
    assert res_small.is_valid
    assert res_small.unique_ratio == pytest.approx(1.0)


def test_unique_min_unique_ratio_lower_bound_applies():
    # N=1000: required = max(5, 0.05*1000) = 50 → 20 уникальных — провал.
    y = np.arange(1000, dtype=float)
    pred = np.tile(np.arange(20, dtype=float), 50)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=pred)
    assert not result.is_valid
    assert "min_unique_ratio" in result.reasons[0]


# ──────────────────────────────────────────────────────────────────────────────
#  Контур В — Dead Feature Check
# ──────────────────────────────────────────────────────────────────────────────


def test_dead_features_lasso_many_zeros_is_signal_not_disqualification():
    rng = np.random.RandomState(11)
    X = rng.randn(200, 30)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(200)
    lasso = Lasso(alpha=0.06, random_state=0).fit(X, y)
    gate = ModelSanityGate()
    # Порог нулевых коэффициентов берём из самого гейта, чтобы тест не
    # рассинхронизировался с реализацией (ревью PR #32).
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(lasso.coef_) <= tol).mean())
    assert zero_ratio > 0.5  # ~57–60% нулевых коэффициентов

    result = gate.check(
        y=y, y_pred_oof=lasso.predict(X), y_pred_full=lasso.predict(X), X=X, model=lasso
    )
    assert result.is_valid
    assert result.reasons == []
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert result.dead_features_count == int((np.abs(lasso.coef_) <= tol).sum())


def test_dead_features_linear_combo_with_low_diversity_disqualifies():
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))  # 100% нулей
    gate = ModelSanityGate(check_unique=False, check_generalization_gap=False)
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),  # diversity = 0 → контур А провален
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    # Комбинация «много нулей И низкий diversity»: контур В дисквалифицирует
    # только потому, что контур А уже провалился (diversity_failed=True).
    assert not result.is_valid
    assert any("combination" in r and "Circuit C" in r for r in result.reasons)
    assert any("Circuit A" in r for r in result.reasons)
    # Сигнал тоже присутствует.
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_combo_disabled_when_require_low_diversity_false():
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(
        dead_features_require_low_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    # Дисквалификация только по контуру А; контур В — сигнал.
    assert not result.is_valid
    assert all("Circuit C" not in r for r in result.reasons)
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_combo_inactive_reported_when_check_diversity_off():
    # check_diversity=False: комбинация C+A не вычислима — гейт явно
    # сообщает об этом soft_warning'ом, а не молча отключает правило.
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    assert result.is_valid  # без контура А нет дисквалификации
    assert any("combination rule inactive" in w for w in result.soft_warnings)


def test_dead_features_zero_ratio_boundary_equality_no_warning():
    model = _CoefModel([1.0, 0.0, 0.0, 0.0])  # ровно 0.75 нулей
    y = np.arange(40, dtype=float)
    gate = ModelSanityGate(
        max_dead_feature_ratio=0.75,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=np.arange(40, dtype=float), model=model)
    assert result.is_valid
    assert result.soft_warnings == []  # равенство порогу — не превышение


def test_dead_features_nonlinear_permutation_positive_rf():
    rng = np.random.RandomState(3)
    n, p = 150, 6
    X = rng.randn(n, p)
    y = X.sum(axis=1) + rng.randn(n)  # все признаки информативны
    rf = RandomForestRegressor(max_depth=4, n_estimators=60, random_state=0)
    oof, full, model = _honest_oof_full(
        lambda: RandomForestRegressor(max_depth=4, n_estimators=60, random_state=0),
        X,
        y,
    )
    result = ModelSanityGate(check_diversity=False, check_generalization_gap=False).check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert result.is_valid
    assert result.dead_features_count == 0
    assert result.soft_warnings == []


def test_dead_features_nonlinear_correlated_noise_soft_warning_only():
    # «Ложно мёртвые» признаки (коррелирующие колонки): модель использует
    # только первую колонку, остальные — почти точные копии (информация
    # берётся у соседа). Пермутация копий не меняет предсказания → высокая
    # доля «мёртвых» признаков → soft_warning, а НЕ дисквалификация.
    rng = np.random.RandomState(3)
    n, p = 120, 10
    x0 = rng.randn(n)
    X = np.column_stack([x0] + [x0 + rng.randn(n) * 0.05 for _ in range(p - 1)])
    y = 3.0 * x0 + rng.randn(n)

    class _FirstColumnModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.asarray(X)[:, 0] * 3.0

    model = _FirstColumnModel()
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result.is_valid  # не дисквалификация
    assert result.dead_features_count == p - 1
    assert any("dead-feature ratio" in w for w in result.soft_warnings)


def test_dead_features_permutation_disabled_when_sensitivity_off():
    # T5: check_permutation_sensitivity=False отключает пермутационный путь
    # контура В для нелинейных моделей; линейный путь по coef_ остаётся.
    rng = np.random.RandomState(3)
    n, p = 120, 10
    x0 = rng.randn(n)
    X = np.column_stack([x0] + [x0 + rng.randn(n) * 0.05 for _ in range(p - 1)])
    y = 3.0 * x0 + rng.randn(n)

    class _FirstColumnModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.asarray(X)[:, 0] * 3.0

    model = _FirstColumnModel()
    # С выключенной чувствительностью пермутационный аудит не выполняется:
    # «мёртвые» признаки не считаются, мягких предупреждений нет.
    gate = ModelSanityGate(
        check_permutation_sensitivity=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result.is_valid
    assert result.dead_features_count == 0
    assert result.soft_warnings == []

    # Контроль: тот же инпут с включённой чувствительностью даёт «мёртвые»
    # признаки (пермутационный путь реально работает).
    gate_on = ModelSanityGate(check_generalization_gap=False)
    result_on = gate_on.check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result_on.dead_features_count == p - 1


def test_dead_features_permutation_disabled_keeps_linear_path():
    # T5: выключение пермутационной чувствительности НЕ влияет на линейный
    # путь контура В (модель с coef_): сигнал и комбинация работают как раньше.
    rng = np.random.RandomState(5)
    n, p = 200, 12
    X = rng.randn(n, p)
    y = X[:, 0] + rng.randn(n) * 0.01
    lasso = Lasso(alpha=1e3).fit(X, y)
    assert np.mean(np.abs(lasso.coef_) <= 1e-12) > 0.8  # много нулей
    gate = ModelSanityGate(
        check_permutation_sensitivity=False,
        max_dead_feature_ratio=0.4,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=lasso.predict(X),
        y_pred_full=lasso.predict(X),
        X=X,
        model=lasso,
    )
    assert result.is_valid  # сигнал, не дисквалификация
    assert result.dead_features_count > 0
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_permutation_cost_controls_rows_and_features():
    rng = np.random.RandomState(9)
    n, p = 120, 40
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    rf = RandomForestRegressor(max_depth=3, n_estimators=30, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        permutation_max_rows=40,
        permutation_max_features=5,
        permutation_repeats=2,
        permutation_seed=1,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    # Проверено не более 5 колонок.
    assert 0 <= result.dead_features_count <= 5


def test_dead_features_permutation_deterministic_with_seed():
    rng = np.random.RandomState(9)
    n, p = 120, 12
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    rf = RandomForestRegressor(max_depth=3, n_estimators=30, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        permutation_max_rows=60,
        permutation_max_features=8,
        permutation_repeats=3,
        permutation_seed=123,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    r1 = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    r2 = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    # NaN-поля не участвуют в сравнении (nan != nan), сравниваем суть.
    assert r1.is_valid == r2.is_valid
    assert r1.reasons == r2.reasons
    assert r1.soft_warnings == r2.soft_warnings
    assert r1.dead_features_count == r2.dead_features_count
    assert r1.severity == r2.severity


def test_dead_features_permutation_works_with_dataframe_x():
    rng = np.random.RandomState(4)
    n, p = 100, 6
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] * 2.0 + rng.randn(n)
    rf = RandomForestRegressor(max_depth=4, n_estimators=40, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    assert result.is_valid


def test_dead_features_permutation_works_with_sparse_x():
    # Sparse X — штатный выход препроцессинга репозитория (ревью PR #32:
    # HashingEncodingTransformer → csr_matrix). Пермутации не должны падать
    # на np.array(sparse) (0-D object-array).
    from scipy import sparse

    rng = np.random.RandomState(0)
    n, p = 120, 10
    X = sparse.csr_matrix(rng.randn(n, p))
    y = np.asarray(X[:, 0].toarray()).ravel() * 2.0 + rng.randn(n)
    rf = RandomForestRegressor(max_depth=4, n_estimators=40, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(
        y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf
    )
    assert result.is_valid
    assert 0 <= result.dead_features_count <= p


def test_dead_features_exception_in_predict_becomes_reason():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = rng.randn(60)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_BrokenPredictModel(), X=X)
    assert not result.is_valid
    assert any("prediction failed" in r for r in result.reasons)


def test_dead_features_nonlinear_without_x_is_reason():
    rng = np.random.RandomState(0)
    y = rng.randn(60)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=None)
    assert not result.is_valid
    assert any("X is not provided" in r for r in result.reasons)


def test_dead_features_model_not_provided_is_reason():
    y = np.arange(50, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y)
    assert not result.is_valid
    assert any("model is not provided" in r for r in result.reasons)


def test_dead_features_pipeline_unwrapped_to_linear_estimator():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(6)
    n, p = 100, 8
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=0.3, random_state=0)).fit(X, y)
    gate = ModelSanityGate(check_diversity=False, check_generalization_gap=False)
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(pipe.named_steps["lasso"].coef_) <= tol).mean())
    result = gate.check(
        y=y, y_pred_oof=pipe.predict(X), y_pred_full=pipe.predict(X), X=X, model=pipe
    )
    # Пайплайн распакован до финального оценщика: считаем нули его coef_.
    assert result.dead_features_count == int(zero_ratio * p)
    # Доля нулей заметная → сигнал; контур А отключён, поэтому комбинация
    # C+A не применяется — причин Circuit C быть не должно.
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert not any("Circuit C" in r for r in result.reasons)
    assert "combination rule inactive" in " ".join(result.soft_warnings)


# ──────────────────────────────────────────────────────────────────────────────
#  Контур Г — Generalization Gap
# ──────────────────────────────────────────────────────────────────────────────


def _gap_only_gate(max_gap: float) -> ModelSanityGate:
    return ModelSanityGate(
        max_generalization_gap=max_gap,
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=True,
    )


def test_gap_honest_model_passes():
    X, y = _dataset(n=500)
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=3.0), X, y)
    result = ModelSanityGate(
        check_diversity=False, check_unique=False, check_dead_features=False
    ).check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert 1.0 <= result.generalization_gap <= 1.5


def test_gap_issue_example_0_1_vs_0_19_fails_at_1_5_passes_at_2_0():
    # Пример из issue: RMSE_full=0.1, RMSE_oof=0.19 — явное переобучение.
    rng = np.random.RandomState(1)
    y = rng.randn(500) * 10.0
    full = _with_rmse(y, 0.1, rng)
    oof = _with_rmse(y, 0.19, rng)
    strict = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not strict.is_valid
    assert any("Circuit D" in r for r in strict.reasons)
    assert strict.generalization_gap == pytest.approx(1.9, rel=1e-6)

    soft = _gap_only_gate(2.0).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert soft.is_valid


def test_gap_deep_tree_rmse_full_zero_fails():
    rng = np.random.RandomState(2)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    full = tree.predict(X)
    # Предпосылка сценария: дерево глубины 20 идеально обучается на train
    # (RMSE_full ≈ 0 в пределах точности float).
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not result.is_valid
    assert any("division by zero" in r or "RMSE_full" in r for r in result.reasons)
    assert np.isinf(result.generalization_gap)


def test_gap_one_nn_fails():
    rng = np.random.RandomState(2)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    full = knn.predict(X)
    # Предпосылка: 1-NN имеет нулевую ошибку на train (RMSE_full ≈ 0).
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not result.is_valid
    assert any("Circuit D" in r for r in result.reasons)


def test_gap_boundary_equality_passes():
    rng = np.random.RandomState(3)
    y = rng.randn(300) * 10.0
    full = _with_rmse(y, 1.0, rng)
    oof = _with_rmse(y, 1.0, rng)  # gap = 1.0 ровно
    result = _gap_only_gate(1.0).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert result.is_valid
    assert result.generalization_gap == pytest.approx(1.0, rel=1e-6)

    oof_over = _with_rmse(y, 2.0, rng)  # gap = 2.0 > 1.0 → провал
    over = _gap_only_gate(1.0).check(y=y, y_pred_oof=oof_over, y_pred_full=full)
    assert not over.is_valid


def test_gap_missing_y_pred_full_is_reason():
    y = np.arange(50, dtype=float)
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=y)
    assert not result.is_valid
    assert any("y_pred_full is not provided" in r for r in result.reasons)


def test_gap_rmse_oof_zero_is_division_by_zero_reason():
    y = np.arange(100, dtype=float)
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=y, y_pred_full=y)
    assert not result.is_valid
    assert any("RMSE_oof" in r and "division by zero" in r for r in result.reasons)
    assert np.isinf(result.generalization_gap)


# ──────────────────────────────────────────────────────────────────────────────
#  Комплексные сценарии (positive / negative из issue)
# ──────────────────────────────────────────────────────────────────────────────


def test_positive_honest_nonlinear_passes_all_circuits_clean_data():
    rng = np.random.RandomState(7)
    n = 500
    X = rng.randn(n, 6)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n)
    oof, full, model = _honest_oof_full(
        lambda: RandomForestRegressor(max_depth=5, n_estimators=100, random_state=0),
        X,
        y,
    )
    result = ModelSanityGate().check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.reasons == []
    assert result.diversity_ratio > 0.15
    assert result.generalization_gap <= 1.5
    assert result.dead_features_count == 0


def test_positive_honest_model_passes_on_noisy_r2_about_0_3():
    # Зашумлённые данные с теоретическим R²≈0.3: честная модель НЕ
    # дисквалифицируется (мягкий порог diversity — ключевое изменение v2).
    # Var(signal)=6 (сумма 6 независимых N(0,1)); шум выбираем так, чтобы
    # R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3 ровно.
    rng = np.random.RandomState(3)
    n = 200
    X = rng.randn(n, 6)
    signal = X.sum(axis=1)
    noise_var = 6.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=2.0), X, y)
    result = ModelSanityGate().check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.reasons == []
    assert 0.1 < result.diversity_ratio < 0.5  # ≈ R², много выше жёсткого 0.15
    assert result.generalization_gap <= 1.5


def test_negative_elasticnet_zero_weights_and_zero_diversity():
    # Реальный ElasticNet с большой регуляризацией зануляет ВСЕ веса
    # (проверяем предпосылку); сам гейт получает копию этих весов через
    # _CoefModel, чтобы результат не зависел от версии sklearn.
    rng = np.random.RandomState(5)
    X = rng.randn(120, 15)
    y = 2.0 * X[:, 0] + rng.randn(120)
    en = ElasticNet(alpha=500.0, random_state=0).fit(X, y)
    assert float((np.abs(en.coef_) <= 1e-12).mean()) == 1.0  # все веса нулевые
    model = _CoefModel(en.coef_)
    flat = np.full(120, y.mean())
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=flat, y_pred_full=en.predict(X), X=X, model=model
    )
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert any("Circuit C" in r and "combination" in r for r in result.reasons)


def test_negative_svr_sigmoid_tanh_plateau():
    # Сценарий из issue: SVR с сигмоидным ядром при малом gamma и coef0=0
    # упирается в плато tanh и выдаёт почти константные предсказания.
    # Сначала проверяем предпосылку (разброс предсказаний ≪ разброса y),
    # затем — что контур А это детектирует.
    X, y = _dataset(n=150)
    sigmoid = SVR(kernel="sigmoid", gamma=0.001, coef0=0.0).fit(X, y)
    pred = sigmoid.predict(X)
    assert pred.std() < 0.1 * y.std()  # плато tanh: предсказания почти const
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=pred, y_pred_full=pred, X=X, model=sigmoid
    )
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert result.diversity_ratio < 0.15


def test_negative_deep_tree_rmse_full_zero():
    X, y = _dataset(n=120)
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    result = ModelSanityGate(check_diversity=False, check_unique=False).check(
        y=y, y_pred_oof=tree.predict(X), y_pred_full=tree.predict(X), X=X, model=tree
    )
    assert not result.is_valid
    assert any("Circuit D" in r for r in result.reasons)


def test_negative_one_nn():
    X, y = _dataset(n=120)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    result = ModelSanityGate(check_diversity=False, check_unique=False).check(
        y=y, y_pred_oof=knn.predict(X), y_pred_full=knn.predict(X), X=X, model=knn
    )
    assert not result.is_valid
    assert any("Circuit D" in r for r in result.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Режимы и edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_warn_only_returns_counts_without_disqualification():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(warn_only=True, check_dead_features=False)
    result = gate.check(y=y, y_pred_oof=np.full(100, y.mean()), y_pred_full=y)
    assert result.is_valid is True  # нет права дисквалификации
    assert len(result.reasons) >= 2  # причины посчитаны для отчёта
    assert result.severity == 1  # capped на уровне «предупреждение»
    assert result.diversity_ratio == 0.0


def test_all_circuits_disabled_always_valid():
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=np.arange(50, dtype=float), y_pred_oof=np.full(50, 1.0))
    assert result.is_valid
    assert result.reasons == []
    assert result.soft_warnings == []
    assert np.isnan(result.diversity_ratio)
    assert np.isnan(result.unique_ratio)
    assert result.dead_features_count == 0
    assert np.isnan(result.generalization_gap)
    assert result.severity == 0


def test_nan_predictions_are_masked_like_oof_rmse():
    # Структурированные предсказания: каждая строка уникальна после
    # округления, поэтому unique_ratio = 1.0 и с маской, и без неё —
    # сравнение строгое (ревью PR #32: abs=0.2 было слишком мягким).
    rng = np.random.RandomState(0)
    n = 100
    y = rng.randn(n) * 10.0
    pred = np.linspace(-5.0, 5.0, n)  # шаг ≈ 0.1 > granularity округления
    pred_masked = pred.copy()
    pred_masked[::5] = np.nan  # 20% непокрытых строк
    gate = ModelSanityGate(
        check_dead_features=False, check_generalization_gap=False
    )
    full_result = gate.check(y=y, y_pred_oof=pred, y_pred_full=y)
    masked_result = gate.check(y=y, y_pred_oof=pred_masked, y_pred_full=y)
    # Маскирование не меняет семантику: оба проходят/проваливают одинаково,
    # а доля уникальных среди валидных строк одинакова.
    assert full_result.is_valid == masked_result.is_valid
    assert masked_result.unique_ratio == pytest.approx(1.0)
    assert masked_result.unique_ratio == pytest.approx(full_result.unique_ratio)


def test_all_nan_predictions_fail_with_reason():
    y = np.arange(50, dtype=float)
    gate = ModelSanityGate(check_dead_features=False, check_generalization_gap=False)
    result = gate.check(y=y, y_pred_oof=np.full(50, np.nan))
    assert not result.is_valid
    assert any("no finite" in r for r in result.reasons)


def test_nan_in_y_is_masked():
    rng = np.random.RandomState(0)
    y = rng.randn(60)
    y[::10] = np.nan
    gate = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=rng.randn(60))
    assert result.is_valid  # считается по конечным парам, без падения


def test_severity_mapping():
    y = np.arange(100, dtype=float)
    one_reason = ModelSanityGate(
        check_diversity=True,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert one_reason.severity == 2

    two_reasons = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert two_reasons.severity == 3

    three_reasons = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=True,
        check_generalization_gap=False,
    ).check(
        y=y,
        y_pred_oof=np.full(100, y.mean()),
        model=_CoefModel(np.zeros(4)),
    )
    assert three_reasons.severity == 4

    clean = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.arange(100, dtype=float))
    assert clean.severity == 0


def test_constructor_validation():
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=-0.1)
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=1.5)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_count=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_ratio=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_ratio=1.5)
    with pytest.raises(ValueError):
        ModelSanityGate(max_dead_feature_ratio=-0.1)
    with pytest.raises(ValueError):
        ModelSanityGate(max_generalization_gap=0.5)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_repeats=0)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_max_rows=0)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_max_features=-1)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_tolerance=-1e-6)


def test_input_contract_validation():
    gate = ModelSanityGate(check_diversity=False, check_unique=False)
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(11))
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(10), X=np.zeros((11, 2)))
    with pytest.raises(ValueError):
        gate.check(y=None, y_pred_oof=np.arange(10))
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10).reshape(2, 5), y_pred_oof=np.arange(10))
    # list-X не имеет .shape → ValueError, а не AttributeError (ревью PR #32).
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(10), X=[[0.0] * 3] * 10)


def test_module_is_standalone_clean_import():
    # Модуль не зависит от остального движка: импортируем напрямую.
    import importlib

    mod = importlib.import_module(
        "configurable_automl_engine.training_engine.sanity_gate"
    )
    assert hasattr(mod, "ModelSanityGate")
    assert hasattr(mod, "SanityCheckResult")


# ──────────────────────────────────────────────────────────────────────────────
#  Дополнительное покрытие edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_input_2d_column_vector_is_accepted():
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    y = np.arange(10).reshape(-1, 1)
    result = gate.check(y=y, y_pred_oof=np.arange(10).reshape(-1, 1))
    assert result.is_valid


def test_input_3d_vector_rejected():
    gate = ModelSanityGate(check_diversity=False, check_unique=False)
    with pytest.raises(ValueError):
        gate.check(y=np.zeros((2, 2, 2)), y_pred_oof=np.zeros((2, 2, 2)))


def test_gap_no_finite_pairs_for_rmse():
    y = np.arange(50, dtype=float)
    result = _gap_only_gate(1.5).check(
        y=y, y_pred_oof=y, y_pred_full=np.full(50, np.nan)
    )
    assert not result.is_valid
    assert any("no finite pairs to compute RMSE" in r for r in result.reasons)


def test_unique_constant_target_uses_fallback_precision():
    # Константный y: std(y)=0 → fallback-точность округления; константный
    # прогноз даёт nunique=1 → провал контура Б (см. edge case issue).
    y = np.full(60, 7.0)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(60, 7.0))
    assert not result.is_valid
    assert any("Circuit B" in r for r in result.reasons)


def test_dead_features_empty_coef_is_reason():
    y = np.arange(40, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_CoefModel(np.array([])))
    assert not result.is_valid
    assert any("empty coef_" in r for r in result.reasons)


def test_dead_features_1d_x_is_reason():
    rng = np.random.RandomState(0)
    y = rng.randn(50)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=rng.randn(50))
    assert not result.is_valid
    assert any("2-D matrix" in r for r in result.reasons)


def test_dead_features_nan_predictions_on_subsample_is_reason():
    class _NanPredictModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.full(len(X), np.nan)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_NanPredictModel(), X=X)
    assert not result.is_valid
    assert any("no finite" in r for r in result.reasons)


def test_dead_features_predict_fails_only_on_permuted_x():
    class _FailsAfterFirstCall:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X: Any) -> np.ndarray:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("cannot predict permuted X")
            return np.full(len(X), 1.0)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_FailsAfterFirstCall(), X=X)
    assert not result.is_valid
    assert any("prediction failed on permuted X" in r for r in result.reasons)


def test_dead_features_predict_wrong_output_length_is_reason():
    class _WrongLengthModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(len(X) + 3)  # неверная длина выхода

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_WrongLengthModel(), X=X)
    assert not result.is_valid
    assert any("values for" in r and "Circuit C" in r for r in result.reasons)


def test_dead_features_permuted_column_nan_pairs_are_skipped():
    class _NanOnPermutedX:
        """Базовое предсказание ок, на перемешанном X — NaN (не падение)."""

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X: Any) -> np.ndarray:
            self.calls += 1
            if self.calls > 1:
                return np.full(len(X), np.nan)
            return np.full(len(X), 1.0)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_NanOnPermutedX(), X=X)
    # Все колонки пропущены (нет валидных пар) — не падение, dead_count=0.
    assert result.is_valid
    assert result.dead_features_count == 0


def test_dead_features_row_subsampling_failure_is_reason():
    class _UnsliceableX:
        """Имеет .shape, но не умеет слайситься по строкам."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        def __getitem__(self, idx: Any) -> Any:
            raise RuntimeError("cannot slice")

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=_UnsliceableX())
    assert not result.is_valid
    assert any("row subsampling" in r and "Circuit C" in r for r in result.reasons)


def test_dead_features_unexpected_permutation_error_is_reason():
    class _ListReturningX:
        """__getitem__ возвращает list без .ndim → падение внутри impl."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        def __getitem__(self, idx: Any) -> list[list[float]]:
            return [[0.0] * 4 for _ in range(len(idx))]

    class _DummyModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(len(X))

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_DummyModel(), X=_ListReturningX())
    # Сбой внутри пермутационного пути превращается в reason, а не падение.
    assert not result.is_valid
    assert any("permutation check failed" in r for r in result.reasons)


def test_dead_features_column_permutation_failure_is_reason():
    class _BrokenSparseLike:
        """toarray() падает — имитация сбойного sparse-объекта."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        @property
        def ndim(self) -> int:
            return 2

        @property
        def toarray(self) -> Any:
            raise RuntimeError("toarray boom")

        def __getitem__(self, idx: Any) -> "_BrokenSparseLike":
            return self

    class _DummyModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(X.shape[0])

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(
        y=y, y_pred_oof=y, model=_DummyModel(), X=_BrokenSparseLike()
    )
    assert not result.is_valid
    assert any("column permutation failed" in r for r in result.reasons)


def test_core_estimator_unwraps_nested_pipeline_objects():
    from configurable_automl_engine.training_engine.sanity_gate import _core_estimator

    class Step:
        def __init__(self, name: str, obj: Any) -> None:
            self.name = name
            self.obj = obj

    class FakePipeline:
        def __init__(self, steps: list[Step]) -> None:
            self._steps = {s.name: s.obj for s in steps}

        @property
        def named_steps(self) -> dict[str, Any]:
            return self._steps

    inner = FakePipeline([Step("model", _CoefModel([1.0, 0.0]))])
    outer = FakePipeline([Step("scale", object()), Step("inner", inner)])
    est = _core_estimator(outer)
    assert np.array_equal(est.coef_, [1.0, 0.0])

    # Пайплайн без шагов и цикличный объект не зацикливаются.
    empty = FakePipeline([])
    assert _core_estimator(empty) is empty
    cyclic = FakePipeline([Step("self", None)])
    cyclic._steps["self"] = cyclic
    assert _core_estimator(cyclic) is cyclic

    class WeirdPipeline:
        @property
        def named_steps(self) -> Any:
            return object()  # нет .values()

    weird = WeirdPipeline()
    assert _core_estimator(weird) is weird


def test_gap_y_pred_full_length_mismatch_rejected():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        _gap_only_gate(1.5).check(y=y, y_pred_oof=y, y_pred_full=np.arange(11))


def test_constructor_zero_coef_tolerance_validation():
    with pytest.raises(ValueError):
        ModelSanityGate(zero_coef_tolerance=-1e-6)

# ──────────────────────────────────────────────────────────────────────────────
#  T6 (issue #66): уровни логирования soft_warnings
# ──────────────────────────────────────────────────────────────────────────────


def test_soft_warnings_logged_at_debug_not_warning(caplog):
    """soft_warnings — DEBUG/INFO, никогда WARNING (T6).

    Dead-признаки у линейных моделей («сигнал») логируются на уровне
    DEBUG/INFO: они не дисквалифицируют сами по себе и не должны
    подниматься до WARNING.
    """
    import logging

    rng = np.random.RandomState(11)
    X = rng.randn(200, 30)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(200)
    lasso = Lasso(alpha=0.06, random_state=0).fit(X, y)
    gate = ModelSanityGate()

    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        result = gate.check(
            y=y,
            y_pred_oof=lasso.predict(X),
            y_pred_full=lasso.predict(X),
            X=X,
            model=lasso,
        )

    assert result.is_valid
    assert any("soft warning" in r.message for r in caplog.records)
    # Детали каждого soft_warning — на DEBUG.
    assert any("Sanity gate soft warning" in r.message for r in caplog.records)
    # Ни одного WARNING и выше (требование T6).
    assert all(r.levelno < logging.WARNING for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
#  T7 (issue #69): «золотые» сценарии на синтетических данных N=10–30
# ──────────────────────────────────────────────────────────────────────────────


def _small_dataset(n: int = 25, seed: int = 7, noise: float = 0.5, p: int = 4):
    """Синтетический датасет малого размера: X0/X1 информативны, остальное — шум."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def test_golden_small_n_honest_gbr_passes_all_circuits():
    """Positive (T7, п. 5e): честная модель (градиентный бустинг) на N=30.

    Маленькие датасеты (N=10–30) не должны ложно дисквалифицировать честные
    модели: diversity > 0.15, unique достаточен, gap ≤ 1.5.
    """
    X, y = _small_dataset(n=30, seed=7)
    oof, full, model = _honest_oof_full(
        lambda: GradientBoostingRegressor(
            max_depth=1, n_estimators=20, learning_rate=0.05, random_state=0
        ),
        X,
        y,
        folds=5,
    )
    result = ModelSanityGate().check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert result.is_valid
    assert result.reasons == []
    assert result.diversity_ratio > 0.15
    assert result.unique_ratio >= 0.05
    assert result.generalization_gap <= 1.5


@pytest.mark.parametrize(
    "name, factory, expected_circuit",
    [
        (
            "elasticnet_zero",
            lambda: ElasticNet(alpha=500.0, random_state=0),
            "Circuit A",
        ),
        (
            "svr_sigmoid",
            lambda: SVR(kernel="sigmoid", gamma=0.001, coef0=0.0),
            "Circuit A",
        ),
        (
            "deep_tree",
            lambda: DecisionTreeRegressor(max_depth=20, random_state=0),
            "Circuit D",
        ),
        (
            "knn_1",
            lambda: KNeighborsRegressor(n_neighbors=1),
            "Circuit D",
        ),
    ],
)
def test_golden_small_n_degenerate_models_fail(
    name: str, factory: Callable[[], Any], expected_circuit: str
):
    """Negative (T7, п. 5a–d): вырожденные модели на N=25 дисквалифицируются.

    (a) ElasticNet с занулёнными коэффициентами («полка»);
    (b) SVR-sigmoid, упёршийся в плато tanh;
    (c) дерево глубины 20 (RMSE_train≈0);
    (d) 1-NN.
    """
    X, y = _small_dataset(n=25, seed=7)
    oof, full, model = _honest_oof_full(factory, X, y, folds=5)
    result = ModelSanityGate().check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert not result.is_valid
    assert any(expected_circuit in r for r in result.reasons)
    # У вырожденных моделей обязана быть хотя бы одна причина.
    assert result.reasons


def test_v2_wide_data_lasso_high_zero_ratio_soft_warning_only():
    """v2 (T7, п. 7): широкие данные, Lasso с 60–90% нулевых коэффициентов.

    Высокая доля нулевых коэффициентов — soft_warning, но НЕ дисквалификация:
    на широких зашумлённых данных Lasso легитимно зануляет признаки, а
    diversity при нормальном сигнале высокий (комбинация C+A не срабатывает).
    """
    rng = np.random.RandomState(21)
    n, p = 100, 80
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(n)
    lasso = Lasso(alpha=0.1, random_state=0).fit(X, y)
    gate = ModelSanityGate()
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(lasso.coef_) <= tol).mean())
    assert 0.6 <= zero_ratio <= 0.95  # предпосылка: 60–90% нулей

    result = gate.check(
        y=y, y_pred_oof=lasso.predict(X), y_pred_full=lasso.predict(X), X=X, model=lasso
    )
    assert result.is_valid  # не дисквалификация
    assert result.reasons == []
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert result.diversity_ratio > 0.15  # нормальный diversity


def test_v2_noisy_honest_model_not_disqualified_small_n():
    """v2 (T7, п. 7): зашумлённые данные, честная модель — гейт НЕ дисквалифицирует.

    R²≈0.3 (Var(signal)=3 при 3 информативных признаках, шум подобран так,
    чтобы R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3) — мягкий порог
    diversity (0.15) пропускает честную модель даже на малой выборке N=30.
    """
    rng = np.random.RandomState(3)
    n = 30
    X = rng.randn(n, 3)
    signal = X.sum(axis=1)
    noise_var = 3.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)

    class _LinearHonest:
        """Линейная честная модель: восстанавливает сигнал с шумом."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self._X = X
            self._y = y

        def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearHonest":
            # OLS по методу наименьших квадратов — без sklearn-зависимости.
            Xd = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            self._coef = coef
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            Xd = np.column_stack([np.ones(len(X)), X])
            return Xd @ self._coef

    model = _LinearHonest(X, y).fit(X, y)
    pred = model.predict(X)
    # На OOF-семантике теста считаем честный OOF через KFold.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        m = _LinearHonest(X, y).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=oof, y_pred_full=pred, X=X, model=model
    )
    assert result.is_valid  # НЕ дисквалифицируется (мягкий diversity-порог)
    assert result.reasons == []
    # Diversity ≈ R² зашумлённой честной модели: много выше жёсткого порога
    # 0.15, но заметно ниже «идеального» (шум не даёт R² ≈ 1).
    assert result.diversity_ratio > 0.15
    assert result.diversity_ratio < 0.9
