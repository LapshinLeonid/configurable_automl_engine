"""Тесты адаптивного диапазона ``epsilon`` для SVR (issue #55).

Покрытие:
- Модульные тесты хелпера ``compute_svr_epsilon_bounds`` (границы из σ(y),
  fallback для константного/NaN-таргета, клампинг для float_log, IQR).
- Интеграционные тесты тюнера: epsilon предлагается Optuna строго в
  адаптивных границах; приоритет пользовательского ``epsilon``.
- e2e через ``train_best_model``: ни один триал не выходит за адаптивный
  диапазон, ``best_params["epsilon"]`` внутри диапазона.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import configurable_automl_engine.tuner as tuner_module
from configurable_automl_engine.common.hyperopt_defaults import (
    SVR_EPSILON_FALLBACK,
    SearchSpaceEntry,
    compute_svr_epsilon_bounds,
)
from configurable_automl_engine.training_engine.component import (
    PreparedData,
    execute_phases,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    GeneralCfg,
    HPOPhaseCfg,
    ValidationStrategy,
)
from configurable_automl_engine.tuner import (
    _make_svr_space,
    optimize,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Хелпер: границы из разброса y
# ──────────────────────────────────────────────────────────────────────────────
def test_helper_known_sigma_returns_relative_bounds():
    """Для y с известным σ возвращает (0.005·σ, 0.10·σ)."""
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sigma = float(np.std(y))  # σ ≈ 1.41421356 (population std, ddof=0)
    low, high = compute_svr_epsilon_bounds(y)

    assert low == pytest.approx(0.005 * sigma)
    assert high == pytest.approx(0.10 * sigma)
    assert 0.0 < low < high


def test_helper_accepts_series_and_dataframe():
    """Хелпер принимает pd.Series и pd.DataFrame (одна колонка)."""
    y_series = pd.Series(np.arange(20, dtype=float))
    y_frame = pd.DataFrame({"y": np.arange(20, dtype=float)})
    y_list = list(range(20))

    expected = compute_svr_epsilon_bounds(y_series)
    assert compute_svr_epsilon_bounds(y_frame) == expected
    assert compute_svr_epsilon_bounds(y_list) == expected


def test_helper_custom_rel_factors():
    """Параметры rel_min/rel_max переопределяются."""
    y = np.array([0.0, 2.0, 4.0])
    low, high = compute_svr_epsilon_bounds(y, rel_min=0.01, rel_max=0.2)
    sigma = float(np.std(y))
    assert low == pytest.approx(0.01 * sigma)
    assert high == pytest.approx(0.2 * sigma)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([1.0, 1.0, 1.0, 1.0]),  # константный y → σ = 0
        np.array([1.0, np.nan, 3.0]),  # NaN
        np.array([np.nan] * 4),  # только NaN
        np.array([1.0, np.inf]),  # +inf
        np.array([-np.inf, 2.0]),  # -inf
        np.array([]),  # пустой вектор
        np.array([0.0, 0.0]),  # нулевой разброс
    ],
)
def test_helper_fallback_for_constant_or_non_finite_y(bad_y):
    """Константный/NaN/±inf/пустой y → fallback (1e-4, 1e-2) без исключений."""
    assert compute_svr_epsilon_bounds(bad_y) == SVR_EPSILON_FALLBACK


def test_helper_very_small_sigma_clamps_low_positive():
    """Очень малый σ → low клампится > 0 (инвариант float_log), low < high."""
    y = np.array([1e-12, 1.1e-12, 1.2e-12, 1.3e-12])
    low, high = compute_svr_epsilon_bounds(y)

    # Без клампинга low был бы ~5.6e-16; гарантируем строго положительный минимум.
    assert low > 0.0
    assert low == pytest.approx(1e-8)  # клампинг к SVR_EPSILON_MIN_LOW
    assert low < high


def test_helper_iqr_strategy_robust_to_outliers():
    """IQR-вариант робастен к выбросам: разброс меньше, чем у std."""
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0], dtype=float)

    std_low, std_high = compute_svr_epsilon_bounds(y, strategy="std")
    iqr_low, iqr_high = compute_svr_epsilon_bounds(y, strategy="iqr")

    expected_spread = (np.percentile(y, 75) - np.percentile(y, 25)) / 1.349
    assert iqr_low == pytest.approx(0.005 * expected_spread)
    assert iqr_high == pytest.approx(0.10 * expected_spread)
    # Выброс завышает σ, но не IQR → robust-диапазон уже.
    assert iqr_high < std_high


def test_helper_invalid_strategy_raises():
    """Неизвестная стратегия разброса → ValueError."""
    with pytest.raises(ValueError, match="Unknown epsilon spread strategy"):
        compute_svr_epsilon_bounds([1, 2, 3], strategy="mad")


def test_helper_non_positive_min_low_raises():
    """min_low <= 0 → ValueError (log-шкала требует low > 0)."""
    with pytest.raises(ValueError, match="min_low must be > 0"):
        compute_svr_epsilon_bounds([1, 2, 3], min_low=0.0)
    with pytest.raises(ValueError, match="min_low must be > 0"):
        compute_svr_epsilon_bounds([1, 2, 3], min_low=-1e-3)


def test_helper_target_scenario_upper_bound_near_025():
    """Регрессия целевого сценария: при σ ≈ 0.244 верхняя граница ≈ 0.025.

    Зона 0.06–0.08 (вырождение в плоскую линию) физически недостижима.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.244, size=500)
    low, high = compute_svr_epsilon_bounds(y)

    assert high == pytest.approx(0.10 * float(np.std(y)))
    assert high < 0.05
    assert 0.02 <= high <= 0.03


# ──────────────────────────────────────────────────────────────────────────────
# 2. Тюнер: фабрика пространства SVR и интеграция
# ──────────────────────────────────────────────────────────────────────────────
def test_make_svr_space_uses_adaptive_epsilon_bounds():
    """Фабрика предлагает epsilon строго в границах, производных от y."""
    y = np.arange(50, dtype=float)
    low, high = compute_svr_epsilon_bounds(y)

    space_fn = _make_svr_space(y)
    trial = MagicMock()
    space_fn(trial)

    trial.suggest_float.assert_any_call("epsilon", low, high, log=True)


def _mock_optimize_infra(
    monkeypatch,
    collect_epsilons: list[float],
    model_factory=None,
):
    """Замокать инфраструктуру оптимизации и собирать epsilon из триалов."""

    def default_factory(algo: str, **params):
        return Ridge()

    factory = model_factory or default_factory

    def mock_create_model(algo: str, **params):
        if "epsilon" in params:
            collect_epsilons.append(float(params["epsilon"]))
        return factory(algo, **params)

    monkeypatch.setattr(
        "configurable_automl_engine.tuner.create_model", mock_create_model
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.get_scorer_object",
        lambda name, global_y=None: lambda model, X, y: 0.5,
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.model_selection.cross_val_score",
        lambda *a, **kw: [0.5, 0.5],
    )


def test_tuner_svr_default_space_all_trials_within_adaptive_bounds(monkeypatch):
    """Тюнер (svr, дефолтный спейс): все значения epsilon триалов ∈ [eps_min, eps_max]."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    _, params, _ = optimize("svr", X, y, n_trials=6, random_state=42)

    assert len(seen) >= 6
    for eps in seen:
        assert low <= eps <= high
    assert low <= params["epsilon"] <= high


def test_tuner_svr_dict_override_without_epsilon_gets_adaptive(monkeypatch):
    """space_overrides[svr] — dict без epsilon → добавляются адаптивные границы."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    space = {"C": SearchSpaceEntry.model_validate([1e-2, 100.0, "float_log"])}
    _, params, _ = optimize(
        "svr",
        X,
        y,
        n_trials=5,
        random_state=42,
        space_overrides={"svr": space},
    )

    assert seen
    for eps in seen:
        assert low <= eps <= high
    assert low <= params["epsilon"] <= high


def test_tuner_svr_user_epsilon_override_has_priority(monkeypatch):
    """Явный пользовательский epsilon (dict override) → адаптивная логика не применяется."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    # Адаптивный диапазон для такого y заведомо меньше 0.1 (rel_max=0.10).
    _, adaptive_high = compute_svr_epsilon_bounds(y)
    assert adaptive_high < 0.5

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    user_eps = SearchSpaceEntry.model_validate([0.5, 1.0, "float_log"])
    _, params, _ = optimize(
        "svr",
        X,
        y,
        n_trials=5,
        random_state=42,
        space_overrides={"svr": {"epsilon": user_eps}},
    )

    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0
        assert eps > adaptive_high  # вне адаптивного диапазона — override применён
    assert 0.5 <= params["epsilon"] <= 1.0


def test_tuner_svr_callable_override_unchanged(monkeypatch):
    """space_overrides[svr] как callable (старый механизм) → поведение не меняется."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    def custom_space(trial):
        return {"epsilon": trial.suggest_float("epsilon", 0.2, 0.3)}

    _, params, _ = optimize(
        "svr",
        X,
        y,
        n_trials=4,
        random_state=42,
        space_overrides={"svr": custom_space},
    )

    assert seen
    for eps in seen:
        assert 0.2 <= eps <= 0.3
    assert 0.2 <= params["epsilon"] <= 0.3


def test_tuner_svr_initial_params_epsilon_valid_in_adaptive_range(monkeypatch):
    """enqueue_trial: epsilon из предыдущей фазы (refine_winner) валиден в адаптивном диапазоне.

    Границы считаются от того же y_train в обеих фазах, поэтому значение
    победителя первой фазы гарантированно лежит внутри диапазона второй.
    """
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)
    prev_epsilon = (low + high) / 2.0  # «победитель» предыдущей фазы

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    _, params, _ = optimize(
        "svr",
        X,
        y,
        n_trials=3,
        random_state=42,
        initial_params={
            "C": 1.0,
            "epsilon": prev_epsilon,
            "kernel": "rbf",
            "gamma": "scale",
        },
    )

    assert low <= prev_epsilon <= high
    assert seen[0] == prev_epsilon  # первый триал — enqueued значение
    assert low <= params["epsilon"] <= high


def test_tuner_svr_constant_y_with_user_epsilon_works(monkeypatch):
    """σ = 0 + пользовательский epsilon → работает пользовательское значение, не fallback."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([1.0] * 60)  # константный y → σ = 0

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    user_eps = SearchSpaceEntry.model_validate([0.5, 1.0, "float_log"])
    _, params, _ = optimize(
        "svr",
        X,
        y,
        n_trials=4,
        random_state=42,
        space_overrides={"svr": {"epsilon": user_eps}},
    )

    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0  # пользовательский диапазон, а не fallback (1e-4, 1e-2)
    assert 0.5 <= params["epsilon"] <= 1.0


def test_tuner_svr_constant_y_falls_back(monkeypatch):
    """Константный y → fallback (1e-4, 1e-2), запуск без исключений."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([2.5] * 60)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    _, params, _ = optimize("svr", X, y, n_trials=4, random_state=42)

    assert seen
    for eps in seen:
        assert SVR_EPSILON_FALLBACK[0] <= eps <= SVR_EPSILON_FALLBACK[1]
    assert SVR_EPSILON_FALLBACK[0] <= params["epsilon"] <= SVR_EPSILON_FALLBACK[1]


def test_tuner_svr_nan_y_falls_back(monkeypatch):
    """y с NaN → fallback, без исключений."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([1.0, np.nan] + list(np.random.randn(58)))

    class _FitOnlyEstimator:
        """Заглушка вместо реального регрессора: Ridge падает на y с NaN."""

        def fit(self, *args, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen, model_factory=lambda algo, **p: _FitOnlyEstimator())

    _, params, _ = optimize("svr", X, y, n_trials=4, random_state=42)

    assert seen
    for eps in seen:
        assert SVR_EPSILON_FALLBACK[0] <= eps <= SVR_EPSILON_FALLBACK[1]
    assert SVR_EPSILON_FALLBACK[0] <= params["epsilon"] <= SVR_EPSILON_FALLBACK[1]


def test_tuner_other_algorithms_unaffected(monkeypatch):
    """Поведение остальных алгоритмов не меняется (например, ridge)."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series(np.random.randn(60))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    _, params, _ = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        random_state=42,
        space_overrides={"ridge": {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float_log"])}},
    )

    assert 0.1 <= params["alpha"] <= 1.0
    assert "epsilon" not in params


# ──────────────────────────────────────────────────────────────────────────────
# 3. e2e через train_best_model
# ──────────────────────────────────────────────────────────────────────────────
def _svr_only_cfg(model_path: str, n_trials: int = 3) -> str:
    return f"""
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: {n_trials}
      action: "all_algorithms"
algorithms:
  svr:
    enable: true
    limit_hyperparameters: true
"""


def test_e2e_svr_adaptive_epsilon_all_trials_and_best_within_bounds(
    tmp_path: Path, monkeypatch
):
    """e2e: ни один триал не выходит за адаптивный диапазон; best_params внутри."""
    rows = 60
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((rows, 4)), columns=[f"f{i}" for i in range(4)])
    # Целевая переменная с детерминированным разбросом σ(y).
    df["target"] = np.arange(rows, dtype=float)
    sigma = float(np.std(df["target"].to_numpy()))
    low, high = 0.005 * sigma, 0.10 * sigma

    # Перехватываем каждый вызов create_model в тюнере (все триалы + финальный fit).
    real_create_model = tuner_module.create_model
    seen: list[float] = []

    def capturing_create_model(algo: str, **params):
        if algo == "svr" and "epsilon" in params:
            seen.append(float(params["epsilon"]))
        return real_create_model(algo, **params)

    monkeypatch.setattr(tuner_module, "create_model", capturing_create_model)

    model_path = tmp_path / "model.pkl"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(_svr_only_cfg(str(model_path)), encoding="utf-8")

    res = train_best_model(cfg_file, df, target="target", model_path_override=model_path)

    assert res["algorithm"] == "svr"
    assert seen, "ни один epsilon не был предложен — тест бесполезен"
    for eps in seen:
        assert low <= eps <= high
    assert low <= res["params"]["epsilon"] <= high


def test_e2e_svr_user_epsilon_override_respected(tmp_path: Path, monkeypatch):
    """e2e: пользовательский epsilon в YAML → адаптивный диапазон не применяется."""
    rows = 60
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.standard_normal((rows, 4)), columns=[f"f{i}" for i in range(4)])
    # Масштабируем таргет, чтобы адаптивный диапазон был заведомо ниже [0.5, 1.0].
    df["target"] = np.arange(rows, dtype=float) * 0.02
    sigma = float(np.std(df["target"].to_numpy()))
    adaptive_high = 0.10 * sigma
    assert adaptive_high < 0.5  # пользовательский диапазон [0.5, 1.0] вне адаптивного

    model_path = tmp_path / "model.pkl"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_text = f"""
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: 3
      action: "all_algorithms"
algorithms:
  svr:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      epsilon: [0.5, 1.0]
"""
    cfg_file.write_text(cfg_text, encoding="utf-8")

    real_create_model = tuner_module.create_model
    seen: list[float] = []

    def capturing_create_model(algo: str, **params):
        if algo == "svr" and "epsilon" in params:
            seen.append(float(params["epsilon"]))
        return real_create_model(algo, **params)

    monkeypatch.setattr(tuner_module, "create_model", capturing_create_model)

    res = train_best_model(cfg_file, df, target="target", model_path_override=model_path)

    assert res["algorithm"] == "svr"
    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0
        assert eps > adaptive_high  # вне адаптивного диапазона — override применён
    assert 0.5 <= res["params"]["epsilon"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Компонент: prepare_search_space (контракт приоритета пользователя)
# ──────────────────────────────────────────────────────────────────────────────
def _svr_config(tmp_path: Path, hyperparameters: dict | None = None) -> Config:
    """Config с единственным включённым алгоритмом svr."""
    algos: dict[str, dict] = {"svr": {"enable": True}}
    if hyperparameters:
        algos["svr"]["hyperparameters"] = hyperparameters
    return Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            path_to_model=str(tmp_path / "m.pkl"),
            phases=[HPOPhaseCfg(name="Coarse Search", n_trials=1, action="all_algorithms")],
        ),
        algorithms=algos,
    )


def _prepared_data() -> PreparedData:
    return PreparedData(
        X=pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0]),
        metric_user="rmse",
        metric_sklearn="neg_root_mean_squared_error",
        resolved_validation=ValidationStrategy.k_fold,
        resolved_n_folds=2,
        resolved_test_size=0.2,
        categorical_features=[],
        numerical_features=["f1", "f2"],
    )


def test_prepare_search_space_svr_default_omits_epsilon(
    tmp_path: Path, monkeypatch
):
    """prepare_search_space: svr без user-epsilon → epsilon опущен (адаптация в тюнере).

    Дефолтный слепой диапазон DEFAULT_SPACES["svr"]["epsilon"]=[1e-3, 1.0] не
    должен попадать в итоговое пространство: иначе тюнер не сможет отличить
    его от явного пользовательского значения.
    """
    cfg = _svr_config(tmp_path)
    prepared = _prepared_data()

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return (0.9, {"C": 1.0})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    results, _ = execute_phases(cfg, prepared)

    assert "svr" in results
    space = captured["search_space"]
    assert "epsilon" not in space, (
        "дефолтный epsilon не должен попадать в пространство — "
        "его место занимают адаптивные границы в тюнере"
    )
    # Остальные дефолтные параметры SVR сохраняются.
    assert "C" in space
    assert "kernel" in space
    assert "gamma" in space


def test_prepare_search_space_svr_user_epsilon_kept(tmp_path: Path, monkeypatch):
    """prepare_search_space: явный пользовательский epsilon сохраняется (приоритет)."""
    cfg = _svr_config(tmp_path, hyperparameters={"epsilon": [0.5, 1.0]})
    prepared = _prepared_data()

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return (0.9, {"C": 1.0})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    results, _ = execute_phases(cfg, prepared)

    assert "svr" in results
    space = captured["search_space"]
    assert "epsilon" in space
    assert space["epsilon"].low == 0.5
    assert space["epsilon"].high == 1.0


def test_prepare_search_space_other_algorithms_unaffected(tmp_path: Path, monkeypatch):
    """prepare_search_space: для остальных алгоритмов поведение не меняется."""
    cfg_ridge = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            path_to_model=str(tmp_path / "m.pkl"),
            phases=[
                HPOPhaseCfg(name="Coarse Search", n_trials=1, action="all_algorithms")
            ],
        ),
        algorithms={"ridge": {"enable": True}},
    )

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return (0.9, {"alpha": 0.1})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    results, _ = execute_phases(cfg_ridge, _prepared_data())

    assert "ridge" in results
    space = captured["search_space"]
    # Дефолтный alpha ridge сохраняется как раньше (ничего не опускается).
    assert "alpha" in space
    assert space["alpha"].dist_type == "float_log"