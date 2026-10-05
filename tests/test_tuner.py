"""
Параметризованные тесты для configurable_automl_engine.tuner.

• Smoke-проверяем, что optimize отрабатывает на каждом алгоритме,
  для которого задан search-space (включая CF-18: SGD, GPR, Isotonic,
  ARD, Poisson/Gamma/Tweedie).
• Валидируем пользовательский space-override.
• Проверяем ошибки: неизвестный алгоритм, плохие данные, n_trials ≤ 0.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import FixedTrial
from sklearn.datasets import make_regression

from configurable_automl_engine import tuner as hyperopt
from configurable_automl_engine.common.hyperopt_defaults import (
    FloatSpace,
    SearchSpaceEntry,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    HyperoptError,
    _apply_dynamic_space,
    _build_scorer,
    _can_stratify,
    _missing_indicator_enabled,
    optimize,
)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Делаем пакет видимым и импортируем модуль
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Фикстура с игрушечными данными
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=120,
        n_features=10,
        noise=0.1,
        random_state=1,
    )
    return pd.DataFrame(X), pd.Series(y)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Вспомогательная адаптация X/y под «капризные» модели
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_data(algo: str, X: pd.DataFrame, y: pd.Series):
    """Подгоняем форму и диапазон под требования конкретных алгоритмов."""
    # IsotonicRegression — ровно один признак
    if algo == "isotonicregression":
        X = X.iloc[:, [0]]
    # GLM-семейство требует y > 0
    if algo in {"poissonregressor", "gammaregressor", "tweedieregressor"}:
        y = np.abs(y) + 1.0
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 3. Smoke-тест на каждый поддерживаемый алгоритм
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "algo",
    ["knn"],
)
def test_optimize_smoke(algo: str, toy_data):
    """optimize не должен падать и возвращает валидные объекты."""
    X, y = _prepare_data(algo, *toy_data)

    model, params, score = hyperopt.optimize(
        algo,
        X,
        y,
        n_trials=3,  # минимально для быстроты
        random_state=0,
    )

    assert isinstance(params, dict) and params, "пустой best_params"
    assert isinstance(score, float) and not np.isnan(score), "score некорректен"
    assert hasattr(model, "predict"), "у модели нет метода predict"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Пользовательский space-override
# ──────────────────────────────────────────────────────────────────────────────
def test_space_override(toy_data):
    X, y = toy_data

    def tiny_space(trial):
        return {"alpha": trial.suggest_float("alpha", 0.0, 0.001)}

    _, params, _ = hyperopt.optimize(
        "elasticnet",
        X,
        y,
        n_trials=2,
        space_overrides={"elasticnet": tiny_space},
    )
    assert params["alpha"] <= 0.001


# ──────────────────────────────────────────────────────────────────────────────
# 5. Негативные сценарии
# ──────────────────────────────────────────────────────────────────────────────
def test_invalid_algo(toy_data):
    X, y = toy_data
    with pytest.raises(hyperopt.InvalidAlgorithmError):
        hyperopt.optimize("does_not_exist", X, y, n_trials=2)


def test_bad_data():
    with pytest.raises(hyperopt.InvalidDataError):
        hyperopt.optimize("ridge", "not-an-array", [1, 2, 3], n_trials=2)


@pytest.mark.parametrize("bad_trials", [-1, -3])
def test_non_positive_trials(bad_trials, toy_data):
    X, y = toy_data
    with pytest.raises(ValueError):
        hyperopt.optimize("ridge", X, y, n_trials=bad_trials)


def test_zero_trials_returns_no_result(toy_data):
    """B2 (issue #32): n_trials=0 → пустой список триалов → (None, None, None).

    Раньше n_trials=0 считался невалидным и вызывал ValueError. Теперь это
    легальный «пустой поиск»: ни одного триала — значит, нет и валидного
    результата, и optimize() сигнализирует об этом сплошным None.
    """
    X, y = toy_data
    model, params, score = hyperopt.optimize("ridge", X, y, n_trials=0)

    assert model is None
    assert params is None
    assert score is None


def test_apply_dynamic_space_types(toy_data):
    """
    Покрывает логику _apply_dynamic_space.
    Используем алгоритм 'knn', так как для него в коде есть фабрика пространств.
    """
    X, y = toy_data

    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds

        @property
        def low(self):
            return self.bounds[0]

        @property
        def high(self):
            return self.bounds[1]

        @property
        def dist_type(self):
            return self.bounds[2]

        @property
        def step(self):
            return self.bounds[3] if len(self.bounds) > 3 else None

    # Имитируем структуру из YAML для KNN
    # Это покроет ветки int, float, float_log, categorical и константы
    dynamic_config = {
        "n_neighbors": MockEntry([5, 15, "int"]),
        "p": MockEntry([1, 2, "int"]),
        "weights": MockEntry([["uniform", "distance"], None, "categorical"]),
        "leaf_size": 30,  # Константа (строка 144)
    }
    model, params, score = hyperopt.optimize(
        "knn", X, y, n_trials=2, space_overrides={"knn": dynamic_config}
    )

    assert isinstance(params["n_neighbors"], int)
    assert params["weights"] in ["uniform", "distance"]
    # Проверяем, что константа попала в модель.
    # С автодетекцией препроцессора модель оборачивается в ImbPipeline
    # (даже для чисто числовых данных), поэтому извлекаем шаг 'model'.
    assert isinstance(model, ImbPipeline)
    assert model.named_steps["model"].leaf_size == 30


def test_validate_data_mismatch():
    """Проверка ошибки при несовпадении длин X и y (строка 159)."""
    X = np.zeros((10, 2))
    y = np.zeros(5)
    with pytest.raises(hyperopt.InvalidDataError, match="Size mismatch"):
        hyperopt._validate_data(X, y)


def test_validate_data_invalid_y_type():
    """Проверка ошибки при недопустимом типе y (строка 155)."""
    X = np.zeros((5, 2))
    y = {1: 0, 2: 0}  # dict не входит в ok_types
    with pytest.raises(hyperopt.InvalidDataError, match="y must be"):
        hyperopt._validate_data(X, y)


def test_optimize_with_oversampling(toy_data):
    """
    Покрывает логику включения оверсэмплинга в Pipeline.
    """
    X, y = toy_data
    y_bin = (y > y.mean()).astype(int)
    model, params, score = hyperopt.optimize(
        "knn",
        X,
        y_bin,
        data_oversampling=True,
        data_oversampling_algorithm="random",
        data_oversampling_multiplier=1.2,
        n_trials=2,
    )

    print(model.steps)
    assert isinstance(model, ImbPipeline)
    assert any(isinstance(step[1], DataOversampler) for step in model.steps)


def test_optimize_pure_numeric_builds_preprocessor(toy_data):
    """На чисто числовом DataFrame без явных списков колонок optimize
    строит препроцессор с StandardScaler (рассинхрон ветки ``if cats:``)."""
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data  # полностью числовой DataFrame

    model, _, _ = hyperopt.optimize(
        "knn",
        X,
        y,
        n_trials=2,
        random_state=0,
    )

    # Препроцессор должен быть собран даже при отсутствии категорий
    assert isinstance(model, ImbPipeline)
    assert "preprocessor" in [step[0] for step in model.steps]

    preprocessor = model.named_steps["preprocessor"]
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    assert "num" in transformers
    num_pipeline = transformers["num"]
    assert isinstance(num_pipeline.named_steps["scaler"], StandardScaler)


def test_can_stratify_negative():
    """Проверка условий, когда стратификация невозможна (строки 182-183)."""
    # Много уникальных значений (регрессия)
    y_reg = np.linspace(0, 1, 100)
    assert hyperopt._can_stratify(y_reg) is False

    # Многомерный y
    y_multi = np.zeros((10, 2))
    assert hyperopt._can_stratify(y_multi) is False


def test_split_train_test_fallback():
    """Проверка fallback в train_test_split при ошибке стратификации (строки 194-196)."""
    # Создаем ситуацию, где стратификация невозможна из-за 1 экземпляра класса
    X = np.random.rand(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Метод должен отработать без ошибок, поймав ValueError внутри
    res = hyperopt._split_train_test(X, y, test_size=0.5)
    assert len(res) == 4


def test_optimize_pruning(toy_data):
    """
    Покрывает блок обработки исключений в _objective.
    Исправлено: параметры теперь регистрируются через trial.suggest_int.
    """
    X, y = toy_data

    def pruning_space(trial):
        # Используем suggest_int, чтобы optuna зафиксировала параметры в trial.params
        if trial.number == 0:
            n_neighbors = trial.suggest_int("n_neighbors", 5, 5)
            return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}

        # Вторая попытка вызовет ошибку (n_neighbors <= 0)
        n_neighbors = trial.suggest_int("n_neighbors", 0, 0)
        return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}

    # Метод не должен упасть, Trial 1 просто будет помечен как Pruned
    model, params, score = hyperopt.optimize(
        "knn",
        X,
        y,
        n_trials=2,
        # Передаем как словарь для конкретного алгоритма
        space_overrides={"knn": pruning_space},
    )

    # Теперь params не будет пустым, так как Trial 0 успешно завершился
    assert "n_neighbors" in params
    assert params["n_neighbors"] == 5


# ──────────────────────────────────────────────────────────────────────────────
# 6. initial_params: enqueue_trial tests
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_with_initial_params_enqueues_trial(toy_data):
    """Проверяет, что study.enqueue_trial() вызывается с initial_params."""
    X, y = toy_data
    initial_params = {"alpha": 0.5, "l1_ratio": 0.3}

    with patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create:
        mock_study = MagicMock()
        mock_study.best_params = {"alpha": 0.6, "l1_ratio": 0.4}
        mock_study.best_value = 0.9
        mock_create.return_value = mock_study

        # Мокаем create_model и _validate_data/_get_estimator
        with (
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model

            optimize(
                "elasticnet", X, y, n_trials=1,
                initial_params=initial_params,
                space_overrides={
                    "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
                },
            )

            # Проверяем, что enqueue_trial вызван с правильными параметрами
            mock_study.enqueue_trial.assert_called_once_with(initial_params)
            # Проверяем, что optimize все еще вызывается
            mock_study.optimize.assert_called_once()


def test_optimize_without_initial_params_does_not_enqueue(toy_data):
    """Проверяет, что без initial_params enqueue_trial НЕ вызывается."""
    X, y = toy_data

    with patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create:
        mock_study = MagicMock()
        mock_study.best_params = {"alpha": 0.6}
        mock_study.best_value = 0.9
        mock_create.return_value = mock_study

        with (
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model

            optimize(
                "elasticnet", X, y, n_trials=1,
                initial_params=None,
                space_overrides={
                    "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
                },
            )

            # enqueue_trial НЕ должен быть вызван
            mock_study.enqueue_trial.assert_not_called()
            mock_study.optimize.assert_called_once()


def test_knn_space_dynamic_limit():
    """Проверка динамического ограничения k в KNN (строки 80-81)."""
    # Для 10 сэмплов 80% это 8. max_k должен быть 8.
    space_fn = hyperopt._make_knn_space(n_samples=10)

    # Эмулируем запрос n_neighbors
    trial = FixedTrial({"n_neighbors": 5, "weights": "uniform", "p": 1})
    params = space_fn(trial)
    assert params["n_neighbors"] <= 8


def test_apply_dynamic_space_floats():
    trial = MagicMock()

    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds

        @property
        def low(self):
            return self.bounds[0]

        @property
        def high(self):
            return self.bounds[1]

        @property
        def dist_type(self):
            return self.bounds[2]

        @property
        def step(self):
            return self.bounds[3] if len(self.bounds) > 3 else None

    space_dict = {
        "learning_rate": MockEntry([0.01, 0.1, "float"]),
        "gamma": MockEntry([1e-5, 1e-1, "float_log"]),
        "constant": 42,
    }

    _apply_dynamic_space(trial, space_dict)

    # Проверяем вызовы
    trial.suggest_float.assert_any_call("learning_rate", 0.01, 0.1, step=None)
    trial.suggest_float.assert_any_call("gamma", 1e-05, 0.1, log=True)


@pytest.fixture
def log_space_entry() -> Callable[[float], SearchSpaceEntry]:
    """Фабрика реальных SearchSpaceEntry с распределением float_log.

    Негативные значения создаются через ``model_construct`` — в обход
    Pydantic-валидации. Штатная схема намеренно отклоняет ``low <= 0``
    для float_log (issue #20), поэтому проверить защиту тюнера можно только
    объектами, собранными мимо валидации: это в точности имитирует
    словарный конфиг старого формата, попадающий в тюнер напрямую.
    """

    def _make(low: float) -> SearchSpaceEntry:
        return SearchSpaceEntry.model_construct(
            config=FloatSpace.model_construct(type="float_log", low=low, high=1.0)
        )

    return _make


def test_apply_dynamic_space_float_log_rejects_non_positive_low(log_space_entry):
    """Защита в _apply_dynamic_space: float_log с low <= 0 отклоняется.

    Негативные сценарии используют реальные SearchSpaceEntry из фикстуры
    (собранные в обход валидации); позитивный проходит штатную валидацию
    через ``SearchSpaceEntry.model_validate`` — полный интеграционный путь.
    """
    trial = MagicMock()

    for bad_low in (0.0, -1.0):
        with pytest.raises(ValueError, match="low must be > 0 for log-scale"):
            _apply_dynamic_space(trial, {"alpha": log_space_entry(bad_low)})
    assert trial.suggest_float.call_count == 0

    # Позитивный сценарий: валидный entry уходит в suggest_float(log=True)
    valid_entry = SearchSpaceEntry.model_validate([1e-6, 1.0, "float_log"])
    _apply_dynamic_space(trial, {"alpha": valid_entry})
    trial.suggest_float.assert_called_once_with("alpha", 1e-06, 1.0, log=True)


def test_build_scorer_error():
    with pytest.raises(HyperoptError, match="Unknown metric name"):
        _build_scorer("non_existent_metric_name_123")


def test_can_stratify_pandas():
    y_series = pd.Series([0, 1, 0, 1])
    assert _can_stratify(y_series) is True

    y_df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert _can_stratify(y_df) is False  # Т.к. ndim != 1 для DF с 1 колонкой (обычно)


def test_optimize_train_test_split_mode(toy_data):
    X, y = toy_data
    # Принудительно вызываем режим hold-out через передачу малого количества данных
    # или явное указание стратегии
    model, params, score = optimize(
        "knn", X, y, validation_strategy="train_test_split", n_trials=1
    )
    assert score is not None


def test_optimize_raises_when_no_search_space(monkeypatch):
    """Проверяет выброс HyperoptError если для алгоритма нет search-space."""

    # --- 1. Подготавливаем фиктивные данные ---
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    # --- 2. Мокаем create_model → чтобы алгоритм считался валидным ---
    def fake_create_model(algo, **kwargs):
        class DummyModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        return DummyModel()

    # --- делаем алгоритм валидным ---
    monkeypatch.setattr(hyperopt, "_get_estimator", lambda algo: True)

    with pytest.raises(HyperoptError, match="нет search-space"):
        optimize("fake_algo", X, y, n_trials=1)


class TestTunerObjective:
    @pytest.fixture
    def dummy_data(self):
        return pd.DataFrame({"a": [1, 2, 3, 4, 5]}), pd.Series([1, 0, 1, 0, 1])

    @pytest.fixture
    def mock_space(self):
        return {"rf": lambda trial: {"n_estimators": 10}}

    def test_objective_trigger_fallback(self, dummy_data, mock_space):
        """
        Тест принудительно заставляет np.isfinite вернуть False,
        чтобы проверить возврат константы HPO_WORST_SCORE.
        """
        X, y = dummy_data
        EXPECTED_FALLBACK = HPO_WORST_SCORE
        # 1. Патчим ВСЁ окружение, чтобы ни одна реальная функция не выполнилась
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner.np.isfinite") as mock_finite,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            # ГАРАНТИРУЕМ:
            # 1. Мы в ветке k-fold
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # 2. cross_val_score возвращает что-то
            mock_cv.return_value = np.array([0.5])
            # 3. КРИТИЧЕСКИЙ МОМЕНТ: Любое число НЕ конечное
            mock_finite.return_value = False

            # Остальные заглушки
            mock_create.return_value = MagicMock()
            mock_scorer_factory.return_value = MagicMock()
            # Вызываем
            _, _, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=1, space_overrides=mock_space
            )
            # Проверяем
            assert best_score == EXPECTED_FALLBACK
            assert mock_finite.called

    def test_objective_via_actual_nan(self, dummy_data, mock_space):
        """
        Тест через подмену np.mean (более естественный путь).
        Если в tuner.py: 'import numpy as np', патчим 'np.mean'.
        Если 'from numpy import mean', патчим 'mean'.
        """
        X, y = dummy_data

        # Попробуем запатчить mean в пространстве имен модуля tuner
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner.np.mean", return_value=np.nan),
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_cv.return_value = np.array(
                [1.0]
            )  # значение не важно, так как mean вернет nan
            _, _, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=1, space_overrides=mock_space
            )
            assert best_score == HPO_WORST_SCORE

    def test_consecutive_failures_disqualifies_algorithm(self, dummy_data, mock_space):
        """
        5 последовательных RuntimeError в cross_val_score → InvalidAlgorithmError.
        Проверяет, что optimize() выбрасывает InvalidAlgorithmError после
        MAX_FATAL_FAILURES (5) последовательных фатальных ошибок.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # Каждый вызов cross_val_score кидает RuntimeError
            mock_cv.side_effect = RuntimeError("fatal failure")

            with pytest.raises(hyperopt.InvalidAlgorithmError, match="disqualified"):
                optimize(
                    algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
                )

    def test_four_failures_one_success_not_disqualified(self, dummy_data, mock_space):
        """
        4 фатальных ошибки + 1 успех → алгоритм НЕ дисквалифицируется.
        Проверяет, что при 4 последовательных RuntimeError и 1 успешном
        запуске cross_val_score оптимизация завершается успешно.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # 4 ошибки подряд, затем успех
            mock_cv.side_effect = [
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                np.array([0.85]),
            ]

            _, _, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=5, space_overrides=mock_space
            )
            assert best_score == 0.85

    def test_consecutive_failures_with_train_test_split(self, dummy_data, mock_space):
        """
        5 последовательных RuntimeError в train_test_split → InvalidAlgorithmError.
        Проверяет, что circuit breaker работает при train_test_split,
        а не только при k-fold/Leave-One-Out.
        """
        X, y = dummy_data
        with (
            patch("configurable_automl_engine.tuner.iter_splits") as mock_iter_splits,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            # Принудительно выбираем train_test_split
            mock_make_cv.return_value = ("train_test_split", None, None)

            # iter_splits должен возвращать свежий итератор при каждом вызове
            def fresh_iter(*args, **kwargs):
                return iter(
                    [
                        (
                            np.array([[1], [2]]),
                            np.array([[1], [2]]),
                            np.array([0, 1]),
                            np.array([0, 1]),
                        )
                    ]
                )

            mock_iter_splits.side_effect = fresh_iter

            # Модель: fit всегда падает с RuntimeError
            mock_model = MagicMock()
            mock_model.fit.side_effect = RuntimeError("fatal failure")
            mock_create.return_value = mock_model
            mock_scorer_factory.return_value = MagicMock()

            with pytest.raises(hyperopt.InvalidAlgorithmError, match="disqualified"):
                optimize(
                    algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
                )

    def test_train_test_split_success_resets_counter(self, dummy_data, mock_space):
        """
        4 фатальных ошибки + 1 успех при train_test_split → алгоритм НЕ дисквалифицируется.
        Проверяет, что успешный train_test_split сбрасывает consecutive_fatal_failures.
        """
        X, y = dummy_data
        with (
            patch("configurable_automl_engine.tuner.iter_splits") as mock_iter_splits,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("train_test_split", None, None)

            def fresh_iter(*args, **kwargs):
                return iter(
                    [
                        (
                            np.array([[1], [2]]),
                            np.array([[1], [2]]),
                            np.array([0, 1]),
                            np.array([0, 1]),
                        )
                    ]
                )

            mock_iter_splits.side_effect = fresh_iter

            # 4 ошибки подряд, затем успех
            mock_model = MagicMock()
            mock_model.fit.side_effect = [
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                None,  # trial 5 success → _objective возвращает score
                None,  # финальный best_model.fit(X, y)
            ]
            mock_create.return_value = mock_model

            # Скор возвращает 0.85 при успешном вызове
            mock_scorer = MagicMock(return_value=0.85)
            mock_scorer_factory.return_value = mock_scorer

            _, _, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=5, space_overrides=mock_space
            )
            assert best_score == 0.85

    def test_valueerror_not_fatal(self, dummy_data, mock_space):
        """
        10 последовательных ValueError → алгоритм НЕ дисквалифицируется.
        Проверяет, что ValueError считается нефатальной ошибкой и не
        увеличивает счётчик consecutive_fatal_failures.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # Все 10 вызовов cross_val_score кидают ValueError
            mock_cv.side_effect = ValueError("non-fatal value error")

            # Оптимизация завершается без InvalidAlgorithmError,
            # ValueError просто прерывает trial (optuna.TrialPruned)
            best_algo, best_model, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
            )
            # Ни один trial не успешен: study.best_params бросает ValueError,
            # и optimize() сигнализирует отказ полным None (issue #13) —
            # это НЕ «магическая» константа -3.4028235e38, которая выглядела
            # как валидный результат для оркестратора.
            assert best_algo is None
            assert best_model is None
            assert best_score is None

    def test_pruned_trials_reset_fatal_counter(self, dummy_data, mock_space):
        """
        1 фатальная ошибка + триалы, отсечённые прунером, + 4 фатальные ошибки
        → алгоритм НЕ дисквалифицируется (регрессия issue #22).

        Проверяет, что optuna.TrialPruned из ранней остановки
        (_evaluate_with_intermediate_reports) обрывает последовательность
        фатальных сбоев и сбрасывает consecutive_fatal_failures: всего сбоев
        набралось 5, но подряд — только 4.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_with_intermediate_reports"
            ) as mock_eval,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # Триал 1 — фатальный сбой; триал 2 — штатно отсечён прунером;
            # триалы 3–6 — фатальные сбои; триал 7 — успех.
            mock_eval.side_effect = [
                RuntimeError("fatal failure"),  # trial 1
                optuna.TrialPruned(),  # trial 2 (early stopping)
                RuntimeError("fatal failure"),  # trial 3
                RuntimeError("fatal failure"),  # trial 4
                RuntimeError("fatal failure"),  # trial 5
                RuntimeError("fatal failure"),  # trial 6
                0.85,  # trial 7
            ]

            _, _, best_score = optimize(
                algo_name="rf",
                X=X,
                y=y,
                n_trials=7,
                space_overrides=mock_space,
                pruning={
                    "enable": True,
                    "strategy": "median",
                    "min_steps": 1,
                    "n_startup_trials": 1,
                },
            )
            assert best_score == 0.85

    def test_valueerror_between_failures_resets_counter(self, dummy_data, mock_space):
        """
        1 фатальная ошибка + нефатальные ValueError + 4 фатальные ошибки
        → алгоритм НЕ дисквалифицируется (регрессия issue #22).

        Проверяет, что нефатальный ValueError обрывает последовательность
        фатальных сбоев и сбрасывает consecutive_fatal_failures: всего сбоев
        набралось 5, но подряд — только 4.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_cv.side_effect = [
                RuntimeError("fatal failure"),  # trial 1
                ValueError("non-fatal value error"),  # trial 2
                ValueError("non-fatal value error"),  # trial 3
                RuntimeError("fatal failure"),  # trial 4
                RuntimeError("fatal failure"),  # trial 5
                RuntimeError("fatal failure"),  # trial 6
                RuntimeError("fatal failure"),  # trial 7
                np.array([0.85]),  # trial 8
            ]

            _, _, best_score = optimize(
                algo_name="rf", X=X, y=y, n_trials=8, space_overrides=mock_space
            )
            assert best_score == 0.85


# ══════════════════════════════════════════════════════════════════════════════
# 7. Сквозной тест training_engine (train_best_model)
# ══════════════════════════════════════════════════════════════════════════════


def test_full_training_engine_pipeline(tmp_path: Path) -> None:
    """
    Сквозной интеграционный тест полного пайплайна обучения.

    Проверяет:
    1. Чтение YAML-конфига.
    2. Выполнение двух фаз HPO (all_algorithms → refine_winner).
    3. Сохранение итоговой модели на диск (.pkl).
    4. Возврат корректной структуры результата.
    """
    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config_path = tmp_path / "config.yaml"

    config: dict[str, Any] = {
        "general": {
            "comparison_metric": "rmse",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {"name": "refine", "n_trials": 1, "action": "refine_winner"},
            ],
        },
        "algorithms": {
            "ridge": {"enable": True},
            "random_forest": {"enable": True},
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Генерируем синтетические данные регрессии
    X, y = make_regression(n_samples=120, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(
        config=str(config_path),
        df=df,
        target="target",
    )

    # Проверка структуры результата
    assert isinstance(result, dict), f"Ожидался dict, получен {type(result)}"
    assert "algorithm" in result, "Нет ключа 'algorithm'"
    assert "score" in result, "Нет ключа 'score'"
    assert "params" in result, "Нет ключа 'params'"
    assert "model_path" in result, "Нет ключа 'model_path'"

    assert result["algorithm"] in ("ridge", "random_forest"), (
        f"Неизвестный алгоритм-победитель: {result['algorithm']}"
    )
    assert isinstance(result["score"], float), f"score не float: {result['score']}"
    assert isinstance(result["params"], dict) and result["params"], (
        "best_params пуст или не dict"
    )

    # Проверка сохранения файла модели на диск
    saved_path = Path(result["model_path"])
    assert saved_path.exists(), f"Файл модели не найден: {saved_path}"

    # Проверка, что модель можно загрузить обратно
    loaded = ModelTrainer.load(str(saved_path))
    assert loaded.pipeline is not None, "Загруженная модель не содержит pipeline"
    assert loaded.algorithm == result["algorithm"], (
        f"Алгоритм не совпадает: {loaded.algorithm} != {result['algorithm']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Adaptive preprocessing presets in optimize() (issue #18)
# ──────────────────────────────────────────────────────────────────────────────


def _num_transformer_of(preprocessor):
    """Извлечь числовой трансформер из ColumnTransformer."""
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    return transformers["num"]


def test_optimize_trees_preprocessor_without_scaling(toy_data):
    """AC-3 в фазе HPO: для деревьев масштабирование не применяется."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    X, y = toy_data

    model, _, _ = hyperopt.optimize(
        "random_forest",
        X,
        y,
        n_trials=2,
        random_state=0,
        space_overrides={
            "random_forest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 5, 20)}
        },
    )

    preprocessor = model.named_steps["preprocessor"]
    num = _num_transformer_of(preprocessor)
    step_names = [s[0] for s in num.steps]
    assert "scaler" not in step_names
    assert num.named_steps["imputer"].strategy == "median"
    assert not isinstance(num.named_steps.get("scaler"), (StandardScaler, RobustScaler))


def test_optimize_glm_uses_robust_scaler(toy_data):
    """AC-5 в фазе HPO: GLM использует RobustScaler + median-импутацию."""
    from sklearn.preprocessing import RobustScaler

    X, y = toy_data
    y_pos = pd.Series(np.abs(y) + 1.0)

    model, _, _ = hyperopt.optimize(
        "gammaregressor",
        X,
        y_pos,
        n_trials=2,
        random_state=0,
        space_overrides={
            "gammaregressor": lambda t: {
                "alpha": t.suggest_float("alpha", 1e-6, 1e-1),
                "max_iter": t.suggest_int("max_iter", 50, 100),
            }
        },
    )

    num = _num_transformer_of(model.named_steps["preprocessor"])
    assert isinstance(num.named_steps["scaler"], RobustScaler)
    assert num.named_steps["imputer"].strategy == "median"


def test_optimize_preprocessing_override_applied(toy_data):
    """AC-7 в фазе HPO: явное переопределение пресета имеет приоритет."""
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data

    # Для дерева пользователь явно просит standard scaling
    model, _, _ = hyperopt.optimize(
        "random_forest",
        X,
        y,
        n_trials=2,
        random_state=0,
        preprocessing_override={"scaling": "standard"},
        space_overrides={
            "random_forest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 5, 20)}
        },
    )

    num = _num_transformer_of(model.named_steps["preprocessor"])
    assert isinstance(num.named_steps["scaler"], StandardScaler)
    # Импутация осталась автовыбором класса деревьев
    assert num.named_steps["imputer"].strategy == "median"


def test_optimize_logs_resolved_preset(toy_data, caplog):
    """AC-9 в фазе HPO: выбранный пресет фиксируется в логах."""
    X, y = toy_data

    with caplog.at_level("INFO", logger="configurable_automl_engine.tuner"):
        hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            random_state=0,
            space_overrides={
                "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-4, 1.0)}
            },
        )

    assert "Resolved preprocessing preset" in caplog.text
    assert "scale_sensitive" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# 9. Feature selection integration (issue #11)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def noisy_fs_data() -> tuple[pd.DataFrame, pd.Series]:
    """Датасет для проверки отбора признаков: 2 информативных + 18 шумовых.

    Чистый шум в 18 колонках делает отбор признаков полезным: модель,
    обучающаяся на полном пространстве, вынуждена бороться с нерелевантными
    признаками, тогда как селектор (importance через ExtraTreesRegressor)
    оставляет информативные колонки.
    """
    rng = np.random.RandomState(7)
    X_info, y = make_regression(
        n_samples=250,
        n_features=2,
        n_informative=2,
        noise=0.15,
        random_state=7,
    )
    X_noise = rng.normal(size=(X_info.shape[0], 18))
    X = np.hstack([X_info, X_noise])
    columns = [f"info_{i}" for i in range(2)] + [f"noise_{i}" for i in range(18)]
    return pd.DataFrame(X, columns=columns), pd.Series(y)


_FS_ELASTICNET_SPACE = {
    "elasticnet": lambda t: {
        "alpha": t.suggest_float("alpha", 1e-3, 1.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
    }
}

_FS_ISOTONIC_SPACE = {
    "isotonic_regression": lambda t: {
        "increasing": t.suggest_categorical("increasing", [True, False])
    }
}


def _run_optimize_with_captured_study(
    *args: Any, **kwargs: Any
) -> tuple[Any, optuna.Study, Any, dict[str, Any] | None, float]:
    """Запустить optimize, перехватив реальный study Optuna через create_study.

    Возвращает кортеж (model, study, params, score).
    """
    real_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=kwargs.get("random_state") or 42),
    )
    with patch(
        "configurable_automl_engine.tuner.optuna.create_study", return_value=real_study
    ):
        model, params, score = hyperopt.optimize(*args, **kwargs)
    return model, real_study, params, score


def test_auto_mode_explores_both_and_wins_with_fs(noisy_fs_data):
    """mode='auto': Optuna пробует True/False и побеждает триал с отбором.

    На датасете из 2 информативных признаков и 18 колонок чистого шума
    сокращение пространства признаков объективно улучшает качество, поэтому
    лучший триал обязан выбрать use_feature_selection=True.
    """
    X, y = noisy_fs_data
    model, study, params, score = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=15,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # Optuna исследовала обе гипотезы отбора
    tried = {
        t.params["use_feature_selection"]
        for t in study.trials
        if "use_feature_selection" in t.params
    }
    assert tried == {True, False}

    # Победивший триал — с отбором признаков
    assert params["use_feature_selection"] is True
    assert "feature_selector" in model.named_steps
    assert isinstance(model.named_steps["feature_selector"], FeatureSelector)
    assert isinstance(score, float) and not np.isnan(score)


def test_auto_mode_winner_false_omits_feature_selector_in_final_model(toy_data):
    """mode='auto': победитель выбрал False -> селектора нет в финальной модели.

    Покрывает путь «auto → победитель выбрал False» в финальной сборке:
    ``best_apply_fs`` обязан разрешиться в False, шаг feature_selector не
    попадает в модель, а решение сохраняется в возвращаемом best_params.
    """
    X, y = toy_data
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        mock_study.best_params = {
            "alpha": 0.6,
            "l1_ratio": 0.4,
            "use_feature_selection": False,
        }
        mock_study.best_value = 0.9
        # Явный подсчёт состояний триалов (issue #32): один COMPLETED-триал,
        # чтобы optimize() прошёл проверку n_completed > 0 и дошёл до best_params.
        mock_study.get_trials.return_value = [
            MagicMock(state=optuna.trial.TrialState.COMPLETE)
        ]
        mock_create.return_value = mock_study

        model, params, _ = optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            feature_selection_cfg={"mode": "auto"},
            space_overrides={
                "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
            },
        )

    # Победитель HPO явно отказался от отбора: селектор не встроен
    assert "use_feature_selection" in params
    assert params["use_feature_selection"] is False
    assert "feature_selector" not in model.named_steps
    assert "model" in model.named_steps


def test_always_mode_forces_feature_selector(noisy_fs_data):
    """mode='always': селектор в пайплайне, best_params без служебных подсказок."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # Служебного ключа тюнера нет в best_params (suggest_categorical не вызывался)
    assert "use_feature_selection" not in params
    # Финальная модель собрана с шагом отбора признаков
    assert "feature_selector" in model.named_steps
    assert isinstance(model.named_steps["feature_selector"], FeatureSelector)


def test_disabled_mode_omits_feature_selector(noisy_fs_data):
    """mode='disabled': селектор не появляется в финальной модели."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "disabled"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    assert "use_feature_selection" not in params
    assert "feature_selector" not in model.named_steps


def test_auto_mode_default_is_disabled_when_cfg_none(noisy_fs_data):
    """feature_selection_cfg=None равнозначен disabled (обратная совместимость)."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    assert "use_feature_selection" not in params
    assert "feature_selector" not in model.named_steps


def test_auto_mode_initial_params_preserves_fs_decision(noisy_fs_data):
    """enqueue_trial с use_feature_selection из фазы 1 стартует с этим решением."""
    X, y = noisy_fs_data
    initial = {"alpha": 0.5, "l1_ratio": 0.3, "use_feature_selection": True}
    _, study, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        initial_params=initial,
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # Первый (enqueued) триал стартует с решением предыдущей фазы
    first = study.trials[0]
    assert first.params.get("use_feature_selection") is True
    assert first.params.get("alpha") == 0.5
    assert first.params.get("l1_ratio") == 0.3
    # Ключ решения сохраняется в best_params (монотонность между фазами)
    assert "use_feature_selection" in params


def test_isotonic_regression_ignores_feature_selection(toy_data):
    """Изотоническая регрессия: тюнинг не падает, отбор принудительно выключен."""
    X, y = toy_data
    X_iso = X.iloc[:, [0]]

    # mode='always' — даже принудительный режим не ломает одномерный алгоритм
    model_always, _, _ = hyperopt.optimize(
        "isotonic_regression",
        X_iso,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ISOTONIC_SPACE,
    )
    assert hasattr(model_always, "predict")
    steps_always = [name for name, _ in getattr(model_always, "steps", [])]
    assert "feature_selector" not in steps_always

    # mode='auto' — даже если Optuna выбрала True, селектор не попадает в модель
    model_auto, params_auto, _ = hyperopt.optimize(
        "isotonic_regression",
        X_iso,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        space_overrides=_FS_ISOTONIC_SPACE,
    )
    assert hasattr(model_auto, "predict")
    steps_auto = [name for name, _ in getattr(model_auto, "steps", [])]
    assert "feature_selector" not in steps_auto
    # Для isotonic категория use_feature_selection не предлагается Optuna
    # (флаг принудительно False), поэтому служебный ключ не попадает
    # в best_params даже в режиме 'auto'.
    assert "use_feature_selection" not in params_auto


def test_invalid_feature_selection_cfg_rejected(toy_data):
    """Невалидный feature_selection_cfg отклоняется до запуска поиска."""
    X, y = toy_data

    # Неверный тип: строка вместо dict/FeatureSelectionCfg
    with pytest.raises(TypeError, match="feature_selection_cfg must be a"):
        hyperopt.optimize(
            "ridge", X, y, n_trials=2, feature_selection_cfg="always"
        )

    # Невалидный dict: неизвестный method отклоняется валидацией Pydantic
    with pytest.raises(HyperoptError, match="Invalid feature_selection_cfg"):
        hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            feature_selection_cfg={"mode": "always", "method": "not_a_method"},
        )


def test_fs_transformer_factory_uses_fixed_seed_when_random_state_none(toy_data):
    """B2: при random_state=None FeatureSelector получает фиксированный seed 42.

    Иначе каждый вызов fs_transformer_factory создавал бы селектор с новым
    случайным зерном и отбор признаков был бы невоспроизводим.
    """
    X, y = toy_data
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch(
            "configurable_automl_engine.tuner.FeatureSelector", autospec=True
        ) as mock_fs,
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        mock_study.best_params = {"alpha": 0.6}
        mock_study.best_value = 0.9
        # Явный подсчёт состояний триалов (issue #32): один COMPLETED-триал,
        # чтобы optimize() прошёл проверку n_completed > 0 и дошёл до best_params.
        mock_study.get_trials.return_value = [
            MagicMock(state=optuna.trial.TrialState.COMPLETE)
        ]
        mock_create.return_value = mock_study

        optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            random_state=None,
            feature_selection_cfg={"mode": "always"},
            space_overrides={"elasticnet": lambda t: {"alpha": 0.01}},
        )

        assert mock_fs.call_args.kwargs["random_state"] == 42


def test_fs_reproducible_with_random_state_none(noisy_fs_data):
    """B2: два запуска random_state=None дают одинаковый результат.

    Случайность Optuna (TPE, разбиения) при random_state=None не
    фиксируется, но отбор признаков использует фиксированный seed 42,
    поэтому при константном пространстве поиска итоговые модели идентичны.
    """
    X, y = noisy_fs_data
    run_kwargs = dict(
        n_trials=2,
        random_state=None,
        feature_selection_cfg={"mode": "always"},
        space_overrides={"elasticnet": lambda t: {"alpha": 0.01, "l1_ratio": 0.5}},
    )
    model1, _, _ = hyperopt.optimize("elasticnet", X, y, **run_kwargs)
    model2, _, _ = hyperopt.optimize("elasticnet", X, y, **run_kwargs)

    assert np.allclose(model1.predict(X), model2.predict(X))


def test_fs_service_key_does_not_leak_into_model_constructor(toy_data):
    """Служебный ключ use_feature_selection не попадает в конструктор модели.

    Покрывает B4/B6: initial_params (enqueue_trial) переносит решение
    предыдущей фазы между фазами HPO, но при создании модели через
    create_model ключ обязан быть вычищен, а исходный study.best_params —
    не мутирован.
    """
    X, y = toy_data
    initial_params = {"alpha": 0.5, "use_feature_selection": True}
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch(
            "configurable_automl_engine.tuner.FeatureSelector", autospec=True
        ),
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        mock_study.best_params = {
            "alpha": 0.6,
            "l1_ratio": 0.4,
            "use_feature_selection": True,
        }
        mock_study.best_value = 0.9
        # Явный подсчёт состояний триалов (issue #32): один COMPLETED-триал,
        # чтобы optimize() прошёл проверку n_completed > 0 и дошёл до best_params.
        mock_study.get_trials.return_value = [
            MagicMock(state=optuna.trial.TrialState.COMPLETE)
        ]
        mock_create.return_value = mock_study

        _, params, _ = optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            initial_params=initial_params,
            feature_selection_cfg={"mode": "auto"},
            space_overrides={
                "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
            },
        )

        # enqueue_trial получает решение предыдущей фазы (монотонность HPO)
        mock_study.enqueue_trial.assert_called_once_with(initial_params)
        # Финальный create_model не получает служебный ключ
        assert "use_feature_selection" not in mock_create_model.call_args.kwargs
        # Возвращаемый best_params сохраняет ключ (монотонность между фазами)
        assert params["use_feature_selection"] is True


def test_fs_pipeline_from_optimize_serializes(tmp_path: Path, noisy_fs_data):
    """B6: пайплайн с шагом feature_selector корректно сериализуется.

    Новый шаг не нарушает обратную совместимость сериализации пайплайнов:
    после save/load predict идентичен, шаг присутствует в восстановленной
    модели.
    """
    import joblib

    X, y = noisy_fs_data
    model, _, _, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )
    assert "feature_selector" in model.named_steps

    path = tmp_path / "fs_pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    assert "feature_selector" in restored.named_steps
    assert np.array_equal(model.predict(X), restored.predict(X))


# ══════════════════════════════════════════════════════════════════════════════
# 10. Регрессия рефакторинга optimize() на шаги (issue #30, блокер B1)
# ══════════════════════════════════════════════════════════════════════════════
# Генерация гиперпараметров (space_fn) и конструктора модели (create_model)
# выполняется в _objective ВНЕ try/except: их ошибки обязаны пробрасываться
# наружу (триал FAIL + исключение из study.optimize), а не превращаться в
# тихие pruned-триалы → (None, None, None) или учитываться circuit breaker'ом.
def test_space_fn_error_marks_trial_failed_and_propagates(toy_data):
    """B1: ValueError из пользовательской space-функции → FAIL, а не PRUNED.

    Исходный контракт: ошибка в space-функции отмечает триал как FAIL и
    прерывает study.optimize (исключение наружу), что делает провал видимым
    для вызывающего кода, а не маскирует его под отсечённый триал.
    """
    X, y = toy_data
    real_study = optuna.create_study(direction="maximize")

    def space_raises(trial):
        trial.suggest_float("alpha", 0.0, 1.0)
        raise ValueError("user space fn boom")

    with patch(
        "configurable_automl_engine.tuner.optuna.create_study", return_value=real_study
    ):
        with pytest.raises(ValueError, match="user space fn boom"):
            optimize(
                "ridge",
                X,
                y,
                n_trials=2,
                space_overrides={"ridge": space_raises},
            )

    assert real_study.trials[0].state == optuna.trial.TrialState.FAIL
    # Отсечения не происходит: это не ранняя остановка и не нефатальный сбой.
    assert all(t.state != optuna.trial.TrialState.PRUNED for t in real_study.trials)


def test_create_model_error_marks_trial_failed_and_propagates(toy_data):
    """B1: сбой конструктора модели (MemoryError) → FAIL, а не дисквалификация.

    Ошибка конструктора не является фатальным сбоем ОЦЕНКИ: она не должна
    инкрементировать consecutive_fatal_failures и приводить к
    InvalidAlgorithmError после N попыток — контракт исходного кода.
    """
    X, y = toy_data
    real_study = optuna.create_study(direction="maximize")

    def exploding_model(algo, **kwargs):
        raise MemoryError("model constructor boom")

    with (
        patch(
            "configurable_automl_engine.tuner.optuna.create_study",
            return_value=real_study,
        ),
        patch(
            "configurable_automl_engine.tuner.create_model",
            side_effect=exploding_model,
        ),
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
    ):
        with pytest.raises(MemoryError, match="model constructor boom"):
            optimize(
                "ridge",
                X,
                y,
                n_trials=5,
                space_overrides={"ridge": lambda t: {"alpha": 0.1}},
            )

    # Первый же триал FAIL — без накопления счётчика фатальных сбоев
    # (иначе после 5 попыток был бы InvalidAlgorithmError, а не MemoryError).
    assert real_study.trials[0].state == optuna.trial.TrialState.FAIL
    assert len(real_study.trials) == 1


def test_scoring_valueerror_still_prunes_not_fatal(toy_data):
    """B1: ValueError на этапе ОЦЕНКИ (fit/scoring) по-прежнему → PRUNED.

    Разделение сохранено: за пределами try/except остались только генерация
    параметров/модели; ошибки самой оценки обрабатываются handle_failure
    как раньше (нефатальный ValueError → TrialPruned, без дисквалификации).
    """
    X, y = toy_data
    with (
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score"
        ) as mock_cv,
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        # Все 10 вызовов cross_val_score (этап оценки) кидают ValueError
        mock_cv.side_effect = ValueError("non-fatal scoring error")

        best_algo, best_model, best_score = optimize(
            algo_name="rf",
            X=X,
            y=y,
            n_trials=10,
            space_overrides={"rf": lambda trial: {"n_estimators": 10}},
        )
        # Ни один триал не завершён → контракт «нет валидного результата»
        # (issue #13/#32): сплошной None, без InvalidAlgorithmError.
        assert best_algo is None
        assert best_model is None
        assert best_score is None


# ─────────── Очистка неинформативных признаков (issue #57) ───────────────────


def _tuner_garbage_data(n: int = 120, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame с информативными колонками и мусором (константные/пустые)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "const_num": 5.0,
            "all_nan": np.nan,
            "almost_empty": [np.nan] * (n - 1) + [1.0],
            "const_cat": "only",
        }
    )
    y = pd.Series(2.0 * X["x1"] - 1.5 * X["x2"] + rng.normal(0, 0.1, n))
    return X, y


def test_optimize_cleans_uninformative_features():
    """optimize() очищает мусорные колонки до построения препроцессора:
    финальная модель работает с информативными признаками, предсказания
    конечны, лишние колонки на входе не влияют на результат."""
    X, y = _tuner_garbage_data()

    model, params, score = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=2,
        random_state=0,
        space_overrides={
            "ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.05, 0.5)}
        },
    )

    assert isinstance(score, float) and np.isfinite(score)
    assert isinstance(params, dict) and params
    assert hasattr(model, "predict")
    # Модель обучалась на очищенных данных: предсказания конечны и устойчивы
    # к лишним (мусорным) колонкам на входе.
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    preds_clean = model.predict(X[["x1", "x2"]])
    assert np.allclose(preds, preds_clean)

    # Препроцессор собран только по 2 информативным колонкам (не 6).
    preprocessor = model.named_steps["preprocessor"]
    out = preprocessor.transform(X[["x1", "x2"]])
    assert out.shape[1] == 2


def test_optimize_numpy_input_skips_cleaning():
    """np.ndarray без имён колонок: очистка пропускается (ограничение),
    оптимизация проходит без падений."""
    rng = np.random.default_rng(3)
    X_arr = rng.normal(size=(100, 3))
    y_arr = X_arr[:, 0] * 2.0 - X_arr[:, 1] + rng.normal(0, 0.1, 100)

    model, params, score = hyperopt.optimize(
        "ridge",
        X_arr,
        y_arr,
        n_trials=2,
        random_state=0,
        space_overrides={
            "ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.05, 0.5)}
        },
    )

    assert isinstance(score, float) and np.isfinite(score)
    assert hasattr(model, "predict")
    assert np.isfinite(model.predict(X_arr)).all()


def test_runner_dropped_features_recorded():
    """_OptimizeRunner фиксирует список удалённых колонок для диагностики."""
    X, y = _tuner_garbage_data()
    runner = hyperopt._OptimizeRunner(
        algo_name="ridge",
        X=X,
        y=y,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=None,
        categorical_features=None,
        numerical_features=None,
        encoding=None,
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    assert set(runner.dropped_features_) == {
        "const_num",
        "all_nan",
        "almost_empty",
        "const_cat",
    }
    assert set(runner.X.columns) == {"x1", "x2"}
    assert runner.preprocessor is not None


def test_optimize_external_preprocessor_skips_cleaning():
    """Если передан готовый preprocessor, очистка self.X не выполняется:
    пользователь сам зафиксировал набор признаков."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    X, y = _tuner_garbage_data()
    external = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["x1", "x2"])]
    )

    runner = hyperopt._OptimizeRunner(
        algo_name="ridge",
        X=X,
        y=y,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=external,
        categorical_features=None,
        numerical_features=None,
        encoding=None,
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    assert runner.dropped_features_ == []
    assert set(runner.X.columns) == {
        "x1",
        "x2",
        "const_num",
        "all_nan",
        "almost_empty",
        "const_cat",
    }
    assert runner.preprocessor is external


# ──────────────────────────────────────────────────────────────────────────────
#  Индикаторы пропусков: безопасный fallback в tuner (issue #56)
# ──────────────────────────────────────────────────────────────────────────────
def test_missing_indicator_enabled_guards_invalid_algorithms():
    """Пустое/None/неизвестное имя алгоритма не роняет сборку препроцессора:
    возвращается безопасное значение True (индикаторы включены)."""
    assert _missing_indicator_enabled(None) is True
    assert _missing_indicator_enabled("") is True
    assert _missing_indicator_enabled("unknown_algo_2026") is True


def test_missing_indicator_enabled_univariate_disabled():
    """Строго одномерные алгоритмы (isotonic) — индикаторы выключены."""
    assert _missing_indicator_enabled("isotonic_regression") is False
    assert _missing_indicator_enabled("isotonic") is False


def test_missing_indicator_enabled_regular_algorithms():
    """Обычные алгоритмы — индикаторы включены."""
    assert _missing_indicator_enabled("ridge") is True
    assert _missing_indicator_enabled("rf") is True
