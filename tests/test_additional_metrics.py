"""Tests for the ``additional_metrics`` feature (issue #19).

Покрытие:
    1. Конфигурация: новый опциональный параметр ``general.additional_metrics``,
       дедупликация, исключение основной метрики сравнения, ошибка валидации
       для неизвестной метрики.
    2. ModelTrainer: расчёт дополнительных метрик для обученной модели тем же
       способом, что и основная метрика; нечисловые значения (inf/NaN);
       сбой отдельной метрики не влияет на обучение.
    3. train_best_model: возврат ``additional_metrics`` в результатах,
       обратная совместимость, отсутствие влияния на выбор победителя.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from configurable_automl_engine.trainer import ModelTrainer, TrainingError
from configurable_automl_engine.training_engine.component import (
    _fit_and_save,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    GeneralCfg,
)

# ──────────────────────────────────────────────────────────────────────────────
#  1. Валидация конфигурации
# ──────────────────────────────────────────────────────────────────────────────

BASE = {
    "general": {
        "comparison_metric": "rmse",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": [{"name": "search", "n_trials": 1, "action": "all_algorithms"}],
    },
    "algorithms": {"elasticnet": {"enable": True}},
}


def _with_additional(metrics: list[str]) -> dict:
    return {
        **BASE,
        "general": {**BASE["general"], "additional_metrics": metrics},
    }


def test_additional_metrics_default_empty():
    """Параметр не задан — список пуст (обратная совместимость)."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.additional_metrics == []


def test_additional_metrics_explicit_empty_list():
    """Пустой список допустим и эквивалентен отсутствию параметра."""
    cfg = Config.model_validate(_with_additional([]))
    assert cfg.general.additional_metrics == []


def test_additional_metrics_valid_list_accepted():
    """Корректный список метрик из поддерживаемого набора принимается."""
    cfg = Config.model_validate(_with_additional(["r2", "mae", "mse"]))
    assert cfg.general.additional_metrics == ["r2", "mae", "mse"]


def test_additional_metrics_unknown_metric_rejected():
    """Неизвестная метрика → ошибка валидации до запуска обучения."""
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_with_additional(["r2", "bogus_metric"]))
    msg = str(exc_info.value)
    assert "additional_metrics" in msg
    assert "bogus_metric" in msg
    # Сообщение перечисляет допустимые значения
    for allowed in ("nrmse", "rmse", "mae", "mse", "r2"):
        assert allowed in msg


def test_additional_metrics_duplicates_deduplicated():
    """Дубликаты в списке считаются один раз (порядок первого вхождения)."""
    cfg = Config.model_validate(_with_additional(["r2", "mae", "r2", "mae"]))
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_comparison_metric_excluded():
    """Совпадение с основной метрикой сравнения исключается из результатов."""
    cfg = Config.model_validate(_with_additional(["r2", "rmse", "mae"]))
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_alias_of_comparison_excluded(caplog):
    """Алиас основной метрики (rmse ↔ neg_root_mean_squared_error) исключается."""
    data = {
        **BASE,
        "general": {
            **BASE["general"],
            "additional_metrics": ["neg_root_mean_squared_error", "mae"],
        },
    }
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["mae"]
    assert "comparison metric" in caplog.text


def test_additional_metrics_all_filtered_to_empty():
    """Если все метрики совпадают с основной — результат пуст (без дублей)."""
    cfg = Config.model_validate(_with_additional(["rmse"]))
    assert cfg.general.additional_metrics == []


def test_additional_metrics_non_string_element_rejected():
    """Нестроковый элемент даёт ValidationError, а не AttributeError.

    Валидатор ``_deduplicate_additional_metrics`` типизирован как
    ``list[ComparisonMetric]``: элементы списка обязаны быть строками из
    допустимого набора, и проверка типа выполняется до вызова ``.lower()``.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_with_additional(["r2", 123]))
    msg = str(exc_info.value)
    assert "additional_metrics" in msg
    assert "AttributeError" not in msg


def test_additional_metrics_tuple_input_accepted():
    """Кортеж метрик (не только список) принимается и приводится к списку."""
    data = {
        **BASE,
        "general": {**BASE["general"], "additional_metrics": ("r2", "mae")},
    }
    cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_post_validation_list_may_differ(caplog):
    """Итоговый список после валидации отличается от введённого пользователем.

    Задокументированное поведение: дедупликация и исключение основной метрики
    сравнения выполняются на этапе валидации конфигурации, поэтому
    ``cfg.general.additional_metrics`` может не совпадать с исходным списком.
    """
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(
            _with_additional(["rmse", "mae", "mae", "neg_root_mean_squared_error"])
        )
    # Основная метрика ('rmse' и её алиас) исключена, дубликат 'mae' устранён
    assert cfg.general.additional_metrics == ["mae"]
    assert len(caplog.text) > 0
    assert "comparison metric" in caplog.text


def test_additional_metrics_works_with_default_comparison():
    """Дефолтная comparison_metric='r2' также исключается из additional_metrics."""
    data = {
        "general": {
            "phases": [{"name": "p", "n_trials": 1}],
            "additional_metrics": ["r2", "mae"],
        },
        "algorithms": {"elasticnet": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["mae"]


def test_general_cfg_additional_metrics_field():
    """Поле существует в GeneralCfg и по умолчанию пусто."""
    cfg = GeneralCfg(phases=[])
    assert cfg.additional_metrics == []


# ──────────────────────────────────────────────────────────────────────────────
#  2. ModelTrainer: расчёт дополнительных метрик
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {"a": np.arange(50, dtype=float), "b": np.arange(50, dtype=float) * 2}
    )
    y = pd.Series(X["a"] * 1.5 + X["b"] * -0.5 + 1.0)
    return X, y


def test_trainer_computes_additional_scores(simple_regression_data):
    """Дополнительные метрики считаются для финальной модели."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["rmse", "mae", "mse"],
    )
    trainer.fit(X, y)

    assert set(trainer.additional_scores.keys()) == {"rmse", "mae", "mse"}
    # Метрики-ошибки возвращаются положительными «честными» значениями
    assert trainer.additional_scores["rmse"] >= 0
    assert trainer.additional_scores["mae"] >= 0
    assert trainer.additional_scores["mse"] >= 0
    assert all(isinstance(v, float) for v in trainer.additional_scores.values())


def test_trainer_additional_scores_match_main_metric_when_same(
    simple_regression_data,
):
    """Для одной и той же метрики значение совпадает с основной (r2)."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", metric="r2", additional_metrics=["r2"])
    trainer.fit(X, y)
    assert trainer.additional_scores["r2"] == trainer.val_score


def test_trainer_error_metric_additional_values(simple_regression_data):
    """Ошибки (rmse) в дополнительных метриках возвращаются без знака, как val_score."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge", metric="rmse", additional_metrics=["r2", "nrmse"]
    )
    trainer.fit(X, y)
    assert trainer.val_score >= 0
    assert trainer.additional_scores["r2"] <= 1.0
    assert trainer.additional_scores["nrmse"] >= 0


def test_trainer_non_finite_additional_value_kept():
    """inf (например, NRMSE при константном таргете) возвращается как есть."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([5.0, 5.0, 5.0])
    trainer = ModelTrainer(
        algorithm="ridge", metric="mse", additional_metrics=["nrmse"]
    )
    trainer.fit(X, y)
    assert trainer.additional_scores["nrmse"] == float("inf")
    assert trainer.val_score is not None  # обучение завершилось штатно


def test_trainer_additional_metric_failure_skipped_with_warning(caplog):
    """Сбой одной дополнительной метрики не прерывает обучение."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def flaky_scorer_factory(name: str, global_y=None):
        if name == "mae":
            raise RuntimeError("scorer exploded")
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            side_effect=flaky_scorer_factory,
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="r2", additional_metrics=["mae", "rmse"]
        )
        trainer.fit(X, y)

    assert trainer.val_score is not None
    assert "rmse" in trainer.additional_scores  # успешная метрика посчитана
    assert "mae" not in trainer.additional_scores  # упавшая пропущена
    assert "could not be computed" in caplog.text


def test_trainer_additional_metric_none_skipped_with_warning(caplog):
    """Скорер, вернувший None для дополнительной метрики, пропускается с логом."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def none_scorer_factory(name: str, global_y=None):
        if name == "r2":
            return lambda est, Xv, yv: None
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            side_effect=none_scorer_factory,
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="mse", additional_metrics=["r2", "rmse"]
        )
        trainer.fit(X, y)

    assert trainer.val_score is not None
    assert "r2" not in trainer.additional_scores
    assert "rmse" in trainer.additional_scores
    assert "Scorer returned None for additional metric 'r2'" in caplog.text


def test_trainer_no_additional_metrics_by_default(simple_regression_data):
    """Без параметра additional_scores пуст, поведение не меняется."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", metric="r2")
    trainer.fit(X, y)
    assert trainer.additional_scores == {}
    assert trainer.val_score is not None


@pytest.mark.parametrize(
    "bad_value",
    ["r2", b"r2", {"r2": 1}, [1, 2], [None], ("r2", 1), {1, 2}, [1.5]],
)
def test_trainer_invalid_additional_metrics_type_rejected(bad_value):
    """Некорректный тип additional_metrics отклоняется в конструкторе.

    Отклоняются строки-скаляры (символьный разбор недопустим), отображения
    (итерация по ключам неявна) и любые итерируемые объекты, содержащие
    элементы нестрокового типа.
    """
    with pytest.raises(
        TrainingError, match="additional_metrics must be an iterable of strings"
    ):
        ModelTrainer(additional_metrics=bad_value)  # type: ignore[arg-type]


def test_trainer_additional_metrics_accepts_tuple(simple_regression_data):
    """Кортеж строк принимается и нормализуется так же, как список."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics=("R2", "mae"))
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


def test_trainer_additional_metrics_accepts_set(simple_regression_data):
    """Множество строк принимается (порядок не имеет значения)."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics={"rmse", "mse"})
    assert sorted(trainer.additional_metrics) == ["mse", "rmse"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"rmse", "mse"}


def test_trainer_additional_metrics_accepts_generator(simple_regression_data):
    """Генератор строк принимается и материализуется в список."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge",
        additional_metrics=(m for m in ["r2", "mae"]),
    )
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


def test_trainer_additional_metrics_accepts_empty_tuple():
    """Пустой кортеж эквивалентен отсутствию метрик."""
    trainer = ModelTrainer(additional_metrics=())
    assert trainer.additional_metrics == []
    assert trainer.additional_scores == {}


def test_trainer_additional_metrics_lowercased(simple_regression_data):
    """Имена дополнительных метрик нормализуются к нижнему регистру."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics=["R2", "MAE"])
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


# ──────────────────────────────────────────────────────────────────────────────
#  3. train_best_model / _fit_and_save
# ──────────────────────────────────────────────────────────────────────────────


def _component_config(additional: list[str] | None = None) -> dict:
    general: dict = {
        "comparison_metric": "rmse",
        "validation_strategy": "train_test_split",
        "phases": [{"name": "p1", "n_trials": 1, "action": "all_algorithms"}],
        "path_to_model": "model.pkl",
    }
    if additional is not None:
        general["additional_metrics"] = additional
    return {
        "general": general,
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "tuner": "unittest.mock",
                "trainer_module": "unittest.mock",
            }
        },
        "oversampling": {"enable": False},
    }


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 0.0, 1.0]})


def test_train_best_model_returns_additional_metrics(tiny_df):
    """Результат обучения дополняется значениями дополнительных метрик."""
    fake_trainer = SimpleNamespace(additional_scores={"r2": 0.87, "mae": 0.12})
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ) as mock_save,
    ):
        result = train_best_model(
            config=_component_config(additional=["r2", "mae"]),
            df=tiny_df,
            target="target",
        )

    assert result["additional_metrics"] == {"r2": 0.87, "mae": 0.12}
    assert result["score"] == 0.9
    mock_save.assert_called_once()
    # Дополнительные метрики проброшены из конфига в финальное обучение
    cfg_passed = mock_save.call_args.args[6]
    assert cfg_passed.general.additional_metrics == ["r2", "mae"]


def test_train_best_model_no_additional_key_when_not_configured(tiny_df):
    """Без параметра результаты идентичны текущему поведению (без ключа)."""
    fake_trainer = SimpleNamespace(additional_scores={})
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ),
    ):
        result = train_best_model(
            config=_component_config(), df=tiny_df, target="target"
        )

    assert "additional_metrics" not in result
    assert set(result.keys()) == {"algorithm", "score", "params", "model_path"}


def test_train_best_model_no_key_when_additional_empty(tiny_df):
    """Пустой список дополнительных метрик — поведение как сейчас."""
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={}),
        ),
    ):
        result = train_best_model(
            config=_component_config(additional=[]), df=tiny_df, target="target"
        )
    assert "additional_metrics" not in result


def test_train_best_model_comparison_only_not_duplicated(tiny_df):
    """Метрика, совпадающая с основной, в results не дублируется."""
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={}),
        ),
    ):
        result = train_best_model(
            config=_component_config(additional=["rmse"]), df=tiny_df, target="target"
        )
    # Значение rmse присутствует ровно один раз — как score
    assert result["score"] == 0.9
    assert "additional_metrics" not in result


def test_additional_metrics_do_not_affect_winner_selection(tiny_df):
    """Победитель определяется только по основной метрике сравнения."""
    cfg = _component_config(additional=["r2", "mae"])
    cfg["algorithms"] = {
        "elasticnet": {
            "enable": True,
            "tuner": "unittest.mock",
            "trainer_module": "unittest.mock",
        },
        "ridge": {
            "enable": True,
            "tuner": "unittest.mock",
            "trainer_module": "unittest.mock",
        },
    }

    def hpo_side_effect(**kwargs):
        # ridge даёт лучший score — он и должен победить
        if kwargs["algo_name"] == "ridge":
            return 0.95, {"alpha": 0.5}
        return 0.60, {"alpha": 0.1}

    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            side_effect=hpo_side_effect,
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={"r2": 0.9, "mae": 0.1}),
        ),
    ):
        result = train_best_model(config=cfg, df=tiny_df, target="target")

    assert result["algorithm"] == "ridge"
    assert result["score"] == 0.95
    assert result["additional_metrics"] == {"r2": 0.9, "mae": 0.1}


def test_fit_and_save_forwards_additional_metrics(tmp_path, tiny_df):
    """_fit_and_save передаёт additional_metrics в ModelTrainer."""

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {"r2": 0.5}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_module = SimpleNamespace(ModelTrainer=FakeTrainer)

    cfg = Config.model_validate(_component_config(additional=["r2", "mae"]))
    algo_cfg = cfg.algorithms.elasticnet

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=fake_module,
    ):
        trainer = _fit_and_save(
            "elasticnet",
            algo_cfg,
            tiny_df.drop(columns="target"),
            tiny_df["target"],
            {},
            tmp_path / "m.pkl",
            cfg,
        )

    assert isinstance(trainer, FakeTrainer)
    assert trainer.kwargs["additional_metrics"] == ["r2", "mae"]


def test_fit_and_save_forwards_empty_additional_metrics(tmp_path, tiny_df):
    """При отсутствии дополнительных метрик передаётся пустой список."""

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_module = SimpleNamespace(ModelTrainer=FakeTrainer)
    cfg = Config.model_validate(_component_config())
    algo_cfg = cfg.algorithms.elasticnet

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=fake_module,
    ):
        trainer = _fit_and_save(
            "elasticnet",
            algo_cfg,
            tiny_df.drop(columns="target"),
            tiny_df["target"],
            {},
            tmp_path / "m.pkl",
            cfg,
        )

    assert trainer.kwargs["additional_metrics"] == []


# ──────────────────────────────────────────────────────────────────────────────
#  4. Сквозной тест с реальным обучением
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_train_best_model_with_additional_metrics(tmp_path):
    """Сквозной сценарий: дополнительные метрики возвращаются для финальной модели."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.arange(80, dtype=float),
            "b": rng.normal(size=80),
            "target": np.arange(80, dtype=float) * 2.5 + rng.normal(size=80),
        }
    )
    model_path = tmp_path / "model.pkl"
    cfg = {
        "general": {
            "comparison_metric": "rmse",
            "additional_metrics": ["r2", "mae"],
            "path_to_model": str(model_path),
            "phases": [{"name": "search", "n_trials": 2, "action": "all_algorithms"}],
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            }
        },
    }

    result = train_best_model(config=cfg, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert model_path.exists()
    assert set(result["additional_metrics"].keys()) == {"r2", "mae"}
    assert "rmse" not in result["additional_metrics"]
    assert all(isinstance(v, float) for v in result["additional_metrics"].values())
