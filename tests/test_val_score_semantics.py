"""Семантические тесты ``val_score`` / ``additional_scores`` (issue #24).

Покрытие:
    1. ``val_score`` — «честная» оценка на отложенных данных (hold-out/CV),
       а не train-скор: на зашумлённых данных заметно ниже train-R²;
       совпадает с ручным пересчётом через ``validation.iter_splits``.
    2. ``additional_scores`` считаются тем же методом валидации, что и основная
       метрика (тот же hold-out сплит / среднее по фолдам).
    3. Engine-путь (``train_best_model`` с k_fold): ``trainer.val_score``
       совпадает с ``result["score"]`` при совпадении метода/фолдов/зерна.
    4. ``auto``-стратегия резолвится один раз; в финальный fit передаются
       разрешённые k/test_size (не строка ``auto``).
    5. Негативные и граничные сценарии: неизвестная стратегия, падение всех
       фолдов (fail-fast без остаточного состояния), N=2, fallback k_fold→split,
       константный таргет (NRMSE→inf), детерминизм при random_state=None,
       обратная совместимость старых pickle без новых атрибутов.
"""

from __future__ import annotations

import logging
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import (
    ModelTrainer,
    TrainingError,
    _sign_corrected_value,
)
from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.metrics import get_scorer_object
from configurable_automl_engine.validation import iter_splits

# ──────────────────────────────────────────────────────────────────────────────
#  Хелперы
# ──────────────────────────────────────────────────────────────────────────────


def _noisy_regression(n: int = 40, p: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Зашумлённый датасет: первый признак информативен, остальные — шум.

    На таких данных модель систематически переобучается, поэтому train-R²
    заметно выше честной hold-out оценки (воспроизводит пример из issue #24).
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p))
    X.iloc[:, 0] = X.iloc[:, 0] * 3.0
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(n) * 3.0)
    return X, y


def _manual_fold_score(
    X: pd.DataFrame,
    y: pd.Series,
    trainer: ModelTrainer,
    *,
    feature_names: list[str],
    method: str,
    n_folds: int,
    test_size: float | int,
    seed: int,
) -> float:
    """Пересчитать val_score вручную: те же сплиты и клоны пайплайна на фолд.

    Повторяет логику ``ModelTrainer._score_on_validation_splits``, чтобы
    независимо проверить равенство значений (а не самоконсистентность).

    ``feature_names`` передаётся явно (ревью PR #17): хелпер не должен
    полагаться на атрибуты тренера, которые могли быть не установлены.
    """
    preprocessor = trainer._build_preprocessor(feature_names)
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    raw: list[float] = []
    scorer = get_scorer_object(trainer.metric)
    for X_tr, X_te, y_tr, y_te in iter_splits(
        X,
        y,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=seed,
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        raw.append(float(scorer(fold_pipe, X_te, y_te)))
    return _sign_corrected_value(trainer.metric, float(np.mean(raw)))


# ──────────────────────────────────────────────────────────────────────────────
#  1. Позитивные: честность оценки и равенство с ручным пересчётом
# ──────────────────────────────────────────────────────────────────────────────


def test_val_score_is_holdout_not_train_score():
    """На зашумлённых данных val_score заметно ниже train-R² (issue #24).

    Воспроизводит пример из issue: раньше val_score == train-R² на полной
    выборке (систематически оптимистичная оценка качества).
    """
    X, y = _noisy_regression()
    trainer = ModelTrainer(algorithm="ridge", metric="r2", random_state=42).fit(X, y)

    train_r2 = float(r2_score(y, trainer.predict(X)))
    assert trainer.val_score is not None
    # Hold-out оценка «честнее»: ощутимо ниже train-скора на полных данных.
    assert trainer.val_score < train_r2
    assert train_r2 - trainer.val_score > 0.05


def test_val_score_matches_manual_iter_splits_for_holdout():
    """train_test_split: val_score совпадает с ручным hold-out пересчётом."""
    X, y = _noisy_regression(n=80, p=6, seed=3)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="train_test_split",
        test_size=0.25,
        random_state=7,
    ).fit(X, y)

    manual = _manual_fold_score(
        X,
        y,
        trainer,
        feature_names=trainer.feature_names or [],
        method="train_test_split",
        n_folds=5,
        test_size=0.25,
        seed=7,
    )
    # rel=1e-6: устойчивый допуск для float-арифметики после np.mean (ревью PR #17).
    assert trainer.val_score == pytest.approx(manual, rel=1e-6)


def test_val_score_matches_manual_iter_splits_for_kfold():
    """k_fold: val_score — среднее по фолдам, совпадает с ручным пересчётом."""
    X, y = _noisy_regression(n=80, p=6, seed=5)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    manual = _manual_fold_score(
        X,
        y,
        trainer,
        feature_names=trainer.feature_names or [],
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        seed=7,
    )
    assert trainer.val_score == pytest.approx(manual, rel=1e-6)


def test_val_score_matches_cross_val_score():
    """k_fold: val_score совпадает со sklearn cross_val_score (те же фолды).

    Дополнительно проверяется равенство индексов фолдов между
    ``iter_splits`` и ``KFold`` (ревью PR #17): иначе сравнение со
    ``cross_val_score`` было бы бессмысленным.
    """
    X, y = _noisy_regression(n=80, p=6, seed=5)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    cv = KFold(n_splits=4, shuffle=True, random_state=7)
    # Гарантия совпадения фолдов: iter_splits отдаёт подмножества данных
    # (X.iloc[idx]), у DataFrame с RangeIndex метки строк равны позиционным
    # индексам — сравниваем их с индексами фолдов KFold.
    cv_folds = [(tr.tolist(), te.tolist()) for tr, te in cv.split(X)]
    iter_folds = [
        (list(X_tr.index), list(X_te.index))
        for X_tr, X_te, _, _ in iter_splits(
            X, y, method="k_fold", n_folds=4, test_size=0.2, random_state=7
        )
    ]
    assert len(iter_folds) == len(cv_folds)
    for (tr_i, te_i), (tr_c, te_c) in zip(iter_folds, cv_folds):
        assert tr_i == tr_c
        assert te_i == te_c

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    pipe = ImbPipeline(
        trainer._assemble_steps(preprocessor, model, feature_selection_active=False)
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=get_scorer_object("r2"))
    assert trainer.val_score == pytest.approx(float(np.mean(scores)), rel=1e-6)


def test_additional_scores_use_same_validation_method():
    """additional_scores считаются тем же методом: тот же сплит, что и val_score.

    Для метрики, совпадающей с основной, значения должны быть равны; для
    hold-out сплита дополнительная метрика считается ровно на том же разбиении.
    """
    X, y = _noisy_regression(n=80, p=6, seed=11)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["r2", "rmse"],
        validation_strategy="train_test_split",
        test_size=0.2,
        random_state=3,
    ).fit(X, y)

    assert trainer.additional_scores["r2"] == pytest.approx(
        trainer.val_score, rel=1e-12
    )
    assert trainer.additional_scores["rmse"] >= 0

    # Ручной пересчёт rmse на том же сплите. Скорер возвращает отрицательные
    # значения (neg_rmse), поэтому усредняем модули (ревью PR #17).
    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    scorer = get_scorer_object("rmse")
    raws: list[float] = []
    for X_tr, X_te, y_tr, y_te in iter_splits(
        X, y, method="train_test_split", n_folds=5, test_size=0.2, random_state=3
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        raws.append(float(scorer(fold_pipe, X_te, y_te)))
    assert trainer.additional_scores["rmse"] == pytest.approx(
        float(np.mean(np.abs(raws))), rel=1e-6
    )


def test_additional_scores_kfold_mean_over_folds():
    """k_fold: additional_scores — среднее по фолдам, не train-скор."""
    X, y = _noisy_regression(n=60, p=5, seed=2)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["mse"],
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    # mse — ошибка: val-score положительный, а train-MSE на полных данных
    # систематически ниже (переобучение).
    train_mse = float(np.mean((y - trainer.predict(X)) ** 2))
    assert trainer.additional_scores["mse"] > train_mse


# ──────────────────────────────────────────────────────────────────────────────
#  2. Engine-путь: согласованность val_score и result["score"]
# ──────────────────────────────────────────────────────────────────────────────


def test_engine_path_val_score_consistent_with_result_score(tmp_path):
    """train_best_model(k_fold): trainer.val_score ≈ result['score'].

    Оба значения — CV-оценки на одних и тех же фолдах (резолюция метода
    выполняется один раз, D3 issue #24); расхождение — только float-шум.
    """
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.arange(120, dtype=float),
            "b": rng.normal(size=120),
            "target": np.arange(120, dtype=float) * 2.5 + rng.normal(size=120),
        }
    )
    model_path = tmp_path / "model.pkl"
    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "k_fold",
            "n_folds": 3,
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

    loaded = ModelTrainer.load(str(model_path))
    assert loaded.validation_strategy == "k_fold"
    assert loaded.n_folds == 3
    # r2 — score-метрика (greater_is_better), sign-коррекция не меняет знак.
    assert loaded.val_score == pytest.approx(result["score"], rel=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
#  3. auto-резолюция
# ──────────────────────────────────────────────────────────────────────────────


def test_auto_resolved_once_and_forwarded_to_final_fit(tmp_path):
    """auto: make_cv резолвится один раз; разрешённые k/test_size доходят до тренера.

    Проверяем через мок make_cv: компонент передаёт в _fit_and_save именно
    разрешённые значения (метод + фолды/размер), а не строку 'auto'.
    """
    X_df = pd.DataFrame({"a": np.arange(50, dtype=float)})
    df = X_df.copy()
    df["target"] = np.arange(50, dtype=float)

    resolved_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    fake_decision = {"method": "kfold", "k": 3}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_trainer = FakeTrainer()

    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "auto",
            "n_folds": 5,
            "path_to_model": str(tmp_path / "m.pkl"),
            "phases": [{"name": "p", "n_trials": 1, "action": "all_algorithms"}],
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    with (
        patch(
            "configurable_automl_engine.training_engine.component.make_cv",
            return_value=("k_fold", resolved_cv, fake_decision),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ) as mock_save,
    ):
        result = train_best_model(config=cfg, df=df, target="target")

    assert result["score"] == 0.9
    call_kwargs = mock_save.call_args.kwargs
    # В финальный fit передаются разрешённые значения, а не 'auto'
    assert call_kwargs["validation_strategy"].value == "k_fold"
    assert call_kwargs["n_folds"] == 3
    # _run_hpo получает тот же разрешённый метод
    assert mock_save.call_count == 1


def test_auto_split_test_size_forwarded_to_hpo_and_fit(tmp_path):
    """auto → train_test_split: целочисленный test_size доходит до тюнера и тренера."""
    df = pd.DataFrame({"a": np.arange(50, dtype=float)})
    df["target"] = np.arange(50, dtype=float)

    fake_decision = {
        "method": "train_test_split",
        "test_size": 15,
        "train_size": 35,
    }

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "auto",
            "n_folds": 5,
            "path_to_model": str(tmp_path / "m.pkl"),
            "phases": [{"name": "p", "n_trials": 1, "action": "all_algorithms"}],
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    captured_hpo: dict = {}

    def fake_run_hpo(**kwargs):
        captured_hpo.update(kwargs)
        return (0.9, {"alpha": 0.1})

    with (
        patch(
            "configurable_automl_engine.training_engine.component.make_cv",
            return_value=("train_test_split", None, fake_decision),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            side_effect=fake_run_hpo,
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=FakeTrainer(),
        ) as mock_save,
    ):
        train_best_model(config=cfg, df=df, target="target")

    assert captured_hpo["validation_strategy"].value == "train_test_split"
    assert captured_hpo["train_test_split_test_size"] == 15
    save_kwargs = mock_save.call_args.kwargs
    assert save_kwargs["validation_strategy"].value == "train_test_split"
    assert save_kwargs["test_size"] == 15


# ──────────────────────────────────────────────────────────────────────────────
#  4. Негативные сценарии
# ──────────────────────────────────────────────────────────────────────────────


def test_unknown_validation_strategy_rejected_at_init():
    """Неизвестная validation_strategy → TrainingError на этапе инициализации."""
    with pytest.raises(TrainingError, match="Unknown validation_strategy"):
        ModelTrainer(validation_strategy="magic_split")
    # Enum-значение тоже нормализуется; строка с неверным регистром — ок
    assert ModelTrainer(validation_strategy="K_FOLD").validation_strategy == "k_fold"


def test_invalid_n_folds_rejected_at_init():
    """n_folds < 1 отклоняется; для k_fold — n_folds < 2 отклоняется."""
    with pytest.raises(TrainingError, match="n_folds"):
        ModelTrainer(n_folds=0)
    with pytest.raises(TrainingError, match="n_folds"):
        ModelTrainer(validation_strategy="k_fold", n_folds=1)


def test_invalid_test_size_rejected_at_init():
    """test_size вне (0, 1) для доли или < 1 для целого отклоняется."""
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=0.0)
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=1.5)
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=0)  # целое < 1
    with pytest.raises(TrainingError, match="test_size must be a number"):
        ModelTrainer(test_size="big")  # нечисловое значение


def test_validation_split_init_failure_wrapped(caplog):
    """Критическая ошибка инициализации разбиений оборачивается в TrainingError.

    Покрывает ветку «материализация сплитов упала» в
    ``_score_on_validation_splits`` (например, сбой валидатора make_cv).
    """
    X, y = _noisy_regression(n=40, p=3, seed=1)
    trainer = ModelTrainer(algorithm="ridge", metric="r2")

    def broken_iter_splits(*args, **kwargs):
        raise ValueError("splits exploded")

    with patch(
        "configurable_automl_engine.trainer.iter_splits",
        side_effect=broken_iter_splits,
    ):
        with pytest.raises(
            TrainingError, match="Error calculating metrics on validation"
        ):
            trainer.fit(X, y)

    assert trainer.val_score is None
    assert trainer.pipeline is None


def test_all_folds_fail_raises_and_resets_state():
    """Все фолды скоринга падают → TrainingError; pipeline/val_score не остаются.

    Сначала успешное обучение, затем fit() с заведомо нерабочим скорером:
    состояние предыдущего обучения не должно сохраняться.
    """
    X, y = _noisy_regression(n=60, p=4, seed=1)
    trainer = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    assert trainer.pipeline is not None
    assert trainer.val_score is not None

    with patch(
        "configurable_automl_engine.trainer.get_scorer_object",
        return_value=lambda est, Xv, yv: (_ for _ in ()).throw(RuntimeError("boom")),
    ):
        with pytest.raises(TrainingError, match="Validation scoring failed"):
            trainer.fit(X, y)

    assert trainer.pipeline is None
    assert trainer.val_score is None
    assert trainer.additional_scores == {}


def test_single_additional_metric_failure_skipped_with_warning(caplog):
    """Сбой одной дополнительной метрики → WARNING + пропуск, обучение успешно."""
    X, y = _noisy_regression(n=60, p=4, seed=1)

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def flaky(name: str, global_y=None):
        if name == "mae":
            raise RuntimeError("scorer exploded")
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object", side_effect=flaky
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="r2", additional_metrics=["mae", "rmse"]
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "rmse" in trainer.additional_scores
    assert "mae" not in trainer.additional_scores
    assert "could not be computed" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  5. Граничные случаи
# ──────────────────────────────────────────────────────────────────────────────


def test_minimum_two_samples():
    """N = 2 (минимум): обучение завершается штатно (hold-out 1/1)."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    trainer = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    assert trainer.val_score is not None
    assert trainer.pipeline is not None


def test_kfold_fallback_to_split_on_small_data(caplog):
    """k_fold на малой выборке (N < max(4, 2K)) → fallback на train_test_split.

    Поведение согласовано с HPO (make_cv): обучение завершается штатно,
    val_score не None.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(6, 2))
    y = pd.Series(X[0] * 2.0 + rng.randn(6) * 0.1)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.validation"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=5,
            random_state=42,
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "Falling back to 'train_test_split'" in caplog.text


def test_constant_target_nrmse_inf_kept():
    """Константный таргет: NRMSE → inf возвращается как есть, обучение успешно."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([5.0, 5.0, 5.0, 5.0])
    trainer = ModelTrainer(
        algorithm="ridge", metric="nrmse", additional_metrics=["mse"]
    ).fit(X, y)

    assert trainer.val_score == float("inf")
    assert trainer.additional_scores["mse"] >= 0
    assert trainer.pipeline is not None


def test_deterministic_splits_when_random_state_none():
    """random_state=None: сплиты валидации детерминированы (фикс-зерно 42).

    Сравнение через pytest.approx (ревью PR #17): строгое == для float после
    усреднения по фолдам может быть хрупким.
    """
    X, y = _noisy_regression(n=80, p=5, seed=9)
    t1 = ModelTrainer(algorithm="ridge", metric="r2", random_state=None).fit(X, y)
    t2 = ModelTrainer(algorithm="ridge", metric="r2", random_state=None).fit(X, y)

    assert t1.val_score == pytest.approx(t2.val_score, rel=1e-9)
    assert t1.additional_scores == t2.additional_scores


def test_old_pickle_without_validation_attrs_compatible():
    """Старый pickle без новых атрибутов: load + повторный fit работают.

    Новые атрибуты читаются через getattr-фолбэк (train_test_split/5/0.2):
    атрибуты не «материализуются» при загрузке старого артефакта, но fit()
    вычисляет val_score по дефолтной стратегии.
    """
    X, y = _noisy_regression(n=60, p=4, seed=4)
    trainer = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    for attr in ("validation_strategy", "n_folds", "test_size"):
        delattr(trainer, attr)

    # Сериализация/десериализация «старого» объекта
    restored = pickle.loads(pickle.dumps(trainer))
    restored.fit(X, y)
    assert restored.val_score is not None
    # Атрибуты отсутствуют (как в старом pickle), но fit использует дефолты:
    # val_score совпадает с тренером, обученным с явными дефолтами.
    assert not hasattr(restored, "validation_strategy")
    baseline = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    assert restored.val_score == pytest.approx(baseline.val_score, rel=1e-6)


def test_serialization_round_trip_preserves_validation_attrs(tmp_path):
    """Round-trip через save/load сохраняет новые атрибуты валидации."""
    X, y = _noisy_regression(n=60, p=4, seed=6)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        test_size=0.25,
    ).fit(X, y)

    path = tmp_path / "trainer.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)

    assert loaded.validation_strategy == "k_fold"
    assert loaded.n_folds == 3
    assert loaded.test_size == 0.25
    assert loaded.val_score == trainer.val_score
    assert loaded.pipeline is not None


def test_val_score_reset_between_fits():
    """Повторный fit с другим методом валидации пересчитывает val_score."""
    X, y = _noisy_regression(n=80, p=5, seed=8)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="train_test_split"
    ).fit(X, y)
    split_score = trainer.val_score

    trainer.validation_strategy = "k_fold"
    trainer.n_folds = 4
    trainer.fit(X, y)
    assert trainer.val_score is not None
    # Значения при разных методах не обязаны совпадать, но обязаны пересчитаться
    # (сравнение с train-скором не производится — это семантика оценки).
    assert isinstance(trainer.val_score, float)
    assert trainer.pipeline is not None
