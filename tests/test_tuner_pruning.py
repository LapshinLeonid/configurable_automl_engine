"""
Тесты ранней остановки (pruning) в configurable_automl_engine.tuner.

Покрывают ключевые сценарии фичи:
• AC-1 — публикация промежуточных результатов и отсечение безнадёжных триалов.
• AC-2 — реальный MedianPruner/HyperbandPruner отсекает часть триалов,
  и это фиксируется в логах.
• AC-3 — best_model/best_params/best_score выбираются только по завершённым
  триалам (правило максимизации метрики).
• AC-4 — при выключенном pruning поведение идентично текущему
  (trial.report не вызывается, cross_val_score не меняется).
• AC-5/AC-6 — отсечённые триалы не считаются фатальными сбоями; при
  отсечении всех триалов сохраняется сценарий «нет результатов».
• AC-7 — hold-out (train_test_split) без естественных шагов: прайнер
  не применяется, поведение безопасно.
• AC-9 — в логах фиксируется факт/количество отсечённых триалов.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from configurable_automl_engine import tuner
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    HyperoptError,
    _build_pruner,
    _evaluate_with_intermediate_reports,
    _normalize_pruning_config,
    optimize,
)


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные объекты: детерминированная «модель» и «скорер»
# ──────────────────────────────────────────────────────────────────────────────
class FakeQualityModel(BaseEstimator, RegressorMixin):
    """Модель, чей predict возвращает константу из гиперпараметра ``quality``.

    Позволяет детерминированно управлять качеством триала: оценка каждого
    фолда равна ``quality``, поэтому «плохие» триалы гарантированно хуже
    «хороших» на каждом промежуточном шаге.
    """

    def __init__(self, quality: float = 0.5, **kwargs: object) -> None:
        self.quality = quality

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.quality)


def _quality_scorer_factory(name: str):
    """Фабрика скорера: возвращает среднее предсказание модели."""
    del name
    return lambda est, X, y: float(np.mean(est.predict(X)))


def _make_quality_space(good: float = 0.9, bad: float = 0.05):
    """Пространство поиска: чётные триалы — хорошие, нечётные — заведомо плохие."""

    def space_fn(trial):
        trial.suggest_float("quality", 0.0, 1.0)
        quality = good if trial.number % 2 == 0 else bad
        return {"quality": quality}

    return space_fn


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((120, 5)))
    y = pd.Series(rng.random(120))
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Юнит-тесты помощников
# ──────────────────────────────────────────────────────────────────────────────
def test_normalize_pruning_config_disabled():
    """enable=False или None → конфигурация прайнера отсутствует (AC-4)."""
    assert _normalize_pruning_config(None) is None
    assert _normalize_pruning_config({}) is None
    assert _normalize_pruning_config({"enable": False}) is None


def test_normalize_pruning_config_defaults():
    """Недостающие поля заполняются значениями по умолчанию."""
    cfg = _normalize_pruning_config({"enable": True})
    assert cfg == {
        "strategy": "median",
        "min_steps": 1,
        "n_startup_trials": 5,
        "reduction_factor": 3,
    }


def test_build_pruner_median():
    pruner = _build_pruner({"strategy": "median", "min_steps": 2, "n_startup_trials": 3})
    assert isinstance(pruner, MedianPruner)
    assert pruner._n_warmup_steps == 2
    assert pruner._n_startup_trials == 3


def test_build_pruner_hyperband():
    pruner = _build_pruner(
        {"strategy": "hyperband", "min_steps": 1, "reduction_factor": 4}
    )
    assert isinstance(pruner, HyperbandPruner)
    assert pruner._min_resource == 1
    assert pruner._reduction_factor == 4


def test_build_pruner_unknown_strategy_raises():
    with pytest.raises(HyperoptError, match="Unknown pruning strategy"):
        _build_pruner({"strategy": "quantile", "min_steps": 1})


def test_build_pruner_invalid_min_steps_raises():
    with pytest.raises(HyperoptError, match="min_steps"):
        _build_pruner({"strategy": "median", "min_steps": 0})


def test_build_pruner_invalid_n_startup_trials_raises():
    with pytest.raises(HyperoptError, match="n_startup_trials"):
        _build_pruner({"strategy": "median", "min_steps": 1, "n_startup_trials": 0})


def test_build_pruner_invalid_reduction_factor_raises():
    with pytest.raises(HyperoptError, match="reduction_factor"):
        _build_pruner({"strategy": "hyperband", "min_steps": 1, "reduction_factor": 1})


def test_evaluate_with_intermediate_reports_publishes_and_prunes():
    """Публикуются промежуточные результаты; should_prune=True → TrialPruned (AC-1)."""
    trial = MagicMock()
    trial.number = 0
    trial.report = MagicMock()
    trial.should_prune = MagicMock(side_effect=[False, True])

    model = FakeQualityModel(quality=0.7)
    scorer = lambda est, X, y: float(np.mean(est.predict(X)))

    X = pd.DataFrame(np.random.rand(30, 3))
    y = pd.Series(np.random.rand(30))

    with pytest.raises(optuna.TrialPruned):
        _evaluate_with_intermediate_reports(
            trial,
            model,
            X,
            y,
            method="k_fold",
            n_folds=3,
            test_size=0.2,
            random_state=42,
            scorer=scorer,
        )

    # На каждом фолде публикуется кумулятивное среднее (шаги с 1).
    assert trial.report.call_count == 2
    steps = [call.kwargs["step"] for call in trial.report.call_args_list]
    assert steps == [1, 2]
    values = [call.args[0] for call in trial.report.call_args_list]
    assert all(v == pytest.approx(0.7) for v in values)


def test_evaluate_with_intermediate_reports_returns_mean():
    """Без отсечения возвращается среднее значение по всем фолдам."""
    trial = MagicMock()
    trial.number = 0
    trial.should_prune = MagicMock(return_value=False)

    model = FakeQualityModel(quality=0.7)
    scorer = lambda est, X, y: float(np.mean(est.predict(X)))

    X = pd.DataFrame(np.random.rand(30, 3))
    y = pd.Series(np.random.rand(30))

    score = _evaluate_with_intermediate_reports(
        trial,
        model,
        X,
        y,
        method="k_fold",
        n_folds=3,
        test_size=0.2,
        random_state=42,
        scorer=scorer,
    )
    assert score == pytest.approx(0.7)
    assert trial.report.call_count == 3


# ──────────────────────────────────────────────────────────────────────────────
# Сквозные тесты optimize() с реальными прайнерами Optuna
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def quality_patches():
    """Патчи активны на время теста: детерминированная модель и скорер."""
    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
    ):
        yield


def test_optimize_pruning_median_prunes_bad_trials(
    toy_data, quality_patches, caplog
):
    """MedianPruner отсекает заведомо плохие триалы до завершения всех фолдов (AC-2).

    Чётные триалы хорошие (quality=0.9), нечётные — заведомо плохие (0.05).
    После первого завершённого триала каждый плохой триал отсекается на первом
    же фолде. Лучший результат выбирается только из завершённых триалов (AC-3),
    а факт отсечений фиксируется в логах (AC-9).
    """
    X, y = toy_data

    with caplog.at_level(logging.INFO):
        model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    # Лучший триал завершён полностью и имеет максимальное качество.
    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None

    # Факт и количество отсечений зафиксированы в логах.
    assert "Early stopping: pruned 2 of 4 trials" in caplog.text
    assert "Early stopping enabled: strategy=median" in caplog.text


def test_optimize_pruning_hyperband_prunes_bad_trials(
    toy_data, quality_patches, caplog
):
    """HyperbandPruner также отсекает часть триалов (AC-2, AC-3)."""
    X, y = toy_data

    with caplog.at_level(logging.INFO):
        model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=8,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "hyperband",
                "min_steps": 1,
                "reduction_factor": 2,
            },
        )

    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None
    assert "Early stopping: pruned" in caplog.text
    assert "strategy=hyperband" in caplog.text


def test_optimize_pruning_all_trials_pruned_returns_no_results(
    toy_data, quality_patches, mocker, caplog
):
    """Все триалы отсечены → сценарий «нет результатов» (AC-6, issue #13).

    Раньше optimize() возвращал «магическую» константу -3.4028235e38,
    которая выглядела как валидный результат для оркестратора. Теперь
    полный провал сигнализируется сплошным None.

    Дополнительно (issue #32): решение принимается явным подсчётом состояний
    триалов (n_completed == 0), а не перехватом ValueError от study.best_params;
    счётчики состояний логируются для наблюдаемости.
    """
    X, y = toy_data

    # Принудительно отсекаем каждый триал на первом же шаге.
    mocker.patch.object(
        tuner.optuna.trial.Trial, "should_prune", return_value=True
    )

    with caplog.at_level(logging.INFO):
        model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert model is None
    assert params is None
    assert score is None
    # Лог со счётчиками состояний: 4 отсечено, 0 завершено, 0 упало.
    assert "Trial states for algorithm 'ridge'" in caplog.text
    assert "completed=0, pruned=4, failed=0" in caplog.text
    assert "no completed trials" in caplog.text


def test_optimize_pruning_single_trial_pruned_returns_no_results(
    toy_data, quality_patches, mocker
):
    """B1 (issue #32): n_trials=1 и единственный триал отсечён → (None, None, None).

    Граничный случай «все pruned» при минимально возможном числе триалов.
    """
    X, y = toy_data

    mocker.patch.object(
        tuner.optuna.trial.Trial, "should_prune", return_value=True
    )

    model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=1,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert model is None
    assert params is None
    assert score is None


def _make_study_with_states(
    states: list[optuna.trial.TrialState],
) -> optuna.Study:
    """Собрать реальный study Optuna с триалами в заданных состояниях.

    Опция ``catch`` в ``study.optimize`` не используется тюнером, поэтому
    «живые» FAIL-триалы (исключение из objective) пробрасываются наружу и
    прерывают запуск. Чтобы проверить явный подсчёт состояний (issue #32),
    наполняем настоящий study заранее созданными триалами и подменяем
    ``study.optimize`` на no-op — get_trials() вернёт именно эти состояния.
    """
    study = optuna.create_study(direction="maximize")
    for state in states:
        study.add_trial(
            optuna.trial.create_trial(state=state, params={}, distributions={})
        )
    return study


def test_optimize_all_trials_failed_returns_no_results(
    toy_data, quality_patches, mocker
):
    """N2 (issue #32): все триалы FAILED → (None, None, None).

    Триалы в состоянии FAILED появляются у Optuna при падении objective с
    исключением, не превращённым в optuna.TrialPruned. Явный подсчёт
    состояний (n_completed == 0) обязан исключить алгоритм и в этом случае,
    не полагаясь на ValueError от study.best_params.
    """
    X, y = toy_data
    study = _make_study_with_states([optuna.trial.TrialState.FAIL] * 3)
    mocker.patch.object(tuner.optuna, "create_study", return_value=study)
    mocker.patch.object(study, "optimize")

    model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert model is None
    assert params is None
    assert score is None


def test_optimize_pruned_and_failed_mix_returns_no_results(
    toy_data, quality_patches, mocker
):
    """N3 (issue #32): mix PRUNED + FAILED без COMPLETED → (None, None, None).

    Завершённых триалов нет ни одного — алгоритм исключается, даже если
    часть триалов отсечена прайнером, а часть упала.
    """
    X, y = toy_data
    study = _make_study_with_states(
        [
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ]
    )
    mocker.patch.object(tuner.optuna, "create_study", return_value=study)
    mocker.patch.object(study, "optimize")

    model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert model is None
    assert params is None
    assert score is None


def test_optimize_pruning_nonfinite_score_uses_worst_score(
    toy_data, quality_patches
):
    """Нефинитный avg_score в pruning-ветке → триал получает HPO_WORST_SCORE.

    Покрывает ветку ``if not np.isfinite(avg_score)`` внутри
    ``_evaluate_with_intermediate_reports``-пути (issue #13): все триалы
    завершаются с «худшим скором», но не отсекаются, поэтому optimize()
    возвращает HPO_WORST_SCORE с валидными параметрами — а отбрасывает такой
    результат уже фильтр valid_results в оркестраторе.
    """
    X, y = toy_data

    def fake_evaluate(
        trial, estimator, X, y, *, method, n_folds, test_size, random_state, scorer
    ):
        del trial, estimator, X, y, method, n_folds, test_size, random_state, scorer
        return np.nan

    with patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate):
        model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert score == HPO_WORST_SCORE
    assert params is not None
    assert model is not None


def test_optimize_pruning_not_applied_for_train_test_split(
    toy_data, quality_patches, caplog
):
    """Hold-out без шагов: прайнер не применяется, все триалы выполняются (AC-7, AC-4)."""
    X, y = toy_data

    with caplog.at_level(logging.WARNING):
        _model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="train_test_split",
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    # Все триалы завершились — лучший найден обычным правилом максимизации.
    assert score == pytest.approx(0.9)
    assert params is not None
    assert "has no intermediate steps — pruning will not be applied" in caplog.text


def test_optimize_pruning_loo_works(toy_data, quality_patches):
    """LOO как стратегия с естественными шагами работает с pruning (AC-7)."""
    X, y = toy_data

    model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="loo",
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None


def test_optimize_pruning_with_initial_params_refine_winner(
    toy_data, quality_patches
):
    """refine_winner (initial_params + enqueue_trial) совместим с pruning."""
    X, y = toy_data

    _model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        initial_params={"quality": 0.9},
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert score == pytest.approx(0.9)
    assert params is not None


def test_optimize_pruning_two_folds_edge_case(toy_data, quality_patches):
    """Малое число фолдов (2) работает с pruning (граничный сценарий)."""
    X, y = toy_data

    model, params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=2,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Обратная совместимость (AC-4)
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_without_pruning_never_reports(toy_data):
    """Без блока pruning промежуточные результаты не публикуются — поведение
    идентично текущему (каждый триал выполняется полностью)."""
    X, y = toy_data

    # Мокаем окружение так, чтобы гарантированно попасть в ветку
    # cross_val_score (как в старом коде), и контролируем вызовы trial.report.
    report_mock = MagicMock()
    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.model_selection,
            "cross_val_score",
            lambda est, Xt, yt, cv=None, scoring=None, n_jobs=1: [0.5, 0.5, 0.5],
        ),
        patch.object(tuner.optuna.trial.Trial, "report", report_mock),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )

    report_mock.assert_not_called()


def test_optimize_disabled_pruning_uses_nop_pruner(toy_data):
    """При выключенном pruning study создаётся с NopPruner — полное выполнение
    триалов гарантировано даже при случайных вызовах report()."""
    X, y = toy_data

    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.optuna, "create_study", wraps=tuner.optuna.create_study
        ) as mock_create_study,
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )

    assert isinstance(mock_create_study.call_args.kwargs["pruner"], NopPruner)


def test_optimize_enabled_pruning_uses_configured_pruner(toy_data):
    """При включённом pruning в study подключается настроенный прайнер."""
    X, y = toy_data

    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.optuna, "create_study", wraps=tuner.optuna.create_study
        ) as mock_create_study,
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    pruner = mock_create_study.call_args.kwargs["pruner"]
    assert isinstance(pruner, MedianPruner)
    assert pruner._n_warmup_steps == 1


def test_optimize_pruned_trials_not_fatal(toy_data, quality_patches, caplog):
    """Отсечённые триалы не увеличивают счётчик фатальных сбоев (AC-5).

    Много подряд отсечённых триалов не приводит к дисквалификации алгоритма.
    """
    X, y = toy_data
    n_trials = 12

    _model, _params, score = optimize(
        "ridge",
        X,
        y,
        n_trials=n_trials,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )

    # Оптимизация завершилась без InvalidAlgorithmError.
    assert score == pytest.approx(0.9)
    assert "disqualified" not in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# Сквозной сценарий через training_engine (конфиг → train_best_model)
# ──────────────────────────────────────────────────────────────────────────────
def test_train_best_model_with_pruning_e2e(tmp_path: Path) -> None:
    """Сквозной сценарий: конфиг с general.pruning.enable=true проходит через
    train_best_model без ошибок (AC-1, AC-3, AC-9)."""
    from pathlib import Path as _Path

    from sklearn.datasets import make_regression

    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"}
            ],
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    X, y = make_regression(
        n_samples=120, n_features=5, noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(config=config, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert result["params"]
    assert _Path(result["model_path"]).exists()


# ──────────────────────────────────────────────────────────────────────────────
# auto + pruning: синхронизация числа фолдов (issue #6)
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_auto_kfold_pruning_passes_resolved_k():
    """auto→kfold (k=10) + pruning: в _evaluate_with_intermediate_reports
    передаётся вычисленное k, а не исходный n_folds.

    Красный тест на старом коде: до фикса ветка ранней остановки получала
    n_folds=5 (дефолт) вместо auto_decision['k']=10, и iter_splits откатывался
    на дефолтное число фолдов (рассинхронизация с cross_val_score(cv=cv_obj)).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((400, 5)))
    y = pd.Series(rng.random(400))

    captured = {}

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        # auto резолвится в kfold c k=10 (решение вернул make_cv).
        return (
            "k_fold",
            KFold(n_splits=10, shuffle=True, random_state=random_state),
            {"method": "kfold", "k": 10, "average_test_size": 40.0},
        )

    def fake_evaluate(
        trial, estimator, X, y, *, method, n_folds, test_size, random_state, scorer
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "make_cv", fake_make_cv),
        patch.object(tuner, "get_effective_train_size", lambda *a, **k: 360),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 10


def test_optimize_auto_kfold_pruning_integration_seven_reports():
    """Интеграция: N=150/P=5 → auto резолвится в k=7; pruning-ветка выполняет
    ровно 7 фолдов и публикует 7 промежуточных шагов (AC-1).

    На старом коде здесь было бы 5 шагов (дефолтный n_folds), т.е. оценка
    расходилась бы с non-pruning веткой cross_val_score(cv=KFold(7)).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 5)))
    y = pd.Series(rng.random(150))

    # Эвристика для этих данных обязана выбрать k-fold c k=7.
    from configurable_automl_engine.common.validation_utils import (
        choose_validation_method,
    )

    decision = choose_validation_method(150, 5)
    assert decision == {"method": "kfold", "k": 7, "average_test_size": 21.4}

    report_steps: list[int] = []

    def fake_report(self, value, step):
        report_steps.append(step)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner.optuna.trial.Trial, "report", fake_report),
    ):
        _model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )

    assert score == pytest.approx(0.9)
    assert params is not None
    # Ровно k=7 промежуточных отчётов (шаги 1..7) — по одному на фолд.
    assert report_steps == list(range(1, 8))


def test_optimize_auto_pruning_parity_with_cross_val():
    """Parity: auto→kfold использует одно и то же число фолдов (k) и с pruning,
    и без него — оценки не расходятся между ветками."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 5)))  # auto -> k=7
    y = pd.Series(rng.random(150))

    cross_val_folds: list[int] = []
    iter_splits_calls: list[tuple[str | None, int | None]] = []
    real_iter_splits = tuner.iter_splits

    def spy_cross_val(est, Xt, yt, cv=None, scoring=None, n_jobs=1):
        cross_val_folds.append(cv.get_n_splits())
        return [0.5] * cv.get_n_splits()

    def spy_iter_splits(*args, **kwargs):
        iter_splits_calls.append((kwargs.get("method"), kwargs.get("n_folds")))
        yield from real_iter_splits(*args, **kwargs)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner.model_selection, "cross_val_score", spy_cross_val),
        patch.object(tuner, "iter_splits", spy_iter_splits),
    ):
        # Без pruning: оценка через cross_val_score(cv=cv_obj из make_cv).
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )
        # С pruning: оценка через iter_splits(method='k_fold', n_folds=k).
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )

    assert cross_val_folds == [7]
    assert iter_splits_calls == [("k_fold", 7)]


def test_optimize_auto_no_features_fallback_clamps_n_folds():
    """Негативный сценарий: auto без P (fallback, decision=None) с n_folds=1.

    make_cv клампит k = max(2, n_folds) через resolve_auto_no_features_fallback;
    pruning-ветка обязана повторить тот же клампинг, иначе KFold(n_splits=1)
    внутри iter_splits упал бы (латентный крах).
    """
    # Ноль признаков — P недоступен, включается fallback-ветка make_cv.
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((400, 0)))
    y = pd.Series(rng.random(400))

    captured = {}

    def fake_evaluate(
        trial, estimator, X, y, *, method, n_folds, test_size, random_state, scorer
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=1,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    # Fallback: method='k_fold' (N достаточно), число фолдов клампится до 2.
    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 2


def test_optimize_auto_kfold_low_confidence_k2():
    """Граничный сценарий: auto→kfold с k=2 (low confidence) — k передаётся
    как есть (клампинг max(2, k) не изменяет значение)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((31, 16)))  # auto -> kfold k=2
    y = pd.Series(rng.random(31))

    captured = {}

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        return (
            "k_fold",
            KFold(n_splits=2, shuffle=True, random_state=random_state),
            {"method": "kfold", "k": 2, "average_test_size": 15.5},
        )

    def fake_evaluate(
        trial, estimator, X, y, *, method, n_folds, test_size, random_state, scorer
    ):
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "make_cv", fake_make_cv),
        patch.object(tuner, "get_effective_train_size", lambda *a, **k: 16),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["n_folds"] == 2


def test_optimize_explicit_kfold_pruning_keeps_original_n_folds():
    """Явный k_fold без регрессии: pruning-ветка получает исходный n_folds
    без изменений (эффективный клампинг не применяется)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((120, 5)))
    y = pd.Series(rng.random(120))

    captured = {}

    def fake_evaluate(
        trial, estimator, X, y, *, method, n_folds, test_size, random_state, scorer
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 3


def test_optimize_auto_loo_with_pruning_works():
    """auto→LOO (N=10/P=3) + pruning: стратегия с естественными шагами работает;
    число фолдов для k-fold не используется."""
    X = pd.DataFrame(np.random.rand(10, 3))
    y = pd.Series(np.random.rand(10))

    report_steps: list[int] = []

    def fake_report(self, value, step):
        report_steps.append(step)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner.optuna.trial.Trial, "report", fake_report),
    ):
        _model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )

    assert score == pytest.approx(0.9)
    assert params is not None
    # LOO: по одному шагу на объект (10 объектов → 10 отчётов).
    assert report_steps == list(range(1, 11))


def test_optimize_auto_train_test_split_pruning_not_applied(caplog):
    """auto→train_test_split: естественных шагов нет — прайнер не применяется,
    все триалы выполняются полностью (поведение задокументировано)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((1000, 5)))  # auto -> train_test_split
    y = pd.Series(rng.random(1000))

    with (
        caplog.at_level(logging.WARNING),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
    ):
        _model, params, score = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert score == pytest.approx(0.9)
    assert params is not None
    assert "has no intermediate steps — pruning will not be applied" in caplog.text


def test_train_best_model_auto_pruning_e2e(tmp_path: Path) -> None:
    """Сквозной сценарий: validation_strategy='auto' + pruning.enable=true
    проходит через train_best_model; число фолдов берётся из auto-решения
    (N=150/P=5 → k=7), а не из дефолтного n_folds=5."""
    from pathlib import Path as _Path

    from sklearn.datasets import make_regression

    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "auto",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"}
            ],
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    X, y = make_regression(
        n_samples=150, n_features=5, noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(config=config, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert result["params"]
    assert _Path(result["model_path"]).exists()