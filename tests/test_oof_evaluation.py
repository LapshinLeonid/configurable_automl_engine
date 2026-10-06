"""Тесты OOF-канала оценки кандидатов (issue #62).

Покрытие:
    1. ``iter_splits(include_indices=True)``: позиционные индексы валидационных
       строк соответствуют подмножествам X_val/y_val (k_fold и train_test_split,
       в т.ч. при random_state=None).
    2. ``metrics.oof_rmse``: метрика по всему вектору сразу; маскирование
       NaN/None/inf пар; ошибка при отсутствии валидных пар.
    3. Positive: RMSE_oof ≈ среднему по фолдам при равных фолдах; точный
       pooled-пересчёт для неравных фолдов; каждая строка предсказана моделью,
       не видевшей её (совпадение с ручным per-fold пересчётом — отсутствие
       утечки); покрытие 100% для k_fold/loo; выравнивание по pandas-индексу.
    4. Negative: при утечке (модель обучалась на строке) OOF-вектор отличается
       от честного — тест это детектирует (in-sample RMSE заметно ниже RMSE_oof).
    5. Boundary: N меньше числа фолдов (fallback на train_test_split);
       train_test_split покрывает только валидационную часть; фолды неравного
       размера; NaN в предсказаниях; очень малый N; дублирующийся индекс.
    6. Обратная совместимость: OOF — отдельный канал, ``val_score`` фаз HPO
       не меняется; сериализация сохраняет OOF-атрибуты; сброс между fit().
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error

from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.metrics import oof_rmse
from configurable_automl_engine.validation import iter_splits


# ──────────────────────────────────────────────────────────────────────────────
#  Хелперы
# ──────────────────────────────────────────────────────────────────────────────


def _noisy_regression(
    n: int = 60, p: int = 6, seed: int = 7, noise: float = 3.0
) -> tuple[pd.DataFrame, pd.Series]:
    """Зашумлённый датасет: первый признак информативен, остальные — шум.

    На таких данных модели переобучаются, поэтому in-sample RMSE заметно ниже
    честного OOF-RMSE (нужно для негативного теста на утечку).
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p))
    X.iloc[:, 0] = X.iloc[:, 0] * 3.0
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(n) * noise)
    return X, y


def _manual_oof(
    X: Any,
    y: Any,
    trainer: ModelTrainer,
    *,
    method: str,
    n_folds: int,
    seed: int,
    test_size: float = 0.2,
) -> np.ndarray:
    """Пересчитать OOF-вектор вручную: те же сплиты и клоны пайплайна на фолд.

    Повторяет логику ``_score_on_validation_splits`` независимо, чтобы проверить
    отсутствие утечки: каждая строка предсказывается пайплайном, обученным без
    её участия (test-часть фолда).
    """
    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    oof = np.full(len(X), np.nan, dtype=float)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X,
        y,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=seed,
        include_indices=True,
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        oof[np.asarray(test_idx)] = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
    return oof


class _ConstRegressor(BaseEstimator):
    """Минимальный регрессор для белого ящика: константа + опциональный NaN."""

    def __init__(self, value: float = 1.0, nan_condition=None):
        self.value = value
        self.nan_condition = nan_condition

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = np.full(len(X), self.value, dtype=float)
        if self.nan_condition is not None:
            mask = self.nan_condition(np.asarray(X))
            preds[mask] = np.nan
        return preds


# ──────────────────────────────────────────────────────────────────────────────
#  1. iter_splits(include_indices=True)
# ──────────────────────────────────────────────────────────────────────────────


def test_iter_splits_include_indices_kfold_match_positions():
    """k_fold: test_idx указывает на те же строки, что X_val (numpy и pandas)."""
    rng = np.random.RandomState(1)
    X_np = rng.randn(30, 3)
    y_np = rng.randn(30)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X_np, y_np, method="k_fold", n_folds=3, random_state=5, include_indices=True
    ):
        idx = np.asarray(test_idx)
        assert np.array_equal(X_te, X_np[idx])
        assert np.array_equal(np.asarray(y_te), y_np[idx])

    X_df = pd.DataFrame(X_np)
    y_s = pd.Series(y_np)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X_df, y_s, method="k_fold", n_folds=3, random_state=5, include_indices=True
    ):
        idx = np.asarray(test_idx)
        assert np.array_equal(X_te.to_numpy(), X_df.to_numpy()[idx])
        assert np.array_equal(y_te.to_numpy(), y_s.to_numpy()[idx])


def test_iter_splits_include_indices_tts_match_positions():
    """train_test_split: test_idx согласован с X_val, в т.ч. при random_state=None.

    До issue #62 сплит строился вторым вызовом train_test_split, что при
    random_state=None давало бы другую перестановку и рассинхрон индексов.
    """
    rng = np.random.RandomState(2)
    X = rng.randn(40, 3)
    y = rng.randn(40)
    for random_state in (7, None):
        for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
            X,
            y,
            method="train_test_split",
            test_size=0.25,
            random_state=random_state,
            include_indices=True,
        ):
            idx = np.asarray(test_idx)
            assert np.array_equal(X_te, X[idx])
            assert np.array_equal(np.asarray(y_te), y[idx])


def test_iter_splits_default_contract_unchanged():
    """Без include_indices контракт прежний: 4 элемента, как раньше."""
    rng = np.random.RandomState(3)
    X = rng.randn(20, 2)
    y = rng.randn(20)
    for split in iter_splits(X, y, method="k_fold", n_folds=2):
        assert len(split) == 4


# ──────────────────────────────────────────────────────────────────────────────
#  2. metrics.oof_rmse
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_rmse_over_vector():
    """RMSE по всему вектору сразу (не усреднение по фолдам)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 4.0])
    expected = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert oof_rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-12)


def test_oof_rmse_masks_non_finite_pairs():
    """NaN/None/inf в y_pred или y_true отбрасываются, а не роняют метрику."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    # Валидные пары: 0, 2, 4 — ошибки равны нулю.
    assert oof_rmse(y_true, y_pred) == pytest.approx(0.0)

    y_true_nan = np.array([1.0, np.nan, 3.0])
    y_pred_ok = np.array([1.0, 2.0, 3.0])
    assert oof_rmse(y_true_nan, y_pred_ok) == pytest.approx(0.0)


def test_oof_rmse_no_valid_pairs_raises():
    """Нет ни одной валидной пары → ValueError (OOF-оценка невозможна)."""
    with pytest.raises(ValueError, match="No valid"):
        oof_rmse(np.array([1.0, 2.0]), np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="No valid"):
        oof_rmse(np.array([1.0, 2.0]), np.array([np.inf, -np.inf]))


# ──────────────────────────────────────────────────────────────────────────────
#  3. Positive: честность и покрытие 100%
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_rmse_approx_mean_fold_rmse_equal_folds():
    """Равные фолды: RMSE_oof ≈ среднему RMSE по фолдам.

    Точное соотношение — pooled: oof² = mean(MSE_fold) (фолды равного размера);
    приближённо (в пределах ~5%) RMSE_oof ≈ mean(RMSE_fold) — утверждение из
    issue #62.
    """
    X, y = _noisy_regression(n=60, p=6, seed=7)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=7,
    ).fit(X, y)

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    fold_rmse: list[float] = []
    fold_mse: list[float] = []
    for X_tr, X_te, y_tr, y_te, _ in iter_splits(
        X, y, method="k_fold", n_folds=3, test_size=0.2, random_state=7, include_indices=True
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        preds = fold_pipe.predict(X_te)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_te, preds))))
        fold_mse.append(float(mean_squared_error(y_te, preds)))

    assert trainer.oof_score_ is not None
    # Точное pooled-соотношение для равных фолдов.
    assert trainer.oof_score_ == pytest.approx(
        float(np.sqrt(np.mean(fold_mse))), rel=1e-9
    )
    # Приближённое равенство среднему RMSE по фолдам (Jensen: mean ≤ pooled).
    assert trainer.oof_score_ == pytest.approx(float(np.mean(fold_rmse)), rel=0.05)


def test_oof_no_leakage_matches_manual_reconstruction():
    """Отсутствие утечки: каждая строка предсказана моделью без её участия.

    Ручной per-fold пересчёт (те же сплиты/клоны) даёт в точности тот же
    OOF-вектор, что собрал тренер, — по построению это честные out-of-fold
    предсказания, а не train-предсказания.
    """
    X, y = _noisy_regression(n=48, p=5, seed=11)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=4, seed=7)
    assert trainer.oof_predictions_ is not None
    np.testing.assert_allclose(
        np.asarray(trainer.oof_predictions_), manual, rtol=1e-9, atol=1e-9
    )
    # Полное покрытие: ни одна строка не осталась без честного предсказания.
    assert np.isfinite(manual).all()
    assert trainer.oof_coverage_ == 1.0


def test_oof_predictions_aligned_with_pandas_index():
    """Нестандартный pandas-индекс сохраняется в y_pred_oof (требование 2)."""
    X, y = _noisy_regression(n=40, p=4, seed=13)
    X = X.copy()
    X.index = np.arange(len(X)) * 10 + 5
    y = y.copy()
    y.index = X.index

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=3,
    ).fit(X, y)

    assert isinstance(trainer.oof_predictions_, pd.Series)
    assert list(trainer.oof_predictions_.index) == list(X.index)
    assert trainer.oof_coverage_ == 1.0
    # Выравнивание: значение для строки с меткой idx совпадает с ручным OOF.
    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=4, seed=3)
    np.testing.assert_allclose(
        trainer.oof_predictions_.to_numpy(), manual, rtol=1e-9, atol=1e-9
    )


def test_oof_numpy_input_positional_alignment():
    """numpy-вход без индекса: y_pred_oof — np.ndarray, выровненный по позиции."""
    X_np, y_np = _noisy_regression(n=50, p=4, seed=5)
    X_np = X_np.to_numpy()
    y_np = y_np.to_numpy()

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=5,
    ).fit(X_np, y_np)

    assert isinstance(trainer.oof_predictions_, np.ndarray)
    assert trainer.oof_coverage_ == 1.0
    assert trainer.oof_predictions_.shape == (len(X_np),)
    # Позиционное выравнивание против ручного OOF на тех же numpy-массивах.
    manual = _manual_oof(X_np, y_np, trainer, method="k_fold", n_folds=4, seed=5)
    np.testing.assert_allclose(trainer.oof_predictions_, manual, rtol=1e-9, atol=1e-9)


def test_oof_loo_full_coverage():
    """LOO: каждая строка — отдельный фолд, покрытие 100%."""
    X, y = _noisy_regression(n=12, p=3, seed=17)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="loo"
    ).fit(X, y)

    assert trainer.oof_coverage_ == 1.0
    assert trainer.oof_score_ is not None
    assert trainer.oof_score_ >= 0
    assert trainer.oof_predictions_ is not None
    assert np.isfinite(np.asarray(trainer.oof_predictions_)).all()


# ──────────────────────────────────────────────────────────────────────────────
#  4. Negative: детекция утечки
# ──────────────────────────────────────────────────────────────────────────────


def test_leaky_in_sample_predictions_detected():
    """Утечка детектируется: in-sample RMSE заметно ниже честного RMSE_oof.

    Финальный пайплайн обучен на 100% строк — каждая строка «предсказана»
    моделью, видевшей её. Такой (утечный) OOF-вектор систематически отличается
    от честного и даёт оптимистично низкую ошибку на зашумлённых данных.
    """
    X, y = _noisy_regression(n=80, p=12, seed=0, noise=4.0)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=42,
    ).fit(X, y)

    leaky_preds = trainer.predict(X)  # модель видела все строки
    leaky_rmse = float(np.sqrt(mean_squared_error(y, leaky_preds)))

    assert trainer.oof_score_ is not None
    # Честный OOF-вектор не совпадает с утечным (построчно).
    honest = np.asarray(trainer.oof_predictions_)
    assert not np.allclose(honest, leaky_preds, rtol=1e-6, atol=1e-6)
    # In-sample ошибка систематически ниже честной out-of-fold.
    assert leaky_rmse < trainer.oof_score_
    assert trainer.oof_score_ - leaky_rmse > 0.5


# ──────────────────────────────────────────────────────────────────────────────
#  5. Boundary
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_unequal_fold_sizes_pooled_rmse_exact():
    """Неравные фолды: конкатенация не «разъезжается»; pooled-RMSE точен."""
    rng = np.random.RandomState(3)
    n = 11
    X = pd.DataFrame(rng.randn(n, 3))
    X.iloc[:, 0] *= 2.0
    y = pd.Series(X.iloc[:, 0] * 1.5 + rng.randn(n))

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=3,
    ).fit(X, y)

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    fold_mse: list[float] = []
    fold_n: list[int] = []
    for X_tr, X_te, y_tr, y_te, _ in iter_splits(
        X, y, method="k_fold", n_folds=3, test_size=0.2, random_state=3, include_indices=True
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        preds = fold_pipe.predict(X_te)
        fold_mse.append(float(mean_squared_error(y_te, preds)))
        fold_n.append(len(y_te))

    assert trainer.oof_coverage_ == 1.0
    pooled = float(np.sqrt(np.sum(np.array(fold_n) * np.array(fold_mse)) / n))
    assert trainer.oof_score_ == pytest.approx(pooled, rel=1e-9)


def test_oof_train_test_split_partial_coverage_warning(caplog):
    """train_test_split: OOF покрывает только валидационную часть (ограничение).

    Ограничение явно задокументировано в логе; RMSE_oof считается по покрытому
    подмножеству и равен RMSE на hold-out части.
    """
    X, y = _noisy_regression(n=80, p=5, seed=21)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="train_test_split",
            test_size=0.25,
            random_state=9,
        ).fit(X, y)

    assert trainer.oof_coverage_ is not None
    assert 0.0 < trainer.oof_coverage_ < 1.0
    assert trainer.oof_score_ is not None
    assert "covers only the validation part" in caplog.text

    # Точность: RMSE_oof = RMSE по покрытой (валидационной) части hold-out.
    manual = _manual_oof(
        X, y, trainer, method="train_test_split", n_folds=5, seed=9, test_size=0.25
    )
    valid = np.isfinite(manual)
    assert trainer.oof_score_ == pytest.approx(
        float(np.sqrt(mean_squared_error(y[valid], manual[valid]))), rel=1e-6
    )


def test_oof_small_data_kfold_fallback_to_split(caplog):
    """N меньше числа фолдов: k_fold откатывается на train_test_split.

    OOF покрывает только hold-out часть; обучение завершается штатно.
    Предупреждение опирается на фактически применённый метод (ревью PR #31):
    логируется ограничение «covers only the validation part», а не вводящий
    в заблуждение лог «non-finite».
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(6, 2))
    y = pd.Series(X[0] * 2.0 + rng.randn(6) * 0.1)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.validation"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=10,
            random_state=42,
        ).fit(X, y)

    assert "Falling back to 'train_test_split'" in caplog.text
    assert trainer.oof_coverage_ is not None
    assert 0.0 < trainer.oof_coverage_ < 1.0
    assert trainer.oof_score_ is not None
    # Fallback-ветка: ограничение OOF задокументировано честно (не «non-finite»).
    assert "covers only the validation part" in caplog.text
    assert "excluded from RMSE_oof" not in caplog.text


def test_oof_nan_predictions_reduced_coverage(caplog):
    """NaN в предсказаниях фолда: строка исключается, RMSE_oof по остальным."""
    rng = np.random.RandomState(4)
    X = pd.DataFrame(rng.randn(24, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(24) * 0.5)

    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=3
    )
    trainer.feature_names = ["f0", "f1"]
    trainer.categorical_features = []
    trainer.numerical_features = ["f0", "f1"]
    preprocessor = trainer._build_preprocessor(["f0", "f1"])
    # Модель возвращает NaN для строк с f0 > 0.5 (белый ящик).
    model = _ConstRegressor(value=1.5, nan_condition=lambda Xa: Xa[:, 0] > 0.5)

    with (
        caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            return_value=lambda est, Xv, yv: 0.5,
        ),
    ):
        trainer._score_on_validation_splits(
            X, y, preprocessor, model, feature_selection_active=False
        )

    valid_mask = X["f0"] <= 0.5
    expected = float(
        np.sqrt(mean_squared_error(y[valid_mask], np.full(valid_mask.sum(), 1.5)))
    )
    assert trainer.oof_coverage_ == pytest.approx(valid_mask.mean(), rel=1e-9)
    assert trainer.oof_score_ == pytest.approx(expected, rel=1e-9)
    assert "excluded from RMSE_oof" in caplog.text
    assert trainer.val_score == pytest.approx(0.5)


def test_oof_nan_in_y_reduced_coverage(caplog):
    """NaN в y: строки с нефинитным таргетом исключаются из RMSE_oof.

    Edge case из issue #62: метрика по вектору не должна «разъезжаться» —
    пары с NaN в y_true маскируются, обучение завершается штатно.
    """
    rng = np.random.RandomState(6)
    X = pd.DataFrame(rng.randn(24, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(24) * 0.5)
    y.iloc[[3, 9, 17]] = np.nan  # три строки с нефинитным таргетом

    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=3
    )
    trainer.feature_names = ["f0", "f1"]
    trainer.categorical_features = []
    trainer.numerical_features = ["f0", "f1"]
    preprocessor = trainer._build_preprocessor(["f0", "f1"])
    # Модель игнорирует y при fit — белый ящик для изоляции OOF-канала.
    model = _ConstRegressor(value=1.5)

    with (
        caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            return_value=lambda est, Xv, yv: 0.5,
        ),
    ):
        trainer._score_on_validation_splits(
            X, y, preprocessor, model, feature_selection_active=False
        )

    valid_mask = y.notna()
    expected = float(
        np.sqrt(mean_squared_error(y[valid_mask], np.full(valid_mask.sum(), 1.5)))
    )
    assert trainer.oof_coverage_ == pytest.approx(valid_mask.mean(), rel=1e-9)
    assert trainer.oof_score_ == pytest.approx(expected, rel=1e-9)
    assert "excluded from RMSE_oof" in caplog.text


def test_oof_very_small_n_warning(caplog):
    """Очень малый N (2–5): предупреждение о высокой дисперсии OOF-оценки."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=2,
        ).fit(X, y)

    assert "very small data" in caplog.text
    assert trainer.oof_score_ is not None
    assert trainer.oof_coverage_ == 1.0


def test_oof_duplicate_index_falls_back_to_positional():
    """Невосстановимый индекс (дубликаты): выравнивание по позиции (ndarray)."""
    X, y = _noisy_regression(n=30, p=3, seed=23)
    X = X.copy()
    X.index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14]
    y = y.copy()
    y.index = X.index

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    # Неуникальный индекс — серия была бы неоднозначной; отдаём ndarray по позиции.
    assert isinstance(trainer.oof_predictions_, np.ndarray)
    assert trainer.oof_coverage_ == 1.0
    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=3, seed=5)
    np.testing.assert_allclose(trainer.oof_predictions_, manual, rtol=1e-9, atol=1e-9)


def test_oof_multiindex_supported():
    """MultiIndex-индекс: fit() не падает, метки строк сохраняются.

    Регрессия блокера ревью PR #31: ``Index.hasnans`` не определён для
    MultiIndex (NotImplementedError), из-за чего ``fit()`` падал целиком,
    хотя на main такой вход работал. Теперь уникальный MultiIndex сохраняется
    в ``y_pred_oof`` как ``pd.Series``.
    """
    idx = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2, 3]], names=["grp", "num"]
    )
    rng = np.random.RandomState(11)
    X = pd.DataFrame(rng.randn(6, 2), index=idx)
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(6) * 0.5, index=idx)

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=2,
        random_state=5,
    ).fit(X, y)

    assert trainer.oof_score_ is not None
    assert trainer.oof_coverage_ == 1.0
    assert isinstance(trainer.oof_predictions_, pd.Series)
    assert isinstance(trainer.oof_predictions_.index, pd.MultiIndex)
    assert list(trainer.oof_predictions_.index) == list(idx)
    np.testing.assert_allclose(
        trainer.oof_predictions_.to_numpy(),
        _manual_oof(X, y, trainer, method="k_fold", n_folds=2, seed=5),
        rtol=1e-9,
        atol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  6. Обратная совместимость и состояние
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_is_separate_channel_val_score_unchanged():
    """OOF — отдельный канал: val_score/additional_scores не затрагиваются.

    Для score-метрики (r2) RMSE_oof — это именно RMSE (>= 0), а не R²; значения
    каналов не смешиваются.
    """
    X, y = _noisy_regression(n=50, p=4, seed=29)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["rmse"],
        validation_strategy="k_fold",
        n_folds=4,
        random_state=5,
    ).fit(X, y)

    assert trainer.val_score is not None and trainer.val_score <= 1.0
    assert "rmse" in trainer.additional_scores
    assert trainer.oof_score_ is not None and trainer.oof_score_ >= 0
    # RMSE_oof не равен R²-каналу и не является train-скором.
    assert trainer.oof_score_ != trainer.val_score
    # OOF-RMSE согласован с каналом дополнительной метрики на полном векторе.
    assert trainer.oof_score_ == pytest.approx(
        float(
            np.sqrt(mean_squared_error(y, np.asarray(trainer.oof_predictions_)))
        ),
        rel=1e-9,
    )


def test_oof_vector_persisted_in_candidate_result(tmp_path):
    """OOF-вектор — часть результата оценки кандидата: сохраняется в артефакте.

    Требование 4 issue #62: diversity/unique контуры аудита (T2) считаются по
    y_pred_oof, поэтому вектор (не только метрика) обязан переживать
    сериализацию вместе с pandas-индексом строк.
    """
    X, y = _noisy_regression(n=45, p=4, seed=31)
    X = X.copy()
    X.index = np.arange(len(X)) * 100 + 7
    y = y.copy()
    y.index = X.index
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    path = tmp_path / "trainer_oof.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)

    assert loaded.oof_score_ == pytest.approx(trainer.oof_score_, rel=1e-12)
    assert loaded.oof_coverage_ == trainer.oof_coverage_
    # Вектор сохранён как часть результата: тип и индекс строк целы.
    assert isinstance(loaded.oof_predictions_, pd.Series)
    assert list(loaded.oof_predictions_.index) == list(X.index)
    np.testing.assert_allclose(
        loaded.oof_predictions_.to_numpy(),
        trainer.oof_predictions_.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_oof_reset_between_fits():
    """Повторный fit сбрасывает OOF-состояние (нет остатков прошлого обучения)."""
    X, y = _noisy_regression(n=50, p=4, seed=37)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=4
    ).fit(X, y)
    assert trainer.oof_score_ is not None

    # Неудачный fit не должен оставлять OOF-значения от предыдущего обучения.
    with patch(
        "configurable_automl_engine.trainer.get_scorer_object",
        return_value=lambda est, Xv, yv: (_ for _ in ()).throw(RuntimeError("boom")),
    ):
        with pytest.raises(Exception, match="Validation scoring failed"):
            trainer.fit(X, y)

    assert trainer.oof_score_ is None
    assert trainer.oof_predictions_ is None
    assert trainer.oof_coverage_ is None
    assert trainer.val_score is None