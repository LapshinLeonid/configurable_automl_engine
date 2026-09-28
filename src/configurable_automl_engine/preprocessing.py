"""Preprocessing: единая точка построения препроцессора признаков.

Модуль инкапсулирует два низкоуровневых кирпича подготовки данных:

    1. :func:`detect_feature_types` — автоопределение категориальных и
       числовых колонок по ``pd.DataFrame``.
    2. :func:`build_preprocessor` — сборка ``sklearn.ColumnTransformer``
       с предобработкой категорий (импутация ``most_frequent`` + выбранная
       стратегия кодирования) и скалированием для числовых признаков.
       Стратегия обработки числовых признаков (стратегия импутации и тип
       масштабирования) задаётся параметрами
       ``imputation_strategy``/``scaling`` и обычно берётся из адаптивного
       пресета предобработки (:mod:`preprocessing_presets`), который
       автоматически выбирается по регрессионному алгоритму.

Поддерживаемые стратегии кодирования категориальных признаков (issue #20):

    * **one_hot** (по умолчанию) — ``OneHotEncoder``: каждая категория
      становится бинарной колонкой; неизвестные категории игнорируются.
    * **ordinal** — ``OrdinalEncoder``: одна числовая колонка на категорию;
      неизвестные категории отображаются в ``-1``.
    * **target** — :class:`TargetEncodingTransformer`: среднее целевой
      переменной по категории со сглаживанием; статистики считаются только
      по обучающей части (в ``fit``), fallback для неизвестных категорий —
      глобальное среднее (или настраиваемое значение).
    * **frequency** — :class:`FrequencyEncodingTransformer`: относительная
      частота категории в обучающей выборке; неизвестные категории -> ``0.0``.
    * **hashing** — :class:`HashingEncodingTransformer`: детерминированное
      хеширование категорий в фиксированное число бинарных колонок
      (``n_components``); неизвестные категории хешируются тем же способом.

Для колонок высокой кардинальности поддерживается автоматический режим:
колонки, число уникальных значений которых превышает
``high_cardinality_threshold``, кодируются стратегией
``high_cardinality_encoding``, остальные — стратегией ``encoding``.
Разделение вычисляется в :meth:`SplitCategoricalEncoder.fit` по фактическим
данным (обучающей части), поэтому решение не зависит от валидационных данных.

Единая точка построения препроцессора используется в ОБОИХ местах обучения —
фазе подбора гиперпараметров (``tuner.optimize``) и финальном обучении
(``trainer.ModelTrainer``), что исключает рассинхрон логики предобработки
между этапами (FR-4).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from configurable_automl_engine.preprocessing_presets import (
    ImputationStrategy,
    ScalingType,
)

logger = logging.getLogger(__name__)

EncodingStrategy = Literal["one_hot", "ordinal", "target", "frequency", "hashing"]

_VALID_ENCODING_STRATEGIES = (
    "one_hot",
    "ordinal",
    "target",
    "frequency",
    "hashing",
)


def _to_string_array(X):
    """Привести категориальную матрицу к строковому объектному массиву.

    ``np.ndarray.astype(str)`` даёт fixed-width unicode (``<U``), а
    ``SimpleImputer`` принимает только ``object``-массивы, поэтому после
    преобразования в строки выполняется повторный каст в ``object`` dtype.
    """
    return np.asarray(X).astype(str).astype(object)


def _as_2d(X: Any) -> np.ndarray:
    """Привести вход трансформера к двумерному массиву (n_samples, n_cols)."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


# --------------------------------------------------------------------------- #
#                       Кастомные энкодеры категорий                          #
# --------------------------------------------------------------------------- #


class TargetEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Target encoding (mean encoding) со сглаживанием и fallback-значением.

    Для каждой категориальной колонки среднее целевой переменной по категории
    смешивается с глобальным средним через параметр сглаживания ``m``:

        encoded(cat) = (n_cat * mean_cat + m * global_mean) / (n_cat + m)

    Все статистики вычисляются **только** в :meth:`fit` из переданного ``y``,
    поэтому информация из валидационной/тестовой части не влияет на кодирование
    (нет data leakage). Категории, отсутствовавшие в обучающей выборке,
    отображаются в fallback-значение (глобальное среднее либо настраиваемое
    через ``fallback``).

    Args:
        smoothing: Параметр сглаживания ``m`` (``>= 0``). При ``m=0``
            используется чистое среднее по категории; при больших ``m``
            редкие категории приближаются к глобальному среднему.
        fallback: Значение для категорий, отсутствующих в обучающей выборке.
            ``None`` — использовать глобальное среднее целевой переменной.
    """

    def __init__(self, smoothing: float = 20.0, fallback: float | None = None):
        self.smoothing = smoothing
        self.fallback = fallback

    def fit(self, X: Any, y: Any = None) -> TargetEncodingTransformer:
        """Вычислить target-статистики по обучающим данным.

        Args:
            X: Матрица категориальных признаков (строки после импутации).
            y: Целевая переменная. Обязательна — без неё кодирование
                невозможно.

        Returns:
            Обученный трансформер.

        Raises:
            ValueError: Если ``y`` не передан либо содержит NaN.
        """
        if y is None:
            raise ValueError("TargetEncodingTransformer requires y during fit().")
        X_arr = _as_2d(X)
        y_arr = np.asarray(y).ravel()
        try:
            y_float = y_arr.astype(np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "TargetEncodingTransformer requires a numeric target y."
            ) from e
        if not np.isfinite(y_float).all():
            raise ValueError(
                "TargetEncodingTransformer does not support non-finite values in y."
            )

        self.global_mean_ = float(np.mean(y_float))
        self.fallback_ = (
            self.global_mean_ if self.fallback is None else float(self.fallback)
        )
        self.statistics_: list[dict[str, float]] = []
        for col in range(X_arr.shape[1]):
            group = pd.DataFrame(
                {"cat": pd.Series(X_arr[:, col]), "target": y_float}
            ).groupby("cat")["target"]
            agg = group.agg(["mean", "count"])
            smoothed = (
                agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing
            ) / (agg["count"] + self.smoothing)
            self.statistics_.append(
                {str(cat): float(value) for cat, value in smoothed.items()}
            )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Заменить категории их target-значениями.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Числовая матрица той же размерности (1 колонка на входную колонку).
            Неизвестные категории отображаются в fallback-значение.
        """
        check_is_fitted(self, attributes=["statistics_"])
        X_arr = _as_2d(X)
        out = np.empty((X_arr.shape[0], X_arr.shape[1]), dtype=np.float64)
        for col, stats in enumerate(self.statistics_):
            mapped = pd.Series(X_arr[:, col]).map(stats)
            out[:, col] = mapped.fillna(self.fallback_).to_numpy(dtype=np.float64)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Вернуть имена выходных признаков (1 колонка на входную)."""
        check_is_fitted(self, attributes=["statistics_"])
        names = _feature_names(input_features, len(self.statistics_))
        return np.asarray(names, dtype=object)


class FrequencyEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Frequency encoding: относительная частота категории в обучающей выборке.

    Детерминированное кодирование, не требующее ``y``. Неизвестные категории
    на этапе предсказания отображаются в ``0.0`` (задокументированный
    fallback). Частоты вычисляются только по обучающей части в :meth:`fit`.
    """

    def fit(self, X: Any, y: Any = None) -> FrequencyEncodingTransformer:
        """Вычислить частоты категорий по обучающим данным.

        Args:
            X: Матрица категориальных признаков.
            y: Не используется (принимается для совместимости с API sklearn).

        Returns:
            Обученный трансформер.
        """
        X_arr = _as_2d(X)
        n_rows = X_arr.shape[0]
        self.frequencies_: list[dict[str, float]] = []
        self.n_columns_ = X_arr.shape[1]
        for col in range(X_arr.shape[1]):
            counts = pd.Series(X_arr[:, col]).value_counts(dropna=False)
            self.frequencies_.append((counts / n_rows).to_dict())
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Заменить категории их частотами.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Числовая матрица той же размерности. Неизвестные категории -> ``0.0``.
        """
        check_is_fitted(self, attributes=["frequencies_"])
        X_arr = _as_2d(X)
        out = np.empty((X_arr.shape[0], X_arr.shape[1]), dtype=np.float64)
        for col, freqs in enumerate(self.frequencies_):
            mapped = pd.Series(X_arr[:, col]).map(freqs)
            out[:, col] = mapped.fillna(0.0).to_numpy(dtype=np.float64)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Вернуть имена выходных признаков (1 колонка на входную)."""
        check_is_fitted(self, attributes=["frequencies_"])
        names = _feature_names(input_features, self.n_columns_)
        return np.asarray(names, dtype=object)


class HashingEncodingTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Feature hashing: детерминированное кодирование категорий хешем.

    Каждая категория хешируется в индекс ``[0, n_components)`` с помощью
    ``hashlib.blake2b`` (соль зависит от ``random_state``), после чего
    выставляется бинарная колонка. Число выходных признаков фиксировано
    (``n_components`` на входную колонку) и управляется конфигурацией — рост
    размерности не зависит от кардинальности колонки.

    Хеширование полностью детерминировано при фиксированных настройках и
    ``random_state``: неизвестные категории на этапе предсказания хешируются
    тем же способом, поэтому падений не возникает. Возможны коллизии хеша
    (разные категории -> одна колонка) — задокументированное ограничение.

    Args:
        n_components: Число бинарных колонок на одну входную категориальную
            колонку (``>= 1``).
        random_state: Зерно для соли хеширования (воспроизводимость).
    """

    def __init__(self, n_components: int = 16, random_state: int | None = 42):
        self.n_components = n_components
        self.random_state = random_state

    def _hash_token(self, token: Any, salt: str) -> int:
        """Детерминированно захешировать категорию в ``[0, n_components)``.

        Соль включается в начало сообщения (а не в параметр ``salt`` blake2b,
        который ограничен 16 байтами), поэтому длина соли ничем не ограничена
        и разные ``random_state`` гарантированно дают разные проекции.
        """
        digest = hashlib.blake2b(digest_size=8)
        digest.update(salt.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(token).encode("utf-8", errors="replace"))
        return int.from_bytes(digest.digest(), byteorder="little") % self.n_components

    def fit(self, X: Any, y: Any = None) -> HashingEncodingTransformer:
        """Подготовить соли для каждой колонки.

        Args:
            X: Матрица категориальных признаков.
            y: Не используется (принимается для совместимости с API sklearn).

        Returns:
            Обученный трансформер.
        """
        X_arr = _as_2d(X)
        self.n_columns_ = X_arr.shape[1]
        self.salts_ = [
            f"hashing_col_{i}_seed_{self.random_state}" for i in range(self.n_columns_)
        ]
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Закодировать категории бинарными хеш-колонками.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Бинарная матрица размерности ``(n_samples, n_columns * n_components)``.
        """
        check_is_fitted(self, attributes=["salts_"])
        X_arr = _as_2d(X)
        n_rows, n_cols = X_arr.shape
        out = np.zeros((n_rows, n_cols * self.n_components), dtype=np.float64)
        for col in range(n_cols):
            uniq, inverse = np.unique(X_arr[:, col], return_inverse=True)
            hashed = np.array(
                [self._hash_token(token, self.salts_[col]) for token in uniq],
                dtype=np.int64,
            )
            flat_indices = col * self.n_components + hashed[inverse]
            out[np.arange(n_rows), flat_indices] = 1.0
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Вернуть имена выходных признаков (``n_components`` на входную)."""
        check_is_fitted(self, attributes=["salts_"])
        base = _feature_names(input_features, self.n_columns_)
        names = [
            f"{base[i]}__h{j}"
            for i in range(self.n_columns_)
            for j in range(self.n_components)
        ]
        return np.asarray(names, dtype=object)


def _feature_names(input_features: Any, n_columns: int) -> list[str]:
    """Сформировать список имён для ``n_columns`` входных признаков."""
    if input_features is None:
        return [f"cat_{i}" for i in range(n_columns)]
    features = list(input_features)
    if len(features) != n_columns:
        raise ValueError(
            f"input_features has {len(features)} elements, expected {n_columns}."
        )
    return features


class SplitCategoricalEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """По-колоночный категориальный энкодер с опциональным high-cardinality режимом.

    На этапе :meth:`fit` по фактическим данным вычисляется кардинальность
    каждой колонки. Колонки, чья кардинальность превышает
    ``high_cardinality_threshold``, кодируются стратегией
    ``high_cardinality_encoding``; остальные — стратегией ``encoding``.
    Если порог не задан (``None``), все колонки кодируются стратегией
    ``encoding`` (поведение по умолчанию).

    Выходы энкодеров конкатенируются горизонтально в порядке «default-колонки,
    затем high-cardinality колонки». Разделение зависит только от обучающих
    данных, поэтому решение о стратегии не «подглядывает» в валидацию.

    Args:
        encoding: Стратегия кодирования для обычных колонок.
        high_cardinality_threshold: Порог кардинальности (``>= 0``); колонки с
            числом уникальных значений строго больше порога считаются
            high-cardinality. ``None`` — режим отключён.
        high_cardinality_encoding: Стратегия для high-cardinality колонок.
            Должна быть задана вместе с ``high_cardinality_threshold``.
        hashing_n_components: Число бинарных колонок для hashing-кодирования.
        target_encoding_smoothing: Параметр сглаживания target encoding.
        target_encoding_fallback: Fallback-значение target encoding
            (``None`` — глобальное среднее).
        random_state: Зерно для детерминированного hashing-кодирования.
    """

    def __init__(
        self,
        encoding: EncodingStrategy = "one_hot",
        high_cardinality_threshold: int | None = None,
        high_cardinality_encoding: EncodingStrategy | None = None,
        hashing_n_components: int = 16,
        target_encoding_smoothing: float = 20.0,
        target_encoding_fallback: float | None = None,
        random_state: int | None = 42,
    ):
        self.encoding = encoding
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        self.random_state = random_state
        # Внутренние энкодеры создаются на этапе конструирования, чтобы их
        # структура была интроспектируема без данных; обучение выполняется в
        # fit(). При клонировании (clone) создаётся новый экземпляр со свежими
        # внутренними энкодерами, поэтому состояние между фолдами не накапливается.
        self.default_encoder_ = self._make_encoder(encoding)
        self.hc_encoder_: BaseEstimator | None = (
            self._make_encoder(high_cardinality_encoding)
            if high_cardinality_encoding is not None
            else None
        )

    def _make_encoder(self, strategy: EncodingStrategy) -> BaseEstimator:
        """Создать энкодер по имени стратегии."""
        if strategy == "one_hot":
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if strategy == "ordinal":
            return OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        if strategy == "target":
            return TargetEncodingTransformer(
                smoothing=self.target_encoding_smoothing,
                fallback=self.target_encoding_fallback,
            )
        if strategy == "frequency":
            return FrequencyEncodingTransformer()
        if strategy == "hashing":
            return HashingEncodingTransformer(
                n_components=self.hashing_n_components,
                random_state=self.random_state,
            )
        raise ValueError(
            f"Unknown encoding strategy: {strategy!r}. "
            f"Expected one of {_VALID_ENCODING_STRATEGIES}."
        )

    def set_params(self, **params: Any) -> SplitCategoricalEncoder:
        """Обновить параметры и пересоздать внутренние энкодеры.

        Внутренние энкодеры создаются в :meth:`__init__`, поэтому изменение
        параметров через ``set_params`` (механизм sklearn: GridSearchCV,
        кастомные пайплайны) требует их пересоздания — иначе поведение
        расходится с новыми значениями параметров (например,
        ``set_params(encoding='target')`` оставило бы ``OneHotEncoder``).

        Args:
            **params: Параметры конструктора (см. :meth:`__init__`).

        Returns:
            Обновлённый энкодер.
        """
        super().set_params(**params)
        self.default_encoder_ = self._make_encoder(self.encoding)
        self.hc_encoder_ = (
            self._make_encoder(self.high_cardinality_encoding)
            if self.high_cardinality_encoding is not None
            else None
        )
        return self

    def fit(self, X: Any, y: Any = None) -> SplitCategoricalEncoder:
        """Разделить колонки по кардинальности и обучить энкодеры.

        Args:
            X: Матрица категориальных признаков (строки после импутации).
            y: Целевая переменная (требуется только для target encoding).

        Returns:
            Обученный энкодер.
        """
        X_arr = _as_2d(X)
        n_columns = X_arr.shape[1]
        cardinalities = [len(np.unique(X_arr[:, col])) for col in range(n_columns)]
        self.cardinalities_ = cardinalities

        threshold = self.high_cardinality_threshold
        default_columns: list[int] = []
        hc_columns: list[int] = []
        if threshold is not None:
            for col, card in enumerate(cardinalities):
                if card > threshold:
                    hc_columns.append(col)
                else:
                    default_columns.append(col)
        else:
            default_columns = list(range(n_columns))
        self.default_columns_ = default_columns
        self.hc_columns_ = hc_columns

        if default_columns:
            self.default_encoder_.fit(X_arr[:, default_columns], y)

        if hc_columns:
            if self.high_cardinality_encoding is None or self.hc_encoder_ is None:
                raise ValueError(
                    "high_cardinality_encoding must be set when "
                    "high_cardinality_threshold is provided."
                )
            self.hc_encoder_.fit(X_arr[:, hc_columns], y)

        logger.debug(
            "SplitCategoricalEncoder: %d default columns, %d high-cardinality "
            "columns (threshold=%r), cardinalities=%s",
            len(default_columns),
            len(hc_columns),
            threshold,
            cardinalities,
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Закодировать колонки выбранными стратегиями и склеить результат.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Числовая матрица, объединяющая выходы default- и
            high-cardinality энкодеров.
        """
        check_is_fitted(self, attributes=["default_columns_"])
        X_arr = _as_2d(X)
        parts: list[np.ndarray] = []
        if self.default_columns_:
            parts.append(
                self.default_encoder_.transform(X_arr[:, self.default_columns_])
            )
        if self.hc_columns_ and self.hc_encoder_ is not None:
            parts.append(self.hc_encoder_.transform(X_arr[:, self.hc_columns_]))
        if not parts:
            return np.empty((X_arr.shape[0], 0), dtype=np.float64)
        return np.hstack(parts)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Вернуть имена выходных признаков в порядке конкатенации."""
        check_is_fitted(self, attributes=["default_columns_"])
        n_columns = len(self.default_columns_) + len(self.hc_columns_)
        base = _feature_names(input_features, n_columns)
        default_names = list(self.default_columns_)
        hc_names = list(self.hc_columns_)
        names: list[str] = []
        if default_names:
            names.extend(
                self.default_encoder_.get_feature_names_out(
                    [base[i] for i in default_names]
                )
            )
        if hc_names and self.hc_encoder_ is not None:
            names.extend(
                self.hc_encoder_.get_feature_names_out([base[i] for i in hc_names])
            )
        return np.asarray(names, dtype=object)


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Автоматически классифицировать колонки на числовые и категориальные.

    Категориальными считаются колонки с dtype ``object``, ``category`` и
    ``bool``; числовыми — колонки с dtype из семейства ``number``.

    Args:
        X: DataFrame признаков (без целевой переменной).

    Returns:
        Кортеж ``(categorical_features, numerical_features)`` — списки имён
        колонок каждого типа.

    Note:
        Функция ожидает именно ``pd.DataFrame``; для ``np.ndarray`` без имён
        колонок автоопределение невозможно (задокументированное ограничение).

    Note:
        Классификация основана на ``dtype`` колонок. Это осознанно отличается от
        value-based-детекции в ``oversampling._is_categorical_col``: здесь
        препроцессор обрабатывает данные до оверсэмплинга, поэтому ``bool`` и
        'числовые строки' (``'10','20'``) рассматриваются как категории (one-hot).
        Оверсэмплер же получает уже закодированные числовые признаки и применяет
        собственную value-based-логику для standalone-вызовов.
    """
    categorical = X.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical, numerical


def build_preprocessor(
    feature_names: list[str],
    categorical_features: list[str],
    numerical_features: list[str],
    encoding: EncodingStrategy = "one_hot",
    imputation_strategy: ImputationStrategy = "mean",
    scaling: ScalingType = "standard",
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    random_state: int | None = 42,
) -> ColumnTransformer:
    """Сконструировать ColumnTransformer для раздельной обработки типов данных.

    Для категориальных признаков поддерживаются стратегии (issue #20):

    * ``'one_hot'`` (по умолчанию) — ``OneHotEncoder``;
    * ``'ordinal'`` — ``OrdinalEncoder`` (1 выходной столбец на колонку);
    * ``'target'`` — target encoding со сглаживанием
      (:class:`TargetEncodingTransformer`);
    * ``'frequency'`` — frequency encoding
      (:class:`FrequencyEncodingTransformer`);
    * ``'hashing'`` — детерминированное хеширование в фиксированное число
      колонок (:class:`HashingEncodingTransformer`).

    При заданных ``high_cardinality_threshold`` и ``high_cardinality_encoding``
    включается автоматический режим: колонки с числом уникальных значений
    строго больше порога кодируются high-cardinality стратегией, остальные —
    стратегией ``encoding``. Разделение вычисляется по обучающим данным внутри
    :class:`SplitCategoricalEncoder`, поэтому единая логика применяется и в HPO,
    и в финальном обучении, и при предсказании.

    Статистики target/frequency-кодирования вычисляются только по обучающей
    части (в ``fit``), что исключает утечку данных (data leakage), включая
    кросс-валидацию: на каждом фолде препроцессор переобучается на обучающей
    части фолда.

    Стратегия предобработки числовых признаков задаётся параметрами
    ``imputation_strategy`` и ``scaling`` и обычно берётся из адаптивного
    пресета предобработки (:mod:`preprocessing_presets`).

    Args:
        feature_names: Полный список имён признаков в порядке следования колонок.
        categorical_features: Имена колонок, кодируемых выбранной стратегией.
        numerical_features: Имена колонок, подлежащих импутации и скалированию.
        encoding: Стратегия кодирования категорий: ``'one_hot'`` (по умолчанию),
            ``'ordinal'``, ``'target'``, ``'frequency'`` или ``'hashing'``.
        imputation_strategy: Стратегия заполнения пропусков для числовых
            признаков: ``'mean'`` (по умолчанию) или ``'median'`` (устойчива
            к выбросам, учитывает скошенность распределений).
        scaling: Масштабирование числовых признаков: ``'standard'``
            (``StandardScaler``, по умолчанию), ``'robust'`` (``RobustScaler``,
            устойчив к выбросам) или ``'none'`` (масштабирование не
            применяется — данные передаются в модель без него).
        high_cardinality_threshold: Порог кардинальности (``>= 0``). Колонки с
            числом уникальных значений строго больше порога считаются
            high-cardinality и кодируются стратегией ``high_cardinality_encoding``.
            ``None`` (по умолчанию) — автоматический режим отключён.
        high_cardinality_encoding: Стратегия кодирования high-cardinality колонок.
            Задаётся вместе с ``high_cardinality_threshold``.
        hashing_n_components: Число бинарных колонок на одну категориальную
            колонку при ``encoding='hashing'`` (``>= 1``).
        target_encoding_smoothing: Параметр сглаживания target encoding
            (``>= 0``); при ``m=0`` используется чистое среднее по категории.
        target_encoding_fallback: Fallback-значение target encoding для
            неизвестных категорий. ``None`` — глобальное среднее целевой
            переменной.
        random_state: Зерно для детерминированного hashing-кодирования.

    Returns:
        ``ColumnTransformer``, преобразующий исходный DataFrame в числовую
        матрицу, готовую к подаче в модель. Если ни одна колонка не совпала,
        возвращается passthrough-трансформер (поведение сохранено из тренера).

    Note:
        Категориальные колонки приводятся к строковому объектному массиву перед
        импутацией и кодированием, поэтому ``bool``-колонки обрабатываются
        корректно (без падения ``SimpleImputer`` на dtype bool).

    Note:
        При ``encoding='ordinal'`` категории кодируются целочисленными кодами,
        которые НЕ масштабируются (в отличие от числовых колонок, проходящих
        через скалер). Для линейных моделей (например, ``elasticnet``)
        несопоставимый масштаб кодов с категориями высокого порядка может
        доминировать над числовыми признаками.

    Raises:
        ValueError: Если имя колонки отсутствует в ``feature_names`` либо
            передано невалидное значение ``encoding``, ``imputation_strategy``,
            ``scaling``, нарушена согласованность high-cardinality параметров
            или некорректны ``hashing_n_components``/``target_encoding_smoothing``.
    """
    if encoding not in _VALID_ENCODING_STRATEGIES:
        raise ValueError(
            f"Unknown encoding strategy: {encoding!r}. "
            f"Expected one of {_VALID_ENCODING_STRATEGIES}."
        )
    if (
        high_cardinality_encoding is not None
        and high_cardinality_encoding not in _VALID_ENCODING_STRATEGIES
    ):
        raise ValueError(
            f"Unknown high_cardinality_encoding: {high_cardinality_encoding!r}. "
            f"Expected one of {_VALID_ENCODING_STRATEGIES}."
        )
    if (high_cardinality_threshold is None) != (high_cardinality_encoding is None):
        raise ValueError(
            "high_cardinality_threshold and high_cardinality_encoding must be "
            "set together (both provided or both None)."
        )
    if high_cardinality_threshold is not None and high_cardinality_threshold < 0:
        raise ValueError(
            f"high_cardinality_threshold must be >= 0, got "
            f"{high_cardinality_threshold}."
        )
    if hashing_n_components < 1:
        raise ValueError(
            f"hashing_n_components must be >= 1, got {hashing_n_components}."
        )
    if target_encoding_smoothing < 0:
        raise ValueError(
            f"target_encoding_smoothing must be >= 0, got {target_encoding_smoothing}."
        )
    if imputation_strategy not in ("mean", "median"):
        raise ValueError(
            f"Unknown imputation strategy: {imputation_strategy!r}. "
            "Expected one of ('mean', 'median')."
        )
    if scaling not in ("standard", "robust", "none"):
        raise ValueError(
            f"Unknown scaling type: {scaling!r}. "
            "Expected one of ('standard', 'robust', 'none')."
        )

    # Сопоставляем имена колонок с порядковыми номерами
    cat_indices = [
        feature_names.index(col) for col in categorical_features if col in feature_names
    ]
    num_indices = [
        feature_names.index(col) for col in numerical_features if col in feature_names
    ]

    if not cat_indices and not num_indices:
        logger.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
        return ColumnTransformer(
            [("bypass", "passthrough", slice(None))], remainder="drop"
        )

    # Пайплайн трансформации числовых признаков: импутация + (опционально) скалер.
    # При scaling='none' шаг масштабирования отсутствует — данные передаются
    # в модель без него (AC-3 для деревьев и ансамблей).
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=imputation_strategy))
    ]
    if scaling != "none":
        scaler = RobustScaler() if scaling == "robust" else StandardScaler()
        num_steps.append(("scaler", scaler))
    num_transformer = Pipeline(steps=num_steps)

    cat_transformer = Pipeline(
        steps=[
            # Приводим категориальные колонки к строковому объектному массиву.
            # Это делает пайплайн устойчивым к bool-колонкам: SimpleImputer со
            # strategy="most_frequent" падает на numpy-bool ("SimpleImputer does
            # not support data with dtype bool"), поэтому bool-признаки приводятся
            # к строкам ("True"/"False") и кодируются как обычные категории.
            ("to_string", FunctionTransformer(_to_string_array)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                SplitCategoricalEncoder(
                    encoding=encoding,
                    high_cardinality_threshold=high_cardinality_threshold,
                    high_cardinality_encoding=high_cardinality_encoding,
                    hashing_n_components=hashing_n_components,
                    target_encoding_smoothing=target_encoding_smoothing,
                    target_encoding_fallback=target_encoding_fallback,
                    random_state=random_state,
                ),
            ),
        ]
    )

    transformers = []
    if cat_indices:
        transformers.append(("cat", cat_transformer, cat_indices))
    if num_indices:
        transformers.append(("num", num_transformer, num_indices))

    return ColumnTransformer(
        transformers=(transformers if transformers else [("pass", "passthrough", [0])]),
        remainder="drop",
    )
