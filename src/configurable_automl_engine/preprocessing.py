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
      Выход разреженный (``scipy.sparse.csr_matrix``), как у стандартного
      ``FeatureHasher`` из sklearn, — это исключает OOM на колонках высокой
      кардинальности больших датасетов. Для алгоритмов, отвергающих sparse
      (GPR/Isotonic/ARD), ``build_preprocessor`` принудительно возвращает
      плотную матрицу (``force_dense_output``).

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
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
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

    Note:
        Ключи статистик сохраняют исходные типы категорий (как и в
        :class:`FrequencyEncodingTransformer`): standalone-вызовы с числовыми
        категориями (``int``-коды, ``float``, ``bool``) корректно находят
        соответствия в :meth:`transform`. Соответствие подбирается pandas по
        совместимости dtype: ``int`` и ``float`` взаимозаменяемы, а ``bool`` —
        отдельный dtype. Из-за стандартной Python-семантики равенства
        (``True == 1``) при смешивании ``bool`` и ``int`` ``0/1`` возможны
        неочевидные результаты, поэтому категории одного признака должны
        сохранять один dtype в ``fit`` и ``transform``. При использовании через
        :func:`build_preprocessor` категории заранее приводятся к строкам
        (шаг ``to_string``), что исключает рассинхрон типов.
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
            Обученный трансформер. Статистики для каждой колонки хранятся в
            ``statistics_`` как ``dict[Any, float]``: ключи повторяют исходные
            значения категорий (типы не нормализуются к строкам).

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
        self.statistics_: list[dict[Any, float]] = []
        for col in range(X_arr.shape[1]):
            group = pd.DataFrame(
                {"cat": pd.Series(X_arr[:, col]), "target": y_float}
            ).groupby("cat")["target"]
            agg = group.agg(["mean", "count"])
            smoothed = (
                agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing
            ) / (agg["count"] + self.smoothing)
            self.statistics_.append(
                {cat: float(value) for cat, value in smoothed.items()}
            )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Заменить категории их target-значениями.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Числовая матрица той же размерности (1 колонка на входную колонку).
            Неизвестные категории отображаются в fallback-значение. Соответствия
            ищутся по dtype-совместимым ключам (см. docstring класса): числовые
            категории в ``X`` должны иметь тот же dtype, что и в обучающей выборке.
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

    Выход :meth:`transform` — разреженная матрица ``scipy.sparse.csr_matrix``
    (nnz = n_rows · n_cols), как у стандартного ``FeatureHasher``: плотная
    ``float64``-матрица размерности ``(n_rows, n_cols · n_components)``
    вызвала бы OOM на реальных таблицах с колонками высокой кардинальности.

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

    def transform(self, X: Any) -> sp.csr_matrix:
        """Закодировать категории бинарными хеш-колонками.

        Возвращает разреженную матрицу ``scipy.sparse.csr_matrix``
        (COO → CSR, ``nnz = n_samples * n_columns``, значения ``1.0``),
        а не плотный массив: hashing предназначен для колонок с очень
        высокой кардинальностью на больших датасетах, и плотная
        ``float64``-матрица размерности ``(n_rows, n_cols * n_components)``
        вызвала бы OOM. Формат соответствует стандартному
        ``sklearn.feature_extraction.FeatureHasher``.

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Разреженная бинарная матрица размерности
            ``(n_samples, n_columns * n_components)``.
        """
        check_is_fitted(self, attributes=["salts_"])
        X_arr = _as_2d(X)
        n_rows, n_cols = X_arr.shape
        # Каждая строка каждой колонки даёт ровно один ненулевой элемент,
        # поэтому nnz = n_rows * n_cols. Коллизии хеша в рамках одной колонки
        # невозможны для одной строки (одна категория на строку), дубликатов
        # в COO не возникает, сумма по строкам всегда равна n_cols (бинарность).
        col_blocks: list[np.ndarray] = []
        for col in range(n_cols):
            uniq, inverse = np.unique(X_arr[:, col], return_inverse=True)
            hashed = np.array(
                [self._hash_token(token, self.salts_[col]) for token in uniq],
                dtype=np.int64,
            )
            col_blocks.append(col * self.n_components + hashed[inverse])
        rows = np.tile(np.arange(n_rows), n_cols)
        cols = np.concatenate(col_blocks) if col_blocks else np.empty(0, dtype=np.int64)
        data = np.ones(n_rows * n_cols, dtype=np.float64)
        out = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n_rows, n_cols * self.n_components),
            dtype=np.float64,
        )
        return out.tocsr()

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

    def _make_encoder(self, strategy: EncodingStrategy) -> BaseEstimator:
        """Создать энкодер по имени стратегии.

        Ветки покрывают все допустимые значения ``EncodingStrategy``;
        валидация стратегии выполняется в :meth:`fit`, поэтому эта ветка
        недостижима для корректных вызовов.
        """
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
        raise AssertionError(
            f"Unreachable: strategy {strategy!r} must be validated in fit(), "
            f"expected one of {_VALID_ENCODING_STRATEGIES}."
        )

    def fit(self, X: Any, y: Any = None) -> SplitCategoricalEncoder:
        """Разделить колонки по кардинальности и обучить энкодеры.

        Внутренние энкодеры (``default_encoder_``/``hc_encoder_``) создаются
        только здесь, в :meth:`fit`: атрибуты с завершающим подчёркиванием
        хранят обученное состояние, поэтому конструктор лишь сохраняет
        параметры, а ``clone()``/``set_params()``/``get_params()`` работают
        через базовую реализацию ``BaseEstimator`` без переопределений.

        Args:
            X: Матрица категориальных признаков (строки после импутации).
            y: Целевая переменная (требуется только для target encoding).

        Returns:
            Обученный энкодер.

        Raises:
            ValueError: Если задана неизвестная стратегия кодирования либо
                high-cardinality режим настроен некорректно.
        """
        if self.encoding not in _VALID_ENCODING_STRATEGIES:
            raise ValueError(
                f"Unknown encoding strategy: {self.encoding!r}. "
                f"Expected one of {_VALID_ENCODING_STRATEGIES}."
            )
        if (
            self.high_cardinality_encoding is not None
            and self.high_cardinality_encoding not in _VALID_ENCODING_STRATEGIES
        ):
            raise ValueError(
                f"Unknown high_cardinality_encoding: "
                f"{self.high_cardinality_encoding!r}. "
                f"Expected one of {_VALID_ENCODING_STRATEGIES}."
            )
        if (
            self.high_cardinality_threshold is not None
            and self.high_cardinality_encoding is None
        ):
            raise ValueError(
                "high_cardinality_encoding must be set when "
                "high_cardinality_threshold is provided."
            )

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

        # Внутренние энкодеры создаются и обучаются здесь, в fit(): атрибуты с
        # завершающим подчёркиванием появляются только после обучения.
        self.default_encoder_ = self._make_encoder(self.encoding)
        self.hc_encoder_: BaseEstimator | None = (
            self._make_encoder(self.high_cardinality_encoding)
            if self.high_cardinality_encoding is not None
            else None
        )

        if default_columns:
            self.default_encoder_.fit(X_arr[:, default_columns], y)

        if hc_columns:
            # Инвариант гарантирован валидацией выше: при заданном пороге
            # HC-стратегия обязана быть указана, значит hc_encoder_ создан.
            assert self.hc_encoder_ is not None
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

    def transform(self, X: Any) -> np.ndarray | sp.spmatrix:
        """Закодировать колонки выбранными стратегиями и склеить результат.

        Если хотя бы один из энкодеров вернул разреженную матрицу
        (hashing-кодирование), части склеиваются через ``sp.hstack``
        (CSR); иначе — через ``np.hstack`` (плотный случай).

        Args:
            X: Матрица категориальных признаков.

        Returns:
            Числовая матрица (плотная или разреженная), объединяющая
            выходы default- и high-cardinality энкодеров.
        """
        check_is_fitted(self, attributes=["default_columns_"])
        X_arr = _as_2d(X)
        parts: list[np.ndarray | sp.spmatrix] = []
        if self.default_columns_:
            parts.append(
                self.default_encoder_.transform(X_arr[:, self.default_columns_])
            )
        if self.hc_columns_ and self.hc_encoder_ is not None:
            parts.append(self.hc_encoder_.transform(X_arr[:, self.hc_columns_]))
        if not parts:
            return np.empty((X_arr.shape[0], 0), dtype=np.float64)
        if any(sp.issparse(part) for part in parts):
            # scipy-stubs не описывают гетерогенные последовательности
            # (ndarray | spmatrix) для hstack: приводим блоки к общему типу.
            sparse_blocks = cast(list[sp.spmatrix], parts)
            return sp.hstack(sparse_blocks).tocsr()
        # Все части плотные (sparse-ветка выше) — обычный np.hstack.
        dense_blocks = cast(list[np.ndarray], parts)
        return np.hstack(dense_blocks)

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


class ColumnNameSelector:
    """Сериализуемый селектор колонок по именам для ``ColumnTransformer``.

    Заменяет списки позиционных индексов при сборке препроцессора (issue #2):
    ``ColumnTransformer`` трактует список ``int`` как срез по позициям и
    игнорирует имена колонок ``pd.DataFrame``, из-за чего предсказание на
    переупорядоченном DataFrame молча применяет энкодер к числовым колонкам,
    а скалер — к категориальным.

    Селектор — picklable-объект (обычный класс с простыми атрибутами, не
    замыкание), поэтому он корректно переживает ``trainer.save()``/pickle.
    На каждом вызове (sklearn резолвит callable-селекторы на этапе
    ``fit_transform``/``transform``) имена сопоставляются с позициями по
    фактическому входу ``X``:

    * ``pd.DataFrame`` — сопоставление по имени колонки; порядок колонок не
      важен, отсутствующие или дублирующиеся колонки дают явную ошибку;
    * любые другие входы (``np.ndarray``, ``pd.Series`` и т.п.) — позиционный
      fallback по обучающему порядку (``feature_names``).

    Args:
        feature_names: Полный список имён признаков в обучающем порядке.
        column_names: Имена колонок, которые должен выбирать этот селектор.

    Raises:
        ValueError: Если для ``pd.DataFrame`` часть ``column_names``
            отсутствует либо встречается более одного раза, или имя не
            разрешается в обучающем порядке для не-DataFrame входов.
    """

    def __init__(self, feature_names: list[str], column_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.column_names = list(column_names)

    def __call__(self, X: Any) -> list[int]:
        if isinstance(X, pd.DataFrame):
            return _resolve_column_positions_by_name(X, self.column_names)
        return _resolve_column_positions_by_order(self.feature_names, self.column_names)


def _resolve_column_positions_by_name(
    X: pd.DataFrame, column_names: list[str]
) -> list[int]:
    """Сопоставить имена колонок с позициями в фактическом DataFrame.

    Args:
        X: Входной DataFrame, против которого выполняется сопоставление.
        column_names: Имена колонок для выбора.

    Returns:
        Список позиций выбранных колонок в порядке ``column_names``.

    Raises:
        ValueError: Если часть имён отсутствует в ``X`` либо встречается
            в ``X`` более одного раза (неоднозначный выбор по имени).
    """
    missing = [name for name in column_names if name not in X.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in the input DataFrame. "
            f"Available columns: {list(X.columns)}."
        )
    duplicated = [name for name in column_names if list(X.columns).count(name) > 1]
    if duplicated:
        raise ValueError(
            f"Column(s) {duplicated} appear more than once in the input "
            "DataFrame; ambiguous selection by name is not supported."
        )
    # После проверки уникальности index() даёт точную позицию колонки
    # (pandas-stubs: get_loc может вернуть slice/маску для дубликатов).
    return [list(X.columns).index(name) for name in column_names]


def _resolve_column_positions_by_order(
    feature_names: list[str], column_names: list[str]
) -> list[int]:
    """Сопоставить имена колонок с позициями по обучающему порядку.

    Используется как fallback для входов без имён колонок (``np.ndarray``):
    позиция имени определяется его положением в ``feature_names``.

    Args:
        feature_names: Полный список имён признаков в обучающем порядке.
        column_names: Имена колонок для выбора.

    Returns:
        Список позиций выбранных колонок в порядке ``column_names``.

    Raises:
        ValueError: Если имя не встречается в ``feature_names`` ровно один раз.
    """
    positions: list[int] = []
    for name in column_names:
        occurrences = feature_names.count(name)
        if occurrences != 1:
            raise ValueError(
                f"Cannot resolve column {name!r} by training order: it occurs "
                f"{occurrences} time(s) in feature_names {feature_names}."
            )
        positions.append(feature_names.index(name))
    return positions


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
    force_dense_output: bool = False,
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
        force_dense_output: Принудительно вернуть плотную матрицу вместо
            разреженной. Включается для алгоритмов, отвергающих
            ``scipy.sparse`` (GPR/Isotonic/ARD, см.
            :func:`~configurable_automl_engine.models.requires_dense_input`):
            с ``sparse_threshold=0.0`` ``ColumnTransformer`` конвертирует
            разреженные части в плотные. По умолчанию ``False`` — выход
            остаётся разреженным при hashing-кодировании (экономия памяти).

    Returns:
        ``ColumnTransformer``, преобразующий исходный DataFrame в числовую
        матрицу, готовую к подаче в модель. Если ни одна колонка не совпала,
        возвращается passthrough-трансформер (поведение сохранено из тренера).

    Note:
        Категориальные колонки приводятся к строковому объектному массиву перед
        импутацией и кодированием, поэтому ``bool``-колонки обрабатываются
        корректно (без падения ``SimpleImputer`` на dtype bool).

    Note:
        Колонки выбираются по имени через :class:`ColumnNameSelector` (issue #2):
        для ``pd.DataFrame`` порядок колонок на ``fit``/``transform`` не важен —
        энкодер всегда применяется к категориальным колонкам, а скалер — к
        числовым; отсутствующие/дублирующиеся колонки дают явную ошибку. Для
        входов без имён (``np.ndarray``) используется позиционный fallback по
        обучающему порядку ``feature_names``.

    Note:
        При ``encoding='ordinal'`` категории кодируются целочисленными кодами,
        которые НЕ масштабируются (в отличие от числовых колонок, проходящих
        через скалер). Для линейных моделей (например, ``elasticnet``)
        несопоставимый масштаб кодов с категориями высокого порядка может
        доминировать над числовыми признаками.

    Raises:
        ValueError: Если имя колонки из ``categorical_features``/
            ``numerical_features`` отсутствует в ``feature_names`` либо передано
            невалидное значение ``encoding``, ``imputation_strategy``,
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

    # Валидируем, что все запрошенные имена колонок присутствуют в обучающем
    # наборе (feature_names). Раньше отсутствующие имена молча пропускались;
    # теперь это явная ошибка — селектор по именам не сможет их разрешить.
    specified_features = list(categorical_features) + list(numerical_features)
    missing_features = [col for col in specified_features if col not in feature_names]
    if missing_features:
        raise ValueError(
            f"Unknown feature name(s): {missing_features}. "
            f"Expected columns: {feature_names}."
        )

    # Селекторы по именам (issue #2): для pd.DataFrame колонки выбираются по
    # имени на каждом fit/transform (порядок колонок не важен), для входов без
    # имён (np.ndarray) используется позиционный fallback по обучающему порядку.
    # Селекторы — picklable-классы, поэтому trainer.save()/pickle работают.
    cat_selector = ColumnNameSelector(feature_names, list(categorical_features))
    num_selector = ColumnNameSelector(feature_names, list(numerical_features))

    if not categorical_features and not numerical_features:
        logger.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
        return ColumnTransformer(
            [("bypass", "passthrough", slice(None))],
            remainder="drop",
            sparse_threshold=0.0 if force_dense_output else 0.3,
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
    if categorical_features:
        transformers.append(("cat", cat_transformer, cat_selector))
    if numerical_features:
        transformers.append(("num", num_transformer, num_selector))

    return ColumnTransformer(
        transformers=(transformers if transformers else [("pass", "passthrough", [0])]),
        remainder="drop",
        # При force_dense_output всегда возвращаем плотную матрицу:
        # разреженные части (hashing) конвертируются в ndarray, иначе —
        # стандартный sparse_threshold=0.3 (csr при любой sparse-части).
        sparse_threshold=0.0 if force_dense_output else 0.3,
    )
