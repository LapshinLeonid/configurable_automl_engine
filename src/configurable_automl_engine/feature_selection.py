"""Селектор признаков для сокращения пространства признаков (issue #29).

Модуль реализует :class:`FeatureSelector` — независимый sklearn-трансформер
(``BaseEstimator`` + ``TransformerMixin``), который выполняет сокращение
пространства признаков по одному из четырёх алгоритмов:

    * **importance** — ``SelectFromModel`` на основе важности признаков
      деревянного ансамбля (``ExtraTreesRegressor``), порог отбора — средняя
      важность (поведение ``SelectFromModel`` по умолчанию);
    * **percentile** — ``SelectPercentile`` со скорингом ``f_regression``
      (линейные зависимости);
    * **mutual_info** — ``SelectPercentile`` со скорингом
      ``mutual_info_regression`` (захватывает нелинейные зависимости);
    * **variance** — ``VarianceThreshold`` (отсечение константных и
      низковариативных колонок).

Трансформер гарантирует:

    1. Сохранение типа матрицы при трансформации: ``np.ndarray`` остаётся
       плотным массивом, ``scipy.sparse.csr_matrix`` — разреженной матрицей
       (срез выполняется без ``.toarray()``, что исключает OOM на больших
       данных), ``pd.DataFrame`` — DataFrame'ом.
    2. Защиту от опустошения признакового пространства
       (``min_features_guard``): если базовый селектор отобрал меньше
       ``min_features`` признаков (вплоть до нуля — все признаки признаны
       неважными), маска поддержки принудительно активирует
       ``top-min(min_features, P)`` признаков с максимальными оценками.
    3. Режим passthrough при ``P <= min_features``: отбор не выполняется,
       маска ``support_`` оставляет все входные признаки.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_VALID_METHODS: tuple[str, ...] = (
    "importance",
    "percentile",
    "mutual_info",
    "variance",
)

# Методы, которым в fit() обязательно нужен таргет y.
_TARGET_METHODS: tuple[str, ...] = ("importance", "percentile", "mutual_info")


class FeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Трансформер сокращения пространства признаков с несколькими бэкендами.

    Выполняет отбор признаков по выбранному алгоритму (``method``), сохраняет
    тип матрицы при трансформации (dense / sparse / DataFrame) и защищает
    пайплайн от опустошения признакового пространства:

    * если число входных признаков ``P <= min_features``, трансформер
      работает в режиме passthrough (отбор не выполняется);
    * если базовый селектор отобрал ``K < min_features`` признаков (включая
      ``K = 0``), маска ``support_`` принудительно активирует
      ``top-min(min_features, P)`` признаков с максимальными оценками
      (важностями/скорами/дисперсиями).

    Args:
        method: Алгоритм отбора: ``'importance'`` (важность деревьев),
            ``'percentile'`` (f-статистика регрессии), ``'mutual_info'``
            (взаимная информация, нелинейные зависимости) или ``'variance'``
            (порог дисперсии).
        percentile: Доля признаков, сохраняемых методами ``'percentile'`` и
            ``'mutual_info'``, в процентах (``0 < percentile <= 100``).
        min_features: Абсолютная нижняя граница числа сохраняемых признаков
            (``>= 1``). При ``P <= min_features`` отбор не выполняется.
        variance_threshold: Порог дисперсии для метода ``'variance'``: колонки
            с дисперсией ``<= threshold`` удаляются.
        n_estimators: Число деревьев в ``ExtraTreesRegressor`` для метода
            ``'importance'``.
        random_state: Зерно генератора случайных чисел для воспроизводимости
            (``'importance'`` и ``'mutual_info'``).

    Attributes:
        support_: Булева маска отобранных признаков длины ``n_features``
            (доступна после ``fit``).
        n_selected_: Число отобранных признаков.
        base_selector_: Обученный базовый селектор sklearn либо ``None`` в
            режиме passthrough.
        n_features_in_: Число входных признаков.
        feature_names_in_: Имена колонок (для ``pd.DataFrame`` на входе).
    """

    def __init__(
        self,
        method: str = "importance",
        percentile: float = 50.0,
        min_features: int = 2,
        variance_threshold: float = 0.0,
        n_estimators: int = 50,
        random_state: int | None = 42,
    ) -> None:
        self.method = method
        self.percentile = percentile
        self.min_features = min_features
        self.variance_threshold = variance_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> FeatureSelector:
        """Обучить селектор на обучающих данных.

        Args:
            X: Матрица признаков ``(n_samples, n_features)``. Поддерживаются
                ``np.ndarray``, ``pd.DataFrame`` и разреженные
                ``scipy.sparse``-матрицы (``csr_matrix`` и др.).
            y: Целевая переменная. Обязательна для методов ``'importance'``,
                ``'percentile'`` и ``'mutual_info'``; для ``'variance'``
                игнорируется.

        Returns:
            Обученный трансформер (``self``). Отобранные признаки доступны
            через :meth:`get_support` / атрибут ``support_``.

        Raises:
            ValueError: Если задан неизвестный ``method``, отсутствует ``y``
                для методов, требующих таргет, либо некорректны
                ``percentile``/``min_features``.
        """
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method: {self.method!r}. Expected one of {_VALID_METHODS}."
            )
        if self.method in _TARGET_METHODS and y is None:
            raise ValueError(f"Method {self.method!r} requires y during fit().")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}.")
        if not 0.0 < self.percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {self.percentile}.")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")

        if y is None:
            X_arr = check_array(X, accept_sparse=True, dtype="numeric")
            y_arr = None
        else:
            X_arr, y_arr = check_X_y(X, y, accept_sparse=True, dtype="numeric")

        self.n_features_in_ = X_arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)

        n_features = X_arr.shape[1]

        # Edge case: P <= min_features -> passthrough без отбора.
        if n_features <= self.min_features:
            self.support_ = np.ones(n_features, dtype=bool)
            self.n_selected_ = n_features
            self.base_selector_ = None
            return self

        # sklearn считает признаки разреженных матриц дискретными, и
        # mutual_info_regression падает невнятной ошибкой на непрерывных
        # значениях — выдаём понятное сообщение заранее.
        if self.method == "mutual_info" and sp.issparse(X_arr):
            _validate_sparse_mutual_info(X_arr)

        if self.method == "variance":
            support, k_selected = self._fit_variance(X_arr, n_features)
        else:
            assert y_arr is not None
            support, k_selected = self._fit_target_based(X_arr, y_arr, n_features)

        self.support_ = support
        self.n_selected_ = k_selected
        return self

    def _fit_variance(self, X_arr: Any, n_features: int) -> tuple[np.ndarray, int]:
        """Обучить ветку ``'variance'`` с обработкой крайнего случая ``K = 0``.

        ``VarianceThreshold.fit`` падает с ``ValueError``, когда ни один
        признак не превышает порог (удаляются все колонки). Такая ситуация
        штатно обрабатывается ``min_features_guard``: дисперсии считаются
        заранее **тем же алгоритмом, что и sklearn** (см.
        :func:`_compute_variances`), поэтому решение о вызове
        ``VarianceThreshold`` всегда совпадает с его собственным, и падение
        исключено. При пустом отборе guard сразу активирует
        ``top-min(min_features, P)`` признаков с максимальной дисперсией.
        """
        variances = _compute_variances(X_arr)
        if np.any(variances > self.variance_threshold):
            selector = self._build_selector()
            selector.fit(X_arr)
            self.base_selector_ = selector
            support = np.asarray(selector.get_support(), dtype=bool)
            k_selected = int(np.count_nonzero(support))
            if k_selected < self.min_features:
                support, k_selected = self._apply_guard(
                    self._extract_scores(selector), n_features
                )
        else:
            self.base_selector_ = None
            support, k_selected = self._apply_guard(variances, n_features)
        return support, k_selected

    def _fit_target_based(
        self, X_arr: Any, y_arr: Any, n_features: int
    ) -> tuple[np.ndarray, int]:
        """Обучить ветки, требующие таргет (importance/percentile/mutual_info)."""
        selector = self._build_selector()
        selector.fit(X_arr, y_arr)
        self.base_selector_ = selector
        support = np.asarray(selector.get_support(), dtype=bool)
        k_selected = int(np.count_nonzero(support))
        if k_selected < self.min_features:
            scores = self._extract_scores(selector)
            support, k_selected = self._apply_guard(scores, n_features)
        return support, k_selected

    def _build_selector(self) -> Any:
        """Создать базовый селектор sklearn по выбранному методу.

        Метод уже провалидирован в :meth:`fit`, поэтому для некорректных
        значений эта ветка недостижима.
        """
        if self.method == "importance":
            return SelectFromModel(
                ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                ),
                threshold="mean",
            )
        if self.method == "percentile":
            return SelectPercentile(score_func=f_regression, percentile=self.percentile)
        if self.method == "mutual_info":
            # Зерно фиксирует стохастику kNN-оценки взаимной информации.
            score_func = partial(mutual_info_regression, random_state=self.random_state)
            return SelectPercentile(score_func=score_func, percentile=self.percentile)
        if self.method == "variance":
            return VarianceThreshold(threshold=self.variance_threshold)
        raise AssertionError(
            f"Unreachable: method {self.method!r} must be validated in fit()."
        )

    def _extract_scores(self, selector: Any) -> np.ndarray:
        """Извлечь оценки признаков из обученного базового селектора."""
        if self.method == "importance":
            return np.asarray(selector.estimator_.feature_importances_, dtype=float)
        if self.method in ("percentile", "mutual_info"):
            return np.asarray(selector.scores_, dtype=float)
        if self.method == "variance":
            return np.asarray(selector.variances_, dtype=float)
        raise AssertionError(
            f"Unreachable: method {self.method!r} must be validated in fit()."
        )

    def _apply_guard(
        self, scores: np.ndarray, n_features: int
    ) -> tuple[np.ndarray, int]:
        """Принудительно активировать ``top-min(min_features, P)`` признаков.

        Args:
            scores: Оценки признаков (важности, скоры или дисперсии); большее
                значение — «лучше».
            n_features: Общее число входных признаков ``P``.

        Returns:
            Кортеж ``(support_, n_selected)``: булева маска с ровно
            ``min(min_features, P)`` активными позициями.
        """
        top_k = min(self.min_features, n_features)
        top_indices = _top_indices(scores, top_k)
        support = np.zeros(n_features, dtype=bool)
        support[top_indices] = True
        return support, int(np.count_nonzero(support))

    def transform(self, X: Any) -> np.ndarray | sp.spmatrix | pd.DataFrame:
        """Сократить пространство признаков по обученной маске ``support_``.

        Тип матрицы сохраняется: ``np.ndarray`` -> ``np.ndarray``,
        ``scipy.sparse`` -> ``scipy.sparse`` (срез ``X[:, support_]`` без
        ``.toarray()``), ``pd.DataFrame`` -> ``pd.DataFrame``.

        Args:
            X: Матрица признаков ``(n_samples, n_features)`` того же типа,
                что и на этапе ``fit``.

        Returns:
            Матрица того же типа с колонками, отобранными в :meth:`fit`.

        Raises:
            sklearn.exceptions.NotFittedError: Если трансформер не обучен.
            ValueError: Если число колонок ``X`` не совпадает с обучающим,
                либо имена колонок ``pd.DataFrame`` не совпадают с
                ``feature_names_in_``.
        """
        check_is_fitted(self, attributes=["support_"])
        n_input = X.shape[1]
        if n_input != len(self.support_):
            raise ValueError(
                f"X has {n_input} features, but FeatureSelector was fitted "
                f"with {len(self.support_)} features."
            )
        if isinstance(X, pd.DataFrame):
            # Сверка имён, как в sklearn: при том же числе колонок, но другом
            # порядке имён позиционный срез тихо взял бы не те признаки.
            if hasattr(self, "feature_names_in_") and list(X.columns) != list(
                self.feature_names_in_
            ):
                raise ValueError(
                    "The feature names of the input DataFrame do not match "
                    "those seen during fit. Expected "
                    f"{list(self.feature_names_in_)}, got {list(X.columns)}."
                )
            return X.iloc[:, self.support_]
        if sp.issparse(X):
            # Срез выполняется в CSR (COO не поддерживает индексацию), затем
            # результат возвращается в исходном формате: CSR -> CSR,
            # CSC -> CSC, COO -> COO и т.д. Без .toarray(): плотная копия
            # не создаётся (исключает OOM). scipy-stubs не типизируют
            # индексацию и динамический to<format> для SparseABC — к Any.
            sparse_X = cast(Any, X)
            sliced = sparse_X.tocsr()[:, self.support_]
            return getattr(sliced, "to" + sparse_X.format)()
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Вернуть маску (или индексы) отобранных признаков.

        Args:
            indices: ``True`` — вернуть индексы отобранных признаков вместо
                булевой маски.

        Returns:
            Булев массив длины ``n_features`` либо массив целочисленных
            индексов отобранных признаков.
        """
        check_is_fitted(self, attributes=["support_"])
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Вернуть имена выходных (отобранных) признаков.

        Args:
            input_features: Имена входных признаков. При ``None``
                используются имена, сохранённые в :meth:`fit` для
                ``pd.DataFrame`` (``feature_names_in_``), либо позиционные
                имена ``x0...xN``.

        Returns:
            Массив имён только для отобранных признаков.

        Raises:
            ValueError: Если длина ``input_features`` не совпадает с числом
                входных признаков.
        """
        check_is_fitted(self, attributes=["support_"])
        n_features = len(self.support_)
        if input_features is None and hasattr(self, "feature_names_in_"):
            input_features = self.feature_names_in_
        if input_features is None:
            names = np.asarray([f"x{i}" for i in range(n_features)], dtype=object)
        else:
            features = np.asarray(input_features, dtype=object)
            if len(features) != n_features:
                raise ValueError(
                    f"input_features has {len(features)} elements, "
                    f"expected {n_features}."
                )
            names = features
        return names[self.support_]


def _compute_variances(X_arr: Any) -> np.ndarray:
    """Дисперсия каждой колонки — тем же способом, что и ``VarianceThreshold``.

    Для разреженных матриц используется
    ``sklearn.utils.sparsefuncs.mean_variance_axis`` после приведения к
    ``csc float64`` — ровно то, что делает ``VarianceThreshold.fit`` внутри
    (``accept_sparse='csc', dtype=np.float64``). Для плотных — ``np.var`` на
    ``float64``-копии.

    Почему нельзя использовать формулу ``E[x^2] - E[x]^2``: на разреженном
    входе она даёт шумовые ненулевые «дисперсии» порядка ``1e-14`` для
    константных колонок (потеря точности при суммировании квадратов в
    ``scipy.sparse.mean``), из-за чего предрасчёт расходится с собственным
    вычислением sklearn: ``VarianceThreshold.fit`` падает с ``ValueError``
    («No feature in X meets the variance threshold») вместо срабатывания
    ``min_features_guard``.
    """
    if sp.issparse(X_arr):
        # Сначала tocsc(), затем float64 — как в _validate_data VarianceThreshold.
        # scipy-stubs не типизируют tocsc для SparseABC — приводим к Any.
        X_csc = cast(Any, X_arr).tocsc().astype(np.float64)
        _, variances = mean_variance_axis(X_csc, axis=0)
        return np.asarray(variances, dtype=float).ravel()
    return np.var(np.asarray(X_arr, dtype=np.float64), axis=0)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы ``k`` признаков с максимальными оценками.

    Нечисловые оценки (NaN) трактуются как ``-inf`` (не могут попасть в
    топ). Связи разрешаются стабильно по индексу (``argsort(kind='stable')``
    по убыванию): при равных оценках выживают младшие индексы, поэтому
    результат детерминирован даже при полностью равных оценках (например,
    нулевые дисперсии у всех константных колонок).
    """
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    if k >= safe.shape[0]:
        return np.arange(safe.shape[0], dtype=int)
    # argsort по убыванию со stable-сортировкой: при равных оценках
    # первыми идут младшие индексы (детерминированный tie-break).
    return np.argsort(-safe, kind="stable")[:k]


def _validate_sparse_mutual_info(X_arr: Any) -> None:
    """Проверить, что sparse-вход для ``mutual_info`` дискретный.

    sklearn считает все признаки разреженных матриц дискретными
    (``discrete_features='auto'`` -> ``True`` для sparse), а
    ``mutual_info_regression`` на непрерывных значениях падает невнятной
    ошибкой (пустая выборка в kNN-оценке). Вместо неё выдаём понятное
    ``ValueError`` с требованием целочисленных признаков.

    Args:
        X_arr: Разреженная матрица (уже приведена к числовому dtype).

    Raises:
        ValueError: Если среди значений есть нецелые (непрерывные) признаки.
    """
    data = cast(Any, X_arr).data
    if np.issubdtype(X_arr.dtype, np.integer):
        return
    if np.all(np.isfinite(data)) and np.all(data == np.trunc(data)):
        return
    raise ValueError(
        "Method 'mutual_info' requires discrete (integer-valued) features "
        "for sparse input: sklearn treats all features of sparse matrices "
        "as discrete, and continuous values break the kNN-based mutual "
        "information estimation."
    )
