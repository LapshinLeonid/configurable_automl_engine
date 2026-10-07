"""Модуль управления жизненным циклом обучения моделей (Training Engine).
Обеспечивает автоматизированный процесс от валидации входных данных до
сохранения финальной модели. Поддерживает многофазовый поиск гиперпараметров
(HPO), динамическую загрузку алгоритмов и параллельное выполнение вычислений.
Публичный цикл обучения декомпозирован на шаги (issue #31):
    - load_config: Загрузка и валидация конфигурации и входных данных.
    - prepare_dataset: Подготовка X/y, резолюция валидации и типов колонок.
    - execute_phases: Многофазовый поиск гиперпараметров (HPO).
    - select_winner: Детерминированный выбор алгоритма-победителя.
    - persist_artifact: Финальное обучение, сохранение модели и сборка отчёта.
Двухстадийный выбор победителя (эпик #61, задача T4, issue #64):
    - select_finalists (T3): пул финалистов — коридор δ + top_k_candidates.
    - select_robust_winner (T4): аудит каждого финалиста через
      ModelSanityGate (T2) и ранжирование прошедших аудит по RMSE_oof;
      режимы off / warn_only / active из конфига general.sanity_gate.
Основные компоненты:
    - train_best_model: Публичный интерфейс — оркестрация перечисленных шагов.
    - _run_hpo: Обертка для поиска гиперпараметров через внешние тюнеры.
    - _fit_and_save: Финальное обучение модели на полном наборе данных.
    - _load_module: Помощник для динамического импорта модулей по пути.
Пример использования:
    results = train_best_model(
        config="path/to/config.yaml",
        df=my_dataframe,
        target="target_col",
        model_path_override="models/best.pkl"
    )
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.common.validation_utils import (
    check_target_exists,
    prepare_X_y,
    validate_df_not_empty,
)
from configurable_automl_engine.preprocessing import detect_feature_types
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    CorridorMode,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    HPOPhaseCfg,
    SanityGateMode,
    ValidationStrategy,
    read_config,
)
from configurable_automl_engine.validation import RANDOM_STATE, make_cv

# ───────────────────────── canonical IAE ─────────────────────── #
from ..tuner import WORST_SCORE_THRESHOLD
from ..tuner import InvalidAlgorithmError as _CanonicalIAE
from .logger import setup_logging
from .metrics import (
    direction_label,
    is_error_metric,
    oof_rmse,
    to_sklearn_name,
    to_user_value,
    user_direction,
)
from .sanity_gate import ModelSanityGate, SanityCheckResult
from .thread_pool import run_parallel

_LOG = logging.getLogger("training_engine")


def is_valid_winner_score(score: Any) -> bool:
    """Проверить, что «сырое» значение скора может быть скором победителя.

    Валидный скор победителя обязан быть:
    - числом (int/float/np.floating): None, строки (в т.ч. числовые, например
      ``"0.5"``) и прочие «мусорные» типы от кастомных тюнеров не проходят
      (issue #32);
    - конечным (NaN и ±inf — сигналы отсутствия валидной метрики);
    - строго выше класса worst-score сентинела: любой скор <= точного
      float32-минимума (``WORST_SCORE_THRESHOLD``) трактуется как сентинел.
      Порог покрывает и константу ``HPO_WORST_SCORE`` (возвращается тюнером
      для нефинитных метрик), и «сырое» значение ``float(np.finfo(np.float32).min)``
      (возможно от numpy-cast метрик или кастомных тюнеров), которое старый
      фильтр через ``math.isclose(rel_tol=1e-9)`` пропускал (относительная
      разница ≈ 9.88e-9 > 1e-9). Реальные метрики такой величины практически
      невозможны, поэтому ложный отсев исключён.

    Args:
        score (Any): «Сырое» значение скора из фазы HPO.

    Returns:
        bool: True, если скор может попасть в ``result["score"]``.
    """
    if not isinstance(score, (int, float, np.floating)):
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > WORST_SCORE_THRESHOLD


def select_winner(results: dict[str, tuple[float, dict[str, Any]]]) -> str:
    """Детерминированно выбрать алгоритм-победителя по «сырому» скору.

    Побеждает алгоритм с максимальным значением скора (семантика
    оптимизатора — «больше лучше»). Ничьи разрешаются порядком итерации по
    ``results``: ``max`` стабилен, а словарь сохраняет порядок вставки —
    для последовательного пути это порядок конфигурации, поэтому при равных
    скорах побеждает первый настроенный алгоритм (задокументированный
    tie-break, issue #32).

    Args:
        results (dict[str, tuple[float, dict[str, Any]]]): Словарь
            «алгоритм → (скор, параметры)» текущей фазы. Не должен быть пустым.

    Returns:
        str: Имя алгоритма-победителя.

    Raises:
        ValueError: Если ``results`` пуст — победитель не существует.
    """
    if not results:
        raise ValueError("Cannot select a winner from empty results")
    return max(results.items(), key=lambda kv: kv[1][0])[0]


# --------------------------------------------------------------------------- #
#  Finalists pool Top-K (epic #61, task T3)                                    #
#  Формирование пула финалистов: коридор δ от лидера в пользовательской       #
#  семантике + жёсткий кап top_k_candidates. Тип ``CorridorMode`` определён   #
#  в ``config_parser`` (используется также конфигом ``SanityGateCfg``).       #
# --------------------------------------------------------------------------- #
FAMILY_DIVERSITY_MULTIPLIER_DEFAULT = 1.5

# Малый относительный допуск для сравнений на границе коридора: артефакты
# плавающей арифметики (например, 0.55 - 0.5 = 0.050000000000000044) не должны
# исключать кандидата, математически находящегося ровно на границе (правило
# «равенство границе — включение», задокументированный tie-break). Допуск
# масштабируется величиной операндов (относительная точность float64), поэтому
# для скоров порядка 1e-300 он не «расширяет» коридор до 1e-12.
_CORRIDOR_EPS = 1e-12


def _tol_scale(value: float, limit: float) -> float:
    """Шкала допуска: максимальная величина операндов (0 при обоих нулях)."""
    scale = max(abs(value), abs(limit))
    return _CORRIDOR_EPS * scale if scale > 0.0 else 0.0


def _le_tol(value: float, limit: float) -> bool:
    """``value <= limit`` с малым относительным допуском (граница включается)."""
    return value <= limit + _tol_scale(value, limit)


def _ge_tol(value: float, limit: float) -> bool:
    """``value >= limit`` с малым относительным допуском (граница включается)."""
    return value >= limit - _tol_scale(value, limit)


def _family_of(algo_name: str, algorithm_families: Mapping[str, str] | None) -> str:
    """Resolve the algorithm family (unmapped algorithms are their own family).

    Args:
        algo_name (str): Algorithm name.
        algorithm_families (Mapping[str, str] | None): Algorithm → family map,
            or None when no family classification is provided.

    Returns:
        str: The family name (the algorithm name itself when unmapped).
    """
    if algorithm_families is None:
        return algo_name
    return algorithm_families.get(algo_name, algo_name)


def select_finalists(
    results: dict[str, tuple[float, dict[str, Any]]],
    *,
    metric_user: str,
    top_k_candidates: int = 3,
    corridor_delta: float = 0.15,
    corridor_mode: CorridorMode = "auto",
    enforce_family_diversity: bool = False,
    algorithm_families: Mapping[str, str] | None = None,
    allowed_families: set[str] | None = None,
    family_diversity_multiplier: float = FAMILY_DIVERSITY_MULTIPLIER_DEFAULT,
    disqualified: Iterable[str] = (),
) -> list[str]:
    """Build the deterministic finalists pool (epic #61, task T3, issue #65).

    Replaces the "blind" CV-first-place selection with a pool of candidates for
    the second phase: candidates whose CV score falls inside the δ corridor from
    the leader, intersected with the hard ``top_k_candidates`` cap. The corridor
    is expressed in **user** metric semantics (see ``metrics.user_direction``
    and ``metrics.to_user_value``): raw optimizer scores from ``results`` are
    converted back to natural values first (issue #54 — the direction comes from
    ``user_direction``, not from the optimizer's ``greater_is_better``).

    Corridor forms (``corridor_mode``):
    - ``multiplicative`` — min-better metrics (errors):
      ``Score_CV(m) <= Score_CV_best * (1 + δ)``; max-better metrics (R² etc.):
      ``Score_CV(m) >= Score_CV_best * (1 - δ)``.
    - ``additive`` — for metrics where the multiplicative form is incorrect
      (zero/negative values: R² < 0, MAE ≈ 0):
      ``|Score_CV(m) - Score_CV_best| <= δ * (max_score - min_score)``, where
      the range is computed over all valid candidates.
    - ``auto`` (default) — chosen by the metric type: error → multiplicative,
      score metric with a possible sign → additive. For error metrics with a
      practically zero best score (MAE ≈ 0) the multiplicative form would
      collapse the pool to the leader, so the additive form is used instead.
    When the multiplicative corridor would exclude the leader itself (e.g. a
    max-better metric with a negative best score such as R² < 0), the additive
    form is used instead (documented edge case).

    Combination of filters (explicit): pool = (candidates inside the corridor)
    ∩ (first ``top_k_candidates`` by CV score). The corridor is the main
    filter; if it yields fewer candidates than ``top_k_candidates`` the pool
    contains exactly those — no topping up beyond the corridor, except the
    optional family-diversity rule below.

    Optional soft family top-up (``enforce_family_diversity``, default False):
    when the pool represents exactly one algorithm family (linear / ensembles /
    kernel-GPR) **and that family is permitted by ``allowed_families``**, the
    best representative of each missing family is added from the *extended*
    corridor (``family_diversity_multiplier * δ``), never exceeding
    ``top_k_candidates``. The added families are also restricted to
    ``allowed_families``; when the pool's single family is not permitted, no
    top-up happens at all.

    Algorithms disqualified by the circuit breaker (issue #12) never enter the
    pool.

    Determinism: the pool is ordered by the raw score descending (optimizer
    semantics — the same ordering as ``select_winner``); ties keep the
    insertion order of ``results`` (the configuration order), which is the
    documented tie-break.

    Args:
        results (dict[str, tuple[float, dict[str, Any]]]): HPO phase results
            "algorithm → (raw score, params)". Must not be empty after removing
            the disqualified algorithms.
        metric_user (str): User-facing metric name (``general.comparison_metric``);
            used to convert raw scores to user semantics and to pick the
            corridor direction/form.
        top_k_candidates (int): Hard cap on the pool size (>= 1, default 3).
        corridor_delta (float): Relative corridor width δ (>= 0, default 0.15).
        corridor_mode (CorridorMode): 'multiplicative' | 'additive' | 'auto'
            (default 'auto').
        enforce_family_diversity (bool): Enable the optional soft family
            top-up (default False).
        algorithm_families (Mapping[str, str] | None): Algorithm → family map
            (e.g. 'linear', 'ensemble', 'kernel'). Algorithms absent from the
            map are treated as their own family.
        allowed_families (set[str] | None): Families participating in the
            top-up rule: the pool's single family must be permitted AND only
            these families may be added; None — all families are permitted.
        family_diversity_multiplier (float): Width multiplier of the extended
            corridor for the top-up (default 1.5, must be > 0).
        disqualified (Iterable[str]): Algorithms disqualified by the circuit
            breaker (issue #12); they are excluded from the pool.

    Returns:
        list[str]: Ordered finalist names (best first), length in
            [1, top_k_candidates].

    Raises:
        RuntimeError: If ``results`` is empty (after removing disqualified
            algorithms) — no finalists exist.
        ValueError: If parameters are out of bounds (``top_k_candidates`` < 1,
            ``corridor_delta`` < 0, ``family_diversity_multiplier`` <= 0,
            unknown ``corridor_mode``) or the metric name cannot be resolved.
    """
    if not results:
        raise RuntimeError("Cannot build a finalists pool from empty results")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1")
    if corridor_delta < 0:
        raise ValueError("corridor_delta must be >= 0")
    if corridor_mode not in ("multiplicative", "additive", "auto"):
        raise ValueError(
            f"corridor_mode must be 'multiplicative', 'additive' or 'auto'; "
            f"got {corridor_mode!r}"
        )
    if family_diversity_multiplier <= 0:
        raise ValueError("family_diversity_multiplier must be > 0")

    # Circuit breaker (issue #12): disqualified algorithms never enter the pool.
    # Iterable материализуется ДО dict-comprehension: генератор, переданный
    # как ``disqualified``, был бы потреблён ``set(...)`` на первой итерации
    # comprehension, и дисквалифицированные «протекли» бы в пул (ревью PR #33).
    disqualified_set = set(disqualified)
    candidates = {
        name: value for name, value in results.items() if name not in disqualified_set
    }
    if not candidates:
        raise RuntimeError(
            "Cannot build a finalists pool: no valid results after removing "
            "disqualified algorithms"
        )

    # Пользовательская семантика (issue #54): «сырые» скоры переводятся в
    # естественные значения, направление берётся из user_direction (для ошибок
    # это "minimize", хотя оптимизатор максимизирует их инверсии).
    user_scores = {
        name: to_user_value(metric_user, score)
        for name, (score, _) in candidates.items()
    }
    direction = user_direction(metric_user)

    # Лидер по пользовательской семантике; ничья — первый в порядке вставки
    # (тот же tie-break, что и в select_winner).
    if direction == "minimize":
        best_name = min(user_scores, key=lambda n: user_scores[n])
    else:
        best_name = max(user_scores, key=lambda n: user_scores[n])
    best_score = user_scores[best_name]

    # Размах скоров по валидным кандидатам — вычисляется один раз (аддитивная
    # форма коридора), а не внутри _in_corridor на каждого кандидата (O(n²),
    # ревью PR #33).
    score_spread = max(user_scores.values()) - min(user_scores.values())

    if corridor_mode == "auto":
        if is_error_metric(metric_user):
            # Ошибка → multiplicative, кроме вырожденного случая best≈0
            # (MAE≈0): мультипликативная форма даёт порог ≈0 и схлопывает пул
            # до лидера — переходим на аддитивную форму (ревью PR #33).
            effective_mode: CorridorMode = (
                "additive" if abs(best_score) <= _CORRIDOR_EPS else "multiplicative"
            )
        else:
            effective_mode = "additive"
    else:
        effective_mode = corridor_mode

    def _in_corridor(name: str, mode: CorridorMode, delta: float) -> bool:
        """Проверить попадание кандидата в коридор заданной формы и ширины.

        Равенство границе включается (документированный tie-break); малый
        относительный допуск ``_CORRIDOR_EPS`` нейтрализует артефакты
        плавающей арифметики.
        """
        score = user_scores[name]
        if mode == "additive":
            limit = delta * score_spread
            return _le_tol(score, best_score + limit) and _ge_tol(
                score, best_score - limit
            )
        if direction == "minimize":
            return _le_tol(score, best_score * (1.0 + delta))
        return _ge_tol(score, best_score * (1.0 - delta))

    base_members = [
        name
        for name in candidates
        if _in_corridor(name, effective_mode, corridor_delta)
    ]
    if best_name not in base_members:
        # Коридор пуст при max-better метрике с отрицательным лучшим скором
        # (например, R² < 0): мультипликативная форма исключает самого лидера
        # (best*(1-δ) > best при best < 0). Выбираем аддитивную форму
        # (документированный edge case из постановки).
        _LOG.warning(
            "Finalists corridor (mode=%s, direction=%s) excludes the leader "
            "%s (best %s); falling back to the additive corridor form",
            effective_mode,
            direction,
            best_name,
            best_score,
        )
        effective_mode = "additive"
        base_members = [
            name
            for name in candidates
            if _in_corridor(name, effective_mode, corridor_delta)
        ]

    # Детерминированный порядок: «сырой» скор по убыванию (семантика
    # оптимизатора, как в select_winner); ничьи — порядок вставки (порядок
    # конфигурации). Сортировка стабильна.
    ordered = sorted(candidates, key=lambda n: candidates[n][0], reverse=True)

    # Явная комбинация (пересечение): (кандидаты внутри коридора) ∩
    # (первые top_k по CV-скору). Добор за пределы коридора не производится.
    base_set = set(base_members)
    pool = [name for name in ordered if name in base_set][:top_k_candidates]

    # Опциональный мягкий добор семейств (п. 4 постановки): если в пуле
    # представлено ровно одно семейство И это семейство разрешено конфигом
    # (allowed_families), добавляем лучшего представителя недостающих семейств
    # из расширенного коридора (family_diversity_multiplier × δ), не превышая
    # top_k_candidates. Добавляемые семейства также ограничиваются
    # allowed_families. Поведение приведено к документации (ревью PR #33).
    if enforce_family_diversity and len(pool) < top_k_candidates:
        pool_families = {_family_of(name, algorithm_families) for name in pool}
        single_pool_family = (
            next(iter(pool_families)) if len(pool_families) == 1 else None
        )
        if single_pool_family is not None and (
            allowed_families is None or single_pool_family in allowed_families
        ):
            extended_delta = corridor_delta * family_diversity_multiplier
            extended_members = [
                name
                for name in candidates
                if _in_corridor(name, effective_mode, extended_delta)
            ]
            extended_set = set(extended_members)
            missing_families = {
                _family_of(name, algorithm_families)
                for name in candidates
                if _family_of(name, algorithm_families) != single_pool_family
            }
            if allowed_families is not None:
                missing_families &= allowed_families

            pool_names = set(pool)
            remaining = set(missing_families)
            while len(pool) < top_k_candidates and remaining:
                picked: str | None = None
                for name in ordered:
                    family = _family_of(name, algorithm_families)
                    if (
                        name not in pool_names
                        and name in extended_set
                        and family in remaining
                    ):
                        picked = name
                        break
                if picked is None:
                    break
                pool.append(picked)
                pool_names.add(picked)
                remaining.remove(_family_of(picked, algorithm_families))

    # Финальный стабильный порядок: добавленные семейства слабее участников
    # базового пула, но пересортировка делает инвариант явным.
    pool.sort(key=lambda n: candidates[n][0], reverse=True)
    return pool


# --------------------------------------------------------------------------- #
#  Robust winner selection (epic #61, task T4, issue #64)                      #
#  Двухстадийный выбор победителя: пул финалистов (T3) → обучение каждого     #
#  финалиста на 100% данных ОДИН раз → аудит ModelSanityGate (T2) →           #
#  ранжирование прошедших аудит по RMSE_oof.                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FinalistCandidate:
    """Финалист двухстадийного выбора: модель, обученная на 100% данных.

    Собирается до ``select_robust_winner`` (обучение финалистов на полных
    данных выполняется ровно один раз, результат переиспользуется для
    ``RMSE_full``, аудита и отчёта — без дублирования обучения, требование 4
    постановки T4).

    Attributes:
        algo_name (str): Имя алгоритма.
        model (Any): Обученный на 100% данных пайплайн (``trainer.pipeline``);
            используется аудитом (контур В) и для предсказаний ``y_pred_full``.
        params (dict[str, Any]): Лучшие гиперпараметры из HPO
            (``phase_results[algo_name][1]``).
        cv_score (float): «Сырой» скор из HPO-фазы (семантика оптимизатора);
            используется в режимах ``off``/``warn_only`` для выбора победителя
            как при старом ``select_winner``.
        rmse_oof (float | None): ``RMSE_oof`` финалиста (``trainer.oof_score_``);
            ``None``, если OOF-оценка невозможна.
        y_pred_oof (Any): OOF-вектор предсказаний (``trainer.oof_predictions_``),
            выровненный по строкам X.
        y_pred_full (Any): Предсказания full-fit модели на тех же строках
            (``trainer.predict(X)``); используется контуром Г (RMSE_full) и
            отчётом.
        rmse_full (float | None): ``RMSE_full = RMSE(y, y_pred_full)``;
            ``None``, если вычислить нельзя.
        trainer (Any | None): Сам ``ModelTrainer`` финалиста; передаётся в
            ``persist_artifact``, чтобы артефакт сохранялся БЕЗ повторного
            обучения победителя.
    """

    algo_name: str
    model: Any
    params: dict[str, Any]
    cv_score: float
    rmse_oof: float | None
    y_pred_oof: Any
    y_pred_full: Any
    rmse_full: float | None = None
    trainer: Any | None = None


@dataclass(frozen=True)
class RobustWinnerResult:
    """Результат robust-выбора победителя (T4).

    Attributes:
        winner_algo (str): Имя алгоритма-победителя.
        mode (SanityGateMode): Режим работы гейта ('off'/'warn_only'/'active').
        pool (list[str]): Порядок пула финалистов (из T3, детерминированный
            tie-break при равных RMSE_oof).
        audit (dict[str, SanityCheckResult]): Аудит каждого финалиста
            (имя → результат). Заполнен при mode != 'off'.
        disqualified (dict[str, list[str]]): Дисквалифицированные финалисты
            (имя → причины). В режиме ``active`` причины применяются;
            в ``warn_only`` — гипотетические (победитель не меняется).
        fallback_used (bool): True, если все кандидаты забракованы и победитель
            выбран fallback-критерием («наименее проблемная» модель).
        winner_rmse_oof (float | None): ``RMSE_oof`` победителя.
        winner_if_active (str | None): Гипотетический победитель в режиме
            ``active`` (лучший ``RMSE_oof`` среди прошедших аудит; при полном
            провале — fallback-критерий). Заполняется только в ``warn_only``
            (T6): ключевая статистика перед включением гейта. В остальных
            режимах — None.
    """

    winner_algo: str
    mode: SanityGateMode
    pool: list[str]
    audit: dict[str, SanityCheckResult] = field(default_factory=dict)
    disqualified: dict[str, list[str]] = field(default_factory=dict)
    fallback_used: bool = False
    winner_rmse_oof: float | None = None
    winner_if_active: str | None = None


def _fit_finalist(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    cfg: Config,
    metric_name_sklearn: str,
    *,
    validation_strategy: ValidationStrategy | str | None,
    n_folds: int | None,
    test_size: float | None,
) -> Any:
    """Обучить финалиста на 100% данных и вернуть его ``ModelTrainer``.

    Выполняет ровно одно обучение (``ModelTrainer.fit``): валидационный
    скоринг с OOF-каналом (issue #62) + финальный fit на полном наборе.
    Результат переиспользуется далее для ``RMSE_full``, аудита и отчёта —
    дублирование обучения отсутствует (требование 4 постановки T4).

    Args:
        algo_name (str): Имя алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма.
        X (pd.DataFrame): Полная матрица признаков.
        y (pd.Series): Полный вектор целевой переменной.
        best_params (dict[str, Any]): Лучшие гиперпараметры из HPO.
        cfg (Config): Корневой конфиг.
        metric_name_sklearn (str): Имя метрики в sklearn-семантике.
        validation_strategy (ValidationStrategy | str | None): Разрешённая
            стратегия валидации.
        n_folds (int | None): Число фолдов (для 'k_fold').
        test_size (float | int | None): Размер hold-out части.

    Returns:
        Any: Обученный ``ModelTrainer`` (pipeline на 100% данных, OOF-атрибуты).

    Raises:
        Exception: Любая ошибка обучения пробрасывается наверх — вызывающий
            код (``_train_finalists``) ловит её и исключает финалиста
            (требование 5 постановки T4).
    """
    trainer = _build_trainer(
        algo_name,
        algo_cfg,
        best_params,
        cfg,
        metric_name_sklearn=metric_name_sklearn,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        test_size=test_size,
    )
    trainer.fit(X, y)
    return trainer


def _to_candidate(
    algo_name: str,
    trainer: Any,
    params: dict[str, Any],
    cv_score: float,
    X: pd.DataFrame,
    y: pd.Series,
) -> FinalistCandidate:
    """Собрать ``FinalistCandidate`` из обученного ``ModelTrainer``.

    Вычисляет производные величины (``y_pred_full``, ``rmse_full``) ровно
    один раз из уже обученного тренера; OOF-вектор и ``RMSE_oof`` берутся из
    атрибутов тренера (issue #62).

    Args:
        algo_name (str): Имя алгоритма.
        trainer (Any): Обученный ``ModelTrainer``.
        params (dict[str, Any]): Лучшие гиперпараметры.
        cv_score (float): «Сырой» CV-скор из HPO.
        X (pd.DataFrame): Полная матрица признаков.
        y (pd.Series): Полный вектор целевой переменной.

    Returns:
        FinalistCandidate: Кандидат с переиспользуемыми full-fit векторами.
    """
    y_pred_full = np.asarray(trainer.predict(X), dtype=float).reshape(-1)
    rmse_full: float | None = None
    try:
        rmse_full = oof_rmse(y, y_pred_full)
    except ValueError:
        rmse_full = None
    return FinalistCandidate(
        algo_name=algo_name,
        model=trainer.pipeline,
        params=params,
        cv_score=cv_score,
        rmse_oof=trainer.oof_score_,
        y_pred_oof=trainer.oof_predictions_,
        y_pred_full=y_pred_full,
        rmse_full=rmse_full,
        trainer=trainer,
    )


def _train_finalists(
    pool: list[str],
    phase_results: dict[str, tuple[float, dict[str, Any]]],
    cfg: Config,
    prepared: PreparedData,
) -> dict[str, FinalistCandidate]:
    """Обучить каждый финалист пула на 100% данных ровно один раз.

    Провал обучения отдельного финалиста не прерывает запуск: исключение
    ловится, причина логируется (WARNING), кандидат пропускается (переход к
    следующему, требование 5 постановки T4).

    Args:
        pool (list[str]): Упорядоченный пул финалистов (из ``select_finalists``).
        phase_results (dict[str, tuple[float, dict[str, Any]]]): Результаты HPO
            «алгоритм → (скор, параметры)».
        cfg (Config): Корневой конфиг.
        prepared (PreparedData): Подготовленные данные и настройки.

    Returns:
        dict[str, FinalistCandidate]: Обученные кандидаты (ключ — имя
            алгоритма), в порядке пула (dict сохраняет порядок вставки).
    """
    algos = _algorithms_as_dict(cfg.algorithms)
    candidates: dict[str, FinalistCandidate] = {}
    for name in pool:
        if name not in phase_results:
            continue
        score, params = phase_results[name]
        try:
            trainer = _fit_finalist(
                name,
                algos[name],
                prepared.X,
                prepared.y,
                params,
                cfg,
                prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
            )
        except Exception as err:  # noqa: BLE001 — финалист исключается
            _LOG.warning(
                "Finalist %s failed to train on the full dataset: %s. "
                "The candidate is skipped (T4 requirement 5).",
                name,
                err,
                exc_info=True,
            )
            continue
        candidates[name] = _to_candidate(
            name, trainer, params, score, prepared.X, prepared.y
        )
    return candidates


# Значимость контуров для fallback-критерия (T4, п. 3): Г > А > Б > В.
# Чем больше ранг, тем «тяжелее» нарушение. Ранги фиксированы в коде.
_CIRCUIT_RANK_BY_LETTER = {
    "A": 2,  # А — diversity
    "B": 1,  # Б — unique
    "C": 0,  # В — dead features
    "D": 3,  # Г — generalization gap (самое тяжёлое)
}


def _circuit_rank(reason: str) -> int:
    """Ранг значимости контура по строке reason (Г > А > Б > В).

    Разбирает префикс ``Circuit <letter>`` в reason (формат
    ``ModelSanityGate``). Неизвестный формат трактуется как наименее значимый
    контур (ранг 0) — defensive fallback, чтобы fallback-ранжирование никогда
    не падало на кастомных причинах.

    Args:
        reason (str): Текст причины дисквалификации.

    Returns:
        int: Ранг значимости (0..3); 3 — контур Г, 2 — А, 1 — Б, 0 — В/unknown.
    """
    match = re.search(r"Circuit\s+([A-D])", reason)
    if match is None:
        return 0
    return _CIRCUIT_RANK_BY_LETTER.get(match.group(1), 0)


def _severity_signature(reasons: list[str]) -> tuple[int, ...]:
    """Подпись тяжести нарушений для fallback-критерия (T4, п. 3).

    Кортеж рангов значимости контуров, отсортированный по убыванию (самое
    тяжёлое нарушение первым). Лексикографически меньшая подпись = менее
    проблемная модель при равном числе нарушенных контуров.

    Args:
        reasons (list[str]): Причины дисквалификации кандидата.

    Returns:
        tuple[int, ...]: Отсортированная по убыванию подпись тяжести.
    """
    return tuple(sorted((_circuit_rank(r) for r in reasons), reverse=True))


def _best_by_rmse_oof(
    names: list[str], candidates: Mapping[str, FinalistCandidate]
) -> str:
    """Выбрать кандидата с лучшим (минимальным) ``RMSE_oof``.

    Tie-break: при равных ``RMSE_oof`` побеждает первый в порядке списка
    (порядок пула из T3 — детерминированный, требование постановки).
    Кандидат с недоступным ``RMSE_oof`` (None) ранжируется последним.

    Args:
        names (list[str]): Кандидаты в детерминированном порядке.
        candidates (Mapping[str, FinalistCandidate]): Кандидаты.

    Returns:
        str: Имя победителя.
    """

    def _key(name: str) -> tuple[int, float]:
        rmse = candidates[name].rmse_oof
        return (rmse is None, rmse if rmse is not None else float("inf"))

    return min(names, key=_key)


def _select_active_winner(
    pool_order: list[str],
    audits: Mapping[str, SanityCheckResult],
    candidate_models: Mapping[str, FinalistCandidate],
) -> str:
    """Выбрать победителя в режиме ``active`` (T4; переиспользуется в T6).

    Сначала — лучший ``RMSE_oof`` среди прошедших аудит (CV и ``RMSE_full``
    в ранжировании не участвуют). Если валидных кандидатов нет (тотальный
    провал пула) — формальный fallback-критерий «наименее проблемной» модели:
    минимум числа нарушенных контуров → минимум тяжести (значимость контуров
    зафиксирована в коде: Г > А > Б > В, см. ``_circuit_rank``) → лучший
    ``RMSE_oof``. Запуск не падает.

    В режиме ``warn_only`` функция даёт гипотетического победителя
    (``winner_if_active``) — ключевую статистику перед включением гейта (T6).

    Args:
        pool_order (list[str]): Упорядоченный пул финалистов.
        audits (Mapping[str, SanityCheckResult]): Аудит каждого финалиста.
        candidate_models (Mapping[str, FinalistCandidate]): Кандидаты.

    Returns:
        str: Имя победителя (актуального в ``active`` либо гипотетического
            в ``warn_only``).
    """
    valid = [n for n in pool_order if audits[n].is_valid]
    if valid:
        return _best_by_rmse_oof(valid, candidate_models)

    def _fallback_key(name: str) -> tuple[int, tuple[int, ...], bool, float]:
        reasons = audits[name].reasons
        rmse = candidate_models[name].rmse_oof
        return (
            len(reasons),
            _severity_signature(reasons),
            rmse is None,
            rmse if rmse is not None else float("inf"),
        )

    return min(pool_order, key=_fallback_key)


def _fmt_audit_value(value: float) -> str:
    """Отформатировать числовую метрику аудита для логов (T6).

    NaN/±inf (контур отключён либо не вычислим) выводятся как ``"n/a"``,
    чтобы лог итогов аудита победителя оставался компактным и читаемым.

    Args:
        value (float): Значение метрики аудита (diversity/unique/gap).

    Returns:
        str: ``"%.4f"`` для конечных значений, ``"n/a"`` иначе.
    """
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def select_robust_winner(
    candidate_models: Mapping[str, FinalistCandidate],
    X: pd.DataFrame,
    y: pd.Series,
    sanity_gate: ModelSanityGate,
    *,
    mode: SanityGateMode = "off",
    audit_time_budget_seconds: float = 0,
) -> RobustWinnerResult:
    """Двухстадийный выбор победителя с аудитом Sanity Gate (эпик #61, T4).

    Перебирает пул кандидатов (уже обученных на 100% данных), прогоняет
    каждого через ``ModelSanityGate.check`` (diversity/unique считаются на
    OOF-векторе; контур Г использует full-fit предсказания) и ранжирует
    прошедших аудит по ``RMSE_oof`` (CV и ``RMSE_full`` в ранжировании НЕ
    участвуют — композитный скор удалён по итогам ревью v2).

    Режимы (``mode``, из T5):
    - ``off``: старое поведение ``select_winner`` — победитель = лучший
      ``cv_score`` (без аудита; ``audit`` остаётся пустым);
    - ``warn_only``: аудит выполняется, дисквалификации гипотетические
      (фиксируются в ``disqualified``/``audit`` для отчёта T6), но победитель
      выбирается как при ``off`` (сбор статистики на реальных данных);
    - ``active``: дисквалификации применяются; победитель = лучший
      ``RMSE_oof`` среди прошедших все проверки.

    Бюджет времени аудита (``audit_time_budget_seconds``, T5): суммарное
    время фазы аудита ограничено; при исчерпании бюджета оставшиеся
    кандидаты не аудируются и получают явный reason «audit skipped: budget
    exceeded» (в ``active`` они не могут выиграть, в ``warn_only`` попадают
    в отчёт как гипотетические). ``0`` — без ограничения.

    Fallback (формализован, п. 3 постановки): если все кандидаты забракованы,
    выбирается «наименее проблемная» модель — минимум числа нарушенных
    контуров, при равенстве — минимум тяжести нарушений (значимость контуров
    зафиксирована в коде: Г > А > Б > В, см. ``_circuit_rank``), при равенстве
    — лучший ``RMSE_oof``. Запуск не падает, в лог пишется WARNING с причинами.

    Args:
        candidate_models (Mapping[str, FinalistCandidate]): Пул кандидатов
            (обучены на 100% данных, OOF-векторы доступны). Порядок вставки
            — детерминированный порядок пула из T3.
        X (pd.DataFrame): Полная матрица признаков.
        y (pd.Series): Полный вектор целевой переменной.
        sanity_gate (ModelSanityGate): Настроенный экземпляр гейта (T2).
        mode (SanityGateMode): Режим работы ('off'/'warn_only'/'active').
        audit_time_budget_seconds (float): Суммарный бюджет времени фазы
            аудита в секундах; 0 — без ограничения.

    Returns:
        RobustWinnerResult: Победитель, аудит по каждому кандидату,
            применённые/гипотетические дисквалификации и флаг fallback.

    Raises:
        RuntimeError: Если пул кандидатов пуст — победитель не существует.
        ValueError: Если ``mode`` неизвестен.
    """
    if not candidate_models:
        raise RuntimeError("Cannot select a robust winner from an empty candidate pool")

    pool_order = list(candidate_models)

    def _cv_winner() -> str:
        """Старое поведение select_winner: лучший «сырой» скор, tie — порядок."""
        return max(pool_order, key=lambda n: candidate_models[n].cv_score)

    if mode == "off":
        winner = _cv_winner()
        return RobustWinnerResult(
            winner_algo=winner,
            mode=mode,
            pool=pool_order,
            winner_rmse_oof=candidate_models[winner].rmse_oof,
        )

    if mode not in ("warn_only", "active"):
        raise ValueError(f"mode must be 'off', 'warn_only' or 'active'; got {mode!r}")

    # Аудит каждого финалиста: diversity/unique — на OOF-векторе, контур Г —
    # на full-fit предсказаниях (требование 1 постановки T4). Суммарное время
    # фазы ограничено бюджетом audit_time_budget_seconds (0 — без лимита).
    audits: dict[str, SanityCheckResult] = {}
    audit_started = time.monotonic()

    def _audit_budget_exhausted() -> bool:
        return (
            audit_time_budget_seconds > 0
            and time.monotonic() - audit_started >= audit_time_budget_seconds
        )

    for name in pool_order:
        if _audit_budget_exhausted():
            _LOG.warning(
                "[sanity_gate] audit time budget (%.3fs) exceeded; "
                "skipping audit of candidate %s",
                audit_time_budget_seconds,
                name,
            )
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[
                    (
                        "Sanity audit skipped: audit time budget "
                        f"({audit_time_budget_seconds:g}s) exceeded"
                    )
                ],
                severity=0,
            )
            continue
        cand = candidate_models[name]
        try:
            audits[name] = sanity_gate.check(
                y=y,
                y_pred_oof=cand.y_pred_oof,
                y_pred_full=cand.y_pred_full,
                X=X,
                model=cand.model,
            )
        except Exception as err:  # noqa: BLE001 — сбой аудита = дисквалификация
            _LOG.warning("Sanity audit failed for candidate %s: %s", name, err)
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[f"Sanity audit failed: {err}"],
                severity=4,
            )

    disqualified = {
        name: list(audits[name].reasons) for name in pool_order if audits[name].reasons
    }

    # Требование T6: для каждого забракованного кандидата — WARNING с перечнем
    # причин. В warn_only — «был бы дисквалифицирован» (гипотетически, выбор
    # не меняется); в active — дисквалификация применена.
    for name, reasons in disqualified.items():
        if mode == "warn_only":
            _LOG.warning(
                "[sanity_gate warn_only] Candidate %s WOULD BE disqualified "
                "(hypothetical, not applied): %s",
                name,
                reasons,
            )
        else:
            _LOG.warning(
                "[sanity_gate active] Candidate %s disqualified by the sanity gate: %s",
                name,
                reasons,
            )

    if mode == "warn_only":
        # Победитель — как при off (CV-лидер); дисквалификации гипотетические
        # и попадают в отчёт (T6), фактический выбор не меняют.
        winner = _cv_winner()
        return RobustWinnerResult(
            winner_algo=winner,
            mode=mode,
            pool=pool_order,
            audit=audits,
            disqualified=disqualified,
            winner_rmse_oof=candidate_models[winner].rmse_oof,
            winner_if_active=_select_active_winner(
                pool_order, audits, candidate_models
            ),
        )

    # mode == "active": применяем дисквалификации, ранжируем по RMSE_oof.
    winner = _select_active_winner(pool_order, audits, candidate_models)
    all_failed = not any(audits[n].is_valid for n in pool_order)
    if all_failed:
        # Fallback (п. 3 постановки): все кандидаты забракованы — «наименее
        # проблемная» модель. Формальный критерий указывается в логе явно (T6).
        _LOG.warning(
            "[sanity_gate active] All %d candidate(s) failed the sanity audit; "
            "falling back to the least problematic model %s. Formal criterion "
            "(T4): min violated circuits -> min severity (D > A > B > C) -> "
            "best RMSE_oof. Reasons: %s",
            len(pool_order),
            winner,
            {n: audits[n].reasons for n in pool_order},
        )
    return RobustWinnerResult(
        winner_algo=winner,
        mode=mode,
        pool=pool_order,
        audit=audits,
        disqualified=disqualified,
        fallback_used=all_failed,
        winner_rmse_oof=candidate_models[winner].rmse_oof,
    )


def _algorithms_as_dict(algorithms_cfg: Any) -> dict[str, AlgoCfg]:
    """Преобразует AlgorithmsConfig в обычный словарь {name: AlgoCfg}."""
    # model_fields через экземпляр тоже работает, но deprecated
    # (PydanticDeprecatedSince211, удаление в V3.0) — используем класс
    return {
        name: algo_cfg
        for name in type(algorithms_cfg).model_fields
        if (algo_cfg := getattr(algorithms_cfg, name)) is not None
    }


def _feature_selection_mode(
    cfg: FeatureSelectionCfg | dict[str, Any] | None,
) -> str | None:
    """Извлечь строковое значение ``mode`` из конфигурации отбора признаков.

    Args:
        cfg: Объект ``FeatureSelectionCfg``, словарь конфигурации или ``None``.

    Returns:
        str | None: Режим ('disabled'/'always'/'auto') либо ``None``, если
            конфигурация не задана или не содержит режима.
    """
    if isinstance(cfg, FeatureSelectionCfg):
        return cfg.mode.value
    if isinstance(cfg, dict):
        mode = cfg.get("mode")
        if isinstance(mode, FeatureSelectionMode):
            return mode.value
        return mode if isinstance(mode, str) else None
    return None


# --------------------------------------------------------------------------- #
#  Dyn-import helper                                                          #
# --------------------------------------------------------------------------- #
def _load_module(path: str) -> ModuleType:
    """Импортировать модуль по указанному пути.
    Динамически загружает Python-модуль, используя dotted-path нотацию.
    В случае неудачи генерирует информативное исключение.
    Args:
        path (str): Путь к модулю в формате 'package.module'.
    Returns:
        ModuleType: Объект импортированного модуля.
    Raises:
        ImportError: Если модуль не найден по указанному пути.
    """
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(f"Can't import module '{path}'") from exc


# --------------------------------------------------------------------------- #
#  HPO wrapper                                                                #
# --------------------------------------------------------------------------- #
def _run_hpo(
    *,
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    metric_name_sklearn: str,
    n_trials: int,
    validation_strategy: ValidationStrategy,
    n_folds: int | None = None,
    train_test_split_test_size: float = 0.2,
    search_space_override: dict[str, Any] | None = None,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    initial_params: dict[str, Any] | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    encoding: str = "one_hot",
    pruning: dict[str, Any] | None = None,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: str | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]] | None:
    """Запустить поиск оптимальных гиперпараметров для алгоритма.
    Логика работы:
    1. Загрузка тюнера: Импортируется модуль, указанный в `algo_cfg.tuner`.
    2. Проверка сигнатуры: Метод `optimize` тюнера проверяется на поддержку
       специфичных параметров (n_folds, oversampling, space_overrides).
    3. Выполнение: Запускается оптимизация. Если алгоритм признан невалидным
       через `InvalidAlgorithmError`, выбрасывается каноничное исключение.
    Args:
        algo_name (str): Название алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путями к тюнеру.
        X (pd.DataFrame): Матрица признаков.
        y (pd.Series): Вектор целевой переменной.
        metric_name_sklearn (str): Название метрики в формате sklearn.
        n_trials (int): Количество итераций поиска.
        validation_strategy (ValidationStrategy): Стратегия валидации
            (k_fold, loo и т.д.).
        n_folds (int | None): Количество фолдов для кросс-валидации.
        train_test_split_test_size (float): Размер hold-out части для стратегии
            'train_test_split' (доля в (0, 1) либо целое число строк после
            резолюции 'auto'). Прокидывается тюнеру, поддерживающему параметр,
            чтобы оценка HPO не расходилась с финальным val_score (D3 issue #24).
        search_space_override (Dict[str, Any] | None):
            Переопределенное пространство поиска.
        data_oversampling (bool): Флаг включения оверсэмплинга.
        data_oversampling_multiplier (float): Коэффициент увеличения выборки.
        data_oversampling_algorithm (str): Название алгоритма оверсэмплинга.
        initial_params (dict[str, Any] | None): Гиперпараметры из предыдущей фазы HPO
            для enqueue_trial (монотонность улучшения между фазами).
        categorical_features (list[str] | None): Имена категориальных колонок,
            определённые один раз в ``train_best_model`` и передаваемые тюнеру.
        numerical_features (list[str] | None): Имена числовых колонок.
        encoding (str): Стратегия кодирования категорий ('one_hot' или 'ordinal'),
            прокидываемая тюнеру для согласованной предобработки с финальным fit.
        pruning (dict[str, Any] | None): Настройки ранней остановки (pruning)
            из секции ``general.pruning``. Передаются только тюнерам, которые
            поддерживают аргумент ``pruning``; кастомные тюнеры не затрагиваются.
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Конфигурация
            отбора признаков из корневого ``Config.general.feature_selection``.
            Передаётся только тюнерам, которые поддерживают аргумент
            ``feature_selection_cfg`` или принимают ``**kwargs`` (сигнатура
            проверяется через ``inspect.signature``); кастомные тюнеры не
            затрагиваются.
    Returns:
        Optional[Tuple[float, Dict[str, Any]]]:
            Кортеж (лучшая метрика, лучшие параметры)
            или None, если произошла ошибка при выполнении HPO.
    Raises:
        _CanonicalIAE: Если тюнер сообщает о несовместимости алгоритма с данными.
        AttributeError: Если в модуле тюнера отсутствует функция `optimize`.
    """
    if algo_cfg.tuner is None:
        raise ValueError("Tuner path is not configured")
    tuner = _load_module(algo_cfg.tuner)
    if not hasattr(tuner, "optimize"):
        raise AttributeError(f"Module {algo_cfg.tuner} lacks `optimize`")

    sig = inspect.signature(tuner.optimize)
    # Сигнатура с **kwargs считается совместимой (аналогично _fit_and_save):
    # тюнер сам решает, что делать с лишними ключами, поэтому служебные
    # аргументы отбора признаков можно безопасно прокидывать.
    tuner_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    kwargs: dict[str, Any] = {
        "algo_name": algo_name,
        "X": X,
        "y": y,
        "metric": metric_name_sklearn,
        "n_trials": n_trials,
        "validation_strategy": validation_strategy,
    }

    # прокидываем n_folds, если модуль это умеет
    if (
        validation_strategy == ValidationStrategy.k_fold
        and n_folds is not None
        and "n_folds" in sig.parameters
    ):
        kwargs["n_folds"] = n_folds

    # прокидываем разрешённый размер hold-out (issue #24, D3): при 'auto' →
    # 'train_test_split' choose_validation_method возвращает целое число строк,
    # которое обязано совпадать и в HPO, и в финальном fit.
    if "train_test_split_test_size" in sig.parameters:
        kwargs["train_test_split_test_size"] = train_test_split_test_size

    # прокидываем oversampling параметры, если tuner их поддерживает
    if "data_oversampling" in sig.parameters:
        kwargs["data_oversampling"] = data_oversampling

    if "data_oversampling_multiplier" in sig.parameters:
        kwargs["data_oversampling_multiplier"] = data_oversampling_multiplier

    if "data_oversampling_algorithm" in sig.parameters:
        kwargs["data_oversampling_algorithm"] = data_oversampling_algorithm

    # прокидываем кастомный search space, если предусмотрен
    if search_space_override is not None:
        # Мы передаем словарь {algo_name: hyperparameters_dict}
        overrides = {algo_name: search_space_override}
        if "space_overrides" in sig.parameters:
            kwargs["space_overrides"] = overrides

    # прокидываем initial_params, если есть (для refine_winner phase)
    if initial_params is not None and "initial_params" in sig.parameters:
        kwargs["initial_params"] = initial_params

    # прокидываем детектированные категориальные/числовые колонки,
    # чтобы фаза HPO строила препроцессор согласованно с финальным обучением
    if "categorical_features" in sig.parameters:
        kwargs["categorical_features"] = categorical_features

    if "numerical_features" in sig.parameters:
        kwargs["numerical_features"] = numerical_features

    # прокидываем стратегию кодирования, если тюнер её поддерживает
    if "encoding" in sig.parameters:
        kwargs["encoding"] = encoding
    else:
        # Кастомный тюнер не принимает encoding: стратегия молча игнорируется
        # в фазе HPO, но применяется в финальном обучении -> рассинхрон
        # представления признаков. Предупреждаем, чтобы пользователь мог
        # обновить тюнер до поддерживающего аргумент `encoding`.
        _LOG.warning(
            "Tuner %s does not accept `encoding`; categorical_encoding=%r "
            "will NOT be applied during HPO (final fit may diverge).",
            algo_cfg.tuner,
            encoding,
        )

    # прокидываем переопределение пресета предобработки (FR-5): явное указание
    # пользователя из конфигурации применяется и в HPO, и в финальном обучении
    preprocessing_override = getattr(algo_cfg, "preprocessing", None)
    if "preprocessing_override" in sig.parameters:
        kwargs["preprocessing_override"] = preprocessing_override

    # прокидываем настройки ранней остановки (pruning), если тюнер их
    # поддерживает; кастомные тюнеры без аргумента `pruning` не затрагиваются
    if pruning is not None and "pruning" in sig.parameters:
        kwargs["pruning"] = pruning

    # прокидываем параметры high-cardinality кодирования и новых стратегий
    # (issue #20), если тюнер их поддерживает; согласованно с финальным обучением
    if "high_cardinality_threshold" in sig.parameters:
        kwargs["high_cardinality_threshold"] = high_cardinality_threshold

    if "high_cardinality_encoding" in sig.parameters:
        kwargs["high_cardinality_encoding"] = high_cardinality_encoding

    if "hashing_n_components" in sig.parameters:
        kwargs["hashing_n_components"] = hashing_n_components

    if "target_encoding_smoothing" in sig.parameters:
        kwargs["target_encoding_smoothing"] = target_encoding_smoothing

    if "target_encoding_fallback" in sig.parameters:
        kwargs["target_encoding_fallback"] = target_encoding_fallback

    # прокидываем конфигурацию отбора признаков (issue #11), если тюнер
    # поддерживает аргумент `feature_selection_cfg` или принимает **kwargs;
    # кастомные тюнеры без поддержки не затрагиваются. При активном режиме
    # отбора предупреждаем о рассинхроне HPO ↔ финальный fit (аналогично
    # encoding): конфигурация не будет применена в фазе поиска.
    if "feature_selection_cfg" in sig.parameters or tuner_accepts_var_kwargs:
        kwargs["feature_selection_cfg"] = feature_selection_cfg
    elif _feature_selection_mode(feature_selection_cfg) not in (None, "disabled"):
        _LOG.warning(
            "Tuner %s does not accept `feature_selection_cfg`; feature "
            "selection mode=%r will NOT be applied during HPO "
            "(final fit may diverge).",
            algo_cfg.tuner,
            _feature_selection_mode(feature_selection_cfg),
        )

    try:
        _, best_params, best_score = tuner.optimize(**kwargs)
        if best_score is None or best_params is None:
            # Tuner couldn't get a valid result: either all trials failed
            # (optuna.TrialPruned) or it failed otherwise. Return None so the
            # main loop knows the algorithm didn't pass (issue #13): the dummy
            # candidate with score -3.4e38 and params=None should no longer
            # end up in phase_results.
            _LOG.error(
                "Algorithm %s failed during HPO: no valid result "
                "(best_score=%r, best_params=%r)",
                algo_name,
                best_score,
                best_params,
            )
            return None
        return best_score, best_params
    except Exception as err:
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # Return None so the main loop knows the algorithm didn't pass.
        return None


# --------------------------------------------------------------------------- #
#  Final fit & save                                                           #
# --------------------------------------------------------------------------- #
def _build_trainer(
    algo_name: str,
    algo_cfg: AlgoCfg,
    best_params: dict[str, Any],
    cfg: Config,
    metric_name_sklearn: str = "r2",
    *,
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int | None = None,
    test_size: float | None = None,
) -> Any:
    """Сконструировать ``ModelTrainer`` для алгоритма по конфигурации.

    Единая точка сборки тренера, используемая и финальным обучением
    победителя (``_fit_and_save``), и обучением финалистов на 100% данных
    (T4, ``_fit_finalist``): гарантирует, что оба пути передают тренеру
    одинаковые настройки предобработки, оверсэмплинга, кодирования,
    отбора признаков и валидации (issue #24, D3).

    Args:
        algo_name (str): Название выбранного алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путем к тренеру.
        best_params (Dict[str, Any]): Найденные оптимальные гиперпараметры.
        cfg (Config): Общий объект конфигурации.
        metric_name_sklearn (str): Имя основной метрики в формате sklearn.
        validation_strategy (ValidationStrategy | str | None): Разрешённая
            стратегия валидации (issue #24, D3). ``None`` — дефолт
            ``ModelTrainer`` ('train_test_split').
        n_folds (int | None): Число фолдов (для 'k_fold').
        test_size (float | int | None): Размер hold-out части.

    Returns:
        Any: Сконструированный (но ещё не обученный) ``ModelTrainer``.

    Raises:
        ValueError: Если ``algo_cfg.trainer_module`` не задан либо
            ``best_params`` равен ``None`` (сигнал полного отказа HPO,
            issue #13).
        AttributeError: Если в модуле тренера отсутствует класс
            ``ModelTrainer``.
    """
    if algo_cfg.trainer_module is None:
        raise ValueError("Trainer module path is not configured")
    if best_params is None:
        raise ValueError(
            f"Algorithm '{algo_name}' produced no hyperparameters "
            "(best_params is None): HPO phase failed completely. The algorithm "
            "should have been excluded before the final fit."
        )
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    # Решение победителя HPO по отбору признаков (issue #11): в режиме
    # 'auto' ключ "use_feature_selection" зафиксирован в best_params Optuna;
    # в режимах 'always'/'disabled' ключа нет -> None, и ModelTrainer сам
    # разрешает активность по конфигурации. Служебный ключ извлекается из
    # копии: исходный best_params (он же возвращается в result["params"]) не
    # мутируется, а "use_feature_selection" гарантированно не попадает в
    # гиперпараметры ModelTrainer (раньше выпадал только неявно через
    # clean_hyperparameters).
    trainer_params = dict(best_params)
    fs_active = trainer_params.pop("use_feature_selection", None)

    # Проверяем сигнатуру ModelTrainer (аналогично тюнерам в _run_hpo):
    # кастомные тренеры без поддержки отбора признаков не должны получать
    # новые аргументы (иначе TypeError). Сигнатура с **kwargs считается
    # совместимой — тренер сам решает, что делать с лишними ключами.
    trainer_sig = inspect.signature(trainer_module.ModelTrainer)
    trainer_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()
    )
    trainer_has_fs_cfg = "feature_selection_cfg" in trainer_sig.parameters
    trainer_has_fs_active = "feature_selection_active" in trainer_sig.parameters

    trainer_kwargs: dict[str, Any] = {
        "algorithm": algo_name,
        "hyperparams": trainer_params,
        "metric": metric_name_sklearn,
        # Пробрасываем настройки оверсэмплинга из конфига в тренер
        "data_oversampling": cfg.oversampling.enable,
        "data_oversampling_multiplier": cfg.oversampling.multiplier,
        "data_oversampling_algorithm": cfg.oversampling.algorithm,
        "serialization_format": cfg.general.serialization_format,
        "encoding_strategy": cfg.general.categorical_encoding,
        "additional_metrics": cfg.general.additional_metrics,
        # Явное переопределение пресета предобработки из конфига (FR-5):
        # применяется согласованно с фазой HPO (AC-7).
        "preprocessing_override": getattr(algo_cfg, "preprocessing", None),
        # Параметры high-cardinality кодирования и новых стратегий (issue #20)
        "high_cardinality_threshold": cfg.general.high_cardinality_threshold,
        "high_cardinality_encoding": cfg.general.high_cardinality_encoding,
        "hashing_n_components": cfg.general.hashing_n_components,
        "target_encoding_smoothing": cfg.general.target_encoding_smoothing,
        "target_encoding_fallback": cfg.general.target_encoding_fallback,
    }
    # Отбор признаков: конфигурация из корневого Config + решение
    # победителя HPO (issue #11). Аргументы передаются только тренерам,
    # которые их поддерживают; при активном режиме отбора предупреждаем
    # о молчаливом игнорировании.
    if trainer_has_fs_cfg or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_cfg"] = cfg.general.feature_selection
    elif _feature_selection_mode(cfg.general.feature_selection) not in (
        None,
        "disabled",
    ):
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_cfg`; "
            "feature selection mode=%r will NOT be applied in the final fit.",
            algo_cfg.trainer_module,
            _feature_selection_mode(cfg.general.feature_selection),
        )

    if trainer_has_fs_active or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_active"] = fs_active
    elif fs_active is not None:
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_active`; "
            "the HPO decision use_feature_selection=%r will NOT be applied "
            "in the final fit.",
            algo_cfg.trainer_module,
            fs_active,
        )

    # Параметры валидации (issue #24, D3): в финальный fit передаются
    # разрешённые значения (метод + число фолдов/размер hold-out), а не строка
    # 'auto', чтобы финальный val_score считался тем же методом, что и оценка
    # сравнения моделей в HPO. Кастомным тренерам без поддержки аргументов
    # новые ключи не передаются (сигнатура проверяется, как для отбора
    # признаков); поведение по умолчанию ModelTrainer при этом совпадает.
    if validation_strategy is not None:
        if (
            "validation_strategy" in trainer_sig.parameters
            or trainer_accepts_var_kwargs
        ):
            trainer_kwargs["validation_strategy"] = validation_strategy
        else:
            _LOG.warning(
                "ModelTrainer %s does not accept `validation_strategy`; "
                "the resolved strategy %r will NOT be applied in the final fit.",
                algo_cfg.trainer_module,
                validation_strategy,
            )
    if n_folds is not None and (
        "n_folds" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["n_folds"] = n_folds
    if test_size is not None and (
        "test_size" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["test_size"] = test_size

    return trainer_module.ModelTrainer(**trainer_kwargs)


def _fit_and_save(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    model_path: Path,
    cfg: Config,
    metric_name_sklearn: str = "r2",
    *,
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int | None = None,
    test_size: float | None = None,
) -> Any:
    """Выполнить финальное обучение модели и сохранить результат на диск.
    Args:
        algo_name (str): Название выбранного алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путем к тренеру.
        X (pd.DataFrame): Полная матрица признаков для обучения.
        y (pd.Series): Полный вектор целевой переменной.
        best_params (Dict[str, Any]): Найденные оптимальные гиперпараметры.
        model_path (Path): Путь для сохранения файла модели.
        cfg (Config): Общий объект конфигурации для получения настроек оверсэмплинга.
        metric_name_sklearn (str): Имя основной метрики в формате sklearn.
        validation_strategy (ValidationStrategy | str | None): Разрешённая
            стратегия валидации финальной модели (issue #24, D3). ``None`` —
            дефолт ``ModelTrainer`` ('train_test_split').
        n_folds (int | None): Число фолдов (для 'k_fold').
        test_size (float | int | None): Размер hold-out части (доля либо число
            строк после резолюции 'auto').
    Returns:
        Any: Экземпляр ``ModelTrainer`` после обучения и сохранения
            (используется для доступа к значениям дополнительных метрик).
    Raises:
        AttributeError: Если в модуле тренера отсутствует класс `ModelTrainer`.
        ValueError: If ``best_params`` is ``None`` — a sign that HPO returned
            no valid hyperparameters (protection against the TypeError
            'NoneType' object is not iterable, issue #13).
    """
    trainer = _build_trainer(
        algo_name,
        algo_cfg,
        best_params,
        cfg,
        metric_name_sklearn=metric_name_sklearn,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        test_size=test_size,
    )
    trainer.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    return trainer


# --------------------------------------------------------------------------- #
#  Orchestration helpers (issue #31)                                          #
#  Шаги пайплайна, вынесенные из монолитного ``train_best_model``:           #
#  load_config → prepare_dataset → execute_phases → select_winner →          #
#  persist_artifact.                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreparedData:
    """Подготовленные данные и разрешённые настройки для фаз HPO.

    Единый контейнер, который ``prepare_dataset`` создаёт ровно один раз,
    а ``execute_phases`` / ``persist_artifact`` потребляют. Гарантирует, что
    все этапы пайплайна используют одинаковое представление признаков,
    метрик и стратегии валидации (issue #24, D3) без дублирования
    вычислений.

    Attributes:
        X (pd.DataFrame): Матрица признаков без целевой колонки.
        y (pd.Series): Вектор целевой переменной.
        metric_user (str): Пользовательское имя метрики сравнения
            (``general.comparison_metric``).
        metric_sklearn (str): Имя метрики в sklearn-семантике
            (например, ``neg_root_mean_squared_error``).
        resolved_validation (ValidationStrategy): Разрешённая стратегия
            валидации: 'auto' уже раскрыта в конкретный метод
            ('k_fold'/'train_test_split'/'loo').
        resolved_n_folds (int): Число фолдов (актуально для 'k_fold').
        resolved_test_size (float | int): Размер hold-out части — доля либо
            целое число строк после резолюции 'auto'.
        categorical_features (list[str]): Детектированные категориальные
            колонки.
        numerical_features (list[str]): Детектированные числовые колонки.
    """

    X: pd.DataFrame
    y: pd.Series
    metric_user: str
    metric_sklearn: str
    resolved_validation: ValidationStrategy
    resolved_n_folds: int
    resolved_test_size: float | int
    categorical_features: list[str]
    numerical_features: list[str]


def load_config(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[Config, str]:
    """Загрузить и провалидировать конфигурацию запуска.

    Шаг 1 пайплайна (issue #31). Проверяет входные данные и целевую колонку,
    приводит конфигурацию к объекту ``Config`` (из файла, словаря или уже
    готового объекта) и при необходимости настраивает файловое логирование.

    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): Конфигурация
            обучения.
        df (pd.DataFrame): Исходные данные.
        target (str | None): Имя целевого столбца. По умолчанию 'target'.

    Returns:
        Tuple[Config, str]: Кортеж (объект ``Config``, имя целевой колонки).

    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        ValueError: Если DataFrame пуст или целевая колонка отсутствует.
    """
    # Centralized validation входных данных до инициализации тяжёлых ресурсов
    validate_df_not_empty(df)
    target_col = target or "target"
    check_target_exists(df, target_col)

    # Если передана строка или Path, читаем файл.
    # Если объект Config или dict, обрабатываем их.
    if isinstance(config, Config):
        cfg = config
    elif isinstance(config, dict):
        _LOG.debug("CONFIG TYPE: %s", type(config))
        _LOG.debug(
            "ALGORITHMS: %s",
            (config.get("algorithms") if isinstance(config, dict) else "N/A"),
        )
        cfg = Config.model_validate(config)
    elif isinstance(config, (str, Path)):
        cfg = read_config(config)
    else:
        raise TypeError(
            f"Unsupported config type: {type(config)}. "
            f"Expected Config, dict, str, or Path."
        )

    # Если в конфиге указан путь к лог-файлу, настраиваем логирование
    if cfg.general.log_to_file:
        setup_logging(cfg.general.log_to_file)
    return cfg, target_col


def prepare_dataset(cfg: Config, df: pd.DataFrame, target_col: str) -> PreparedData:
    """Подготовить данные и разрешить настройки для фаз HPO.

    Шаг 2 пайплайна (issue #31). Разделяет DataFrame на X/y, резолвит
    стратегию валидации ('auto' → конкретный метод/фолды/test_size, issue
    #24, D3) и детектирует типы колонок ровно один раз — гарантия
    согласованной предобработки между HPO и финальным обучением.

    Args:
        cfg (Config): Объект конфигурации.
        df (pd.DataFrame): Исходные данные.
        target_col (str): Имя целевой колонки.

    Returns:
        PreparedData: Подготовленный контейнер данных и настроек.
    """
    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)

    # Centralized splitting
    X, y = prepare_X_y(df, target_col)

    # Единая резолюция стратегии валидации (issue #24, D3): 'auto' разрешается
    # ровно один раз по сырым X/y — до фаз HPO. Разрешённые значения (метод +
    # число фолдов / размер hold-out) передаются и в HPO, и в финальный fit,
    # чтобы оценка сравнения моделей и финальный val_score не расходились
    # (например, auto → train_test_split с целочисленным test_size из
    # choose_validation_method, а не фиксированным 0.2).
    val_method_eff, cv_obj, auto_decision = make_cv(
        n_samples=len(X),
        val_method=cfg.general.validation_strategy,
        n_folds=cfg.general.n_folds,
        random_state=RANDOM_STATE,
        test_size=0.2,
        n_features=X.shape[1],
    )
    resolved_validation = ValidationStrategy(val_method_eff)
    if val_method_eff == "k_fold":
        # Число фолдов берём из готового cv_obj: для 'auto'→kfold он содержит
        # k из решения choose_validation_method, отличное от cfg.general.n_folds.
        assert cv_obj is not None
        resolved_n_folds = int(cv_obj.get_n_splits())
    else:
        resolved_n_folds = cfg.general.n_folds
    # Для auto → train_test_split choose_validation_method возвращает целое
    # число строк; для явной стратегии — фиксированная доля 0.2.
    if val_method_eff == "train_test_split" and auto_decision is not None:
        resolved_test_size: float | int = int(auto_decision["test_size"])
    else:
        resolved_test_size = 0.2

    # Определяем типы колонок ровно один раз на входном DataFrame и прокидываем
    # в фазу HPO. Это гарантирует одинаковую предобработку (one-hot) между HPO
    # и финальным обучением ModelTrainer.
    categorical_features, numerical_features = detect_feature_types(X)

    return PreparedData(
        X=X,
        y=y,
        metric_user=metric_user,
        metric_sklearn=metric_sklearn,
        resolved_validation=resolved_validation,
        resolved_n_folds=resolved_n_folds,
        resolved_test_size=resolved_test_size,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )


def execute_phases(
    cfg: Config,
    prepared: PreparedData,
) -> tuple[dict[str, tuple[float, dict[str, Any]]], dict[str, str]]:
    """Выполнить все фазы HPO и вернуть результаты последней фазы.

    Шаг 3 пайплайна (issue #31). Содержит вложенные помощники
    ``prepare_search_space`` и ``_execute_hpo_phase`` (ранее объявленные
    внутри ``train_best_model``), цикл по фазам, параллельное выполнение,
    circuit breaker дисквалификации и фильтрацию невалидных результатов.

    Multi-phase semantics (issue #13): phase results are not accumulated —
    each phase rebuilds its results from scratch, so an algorithm that fails
    in the current phase (total HPO failure: returned ``None`` or an invalid
    score) is fully excluded and does not keep a stale record from a previous
    phase. The final winner is chosen by the results of the LAST phase: for
    the typical ``all_algorithms → refine_winner`` pipeline this is the
    refined winner; a winner failure during ``refine_winner`` raises
    ``RuntimeError`` instead of silently rolling back to the previous phase
    results.

    Args:
        cfg (Config): Объект конфигурации.
        prepared (PreparedData): Подготовленные данные и разрешённые настройки.

    Returns:
        Tuple[Dict[str, Tuple[float, Dict[str, Any]]], Dict[str, str]]:
            Кортеж (результаты последней фазы «алгоритм → (скор, параметры)»,
            дисквалифицированные алгоритмы «имя → причина»). Результаты пусты,
            если фазы не выполнялись (``general.phases`` пуст) или ни один
            алгоритм не дал валидного результата — тогда ``select_winner``
            не вызывается (защита в ``train_best_model``).

    Raises:
        RuntimeError: Если фаза 'refine_winner' не имеет предыдущих
            результатов, либо ни один алгоритм не дал валидного скора
            в фазе.
    """
    X = prepared.X
    y = prepared.y
    metric_sklearn = prepared.metric_sklearn
    resolved_validation = prepared.resolved_validation
    resolved_n_folds = prepared.resolved_n_folds
    resolved_test_size = prepared.resolved_test_size
    categorical_features = prepared.categorical_features
    numerical_features = prepared.numerical_features

    def prepare_search_space(
        algo_name: str, user_overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Подготовить пространство поиска гиперпараметров.
        Объединяет системные значения по умолчанию с пользовательскими
        переопределениями из конфигурации.
        Args:
            algo_name (str): Имя алгоритма для поиска дефолтов.
            user_overrides (Dict[str, Any] | None): Словарь с параметрами для замены.
        Returns:
            Dict[str, Any]: Итоговое пространство поиска.
        """
        # Получаем базовый спейс для алгоритма
        # (копируем, чтобы не менять глобальный объект)
        space = DEFAULT_SPACES.get(algo_name, {}).copy()

        if user_overrides:
            # Перезаписываем или добавляем параметры из конфига пользователя
            space.update(user_overrides)

        # Адаптивный epsilon для SVR (issue #55): границы поиска epsilon
        # масштабируются разбросом y_train в рантайме (tuner.resolve_search_space),
        # поэтому дефолтный слепой диапазон [1e-3, 1.0] из DEFAULT_SPACES
        # опускается. Если пользователь задал epsilon явно — он уже в space
        # (space.update(user_overrides) выше), и приоритет пользователя
        # сохраняется: адаптивная логика не вмешивается.
        if algo_name == "svr" and not (user_overrides and "epsilon" in user_overrides):
            space.pop("epsilon", None)

        return space

    def _execute_hpo_phase(
        phase_name: str,
        algo: str,
        a_cfg: AlgoCfg,
        n_trials: int,
        search_space: dict[str, Any] | None = None,
        initial_params: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]] | None:
        """Выполнить конкретную фазу HPO для алгоритма.
        Обеспечивает логирование этапа и обработку результатов оверсэмплинга.
        Args:
            phase_name (str): Название текущей фазы (напр. 'coarse').
            algo (str): Имя алгоритма.
            a_cfg (AlgoCfg): Конфигурация алгоритма.
            n_trials (int): Количество итераций в этой фазе.
            search_space (Dict[str, Any] | None): Пространство поиска для фазы.
        Returns:
            Tuple[float, Dict[str, Any]] | None: tuple (score, params),
                or ``None`` when HPO did not return a valid result (the
                algorithm is excluded from candidates, issue #13).
        Raises:
            Exception: Exceptions from ``_run_hpo`` are logged and re-raised
                (including ``InvalidAlgorithmError``).
        """
        _LOG.info(f"=== {phase_name} phase: {algo} ({n_trials} tries) ===")

        # Обращаемся к полям согласно определению в config_parser.py
        ovr = cfg.oversampling
        pruning_cfg: dict[str, Any] | None = None
        if cfg.general.pruning.enable:
            pruning_cfg = {
                "enable": cfg.general.pruning.enable,
                "strategy": cfg.general.pruning.strategy.value,
                "min_steps": cfg.general.pruning.min_steps,
                "n_startup_trials": cfg.general.pruning.n_startup_trials,
                "reduction_factor": cfg.general.pruning.reduction_factor,
            }

        try:
            result = _run_hpo(
                algo_name=algo,
                algo_cfg=a_cfg,
                X=X,
                y=y,
                metric_name_sklearn=metric_sklearn,
                n_trials=n_trials,
                validation_strategy=resolved_validation,
                n_folds=resolved_n_folds,
                train_test_split_test_size=resolved_test_size,
                search_space_override=search_space,
                data_oversampling=ovr.enable,
                data_oversampling_multiplier=ovr.multiplier,
                data_oversampling_algorithm=ovr.algorithm.value,  # .value т.к. это Enum
                initial_params=initial_params,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                encoding=cfg.general.categorical_encoding,
                pruning=pruning_cfg,
                high_cardinality_threshold=cfg.general.high_cardinality_threshold,
                high_cardinality_encoding=cfg.general.high_cardinality_encoding,
                hashing_n_components=cfg.general.hashing_n_components,
                target_encoding_smoothing=cfg.general.target_encoding_smoothing,
                target_encoding_fallback=cfg.general.target_encoding_fallback,
                feature_selection_cfg=cfg.general.feature_selection,
            )

            if result is None:
                return None

            score, params = result

            # Логируем значение в пользовательской семантике: для neg_*-метрик
            # (например, neg_root_mean_squared_error) «сырое» значение скорера
            # инвертировано, пользователю показываем естественное (положительное).
            # Направление сопровождает значение явно («min better» для ошибок,
            # issue #54): флаг оптимизатора greater_is_better не применяется.
            disp = to_user_value(metric_sklearn, score)
            _LOG.info(
                f"{phase_name} {algo:15} | score {disp:.5f} "
                f"({direction_label(metric_sklearn)}) | params {params}"
            )
            return score, params
        except Exception as err:
            _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
            raise

    # Начальный список кандидатов (все включенные алгоритмы)
    all_algorithms = _algorithms_as_dict(cfg.algorithms)
    phase_results: dict[str, tuple[float, dict[str, Any]]] = {}
    # Дисквалифицированные алгоритмы (circuit breaker): имя -> причина.
    # Дисквалификация не прерывает запуск: алгоритм исключается из
    # кандидатов и результатов, остальные продолжают обучение (issue #12).
    disqualified_algorithms: dict[str, str] = {}
    for phase in cfg.general.phases:
        _LOG.info(
            f"--- Starting Phase: {phase.name} ({phase.n_trials}"
            f" trials, action: {phase.action}) ---"
        )

        # Snapshot of the previous phases' results. Used ONLY to select the
        # refine_winner and to pass initial_params to the worker. Records from
        # previous phases are NOT carried into the current one: phase_results
        # is rebuilt from the current phase only, so an algorithm that failed
        # in it (returned None or an invalid score) does not keep a stale
        # record from a previous phase and does not compete with real results
        # (issue #13). Consequence: the final winner is chosen by the results
        # of the LAST phase.
        prev_results = phase_results

        if phase.action == "refine_winner":
            if not prev_results:
                raise RuntimeError(
                    f"Phase '{phase.name}' requires a winner,"
                    f" but no previous results exist."
                )

            # Детерминированный выбор победителя (issue #32): максимальный
            # «сырой» скор; ничья разрешается порядком конфигурации.
            winner_algo = select_winner(prev_results)
            _LOG.info(f"Phase '{phase.name}' filtering for winner: {winner_algo}")
            current_candidates = {winner_algo: all_algorithms[winner_algo]}
            # Only the previous phase's winner takes part in refine_winner:
            # the losers' records must not "resurrect" if the winner fails
            # during refinement. The refine result replaces the winner's
            # record; if the winner fails (None/invalid score) the phase ends
            # with RuntimeError — no silent rollback to phase 1.
        else:
            # all_algorithms phase: every enabled algorithm, results from
            # previous phases are not accumulated. Algorithms disqualified by
            # the circuit breaker do not return to the phase (issue #12).
            current_candidates = {
                n: a
                for n, a in all_algorithms.items()
                if a.enable and n not in disqualified_algorithms
            }

        # Each phase starts with an empty phase_results: it is rebuilt from the
        # current phase only, so an algorithm that fails in this phase does not
        # keep a stale record from a previous phase (issue #13).
        phase_results = {}

        # Кандидаты, участвующие в текущей фазе (нужны для диагностики
        # в сообщении об отсутствии валидных результатов)
        phase_candidates = list(current_candidates.keys())

        def _worker(
            algo_name: str,
            algo_cfg: AlgoCfg,
            p: HPOPhaseCfg = phase,
            pr: dict[str, tuple[float, dict[str, Any]]] = prev_results,
        ) -> tuple[str, float, dict[str, Any]] | None:
            """Воркер для параллельного или последовательного запуска задачи HPO.
            Args:
                algo_name (str): Имя алгоритма.
                algo_cfg (AlgoCfg): Конфигурация алгоритма.
            Returns:
                Optional[Tuple[str, float, Dict[str, Any]]]: Название, скор и параметры
                    или None в случае ошибки.
            """
            # Determine initial_params for refine_winner: keep the best parameters of
            # the previous phase for enqueue_trial. The prev_results snapshot is
            # fixed via a default argument (like `p`), so the worker uses the
            # state at definition time (B023). Workers only READ this dict;
            # writes to phase_results happen after all phase workers finish, and
            # at the end of the phase phase_results is re-bound to the filtered
            # valid_results (objects are never mutated) — therefore the snapshot
            # is stable in both parallel and sequential modes.
            init_params = None
            if p.action == "refine_winner" and algo_name in pr:
                _, prev_params = pr[algo_name]
                init_params = prev_params

            # 1. Берем системные дефолты + накладываем то, что в AlgoCfg (из YAML/JSON)
            full_search_space = prepare_search_space(
                algo_name,
                algo_cfg.hyperparameters,  # это dict из вашего config_parser
            )
            try:
                result = _execute_hpo_phase(
                    p.name,
                    algo_name,
                    algo_cfg,
                    p.n_trials,
                    full_search_space,
                    initial_params=init_params,
                )

                if result is None:
                    return None

                score, params = result

                return algo_name, score, params
            except _CanonicalIAE as err:
                # Circuit breaker (issue #12): алгоритм дисквалифицирован после
                # MAX_FATAL_FAILURES подряд фатальных ошибок. Это НЕ аварийный
                # стоп всего запуска — алгоритм исключается из кандидатов,
                # остальные продолжают обучение (аналогично None-результату).
                _LOG.warning(
                    "Algorithm %s disqualified in phase %s: %s",
                    algo_name,
                    p.name,
                    err,
                )
                disqualified_algorithms[algo_name] = str(err)
                return None
            except Exception as e:  # noqa: BLE001
                _LOG.warning(f"Algorithm {algo_name} failed in phase {p.name}: {e}")
                return None

        # Выполнение (параллельное или последовательное)
        if (
            cfg.general.parallel_strategy == "algorithms"
            and len(current_candidates) > 1
        ):
            results = run_parallel(
                _worker,
                args_seq=[(n, a) for n, a in current_candidates.items()],
                max_workers=cfg.general.max_workers,
                mode=cfg.general.parallel_mode,
                timeout=cfg.general.phase_timeout or 3600,
                task_timeout=cfg.general.task_timeout,
            )

            for res in results:
                if res:
                    name, sc, pr = res
                    phase_results[name] = (sc, pr)
        else:
            for n, a in current_candidates.items():
                res = _worker(n, a)
                if res:
                    name, sc, pr = res
                    phase_results[name] = (sc, pr)

        # Disqualified algorithms (circuit breaker) are removed from the results and
        # from the candidates of the following phases: they must not win and must
        # not waste compute again.
        # phase_results.pop is a safety net for disqualification in the same
        # phase (a no-op for the already-rebuilt dict).
        for name in list(disqualified_algorithms):
            phase_results.pop(name, None)
            current_candidates.pop(name, None)

        # Filter out invalid phase results: None/NaN/±inf and the worst-score
        # sentinel class. The sentinel class is defined as any score at or below
        # the exact float32 minimum (WORST_SCORE_THRESHOLD): it covers both the
        # HPO_WORST_SCORE constant (returned by the tuner when a trial metric is
        # non-finite) and the raw float(np.finfo(np.float32).min) value (possible
        # from numpy-cast metrics or custom tuners), which the old
        # math.isclose(rel_tol=1e-9) check missed (relative difference ≈ 9.88e-9,
        # issue #32). Real metrics of such magnitude are practically impossible,
        # so the threshold cannot drop a legitimate winner. params must be a
        # dict (the tuner failure signal is params=None, issue #13); garbage
        # score types from custom tuners (None, strings, etc.) are rejected by
        # is_valid_winner_score.
        valid_results: dict[str, tuple[float, dict[str, Any]]] = {}
        for name, (score, params) in phase_results.items():
            if not isinstance(params, dict):
                _LOG.warning(
                    "Algorithm %s produced invalid params %r (expected dict); "
                    "excluding from phase candidates",
                    name,
                    params,
                )
                continue
            if not is_valid_winner_score(score):
                _LOG.warning(
                    "Algorithm %s produced invalid score %r (non-finite or "
                    "worst-score sentinel); excluding from phase candidates",
                    name,
                    score,
                )
                continue
            valid_results[name] = (score, params)

        # Keep only the valid results of the current phase in phase_results: entries
        # with an invalid score (HPO_WORST_SCORE etc.) are removed. Since
        # phase_results started the phase empty, here it is guaranteed to
        # contain only this phase's results (issue #13).
        phase_results = valid_results

        if not valid_results:
            failed_algos = [n for n in phase_candidates if n not in valid_results]
            raise RuntimeError(
                f"No algorithms produced valid scores in phase '{phase.name}'. "
                f"Failed algorithms: {failed_algos}"
            )
    return phase_results, disqualified_algorithms


def persist_artifact(
    *,
    cfg: Config,
    prepared: PreparedData,
    winner_algo: str,
    final_score: float,
    final_params: dict[str, Any],
    model_path_override: str | Path | None,
    disqualified_algorithms: dict[str, str],
    pre_trained_trainer: Any | None = None,
    disqualified_by_sanity_gate: dict[str, list[str]] | None = None,
    sanity_gate_warn_only: dict[str, Any] | None = None,
    sanity_gate_overhead_seconds: float | None = None,
) -> dict[str, Any]:
    """Финальное обучение победителя, сохранение артефакта и сборка отчёта.

    Шаг 5 пайплайна (issue #31). Обучает модель алгоритма-победителя на
    полном наборе данных через ``_fit_and_save`` (либо переиспользует уже
    обученного финалиста при двухстадийном выборе T4), сохраняет артефакт на
    диск и формирует словарь результата: ``algorithm``, ``score`` (в
    пользовательской семантике, issue #26), ``metric``, ``params``,
    ``model_path``; при наличии добавляются ``disqualified_algorithms``
    (issue #12), ``additional_metrics`` и ключи Sanity Gate (T6):
    ``disqualified_by_sanity_gate``, ``sanity_gate_warn_only``,
    ``sanity_gate_overhead_seconds``. Источники дисквалификаций не смешиваются:
    circuit breaker (``disqualified_algorithms``) и Sanity Gate
    (``disqualified_by_sanity_gate``) — отдельные ключи.

    Args:
        cfg (Config): Объект конфигурации.
        prepared (PreparedData): Подготовленные данные и настройки.
        winner_algo (str): Имя алгоритма-победителя.
        final_score (float): «Сырой» скор победителя (уже прошёл инвариант
            ``is_valid_winner_score``).
        final_params (Dict[str, Any]): Лучшие гиперпараметры победителя.
        model_path_override (str | Path | None): Альтернативный путь
            сохранения модели.
        disqualified_algorithms (Dict[str, str]): Дисквалифицированные
            алгоритмы (имя → причина); добавляются в результат при наличии.
        pre_trained_trainer (Any | None): Уже обученный на 100% данных
            ``ModelTrainer`` победителя (двухстадийный выбор T4). Если задан,
            повторное обучение НЕ выполняется — артефакт сохраняется из него
            (требование 4 постановки T4: финалисты обучаются один раз).
        disqualified_by_sanity_gate (Dict[str, list[str]] | None): Применённые
            дисквалификации Sanity Gate «алгоритм → причины» (режим ``active``,
            T6). Ключ ``disqualified_by_sanity_gate`` добавляется в результат
            только при наличии дисквалификаций.
        sanity_gate_warn_only (Dict[str, Any] | None): Секция статистики
            режима ``warn_only`` (T6): гипотетические дисквалификации
            (``disqualified``) и гипотетический победитель при ``active``
            (``winner_if_active``). Передаётся только в режиме ``warn_only``.
        sanity_gate_overhead_seconds (float | None): Накладные расходы гейта
            (время обучения финалистов на 100% данных + аудит, T6).
            Добавляется в результат только в режимах с пулом/аудитом.

    Returns:
        Dict[str, Any]: Словарь с результатами обучения.

    Raises:
        Exception: Пробрасывается ошибка финального обучения/сохранения
            (логируется перед повторным поднятием).
    """
    model_path = Path(model_path_override or cfg.general.path_to_model)
    winner_cfg = _algorithms_as_dict(cfg.algorithms)[winner_algo]

    try:
        if pre_trained_trainer is not None:
            trainer = pre_trained_trainer
            model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(model_path)
        else:
            trainer = _fit_and_save(
                winner_algo,
                winner_cfg,
                prepared.X,
                prepared.y,
                final_params,
                model_path,
                cfg,
                metric_name_sklearn=prepared.metric_sklearn,
                validation_strategy=prepared.resolved_validation,
                n_folds=prepared.resolved_n_folds,
                test_size=prepared.resolved_test_size,
            )
        _LOG.info("Model saved to %s", model_path.resolve())
    except Exception as e:
        _LOG.error(f"Failed to save final model: {e}")
        raise

    result: dict[str, Any] = {
        "algorithm": winner_algo,
        # Единая пользовательская семантика (issue #26): score всегда отражает
        # естественное значение метрики (положительный RMSE/MAE, обычный R²),
        # metric — пользовательское имя метрики сравнения из конфигурации.
        # «Сырое» (инвертированное для neg_*-скореров) значение остаётся
        # внутренней деталью оптимизатора и до пользователя не доходит.
        "score": to_user_value(prepared.metric_user, final_score),
        "metric": prepared.metric_user,
        "params": final_params,
        "model_path": str(model_path),
    }
    # Диагностика circuit breaker (issue #12): какие алгоритмы и почему были
    # дисквалифицированы. Ключ добавляется только при наличии дисквалификаций,
    # иначе результаты идентичны прежнему поведению.
    if disqualified_algorithms:
        result["disqualified_algorithms"] = dict(disqualified_algorithms)
    # Дисквалификации Sanity Gate (T6, режим active): причины отдельным ключом
    # от circuit breaker — источники не смешиваются. Ключ добавляется только
    # при наличии дисквалификаций.
    if disqualified_by_sanity_gate:
        result["disqualified_by_sanity_gate"] = {
            name: list(reasons) for name, reasons in disqualified_by_sanity_gate.items()
        }
    # Статистика warn_only (T6): гипотетические дисквалификации и
    # гипотетический победитель при mode=active. Ключ присутствует только в
    # режиме warn_only (None в остальных).
    if sanity_gate_warn_only is not None:
        result["sanity_gate_warn_only"] = dict(sanity_gate_warn_only)
    # Накладные расходы гейта (T6): обучение финалистов на 100% данных + аудит.
    # Добавляется только в режимах с пулом/аудитом (mode != 'off').
    if sanity_gate_overhead_seconds is not None:
        result["sanity_gate_overhead_seconds"] = float(sanity_gate_overhead_seconds)
    # Дополнительные информационные метрики финальной модели. Ключ добавляется
    # только при наличии метрик (после дедупликации и исключения основной
    # метрики сравнения), иначе результаты идентичны прежнему поведению.
    if cfg.general.additional_metrics:
        result["additional_metrics"] = dict(trainer.additional_scores)
    return result


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #


def train_best_model(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
    model_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Основной интерфейс обучения лучшей модели.

    Оркестрирует декомпозированный пайплайн (issue #31):
    ``load_config → prepare_dataset → execute_phases → select_winner →
    persist_artifact``. Логика каждого шага вынесена в отдельные функции
    модуля; здесь остаются только выбор победителя и инвариант winner-скора.

    Двухстадийный выбор победителя (эпик #61, T4): при
    ``general.sanity_gate.mode != 'off'`` после фаз HPO строится пул
    финалистов (``select_finalists``, T3), каждый финалист обучается на 100%
    данных ровно один раз, прогоняется через ``ModelSanityGate`` (T2) и
    победитель ранжируется по ``RMSE_oof`` (``select_robust_winner``).
    Режим ``warn_only`` собирает статистику аудита без применения
    дисквалификаций; режим ``active`` применяет их. При ``mode='off'``
    поведение полностью идентично старому одностадийному выбору.

    Multi-phase semantics (issue #13): phase results are not accumulated — each
    phase rebuilds its results from scratch, so an algorithm that fails in the
    current phase (total HPO failure: returned ``None`` or an invalid score) is
    fully excluded and does not keep a stale record from a previous phase.
    The final winner is chosen by the results of the LAST phase: for the typical
    ``all_algorithms → refine_winner`` pipeline this is the refined winner;
    a winner failure during ``refine_winner`` raises ``RuntimeError`` instead of
    silently rolling back to the previous phase results.
    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): Конфигурация обучения.
            Может быть путем к файлу, словарем или объектом Config.
        df (pd.DataFrame): Исходные данные.
        target (str | None): Имя целевого столбца. По умолчанию 'target'.
        model_path_override (str | Path | None): Альтернативный путь сохранения модели.
    Returns:
        Dict[str, Any]: Словарь с результатами: название алгоритма,
            ``score`` — значение метрики в пользовательской семантике
            (положительный RMSE/MAE, обычный R²; для neg_-скореров значение
            инвертировано обратно), ``metric`` — пользовательское имя метрики
            сравнения, параметры и путь к файлу. Если в конфигурации заданы
            дополнительные метрики (``general.additional_metrics``),
            в результат добавляется ключ ``additional_metrics`` — словарь
            {метрика: значение}, рассчитанных для финальной модели на том же
            наборе данных, что и основная метрика. Если в ходе запуска
            какие-либо алгоритмы были дисквалифицированы circuit breaker'ом
            (5 подряд фатальных ошибок), в результат добавляется ключ
            ``disqualified_algorithms`` — словарь {имя алгоритма: причина
            дисквалификации}. При ``general.sanity_gate.mode != 'off'``
            добавляется ключ ``sanity_gate`` со статистикой аудита для отчёта
            (T6): режим, пул, дисквалификации, fallback-флаг и RMSE победителя.
            Дополнительно (T6): ``sanity_gate_overhead_seconds`` — накладные
            расходы гейта (переобучение финалистов на 100% данных + аудит);
            в режиме ``active`` при наличии дисквалификаций —
            ``disqualified_by_sanity_gate`` {алгоритм: [причины]} (отдельный
            ключ от circuit breaker, источники не смешиваются); в режиме
            ``warn_only`` — ``sanity_gate_warn_only`` с гипотетическими
            дисквалификациями (``disqualified``) и гипотетическим победителем
            при ``mode=active`` (``winner_if_active``). При ``mode='off'`` все
            новые ключи отсутствуют (результат обратно совместим).
    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        RuntimeError: Если ни один алгоритм не смог успешно завершить фазу HPO
            либо ни один финалист не обучился на полном наборе данных.
            Дисквалификация отдельного алгоритма (``InvalidAlgorithmError``)
            запуск НЕ прерывает: алгоритм исключается из кандидатов, остальные
            продолжают обучение (issue #12).
    """
    cfg, target_col = load_config(config, df, target)
    prepared = prepare_dataset(cfg, df, target_col)
    phase_results, disqualified_algorithms = execute_phases(cfg, prepared)

    # After all phases, pick the final winner. Since phase_results is rebuilt per
    # phase, the winner is chosen by the LAST phase's results (for the typical
    # all_algorithms → refine_winner pipeline this is the refined winner).
    # Детерминированный tie-break при равных скорах: побеждает первый
    # алгоритм в порядке конфигурации (select_winner, issue #32).
    if not phase_results:
        # Защита от пустого списка фаз (конфиг допускает phases: []): цикл
        # в execute_phases не выполнялся, и select_winner({}) бросил бы
        # ValueError — поднимаем понятную RuntimeError (ревью PR #22).
        raise RuntimeError("No valid results after HPO phases; cannot select a winner.")

    sg_cfg = cfg.general.sanity_gate

    # ── Двухстадийный выбор победителя (эпик #61, T4) ────────────────────── #
    # Режим 'off' — старое поведение select_winner: пул и аудит НЕ выполняются,
    # поведение пайплайна идентично прежнему (первый финалист обучается один
    # раз внутри persist_artifact).
    if sg_cfg.mode == "off":
        winner_algo = select_winner(phase_results)
        final_score, final_params = phase_results[winner_algo]
        # Финальный инвариант (issue #32): score победителя обязан быть конечным
        # и строго выше класса worst-score сентинела, прежде чем попасть в
        # result["score"]. Иначе сентинел мог бы протечь в отчёт — для neg_-
        # метрик to_user_value инвертировал бы его в абсурдные +3.4e38.
        if not is_valid_winner_score(final_score):
            raise RuntimeError(
                f"Winner '{winner_algo}' produced an invalid final score "
                f"{final_score!r} (non-finite or worst-score sentinel); "
                f"refusing to report it."
            )
        return persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo=winner_algo,
            final_score=final_score,
            final_params=final_params,
            model_path_override=model_path_override,
            disqualified_algorithms=disqualified_algorithms,
        )

    # Режимы warn_only / active: пул финалистов (T3) → обучение на 100%
    # данных один раз → аудит ModelSanityGate (T2) → ранжирование по RMSE_oof.
    pool = select_finalists(
        phase_results,
        metric_user=prepared.metric_user,
        top_k_candidates=sg_cfg.top_k_candidates,
        corridor_delta=sg_cfg.corridor_delta,
        corridor_mode=sg_cfg.corridor_mode,
        enforce_family_diversity=sg_cfg.enforce_family_diversity,
        algorithm_families=sg_cfg.algorithm_families,
        allowed_families=(
            set(sg_cfg.allowed_families)
            if sg_cfg.allowed_families is not None
            else None
        ),
        family_diversity_multiplier=sg_cfg.family_diversity_multiplier,
        disqualified=disqualified_algorithms,
    )
    _LOG.info(
        "[sanity_gate] mode=%s: finalists pool (top_k=%d, delta=%s): %s",
        sg_cfg.mode,
        sg_cfg.top_k_candidates,
        sg_cfg.corridor_delta,
        pool,
    )

    # Накладные расходы гейта (T6): время переобучения финалистов на 100%
    # данных + фазы аудита. Замеряется от старта обучения финалистов до
    # завершения выбора robust-победителя; попадает в отчёт отдельным ключом.
    gate_overhead_started = time.monotonic()
    candidates = _train_finalists(pool, phase_results, cfg, prepared)
    if not candidates:
        raise RuntimeError(
            "No finalist could be trained on the full dataset; "
            "cannot select a robust winner."
        )
    failed_training = [name for name in pool if name not in candidates]

    gate = ModelSanityGate(
        min_prediction_diversity=sg_cfg.min_prediction_diversity,
        adaptive_diversity=sg_cfg.adaptive_diversity,
        min_unique_count=sg_cfg.min_unique_count,
        min_unique_ratio=sg_cfg.min_unique_ratio,
        max_dead_feature_ratio=sg_cfg.max_dead_feature_ratio,
        dead_features_require_low_diversity=sg_cfg.dead_features_require_low_diversity,
        zero_coef_tolerance=sg_cfg.zero_coef_tolerance,
        max_generalization_gap=sg_cfg.max_generalization_gap,
        check_permutation_sensitivity=sg_cfg.check_permutation_sensitivity,
        permutation_max_rows=sg_cfg.permutation_max_rows,
        permutation_max_features=sg_cfg.permutation_max_features,
        permutation_repeats=sg_cfg.permutation_repeats,
        permutation_seed=sg_cfg.permutation_seed,
        permutation_tolerance=sg_cfg.permutation_tolerance,
        check_diversity=sg_cfg.check_diversity,
        check_unique=sg_cfg.check_unique,
        check_dead_features=sg_cfg.check_dead_features,
        check_generalization_gap=sg_cfg.check_generalization_gap,
    )
    robust_result = select_robust_winner(
        candidates,
        prepared.X,
        prepared.y,
        gate,
        mode=sg_cfg.mode,
        audit_time_budget_seconds=sg_cfg.audit_time_budget_seconds,
    )
    sanity_gate_overhead_seconds = time.monotonic() - gate_overhead_started
    winner_algo = robust_result.winner_algo
    final_score, final_params = phase_results[winner_algo]
    if not is_valid_winner_score(final_score):
        raise RuntimeError(
            f"Winner '{winner_algo}' produced an invalid final score "
            f"{final_score!r} (non-finite or worst-score sentinel); "
            f"refusing to report it."
        )

    # Лог итогов аудита победителя (T6): по логам запуска можно восстановить,
    # почему модель стала победителем (diversity/unique/gap/dead features).
    winner_audit = robust_result.audit.get(winner_algo)
    if winner_audit is not None:
        _LOG.info(
            "[sanity_gate] Winner %s audit summary: diversity_ratio=%s, "
            "unique_ratio=%s, dead_features_count=%d, generalization_gap=%s",
            winner_algo,
            _fmt_audit_value(winner_audit.diversity_ratio),
            _fmt_audit_value(winner_audit.unique_ratio),
            winner_audit.dead_features_count,
            _fmt_audit_value(winner_audit.generalization_gap),
        )

    # Секции отчёта T6. Источники дисквалификаций не смешиваются: в active —
    # disqualified_by_sanity_gate (применённые), в warn_only —
    # sanity_gate_warn_only (гипотетические + победитель при mode=active).
    disqualified_by_sanity_gate: dict[str, list[str]] | None = None
    sanity_gate_warn_only: dict[str, Any] | None = None
    if sg_cfg.mode == "warn_only":
        sanity_gate_warn_only = {
            "disqualified": {
                name: list(reasons)
                for name, reasons in robust_result.disqualified.items()
            },
            "winner_if_active": robust_result.winner_if_active,
        }
    else:
        # mode == "active": применяются фактически.
        if robust_result.disqualified:
            disqualified_by_sanity_gate = {
                name: list(reasons)
                for name, reasons in robust_result.disqualified.items()
            }

    result = persist_artifact(
        cfg=cfg,
        prepared=prepared,
        winner_algo=winner_algo,
        final_score=final_score,
        final_params=final_params,
        model_path_override=model_path_override,
        disqualified_algorithms=disqualified_algorithms,
        pre_trained_trainer=candidates[winner_algo].trainer,
        disqualified_by_sanity_gate=disqualified_by_sanity_gate,
        sanity_gate_warn_only=sanity_gate_warn_only,
        sanity_gate_overhead_seconds=sanity_gate_overhead_seconds,
    )

    # Статистика Sanity Gate для отчёта (T6): причины дисквалификаций,
    # результаты аудита и режим. Добавляется только в режимах с пулом/аудитом,
    # в 'off' результат идентичен прежнему поведению.
    result["sanity_gate"] = {
        "mode": robust_result.mode,
        "pool": list(robust_result.pool),
        "fallback_used": robust_result.fallback_used,
        "winner_rmse_oof": robust_result.winner_rmse_oof,
        "winner_rmse_full": candidates[winner_algo].rmse_full,
        "failed_training": list(failed_training),
        "disqualified": {
            name: list(reasons) for name, reasons in robust_result.disqualified.items()
        },
        "audit": {
            name: {
                "is_valid": audit.is_valid,
                "reasons": list(audit.reasons),
                "soft_warnings": list(audit.soft_warnings),
                "diversity_ratio": audit.diversity_ratio,
                "unique_ratio": audit.unique_ratio,
                "dead_features_count": audit.dead_features_count,
                "generalization_gap": audit.generalization_gap,
                "severity": audit.severity,
            }
            for name, audit in robust_result.audit.items()
        },
    }
    return result
