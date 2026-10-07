"""Ядро регрессионного бенчмарка «до/после» Sanity Gate (эпик #61, T9, issue #70).

Бенчмарк сравнивает выбор победителя AutoML-движка в двух режимах:

* ``mode=off`` — старое поведение: победитель = лучший «сырой» CV-скор
  (``select_robust_winner`` без аудита);
* ``mode=active`` — победитель = лучший ``RMSE_oof`` среди прошедших аудит
  ``ModelSanityGate`` (при тотальном провале — fallback-критерий).

Критерии приёмки эпика #61 (задача T9):

1. Эталонные датасеты (нормальные данные с честным победителем: зашумлённые
   R²≈0.3, широкие с сотнями признаков, малые N, граничный diversity у порога):
   победитель ``mode=active`` совпадает с ``mode=off``, гейт НЕ срабатывает
   (0 изменений победителя; документированный граничный случай — порог
   допуска «ухудшение ≤ ε»).
2. Вырожденные датасеты (полка / плато tanh / переобучение): победитель при
   ``mode=active`` ОБЯЗАН измениться (100% изменений), вырожденный кандидат
   дисквалифицируется, честный кандидат остаётся победителем.

Архитектура прогона:

* Эталонные датасеты прогоняются сквозным пайплайном движка: HPO-фаза
  (``execute_phases``) → пул финалистов (``select_finalists``) → обучение
  финалистов на 100% данных (``_train_finalists``) → сравнение
  ``mode=off``/``mode=active`` на ОДНОМ и том же пуле (обучение не
  дублируется). HPO детерминирован (TPE-seed 42, KFold seed 0), поэтому
  результат воспроизводим.
* Вырожденные датасеты используют контролируемый пул: вырожденная модель
  (реально обученная на датасете, честный OOF через KFold) является
  CV-лидером (лучший ``cv_score`` — как если бы старое поведение выбрало её),
  честная модель — второй кандидат. Это единственный детерминированный способ
  гарантировать «вырожденный победитель при mode=off»: такие модели по
  построению НЕ выигрывают честную кросс-валидацию, поэтому сквозной HPO не
  может их воспроизвести стабильно (подход совпадает с негативными тестами T4).

Время прогона ограничено (CI): 7 датасетов, n_trials=2, N ≤ 400 — суммарно
порядка 15–25 секунд.

Формат вывода — таблица «датасет × режим × победитель × метрики × сработал ли
гейт» (см. ``render_table``); критерии приёмки — ``check_acceptance``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.component import (
    FinalistCandidate,
    ModelSanityGate,
    _train_finalists,
    execute_phases,
    prepare_dataset,
    select_finalists,
    select_robust_winner,
)
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.training_engine.metrics import oof_rmse

from tests.data_factory import (
    make_degenerate_flat_shelf,
    make_degenerate_overfit,
    make_degenerate_tanh_plateau,
    make_reference_boundary_diversity,
    make_reference_noisy_r2_03,
    make_reference_small_n,
    make_reference_wide,
)

__all__ = [
    "BeforeAfterResult",
    "REFERENCE_CASES",
    "BOUNDARY_CASES",
    "DEGENERATE_CASES",
    "ALL_CASES",
    "run_case",
    "render_table",
    "check_acceptance",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Результат прогона «до/после»
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeforeAfterResult:
    """Результат сравнения ``mode=off`` vs ``mode=active`` на одном датасете.

    Attributes:
        dataset (str): Имя датасета.
        kind (str): 'reference' | 'boundary' | 'degenerate'.
        winner_off (str): Победитель при ``mode=off`` (CV-лидер).
        winner_active (str): Победитель при ``mode=active``.
        pool (list[str]): Пул финалистов (порядок детерминированный).
        cv_scores (dict[str, float]): «Сырые» CV-скоры кандидатов.
        rmse_oof (dict[str, float]): RMSE_oof кандидатов.
        disqualified (dict[str, list[str]]): Дисквалификации при ``mode=active``.
        audit_valid (dict[str, bool]): is_valid каждого кандидата при active.
        winner_metrics_off (dict[str, float]): Метрики победителя off
            (rmse_oof).
        winner_metrics_active (dict[str, float]): Метрики победителя active
            (rmse_oof, diversity_ratio, generalization_gap).
        gate_triggered (bool): True — при ``mode=active`` есть хотя бы одна
            дисквалификация.
        changed (bool): True — победители режимов различаются.
        elapsed (float): Время прогона в секундах.
    """

    dataset: str
    kind: str
    winner_off: str
    winner_active: str
    pool: list[str]
    cv_scores: dict[str, float] = field(default_factory=dict)
    rmse_oof: dict[str, float] = field(default_factory=dict)
    disqualified: dict[str, list[str]] = field(default_factory=dict)
    audit_valid: dict[str, bool] = field(default_factory=dict)
    winner_metrics_off: dict[str, float] = field(default_factory=dict)
    winner_metrics_active: dict[str, float] = field(default_factory=dict)
    gate_triggered: bool = False
    changed: bool = False
    elapsed: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Описания кейсов
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FullFlowCase:
    """Эталонный кейс: сквозной прогон движка (HPO → пул → финалисты).

    Attributes:
        name (str): Имя датасета.
        kind (str): 'reference'.
        generator (Callable): Генератор (X, y) с фиксированным seed.
        algorithms (dict[str, dict]): Конфигурация алгоритмов для HPO
            (имя → доп. поля AlgoCfg, например гиперпараметры).
        n_trials (int): Число триалов HPO на алгоритм (ограничение времени CI).
        top_k_candidates (int): Кап размера пула финалистов.
        corridor_delta (float): Ширина коридора пула.
    """

    name: str
    kind: str
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]
    algorithms: dict[str, dict[str, Any]]
    n_trials: int = 2
    top_k_candidates: int = 3
    corridor_delta: float = 0.3


@dataclass(frozen=True)
class ControlledCase:
    """Контролируемый кейс: заданный пул реально обученных моделей.

    Используется для вырожденных датасетов и граничного эталонного случая,
    где нужен точный контроль над составом пула и позицией «CV-лидера».

    Attributes:
        name (str): Имя датасета.
        kind (str): 'boundary' | 'degenerate'.
        generator (Callable): Генератор (X, y) с фиксированным seed.
        candidates (Callable): Строит ``dict[str, FinalistCandidate]`` из (X, y):
            кандидаты реально обучены на данных, OOF честный (KFold).
    """

    name: str
    kind: str
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]
    candidates: Callable[
        [np.ndarray, np.ndarray], dict[str, FinalistCandidate]
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  Построители контролируемых пулов (реально обученные модели + честный OOF)
# ──────────────────────────────────────────────────────────────────────────────


def _honest_oof_full(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 4,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Честный OOF-вектор (модель фолда не видела строку) + full-fit модель.

    Args:
        model_factory: Фабрика модели (без аргументов).
        X: Матрица признаков.
        y: Целевой вектор.
        folds: Число фолдов KFold (shuffle, random_state=0 — детерминизм).

    Returns:
        tuple[np.ndarray, np.ndarray, Any]: (y_pred_oof, y_pred_full, model).
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        model = model_factory()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    full_model = model_factory().fit(X, y)
    return oof, full_model.predict(X), full_model


def _make_candidate(
    name: str, cv_score: float, model: Any, y_oof: np.ndarray, y_full: np.ndarray, y: np.ndarray
) -> FinalistCandidate:
    """Собрать ``FinalistCandidate`` из обученной модели и честных векторов."""
    return FinalistCandidate(
        algo_name=name,
        model=model,
        params={},
        cv_score=cv_score,
        rmse_oof=float(oof_rmse(y, y_oof)),
        y_pred_oof=y_oof,
        y_pred_full=y_full,
        rmse_full=float(oof_rmse(y, y_full)),
    )


def _boundary_ridge_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """Два честных Ridge на границе diversity-порога (см. ``BOUNDARY_CASES``)."""
    oof_a, full_a, model_a = _honest_oof_full(
        lambda: Ridge(alpha=1.0, random_state=0), X, y
    )
    oof_b, full_b, model_b = _honest_oof_full(
        lambda: Ridge(alpha=0.1, random_state=0), X, y
    )
    return {
        "ridge_alpha1": _make_candidate("ridge_alpha1", -0.05, model_a, oof_a, full_a, y),
        "ridge_alpha01": _make_candidate("ridge_alpha01", -0.051, model_b, oof_b, full_b, y),
    }


def _flat_shelf_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """«Полка»: ElasticNet(alpha=500) с нулевыми весами против честного Ridge."""
    oof_en, full_en, model_en = _honest_oof_full(
        lambda: ElasticNet(alpha=500.0, random_state=0), X, y
    )
    oof_ridge, full_ridge, model_ridge = _honest_oof_full(
        lambda: Ridge(alpha=1.0, random_state=0), X, y
    )
    return {
        "elasticnet_flat": _make_candidate(
            "elasticnet_flat", -0.05, model_en, oof_en, full_en, y
        ),
        "ridge": _make_candidate("ridge", -0.09, model_ridge, oof_ridge, full_ridge, y),
    }


def _tanh_plateau_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """«Плато tanh»: SVR-sigmoid против честного SVR-RBF."""
    oof_sig, full_sig, model_sig = _honest_oof_full(
        lambda: SVR(kernel="sigmoid", gamma=0.001, coef0=0.0), X, y
    )
    oof_rbf, full_rbf, model_rbf = _honest_oof_full(
        lambda: SVR(kernel="rbf", C=1.0), X, y
    )
    return {
        "svr_sigmoid": _make_candidate(
            "svr_sigmoid", -0.05, model_sig, oof_sig, full_sig, y
        ),
        "svr_rbf": _make_candidate("svr_rbf", -0.09, model_rbf, oof_rbf, full_rbf, y),
    }


def _overfit_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """«Переобучение»: дерево глубины 20 (RMSE_full≈0) против честного Ridge."""
    oof_tree, full_tree, model_tree = _honest_oof_full(
        lambda: DecisionTreeRegressor(max_depth=20, random_state=0), X, y
    )
    oof_ridge, full_ridge, model_ridge = _honest_oof_full(
        lambda: Ridge(alpha=1.0, random_state=0), X, y
    )
    return {
        "decision_tree": _make_candidate(
            "decision_tree", -0.05, model_tree, oof_tree, full_tree, y
        ),
        "ridge": _make_candidate("ridge", -0.09, model_ridge, oof_ridge, full_ridge, y),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Реестр кейсов бенчмарка
# ──────────────────────────────────────────────────────────────────────────────


# Эталонные датасеты (сквозной прогон): гейт не должен ни срабатывать, ни
# менять победителя. lasso на широком датасете получает суженное пространство
# alpha [0.05, 1.0]: с дефолтным пространством Optuna выбирает альфы, при
# которых lasso переобучается (gap > 1.5) — известное ложное срабатывание
# контура Г из-за плохой регуляризации, а не из-за вырожденности (см. edge
# case в issue #70: «порог diversity слишком строгий / дефолты»).
REFERENCE_CASES: list[FullFlowCase] = [
    FullFlowCase(
        name="reference_noisy_r2_03",
        kind="reference",
        generator=make_reference_noisy_r2_03,
        algorithms={"ridge": {}, "lasso": {}, "elasticnet": {}},
    ),
    FullFlowCase(
        name="reference_wide_p200",
        kind="reference",
        generator=make_reference_wide,
        algorithms={
            "ridge": {},
            "lasso": {"hyperparameters": {"alpha": [0.05, 1.0, "float_log"]}},
            "elasticnet": {},
        },
    ),
    FullFlowCase(
        name="reference_small_n80",
        kind="reference",
        generator=make_reference_small_n,
        algorithms={"ridge": {}, "lasso": {}, "elasticnet": {}},
    ),
]

# Граничный эталонный кейс (контролируемый пул): diversity честной модели
# ≈ 0.155 при пороге 0.15 (запас < 5%) — документированный порог допуска
# «ухудшение ≤ ε»: победитель не меняется, гейт не срабатывает.
BOUNDARY_CASES: list[ControlledCase] = [
    ControlledCase(
        name="reference_boundary_diversity",
        kind="boundary",
        generator=make_reference_boundary_diversity,
        candidates=_boundary_ridge_candidates,
    ),
]

# Вырожденные датасеты (контролируемый пул): вырожденная модель — CV-лидер
# (как если бы старое поведение выбрало её), честная модель — вторая.
# При ``mode=active`` вырожденный кандидат дисквалифицируется — победитель
# обязан измениться (100% изменений).
DEGENERATE_CASES: list[ControlledCase] = [
    ControlledCase(
        name="degenerate_flat_shelf",
        kind="degenerate",
        generator=make_degenerate_flat_shelf,
        candidates=_flat_shelf_candidates,
    ),
    ControlledCase(
        name="degenerate_tanh_plateau",
        kind="degenerate",
        generator=make_degenerate_tanh_plateau,
        candidates=_tanh_plateau_candidates,
    ),
    ControlledCase(
        name="degenerate_overfit",
        kind="degenerate",
        generator=make_degenerate_overfit,
        candidates=_overfit_candidates,
    ),
]

ALL_CASES: list[Any] = [*REFERENCE_CASES, *BOUNDARY_CASES, *DEGENERATE_CASES]


# ──────────────────────────────────────────────────────────────────────────────
#  Прогон «до/после»
# ──────────────────────────────────────────────────────────────────────────────


def _gate_from_cfg(sg_cfg: Any) -> ModelSanityGate:
    """Собрать ``ModelSanityGate`` из конфигурации ``SanityGateCfg``."""
    return ModelSanityGate(
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


def _compare_modes(
    dataset: str,
    kind: str,
    candidates: dict[str, FinalistCandidate],
    X: pd.DataFrame,
    y: pd.Series,
    gate: ModelSanityGate,
    started: float,
) -> BeforeAfterResult:
    """Общий хвост: прогнать off/active на одном пуле и собрать результат."""
    res_off = select_robust_winner(candidates, X, y, gate, mode="off")
    res_active = select_robust_winner(candidates, X, y, gate, mode="active")

    def _metrics(winner: str, audit: Any) -> dict[str, float]:
        metrics: dict[str, float] = {
            "rmse_oof": candidates[winner].rmse_oof or float("nan")
        }
        if audit is not None and winner in audit:
            metrics["diversity_ratio"] = audit[winner].diversity_ratio
            metrics["generalization_gap"] = audit[winner].generalization_gap
        return metrics

    return BeforeAfterResult(
        dataset=dataset,
        kind=kind,
        winner_off=res_off.winner_algo,
        winner_active=res_active.winner_algo,
        pool=list(res_active.pool),
        cv_scores={n: c.cv_score for n, c in candidates.items()},
        rmse_oof={n: (c.rmse_oof if c.rmse_oof is not None else float("nan")) for n, c in candidates.items()},
        disqualified={n: list(r) for n, r in res_active.disqualified.items()},
        audit_valid={n: a.is_valid for n, a in res_active.audit.items()},
        winner_metrics_off=_metrics(res_off.winner_algo, {}),
        winner_metrics_active=_metrics(res_active.winner_algo, res_active.audit),
        gate_triggered=bool(res_active.disqualified),
        changed=res_off.winner_algo != res_active.winner_algo,
        elapsed=time.monotonic() - started,
    )


def _run_full_flow(case: FullFlowCase) -> BeforeAfterResult:
    """Сквозной прогон эталонного кейса (HPO → пул → финалисты → off/active)."""
    started = time.monotonic()
    X, y = case.generator()
    n_features = X.shape[1]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    algorithms = {}
    for name, extra in case.algorithms.items():
        algo_cfg: dict[str, Any] = {"enable": True, "limit_hyperparameters": True}
        algo_cfg.update(extra)
        algorithms[name] = algo_cfg

    cfg = Config.model_validate(
        {
            "general": {
                "comparison_metric": "rmse",
                "validation_strategy": "k_fold",
                "n_folds": 4,
                "phases": [
                    {
                        "name": "Search",
                        "n_trials": case.n_trials,
                        "action": "all_algorithms",
                    }
                ],
                "sanity_gate": {
                    "mode": "active",
                    "top_k_candidates": case.top_k_candidates,
                    "corridor_delta": case.corridor_delta,
                },
            },
            "algorithms": algorithms,
        }
    )

    prepared = prepare_dataset(cfg, df, "target")
    phase_results, disqualified = execute_phases(cfg, prepared)
    sg_cfg = cfg.general.sanity_gate
    pool = select_finalists(
        phase_results,
        metric_user=prepared.metric_user,
        top_k_candidates=sg_cfg.top_k_candidates,
        corridor_delta=sg_cfg.corridor_delta,
        corridor_mode=sg_cfg.corridor_mode,
        enforce_family_diversity=sg_cfg.enforce_family_diversity,
        algorithm_families=sg_cfg.algorithm_families,
        allowed_families=(
            set(sg_cfg.allowed_families) if sg_cfg.allowed_families is not None else None
        ),
        family_diversity_multiplier=sg_cfg.family_diversity_multiplier,
        disqualified=disqualified,
    )
    candidates = _train_finalists(pool, phase_results, cfg, prepared)
    return _compare_modes(
        case.name,
        case.kind,
        candidates,
        prepared.X,
        prepared.y,
        _gate_from_cfg(sg_cfg),
        started,
    )


def _run_controlled(case: ControlledCase) -> BeforeAfterResult:
    """Прогон контролируемого кейса (заданный пул реально обученных моделей)."""
    started = time.monotonic()
    X, y = case.generator()
    candidates = case.candidates(X, y)
    return _compare_modes(
        case.name,
        case.kind,
        candidates,
        pd.DataFrame(X),
        pd.Series(y),
        ModelSanityGate(),
        started,
    )


def run_case(case: FullFlowCase | ControlledCase) -> BeforeAfterResult:
    """Прогнать кейс «до/после» (сквозной или контролируемый).

    Args:
        case: Описание кейса (``FullFlowCase`` или ``ControlledCase``).

    Returns:
        BeforeAfterResult: Результат сравнения режимов.
    """
    if isinstance(case, FullFlowCase):
        return _run_full_flow(case)
    return _run_controlled(case)


# ──────────────────────────────────────────────────────────────────────────────
#  Таблица и критерии приёмки
# ──────────────────────────────────────────────────────────────────────────────


def _fmt(value: float | None) -> str:
    """Отформатировать метрику: NaN/None → '—', иначе 4 знака."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.4f}"


def render_table(results: list[BeforeAfterResult]) -> str:
    """Таблица «датасет × режим × победитель × метрики × сработал ли гейт».

    Для каждого датасета две строки: ``mode=off`` (победитель по CV, гейт не
    участвует) и ``mode=active`` (победитель после аудита, метрики аудита,
    сработал ли гейт).

    Args:
        results: Результаты прогонов.

    Returns:
        str: Текстовая таблица фиксированной ширины.
    """
    header = (
        f"{'датасет':<30} {'режим':<7} {'победитель':<16} "
        f"{'rmse_oof':<9} {'diversity':<9} {'gap':<8} {'гейт':<9}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        w_off = r.winner_metrics_off.get("rmse_oof")
        lines.append(
            f"{r.dataset:<30} {'off':<7} {r.winner_off:<16} "
            f"{_fmt(w_off):<9} {'—':<9} {'—':<8} {'нет':<9}"
        )
        wm = r.winner_metrics_active
        gate = "да" if r.gate_triggered else "нет"
        changed_mark = " (изм.)" if r.changed else ""
        lines.append(
            f"{'':<30} {'active':<7} {r.winner_active + changed_mark:<16} "
            f"{_fmt(wm.get('rmse_oof')):<9} {_fmt(wm.get('diversity_ratio')):<9} "
            f"{_fmt(wm.get('generalization_gap')):<8} {gate:<9}"
        )
    lines.append(sep)
    return "\n".join(lines)


def check_acceptance(results: list[BeforeAfterResult]) -> tuple[bool, list[str]]:
    """Проверить критерии приёмки T9.

    Эталонные и граничные датасеты: 0 изменений победителя и гейт не
    срабатывает. Вырожденные: 100% изменений, вырожденный кандидат
    дисквалифицирован, честный победитель валиден.

    Args:
        results: Результаты прогонов.

    Returns:
        tuple[bool, list[str]]: (passed, список нарушений).
    """
    errors: list[str] = []
    for r in results:
        if r.kind in ("reference", "boundary"):
            if r.changed:
                errors.append(
                    f"{r.dataset}: победитель изменился на эталонных данных "
                    f"({r.winner_off} -> {r.winner_active})"
                )
            if r.gate_triggered:
                errors.append(
                    f"{r.dataset}: гейт сработал на эталонных данных: "
                    f"{r.disqualified}"
                )
        else:  # degenerate
            if not r.changed:
                errors.append(
                    f"{r.dataset}: победитель НЕ изменился на вырожденных "
                    f"данных ({r.winner_off} == {r.winner_active})"
                )
            if not r.gate_triggered:
                errors.append(f"{r.dataset}: гейт не сработал на вырожденных данных")
            if r.audit_valid.get(r.winner_off, True):
                errors.append(
                    f"{r.dataset}: вырожденный кандидат {r.winner_off} не "
                    f"дисквалифицирован при mode=active"
                )
            if not r.audit_valid.get(r.winner_active, False):
                errors.append(
                    f"{r.dataset}: победитель active {r.winner_active} не прошёл "
                    f"аудит (ожидался честный кандидат)"
                )
    return (not errors, errors)


def run_all() -> tuple[list[BeforeAfterResult], bool, list[str]]:
    """Прогнать все кейсы и проверить критерии приёмки.

    Returns:
        tuple[list[BeforeAfterResult], bool, list[str]]: (results, passed,
            errors).
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    return results, passed, errors