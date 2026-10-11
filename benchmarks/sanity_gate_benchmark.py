"""Ядро регрессионного бенчмарка Sanity Gate на целевом пайплайне (эпик #74, R4).

Бенчмарк сравнивает два варианта пайплайна движка на одном и том же датасете.
Сравнение «до/после» времен эпика #61 переосмыслено: конфиг со старой схемой
(``refine_winner`` / ``sanity_gate.mode``) недоступен и отвергается валидацией,
поэтому базовая линия — современный аналог старого поведения выбора победителя:

* ``baseline``: ``all_algorithms → select(strategy: cv_best, output: winner)``
  — победитель по «сырому» CV-скору без аудита Sanity Gate;
* ``target``: ``all_algorithms → select(sanity_gate, top_k) → refine →
  select(sanity_gate, winner)`` — целевой DAG эпика #74: пул финалистов с
  аудитом (T3+T4), дообучение (warm-start), повторный аудит обновлённых
  моделей (честная цена — финалисты обучаются заново).

Критерии приёмки:

1. Эталонные датасеты (нормальные данные с честным победителем: зашумлённые
   R²≈0.3, широкие с сотнями признаков, малые N): победитель ``target``
   совпадает с победителем ``baseline``, гейт НЕ срабатывает (0
   дисквалификаций ни в одном select-шаге).
2. Вырожденные датасеты (полка / плато tanh / переобучение): CV-лидер —
   вырожденная модель, гейт дисквалифицирует её — победитель ``target``
   ОБЯЗАН отличаться от победителя ``baseline`` (100% изменений).

Архитектура прогона:

* Эталонные датасеты прогоняются сквозным пайплайном движка
  (``execute_phases``) с реальным HPO (``n_trials=1``, N ≤ 300) — проверяется,
  что аудит не ухудшает отбор на нормальных данных.
* Вырожденные датасеты используют детерминированную подмену ``_run_hpo``:
  вырожденный кандидат получает параметры, при которых реальное обучение на
  100% данных даёт вырожденную модель (``alpha=500`` для ElasticNet,
  ``max_depth=20`` для дерева, sigmoid-ядро для SVR), и «сырой» CV-скор лучше
  честного кандидата. Это единственный детерминированный способ гарантировать
  «вырожденный CV-лидер»: такие модели по построению НЕ выигрывают честную
  кросс-валидацию, поэтому сквозной HPO не может их воспроизвести стабильно
  (подход наследует негативные сценарии T4). Финалисты при этом обучаются и
  аудируются по-настоящему.

Время прогона ограничено (CI): 6 датасетов, n_trials=1, N ≤ 300 — суммарно
порядка 10–20 секунд.

Формат вывода — таблица «датасет × вариант × победитель × сработал ли гейт»
(см. ``render_table``); критерии приёмки — ``check_acceptance``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import pandas as pd

import configurable_automl_engine.training_engine.component as _component
from configurable_automl_engine.training_engine.component import (
    PhaseContext,
    execute_phases,
    prepare_dataset,
)
from configurable_automl_engine.training_engine.config_parser import Config

from tests.data_factory import (
    make_degenerate_flat_shelf,
    make_degenerate_overfit,
    make_degenerate_tanh_plateau,
    make_reference_noisy_r2_03,
    make_reference_small_n,
    make_reference_wide,
)

__all__ = [
    "VariantResult",
    "ReferenceCase",
    "DegenerateCase",
    "REFERENCE_CASES",
    "DEGENERATE_CASES",
    "ALL_CASES",
    "run_case",
    "render_table",
    "check_acceptance",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Результат прогона вариантов пайплайна
# ──────────────────────────────────────────────────────────────────────────────

#: Имя фазы первого sanity-select целевого DAG (пул финалистов T3, аудит T2).
GATE_SELECT_PHASE = "GateSelect"


@dataclass(frozen=True)
class VariantResult:
    """Результат сравнения ``baseline`` vs ``target`` на одном датасете.

    Attributes:
        dataset (str): Имя датасета.
        kind (str): 'reference' | 'degenerate'.
        winner_baseline (str): Победитель пайплайна ``cv_best`` (CV-лидер).
        winner_target (str): Победитель целевого DAG с аудитом и refine.
        pool (list[str]): Пул финалистов первого sanity-select целевого DAG.
        disqualified (dict[str, list[str]]): Дисквалификации Sanity Gate
            (имя select-фазы → дисквалифицированные кандидаты).
        audit_valid (dict[str, bool]): is_valid каждого кандидата пула
            (аудит первого sanity-select).
        gate_triggered (bool): True — хотя бы в одном sanity-select целевого
            DAG есть дисквалификации.
        changed (bool): True — победители вариантов различаются.
        elapsed (float): Время прогона в секундах.
    """

    dataset: str
    kind: str
    winner_baseline: str
    winner_target: str
    pool: list[str]
    disqualified: dict[str, list[str]] = field(default_factory=dict)
    audit_valid: dict[str, bool] = field(default_factory=dict)
    gate_triggered: bool = False
    changed: bool = False
    elapsed: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Описания кейсов
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReferenceCase:
    """Эталонный кейс: сквозной прогон с реальным HPO.

    Attributes:
        name (str): Имя датасета.
        generator (Callable): Генератор (X, y) с фиксированным seed.
        algorithms (dict[str, dict]): Конфигурация алгоритмов для HPO
            (имя → доп. поля AlgoCfg, например гиперпараметры).
        n_trials (int): Число триалов HPO на алгоритм (ограничение времени CI).
        top_k_candidates (int): Кап размера пула финалистов.
        corridor_delta (float): Ширина коридора пула.
    """

    name: str
    kind: str = "reference"
    generator: Callable[[], tuple[np.ndarray, np.ndarray]] = field(
        default=lambda: (np.zeros((1, 1)), np.zeros(1))
    )
    algorithms: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_trials: int = 1
    top_k_candidates: int = 3
    corridor_delta: float = 0.3


@dataclass(frozen=True)
class DegenerateCase:
    """Вырожденный кейс: сквозной прогон с детерминированной подменой HPO.

    ``stub_hpo`` возвращает ``(score, params)`` для каждого имени алгоритма:
    вырожденный кандидат получает параметры, при которых реальное обучение
    финалиста даёт вырожденную модель, и «сырой» CV-скор лучше честного
    кандидата (он и был бы «победителем» старого поведения по CV-скору).

    Attributes:
        name (str): Имя датасета.
        generator (Callable): Генератор (X, y) с фиксированным seed.
        algorithms (dict[str, dict]): Конфигурация алгоритмов.
        stub_hpo (Callable[[str], tuple[float, dict[str, Any]]]): Детермини-
            рованный «тюнер»: имя алгоритма → (сырой CV-скор, параметры).
        top_k_candidates (int): Кап размера пула финалистов.
        corridor_delta (float): Ширина коридора пула.
    """

    name: str
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]
    algorithms: dict[str, dict[str, Any]]
    stub_hpo: Callable[[str], tuple[float, dict[str, Any]]]
    kind: str = "degenerate"
    n_trials: int = 1
    top_k_candidates: int = 2
    corridor_delta: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
#  Реестр кейсов бенчмарка
# ──────────────────────────────────────────────────────────────────────────────

# Эталонные датасеты (сквозной прогон): гейт не должен ни срабатывать, ни
# менять победителя. lasso на широком датасете получает суженное пространство
# alpha [0.05, 1.0]: с дефолтным пространством Optuna выбирает альфы, при
# которых lasso переобучается (gap > 1.5) — известное ложное срабатывание
# контура Г из-за плохой регуляризации, а не из-за вырожденности (см. edge
# case в issue #70: «порог diversity слишком строгий / дефолты»).
_REFERENCE_ALGORITHMS: dict[str, dict[str, Any]] = {
    "ridge": {"enable": True, "limit_hyperparameters": True},
    "lasso": {"enable": True, "limit_hyperparameters": True},
    "elasticnet": {"enable": True, "limit_hyperparameters": True},
}

REFERENCE_CASES: list[ReferenceCase] = [
    ReferenceCase(
        name="reference_noisy_r2_03",
        generator=make_reference_noisy_r2_03,
        algorithms=dict(_REFERENCE_ALGORITHMS),
    ),
    ReferenceCase(
        name="reference_wide_p200",
        generator=make_reference_wide,
        algorithms={
            "ridge": {"enable": True, "limit_hyperparameters": True},
            "lasso": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.05, 1.0, "float_log"]},
            },
            "elasticnet": {"enable": True, "limit_hyperparameters": True},
        },
    ),
    ReferenceCase(
        name="reference_small_n80",
        generator=make_reference_small_n,
        algorithms=dict(_REFERENCE_ALGORITHMS),
    ),
]

# Вырожденные датасеты (детерминированная подмена HPO): вырожденный кандидат —
# CV-лидер (как если бы старое поведение выбрало его), честный кандидат —
# второй. При аудите Sanity Gate вырожденный кандидат дисквалифицируется —
# победитель целевого DAG обязан измениться (100% изменений).
#
# «Сырые» скоры (метрика rmse, семантика оптимизатора — больше лучше) заданы
# так, чтобы оба кандидата попали в пул финалистов (коридор δ=0.3 в
# пользовательской семантике: RMSE 0.08 против 0.09) и при этом вырожденный
# кандидат оставался CV-лидером.
DEGENERATE_CASES: list[DegenerateCase] = [
    DegenerateCase(
        name="degenerate_flat_shelf",
        generator=make_degenerate_flat_shelf,
        algorithms={
            "elasticnet": {"enable": True, "limit_hyperparameters": True},
            "ridge": {"enable": True, "limit_hyperparameters": True},
        },
        stub_hpo=lambda name: {
            "elasticnet": (-0.08, {"alpha": 500.0}),
            "ridge": (-0.09, {"alpha": 1.0}),
        }[name],
    ),
    DegenerateCase(
        name="degenerate_tanh_plateau",
        generator=make_degenerate_tanh_plateau,
        algorithms={
            "svr": {"enable": True, "limit_hyperparameters": True},
            "ridge": {"enable": True, "limit_hyperparameters": True},
        },
        stub_hpo=lambda name: {
            "svr": (
                -0.08,
                {"kernel": "sigmoid", "gamma": 0.001, "coef0": 0.0, "C": 1.0},
            ),
            "ridge": (-0.09, {"alpha": 1.0}),
        }[name],
    ),
    DegenerateCase(
        name="degenerate_overfit",
        generator=make_degenerate_overfit,
        algorithms={
            "decision_tree": {"enable": True, "limit_hyperparameters": True},
            "ridge": {"enable": True, "limit_hyperparameters": True},
        },
        stub_hpo=lambda name: {
            "decision_tree": (-0.08, {"max_depth": 20}),
            "ridge": (-0.09, {"alpha": 1.0}),
        }[name],
    ),
]

ALL_CASES: list[ReferenceCase | DegenerateCase] = [
    *REFERENCE_CASES,
    *DEGENERATE_CASES,
]


# ──────────────────────────────────────────────────────────────────────────────
#  Конфигурации вариантов пайплайна
# ──────────────────────────────────────────────────────────────────────────────


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    """Нормализовать конфигурацию алгоритма из кейса в AlgoCfg-словарь."""
    cfg: dict[str, Any] = {"enable": True, "limit_hyperparameters": True}
    cfg.update(extra)
    return cfg


def _phase(name: str, n_trials: int, action: str, **extra: Any) -> dict[str, Any]:
    """Собрать фазу пайплайна."""
    phase: dict[str, Any] = {"name": name, "n_trials": n_trials, "action": action}
    phase.update(extra)
    return phase


def _cfg_dict(
    case: ReferenceCase | DegenerateCase,
    *,
    target_dag: bool,
) -> dict[str, Any]:
    """Собрать конфигурацию варианта пайплайна (baseline или target DAG).

    Args:
        case: Кейс бенчмарка.
        target_dag (bool): True — целевой DAG с аудитом и refine; False —
            базовая линия ``cv_best`` без аудита.

    Returns:
        dict[str, Any]: Конфигурация для ``Config.model_validate``.
    """
    n_trials = case.n_trials
    if target_dag:
        phases = [
            _phase("Search", n_trials, "all_algorithms"),
            _phase(
                GATE_SELECT_PHASE,
                1,
                "select",
                strategy="sanity_gate",
                output="top_k",
                k=case.top_k_candidates,
            ),
            _phase("Refine", n_trials, "refine"),
            _phase("FinalSelect", 1, "select", strategy="sanity_gate", output="winner"),
        ]
    else:
        phases = [
            _phase("Search", n_trials, "all_algorithms"),
            _phase(
                "FinalSelect", 1, "select", strategy="cv_best", output="winner"
            ),
        ]
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": phases,
        "sanity_gate": {
            "top_k_candidates": case.top_k_candidates,
            "corridor_delta": case.corridor_delta,
        },
    }
    return {
        "general": general,
        "algorithms": {
            name: _algo_cfg(extra) for name, extra in case.algorithms.items()
        },
        "oversampling": {"enable": False},
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Прогон вариантов
# ──────────────────────────────────────────────────────────────────────────────


def _df_from_case(case: ReferenceCase | DegenerateCase) -> pd.DataFrame:
    """Собрать DataFrame из генератора кейса."""
    X, y = case.generator()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _execute(
    cfg_dict: dict[str, Any],
    df: pd.DataFrame,
    *,
    stub_hpo: Callable[[str], tuple[float, dict[str, Any]]] | None = None,
) -> PhaseContext:
    """Прогнать ``execute_phases`` на конфигурации, опционально подменяя HPO.

    Args:
        cfg_dict: Конфигурация варианта пайплайна.
        df: Датасет с целевой колонкой.
        stub_hpo: Детерминированная подмена ``_run_hpo`` (вырожденные кейсы);
            ``None`` — реальный HPO.

    Returns:
        PhaseContext: Итоговый контекст фаз (победитель в ``winner_algo``).
    """
    cfg = Config.model_validate(cfg_dict)
    prepared = prepare_dataset(cfg, df, "target")
    if stub_hpo is None:
        return execute_phases(cfg, prepared)
    with patch.object(
        _component,
        "_run_hpo",
        side_effect=lambda **kwargs: stub_hpo(kwargs["algo_name"]),
    ):
        return execute_phases(cfg, prepared)


def _audit_valid_from_section(section: dict[str, Any]) -> dict[str, bool]:
    """is_valid каждого кандидата из gate-секции (T6)."""
    return {name: bool(audit["is_valid"]) for name, audit in section["audit"].items()}


def run_case(case: ReferenceCase | DegenerateCase) -> VariantResult:
    """Прогнать кейс «baseline vs target DAG».

    Args:
        case: Описание кейса (``ReferenceCase`` или ``DegenerateCase``).

    Returns:
        VariantResult: Результат сравнения вариантов пайплайна.
    """
    started = time.monotonic()
    df = _df_from_case(case)
    stub: Callable[[str], tuple[float, dict[str, Any]]] | None = (
        case.stub_hpo if isinstance(case, DegenerateCase) else None
    )

    ctx_baseline = _execute(_cfg_dict(case, target_dag=False), df, stub_hpo=stub)
    ctx_target = _execute(_cfg_dict(case, target_dag=True), df, stub_hpo=stub)

    sections = ctx_target.sanity_gate_sections
    gate_select = sections.get(GATE_SELECT_PHASE, {})
    disqualified = {
        name: list(section["disqualified"]) for name, section in sections.items()
    }
    gate_triggered = any(section["disqualified"] for section in sections.values())
    winner_baseline = ctx_baseline.winner_algo
    winner_target = ctx_target.winner_algo
    return VariantResult(
        dataset=case.name,
        kind=case.kind,
        winner_baseline=winner_baseline or "",
        winner_target=winner_target or "",
        pool=list(gate_select.get("pool", [])),
        disqualified=disqualified,
        audit_valid=(
            _audit_valid_from_section(gate_select) if gate_select else {}
        ),
        gate_triggered=gate_triggered,
        changed=winner_baseline != winner_target,
        elapsed=time.monotonic() - started,
    )


def run_all() -> tuple[list[VariantResult], bool, list[str]]:
    """Прогнать все кейсы и проверить критерии приёмки.

    Returns:
        tuple[list[VariantResult], bool, list[str]]: (results, passed, errors).
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    return results, passed, errors


# ──────────────────────────────────────────────────────────────────────────────
#  Таблица и критерии приёмки
# ──────────────────────────────────────────────────────────────────────────────


def render_table(results: list[VariantResult]) -> str:
    """Таблица «датасет × вариант × победитель × сработал ли гейт».

    Для каждого датасета две строки: ``baseline`` (CV-лидер без аудита) и
    ``target`` (победитель целевого DAG с аудитом и refine, дисквалификации
    гейта по обоим select-шагам).

    Args:
        results: Результаты прогонов.

    Returns:
        str: Текстовая таблица фиксированной ширины.
    """
    header = (
        f"{'датасет':<28} {'вариант':<9} {'победитель':<18} "
        f"{'гейт':<9} {'дисквалификации':<16}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        gate_b = "нет"
        changed_mark = " (изм.)" if r.changed else ""
        gate_t = "да" if r.gate_triggered else "нет"
        disq = "; ".join(
            f"{phase}: {','.join(names)}"
            for phase, names in r.disqualified.items()
            if names
        )
        lines.append(
            f"{r.dataset:<28} {'baseline':<9} {r.winner_baseline:<18} "
            f"{gate_b:<9} {'—':<16}"
        )
        lines.append(
            f"{'':<28} {'target':<9} {r.winner_target + changed_mark:<18} "
            f"{gate_t:<9} {disq:<16}"
        )
    lines.append(sep)
    return "\n".join(lines)


def check_acceptance(results: list[VariantResult]) -> tuple[bool, list[str]]:
    """Проверить критерии приёмки эпика #74 (R4).

    Эталонные датасеты: 0 изменений победителя и гейт не срабатывает.
    Вырожденные: 100% изменений, вырожденный CV-лидер дисквалифицирован,
    честный победитель валиден в первом sanity-select.

    Args:
        results: Результаты прогонов.

    Returns:
        tuple[bool, list[str]]: (passed, список нарушений).
    """
    errors: list[str] = []
    for r in results:
        if r.kind == "reference":
            if r.changed:
                errors.append(
                    f"{r.dataset}: победитель изменился на эталонных данных "
                    f"({r.winner_baseline} -> {r.winner_target})"
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
                    f"данных ({r.winner_baseline} == {r.winner_target})"
                )
            if not r.gate_triggered:
                errors.append(
                    f"{r.dataset}: гейт не сработал на вырожденных данных"
                )
            if r.winner_baseline not in r.disqualified.get(GATE_SELECT_PHASE, []):
                errors.append(
                    f"{r.dataset}: вырожденный CV-лидер {r.winner_baseline} "
                    f"не дисквалифицирован в первом sanity-select"
                )
            if not r.audit_valid.get(r.winner_target, False):
                errors.append(
                    f"{r.dataset}: победитель target {r.winner_target} не "
                    f"прошёл аудит в первом sanity-select (ожидался честный "
                    f"кандидат)"
                )
    return (not errors, errors)