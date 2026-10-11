"""Тесты явного select-шага пайплайна (эпик #74, задача R2).

Покрытие:

1. Позитивные: select ``cv_best`` с каждым ``output`` (winner/top_k/pool);
   select ``sanity_gate`` с ``winner``/``top_k``/``pool``; select после
   select; persist_artifact терминального select без переобучения (тренер
   финалиста переиспользуется через ``pre_trained_trainer``).
2. Негативные: полная дисквалификация select'а → fallback T4; пустое входное
   множество select'а → RuntimeError.
3. Отчёт (T6): секции ``sanity_gate`` по именам фаз; отсутствие gate-секции
   у ``cv_best``; удалённые ключи ``warn_only``/``winner_if_active`` больше
   не появляются; дисквалификации гейта попадают в circuit breaker.
4. Unit-уровень обработчиков: ранжир стратегии (``_rank_sanity_candidates``),
   сериализация аудита, сборка гейта из секции порогов, повторный sanity-select
   переобучает финалистов (старые full-fit тренеры не переиспользуются).

Схема конфига (эпик #74) обязывает последней фазой быть терминальный select
(``output: winner``), поэтому содержимое промежуточных select-шагов
проверяется прямыми вызовами обработчиков на handcrafted ``PhaseContext``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine.training_engine.component as component
from configurable_automl_engine.training_engine.component import (
    FinalistCandidate,
    PhaseContext,
    PreparedData,
    RobustWinnerResult,
    _build_sanity_gate,
    _rank_sanity_candidates,
    _serialize_audit_result,
    execute_phases,
    persist_artifact,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    HPOPhaseCfg,
    SanityGateCfg,
    ValidationStrategy,
)
from configurable_automl_engine.training_engine.sanity_gate import SanityCheckResult

# ──────────────────────────────────────────────────────────────────────────────
# Общие помощники
# ──────────────────────────────────────────────────────────────────────────────


def _select_phase(**overrides: Any) -> HPOPhaseCfg:
    """Собрать HPOPhaseCfg select-фазы поверх дефолтов."""
    base: dict[str, Any] = {
        "name": "sel",
        "n_trials": 1,
        "action": "select",
        "strategy": "cv_best",
        "output": "winner",
    }
    base.update(overrides)
    return HPOPhaseCfg(**base)


def _cfg_dict(
    phases: list[dict[str, Any]],
    *,
    extra_algorithms: dict[str, dict[str, Any]] | None = None,
    **general_extra: Any,
) -> dict[str, Any]:
    """Валидный конфиг: HPO-фаза + заданные фазы (терминальный — с конца).

    Args:
        phases: Фазы пайплайна (добавляются после обязательной HPO-фазы).
        extra_algorithms: Дополнительные секции алгоритмов (сливаются с
            базовыми ridge/extra_trees).
        **general_extra: Дополнительные поля ``general`` (например,
            ``sanity_gate``).
    """
    algorithms: dict[str, dict[str, Any]] = {
        "ridge": {
            "enable": True,
            "tuner": "mock.mock_tuner",
            "trainer_module": "mock.mock_trainer",
        },
        "extra_trees": {
            "enable": True,
            "tuner": "mock.mock_tuner",
            "trainer_module": "mock.mock_trainer",
        },
    }
    if extra_algorithms:
        algorithms.update(extra_algorithms)
    return {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                *phases,
            ],
            "path_to_model": "model.pkl",
            **general_extra,
        },
        "algorithms": algorithms,
        "oversampling": {"data_oversampling": False},
    }


def _df(n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """Числовой DataFrame с целевой колонкой."""
    rng = np.random.RandomState(seed)
    data = {f"f{i}": rng.randn(n_rows) for i in range(3)}
    data["target"] = 2.0 * data["f0"] + rng.randn(n_rows) * 0.1
    return pd.DataFrame(data)


def _prepared(y: pd.Series | None = None) -> PreparedData:
    """Минимальный PreparedData с заполненными дефолтами."""
    if y is None:
        y = pd.Series([1.0, 2.0, 3.0])
    return PreparedData(
        X=pd.DataFrame({"a": np.arange(len(y), dtype=float)}),
        y=y,
        metric_user="mae",
        metric_sklearn="mae",
        resolved_validation=ValidationStrategy.train_test_split,
        resolved_n_folds=5,
        resolved_test_size=0.2,
        categorical_features=[],
        numerical_features=["a"],
    )


class _FakeTrainer:
    """Фейк ModelTrainer с контролируемым качеством OOF/full предсказаний.

    ``oof_rmse`` — RMSE OOF-вектора относительно y; ``full_rmse`` — RMSE
    full-fit предсказаний (контур Г: gap = oof_rmse / full_rmse).
    """

    def __init__(self, y: np.ndarray, oof_rmse: float, full_rmse: float) -> None:
        rng = np.random.RandomState(0)

        def _noise(scale: float) -> np.ndarray:
            noise = rng.randn(len(y))
            return noise / np.sqrt(np.mean(noise**2)) * scale

        # pipeline с методом predict: контур В аудита предсказывает на
        # переставленном X (pipeline = self достаточно).
        self.pipeline = self
        self.oof_predictions_ = y + _noise(oof_rmse)
        self.oof_score_ = oof_rmse
        self._full_pred = y + _noise(full_rmse)

    def predict(self, X: Any) -> np.ndarray:
        return self._full_pred[: len(X)]

    def save(self, path: Any) -> None:
        """Артефакт «сохраняется» созданием файла (persist_artifact)."""
        from pathlib import Path

        Path(path).write_bytes(b"fake-model")


def _make_ctx(scores: dict[str, tuple[float, dict[str, Any]]]) -> PhaseContext:
    """Контекст фаз с заданным рабочим множеством."""
    return PhaseContext(working_set=dict(scores))


def _hpo_cfg(**general_extra: Any) -> Config:
    """Конфиг для прямых вызовов обработчиков (фазы не исполняются)."""
    return Config.model_validate(
        _cfg_dict(
            [
                {"name": "final", "n_trials": 1, "action": "all_algorithms"},
                _select_phase(),
            ],
            **general_extra,
        )
    )


_GATE_THRESHOLDS = {"sanity_gate": {"top_k_candidates": 2, "corridor_delta": 0.9}}


def _patch_hpo():
    """Замокать HPO: детерминированные скоры «ridge» хуже, чем у «extra_trees»."""
    return patch.object(
        component,
        "_run_hpo",
        side_effect=lambda **kw: (
            (-0.10, {"p": 1}) if kw["algo_name"] == "ridge" else (-0.08, {"p": 1})
        ),
    )


def _patch_honest_finalists(y: pd.Series):
    """Замокать обучение финалистов честным тренером (gap ≈ 1.2 — валиден)."""
    return patch.object(
        component,
        "_fit_finalist",
        side_effect=lambda *a, **kw: _FakeTrainer(y.to_numpy(), 0.30, 0.25),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  select cv_best: каждый output (unit-уровень обработчика)
# ──────────────────────────────────────────────────────────────────────────────
def test_select_cv_best_winner():
    """select(cv_best, winner): рабочее множество = CV-лидер, gate-секции нет."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo == "extra_trees"
    assert ctx.pre_trained_trainer is None
    assert ctx.sanity_gate_sections == {}


def test_select_cv_best_top_k():
    """select(cv_best, top_k): k лучших по CV-скору."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(output="top_k", k=1), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo is None  # нетерминальный select не фиксирует победителя


def test_select_cv_best_top_k_tie_keeps_insertion_order():
    """select(cv_best, top_k): ничья CV-скоров — порядок вставки (tie-break)."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.10, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(output="top_k", k=1), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"ridge"}


def test_select_cv_best_pool_corridor_and_cap():
    """select(cv_best, pool): коридорный пул T3 (δ ∩ top_k_candidates)."""
    cfg = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.05})
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(output="pool"), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    # Узкий коридор δ=0.05: 0.10 > 0.08 * 1.05 → только лидер.
    assert set(ctx.working_set) == {"extra_trees"}

    # Широкий коридор + кап top_k_candidates=2: оба кандидата.
    cfg2 = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9})
    ctx2 = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(name="sel2", output="pool"),
        cfg=cfg2,
        prepared=_prepared(),
        ctx=ctx2,
    )
    assert list(ctx2.working_set) == ["extra_trees", "ridge"]
    assert ctx2.sanity_gate_sections == {}  # gate-секция не создаётся

    # Boundary капа: пул широк (оба в коридоре δ=0.9), но top_k_candidates=1
    # отсекает его до лидера.
    cfg3 = _hpo_cfg(sanity_gate={"top_k_candidates": 1, "corridor_delta": 0.9})
    ctx3 = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    component._execute_select_phase(
        phase=_select_phase(name="sel3", output="pool"),
        cfg=cfg3,
        prepared=_prepared(),
        ctx=ctx3,
    )
    assert list(ctx3.working_set) == ["extra_trees"]


def test_select_cv_best_pool_excludes_disqualified():
    """select(cv_best, pool): дисквалифицированные circuit breaker'ом не в пуле."""
    cfg = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9})
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    ctx.disqualified_algorithms = {"ridge": "HPO failure"}
    component._execute_select_phase(
        phase=_select_phase(output="pool"), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}


# ──────────────────────────────────────────────────────────────────────────────
#  select sanity_gate: каждый output (unit-уровень обработчика)
# ──────────────────────────────────────────────────────────────────────────────
def _result(
    pool: list[str],
    audits: dict[str, SanityCheckResult],
    winner: str,
    fallback: bool = False,
) -> RobustWinnerResult:
    """RobustWinnerResult для мока ``select_robust_winner``."""
    return RobustWinnerResult(
        winner_algo=winner,
        mode="active",
        pool=pool,
        audit=audits,
        disqualified={
            name: list(res.reasons) for name, res in audits.items() if res.reasons
        },
        fallback_used=fallback,
        winner_rmse_oof=0.2,
    )


def test_select_sanity_gate_winner_sets_pre_trained_trainer():
    """select(sanity_gate, winner): full-fit тренер победителя в контексте."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    trainer = object()

    def fake_fit(algo_name, *args, **kwargs):
        return _FakeTrainer(_prepared().y.to_numpy(), 0.2, 0.25)

    with (
        patch.object(component, "_fit_finalist", side_effect=fake_fit),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ) as mock_select,
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="winner"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    mock_select.assert_called_once()
    assert mock_select.call_args.kwargs["mode"] == "active"
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo == "extra_trees"
    assert ctx.pre_trained_trainer is ctx.full_fit_trainers["extra_trees"]


def test_select_sanity_gate_top_k_ranks_by_rmse_oof():
    """select(sanity_gate, top_k): k лучших по RMSE_oof (не по CV-скору)."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    # CV-лидер — «ridge», но его RMSE_oof хуже: top_k по RMSE_oof берёт
    # «extra_trees».
    ctx = _make_ctx({"ridge": (-0.08, {"p": 1}), "extra_trees": (-0.09, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), {"ridge": 0.5, "extra_trees": 0.2}[a[0]], 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["ridge", "extra_trees"],
                {
                    "ridge": SanityCheckResult(is_valid=True),
                    "extra_trees": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees"}
    section = ctx.sanity_gate_sections["gate"]
    assert section["pool"] == ["ridge", "extra_trees"]
    assert section["fallback_used"] is False
    assert section["overhead_seconds"] >= 0
    assert section["winner_rmse_oof"] == 0.2


def test_select_sanity_gate_pool_keeps_passed_candidates():
    """select(sanity_gate, pool): все прошедшие аудит остаются в рабочем множестве."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert list(ctx.working_set) == ["extra_trees", "ridge"]


def test_select_sanity_gate_pool_full_disqualification_falls_back_to_ranking():
    """select(sanity_gate, pool): полная дисквалификация → fallback-ранжир.

    Пайплайн не остаётся с пустым множеством (документированный fallback).
    Инвариант выхода (issue #102): вернувшиеся в рабочее множество
    fallback-кандидаты снимаются с circuit breaker — иначе следующая фаза
    отфильтровала бы их и упала. Аудит сохраняется в gate-секции.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                    "ridge": SanityCheckResult(
                        is_valid=False,
                        reasons=["Circuit A", "Circuit B"],
                        severity=3,
                    ),
                },
                winner="extra_trees",
                fallback=True,
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Меньше нарушений (extra_trees) — выше в fallback-ранжире.
    assert list(ctx.working_set) == ["extra_trees", "ridge"]
    # Инвариант (issue #102): рабочее множество не пересекается с circuit
    # breaker — fallback-кандидаты не дисквалифицированы навсегда, иначе
    # следующий refine/select отфильтровал бы их и упал с RuntimeError.
    assert ctx.disqualified_algorithms == {}
    section = ctx.sanity_gate_sections["gate"]
    assert section["fallback_used"] is True
    # Детальный аудит сохраняется в gate-секции отчёта.
    assert set(section["disqualified"]) == {"extra_trees", "ridge"}


def test_select_sanity_gate_top_k_fallback_kept_candidate_not_disqualified():
    """select(sanity_gate, top_k): fallback-кандидат в выходе не дисквалифицирован.

    Ранжир стратегии ставит валидных первыми; если ``k`` покрывает и
    забракованных (fallback-семантика top_k, issue #102), вернувшиеся в
    рабочее множество кандидаты снимаются с circuit breaker — иначе
    следующий refine/select отфильтровал бы их и упал с RuntimeError.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=2
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees", "ridge"}
    # Инвариант (issue #102): ridge вернулся в выход top_k → не в circuit breaker.
    assert ctx.disqualified_algorithms == {}


def test_select_sanity_gate_top_k_drops_failed_outside_k():
    """select(sanity_gate, top_k): забракованный вне slice остаётся дисквалифицированным."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees"}
    # ridge вне top_k: исключён из рабочего множества и остаётся в circuit breaker.
    assert set(ctx.disqualified_algorithms) == {"ridge"}
    assert ctx.disqualified_algorithms["ridge"] == "Circuit A"


def test_select_sanity_gate_pool_partial_failure_keeps_only_passed():
    """select(sanity_gate, pool): частичный провал аудита.

    Прошедшие остаются в рабочем множестве; реально забракованные — вне
    выхода и в circuit breaker (не вернутся в следующие фазы, эпик #74).
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {"ridge": "Circuit A"}
    assert ctx.sanity_gate_sections["gate"]["fallback_used"] is False


def test_select_sanity_gate_budget_skipped_not_permanently_disqualified():
    """select(sanity_gate, pool): кандидаты, не проверенные из-за
    ``audit_time_budget``, не дисквалифицируются навсегда (issue #102).

    Пропуск аудита — ресурсное ограничение фазы, а не качество модели:
    такой кандидат не может победить в текущем select (нет аудита), но не
    попадает в circuit breaker и может вернуться в последующие фазы.
    Причины остаются в gate-секции отчёта.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    # Формат причины берётся из кода (единый источник, ревью PR #49).
    budget_reason = component._audit_budget_skip_reason(5.0)

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=[budget_reason], severity=0
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Прошедший остаётся; пропущенный не дисквалифицирован навсегда.
    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {}
    # Отчёт сохраняет причину пропуска.
    section = ctx.sanity_gate_sections["gate"]
    assert section["audit"]["ridge"]["is_valid"] is False
    assert section["disqualified"]["ridge"] == [budget_reason]


def test_select_sanity_gate_budget_skipped_outside_output_not_disqualified():
    """Budget-only причины кандидата ВНЕ рабочего множества (issue #102).

    Пропущенный по бюджету кандидат не прошёл в выход ``pool`` (прошли
    только валидные), но НЕ дисквалифицируется навсегда — в отличие от
    реального провала, который остаётся в circuit breaker. Контраст обеих
    категорий в одном select-шаге.
    """
    cfg = Config.model_validate(
        _cfg_dict(
            [
                _select_phase(name="gate", strategy="sanity_gate", output="pool"),
                _select_phase(name="final"),  # терминальный select (валидность схемы)
            ],
            sanity_gate={"top_k_candidates": 3, "corridor_delta": 0.9},
            extra_algorithms={
                "svr": {
                    "enable": True,
                    "tuner": "mock.mock_tuner",
                    "trainer_module": "mock.mock_trainer",
                }
            },
        )
    )
    ctx = _make_ctx(
        {
            "extra_trees": (-0.08, {"p": 1}),
            "ridge": (-0.10, {"p": 1}),
            "svr": (-0.12, {"p": 1}),
        }
    )
    budget_reason = component._audit_budget_skip_reason(5.0)

    with (
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge", "svr"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                    "svr": SanityCheckResult(
                        is_valid=False, reasons=[budget_reason], severity=0
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Валидный — в выходе; реальный провал — в circuit breaker; пропущенный
    # по бюджету — ни в выходе, ни в circuit breaker (issue #102).
    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {"ridge": "Circuit A"}
    assert "svr" not in ctx.disqualified_algorithms
    # Отчёт сохраняет причины обеих невалидных категорий.
    section = ctx.sanity_gate_sections["gate"]
    assert section["disqualified"]["ridge"] == ["Circuit A"]
    assert section["disqualified"]["svr"] == [budget_reason]


def test_select_sanity_gate_pool_keeps_pre_existing_disqualification():
    """select(sanity_gate, pool): предсуществующая дисквалификация не снимается.

    (ревью PR #49) Снимаются только провалы, дисквалифицированные этим
    шагом и возвращённые fallback'ом в выход. Кандидат, дисквалифицированный
    ранее (например, HPO circuit breaker), остаётся в circuit breaker, даже
    если оказался в выходе шага.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    ctx.disqualified_algorithms = {"ridge": "HPO failure"}

    with (
        patch.object(
            component,
            "_build_finalists_pool",
            return_value=["extra_trees", "ridge"],
        ),
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            component,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        component._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Оба валидны → оба в выходе pool; ridge НЕ провалил аудит в этом шаге,
    # поэтому его предсуществующая дисквалификация сохраняется.
    assert list(ctx.working_set) == ["extra_trees", "ridge"]
    assert ctx.disqualified_algorithms == {"ridge": "HPO failure"}


def test_is_audit_budget_skip_rejects_partial_prefix_match():
    """Fullmatch-детекция (ревью PR #49): частичное совпадение префикса не
    считается бюджетным пропуском.

    Раньше ``startswith`` давал ложное срабатывание на произвольной причине,
    лишь начинающейся с префикса; теперь причина обязана полностью
    соответствовать формату «{prefix} (Ns) exceeded».
    """
    assert (
        component._is_audit_budget_skip(
            ["Sanity audit skipped: audit time budget is not a reason"]
        )
        is False
    )
    assert component._is_audit_budget_skip(["Sanity audit skipped by user"]) is False
    # Кандидат с реальной причиной + «похожей» — тоже не «только пропуск».
    assert (
        component._is_audit_budget_skip(
            [component._audit_budget_skip_reason(5.0), "Sanity audit skipped by user"]
        )
        is False
    )


def test_is_audit_budget_skip_detects_only_budget_reasons():
    """``_is_audit_budget_skip`` отличает пропуск по бюджету от реального провала."""
    reason = component._audit_budget_skip_reason(5.0)
    assert component._is_audit_budget_skip([reason]) is True
    # Смешанные причины (бюджет + контур) — не «только пропуск».
    assert component._is_audit_budget_skip([reason, "Circuit A"]) is False
    assert component._is_audit_budget_skip(["Circuit A"]) is False
    assert component._is_audit_budget_skip([]) is False


def test_select_after_select():
    """select после select: выход первого замещает вход второго."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1}), "extra_trees": (-0.08, {"p": 1})})
    prepared = _prepared()

    with _patch_honest_finalists(prepared.y):
        # Первый select: sanity_gate top_k=1 (лучший по RMSE_oof — по ранжиру).
        component._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=prepared,
            ctx=ctx,
        )
        assert len(ctx.working_set) == 1
        # Второй select: cv_best winner сужает до терминального победителя.
        component._execute_select_phase(
            phase=_select_phase(name="final"),
            cfg=cfg,
            prepared=prepared,
            ctx=ctx,
        )

    assert len(ctx.working_set) == 1
    assert ctx.winner_algo == list(ctx.working_set)[0]


def test_select_top_k_without_k_raises():
    """select(sanity_gate, top_k) без k → ValueError (защита дублирует схему)."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1})})
    phase = HPOPhaseCfg.model_construct(
        name="gate",
        n_trials=1,
        action="select",
        strategy="sanity_gate",
        output="top_k",
        k=None,
    )
    with pytest.raises(ValueError, match="requires"):
        component._execute_select_phase(
            phase=phase, cfg=cfg, prepared=_prepared(), ctx=ctx
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Интеграция через train_best_model / execute_phases
# ──────────────────────────────────────────────────────────────────────────────
def test_terminal_sanity_select_persists_without_retraining(tmp_path):
    """Терминальный sanity-select: артефакт без переобучения победителя."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)
    with (
        _patch_hpo(),
        _patch_honest_finalists(df["target"]),
        patch.object(component, "_fit_and_save") as mock_fit,
    ):
        res = train_best_model(cfg, df, target="target")

    mock_fit.assert_not_called()  # тренер финалиста переиспользуется
    import pathlib

    assert pathlib.Path(res["model_path"]).exists()
    sg = res["sanity_gate"]["gate"]
    assert res["algorithm"] in sg["pool"]
    assert sg["fallback_used"] is False
    assert sg["winner_rmse_oof"] is not None


def test_select_empty_working_set_raises():
    """Пустое входное множество select'а → RuntimeError с понятным текстом."""
    cfg = _hpo_cfg()
    ctx = PhaseContext()
    with pytest.raises(RuntimeError, match="empty working set"):
        component._execute_select_phase(
            phase=_select_phase(), cfg=cfg, prepared=_prepared(), ctx=ctx
        )


def test_select_sanity_gate_full_disqualification_falls_back(tmp_path):
    """Полная дисквалификация select'а → fallback T4, запуск не падает.

    Строгий порог контура Г (gap > 1.0) при full-fit лучше OOF
    дисквалифицирует весь пул: победитель выбирается fallback-критерием,
    дисквалификации попадают в circuit breaker и gate-секцию отчёта.
    """
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")],
        sanity_gate={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # RMSE_oof=0.5 против RMSE_full=0.05: gap=10 > 1.0 → контур Г.
        return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.05)

    with (
        _patch_hpo(),
        patch.object(component, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    sg = res["sanity_gate"]["gate"]
    assert sg["fallback_used"] is True
    assert sg["disqualified"]  # все финалисты забракованы
    # Дисквалификации гейта попадают в circuit breaker (эпик #74).
    assert set(res["disqualified_algorithms"]) == set(sg["disqualified"])
    assert all("Circuit D" in r for rs in sg["disqualified"].values() for r in rs)
    # Победитель (fallback) — из пула финалистов.
    assert res["algorithm"] in sg["pool"]


def test_pipeline_pool_fallback_survives_following_refine_and_select(tmp_path):
    """Issue #102: fallback select(pool) не ломает последующие refine/select.

    Полная дисквалификация первого sanity-select возвращает весь ранжир в
    рабочее множество. Без снятия вернувшихся кандидатов с circuit breaker
    следующий refine отфильтровал бы всех и упал с RuntimeError («no valid
    results»), а select(sanity_gate) — в ``select_finalists``. С фиксом DAG
    …select(pool) → refine → select(winner)… доходит до терминального
    победителя.
    """
    cfg_dict = _cfg_dict(
        [
            _select_phase(name="gate_pool", strategy="sanity_gate", output="pool"),
            {"name": "refine", "n_trials": 2, "action": "refine"},
            _select_phase(name="gate_final", strategy="sanity_gate"),
        ],
        sanity_gate={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # RMSE_oof=0.5 против RMSE_full=0.05: gap=10 > 1.0 → контур Г.
        return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.05)

    with (
        _patch_hpo(),
        patch.object(component, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    assert res["algorithm"] in {"ridge", "extra_trees"}
    pool_section = res["sanity_gate"]["gate_pool"]
    final_section = res["sanity_gate"]["gate_final"]
    # Первый select: все финалисты провалили аудит → fallback-ранжир.
    assert pool_section["fallback_used"] is True
    assert set(pool_section["disqualified"]) == {"ridge", "extra_trees"}
    # Второй select тоже с fallback (финалисты по-прежнему вырожденные) —
    # финальный победитель выбирается формальным критерием T4.
    assert final_section["fallback_used"] is True
    assert res["algorithm"] in final_section["pool"]
    # Терминальный select (winner) сохраняет дисквалификации в circuit breaker
    # (эпик #74): победитель fallback'а по-прежнему числится в отчёте.
    assert res["algorithm"] in res["disqualified_algorithms"]


def test_select_sanity_gate_warn_mode_keeps_gap_winner(tmp_path):
    """Warn-режим контура Г (дефолт, issue #108): переобучение не дисквалифицирует.

    Кандидат с конечным gap > 1.5 остаётся валидным (soft_warning Circuit D),
    выигрывает по RMSE_oof, не попадает в ``disqualified`` и в circuit breaker
    (``disqualified_algorithms``) — иначе следующий refine/select его потеряет
    (та же ловушка, что в #102). В gate-секции видны warning и применённый
    режим.
    """
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")],
        sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9},
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # gap = 2.0 > 1.5 у обоих (мягкое предупреждение), но RMSE_oof у
        # extra_trees лучше — он обязан выиграть по RMSE_oof, а не Ridge.
        if a[0] == "ridge":
            return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.25)  # gap = 2.0
        return _FakeTrainer(df["target"].to_numpy(), 0.2, 0.1)  # gap = 2.0, RMSE_oof лучше

    with (
        _patch_hpo(),
        patch.object(component, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    sg = res["sanity_gate"]["gate"]
    assert sg["fallback_used"] is False
    assert sg["disqualified"] == {}
    assert "disqualified_algorithms" not in res  # circuit breaker пуст
    assert sg["generalization_gap_mode"] == "warn"
    assert all(
        audit["generalization_gap_mode"] == "warn"
        for audit in sg["audit"].values()
    )
    # Лучший по RMSE_oof кандидат выиграл; оба — валидны с Circuit D-warning.
    assert res["algorithm"] == "extra_trees"
    assert sg["winner_rmse_oof"] == pytest.approx(0.2, rel=1e-6)
    for audit in sg["audit"].values():
        assert audit["is_valid"] is True
        assert audit["generalization_gap"] == pytest.approx(2.0, rel=1e-4)
        assert any(
            "Circuit D" in w and "max_generalization_gap=1.500" in w
            for w in audit["soft_warnings"]
        )
        assert audit["severity"] == 1


def test_select_sanity_gate_all_finalists_fail_to_train():
    """Ни один финалист не обучился на 100% данных → RuntimeError."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    prepared = prepare_dataset(cfg, _df(), "target")

    with (
        _patch_hpo(),
        patch.object(component, "_fit_finalist", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="No finalist could be trained"):
            execute_phases(cfg, prepared)


def test_repeat_sanity_select_retrains_finalists():
    """Повторный sanity-select после HPO-фазы: финалисты обучаются заново.

    Full-fit тренеры первой sanity-фазы не переиспользуются после фазы HPO.
    """
    cfg_dict = _cfg_dict(
        [
            _select_phase(name="gate", strategy="sanity_gate", output="top_k", k=2),
            {"name": "second_hpo", "n_trials": 2, "action": "all_algorithms"},
            _select_phase(name="gate_final", strategy="sanity_gate"),
        ],
        **_GATE_THRESHOLDS,
    )
    cfg = Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    # Тренеры накапливаются по имени sanity-фазы. Имя фазы определяется
    # динамически: пул финалистов строится в НАЧАЛЕ sanity-select, поэтому
    # секции предыдущих sanity-фаз уже в отчёте, а текущей — ещё нет.
    trainers_by_phase: dict[str, list[_FakeTrainer]] = {}
    current_phase: list[str] = []

    real_build_pool = component._build_finalists_pool

    def _build_pool(cfg_arg: Any, prepared_arg: Any, ctx_arg: Any) -> list[str]:
        pending = [
            p.name
            for p in cfg_arg.general.phases
            if p.action == "select"
            and p.strategy == "sanity_gate"
            and p.name not in ctx_arg.sanity_gate_sections
        ]
        current_phase.append(pending[0])
        return real_build_pool(cfg_arg, prepared_arg, ctx_arg)

    def _fit(*a: Any, **kw: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(prepared.y.to_numpy(), 0.30, 0.25)
        trainers_by_phase.setdefault(current_phase[-1], []).append(trainer)
        return trainer

    with (
        _patch_hpo(),
        patch.object(component, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(component, "_fit_finalist", side_effect=_fit),
    ):
        ctx = execute_phases(cfg, prepared)

    # Финалисты обучались в обеих sanity-фазах новыми экземплярами тренеров;
    # разбиение по фазам не зависит от размеров пулов и набора алгоритмов.
    first_phase = trainers_by_phase["gate"]
    final_phase = trainers_by_phase["gate_final"]
    assert first_phase and final_phase
    # Терминальный sanity-select отдаёт тренера победителя, обученного в
    # НЕЙ фазе (не переиспользуя тренеры первой sanity-фазы).
    assert ctx.pre_trained_trainer in final_phase
    assert ctx.pre_trained_trainer not in first_phase
    # Полная дискретность: пересечение множеств тренеров пусто.
    assert not set(map(id, first_phase)) & set(map(id, final_phase))


# ──────────────────────────────────────────────────────────────────────────────
#  Отчёт (T6)
# ──────────────────────────────────────────────────────────────────────────────
def test_report_warn_only_keys_removed():
    """Удалённые ключи warn_only (эпик #74) не появляются в отчёте."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    prepared = _prepared()
    ctx = _make_ctx({"ridge": (-0.10, {"p": 1})})
    ctx.winner_algo = "ridge"
    ctx.sanity_gate_sections = {
        "gate": {
            "strategy": "sanity_gate",
            "pool": ["ridge"],
            "audit": {},
            "disqualified": {},
            "fallback_used": False,
            "winner_rmse_oof": 0.2,
            "overhead_seconds": 0.1,
        }
    }
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=-0.10,
            final_params={"p": 1},
            model_path_override=None,
            disqualified_algorithms=ctx.disqualified_algorithms,
            sanity_gate_sections=ctx.sanity_gate_sections,
        )
    for key in (
        "disqualified_by_sanity_gate",
        "sanity_gate_warn_only",
        "sanity_gate_overhead_seconds",
    ):
        assert key not in result
    assert "winner_if_active" not in result["sanity_gate"]["gate"]


def test_report_audit_section_serialized():
    """Аудит в gate-секции сериализован в JSON-совместимые словари."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")
    with _patch_hpo(), _patch_honest_finalists(df["target"]):
        ctx = execute_phases(cfg, prepared)
    section = ctx.sanity_gate_sections["gate"]
    for audit in section["audit"].values():
        assert isinstance(audit, dict)
        assert set(audit) == {
            "is_valid",
            "reasons",
            "soft_warnings",
            "diversity_ratio",
            "unique_ratio",
            "dead_features_count",
            "generalization_gap",
            "generalization_gap_mode",
            "severity",
        }
    # Применённый режим контура Г (issue #108) присутствует в отчёте.
    assert section["generalization_gap_mode"] == "warn"
    assert all(
        audit["generalization_gap_mode"] == "warn" for audit in section["audit"].values()
    )
    # Дисквалификации гейта синхронизированы с circuit breaker.
    assert set(section["disqualified"]) <= set(ctx.disqualified_algorithms)


# ──────────────────────────────────────────────────────────────────────────────
#  Unit-уровень вспомогательных функций
# ──────────────────────────────────────────────────────────────────────────────
def test_rank_sanity_candidates_valid_first_by_rmse_oof():
    """Ранжир стратегии sanity_gate: валидные по RMSE_oof, затем fallback-порядок."""
    result = MagicMock()
    result.pool = ["flat", "honest", "second"]
    result.audit = {
        "flat": SanityCheckResult(is_valid=False, reasons=["Circuit A"], severity=2),
        "honest": SanityCheckResult(is_valid=True, reasons=[], severity=0),
        "second": SanityCheckResult(is_valid=True, reasons=[], severity=0),
    }
    cands = {
        "flat": FinalistCandidate(
            algo_name="flat",
            model=object(),
            params={},
            cv_score=-0.08,
            rmse_oof=0.01,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "honest": FinalistCandidate(
            algo_name="honest",
            model=object(),
            params={},
            cv_score=-0.09,
            rmse_oof=0.3,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "second": FinalistCandidate(
            algo_name="second",
            model=object(),
            params={},
            cv_score=-0.10,
            rmse_oof=0.2,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
    }
    ranked = _rank_sanity_candidates(result, cands)
    assert ranked == ["second", "honest", "flat"]  # RMSE_oof: 0.2 < 0.3, flat — после


def test_rank_sanity_candidates_none_rmse_last():
    """Кандидат с недоступным RMSE_oof (None) ранжируется последним."""
    result = MagicMock()
    result.pool = ["a", "b"]
    result.audit = {
        "a": SanityCheckResult(is_valid=True, reasons=[], severity=0),
        "b": SanityCheckResult(is_valid=True, reasons=[], severity=0),
    }
    cands = {
        "a": FinalistCandidate(
            algo_name="a",
            model=object(),
            params={},
            cv_score=-0.08,
            rmse_oof=None,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "b": FinalistCandidate(
            algo_name="b",
            model=object(),
            params={},
            cv_score=-0.09,
            rmse_oof=0.5,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
    }
    assert _rank_sanity_candidates(result, cands) == ["b", "a"]


def test_serialize_audit_result_converts_nan_to_none():
    """NaN-статистики отключённых контуров заменяются на None (JSON-безопасность)."""
    audit = SanityCheckResult(
        is_valid=False,
        reasons=["Circuit A"],
        soft_warnings=["w"],
        diversity_ratio=float("nan"),
        unique_ratio=0.5,
        dead_features_count=2,
        generalization_gap=float("nan"),
        severity=2,
    )
    serialized = _serialize_audit_result(audit, "reject")
    assert serialized["diversity_ratio"] is None
    assert serialized["generalization_gap"] is None
    assert serialized["generalization_gap_mode"] == "reject"
    assert serialized["unique_ratio"] == 0.5
    assert serialized["dead_features_count"] == 2


def test_build_sanity_gate_passes_thresholds():
    """Секция ``general.sanity_gate`` — источник порогов для ModelSanityGate."""
    sg_cfg = SanityGateCfg(
        min_prediction_diversity=0.25,
        max_generalization_gap=2.0,
        generalization_gap_mode="reject",
        check_dead_features=False,
        audit_time_budget_seconds=12.5,
    )
    gate = _build_sanity_gate(sg_cfg)
    assert gate.min_prediction_diversity == 0.25
    assert gate.max_generalization_gap == 2.0
    assert gate.generalization_gap_mode == "reject"
    assert gate.check_dead_features is False
    # warn_only гейта не используется: sanity-select дисквалифицирует.
    assert gate.warn_only is False
