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


def _cfg_dict(phases: list[dict[str, Any]], **general_extra: Any) -> dict[str, Any]:
    """Валидный конфиг: HPO-фаза + заданные фазы (терминальный — с конца)."""
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
        "algorithms": {
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
        },
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
    # Дисквалификации гейта попали в circuit breaker (эпик #74).
    assert set(ctx.disqualified_algorithms) == {"extra_trees", "ridge"}
    section = ctx.sanity_gate_sections["gate"]
    assert section["fallback_used"] is True
    assert set(section["disqualified"]) == {"extra_trees", "ridge"}


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

    trainers: list[_FakeTrainer] = []

    def _fit(*a: Any, **kw: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(prepared.y.to_numpy(), 0.30, 0.25)
        trainers.append(trainer)
        return trainer

    with _patch_hpo(), patch.object(component, "_fit_finalist", side_effect=_fit):
        ctx = execute_phases(cfg, prepared)

    # Финалисты обучались в обеих sanity-фазах (новые экземпляры тренеров).
    assert len(trainers) >= 4  # ≥ 2 финалиста × 2 sanity-select
    # Терминальный sanity-select отдаёт тренера победителя, обученного в
    # НЕЙ фазе (не переиспользуя тренеры первой sanity-фазы).
    assert ctx.pre_trained_trainer in trainers[2:]
    assert ctx.pre_trained_trainer not in trainers[:2]


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
    with _patch_hpo(), _patch_honest_finalists(df["target"]):
        ctx = execute_phases(cfg, prepared := prepare_dataset(cfg, df, "target"))
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
            "severity",
        }
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
    serialized = _serialize_audit_result(audit)
    assert serialized["diversity_ratio"] is None
    assert serialized["generalization_gap"] is None
    assert serialized["unique_ratio"] == 0.5
    assert serialized["dead_features_count"] == 2


def test_build_sanity_gate_passes_thresholds():
    """Секция ``general.sanity_gate`` — источник порогов для ModelSanityGate."""
    sg_cfg = SanityGateCfg(
        min_prediction_diversity=0.25,
        max_generalization_gap=2.0,
        check_dead_features=False,
        audit_time_budget_seconds=12.5,
    )
    gate = _build_sanity_gate(sg_cfg)
    assert gate.min_prediction_diversity == 0.25
    assert gate.max_generalization_gap == 2.0
    assert gate.check_dead_features is False
    # warn_only гейта не используется: sanity-select дисквалифицирует.
    assert gate.warn_only is False
