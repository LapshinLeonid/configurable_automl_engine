"""E2E целевого пайплайна эпика #74 (задача R4, issue #78).

Целевой DAG: ``all_algorithms → select(sanity_gate, top_k) → refine →
select(sanity_gate, winner)``.

Покрытие:

1. E2E-сценарий: сквозной прогон DAG на тестовом датасете —
   gate-победитель дообучается (refine получает его параметры как
   ``initial_params`` из gate-контекста), повторный select переаудитит
   обновлённые модели (финалисты обучаются заново — честная цена),
   финальный артефакт сохраняется через ``pre_trained_trainer`` без
   переобучения победителя, отчёт содержит секции ``sanity_gate`` по обоим
   select-шагам с их именами фаз.
2. Негативные сквозные сценарии: конфиг с устаревшими сущностями
   (``refine_winner``, ``sanity_gate.mode``) доходит до дружелюбной ошибки
   валидации на входе ``train_best_model``.
3. Пограничные: refine после select(top_k) не теряет ни одного из k
   кандидатов; повторная дисквалификация всех после refine → fallback T4
   в терминальном select.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import configurable_automl_engine.training_engine.component as component
from configurable_automl_engine.training_engine.component import (
    execute_phases,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    ERR_REMOVED_REFINE_WINNER,
    ERR_REMOVED_SANITY_GATE_MODE,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Общие помощники
# ──────────────────────────────────────────────────────────────────────────────


def _target_dag_phases(
    top_k: int = 2,
    *,
    first_select_name: str = "gate_select",
    refine_name: str = "refine",
    terminal_select_name: str = "gate_final",
) -> list[dict[str, Any]]:
    """Фазы целевого DAG эпика #74."""
    return [
        {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
        {
            "name": first_select_name,
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "top_k",
            "k": top_k,
        },
        {"name": refine_name, "n_trials": 2, "action": "refine"},
        {
            "name": terminal_select_name,
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "winner",
        },
    ]


def _cfg_dict(
    phases: list[dict[str, Any]],
    *,
    gate_thresholds: dict[str, Any] | None = None,
    **general_extra: Any,
) -> dict[str, Any]:
    """Валидный конфиг с заданными фазами (последняя — терминальный select)."""
    general: dict[str, Any] = {
        "comparison_metric": "mae",
        "validation_strategy": "train_test_split",
        "phases": phases,
        "path_to_model": "model.pkl",
        **general_extra,
    }
    if gate_thresholds is not None:
        general["sanity_gate"] = gate_thresholds
    return {
        "general": general,
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


def _df(n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    """Числовой DataFrame с целевой колонкой."""
    rng = np.random.RandomState(seed)
    data = {f"f{i}": rng.randn(n_rows) for i in range(3)}
    data["target"] = 2.0 * data["f0"] + rng.randn(n_rows) * 0.1
    return pd.DataFrame(data)


_GATE_THRESHOLDS = {"top_k_candidates": 2, "corridor_delta": 0.9}


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

        self.pipeline = self
        self.oof_predictions_ = y + _noise(oof_rmse)
        self.oof_score_ = oof_rmse
        self._full_pred = y + _noise(full_rmse)

    def predict(self, X: Any) -> np.ndarray:
        return self._full_pred[: len(X)]

    def save(self, path: Any) -> None:
        Path(path).write_bytes(b"fake-model")


def _houndstooth_hpo(**kwargs: Any) -> tuple[float, dict[str, Any]]:
    """Детерминированный HPO: extra_trees лучше ridge; refine — boost.

    Coarse-фаза возвращает фиксированные (score, params). Refine-фаза
    вызывается с ``initial_params`` из gate-контекста (проверяется ниже)
    и возвращает улучшенные (score', params').
    """
    algo = kwargs["algo_name"]
    if kwargs.get("initial_params") is None:
        # coarse (all_algorithms): gate-контекстные параметры фазы 1.
        return {
            "ridge": (-0.10, {"alpha": 1.0}),
            "extra_trees": (-0.08, {"n_estimators": 10}),
        }[algo]
    # refine: warm-start от параметров выхода первого select (gate-контекст).
    if algo == "ridge":
        assert kwargs["initial_params"] == {"alpha": 1.0}, kwargs["initial_params"]
        return -0.09, {"alpha": 1.5}
    assert kwargs["initial_params"] == {"n_estimators": 10}, kwargs["initial_params"]
    return -0.07, {"n_estimators": 20}


# ──────────────────────────────────────────────────────────────────────────────
#  1. E2E целевого DAG
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_target_dag_refine_select_persist_without_retraining(tmp_path: Path):
    """Сквозной прогон целевого DAG: дообучение, повторный аудит, артефакт.

    Проверки:
    - refine получает параметры gate-победителя как ``initial_params``
      (тюнер победителя стартует с параметров из gate-контекста);
    - терминальный select переаудитит обновлённые модели — финалисты
      обучаются заново (честная цена, старые full-fit тренеры не
      переиспользуются);
    - артефакт сохраняется через ``pre_trained_trainer`` — ``_fit_and_save``
      не вызывается;
    - отчёт содержит секции ``sanity_gate`` с именами обоих select-фаз.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds=_GATE_THRESHOLDS,
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    # Тренировка финалистов: фаза 1 (gate_select) и фаза 2 (gate_final)
    # обучают финалистов новыми экземплярами; refine финалистов не обучает.
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

    def _fit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(prepared.y.to_numpy(), 0.30, 0.25)
        trainers_by_phase.setdefault(current_phase[-1], []).append(trainer)
        return trainer

    with (
        patch.object(
            component, "_run_hpo", side_effect=_houndstooth_hpo
        ) as mock_hpo,
        patch.object(component, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(component, "_fit_finalist", side_effect=_fit_finalist),
        patch.object(component, "_fit_and_save") as mock_fit,
    ):
        result = train_best_model(cfg, df, target="target", model_path_override=tmp_path / "m.pkl")

    # 1a. Артефакт сохранён без переобучения победителя (pre_trained_trainer).
    mock_fit.assert_not_called()
    assert Path(result["model_path"]).exists()

    # 1b. Refine стартовал warm-start'ом от параметров из gate-контекста.
    # Вызовы _run_hpo: 2 (coarse) + 2 (refine). Для каждого кандидата
    # initial_params refine == параметры из выхода первого select.
    refine_calls = [
        call
        for call in mock_hpo.call_args_list
        if call.kwargs.get("initial_params") is not None
    ]
    assert len(refine_calls) == 2
    refine_params = {
        call.kwargs["algo_name"]: call.kwargs["initial_params"]
        for call in refine_calls
    }
    assert refine_params == {
        "ridge": {"alpha": 1.0},
        "extra_trees": {"n_estimators": 10},
    }

    # 1c. Повторный select переаудитит обновлённые модели: финалисты
    # терминального select обучаются заново (не переиспользуются тренеры
    # первого select).
    assert set(trainers_by_phase) == {"gate_select", "gate_final"}
    assert all(trainers_by_phase.values())
    first = trainers_by_phase["gate_select"]
    final = trainers_by_phase["gate_final"]
    assert not set(map(id, first)) & set(map(id, final))
    # Победитель терминального select — из пула финалистов ЕГО фазы.
    assert result["algorithm"] in result["sanity_gate"]["gate_final"]["pool"]

    # 1d. Отчёт: секции sanity_gate по обоим select-шагам с именами фаз.
    assert set(result["sanity_gate"]) == {"gate_select", "gate_final"}
    for section in result["sanity_gate"].values():
        assert section["strategy"] == "sanity_gate"
        assert section["fallback_used"] is False
        assert section["winner_rmse_oof"] is not None
        assert section["overhead_seconds"] >= 0


def test_e2e_target_dag_winner_refined_params_in_gate_context():
    """Победитель терминального select действительно дообучен (R3+R4).

    refine заменяет параметры победителя улучшенными (warm-start): терминальный
    select аудитит обновлённую модель и выдаёт её параметры в result.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=1),
        gate_thresholds={"top_k_candidates": 2, "corridor_delta": 0.9},
    )
    cfg = component.Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                prepared.y.to_numpy(), 0.30, 0.25
            ),
        ),
    ):
        ctx = execute_phases(cfg, prepared)

    # top_k=1: после первого select в рабочем множестве один кандидат —
    # лучший по RMSE_oof. Его refine-параметры — улучшенная версия
    # coarse-параметров (warm-start шёл от gate-контекста).
    assert len(ctx.working_set) == 1
    winner = ctx.winner_algo
    assert winner is not None
    score, params = ctx.working_set[winner]
    # Победитель обновлён refine-фазой: параметры из refine-стаба.
    assert params in ({"alpha": 1.5}, {"n_estimators": 20})
    assert score in (-0.09, -0.07)
    assert ctx.pre_trained_trainer is not None
    assert set(ctx.sanity_gate_sections) == {"gate_select", "gate_final"}


# ──────────────────────────────────────────────────────────────────────────────
#  2. Негативные сквозные сценарии: legacy-сущности → дружелюбная валидация
# ──────────────────────────────────────────────────────────────────────────────


def test_train_best_model_rejects_legacy_refine_winner_action():
    """Конфиг с ``refine_winner`` падает на входе ``train_best_model``.

    Ошибка — дружелюбная валидация со стабильным кодом
    ``removed_refine_winner_action`` и подсказкой миграции.
    """
    cfg_dict = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                {"name": "refine", "n_trials": 1, "action": "refine_winner"},
            ],
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError) as excinfo:
        train_best_model(config=cfg_dict, df=_df(), target="target")

    errors = excinfo.value.errors()
    assert any(
        error["type"] == ERR_REMOVED_REFINE_WINNER for error in errors
    ), errors
    assert any("refine_winner" in error.get("msg", "") for error in errors)


def test_train_best_model_rejects_legacy_sanity_gate_mode():
    """Конфиг с ``sanity_gate.mode`` падает на входе ``train_best_model``.

    Ошибка — дружелюбная валидация со стабильным кодом
    ``removed_sanity_gate_mode`` и подсказкой миграции.
    """
    cfg_dict = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "sanity_gate",
                    "output": "winner",
                },
            ],
            "sanity_gate": {"mode": "active"},
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError) as excinfo:
        train_best_model(config=cfg_dict, df=_df(), target="target")

    errors = excinfo.value.errors()
    assert any(
        error["type"] == ERR_REMOVED_SANITY_GATE_MODE for error in errors
    ), errors
    assert any("sanity_gate.mode" in error.get("msg", "") for error in errors)


# ──────────────────────────────────────────────────────────────────────────────
#  3. Пограничные сценарии
# ──────────────────────────────────────────────────────────────────────────────


def test_refine_after_select_top_k_keeps_all_k_candidates():
    """refine после select(sanity_gate, top_k) не теряет ни одного из k.

    Схема требует терминальный select последней фазой, поэтому конфиг
    валидируется с ним, а для прогона фазы усекаются до
    [all_algorithms, select(top_k), refine] (подход юнит-тестов R2/R3).
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds=_GATE_THRESHOLDS,
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.phases = cfg.general.phases[:-1]  # убрать терминальный select
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                prepared.y.to_numpy(), 0.30, 0.25
            ),
        ),
    ):
        ctx = execute_phases(cfg, prepared)

    # Первый select (top_k=2) не сузил множество: оба кандидата остались,
    # refine не теряет ни одного (все прошли refine-HPO).
    assert set(ctx.working_set) == {"ridge", "extra_trees"}
    assert len(ctx.working_set) == 2


def test_all_disqualified_after_refine_terminal_select_falls_back_t4():
    """Повторная дисквалификация всех после refine → fallback T4.

    Первый sanity-select пропускает честных финалистов; после refine
    терминальный select аудитит обновлённые (вырожденные) модели — весь
    пул дисквалифицирован, победитель выбирается fallback-критерием T4,
    запуск не падает.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.path_to_model = "m.pkl"
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

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

    def _fit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        if current_phase[-1] == "gate_select":
            # Первый select: честные финалисты (gap=0.83 < 1.0 — строго ниже
            # порога контура Г; равенство плавающей арифметикой не срабатывает).
            return _FakeTrainer(prepared.y.to_numpy(), 0.25, 0.30)
        # Терминальный select: вырожденные модели (gap=10 > 1.0 → контур Г).
        return _FakeTrainer(prepared.y.to_numpy(), 0.5, 0.05)

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(component, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(component, "_fit_finalist", side_effect=_fit_finalist),
        patch.object(component, "_fit_and_save"),
    ):
        result = train_best_model(cfg, df, target="target")

    terminal = result["sanity_gate"]["gate_final"]
    assert terminal["fallback_used"] is True
    assert terminal["disqualified"]  # все финалисты забракованы после refine
    assert all("Circuit D" in r for rs in terminal["disqualified"].values() for r in rs)
    # Победитель (fallback) — из пула финалистов терминального select.
    assert result["algorithm"] in terminal["pool"]
    assert result["algorithm"] in result["sanity_gate"]["gate_select"]["pool"]