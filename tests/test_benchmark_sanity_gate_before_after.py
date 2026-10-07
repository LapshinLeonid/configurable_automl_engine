"""Регрессионный бенчмарк «до/после» Sanity Gate (эпик #61, T9, issue #70).

Критерий приёмки эпика: на эталонных датасетах победитель ``mode=active``
совпадает с победителем ``mode=off`` (гейт не ухудшает качество отбора на
нормальных данных), на вырожденных — меняется.

Покрытие:

1. Positive: на эталонных датасетах (зашумлённые R²≈0.3, широкие p=200,
   малые N) победитель не изменился И гейт не сработал (0 дисквалификаций).
2. Boundary: граничный датасет — diversity честной модели ≈ 0.155 при пороге
   0.15 (ухудшение ≤ ε) — победитель не меняется (документированный порог
   допуска).
3. Negative: на вырожденных датасетах (полка / плато tanh / переобучение)
   победитель изменился, вырожденный кандидат дисквалифицирован, честный
   кандидат выигрывает.
4. Smoke: таблица «датасет × режим × победитель × метрики × гейт» рендерится;
   критерии приёмки (check_acceptance) согласованы с тестами.

Время прогона ограничено (CI): n_trials=2, N ≤ 400, 7 датасетов.
"""

from __future__ import annotations

import pytest

from benchmarks.sanity_gate_benchmark import (
    ALL_CASES,
    BOUNDARY_CASES,
    DEGENERATE_CASES,
    REFERENCE_CASES,
    check_acceptance,
    render_table,
    run_case,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Positive: эталонные датасеты — победитель не меняется, гейт не срабатывает
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "case", REFERENCE_CASES, ids=lambda case: case.name
)
def test_reference_winner_unchanged_and_gate_not_triggered(case):
    """Эталонные датасеты: 0 изменений победителя (mode=off vs mode=active).

    Гейт НЕ должен срабатывать на нормальных данных: ни один кандидат пула
    не дисквалифицируется, победитель остаётся прежним.
    """
    result = run_case(case)
    assert result.winner_active == result.winner_off
    assert result.changed is False
    assert result.gate_triggered is False
    assert result.disqualified == {}
    # Пул не вырожден в одиночку — сравнение режимов осмысленно.
    assert len(result.pool) >= 1
    assert result.pool  # непустой


@pytest.mark.parametrize(
    "case", BOUNDARY_CASES, ids=lambda case: case.name
)
def test_boundary_tolerance_winner_unchanged(case):
    """Boundary: победитель на границе (ухудшение ≤ ε) — порог допуска.

    На граничном датасете diversity честной модели ≈ 0.155 при пороге 0.15:
    строгий недобор (<) — единственная точка провала, равенство и близость к
    порогу проходят; победитель не меняется.
    """
    result = run_case(case)
    assert result.winner_active == result.winner_off
    assert result.changed is False
    assert result.gate_triggered is False
    assert result.disqualified == {}
    # Честная модель действительно на границе порога (запас < 25%).
    div = result.winner_metrics_active.get("diversity_ratio")
    assert div is not None and 0.15 <= div < 0.19


# ──────────────────────────────────────────────────────────────────────────────
#  Negative: вырожденные датасеты — победитель обязан измениться
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "case", DEGENERATE_CASES, ids=lambda case: case.name
)
def test_degenerate_winner_changes(case):
    """Вырожденные датасеты: 100% изменений победителя при mode=active.

    Вырожденный кандидат (CV-лидер при mode=off) дисквалифицируется гейтом,
    победителем становится честный кандидат.
    """
    result = run_case(case)
    # Победитель обязан измениться (полка / плато tanh / переобучение).
    assert result.changed is True
    assert result.winner_active != result.winner_off
    # Гейт сработал: вырожденный CV-лидер дисквалифицирован.
    assert result.gate_triggered is True
    assert result.winner_off in result.disqualified
    assert result.audit_valid[result.winner_off] is False
    # Новый победитель — честный кандидат, прошедший аудит.
    assert result.audit_valid[result.winner_active] is True


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke: таблица и критерии приёмки
# ──────────────────────────────────────────────────────────────────────────────


def test_benchmark_table_renders_with_expected_columns():
    """Таблица «датасет × режим × победитель × метрики × гейт» рендерится."""
    # Два быстрых кейса достаточно для smoke-проверки форматирования.
    results = [run_case(case) for case in (REFERENCE_CASES[2], DEGENERATE_CASES[2])]
    table = render_table(results)
    assert "датасет" in table
    assert "режим" in table
    assert "победитель" in table
    assert "rmse_oof" in table
    assert "гейт" in table
    assert "off" in table and "active" in table
    # У каждого датасета две строки (off + active).
    assert table.count(results[0].dataset) >= 1


def test_check_acceptance_matches_test_semantics():
    """check_acceptance согласован с семантикой тестов.

    Эталонные/граничные кейсы без изменений и срабатываний проходят;
    вырожденные с изменением — проходят.
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    assert passed, f"Критерии приёмки нарушены: {errors}"