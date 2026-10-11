"""Регрессионный бенчмарк Sanity Gate на целевом пайплайне (эпик #74, R4).

Сравнение вариантов пайплайна на одних и тех же датасетах:

* ``baseline`` — ``all_algorithms → select(cv_best, winner)`` (CV-лидер без
  аудита, современный аналог старого поведения);
* ``target`` — ``all_algorithms → select(sanity_gate, top_k) → refine →
  select(sanity_gate, winner)`` (целевой DAG эпика #74).

Покрытие:

1. Positive: на эталонных датасетах (зашумлённые R²≈0.3, широкие p=200,
   малые N) победитель не изменился И гейт не сработал (0 дисквалификаций
   ни в одном sanity-select).
2. Negative: на вырожденных датасетах (полка / плато tanh / переобучение)
   победитель изменился, вырожденный CV-лидер дисквалифицирован, честный
   кандидат выигрывает в целевом DAG.
3. Smoke: таблица «датасет × вариант × победитель × гейт» рендерится;
   критерии приёмки (check_acceptance) согласованы с тестами.

Время прогона ограничено (CI): n_trials=1, N ≤ 300, 6 датасетов.
"""

from __future__ import annotations

from benchmarks.sanity_gate_benchmark import (
    ALL_CASES,
    DEGENERATE_CASES,
    GATE_SELECT_PHASE,
    REFERENCE_CASES,
    check_acceptance,
    render_table,
    run_case,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Positive: эталонные датасеты — победитель не меняется, гейт не срабатывает
# ──────────────────────────────────────────────────────────────────────────────


def test_reference_winner_unchanged_and_gate_not_triggered():
    """Эталонные датасеты: 0 изменений победителя (baseline vs target).

    Гейт НЕ должен срабатывать на нормальных данных: ни один кандидат пула
    не дисквалифицируется, победитель целевого DAG совпадает с CV-лидером.
    """
    for case in REFERENCE_CASES:
        result = run_case(case)
        assert result.winner_target == result.winner_baseline, (
            f"{case.name}: {result.winner_baseline} -> {result.winner_target}"
        )
        assert result.changed is False
        assert result.gate_triggered is False, result.disqualified
        assert all(not names for names in result.disqualified.values())
        # Пул финалистов первого sanity-select непуст.
        assert result.pool


# ──────────────────────────────────────────────────────────────────────────────
#  Negative: вырожденные датасеты — победитель обязан измениться
# ──────────────────────────────────────────────────────────────────────────────


def test_degenerate_winner_changes():
    """Вырожденные датасеты: 100% изменений победителя.

    Вырожденный кандидат (CV-лидер базовой линии) дисквалифицируется гейтом
    в первом sanity-select целевого DAG, победителем становится честный
    кандидат.
    """
    for case in DEGENERATE_CASES:
        result = run_case(case)
        # Победитель обязан измениться (полка / плато tanh / переобучение).
        assert result.changed is True, (
            f"{case.name}: winners equal ({result.winner_baseline})"
        )
        assert result.winner_target != result.winner_baseline
        # Гейт сработал: вырожденный CV-лидер дисквалифицирован в первом
        # sanity-select; честный победитель прошёл аудит.
        assert result.gate_triggered is True
        assert result.winner_baseline in result.disqualified[GATE_SELECT_PHASE]
        assert result.audit_valid[result.winner_baseline] is False
        assert result.audit_valid[result.winner_target] is True


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke: таблица и критерии приёмки
# ──────────────────────────────────────────────────────────────────────────────


def test_benchmark_table_renders_with_expected_columns():
    """Таблица «датасет × вариант × победитель × гейт» рендерится."""
    # Два быстрых кейса достаточно для smoke-проверки форматирования.
    results = [run_case(case) for case in (REFERENCE_CASES[2], DEGENERATE_CASES[2])]
    table = render_table(results)
    assert "датасет" in table
    assert "вариант" in table
    assert "победитель" in table
    assert "гейт" in table
    assert "baseline" in table and "target" in table
    # У каждого датасета две строки (baseline + target).
    assert table.count(results[0].dataset) >= 1


def test_check_acceptance_matches_test_semantics():
    """check_acceptance согласован с семантикой тестов.

    Эталонные кейсы без изменений и срабатываний проходят; вырожденные
    с изменением и дисквалификацией CV-лидера — проходят.
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    assert passed, f"Критерии приёмки нарушены: {errors}"