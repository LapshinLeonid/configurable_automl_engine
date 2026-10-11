#!/usr/bin/env python3
"""CLI регрессионного бенчмарка Sanity Gate на целевом пайплайне (эпик #74, R4).

Запуск из корня репозитория:

    python benchmarks/run_sanity_gate_benchmark.py

Печатает таблицу «датасет × вариант (baseline/target) × победитель ×
сработал ли гейт» и проверяет критерии приёмки эпика #74:

* эталонные датасеты (зашумлённые R²≈0.3, широкие, малые N) — победитель
  целевого DAG совпадает с CV-лидером, гейт не срабатывает;
* вырожденные датасеты (полка, плато tanh, переобучение) — 100% изменений
  победителя: вырожденный CV-лидер дисквалифицируется гейтом.

Код возврата: 0 — критерии выполнены; 1 — нарушение (CI блокирует приёмку).

Датасеты зафиксированы (seed, генератор): прогон воспроизводим, чувствительные
реальные данные в git не добавляются.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Корень репозитория — чтобы импорты tests.data_factory и benchmarks.* работали
# независимо от текущего рабочего каталога.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.sanity_gate_benchmark import (  # noqa: E402
    check_acceptance,
    render_table,
    run_all,
)


def main() -> int:
    """Запустить бенчмарк, вывести таблицу и вернуть код результата."""
    # INFO-логи Optuna (каждый триал) засоряют таблицу — оставляем только
    # предупреждения и ошибки. Логи движка на WARNING (дисквалификации гейта)
    # сохраняются: они показывают, кто и почему отсеян.
    logging.getLogger("optuna").setLevel(logging.WARNING)

    print("Sanity Gate: регрессионный бенчмарк вариантов пайплайна (эпик #74)")
    print("Варианты: baseline = all_algorithms + select(cv_best, winner);")
    print("          target   = all_algorithms + select(sanity_gate, top_k)"
          " + refine + select(sanity_gate, winner)")
    print("Критерий: эталонные датасеты — 0 изменений победителя; "
          "вырожденные — 100% изменений.\n")
    results, passed, errors = run_all()
    print(render_table(results))
    print()
    if passed:
        print("ПРИЁМКА: все критерии выполнены (эталонные датасеты без "
              "изменений победителя, вырожденные — с изменением).")
        return 0
    print("ПРИЁМКА НЕ ПРОЙДЕНА:")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())