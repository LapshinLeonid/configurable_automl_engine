#!/usr/bin/env python3
"""CLI регрессионного бенчмарка «до/после» Sanity Gate (эпик #61, T9, issue #70).

Запуск из корня репозитория:

    python benchmarks/run_sanity_gate_benchmark.py

Печатает таблицу «датасет × режим × победитель × метрики × сработал ли гейт» и
проверяет критерии приёмки эпика #61:

* эталонные датасеты (зашумлённые R²≈0.3, широкие, малые N, граничный
  diversity) — 0 изменений победителя, гейт не срабатывает;
* вырожденные датасеты (полка, плато tanh, переобучение) — 100% изменений
  победителя.

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

    print("Sanity Gate: регрессионный бенчмарк «до/после» (mode=off vs mode=active)")
    print("Критерий: эталонные датасеты — 0 изменений победителя; "
          "вырожденные — 100% изменений.\n")
    results, passed, errors = run_all()
    print(render_table(results))
    print()
    if passed:
        print("ПРИЁМКА: все критерии выполнены (эталонные датасеты без изменений "
              "победителя, вырожденные — с изменением).")
        return 0
    print("ПРИЁМКА НЕ ПРОЙДЕНА:")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())