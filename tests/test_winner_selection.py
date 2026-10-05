"""
Юнит-тесты чистой логики выбора победителя и инварианта winner-скора.

Покрывают функции ``select_winner`` и ``is_valid_winner_score`` из
``training_engine/component.py`` (issue #32):

• select_winner — детерминированный tie-break при равных скорах
  (побеждает первый в порядке конфигурации);
• is_valid_winner_score — порог класса worst-score сентинела
  (score <= float32 minimum → невалиден), включая «сырой» float32 minimum,
  который старый фильтр через ``math.isclose(rel_tol=1e-9)`` пропускал.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from configurable_automl_engine.training_engine.component import (
    is_valid_winner_score,
    select_winner,
)
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# select_winner: детерминированный выбор победителя
# ──────────────────────────────────────────────────────────────────────────────
def test_select_winner_single_algorithm():
    """Единственный алгоритм — он и есть победитель (возвращается имя-строка)."""
    results = {"ridge": (0.7, {"alpha": 1.0})}
    winner = select_winner(results)
    assert isinstance(winner, str)
    assert winner == "ridge"


def test_select_winner_max_score_wins():
    """Побеждает алгоритм с максимальным «сырым» скором."""
    results = {
        "ridge": (0.4, {"alpha": 1.0}),
        "random_forest": (0.9, {"n_estimators": 10}),
        "extra_trees": (0.6, {"n_estimators": 20}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_breaks_by_configuration_order():
    """B6 (issue #32): равные скоры → первый в порядке конфигурации.

    ``max`` стабилен, а dict сохраняет порядок вставки, поэтому ничья
    разрешается порядком итерации (порядок конфигурации для
    последовательного пути).
    """
    results = {
        "random_forest": (0.5, {"n_estimators": 10}),
        "extra_trees": (0.5, {"n_estimators": 20}),
        "elasticnet": (0.5, {"alpha": 0.5}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_reorders_with_negative_scores():
    """Ничья корректна и для отрицательных (инвертированных) скоров."""
    results = {
        "extra_trees": (-0.5, {}),
        "random_forest": (-0.5, {}),
    }
    assert select_winner(results) == "extra_trees"


def test_select_winner_empty_raises_value_error():
    """Пустой словарь результатов → ValueError (победителя нет)."""
    with pytest.raises(ValueError, match="empty results"):
        select_winner({})


# ──────────────────────────────────────────────────────────────────────────────
# is_valid_winner_score: инвариант winner-скора
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "score",
    [
        0.42,
        1.0,
        -0.9,
        -3.4e38,  # очень маленький, но валидный скор (выше float32 min)
        -3.39e38,
        0.0,
        5,  # целочисленный скор
        np.float32(0.5),
        np.float64(-0.75),
    ],
)
def test_is_valid_winner_score_accepts_finite_above_threshold(score: Any):
    """Валидные скоры: числа, конечные и строго выше float32-min порога."""
    assert is_valid_winner_score(score) is True


@pytest.mark.parametrize(
    "score",
    [
        None,
        "not-a-number",
        "0.5",  # числовая строка — всё равно мусор (issue #32, ревью PR #22)
        b"0.5",
        (0.5,),
        [0.5],
        {"value": 0.5},
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32("nan"),
        np.float32("inf"),
        np.float32("-inf"),
        HPO_WORST_SCORE,
        float(np.finfo(np.float32).min),  # B4: «сырой» float32 min
        np.float32(np.finfo(np.float32).min),
        -1e39,  # ниже float32 min — класс сентинела
        float("-inf"),
    ],
)
def test_is_valid_winner_score_rejects_sentinel_and_nonfinite(score: Any):
    """Невалидные скоры: не-числа, мусор, NaN/±inf и класс worst-score сентинела."""
    assert is_valid_winner_score(score) is False


def test_threshold_covers_hpo_worst_score_and_raw_f32_min():
    """Порог класса сентинела покрывает обе известные сигнатуры.

    HPO_WORST_SCORE (-3.4028235e38) и «сырой» float(np.finfo(np.float32).min)
    отличаются на ~9.88e-9 относительно — старый isclose(rel_tol=1e-9) второй
    не ловил. Оба обязаны попадать в класс сентинела (issue #32).
    """
    raw_f32_min = float(np.finfo(np.float32).min)
    assert HPO_WORST_SCORE != raw_f32_min  # разные double-значения
    assert not math.isclose(raw_f32_min, HPO_WORST_SCORE, rel_tol=1e-9)
    assert raw_f32_min <= WORST_SCORE_THRESHOLD
    assert HPO_WORST_SCORE <= WORST_SCORE_THRESHOLD
    # Валидный скор обязан быть строго выше порога
    assert -3.4e38 > WORST_SCORE_THRESHOLD