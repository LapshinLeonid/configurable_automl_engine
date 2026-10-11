"""
AutoML Config Engine: Валидация и парсинг иерархических YAML-конфигураций.
Модуль реализует строго типизированную объектную модель конфигурации
на базе Pydantic v2, обеспечивая проверку целостности параметров
эксперимента, стратегий валидации и пространств поиска гиперпараметров
перед запуском пайплайна AutoML.
Ключевые возможности:
    1. Multi-Stage Pipeline: Конфигурирование последовательности шагов
       (фазы HPO 'all_algorithms' / 'refine' и явные шаги отбора 'select'
       со стратегиями cv_best / sanity_gate) с кросс-валидацией фаз
       (эпик #74).
    2. Intelligent Validation Logic: Автоматическая проверка согласованности
       настроек, включая контроль количества фолдов для k-fold стратегии
       и валидацию границ числовых диапазонов.
    3. Flexible Search Space DSL: Поддержка компактного YAML-синтаксиса
       для определения пространств поиска (Categorical, Float, Int, Log)
       с автоматическим приведением типов из списочных структур.
    4. Dependency Guard: Встроенный механизм проверки наличия необходимых
       сторонних пакетов (XGBoost и др.) для всех включенных
       в конфиг алгоритмов еще на этапе инициализации.
    5. Data Balancing Schema: Управление стратегиями оверсэмплинга
       (SMOTE, ADASYN) с поддержкой псевдонимов полей (aliasing)
       для чистоты структуры YAML-файла.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

# Стабильные коды ошибок миграции (эпик #74): тесты могут завязываться на
# ``error['type']`` из ValidationError вместо текста подсказки, который
# может меняться.
ERR_REMOVED_REFINE_WINNER = "removed_refine_winner_action"
ERR_REMOVED_SANITY_GATE_MODE = "removed_sanity_gate_mode"

from configurable_automl_engine.common.definitions import (
    ALGO_PACKAGE_MAPPING,
    ParallelStrategy,
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.hyperopt_defaults import SearchSpaceEntry
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    get_allowed_hyperparameters,
)
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride
from configurable_automl_engine.training_engine.metrics import (
    AVAILABLE_METRICS,
    to_sklearn_name,
)

# Создаем тип на лету. *AVAILABLE_METRICS распакует список в аргументы Literal
ComparisonMetric = Literal[*AVAILABLE_METRICS]  # type: ignore


_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

__all__ = [
    "AlgoCfg",
    "Config",
    "CorridorMode",
    "FeatureSelectionCfg",
    "FeatureSelectionMethod",
    "FeatureSelectionMode",
    "GeneralizationGapMode",
    "HPOPhaseAction",
    "HPOPhaseCfg",
    "SanityGateCfg",
    "SelectOutput",
    "SelectStrategy",
    "ValidationStrategy",
    "read_config",
]

#: Режим robust-выбора победителя ``select_robust_winner`` (эпик #61, T4),
#: используемый select-шагами пайплайна (эпик #74): 'off' — ранжирование
#: только по CV-скору (без аудита); 'active' — дисквалификации применяются,
#: победитель = лучший RMSE_oof. Select-шаги со ``strategy: sanity_gate``
#: выполняют режим 'active'. Режим 'warn_only' удалён вместе с ключами
#: отчёта ``sanity_gate_warn_only``/``winner_if_active`` (эпик #74, задача
#: R2). Это внутренний режим шага отбора, а НЕ поле конфигурации: поле
#: ``general.sanity_gate.mode`` удалено (эпик #74), стратегия задается
#: явно в action='select' фазах. Определение остаётся здесь (без экспорта
#: в ``__all__``), так как механизм robust-выбора в
#: ``training_engine/component.py`` импортирует этот тип напрямую; в
#: конфиг-схеме он не используется.
SanityGateMode = Literal["off", "active"]

#: Форма коридора пула финалистов (T3, issue #65): multiplicative / additive /
#: auto (выбор по типу метрики). Определяется здесь, чтобы ``SanityGateCfg``
#: и ``select_finalists`` использовали один тип без циклического импорта.
CorridorMode = Literal["multiplicative", "additive", "auto"]

#: Режим контура Г ``_check_generalization_gap`` (issue #108):
#: ``"warn"`` (дефолт) — конечный ``gap > max_generalization_gap`` уходит в
#: ``soft_warnings`` (мягкое предупреждение, не дисквалифицирует);
#: ``"reject"`` — строгая дисквалификация (старое поведение дефолта).
#: Вырожденные случаи деления на ноль (``RMSE_full ≈ 0`` / ``RMSE_oof ≈ 0``)
#: и ``y_pred_full = None`` остаются reason в обоих режимах. Имя поля
#: НЕ ``mode``: это имя зарезервировано за удалённым в эпике #74 полем
#: ``sanity_gate.mode`` (валидатор ``_reject_removed_mode`` отвергает его с
#: ошибкой миграции). Тип объявлен здесь и в ``sanity_gate.py`` (standalone)
#: без импорта, чтобы не связывать модули.
GeneralizationGapMode = Literal["warn", "reject"]

#: Действия фазы пайплайна (эпик #74): поиск по всем алгоритмам, дообучение
#: всего входного множества (warm-start) и явный шаг отбора. Устаревшее
#: значение 'refine_winner' удалено — конфиги с ним отвергаются валидацией
#: с подсказкой миграции.
HPOPhaseAction = Literal["all_algorithms", "refine", "select"]

#: Стратегия select-шага: ранжирование по CV-скору или аудит Sanity Gate.
SelectStrategy = Literal["cv_best", "sanity_gate"]

#: Что отдаёт select-шаг на выходе: единственный победитель, топ-K кандидатов
#: или весь коридорный пул финалистов.
SelectOutput = Literal["winner", "top_k", "pool"]


# ─────────────────── pruning (early stopping) ──────────────────── #
class PruningStrategy(str, Enum):
    """Поддерживаемые стратегии ранней остановки (pruning) триалов Optuna."""

    median = "median"
    hyperband = "hyperband"


class PruningCfg(BaseModel):
    """Настройки ранней остановки (pruning) триалов в фазе HPO.

    Механизм промежуточной оценки: при включении тюнер публикует оценку
    качества после каждого естественного шага валидации (например, после
    каждого фолда кросс-валидации), на основе которой прайнер Optuna может
    отсечь заведомо безнадёжный триал до завершения полной оценки.
    Для стратегий валидации без естественных шагов (hold-out /
    train_test_split) прайнер не применяется — поведение безопасно и
    задокументировано (см. API_REFERENCE.md).

    Attributes:
        enable (bool): Глобальный флаг включения ранней остановки.
            По умолчанию False — фича выключена, триалы выполняются полностью.
        strategy (PruningStrategy): Стратегия отсечения: 'median'
            (optuna.pruners.MedianPruner) или 'hyperband'
            (optuna.pruners.HyperbandPruner).
        min_steps (int): Минимальное число промежуточных шагов (например,
            фолдов CV) до первого решения об отсечении. Для 'median' это
            n_warmup_steps, для 'hyperband' — min_resource.
        n_startup_trials (int): Число завершённых триалов, после которого
            включается отсечение (только для 'median'; аналог
            n_startup_trials в Optuna).
        reduction_factor (int): Коэффициент сокращения бюджета между раундами
            Hyperband (только для 'hyperband'; контролирует агрессивность
            отсечения).
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description=(
            "Флаг включения ранней остановки (pruning) триалов Optuna. "
            "По умолчанию False — каждый триал выполняется полностью."
        ),
    )
    strategy: PruningStrategy = Field(
        default=PruningStrategy.median,
        description=(
            "Стратегия отсечения триалов: 'median' (MedianPruner) "
            "или 'hyperband' (HyperbandPruner)"
        ),
    )
    min_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Минимальное число промежуточных шагов (например, фолдов "
            "кросс-валидации) до первого решения об отсечении. Для 'median' "
            "соответствует n_warmup_steps, для 'hyperband' — min_resource"
        ),
    )
    n_startup_trials: int = Field(
        default=5,
        ge=1,
        description=(
            "Число завершённых триалов, после которого включается отсечение "
            "(только для 'median'; аналог n_startup_trials в Optuna)"
        ),
    )
    reduction_factor: int = Field(
        default=3,
        ge=2,
        description=(
            "Коэффициент сокращения бюджета между раундами Hyperband "
            "(только для 'hyperband'; агрессивность отсечения)"
        ),
    )


# ─────────────────── phases ──────────────────── #
class HPOPhaseCfg(BaseModel):
    """Конфигурация отдельного шага пайплайна (эпик #74).

    Класс описывает параметры конкретного шага: фазы HPO
    (``all_algorithms`` / ``refine``) или явного шага отбора (``select``),
    позволяя выстраивать многоступенчатые стратегии поиска и отбора
    (например: поиск по всем моделям → select с аудитом → дообучение
    финалистов → финальный select победителя).

    Attributes:
        name (str): Уникальный идентификатор фазы.
        n_trials (int): Лимит итераций (испытаний) для фаз HPO
            (``all_algorithms`` / ``refine``). Для ``select`` игнорируется,
            но поле остаётся обязательным для единообразия схемы.
        action (str): Тип действия: 'all_algorithms' (поиск по всем
            алгоритмам), 'refine' (дообучение всего входного множества
            с warm-start) или 'select' (шаг отбора). Устаревшее значение
            'refine_winner' удалено (эпик #74): конфиги с ним отвергаются
            валидацией с подсказкой миграции.
        strategy (Literal['cv_best', 'sanity_gate'] | None): Стратегия
            select-шага (только для action='select'): ранжирование по
            CV-скору ('cv_best') или аудит Sanity Gate ('sanity_gate').
        output (Literal['winner', 'top_k', 'pool'] | None): Что отдаёт
            select-шаг на выходе (только для action='select'): 'winner' —
            единственный победитель (терминальный шаг), 'top_k' — топ-K
            кандидатов, 'pool' — весь коридорный пул финалистов.
        k (int | None): Количество кандидатов (>= 1) для output='top_k'.
            Обязателен только при output='top_k'; при output='winner'/'pool'
            запрещён.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="phase",
        description=(
            "Определенное пользователем название фазы оптимизации"
            " (например, 'coarse_search' или 'fine_tuning')"
        ),
    )
    n_trials: int = Field(
        ge=1,
        description=(
            "Количество итераций (испытаний) для фаз HPO (all_algorithms / "
            "refine). Для action='select' игнорируется"
        ),
    )
    action: HPOPhaseAction = Field(
        default="all_algorithms",
        description=(
            "Действие фазы: 'all_algorithms' — поиск по всем алгоритмам, "
            "'refine' — дообучение всего входного множества с warm-start, "
            "'select' — явный шаг отбора (стратегия strategy, форма выхода "
            "output). Устаревшее 'refine_winner' удалено: используйте пару "
            "'select' (cv_best) + 'refine'"
        ),
    )
    strategy: SelectStrategy | None = Field(
        default=None,
        description=(
            "Стратегия select-шага (только для action='select'): 'cv_best' — "
            "ранжирование по CV-скору, 'sanity_gate' — пул финалистов + "
            "аудит ModelSanityGate + ранжирование по RMSE_oof. Для фаз "
            "all_algorithms/refine запрещено"
        ),
    )
    output: SelectOutput | None = Field(
        default=None,
        description=(
            "Форма выхода select-шага (только для action='select'): "
            "'winner' — единственный победитель (терминальный шаг), "
            "'top_k' — топ-K кандидатов (требует параметр k), 'pool' — "
            "весь коридорный пул финалистов. Для фаз all_algorithms/refine "
            "запрещено"
        ),
    )
    k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Количество кандидатов (>= 1) для output='top_k'; обязателен "
            "только при output='top_k', при output='winner'/'pool' запрещён. "
            "Для фаз all_algorithms/refine запрещено"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_refine_winner(cls, data: Any) -> Any:
        """Отвергнуть устаревший action 'refine_winner' с подсказкой миграции.

        Значение удалено осознанно (эпик #74) без обратной совместимости:
        молчаливое игнорирование или автоматическая трансляция скрыли бы
        смену семантики (refine_winner сужал множество до одного кандидата,
        новая пара select + refine этого не делает). Ошибка сопровождается
        подсказкой миграции.

        Args:
            data: Сырые входные данные фазы (dict при YAML/JSON-парсинге).

        Returns:
            Any: Данные без изменений, если legacy-значения нет.

        Raises:
            ValueError: Если action равен 'refine_winner'.
        """
        if isinstance(data, dict) and data.get("action") == "refine_winner":
            # Стабильный код ошибки (ERR_REMOVED_REFINE_WINNER) позволяет
            # тестам завязываться на error['type'], а не на текст подсказки.
            raise PydanticCustomError(
                ERR_REMOVED_REFINE_WINNER,
                "action 'refine_winner' has been removed (epic #74). "
                "Migration: replace the 'refine_winner' phase with explicit "
                "'select' and 'refine' phases, e.g. "
                "all_algorithms -> select(strategy: cv_best, output: top_k, "
                "k: N) -> refine -> select(strategy: cv_best, output: winner); "
                "for a single-winner pipeline use "
                "all_algorithms -> select(strategy: cv_best, output: winner). "
                "See the migration table in README.md.",
            )
        return data

    @model_validator(mode="after")
    def _validate_select_fields(self) -> HPOPhaseCfg:
        """Проверить согласованность strategy/output/k с действием фазы.

        Логика проверки:
        1. Для action='select' поля strategy и output обязательны.
        2. При output='top_k' параметр k обязателен (k >= 1 накладывается
           ограничением поля); при output='winner'/'pool' параметр k
           запрещён.
        3. Для action='all_algorithms'/'refine' поля strategy/output/k
           запрещены (сужение множества — работа select-шага).

        Returns:
            HPOPhaseCfg: Валидированный объект фазы.

        Raises:
            ValueError: Если набор полей не согласован с действием фазы.
        """
        if self.action == "select":
            missing = [
                field
                for field in ("strategy", "output")
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    f"action 'select' requires fields {missing} to be set "
                    "(strategy: 'cv_best' | 'sanity_gate', "
                    "output: 'winner' | 'top_k' | 'pool')"
                )
            if self.output == "top_k" and self.k is None:
                raise ValueError(
                    "output 'top_k' requires the integer parameter k (>= 1); "
                    "set k on the select phase or use output 'winner'/'pool'"
                )
            if self.output in ("winner", "pool") and self.k is not None:
                raise ValueError(
                    f"parameter k is not allowed for output '{self.output}' "
                    "(k is only valid for output 'top_k'); remove k or use "
                    "output 'top_k'"
                )
            return self

        forbidden = [
            field
            for field in ("strategy", "output", "k")
            if getattr(self, field) is not None
        ]
        if forbidden:
            raise ValueError(
                f"fields {forbidden} are only allowed for action 'select' "
                f"(got action '{self.action}'): candidate narrowing is the "
                "job of explicit select phases (epic #74)"
            )
        return self


# ─────────────────── feature selection ──────────────────── #
class FeatureSelectionMode(str, Enum):
    """Operation modes of the feature space reduction mechanism."""

    disabled = "disabled"  # Feature selection is off (default behavior)
    always = "always"  # Feature selection is always enabled
    auto = "auto"  # Auto mode (Optuna checks whether selection improves the metric)


class FeatureSelectionMethod(str, Enum):
    """Supported algorithms for reducing the feature space."""

    importance = "importance"  # SelectFromModel based on ExtraTreesRegressor
    percentile = "percentile"  # SelectPercentile based on f_regression
    mutual_info = "mutual_info"  # SelectPercentile based on mutual_info_regression
    variance = "variance"  # VarianceThreshold (drops quasi-constant features)


class FeatureSelectionCfg(BaseModel):
    """Settings of the feature selection mechanism.

    Attributes:
        mode (FeatureSelectionMode): Operation mode: 'disabled', 'always'
            or 'auto'.
        method (FeatureSelectionMethod): Selection algorithm: 'importance',
            'percentile', 'mutual_info' or 'variance'.
        percentile (float): Percentage of the best features to keep
            (0 < percentile <= 100) for the 'percentile' and 'mutual_info'
            methods.
        min_features (int): Guaranteed minimum number of features (>= 1),
            protects against an empty feature matrix.
        variance_threshold (float): Minimum feature variance (>= 0.0)
            for the 'variance' method.
        n_estimators (int): Number of trees in the ensemble (>= 10)
            for the 'importance' method.
    """

    model_config = ConfigDict(extra="forbid")

    mode: FeatureSelectionMode = Field(
        default=FeatureSelectionMode.disabled,
        description=(
            "Feature selection mode: 'disabled', 'always' or 'auto'. "
            "By default 'disabled' — feature selection is not applied."
        ),
    )
    method: FeatureSelectionMethod = Field(
        default=FeatureSelectionMethod.importance,
        description=(
            "Selection algorithm: 'importance' (tree-based), 'percentile' "
            "(f_regression), 'mutual_info' (mutual information) or "
            "'variance' (variance filter)."
        ),
    )
    percentile: float = Field(
        default=50.0,
        gt=0.0,
        le=100.0,
        description=(
            "Percentage of features to keep (for the 'percentile' and "
            "'mutual_info' methods)."
        ),
    )
    min_features: int = Field(
        default=2,
        ge=1,
        description=(
            "Lower bound on the number of features kept (protection against "
            "an empty feature matrix)."
        ),
    )
    variance_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum feature variance for the 'variance' method.",
    )
    n_estimators: int = Field(
        default=50,
        ge=10,
        description=(
            "Number of trees used to estimate feature importance in the "
            "'importance' method."
        ),
    )


# ─────────────────── sanity gate / select-step thresholds ─────────── #
class SanityGateCfg(BaseModel):
    """Пороги и настройки аудита Sanity Gate (эпик #61, #74).

    С версии эпика #74 секция является исключительно **источником порогов**
    для select-шагов со ``strategy: sanity_gate`` (пул финалистов: коридор δ
    от лидера ∩ жёсткий кап ``top_k_candidates``, опциональный мягкий добор
    семейств; пороги контуров аудита ``ModelSanityGate`` передаются в
    конструктор гейта без изменений). Глобального переключателя больше нет:
    поле ``mode`` удалено вместе с семантикой ``warn_only`` — отбор
    происходит ТОЛЬКО явным шагом ``select`` в ``general.phases``, а
    диагностика доступна в gate-секции отчёта каждого select-шага.

    Наличие секции без select-шагов со ``strategy: sanity_gate`` валидно:
    секция — опциональный источник порогов, а не поведенческий флаг.

    Attributes:
        top_k_candidates (int): Жёсткий кап размера пула финалистов (>= 1).
        corridor_delta (float): Относительная ширина коридора δ (в (0, 1)).
        corridor_mode (CorridorMode): 'multiplicative' | 'additive' | 'auto'.
        enforce_family_diversity (bool): Включить мягкий добор семейств.
        algorithm_families (dict[str, str] | None): Карта «алгоритм → семейство».
        allowed_families (list[str] | None): Семейства, участвующие в доборе.
        family_diversity_multiplier (float): Множитель расширенного коридора.
        min_prediction_diversity (float): Порог контура А (в (0, 1]);
            считается на OOF-предсказаниях.
        adaptive_diversity (bool): Адаптивный режим контура А.
        min_unique_count (int): Абсолютный минимум nunique контура Б (>= 1).
        min_unique_ratio (float): Нижняя граница доли уникальных (в (0, 1]).
        max_dead_feature_ratio (float): Порог контура В (в [0, 1]) — сигнал
            для линейных моделей; самостоятельная дисквалификация только при
            ``dead_features_require_low_diversity=true`` и низком diversity.
        dead_features_require_low_diversity (bool): Комбинированное правило
            «много мёртвых признаков И низкий diversity».
        zero_coef_tolerance (float): Допуск нулевых коэффициентов (>= 0).
        max_generalization_gap (float): Порог контура Г (>= 1, дефолт 1.5).
        generalization_gap_mode (GeneralizationGapMode): Режим контура Г
            (issue #108): 'warn' (дефолт) — конечный gap > порога уходит в
            soft_warnings (мягкое предупреждение); 'reject' — провал →
            reasons → дисквалификация (старое строгое поведение).
            Вырожденные случаи деления на ноль остаются reason в обоих
            режимах.
        check_permutation_sensitivity (bool): Включить пермутационный аудит
            чувствительности признаков (контур В, нелинейные модели).
        permutation_max_rows (int): Подвыборка строк для пермутаций (>= 1).
        permutation_max_features (int): Кап проверяемых колонок (>= 1).
        permutation_repeats (int): Повторы пермутаций (>= 1).
        permutation_seed (int): Фиксированное зерно пермутаций.
        permutation_tolerance (float): Порог «мёртвости» признака (>= 0).
        audit_time_budget_seconds (float): Бюджет времени на фазу аудита
            (>= 0; 0 — без ограничения).
        check_diversity (bool): Включить контур А.
        check_unique (bool): Включить контур Б.
        check_dead_features (bool): Включить контур В.
        check_generalization_gap (bool): Включить контур Г.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mode(cls, data: Any) -> Any:
        """Отвергнуть удалённое поле ``mode`` с подсказкой миграции.

        Поле удалено целиком вместе с семантикой ``warn_only`` (осознанное
        решение эпика #74): отбор происходит только явным select-шагом.
        Молчаливое игнорирование скрыло бы смену семантики, поэтому конфиг
        с ``sanity_gate.mode`` отвергается с подсказкой миграции.

        Args:
            data: Сырые входные данные секции (dict при YAML/JSON-парсинге).

        Returns:
            Any: Данные без изменений, если ключа ``mode`` нет.

        Raises:
            ValueError: Если в данных присутствует ключ ``mode``.
        """
        if isinstance(data, dict) and "mode" in data:
            # Стабильный код ошибки (ERR_REMOVED_SANITY_GATE_MODE) позволяет
            # тестам завязываться на error['type'], а не на текст подсказки.
            raise PydanticCustomError(
                ERR_REMOVED_SANITY_GATE_MODE,
                "general.sanity_gate.mode has been removed (epic #74). "
                "Migration: remove sanity_gate.mode and add an explicit "
                "select phase, e.g. {action: 'select', strategy: "
                "'sanity_gate', output: 'winner'} as the last phase; "
                "gate diagnostics are available in the gate section of the "
                "select step report. The sanity_gate section remains the "
                "source of thresholds (corridor, top_k_candidates, audit "
                "limits, ...). See the migration table in README.md.",
            )
        return data

    top_k_candidates: int = Field(
        default=3,
        ge=1,
        description="Жёсткий кап размера пула финалистов (>= 1, дефолт 3).",
    )
    corridor_delta: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description=(
            "Относительная ширина коридора δ от лидера (строго в (0, 1), дефолт 0.15)."
        ),
    )
    corridor_mode: CorridorMode = Field(
        default="auto",
        description=(
            "Форма коридора пула финалистов: 'multiplicative', 'additive' "
            "или 'auto' (выбор по типу метрики, дефолт 'auto')."
        ),
    )
    enforce_family_diversity: bool = Field(
        default=False,
        description=(
            "Опциональный мягкий добор недостающих семейств в пул финалистов "
            "(дефолт False — выключен)."
        ),
    )
    algorithm_families: dict[str, str] | None = Field(
        default=None,
        description=(
            "Карта «алгоритм → семейство» для правила добора семейств. "
            "None — алгоритмы без классификации считаются собственным семейством."
        ),
    )
    allowed_families: list[str] | None = Field(
        default=None,
        description=("Семейства, участвующие в правиле добора (None — все разрешены)."),
    )
    family_diversity_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        description="Множитель расширенного коридора для добора семейств (> 0).",
    )
    min_prediction_diversity: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description=(
            "Порог контура А (diversity): Var(y_pred_oof)/Var(y) >= порога; "
            "считается на OOF-предсказаниях (строго в (0, 1], дефолт 0.15)."
        ),
    )
    adaptive_diversity: bool = Field(
        default=False,
        description=(
            "Адаптивный режим контура А: сравнение с константной моделью "
            "(нулевой разброс предсказаний — провал)."
        ),
    )
    min_unique_count: int = Field(
        default=5,
        ge=1,
        description="Абсолютный нижний предел nunique контура Б (>= 1).",
    )
    min_unique_ratio: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Нижняя граница доли уникальных предсказаний контура Б (в (0, 1]).",
    )
    max_dead_feature_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Порог доли «мёртвых» признаков контура В (в [0, 1]) — для "
            "линейных моделей это сигнал; самостоятельная дисквалификация "
            "только при dead_features_require_low_diversity=true и низком "
            "diversity."
        ),
    )
    dead_features_require_low_diversity: bool = Field(
        default=True,
        description=(
            "Комбинированное правило контура В: дисквалификация только при "
            "сочетании «много мёртвых признаков И низкий diversity»."
        ),
    )
    zero_coef_tolerance: float = Field(
        default=1e-12,
        ge=0.0,
        description="Коэффициенты с |coef| <= порога считаются нулевыми (контур В).",
    )
    max_generalization_gap: float = Field(
        default=1.5,
        ge=1.0,
        description="Порог контура Г: RMSE_oof/RMSE_full > порога — переобучение.",
    )
    generalization_gap_mode: GeneralizationGapMode = Field(
        default="warn",
        description=(
            "Режим контура Г (generalization gap): 'warn' (дефолт) — конечный "
            "gap > max_generalization_gap записывается в soft_warnings "
            "(не дисквалифицирует); 'reject' — провал → reasons → "
            "дисквалификация (старое строгое поведение). Вырожденные случаи "
            "деления на ноль (RMSE_full≈0 / RMSE_oof≈0) остаются reason в "
            "обоих режимах. Поле намеренно НЕ называется mode (коллизия с "
            "удалённым в эпике #74 полем sanity_gate.mode)."
        ),
    )
    check_permutation_sensitivity: bool = Field(
        default=True,
        description=(
            "Включить пермутационный аудит чувствительности признаков "
            "(контур В для нелинейных моделей; контроль стоимости аудита). "
            "По умолчанию True; False — пермутационный путь пропускается, "
            "линейный путь по коэффициентам остаётся."
        ),
    )
    permutation_max_rows: int = Field(
        default=5000,
        ge=1,
        description="Подвыборка строк для пермутационного пути контура В (>= 1, дефолт 5000).",
    )
    permutation_max_features: int = Field(
        default=100,
        ge=1,
        description="Кап проверяемых колонок в пермутационном пути (>= 1, дефолт 100).",
    )
    permutation_repeats: int = Field(
        default=3,
        ge=1,
        description="Число повторов пермутаций на колонку (>= 1).",
    )
    permutation_seed: int = Field(
        default=42,
        description="Фиксированное зерно пермутаций (детерминизм).",
    )
    permutation_tolerance: float = Field(
        default=1e-3,
        ge=0.0,
        description="Относительный порог «мёртвости» признака в пермутациях.",
    )
    audit_time_budget_seconds: float = Field(
        default=0,
        ge=0.0,
        description=(
            "Бюджет времени на фазу аудита Sanity Gate (в секундах). "
            "0 — без ограничения (дефолт)."
        ),
    )
    check_diversity: bool = Field(default=True, description="Включить контур А.")
    check_unique: bool = Field(default=True, description="Включить контур Б.")
    check_dead_features: bool = Field(default=True, description="Включить контур В.")
    check_generalization_gap: bool = Field(
        default=True, description="Включить контур Г."
    )


# ─────────────────── general ─────────────────── #
class GeneralCfg(BaseModel):
    """Общие настройки процесса AutoML, валидации и параллелизма.

    Центральный узел управления экспериментом, отвечающий за выбор
    метрик, стратегий оценки качества (Cross-Validation / Hold-out)
    и распределение вычислительных ресурсов.
    Attributes:
        comparison_metric (ComparisonMetric): Основная метрика для ранжирования моделей.
        additional_metrics (List[ComparisonMetric]): Дополнительные информационные
            метрики, рассчитываемые для финальной модели тем же методом валидации,
            что и сравнение моделей (hold-out/CV, а не train-скор).
        path_to_model (Path): Путь в файловой системе для экспорта артефакта модели.
        serialization_format (SerializationFormat): Формат сохранения (pickle/joblib).
        log_to_file (Path | None): Файл для записи логов работы движка.
        phases (List[HPOPhaseCfg]): Последовательность шагов пайплайна:
            фазы HPO ('all_algorithms' / 'refine') и явные шаги отбора
            ('select'); последняя фаза обязана быть select с output: winner.
        validation_strategy (ValidationStrategy): Метод оценки (k-fold или hold-out).
        n_folds (int): Количество разбиений для кросс-валидации.
        categorical_encoding (Literal['one_hot', 'ordinal']): Стратегия кодирования
            категориальных признаков. По умолчанию 'one_hot'.
        parallel_strategy (str): Уровень распараллеливания (по алгоритмам/фолдам).
        max_workers (int | None): Лимит потоков или процессов.
        parallel_mode (str): Технический режим исполнения ('threads' или 'processes').
        feature_selection (FeatureSelectionCfg): Настройки сокращения пространства
            признаков. По умолчанию mode='disabled' — отбор признаков не применяется.
    """

    model_config = ConfigDict(extra="forbid")

    comparison_metric: ComparisonMetric = Field(
        default="r2",
        description=(
            "Метрика для сравнения моделей"
            " и выбора лучшей (например, r2, rmse, accuracy)"
        ),
    )
    additional_metrics: list[ComparisonMetric] = Field(
        default_factory=list,
        description=(
            "Список дополнительных метрик качества, значения которых рассчитываются "
            "для финальной обученной модели тем же методом валидации, что и "
            "сравнение моделей (hold-out скоринг либо среднее по фолдам CV), и "
            "возвращаются вместе с результатами обучения (ключ 'additional_metrics'). "
            "Метрики носят исключительно информационный характер: они не участвуют "
            "в оптимизации гиперпараметров, сравнении моделей и выборе победителя. "
            "Допустимые значения совпадают с набором значений comparison_metric. "
            "Дубликаты в списке игнорируются (значение считается один раз), а "
            "совпадение с основной метрикой сравнения (включая алиасы, например "
            "rmse ↔ neg_root_mean_squared_error) исключается из списка — её значение "
            "и так возвращается как 'score'. Оба правила применяются на этапе "
            "валидации конфигурации, поэтому итоговый список в "
            "general.additional_metrics может отличаться от исходного, переданного "
            "пользователем."
        ),
    )
    path_to_model: Path = Field(
        default=Path("model.pkl"),
        description="Путь для сохранения/загрузки результирующей обученной модели",
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.pickle,
        description="Формат сериализации модели (pickle, joblib и т.д.)",
    )
    log_to_file: Path | None = Field(
        default=None,
        description=(
            "Путь к файлу логов. Если не указан,логи выводятся только в консоль"
        ),
    )
    phases: list[HPOPhaseCfg] = Field(
        ..., description="Список последовательных фаз оптимизации гиперпараметров"
    )
    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.auto,
        description=(
            "Стратегия оценки качества:"
            "k-fold кросс-валидация, фиксированный hold-out,"
            "LOO или автоматический выбор"
        ),
    )
    n_folds: int = Field(
        default=5,
        description=(
            "Количество блоков (фолдов) для кросс-валидации."
            "Используется только если validation_strategy = 'k_fold'"
        ),
    )
    categorical_encoding: EncodingStrategy = Field(
        default="one_hot",
        description=(
            "Стратегия кодирования категориальных признаков: 'one_hot' (каждая "
            "категория -> отдельный бинарный столбец), 'ordinal' (каждая "
            "категориальная колонка -> один числовой столбец с наложенным "
            "порядком), 'target' (target encoding со сглаживанием), 'frequency' "
            "(частота категории) или 'hashing' (детерминированное хеширование "
            "в фиксированное число бинарных колонок). Ordinal кодирование не "
            "расширяет число колонок и полезно для линейных моделей, однако "
            "навязывает искусственный порядок категориям. Target/frequency/"
            "hashing эффективны для колонок высокой кардинальности."
        ),
    )
    high_cardinality_threshold: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Порог кардинальности для автоматического режима (>= 0). Колонки "
            "с числом уникальных значений строго больше порога кодируются "
            "стратегией high_cardinality_encoding, остальные — стратегией "
            "categorical_encoding. Должен задаваться вместе с "
            "high_cardinality_encoding. None — автоматический режим отключён."
        ),
    )
    high_cardinality_encoding: EncodingStrategy | None = Field(
        default=None,
        description=(
            "Стратегия кодирования для колонок высокой кардинальности "
            "(выше high_cardinality_threshold). Должна задаваться вместе с "
            "high_cardinality_threshold. None — режим отключён."
        ),
    )
    hashing_n_components: int = Field(
        default=16,
        ge=1,
        description=(
            "Число бинарных колонок на одну категориальную колонку при "
            "кодировании 'hashing' (>= 1). Управляет размерностью выходных "
            "признаков: рост числа признаков не зависит от кардинальности колонки."
        ),
    )
    target_encoding_smoothing: float = Field(
        default=20.0,
        ge=0,
        description=(
            "Параметр сглаживания (m) target encoding (>= 0). При m=0 "
            "используется чистое среднее целевой переменной по категории; "
            "при больших m редкие категории приближаются к глобальному среднему."
        ),
    )
    target_encoding_fallback: float | None = Field(
        default=None,
        description=(
            "Fallback-значение target encoding для категорий, отсутствующих "
            "в обучающей выборке. None — глобальное среднее целевой переменной."
        ),
    )
    parallel_strategy: ParallelStrategy = Field(
        default=next(iter(ParallelStrategy)),
        description=(
            "Стратегия распараллеливания."
            "Сейчас поддерживается только 'algorithms'"
            " (каждый алгоритм в своем потоке/процессе)."
        ),
    )
    max_workers: int | None = Field(
        default=None,
        description=(
            "Максимальное количество потоков/процессов."
            "Если null, используется количество ядер CPU"
        ),
    )
    parallel_mode: Literal["threads", "processes"] = Field(
        default="threads",
        description=(
            "Режим многозадачности: потоки (для I/O задач)"
            " или процессы (для CPU-интенсивных вычислений)"
        ),
    )
    phase_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Глобальный таймаут на всю фазу HPO (в секундах). "
            "Если None — используется значение по умолчанию 3600 секунд (1 час). "
            "В режиме parallel_mode='threads' таймаут мягкий (soft timeout): "
            "при его истечении результат задачи отбрасывается, но уже запущенная "
            "Python-функция не может быть остановлена и продолжает выполняться "
            "до естественного завершения."
        ),
    )
    task_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Таймаут на одну задачу (алгоритм) внутри фазы (в секундах). "
            "Если None — используется phase_timeout или глобальный timeout. "
            "В режиме parallel_mode='threads' таймаут мягкий (soft timeout): "
            "при его истечении результат задачи отбрасывается (None), но функция "
            "продолжает выполняться в фоне до своего завершения."
        ),
    )
    pruning: PruningCfg = Field(
        default_factory=PruningCfg,
        description=(
            "Настройки ранней остановки (pruning) триалов Optuna. "
            "По умолчанию выключены: каждый триал выполняется полностью."
        ),
    )
    feature_selection: FeatureSelectionCfg = Field(
        default_factory=FeatureSelectionCfg,
        description=(
            "Настройки сокращения пространства признаков. По умолчанию "
            "mode='disabled' — отбор признаков не применяется."
        ),
    )
    sanity_gate: SanityGateCfg = Field(
        default_factory=SanityGateCfg,
        description=(
            "Пороги и настройки аудита Sanity Gate для select-шагов со "
            "strategy: sanity_gate (эпик #61, #74). Секция — опциональный "
            "источник порогов (коридор, top_k_candidates, лимиты аудита), "
            "а не поведенческий флаг: отбор выполняется только явными "
            "select-фазами в general.phases. Наличие секции без "
            "sanity-select-шагов валидно"
        ),
    )

    @field_validator("additional_metrics")
    @classmethod
    def _deduplicate_additional_metrics(
        cls, v: list[ComparisonMetric]
    ) -> list[ComparisonMetric]:
        """Устранить дубликаты в списке дополнительных метрик.

        Задокументированное поведение: одна и та же метрика, указанная
        несколько раз, вычисляется и возвращается ровно один раз
        (порядок первого вхождения сохраняется). Дедупликация выполняется
        по строковому представлению метрики в нижнем регистре: элементы
        ``ComparisonMetric`` являются строками, поэтому вызов ``.lower()``
        типобезопасен и не зависит от произвольного типа элемента.
        """
        seen: set[str] = set()
        result: list[ComparisonMetric] = []
        for name in v:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append(name)
        return result

    @model_validator(mode="after")
    def _exclude_comparison_from_additional(self) -> GeneralCfg:
        """Исключить основную метрику сравнения из списка дополнительных.

        Значение основной метрики сравнения уже возвращается в результатах
        обучения как ``score``, поэтому дублировать его в
        ``additional_metrics`` не нужно. Сравнение выполняется по
        sklearn-имени метрики, поэтому алиасы одной и той же метрики
        (например, 'rmse' и 'neg_root_mean_squared_error') считаются
        совпадением.
        """
        comparison_sklearn = to_sklearn_name(self.comparison_metric)
        filtered = [
            m
            for m in self.additional_metrics
            if to_sklearn_name(m) != comparison_sklearn
        ]
        if len(filtered) != len(self.additional_metrics):
            logging.getLogger(__name__).warning(
                "additional_metrics contains the comparison metric '%s'; "
                "its value is already returned as 'score' and will not be "
                "duplicated in 'additional_metrics'.",
                self.comparison_metric,
            )
        self.additional_metrics = filtered
        return self

    @model_validator(mode="after")
    def _check_n_folds(self) -> GeneralCfg:
        """Проверить логическую целостность настроек валидации и сериализации.

        Логика проверки:
        1. Проверка доступности пакета `joblib`, если выбран соответствующий формат.
        2. Гарантия, что `n_folds` является положительным числом для любых стратегий.
        3. Строгая проверка для `k_fold`: количество блоков должно быть не менее 2.

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если пакет 'joblib' не установлен или значение `n_folds`
                недопустимо для выбранной стратегии.
        """
        if self.serialization_format == SerializationFormat.joblib and not is_installed(
            "joblib"
        ):
            raise ValueError(
                "serialization_format='joblib' requires"
                " the 'joblib' package to be installed"
            )
        if self.n_folds < 1:
            raise ValueError("`n_folds` must be at least 1")
        # Строгая проверка только для k-fold
        if self.validation_strategy == ValidationStrategy.k_fold and self.n_folds < 2:
            raise ValueError("n_folds must be ≥ 2 for k-fold validation")
        return self

    @model_validator(mode="after")
    def _check_pruning_consistency(self) -> GeneralCfg:
        """Проверить согласованность настроек ранней остановки с валидацией.

        Логика проверки:
        1. Для 'k_fold' прайнер должен иметь возможность сработать раньше
           окончания оценки: min_steps не может превышать n_folds.
        2. Для 'train_test_split' естественных шагов нет — ранняя остановка
           не применяется; выдаём предупреждение, а не ошибку (поведение
           задокументировано и безопасно).

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если настройки pruning конфликтуют с валидацией.
        """
        if not self.pruning.enable:
            return self

        if (
            self.validation_strategy == ValidationStrategy.k_fold
            and self.pruning.min_steps > self.n_folds
        ):
            raise ValueError(
                f"pruning.min_steps ({self.pruning.min_steps}) cannot be greater "
                f"than general.n_folds ({self.n_folds}) for k-fold validation: "
                "the pruner would never get enough intermediate steps. "
                "Decrease pruning.min_steps or increase general.n_folds."
            )

        if self.validation_strategy == ValidationStrategy.train_test_split:
            logging.getLogger(__name__).warning(
                "pruning.enable=true with validation_strategy='train_test_split': "
                "early stopping will NOT be applied because hold-out evaluation "
                "has no intermediate steps."
            )
        return self

    @model_validator(mode="after")
    def _validate_phase_pipeline(self) -> GeneralCfg:
        """Кросс-валидация последовательности фаз пайплайна (эпик #74).

        Порядок after-валидаторов в Pydantic v2 детерминирован порядком
        определения, поэтому этот валидатор осознанно объявлен РАНЬШЕ
        ``_check_high_cardinality_encoding``: ошибки структуры пайплайна
        (первопричина конфига) гарантированно сообщаются раньше ошибок
        кодирования категориальных колонок.

        Логика проверки (пустой список фаз не проверяется — защита от
        отсутствия фаз остаётся в рантайме ``train_best_model``):
        1. ``select`` не может быть первой фазой: до какого-либо поиска
           аудировать/ранжировать нечего.
        2. Фазы после терминального select (``output: winner``) запрещены:
           такой шаг отдаёт единственного победителя, и последующее
           дообучение скрыло бы результат отбора.
        3. Последняя фаза обязана быть ``select`` с ``output: winner``:
           финальный артефакт строится только из результата терминального
           шага отбора.

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если последовательность фаз нарушает правила пайплайна.
        """
        phases = self.phases
        if not phases:
            return self

        if phases[0].action == "select":
            raise ValueError(
                f"phase '{phases[0].name}' (index 0): action 'select' cannot "
                "be the first phase — there is nothing to rank/audit yet. "
                "Start the pipeline with an 'all_algorithms' (or 'refine') "
                "phase."
            )

        for index, phase in enumerate(phases[:-1]):
            if phase.action == "select" and phase.output == "winner":
                raise ValueError(
                    f"phase '{phase.name}' (index {index}): select with "
                    "output 'winner' is terminal — no phases are allowed "
                    "after it. Move it to the last position or use "
                    "output 'top_k'/'pool' to keep the pipeline going."
                )

        last = phases[-1]
        if not (last.action == "select" and last.output == "winner"):
            raise ValueError(
                f"the last phase '{last.name}' (action '{last.action}') must "
                "be a 'select' phase with output 'winner': the final artifact "
                "is built from the result of the terminal select step "
                "(epic #74). Migration: append "
                "{action: 'select', strategy: 'cv_best', output: 'winner'} "
                "as the last phase. See the migration table in README.md."
            )
        return self

    @model_validator(mode="after")
    def _check_high_cardinality_encoding(self) -> GeneralCfg:
        """Проверить согласованность настроек кодирования high-cardinality колонок.

        Логика проверки:
        1. ``high_cardinality_threshold`` и ``high_cardinality_encoding``
           задаются только вместе (иначе поведение неоднозначно — ошибка
           отклоняется до запуска обучения).
        2. Значения ``categorical_encoding`` / ``high_cardinality_encoding``
           валидируются типом ``EncodingStrategy`` (неподдерживаемая стратегия
           отклоняется на этапе парсинга конфигурации).

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если задан только один из пары параметров
                high-cardinality кодирования.
        """
        threshold = self.high_cardinality_threshold
        hc_encoding = self.high_cardinality_encoding
        if (threshold is None) != (hc_encoding is None):
            raise ValueError(
                "general.high_cardinality_threshold and "
                "general.high_cardinality_encoding must be set together: "
                f"got threshold={threshold!r}, encoding={hc_encoding!r}. "
                "Provide both parameters to enable automatic high-cardinality "
                "encoding, or omit both to disable it."
            )
        return self


# ──────────────── oversampling ──────────────── #
class OversamplingAlgorithm(str, Enum):
    random = "random"
    smote = "smote"
    adasyn = "adasyn"


class OversamplingCfg(BaseModel):
    """Параметры балансировки классов и синтеза данных.

    Обеспечивает конфигурацию методов устранения дисбаланса целевой переменной.
    Использует механизм псевдонимов (aliases) для маппинга плоских ключей
    YAML в структурированный объект.
    Attributes:
        enable (bool): Глобальный флаг включения оверсэмплинга.
        multiplier (float): Целевой коэффициент увеличения выборки.
        algorithm (OversamplingAlgorithm): Выбранный метод генерации
            (SMOTE, ADASYN и др.).
    """

    # принимать alias‑имена
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    enable: bool = Field(
        default=False,
        alias="data_oversampling",
        description=(
            "Флаг включения балансировки данных.Применяется только к обучающей выборке"
        ),
    )
    multiplier: float = Field(
        default=1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="Во сколько раз увеличить количество примеров миноритарных классов",
    )
    algorithm: OversamplingAlgorithm = Field(
        default=OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Алгоритм синтеза новых данных (Random, SMOTE, ADASYN)",
    )

    @model_validator(mode="after")
    def _validate_oversampling_logic(self) -> OversamplingCfg:
        if self.enable and self.multiplier == 1.0:
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ class balance will not change."
            )
        return self


# ───────────────── algorithms ───────────────── #


class AlgoCfg(BaseModel):
    """Техническая конфигурация конкретного ML-алгоритма в пайплайне.

    Хранит настройки включения алгоритма, переопределенные пространства
    поиска и пути к программным модулям тренера и тюнера.
    Attributes:
        enable (bool): Флаг участия алгоритма в текущем эксперименте.
        limit_hyperparameters (bool): Режим сокращенного пространства поиска.
        hyperparameters (Dict | None): Кастомные границы параметров,
            перекрывающие значения по умолчанию.
        tuner (str): Dotted-path к модулю оптимизатора.
        trainer_module (str): Dotted-path к реализации обучения модели.
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=True, description="Использовать ли данный алгоритм в пайплайне AutoML"
    )
    limit_hyperparameters: bool = Field(
        default=False, description=("Ограничить гиперпараметры пространства поиска")
    )
    hyperparameters: dict[str, SearchSpaceEntry] | None = Field(
        default=None,
        description=(
            "Ключи должны соответствовать допустимым гиперпараметрам алгоритма: "
            "параметрам конструктора sklearn-оценщика (включая legacy-имена из "
            "LEGACY_PARAM_MAPPINGS). Источник истины — "
            "models.get_allowed_hyperparameters."
        ),
    )
    preprocessing: PreprocessingOverride | None = Field(
        default=None,
        description=(
            "Явное переопределение пресета предобработки признаков для данного "
            "алгоритма (FR-5). Задаётся частично или полностью: 'imputation_strategy' "
            "('mean'|'median') и/или 'scaling' ('standard'|'robust'|'none'). "
            "Имеет приоритет над автоматическим выбором по классу алгоритма. "
            "None — автоматический выбор."
        ),
    )
    tuner: str | None = Field(
        default="configurable_automl_engine.tuner",
        description="Путь к модулю тюнера для оптимизации гиперпараметров",
    )
    trainer_module: str | None = Field(
        default="configurable_automl_engine.trainer",
        description=(
            "Dotted-path к модулю, содержащему класс `ModelTrainer`"
            "(например, 'configurable_automl_engine.trainer')."
        ),
    )

    def get_required_package(self, algo_name: str) -> str | None:
        """Определить имя внешнего Python-пакета, необходимого для работы алгоритма.

        Логика поиска:
        1. Обращается к глобальному маппингу `ALGO_PACKAGE_MAPPING`.
        2. Сопоставляет внутреннее имя алгоритма (например, 'xgboost')
            с названием в PyPI.

        Args:
            algo_name (str): Уникальный идентификатор алгоритма.
        Returns:
            Optional[str]: Название пакета для установки через pip или None,
                если зависимость не определена.
        """
        return ALGO_PACKAGE_MAPPING.get(algo_name)

    def get_unknown_hyperparameters(self, algo_name: str) -> list[str]:
        """Вернуть список гиперпараметров, несовместимых с данным алгоритмом.

        Сверяет ключи `self.hyperparameters` с допустимым множеством
        параметров конструктора оценки (см.
        `models.get_allowed_hyperparameters`), включая legacy-имена из
        `LEGACY_PARAM_MAPPINGS`. Если допустимое множество определить нельзя
        (алгоритм неизвестен, зависимость не установлена или конструктор
        принимает **kwargs) — проверка пропускается (мягкий fallback).

        Args:
            algo_name (str): Уникальный идентификатор алгоритма.
        Returns:
            List[str]: Список недопустимых ключей. Пустой список — всё корректно.
        """
        if self.hyperparameters is None:
            return []
        allowed = get_allowed_hyperparameters(algo_name)

        if allowed is None:
            return []

        return [k for k in self.hyperparameters if k not in allowed]

    @field_validator("tuner", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if v is None:
            return v
        if not _DOTTED_PATH_RE.fullmatch(v):
            raise ValueError(
                f"'{v}' is not a valid dotted path "
                "(expecting 'package.module' or 'a.b.c.Class' format)"
            )
        return v


class _AlgorithmsConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


AlgorithmsConfig = create_model(
    "AlgorithmsConfig",
    __base__=_AlgorithmsConfigBase,
    **{name: (AlgoCfg | None, Field(default=None)) for name in AVAILABLE_ALGORITHMS},
)  # type: ignore[call-overload]

if TYPE_CHECKING:
    # для mypy используем статический alias
    AlgorithmsConfigType = _AlgorithmsConfigBase
else:
    # runtime — динамический create_model
    AlgorithmsConfigType = AlgorithmsConfig


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Корневой объект всей системы конфигурации AutoML.

    Агрегирует все секции настроек и выполняет финальную валидацию
    целостности графа параметров, включая проверку системных зависимостей.
    Attributes:
        general (GeneralCfg): Общие параметры эксперимента.
        oversampling (OversamplingCfg): Настройки предобработки данных.
        algorithms (AlgorithmsConfig): Реестр доступных и активных алгоритмов.
    """

    model_config = ConfigDict(extra="forbid")

    general: GeneralCfg = Field(
        ..., description="Общие настройки эксперимента и валидации"
    )
    oversampling: OversamplingCfg = Field(
        default=OversamplingCfg(), description="Настройки балансировки данных"
    )
    algorithms: AlgorithmsConfigType = Field(
        ...,
        description=(
            "Словарь алгоритмов, где ключ — имя алгоритма "
            "(например, 'xgboost', 'random_forest')"
        ),
    )

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Any) -> Any:
        # Поскольку 'v' теперь — это AlgorithmsConfig (объект),
        # мы получаем все его поля через .model_dump()
        enabled_algorithms = [
            algo
            for algo in v.model_dump().values()
            if algo is not None and algo.get("enable") is True
        ]

        if not enabled_algorithms:
            raise ValueError("At least one algorithm must be enabled (enable: true)")
        return v

    @model_validator(mode="after")
    def _check_algorithm_dependencies(self) -> Config:
        # model_fields через экземпляр тоже работает, но deprecated
        # (PydanticDeprecatedSince211, удаление в V3.0) — используем класс
        for name in type(self.algorithms).model_fields:
            # Get the attribute value (could be None if not provided in data)
            algo_cfg = getattr(self.algorithms, name)

            # Skip if the algorithm entry is missing (None)
            if algo_cfg is None:
                continue

            # Ensure it has an 'enable' attribute before checking it
            # If it's a Pydantic model, check attribute, otherwise get() as dict
            enabled = getattr(algo_cfg, "enable", False)

            if enabled:
                # Safely get the required package
                required_pkg = getattr(
                    algo_cfg, "get_required_package", lambda n: None
                )(name)

                if required_pkg and not is_installed(required_pkg):
                    raise ValueError(
                        f"Algorithm '{name}' is enabled, but the package "
                        f"'{required_pkg}' is not installed. "
                        f"Please run: pip install {required_pkg}"
                    )
        return self

    @model_validator(mode="after")
    def _check_hyperparameter_compatibility(self) -> Config:
        errors = []
        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            unknown = algo_cfg.get_unknown_hyperparameters(name)
            if unknown:
                allowed = sorted(get_allowed_hyperparameters(name) or set())
                errors.append(
                    f"Algorithm '{name}': unknown hyperparameters {unknown}. "
                    f"Allowed parameters: {allowed}"
                )
        if errors:
            raise ValueError("\n".join(errors))
        return self


# ────────────────── API ────────────────────── #
def read_config(path: str | Path) -> Config:
    """Загрузить, распарсить и валидировать конфигурационный файл эксперимента.

    Логика работы:
    1. Открывает файл по указанному пути в кодировке UTF-8.
    2. Выполняет безопасную загрузку YAML-структуры в Python-словарь.
    3. Инициирует каскадную валидацию Pydantic
        для создания типизированного объекта `Config`.
    Args:
        path (str | Path): Путь к файлу конфигурации в формате .yaml или .yml.
    Returns:
        Config: Корневой объект конфигурации, готовый к использованию в движке AutoML.

    Raises:
        FileNotFoundError: Если файл по указанному пути не найден.
        ValidationError: Если структура файла не соответствует схеме или
            нарушена логика параметров.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
