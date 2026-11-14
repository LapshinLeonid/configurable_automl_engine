# CONTRACTS.md — аудит внешних контрактов `configurable-automl-engine`

Документ подготовлен в рамках C1 эпика [#79](https://sourcecraft.dev/ooo-ainod/configurable-automl-engine/issues/79)
(задача [#80](https://sourcecraft.dev/ooo-ainod/configurable-automl-engine/issues/80)).
Код в рамках C1 не изменялся: документ фиксирует текущее состояние, границы и
решения, на которые должны опираться C2–C7.

## 1. Назначение и границы аудита

Аудит отвечает на три вопроса:

1. Какие определения `configurable_automl_engine` являются публичным
   контрактом, какие — контрактом конфигурации, а какие — реализацией движка.
2. Какие пути импорта уже используются потребителями и какие из них C3 обязан
   сохранить через реэкспорт.
3. Что именно можно перенести в `automl-contracts`, не притащив в контракт
   `scikit-learn`, `pandas`, `numpy`, `optuna`, `PyYAML` и другие runtime-зависимости.

Три уровня из эпика применяются буквально:

* **публичный контракт библиотеки** — DTO, модели данных, перечисления,
  `Literal`/`TypeAlias`, публичные исключения, стабильные коды ошибок,
  публичные константы, `Protocol`, JSON Schema, типы результатов и статусов;
* **контракт конфигурации** — структура конфигурационных данных и правила их
  структурной и типовой валидации;
* **реализация движка** — алгоритмы, оркестрация, выполнение экспериментов,
  интеграция со `scikit-learn`, `pandas`, `numpy`, `optuna` и окружением.

Не все публично импортируемые символы автоматически становятся частью
`automl-contracts`: публичный статус и решение о переносе — разные поля матрицы.

### 1.1. Что изучено

* исходный код `src/configurable_automl_engine/` (в первую очередь модули из
  скоупа задачи и фактические импорты между ними);
* корневой `__init__.py` и `__all__` подпакетов;
* `API_REFERENCE.md`, `README.md`, `example.py`;
* `tests/`, `scripts/`, `benchmarks/`, `tools/` — как фактические потребители
  внутри репозитория;
* `config.schema.json` и `schema_generator.py`;
* `pyproject.toml` (зависимости, версия Python, конфигурация mypy/import-linter).

### 1.2. Ограничение по внешним потребителям

В окружении выполнения C1 не было доступа к репозиториям-потребителям
организации: перечисление репозиториев организации недоступно текущей сессии.
Поэтому матрица «известные потребители» ниже опирается на:

* документированные публичные примеры и пути импорта;
* реальные импорты в тестах, скриптах и бенчмарках этого репозитория;
* публичные экспорты пакета.

Это не отменяет требования C5/C6: перед стабилизацией `1.0.0` нужно отдельно
проверить downstream-репозитории и зафиксировать найденные пути импорта в
матрице совместимости. Все места, где вывод основан только на документации,
помечены в матрице как «документация/тесты», а не как подтверждённые внешние
потребители.

## 2. Легенда

### 2.1. Статус публичности

* `public` — символ является частью внешнего API: на него можно опираться.
* `internal` — символ относится к реализации движка и внешним контрактом не является.
* `deprecated` — символ сохранён для совместимости, но не рекомендуется к
  использованию; удаление — только по политике SemVer.
* `needs-decision` — статус требует решения C2/C6; в документе приведена
  рекомендация и аргументы.

### 2.2. Решение

* `move` — определение переносится в `automl_contracts`, старый путь получает
  реэкспорт;
* `re-export` — определение остаётся там, где целесообразно, но старый путь
  сохраняется через реэкспорт;
* `internal` — остаётся в движке и в контракт не входит;
* `deprecated` — сохраняется, но помечается устаревшим;
* `needs-decision` — решение откладывается до C2/C6 с явным вопросом.

### 2.3. Стабильность

* `stable` — обещание обратной совместимости по SemVer;
* `beta` — публичный, но до `1.0.0` допускает изменения с повышением MINOR и
  записью в `CHANGELOG.md`;
* `internal` — без обещаний совместимости.

Предлагаемые новые пути ниже — **целевая схема для C2**, а не уже принятое
решение о раскладке дистрибутивов. Имена модулей внутри `automl_contracts`
могут быть уточнены ADR; важна фиксированная граница «контракт ↔ движок».

## 3. Итоговая классификация

| Уровень | Что входит | Что не входит |
|---|---|---|
| Публичный контракт библиотеки | `Config` и секционные Pydantic-модели, перечисления, `Literal`/`TypeAlias`, `SearchSpaceEntry` и distribution-модели, `PreprocessingOverride`, `FeatureSelectionCfg`, `CustomMetricSpec` (декларация), `ExperimentResult`, `AlgorithmRun`, статусы, `TrainingError`, стабильные коды ошибок, `WORST_SCORE_THRESHOLD`, JSON Schema | `ModelTrainer`, `train_model`, `optimize`, `create_model`, `FeatureSelector`, `DataOversampler`, `build_preprocessor`, `DEFAULT_SPACES`, HPO runner/evaluation, оркестрация `training_engine` |
| Контракт конфигурации | Структура `Config`, типы полей, диапазоны, обязательность, структурные cross-field правила, стабильные коды миграционных ошибок | Проверка установленного backend/пакета, доступность `joblib`, разрешённые гиперпараметры по сигнатуре sklearn, проверки ресурсов и окружения |
| Реализация движка | Обучение, препроцессинг, HPO, отбор, артефакты, отчёты, store, explain, параллельное исполнение | Определения DTO и конфигурационных моделей, которые нужны потребителю без движка |

Практическое правило для C3: **переносится только определение и чистые
pydantic-правила; всё, что требует импорта sklearn/pandas/numpy/optuna/PyYAML
или чтения окружения, остаётся в движке.**

## 4. Матрица публичного API по модулям

В таблицах:
* «Текущий путь» — фактический модуль определения;
* «Новый путь» — предлагаемый путь в `automl-contracts` (C2 может уточнить);
* «Потребители» — документация, тесты, скрипты и внутренние импорты;
* «Статус» — публичность символа;
* «Решение» — судьба при миграции;
* «Стабильность» — уровень обещаний.

### 4.1. `common/definitions.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `ValidationStrategy` | enum | `GeneralCfg`, `tuner.optimize`, `validation`, `ExperimentResult`, тесты, API_REFERENCE | `configurable_automl_engine.common.definitions.ValidationStrategy` | `automl_contracts.definitions.ValidationStrategy` | public | move + re-export | stable |
| `EncodingStrategy` | `Literal`/`TypeAlias` | `GeneralCfg`, `preprocessing`, тесты, документация по categorical encoding | `...common.definitions.EncodingStrategy` | `automl_contracts.definitions.EncodingStrategy` | public | move + re-export | stable |
| `SerializationFormat` | enum | `GeneralCfg`, `ModelTrainer`, тесты, API_REFERENCE | `...common.definitions.SerializationFormat` | `automl_contracts.definitions.SerializationFormat` | public | move + re-export | stable |
| `TargetTransform` | enum | `GeneralCfg`, `ModelTrainer`, `preprocessing.target_transform`, тесты | `...common.definitions.TargetTransform` | `automl_contracts.definitions.TargetTransform` | public | move + re-export | stable |
| `ALGO_PACKAGE_MAPPING` | `dict[str, str]` | `AlgoCfg.get_required_package`, runtime-проверка зависимостей | `...common.definitions.ALGO_PACKAGE_MAPPING` | остаётся в движке; в контракт не входит | internal | internal | internal |
| `ParallelStrategy` | enum | `GeneralCfg` | `...common.definitions.ParallelStrategy` | `automl_contracts.definitions.ParallelStrategy` | public | move + re-export | stable |
| `RunStatus` | enum | `ExperimentResult`, reporters, tests | `...common.definitions.RunStatus` | `automl_contracts.definitions.RunStatus` | public | move + re-export | stable |
| `RunStepStatus` | enum | `AlgorithmRun`, reporters, tests | `...common.definitions.RunStepStatus` | `automl_contracts.definitions.RunStepStatus` | public | move + re-export | stable |
| `RunPhaseAction` | enum | `AlgorithmRun`, reporters, tests | `...common.definitions.RunPhaseAction` | `automl_contracts.definitions.RunPhaseAction` | public | move + re-export | stable |
| `MetricDirection` | enum | `ExperimentResult`, ensemble, reporters, tests | `...common.definitions.MetricDirection` | `automl_contracts.definitions.MetricDirection` | public | move + re-export | stable |
| `WORST_SCORE_THRESHOLD` | `float` | `ExperimentResult`, `hpo.types`, `tuner`, tests | `...common.definitions.WORST_SCORE_THRESHOLD` | `automl_contracts.definitions.WORST_SCORE_THRESHOLD` | public | move + re-export; убрать `numpy` из определения | beta |

**Проблема модуля.** `definitions.py` импортирует `numpy` только для
`WORST_SCORE_THRESHOLD = float(np.finfo(np.float32).min)`. Контракт не может
зависеть от `numpy`. Значение нужно зафиксировать литералом
`-3.4028234663852886e+38` (точное значение `float(np.finfo(np.float32).min)`)
с комментарием, что это нижняя граница sentinel-класса; движок должен
продолжать сравнивать с тем же значением. Это также устраняет единственную
runtime-зависимость «нижнего» модуля.

`ALGO_PACKAGE_MAPPING` — не контракт данных, а runtime-метаданные о пакетах.
Он должен остаться в движке (например, рядом с runtime-валидацией `AlgoCfg`);
старый путь `common.definitions` можно сохранить только как реэкспорт на
переходный период, но в `automl_contracts` его быть не должно.

### 4.2. `common/errors.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `TrainingError` | exception | `ModelTrainer`, `trainer_params`, `trainer_feature_selection`, API_REFERENCE, tests | `...common.errors.TrainingError`, реэкспорт `...trainer.TrainingError` | `automl_contracts.errors.TrainingError` | public | move + re-export из `common.errors`, `trainer`, `trainer_params` | stable |
| `InvalidPredictionError` | exception | внутренние проверки метрик, `tests/test_non_finite_predictions.py` | `...common.errors.InvalidPredictionError` | `automl_contracts.errors.InvalidPredictionError` (рекомендуется) | needs-decision; рекомендация — public beta | move + re-export после решения C2/C6 | beta |

`TrainingError` — документированный публичный тип: `ModelTrainer` и
`train_model` обещают его в API_REFERENCE. Его нужно перенести в контракт и
сохранить идентичность класса (`...trainer.TrainingError is
automl_contracts.errors.TrainingError`).

`InvalidPredictionError` наследуется от `TrainingError` и `ValueError`, но не
входит в `__all__` и не имеет документированного пути импорта. При этом он
наблюдаем снаружи: его может поймать потребитель, вызвавший `ModelTrainer.fit`/
`predict`. Рекомендуется сделать его публичным beta и перенести вместе с
`TrainingError`; окончательное решение — за C2/C6. Если он останется internal,
C3 не должен обещать его стабильность, но реэкспорт из `common.errors` следует
сохранить.

`NRMSEZeroRangeError` и прочие исключения из `common.metrics` — runtime-
исключения метрик, не контракт.

### 4.3. Пакет `training_engine/config/` и `config_parser.py`

`config_parser.py` — документированная публичная точка импорта
(`API_REFERENCE.md`, docstring модуля). Пакет `training_engine/config/` — его
внутренняя раскладка; прямой импорт из `training_engine.config.general` и
аналогичных модулей в документации не обещан, но используется тестами.

#### 4.3.1. Корневая модель и чтение конфигурации

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `Config` | Pydantic-модель | корневой `__init__`, `config_parser`, `train_best_model`, `AutoMLExperiment`, тесты, API_REFERENCE | `...training_engine.config_parser.Config`, `configurable_automl_engine.Config` | `automl_contracts.config.Config` | public | move + re-export | stable |
| `read_config` | функция | `config_parser`, `data_prep.load_config`, тесты; в API_REFERENCE путь не документирован | `...training_engine.config_parser.read_config` | остаётся в движке; контракт получает только `Config.model_validate` | public | internal (движок), старый путь сохранить | beta |
| runtime-валидаторы `Config` | методы Pydantic | оркестратор | `...training_engine.config.root` | отдельная runtime-проверка движка | internal | internal | internal |

`read_config` читает YAML через `PyYAML`. `PyYAML` не входит в разрешённые
зависимости контракта, поэтому переносить функцию в `automl-contracts` нельзя
без ADR. В контракте остаётся `Config` и его структурная валидация; чтение
файла — удобство движка.

`Config` содержит три runtime-валидатора, которые нельзя переносить в контракт
как есть:

* `_check_algorithm_dependencies` — проверяет установленный пакет через
  `is_installed` и `ALGO_PACKAGE_MAPPING`;
* `_check_hyperparameter_compatibility` — вызывает
  `models.get_allowed_hyperparameters` (sklearn);
* `_normalize_hyperparameter_types` — вызывает `models.get_param_numeric_type`
  (sklearn).

Их нужно вынести в отдельную функцию/слой движка (например,
`validate_runtime_config(cfg)`), а контрактный `Config` оставить структурной
моделью. Это меняет точку возникновения части ошибок: структурные ошибки
по-прежнему возникают при `model_validate`, runtime-ошибки — при запуске
движка. Поведение публичного `train_best_model` должно сохраниться: движок
вызывает runtime-валидацию сразу после разбора конфигурации, до обучения.

#### 4.3.2. Секция `general`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `GeneralCfg` | Pydantic-модель | `Config`, `config_parser`, тесты, API_REFERENCE | `...training_engine.config.general.GeneralCfg` | `automl_contracts.config.general.GeneralCfg` | public | move; чистые cross-field правила — вместе с моделью | stable |
| `ComparisonMetric` | `TypeAlias = str` | `GeneralCfg`, `config_parser` | `...config.general.ComparisonMetric` | `automl_contracts.config.general.ComparisonMetric` | public | move + re-export | beta |
| `AUTO_HOLDOUT_FRACTION` | `float` | `GeneralCfg`, `config_parser`, тесты | `...config.general.AUTO_HOLDOUT_FRACTION` | `automl_contracts.config.general.AUTO_HOLDOUT_FRACTION` | public | move + re-export | beta |
| `AUTO_HOLDOUT_MIN_ROWS` | `int` | `GeneralCfg`, `config_parser`, тесты | `...config.general.AUTO_HOLDOUT_MIN_ROWS` | `automl_contracts.config.general.AUTO_HOLDOUT_MIN_ROWS` | public | move + re-export | beta |
| `InfPolicy` | `Literal` | `GeneralCfg` | `...config.general.InfPolicy` | `automl_contracts.config.general.InfPolicy` | public | move + re-export | stable |
| `DatetimePolicy` | `Literal` | `GeneralCfg` | `...config.general.DatetimePolicy` | `automl_contracts.config.general.DatetimePolicy` | public | move + re-export | stable |
| `SplitAggregation` | `Literal` | `GeneralCfg` | `...config.general.SplitAggregation` | `automl_contracts.config.general.SplitAggregation` | public | move + re-export | stable |

`GeneralCfg` — это контракт структуры `general`. Все его поля (`comparison_metric`,
`additional_metrics`, `custom_metrics`, `path_to_model`, `serialization_format`,
`log_to_file`, `phases`, `validation_strategy`, `n_folds`, `n_repeats`,
`split_aggregation`, `random_state`, `categorical_encoding`, `parallel_strategy`,
`max_workers`, `parallel_mode`, `start_method`, `feature_selection`,
`preprocessing`, `target_transform`, `target_transform_skew_threshold` и
остальные поля модели) входят в контракт конфигурации. Отдельные строки для
каждого поля не нужны: их контракт — это тип и `Field(...)`-ограничения модели.

Чистые cross-field правила из `general_rules.py` (`check_metric_references`,
`deduplicate_additional_metrics`, `exclude_comparison_from_additional`,
`check_multi_split`, `validate_phase_pipeline`, `check_pruning_consistency`,
`check_high_cardinality_encoding`) переносятся в контрактный слой. Проверки,
требующие `joblib`, `multiprocessing`, `sklearn` или данных (`check_n_folds` в
части `joblib`, `check_start_method`, `check_threads_timeouts`), остаются в
движке.

#### 4.3.3. Секция `phases` и pruning

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `HPOPhaseAction` | `Literal` | `HPOPhaseCfg`, `AlgorithmRun`, тесты | `...config.phases.HPOPhaseAction` | `automl_contracts.config.phases.HPOPhaseAction` | public | move + re-export | stable |
| `SelectStrategy` | `Literal` | `HPOPhaseCfg`, тесты | `...config.phases.SelectStrategy` | `automl_contracts.config.phases.SelectStrategy` | public | move + re-export | stable |
| `SelectOutput` | `Literal` | `HPOPhaseCfg`, тесты | `...config.phases.SelectOutput` | `automl_contracts.config.phases.SelectOutput` | public | move + re-export | stable |
| `PruningCfg` | Pydantic-модель | `GeneralCfg`, тесты | `...config.phases.PruningCfg` | `automl_contracts.config.phases.PruningCfg` | public | move + re-export | beta |
| `PruningStrategy` | enum | `PruningCfg` | `...config.phases.PruningStrategy` | `automl_contracts.config.phases.PruningStrategy` | public | move + re-export | stable |
| `HPOPhaseCfg` | Pydantic-модель | `GeneralCfg`, тесты | `...config.phases.HPOPhaseCfg` | `automl_contracts.config.phases.HPOPhaseCfg` | public | move + re-export | stable |
| `DEFAULT_PHASE_N_TRIALS` | `int` | `config_parser`, тесты | `...config.phases.DEFAULT_PHASE_N_TRIALS` | `automl_contracts.config.phases.DEFAULT_PHASE_N_TRIALS` | public | move + re-export | beta |
| `ERR_REMOVED_REFINE_WINNER` | `str` (код ошибки) | `config_parser`, тесты; документированный migration hint | `...config.phases.ERR_REMOVED_REFINE_WINNER` | `automl_contracts.config.phases.ERR_REMOVED_REFINE_WINNER` | public | move + re-export | stable |
| `_default_phases` | функция | внутренняя реализация `GeneralCfg` | `...config.phases._default_phases` | остаётся внутренней функцией контрактного модуля | internal | move как internal | internal |

`HPOPhaseCfg._reject_legacy_refine_winner` и `_validate_select_fields` — чистые
pydantic-валидаторы, переносятся вместе с моделью. `_default_phases` не
публичен, но его результат (pipeline по умолчанию) входит в контракт поведения
`GeneralCfg`; функция может остаться внутренней в контрактном модуле.

#### 4.3.4. Sanity Gate

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `SanityGateCfg` | Pydantic-модель | `GeneralCfg`, тесты, API_REFERENCE | `...config.sanity_gate.SanityGateCfg` | `automl_contracts.config.sanity_gate.SanityGateCfg` | public | move + re-export | stable |
| `CorridorMode` | `Literal` | `SanityGateCfg`, `select_finalists` | `...config.sanity_gate.CorridorMode` | `automl_contracts.config.sanity_gate.CorridorMode` | public | move + re-export | stable |
| `GeneralizationGapMode` | `Literal` | `SanityGateCfg`, `training_engine/sanity_gate.py` | `...config.sanity_gate.GeneralizationGapMode` | `automl_contracts.config.sanity_gate.GeneralizationGapMode` | public | move + re-export | beta |
| `SanityGateMode` | `Literal` | внутренний `training_engine/state.py`, `finalists.py` | `...config.sanity_gate.SanityGateMode`, реэкспорт `config_parser` | остаётся в движке; реэкспорт из `config_parser` сохранить | internal | internal | internal |
| `ERR_REMOVED_SANITY_GATE_MODE` | `str` (код ошибки) | `config_parser`, тесты; документированный migration hint | `...config.sanity_gate.ERR_REMOVED_SANITY_GATE_MODE` | `automl_contracts.config.sanity_gate.ERR_REMOVED_SANITY_GATE_MODE` | public | move + re-export | stable |

`SanityGateMode` — внутренний режим select-шага, а не поле конфигурации; его
определение оставлено в движке. Это пример сущности, которую нельзя считать
публичной только потому, что она импортируется из `config_parser`.

#### 4.3.5. Complexity guard и ensemble

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `ComplexityGuardCfg` | Pydantic-модель | `GeneralCfg`, тесты, API_REFERENCE | `...config.complexity_guard.ComplexityGuardCfg` | `automl_contracts.config.complexity_guard.ComplexityGuardCfg` | public | move + re-export | stable |
| `ComplexityGuardMode` | enum | `ComplexityGuardCfg` | `...config.complexity_guard.ComplexityGuardMode` | `automl_contracts.config.complexity_guard.ComplexityGuardMode` | public | move + re-export | stable |
| `ComplexityGuardUserRangePolicy` | enum | `ComplexityGuardCfg` | `...config.complexity_guard.ComplexityGuardUserRangePolicy` | `automl_contracts.config.complexity_guard.ComplexityGuardUserRangePolicy` | public | move + re-export | stable |
| `ComplexityGuardSizeBasis` | enum | `ComplexityGuardCfg` | `...config.complexity_guard.ComplexityGuardSizeBasis` | `automl_contracts.config.complexity_guard.ComplexityGuardSizeBasis` | public | move + re-export | stable |
| `EnsembleCfg` | Pydantic-модель | `GeneralCfg`, тесты, API_REFERENCE | `...config.ensemble.EnsembleCfg` | `automl_contracts.config.ensemble.EnsembleCfg` | public | move + re-export | beta |
| `EnsembleWeightMethod` | `Literal` | `EnsembleCfg` | `...config.ensemble.EnsembleWeightMethod` | `automl_contracts.config.ensemble.EnsembleWeightMethod` | public | move + re-export | beta |

`ComplexityGuardCfg` — контракт конфигурации. Не путать с
`hpo.complexity_guard.ComplexityGuardConfig` — это runtime-представление
guard-а, которое остаётся в движке.

#### 4.3.6. Секция `algorithms` и oversampling

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `AlgoCfg` | Pydantic-модель | `Config`, `config_parser`, тесты | `...config.algorithms.AlgoCfg` | `automl_contracts.config.algorithms.AlgoCfg` | public | move; runtime-методы вынести в движок | stable |
| `OversamplingCfg` | Pydantic-модель | `Config`, `config_parser`, тесты | `...config.algorithms.OversamplingCfg` | `automl_contracts.config.algorithms.OversamplingCfg` | public | move + re-export | stable |
| `OversamplingAlgorithm` | enum | `OversamplingCfg` | `...config.algorithms.OversamplingAlgorithm` | `automl_contracts.config.algorithms.OversamplingAlgorithm` | public | move + re-export | stable |
| `AlgorithmsConfig` | динамическая Pydantic-модель | `Config.algorithms`, `config_parser` | `...config.algorithms.AlgorithmsConfig` | `automl_contracts.config.algorithms.AlgorithmsConfig` | public | move после фиксации списка алгоритмов в контракте | beta |
| `AlgorithmsConfigType` | `TypeAlias` | статическая типизация | `...config.algorithms.AlgorithmsConfigType` | внутренний тип контрактного модуля | internal | move как internal | internal |
| `ERR_REMOVED_PLUGIN_KEY` | `str` (код ошибки) | `config_parser`, тесты; migration hint | `...config.algorithms.ERR_REMOVED_PLUGIN_KEY` | `automl_contracts.config.algorithms.ERR_REMOVED_PLUGIN_KEY` | public | move + re-export | stable |
| `REMOVED_ALGORITHM_KEYS` | `tuple[str, ...]` | внутренняя валидация | `...config.algorithms.REMOVED_ALGORITHM_KEYS` | остаётся в движке или internal-модуле контракта | internal | internal | internal |

`AlgoCfg.get_required_package` и `get_unknown_hyperparameters` — runtime-методы:
они обращаются к `ALGO_PACKAGE_MAPPING` и `models.get_allowed_hyperparameters`.
В контрактной модели остаются только поля и структурные валидаторы
(`limit_hyperparameters` без `hyperparameters` и т.п.). Runtime-методы нужно
либо вынести в функцию движка, либо сохранить в движке через отдельный
подкласс/миксин, не создавая второго определения `AlgoCfg`.

`AlgorithmsConfig` создаётся через `create_model` по списку
`AVAILABLE_ALGORITHMS` из `models.py`. Чтобы контракт не импортировал
`models`, канонический список алгоритмов нужно определить в контракте
(см. 4.7), а движок — импортировать его оттуда и реэкспортировать как
`configurable_automl_engine.models.AVAILABLE_ALGORITHMS`.

#### 4.3.7. Cross-field правила `general_rules.py`

| Символ | Тип | Решение | Где остаётся |
|---|---|---|---|
| `check_custom_metric_names` | функция | move в контракт | чистая проверка имён; runtime-проверка callable — в движке |
| `check_metric_references` | функция | move в контракт | использует контрактный набор встроенных метрик + имена `custom_metrics` |
| `deduplicate_additional_metrics` | функция | move в контракт | чистая функция |
| `exclude_comparison_from_additional` | функция | move в контракт | нужен контрактный alias-map `to_sklearn_name` |
| `check_n_folds` | функция | split | структурная часть (`n_folds >= 1`, `k_fold` требует `>= 2`) — контракт; проверка `joblib` — движок |
| `check_multi_split` | функция | move в контракт | бюджетный guard не требует runtime-зависимостей |
| `check_pruning_consistency` | функция | move в контракт | чистая проверка |
| `validate_phase_pipeline` | функция | move в контракт | чистая проверка |
| `check_high_cardinality_encoding` | функция | move в контракт | чистая проверка |
| `check_start_method` | функция | internal | зависит от платформы/окружения |
| `check_threads_timeouts` | функция | internal | зависит от `multiprocessing`/ресурсов |
| `MULTI_SPLIT_BUDGET_*`, `MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT` | константы | move как internal-константы контракта | значения — часть контрактной валидации, но не публичный API |

#### 4.3.8. JSON Schema

| Сущность | Тип | Потребители | Текущее состояние | Предлагаемое состояние | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `config.schema.json` | JSON Schema (производный артефакт) | API_REFERENCE, внешние валидаторы, CI | генерируется из `Config.model_json_schema()` | генерируется из контрактной `Config`; включается в wheel `automl-contracts` | public | move (как артефакт контракта) | beta |
| `schema_generator.py` | скрипт | разработка/CI | импортирует `config_parser.Config` | импортирует контрактную `Config`; остаётся инструментом, не публичным API | internal | re-export/internal | internal |

Схема — не независимый источник истины. C3/C4 должны генерировать её из
контрактных моделей и проверять в CI, что сохранённый файл совпадает с
результатом генерации. В корне репозитория `config.schema.json` может остаться
как совместимый артефакт, но он не должен редактироваться вручную отдельно от
контрактной схемы.

Дополнительно: текущий `schema_generator.py` вызывает `save_schema()` на уровне
модуля, то есть запись файла происходит уже при импорте. Для C3 это нельзя
сохранять как публичное поведение: генератор должен остаться скриптом/CI-шагом,
который вызывается явно, и не иметь побочных эффектов при импорте.

### 4.4. `hpo/default_spaces.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `SearchSpaceEntry` | Pydantic-модель | `AlgoCfg.hyperparameters`, `optimize(space_overrides=...)`, тесты, API_REFERENCE | `...hpo.default_spaces.SearchSpaceEntry` | `automl_contracts.search_space.SearchSpaceEntry` | public | move + re-export | stable |
| `BaseDistribution` | Pydantic-модель | типовая база distributions | `...hpo.default_spaces.BaseDistribution` | `automl_contracts.search_space.BaseDistribution` | public | move + re-export | beta |
| `CategoricalSpace` | Pydantic-модель | `SearchSpaceEntry`, тесты | `...hpo.default_spaces.CategoricalSpace` | `automl_contracts.search_space.CategoricalSpace` | public | move + re-export | beta |
| `NumericSpace` | Pydantic-модель | база `FloatSpace`/`IntSpace` | `...hpo.default_spaces.NumericSpace` | `automl_contracts.search_space.NumericSpace` | public | move + re-export | beta |
| `FloatSpace` | Pydantic-модель | `SearchSpaceEntry`, тесты | `...hpo.default_spaces.FloatSpace` | `automl_contracts.search_space.FloatSpace` | public | move + re-export | beta |
| `IntSpace` | Pydantic-модель | `SearchSpaceEntry`, тесты | `...hpo.default_spaces.IntSpace` | `automl_contracts.search_space.IntSpace` | public | move + re-export | beta |
| `DEFAULT_SPACES` | `dict[str, dict[str, SearchSpaceEntry]]` | `tuner.optimize`, бенчмарки, тесты, API_REFERENCE | `...hpo.default_spaces.DEFAULT_SPACES` | остаётся в движке | public | internal | beta |
| `GLM_COMMON` | `dict` | построение `DEFAULT_SPACES` | `...hpo.default_spaces.GLM_COMMON` | остаётся в движке | internal | internal | internal |
| `ALGO_HYPERPARAMETER_REGISTRY` | `dict` | информационная проекция | `...hpo.default_spaces.ALGO_HYPERPARAMETER_REGISTRY` | остаётся в движке | internal | internal | internal |
| `DATA_DEPENDENT_CONSTRAINTS` | `dict` | clipping search space | `...hpo.default_spaces.DATA_DEPENDENT_CONSTRAINTS` | остаётся в движке | internal | internal | internal |
| `SVR_EPSILON_*` | константы | адаптивные bounds SVR | `...hpo.default_spaces` | остаются в движке | internal | internal | internal |
| `align_high_to_step`, `align_entry_bounds` | функции | clipping/guard | `...hpo.default_spaces` | остаются в движке | internal | internal | internal |
| `compute_svr_epsilon_bounds` | функция | HPO | `...hpo.default_spaces` | остаётся в движке | internal | internal | internal |
| `clip_search_space` | функция | HPO | `...hpo.default_spaces` | остаётся в движке | internal | internal | internal |
| `SCALE_PARAMS` | `frozenset` | типизация диапазонов | `...hpo.default_spaces` | остаётся в движке | internal | internal | internal |
| `infer_param_numeric_type`, `normalize_hyperparameter_entry`, `resolve_implicit_range_type` | функции | runtime-типизация по sklearn | `...hpo.default_spaces` | остаются в движке | internal | internal | internal |

`SearchSpaceEntry` и distribution-модели — часть публичного DSL конфигурации:
пользователь задаёт `hyperparameters` списками/словарями, а `SearchSpaceEntry`
является канонической формой. `DEFAULT_SPACES` — публичный **реестр движка**
(документированный), но не контракт данных: он описывает стратегию поиска
конкретных алгоритмов и меняется вместе с движком. Переносить его в контракт
не следует.

### 4.5. `feature_selection.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `FeatureSelectionMode` | enum | `FeatureSelectionCfg`, `GeneralCfg`, тесты | `...feature_selection.FeatureSelectionMode` | `automl_contracts.feature_selection.FeatureSelectionMode` | public | move + re-export | stable |
| `FeatureSelectionMethod` | enum | `FeatureSelectionCfg`, тесты | `...feature_selection.FeatureSelectionMethod` | `automl_contracts.feature_selection.FeatureSelectionMethod` | public | move + re-export | stable |
| `FeatureSelectionCfg` | Pydantic-модель | `GeneralCfg`, `ModelTrainer`, `optimize`, тесты, API_REFERENCE | `...feature_selection.FeatureSelectionCfg` | `automl_contracts.feature_selection.FeatureSelectionCfg` | public | move + re-export | stable |
| `FeatureSelector` | sklearn transformer | `ModelTrainer`, HPO, тесты | `...feature_selection.FeatureSelector` | остаётся в движке | internal | internal | internal |
| `_VALID_METHODS`, `_TARGET_METHODS`, `_top_indices`, `_compute_variances`, `_validate_sparse_mutual_info` | детали реализации | `FeatureSelector` | `...feature_selection` | остаются в движке | internal | internal | internal |

Модуль сейчас смешивает контрактную модель и sklearn-реализацию. C3 должен
разделить его: контрактный модуль содержит только `FeatureSelectionCfg` и
перечисления; движковый модуль импортирует их из контракта, определяет
`FeatureSelector` и реэкспортирует старые имена. Идентичность
`...feature_selection.FeatureSelectionCfg is
automl_contracts.feature_selection.FeatureSelectionCfg` обязательна.

### 4.6. `preprocessing_presets.py` и `outliers_presets.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `ImputationStrategy` | `Literal` | `PreprocessingPreset`, `PreprocessingOverride` | `...preprocessing_presets.ImputationStrategy` | `automl_contracts.preprocessing.ImputationStrategy` | public | move + re-export | stable |
| `ScalingType` | `Literal` (deprecated alias) | старые импорты | `...preprocessing_presets.ScalingType` | `automl_contracts.preprocessing.ScalingType` | deprecated | re-export; пометить deprecated | beta |
| `StandardScaling` | Pydantic-модель | `ScalingSpec`, `PreprocessingPreset`, тесты | `...preprocessing_presets.StandardScaling` | `automl_contracts.preprocessing.StandardScaling` | public | move + re-export | stable |
| `RobustScaling` | Pydantic-модель | `ScalingSpec`, `PreprocessingPreset`, тесты | `...preprocessing_presets.RobustScaling` | `automl_contracts.preprocessing.RobustScaling` | public | move + re-export | stable |
| `QuantileScaling` | Pydantic-модель | `ScalingSpec`, `PreprocessingPreset`, тесты | `...preprocessing_presets.QuantileScaling` | `automl_contracts.preprocessing.QuantileScaling` | public | move + re-export | stable |
| `NoScaling` | Pydantic-модель | `ScalingSpec`, `PreprocessingPreset`, тесты | `...preprocessing_presets.NoScaling` | `automl_contracts.preprocessing.NoScaling` | public | move + re-export | stable |
| `ObjectScalingSpec`, `StringScaling`, `ScalingSpec` | `TypeAlias` | поля `scaling`, тесты, JSON Schema | `...preprocessing_presets` | `automl_contracts.preprocessing` | public | move + re-export | beta |
| `coerce_scaling`, `scaling_type` | функции | внутренние helper-ы и тесты | `...preprocessing_presets` | остаются internal-функциями контрактного модуля или движка | internal | move как internal | internal |
| `AlgorithmClass` | enum | `PreprocessingPreset`, тесты | `...preprocessing_presets.AlgorithmClass` | `automl_contracts.preprocessing.AlgorithmClass` | public | move + re-export | stable |
| `PreprocessingPreset` | Pydantic-модель | `ModelTrainer.preprocessing_preset_`, тесты, API_REFERENCE | `...preprocessing_presets.PreprocessingPreset` | `automl_contracts.preprocessing.PreprocessingPreset` | public | move + re-export | stable |
| `PreprocessingOverride` | Pydantic-модель | `GeneralCfg`, `AlgoCfg`, `ModelTrainer`, `optimize`, тесты | `...preprocessing_presets.PreprocessingOverride` | `automl_contracts.preprocessing.PreprocessingOverride` | public | move + re-export | stable |
| `ALGORITHM_CLASS_MAPPING` | `dict` | движок, тесты, документация | `...preprocessing_presets.ALGORITHM_CLASS_MAPPING` | остаётся в движке | public | internal | beta |
| `PRESET_BY_CLASS` | `dict` | движок, тесты | `...preprocessing_presets.PRESET_BY_CLASS` | остаётся в движке | public | internal | beta |
| `normalize_algorithm` | функция | движок, тесты | `...preprocessing_presets.normalize_algorithm` | остаётся в движке | internal | internal | internal |
| `merge_preprocessing_overrides` | функция | документированный helper слияния override-ов | `...preprocessing_presets.merge_preprocessing_overrides` | `automl_contracts.preprocessing.merge_preprocessing_overrides` | public | move + re-export | beta |
| `resolve_preprocessing_preset` | функция | движок, тесты | `...preprocessing_presets.resolve_preprocessing_preset` | остаётся в движке | internal | internal | internal |
| `requires_missing_indicator` | функция | движок | `...preprocessing_presets.requires_missing_indicator` | остаётся в движке | internal | internal | internal |
| `log_tree_class_scaling_ineffective` | функция | движок | `...preprocessing_presets.log_tree_class_scaling_ineffective` | остаётся в движке | internal | internal | internal |
| `NoOutliers` | Pydantic-модель | `OutlierSpec`, `PreprocessingPreset`, тесты | `...outliers_presets.NoOutliers` | `automl_contracts.preprocessing.NoOutliers` | public | move + re-export | stable |
| `QuantileOutliers` | Pydantic-модель | `OutlierSpec`, `PreprocessingPreset`, тесты | `...outliers_presets.QuantileOutliers` | `automl_contracts.preprocessing.QuantileOutliers` | public | move + re-export | stable |
| `IqrOutliers` | Pydantic-модель | `OutlierSpec`, `PreprocessingPreset`, тесты | `...outliers_presets.IqrOutliers` | `automl_contracts.preprocessing.IqrOutliers` | public | move + re-export | stable |
| `ObjectOutlierSpec`, `StringOutlierSpec`, `OutlierSpec` | `TypeAlias` | поля `outliers`, тесты, JSON Schema | `...outliers_presets` | `automl_contracts.preprocessing` | public | move + re-export | beta |
| `NO_OUTLIERS` | `NoOutliers` | default `PreprocessingPreset.outliers` | `...outliers_presets.NO_OUTLIERS` | `automl_contracts.preprocessing.NO_OUTLIERS` | public | move + re-export | beta |
| `coerce_outliers`, `format_outliers` | функции | внутренние helper-ы, `PreprocessingPreset.__str__` | `...outliers_presets` | `automl_contracts.preprocessing` (internal/public beta) | internal | move как internal | internal |

`preprocessing_presets.py` импортирует `models` (`AVAILABLE_ALGORITHMS`,
`resolve_algorithm_name`) — это runtime-зависимость. Для контракта нужно
отделить модели `ScalingSpec`/`OutlierSpec`/`PreprocessingOverride`/
`PreprocessingPreset` от таблиц и функций выбора пресета. `PRESET_BY_CLASS`
описывает поведение движка, а не формат данных, поэтому в контракт не входит.

`ScalingType` сохраняется как deprecated alias: у него нет отдельной
семантики, но старые импорты не должны ломаться.

### 4.7. `models.py` и списки алгоритмов

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `AVAILABLE_ALGORITHMS` | `list[str]` | `AlgorithmsConfig`, `preprocessing_presets`, тесты, scripts, API_REFERENCE (алгоритмы) | `configurable_automl_engine.models.AVAILABLE_ALGORITHMS` | `automl_contracts.definitions.AVAILABLE_ALGORITHMS`; в `models` — реэкспорт | public | move canonical list + re-export | beta |
| `create_model` | функция | публичный API движка, тесты, API_REFERENCE | `...models.create_model` | остаётся в движке | public | internal | stable |
| `get_allowed_hyperparameters`, `get_param_numeric_type`, `clean_hyperparameters`, `remap_legacy_param_names`, `resolve_algorithm_name` | функции | runtime-валидация/HPO | `...models` | остаются в движке | internal | internal | internal |
| `LEGACY_PARAM_MAPPINGS` | `dict` | runtime-валидация | `...models.LEGACY_PARAM_MAPPINGS` | остаётся в движке | internal | internal | internal |
| `MODEL_RANDOM_STATE`, `SPARSE_UNSUPPORTED_ALGORITHMS`, `GPR_KERNEL_TYPES`, `_FACTORY`, `_get_factory` | константы/фабрика | движок | `...models` | остаются в движке | internal | internal | internal |

`AVAILABLE_ALGORITHMS` — де-факто публичный список: он определяет допустимые
ключи `algorithms` и используется тестами/скриптами. Его надо определить в
контракте независимо, без импорта `models` (иначе контракт потянет sklearn).
Движок должен импортировать этот список из контракта, а не дублировать.
Тип (`list` или `tuple`) и вопрос мутабельности нужно решить в C2; для
совместимости на первом этапе рекомендуется оставить `list`, но не давать
потребителям менять его как источник истины.

### 4.8. `common/metrics.py`, `common/custom_metrics.py`

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `AVAILABLE_METRICS` | `list[str]` | `GeneralCfg` (описание поля), тесты, API_REFERENCE (allowed values) | `...common.metrics.AVAILABLE_METRICS` | `automl_contracts.metrics.AVAILABLE_METRICS` (независимый статический список) | public | move + re-export | beta |
| `to_sklearn_name` | функция (alias-map) | `general_rules`, `ExperimentResult`, тесты | `...common.metrics.to_sklearn_name` | `automl_contracts.metrics.to_sklearn_name` (internal helper контракта) | internal | move + re-export старого пути | internal |
| `user_direction`, `is_error_metric`, `is_greater_better`, `get_scorer_object` | функции | движок, тесты | `...common.metrics` | остаются в движке | internal | internal | internal |
| `reserved_metric_names` | функция | runtime-валидация custom metrics | `...common.metrics` | остаётся в движке | internal | internal | internal |
| `CustomMetricKind` | `Literal` | `CustomMetricSpec` | `...common.custom_metrics.CustomMetricKind` | `automl_contracts.custom_metrics.CustomMetricKind` | public | move + re-export | beta |
| `CustomMetricSpec` | Pydantic-модель | корневой `__init__`, `GeneralCfg`, `optimize`, API_REFERENCE | `...common.custom_metrics.CustomMetricSpec`, `configurable_automl_engine.CustomMetricSpec` | `automl_contracts.custom_metrics.CustomMetricSpec` | public | needs-decision: рекомендуется move полей/структуры + runtime-валидация в движке | beta |
| `ERR_IMPORT`, `ERR_NOT_CALLABLE`, `ERR_BAD_SIGNATURE`, `ERR_DIRECTION_REQUIRED`, `ERR_DIRECTION_CONFLICT`, `ERR_KWARGS_UNSUPPORTED` | коды ошибок | pydantic-ошибки custom metrics | `...common.custom_metrics` | остаются в движке | internal | internal | internal |
| `ERR_CUSTOM_NAME_INVALID` | `str` (код ошибки) | структурная проверка имени custom metric | `...common.metrics.ERR_CUSTOM_NAME_INVALID` | `automl_contracts.custom_metrics.ERR_CUSTOM_NAME_INVALID` | public | move + re-export | stable |
| `ERR_CUSTOM_NAME_COLLISION` | `str` (код ошибки) | runtime-проверка коллизий со sklearn | `...common.metrics.ERR_CUSTOM_NAME_COLLISION` | остаётся в движке | internal | internal | internal |
| `HPO_WORST_SCORE` | `float` | `tuner`, HPO, тесты, API_REFERENCE | `...hpo.types.HPO_WORST_SCORE`, реэкспорт `...tuner` | остаётся в движке | public | internal | stable |
| `WORST_SCORE_THRESHOLD` | `float` | `ExperimentResult`, HPO, тесты | `...common.definitions` | `automl_contracts.definitions` | public | move + re-export | beta |

`AVAILABLE_METRICS` сейчас вычисляется из `_SCORER_OBJECTS`, которые содержат
`make_scorer` из sklearn. Для контракта нужен независимый статический набор
встроенных имён:

`nrmse`, `rmse`, `neg_root_mean_squared_error`, `mae`, `mse`, `r2`.

`global_nrmse` поддерживается движком как специальный динамический scorer, но
не входит в `AVAILABLE_METRICS`; это нужно явно зафиксировать в C2/C6 и не
делать вид, что множество встроенных метрик автоматически совпадает с
`sklearn.metrics.get_scorer_names()`.

`CustomMetricSpec` — публичная декларация (корневой экспорт). В контракте
остаются поля `path`, `kind`, `greater_is_better`, `additive`, `kwargs` и
структурные проверки. Импорт callable, проверка сигнатуры, разрешение
направления по sklearn scorer-у и проверка коллизий имён со sklearn — runtime-
валидация движка. Старый `CustomMetricSpec` должен продолжать проверять
callable при создании в движке; в контракте эта проверка недоступна.

### 4.9. DTO результатов

| Символ | Тип | Известные потребители | Текущий путь | Новый путь | Статус | Решение | Стабильность |
|---|---|---|---|---|---|---|---|
| `AlgorithmRun` | Pydantic-модель | `ExperimentResult`, reporters, tests | `...experiment.result.AlgorithmRun`, `...experiment.AlgorithmRun`, корневой `__init__` | `automl_contracts.result.AlgorithmRun` | public | move + re-export | stable |
| `ExperimentResult` | Pydantic-модель | reporters, store, leaderboard, export, `AutoMLExperiment`, tests, API_REFERENCE | `...experiment.result.ExperimentResult`, `...experiment.ExperimentResult`, корневой `__init__` | `automl_contracts.result.ExperimentResult` | public | move после отделения numpy/sklearn-валидации + re-export | stable |
| `ScoreValue`, `ScoreValueOrNone` | `TypeAlias` | поля `AlgorithmRun`/`ExperimentResult` | `...experiment.result` | `automl_contracts.result` (internal type aliases) | internal | move как internal | internal |
| `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection` | enums | см. 4.1 | `...common.definitions`, `...experiment` | `automl_contracts.definitions` | public | move + re-export | stable |

`ExperimentResult` — самый важный публичный DTO для потребителей отчётов и
store. Сейчас модуль импортирует `numpy` и `common.metrics` (sklearn). Для
переноса в контракт нужно:

1. убрать `numpy` из валидаторов, принимая только стандартные числа;
   нормализацию numpy-скаляров выполнять в движке на границе;
2. заменить `reserved_metric_names()`/`user_direction()` на статическую
   контрактную таблицу направлений для встроенных метрик и не пытаться
   разрешать произвольные sklearn-имена в контракте;
3. для custom metrics не обращаться к process-local registry — направление
   уже приходит в payload (`metric_direction`), как и описано в docstring.

Это сохранит JSON-совместимость и сериализацию, но изменит место проверки
некоторых инвариантов: runtime-проверки метрик останутся в движке. C5 должен
проверить round-trip `ExperimentResult.to_json()`/`from_json()` и идентичность
полей.

`RunSummary`, `LeaderboardEntry`, `FeatureImportance`, `EnsembleMember`,
`ReporterArtifact` — тоже DTO, но они принадлежат engine/experiment API.
Включать их в `automl_contracts` в C3 не обязательно; если C6 решит сделать
их публичными стабильными, это отдельное расширение. Сейчас они классифицируются
как `internal`/`beta` движка и не входят в минимальный контракт.

### 4.10. Документация, тесты, примеры и скрипты

Фактические публичные пути, найденные в документации:

* `README.md` и `example.py` используют только `import configurable_automl_engine
  as caml` и `caml.train_best_model(...)`;
* `API_REFERENCE.md` документирует `from configurable_automl_engine import
  train_best_model`, `from configurable_automl_engine import ModelTrainer`,
  `from configurable_automl_engine import CustomMetricSpec` (через
  `configurable_automl_engine.CustomMetricSpec`), `from
  configurable_automl_engine.tuner import optimize`,
  `from configurable_automl_engine.models import create_model`,
  `from configurable_automl_engine.trainer import train_model`,
  `from configurable_automl_engine.training_engine.thread_pool import
  run_parallel`, `from configurable_automl_engine.experiment import ...`,
  `from configurable_automl_engine.oversampling import DataOversampler`;
* `Config` документирован как принимаемый объект (`config=Config`), но не как
  отдельный документированный путь импорта; фактически он экспортируется из
  корня и из `training_engine.config_parser`;
* конфигурационные модели документированы по полям и секциям, а не по путям
  импорта; `config_parser` — единственный явно названный публичный модуль
  конфигурации.

Фактические потребители внутри репозитория:

* `tests/` импортируют `Config`, `GeneralCfg`, `AlgoCfg`, `HPOPhaseCfg`,
  `ValidationStrategy`, `SerializationFormat`, `TargetTransform`,
  `MetricDirection`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`,
  `SearchSpaceEntry`, `FeatureSelectionCfg`, `PreprocessingOverride`,
  `TrainingError`, `CustomMetricSpec`, `ExperimentResult`, `AlgorithmRun` и др.;
* `scripts/` и `benchmarks/` используют `models.AVAILABLE_ALGORITHMS`,
  `models.create_model`, `hpo.default_spaces.DEFAULT_SPACES`,
  `hpo.complexity_guard` и runtime-модули; это engine-потребители, а не
  контракт;
* `tests/` также импортируют runtime-функции (`build_preprocessor`,
  `iter_splits`, `run_optimization`, `prepare_dataset` и т.п.) — они остаются
  в движке и не должны попадать в `automl_contracts` только из-за тестов.

Отдельно зафиксировано: `tests/test_experiment_result.py` уже проверяет
идентичность реэкспортов статусов из `experiment` и
`common.definitions`; C5 должен расширить такие проверки на все контрактные
DTO/enum-ы и на старые пути из таблицы 5.1.

### 4.11. Идентичность классов и runtime-поведение

Реэкспорт — это не только успешный импорт. Для C3 обязательны проверки
идентичности (`is`) для:

* `Config`, `GeneralCfg`, `AlgoCfg`, `HPOPhaseCfg`, `SanityGateCfg`,
  `ComplexityGuardCfg`, `EnsembleCfg`, `OversamplingCfg`;
* `FeatureSelectionCfg`, `PreprocessingOverride`, `PreprocessingPreset`,
  `SearchSpaceEntry`, `CustomMetricSpec`;
* `ExperimentResult`, `AlgorithmRun`;
* всех enum-ов и `TrainingError`/`InvalidPredictionError`.

При этом нужно отдельно проверить поведение:

* `Config(...)` в контракте валидирует структуру, а runtime-проверки
  (backend, гиперпараметры) выполняет движок при запуске;
* `CustomMetricSpec(...)` в контракте валидирует структуру, а импорт callable и
  проверку сигнатуры выполняет движок на входе в `train_best_model`/`optimize`;
* `ExperimentResult(...)` принимает стандартные числа; numpy-скаляры
  нормализуются движком до передачи в модель.

Если C2 решит сохранить старый конструктор `CustomMetricSpec` с немедленной
проверкой callable, это потребует отдельного engine-класса/обёртки и явного
решения о том, что именно реэкспортируется. Нельзя одновременно обещать
идентичность класса и разное поведение конструктора без такого решения.

## 5. Матрица обратной совместимости путей импорта

Правило: **старый публичный путь сохраняется через реэкспорт; повторного
определения класса/перечисления/исключения быть не должно.** C3 обязан
проверять не только успешность импорта, но и идентичность (`is`) для классов,
enum-ов и исключений. Для type aliases и констант проверяется равенство
значений/типов.

### 5.1. Документированные и корневые пути

| Старый путь | Символы | Статус пути | Действие C3 | Проверка |
|---|---|---|---|---|
| `configurable_automl_engine` | `AlgorithmRun`, `AutoMLExperiment`, `Config`, `CustomMetricSpec`, `ExperimentResult`, `ModelTrainer`, `__version__`, `train_best_model` | public, documented/root `__all__` | сохранить; контрактные DTO реэкспортировать | импорт + `is` для `Config`, `AlgorithmRun`, `ExperimentResult`, `CustomMetricSpec` |
| `configurable_automl_engine.common.definitions` | `ValidationStrategy`, `EncodingStrategy`, `SerializationFormat`, `TargetTransform`, `ParallelStrategy`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection`, `WORST_SCORE_THRESHOLD` | public, de facto | реэкспорт из контракта; `ALGO_PACKAGE_MAPPING` оставить в движке | импорт + `is` для enum-ов; значение `WORST_SCORE_THRESHOLD` |
| `configurable_automl_engine.common.errors` | `TrainingError`, `InvalidPredictionError` | public для `TrainingError`; для второго — по решению C2 | реэкспорт из контракта (если `InvalidPredictionError` признан публичным) | `is` для `TrainingError`; импорт для второго |
| `configurable_automl_engine.training_engine.config_parser` | все имена из `__all__`, а также `noqa`-реэкспорты: `AlgorithmsConfig`, `OversamplingCfg`, `OversamplingAlgorithm`, `GeneralCfg`, `PruningCfg`, `PruningStrategy`, `SearchSpaceEntry`, `AUTO_HOLDOUT_*`, `ComparisonMetric`, `InfPolicy`, `DatetimePolicy`, `DEFAULT_PHASE_N_TRIALS`, `ERR_*`, `SanityGateMode` | public, documented (модуль) | сохранить все перечисленные имена; контрактные — реэкспорт, runtime — оставить | импорт каждого имени + `is` для моделей/enum-ов; `read_config` продолжает работать |
| `configurable_automl_engine.feature_selection` | `FeatureSelectionCfg`, `FeatureSelectionMethod`, `FeatureSelectionMode`, `FeatureSelector` | public для config-типов; `FeatureSelector` — engine API | config-типы реэкспортировать из контракта, `FeatureSelector` оставить | `is` для config-типов |
| `configurable_automl_engine.preprocessing_presets` | `ImputationStrategy`, `ScalingType`, `StandardScaling`, `RobustScaling`, `QuantileScaling`, `NoScaling`, `ObjectScalingSpec`, `StringScaling`, `ScalingSpec`, `AlgorithmClass`, `PreprocessingPreset`, `PreprocessingOverride`, `ALGORITHM_CLASS_MAPPING`, `PRESET_BY_CLASS`, `merge_preprocessing_overrides`, `resolve_preprocessing_preset`, `coerce_scaling`, `scaling_type`, `normalize_algorithm`, `requires_missing_indicator`, а также реэкспорты из `outliers_presets` | смешанный: config-типы public, runtime-helper-ы internal | config-типы и `merge_preprocessing_overrides` — реэкспорт из контракта; runtime — оставить; deprecated `ScalingType` — сохранить | `is` для моделей/enum-ов; импорт helper-ов |
| `configurable_automl_engine.outliers_presets` | `NoOutliers`, `QuantileOutliers`, `IqrOutliers`, `ObjectOutlierSpec`, `OutlierSpec`, `StringOutlierSpec`, `NO_OUTLIERS`, `coerce_outliers`, `format_outliers` | public для spec-моделей, helper-ы — beta/internal | реэкспорт из контракта для spec-моделей; helper-ы сохранить | `is` для моделей; импорт helper-ов |
| `configurable_automl_engine.hpo.default_spaces` | `SearchSpaceEntry`, `BaseDistribution`, `CategoricalSpace`, `NumericSpace`, `FloatSpace`, `IntSpace`, `DEFAULT_SPACES`, `align_*`, `compute_svr_epsilon_bounds`, `clip_search_space`, `normalize_hyperparameter_entry`, `infer_param_numeric_type` | `SearchSpaceEntry`/distribution — public beta; `DEFAULT_SPACES` — engine API | distribution-модели реэкспортировать из контракта; runtime-функции и `DEFAULT_SPACES` оставить | `is` для моделей; импорт `DEFAULT_SPACES` |
| `configurable_automl_engine.experiment.result` | `AlgorithmRun`, `ExperimentResult`, `MetricDirection`, `RunPhaseAction`, `RunStatus`, `RunStepStatus`, `ScoreValue`, `ScoreValueOrNone` | public для DTO/enum-ов; `ScoreValue*` — internal | DTO/enum-ы реэкспортировать из контракта; aliases оставить | `is` для DTO/enum-ов; JSON round-trip |
| `configurable_automl_engine.experiment` | `AlgorithmRun`, `ExperimentResult`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection`, reporters/store/leaderboard/export/explain | DTO/enum-ы public; остальное — engine API | DTO/enum-ы реэкспортировать; остальное сохранить | `is` для DTO/enum-ов |
| `configurable_automl_engine.models` | `AVAILABLE_ALGORITHMS`, `create_model`; также runtime-функции HPO | `AVAILABLE_ALGORITHMS` public beta; `create_model` public engine API | `AVAILABLE_ALGORITHMS` — реэкспорт из контракта; `create_model` и runtime — оставить | значение/тип списка; импорт `create_model` |
| `configurable_automl_engine.common.metrics` | `AVAILABLE_METRICS`, `to_sklearn_name`, `ERR_CUSTOM_NAME_INVALID`, `ERR_CUSTOM_NAME_COLLISION`, `get_scorer_object`, `oof_rmse`, `user_direction`, `to_user_value` и др. | `AVAILABLE_METRICS`/`to_sklearn_name`/`ERR_CUSTOM_NAME_INVALID` — контрактные; остальное — engine/internal | контрактные имена реэкспортировать; runtime-метрики оставить | равенство значений `AVAILABLE_METRICS`; импорт runtime-функций и кодов |
| `configurable_automl_engine.common.custom_metrics` | `CustomMetricSpec`, `CustomMetricKind`, helpers | `CustomMetricSpec` public; runtime helpers internal | модель/kind реэкспортировать из контракта; runtime-валидацию оставить в движке | `is` для `CustomMetricSpec`; проверка callable в движке |
| `configurable_automl_engine.trainer` | `ModelTrainer`, `TrainingError`, `train_model` | public engine API | `TrainingError` реэкспортировать из контракта; остальное сохранить | `is` для `TrainingError` |
| `configurable_automl_engine.tuner` | `optimize`, `HPO_WORST_SCORE`, `WORST_SCORE_THRESHOLD`, `HyperoptError`, `InvalidAlgorithmError`, `InvalidDataError`, `HPOHistoryError`, `OptimizeResult`, `TrialStats` | public engine API | `WORST_SCORE_THRESHOLD` реэкспортировать из контракта; остальное сохранить | импорт; значение threshold |
| `configurable_automl_engine.preprocessing` | `EncodingStrategy`, runtime-трансформеры | `EncodingStrategy` public; runtime — internal | `EncodingStrategy` реэкспортировать из контракта | `is` для enum/alias |
| `configurable_automl_engine.training_engine` | `train_best_model` | public root/engine path | сохранить | импорт |

### 5.2. Прямые импорты из подмодулей конфигурации

Следующие пути не документированы как публичные (docstring `config/__init__.py`
называет публичным только `config_parser`), но используются тестами:

* `configurable_automl_engine.training_engine.config.general.GeneralCfg`;
* `configurable_automl_engine.training_engine.config.algorithms.AlgoCfg`;
* `configurable_automl_engine.training_engine.config.phases.HPOPhaseCfg`;
* `configurable_automl_engine.training_engine.config.sanity_gate.SanityGateCfg`;
* аналогичные прямые импорты секционных моделей.

Решение C1: считать их **de facto тестовыми, но не публичными** путями.
C3 должен либо сохранить их через тонкие реэкспортные шимы, либо обновить
тесты и явно обосновать изменение. Поскольку стоимость шимов мала, рекомендуется
сохранить их на переходный период и пометить как internal/deprecated в
документации C6. В матрицу публичной совместимости они не входят, но C3 не
должен ломать их молча.

### 5.3. Правило изменения путей

1. Старый путь из таблицы 5.1 остаётся рабочим.
2. C3 не создаёт второй класс с тем же именем: только реэкспорт одного
   определения из `automl_contracts`.
3. Для каждого реэкспорта C5 добавляет проверку идентичности классов/enum-ов.
4. Несовместимое изменение (удаление пути, изменение типа/семантики) возможно
   только с явным обоснованием в ADR/CHANGELOG и повышением версии по политике
   эпика.
5. `read_config` и HPO/runtime-функции не обязаны появляться в `automl_contracts`;
   их старые пути сохраняются в движке.

### 5.4. Документированные engine-пути вне контракта

Эти пути не входят в `automl_contracts`, но являются публичным API движка и
не должны ломаться при C3:

| Старый путь | Символы | Действие C3 |
|---|---|---|
| `configurable_automl_engine.tuner` | `optimize`, `OptimizeResult`, `TrialStats`, HPO-исключения, `HPO_WORST_SCORE`, `WORST_SCORE_THRESHOLD` | сохранить; `WORST_SCORE_THRESHOLD` реэкспортировать из контракта |
| `configurable_automl_engine.models` | `create_model`, `AVAILABLE_ALGORITHMS` | `create_model` сохранить; `AVAILABLE_ALGORITHMS` реэкспортировать |
| `configurable_automl_engine.trainer` | `ModelTrainer`, `train_model`, `TrainingError` | сохранить; `TrainingError` реэкспортировать |
| `configurable_automl_engine.training_engine.thread_pool` | `run_parallel`, `run_parallel_outcomes` | сохранить |
| `configurable_automl_engine.training_engine.component` | `train_best_model`, `last_experiment_result`, `execute_phases` | сохранить |
| `configurable_automl_engine.experiment` | reporters, store, leaderboard, export, explain, `AutoMLExperiment` | сохранить; контрактные DTO/enum-ы реэкспортировать |
| `configurable_automl_engine.experiment.explain` | `explain_permutation_importance`, `explain_shap`, `aggregate_to_original_features`, `render_importance_markdown` | сохранить |
| `configurable_automl_engine.experiment.store` | `ExperimentStore`, `RunSummary`, `DuplicateRunError`, `ExperimentStoreError` | сохранить |
| `configurable_automl_engine.oversampling` | `DataOversampler`, `oversample` и публичные имена пакета | сохранить |
| `configurable_automl_engine.preprocessing` | `EncodingStrategy` и публичные transformer-ы пакета | `EncodingStrategy` реэкспортировать; runtime-классы сохранить |

## 6. Граница контрактной и runtime-валидации

### 6.1. Контрактная валидация

В `automl_contracts` переносятся проверки, которые не требуют runtime-окружения:

* обязательные поля и типы Pydantic-моделей;
* `Field(ge=..., gt=..., le=..., lt=...)`-ограничения конфигурации;
* допустимые значения enum/`Literal` (структурное отклонение неизвестных
  значений);
* discriminated unions `SearchSpaceEntry` (`categorical` / `int` / `float` /
  `float_log`), `ScalingSpec`, `OutlierSpec`;
* структурная корректность поискового пространства (`low <= high`,
  положительный `step`, `float_log` требует `low > 0`);
* корректность `RobustScaling.quantile_range`, `QuantileScaling.subsample >=
  n_quantiles`, `QuantileOutliers`/`IqrOutliers`;
* согласованность фаз (`select` требует `strategy`/`output`, `top_k` требует
  `k`, `k` запрещён для `winner`/`pool`; `refine_winner` отклоняется);
* структурные cross-field правила `general` (метрики, дубликаты, бюджет
  `n_repeats * n_folds`, консистентность pruning, порядок фаз);
* коды миграционных ошибок `removed_plugin_module_key`,
  `removed_refine_winner_action`, `removed_sanity_gate_mode`;
* JSON Schema, генерируемая из контрактных моделей.

### 6.2. Runtime-валидация движка

Остаётся в `configurable_automl_engine`:

* наличие установленного пакета алгоритма (`is_installed`,
  `ALGO_PACKAGE_MAPPING`, optional extra `xgboost`);
* доступность `joblib` для `serialization_format="joblib"`;
* разрешённые гиперпараметры по сигнатуре конкретного sklearn-эстиматора
  (`get_allowed_hyperparameters`), legacy-имена, `**kwargs`-исключения
  (например, `xgboosting`);
* нормализация типа диапазона по типу параметра
  (`get_param_numeric_type`, `normalize_hyperparameter_entry`);
* поддержка метрики установленной версией sklearn/движка, импорт custom metric
  callable, проверка сигнатуры, разрешение направления scorer-а, коллизии имён
  со sklearn;
* `validation_strategy="auto"` и выбор hold-out/k-fold/LOO по данным;
* проверка `n_samples` для repeats/folds, достаточность данных, пустой/константный
  target, NaN/inf-политика, datetime-политика, ID-like колонки;
* доменные проверки target transform (`log1p`, `boxcox`) и GLM target-domain
  guard, потому что они требуют данных;
* доступность ресурсов, `multiprocessing` start method, timeouts, CPU count;
* реальное поведение препроцессинга, feature selection, oversampling, HPO,
  sanity gate и orchestration.

### 6.3. Спорные проверки

| Проверка | Решение C1 | Обоснование |
|---|---|---|
| `serialization_format="joblib"` | runtime | требует установленного `joblib` |
| `comparison_metric` входит в `AVAILABLE_METRICS` или custom | контракт для встроенного набора; runtime для custom callable | структурная проверка имён отделена от импорта функции |
| `custom_metrics` path/kind/direction | структура — контракт, импорт и сигнатура — runtime | контракт не должен импортировать пользовательский код |
| target transform domain | runtime | требует значений `y` |
| `n_repeats`/`n_folds` бюджет | контракт | считается только из конфигурации |
| `n_samples` vs `n_folds * n_repeats` | runtime | требует данных |
| `xgb`/sklearn backend | runtime | требует окружения |
| `sanity_gate` thresholds/ordering | контракт | структурные ограничения полей |
| `algorithm_families`/`allowed_families` | контракт по структуре; runtime по применимости | имена семейств — данные конфигурации |

### 6.4. Стабильные коды ошибок

Коды, которые являются частью контрактной валидации и должны сохранять
значение `error['type']`:

| Код | Где определён | Почему контракт | Решение |
|---|---|---|---|
| `removed_plugin_module_key` | `config/algorithms.py` | стабильный migration error для удалённых `tuner`/`trainer_module` | move в контракт |
| `removed_refine_winner_action` | `config/phases.py` | стабильный migration error для `refine_winner` | move в контракт |
| `removed_sanity_gate_mode` | `config/sanity_gate.py` | стабильный migration error для `sanity_gate.mode` | move в контракт |
| `custom_metric_invalid_name` | `common/metrics.py` | структурный формат имени custom metric (`^[a-z][a-z0-9_]*$`) не зависит от runtime | move в контракт |
| `custom_metric_name_collision` | `common/metrics.py` | коллизия со sklearn-именами требует sklearn | остаётся в движке |

Остальные коды custom metrics (`custom_metric_import_error`,
`custom_metric_not_callable`, `custom_metric_bad_signature`,
`custom_metric_direction_required`, `custom_metric_direction_conflict`,
`custom_metric_kwargs_unsupported`) относятся к runtime-валидации callable и
остаются в движке. Это не отменяет их стабильности как engine API, но в
`automl_contracts` они не входят.

## 7. `Protocol`-кандидаты

`Protocol` предлагается только для устойчивых публичных точек взаимодействия,
а не для внутренних интерфейсов движка.

| Кандидат | Предлагаемый контракт | Почему публичный и устойчивый | Зависимости | Решение |
|---|---|---|---|---|
| `ReporterProtocol` | `report(result: ExperimentResult) -> ReporterArtifact` | Reporter — документированная точка расширения: пользователь пишет свой sink и передаёт его в `CompositeReporter`/`AutoMLExperiment.report` | `ExperimentResult`, `ReporterArtifact`; без sklearn | needs-decision: определить в контракте или оставить в `experiment.reporting` |
| `MetricFunction` | `(y_true, y_pred, **kwargs) -> float` | `general.custom_metrics` — публичный способ добавить метрику; сигнатура уже документирована | только stdlib/`Any`; без numpy/sklearn | move в контракт как public beta |
| `ScorerCallable` | `(estimator: Any, X: Any, y: Any) -> float` | `kind="scorer"` в `custom_metrics` — публичная альтернатива функции метрики | только `Any`; без sklearn | move в контракт как public beta |
| `SearchSpaceCallable` | `(trial: Any) -> dict[str, Any]` | `optimize(space_overrides=...)` документирует callable `space_fn(trial)` | Optuna `Trial` — runtime; в контракте можно заменить на `Any`, но это менее полезно | needs-decision; скорее engine-only Protocol |
| `PredictorProtocol` | `fit`/`predict` | `ModelTrainer` работает с sklearn-совместимыми моделями, но это runtime-интерфейс | sklearn | internal; не в контракт |
| `ExperimentStoreProtocol` | `register/get/find_by_label` | store — engine API, не контракт данных | `ExperimentResult` | internal; не в контракт |

Рекомендация: в C3 добавить в контракт `MetricFunction` и `ScorerCallable`
(они не тянут runtime-зависимости), а `ReporterProtocol` и
`SearchSpaceCallable` вынести в C2/C6 как отдельное решение. Существующий
`Reporter` ABC нельзя молча заменять на `Protocol`: это сломает наследников;
`ReporterProtocol` должен быть дополнительным структурным интерфейсом.

## 8. Судьба наборов значений `Literal[...]` и enum

Общее правило: **закрытые наборы не расширяются «молча»**. Добавление значения
в enum/`Literal`, который разбирает Pydantic-модель, — потенциально ломающее
изменение для старого потребителя: old-версия контракта отклонит новый payload.
До `1.0.0` это MINOR с записью в `CHANGELOG.md`; после `1.0.0` — оценивается
как MINOR или MAJOR по фактическому влиянию. Потребители не должны считать
неизвестные значения допустимыми, если это не оговорено отдельно.

| Набор | Где определён | Значения | Публичный? | Расширение | Неизвестные значения у потребителя |
|---|---|---|---|---|---|
| `ValidationStrategy` | `common/definitions.py` | `train_test_split`, `k_fold`, `loo`, `auto` | да, stable | новое значение — MINOR/MAJOR по политике; требует поддержки движка | отклонять (enum) |
| `EncodingStrategy` | `common/definitions.py` | `one_hot`, `ordinal`, `target`, `frequency`, `hashing` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `SerializationFormat` | `common/definitions.py` | `pickle`, `joblib` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `TargetTransform` | `common/definitions.py` | `none`, `log1p`, `boxcox`, `yeo-johnson`, `auto` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `ParallelStrategy` | `common/definitions.py` | `algorithms` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `RunStatus` | `common/definitions.py` | `completed`, `failed` | да, stable | новое значение — потенциально MAJOR | отклонять |
| `RunStepStatus` | `common/definitions.py` | `completed`, `failed`, `disqualified` | да, stable | новое значение — потенциально MAJOR | отклонять |
| `RunPhaseAction` | `common/definitions.py` | `all_algorithms`, `refine`, `select` | да, stable | новое значение — MAJOR для `AlgorithmRun`-потребителей | отклонять |
| `MetricDirection` | `common/definitions.py` | `minimize`, `maximize` | да, stable | новое значение — MAJOR | отклонять |
| `ComparisonMetric` | `config/general.py` | `str`; встроенные имена + custom names | да, beta | встроенное имя — MINOR/MAJOR; custom — расширяемо через `custom_metrics` | для custom — разрешать только объявленные |
| `AVAILABLE_METRICS` | `common/metrics.py` (сейчас) | `nrmse`, `rmse`, `neg_root_mean_squared_error`, `mae`, `mse`, `r2` | да, beta | новое встроенное имя — MINOR/MAJOR; не импортировать из движка в контракт | отклонять неизвестное встроенное имя; custom — только через декларацию |
| `AVAILABLE_ALGORITHMS` | `models.py` (сейчас) | 19 алгоритмов | да, beta | новый алгоритм — MINOR для producers, но old consumer `Config` его отклонит | отклонять неизвестный ключ `Config`; новые имена — только с обновлением контракта |
| `HPOPhaseAction` | `config/phases.py` | `all_algorithms`, `refine`, `select` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `SelectStrategy` | `config/phases.py` | `cv_best`, `sanity_gate` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `SelectOutput` | `config/phases.py` | `winner`, `top_k`, `pool` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `PruningStrategy` | `config/phases.py` | `median`, `hyperband` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `CorridorMode` | `config/sanity_gate.py` | `multiplicative`, `additive`, `auto` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `GeneralizationGapMode` | `config/sanity_gate.py` | `warn`, `reject` | да, beta | новое значение — MINOR/MAJOR | отклонять |
| `SanityGateMode` | `config/sanity_gate.py` | `off`, `active` | нет (internal) | только внутреннее использование | n/a |
| `InfPolicy` | `config/general.py` | `as_missing`, `error` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `DatetimePolicy` | `config/general.py` | `expand`, `drop`, `error` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `SplitAggregation` | `config/general.py` | `mean`, `median` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `OversamplingAlgorithm` | `config/algorithms.py` | `random`, `smote`, `adasyn` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `ComplexityGuardMode` | `config/complexity_guard.py` | `off`, `warn`, `apply` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `ComplexityGuardUserRangePolicy` | `config/complexity_guard.py` | `user_wins`, `engine_wins` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `ComplexityGuardSizeBasis` | `config/complexity_guard.py` | `rows`, `groups` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `EnsembleWeightMethod` | `config/ensemble.py` | `score_weighted`, `equal` | да, beta | новое значение — MINOR/MAJOR | отклонять |
| `CustomMetricKind` | `common/custom_metrics.py` | `metric`, `scorer` | да, beta | новое значение — MINOR/MAJOR | отклонять |
| `ScalingString` / `StringScaling` | `preprocessing_presets.py` | `standard`, `robust`, `quantile`, `none` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `OutlierString` / `StringOutlierSpec` | `outliers_presets.py` | `none`, `quantile`, `iqr` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `ImputationStrategy` | `preprocessing_presets.py` | `mean`, `median` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `QuantileScaling.output_distribution` | `preprocessing_presets.py` | `uniform`, `normal` | да, stable | новое значение — MINOR/MAJOR | отклонять |
| `SearchSpaceEntry` discriminator | `hpo/default_spaces.py` | `categorical`, `int`, `float`, `float_log` | да, beta | новое значение — MINOR/MAJOR | отклонять |

Отдельно про `Literal[*AVAILABLE_METRICS]`: в текущем коде такого типа **нет**;
`ComparisonMetric = str`, а `AVAILABLE_METRICS` — вычисляемый список. В контракте
нельзя делать `Literal[*AVAILABLE_METRICS]` с импортом из движка. Если C2 решит
ввести более строгий тип, он должен:

1. определить значения независимо в контракте;
2. либо оставить `str`/`TypeAlias` с отдельным набором `BUILTIN_METRIC_NAMES`;
3. явно решить, допускаются ли custom-имена и произвольные sklearn-scorer-имена;
4. зафиксировать, что расширение закрытого `Literal` — потенциально ломающее
   изменение.

Рекомендация C1: оставить `ComparisonMetric = str` (потому что custom names
являются частью публичного контракта), а `AVAILABLE_METRICS` определить как
независимый стабильный список в контракте. Не превращать его в закрытый
`Literal` без отдельного ADR.

## 9. Проблемные зависимости и циклы

### 9.1. Основные зависимости, блокирующие перенос

| Модуль | Что тянет | Почему блокирует контракт | Как разрезать |
|---|---|---|---|
| `common/definitions.py` | `numpy` | `WORST_SCORE_THRESHOLD` | заменить на литерал `float`; numpy убрать |
| `training_engine/config/algorithms.py` | `models` (sklearn), `hpo.default_spaces` (numpy/models), `preprocessing_presets` (models) | контрактные модели зависят от движка | вынести структуры в контракт; список алгоритмов — в контракт; runtime-проверки — в движок |
| `training_engine/config/general.py` | `common.metrics` (sklearn), `common.custom_metrics` (sklearn), `feature_selection` (sklearn), `preprocessing` (sklearn), `preprocessing_presets` (models) | контракт `GeneralCfg` тянет runtime | заменить импорты на контрактные модели/константы; runtime-проверки — отдельно |
| `training_engine/config/general_rules.py` | `dependency_utils`, `multiprocessing`, `common.metrics` | часть правил runtime | чистые правила — в контракт; `joblib`/`multiprocessing`/данные — в движок |
| `training_engine/config/root.py` | `yaml`, `dependency_utils`, `hpo.default_spaces`, `models` | `read_config` и runtime-валидаторы | контрактная `Config` без YAML и models; `read_config`/runtime — в движке |
| `hpo/default_spaces.py` | `numpy`, `models` | distribution-модели и clipping в одном модуле | distribution-модели/`SearchSpaceEntry` — в контракт; `DEFAULT_SPACES`/clipping/типизация — в движок |
| `feature_selection.py` | `numpy`, `pandas`, `scipy`, `sklearn` | config-модель в одном модуле с transformer-ом | `FeatureSelectionCfg`/enum-ы — в контракт; `FeatureSelector` — в движок |
| `preprocessing_presets.py` | `models` (sklearn), `outliers_presets` | config-модели и выбор пресета в одном модуле | модели/`merge_preprocessing_overrides` — в контракт; таблицы/`resolve_*` — в движок |
| `experiment/result.py` | `numpy`, `common.metrics` (sklearn) | DTO результата зависит от runtime | стандартные числа + статическая таблица метрик; нормализация numpy — на границе движка |
| `common/custom_metrics.py` | `sklearn` | декларация и проверка callable в одном модуле | поля/структура — в контракт; импорт/сигнатура/sklearn — в движок |
| `common/metrics.py` | `sklearn`, `numpy`, `pydantic_core` | `AVAILABLE_METRICS` и alias-map в runtime-модуле | статический набор и alias-map — в контракт; scorer-фабрика — в движок |
| `schema_generator.py` | `config_parser` -> runtime-импорты | генератор зависит от движка | генерировать из контрактной `Config` |

### 9.2. Кажущийся цикл `config_parser ↔ models/metrics`

Формального цикла импортов может не быть, но есть нарушение слоёв:
`config` (контракт) импортирует `models` и `metrics` (runtime), а те, в свою
очередь, используют определения из `common`. После разрезания направление
должно стать строго таким:

```text
automl_contracts (stdlib + pydantic)
        ▲
        │
configurable_automl_engine (sklearn/pandas/numpy/optuna/yaml)
```

Внутри контракта:

```text
definitions.py, errors.py, metrics.py (статические значения)
        ▲
        ├── search_space.py
        ├── preprocessing.py
        ├── feature_selection.py
        ├── config/*.py
        └── result.py
```

`automl_contracts` не импортирует `configurable_automl_engine` ни напрямую, ни
через `TYPE_CHECKING`, если это создаёт обязательную структурную связь.

### 9.3. Цикл `preprocessing_presets ↔ outliers_presets`

Сейчас `preprocessing_presets` импортирует `outliers_presets` в середине файла,
чтобы обойти цикл `outliers_presets -> preprocessing package -> builder ->
preprocessing_presets`. При переносе в контракт оба модуля становятся чистыми
pydantic-модулями; `outliers_presets` больше не должен импортировать
preprocessing-пакет. После этого цикл устраняется, а старый путь
`preprocessing_presets` сохраняет реэкспорт outlier-имён.

## 10. Что не является контрактом

Следующие сущности не должны попасть в `automl-contracts` в C3:

* `trainer.py`: `ModelTrainer`, `train_model`, `trainer_params`,
  `trainer_data`, `trainer_artifact`, `trainer_scoring`,
  `trainer_feature_selection`, `trainer_quick`, `trainer_state`;
* `tuner.py`: `optimize`, `OptimizeResult`, `TrialStats`, HPO-исключения,
  `HPO_WORST_SCORE` (движковый sentinel; `WORST_SCORE_THRESHOLD` — отдельно
  в контракте);
* `hpo/*`: runner, evaluation, search_space, pruning, trial_pipeline, repeats,
  complexity_guard, target_domain, `DEFAULT_SPACES`;
* `training_engine/*`: `component`, `data_prep`, `artifact`, `state`,
  `hpo_phases`, `select_steps`, `finalists`, `robust_winner`, `sanity_gate`,
  `sanity_circuits`, `sanity_repeats`, parallel-модули, `thread_pool`;
* `feature_selection.FeatureSelector` — sklearn transformer;
* `preprocessing/*`: `build_preprocessor`, `Winsorizer`, encoders, quantile
  transformer, `TargetTransformer`, `AutoTargetTransformer`,
  `resolve_target_transform`, `validate_target_for_transform`;
* `oversampling/*`: `DataOversampler`, `oversample`, SMOTER-генератор;
* `models.py`: `create_model`, `get_allowed_hyperparameters`,
  `LEGACY_PARAM_MAPPINGS`, `SPARSE_UNSUPPORTED_ALGORITHMS`, `MODEL_RANDOM_STATE`;
* `validation.py`: `make_cv`, `iter_splits`, `aggregate_split_scores`,
  `RepeatedSplit`, `ValidationError`, `InvalidDataError`;
* `experiment/*`: `AutoMLExperiment`, `ExperimentStore`, `RunSummary`,
  `Leaderboard`, `LeaderboardEntry`, `Reporter`, `JSONReporter`,
  `MarkdownReporter`, `MLflowReporter`, `export_experiment`, explain-хелперы;
* `common/*`: `metrics` runtime-функции, `validation_utils`, `shared_frame`,
  `serialization_utils`, `log_setup`, `settings`, `dependency_utils`,
  `sklearn_compat`, `datetime_features`, `bbc_cv`, `isotonic`;
* `schema_generator.py` — инструмент, а не публичный модуль;
* `_version.py` — генерируемый артефакт.

`RunSummary`, `LeaderboardEntry`, `FeatureImportance`, `EnsembleMember` и
`ReporterArtifact` — пограничные DTO. В C1 они не включены в минимальный
контракт; если C6 захочет сделать их публичными стабильными, это нужно
оформить отдельным расширением с проверками сериализации.

## 11. Разрешённые runtime-зависимости контракта

Базовое правило эпика:

* стандартная библиотека Python;
* `pydantic` (Pydantic v2; текущий `pyproject.toml` требует `pydantic>=2.12.5`);
* никаких `scikit-learn`, `pandas`, `numpy`, `optuna`, `PyYAML`, `scipy`,
  `joblib`, `imbalanced-learn`, `pyarrow`, `mlflow`, `shap`.

Дополнительно требуется решение C2 по `pydantic_core`:

* сейчас код использует `from pydantic_core import PydanticCustomError` для
  стабильных кодов ошибок;
* `pydantic_core` — транзитивная зависимость `pydantic`, но формально это
  отдельный пакет;
* C2 должен либо включить `pydantic-core` в разрешённый runtime-набор
  контракта (как часть pydantic), либо заменить прямые импорты на публичный
  API pydantic/`ValueError` без потери стабильных `error['type']`.

`read_config` использует `PyYAML` и поэтому остаётся в движке. Если контракту
понадобится чтение YAML, это будет отдельное обоснование в ADR.

## 12. JSON Schema и совместимость

`config.schema.json` — производный артефакт. Правила:

* источник истины — контрактные Pydantic-модели;
* схема генерируется из `Config.model_json_schema()`;
* сохранённый файл включается в wheel `automl-contracts` как package data;
* CI (C4/C5) проверяет воспроизводимость и отсутствие расхождений;
* корневой `config.schema.json` может остаться как публикуемый артефакт движка,
  но не должен быть независимым источником истины;
* `schema_generator.py` остаётся внутренним инструментом и после C3 импортирует
  контрактную `Config`.

Классификация изменений схемы:

| Изменение | Оценка |
|---|---|
| Новое необязательное поле с безопасным default | MINOR, если не ломает сериализацию |
| Новое значение enum/`Literal` | потенциально MINOR/MAJOR: старый потребитель может отклонять payload |
| Новое обязательное поле | MAJOR |
| Удаление поля/значения/типа | MAJOR |
| Изменение типа или семантики поля | MAJOR |
| Изменение `description`/порядка полей без семантики | PATCH |
| Переименование поля без alias | MAJOR |

Схема не должна анализироваться только по факту изменения файла: важна
направленность совместимости (producer/consumer) и поведение старых
валидаторов.

## 13. Открытые вопросы для C2/C6

1. **`InvalidPredictionError`**: публичный beta или internal? Рекомендация —
   public beta с реэкспортом из `common.errors`.
2. **`read_config`**: подтвердить, что остаётся только в движке из-за PyYAML.
3. **`ExperimentResult`**: как именно контрактная модель проверяет
   `metric_direction` без sklearn; нужен ли статический реестр встроенных
   метрик и как обрабатывать произвольные sklearn-имена.
4. **`pydantic_core`**: разрешённая зависимость контракта или замена
   `PydanticCustomError`.
5. **Тип `AVAILABLE_ALGORITHMS` / `AVAILABLE_METRICS`**: `list` или `tuple`;
   кто владеет каноническим значением; нужна ли проверка идентичности.
6. **`ReporterProtocol`**: контракт или engine-only; как совместить с текущим
   `Reporter` ABC.
7. **Прямые импорты `training_engine.config.*`**: сохранять шимы или обновить
   тесты и объявить пути internal.
8. **`DEFAULT_SPACES`, `PRESET_BY_CLASS`, `ALGORITHM_CLASS_MAPPING`**: оставить
   engine API или сделать публичными данными контракта. Рекомендация C1 —
   оставить в движке.
9. **Схема**: один файл в контракте или синхронизируемая копия в корне;
   как публиковать `config.schema.json` в wheel.
10. **Пограничные DTO** (`RunSummary`, `LeaderboardEntry`, `FeatureImportance`,
    `EnsembleMember`, `ReporterArtifact`): включать ли в контракт в C3 или
    отложить до C6.
11. **`CustomMetricSpec`**: как сохранить проверку callable при создании и при
    этом не тянуть runtime в контракт; допустима ли смена поведения прямого
    конструктора при сохранении поведения `train_best_model`/`optimize`.

## 14. Проверочный список приёмки C1

- [x] `CONTRACTS.md` лежит в корне репозитория.
- [x] Покрыты все модули из скоупа: `common/definitions.py`, `common/errors.py`,
  `training_engine/config/*`, `config_parser.py`, `hpo/default_spaces.py`,
  `feature_selection.py`, `preprocessing_presets.py` (+ `outliers_presets.py`),
  `config.schema.json`, `schema_generator.py`, `experiment/result.py`,
  `common/metrics.py`, `common/custom_metrics.py`, `models.py`.
- [x] Для каждой сущности указан статус публичности и решение
  (`move` / `re-export` / `internal` / `deprecated` / `needs-decision`).
- [x] Зафиксированы матрица старых публичных путей импорта (включая подмодули)
  и правило их сохранения/изменения.
- [x] Разделены контрактная и runtime-валидация; перечислены проверки,
  остающиеся в движке.
- [x] Определены `Protocol`-кандидаты и судьба `Literal`/enum-наборов.
- [x] Зафиксирован перечень разрешённых зависимостей контракта: stdlib +
  `pydantic`; `pydantic-core` вынесен в решение C2.
- [x] Код в рамках C1 не изменён: добавлен только `CONTRACTS.md`.
