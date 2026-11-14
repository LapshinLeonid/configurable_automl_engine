# ADR 0001: два независимых дистрибутива в одном репозитории

- **Статус:** accepted
- **Дата:** 2026-11-14
- **Контекст:** эпик [#79](https://sourcecraft.dev/ooo-ainod/configurable-automl-engine/issues/79),
  задача C1 [#80](https://sourcecraft.dev/ooo-ainod/configurable-automl-engine/issues/80),
  задача C2 [#81](https://sourcecraft.dev/ooo-ainod/configurable-automl-engine/issues/81)
- **Связанный аудит:** [CONTRACTS.md](../../CONTRACTS.md)

## 1. Контекст

Сейчас `configurable-automl-engine` — один дистрибутив: корневой `pyproject.toml`,
пакет `src/configurable_automl_engine/`, общий каталог `tests/`. Публичные
контрактные сущности (модели конфигурации, DTO, перечисления, исключения,
`Protocol`) живут в тех же модулях, что и реализация движка, а часть из них
транзитивно зависит от `numpy`, `scikit-learn`, `pandas`, `optuna` и `PyYAML`.

C1 зафиксировал в `CONTRACTS.md` матрицу публичного API, старые пути импорта,
границу контрактной/runtime-валидации и список сущностей, которые переносятся в
`automl-contracts`. Следующий шаг — принять архитектурное решение о форме
дистрибутивов до массовой миграции C3.

Цель решения:

- дать потребителям возможность установить контракт без движка и его тяжёлых
  зависимостей;
- сохранить один источник истины для контракта и реализации;
- сохранить обратную совместимость публичных путей импорта;
- обеспечить независимые сборку, версионирование, тестирование и публикацию
  двух дистрибутивов;
- не превращать миграцию в перемещение всего репозитория.

## 2. Решение

Принят **монорепозиторий с двумя независимо собираемыми и версионируемыми
дистрибутивами**:

1. `automl-contracts` — контракт, импортируемый модуль `automl_contracts`;
2. `configurable-automl-engine` — реализация движка, импортируемый модуль
   `configurable_automl_engine`.

Раскладка выбрана гибридная:

- движок остаётся в текущем корне репозитория: корневой `pyproject.toml` и
  `src/configurable_automl_engine/`;
- контракт получает собственный корень `automl-contracts/` с собственным
  `pyproject.toml` и `src/automl_contracts/`;
- тесты разделяются на `tests/contracts/` и `tests/engine/`;
- `CONTRACTS.md` остаётся в корне как результат аудита C1;
- ADR и остальная документация остаются в `docs/`.

Полная симметричная раскладка из эпика
(`distributions/automl-contracts/` и
`distributions/configurable-automl-engine/`) не выбрана: она требует
перемещения всего исходного кода и тестов, ломает текущие команды разработки,
пути `setuptools-scm`, `MANIFEST.in`, CI и `import-linter`, но не даёт
функциональных преимуществ по сравнению с гибридной раскладкой. Обоснование —
в разделе 3.4.

## 3. Рассмотренные варианты

### 3.1. Монорепозиторий с двумя дистрибутивами (выбран)

Плюсы:

- единый источник истины: изменение контракта и его использование в движке
  делается одним атомарным PR;
- нет синхронизации двух репозиториев, двух CI и двух наборов прав;
- общие тесты совместимости и общая история изменений;
- контракт можно собирать и устанавливать отдельно, не устанавливая движок;
- рефакторинг старых путей импорта можно делать постепенно, сохраняя
  реэкспорты.

Минусы:

- две сборочные конфигурации и два релизных цикла;
- требуется дисциплина тегов, чтобы `contracts-v*` и `v*` не путались;
- CI сложнее: нужны независимые job-ы и матрица совместимости;
- сохраняется риск случайной связи контракта с движком, который нужно
  подавлять `import-linter` и тестами.

Принят, потому что минусы управляемы, а плюсы напрямую нужны для C3–C7.

### 3.2. Отдельный репозиторий под контракты (отклонён)

Плюсы:

- жёсткая изоляция и независимые права/CI;
- невозможно случайно импортировать движок из контракта;
- более простая история релизов контракта.

Минусы:

- совместное изменение контракта и движка требует двух PR в двух репозиториях;
- теряется атомарность: движок может оказаться на несовместимой версии
  контракта;
- нужна синхронизация версий и схемы между репозиториями;
- CI совместимости требует доступа к двум репозиториям и токенов;
- миграция C3 и проверки старых путей импорта усложняются.

Отклонён. Изоляцию можно обеспечить внутри монорепозитория правилами импортов и
отдельной сборкой.

### 3.3. Копия контрактов в зависимых репозиториях (отклонён)

Плюсы: локальные импорты и отсутствие внешней зависимости.

Минусы:

- копии расходятся;
- непонятно, какая версия контракта является истинной;
- невозможно проверить идентичность классов и типов;
- исправления нужно дублировать;
- JSON Schema и `Literal`/enum-наборы теряют единый источник.

Отклонён как не решающий задачу версионирования.

### 3.4. Варианты раскладки внутри монорепозитория

#### 3.4.1. Полная раскладка `distributions/...` (не выбрана)

Плюсы:

- симметрия и очевидная граница между дистрибутивами;
- корень репозитория содержит только общие файлы;
- проще объяснить внешнему участнику.

Минусы для текущего репозитория:

- нужно переместить `src/configurable_automl_engine/` и более 100 тестовых
  файлов;
- сломаются текущие команды `ruff check src/`, `mypy src`, `pytest --cov=src`,
  `lint-imports`, `python -m build`, скрипты и GitHub workflows;
- `setuptools-scm` пишет `_version.py` по текущему пути; его настройка,
  `MANIFEST.in` и package discovery также привязаны к корню;
- большой механический diff затруднит ревью C3 и повысит риск регрессий;
- нет функциональной выгоды: обе раскладки одинаково разделяют дистрибутивы.

Не выбрана. Переезд движка под `distributions/` можно сделать отдельным ADR
после стабилизации контракта, если появится практическая необходимость.

#### 3.4.2. Гибридная раскладка (выбрана)

- движок остаётся на текущем месте;
- контракт получает отдельный корень `automl-contracts/`;
- тесты разделяются, но переносятся постепенно;
- общие файлы (`CONTRACTS.md`, `docs/`, корневой `pyproject.toml`) остаются в
  корне.

Плюсы:

- минимальный механический diff в C3;
- сохраняются текущие команды и workflows движка;
- контракт сразу имеет собственный `pyproject.toml`, `src/` и `CHANGELOG.md`;
- отдельная сборка контракта не зависит от корневого `pyproject.toml`.

Минус — асимметрия: движок в корне, контракт в подкаталоге. Она компенсируется
явными правилами package discovery и `MANIFEST.in`, которые не дают сборкам
смешиваться.

## 4. Имена и границы дистрибутивов

| Дистрибутив | Импортируемый модуль | Корень исходников | Содержимое |
|---|---|---|---|
| `automl-contracts` | `automl_contracts` | `automl-contracts/src/automl_contracts/` | Контрактные модели, DTO, перечисления, исключения, `Protocol`, константы, JSON Schema |
| `configurable-automl-engine` | `configurable_automl_engine` | `src/configurable_automl_engine/` | Обучение, HPO, оркестрация, препроцессинг, отчёты, store, runtime-интеграции |

Границы:

- `automl_contracts` не содержит кода обучения, оркестрации, HPO, отчётов,
  препроцессинга и runtime-интеграций;
- `configurable_automl_engine` может импортировать `automl_contracts` и
  реэкспортировать его публичные имена;
- потребитель, которому нужны только типы, модели конфигурации, `Protocol` или
  JSON Schema, устанавливает только `automl-contracts`;
- потребитель, которому нужен запуск экспериментов, устанавливает
  `configurable-automl-engine`, который притягивает `automl-contracts` как
  обязательную зависимость.

## 5. Структура репозитория

Целевая структура после C3:

```text
configurable-automl-engine/
├── pyproject.toml                         # дистрибутив движка
├── MANIFEST.in
├── src/
│   └── configurable_automl_engine/        # существующий пакет движка
│       ├── __init__.py
│       ├── py.typed
│       ├── common/
│       ├── hpo/
│       ├── training_engine/
│       └── ...
├── automl-contracts/                      # дистрибутив контракта
│   ├── pyproject.toml
│   ├── MANIFEST.in
│   ├── CHANGELOG.md
│   ├── scripts/
│   │   └── generate_schema.py
│   └── src/
│       └── automl_contracts/
│           ├── __init__.py
│           ├── py.typed
│           ├── definitions.py
│           ├── errors.py
│           ├── metrics.py
│           ├── custom_metrics.py
│           ├── protocols.py
│           ├── search_space.py
│           ├── feature_selection.py
│           ├── preprocessing.py
│           ├── result.py
│           ├── config/
│           │   ├── __init__.py
│           │   ├── root.py
│           │   ├── general.py
│           │   ├── general_rules.py
│           │   ├── phases.py
│           │   ├── sanity_gate.py
│           │   ├── complexity_guard.py
│           │   ├── ensemble.py
│           │   └── algorithms.py
│           └── schemas/
│               └── config.schema.json
├── tests/
│   ├── contracts/
│   └── engine/                            # существующие тесты после C3
├── scripts/                               # существующие engine-скрипты
├── docs/
│   └── adr/
│       └── 0001-contracts-package.md
├── CONTRACTS.md
├── schema_generator.py                    # совместимый генератор-обёртка
├── config.schema.json                     # совместимая копия-зеркало
└── .sourcecraft/
    └── ci.yaml
```

`automl_contracts/config/root.py`, `general_rules.py` и `schemas/` — внутренняя
раскладка, а не публичные Python-модули. Публичные модели конфигурации
доступны через `automl_contracts.config` и его секционные модули, перечисленные
в разделе 7.2.

### 5.1. Перемещение тестов

C3 выполняет механическое разделение:

- `tests/contracts/` — только тесты контракта, которые не импортируют
  `configurable_automl_engine` и тяжёлые зависимости;
- `tests/engine/` — текущие тесты движка, перенесённые из `tests/`;
- общий `tests/conftest.py` и фикстуры движка переезжают в
  `tests/engine/conftest.py`; контрактные тесты не должны зависеть от
  engine-фикстур;
- импорты в тестах обновляются на новые пути, но старые пути проверяются
  отдельным набором тестов обратной совместимости.

## 6. Сборочная конфигурация

### 6.1. Общий backend

Оба дистрибутива используют `setuptools` с backend `setuptools.build_meta`.
Причины:

- текущий движок уже собран на `setuptools` и `setuptools-scm`;
- не требуется менять существующий release flow движка;
- `setuptools` поддерживает `src`-layout, package data и статическую версию для
  контракта.

Для контракта backend фиксируется так:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
```

`setuptools-scm` в контракте не используется: версия статическая, а теги
`contracts-v*` не должны участвовать в вычислении версии движка.

### 6.2. Дистрибутив движка

Корневой `pyproject.toml` остаётся конфигурацией движка и дополняется явным
package discovery, чтобы сборка не захватила контракт:

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["configurable_automl_engine*"]
```

Версия движка остаётся динамической через `setuptools-scm`, но получает
явный шаблон тега, который не совпадает с `contracts-v*`:

```toml
[tool.setuptools_scm]
tag_regex = "^v(?P<version>\\d+\\.\\d+\\.\\d+.*)$"
```

Это сохраняет текущие теги движка (`v0.6.0` и т. п.) и исключает случайное
версионирование движка по тегу контракта.

Корневой `MANIFEST.in` дополняется `prune automl-contracts`, чтобы sdist
движка не содержал исходники второго дистрибутива. `setuptools-scm` не должен
включать контракт в sdist движка.

### 6.3. Дистрибутив контракта

`automl-contracts/pyproject.toml` содержит:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "automl-contracts"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.12.5,<3",
    "pydantic-core>=2,<3",
]

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["automl_contracts*"]

[tool.setuptools.package-data]
automl_contracts = ["py.typed", "schemas/*.json"]
```

`automl-contracts/MANIFEST.in` включает `CHANGELOG.md`, `py.typed` и
`schemas/*.json` в sdist; служебные `scripts/` и тесты в sdist не входят.

### 6.4. Команды сборки и отсутствие неявной сборки

Сборка каждого дистрибутива выполняется отдельно и только из его корня:

```bash
# движок
python -m build . --outdir dist/engine

# контракт
python -m build automl-contracts --outdir automl-contracts/dist
```

Правила:

- `python -m build` в корне собирает только движок: package discovery ограничен
  `src`, а `MANIFEST.in` исключает `automl-contracts`;
- сборка контракта читает только `automl-contracts/pyproject.toml` и
  `automl-contracts/src`;
- ни одна команда сборки не публикует второй дистрибутив;
- CI использует отдельные job-ы и отдельные артефакты для `dist/engine` и
  `automl-contracts/dist`;
- установка из исходников в CI выполняется явно и по порядку:
  `pip install -e automl-contracts`, затем `pip install -e .`.

### 6.5. Package data

В wheel контракта обязательно попадают:

- `automl_contracts/py.typed`;
- `automl_contracts/schemas/config.schema.json`.

`py.typed` находится в корне пакета и включается через `package-data`.
JSON Schema лежит внутри пакета, а не только в корне репозитория, чтобы
потребитель мог получить её через `importlib.resources`.

В wheel движка остаётся `configurable_automl_engine/py.typed`. Корневой
`config.schema.json` не является частью wheel движка; каноническая схема
публикуется в wheel контракта.

## 7. Публичная точка входа `automl_contracts`

### 7.1. Назначение `__init__.py`

`automl_contracts/__init__.py`:

- реэкспортирует публичные определения верхнего уровня;
- фиксирует их в `__all__`;
- не содержит бизнес-логики, чтения файлов, генерации схемы и импортов движка;
- определяет `__version__` из метаданных установленного дистрибутива:

  ```python
  from importlib.metadata import PackageNotFoundError, version

  try:
      __version__ = version("automl-contracts")
  except PackageNotFoundError:  # исходное дерево без установки
      __version__ = "0.0.0"
  ```

  Версия `0.0.0` — только fallback для незапущенного из установленного
  дистрибутива источника; CI и публикация используют установленную версию.

Импорт `automl_contracts` не должен импортировать `configurable_automl_engine`
ни напрямую, ни через `TYPE_CHECKING`, и не должен требовать тяжёлых
зависимостей.

### 7.2. Стабильные публичные подмодули

Потребителям разрешено опираться на следующие подмодули:

| Подмодуль | Публичные символы | Стабильность |
|---|---|---|
| `automl_contracts.definitions` | `ValidationStrategy`, `EncodingStrategy`, `SerializationFormat`, `TargetTransform`, `ParallelStrategy`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection`, `WORST_SCORE_THRESHOLD`, `AVAILABLE_ALGORITHMS` | stable, кроме `AVAILABLE_ALGORITHMS` и `WORST_SCORE_THRESHOLD` — beta |
| `automl_contracts.errors` | `TrainingError`, `InvalidPredictionError` | `TrainingError` — stable, `InvalidPredictionError` — beta |
| `automl_contracts.metrics` | `AVAILABLE_METRICS` | beta |
| `automl_contracts.custom_metrics` | `CustomMetricKind`, `CustomMetricSpec`, `ERR_CUSTOM_NAME_INVALID` | `CustomMetricKind` — beta; `CustomMetricSpec` — beta; `ERR_CUSTOM_NAME_INVALID` — stable |
| `automl_contracts.protocols` | `MetricFunction`, `ScorerCallable` | beta |
| `automl_contracts.search_space` | `SearchSpaceEntry`, `BaseDistribution`, `CategoricalSpace`, `NumericSpace`, `FloatSpace`, `IntSpace` | `SearchSpaceEntry` — stable, distribution-модели — beta |
| `automl_contracts.feature_selection` | `FeatureSelectionMode`, `FeatureSelectionMethod`, `FeatureSelectionCfg` | stable |
| `automl_contracts.preprocessing` | `ImputationStrategy`, `ScalingType`, `StandardScaling`, `RobustScaling`, `QuantileScaling`, `NoScaling`, `ObjectScalingSpec`, `StringScaling`, `ScalingSpec`, `AlgorithmClass`, `PreprocessingPreset`, `PreprocessingOverride`, `merge_preprocessing_overrides`, `NoOutliers`, `QuantileOutliers`, `IqrOutliers`, `ObjectOutlierSpec`, `StringOutlierSpec`, `OutlierSpec`, `NO_OUTLIERS` | `ImputationStrategy`, `StandardScaling`, `RobustScaling`, `QuantileScaling`, `NoScaling`, `AlgorithmClass`, `PreprocessingPreset`, `PreprocessingOverride`, `NoOutliers`, `QuantileOutliers`, `IqrOutliers` — stable; `ObjectScalingSpec`, `StringScaling`, `ScalingSpec`, `merge_preprocessing_overrides`, `ObjectOutlierSpec`, `StringOutlierSpec`, `OutlierSpec`, `NO_OUTLIERS` — beta; `ScalingType` — deprecated alias |
| `automl_contracts.result` | `AlgorithmRun`, `ExperimentResult` | stable |
| `automl_contracts.config` | `Config`, `GeneralCfg`, `HPOPhaseCfg`, `SanityGateCfg`, `ComplexityGuardCfg`, `EnsembleCfg`, `AlgoCfg`, `OversamplingCfg`, `AlgorithmsConfig` | `Config`, `GeneralCfg`, `HPOPhaseCfg`, `SanityGateCfg`, `ComplexityGuardCfg`, `AlgoCfg`, `OversamplingCfg` — stable; `EnsembleCfg`, `AlgorithmsConfig` — beta |
| `automl_contracts.config.general` | `GeneralCfg`, `ComparisonMetric`, `AUTO_HOLDOUT_FRACTION`, `AUTO_HOLDOUT_MIN_ROWS`, `InfPolicy`, `DatetimePolicy`, `SplitAggregation` | `GeneralCfg`, `InfPolicy`, `DatetimePolicy`, `SplitAggregation` — stable; `ComparisonMetric`, `AUTO_HOLDOUT_FRACTION`, `AUTO_HOLDOUT_MIN_ROWS` — beta |
| `automl_contracts.config.phases` | `HPOPhaseAction`, `SelectStrategy`, `SelectOutput`, `PruningCfg`, `PruningStrategy`, `HPOPhaseCfg`, `DEFAULT_PHASE_N_TRIALS`, `ERR_REMOVED_REFINE_WINNER` | `HPOPhaseAction`, `SelectStrategy`, `SelectOutput`, `PruningStrategy`, `HPOPhaseCfg`, `ERR_REMOVED_REFINE_WINNER` — stable; `PruningCfg`, `DEFAULT_PHASE_N_TRIALS` — beta |
| `automl_contracts.config.sanity_gate` | `SanityGateCfg`, `CorridorMode`, `GeneralizationGapMode`, `ERR_REMOVED_SANITY_GATE_MODE` | `SanityGateCfg`, `CorridorMode`, `ERR_REMOVED_SANITY_GATE_MODE` — stable; `GeneralizationGapMode` — beta |
| `automl_contracts.config.complexity_guard` | `ComplexityGuardCfg`, `ComplexityGuardMode`, `ComplexityGuardUserRangePolicy`, `ComplexityGuardSizeBasis` | stable |
| `automl_contracts.config.ensemble` | `EnsembleCfg`, `EnsembleWeightMethod` | beta |
| `automl_contracts.config.algorithms` | `AlgoCfg`, `OversamplingCfg`, `OversamplingAlgorithm`, `AlgorithmsConfig`, `ERR_REMOVED_PLUGIN_KEY` | `AlgoCfg`, `OversamplingCfg`, `OversamplingAlgorithm`, `ERR_REMOVED_PLUGIN_KEY` — stable; `AlgorithmsConfig` — beta |

`automl_contracts.config.root` и `automl_contracts.config.general_rules`
являются внутренними модулями реализации. `Schema`-файл
`automl_contracts/schemas/config.schema.json` — публикуемый data-артефакт, но
не Python-модуль.

Публичные имена доступны также из корня `automl_contracts`. Если имя есть в
подмодуле, но отсутствует в корневом `__all__`, это осознанное решение:
например, `ScalingType` остаётся deprecated-алиасом в
`automl_contracts.preprocessing` и не рекламируется на верхнем уровне.

### 7.3. `__all__` верхнего уровня

```python
__all__ = [
    "__version__",
    # definitions
    "ValidationStrategy",
    "EncodingStrategy",
    "SerializationFormat",
    "TargetTransform",
    "ParallelStrategy",
    "RunStatus",
    "RunStepStatus",
    "RunPhaseAction",
    "MetricDirection",
    "WORST_SCORE_THRESHOLD",
    "AVAILABLE_ALGORITHMS",
    # errors
    "TrainingError",
    "InvalidPredictionError",
    # metrics
    "AVAILABLE_METRICS",
    # custom metrics
    "CustomMetricKind",
    "CustomMetricSpec",
    "ERR_CUSTOM_NAME_INVALID",
    # protocols
    "MetricFunction",
    "ScorerCallable",
    # search space
    "SearchSpaceEntry",
    "BaseDistribution",
    "CategoricalSpace",
    "NumericSpace",
    "FloatSpace",
    "IntSpace",
    # feature selection
    "FeatureSelectionMode",
    "FeatureSelectionMethod",
    "FeatureSelectionCfg",
    # preprocessing
    "ImputationStrategy",
    "StandardScaling",
    "RobustScaling",
    "QuantileScaling",
    "NoScaling",
    "ObjectScalingSpec",
    "StringScaling",
    "ScalingSpec",
    "AlgorithmClass",
    "PreprocessingPreset",
    "PreprocessingOverride",
    "merge_preprocessing_overrides",
    "NoOutliers",
    "QuantileOutliers",
    "IqrOutliers",
    "ObjectOutlierSpec",
    "StringOutlierSpec",
    "OutlierSpec",
    "NO_OUTLIERS",
    # result
    "AlgorithmRun",
    "ExperimentResult",
    # config: root
    "Config",
    # config: general
    "GeneralCfg",
    "ComparisonMetric",
    "AUTO_HOLDOUT_FRACTION",
    "AUTO_HOLDOUT_MIN_ROWS",
    "InfPolicy",
    "DatetimePolicy",
    "SplitAggregation",
    # config: phases
    "HPOPhaseAction",
    "SelectStrategy",
    "SelectOutput",
    "PruningCfg",
    "PruningStrategy",
    "HPOPhaseCfg",
    "DEFAULT_PHASE_N_TRIALS",
    "ERR_REMOVED_REFINE_WINNER",
    # config: sanity gate
    "SanityGateCfg",
    "CorridorMode",
    "GeneralizationGapMode",
    "ERR_REMOVED_SANITY_GATE_MODE",
    # config: complexity guard
    "ComplexityGuardCfg",
    "ComplexityGuardMode",
    "ComplexityGuardUserRangePolicy",
    "ComplexityGuardSizeBasis",
    # config: ensemble
    "EnsembleCfg",
    "EnsembleWeightMethod",
    # config: algorithms
    "AlgoCfg",
    "OversamplingCfg",
    "OversamplingAlgorithm",
    "AlgorithmsConfig",
    "ERR_REMOVED_PLUGIN_KEY",
]
```

`ScalingType` не входит в `__all__`, но импортируется из
`automl_contracts.preprocessing` для совместимости. `MetricFunction` и
`ScorerCallable` — публичные beta `Protocol`; `ReporterProtocol` и
`SearchSpaceCallable` в контракт не входят (см. раздел 10.4).

### 7.4. `py.typed`

`automl_contracts/py.typed` создаётся пустым и включается в wheel через
`package-data`. CI проверяет:

- файл присутствует в собранном wheel;
- `pip install` из wheel сохраняет его;
- `mypy` видит типы контракта при установке без движка.

## 8. Зависимости, Python и Pydantic

### 8.1. Разрешённые runtime-зависимости

`automl-contracts` зависит только от:

- стандартной библиотеки Python;
- `pydantic>=2.12.5,<3`;
- `pydantic-core>=2,<3` — единственное явно обоснованное исключение.

`pydantic-core` разрешён, потому что контракт обязан сохранять стабильные коды
ошибок в `error['type']` (`removed_plugin_module_key`,
`removed_refine_winner_action`, `removed_sanity_gate_mode`,
`custom_metric_invalid_name`). Публичный способ задать такой код в Pydantic v2 —
`pydantic_core.PydanticCustomError`. Пакет уже является обязательной
транзитивной зависимостью `pydantic`; контракт объявляет его напрямую, чтобы не
зависеть от неявного транзитивного импорта. Версия `pydantic-core` ограничена
той же major-версией, что и `pydantic` v2.

Запрещены как runtime-зависимости контракта:

`scikit-learn`, `pandas`, `numpy`, `scipy`, `optuna`, `PyYAML`, `joblib`,
`imbalanced-learn`, `pyarrow`, `mlflow`, `shap`, `matplotlib` и любые другие
библиотеки, нужные только для исполнения движка или отчётов.

`pydantic_core` не должен использоваться для чего-либо, кроме стабильных кодов
ошибок; импорт этого пакета в контракте локализуется в модулях валидации
конфигурации и custom-metrics.

### 8.2. Python

- Минимальная версия: Python 3.11.
- Поддерживаются Python 3.11, 3.12, 3.13, 3.14 — те же версии, что заявлены у
  движка.
- Python 3.10 и ниже не поддерживается.
- Поддержка PyPy не заявляется без отдельного тестирования.

### 8.3. Pydantic

- Поддерживается только Pydantic v2.
- Нижняя граница: `2.12.5` — та же, что у движка.
- Верхняя граница: `<3`.
- Pydantic v1 не поддерживается: контракт использует `model_validator`,
  `field_validator`, `model_json_schema`, `Discriminator`/`Tag` и другие
  возможности v2.
- Нельзя заявлять поддержку нескольких major-версий Pydantic без тестирования
  сериализации, валидации и генерации JSON Schema на каждой версии.
- CI проверяет минимальную (`2.12.5`) и последнюю доступную версию Pydantic v2
  джобом `contracts-pydantic-matrix` (§15.2); при изменении нижней границы
  матрица обновляется в том же PR.

## 9. Направление зависимостей и правила слоёв

### 9.1. Направление между дистрибутивами

```text
consumer applications ───────┐
                             ▼
                       automl-contracts
                             ▲
                             │
              configurable-automl-engine
```

Правила:

- `automl_contracts` никогда не импортирует `configurable_automl_engine`;
- `automl_contracts` не зависит от библиотек, нужных исключительно движку;
- `configurable_automl_engine` зависит от `automl-contracts` и использует его
  определения;
- импорт `configurable_automl_engine` из контракта запрещён и напрямую, и через
  `TYPE_CHECKING`;
- реэкспорты из контракта в основном пакете разрешены и обязательны для
  обратной совместимости.

### 9.2. Слои внутри контракта

```text
errors, definitions, protocols
        ▲
        ├── metrics, custom_metrics
        ├── search_space, feature_selection, preprocessing
        ├── result
        └── config/*
```

`config/*` может импортировать определения нижних слоёв. Обратные импорты из
нижних слоёв в `config/*` запрещены. Циклы внутри контракта запрещены.

### 9.3. Принудительное соблюдение

- В `automl-contracts/pyproject.toml` добавляется `import-linter`-контракт с
  `root_package = "automl_contracts"` и запретом
  `configurable_automl_engine`; `exclude_type_checking_imports = false`.
- В CI контракта `lint-imports` запускается до тестов и обязательно с
  конфигурацией контракта: шаг выполняется с `working-directory:
  automl-contracts` (эквивалент `lint-imports --config
  automl-contracts/pyproject.toml` из корня репозитория). Голый
  `lint-imports` из корня читает корневой `pyproject.toml`, применяет только
  layer-контракты движка и не проверяет запрет импорта
  `configurable_automl_engine` — такой запуск не считается проверкой.
- Отдельный тест сканирует AST контрактных модулей и падает при любом импорте
  `configurable_automl_engine` или запрещённой библиотеки, включая
  `TYPE_CHECKING`-блоки.
- Изолированная установка wheel контракта в venv без движка и тяжёлых
  зависимостей доказывает отсутствие скрытых runtime-импортов.
- В движке `import-linter` дополняется проверкой, что новые engine-модули не
  импортируют контракт в обратном направлении и не создают второй определения
  контрактных классов.

### 9.4. Реэкспорты и запрет дублирования

Определение каждого контрактного класса, перечисления, исключения и типа
существует только в `automl_contracts`. В движке допускаются только импорты и
реэкспорты. Копирование определения или создание второго класса с тем же именем
запрещено. Для классов, enum-ов и исключений тесты проверяют идентичность
(`is`); для type aliases и констант — равенство значений и типов.

## 10. Граница контрактной и runtime-валидации

### 10.1. Что входит в контракт

- обязательные поля, типы и ограничения `Field(...)`;
- допустимые значения enum и закрытых `Literal`;
- discriminated unions `SearchSpaceEntry`, `ScalingSpec`, `OutlierSpec`;
- структурная корректность поискового пространства (`low <= high`,
  положительный `step`, `float_log` требует `low > 0`);
- ограничения `RobustScaling.quantile_range`,
  `QuantileScaling.subsample >= n_quantiles`, `QuantileOutliers`,
  `IqrOutliers`;
- структурная согласованность фаз (`select` требует `strategy`/`output`,
  `top_k` требует `k`, `k` запрещён для `winner`/`pool`);
- чистые cross-field правила `general` (метрики, дубликаты, бюджет
  `n_repeats * n_folds`, консистентность pruning, порядок фаз);
- стабильные коды контрактной валидации;
- JSON Schema, генерируемая из контрактных моделей.

### 10.2. Что остаётся runtime-валидацией движка

- наличие установленного пакета алгоритма (`is_installed`,
  `ALGO_PACKAGE_MAPPING`, optional extra `xgboost`);
- доступность `joblib` для `serialization_format="joblib"`;
- разрешённые гиперпараметры по сигнатуре конкретного sklearn-эстиматора,
  legacy-имена и исключения `**kwargs`;
- нормализация типа диапазона по типу параметра;
- поддержка метрики установленной версией sklearn, импорт custom-metric
  callable, проверка сигнатуры, определение направления scorer-а и коллизии имён;
- `validation_strategy="auto"`, проверки `n_samples`, достаточности данных,
  пустого/константного target, NaN/inf-политики, datetime-политики, ID-like
  колонок;
- доменные проверки target transform и GLM target-domain guard;
- доступность ресурсов, `multiprocessing` start method, timeouts, CPU count;
- реальное поведение препроцессинга, feature selection, oversampling, HPO,
  sanity gate и оркестрации.

### 10.3. Правило разрезания `config_parser` и `training_engine/config/`

Целевое разделение:

1. `automl_contracts/config/` содержит чистые Pydantic-модели и чистые
   cross-field правила. `Config` не читает YAML, не проверяет установленные
   пакеты, не обращается к sklearn и не импортирует `models`.
2. `configurable_automl_engine.training_engine.config_parser` остаётся
   документированной публичной точкой импорта. Он:
   - импортирует и реэкспортирует контрактные модели;
   - сохраняет `read_config` на `PyYAML`: YAML-разбор → структурная
     валидация контрактной `Config` → `validate_runtime_config` из
     `config_runtime`. Публичное поведение `read_config` не меняется — он
     по-прежнему возвращает конфигурацию с выполненными runtime-проверками.
     Это единственный владелец runtime-валидации для файлового ввода;
   - сохраняет `SanityGateMode`, `REMOVED_ALGORITHM_KEYS` и другие
     engine-имена.
3. `configurable_automl_engine.training_engine.config_runtime` (новый
   внутренний модуль) содержит `validate_runtime_config(cfg)` и runtime-помощники,
   включая бывшие методы `AlgoCfg.get_required_package` и
   `AlgoCfg.get_unknown_hyperparameters`.
4. `configurable_automl_engine.training_engine.config/*` становится набором
   тонких модулей-шимов: контрактные имена реэкспортируются из
   `automl_contracts`, engine-only имена остаются в движке. Вторых определений
   моделей нет.
5. `train_best_model` и `AutoMLExperiment` — публичные entry point, которые
   принимают `str | Path | Config | dict` и проходят через шаг `load_config`
   (`read_config`/`Config.model_validate`). Runtime-валидация применяется к
   каждой конфигурации ровно один раз, у каждого вида ввода свой владелец:

   - `str`/`Path` — `read_config` (YAML-разбор, структурная валидация,
     `validate_runtime_config`); `load_config` полученный `Config` повторно не
     валидирует;
   - `dict` — `load_config`: `Config.model_validate` (структурная проверка),
     затем `validate_runtime_config`;
   - готовый `Config` — `load_config`: `validate_runtime_config` перед
     обучением, потому что контрактная `Config` runtime-проверок не содержит.

   `optimize` и `ModelTrainer` не принимают `Config`: у них runtime-проверки
   выполняются в собственной валидации аргументов, а
   `validate_runtime_config(cfg)` к ним не применяется. Структурные ошибки
   возникают при создании контрактной модели, runtime-ошибки конфигурации — в
   `read_config` для файла или при запуске `train_best_model`/`AutoMLExperiment`
   для остальных форм.

Прямые импорты из `training_engine.config.general` и аналогичных модулей,
используемые тестами, сохраняются через шимы на переходный период и помечаются
внутренними/deprecated в документации. C3 не ломает их молча.

### 10.4. `Protocol`

В контракт входят только устойчивые публичные интерфейсы без runtime-зависимостей:

- `MetricFunction` — `(y_true, y_pred, **kwargs) -> float`;
- `ScorerCallable` — `(estimator, X, y) -> float`.

Остаются в движке:

- `ReporterProtocol` — связан с engine-DTO `ReporterArtifact` и системой
  отчётов; в контракт не входит до отдельного решения;
- `SearchSpaceCallable` — зависит от runtime-объекта trial (Optuna);
- `PredictorProtocol` — sklearn-совместимый runtime-интерфейс;
- `ExperimentStoreProtocol` — engine API store.

Существующий `Reporter` ABC не заменяется на `Protocol`: это сломало бы
наследников. Если `ReporterProtocol` появится позже, он будет дополнительным
структурным интерфейсом.

## 11. Обратная совместимость

### 11.1. Общие правила

1. Старый публичный путь из матрицы `CONTRACTS.md` §5.1 остаётся рабочим.
2. C3 не создаёт второй класс с тем же именем: только реэкспорт одного
   определения из `automl_contracts`.
3. Для каждого реэкспорта класса/enum-а/исключения тест проверяет `is`; для
   type alias и констант — равенство значений/типов.
4. Несовместимое изменение (удаление пути, изменение типа или семантики)
   допускается только с явным обоснованием в ADR и `CHANGELOG.md` и повышением
   версии по политике SemVer.
5. `read_config`, HPO- и runtime-функции не обязаны появляться в
   `automl_contracts`; их старые пути сохраняются в движке.

### 11.2. Сохраняемые пути

Через реэкспорт сохраняются как минимум:

- `configurable_automl_engine` (корень): `Config`, `AlgorithmRun`,
  `ExperimentResult`, `CustomMetricSpec`, `TrainingError`;
- `configurable_automl_engine.common.definitions` — контрактные enum-ы и
  `WORST_SCORE_THRESHOLD`; `ALGO_PACKAGE_MAPPING` остаётся в движке;
- `configurable_automl_engine.common.errors` — `TrainingError`,
  `InvalidPredictionError`;
- `configurable_automl_engine.training_engine.config_parser` — все имена из
  `__all__` и `noqa`-реэкспорты;
- `configurable_automl_engine.training_engine.config.*` — шимы для
  контрактных имён и engine-only имён;
- `configurable_automl_engine.feature_selection` — контрактные config-типы
  реэкспортируются, `FeatureSelector` остаётся в движке;
- `configurable_automl_engine.preprocessing_presets`;
- `configurable_automl_engine.outliers_presets`;
- `configurable_automl_engine.hpo.default_spaces` — distribution-модели;
- `configurable_automl_engine.experiment.result` и
  `configurable_automl_engine.experiment`;
- `configurable_automl_engine.models` — `AVAILABLE_ALGORITHMS`;
- `configurable_automl_engine.common.metrics` — `AVAILABLE_METRICS`,
  `to_sklearn_name`, `ERR_CUSTOM_NAME_INVALID`;
- `configurable_automl_engine.common.custom_metrics`;
- `configurable_automl_engine.trainer` и
  `configurable_automl_engine.trainer_params` — `TrainingError`;
- `configurable_automl_engine.tuner` — `WORST_SCORE_THRESHOLD`;
- `configurable_automl_engine.preprocessing` — `EncodingStrategy`.

Полный перечень и проверки — в `CONTRACTS.md` §5.1–5.4. Он остаётся
обязательным для C3 и C5.

### 11.3. Осознанные изменения поведения

Перенос в контракт меняет место некоторых проверок. Эти изменения должны быть
описаны в `CHANGELOG.md` контракта и проверены в C5:

- `CustomMetricSpec` в контракте валидирует структуру полей (`path`, `kind`,
  `greater_is_better`, `additive`, `kwargs`), но не импортирует callable и не
  проверяет его сигнатуру. Runtime-проверка выполняется движком на входе в
  `train_best_model`, `optimize` и `ModelTrainer` через их собственные
  валидаторы аргументов, а не через `validate_runtime_config(cfg)`. Прямой
  конструктор контрактного класса может принять некорректный callable, но
  публичные entry point движка по-прежнему отвергают его до обучения.
- `ExperimentResult` принимает стандартные числа. Нормализация numpy-скаляров
  выполняется движком на границе; прямой конструктор с numpy-скаляром может
  изменить поведение.
- `AlgoCfg.get_required_package` и `AlgoCfg.get_unknown_hyperparameters` —
  runtime-методы, а не часть контракта. Их заменяют функции
  `config_runtime`; это изменение класса `AlgoCfg` относится к 0.x MINOR и
  документируется.
- `read_config` остаётся в движке из-за `PyYAML`; контракт предоставляет
  `Config.model_validate`. Публичное поведение `read_config` не меняется: он
  выполняет YAML-разбор, структурную валидацию контрактной `Config` и
  runtime-валидацию через `validate_runtime_config`, то есть по-прежнему
  возвращает конфигурацию с проверенными зависимостями алгоритмов,
  совместимостью гиперпараметров и нормализованными типами диапазонов.
- Меняется только место runtime-проверок: три бывших `model_validator`-а
  контрактной `Config` (`_check_algorithm_dependencies`,
  `_check_hyperparameter_compatibility`, `_normalize_hyperparameter_types`)
  в контракт не переносятся. Прямой вызов `Config.model_validate(dict)` или
  конструктора контрактного класса выполняет только структурную валидацию.
  Публичные entry point движка (`train_best_model`, `AutoMLExperiment`) и
  `read_config` по-прежнему выполняют runtime-проверки — ровно один раз на
  конфигурацию, по правилу §10.3; это фиксируется в `CHANGELOG.md` и в
  docstring `automl_contracts.config.Config`.
- `SanityGateMode`, `ALGO_PACKAGE_MAPPING`, `DEFAULT_SPACES`,
  `PRESET_BY_CLASS`, `ALGORITHM_CLASS_MAPPING`, `REMOVED_ALGORITHM_KEYS` и
  runtime-помощники остаются в движке.

Это единственные запланированные семантические изменения. Если C3 обнаружит
ещё одно, оно оформляется отдельным решением до реализации.

### 11.4. Deprecation

- `ScalingType` сохраняется как deprecated alias в
  `automl_contracts.preprocessing` и в старом engine-пути.
- Прямые импорты из `training_engine.config.*` сохраняются через шимы и
  помечаются внутренними/deprecated в документации.
- Удаление deprecated-пути допускается не раньше следующего major-релиза
  контракта и после периода предупреждения; для движка — по его собственной
  политике версий.
- Deprecation фиксируется в `CHANGELOG.md` с указанием замены.

## 12. JSON Schema

`config.schema.json` — производный артефакт контрактных моделей, а не
независимый источник истины.

- **Владелец:** дистрибутив `automl-contracts`.
- **Источник истины:** `automl_contracts.config.Config` и вложенные модели.
- **Канонический файл:**
  `automl-contracts/src/automl_contracts/schemas/config.schema.json`.
- **Генератор:** `automl-contracts/scripts/generate_schema.py` с явным
  `main()`/CLI. Импорт генератора не пишет файл; побочных эффектов нет.
- **Генерация:** `Config.model_json_schema()` с детерминированной очисткой
  описаний (`inspect.cleandoc`), `indent=2`, UTF-8, завершающий перевод строки,
  без временных меток и абсолютных путей.
- **Package data:** файл входит в wheel контракта как
  `automl_contracts/schemas/config.schema.json` и доступен через
  `importlib.resources.files("automl_contracts")`.
- **Корневая копия:** `config.schema.json` в корне остаётся совместимым
  зеркалом для существующих потребителей и генерируется из канонической схемы.
  Он не редактируется вручную и не является вторым источником истины.
- **CI:** генерация во временный файл и сравнение с сохранённым каноническим
  файлом; расхождение — ошибка. Отдельно проверяется наличие файла в wheel.
- **Совместимость:** изменение схемы классифицируется по влиянию на
  производителей и потребителей, а не по факту изменения файла.

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

## 13. Версионирование

### 13.1. Независимый SemVer контракта

`automl-contracts` использует собственную версию SemVer, независимую от версии
`configurable-automl-engine`.

Стартовая версия: **`0.1.0`**.

Способ задания версии: **статическая версия в
`automl-contracts/pyproject.toml`**. `setuptools-scm` для контракта не
используется.

Причины выбора статической версии:

- независимый релизный цикл не должен зависеть от тегов движка;
- теги `contracts-v*` и `v*` не конкурируют за одну SCM-версию;
- версия видна в `pyproject.toml` без запуска сборки;
- CI может сравнить версию в `pyproject.toml` с тегом до публикации.

`automl_contracts.__version__` читается из метаданных установленного
дистрибутива. CI проверяет, что метаданные, `__version__` и тег
`contracts-v<version>` совпадают.

### 13.2. Правила PATCH/MINOR/MAJOR

Для версий `1.0.0` и выше:

- **PATCH** — обратно совместимые исправления, не меняющие публичное поведение,
  публичные типы и схему (кроме незначимых описаний);
- **MINOR** — обратно совместимое расширение публичного API;
- **MAJOR** — изменения, нарушающие обратную совместимость.

Ориентировочная классификация:

| Изменение | Классификация после `1.0.0` |
|---|---|
| Исправление ошибки без изменения публичного поведения | PATCH |
| Новое необязательное поле с безопасным default | MINOR, если не ломает сериализацию |
| Новый публичный тип или `Protocol` | MINOR |
| Новое значение enum/закрытого `Literal` | MINOR или MAJOR по фактическому влиянию на потребителей |
| Новое обязательное поле | MAJOR |
| Удаление публичного типа, поля, метода или значения | MAJOR |
| Изменение типа или семантики публичного поля/метода | MAJOR |
| Удаление старого публичного пути импорта | MAJOR |
| Изменение JSON Schema | По фактическому влиянию, а не только по факту изменения файла |

### 13.3. Политика `0.x`

До `1.0.0` полная стабильность API не обещается:

- потенциально несовместимые изменения получают повышение MINOR;
- PATCH используется только для строго совместимых исправлений;
- несовместимое изменение нельзя маскировать под PATCH;
- каждое изменение публичного API и схемы описывается в `CHANGELOG.md`.

### 13.4. `CHANGELOG.md`

Контракт ведёт `automl-contracts/CHANGELOG.md` в стиле Keep a Changelog:

- раздел `Unreleased`;
- разделы `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`;
- для каждого релиза — версия и дата;
- изменения публичного API и JSON Schema отражаются явно;
- несовместимые изменения помечаются как `BREAKING` и содержат обоснование и
  инструкцию миграции.

Changelog движка не смешивается с changelog контракта. Если движку понадобится
собственный файл, он будет отдельным.

### 13.5. Связь коммитов и тегов

- тег `contracts-v<version>` относится только к `automl-contracts`;
- тег `v<version>` относится только к `configurable-automl-engine`;
- `setuptools-scm` движка настраивается на `^v...`, поэтому теги контракта не
  влияют на версию движка;
- публикация контракта не запускается на тег движка и наоборот;
- коммит, затронувший оба дистрибутива, может быть один; релизные теги ставятся
  на него раздельно и в нужном порядке.

## 14. Матрица совместимости версий

Основной пакет объявляет диапазон версий контракта средствами Python
packaging:

```toml
dependencies = [
    "automl-contracts>=0.1.0,<0.2.0",
    # остальные зависимости движка
]
```

Начальный диапазон `>=0.1.0,<0.2.0` выбран потому, что до `1.0.0` MINOR
контракта может содержать несовместимое изменение. Движок не должен
автоматически принимать `0.2.0` без проверки.

Поддерживаемая матрица:

| Контракт | Движок | Статус |
|---|---|---|
| `0.1.x` | версия движка, в которой объявлен `>=0.1.0,<0.2.0` | поддерживается и проверяется |
| `<0.1.0` | — | не поддерживается |
| `>=0.2.0,<1.0.0` | — | требует нового релиза движка с обновлённым диапазоном |
| `>=1.0.0,<2.0.0` | будущий движок с диапазоном `>=1.0.0,<2.0.0` | поддерживается после отдельного обновления политики |

Правила:

- CI движка проверяет нижнюю (`0.1.0`) и последнюю доступную версию из
  разрешённого диапазона (`0.1.*`);
- CI контракта проверяет минимальную и последнюю версию Pydantic v2
  (`contracts-pydantic-matrix`, §15.2);
- после `1.0.0` контракта диапазон движка становится `>=1.0.0,<2.0.0`, а
  матрица покрывает все поддерживаемые minor-версии контракта;
- сочетания вне объявленного диапазона не считаются поддерживаемыми, даже если
  импорт случайно работает;
- изменение диапазона в движке — отдельное изменение с записью в changelog
  движка и проверкой в CI.

Совместное изменение контракта и движка в одном PR обновляет диапазон и обе
сборки атомарно.

## 15. Тесты и CI

### 15.1. Разделение тестов

- `tests/contracts/` — тесты только контракта. Они не импортируют
  `configurable_automl_engine`, `numpy`, `pandas`, `sklearn`, `optuna`,
  `yaml` и другие тяжёлые пакеты.
- `tests/engine/` — существующие тесты движка и тесты обратной совместимости.
  Они могут импортировать оба пакета.
- Общие фикстуры движка не используются контрактными тестами.
- Тесты контракта покрывают:
  - импорт и `__all__`;
  - стабильные публичные подмодули;
  - валидацию моделей и коды ошибок;
  - `Protocol`-совместимость;
  - JSON Schema и её воспроизводимость;
  - `py.typed` и package data;
  - отсутствие запрещённых зависимостей и обратных импортов.

### 15.2. Независимые CI-сценарии

Для `automl-contracts`:

1. `contracts-lint-type-test`: из корня репозитория установка контракта без
   движка (`pip install -e automl-contracts`), `ruff` и `mypy` по
   `automl-contracts/src` и `tests/contracts`, `pytest tests/contracts` из
   корня репозитория (путь `tests/contracts` задан относительно корня), затем
   `lint-imports` с `working-directory: automl-contracts`, чтобы применялась
   конфигурация контракта, а не корневая (см. §9.3).
2. `contracts-pydantic-matrix`: та же установка контракта без движка и
   `pytest tests/contracts` из корня репозитория, но в матрице по Pydantic:
   `2.12.5` (нижняя граница) и последняя версия из диапазона
   `>=2.12.5,<3` (резолвится `pip install "pydantic<3"`). Джоб также повторно
   генерирует JSON Schema и сравнивает с каноническим файлом на обеих
   версиях, чтобы поддержка диапазона была доказана, а не только заявлена.
   При изменении нижней границы (§8.3) матрица обновляется в том же PR.
3. `contracts-build`: `python -m build automl-contracts`, проверка wheel/sdist,
   наличия `py.typed` и `schemas/config.schema.json`.
4. `contracts-schema`: генерация схемы и сравнение с каноническим файлом.
5. `contracts-isolated-install`: установка wheel в чистый venv без движка,
   проверка импорта, метаданных и отсутствия тяжёлых зависимостей.
6. `contracts-publish`: запускается только по тегу `contracts-v*`.

Для `configurable-automl-engine`:

1. `engine-lint-type-test`: из корня репозитория установка контракта из
   исходников, затем движка; `ruff`, `mypy`, `pytest tests/engine`,
   `lint-imports` с корневой конфигурацией движка (без смены
   `working-directory`).
2. `engine-build`: сборка wheel/sdist движка, установка с контрактом, smoke-тест
   публичного API.
3. `engine-backward-compat`: старые пути импорта и проверки идентичности
   классов/enum-ов.
4. `engine-matrix`: проверка нижней и последней поддерживаемой версии контракта.
5. `engine-publish`: существующий релизный цикл движка, переведённый на
   SourceCraft Registry в C7.

### 15.3. Правило публикации по тегам

- `contracts-v*` запускает только сборку, проверку и публикацию контракта.
- `v*` запускает только релизный цикл движка.
- Перед публикацией CI проверяет:
  - тег соответствует версии в `pyproject.toml`;
  - сборка выполнена из чистого checkout;
  - все обязательные проверки прошли;
  - публикуется только один дистрибутив.
- Публикация не запускается из ветки или PR.

### 15.4. Совместные изменения

Один PR может менять контракт и движок. В нём:

- оба дистрибутива собираются независимо;
- контрактные тесты не требуют движка;
- engine-тесты устанавливают контракт из исходников;
- проверяется новая версия/диапазон;
- версия контракта повышается по SemVer;
- диапазон в `pyproject.toml` движка обновляется в том же PR;
- публикация выполняется после мержа и установки тегов, а не из PR.

## 16. Публикация

Канонический реестр для контракта — **PyPI-реестр SourceCraft
`automl-packages`** организации `ooo-ainod`.

Параметры:

- upload endpoint:
  `https://pkg.sourcecraft.tech/pypi/ooo-ainod/${SOURCECRAFT_PYPI_REGISTRY_ID}/legacy/`;
- install endpoint:
  `https://pkg.sourcecraft.tech/pypi/ooo-ainod/${SOURCECRAFT_PYPI_REGISTRY_ID}/simple/`;
- идентификатор реестра передаётся через переменную/секрет
  `SOURCECRAFT_PYPI_REGISTRY_ID`;
- PAT передаётся через защищённый секрет `PYPI_REGISTRY_PAT`;
- публикация выполняется через `twine` в доверенном CI-контексте.

Требования безопасности:

- PAT не хранится в репозитории, не выводится в логи и не попадает в артефакты;
- права PAT минимальны и достаточны только для публикации пакета;
- публикация разрешена только из доверенного CI по тегу;
- перед публикацией проверяется версия и состав артефактов.

Порядок:

1. CI собирает и проверяет `automl-contracts` из тега `contracts-v<version>`.
2. `twine upload` отправляет wheel и sdist в `automl-packages`.
3. E2E-проверка (C7) устанавливает контракт из реестра в чистый проект без
   движка и проверяет импорт, модели, `Protocol` и JSON Schema.
4. Для CI монорепозитория основной путь — установка обоих пакетов из исходников;
   реестр используется для внешних потребителей и E2E-проверки.

Движок в C7 также переводится на публикацию в `automl-packages`. До завершения
C7 существующий релизный цикл движка остаётся переходным; контракт от него не
зависит. Если движок временно публикуется вне SourceCraft Registry, документация
обязана указывать дополнительный индекс для установки `automl-contracts`.

## 17. Риски и граничные случаи

### 17.1. Динамические `Literal` и наборы значений

Сейчас `Literal[*AVAILABLE_METRICS]` в коде нет: `ComparisonMetric = str`, а
`AVAILABLE_METRICS` — вычисляемый список. В контракте запрещено строить
`Literal` из значений, импортированных из движка.

Правило:

- канонические `AVAILABLE_METRICS` и `AVAILABLE_ALGORITHMS` определяются
  статически в контракте;
- движок импортирует их из контракта и реэкспортирует старые имена;
- `ComparisonMetric` остаётся `str`, потому что custom-имена — часть публичного
  контракта;
- если в будущем понадобится закрытый `Literal`, его значения определяются в
  контракте, а расширение набора считается потенциально ломающим изменением;
- JSON Schema генерируется из контрактных моделей; импорт движка для генерации
  не используется.

`AVAILABLE_METRICS` сохраняется как `list`, а `AVAILABLE_ALGORITHMS` — как
`list` для совместимости с текущим типом. Оба списка документируются как
read-only: движок и потребители не должны их мутировать. Переход на `tuple`
возможен только отдельным решением с повышением версии.

`global_nrmse` остаётся специальным динамическим scorer-ом движка и не входит в
`AVAILABLE_METRICS`.

### 17.2. `config.schema.json`

- владелец — контракт;
- канонический файл — в wheel контракта;
- корневая копия — только совместимое зеркало;
- генератор — явный скрипт без побочных эффектов при импорте;
- CI сравнивает сгенерированную схему с сохранённой;
- изменение схемы анализируется на совместимость.

### 17.3. Тесты, завязанные на `config_parser`

- Старые импорты из `training_engine.config_parser` и
  `training_engine.config.*` сохраняются через шимы.
- Контрактные имена в шимах — только реэкспорты; engine-only имена остаются в
  движке.
- C3 обновляет тесты на контрактные пути там, где это не ломает сценарии, и
  сохраняет проверки старых путей в отдельном наборе.
- Проверка идентичности классов/enum-ов ловит случайное создание второй
  реализации.

### 17.4. Совместные изменения контракта и реализации

- Один PR, две независимые сборки.
- Версия контракта повышается по SemVer, диапазон движка обновляется в том же
  PR.
- CI проверяет контракт без движка и движок с контрактом из исходников.
- Публикация контракта выполняется до релиза движка, который на него
  ссылается.
- Атомарность обеспечивается merge одного PR, а не синхронизацией двух
  репозиториев.

### 17.5. Ассоциация коммитов, тегов и CI

- Префиксы тегов разделяют релизы.
- `setuptools-scm` движка игнорирует `contracts-v*`.
- CI-фильтры тегов не пересекаются.
- Коммит может относиться к обоим дистрибутивам; тег и публикация определяют
  конкретный артефакт.
- Изменение версии контракта без тега не публикуется.

### 17.6. Дублирование определений

Дублирование запрещено. Если символ нужен в движке, он импортируется из
`automl_contracts` и реэкспортируется. Второй класс, enum или исключение с тем
же именем не создаётся. Тесты идентичности обязательны.

### 17.7. Прочие риски

- `pydantic-core` как исключение: требует контроля версии и тестов кодов ошибок;
  при переходе на публичный API Pydantic исключение можно убрать.
- Изменения `CustomMetricSpec`, `ExperimentResult` и runtime-методов `AlgoCfg`
  требуют явного changelog и проверки публичных entry point.
- Sdist движка не должен содержать `automl-contracts`; это обеспечивается
  `MANIFEST.in` и package discovery.
- Editable-установка в CI должна ставить контракт до движка, иначе pip может
  попытаться разрешить зависимость из внешнего индекса.
- Внешние потребители не были доступны в C1; C5/C6 обязаны проверить
  downstream-репозитории и дополнить матрицу совместимости.

## 18. Последствия

Положительные:

- контракт устанавливается без `scikit-learn`, `pandas`, `numpy`, `optuna` и
  `PyYAML`;
- один источник истины и атомарные PR;
- независимые версии и публикация;
- старые пути импорта сохраняются через реэкспорты;
- JSON Schema имеет явного владельца и воспроизводимую генерацию.

Отрицательные:

- две сборочные конфигурации и два набора CI-проверок;
- нужна дисциплина тегов и релизного порядка;
- временная асимметрия раскладки (движок в корне, контракт в подкаталоге);
- часть runtime-проверок меняет место возникновения ошибок;
- появляются шимы и deprecated-пути на переходный период.

Нейтральные:

- раскладку движка можно симметрично перенести под `distributions/` отдельным
  ADR после стабилизации контракта;
- `pydantic-core` остаётся обоснованным исключением, пока нужны стабильные коды
  ошибок.

## 19. План миграции

- **C3:** создать `automl-contracts`, перенести определения из
  `CONTRACTS.md`, разрезать `config_parser`/`config`, добавить реэкспорты и
  шимы, разделить тесты.
- **C4:** перенести генератор схемы в контракт, включить `py.typed` и Schema в
  wheel, добавить CI-проверку воспроизводимости.
- **C5:** снимок публичного API, тесты идентичности, JSON round-trip,
  обратная совместимость старых путей.
- **C6:** обновить `API_REFERENCE.md`/`README.md`, зафиксировать deprecation и
  оставшиеся пограничные DTO.
- **C7:** публикация контракта и движка в `automl-packages`, E2E-установка из
  реестра, документация индекса.

## 20. Соответствие критериям приёмки

| Критерий | Раздел |
|---|---|
| ADR добавлен, все пункты зафиксированы | весь документ |
| Выбор варианта обоснован | 3.1–3.4 |
| Структура каталогов и сборка | 5, 6 |
| Публичная точка входа, `py.typed`, Schema в wheel | 7, 12 |
| Зависимости, Python/Pydantic | 8 |
| Направление импортов и граница валидации | 9, 10 |
| Обратная совместимость, реэкспорты, SemVer, матрица | 11, 13, 14 |
| Финальный список символов соответствует `CONTRACTS.md` | 21 |
| Имя дистрибутива, версионирование, публикация | 4, 13, 16 |

## 21. Финальный список переносимых символов

Список соответствует `CONTRACTS.md` §§4.1–4.9. Публичные символы с решением
`move`/`move + re-export` перечислены ниже; внутренние символы, которые
переносятся как `internal`, — в разделе 21.1; символы с решением `internal`,
которые остаются в движке, — в списке после таблиц. Решения по строкам
`needs-decision` зафиксированы в этом ADR: `InvalidPredictionError` — public
beta, `ReporterProtocol` и `SearchSpaceCallable` — engine-only,
`CustomMetricSpec` — структура в контракте, runtime-проверка в движке.

| Модуль контракта | Переносимые символы |
|---|---|
| `automl_contracts.definitions` | `ValidationStrategy`, `EncodingStrategy`, `SerializationFormat`, `TargetTransform`, `ParallelStrategy`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection`, `WORST_SCORE_THRESHOLD`, `AVAILABLE_ALGORITHMS` |
| `automl_contracts.errors` | `TrainingError`, `InvalidPredictionError` |
| `automl_contracts.metrics` | `AVAILABLE_METRICS` |
| `automl_contracts.custom_metrics` | `CustomMetricKind`, `CustomMetricSpec`, `ERR_CUSTOM_NAME_INVALID` |
| `automl_contracts.protocols` | `MetricFunction`, `ScorerCallable` |
| `automl_contracts.search_space` | `SearchSpaceEntry`, `BaseDistribution`, `CategoricalSpace`, `NumericSpace`, `FloatSpace`, `IntSpace` |
| `automl_contracts.feature_selection` | `FeatureSelectionMode`, `FeatureSelectionMethod`, `FeatureSelectionCfg` |
| `automl_contracts.preprocessing` | `ImputationStrategy`, `ScalingType` (deprecated), `StandardScaling`, `RobustScaling`, `QuantileScaling`, `NoScaling`, `ObjectScalingSpec`, `StringScaling`, `ScalingSpec`, `AlgorithmClass`, `PreprocessingPreset`, `PreprocessingOverride`, `merge_preprocessing_overrides`, `NoOutliers`, `QuantileOutliers`, `IqrOutliers`, `ObjectOutlierSpec`, `StringOutlierSpec`, `OutlierSpec`, `NO_OUTLIERS` |
| `automl_contracts.result` | `AlgorithmRun`, `ExperimentResult` |
| `automl_contracts.config` | `Config`, `GeneralCfg`, `HPOPhaseCfg`, `SanityGateCfg`, `ComplexityGuardCfg`, `EnsembleCfg`, `AlgoCfg`, `OversamplingCfg`, `AlgorithmsConfig` |
| `automl_contracts.config.general` | `GeneralCfg`, `ComparisonMetric`, `AUTO_HOLDOUT_FRACTION`, `AUTO_HOLDOUT_MIN_ROWS`, `InfPolicy`, `DatetimePolicy`, `SplitAggregation` |
| `automl_contracts.config.phases` | `HPOPhaseAction`, `SelectStrategy`, `SelectOutput`, `PruningCfg`, `PruningStrategy`, `HPOPhaseCfg`, `DEFAULT_PHASE_N_TRIALS`, `ERR_REMOVED_REFINE_WINNER` |
| `automl_contracts.config.sanity_gate` | `SanityGateCfg`, `CorridorMode`, `GeneralizationGapMode`, `ERR_REMOVED_SANITY_GATE_MODE` |
| `automl_contracts.config.complexity_guard` | `ComplexityGuardCfg`, `ComplexityGuardMode`, `ComplexityGuardUserRangePolicy`, `ComplexityGuardSizeBasis` |
| `automl_contracts.config.ensemble` | `EnsembleCfg`, `EnsembleWeightMethod` |
| `automl_contracts.config.algorithms` | `AlgoCfg`, `OversamplingCfg`, `OversamplingAlgorithm`, `AlgorithmsConfig`, `ERR_REMOVED_PLUGIN_KEY` |

### 21.1. Внутренние символы, переносимые в контракт

Эти символы не входят в публичный API и не попадают в корневой `__all__`, но
их определения должны жить в контракте, чтобы у моделей был один источник
истины.

| Модуль контракта | Внутренние символы |
|---|---|
| `automl_contracts.config.general_rules` | `check_custom_metric_names`, `check_metric_references`, `deduplicate_additional_metrics`, `exclude_comparison_from_additional`, `check_multi_split`, `check_pruning_consistency`, `validate_phase_pipeline`, `check_high_cardinality_encoding`, структурная часть `check_n_folds`, константы `MULTI_SPLIT_*` |
| `automl_contracts.config.phases` | `_default_phases` |
| `automl_contracts.config.algorithms` | `AlgorithmsConfigType` |
| `automl_contracts.preprocessing` | `coerce_scaling`, `scaling_type`, `coerce_outliers`, `format_outliers` |
| `automl_contracts.result` | `ScoreValue`, `ScoreValueOrNone` |
| `automl_contracts.metrics` | `to_sklearn_name` |

В контракт **не переносятся**:

- `read_config` — остаётся в движке из-за `PyYAML`;
- `ALGO_PACKAGE_MAPPING`, `SanityGateMode`, `REMOVED_ALGORITHM_KEYS`;
- `DEFAULT_SPACES`, `PRESET_BY_CLASS`, `ALGORITHM_CLASS_MAPPING`, `GLM_COMMON`,
  `ALGO_HYPERPARAMETER_REGISTRY`, `DATA_DEPENDENT_CONSTRAINTS`;
- runtime-функции и константы HPO, препроцессинга, oversampling, отчётов и
  store;
- runtime-часть `check_n_folds` (проверка `joblib`), `check_start_method`,
  `check_threads_timeouts`;
- `to_sklearn_name` как публичное имя верхнего уровня (только внутренний helper
  контракта и старый engine-путь);
- `ERR_CUSTOM_NAME_COLLISION` и остальные runtime-коды custom metrics.

Если C3 обнаружит расхождение с этой таблицей, сначала обновляется
`CONTRACTS.md` и это ADR, затем реализация. Молчаливое добавление или удаление
переносимого символа не допускается.
