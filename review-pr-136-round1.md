# Review: PR 136 — T10: ensemble — комбинирование топ-K моделей из leaderboard

Дата: 2026-11-06. Ревьюер: codereview (AI). Раунд 1.

## Верификация (воспроизведено локально на Python 3.11, ветка t10-ensemble @ 3078d66)

| Проверка | Результат |
|---|---|
| `ruff check src/` | 0 ошибок |
| `ruff format --check src/` | чисто (98 файлов) |
| `mypy src` | 0 ошибок в файлах PR; 6 ошибок окружения (`import-untyped`: scipy, yaml) в файлах, которые PR не меняет — требуется `scipy-stubs`/`types-PyYAML` в dev-зависимостях (не блокирует PR, см. N-2) |
| `pytest` (полный набор) | 4921 passed, 2 skipped |
| `lint-imports` | 4 контракта KEPT |
| Новые тесты | `tests/test_ensemble.py`: 16 passed |

## Оценка по чек-листу

- **Функциональность:** требования T10 выполнены. Ядро (`select_top_k`, `compute_weights`, `EnsembleModel`) корректно переиспользует документированную семантику select-шагов; инвариант winner-скора (`validate_ensemble_score`) последовательно проведён через все входы весов и топ-K; best-effort контракт шага (сбой никогда не маскирует победителя) выдержан на всех путях, включая `monkeypatch`-тест N4. `min(k, n)` с RuntimeWarning, K=1, fallback равных весов при неположительных R² — всё покрыто.
- **Тесты:** хорошее соотношение позитив/негатив/границы (P1–P5, N1–N4, B1–B3), включая полный путь config → engine → result → facade → artifact и JSON round-trip секции.
- **Типизация:** строгая, `Any` только на границах предикторов/конфига, как и в остальном коде.
- **Docstrings:** Google-style на английском, полные, с Raises-секциями — за исключением одного места, где контракт задокументирован, но не enforced (см. BLOCKER-1).
- **Производительность/безопасность:** ретрейн каждого члена ровно один раз; живые ссылки на предикторы задокументированы как стоимость памяти; pickle-артефакт с маркером и версией payload; чужой pickle отклоняется с понятной ошибкой.

## Замечания

### BLOCKER-1 — `EnsembleModel.__init__`: контракт «веса положительны» не enforced (tests/test_ensemble.py — недостающий негативный сценарий)

`src/configurable_automl_engine/experiment/ensemble.py:350` (docstring «Raises: ... a non-positive/non-finite weight») и `EnsembleMember` (:307, «weights of all members sum to 1», «positive»). Фактическая проверка — только на **сумму** (:368–372):

```python
total = sum(member.weight for member in members)
if not math.isfinite(total) or total <= 0.0:
    raise ValueError(...)
```

Конструктор класса с отрицательным весом у одного члена (например, веса `[2.0, -1.0]`, сумма = 1.0) проходит без ошибки: нормализованный вес `-1.0` попадает в `members`, в `report_section()` (секция обещает «positive; the weights of all members sum to 1»), в Markdown-таблицу и в сериализованный артефакт. Двигатель всегда передаёт уже нормализованные положительные веса из `compute_weights`, поэтому боевой путь не задет — но модуль экспортирован в публичный API (`experiment/__init__.py`), и документированный инвариант нарушается молча.

**Что сделать:** добавить per-member проверку (`math.isfinite(member.weight) and member.weight > 0`) перед проверкой суммы + негативный тест в `test_ensemble.py` (в духе N4: `EnsembleModel(..., members=[EnsembleMember("a", 1.0, 2.0, 1), EnsembleMember("b", 1.0, -1.0, 2)], ...)` → `ValueError`).

### MAJOR-1 — `top_k_from_leaderboard`: ретрейн члена по snapshot-параметрам может отличаться от артефакта победителя

`src/configurable_automl_engine/training_engine/ensemble_step.py:85–92, 167–181`. Если tie-break терминального select и снапшота разошёлся и победитель не вошёл в топ-K, он дописывается **последним** членом с рангом K. Против documented-инварианта «ранги — порядок лучшести» это выглядит противоречиво: победитель лучшая модель рана, но стоит последней строкой таблицы отчёта и получает наименьший ранг.

Также при K=1 комментарий «K=1 keeps the single best member» — а при K>1 порядок добавления победителя в конец меняет и веса (при `score_weighted` вес не зависит от позиции, но ранг в отчёте — да). Рекомендация: вставлять победителя на его место по скору (`bisect` по отсортированному списку) и перенумеровывать ранги, либо явно задокументировать в docstring `top_k_from_leaderboard`, что добавленный победитель получает последний ранг и почему это допустимо.

### MAJOR-2 — `build_ensemble_from_context`: дрейф kwargs к `_fit_finalist`

`src/configurable_automl_engine/training_engine/ensemble_step.py:167–181` — единственное место ручного дублирования сигнатуры `_fit_finalist` (validation_strategy/n_folds/test_size/random_state/datetime_expansion). Остальные call-сайты (finalists, audit) собирают эти параметры из `prepared` централизованно. При добавлении нового обязательного параметра в `_fit_finalist` mypy это поймает, но при изменении *семантики* резолва (например, иной holdout-политики) ансамбль молча откатится к другой валидации. Рекомендация: выделить общий помощник сборки kwargs (либо передавать `prepared` в `_fit_finalist`) в отдельном PR-следования.

### MINOR-1 — `_build_and_attach_ensemble`: ветка «секция без артефакта» приводит к ошибке фасада

`src/configurable_automl_engine/training_engine/component.py:876–880`: при неудаче `ensemble.save()` секция не добавляется — консистентно. Но обратная ветка, когда `build_ensemble_from_context` вернул модель, а сериализация упала, логируется WARNING-ом без упоминания, что фасад `ensemble_model()` в этом ране недоступен. Пользователь, читающий только логи, может не связать «artifact could not be saved» с последующим `RuntimeError: No ensemble available`. Рекомендация: дополнить сообщение WARNING-а подсказкой («the result carries no ensemble section; ensemble_model() will raise»).

### MINOR-2 — `_build_and_attach_ensemble`: fallback `result.get("model_path", cfg.general.path_to_model)`

`src/configurable_automl_engine/training_engine/component.py:872`. `persist_artifact` всегда кладёт `"model_path"` в result (artifact.py:510) и сам применяет `model_path_override`; fallback на `cfg.general.path_to_model` — недостижимая защитная ветка, которая при неожиданном расхождении записала бы ансамбль не рядом с артефактом победителя. Лучше явная защита: `if "model_path" not in result: skip with WARNING`.

### MINOR-3 — `_render_ensemble`: ранний return после заголовка

`src/configurable_automl_engine/experiment/reporting/markdown_reporter.py:286–290`: при `members == []` выводится висячий заголовок `## Ensemble (experimental)` без содержимого. На практике недостижимо (валидатор секции + сборка в движке всегда кладут непустой список), но тривиально исправляется перестановкой проверки до `lines.append`.

### N-1 — Документация пользователя

`general.ensemble` добавлен в `config.schema.json` (описания полные), но в `API_REFERENCE.md` новой секции нет — при том, что смежные экспериментальные блоки (`complexity_guard`, `sanity_gate`, `feature_selection`) там документированы отдельно. Рекомендуется дополнить (можно следующим PR).

### N-2 — Окружение mypy (вне PR)

`mypy src` вне подготовленного CI-окружения падает на `import-untyped` для `scipy`/`yaml` в файлах, не затронутых PR. Стоит добавить `scipy-stubs` и `types-PyYAML` в dev-зависимости для воспроизводимости локальной проверки.

## Итог

Реализация качественная и хорошо протестированная, замечания MINOR/N не блокируют. BLOCKER-1 (не enforced документированный инвариант положительности весов в публичном конструкторе + отсутствующий негативный тест) предлагается устранить до мержа.

Резолюция: **block** (до устранения BLOCKER-1; после фикса — approve).
