"""The engine config schema (Pydantic v2), one module per config section.

* :mod:`.phases` — the pipeline steps (``general.phases``) and pruning;
* :mod:`.sanity_gate` — ``general.sanity_gate``;
* :mod:`.complexity_guard` — ``general.complexity_guard``;
* :mod:`.general` — the ``general`` section (``GeneralCfg``, fields only)
  and :mod:`.general_rules` — its cross-field validation rules;
* :mod:`.algorithms` — the ``algorithms`` and ``oversampling`` sections;
* :mod:`.root` — the root ``Config`` and ``read_config``.

The public import path is ``training_engine.config_parser``, which
re-exports these names; this ``__init__`` re-exports nothing.
"""
