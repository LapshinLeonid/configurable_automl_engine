"""Hyperparameter optimisation (HPO) engine.

The implementation behind ``configurable_automl_engine.tuner.optimize``:
``types`` (exceptions, ``OptimizeResult``, sentinel constants), ``request``
(``HpoRequest`` — the settings of one run), ``plans`` (the results of the
pure setup steps), ``search_space``, ``evaluation``, ``pruning``,
``trial_pipeline`` and ``runner`` (``_OptimizeRunner``), plus the search space
building blocks ``default_spaces`` (the default search spaces, clipping and
the range DSL), ``complexity_guard`` and ``target_domain``. The public entry
point and the documented import path stay ``configurable_automl_engine.tuner``.
The package ``__init__`` imports nothing, so importing a single submodule
stays cheap (``config_parser`` imports ``default_spaces`` without loading
Optuna), and no module of the package may import
``configurable_automl_engine.tuner`` or the trainer modules.
"""
