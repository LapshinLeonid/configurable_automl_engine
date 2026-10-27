"""Hyperparameter optimisation (HPO) engine.

The implementation behind ``configurable_automl_engine.tuner.optimize``:
``types`` (exceptions, ``OptimizeResult``, sentinel constants),
``search_space``, ``evaluation``, ``pruning``, ``trial_pipeline`` and
``runner`` (``_OptimizeRunner``). The public entry point and the documented
import path stay ``configurable_automl_engine.tuner``. The package
``__init__`` imports nothing, so importing a single submodule stays cheap,
and no module of the package may import ``configurable_automl_engine.tuner``.
"""
