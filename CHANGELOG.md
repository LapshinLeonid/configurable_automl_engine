# Changelog

## 0.5.0 — unreleased

Released as a minor version: results of existing direct `optimize()` calls can change (see "Changed").

### Fixed

- `tuner.optimize(algo, X, y)` without `space_overrides` no longer raises `HyperoptError: No search space ...` for `ridge`, `lasso`, `elasticnet`, `sgdregressor`, `ardregression`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `adaboost`, `xgboosting`, `poissonregressor`, `gammaregressor`, `tweedieregressor` and `glm`. Every algorithm with a `DEFAULT_SPACES` entry now searches the same default space as a config run without `hyperparameters`.
- The direct path and the engine path build the search space with one merge function (`hpo.search_space.build_search_space`), so a direct call and a config run with the same ranges search the same space.
- An algorithm without a default space and without `space_overrides` still raises `HyperoptError` (`No search space for algorithm '<name>' ...`); the message now says to pass `space_overrides={'<name>': {...}}`.

### Changed

- Direct `optimize()` calls: a partial dict override (`limit_hyperparameters=False`) is now merged with `DEFAULT_SPACES`, as in a config run. Before, only the listed parameters were searched; pass `limit_hyperparameters=True` for that behavior.
- Direct `optimize("nearest_neighbors_regression", ...)` without overrides now searches `n_neighbors` in 1–50 (clipped to `n_samples − 1`) instead of 1–30.
- Physical clipping (`clip_search_space`) and the complexity guard now apply to direct calls without overrides too (they already applied to dict overrides and to config runs).
- Direct calls without `user_hyperparameter_keys` treat the keys of the dict `space_overrides` entry as the user ranges of the complexity guard. The training engine now always passes `user_hyperparameter_keys` to the tuner (an empty list included).
