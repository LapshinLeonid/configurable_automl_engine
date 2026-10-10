from sklearn.datasets import load_diabetes

import configurable_automl_engine as caml

data = load_diabetes(as_frame=True)
df = data.frame

# NOTE (epic #74): the config below uses the target pipeline schema.
# The `select` steps are fully implemented (task R2); the `refine` action is
# declared in the config schema but its execution is not implemented yet
# (task R3): running this example raises NotImplementedError until then.
# Remove the "refine" phase (or replace it with another "all_algorithms"
# phase) to run the example today.

config = {
    "general": {
        "comparison_metric": "r2",
        "phases": [
            # 1. Coarse search: HPO for every enabled algorithm.
            {"name": "coarse", "n_trials": 100, "action": "all_algorithms"},
            # 2. Sanity Gate keeps the top-3 candidates that pass the audit.
            {
                "name": "gate_select",
                "n_trials": 1,
                "action": "select",
                "strategy": "sanity_gate",
                "output": "top_k",
                "k": 3,
            },
            # 3. Refine: warm-started HPO for every surviving candidate.
            {"name": "refine", "n_trials": 200, "action": "refine"},
            # 4. Terminal step: re-audit the refined models, pick the winner.
            {
                "name": "gate_final",
                "n_trials": 1,
                "action": "select",
                "strategy": "sanity_gate",
                "output": "winner",
            },
        ],
        "path_to_model": "diabetes_model.pkl",
    },
    "algorithms": {
        "random_forest": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"n_estimators": [10, 500], "max_depth": [3, 20]},
        },
        "ridge": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"alpha": [0.1, 1.0]},
        },
        "xgboosting": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {
                "n_estimators": [100, 1000],
                "max_depth": [3, 10],
                "learning_rate": [0.01, 0.3],
                "subsample": [0.5, 1.0],
            },
        },
    },
}

results = caml.train_best_model(config=config, df=df, target="target")

print(f"Winner: {results['algorithm']}, Score: {results['score']:.4f}")
