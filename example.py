from sklearn.datasets import load_diabetes

import configurable_automl_engine as caml

data = load_diabetes(as_frame=True)
df = data.frame

config = {
    "general": {
        "comparison_metric": "r2",
        "phases": [
            {"n_trials": 100, "action": "all_algorithms"},
            {"n_trials": 200, "action": "refine_winner"}
        ],
        "path_to_model": "diabetes_model.joblib"
    },
    "algorithms": {
        "random_forest": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"n_estimators": [10, 500], "max_depth": [3, 20]}
        },
        "ridge": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"alpha": [0.1, 1.0]}
        },
        "xgboosting": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {
                "n_estimators": [100, 1000],
                "max_depth": [3, 10],
                "learning_rate": [0.01, 0.3],
                "subsample": [0.5, 1.0]
            }
        },
    }
}

results = caml.train_best_model(config=config, df=df, target='target')

print(f"Winner: {results['algorithm']}, Score: {results['score']:.4f}")