from sklearn.datasets import load_diabetes

import configurable_automl_engine as caml

data = load_diabetes(as_frame=True)
df = data.frame

config = {
    "general": {
        "comparison_metric": "mae",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "path_to_model": "diabetes_model.joblib",
        "phases": [
            {"name": "Coarse Search", "n_trials": 100, "action": "all_algorithms"},
            {"name": "Fine Tuning", "n_trials": 200, "action": "refine_winner"}
        ],
        "log_to_file": None,
        "parallel_strategy": "serial",
        "max_workers": 1
    },
    "oversampling": {
        "enable": False,
        "multiplier": 1.0,
        "algorithm": "smote"
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
        "xgboost": {
            "enable": True,
            "limit_hyperparameters": False,
            "hyperparameters": {
                "n_estimators": [100, 1000],
                "max_depth": [3, 10],
                "learning_rate": [0.01, 0.3],
                "subsample": [0.5, 1.0]
            }
        },
    }
}
if __name__ == "__main__":
    results = caml.train_best_model(config=config, df=df, target='target')
    print(f"Winner: {results['algorithm']}, Score: {results['score']:.4f}")