from sklearn.datasets import load_diabetes

import configurable_automl_engine as caml

df = load_diabetes(as_frame=True).frame

config = {
    "general": {
        "comparison_metric": "r2",
        "path_to_model": "diabetes_model.pkl",
    },
    # {} = use the built-in search space of the algorithm
    "algorithms": {
        "ridge": {},
        "random_forest": {},
    },
}

results = caml.train_best_model(config=config, df=df, target="target")

print(f"Winner: {results['algorithm']}")
print(f"Hold-out R2 (honest estimate): {results['holdout_score']:.4f}")
