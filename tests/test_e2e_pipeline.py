import pandas as pd
import yaml # не забудьте import yaml
from pathlib import Path
from configurable_automl_engine.training_engine.component import train_best_model

def test_full_pipeline(tmp_path):
    # 1. Создаем временный конфиг БЕЗ ошибок (без extra_trees)
    # Используем tmp_path, чтобы модель сохранялась в изолированную папку
    model_save_path = tmp_path / "best_model.pkl"
    
    config_dict = {
        "general": {
            "comparison_metric": "rmse",
            "n_rude_tries": 2,
            "n_accurate_tries": 3,
            "path_to_model": str(model_save_path)
        },
        "algorithms": {
            "random_forest": { 
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"n_estimators": [5, 10]}
            },
            "decision_tree": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"max_depth": [2, 5]}
            }
        }
    }
    
    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)
    # 2. Данные
    df = pd.DataFrame({
        "feat1": range(10),
        "feat2": [x * 0.1 for x in range(10)],
        "target": [1.0 + x * 0.05 for x in range(10)],
    })
    # 3. Запуск
    # Передаем путь к временному конфигу
    res = train_best_model(str(cfg_file), df, target="target")
    # 4. Проверки
    assert res["algorithm"] in ["rf", "dt", "random_forest", "decision_tree"]
    assert "score" in res
    assert Path(res["model_path"]).exists()
    assert res["model_path"] == str(model_save_path)