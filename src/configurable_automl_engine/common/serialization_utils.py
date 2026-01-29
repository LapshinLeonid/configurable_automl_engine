import pickle
from pathlib import Path
from typing import Any, Union
from configurable_automl_engine.common.definitions import SerializationFormat

def save_artifact(obj: Any, path: Union[str, Path], fmt: SerializationFormat) -> None:
    """
    Сохраняет объект на диск в выбранном формате.
    """
    path = Path(path)
    
    if fmt == SerializationFormat.joblib:
        import joblib
        joblib.dump(obj, path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

def load_artifact(path: Union[str, Path], fmt: SerializationFormat) -> Any:
    """
    Загружает объект с диска в выбранном формате.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")

    if fmt == SerializationFormat.joblib:
        import joblib
        return joblib.load(path)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)