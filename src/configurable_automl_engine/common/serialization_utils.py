import logging
import pickle
from pathlib import Path
from typing import Any

import joblib

from configurable_automl_engine.common.definitions import SerializationFormat

logger = logging.getLogger(__name__)

#: Ожидаемые расширения файлов для каждого формата сериализации.
#: При добавлении нового значения в SerializationFormat справочник
#: необходимо дополнить соответствующими расширениями.
_FORMAT_EXTENSIONS: dict[SerializationFormat, tuple[str, ...]] = {
    SerializationFormat.pickle: (".pkl", ".pickle"),
    SerializationFormat.joblib: (".joblib",),
}


def _warn_on_extension_mismatch(path: Path, fmt: SerializationFormat) -> None:
    """Логирует предупреждение, если расширение файла не соответствует формату.

    Расширение файла является лишь соглашением и не влияет на фактический
    способ сериализации: например, артефакт ``model.joblib`` при
    ``fmt=pickle`` будет записан через ``pickle.dump``. Это вводит в
    заблуждение и может привести к ошибке загрузки у пользователя, поэтому
    при несоответствии расширения формату выдаётся предупреждение. Если
    формат отсутствует в справочнике расширений, предупреждение не
    выдаётся — корректность формата проверяется в ``save_artifact``
    и ``load_artifact``.

    Args:
        path: Путь к артефакту.
        fmt: Запрошенный формат сериализации.
    """
    expected_extensions = _FORMAT_EXTENSIONS.get(fmt)
    if expected_extensions and path.suffix.lower() not in expected_extensions:
        logger.warning(
            "File extension %r does not match serialization format %r; "
            "expected one of %s. The extension may mislead tools or users "
            "trying to load this artifact.",
            path.suffix,
            fmt.value,
            ", ".join(expected_extensions),
        )


def save_artifact(obj: Any, path: str | Path, fmt: SerializationFormat) -> None:
    """
    Сохраняет объект на диск в выбранном формате.
    """
    path = Path(path)
    _warn_on_extension_mismatch(path, fmt)

    if fmt == SerializationFormat.joblib:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def load_artifact(path: str | Path, fmt: SerializationFormat) -> Any:
    """
    Загружает объект с диска в выбранном формате.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    _warn_on_extension_mismatch(path, fmt)

    if fmt == SerializationFormat.joblib:
        return joblib.load(path)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
