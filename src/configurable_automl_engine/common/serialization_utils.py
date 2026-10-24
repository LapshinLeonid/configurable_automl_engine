import logging
import pickle
from pathlib import Path
from typing import Any

import joblib

from configurable_automl_engine.common.definitions import SerializationFormat

logger = logging.getLogger(__name__)

#: Expected file extensions for each serialization format.
#: When a new value is added to SerializationFormat, this reference
#: must be extended with the corresponding extensions.
_FORMAT_EXTENSIONS: dict[SerializationFormat, tuple[str, ...]] = {
    SerializationFormat.pickle: (".pkl", ".pickle"),
    SerializationFormat.joblib: (".joblib",),
}


def resolve_load_formats(
    path: str | Path, fmt: SerializationFormat | None
) -> list[SerializationFormat]:
    """Resolve the candidate serialization formats for loading an artifact.

    When ``fmt`` is given, it is the single candidate. Otherwise the format
    is auto-detected from the file extension (``.joblib`` → joblib,
    ``.pkl``/``.pickle`` → pickle), and the other format is kept as a fallback
    for the case when the primary detection fails (e.g. a wrongly named file).
    For an unknown extension the candidates are joblib first, then pickle —
    both are tried in order until one succeeds.

    Args:
        path: Path to the artifact.
        fmt: Explicit serialization format, or ``None`` for auto-detection.

    Returns:
        The ordered list of formats to try.
    """
    if fmt is not None:
        return [fmt]

    suffix = Path(path).suffix.lower()
    for primary in (SerializationFormat.pickle, SerializationFormat.joblib):
        if suffix in _FORMAT_EXTENSIONS[primary]:
            fallback = (
                SerializationFormat.joblib
                if primary is SerializationFormat.pickle
                else SerializationFormat.pickle
            )
            return [primary, fallback]
    # Unknown extension: try joblib first, then pickle.
    return [SerializationFormat.joblib, SerializationFormat.pickle]


def _warn_on_extension_mismatch(path: Path, fmt: SerializationFormat) -> None:
    """Log a warning if the file extension does not match the format.

    The file extension is merely a convention and does not affect the actual
    serialization method: for example, the artifact ``model.joblib`` with
    ``fmt=pickle`` will be written via ``pickle.dump``. This is misleading
    and may cause the user to fail when loading, so a warning is emitted
    when the extension does not match the format. If the format is missing
    from the extension reference, no warning is emitted — format correctness
    is validated in ``save_artifact`` and ``load_artifact``.

    Args:
        path: Path to the artifact.
        fmt: Requested serialization format.
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
    Save an object to disk in the chosen format.
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
    Load an object from disk in the chosen format.
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
