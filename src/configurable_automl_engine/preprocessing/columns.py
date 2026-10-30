"""Column selection of the preprocessor by column name.

Holds :class:`ColumnNameSelector` — the picklable callable that the
``ColumnTransformer`` uses to pick the categorical and numerical columns — and
the helpers that resolve the names to positions. The selector is pickled
inside saved models, so its module path is part of the artifact format. The
module must not import the other preprocessing modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class ColumnNameSelector:
    """Serializable column selector by names for ``ColumnTransformer``.

    Replaces positional index lists when assembling the preprocessor:
    ``ColumnTransformer`` treats a list of ``int`` as a positional slice and
    ignores the ``pd.DataFrame`` column names, so prediction on a reordered
    DataFrame silently applies the encoder to numerical columns and the scaler
    to categorical ones.

    The selector is a picklable object (an ordinary class with plain attributes,
    not a closure), so it survives ``trainer.save()``/pickle correctly.
    On each call (sklearn resolves callable selectors at the
    ``fit_transform``/``transform`` stage) the names are mapped to positions
    against the actual input ``X``:

    * ``pd.DataFrame`` — mapping by column name; column order does not
      matter, missing or duplicated columns raise an explicit error;
    * any other inputs (``np.ndarray``, ``pd.Series``, etc.) — positional
      fallback by training order (``feature_names``).

    Args:
        feature_names: Full list of feature names in training order.
        column_names: Column names that this selector must select.

    Raises:
        ValueError: If for a ``pd.DataFrame`` some of the ``column_names`` are
            missing or occur more than once, or a name cannot be resolved
            against the training order for non-DataFrame inputs.
    """

    def __init__(self, feature_names: list[str], column_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.column_names = list(column_names)

    def __call__(self, X: Any) -> list[int]:
        if isinstance(X, pd.DataFrame):
            return _resolve_column_positions_by_name(X, self.column_names)
        return _resolve_column_positions_by_order(self.feature_names, self.column_names)


def _resolve_column_positions_by_name(
    X: pd.DataFrame, column_names: list[str]
) -> list[int]:
    """Map column names to positions in the actual DataFrame.

    Args:
        X: Input DataFrame against which the mapping is performed.
        column_names: Column names to select.

    Returns:
        List of positions of the selected columns in ``column_names`` order.

    Raises:
        ValueError: If some of the names are missing from ``X`` or occur more
            than once in ``X`` (ambiguous selection by name).
    """
    missing = [name for name in column_names if name not in X.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in the input DataFrame. "
            f"Available columns: {list(X.columns)}."
        )
    duplicated = [name for name in column_names if list(X.columns).count(name) > 1]
    if duplicated:
        raise ValueError(
            f"Column(s) {duplicated} appear more than once in the input "
            "DataFrame; ambiguous selection by name is not supported."
        )
    # After the uniqueness check, index() gives the exact column position
    # (pandas-stubs: get_loc may return a slice/mask for duplicates).
    return [list(X.columns).index(name) for name in column_names]


def _resolve_column_positions_by_order(
    feature_names: list[str], column_names: list[str]
) -> list[int]:
    """Map column names to positions by training order.

    Used as a fallback for inputs without column names (``np.ndarray``):
    the position of a name is determined by its place in ``feature_names``.

    Args:
        feature_names: Full list of feature names in training order.
        column_names: Column names to select.

    Returns:
        List of positions of the selected columns in ``column_names`` order.

    Raises:
        ValueError: If a name does not occur in ``feature_names`` exactly once.
    """
    positions: list[int] = []
    for name in column_names:
        occurrences = feature_names.count(name)
        if occurrences != 1:
            raise ValueError(
                f"Cannot resolve column {name!r} by training order: it occurs "
                f"{occurrences} time(s) in feature_names {feature_names}."
            )
        positions.append(feature_names.index(name))
    return positions
