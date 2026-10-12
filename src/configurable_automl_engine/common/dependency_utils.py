"""
Utilities for checking whether dependencies are available without importing them.
Used to validate configuration and avoid side effects.
"""

from __future__ import annotations

import importlib.util


def is_installed(package_name: str) -> bool:
    """
    Check whether a package is installed in the environment without importing it.

    Args:
        package_name: Name of the Python package (as in import).

    Returns:
        True if the package is available, False otherwise.
    """
    if not package_name:
        return False
    return importlib.util.find_spec(package_name) is not None
