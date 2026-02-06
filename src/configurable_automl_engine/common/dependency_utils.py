"""
Утилиты для проверки наличия зависимостей без их импорта.
Используется для валидации конфигурации и избежания побочных эффектов.
"""
from __future__ import annotations

import importlib.util


def is_installed(package_name: str) -> bool:
    """
    Проверяет, установлен ли пакет в окружении, не импортируя его.

    Args:
        package_name: Имя Python-пакета (как в import).

    Returns:
        True, если пакет доступен, иначе False.
    """
    if not package_name:
        return False
    return importlib.util.find_spec(package_name) is not None