"""Helpers for converting runtime payloads into JSON-safe structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def to_json_safe(value: Any) -> Any:
    """Recursively convert runtime values into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in value]
    if hasattr(value, "get_fields_and_field_types"):
        return {
            field: to_json_safe(getattr(value, field))
            for field in value.get_fields_and_field_types().keys()
        }

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return to_json_safe(value.tolist())
        if isinstance(value, np.generic):
            return to_json_safe(value.item())
    except Exception:
        pass

    return str(value)
