from __future__ import annotations

import re


_DECIMAL_UNITS = {
    "b": 1,
    "kb": 1_000,
    "k": 1_000,
    "mb": 1_000_000,
    "m": 1_000_000,
    "gb": 1_000_000_000,
    "g": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "t": 1_000_000_000_000,
}

_BINARY_UNITS = {
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_size(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().replace("_", "")
    match = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|[kmgt])?\s*", text)
    if not match:
        raise ValueError(f"Invalid size: {value!r}")

    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multiplier = _BINARY_UNITS.get(unit, _DECIMAL_UNITS.get(unit))
    if multiplier is None:
        raise ValueError(f"Invalid size unit: {unit!r}")
    return int(number * multiplier)


def parse_context(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Context must be positive")
        return value

    text = str(value).strip().replace("_", "")
    match = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", text)
    if not match:
        raise ValueError(f"Invalid context: {value!r}")

    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = 1
    if suffix == "k":
        multiplier = 1024
    elif suffix == "m":
        multiplier = 1024 * 1024

    result = int(number * multiplier)
    if result <= 0:
        raise ValueError("Context must be positive")
    return result


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = [("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]
    for suffix, scale in units:
        if abs(value) >= scale:
            return f"{value / scale:.2f} {suffix}"
    return f"{value} B"
