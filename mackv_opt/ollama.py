from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from .models import ModelProfile


def load_model_profile(model_name: str) -> ModelProfile | None:
    payload = load_ollama_show_payload(model_name)
    if payload is None:
        return None
    return normalize_show_payload(model_name, payload)


def load_ollama_show_payload(model_name: str) -> dict[str, Any] | None:
    if not shutil.which("ollama"):
        return None
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--json"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload


def normalize_show_payload(model_name: str, payload: dict[str, Any]) -> ModelProfile:
    details = payload.get("details") or {}
    info = payload.get("model_info") or payload.get("info") or {}
    architecture = str(info.get("general.architecture") or details.get("family") or "unknown")
    prefix = architecture

    parameter_count = _parse_parameters(
        info.get("general.parameter_count")
        or info.get(f"{prefix}.parameter_count")
        or details.get("parameter_size")
    )
    size = _int_or_none(payload.get("size") or details.get("size"))
    return ModelProfile(
        name=model_name,
        family=str(details.get("family") or architecture or "unknown"),
        parameter_count=parameter_count,
        size_bytes=size,
        hidden_size=_int_or_none(
            info.get(f"{prefix}.embedding_length")
            or info.get("llama.embedding_length")
            or info.get("general.embedding_length")
        ),
        layer_count=_int_or_none(
            info.get(f"{prefix}.block_count")
            or info.get("llama.block_count")
            or info.get("general.block_count")
        ),
        attention_head_count=_int_or_none(
            info.get(f"{prefix}.attention.head_count")
            or info.get("llama.attention.head_count")
            or info.get("general.attention.head_count")
        ),
        kv_head_count=_int_or_none(
            info.get(f"{prefix}.attention.head_count_kv")
            or info.get("llama.attention.head_count_kv")
        ),
        architecture=architecture,
        max_context=_int_or_none(
            info.get(f"{prefix}.context_length")
            or info.get("llama.context_length")
            or info.get("general.context_length")
        ),
    )


def build_run_command(
    model_name: str,
    options: dict[str, int | float | str | bool],
    *,
    prompt: str | None = None,
) -> list[str]:
    command = ["ollama", "run", model_name]
    for key, value in options.items():
        command.extend(["--option", f"{key}={_format_option_value(value)}"])
    if prompt:
        command.append(prompt)
    return command


def run_model(
    model_name: str,
    options: dict[str, int | float | str | bool],
    *,
    prompt: str | None = None,
) -> int:
    command = build_run_command(model_name, options, prompt=prompt)
    completed = subprocess.run(command)
    return completed.returncode


def _format_option_value(value: int | float | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_parameters(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(" ", "")
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([bmk]?)", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    return int(number * multiplier)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
