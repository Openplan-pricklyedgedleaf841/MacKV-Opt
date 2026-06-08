from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    machine: str
    chip: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    pressure: str = "unknown"
    os_version: str = "unknown"
    kernel_version: str = "unknown"
    power_source: str = "unknown"
    power_mode: str = "unknown"
    thermal_state: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str = "unknown"
    parameter_count: int | None = None
    size_bytes: int | None = None
    hidden_size: int | None = None
    layer_count: int | None = None
    attention_head_count: int | None = None
    kv_head_count: int | None = None
    architecture: str = "unknown"
    max_context: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KVStrategy:
    cache_type_k: str
    cache_type_v: str
    bytes_per_k: float
    bytes_per_v: float
    quality_note: str


@dataclass(frozen=True)
class OptimizationPlan:
    model_name: str
    status: str
    num_ctx: int
    target_context: int
    cache_type_k: str
    cache_type_v: str
    kv_offload: bool
    concurrency: int
    memory_budget_bytes: int
    estimated_model_bytes: int
    estimated_kv_bytes: int
    estimated_runtime_overhead_bytes: int
    estimated_total_bytes: int
    ollama_options: dict[str, int | float | str | bool]
    llama_cpp_args: list[str]
    reasons: list[str]
    warnings: list[str]
    runtime_advice: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
