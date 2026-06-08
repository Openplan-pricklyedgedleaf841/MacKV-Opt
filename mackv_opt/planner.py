from __future__ import annotations

from dataclasses import dataclass

from .capabilities import RuntimeCapabilities
from .models import HardwareProfile, KVStrategy, ModelProfile, OptimizationPlan


DEFAULT_CONTEXT = 8192
MIN_CONTEXT = 4096
RUNTIME_OVERHEAD_BYTES = 768 * 1024**2

KV_STRATEGIES = [
    KVStrategy("f16", "f16", 2.0, 2.0, "highest KV precision"),
    KVStrategy("q8_0", "q8_0", 1.0, 1.0, "balanced default for long context"),
    KVStrategy("q5_1", "q5_1", 0.70, 0.70, "moderate KV compression"),
    KVStrategy("q4_1", "q4_1", 0.56, 0.56, "aggressive KV compression"),
    KVStrategy("q4_0", "q4_0", 0.50, 0.50, "most compact broadly available KV type"),
]


@dataclass(frozen=True)
class PlannerConfig:
    target_context: int = DEFAULT_CONTEXT
    memory_budget_bytes: int | None = None
    safety_fraction: float = 0.72
    max_concurrency: int = 1
    prefer_quality: bool = True


def create_plan(
    model: ModelProfile,
    hardware: HardwareProfile,
    config: PlannerConfig | None = None,
    capabilities: RuntimeCapabilities | dict[str, object] | None = None,
) -> OptimizationPlan:
    config = config or PlannerConfig()
    target_context = _normalize_context(config.target_context)
    memory_budget = _memory_budget(hardware, config)
    estimated_model = _model_bytes(model)
    overhead = RUNTIME_OVERHEAD_BYTES
    warnings: list[str] = []
    reasons: list[str] = []

    if model.max_context and target_context > model.max_context:
        warnings.append(
            f"Target context {target_context} exceeds model metadata context {model.max_context}."
        )

    strategy, context, kv_bytes = _find_strategy(model, target_context, memory_budget, estimated_model, overhead)
    status = "fits" if context == target_context else "reduced"
    if context <= MIN_CONTEXT and context < target_context:
        status = "minimal"

    if context < target_context:
        reasons.append(
            f"Reduced context from {target_context} to {context} to stay within memory budget."
        )
    reasons.append(
        f"Selected K/V cache types {strategy.cache_type_k}/{strategy.cache_type_v}: {strategy.quality_note}."
    )

    estimated_total = estimated_model + kv_bytes + overhead
    concurrency = max(1, min(config.max_concurrency, _safe_concurrency(memory_budget, estimated_total)))
    ollama_options: dict[str, int | float | str | bool] = {
        "num_ctx": context,
        "num_gpu": 999,
    }
    llama_cpp_args = [
        "--ctx-size",
        str(context),
        "--cache-type-k",
        strategy.cache_type_k,
        "--cache-type-v",
        strategy.cache_type_v,
        "--kv-offload",
    ]

    if hardware.platform.lower() != "darwin" or "arm" not in hardware.machine.lower():
        warnings.append("This profile is not Apple Silicon; recommendations are still usable as estimates.")
    if estimated_total > memory_budget:
        warnings.append("Even the minimum strategy exceeds the requested memory budget.")

    runtime_advice = advise_plan_runtime(
        ollama_options=ollama_options,
        llama_cpp_args=llama_cpp_args,
        capabilities=capabilities,
    )
    warnings.extend(str(warning) for warning in runtime_advice.get("warnings", []))

    return OptimizationPlan(
        model_name=model.name,
        status=status,
        num_ctx=context,
        target_context=target_context,
        cache_type_k=strategy.cache_type_k,
        cache_type_v=strategy.cache_type_v,
        kv_offload=True,
        concurrency=concurrency,
        memory_budget_bytes=memory_budget,
        estimated_model_bytes=estimated_model,
        estimated_kv_bytes=kv_bytes,
        estimated_runtime_overhead_bytes=overhead,
        estimated_total_bytes=estimated_total,
        ollama_options=ollama_options,
        llama_cpp_args=llama_cpp_args,
        reasons=reasons,
        warnings=warnings,
        runtime_advice=runtime_advice,
    )


def advise_plan_runtime(
    *,
    ollama_options: dict[str, int | float | str | bool],
    llama_cpp_args: list[str],
    capabilities: RuntimeCapabilities | dict[str, object] | None,
) -> dict[str, object]:
    if capabilities is None:
        return {
            "checked": False,
            "warnings": ["Runtime capabilities were not checked; run `mackv-opt capabilities --json` for a reproducible manifest."],
        }
    caps = capabilities.to_dict() if hasattr(capabilities, "to_dict") else dict(capabilities)
    ollama = caps.get("ollama") if isinstance(caps.get("ollama"), dict) else {}
    llama_cpp = caps.get("llama_cpp") if isinstance(caps.get("llama_cpp"), dict) else {}
    missing_ollama: list[str] = []
    if "num_ctx" in ollama_options and not caps.get("supports_ollama_num_ctx"):
        missing_ollama.append("num_ctx")
    if "num_gpu" in ollama_options and not caps.get("supports_ollama_num_gpu"):
        missing_ollama.append("num_gpu")

    required_llama_flags = [flag for flag in ["--ctx-size", "--cache-type-k", "--cache-type-v", "--kv-offload"] if flag in llama_cpp_args]
    flag_support = {
        "--ctx-size": bool(caps.get("supports_llama_cpp_ctx_size")),
        "--cache-type-k": bool(caps.get("supports_llama_cpp_cache_type_k")),
        "--cache-type-v": bool(caps.get("supports_llama_cpp_cache_type_v")),
        "--kv-offload": bool(caps.get("supports_llama_cpp_kv_offload")),
    }
    missing_llama = [flag for flag in required_llama_flags if not flag_support.get(flag, False)]
    warnings: list[str] = []
    if not ollama.get("available"):
        warnings.append("Ollama CLI was not detected; Ollama run commands may not be executable on this machine.")
    if missing_ollama:
        warnings.append("Ollama option support could not be confirmed for: " + ", ".join(missing_ollama) + ".")
    if not llama_cpp.get("available"):
        warnings.append("llama.cpp command was not detected; llama.cpp args are a portable plan, not a verified local command.")
    if missing_llama:
        warnings.append("llama.cpp flag support could not be confirmed for: " + ", ".join(missing_llama) + ".")
    return {
        "checked": True,
        "ollama_ready": bool(ollama.get("available")) and not missing_ollama,
        "llama_cpp_ready": bool(llama_cpp.get("available")) and not missing_llama,
        "ollama_command": ollama.get("command") or "ollama",
        "llama_cpp_command": llama_cpp.get("command") or "llama.cpp",
        "missing_ollama_options": missing_ollama,
        "missing_llama_cpp_flags": missing_llama,
        "warnings": warnings,
    }


def estimate_kv_cache_bytes(model: ModelProfile, context: int, strategy: KVStrategy) -> int:
    hidden_size = model.hidden_size or _infer_hidden_size(model)
    layer_count = model.layer_count or _infer_layer_count(model)
    kv_ratio = _kv_ratio(model)
    bytes_per_token_per_layer = hidden_size * kv_ratio * (strategy.bytes_per_k + strategy.bytes_per_v)
    return int(context * layer_count * bytes_per_token_per_layer)


def _find_strategy(
    model: ModelProfile,
    target_context: int,
    memory_budget: int,
    estimated_model: int,
    overhead: int,
) -> tuple[KVStrategy, int, int]:
    contexts = _candidate_contexts(target_context)
    for context in contexts:
        for strategy in KV_STRATEGIES:
            kv_bytes = estimate_kv_cache_bytes(model, context, strategy)
            if estimated_model + overhead + kv_bytes <= memory_budget:
                return strategy, context, kv_bytes

    strategy = KV_STRATEGIES[-1]
    context = min(contexts[-1], MIN_CONTEXT)
    kv_bytes = estimate_kv_cache_bytes(model, context, strategy)
    return strategy, context, kv_bytes


def _candidate_contexts(target_context: int) -> list[int]:
    contexts = []
    current = _normalize_context(target_context)
    while current >= MIN_CONTEXT:
        contexts.append(current)
        current //= 2
    if MIN_CONTEXT not in contexts:
        contexts.append(MIN_CONTEXT)
    return contexts


def _normalize_context(context: int) -> int:
    if context <= MIN_CONTEXT:
        return MIN_CONTEXT
    return int(context)


def _memory_budget(hardware: HardwareProfile, config: PlannerConfig) -> int:
    if config.memory_budget_bytes:
        return config.memory_budget_bytes
    candidates = [value for value in [hardware.available_memory_bytes, hardware.total_memory_bytes] if value]
    if not candidates:
        return 8 * 1024**3
    base = min(candidates)
    if hardware.total_memory_bytes and base == hardware.total_memory_bytes:
        return int(base * config.safety_fraction)
    return int(base)


def _model_bytes(model: ModelProfile) -> int:
    if model.size_bytes:
        return model.size_bytes
    if model.parameter_count:
        return int(model.parameter_count * 0.6)
    return 4 * 1024**3


def _safe_concurrency(memory_budget: int, estimated_total: int) -> int:
    if estimated_total <= 0:
        return 1
    return max(1, memory_budget // estimated_total)


def _infer_hidden_size(model: ModelProfile) -> int:
    params = model.parameter_count or 8_000_000_000
    if params >= 60_000_000_000:
        return 8192
    if params >= 30_000_000_000:
        return 6656
    if params >= 13_000_000_000:
        return 5120
    return 4096


def _infer_layer_count(model: ModelProfile) -> int:
    params = model.parameter_count or 8_000_000_000
    if params >= 60_000_000_000:
        return 80
    if params >= 30_000_000_000:
        return 64
    if params >= 13_000_000_000:
        return 40
    return 32


def _kv_ratio(model: ModelProfile) -> float:
    if model.kv_head_count is not None and model.attention_head_count:
        return max(0.05, min(1.0, model.kv_head_count / model.attention_head_count))
    if model.kv_head_count is None:
        return 1.0
    # Most modern GQA models have fewer KV heads than attention heads. Without
    # total head count metadata, this conservative ratio keeps estimates useful.
    if model.kv_head_count <= 8:
        return 0.25
    if model.kv_head_count <= 16:
        return 0.5
    return 1.0
