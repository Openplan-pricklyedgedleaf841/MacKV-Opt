from mackv_opt.models import HardwareProfile, ModelProfile
from mackv_opt.planner import PlannerConfig, advise_plan_runtime, create_plan
from mackv_opt.profiler import get_hardware_profile


def hardware(total_gib: int = 16) -> HardwareProfile:
    return HardwareProfile(
        platform="Darwin",
        machine="arm64",
        chip="Apple M2",
        total_memory_bytes=total_gib * 1024**3,
        available_memory_bytes=int(total_gib * 1024**3 * 0.72),
        pressure="normal",
    )


def model(size_gib: float = 4.8) -> ModelProfile:
    return ModelProfile(
        name="llama3.1:8b",
        family="llama",
        parameter_count=8_000_000_000,
        size_bytes=int(size_gib * 1024**3),
        hidden_size=4096,
        layer_count=32,
        attention_head_count=32,
        kv_head_count=8,
        architecture="llama",
    )


def test_planner_selects_long_context_with_compressed_kv_under_budget():
    plan = create_plan(
        model(),
        hardware(),
        PlannerConfig(target_context=65_536, memory_budget_bytes=12 * 1024**3),
    )

    assert plan.num_ctx == 65_536
    assert plan.cache_type_k in {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16"}
    assert plan.cache_type_v in {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16"}
    assert plan.estimated_total_bytes <= plan.memory_budget_bytes
    assert plan.ollama_options["num_ctx"] == 65_536
    assert plan.status == "fits"
    assert plan.runtime_advice["checked"] is False


def test_planner_reduces_context_when_target_cannot_fit():
    plan = create_plan(
        model(size_gib=11.5),
        hardware(total_gib=16),
        PlannerConfig(target_context=131_072, memory_budget_bytes=13 * 1024**3),
    )

    assert plan.num_ctx < 131_072
    assert plan.num_ctx >= 4096
    assert plan.status in {"reduced", "minimal"}
    assert any("Reduced context" in reason for reason in plan.reasons)


def test_planner_uses_memory_budget_safety_margin_when_budget_omitted():
    plan = create_plan(
        model(),
        hardware(total_gib=32),
        PlannerConfig(target_context=32_768),
    )

    assert plan.memory_budget_bytes < 32 * 1024**3
    assert plan.memory_budget_bytes <= int(32 * 1024**3 * 0.72)
    assert plan.concurrency >= 1


def test_hardware_memory_override_uses_matching_available_memory_budget():
    profile = get_hardware_profile(total_memory_bytes=16 * 1024**3)

    assert profile.total_memory_bytes == 16 * 1024**3
    assert profile.available_memory_bytes == int(16 * 1024**3 * 0.72)


def test_planner_runtime_advice_marks_supported_runtime_ready():
    capabilities = {
        "ollama": {"available": True, "command": "ollama"},
        "llama_cpp": {"available": True, "command": "llama-cli"},
        "supports_ollama_num_ctx": True,
        "supports_ollama_num_gpu": True,
        "supports_llama_cpp_ctx_size": True,
        "supports_llama_cpp_cache_type_k": True,
        "supports_llama_cpp_cache_type_v": True,
        "supports_llama_cpp_kv_offload": True,
    }

    plan = create_plan(
        model(),
        hardware(),
        PlannerConfig(target_context=8192, memory_budget_bytes=12 * 1024**3),
        capabilities=capabilities,
    )

    assert plan.runtime_advice["checked"] is True
    assert plan.runtime_advice["ollama_ready"] is True
    assert plan.runtime_advice["llama_cpp_ready"] is True
    assert plan.runtime_advice["warnings"] == []


def test_runtime_advice_reports_missing_runtime_support():
    advice = advise_plan_runtime(
        ollama_options={"num_ctx": 8192, "num_gpu": 999},
        llama_cpp_args=["--ctx-size", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--kv-offload"],
        capabilities={
            "ollama": {"available": False},
            "llama_cpp": {"available": False},
            "supports_ollama_num_ctx": False,
            "supports_ollama_num_gpu": False,
            "supports_llama_cpp_ctx_size": False,
            "supports_llama_cpp_cache_type_k": False,
            "supports_llama_cpp_cache_type_v": False,
            "supports_llama_cpp_kv_offload": False,
        },
    )

    assert advice["checked"] is True
    assert advice["ollama_ready"] is False
    assert advice["llama_cpp_ready"] is False
    assert advice["missing_ollama_options"] == ["num_ctx", "num_gpu"]
    assert "--cache-type-k" in advice["missing_llama_cpp_flags"]
    assert advice["warnings"]
