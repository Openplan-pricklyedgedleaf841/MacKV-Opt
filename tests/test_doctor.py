from __future__ import annotations

from mackv_opt.doctor import doctor_payload, render_doctor_text
from mackv_opt.models import HardwareProfile


def _hardware(platform: str = "Darwin", machine: str = "arm64", pressure: str = "normal") -> HardwareProfile:
    return HardwareProfile(
        platform=platform,
        machine=machine,
        chip="Apple M3 Pro" if platform == "Darwin" else "unknown",
        total_memory_bytes=36 * 1024**3,
        available_memory_bytes=24 * 1024**3,
        pressure=pressure,
        os_version="macOS 15.5 (24F74)" if platform == "Darwin" else "unknown",
        kernel_version="25.0.0" if platform == "Darwin" else "unknown",
        power_source="ac" if platform == "Darwin" else "unknown",
        power_mode="normal" if platform == "Darwin" else "unknown",
        thermal_state="0" if platform == "Darwin" else "unknown",
    )


def _capabilities(*, ollama: bool = True, llama_cpp: bool = True, cache_types: bool = True) -> dict[str, object]:
    return {
        "ollama": {"available": ollama, "version": "0.12.1", "command": "ollama"},
        "llama_cpp": {"available": llama_cpp, "version": "b6500", "command": "llama-cli"},
        "supports_ollama_num_ctx": ollama,
        "supports_ollama_num_gpu": ollama,
        "supports_llama_cpp_ctx_size": llama_cpp,
        "supports_llama_cpp_kv_offload": llama_cpp,
        "supports_llama_cpp_cache_type_k": llama_cpp and cache_types,
        "supports_llama_cpp_cache_type_v": llama_cpp and cache_types,
        "warnings": [],
    }


def test_doctor_payload_passes_for_apple_silicon_ready_environment():
    payload = doctor_payload(
        hardware=_hardware(),
        capabilities=_capabilities(),
        memory_state={"memory_pressure": "normal", "swap_bytes": 0},
        process_memory_bytes=512 * 1024**2,
        models=[{"name": "llama3.1:8b"}],
    )

    assert payload["status"] == "pass"
    assert payload["ollama_model_count"] == 1
    assert {check["name"]: check["status"] for check in payload["checks"]}["run-readiness"] == "pass"
    assert {check["name"]: check["status"] for check in payload["checks"]}["runtime-environment"] == "pass"
    assert payload["next_steps"] == ["Run `mackv-opt experiment ... --execute --repeats 3`."]


def test_doctor_payload_warns_when_macos_environment_fields_are_missing():
    hardware = HardwareProfile(
        platform="Darwin",
        machine="arm64",
        chip="Apple M3 Pro",
        total_memory_bytes=36 * 1024**3,
        available_memory_bytes=24 * 1024**3,
        pressure="normal",
    )

    payload = doctor_payload(
        hardware=hardware,
        capabilities=_capabilities(),
        memory_state={"memory_pressure": "normal", "swap_bytes": 0},
        process_memory_bytes=512 * 1024**2,
        models=[{"name": "llama3.1:8b"}],
    )

    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["status"] == "warn"
    assert checks["runtime-environment"]["status"] == "warn"
    assert "power_source" in checks["runtime-environment"]["evidence"]["missing_environment_fields"]


def test_doctor_payload_warns_on_non_mac_development_environment():
    payload = doctor_payload(
        hardware=_hardware(platform="Windows", machine="AMD64", pressure="unknown"),
        capabilities=_capabilities(),
        memory_state={"memory_pressure": "unknown", "swap_bytes": None},
        process_memory_bytes=None,
        models=[{"name": "llama3.1:8b"}],
    )

    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["status"] == "warn"
    assert checks["hardware"]["status"] == "warn"
    assert "not macOS" in checks["hardware"]["message"]
    assert checks["run-readiness"]["status"] == "warn"


def test_doctor_payload_fails_when_ollama_is_missing():
    payload = doctor_payload(
        hardware=_hardware(),
        capabilities=_capabilities(ollama=False),
        memory_state={"memory_pressure": "normal", "swap_bytes": 0},
        process_memory_bytes=None,
        models=[],
    )

    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["status"] == "fail"
    assert checks["ollama"]["status"] == "fail"
    assert "Ollama was not found" in checks["ollama"]["message"]


def test_doctor_payload_warns_when_llama_cpp_cache_flags_are_missing():
    payload = doctor_payload(
        hardware=_hardware(),
        capabilities=_capabilities(llama_cpp=True, cache_types=False),
        memory_state={"memory_pressure": "normal", "swap_bytes": 0},
        process_memory_bytes=1024,
        models=[{"name": "llama3.1:8b"}],
    )

    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["status"] == "warn"
    assert checks["llama.cpp"]["status"] == "warn"
    assert "cache-type flag support was not confirmed" in checks["llama.cpp"]["message"]


def test_doctor_payload_warns_when_memory_sampler_is_partial():
    payload = doctor_payload(
        hardware=_hardware(),
        capabilities=_capabilities(),
        memory_state={"memory_pressure": "normal", "swap_bytes": None},
        process_memory_bytes=1024,
        models=[{"name": "llama3.1:8b"}],
    )

    checks = {check["name"]: check for check in payload["checks"]}
    assert payload["status"] == "warn"
    assert checks["memory-sampler"]["status"] == "warn"
    assert "partial memory sampling" in checks["memory-sampler"]["message"]


def test_render_doctor_text_includes_checks_and_next_steps():
    payload = doctor_payload(
        hardware=_hardware(platform="Linux", machine="x86_64", pressure="unknown"),
        capabilities=_capabilities(ollama=False, llama_cpp=False),
        memory_state={"memory_pressure": "unknown", "swap_bytes": None},
        process_memory_bytes=None,
        models=[],
    )

    rendered = render_doctor_text(payload)

    assert rendered.startswith("MacKV-Opt doctor: fail")
    assert "- [fail] ollama:" in rendered
    assert "Next steps:" in rendered
