from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .bench import sample_memory_state, sample_ollama_process_memory
from .capabilities import RuntimeCapabilities, detect_runtime_capabilities
from .models import HardwareProfile
from .profiler import get_hardware_profile, ollama_models


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    evidence: dict[str, Any]
    next_step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def doctor_payload(
    *,
    hardware: HardwareProfile | None = None,
    capabilities: RuntimeCapabilities | dict[str, Any] | None = None,
    memory_state: dict[str, Any] | None = None,
    process_memory_bytes: int | None = None,
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hardware = hardware or get_hardware_profile()
    caps = _capabilities_dict(capabilities)
    memory = memory_state if memory_state is not None else sample_memory_state()
    process_memory = process_memory_bytes if process_memory_bytes is not None else sample_ollama_process_memory()
    local_models = models if models is not None else ollama_models()

    checks = [
        _hardware_check(hardware),
        _runtime_environment_check(hardware),
        _memory_pressure_check(hardware, memory),
        _ollama_check(caps, local_models),
        _llama_cpp_check(caps),
        _memory_sampler_check(memory, process_memory),
        _run_readiness_check(hardware, caps, memory, local_models),
    ]
    status = _overall_status(checks)
    return {
        "task": "doctor",
        "status": status,
        "hardware": hardware.to_dict(),
        "memory_state": memory,
        "ollama_process_memory_bytes": process_memory,
        "capabilities": caps,
        "ollama_model_count": len(local_models),
        "checks": [check.to_dict() for check in checks],
        "next_steps": _next_steps(checks),
    }


def render_doctor_text(payload: dict[str, Any]) -> str:
    lines = [
        f"MacKV-Opt doctor: {payload.get('status', 'unknown')}",
        "",
        "Checks:",
    ]
    for check in payload.get("checks", []):
        lines.append(f"- [{check.get('status')}] {check.get('name')}: {check.get('message')}")
        next_step = check.get("next_step")
        if next_step:
            lines.append(f"  next: {next_step}")
    next_steps = payload.get("next_steps") or []
    if next_steps:
        lines.extend(["", "Next steps:"])
        lines.extend(f"- {step}" for step in next_steps)
    return "\n".join(lines)


def _hardware_check(hardware: HardwareProfile) -> DoctorCheck:
    is_darwin = hardware.platform == "Darwin"
    is_arm = "arm" in hardware.machine.lower() or "aarch" in hardware.machine.lower()
    if is_darwin and is_arm:
        status = "pass"
        message = "Apple Silicon hardware profile detected."
        next_step = ""
    elif is_darwin:
        status = "warn"
        message = "macOS was detected, but the machine does not look like Apple Silicon."
        next_step = "Apple Silicon validation needs arm64 Mac hardware."
    else:
        status = "warn"
        message = "This machine is not macOS; use it for development or smoke tests only."
        next_step = "Run memory-pressure validation on Apple Silicon."
    return DoctorCheck(
        name="hardware",
        status=status,
        message=message,
        evidence=hardware.to_dict(),
        next_step=next_step,
    )


def _memory_pressure_check(hardware: HardwareProfile, memory: dict[str, Any]) -> DoctorCheck:
    pressure = str(memory.get("memory_pressure") or hardware.pressure or "unknown")
    if pressure == "critical":
        status = "fail"
        message = "Memory pressure is critical."
        next_step = "Close heavy apps or reduce context before running benchmarks."
    elif pressure == "warning":
        status = "warn"
        message = "Memory pressure is elevated."
        next_step = "Record this state and consider a clean benchmark run."
    elif pressure == "normal":
        status = "pass"
        message = "Memory pressure is normal."
        next_step = ""
    else:
        status = "warn"
        message = "Memory pressure could not be verified on this platform."
        next_step = "On macOS, confirm `memory_pressure` works before executable validation."
    return DoctorCheck(
        name="memory-pressure",
        status=status,
        message=message,
        evidence={"hardware_pressure": hardware.pressure, **memory},
        next_step=next_step,
    )


def _runtime_environment_check(hardware: HardwareProfile) -> DoctorCheck:
    missing = [
        name
        for name in ["os_version", "power_source", "thermal_state"]
        if getattr(hardware, name, "unknown") in {"", "unknown"}
    ]
    if hardware.platform != "Darwin":
        return DoctorCheck(
            name="runtime-environment",
            status="warn",
            message="macOS runtime environment details were not collected on this platform.",
            evidence=hardware.to_dict(),
            next_step="Run collection on the target Mac to record macOS, power, and thermal state.",
        )
    if missing:
        return DoctorCheck(
            name="runtime-environment",
            status="warn",
            message="Some macOS runtime environment fields are missing.",
            evidence={**hardware.to_dict(), "missing_environment_fields": missing},
            next_step="Record macOS version, power source, and thermal state before executable validation.",
        )
    return DoctorCheck(
        name="runtime-environment",
        status="pass",
        message="macOS version, power source, and thermal state were recorded.",
        evidence=hardware.to_dict(),
    )


def _ollama_check(caps: dict[str, Any], models: list[dict[str, Any]]) -> DoctorCheck:
    ollama = caps.get("ollama") if isinstance(caps.get("ollama"), dict) else {}
    available = bool(ollama.get("available"))
    if available and models:
        status = "pass"
        message = f"Ollama is available with {len(models)} local model(s)."
        next_step = ""
    elif available:
        status = "warn"
        message = "Ollama is available, but no local models were listed."
        next_step = "Pull or create at least one model before executable validation."
    else:
        status = "fail"
        message = "Ollama was not found on PATH."
        next_step = "Install Ollama or use manual metadata for planning-only workflows."
    return DoctorCheck(
        name="ollama",
        status=status,
        message=message,
        evidence={"ollama": ollama, "models": models[:10]},
        next_step=next_step,
    )


def _llama_cpp_check(caps: dict[str, Any]) -> DoctorCheck:
    llama_cpp = caps.get("llama_cpp") if isinstance(caps.get("llama_cpp"), dict) else {}
    available = bool(llama_cpp.get("available"))
    supports_cache = bool(caps.get("supports_llama_cpp_cache_type_k")) and bool(
        caps.get("supports_llama_cpp_cache_type_v")
    )
    if available and supports_cache:
        status = "pass"
        message = "llama.cpp is available and exposes K/V cache type flags."
        next_step = ""
    elif available:
        status = "warn"
        message = "llama.cpp is available, but cache-type flag support was not confirmed."
        next_step = "Record the llama.cpp build and verify `--cache-type-k/v` manually."
    else:
        status = "warn"
        message = "llama.cpp was not found on PATH."
        next_step = "Install llama.cpp for direct KV cache type validation."
    return DoctorCheck(
        name="llama.cpp",
        status=status,
        message=message,
        evidence={"llama_cpp": llama_cpp, "supports_cache_type_kv": supports_cache},
        next_step=next_step,
    )


def _memory_sampler_check(memory: dict[str, Any], process_memory_bytes: int | None) -> DoctorCheck:
    pressure_known = str(memory.get("memory_pressure") or "unknown") != "unknown"
    swap_known = memory.get("swap_bytes") is not None
    process_known = process_memory_bytes is not None
    if pressure_known and swap_known:
        status = "pass"
        message = "Memory pressure and swap sampling are available."
        next_step = ""
    elif pressure_known or swap_known or process_known:
        status = "warn"
        message = "Only partial memory sampling is available."
        next_step = "Pair benchmark logs with Activity Monitor or powermetrics for memory attribution."
    else:
        status = "warn"
        message = "Memory sampling is not fully available in this environment."
        next_step = "Run on macOS to collect pressure and swap evidence."
    return DoctorCheck(
        name="memory-sampler",
        status=status,
        message=message,
        evidence={"memory_state": memory, "ollama_process_memory_bytes": process_memory_bytes},
        next_step=next_step,
    )


def _run_readiness_check(
    hardware: HardwareProfile,
    caps: dict[str, Any],
    memory: dict[str, Any],
    models: list[dict[str, Any]],
) -> DoctorCheck:
    ready = (
        hardware.platform == "Darwin"
        and ("arm" in hardware.machine.lower() or "aarch" in hardware.machine.lower())
        and bool(caps.get("ollama", {}).get("available") if isinstance(caps.get("ollama"), dict) else False)
        and bool(models)
        and str(memory.get("memory_pressure") or "unknown") in {"normal", "warning"}
    )
    if ready:
        status = "pass"
        message = "This machine looks ready for MacKV-Opt executable validation."
        next_step = "Run `mackv-opt experiment ... --execute --repeats 3`."
    else:
        status = "warn"
        message = "This environment is not yet sufficient for executable Mac validation."
        next_step = "Resolve failed or warning checks, then record profile and capabilities JSON."
    return DoctorCheck(
        name="run-readiness",
        status=status,
        message=message,
        evidence={
            "platform": hardware.platform,
            "machine": hardware.machine,
            "ollama_available": bool(caps.get("ollama", {}).get("available"))
            if isinstance(caps.get("ollama"), dict)
            else False,
            "model_count": len(models),
            "memory_pressure": memory.get("memory_pressure"),
        },
        next_step=next_step,
    )


def _capabilities_dict(capabilities: RuntimeCapabilities | dict[str, Any] | None) -> dict[str, Any]:
    if capabilities is None:
        return detect_runtime_capabilities().to_dict()
    if hasattr(capabilities, "to_dict"):
        return capabilities.to_dict()  # type: ignore[union-attr]
    return dict(capabilities)


def _overall_status(checks: list[DoctorCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _next_steps(checks: list[DoctorCheck]) -> list[str]:
    steps: list[str] = []
    for check in checks:
        if check.next_step and check.next_step not in steps:
            steps.append(check.next_step)
    return steps
