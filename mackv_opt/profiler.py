from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from typing import Any

from .capabilities import detect_runtime_capabilities
from .models import HardwareProfile


def get_hardware_profile(
    *,
    total_memory_bytes: int | None = None,
    available_memory_bytes: int | None = None,
) -> HardwareProfile:
    system = platform.system()
    machine = platform.machine()
    chip = _detect_chip(system)
    total = total_memory_bytes if total_memory_bytes is not None else _detect_total_memory(system)
    available = (
        available_memory_bytes
        if available_memory_bytes is not None
        else int(total_memory_bytes * 0.72)
        if total_memory_bytes is not None
        else _detect_available_memory(system, total)
    )
    pressure = _detect_pressure(system)
    return HardwareProfile(
        platform=system,
        machine=machine,
        chip=chip,
        total_memory_bytes=total,
        available_memory_bytes=available,
        pressure=pressure,
        os_version=_detect_os_version(system),
        kernel_version=platform.release() or "unknown",
        power_source=_detect_power_source(system),
        power_mode=_detect_power_mode(system),
        thermal_state=_detect_thermal_state(system),
    )


def ollama_models() -> list[dict[str, Any]]:
    if not shutil.which("ollama"):
        return []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    models: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append({"name": parts[0], "raw": line})
    return models


def profile_payload(hardware: HardwareProfile | None = None) -> dict[str, Any]:
    hardware = hardware or get_hardware_profile()
    return {
        "hardware": hardware.to_dict(),
        "ollama": {
            "available": shutil.which("ollama") is not None,
            "models": ollama_models(),
        },
        "capabilities": detect_runtime_capabilities().to_dict(),
    }


def _detect_chip(system: str) -> str:
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return result.stdout.strip() or "Apple Silicon"
        except (OSError, subprocess.CalledProcessError):
            return "Apple Silicon"
    return platform.processor() or "unknown"


def _detect_total_memory(system: str) -> int | None:
    if system == "Darwin":
        return _sysctl_int("hw.memsize")
    if system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory = MEMORYSTATUSEX()
            memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
            return int(memory.ullTotalPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _detect_available_memory(system: str, total: int | None) -> int | None:
    if system == "Windows":
        try:
            output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                text=True,
                encoding="utf-8",
            )
            return int(output.strip()) * 1024
        except (OSError, subprocess.CalledProcessError, ValueError):
            return int(total * 0.72) if total else None
    if total:
        return int(total * 0.72)
    return None


def _detect_pressure(system: str) -> str:
    if system != "Darwin":
        return "unknown"
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=2,
        )
        text = result.stdout.lower()
        if "critical" in text:
            return "critical"
        if "warn" in text:
            return "warning"
        if text:
            return "normal"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _detect_os_version(system: str) -> str:
    if system == "Darwin":
        version = _command_stdout(["sw_vers", "-productVersion"])
        build = _command_stdout(["sw_vers", "-buildVersion"])
        if version and build:
            return f"macOS {version} ({build})"
        if version:
            return f"macOS {version}"
    value = platform.platform()
    return value or "unknown"


def _detect_power_source(system: str) -> str:
    if system != "Darwin":
        return "unknown"
    text = _command_stdout(["pmset", "-g", "batt"]).lower()
    if "ac power" in text:
        return "ac"
    if "battery power" in text:
        return "battery"
    return "unknown"


def _detect_power_mode(system: str) -> str:
    if system != "Darwin":
        return "unknown"
    text = _command_stdout(["pmset", "-g", "custom"])
    if not text:
        return "unknown"
    lowered = text.lower()
    if "lowpowermode" in lowered:
        for line in lowered.splitlines():
            if "lowpowermode" not in line:
                continue
            parts = line.split()
            value = parts[-1] if parts else ""
            if value == "1":
                return "low-power"
            if value == "0":
                return "normal"
    return "recorded"


def _detect_thermal_state(system: str) -> str:
    if system != "Darwin":
        return "unknown"
    text = _command_stdout(["pmset", "-g", "therm"])
    if not text:
        return "unknown"
    lowered = text.lower()
    if "thermal warning level" in lowered:
        for line in lowered.splitlines():
            if "thermal warning level" not in line:
                continue
            value = line.split(":", 1)[-1].strip() if ":" in line else line.split()[-1]
            return value or "recorded"
    if "cpu_speed_limit" in lowered or "scheduler_limit" in lowered:
        return "recorded"
    return "recorded"


def _sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return int(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _command_stdout(command: list[str], *, timeout_seconds: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "").strip()


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
