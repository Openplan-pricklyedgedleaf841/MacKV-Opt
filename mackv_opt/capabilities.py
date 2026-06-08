from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Iterable


LLAMA_CPP_CANDIDATES = ("llama-cli", "llama-server", "server", "main")
KV_CACHE_FLAGS = ("--cache-type-k", "--cache-type-v")


@dataclass(frozen=True)
class CommandProbe:
    command: str
    path: str | None
    available: bool
    version: str | None = None
    help_available: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCapabilities:
    ollama: CommandProbe
    llama_cpp: CommandProbe
    supports_ollama_num_ctx: bool
    supports_ollama_num_gpu: bool
    supports_llama_cpp_ctx_size: bool
    supports_llama_cpp_kv_offload: bool
    supports_llama_cpp_cache_type_k: bool
    supports_llama_cpp_cache_type_v: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ollama"] = self.ollama.to_dict()
        payload["llama_cpp"] = self.llama_cpp.to_dict()
        payload["warnings"] = list(self.warnings)
        return payload


def detect_runtime_capabilities() -> RuntimeCapabilities:
    ollama = _probe_ollama()
    llama_cpp = _probe_llama_cpp()
    warnings: list[str] = []
    if not ollama.available:
        warnings.append("Ollama CLI was not found on PATH; profile and run wrappers will need manual metadata.")
    if not llama_cpp.available:
        warnings.append("llama.cpp CLI/server was not found on PATH; llama.cpp cache-type support could not be verified.")
    if ollama.available:
        warnings.append("Ollama run options are model/runtime dependent; KV cache type support is not assumed unless exposed by the runtime.")
    return RuntimeCapabilities(
        ollama=ollama,
        llama_cpp=llama_cpp,
        supports_ollama_num_ctx=_help_has_any(ollama, ("num_ctx", "--option")),
        supports_ollama_num_gpu=_help_has_any(ollama, ("num_gpu", "--option")),
        supports_llama_cpp_ctx_size=_help_has_any(llama_cpp, ("--ctx-size", "-c")),
        supports_llama_cpp_kv_offload=_help_has_any(llama_cpp, ("--kv-offload", "--no-kv-offload")),
        supports_llama_cpp_cache_type_k=_help_has_any(llama_cpp, ("--cache-type-k",)),
        supports_llama_cpp_cache_type_v=_help_has_any(llama_cpp, ("--cache-type-v",)),
        warnings=tuple(warnings),
    )


def _probe_ollama() -> CommandProbe:
    path = shutil.which("ollama")
    if not path:
        return CommandProbe(command="ollama", path=None, available=False)
    version = _command_text(["ollama", "--version"])
    help_text = _command_text(["ollama", "run", "--help"]) or _command_text(["ollama", "--help"])
    return CommandProbe(
        command="ollama",
        path=path,
        available=True,
        version=_extract_version(version),
        help_available=bool(help_text),
        error=None if help_text or version else "Ollama command exists but version/help probing returned no output.",
    )


def _probe_llama_cpp(candidates: Iterable[str] = LLAMA_CPP_CANDIDATES) -> CommandProbe:
    for command in candidates:
        path = shutil.which(command)
        if not path:
            continue
        version = _command_text([command, "--version"])
        help_text = _command_text([command, "--help"])
        if not _looks_like_llama_cpp(command, path, help_text, version):
            continue
        return CommandProbe(
            command=command,
            path=path,
            available=True,
            version=_extract_version(version),
            help_available=bool(help_text),
            error=None if help_text or version else f"{command} exists but version/help probing returned no output.",
        )
    return CommandProbe(command="llama.cpp", path=None, available=False)


def _looks_like_llama_cpp(command: str, path: str, help_text: str, version_text: str) -> bool:
    command_lower = command.lower()
    path_lower = path.lower()
    text = (help_text + "\n" + version_text).lower()
    if command_lower.startswith("llama-") or "llama" in path_lower:
        return True
    return any(marker in text for marker in ["llama.cpp", "--ctx-size", "--cache-type-k", "--kv-offload"])


def _help_has_any(probe: CommandProbe, needles: Iterable[str]) -> bool:
    if not probe.available:
        return False
    help_text = _command_text([probe.command, "run", "--help"]) if probe.command == "ollama" else _command_text([probe.command, "--help"])
    if not help_text:
        return False
    lowered = help_text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _command_text(command: list[str], *, timeout_seconds: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _extract_version(text: str) -> str | None:
    clean = " ".join(text.split())
    if not clean:
        return None
    match = re.search(r"v?\d+(?:\.\d+)+(?:[-+][\w.]+)?", clean)
    return match.group(0) if match else clean[:120]
