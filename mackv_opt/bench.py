from __future__ import annotations

import json
import platform
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

from .stats import repeat_count, repeat_runs, summarize_repeated_runs
from .stability import StabilityConfig, annotate_runs_with_stability, summarize_stability
from .units import parse_context

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_PROMPT = "Write one concise paragraph about local LLM inference on Apple Silicon."


@dataclass(frozen=True)
class BenchJob:
    model: str
    context: int
    mode: str = "plan"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class BenchRunResult:
    model: str
    context: int
    method: str
    status: str
    started_at: str
    ended_at: str
    wall_time_seconds: float
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    first_token_latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    memory_pressure: str = "unknown"
    swap_bytes: int | None = None
    pageins_delta: int | None = None
    pageout_delta: int | None = None
    pageout_bytes_delta: int | None = None
    swapins_delta: int | None = None
    swapouts_delta: int | None = None
    memory_samples: int = 0
    memory_sample_interval_seconds: float | None = None
    memory_series: list[dict[str, int | str | None]] | None = None
    ollama_default_options: bool = False
    done_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_bench_matrix(models: Iterable[str], contexts: Iterable[str | int]) -> list[BenchJob]:
    clean_models = [model.strip() for model in models if model and model.strip()]
    clean_contexts = [parse_context(context) for context in contexts]
    return [BenchJob(model=model, context=context) for model in clean_models for context in clean_contexts]


def dry_run_payload(models: Iterable[str], contexts: Iterable[str | int], *, repeats: int = 1) -> dict[str, object]:
    jobs = build_bench_matrix(models, contexts)
    return {
        "dry_run": True,
        "repeats": repeat_count(repeats),
        "planned_run_count": len(jobs) * repeat_count(repeats),
        "jobs": [job.to_dict() for job in jobs],
        "metrics": [
            "max_stable_context",
            "first_token_latency_ms",
            "tokens_per_second",
            "peak_memory_bytes",
            "memory_pressure",
            "swap_bytes",
            "pageout_delta",
            "pageout_bytes_delta",
            "swapouts_delta",
        ],
    }


def execute_bench_payload(
    models: Iterable[str],
    contexts: Iterable[str | int],
    *,
    prompt: str = DEFAULT_PROMPT,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 128,
    memory_sample_interval_seconds: float = 0.5,
    include_memory_series: bool = False,
    repeats: int = 1,
    stability_config: StabilityConfig | None = None,
    use_ollama_default_options: bool = False,
) -> dict[str, object]:
    jobs = build_bench_matrix(models, contexts)
    started_at = _utc_now()
    selected_stability_config = (stability_config or StabilityConfig()).normalized()
    runs = annotate_runs_with_stability(
        repeat_runs(
            jobs,
            repeat_count(repeats),
            lambda job: run_ollama_generate_benchmark(
                job,
                prompt=prompt,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                num_predict=num_predict,
                memory_sample_interval_seconds=memory_sample_interval_seconds,
                include_memory_series=include_memory_series,
                apply_num_ctx=not use_ollama_default_options,
            ),
        ),
        selected_stability_config,
    )
    return {
        "dry_run": False,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "base_url": base_url,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "repeats": repeat_count(repeats),
        "ollama_default_options": bool(use_ollama_default_options),
        "stability_config": selected_stability_config.to_dict(),
        "runs": runs,
        "repeat_summaries": summarize_repeated_runs(runs),
        "stability_summary": summarize_stability(runs, selected_stability_config),
        "metrics": [
            "prompt_eval_count",
            "prompt_eval_duration_ns",
            "eval_count",
            "eval_duration_ns",
            "tokens_per_second",
            "prompt_tokens_per_second",
            "total_duration_ns",
            "load_duration_ns",
            "wall_time_seconds",
            "memory_pressure",
            "swap_bytes",
            "pageout_delta",
            "pageout_bytes_delta",
            "swapouts_delta",
            "memory_samples",
            "stable",
            "stability_reason",
        ],
    }


def _execute_bench_runs(
    jobs: Iterable[BenchJob],
    *,
    prompt: str,
    base_url: str,
    timeout_seconds: float,
    num_predict: int | None,
    memory_sample_interval_seconds: float,
    include_memory_series: bool,
    apply_num_ctx: bool,
) -> list[dict[str, object]]:
    return [
        run_ollama_generate_benchmark(
            job,
            prompt=prompt,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
            memory_sample_interval_seconds=memory_sample_interval_seconds,
            include_memory_series=include_memory_series,
            apply_num_ctx=apply_num_ctx,
        ).to_dict()
        for job in jobs
    ]


def run_ollama_generate_benchmark(
    job: BenchJob,
    *,
    prompt: str = DEFAULT_PROMPT,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 128,
    memory_sample_interval_seconds: float = 0.5,
    include_memory_series: bool = False,
    apply_num_ctx: bool = True,
) -> BenchRunResult:
    started_at = _utc_now()
    wall_start = time.perf_counter()
    sampler = MemorySampler(interval_seconds=memory_sample_interval_seconds)
    sampler.start()
    try:
        payload = _call_ollama_generate(
            model=job.model,
            context=job.context,
            prompt=prompt,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
            apply_num_ctx=apply_num_ctx,
        )
        status = "ok" if payload.get("done", True) else "incomplete"
        error = None
    except Exception as exc:
        payload = {}
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    memory_summary = sampler.stop(include_series=include_memory_series)
    wall_seconds = time.perf_counter() - wall_start
    ended_at = _utc_now()

    eval_count = _int_or_none(payload.get("eval_count"))
    eval_duration = _int_or_none(payload.get("eval_duration"))
    prompt_eval_count = _int_or_none(payload.get("prompt_eval_count"))
    prompt_eval_duration = _int_or_none(payload.get("prompt_eval_duration"))
    return BenchRunResult(
        model=job.model,
        context=job.context,
        method="ollama-api",
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        wall_time_seconds=round(wall_seconds, 6),
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration_ns=prompt_eval_duration,
        eval_count=eval_count,
        eval_duration_ns=eval_duration,
        total_duration_ns=_int_or_none(payload.get("total_duration")),
        load_duration_ns=_int_or_none(payload.get("load_duration")),
        tokens_per_second=_tokens_per_second(eval_count, eval_duration),
        prompt_tokens_per_second=_tokens_per_second(prompt_eval_count, prompt_eval_duration),
        first_token_latency_ms=_first_token_latency_ms(payload),
        peak_memory_bytes=memory_summary["peak_memory_bytes"],
        memory_pressure=str(memory_summary["memory_pressure"]),
        swap_bytes=memory_summary["swap_bytes"],
        pageins_delta=memory_summary["pageins_delta"],
        pageout_delta=memory_summary["pageout_delta"],
        pageout_bytes_delta=memory_summary["pageout_bytes_delta"],
        swapins_delta=memory_summary["swapins_delta"],
        swapouts_delta=memory_summary["swapouts_delta"],
        memory_samples=int(memory_summary["memory_samples"] or 0),
        memory_sample_interval_seconds=memory_sample_interval_seconds,
        memory_series=memory_summary.get("memory_series") if include_memory_series else None,
        ollama_default_options=not apply_num_ctx,
        done_reason=_str_or_none(payload.get("done_reason")),
        error=error,
    )


def sample_memory_state() -> dict[str, int | str | None]:
    system = platform.system()
    if system == "Darwin":
        return {
            "memory_pressure": _darwin_memory_pressure(),
            "swap_bytes": _darwin_swap_used_bytes(),
            **_darwin_vm_stat(),
        }
    return {"memory_pressure": "unknown", "swap_bytes": None}


def sample_ollama_process_memory() -> int | None:
    system = platform.system()
    if system == "Windows":
        return _windows_process_rss_bytes("ollama")
    if system in {"Darwin", "Linux"}:
        return _posix_process_rss_bytes("ollama")
    return None


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._samples: list[dict[str, int | str | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._record_sample()
        self._thread = threading.Thread(target=self._run, name="mackv-opt-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self, *, include_series: bool = False) -> dict[str, object]:
        self._record_sample()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds * 2)
        self._record_sample()
        return self.summary(include_series=include_series)

    def summary(self, *, include_series: bool = False) -> dict[str, object]:
        process_values = [
            sample.get("process_memory_bytes")
            for sample in self._samples
            if isinstance(sample.get("process_memory_bytes"), int)
        ]
        swap_values = [sample.get("swap_bytes") for sample in self._samples if isinstance(sample.get("swap_bytes"), int)]
        page_size = _last_int_sample(self._samples, "page_size_bytes")
        pageout_delta = _counter_delta(self._samples, "pageouts")
        pressure = "unknown"
        for sample in self._samples:
            pressure = _merge_pressure(pressure, sample.get("memory_pressure"))
        result: dict[str, object] = {
            "peak_memory_bytes": max(process_values) if process_values else None,
            "memory_pressure": pressure,
            "swap_bytes": _swap_delta(swap_values[0], swap_values[-1]) if swap_values else None,
            "pageins_delta": _counter_delta(self._samples, "pageins"),
            "pageout_delta": pageout_delta,
            "pageout_bytes_delta": pageout_delta * page_size
            if pageout_delta is not None and page_size is not None
            else None,
            "swapins_delta": _counter_delta(self._samples, "swapins"),
            "swapouts_delta": _counter_delta(self._samples, "swapouts"),
            "memory_samples": len(self._samples),
        }
        if include_series:
            result["memory_series"] = [dict(sample) for sample in self._samples]
        return result

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record_sample()

    def _record_sample(self) -> None:
        memory = sample_memory_state()
        self._samples.append({"timestamp": _utc_now(), **memory, "process_memory_bytes": sample_ollama_process_memory()})


def _call_ollama_generate(
    *,
    model: str,
    context: int,
    prompt: str,
    base_url: str,
    timeout_seconds: float,
    num_predict: int | None,
    apply_num_ctx: bool = True,
) -> dict[str, object]:
    url = base_url.rstrip("/") + "/api/generate"
    options: dict[str, int] = {}
    if apply_num_ctx:
        options["num_ctx"] = context
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API returned HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama API is unavailable at {url}: {exc.reason}") from exc
    return json.loads(data)


def _darwin_memory_pressure() -> str:
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = (result.stdout + result.stderr).lower()
    if "critical" in text:
        return "critical"
    if "warn" in text:
        return "warning"
    if text:
        return "normal"
    return "unknown"


def _darwin_swap_used_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # Example: "total = 2048.00M  used = 512.00M  free = 1536.00M ..."
    text = result.stdout
    marker = "used ="
    if marker not in text:
        return None
    used = text.split(marker, 1)[1].strip().split()[0]
    return _parse_swap_amount(used)


def _darwin_vm_stat() -> dict[str, int]:
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    return _parse_darwin_vm_stat(result.stdout)


def _parse_darwin_vm_stat(text: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    page_size_match = re.search(r"page size of\s+(\d+)\s+bytes", text, re.IGNORECASE)
    if page_size_match:
        stats["page_size_bytes"] = int(page_size_match.group(1))

    key_map = {
        "pages free": "pages_free",
        "pages active": "pages_active",
        "pages inactive": "pages_inactive",
        "pages speculative": "pages_speculative",
        "pages wired down": "pages_wired",
        "pages purgeable": "pages_purgeable",
        "pages throttled": "pages_throttled",
        "pages stored in compressor": "pages_stored_in_compressor",
        "pages occupied by compressor": "pages_occupied_by_compressor",
        "compressions": "compressions",
        "decompressions": "decompressions",
        "pageins": "pageins",
        "pageouts": "pageouts",
        "swapins": "swapins",
        "swapouts": "swapouts",
    }
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        label, raw_value = raw_line.split(":", 1)
        key = key_map.get(label.strip().lower())
        if not key:
            continue
        match = re.search(r"([0-9][0-9,]*)", raw_value)
        if match:
            stats[key] = int(match.group(1).replace(",", ""))
    return stats


def _posix_process_rss_bytes(process_name: str) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-axo", "comm,rss"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    peak_kib = 0
    for line in result.stdout.splitlines()[1:]:
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        command, rss = parts
        if process_name.lower() not in command.lower():
            continue
        try:
            peak_kib = max(peak_kib, int(rss))
        except ValueError:
            continue
    return peak_kib * 1024 if peak_kib else None


def _windows_process_rss_bytes(process_name: str) -> int | None:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-Process -Name {process_name} -ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty WorkingSet64",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    values: list[int] = []
    for line in result.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def _parse_swap_amount(value: str) -> int | None:
    import re

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?)", value.strip(), re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).upper()
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[suffix]
    return int(number * scale)


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return round(count / duration_ns * 1_000_000_000, 6)


def _first_token_latency_ms(payload: dict[str, object]) -> float | None:
    # Ollama's non-streaming API does not expose true first-token latency. The
    # closest repeatable proxy is load + prompt evaluation time.
    load_duration = _int_or_none(payload.get("load_duration"))
    prompt_eval_duration = _int_or_none(payload.get("prompt_eval_duration"))
    if load_duration is None and prompt_eval_duration is None:
        return None
    return round(((load_duration or 0) + (prompt_eval_duration or 0)) / 1_000_000, 3)


def _swap_delta(before: object, after: object) -> int | None:
    if not isinstance(before, int) or not isinstance(after, int):
        return None
    return max(0, after - before)


def _counter_delta(samples: list[dict[str, int | str | None]], key: str) -> int | None:
    values = [sample.get(key) for sample in samples if isinstance(sample.get(key), int)]
    if not values:
        return None
    return max(0, values[-1] - values[0])


def _last_int_sample(samples: list[dict[str, int | str | None]], key: str) -> int | None:
    for sample in reversed(samples):
        value = sample.get(key)
        if isinstance(value, int):
            return value
    return None


def _peak_process_memory(before: int | None, after: int | None) -> int | None:
    values = [value for value in [before, after] if value is not None]
    return max(values) if values else None


def _merge_pressure(before: object, after: object) -> str:
    order = {"unknown": 0, "normal": 1, "warning": 2, "critical": 3}
    before_text = before if isinstance(before, str) else "unknown"
    after_text = after if isinstance(after, str) else "unknown"
    return before_text if order.get(before_text, 0) >= order.get(after_text, 0) else after_text


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
