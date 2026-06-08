from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


NUMERIC_METRICS = [
    "tokens_per_second",
    "prompt_tokens_per_second",
    "first_token_latency_ms",
    "wall_time_seconds",
    "peak_memory_bytes",
    "swap_bytes",
    "pageins_delta",
    "pageout_delta",
    "pageout_bytes_delta",
    "swapins_delta",
    "swapouts_delta",
    "quality_score",
]


def repeat_count(value: int | str | None) -> int:
    try:
        parsed = int(value) if value is not None else 1
    except (TypeError, ValueError):
        parsed = 1
    return max(1, parsed)


def repeat_runs(jobs: Iterable[Any], repeats: int, runner) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for job in jobs:
        for repeat_index in range(repeat_count(repeats)):
            result = runner(job)
            record = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            record["repeat_index"] = repeat_index
            record["repeat_count"] = repeat_count(repeats)
            runs.append(record)
    return runs


def summarize_repeated_runs(
    runs: Iterable[dict[str, Any]],
    *,
    group_keys: Iterable[str] = ("model", "context", "method"),
    numeric_metrics: Iterable[str] = NUMERIC_METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = list(group_keys)
    for run in runs:
        groups[tuple(run.get(key) for key in keys)].append(run)

    summaries: list[dict[str, Any]] = []
    for group_values, records in groups.items():
        summary = {key: value for key, value in zip(keys, group_values)}
        total = len(records)
        ok = sum(1 for record in records if record.get("status") == "ok")
        summary.update(
            {
                "runs": total,
                "ok_runs": ok,
                "error_runs": sum(1 for record in records if record.get("status") == "error"),
                "success_rate": round(ok / total, 6) if total else 0.0,
            }
        )
        if any("found" in record for record in records):
            found = sum(1 for record in records if record.get("found") is True)
            summary["found_runs"] = found
            summary["accuracy"] = round(found / total, 6) if total else 0.0
        for metric in numeric_metrics:
            values = [_float_or_none(record.get(metric)) for record in records]
            clean = [value for value in values if value is not None]
            if not clean:
                continue
            summary[f"{metric}_mean"] = round(sum(clean) / len(clean), 6)
            summary[f"{metric}_stdev"] = round(_sample_stdev(clean), 6)
            summary[f"{metric}_min"] = round(min(clean), 6)
            summary[f"{metric}_max"] = round(max(clean), 6)
        summaries.append(summary)
    return summaries


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
