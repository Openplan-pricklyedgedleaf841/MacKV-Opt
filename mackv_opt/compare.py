from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Iterable

from .report import load_report_payload, normalize_report_rows

COMPARE_COLUMNS = [
    "label",
    "baseline_label",
    "model",
    "max_stable_context",
    "max_stable_context_vs_baseline",
    "best_context",
    "best_context_vs_baseline",
    "tokens_per_second_mean",
    "tokens_per_second",
    "tokens_per_second_vs_baseline",
    "first_token_latency_ms_mean",
    "first_token_latency_ms",
    "first_token_latency_ms_vs_baseline",
    "peak_memory_bytes_mean",
    "peak_memory_bytes",
    "peak_memory_bytes_vs_baseline",
    "swap_bytes_mean",
    "memory_pressure",
    "success_rate",
    "quality_accuracy",
    "quality_score_mean",
    "quality_accuracy_vs_baseline",
    "stable_context_policy",
    "source",
]


@dataclass(frozen=True)
class CompareInput:
    label: str
    path: str


def parse_compare_input(value: str) -> CompareInput:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if label and path:
            return CompareInput(label=label, path=path)
    path = value.strip()
    return CompareInput(label=_label_from_path(path), path=path)


def build_compare_payload(
    inputs: Iterable[CompareInput | str],
    *,
    baseline_label: str | None = None,
) -> dict[str, object]:
    parsed = [parse_compare_input(item) if isinstance(item, str) else item for item in inputs]
    rows = [_compare_input_row(item) for item in parsed]
    selected_baseline = baseline_label or (rows[0]["label"] if rows else "")
    rows = _annotate_vs_baseline(rows, selected_baseline)
    return {
        "task": "compare",
        "baseline_label": selected_baseline,
        "inputs": [{"label": item.label, "path": item.path} for item in parsed],
        "rows": rows,
        "columns": COMPARE_COLUMNS,
    }


def render_compare_markdown(rows: Iterable[dict[str, Any]]) -> str:
    return _render_markdown_table(rows, COMPARE_COLUMNS)


def render_compare_csv(rows: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COMPARE_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in COMPARE_COLUMNS})
    return output.getvalue()


def _compare_input_row(item: CompareInput) -> dict[str, Any]:
    payload = load_report_payload(item.path)
    rows = _artifact_rows(payload)
    model = _first_non_empty(row.get("model") for row in rows)
    stability_rows = [row for row in rows if row.get("method") == "stability-summary"]
    performance_rows = [row for row in rows if row.get("method") == "ollama-api"]
    quality_rows = [row for row in rows if row.get("method") in {"needle", "qa"} or row.get("accuracy") != ""]

    best_performance = _best_performance_row(performance_rows)
    stability = _best_stability_row(stability_rows)
    quality = _best_quality_row(quality_rows)

    return {
        "label": item.label,
        "baseline_label": "",
        "model": model or _first_non_empty(row.get("model") for row in performance_rows + stability_rows + quality_rows),
        "max_stable_context": _value_or_empty(stability.get("max_stable_context")),
        "max_stable_context_vs_baseline": "",
        "best_context": _value_or_empty(best_performance.get("context")),
        "best_context_vs_baseline": "",
        "tokens_per_second_mean": _value_or_empty(best_performance.get("tokens_per_second_mean")),
        "tokens_per_second": _value_or_empty(best_performance.get("tokens_per_second")),
        "tokens_per_second_vs_baseline": "",
        "first_token_latency_ms_mean": _value_or_empty(best_performance.get("first_token_latency_ms_mean")),
        "first_token_latency_ms": _value_or_empty(best_performance.get("first_token_latency_ms")),
        "first_token_latency_ms_vs_baseline": "",
        "peak_memory_bytes_mean": _value_or_empty(best_performance.get("peak_memory_bytes_mean")),
        "peak_memory_bytes": _value_or_empty(best_performance.get("peak_memory_bytes")),
        "peak_memory_bytes_vs_baseline": "",
        "swap_bytes_mean": _value_or_empty(best_performance.get("swap_bytes_mean")),
        "memory_pressure": best_performance.get("memory_pressure") or "",
        "success_rate": _value_or_empty(best_performance.get("success_rate")),
        "quality_accuracy": _value_or_empty(quality.get("accuracy")),
        "quality_score_mean": _value_or_empty(quality.get("quality_score_mean") or quality.get("quality_score")),
        "quality_accuracy_vs_baseline": "",
        "stable_context_policy": stability.get("stable_context_policy") or "",
        "source": item.path,
    }


def _annotate_vs_baseline(rows: list[dict[str, Any]], baseline_label: str) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row.get("label") == baseline_label), rows[0] if rows else {})
    annotated: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["baseline_label"] = baseline.get("label") or ""
        record["max_stable_context_vs_baseline"] = _ratio(
            record.get("max_stable_context"),
            baseline.get("max_stable_context"),
        )
        record["best_context_vs_baseline"] = _ratio(record.get("best_context"), baseline.get("best_context"))
        record["tokens_per_second_vs_baseline"] = _ratio(
            record.get("tokens_per_second_mean") or record.get("tokens_per_second"),
            baseline.get("tokens_per_second_mean") or baseline.get("tokens_per_second"),
        )
        record["first_token_latency_ms_vs_baseline"] = _ratio(
            record.get("first_token_latency_ms_mean") or record.get("first_token_latency_ms"),
            baseline.get("first_token_latency_ms_mean") or baseline.get("first_token_latency_ms"),
        )
        record["peak_memory_bytes_vs_baseline"] = _ratio(
            record.get("peak_memory_bytes_mean") or record.get("peak_memory_bytes"),
            baseline.get("peak_memory_bytes_mean") or baseline.get("peak_memory_bytes"),
        )
        record["quality_accuracy_vs_baseline"] = _delta(
            record.get("quality_accuracy"),
            baseline.get("quality_accuracy"),
        )
        annotated.append(record)
    return annotated


def _best_stability_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: _number(row.get("max_stable_context")), default={})


def _best_performance_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            _number(row.get("tokens_per_second_mean") or row.get("tokens_per_second")),
            1 if row.get("tokens_per_second_mean") not in (None, "") else 0,
            _number(row.get("context")),
        ),
        default={},
    )


def _best_quality_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            _number(row.get("accuracy")),
            _number(row.get("quality_score_mean") or row.get("quality_score")),
        ),
        default={},
    )


def _artifact_rows(payload: Any) -> list[dict[str, Any]]:
    rows = normalize_report_rows(payload)
    if not isinstance(payload, dict) or payload.get("task") == "experiment":
        return rows
    stability_summary = payload.get("stability_summary")
    if isinstance(stability_summary, dict):
        rows.extend(normalize_report_rows({"runs": [], "stability_summary": stability_summary}))
    for section_name in ["bench", "needle", "qa"]:
        section = payload.get(section_name)
        if isinstance(section, dict):
            rows.extend(normalize_report_rows(section))
    return rows


def _render_markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _label_from_path(path: str) -> str:
    clean = path.replace("\\", "/").rstrip("/")
    name = clean.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] or "artifact"


def _first_non_empty(values: Iterable[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _number(value: Any) -> float:
    if value in (None, "") or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _number_or_none(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(value: Any, baseline: Any) -> float | str:
    numerator = _number_or_none(value)
    denominator = _number_or_none(baseline)
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return round(numerator / denominator, 6)


def _delta(value: Any, baseline: Any) -> float | str:
    current = _number_or_none(value)
    reference = _number_or_none(baseline)
    if current is None or reference is None:
        return ""
    return round(current - reference, 6)


def _value_or_empty(value: Any) -> Any:
    return "" if value is None else value


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
