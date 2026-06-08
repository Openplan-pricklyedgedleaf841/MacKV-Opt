from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .compare import CompareInput, build_compare_payload

RQ1_LABELS = ("default", "manual-num-ctx", "mackv-opt")

RQ1_COLUMNS = [
    "model",
    "default_max_stable_context",
    "manual_num_ctx_max_stable_context",
    "mackv_opt_max_stable_context",
    "mackv_opt_vs_default",
    "mackv_opt_vs_manual_num_ctx",
    "best_label",
    "best_max_stable_context",
    "default_tokens_per_second",
    "manual_num_ctx_tokens_per_second",
    "mackv_opt_tokens_per_second",
    "default_quality_accuracy",
    "manual_num_ctx_quality_accuracy",
    "mackv_opt_quality_accuracy",
    "complete",
    "source",
]


def build_rq1_summary_payload(machine_dir: str) -> dict[str, Any]:
    root = Path(machine_dir)
    rows = [_model_rq1_row(model_dir) for model_dir in sorted(root.iterdir()) if model_dir.is_dir()]
    rows = [row for row in rows if row]
    return {
        "task": "rq1-summary",
        "machine_dir": str(root),
        "labels": list(RQ1_LABELS),
        "columns": RQ1_COLUMNS,
        "rows": rows,
        "summary": {
            "model_count": len(rows),
            "complete_model_count": sum(1 for row in rows if row.get("complete") is True),
            "models_missing_baselines": [
                row["model"] for row in rows if row.get("complete") is not True
            ],
        },
    }


def render_rq1_markdown(payload: dict[str, Any]) -> str:
    return _render_markdown_table(payload.get("rows") or [], RQ1_COLUMNS)


def render_rq1_csv(payload: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=RQ1_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in payload.get("rows") or []:
        writer.writerow({column: row.get(column, "") for column in RQ1_COLUMNS})
    return output.getvalue()


def render_rq1_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _model_rq1_row(model_dir: Path) -> dict[str, Any] | None:
    artifacts = {
        label: model_dir / label / "full-run.json"
        for label in RQ1_LABELS
    }
    existing = {label: path for label, path in artifacts.items() if path.exists()}
    if not existing:
        return None
    compare = build_compare_payload(
        [CompareInput(label=label, path=str(path)) for label, path in existing.items()],
        baseline_label="default" if "default" in existing else next(iter(existing)),
    )
    by_label = {row.get("label"): row for row in compare.get("rows", []) if isinstance(row, dict)}
    model = _first_non_empty(row.get("model") for row in by_label.values()) or model_dir.name
    default = by_label.get("default", {})
    manual = by_label.get("manual-num-ctx", {})
    optimized = by_label.get("mackv-opt", {})
    best = _best_label(by_label)
    return {
        "model": model,
        "default_max_stable_context": _value_or_empty(default.get("max_stable_context")),
        "manual_num_ctx_max_stable_context": _value_or_empty(manual.get("max_stable_context")),
        "mackv_opt_max_stable_context": _value_or_empty(optimized.get("max_stable_context")),
        "mackv_opt_vs_default": _ratio(
            optimized.get("max_stable_context"),
            default.get("max_stable_context"),
        ),
        "mackv_opt_vs_manual_num_ctx": _ratio(
            optimized.get("max_stable_context"),
            manual.get("max_stable_context"),
        ),
        "best_label": best,
        "best_max_stable_context": _value_or_empty(by_label.get(best, {}).get("max_stable_context")),
        "default_tokens_per_second": _value_or_empty(
            default.get("tokens_per_second_mean") or default.get("tokens_per_second")
        ),
        "manual_num_ctx_tokens_per_second": _value_or_empty(
            manual.get("tokens_per_second_mean") or manual.get("tokens_per_second")
        ),
        "mackv_opt_tokens_per_second": _value_or_empty(
            optimized.get("tokens_per_second_mean") or optimized.get("tokens_per_second")
        ),
        "default_quality_accuracy": _value_or_empty(default.get("quality_accuracy")),
        "manual_num_ctx_quality_accuracy": _value_or_empty(manual.get("quality_accuracy")),
        "mackv_opt_quality_accuracy": _value_or_empty(optimized.get("quality_accuracy")),
        "complete": all(label in existing for label in RQ1_LABELS),
        "source": str(model_dir),
    }


def _best_label(rows: dict[Any, dict[str, Any]]) -> str:
    best = ""
    best_context = -1.0
    for label, row in rows.items():
        context = _number_or_none(row.get("max_stable_context"))
        if context is not None and context > best_context:
            best = str(label)
            best_context = context
    return best


def _ratio(value: Any, baseline: Any) -> float | str:
    numerator = _number_or_none(value)
    denominator = _number_or_none(baseline)
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return round(numerator / denominator, 6)


def _number_or_none(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty(values) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _value_or_empty(value: Any) -> Any:
    return "" if value is None else value


def _render_markdown_table(rows, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
