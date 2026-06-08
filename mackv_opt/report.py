from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from .models import HardwareProfile, ModelProfile, OptimizationPlan
from .units import format_bytes

REPORT_COLUMNS = [
    "model",
    "context",
    "method",
    "status",
    "cache_type_k",
    "cache_type_v",
    "depth",
    "source",
    "question_id",
    "found",
    "first_token_latency_ms",
    "first_token_latency_ms_mean",
    "first_token_latency_ms_stdev",
    "tokens_per_second",
    "tokens_per_second_mean",
    "tokens_per_second_stdev",
    "peak_memory_bytes",
    "peak_memory_bytes_mean",
    "memory_pressure",
    "swap_bytes",
    "swap_bytes_mean",
    "pageout_delta",
    "pageout_delta_mean",
    "pageout_bytes_delta",
    "pageout_bytes_delta_mean",
    "swapins_delta",
    "swapouts_delta",
    "stable",
    "stability_reason",
    "instability_reasons",
    "context_stable",
    "stable_fraction",
    "stable_context_policy",
    "min_stable_fraction",
    "max_stable_context",
    "stable_runs",
    "unstable_runs",
    "quality_score",
    "quality_score_mean",
    "quality_score_stdev",
    "success_rate",
    "accuracy",
    "runs",
    "response_excerpt",
]

PAPER_TABLES: dict[str, list[str]] = {
    "readiness-compact": [
        "paper_ready",
        "status",
        "artifact_type",
        "component_count",
        "model_count",
        "platform",
        "machine",
        "chip",
        "is_apple_silicon",
        "os_version",
        "power_source",
        "power_mode",
        "thermal_state",
        "memory_pressure",
        "ollama_available",
        "ollama_model_count",
        "llama_cpp_available",
        "supports_ollama_num_ctx",
        "supports_llama_cpp_cache_type_k",
        "supports_llama_cpp_cache_type_v",
        "metadata_complete",
        "metadata_override_models",
        "failed_checks",
        "warning_checks",
        "next_step",
    ],
    "readiness": [
        "artifact_type",
        "component",
        "status",
        "model",
        "model_count",
        "check",
        "message",
        "platform",
        "machine",
        "chip",
        "is_apple_silicon",
        "os_version",
        "kernel_version",
        "total_memory_bytes",
        "available_memory_bytes",
        "power_source",
        "power_mode",
        "thermal_state",
        "memory_pressure",
        "swap_bytes",
        "ollama_available",
        "ollama_version",
        "ollama_model_count",
        "llama_cpp_available",
        "llama_cpp_version",
        "supports_ollama_num_ctx",
        "supports_llama_cpp_cache_type_k",
        "supports_llama_cpp_cache_type_v",
        "model_status",
        "metadata_status",
        "missing_required_metadata",
        "override_applied",
        "failed_checks",
        "warning_checks",
        "next_step",
    ],
    "context": [
        "model",
        "context",
        "status",
        "cache_type_k",
        "cache_type_v",
        "peak_memory_bytes",
        "memory_pressure",
        "stable",
        "stability_reason",
        "max_stable_context",
        "stable_runs",
        "unstable_runs",
    ],
    "performance": [
        "model",
        "context",
        "method",
        "tokens_per_second",
        "tokens_per_second_mean",
        "tokens_per_second_stdev",
        "first_token_latency_ms",
        "first_token_latency_ms_mean",
        "first_token_latency_ms_stdev",
        "memory_pressure",
        "stable",
        "stability_reason",
        "max_stable_context",
        "success_rate",
    ],
    "memory": [
        "model",
        "context",
        "method",
        "peak_memory_bytes",
        "peak_memory_bytes_mean",
        "swap_bytes",
        "swap_bytes_mean",
        "pageout_delta",
        "pageout_delta_mean",
        "pageout_bytes_delta",
        "pageout_bytes_delta_mean",
        "swapouts_delta",
        "memory_pressure",
        "success_rate",
    ],
    "quality": [
        "model",
        "context",
        "depth",
        "source",
        "question_id",
        "found",
        "quality_score",
        "quality_score_mean",
        "quality_score_stdev",
        "accuracy",
        "tokens_per_second",
        "tokens_per_second_mean",
    ],
    "stability": [
        "model",
        "context",
        "stable_context_policy",
        "min_stable_fraction",
        "context_stable",
        "stable_fraction",
        "stable_runs",
        "unstable_runs",
        "runs",
        "instability_reasons",
        "max_stable_context",
    ],
}


def render_plan_text(
    plan: OptimizationPlan,
    model: ModelProfile,
    hardware: HardwareProfile,
) -> str:
    lines = [
        f"MacKV-Opt plan for {model.name}",
        "",
        f"Status: {plan.status}",
        f"Hardware: {hardware.chip} ({hardware.platform}/{hardware.machine})",
        f"Memory budget: {format_bytes(plan.memory_budget_bytes)}",
        f"Recommended context: {plan.num_ctx} tokens",
        f"KV cache: K={plan.cache_type_k}, V={plan.cache_type_v}, offload={plan.kv_offload}",
        f"Estimated model memory: {format_bytes(plan.estimated_model_bytes)}",
        f"Estimated KV memory: {format_bytes(plan.estimated_kv_bytes)}",
        f"Estimated total memory: {format_bytes(plan.estimated_total_bytes)}",
        "",
        "Ollama options:",
    ]
    for key, value in plan.ollama_options.items():
        lines.append(f"  --option {key}={value}")
    lines.extend(["", "llama.cpp args:", "  " + " ".join(plan.llama_cpp_args)])
    if plan.runtime_advice:
        lines.append("")
        lines.append("Runtime advice:")
        lines.append(f"  - capability check: {plan.runtime_advice.get('checked')}")
        if plan.runtime_advice.get("checked"):
            lines.append(f"  - Ollama ready: {plan.runtime_advice.get('ollama_ready')}")
            lines.append(f"  - llama.cpp ready: {plan.runtime_advice.get('llama_cpp_ready')}")
    if plan.reasons:
        lines.append("")
        lines.append("Reasons:")
        lines.extend(f"  - {reason}" for reason in plan.reasons)
    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def load_report_payload(path: str) -> Any:
    source_path = Path(path)
    if source_path.is_dir():
        manifest = source_path / "manifest.json"
        if manifest.exists():
            bundle = [_load_json_with_source(manifest)]
            audit = source_path / "audit.json"
            if audit.exists():
                bundle.append(_load_json_with_source(audit))
            return bundle
        bundle = [
            _load_json_with_source(candidate)
            for candidate in [
                source_path / "audit.json",
                source_path / "doctor.json",
                source_path / "machine-profile.json",
                source_path / "runtime-capabilities.json",
            ]
            if candidate.exists()
        ]
        if bundle:
            return bundle
        for name in ["full-run.json", "experiment.json"]:
            candidate = source_path / name
            if candidate.exists():
                source_path = candidate
                break
    with open(source_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def normalize_report_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if any(_is_report_payload(item) for item in payload if isinstance(item, dict)):
            rows: list[dict[str, Any]] = []
            for item in payload:
                rows.extend(normalize_report_rows(item))
            return rows
        records = payload
    elif isinstance(payload, dict) and payload.get("task") == "collect":
        records = _collect_readiness_records(payload)
    elif isinstance(payload, dict) and payload.get("task") == "audit":
        records = _audit_readiness_records(payload)
    elif isinstance(payload, dict) and payload.get("task") == "doctor":
        records = _doctor_readiness_records(payload)
    elif isinstance(payload, dict) and _is_profile_payload(payload):
        records = _profile_readiness_records(payload)
    elif isinstance(payload, dict) and _is_capabilities_payload(payload):
        records = _capabilities_readiness_records(payload)
    elif isinstance(payload, dict) and payload.get("task") == "experiment":
        records = _experiment_records(payload)
    elif isinstance(payload, dict) and any(isinstance(payload.get(key), list) for key in ["runs", "jobs", "repeat_summaries"]):
        records = _section_records(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("plan"), dict):
        records = [payload]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        records = []

    return [_normalize_row(record) for record in records if isinstance(record, dict)]


def _experiment_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    model_name = model.get("name") or payload.get("model")
    for plan in payload.get("plans") or []:
        if isinstance(plan, dict):
            records.append({"model": model_name, "method": "plan", "plan": plan})
    for section_name in ["bench", "needle", "qa"]:
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        records.extend(_section_records(section))
    return records


def _section_records(section: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ["runs", "jobs", "repeat_summaries"]:
        for record in section.get(key) or []:
            if isinstance(record, dict):
                records.append(record)
    stability_summary = section.get("stability_summary")
    if isinstance(stability_summary, dict):
        for record in stability_summary.get("models") or []:
            if isinstance(record, dict):
                records.append({**record, "method": "stability-summary"})
                for context_record in _stability_context_records(record, stability_summary):
                    records.append(context_record)
    return records


def render_report_markdown(rows: Iterable[dict[str, Any]]) -> str:
    return _render_markdown_table(list(rows), REPORT_COLUMNS)


def render_report_csv(rows: Iterable[dict[str, Any]]) -> str:
    return _render_csv_table(rows, REPORT_COLUMNS)


def render_paper_table_markdown(rows: Iterable[dict[str, Any]], table: str) -> str:
    columns = _paper_table_columns(table)
    filtered = _paper_table_rows((_normalize_row(row) for row in rows), table)
    return _render_markdown_table(filtered, columns)


def render_paper_table_csv(rows: Iterable[dict[str, Any]], table: str) -> str:
    columns = _paper_table_columns(table)
    filtered = _paper_table_rows((_normalize_row(row) for row in rows), table)
    return _render_csv_table(filtered, columns)


def write_paper_tables(
    rows: Iterable[dict[str, Any]],
    output_dir: str,
    *,
    prefix: str = "mackv-opt-table",
    tables: Iterable[str] | None = None,
    fmt: str = "markdown",
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    normalized = list(rows)
    selected = list(tables) if tables is not None else list(PAPER_TABLES)
    extension = "csv" if fmt == "csv" else "md"
    written: dict[str, str] = {}
    for table in selected:
        name = table.strip()
        if not name:
            continue
        if fmt == "csv":
            rendered = render_paper_table_csv(normalized, name)
        else:
            rendered = render_paper_table_markdown(normalized, name) + "\n"
        path = target / f"{prefix}-{name}.{extension}"
        path.write_text(rendered, encoding="utf-8")
        written[name] = str(path)
    return written


def _render_markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    normalized = list(rows)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in normalized:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _render_csv_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


def _paper_table_columns(table: str) -> list[str]:
    try:
        return PAPER_TABLES[table]
    except KeyError as exc:
        choices = ", ".join(sorted(PAPER_TABLES))
        raise ValueError(f"Unknown paper table {table!r}; choose one of: {choices}") from exc


def _paper_table_rows(rows: Iterable[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    normalized = list(rows)
    if table == "readiness":
        return [row for row in normalized if row.get("method") == "readiness"]
    if table == "readiness-compact":
        return [_normalize_row(_compact_readiness_row([row for row in normalized if row.get("method") == "readiness"]))]
    if table == "context":
        return [row for row in normalized if row.get("method") in {"plan", "mackv-opt", "stability-summary"}]
    if table == "performance":
        return [row for row in normalized if row.get("method") == "ollama-api"]
    if table == "memory":
        return [row for row in normalized if row.get("method") in {"ollama-api", "plan"}]
    if table == "quality":
        return [row for row in normalized if row.get("method") in {"needle", "qa"} or row.get("found") != ""]
    if table == "stability":
        return [row for row in normalized if row.get("method") in {"stability-context", "stability-summary"}]
    _paper_table_columns(table)
    return normalized


def write_experiment_artifacts(
    payload: Any,
    output_dir: str,
    *,
    prefix: str = "mackv-opt-bench",
    formats: Iterable[str] = ("json", "markdown", "csv"),
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected = {item.strip().lower() for item in formats if item and item.strip()}
    rows = normalize_report_rows(payload)
    written: dict[str, str] = {}

    if "json" in selected:
        path = target / f"{prefix}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        written["json"] = str(path)
    if "markdown" in selected or "md" in selected:
        path = target / f"{prefix}.md"
        path.write_text(render_report_markdown(rows) + "\n", encoding="utf-8")
        written["markdown"] = str(path)
    if "csv" in selected:
        path = target / f"{prefix}.csv"
        path.write_text(render_report_csv(rows), encoding="utf-8")
        written["csv"] = str(path)
    return written


def _normalize_row(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    source = {**record, **plan}
    return {
        "model": _model_name(source, model, record),
        "context": source.get("context") or source.get("num_ctx") or source.get("target_context") or "",
        "method": source.get("method") or source.get("mode") or "mackv-opt",
        "status": source.get("status") or "",
        "cache_type_k": source.get("cache_type_k") or "",
        "cache_type_v": source.get("cache_type_v") or "",
        "depth": source.get("depth") if source.get("depth") is not None else "",
        "source": source.get("source") or "",
        "question_id": source.get("question_id") or source.get("id") or "",
        "found": source.get("found") if source.get("found") is not None else "",
        "first_token_latency_ms": source.get("first_token_latency_ms") or "",
        "first_token_latency_ms_mean": _value_or_empty(source.get("first_token_latency_ms_mean")),
        "first_token_latency_ms_stdev": _value_or_empty(source.get("first_token_latency_ms_stdev")),
        "tokens_per_second": source.get("tokens_per_second") or "",
        "tokens_per_second_mean": _value_or_empty(source.get("tokens_per_second_mean")),
        "tokens_per_second_stdev": _value_or_empty(source.get("tokens_per_second_stdev")),
        "peak_memory_bytes": source.get("peak_memory_bytes") or "",
        "peak_memory_bytes_mean": _value_or_empty(source.get("peak_memory_bytes_mean")),
        "memory_pressure": source.get("memory_pressure") or "",
        "swap_bytes": _value_or_empty(source.get("swap_bytes")),
        "swap_bytes_mean": _value_or_empty(source.get("swap_bytes_mean")),
        "pageout_delta": _value_or_empty(source.get("pageout_delta")),
        "pageout_delta_mean": _value_or_empty(source.get("pageout_delta_mean")),
        "pageout_bytes_delta": _value_or_empty(source.get("pageout_bytes_delta")),
        "pageout_bytes_delta_mean": _value_or_empty(source.get("pageout_bytes_delta_mean")),
        "swapins_delta": _value_or_empty(source.get("swapins_delta")),
        "swapouts_delta": _value_or_empty(source.get("swapouts_delta")),
        "stable": source.get("stable") if source.get("stable") is not None else "",
        "stability_reason": source.get("stability_reason") or "",
        "instability_reasons": _format_reasons(source.get("instability_reasons") or source.get("reasons")),
        "context_stable": source.get("context_stable") if source.get("context_stable") is not None else "",
        "stable_fraction": _value_or_empty(source.get("stable_fraction")),
        "stable_context_policy": source.get("stable_context_policy") or "",
        "min_stable_fraction": _value_or_empty(source.get("min_stable_fraction")),
        "max_stable_context": _value_or_empty(source.get("max_stable_context")),
        "stable_runs": _value_or_empty(source.get("stable_runs")),
        "unstable_runs": _value_or_empty(source.get("unstable_runs")),
        "quality_score": _value_or_empty(source.get("quality_score")),
        "quality_score_mean": _value_or_empty(source.get("quality_score_mean")),
        "quality_score_stdev": _value_or_empty(source.get("quality_score_stdev")),
        "success_rate": _value_or_empty(source.get("success_rate")),
        "accuracy": _value_or_empty(source.get("accuracy")),
        "runs": _value_or_empty(source.get("runs")),
        "response_excerpt": source.get("response_excerpt") or "",
        "artifact_type": source.get("artifact_type") or "",
        "component": source.get("component") or "",
        "model_count": _value_or_empty(source.get("model_count")),
        "check": source.get("check") or "",
        "message": source.get("message") or "",
        "platform": source.get("platform") or "",
        "machine": source.get("machine") or "",
        "chip": source.get("chip") or "",
        "is_apple_silicon": source.get("is_apple_silicon")
        if source.get("is_apple_silicon") is not None
        else "",
        "os_version": source.get("os_version") or "",
        "kernel_version": source.get("kernel_version") or "",
        "total_memory_bytes": _value_or_empty(source.get("total_memory_bytes")),
        "available_memory_bytes": _value_or_empty(source.get("available_memory_bytes")),
        "power_source": source.get("power_source") or "",
        "power_mode": source.get("power_mode") or "",
        "thermal_state": source.get("thermal_state") or "",
        "ollama_available": source.get("ollama_available")
        if source.get("ollama_available") is not None
        else "",
        "ollama_version": source.get("ollama_version") or "",
        "ollama_model_count": _value_or_empty(source.get("ollama_model_count")),
        "llama_cpp_available": source.get("llama_cpp_available")
        if source.get("llama_cpp_available") is not None
        else "",
        "llama_cpp_version": source.get("llama_cpp_version") or "",
        "supports_ollama_num_ctx": source.get("supports_ollama_num_ctx")
        if source.get("supports_ollama_num_ctx") is not None
        else "",
        "supports_llama_cpp_cache_type_k": source.get("supports_llama_cpp_cache_type_k")
        if source.get("supports_llama_cpp_cache_type_k") is not None
        else "",
        "supports_llama_cpp_cache_type_v": source.get("supports_llama_cpp_cache_type_v")
        if source.get("supports_llama_cpp_cache_type_v") is not None
        else "",
        "model_status": source.get("model_status") or "",
        "metadata_status": source.get("metadata_status") or "",
        "missing_required_metadata": _format_list(source.get("missing_required_metadata")),
        "override_applied": source.get("override_applied")
        if source.get("override_applied") is not None
        else "",
        "failed_checks": _format_list(source.get("failed_checks")),
        "warning_checks": _format_list(source.get("warning_checks")),
        "next_step": source.get("next_step") or "",
        "paper_ready": source.get("paper_ready") if source.get("paper_ready") is not None else "",
        "component_count": _value_or_empty(source.get("component_count")),
        "metadata_complete": source.get("metadata_complete")
        if source.get("metadata_complete") is not None
        else "",
        "metadata_override_models": _format_list(source.get("metadata_override_models")),
    }


def _collect_readiness_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_path = payload.get("_mackv_opt_source_path") or payload.get("artifacts", {}).get("manifest")
    manifest_dir = Path(source_path).parent if source_path else Path.cwd()
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    records.append(
        {
            "method": "readiness",
            "artifact_type": "collect-manifest",
            "component": "summary",
            "status": payload.get("doctor_status") or "unknown",
            "model_count": payload.get("model_count", len(models)),
            "message": "Collection manifest summary.",
            "source": source_path or "",
        }
    )
    for name, value in sorted(_dict_value(payload.get("artifacts")).items()):
        records.append(
            {
                "method": "readiness",
                "artifact_type": "collect-manifest",
                "component": "artifact",
                "status": "recorded" if value else "missing",
                "check": str(name),
                "message": str(value or ""),
                "source": source_path or "",
            }
        )
    for record in models:
        if not isinstance(record, dict):
            continue
        audit = _dict_value(record.get("metadata_audit"))
        override = _dict_value(record.get("metadata_override"))
        profile = _dict_value(record.get("profile"))
        records.append(
            {
                "method": "readiness",
                "artifact_type": "collect-manifest",
                "component": "model-metadata",
                "status": audit.get("status") or record.get("status") or "unknown",
                "model": record.get("name") or profile.get("name") or "",
                "message": _metadata_message(audit),
                "model_status": record.get("status") or "",
                "metadata_status": audit.get("status") or "",
                "missing_required_metadata": audit.get("missing_required_fields") or [],
                "override_applied": override.get("applied"),
                "source": record.get("normalized_profile_json") or source_path or "",
            }
        )
    for artifact_name, converter in [
        ("doctor", _doctor_readiness_records),
        ("profile", _profile_readiness_records),
        ("capabilities", _capabilities_readiness_records),
    ]:
        artifact_payload = _load_referenced_artifact(payload, artifact_name, manifest_dir)
        if isinstance(artifact_payload, dict):
            records.extend(converter(artifact_payload))
    return records


def _audit_readiness_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = _dict_value(payload.get("summary"))
    records: list[dict[str, Any]] = [
        {
            "method": "readiness",
            "artifact_type": "collect-audit",
            "component": "summary",
            "status": payload.get("status") or "unknown",
            "model_count": summary.get("model_count"),
            "message": "Collection audit summary.",
            "failed_checks": summary.get("failed_checks") or [],
            "warning_checks": summary.get("warning_checks") or [],
            "source": payload.get("_mackv_opt_source_path") or payload.get("manifest_path") or "",
        }
    ]
    for check in payload.get("checks") or []:
        if not isinstance(check, dict):
            continue
        evidence = _dict_value(check.get("evidence"))
        records.append(
            {
                "method": "readiness",
                "artifact_type": "collect-audit",
                "component": check.get("name") or "check",
                "check": check.get("name") or "",
                "status": check.get("status") or "unknown",
                "message": check.get("message") or "",
                "next_step": check.get("next_step") or "",
                "source": payload.get("_mackv_opt_source_path") or payload.get("manifest_path") or "",
                **_readiness_evidence_fields(evidence),
            }
        )
        records.extend(_audit_model_issue_records(check, payload))
    return records


def _doctor_readiness_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hardware = _dict_value(payload.get("hardware"))
    caps = _dict_value(payload.get("capabilities"))
    memory = _dict_value(payload.get("memory_state"))
    records: list[dict[str, Any]] = [
        {
            "method": "readiness",
            "artifact_type": "doctor",
            "component": "summary",
            "status": payload.get("status") or "unknown",
            "message": "Doctor preflight summary.",
            "ollama_model_count": payload.get("ollama_model_count"),
            "source": payload.get("_mackv_opt_source_path") or "",
            **_hardware_fields(hardware),
            **_capability_fields(caps),
            **_memory_fields(memory),
        }
    ]
    for check in payload.get("checks") or []:
        if not isinstance(check, dict):
            continue
        evidence = _dict_value(check.get("evidence"))
        records.append(
            {
                "method": "readiness",
                "artifact_type": "doctor",
                "component": check.get("name") or "check",
                "check": check.get("name") or "",
                "status": check.get("status") or "unknown",
                "message": check.get("message") or "",
                "next_step": check.get("next_step") or "",
                "source": payload.get("_mackv_opt_source_path") or "",
                **_readiness_evidence_fields(evidence),
            }
        )
    return records


def _profile_readiness_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hardware = _dict_value(payload.get("hardware"))
    ollama = _dict_value(payload.get("ollama"))
    models = ollama.get("models") if isinstance(ollama.get("models"), list) else []
    caps = _dict_value(payload.get("capabilities"))
    return [
        {
            "method": "readiness",
            "artifact_type": "machine-profile",
            "component": "hardware",
            "status": "pass" if _is_apple_silicon(hardware) else "warn",
            "message": "Machine profile hardware summary.",
            "source": payload.get("_mackv_opt_source_path") or "",
            **_hardware_fields(hardware),
        },
        {
            "method": "readiness",
            "artifact_type": "machine-profile",
            "component": "runtime",
            "status": _ollama_status(_dict_value(caps.get("ollama") or ollama), len(models)),
            "message": "Machine profile runtime summary.",
            "ollama_model_count": len(models),
            "source": payload.get("_mackv_opt_source_path") or "",
            **_capability_fields(caps),
        },
    ]


def _capabilities_readiness_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    caps = dict(payload)
    return [
        {
            "method": "readiness",
            "artifact_type": "runtime-capabilities",
            "component": "runtime",
            "status": _runtime_status(caps),
            "message": "Runtime capability probe summary.",
            "source": payload.get("_mackv_opt_source_path") or "",
            **_capability_fields(caps),
        }
    ]


def _audit_model_issue_records(check: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    if check.get("name") != "model-metadata":
        return []
    evidence = _dict_value(check.get("evidence"))
    records: list[dict[str, Any]] = []
    for item in evidence.get("incomplete_models") or []:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "method": "readiness",
                "artifact_type": "collect-audit",
                "component": "model-metadata",
                "status": check.get("status") or "unknown",
                "model": item.get("name") or "",
                "message": check.get("message") or "",
                "missing_required_metadata": item.get("missing_required_fields") or [],
                "next_step": check.get("next_step") or "",
                "source": payload.get("_mackv_opt_source_path") or payload.get("manifest_path") or "",
            }
        )
    for model_name in evidence.get("missing_models") or []:
        records.append(
            {
                "method": "readiness",
                "artifact_type": "collect-audit",
                "component": "model-availability",
                "status": check.get("status") or "unknown",
                "model": model_name,
                "message": check.get("message") or "",
                "next_step": check.get("next_step") or "",
                "source": payload.get("_mackv_opt_source_path") or payload.get("manifest_path") or "",
            }
        )
    return records


def _readiness_evidence_fields(evidence: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    fields.update(_hardware_fields(evidence))
    fields.update(_memory_fields(evidence))
    if isinstance(evidence.get("memory_state"), dict):
        fields.update(_memory_fields(evidence["memory_state"]))
    if "model_count" in evidence:
        fields["ollama_model_count"] = evidence.get("model_count")
    if isinstance(evidence.get("ollama"), dict):
        fields.update(_ollama_fields(evidence["ollama"]))
    if isinstance(evidence.get("llama_cpp"), dict):
        fields.update(_llama_cpp_fields(evidence["llama_cpp"]))
    if "ollama_available" in evidence:
        fields["ollama_available"] = evidence.get("ollama_available")
    for name in [
        "supports_ollama_num_ctx",
        "supports_llama_cpp_cache_type_k",
        "supports_llama_cpp_cache_type_v",
    ]:
        if name in evidence:
            fields[name] = evidence.get(name)
    if "supports_cache_type_kv" in evidence:
        fields["supports_llama_cpp_cache_type_k"] = evidence.get("supports_cache_type_kv")
        fields["supports_llama_cpp_cache_type_v"] = evidence.get("supports_cache_type_kv")
    return fields


def _compact_readiness_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "method": "readiness",
            "paper_ready": False,
            "status": "missing",
            "artifact_type": "readiness-compact",
            "message": "No readiness rows were available.",
            "next_step": "Run `mackv-opt collect` and `mackv-opt audit`, then render `report --table readiness`.",
        }
    normalized = [_normalize_row(row) for row in rows]
    failed = _unique(
        row.get("check") or row.get("component") or row.get("artifact_type")
        for row in normalized
        if row.get("status") == "fail"
    )
    warning = _unique(
        row.get("check") or row.get("component") or row.get("artifact_type")
        for row in normalized
        if row.get("status") == "warn"
    )
    metadata_rows = [row for row in normalized if row.get("component") == "model-metadata"]
    incomplete_metadata = [
        row.get("model")
        for row in metadata_rows
        if row.get("metadata_status") not in {"pass", ""}
        or bool(row.get("missing_required_metadata"))
    ]
    override_models = _unique(
        row.get("model")
        for row in metadata_rows
        if row.get("override_applied") is True or row.get("override_applied") == "True"
    )
    model_count = _first_value(normalized, "model_count")
    ollama_model_count = _first_value(normalized, "ollama_model_count")
    environment_missing = [
        key
        for key in ["os_version", "power_source", "thermal_state"]
        if not _first_value(normalized, key)
    ]
    is_apple_silicon = _first_value(normalized, "is_apple_silicon")
    ollama_available = _first_value(normalized, "ollama_available")
    paper_ready = (
        not failed
        and is_apple_silicon is True
        and ollama_available is True
        and bool(_int_value(ollama_model_count) or _int_value(model_count))
        and not incomplete_metadata
        and not environment_missing
    )
    next_steps = _unique(row.get("next_step") for row in normalized if row.get("next_step"))
    if environment_missing:
        next_steps.append("Record missing environment fields: " + ", ".join(environment_missing) + ".")
    if incomplete_metadata:
        next_steps.append("Fill missing KV metadata for: " + ", ".join(str(model) for model in incomplete_metadata) + ".")
    return {
        "method": "readiness",
        "paper_ready": paper_ready,
        "status": "pass" if paper_ready else "fail" if failed else "warn",
        "artifact_type": "readiness-compact",
        "component_count": len(normalized),
        "model_count": model_count,
        "platform": _first_value(normalized, "platform"),
        "machine": _first_value(normalized, "machine"),
        "chip": _first_value(normalized, "chip"),
        "is_apple_silicon": is_apple_silicon,
        "os_version": _first_value(normalized, "os_version"),
        "power_source": _first_value(normalized, "power_source"),
        "power_mode": _first_value(normalized, "power_mode"),
        "thermal_state": _first_value(normalized, "thermal_state"),
        "memory_pressure": _first_value(normalized, "memory_pressure"),
        "ollama_available": ollama_available,
        "ollama_model_count": ollama_model_count,
        "llama_cpp_available": _first_value(normalized, "llama_cpp_available"),
        "supports_ollama_num_ctx": _first_value(normalized, "supports_ollama_num_ctx"),
        "supports_llama_cpp_cache_type_k": _first_value(normalized, "supports_llama_cpp_cache_type_k"),
        "supports_llama_cpp_cache_type_v": _first_value(normalized, "supports_llama_cpp_cache_type_v"),
        "metadata_complete": not incomplete_metadata,
        "metadata_override_models": override_models,
        "failed_checks": failed,
        "warning_checks": warning,
        "next_step": " ".join(next_steps),
    }


def _hardware_fields(hardware: dict[str, Any]) -> dict[str, Any]:
    platform = hardware.get("platform")
    machine = hardware.get("machine")
    chip = hardware.get("chip")
    has_hardware = bool(platform or machine or chip or hardware.get("total_memory_bytes") is not None)
    return {
        "platform": platform or "",
        "machine": machine or "",
        "chip": chip or "",
        "is_apple_silicon": _is_apple_silicon(hardware) if has_hardware else None,
        "os_version": hardware.get("os_version") or "",
        "kernel_version": hardware.get("kernel_version") or "",
        "total_memory_bytes": hardware.get("total_memory_bytes"),
        "available_memory_bytes": hardware.get("available_memory_bytes"),
        "power_source": hardware.get("power_source") or "",
        "power_mode": hardware.get("power_mode") or "",
        "thermal_state": hardware.get("thermal_state") or "",
        "memory_pressure": hardware.get("pressure") or hardware.get("memory_pressure") or "",
    }


def _memory_fields(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_pressure": memory.get("memory_pressure") or "",
        "swap_bytes": memory.get("swap_bytes"),
        "pageout_delta": memory.get("pageout_delta"),
        "pageout_bytes_delta": memory.get("pageout_bytes_delta"),
        "swapins_delta": memory.get("swapins_delta"),
        "swapouts_delta": memory.get("swapouts_delta"),
    }


def _capability_fields(caps: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "supports_ollama_num_ctx": caps.get("supports_ollama_num_ctx"),
        "supports_llama_cpp_cache_type_k": caps.get("supports_llama_cpp_cache_type_k"),
        "supports_llama_cpp_cache_type_v": caps.get("supports_llama_cpp_cache_type_v"),
    }
    fields.update(_ollama_fields(_dict_value(caps.get("ollama"))))
    fields.update(_llama_cpp_fields(_dict_value(caps.get("llama_cpp"))))
    return fields


def _ollama_fields(ollama: dict[str, Any]) -> dict[str, Any]:
    return {
        "ollama_available": ollama.get("available"),
        "ollama_version": ollama.get("version") or "",
    }


def _llama_cpp_fields(llama_cpp: dict[str, Any]) -> dict[str, Any]:
    return {
        "llama_cpp_available": llama_cpp.get("available"),
        "llama_cpp_version": llama_cpp.get("version") or "",
    }


def _metadata_message(audit: dict[str, Any]) -> str:
    missing = audit.get("missing_required_fields") or []
    if missing:
        return "Missing KV-budget-critical metadata."
    if audit.get("status") == "pass":
        return "Required KV-budget metadata is present."
    return "Model metadata audit did not pass."


def _runtime_status(caps: dict[str, Any]) -> str:
    ollama = _dict_value(caps.get("ollama"))
    llama_cpp = _dict_value(caps.get("llama_cpp"))
    if not ollama.get("available"):
        return "fail"
    if llama_cpp.get("available") and caps.get("supports_llama_cpp_cache_type_k") and caps.get(
        "supports_llama_cpp_cache_type_v"
    ):
        return "pass"
    return "warn"


def _ollama_status(ollama: dict[str, Any], model_count: int) -> str:
    if not ollama.get("available"):
        return "fail"
    return "pass" if model_count else "warn"


def _is_profile_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("hardware"), dict) and isinstance(payload.get("ollama"), dict)


def _is_capabilities_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("ollama"), dict) and isinstance(payload.get("llama_cpp"), dict) and any(
        key.startswith("supports_") for key in payload
    )


def _is_apple_silicon(hardware: dict[str, Any]) -> bool:
    platform = str(hardware.get("platform") or "")
    machine = str(hardware.get("machine") or "").lower()
    return platform == "Darwin" and ("arm" in machine or "aarch" in machine)


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_report_payload(value: dict[str, Any]) -> bool:
    if value.get("task") in {"collect", "audit", "doctor", "experiment"}:
        return True
    if _is_profile_payload(value) or _is_capabilities_payload(value):
        return True
    if isinstance(value.get("plan"), dict):
        return True
    return any(isinstance(value.get(key), list) for key in ["runs", "jobs", "repeat_summaries"])


def _load_json_with_source(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return {**payload, "_mackv_opt_source_path": str(path)}
    return payload


def _load_referenced_artifact(payload: dict[str, Any], name: str, manifest_dir: Path) -> Any:
    artifacts = _dict_value(payload.get("artifacts"))
    value = artifacts.get(name)
    if not value:
        return None
    path = _resolve_artifact_path(str(value), manifest_dir)
    if not path:
        return None
    try:
        return _load_json_with_source(path)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_artifact_path(value: str, manifest_dir: Path) -> Path | None:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(manifest_dir / path)
        candidates.append(manifest_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _model_name(source: dict[str, Any], model: dict[str, Any], record: dict[str, Any]) -> str:
    value = source.get("model")
    if isinstance(value, str):
        return value
    if model.get("name"):
        return str(model["name"])
    if source.get("model_name"):
        return str(source["model_name"])
    if record.get("name"):
        return str(record["name"])
    return ""


def _stability_context_records(
    model_record: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    contexts = model_record.get("contexts")
    if not isinstance(contexts, dict):
        return []
    records: list[dict[str, Any]] = []
    for context_state in contexts.values():
        if not isinstance(context_state, dict):
            continue
        records.append(
            {
                **context_state,
                "model": model_record.get("model"),
                "method": "stability-context",
                "stable_context_policy": model_record.get("stable_context_policy")
                or summary.get("stable_context_policy"),
                "min_stable_fraction": model_record.get("min_stable_fraction")
                if model_record.get("min_stable_fraction") is not None
                else summary.get("min_stable_fraction"),
                "max_stable_context": model_record.get("max_stable_context"),
                "instability_reasons": context_state.get("reasons"),
            }
        )
    return sorted(records, key=lambda item: int(item.get("context") or 0))


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _format_reasons(value: Any) -> str:
    if not isinstance(value, dict):
        return "" if value is None else str(value)
    items = []
    for reason, count in sorted(value.items(), key=lambda item: str(item[0])):
        items.append(f"{reason}:{count}")
    return "; ".join(items)


def _format_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _unique(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        if value in {None, ""}:
            continue
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return ""


def _int_value(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _value_or_empty(value: Any) -> Any:
    return "" if value is None else value
