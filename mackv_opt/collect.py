from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .capabilities import detect_runtime_capabilities
from .doctor import doctor_payload
from .ollama import load_ollama_show_payload, normalize_show_payload
from .profiler import ollama_models, profile_payload


def collect_artifacts(
    output_dir: str,
    *,
    models: Iterable[str] | None = None,
    include_raw_model_json: bool = True,
    model_metadata_overrides: dict[str, dict[str, Any]] | None = None,
    model_metadata_override_path: str | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_names = _resolve_model_names(models)
    overrides = model_metadata_overrides or {}

    doctor = doctor_payload()
    profile = profile_payload()
    capabilities = detect_runtime_capabilities().to_dict()
    model_records = [
        _collect_model(
            name,
            target / "models",
            include_raw_model_json,
            override=overrides.get(name),
            override_path=model_metadata_override_path,
        )
        for name in model_names
    ]

    markdown_path = str(target / "manifest.md")
    manifest_path = str(target / "manifest.json")
    manifest = {
        "task": "collect",
        "created_at": _utc_now(),
        "output_dir": str(target),
        "models_requested": model_names,
        "model_metadata_override_path": model_metadata_override_path,
        "doctor_status": doctor.get("status"),
        "model_count": len(model_records),
        "artifacts": {
            "doctor": _write_json(target / "doctor.json", doctor),
            "profile": _write_json(target / "machine-profile.json", profile),
            "capabilities": _write_json(target / "runtime-capabilities.json", capabilities),
            "manifest": manifest_path,
            "markdown": markdown_path,
        },
        "models": model_records,
    }
    _write_json(target / "manifest.json", manifest)
    _write_text(target / "manifest.md", render_collect_markdown(manifest))
    return manifest


def render_collect_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# MacKV-Opt Collection Manifest",
        "",
        f"- Created: {manifest.get('created_at', '')}",
        f"- Doctor status: {manifest.get('doctor_status', '')}",
        f"- Model count: {manifest.get('model_count', 0)}",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for name in sorted(artifacts):
        lines.append(f"- {name}: `{artifacts[name]}`")

    lines.extend(["", "## Models", ""])
    models = manifest.get("models") if isinstance(manifest.get("models"), list) else []
    if not models:
        lines.append("- No models were collected.")
    for record in models:
        if not isinstance(record, dict):
            continue
        audit = record.get("metadata_audit") if isinstance(record.get("metadata_audit"), dict) else {}
        missing = audit.get("missing_required_fields") or []
        status = audit.get("status", "unknown")
        lines.append(f"- {record.get('name', '')}: {record.get('status', '')}, metadata={status}")
        if missing:
            lines.append(f"  - missing: {', '.join(str(item) for item in missing)}")
    return "\n".join(lines) + "\n"


def load_collect_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"task": "unknown", "raw": payload}


def load_model_metadata_overrides(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    source = payload.get("models", payload)
    if isinstance(source, dict):
        return {
            str(name): _clean_model_override(value)
            for name, value in source.items()
            if isinstance(value, dict)
        }
    if isinstance(source, list):
        overrides: dict[str, dict[str, Any]] = {}
        for item in source:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            overrides[str(item["name"])] = _clean_model_override(item)
        return overrides
    return {}


def audit_collect_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: str | None = None,
    fail_on_missing_metadata: bool = False,
    require_artifacts: bool = True,
    require_apple_silicon: bool = False,
) -> dict[str, Any]:
    manifest_dir = Path(manifest_path).parent if manifest_path else Path.cwd()
    checks = [
        _check_manifest_shape(manifest),
        _check_manifest_artifacts(manifest, manifest_dir, require_artifacts=require_artifacts),
        _check_hardware_profile(manifest, manifest_dir, require_apple_silicon=require_apple_silicon),
        _check_doctor_status(manifest),
        _check_model_metadata(manifest, fail_on_missing_metadata=fail_on_missing_metadata),
    ]
    return {
        "task": "audit",
        "status": _overall_status(checks),
        "manifest_path": manifest_path,
        "policy": {
            "fail_on_missing_metadata": fail_on_missing_metadata,
            "require_artifacts": require_artifacts,
            "require_apple_silicon": require_apple_silicon,
        },
        "summary": _audit_summary(manifest, checks),
        "checks": checks,
    }


def render_collect_audit_text(payload: dict[str, Any]) -> str:
    lines = [
        f"MacKV-Opt audit: {payload.get('status', 'unknown')}",
        "",
        "Checks:",
    ]
    for check in payload.get("checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(f"- [{check.get('status')}] {check.get('name')}: {check.get('message')}")
        next_step = check.get("next_step")
        if next_step:
            lines.append(f"  next: {next_step}")
    return "\n".join(lines)


def audit_model_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {
            "status": "missing",
            "missing_required_fields": [
                "size_bytes",
                "hidden_size",
                "layer_count",
                "attention_head_count",
                "kv_head_count",
            ],
            "missing_recommended_fields": ["parameter_count", "max_context", "architecture", "family"],
            "warnings": ["Model metadata could not be normalized from Ollama."],
        }

    required = [
        "size_bytes",
        "hidden_size",
        "layer_count",
        "attention_head_count",
        "kv_head_count",
    ]
    recommended = ["parameter_count", "max_context", "architecture", "family"]
    missing_required = [field for field in required if profile.get(field) is None]
    missing_recommended = [field for field in recommended if profile.get(field) in {None, "", "unknown"}]
    warnings: list[str] = []
    if missing_required:
        warnings.append(
            "Missing required planner metadata; KV budget estimates will rely on conservative inference."
        )
    if missing_recommended:
        warnings.append("Missing recommended metadata for repeatable validation.")
    return {
        "status": "pass" if not missing_required else "warn",
        "missing_required_fields": missing_required,
        "missing_recommended_fields": missing_recommended,
        "warnings": warnings,
    }


def _collect_model(
    model_name: str,
    model_dir: Path,
    include_raw_model_json: bool,
    *,
    override: dict[str, Any] | None = None,
    override_path: str | None = None,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(model_name)
    raw_payload = load_ollama_show_payload(model_name)
    raw_path: str | None = None
    normalized: dict[str, Any] | None = None
    status = "missing"
    error: str | None = None

    if raw_payload is not None:
        status = "ok"
        if include_raw_model_json:
            raw_path = _write_json(model_dir / f"{safe_name}-ollama-show.json", raw_payload)
        try:
            normalized = normalize_show_payload(model_name, raw_payload).to_dict()
        except Exception as exc:  # pragma: no cover - defensive around external JSON shape
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

    override_fields = _clean_model_override(override)
    if override_fields:
        normalized = _apply_model_override(model_name, normalized, override_fields)
        status = "ok-with-override" if status != "ok" else "ok"

    normalized_path = _write_json(model_dir / f"{safe_name}-profile.json", normalized or {})
    return {
        "name": model_name,
        "status": status,
        "raw_ollama_show_json": raw_path,
        "normalized_profile_json": normalized_path,
        "profile": normalized,
        "metadata_audit": audit_model_profile(normalized),
        "metadata_override": {
            "applied": bool(override_fields),
            "fields": sorted(override_fields),
            "source": override_path,
        },
        "error": error,
    }


def _resolve_model_names(models: Iterable[str] | None) -> list[str]:
    if models is not None:
        return [model.strip() for model in models if model and model.strip()]
    return [str(model["name"]) for model in ollama_models() if model.get("name")]


def _check_manifest_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("task") == "collect":
        return _audit_check(
            "manifest",
            "pass",
            "Collection manifest shape is valid.",
            {"task": manifest.get("task"), "created_at": manifest.get("created_at")},
        )
    return _audit_check(
        "manifest",
        "fail",
        "Input is not a MacKV-Opt collection manifest.",
        {"task": manifest.get("task")},
        "Run `mackv-opt collect --output-dir ...` before executable validation.",
    )


def _check_manifest_artifacts(
    manifest: dict[str, Any],
    manifest_dir: Path,
    *,
    require_artifacts: bool,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    expected = ["doctor", "profile", "capabilities", "manifest", "markdown"]
    missing = [
        name
        for name in expected
        if not artifacts.get(name) or not _artifact_exists(str(artifacts.get(name)), manifest_dir)
    ]
    if not missing:
        return _audit_check(
            "artifacts",
            "pass",
            "Required collection artifacts are present.",
            {"checked": expected},
        )
    status = "fail" if require_artifacts else "warn"
    return _audit_check(
        "artifacts",
        status,
        "Some required collection artifacts are missing.",
        {"missing": missing},
        "Regenerate the collection bundle or keep all files next to the manifest.",
    )


def _check_doctor_status(manifest: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("doctor_status") or "unknown")
    if status == "pass":
        return _audit_check("doctor", "pass", "Doctor preflight passed.", {"doctor_status": status})
    if status == "warn":
        return _audit_check(
            "doctor",
            "warn",
            "Doctor preflight has warnings.",
            {"doctor_status": status},
            "Resolve or explicitly note doctor warnings before benchmark runs.",
        )
    return _audit_check(
        "doctor",
        "fail",
        "Doctor preflight did not pass.",
        {"doctor_status": status},
        "Run `mackv-opt doctor` and resolve failed checks before executable experiments.",
    )


def _check_hardware_profile(
    manifest: dict[str, Any],
    manifest_dir: Path,
    *,
    require_apple_silicon: bool,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    profile_path = artifacts.get("profile")
    profile = _load_json_artifact(str(profile_path), manifest_dir) if profile_path else None
    hardware = profile.get("hardware") if isinstance(profile, dict) and isinstance(profile.get("hardware"), dict) else {}
    platform = str(hardware.get("platform") or "unknown")
    machine = str(hardware.get("machine") or "unknown")
    chip = str(hardware.get("chip") or "unknown")
    is_apple_silicon = platform == "Darwin" and ("arm" in machine.lower() or "aarch" in machine.lower())
    evidence = {
        "platform": platform,
        "machine": machine,
        "chip": chip,
        "is_apple_silicon": is_apple_silicon,
        "profile_artifact": profile_path,
    }
    if is_apple_silicon:
        return _audit_check("hardware", "pass", "Apple Silicon hardware profile was verified.", evidence)
    if require_apple_silicon:
        return _audit_check(
            "hardware",
            "fail",
            "Apple Silicon hardware is required but was not verified.",
            evidence,
            "Run collection on the target Apple Silicon Mac before executable validation.",
        )
    return _audit_check(
        "hardware",
        "warn",
        "Apple Silicon hardware was not verified.",
        evidence,
        "Use `--require-apple-silicon` when validating Apple Silicon-specific results.",
    )


def _check_model_metadata(manifest: dict[str, Any], *, fail_on_missing_metadata: bool) -> dict[str, Any]:
    models = manifest.get("models") if isinstance(manifest.get("models"), list) else []
    if not models:
        return _audit_check(
            "model-metadata",
            "warn",
            "No model metadata records were collected.",
            {"model_count": 0},
            "Pass `--models` or make sure Ollama lists local models.",
        )

    missing_models: list[str] = []
    incomplete_models: list[dict[str, Any]] = []
    for record in models:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "")
        if record.get("status") not in {"ok", "ok-with-override"}:
            missing_models.append(name)
            continue
        metadata_audit = record.get("metadata_audit") if isinstance(record.get("metadata_audit"), dict) else {}
        missing_fields = metadata_audit.get("missing_required_fields") or []
        if missing_fields:
            incomplete_models.append({"name": name, "missing_required_fields": list(missing_fields)})

    if missing_models:
        return _audit_check(
            "model-metadata",
            "fail",
            "One or more requested models could not be collected.",
            {"missing_models": missing_models},
            "Pull the models with Ollama or remove unavailable models from the experiment matrix.",
        )
    if incomplete_models:
        status = "fail" if fail_on_missing_metadata else "warn"
        return _audit_check(
            "model-metadata",
            status,
            "Some model profiles are missing KV-budget-critical metadata.",
            {"incomplete_models": incomplete_models},
            "Fill missing metadata manually or treat planner fallback estimates as approximate.",
        )
    return _audit_check(
        "model-metadata",
        "pass",
        "Collected models have the required KV budget metadata.",
        {"model_count": len(models)},
    )


MODEL_OVERRIDE_FIELDS = {
    "family",
    "parameter_count",
    "size_bytes",
    "hidden_size",
    "layer_count",
    "attention_head_count",
    "kv_head_count",
    "architecture",
    "max_context",
}


MODEL_OVERRIDE_ALIASES = {
    "parameters": "parameter_count",
    "parameter_size": "parameter_count",
    "model_size": "size_bytes",
    "size": "size_bytes",
    "layers": "layer_count",
    "heads": "attention_head_count",
    "kv_heads": "kv_head_count",
    "context_length": "max_context",
    "max_ctx": "max_context",
}


def _clean_model_override(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, raw in value.items():
        canonical = MODEL_OVERRIDE_ALIASES.get(str(key), str(key))
        if canonical not in MODEL_OVERRIDE_FIELDS:
            continue
        cleaned[canonical] = _coerce_model_override_value(canonical, raw)
    return cleaned


def _apply_model_override(
    model_name: str,
    normalized: dict[str, Any] | None,
    override: dict[str, Any],
) -> dict[str, Any]:
    profile = dict(normalized or {})
    profile.setdefault("name", model_name)
    profile.update(override)
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    profile["metadata"] = {
        **metadata,
        "mackv_opt_override_applied": True,
        "mackv_opt_override_fields": sorted(override),
    }
    return profile


def _coerce_model_override_value(field: str, value: Any) -> Any:
    if field in {"family", "architecture"}:
        return str(value) if value is not None else "unknown"
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def _audit_summary(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_count": manifest.get("model_count", 0),
        "doctor_status": manifest.get("doctor_status"),
        "failed_checks": [check["name"] for check in checks if check.get("status") == "fail"],
        "warning_checks": [check["name"] for check in checks if check.get("status") == "warn"],
    }


def _audit_check(
    name: str,
    status: str,
    message: str,
    evidence: dict[str, Any],
    next_step: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "evidence": evidence,
        "next_step": next_step,
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check.get("status") == "fail" for check in checks):
        return "fail"
    if any(check.get("status") == "warn" for check in checks):
        return "warn"
    return "pass"


def _artifact_exists(value: str, manifest_dir: Path) -> bool:
    return _resolve_artifact_path(value, manifest_dir) is not None


def _load_json_artifact(value: str, manifest_dir: Path) -> Any:
    path = _resolve_artifact_path(value, manifest_dir)
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
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


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    return cleaned or "model"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
