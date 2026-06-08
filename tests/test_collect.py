from __future__ import annotations

import json

from mackv_opt.collect import (
    audit_collect_manifest,
    audit_model_profile,
    collect_artifacts,
    load_model_metadata_overrides,
    render_collect_audit_text,
    render_collect_markdown,
)


def _raw_show_payload() -> dict[str, object]:
    return {
        "details": {"family": "llama", "parameter_size": "8B"},
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
        "size": 5_100_000_000,
    }


def test_audit_model_profile_passes_with_required_metadata():
    audit = audit_model_profile(
        {
            "family": "llama",
            "architecture": "llama",
            "parameter_count": 8_000_000_000,
            "size_bytes": 5_100_000_000,
            "hidden_size": 4096,
            "layer_count": 32,
            "attention_head_count": 32,
            "kv_head_count": 8,
            "max_context": 131072,
        }
    )

    assert audit["status"] == "pass"
    assert audit["missing_required_fields"] == []


def test_audit_model_profile_warns_when_kv_budget_metadata_is_missing():
    audit = audit_model_profile({"name": "llama3.1:8b", "size_bytes": None, "hidden_size": 4096})

    assert audit["status"] == "warn"
    assert "layer_count" in audit["missing_required_fields"]
    assert audit["warnings"]


def test_collect_artifacts_writes_manifest_and_model_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("mackv_opt.collect.doctor_payload", lambda: {"task": "doctor", "status": "pass"})
    monkeypatch.setattr("mackv_opt.collect.profile_payload", lambda: {"hardware": {"platform": "Darwin"}})
    monkeypatch.setattr(
        "mackv_opt.collect.detect_runtime_capabilities",
        lambda: type(
            "FakeCapabilities",
            (),
            {"to_dict": lambda self: {"ollama": {"available": True}, "llama_cpp": {"available": True}}},
        )(),
    )
    monkeypatch.setattr("mackv_opt.collect.load_ollama_show_payload", lambda model: _raw_show_payload())

    manifest = collect_artifacts(str(tmp_path), models=["llama3.1:8b"])

    assert manifest["task"] == "collect"
    assert manifest["doctor_status"] == "pass"
    assert manifest["model_count"] == 1
    assert manifest["models"][0]["metadata_audit"]["status"] == "pass"
    assert (tmp_path / "doctor.json").exists()
    assert (tmp_path / "machine-profile.json").exists()
    assert (tmp_path / "runtime-capabilities.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest.md").exists()
    assert (tmp_path / "models" / "llama3.1-8b-ollama-show.json").exists()
    assert (tmp_path / "models" / "llama3.1-8b-profile.json").exists()

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["artifacts"]["doctor"] == str(tmp_path / "doctor.json")


def test_collect_artifacts_records_missing_model_without_raw_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("mackv_opt.collect.doctor_payload", lambda: {"task": "doctor", "status": "fail"})
    monkeypatch.setattr("mackv_opt.collect.profile_payload", lambda: {"hardware": {}})
    monkeypatch.setattr(
        "mackv_opt.collect.detect_runtime_capabilities",
        lambda: type("FakeCapabilities", (), {"to_dict": lambda self: {}})(),
    )
    monkeypatch.setattr("mackv_opt.collect.load_ollama_show_payload", lambda model: None)

    manifest = collect_artifacts(str(tmp_path), models=["missing:model"], include_raw_model_json=False)

    model = manifest["models"][0]
    assert model["status"] == "missing"
    assert model["raw_ollama_show_json"] is None
    assert model["metadata_audit"]["status"] == "missing"
    assert not (tmp_path / "models" / "missing-model-ollama-show.json").exists()


def test_load_model_metadata_overrides_accepts_mapping_and_aliases(tmp_path):
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "models": {
                    "llama3.1:8b": {
                        "model_size": 5_100_000_000,
                        "layers": 32,
                        "heads": 32,
                        "kv_heads": 8,
                        "hidden_size": 4096,
                        "context_length": 131072,
                        "ignored": "nope",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_model_metadata_overrides(str(overrides))

    assert loaded["llama3.1:8b"]["size_bytes"] == 5_100_000_000
    assert loaded["llama3.1:8b"]["layer_count"] == 32
    assert loaded["llama3.1:8b"]["attention_head_count"] == 32
    assert loaded["llama3.1:8b"]["kv_head_count"] == 8
    assert loaded["llama3.1:8b"]["max_context"] == 131072
    assert "ignored" not in loaded["llama3.1:8b"]


def test_collect_artifacts_uses_override_when_ollama_metadata_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("mackv_opt.collect.doctor_payload", lambda: {"task": "doctor", "status": "pass"})
    monkeypatch.setattr("mackv_opt.collect.profile_payload", lambda: {"hardware": {"platform": "Darwin"}})
    monkeypatch.setattr(
        "mackv_opt.collect.detect_runtime_capabilities",
        lambda: type("FakeCapabilities", (), {"to_dict": lambda self: {}})(),
    )
    monkeypatch.setattr("mackv_opt.collect.load_ollama_show_payload", lambda model: None)

    manifest = collect_artifacts(
        str(tmp_path),
        models=["llama3.1:8b"],
        model_metadata_overrides={
            "llama3.1:8b": {
                "size_bytes": 5_100_000_000,
                "parameter_count": 8_000_000_000,
                "hidden_size": 4096,
                "layer_count": 32,
                "attention_head_count": 32,
                "kv_head_count": 8,
                "max_context": 131072,
                "family": "llama",
                "architecture": "llama",
            }
        },
        model_metadata_override_path="overrides.json",
    )

    model = manifest["models"][0]
    assert model["status"] == "ok-with-override"
    assert model["metadata_audit"]["status"] == "pass"
    assert model["metadata_override"]["applied"] is True
    assert model["metadata_override"]["source"] == "overrides.json"
    assert model["profile"]["hidden_size"] == 4096
    assert model["profile"]["metadata"]["mackv_opt_override_applied"] is True


def test_render_collect_markdown_summarizes_missing_metadata():
    rendered = render_collect_markdown(
        {
            "created_at": "2026-06-07T00:00:00Z",
            "doctor_status": "warn",
            "model_count": 1,
            "artifacts": {"doctor": "doctor.json"},
            "models": [
                {
                    "name": "llama3.1:8b",
                    "status": "ok",
                    "metadata_audit": {
                        "status": "warn",
                        "missing_required_fields": ["kv_head_count"],
                    },
                }
            ],
        }
    )

    assert "Doctor status: warn" in rendered
    assert "llama3.1:8b" in rendered
    assert "missing: kv_head_count" in rendered


def test_audit_collect_manifest_passes_for_complete_bundle(tmp_path):
    for name in ["doctor.json", "machine-profile.json", "runtime-capabilities.json", "manifest.json", "manifest.md"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "machine-profile.json").write_text(
        json.dumps({"hardware": {"platform": "Darwin", "machine": "arm64", "chip": "Apple M3 Pro"}}),
        encoding="utf-8",
    )
    manifest = {
        "task": "collect",
        "doctor_status": "pass",
        "model_count": 1,
        "artifacts": {
            "doctor": str(tmp_path / "doctor.json"),
            "profile": str(tmp_path / "machine-profile.json"),
            "capabilities": str(tmp_path / "runtime-capabilities.json"),
            "manifest": str(tmp_path / "manifest.json"),
            "markdown": str(tmp_path / "manifest.md"),
        },
        "models": [
            {
                "name": "llama3.1:8b",
                "status": "ok",
                "metadata_audit": {"status": "pass", "missing_required_fields": []},
            }
        ],
    }

    audit = audit_collect_manifest(
        manifest,
        manifest_path=str(tmp_path / "manifest.json"),
        require_apple_silicon=True,
    )

    assert audit["status"] == "pass"
    assert audit["summary"]["failed_checks"] == []


def test_audit_collect_manifest_fails_when_doctor_failed():
    audit = audit_collect_manifest(
        {
            "task": "collect",
            "doctor_status": "fail",
            "artifacts": {},
            "models": [],
        },
        require_artifacts=False,
    )

    checks = {check["name"]: check for check in audit["checks"]}
    assert audit["status"] == "fail"
    assert checks["doctor"]["status"] == "fail"


def test_audit_collect_manifest_fails_when_apple_silicon_is_required(tmp_path):
    profile = tmp_path / "machine-profile.json"
    profile.write_text(
        json.dumps({"hardware": {"platform": "Windows", "machine": "AMD64", "chip": "x86"}}),
        encoding="utf-8",
    )
    manifest = {
        "task": "collect",
        "doctor_status": "pass",
        "artifacts": {"profile": str(profile)},
        "models": [
            {
                "name": "llama3.1:8b",
                "status": "ok",
                "metadata_audit": {"missing_required_fields": []},
            }
        ],
    }

    audit = audit_collect_manifest(manifest, require_artifacts=False, require_apple_silicon=True)

    checks = {check["name"]: check for check in audit["checks"]}
    assert audit["status"] == "fail"
    assert checks["hardware"]["status"] == "fail"
    assert checks["hardware"]["evidence"]["is_apple_silicon"] is False


def test_audit_collect_manifest_warns_on_non_apple_silicon_without_requirement(tmp_path):
    profile = tmp_path / "machine-profile.json"
    profile.write_text(
        json.dumps({"hardware": {"platform": "Linux", "machine": "x86_64", "chip": "x86"}}),
        encoding="utf-8",
    )
    manifest = {
        "task": "collect",
        "doctor_status": "pass",
        "artifacts": {"profile": str(profile)},
        "models": [
            {
                "name": "llama3.1:8b",
                "status": "ok",
                "metadata_audit": {"missing_required_fields": []},
            }
        ],
    }

    audit = audit_collect_manifest(manifest, require_artifacts=False, require_apple_silicon=False)

    checks = {check["name"]: check for check in audit["checks"]}
    assert audit["status"] == "warn"
    assert checks["hardware"]["status"] == "warn"


def test_audit_collect_manifest_can_fail_on_missing_metadata():
    manifest = {
        "task": "collect",
        "doctor_status": "pass",
        "artifacts": {},
        "models": [
            {
                "name": "qwen2.5:7b",
                "status": "ok",
                "metadata_audit": {"status": "warn", "missing_required_fields": ["kv_head_count"]},
            }
        ],
    }

    loose = audit_collect_manifest(manifest, fail_on_missing_metadata=False, require_artifacts=False)
    strict = audit_collect_manifest(manifest, fail_on_missing_metadata=True, require_artifacts=False)

    assert loose["status"] == "warn"
    assert strict["status"] == "fail"
    assert "model-metadata" in strict["summary"]["failed_checks"]


def test_render_collect_audit_text_includes_failed_checks():
    rendered = render_collect_audit_text(
        {
            "status": "fail",
            "checks": [
                {
                    "name": "doctor",
                    "status": "fail",
                    "message": "Doctor preflight did not pass.",
                    "next_step": "Run doctor.",
                }
            ],
        }
    )

    assert rendered.startswith("MacKV-Opt audit: fail")
    assert "- [fail] doctor:" in rendered
    assert "next: Run doctor." in rendered
