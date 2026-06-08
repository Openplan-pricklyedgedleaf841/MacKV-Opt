from pathlib import Path
import importlib.util
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_run_macos_matrix_script_documents_safe_defaults_and_commands():
    script = ROOT / "scripts" / "run_macos_matrix.sh"

    text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert "MACKV_EXECUTE" in text
    assert "Default mode is dry-run" in text
    assert 'RUN_MODE="--dry-run"' in text
    assert "mackv-opt experiment" in text
    assert "mackv-opt report" in text
    assert "mackv-opt collect" in text
    assert "$BASE_DIR/collect" in text
    assert "collect.stdout.json" in text
    assert "$BASE_DIR/collect/doctor.json" in text
    assert "$BASE_DIR/collect/machine-profile.json" in text
    assert "$BASE_DIR/collect/runtime-capabilities.json" in text
    assert "doctor.json" in text
    assert "MACKV_MODEL_METADATA_OVERRIDES" in text
    assert "--model-metadata-overrides" in text
    assert "--memory-sample-interval" in text
    assert "--include-memory-series" in text
    assert "MACKV_REPEATS" in text
    assert "--repeats" in text
    assert "MACKV_QA_NUM_PREDICT" in text
    assert "MACKV_MAX_SWAP" in text
    assert "--max-swap" in text
    assert "MACKV_MIN_TOKENS_PER_SEC" in text
    assert "--min-tokens-per-second" in text
    assert "MACKV_STABLE_POLICY" in text
    assert "--stable-context-policy" in text
    assert "MACKV_MIN_STABLE_FRACTION" in text
    assert "--min-stable-fraction" in text
    assert "paper-tables" in text
    assert "MACKV_COMPARE_LABELS" in text
    assert "default,manual-num-ctx,mackv-opt" in text
    assert "MACKV_COMPARE_BASELINE" in text
    assert "MACKV_COMPARE_CURRENT" in text
    assert "MACKV_WRITE_BASELINE_TEMPLATES" in text
    assert "mackv-opt baseline-template" in text
    assert "manual-num-ctx" in text
    assert "MACKV_FAIL_ON_MISSING_METADATA" in text
    assert "MACKV_REQUIRE_APPLE_SILICON" in text
    assert "mackv-opt audit" in text
    assert "--require-apple-silicon" in text
    assert "collect-audit.json" in text
    assert "collect-audit.md" in text
    assert "collect-audit.stdout.json" in text
    assert "Refusing executable runs" in text
    assert "--table readiness-compact" in text
    assert "--table readiness" in text
    assert "$BASE_DIR/paper-tables" in text
    assert "mackv-opt compare" in text
    assert "matrix-compare.md" in text
    assert "matrix-compare.csv" in text
    assert "-compare.md" in text
    assert "-compare.csv" in text


def load_metadata_sync_module():
    script = ROOT / "scripts" / "sync_github_metadata.py"
    spec = importlib.util.spec_from_file_location("sync_github_metadata", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_github_metadata_sync_payload_matches_repository_profile():
    module = load_metadata_sync_module()

    metadata = module.build_metadata()
    payload = module.dry_run_payload(metadata)

    assert payload["repo"] == "Lin-Aurora/MacKV-Opt"
    assert payload["description"] == (
        "KV cache and context planner for running longer local LLM contexts on "
        "Apple Silicon Macs with Ollama-compatible benchmarks."
    )
    assert payload["homepage"] == "https://github.com/Lin-Aurora/MacKV-Opt#readme"
    assert payload["topics_payload"]["names"] == [
        "apple-silicon",
        "ollama",
        "llama-cpp",
        "kv-cache",
        "local-llm",
        "llm-inference",
        "macos",
        "benchmark",
        "long-context",
        "mlx",
        "gguf",
        "research-artifact",
    ]


def test_github_metadata_sync_script_dry_run_outputs_json():
    script = ROOT / "scripts" / "sync_github_metadata.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["repo"] == "Lin-Aurora/MacKV-Opt"
    assert payload["repo_patch_payload"]["homepage"] == (
        "https://github.com/Lin-Aurora/MacKV-Opt#readme"
    )
    assert "research-artifact" in payload["topics"]
