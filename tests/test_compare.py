import json

from mackv_opt.compare import (
    build_compare_payload,
    parse_compare_input,
    render_compare_csv,
    render_compare_markdown,
)


def _write_artifact(path, *, model="m", max_context=8192, tps=12.0, peak_memory=1024, accuracy=0.75):
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": model,
                        "context": max_context,
                        "method": "ollama-api",
                        "tokens_per_second": tps,
                        "first_token_latency_ms": 120.0,
                        "peak_memory_bytes": peak_memory,
                        "memory_pressure": "normal",
                        "success_rate": 1.0,
                    }
                ],
                "repeat_summaries": [
                    {
                        "model": model,
                        "context": max_context,
                        "method": "ollama-api",
                        "tokens_per_second_mean": tps,
                        "first_token_latency_ms_mean": 120.0,
                        "peak_memory_bytes_mean": peak_memory,
                        "success_rate": 1.0,
                    }
                ],
                "stability_summary": {
                    "stable_context_policy": "all",
                    "models": [
                        {
                            "model": model,
                            "max_stable_context": max_context,
                            "stable_runs": 3,
                            "unstable_runs": 0,
                            "stable_context_policy": "all",
                        }
                    ],
                },
                "qa": {
                    "repeat_summaries": [
                        {
                            "model": model,
                            "context": max_context,
                            "method": "qa",
                            "accuracy": accuracy,
                            "quality_score_mean": accuracy,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_parse_compare_input_accepts_label_prefix():
    parsed = parse_compare_input("mackv=run.json")

    assert parsed.label == "mackv"
    assert parsed.path == "run.json"


def test_build_compare_payload_summarizes_artifacts(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    _write_artifact(baseline, max_context=8192, tps=10.0)
    _write_artifact(optimized, max_context=16384, tps=12.0)

    payload = build_compare_payload([f"baseline={baseline}", f"mackv-opt={optimized}"])

    assert payload["task"] == "compare"
    assert payload["baseline_label"] == "baseline"
    assert payload["rows"][0]["label"] == "baseline"
    assert payload["rows"][0]["max_stable_context"] == 8192
    assert payload["rows"][0]["max_stable_context_vs_baseline"] == 1.0
    assert payload["rows"][1]["label"] == "mackv-opt"
    assert payload["rows"][1]["max_stable_context"] == 16384
    assert payload["rows"][1]["max_stable_context_vs_baseline"] == 2.0
    assert payload["rows"][1]["best_context_vs_baseline"] == 2.0
    assert payload["rows"][1]["tokens_per_second_mean"] == 12.0
    assert payload["rows"][1]["tokens_per_second_vs_baseline"] == 1.2
    assert payload["rows"][1]["quality_accuracy"] == 0.75
    assert payload["rows"][1]["quality_accuracy_vs_baseline"] == 0.0
    assert payload["rows"][1]["stable_context_policy"] == "all"


def test_build_compare_payload_can_select_baseline_label(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    _write_artifact(baseline, max_context=8192, tps=10.0, peak_memory=1000, accuracy=0.8)
    _write_artifact(optimized, max_context=16384, tps=12.0, peak_memory=1200, accuracy=0.7)

    payload = build_compare_payload(
        [f"baseline={baseline}", f"mackv-opt={optimized}"],
        baseline_label="mackv-opt",
    )

    assert payload["baseline_label"] == "mackv-opt"
    assert payload["rows"][0]["baseline_label"] == "mackv-opt"
    assert payload["rows"][0]["max_stable_context_vs_baseline"] == 0.5
    assert payload["rows"][0]["tokens_per_second_vs_baseline"] == 0.833333
    assert payload["rows"][0]["peak_memory_bytes_vs_baseline"] == 0.833333
    assert payload["rows"][0]["quality_accuracy_vs_baseline"] == 0.1


def test_render_compare_markdown_and_csv(tmp_path):
    artifact = tmp_path / "run.json"
    _write_artifact(artifact)
    payload = build_compare_payload([f"run={artifact}"])

    markdown = render_compare_markdown(payload["rows"])
    csv_text = render_compare_csv(payload["rows"])

    assert "| label | baseline_label | model | max_stable_context |" in markdown
    assert "| run | run | m | 8192 | 1.0 |" in markdown
    assert "label,baseline_label,model,max_stable_context" in csv_text
    assert "run,run,m,8192,1.0" in csv_text


def test_build_compare_payload_handles_multiple_stability_rows(tmp_path):
    artifact = tmp_path / "multi.json"
    artifact.write_text(
        json.dumps(
            {
                "stability_summary": {
                    "models": [
                        {"model": "a", "max_stable_context": 8192},
                        {"model": "b", "max_stable_context": 32768},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_compare_payload([f"multi={artifact}"])

    assert payload["rows"][0]["max_stable_context"] == 32768
