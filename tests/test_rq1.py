import json

from mackv_opt.cli import main
from mackv_opt.rq1 import build_rq1_summary_payload, render_rq1_markdown


def _write_artifact(path, *, model, label, max_context, tps=10.0, memory=1024, accuracy=0.8):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": model,
                        "context": max_context,
                        "method": "ollama-api",
                        "tokens_per_second": tps,
                        "peak_memory_bytes": memory,
                        "memory_pressure": "normal",
                    }
                ],
                "repeat_summaries": [
                    {
                        "model": model,
                        "context": max_context,
                        "method": "ollama-api",
                        "tokens_per_second_mean": tps,
                        "peak_memory_bytes_mean": memory,
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
                "label": label,
            }
        ),
        encoding="utf-8",
    )


def test_build_rq1_summary_payload_scans_three_baselines(tmp_path):
    model_dir = tmp_path / "llama3.1-8b"
    _write_artifact(model_dir / "default" / "full-run.json", model="llama3.1:8b", label="default", max_context=8192)
    _write_artifact(
        model_dir / "manual-num-ctx" / "full-run.json",
        model="llama3.1:8b",
        label="manual-num-ctx",
        max_context=16384,
    )
    _write_artifact(
        model_dir / "mackv-opt" / "full-run.json",
        model="llama3.1:8b",
        label="mackv-opt",
        max_context=32768,
    )

    payload = build_rq1_summary_payload(str(tmp_path))

    assert payload["task"] == "rq1-summary"
    assert payload["machine_dir"] == str(tmp_path)
    assert payload["summary"]["model_count"] == 1
    assert payload["summary"]["complete_model_count"] == 1
    row = payload["rows"][0]
    assert row["model"] == "llama3.1:8b"
    assert row["default_max_stable_context"] == 8192
    assert row["manual_num_ctx_max_stable_context"] == 16384
    assert row["mackv_opt_max_stable_context"] == 32768
    assert row["mackv_opt_vs_default"] == 4.0
    assert row["mackv_opt_vs_manual_num_ctx"] == 2.0
    assert row["best_label"] == "mackv-opt"


def test_render_rq1_markdown_outputs_paper_table(tmp_path):
    model_dir = tmp_path / "m"
    _write_artifact(model_dir / "default" / "full-run.json", model="m", label="default", max_context=8192)
    _write_artifact(model_dir / "manual-num-ctx" / "full-run.json", model="m", label="manual", max_context=8192)
    _write_artifact(model_dir / "mackv-opt" / "full-run.json", model="m", label="mackv-opt", max_context=16384)

    payload = build_rq1_summary_payload(str(tmp_path))
    table = render_rq1_markdown(payload)

    assert "| model | default_max_stable_context |" in table
    assert "| m | 8192 | 8192 | 16384 | 2.0 | 2.0 | mackv-opt |" in table


def test_cli_rq1_summary_writes_output(tmp_path, capsys):
    model_dir = tmp_path / "m"
    output = tmp_path / "rq1.md"
    _write_artifact(model_dir / "default" / "full-run.json", model="m", label="default", max_context=8192)
    _write_artifact(model_dir / "manual-num-ctx" / "full-run.json", model="m", label="manual", max_context=8192)
    _write_artifact(model_dir / "mackv-opt" / "full-run.json", model="m", label="mackv-opt", max_context=16384)

    code = main(["rq1-summary", str(tmp_path), "--output", str(output), "--format", "markdown"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert output.exists()
    assert "mackv_opt_vs_default" in output.read_text(encoding="utf-8")
