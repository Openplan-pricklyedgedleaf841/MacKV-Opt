import json

from mackv_opt.cli import main
from mackv_opt.models import HardwareProfile, ModelProfile, OptimizationPlan
from mackv_opt.report import (
    load_report_payload,
    normalize_report_rows,
    render_paper_table_csv,
    render_paper_table_markdown,
    render_plan_text,
    render_report_csv,
    render_report_markdown,
    write_experiment_artifacts,
    write_paper_tables,
)


def test_report_normalizes_plan_payload():
    rows = normalize_report_rows(
        {
            "model": {"name": "llama3.1:8b"},
            "plan": {
                "status": "fits",
                "num_ctx": 65536,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        }
    )

    assert rows[0]["model"] == "llama3.1:8b"
    assert rows[0]["context"] == 65536
    assert rows[0]["method"] == "mackv-opt"
    assert rows[0]["status"] == "fits"
    assert rows[0]["cache_type_k"] == "q8_0"
    assert rows[0]["cache_type_v"] == "q8_0"
    assert rows[0]["depth"] == ""
    assert rows[0]["found"] == ""
    assert rows[0]["response_excerpt"] == ""


def test_render_plan_text_includes_runtime_advice():
    plan = OptimizationPlan(
        model_name="llama3.1:8b",
        status="fits",
        num_ctx=8192,
        target_context=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        kv_offload=True,
        concurrency=1,
        memory_budget_bytes=12 * 1024**3,
        estimated_model_bytes=5 * 1024**3,
        estimated_kv_bytes=1024,
        estimated_runtime_overhead_bytes=1024,
        estimated_total_bytes=5 * 1024**3 + 2048,
        ollama_options={"num_ctx": 8192},
        llama_cpp_args=["--ctx-size", "8192"],
        reasons=[],
        warnings=[],
        runtime_advice={"checked": True, "ollama_ready": True, "llama_cpp_ready": False},
    )
    text = render_plan_text(
        plan,
        ModelProfile(name="llama3.1:8b"),
        HardwareProfile("Darwin", "arm64", "Apple M2", 16 * 1024**3, 12 * 1024**3),
    )

    assert "Runtime advice:" in text
    assert "Ollama ready: True" in text
    assert "llama.cpp ready: False" in text


def test_report_renders_markdown_and_csv():
    rows = [{"model": "qwen2.5:7b", "context": 32768, "status": "fits"}]

    markdown = render_report_markdown(rows)
    csv_text = render_report_csv(rows)

    assert "| model | context |" in markdown
    assert "| qwen2.5:7b | 32768 |" in markdown
    assert "model,context,method" in csv_text
    assert "qwen2.5:7b,32768" in csv_text


def test_cli_report_renders_json_file(tmp_path, capsys):
    report_file = tmp_path / "runs.json"
    report_file.write_text(
        json.dumps({"runs": [{"model": "llama3.1:8b", "context": 8192, "status": "ok"}]}),
        encoding="utf-8",
    )

    code = main(["report", str(report_file), "--format", "markdown"])

    assert code == 0
    assert "| llama3.1:8b | 8192 |" in capsys.readouterr().out


def test_report_loader_accepts_utf8_bom(tmp_path):
    report_file = tmp_path / "bom.json"
    report_file.write_bytes(b"\xef\xbb\xbf{\"runs\": []}")

    assert load_report_payload(str(report_file)) == {"runs": []}


def test_report_loader_accepts_collect_directory_bundle(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"task": "collect", "doctor_status": "pass", "model_count": 0, "models": []}),
        encoding="utf-8",
    )
    (tmp_path / "audit.json").write_text(
        json.dumps({"task": "audit", "status": "pass", "summary": {"failed_checks": [], "warning_checks": []}, "checks": []}),
        encoding="utf-8",
    )

    payload = load_report_payload(str(tmp_path))

    assert isinstance(payload, list)
    assert payload[0]["task"] == "collect"
    assert payload[0]["_mackv_opt_source_path"].endswith("manifest.json")
    assert payload[1]["task"] == "audit"


def test_write_experiment_artifacts_writes_json_markdown_and_csv(tmp_path):
    payload = {"runs": [{"model": "llama3.1:8b", "context": 8192, "status": "ok"}]}

    written = write_experiment_artifacts(payload, str(tmp_path), prefix="exp")

    assert set(written) == {"json", "markdown", "csv"}
    assert (tmp_path / "exp.json").exists()
    assert "| llama3.1:8b | 8192 |" in (tmp_path / "exp.md").read_text(encoding="utf-8")
    assert "llama3.1:8b,8192" in (tmp_path / "exp.csv").read_text(encoding="utf-8")


def test_report_normalizes_nested_experiment_payload():
    rows = normalize_report_rows(
        {
            "task": "experiment",
            "model": {"name": "llama3.1:8b"},
            "plans": [{"status": "fits", "num_ctx": 8192, "cache_type_k": "q8_0"}],
            "bench": {
                "runs": [{"model": "llama3.1:8b", "context": 8192, "tokens_per_second": 20.0}],
                "repeat_summaries": [
                    {
                        "model": "llama3.1:8b",
                        "context": 8192,
                        "method": "ollama-api",
                        "runs": 3,
                        "tokens_per_second_mean": 19.5,
                        "tokens_per_second_stdev": 1.0,
                        "success_rate": 1.0,
                    }
                ],
                "stability_summary": {
                    "stable_context_policy": "all",
                    "min_stable_fraction": 1.0,
                    "models": [
                        {
                            "model": "llama3.1:8b",
                            "max_stable_context": 8192,
                            "stable_runs": 1,
                            "unstable_runs": 0,
                            "stable_context_policy": "all",
                            "min_stable_fraction": 1.0,
                            "contexts": {
                                "8192": {
                                    "context": 8192,
                                    "context_stable": True,
                                    "stable_fraction": 1.0,
                                    "stable_runs": 1,
                                    "unstable_runs": 0,
                                    "runs": 1,
                                    "reasons": {},
                                }
                            },
                        }
                    ]
                },
            },
            "needle": {"runs": [{"model": "llama3.1:8b", "context": 8192, "depth": 0.5, "found": True}]},
            "qa": {
                "runs": [
                    {
                        "model": "llama3.1:8b",
                        "context": 8192,
                        "method": "qa",
                        "question_id": "q1",
                        "source": "mini",
                        "found": False,
                        "quality_score": 0.0,
                    }
                ]
            },
        }
    )

    assert len(rows) == 7
    assert rows[0]["method"] == "plan"
    assert rows[1]["tokens_per_second"] == 20.0
    assert rows[2]["tokens_per_second_mean"] == 19.5
    assert rows[3]["method"] == "stability-summary"
    assert rows[3]["max_stable_context"] == 8192
    assert rows[3]["stable_runs"] == 1
    assert rows[4]["method"] == "stability-context"
    assert rows[4]["context_stable"] is True
    assert rows[4]["stable_fraction"] == 1.0
    assert rows[5]["found"] is True
    assert rows[6]["method"] == "qa"
    assert rows[6]["question_id"] == "q1"
    assert rows[6]["source"] == "mini"


def test_report_normalizes_top_level_repeat_summaries():
    rows = normalize_report_rows(
        {
            "runs": [{"model": "m", "context": 8192, "method": "ollama-api", "status": "ok"}],
            "repeat_summaries": [
                {
                    "model": "m",
                    "context": 8192,
                    "method": "ollama-api",
                    "runs": 2,
                    "success_rate": 1.0,
                    "tokens_per_second_mean": 12.0,
                }
            ],
        }
    )

    assert len(rows) == 2
    assert rows[1]["runs"] == 2
    assert rows[1]["success_rate"] == 1.0
    assert rows[1]["tokens_per_second_mean"] == 12.0


def test_paper_table_renderers_filter_context_performance_memory_quality():
    rows = [
        {"model": "m", "context": 8192, "method": "plan", "status": "fits", "cache_type_k": "f16"},
        {"model": "m", "method": "stability-summary", "max_stable_context": 32768, "stable_runs": 2, "unstable_runs": 1},
        {
            "model": "m",
            "context": 8192,
            "method": "ollama-api",
            "tokens_per_second": 12.5,
            "pageout_delta": 2,
            "pageout_bytes_delta": 8192,
            "swapouts_delta": 1,
        },
        {"model": "m", "context": 8192, "method": "ollama-api", "runs": 3, "tokens_per_second_mean": 11.0, "tokens_per_second_stdev": 1.0, "success_rate": 1.0},
        {"model": "m", "context": 8192, "method": "needle", "depth": 0.5, "found": True, "quality_score": 1.0},
        {"model": "m", "context": 8192, "method": "qa", "source": "mini", "question_id": "q1", "found": False, "quality_score": 0.0},
        {"model": "m", "context": 8192, "method": "qa", "source": "mini", "question_id": "q1", "runs": 3, "accuracy": 0.667, "quality_score_mean": 0.667},
        {
            "model": "m",
            "context": 8192,
            "method": "stability-context",
            "stable_context_policy": "all",
            "min_stable_fraction": 1.0,
            "context_stable": False,
            "stable_fraction": 0.667,
            "stable_runs": 2,
            "unstable_runs": 1,
            "runs": 3,
            "instability_reasons": {"status=error": 1},
            "max_stable_context": 32768,
        },
    ]

    context_table = render_paper_table_markdown(rows, "context")
    performance_csv = render_paper_table_csv(rows, "performance")
    memory_table = render_paper_table_markdown(rows, "memory")
    quality_table = render_paper_table_markdown(rows, "quality")
    stability_table = render_paper_table_markdown(rows, "stability")

    assert "| model | context | status |" in context_table
    assert "| m | 8192 | fits |" in context_table
    assert "| m |  |  |  |  |  |  |  |  | 32768 | 2 | 1 |" in context_table
    assert "tokens_per_second" in performance_csv
    assert "12.5" in performance_csv
    assert "11.0" in performance_csv
    assert "peak_memory_bytes" in memory_table
    assert "pageout_delta" in memory_table
    assert "| m | 8192 | ollama-api |  |  |  |  | 2 |  | 8192 |  | 1 |  |  |" in memory_table
    assert "| m | 8192 | 0.5 |  |  | True | 1.0 |" in quality_table
    assert "| m | 8192 |  | mini | q1 | False | 0.0 |" in quality_table
    assert "| m | 8192 |  | mini | q1 |  |  | 0.667 |" in quality_table
    assert "| model | context | stable_context_policy |" in stability_table
    assert "| m | 8192 | all | 1.0 | False | 0.667 | 2 | 1 | 3 | status=error:1 | 32768 |" in stability_table


def test_readiness_table_normalizes_collect_manifest_and_referenced_artifacts(tmp_path):
    doctor = {
        "task": "doctor",
        "status": "pass",
        "hardware": {
            "platform": "Darwin",
            "machine": "arm64",
            "chip": "Apple M3 Pro",
            "total_memory_bytes": 36 * 1024**3,
            "available_memory_bytes": 24 * 1024**3,
            "pressure": "normal",
            "os_version": "macOS 15.5 (24F74)",
            "kernel_version": "25.0.0",
            "power_source": "ac",
            "power_mode": "normal",
            "thermal_state": "0",
        },
        "memory_state": {"memory_pressure": "normal", "swap_bytes": 0},
        "capabilities": {
            "ollama": {"available": True, "version": "0.12.1"},
            "llama_cpp": {"available": True, "version": "b6500"},
            "supports_ollama_num_ctx": True,
            "supports_llama_cpp_cache_type_k": True,
            "supports_llama_cpp_cache_type_v": True,
        },
        "ollama_model_count": 1,
        "checks": [],
    }
    profile = {
        "hardware": doctor["hardware"],
        "ollama": {"available": True, "models": [{"name": "llama3.1:8b"}]},
        "capabilities": doctor["capabilities"],
    }
    capabilities = doctor["capabilities"]
    (tmp_path / "doctor.json").write_text(json.dumps(doctor), encoding="utf-8")
    (tmp_path / "machine-profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (tmp_path / "runtime-capabilities.json").write_text(json.dumps(capabilities), encoding="utf-8")
    manifest = {
        "task": "collect",
        "doctor_status": "pass",
        "model_count": 1,
        "artifacts": {
            "doctor": str(tmp_path / "doctor.json"),
            "profile": str(tmp_path / "machine-profile.json"),
            "capabilities": str(tmp_path / "runtime-capabilities.json"),
        },
        "models": [
            {
                "name": "llama3.1:8b",
                "status": "ok-with-override",
                "normalized_profile_json": "models/llama3.1-8b-profile.json",
                "metadata_audit": {"status": "pass", "missing_required_fields": []},
                "metadata_override": {"applied": True},
                "profile": {"name": "llama3.1:8b"},
            }
        ],
    }

    rows = normalize_report_rows(manifest)
    table = render_paper_table_markdown(rows, "readiness")

    assert any(row["artifact_type"] == "collect-manifest" and row["component"] == "model-metadata" for row in rows)
    assert any(row["artifact_type"] == "doctor" and row["component"] == "summary" for row in rows)
    assert any(row["artifact_type"] == "machine-profile" and row["component"] == "hardware" for row in rows)
    assert any(row["artifact_type"] == "runtime-capabilities" for row in rows)
    assert "| artifact_type | component | status |" in table
    assert "| collect-manifest | model-metadata | pass | llama3.1:8b |" in table
    assert "Apple M3 Pro" in table
    assert "macOS 15.5" in table
    assert "ac" in table
    assert "0.12.1" in table

    compact = render_paper_table_markdown(rows, "readiness-compact")

    assert "| paper_ready | status | artifact_type |" in compact
    assert "| True | pass | readiness-compact |" in compact
    assert "llama3.1:8b" in compact


def test_readiness_table_normalizes_audit_model_metadata_issues():
    rows = normalize_report_rows(
        {
            "task": "audit",
            "status": "fail",
            "summary": {
                "model_count": 1,
                "failed_checks": ["model-metadata"],
                "warning_checks": ["hardware"],
            },
            "checks": [
                {
                    "name": "model-metadata",
                    "status": "fail",
                    "message": "Some model profiles are missing KV-budget-critical metadata.",
                    "next_step": "Fill missing metadata.",
                    "evidence": {
                        "incomplete_models": [
                            {"name": "qwen2.5:7b", "missing_required_fields": ["kv_head_count", "layer_count"]}
                        ]
                    },
                }
            ],
        }
    )

    table = render_paper_table_markdown(rows, "readiness")

    assert rows[0]["failed_checks"] == "model-metadata"
    assert rows[0]["warning_checks"] == "hardware"
    assert any(row["model"] == "qwen2.5:7b" for row in rows)
    assert "kv_head_count, layer_count" in table
    assert "readiness-compact" in render_paper_table_markdown(rows, "readiness-compact")


def test_report_preserves_zero_quality_and_swap_values():
    rows = normalize_report_rows(
        {
            "runs": [
                {
                    "model": "m",
                    "context": 8192,
                    "method": "needle",
                    "quality_score": 0.0,
                    "swap_bytes": 0,
                }
            ]
        }
    )

    assert rows[0]["quality_score"] == 0.0
    assert rows[0]["swap_bytes"] == 0


def test_report_normalizes_top_level_stability_summary():
    rows = normalize_report_rows(
        {
            "runs": [
                {
                    "model": "m",
                    "context": 8192,
                    "method": "ollama-api",
                    "status": "ok",
                    "stable": True,
                    "stability_reason": "stable",
                }
            ],
            "stability_summary": {
                "stable_context_policy": "all",
                "min_stable_fraction": 1.0,
                "models": [
                    {
                        "model": "m",
                        "max_stable_context": 8192,
                        "stable_runs": 1,
                        "unstable_runs": 0,
                        "contexts": {
                            "8192": {
                                "context": 8192,
                                "context_stable": True,
                                "stable_fraction": 1.0,
                                "stable_runs": 1,
                                "unstable_runs": 0,
                                "runs": 1,
                                "reasons": {},
                            }
                        },
                    }
                ]
            },
        }
    )

    assert len(rows) == 3
    assert rows[0]["stable"] is True
    assert rows[0]["stability_reason"] == "stable"
    assert rows[1]["method"] == "stability-summary"
    assert rows[1]["max_stable_context"] == 8192
    assert rows[2]["method"] == "stability-context"
    assert rows[2]["context_stable"] is True


def test_context_paper_table_includes_stability_summary_rows():
    rows = normalize_report_rows(
        {
            "runs": [{"model": "m", "context": 8192, "method": "ollama-api", "status": "ok"}],
            "stability_summary": {
                "stable_context_policy": "all",
                "min_stable_fraction": 1.0,
                "models": [
                    {
                        "model": "m",
                        "max_stable_context": 16384,
                        "stable_runs": 2,
                        "unstable_runs": 1,
                        "contexts": {
                            "8192": {
                                "context": 8192,
                                "context_stable": True,
                                "stable_fraction": 1.0,
                                "stable_runs": 2,
                                "unstable_runs": 0,
                                "runs": 2,
                                "reasons": {},
                            },
                            "16384": {
                                "context": 16384,
                                "context_stable": False,
                                "stable_fraction": 0.0,
                                "stable_runs": 0,
                                "unstable_runs": 1,
                                "runs": 1,
                                "reasons": {"status=error": 1},
                            },
                        },
                    }
                ]
            },
        }
    )

    context_table = render_paper_table_markdown(rows, "context")

    assert "max_stable_context" in context_table
    assert "| m |  |  |  |  |  |  |  |  | 16384 | 2 | 1 |" in context_table


def test_stability_paper_table_includes_context_breakdown():
    rows = normalize_report_rows(
        {
            "runs": [{"model": "m", "context": 8192, "method": "ollama-api", "status": "ok"}],
            "stability_summary": {
                "stable_context_policy": "fraction",
                "min_stable_fraction": 0.67,
                "models": [
                    {
                        "model": "m",
                        "max_stable_context": 8192,
                        "stable_runs": 2,
                        "unstable_runs": 1,
                        "contexts": {
                            "8192": {
                                "context": 8192,
                                "context_stable": True,
                                "stable_fraction": 0.666667,
                                "stable_runs": 2,
                                "unstable_runs": 1,
                                "runs": 3,
                                "reasons": {"status=error": 1},
                            }
                        },
                    }
                ],
            },
        }
    )

    stability_table = render_paper_table_markdown(rows, "stability")

    assert "| model | context | stable_context_policy | min_stable_fraction |" in stability_table
    assert "| m | 8192 | fraction | 0.67 | True | 0.666667 | 2 | 1 | 3 | status=error:1 | 8192 |" in stability_table


def test_cli_report_renders_fixed_paper_table(tmp_path, capsys):
    report_file = tmp_path / "experiment.json"
    report_file.write_text(
        json.dumps(
            {
                "task": "experiment",
                "model": {"name": "llama3.1:8b"},
                "needle": {"runs": [{"model": "llama3.1:8b", "context": 8192, "depth": 0.5, "found": True, "quality_score": 1.0}]},
            }
        ),
        encoding="utf-8",
    )

    code = main(["report", str(report_file), "--table", "quality", "--format", "markdown"])

    assert code == 0
    output = capsys.readouterr().out
    assert "| model | context | depth | source | question_id | found | quality_score |" in output
    assert "| llama3.1:8b | 8192 | 0.5 |  |  | True | 1.0 |" in output


def test_write_paper_tables_writes_selected_tables(tmp_path):
    rows = [
        {"model": "m", "context": 8192, "method": "plan", "status": "fits"},
        {"model": "m", "context": 8192, "method": "needle", "depth": 0.5, "found": False, "quality_score": 0.0},
    ]

    written = write_paper_tables(rows, str(tmp_path), prefix="paper", tables=["context", "quality", "stability"])

    assert set(written) == {"context", "quality", "stability"}
    assert (tmp_path / "paper-context.md").exists()
    assert "| m | 8192 | fits |" in (tmp_path / "paper-context.md").read_text(encoding="utf-8")
    assert "0.0" in (tmp_path / "paper-quality.md").read_text(encoding="utf-8")
    assert (tmp_path / "paper-stability.md").exists()


def test_cli_report_writes_all_fixed_tables(tmp_path, capsys):
    report_file = tmp_path / "experiment.json"
    report_file.write_text(
        json.dumps(
            {
                "task": "experiment",
                "model": {"name": "llama3.1:8b"},
                "plans": [{"status": "fits", "num_ctx": 8192}],
                "bench": {
                    "runs": [{"model": "llama3.1:8b", "context": 8192, "method": "ollama-api"}],
                    "stability_summary": {
                        "models": [
                            {
                                "model": "llama3.1:8b",
                                "max_stable_context": 8192,
                                "stable_runs": 1,
                                "unstable_runs": 0,
                            }
                        ]
                    },
                },
                "needle": {"runs": [{"model": "llama3.1:8b", "context": 8192, "method": "needle", "found": False, "quality_score": 0.0}]},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "report",
            str(report_file),
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "tables",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["artifacts"]) == {
        "readiness-compact",
        "readiness",
        "context",
        "performance",
        "memory",
        "quality",
        "stability",
    }
    assert (tmp_path / "tables-readiness-compact.md").exists()
    assert (tmp_path / "tables-readiness.md").exists()
    assert (tmp_path / "tables-context.md").exists()
    assert (tmp_path / "tables-quality.md").exists()
    assert (tmp_path / "tables-stability.md").exists()
