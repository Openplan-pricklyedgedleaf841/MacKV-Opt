import json

from mackv_opt.cli import main


def test_cli_plan_json_uses_manual_model_metadata(monkeypatch, capsys):
    monkeypatch.setattr("mackv_opt.cli.detect_runtime_capabilities", lambda: None)

    code = main(
        [
            "plan",
            "llama3.1:8b",
            "--target-context",
            "64k",
            "--memory-budget",
            "12GiB",
            "--model-size",
            "4.8GiB",
            "--hidden-size",
            "4096",
            "--layers",
            "32",
            "--kv-heads",
            "8",
            "--hardware-memory",
            "16GiB",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["name"] == "llama3.1:8b"
    assert payload["plan"]["num_ctx"] == 65536
    assert payload["plan"]["ollama_options"]["num_ctx"] == 65536
    assert payload["plan"]["runtime_advice"]["checked"] is False


def test_cli_bench_dry_run_outputs_matrix(capsys):
    code = main(
        [
            "bench",
            "--models",
            "llama3.1:8b,qwen2.5:7b",
            "--contexts",
            "8k,16k",
            "--dry-run",
            "--json",
            "--repeats",
            "3",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["jobs"]) == 4
    assert payload["planned_run_count"] == 12
    assert payload["jobs"][0]["context"] == 8192


def test_cli_capabilities_json_outputs_runtime_probe(monkeypatch, capsys):
    monkeypatch.setattr(
        "mackv_opt.cli.detect_runtime_capabilities",
        lambda: type(
            "FakeCapabilities",
            (),
            {
                "to_dict": lambda self: {
                    "ollama": {"available": True},
                    "llama_cpp": {"available": False},
                    "supports_ollama_num_ctx": True,
                    "supports_ollama_num_gpu": True,
                    "supports_llama_cpp_cache_type_k": False,
                    "supports_llama_cpp_cache_type_v": False,
                    "warnings": [],
                }
            },
        )(),
    )

    code = main(["capabilities", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ollama"]["available"] is True
    assert payload["supports_ollama_num_ctx"] is True


def test_cli_doctor_json_outputs_preflight_payload(monkeypatch, capsys):
    monkeypatch.setattr(
        "mackv_opt.cli.doctor_payload",
        lambda: {
            "task": "doctor",
            "status": "pass",
            "checks": [{"name": "hardware", "status": "pass", "message": "ok"}],
            "next_steps": [],
        },
    )

    code = main(["doctor", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "doctor"
    assert payload["status"] == "pass"
    assert payload["checks"][0]["name"] == "hardware"


def test_cli_doctor_text_outputs_rendered_preflight(monkeypatch, capsys):
    monkeypatch.setattr("mackv_opt.cli.doctor_payload", lambda: {"status": "warn", "checks": []})
    monkeypatch.setattr("mackv_opt.cli.render_doctor_text", lambda payload: f"doctor={payload['status']}")

    code = main(["doctor"])

    assert code == 0
    assert capsys.readouterr().out.strip() == "doctor=warn"


def test_cli_collect_json_outputs_manifest(monkeypatch, tmp_path, capsys):
    overrides = tmp_path / "overrides.json"
    overrides.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("mackv_opt.cli.load_model_metadata_overrides", lambda path: {"llama3.1:8b": {"hidden_size": 4096}})

    def fake_collect(output_dir, **kwargs):
        assert output_dir == str(tmp_path)
        assert kwargs["models"] == ["llama3.1:8b", "qwen2.5:7b"]
        assert kwargs["include_raw_model_json"] is False
        assert kwargs["model_metadata_overrides"] == {"llama3.1:8b": {"hidden_size": 4096}}
        assert kwargs["model_metadata_override_path"] == str(overrides)
        return {"task": "collect", "model_count": 2, "artifacts": {"manifest": str(tmp_path / "manifest.json")}}

    monkeypatch.setattr("mackv_opt.cli.collect_artifacts", fake_collect)

    code = main(
        [
            "collect",
            "--output-dir",
            str(tmp_path),
            "--models",
            "llama3.1:8b,qwen2.5:7b",
            "--skip-raw-model-json",
            "--model-metadata-overrides",
            str(overrides),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "collect"
    assert payload["model_count"] == 2


def test_cli_collect_text_outputs_manifest_markdown(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("mackv_opt.cli.collect_artifacts", lambda output_dir, **kwargs: {"task": "collect"})
    monkeypatch.setattr("mackv_opt.cli.render_collect_markdown", lambda payload: "# manifest\n")

    code = main(["collect", "--output-dir", str(tmp_path)])

    assert code == 0
    assert capsys.readouterr().out == "# manifest\n"


def test_cli_audit_json_outputs_payload_and_exit_zero(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("mackv_opt.cli.load_collect_manifest", lambda path: {"task": "collect"})

    def fake_audit(manifest_payload, **kwargs):
        assert manifest_payload["task"] == "collect"
        assert kwargs["manifest_path"] == str(manifest)
        assert kwargs["fail_on_missing_metadata"] is True
        assert kwargs["require_artifacts"] is False
        assert kwargs["require_apple_silicon"] is True
        return {"task": "audit", "status": "pass", "checks": []}

    monkeypatch.setattr("mackv_opt.cli.audit_collect_manifest", fake_audit)

    code = main(
        [
            "audit",
            str(manifest),
            "--fail-on-missing-metadata",
            "--no-require-artifacts",
            "--require-apple-silicon",
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "audit"
    assert payload["status"] == "pass"
    assert output.exists()
    assert markdown.exists()


def test_cli_audit_text_returns_one_on_failure(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("mackv_opt.cli.load_collect_manifest", lambda path: {"task": "collect"})
    monkeypatch.setattr(
        "mackv_opt.cli.audit_collect_manifest",
        lambda manifest_payload, **kwargs: {"task": "audit", "status": "fail", "checks": []},
    )
    monkeypatch.setattr("mackv_opt.cli.render_collect_audit_text", lambda payload: "audit failed")

    code = main(["audit", str(manifest)])

    assert code == 1
    assert capsys.readouterr().out.strip() == "audit failed"


def test_cli_bench_execute_outputs_run_payload(monkeypatch, capsys):
    def fake_execute(models, contexts, **kwargs):
        assert kwargs["include_memory_series"] is True
        assert kwargs["repeats"] == 2
        assert kwargs["stability_config"].max_swap_bytes == 0
        assert kwargs["stability_config"].min_tokens_per_second == 1.5
        assert kwargs["stability_config"].stable_context_policy == "fraction"
        assert kwargs["stability_config"].min_stable_fraction == 0.67
        return {
            "dry_run": False,
            "runs": [
                {
                    "model": list(models)[0],
                    "context": 8192,
                    "status": "ok",
                    "tokens_per_second": 12.5,
                }
            ],
        }

    monkeypatch.setattr("mackv_opt.cli.execute_bench_payload", fake_execute)

    code = main(
        [
            "bench",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--execute",
            "--json",
            "--include-memory-series",
            "--repeats",
            "2",
            "--max-swap",
            "0",
            "--min-tokens-per-second",
            "1.5",
            "--stable-context-policy",
            "fraction",
            "--min-stable-fraction",
            "0.67",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert payload["runs"][0]["tokens_per_second"] == 12.5


def test_cli_bench_writes_artifacts(tmp_path, capsys):
    code = main(
        [
            "bench",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--dry-run",
            "--json",
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "matrix",
            "--save-formats",
            "json,markdown",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["artifacts"]) == {"json", "markdown"}
    assert (tmp_path / "matrix.json").exists()
    assert (tmp_path / "matrix.md").exists()


def test_cli_baseline_template_writes_comparison_directories(tmp_path, capsys):
    code = main(
        [
            "baseline-template",
            "--output-dir",
            str(tmp_path),
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k,16k",
            "--memory-budget",
            "12GiB",
            "--manual-context",
            "16k",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    model_dir = tmp_path / "llama3.1-8b"
    assert payload["task"] == "baseline-template"
    assert payload["baselines"] == ["default", "manual-num-ctx", "mackv-opt"]
    assert (model_dir / "default" / "README.md").exists()
    assert (model_dir / "manual-num-ctx" / "run.sh").exists()
    assert (model_dir / "mackv-opt" / "manifest.json").exists()
    default_run = (model_dir / "default" / "run.sh").read_text(encoding="utf-8")
    manual_run = (model_dir / "manual-num-ctx" / "run.sh").read_text(encoding="utf-8")
    optimized_readme = (model_dir / "mackv-opt" / "README.md").read_text(encoding="utf-8")
    assert "--use-ollama-default-options" in default_run
    assert "mackv-opt bench" in default_run
    assert "--contexts 16k" in manual_run
    assert "--memory-budget 12GiB" in optimized_readme


def test_cli_compare_renders_markdown_and_writes_output(tmp_path, capsys):
    artifact = tmp_path / "run.json"
    output = tmp_path / "compare.md"
    artifact.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "m",
                        "context": 8192,
                        "method": "ollama-api",
                        "tokens_per_second": 12.0,
                    }
                ],
                "stability_summary": {
                    "models": [
                        {
                            "model": "m",
                            "max_stable_context": 8192,
                            "stable_runs": 1,
                            "unstable_runs": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    code = main(["compare", f"baseline={artifact}", "--output", str(output)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert "| baseline | m | 8192 |" in output.read_text(encoding="utf-8")


def test_cli_report_readiness_accepts_multiple_inputs(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    audit = tmp_path / "audit.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "collect",
                "doctor_status": "warn",
                "model_count": 1,
                "models": [
                    {
                        "name": "llama3.1:8b",
                        "status": "ok",
                        "metadata_audit": {"status": "warn", "missing_required_fields": ["kv_head_count"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "task": "audit",
                "status": "warn",
                "summary": {"model_count": 1, "failed_checks": [], "warning_checks": ["model-metadata"]},
                "checks": [],
            }
        ),
        encoding="utf-8",
    )

    code = main(["report", str(manifest), str(audit), "--table", "readiness"])

    assert code == 0
    output = capsys.readouterr().out
    assert "| artifact_type | component | status |" in output
    assert "| collect-manifest | model-metadata | warn | llama3.1:8b |" in output
    assert "| collect-audit | summary | warn |" in output

    code = main(["report", str(manifest), str(audit), "--table", "readiness-compact"])

    assert code == 0
    compact = capsys.readouterr().out
    assert "| paper_ready | status | artifact_type |" in compact
    assert "| False | warn | readiness-compact |" in compact


def test_cli_compare_json_outputs_rows(tmp_path, capsys):
    artifact = tmp_path / "run.json"
    artifact.write_text(
        json.dumps({"runs": [{"model": "m", "context": 8192, "method": "ollama-api"}]}),
        encoding="utf-8",
    )

    code = main(["compare", f"default={artifact}", "--format", "json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "compare"
    assert payload["rows"][0]["label"] == "default"
    assert payload["rows"][0]["baseline_label"] == "default"


def test_cli_compare_accepts_baseline_label(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(
        json.dumps(
            {
                "runs": [{"model": "m", "context": 8192, "method": "ollama-api", "tokens_per_second": 10.0}],
                "stability_summary": {
                    "models": [{"model": "m", "max_stable_context": 8192, "stable_runs": 1, "unstable_runs": 0}]
                },
            }
        ),
        encoding="utf-8",
    )
    optimized.write_text(
        json.dumps(
            {
                "runs": [{"model": "m", "context": 16384, "method": "ollama-api", "tokens_per_second": 20.0}],
                "stability_summary": {
                    "models": [{"model": "m", "max_stable_context": 16384, "stable_runs": 1, "unstable_runs": 0}]
                },
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "compare",
            f"default={baseline}",
            f"mackv={optimized}",
            "--baseline-label",
            "default",
            "--format",
            "json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_label"] == "default"
    assert payload["rows"][1]["max_stable_context_vs_baseline"] == 2.0
    assert payload["rows"][1]["tokens_per_second_vs_baseline"] == 2.0


def test_cli_experiment_passes_stability_config(monkeypatch, capsys):
    def fake_build(model, hardware, contexts, **kwargs):
        assert model.name == "llama3.1:8b"
        assert list(contexts) == ["8k"]
        assert kwargs["stability_config"].max_swap_bytes == 0
        assert kwargs["stability_config"].min_tokens_per_second == 2.5
        assert kwargs["stability_config"].stable_context_policy == "all"
        assert kwargs["stability_config"].min_stable_fraction == 1.0
        return {"task": "experiment", "dry_run": False, "plans": [], "summary": {}}

    monkeypatch.setattr("mackv_opt.cli.build_experiment_payload", fake_build)

    code = main(
        [
            "experiment",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--model-size",
            "4.8GiB",
            "--hidden-size",
            "4096",
            "--layers",
            "32",
            "--heads",
            "32",
            "--kv-heads",
            "8",
            "--hardware-memory",
            "16GiB",
            "--execute",
            "--json",
            "--skip-capability-check",
            "--max-swap",
            "0",
            "--min-tokens-per-second",
            "2.5",
            "--stable-context-policy",
            "all",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "experiment"


def test_cli_needle_dry_run_outputs_quality_matrix(capsys):
    code = main(
        [
            "needle",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--depths",
            "0.1,0.9",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "needle"
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["needle"].startswith("MACKV-")


def test_cli_needle_execute_writes_artifacts(monkeypatch, tmp_path, capsys):
    def fake_execute(models, contexts, depths, **kwargs):
        assert kwargs["repeats"] == 2
        return {
            "dry_run": False,
            "task": "needle",
            "runs": [{"model": "llama3.1:8b", "context": 8192, "found": True, "quality_score": 1.0}],
            "summary": {"total": 1, "found": 1, "accuracy": 1.0},
        }

    monkeypatch.setattr("mackv_opt.cli.execute_needle_payload", fake_execute)

    code = main(
        [
            "needle",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--execute",
            "--json",
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "needle",
            "--repeats",
            "2",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["accuracy"] == 1.0
    assert (tmp_path / "needle.json").exists()
    assert (tmp_path / "needle.md").exists()


def test_cli_qa_dry_run_outputs_quality_matrix(capsys):
    code = main(
        [
            "qa",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--depths",
            "0.1,0.9",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "qa"
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["expected_answer"].startswith("MACKV-QA-")
    assert "document" not in payload["jobs"][0]


def test_cli_qa_execute_writes_artifacts(monkeypatch, tmp_path, capsys):
    def fake_execute(models, contexts, **kwargs):
        assert kwargs["dataset_path"] is None
        assert kwargs["repeats"] == 2
        return {
            "dry_run": False,
            "task": "qa",
            "runs": [{"model": "llama3.1:8b", "context": 8192, "method": "qa", "found": True, "quality_score": 1.0}],
            "summary": {"total": 1, "found": 1, "accuracy": 1.0},
        }

    monkeypatch.setattr("mackv_opt.cli.execute_qa_payload", fake_execute)

    code = main(
        [
            "qa",
            "--models",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--execute",
            "--json",
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "qa",
            "--repeats",
            "2",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["accuracy"] == 1.0
    assert (tmp_path / "qa.json").exists()
    assert (tmp_path / "qa.md").exists()


def test_cli_experiment_dry_run_writes_artifacts(tmp_path, capsys):
    code = main(
        [
            "experiment",
            "llama3.1:8b",
            "--contexts",
            "8k",
            "--memory-budget",
            "12GiB",
            "--model-size",
            "4.8GiB",
            "--hidden-size",
            "4096",
            "--layers",
            "32",
            "--heads",
            "32",
            "--kv-heads",
            "8",
            "--hardware-memory",
            "16GiB",
            "--dry-run",
            "--json",
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "experiment",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "experiment"
    assert len(payload["plans"]) == 1
    assert (tmp_path / "experiment.json").exists()
    assert (tmp_path / "experiment.md").exists()


def test_cli_plot_memory_writes_svg(tmp_path, capsys):
    input_file = tmp_path / "bench.json"
    output_file = tmp_path / "memory.svg"
    input_file.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "llama3.1:8b",
                        "context": 8192,
                        "memory_series": [
                            {"timestamp": "t0", "process_memory_bytes": 1024},
                            {"timestamp": "t1", "process_memory_bytes": 2048},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(["plot-memory", str(input_file), "--output", str(output_file)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["svg"] == str(output_file)
    assert output_file.exists()
