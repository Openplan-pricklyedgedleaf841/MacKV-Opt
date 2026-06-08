from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .baseline import write_baseline_template
from .baseline_summary import (
    build_baseline_summary_payload,
    render_baseline_summary_csv,
    render_baseline_summary_json,
    render_baseline_summary_markdown,
)
from .bench import DEFAULT_OLLAMA_BASE_URL, DEFAULT_PROMPT, dry_run_payload, execute_bench_payload
from .capabilities import detect_runtime_capabilities
from .collect import (
    audit_collect_manifest,
    collect_artifacts,
    load_collect_manifest,
    load_model_metadata_overrides,
    render_collect_audit_text,
    render_collect_markdown,
)
from .compare import build_compare_payload, parse_compare_input, render_compare_csv, render_compare_markdown
from .doctor import doctor_payload, render_doctor_text
from .experiment import build_experiment_payload
from .models import ModelProfile
from .ollama import build_run_command, load_model_profile
from .planner import PlannerConfig, create_plan
from .plot import write_memory_svg
from .profiler import dumps_json, get_hardware_profile, profile_payload
from .quality import dry_run_needle_payload, dry_run_qa_payload, execute_needle_payload, execute_qa_payload
from .report import (
    load_report_payload,
    normalize_report_rows,
    render_fixed_table_csv,
    render_fixed_table_markdown,
    render_plan_text,
    render_report_csv,
    render_report_markdown,
    write_experiment_artifacts,
    write_fixed_tables,
)
from .stability import StabilityConfig
from .units import parse_context, parse_size


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "profile":
        return _cmd_profile(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "collect":
        return _cmd_collect(args)
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "capabilities":
        return _cmd_capabilities(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "auto":
        return _cmd_auto(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "baseline-template":
        return _cmd_baseline_template(args)
    if args.command == "needle":
        return _cmd_needle(args)
    if args.command == "qa":
        return _cmd_qa(args)
    if args.command == "experiment":
        return _cmd_experiment(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "baseline-summary":
        return _cmd_baseline_summary(args)
    if args.command == "plot-memory":
        return _cmd_plot_memory(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mackv-opt",
        description="Plan KV cache and context settings for Mac local LLM inference.",
    )
    sub = parser.add_subparsers(dest="command")

    profile = sub.add_parser("profile", help="Detect hardware and local Ollama models.")
    profile.add_argument("--json", action="store_true", help="Emit JSON.")

    doctor = sub.add_parser("doctor", help="Run a read-only local readiness check.")
    doctor.add_argument("--json", action="store_true", help="Emit JSON.")

    collect = sub.add_parser("collect", help="Collect read-only diagnostics for repeatable local validation.")
    collect.add_argument("--output-dir", required=True, help="Directory for doctor/profile/capability/model outputs.")
    collect.add_argument("--models", help="Comma-separated model names. Defaults to local Ollama model list.")
    collect.add_argument(
        "--skip-raw-model-json",
        action="store_true",
        help="Do not write raw `ollama show --json` payloads for models.",
    )
    collect.add_argument(
        "--model-metadata-overrides",
        help="Optional JSON file with manual model metadata overrides keyed by model name.",
    )
    collect.add_argument("--json", action="store_true", help="Emit JSON manifest.")

    audit = sub.add_parser("audit", help="Audit a collect manifest before executable validation runs.")
    audit.add_argument("manifest", help="Path to collect/manifest.json.")
    audit.add_argument(
        "--fail-on-missing-metadata",
        action="store_true",
        help="Treat missing KV-budget-critical model metadata as a failure.",
    )
    audit.add_argument(
        "--no-require-artifacts",
        action="store_true",
        help="Warn instead of failing when referenced collection artifact files are missing.",
    )
    audit.add_argument(
        "--require-apple-silicon",
        action="store_true",
        help="Fail unless the collected machine profile is Apple Silicon.",
    )
    audit.add_argument("--output", help="Write JSON audit payload to this path.")
    audit.add_argument("--markdown-output", help="Write Markdown audit report to this path.")
    audit.add_argument("--json", action="store_true", help="Emit JSON.")

    capabilities = sub.add_parser("capabilities", help="Detect local Ollama and llama.cpp runtime capabilities.")
    capabilities.add_argument("--json", action="store_true", help="Emit JSON.")

    plan = sub.add_parser("plan", help="Create an optimization plan for a model.")
    _add_model_args(plan)
    plan.add_argument("model")
    plan.add_argument("--target-context", default="8k")
    plan.add_argument("--memory-budget")
    plan.add_argument("--hardware-memory")
    plan.add_argument("--skip-capability-check", action="store_true")
    plan.add_argument("--json", action="store_true")

    auto = sub.add_parser("auto", help="Plan the largest safe context and print or execute an Ollama command.")
    _add_model_args(auto)
    auto.add_argument("model")
    auto.add_argument("--target-context", default="128k")
    auto.add_argument("--memory-budget")
    auto.add_argument("--hardware-memory")
    auto.add_argument("--prompt")
    auto.add_argument("--skip-capability-check", action="store_true")
    auto.add_argument("--execute", action="store_true", help="Execute local Ollama command instead of printing it.")
    auto.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="Print or execute an Ollama run command with planned options.")
    _add_model_args(run)
    run.add_argument("model")
    run.add_argument("--target-context", default="8k")
    run.add_argument("--memory-budget")
    run.add_argument("--hardware-memory")
    run.add_argument("--prompt")
    run.add_argument("--skip-capability-check", action="store_true")
    run.add_argument("--execute", action="store_true", help="Execute command instead of printing it.")
    run.add_argument("--json", action="store_true")

    bench = sub.add_parser("bench", help="Build a benchmark matrix.")
    bench.add_argument("--models", required=True, help="Comma-separated model names.")
    bench.add_argument("--contexts", required=True, help="Comma-separated contexts, e.g. 8k,16k,32k.")
    bench.add_argument("--dry-run", action="store_true")
    bench.add_argument("--execute", action="store_true", help="Run the benchmark through the local Ollama API.")
    bench.add_argument("--prompt", default=DEFAULT_PROMPT)
    bench.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    bench.add_argument("--timeout", type=float, default=300.0, help="Per-run timeout in seconds.")
    bench.add_argument("--num-predict", type=int, default=128, help="Generated token limit for each run.")
    bench.add_argument(
        "--use-ollama-default-options",
        action="store_true",
        help="Do not send num_ctx; useful for default Ollama baseline outputs.",
    )
    bench.add_argument("--repeats", type=int, default=1, help="Repeat each benchmark cell this many times.")
    bench.add_argument("--memory-sample-interval", type=float, default=0.5, help="Seconds between memory samples.")
    bench.add_argument("--include-memory-series", action="store_true", help="Include every memory sample in JSON output.")
    _add_stability_args(bench)
    bench.add_argument("--output-dir", help="Write benchmark artifacts into this directory.")
    bench.add_argument("--output-prefix", default="mackv-opt-bench")
    bench.add_argument(
        "--save-formats",
        default="json,markdown,csv",
        help="Comma-separated artifact formats: json,markdown,csv.",
    )
    bench.add_argument("--json", action="store_true")

    baseline_template = sub.add_parser(
        "baseline-template",
        help="Create default, manual-num-ctx, and mackv-opt baseline output directories.",
    )
    baseline_template.add_argument("--output-dir", required=True, help="Directory that will receive template folders.")
    baseline_template.add_argument("--models", required=True, help="Comma-separated model names.")
    baseline_template.add_argument("--contexts", required=True, help="Comma-separated contexts, e.g. 8k,16k,32k.")
    baseline_template.add_argument("--memory-budget", help="Memory budget for the mackv-opt baseline command.")
    baseline_template.add_argument("--manual-context", help="Context for the manual num_ctx baseline. Defaults to max context.")
    baseline_template.add_argument("--prompt", default=DEFAULT_PROMPT)
    baseline_template.add_argument("--repeats", type=int, default=3)
    baseline_template.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="Render run or readiness JSON as a table.")
    report.add_argument(
        "inputs",
        nargs="+",
        help="JSON file or collect directory from plan, run, bench, experiment, collect, audit, doctor, profile, or capabilities logs.",
    )
    report.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    report.add_argument(
        "--table",
        choices=["all", "readiness", "readiness-compact", "context", "performance", "memory", "quality", "stability"],
        default="all",
    )
    report.add_argument("--output-dir", help="Write fixed tables into this directory.")
    report.add_argument("--output-prefix", default="mackv-opt-table")
    report.add_argument("--tables", default="readiness-compact,readiness,context,performance,memory,quality,stability")

    compare = sub.add_parser("compare", help="Compare multiple run outputs as a baseline table.")
    compare.add_argument(
        "inputs",
        nargs="+",
        help="Output paths, optionally labeled as LABEL=path.",
    )
    compare.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    compare.add_argument("--output", help="Write the rendered comparison table to this path.")
    compare.add_argument("--baseline-label", help="Label to use as the relative-improvement baseline.")

    baseline_summary = sub.add_parser(
        "baseline-summary",
        help="Summarize default/manual-num-ctx/mackv-opt baseline outputs.",
    )
    baseline_summary.add_argument("machine_dir", help="Directory containing per-model baseline folders.")
    baseline_summary.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    baseline_summary.add_argument("--output", help="Write the rendered baseline table to this path.")

    plot_memory = sub.add_parser("plot-memory", help="Render memory_series from JSON as an SVG plot.")
    plot_memory.add_argument("input", help="Benchmark or experiment JSON containing memory_series.")
    plot_memory.add_argument("--output", required=True, help="Output SVG path.")
    plot_memory.add_argument("--series-index", type=int, default=0)

    needle = sub.add_parser("needle", help="Run Needle-in-a-Haystack long-context quality checks.")
    needle.add_argument("--models", required=True, help="Comma-separated model names.")
    needle.add_argument("--contexts", required=True, help="Comma-separated contexts, e.g. 8k,16k,32k.")
    needle.add_argument("--depths", default="0.1,0.5,0.9", help="Comma-separated depths, e.g. 0.1,0.5,90%.")
    needle.add_argument("--dry-run", action="store_true")
    needle.add_argument("--execute", action="store_true", help="Run the quality check through the local Ollama API.")
    needle.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    needle.add_argument("--timeout", type=float, default=300.0)
    needle.add_argument("--num-predict", type=int, default=64)
    needle.add_argument("--repeats", type=int, default=1, help="Repeat each quality cell this many times.")
    needle.add_argument("--output-dir", help="Write quality artifacts into this directory.")
    needle.add_argument("--output-prefix", default="mackv-opt-needle")
    needle.add_argument("--save-formats", default="json,markdown,csv")
    needle.add_argument("--json", action="store_true")

    qa = sub.add_parser("qa", help="Run LongBench-style long-document QA checks.")
    qa.add_argument("--models", required=True, help="Comma-separated model names.")
    qa.add_argument("--contexts", required=True, help="Comma-separated contexts, e.g. 8k,16k,32k.")
    qa.add_argument("--dataset", help="Optional JSONL dataset with document, question, and expected_answer fields.")
    qa.add_argument("--depths", default="10%,50%,90%", help="Synthetic-only comma-separated answer depths.")
    qa.add_argument("--dry-run", action="store_true")
    qa.add_argument("--execute", action="store_true", help="Run the QA check through the local Ollama API.")
    qa.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    qa.add_argument("--timeout", type=float, default=300.0)
    qa.add_argument("--num-predict", type=int, default=96)
    qa.add_argument("--repeats", type=int, default=1, help="Repeat each QA cell this many times.")
    qa.add_argument("--output-dir", help="Write QA artifacts into this directory.")
    qa.add_argument("--output-prefix", default="mackv-opt-qa")
    qa.add_argument("--save-formats", default="json,markdown,csv")
    qa.add_argument("--json", action="store_true")

    experiment = sub.add_parser("experiment", help="Run plan, benchmark, and quality checks as one validation flow.")
    _add_model_args(experiment)
    experiment.add_argument("model")
    experiment.add_argument("--contexts", required=True, help="Comma-separated contexts, e.g. 8k,16k,32k.")
    experiment.add_argument("--memory-budget")
    experiment.add_argument("--hardware-memory")
    experiment.add_argument("--depths", default="10%,50%,90%")
    experiment.add_argument("--dry-run", action="store_true")
    experiment.add_argument("--execute", action="store_true")
    experiment.add_argument("--skip-bench", action="store_true")
    experiment.add_argument("--skip-needle", action="store_true")
    experiment.add_argument("--skip-qa", action="store_true")
    experiment.add_argument("--qa-dataset", help="Optional JSONL dataset for LongBench-style QA checks.")
    experiment.add_argument("--skip-capability-check", action="store_true")
    experiment.add_argument("--prompt", default=DEFAULT_PROMPT)
    experiment.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    experiment.add_argument("--timeout", type=float, default=300.0)
    experiment.add_argument("--bench-num-predict", type=int, default=128)
    experiment.add_argument("--needle-num-predict", type=int, default=64)
    experiment.add_argument("--qa-num-predict", type=int, default=96)
    experiment.add_argument("--repeats", type=int, default=1, help="Repeat each executable bench and quality cell.")
    experiment.add_argument("--memory-sample-interval", type=float, default=0.5)
    experiment.add_argument("--include-memory-series", action="store_true")
    _add_stability_args(experiment)
    experiment.add_argument("--output-dir", help="Write experiment artifacts into this directory.")
    experiment.add_argument("--output-prefix", default="mackv-opt-experiment")
    experiment.add_argument("--save-formats", default="json,markdown,csv")
    experiment.add_argument("--json", action="store_true")
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-size", help="Manual model size, e.g. 4.8GiB.")
    parser.add_argument("--parameters", help="Manual parameter count, e.g. 8B.")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int, help="Manual attention head count.")
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--family", default="unknown")
    parser.add_argument("--architecture", default="unknown")


def _add_stability_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-swap",
        default="512MiB",
        help="Mark a run unstable if swap growth exceeds this amount; use 0 to disallow swap.",
    )
    parser.add_argument(
        "--min-tokens-per-second",
        type=float,
        help="Mark a run unstable if output tokens/s falls below this threshold.",
    )
    parser.add_argument(
        "--stable-context-policy",
        choices=["any", "all", "fraction"],
        default="any",
        help="Policy for counting a context as stable across repeats.",
    )
    parser.add_argument(
        "--min-stable-fraction",
        type=float,
        default=1.0,
        help="Required stable-run fraction when --stable-context-policy=fraction.",
    )


def _cmd_profile(args: argparse.Namespace) -> int:
    payload = profile_payload()
    if args.json:
        print(dumps_json(payload))
    else:
        hardware = payload["hardware"]
        print(f"Platform: {hardware['platform']}/{hardware['machine']}")
        print(f"Chip: {hardware['chip']}")
        print(f"Ollama available: {payload['ollama']['available']}")
        caps = payload.get("capabilities", {})
        if isinstance(caps, dict):
            llama_cpp = caps.get("llama_cpp", {})
            print(f"llama.cpp available: {bool(isinstance(llama_cpp, dict) and llama_cpp.get('available'))}")
        for model in payload["ollama"]["models"]:
            print(f"- {model['name']}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    payload = doctor_payload()
    if args.json:
        print(dumps_json(payload))
    else:
        print(render_doctor_text(payload))
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    models = args.models.split(",") if args.models else None
    overrides = load_model_metadata_overrides(args.model_metadata_overrides)
    payload = collect_artifacts(
        args.output_dir,
        models=models,
        include_raw_model_json=not args.skip_raw_model_json,
        model_metadata_overrides=overrides,
        model_metadata_override_path=args.model_metadata_overrides,
    )
    if args.json:
        print(dumps_json(payload))
    else:
        print(render_collect_markdown(payload), end="")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    manifest = load_collect_manifest(args.manifest)
    payload = audit_collect_manifest(
        manifest,
        manifest_path=args.manifest,
        fail_on_missing_metadata=args.fail_on_missing_metadata,
        require_artifacts=not args.no_require_artifacts,
        require_apple_silicon=args.require_apple_silicon,
    )
    _maybe_write_text(args.output, dumps_json(payload))
    _maybe_write_text(args.markdown_output, render_collect_audit_text(payload) + "\n")
    if args.json:
        print(dumps_json(payload))
    else:
        print(render_collect_audit_text(payload))
    return 1 if payload.get("status") == "fail" else 0


def _cmd_capabilities(args: argparse.Namespace) -> int:
    payload = detect_runtime_capabilities().to_dict()
    if args.json:
        print(dumps_json(payload))
    else:
        print(f"Ollama: {'available' if payload['ollama']['available'] else 'missing'}")
        print(f"llama.cpp: {'available' if payload['llama_cpp']['available'] else 'missing'}")
        print(f"Ollama num_ctx option surface: {payload['supports_ollama_num_ctx']}")
        print(f"llama.cpp cache-type-k/v: {payload['supports_llama_cpp_cache_type_k']}/{payload['supports_llama_cpp_cache_type_v']}")
        for warning in payload.get("warnings", []):
            print(f"Warning: {warning}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    model = _resolve_model(args)
    hardware = _resolve_hardware(args)
    config = PlannerConfig(
        target_context=parse_context(args.target_context),
        memory_budget_bytes=parse_size(args.memory_budget),
    )
    plan = create_plan(model, hardware, config, capabilities=_resolve_capabilities(args))
    payload = {"hardware": hardware.to_dict(), "model": model.to_dict(), "plan": plan.to_dict()}
    if args.json:
        print(dumps_json(payload))
    else:
        print(render_plan_text(plan, model, hardware))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    model = _resolve_model(args)
    hardware = _resolve_hardware(args)
    config = PlannerConfig(
        target_context=parse_context(args.target_context),
        memory_budget_bytes=parse_size(args.memory_budget),
    )
    plan = create_plan(model, hardware, config, capabilities=_resolve_capabilities(args))
    command = build_run_command(args.model, plan.ollama_options, prompt=args.prompt)
    if args.json:
        print(dumps_json({"plan": plan.to_dict(), "command": command}))
        return 0
    print(_quote_command(command))
    if args.execute:
        import subprocess

        return subprocess.run(command).returncode
    return 0


def _cmd_auto(args: argparse.Namespace) -> int:
    model = _resolve_model(args)
    hardware = _resolve_hardware(args)
    config = PlannerConfig(
        target_context=parse_context(args.target_context),
        memory_budget_bytes=parse_size(args.memory_budget),
    )
    plan = create_plan(model, hardware, config, capabilities=_resolve_capabilities(args))
    command = build_run_command(args.model, plan.ollama_options, prompt=args.prompt)
    payload = {
        "hardware": hardware.to_dict(),
        "model": model.to_dict(),
        "plan": plan.to_dict(),
        "command": command,
        "next_step": "Run with --execute to call local Ollama." if not args.execute else "",
    }
    if args.json:
        print(dumps_json(payload))
        if args.execute:
            import subprocess

            return subprocess.run(command).returncode
        return 0

    print(render_plan_text(plan, model, hardware))
    print("")
    print("Recommended Ollama command:")
    print(_quote_command(command))
    if not args.execute:
        print("")
        print("Add --execute to run it through local Ollama.")
        return 0

    import subprocess

    return subprocess.run(command).returncode


def _cmd_bench(args: argparse.Namespace) -> int:
    models = args.models.split(",")
    contexts = args.contexts.split(",")
    if args.execute:
        payload = execute_bench_payload(
            models,
            contexts,
            prompt=args.prompt,
            base_url=args.ollama_url,
            timeout_seconds=args.timeout,
            num_predict=args.num_predict,
            memory_sample_interval_seconds=args.memory_sample_interval,
            include_memory_series=args.include_memory_series,
            repeats=args.repeats,
            stability_config=_resolve_stability_config(args),
            use_ollama_default_options=args.use_ollama_default_options,
        )
        written = _maybe_write_artifacts(args, payload)
        if written:
            payload["artifacts"] = written
        print(dumps_json(payload))
        return 0

    payload = dry_run_payload(models, contexts, repeats=args.repeats)
    written = _maybe_write_artifacts(args, payload)
    if written:
        payload["artifacts"] = written
    if args.json or args.dry_run:
        print(dumps_json(payload))
        return 0

    for job in payload["jobs"]:
        print(f"{job['model']} @ {job['context']}")
    return 0


def _cmd_baseline_template(args: argparse.Namespace) -> int:
    payload = write_baseline_template(
        args.output_dir,
        args.models.split(","),
        args.contexts.split(","),
        memory_budget=args.memory_budget,
        prompt=args.prompt,
        manual_context=args.manual_context,
        repeats=args.repeats,
    )
    if args.json:
        print(dumps_json(payload))
    else:
        print(f"Baseline template written to {payload['output_dir']}")
        for item in payload.get("model_directories", []):
            if isinstance(item, dict):
                print(f"- {item.get('model')}: {item.get('path')}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for input_path in args.inputs:
        rows.extend(normalize_report_rows(load_report_payload(input_path)))
    if args.output_dir:
        tables = args.tables.split(",")
        if args.table != "all":
            tables = [args.table]
        written = write_fixed_tables(
            rows,
            args.output_dir,
            prefix=args.output_prefix,
            tables=tables,
            fmt=args.format,
        )
        print(dumps_json({"artifacts": written}))
        return 0
    if args.format == "csv":
        if args.table == "all":
            print(render_report_csv(rows), end="")
        else:
            print(render_fixed_table_csv(rows, args.table), end="")
    else:
        if args.table == "all":
            print(render_report_markdown(rows))
        else:
            print(render_fixed_table_markdown(rows, args.table))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    payload = build_compare_payload(
        (parse_compare_input(item) for item in args.inputs),
        baseline_label=args.baseline_label,
    )
    if args.format == "json":
        rendered = dumps_json(payload)
    elif args.format == "csv":
        rendered = render_compare_csv(payload["rows"])
    else:
        rendered = render_compare_markdown(payload["rows"])

    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        print(dumps_json({"output": args.output}))
    else:
        print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 0


def _cmd_baseline_summary(args: argparse.Namespace) -> int:
    payload = build_baseline_summary_payload(args.machine_dir)
    if args.format == "json":
        rendered = render_baseline_summary_json(payload)
    elif args.format == "csv":
        rendered = render_baseline_summary_csv(payload)
    else:
        rendered = render_baseline_summary_markdown(payload)
    if args.output:
        _maybe_write_text(args.output, rendered + ("" if rendered.endswith("\n") else "\n"))
        print(dumps_json({"output": args.output, "rows": len(payload.get("rows", []))}))
    else:
        print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 0


def _cmd_plot_memory(args: argparse.Namespace) -> int:
    payload = load_report_payload(args.input)
    path = write_memory_svg(payload, args.output, series_index=args.series_index)
    print(dumps_json({"svg": path}))
    return 0


def _cmd_needle(args: argparse.Namespace) -> int:
    models = args.models.split(",")
    contexts = args.contexts.split(",")
    depths = args.depths.split(",")
    if args.execute:
        payload = execute_needle_payload(
            models,
            contexts,
            depths,
            base_url=args.ollama_url,
            timeout_seconds=args.timeout,
            num_predict=args.num_predict,
            repeats=args.repeats,
        )
    else:
        payload = dry_run_needle_payload(models, contexts, depths, repeats=args.repeats)

    written = _maybe_write_artifacts(args, payload)
    if written:
        payload["artifacts"] = written
    if args.json or args.dry_run or args.execute:
        print(dumps_json(payload))
    else:
        for job in payload["jobs"]:
            print(f"{job['model']} @ {job['context']} depth={job['depth']}")
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    models = args.models.split(",")
    contexts = args.contexts.split(",")
    depths = args.depths.split(",")
    if args.execute:
        payload = execute_qa_payload(
            models,
            contexts,
            dataset_path=args.dataset,
            depths=depths,
            base_url=args.ollama_url,
            timeout_seconds=args.timeout,
            num_predict=args.num_predict,
            repeats=args.repeats,
        )
    else:
        payload = dry_run_qa_payload(
            models,
            contexts,
            dataset_path=args.dataset,
            depths=depths,
            repeats=args.repeats,
        )

    written = _maybe_write_artifacts(args, payload)
    if written:
        payload["artifacts"] = written
    if args.json or args.dry_run or args.execute:
        print(dumps_json(payload))
    else:
        for job in payload["jobs"]:
            source = job.get("source", "synthetic")
            question_id = job.get("id", "")
            print(f"{job['model']} @ {job['context']} source={source} id={question_id}")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    model = _resolve_model(args)
    hardware = _resolve_hardware(args)
    payload = build_experiment_payload(
        model,
        hardware,
        args.contexts.split(","),
        memory_budget=args.memory_budget,
        execute=args.execute,
        include_bench=not args.skip_bench,
        include_needle=not args.skip_needle,
        include_qa=not args.skip_qa,
        depths=args.depths.split(","),
        qa_dataset_path=args.qa_dataset,
        prompt=args.prompt,
        base_url=args.ollama_url,
        timeout_seconds=args.timeout,
        bench_num_predict=args.bench_num_predict,
        needle_num_predict=args.needle_num_predict,
        qa_num_predict=args.qa_num_predict,
        repeats=args.repeats,
        memory_sample_interval_seconds=args.memory_sample_interval,
        include_memory_series=args.include_memory_series,
        capabilities=_resolve_capabilities(args),
        stability_config=_resolve_stability_config(args),
    )
    written = _maybe_write_artifacts(args, payload)
    if written:
        payload["artifacts"] = written
    if args.json or args.dry_run or args.execute:
        print(dumps_json(payload))
    else:
        print(f"MacKV-Opt experiment for {model.name}")
        for plan in payload["plans"]:
            print(f"- context {plan['target_context']}: {plan['status']} -> {plan['num_ctx']}")
    return 0


def _maybe_write_artifacts(args: argparse.Namespace, payload: dict[str, object]) -> dict[str, str]:
    if not getattr(args, "output_dir", None):
        return {}
    formats = str(getattr(args, "save_formats", "")).split(",")
    return write_experiment_artifacts(
        payload,
        args.output_dir,
        prefix=getattr(args, "output_prefix", "mackv-opt-bench"),
        formats=formats,
    )


def _maybe_write_text(path: str | None, text: str) -> None:
    if not path:
        return
    from pathlib import Path

    Path(path).write_text(text, encoding="utf-8")


def _resolve_model(args: argparse.Namespace) -> ModelProfile:
    profile = load_model_profile(args.model)
    if profile and not any(
        [args.model_size, args.parameters, args.hidden_size, args.layers, args.heads, args.kv_heads]
    ):
        return profile
    return ModelProfile(
        name=args.model,
        family=args.family,
        parameter_count=_parse_parameters(args.parameters),
        size_bytes=parse_size(args.model_size),
        hidden_size=args.hidden_size,
        layer_count=args.layers,
        attention_head_count=args.heads,
        kv_head_count=args.kv_heads,
        architecture=args.architecture,
    )


def _resolve_hardware(args: argparse.Namespace):
    total = parse_size(getattr(args, "hardware_memory", None))
    return get_hardware_profile(total_memory_bytes=total)


def _resolve_capabilities(args: argparse.Namespace):
    if getattr(args, "skip_capability_check", False):
        return None
    return detect_runtime_capabilities()


def _resolve_stability_config(args: argparse.Namespace) -> StabilityConfig:
    return StabilityConfig(
        max_swap_bytes=parse_size(getattr(args, "max_swap", None)),
        min_tokens_per_second=getattr(args, "min_tokens_per_second", None),
        stable_context_policy=getattr(args, "stable_context_policy", "any"),
        min_stable_fraction=getattr(args, "min_stable_fraction", 1.0),
    ).normalized()


def _parse_parameters(value: str | None) -> int | None:
    if value is None:
        return None
    from .ollama import _parse_parameters as parse_parameters

    return parse_parameters(value)


def _quote_command(command: list[str]) -> str:
    return " ".join(_quote_arg(arg) for arg in command)


def _quote_arg(arg: str) -> str:
    if not arg or any(ch.isspace() for ch in arg):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


if __name__ == "__main__":
    sys.exit(main())
