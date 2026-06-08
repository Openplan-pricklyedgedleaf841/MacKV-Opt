from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .bench import DEFAULT_OLLAMA_BASE_URL, DEFAULT_PROMPT, dry_run_payload, execute_bench_payload
from .capabilities import RuntimeCapabilities
from .models import HardwareProfile, ModelProfile
from .planner import PlannerConfig, create_plan
from .quality import dry_run_needle_payload, dry_run_qa_payload, execute_needle_payload, execute_qa_payload
from .stability import StabilityConfig
from .units import parse_context, parse_size


def build_experiment_payload(
    model: ModelProfile,
    hardware: HardwareProfile,
    contexts: Iterable[str | int],
    *,
    memory_budget: str | int | None = None,
    execute: bool = False,
    include_bench: bool = True,
    include_needle: bool = True,
    include_qa: bool = True,
    depths: Iterable[str | float] = ("10%", "50%", "90%"),
    qa_dataset_path: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    bench_num_predict: int | None = 128,
    needle_num_predict: int | None = 64,
    qa_num_predict: int | None = 96,
    repeats: int = 1,
    memory_sample_interval_seconds: float = 0.5,
    include_memory_series: bool = False,
    capabilities: RuntimeCapabilities | dict[str, object] | None = None,
    stability_config: StabilityConfig | None = None,
) -> dict[str, object]:
    parsed_contexts = [parse_context(context) for context in contexts]
    budget_bytes = parse_size(memory_budget)
    started_at = _utc_now()
    plans = [
        create_plan(
            model,
            hardware,
            PlannerConfig(target_context=context, memory_budget_bytes=budget_bytes),
            capabilities=capabilities,
        ).to_dict()
        for context in parsed_contexts
    ]

    payload: dict[str, object] = {
        "task": "experiment",
        "dry_run": not execute,
        "started_at": started_at,
        "ended_at": None,
        "hardware": hardware.to_dict(),
        "model": model.to_dict(),
        "contexts": parsed_contexts,
        "memory_budget_bytes": budget_bytes,
        "plans": plans,
        "bench": None,
        "needle": None,
        "qa": None,
    }

    model_names = [model.name]
    if include_bench:
        if execute:
            payload["bench"] = execute_bench_payload(
                model_names,
                parsed_contexts,
                prompt=prompt,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                num_predict=bench_num_predict,
                memory_sample_interval_seconds=memory_sample_interval_seconds,
                include_memory_series=include_memory_series,
                repeats=repeats,
                stability_config=stability_config,
            )
        else:
            payload["bench"] = dry_run_payload(model_names, parsed_contexts, repeats=repeats)

    if include_needle:
        if execute:
            payload["needle"] = execute_needle_payload(
                model_names,
                parsed_contexts,
                depths,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                num_predict=needle_num_predict,
                repeats=repeats,
            )
        else:
            payload["needle"] = dry_run_needle_payload(model_names, parsed_contexts, depths, repeats=repeats)

    if include_qa:
        if execute:
            payload["qa"] = execute_qa_payload(
                model_names,
                parsed_contexts,
                dataset_path=qa_dataset_path,
                depths=depths,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                num_predict=qa_num_predict,
                repeats=repeats,
            )
        else:
            payload["qa"] = dry_run_qa_payload(
                model_names,
                parsed_contexts,
                dataset_path=qa_dataset_path,
                depths=depths,
                repeats=repeats,
            )

    payload["ended_at"] = _utc_now()
    payload["summary"] = summarize_experiment(payload)
    return payload


def summarize_experiment(payload: dict[str, object]) -> dict[str, object]:
    plans = payload.get("plans") if isinstance(payload.get("plans"), list) else []
    bench = payload.get("bench") if isinstance(payload.get("bench"), dict) else {}
    needle = payload.get("needle") if isinstance(payload.get("needle"), dict) else {}
    qa = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}
    bench_runs = bench.get("runs") if isinstance(bench.get("runs"), list) else []
    bench_repeat_summaries = bench.get("repeat_summaries") if isinstance(bench.get("repeat_summaries"), list) else []
    stability_summary = bench.get("stability_summary") if isinstance(bench.get("stability_summary"), dict) else {}
    needle_summary = needle.get("summary") if isinstance(needle.get("summary"), dict) else {}
    needle_repeat_summaries = needle.get("repeat_summaries") if isinstance(needle.get("repeat_summaries"), list) else []
    qa_summary = qa.get("summary") if isinstance(qa.get("summary"), dict) else {}
    qa_repeat_summaries = qa.get("repeat_summaries") if isinstance(qa.get("repeat_summaries"), list) else []
    return {
        "plan_count": len(plans),
        "bench_run_count": len(bench_runs),
        "bench_repeat_summary_count": len(bench_repeat_summaries),
        "max_stable_context_by_model": stability_summary.get("max_stable_context_by_model"),
        "needle_accuracy": needle_summary.get("accuracy"),
        "needle_repeat_summary_count": len(needle_repeat_summaries),
        "qa_accuracy": qa_summary.get("accuracy"),
        "qa_repeat_summary_count": len(qa_repeat_summaries),
        "max_planned_context": max([plan.get("num_ctx", 0) for plan in plans if isinstance(plan, dict)] or [0]),
        "all_plans_fit": all(plan.get("status") == "fits" for plan in plans if isinstance(plan, dict)),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
