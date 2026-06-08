from mackv_opt.experiment import build_experiment_payload, summarize_experiment
from mackv_opt.models import HardwareProfile, ModelProfile
from mackv_opt.stability import StabilityConfig


def hardware() -> HardwareProfile:
    return HardwareProfile(
        platform="Darwin",
        machine="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=int(16 * 1024**3 * 0.72),
        pressure="normal",
    )


def model() -> ModelProfile:
    return ModelProfile(
        name="llama3.1:8b",
        size_bytes=int(4.8 * 1024**3),
        hidden_size=4096,
        layer_count=32,
        attention_head_count=32,
        kv_head_count=8,
    )


def test_build_experiment_payload_dry_run_includes_plans_bench_needle_and_qa():
    payload = build_experiment_payload(
        model(),
        hardware(),
        ["8k", "16k"],
        memory_budget="12GiB",
        execute=False,
        depths=["50%"],
    )

    assert payload["task"] == "experiment"
    assert payload["dry_run"] is True
    assert len(payload["plans"]) == 2
    assert len(payload["bench"]["jobs"]) == 2
    assert len(payload["needle"]["jobs"]) == 2
    assert len(payload["qa"]["jobs"]) == 2
    assert payload["summary"]["plan_count"] == 2


def test_build_experiment_payload_execute_uses_bench_and_needle_runners(monkeypatch):
    seen = {}

    def fake_bench(*args, **kwargs):
        seen["memory_sample_interval_seconds"] = kwargs["memory_sample_interval_seconds"]
        seen["include_memory_series"] = kwargs["include_memory_series"]
        seen["bench_repeats"] = kwargs["repeats"]
        seen["stability_config"] = kwargs["stability_config"]
        return {
            "runs": [{"model": "llama3.1:8b", "context": 8192, "status": "ok"}],
            "repeat_summaries": [{"model": "llama3.1:8b", "context": 8192, "runs": 2}],
            "stability_summary": {"max_stable_context_by_model": {"llama3.1:8b": 8192}},
        }

    monkeypatch.setattr(
        "mackv_opt.experiment.execute_bench_payload",
        fake_bench,
    )
    monkeypatch.setattr(
        "mackv_opt.experiment.execute_needle_payload",
        lambda *args, **kwargs: {
            "runs": [{"model": "llama3.1:8b", "context": 8192, "found": True}],
            "repeat_summaries": [{"model": "llama3.1:8b", "context": 8192, "runs": 2}],
            "summary": {"total": 1, "found": 1, "accuracy": 1.0},
        },
    )
    monkeypatch.setattr(
        "mackv_opt.experiment.execute_qa_payload",
        lambda *args, **kwargs: {
            "runs": [{"model": "llama3.1:8b", "context": 8192, "found": False, "method": "qa"}],
            "repeat_summaries": [{"model": "llama3.1:8b", "context": 8192, "runs": 2}],
            "summary": {"total": 1, "found": 0, "accuracy": 0.0},
        },
    )

    payload = build_experiment_payload(
        model(),
        hardware(),
        ["8k"],
        execute=True,
        memory_sample_interval_seconds=0.25,
        include_memory_series=True,
        repeats=2,
        stability_config=StabilityConfig(max_swap_bytes=0, min_tokens_per_second=2.0),
    )

    assert payload["dry_run"] is False
    assert payload["summary"]["bench_run_count"] == 1
    assert payload["summary"]["bench_repeat_summary_count"] == 1
    assert payload["summary"]["needle_repeat_summary_count"] == 1
    assert payload["summary"]["qa_repeat_summary_count"] == 1
    assert payload["summary"]["needle_accuracy"] == 1.0
    assert payload["summary"]["qa_accuracy"] == 0.0
    assert seen["bench_repeats"] == 2
    assert seen["memory_sample_interval_seconds"] == 0.25
    assert seen["include_memory_series"] is True
    assert seen["stability_config"].max_swap_bytes == 0
    assert seen["stability_config"].min_tokens_per_second == 2.0
    assert payload["summary"]["max_stable_context_by_model"] == {"llama3.1:8b": 8192}


def test_build_experiment_payload_can_skip_qa():
    payload = build_experiment_payload(
        model(),
        hardware(),
        ["8k"],
        execute=False,
        include_qa=False,
    )

    assert payload["qa"] is None
    assert payload["summary"]["qa_accuracy"] is None


def test_summarize_experiment_handles_missing_sections():
    summary = summarize_experiment({"plans": [{"status": "fits", "num_ctx": 8192}]})

    assert summary["max_planned_context"] == 8192
    assert summary["all_plans_fit"] is True
