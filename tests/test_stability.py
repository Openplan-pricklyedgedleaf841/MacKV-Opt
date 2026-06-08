from mackv_opt.stability import (
    StabilityConfig,
    annotate_runs_with_stability,
    classify_run_stability,
    summarize_stability,
)


def test_classify_run_stability_marks_ok_run_stable():
    result = classify_run_stability(
        {
            "status": "ok",
            "memory_pressure": "normal",
            "swap_bytes": 0,
            "tokens_per_second": 12.5,
        },
        StabilityConfig(max_swap_bytes=1024, min_tokens_per_second=1.0),
    )

    assert result == {"stable": True, "stability_reason": "stable"}


def test_classify_run_stability_marks_status_error_unstable():
    result = classify_run_stability({"status": "error"}, StabilityConfig())

    assert result["stable"] is False
    assert "status=error" in result["stability_reason"]


def test_classify_run_stability_marks_critical_pressure_unstable():
    result = classify_run_stability(
        {"status": "ok", "memory_pressure": "critical"},
        StabilityConfig(),
    )

    assert result["stable"] is False
    assert "memory_pressure=critical" in result["stability_reason"]


def test_classify_run_stability_marks_swap_over_threshold_unstable():
    result = classify_run_stability(
        {"status": "ok", "swap_bytes": 2048},
        StabilityConfig(max_swap_bytes=1024),
    )

    assert result["stable"] is False
    assert "swap_bytes>1024" in result["stability_reason"]


def test_classify_run_stability_marks_low_tokens_per_second_unstable():
    result = classify_run_stability(
        {"status": "ok", "tokens_per_second": 0.5},
        StabilityConfig(min_tokens_per_second=1.0),
    )

    assert result["stable"] is False
    assert "tokens_per_second<1.0" in result["stability_reason"]


def test_annotate_runs_with_stability_copies_records():
    original = {"model": "m", "context": 8192, "status": "ok"}

    annotated = annotate_runs_with_stability([original])

    assert annotated[0]["stable"] is True
    assert "stable" not in original


def test_summarize_stability_tracks_max_stable_context_by_model():
    summary = summarize_stability(
        [
            {"model": "m", "context": 8192, "stable": True},
            {"model": "m", "context": 16384, "stable": False, "stability_reason": "status=error"},
            {"model": "m", "context": 32768, "stable": True},
            {"model": "n", "context": 4096, "stable": False, "stability_reason": "memory_pressure=critical"},
        ]
    )

    assert summary["max_stable_context_by_model"] == {"m": 32768, "n": 0}
    assert summary["stable_context_policy"] == "any"
    assert summary["min_stable_fraction"] == 1.0
    models = {record["model"]: record for record in summary["models"]}
    assert models["m"]["stable_runs"] == 2
    assert models["m"]["unstable_runs"] == 1
    assert models["m"]["contexts"]["16384"]["runs"] == 1
    assert models["m"]["contexts"]["16384"]["context_stable"] is False
    assert models["m"]["contexts"]["16384"]["reasons"]["status=error"] == 1
    assert models["n"]["max_stable_context"] == 0


def test_summarize_stability_can_require_all_repeats_for_context():
    runs = [
        {"model": "m", "context": 8192, "stable": True},
        {"model": "m", "context": 8192, "stable": False, "stability_reason": "status=error"},
        {"model": "m", "context": 16384, "stable": True},
        {"model": "m", "context": 16384, "stable": True},
    ]

    summary = summarize_stability(runs, StabilityConfig(stable_context_policy="all"))

    assert summary["stable_context_policy"] == "all"
    assert summary["min_stable_fraction"] == 1.0
    assert summary["max_stable_context_by_model"] == {"m": 16384}
    contexts = summary["models"][0]["contexts"]
    assert contexts["8192"]["stable_fraction"] == 0.5
    assert contexts["8192"]["context_stable"] is False
    assert contexts["16384"]["context_stable"] is True


def test_summarize_stability_can_require_stable_fraction():
    runs = [
        {"model": "m", "context": 8192, "stable": True},
        {"model": "m", "context": 8192, "stable": False, "stability_reason": "status=error"},
        {"model": "m", "context": 8192, "stable": True},
        {"model": "m", "context": 16384, "stable": True},
        {"model": "m", "context": 16384, "stable": False, "stability_reason": "status=error"},
        {"model": "m", "context": 16384, "stable": False, "stability_reason": "status=error"},
    ]

    summary = summarize_stability(
        runs,
        StabilityConfig(stable_context_policy="fraction", min_stable_fraction=0.67),
    )

    assert summary["stable_context_policy"] == "fraction"
    assert summary["min_stable_fraction"] == 0.67
    assert summary["max_stable_context_by_model"] == {"m": 0}
    contexts = summary["models"][0]["contexts"]
    assert contexts["8192"]["stable_fraction"] == 0.666667
    assert contexts["8192"]["context_stable"] is False


def test_summarize_stability_fraction_policy_accepts_threshold():
    summary = summarize_stability(
        [
            {"model": "m", "context": 8192, "stable": True},
            {"model": "m", "context": 8192, "stable": False, "stability_reason": "status=error"},
            {"model": "m", "context": 8192, "stable": True},
        ],
        StabilityConfig(stable_context_policy="fraction", min_stable_fraction=0.66),
    )

    assert summary["max_stable_context_by_model"] == {"m": 8192}
    assert summary["models"][0]["contexts"]["8192"]["context_stable"] is True


def test_stability_config_rejects_unknown_context_policy():
    try:
        StabilityConfig(stable_context_policy="sometimes").normalized()
    except ValueError as exc:
        assert "Invalid stable context policy" in str(exc)
    else:
        raise AssertionError("expected ValueError")
