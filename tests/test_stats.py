from mackv_opt.stats import repeat_count, summarize_repeated_runs


def test_repeat_count_clamps_invalid_values_to_one():
    assert repeat_count(None) == 1
    assert repeat_count(0) == 1
    assert repeat_count("3") == 3
    assert repeat_count("bad") == 1


def test_summarize_repeated_runs_groups_and_calculates_stats():
    summaries = summarize_repeated_runs(
        [
            {"model": "m", "context": 8192, "method": "ollama-api", "status": "ok", "tokens_per_second": 10.0},
            {"model": "m", "context": 8192, "method": "ollama-api", "status": "ok", "tokens_per_second": 14.0},
            {"model": "m", "context": 8192, "method": "ollama-api", "status": "error"},
        ]
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["runs"] == 3
    assert summary["ok_runs"] == 2
    assert summary["error_runs"] == 1
    assert summary["success_rate"] == 0.666667
    assert summary["tokens_per_second_mean"] == 12.0
    assert summary["tokens_per_second_stdev"] == 2.828427
