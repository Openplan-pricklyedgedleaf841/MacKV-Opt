import json

from mackv_opt.bench import (
    BenchJob,
    MemorySampler,
    _parse_darwin_vm_stat,
    _call_ollama_generate,
    dry_run_payload,
    execute_bench_payload,
    sample_ollama_process_memory,
    sample_memory_state,
    run_ollama_generate_benchmark,
)


def test_run_ollama_generate_benchmark_calculates_ollama_metrics(monkeypatch):
    def fake_call(**kwargs):
        assert kwargs["model"] == "llama3.1:8b"
        assert kwargs["context"] == 8192
        assert kwargs["apply_num_ctx"] is True
        return {
            "done": True,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 50,
            "eval_duration": 1_000_000_000,
            "total_duration": 4_000_000_000,
            "load_duration": 500_000_000,
            "done_reason": "stop",
        }

    monkeypatch.setattr("mackv_opt.bench._call_ollama_generate", fake_call)
    monkeypatch.setattr(
        "mackv_opt.bench.sample_memory_state",
        lambda: {"memory_pressure": "normal", "swap_bytes": 1024},
    )
    monkeypatch.setattr("mackv_opt.bench.sample_ollama_process_memory", lambda: None)

    result = run_ollama_generate_benchmark(BenchJob("llama3.1:8b", 8192))

    assert result.status == "ok"
    assert result.tokens_per_second == 50.0
    assert result.prompt_tokens_per_second == 50.0
    assert result.first_token_latency_ms == 2500.0
    assert result.peak_memory_bytes is None
    assert result.memory_pressure == "normal"
    assert result.swap_bytes == 0
    assert result.memory_samples >= 2
    assert result.memory_series is None
    assert result.done_reason == "stop"


def test_run_ollama_generate_benchmark_records_structured_errors(monkeypatch):
    def fake_call(**kwargs):
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr("mackv_opt.bench._call_ollama_generate", fake_call)
    monkeypatch.setattr(
        "mackv_opt.bench.sample_memory_state",
        lambda: {"memory_pressure": "unknown", "swap_bytes": None},
    )
    monkeypatch.setattr("mackv_opt.bench.sample_ollama_process_memory", lambda: None)

    result = run_ollama_generate_benchmark(BenchJob("qwen2.5:7b", 16384))

    assert result.status == "error"
    assert result.error == "RuntimeError: Ollama is not running"
    assert result.model == "qwen2.5:7b"
    assert result.context == 16384


def test_execute_bench_payload_includes_runs_and_metrics(monkeypatch):
    monkeypatch.setattr(
        "mackv_opt.bench.run_ollama_generate_benchmark",
        lambda job, **kwargs: type(
            "FakeResult",
            (),
            {
                "to_dict": lambda self: {
                    "model": job.model,
                    "context": job.context,
                    "status": "ok",
                }
            },
        )(),
    )

    payload = execute_bench_payload(["a", "b"], ["8k"])

    assert payload["dry_run"] is False
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["model"] == "a"
    assert payload["runs"][0]["context"] == 8192
    assert payload["runs"][0]["status"] == "ok"
    assert payload["runs"][0]["repeat_index"] == 0
    assert payload["runs"][0]["repeat_count"] == 1
    assert payload["runs"][0]["stable"] is True
    assert payload["runs"][0]["stability_reason"] == "stable"
    assert payload["repeat_summaries"][0]["runs"] == 1
    assert payload["stability_config"]["max_swap_bytes"] == 512 * 1024**2
    assert payload["stability_config"]["stable_context_policy"] == "any"
    assert payload["stability_summary"]["max_stable_context_by_model"] == {"a": 8192, "b": 8192}
    assert "tokens_per_second" in payload["metrics"]
    assert "stable" in payload["metrics"]


def test_execute_bench_payload_can_use_ollama_default_options(monkeypatch):
    seen = {}

    def fake_run(job, **kwargs):
        seen["apply_num_ctx"] = kwargs["apply_num_ctx"]
        return type(
            "FakeResult",
            (),
            {
                "to_dict": lambda self: {
                    "model": job.model,
                    "context": job.context,
                    "status": "ok",
                    "ollama_default_options": True,
                }
            },
        )()

    monkeypatch.setattr("mackv_opt.bench.run_ollama_generate_benchmark", fake_run)

    payload = execute_bench_payload(["a"], ["8k"], use_ollama_default_options=True)

    assert seen["apply_num_ctx"] is False
    assert payload["runs"][0]["ollama_default_options"] is True


def test_call_ollama_generate_can_omit_num_ctx(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"done": true}'

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("mackv_opt.bench.urllib.request.urlopen", fake_urlopen)

    payload = _call_ollama_generate(
        model="llama3.1:8b",
        context=8192,
        prompt="hello",
        base_url="http://localhost:11434",
        timeout_seconds=3,
        num_predict=16,
        apply_num_ctx=False,
    )

    assert payload["done"] is True
    assert captured["body"]["options"] == {"num_predict": 16}


def test_execute_bench_payload_uses_strict_stability_policy(monkeypatch):
    statuses = iter(["ok", "error", "ok", "ok"])

    def fake_run(job, **kwargs):
        status = next(statuses)
        return type(
            "FakeResult",
            (),
            {
                "to_dict": lambda self: {
                    "model": job.model,
                    "context": job.context,
                    "method": "ollama-api",
                    "status": status,
                }
            },
        )()

    monkeypatch.setattr("mackv_opt.bench.run_ollama_generate_benchmark", fake_run)

    from mackv_opt.stability import StabilityConfig

    payload = execute_bench_payload(
        ["a"],
        ["8k", "16k"],
        repeats=2,
        stability_config=StabilityConfig(stable_context_policy="all"),
    )

    assert payload["stability_config"]["stable_context_policy"] == "all"
    assert payload["stability_summary"]["max_stable_context_by_model"] == {"a": 16384}
    contexts = payload["stability_summary"]["models"][0]["contexts"]
    assert contexts["8192"]["context_stable"] is False
    assert contexts["16384"]["context_stable"] is True


def test_bench_payloads_include_repeat_counts(monkeypatch):
    monkeypatch.setattr(
        "mackv_opt.bench.run_ollama_generate_benchmark",
        lambda job, **kwargs: type(
            "FakeResult",
            (),
            {
                "to_dict": lambda self: {
                    "model": job.model,
                    "context": job.context,
                    "method": "ollama-api",
                    "status": "ok",
                    "tokens_per_second": 10.0,
                }
            },
        )(),
    )

    dry = dry_run_payload(["a"], ["8k"], repeats=3)
    executed = execute_bench_payload(["a"], ["8k"], repeats=3)

    assert dry["planned_run_count"] == 3
    assert dry["repeats"] == 3
    assert len(executed["runs"]) == 3
    assert executed["runs"][2]["repeat_index"] == 2
    assert executed["repeat_summaries"][0]["tokens_per_second_mean"] == 10.0


def test_run_ollama_generate_benchmark_records_peak_process_memory(monkeypatch):
    samples = iter([1000, 1500, 1250])

    monkeypatch.setattr("mackv_opt.bench._call_ollama_generate", lambda **kwargs: {"done": True})
    monkeypatch.setattr(
        "mackv_opt.bench.sample_memory_state",
        lambda: {"memory_pressure": "normal", "swap_bytes": 0},
    )
    monkeypatch.setattr("mackv_opt.bench.sample_ollama_process_memory", lambda: next(samples))

    result = run_ollama_generate_benchmark(BenchJob("llama3.1:8b", 8192))

    assert result.peak_memory_bytes == 1500


def test_sample_ollama_process_memory_uses_platform_specific_sampler(monkeypatch):
    monkeypatch.setattr("mackv_opt.bench.platform.system", lambda: "Windows")
    monkeypatch.setattr("mackv_opt.bench._windows_process_rss_bytes", lambda name: 123)

    assert sample_ollama_process_memory() == 123


def test_memory_sampler_summary_tracks_peak_pressure_and_swap(monkeypatch):
    memory_samples = iter(
        [
            {
                "memory_pressure": "normal",
                "swap_bytes": 100,
                "page_size_bytes": 4096,
                "pageins": 10,
                "pageouts": 7,
                "swapins": 3,
                "swapouts": 4,
            },
            {
                "memory_pressure": "critical",
                "swap_bytes": 180,
                "page_size_bytes": 4096,
                "pageins": 16,
                "pageouts": 9,
                "swapins": 5,
                "swapouts": 9,
            },
        ]
    )
    process_samples = iter([1024, 4096])

    monkeypatch.setattr("mackv_opt.bench.sample_memory_state", lambda: next(memory_samples))
    monkeypatch.setattr("mackv_opt.bench.sample_ollama_process_memory", lambda: next(process_samples))

    sampler = MemorySampler()
    sampler._record_sample()
    sampler._record_sample()
    summary = sampler.summary(include_series=True)

    assert summary["peak_memory_bytes"] == 4096
    assert summary["memory_pressure"] == "critical"
    assert summary["swap_bytes"] == 80
    assert summary["pageins_delta"] == 6
    assert summary["pageout_delta"] == 2
    assert summary["pageout_bytes_delta"] == 8192
    assert summary["swapins_delta"] == 2
    assert summary["swapouts_delta"] == 5
    assert summary["memory_samples"] == 2
    assert len(summary["memory_series"]) == 2
    assert summary["memory_series"][0]["pageouts"] == 7


def test_parse_darwin_vm_stat_extracts_page_and_swap_counters():
    text = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10.
Pages active:                             20.
Pages inactive:                           30.
Pages speculative:                        40.
Pages wired down:                         50.
Pages stored in compressor:               60.
Pages occupied by compressor:             70.
Pageins:                                  80.
Pageouts:                                 90.
Swapins:                                  100.
Swapouts:                                 110.
"""

    stats = _parse_darwin_vm_stat(text)

    assert stats["page_size_bytes"] == 16384
    assert stats["pages_free"] == 10
    assert stats["pages_active"] == 20
    assert stats["pages_inactive"] == 30
    assert stats["pages_speculative"] == 40
    assert stats["pages_wired"] == 50
    assert stats["pages_stored_in_compressor"] == 60
    assert stats["pages_occupied_by_compressor"] == 70
    assert stats["pageins"] == 80
    assert stats["pageouts"] == 90
    assert stats["swapins"] == 100
    assert stats["swapouts"] == 110


def test_sample_memory_state_includes_darwin_vm_stat(monkeypatch):
    monkeypatch.setattr("mackv_opt.bench.platform.system", lambda: "Darwin")
    monkeypatch.setattr("mackv_opt.bench._darwin_memory_pressure", lambda: "normal")
    monkeypatch.setattr("mackv_opt.bench._darwin_swap_used_bytes", lambda: 2048)
    monkeypatch.setattr(
        "mackv_opt.bench._darwin_vm_stat",
        lambda: {"page_size_bytes": 16384, "pageouts": 9, "swapouts": 3},
    )

    state = sample_memory_state()

    assert state["memory_pressure"] == "normal"
    assert state["swap_bytes"] == 2048
    assert state["page_size_bytes"] == 16384
    assert state["pageouts"] == 9
    assert state["swapouts"] == 3


def test_run_ollama_generate_benchmark_can_include_memory_series(monkeypatch):
    process_samples = iter([1000, 2000, 1500])

    monkeypatch.setattr("mackv_opt.bench._call_ollama_generate", lambda **kwargs: {"done": True})
    monkeypatch.setattr(
        "mackv_opt.bench.sample_memory_state",
        lambda: {"memory_pressure": "normal", "swap_bytes": 0},
    )
    monkeypatch.setattr("mackv_opt.bench.sample_ollama_process_memory", lambda: next(process_samples))

    result = run_ollama_generate_benchmark(
        BenchJob("llama3.1:8b", 8192),
        include_memory_series=True,
    )

    assert result.peak_memory_bytes == 2000
    assert result.memory_series is not None
    assert len(result.memory_series) >= 2
