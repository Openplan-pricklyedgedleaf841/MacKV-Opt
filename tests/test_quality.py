from mackv_opt.quality import (
    NeedleJob,
    QAJob,
    build_haystack_document,
    build_needle_jobs,
    build_qa_document,
    build_synthetic_qa_jobs,
    dry_run_needle_payload,
    dry_run_qa_payload,
    execute_needle_payload,
    execute_qa_payload,
    load_qa_jobs_jsonl,
    render_needle_prompt,
    render_qa_prompt,
    run_needle_benchmark,
    run_qa_benchmark,
    summarize_qa_runs,
    summarize_needle_runs,
)


def test_build_needle_jobs_creates_deterministic_matrix():
    jobs = build_needle_jobs(["llama3.1:8b"], ["8k"], ["10%", "0.9"])

    assert len(jobs) == 2
    assert jobs[0].context == 8192
    assert jobs[0].depth == 0.1
    assert jobs[0].needle.startswith("MACKV-")
    assert jobs[0].needle != jobs[1].needle


def test_render_needle_prompt_contains_secret_once():
    job = NeedleJob("llama3.1:8b", 4096, 0.5, "MACKV-SECRET")

    document = build_haystack_document(job.context, job.depth, job.needle)
    prompt = render_needle_prompt(job)

    assert document.count("MACKV-SECRET") == 1
    assert "Question: What is the exact secret key" in prompt


def test_run_needle_benchmark_scores_found_response(monkeypatch):
    def fake_call(**kwargs):
        assert "MACKV-ABC" in kwargs["prompt"]
        return {
            "done": True,
            "response": "MACKV-ABC",
            "eval_count": 4,
            "eval_duration": 1_000_000_000,
        }

    monkeypatch.setattr("mackv_opt.quality._call_ollama_generate", fake_call)

    result = run_needle_benchmark(NeedleJob("llama3.1:8b", 4096, 0.5, "MACKV-ABC"))

    assert result.status == "ok"
    assert result.found is True
    assert result.quality_score == 1.0
    assert result.tokens_per_second == 4.0


def test_run_needle_benchmark_scores_missing_response(monkeypatch):
    monkeypatch.setattr("mackv_opt.quality._call_ollama_generate", lambda **kwargs: {"response": "wrong"})

    result = run_needle_benchmark(NeedleJob("qwen2.5:7b", 4096, 0.9, "MACKV-MISS"))

    assert result.found is False
    assert result.quality_score == 0.0


def test_needle_payloads_and_summary(monkeypatch):
    monkeypatch.setattr(
        "mackv_opt.quality.run_needle_benchmark",
        lambda job, **kwargs: type(
            "FakeResult",
            (),
            {"to_dict": lambda self: {"model": job.model, "context": job.context, "found": True}},
        )(),
    )

    dry = dry_run_needle_payload(["a"], ["8k"], ["0.5"], repeats=2)
    executed = execute_needle_payload(["a"], ["8k"], ["0.5"], repeats=2)

    assert dry["task"] == "needle"
    assert dry["planned_run_count"] == 2
    assert dry["jobs"][0]["depth"] == 0.5
    assert executed["summary"] == {"total": 2, "found": 2, "accuracy": 1.0}
    assert executed["repeat_summaries"][0]["runs"] == 2
    assert summarize_needle_runs([{"found": True}, {"found": False}])["accuracy"] == 0.5


def test_build_synthetic_qa_jobs_creates_depth_matrix():
    jobs = build_synthetic_qa_jobs(["llama3.1:8b"], ["8k"], ["10%", "90%"])

    assert len(jobs) == 2
    assert jobs[0].context == 8192
    assert jobs[0].source == "synthetic"
    assert jobs[0].depth == 0.1
    assert jobs[0].expected_answer.startswith("MACKV-QA-")
    assert jobs[0].expected_answer != jobs[1].expected_answer


def test_render_qa_prompt_contains_document_question_and_answer_once():
    document = build_qa_document(4096, 0.5, "MACKV-QA-SECRET")
    job = QAJob(
        model="llama3.1:8b",
        context=4096,
        document=document,
        question="What is the calibration code?",
        expected_answer="MACKV-QA-SECRET",
        id="q1",
    )

    prompt = render_qa_prompt(job)

    assert document.count("MACKV-QA-SECRET") == 1
    assert "Question: What is the calibration code?" in prompt
    assert "MACKV-QA-SECRET" in prompt


def test_load_qa_jobs_jsonl_expands_models_and_contexts(tmp_path):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"id":"q1","document":"The answer is Alpine-42.","question":"What is the code?","answer":"Alpine-42","source":"mini"}\n',
        encoding="utf-8",
    )

    jobs = load_qa_jobs_jsonl(str(dataset), ["a", "b"], ["8k", "16k"])

    assert len(jobs) == 4
    assert jobs[0].id == "q1"
    assert jobs[0].expected_answer == "Alpine-42"
    assert jobs[0].source == "mini"
    assert jobs[-1].model == "b"
    assert jobs[-1].context == 16384


def test_dry_run_qa_payload_omits_full_document(tmp_path):
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        '{"document":"The answer is Alpine-42.","question":"What is the code?","expected_answer":"Alpine-42"}\n',
        encoding="utf-8",
    )

    payload = dry_run_qa_payload(["llama3.1:8b"], ["8k"], dataset_path=str(dataset))

    assert payload["task"] == "qa"
    assert payload["jobs"][0]["document_bytes"] > 0
    assert payload["jobs"][0]["document_sha1"]
    assert "document" not in payload["jobs"][0]


def test_run_qa_benchmark_scores_found_response(monkeypatch):
    def fake_call(**kwargs):
        assert "The answer is Alpine-42." in kwargs["prompt"]
        return {
            "done": True,
            "response": "Alpine-42",
            "eval_count": 6,
            "eval_duration": 2_000_000_000,
        }

    monkeypatch.setattr("mackv_opt.quality._call_ollama_generate", fake_call)

    result = run_qa_benchmark(
        QAJob(
            model="llama3.1:8b",
            context=4096,
            document="The answer is Alpine-42.",
            question="What is the code?",
            expected_answer="Alpine-42",
            source="mini",
            id="q1",
        )
    )

    assert result.method == "qa"
    assert result.question_id == "q1"
    assert result.found is True
    assert result.quality_score == 1.0
    assert result.tokens_per_second == 3.0


def test_execute_qa_payload_and_summary(monkeypatch):
    monkeypatch.setattr(
        "mackv_opt.quality.run_qa_benchmark",
        lambda job, **kwargs: type(
            "FakeResult",
            (),
            {"to_dict": lambda self: {"model": job.model, "context": job.context, "found": True, "method": "qa"}},
        )(),
    )

    executed = execute_qa_payload(["a"], ["8k"], depths=["50%"], repeats=2)

    assert executed["task"] == "qa"
    assert executed["summary"] == {"total": 2, "found": 2, "accuracy": 1.0}
    assert executed["repeat_summaries"][0]["runs"] == 2
    assert summarize_qa_runs([{"found": True}, {"found": False}])["accuracy"] == 0.5
