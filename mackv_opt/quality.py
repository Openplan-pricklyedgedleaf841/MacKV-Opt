from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .bench import DEFAULT_OLLAMA_BASE_URL, _call_ollama_generate, _int_or_none, _tokens_per_second
from .stats import repeat_count, repeat_runs, summarize_repeated_runs
from .units import parse_context

FILLER_SENTENCE = (
    "This background sentence discusses local inference scheduling, memory budgets, "
    "and document analysis without containing the secret key."
)

QA_FILLER_SENTENCE = (
    "This paragraph describes benchmark setup, local model behavior, and memory "
    "planning without containing the requested answer."
)

DEFAULT_QA_DEPTHS = ("10%", "50%", "90%")


@dataclass(frozen=True)
class NeedleJob:
    model: str
    context: int
    depth: float
    needle: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class NeedleRunResult:
    model: str
    context: int
    depth: float
    method: str
    status: str
    needle: str
    found: bool
    quality_score: float
    response_excerpt: str
    started_at: str
    ended_at: str
    wall_time_seconds: float
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    tokens_per_second: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class QAJob:
    model: str
    context: int
    document: str
    question: str
    expected_answer: str
    source: str = "synthetic"
    id: str = ""
    depth: float | None = None

    def to_dict(self, *, include_document: bool = True) -> dict[str, float | int | str | None]:
        payload = asdict(self)
        if include_document:
            return payload
        document = payload.pop("document")
        payload["document_bytes"] = len(str(document).encode("utf-8"))
        payload["document_sha1"] = hashlib.sha1(str(document).encode("utf-8")).hexdigest()
        payload["document_excerpt"] = str(document)[:240]
        return payload


@dataclass(frozen=True)
class QARunResult:
    model: str
    context: int
    question_id: str
    source: str
    depth: float | None
    method: str
    status: str
    question: str
    expected_answer: str
    found: bool
    quality_score: float
    response_excerpt: str
    started_at: str
    ended_at: str
    wall_time_seconds: float
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    tokens_per_second: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


def build_needle_jobs(
    models: Iterable[str],
    contexts: Iterable[str | int],
    depths: Iterable[str | float],
    *,
    needle_prefix: str = "MACKV",
) -> list[NeedleJob]:
    clean_models = [model.strip() for model in models if model and model.strip()]
    clean_contexts = [parse_context(context) for context in contexts]
    clean_depths = [_parse_depth(depth) for depth in depths]
    jobs: list[NeedleJob] = []
    for model in clean_models:
        for context in clean_contexts:
            for depth in clean_depths:
                jobs.append(NeedleJob(model=model, context=context, depth=depth, needle=_needle(model, context, depth, needle_prefix)))
    return jobs


def build_synthetic_qa_jobs(
    models: Iterable[str],
    contexts: Iterable[str | int],
    depths: Iterable[str | float] = DEFAULT_QA_DEPTHS,
    *,
    answer_prefix: str = "MACKV-QA",
) -> list[QAJob]:
    clean_models = [model.strip() for model in models if model and model.strip()]
    clean_contexts = [parse_context(context) for context in contexts]
    clean_depths = [_parse_depth(depth) for depth in depths]
    jobs: list[QAJob] = []
    for model in clean_models:
        for context in clean_contexts:
            for depth in clean_depths:
                answer = _qa_answer(model, context, depth, answer_prefix)
                question_id = f"synthetic-{_slug_model(model)}-{context}-{int(depth * 100)}"
                jobs.append(
                    QAJob(
                        model=model,
                        context=context,
                        document=build_qa_document(context, depth, answer),
                        question="What is the exact MacKV-Opt calibration code in the document?",
                        expected_answer=answer,
                        source="synthetic",
                        id=question_id,
                        depth=depth,
                    )
                )
    return jobs


def load_qa_jobs_jsonl(
    path: str,
    models: Iterable[str],
    contexts: Iterable[str | int],
) -> list[QAJob]:
    clean_models = [model.strip() for model in models if model and model.strip()]
    clean_contexts = [parse_context(context) for context in contexts]
    jobs: list[QAJob] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc.msg}") from exc
            document = _required_text(record, "document", path, line_number)
            question = _required_text(record, "question", path, line_number)
            expected = _optional_text(record, "expected_answer") or _optional_text(record, "answer")
            if not expected:
                raise ValueError(f"{path}:{line_number}: missing expected_answer or answer")
            source = _optional_text(record, "source") or Path(path).name
            record_id = _optional_text(record, "id") or _optional_text(record, "question_id") or f"jsonl-{line_number}"
            depth = _optional_depth(record.get("depth"))
            for model in clean_models:
                for context in clean_contexts:
                    jobs.append(
                        QAJob(
                            model=model,
                            context=context,
                            document=document,
                            question=question,
                            expected_answer=expected,
                            source=source,
                            id=record_id,
                            depth=depth,
                        )
                    )
    return jobs


def dry_run_qa_payload(
    models: Iterable[str],
    contexts: Iterable[str | int],
    *,
    dataset_path: str | None = None,
    depths: Iterable[str | float] = DEFAULT_QA_DEPTHS,
    repeats: int = 1,
) -> dict[str, object]:
    jobs = _qa_jobs(models, contexts, dataset_path=dataset_path, depths=depths)
    return {
        "dry_run": True,
        "task": "qa",
        "dataset": dataset_path or "synthetic",
        "repeats": repeat_count(repeats),
        "planned_run_count": len(jobs) * repeat_count(repeats),
        "jobs": [job.to_dict(include_document=False) for job in jobs],
        "metrics": ["found", "quality_score", "tokens_per_second", "wall_time_seconds"],
    }


def execute_qa_payload(
    models: Iterable[str],
    contexts: Iterable[str | int],
    *,
    dataset_path: str | None = None,
    depths: Iterable[str | float] = DEFAULT_QA_DEPTHS,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 96,
    repeats: int = 1,
) -> dict[str, object]:
    jobs = _qa_jobs(models, contexts, dataset_path=dataset_path, depths=depths)
    started_at = _utc_now()
    runs = repeat_runs(
        jobs,
        repeat_count(repeats),
        lambda job: run_qa_benchmark(
            job,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        ),
    )
    return {
        "dry_run": False,
        "task": "qa",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "base_url": base_url,
        "dataset": dataset_path or "synthetic",
        "repeats": repeat_count(repeats),
        "runs": runs,
        "repeat_summaries": summarize_repeated_runs(
            runs,
            group_keys=("model", "context", "method", "source", "question_id", "depth"),
        ),
        "metrics": ["found", "quality_score", "eval_count", "eval_duration_ns", "tokens_per_second", "wall_time_seconds"],
        "summary": summarize_qa_runs(runs),
    }


def dry_run_needle_payload(
    models: Iterable[str],
    contexts: Iterable[str | int],
    depths: Iterable[str | float],
    *,
    repeats: int = 1,
) -> dict[str, object]:
    jobs = build_needle_jobs(models, contexts, depths)
    return {
        "dry_run": True,
        "task": "needle",
        "repeats": repeat_count(repeats),
        "planned_run_count": len(jobs) * repeat_count(repeats),
        "jobs": [job.to_dict() for job in jobs],
        "metrics": ["found", "quality_score", "tokens_per_second", "wall_time_seconds"],
    }


def execute_needle_payload(
    models: Iterable[str],
    contexts: Iterable[str | int],
    depths: Iterable[str | float],
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 64,
    repeats: int = 1,
) -> dict[str, object]:
    jobs = build_needle_jobs(models, contexts, depths)
    started_at = _utc_now()
    runs = repeat_runs(
        jobs,
        repeat_count(repeats),
        lambda job: run_needle_benchmark(
            job,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        ),
    )
    return {
        "dry_run": False,
        "task": "needle",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "base_url": base_url,
        "repeats": repeat_count(repeats),
        "runs": runs,
        "repeat_summaries": summarize_repeated_runs(
            runs,
            group_keys=("model", "context", "method", "depth"),
        ),
        "metrics": ["found", "quality_score", "eval_count", "eval_duration_ns", "tokens_per_second", "wall_time_seconds"],
        "summary": summarize_needle_runs(runs),
    }


def run_needle_benchmark(
    job: NeedleJob,
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 64,
) -> NeedleRunResult:
    started_at = _utc_now()
    wall_start = time.perf_counter()
    try:
        payload = _call_ollama_generate(
            model=job.model,
            context=job.context,
            prompt=render_needle_prompt(job),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        )
        response = str(payload.get("response") or "")
        status = "ok" if payload.get("done", True) else "incomplete"
        error = None
    except Exception as exc:
        payload = {}
        response = ""
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - wall_start
    ended_at = _utc_now()
    eval_count = _int_or_none(payload.get("eval_count"))
    eval_duration = _int_or_none(payload.get("eval_duration"))
    found = job.needle.lower() in response.lower()
    return NeedleRunResult(
        model=job.model,
        context=job.context,
        depth=job.depth,
        method="needle",
        status=status,
        needle=job.needle,
        found=found,
        quality_score=1.0 if found else 0.0,
        response_excerpt=response[:240],
        started_at=started_at,
        ended_at=ended_at,
        wall_time_seconds=round(wall_seconds, 6),
        eval_count=eval_count,
        eval_duration_ns=eval_duration,
        tokens_per_second=_tokens_per_second(eval_count, eval_duration),
        error=error,
    )


def run_qa_benchmark(
    job: QAJob,
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: float = 300.0,
    num_predict: int | None = 96,
) -> QARunResult:
    started_at = _utc_now()
    wall_start = time.perf_counter()
    try:
        payload = _call_ollama_generate(
            model=job.model,
            context=job.context,
            prompt=render_qa_prompt(job),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        )
        response = str(payload.get("response") or "")
        status = "ok" if payload.get("done", True) else "incomplete"
        error = None
    except Exception as exc:
        payload = {}
        response = ""
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - wall_start
    ended_at = _utc_now()
    eval_count = _int_or_none(payload.get("eval_count"))
    eval_duration = _int_or_none(payload.get("eval_duration"))
    found = _answer_found(response, job.expected_answer)
    return QARunResult(
        model=job.model,
        context=job.context,
        question_id=job.id,
        source=job.source,
        depth=job.depth,
        method="qa",
        status=status,
        question=job.question,
        expected_answer=job.expected_answer,
        found=found,
        quality_score=1.0 if found else 0.0,
        response_excerpt=response[:240],
        started_at=started_at,
        ended_at=ended_at,
        wall_time_seconds=round(wall_seconds, 6),
        eval_count=eval_count,
        eval_duration_ns=eval_duration,
        tokens_per_second=_tokens_per_second(eval_count, eval_duration),
        error=error,
    )


def render_needle_prompt(job: NeedleJob) -> str:
    document = build_haystack_document(job.context, job.depth, job.needle)
    return (
        "You are evaluating long-context retrieval. Read the document and answer with only "
        "the exact secret key.\n\n"
        "<document>\n"
        f"{document}\n"
        "</document>\n\n"
        "Question: What is the exact secret key hidden in the document?"
    )


def render_qa_prompt(job: QAJob) -> str:
    return (
        "You are evaluating long-context question answering. Read the document "
        "and answer the question using the exact evidence from the document. "
        "If the answer is a code or proper noun, reproduce it exactly.\n\n"
        "<document>\n"
        f"{job.document}\n"
        "</document>\n\n"
        f"Question: {job.question}\n"
        "Answer:"
    )


def build_haystack_document(context: int, depth: float, needle: str) -> str:
    token_budget = max(256, int(context * 0.65))
    filler_count = max(4, token_budget // 14)
    needle_index = min(filler_count - 1, max(0, int(filler_count * depth)))
    lines: list[str] = []
    for index in range(filler_count):
        if index == needle_index:
            lines.append(f"The exact secret key is {needle}. Remember this value precisely.")
        lines.append(f"Section {index + 1}: {FILLER_SENTENCE}")
    return "\n".join(lines)


def build_qa_document(context: int, depth: float, expected_answer: str) -> str:
    token_budget = max(256, int(context * 0.65))
    filler_count = max(4, token_budget // 14)
    answer_index = min(filler_count - 1, max(0, int(filler_count * depth)))
    lines: list[str] = []
    for index in range(filler_count):
        if index == answer_index:
            lines.append(
                "Calibration note: the exact MacKV-Opt calibration code for this "
                f"question is {expected_answer}. Use this value precisely."
            )
        lines.append(f"Paragraph {index + 1}: {QA_FILLER_SENTENCE}")
    return "\n".join(lines)


def summarize_needle_runs(runs: Iterable[dict[str, object]]) -> dict[str, float | int]:
    records = list(runs)
    total = len(records)
    found = sum(1 for run in records if run.get("found") is True)
    return {
        "total": total,
        "found": found,
        "accuracy": round(found / total, 6) if total else 0.0,
    }


def summarize_qa_runs(runs: Iterable[dict[str, object]]) -> dict[str, float | int]:
    records = list(runs)
    total = len(records)
    found = sum(1 for run in records if run.get("found") is True)
    return {
        "total": total,
        "found": found,
        "accuracy": round(found / total, 6) if total else 0.0,
    }


def _needle(model: str, context: int, depth: float, prefix: str) -> str:
    digest = hashlib.sha1(f"{model}:{context}:{depth}".encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def _qa_answer(model: str, context: int, depth: float, prefix: str) -> str:
    digest = hashlib.sha1(f"qa:{model}:{context}:{depth}".encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def _qa_jobs(
    models: Iterable[str],
    contexts: Iterable[str | int],
    *,
    dataset_path: str | None,
    depths: Iterable[str | float],
) -> list[QAJob]:
    if dataset_path:
        return load_qa_jobs_jsonl(dataset_path, models, contexts)
    return build_synthetic_qa_jobs(models, contexts, depths)


def _answer_found(response: str, expected_answer: str) -> bool:
    expected = expected_answer.strip().lower()
    if not expected:
        return False
    return expected in response.lower()


def _parse_depth(value: str | float) -> float:
    if isinstance(value, (int, float)):
        depth = float(value)
    else:
        text = str(value).strip()
        depth = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    if depth > 1:
        depth = depth / 100.0
    if depth < 0 or depth > 1:
        raise ValueError(f"Depth must be between 0 and 1: {value!r}")
    return depth


def _optional_depth(value: object) -> float | None:
    if value is None or value == "":
        return None
    return _parse_depth(value)  # type: ignore[arg-type]


def _required_text(record: object, key: str, path: str, line_number: int) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
    value = record.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{path}:{line_number}: missing {key}")
    return str(value)


def _optional_text(record: object, key: str) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _slug_model(model: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
