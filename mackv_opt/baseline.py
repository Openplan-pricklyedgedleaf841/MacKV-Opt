from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .bench import DEFAULT_PROMPT
from .units import parse_context

BASELINE_LABELS = ("default", "manual-num-ctx", "mackv-opt")


def write_baseline_template(
    output_dir: str,
    models: Iterable[str],
    contexts: Iterable[str | int],
    *,
    memory_budget: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    manual_context: str | int | None = None,
    repeats: int = 3,
) -> dict[str, object]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    clean_models = [model.strip() for model in models if model and model.strip()]
    context_values = [str(context).strip() for context in contexts if str(context).strip()]
    manual_context_text = str(manual_context or _largest_context_label(context_values)).strip()
    payload: dict[str, object] = {
        "task": "baseline-template",
        "created_at": _utc_now(),
        "output_dir": str(target),
        "models": clean_models,
        "contexts": context_values,
        "memory_budget": memory_budget,
        "manual_context": manual_context_text,
        "baselines": list(BASELINE_LABELS),
        "model_directories": [],
    }
    for model in clean_models:
        model_dir = target / _safe_model_dir(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "report-tables").mkdir(parents=True, exist_ok=True)
        _write_model_readme(model_dir, model, context_values, manual_context_text, memory_budget)
        baseline_dirs: dict[str, str] = {}
        for label in BASELINE_LABELS:
            baseline_dir = model_dir / label
            baseline_dir.mkdir(parents=True, exist_ok=True)
            manifest = _baseline_manifest(
                label,
                model,
                context_values,
                manual_context_text,
                memory_budget,
                prompt,
                repeats,
            )
            (baseline_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "README.md").write_text(
                _baseline_readme(manifest),
                encoding="utf-8",
            )
            run_path = baseline_dir / "run.sh"
            run_path.write_text(_run_script(manifest), encoding="utf-8")
            _make_executable(run_path)
            baseline_dirs[label] = str(baseline_dir)
        payload["model_directories"].append(
            {
                "model": model,
                "path": str(model_dir),
                "baselines": baseline_dirs,
                "compare_command": _compare_command(model_dir),
            }
        )
    (target / "baseline-template-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _baseline_manifest(
    label: str,
    model: str,
    contexts: list[str],
    manual_context: str,
    memory_budget: str | None,
    prompt: str,
    repeats: int,
) -> dict[str, object]:
    return {
        "label": label,
        "model": model,
        "contexts": contexts,
        "manual_context": manual_context,
        "memory_budget": memory_budget,
        "prompt": prompt,
        "repeats": max(1, int(repeats)),
        "artifact": "full-run.json",
        "command": _command_for(label, model, contexts, manual_context, memory_budget, prompt, repeats),
        "notes": _notes_for(label),
    }


def _command_for(
    label: str,
    model: str,
    contexts: list[str],
    manual_context: str,
    memory_budget: str | None,
    prompt: str,
    repeats: int,
) -> list[str]:
    if label == "default":
        return [
            "mackv-opt",
            "bench",
            "--models",
            model,
            "--contexts",
            _first_context_label(contexts),
            "--use-ollama-default-options",
            "--execute",
            "--json",
            "--prompt",
            prompt,
            "--repeats",
            str(max(1, int(repeats))),
            "--output-dir",
            ".",
            "--output-prefix",
            "full-run",
        ]
    if label == "manual-num-ctx":
        return [
            "mackv-opt",
            "bench",
            "--models",
            model,
            "--contexts",
            manual_context,
            "--execute",
            "--json",
            "--prompt",
            prompt,
            "--repeats",
            str(max(1, int(repeats))),
            "--output-dir",
            ".",
            "--output-prefix",
            "full-run",
        ]
    command = [
        "mackv-opt",
        "experiment",
        model,
        "--contexts",
        ",".join(contexts),
        "--execute",
        "--json",
        "--prompt",
        prompt,
        "--repeats",
        str(max(1, int(repeats))),
        "--output-dir",
        ".",
        "--output-prefix",
        "full-run",
    ]
    if memory_budget:
        command[5:5] = ["--memory-budget", memory_budget]
    return command


def _baseline_readme(manifest: dict[str, object]) -> str:
    command = " ".join(shlex.quote(part) for part in manifest["command"])  # type: ignore[index]
    return "\n".join(
        [
            f"# {manifest['label']} baseline",
            "",
            f"Model: `{manifest['model']}`",
            f"Output file: `{manifest['artifact']}`",
            f"Memory budget: `{manifest.get('memory_budget') or 'not used'}`",
            "",
            "Run from this directory on the target Mac:",
            "",
            "```bash",
            "./run.sh",
            "```",
            "",
            "Equivalent command:",
            "",
            "```bash",
            command,
            "```",
            "",
            "The default baseline uses `bench --use-ollama-default-options`, so it records JSON without sending `num_ctx`.",
            "Save any external sampler logs next to this README when using Activity Monitor or powermetrics.",
            "",
            f"Notes: {manifest['notes']}",
            "",
        ]
    )


def _run_script(manifest: dict[str, object]) -> str:
    command = " ".join(shlex.quote(part) for part in manifest["command"])  # type: ignore[index]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            command,
            "",
        ]
    )


def _write_model_readme(
    model_dir: Path,
    model: str,
    contexts: list[str],
    manual_context: str,
    memory_budget: str | None,
) -> None:
    text = "\n".join(
        [
            f"# Baseline artifacts for {model}",
            "",
            "This directory is a baseline comparison template.",
            "",
            "- `default/`: default Ollama API run at the smallest context in the matrix.",
            "- `manual-num-ctx/`: manual `num_ctx` run at the selected manual context.",
            "- `mackv-opt/`: MacKV-Opt planned experiment across the full context matrix.",
            "",
            f"Contexts: `{','.join(contexts)}`",
            f"Manual context: `{manual_context}`",
            f"Memory budget: `{memory_budget or 'not set'}`",
            "",
            "After at least two directories contain `full-run.json`, run:",
            "",
            "```bash",
            _compare_command(model_dir),
            "```",
            "",
        ]
    )
    (model_dir / "README.md").write_text(text, encoding="utf-8")


def _compare_command(model_dir: Path) -> str:
    return (
        "mackv-opt compare "
        "default=default/full-run.json "
        "manual-num-ctx=manual-num-ctx/full-run.json "
        "mackv-opt=mackv-opt/full-run.json "
        "--baseline-label default "
        "--format markdown "
            "> report-tables/baseline-compare.md"
    )


def _notes_for(label: str) -> str:
    if label == "default":
        return "Captures out-of-box Ollama API behavior without sending num_ctx."
    if label == "manual-num-ctx":
        return "Captures the user choosing num_ctx manually."
    return "Captures the MacKV-Opt automatic planner plus benchmark and quality checks."


def _safe_model_dir(model: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in model.strip())
    return safe.strip("-") or "model"


def _make_executable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | 0o755)
    except OSError:
        return


def _first_context_label(contexts: list[str]) -> str:
    return contexts[0] if contexts else "8k"


def _largest_context_label(contexts: list[str]) -> str:
    if not contexts:
        return "8k"
    return max(contexts, key=parse_context)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
