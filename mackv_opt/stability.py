from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

STABLE_CONTEXT_POLICIES = ("any", "all", "fraction")


@dataclass(frozen=True)
class StabilityConfig:
    max_swap_bytes: int | None = 512 * 1024**2
    min_tokens_per_second: float | None = None
    require_status_ok: bool = True
    critical_pressures: tuple[str, ...] = ("critical",)
    stable_context_policy: str = "any"
    min_stable_fraction: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def normalized(self) -> "StabilityConfig":
        policy = str(self.stable_context_policy or "any").lower()
        if policy not in STABLE_CONTEXT_POLICIES:
            raise ValueError(
                f"Invalid stable context policy {self.stable_context_policy!r}; "
                f"choose one of: {', '.join(STABLE_CONTEXT_POLICIES)}"
            )
        fraction = _clamp_fraction(self.min_stable_fraction)
        if policy == "all":
            fraction = 1.0
        return StabilityConfig(
            max_swap_bytes=self.max_swap_bytes,
            min_tokens_per_second=self.min_tokens_per_second,
            require_status_ok=self.require_status_ok,
            critical_pressures=tuple(self.critical_pressures),
            stable_context_policy=policy,
            min_stable_fraction=fraction,
        )


def classify_run_stability(run: dict[str, Any], config: StabilityConfig | None = None) -> dict[str, object]:
    config = (config or StabilityConfig()).normalized()
    reasons: list[str] = []
    if config.require_status_ok and run.get("status") != "ok":
        reasons.append(f"status={run.get('status') or 'missing'}")
    pressure = str(run.get("memory_pressure") or "unknown").lower()
    if pressure in {item.lower() for item in config.critical_pressures}:
        reasons.append(f"memory_pressure={pressure}")
    swap = _number_or_none(run.get("swap_bytes"))
    if config.max_swap_bytes is not None and swap is not None and swap > config.max_swap_bytes:
        reasons.append(f"swap_bytes>{config.max_swap_bytes}")
    tps = _number_or_none(run.get("tokens_per_second"))
    if config.min_tokens_per_second is not None and tps is not None and tps < config.min_tokens_per_second:
        reasons.append(f"tokens_per_second<{config.min_tokens_per_second}")
    return {
        "stable": not reasons,
        "stability_reason": "stable" if not reasons else "; ".join(reasons),
    }


def annotate_runs_with_stability(
    runs: Iterable[dict[str, Any]],
    config: StabilityConfig | None = None,
) -> list[dict[str, Any]]:
    selected = (config or StabilityConfig()).normalized()
    annotated: list[dict[str, Any]] = []
    for run in runs:
        record = dict(run)
        record.update(classify_run_stability(record, selected))
        annotated.append(record)
    return annotated


def summarize_stability(
    runs: Iterable[dict[str, Any]],
    config: StabilityConfig | None = None,
) -> dict[str, Any]:
    selected = (config or StabilityConfig()).normalized()
    records = list(runs)
    by_model: dict[str, dict[str, Any]] = {}
    for run in records:
        model = str(run.get("model") or "")
        if not model:
            continue
        context = _int_or_none(run.get("context"))
        if context is None:
            continue
        state = by_model.setdefault(
            model,
            {
                "model": model,
                "max_stable_context": 0,
                "stable_runs": 0,
                "unstable_runs": 0,
                "contexts": {},
                "stable_context_policy": selected.stable_context_policy,
                "min_stable_fraction": selected.min_stable_fraction,
            },
        )
        context_key = str(context)
        context_state = state["contexts"].setdefault(
            context_key,
            {
                "context": context,
                "stable_runs": 0,
                "unstable_runs": 0,
                "runs": 0,
                "stable_fraction": 0.0,
                "context_stable": False,
                "reasons": {},
            },
        )
        context_state["runs"] += 1
        if run.get("stable") is True:
            state["stable_runs"] += 1
            context_state["stable_runs"] += 1
        else:
            state["unstable_runs"] += 1
            context_state["unstable_runs"] += 1
            reason = str(run.get("stability_reason") or "unknown")
            context_state["reasons"][reason] = context_state["reasons"].get(reason, 0) + 1
    for state in by_model.values():
        for context_state in state["contexts"].values():
            stable_runs = int(context_state["stable_runs"])
            runs = int(context_state["runs"])
            context_state["stable_fraction"] = round(stable_runs / runs, 6) if runs else 0.0
            context_state["context_stable"] = _context_is_stable(context_state, selected)
            if context_state["context_stable"]:
                state["max_stable_context"] = max(state["max_stable_context"], int(context_state["context"]))
    return {
        "stable_context_policy": selected.stable_context_policy,
        "min_stable_fraction": selected.min_stable_fraction,
        "models": list(by_model.values()),
        "max_stable_context_by_model": {
            model: record["max_stable_context"] for model, record in by_model.items()
        },
    }


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _context_is_stable(context_state: dict[str, Any], config: StabilityConfig) -> bool:
    stable_runs = int(context_state.get("stable_runs") or 0)
    runs = int(context_state.get("runs") or 0)
    if runs <= 0:
        return False
    if config.stable_context_policy == "any":
        return stable_runs > 0
    if config.stable_context_policy == "all":
        return stable_runs == runs
    return stable_runs / runs >= config.min_stable_fraction


def _clamp_fraction(value: Any) -> float:
    number = _number_or_none(value)
    if number is None:
        return 1.0
    return min(1.0, max(0.0, number))
