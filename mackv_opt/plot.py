from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


def extract_memory_series(payload: Any) -> list[dict[str, Any]]:
    runs = _extract_runs(payload)
    series: list[dict[str, Any]] = []
    for run in runs:
        samples = run.get("memory_series") if isinstance(run, dict) else None
        if isinstance(samples, list):
            series.append(
                {
                    "model": run.get("model", ""),
                    "context": run.get("context", ""),
                    "samples": [sample for sample in samples if isinstance(sample, dict)],
                }
            )
    return series


def render_memory_svg(series: dict[str, Any], *, width: int = 960, height: int = 420) -> str:
    samples = series.get("samples") if isinstance(series.get("samples"), list) else []
    values = [_int_or_none(sample.get("process_memory_bytes")) for sample in samples]
    points = [(index, value) for index, value in enumerate(values) if value is not None]
    title = f"MacKV-Opt memory series: {series.get('model', '')} @ {series.get('context', '')}"
    margin_left = 72
    margin_right = 28
    margin_top = 48
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#6b7280}.grid{stroke:#e5e7eb}.line{fill:none;stroke:#2563eb;stroke-width:2.5}.dot{fill:#1d4ed8}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
    ]

    if not points:
        parts.append(
            f'<text x="{margin_left}" y="{margin_top + 40}" font-size="14">No process_memory_bytes samples were available.</text>'
        )
        parts.append("</svg>")
        return "\n".join(parts)

    min_value = min(value for _, value in points)
    max_value = max(value for _, value in points)
    if min_value == max_value:
        min_value = max(0, min_value - 1)
        max_value += 1

    for tick in range(5):
        ratio = tick / 4
        y = margin_top + plot_height * ratio
        value = max_value - (max_value - min_value) * ratio
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        parts.append(f'<text x="8" y="{y + 4:.2f}" font-size="12">{_format_bytes(value)}</text>')

    max_index = max(index for index, _ in points) or 1
    coords = []
    for index, value in points:
        x = margin_left + (index / max_index) * plot_width
        y = margin_top + (1 - ((value - min_value) / (max_value - min_value))) * plot_height
        coords.append((x, y))
    path_data = " ".join(("M" if idx == 0 else "L") + f"{x:.2f},{y:.2f}" for idx, (x, y) in enumerate(coords))
    parts.append(f'<path class="line" d="{path_data}"/>')
    for x, y in coords:
        parts.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="3"/>')

    parts.append(f'<text x="{margin_left}" y="{height - 18}" font-size="12">sample index</text>')
    parts.append(f'<text x="{width - 230}" y="{height - 18}" font-size="12">peak: {_format_bytes(max_value)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_memory_svg(payload: Any, output: str, *, series_index: int = 0) -> str:
    series_list = extract_memory_series(payload)
    selected = series_list[series_index] if 0 <= series_index < len(series_list) else {"samples": []}
    svg = render_memory_svg(selected)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _extract_runs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return [run for run in payload["runs"] if isinstance(run, dict)]
    if isinstance(payload, dict) and payload.get("task") == "experiment":
        bench = payload.get("bench")
        if isinstance(bench, dict) and isinstance(bench.get("runs"), list):
            return [run for run in bench["runs"] if isinstance(run, dict)]
    return []


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_bytes(value: float) -> str:
    units = [("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]
    for suffix, scale in units:
        if abs(value) >= scale:
            return f"{value / scale:.2f} {suffix}"
    return f"{value:.0f} B"
