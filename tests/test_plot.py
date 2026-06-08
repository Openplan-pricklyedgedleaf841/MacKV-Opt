from mackv_opt.plot import extract_memory_series, render_memory_svg, write_memory_svg


def payload():
    return {
        "runs": [
            {
                "model": "llama3.1:8b",
                "context": 8192,
                "memory_series": [
                    {"timestamp": "2026-06-07T00:00:00Z", "process_memory_bytes": 1024, "memory_pressure": "normal"},
                    {"timestamp": "2026-06-07T00:00:01Z", "process_memory_bytes": 2048, "memory_pressure": "normal"},
                ],
            }
        ]
    }


def test_extract_memory_series_from_benchmark_payload():
    series = extract_memory_series(payload())

    assert len(series) == 1
    assert series[0]["model"] == "llama3.1:8b"
    assert series[0]["samples"][1]["process_memory_bytes"] == 2048


def test_render_memory_svg_draws_line():
    svg = render_memory_svg(extract_memory_series(payload())[0])

    assert svg.startswith("<svg")
    assert "MacKV-Opt memory series" in svg
    assert "<path" in svg
    assert "peak:" in svg


def test_render_memory_svg_handles_missing_values():
    svg = render_memory_svg({"samples": [{"process_memory_bytes": None}]})

    assert "No process_memory_bytes samples" in svg


def test_write_memory_svg_writes_file(tmp_path):
    output = tmp_path / "memory.svg"

    written = write_memory_svg(payload(), str(output))

    assert written == str(output)
    assert output.exists()
    assert "<svg" in output.read_text(encoding="utf-8")
