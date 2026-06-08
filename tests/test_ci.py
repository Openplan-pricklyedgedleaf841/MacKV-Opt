from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_ci_workflow_covers_tests_and_cli_smoke():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8")

    assert workflow.exists()
    assert "ubuntu-latest" in text
    assert "macos-latest" in text
    assert 'python -m pip install -e ".[dev]"' in text
    assert "python -m pytest -q" in text
    assert "mackv-opt doctor --json" in text
    assert "mackv-opt collect" in text
    assert "ci-collect" in text
    assert "mackv-opt audit ci-collect/manifest.json" in text
    assert "--no-require-artifacts" in text
    assert "--table readiness-compact" in text
    assert "--table readiness" in text
    assert "mackv-opt capabilities --json" in text
    assert "--skip-capability-check" in text
    assert "mackv-opt experiment" in text
    assert "mackv-opt qa" in text
    assert "mackv-opt report" in text
    assert "mackv-opt baseline-summary" in text
    assert "mackv-opt plot-memory" in text
