from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_language_switch_and_existing_chart_assets():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    chinese_text = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    assert "[English](README.md)" in text
    assert "[简体中文](README.zh-CN.md)" in text
    assert (ROOT / "README.zh-CN.md").exists()
    assert "[English](README.md)" in chinese_text
    assert "优化什么" in chinese_text

    for asset in [
        "docs/assets/mackv-opt-architecture.svg",
        "docs/assets/validation-snapshot.svg",
        "docs/assets/baseline-summary-example.svg",
    ]:
        assert asset in text
        assert asset in chinese_text
        assert (ROOT / asset).exists()


def test_project_has_license_and_repository_metadata_notes():
    assert (ROOT / "LICENSE").exists()
    metadata = (ROOT / "docs" / "GITHUB_REPOSITORY_METADATA.md")
    assert metadata.exists()

    text = metadata.read_text(encoding="utf-8")
    assert "KV cache and context planner" in text
    assert "https://github.com/Lin-Aurora/MacKV-Opt#readme" in text
    assert "python scripts/sync_github_metadata.py --apply" in text
    for topic in ["apple-silicon", "ollama", "kv-cache", "long-context", "ollama-tools"]:
        assert topic in text
