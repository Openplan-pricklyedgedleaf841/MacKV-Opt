from mackv_opt.capabilities import detect_runtime_capabilities


def test_detect_runtime_capabilities_handles_missing_commands(monkeypatch):
    monkeypatch.setattr("mackv_opt.capabilities.shutil.which", lambda command: None)

    payload = detect_runtime_capabilities().to_dict()

    assert payload["ollama"]["available"] is False
    assert payload["llama_cpp"]["available"] is False
    assert payload["supports_llama_cpp_cache_type_k"] is False
    assert payload["warnings"]


def test_detect_runtime_capabilities_parses_ollama_and_llama_cpp_help(monkeypatch):
    paths = {"ollama": "/usr/local/bin/ollama", "llama-cli": "/usr/local/bin/llama-cli"}

    def fake_which(command):
        return paths.get(command)

    def fake_command_text(command, **kwargs):
        joined = " ".join(command)
        if joined == "ollama --version":
            return "ollama version 0.12.1"
        if joined == "ollama run --help":
            return "Usage: ollama run MODEL --option num_ctx=8192 --option num_gpu=999"
        if joined == "llama-cli --version":
            return "llama.cpp build 6500"
        if joined == "llama-cli --help":
            return "--ctx-size N --cache-type-k TYPE --cache-type-v TYPE --kv-offload"
        return ""

    monkeypatch.setattr("mackv_opt.capabilities.shutil.which", fake_which)
    monkeypatch.setattr("mackv_opt.capabilities._command_text", fake_command_text)

    payload = detect_runtime_capabilities().to_dict()

    assert payload["ollama"]["available"] is True
    assert payload["ollama"]["version"] == "0.12.1"
    assert payload["llama_cpp"]["available"] is True
    assert payload["llama_cpp"]["command"] == "llama-cli"
    assert payload["supports_ollama_num_ctx"] is True
    assert payload["supports_ollama_num_gpu"] is True
    assert payload["supports_llama_cpp_ctx_size"] is True
    assert payload["supports_llama_cpp_cache_type_k"] is True
    assert payload["supports_llama_cpp_cache_type_v"] is True
    assert payload["supports_llama_cpp_kv_offload"] is True


def test_detect_runtime_capabilities_does_not_treat_unrelated_main_as_llama_cpp(monkeypatch):
    def fake_which(command):
        return "C:/Windows/System32/main.CPL" if command == "main" else None

    monkeypatch.setattr("mackv_opt.capabilities.shutil.which", fake_which)
    monkeypatch.setattr("mackv_opt.capabilities._command_text", lambda command, **kwargs: "")

    payload = detect_runtime_capabilities().to_dict()

    assert payload["llama_cpp"]["available"] is False
