from mackv_opt.models import ModelProfile
from mackv_opt.ollama import build_run_command, normalize_show_payload


def test_normalize_show_payload_extracts_llama_metadata():
    payload = {
        "details": {"family": "llama", "parameter_size": "8B"},
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
        "size": 5_100_000_000,
    }

    profile = normalize_show_payload("llama3.1:8b", payload)

    assert profile == ModelProfile(
        name="llama3.1:8b",
        family="llama",
        parameter_count=8_000_000_000,
        size_bytes=5_100_000_000,
        hidden_size=4096,
        layer_count=32,
        attention_head_count=32,
        kv_head_count=8,
        architecture="llama",
        max_context=131072,
    )


def test_build_run_command_sets_ollama_options_without_changing_model():
    command = build_run_command(
        "llama3.1:8b",
        {"num_ctx": 65536, "num_gpu": 999, "temperature": 0.2},
        prompt="hello",
    )

    assert command == [
        "ollama",
        "run",
        "llama3.1:8b",
        "--option",
        "num_ctx=65536",
        "--option",
        "num_gpu=999",
        "--option",
        "temperature=0.2",
        "hello",
    ]
