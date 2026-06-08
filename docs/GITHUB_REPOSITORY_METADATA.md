# GitHub Repository Metadata

Use this file as the canonical public repository profile for MacKV-Opt.

## About

KV cache and context planner for running longer local LLM contexts on Apple
Silicon Macs with Ollama-compatible benchmarks.

## Homepage

https://github.com/Lin-Aurora/MacKV-Opt#readme

## Topics

- apple-silicon
- ollama
- llama-cpp
- kv-cache
- local-llm
- llm-inference
- macos
- benchmark
- long-context
- mlx
- gguf
- research-artifact

## API Commands

If the GitHub CLI is available:

```bash
gh repo edit Lin-Aurora/MacKV-Opt \
  --description "KV cache and context planner for running longer local LLM contexts on Apple Silicon Macs with Ollama-compatible benchmarks." \
  --homepage "https://github.com/Lin-Aurora/MacKV-Opt#readme" \
  --add-topic apple-silicon \
  --add-topic ollama \
  --add-topic llama-cpp \
  --add-topic kv-cache \
  --add-topic local-llm \
  --add-topic llm-inference \
  --add-topic macos \
  --add-topic benchmark \
  --add-topic long-context \
  --add-topic mlx \
  --add-topic gguf \
  --add-topic research-artifact
```

If only the GitHub REST API is available, patch `description` and `homepage`
with `PATCH /repos/Lin-Aurora/MacKV-Opt`, then replace topics with
`PUT /repos/Lin-Aurora/MacKV-Opt/topics`.
