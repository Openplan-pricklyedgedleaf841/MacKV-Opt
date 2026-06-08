#!/usr/bin/env python3
"""Sync the public GitHub repository profile for MacKV-Opt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_REPO = "Lin-Aurora/MacKV-Opt"
DESCRIPTION = (
    "KV cache and context planner for longer local LLM contexts on Apple "
    "Silicon Macs with Ollama-compatible benchmarks."
)
HOMEPAGE = "https://github.com/Lin-Aurora/MacKV-Opt#readme"
TOPICS = [
    "apple-silicon",
    "ollama",
    "llama-cpp",
    "kv-cache",
    "local-llm",
    "llm-inference",
    "macos",
    "benchmark",
    "long-context",
    "mlx",
    "gguf",
    "ollama-tools",
]


@dataclass(frozen=True)
class RepositoryMetadata:
    repo: str
    description: str
    homepage: str
    topics: list[str]

    @property
    def repo_patch_payload(self) -> dict[str, str]:
        return {"description": self.description, "homepage": self.homepage}

    @property
    def topics_payload(self) -> dict[str, list[str]]:
        return {"names": self.topics}


def build_metadata(repo: str = DEFAULT_REPO) -> RepositoryMetadata:
    return RepositoryMetadata(
        repo=repo,
        description=DESCRIPTION,
        homepage=HOMEPAGE,
        topics=list(TOPICS),
    )


def token_from_environment(names: list[str]) -> tuple[str | None, str | None]:
    for name in names:
        token = os.environ.get(name)
        if token:
            return token, name
    return None, None


def github_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    headers = {
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "MacKV-Opt-metadata-sync",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        f"https://api.github.com{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")
    if not content:
        return {}
    return json.loads(content)


def read_current_metadata(repo: str, token: str | None = None) -> dict[str, Any]:
    current = github_request("GET", f"/repos/{repo}", token=token)
    return {
        "description": current.get("description"),
        "homepage": current.get("homepage"),
        "topics": current.get("topics", []),
    }


def compare_metadata(expected: RepositoryMetadata, current: dict[str, Any]) -> dict[str, Any]:
    current_topics = set(current.get("topics") or [])
    expected_topics = set(expected.topics)
    return {
        "description_matches": current.get("description") == expected.description,
        "homepage_matches": current.get("homepage") == expected.homepage,
        "topics_match": current_topics == expected_topics,
        "missing_topics": sorted(expected_topics - current_topics),
        "extra_topics": sorted(current_topics - expected_topics),
    }


def dry_run_payload(metadata: RepositoryMetadata) -> dict[str, Any]:
    return {
        "repo": metadata.repo,
        "description": metadata.description,
        "homepage": metadata.homepage,
        "topics": metadata.topics,
        "repo_patch_payload": metadata.repo_patch_payload,
        "topics_payload": metadata.topics_payload,
    }


def apply_metadata(metadata: RepositoryMetadata, token: str) -> dict[str, Any]:
    repo_result = github_request(
        "PATCH",
        f"/repos/{metadata.repo}",
        token=token,
        payload=metadata.repo_patch_payload,
    )
    topics_result = github_request(
        "PUT",
        f"/repos/{metadata.repo}/topics",
        token=token,
        payload=metadata.topics_payload,
    )
    return {
        "repo": metadata.repo,
        "description": repo_result.get("description"),
        "homepage": repo_result.get("homepage"),
        "topics": topics_result.get("names", []),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the MacKV-Opt GitHub About, homepage, and topics."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo as owner/name.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the metadata through the GitHub REST API.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read current metadata and compare it with the expected profile.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --check, return a non-zero exit code if the repository differs.",
    )
    parser.add_argument(
        "--token-env",
        action="append",
        default=None,
        help="Environment variable containing a GitHub token. Can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    metadata = build_metadata(args.repo)
    token_names = args.token_env or ["GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"]
    token, token_name = token_from_environment(token_names)

    try:
        if args.apply:
            if not token:
                print(
                    "No GitHub token found. Set one of: " + ", ".join(token_names),
                    file=sys.stderr,
                )
                return 2
            result = apply_metadata(metadata, token)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        payload = dry_run_payload(metadata)
        if args.check:
            current = read_current_metadata(metadata.repo, token=token)
            comparison = compare_metadata(metadata, current)
            payload = {
                **payload,
                "current": current,
                "comparison": comparison,
                "checked_with_token_env": token_name,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            if args.strict and not all(
                [
                    comparison["description_matches"],
                    comparison["homepage_matches"],
                    comparison["topics_match"],
                ]
            ):
                return 1
            return 0

        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"GitHub API error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"GitHub API request failed: {exc.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
