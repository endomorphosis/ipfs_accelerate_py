#!/usr/bin/env python3
"""Initialize GitHub API cache for composite actions / workflows.

Kept as a standalone script so composite action YAML does not embed a
heredoc that breaks YAML block-scalar indentation rules.
"""

from __future__ import annotations

import os
import sys
import traceback


def _write_output(**fields: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in fields.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    try:
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
    except ImportError as exc:
        print(f"::warning::Cache modules not available: {exc}")
        print("::notice::Continuing without caching")
        _write_output(peer_count="0", status="disabled")
        return 0

    try:
        cache = GitHubAPICache(
            cache_dir=os.environ.get("CACHE_DIR"),
            enable_p2p=os.environ.get("ENABLE_P2P_CACHE", "true").lower() == "true",
            enable_peer_discovery=(
                os.environ.get("ENABLE_PEER_DISCOVERY", "true").lower() == "true"
            ),
            github_repo=os.environ.get("GITHUB_REPOSITORY"),
            max_cache_size=int(os.environ.get("GITHUB_CACHE_SIZE", 5000)),
            default_ttl=int(os.environ.get("CACHE_DEFAULT_TTL", 300)),
            p2p_listen_port=int(os.environ.get("P2P_LISTEN_PORT", 9000)),
        )

        print("GitHub API cache initialized successfully")
        stats = cache.get_stats()
        print(f"Cache entries: {stats.get('cache_size', 0)}")
        print(f"P2P enabled: {stats.get('p2p_enabled', False)}")

        if stats.get("p2p_enabled"):
            peer_count = int(stats.get("connected_peers", 0) or 0)
            print(f"Connected peers: {peer_count}")
            _write_output(peer_count=str(peer_count), status="success")
        else:
            _write_output(peer_count="0", status="fallback")

        print("::notice::GitHub API cache is ready - API calls will be cached")
        return 0
    except Exception as exc:  # pragma: no cover - defensive CI path
        print(f"::error::Cache initialization failed: {exc}")
        traceback.print_exc()
        _write_output(peer_count="0", status="fallback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
