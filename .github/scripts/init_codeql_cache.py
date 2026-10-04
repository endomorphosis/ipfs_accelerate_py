#!/usr/bin/env python3
"""Initialize CodeQL result cache for composite actions / workflows."""

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
        from ipfs_accelerate_py.github_cli.codeql_cache import CodeQLCache
    except ImportError as exc:
        print(f"::warning::CodeQL cache modules not available: {exc}")
        print("::notice::Continuing without CodeQL caching")
        _write_output(status="disabled")
        return 0

    try:
        cache = CodeQLCache(
            cache_dir=os.environ.get("CODEQL_CACHE_DIR"),
            default_ttl=int(os.environ.get("CODEQL_CACHE_TTL", 86400)),
        )
        print("CodeQL cache initialized successfully")
        stats = cache.get_stats()
        print(f"Cached scans: {stats.get('scans_cached', 0)}")
        print(f"Scans retrieved: {stats.get('scans_retrieved', 0)}")
        print(f"Time saved: {stats.get('scan_time_saved_hours', 0):.1f} hours")
        _write_output(status="success")
        print("::notice::CodeQL cache is ready - redundant scans will be skipped")
        return 0
    except Exception as exc:  # pragma: no cover - defensive CI path
        print(f"::error::CodeQL cache initialization failed: {exc}")
        traceback.print_exc()
        _write_output(status="error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
