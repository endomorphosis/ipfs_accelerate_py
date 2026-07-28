#!/usr/bin/env python3
"""Decide whether a CodeQL scan can be skipped from cache."""

from __future__ import annotations

import os
import sys
import time


def _write_output(**fields: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in fields.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    try:
        from ipfs_accelerate_py.github_cli.codeql_cache import get_global_codeql_cache

        cache = get_global_codeql_cache()
        repo = os.environ.get("GITHUB_REPOSITORY")
        commit_sha = os.environ.get("COMMIT_SHA") or ""
        scan_config = {
            "queries": "security-extended",
            "languages": "auto",
        }
        should_skip, cached_result = cache.should_skip_scan(
            repo=repo,
            commit_sha=commit_sha,
            scan_config=scan_config,
        )
        fields = {"should_skip": str(should_skip).lower()}
        if cached_result and getattr(cached_result, "sarif_location", None):
            fields["sarif_path"] = str(cached_result.sarif_location)
            age_hours = (time.time() - float(cached_result.timestamp)) / 3600.0
            print(f"Cached scan found for {commit_sha[:8]}")
            print(f"  - Alerts: {cached_result.alerts_count}")
            print(f"  - Age: {age_hours:.1f} hours")
            print(f"  - SARIF: {cached_result.sarif_location}")
        else:
            fields["sarif_path"] = ""
            print(f"No cached scan for {commit_sha[:8]}, scan required")
        _write_output(**fields)
        if should_skip:
            print("::notice::Skipping CodeQL scan - using cached results")
        else:
            print("::notice::Running CodeQL scan")
        return 0
    except Exception as exc:  # pragma: no cover - defensive CI path
        print(f"::warning::Cache check failed: {exc}")
        _write_output(should_skip="false", sarif_path="")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
