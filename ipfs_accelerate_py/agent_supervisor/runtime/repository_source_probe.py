"""Qualify an installed repository-tool runtime on disposable authored source.

No provider or target code is invoked. The actual stdio server must initialize,
read, edit, refresh and refuse an old generation. Source-bearing replies remain
in memory; only bounded checks and counters are returned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

from . import repository_source_policy as policy

SCHEMA = "repository-source-runtime-probe@1"


def _request(identity, method, params=None):
    value = dict(jsonrpc="2.0", method=method)
    if identity is not None:
        value["id"] = identity
    if params is not None:
        value["params"] = params
    return value


def run_probe() -> dict:
    """Exercise the fixed server using the runtime's exported import paths."""
    with tempfile.TemporaryDirectory(prefix="repository-runtime-probe-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        original = b"# Authored source comment retained.\ndef value():\n    return 1\n"
        (workspace / "probe.py").write_bytes(original)
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__/probe.pyc").write_bytes(b"\x00\xffauthored cache")
        (workspace / ".pytest_cache/v/cache").mkdir(parents=True)
        (workspace / ".pytest_cache/v/cache/nodeids").write_text("[]\n")
        cache_entries = {}
        for path in sorted(workspace.rglob("*")):
            name = path.relative_to(workspace).as_posix()
            if name == "probe.py":
                continue
            entry = dict(kind="directory" if path.is_dir() else "file", mode=stat.S_IMODE(path.stat().st_mode))
            if path.is_file():
                raw = path.read_bytes()
                entry.update(original_sha256=hashlib.sha256(raw).hexdigest(), original_bytes=len(raw))
            cache_entries[name] = entry
        inventory = dict(schema="repository-source-retained-opaque-cache@1",
            roots=[".pytest_cache", "__pycache__"], entries=cache_entries)
        cache_digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode()).hexdigest()
        binding = policy.create_binding(workspace=workspace, requested_policy=policy.POLICIES[2])
        identity = policy.create_runtime_store(binding)
        checks = {}
        try:
            policy.validate_binding(binding, materialized=True)
            def tool(number, name, **arguments):
                return _request(number, "tools/call", dict(name=name, arguments=arguments))
            requests = [
                _request(1, "initialize", dict(protocolVersion="2025-06-18", capabilities={},
                    clientInfo={"name": "installed-runtime-probe", "version": "1"})),
                _request(None, "notifications/initialized"),
                _request(2, "tools/list"),
                tool(3, "repository_status"),
                tool(4, "repository_read", generation=1, path="probe.py"),
                tool(5, "repository_replace_symbol", generation=1, path="probe.py", symbol="value",
                    expected_sha256=hashlib.sha256(original).hexdigest(),
                    replacement="def value():\n    return 2\n"),
                tool(6, "repository_read", generation=1, path="probe.py"),
                tool(7, "repository_read", generation=2, path="probe.py", representation="original"),
                tool(8, "repository_symbols", generation=2, query="value"),
                tool(9, "repository_cache_inventory", generation=2, expected_inventory_sha256=cache_digest),
            ]
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in requests)
            process = subprocess.run([sys.executable, binding["helper_script"],
                "--config", binding["config_path"], "--config-sha256", binding["config_sha256"]],
                input=encoded, capture_output=True, text=True, cwd="/tmp", timeout=60, check=False)
            if process.returncode != 0 or len(process.stdout.encode()) > 128 * 1024:
                raise ValueError("repository runtime probe process failed")
            responses = [json.loads(line) for line in process.stdout.splitlines()]
            if len(responses) != 9 or [row.get("id") for row in responses] != list(range(1, 10)):
                raise ValueError("repository runtime probe response population differs")
            if any("error" in row for row in responses):
                raise ValueError("repository runtime probe initialization or protocol failed")
            values = {}
            for row in responses[2:]:
                result = row["result"]
                if result.get("isError") is not (row["id"] == 6):
                    raise ValueError("repository runtime probe tool disposition differs")
                content = result["content"]
                if len(content) != 1 or content[0]["type"] != "text":
                    raise ValueError("repository runtime probe tool content differs")
                values[row["id"]] = json.loads(content[0]["text"])
            status, read, edit, stale, refreshed, symbols = (values[i] for i in range(3, 9))
            cache = status["native_index"].get("retained_opaque_cache", {})
            page = values[9]
            observed_entries = {row["path"]: {key: value for key, value in row.items()
                                if key not in {"path", "cache_root"}} for row in page["entries"]}
            expected_cache_hashes = {
                "__pycache__/probe.pyc": hashlib.sha256(b"\x00\xffauthored cache").hexdigest(),
                ".pytest_cache/v/cache/nodeids": hashlib.sha256(b"[]\n").hexdigest(),
            }
            checks = dict(
                initialized=responses[0]["result"]["serverInfo"]["name"] == "repository_source",
                complete_tool_inventory={row["name"] for row in responses[1]["result"]["tools"]} == set(policy.TOOLS),
                source_current=status["source_current"] is True,
                native_scope_joined=status["native_index"]["source_scope_verified"] is True,
                cache_coverage_explicit=(cache.get("roots_count") == 2 and "entries" not in cache and "roots" not in cache
                    and cache.get("inventory_sha256") == cache_digest
                    and page["inventory_sha256"] == cache_digest and page["total"] == 6 and page["next_offset"] is None
                    and observed_entries == cache_entries
                    and sorted(row["path"] for row in page["entries"] if row["cache_root"]) == inventory["roots"]
                    and cache.get("entry_count") == 6 and cache.get("file_count") == 2
                    and cache.get("directory_count") == 4 and cache.get("native_indexed") is False
                    and cache.get("proof_authority") is False
                    and status["native_index"]["coverage"]["inventory_entries"] == 1
                    and all(observed_entries.get(path, {}).get("original_sha256") == digest
                            for path, digest in expected_cache_hashes.items())),
                compact_read=read["representation"] == "compact" and "Authored source comment" not in read["content"],
                original_hash_bound=read["source_sha256"] == hashlib.sha256(original).hexdigest(),
                canonical_edit=edit["canonical_changed"] is True and edit["generation"] == 2 and edit["refresh_required"] is False,
                stale_generation_refused=stale.get("error") == "stale_generation",
                refreshed_original_exact=refreshed["content"].encode() == (workspace / "probe.py").read_bytes()
                    and "return 2" in refreshed["content"] and "Authored source comment" in refreshed["content"],
                native_symbol_observed=any(row["name"] == "probe.value" for row in symbols["symbols"]),
                no_proof_authority=all(values[i]["proof_authority"] is False for i in (3, 4, 5, 7, 8, 9)),
            )
        finally:
            usage = policy.finish_runtime_store(binding, expected_identity=identity)
        checks["private_state_removed"] = usage["store_cleanup_complete"] is True
        checks["actual_tool_usage_observed"] = (usage["initializations"] == 1 and usage["compact_reads"] == 1
            and usage["canonical_edits"] == 1 and usage["refreshes"] == 1 and usage["errors"] == 1)
        return dict(schema=SCHEMA, qualified=all(checks.values()), checks=checks, usage=usage,
                    provider_calls=0, target_code_executed=False, source_messages_retained=False,
                    live_model_adoption_qualified=False)


def validate_probe(value):
    """Deployment cannot admit missing, partial or contradictory probe output."""
    expected_checks = {"initialized", "complete_tool_inventory", "source_current", "native_scope_joined",
        "cache_coverage_explicit", "compact_read", "original_hash_bound", "canonical_edit",
        "stale_generation_refused", "refreshed_original_exact", "native_symbol_observed",
        "no_proof_authority", "private_state_removed", "actual_tool_usage_observed"}
    if (type(value) is not dict or set(value) != {"schema", "qualified", "checks", "usage",
            "provider_calls", "target_code_executed", "source_messages_retained", "live_model_adoption_qualified"}
            or value.get("schema") != SCHEMA or value.get("qualified") is not True
            or type(value.get("checks")) is not dict or set(value["checks"]) != expected_checks
            or any(item is not True for item in value["checks"].values())
            or type(value.get("provider_calls")) is not int or value["provider_calls"] != 0
            or any(value.get(key) is not False for key in ("target_code_executed", "source_messages_retained", "live_model_adoption_qualified"))):
        raise ValueError("repository runtime probe did not qualify")
    usage = value["usage"]
    expected = dict(initializations=1, tool_calls=7, successful_reads=2, compact_reads=1,
                    canonical_edits=1, refreshes=1, errors=1, uncertain_edits=0)
    usage_fields = {"status", *expected, "returned_bytes", "store_cleanup_complete",
                    "artifacts_retained_after_invocation"}
    if (type(usage) is not dict or set(usage) != usage_fields or usage.get("status") != "observed"
            or any(type(usage.get(key)) is not int or usage[key] != number for key, number in expected.items())
            or type(usage.get("returned_bytes")) is not int or not 0 < usage["returned_bytes"] <= 128 * 1024
            or usage.get("store_cleanup_complete") is not True
            or usage.get("artifacts_retained_after_invocation") is not False):
        raise ValueError("repository runtime probe usage differs")
    return value


def main():
    try:
        result = validate_probe(run_probe())
    except Exception:
        print(json.dumps(dict(schema=SCHEMA, qualified=False, error="runtime_probe_failed")))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
