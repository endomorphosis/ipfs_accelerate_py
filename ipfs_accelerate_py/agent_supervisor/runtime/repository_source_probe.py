"""Qualify an installed repository-tool runtime on disposable authored source.

No provider or target code is invoked. The actual stdio server must initialize,
read, edit, refresh and refuse an old generation. Schema @2 also requires an
indexed symbol read/edit round trip and its retrieval audit. Source-bearing
replies remain in memory; only bounded checks and counters are returned.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import tempfile
import time

from . import repository_source_policy as policy

SCHEMA = "repository-source-runtime-probe@2"
LEGACY_SCHEMA = "repository-source-runtime-probe@1"


def _request(identity, method, params=None):
    value = dict(jsonrpc="2.0", method=method)
    if identity is not None:
        value["id"] = identity
    if params is not None:
        value["params"] = params
    return value


def _responses(binding, requests):
    """Bounded interactive stdio so expansion uses the child's actual root."""
    process = subprocess.Popen([sys.executable, binding["helper_script"],
        "--config", binding["config_path"], "--config-sha256", binding["config_sha256"]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd="/tmp", bufsize=0)
    deadline, pending, received = time.monotonic() + 60, b"", 0
    responses = []
    try:
        def exchange(request):
            nonlocal pending, received
            raw = (json.dumps(request, separators=(",", ":")) + "\n").encode()
            if process.stdin.write(raw) != len(raw):
                raise ValueError("repository probe request write failed")
            if "id" not in request:
                return
            while b"\n" not in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise ValueError("repository probe response deadline")
                chunk = os.read(process.stdout.fileno(), min(65536, 128 * 1024 + 1 - received))
                received += len(chunk)
                if not chunk or received > 128 * 1024:
                    raise ValueError("repository probe output bound or EOF")
                pending += chunk
            line, pending = pending.split(b"\n", 1)
            response = json.loads(line)
            if response.get("id") != request["id"]:
                raise ValueError("repository probe response identity differs")
            responses.append(response)
        for request in requests:
            exchange(request)
        context = json.loads(responses[-1]["result"]["content"][0]["text"])
        if len(context["candidates"]) != 1:
            raise ValueError("repository probe context population differs")
        exchange(_request(11, "tools/call", dict(name="repository_capsule_expand", arguments=dict(
            generation=2, semantic_root_cid=context["semantic_root_cid"],
            stable_symbol_ids=[context["candidates"][0]["symbol_id"]]))))
        candidate = context["candidates"][0]
        exchange(_request(12, "tools/call", dict(name="repository_read", arguments=dict(
            generation=2, path=candidate["path"], symbol_id=candidate["symbol_id"]))))
        selected = json.loads(responses[-1]["result"]["content"][0]["text"])
        exchange(_request(13, "tools/call", dict(name="repository_replace_symbol", arguments=dict(
            generation=selected["generation"], path=selected["path"], symbol=selected["symbol"],
            expected_sha256=selected["source_sha256"], representation=selected["representation"],
            replacement=selected["content"].replace("return 2", "return 3")))))
        exchange(_request(14, "tools/call", dict(name="repository_read", arguments=dict(
            generation=2, path=candidate["path"], symbol_id=candidate["symbol_id"]))))
        exchange(_request(15, "tools/call", dict(name="repository_read", arguments=dict(
            generation=3, path=candidate["path"], symbol_id=candidate["symbol_id"],
            representation="original"))))
        exchange(_request(16, "tools/call", dict(name="repository_read", arguments=dict(
            generation=3, path=candidate["path"], representation="original"))))
        process.stdin.close()
        code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if code != 0 or pending or process.stdout.read(1):
            raise ValueError("repository probe exit or response population differs")
        return responses
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


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
                    replacement="def value():\n    # Authored definition context retained until compact edit.\n    return 2\n"),
                tool(6, "repository_read", generation=1, path="probe.py"),
                tool(7, "repository_read", generation=2, path="probe.py", representation="original"),
                tool(8, "repository_symbols", generation=2, query="value"),
                tool(9, "repository_cache_inventory", generation=2, expected_inventory_sha256=cache_digest),
                tool(10, "repository_context_query", generation=2, query="value", limit=1),
            ]
            responses = _responses(binding, requests)
            if len(responses) != 16 or [row.get("id") for row in responses] != list(range(1, 17)):
                raise ValueError("repository runtime probe response population differs")
            if any("error" in row for row in responses):
                raise ValueError("repository runtime probe initialization or protocol failed")
            values = {}
            for row in responses[2:]:
                result = row["result"]
                if result.get("isError") is not (row["id"] in {6, 14}):
                    raise ValueError("repository runtime probe tool disposition differs")
                content = result["content"]
                if len(content) != 1 or content[0]["type"] != "text":
                    raise ValueError("repository runtime probe tool content differs")
                values[row["id"]] = json.loads(content[0]["text"])
            status, read, edit, stale, refreshed, symbols = (values[i] for i in range(3, 9))
            cache = status["native_index"].get("retained_opaque_cache", {})
            page = values[9]
            context, expanded = values[10], values[11]
            candidate = context["candidates"][0]
            selected, targeted_edit, targeted_stale, targeted_current, final_original = (
                values[i] for i in range(12, 17))
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
                context_query_current=(context["generation"] == 2 and context["selected_count"] == 1
                    and candidate["path"] == "probe.py"
                    and candidate["source_sha256"] == refreshed["source_sha256"]),
                capsule_expansion_bound=(expanded["generation"] == 2
                    and expanded["semantic_root_cid"] == context["semantic_root_cid"]
                    and len(expanded["capsules"]) == 1
                    and expanded["capsules"][0]["capsule_cid"] == candidate["capsule_cid"]
                    and expanded["raw_source_fetch_required"]["probe.py"]["source_sha256"] == candidate["source_sha256"]),
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
                refreshed_original_exact=(final_original["content"].encode() == (workspace / "probe.py").read_bytes()
                    and final_original["source_sha256"] == hashlib.sha256((workspace / "probe.py").read_bytes()).hexdigest()
                    and "return 3" in final_original["content"] and "Authored source comment" in final_original["content"]),
                native_symbol_observed=any(row["name"] == "probe.value" for row in symbols["symbols"]),
                indexed_symbol_read=(selected["generation"] == 2 and selected["path"] == candidate["path"]
                    and selected["symbol_id"] == candidate["symbol_id"] and selected["symbol"] == "value"
                    and selected["source_sha256"] == candidate["source_sha256"] and selected["edit_supported"] is True
                    and selected["scope"] == "python_definition" and selected["representation"] == "compact"
                    and "return 2" in selected["content"] and "Authored" not in selected["content"]),
                indexed_symbol_edit=(targeted_edit["canonical_changed"] is True and targeted_edit["generation"] == 3
                    and targeted_edit["refresh_required"] is False
                    and targeted_edit["source_sha256"] == final_original["source_sha256"]),
                indexed_symbol_stale_refused=targeted_stale.get("error") == "stale_generation",
                indexed_symbol_refresh=(targeted_current["generation"] == 3
                    and targeted_current["symbol_id"] == selected["symbol_id"]
                    and targeted_current["source_sha256"] == final_original["source_sha256"]
                    and targeted_current["source_sha256"] != selected["source_sha256"]
                    and targeted_current["representation"] == "original" and "return 3" in targeted_current["content"]),
                no_proof_authority=all(values[i]["proof_authority"] is False for i in (3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16)),
            )
        finally:
            usage = policy.finish_runtime_store(binding, expected_identity=identity)
        checks["private_state_removed"] = usage["store_cleanup_complete"] is True
        checks["actual_tool_usage_observed"] = (usage["initializations"] == 1 and usage["compact_reads"] == 2
            and usage["canonical_edits"] == 2 and usage["refreshes"] == 2 and usage["errors"] == 2)
        try:
            _validate_retrieval(usage)
            checks["retrieval_usage_bound"] = True
        except ValueError:
            checks["retrieval_usage_bound"] = False
        return dict(schema=SCHEMA, qualified=all(checks.values()), checks=checks, usage=usage,
                    provider_calls=0, target_code_executed=False, source_messages_retained=False,
                    live_model_adoption_qualified=False)


def _validate_retrieval(usage):
    from .repository_source_mcp import RETRIEVAL_COUNTERS, validate_retrieval_observation
    retrieval = validate_retrieval_observation(usage.get("retrieval_observation"),
        tool_calls=usage.get("tool_calls"), errors=usage.get("errors"))
    expected = dict(query_calls=1, successful_queries=1, nonempty_queries=1, candidates_returned=1,
        expansion_calls=1, successful_expansions=1, nonempty_expansions=1, capsules_returned=1,
        generation_bindings_observed=2)
    if (retrieval["complete"] is not True
            or any(retrieval[key] != expected.get(key, 0) for key in RETRIEVAL_COUNTERS)):
        raise ValueError("repository runtime probe retrieval usage differs")


def validate_probe(value, *, allow_legacy=False):
    """Admit current qualification; old receipts require explicit review mode.

    Historical @1 evidence remains interpretable but never qualifies the new
    indexed symbol bridge for deployment. No source or provider calls replay.
    """
    if type(allow_legacy) is not bool:
        raise ValueError("explicit legacy probe review policy required")
    legacy = type(value) is dict and value.get("schema") == LEGACY_SCHEMA
    if legacy and not allow_legacy:
        raise ValueError("historical repository runtime probe requires current requalification")
    expected_checks = {"initialized", "complete_tool_inventory", "source_current", "native_scope_joined",
        "cache_coverage_explicit", "compact_read", "original_hash_bound", "canonical_edit",
        "stale_generation_refused", "refreshed_original_exact", "native_symbol_observed",
        "context_query_current", "capsule_expansion_bound",
        "no_proof_authority", "private_state_removed", "actual_tool_usage_observed"}
    if not legacy:
        expected_checks.update({"indexed_symbol_read", "indexed_symbol_edit", "indexed_symbol_stale_refused",
                                "indexed_symbol_refresh", "retrieval_usage_bound"})
    if (type(value) is not dict or set(value) != {"schema", "qualified", "checks", "usage",
            "provider_calls", "target_code_executed", "source_messages_retained", "live_model_adoption_qualified"}
            or value.get("schema") != (LEGACY_SCHEMA if legacy else SCHEMA) or value.get("qualified") is not True
            or type(value.get("checks")) is not dict or set(value["checks"]) != expected_checks
            or any(item is not True for item in value["checks"].values())
            or type(value.get("provider_calls")) is not int or value["provider_calls"] != 0
            or any(value.get(key) is not False for key in ("target_code_executed", "source_messages_retained", "live_model_adoption_qualified"))):
        raise ValueError("repository runtime probe did not qualify")
    usage = value["usage"]
    expected = (dict(initializations=1, tool_calls=9, successful_reads=2, compact_reads=1,
                    canonical_edits=1, refreshes=1, errors=1, uncertain_edits=0) if legacy else
                dict(initializations=1, tool_calls=14, successful_reads=5, compact_reads=2,
                    canonical_edits=2, refreshes=2, errors=2, uncertain_edits=0))
    usage_fields = {"status", *expected, "returned_bytes", "store_cleanup_complete",
                    "artifacts_retained_after_invocation"}
    if not legacy or type(usage) is dict and "retrieval_observation" in usage:
        usage_fields.add("retrieval_observation")
    if (type(usage) is not dict or set(usage) != usage_fields or usage.get("status") != "observed"
            or any(type(usage.get(key)) is not int or usage[key] != number for key, number in expected.items())
            or type(usage.get("returned_bytes")) is not int or not 0 < usage["returned_bytes"] <= 128 * 1024
            or usage.get("store_cleanup_complete") is not True
            or usage.get("artifacts_retained_after_invocation") is not False):
        raise ValueError("repository runtime probe usage differs")
    if "retrieval_observation" in usage:
        _validate_retrieval(usage)
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
