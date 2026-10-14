"""Measure bounded status and explicit cache pages through real local MCP stdio.

Authored files only. No target code, providers or containers run. Source and
cache contents stay in memory; the report retains counters and Boolean checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .terminal_repository_tools_experiment import _Client, _sha, _wire
from .terminal_source_view_experiment import _render, local_token_counter

SCHEMA = "terminal-repository-cache-budget-experiment@1"
MAX_CACHE_FILES = 384


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _cache_case(*, cache_files, status_repeats, page_limit, counter):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_mcp import summarize_repository_audit
    checks, failure, audit = {}, None, {}
    with tempfile.TemporaryDirectory(prefix="repository-cache-budget-", dir="/tmp") as temporary:
        root = Path(temporary)
        workspace, private = root / "workspace", root / "private"
        workspace.mkdir()
        source = b"# Authored cache-budget source retained exactly.\ndef authored_value():\n    return 7\n"
        (workspace / "source.py").write_bytes(source)
        cached = {}
        for index in range(cache_files):
            name = ("__pycache__/entry_%04d.pyc" if index % 2 == 0
                    else ".pytest_cache/v/cache/entry_%04d") % index
            raw = b"\x00\xffauthored opaque cache " + str(index).encode()
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            cached[name] = raw
        directories = {str(parent.relative_to(workspace)) for name in cached
                       for parent in (workspace / name).parents if parent != workspace and parent.is_relative_to(workspace)}
        private.mkdir(mode=0o700)
        (private / "state").mkdir(mode=0o700)
        config = private / "config.json"
        config.write_bytes(_wire(dict(schema="repository-source-mcp-config@1", workspace=str(workspace),
            state_dir=str(private / "state"), audit_store=str(private / "operations.jsonl"), representation="compact")))
        config.chmod(0o600)
        client = _Client(config, counter)
        summary_size, inventory_size, pages, status_count, final_hash = 0, 0, 0, 0, None
        try:
            initialized = client.request("initialize", dict(protocolVersion="2025-06-18", capabilities={},
                clientInfo={"name": "authored-cache-budget", "version": "1"}), phase="setup")
            checks["initialized"] = initialized["serverInfo"]["name"] == "repository_source"
            client.request("notifications/initialized", phase="setup", notification=True)
            tools = client.request("tools/list", phase="setup")
            checks["cache_paging_advertised"] = any(row["name"] == "repository_cache_inventory" for row in tools["tools"])
            statuses = [client.tool("repository_status") for _ in range(status_repeats)]
            status_count += len(statuses)
            cache = statuses[0]["native_index"]["retained_opaque_cache"]
            generation, digest = statuses[0]["generation"], cache["inventory_sha256"]
            summary_size = len(_canonical(cache))
            checks["status_omits_entry_and_root_paths"] = all(
                "entries" not in row["native_index"]["retained_opaque_cache"]
                and "roots" not in row["native_index"]["retained_opaque_cache"] for row in statuses)
            checks["summary_counts_exact"] = (cache["file_count"] == cache_files
                and cache["directory_count"] == len(directories)
                and cache["entry_count"] == cache_files + len(directories)
                and cache["original_bytes"] == sum(map(len, cached.values()))
                and cache["native_indexed"] is False and cache["proof_authority"] is False)
            checks["status_summary_stable"] = all(
                row["native_index"]["retained_opaque_cache"] == cache for row in statuses)
            entries, roots, offset = {}, [], 0
            while True:
                page = client.tool("repository_cache_inventory", dict(generation=generation,
                    expected_inventory_sha256=digest, offset=offset, limit=page_limit))
                pages += 1
                if page["offset"] != offset or page["inventory_sha256"] != digest or len(page["entries"]) > page_limit:
                    raise ValueError("cache page binding differs")
                for row in page["entries"]:
                    name = row["path"]
                    if name in entries:
                        raise ValueError("duplicate cache page entry")
                    entries[name] = {key: value for key, value in row.items() if key not in {"path", "cache_root"}}
                    if row["cache_root"]:
                        roots.append(name)
                if page["next_offset"] is None:
                    break
                if page["next_offset"] != offset + len(page["entries"]):
                    raise ValueError("cache cursor differs")
                offset = page["next_offset"]
                if pages > 32:
                    raise ValueError("cache page population exceeds fixture bound")
            inventory = dict(schema=cache["inventory_schema"], roots=sorted(roots), entries=entries)
            inventory_size = len(_canonical(inventory))
            checks["complete_pages_reconstruct_inventory_digest"] = _sha(_canonical(inventory)) == digest
            checks["page_population_exact"] = set(entries) == set(cached) | directories
            checks["page_source_hashes_exact"] = all(entries[name]["kind"] == "file"
                and entries[name]["original_sha256"] == _sha(raw)
                and entries[name]["original_bytes"] == len(raw) for name, raw in cached.items())
            checks["page_directories_exact"] = all(entries[name]["kind"] == "directory" for name in directories)
            read = client.tool("repository_read", dict(generation=generation, path="source.py"))
            checks["canonical_read_hash_bound"] = read["source_sha256"] == _sha(source)
            if cached:
                name = sorted(cached)[0]
                cached[name] += b"\nAuthored cache update."
                (workspace / name).write_bytes(cached[name])
                stale = client.tool("repository_cache_inventory", dict(generation=generation,
                    expected_inventory_sha256=digest, limit=page_limit), expect_error=True)
                checks["cache_change_fences_pages"] = stale["error"] == "stale_source"
                refreshed = client.tool("repository_refresh", dict(generation=generation))
                old = client.tool("repository_cache_inventory", dict(generation=generation,
                    expected_inventory_sha256=digest, limit=page_limit), expect_error=True)
                checks["refresh_fences_old_generation"] = old["error"] == "stale_generation"
                wrong = client.tool("repository_cache_inventory", dict(generation=refreshed["generation"],
                    expected_inventory_sha256=digest, limit=page_limit), expect_error=True)
                checks["refresh_fences_old_inventory"] = wrong["error"] == "stale_inventory_hash"
                final = client.tool("repository_status")
                status_count += 1
                # Unchanged native files are skipped before parser-cache lookup;
                # a cache-only refresh need not increment its reuse counter.
                checks["refresh_skips_unchanged_source_parse"] = (
                    final["native_index"]["ast_stats_delta"]["parse_invocations"] == 0)
                checks["refreshed_native_scope_current"] = (
                    final["source_current"] is True
                    and final["native_index"]["source_scope_verified"] is True
                    and final["native_index"]["coverage"] == statuses[0]["native_index"]["coverage"])
                current = client.tool("repository_read", dict(generation=refreshed["generation"], path="source.py"))
                checks["refreshed_canonical_read_hash_bound"] = current["source_sha256"] == _sha(source)
                checks["new_inventory_digest_observed"] = final["native_index"]["retained_opaque_cache"]["inventory_sha256"] != digest
            checks["canonical_source_unchanged"] = (workspace / "source.py").read_bytes() == source
            checks["opaque_cache_originals_retained"] = all((workspace / name).read_bytes() == raw for name, raw in cached.items())
            final_hash = _sha((workspace / "source.py").read_bytes())
        except Exception as error:
            checks["workflow_completed"] = False
            failure = dict(error_type=type(error).__name__)
        finally:
            exit_code = client.close()
        protocol = client.metrics()
        audit = summarize_repository_audit(private / "operations.jsonl")
        checks["clean_subprocess_exit"] = exit_code == 0
        checks["component_totals_match"] = protocol["component_totals_match"]
        checks["status_transmissions_counted"] = protocol["components"]["status"]["requests"] == status_count
        result = dict(cache_file_count=cache_files, requested_status_repeats=status_repeats,
            requested_page_limit=page_limit, successful_inventory_pages=pages,
            cache_summary_json_bytes=summary_size, reconstructed_inventory_json_bytes=inventory_size,
            protocol=protocol, audit=audit, final_source_sha256=final_hash, checks=checks,
            failure=failure, content_bearing_messages_retained=False)
    result["checks"]["private_fixture_removed"] = not root.exists()
    result["qualified"] = all(result["checks"].values())
    return result


def run_experiment(*, output, cache_counts=(0, 128), status_repeats=4, page_limit=32, tokenizer_cache=None):
    cache_counts = tuple(cache_counts)
    if (not 1 <= len(cache_counts) <= 4 or len(set(cache_counts)) != len(cache_counts)
            or any(type(value) is not int or not 0 <= value <= MAX_CACHE_FILES for value in cache_counts)
            or type(status_repeats) is not int or not 1 <= status_repeats <= 8
            or type(page_limit) is not int or not 16 <= page_limit <= 100):
        raise ValueError("explicit bounded cache experiment population required")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("cache experiment output must be a new canonical directory")
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output.mkdir(parents=True)
    rows = []
    for count in cache_counts:
        try:
            rows.append(_cache_case(cache_files=count, status_repeats=status_repeats,
                                    page_limit=page_limit, counter=counter))
        except Exception as error:
            rows.append(dict(cache_file_count=count, qualified=False, error_type=type(error).__name__,
                             content_bearing_messages_retained=False))
    qualified = all(row["qualified"] for row in rows)
    same_source = qualified and len({row["final_source_sha256"] for row in rows}) == 1
    result = dict(schema=SCHEMA, qualified=qualified and same_source, cases=rows,
        paired_canonical_source_identical=same_source, local_tokenizer=tokenizer,
        provider_calls=0, containers_started=0, target_code_executed=False,
        provider_usage_tokens=None, official_reward=None, content_bearing_messages_retained=False,
        measurement_scope="actual authored stdio wire traffic; per-component transmission and local tokenizer proxies",
        limitations=["Scripted tool use does not establish live model adoption, task solving or provider tokens.",
            "Complete inventory is explicitly requested and its pagination traffic is included in totals.",
            "Reconstructed inventory bytes are an in-memory diagnostic, not additional transmitted bytes.",
            "Cache files are classified by directory name, without source meaning or proof authority."])
    (output / "qualification.json").write_text(_render(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-files", type=int, action="append", dest="cache_counts")
    parser.add_argument("--status-repeats", type=int, default=4)
    parser.add_argument("--page-limit", type=int, default=32)
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(output=args.output, cache_counts=args.cache_counts or (0, 128),
        status_repeats=args.status_repeats, page_limit=args.page_limit, tokenizer_cache=args.tokenizer_cache)
    print(_render(dict(schema=SCHEMA, qualified=result["qualified"], provider_calls=0,
                       output=str(args.output))), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
