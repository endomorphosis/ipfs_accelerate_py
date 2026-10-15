"""Authored paired source lookup and native capsule workflows over real MCP.

Only disposable authored source is inspected and edited. No target code,
provider, container, proof search or model training runs. Reports retain byte
counts, local token proxies and checks, never source or capsule bodies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .terminal_repository_tools_experiment import _Client, _sha, _wire, _TOOL_COMPONENTS
from .terminal_source_view_experiment import _render, local_token_counter

SCHEMA = "terminal-repository-capsule-experiment@1"
ARMS = ("full_source", "native_symbol_targeted", "context_targeted", "capsule_targeted")
QUERY = "compute_invoice_total"
TARGET = "ledger.py"
ORIGINAL_DEFINITION = "def compute_invoice_total(subtotal, tax):\n    return subtotal + tax\n"
CHANGED_DEFINITION = "def compute_invoice_total(subtotal, tax):\n    return subtotal + tax + 1\n"
IDENTITIES = ("semantic_root_cid", "capsule_index_cid", "semantic_state_cid",
              "source_tree_sha256")


def authored_sources():
    return {
        TARGET: ("# Authored invoice calculation; preserve this comment and original bytes.\n"
                 + ORIGINAL_DEFINITION).encode(),
        "display.py": ("# Authored display examples, unrelated to invoice arithmetic.\n" * 12
                       + "def display_label(value):\n    return str(value)\n").encode(),
        "storage.py": ("# Authored storage examples, unrelated to invoice arithmetic.\n" * 12
                       + "def storage_key(value):\n    return str(value)\n").encode(),
    }


def _delta(before, after):
    def subtract(left, right):
        return {key: None if left[key] is None or value is None else value - left[key]
                for key, value in right.items()}
    return {"total": subtract(before["total"], after["total"]), "components": {
        key: subtract(before["components"][key], row) for key, row in after["components"].items()}}


def _query(client, generation):
    return client.tool("repository_context_query", dict(generation=generation, query=QUERY, limit=1))


def _expand(client, query, *, generation=None, expect_error=False):
    return client.tool("repository_capsule_expand", dict(
        generation=query["generation"] if generation is None else generation,
        semantic_root_cid=query["semantic_root_cid"],
        stable_symbol_ids=[query["candidates"][0]["symbol_id"]]), expect_error=expect_error)


def _validate_candidate(query, source):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    candidate = query["candidates"][0]
    return (len(query["candidates"]) == query["selected_count"] == 1
        and candidate["path"] == TARGET and candidate["name"].endswith("." + QUERY)
        and candidate["source_sha256"] == _sha(source)
        and candidate["source_cid"] == cid_for_bytes(source)
        and query["ranking"] == "lexical_symbol_path")


def _validate_context(query, expanded, source):
    """Reverify native capsule identity and original-source binding in memory."""
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
    candidate = query["candidates"][0]
    capsules = expanded["capsules"]
    # The native producer's serialized capsule retains its own CID. Parsing it
    # recomputes that identity, instead of accepting a decorative CID string.
    capsule = SemanticCapsule.from_dict(capsules[0])
    return dict(
        bounded_exact_candidate=_validate_candidate(query, source),
        native_capsule_identity=(len(capsules) == 1
            and capsule.stable_symbol_id == candidate["symbol_id"]
            and capsule.capsule_cid == candidate["capsule_cid"]
            and capsule.source_cid == candidate["source_cid"]
            and expanded["requested_stable_symbol_ids"] == [candidate["symbol_id"]]),
        same_generation_and_native_root=(query["generation"] == expanded["generation"]
            and all(query[key] == expanded[key] for key in IDENTITIES)),
        exact_required_source_binding=expanded["raw_source_fetch_required"].get(TARGET) == {
            "source_sha256": _sha(source), "source_cid": cid_for_bytes(source)},
        native_admissions_retained=bool(expanded["admissions"]),
        structural_authority_only=all(row["authority"] == "structural_only"
            and all(row[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
            for row in (query, expanded)),
    )


def _duckdb_head_matches(path, head):
    """Read the closed authored native database, with bounded SQL resources."""
    import duckdb
    fields = ("repository_id", "generation", "manifest_cid", "snapshot_cid", "ast_revision_id", "receipt_cid")
    with duckdb.connect(str(path), read_only=True, config={"threads": "1", "memory_limit": "128MB"}) as connection:
        connection.execute("SET enable_external_access = false")
        rows = connection.execute("SELECT repository_id,generation,manifest_cid,snapshot_cid,ast_revision_id,receipt_cid "
                                  "FROM codebase_control.heads LIMIT 2").fetchall()
    return len(rows) == 1 and dict(zip(fields, rows[0])) == {key: head[key] for key in fields}


def _arm(*, mode, counter):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_mcp import summarize_repository_audit
    sources = authored_sources()
    checks, failure, lookup, final_hashes, final_head = {}, None, None, {}, None
    with tempfile.TemporaryDirectory(prefix="repository-capsule-benchmark-", dir="/tmp") as temporary:
        root = Path(temporary)
        workspace, private = root / "workspace", root / "private"
        workspace.mkdir()
        for name, raw in sources.items():
            (workspace / name).write_bytes(raw)
        private.mkdir(mode=0o700)
        (private / "state").mkdir(mode=0o700)
        config = private / "config.json"
        config.write_bytes(_wire(dict(schema="repository-source-mcp-config@1", workspace=str(workspace),
            state_dir=str(private / "state"), audit_store=str(private / "operations.jsonl"), representation="original")))
        config.chmod(0o600)
        client = _Client(config, counter)
        try:
            initialized = client.request("initialize", dict(protocolVersion="2025-06-18", capabilities={},
                clientInfo={"name": "authored-capsule-benchmark", "version": "1"}), phase="setup")
            checks["initialized"] = initialized["serverInfo"]["name"] == "repository_source"
            client.request("notifications/initialized", phase="setup", notification=True)
            tools = client.request("tools/list", phase="setup")
            checks["complete_tool_inventory"] = {row["name"] for row in tools["tools"]} == set(_TOOL_COMPONENTS)
            status = client.tool("repository_status", phase="setup")
            generation = status["generation"]
            checks["native_scope_current"] = status["source_current"] is True and status["native_index"]["source_scope_verified"] is True
            before = client.metrics()
            lookup_symbol_id = None
            if mode == "full_source":
                listed = client.tool("repository_list", dict(generation=generation, limit=100))
                paths = [row["path"] for row in listed["entries"] if row["kind"] == "file"]
                checks["lookup_population_exact"] = set(paths) == set(sources) and listed["next_offset"] is None
                reads = {name: client.tool("repository_read", dict(generation=generation, path=name)) for name in paths}
                checks["lookup_sources_exact"] = all(row["content"].encode() == sources[name] for name, row in reads.items())
                target_read = reads[TARGET]
            elif mode == "native_symbol_targeted":
                symbols = client.tool("repository_symbols", dict(generation=generation, query=QUERY, limit=1))
                candidate = symbols["symbols"][0]
                checks["lookup_candidate_exact"] = (len(symbols["symbols"]) == 1 and candidate["path"] == TARGET
                    and candidate["name"].endswith("." + QUERY))
                lookup_symbol_id = candidate["symbol_id"]
                target_read = client.tool("repository_read", dict(generation=generation, path=candidate["path"]))
                checks["lookup_sources_exact"] = target_read["content"].encode() == sources[TARGET]
            elif mode in {"context_targeted", "capsule_targeted"}:
                selected = _query(client, generation)
                checks["lookup_candidate_exact"] = _validate_candidate(selected, sources[TARGET])
                lookup_symbol_id = selected["candidates"][0]["symbol_id"]
                if mode == "capsule_targeted":
                    expanded = _expand(client, selected)
                    checks.update({"lookup_" + name: result for name, result in _validate_context(selected, expanded, sources[TARGET]).items()})
                target_read = client.tool("repository_read", dict(generation=generation, path=selected["candidates"][0]["path"]))
                checks["lookup_sources_exact"] = target_read["content"].encode() == sources[TARGET]
            else:
                raise ValueError("unknown authored comparison arm")
            checks["lookup_target_hash_bound"] = target_read["source_sha256"] == _sha(sources[TARGET])
            lookup = _delta(before, client.metrics())
            # Identical contract qualification follows either lookup. Its full
            # traffic is charged separately from the lookup-only comparison.
            selected = _query(client, generation)
            expanded = _expand(client, selected)
            checks["lookup_same_capability"] = lookup_symbol_id is None or lookup_symbol_id == selected["candidates"][0]["symbol_id"]
            checks.update(_validate_context(selected, expanded, sources[TARGET]))
            checks["context_matches_status"] = (
                selected["semantic_state_cid"] == status["native_index"]["semantic_state_cid"]
                and selected["source_tree_sha256"] == status["source_tree_sha256"])
            edit = client.tool("repository_replace_symbol", dict(generation=generation, path=TARGET,
                symbol=QUERY, expected_sha256=target_read["source_sha256"], replacement=CHANGED_DEFINITION))
            checks["canonical_edit_refreshed"] = edit["canonical_changed"] is True and edit["refresh_required"] is False
            stale = _expand(client, selected, expect_error=True)
            checks["old_generation_refused"] = stale["error"] == "stale_generation"
            wrong_root = _expand(client, selected, generation=edit["generation"], expect_error=True)
            checks["old_root_refused"] = wrong_root["error"] == "capsule_root_differs"
            changed = _query(client, edit["generation"])
            changed_expansion = _expand(client, changed)
            current_bytes = (workspace / TARGET).read_bytes()
            checks.update({"edited_" + key: value for key, value in _validate_context(changed, changed_expansion, current_bytes).items()})
            checks["edited_capsule_changes"] = (changed["semantic_root_cid"] != selected["semantic_root_cid"]
                and changed["candidates"][0]["capsule_cid"] != selected["candidates"][0]["capsule_cid"])
            current = client.tool("repository_read", dict(generation=edit["generation"], path=TARGET))
            checks["edited_original_exact"] = current["content"].encode() == current_bytes
            restored = client.tool("repository_replace_symbol", dict(generation=edit["generation"], path=TARGET,
                symbol=QUERY, expected_sha256=current["source_sha256"], replacement=ORIGINAL_DEFINITION))
            checks["canonical_restore_refreshed"] = restored["canonical_changed"] is True and restored["refresh_required"] is False
            refreshed = client.tool("repository_refresh", dict(generation=restored["generation"]))
            stale = client.tool("repository_context_query", dict(generation=restored["generation"], query=QUERY), expect_error=True)
            checks["refreshed_query_generation_refused"] = stale["error"] == "stale_generation"
            final_context = _query(client, refreshed["generation"])
            final_expansion = _expand(client, final_context)
            checks.update({"restored_" + key: value for key, value in _validate_context(final_context, final_expansion, sources[TARGET]).items()})
            final_status = client.tool("repository_status")
            final_head = final_status["datasets_head"]
            checks["final_native_head_current"] = (final_head == final_status["native_index"]["head"]
                and final_status["source_current"] is True
                and final_context["source_tree_sha256"] == final_status["source_tree_sha256"]
                and final_context["semantic_state_cid"] == final_status["native_index"]["semantic_state_cid"])
            checks["all_original_bytes_restored"] = all((workspace / name).read_bytes() == raw for name, raw in sources.items())
            final_hashes = {name: _sha((workspace / name).read_bytes()) for name in sources}
        except Exception as error:
            failure = {"error_type": type(error).__name__}
            checks["workflow_completed"] = False
            client._incomplete("workflow_incomplete")
        finally:
            exit_code = client.close()
        protocol = client.metrics()
        audit = summarize_repository_audit(private / "operations.jsonl")
        checks["clean_subprocess_exit"] = exit_code == 0
        checks["wire_accounting_complete"] = protocol["wire_accounting_complete"]
        checks["component_totals_match"] = protocol["component_totals_match"]
        checks["native_duckdb_head_persisted"] = final_head is not None and _duckdb_head_matches(private / "state/index/native.duckdb", final_head)
        checks["actual_native_operations_observed"] = (audit["initializations"] == 1 and audit["canonical_edits"] == 2
            and audit["refreshes"] == 3 and audit["errors"] == 3
            and audit["tool_calls"] == sum(protocol["components"][name]["requests"] for name in _TOOL_COMPONENTS.values())
            and audit["successful_reads"] == (4 if mode == "full_source" else 2))
        result = dict(mode=mode, checks=checks, failure=failure, lookup=lookup, protocol=protocol,
                      audit=audit, final_source_hashes=final_hashes, content_bearing_messages_retained=False)
    checks["private_fixture_removed"] = not root.exists()
    result["qualified"] = all(checks.values())
    return result


def run_experiment(*, output, tokenizer_cache=None):
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("capsule experiment output must be a new canonical directory")
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output.mkdir(parents=True)
    arms = {}
    for mode in ARMS:
        try:
            arms[mode] = _arm(mode=mode, counter=counter)
        except Exception as error:
            arms[mode] = dict(qualified=False, error_type=type(error).__name__, content_bearing_messages_retained=False)
    paired = all(row["qualified"] for row in arms.values()) and all(
        arms[ARMS[0]]["final_source_hashes"] == row["final_source_hashes"] for row in arms.values())
    result = dict(schema=SCHEMA, qualified=paired, arms=arms, paired_final_sources_identical=paired,
        local_tokenizer=tokenizer, representation="original", fixture_files=len(authored_sources()),
        provider_calls=0, containers_started=0, target_code_executed=False, training_executed=False,
        proof_search_executed=False, provider_usage_tokens=None, official_reward=None,
        complete_multi_ir_qualified=False, content_bearing_messages_retained=False,
        measurement_scope="authored capability lookup plus identical edit/restore qualification; all actual stdio traffic charged",
        limitations=["Lookup counters are a subset of full traffic, never an extra cost to add to totals.",
            "Four fixed-source arms separate exhaustive reads, native symbol selection, context references, and full capsule expansion; no token winner is assumed.",
            "Lexical symbol selection and structural native capsules do not prove behavior or cover complete intent/formal/codebase IR.",
            "The scripted fixture does not establish model adoption, task success or provider-token savings.",
            "Native DuckDB/CAS work and local capsule revalidation are host computation, not extra model transmissions.",
            "The optional tokenizer is a local wire proxy and does not enable token-aware source selection."])
    (output / "qualification.json").write_text(_render(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(output=args.output, tokenizer_cache=args.tokenizer_cache)
    print(_render(dict(schema=SCHEMA, qualified=result["qualified"], output=str(args.output), provider_calls=0)), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
