"""Real native capsule queries across stdio, source edits and freshness fences.

Authored source only: these tests invoke no providers, training or proof solvers.
"""
import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from test.api.test_repository_source_mcp import configuration, real_client


def query(client, *, generation=1, text="calc", **options):
    result, failed = client.call("repository_context_query", generation=generation,
                                 query=text, **options)
    assert not failed, result
    return result


def expand(client, context, *, candidates=None, **options):
    rows = context["candidates"] if candidates is None else candidates
    result, failed = client.call("repository_capsule_expand", generation=context["generation"],
        semantic_root_cid=context["semantic_root_cid"],
        stable_symbol_ids=[row["symbol_id"] for row in rows], **options)
    assert not failed, result
    return result


def wire_bytes(payload):
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode())


def assert_native_binding(context, expansion):
    for key in ("generation", "semantic_root_cid", "capsule_index_cid", "semantic_state_cid",
                "source_tree_sha256"):
        assert context[key] == expansion[key], key
    assert context["proof_authority"] is expansion["proof_authority"] is False


def test_actual_stdio_query_expand_edit_and_successor_roots(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"]) / "calc.py"
    before = source.read_bytes()
    with real_client(path, digest) as client:
        tools = {tool["name"] for tool in client.request("tools/list")["result"]["tools"]}
        assert {"repository_context_query", "repository_capsule_expand"} <= tools
        context = query(client, text="add")
        assert context["schema"] == "repository-source-context@1"
        assert context["ranking"] == "lexical_symbol_path"
        assert context["selected_count"] == len(context["candidates"]) == 1
        assert context["matching_count"] == 1
        assert "capsules" not in context and "admissions" not in context
        candidate = context["candidates"][0]
        assert candidate["path"] == "calc.py" and candidate["name"] == "calc.add"
        assert candidate["source_sha256"] == hashlib.sha256(before).hexdigest()
        assert candidate["confidence"] in {"exact", "conservative", "heuristic", "opaque"}
        assert type(candidate["raw_source_required"]) is bool
        assert "Original public comment" not in json.dumps(context)
        assert "return value" not in json.dumps(context)

        expansion = expand(client, context)
        assert expansion["schema"] == "repository-source-capsule-expansion@1"
        assert_native_binding(context, expansion)
        assert len(expansion["capsules"]) == len(expansion["admissions"]) == 1
        capsule = expansion["capsules"][0]
        assert capsule["stable_symbol_id"] == candidate["symbol_id"]
        assert capsule["capsule_cid"] == candidate["capsule_cid"]
        assert capsule["source_cid"] == candidate["source_cid"]
        assert capsule["confidence"] == candidate["confidence"]
        assert expansion["raw_source_fetch_required"]["calc.py"] == {
            "source_sha256": candidate["source_sha256"], "source_cid": candidate["source_cid"]}
        admission = expansion["admissions"][0]
        assert admission["freshness"] == "fresh"
        assert admission["ref"]["capsule_cid"] == candidate["capsule_cid"]
        # Independently reopen the native capsule model/CID, not merely labels.
        from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
        native = SemanticCapsule.from_dict(capsule)
        assert native.capsule_cid == candidate["capsule_cid"]

        exact, failed = client.call("repository_read", generation=1,
                                     path="calc.py", representation="original")
        assert not failed and exact["content"].encode() == before
        assert exact["source_sha256"] == candidate["source_sha256"]
        edited, failed = client.call("repository_replace_symbol", generation=1, path="calc.py",
            symbol="add", expected_sha256=exact["source_sha256"],
            replacement="def add(value):\n    return value + 10\n")
        assert not failed and edited["canonical_changed"] is True and edited["generation"] == 2
        assert b"Original public comment" in source.read_bytes()
        assert b"return value + 10" in source.read_bytes()

        stale, failed = client.call("repository_context_query", generation=1, query="add")
        assert failed and stale == {"error": "stale_generation"}
        stale, failed = client.call("repository_capsule_expand", generation=1,
            semantic_root_cid=context["semantic_root_cid"], stable_symbol_ids=[candidate["symbol_id"]])
        assert failed and stale == {"error": "stale_generation"}
        wrong_root, failed = client.call("repository_capsule_expand", generation=2,
            semantic_root_cid=context["semantic_root_cid"], stable_symbol_ids=[candidate["symbol_id"]])
        assert failed and wrong_root == {"error": "capsule_root_differs"}

        current = query(client, generation=2, text="add")
        assert current["semantic_root_cid"] != context["semantic_root_cid"]
        assert current["semantic_state_cid"] != context["semantic_state_cid"]
        assert current["source_tree_sha256"] != context["source_tree_sha256"]
        successor = expand(client, current)
        assert_native_binding(current, successor)
        assert current["candidates"][0]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert successor["capsules"][0]["capsule_cid"] != capsule["capsule_cid"]
        assert successor["raw_source_fetch_required"]["calc.py"]["source_sha256"] != candidate["source_sha256"]
    summary = mcp.summarize_repository_audit(config["audit_store"])
    assert summary["status"] == "observed" and summary["successful_reads"] == 1
    assert summary["canonical_edits"] == 1 and summary["errors"] == 3
    audit = Path(config["audit_store"]).read_text()
    assert "return value" not in audit and "calc.py" not in audit


def test_external_change_to_unrequested_source_invalidates_capsule_population(tmp_path):
    config, path, digest = configuration(tmp_path)
    workspace = Path(config["workspace"])
    dependency = workspace / "consumer.py"
    dependency.write_text("from calc import add\n\ndef use(value):\n    return add(value)\n")
    with real_client(path, digest) as client:
        context = query(client, text="calc")
        candidate = next(row for row in context["candidates"] if row["name"] == "calc.add")
        expand(client, context, candidates=[candidate])
        dependency.write_text("from calc import add\n\ndef use(value):\n    return add(value) + 2\n")
        stale, failed = client.call("repository_capsule_expand", generation=1,
            semantic_root_cid=context["semantic_root_cid"], stable_symbol_ids=[candidate["symbol_id"]])
        assert failed and stale == {"error": "stale_source"}
        stale, failed = client.call("repository_context_query", generation=1, query="calc")
        assert failed and stale["error"] in {"stale_source", "refresh_required"}
        refreshed, failed = client.call("repository_refresh", generation=1)
        assert not failed and refreshed["generation"] == 2
        current = query(client, generation=2, text="calc")
        assert current["semantic_root_cid"] != context["semantic_root_cid"]
        current_candidate = next(row for row in current["candidates"] if row["name"] == "calc.add")
        # Unchanged requested bytes do not make the old repository root current.
        assert current_candidate["source_sha256"] == candidate["source_sha256"]
        expansion = expand(client, current, candidates=[current_candidate])
        assert_native_binding(current, expansion)


def test_query_and_expansion_are_bounded_and_never_return_partial_expansions(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"]) / "calc.py"
    source.write_text("\n".join(f"def item_{number}(value):\n    return value + {number}\n" for number in range(6)))
    with real_client(path, digest) as client:
        whole = query(client, limit=32)
        # The native producer retains the module symbol as well as six
        # definitions; query coverage must account for that complete population.
        assert whole["population_count"] == whole["matching_count"] == whole["selected_count"] == 7
        assert whole["omitted_count"] == 0 and whole["truncated"] is False
        small = query(client, limit=2, max_bytes=32768)
        assert small["population_count"] == small["matching_count"] == 7
        assert small["selected_count"] == len(small["candidates"]) == 2
        assert small["omitted_count"] == 5 and small["truncated"] is True
        assert wire_bytes(small) <= 32768
        assert [row["symbol_id"] for row in small["candidates"]] == [row["symbol_id"] for row in whole["candidates"][:2]]
        expansion = expand(client, small, max_bytes=32768)
        assert len(expansion["capsules"]) == 2 and wire_bytes(expansion) <= 32768
        for tool, arguments in (
            ("repository_context_query", {"query": "calc " + "z" * 1100, "max_bytes": 1024}),
            ("repository_capsule_expand", {"semantic_root_cid": whole["semantic_root_cid"],
                "stable_symbol_ids": [row["symbol_id"] for row in small["candidates"]], "max_bytes": 1024}),
        ):
            refused, failed = client.call(tool, generation=1, **arguments)
            assert failed and refused == {"error": "capsule_response_limit"}

        from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
        unknown = cid_for_structured({"authored_symbol": "absent"})
        mixed, failed = client.call("repository_capsule_expand", generation=1,
            semantic_root_cid=whole["semantic_root_cid"],
            stable_symbol_ids=[small["candidates"][0]["symbol_id"], unknown])
        assert failed and mixed == {"error": "capsule_symbol_unavailable"}
        # Failed bounded/unknown requests do not disable ordinary reads or a
        # subsequent valid context query for the unchanged generation.
        read, failed = client.call("repository_read", generation=1, path="calc.py", representation="original")
        assert not failed and read["content"] == source.read_text()
        assert query(client, limit=2)["candidates"] == small["candidates"]


def test_no_match_and_opaque_only_repository_report_explicit_empty_populations(tmp_path):
    config, path, digest = configuration(tmp_path)
    workspace = Path(config["workspace"])
    (workspace / "calc.py").unlink()
    (workspace / "README.txt").write_text("Authored nonprogram data, not a proved requirement.\n")
    (workspace / "binary.dat").write_bytes(b"\xff\xfe\x00opaque")
    with real_client(path, digest) as client:
        for text in ("absent_symbol", "?!"):
            context = query(client, text=text)
            assert context["candidates"] == []
            assert context["population_count"] == context["matching_count"] == context["selected_count"] == 0
            assert context["omitted_count"] == 0 and context["truncated"] is False
            assert context["proof_authority"] is False
        unknown, failed = client.call("repository_capsule_expand", generation=1,
            semantic_root_cid=context["semantic_root_cid"], stable_symbol_ids=[])
        assert failed and set(unknown) == {"error"}
        files, failed = client.call("repository_list", generation=1)
        assert not failed and {row["path"] for row in files["entries"]} == {"README.txt", "binary.dat"}


def test_unknown_query_and_invalid_requests_preserve_complete_inventory(tmp_path):
    config, path, digest = configuration(tmp_path)
    with real_client(path, digest) as client:
        missing = query(client, text="nonexistent_symbol")
        assert missing["population_count"] == 2 and missing["matching_count"] == 0
        assert missing["selected_count"] == 0 and missing["candidates"] == []
        known = query(client)
        identity = known["candidates"][0]["symbol_id"]
        for arguments in ({"query": ""}, {"query": "x" * 4097}, {"query": "é" * 2049},
                          {"query": "calc", "limit": True}, {"query": "calc", "limit": 33},
                          {"query": "calc", "max_bytes": 65537}):
            refused, failed = client.call("repository_context_query", generation=1, **arguments)
            assert failed and set(refused) == {"error"}
        for ids in ([identity, identity], [identity] * 33):
            refused, failed = client.call("repository_capsule_expand", generation=1,
                semantic_root_cid=known["semantic_root_cid"], stable_symbol_ids=ids)
            assert failed and set(refused) == {"error"}
        assert query(client)["candidates"] == known["candidates"]
