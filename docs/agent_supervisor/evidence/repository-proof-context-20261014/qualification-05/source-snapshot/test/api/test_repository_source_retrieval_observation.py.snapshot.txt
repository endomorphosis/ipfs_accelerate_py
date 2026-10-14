"""Retained retrieval evidence with real local producers, runner and stdio.

Provider output is authored. These tests establish delivery and stale-handle
refusal, not live model adoption, proof, task correctness or token savings.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as benchmark
from test.api.test_repository_source_mcp import real_client
from test.api.test_repository_source_router import route, _run
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke


def payload(*, generation=1, count=1, expansion=False):
    return dict(generation=generation, semantic_root_cid="retained-native-root",
        semantic_state_cid="retained-native-state", source_tree_sha256="1" * 64,
        **{"capsules" if expansion else "candidates": [{} for _ in range(count)]})


def record(audit, tool, *, success=True, value=None, generation=1, legacy=False):
    value = payload(expansion=tool == "repository_capsule_expand") if value is None else value
    event = None if legacy else mcp._retrieval_event(tool, {"generation": generation}, value, success)
    audit.record("tool", tool=tool, success=success, returned_bytes=128, retrieval=event)


def journal(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "operations.jsonl"
    audit = mcp._Audit(path)
    audit.record("initialize")
    return path, audit


def test_summary_distinguishes_empty_delivery_errors_and_generation_bindings(tmp_path):
    path, audit = journal(tmp_path)
    record(audit, "repository_context_query", value=payload(count=0))
    record(audit, "repository_context_query", value=payload(count=2))
    record(audit, "repository_capsule_expand", value=payload(count=2, expansion=True))
    for error in ("stale_generation", "stale_source", "capsule_root_differs", "tool_failed"):
        record(audit, "repository_capsule_expand", success=False, value={"error": error})
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    observed = summary["retrieval_observation"]
    assert observed["complete"] is True
    assert observed["query_calls"] == observed["successful_queries"] == 2
    assert observed["empty_queries"] == observed["nonempty_queries"] == 1
    assert observed["candidates_returned"] == observed["capsules_returned"] == 2
    assert observed["expansion_calls"] == 5 and observed["successful_expansions"] == 1
    assert observed["empty_expansions"] == 0 and observed["nonempty_expansions"] == 1
    assert observed["generation_bindings_observed"] == 3 and observed["errors"] == 4
    assert all(observed[key] == 1 for key in (
        "stale_generation_errors", "stale_source_errors", "root_mismatch_errors"))
    assert all(observed[key] is False for key in mcp.RETRIEVAL_FALSE)
    raw = path.read_text()
    assert "retained-native-root" not in raw and "retained-native-state" not in raw
    assert mcp.validate_retrieval_observation(observed, tool_calls=7, errors=4) == observed


@pytest.mark.parametrize("mixed", [False, True])
def test_historical_retrieval_remains_unknown_or_explicitly_incomplete(tmp_path, mixed):
    path, audit = journal(tmp_path)
    record(audit, "repository_context_query", legacy=True)
    if mixed:
        record(audit, "repository_capsule_expand")
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    assert summary["status"] == "observed"
    if mixed:
        observed = summary["retrieval_observation"]
        assert observed["legacy_retrieval_calls"] == 1 and observed["complete"] is False
    else:
        assert "retrieval_observation" not in summary


def test_unbound_success_does_not_claim_complete_retrieval(tmp_path):
    path, audit = journal(tmp_path)
    record(audit, "repository_capsule_expand", value={"capsules": [{}]})
    audit.close()
    observed = mcp.summarize_repository_audit(path)["retrieval_observation"]
    assert observed["successful_expansions"] == observed["unbound_successes"] == 1
    assert observed["capsules_returned"] == observed["generation_bindings_observed"] == 0
    assert observed["complete"] is False


@pytest.mark.parametrize("kind", ["read", "query", "expansion", "legacy_mix", "unconfigured", "unaudited"])
def test_benchmark_observes_expansion_separately_from_reads_and_omission_claims(tmp_path, kind):
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_policy import dispatch_report

    path, audit = journal(tmp_path)
    if kind == "read":
        audit.record("tool", tool="repository_read", returned_bytes=128)
    elif kind == "query":
        record(audit, "repository_context_query")
    else:
        if kind == "legacy_mix":
            record(audit, "repository_context_query", legacy=True)
        record(audit, "repository_capsule_expand")
    audit.close()
    report = dispatch_report(policy.POLICIES[1])
    row = report["provider_invocations"][-1]
    row["repository_source_policy"]["usage"] = {**mcp.summarize_repository_audit(path),
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}
    if kind == "unconfigured": row["usage"]["codex_repository_source_configured"] = False
    if kind == "unaudited": report["context_input_audit"]["matches"] = []
    observation = benchmark.repository_source_dispatch_observation(report, policy.POLICIES[1])["invocations"][0]
    assert observation.get("capsule_retrieval_observed", False) is (kind == "expansion")
    assert observation["actual_reads_observed"] is (kind == "read")
    if "retrieval_observation" in observation:
        assert observation["retrieval_observation"]["omitted_capsule_recovery_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("capsules_returned", 33), ("successful_expansions", True), ("generation_bindings_observed", 0),
    ("empty_expansions", 1), ("errors", 1), ("complete", False),
    ("omitted_capsule_recovery_verified", True), ("event_chain_sha256", "foreign"),
])
def test_forged_retrieval_aggregate_cannot_qualify(tmp_path, field, value):
    path, audit = journal(tmp_path)
    record(audit, "repository_capsule_expand")
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    changed = deepcopy(summary["retrieval_observation"])
    changed[field] = value
    with pytest.raises(ValueError):
        mcp.validate_retrieval_observation(changed, tool_calls=1, errors=0)
    usage = {**summary, "retrieval_observation": changed,
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}
    observed = benchmark._usage_observation(usage)
    assert observed["server_audit_valid"] is True
    assert observed["retrieval_observation_valid"] is False
    assert observed["retrieval_observation"] is None


@pytest.mark.parametrize("tool_calls,errors", [(-1, 0), (True, 0), (513, 0), (1, -1), (1, True), (1, 3), (1, 2)])
def test_retrieval_validator_bounds_global_population(tmp_path, tool_calls, errors):
    path, audit = journal(tmp_path)
    record(audit, "repository_capsule_expand")
    audit.close()
    retrieval = mcp.summarize_repository_audit(path)["retrieval_observation"]
    with pytest.raises(ValueError, match="invalid_retrieval_observation"):
        mcp.validate_retrieval_observation(retrieval, tool_calls=tool_calls, errors=errors)


@pytest.mark.parametrize("changes", [{"errors": 1}, {"successful_reads": 1}, {"refreshes": 1}])
def test_retrieval_cannot_reuse_failed_read_or_refresh_call_population(tmp_path, changes):
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_policy import dispatch_report

    path, audit = journal(tmp_path)
    record(audit, "repository_capsule_expand")
    audit.close()
    usage = {**mcp.summarize_repository_audit(path), **changes,
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}
    observed = benchmark._usage_observation(usage)
    assert observed["server_audit_valid"] is True
    assert observed["retrieval_observation_valid"] is False
    assert observed["retrieval_observation"] is None
    report = dispatch_report(policy.POLICIES[1])
    report["provider_invocations"][-1]["repository_source_policy"]["usage"] = usage
    invocation = benchmark.repository_source_dispatch_observation(report, policy.POLICIES[1])["invocations"][0]
    assert invocation.get("capsule_retrieval_observed", False) is False


def test_distinct_read_refresh_and_retrieval_successes_remain_valid(tmp_path):
    path, audit = journal(tmp_path)
    record(audit, "repository_context_query")
    record(audit, "repository_capsule_expand")
    audit.record("tool", tool="repository_read", returned_bytes=128)
    audit.record("tool", tool="repository_refresh", refreshed=True, returned_bytes=128)
    record(audit, "repository_context_query", success=False, value={"error": "stale_generation"})
    audit.close()
    observed = benchmark._usage_observation({**mcp.summarize_repository_audit(path),
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False})
    assert observed["server_audit_valid"] is True
    assert observed["retrieval_observation_valid"] is True


@pytest.mark.parametrize("change", ["boolean_count", "missing_binding", "foreign_field", "wrong_tool"])
def test_malformed_retrieval_journal_is_unavailable(tmp_path, change):
    path, audit = journal(tmp_path)
    record(audit, "repository_capsule_expand")
    audit.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    event = rows[-1]["retrieval"]
    if change == "boolean_count": event["returned_items"] = True
    elif change == "missing_binding": event["binding_sha256"] = None
    elif change == "foreign_field": event["source_text"] = "must not be accepted"
    else: rows[-1]["tool"] = "repository_read"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = mcp.summarize_repository_audit(path)
    assert summary["status"] == "unavailable"
    assert "retrieval_observation" not in summary


def test_real_source_read_alone_does_not_qualify_capsule_retrieval(route):
    _workspace, state = route
    def read(binding):
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            value, failed = client.call("repository_read", generation=1, path="module.py", representation="original")
            assert not failed and "square" in value["content"]
    state["callback"] = read
    _, receipt = _run(repository_source_policy=policy.POLICIES[1])
    usage = receipt["repository_source_policy"]["usage"]
    assert usage["successful_reads"] == 1
    assert "retrieval_observation" not in usage
    assert "retrieval_observation_valid" not in benchmark._usage_observation(usage)


@pytest.fixture
def native(tmp_path, request):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from benchmarks.agent_supervisor.container_coding.terminal_source_input_experiment import _native_prompt

    root = tmp_path / "repository"
    root.mkdir()
    source = "# Authored local fixture.\n" + "\n".join(
        f"def item_{index}(value):\n    return value + {index}\n" for index in range(6))
    (root / "mod.py").write_text(source)
    (root / "instruction.md").write_text("Inspect the selected function in this authored local repository.\n")
    # dispatch_case commits mod.py; stage the separate required input here so
    # it is also present in the allocated worktree without staging artifacts.
    for argv in (["init", "-q"], ["add", "instruction.md"]):
        subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True)
    artifact = root / ".runtime/semantic/worker-context.json"
    prepared = prepare_semantic_context(repository=root, paths=["mod.py", "instruction.md"], program_paths=["mod.py"],
        required_raw_paths=["instruction.md"], objective="Inspect item_0", task_id="RETRIEVAL-TEST", output=artifact.parent,
        worker_query="item_0", worker_selector="duckdb-query-symbol-name@1", worker_capsule_limit=request.param,
        worker_max_bytes=32768, context_input_tokens=8192)
    assert prepared["worker_capsules"] == request.param < prepared["capsules"]
    return root, artifact, _native_prompt(root, artifact, "RETRIEVAL-TEST", ["mod.py", "instruction.md"]), source


@pytest.mark.parametrize("native", [1, 2], indirect=True)
def test_real_runner_recovers_initially_omitted_symbol_and_refuses_stale_handles(
        dispatch_case, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as context_audit

    case = dispatch_case
    initial = json.loads((case["repository"] / ".runtime/semantic/worker-context.json").read_text())
    selected = {row["qualified_name"] for row in initial["worker_projection"]["selected_symbols"]}
    target = next(f"item_{index}" for index in range(6) if f"mod.item_{index}" not in selected)
    original = (case["workspace"] / "mod.py").read_bytes()
    original_generate = llm_router.generate_text
    delivered = []

    def generate(prompt, **kwargs):
        binding = kwargs["codex_repository_source_binding"]
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            empty, failed = client.call("repository_context_query", generation=1, query="absent_symbol")
            assert not failed and empty["selected_count"] == 0
            context, failed = client.call("repository_context_query", generation=1, query=target, limit=1)
            assert not failed and context["selected_count"] == 1
            candidate = context["candidates"][0]
            assert candidate["name"] == "mod." + target and candidate["path"] == "mod.py"
            assert candidate["source_sha256"] == hashlib.sha256(original).hexdigest()
            # The session owns a distinct index identity; join by native symbol
            # identity and exact source bytes instead of reusing initial CIDs.
            assert context["semantic_root_cid"] != initial["semantic_root_cid"]
            args = dict(generation=1, semantic_root_cid=context["semantic_root_cid"],
                stable_symbol_ids=[candidate["symbol_id"]])
            expanded, failed = client.call("repository_capsule_expand", **args)
            assert not failed and len(expanded["capsules"]) == 1
            assert expanded["capsules"][0]["capsule_cid"] == candidate["capsule_cid"]
            assert expanded["semantic_root_cid"] == context["semantic_root_cid"]
            read, failed = client.call("repository_read", generation=1, path="mod.py", representation="original")
            assert not failed and read["content"].encode() == original
            assert read["source_sha256"] == candidate["source_sha256"]
            delivered.append(candidate["name"])
            (case["workspace"] / "mod.py").write_bytes(original + b"# External local edit.\n")
            stale, failed = client.call("repository_capsule_expand", **args)
            assert failed and stale["error"] == "stale_source"
            refreshed, failed = client.call("repository_refresh", generation=1)
            assert not failed and refreshed["generation"] == 2
            stale, failed = client.call("repository_context_query", generation=1, query=target)
            assert failed and stale["error"] == "stale_generation"
            stale, failed = client.call("repository_capsule_expand", **{**args, "generation": 2})
            assert failed and stale["error"] == "capsule_root_differs"
            current, failed = client.call("repository_context_query", generation=2, query=target, limit=1)
            assert not failed and current["semantic_root_cid"] != context["semantic_root_cid"]
            expanded, failed = client.call("repository_capsule_expand", generation=2,
                semantic_root_cid=current["semantic_root_cid"],
                stable_symbol_ids=[current["candidates"][0]["symbol_id"]])
            assert not failed and expanded["generation"] == 2 and len(expanded["capsules"]) == 1
        output = original_generate(prompt, **kwargs)
        observation = get_last_cli_observation("codex_cli")
        observation.update(codex_repository_source_configured=True,
            codex_repository_source_config_sha256=binding["config_sha256"])
        set_last_cli_observation("codex_cli", observation)
        return output

    monkeypatch.setattr(llm_router, "generate_text", generate)
    _, receipt = invoke(case, repository_source_policy=policy.POLICIES[1])
    assert delivered == ["mod." + target] and len(case["calls"]) == 1
    usage = receipt["repository_source_policy"]["usage"]
    retrieval = usage["retrieval_observation"]
    assert retrieval["complete"] is True and retrieval["query_calls"] == 4
    assert retrieval["successful_queries"] == 3 and retrieval["empty_queries"] == 1
    assert retrieval["nonempty_queries"] == retrieval["candidates_returned"] == 2
    assert retrieval["expansion_calls"] == 4
    assert retrieval["successful_expansions"] == retrieval["nonempty_expansions"] == retrieval["capsules_returned"] == 2
    assert retrieval["generation_bindings_observed"] == 5 and retrieval["errors"] == 3
    assert all(retrieval[key] == 1 for key in ("stale_generation_errors", "stale_source_errors", "root_mismatch_errors"))
    assert all(retrieval[key] is False for key in mcp.RETRIEVAL_FALSE)
    assert usage["store_cleanup_complete"] is True and not Path(receipt["repository_source_policy"]["invocation_dir"]).exists()
    assert benchmark._usage_observation(usage)["retrieval_observation_valid"] is True
    parsed = context_audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    replayed, _, checks = context_audit._model_projection(
        rendered=case["prompt"], receipt=parsed, repository=case["repository"])
    assert replayed == case["calls"][0][0] and all(checks.values())
    assert target not in json.dumps(retrieval) and original.decode() not in json.dumps(receipt)
