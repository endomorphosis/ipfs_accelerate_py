"""Native obligation summaries reach both routers without learned model calls.

The grammar, source replay, DuckLake owners, planner admission, worker context
compiler, and router boundary are real. Only planner/provider responses are
authored fixtures; this does not establish task success or numerical proofs.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as resources
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as cache
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_policy import seal
from benchmarks.agent_supervisor.container_coding.test_terminal_setup_cache_advice import selected_cache, selected_cache_v2  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as worker_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
from test.api.test_semantic_router_integration import provider  # noqa: F401


SOURCE = "import numpy as np\n\n" + "\n".join(
    f"def candidate_{index}(matrix):\n"
    "    values, vectors = np.linalg.eig(matrix)\n"
    "    chosen = np.argmax(np.abs(values))\n"
    "    return values[chosen], vectors[:, chosen]\n"
    for index in range(4)
)
RETRIEVAL_API = (
    "ipfs_accelerate_py.agent_supervisor.runtime."
    "numerical_formalization_context.retrieve_numerical_obligations"
)


@pytest.fixture
def selected(original, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    root, instruction, state = original
    (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    initial = prep.initial_context(state=state, numerical_formalization_profile=numerical.PROFILE)
    return root, state, prepared, initial["numerical_formalization"]


def _retained_obligations(receipt):
    output = Path(receipt["output"])
    index = json.loads((output / "reports.json").read_bytes())
    return [row for ref in index.values()
            for row in json.loads((output / ref["artifact"]).read_bytes())["pending_obligations"]]


def _assert_discovery(receipt):
    summary = receipt["summary"]
    obligations = _retained_obligations(receipt)
    expected = dict(sorted(Counter(row["kind"] for row in obligations).items()))
    assert summary["pending_obligation_kinds"] == expected
    assert sum(expected.values()) == summary["pending_obligation_count"] == len(obligations)
    assert summary["omitted_obligation_summaries"] > 0
    assert len(summary["pending_obligations"]) + summary["omitted_obligation_summaries"] == len(obligations)
    assert summary["obligation_retrieval"] == {
        "schema": "supervisor-numerical-obligation-retrieval@1",
        "api": RETRIEVAL_API,
        "isolated_worker_command": ("/opt/ipfs-supervisor/venv/bin/python -I "
            "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/"
            "numerical_obligation_client.py --path <source_path>"),
        "isolated_worker_usage": ("Run from the allocated workspace while the coding session is active. "
            "Optional: --function-id, --kind, --limit, --max-bytes, --cursor '<next_cursor JSON>'. "
            "Responses include full pending rows; source edits invalidate retrieval."),
        "requires_validated_context_receipt": True,
        "max_page_bytes": 8192,
        "max_page_rows": 8,
        "source_paths": ["bottle.py"],
        "omitted_source_paths": 0,
    }
    assert len(numerical._bytes(summary)) <= 8192
    assert summary["source_semantics_verified"] is summary["proof_authority"] is False
    assert summary["source_obligations_discharged"] == 0
    return summary


def test_deterministic_obligation_profile_can_share_the_existing_formalization_cache(selected_cache_v2):
    archive, manifest, selection, auth = selected_cache_v2
    profile = resources.FORMALIZATION_SOURCE384_PROFILE
    config = benchmark.config_for(Path("/dataset"), Path("/output"), archive, "full",
        resource_profile=profile, setup_cache_selection=selection,
        numerical_formalization_profile=numerical.PROFILE)
    cache.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth,
        arm="full", resource_profile=profile)
    assert cache.validate_setup_cache_selection(archive, selection) == (manifest, manifest["setup_cache"])
    assert manifest.get("numerical_selection_384") is None
    assert config["agents"][0]["kwargs"]["numerical_formalization_profile"] == numerical.PROFILE
    assert config["environment"]["override_cpus"] == 5
    assert config["environment"]["override_memory_mb"] == 20480
    declaration = seal(config)
    assert declaration["schema"] == controls.NUMERICAL_SCHEMA
    assert controls.validate_controls(declaration)
    absent = deepcopy(config)
    del absent["agents"][0]["kwargs"]["numerical_formalization_profile"]
    assert not controls.compare_controls(
        dict(status="observed", configuration_unchanged=True, declared=declaration),
        dict(status="observed", configuration_unchanged=True, declared=seal(absent)))["matches"]


def test_exact_obligation_discovery_reaches_planner_and_required_coding_dispatch(selected, provider, monkeypatch):
    root, state, prepared, formal = selected
    summary = _assert_discovery(formal)
    calls = []

    def planner(prompt, **kwargs):
        calls.append(prompt)
        wire = json.loads(prompt)
        actual = [row for row in wire["constraints"]["constraint_summaries"]
                  if isinstance(row, dict) and row.get("schema") == summary["schema"]]
        assert actual == [summary]
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)

    _version(monkeypatch)
    assert prep.plan(state=state, provider_callable=planner)["qualified"] is True
    assert len(calls) == 1
    context = prep.context(state=state)
    assert context["initial_indexes_reused"] is True
    assert context["numerical_formalization"] == formal
    bundle = context["context_bundle"]
    task = json.loads((root / bundle["artifact"]).read_bytes())["tasks"][0]
    daemon = TodoImplementationDaemon(todo_path=root / ".runtime/numerical.todo.md", repo_root=root,
        state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
        events_path=state / "worker/events.jsonl", task_header_prefix="## TB-",
        implementation_log_dir=state / "launch/implementation-logs")
    daemon._task_context_nomination_bundle = bundle
    portal = PortalTask(task_id=task["task_id"], canonical_task_cid=task["task_cid"],
        title="Numerical obligation discovery", status="ready", completion="manual", priority="P2",
        track="implementation", outputs=["bottle.py", "report.jsonl"], validation=[],
        acceptance="Observe the exact task-bound numerical obligation discovery.",
        metadata={"database task cid": task["task_cid"]})
    try:
        prompt = daemon._build_implementation_prompt(portal, attempt=1)
        daemon._persist_implementation_context_receipt(portal, attempt=1)
        wire, _ = json.JSONDecoder().raw_decode(prompt)
        refs = [row for row in wire["evidence"] if row["kind"] == "source-formalization-context"]
        assert refs and all(row["metadata"]["required"] for row in refs)
        evidence = json.loads("".join(row["summary"] for row in refs))
        actual = [row for row in evidence["contexts"] if row["kind"] == "numerical-formalization-context"]
        assert len(actual) == 1 and actual[0]["summary"] == summary
        assert evidence["proof_authority"] is evidence["execution_authority"] is False
        assert evidence["source_obligations_discharged"] == 0
        # Resolve the advertised API using the exact coding evidence's owner
        # reference, rather than accepting a formula supplied by the model.
        reference = actual[0]["receipt_reference"]
        retained = json.loads((Path(reference["output"]) / "receipt.json").read_bytes())
        retained["receipt_sha256"] = reference["receipt_sha256"]
        assert retained == formal
        visible_ids = {row["obligation_id"] for row in summary["pending_obligations"]}
        omitted = next(row for row in _retained_obligations(retained)
                       if row["obligation_id"] not in visible_ids)
        page = numerical.retrieve_numerical_obligations(repository=root, expected_receipt=retained,
            path="bottle.py", function_id=omitted["function_ids"][0], kind=omitted["kind"])
        assert omitted in page["rows"]
        assert page["schema"] == "supervisor-numerical-obligation-page@1"
        assert page["context_receipt_sha256"] == reference["receipt_sha256"]
        assert page["binding"]["source_sha256"] == formal["source_hashes"]["bottle.py"]
        assert page["total_count"] == summary["pending_obligation_count"]
        assert len(numerical._page_bytes(page)) <= summary["obligation_retrieval"]["max_page_bytes"]
        assert page["source_obligations_discharged"] == 0 and page["proof_authority"] is False
        workspace = state / "worker-worktree"
        subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)],
            check=True, capture_output=True)
        for name in formal["source_hashes"]:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        monkeypatch.chdir(workspace)
        _, observed = provider
        for path in (root / "bottle.py", workspace / "bottle.py", Path(formal["output"]) / "receipt.json"):
            before = path.read_bytes()
            try:
                path.write_bytes(before + b" ")
                with pytest.raises(ValueError):
                    worker_router.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
                        max_output_tokens=128, semantic_repository=root)
                assert observed == [], str(path)
            finally:
                path.write_bytes(before)
        output, receipt = worker_router.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
            max_output_tokens=128, semantic_repository=root)
        assert output == "literal summary" and len(observed) == 1
        assert RETRIEVAL_API in observed[0][0]
        assert receipt["model_prompt_sha256"] == hashlib.sha256(observed[0][0].encode()).hexdigest()
        dispatch = receipt["source_formalization_context"]
        assert dispatch["validation_phase"] == "before_provider"
        assert "numerical-formalization-context" in dispatch["context_kinds"]
        assert dispatch["proof_authority"] is False
    finally:
        daemon.close_event_runtime()


@pytest.mark.parametrize("mutation", ["source", "summary", "receipt"])
def test_obligation_discovery_cannot_reach_planner_after_context_mutation(selected, monkeypatch, mutation):
    root, state, _, formal = selected
    _assert_discovery(formal)
    output = Path(formal["output"])
    path = {"source": root / "bottle.py", "summary": output / "planning-summary.json",
            "receipt": output / "receipt.json"}[mutation]
    path.write_bytes(path.read_bytes() + b" ")
    _version(monkeypatch)
    with pytest.raises(ValueError):
        prep.plan(state=state, provider_callable=lambda *a, **k: pytest.fail("stale context reached provider"))
    assert not (state / "planner-invoked.json").exists()
    with pytest.raises(ValueError):
        numerical.retrieve_numerical_obligations(repository=root, expected_receipt=formal, path="bottle.py")
