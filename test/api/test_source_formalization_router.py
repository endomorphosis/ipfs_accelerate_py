"""Native worker context reaches the shared router with current source advice.

Numerical grammar, DuckLake persistence, daemon compilation, Git worktree,
translation and dispatch checks are real. The coding provider and Source384
head are authored doubles; these tests claim neither model accuracy nor a
benchmark score and access no host credentials or rollout records.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import canonical_json
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
from test.api.test_semantic_router_integration import provider, _run
from test.api.test_semantic_router_translation import _reply


SOURCE = """import numpy as np

def choose_pair(matrix):
    values, vectors = np.linalg.eig(matrix)
    chosen = np.argmax(np.abs(values))
    return values[chosen], vectors[:, chosen]

def compare(left: int, right: int) -> bool:
    return left < right
"""


def _sha(raw):
    return hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()


def _authored_candidate(root, output):
    """A controlled head prediction passes the real scalar source guard."""
    from ipfs_datasets_py.logic.formalization.autoencoder.source_function_units import extract_function_units
    from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384 import SourceProgramDecoder384
    from ipfs_datasets_py.logic.software_verification.program import ProgramExpression

    raw = (root / "mod.py").read_bytes()
    inventory = extract_function_units(source_bytes=raw, source_sha256=_sha(raw), source_path="mod.py")
    unit = next(row for row in inventory["units"] if row["qualified_name"] == "compare")
    operands = ("expr:left", "expr:right")
    candidate = dict(kind="program_expression", document=ProgramExpression(
        "expr:result", "binary", "boolean", operand_ids=operands, evaluation_order=operands,
        operator="<", source_ref_ids=("source",)).to_dict())
    prediction = dict(id=unit["unit_id"], source_sha256=unit["normalized_body_sha256"],
                      candidate_ir=candidate, status="unqualified_candidate")
    decoder = SourceProgramDecoder384(SimpleNamespace(
        describe=lambda: dict(domain_id="security_ir"),
        infer=lambda rows: dict(domain_id="security_ir", rows=[prediction])), checkpoint_sha256="c" * 64)
    row = decoder.infer([dict(id=unit["unit_id"], source_text=unit["normalized_source_text"],
                             embedding=[0.125] * 384)])["rows"][0]
    assert row["source_contract"]["status"] == "qualified"
    inference = dict(report=dict(key=dict(source_head={"authored": True}, version_id="authored-head"),
        preparation=dict(files=[dict(path="mod.py", extraction=inventory)]), coverage={},
        output=dict(rows=[dict(id=row["id"], candidate=row)])))
    summary = source384._summary(inference, inventory=[dict(path="mod.py", disposition="captured")],
        checkpoint_sha256="c" * 64, inference_sha256=_sha(source384._raw(inference)))
    output.mkdir()
    receipt = dict(schema=source384.SCHEMA, repository=str(root), output=str(output),
        source_hashes={"mod.py": _sha(raw)}, summary=summary, **source384.AUTHORITY)
    (output / "receipt.json").write_bytes(bundles._raw(receipt))
    return receipt


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    root = tmp_path / "repository"; root.mkdir()
    (root / "mod.py").write_text(SOURCE)
    for args in (("init", "-q"), ("add", "mod.py"),
                 ("-c", "user.name=Qualification", "-c", "user.email=qualification@example.invalid",
                  "commit", "-qm", "authored source")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    workspace = tmp_path / "allocated-worktree"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)],
                   check=True, capture_output=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    semantic_output = root / ".runtime/semantic"
    prepare_semantic_context(repository=root, paths=["mod.py"], required_raw_paths=["mod.py"],
        objective="Review the selected pair using current source advice.", task_id="TASK-1", output=semantic_output)
    semantic_artifact = semantic_output / "worker-context.json"
    formal = numerical.prepare_numerical_context(repository=root, source_hashes={"mod.py": _sha(SOURCE)},
        output=tmp_path / "numerical", profile=numerical.PROFILE)
    learned = _authored_candidate(root, tmp_path / "source384")
    source384_checks = []
    def validate(**kwargs):
        assert kwargs["repository"] == root and kwargs["expected_receipt"] == learned
        source384_checks.append(True)
    monkeypatch.setattr(source384, "validate_source384_context", validate)
    task = PortalTask(task_id="TASK-1", canonical_task_cid="task:source-advice", title="Review selected pair",
        status="ready", completion="manual", priority="P1", track="implementation", outputs=["mod.py"],
        validation=[], acceptance="Use current evidence without claiming proven NumPy behavior.",
        metadata={})
    prepared = dict(schema="supervisor-task-context-preparation@1", task_cid=task.canonical_task_cid,
        task_id=task.task_id, metadata={"semantic context artifact": semantic_artifact.relative_to(root).as_posix(),
            "semantic context sha256": _sha(semantic_artifact.read_bytes())},
        numerical_formalization=formal, source384_context=learned)
    bundle = bundles.write_task_context_bundle(repository=root, prepared=[prepared], output=root / ".runtime/bundle.json")
    daemon = TodoImplementationDaemon(todo_path=root / ".runtime/tasks.todo.md", repo_root=root,
        state_path=tmp_path / "state/tasks.json", strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl", task_header_prefix="## TASK-")
    daemon._task_context_nomination_bundle = bundle
    try:
        prompt = daemon._build_implementation_prompt(task, attempt=1)
        wire, _ = json.JSONDecoder().raw_decode(prompt)
        refs = [row for row in wire["evidence"] if row["kind"] == "source-formalization-context"]
        assert refs and all(row["metadata"]["required"] for row in refs)
        evidence = json.loads("".join(row["summary"] for row in refs))
        yield SimpleNamespace(root=root, workspace=workspace, prompt=prompt, task=task,
            formal=formal, learned=learned, bundle=bundle, evidence=evidence, checks=source384_checks)
    finally:
        daemon.close_event_runtime()


@pytest.mark.parametrize("transport_schema", [codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA])
def test_native_worker_advice_reaches_actual_shared_router_and_restores_exactly(worker_case, provider, transport_schema):
    case = worker_case
    state, observed = provider
    assert case.checks == [True]
    encoded = codec.encode_semantic_router_prompt(prompt=case.prompt, repository=case.root,
                                                 transport_schema=transport_schema)
    reply, symbol = _reply(encoded)
    state["reply"] = json.dumps(reply)
    output, receipt = _run(case.root, case.prompt, semantic_transport_schema=transport_schema)
    assert len(observed) == 1 and case.checks == [True, True]
    actual, options = observed[0]
    expected, advisory = runner.render_model_prompt(prompt=encoded.provider_prompt,
        purpose="coding", workspace=case.workspace, semantic_transport=True)
    assert actual == expected
    native = codec.restore_semantic_router_prompt(provider_prompt=actual[len(advisory):],
        table=encoded.table, repository=case.root)
    assert native == case.prompt
    transported = json.loads(actual[len(advisory):])["native_context"]
    refs = [row for row in transported["evidence"] if row["kind"] == "source-formalization-context"]
    assert all(row["metadata"]["required"] is True for row in refs)
    evidence = json.loads("".join(row["summary"] for row in refs))
    assert evidence == case.evidence
    summaries = {row["kind"]: row["summary"] for row in evidence["contexts"]}
    assert summaries["numerical-formalization-context"] == case.formal["summary"]
    assert summaries["source384-candidate-context"] == case.learned["summary"]
    link = case.formal["summary"]["source_unit_links"][0]
    for key in ("unit_id", "function_id", "link_report_sha256"):
        assert link[key] in actual
    candidate = case.learned["summary"]["candidate_samples"][0]
    assert candidate["candidate_sha256"] in actual
    assert candidate["candidate_ir_json"] == summaries["source384-candidate-context"]["candidate_samples"][0]["candidate_ir_json"]
    assert candidate["source_validation_status"] == "qualified"
    assert case.formal["hydration"]["ducklake_verified"] is True
    assert evidence["source_obligations_discharged"] == 0
    assert json.loads(output)["structured_payload"]["symbol_ids"] == [symbol]
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    dispatch = receipt["source_formalization_context"]
    assert dispatch["worker_evidence_sha256"] == _sha(canonical_json(evidence))
    assert dispatch["context_bundle_sha256"] == case.bundle["sha256"]
    assert dispatch["task_id"] == case.task.task_id and dispatch["task_cid"] == case.task.canonical_task_cid
    assert dispatch["source_files"] == 1 and dispatch["validation_phase"] == "before_provider"
    assert dispatch["workspace_source_current"] is dispatch["canonical_source_current"] is True
    assert dispatch["neural_inference_replayed"] is dispatch["proof_authority"] is False
    assert receipt["model_prompt_sha256"] == _sha(actual)
    assert receipt["native_prompt_sha256"] == _sha(case.prompt)
    assert receipt["usage"]["prompt_tokens"] == 123  # Authored observations, not measured billing.
    assert receipt["native_rollout_usage"] is None  # No session identity: no host rollout read.


@pytest.mark.parametrize("mutation", ["canonical_source", "workspace_source", "numerical_receipt", "task", "bundle"])
def test_native_dispatch_refuses_changed_task_source_or_receipt_before_provider(worker_case, provider, mutation):
    case = worker_case
    _, observed = provider
    prompt = case.prompt
    if mutation in {"canonical_source", "workspace_source"}:
        root = case.root if mutation == "canonical_source" else case.workspace
        (root / "mod.py").write_text(SOURCE + "\n# concurrent authored edit\n")
    elif mutation == "numerical_receipt":
        path = Path(case.formal["output"]) / "receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "bundle":
        path = case.root / case.bundle["artifact"]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        wire, end = json.JSONDecoder().raw_decode(prompt)
        evidence = deepcopy(case.evidence)
        evidence["task_cid"] = "task:foreign"
        refs = build_text_context_references(canonical_json(evidence), reference_prefix="source-formalization",
            kind="source-formalization-context", path=case.bundle["artifact"], repository_id=wire["repository_id"],
            tree_id=wire["tree_id"], required=True, chunk_bytes=6144)
        wire["evidence"] = [row for row in wire["evidence"] if row["kind"] != "source-formalization-context"]
        wire["evidence"].extend(row.to_record() for row in refs)
        prompt = canonical_json(wire) + prompt[end:]
    with pytest.raises(ValueError):
        _run(case.root, prompt)
    assert observed == []
