"""Complete package inputs through signed indexed planning and real Doctor proof.

The intent interpretation is authored. These fixtures qualify a finite import
binding repair, not learned interpretation, whole-program correctness or a
Terminal-Bench reward.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_repair_pipeline import (
    _git, container_umask,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import validate_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


INIT = '"""Authored inert package documentation."""\n'
DONOR = 'def transform(value):\n    return value * 2\n'


def prepared_package_alias(tmp_path, *, nested=False, missing_init=False):
    root = tmp_path / "repository"
    root.mkdir(parents=True)
    selected = "pkg/sub/worker.py" if nested else "pkg/worker.py"
    source = ("from " + (".." if nested else ".") + "helpers import transform as normalize\n"
              "def answer(value):\n    return transform(value)\n")
    sources = {"pkg/__init__.py": INIT, "pkg/helpers.py": DONOR, selected: source}
    if nested:
        sources["pkg/sub/__init__.py"] = INIT
    if missing_init:
        del sources["pkg/__init__.py"]
    for name, body in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    _git(root, "init", "-q")
    _git(root, "add", *sorted(sources))
    _git(root, "-c", "user.name=Qualification", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "authored package alias mismatch")
    instruction = tmp_path / "instruction.md"
    instruction.write_text(f"Repair the unique import alias mismatch in {selected}.\n")
    outputs = [{"path": selected, "effect": "modify", "media_type": "text/x-python"}]
    profile = {"schema": profiles.SCHEMA,
        "instruction_sha256": profiles.instruction_sha256(instruction.read_text()),
        "input_paths": sorted(sources), "outputs": outputs}
    contract = _requirements(instruction, symbolic=True)
    for item in contract["requirements"]:
        item["outputs"] = deepcopy(outputs)
    for item in contract["symbolic_operations"]["operations"]:
        item["outputs"] = deepcopy(outputs)
    contract = validate_intent_requirement_contract(contract, source_text=instruction.read_text())
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps(contract))
    state = tmp_path / "state"
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, intent_requirement_contract=requirements, disable_intent_autoencoder=True)
    initial = prep.initial_context(state=state)
    assert initial["indexed_symbols"] == 2 and initial["provider_calls"] == 0
    assert initial["public_task_index_scope"]["task_source_paths"] == sorted(sources)
    descriptor = json.loads((root / initial["descriptor"]["artifact"]).read_text())
    assert descriptor["semantic"]["ducklake"]["status"] == "projected"
    assert descriptor["semantic"]["ducklake"]["stored_catalogs"] == 2
    assert descriptor["semantic"]["ducklake"]["stored_links"] == 2
    assert descriptor["semantic"]["ducklake"]["authoritative"] is False
    planned = prep.plan(state, provider_callable=None)
    assert planned["qualified"], planned.get("failure")
    assert planned["planning_strategy"] == "intent_symbolic" and planned["provider_calls"] == 0
    assert planned["goals"] == 2 and planned["tasks"] == 1
    context = prep.context(state=state)
    assert context["initial_indexes_reused"] is True
    admission = json.loads((state / "admission.json").read_text())
    task_cid, = planned["task_cids"]
    return {"repository": root, "state": state, "prepared": prepared, "admission": admission,
        "task_cid": task_cid, "selected": selected, "sources": sources,
        "source": source, "after": source.replace("return transform(", "return normalize("),
        "initial": initial, "context": context}


@pytest.fixture(autouse=True)
def provider_forbidden(monkeypatch):
    from ipfs_accelerate_py import llm_router

    def forbidden(*args, **kwargs):
        pytest.fail("package alias qualification must not call an LLM provider")

    for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
                 "generate_text_mesh_batch", "get_llm_provider"):
        monkeypatch.setattr(llm_router, name, forbidden)


@pytest.mark.parametrize("nested", [False, True], ids=["relative", "parent-relative"])
def test_indexed_package_alias_reaches_proof_and_staged_candidate(tmp_path, nested):
    case = prepared_package_alias(tmp_path, nested=nested)
    root, state, task_cid = case["repository"], case["state"], case["task_cid"]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        before = intent.get_task(task_cid)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state,
        admission=case["admission"], task_cid=task_cid)
    assert result["status"] == "candidate_ready", result
    assert result["route"] == "doctor_candidate" and result["provider_calls"] == 0
    capabilities = result["symbolic_capabilities"]
    assert capabilities["proof"]["local_contract_proof_reported"] is True
    assert capabilities["proof"]["whole_program_verified"] is False
    assert capabilities["operators"]["selected_workflow"] == "closed_package_imported_alias_call"
    raw = Path(result["artifact"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    handoff = json.loads(raw)
    assert handoff["proof_receipt_id"] and "projected supplied parameters satisfy" in handoff["proof_scope"]
    native = json.loads(Path(result["result_artifact"]).read_text())
    assert native["source_partition"]["program_paths"] == sorted(case["sources"])
    assert native["stages"]["tactician"]["status"] == "executed"
    assert native["stages"]["proof"]["status"] == "executed"
    assert native["stages"]["proof"]["disposition"] == "verified"
    assert native["stages"]["proof"]["receipt_id"] == handoff["proof_receipt_id"]
    assert _git(root, "show", handoff["candidate_commit"] + ":" + case["selected"]) == case["after"].strip()
    assert _git(root, "diff", "--name-only", handoff["base_commit"], handoff["candidate_commit"]) == case["selected"]
    workspace = tmp_path / "allocated"
    _git(root, "worktree", "add", "--detach", str(workspace), handoff["base_commit"])
    observed = materialize_doctor_candidate(artifact=Path(result["artifact"]),
        expected_sha256=result["sha256"], task_cid=task_cid,
        prompt=json.dumps({"objective_id": handoff["task_id"]}), workspace=workspace)
    assert observed["provider_calls"] == 0
    assert observed["publication_authority"] is observed["completion_authority"] is False
    for name, source in case["sources"].items():
        assert (workspace / name).read_text() == (case["after"] if name == case["selected"] else source)
        assert (root / name).read_text() == source
    subprocess.run(case["prepared"]["spec"]["validations"][0]["argv"], cwd=workspace,
                   check=True, capture_output=True, timeout=10)
    module = "pkg.sub.worker" if nested else "pkg.worker"
    subprocess.run(["python3", "-B", "-c",
        f"from {module} import answer; assert answer(3) == 6; assert answer(-2) == -4"],
        cwd=workspace, check=True, capture_output=True, timeout=10)
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(task_cid) == before


@pytest.mark.parametrize("failure", ["missing_init", "failed_prover", "source_drift"])
def test_package_alias_cannot_publish_incomplete_unproved_or_stale_graph(tmp_path, monkeypatch, failure):
    case = prepared_package_alias(tmp_path, missing_init=failure == "missing_init")
    root, state = case["repository"], case["state"]
    if failure == "failed_prover":
        _, kernel = dispatch._installed_provers()
        monkeypatch.setattr(dispatch, "_installed_provers", lambda: (Path("/usr/bin/false"), kernel))
    if failure == "source_drift":
        (root / "pkg/__init__.py").write_text('"""Changed package documentation."""\n')
        with pytest.raises(ValueError):
            dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state,
                admission=case["admission"], task_cid=case["task_cid"])
    else:
        result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state,
            admission=case["admission"], task_cid=case["task_cid"])
        assert result["status"] == "residual" and result["route"] == "model_router"
        assert result["provider_calls"] == 0 and result["completion_authority"] is False
    assert not (root / ".runtime/doctor-handoffs").exists()
    assert (root / case["selected"]).read_text() == case["source"]
