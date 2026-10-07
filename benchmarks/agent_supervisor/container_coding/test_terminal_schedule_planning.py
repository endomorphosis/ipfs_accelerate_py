"""Public scheduling declaration through indexed planning and Doctor synthesis.

The reviewed interpretation is authored. Public profile validation remains a
structural smoke check; the shared finite witness checker supplies separately
scoped feasibility evidence, without a kernel or Terminal-Bench reward claim.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as full
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial_runtime
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_repair_pipeline import (
    _git, container_umask,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_source_partition import terminal_profile_partition
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
from ipfs_datasets_py.logic.software_contracts.finite_interval_schedule import (
    FiniteIntervalScheduleContract, check_finite_interval_schedule, verify_finite_schedule_check,
)
from test.api.test_intent_interval_schedule import _schedule_case
from test.api.test_doctor_schedule_contract import isolated_schedule_resources  # noqa: F401


def test_public_json_schedule_reaches_indexed_symbolic_candidate_without_provider(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router

    def forbidden(*args, **kwargs):
        pytest.fail("reviewed schedule planning and synthesis must not call a model provider")

    for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
                 "generate_text_mesh_batch", "get_llm_provider"):
        monkeypatch.setattr(llm_router, name, forbidden)
    monkeypatch.setattr(prep, "generate_prompt_goal_graph", forbidden)
    monkeypatch.setattr(prep, "build_prompt_goal_provider_request", forbidden)

    authored = _schedule_case(tmp_path / "authored")
    root = tmp_path / "public-repository"
    root.mkdir()
    (root / "input.json").write_bytes(authored["input_bytes"])
    (root / "public_check.py").write_bytes(authored["public_check_bytes"])
    _git(root, "init", "-q")
    _git(root, "add", "input.json", "public_check.py")
    _git(root, "-c", "user.name=Qualification", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "authored public scheduling inputs")
    instruction = tmp_path / "instruction.md"
    instruction.write_text(authored["source"])
    outputs = [{"path": "output.json", "effect": "create", "media_type": "application/json"}]
    profile = {"schema": profiles.DATA_SCHEMA,
        "instruction_sha256": profiles.instruction_sha256(authored["source"]),
        "input_paths": ["input.json", "public_check.py"],
        "data_inputs": [{"path": "input.json", "media_type": "application/json"}],
        "outputs": outputs}

    # Rebind the authored source report to this public instruction and the
    # canonical public profile's validation key. No prior admission is reused.
    original = authored["contract"]
    ledger = build_intent_requirement_ledger(authored["source"],
        source_report=original["ledger"]["source_report"],
        source_identity={"path": prep.INSTRUCTION, "revision": _git(root, "rev-parse", "HEAD")})
    operations = deepcopy(original["symbolic_operations"])
    operations["ledger_sha256"] = ledger["ledger_sha256"]
    operation = operations["operations"][0]
    operation["task_key"] = "TB-CODE-TASK"
    operation["validation_keys"] = ["public-structural-smoke"]
    selector = deepcopy(original["reviewed_interval_schedule"])
    selector["validation_key"] = "public-structural-smoke"
    requirements = deepcopy(original["requirements"])
    for row in requirements:
        row["validation_keys"] = ["public-structural-smoke"]
    contract = build_intent_requirement_contract(source_path=prep.INSTRUCTION, ledger=ledger,
        requirements=requirements, source_text=authored["source"], symbolic_operations=operations,
        reviewed_interval_schedule=selector)
    contract_file = tmp_path / "public-requirements.json"
    contract_file.write_text(json.dumps(contract))
    selection = full._intent_selection(full.load_intent_requirements_for_instruction(contract_file, instruction))
    assert selection["planning_strategy"] == "intent_symbolic"
    assert selection["intent_requirement_contract_cid"] == local.content_identity(contract)

    state = tmp_path / "public-state"
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, intent_requirement_contract=contract_file, disable_intent_autoencoder=True)
    assert prepared["planning_strategy"] == "intent_symbolic"
    assert prepared["request"]["planning_policy"]["allow_model"] is False
    assert not (state / "provider-request.json").exists()
    manifest = prepared["manifest"]["payload"]
    partition = terminal_profile_partition(repository=root, manifest=manifest)
    assert partition.program_paths == ("public_check.py",)
    assert ("input.json", "task_data", hashlib.sha256(authored["input_bytes"]).hexdigest()) in partition.support_hashes
    assert manifest["tasks"][0]["validations"][0]["validation_key"] == "public-structural-smoke"
    assert "benchmark correctness remains unverified" in manifest["tasks"][0]["acceptance"][0]["criterion"]

    initial = prep.initial_context(state=state)
    assert initial["indexed_symbols"] >= 2 and initial["provider_calls"] == 0
    loaded = initial_runtime.load_initial_context(state=state, prepared=prepared,
        require_empty_owner=True)
    assert loaded["retrieval"]["status"] == "current"
    assert loaded["retrieval"]["hits"] == []  # Instruction has no symbol-name overlap.
    assert loaded["indexed"]["symbols"] == initial["indexed_symbols"]
    descriptor = json.loads((root / initial["descriptor"]["artifact"]).read_text())
    projected = descriptor["semantic"]["ducklake"]
    assert projected["status"] == "projected"
    assert projected["stored_catalogs"] == projected["stored_links"] == 2
    assert projected["authoritative"] is False
    planned = prep.plan(state, provider_callable=forbidden)
    assert planned["qualified"], planned.get("failure")
    assert planned["planning_strategy"] == "intent_symbolic" and planned["provider_calls"] == 0
    assert planned["symbolic_planning"]["observed_facts_supplied"] == 0
    assert planned["symbolic_planning"]["source_semantics_verified"] is False
    assert planned["tasks"] == 1 and planned["benchmark_success"] is None
    context = prep.context(state=state)
    assert context["initial_indexes_reused"] is True
    admission = json.loads((state / "admission.json").read_text())
    task_cid, = planned["task_cids"]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        before = intent.get_task(task_cid)

    result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state,
        admission=admission, task_cid=task_cid)
    assert result["status"] == "candidate_ready", result
    assert result["route"] == "doctor_contract_candidate" and result["provider_calls"] == 0
    workflow = result["contract_workflow"]
    assert workflow["solver_status"] == "sat" and workflow["kernel_proved"] is False
    assert workflow["contract_index"]["hydrated"] is True
    assert workflow["contract_index"]["active_receipt_ids"] == []
    raw = Path(result["artifact"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    candidate = json.loads(raw)
    edit, = candidate["edits"]
    assert edit["path"] == "output.json" and edit["effect"] == "create"
    output_bytes = base64.b64decode(edit["after_bytes_base64"], validate=True)
    assert hashlib.sha256(output_bytes).hexdigest() == edit["after_sha256"]
    checked = check_finite_interval_schedule(authored["input_bytes"], output_bytes,
        FiniteIntervalScheduleContract())
    assert checked == workflow["check"]["check"]
    assert verify_finite_schedule_check(checked, input_bytes=authored["input_bytes"],
        output_bytes=output_bytes, contract=FiniteIntervalScheduleContract()) == checked
    capabilities = result["symbolic_capabilities"]
    assert capabilities["operators"]["selected_workflow"] == "reviewed_finite_interval_schedule"
    assert capabilities["finite_schedule_check"]["reported"] is True
    assert capabilities["finite_schedule_check"]["kernel_proof"] is False
    assert capabilities["contracts"]["whole_task_behavior_verified"] is False
    assert candidate["publication_authority"] is candidate["completion_authority"] is False
    assert not (root / "output.json").exists()
    assert (root / "input.json").read_bytes() == authored["input_bytes"]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(task_cid) == before
