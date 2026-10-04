"""Real captured v3 header proof joins native publication and observation.

The repository, instruction mapping and scheduler telemetry are authored. Z3,
Git publication, Quack owner, signed validation and typed completion are real.
No neural model, provider, hidden verifier or live benchmark is involved.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test.api.test_header_intent_applicability import PROGRAM, case  # noqa: F401
from test.api.test_local_completion_bridge import complete, native_published_transition
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as planning
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import (
    run_owner_local_task_validations,
    verify_owner_local_benchmark_observation,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


@contextmanager
def _typed_header_claim(c, admission, intent, task_cid):
    capability = probe_quack_capabilities()
    assert capability.passes_health_check, "actual native Quack is required for this regression"
    local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    task = intent.get_task(task_cid)
    with open_existing_native_owner(
        database=intent.database_path, checkout=c.repo, state_dir=c.root / "owner",
        repository_id=c.manifest["payload"]["repository_cid"],
        execution_routes={task["task_alias"]: GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        daemon = DatabaseImplementationDaemon(
            database_path=owner.database, coordination_path=c.root / "coordination.duckdb",
            execution_path=c.root / "execution.duckdb", authority_mode="quack",
            task_source_kind="duckdb", owner_session_id="session:header-observation-test",
            process_instance_id=owner.identity.process_birth_id,
            quack_uri=owner.identity.listen_uri, task_source=owner.source,
            close_task_source=False, state_owner_bootstrap_credentials=owner.credentials,
            strict_task_sharding=True, max_task_attempts=2, lease_ms=60_000,
            require_real_execution=True,
        ).open()
        try:
            attempt = daemon.claim_next()
            assert attempt is not None and attempt.task_cid == task_cid
            yield owner, attempt
        finally:
            daemon.close()


@pytest.fixture
def completed_header(case, record_property):
    from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as headers

    c = case
    planned = planning.build_intent_symbolic_plan(c.contract, manifest=c.manifest,
        source_applicability_nomination=c.nomination)
    evidence = planned["receipt"]["source_applicability"]
    assert evidence["solver_calls"] == 6
    assert {row["solver_answer"] for row in evidence["checked_obligations"]} == {"sat", "unsat"}
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=c.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=c.nomination)
    assert local._header_nomination(admission["receipt"]["payload"]) == c.nomination
    candidate = headers.analyze_http_header_contracts(PROGRAM,
        protocol=headers.WsgiHeaderProtocolContract("review:authored-header-protocol@1", "emit")).candidate
    assert candidate is not None and candidate.source != PROGRAM
    task_cid = planned["graph"].tasks[0].task_cid
    with IntentRepository(c.root / "intent.duckdb") as intent:
        with _typed_header_claim(c, admission, intent, task_cid) as (owner, attempt):
            task = owner.source.get_task(task_cid)
            transition = native_published_transition({"repository": c.repo}, c.root, owner, attempt,
                modified_outputs={"headers.py": candidate.source})
            assert transition["payload"]["changed_paths"] == ["headers.py"]
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert passed["passed"] is True
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            completed = owner.source.get_task(task_cid)
            assert completed.status == "completed" and completed.revision == task.revision + 1
            record_property("header_observation_mechanism", json.dumps({
                "captured_header": True, "actual_z3_solver_calls_per_replay": 6,
                "native_git_merge": True, "native_quack_owner": True,
                "actual_public_validation": True, "typed_task_completed": True,
                "provider_or_model_calls": 0, "authored_scheduler_telemetry": True,
                "live_benchmark": False,
            }, sort_keys=True))
            yield SimpleNamespace(case=c, owner=owner, admission=admission,
                completed=completed, passed=passed)


def test_completed_v3_header_observation_preserves_signed_nomination(completed_header):
    c = completed_header
    observed = verify_owner_local_benchmark_observation(server=c.owner.server, admission=c.admission)
    assert observed["current_source_tree_id"] == c.passed["source_tree_id"]
    assert observed["receipt"] == c.admission["receipt"]["payload"]
    assert local._header_nomination(observed["receipt"]) == c.case.nomination
    assert observed["receipt"]["completion_authority"] is False
    assert c.owner.source.get_task(c.completed.task_cid) == c.completed


@pytest.mark.parametrize("change", ["missing", "altered"])
def test_completed_header_rejects_resigned_nomination_changes(completed_header, change):
    c = completed_header
    forged = deepcopy(c.admission)
    symbolic = forged["receipt"]["payload"]["intent_symbolic_planning"]
    if change == "missing":
        symbolic.pop("source_applicability_nomination")
    else:
        symbolic["source_applicability_nomination"]["captured_receipt"]["source_head"]["generation"] += 1
    # Re-sign deliberately: refusal must not depend only on a stale signature.
    forged["receipt"] = local._signed(forged["receipt"]["payload"], c.case.manifest["payload"])
    with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
        verify_owner_local_benchmark_observation(server=c.owner.server, admission=forged)
    assert c.owner.source.get_task(c.completed.task_cid) == c.completed
    assert verify_owner_local_benchmark_observation(server=c.owner.server,
        admission=c.admission)["current_source_tree_id"] == c.passed["source_tree_id"]


def test_completed_header_observation_rejects_live_source_drift(completed_header):
    c = completed_header
    assert verify_owner_local_benchmark_observation(server=c.owner.server,
        admission=c.admission)["current_source_tree_id"] == c.passed["source_tree_id"]
    (c.case.repo / "test_headers.py").write_text("raise AssertionError('authored source drift')\n")
    with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
        verify_owner_local_benchmark_observation(server=c.owner.server, admission=c.admission)
    assert c.owner.source.get_task(c.completed.task_cid) == c.completed


@pytest.mark.parametrize("consumer", ["requirement_observation", "router_instruction"])
def test_signed_header_nomination_reaches_other_replay_consumers(case, consumer):
    from ipfs_accelerate_py.agent_supervisor.runtime import intent_requirement_observation as observation
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction

    c = case
    planned = planning.build_intent_symbolic_plan(c.contract, manifest=c.manifest,
        source_applicability_nomination=c.nomination)
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=c.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=c.nomination)
    verified = local.verify_local_benchmark_admission(admission)
    task = planned["graph"].tasks[0]

    def replay(value):
        if consumer == "requirement_observation":
            result = observation._verify_admission(value)
            assert result["receipt"] == value["receipt"]["payload"]
        else:
            payload = {"schema": instruction.INTENT_SCHEMA,
                "manifest": value["manifest"], "task_cid": task.task_cid, "task_id": task.task_key,
                "owner_identity": verified["profile"].identity_did,
                "owner_profile_id": verified["profile"].profile_id,
                "intent_plan_admission": {key: value[key] for key in ("graph", "receipt", "requirement_bindings")}}
            result = instruction._intent_context(payload, verified["manifest"],
                (c.repo / "instruction.txt").read_text())
            assert result["task_cid"] == task.task_cid
        return result

    accepted = replay(admission)
    for change in ("missing", "altered"):
        forged = deepcopy(admission)
        symbolic = forged["receipt"]["payload"]["intent_symbolic_planning"]
        if change == "missing":
            symbolic.pop("source_applicability_nomination")
        else:
            symbolic["source_applicability_nomination"]["captured_receipt"]["source_head"]["generation"] += 1
        forged["receipt"] = local._signed(forged["receipt"]["payload"], c.manifest["payload"])
        with pytest.raises(ValueError):
            replay(forged)
    assert replay(admission) == accepted
