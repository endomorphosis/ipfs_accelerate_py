"""Finite data candidates require signed intent and freshly checked bytes."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_intent_data_transform import _data_case
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture
def data_case(tmp_path, request):
    case = _data_case(tmp_path / "fixture", mode=getattr(request, "param", "rename"))
    proposed = case["proposed"]
    case["admission"] = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=case["admission"], intent=intent)
        yield {**case, "intent": intent, "task_cid": materialized["task_cids"][0], "state": tmp_path / "doctor"}


def _prepare(case):
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_data_contract import prepare_ndjson_contract_candidate
    return prepare_ndjson_contract_candidate(**{name: case[name] for name in (
        "repository", "admission", "intent", "task_cid", "state")})


@pytest.mark.parametrize("data_case", ["rename", "copy"], indirect=True)
def test_checked_candidate_is_inert_until_allocated_worktree_materialization(data_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import materialize_doctor_contract_candidate

    def forbidden(*args, **kwargs):
        raise AssertionError("finite data candidate must not request a model")

    monkeypatch.setattr(llm_router, "generate_text", forbidden)
    case = data_case
    before = case["intent"].get_task(case["task_cid"])
    result = _prepare(case)
    assert result["status"] == "candidate_ready", result
    assert result["route"] == "doctor_contract_candidate"
    assert result["provider_calls"] == 0
    assert result["publication_authority"] is result["completion_authority"] is False
    check = result["check"]["check"]
    assert check["status"] == "checked" and check["evidence_kind"] == "finite_record_check"
    assert check["record_count"] == 2
    assert result["kernel_proved"] is result["whole_program_proved"] is result["proof_authority"] is False
    assert result["contract_index"]["hydrated"] is True
    assert result["contract_index"]["finite_check_recorded"] is True
    assert result["contract_index"]["active_receipt_ids"] == []
    assert case["intent"].get_task(case["task_cid"]) == before
    assert not (case["repository"] / "output.jsonl").exists()
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    artifact = Path(result["artifact"])
    assert artifact.stat().st_mode & 0o222 == 0
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    payload = json.loads(raw)
    assert payload["manifest_cid"] == local.content_identity(case["manifest"])
    assert payload["task_revision"] == before["revision"]
    assert len(payload["edits"]) == 1
    edit = payload["edits"][0]
    candidate = base64.b64decode(edit["after_bytes_base64"], validate=True)
    assert [json.loads(line) for line in candidate.splitlines()] == case["expected_output"]
    assert edit["before_sha256"] is None and edit["effect"] == "create"
    assert edit["after_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert payload["proof_receipt_id"] == result["check"]["check_receipt_id"]
    assert check["input_sha256"] == hashlib.sha256(case["input_bytes"]).hexdigest()
    assert check["output_sha256"] == edit["after_sha256"]
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "add", "--detach", str(workspace),
        case["baseline_commit"]], check=True, capture_output=True)
    materialized = materialize_doctor_contract_candidate(artifact=artifact,
        expected_sha256=result["sha256"], task_cid=case["task_cid"],
        prompt=json.dumps({"objective_id": before["task_alias"]}), workspace=workspace)
    assert materialized["status"] == "candidate_materialized"
    assert materialized["writes"] == [{"path": "output.jsonl", "effect": "create",
        "after_sha256": edit["after_sha256"], "write_mode": "exclusive_create"}]
    assert (workspace / "output.jsonl").read_bytes() == candidate
    assert (workspace / "input.jsonl").read_bytes() == case["input_bytes"]
    assert not (case["repository"] / "output.jsonl").exists()
    assert case["intent"].get_task(case["task_cid"]) == before
    assert materialized["publication_authority"] is materialized["completion_authority"] is False


@pytest.mark.parametrize("drift", ["input", "public_check", "created_output", "terminal_task"])
def test_drift_refuses_candidate_and_does_not_publish(data_case, drift):
    case = data_case
    if drift == "input":
        (case["repository"] / "input.jsonl").write_bytes(b'{"name":"replacement"}\n')
    elif drift == "public_check":
        (case["repository"] / "public_check.py").write_text("pass\n")
    elif drift == "created_output":
        (case["repository"] / "output.jsonl").write_bytes(b"{}\n")
    else:
        task = case["intent"].get_task(case["task_cid"])
        case["intent"].cas_task_status(task_cid=case["task_cid"], expected_revision=task["revision"],
            new_status="cancelled")
    before = case["intent"].get_task(case["task_cid"])
    with pytest.raises(ValueError):
        _prepare(case)
    assert case["intent"].get_task(case["task_cid"]) == before
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]


def test_unreviewed_symbolic_intent_stays_residual(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
    case = _data_case(tmp_path / "fixture")
    contract = deepcopy(case["contract"])
    contract["schema"] = "intent-plan-requirement-contract@2"
    del contract["reviewed_data_transform"]
    payload = case["manifest"]["payload"]
    manifest = local.author_local_benchmark_manifest(repository=case["repository"], profile_dir=case["profile"],
        lifecycle_dir=case["lifecycle"], task_specs=payload["tasks"], planning_roots=payload["planning_roots"],
        intent_requirements=contract)
    proposed = build_intent_symbolic_plan(contract, manifest=manifest)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        case.update(admission=admission, intent=intent, task_cid=materialized["task_cids"][0], state=tmp_path / "doctor")
        before = intent.get_task(case["task_cid"])
        result = _prepare(case)
        assert result["status"] == "residual"
        assert result["provider_calls"] == 0
        assert result["publication_authority"] is result["completion_authority"] is False
        assert "artifact" not in result
        assert not (case["repository"] / "output.jsonl").exists()
        assert intent.get_task(case["task_cid"]) == before


@pytest.mark.parametrize("attack", ["wrong_output", "source_drift", "task_drift", "forged_check"])
def test_independent_replay_refuses_producer_or_binding_drift(data_case, monkeypatch, attack):
    from ipfs_datasets_py.logic.software_contracts import finite_record_projection as shared
    case = data_case
    synthesize = shared.synthesize_finite_record_projection
    if attack == "forged_check":
        checker = shared.check_finite_record_projection
        observed = []

        def forged(*args, **kwargs):
            receipt = checker(*args, **kwargs)
            if not observed:
                receipt["output_sha256"] = "0" * 64
            observed.append(True)
            return receipt

        monkeypatch.setattr(shared, "check_finite_record_projection", forged)
    else:
        def changed(*args, **kwargs):
            raw = synthesize(*args, **kwargs)
            if attack == "wrong_output":
                return b'{"id":"forged","label":"replaced"}\n'
            if attack == "source_drift":
                (case["repository"] / "input.jsonl").write_bytes(b'{"name":"changed"}\n')
            else:
                task = case["intent"].get_task(case["task_cid"])
                case["intent"].cas_task_status(task_cid=case["task_cid"],
                    expected_revision=task["revision"], new_status="in_progress")
            return raw

        monkeypatch.setattr(shared, "synthesize_finite_record_projection", changed)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not (case["repository"] / "output.jsonl").exists()
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    assert case["intent"].get_task(case["task_cid"])["status"] in {"ready", "in_progress"}
    assert not list((case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


@pytest.mark.parametrize("field", ["check_receipt_id", "task_cid", "output_path"])
def test_index_cannot_replace_bound_check_before_candidate_publication(data_case, monkeypatch, field):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_data_contract as workflow
    case = data_case
    persist = workflow._persist_index

    def mutate(**kwargs):
        result = persist(**kwargs)
        kwargs["check"][field] = "untrusted-index-replacement"
        return result

    monkeypatch.setattr(workflow, "_persist_index", mutate)
    before = case["intent"].get_task(case["task_cid"])
    with pytest.raises(ValueError):
        _prepare(case)
    assert case["intent"].get_task(case["task_cid"]) == before
    assert not (case["repository"] / "output.jsonl").exists()
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    assert not list((case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))
