"""Real signed finite parent and native trusted child; no worker authority."""
import base64
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import shutil
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as candidate_module
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError, ResourceLane,
)
from test.api.test_finite_integer_codebase import finite_tools, finite_source
from test.api.test_finite_repository_admission import _build_case, _author, _admit


@pytest.fixture(scope="module")
def candidate_case(tmp_path_factory, finite_tools):
    root = tmp_path_factory.mktemp("finite-repository-candidate")
    # Copy exact native ELF bytes so tool-tampering regressions never alter the
    # machine's shared Python executable or another concurrent qualification.
    python = root / "candidate-python"
    shutil.copyfile(finite_tools["python"]["path"], python)
    python.chmod(0o755)
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    tools = seal_finite_integer_tools(python_executable=python,
                                     lean_executable=Path(finite_tools["lean"]["path"]))
    case = _build_case(root / "native", tools)
    try:
        case["declaration"] = _author(case)
        case["admission"] = _admit(case, output=case["root"] / "signed-evidence")
        case["candidate"] = candidate_module.author_finite_repository_candidate(
            owner=case["owner"], admission=case["admission"],
            review_ref="review:authored-exact-offset-candidate")
        case["python"] = python
        yield case
    finally:
        state = case["scheduler"].snapshot()
        assert state["active_lease_count"] == state["waiting_request_count"] == 0
        case["connection"].close()


def _generate(case, output, **changes):
    arguments = dict(owner=case["owner"], admission=case["admission"], candidate=case["candidate"],
                     output=output, policy_observer=lambda bound: bound.roots)
    arguments.update(changes)
    return candidate_module.generate_finite_repository_candidate(**arguments)


def _assert_unchanged(case):
    assert (case["repository"] / "calc.py").read_bytes() == candidate_module.BEFORE_SOURCE
    assert case["index"].catalog.current(case["expected_head"].repository_id) == case["expected_head"]
    state = case["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0


@pytest.fixture(scope="module")
def generated(candidate_case):
    result = _generate(candidate_case, candidate_case["root"] / "generated")
    _assert_unchanged(candidate_case)
    return result


def test_actual_bounded_child_generates_only_reviewed_proposal(candidate_case, generated):
    assert generated["status"] == "candidate_generated"
    assert generated["proposal_generated"] is True
    assert generated["canonical_source_unchanged"] is True
    assert generated["reservation_released_on_return"] is True
    assert generated["parent_admission"] == candidate_case["admission"]
    assert generated["reviewed_candidate"] == candidate_case["candidate"]
    assert generated["task_cid"] == generated["reviewed_candidate"]["payload"]["task_cid"]
    assert len(generated["reviewed_candidate"]["payload"]["administrator_task_cids"]) == 2
    assert generated["child_process"]["pid"] > 0
    assert generated["child_process"]["returncode"] == 0
    assert generated["child_process"]["workspace_cleaned"] is True
    assert [row["reservation"]["memory_mb"] for row in generated["native_capacity"]["observations"]] == [1024, 512, 512, 1024]
    assert "lease_key" not in canonical_dag_json_bytes(generated).decode()
    assert all(generated[name] is False for name in candidate_module._FALSE)
    assert generated["provider_calls"] == generated["training_steps"] == 0
    raw = Path(generated["artifacts"]["replacement"]["path"]).read_bytes()
    assert raw == candidate_module.AFTER_SOURCE
    verified = candidate_module.verify_generated_finite_repository_candidate(record=generated)
    assert verified["observed_current"] is False
    assert verified["child_origin_authenticated"] is False
    _assert_unchanged(candidate_case)


@pytest.mark.parametrize("field", ["task_cid", "head", "after_base64", "path", "function_name", "parameter", "desired_offset", "environment", "limits", "administrator_task_cids", "mutation_authority"])
def test_even_owner_resigned_candidate_cannot_change_reviewed_effect(candidate_case, field):
    value = deepcopy(candidate_case["candidate"]["payload"])
    if field == "head":
        value[field]["generation"] += 1
    elif field == "after_base64":
        altered = b"def increment(n: int) -> int:\n    return n + 99\n"
        value[field] = base64.b64encode(altered).decode()
        value["after_sha256"] = candidate_module._sha(altered)
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
        value["after_cid"] = cid_for_bytes(altered)
    elif field == "desired_offset":
        value[field] = True
    elif field == "environment":
        value[field]["FAKE_PRIVATE_KEY"] = "authored-not-a-key"
    elif field == "limits":
        value[field]["child_timeout_ms"] = 50_000
    elif field == "administrator_task_cids":
        value[field] = value[field][:1]
    elif field == "mutation_authority":
        value[field] = 0  # Bool/int aliases must not satisfy exact policy.
    else:
        value[field] = "foreign"
    altered = local._signed(value, candidate_case["manifest"]["payload"])
    with pytest.raises(ValueError):
        candidate_module.verify_finite_repository_candidate(admission=candidate_case["admission"], candidate=altered)
    _assert_unchanged(candidate_case)


def test_changed_signed_candidate_cannot_recompute_digest_to_gain_authority(candidate_case):
    altered = deepcopy(candidate_case["candidate"])
    altered["payload"]["review_ref"] = "review:foreign-declaration"
    assert cid_for_structured(altered) != cid_for_structured(candidate_case["candidate"])
    with pytest.raises(ValueError):
        candidate_module.verify_finite_repository_candidate(admission=candidate_case["admission"], candidate=altered)


@pytest.mark.parametrize("control", ["cancelled", "expired", "wrong_head", "foreign_parent", "small_parent"])
def test_launch_controls_refuse_before_output_or_child(candidate_case, tmp_path, control):
    case, output = candidate_case, tmp_path / "refused"
    owner = case["owner"]
    parent = None
    if control == "cancelled":
        event = threading.Event()
        event.set()
        owner = replace(owner, cancel_event=event)
    elif control == "expired":
        owner = replace(owner, timeout_seconds=0.000001)
    elif control == "wrong_head":
        owner = replace(owner, expected_head=replace(owner.expected_head, generation=owner.expected_head.generation + 1))
    elif control == "foreign_parent":
        owner = replace(owner, parent_lease="not-a-native-parent")
    else:
        parent = case["scheduler"].acquire(ResourceLane.VALIDATION, cpu_slots=1, memory_mb=1024,
            child_process_slots=1, timeout=1, request_id="authored-small-parent")
        owner = replace(owner, scheduler=None, parent_lease=parent)
    try:
        with pytest.raises((ValueError, TypeError, LeaseCancelledError, LeaseTimeoutError)):
            _generate(case, output, owner=owner)
        assert not output.exists()
    finally:
        if parent is not None:
            parent.release()
    _assert_unchanged(case)


@pytest.mark.parametrize("control", ["source", "manifest_cas", "ast_cas", "parent_lean", "current_lean", "candidate", "driver", "python", "cancelled_after_child"])
def test_real_child_then_late_mutation_cannot_publish_success(candidate_case, tmp_path, monkeypatch, control):
    case, output = candidate_case, tmp_path / "late-candidate"
    original = candidate_module.native_process.BoundedToolRunner.run
    restore = []
    event = threading.Event()
    owner = replace(case["owner"], cancel_event=event)
    hit = False
    def run(runner, request, *args, **kwargs):
        nonlocal hit
        result = original(runner, request, *args, **kwargs)
        if isinstance(request, list) and len(request) == 5 and request[3] == "driver.py":
            assert result.ok
            hit = True
            if control == "cancelled_after_child":
                event.set()
                return result
            if control == "source":
                path = case["repository"] / "calc.py"
            elif control == "manifest_cas":
                path = case["index"].artifacts.path_for(case["expected_head"].manifest_cid)
            elif control == "ast_cas":
                manifest = case["index"].load(case["expected_head"].manifest_cid)
                entry = next(row for row in manifest.snapshot.entries if row.path == "calc.py")
                unit = next(row for row in manifest.units if row.source_key == entry.source_key)
                path = case["index"].artifacts.path_for(unit.ast_cid)
            elif control == "python":
                path = case["python"]
            elif control == "parent_lean":
                path = Path(case["admission"]["evidence"]["operational_model"]["artifacts"]["lean_olean"]["path"])
            elif control == "current_lean":
                import json
                current = json.loads((output / "current-verification.json").read_bytes())
                path = Path(current["fresh_evidence"]["operational_model"]["artifacts"]["lean_olean"]["path"])
            else:
                path = output / ("candidate.json" if control == "candidate" else "driver.py")
            previous = path.read_bytes()
            restore.append((path, previous, path.stat().st_mode & 0o777))
            path.chmod(0o600)
            path.write_bytes(previous + b"\n")
        return result
    monkeypatch.setattr(candidate_module.native_process.BoundedToolRunner, "run", run)
    try:
        with pytest.raises((ValueError, LeaseCancelledError)):
            _generate(case, output, owner=owner)
        assert hit, "control must mutate after the actual trusted child finishes"
        assert not (output / "result.json").exists()
    finally:
        for path, previous, mode in restore:
            path.write_bytes(previous)
            path.chmod(mode)
    _assert_unchanged(case)


def test_final_historical_callback_cannot_mutate_already_checked_parent_proof(candidate_case, tmp_path, monkeypatch):
    original = candidate_module.verify_generated_finite_repository_candidate
    parent = Path(candidate_case["admission"]["evidence"]["operational_model"]["artifacts"]["lean_olean"]["path"])
    raw, mode = parent.read_bytes(), parent.stat().st_mode & 0o777
    hit = False
    def verify(*, record):
        nonlocal hit
        checked = original(record=record)
        parent.chmod(0o600)
        parent.write_bytes(raw + b"\n")
        hit = True
        return checked
    monkeypatch.setattr(candidate_module, "verify_generated_finite_repository_candidate", verify)
    try:
        with pytest.raises(ValueError):
            _generate(candidate_case, tmp_path / "late-verifier-parent")
        assert hit
    finally:
        parent.write_bytes(raw)
        parent.chmod(mode)
    _assert_unchanged(candidate_case)


def test_actual_parent_revocation_after_final_verification_refuses_return(candidate_case, tmp_path, monkeypatch):
    case = candidate_case
    parent = case["scheduler"].acquire(ResourceLane.VALIDATION, cpu_slots=2, memory_mb=2048,
        child_process_slots=2, timeout=1, request_id="authored-complete-candidate-parent")
    owner = replace(case["owner"], scheduler=None, parent_lease=parent)
    original = candidate_module.verify_generated_finite_repository_candidate
    hit = False
    def verify(*, record):
        nonlocal hit
        checked = original(record=record)
        assert len(record["native_capacity"]["observations"]) == 4
        assert record["native_capacity"]["observations"][0]["reservation"]["parent_lease_id"] == parent.lease_id
        assert parent.cancel()
        hit = True
        return checked
    monkeypatch.setattr(candidate_module, "verify_generated_finite_repository_candidate", verify)
    try:
        with pytest.raises(LeaseCancelledError, match="parent authority"):
            _generate(case, tmp_path / "late-parent-cancel", owner=owner)
        assert hit, "parent must revoke after actual child and final historical validation"
    finally:
        parent.release()
    _assert_unchanged(case)


def test_result_written_then_changed_is_refused(candidate_case, tmp_path, monkeypatch):
    original = candidate_module._write
    output = tmp_path / "result-drift"
    hit = False
    def write(path, raw):
        nonlocal hit
        row = original(path, raw)
        if path == output / "result.json":
            hit = True
            path.chmod(0o600)
            path.write_bytes(raw + b"\n")
        return row
    monkeypatch.setattr(candidate_module, "_write", write)
    with pytest.raises(ValueError):
        _generate(candidate_case, output)
    assert hit
    _assert_unchanged(candidate_case)


@pytest.mark.parametrize("field", ["completion_authority", "task_cid", "replacement_cid", "child_process", "artifacts"])
def test_historical_result_rehash_cannot_change_signed_candidate_or_exact_output(generated, field):
    record = deepcopy(generated)
    if field == "completion_authority":
        record[field] = True
    elif field == "child_process":
        record[field]["command"] = ["/bin/sh", "-c", "foreign"]
    elif field == "artifacts":
        record[field].pop("replacement")
    else:
        record[field] = "foreign"
    record["result_cid"] = cid_for_structured({key: row for key, row in record.items() if key != "result_cid"})
    with pytest.raises(ValueError):
        candidate_module.verify_generated_finite_repository_candidate(record=record)


def test_candidate_api_has_no_command_source_or_completion_bypass(candidate_case, tmp_path):
    with pytest.raises(TypeError):
        _generate(candidate_case, tmp_path / "unused", command=["/bin/sh"])
    with pytest.raises(TypeError):
        candidate_module.author_finite_repository_candidate(owner=candidate_case["owner"],
            admission=candidate_case["admission"], review_ref="review:fixed", after_source=b"foreign")
