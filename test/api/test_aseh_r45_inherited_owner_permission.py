"""R45 launch must retain the existing R23 permission/empty-WAL route."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from test.api import test_agent_supervisor_configured_typed_grant_handoff as native

operator = native.aseh_operator


@pytest.fixture
def admitted(monkeypatch):
    # Reuse the existing exact R45 acceptance fixture and its negative checks.
    # Capture only calls accepted by the real admission verifier.
    original = operator._assert_exact_run_launch_admission
    accepted = []

    def capture(value, **kwargs):
        original(value, **kwargs)
        accepted.append(copy.deepcopy(value))

    monkeypatch.setattr(operator, "_assert_exact_run_launch_admission", capture)
    native.test_aseh_r45_exact_launch_requires_failed_r44_and_no_retry_receipts(
        monkeypatch
    )
    monkeypatch.setattr(operator, "_assert_exact_run_launch_admission", original)
    assert len(accepted) == 1
    value = accepted[0]
    anchor_head, anchor_tree = "c" * 40, "d" * 40
    anchor = native._aseh_r22_witness(anchor_head, anchor_tree)
    durable = {
        "schema": operator.ASEH_R30_DURABLE_CANDIDATE_WITNESS_SCHEMA,
        "head": anchor_head,
        "tree": anchor_tree,
        "branch_ref": anchor["branch_ref"],
        "index_entries_digest": anchor["index_entries_digest"],
        "index_flags_digest": anchor["index_flags_digest"],
        "status_digest": anchor["status_digest"],
        "authorization_v1_witness_cid": operator._identity(anchor),
        "authorization_guard_cid": "sha256:" + "6" * 64,
        "reflog_binding": "strict_v1_authorization_epoch_only",
    }
    durable["witness_cid"] = operator._identity(durable)
    prefix = native._aseh_r30_materialized_chain(
        head=anchor_head, tree=anchor_tree, witness=anchor, durable=durable
    )
    prefix[-1]["receipt_cid"] = value["repair_transition_chain"][-5]["receipt_cid"]
    value["repair_transition_chain"][:-4] = prefix
    value["bootstrap_receipt_id"] = "sha256:" + "1" * 64
    value["historical_live_authorizing_receipt_cid"] = value["repair_transition"]["receipt_cid"]
    value["projection_matches_events"] = True
    witness = native._aseh_r22_witness("a" * 40, "b" * 40)
    calls = []

    def current(observed, **kwargs):
        if observed != witness:
            raise operator.OperatorError("current witness changed")
        assert kwargs["expected_head"] == witness["head"]
        assert kwargs["expected_tree"] == witness["tree"]
        calls.append(kwargs)

    monkeypatch.setattr(operator, "_assert_candidate_authorization_witness", current)
    seal(value)
    return value, witness, calls


def seal(value):
    value.pop("admission_cid", None)
    value["admission_cid"] = operator._identity(value)


def project(value, witness, *, store="data/aseh/control.duckdb"):
    return operator._r23_owner_start_permission_context_from_launch_admission(
        board=SimpleNamespace(resolved_database_program=lambda: SimpleNamespace(store_id=store)),
        launch_admission=value,
        candidate_head=value["runtime_source_head"],
        candidate_tree=value["runtime_repository_tree_id"],
        candidate_authorization_witness=witness,
    )


def test_exact_r45_preserves_permission_context(admitted):
    value, witness, calls = admitted
    context = project(value, witness)
    assert context is not None
    assert context["repair_transition_receipt_cid"] == value["repair_transition"]["receipt_cid"]
    assert context["materialized_launch_admission_cid"] == value["admission_cid"]
    assert context["candidate_authorization_witness"] == witness
    assert calls


@pytest.mark.parametrize("change", [
    "admission_cid", "active_receipt", "active_payload", "r23_receipt",
    "chain_link", "r30_witness", "r30_anchor", "r44_failure",
    "missing_receipt_absence", "authorizing_receipt", "current_witness", "store",
])
def test_r45_permission_refuses_changed_authority(admitted, change):
    original, witness, _ = admitted
    value = copy.deepcopy(original)
    witness = dict(witness)
    store = "data/aseh/control.duckdb"
    if change == "active_receipt":
        value["repair_transition_chain"][-1]["receipt_cid"] = "sha256:" + "9" * 64
    elif change == "active_payload":
        value["repair_transition"] = dict(value["repair_transition"], extra=True)
    elif change == "r23_receipt":
        value["repair_transition_chain"][-10]["receipt_cid"] = "sha256:" + "9" * 64
    elif change == "chain_link":
        value["repair_transition_chain"][-3]["previous_receipt_cid"] = "sha256:" + "9" * 64
    elif change == "r30_witness":
        del value["repair_transition_chain"][-5]["durable_candidate_witness"]
    elif change == "r30_anchor":
        value["repair_transition_chain"][-5]["repair_head"] = "9" * 40
    elif change == "r44_failure":
        del value["repair_transition"]["failed_r44_authorization_attempt_cid"]
    elif change == "missing_receipt_absence":
        del value["r44_receipt_absent"]
    elif change == "authorizing_receipt":
        value["historical_live_authorizing_receipt_cid"] = "sha256:" + "9" * 64
    elif change == "current_witness":
        witness["head"] = "9" * 40
    elif change == "store":
        store = "data/foreign/control.duckdb"
    seal(value)
    if change == "admission_cid":
        value["admission_cid"] = "sha256:" + "9" * 64
    with pytest.raises(operator.OperatorError):
        project(value, witness, store=store)


def test_r45_descendant_requires_current_continuity(admitted, monkeypatch):
    value, witness, _ = admitted
    anchor_head, anchor_tree = "e" * 40, "f" * 40
    value["repair_transition"]["repair_head"] = anchor_head
    value["repair_transition"]["repair_tree"] = anchor_tree
    suffix = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-canonical-merge-suffix@1",
        "base_head": anchor_head, "target_head": "a" * 40, "target_tree": "b" * 40,
        "integrations": [{
            "request_id": "r45-descendant", "task_alias": "ASEH-001",
            "task_cid": "task-cid-r45-descendant", "candidate_commit": "7" * 40,
            "candidate_tree": "8" * 40, "integration_commit": "a" * 40,
            "integration_tree": "b" * 40, "changed_paths": ["bounded.py"],
        }],
    }
    suffix["receipt_cid"] = operator._identity(suffix)
    value["canonical_continuity"] = {
        "repair_to_current": suffix,
        "published_r39_to_historical_live_evidence_revision_closure": value["repair_transition"],
    }

    def git(*args):
        if args == ("show", "-s", "--format=%P", "a" * 40):
            return anchor_head + " " + "7" * 40
        if args == ("merge-base", "--is-ancestor", anchor_head, "a" * 40):
            return ""
        if args == ("rev-parse", "7" * 40 + "^{tree}"):
            return "8" * 40
        if args == ("rev-parse", "a" * 40 + "^{tree}"):
            return "b" * 40
        raise AssertionError(args)

    monkeypatch.setattr(operator, "_git", git)
    seal(value)
    assert project(value, witness) is not None
    value["canonical_continuity"]["repair_to_current"] = {"admitted": True}
    seal(value)
    with pytest.raises(operator.OperatorError, match="continuity"):
        project(value, witness)
