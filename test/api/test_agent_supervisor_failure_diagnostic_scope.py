"""Failed validation locations never grant write scope or invent deliverables."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as runtime
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_failure_review import (
    review_implementation_failure,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ImplementationProposal,
    ProposalValidationPolicy,
    ProposalValidationResult,
    validate_implementation_proposal,
)


PRODUCTION = "src/service.py"
EXISTING_TEST = "tests/test_existing.py"
COMMAND = "python -m pytest tests/test_existing.py"
INCOMPLETE = "diagnostic_path_evidence_incomplete"
CONTRACT_GAP = "task_scope_contract_revision_required"
MARKER = "[truncated original_bytes=41 sha256=" + "a" * 64 + " omitted_items=2]"


def _validation(paths=(PRODUCTION,), *, accepted=True):
    return {
        "passed": False,
        "returncode": 1,
        "reason": "declared_validation_failed",
        "proposal_gate": {
            "accepted": accepted,
            "changed_paths": list(paths),
            "reason_codes": [] if accepted else ["path_outside_scope"],
        },
        "failed_commands": [COMMAND],
        "failed_test_paths": [EXISTING_TEST],
        "validation_impact_paths": [EXISTING_TEST],
    }


def _review(validation, *, outputs=(PRODUCTION,), owned=(), accepted=True):
    receipt = review_implementation_failure(
        task_id="SYNTHETIC-DIAGNOSTIC",
        attempt=1,
        expected_outputs=outputs,
        task_owned_paths=owned,
        proposal_accepted=accepted,
        validation_commands=(COMMAND,),
        validation_result=validation,
    )
    assert not receipt.accepted
    assert not receipt.proof_authoritative and not receipt.completion_authoritative
    return receipt


def _plan(validation, receipt, **kwargs):
    return plan_automatic_implementation_rescue(
        validation_result={**validation, "failure_review": receipt.to_record()},
        expected_outputs=(PRODUCTION,),
        expected_outputs_present_on_disk=True,
        **kwargs,
    )


def test_unchanged_failed_test_preserves_scope_and_one_bounded_repair():
    validation = _validation()
    original = deepcopy(validation)
    receipt = _review(validation)
    assert receipt.changed_paths == (PRODUCTION,)
    assert receipt.contract_gap_paths == ()
    assert CONTRACT_GAP not in receipt.reason_codes
    assert receipt.out_of_scope_paths == () and receipt.denied_paths == ()
    plan = _plan(validation, receipt)
    assert plan.action is AutoRescueAction.INLINE_PROVIDER_RESCUE
    assert plan.max_provider_rescue_passes == 1
    assert plan.expected_outputs == (PRODUCTION,)
    assert plan.remediation_paths == ()
    assert _plan(validation, receipt, provider_rescue_passes_used=1).action is AutoRescueAction.NONE
    assert validation == original


@pytest.mark.parametrize("accepted", [False, None])
def test_unchanged_diagnostic_does_not_imply_test_edit_even_without_admission(accepted):
    validation = _validation(accepted=False)
    validation["proposal_gate"]["reason_codes"] = []
    receipt = _review(validation, accepted=accepted)
    assert receipt.contract_gap_paths == ()
    assert CONTRACT_GAP not in receipt.reason_codes


def test_actual_unauthorized_test_change_remains_rejected():
    validation = _validation((PRODUCTION, EXISTING_TEST), accepted=False)
    receipt = _review(validation, accepted=False)
    assert receipt.contract_gap_paths == (EXISTING_TEST,)
    assert receipt.out_of_scope_paths == (EXISTING_TEST,)
    assert CONTRACT_GAP in receipt.reason_codes
    assert "scope_expansion_denied" in receipt.reason_codes
    assert _plan(validation, receipt).action is AutoRescueAction.NONE


def test_caller_owned_non_output_test_is_not_a_contract_gap():
    validation = _validation((PRODUCTION, EXISTING_TEST), accepted=False)
    validation["proposal_gate"]["reason_codes"] = []
    receipt = _review(validation, owned=(EXISTING_TEST,), accepted=False)
    assert receipt.contract_gap_paths == ()
    assert CONTRACT_GAP not in receipt.reason_codes


def test_admitted_companion_change_keeps_existing_authority():
    validation = _validation((PRODUCTION, EXISTING_TEST))
    receipt = _review(validation, accepted=True)
    assert receipt.contract_gap_paths == ()
    assert _plan(validation, receipt).action is AutoRescueAction.INLINE_PROVIDER_RESCUE


def test_candidate_json_cannot_add_test_edit_authority():
    validation = _validation((PRODUCTION, EXISTING_TEST), accepted=False)
    validation.update({"task_owned_paths": [EXISTING_TEST],
                       "allowed_edit_paths": [EXISTING_TEST],
                       "ast_import_companion_paths": [EXISTING_TEST]})
    receipt = _review(validation, accepted=False)
    assert receipt.contract_gap_paths == (EXISTING_TEST,)
    assert _plan(validation, receipt).action is AutoRescueAction.NONE


@pytest.mark.parametrize("projection", [[MARKER], None, {}, "src/service.py"])
def test_incomplete_candidate_paths_never_get_automatic_rescue(projection):
    validation = _validation()
    validation["failed_test_paths"] = []
    validation["proposal_gate"]["changed_paths"] = projection
    receipt = _review(validation)
    assert INCOMPLETE in receipt.reason_codes
    plan = _plan(validation, receipt)
    assert plan.action is AutoRescueAction.NONE
    assert plan.reason == "hard_deny_reason_codes"


def test_unknown_projection_object_stays_incomplete_without_invoking_hooks():
    class Hostile:
        def __str__(self):
            raise AssertionError("must not stringify unknown object")

        def __iter__(self):
            raise AssertionError("must not iterate unknown object")

    validation = _validation()
    validation["failed_test_paths"] = []
    validation["proposal_validation"] = Hostile()
    receipt = _review(validation)
    assert INCOMPLETE in receipt.reason_codes
    assert _plan(validation, receipt).action is AutoRescueAction.NONE


def test_incomplete_evidence_cannot_take_earlier_helper_removal_route():
    validation = _validation()
    validation["failed_test_paths"] = []
    validation["proposal_gate"]["changed_paths"] = [MARKER]
    validation["scope_adjudication"] = {
        "accepted": False, "denied_paths": ["tmp-selfcheck.py"],
    }
    receipt = _review(validation, accepted=False)
    assert INCOMPLETE in receipt.reason_codes
    plan = _plan(validation, receipt)
    assert plan.action is AutoRescueAction.NONE
    assert plan.reason == "hard_deny_reason_codes"


@pytest.mark.parametrize("real_missing", [(), ("data/required.json",)])
def test_relocation_hints_cannot_replace_missing_output_inventory(real_missing):
    validation = _validation()
    validation["failed_test_paths"] = []
    hint = {"owner_module": EXISTING_TEST,
            "missing": "target_file=adversarial_harness/session.py",
            "candidate": "external/dependency/session.py"}
    before = _review(validation, outputs=(PRODUCTION, *real_missing))
    validation["descriptor_relocation_hints"] = [hint]
    after = _review(validation, outputs=(PRODUCTION, *real_missing))
    assert after.missing_expected_outputs == before.missing_expected_outputs == real_missing
    assert after.expected_outputs == before.expected_outputs
    assert hint["missing"] not in after.expected_outputs
    assert hint["missing"] in after.next_attempt_prompt_addendum


def _bound_proposal():
    proposal = ImplementationProposal(
        task_id="SYNTHETIC-DIAGNOSTIC", accepted_plan_id="plan:synthetic",
        repository_id="repo:synthetic", repository_tree_id="tree:synthetic",
        objective_id="goal:synthetic", baseline_id="baseline:synthetic",
        candidate_diff=(CandidateDiffEntry(old_path=PRODUCTION, new_path=PRODUCTION,
                        change_kind=DiffChangeKind.MODIFY,
                        before_source="VALUE = 1\n", after_source="VALUE = 2\n"),),
        declared_paths=(PRODUCTION,), context_id="context:synthetic",
    )
    policy = ProposalValidationPolicy(
        allowed_paths=(PRODUCTION,), task_owned_paths=(PRODUCTION,),
        expected_task_id=proposal.task_id, expected_plan_id=proposal.accepted_plan_id,
        expected_repository_id=proposal.repository_id,
        expected_repository_tree_id=proposal.repository_tree_id,
        expected_objective_id=proposal.objective_id,
    )
    result = validate_implementation_proposal(proposal, policy=policy)
    assert type(result) is ProposalValidationResult and result.accepted
    return result


@pytest.mark.parametrize("field_kind", ["known", "unknown", "different_known"])
def test_review_producer_projects_only_its_exact_known_live_result(tmp_path, monkeypatch, field_kind):
    proposal = _bound_proposal()
    task = runtime.PortalTask(task_id=proposal.proposal.task_id,
                              title="Synthetic diagnostic", status="ready", completion="auto",
                              priority="P0", track="SYNTHETIC",
                              outputs=[PRODUCTION], validation=[COMMAND])
    daemon = object.__new__(runtime.PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon._implementation_scope_adjudications = {}
    monkeypatch.setattr(runtime, "completion_gap_edit_scope", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "validation_ast_companion_paths", lambda *a, **k: ())
    monkeypatch.setattr(runtime, "validation_ast_relocation_hints", lambda *a, **k: ())
    monkeypatch.setattr(daemon, "_authoritative_validation_environment_guidance", lambda: "")
    events = []
    monkeypatch.setattr(daemon, "_record_event", lambda kind, body: events.append((kind, body)))
    validation = _validation()
    field = proposal if field_kind == "known" else (
        _bound_proposal() if field_kind == "different_known" else object()
    )
    validation["proposal_validation"] = field
    result = daemon._apply_implementation_failure_review(
        task=task, attempt=1, workspace_path=tmp_path,
        validation_result=validation, proposal_validation=proposal,
    )
    receipt = result["failure_review"]
    assert (INCOMPLETE in receipt["reason_codes"]) is (field_kind != "known")
    assert result["proposal_validation"] is field
    assert validation["proposal_validation"] is field
    assert result["passed"] is False and result["returncode"] == 1
    assert not receipt["accepted"] and len(events) == 1
    assert result["proposal_gate"] == validation["proposal_gate"]


def test_denied_test_change_adjudication_still_requires_contract_revision():
    validation = _validation((PRODUCTION, EXISTING_TEST), accepted=False)
    validation["scope_adjudication"] = {
        "accepted": False, "denied_paths": [EXISTING_TEST],
        "decisions": [{"path": EXISTING_TEST, "verdict": "denied",
                       "reason_codes": ["test_change_unverifiable"]}],
    }
    receipt = _review(validation, accepted=False)
    assert CONTRACT_GAP in receipt.reason_codes
    assert receipt.contract_gap_paths == (EXISTING_TEST,)
    assert _plan(validation, receipt).action is AutoRescueAction.NONE
