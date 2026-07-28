from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_context import (
    ContextCompletenessEntry,
    ContextCompletenessWitness,
    DecisionContextRepresentation,
)
from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    ActionEnvelope,
    DecisionAuthority,
    DecisionTarget,
    EffectEnvelope,
    EffectKind,
)
from ipfs_accelerate_py.agent_supervisor.control.execution_permit import (
    ExecutionAttempt,
    ExecutionEvidence,
    ExecutionPermit,
    ExecutionPermitIssuer,
    ExecutionPermitVerifier,
    MandatoryEvidenceState,
    PermitIssuanceError,
    PermitReplayError,
    PermitVerificationCode,
    PermitVerificationError,
    _artifact_token,
    _plain,
    _semantic_roots_digest,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    PlanAdmissionRequest,
    ValidationResult,
    compile_plan_admission,
)
from test.api.test_agent_supervisor_decision_contracts import (
    _request as _base_decision,
)
from test.api.test_agent_supervisor_ir_constraint_compiler import (
    _request as _base_admission,
)


NOW = 10_000


def _fixture() -> tuple[
    PlanAdmissionRequest,
    object,
    ContextCompletenessWitness,
]:
    admission = _base_admission()
    arguments = {"mode": "update", "path": "src/0.py"}
    decision_base = _base_decision()
    decision = replace(
        decision_base,
        objective_id="ASI-136",
        objective_revision="sha256:asi-136",
        authority=replace(
            decision_base.authority,
            capability_ids=("tool:editor",),
            principal_id="principal:worker",
        ),
        action=ActionEnvelope(
            action_id="action:write",
            action="write",
            tool_id="tool:editor",
            authority=DecisionAuthority.MUTATION,
            arguments=arguments,
            targets=(
                DecisionTarget(
                    target_id="resource:repository",
                    resource_type="repository",
                    repository_paths=("src/0.py",),
                ),
            ),
        ),
        expected_effects=(
            EffectEnvelope(
                effect_id="effect:write",
                kind=EffectKind.WRITE,
                authority=DecisionAuthority.MUTATION,
                target_ids=("resource:repository",),
                repository_paths=("src/0.py",),
                description="Update the declared repository file",
                verification={"command": "python -m pytest -q"},
            ),
        ),
        capabilities=(
            replace(
                decision_base.capabilities[0],
                capability_id="tool:editor",
            ),
        ),
    )
    candidate = _plain(admission.candidate_plan)
    candidate["actions"][0]["arguments"] = arguments
    intent_candidate = _plain(admission.intent_request.candidate_plan)
    intent_candidate["actions"][0]["arguments"] = arguments
    intent_request = replace(
        admission.intent_request,
        candidate_plan=intent_candidate,
    )
    program_root = _artifact_token(decision.program_root)
    admission = replace(
        admission,
        candidate_plan=candidate,
        intent_request=intent_request,
        decision_request=decision,
        repository_tree_id=decision.repository_root.cid_v1,
        root_bindings=tuple(
            replace(item, expected=program_root, observed=program_root)
            if item.kind == "program"
            else item
            for item in admission.root_bindings
        ),
        validation_results=tuple(
            ValidationResult(
                requirement_id=item.requirement_id,
                status=item.status,
                repository_tree_id=decision.repository_root.cid_v1,
                evidence_id=item.evidence_id,
                reason_codes=item.reason_codes,
            )
            for item in admission.validation_results
        ),
    )
    receipt = compile_plan_admission(admission)
    assert receipt.admitted
    closure = admission.mandatory_closure
    assert closure is not None
    entry = ContextCompletenessEntry(
        node_id=closure.decision_id,
        node_kind="decision",
        node_content_id=decision.request_id,
        path=(closure.decision_id,),
        path_edge_ids=(),
        reference_id=f"mandatory:{closure.decision_id}",
        reference_content_id=decision.request_id,
        representation=DecisionContextRepresentation.INLINE,
    )
    witness = ContextCompletenessWitness(
        decision_request_id=decision.request_id,
        semantic_graph_root_id=closure.root_id,
        semantic_graph_id="graph:permit-fixture",
        retrieval_receipt_id="retrieval:permit-fixture",
        closure_id=closure.closure_id,
        mandatory_node_ids=closure.node_ids,
        mandatory_edge_ids=closure.edge_ids,
        entries=(entry,),
        inline_reference_ids=(entry.reference_id,),
        roots_digest=_semantic_roots_digest(decision),
    )
    return admission, receipt, witness


def _issuer() -> ExecutionPermitIssuer:
    return ExecutionPermitIssuer(clock_ms=lambda: NOW)


def _permit(
    *,
    issuer: ExecutionPermitIssuer | None = None,
    evidence: tuple[ExecutionEvidence, ...] = (),
    allowed_use_count: int = 1,
):
    admission, receipt, witness = _fixture()
    selected = issuer or _issuer()
    permit = selected.issue(
        admission,
        receipt,
        witness,
        caller="agent-supervisor:implementation-daemon",
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:policy-v1",
        expires_at_ms=NOW + 30_000,
        allowed_use_count=allowed_use_count,
        evidence_receipts=evidence,
    )
    return selected, permit


def test_issues_immutable_exact_short_lived_permit_and_round_trips() -> None:
    issuer, permit = _permit()

    assert issuer.issued(permit)
    assert permit.candidate_action["arguments"]["path"] == "src/0.py"
    assert permit.declared_paths == ("src/0.py",)
    assert permit.context_witness.closure_id == permit.mandatory_closure.closure_id
    assert permit.semantic_roots["dirty_worktree"] == permit.worktree_root_id
    assert {item.domain for item in permit.evidence_receipts} >= {
        "intent",
        "legal",
        "security",
        "security_policy",
        "validation",
    }
    assert not permit.grants_completion_authority
    assert ExecutionPermit.from_json(permit.to_json()) == permit
    with pytest.raises(FrozenInstanceError):
        permit.allowed_use_count = 2  # type: ignore[misc]

    forged = copy.deepcopy(permit.to_dict())
    forged["tool_arguments"]["path"] = "src/broadened.py"
    with pytest.raises(ValueError, match="projection"):
        ExecutionPermit.from_dict(forged)

    unknown_field = permit.to_dict()
    unknown_field["undeclared_authority"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        ExecutionPermit.from_dict(unknown_field)

    forged_action = permit.to_dict()
    forged_action["candidate_action"]["arguments"]["path"] = "src/forged.py"
    with pytest.raises(ValueError, match="arguments"):
        ExecutionPermit.from_dict(forged_action)


def test_issuance_recompiles_admission_and_rejects_partial_or_unknown_state() -> None:
    admission, receipt, witness = _fixture()
    rejected_admission = replace(admission, graph_complete=False)
    rejected = compile_plan_admission(rejected_admission)
    with pytest.raises(PermitIssuanceError, match="rejected"):
        _issuer().issue(
            rejected_admission,
            rejected,
            witness,
            caller="caller",
            policy_id="policy",
            policy_revision="revision",
            expires_at_ms=NOW + 1_000,
        )

    unknown = ExecutionEvidence(
        domain="monitor",
        receipt_id="monitor:unknown",
        subject_ids=("monitor:required",),
        state=MandatoryEvidenceState.UNKNOWN,
    )
    with pytest.raises(PermitIssuanceError, match="unknown"):
        _permit(evidence=(unknown,))

    with pytest.raises(PermitIssuanceError, match="TTL"):
        _issuer().issue(
            admission,
            receipt,
            witness,
            caller="caller",
            policy_id="policy",
            policy_revision="revision",
            expires_at_ms=NOW + 600_000,
        )


def test_verification_consumes_exact_use_and_rejects_replay() -> None:
    issuer, permit = _permit()
    verifier = issuer.verifier()
    attempt = ExecutionAttempt.from_permit(permit, now_ms=NOW + 1)

    use = verifier.verify(permit, attempt)

    assert use.authorizes_effect
    assert not use.authorizes_completion
    assert use.remaining_uses == 0
    with pytest.raises(PermitReplayError) as caught:
        verifier.verify(permit, attempt)
    assert caught.value.code is PermitVerificationCode.REPLAYED


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"caller": "agent-supervisor:other"}, PermitVerificationCode.CALLER_MISMATCH),
        (
            {"policy_revision": "sha256:policy-v2"},
            PermitVerificationCode.POLICY_MISMATCH,
        ),
        ({"active_lease_id": "lease:lost"}, PermitVerificationCode.LEASE_LOST),
        ({"current_fencing_epoch": 999}, PermitVerificationCode.FENCE_LOST),
        ({"repository_tree_id": "tree:stale"}, PermitVerificationCode.STALE_ROOT),
        (
            {"actual_paths": ("src/0.py", "src/escape.py")},
            PermitVerificationCode.PATH_BROADENING,
        ),
        ({"actual_paths": ("src",)}, PermitVerificationCode.PATH_BROADENING),
        ({"actual_paths": ()}, PermitVerificationCode.PARTIAL_AUTHORITY),
        (
            {"completion_requested": True},
            PermitVerificationCode.COMPLETION_AUTHORITY_FORBIDDEN,
        ),
    ),
)
def test_pre_effect_verifier_rejects_changed_scope_and_authority(
    changes: dict[str, object],
    code: PermitVerificationCode,
) -> None:
    issuer, permit = _permit()
    attempt = ExecutionAttempt.from_permit(
        permit,
        now_ms=NOW + 1,
        **changes,
    )

    with pytest.raises(PermitVerificationError) as caught:
        issuer.verifier().verify(permit, attempt)
    assert caught.value.code is code


def test_rejects_changed_arguments_targets_effects_task_principal_and_receipts(
) -> None:
    issuer, permit = _permit()
    base = permit.decision_request
    cases: list[tuple[dict[str, object], PermitVerificationCode]] = []
    cases.append(
        (
            {
                "decision_request": replace(
                    base,
                    action=replace(
                        base.action,
                        arguments={"mode": "delete", "path": "src/0.py"},
                    ),
                )
            },
            PermitVerificationCode.CHANGED_OPERATION,
        )
    )
    changed_target = DecisionTarget(
        target_id="resource:repository",
        resource_type="repository",
        repository_paths=("src/1.py",),
    )
    cases.append(
        (
            {
                "decision_request": replace(
                    base,
                    action=replace(base.action, targets=(changed_target,)),
                    expected_effects=(
                        replace(
                            base.expected_effects[0],
                            repository_paths=("src/1.py",),
                        ),
                    ),
                )
            },
            PermitVerificationCode.CHANGED_TARGET,
        )
    )
    cases.append(
        (
            {
                "decision_request": replace(
                    base,
                    expected_effects=(
                        replace(base.expected_effects[0], description="Changed"),
                    ),
                )
            },
            PermitVerificationCode.CHANGED_EFFECT,
        )
    )
    cases.append(
        (
            {"decision_request": replace(base, objective_id="ASI-other")},
            PermitVerificationCode.TASK_MISMATCH,
        )
    )
    cases.append(
        (
            {
                "decision_request": replace(
                    base,
                    authority=replace(
                        base.authority,
                        principal_id="principal:other",
                    ),
                )
            },
            PermitVerificationCode.PRINCIPAL_MISMATCH,
        )
    )
    cases.append(
        (
            {"evidence_receipts": permit.evidence_receipts[:-1]},
            PermitVerificationCode.STALE_RECEIPT,
        )
    )

    for changes, expected in cases:
        attempt = ExecutionAttempt.from_permit(
            permit,
            now_ms=NOW + 1,
            **changes,
        )
        with pytest.raises(PermitVerificationError) as caught:
            issuer.verifier().verify(permit, attempt)
        assert caught.value.code is expected


def test_rejects_expiry_unknown_or_contradictory_mandatory_state_and_forgery() -> None:
    issuer, permit = _permit()
    expired = ExecutionAttempt.from_permit(
        permit,
        now_ms=permit.expires_at_ms,
    )
    with pytest.raises(PermitVerificationError) as caught:
        issuer.verifier().verify(permit, expired)
    assert caught.value.code is PermitVerificationCode.EXPIRED

    for state, expected in (
        (
            MandatoryEvidenceState.UNKNOWN,
            PermitVerificationCode.MANDATORY_STATE_UNKNOWN,
        ),
        (
            MandatoryEvidenceState.CONTRADICTORY,
            PermitVerificationCode.MANDATORY_STATE_CONTRADICTORY,
        ),
    ):
        changed = (
            replace(permit.evidence_receipts[0], state=state),
            *permit.evidence_receipts[1:],
        )
        attempt = ExecutionAttempt.from_permit(
            permit,
            now_ms=NOW + 1,
            evidence_receipts=changed,
        )
        with pytest.raises(PermitVerificationError) as caught:
            issuer.verifier().verify(permit, attempt)
        assert caught.value.code is expected

    stale_receipt = (
        replace(
            permit.evidence_receipts[0],
            semantic_roots={"program": "sha256:stale"},
        ),
        *permit.evidence_receipts[1:],
    )
    with pytest.raises(PermitVerificationError) as caught:
        issuer.verifier().verify(
            permit,
            ExecutionAttempt.from_permit(
                permit,
                now_ms=NOW + 1,
                evidence_receipts=stale_receipt,
            ),
        )
    assert caught.value.code is PermitVerificationCode.STALE_RECEIPT

    other_issuer = _issuer()
    with pytest.raises(PermitVerificationError) as caught:
        other_issuer.verifier().verify(
            permit,
            ExecutionAttempt.from_permit(permit, now_ms=NOW + 1),
        )
    assert caught.value.code is PermitVerificationCode.UNTRUSTED

    with pytest.raises(PermitVerificationError) as caught:
        ExecutionPermitVerifier().verify(
            permit,
            ExecutionAttempt.from_permit(permit, now_ms=NOW + 1),
        )
    assert caught.value.code is PermitVerificationCode.UNTRUSTED


def test_multi_use_sequence_is_bounded_and_idempotency_cannot_change_meaning() -> None:
    issuer, permit = _permit(allowed_use_count=2)
    verifier = issuer.verifier()
    first = verifier.verify(
        permit,
        ExecutionAttempt.from_permit(permit, now_ms=NOW + 1, use_sequence=1),
    )
    second = verifier.verify(
        permit,
        ExecutionAttempt.from_permit(permit, now_ms=NOW + 2, use_sequence=2),
    )

    assert first.remaining_uses == 1
    assert second.remaining_uses == 0
    with pytest.raises(PermitReplayError):
        verifier.verify(
            permit,
            ExecutionAttempt.from_permit(
                permit,
                now_ms=NOW + 3,
                use_sequence=3,
            ),
        )

    admission, receipt, witness = _fixture()
    with pytest.raises(PermitIssuanceError, match="idempotency"):
        issuer.issue(
            admission,
            receipt,
            witness,
            caller=permit.caller,
            policy_id=permit.policy_id,
            policy_revision="sha256:changed-policy",
            issued_at_ms=NOW,
            expires_at_ms=NOW + 30_000,
            allowed_use_count=2,
        )
