from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.goal_development_contracts import (
    GOAL_DECOMPOSITION_DRAFT_SCHEMA,
    GOAL_DEVELOPMENT_CONTRACT_VERSION,
    GoalAdmissionDecision,
    GoalDecompositionDraft,
    GoalDecompositionProposal,
    GoalDevelopmentAdmissionReceipt,
    GoalDevelopmentAuthority,
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentProposalReceipt,
    GoalDevelopmentRequest,
    GoalDevelopmentTrust,
    GoalProposalDecision,
)


def _policy(
    mode: GoalDevelopmentMode = GoalDevelopmentMode.SHADOW, **changes: object
) -> GoalDevelopmentPolicy:
    values = {
        "mode": mode,
        "max_depth": 3,
        "max_breadth": 3,
        "max_proposals": 8,
        "max_bytes": 32_768,
        "max_tokens": 8_192,
    }
    values.update(changes)
    return GoalDevelopmentPolicy(**values)


def _request(
    policy: GoalDevelopmentPolicy | None = None, **changes: object
) -> GoalDevelopmentRequest:
    policy = policy or _policy()
    values = {
        "root_goal_id": "goal:root",
        "root_goal_content_id": "baguqeera-root-goal",
        "satisfaction_formula_id": "formula:root-satisfied",
        "assumption_ids": ("assumption:reviewed-b", "assumption:reviewed-a"),
        "evidence_requirement_ids": ("evidence:review", "evidence:test"),
        "vocabulary_profile_id": "supervisor-reviewed-dcec-tdfol",
        "vocabulary_version": 1,
        "repository_tree_id": "tree:base",
        "scope_ids": ("path:test/api", "path:ipfs_accelerate_py"),
        "policy_digest": policy.policy_digest,
        "mode": policy.mode,
        "repair_draft_id": (
            "baguqeera-prior-draft"
            if policy.mode is GoalDevelopmentMode.REPAIR_ONLY
            else ""
        ),
    }
    values.update(changes)
    return GoalDevelopmentRequest(**values)


def _proposal(
    proposal_id: str,
    parent_id: str = "goal:root",
    *,
    depends_on: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = ("assumption:reviewed-a",),
    title: str = "",
) -> GoalDecompositionProposal:
    return GoalDecompositionProposal(
        proposal_id=proposal_id,
        parent_id=parent_id,
        satisfaction_formula_id=f"formula:{proposal_id}",
        assumption_ids=assumptions,
        evidence_requirement_ids=("evidence:test",),
        scope_ids=("path:ipfs_accelerate_py",),
        depends_on=depends_on,
        title=title,
    )


def _draft(
    policy: GoalDevelopmentPolicy | None = None,
    request: GoalDevelopmentRequest | None = None,
    proposals: tuple[GoalDecompositionProposal, ...] | None = None,
    **changes: object,
) -> GoalDecompositionDraft:
    policy = policy or _policy()
    request = request or _request(policy)
    values = {
        "request": request,
        "policy": policy,
        "proposals": proposals
        or (
            _proposal("subgoal:contracts"),
            _proposal(
                "subgoal:tests",
                parent_id="subgoal:contracts",
                depends_on=("subgoal:contracts",),
            ),
        ),
        "producer_id": "provider:leanstral-goal-development",
        "token_count": 120,
    }
    values.update(changes)
    return GoalDecompositionDraft(**values)


def test_policy_and_all_modes_are_versioned_content_addressed_contracts() -> None:
    for mode in GoalDevelopmentMode:
        policy = _policy(mode)
        restored = GoalDevelopmentPolicy.from_json(policy.to_json())
        assert restored == policy
        assert restored.content_id == policy.content_id
        assert restored.schema_version == GOAL_DEVELOPMENT_CONTRACT_VERSION
        assert policy.to_dict()["contract_version"] == 1
        assert policy.policy_digest == policy.content_id

    assert {mode.value for mode in GoalDevelopmentMode} == {
        "off",
        "shadow",
        "assist",
        "auto_safe",
        "repair_only",
    }


def test_request_freezes_every_semantic_input_and_round_trips() -> None:
    policy = _policy(GoalDevelopmentMode.ASSIST)
    request = _request(policy)
    restored = GoalDevelopmentRequest.from_json(request.to_json())

    assert restored == request
    assert restored.request_id == request.content_id
    assert restored.root_goal_id == "goal:root"
    assert restored.root_goal_content_id == "baguqeera-root-goal"
    assert restored.satisfaction_formula_id == "formula:root-satisfied"
    assert restored.assumption_ids == (
        "assumption:reviewed-a",
        "assumption:reviewed-b",
    )
    assert restored.evidence_requirement_ids == ("evidence:review", "evidence:test")
    assert restored.vocabulary_profile_id == "supervisor-reviewed-dcec-tdfol"
    assert restored.repository_tree_id == "tree:base"
    assert restored.scope_ids == (
        "path:ipfs_accelerate_py",
        "path:test/api",
    )
    assert restored.policy_digest == policy.policy_digest
    restored.require_policy(policy)


def test_draft_round_trip_is_explicitly_unverified_and_content_addressed() -> None:
    draft = _draft()
    restored = GoalDecompositionDraft.from_json(draft.to_json())

    assert restored == draft
    assert restored.draft_id == draft.content_id
    assert restored.request_id == draft.request.request_id
    assert restored.policy_digest == draft.policy.policy_digest
    assert restored.trust is GoalDevelopmentTrust.UNVERIFIED
    assert restored.assurance is AssuranceLevel.UNVERIFIED
    assert restored.authority is GoalDevelopmentAuthority.NONE
    assert restored.verified is False
    assert restored.admitted is False
    assert restored.implementation_conformant is False
    assert restored.complete is False
    assert restored.to_dict()["schema"] == GOAL_DECOMPOSITION_DRAFT_SCHEMA
    assert restored.to_dict()["proof_claimed"] is False
    assert restored.to_dict()["admission_claimed"] is False
    assert restored.to_dict()["completion_claimed"] is False


def test_contracts_are_frozen_and_set_semantics_are_deterministic() -> None:
    first_policy = _policy()
    second_policy = _policy()
    first_request = _request(first_policy)
    second_request = _request(
        second_policy,
        assumption_ids=("assumption:reviewed-a", "assumption:reviewed-b"),
        evidence_requirement_ids=("evidence:test", "evidence:review"),
        scope_ids=("path:ipfs_accelerate_py", "path:test/api"),
    )
    first = _draft(
        first_policy,
        first_request,
        proposals=(
            _proposal(
                "subgoal:tests",
                parent_id="subgoal:contracts",
                depends_on=("subgoal:contracts",),
            ),
            _proposal("subgoal:contracts"),
        ),
    )
    second = _draft(second_policy, second_request)

    assert first_policy.content_id == second_policy.content_id
    assert first_request.request_id == second_request.request_id
    assert first.to_json() == second.to_json()
    assert first.draft_id == second.draft_id
    with pytest.raises(FrozenInstanceError):
        first.request = second_request  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.proposals[0].parent_id = "goal:changed"  # type: ignore[misc]


def test_policy_rejects_nonpositive_bounds_and_new_assumption_authority() -> None:
    for name in (
        "max_depth",
        "max_breadth",
        "max_proposals",
        "max_bytes",
        "max_tokens",
    ):
        with pytest.raises(ContractValidationError, match=name):
            _policy(**{name: 0})

    with pytest.raises(ContractValidationError, match="new assumptions"):
        _policy(allow_new_assumptions=True)


@pytest.mark.parametrize(
    ("policy_changes", "proposals", "message"),
    [
        (
            {"max_depth": 1},
            (
                _proposal("subgoal:one"),
                _proposal("subgoal:two", parent_id="subgoal:one"),
            ),
            "max_depth",
        ),
        (
            {"max_breadth": 1},
            (_proposal("subgoal:one"), _proposal("subgoal:two")),
            "max_breadth",
        ),
        (
            {"max_proposals": 1},
            (_proposal("subgoal:one"), _proposal("subgoal:two")),
            "max_proposals",
        ),
    ],
)
def test_draft_enforces_depth_breadth_and_count_bounds(
    policy_changes: dict[str, int],
    proposals: tuple[GoalDecompositionProposal, ...],
    message: str,
) -> None:
    policy = _policy(**policy_changes)
    with pytest.raises(ContractValidationError, match=message):
        _draft(policy, _request(policy), proposals=proposals)


def test_draft_enforces_canonical_byte_and_token_bounds() -> None:
    large = _proposal("subgoal:large", title="é" * 1_000)
    byte_policy = _policy(max_bytes=500)
    with pytest.raises(ContractValidationError, match="max_bytes"):
        _draft(byte_policy, _request(byte_policy), proposals=(large,))

    token_policy = _policy(max_tokens=10)
    with pytest.raises(ContractValidationError, match="max_tokens"):
        _draft(token_policy, _request(token_policy), proposals=(_proposal("x"),))

    producer_count_policy = _policy(max_tokens=100)
    with pytest.raises(ContractValidationError, match="max_tokens"):
        _draft(
            producer_count_policy,
            _request(producer_count_policy),
            proposals=(_proposal("x"),),
            token_count=101,
        )


def test_draft_rejects_unknown_parents_cycles_and_dependencies() -> None:
    with pytest.raises(ContractValidationError, match="unknown parent"):
        _draft(proposals=(_proposal("subgoal:x", parent_id="subgoal:missing"),))
    with pytest.raises(ContractValidationError, match="parent graph.*acyclic"):
        _draft(
            proposals=(
                _proposal("subgoal:a", parent_id="subgoal:b"),
                _proposal("subgoal:b", parent_id="subgoal:a"),
            )
        )
    with pytest.raises(ContractValidationError, match="unknown dependencies"):
        _draft(
            proposals=(
                _proposal("subgoal:a", depends_on=("subgoal:missing",)),
            )
        )
    with pytest.raises(ContractValidationError, match="dependency graph.*acyclic"):
        _draft(
            proposals=(
                _proposal("subgoal:a", depends_on=("subgoal:b",)),
                _proposal("subgoal:b", depends_on=("subgoal:a",)),
            )
        )


def test_off_and_repair_only_modes_fail_closed() -> None:
    off = _policy(GoalDevelopmentMode.OFF)
    with pytest.raises(ContractValidationError, match="off mode"):
        _draft(off, _request(off), proposals=(_proposal("subgoal:x"),))

    repair = _policy(GoalDevelopmentMode.REPAIR_ONLY)
    with pytest.raises(ContractValidationError, match="draft being repaired"):
        _request(repair, repair_draft_id="")
    request = _request(repair)
    assert request.repair_draft_id == "baguqeera-prior-draft"
    with pytest.raises(ContractValidationError, match="only valid"):
        _request(_policy(), repair_draft_id="baguqeera-unexpected")


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("root_goal_id", "goal:mutated"),
        ("root_goal_content_id", "baguqeera-mutated"),
        ("satisfaction_formula_id", "formula:mutated"),
        ("assumption_ids", ["assumption:hidden"]),
        ("evidence_requirement_ids", ["evidence:different"]),
        ("vocabulary_profile_id", "unreviewed-vocabulary"),
        ("vocabulary_version", 2),
        ("repository_tree_id", "tree:stale"),
        ("scope_ids", ["path:outside"]),
        ("policy_digest", "baguqeera-other-policy"),
    ],
)
def test_draft_decode_rejects_frozen_root_and_context_mutation(
    field_name: str, replacement: object
) -> None:
    draft = _draft()
    payload = draft.to_record()
    payload["request"][field_name] = replacement

    with pytest.raises(ContractValidationError, match="identity|policy digest"):
        GoalDecompositionDraft.from_dict(payload)


def test_draft_rejects_hidden_assumptions_and_frozen_surface_expansion() -> None:
    with pytest.raises(ContractValidationError, match="hidden or new assumption"):
        _draft(
            proposals=(
                _proposal(
                    "subgoal:x",
                    assumptions=("assumption:reviewed-a", "assumption:hidden"),
                ),
            )
        )
    with pytest.raises(ContractValidationError, match="evidence requirements"):
        GoalDecompositionDraft(
            request=_request(),
            policy=_policy(),
            proposals=(
                GoalDecompositionProposal(
                    proposal_id="subgoal:x",
                    parent_id="goal:root",
                    satisfaction_formula_id="formula:x",
                    evidence_requirement_ids=("evidence:invented",),
                    scope_ids=("path:ipfs_accelerate_py",),
                ),
            ),
            producer_id="provider:model",
        )
    with pytest.raises(ContractValidationError, match="development scope"):
        GoalDecompositionDraft(
            request=_request(),
            policy=_policy(),
            proposals=(
                GoalDecompositionProposal(
                    proposal_id="subgoal:x",
                    parent_id="goal:root",
                    satisfaction_formula_id="formula:x",
                    evidence_requirement_ids=("evidence:test",),
                    scope_ids=("path:outside",),
                ),
            ),
            producer_id="provider:model",
        )


@pytest.mark.parametrize(
    ("field_name", "claim"),
    [
        ("proof_claimed", True),
        ("admission_claimed", True),
        ("admitted", True),
        ("implementation_conformance_claimed", True),
        ("implementation_conformant", True),
        ("completion_claimed", True),
        ("complete", True),
    ],
)
def test_draft_cannot_claim_proof_admission_conformance_or_completion(
    field_name: str, claim: bool
) -> None:
    payload = _draft().to_dict()
    payload[field_name] = claim
    with pytest.raises(ContractValidationError, match="cannot claim"):
        GoalDecompositionDraft.from_dict(payload)


def test_draft_rejects_invalid_assurance_authority_and_unknown_fields() -> None:
    for name, value, message in (
        ("trust", "trusted_fact", "unverified"),
        ("assurance", "kernel_verified", "assurance"),
        ("authority", "kernel", "authority"),
    ):
        payload = _draft().to_dict()
        payload[name] = value
        with pytest.raises(ContractValidationError, match=message):
            GoalDecompositionDraft.from_dict(payload)

    payload = _draft().to_dict()
    payload["canonical_root_override"] = "goal:attacker"
    with pytest.raises(ContractValidationError, match="unsupported fields"):
        GoalDecompositionDraft.from_dict(payload)


def test_proposal_receipt_round_trips_and_validates_all_frozen_bindings() -> None:
    draft = _draft()
    receipt = GoalDevelopmentProposalReceipt.for_draft(
        draft, validator_id="validator:goal-schema-v1"
    )
    restored = GoalDevelopmentProposalReceipt.from_json(receipt.to_json())

    assert restored == receipt
    assert restored.receipt_id == receipt.content_id
    assert restored.authority is GoalDevelopmentAuthority.DETERMINISTIC_VALIDATOR
    assert restored.assurance is AssuranceLevel.UNVERIFIED
    assert restored.decision is GoalProposalDecision.ACCEPTED
    restored.validate_draft(draft)

    mutated = replace(receipt, repository_tree_id="tree:other")
    with pytest.raises(ContractValidationError, match="frozen draft bindings"):
        mutated.validate_draft(draft)


def test_receipts_reject_invalid_authority_and_forged_identity() -> None:
    proposal = GoalDevelopmentProposalReceipt.for_draft(
        _draft(), validator_id="validator:goal-schema-v1"
    )
    payload = proposal.to_record()
    payload["authority"] = "kernel"
    with pytest.raises(ContractValidationError, match="invalid authority"):
        GoalDevelopmentProposalReceipt.from_dict(payload)

    admission = GoalDevelopmentAdmissionReceipt.for_proposal(
        proposal,
        mode=GoalDevelopmentMode.SHADOW,
        admitter_id="supervisor:objective-daemon",
        decision=GoalAdmissionDecision.NOT_ADMITTED,
        reason_codes=("shadow_mode",),
    )
    payload = admission.to_record()
    payload["authority"] = "model"
    with pytest.raises(ContractValidationError, match="invalid authority"):
        GoalDevelopmentAdmissionReceipt.from_dict(payload)

    payload = admission.to_record()
    payload["content_id"] = "baguqeera-forged"
    with pytest.raises(ContractValidationError, match="identity"):
        GoalDevelopmentAdmissionReceipt.from_dict(payload)


def test_admission_receipts_round_trip_and_enforce_mode_authority() -> None:
    auto_policy = _policy(GoalDevelopmentMode.AUTO_SAFE)
    draft = _draft(auto_policy, _request(auto_policy))
    proposal = GoalDevelopmentProposalReceipt.for_draft(
        draft, validator_id="validator:goal-schema-v1"
    )
    admitted = GoalDevelopmentAdmissionReceipt.for_proposal(
        proposal,
        mode=GoalDevelopmentMode.AUTO_SAFE,
        admitter_id="supervisor:objective-daemon",
        decision=GoalAdmissionDecision.ADMITTED,
        authoritative_receipt_ids=("receipt:refinement-kernel",),
    )
    restored = GoalDevelopmentAdmissionReceipt.from_json(admitted.to_json())

    assert restored == admitted
    assert restored.admitted is True
    assert restored.authority is GoalDevelopmentAuthority.SUPERVISOR_ADMISSION
    assert restored.proof_assurance is AssuranceLevel.UNVERIFIED
    assert restored.to_dict()["proof_claimed"] is False
    assert restored.to_dict()["implementation_conformant"] is False
    assert restored.to_dict()["complete"] is False
    restored.validate_proposal_receipt(proposal)

    with pytest.raises(ContractValidationError, match="authoritative receipts"):
        replace(admitted, authoritative_receipt_ids=())
    with pytest.raises(ContractValidationError, match="only auto_safe"):
        replace(admitted, mode=GoalDevelopmentMode.ASSIST)
    with pytest.raises(ContractValidationError, match="does not match"):
        GoalDevelopmentAdmissionReceipt.for_proposal(
            proposal,
            mode=GoalDevelopmentMode.SHADOW,
            admitter_id="supervisor:objective-daemon",
            decision=GoalAdmissionDecision.NOT_ADMITTED,
            reason_codes=("shadow_mode",),
        )
    with pytest.raises(ContractValidationError, match="frozen proposal bindings"):
        replace(admitted, repository_tree_id="tree:other").validate_proposal_receipt(
            proposal
        )


def test_shadow_assist_and_repair_receipts_cannot_mutate_objective_graph() -> None:
    for mode, decision in (
        (GoalDevelopmentMode.SHADOW, GoalAdmissionDecision.NOT_ADMITTED),
        (GoalDevelopmentMode.ASSIST, GoalAdmissionDecision.REVIEW_REQUIRED),
        (GoalDevelopmentMode.REPAIR_ONLY, GoalAdmissionDecision.NOT_ADMITTED),
    ):
        policy = _policy(mode)
        draft = _draft(policy, _request(policy))
        proposal = GoalDevelopmentProposalReceipt.for_draft(
            draft, validator_id="validator:goal-schema-v1"
        )
        receipt = GoalDevelopmentAdmissionReceipt.for_proposal(
            proposal,
            mode=mode,
            admitter_id="supervisor:objective-daemon",
            decision=decision,
            reason_codes=(f"{mode.value}_mode",),
        )
        assert receipt.admitted is False
        assert GoalDevelopmentAdmissionReceipt.from_json(receipt.to_json()) == receipt

        with pytest.raises(ContractValidationError, match="only auto_safe"):
            replace(
                receipt,
                decision=GoalAdmissionDecision.ADMITTED,
                authoritative_receipt_ids=("receipt:claimed",),
                reason_codes=(),
            )
