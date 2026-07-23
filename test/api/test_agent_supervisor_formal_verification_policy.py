from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    OverrideReceipt,
    OverrideReceiptStore,
    PolicyGateDecision,
    PolicySelection,
    PolicyValidationError,
    ProofOutcome,
    ProofPolicyRule,
    ProofResultStatus,
    RiskLevel,
    RolloutMode,
    RolloutTransitionReceipt,
    ValidationOutcome,
    default_formal_verification_policy,
)

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
TREE = "git-tree:policy-test"


def _rule(**changes: object) -> ProofPolicyRule:
    values = {
        "rule_id": "supervisor-high-risk",
        "path_patterns": ("src/agent_supervisor/**",),
        "minimum_risk": RiskLevel.HIGH,
        "invariant_classes": ("lease_safety",),
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "fallback_validations": ("pytest:lease-regression",),
        "allow_fallback": True,
    }
    values.update(changes)
    return ProofPolicyRule(**values)


def _policy(
    mode: RolloutMode = RolloutMode.ENFORCEMENT,
    *,
    rules: tuple[ProofPolicyRule, ...] | None = None,
    **changes: object,
) -> FormalVerificationPolicy:
    values = {
        "name": "supervisor-proof-policy",
        "version": "7",
        "rollout_mode": mode,
        "rules": rules or (_rule(),),
        "canary_salt": "test-canary-v1",
        "minimum_promotion_observations": 2,
    }
    values.update(changes)
    return FormalVerificationPolicy(**values)


def _change(path: str = "src/agent_supervisor/lease.py") -> ChangedScope:
    return ChangedScope(
        path=path,
        ast_scope_ids=("module:lease", "class:LeaseStore.acquire"),
        risk=RiskLevel.CRITICAL,
        invariant_classes=("lease_safety", "state_transition"),
    )


def _selection(policy: FormalVerificationPolicy):
    return policy.select((_change(),), repository_tree_id=TREE)


def _outcome(
    selection,
    status: ProofResultStatus,
    *,
    assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED,
    receipt_id: str = "",
) -> ProofOutcome:
    return ProofOutcome(
        requirement_id=selection.requirements[0].requirement_id,
        status=status,
        authoritative_assurance=assurance,
        receipt_id=receipt_id,
    )


def test_policy_maps_path_ast_risk_and_invariant_to_assurance_and_fallback() -> None:
    policy = _policy(
        rules=(
            _rule(
                rule_id="path-risk",
                invariant_classes=(),
                required_assurance=AssuranceLevel.SOLVER_CHECKED,
            ),
            _rule(
                rule_id="ast-invariant",
                path_patterns=(),
                ast_scope_patterns=("class:LeaseStore.*",),
                minimum_risk=RiskLevel.MEDIUM,
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                fallback_validations=("pytest:lease-model",),
            ),
        )
    )

    selection = _selection(policy)
    requirement = selection.requirements[0]

    assert requirement.matched_rule_ids == ("ast-invariant", "path-risk")
    assert requirement.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert requirement.fallback_validations == (
        "pytest:lease-model",
        "pytest:lease-regression",
    )
    assert requirement.allow_fallback is True
    assert selection.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert selection.fallback_validations == requirement.fallback_validations

    # Each configured selector dimension is conjunctive.
    unrelated = policy.select(
        (
            ChangedScope(
                path="src/agent_supervisor/lease.py",
                ast_scope_ids=("function:unrelated",),
                risk=RiskLevel.LOW,
                invariant_classes=("formatting",),
            ),
        )
    )
    assert unrelated.requirements == ()
    assert len(unrelated.unprotected_scope_ids) == 1


def test_overlapping_rules_merge_conservatively_and_cannot_weaken_fallback() -> None:
    policy = _policy(
        rules=(
            _rule(rule_id="fallback-allowed"),
            _rule(
                rule_id="critical-no-fallback",
                minimum_risk=RiskLevel.CRITICAL,
                fallback_validations=(),
                allow_fallback=False,
            ),
        )
    )

    requirement = _selection(policy).requirements[0]

    assert requirement.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert requirement.allow_fallback is False


@pytest.mark.parametrize(
    "mode",
    (RolloutMode.DISABLED, RolloutMode.SHADOW),
)
def test_nonblocking_modes_report_failure_without_blocking(mode: RolloutMode) -> None:
    policy = _policy(mode)
    selection = _selection(policy)
    decision = policy.evaluate_gate(
        selection, _outcome(selection, ProofResultStatus.UNAVAILABLE)
    )
    result = decision.results[0]

    assert decision.allowed is True
    assert result.requirement_satisfied is False
    if mode is RolloutMode.DISABLED:
        assert result.would_block is False
        assert "proof_policy_disabled" in result.reason_codes
    else:
        assert result.would_block is True
        assert "shadow_would_block" in result.reason_codes


def test_canary_only_blocks_configured_or_deterministically_selected_paths() -> None:
    policy = _policy(
        RolloutMode.CANARY,
        canary_path_patterns=("src/agent_supervisor/lease.py",),
        canary_percent=0,
    )
    selected = _selection(policy)
    selected_decision = policy.evaluate_gate(
        selected, _outcome(selected, ProofResultStatus.MISSING)
    )

    other = policy.select(
        (_change("src/agent_supervisor/other.py"),),
        repository_tree_id=TREE,
    )
    other_decision = policy.evaluate_gate(
        other, _outcome(other, ProofResultStatus.MISSING)
    )

    assert selected_decision.allowed is False
    assert selected_decision.results[0].effective_mode is RolloutMode.CANARY
    assert other_decision.allowed is True
    assert other_decision.results[0].effective_mode is RolloutMode.SHADOW
    assert other_decision.results[0].would_block is True

    full_canary = _policy(RolloutMode.CANARY, canary_percent=100)
    full_selection = _selection(full_canary)
    assert (
        full_canary.effective_mode(full_selection.requirements[0]) is RolloutMode.CANARY
    )


@pytest.mark.parametrize(
    "status",
    (
        ProofResultStatus.UNSUPPORTED,
        ProofResultStatus.UNAVAILABLE,
        ProofResultStatus.TIMED_OUT,
        ProofResultStatus.INCONCLUSIVE,
        ProofResultStatus.ERROR,
        ProofResultStatus.MISSING,
    ),
)
def test_nonproof_results_never_silently_satisfy_enforcement(
    status: ProofResultStatus,
) -> None:
    policy = _policy(
        rules=(
            _rule(
                fallback_validations=(),
                allow_fallback=False,
            ),
        )
    )
    selection = _selection(policy)
    # Even a forged high assurance field and receipt cannot turn a non-proof
    # semantic status into proof success.
    outcome = _outcome(
        selection,
        status,
        assurance=AssuranceLevel.ATTESTED,
        receipt_id="receipt:provider-claimed-success",
    )

    decision = policy.evaluate_gate(selection, outcome)

    assert decision.allowed is False
    assert decision.results[0].proof_satisfied is False
    assert f"proof_{status.value}" in decision.results[0].reason_codes


def test_proved_result_requires_durable_receipt_and_required_assurance() -> None:
    policy = _policy()
    selection = _selection(policy)

    no_receipt = policy.evaluate_gate(
        selection,
        _outcome(
            selection,
            ProofResultStatus.PROVED,
            assurance=AssuranceLevel.KERNEL_VERIFIED,
        ),
    )
    solver_only = policy.evaluate_gate(
        selection,
        _outcome(
            selection,
            ProofResultStatus.PROVED,
            assurance=AssuranceLevel.SOLVER_CHECKED,
            receipt_id="receipt:solver",
        ),
    )
    verified = policy.evaluate_gate(
        selection,
        _outcome(
            selection,
            ProofResultStatus.PROVED,
            assurance=AssuranceLevel.KERNEL_VERIFIED,
            receipt_id="receipt:kernel",
        ),
    )

    assert no_receipt.allowed is False
    assert "proof_receipt_missing" in no_receipt.results[0].reason_codes
    assert solver_only.allowed is False
    assert "required_assurance_not_satisfied" in solver_only.results[0].reason_codes
    assert verified.allowed is True
    assert verified.results[0].proof_satisfied is True


def test_unsupported_can_only_use_an_explicit_complete_fallback() -> None:
    policy = _policy()
    selection = _selection(policy)
    unsupported = _outcome(selection, ProofResultStatus.UNSUPPORTED)

    missing = policy.evaluate_gate(selection, unsupported)
    failed = policy.evaluate_gate(
        selection,
        unsupported,
        validations=(
            ValidationOutcome(
                "pytest:lease-regression",
                False,
                receipt_id="validation:failed",
            ),
        ),
    )
    passed = policy.evaluate_gate(
        selection,
        unsupported,
        validations={
            "pytest:lease-regression": ValidationOutcome(
                "pytest:lease-regression",
                True,
                receipt_id="validation:passed",
            )
        },
    )

    assert missing.allowed is False
    assert failed.allowed is False
    assert passed.allowed is True
    assert passed.results[0].proof_satisfied is False
    assert passed.results[0].fallback_satisfied is True
    assert passed.results[0].reason_codes == (
        "explicit_fallback_validations_satisfied",
    )


def test_override_requires_bounded_actor_reason_tree_policy_and_expiration() -> None:
    policy = _policy()
    selection = _selection(policy)
    outcome = _outcome(selection, ProofResultStatus.TIMED_OUT)
    override = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id=TREE,
        paths=("src/agent_supervisor/lease.py",),
        ast_scope_ids=("module:lease", "class:LeaseStore.acquire"),
        invariant_classes=("lease_safety", "state_transition"),
        actor="did:key:operator-7",
        reason="Emergency provider outage; incident INC-247.",
        ticket_id="INC-247",
        ttl_seconds=900,
        now=NOW,
    )

    decision = policy.evaluate_gate(
        selection,
        outcome,
        override=override,
        now=NOW + timedelta(minutes=1),
    )

    assert decision.allowed is True
    assert decision.override_receipt_id == override.receipt_id
    result = decision.results[0]
    assert result.requirement_satisfied is False
    assert result.proof_status is ProofResultStatus.TIMED_OUT
    assert result.would_block is True
    assert "bounded_override_applied" in result.reason_codes

    with pytest.raises(PolicyValidationError, match="actor is required"):
        OverrideReceipt.create(
            policy_id=policy.policy_id,
            repository_tree_id=TREE,
            paths=("src/agent_supervisor/lease.py",),
            actor="",
            reason="Emergency",
            ttl_seconds=60,
            now=NOW,
        )
    with pytest.raises(PolicyValidationError, match="exact, not globs"):
        OverrideReceipt.create(
            policy_id=policy.policy_id,
            repository_tree_id=TREE,
            paths=("src/**",),
            actor="operator",
            reason="Emergency",
            ttl_seconds=60,
            now=NOW,
        )


def test_expired_wrong_tree_and_out_of_scope_overrides_fail_closed() -> None:
    policy = _policy()
    selection = _selection(policy)
    outcome = _outcome(selection, ProofResultStatus.UNAVAILABLE)

    expired = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id=TREE,
        paths=("src/agent_supervisor/lease.py",),
        actor="operator",
        reason="Temporary incident mitigation",
        ttl_seconds=60,
        now=NOW,
    )
    expired_decision = policy.evaluate_gate(
        selection,
        outcome,
        override=expired,
        now=NOW + timedelta(seconds=61),
    )
    assert expired_decision.allowed is False
    assert expired_decision.override_rejection_reasons == ("override_expired",)

    wrong_tree = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id="git-tree:old",
        paths=("src/agent_supervisor/lease.py",),
        actor="operator",
        reason="Temporary incident mitigation",
        ttl_seconds=60,
        now=NOW,
    )
    wrong_tree_decision = policy.evaluate_gate(
        selection, outcome, override=wrong_tree, now=NOW
    )
    assert wrong_tree_decision.allowed is False
    assert "override_tree_mismatch" in wrong_tree_decision.override_rejection_reasons

    wrong_path = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id=TREE,
        paths=("src/agent_supervisor/unrelated.py",),
        actor="operator",
        reason="Temporary incident mitigation",
        ttl_seconds=60,
        now=NOW,
    )
    wrong_path_decision = policy.evaluate_gate(
        selection, outcome, override=wrong_path, now=NOW
    )
    assert wrong_path_decision.allowed is False
    assert "override_does_not_cover_requirement" in (
        wrong_path_decision.results[0].reason_codes
    )


def test_override_receipt_is_canonical_content_addressed_and_durable(tmp_path) -> None:
    policy = _policy()
    receipt = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id=TREE,
        paths=("src/agent_supervisor/lease.py",),
        actor="did:key:operator",
        reason="Bounded operational exception",
        ttl_seconds=300,
        now=NOW,
    )
    store = OverrideReceiptStore(tmp_path / "overrides")

    path = store.persist(receipt)
    loaded = store.load(receipt.receipt_id)

    assert loaded == receipt
    assert path.name == f"{receipt.receipt_id}.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert store.persist(receipt) == path  # idempotent
    assert json.loads(path.read_text())["actor"] == "did:key:operator"

    tampered = receipt.to_dict()
    tampered["reason"] = "Different reason"
    tampered["receipt_id"] = receipt.receipt_id
    with pytest.raises(PolicyValidationError, match="identity does not match"):
        OverrideReceipt.from_dict(tampered)


def test_rollout_promotion_is_adjacent_evidence_gated_and_content_addressed() -> None:
    shadow = _policy(RolloutMode.SHADOW)
    receipt = RolloutTransitionReceipt(
        policy_id=shadow.policy_id,
        from_mode=RolloutMode.SHADOW,
        target_mode=RolloutMode.CANARY,
        actor="release-manager",
        reason="Shadow observations meet the reviewed rollout threshold.",
        issued_at=NOW.isoformat(),
        observation_count=2,
        blocking_result_count=0,
        override_count=0,
        evidence_receipt_ids=("observation:1", "observation:2"),
    )

    canary = shadow.transition(receipt)

    assert canary.rollout_mode is RolloutMode.CANARY
    assert canary.policy_id != shadow.policy_id
    assert receipt.receipt_id.startswith("baguqeera")

    with pytest.raises(PolicyValidationError, match="insufficient observations"):
        shadow.transition(
            RolloutTransitionReceipt(
                policy_id=shadow.policy_id,
                from_mode=RolloutMode.SHADOW,
                target_mode=RolloutMode.CANARY,
                actor="release-manager",
                reason="Too little evidence",
                issued_at=NOW.isoformat(),
                observation_count=1,
                evidence_receipt_ids=("observation:1",),
            )
        )
    with pytest.raises(PolicyValidationError, match="advances one stage"):
        RolloutTransitionReceipt(
            policy_id=shadow.policy_id,
            from_mode=RolloutMode.SHADOW,
            target_mode=RolloutMode.ENFORCEMENT,
            actor="release-manager",
            reason="Attempted skipped stage",
            issued_at=NOW.isoformat(),
        )


def test_rollout_rollback_is_explicit_and_does_not_need_success_evidence() -> None:
    enforcement = _policy(RolloutMode.ENFORCEMENT)
    receipt = RolloutTransitionReceipt(
        policy_id=enforcement.policy_id,
        from_mode=RolloutMode.ENFORCEMENT,
        target_mode=RolloutMode.SHADOW,
        actor="incident-commander",
        reason="Rollback after elevated provider error rate.",
        issued_at=NOW.isoformat(),
    )

    rolled_back = enforcement.transition(receipt)

    assert rolled_back.rollout_mode is RolloutMode.SHADOW
    assert rolled_back.policy_id != enforcement.policy_id


def test_policy_and_selection_round_trip_with_deterministic_identities() -> None:
    policy = _policy()
    same_policy = FormalVerificationPolicy.from_dict(policy.to_dict())
    selection = _selection(policy)
    outcome = _outcome(selection, ProofResultStatus.UNAVAILABLE)
    decision = policy.evaluate_gate(selection, outcome)

    assert same_policy == policy
    assert same_policy.policy_id == policy.policy_id
    assert PolicySelection.from_json(selection.to_json()) == selection
    assert ProofOutcome.from_json(outcome.to_json()) == outcome
    assert PolicyGateDecision.from_json(decision.to_json()) == decision
    assert selection.to_json() == selection.canonical_json()
    assert selection.selection_id.startswith("baguqeera")
    assert policy.select((_change(),), repository_tree_id=TREE) == selection


def test_baseline_policy_protects_only_modeled_high_risk_changes() -> None:
    policy = default_formal_verification_policy(RolloutMode.SHADOW)
    protected = policy.select(
        (
            ChangedScope(
                path=(
                    "ipfs_datasets_py/ipfs_accelerate_py/ipfs_accelerate_py/"
                    "agent_supervisor/lease_coordination.py"
                ),
                risk=RiskLevel.CRITICAL,
                invariant_classes=("lease_safety",),
            ),
        )
    )
    unrelated = policy.select(
        (
            ChangedScope(
                path="docs/readme.md",
                risk=RiskLevel.LOW,
                invariant_classes=(),
            ),
        )
    )

    assert protected.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert unrelated.requirements == ()


def test_unsafe_paths_and_accidental_global_rules_are_rejected() -> None:
    with pytest.raises(PolicyValidationError, match="repository-relative"):
        ChangedScope(path="/etc/passwd")
    with pytest.raises(PolicyValidationError, match="safe repository path"):
        ChangedScope(path="../outside.py")
    with pytest.raises(PolicyValidationError, match="selector"):
        ProofPolicyRule(
            rule_id="accidental-global",
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        )

    explicit_global = ProofPolicyRule(
        rule_id="reviewed-global",
        required_assurance=AssuranceLevel.SOLVER_CHECKED,
        match_all=True,
    )
    assert explicit_global.matches(ChangedScope(path="any/file.py"))


def test_path_globs_do_not_let_single_star_cross_directories() -> None:
    direct_only = ProofPolicyRule(
        rule_id="direct-python",
        path_patterns=("src/*.py",),
        required_assurance=AssuranceLevel.SOLVER_CHECKED,
    )
    recursive = ProofPolicyRule(
        rule_id="recursive-python",
        path_patterns=("src/**/*.py",),
        required_assurance=AssuranceLevel.SOLVER_CHECKED,
    )

    assert direct_only.matches(ChangedScope(path="src/direct.py"))
    assert not direct_only.matches(ChangedScope(path="src/nested/module.py"))
    assert recursive.matches(ChangedScope(path="src/direct.py"))
    assert recursive.matches(ChangedScope(path="src/nested/deep/module.py"))
