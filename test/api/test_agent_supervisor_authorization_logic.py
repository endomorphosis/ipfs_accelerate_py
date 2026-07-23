from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.authorization_logic import (
    AUTHORIZATION_CONFORMANCE_FIXTURES,
    AuthorizationChecker,
    AuthorizationDecision,
    AuthorizationGrant,
    AuthorizationPolicy,
    AuthorizationRequest,
    AuthorizationValidationError,
    AuthorizationVerdict,
    Capability,
    DatalogAuthorizationAdapter,
    DenialReason,
    EngineSupportStatus,
    GeneratedCodeCorrectness,
    LaneStatus,
    ReferenceAuthorizationEvaluator,
    Revocation,
    SecPALAuthorizationAdapter,
    check_authorization,
    evaluate_authorization,
    probe_authorization_engines,
    render_datalog_policy,
    render_secpal_policy,
)
from ipfs_accelerate_py.agent_supervisor.prover_matrix_registry import (
    CommandRequest,
    CommandResult,
)

NOW = 1_000


def _grant(
    *,
    statement_id: str = "root-grant",
    issuer: str = "root",
    subject: str = "worker",
    capability: Capability = Capability.EXECUTE_TASK,
    task_scope: tuple[str, ...] = ("REF-284",),
    delegation_depth: int = 0,
    parent_statement_id: str | None = None,
    lease_scope: tuple[str, ...] = ("lease-7",),
    worktree_scope: tuple[str, ...] = ("tree-7",),
    path_scope: tuple[str, ...] = ("src",),
    fencing_epoch: int | None = 7,
    proof_authorities: tuple[str, ...] = (),
    override_scopes: tuple[str, ...] = (),
    not_before_ms: int = 100,
    expires_at_ms: int | None = 2_000,
) -> AuthorizationGrant:
    return AuthorizationGrant(
        statement_id=statement_id,
        issuer=issuer,
        subject=subject,
        capability=capability,
        task_scope=task_scope,
        delegation_depth=delegation_depth,
        parent_statement_id=parent_statement_id,
        lease_scope=lease_scope,
        worktree_scope=worktree_scope,
        path_scope=path_scope,
        fencing_epoch=fencing_epoch,
        proof_authorities=proof_authorities,
        override_scopes=override_scopes,
        not_before_ms=not_before_ms,
        expires_at_ms=expires_at_ms,
    )


def _policy(
    *grants: AuthorizationGrant,
    revocations: tuple[Revocation, ...] = (),
    epoch: int = 7,
    lease_id: str = "lease-7",
) -> AuthorizationPolicy:
    return AuthorizationPolicy(
        policy_id="test-authority",
        version="1",
        trusted_roots=("root",),
        grants=grants or (_grant(),),
        revocations=revocations,
        current_fencing_epochs={"REF-284": epoch},
        current_lease_ids={"REF-284": lease_id},
    )


def _request(**changes: object) -> AuthorizationRequest:
    values: dict[str, object] = {
        "principal": "worker",
        "capability": Capability.EXECUTE_TASK,
        "task_id": "REF-284",
        "evaluated_at_ms": NOW,
        "lease_id": "lease-7",
        "worktree_id": "tree-7",
        "path": "src/authorization.py",
        "fencing_epoch": 7,
    }
    values.update(changes)
    return AuthorizationRequest(**values)


def test_reviewed_conformance_fixtures_cover_required_semantics() -> None:
    expected = {
        "positive": AuthorizationVerdict.PERMIT,
        "negative": AuthorizationVerdict.DENY,
        "revocation": AuthorizationVerdict.DENY,
        "confused_deputy": AuthorizationVerdict.DENY,
        "stale_lease": AuthorizationVerdict.DENY,
    }

    assert {
        fixture.category: fixture.expected_verdict
        for fixture in AUTHORIZATION_CONFORMANCE_FIXTURES
    } == expected
    for fixture in AUTHORIZATION_CONFORMANCE_FIXTURES:
        decision = evaluate_authorization(fixture.policy, fixture.request)
        assert decision.verdict is fixture.expected_verdict
        assert decision.policy_identity == fixture.policy.content_id
        assert decision.request_identity == fixture.request.content_id


def test_policy_request_grant_and_decision_are_canonical_round_trips() -> None:
    policy = _policy()
    request = _request()
    decision = evaluate_authorization(policy, request)

    assert (
        AuthorizationGrant.from_dict(policy.grants[0].to_record()) == policy.grants[0]
    )
    assert AuthorizationPolicy.from_dict(policy.to_record()) == policy
    assert AuthorizationRequest.from_dict(request.to_record()) == request
    assert AuthorizationDecision.from_dict(decision.to_record()) == decision
    assert json.loads(policy.to_json()) == policy.to_dict()

    forged = decision.to_record()
    forged["establishes_generated_code_correctness"] = True
    with pytest.raises(
        AuthorizationValidationError,
        match="cannot establish generated-code correctness",
    ):
        AuthorizationDecision.from_dict(forged)


def test_parent_linked_delegation_preserves_capability_and_narrows_scope() -> None:
    root = _grant(
        statement_id="root",
        subject="supervisor",
        task_scope=("*",),
        delegation_depth=2,
        lease_scope=("*",),
        worktree_scope=("*",),
        path_scope=("src",),
    )
    worker = _grant(
        statement_id="worker",
        issuer="supervisor",
        subject="worker",
        parent_statement_id="root",
        delegation_depth=1,
        path_scope=("src/agent_supervisor",),
    )
    policy = _policy(root, worker)

    allowed = evaluate_authorization(
        policy, _request(path="src/agent_supervisor/authorization.py")
    )
    denied = evaluate_authorization(policy, _request(path="src/unrelated.py"))

    assert allowed.permitted
    assert allowed.matched_statement_ids == ("root", "worker")
    assert denied.reason is DenialReason.PATH_SCOPE_MISMATCH


def test_delegation_depth_cannot_be_reused_or_amplified() -> None:
    root = _grant(
        statement_id="root",
        subject="supervisor",
        delegation_depth=1,
    )
    amplified = _grant(
        statement_id="amplified",
        issuer="supervisor",
        parent_statement_id="root",
        delegation_depth=1,
    )
    decision = evaluate_authorization(_policy(root, amplified), _request())

    assert not decision.permitted
    assert decision.reason is DenialReason.DELEGATION_DEPTH_EXCEEDED


def test_confused_deputy_cannot_mix_principal_capability_or_parent_authority() -> None:
    execute_for_worker = _grant(subject="worker")
    merge_for_deputy = _grant(
        statement_id="deputy-merge",
        subject="deputy",
        capability=Capability.MERGE,
    )
    policy = _policy(execute_for_worker, merge_for_deputy)

    assert (
        evaluate_authorization(policy, _request(principal="deputy")).reason
        is DenialReason.NO_APPLICABLE_GRANT
    )
    assert (
        evaluate_authorization(
            policy, _request(principal="worker", capability=Capability.MERGE)
        ).reason
        is DenialReason.NO_APPLICABLE_GRANT
    )

    bad_child = _grant(
        statement_id="bad-child",
        issuer="deputy",
        subject="worker",
        parent_statement_id="deputy-merge",
        capability=Capability.EXECUTE_TASK,
    )
    assert (
        evaluate_authorization(_policy(merge_for_deputy, bad_child), _request()).reason
        is DenialReason.MALFORMED_DELEGATION
    )


def test_expiration_and_not_before_are_distinct_fail_closed_outcomes() -> None:
    future = _grant(not_before_ms=1_200, expires_at_ms=2_000)
    expired = _grant(not_before_ms=100, expires_at_ms=900)

    assert (
        evaluate_authorization(_policy(future), _request()).reason
        is DenialReason.NOT_YET_VALID
    )
    assert (
        evaluate_authorization(_policy(expired), _request()).reason
        is DenialReason.EXPIRED
    )


def test_authorized_revocation_is_transitive_and_unauthorized_one_is_ignored() -> None:
    root = _grant(
        statement_id="root",
        subject="supervisor",
        delegation_depth=1,
    )
    child = _grant(
        statement_id="child",
        issuer="supervisor",
        parent_statement_id="root",
    )
    root_revocation = Revocation(
        "root-revocation", "root", "root", 900, "rotate supervisor"
    )
    unauthorized = Revocation(
        "attacker-revocation", "attacker", "child", 900, "confused deputy attempt"
    )

    assert (
        evaluate_authorization(
            _policy(root, child, revocations=(root_revocation,)), _request()
        ).reason
        is DenialReason.REVOKED
    )
    assert evaluate_authorization(
        _policy(root, child, revocations=(unauthorized,)), _request()
    ).permitted


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"lease_id": None}, DenialReason.LEASE_REQUIRED),
        ({"fencing_epoch": None}, DenialReason.FENCING_EPOCH_REQUIRED),
        ({"lease_id": "old-lease"}, DenialReason.LEASE_SCOPE_MISMATCH),
        ({"fencing_epoch": 6}, DenialReason.STALE_FENCING_EPOCH),
    ],
)
def test_lease_and_fencing_are_required_and_current(
    changes: dict[str, object], reason: DenialReason
) -> None:
    assert evaluate_authorization(_policy(), _request(**changes)).reason is reason


def test_policy_epoch_takeover_rejects_an_otherwise_valid_old_lease_grant() -> None:
    decision = evaluate_authorization(_policy(epoch=8), _request())
    assert decision.verdict is AuthorizationVerdict.DENY
    assert decision.reason is DenialReason.STALE_FENCING_EPOCH


def test_proof_promotion_requires_independent_proof_authority() -> None:
    grant = _grant(
        capability=Capability.PROMOTE_PROOF,
        proof_authorities=("kernel",),
    )
    policy = _policy(grant)

    assert (
        evaluate_authorization(
            policy,
            _request(
                capability=Capability.PROMOTE_PROOF,
                proof_authority=None,
            ),
        ).reason
        is DenialReason.PROOF_AUTHORITY_REQUIRED
    )
    assert (
        evaluate_authorization(
            policy,
            _request(
                capability=Capability.PROMOTE_PROOF,
                proof_authority="solver",
            ),
        ).reason
        is DenialReason.PROOF_AUTHORITY_MISMATCH
    )
    assert evaluate_authorization(
        policy,
        _request(
            capability=Capability.PROMOTE_PROOF,
            proof_authority="kernel",
        ),
    ).permitted


def test_override_is_explicit_and_path_scoped() -> None:
    grant = _grant(
        capability=Capability.OVERRIDE_POLICY,
        override_scopes=("validation/timeout",),
    )
    policy = _policy(grant)

    assert (
        evaluate_authorization(
            policy,
            _request(capability=Capability.OVERRIDE_POLICY, override_scope=None),
        ).reason
        is DenialReason.OVERRIDE_SCOPE_REQUIRED
    )
    assert (
        evaluate_authorization(
            policy,
            _request(
                capability=Capability.OVERRIDE_POLICY,
                override_scope="merge/evidence",
            ),
        ).reason
        is DenialReason.OVERRIDE_SCOPE_MISMATCH
    )
    assert evaluate_authorization(
        policy,
        _request(
            capability=Capability.OVERRIDE_POLICY,
            override_scope="validation/timeout/model-check",
        ),
    ).permitted


def test_authorization_can_permit_action_but_never_code_correctness() -> None:
    decision = evaluate_authorization(_policy(), _request())
    report = check_authorization(_policy(), _request(), adapters=())

    assert decision.permits_action is True
    assert decision.establishes_generated_code_correctness is False
    assert (
        decision.generated_code_correctness is GeneratedCodeCorrectness.NOT_ESTABLISHED
    )
    assert report.permitted is True
    assert report.establishes_generated_code_correctness is False
    assert report.to_dict()["generated_code_correctness"] == "not_established"


def test_datalog_and_secpal_render_every_authority_dimension() -> None:
    proof = _grant(
        capability=Capability.PROMOTE_PROOF,
        proof_authorities=("kernel",),
    )
    policy = _policy(proof)
    request = _request(
        capability=Capability.PROMOTE_PROOF,
        proof_authority="kernel",
    )

    datalog = render_datalog_policy(policy, request)
    secpal = render_secpal_policy(policy, request)

    for token in (
        "trusted",
        "grant",
        "depth",
        "parent",
        "active",
        "revoked",
        "narrows",
        "authorized",
        "request_match",
    ):
        assert token in datalog
    for token in (
        "says",
        "delegation-depth",
        "lease=",
        "worktree=",
        "path=",
        "fence=",
        "proof-authority=",
        "override=",
        "valid=",
        "revokes",
    ):
        assert token in secpal


def test_missing_external_engines_are_unsupported_while_reference_continues() -> None:
    adapters = (
        DatalogAuthorizationAdapter(which=lambda _name: None),
        SecPALAuthorizationAdapter(which=lambda _name: None),
    )
    capabilities = probe_authorization_engines(adapters)
    report = AuthorizationChecker(adapters=adapters).evaluate(_policy(), _request())

    assert all(item.status is EngineSupportStatus.UNSUPPORTED for item in capabilities)
    assert report.permitted
    assert {item.status for item in report.shadow_results} == {LaneStatus.UNSUPPORTED}
    assert report.to_dict()["enforcement_lane"] == "deterministic_reference"
    assert report.to_dict()["external_lanes_are_shadow_only"] is True


class _ReferenceBackedDatalogAdapter(DatalogAuthorizationAdapter):
    """Test double for a lane whose parser implements the rendered semantics."""

    def _execute(
        self,
        executable: str,
        policy: AuthorizationPolicy,
        request: AuthorizationRequest,
    ) -> AuthorizationVerdict | None:
        return ReferenceAuthorizationEvaluator().evaluate(policy, request).verdict


class _AlwaysPermitDatalogAdapter(_ReferenceBackedDatalogAdapter):
    def _execute(
        self,
        executable: str,
        policy: AuthorizationPolicy,
        request: AuthorizationRequest,
    ) -> AuthorizationVerdict | None:
        return AuthorizationVerdict.PERMIT


def _version_runner(request: CommandRequest) -> CommandResult:
    assert request.command[-1] == "--version"
    return CommandResult(returncode=0, stdout="test-engine 1.0\n")


def test_external_lane_requires_complete_fixture_agreement_before_support() -> None:
    adapter = _ReferenceBackedDatalogAdapter(
        which=lambda _name: str(Path("/bin/true")),
        command_runner=_version_runner,
        monotonic=lambda: 1.0,
    )
    capability = adapter.probe()
    report = AuthorizationChecker(adapters=(adapter,)).evaluate(_policy(), _request())

    assert capability.status is EngineSupportStatus.CONFORMANT
    assert capability.conformance_receipt is not None
    assert capability.conformance_receipt.passed
    assert set(capability.conformance_receipt.checked_fixture_ids) == {
        fixture.fixture_id for fixture in AUTHORIZATION_CONFORMANCE_FIXTURES
    }
    assert report.shadow_results[0].status is LaneStatus.AGREED
    assert report.shadow_results[0].observed_verdict is AuthorizationVerdict.PERMIT


def test_fixture_disagreement_keeps_external_lane_unsupported() -> None:
    adapter = _AlwaysPermitDatalogAdapter(
        which=lambda _name: str(Path("/bin/true")),
        command_runner=_version_runner,
        monotonic=lambda: 1.0,
    )
    capability = adapter.probe()

    assert capability.status is EngineSupportStatus.UNSUPPORTED
    assert capability.conformance_receipt is not None
    assert capability.conformance_receipt.status.value == "failed"
    assert set(capability.conformance_receipt.disagreements) == {
        "negative@1",
        "revocation@1",
        "confused-deputy@1",
        "stale-lease@1",
    }
