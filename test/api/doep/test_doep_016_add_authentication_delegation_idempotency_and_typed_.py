"""Independent current-tree checks for DOEP-016 authority admission security."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    CANONICAL_AUTHORITY_ADMISSION_ENTRYPOINT,
    CANONICAL_AUTHORITY_ADMISSION_INTERFACE,
    LOCAL_WORKTREE_DENIED_EFFECTS,
    AuthenticatedPrincipalEvidence,
    AuthorityAdmissionContractError,
    AuthorityAdmissionError,
    AuthorityAdmissionRequest,
    AuthorityAdmissionUnavailableError,
    AuthorityAuthenticationError,
    AuthorityDelegationError,
    AuthorityIdempotencyError,
    AuthorityIdempotencyStore,
    AuthorityResolutionRequest,
    AuthorityResolver,
    DelegationEvidence,
    PrincipalSourceKind,
    SignedProfileEvidence,
    admit_authority,
    bind_delegation,
    install_local_worktree_authority,
    policy_cid_for,
    resolve_authority,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ExpectedEffect,
    InvocationMode,
    ResolutionSource,
    WorktreeStrategy,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "authority_resolver.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-016.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-016.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py",
    "test/api/doep/test_doep_016_add_authentication_delegation_idempotency_and_typed_.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-016.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-016.json",
)
TASK_CID = "sha256:0d12d5694ddf55e6a0c86085aa7a2989eb89ed17b628011b82629bb6c027faec"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _local_authority(principal: str = "did:key:local-owner"):
    return install_local_worktree_authority(
        principal,
        signing_key_handle="key:local-worktree-1",
        installed_at_ms=1_700_000_000_000,
    )


def _signed_profile(
    principal: str = "did:key:mcp-caller",
    *,
    effects: tuple[ExpectedEffect, ...] | None = None,
) -> SignedProfileEvidence:
    return SignedProfileEvidence(
        profile_name="local-worktree",
        profile_cid=_cid(f"signed-profile-{principal}"),
        policy_cid=policy_cid_for("policy:signed-local-worktree@1"),
        principal_ref=principal,
        authority_source_ref="authority:signed-profile",
        allowed_effects=effects or (
            ExpectedEffect.INSPECT_REPOSITORY,
            ExpectedEffect.WRITE_SUPERVISOR_STATE,
            ExpectedEffect.CREATE_ISOLATED_WORKTREE,
            ExpectedEffect.EDIT_ISOLATED_WORKTREE,
            ExpectedEffect.RUN_VALIDATION,
            ExpectedEffect.LAUNCH_LOCAL_PROCESS,
        ),
        evidence_cid=_cid(f"signed-profile-evidence-{principal}"),
        signature_verified=True,
        worktree_strategy=WorktreeStrategy.ISOLATED,
    )


def _mcp_plus_principal(
    principal: str = "did:key:mcp-caller",
    *,
    ucan_verified: bool = False,
) -> AuthenticatedPrincipalEvidence:
    return AuthenticatedPrincipalEvidence(
        principal_ref=principal,
        source=ResolutionSource.AUTHENTICATED_TRANSPORT,
        evidence_cid=_cid(f"transport-{principal}"),
        kind=PrincipalSourceKind.MCP_PLUS_UCAN,
        transport="mcp++",
        signature_verified=True,
        ucan_verified=ucan_verified,
    )


def _valid_delegation(
    audience: str = "did:key:mcp-caller",
    *,
    attenuated_effects: tuple[ExpectedEffect, ...] | None = None,
) -> DelegationEvidence:
    return DelegationEvidence(
        raw_chain=(
            {
                "issuer": "did:user:alice",
                "audience": audience,
                "capabilities": [
                    {"resource": "*", "ability": "agent-supervisor/invoke"}
                ],
            },
        ),
        actor=audience,
        attenuated_effects=attenuated_effects,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_admission_extends_existing_authority_resolver() -> None:
    assert CANONICAL_AUTHORITY_ADMISSION_INTERFACE == "AuthorityAdmission@1"
    assert CANONICAL_AUTHORITY_ADMISSION_ENTRYPOINT == "admit_authority"
    assert (
        admit_authority.__module__
        == "ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver"
    )
    assert hasattr(AuthorityResolver, "resolve")
    assert hasattr(AuthorityResolver, "admit")
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert "def resolve_authority" in source
    assert "def admit_authority" in source
    assert "validate_raw_delegation_chain" in source
    assert "class AuthorityIdempotencyStore" in source
    assert "class CanonicalAuthorityService" not in source
    assert "class AuthService" not in source
    assert "def validate_raw_delegation_chain(" not in source


def test_typed_errors_are_distinct_and_fail_closed() -> None:
    assert issubclass(AuthorityAuthenticationError, AuthorityAdmissionError)
    assert issubclass(AuthorityDelegationError, AuthorityAdmissionError)
    assert issubclass(AuthorityIdempotencyError, AuthorityAdmissionError)
    assert issubclass(AuthorityAdmissionContractError, AuthorityAdmissionError)
    assert issubclass(AuthorityAdmissionUnavailableError, AuthorityAdmissionError)

    soft = resolve_authority(
        AuthorityResolutionRequest(
            credentials_present=True,
            prompt_claimed_principal="did:key:forged",
            username_claim="root",
        )
    )
    assert soft.authorized is False
    assert "no_trusted_principal_evidence" in soft.reason_codes

    with pytest.raises(AuthorityAuthenticationError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    credentials_present=True,
                    prompt_claimed_principal="did:key:forged",
                ),
                require_authenticated_principal=True,
            )
        )

    with pytest.raises(AuthorityIdempotencyError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    mode=InvocationMode.WORKTREE,
                    local_worktree_authority=_local_authority(),
                ),
                require_idempotency_key=True,
                idempotency_key="",
            )
        )

    with pytest.raises(AuthorityDelegationError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    mode=InvocationMode.WORKTREE,
                    authenticated_principal=_mcp_plus_principal(ucan_verified=True),
                    signed_profile=_signed_profile(),
                )
            )
        )


def test_delegation_reuses_mcpplusplus_validator_and_only_narrows() -> None:
    principal = _mcp_plus_principal(ucan_verified=False)
    profile = _signed_profile()
    narrowing = (
        ExpectedEffect.INSPECT_REPOSITORY,
        ExpectedEffect.RUN_VALIDATION,
    )
    admission = admit_authority(
        AuthorityAdmissionRequest(
            resolution=AuthorityResolutionRequest(
                mode=InvocationMode.WORKTREE,
                authenticated_principal=principal,
                signed_profile=profile,
            ),
            delegation=_valid_delegation(attenuated_effects=narrowing),
        )
    )
    assert admission.authorized is True
    assert admission.delegation is not None
    assert admission.delegation.bound is True
    assert admission.delegation.principal_ref == "did:key:mcp-caller"
    assert "mcpplusplus_validate_raw_delegation_chain" in admission.delegation.reason_codes
    for effect in narrowing:
        assert admission.resolution.effect_ceiling.permits(effect)
    assert not admission.resolution.effect_ceiling.permits(
        ExpectedEffect.EDIT_ISOLATED_WORKTREE
    )
    for effect in LOCAL_WORKTREE_DENIED_EFFECTS:
        assert not admission.resolution.effect_ceiling.permits(effect)

    denied = bind_delegation(
        DelegationEvidence(
            raw_chain=(
                {
                    "issuer": "did:user:alice",
                    "audience": "did:key:mcp-caller",
                    "capabilities": [
                        {"resource": "smoke.echo", "ability": "invoke"}
                    ],
                },
                {
                    "issuer": "did:key:mcp-caller",
                    "audience": "did:key:mcp-caller",
                    "capabilities": [{"resource": "*", "ability": "invoke"}],
                },
            ),
            resource="smoke.echo",
            ability="invoke",
            actor="did:key:mcp-caller",
        )
    )
    assert denied.bound is False
    assert any("escalation" in code for code in denied.reason_codes)

    with pytest.raises(AuthorityDelegationError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    mode=InvocationMode.WORKTREE,
                    authenticated_principal=principal,
                    signed_profile=profile,
                ),
                delegation=DelegationEvidence(
                    raw_chain=(
                        {
                            "issuer": "did:user:alice",
                            "audience": "did:key:mcp-caller",
                            "capabilities": [
                                {"resource": "smoke.echo", "ability": "invoke"}
                            ],
                        },
                        {
                            "issuer": "did:key:mcp-caller",
                            "audience": "did:key:mcp-caller",
                            "capabilities": [
                                {"resource": "*", "ability": "invoke"}
                            ],
                        },
                    ),
                    resource="smoke.echo",
                    ability="invoke",
                    actor="did:key:mcp-caller",
                ),
            )
        )


def test_idempotency_replays_exact_digest_and_conflicts_on_divergence() -> None:
    store = AuthorityIdempotencyStore()
    resolver = AuthorityResolver(idempotency_store=store)
    local = _local_authority()
    first = resolver.admit(
        AuthorityAdmissionRequest(
            resolution=AuthorityResolutionRequest(
                mode=InvocationMode.WORKTREE,
                local_worktree_authority=local,
            ),
            idempotency_key="doep-016-admission-1",
        )
    )
    assert first.authorized is True
    assert first.idempotency is not None
    assert first.idempotency.replayed is False

    second = resolver.admit(
        AuthorityAdmissionRequest(
            resolution=AuthorityResolutionRequest(
                mode=InvocationMode.WORKTREE,
                local_worktree_authority=local,
            ),
            idempotency_key="doep-016-admission-1",
        )
    )
    assert second.idempotency is not None
    assert second.idempotency.replayed is True
    assert (
        second.resolution.decision_reference_cid
        == first.resolution.decision_reference_cid
    )

    with pytest.raises(AuthorityIdempotencyError):
        resolver.admit(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    mode=InvocationMode.WORKTREE,
                    local_worktree_authority=_local_authority("did:key:other-owner"),
                ),
                idempotency_key="doep-016-admission-1",
            )
        )


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-016"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True

    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    extension = manifest["canonical_extension"]
    assert extension["identifier"] == CANONICAL_AUTHORITY_ADMISSION_INTERFACE
    assert extension["entrypoint"] == CANONICAL_AUTHORITY_ADMISSION_ENTRYPOINT
    assert extension["carrier"] == "AuthorityResolver"
    assert extension["resolution_delegate"] == "resolve_authority"
    assert extension["idempotency_semantics"] == "exact_request_digest_replay"
    assert extension["ase008_preserved"] is True
    assert extension["callers_supply_authoritative_policy"] is False
    assert extension["completion_authority"] is False
    assert extension["competing_subsystem_created"] is False
    assert "AuthorityAuthenticationError" in extension["typed_errors"]
    assert "AuthorityDelegationError" in extension["typed_errors"]
    assert "AuthorityIdempotencyError" in extension["typed_errors"]
    assert (
        extension["delegation_delegate"]
        == "ipfs_accelerate_py.mcp_server.mcplusplus.delegation.validate_raw_delegation_chain"
    )

    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SERVICE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["required_evidence"]["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
