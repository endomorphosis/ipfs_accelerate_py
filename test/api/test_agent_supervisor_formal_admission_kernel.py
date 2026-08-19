"""FACP-039: restricted Effect Admission Kernel.

Acceptance:
- Valid token requires all declared obligations and exact argument CID.
- Expired / revoked / replayed / changed arguments fail.
- Unknown source policy compiles only to denial / obligation / typed indeterminate.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.admission.formal_kernel import (
    BUNDLE,
    EVIDENCE_SUBSET,
    GOAL_ID,
    KERNEL_ISSUER,
    SCHEMA,
    TASK_ID,
    TOKEN_SCHEMA,
    AdmissionBindings,
    AdmissionError,
    AdmissionErrorCode,
    AdmissionToken,
    AdmissionVerdict,
    EffectAdmissionKernel,
    OperationSpecView,
    PolicyIRVerdict,
    binding_cid,
    compile_source_policy,
    default_kernel,
    derive_token_obligations,
    fresh_nonce,
)


NOW_MS = 1_700_000_000_000
NOT_AFTER = NOW_MS + 60_000


def _spec(**overrides: Any) -> OperationSpecView:
    values: dict[str, Any] = {
        "operation_id": "accelerate.inference",
        "effect_class": "process",
        "authority_obligation": "capability_verified",
        "policy_obligation": "host_policy_required",
        "confirmation_obligation": "one_use_confirmation_required",
        "lease_obligation": "lease_required",
        "observation_obligation": "independent_observation_required",
        "idempotency_class": "at_most_once",
        "reversibility_class": "compensatable",
    }
    values.update(overrides)
    return OperationSpecView(**values)


def _bindings(
    *,
    nonce: str | None = None,
    argument_cid: str | None = None,
    not_after: int = NOT_AFTER,
    **overrides: Any,
) -> AdmissionBindings:
    values: dict[str, Any] = {
        "actor_cid": binding_cid("actor", "operator-1"),
        "device_cid": binding_cid("device", "device-1"),
        "tenant_cid": binding_cid("tenant", "tenant-1"),
        "resource_cid": binding_cid("resource", "model:demo"),
        "operation_id": "accelerate.inference",
        "argument_cid": argument_cid or binding_cid("argument", {"prompt": "hello"}),
        "contract_cid": binding_cid("contract", "facp/operation-spec@1"),
        "delegation_cid": binding_cid("delegation", "ucan:chain-1"),
        "policy_cid": binding_cid(
            "policy",
            {
                "clauses": [
                    {
                        "clause_type": "permission",
                        "actor": "*",
                        "action": "accelerate.inference",
                        "resource": "*",
                    }
                ]
            },
        ),
        "confirmation_cid": binding_cid("confirmation", "confirm:1"),
        "lease_id": "lease:demo-1",
        "not_before": NOW_MS - 1_000,
        "not_after": not_after,
        "nonce": nonce or fresh_nonce(),
        "signature_cid": binding_cid("signature", "sig:1"),
        "revocation_id": "",
    }
    values.update(overrides)
    return AdmissionBindings(**values)


def _allow_policy(*, policy_cid: str | None = None) -> dict[str, Any]:
    body = {
        "name": "allow-inference",
        "clauses": [
            {
                "clause_type": "permission",
                "actor": "*",
                "action": "accelerate.inference",
                "resource": "*",
            }
        ],
    }
    if policy_cid is not None:
        body["policy_cid"] = policy_cid
    return body


def _mint(
    kernel: EffectAdmissionKernel | None = None,
    *,
    spec: OperationSpecView | None = None,
    bindings: AdmissionBindings | None = None,
    source_policy: dict[str, Any] | None = None,
) -> tuple[EffectAdmissionKernel, AdmissionToken, AdmissionBindings, OperationSpecView]:
    kernel = kernel or default_kernel(now_ms=NOW_MS)
    spec = spec or _spec()
    bindings = bindings or _bindings()
    policy = source_policy
    if policy is None and "policy_bound" in derive_token_obligations(spec):
        policy = _allow_policy(policy_cid=bindings.policy_cid)
    token = kernel.mint_token(spec, bindings, source_policy=policy)
    return kernel, token, bindings, spec


# ---------------------------------------------------------------------------
# Metadata / evidence envelope
# ---------------------------------------------------------------------------


def test_facp_039_metadata_and_evidence_subset() -> None:
    assert TASK_ID == "FACP-039"
    assert GOAL_ID == "FACP-G320"
    assert BUNDLE == "facp/admission/kernel"
    assert SCHEMA == "facp/admission-kernel@1"
    assert TOKEN_SCHEMA == "facp/admission-token@1"
    assert KERNEL_ISSUER == "effect_admission_kernel"
    required = {
        "actor",
        "device",
        "tenant",
        "resource",
        "operation",
        "argument",
        "contract",
        "delegation",
        "policy",
        "confirmation",
        "lease",
        "expiry",
        "nonce",
        "signature",
        "revocation",
    }
    assert set(EVIDENCE_SUBSET) == required


def test_derive_token_obligations_matches_facp_038() -> None:
    obligations = derive_token_obligations(_spec())
    assert "kernel_issued" in obligations
    assert "argument_bound" in obligations
    assert "actor_bound" in obligations
    assert "capability_bound" in obligations
    assert "delegation_bound" in obligations
    assert "policy_bound" in obligations
    assert "confirmation_bound" in obligations
    assert "lease_bound" in obligations
    assert "observation_bound" in obligations
    assert derive_token_obligations(_spec(effect_class="pure")) == frozenset()


# ---------------------------------------------------------------------------
# Happy path: all obligations + exact argument CID
# ---------------------------------------------------------------------------


def test_valid_token_requires_all_obligations_and_exact_argument_cid() -> None:
    kernel, token, bindings, spec = _mint()
    required = derive_token_obligations(spec)
    assert required.issubset(set(token.satisfied_obligations))
    assert token.issuer == KERNEL_ISSUER
    assert token.argument_cid == bindings.argument_cid

    decision = kernel.unlock_handler(
        spec=spec,
        typestate="Reserved",
        token=token,
        argument_cid=bindings.argument_cid,
    )
    assert decision.verdict is AdmissionVerdict.ADMIT
    assert decision.unlocked is True
    assert decision.code is None

    # Evidence subset dimensions are retained on the token bindings.
    present = set(AdmissionBindings(**dict(token.bindings)).evidence_present())
    assert {
        "actor",
        "device",
        "tenant",
        "resource",
        "operation",
        "argument",
        "contract",
        "delegation",
        "policy",
        "confirmation",
        "lease",
        "expiry",
        "nonce",
        "signature",
    }.issubset(present)


def test_canonical_token_projection_closed_fields() -> None:
    _, token, _, _ = _mint()
    projected = token.to_canonical_token()
    assert set(projected) == {
        "schema",
        "schema_version",
        "operation_id",
        "actor_cid",
        "argument_cid",
        "nonce",
        "not_after",
    }
    assert projected["schema"] == TOKEN_SCHEMA


# ---------------------------------------------------------------------------
# Negative: expired / revoked / replayed / changed arguments
# ---------------------------------------------------------------------------


def test_expired_token_fails() -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    bindings = _bindings(not_after=NOW_MS + 5_000)
    kernel, token, bindings, spec = _mint(kernel, bindings=bindings)
    decision = kernel.verify_token(
        token,
        operation_id=spec.operation_id,
        argument_cid=bindings.argument_cid,
        now_ms=NOW_MS + 10_000,
    )
    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.code is AdmissionErrorCode.EXPIRED_TOKEN


def test_revoked_token_fails() -> None:
    kernel, token, bindings, spec = _mint()
    kernel.revoke(token)
    decision = kernel.verify_token(
        token,
        operation_id=spec.operation_id,
        argument_cid=bindings.argument_cid,
    )
    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.code is AdmissionErrorCode.REVOKED_TOKEN


def test_replayed_token_fails() -> None:
    kernel, token, bindings, spec = _mint()
    first = kernel.unlock_handler(
        spec=spec,
        typestate="Started",
        token=token,
        argument_cid=bindings.argument_cid,
        consume=True,
    )
    assert first.unlocked is True
    second = kernel.unlock_handler(
        spec=spec,
        typestate="Started",
        token=token,
        argument_cid=bindings.argument_cid,
        consume=True,
    )
    assert second.verdict is AdmissionVerdict.DENY
    assert second.code is AdmissionErrorCode.REPLAYED_TOKEN


def test_changed_argument_cid_fails() -> None:
    kernel, token, bindings, spec = _mint()
    decision = kernel.verify_token(
        token,
        operation_id=spec.operation_id,
        argument_cid=binding_cid("argument", {"prompt": "TAMPERED"}),
    )
    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.code is AdmissionErrorCode.ARGUMENT_MISMATCH


def test_missing_obligation_fails_mint() -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    spec = _spec()
    bindings = _bindings()
    with pytest.raises(AdmissionError) as exc:
        kernel.mint_token(
            spec,
            bindings,
            satisfied_obligations=["kernel_issued", "argument_bound"],
            source_policy=_allow_policy(policy_cid=bindings.policy_cid),
        )
    assert exc.value.code is AdmissionErrorCode.TOKEN_OBLIGATION_MISMATCH


# ---------------------------------------------------------------------------
# Non-kernel / browser / model issuers cannot mint or unlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "issuer",
    ["browser", "browser_consent", "prompt", "model", "peer", "payment", "ui"],
)
def test_forbidden_issuers_cannot_mint(issuer: str) -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    bindings = _bindings()
    with pytest.raises(AdmissionError) as exc:
        kernel.mint_token(
            _spec(),
            bindings,
            issuer=issuer,
            source_policy=_allow_policy(policy_cid=bindings.policy_cid),
        )
    assert exc.value.code in {
        AdmissionErrorCode.FORBIDDEN_ISSUER,
        AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER,
    }


def test_non_kernel_token_fails_verify() -> None:
    kernel, token, bindings, spec = _mint()
    forged = AdmissionToken(
        operation_id=token.operation_id,
        effect_class=token.effect_class,
        argument_cid=token.argument_cid,
        actor_cid=token.actor_cid,
        nonce=fresh_nonce(),
        not_after=token.not_after,
        satisfied_obligations=token.satisfied_obligations,
        issuer="browser_consent",
        derived_obligations=token.derived_obligations,
        bindings=dict(token.bindings),
    )
    decision = kernel.verify_token(
        forged,
        operation_id=spec.operation_id,
        argument_cid=bindings.argument_cid,
    )
    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.code is AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER


def test_pure_effect_forbids_token_mint_and_unlock() -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    pure = _spec(effect_class="pure", authority_obligation="none", policy_obligation="none")
    with pytest.raises(AdmissionError) as exc:
        kernel.mint_token(pure, _bindings())
    assert exc.value.code is AdmissionErrorCode.PURE_TOKEN_FORBIDDEN

    decision = kernel.unlock_handler(
        spec=pure,
        typestate="Reserved",
        token=None,
        argument_cid=binding_cid("argument", {}),
    )
    assert decision.unlocked is False
    assert decision.code is AdmissionErrorCode.HANDLER_NOT_UNLOCKED


# ---------------------------------------------------------------------------
# Source policy compilation: unknown -> deny / obligation / indeterminate
# ---------------------------------------------------------------------------


def test_unknown_source_policy_never_compiles_to_allow() -> None:
    unknown = {
        "name": "mystery",
        "clauses": [
            {"clause_type": "permission", "actor": "*", "action": "*", "resource": "*"},
            {"clause_type": "temporal_until_unless", "magic": True},
        ],
        "consent": "user-said-yes",
    }
    ir = compile_source_policy(
        unknown,
        actor="operator-1",
        action="accelerate.inference",
        resource="model:demo",
        now_ms=NOW_MS,
    )
    assert ir.verdict in {
        PolicyIRVerdict.DENY,
        PolicyIRVerdict.OBLIGATION,
        PolicyIRVerdict.INDETERMINATE,
    }
    assert ir.verdict is not PolicyIRVerdict.ALLOW
    assert ir.fully_translated is False
    assert ir.unknown_constructs


def test_unknown_fields_only_yield_deny_or_indeterminate() -> None:
    ir = compile_source_policy(
        {
            "name": "partial",
            "allow": True,
            "clauses": [],
        },
        now_ms=NOW_MS,
    )
    assert ir.verdict in {
        PolicyIRVerdict.DENY,
        PolicyIRVerdict.OBLIGATION,
        PolicyIRVerdict.INDETERMINATE,
    }
    assert ir.verdict is not PolicyIRVerdict.ALLOW


def test_known_permission_may_allow_when_fully_translated() -> None:
    ir = compile_source_policy(
        _allow_policy(),
        actor="x",
        action="accelerate.inference",
        resource="y",
        now_ms=NOW_MS,
    )
    assert ir.fully_translated is True
    assert ir.verdict is PolicyIRVerdict.ALLOW


def test_prohibition_denies() -> None:
    ir = compile_source_policy(
        {
            "clauses": [
                {
                    "clause_type": "permission",
                    "actor": "*",
                    "action": "*",
                    "resource": "*",
                },
                {
                    "clause_type": "prohibition",
                    "actor": "*",
                    "action": "accelerate.inference",
                    "resource": "*",
                },
            ]
        },
        action="accelerate.inference",
        now_ms=NOW_MS,
    )
    assert ir.verdict is PolicyIRVerdict.DENY


def test_mint_rejects_indeterminate_policy() -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    bindings = _bindings()
    with pytest.raises(AdmissionError) as exc:
        kernel.mint_token(
            _spec(),
            bindings,
            source_policy={
                "policy_cid": bindings.policy_cid,
                "clauses": [
                    {
                        "clause_type": "permission",
                        "actor": "*",
                        "action": "accelerate.inference",
                        "resource": "*",
                    },
                    {"clause_type": "exotic_modality", "payload": "???"},
                ],
            },
        )
    assert exc.value.code in {
        AdmissionErrorCode.POLICY_INDETERMINATE,
        AdmissionErrorCode.POLICY_DENIED,
    }


def test_free_form_authority_on_operation_spec_fails() -> None:
    with pytest.raises(AdmissionError) as exc:
        OperationSpecView.from_mapping(
            {
                "operation_id": "accelerate.inference",
                "effect_class": "process",
                "consent": "yes",
            }
        )
    assert exc.value.code is AdmissionErrorCode.FREE_FORM_AUTHORITY


def test_handler_not_unlocked_outside_reserved_or_started() -> None:
    kernel, token, bindings, spec = _mint()
    decision = kernel.unlock_handler(
        spec=spec,
        typestate="PolicyEvaluated",
        token=token,
        argument_cid=bindings.argument_cid,
        consume=False,
    )
    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.code is AdmissionErrorCode.HANDLER_NOT_UNLOCKED


def test_missing_evidence_binding_fails_mint() -> None:
    kernel = default_kernel(now_ms=NOW_MS)
    bindings = _bindings(delegation_cid="")
    with pytest.raises(AdmissionError) as exc:
        kernel.mint_token(
            _spec(),
            bindings,
            source_policy=_allow_policy(policy_cid=bindings.policy_cid),
        )
    assert exc.value.code is AdmissionErrorCode.MISSING_EVIDENCE
