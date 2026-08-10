"""ASE3-024 signed planning-policy artifact tests."""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.entrypoints.planning_policy import (
    PLANNING_POLICY_SCHEMA,
    PlanningPolicyError,
    revoke_prompt_planning_policy,
    sign_prompt_planning_policy,
    verify_prompt_planning_policy,
)


def test_sign_and_verify_planning_policy_is_independent_of_profile_schema() -> None:
    key = Ed25519PrivateKey.generate()
    policy = sign_prompt_planning_policy(
        private_key=key,
        policy_id="planning-policy-1",
        max_planning_attempts=1,
        allowed_planning_providers=("grok",),
    )
    assert policy.schema == PLANNING_POLICY_SCHEMA
    assert policy.content_id.startswith("sha256:")
    verified = verify_prompt_planning_policy(policy)
    assert verified.content_id == policy.content_id
    assert verified.allow_provider_replay_on_unknown is False


def test_expired_or_revoked_policy_fails_closed() -> None:
    key = Ed25519PrivateKey.generate()
    now = int(time.time() * 1000)
    policy = sign_prompt_planning_policy(
        private_key=key,
        policy_id="planning-policy-exp",
        issued_at_ms=now - 10_000,
        expires_at_ms=now - 1,
    )
    with pytest.raises(PlanningPolicyError, match="expired"):
        verify_prompt_planning_policy(policy, now_ms=now)

    live = sign_prompt_planning_policy(
        private_key=key,
        policy_id="planning-policy-rev",
        issued_at_ms=now - 1_000,
        expires_at_ms=now + 60_000,
    )
    revoked = revoke_prompt_planning_policy(live, private_key=key)
    with pytest.raises(PlanningPolicyError, match="revoked"):
        verify_prompt_planning_policy(revoked, now_ms=now)


def test_tampered_signature_is_rejected() -> None:
    key = Ed25519PrivateKey.generate()
    policy = sign_prompt_planning_policy(
        private_key=key,
        policy_id="planning-policy-tamper",
    )
    payload = policy.to_dict()
    payload["max_prompt_bytes"] = policy.max_prompt_bytes + 1
    with pytest.raises(PlanningPolicyError):
        verify_prompt_planning_policy(payload)
