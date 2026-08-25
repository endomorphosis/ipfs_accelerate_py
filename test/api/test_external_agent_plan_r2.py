from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.planning import external_agent_plan_r2 as r2

NOW_MS = 1_800_000_000_000


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _task(token: str, alias: str, ordinal: int, status: str) -> dict[str, object]:
    task_cid = _sha(token)
    return {
        "task_cid": task_cid,
        "task_alias": alias,
        "goal_cid": _sha("9"),
        "plan_cid": _sha("8"),
        "objective_id": "EAAEF-OBJ-001",
        "ordinal": ordinal,
        "status": status,
        "revision": 1,
        "priority": "high",
        "identity": {"task_cid": task_cid},
        "body": {
            "read_scope": [f"read/{alias}"],
            "write_scope": [f"write/{alias}"],
            "effect_scope": [f"effect/{alias}"],
        },
    }


def _statement() -> dict[str, object]:
    completed = _task("a", "EAAEF-000", 1, "accepted")
    frontier = _task("b", "EAAEF-001", 2, "todo")
    protected = {
        "task_cid": completed["task_cid"],
        "status": completed["status"],
        "revision": completed["revision"],
        "task_row": completed,
        "task_row_cid": r2._cid(completed),
    }
    return r2.prepare_plan_r2_transition_authorization(
        board_namespace="external-agent-autonomous-execution-fabric-v1",
        source_head="1" * 40,
        source_tree="2" * 40,
        source_generation_cid=_sha("3"),
        bootstrap_admission_cid=_sha("4"),
        r1_launch_capsule_cid=_sha("5"),
        quack_owner_qualification_cid=_sha("6"),
        quack_command_fabric_qualification_cid=_sha("7"),
        owner_principal_did="did:key:zOwner",
        shard_id="eaaef-control",
        store_id="eaaef-control-run-v5",
        owner_generation=3,
        expected_epoch=4,
        fencing_token=5,
        lease_id="eaaef-plan-r2-lease",
        expected_version=9,
        expected_active_plan_cid=_sha("c"),
        expected_active_plan_root_cid=_sha("d"),
        expected_active_plan_revision=1,
        expected_event_cursor="event-cursor-9",
        expected_semantic_root_cid=_sha("e"),
        new_plan={
            "plan_cid": _sha("8"),
            "plan_alias": "EAAEF-PLAN-R2",
            "plan_root_cid": _sha("f"),
            "semantic_root_cid": _sha("0"),
            "status": "active",
            "revision": 2,
            "body": {"objective": "continue after reconciliation"},
        },
        tasks=[completed, frontier],
        dependencies=[
            {
                "task_cid": frontier["task_cid"],
                "dependency_task_cid": completed["task_cid"],
                "kind": "requires",
            }
        ],
        protected_tasks=[protected],
        frontier_task_cids=[str(frontier["task_cid"])],
        delta_cid=_sha("1"),
        request_id="eaaef-plan-r2-request-1",
        idempotency_key="eaaef-plan-r2-idempotency-1",
        deadline_ms=NOW_MS + 50_000,
        issued_at_ms=NOW_MS - 1000,
        expires_at_ms=NOW_MS + 100_000,
        one_use_nonce="eaaef-plan-r2-nonce-1",
    )


def _signed_approval(
    statement: dict[str, object], role: str, key: Ed25519PrivateKey
) -> tuple[dict[str, object], str]:
    identity = ed25519_did_key(key.public_key())
    approval = r2.prepare_plan_r2_transition_approval(
        statement,
        role=role,
        identity_did=identity,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    signature = base64.b64encode(
        key.sign(r2._canonical_bytes(approval))
    ).decode("ascii")
    return (
        r2.seal_plan_r2_transition_approval(
            statement,
            approval,
            signature=signature,
        ),
        identity,
    )


def _authorization():
    statement = _statement()
    operator, operator_did = _signed_approval(
        statement, "independent_operator", Ed25519PrivateKey.generate()
    )
    security, security_did = _signed_approval(
        statement,
        "independent_security_reviewer",
        Ed25519PrivateKey.generate(),
    )
    authorization = r2.assemble_plan_r2_transition_authorization(
        statement,
        operator_approval=operator,
        security_approval=security,
        trusted_operator_dids=[operator_did],
        trusted_security_reviewer_dids=[security_did],
        now_ms=NOW_MS,
    )
    return authorization, operator_did, security_did


def _capability(authorization: dict[str, object]):
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    value = r2.plan_r2_operational_capability_signing_payload(
        authorization,
        reviewer_identity_did=reviewer,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    signature = base64.b64encode(
        key.sign(r2._canonical_bytes(value))
    ).decode("ascii")
    return (
        r2.seal_plan_r2_operational_capability(
            value,
            reviewer_signature=signature,
        ),
        reviewer,
    )


class _Repository:
    INTERFACE = r2.AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE

    def __init__(self, authorization: dict[str, object], capability: dict[str, object]):
        self.authorization = authorization
        self.capability = capability
        self.capability_cid = capability.get("capability_cid", "")
        self.command_fabric_qualification_cid = authorization[
            "quack_command_fabric_qualification_cid"
        ]
        self.owner_principal_did = authorization["owner_principal_did"]
        self.shard_id = authorization["shard_id"]
        self.store_id = authorization["store_id"]
        self.owner_generation = authorization["owner_generation"]
        self.owner_epoch = authorization["expected_epoch"]
        self.fence_epoch = authorization["fencing_token"]
        self.prepare_calls = 0
        self.apply_calls = 0
        self.observe_calls = 0
        self.prepared: dict[str, object] | None = None
        self.receipt: dict[str, object] | None = None
        self.observation: dict[str, object] | None = None

    def prepare_authorized_plan_r2_transition(self, authorization):
        self.prepare_calls += 1
        assert authorization == self.authorization
        value: dict[str, object] = {
            "schema": r2.PLAN_R2_PREPARED_PROJECTION_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "capability_cid": self.capability["capability_cid"],
            "authorized_prepare_command_cid": _sha("4"),
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "before_plan_cid": authorization["expected_active_plan_cid"],
            "before_plan_root_cid": authorization["expected_active_plan_root_cid"],
            "before_plan_revision": authorization["expected_active_plan_revision"],
            "before_version": authorization["expected_version"],
            "before_event_cursor": authorization["expected_event_cursor"],
            "before_semantic_root_cid": authorization["expected_semantic_root_cid"],
            "population_cid": authorization["population_cid"],
            "plan_root_cid": authorization["plan_root_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "frontier_cid": authorization["frontier_cid"],
            "prepared_at_ms": NOW_MS,
            "expires_at_ms": NOW_MS + 40_000,
            "authority_mutated": False,
            "process_started": False,
        }
        value["projection_cid"] = r2._cid(value)
        self.prepared = value
        return deepcopy(value)

    def apply_authorized_plan_r2_transition(self, authorization, prepared_projection):
        self.apply_calls += 1
        assert authorization == self.authorization
        assert prepared_projection == self.prepared
        if self.receipt is None:
            value: dict[str, object] = {
                "schema": r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
                "authorization_cid": authorization["authorization_cid"],
                "statement_cid": authorization["statement_cid"],
                "capability_cid": self.capability["capability_cid"],
                "authorized_prepare_command_cid": prepared_projection[
                    "authorized_prepare_command_cid"
                ],
                "authorized_apply_command_cid": _sha("5"),
                "prepared_projection_cid": prepared_projection["projection_cid"],
                "source_head": authorization["source_head"],
                "source_tree": authorization["source_tree"],
                "shard_id": authorization["shard_id"],
                "owner_generation": authorization["owner_generation"],
                "epoch": authorization["expected_epoch"],
                "fence": authorization["fencing_token"],
                "before_plan_cid": authorization["expected_active_plan_cid"],
                "after_plan_cid": authorization["new_plan"]["plan_cid"],
                "before_plan_root_cid": authorization[
                    "expected_active_plan_root_cid"
                ],
                "after_plan_root_cid": authorization["plan_root_cid"],
                "before_plan_revision": authorization[
                    "expected_active_plan_revision"
                ],
                "after_plan_revision": authorization["new_plan"]["revision"],
                "before_version": authorization["expected_version"],
                "after_version": int(authorization["expected_version"]) + 1,
                "before_event_cursor": authorization["expected_event_cursor"],
                "after_event_cursor": "event-cursor-10",
                "before_semantic_root_cid": authorization[
                    "expected_semantic_root_cid"
                ],
                "after_semantic_root_cid": authorization["new_plan"][
                    "semantic_root_cid"
                ],
                "population_cid": authorization["population_cid"],
                "task_population_cid": authorization["task_population_cid"],
                "dependency_population_cid": authorization[
                    "dependency_population_cid"
                ],
                "protected_tasks_root_cid": authorization[
                    "protected_tasks_root_cid"
                ],
                "frontier_cid": authorization["frontier_cid"],
                "frontier_task_cids": authorization["frontier_task_cids"],
                "protected_tasks_unchanged": True,
                "transaction_cid": _sha("2"),
                "replayed": False,
                "committed_at_ms": NOW_MS,
            }
            value["receipt_cid"] = r2._cid(value)
            self.receipt = value
        return deepcopy(self.receipt)

    def observe_authorized_plan_r2_transition(self, authorization, transition_receipt):
        self.observe_calls += 1
        assert authorization == self.authorization
        assert transition_receipt == self.receipt
        if self.observation is None:
            value: dict[str, object] = {
                "schema": r2.PLAN_R2_STATE_OBSERVATION_SCHEMA,
                "authorization_cid": authorization["authorization_cid"],
                "transition_receipt_cid": transition_receipt["receipt_cid"],
                "transaction_cid": transition_receipt["transaction_cid"],
                "authorized_prepare_command_cid": transition_receipt[
                    "authorized_prepare_command_cid"
                ],
                "authorized_apply_command_cid": transition_receipt[
                    "authorized_apply_command_cid"
                ],
                "quack_command_fabric_qualification_cid": authorization[
                    "quack_command_fabric_qualification_cid"
                ],
                "source_head": authorization["source_head"],
                "source_tree": authorization["source_tree"],
                "owner_principal_did": authorization["owner_principal_did"],
                "shard_id": authorization["shard_id"],
                "owner_generation": authorization["owner_generation"],
                "epoch": authorization["expected_epoch"],
                "fence": authorization["fencing_token"],
                "store_version": transition_receipt["after_version"],
                "active_plan_cid": transition_receipt["after_plan_cid"],
                "active_plan_root_cid": transition_receipt[
                    "after_plan_root_cid"
                ],
                "active_plan_revision": transition_receipt[
                    "after_plan_revision"
                ],
                "event_cursor": transition_receipt["after_event_cursor"],
                "semantic_root_cid": transition_receipt[
                    "after_semantic_root_cid"
                ],
                "population_cid": transition_receipt["population_cid"],
                "task_population_cid": transition_receipt[
                    "task_population_cid"
                ],
                "dependency_population_cid": transition_receipt[
                    "dependency_population_cid"
                ],
                "protected_tasks_root_cid": transition_receipt[
                    "protected_tasks_root_cid"
                ],
                "frontier_cid": transition_receipt["frontier_cid"],
                "frontier_task_cids": transition_receipt["frontier_task_cids"],
                "captured_at_ms": NOW_MS,
                "authority_mutated": False,
                "process_started": False,
            }
            value["observation_cid"] = r2._cid(value)
            self.observation = value
        return deepcopy(self.observation)


def test_keyless_transition_approval_sealer_is_exact_and_adversarial() -> None:
    statement = _statement()
    key = Ed25519PrivateKey.generate()
    identity = ed25519_did_key(key.public_key())
    prepared = r2.prepare_plan_r2_transition_approval(
        statement,
        role="independent_operator",
        identity_did=identity,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    assert "signature" not in prepared
    signature = base64.b64encode(
        key.sign(r2._canonical_bytes(prepared))
    ).decode("ascii")
    sealed = r2.seal_plan_r2_transition_approval(
        statement,
        prepared,
        signature=signature,
    )
    assert sealed == {**prepared, "signature": signature}

    mutated = {**prepared, "one_use_nonce": "another-nonce"}
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="differs"):
        r2.seal_plan_r2_transition_approval(
            statement,
            mutated,
            signature=signature,
        )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="signature is absent"):
        r2.seal_plan_r2_transition_approval(
            statement,
            prepared,
            signature="",
        )


def test_operational_capability_payload_and_sealer_are_keyless_and_self_addressed() -> None:
    statement = _statement()
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    arguments = {
        "reviewer_identity_did": reviewer,
        "issued_at_ms": NOW_MS - 500,
        "expires_at_ms": NOW_MS + 50_000,
    }
    prepared = r2.plan_r2_operational_capability_signing_payload(
        statement,
        **arguments,
    )
    assert prepared == r2.plan_r2_operational_capability_signing_payload(
        statement,
        **arguments,
    )
    assert prepared["source_head"] == statement["source_head"]
    assert prepared["owner_principal_did"] == statement["owner_principal_did"]
    assert "reviewer_signature" not in prepared
    assert "capability_cid" not in prepared
    signature = base64.b64encode(
        key.sign(r2._canonical_bytes(prepared))
    ).decode("ascii")
    capability = r2.seal_plan_r2_operational_capability(
        prepared,
        reviewer_signature=signature,
    )
    assert capability["capability_cid"] == r2._cid(
        {
            key: value
            for key, value in capability.items()
            if key != "capability_cid"
        }
    )
    assert r2.verify_plan_r2_operational_capability(
        capability,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    ) == capability

    mutated = {**prepared, "atomic_plan_population_cas": False}
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="payload is invalid"):
        r2.seal_plan_r2_operational_capability(
            mutated,
            reviewer_signature=signature,
        )
    stale_authorization = dict(_authorization()[0])
    stale_authorization["authorization_cid"] = _sha("0")
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="self-addressed"):
        r2.plan_r2_operational_capability_signing_payload(
            stale_authorization,
            **arguments,
        )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="reviewer/lifetime"):
        r2.plan_r2_operational_capability_signing_payload(
            statement,
            reviewer_identity_did=str(statement["owner_principal_did"]),
            issued_at_ms=NOW_MS - 500,
            expires_at_ms=NOW_MS + 50_000,
        )


def test_authorization_is_separate_from_process_birth_and_binds_full_rows() -> None:
    authorization, operator, security = _authorization()
    report = r2.verify_plan_r2_transition_authorization(
        authorization,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    assert report["valid"] is True
    assert authorization["authority"]["process_birth_allowed"] is False
    assert authorization["protected_tasks"][0]["task_row_cid"] == r2._cid(
        authorization["protected_tasks"][0]["task_row"]
    )


def test_missing_or_diagnostic_capability_never_calls_repository() -> None:
    authorization, operator, security = _authorization()
    repository = _Repository(authorization, {})
    decision = r2.assess_plan_r2_transition(
        authorization,
        {},
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    assert decision["allowed"] is False
    assert "typed_quack_plan_transition_unavailable" in decision["blockers"]
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="typed_quack"):
        r2.prepare_authorized_plan_r2_transition(
            repository,
            authorization,
            {},
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            trusted_capability_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    assert repository.prepare_calls == 0
    assert repository.apply_calls == 0


def test_prepare_is_read_only_and_apply_receipt_is_idempotently_immutable() -> None:
    authorization, operator, security = _authorization()
    capability, reviewer = _capability(authorization)
    repository = _Repository(authorization, capability)
    prepared = r2.prepare_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert prepared["authority_mutated"] is False
    assert repository.apply_calls == 0
    receipt = r2.apply_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        prepared,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    replay = r2.apply_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        prepared,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert replay == receipt
    assert receipt["replayed"] is False
    assert receipt["protected_tasks_unchanged"] is True
    observation = r2.observe_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    launch = r2.validate_plan_r2_launch_transition(
        repository=repository,
        authorization=authorization,
        transition_receipt=receipt,
        state_observation=observation,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    assert launch["valid"] is True
    assert launch["process_started"] is False
    assert observation["store_version"] == receipt["after_version"]


def test_plan_r2_crash_windows_require_apply_then_fresh_observation() -> None:
    authorization, operator, security = _authorization()
    capability, reviewer = _capability(authorization)
    repository = _Repository(authorization, capability)

    # Authorization and prepare alone grant no process birth and create no
    # transition/observation artifact.
    prepared = r2.prepare_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert repository.apply_calls == 0
    assert repository.observe_calls == 0
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="receipt"):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt={},
            state_observation={},
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )

    # Apply can survive a crash before capsule issuance, but launch remains
    # unavailable until a later authoritative read-only observation exists.
    receipt = r2.apply_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        prepared,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="observation"):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation={},
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    observation = r2.observe_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    with pytest.raises(
        r2.ExternalAgentPlanR2Error,
        match="owner_authenticated_plan_r2_live_readback_unavailable",
    ):
        r2.validate_plan_r2_launch_transition(
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=observation,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    swapped = deepcopy(observation)
    swapped["fence"] = int(swapped["fence"]) + 1
    swapped["observation_cid"] = r2._cid(
        {key: value for key, value in swapped.items() if key != "observation_cid"}
    )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="re-observed"):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=swapped,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )

    # A self-consistent forged receipt/observation pair has valid content
    # hashes, but the owner has no durable apply result for its transaction.
    forged_receipt = deepcopy(receipt)
    forged_receipt["transaction_cid"] = _sha("7")
    forged_receipt["receipt_cid"] = r2._cid(
        {
            key: value
            for key, value in forged_receipt.items()
            if key != "receipt_cid"
        }
    )
    forged_observation = deepcopy(observation)
    forged_observation["transition_receipt_cid"] = forged_receipt["receipt_cid"]
    forged_observation["transaction_cid"] = forged_receipt["transaction_cid"]
    forged_observation["observation_cid"] = r2._cid(
        {
            key: value
            for key, value in forged_observation.items()
            if key != "observation_cid"
        }
    )
    with pytest.raises(
        r2.ExternalAgentPlanR2Error,
        match="owner_authenticated_plan_r2_live_readback_failed",
    ):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt=forged_receipt,
            state_observation=forged_observation,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )

    repository.observation = deepcopy(observation)
    with pytest.raises(
        r2.ExternalAgentPlanR2Error,
        match="owner_authenticated_plan_r2_live_readback_is_stale",
    ):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=observation,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS + r2._MAX_LAUNCH_READBACK_AGE_MS + 1,
        )

    swapped_command = deepcopy(observation)
    swapped_command["authorized_apply_command_cid"] = _sha("6")
    swapped_command["observation_cid"] = r2._cid(
        {
            key: value
            for key, value in swapped_command.items()
            if key != "observation_cid"
        }
    )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="re-observed"):
        r2.validate_plan_r2_launch_transition(
            repository=repository,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=swapped_command,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )


def test_partial_protected_snapshot_and_authorization_swap_fail_closed() -> None:
    statement = _statement()
    partial = deepcopy(statement)
    partial["protected_tasks"][0].pop("task_row")
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="shape"):
        r2.prepare_plan_r2_transition_authorization(
            **{
                key: partial[key]
                for key in r2._STATEMENT_FIELDS
                if key
                not in {
                    "schema",
                    "statement_cid",
                    "population_cid",
                    "plan_root_cid",
                    "task_population_cid",
                    "dependency_population_cid",
                    "protected_tasks_root_cid",
                    "frontier_cid",
                    "authority",
                }
            }
        )

    authorization, operator, security = _authorization()
    capability, reviewer = _capability(authorization)
    swapped = deepcopy(authorization)
    swapped["fencing_token"] = int(swapped["fencing_token"]) + 1
    decision = r2.assess_plan_r2_transition(
        swapped,
        capability,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert decision["allowed"] is False
    assert any(item.startswith("transition_authorization_invalid") for item in decision["blockers"])


def test_plan_r2_publication_recovers_exact_partial_replay(
    tmp_path: Path,
) -> None:
    authorization, operator, security = _authorization()
    capability, reviewer = _capability(authorization)
    repository = _Repository(authorization, capability)
    prepared = r2.prepare_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    receipt = r2.apply_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        prepared,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    observation = r2.observe_authorized_plan_r2_transition(
        repository,
        authorization,
        capability,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        trusted_capability_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    authority_root = tmp_path / r2.EAAEF_AUTHORITY_REGISTRY_PREFIX
    authority_root.mkdir(parents=True, mode=0o700)
    current = tmp_path
    for part in Path(r2.EAAEF_AUTHORITY_REGISTRY_PREFIX).parts:
        current /= part
        current.chmod(0o700)
    receipt_path = r2.plan_r2_transition_receipt_relative_path(
        str(authorization["source_head"]),
        str(authorization["plan_root_cid"]),
    )
    # Simulate a crash after receipt publication but before observation.
    r2._publish_or_confirm_plan_r2_artifact(
        tmp_path,
        receipt_path,
        receipt,
        noun="Plan R2 transition receipt",
    )
    first = r2.publish_plan_r2_transition_result(
        tmp_path,
        repository=repository,
        authorization=authorization,
        transition_receipt=receipt,
        state_observation=observation,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    replay = r2.publish_plan_r2_transition_result(
        tmp_path,
        repository=repository,
        authorization=authorization,
        transition_receipt=receipt,
        state_observation=observation,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    assert replay == first
    tampered = deepcopy(receipt)
    tampered["transaction_cid"] = _sha("3")
    tampered["receipt_cid"] = r2._cid(
        {key: value for key, value in tampered.items() if key != "receipt_cid"}
    )
    with pytest.raises(r2.ExternalAgentPlanR2Error, match="conflicts"):
        r2._publish_or_confirm_plan_r2_artifact(
            tmp_path,
            receipt_path,
            tampered,
            noun="Plan R2 transition receipt",
        )
