"""Owner-local Plan-R2 operations borrowing the sole CASF transaction boundary.

The historical :mod:`quack_command_fabric` implementation proves the three
Plan-R2 transaction bodies, but its topology owns ingress, operational, and
projection databases.  EAAEF cannot use that topology beside the CASF owner.
This adapter therefore reuses only those closed transaction algorithms while
borrowing the already-open connection and private lock from the exact
``TypedStateOwnerGateway``.  It opens and closes no database, creates no
sidecar, accepts no SQL/callback/path/connection factory, and exposes only the
three typed Plan-R2 operations.

Existing signed Plan-R2 transport admissions name the historical command
fabric qualification.  This implementation consequently remains an explicit
production no-go until independent reviewers sign a single-owner cutover; it
does not reinterpret the old qualification as authority for this topology.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..planning import external_agent_plan_r2 as plan_r2
from ..validation.plan_r2_remote_owner_admission import (
    PLAN_R2_REMOTE_OPERATIONS,
    VerifiedPlanR2RemoteOwnerAdmission,
)
from .control_plane_contracts import CommandKind
from .control_plane_transactions import StateTransaction
from .external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PLAN_R2_OWNER_GATEWAY_INTERFACE,
    PREPARE_PLAN_R2_OPERATION,
)
from .quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)
from .quack_command_fabric import (
    QuackCommandFabric,
    QuackCommandFabricCapabilityError,
    QuackCommandFabricStateError,
)

EAAEF_PLAN_R2_BORROWED_OWNER_INTERFACE: Final = (
    "EAAEFPlanR2BorrowedOwnerService@1"
)
EAAEF_PLAN_R2_BORROWED_OWNER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-borrowed-owner@1"
)
EAAEF_PLAN_R2_BORROWED_OWNER_QUALIFICATION_STATUS: Final = (
    "implemented_unqualified_fail_closed"
)
EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER: Final = (
    "independently_signed_single_owner_plan_r2_cutover_absent"
)

_SERVICE_FACTORY_TOKEN = object()
_CLIENT_FACTORY_TOKEN = object()


class EAAEFPlanR2OwnerServiceError(RuntimeError):
    """The closed owner-local Plan-R2 service rejected an operation."""


def _authorization_policy(
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> QuackCommandAuthorizationPolicy:
    return QuackCommandAuthorizationPolicy(
        board_namespace=str(admission["board_namespace"]),
        shard_id=str(admission["shard_id"]),
        store_id=str(admission["store_id"]),
        authority_ref_cid=str(admission["plan_r2_authorization_cid"]),
        owner_principal_did=str(admission["owner_principal_did"]),
        owner_generation=int(admission["owner_generation"]),
        fence_epoch=int(admission["fence"]),
        trusted_approver_dids=frozenset(
            {str(admission["independent_approver_did"])}
        ),
        authorized_principal_dids=frozenset(
            {str(admission["authorized_principal_did"])}
        ),
        allowed_command_kinds=frozenset(
            {CommandKind.OBSERVE, CommandKind.MIGRATE}
        ),
    )


class EAAEFPlanR2BorrowedOwnerService:
    """Three-operation service sharing the exact CASF connection and lock."""

    INTERFACE: ClassVar[str] = EAAEF_PLAN_R2_BORROWED_OWNER_INTERFACE
    REQUIRED_CUTOVER_GATEWAY_INTERFACE: ClassVar[str] = (
        PLAN_R2_OWNER_GATEWAY_INTERFACE
    )
    SCHEMA: ClassVar[str] = EAAEF_PLAN_R2_BORROWED_OWNER_SCHEMA
    __slots__ = (
        "_owner_gateway",
        "_connection",
        "_transaction_lock",
        "_admission",
        "_plan_capability",
        "_authorization",
        "_policy",
        "_trusted_capability_reviewer_dids",
        "_trusted_operator_dids",
        "_trusted_security_reviewer_dids",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        owner_gateway: Any,
        bootstrap_service: Any,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
        plan_r2_operational_capability: Mapping[str, Any],
        authorization: Mapping[str, Any],
        trusted_capability_reviewer_dids: Sequence[str],
        trusted_operator_dids: Sequence[str],
        trusted_security_reviewer_dids: Sequence[str],
    ) -> None:
        if token is not _SERVICE_FACTORY_TOKEN:
            raise TypeError(
                "borrowed Plan-R2 services come from the typed owner binder"
            )
        from .eaaef_typed_owner_service import EAAEFTypedOwnerCommandService
        from .typed_state_owner import TypedStateOwnerGateway

        if (
            type(owner_gateway) is not TypedStateOwnerGateway
            or type(bootstrap_service) is not EAAEFTypedOwnerCommandService
            or type(admission) is not VerifiedPlanR2RemoteOwnerAdmission
            or not isinstance(plan_r2_operational_capability, Mapping)
            or not isinstance(authorization, Mapping)
        ):
            raise EAAEFPlanR2OwnerServiceError(
                "Plan-R2 service requires exact owner and verified admission"
            )
        capability_reviewers = tuple(
            str(value) for value in trusted_capability_reviewer_dids
        )
        operators = tuple(str(value) for value in trusted_operator_dids)
        security_reviewers = tuple(
            str(value) for value in trusted_security_reviewer_dids
        )
        if (
            not capability_reviewers
            or not operators
            or not security_reviewers
            or len(
                set(capability_reviewers)
                | set(operators)
                | set(security_reviewers)
            )
            != len(set(capability_reviewers))
            + len(set(operators))
            + len(set(security_reviewers))
        ):
            raise EAAEFPlanR2OwnerServiceError(
                "Plan-R2 reviewer trust sets must be present and disjoint"
            )
        self._owner_gateway = owner_gateway
        self._connection = owner_gateway._connection  # noqa: SLF001
        self._transaction_lock = owner_gateway._transaction_lock  # noqa: SLF001
        self._admission = admission
        self._plan_capability = MappingProxyType(
            dict(plan_r2_operational_capability)
        )
        self._authorization = MappingProxyType(dict(authorization))
        self._trusted_capability_reviewer_dids = capability_reviewers
        self._trusted_operator_dids = operators
        self._trusted_security_reviewer_dids = security_reviewers
        self._policy = _authorization_policy(admission)
        self._closed = False
        self._verify_fresh_authority(now_ms=time.time_ns() // 1_000_000)
        self._verify_bootstrap_authority(bootstrap_service)
        identity = owner_gateway.identity
        if (
            owner_gateway.store_id != self._policy.store_id
            or str(identity.get("store_id") or "") != self._policy.store_id
            or int(identity.get("generation") or 0)
            != self._policy.owner_generation
            or int(identity.get("fence_epoch") or 0)
            != self._policy.fence_epoch
            or not callable(getattr(self._connection, "execute", None))
            or not callable(getattr(self._connection, "commit", None))
            or not callable(getattr(self._connection, "rollback", None))
        ):
            raise EAAEFPlanR2OwnerServiceError(
                "typed owner identity differs from Plan-R2 authority"
            )

    def _verify_bootstrap_authority(self, bootstrap_service: Any) -> None:
        bindings = dict(bootstrap_service.authority_bindings)
        admission_expected = {
            "board_namespace": bindings["board_namespace"],
            "source_head": bindings["source_head"],
            "source_tree": bindings["source_tree"],
            "quack_command_fabric_qualification_cid": bindings[
                "quack_command_fabric_qualification_cid"
            ],
            "owner_principal_did": bindings["owner_principal_did"],
            "shard_id": bindings["shard_id"],
            "store_id": bindings["store_id"],
            "owner_generation": bindings["owner_generation"],
            "fence": bindings["fence"],
        }
        authorization_expected = {
            **{
                name: value
                for name, value in admission_expected.items()
                if name != "fence"
            },
            "fencing_token": bindings["fence"],
            "bootstrap_admission_cid": bindings["bootstrap_admission_cid"],
            "source_generation_cid": bindings["source_generation_cid"],
            "r1_launch_capsule_cid": bindings["r1_launch_capsule_cid"],
            "expected_active_plan_root_cid": bindings[
                "active_plan_root_cid"
            ],
            "expected_active_plan_cid": bindings[
                "active_plan_revision_cid"
            ],
            "expected_active_plan_revision": bindings[
                "active_plan_revision"
            ],
        }
        capability_expected = {
            name: bindings[name]
            for name in (
                "source_head",
                "source_tree",
                "quack_command_fabric_qualification_cid",
                "owner_principal_did",
                "shard_id",
                "owner_generation",
            )
        }
        capability_expected.update(
            {
                "bootstrap_admission_cid": bindings[
                    "bootstrap_admission_cid"
                ],
                "fence": bindings["fence"],
            }
        )
        mismatched = {
            f"admission.{name}"
            for name, expected in admission_expected.items()
            if self._admission.get(name) != expected
        }
        mismatched.update(
            f"authorization.{name}"
            for name, expected in authorization_expected.items()
            if self._authorization.get(name) != expected
        )
        mismatched.update(
            f"capability.{name}"
            for name, expected in capability_expected.items()
            if self._plan_capability.get(name) != expected
        )
        if mismatched:
            raise EAAEFPlanR2OwnerServiceError(
                "Plan-R2 authority differs from bound R1: "
                + ", ".join(sorted(mismatched))
            )

    @property
    def operational_capability_cid(self) -> str:
        return str(self._plan_capability["capability_cid"])

    @property
    def remote_capability_cid(self) -> str:
        return str(self._admission["capability_cid"])

    @property
    def authorization_cid(self) -> str:
        return str(self._authorization["authorization_cid"])

    @property
    def maximum_request_bytes(self) -> int:
        return int(self._admission["maximum_request_bytes"])

    @property
    def maximum_response_bytes(self) -> int:
        return int(self._admission["maximum_response_bytes"])

    @property
    def legacy_command_fabric_qualification_cid(self) -> str:
        return str(self._admission["quack_command_fabric_qualification_cid"])

    def _require_open(self) -> None:
        if self._closed:
            raise EAAEFPlanR2OwnerServiceError(
                "borrowed Plan-R2 owner service is closed"
            )

    def _verify_fresh_authority(self, *, now_ms: int) -> None:
        if now_ms >= int(self._admission["expires_at_ms"]):
            raise EAAEFPlanR2OwnerServiceError(
                "single-owner Plan-R2 admission is expired"
            )
        try:
            capability = plan_r2.verify_plan_r2_operational_capability(
                self._plan_capability,
                trusted_reviewer_dids=(
                    self._trusted_capability_reviewer_dids
                ),
                now_ms=now_ms,
            )
            authorization = plan_r2.verify_plan_r2_transition_authorization(
                self._authorization,
                trusted_operator_dids=self._trusted_operator_dids,
                trusted_security_reviewer_dids=(
                    self._trusted_security_reviewer_dids
                ),
                now_ms=now_ms,
            )
        except plan_r2.ExternalAgentPlanR2Error as exc:
            raise EAAEFPlanR2OwnerServiceError(
                "single-owner Plan-R2 signed authority is invalid"
            ) from exc
        verified_expected = {
            "source_head": self._admission["source_head"],
            "source_tree": self._admission["source_tree"],
            "plan_root_cid": self._admission["plan_root_cid"],
            "population_cid": self._admission["population_cid"],
            "authorization_cid": self._admission[
                "plan_r2_authorization_cid"
            ],
            "owner_principal_did": self._admission[
                "owner_principal_did"
            ],
        }
        mismatched = sorted(
            name
            for name, value in verified_expected.items()
            if authorization.get(name) != value
        )
        bound_authorization = {
            "board_namespace": self._admission["board_namespace"],
            "shard_id": self._admission["shard_id"],
            "store_id": self._admission["store_id"],
            "owner_generation": self._admission["owner_generation"],
            "expected_epoch": self._admission["epoch"],
            "fencing_token": self._admission["fence"],
            "quack_command_fabric_qualification_cid": self._admission[
                "quack_command_fabric_qualification_cid"
            ],
        }
        mismatched.extend(
            name
            for name, value in bound_authorization.items()
            if self._authorization.get(name) != value
        )
        if (
            capability.get("capability_cid")
            != self._admission["plan_r2_operational_capability_cid"]
            or capability.get("owner_principal_did")
            != self._admission["owner_principal_did"]
            or capability.get("shard_id") != self._admission["shard_id"]
            or capability.get("owner_generation")
            != self._admission["owner_generation"]
            or capability.get("epoch") != self._admission["epoch"]
            or capability.get("fence") != self._admission["fence"]
            or capability.get("quack_command_fabric_qualification_cid")
            != self._admission["quack_command_fabric_qualification_cid"]
        ):
            mismatched.append("operational_capability")
        if mismatched:
            raise EAAEFPlanR2OwnerServiceError(
                "single-owner Plan-R2 authority join differs: "
                + ", ".join(sorted(set(mismatched)))
            )

    def _new_transaction(self, *, session_id: str = "") -> StateTransaction:
        return StateTransaction(
            self._connection,
            store_id=self._policy.store_id,
            session_id=session_id,
        )

    def _recover_result(
        self,
        *,
        envelope: AuthorizedStateCommand,
        payload_cid: str,
        operation: str,
    ) -> Mapping[str, Any] | None:
        transaction = self._new_transaction(
            session_id=envelope.command.session_id
        )
        try:
            transaction.begin()
            result = QuackCommandFabric._plan_r2_lookup_result(  # noqa: SLF001
                transaction,
                envelope_cid=envelope.envelope_cid,
                operation_payload_cid=payload_cid,
                operation=operation,
            )
            transaction.commit()
            return result
        except Exception:
            if transaction.active:
                transaction.rollback()
            raise

    def _validate_submission(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> tuple[str, dict[str, Any], Mapping[str, Any], str]:
        operation, payload, payload_cid = (
            QuackCommandFabric._plan_r2_submission_replay_identity(  # noqa: SLF001
                envelope,
                operation_payload,
            )
        )
        self._verify_fresh_authority(now_ms=now_ms)
        authorization = dict(payload["authorization"])
        decision = plan_r2.assess_plan_r2_transition(
            authorization,
            self._plan_capability,
            trusted_operator_dids=self._trusted_operator_dids,
            trusted_security_reviewer_dids=(
                self._trusted_security_reviewer_dids
            ),
            trusted_capability_reviewer_dids=(
                self._trusted_capability_reviewer_dids
            ),
            now_ms=now_ms,
        )
        if decision.get("allowed") is not True:
            raise QuackCommandFabricCapabilityError(
                ",".join(
                    str(item) for item in decision.get("blockers") or ()
                )
                or "typed_quack_plan_transition_unavailable"
            )
        if authorization != dict(self._authorization):
            raise QuackCommandAuthorizationError(
                "Plan-R2 payload differs from bound transition authority"
            )
        verify_authorized_state_command(
            envelope,
            policy=self._policy,
            now_ms=now_ms,
        )
        command = envelope.command
        suffix = operation.replace(".", "-")
        if operation == OBSERVE_PLAN_R2_OPERATION:
            suffix = f"{suffix}:{envelope.ingress_slot}"
        expected_kind = (
            CommandKind.MIGRATE.value
            if operation == APPLY_PLAN_R2_OPERATION
            else CommandKind.OBSERVE.value
        )
        expected_parameters = {
            "interface": "AuthorizedPlanR2TransitionRepository@1",
            "operation": operation,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "operation_payload_cid": payload_cid,
            "shard_id": authorization["shard_id"],
            "store_id": authorization["store_id"],
            "expected_event_cursor": authorization[
                "expected_event_cursor"
            ],
            "population_cid": authorization["population_cid"],
            "protected_tasks_root_cid": authorization[
                "protected_tasks_root_cid"
            ],
            "prepared_projection_cid": "",
            "transition_receipt_cid": "",
        }
        if operation == APPLY_PLAN_R2_OPERATION:
            prepared = payload.get("prepared_projection")
            if not isinstance(prepared, Mapping):
                raise QuackCommandAuthorizationError(
                    "Plan-R2 apply lacks its prepared projection"
                )
            plan_r2._validate_prepared(  # noqa: SLF001
                prepared,
                authorization=authorization,
                capability=self._plan_capability,
                now_ms=now_ms,
            )
            expected_parameters["prepared_projection_cid"] = str(
                prepared["projection_cid"]
            )
        elif operation == OBSERVE_PLAN_R2_OPERATION:
            receipt = payload.get("transition_receipt")
            if not isinstance(receipt, Mapping):
                raise QuackCommandAuthorizationError(
                    "Plan-R2 observe lacks its transition receipt"
                )
            plan_r2._validate_transition_receipt_for_launch(  # noqa: SLF001
                receipt,
                authorization=authorization,
                now_ms=now_ms,
            )
            expected_parameters["transition_receipt_cid"] = str(
                receipt["receipt_cid"]
            )
        identity_checks = (
            command.command_kind.value == expected_kind,
            command.command_id
            == f"{authorization['request_id']}:{suffix}",
            command.idempotency_key
            == f"{authorization['idempotency_key']}:{suffix}",
            command.store_id == authorization["store_id"],
            command.session_id == authorization["lease_id"],
            command.expected_generation == authorization["owner_generation"],
            command.expected_revision == authorization["expected_version"],
            command.fence_epoch == authorization["fencing_token"],
            envelope.request_id
            == f"{authorization['request_id']}:{suffix}",
            envelope.authority_ref_cid == authorization["authorization_cid"],
            envelope.board_namespace == authorization["board_namespace"],
            envelope.shard_id == authorization["shard_id"],
            envelope.owner_principal_did
            == authorization["owner_principal_did"],
            envelope.lease_id == authorization["lease_id"],
            envelope.scope_id == authorization["plan_root_cid"],
            envelope.one_use_nonce
            == f"{authorization['one_use_nonce']}:{suffix}",
            dict(command.parameters) == expected_parameters,
            authorization["store_id"] == self._policy.store_id,
            authorization["shard_id"] == self._policy.shard_id,
            authorization["shard_id"] != authorization["store_id"],
        )
        if not all(identity_checks):
            raise QuackCommandAuthorizationError(
                "Plan-R2 command/envelope/payload identity join failed"
            )
        return operation, authorization, self._plan_capability, payload_cid

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply one exact operation under the sole owner's transaction lock."""

        with self._transaction_lock:
            self._require_open()
            operation, _payload, replay_payload_cid = (
                QuackCommandFabric._plan_r2_submission_replay_identity(  # noqa: SLF001
                    envelope,
                    operation_payload,
                )
            )
            recovered = self._recover_result(
                envelope=envelope,
                payload_cid=replay_payload_cid,
                operation=operation,
            )
            if recovered is not None:
                return recovered
            now_ms = time.time_ns() // 1_000_000
            operation, authorization, capability, payload_cid = (
                self._validate_submission(
                    envelope,
                    operation_payload,
                    now_ms=now_ms,
                )
            )
            transaction = self._new_transaction(
                session_id=envelope.command.session_id
            )
            try:
                transaction.begin()
                QuackCommandFabric._plan_r2_assert_live_lease(  # noqa: SLF001
                    transaction,
                    envelope=envelope,
                    authorization=authorization,
                    now_ms=now_ms,
                )
                transaction.consume_authorized_command_replay_claims(
                    request_id=envelope.request_id,
                    one_use_nonce=envelope.one_use_nonce,
                    scope_id=envelope.scope_id,
                    effect=envelope.effect,
                )
                snapshot = QuackCommandFabric._plan_r2_active_snapshot(  # noqa: SLF001
                    transaction,
                    store_id=self._policy.store_id,
                    owner_status="ready",
                )
                if operation != OBSERVE_PLAN_R2_OPERATION:
                    QuackCommandFabric._plan_r2_assert_snapshot(  # noqa: SLF001
                        snapshot,
                        authorization,
                        capability,
                    )
                    QuackCommandFabric._plan_r2_assert_protected_rows(  # noqa: SLF001
                        transaction,
                        authorization,
                    )
                else:
                    generation = snapshot["generation"]
                    if (
                        snapshot["epoch"] != authorization["expected_epoch"]
                        or snapshot["fence"]
                        != authorization["fencing_token"]
                        or generation.generation
                        != authorization["owner_generation"]
                        or generation.fence_epoch
                        != authorization["fencing_token"]
                    ):
                        raise QuackCommandFabricStateError(
                            "Plan-R2 observation owner epoch/fence is stale"
                        )
                if operation == PREPARE_PLAN_R2_OPERATION:
                    result = QuackCommandFabric._prepare_plan_r2_result(  # noqa: SLF001
                        envelope=envelope,
                        authorization=authorization,
                        capability=capability,
                        snapshot=snapshot,
                        now_ms=now_ms,
                    )
                elif operation == APPLY_PLAN_R2_OPERATION:
                    result = QuackCommandFabric._apply_plan_r2_result(  # noqa: SLF001
                        transaction,
                        envelope=envelope,
                        authorization=authorization,
                        capability=capability,
                        prepared=dict(
                            operation_payload["prepared_projection"]
                        ),
                        snapshot=snapshot,
                        now_ms=now_ms,
                    )
                else:
                    result = QuackCommandFabric._observe_plan_r2_result(  # noqa: SLF001
                        transaction,
                        envelope=envelope,
                        authorization=authorization,
                        receipt=dict(
                            operation_payload["transition_receipt"]
                        ),
                        snapshot=snapshot,
                        now_ms=now_ms,
                    )
                QuackCommandFabric._plan_r2_record_result(  # noqa: SLF001
                    transaction,
                    envelope=envelope,
                    operation_payload_cid=payload_cid,
                    operation=operation,
                    result=result,
                    recorded_at=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(now_ms / 1000),
                    ),
                )
                transaction.commit()
                return MappingProxyType(dict(result))
            except Exception:
                if transaction.active:
                    transaction.rollback()
                recovered = self._recover_result(
                    envelope=envelope,
                    payload_cid=payload_cid,
                    operation=operation,
                )
                if recovered is not None:
                    return recovered
                raise

    def close(self) -> None:
        """Retire only this adapter; the outer owner keeps its resources."""

        with self._transaction_lock:
            self._closed = True

    def require_production_admission(self) -> None:
        raise EAAEFPlanR2OwnerServiceError(
            "Plan-R2 single-owner production no-go: "
            + EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER
        )

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "required_cutover_gateway_interface": (
                    self.REQUIRED_CUTOVER_GATEWAY_INTERFACE
                ),
                "schema": self.SCHEMA,
                "qualification_status": (
                    EAAEF_PLAN_R2_BORROWED_OWNER_QUALIFICATION_STATUS
                ),
                "production_admitted": False,
                "production_blocker": (
                    EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER
                ),
                "operations": list(PLAN_R2_REMOTE_OPERATIONS),
                "operation_count": 3,
                "bootstrap_r1_operations_allowed": False,
                "generic_state_command_allowed": False,
                "bound_by_typed_state_owner_gateway": True,
                "owner_catalog_id": self._owner_gateway.catalog_id,
                "owner_server_id": str(
                    self._owner_gateway.identity.get("server_id") or ""
                ),
                "borrows_open_connection": True,
                "shares_owner_transaction_lock": True,
                "opens_database": False,
                "closes_database": False,
                "arbitrary_sql_enabled": False,
                "local_sidecar_enabled": False,
                "downstream_catalog_enabled": False,
                "legacy_command_fabric_started": False,
            }
        )


class EAAEFPlanR2TypedOwnerCommandClient:
    """Narrow three-operation client over an authenticated owner connection."""

    INTERFACE: ClassVar[str] = PLAN_R2_OWNER_GATEWAY_INTERFACE
    __slots__ = (
        "_owner_connection",
        "_remote_capability_cid",
        "_operational_capability_cid",
        "_authorization_cid",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        owner_connection: Any,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
    ) -> None:
        from .typed_state_owner import TypedStateOwnerConnection

        if (
            token is not _CLIENT_FACTORY_TOKEN
            or type(owner_connection) is not TypedStateOwnerConnection
            or type(admission) is not VerifiedPlanR2RemoteOwnerAdmission
        ):
            raise EAAEFPlanR2OwnerServiceError(
                "Plan-R2 client requires one typed owner connection and admission"
            )
        self._owner_connection = owner_connection
        self._remote_capability_cid = str(admission["capability_cid"])
        self._operational_capability_cid = str(
            admission["plan_r2_operational_capability_cid"]
        )
        self._authorization_cid = str(
            admission["plan_r2_authorization_cid"]
        )
        self._closed = False

    def attach(self) -> None:
        if self._closed:
            raise EAAEFPlanR2OwnerServiceError(
                "typed owner Plan-R2 client is closed"
            )

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._closed:
            raise EAAEFPlanR2OwnerServiceError(
                "typed owner Plan-R2 client is closed"
            )
        return self._owner_connection.submit_eaaef_plan_r2_operation(
            envelope,
            dict(operation_payload),
            remote_capability_cid=self._remote_capability_cid,
            plan_r2_operational_capability_cid=(
                self._operational_capability_cid
            ),
            plan_r2_authorization_cid=self._authorization_cid,
        )

    def close(self) -> None:
        # The connection may also carry the R1 client.  Only its owner closes
        # that shared transport; this wrapper retires its narrower view.
        self._closed = True

    def require_production_admission(self) -> None:
        raise EAAEFPlanR2OwnerServiceError(
            "Plan-R2 single-owner production no-go: "
            + EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER
        )


def bind_eaaef_plan_r2_typed_owner_command_client(
    *,
    owner_connection: Any,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> EAAEFPlanR2TypedOwnerCommandClient:
    """Bind a path/token-free Plan-R2 view over an admitted owner channel."""

    return EAAEFPlanR2TypedOwnerCommandClient(
        _CLIENT_FACTORY_TOKEN,
        owner_connection=owner_connection,
        admission=admission,
    )


def bind_eaaef_plan_r2_borrowed_owner_service(
    *,
    owner_server: Any,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
    plan_r2_operational_capability: Mapping[str, Any],
    authorization: Mapping[str, Any],
    trusted_capability_reviewer_dids: Sequence[str],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
) -> EAAEFPlanR2BorrowedOwnerService:
    """Ask the sole typed owner to bind the three-operation adapter."""

    from ..runtime.quack_state_server import QuackStateServer

    if type(owner_server) is not QuackStateServer:
        raise EAAEFPlanR2OwnerServiceError(
            "Plan-R2 binding requires the exact Quack state owner"
        )
    return owner_server.bind_eaaef_plan_r2_owner_service(
        admission=admission,
        plan_r2_operational_capability=plan_r2_operational_capability,
        authorization=authorization,
        trusted_capability_reviewer_dids=(
            trusted_capability_reviewer_dids
        ),
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
    )


def _bind_eaaef_plan_r2_owner_service_from_gateway(
    *,
    owner_gateway: Any,
    bootstrap_service: Any,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
    plan_r2_operational_capability: Mapping[str, Any],
    authorization: Mapping[str, Any],
    trusted_capability_reviewer_dids: Sequence[str],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
) -> EAAEFPlanR2BorrowedOwnerService:
    return EAAEFPlanR2BorrowedOwnerService(
        _SERVICE_FACTORY_TOKEN,
        owner_gateway=owner_gateway,
        bootstrap_service=bootstrap_service,
        admission=admission,
        plan_r2_operational_capability=plan_r2_operational_capability,
        authorization=authorization,
        trusted_capability_reviewer_dids=(
            trusted_capability_reviewer_dids
        ),
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
    )


__all__ = [
    "EAAEF_PLAN_R2_BORROWED_OWNER_INTERFACE",
    "EAAEF_PLAN_R2_BORROWED_OWNER_QUALIFICATION_STATUS",
    "EAAEF_PLAN_R2_BORROWED_OWNER_SCHEMA",
    "EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER",
    "EAAEFPlanR2BorrowedOwnerService",
    "EAAEFPlanR2TypedOwnerCommandClient",
    "EAAEFPlanR2OwnerServiceError",
    "bind_eaaef_plan_r2_borrowed_owner_service",
    "bind_eaaef_plan_r2_typed_owner_command_client",
]
