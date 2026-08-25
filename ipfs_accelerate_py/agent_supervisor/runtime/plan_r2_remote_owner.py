"""Canonical process-remote transport for the distinct Plan-R2 owner path.

The command-fabric owner already implements atomic, exactly replayable
``prepare/apply/observe`` Plan-R2 operations.  This module transports the full
operation payload to that owner without adding those operations to the R1
bootstrap gateway.  Only canonical bytes cross the process boundary.

The client journals the exact signed envelope and wire request before sending
it.  If the response is lost, a retry resends those same bytes; the owner then
adopts its already durable result.  Divergent reuse of a submission identity
is rejected locally.  No filesystem path, token, callback, Portal, DuckDB
handle, generic ``StateCommand``, merge operation, or process-birth authority
is present in the wire protocol.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from ..planning.external_agent_plan_r2 import (
    AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
)
from ..task_sources.control_plane_contracts import CommandKind
from ..task_sources.eaaef_plan_r2_owner_service import (
    EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER,
    EAAEFPlanR2TypedOwnerCommandClient,
    bind_eaaef_plan_r2_typed_owner_command_client,
)
from ..task_sources.external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PLAN_R2_OWNER_GATEWAY_INTERFACE,
    PLAN_R2_OWNER_OPERATION_SCHEMA,
    PREPARE_PLAN_R2_OPERATION,
)
from ..task_sources.plan_revision_store import (
    PlanRevisionStore,
    PlanRevisionStoreError,
)
from ..task_sources.quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
)
from ..task_sources.quack_command_fabric import QuackPlanR2OwnerGateway
from ..validation.plan_r2_remote_owner_admission import (
    PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE,
    PLAN_R2_REMOTE_OPERATIONS,
    PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE,
    PLAN_R2_REMOTE_REQUEST_SCHEMA,
    PLAN_R2_REMOTE_RESPONSE_SCHEMA,
    PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
    VerifiedPlanR2RemoteOwnerAdmission,
)

PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_INTERFACE: Final = "PlanR2RemoteExactEnvelopeJournal@1"
PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-r2-remote-exact-envelope-journal@1"
)
PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS: Final = (
    "source_complete_external_signed_channel_required"
)
PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS: Final = (
    "external_plan_r2_remote_owner_capability_absent",
    "qualified_process_remote_wire_channel_factory_absent",
    "supervisor_plan_r2_remote_repository_wiring_absent",
)
PLAN_R2_TYPED_OWNER_CHANNEL_QUALIFICATION_STATUS: Final = (
    "typed_owner_canonical_channel_implemented_cutover_unqualified"
)

_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "remote_capability_cid",
        "plan_r2_operational_capability_cid",
        "plan_r2_authorization_cid",
        "operation",
        "envelope",
        "operation_payload",
        "request_cid",
    }
)
_RESPONSE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "remote_capability_cid",
        "request_cid",
        "operation",
        "envelope_cid",
        "operation_payload_cid",
        "result",
        "result_cid",
        "response_cid",
    }
)
_PAYLOAD_FIELDS: Final = {
    PREPARE_PLAN_R2_OPERATION: frozenset({"schema", "operation", "authorization"}),
    APPLY_PLAN_R2_OPERATION: frozenset(
        {"schema", "operation", "authorization", "prepared_projection"}
    ),
    OBSERVE_PLAN_R2_OPERATION: frozenset(
        {"schema", "operation", "authorization", "transition_receipt"}
    ),
}
_COMMAND_PARAMETER_FIELDS: Final = frozenset(
    {
        "interface",
        "operation",
        "authorization_cid",
        "statement_cid",
        "operation_payload_cid",
        "shard_id",
        "store_id",
        "expected_event_cursor",
        "population_cid",
        "protected_tasks_root_cid",
        "prepared_projection_cid",
        "transition_receipt_cid",
    }
)
_FORBIDDEN_CHANNEL_AUTHORITY: Final = (
    "database_path",
    "connection",
    "execute",
    "execute_sql",
    "filesystem_path",
    "owner_submit",
    "portal",
    "raw_token",
    "token",
    "transport_token",
)
_JOURNAL_FACTORY_TOKEN = object()
_TYPED_OWNER_CHANNEL_FACTORY_TOKEN = object()


class PlanR2RemoteOwnerError(RuntimeError):
    """The remote Plan-R2 path failed closed."""


class PlanR2RemoteResponseUnavailable(PlanR2RemoteOwnerError):
    """The exact request may have committed, but no response was received."""


class PlanR2RemoteReplayDiverged(PlanR2RemoteOwnerError):
    """A durable request/response differs from the exact replay."""


def _canonical_bytes(value: Any, *, noun: str, maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PlanR2RemoteOwnerError(f"{noun} is not canonical JSON") from exc
    if len(encoded) > maximum:
        raise PlanR2RemoteOwnerError(f"{noun} exceeds its signed byte bound")
    return encoded


def _cid(value: Any, *, maximum: int) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            _canonical_bytes(value, noun="content identity payload", maximum=maximum)
        ).hexdigest()
    )


def _decode_exact_object(raw: object, *, noun: str, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise PlanR2RemoteOwnerError(f"{noun} is not bounded canonical bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanR2RemoteOwnerError(f"{noun} is not canonical JSON") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PlanR2RemoteOwnerError(f"{noun} is not an object")
    if _canonical_bytes(value, noun=noun, maximum=maximum) != raw:
        raise PlanR2RemoteOwnerError(f"{noun} bytes are not canonical")
    return value


def _detached(value: object, *, noun: str, maximum: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PlanR2RemoteOwnerError(f"{noun} is not an object")
    return json.loads(_canonical_bytes(dict(value), noun=noun, maximum=maximum))


def _validate_request(
    value: Mapping[str, Any],
    *,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> tuple[dict[str, Any], AuthorizedStateCommand, dict[str, Any]]:
    request = _detached(
        value,
        noun="remote Plan-R2 request",
        maximum=int(admission["maximum_request_bytes"]),
    )
    body = dict(request)
    request_cid = str(body.pop("request_cid", ""))
    operation = str(request.get("operation") or "")
    if (
        set(request) != _REQUEST_FIELDS
        or request.get("schema") != PLAN_R2_REMOTE_REQUEST_SCHEMA
        or request.get("interface") != PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE
        or request.get("remote_capability_cid") != admission.capability_cid
        or request.get("plan_r2_operational_capability_cid")
        != admission["plan_r2_operational_capability_cid"]
        or request.get("plan_r2_authorization_cid") != admission["plan_r2_authorization_cid"]
        or operation not in PLAN_R2_REMOTE_OPERATIONS
        or request_cid != _cid(body, maximum=int(admission["maximum_request_bytes"]))
    ):
        raise PlanR2RemoteOwnerError("remote Plan-R2 request identity or operation is invalid")
    payload = _detached(
        request.get("operation_payload"),
        noun="remote Plan-R2 operation payload",
        maximum=int(admission["maximum_request_bytes"]),
    )
    if (
        set(payload) != _PAYLOAD_FIELDS[operation]
        or payload.get("schema") != PLAN_R2_OWNER_OPERATION_SCHEMA
        or payload.get("operation") != operation
    ):
        raise PlanR2RemoteOwnerError("remote Plan-R2 operation payload shape is invalid")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PlanR2RemoteOwnerError("remote Plan-R2 operation has no transition authorization")
    authorization_bindings = {
        "authorization_cid": admission["plan_r2_authorization_cid"],
        "board_namespace": admission["board_namespace"],
        "source_head": admission["source_head"],
        "source_tree": admission["source_tree"],
        "plan_root_cid": admission["plan_root_cid"],
        "population_cid": admission["population_cid"],
        "quack_command_fabric_qualification_cid": admission[
            "quack_command_fabric_qualification_cid"
        ],
        "owner_principal_did": admission["owner_principal_did"],
        "shard_id": admission["shard_id"],
        "store_id": admission["store_id"],
        "owner_generation": admission["owner_generation"],
        "expected_epoch": admission["epoch"],
        "fencing_token": admission["fence"],
    }
    mismatched = sorted(
        field
        for field, expected in authorization_bindings.items()
        if authorization.get(field) != expected
    )
    if mismatched:
        raise PlanR2RemoteOwnerError(
            "remote Plan-R2 authorization differs from admission: " + ", ".join(mismatched)
        )
    envelope_value = request.get("envelope")
    if not isinstance(envelope_value, Mapping):
        raise PlanR2RemoteOwnerError("remote Plan-R2 request has no signed envelope")
    try:
        envelope = AuthorizedStateCommand.from_dict(envelope_value)
    except QuackCommandAuthorizationError as exc:
        raise PlanR2RemoteOwnerError("remote Plan-R2 request envelope is invalid") from exc
    if type(envelope) is not AuthorizedStateCommand or envelope.to_dict() != dict(envelope_value):
        raise PlanR2RemoteOwnerError("remote Plan-R2 request envelope is not the exact base type")
    command = envelope.command
    parameters = command.parameters
    payload_cid = _cid(payload, maximum=int(admission["maximum_request_bytes"]))
    expected_kind = (
        CommandKind.MIGRATE if operation == APPLY_PLAN_R2_OPERATION else CommandKind.OBSERVE
    )
    envelope_checks = (
        envelope.principal_did == admission["authorized_principal_did"],
        envelope.approver_did == admission["independent_approver_did"],
        envelope.authority_ref_cid == admission["plan_r2_authorization_cid"],
        envelope.board_namespace == admission["board_namespace"],
        envelope.shard_id == admission["shard_id"],
        envelope.owner_principal_did == admission["owner_principal_did"],
        envelope.scope_id == admission["plan_root_cid"],
        command.store_id == admission["store_id"],
        command.expected_generation == admission["owner_generation"],
        command.fence_epoch == admission["fence"],
        command.command_kind is expected_kind,
        envelope.effect == f"control-plane/{expected_kind.value}",
        set(parameters) == _COMMAND_PARAMETER_FIELDS,
        parameters.get("interface") == AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        parameters.get("operation") == operation,
        parameters.get("authorization_cid") == admission["plan_r2_authorization_cid"],
        parameters.get("operation_payload_cid") == payload_cid,
        parameters.get("shard_id") == admission["shard_id"],
        parameters.get("store_id") == admission["store_id"],
        parameters.get("population_cid") == admission["population_cid"],
    )
    if not all(envelope_checks):
        raise PlanR2RemoteOwnerError(
            "remote Plan-R2 signed envelope differs from its exclusive admission"
        )
    return request, envelope, payload


def build_plan_r2_remote_request(
    *,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
    envelope: AuthorizedStateCommand,
    operation_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build and reparse the exact bytes that may cross the channel."""

    if type(admission) is not VerifiedPlanR2RemoteOwnerAdmission:
        raise PlanR2RemoteOwnerError("remote Plan-R2 request requires exact verified admission")
    if type(envelope) is not AuthorizedStateCommand:
        raise PlanR2RemoteOwnerError("remote Plan-R2 request requires exact AuthorizedStateCommand")
    payload = _detached(
        operation_payload,
        noun="remote Plan-R2 operation payload",
        maximum=int(admission["maximum_request_bytes"]),
    )
    operation = str(payload.get("operation") or "")
    body = {
        "schema": PLAN_R2_REMOTE_REQUEST_SCHEMA,
        "interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "remote_capability_cid": admission.capability_cid,
        "plan_r2_operational_capability_cid": admission["plan_r2_operational_capability_cid"],
        "plan_r2_authorization_cid": admission["plan_r2_authorization_cid"],
        "operation": operation,
        "envelope": envelope.to_dict(),
        "operation_payload": payload,
    }
    request = {
        **body,
        "request_cid": _cid(body, maximum=int(admission["maximum_request_bytes"])),
    }
    validated, _envelope, _payload = _validate_request(request, admission=admission)
    return MappingProxyType(validated)


def encode_plan_r2_remote_request(
    request: Mapping[str, Any], *, admission: VerifiedPlanR2RemoteOwnerAdmission
) -> bytes:
    validated, _envelope, _payload = _validate_request(request, admission=admission)
    return _canonical_bytes(
        validated,
        noun="remote Plan-R2 request",
        maximum=int(admission["maximum_request_bytes"]),
    )


def decode_plan_r2_remote_request(
    raw: bytes, *, admission: VerifiedPlanR2RemoteOwnerAdmission
) -> Mapping[str, Any]:
    value = _decode_exact_object(
        raw,
        noun="remote Plan-R2 request",
        maximum=int(admission["maximum_request_bytes"]),
    )
    request, _envelope, _payload = _validate_request(value, admission=admission)
    return MappingProxyType(request)


def _build_response(
    *,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
    request: Mapping[str, Any],
    envelope: AuthorizedStateCommand,
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    detached_result = _detached(
        result,
        noun="remote Plan-R2 owner result",
        maximum=int(admission["maximum_response_bytes"]),
    )
    body = {
        "schema": PLAN_R2_REMOTE_RESPONSE_SCHEMA,
        "interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "remote_capability_cid": admission.capability_cid,
        "request_cid": request["request_cid"],
        "operation": request["operation"],
        "envelope_cid": envelope.envelope_cid,
        "operation_payload_cid": _cid(payload, maximum=int(admission["maximum_request_bytes"])),
        "result": detached_result,
        "result_cid": _cid(detached_result, maximum=int(admission["maximum_response_bytes"])),
    }
    return {
        **body,
        "response_cid": _cid(body, maximum=int(admission["maximum_response_bytes"])),
    }


def _validate_response(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> dict[str, Any]:
    response = _detached(
        value,
        noun="remote Plan-R2 response",
        maximum=int(admission["maximum_response_bytes"]),
    )
    body = dict(response)
    response_cid = str(body.pop("response_cid", ""))
    envelope = request["envelope"]
    payload = request["operation_payload"]
    result = response.get("result")
    if (
        not isinstance(envelope, Mapping)
        or not isinstance(payload, Mapping)
        or not isinstance(result, Mapping)
    ):
        raise PlanR2RemoteOwnerError("remote Plan-R2 response joins invalid request data")
    expected = {
        "schema": PLAN_R2_REMOTE_RESPONSE_SCHEMA,
        "interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "remote_capability_cid": admission.capability_cid,
        "request_cid": request["request_cid"],
        "operation": request["operation"],
        "envelope_cid": envelope.get("envelope_cid"),
        "operation_payload_cid": _cid(payload, maximum=int(admission["maximum_request_bytes"])),
    }
    if (
        set(response) != _RESPONSE_FIELDS
        or any(response.get(field) != item for field, item in expected.items())
        or response.get("result_cid")
        != _cid(result, maximum=int(admission["maximum_response_bytes"]))
        or response_cid != _cid(body, maximum=int(admission["maximum_response_bytes"]))
    ):
        raise PlanR2RemoteReplayDiverged("remote Plan-R2 response does not bind the exact request")
    return response


def encode_plan_r2_remote_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> bytes:
    value = _validate_response(response, request=request, admission=admission)
    return _canonical_bytes(
        value,
        noun="remote Plan-R2 response",
        maximum=int(admission["maximum_response_bytes"]),
    )


def decode_plan_r2_remote_response(
    raw: bytes,
    *,
    request: Mapping[str, Any],
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> Mapping[str, Any]:
    value = _decode_exact_object(
        raw,
        noun="remote Plan-R2 response",
        maximum=int(admission["maximum_response_bytes"]),
    )
    return MappingProxyType(_validate_response(value, request=request, admission=admission))


@runtime_checkable
class PlanR2CanonicalWireChannel(Protocol):
    """Qualified process channel; its public surface contains no authority."""

    INTERFACE: str
    request_channel_id: str
    response_channel_id: str

    def attach(self) -> None: ...

    def exchange(
        self,
        request_bytes: bytes,
        *,
        request_cid: str,
        maximum_wait_ms: int,
    ) -> bytes: ...

    def close(self) -> None: ...


class TypedStateOwnerPlanR2CanonicalWireChannel:
    """Canonical-byte channel over one authenticated typed-owner session.

    The factory accepts neither a socket path nor token.  It narrows an
    already-authenticated :class:`TypedStateOwnerConnection` to the existing
    three-operation EAAEF Plan-R2 client, then binds both public channel IDs
    directly from the exact signed remote-owner admission.  The underlying
    typed-owner connection remains shared and is not closed by this view.

    This closes the source-level channel implementation gap only.  The
    independently signed single-owner cutover remains a separate production
    gate and is deliberately exposed by :meth:`require_production_admission`.
    """

    INTERFACE: ClassVar[str] = PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE
    __slots__ = (
        "_request_channel_id",
        "_response_channel_id",
        "_admission",
        "_client",
        "_attached",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        client: EAAEFPlanR2TypedOwnerCommandClient,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
    ) -> None:
        if (
            token is not _TYPED_OWNER_CHANNEL_FACTORY_TOKEN
            or type(client) is not EAAEFPlanR2TypedOwnerCommandClient
            or type(admission) is not VerifiedPlanR2RemoteOwnerAdmission
        ):
            raise PlanR2RemoteOwnerError(
                "typed-owner Plan-R2 channels require the exact authenticated "
                "client and verified admission"
            )
        client_bindings = {
            "capability_cid": client._remote_capability_cid,  # noqa: SLF001
            "plan_r2_operational_capability_cid": (
                client._operational_capability_cid  # noqa: SLF001
            ),
            "plan_r2_authorization_cid": client._authorization_cid,  # noqa: SLF001
        }
        mismatched = sorted(
            name
            for name, observed in client_bindings.items()
            if observed != admission[name]
        )
        if mismatched:
            raise PlanR2RemoteOwnerError(
                "typed-owner Plan-R2 client differs from signed admission: "
                + ", ".join(mismatched)
            )
        self._request_channel_id = str(admission["request_channel_id"])
        self._response_channel_id = str(admission["response_channel_id"])
        self._admission = admission
        self._client = client
        self._attached = False
        self._closed = False

    @property
    def request_channel_id(self) -> str:
        return self._request_channel_id

    @property
    def response_channel_id(self) -> str:
        return self._response_channel_id

    def attach(self) -> None:
        if self._closed:
            raise PlanR2RemoteOwnerError("typed-owner Plan-R2 channel is closed")
        if not self._attached:
            self._client.attach()
            self._attached = True

    def exchange(
        self,
        request_bytes: bytes,
        *,
        request_cid: str,
        maximum_wait_ms: int,
    ) -> bytes:
        if self._closed or not self._attached:
            raise PlanR2RemoteOwnerError(
                "typed-owner Plan-R2 channel is not attached"
            )
        if (
            type(request_cid) is not str
            or type(maximum_wait_ms) is not int
            or maximum_wait_ms != int(self._admission["maximum_wait_ms"])
        ):
            raise PlanR2RemoteOwnerError(
                "typed-owner Plan-R2 exchange differs from signed wait authority"
            )
        raw_request = _decode_exact_object(
            request_bytes,
            noun="typed-owner Plan-R2 request",
            maximum=int(self._admission["maximum_request_bytes"]),
        )
        request, envelope, payload = _validate_request(
            raw_request,
            admission=self._admission,
        )
        if request["request_cid"] != request_cid:
            raise PlanR2RemoteReplayDiverged(
                "typed-owner Plan-R2 request CID differs from its exchange"
            )
        result = self._client.submit_authorized_plan_r2_operation(
            envelope,
            payload,
        )
        if not isinstance(result, Mapping):
            raise PlanR2RemoteOwnerError(
                "typed owner returned a non-object Plan-R2 result"
            )
        response = _build_response(
            admission=self._admission,
            request=request,
            envelope=envelope,
            payload=payload,
            result=result,
        )
        return encode_plan_r2_remote_response(
            response,
            request=request,
            admission=self._admission,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._attached = False
        self._closed = True

    def require_production_admission(self) -> None:
        """Preserve the independent single-owner cutover gate."""

        self._client.require_production_admission()

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "qualification_status": (
                    PLAN_R2_TYPED_OWNER_CHANNEL_QUALIFICATION_STATUS
                ),
                "production_admitted": False,
                "production_blocker": (
                    EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER
                ),
                "request_channel_id": self.request_channel_id,
                "response_channel_id": self.response_channel_id,
                "transport": "authenticated_typed_state_owner_connection",
                "canonical_bytes_only": True,
                "r1_operations_allowed": False,
                "generic_state_command_allowed": False,
                "database_authority_exposed": False,
                "filesystem_path_authority_exposed": False,
                "transport_token_exposed": False,
                "sql_exposed": False,
                "closes_shared_owner_connection": False,
                "attached": self._attached,
            }
        )


def bind_typed_state_owner_plan_r2_canonical_wire_channel(
    *,
    owner_connection: object,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> TypedStateOwnerPlanR2CanonicalWireChannel:
    """Narrow one authenticated typed-owner connection to canonical bytes.

    No socket path, token, callback, SQL surface, or database handle is
    accepted by this factory.  The existing typed-owner binder verifies the
    exact connection type and admission before this narrower channel is made.
    """

    if type(admission) is not VerifiedPlanR2RemoteOwnerAdmission:
        raise PlanR2RemoteOwnerError(
            "typed-owner Plan-R2 channel requires exact verified admission"
        )
    client = bind_eaaef_plan_r2_typed_owner_command_client(
        owner_connection=owner_connection,
        admission=admission,
    )
    return TypedStateOwnerPlanR2CanonicalWireChannel(
        _TYPED_OWNER_CHANNEL_FACTORY_TOKEN,
        client=client,
        admission=admission,
    )


class PlanR2RemoteExactEnvelopeJournal:
    """Crash-safe create-once request and response-loss adoption journal."""

    INTERFACE: ClassVar[str] = PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_INTERFACE
    SCHEMA: ClassVar[str] = PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_SCHEMA

    __slots__ = ("_store", "_capability_cid", "_authorization_cid")

    def __init__(
        self,
        token: object,
        *,
        store: PlanRevisionStore,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
    ) -> None:
        if token is not _JOURNAL_FACTORY_TOKEN:
            raise TypeError("remote Plan-R2 journals come from the exact store binder")
        if type(store) is not PlanRevisionStore:
            raise PlanR2RemoteOwnerError("remote Plan-R2 journal requires exact PlanRevisionStore")
        self._store = store
        self._capability_cid = admission.capability_cid
        self._authorization_cid = str(admission["plan_r2_authorization_cid"])

    @staticmethod
    def _key(request: Mapping[str, Any]) -> str:
        envelope = request.get("envelope")
        if not isinstance(envelope, Mapping):
            raise PlanR2RemoteReplayDiverged("journal request envelope is absent")
        submission_id = str(envelope.get("submission_id") or "")
        if not submission_id:
            raise PlanR2RemoteReplayDiverged("journal submission identity is absent")
        digest = hashlib.sha256(
            json.dumps(
                {
                    "schema": "PlanR2RemoteSubmissionJournalIdentity@1",
                    "submission_id": submission_id,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        return f"plan-r2-remote-{digest}"

    def _load_locked(
        self, request: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None] | None:
        state = self._store.load_continuation(self._key(request))
        if state is None:
            return None
        fields = {
            "schema",
            "remote_capability_cid",
            "plan_r2_authorization_cid",
            "operation",
            "request_cid",
            "request_cas_cid",
            "envelope_cid",
            "phase",
            "response_cas_cid",
        }
        envelope = request.get("envelope")
        if not isinstance(envelope, Mapping):
            raise PlanR2RemoteReplayDiverged("journal request envelope is absent")
        if (
            set(state) != fields
            or state.get("schema") != self.SCHEMA
            or state.get("remote_capability_cid") != self._capability_cid
            or state.get("plan_r2_authorization_cid") != self._authorization_cid
            or state.get("operation") != request.get("operation")
            or state.get("request_cid") != request.get("request_cid")
            or state.get("envelope_cid") != envelope.get("envelope_cid")
            or state.get("phase") not in {"plan_r2_remote_prepared", "plan_r2_remote_committed"}
            or (state.get("phase") == "plan_r2_remote_prepared" and state.get("response_cas_cid"))
            or (
                state.get("phase") == "plan_r2_remote_committed"
                and not state.get("response_cas_cid")
            )
        ):
            raise PlanR2RemoteReplayDiverged("durable remote Plan-R2 request identity diverged")
        try:
            prior_request = self._store.get_cas(str(state["request_cas_cid"]))
            prior_response = (
                self._store.get_cas(str(state["response_cas_cid"]))
                if state["phase"] == "plan_r2_remote_committed"
                else None
            )
        except (KeyError, PlanRevisionStoreError) as exc:
            raise PlanR2RemoteReplayDiverged("durable remote Plan-R2 journal is corrupt") from exc
        if prior_request != dict(request):
            raise PlanR2RemoteReplayDiverged("durable remote Plan-R2 request bytes diverged")
        return MappingProxyType(prior_request), (
            MappingProxyType(prior_response) if prior_response is not None else None
        )

    def lookup(
        self, request: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None] | None:
        with self._store._thread_lock, self._store._guard():
            return self._load_locked(request)

    def recover_effect_envelope(
        self,
        operation_payload: Mapping[str, Any],
        *,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
    ) -> AuthorizedStateCommand | None:
        """Recover a prepared/apply envelope before an adapter allocates anew.

        Observe is intentionally excluded.  A later observe is a new live
        read, while prepare/apply are the one logical transition effect and
        must retain their original envelope across response loss or restart.
        """

        payload = _detached(
            operation_payload,
            noun="remote Plan-R2 recovery payload",
            maximum=int(admission["maximum_request_bytes"]),
        )
        operation = str(payload.get("operation") or "")
        if operation not in {PREPARE_PLAN_R2_OPERATION, APPLY_PLAN_R2_OPERATION}:
            return None
        authorization = payload.get("authorization")
        if not isinstance(authorization, Mapping):
            raise PlanR2RemoteReplayDiverged("remote Plan-R2 recovery authorization is absent")
        payload_cid = _cid(payload, maximum=int(admission["maximum_request_bytes"]))
        suffix = operation.replace(".", "-")
        submission_id = f"{authorization.get('request_id')}:{suffix}:{payload_cid}"
        synthetic = {"envelope": {"submission_id": submission_id}}
        with self._store._thread_lock, self._store._guard():
            state = self._store.load_continuation(self._key(synthetic))
            if state is None:
                return None
            try:
                request = self._store.get_cas(str(state["request_cas_cid"]))
            except (KeyError, PlanRevisionStoreError) as exc:
                raise PlanR2RemoteReplayDiverged(
                    "durable remote Plan-R2 recovery request is corrupt"
                ) from exc
            validated, envelope, prior_payload = _validate_request(request, admission=admission)
            if (
                validated["operation"] != operation
                or prior_payload != payload
                or envelope.submission_id != submission_id
            ):
                raise PlanR2RemoteReplayDiverged("durable remote Plan-R2 recovery request diverged")
            # Re-run the full state/response join checks before returning the
            # envelope, rather than trusting only the CAS pointer above.
            self._load_locked(validated)
            return envelope

    def prepare(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(request)
            if existing is not None:
                return existing[0]
            envelope = request["envelope"]
            request_cas_cid = self._store.put_cas(dict(request))
            state = {
                "schema": self.SCHEMA,
                "remote_capability_cid": self._capability_cid,
                "plan_r2_authorization_cid": self._authorization_cid,
                "operation": request["operation"],
                "request_cid": request["request_cid"],
                "request_cas_cid": request_cas_cid,
                "envelope_cid": envelope["envelope_cid"],
                # Namespaced outside PlanRevisionStore's own apply-recovery
                # state vocabulary, so opening the store cannot rewrite it.
                "phase": "plan_r2_remote_prepared",
                "response_cas_cid": "",
            }
            self._store.put_continuation(self._key(request), state)
            return MappingProxyType(dict(request))

    def commit(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(request)
            if existing is None:
                raise PlanR2RemoteReplayDiverged(
                    "remote Plan-R2 response has no prepared exact request"
                )
            if existing[1] is not None:
                if dict(existing[1]) != dict(response):
                    raise PlanR2RemoteReplayDiverged("durable remote Plan-R2 response diverged")
                return existing[1]
            envelope = request["envelope"]
            state = {
                "schema": self.SCHEMA,
                "remote_capability_cid": self._capability_cid,
                "plan_r2_authorization_cid": self._authorization_cid,
                "operation": request["operation"],
                "request_cid": request["request_cid"],
                "request_cas_cid": self._store.put_cas(dict(request)),
                "envelope_cid": envelope["envelope_cid"],
                "phase": "plan_r2_remote_committed",
                "response_cas_cid": self._store.put_cas(dict(response)),
            }
            self._store.put_continuation(self._key(request), state)
            return MappingProxyType(dict(response))


def bind_plan_r2_remote_exact_envelope_journal(
    *,
    store: PlanRevisionStore,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
) -> PlanR2RemoteExactEnvelopeJournal:
    """Bind caller-owned durable storage without exposing it on the wire."""

    if type(admission) is not VerifiedPlanR2RemoteOwnerAdmission:
        raise PlanR2RemoteOwnerError("remote Plan-R2 journal requires exact verified admission")
    return PlanR2RemoteExactEnvelopeJournal(
        _JOURNAL_FACTORY_TOKEN,
        store=store,
        admission=admission,
    )


class PlanR2RemoteOwnerService:
    """Owner-process decoder around the exact admitted Quack owner gateway."""

    INTERFACE: ClassVar[str] = PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE
    __slots__ = ("_admission", "_owner_gateway")

    def __init__(
        self,
        *,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
        owner_gateway: QuackPlanR2OwnerGateway,
    ) -> None:
        if type(admission) is not VerifiedPlanR2RemoteOwnerAdmission:
            raise PlanR2RemoteOwnerError(
                "remote Plan-R2 owner service requires exact verified admission"
            )
        if type(owner_gateway) is not QuackPlanR2OwnerGateway:
            raise PlanR2RemoteOwnerError(
                "remote Plan-R2 owner service requires exact Quack owner gateway"
            )
        if (
            owner_gateway.production_capability_cid
            != admission["plan_r2_operational_capability_cid"]
            or owner_gateway.command_fabric_qualification_cid
            != admission["quack_command_fabric_qualification_cid"]
        ):
            raise PlanR2RemoteOwnerError(
                "remote Plan-R2 owner gateway differs from signed admission"
            )
        self._admission = admission
        self._owner_gateway = owner_gateway

    def handle_exchange(self, request_bytes: bytes) -> bytes:
        request = decode_plan_r2_remote_request(request_bytes, admission=self._admission)
        _validated, envelope, payload = _validate_request(request, admission=self._admission)
        result = self._owner_gateway.submit_authorized_plan_r2_operation(envelope, payload)
        if not isinstance(result, Mapping):
            raise PlanR2RemoteOwnerError("remote Plan-R2 owner returned an untyped result")
        response = _build_response(
            admission=self._admission,
            request=request,
            envelope=envelope,
            payload=payload,
            result=result,
        )
        return encode_plan_r2_remote_response(
            response,
            request=request,
            admission=self._admission,
        )

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "remote_capability_cid": self._admission.capability_cid,
                "plan_r2_operational_capability_cid": self._admission[
                    "plan_r2_operational_capability_cid"
                ],
                "allowed_operations": list(PLAN_R2_REMOTE_OPERATIONS),
                "r1_operations_allowed": False,
                "generic_state_command_allowed": False,
                "database_authority_exposed": False,
                "transport_token_exposed": False,
            }
        )


class PlanR2ProcessRemoteOwnerGateway:
    """Client-side owner gateway consumed by ExternalAgentStateRepository."""

    INTERFACE: ClassVar[str] = PLAN_R2_OWNER_GATEWAY_INTERFACE
    REMOTE_INTERFACE: ClassVar[str] = PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE
    __slots__ = ("_admission", "_channel", "_journal", "_attached", "_closed")

    def __init__(
        self,
        *,
        admission: VerifiedPlanR2RemoteOwnerAdmission,
        channel: PlanR2CanonicalWireChannel,
        journal: PlanR2RemoteExactEnvelopeJournal,
    ) -> None:
        if type(admission) is not VerifiedPlanR2RemoteOwnerAdmission:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway requires exact verified admission")
        if not isinstance(channel, PlanR2CanonicalWireChannel):
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway requires the closed wire channel")
        if (
            getattr(channel, "INTERFACE", "") != PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE
            or channel.request_channel_id != admission["request_channel_id"]
            or channel.response_channel_id != admission["response_channel_id"]
        ):
            raise PlanR2RemoteOwnerError("remote Plan-R2 channel differs from signed admission")
        exposed = sorted(name for name in _FORBIDDEN_CHANNEL_AUTHORITY if hasattr(channel, name))
        if exposed:
            raise PlanR2RemoteOwnerError(
                "remote Plan-R2 channel exposes forbidden authority: " + ", ".join(exposed)
            )
        if type(journal) is not PlanR2RemoteExactEnvelopeJournal:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway requires exact durable journal")
        if (
            journal._capability_cid != admission.capability_cid
            or journal._authorization_cid != admission["plan_r2_authorization_cid"]
        ):
            raise PlanR2RemoteOwnerError("remote Plan-R2 journal differs from signed admission")
        self._admission = admission
        self._channel = channel
        self._journal = journal
        self._attached = False
        self._closed = False

    @property
    def production_capability_cid(self) -> str:
        return str(self._admission["plan_r2_operational_capability_cid"])

    @property
    def remote_capability_cid(self) -> str:
        return self._admission.capability_cid

    @property
    def command_fabric_qualification_cid(self) -> str:
        return str(self._admission["quack_command_fabric_qualification_cid"])

    def attach(self) -> None:
        if self._closed:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway is closed")
        if not self._attached:
            self._channel.attach()
            self._attached = True

    def close(self) -> None:
        if self._closed:
            return
        self._channel.close()
        self._attached = False
        self._closed = True

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._closed:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway is closed")
        if not self._attached:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway is not attached")
        request = build_plan_r2_remote_request(
            admission=self._admission,
            envelope=envelope,
            operation_payload=operation_payload,
        )
        existing = self._journal.lookup(request)
        if existing is not None and existing[1] is not None:
            response = _validate_response(existing[1], request=request, admission=self._admission)
            return MappingProxyType(dict(response["result"]))
        if existing is None:
            if time.time_ns() // 1_000_000 >= int(self._admission["expires_at_ms"]):
                raise PlanR2RemoteOwnerError(
                    "remote Plan-R2 capability expired before request preparation"
                )
            self._journal.prepare(request)
        request_bytes = encode_plan_r2_remote_request(request, admission=self._admission)
        try:
            response_bytes = self._channel.exchange(
                request_bytes,
                request_cid=str(request["request_cid"]),
                maximum_wait_ms=int(self._admission["maximum_wait_ms"]),
            )
        except Exception as exc:
            raise PlanR2RemoteResponseUnavailable(
                "remote Plan-R2 response unavailable; retry the exact envelope"
            ) from exc
        response = decode_plan_r2_remote_response(
            response_bytes,
            request=request,
            admission=self._admission,
        )
        committed = self._journal.commit(request, response)
        return MappingProxyType(dict(committed["result"]))

    def recover_exact_authorized_plan_r2_envelope(
        self, operation_payload: Mapping[str, Any]
    ) -> AuthorizedStateCommand | None:
        """Return only the prior exact prepare/apply envelope, never a new one."""

        if self._closed:
            raise PlanR2RemoteOwnerError("remote Plan-R2 gateway is closed")
        return self._journal.recover_effect_envelope(
            operation_payload,
            admission=self._admission,
        )

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.REMOTE_INTERFACE,
                "owner_gateway_interface": self.INTERFACE,
                "qualification_status": PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS,
                "remote_capability_cid": self.remote_capability_cid,
                "plan_r2_operational_capability_cid": self.production_capability_cid,
                "allowed_operations": list(PLAN_R2_REMOTE_OPERATIONS),
                "canonical_bytes_only": True,
                "durable_exact_envelope_journal": True,
                "response_loss_adoption": True,
                "r1_operations_allowed": False,
                "merge_operations_allowed": False,
                "generic_state_command_allowed": False,
                "path_or_token_authority_exposed": False,
                "attached": self._attached,
            }
        )


def bind_plan_r2_process_remote_owner_gateway(
    *,
    admission: VerifiedPlanR2RemoteOwnerAdmission,
    channel: PlanR2CanonicalWireChannel,
    journal: PlanR2RemoteExactEnvelopeJournal,
) -> PlanR2ProcessRemoteOwnerGateway:
    """Construct the distinct gateway; no endpoint, token, or callback accepted."""

    return PlanR2ProcessRemoteOwnerGateway(
        admission=admission,
        channel=channel,
        journal=journal,
    )


__all__ = (
    "PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_INTERFACE",
    "PLAN_R2_REMOTE_EXACT_ENVELOPE_JOURNAL_SCHEMA",
    "PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS",
    "PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS",
    "PLAN_R2_TYPED_OWNER_CHANNEL_QUALIFICATION_STATUS",
    "PlanR2CanonicalWireChannel",
    "PlanR2ProcessRemoteOwnerGateway",
    "PlanR2RemoteExactEnvelopeJournal",
    "PlanR2RemoteOwnerError",
    "PlanR2RemoteOwnerService",
    "PlanR2RemoteReplayDiverged",
    "PlanR2RemoteResponseUnavailable",
    "TypedStateOwnerPlanR2CanonicalWireChannel",
    "bind_plan_r2_process_remote_owner_gateway",
    "bind_plan_r2_remote_exact_envelope_journal",
    "bind_typed_state_owner_plan_r2_canonical_wire_channel",
    "build_plan_r2_remote_request",
    "decode_plan_r2_remote_request",
    "decode_plan_r2_remote_response",
    "encode_plan_r2_remote_request",
    "encode_plan_r2_remote_response",
)
