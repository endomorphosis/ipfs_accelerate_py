"""Bounded Quack command ingress for the exact DuckDB/Quack 1.5.5 build.

This compatibility slice exposes only closed, independently authorized owner
operations.  Untrusted clients can append canonical, independently signed
``AuthorizedStateCommand@1`` envelopes to a one-table ingress database.  One
in-process repository owner verifies authority and applies the embedded
``StateCommand@1`` records to a distinct private DuckDB through
``QuackStateRepository@1``.  When an independently signed
``PlanR2OperationalCapability@1`` is pinned, the same owner also implements the
three exact prepare/apply/observe Plan-R2 operations as private transactions.
It publishes a separate two-table, read-only projection database and never
exposes a database handle or arbitrary SQL surface to either client.

Quack's authorization callback receives SQL text rather than a typed operation
identity.  Every served endpoint therefore exact-matches a finite set of SQL
strings.  Prefix, substring, and regular-expression authorization are never
used here.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_plane_contracts import (
    CommandOutcome,
    StateCommand,
    canonical_json_bytes,
)
from .control_plane_repository import QuackStateRepository
from .external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PLAN_R2_OWNER_GATEWAY_INTERFACE,
    PLAN_R2_OWNER_OPERATION_SCHEMA,
    PREPARE_PLAN_R2_OPERATION,
)
from .quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)

QUACK_COMMAND_FABRIC_INTERFACE: Final = "QuackCommandFabric@1"
QUACK_COMMAND_FABRIC_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/quack-command-fabric@1"
QUACK_DAEMON_OWNER_GATEWAY_INTERFACE: Final = (
    "AuthorizedStateCommandDaemonOwnerGateway@1"
)
QUACK_DAEMON_CANONICAL_HANDLER_INTERFACE: Final = (
    "QuackDaemonCanonicalOwnerOperationHandler@1"
)
REQUIRED_DUCKDB_VERSION: Final = "1.5.5"
REQUIRED_QUACK_BUILD: Final = "quack@1.5.5+core"
MAX_COMMAND_BYTES: Final = 65_536
MAX_AUTHORIZED_COMMAND_BYTES: Final = 73_728
MAX_ID_BYTES: Final = 512
MAX_INGRESS_ROWS: Final = 1_024
MAX_REMOTE_RECEIPT_ROWS: Final = 256
MAX_PROJECTED_RECEIPTS: Final = 10_000

_CATALOG_SCHEMAS_SQL: Final = """
SELECT catalog_name, schema_name
FROM information_schema.schemata
WHERE catalog_name NOT IN ('system', 'temp')
ORDER BY ALL
\t"""
_CATALOG_RELATIONS_SQL: Final = """
SELECT schema_name, sql, 'table'
FROM duckdb_tables()
UNION ALL
SELECT schema_name, view_name, 'view'
FROM duckdb_views()
\t"""
_INGRESS_REMOTE_INSERT_SQL: Final = (
    # Quack's insert protocol transports a DataChunk separately and represents
    # the complete remote row placeholder as one NULL in the authorization SQL.
    "INSERT INTO main.command_inbox VALUES (NULL)"
)
_STATE_QUERY: Final = (
    "SELECT task_cid, status, revision, body_json FROM state_rows ORDER BY task_cid"
)
_RECEIPT_QUERY: Final = (
    "SELECT submission_id, envelope_cid, request_id, principal_did, approver_did, "
    "authority_ref_cid, lease_id, one_use_nonce, command_id, idempotency_key, "
    "outcome, changed, revision, generation, fence_epoch, result_json, error, "
    "submitted_at, applied_at FROM apply_receipts ORDER BY submitted_at, submission_id "
    f"LIMIT {MAX_PROJECTED_RECEIPTS}"
)
_RECENT_RECEIPT_QUERY: Final = (
    "SELECT submission_id, envelope_cid, request_id, principal_did, approver_did, "
    "authority_ref_cid, lease_id, one_use_nonce, command_id, idempotency_key, "
    "outcome, changed, revision, generation, fence_epoch, result_json, error, "
    "submitted_at, applied_at FROM apply_receipts "
    "ORDER BY submitted_at DESC, submission_id DESC "
    f"LIMIT {MAX_REMOTE_RECEIPT_ROWS}"
)
_PRIVATE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/private-authorized-command-receipt@1"
)
_PRIVATE_RECEIPT_EVENT_TYPE: Final = "authorized_state_command_receipt"
_DIVERGENT_INGRESS_QUARANTINE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "divergent-authorized-command-ingress-quarantine@1"
)
_PLAN_R2_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/private-authorized-plan-r2-owner-result@1"
)
_PLAN_R2_RESULT_EVENT_TYPE: Final = "authorized_plan_r2_owner_result"
_PLAN_R2_APPLIED_EVENT_TYPE: Final = "plan_r2_population_applied"
_PLAN_R2_AUDIT_EVENT_TYPES: Final = frozenset(
    {_PRIVATE_RECEIPT_EVENT_TYPE, _PLAN_R2_RESULT_EVENT_TYPE}
)
_PLAN_R2_OPERATIONS: Final = frozenset(
    {PREPARE_PLAN_R2_OPERATION, APPLY_PLAN_R2_OPERATION, OBSERVE_PLAN_R2_OPERATION}
)
_PLAN_R2_PROTECTED_STATUSES: Final = frozenset(
    {"claimed", "running", "settling", "completed", "accepted"}
)


class QuackCommandFabricError(RuntimeError):
    """Base error for the compatibility slice."""


class QuackCommandFabricCapabilityError(QuackCommandFabricError):
    """The exact locked DuckDB/Quack capability is unavailable."""


class QuackCommandIngressError(QuackCommandFabricError):
    """A command cannot cross the bounded append-only ingress."""


class QuackCommandFabricStateError(QuackCommandFabricError):
    """The fabric lifecycle or topology is invalid."""


class QuackCommandCapabilityDecision(StrEnum):
    GO = "go"
    NO_GO = "no-go"


@dataclass(frozen=True)
class QuackCommandCapabilityResult:
    """Typed exact-build admission result; a mismatch never starts a server."""

    decision: QuackCommandCapabilityDecision
    duckdb_version: str
    quack_build: str
    extension_path: str
    extension_sha256: str
    expected_extension_sha256: str
    reason: str = ""

    @property
    def admitted(self) -> bool:
        return self.decision is QuackCommandCapabilityDecision.GO

    def require(self) -> None:
        if not self.admitted:
            raise QuackCommandFabricCapabilityError(
                self.reason or "exact DuckDB/Quack capability was not admitted"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUACK_COMMAND_FABRIC_SCHEMA,
            "interface": QUACK_COMMAND_FABRIC_INTERFACE,
            "decision": self.decision.value,
            "admitted": self.admitted,
            "duckdb_version": self.duckdb_version,
            "quack_build": self.quack_build,
            "extension_path": self.extension_path,
            "extension_sha256": self.extension_sha256,
            "expected_extension_sha256": self.expected_extension_sha256,
            "reason": self.reason,
            "command_interface": StateCommand.INTERFACE,
            "authorization_interface": AuthorizedStateCommand.INTERFACE,
            "repository_interface": QuackStateRepository.INTERFACE,
            "ingress_relation": "command_inbox",
            "ingress_relation_count": 1,
            "ingress_append_only": True,
            "operational_database_served": False,
            "state_endpoint_read_only": True,
            "transport_token_is_authority": False,
            "local_owner_verifies_effect_authorization": True,
        }


def _locked_quack_sha256(lock_path: Path, machine: str) -> str:
    key = f"profile.extension.quack.{machine}.bin_sha256="
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QuackCommandFabricCapabilityError(f"could not read Quack lock: {lock_path}") from exc
    for line in lines:
        if line.startswith(key):
            value = line.split("=", 1)[1].strip().lower()
            if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
                return value
    raise QuackCommandFabricCapabilityError(f"Quack lock has no valid {machine} binary digest")


def assess_quack_command_capability(
    *,
    duckdb_module: Any,
    extension_path: str | Path,
    lock_path: str | Path,
    machine: str,
) -> QuackCommandCapabilityResult:
    """Verify exact version and locked artifact bytes without installing."""

    path = Path(extension_path).resolve()
    version = str(getattr(duckdb_module, "__version__", "") or "")
    try:
        expected = _locked_quack_sha256(Path(lock_path).resolve(), machine)
    except QuackCommandFabricCapabilityError as exc:
        expected = ""
        lock_error = str(exc)
    else:
        lock_error = ""
    observed = ""
    reason = lock_error
    if not reason and not path.is_file():
        reason = "locked Quack extension artifact is missing"
    elif not reason:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            reason = "Quack extension artifact does not match repository lock"
    if not reason and version != REQUIRED_DUCKDB_VERSION:
        reason = f"DuckDB {REQUIRED_DUCKDB_VERSION} is required; observed {version or 'unknown'}"
    if not reason:
        probe = None
        try:
            probe = duckdb_module.connect(database=":memory:")
            escaped = str(path).replace("'", "''")
            probe.execute(f"LOAD '{escaped}'")
            observed_functions = {
                str(row[0])
                for row in probe.execute(
                    "SELECT DISTINCT function_name FROM duckdb_functions() "
                    "WHERE function_name IN ('quack_serve', 'quack_query', "
                    "'quack_stop', 'quack_check_token')"
                ).fetchall()
            }
            required = {
                "quack_serve",
                "quack_query",
                "quack_stop",
                "quack_check_token",
            }
            missing = sorted(required - observed_functions)
            if missing:
                reason = "locked Quack build lacks required functions: " + ", ".join(missing)
        except Exception as exc:
            reason = f"locked Quack build failed local LOAD: {type(exc).__name__}"
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception:
                    pass
    return QuackCommandCapabilityResult(
        decision=(
            QuackCommandCapabilityDecision.GO
            if not reason
            else QuackCommandCapabilityDecision.NO_GO
        ),
        duckdb_version=version,
        quack_build=REQUIRED_QUACK_BUILD,
        extension_path=str(path),
        extension_sha256=observed,
        expected_extension_sha256=expected,
        reason=reason,
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _receipt_event_id(submission_id: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "AuthorizedStateCommandSubmissionIdentity@1",
                "submission_id": str(submission_id),
            }
        )
    ).hexdigest()
    return f"authorized-command-receipt:sha256:{digest}"


def _plan_r2_result_event_id(envelope_cid: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "AuthorizedPlanR2OwnerResultIdentity@1",
                "envelope_cid": str(envelope_cid),
            }
        )
    ).hexdigest()
    return f"authorized-plan-r2-result:sha256:{digest}"


def _json_object(value: object, *, noun: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise QuackCommandFabricStateError(f"{noun} is not canonical JSON text")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise QuackCommandFabricStateError(f"{noun} is corrupt") from exc
    if not isinstance(decoded, Mapping) or not all(isinstance(key, str) for key in decoded):
        raise QuackCommandFabricStateError(f"{noun} is not an object")
    return dict(decoded)


def _listener_reachable(uri: str) -> bool:
    host_port = uri.removeprefix("quack:").removeprefix("//")
    host, separator, port_text = host_port.rpartition(":")
    if not separator:
        return False
    try:
        with socket.create_connection((host.strip("[]"), int(port_text)), timeout=0.1):
            return True
    except OSError:
        return False


def _validated_loopback_endpoint(uri: str, *, name: str) -> str:
    if not isinstance(uri, str) or not uri.startswith("quack:"):
        raise QuackCommandFabricStateError(f"{name} must be a quack loopback endpoint")
    host_port = uri.removeprefix("quack:").removeprefix("//")
    host, separator, port_text = host_port.rpartition(":")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise QuackCommandFabricStateError(
            f"{name} must contain a numeric loopback address and port"
        ) from exc
    if not separator or not address.is_loopback or not 1024 <= port <= 65535:
        raise QuackCommandFabricStateError(
            f"{name} must contain a non-privileged numeric loopback endpoint"
        )
    return uri


def _validated_transport_token(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 32 <= len(value.encode("utf-8")) <= 512
        or any(character.isspace() for character in value)
    ):
        raise QuackCommandFabricStateError(f"{name} must be a high-entropy opaque transport token")
    return value


class _ExactQuackEndpoint:
    """Own one Quack listener with finite exact-query authorization."""

    def __init__(
        self,
        *,
        duckdb_module: Any,
        database_path: Path,
        extension_path: Path,
        uri: str,
        token: str,
        authorization_name: str,
        allowed_sql: tuple[str, ...],
    ) -> None:
        self._duckdb = duckdb_module
        self.database_path = database_path
        self.extension_path = extension_path
        self.uri = uri
        self._token = token
        self.authorization_name = authorization_name
        self.allowed_sql = allowed_sql
        self.connection: Any | None = None
        self.started = False

    def start(self) -> Any:
        if self.started:
            raise QuackCommandFabricStateError("Quack endpoint is already started")
        connection = self._duckdb.connect(str(self.database_path))
        try:
            connection.execute(f"LOAD {_sql_literal(str(self.extension_path))}")
            comparisons = " OR ".join(
                f"query = {_sql_literal(statement)}" for statement in self.allowed_sql
            )
            connection.execute(
                f"CREATE OR REPLACE MACRO {self.authorization_name}(sid, query) "
                f"AS (sid IS NOT NULL AND query IS NOT NULL AND ({comparisons}))"
            )
            connection.execute("SET GLOBAL quack_authentication_function = 'quack_check_token'")
            connection.execute(
                "SET GLOBAL quack_authorization_function = " + _sql_literal(self.authorization_name)
            )
            connection.execute(
                "CALL quack_serve(?, token := ?, disable_ssl := true)",
                [self.uri, self._token],
            ).fetchall()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not _listener_reachable(self.uri):
                time.sleep(0.01)
            if not _listener_reachable(self.uri):
                raise QuackCommandFabricStateError(f"Quack listener did not start: {self.uri}")
        except Exception:
            connection.close()
            raise
        self.connection = connection
        self.started = True
        return connection

    def stop(self) -> None:
        if not self.started:
            return
        connection = self.connection
        error: BaseException | None = None
        try:
            if connection is None:
                raise QuackCommandFabricStateError("started endpoint lost connection")
            connection.execute("CALL quack_stop(?)", [self.uri]).fetchall()
        except BaseException as exc:
            error = exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _listener_reachable(self.uri):
            time.sleep(0.01)
        if _listener_reachable(self.uri) and error is None:
            error = QuackCommandFabricStateError(
                f"Quack listener remained reachable after stop: {self.uri}"
            )
        if connection is not None:
            try:
                connection.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        self.connection = None
        self.started = False
        if error is not None:
            raise QuackCommandFabricStateError(
                f"Quack endpoint failed clean stop: {self.uri}"
            ) from error


class QuackCommandClient:
    """Fixed-template client; exposes append, never SQL execution."""

    def __init__(
        self,
        *,
        duckdb_module: Any,
        extension_path: str | Path,
        endpoint: str,
        token: str,
        alias: str,
    ) -> None:
        if not alias.replace("_", "a").isalnum():
            raise QuackCommandIngressError("client alias must be an identifier")
        self._connection = duckdb_module.connect(database=":memory:")
        self._alias = alias
        self._endpoint = str(endpoint)
        path = str(Path(extension_path).resolve())
        try:
            self._connection.execute(f"LOAD {_sql_literal(path)}")
            self._connection.execute(
                f"ATTACH {_sql_literal(endpoint)} AS {alias} (TOKEN {_sql_literal(token)})"
            )
        except Exception:
            self._connection.close()
            raise
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def append(
        self,
        envelope: AuthorizedStateCommand,
    ) -> None:
        if self._closed:
            raise QuackCommandIngressError("command client is closed")
        if not isinstance(envelope, AuthorizedStateCommand):
            raise QuackCommandIngressError("command must be an AuthorizedStateCommand")
        ingress_slot = envelope.ingress_slot
        if (
            isinstance(ingress_slot, bool)
            or not isinstance(ingress_slot, int)
            or not 1 <= ingress_slot <= MAX_INGRESS_ROWS
        ):
            raise QuackCommandIngressError("ingress_slot is outside relation bounds")
        submission = str(envelope.submission_id or "").strip()
        if not submission or len(submission.encode("utf-8")) > MAX_ID_BYTES:
            raise QuackCommandIngressError("submission_id is outside ingress bounds")
        payload = canonical_json_bytes(envelope.to_dict()).decode("utf-8")
        if len(payload.encode("utf-8")) > MAX_AUTHORIZED_COMMAND_BYTES:
            raise QuackCommandIngressError(
                "serialized AuthorizedStateCommand exceeds ingress bound"
            )
        submitted_at = time.time_ns()
        try:
            self._connection.execute(
                f"INSERT INTO {self._alias}.main.command_inbox VALUES (?, ?, ?, ?, ?)",
                [
                    ingress_slot,
                    submission,
                    envelope.envelope_cid,
                    payload,
                    submitted_at,
                ],
            ).fetchall()
        except Exception as exc:
            raise QuackCommandIngressError("command append was rejected") from exc

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.execute(f"DETACH {self._alias}")
        finally:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> QuackCommandClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class QuackReadClient:
    """Read-only fixed-query client for projected state and apply receipts."""

    def __init__(
        self,
        *,
        duckdb_module: Any,
        extension_path: str | Path,
        endpoint: str,
        token: str,
    ) -> None:
        self._connection = duckdb_module.connect(database=":memory:")
        self._connection.execute(f"LOAD {_sql_literal(str(Path(extension_path).resolve()))}")
        self._endpoint = endpoint
        self._token = token
        self._closed = False

    @property
    def endpoint(self) -> str:
        return str(self._endpoint)

    @staticmethod
    def _rows(result: Any) -> tuple[Mapping[str, Any], ...]:
        columns = [str(item[0]) for item in (result.description or ())]
        return tuple(
            MappingProxyType(dict(zip(columns, row, strict=True))) for row in result.fetchall()
        )

    def _query(self, statement: str) -> tuple[Mapping[str, Any], ...]:
        if self._closed:
            raise QuackCommandFabricStateError("read client is closed")
        result = self._connection.execute(
            "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
            [self._endpoint, statement, self._token],
        )
        return self._rows(result)

    def list_state(self) -> tuple[Mapping[str, Any], ...]:
        return self._query(_STATE_QUERY)

    def list_receipts(self) -> tuple[Mapping[str, Any], ...]:
        return self._query(_RECEIPT_QUERY)

    def list_recent_receipts(self) -> tuple[Mapping[str, Any], ...]:
        """Return the fixed, bounded newest receipt window for remote polling."""

        return self._query(_RECENT_RECEIPT_QUERY)

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> QuackReadClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class QuackPlanR2OwnerGateway:
    """Narrow production owner surface for the admitted Plan-R2 operations.

    Construction is private to :class:`QuackCommandFabric`.  The object exposes
    neither a database location, connection, transaction, nor SQL method.  Its
    production capability is the independently signed Plan-R2 operational
    capability already verified and pinned by the owning fabric.
    """

    INTERFACE = PLAN_R2_OWNER_GATEWAY_INTERFACE

    def __init__(self, fabric: QuackCommandFabric) -> None:
        self.__fabric = fabric

    @property
    def production_capability_cid(self) -> str:
        return self.__fabric.plan_r2_production_capability_cid

    @property
    def command_fabric_qualification_cid(self) -> str:
        capability = self.__fabric._require_plan_r2_capability()  # noqa: SLF001
        return str(capability["quack_command_fabric_qualification_cid"])

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.__fabric._submit_authorized_plan_r2_operation(  # noqa: SLF001
            envelope,
            operation_payload,
        )


class QuackDaemonOwnerGateway:
    """Narrow owner-only surface for the signed 39-operation daemon fabric."""

    INTERFACE = QUACK_DAEMON_OWNER_GATEWAY_INTERFACE

    def __init__(self, fabric: QuackCommandFabric) -> None:
        self.__fabric = fabric

    @property
    def production_capability_cid(self) -> str:
        return self.__fabric.daemon_production_capability_cid

    @property
    def command_fabric_qualification_cid(self) -> str:
        capability = self.__fabric._require_daemon_capability()  # noqa: SLF001
        return str(capability["command_fabric_qualification_cid"])

    def submit_authorized_daemon_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_intent: Mapping[str, Any],
    ) -> Any:
        return self.__fabric._submit_authorized_daemon_operation(  # noqa: SLF001
            envelope,
            operation_intent,
        )


class QuackCommandFabric:
    """Single local owner for ingress, private repository, and read projection."""

    INTERFACE = QUACK_COMMAND_FABRIC_INTERFACE
    SCHEMA = QUACK_COMMAND_FABRIC_SCHEMA

    def __init__(
        self,
        *,
        duckdb_module: Any,
        extension_path: str | Path,
        lock_path: str | Path,
        machine: str,
        ingress_database: str | Path,
        operational_database: str | Path,
        projection_database: str | Path,
        ingress_endpoint: str,
        state_endpoint: str,
        ingress_token: str,
        state_token: str,
        authorization_policy: QuackCommandAuthorizationPolicy,
        plan_r2_operational_capability: Mapping[str, Any] | None = None,
        command_fabric_qualification_cid: str = "",
        trusted_plan_r2_capability_reviewer_dids: Sequence[str] = (),
        trusted_plan_r2_operator_dids: Sequence[str] = (),
        trusted_plan_r2_security_reviewer_dids: Sequence[str] = (),
        daemon_operational_capability: Mapping[str, Any] | None = None,
        trusted_daemon_capability_reviewer_dids: Sequence[str] = (),
        eaaef_bootstrap_operational_capability: Any = None,
        daemon_operation_handler: Any = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._duckdb = duckdb_module
        self.extension_path = Path(extension_path).resolve()
        self.ingress_database = Path(ingress_database).resolve()
        self.operational_database = Path(operational_database).resolve()
        self.projection_database = Path(projection_database).resolve()
        if (
            len(
                {
                    self.ingress_database,
                    self.operational_database,
                    self.projection_database,
                }
            )
            != 3
        ):
            raise QuackCommandFabricStateError(
                "ingress, operational, and projection databases must be distinct"
            )
        self.capability = assess_quack_command_capability(
            duckdb_module=duckdb_module,
            extension_path=self.extension_path,
            lock_path=lock_path,
            machine=machine,
        )
        self.ingress_endpoint = _validated_loopback_endpoint(
            ingress_endpoint, name="ingress_endpoint"
        )
        self.state_endpoint = _validated_loopback_endpoint(state_endpoint, name="state_endpoint")
        if self.ingress_endpoint == self.state_endpoint:
            raise QuackCommandFabricStateError("ingress and state endpoints must be distinct")
        self._ingress_token = _validated_transport_token(ingress_token, name="ingress_token")
        self._state_token = _validated_transport_token(state_token, name="state_token")
        if self._ingress_token == self._state_token:
            raise QuackCommandFabricStateError(
                "ingress and state transport tokens must be distinct"
            )
        if not isinstance(authorization_policy, QuackCommandAuthorizationPolicy):
            raise QuackCommandFabricStateError("an exact command authorization policy is required")
        self.authorization_policy = authorization_policy
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._plan_r2_operational_capability = (
            None
            if plan_r2_operational_capability is None
            else MappingProxyType(dict(plan_r2_operational_capability))
        )
        self._command_fabric_qualification_cid = str(command_fabric_qualification_cid or "")
        self._trusted_plan_r2_capability_reviewer_dids = tuple(
            str(value) for value in trusted_plan_r2_capability_reviewer_dids
        )
        self._trusted_plan_r2_operator_dids = tuple(
            str(value) for value in trusted_plan_r2_operator_dids
        )
        self._trusted_plan_r2_security_reviewer_dids = tuple(
            str(value) for value in trusted_plan_r2_security_reviewer_dids
        )
        self._daemon_operational_capability = (
            None
            if daemon_operational_capability is None
            else MappingProxyType(dict(daemon_operational_capability))
        )
        self._trusted_daemon_capability_reviewer_dids = tuple(
            str(value) for value in trusted_daemon_capability_reviewer_dids
        )
        self._eaaef_bootstrap_operational_capability = (
            eaaef_bootstrap_operational_capability
        )
        # The owner implementation is a source-reviewed built-in, never an
        # arbitrary callback supplied by a supervisor or worker.  Keep the
        # constructor parameter only as a compatibility seam for callers that
        # explicitly pass the exact final built-in type; subclasses and duck
        # types are rejected below.
        self._daemon_operation_handler = daemon_operation_handler
        configured_plan_r2 = self._plan_r2_operational_capability is not None
        trust_sets = (
            self._trusted_plan_r2_capability_reviewer_dids,
            self._trusted_plan_r2_operator_dids,
            self._trusted_plan_r2_security_reviewer_dids,
        )
        if configured_plan_r2 != all(bool(value) for value in trust_sets):
            raise QuackCommandFabricStateError(
                "Plan-R2 owner dispatch requires one signed operational capability "
                "and all three independently pinned reviewer sets"
            )
        if configured_plan_r2 and (
            not self._command_fabric_qualification_cid.startswith("sha256:")
            or len(self._command_fabric_qualification_cid) != 71
        ):
            raise QuackCommandFabricStateError(
                "Plan-R2 owner dispatch requires the exact signed command-fabric "
                "qualification identity"
            )
        if len(set().union(*(set(value) for value in trust_sets))) != sum(
            len(set(value)) for value in trust_sets
        ):
            raise QuackCommandFabricStateError(
                "Plan-R2 capability, operator, and security trust roots must be distinct"
            )
        plan_reviewers = set().union(*(set(value) for value in trust_sets))
        if configured_plan_r2 and plan_reviewers.intersection(
            set(authorization_policy.trusted_approver_dids)
            | set(authorization_policy.authorized_principal_dids)
            | {authorization_policy.owner_principal_did}
        ):
            raise QuackCommandFabricStateError(
                "Plan-R2 reviewers must be independent of owner, command "
                "principal, and command approver identities"
            )
        configured_daemon = self._daemon_operational_capability is not None
        configured_eaaef = self._eaaef_bootstrap_operational_capability is not None
        if configured_daemon and configured_eaaef:
            raise QuackCommandFabricStateError(
                "generic 39-operation and EAAEF 31-operation daemon capabilities "
                "are mutually exclusive"
            )
        if configured_daemon != bool(self._trusted_daemon_capability_reviewer_dids):
            raise QuackCommandFabricStateError(
                "daemon owner dispatch requires one signed operational capability "
                "and an independently pinned reviewer set"
            )
        if self._daemon_operation_handler is not None and not (
            configured_daemon or configured_eaaef
        ):
            raise QuackCommandFabricStateError(
                "daemon owner handler cannot exist without its signed capability"
            )
        if configured_daemon:
            from .quack_daemon_gateway import (
                QuackDaemonCanonicalOwnerOperationHandler,
            )

            if self._daemon_operation_handler is None:
                self._daemon_operation_handler = (
                    QuackDaemonCanonicalOwnerOperationHandler()
                )
            elif type(self._daemon_operation_handler) is not (
                QuackDaemonCanonicalOwnerOperationHandler
            ):
                raise QuackCommandFabricStateError(
                    "arbitrary daemon operation handler injection is forbidden"
                )
        if configured_eaaef:
            from ..validation.eaaef_bootstrap_gateway_launch import (
                VerifiedEAAEFBootstrapOperationalCapability,
            )
            from .eaaef_borrowed_transaction import (
                EAAEFBootstrapBorrowedTransactionOperationHandler,
            )

            capability = self._eaaef_bootstrap_operational_capability
            if type(capability) is not VerifiedEAAEFBootstrapOperationalCapability:
                raise QuackCommandFabricStateError(
                    "EAAEF daemon dispatch requires the verifier-owned typed "
                    "operational capability"
                )
            if self._trusted_daemon_capability_reviewer_dids:
                raise QuackCommandFabricStateError(
                    "EAAEF typed capability cannot reuse generic daemon trust roots"
                )
            expected_handler = EAAEFBootstrapBorrowedTransactionOperationHandler
            if self._daemon_operation_handler is None:
                self._daemon_operation_handler = expected_handler(
                    board_namespace=str(capability["board_namespace"]),
                    shard_id=str(capability["shard_id"]),
                    owner_principal_did=str(capability["owner_principal_did"]),
                    command_principal_did=str(capability["command_principal_did"]),
                    owner_session_id=str(capability["owner_session_id"]),
                    owner_generation=int(capability["owner_generation"]),
                    fence_epoch=int(capability["fence_epoch"]),
                    gateway_binding_cid=str(capability["gateway_binding_cid"]),
                    control_plane_schema_version=str(
                        capability["control_plane_schema_version"]
                    ),
                    state_schema_revision=str(capability["state_schema_revision"]),
                )
            elif type(self._daemon_operation_handler) is not expected_handler:
                raise QuackCommandFabricStateError(
                    "arbitrary EAAEF daemon operation handler injection is forbidden"
                )
        if (configured_daemon or configured_eaaef) and (
            not self._command_fabric_qualification_cid.startswith("sha256:")
            or len(self._command_fabric_qualification_cid) != 71
        ):
            raise QuackCommandFabricStateError(
                "daemon owner dispatch requires the exact signed command-fabric "
                "qualification identity"
            )
        expected_handler_interface = QUACK_DAEMON_CANONICAL_HANDLER_INTERFACE
        if configured_eaaef:
            from .eaaef_borrowed_transaction import (
                EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
            )

            expected_handler_interface = EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        if self._daemon_operation_handler is not None and (
            getattr(self._daemon_operation_handler, "INTERFACE", "")
            != expected_handler_interface
        ):
            raise QuackCommandFabricStateError(
                "daemon operation handler does not implement the closed owner interface"
            )
        if self._daemon_operation_handler is not None:
            missing = [
                name
                for name in ("apply_authorized_daemon_operation",)
                if not callable(getattr(self._daemon_operation_handler, name, None))
            ]
            forbidden = [
                name
                for name in (
                    "database_path",
                    "connection",
                    "execute",
                    "execute_sql",
                    "portal",
                    "local_sidecar",
                )
                if hasattr(self._daemon_operation_handler, name)
            ]
            if missing or forbidden:
                raise QuackCommandFabricStateError(
                    "daemon operation handler is incomplete or exposes forbidden authority"
                )
            daemon_reviewers = set(self._trusted_daemon_capability_reviewer_dids)
            if daemon_reviewers.intersection(
                set(authorization_policy.trusted_approver_dids)
                | set(authorization_policy.authorized_principal_dids)
                | {authorization_policy.owner_principal_did}
            ):
                raise QuackCommandFabricStateError(
                    "daemon capability reviewers must be independent of owner, "
                    "command principals, and command approvers"
                )
        self._plan_r2_after_commit_hook: Callable[[Mapping[str, Any]], None] = lambda _result: None
        self._ingress_server: _ExactQuackEndpoint | None = None
        self._state_server: _ExactQuackEndpoint | None = None
        self._repository: QuackStateRepository | None = None
        self.started = False

    @property
    def plan_r2_production_capability_cid(self) -> str:
        capability = self._require_plan_r2_capability()
        return str(capability["capability_cid"])

    def plan_r2_owner_gateway(self) -> QuackPlanR2OwnerGateway:
        """Return the narrow owner gateway only after production admission."""

        self._require_plan_r2_capability()
        return QuackPlanR2OwnerGateway(self)

    @property
    def daemon_production_capability_cid(self) -> str:
        capability = self._require_daemon_capability()
        return str(capability["capability_cid"])

    def daemon_owner_gateway(self) -> QuackDaemonOwnerGateway:
        """Return the narrow daemon owner surface only after signed admission."""

        self._require_daemon_capability()
        return QuackDaemonOwnerGateway(self)

    def _require_daemon_capability(self) -> Mapping[str, Any]:
        verified = self._verify_daemon_capability_record()
        if self._daemon_operation_handler is None:
            raise QuackCommandFabricCapabilityError(
                "canonical_39_operation_owner_handler_unqualified"
            )
        evidence = self._daemon_operation_handler.evidence()
        if self._eaaef_bootstrap_operational_capability is not None:
            if (
                evidence.get("handler_source_evidence_cid")
                != verified.get(
                    "borrowed_transaction_handler_source_evidence_cid"
                )
                or evidence.get("operation_count") != 31
                or evidence.get("production_admitted") is not False
            ):
                raise QuackCommandFabricCapabilityError(
                    "EAAEF 31-operation owner handler evidence is invalid"
                )
            return MappingProxyType(dict(verified))
        if (
            evidence.get("all_operations_recognized") is not True
            or evidence.get("production_admitted") is not False
        ):
            raise QuackCommandFabricCapabilityError(
                "canonical daemon owner handler evidence is invalid"
            )
        del verified
        raise QuackCommandFabricCapabilityError(
            "canonical_39_operation_owner_handler_unqualified: all operations "
            "are recognized, but the handler reports explicit per-operation "
            "no-go dispositions and cannot support a broad production gateway"
        )

    def _require_daemon_operation_capability(
        self, operation: str
    ) -> tuple[Mapping[str, Any], Any]:
        """Admit one exact built-in owner operation, never the broad gateway."""

        verified = self._verify_daemon_capability_record()
        handler = self._daemon_operation_handler
        if handler is None:
            raise QuackCommandFabricCapabilityError(
                "canonical_39_operation_owner_handler_unqualified"
            )
        try:
            handler.require_operation(str(operation))
        except Exception as exc:
            # Preserve the stable per-operation no-go vocabulary in the public
            # error without exposing a database or callback surface.
            raise QuackCommandFabricCapabilityError(str(exc)) from exc
        return MappingProxyType(dict(verified)), handler

    def _verify_daemon_capability_record(self) -> Mapping[str, Any]:
        if self._eaaef_bootstrap_operational_capability is not None:
            return self._verify_eaaef_bootstrap_capability_record()
        capability = self._daemon_operational_capability
        if capability is None:
            raise QuackCommandFabricCapabilityError(
                "typed_quack_daemon_owner_dispatch_unavailable"
            )
        from .quack_daemon_gateway import (
            QuackDaemonGatewayError,
            verify_quack_daemon_operational_capability,
        )

        try:
            verified = verify_quack_daemon_operational_capability(
                capability,
                trusted_reviewer_dids=self._trusted_daemon_capability_reviewer_dids,
                now_ms=int(self._clock_ms()),
            )
        except QuackDaemonGatewayError as exc:
            raise QuackCommandFabricCapabilityError(
                "typed_quack_daemon_owner_dispatch_unavailable"
            ) from exc
        policy = self.authorization_policy
        comparisons = {
            "board_namespace": policy.board_namespace,
            "shard_id": policy.shard_id,
            "store_id": policy.store_id,
            "owner_principal_did": policy.owner_principal_did,
            "owner_generation": policy.owner_generation,
            "fence_epoch": policy.fence_epoch,
            "authorization_policy_cid": policy.policy_cid,
            "command_endpoint": self.ingress_endpoint,
            "state_endpoint": self.state_endpoint,
            "command_fabric_qualification_cid": self._command_fabric_qualification_cid,
        }
        mismatched = sorted(
            name for name, expected in comparisons.items() if verified.get(name) != expected
        )
        if mismatched:
            raise QuackCommandFabricCapabilityError(
                "daemon production capability differs from command owner: "
                + ", ".join(mismatched)
            )
        return MappingProxyType(dict(verified))

    def _verify_eaaef_bootstrap_capability_record(self) -> Mapping[str, Any]:
        """Rejoin the verifier-owned @2 capability to this exact owner."""

        from ..validation.eaaef_bootstrap_gateway_launch import (
            VerifiedEAAEFBootstrapOperationalCapability,
            eaaef_bootstrap_gateway_binding_cid,
        )
        from .control_plane_repository import QUACK_STATE_REPOSITORY_INTERFACE
        from .eaaef_bootstrap_daemon_gateway import (
            EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE,
            EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
        )
        from .eaaef_borrowed_transaction import (
            EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
            eaaef_bootstrap_handler_source_evidence,
        )
        from .eaaef_operational_schema import (
            EAAEF_OPERATIONAL_PROFILE_ID,
            eaaef_operation_vocabulary_cid,
        )

        capability = self._eaaef_bootstrap_operational_capability
        if type(capability) is not VerifiedEAAEFBootstrapOperationalCapability:
            raise QuackCommandFabricCapabilityError(
                "typed_eaaef_bootstrap_daemon_owner_dispatch_unavailable"
            )
        policy = self.authorization_policy
        handler_evidence = eaaef_bootstrap_handler_source_evidence(
            board_namespace=policy.board_namespace,
            shard_id=policy.shard_id,
        )
        try:
            binding_cid = eaaef_bootstrap_gateway_binding_cid(capability)
        except Exception as exc:
            raise QuackCommandFabricCapabilityError(
                "typed_eaaef_bootstrap_daemon_owner_dispatch_unavailable"
            ) from exc
        comparisons = {
            "board_namespace": policy.board_namespace,
            "shard_id": policy.shard_id,
            "store_id": policy.store_id,
            "owner_principal_did": policy.owner_principal_did,
            "owner_generation": policy.owner_generation,
            "fence_epoch": policy.fence_epoch,
            "authorization_policy_cid": policy.policy_cid,
            "command_endpoint": self.ingress_endpoint,
            "state_endpoint": self.state_endpoint,
            "command_fabric_qualification_cid": (
                self._command_fabric_qualification_cid
            ),
            "gateway_binding_cid": binding_cid,
            "control_plane_schema_version": QUACK_STATE_REPOSITORY_INTERFACE,
            "state_schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
            "operational_profile_id": EAAEF_OPERATIONAL_PROFILE_ID,
            "schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
            "gateway_interface": EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE,
            "borrowed_transaction_handler_interface": (
                EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
            ),
            "borrowed_transaction_handler_source_evidence_cid": (
                handler_evidence["handler_source_evidence_cid"]
            ),
            "operation_vocabulary_cid": eaaef_operation_vocabulary_cid(
                EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
            ),
        }
        mismatched = sorted(
            name
            for name, expected in comparisons.items()
            if capability.get(name) != expected
        )
        expected_principals = frozenset(
            {str(capability.get("command_principal_did") or "")}
        )
        service = capability.get("command_authorization_service")
        expected_approvers = (
            frozenset({str(service.get("approver_principal_did") or "")})
            if isinstance(service, Mapping)
            else frozenset()
        )
        role_dids = (
            set(policy.authorized_principal_dids)
            | set(policy.trusted_approver_dids)
            | {policy.owner_principal_did}
        )
        reviewer = str(capability.get("reviewer_did") or "")
        if (
            mismatched
            or capability.get("operations")
            != sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)
            or policy.authorized_principal_dids != expected_principals
            or policy.trusted_approver_dids != expected_approvers
            or reviewer in role_dids
        ):
            detail = ", ".join(mismatched) or "authority_or_operation_set"
            raise QuackCommandFabricCapabilityError(
                "EAAEF bootstrap capability differs from command owner: " + detail
            )
        return MappingProxyType(dict(capability))

    def _verify_daemon_operation_authorization(
        self,
        envelope: AuthorizedStateCommand,
        operation_intent: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> Mapping[str, Any]:
        """Verify exactly one mutually exclusive daemon protocol submission."""

        if self._eaaef_bootstrap_operational_capability is not None:
            from ..validation.eaaef_bootstrap_gateway_launch import (
                EAAEFBootstrapGatewayLaunchError,
                verify_eaaef_bootstrap_operation_submission,
            )

            verified = self._verify_eaaef_bootstrap_capability_record()
            # Keep the verifier-owned typed object: copying it to a plain
            # Mapping would erase the admission token and must fail closed.
            capability = self._eaaef_bootstrap_operational_capability
            try:
                authorization = verify_eaaef_bootstrap_operation_submission(
                    envelope,
                    operation_intent,
                    verified_capability=capability,
                    authorization_policy=self.authorization_policy,
                    now_ms=now_ms,
                )
            except EAAEFBootstrapGatewayLaunchError as exc:
                raise QuackCommandAuthorizationError(
                    "EAAEF daemon operation capability/envelope verification failed"
                ) from exc
            if (
                authorization.get("operation") not in verified.get("operations", ())
                or authorization.get("scope_id") != envelope.scope_id
                or authorization.get("lease_id") != envelope.lease_id
                or authorization.get("effect") != envelope.effect
            ):
                raise QuackCommandAuthorizationError(
                    "EAAEF daemon operation authorization projection diverged"
                )
            return MappingProxyType(dict(authorization))

        from .quack_daemon_gateway import (
            QuackDaemonGatewayCapability,
            QuackDaemonGatewayError,
            verify_quack_daemon_operation_submission,
        )

        capability_record = self._verify_daemon_capability_record()
        capability = QuackDaemonGatewayCapability.from_verified_operational_capability(
            capability_record
        )
        try:
            authorization = verify_quack_daemon_operation_submission(
                envelope,
                operation_intent,
                capability=capability,
                authorization_policy=self.authorization_policy,
                now_ms=now_ms,
            )
        except QuackDaemonGatewayError as exc:
            raise QuackCommandAuthorizationError(
                "daemon operation capability/envelope verification failed"
            ) from exc
        return MappingProxyType(dict(authorization))

    def _require_plan_r2_capability(self) -> Mapping[str, Any]:
        capability = self._plan_r2_operational_capability
        if capability is None:
            raise QuackCommandFabricCapabilityError(
                "typed_quack_plan_transition_atomic_owner_operation_unavailable"
            )
        from ..planning.external_agent_plan_r2 import (
            ExternalAgentPlanR2Error,
            verify_plan_r2_operational_capability,
        )

        try:
            verified = verify_plan_r2_operational_capability(
                capability,
                trusted_reviewer_dids=(self._trusted_plan_r2_capability_reviewer_dids),
                now_ms=int(self._clock_ms()),
            )
        except ExternalAgentPlanR2Error as exc:
            raise QuackCommandFabricCapabilityError(
                "typed_quack_plan_transition_atomic_owner_operation_unavailable"
            ) from exc
        policy = self.authorization_policy
        comparisons = {
            "owner_principal_did": policy.owner_principal_did,
            "shard_id": policy.shard_id,
            "owner_generation": policy.owner_generation,
            "fence": policy.fence_epoch,
            "quack_command_fabric_qualification_cid": (self._command_fabric_qualification_cid),
        }
        mismatched = sorted(
            field for field, expected in comparisons.items() if verified.get(field) != expected
        )
        if mismatched:
            raise QuackCommandFabricCapabilityError(
                "Plan-R2 production capability differs from command owner: " + ", ".join(mismatched)
            )
        return MappingProxyType(dict(verified))

    @property
    def repository(self) -> QuackStateRepository:
        if self._repository is None:
            raise QuackCommandFabricStateError("repository owner is not started")
        return self._repository

    def _install_ingress(self) -> None:
        self.ingress_database.parent.mkdir(parents=True, exist_ok=True)
        connection = self._duckdb.connect(str(self.ingress_database))
        try:
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS command_inbox (
                    ingress_slot INTEGER PRIMARY KEY
                        CHECK (ingress_slot BETWEEN 1 AND {MAX_INGRESS_ROWS}),
                    submission_id VARCHAR UNIQUE NOT NULL
                        CHECK (octet_length(encode(submission_id)) BETWEEN 1 AND {MAX_ID_BYTES}),
                    envelope_cid VARCHAR UNIQUE NOT NULL
                        CHECK (octet_length(encode(envelope_cid))
                               BETWEEN 8 AND {MAX_ID_BYTES}),
                    envelope_json VARCHAR NOT NULL
                        CHECK (json_valid(envelope_json)
                               AND octet_length(encode(envelope_json)) BETWEEN 2 AND {MAX_AUTHORIZED_COMMAND_BYTES}),
                    submitted_at UBIGINT NOT NULL
                )
                """)
            rows = connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE NOT internal ORDER BY table_name"
            ).fetchall()
            if rows != [("command_inbox",)]:
                raise QuackCommandFabricStateError(
                    "ingress database must contain exactly command_inbox"
                )
            columns = [
                str(row[1])
                for row in connection.execute("PRAGMA table_info('command_inbox')").fetchall()
            ]
            if columns != [
                "ingress_slot",
                "submission_id",
                "envelope_cid",
                "envelope_json",
                "submitted_at",
            ]:
                raise QuackCommandFabricStateError(
                    "ingress database schema is not AuthorizedStateCommand@1"
                )
        finally:
            connection.close()

    def _install_projection(self) -> None:
        self.projection_database.parent.mkdir(parents=True, exist_ok=True)
        connection = self._duckdb.connect(str(self.projection_database))
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS state_rows (
                    task_cid VARCHAR PRIMARY KEY,
                    status VARCHAR NOT NULL,
                    revision BIGINT NOT NULL,
                    body_json VARCHAR NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS apply_receipts (
                    submission_id VARCHAR PRIMARY KEY,
                    envelope_cid VARCHAR NOT NULL,
                    request_id VARCHAR NOT NULL,
                    principal_did VARCHAR NOT NULL,
                    approver_did VARCHAR NOT NULL,
                    authority_ref_cid VARCHAR NOT NULL,
                    lease_id VARCHAR NOT NULL,
                    one_use_nonce VARCHAR NOT NULL,
                    command_id VARCHAR NOT NULL,
                    idempotency_key VARCHAR NOT NULL,
                    outcome VARCHAR NOT NULL,
                    changed BOOLEAN NOT NULL,
                    revision BIGINT NOT NULL,
                    generation BIGINT NOT NULL,
                    fence_epoch BIGINT NOT NULL,
                    result_json VARCHAR NOT NULL,
                    error VARCHAR NOT NULL,
                    submitted_at UBIGINT NOT NULL,
                    applied_at UBIGINT NOT NULL
                )
                """)
            relations = connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE NOT internal ORDER BY table_name"
            ).fetchall()
            if relations != [("apply_receipts",), ("state_rows",)]:
                raise QuackCommandFabricStateError(
                    "projection database contains unexpected mutable relations"
                )
            receipt_columns = [
                str(row[1])
                for row in connection.execute("PRAGMA table_info('apply_receipts')").fetchall()
            ]
            if receipt_columns != [
                "submission_id",
                "envelope_cid",
                "request_id",
                "principal_did",
                "approver_did",
                "authority_ref_cid",
                "lease_id",
                "one_use_nonce",
                "command_id",
                "idempotency_key",
                "outcome",
                "changed",
                "revision",
                "generation",
                "fence_epoch",
                "result_json",
                "error",
                "submitted_at",
                "applied_at",
            ]:
                raise QuackCommandFabricStateError(
                    "projection receipt schema is not AuthorizedStateCommand@1"
                )
        finally:
            connection.close()

    def start(self) -> None:
        if self.started:
            raise QuackCommandFabricStateError("fabric is already started")
        self.capability.require()
        if self._plan_r2_operational_capability is not None:
            self._require_plan_r2_capability()
        if (
            self._daemon_operational_capability is not None
            or self._eaaef_bootstrap_operational_capability is not None
        ):
            self._verify_daemon_capability_record()
        if not self.operational_database.is_file():
            raise QuackCommandFabricStateError(
                "private operational database must be provisioned before start"
            )
        self._install_ingress()
        self._install_projection()
        ingress = _ExactQuackEndpoint(
            duckdb_module=self._duckdb,
            database_path=self.ingress_database,
            extension_path=self.extension_path,
            uri=self.ingress_endpoint,
            token=self._ingress_token,
            authorization_name="command_inbox_exact_authorization",
            allowed_sql=(
                _CATALOG_SCHEMAS_SQL,
                _CATALOG_RELATIONS_SQL,
                _INGRESS_REMOTE_INSERT_SQL,
            ),
        )
        state = _ExactQuackEndpoint(
            duckdb_module=self._duckdb,
            database_path=self.projection_database,
            extension_path=self.extension_path,
            uri=self.state_endpoint,
            token=self._state_token,
            authorization_name="state_projection_exact_authorization",
            allowed_sql=(_STATE_QUERY, _RECEIPT_QUERY, _RECENT_RECEIPT_QUERY),
        )
        repository: QuackStateRepository | None = None
        try:
            ingress.start()
            state.start()
            repository = QuackStateRepository(
                "quack:127.0.0.1:1",
                owner_id=self.authorization_policy.owner_principal_did,
                store_id=self.authorization_policy.store_id,
                connection_factory=lambda _endpoint: self._duckdb.connect(
                    str(self.operational_database)
                ),
                seed_generation=False,
            )
            repository.attach()
            generation = repository.load_generation()
            if (
                generation.store_id != self.authorization_policy.store_id
                or generation.generation != self.authorization_policy.owner_generation
                or generation.fence_epoch != self.authorization_policy.fence_epoch
            ):
                raise QuackCommandFabricStateError(
                    "private owner generation does not match authorization policy"
                )
        except Exception:
            try:
                if repository is not None:
                    repository.close()
            finally:
                try:
                    state.stop()
                finally:
                    ingress.stop()
            raise
        self._ingress_server = ingress
        self._state_server = state
        self._repository = repository
        self.started = True
        self._rebuild_projection()

    def command_client(self, *, alias: str) -> QuackCommandClient:
        if not self.started:
            raise QuackCommandFabricStateError("fabric is not started")
        return QuackCommandClient(
            duckdb_module=self._duckdb,
            extension_path=self.extension_path,
            endpoint=self.ingress_endpoint,
            token=self._ingress_token,
            alias=alias,
        )

    def read_client(self) -> QuackReadClient:
        if not self.started:
            raise QuackCommandFabricStateError("fabric is not started")
        return QuackReadClient(
            duckdb_module=self._duckdb,
            extension_path=self.extension_path,
            endpoint=self.state_endpoint,
            token=self._state_token,
        )

    def _project_state(self) -> None:
        page = self.repository.list_tasks(cursor=0, limit=500)
        if not page.exhausted:
            raise QuackCommandFabricStateError(
                "state projection exceeded the compatibility slice bound"
            )
        connection = self._state_server.connection if self._state_server else None
        if connection is None:
            raise QuackCommandFabricStateError("state projection owner is absent")
        connection.execute("DELETE FROM state_rows")
        for row in page.items:
            connection.execute(
                "INSERT INTO state_rows VALUES (?, ?, ?, ?)",
                [
                    str(row["task_cid"]),
                    str(row["status"]),
                    int(row["revision"]),
                    str(row.get("body_json") or "{}"),
                ],
            )

    def _private_receipts(self) -> tuple[Mapping[str, Any], ...]:
        """Read authoritative receipts from private committed domain events."""

        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        try:
            transaction.begin()
            receipts = transaction.list_authorized_command_receipts(
                limit=MAX_PROJECTED_RECEIPTS
            )
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        for payload in receipts:
            if payload.get("schema") != _PRIVATE_RECEIPT_SCHEMA:
                raise QuackCommandFabricStateError(
                    "private authorized-command receipt schema is invalid"
                )
        return receipts

    def _private_receipt(self, submission_id: str) -> Mapping[str, Any] | None:
        event_id = _receipt_event_id(submission_id)
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        try:
            transaction.begin()
            receipt = transaction.lookup_authorized_command_receipt(receipt_event_id=event_id)
            transaction.commit()
            return receipt
        except Exception:
            transaction.rollback()
            raise

    @staticmethod
    def _projection_receipt_values(receipt: Mapping[str, Any]) -> list[Any]:
        return [
            receipt.get("submission_id", ""),
            receipt.get("envelope_cid", ""),
            receipt.get("request_id", ""),
            receipt.get("principal_did", ""),
            receipt.get("approver_did", ""),
            receipt.get("authority_ref_cid", ""),
            receipt.get("lease_id", ""),
            receipt.get("one_use_nonce", ""),
            receipt.get("command_id", ""),
            receipt.get("idempotency_key", ""),
            receipt.get("outcome", CommandOutcome.REJECTED.value),
            bool(receipt.get("changed", False)),
            int(receipt.get("revision", 0)),
            int(receipt.get("generation", 0)),
            int(receipt.get("fence_epoch", 0)),
            receipt.get("result_json", "{}"),
            receipt.get("error", ""),
            int(receipt.get("submitted_at", 0)),
            int(receipt.get("applied_at", 0)),
        ]

    def _rebuild_projection(self) -> None:
        """Rebuild the disposable read model from private authority state."""

        connection = self._state_server.connection if self._state_server else None
        if connection is None:
            raise QuackCommandFabricStateError("state projection owner is absent")
        self._project_state()
        connection.execute("DELETE FROM apply_receipts")
        for receipt in self._private_receipts():
            connection.execute(
                "INSERT INTO apply_receipts VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._projection_receipt_values(receipt),
            )

    def _persist_rejected_receipt(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        """Durably audit a non-admitted submission without consuming authority."""

        submission_id = str(receipt.get("submission_id") or "")
        event_id = _receipt_event_id(submission_id)
        prior = self._private_receipt(submission_id)
        if prior is not None:
            if prior.get("envelope_cid") != receipt.get("envelope_cid"):
                raise QuackCommandFabricStateError(
                    "submission identity is bound to a different envelope"
                )
            return prior
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        try:
            transaction.begin()
            transaction.record_authorized_command_receipt(
                receipt_event_id=event_id,
                stream_id=f"authorized-command:{self.authorization_policy.shard_id}",
                task_cid=str(receipt.get("scope_id") or ""),
                session_id="",
                receipt=receipt,
            )
            transaction.commit()
        except Exception:
            transaction.rollback()
            recovered = self._private_receipt(submission_id)
            if recovered is not None and recovered.get("envelope_cid") == receipt.get(
                "envelope_cid"
            ):
                return recovered
            raise
        return MappingProxyType(dict(receipt))

    def _quarantine_divergent_ingress_replay(
        self,
        *,
        ingress_slot: int,
        submission_id: str,
        durable_envelope_cid: str,
        divergent_envelope_cid: str,
    ) -> Mapping[str, Any]:
        """Durably reject one poison row without reopening prior authority."""

        body = {
            "schema": _DIVERGENT_INGRESS_QUARANTINE_SCHEMA,
            "disposition": "quarantined",
            "original_submission_id": str(submission_id),
            "ingress_slot": int(ingress_slot),
            "durable_envelope_cid": str(durable_envelope_cid),
            "divergent_envelope_cid": str(divergent_envelope_cid),
            "authority_reopened": False,
        }
        quarantine_digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        quarantine_event_id = (
            f"authorized-command-ingress-quarantine:sha256:{quarantine_digest}"
        )
        quarantine = {
            **body,
            "quarantine_event_id": quarantine_event_id,
            "submission_id": str(submission_id),
            "envelope_cid": str(divergent_envelope_cid),
            "outcome": CommandOutcome.REJECTED.value,
            "changed": False,
            "result_json": canonical_json_bytes(body).decode("utf-8"),
            "error": (
                "QuackCommandFabricStateError: "
                "ingress_divergent_durable_submission_quarantined"
            ),
        }
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        try:
            transaction.begin()
            prior = transaction.lookup_authorized_command_ingress_quarantine(
                quarantine_event_id=quarantine_event_id
            )
            if prior is None:
                transaction.record_authorized_command_ingress_quarantine(
                    quarantine_event_id=quarantine_event_id,
                    stream_id=(
                        "authorized-command-ingress-quarantine:"
                        f"{self.authorization_policy.shard_id}"
                    ),
                    quarantine=quarantine,
                )
                transaction.commit()
                return MappingProxyType(dict(quarantine))
            transaction.commit()
            if dict(prior) != quarantine:
                raise QuackCommandFabricStateError(
                    "authorized command ingress quarantine replay diverged"
                )
            return MappingProxyType(dict(prior))
        except Exception:
            if transaction.active:
                transaction.rollback()
            raise

    def apply_pending(self, *, limit: int = 100) -> tuple[Mapping[str, Any], ...]:
        if not self.started or self._ingress_server is None or self._state_server is None:
            raise QuackCommandFabricStateError("fabric is not started")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise QuackCommandFabricStateError("apply limit must be between 1 and 100")
        ingress = self._ingress_server.connection
        projection = self._state_server.connection
        if ingress is None or projection is None:
            raise QuackCommandFabricStateError("fabric owner connections are absent")
        rows = ingress.execute(
            """
            SELECT ingress_slot, submission_id, envelope_cid, envelope_json,
                   submitted_at
            FROM command_inbox
            ORDER BY submitted_at, submission_id
            LIMIT ?
            """,
            [MAX_INGRESS_ROWS],
        ).fetchall()
        existing = {
            str(receipt.get("submission_id") or ""): str(receipt.get("envelope_cid") or "")
            for receipt in self._private_receipts()
        }
        pending: list[tuple[Any, ...]] = []
        durable_replay_rows: list[tuple[int, str, str]] = []
        quarantined_rows: list[tuple[int, str, str]] = []
        quarantined_receipts: list[Mapping[str, Any]] = []
        for row in rows:
            ingress_slot, submission_id, envelope_cid, _json, _submitted_at = row
            prior = existing.get(str(submission_id))
            if prior is None:
                pending.append(row)
                continue
            # An exact transport replay has an already durable receipt.  Keep
            # the ingress row until the disposable projection has been rebuilt
            # from private authority state.  This is the recovery path for a
            # commit followed by projection-publication failure; it deliberately
            # does not re-open freshness, lease, fence, or nonce authorization.
            # A divergent reuse is durably quarantined under a distinct audit
            # identity.  It never re-opens the original submission authority
            # and cannot starve later valid rows across owner restarts.
            if prior != str(envelope_cid):
                quarantined_receipts.append(
                    self._quarantine_divergent_ingress_replay(
                        ingress_slot=int(ingress_slot),
                        submission_id=str(submission_id),
                        durable_envelope_cid=prior,
                        divergent_envelope_cid=str(envelope_cid),
                    )
                )
                quarantined_rows.append(
                    (int(ingress_slot), str(submission_id), str(envelope_cid))
                )
                continue
            durable_replay_rows.append(
                (int(ingress_slot), str(submission_id), str(envelope_cid))
            )
        pending = pending[:limit]
        emitted: list[Mapping[str, Any]] = [
            MappingProxyType(dict(receipt)) for receipt in quarantined_receipts
        ]
        applied_rows: list[tuple[int, str, str]] = []
        for ingress_slot, submission_id, content_id, command_json, submitted_at in pending:
            envelope_cid = str(content_id)
            request_id = ""
            principal_did = ""
            approver_did = ""
            authority_ref_cid = ""
            lease_id = ""
            one_use_nonce = ""
            scope_id = ""
            effect = ""
            command_id = ""
            idempotency_key = ""
            outcome = CommandOutcome.REJECTED.value
            changed = False
            revision = 0
            generation = 0
            fence_epoch = 0
            result_json = "{}"
            error = ""
            transaction = None
            try:
                payload = json.loads(str(command_json))
                envelope = AuthorizedStateCommand.from_dict(payload)
                if envelope.envelope_cid != envelope_cid:
                    raise QuackCommandAuthorizationError(
                        "authorized envelope content identity mismatch"
                    )
                if envelope.submission_id != str(submission_id) or envelope.ingress_slot != int(
                    ingress_slot
                ):
                    raise QuackCommandAuthorizationError(
                        "authorized envelope does not bind its ingress identity"
                    )
                request_id = envelope.request_id
                principal_did = envelope.principal_did
                approver_did = envelope.approver_did
                authority_ref_cid = envelope.authority_ref_cid
                lease_id = envelope.lease_id
                one_use_nonce = envelope.one_use_nonce
                scope_id = envelope.scope_id
                effect = envelope.effect
                now_ms = int(self._clock_ms())
                transaction = self.repository.transaction(
                    expected_generation=self.repository.load_generation()
                )
                transaction.begin()
                prior = transaction.lookup_authorized_command_receipt(
                    receipt_event_id=_receipt_event_id(str(submission_id))
                )
                if prior is not None:
                    if prior.get("envelope_cid") != envelope_cid:
                        transaction.commit()
                        quarantine = self._quarantine_divergent_ingress_replay(
                            ingress_slot=int(ingress_slot),
                            submission_id=str(submission_id),
                            durable_envelope_cid=str(prior.get("envelope_cid") or ""),
                            divergent_envelope_cid=envelope_cid,
                        )
                        emitted.append(MappingProxyType(dict(quarantine)))
                        quarantined_rows.append(
                            (int(ingress_slot), str(submission_id), envelope_cid)
                        )
                        continue
                    transaction.commit()
                    emitted.append(MappingProxyType(dict(prior)))
                    durable_replay_rows.append(
                        (int(ingress_slot), str(submission_id), envelope_cid)
                    )
                    continue
                verify_authorized_state_command(
                    envelope,
                    policy=self.authorization_policy,
                    now_ms=now_ms,
                )
                command = envelope.command
                command_id = command.command_id
                idempotency_key = command.idempotency_key
                live_lease = transaction.assert_live_authorized_command_lease(
                    lease_id=lease_id,
                    scope_id=scope_id,
                    principal_did=principal_did,
                    effect=effect,
                    command_kind=command.command_kind,
                    fence_epoch=command.fence_epoch,
                    now_ms=now_ms,
                )
                transaction.consume_authorized_command_replay_claims(
                    request_id=request_id,
                    one_use_nonce=one_use_nonce,
                    scope_id=scope_id,
                    effect=effect,
                )
                daemon_operation = ""
                daemon_intent_cid = ""
                daemon_handler: Any = None
                daemon_arguments: dict[str, Any] = {}
                if "daemon_operation" in command.parameters:
                    from .quack_daemon_gateway import (
                        quack_daemon_operation_intent_from_envelope,
                    )

                    daemon_intent = quack_daemon_operation_intent_from_envelope(
                        envelope
                    )
                    daemon_authorization = self._verify_daemon_operation_authorization(
                        envelope,
                        daemon_intent,
                        now_ms=now_ms,
                    )
                    if int(live_lease.get("fencing_token") or 0) != int(
                        daemon_authorization["fencing_token"]
                    ):
                        raise QuackCommandAuthorizationError(
                            "daemon operation task fencing token is stale"
                        )
                    daemon_operation = str(daemon_authorization["operation"])
                    daemon_intent_cid = str(daemon_authorization["intent_cid"])
                    daemon_arguments = dict(daemon_authorization["arguments"])
                    _record, daemon_handler = (
                        self._require_daemon_operation_capability(daemon_operation)
                    )

                if daemon_handler is None and (
                    self._plan_r2_operational_capability is not None
                    or self._daemon_operational_capability is not None
                    or self._eaaef_bootstrap_operational_capability is not None
                ):
                    raise QuackCommandAuthorizationError(
                        "specialized owner command fabric rejects bare generic "
                        "StateCommand fallback"
                    )
                if daemon_handler is None:
                    apply_command = self.repository.client.apply_command_in_transaction
                else:

                    def apply_command(
                        active_transaction: Any,
                        active_command: StateCommand,
                        _live: Any,
                        _handler: Any = daemon_handler,
                        _operation: str = daemon_operation,
                        _arguments: Mapping[str, Any] = daemon_arguments,
                        _lease: Mapping[str, Any] = live_lease,
                        _intent_cid: str = daemon_intent_cid,
                    ) -> Mapping[str, Any]:
                        owner_result = _handler.apply_authorized_daemon_operation(
                            operation=_operation,
                            arguments=dict(_arguments),
                            transaction=active_transaction,
                            command=active_command,
                            lease=dict(_lease),
                        )
                        if not isinstance(owner_result, Mapping) or set(owner_result) != {
                            "value"
                        }:
                            raise QuackCommandFabricStateError(
                                "canonical daemon handler returned a non-closed result"
                            )
                        body = {
                            "daemon_operation": _operation,
                            "intent_cid": _intent_cid,
                            "value": owner_result["value"],
                        }
                        canonical_json_bytes(body)
                        return body
                result = transaction.execute_command(
                    command,
                    apply=apply_command,
                    auto_commit=False,
                )
                outcome = result.outcome.value
                changed = bool(result.changed)
                revision = int(result.revision)
                generation = int(result.generation)
                fence_epoch = int(result.fence_epoch)
                result_json = canonical_json_bytes(dict(result.result)).decode("utf-8")
                receipt = {
                    "schema": _PRIVATE_RECEIPT_SCHEMA,
                    "submission_id": str(submission_id),
                    "envelope_cid": envelope_cid,
                    "request_id": request_id,
                    "principal_did": principal_did,
                    "approver_did": approver_did,
                    "authority_ref_cid": authority_ref_cid,
                    "lease_id": lease_id,
                    "scope_id": scope_id,
                    "effect": effect,
                    "one_use_nonce": one_use_nonce,
                    "command_id": command_id,
                    "idempotency_key": idempotency_key,
                    "outcome": outcome,
                    "changed": changed,
                    "revision": revision,
                    "generation": generation,
                    "fence_epoch": fence_epoch,
                    "result_json": result_json,
                    "error": "",
                    "submitted_at": int(submitted_at),
                    "applied_at": time.time_ns(),
                }
                if daemon_operation:
                    receipt["daemon_operation"] = daemon_operation
                    receipt["daemon_operation_intent_cid"] = daemon_intent_cid
                transaction.record_authorized_command_receipt(
                    receipt_event_id=_receipt_event_id(str(submission_id)),
                    stream_id=f"authorized-command:{self.authorization_policy.shard_id}",
                    task_cid=scope_id,
                    session_id=command.session_id,
                    receipt=receipt,
                )
                transaction.commit()
            except Exception as exc:
                if transaction is not None:
                    transaction.rollback()
                recovered = self._private_receipt(str(submission_id))
                if recovered is not None:
                    if recovered.get("envelope_cid") != envelope_cid:
                        receipt = self._quarantine_divergent_ingress_replay(
                            ingress_slot=int(ingress_slot),
                            submission_id=str(submission_id),
                            durable_envelope_cid=str(
                                recovered.get("envelope_cid") or ""
                            ),
                            divergent_envelope_cid=envelope_cid,
                        )
                        quarantined_rows.append(
                            (int(ingress_slot), str(submission_id), envelope_cid)
                        )
                    else:
                        receipt = recovered
                        error = str(receipt.get("error") or "")
                        outcome = str(
                            receipt.get("outcome") or CommandOutcome.REJECTED.value
                        )
                        changed = bool(receipt.get("changed", False))
                        revision = int(receipt.get("revision", 0))
                        generation = int(receipt.get("generation", 0))
                        fence_epoch = int(receipt.get("fence_epoch", 0))
                        result_json = str(receipt.get("result_json") or "{}")
                else:
                    error = f"{type(exc).__name__}: {exc}"
                    receipt = {
                        "schema": _PRIVATE_RECEIPT_SCHEMA,
                        "submission_id": str(submission_id),
                        "envelope_cid": envelope_cid,
                        "request_id": request_id,
                        "principal_did": principal_did,
                        "approver_did": approver_did,
                        "authority_ref_cid": authority_ref_cid,
                        "lease_id": lease_id,
                        "scope_id": scope_id,
                        "effect": effect,
                        "one_use_nonce": one_use_nonce,
                        "command_id": command_id,
                        "idempotency_key": idempotency_key,
                        "outcome": CommandOutcome.REJECTED.value,
                        "changed": False,
                        "revision": 0,
                        "generation": 0,
                        "fence_epoch": 0,
                        "result_json": "{}",
                        "error": error,
                        "submitted_at": int(submitted_at),
                        "applied_at": time.time_ns(),
                    }
                    receipt = self._persist_rejected_receipt(receipt)
            applied_rows.append(
                (int(ingress_slot), str(submission_id), envelope_cid)
            )
            emitted.append(MappingProxyType(receipt))
        if emitted or durable_replay_rows or quarantined_rows:
            self._rebuild_projection()
        for ingress_slot, submission_id, envelope_cid in (
            durable_replay_rows + quarantined_rows + applied_rows
        ):
            ingress.execute(
                "DELETE FROM command_inbox WHERE ingress_slot = ? "
                "AND submission_id = ? AND envelope_cid = ?",
                [ingress_slot, submission_id, envelope_cid],
            )
        return tuple(emitted)

    @staticmethod
    def _daemon_result_from_receipt(
        receipt: Mapping[str, Any],
        *,
        envelope: AuthorizedStateCommand,
        intent_cid: str,
    ) -> Any:
        if (
            receipt.get("envelope_cid") != envelope.envelope_cid
            or receipt.get("daemon_operation_intent_cid") != intent_cid
            or receipt.get("outcome")
            not in {
                CommandOutcome.ACCEPTED.value,
                CommandOutcome.IDEMPOTENT_REPLAY.value,
            }
        ):
            raise QuackCommandFabricStateError(
                "daemon operation replay is divergent or was not accepted"
            )
        try:
            result = json.loads(str(receipt.get("result_json") or "{}"))
        except (TypeError, ValueError) as exc:
            raise QuackCommandFabricStateError(
                "daemon operation receipt result is corrupt"
            ) from exc
        if not isinstance(result, Mapping) or set(result) != {
            "daemon_operation",
            "intent_cid",
            "value",
        }:
            raise QuackCommandFabricStateError(
                "daemon operation receipt result is not the closed owner result"
            )
        return result["value"]

    def _submit_authorized_daemon_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_intent: Mapping[str, Any],
    ) -> Any:
        """Verify and atomically apply one exact daemon operation at the owner.

        The handler executes inside the same private StateTransaction that
        checks the live lease/fence, consumes request and nonce identities,
        applies the store-revision CAS, records command idempotency, and writes
        the durable receipt.  The handler is an owner-local implementation
        detail and is never returned to a remote supervisor.
        """

        # Reject subclass-controlled virtual serialization before even looking
        # up a durable result.  Exact base objects are the canonical authority
        # boundary for both new application and response-loss adoption.
        if type(envelope) is not AuthorizedStateCommand:
            raise QuackCommandAuthorizationError(
                "daemon operation command envelope is untyped"
            )
        if type(envelope.command) is not StateCommand:
            raise QuackCommandAuthorizationError(
                "daemon operation embedded command is untyped"
            )

        # Exact durable adoption is intentionally ahead of current capability,
        # envelope-freshness, lease, fence, and nonce checks.  The authority
        # decision and effects were committed atomically with this receipt; a
        # response or projection-publication loss must remain recoverable after
        # those transient authorities expire.  Reconstructing the transported
        # content-addressed intent still rejects any divergent replay.
        prior = self._private_receipt(envelope.submission_id)
        if prior is not None:
            from .quack_daemon_gateway import (
                QuackDaemonGatewayError,
                quack_daemon_operation_intent_from_envelope,
            )

            try:
                transported_intent = dict(
                    quack_daemon_operation_intent_from_envelope(envelope)
                )
            except QuackDaemonGatewayError as exc:
                raise QuackCommandFabricStateError(
                    "durable daemon operation replay intent is malformed"
                ) from exc
            if not isinstance(operation_intent, Mapping) or dict(
                operation_intent
            ) != transported_intent:
                raise QuackCommandFabricStateError(
                    "durable daemon operation replay intent is divergent"
                )
            value = self._daemon_result_from_receipt(
                prior,
                envelope=envelope,
                intent_cid=str(transported_intent["intent_cid"]),
            )
            if self.started:
                self._rebuild_projection()
            return value

        now_ms = int(self._clock_ms())
        authorization = self._verify_daemon_operation_authorization(
            envelope,
            operation_intent,
            now_ms=now_ms,
        )
        _verified, operation_handler = self._require_daemon_operation_capability(
            str(authorization["operation"])
        )
        intent_cid = str(authorization["intent_cid"])
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        submitted_at = time.time_ns()
        try:
            transaction.begin()
            lease = transaction.assert_live_authorized_command_lease(
                lease_id=envelope.lease_id,
                scope_id=envelope.scope_id,
                principal_did=envelope.principal_did,
                effect=envelope.effect,
                command_kind=envelope.command.command_kind,
                fence_epoch=envelope.command.fence_epoch,
                now_ms=now_ms,
            )
            if int(lease.get("fencing_token") or 0) != int(
                authorization["fencing_token"]
            ):
                raise QuackCommandAuthorizationError(
                    "daemon operation task fencing token is stale"
                )
            transaction.consume_authorized_command_replay_claims(
                request_id=envelope.request_id,
                one_use_nonce=envelope.one_use_nonce,
                scope_id=envelope.scope_id,
                effect=envelope.effect,
            )

            def apply_owner_operation(
                active_transaction: Any,
                command: StateCommand,
                _live: Any,
            ) -> Mapping[str, Any]:
                result = operation_handler.apply_authorized_daemon_operation(
                    operation=str(authorization["operation"]),
                    arguments=dict(authorization["arguments"]),
                    transaction=active_transaction,
                    command=command,
                    lease=dict(lease),
                )
                if not isinstance(result, Mapping) or set(result) != {"value"}:
                    raise QuackCommandFabricStateError(
                        "canonical daemon handler must return the closed {'value': ...} result"
                    )
                body = {
                    "daemon_operation": str(authorization["operation"]),
                    "intent_cid": intent_cid,
                    "value": result["value"],
                }
                canonical_json_bytes(body)
                return body

            result = transaction.execute_command(
                envelope.command,
                apply=apply_owner_operation,
                auto_commit=False,
            )
            result_body = dict(result.result)
            receipt = {
                "schema": _PRIVATE_RECEIPT_SCHEMA,
                "submission_id": envelope.submission_id,
                "envelope_cid": envelope.envelope_cid,
                "request_id": envelope.request_id,
                "principal_did": envelope.principal_did,
                "approver_did": envelope.approver_did,
                "authority_ref_cid": envelope.authority_ref_cid,
                "lease_id": envelope.lease_id,
                "scope_id": envelope.scope_id,
                "effect": envelope.effect,
                "one_use_nonce": envelope.one_use_nonce,
                "command_id": envelope.command.command_id,
                "idempotency_key": envelope.command.idempotency_key,
                "outcome": result.outcome.value,
                "changed": bool(result.changed),
                "revision": int(result.revision),
                "generation": int(result.generation),
                "fence_epoch": int(result.fence_epoch),
                "result_json": canonical_json_bytes(result_body).decode("utf-8"),
                "error": "",
                "submitted_at": submitted_at,
                "applied_at": time.time_ns(),
                "daemon_operation": str(authorization["operation"]),
                "daemon_operation_intent_cid": intent_cid,
            }
            transaction.record_authorized_command_receipt(
                receipt_event_id=_receipt_event_id(envelope.submission_id),
                stream_id=f"authorized-command:{self.authorization_policy.shard_id}",
                task_cid=envelope.scope_id,
                session_id=envelope.command.session_id,
                receipt=receipt,
            )
            transaction.commit()
        except Exception:
            transaction.rollback()
            recovered = self._private_receipt(envelope.submission_id)
            if recovered is None:
                raise
            value = self._daemon_result_from_receipt(
                recovered,
                envelope=envelope,
                intent_cid=intent_cid,
            )
            if self.started:
                self._rebuild_projection()
            return value
        if self.started:
            self._rebuild_projection()
        return result_body["value"]

    @staticmethod
    def _plan_r2_connection(transaction: Any) -> Any:
        """Return the private owner's active connection, never a public surface."""

        if not getattr(transaction, "active", False):
            raise QuackCommandFabricStateError(
                "Plan-R2 owner operation requires an active private transaction"
            )
        connection = getattr(transaction, "_connection", None)
        if connection is None or not callable(getattr(connection, "execute", None)):
            raise QuackCommandFabricStateError(
                "Plan-R2 owner transaction lost its private connection"
            )
        return connection

    @staticmethod
    def _plan_r2_rows(result: Any) -> list[tuple[Any, ...]]:
        fetchall = getattr(result, "fetchall", None)
        rows = list(fetchall() or ()) if callable(fetchall) else []
        return [
            (
                tuple(row[name] for name in row)
                if isinstance(row, Mapping)
                else tuple(row)
            )
            for row in rows
        ]

    @classmethod
    def _plan_r2_one(cls, result: Any, *, noun: str) -> tuple[Any, ...]:
        rows = cls._plan_r2_rows(result)
        if len(rows) != 1:
            raise QuackCommandFabricStateError(f"{noun} must resolve to exactly one canonical row")
        return rows[0]

    @staticmethod
    def _plan_r2_event_cursor(connection: Any) -> str:
        placeholders = ", ".join("?" for _ in _PLAN_R2_AUDIT_EVENT_TYPES)
        row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events "
            f"WHERE event_type NOT IN ({placeholders})",
            sorted(_PLAN_R2_AUDIT_EVENT_TYPES),
        ).fetchone()
        sequence = int(row[0] if row is not None else 0)
        return f"event-cursor-{sequence}"

    @staticmethod
    def _plan_r2_next_event_position(connection: Any, *, stream_id: str) -> tuple[int, int]:
        stream_row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM domain_events WHERE stream_id = ?",
            [stream_id],
        ).fetchone()
        global_row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        return (
            int(stream_row[0] if stream_row is not None else 0) + 1,
            int(global_row[0] if global_row is not None else 0) + 1,
        )

    @classmethod
    def _plan_r2_record_event(
        cls,
        connection: Any,
        *,
        event_id: str,
        stream_id: str,
        event_type: str,
        task_cid: str,
        session_id: str,
        recorded_at: str,
        body: Mapping[str, Any],
        expected_global_sequence: int | None = None,
    ) -> int:
        sequence, global_sequence = cls._plan_r2_next_event_position(
            connection, stream_id=stream_id
        )
        if expected_global_sequence is not None and global_sequence != expected_global_sequence:
            raise QuackCommandFabricStateError(
                "Plan-R2 event cursor changed inside the owner transaction"
            )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?)
            """,
            [
                event_id,
                stream_id,
                sequence,
                global_sequence,
                event_type,
                task_cid,
                session_id,
                recorded_at,
                canonical_json_bytes(dict(body)).decode("utf-8"),
            ],
        )
        return global_sequence

    @classmethod
    def _plan_r2_lookup_result(
        cls,
        transaction: Any,
        *,
        envelope_cid: str,
        operation_payload_cid: str | None,
        operation: str,
    ) -> Mapping[str, Any] | None:
        connection = cls._plan_r2_connection(transaction)
        event_id = _plan_r2_result_event_id(envelope_cid)
        rows = cls._plan_r2_rows(
            connection.execute(
                "SELECT body_json FROM domain_events WHERE event_id = ? AND event_type = ? LIMIT 1",
                [event_id, _PLAN_R2_RESULT_EVENT_TYPE],
            )
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise QuackCommandFabricStateError("Plan-R2 result identity is not unique")
        wrapper = _json_object(rows[0][0], noun="Plan-R2 owner result")
        expected_fields = {
            "schema",
            "envelope_cid",
            "operation_payload_cid",
            "operation",
            "result",
        }
        if (
            set(wrapper) != expected_fields
            or wrapper.get("schema") != _PLAN_R2_RESULT_SCHEMA
            or wrapper.get("envelope_cid") != envelope_cid
            or (
                operation_payload_cid is not None
                and wrapper.get("operation_payload_cid") != operation_payload_cid
            )
            or wrapper.get("operation") != operation
            or not isinstance(wrapper.get("result"), Mapping)
        ):
            raise QuackCommandFabricStateError("durable Plan-R2 result is divergent or corrupt")
        return MappingProxyType(dict(wrapper["result"]))

    @classmethod
    def _plan_r2_record_result(
        cls,
        transaction: Any,
        *,
        envelope: AuthorizedStateCommand,
        operation_payload_cid: str,
        operation: str,
        result: Mapping[str, Any],
        recorded_at: str,
    ) -> None:
        connection = cls._plan_r2_connection(transaction)
        wrapper = {
            "schema": _PLAN_R2_RESULT_SCHEMA,
            "envelope_cid": envelope.envelope_cid,
            "operation_payload_cid": operation_payload_cid,
            "operation": operation,
            "result": dict(result),
        }
        cls._plan_r2_record_event(
            connection,
            event_id=_plan_r2_result_event_id(envelope.envelope_cid),
            stream_id=f"plan-r2-owner:{envelope.shard_id}",
            event_type=_PLAN_R2_RESULT_EVENT_TYPE,
            task_cid=envelope.scope_id,
            session_id=envelope.command.session_id,
            recorded_at=recorded_at,
            body=wrapper,
        )

    @classmethod
    def _plan_r2_task_row(cls, connection: Any, task_cid: str) -> dict[str, Any]:
        row = cls._plan_r2_one(
            connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                       ordinal, status, revision, priority, identity_json,
                       body_json
                FROM tasks WHERE task_cid = ? LIMIT 1
                """,
                [task_cid],
            ),
            noun=f"Plan-R2 task {task_cid}",
        )
        return {
            "task_cid": str(row[0]),
            "task_alias": str(row[1]),
            "goal_cid": str(row[2]),
            "plan_cid": str(row[3]),
            "objective_id": str(row[4]),
            "ordinal": int(row[5]),
            "status": str(row[6]),
            "revision": int(row[7]),
            "priority": str(row[8]),
            "identity": _json_object(row[9], noun="task identity_json"),
            "body": _json_object(row[10], noun="task body_json"),
        }

    @classmethod
    def _plan_r2_active_snapshot(
        cls,
        transaction: Any,
        *,
        store_id: str,
        owner_status: str = "running",
    ) -> dict[str, Any]:
        if type(owner_status) is not str or owner_status not in {"ready", "running"}:
            raise QuackCommandFabricStateError(
                "Plan-R2 owner status is outside the closed vocabulary"
            )
        connection = cls._plan_r2_connection(transaction)
        generation = transaction.load_generation()
        server_row = cls._plan_r2_one(
            connection.execute(
                """
                SELECT s.server_id, e.epoch, e.fence_epoch
                FROM state_servers AS s
                JOIN server_epochs AS e ON e.server_id = s.server_id
                WHERE s.store_id = ? AND s.generation = ?
                  AND s.status = ? AND e.ended_at IS NULL
                """,
                [store_id, generation.generation, owner_status],
            ),
            noun="live Plan-R2 owner epoch",
        )
        plan_row = cls._plan_r2_one(
            connection.execute(
                """
                SELECT plan_cid, goal_cid, plan_alias, revision, body_json
                FROM plans WHERE status = 'active'
                """
            ),
            noun="active Plan-R2 predecessor plan",
        )
        body = _json_object(plan_row[4], noun="active plan body_json")
        plan_cid = str(plan_row[0])
        plan_root_cid = str(body.get("plan_root_cid") or plan_cid)
        semantic_root_cid = str(body.get("semantic_root_cid") or "")
        if not semantic_root_cid:
            semantic_roots: set[str] = set()
            for task_body_row in cls._plan_r2_rows(
                connection.execute(
                    "SELECT body_json FROM tasks WHERE plan_cid = ? ORDER BY task_cid",
                    [plan_cid],
                )
            ):
                task_body = _json_object(task_body_row[0], noun="active plan task body_json")
                nested = task_body.get("body")
                if isinstance(nested, Mapping):
                    candidate = str(nested.get("source_semantic_state_root") or "")
                    if candidate.startswith("sha256:") and len(candidate) == 71:
                        semantic_roots.add(candidate)
            if len(semantic_roots) == 1:
                semantic_root_cid = next(iter(semantic_roots))
        if (
            not plan_root_cid.startswith("sha256:")
            or len(plan_root_cid) != 71
            or not semantic_root_cid.startswith("sha256:")
            or len(semantic_root_cid) != 71
        ):
            raise QuackCommandFabricCapabilityError(
                "canonical active plan lacks an exact plan/semantic root; "
                "a reviewed source-schema materialization is required"
            )
        return {
            "generation": generation,
            "server_id": str(server_row[0]),
            "epoch": int(server_row[1]),
            "fence": int(server_row[2]),
            "plan_cid": plan_cid,
            "goal_cid": str(plan_row[1]),
            "plan_alias": str(plan_row[2]),
            "plan_revision": int(plan_row[3]),
            "plan_root_cid": plan_root_cid,
            "semantic_root_cid": semantic_root_cid,
            "event_cursor": cls._plan_r2_event_cursor(connection),
            "version": int(generation.revision),
        }

    @classmethod
    def _plan_r2_assert_protected_rows(
        cls,
        transaction: Any,
        authorization: Mapping[str, Any],
    ) -> None:
        connection = cls._plan_r2_connection(transaction)
        protected = {str(item["task_cid"]): dict(item) for item in authorization["protected_tasks"]}
        current = {
            str(row[0])
            for row in cls._plan_r2_rows(
                connection.execute(
                    "SELECT task_cid FROM tasks WHERE status IN "
                    "('claimed','running','settling','completed','accepted') "
                    "ORDER BY task_cid"
                )
            )
        }
        if current != set(protected):
            raise QuackCommandFabricStateError(
                "Plan-R2 protected population differs from current authority"
            )
        for task_cid, expected in protected.items():
            observed = cls._plan_r2_task_row(connection, task_cid)
            if (
                observed != expected.get("task_row")
                or observed.get("status") != expected.get("status")
                or observed.get("revision") != expected.get("revision")
            ):
                raise QuackCommandFabricStateError(
                    f"Plan-R2 protected full-row CAS failed for {task_cid}"
                )

    @staticmethod
    def _plan_r2_assert_snapshot(
        snapshot: Mapping[str, Any],
        authorization: Mapping[str, Any],
        capability: Mapping[str, Any],
    ) -> None:
        exact = {
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "version": authorization["expected_version"],
            "plan_cid": authorization["expected_active_plan_cid"],
            "plan_root_cid": authorization["expected_active_plan_root_cid"],
            "plan_revision": authorization["expected_active_plan_revision"],
            "event_cursor": authorization["expected_event_cursor"],
            "semantic_root_cid": authorization["expected_semantic_root_cid"],
        }
        mismatched = sorted(
            field for field, expected in exact.items() if snapshot.get(field) != expected
        )
        generation = snapshot["generation"]
        if (
            generation.generation != authorization["owner_generation"]
            or generation.fence_epoch != authorization["fencing_token"]
            or capability.get("epoch") != snapshot.get("epoch")
            or capability.get("fence") != snapshot.get("fence")
        ):
            mismatched.append("live_owner_generation_epoch_fence")
        if mismatched:
            raise QuackCommandFabricStateError(
                "Plan-R2 source CAS is stale: " + ", ".join(sorted(set(mismatched)))
            )

    @classmethod
    def _plan_r2_dependencies(cls, connection: Any, task_cids: set[str]) -> list[dict[str, str]]:
        values = [
            {
                "task_cid": str(row[0]),
                "dependency_task_cid": str(row[1]),
                "kind": str(row[2]),
            }
            for row in cls._plan_r2_rows(
                connection.execute(
                    "SELECT task_cid, dependency_task_cid, kind "
                    "FROM task_dependencies ORDER BY task_cid, "
                    "dependency_task_cid, kind"
                )
            )
            if str(row[0]) in task_cids
        ]
        return values

    @classmethod
    def _plan_r2_assert_population_readback(
        cls,
        transaction: Any,
        authorization: Mapping[str, Any],
    ) -> None:
        connection = cls._plan_r2_connection(transaction)
        expected_tasks = [dict(value) for value in authorization["tasks"]]
        observed_tasks = [
            cls._plan_r2_task_row(connection, str(value["task_cid"])) for value in expected_tasks
        ]
        if observed_tasks != expected_tasks:
            raise QuackCommandFabricStateError(
                "Plan-R2 task population readback differs from authorization"
            )
        task_cids = {str(value["task_cid"]) for value in expected_tasks}
        if cls._plan_r2_dependencies(connection, task_cids) != [
            dict(value) for value in authorization["dependencies"]
        ]:
            raise QuackCommandFabricStateError(
                "Plan-R2 dependency population readback differs from authorization"
            )

    @classmethod
    def _plan_r2_apply_population(
        cls,
        transaction: Any,
        *,
        authorization: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        envelope: AuthorizedStateCommand,
        committed_at_ms: int,
    ) -> dict[str, Any]:
        connection = cls._plan_r2_connection(transaction)
        new_plan = dict(authorization["new_plan"])
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(committed_at_ms / 1000))
        returned = cls._plan_r2_rows(
            connection.execute(
                """
                UPDATE plans SET status = 'superseded', updated_at = ?
                WHERE plan_cid = ? AND status = 'active' AND revision = ?
                RETURNING plan_cid
                """,
                [now_iso, snapshot["plan_cid"], snapshot["plan_revision"]],
            )
        )
        if returned != [(snapshot["plan_cid"],)]:
            raise QuackCommandFabricStateError(
                "Plan-R2 predecessor plan CAS lost its owner transaction"
            )
        if cls._plan_r2_rows(
            connection.execute(
                "SELECT plan_cid FROM plans WHERE plan_cid = ?",
                [new_plan["plan_cid"]],
            )
        ):
            raise QuackCommandFabricStateError(
                "Plan-R2 plan identity already exists without its durable receipt"
            )
        plan_json = canonical_json_bytes(new_plan).decode("utf-8")
        connection.execute(
            """
            INSERT INTO plans (
                plan_cid, goal_cid, plan_alias, status, created_at, updated_at,
                revision, body_json
            ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            [
                new_plan["plan_cid"],
                snapshot["goal_cid"],
                new_plan["plan_alias"],
                now_iso,
                now_iso,
                new_plan["revision"],
                plan_json,
            ],
        )
        connection.execute(
            "INSERT INTO plan_revisions VALUES (?, ?, ?, ?)",
            [new_plan["plan_cid"], new_plan["revision"], plan_json, now_iso],
        )

        proposed = {str(value["task_cid"]): dict(value) for value in authorization["tasks"]}
        protected = {str(value["task_cid"]) for value in authorization["protected_tasks"]}
        old_rows = cls._plan_r2_rows(
            connection.execute(
                "SELECT task_cid, revision, status FROM tasks WHERE plan_cid = ?",
                [snapshot["plan_cid"]],
            )
        )
        for task_cid, revision, status in old_rows:
            task_id = str(task_cid)
            if task_id in proposed or task_id in protected:
                continue
            if str(status) in _PLAN_R2_PROTECTED_STATUSES:
                raise QuackCommandFabricStateError(
                    "Plan-R2 attempted to omit a protected predecessor task"
                )
            connection.execute(
                "UPDATE tasks SET status = 'superseded', revision = ?, "
                "updated_at = ? WHERE task_cid = ? AND revision = ?",
                [int(revision) + 1, now_iso, task_id, int(revision)],
            )

        for task_id, task in proposed.items():
            existing_rows = cls._plan_r2_rows(
                connection.execute("SELECT revision FROM tasks WHERE task_cid = ?", [task_id])
            )
            if task_id in protected:
                continue
            identity_json = canonical_json_bytes(task["identity"]).decode("utf-8")
            body_json = canonical_json_bytes(task["body"]).decode("utf-8")
            if existing_rows:
                current_revision = int(existing_rows[0][0])
                if int(task["revision"]) <= current_revision:
                    raise QuackCommandFabricStateError(
                        f"Plan-R2 task revision does not advance for {task_id}"
                    )
                returned_task = cls._plan_r2_rows(
                    connection.execute(
                        """
                        UPDATE tasks SET task_alias = ?, goal_cid = ?, plan_cid = ?,
                            objective_id = ?, ordinal = ?, status = ?, revision = ?,
                            priority = ?, updated_at = ?, identity_json = ?, body_json = ?
                        WHERE task_cid = ? AND revision = ? RETURNING task_cid
                        """,
                        [
                            task["task_alias"],
                            task["goal_cid"],
                            task["plan_cid"],
                            task["objective_id"],
                            task["ordinal"],
                            task["status"],
                            task["revision"],
                            task["priority"],
                            now_iso,
                            identity_json,
                            body_json,
                            task_id,
                            current_revision,
                        ],
                    )
                )
                if returned_task != [(task_id,)]:
                    raise QuackCommandFabricStateError(f"Plan-R2 task CAS failed for {task_id}")
            else:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_cid, objective_id,
                        ordinal, status, revision, priority, created_at, updated_at,
                        identity_json, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        task_id,
                        task["task_alias"],
                        task["goal_cid"],
                        task["plan_cid"],
                        task["objective_id"],
                        task["ordinal"],
                        task["status"],
                        task["revision"],
                        task["priority"],
                        now_iso,
                        now_iso,
                        identity_json,
                        body_json,
                    ],
                )
            revision_body = canonical_json_bytes(task).decode("utf-8")
            prior_revision = cls._plan_r2_rows(
                connection.execute(
                    "SELECT status, body_json FROM task_revisions "
                    "WHERE task_cid = ? AND revision = ?",
                    [task_id, task["revision"]],
                )
            )
            if prior_revision and prior_revision != [(task["status"], revision_body)]:
                raise QuackCommandFabricStateError(
                    f"Plan-R2 task revision identity conflicts for {task_id}"
                )
            if not prior_revision:
                connection.execute(
                    "INSERT INTO task_revisions VALUES (?, ?, ?, ?, ?)",
                    [
                        task_id,
                        task["revision"],
                        task["status"],
                        revision_body,
                        now_iso,
                    ],
                )

        for task_id in sorted(proposed):
            connection.execute("DELETE FROM task_dependencies WHERE task_cid = ?", [task_id])
        for dependency in authorization["dependencies"]:
            connection.execute(
                "INSERT INTO task_dependencies VALUES (?, ?, ?)",
                [
                    dependency["task_cid"],
                    dependency["dependency_task_cid"],
                    dependency["kind"],
                ],
            )
        cls._plan_r2_assert_protected_rows(transaction, authorization)
        cls._plan_r2_assert_population_readback(transaction, authorization)

        event_stream = f"plan-r2:{authorization['shard_id']}"
        _, next_global = cls._plan_r2_next_event_position(connection, stream_id=event_stream)
        after_cursor = f"event-cursor-{next_global}"
        transaction_cid = (
            "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": "AuthorizedPlanR2OwnerTransactionIdentity@1",
                        "authorization_cid": authorization["authorization_cid"],
                        "authorized_apply_command_cid": envelope.envelope_cid,
                        "before_version": snapshot["version"],
                        "after_version": snapshot["version"] + 1,
                        "before_event_cursor": snapshot["event_cursor"],
                        "after_event_cursor": after_cursor,
                        "population_cid": authorization["population_cid"],
                    }
                )
            ).hexdigest()
        )
        applied_body = {
            "schema": "AuthorizedPlanR2PopulationApplied@1",
            "authorization_cid": authorization["authorization_cid"],
            "command_cid": envelope.envelope_cid,
            "transaction_cid": transaction_cid,
            "population_cid": authorization["population_cid"],
            "plan_cid": new_plan["plan_cid"],
            "plan_root_cid": new_plan["plan_root_cid"],
            "semantic_root_cid": new_plan["semantic_root_cid"],
            "after_version": snapshot["version"] + 1,
            "after_event_cursor": after_cursor,
        }
        cls._plan_r2_record_event(
            connection,
            event_id=f"plan-r2-applied:{transaction_cid}",
            stream_id=event_stream,
            event_type=_PLAN_R2_APPLIED_EVENT_TYPE,
            task_cid=envelope.scope_id,
            session_id=envelope.command.session_id,
            recorded_at=now_iso,
            body=applied_body,
            expected_global_sequence=next_global,
        )
        return {
            "transaction_cid": transaction_cid,
            "after_event_cursor": after_cursor,
            "committed_at_ms": committed_at_ms,
        }

    @staticmethod
    def _plan_r2_submission_replay_identity(
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        """Return only the immutable identity needed for durable adoption."""

        if type(envelope) is not AuthorizedStateCommand:
            raise QuackCommandIngressError("Plan-R2 owner requires AuthorizedStateCommand@1")
        if type(envelope.command) is not StateCommand:
            raise QuackCommandIngressError("Plan-R2 owner requires exact StateCommand@1")
        if not isinstance(operation_payload, Mapping) or not all(
            isinstance(key, str) for key in operation_payload
        ):
            raise QuackCommandIngressError("Plan-R2 operation payload is not an object")
        payload = dict(operation_payload)
        operation = str(payload.get("operation") or "")
        expected_fields = {
            PREPARE_PLAN_R2_OPERATION: {"schema", "operation", "authorization"},
            APPLY_PLAN_R2_OPERATION: {
                "schema",
                "operation",
                "authorization",
                "prepared_projection",
            },
            OBSERVE_PLAN_R2_OPERATION: {
                "schema",
                "operation",
                "authorization",
                "transition_receipt",
            },
        }
        if (
            operation not in _PLAN_R2_OPERATIONS
            or payload.get("schema") != PLAN_R2_OWNER_OPERATION_SCHEMA
            or set(payload) != expected_fields[operation]
            or not isinstance(payload.get("authorization"), Mapping)
        ):
            raise QuackCommandIngressError(
                "Plan-R2 operation does not use its exact closed owner schema"
            )
        payload_cid = "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        return operation, payload, payload_cid

    def _validate_plan_r2_submission(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], Mapping[str, Any], str]:
        from ..planning import external_agent_plan_r2 as plan_r2

        if not self.started:
            raise QuackCommandFabricStateError("fabric owner is not started")
        operation, payload, payload_cid = self._plan_r2_submission_replay_identity(
            envelope,
            operation_payload,
        )
        authorization = dict(payload["authorization"])
        capability = self._require_plan_r2_capability()
        decision = plan_r2.assess_plan_r2_transition(
            authorization,
            capability,
            trusted_operator_dids=self._trusted_plan_r2_operator_dids,
            trusted_security_reviewer_dids=(self._trusted_plan_r2_security_reviewer_dids),
            trusted_capability_reviewer_dids=(self._trusted_plan_r2_capability_reviewer_dids),
            now_ms=int(self._clock_ms()),
        )
        if decision.get("allowed") is not True:
            raise QuackCommandFabricCapabilityError(
                ",".join(str(item) for item in decision.get("blockers") or ())
                or "typed_quack_plan_transition_atomic_owner_operation_unavailable"
            )
        verify_authorized_state_command(
            envelope,
            policy=self.authorization_policy,
            now_ms=int(self._clock_ms()),
        )
        command = envelope.command
        suffix = operation.replace(".", "-")
        if operation == OBSERVE_PLAN_R2_OPERATION:
            suffix = f"{suffix}:{envelope.ingress_slot}"
        expected_kind = "migrate" if operation == APPLY_PLAN_R2_OPERATION else "observe"
        parameters = dict(command.parameters)
        expected_parameters = {
            "interface": "AuthorizedPlanR2TransitionRepository@1",
            "operation": operation,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "operation_payload_cid": payload_cid,
            "shard_id": authorization["shard_id"],
            "store_id": authorization["store_id"],
            "expected_event_cursor": authorization["expected_event_cursor"],
            "population_cid": authorization["population_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "prepared_projection_cid": "",
            "transition_receipt_cid": "",
        }
        if operation == APPLY_PLAN_R2_OPERATION:
            prepared = payload.get("prepared_projection")
            if not isinstance(prepared, Mapping):
                raise QuackCommandIngressError("Plan-R2 apply lacks its prepared projection")
            plan_r2._validate_prepared(
                prepared,
                authorization=authorization,
                capability=capability,
                now_ms=int(self._clock_ms()),
            )
            expected_parameters["prepared_projection_cid"] = str(prepared["projection_cid"])
        elif operation == OBSERVE_PLAN_R2_OPERATION:
            receipt = payload.get("transition_receipt")
            if not isinstance(receipt, Mapping):
                raise QuackCommandIngressError("Plan-R2 observe lacks its transition receipt")
            plan_r2._validate_transition_receipt_for_launch(
                receipt,
                authorization=authorization,
                now_ms=int(self._clock_ms()),
            )
            expected_parameters["transition_receipt_cid"] = str(receipt["receipt_cid"])
        identity_checks = (
            command.command_kind.value == expected_kind,
            command.command_id == f"{authorization['request_id']}:{suffix}",
            command.idempotency_key == f"{authorization['idempotency_key']}:{suffix}",
            command.store_id == authorization["store_id"],
            command.session_id == authorization["lease_id"],
            command.expected_generation == authorization["owner_generation"],
            command.expected_revision == authorization["expected_version"],
            command.fence_epoch == authorization["fencing_token"],
            envelope.request_id == f"{authorization['request_id']}:{suffix}",
            envelope.authority_ref_cid == authorization["authorization_cid"],
            envelope.board_namespace == authorization["board_namespace"],
            envelope.shard_id == authorization["shard_id"],
            envelope.owner_principal_did == authorization["owner_principal_did"],
            envelope.lease_id == authorization["lease_id"],
            envelope.scope_id == authorization["plan_root_cid"],
            envelope.one_use_nonce == f"{authorization['one_use_nonce']}:{suffix}",
            parameters == expected_parameters,
            authorization["store_id"] == self.authorization_policy.store_id,
            authorization["shard_id"] == self.authorization_policy.shard_id,
            authorization["shard_id"] != authorization["store_id"],
        )
        if not all(identity_checks):
            raise QuackCommandAuthorizationError(
                "Plan-R2 command/envelope/payload identity join failed"
            )
        return operation, authorization, capability, payload_cid

    @staticmethod
    def _plan_r2_assert_live_lease(
        transaction: Any,
        *,
        envelope: AuthorizedStateCommand,
        authorization: Mapping[str, Any],
        now_ms: int,
    ) -> None:
        lease = transaction.assert_live_authorized_command_lease(
            lease_id=envelope.lease_id,
            scope_id=envelope.scope_id,
            principal_did=envelope.principal_did,
            effect=envelope.effect,
            command_kind=envelope.command.command_kind,
            fence_epoch=envelope.command.fence_epoch,
            now_ms=now_ms,
        )
        if (
            lease.get("fencing_token") != authorization["fencing_token"]
            or lease.get("fence_epoch") != authorization["fencing_token"]
        ):
            raise QuackCommandAuthorizationError(
                "Plan-R2 lease does not bind the exact fencing token"
            )

    def _plan_r2_recover_result(
        self,
        *,
        envelope: AuthorizedStateCommand,
        payload_cid: str,
        operation: str,
    ) -> Mapping[str, Any] | None:
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        try:
            transaction.begin()
            result = self._plan_r2_lookup_result(
                transaction,
                envelope_cid=envelope.envelope_cid,
                operation_payload_cid=payload_cid,
                operation=operation,
            )
            transaction.commit()
            return result
        except Exception:
            transaction.rollback()
            raise

    def _submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.started:
            raise QuackCommandFabricStateError("fabric owner is not started")
        replay_operation, _replay_payload, replay_payload_cid = (
            self._plan_r2_submission_replay_identity(envelope, operation_payload)
        )
        recovered = self._plan_r2_recover_result(
            envelope=envelope,
            payload_cid=replay_payload_cid,
            operation=replay_operation,
        )
        if recovered is not None:
            return recovered
        operation, authorization, capability, payload_cid = self._validate_plan_r2_submission(
            envelope, operation_payload
        )
        transaction = self.repository.transaction(
            expected_generation=self.repository.load_generation()
        )
        now_ms = int(self._clock_ms())
        try:
            transaction.begin()
            self._plan_r2_assert_live_lease(
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
            snapshot = self._plan_r2_active_snapshot(
                transaction,
                store_id=self.authorization_policy.store_id,
            )
            if operation != OBSERVE_PLAN_R2_OPERATION:
                self._plan_r2_assert_snapshot(snapshot, authorization, capability)
                self._plan_r2_assert_protected_rows(transaction, authorization)
            else:
                generation = snapshot["generation"]
                if (
                    snapshot["epoch"] != authorization["expected_epoch"]
                    or snapshot["fence"] != authorization["fencing_token"]
                    or generation.generation != authorization["owner_generation"]
                    or generation.fence_epoch != authorization["fencing_token"]
                ):
                    raise QuackCommandFabricStateError(
                        "Plan-R2 observation owner epoch/fence is stale"
                    )
            if operation == PREPARE_PLAN_R2_OPERATION:
                result = self._prepare_plan_r2_result(
                    envelope=envelope,
                    authorization=authorization,
                    capability=capability,
                    snapshot=snapshot,
                    now_ms=now_ms,
                )
            elif operation == APPLY_PLAN_R2_OPERATION:
                result = self._apply_plan_r2_result(
                    transaction,
                    envelope=envelope,
                    authorization=authorization,
                    capability=capability,
                    prepared=dict(operation_payload["prepared_projection"]),
                    snapshot=snapshot,
                    now_ms=now_ms,
                )
            else:
                result = self._observe_plan_r2_result(
                    transaction,
                    envelope=envelope,
                    authorization=authorization,
                    receipt=dict(operation_payload["transition_receipt"]),
                    snapshot=snapshot,
                    now_ms=now_ms,
                )
            self._plan_r2_record_result(
                transaction,
                envelope=envelope,
                operation_payload_cid=payload_cid,
                operation=operation,
                result=result,
                recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ms / 1000)),
            )
            transaction.commit()
            self._plan_r2_after_commit_hook(result)
            return MappingProxyType(dict(result))
        except Exception as exc:
            transaction.rollback()
            recovered = self._plan_r2_recover_result(
                envelope=envelope,
                payload_cid=payload_cid,
                operation=operation,
            )
            if recovered is not None:
                return recovered
            raise exc

    @staticmethod
    def _prepare_plan_r2_result(
        *,
        envelope: AuthorizedStateCommand,
        authorization: Mapping[str, Any],
        capability: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        now_ms: int,
    ) -> dict[str, Any]:
        from ..planning import external_agent_plan_r2 as plan_r2

        value = {
            "schema": plan_r2.PLAN_R2_PREPARED_PROJECTION_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "capability_cid": capability["capability_cid"],
            "authorized_prepare_command_cid": envelope.envelope_cid,
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": snapshot["epoch"],
            "fence": snapshot["fence"],
            "before_plan_cid": snapshot["plan_cid"],
            "before_plan_root_cid": snapshot["plan_root_cid"],
            "before_plan_revision": snapshot["plan_revision"],
            "before_version": snapshot["version"],
            "before_event_cursor": snapshot["event_cursor"],
            "before_semantic_root_cid": snapshot["semantic_root_cid"],
            "population_cid": authorization["population_cid"],
            "plan_root_cid": authorization["plan_root_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "frontier_cid": authorization["frontier_cid"],
            "prepared_at_ms": now_ms,
            "expires_at_ms": min(int(authorization["expires_at_ms"]), envelope.expires_at_ms),
            "authority_mutated": False,
            "process_started": False,
        }
        value["projection_cid"] = plan_r2._cid(value)
        return value

    @classmethod
    def _apply_plan_r2_result(
        cls,
        transaction: Any,
        *,
        envelope: AuthorizedStateCommand,
        authorization: Mapping[str, Any],
        capability: Mapping[str, Any],
        prepared: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        now_ms: int,
    ) -> dict[str, Any]:
        from ..planning import external_agent_plan_r2 as plan_r2

        prepared_record = cls._plan_r2_lookup_result(
            transaction,
            envelope_cid=str(prepared["authorized_prepare_command_cid"]),
            operation_payload_cid=None,
            operation=PREPARE_PLAN_R2_OPERATION,
        )
        if prepared_record is None or dict(prepared_record) != dict(prepared):
            raise QuackCommandFabricStateError(
                "Plan-R2 apply is not joined to the durable prepare result"
            )

        def apply_population(_transaction: Any, _command: Any, live: Any) -> Mapping[str, Any]:
            if live.revision != snapshot["version"]:
                raise QuackCommandFabricStateError("Plan-R2 store version CAS is stale")
            return cls._plan_r2_apply_population(
                _transaction,
                authorization=authorization,
                snapshot=snapshot,
                envelope=envelope,
                committed_at_ms=now_ms,
            )

        cas = transaction.execute_command(
            envelope.command,
            apply=apply_population,
            auto_commit=False,
        )
        if cas.outcome is not CommandOutcome.ACCEPTED or cas.changed is not True:
            raise QuackCommandFabricStateError(
                "Plan-R2 apply did not produce one newly accepted mutation"
            )
        mutation = dict(cas.result)
        new_plan = authorization["new_plan"]
        value = {
            "schema": plan_r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "capability_cid": capability["capability_cid"],
            "authorized_prepare_command_cid": prepared["authorized_prepare_command_cid"],
            "authorized_apply_command_cid": envelope.envelope_cid,
            "prepared_projection_cid": prepared["projection_cid"],
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "before_plan_cid": snapshot["plan_cid"],
            "after_plan_cid": new_plan["plan_cid"],
            "before_plan_root_cid": snapshot["plan_root_cid"],
            "after_plan_root_cid": new_plan["plan_root_cid"],
            "before_plan_revision": snapshot["plan_revision"],
            "after_plan_revision": new_plan["revision"],
            "before_version": snapshot["version"],
            "after_version": cas.revision,
            "before_event_cursor": snapshot["event_cursor"],
            "after_event_cursor": mutation["after_event_cursor"],
            "before_semantic_root_cid": snapshot["semantic_root_cid"],
            "after_semantic_root_cid": new_plan["semantic_root_cid"],
            "population_cid": authorization["population_cid"],
            "task_population_cid": authorization["task_population_cid"],
            "dependency_population_cid": authorization["dependency_population_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "frontier_cid": authorization["frontier_cid"],
            "frontier_task_cids": authorization["frontier_task_cids"],
            "protected_tasks_unchanged": True,
            "transaction_cid": mutation["transaction_cid"],
            "replayed": False,
            "committed_at_ms": mutation["committed_at_ms"],
        }
        value["receipt_cid"] = plan_r2._cid(value)
        return value

    @classmethod
    def _observe_plan_r2_result(
        cls,
        transaction: Any,
        *,
        envelope: AuthorizedStateCommand,
        authorization: Mapping[str, Any],
        receipt: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        now_ms: int,
    ) -> dict[str, Any]:
        from ..planning import external_agent_plan_r2 as plan_r2

        connection = cls._plan_r2_connection(transaction)
        row = cls._plan_r2_one(
            connection.execute(
                "SELECT body_json FROM domain_events WHERE event_id = ? AND event_type = ?",
                [
                    _plan_r2_result_event_id(str(receipt["authorized_apply_command_cid"])),
                    _PLAN_R2_RESULT_EVENT_TYPE,
                ],
            ),
            noun="durable Plan-R2 apply result",
        )
        wrapper = _json_object(row[0], noun="durable Plan-R2 apply result")
        if dict(wrapper.get("result") or {}) != dict(receipt):
            raise QuackCommandFabricStateError(
                "Plan-R2 observation is not joined to the durable apply result"
            )
        expected_after = {
            "version": receipt["after_version"],
            "event_cursor": receipt["after_event_cursor"],
            "plan_cid": receipt["after_plan_cid"],
            "plan_root_cid": receipt["after_plan_root_cid"],
            "plan_revision": receipt["after_plan_revision"],
            "semantic_root_cid": receipt["after_semantic_root_cid"],
            "epoch": receipt["epoch"],
            "fence": receipt["fence"],
        }
        mismatched = sorted(
            field for field, expected in expected_after.items() if snapshot.get(field) != expected
        )
        if mismatched:
            raise QuackCommandFabricStateError(
                "Plan-R2 readback is stale: " + ", ".join(mismatched)
            )
        cls._plan_r2_assert_population_readback(transaction, authorization)
        cls._plan_r2_assert_protected_rows(transaction, authorization)
        value = {
            "schema": plan_r2.PLAN_R2_STATE_OBSERVATION_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "transition_receipt_cid": receipt["receipt_cid"],
            "transaction_cid": receipt["transaction_cid"],
            "authorized_prepare_command_cid": receipt["authorized_prepare_command_cid"],
            "authorized_apply_command_cid": receipt["authorized_apply_command_cid"],
            "quack_command_fabric_qualification_cid": authorization[
                "quack_command_fabric_qualification_cid"
            ],
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "owner_principal_did": authorization["owner_principal_did"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": snapshot["epoch"],
            "fence": snapshot["fence"],
            "store_version": snapshot["version"],
            "active_plan_cid": snapshot["plan_cid"],
            "active_plan_root_cid": snapshot["plan_root_cid"],
            "active_plan_revision": snapshot["plan_revision"],
            "event_cursor": snapshot["event_cursor"],
            "semantic_root_cid": snapshot["semantic_root_cid"],
            "population_cid": receipt["population_cid"],
            "task_population_cid": receipt["task_population_cid"],
            "dependency_population_cid": receipt["dependency_population_cid"],
            "protected_tasks_root_cid": receipt["protected_tasks_root_cid"],
            "frontier_cid": receipt["frontier_cid"],
            "frontier_task_cids": receipt["frontier_task_cids"],
            "captured_at_ms": now_ms,
            "authority_mutated": False,
            "process_started": False,
        }
        value["observation_cid"] = plan_r2._cid(value)
        return value

    def stop(self) -> None:
        errors: list[BaseException] = []
        repository = self._repository
        self._repository = None
        if repository is not None:
            try:
                repository.close()
            except BaseException as exc:
                errors.append(exc)
        for endpoint in (self._state_server, self._ingress_server):
            if endpoint is not None:
                try:
                    endpoint.stop()
                except BaseException as exc:
                    errors.append(exc)
        self._state_server = None
        self._ingress_server = None
        self.started = False
        if errors:
            raise QuackCommandFabricStateError(
                f"fabric failed clean stop: {type(errors[0]).__name__}"
            ) from errors[0]

    def __enter__(self) -> QuackCommandFabric:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


__all__ = [
    "MAX_COMMAND_BYTES",
    "MAX_ID_BYTES",
    "MAX_INGRESS_ROWS",
    "QUACK_COMMAND_FABRIC_INTERFACE",
    "QUACK_COMMAND_FABRIC_SCHEMA",
    "QUACK_DAEMON_CANONICAL_HANDLER_INTERFACE",
    "QUACK_DAEMON_OWNER_GATEWAY_INTERFACE",
    "QuackCommandCapabilityDecision",
    "QuackCommandCapabilityResult",
    "QuackCommandClient",
    "QuackCommandFabric",
    "QuackCommandFabricCapabilityError",
    "QuackCommandFabricError",
    "QuackCommandFabricStateError",
    "QuackCommandIngressError",
    "QuackDaemonOwnerGateway",
    "QuackPlanR2OwnerGateway",
    "QuackReadClient",
    "assess_quack_command_capability",
]
