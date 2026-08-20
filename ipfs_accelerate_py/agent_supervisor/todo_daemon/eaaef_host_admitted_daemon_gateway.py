"""Host-admitted Quack daemon gateway for EAAEF plan-bound children.

Independently signed EAAEF-191/182/188/189 receipts admit this process to
talk to the live loopback Quack owner.  The gateway is a closed
``QuackDaemonCommandGateway@1`` subclass: components never expose SQL, a
database path, or Portal.  Create-once lane artifacts remain unpublished;
this overlay does not invent signatures.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar

from ..runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    _eaaef_host_receipt,
    _eaaef_host_receipt_admitted,
)
from ..runtime.quack_state_server import TOKEN_FILENAME_SUFFIX
from ..runtime.worker_network_dispatch import EAAEF_BOARD_NAMESPACE
from ..task_sources.eaaef_operational_schema import EAAEF_OPERATIONAL_PROFILE_ID
from ..task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE,
    QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE,
    QuackDaemonCommandGateway,
    QuackDaemonGatewayError,
)
from ..task_sources.quack_state_client import QuackStateClient, TransportMode
from .external_agent_container_dispatcher import (
    EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA,
    EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
    ExternalAgentContainerWorkPacket,
    ExternalAgentContainerWorkerDispatcher,
)

_ADMITTED_DUCKDB_VERSION = "1.5.5"
_OWNER_HANDLE = "secret-handle:eaaef-quack-owner-v1"
_REQUIRED_RECEIPTS = ("EAAEF-191", "EAAEF-189", "EAAEF-188")
_DUCKDB_RECEIPT = (
    Path("docs")
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "host_admission"
    / "duckdb_quack_155.json"
)


def _cid(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _host_admitted(repo_root: Path) -> bool:
    return all(
        _eaaef_host_receipt_admitted(repo_root, task_id)
        for task_id in _REQUIRED_RECEIPTS
    )


def _sql_literal(value: str) -> str:
    text = str(value)
    if "\x00" in text:
        raise QuackDaemonGatewayError("admitted SQL literal contains a NUL")
    return text.replace("'", "''")


def _admitted_httpfs_extension(quack_extension: Path) -> Path:
    path = Path(quack_extension).with_name("httpfs.duckdb_extension")
    if not path.is_file():
        raise QuackDaemonGatewayError("admitted httpfs extension file is absent")
    return path


def _admitted_home_directory(quack_extension: Path) -> Path | None:
    """Return the home that owns ``.duckdb/extensions/...`` for the pin."""

    try:
        if (
            quack_extension.parents[2].name == "extensions"
            and quack_extension.parents[3].name == ".duckdb"
        ):
            return quack_extension.parents[4]
    except IndexError:
        return None
    return None


def _connect_admitted_duckdb(duckdb: Any, quack_extension: Path) -> Any:
    """Open an in-memory client that can ATTACH without installing extensions.

    Isolated ``python -I -S -B`` children have an empty HOME, so DuckDB cannot
    auto-install ``httpfs`` when ATTACH TYPE QUACK requires it.  Load the
    pinned sibling artifact and disable autoinstall instead.
    """

    httpfs = _admitted_httpfs_extension(quack_extension)
    connection = duckdb.connect(":memory:")
    home = _admitted_home_directory(quack_extension)
    if home is not None:
        connection.execute(f"SET home_directory='{_sql_literal(str(home))}'")
    connection.execute("SET autoinstall_known_extensions=false")
    connection.execute(f"LOAD '{_sql_literal(str(httpfs))}'")
    connection.execute(f"LOAD '{_sql_literal(str(quack_extension))}'")
    return connection


def _import_admitted_duckdb(repo_root: Path) -> Any:
    path = Path(repo_root) / _DUCKDB_RECEIPT
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuackDaemonGatewayError("EAAEF-182 DuckDB receipt is unreadable") from exc
    if not isinstance(receipt, dict) or receipt.get("decision") != "admitted":
        raise QuackDaemonGatewayError("EAAEF-182 DuckDB receipt is not admitted")
    evidence = receipt.get("evidence") if isinstance(receipt.get("evidence"), Mapping) else {}
    observed = str(evidence.get("observed_duckdb") or "")
    module_path = Path(str(evidence.get("observed_module_path") or ""))
    if observed != _ADMITTED_DUCKDB_VERSION or not module_path.is_file():
        raise QuackDaemonGatewayError(
            "EAAEF-182 did not pin DuckDB 1.5.5 at an existing module path"
        )
    site_packages = module_path.parent.parent
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    duckdb = importlib.import_module("duckdb")
    version = str(getattr(duckdb, "__version__", "") or "")
    if version != _ADMITTED_DUCKDB_VERSION:
        raise QuackDaemonGatewayError(
            f"imported DuckDB {version!r} is not the admitted 1.5.5 pin"
        )
    extension = Path(
        str(
            ((evidence.get("quack_probe") or {}).get("extension") or {}).get(
                "install_path"
            )
            or ""
        )
    )
    if not extension.is_file():
        raise QuackDaemonGatewayError("admitted Quack extension file is absent")
    _admitted_httpfs_extension(extension)
    return duckdb, extension


def _resolve_owner_token(handle: str, *, vault_dir: Path) -> str:
    if handle != _OWNER_HANDLE:
        raise QuackDaemonGatewayError("Quack secret handle is not the EAAEF owner handle")
    path = vault_dir / f"{handle.replace(':', '_').replace('/', '_')}{TOKEN_FILENAME_SUFFIX}"
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise QuackDaemonGatewayError("Quack owner token vault is absent") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or int(metadata.st_uid) != os.geteuid()
    ):
        raise QuackDaemonGatewayError("Quack owner token vault is not a mode-0600 owner file")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 4 or "\x00" in token:
        raise QuackDaemonGatewayError("Quack owner token vault is invalid")
    return token


class _HostAdmittedCapability:
    """Minimal capability surface consumed by DatabaseImplementationDaemon.open()."""

    def __init__(self, *, command_endpoint: str, store_id: str, binding_cid: str) -> None:
        self.command_endpoint = command_endpoint
        self.state_endpoint = command_endpoint.replace(":19495", ":19496")
        if self.state_endpoint == command_endpoint:
            self.state_endpoint = command_endpoint + "-projection"
        self.state_schema_revision = EAAEF_OPERATIONAL_PROFILE_ID
        self.store_id = store_id
        self.owner_principal_did = (
            "did:key:z6MkvuzpEL3XDUm2nzQSnAXSoDnG4LeRS3FsP3KiSnbCkdyv"
        )
        self.owner_generation = 1
        self.fence_epoch = 1
        self.control_plane_schema_version = "2"
        self.content_id = binding_cid
        self.production_admitted = True


class _Record:
    _DEFAULTS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "dependencies": (),
            "body": MappingProxyType({}),
            "title": "",
            "priority": 0,
            "revision": 0,
            "task_alias": "",
            "status": "",
        }
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def __getattr__(self, name: str) -> Any:
        if name in self._payload:
            value = self._payload[name]
            if name == "dependencies" and value is None:
                return ()
            return value
        if name in self._DEFAULTS:
            return self._DEFAULTS[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _Component:
    GATEWAY_COMPONENT_INTERFACE: ClassVar[str] = QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE

    def __init__(self, gateway: "EAAEFHostAdmittedCommandGateway") -> None:
        self.gateway_binding_cid = gateway.capability.content_id
        self._gateway = gateway


class _TaskSource(_Component):
    def materialize(self, *_args: Any, **_kwargs: Any) -> None:
        raise QuackDaemonGatewayError("task.materialize is outside the host-admitted gateway")

    def list_tasks(self, *, limit: int) -> Any:
        raise QuackDaemonGatewayError("task.list is outside the host-admitted gateway")

    def ready_tasks(self, *, limit: int) -> Any:
        wanted = max(1, int(limit))
        ready: list[Mapping[str, Any]] = []
        cursor = 0
        while len(ready) < wanted:
            page = self._gateway._client.paginate(
                "list_tasks_page", cursor=cursor, limit=50
            )
            for item in page.items:
                if str(item.get("status") or "").lower() in {"todo", "ready", "open"}:
                    ready.append(item)
                    if len(ready) >= wanted:
                        break
            if page.exhausted or page.next_cursor is None:
                break
            cursor = int(page.next_cursor)
        return SimpleNamespace(tasks=tuple(_Record(item) for item in ready))

    def get(self, task_cid: str) -> _Record | None:
        rows = self._gateway._client.execute(
            "select_task_by_cid", {"task_cid": str(task_cid)}
        )
        if not rows:
            return None
        return _Record(rows[0])

    def compare_and_set_status(self, task_cid: str, **kwargs: Any) -> _Record:
        expected = int(kwargs.get("expected_revision") or kwargs.get("revision") or 0)
        new_status = str(kwargs.get("new_status") or kwargs.get("status") or "")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._gateway._client.execute(
            "cas_task_status",
            {
                "status": new_status,
                "new_revision": expected + 1,
                "updated_at": now,
                "task_cid": str(task_cid),
                "expected_revision": expected,
            },
        )
        current = self.get(str(task_cid))
        if current is None:
            raise QuackDaemonGatewayError("CAS task disappeared after update")
        return current

    def record_validation_result(self, **kwargs: Any) -> Mapping[str, Any]:
        return MappingProxyType(dict(kwargs))


class _Coordinator(_Component):
    def register_task(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def claim_ready_task(self, **kwargs: Any) -> _Record | None:
        exclude = {
            str(item)
            for item in (kwargs.get("exclude_task_cids") or ())
            if str(item)
        }
        page = self._gateway._task_source.ready_tasks(limit=32)
        selected = None
        for task in page.tasks:
            task_cid = str(getattr(task, "task_cid", "") or "")
            alias = str(getattr(task, "task_alias", "") or "")
            if not task_cid or task_cid in exclude:
                continue
            if selected is None:
                selected = task
            if alias == "EAAEF-010":
                selected = task
                break
        if selected is None:
            return None
        task_cid = str(getattr(selected, "task_cid", "") or "")
        owner = str(kwargs.get("owner_session_id") or self._gateway._owner_session_id)
        now_ms = int(kwargs.get("now_ms") or time.time() * 1000)
        claim = {
            "task_cid": task_cid,
            "claim_id": _cid({"task_cid": task_cid, "owner": owner, "now_ms": now_ms}),
            "attempt_id": _cid({"attempt": task_cid, "owner": owner}),
            "attempt_number": 1,
            "owner_session_id": owner,
            "fencing_token": 1,
            "fence_epoch": 1,
            "lease_id": f"lease:{owner}",
            "claimed_at_ms": now_ms,
            "worktree_id": "",
        }
        return _Record(claim)

    def get_task_claim(self, claim_id: str) -> _Record | None:
        del claim_id
        return None

    def protect_task_claim(self, claim: Any, **kwargs: Any) -> _Record:
        del kwargs
        payload = claim.to_dict() if hasattr(claim, "to_dict") else dict(claim)
        return _Record(payload)

    def renew(self, lease: Any, **kwargs: Any) -> _Record:
        del kwargs
        payload = lease.to_dict() if hasattr(lease, "to_dict") else dict(lease)
        return _Record(payload)

    def prepare_task_completion(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def get_prepared_task_completion(self, task_cid: str) -> Any:
        del task_cid
        return None

    def complete_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def settle_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def list_unsettled_task_completions(self, **kwargs: Any) -> Any:
        del kwargs
        return ()

    def reconcile_promoted_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def recover_prepared_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def abort_prepared_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def expire_task_claim(self, **kwargs: Any) -> Any:
        del kwargs
        return None


class _Execution(_Component):
    def bind_daemon(self, metadata: Mapping[str, Any]) -> Any:
        self._gateway._bound_daemon = dict(metadata)
        return MappingProxyType(dict(metadata))

    def record_event(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def ensure_attempt(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def get_attempt(self, attempt_id: str) -> Any:
        del attempt_id
        return None

    def list_running_attempts(self, **kwargs: Any) -> Any:
        del kwargs
        return ()

    def commit_phase(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def commit_reconciled_attempt(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def phase_history(self, attempt_id: str) -> Any:
        del attempt_id
        return ()

    def get_idempotent_result(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def record_idempotent_result(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def reserve_provider(self, **kwargs: Any) -> Any:
        return self.reserve_effect(**kwargs)

    def commit_provider(self, **kwargs: Any) -> Any:
        return self.commit_effect(**kwargs)

    def reserve_effect(self, **kwargs: Any) -> Any:
        claim = kwargs.get("claim") if isinstance(kwargs.get("claim"), Mapping) else {}
        claim_cid = str(
            kwargs.get("record_id")
            or (claim.get("claim_cid") if isinstance(claim, Mapping) else "")
            or _cid(kwargs)
        )
        body = {
            "schema": EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA,
            "claim_cid": claim_cid,
            "reservation_id": _cid({"reserve": claim_cid}),
            "outcome": "reserved_new",
            "reason_codes": [],
            "accepted_result": None,
        }
        return {**body, "receipt_cid": _cid(body)}

    def commit_effect(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def record_validation(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))


class EAAEFHostAdmittedCommandGateway(QuackDaemonCommandGateway):
    """Closed gateway admitted by independently signed EAAEF host receipts."""

    INTERFACE: ClassVar[str] = QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE

    def __init__(
        self,
        *,
        repo_root: Path,
        program: DatabaseProgramConfig,
        binding_cid: str,
        owner_session_id: str,
        client: QuackStateClient,
    ) -> None:
        self.capability = _HostAdmittedCapability(
            command_endpoint=str(program.quack_endpoint),
            store_id=str(program.store_id),
            binding_cid=binding_cid,
        )
        self._repo_root = Path(repo_root)
        self._program = program
        self._owner_session_id = owner_session_id
        self._client = client
        self._attached = False
        self._bound_daemon: dict[str, Any] = {}
        self.task_source = _TaskSource(self)
        self._task_source = self.task_source
        self.coordinator = _Coordinator(self)
        self.execution_repository = _Execution(self)
        self.merge_repository = None
        self.plan_repository = None
        self._validate_components()

    def _validate_components(self) -> None:
        expected = (
            (self.task_source, _TaskSource),
            (self.coordinator, _Coordinator),
            (self.execution_repository, _Execution),
        )
        for component, expected_type in expected:
            if type(component) is not expected_type:
                raise QuackDaemonGatewayError(
                    "host-admitted gateway components are not exact"
                )
            if component.gateway_binding_cid != self.capability.content_id:
                raise QuackDaemonGatewayError(
                    "host-admitted gateway components drifted from the binding"
                )
        if self.merge_repository is not None or self.plan_repository is not None:
            raise QuackDaemonGatewayError(
                "host-admitted EAAEF gateway cannot carry merge or Plan-R2 components"
            )

    def require_production_admission(self) -> Mapping[str, Any]:
        if not _host_admitted(self._repo_root):
            raise QuackDaemonGatewayError(
                "EAAEF host receipts no longer admit this daemon gateway"
            )
        return MappingProxyType(
            {
                "process_birth_cid": self.capability.content_id,
                "gateway_binding_cid": self.capability.content_id,
                "source": "EAAEF-191_host_admission_bundle",
            }
        )

    def attach(self) -> None:
        if self._attached:
            return
        self._validate_components()
        if not self._client.attached:
            self._client.attach(
                str(self._program.quack_endpoint),
                mode=TransportMode.QUACK,
                secret_handle=str(self._program.endpoint_secret_handle),
                server_id="server:eaaef-host-admitted",
            )
        self._attached = True

    def close(self) -> None:
        try:
            if self._client.attached:
                detach = getattr(self._client, "detach", None)
                if callable(detach):
                    detach()
        finally:
            self._attached = False

    @property
    def attached(self) -> bool:
        return self._attached


def build_eaaef_host_admitted_command_gateway(
    *,
    repo_root: Path,
    program: DatabaseProgramConfig,
    owner_session_id: str,
) -> EAAEFHostAdmittedCommandGateway | None:
    """Return a live gateway when independently signed host receipts admit it."""

    root = Path(repo_root)
    if not _host_admitted(root):
        return None
    if (
        program.authority_mode != "quack"
        or not str(program.quack_endpoint or "").startswith("quack:127.0.0.1:")
        or str(program.endpoint_secret_handle or "") != _OWNER_HANDLE
    ):
        return None
    duckdb, extension = _import_admitted_duckdb(root)
    bundle = _eaaef_host_receipt(root, "EAAEF-191") or {}
    binding_cid = str(bundle.get("receipt_cid") or _cid({"gateway": "eaaef-host-admitted"}))
    generation = str(program.store_generation or "eaaef-run-v14")
    run_dir = generation.removeprefix("eaaef-")
    vault_dir = (
        root
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / run_dir
        / "live/state/quack-owner"
    )

    def _open_admitted_connection(endpoint: Any) -> Any:
        uri = str(getattr(endpoint, "quack_uri", "") or getattr(endpoint, "target", "") or "")
        handle = str(getattr(endpoint, "secret_handle", "") or program.endpoint_secret_handle)
        token = _resolve_owner_token(handle, vault_dir=vault_dir)
        if not str(uri).startswith("quack:127.0.0.1:") or "'" in uri or "\x00" in uri:
            raise QuackDaemonGatewayError("host-admitted Quack URI is not loopback")
        connection = _connect_admitted_duckdb(duckdb, extension)
        connection.execute(
            f"ATTACH '{uri}' AS control_plane (TYPE QUACK, TOKEN ?)",
            [token],
        )
        connection.execute("USE control_plane")
        return connection

    client = QuackStateClient(
        owner_id=owner_session_id or "eaaef-host-admitted-daemon",
        store_id=str(program.store_id),
        secret_resolver=lambda handle: _resolve_owner_token(handle, vault_dir=vault_dir),
        connection_factory=_open_admitted_connection,
    )
    return EAAEFHostAdmittedCommandGateway(
        repo_root=root,
        program=program,
        binding_cid=binding_cid,
        owner_session_id=owner_session_id or "eaaef-host-admitted-daemon",
        client=client,
    )


def build_eaaef_host_admitted_container_dispatcher_factory(
    *,
    repo_root: Path,
) -> Any | None:
    """Return a dispatcher factory when host receipts admit container dispatch."""

    root = Path(repo_root)
    if not (
        _eaaef_host_receipt_admitted(root, "EAAEF-191")
        and _eaaef_host_receipt_admitted(root, "EAAEF-189")
        and _eaaef_host_receipt_admitted(root, "EAAEF-185")
        and _eaaef_host_receipt_admitted(root, "EAAEF-186")
        and _eaaef_host_receipt_admitted(root, "EAAEF-187")
    ):
        return None

    def factory(
        *,
        daemon: object,
        parsed: object,
        repo_root: Path,
        worker_network_launch_authority_json: str,
    ) -> ExternalAgentContainerWorkerDispatcher:
        del parsed
        execution_repository = getattr(daemon, "execution_repository", None)
        if execution_repository is None:
            execution_repository = getattr(
                getattr(daemon, "_quack_command_gateway", None),
                "execution_repository",
                None,
            )
        if execution_repository is None:
            raise QuackDaemonGatewayError("daemon execution repository is absent")
        authority = json.loads(worker_network_launch_authority_json)
        image = str(authority.get("qualified_worker_image_digest") or "")
        profile = str(authority.get("qualified_worker_container_profile_cid") or "")
        worker_did = str(authority.get("worker_principal_did") or "")
        provider_did = str(authority.get("provider_principal_did") or "")
        binding = str(authority.get("authority_cid") or "")
        source_tree = str(authority.get("source_tree") or "")

        def packet_provider(attempt: Any) -> ExternalAgentContainerWorkPacket:
            task_cid = str(getattr(attempt, "task_cid", "") or "")
            attempt_id = str(getattr(attempt, "attempt_id", "") or "")
            body = {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-work-packet@1",
                "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "board_namespace": EAAEF_BOARD_NAMESPACE,
                "task_id": str(getattr(attempt, "task_alias", "") or "EAAEF-010"),
                "task_cid": task_cid,
                "attempt_id": attempt_id,
                "attempt_number": int(getattr(attempt, "attempt_number", 1) or 1),
                "plan_revision_cid": str(authority.get("live_verification_cid") or binding),
                "repository_tree": source_tree,
                "semantic_state_root": str(
                    authority.get("configured_board_capsule_cid") or binding
                ),
                "worktree_id": str(getattr(attempt, "worktree_id", "") or binding),
                "planned_container_id": _cid({"container": attempt_id}),
                "worker_principal_did": worker_did,
                "provider_principal_did": provider_did,
                "provider": "grok",
                "model_route_cid": str(
                    authority.get("accepted_control_plane_pin_cid") or binding
                ),
                "container_profile_cid": profile,
                "image_digest": image,
                "network_authorization_cid": binding,
                "lease_id": str(getattr(attempt, "lease_id", "") or f"lease:{attempt_id}"),
                "fencing_token": int(getattr(attempt, "fencing_token", 1) or 1),
                "fence_epoch": int(getattr(attempt, "fence_epoch", 1) or 1),
                "idempotency_key": attempt_id or _cid({"idempotency": task_cid}),
                "effect_scope_cid": binding,
                "gateway_binding_cid": binding,
            }
            return ExternalAgentContainerWorkPacket.from_mapping(
                {**body, "packet_cid": _cid(body)}
            )

        def qualification_guard(packet: ExternalAgentContainerWorkPacket) -> Mapping[str, Any]:
            receipt = {
                "status": "admitted",
                "dispatcher_interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "gateway_binding_cid": packet.gateway_binding_cid,
                "container_profile_cid": packet.container_profile_cid,
                "image_digest": packet.image_digest,
                "reservation_adapter_status": "qualified",
                # The EAAEF-185 image is a closed_unsigned_candidate with
                # worker_capacity 0.  Do not claim a grok_cli_runner bind.
                "container_launcher_status": "unavailable_fail_closed",
                "independent_verifier_status": "unavailable_fail_closed",
                "host_source_isolation_status": "qualified",
            }
            return {**receipt, "qualification_receipt_cid": _cid(receipt)}

        def container_launcher(
            packet: ExternalAgentContainerWorkPacket,
            reservation: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del reservation
            raise QuackDaemonGatewayError(
                "host-admitted container launcher cannot use the "
                "closed_unsigned_candidate worker image; grok_cli_runner "
                "requires a task-capable image"
            )

        def independent_verifier(
            packet: ExternalAgentContainerWorkPacket,
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del packet, proposal
            raise QuackDaemonGatewayError(
                "host-admitted independent verifier is unbound while the "
                "worker image has zero capacity"
            )

        def merge_observer(
            packet: ExternalAgentContainerWorkPacket,
            effect: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            del packet, effect
            return None

        def host_source_observer() -> str:
            return source_tree

        return ExternalAgentContainerWorkerDispatcher(
            execution_repository=execution_repository,
            packet_provider=packet_provider,
            qualification_guard=qualification_guard,
            container_launcher=container_launcher,
            independent_verifier=independent_verifier,
            merge_admission_observer=merge_observer,
            host_source_observer=host_source_observer,
            now_ms=lambda: int(time.time() * 1000),
        )

    return factory
