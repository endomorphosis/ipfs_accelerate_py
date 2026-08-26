"""Guarded offline-population prefix for the future EAAEF CASF owner.

This module deliberately stops before Plan R2.  A code-bound CASF lifecycle
must hold its real exclusive owner lease continuously while the adapter writes
the fresh population and provisionally hands that same lease to the new owner.
The guard commits that provisional owner only after the final durable registry
record.  Every post-start failure requests an exact abort first.

The final registry write and process commit cannot be one OS transaction.  If
commit and subsequent abort both fail, the generation is permanently abandoned
for operator recovery: its monotonic record is never rolled back or reused.
The registry is lifecycle evidence, not live-owner or task-status authority.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol

EAAEF_RECONCILIATION_OWNER_INTERFACE: Final = "EAAEFTypedReconciliationOwner@1"
EAAEF_OWNER_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-typed-owner-qualification@1"
)
DATABASE_TASK_SOURCE_INTERFACE: Final = "DatabaseTaskSource@1"
PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS: Final = (
    "source_complete_external_signed_channel_required"
)
PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS: Final = (
    "external_plan_r2_remote_owner_capability_absent",
    "qualified_process_remote_wire_channel_factory_absent",
    "supervisor_plan_r2_remote_repository_wiring_absent",
)
EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS: Final = (
    "casf_quack_exclusive_owner_lifecycle_not_bound",
    "typed_database_task_source_runtime_adapter_not_bound",
    *PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS,
    "owner_status_snapshot_not_bound",
    "exact_birth_stop_tracks_not_bound",
    "supervisor_launch_not_bound",
)
EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE: Final = (
    "EAAEFCASFBootstrapOwnerLifecycle@1"
)
EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE: Final = (
    "EAAEFCASFBootstrapOwnerGuard@1"
)
EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS: Final = (
    "persistent_quack_handoff_source_complete_cutover_unqualified"
)
EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-absence-attestation@1"
)
EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-start-receipt@1"
)
EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-abort-receipt@1"
)
EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-commit-receipt@1"
)
EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-bootstrap-registry-record@1"
)

_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGISTRY_FIELDS: Final = frozenset(
    {
        "schema",
        "generation_id",
        "phase",
        "request_cid",
        "source_forest_root",
        "population_cid",
        "owner_lifecycle_interface",
        "absence_attestation_cid",
        "offline_materialization_receipt_cid",
        "owner_start_receipt_cid",
        "canonical_bootstrap_receipt_cid",
        "owner_process_birth",
        "record_cid",
    }
)
_ABSENCE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "source_forest_root",
        "owner_absent",
        "exclusive_owner_lease_held",
        "observed_owner_process_birth",
        "attestation_cid",
    }
)
_START_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "source_forest_root",
        "population_cid",
        "absence_attestation_cid",
        "offline_materialization_receipt_cid",
        "owner_started_after_bootstrap",
        "exclusive_owner_lease_handoff_complete",
        "owner_start_commit_pending",
        "provider_process_started",
        "owner_process_birth",
        "bootstrap_snapshot",
        "start_receipt_cid",
    }
)
_ABORT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "owner_start_receipt_cid",
        "abort_reason_code",
        "owner_abort_completed",
        "remaining_started_owner_count",
        "owner_process_birth",
        "owner_process_alive",
        "task_state_mutated",
        "abort_receipt_cid",
    }
)
_COMMIT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "owner_start_receipt_cid",
        "final_record_cid",
        "owner_commit_completed",
        "owner_process_birth",
        "owner_process_alive",
        "provider_process_started",
        "commit_receipt_cid",
    }
)
_PROCESS_BIRTH_FIELDS: Final = frozenset(
    {"pid", "start_time_ticks", "parent_pid", "boot_id", "argv_sha256"}
)
_REGISTRY_CAPABILITY_TOKEN = object()


class EAAEFCASFBootstrapOwnerError(RuntimeError):
    """The bounded CASF bootstrap transition rejected work."""


@dataclass(frozen=True, slots=True)
class EAAEFCASFBootstrapBinding:
    """Owner-local authority; paths never cross the public owner protocol."""

    generation_id: str
    source_head: str
    source_tree: str
    source_forest_root: str
    board_cid: str
    population_cid: str
    bootstrap_population_cid: str
    plan_r1_cid: str
    database_path: Path
    owner_state_dir: Path


class EAAEFCASFBootstrapOwnerGuard(Protocol):
    """One held CASF lease spanning offline commit and provisional owner birth.

    ``start_after_offline_commit`` must either return a receipt for every
    provisional owner it created or internally abort before raising.  Abort is
    idempotent and remains available after a partially failed commit.  Before
    ``commit_started_owner`` succeeds, the provisional owner must be unable to
    mutate task state and must lose its lease or terminate if the guard process
    exits without an explicit commit.  That crash coupling is part of the real
    CASF/Quack implementation contract; this adapter cannot synthesize it.
    """

    INTERFACE: str

    def owner_absence_attestation(self) -> Mapping[str, Any]: ...

    def start_after_offline_commit(
        self,
        *,
        offline_materialization_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def abort_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any] | None,
        reason_code: str,
    ) -> Mapping[str, Any]: ...

    def commit_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any],
        final_record_cid: str,
    ) -> Mapping[str, Any]: ...


class EAAEFCASFBootstrapOwnerLifecycle(Protocol):
    """Code-bound factory for the continuous exclusive-owner guard."""

    INTERFACE: str

    def hold_exclusive_bootstrap(
        self,
        binding: EAAEFCASFBootstrapBinding,
    ) -> AbstractContextManager[EAAEFCASFBootstrapOwnerGuard]: ...


def _verified(
    raw: Mapping[str, Any],
    *,
    schema: str,
    cid_field: str,
    fields: frozenset[str],
    noun: str,
) -> dict[str, Any]:
    from ..runtime.eaaef_reconciliation_lifecycle import _cid

    value = dict(raw)
    body = dict(value)
    observed_cid = body.pop(cid_field, "")
    if (
        set(value) != fields
        or value.get("schema") != schema
        or type(observed_cid) is not str
        or observed_cid != _cid(body)
    ):
        raise EAAEFCASFBootstrapOwnerError(f"{noun} identity differs")
    return value


def _exact_process_birth(raw: object) -> Any:
    from ..runtime import eaaef_reconciliation_lifecycle as reconciliation

    if type(raw) is not dict or set(raw) != _PROCESS_BIRTH_FIELDS:
        raise EAAEFCASFBootstrapOwnerError("CASF owner process birth shape differs")
    integer_fields = ("pid", "start_time_ticks", "parent_pid")
    if any(type(raw[name]) is not int for name in integer_fields) or any(
        type(raw[name]) is not str for name in ("boot_id", "argv_sha256")
    ):
        raise EAAEFCASFBootstrapOwnerError("CASF owner process birth types differ")
    return reconciliation.ProcessBirth.from_mapping(raw)


class _HeldRegistryWriteCapability:
    """Unexported identity proving this registry's local and file locks are held."""

    __slots__ = ("_active", "_descriptor", "_registry")

    def __init__(
        self,
        token: object,
        *,
        registry: EAAEFCASFBootstrapRegistry,
        descriptor: int,
    ) -> None:
        if token is not _REGISTRY_CAPABILITY_TOKEN:
            raise TypeError("registry capabilities are issued only under the held lock")
        self._registry = registry
        self._descriptor = descriptor
        self._active = True


class EAAEFCASFBootstrapRegistry:
    """Private crash-fail-closed registry; never a task-status authority."""

    def __init__(self, root: str | Path) -> None:
        from ..runtime.eaaef_reconciliation_lifecycle import ReconciliationStateStore

        raw = Path(root).expanduser()
        for candidate in (raw, *raw.parents):
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise EAAEFCASFBootstrapOwnerError(
                    "CASF bootstrap registry path contains a symlink"
                )
        self._store = ReconciliationStateStore(raw)
        self.root = self._store.root
        self.lock_path = self.root / ".bootstrap-owner.lock"
        self._local_lock = threading.Lock()
        self._active_capability: _HeldRegistryWriteCapability | None = None

    def generation_dir(self, generation_id: str) -> Path:
        return self._store.generation_dir(generation_id)

    @staticmethod
    def _sync_private_directory(path: Path, *, noun: str) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise EAAEFCASFBootstrapOwnerError(f"{noun} is unsafe") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise EAAEFCASFBootstrapOwnerError(f"{noun} is unsafe")
            os.fsync(descriptor)
            path_metadata = os.lstat(path)
            if (
                stat.S_ISLNK(path_metadata.st_mode)
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise EAAEFCASFBootstrapOwnerError(f"{noun} changed while syncing")
        except OSError as exc:
            raise EAAEFCASFBootstrapOwnerError(f"{noun} is not durable") from exc
        finally:
            os.close(descriptor)

    @contextmanager
    def exclusive(self) -> Iterator[_HeldRegistryWriteCapability]:
        with self._local_lock:
            self._store.initialize()
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.lock_path, flags, 0o600)
            capability: _HeldRegistryWriteCapability | None = None
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                ):
                    raise EAAEFCASFBootstrapOwnerError("CASF bootstrap lock is unsafe")
                os.fchmod(descriptor, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                capability = _HeldRegistryWriteCapability(
                    _REGISTRY_CAPABILITY_TOKEN,
                    registry=self,
                    descriptor=descriptor,
                )
                self._active_capability = capability
                yield capability
            finally:
                self._active_capability = None
                if capability is not None:
                    capability._active = False  # noqa: SLF001
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _require_capability(self, capability: object) -> _HeldRegistryWriteCapability:
        if (
            type(capability) is not _HeldRegistryWriteCapability
            or capability is not self._active_capability
            or capability._registry is not self  # noqa: SLF001
            or capability._active is not True  # noqa: SLF001
        ):
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap registry write requires its held-lock capability"
            )
        descriptor_metadata = os.fstat(capability._descriptor)  # noqa: SLF001
        path_metadata = os.stat(self.lock_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap held-lock capability lost its sealed lock file"
            )
        return capability

    def prepare_generation(self, capability: object, generation_id: str) -> Path:
        self._require_capability(capability)
        directory = self.generation_dir(generation_id)
        directory.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = os.lstat(directory.parent)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap parent is unsafe")
        os.chmod(directory.parent, 0o700)
        self._sync_private_directory(
            directory.parent,
            noun="CASF bootstrap generations parent",
        )
        try:
            directory.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap generation already has durable state"
            ) from exc
        os.chmod(directory, 0o700)
        self._sync_private_directory(directory, noun="CASF bootstrap generation")
        self._sync_private_directory(
            directory.parent,
            noun="CASF bootstrap generations parent",
        )
        return directory

    @staticmethod
    def _validate_phase_record(value: Mapping[str, Any]) -> None:
        phase = value.get("phase")
        cid_fields = (
            "absence_attestation_cid",
            "offline_materialization_receipt_cid",
            "owner_start_receipt_cid",
            "canonical_bootstrap_receipt_cid",
        )
        if phase not in {"absent", "offline_committed", "owner_started"} or any(
            type(value.get(name)) is not str for name in cid_fields
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap record phase differs")
        absence, offline, started, canonical = (str(value[name]) for name in cid_fields)
        birth = value.get("owner_process_birth")
        if not _SHA256_RE.fullmatch(absence):
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap absence CID is invalid")
        if phase == "absent":
            valid = not offline and not started and not canonical and birth is None
        elif phase == "offline_committed":
            valid = (
                _SHA256_RE.fullmatch(offline) is not None
                and not started
                and not canonical
                and birth is None
            )
        else:
            valid = all(
                _SHA256_RE.fullmatch(item) is not None
                for item in (offline, started, canonical)
            )
            if valid:
                _exact_process_birth(birth)
        if not valid:
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap record payload differs")

    def _read_record(self, generation_id: str) -> dict[str, Any] | None:
        from ..runtime.eaaef_reconciliation_lifecycle import _private_json_object

        path = self.generation_dir(generation_id) / "bootstrap-owner.json"
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        value = _verified(
            _private_json_object(path, noun="CASF bootstrap registry record"),
            schema=EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
            cid_field="record_cid",
            fields=_REGISTRY_FIELDS,
            noun="CASF bootstrap registry record",
        )
        self._validate_phase_record(value)
        return value

    def write_record(
        self,
        capability: object,
        generation_id: str,
        record: Mapping[str, Any],
    ) -> None:
        self._require_capability(capability)
        value = _verified(
            record,
            schema=EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
            cid_field="record_cid",
            fields=_REGISTRY_FIELDS,
            noun="CASF bootstrap registry record",
        )
        if value.get("generation_id") != generation_id:
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap record state differs")
        self._validate_phase_record(value)
        previous = self._read_record(generation_id)
        expected_previous = {
            "absent": None,
            "offline_committed": "absent",
            "owner_started": "offline_committed",
        }[str(value["phase"])]
        observed_previous = None if previous is None else previous.get("phase")
        if observed_previous != expected_previous:
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap registry transition is not monotonic"
            )
        continuity_fields = (
            "generation_id",
            "request_cid",
            "source_forest_root",
            "population_cid",
            "owner_lifecycle_interface",
            "absence_attestation_cid",
        )
        if previous is not None and any(
            value.get(name) != previous.get(name) for name in continuity_fields
        ):
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap registry transition identity differs"
            )
        if (
            previous is not None
            and previous.get("phase") == "offline_committed"
            and value.get("offline_materialization_receipt_cid")
            != previous.get("offline_materialization_receipt_cid")
        ):
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap offline commit identity changed"
            )
        self._store._write_private_json(  # noqa: SLF001
            self.generation_dir(generation_id) / "bootstrap-owner.json",
            value,
        )
        self._sync_private_directory(
            self.generation_dir(generation_id),
            noun="CASF bootstrap generation",
        )
        if self._read_record(generation_id) != value:
            raise EAAEFCASFBootstrapOwnerError(
                "CASF bootstrap durable record postcondition differs"
            )


class CASFBootstrapEAAEFTypedReconciliationOwner:
    """Executable guarded bootstrap prefix; every later effect fails closed."""

    INTERFACE: ClassVar[str] = EAAEF_RECONCILIATION_OWNER_INTERFACE

    def __init__(
        self,
        *,
        repo_root: Path,
        registry: EAAEFCASFBootstrapRegistry,
        source_forest_root: str,
        owner_lifecycle: EAAEFCASFBootstrapOwnerLifecycle,
    ) -> None:
        self._repo_root = Path(repo_root).resolve(strict=True)
        self._registry = registry
        self._source_forest_root = str(source_forest_root or "")
        self._owner_lifecycle = owner_lifecycle
        self._persistent_quack_handoff_bound = (
            getattr(owner_lifecycle, "QUALIFICATION_STATUS", "")
            == EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS
        )
        if (
            not self._repo_root.is_dir()
            or not _SHA256_RE.fullmatch(self._source_forest_root)
            or getattr(owner_lifecycle, "INTERFACE", "")
            != EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
            or not callable(getattr(owner_lifecycle, "hold_exclusive_bootstrap", None))
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF bootstrap binding is invalid")

    def reconciliation_qualification(self) -> Mapping[str, Any]:
        from ..runtime.eaaef_reconciliation_lifecycle import _cid

        blockers = list(EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS)
        if self._persistent_quack_handoff_bound:
            blockers.remove("casf_quack_exclusive_owner_lifecycle_not_bound")
        value: dict[str, Any] = {
            "schema": EAAEF_OWNER_QUALIFICATION_SCHEMA,
            "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "source_forest_root": self._source_forest_root,
            "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
            "bootstrap_materialization_before_owner_start": True,
            "offline_population_includes_execution_contracts": True,
            "direct_database_mutation_after_owner_start": False,
            "typed_task_source_interface": DATABASE_TASK_SOURCE_INTERFACE,
            "plan_r2_repository_interface": "",
            "plan_r2_remote_gateway_interface": "",
            "plan_r2_wire_channel_interface": "",
            "plan_r2_remote_runtime_qualification_status": (
                PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS
            ),
            "plan_r2_remote_runtime_blockers": blockers,
            "status_operation": "unavailable",
            "stop_tracks_operation": "unavailable",
            "launch_modes": [],
            "database_authority_crossing_allowed": False,
            "filesystem_path_authority_crossing_allowed": False,
            "transport_token_authority_crossing_allowed": False,
            "sql_crossing_allowed": False,
            "provider_launch_allowed": False,
        }
        value["qualification_cid"] = _cid(value)
        return value

    @staticmethod
    def _blocked() -> None:
        raise EAAEFCASFBootstrapOwnerError(
            "CASF bootstrap is bound but later effects remain unavailable: "
            + ", ".join(EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS)
        )

    def _absence(
        self,
        raw: Mapping[str, Any],
        binding: EAAEFCASFBootstrapBinding,
    ) -> dict[str, Any]:
        value = _verified(
            raw,
            schema=EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
            cid_field="attestation_cid",
            fields=_ABSENCE_FIELDS,
            noun="CASF owner absence attestation",
        )
        expected_strings = {
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "source_forest_root": binding.source_forest_root,
        }
        if (
            any(
                type(value.get(name)) is not str or value.get(name) != selected
                for name, selected in expected_strings.items()
            )
            or type(value.get("owner_absent")) is not bool
            or value.get("owner_absent") is not True
            or type(value.get("exclusive_owner_lease_held")) is not bool
            or value.get("exclusive_owner_lease_held") is not True
            or value.get("observed_owner_process_birth") is not None
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF owner absence evidence differs")
        return value

    def _started(
        self,
        raw: Mapping[str, Any],
        binding: EAAEFCASFBootstrapBinding,
        population: object,
        absence: Mapping[str, Any],
        offline: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from ..runtime import eaaef_reconciliation_lifecycle as reconciliation

        value = _verified(
            raw,
            schema=EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
            cid_field="start_receipt_cid",
            fields=_START_FIELDS,
            noun="CASF owner start receipt",
        )
        expected_strings = {
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "source_forest_root": binding.source_forest_root,
            "population_cid": binding.population_cid,
            "absence_attestation_cid": absence["attestation_cid"],
            "offline_materialization_receipt_cid": offline["receipt_cid"],
        }
        true_fields = (
            "owner_started_after_bootstrap",
            "exclusive_owner_lease_handoff_complete",
            "owner_start_commit_pending",
        )
        if (
            any(
                type(value.get(name)) is not str or value.get(name) != selected
                for name, selected in expected_strings.items()
            )
            or any(
                type(value.get(name)) is not bool or value.get(name) is not True
                for name in true_fields
            )
            or type(value.get("provider_process_started")) is not bool
            or value.get("provider_process_started") is not False
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF owner start evidence differs")
        birth = _exact_process_birth(value.get("owner_process_birth"))
        if reconciliation.inspect_process_birth(birth.pid) != birth:
            raise EAAEFCASFBootstrapOwnerError("CASF owner birth is not corroborated")
        snapshot = value.get("bootstrap_snapshot")
        if type(snapshot) is not dict:
            raise EAAEFCASFBootstrapOwnerError("CASF owner snapshot is absent")
        snapshot = dict(snapshot)
        reconciliation.build_unsigned_fresh_plan_r2_statement(
            population=population,
            bootstrap_snapshot=snapshot,
        )
        reconciliation._assert_no_boundary_authority(value)  # noqa: SLF001
        return value, snapshot

    @staticmethod
    def _claimed_start_receipt_cid(start_receipt: Mapping[str, Any] | None) -> str:
        if start_receipt is None:
            return ""
        value = start_receipt.get("start_receipt_cid")
        return value if type(value) is str and _SHA256_RE.fullmatch(value) else ""

    def _abort_started(
        self,
        guard: EAAEFCASFBootstrapOwnerGuard,
        binding: EAAEFCASFBootstrapBinding,
        start_receipt: Mapping[str, Any] | None,
        *,
        reason_code: str,
    ) -> dict[str, Any]:
        from ..runtime import eaaef_reconciliation_lifecycle as reconciliation

        expected_start_cid = self._claimed_start_receipt_cid(start_receipt)
        try:
            raw = guard.abort_started_owner(
                start_receipt=start_receipt,
                reason_code=reason_code,
            )
            if not isinstance(raw, Mapping):
                raise EAAEFCASFBootstrapOwnerError(
                    "CASF owner abort receipt is malformed"
                )
            value = _verified(
                raw,
                schema=EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
                cid_field="abort_receipt_cid",
                fields=_ABORT_FIELDS,
                noun="CASF owner abort receipt",
            )
            expected_strings = {
                "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
                "generation_id": binding.generation_id,
                "owner_start_receipt_cid": expected_start_cid,
                "abort_reason_code": reason_code,
            }
            if (
                any(
                    type(value.get(name)) is not str or value.get(name) != selected
                    for name, selected in expected_strings.items()
                )
                or type(value.get("owner_abort_completed")) is not bool
                or value.get("owner_abort_completed") is not True
                or type(value.get("remaining_started_owner_count")) is not int
                or value.get("remaining_started_owner_count") != 0
                or type(value.get("owner_process_alive")) is not bool
                or value.get("owner_process_alive") is not False
                or type(value.get("task_state_mutated")) is not bool
                or value.get("task_state_mutated") is not False
            ):
                raise EAAEFCASFBootstrapOwnerError(
                    "CASF owner abort evidence differs"
                )
            abort_birth_raw = value.get("owner_process_birth")
            abort_birth = (
                None if abort_birth_raw is None else _exact_process_birth(abort_birth_raw)
            )
            expected_birth = None
            if start_receipt is not None:
                try:
                    expected_birth = _exact_process_birth(
                        start_receipt.get("owner_process_birth")
                    )
                except EAAEFCASFBootstrapOwnerError:
                    expected_birth = None
            if (
                (expected_birth is not None and abort_birth != expected_birth)
                or (
                    abort_birth is not None
                    and reconciliation.inspect_process_birth(abort_birth.pid)
                    == abort_birth
                )
            ):
                raise EAAEFCASFBootstrapOwnerError(
                    "CASF owner abort process evidence differs"
                )
            reconciliation._assert_no_boundary_authority(value)  # noqa: SLF001
            return value
        except BaseException as exc:
            raise EAAEFCASFBootstrapOwnerError(
                "CASF owner cleanup is unconfirmed; generation is permanently abandoned"
            ) from exc

    @staticmethod
    def _commit_started(
        guard: EAAEFCASFBootstrapOwnerGuard,
        binding: EAAEFCASFBootstrapBinding,
        started: Mapping[str, Any],
        final_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        from ..runtime import eaaef_reconciliation_lifecycle as reconciliation

        raw = guard.commit_started_owner(
            start_receipt=started,
            final_record_cid=final_record["record_cid"],
        )
        if not isinstance(raw, Mapping):
            raise EAAEFCASFBootstrapOwnerError("CASF owner commit receipt is malformed")
        value = _verified(
            raw,
            schema=EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
            cid_field="commit_receipt_cid",
            fields=_COMMIT_FIELDS,
            noun="CASF owner commit receipt",
        )
        expected_strings = {
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "owner_start_receipt_cid": started["start_receipt_cid"],
            "final_record_cid": final_record["record_cid"],
        }
        if (
            any(
                type(value.get(name)) is not str or value.get(name) != selected
                for name, selected in expected_strings.items()
            )
            or type(value.get("owner_commit_completed")) is not bool
            or value.get("owner_commit_completed") is not True
            or type(value.get("owner_process_alive")) is not bool
            or value.get("owner_process_alive") is not True
            or type(value.get("provider_process_started")) is not bool
            or value.get("provider_process_started") is not False
        ):
            raise EAAEFCASFBootstrapOwnerError("CASF owner commit evidence differs")
        birth = _exact_process_birth(value.get("owner_process_birth"))
        if (
            birth != _exact_process_birth(started.get("owner_process_birth"))
            or reconciliation.inspect_process_birth(birth.pid) != birth
        ):
            raise EAAEFCASFBootstrapOwnerError(
                "CASF committed owner birth is not corroborated"
            )
        reconciliation._assert_no_boundary_authority(value)  # noqa: SLF001
        return value

    @staticmethod
    def _record(
        *,
        phase: str,
        request: Mapping[str, Any],
        lifecycle_interface: str,
        absence: Mapping[str, Any],
        offline: Mapping[str, Any] | None = None,
        started: Mapping[str, Any] | None = None,
        canonical: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from ..runtime.eaaef_reconciliation_lifecycle import _cid

        value: dict[str, Any] = {
            "schema": EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
            "generation_id": request["generation_id"],
            "phase": phase,
            "request_cid": request["request_cid"],
            "source_forest_root": request["source_forest_root"],
            "population_cid": request["population_cid"],
            "owner_lifecycle_interface": lifecycle_interface,
            "absence_attestation_cid": absence["attestation_cid"],
            "offline_materialization_receipt_cid": (
                "" if offline is None else offline["receipt_cid"]
            ),
            "owner_start_receipt_cid": (
                "" if started is None else started["start_receipt_cid"]
            ),
            "canonical_bootstrap_receipt_cid": (
                "" if canonical is None else canonical["receipt_cid"]
            ),
            "owner_process_birth": (
                None if started is None else started["owner_process_birth"]
            ),
        }
        value["record_cid"] = _cid(value)
        return value

    @staticmethod
    def _canonical_receipt(
        request: Mapping[str, Any],
        population: Any,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        from ..runtime import eaaef_reconciliation_lifecycle as reconciliation

        value: dict[str, Any] = {
            "schema": reconciliation.EAAEF_OFFLINE_POPULATION_RECEIPT_SCHEMA,
            "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "source_forest_root": population.source_forest_root,
            "population_cid": population.population_cid,
            "goal_population_cid": population.goal_population_cid,
            "execution_contract_population_cid": (
                population.execution_contract_population_cid
            ),
            "bootstrap_population_cid": population.bootstrap_population_cid,
            "held_plan_r2_population_cid": population.plan_r2_population_cid,
            "plan_r1_cid": population.plan_r1_cid,
            "goal_count": reconciliation.EAAEF_GOAL_COUNT,
            "goal_edge_count": reconciliation.EAAEF_GOAL_EDGE_COUNT,
            "plan_count": 1,
            "task_count": reconciliation.EAAEF_TASK_COUNT,
            "bootstrap_task_count": reconciliation.EAAEF_BOOTSTRAP_TASK_COUNT,
            "held_task_count": reconciliation.EAAEF_PLAN_R2_TASK_COUNT,
            "task_status_counts": {
                "blocked": reconciliation.EAAEF_PLAN_R2_TASK_COUNT,
                "todo": reconciliation.EAAEF_BOOTSTRAP_TASK_COUNT,
            },
            "execution_contract_counts": population.execution_contract_counts,
            "execution_contracts_materialized": True,
            "terminal_statuses_imported": 0,
            "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
            "bootstrap_owner_absent_during_materialization": True,
            "owner_started_after_bootstrap": True,
            "direct_database_mutation_after_owner_start": False,
            "provider_process_started": False,
            "bootstrap_snapshot": dict(snapshot),
        }
        value["receipt_cid"] = reconciliation._cid(value)  # noqa: SLF001
        validated, _snapshot = reconciliation._validate_offline_population_receipt(  # noqa: SLF001
            value,
            request=request,
            population=population,
        )
        return validated

    @staticmethod
    def _seal_offline_database(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise EAAEFCASFBootstrapOwnerError(
                "offline CASF bootstrap database is unsafe"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_size <= 0
                or metadata.st_nlink != 1
            ):
                raise EAAEFCASFBootstrapOwnerError(
                    "offline CASF bootstrap database is unsafe"
                )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            final_metadata = os.fstat(descriptor)
            path_metadata = os.lstat(path)
            if (
                not stat.S_ISREG(path_metadata.st_mode)
                or path_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(path_metadata.st_mode) != 0o600
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (final_metadata.st_dev, final_metadata.st_ino)
            ):
                raise EAAEFCASFBootstrapOwnerError(
                    "offline CASF bootstrap database changed while it was sealed"
                )
        except OSError as exc:
            raise EAAEFCASFBootstrapOwnerError(
                "offline CASF bootstrap database is not durable"
            ) from exc
        finally:
            os.close(descriptor)

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: object,
    ) -> Mapping[str, Any]:
        from ..runtime import eaaef_reconciliation_lifecycle as reconciliation
        from ..runtime.eaaef_offline_population import materialize_offline_eaaef_population
        from .database_task_source import DatabaseTaskSource

        if type(population) is not reconciliation.CompiledEAAEFPopulation:
            raise EAAEFCASFBootstrapOwnerError("compiled EAAEF population is not exact")
        expected = reconciliation._build_offline_population_request(  # noqa: SLF001
            generation_id=str(request.get("generation_id") or ""),
            population=population,
        )
        if reconciliation._canonical_bytes(dict(request)) != reconciliation._canonical_bytes(  # noqa: SLF001
            expected
        ):
            raise EAAEFCASFBootstrapOwnerError("offline bootstrap request differs")
        if population.source_forest_root != self._source_forest_root:
            raise EAAEFCASFBootstrapOwnerError("bootstrap source forest differs")
        forest = reconciliation._require_sealed_forest(  # noqa: SLF001
            reconciliation.inspect_current_repository_forest(self._repo_root)
        )
        board = reconciliation._json_object(  # noqa: SLF001
            self._repo_root / reconciliation.EAAEF_BOARD_PATH,
            noun="EAAEF task board",
        )
        population = reconciliation.verify_compiled_eaaef_population_commitments(
            population,
            current_board=board,
            current_forest=forest,
            repo_root=self._repo_root,
        )
        generation_id = expected["generation_id"]
        binding = EAAEFCASFBootstrapBinding(
            generation_id=generation_id,
            source_head=population.source_head,
            source_tree=population.source_tree,
            source_forest_root=population.source_forest_root,
            board_cid=population.board_cid,
            population_cid=population.population_cid,
            bootstrap_population_cid=population.bootstrap_population_cid,
            plan_r1_cid=population.plan_r1_cid,
            database_path=self._registry.generation_dir(generation_id) / "control.duckdb",
            owner_state_dir=self._registry.generation_dir(generation_id) / "casf-owner",
        )
        lifecycle_interface = str(self._owner_lifecycle.INTERFACE)
        with self._registry.exclusive() as registry_capability:
            self._registry.prepare_generation(registry_capability, generation_id)
            with self._owner_lifecycle.hold_exclusive_bootstrap(binding) as guard:
                if (
                    getattr(guard, "INTERFACE", "")
                    != EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE
                    or not callable(getattr(guard, "owner_absence_attestation", None))
                    or not callable(getattr(guard, "start_after_offline_commit", None))
                    or not callable(getattr(guard, "abort_started_owner", None))
                    or not callable(getattr(guard, "commit_started_owner", None))
                ):
                    raise EAAEFCASFBootstrapOwnerError("CASF owner guard is invalid")
                absence = self._absence(guard.owner_absence_attestation(), binding)
                self._registry.write_record(
                    registry_capability,
                    generation_id,
                    self._record(
                        phase="absent",
                        request=expected,
                        lifecycle_interface=lifecycle_interface,
                        absence=absence,
                    ),
                )
                with DatabaseTaskSource(binding.database_path) as task_source:
                    offline = materialize_offline_eaaef_population(
                        task_source,
                        population,
                        current_board=board,
                        current_forest=forest,
                        repo_root=self._repo_root,
                        owner_active=False,
                        historical_task_statuses=None,
                    )
                self._seal_offline_database(binding.database_path)
                self._registry._sync_private_directory(  # noqa: SLF001
                    binding.database_path.parent,
                    noun="CASF bootstrap generation",
                )
                self._registry.write_record(
                    registry_capability,
                    generation_id,
                    self._record(
                        phase="offline_committed",
                        request=expected,
                        lifecycle_interface=lifecycle_interface,
                        absence=absence,
                        offline=offline,
                    ),
                )
                raw_started: Mapping[str, Any] | None = None
                try:
                    raw = guard.start_after_offline_commit(
                        offline_materialization_receipt=offline
                    )
                    if not isinstance(raw, Mapping):
                        raise EAAEFCASFBootstrapOwnerError(
                            "owner start receipt is malformed"
                        )
                    raw_started = dict(raw)
                    started, snapshot = self._started(
                        raw_started,
                        binding,
                        population,
                        absence,
                        offline,
                    )
                    canonical = self._canonical_receipt(expected, population, snapshot)
                    final_record = self._record(
                        phase="owner_started",
                        request=expected,
                        lifecycle_interface=lifecycle_interface,
                        absence=absence,
                        offline=offline,
                        started=started,
                        canonical=canonical,
                    )
                    self._registry.write_record(
                        registry_capability,
                        generation_id,
                        final_record,
                    )
                    self._commit_started(guard, binding, started, final_record)
                    return canonical
                except BaseException as failure:
                    try:
                        self._abort_started(
                            guard,
                            binding,
                            raw_started,
                            reason_code=(
                                "owner_start_operation_failed"
                                if raw_started is None
                                else "post_start_validation_or_commit_failed"
                            ),
                        )
                    except EAAEFCASFBootstrapOwnerError as cleanup_failure:
                        raise cleanup_failure from failure
                    raise

    def apply_signed_plan_r2(
        self,
        request: Mapping[str, Any],
        *,
        population: object,
        authority: object,
    ) -> Mapping[str, Any]:
        del request, population, authority
        self._blocked()

    def launch_reconciliation_supervisor(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        self._blocked()

    def reconciliation_status_snapshot(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        self._blocked()

    def stop_reconciliation_tracks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        self._blocked()


def bind_eaaef_casf_bootstrap_owner(
    *,
    repo_root: Path,
    registry_root: Path,
    source_forest_root: str,
    owner_lifecycle: EAAEFCASFBootstrapOwnerLifecycle,
) -> CASFBootstrapEAAEFTypedReconciliationOwner:
    """Bind the prefix in code; no request can select paths or lifecycle."""

    return CASFBootstrapEAAEFTypedReconciliationOwner(
        repo_root=repo_root,
        registry=EAAEFCASFBootstrapRegistry(registry_root),
        source_forest_root=source_forest_root,
        owner_lifecycle=owner_lifecycle,
    )


__all__ = [
    "CASFBootstrapEAAEFTypedReconciliationOwner",
    "EAAEFCASFBootstrapBinding",
    "EAAEFCASFBootstrapOwnerError",
    "EAAEFCASFBootstrapOwnerGuard",
    "EAAEFCASFBootstrapOwnerLifecycle",
    "EAAEFCASFBootstrapRegistry",
    "EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS",
    "EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE",
    "EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE",
    "EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS",
    "EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA",
    "EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA",
    "EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA",
    "EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA",
    "EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA",
    "bind_eaaef_casf_bootstrap_owner",
]
