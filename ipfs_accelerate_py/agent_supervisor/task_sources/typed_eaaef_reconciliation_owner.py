"""Statically named, non-production EAAEF reconciliation-owner facade.

The EAAEF lifecycle resolves this exact module name so integration can report
the missing final-CASF bindings deterministically.  The default facade does
not open a database, read a credential, inspect or signal a process, start a
provider, or sign authority.  Every effect method fails closed.

The separately exported CASF bootstrap adapter implements only the distinct
``EAAEFBootstrapReconciliationOwner@1`` guarded
owner-absent/offline-commit/owner-start prefix.  Its bootstrap opener is inert
when called with only ``repo_root``: a long-lived host must also bind the exact
private registry root, sealed source forest, and unsigned snapshot bindings.
Those inputs create no production or signing authority.  The bootstrap adapter
is never accepted as the production owner needed for independently signed Plan
R2, production status, track stop, or supervisor launch.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

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
EAAEF_OWNER_PRODUCTION_BLOCKERS: Final = (
    "final_casf_exclusive_typed_owner_adapter_absent",
    "offline_database_task_source_materializer_not_bound",
    *PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS,
    "owner_status_snapshot_not_bound",
    "exact_birth_stop_tracks_not_bound",
    "supervisor_launch_not_bound",
)


class EAAEFTypedReconciliationOwnerUnavailable(RuntimeError):
    """The statically named facade has no admitted production effect path."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


class NonProductionEAAEFTypedReconciliationOwner:
    """Fail-closed facade exposing only truthful integration blockers."""

    INTERFACE: ClassVar[str] = EAAEF_RECONCILIATION_OWNER_INTERFACE

    def reconciliation_qualification(self) -> Mapping[str, Any]:
        qualification: dict[str, Any] = {
            "schema": EAAEF_OWNER_QUALIFICATION_SCHEMA,
            "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "source_forest_root": "",
            "bootstrap_materialization_mode": "translation_only_owner_not_bound",
            "bootstrap_materialization_before_owner_start": False,
            "offline_population_includes_execution_contracts": True,
            "direct_database_mutation_after_owner_start": False,
            "typed_task_source_interface": DATABASE_TASK_SOURCE_INTERFACE,
            "plan_r2_repository_interface": "",
            "plan_r2_remote_gateway_interface": "",
            "plan_r2_wire_channel_interface": "",
            "plan_r2_remote_runtime_qualification_status": (
                PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS
            ),
            "plan_r2_remote_runtime_blockers": list(EAAEF_OWNER_PRODUCTION_BLOCKERS),
            "status_operation": "unavailable",
            "stop_tracks_operation": "unavailable",
            "launch_modes": [],
            "database_authority_crossing_allowed": False,
            "filesystem_path_authority_crossing_allowed": False,
            "transport_token_authority_crossing_allowed": False,
            "sql_crossing_allowed": False,
            "provider_launch_allowed": False,
        }
        qualification["qualification_cid"] = _cid(qualification)
        return qualification

    @staticmethod
    def _blocked() -> None:
        raise EAAEFTypedReconciliationOwnerUnavailable(
            "EAAEF production reconciliation owner is not bound: "
            + ", ".join(EAAEF_OWNER_PRODUCTION_BLOCKERS)
        )

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: object,
    ) -> Mapping[str, Any]:
        del request, population
        self._blocked()

    def apply_signed_plan_r2(
        self,
        request: Mapping[str, Any],
        *,
        population: object,
        authority: object,
    ) -> Mapping[str, Any]:
        del request, population, authority
        self._blocked()

    def launch_reconciliation_supervisor(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del request
        self._blocked()

    def reconciliation_status_snapshot(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del request
        self._blocked()

    def stop_reconciliation_tracks(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del request
        self._blocked()


def open_eaaef_typed_reconciliation_owner(
    *,
    repo_root: Path,
) -> NonProductionEAAEFTypedReconciliationOwner:
    """Return the static blocker facade without inspecting ``repo_root``."""

    del repo_root
    return NonProductionEAAEFTypedReconciliationOwner()


from .eaaef_casf_bootstrap_owner import (  # noqa: E402
    EAAEF_BOOTSTRAP_OWNER_QUALIFICATION_SCHEMA,
    EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS,
    EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
    EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
    EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
    EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS,
    CASFBootstrapEAAEFTypedReconciliationOwner,
    EAAEFCASFBootstrapBinding,
    EAAEFCASFBootstrapOwnerError,
    EAAEFCASFBootstrapOwnerGuard,
    EAAEFCASFBootstrapOwnerLifecycle,
    EAAEFCASFBootstrapRegistry,
    bind_eaaef_casf_bootstrap_owner,
)


def open_eaaef_bootstrap_reconciliation_owner(
    *,
    repo_root: Path,
    registry_root: Path | None = None,
    source_forest_root: str = "",
    snapshot_bindings: object | None = None,
    startup_timeout_seconds: float = 180.0,
    operation_timeout_seconds: float = 180.0,
    shutdown_timeout_seconds: float = 30.0,
) -> CASFBootstrapEAAEFTypedReconciliationOwner | None:
    """Open the bounded owner only from a complete explicit host binding.

    The statically named lifecycle resolver supplies only ``repo_root``.  That
    call intentionally returns ``None`` so the normal command path remains a
    typed no-go rather than deriving paths or owner inputs from environment,
    argv, cached receipts, or historical DuckDB rows.  A long-lived host may
    call this function directly with all three additional bindings and retain
    the returned object across prepare, authenticated reattachment, and exact
    stop.
    """

    if registry_root is None and snapshot_bindings is None and not source_forest_root:
        return None
    if (
        registry_root is None
        or snapshot_bindings is None
        or type(source_forest_root) is not str
        or not source_forest_root
    ):
        raise EAAEFCASFBootstrapOwnerError(
            "CASF bootstrap opener requires the complete explicit host binding"
        )
    from ..runtime.eaaef_casf_bootstrap_lifecycle import (
        EAAEFCASFBootstrapSnapshotBindings,
        QuackEAAEFCASFBootstrapOwnerLifecycle,
    )

    if type(snapshot_bindings) is not EAAEFCASFBootstrapSnapshotBindings:
        raise EAAEFCASFBootstrapOwnerError(
            "CASF bootstrap opener snapshot bindings are not exact"
        )
    selected_registry = Path(registry_root)
    lexical_registry = Path(os.path.abspath(selected_registry))
    if not selected_registry.is_absolute() or selected_registry != lexical_registry:
        raise EAAEFCASFBootstrapOwnerError(
            "CASF bootstrap opener registry root is not an exact absolute path"
        )
    selected_repo = Path(repo_root).resolve(strict=True)
    lifecycle = QuackEAAEFCASFBootstrapOwnerLifecycle(
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=startup_timeout_seconds,
        operation_timeout_seconds=operation_timeout_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
    return bind_eaaef_casf_bootstrap_owner(
        repo_root=selected_repo,
        registry_root=selected_registry,
        source_forest_root=source_forest_root,
        owner_lifecycle=lifecycle,
    )

__all__ = [
    "CASFBootstrapEAAEFTypedReconciliationOwner",
    "DATABASE_TASK_SOURCE_INTERFACE",
    "EAAEFCASFBootstrapBinding",
    "EAAEFCASFBootstrapOwnerError",
    "EAAEFCASFBootstrapOwnerGuard",
    "EAAEFCASFBootstrapOwnerLifecycle",
    "EAAEFCASFBootstrapRegistry",
    "EAAEFTypedReconciliationOwnerUnavailable",
    "EAAEF_BOOTSTRAP_OWNER_QUALIFICATION_SCHEMA",
    "EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE",
    "EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS",
    "EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE",
    "EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE",
    "EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA",
    "EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS",
    "EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA",
    "EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA",
    "EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA",
    "EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA",
    "EAAEF_OWNER_PRODUCTION_BLOCKERS",
    "EAAEF_RECONCILIATION_OWNER_INTERFACE",
    "NonProductionEAAEFTypedReconciliationOwner",
    "bind_eaaef_casf_bootstrap_owner",
    "open_eaaef_bootstrap_reconciliation_owner",
    "open_eaaef_typed_reconciliation_owner",
]
