"""Statically named, non-production EAAEF reconciliation-owner facade.

The EAAEF lifecycle resolves this exact module name so integration can report
the missing final-CASF bindings deterministically.  This facade intentionally
does not open a database, read a credential, inspect or signal a process,
start a provider, or sign authority.  Every effect method fails closed.

Its qualification is evidence, not an admission: the status and blockers are
deliberately incompatible with the production qualification required by the
public lifecycle.
"""

from __future__ import annotations

import hashlib
import json
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


__all__ = [
    "DATABASE_TASK_SOURCE_INTERFACE",
    "EAAEFTypedReconciliationOwnerUnavailable",
    "EAAEF_OWNER_PRODUCTION_BLOCKERS",
    "EAAEF_RECONCILIATION_OWNER_INTERFACE",
    "NonProductionEAAEFTypedReconciliationOwner",
    "open_eaaef_typed_reconciliation_owner",
]
