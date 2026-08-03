"""PlanSupervisorService@1 — shared create/steer facade for control transports.

Python, CLI, and MCP all dispatch the same closed control operations into this
module.  Preview paths are proposal-only; apply paths require the normal
control-plane permit, lease, fence, idempotency, and expected-effects gates
before any revision-store write is attempted.

Default control-service construction binds live handlers from
:func:`build_default_plan_control_handlers` so create/steer and the workflow
compatibility aliases are no longer reported as ``unavailable``.  Handlers
import domain services only at execute time so help/import/discovery remain
provider-free.

Doctor residual refill (PDR-055) enters plan steering only through the
proposal-only preview path: residuals map to append-only delta items or
bounded objective work proposals and never grant completion or mutation
authority.  Derived runtime task-source admission remains gated until
PDR-081.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..control.control_contracts import (
    Operation,
    OperationRequest,
)

# BackendResponse is imported lazily inside handlers to keep module import free of
# the heavy control_plane graph (and avoid cycles when the control plane binds
# default plan handlers).


PLAN_SUPERVISOR_SERVICE_INTERFACE: Final[str] = "PlanSupervisorService@1"
PLAN_SUPERVISOR_SERVICE_VERSION: Final[int] = 1

PLAN_CONTROL_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_CREATE_APPLY,
        Operation.PLAN_STEER_PREVIEW,
        Operation.PLAN_STEER_APPLY,
    }
)

# Workflow aliases keep catalog identity (operation name / tool / CLI command)
# while sharing the create pipeline implemented by this facade.
WORKFLOW_ALIAS_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.WORKFLOW_PREVIEW,
        Operation.WORKFLOW_MATERIALIZE,
    }
)

DEFAULT_PLAN_CONTROL_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    PLAN_CONTROL_OPERATIONS | WORKFLOW_ALIAS_OPERATIONS
)


class PlanSupervisorServiceError(RuntimeError):
    """Raised when the shared plan facade cannot complete a control dispatch."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise PlanSupervisorServiceError(f"{name} must be a mapping")


def _record(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


@dataclass
class PlanSupervisorService:
    """Facade over PlanCreateService, PlanSteerService, and PlanRevisionStore.

    Construction is side-effect free.  Domain services and the revision store
    are resolved lazily so catalog discovery and CLI help never construct
    analysis providers or open task-source backends.
    """

    INTERFACE: Final[str] = PLAN_SUPERVISOR_SERVICE_INTERFACE
    VERSION: Final[int] = PLAN_SUPERVISOR_SERVICE_VERSION

    create_service: Any | None = None
    steer_service: Any | None = None
    revision_store: Any | None = None
    revision_store_root: str | Path | None = None
    create_service_factory: Callable[[], Any] | None = None
    steer_service_factory: Callable[[], Any] | None = None
    revision_store_factory: Callable[[], Any] | None = None
    clock_ms: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    # -- lazy domain wiring -------------------------------------------------

    def _create(self) -> Any:
        if self.create_service is not None:
            return self.create_service
        with self._lock:
            if self.create_service is not None:
                return self.create_service
            if self.create_service_factory is not None:
                self.create_service = self.create_service_factory()
                return self.create_service
            from .plan_create_service import create_default_plan_create_service

            self.create_service = create_default_plan_create_service(
                build_analysis_factory=False,
                clock_ms=self.clock_ms,
            )
            return self.create_service

    def _steer(self) -> Any:
        if self.steer_service is not None:
            return self.steer_service
        with self._lock:
            if self.steer_service is not None:
                return self.steer_service
            if self.steer_service_factory is not None:
                self.steer_service = self.steer_service_factory()
                return self.steer_service
            from .plan_steer_service import PlanSteerService

            self.steer_service = PlanSteerService(clock_ms=self.clock_ms)
            return self.steer_service

    def _store(self, request: OperationRequest | None = None) -> Any:
        if self.revision_store is not None:
            return self.revision_store
        with self._lock:
            if self.revision_store is not None:
                return self.revision_store
            if self.revision_store_factory is not None:
                self.revision_store = self.revision_store_factory()
                return self.revision_store
            from ..task_sources.plan_revision_store import PlanRevisionStore

            root: Path
            if self.revision_store_root is not None:
                root = Path(self.revision_store_root)
            elif request is not None:
                root = Path(request.state_root) / "plan_revision_store"
            else:
                raise PlanSupervisorServiceError(
                    "revision store root is required for apply"
                )
            self.revision_store = PlanRevisionStore(root)
            return self.revision_store

    # -- domain API ---------------------------------------------------------

    def preview_create(
        self,
        request: Any,
        *,
        mode: Any = None,
        materials: Any = None,
        compatibility_alias: str = "",
    ) -> Any:
        """Proposal-only create preview; never writes task sources."""

        service = self._create()
        kwargs: dict[str, Any] = {}
        if mode is not None:
            kwargs["mode"] = mode
        if materials is not None:
            kwargs["materials"] = materials
        if compatibility_alias:
            kwargs["compatibility_alias"] = compatibility_alias
        return service.preview_create(request, **kwargs)

    def preview_steer(
        self,
        request: Any,
        *,
        materials: Any = None,
        live_state: Any = None,
    ) -> Any:
        """Proposal-only steer preview; never writes task sources."""

        service = self._steer()
        if materials is not None or live_state is not None:
            # Prefer the materials-aware path when present; otherwise plain preview.
            if hasattr(service, "preview_steer"):
                if materials is not None and live_state is not None:
                    return service.preview_steer(
                        request, materials=materials, live_state=live_state
                    )
                if materials is not None:
                    try:
                        return service.preview_steer(request, materials=materials)
                    except TypeError:
                        pass
                if live_state is not None:
                    try:
                        return service.preview_steer(request, live_state=live_state)
                    except TypeError:
                        pass
        return service.preview_steer(request)

    def preview_steer_from_doctor_residuals(
        self,
        request: Any,
        *,
        residuals: Sequence[Any] | None = None,
        fixed_point: Any = None,
        live_state: Any = None,
        plan: Any = None,
        memory: Any = None,
        policy: Any = None,
        materials: Any = None,
    ) -> dict[str, Any]:
        """Proposal-only steer preview driven by Doctor residuals (PDR-055).

        Successful residual-free fixed points emit no work.  New residuals are
        deduplicated and mapped to append-only plan successors where possible,
        otherwise to bounded ``ObjectiveWorkProposal`` records.  This path is
        always read-only: it never applies revisions, never edits seed boards,
        and never grants completion or mutation authority.  Derived runtime
        admission remains gated until PDR-081 enables it on the refill policy.
        """

        from ..objectives.doctor_plan_refill import (
            DoctorPlanRefillDisposition,
            doctor_residuals_for_steer,
        )

        package = doctor_residuals_for_steer(
            residuals=residuals,
            fixed_point=fixed_point,
            plan=plan,
            memory=memory,
            policy=policy,
            request=request,
            live_state=live_state,
        )
        receipt = package["receipt"]
        proposed_items = list(package["proposed_delta_items"])

        # Merge caller materials with residual-derived append-only items.
        merged_materials = materials
        if package.get("materials") is not None:
            merged_materials = package["materials"]
            if materials is not None and isinstance(materials, Mapping):
                extra = list(materials.get("proposed_delta_items") or ())
                merged_materials = dict(package["materials"])
                merged_materials["proposed_delta_items"] = (
                    list(merged_materials.get("proposed_delta_items") or [])
                    + extra
                )

        steer_preview: Any = None
        if (
            receipt.disposition
            is not DoctorPlanRefillDisposition.FIXED_POINT_CLOSED
            and proposed_items
            and live_state is not None
        ):
            try:
                steer_preview = self.preview_steer(
                    request,
                    materials=merged_materials,
                    live_state=live_state,
                )
            except TypeError:
                # Older steer services may only accept the request object.
                steer_preview = self.preview_steer(request)
        elif (
            receipt.disposition is DoctorPlanRefillDisposition.FIXED_POINT_CLOSED
        ):
            steer_preview = None
        elif live_state is not None and merged_materials is not None:
            # Residual-free of successors but materials present (proposals only).
            try:
                steer_preview = self.preview_steer(
                    request,
                    materials=merged_materials,
                    live_state=live_state,
                )
            except Exception:
                steer_preview = None

        preview_record = _record(steer_preview) if steer_preview is not None else {}
        return {
            "operation": "plan_steer_preview",
            "status": "proposal",
            "read_only": True,
            "wrote_effects": (),
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "doctor_plan_refill": receipt.to_dict(),
            "disposition": receipt.disposition.value,
            "emits_work": receipt.emits_work,
            "proposed_delta_items": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in proposed_items
            ],
            "work_proposals": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in receipt.work_proposals
            ],
            "steer_preview": preview_record,
            "derived_runtime_admitted": receipt.derived_runtime_admitted,
            "derived_runtime_gate": receipt.to_dict().get("derived_runtime_gate"),
        }

    def refill_doctor_residuals(
        self,
        residuals: Sequence[Any] | None = None,
        *,
        fixed_point: Any = None,
        plan: Any = None,
        memory: Any = None,
        policy: Any = None,
    ) -> Any:
        """Run Doctor residual refill without invoking steer (proposal-only)."""

        from ..objectives.doctor_plan_refill import refill_doctor_plan_residuals

        return refill_doctor_plan_residuals(
            residuals,
            fixed_point=fixed_point,
            plan=plan,
            memory=memory,
            policy=policy,
        )

    def apply_revision(
        self,
        apply_request: Any,
        *,
        control_request: OperationRequest | None = None,
    ) -> Any:
        """Authorized durable apply through the revision store."""

        store = self._store(control_request)
        return store.apply(apply_request)

    plan_create_preview = preview_create
    plan_steer_preview = preview_steer
    plan_create_apply = apply_revision
    plan_steer_apply = apply_revision

    # -- control dispatch ---------------------------------------------------

    def handle_control_request(self, request: OperationRequest) -> Any:
        """Dispatch one closed control operation into the create/steer facade.

        Preview/alias-preview responses never set ``changed`` or claim applied
        effects.  Apply paths claim only effect IDs declared on the request and
        only when the control plane already admitted a real mutation (not
        dry-run).
        """

        from ..control.control_plane import BackendResponse

        operation = request.operation
        parameters = dict(request.parameters)

        if operation is Operation.PLAN_CREATE_PREVIEW:
            return self._handle_create_preview(
                request, parameters, compatibility_alias=""
            )
        if operation is Operation.WORKFLOW_PREVIEW:
            # Preserve catalog identity: result operation remains workflow_preview.
            return self._handle_create_preview(
                request,
                parameters,
                compatibility_alias="workflow_preview",
            )
        if operation is Operation.PLAN_STEER_PREVIEW:
            return self._handle_steer_preview(request, parameters)
        if operation is Operation.PLAN_CREATE_APPLY:
            return self._handle_apply(
                request, parameters, origin="create"
            )
        if operation is Operation.PLAN_STEER_APPLY:
            return self._handle_apply(
                request, parameters, origin="steer"
            )
        if operation is Operation.WORKFLOW_MATERIALIZE:
            return self._handle_apply(
                request, parameters, origin="workflow_materialize"
            )
        raise PlanSupervisorServiceError(
            f"operation {operation.value} is not a plan-control operation"
        )

    def _handle_create_preview(
        self,
        request: OperationRequest,
        parameters: Mapping[str, Any],
        *,
        compatibility_alias: str,
    ) -> Any:
        from ..control.control_plane import BackendResponse

        plan_request = parameters.get("plan_request")
        if plan_request is None:
            # Sparse transport path: surface a stable domain payload without
            # inventing repository authority from prompt/repository text.
            data = {
                "operation": request.operation.value,
                "status": "proposal",
                "read_only": True,
                "wrote_effects": (),
                "compatibility_alias": compatibility_alias
                or str(parameters.get("compatibility_alias") or ""),
                "plan_request_present": False,
                "repository_id": request.repository_id,
                "tree_id": request.tree_id,
                "objective_id": request.objective_id,
                "policy_id": request.policy_id,
                "message": (
                    "plan_create_preview requires parameters.plan_request "
                    "for full pipeline execution"
                ),
            }
            return BackendResponse(
                data=data,
                changed=False,
                checks=("schema", "proposal_only", "provider_free"),
            )

        alias = compatibility_alias or str(
            parameters.get("compatibility_alias") or ""
        )
        receipt = self.preview_create(
            plan_request,
            mode=parameters.get("mode"),
            materials=parameters.get("materials"),
            compatibility_alias=alias,
        )
        data = _record(receipt)
        data.setdefault("operation", request.operation.value)
        data.setdefault("read_only", True)
        data.setdefault("wrote_effects", ())
        if alias:
            data.setdefault("compatibility_alias", alias)
        # Hard guarantee: proposal tier cannot report applied effects.
        return BackendResponse(
            data=data,
            changed=False,
            checks=("schema", "proposal_only", "body_free"),
        )

    def _handle_steer_preview(
        self,
        request: OperationRequest,
        parameters: Mapping[str, Any],
    ) -> Any:
        from ..control.control_plane import BackendResponse

        plan_request = parameters.get("plan_request")
        if plan_request is None:
            data = {
                "operation": request.operation.value,
                "status": "proposal",
                "read_only": True,
                "wrote_effects": (),
                "plan_request_present": False,
                "repository_id": request.repository_id,
                "tree_id": request.tree_id,
                "message": (
                    "plan_steer_preview requires parameters.plan_request "
                    "for full pipeline execution"
                ),
            }
            return BackendResponse(
                data=data,
                changed=False,
                checks=("schema", "proposal_only", "provider_free"),
            )
        receipt = self.preview_steer(
            plan_request,
            materials=parameters.get("materials"),
            live_state=parameters.get("live_state"),
        )
        data = _record(receipt)
        data.setdefault("operation", request.operation.value)
        data.setdefault("read_only", True)
        data.setdefault("wrote_effects", ())
        return BackendResponse(
            data=data,
            changed=False,
            checks=("schema", "proposal_only", "body_free"),
        )

    def _handle_apply(
        self,
        request: OperationRequest,
        parameters: Mapping[str, Any],
        *,
        origin: str,
    ) -> Any:
        from ..control.control_plane import BackendResponse

        if request.dry_run:
            data = {
                "operation": request.operation.value,
                "status": "proposal",
                "origin": origin,
                "dry_run": True,
                "preview_ref": str(parameters.get("preview_ref") or ""),
                "preview_root": str(parameters.get("preview_root") or ""),
                "read_only": True,
                "wrote_effects": (),
            }
            return BackendResponse(
                data=data,
                changed=False,
                checks=("schema", "proposal_only", "dry_run"),
            )

        apply_request = parameters.get("apply_request")
        if apply_request is None:
            # Mutation was already authorized by the control plane.  Without a
            # bound apply payload we refuse rather than inventing a write.
            raise PlanSupervisorServiceError(
                f"{request.operation.value} requires parameters.apply_request"
            )

        receipt = self.apply_revision(
            apply_request, control_request=request
        )
        data = _record(receipt)
        data.setdefault("operation", request.operation.value)
        data.setdefault("origin", origin)
        data["dry_run"] = False
        effect_ids = tuple(
            item.effect_id for item in request.expected_effects
        )
        return BackendResponse(
            data=data,
            changed=True,
            applied_effect_ids=effect_ids,
            checks=("schema", "authorized_apply", "revision_store"),
        )


_default_service: PlanSupervisorService | None = None
_default_service_lock = threading.RLock()


def get_plan_supervisor_service() -> PlanSupervisorService:
    """Return the process-local default plan supervisor facade."""

    global _default_service
    with _default_service_lock:
        if _default_service is None:
            _default_service = PlanSupervisorService()
        return _default_service


def set_plan_supervisor_service(service: PlanSupervisorService | None) -> None:
    """Inject or clear the process-local default facade (tests)."""

    global _default_service
    if service is not None and not isinstance(service, PlanSupervisorService):
        raise TypeError("service must be a PlanSupervisorService")
    with _default_service_lock:
        _default_service = service


def create_default_plan_supervisor_service(
    **kwargs: Any,
) -> PlanSupervisorService:
    """Construct a new facade instance (does not replace the process default)."""

    return PlanSupervisorService(**kwargs)


build_default_plan_supervisor_service = create_default_plan_supervisor_service


def build_default_plan_control_handlers() -> Mapping[Operation, Callable[..., Any]]:
    """Return live plan/workflow handlers for default control-service binding.

    Handlers are pure callables; domain imports occur only when the control
    plane dispatches a request.  Catalog identity for workflow aliases is
    preserved by routing through the original ``request.operation``.
    """

    def _handler(request: OperationRequest) -> BackendResponse:
        return get_plan_supervisor_service().handle_control_request(request)

    return MappingProxyType(
        {
            Operation.PLAN_CREATE_PREVIEW: _handler,
            Operation.PLAN_CREATE_APPLY: _handler,
            Operation.PLAN_STEER_PREVIEW: _handler,
            Operation.PLAN_STEER_APPLY: _handler,
            Operation.WORKFLOW_PREVIEW: _handler,
            Operation.WORKFLOW_MATERIALIZE: _handler,
        }
    )


__all__ = [
    "DEFAULT_PLAN_CONTROL_OPERATIONS",
    "PLAN_CONTROL_OPERATIONS",
    "PLAN_SUPERVISOR_SERVICE_INTERFACE",
    "PLAN_SUPERVISOR_SERVICE_VERSION",
    "WORKFLOW_ALIAS_OPERATIONS",
    "PlanSupervisorService",
    "PlanSupervisorServiceError",
    "build_default_plan_control_handlers",
    "build_default_plan_supervisor_service",
    "create_default_plan_supervisor_service",
    "get_plan_supervisor_service",
    "set_plan_supervisor_service",
]
