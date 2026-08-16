"""Body-free admitted edit plans for single-path autonomous repair.

These plans are **not** source rewrites. They bind identity, doctor operator,
surface anchors, IR evidence, and materialize preconditions so a later
fail-closed materializer (or human) can apply an edit only after span proof
and re-proof commands succeed.

Never forges KERNEL_VERIFIED. Never sets completion_authoritative.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from .contracts import RepairDisposition, RepairWorkItem
from .mcp_surface_resolution import SurfaceHit

ADMITTED_EDIT_PLAN_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/admitted-edit-plan@1"
ADMITTED_EDIT_PLAN_INTERFACE: Final = "AdmittedEditPlan@1"


@dataclass
class AdmittedEditPlan:
    """Body-free edit plan for one single-path (or collapse) work item."""

    plan_id: str
    work_id: str
    operation: str
    disposition: str
    preferred_path: str = ""
    handler: str | None = None
    registration_api: str | None = None
    doctor_operator: str = ""
    aliases: tuple[str, ...] = ()
    idl_methods: tuple[str, ...] = ()
    body_free: bool = True
    proof_admitted: bool = False
    materialize_ready: bool = False
    implementable: bool = False
    source_edit_operator: dict[str, Any] | None = None
    materialization_disposition: str = "analysis_only"
    materialize_preconditions: tuple[str, ...] = ()
    re_proof_commands: tuple[str, ...] = ()
    predicted_files: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    ordered_steps: tuple[str, ...] = ()
    identity_bindings: dict[str, Any] = field(default_factory=dict)
    doctor_proposal: dict[str, Any] | None = None
    ir_family_ok: dict[str, bool] = field(default_factory=dict)
    domain: str = "agent_supervisor"
    recorded_at: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = ADMITTED_EDIT_PLAN_SCHEMA
        payload["interface"] = ADMITTED_EDIT_PLAN_INTERFACE
        payload["completion_authoritative"] = False
        payload["grants_execution_authority"] = False
        payload["required_assurance"] = "solver_checked_then_reproof"
        return payload


def _plan_id(work_id: str, operation: str, path: str) -> str:
    digest = hashlib.sha256(f"{work_id}|{operation}|{path}".encode()).hexdigest()[:16]
    return f"edit-plan:{digest}"


def _default_reproof_commands(domain: str) -> tuple[str, ...]:
    cmds = [
        "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets:external/ipfs_kit "
        "python3 scripts/autonomous_supervisor_repair.py --domain "
        f"{domain} --op <operation>",
    ]
    if domain == "sca":
        cmds.extend(
            [
                "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets:external/ipfs_kit "
                "python3 scripts/sca_symbolic_repair_ready.py",
                "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets:external/ipfs_kit "
                "python3 scripts/sca_bind_kernel_receipts_to_board.py",
            ]
        )
    return tuple(cmds)


def build_body_free_doctor_proposal(
    *,
    operation: str,
    preferred_path: str,
    doctor_operator: str,
    registration_api: str | None,
    domain: str,
) -> dict[str, Any] | None:
    """Best-effort body-free doctor proposal (proof_admitted=False).

    Uses the closed deterministic doctor operator registry when available.
    Failures are non-fatal — the edit plan still stands without a proposal.
    """
    try:
        from ..analysis.deterministic_doctor_contracts import DoctorAuthorityRoots
        from ..planning.deterministic_doctor_transforms import (
            DoctorOperatorKind,
            build_default_doctor_operator_registry,
            make_edit_site,
        )

        fields = [
            f for f in DoctorAuthorityRoots.__dataclass_fields__ if f not in {"SCHEMA", "lease_id"}
        ]
        roots = DoctorAuthorityRoots(**{f: f"auto-repair:{domain}:{f}" for f in fields})
        registry = build_default_doctor_operator_registry(roots)
        site = make_edit_site(preferred_path or f"surface:{operation}", operation)

        # Prefer registration operator when we have a registration API; else rename.
        kind = DoctorOperatorKind.ADD_REGISTRATION
        kwargs: dict[str, Any] = {
            "kind": kind,
            "edit_site": site,
            "obligation_refs": (f"obligation:{operation}",),
            "registration_name": operation,
            "registration_target": registration_api or "register_tool",
            "proof_admitted": False,
        }
        # Fall back if ADD_REGISTRATION not usable
        try:
            proposal = registry.propose(**kwargs)
        except Exception:
            proposal = registry.propose(
                kind=DoctorOperatorKind.EXACT_RENAME,
                edit_site=site,
                obligation_refs=(f"obligation:{operation}",),
                parameter_name=operation,
                previous_parameter_name=operation,
                proof_admitted=False,
            )
        receipt = registry.evaluate(proposal)
        return {
            "proposal": proposal.to_dict()
            if hasattr(proposal, "to_dict")
            else {"proposal_id": getattr(proposal, "proposal_id", "")},
            "evaluate_disposition": str(
                getattr(getattr(receipt, "disposition", None), "value", receipt.disposition)
            ),
            "rejection_reasons": list(getattr(receipt, "rejection_reasons", ()) or ()),
            "admitted": bool(getattr(receipt, "admitted", False)),
            "body_free": True,
            "proof_admitted": False,
            "sca_doctor_operator": doctor_operator,
            "note": "proposal is body-free; render_admitted requires proof_admitted + span hash",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "body_free": True,
            "proof_admitted": False,
            "sca_doctor_operator": doctor_operator,
        }


def materialize_admitted_edit_plan(
    *,
    work: RepairWorkItem,
    disposition: RepairDisposition | str,
    surface: SurfaceHit | None,
    doctor: Mapping[str, Any],
    ir_doc: Mapping[str, Any],
    aliases: Sequence[str],
    idl_methods: Sequence[str],
    allow_code_edit_materialize: bool = False,
    domain: str = "agent_supervisor",
) -> AdmittedEditPlan | None:
    """Build a body-free admitted edit plan for single-path (and optional collapse).

    Returns None for blocked/missing items that cannot form an edit plan.
    """
    disp = (
        disposition
        if isinstance(disposition, RepairDisposition)
        else RepairDisposition(str(disposition))
    )
    if disp not in {
        RepairDisposition.SINGLE_PATH_READY,
        RepairDisposition.MULTI_PATH_COLLAPSE,
    }:
        return None

    preferred = (surface.preferred_path if surface else None) or work.path or ""
    handler = surface.handler if surface else None
    reg_api = surface.registration_api if surface else None
    doctor_op = str(doctor.get("operator") or "analytical_transform")

    # This builder deliberately has no source bytes, diff, inverse, or
    # owner-root admission.  A policy flag cannot turn body-free analysis into
    # a mutable edit; an external typed source-edit operator must be bound
    # later by the materializer.
    materialize_ready = False

    preconditions = [
        "body_free_plan_only_until_span_bound",
        "doctor_operator_closed_set",
        "preferred_path_must_exist_in_tree",
        "reindex_runtime_components_after_edit",
        "reproof_required_before_board_completion",
        "no_llm_implement",
    ]
    if disp is RepairDisposition.MULTI_PATH_COLLAPSE:
        preconditions.append("collapse_to_single_mcp_server_anchor_before_edit")
    if allow_code_edit_materialize:
        preconditions.append("policy_flag_is_not_source_edit_admission")
    preconditions.extend(
        (
            "typed_admitted_source_edit_operator_required",
            "exact_old_new_byte_digests_required",
            "nonempty_forward_diff_and_inverse_required",
            "owner_root_and_relative_path_binding_required",
            "post_write_validation_pending_required",
        )
    )

    postconditions = [
        f"operation `{work.operation}` resolves to unique preferred surface",
        "GUI/ORB/IDL aliases remain consistent with package MCP tools/list",
        "mediation path prefers package_mcp_interop / tools/call",
        "claim KERNEL_VERIFIED only via observation-bound kernel receipts",
    ]

    steps = [
        f"Load admitted edit plan for {work.operation}",
        f"Bind preferred path `{preferred}`" + (f" handler=`{handler}`" if handler else ""),
        f"Execute doctor operator `{doctor_op}` only under closed analytical path",
    ]
    if disp is RepairDisposition.MULTI_PATH_COLLAPSE:
        steps.append("Collapse multi-match anchors before any source rewrite")
    steps.append(
        "Plan is analysis/admission only; a separate typed source-edit operator "
        "must bind exact before/after bytes before any mutation"
    )
    steps.extend(
        [
            "Re-run autonomous_supervisor_repair surface resolution",
            "Run domain re-proof commands",
            "Do not mark board completion authoritative",
        ]
    )

    proposal = None
    if preferred:
        proposal = build_body_free_doctor_proposal(
            operation=work.operation,
            preferred_path=preferred,
            doctor_operator=doctor_op,
            registration_api=reg_api,
            domain=domain,
        )

    identity = {
        "contract_id": work.contract_id,
        "package": work.package,
        "canonical_surface": surface.canonical if surface else "",
        "match_count": surface.match_count if surface else 0,
        "surface_status": surface.status if surface else "unknown",
        "content_binding": hashlib.sha256(
            json.dumps(
                {
                    "op": work.operation,
                    "path": preferred,
                    "handler": handler,
                    "aliases": list(aliases),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }

    return AdmittedEditPlan(
        plan_id=_plan_id(work.work_id, work.operation, preferred),
        work_id=work.work_id,
        operation=work.operation,
        disposition=disp.value,
        preferred_path=preferred,
        handler=handler,
        registration_api=reg_api,
        doctor_operator=doctor_op,
        aliases=tuple(aliases),
        idl_methods=tuple(idl_methods),
        body_free=True,
        proof_admitted=False,
        materialize_ready=materialize_ready,
        implementable=False,  # never implementable without re-proof gate
        source_edit_operator=None,
        materialization_disposition="analysis_only",
        materialize_preconditions=tuple(preconditions),
        re_proof_commands=_default_reproof_commands(domain),
        predicted_files=tuple(
            p for p in ((preferred,) if preferred else ()) + tuple(work.write_paths) if p
        ),
        postconditions=tuple(postconditions),
        ordered_steps=tuple(steps),
        identity_bindings=identity,
        doctor_proposal=proposal,
        ir_family_ok=dict(ir_doc.get("family_ok") or {}),
        domain=domain,
        recorded_at=datetime.now(UTC).isoformat(),
        notes=(
            "Body-free admitted edit plan from AutonomousRepairEngine.",
            "Does not contain source body text.",
            "Cannot count as applied/success/completed without a typed admitted source edit.",
        ),
    )


def write_edit_plans(
    plans: Sequence[AdmittedEditPlan],
    *,
    output_dir: str | Path,
) -> list[str]:
    """Persist each plan as JSON; return written paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    written: list[str] = []
    index: list[dict[str, Any]] = []
    for plan in plans:
        name = f"{plan.plan_id.replace(':', '_')}.json"
        path = out_dir / name
        path.write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written.append(str(path))
        index.append(
            {
                "plan_id": plan.plan_id,
                "work_id": plan.work_id,
                "operation": plan.operation,
                "disposition": plan.disposition,
                "materialize_ready": plan.materialize_ready,
                "path": str(path),
            }
        )
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/admitted-edit-plan-index@1",
                "recorded_at": datetime.now(UTC).isoformat(),
                "count": len(index),
                "materialize_ready_count": sum(1 for p in plans if p.materialize_ready),
                "plans": index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return written


__all__ = [
    "ADMITTED_EDIT_PLAN_INTERFACE",
    "ADMITTED_EDIT_PLAN_SCHEMA",
    "AdmittedEditPlan",
    "build_body_free_doctor_proposal",
    "materialize_admitted_edit_plan",
    "write_edit_plans",
]
