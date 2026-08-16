"""Apply shared-IR + structural IR logic to intermediate representations.

**General agent-supervisor module** — not bound to the SCA taskboard. Any
consumer (planner, doctor, symbolic repair, contract repair, SCA, generic
work items) supplies an :class:`IrWorkSurface` or field kwargs.

Shared IR (Intent / Legal / Security / UI)
  materialize → normalize → compile → (security evaluate) → optional admission

Structural IR (AST / knowledge graph / vector index)
  project body-free AST → index/query → deterministic vector index →
  semantic dependency graph + mandatory closure

Authority rules (fail-closed):

* Intent describes required work; never authorizes.
* Legal applicability is context; never a capability grant.
* Security evaluation alone may PERMIT/DENY/UNKNOWN.
* UI / AST / KG / vector hits are non-authoritative intermediate context.
* KERNEL_VERIFIED remains observation-bound outside this module.
* LLM implement remains proposal_only under RPR where applicable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Mapping, Sequence

IR_LOGIC_APPLICATION_INTERFACE: Final = "IrLogicApplication@1"
SCA_IR_LOGIC_APPLICATOR_INTERFACE: Final = IR_LOGIC_APPLICATION_INTERFACE  # compat
IR_LOGIC_APPLY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-apply@1"
)
SCA_IR_LOGIC_APPLY_SCHEMA: Final = IR_LOGIC_APPLY_SCHEMA  # compat
IR_LOGIC_BATCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-apply-batch@1"
)
SCA_IR_LOGIC_BATCH_SCHEMA: Final = IR_LOGIC_BATCH_SCHEMA  # compat

DEFAULT_APPLY_FAMILIES: Final[tuple[str, ...]] = (
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
    "ast",
    "knowledge_graph",
    "vector_index",
)

SHARED_IR_FAMILIES: Final[tuple[str, ...]] = (
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
)
STRUCTURAL_IR_FAMILIES: Final[tuple[str, ...]] = (
    "ast",
    "knowledge_graph",
    "vector_index",
)

def _source_for_domain(domain: str = "agent_supervisor") -> tuple[dict[str, str], ...]:
    d = (domain or "agent_supervisor").strip() or "agent_supervisor"
    return ({"source_id": f"{d}:ir-logic", "span_id": "work-surface"},)


# Default source (generic supervisor domain)
_SOURCE: Final[tuple[dict[str, str], ...]] = _source_for_domain("agent_supervisor")
_FLOW: Final[dict[str, str]] = {
    "classification": "source",
    "direction": "workspace_to_tool",
}


class IrLogicApplicationError(ValueError):
    """Malformed IR logic application request."""


ScaIrLogicApplicatorError = IrLogicApplicationError  # compat


@dataclass
class IrLogicApplyPolicy:
    """Policy for applying logic families to intermediate representations."""

    families: tuple[str, ...] = DEFAULT_APPLY_FAMILIES
    evaluate_security: bool = True
    security_decision: str = "allow"  # IR policy decision under test / mediation path
    include_plan_admission: bool = False
    legal_modality: str = "obligation"  # work item: mediation obligation
    max_surfaces: int = 8
    fail_closed_on_unsupported: bool = False  # report, don't fail stack by default

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "IrLogicApplyPolicy":
        raw = dict(raw or {})
        fams = raw.get("families") or raw.get("analysisFamilies")
        return cls(
            families=tuple(fams or DEFAULT_APPLY_FAMILIES),
            evaluate_security=bool(
                raw.get("evaluateSecurity", raw.get("evaluate_security", True))
            ),
            security_decision=str(
                raw.get("securityDecision")
                or raw.get("security_decision")
                or "allow"
            ),
            include_plan_admission=bool(
                raw.get(
                    "includePlanAdmission",
                    raw.get("include_plan_admission", False),
                )
            ),
            legal_modality=str(
                raw.get("legalModality") or raw.get("legal_modality") or "obligation"
            ),
            max_surfaces=int(raw.get("maxSurfaces") or raw.get("max_surfaces") or 8),
            fail_closed_on_unsupported=bool(
                raw.get(
                    "failClosedOnUnsupported",
                    raw.get("fail_closed_on_unsupported", False),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IrWorkSurface:
    """Domain-agnostic work surface for IR logic application.

    Not SCA-specific. Construct from planner goals, doctor findings, repair
    tasks, contract operations, or any supervisor work item.
    """

    operation: str
    kind: str = "work_item"
    contract_id: str = ""
    path: str = ""
    symbol: str = ""
    finding_id: str = ""
    goal_id: str = ""
    program_id: str = ""
    domain: str = "agent_supervisor"
    consumer: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "IrWorkSurface":
        raw = dict(value or {})
        op = str(
            raw.get("operation")
            or raw.get("op")
            or raw.get("tool")
            or raw.get("action")
            or ""
        )
        contract_id = str(
            raw.get("contract_id")
            or raw.get("operation_id")
            or raw.get("surface_id")
            or ""
        )
        if not op and contract_id and ":" in contract_id:
            op = contract_id.split(":", 1)[-1]
        if not op:
            op = str(raw.get("symbol") or raw.get("name") or "unknown")
        return cls(
            operation=op,
            kind=str(
                raw.get("kind")
                or raw.get("finding_kind")
                or raw.get("reason_code")
                or raw.get("work_kind")
                or "work_item"
            ),
            contract_id=contract_id or f"surface:{op}",
            path=str(raw.get("path") or raw.get("file") or ""),
            symbol=str(raw.get("symbol") or raw.get("qualified_symbol") or ""),
            finding_id=str(
                raw.get("finding_id") or raw.get("id") or raw.get("task_id") or ""
            ),
            goal_id=str(raw.get("goal_id") or raw.get("objective_id") or ""),
            program_id=str(raw.get("program_id") or raw.get("repository_id") or ""),
            domain=str(raw.get("domain") or raw.get("program") or "agent_supervisor"),
            consumer=str(raw.get("consumer") or "generic"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: str, *, maximum: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "unknown").strip())
    cleaned = cleaned.strip("-._") or "unknown"
    return cleaned[:maximum]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{_slug(parts[-1] if parts else 'x')}:{digest}"


def _result_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return dict(value.to_dict())
        except Exception:  # noqa: BLE001
            pass
    out: dict[str, Any] = {"type": type(value).__name__}
    for attr in (
        "status",
        "outcome",
        "successful",
        "permitted",
        "content_id",
        "compilation_id",
        "query_id",
        "request_id",
        "reason_codes",
        "grants_execution_authority",
    ):
        if hasattr(value, attr):
            v = getattr(value, attr)
            if hasattr(v, "value"):
                v = v.value
            out[attr] = v
    return out


def _status_value(value: Any) -> str:
    status = getattr(value, "status", None)
    if status is None and isinstance(value, Mapping):
        status = value.get("status")
    if hasattr(status, "value"):
        return str(status.value)
    return str(status or "")


def project_candidate_plan(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "work_item",
    finding_id: str = "",
    domain: str = "agent_supervisor",
    goal_id: str = "",
    program_id: str = "",
) -> dict[str, Any]:
    """Project any work surface into a candidate plan graph for IR binding."""
    op = operation or (contract_id.split(":", 1)[-1] if contract_id else "") or "unknown"
    op_slug = _slug(op)
    domain_slug = _slug(domain or "agent_supervisor")
    prepare_id = f"action:index-{op_slug}"
    repair_id = f"action:repair-{op_slug}"
    verify_id = f"action:verify-{op_slug}"
    effect_prepare = {
        "effect_id": f"effect:index-{op_slug}",
        "action_id": prepare_id,
        "operation": "index",
        "target": f"surface:{op_slug}",
    }
    effect_repair = {
        "effect_id": f"effect:register-{op_slug}",
        "action_id": repair_id,
        "operation": "register_tool",
        "target": f"mcp:{op_slug}",
    }
    effect_verify = {
        "effect_id": f"effect:reindex-{op_slug}",
        "action_id": verify_id,
        "operation": "reindex",
        "target": f"surface:{op_slug}",
    }
    actions = [
        {
            "action_id": prepare_id,
            "principal": "principal:agent",
            "action": "index",
            "tool": "tool:mcp-server",
            "target": f"resource:contract:{op_slug}",
            "requested_authority": "analysis",
            "depends_on": [],
            "effects": [effect_prepare],
        },
        {
            "action_id": repair_id,
            "principal": "principal:agent",
            "action": "repair",
            "tool": "tool:mcp-server",
            "target": f"resource:contract:{op_slug}",
            "requested_authority": "mutation",
            "depends_on": [prepare_id],
            "effects": [effect_repair],
        },
        {
            "action_id": verify_id,
            "principal": "principal:agent",
            "action": "verify",
            "tool": "tool:mcp-server",
            "target": f"resource:contract:{op_slug}",
            "requested_authority": "analysis",
            "depends_on": [repair_id],
            "effects": [effect_verify],
        },
    ]
    return {
        "plan_id": _stable_id("plan", domain_slug, op, finding_kind),
        "schema": "ir-candidate-plan@1",
        "operation": op,
        "contract_id": contract_id or f"surface:{op}",
        "finding_kind": finding_kind,
        "finding_id": finding_id,
        "goal_id": goal_id,
        "program_id": program_id,
        "domain": domain,
        "actions": actions,
        "effects": [effect_prepare, effect_repair, effect_verify],
        "mediation_required": True,
        "mcp_preferred": True,
    }


def _load_normalized(family: Any, **sections: Any) -> Any:
    from .ir_adapters import IRAdapterRegistry
    from .ir_registry import (
        IRLoadRequest,
        IRLoadStatus,
        IRRegistry,
        deterministic_ir_fixture,
    )

    reference, encoded = deterministic_ir_fixture(family, **sections)
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    loaded = registry.load(IRLoadRequest(reference=reference, family=family))
    if loaded.status is not IRLoadStatus.VERIFIED:
        raise ScaIrLogicApplicatorError(
            f"IR load not verified for {getattr(family, 'value', family)}: "
            f"{loaded.status}"
        )
    result = IRAdapterRegistry().normalize(loaded)
    return result.require_artifact()


def apply_intent_logic(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build IntentIR + formalization, normalize, compile action constraints."""
    from .intent_constraint_adapter import (
        IntentCompilationStatus,
        compile_intent_constraints,
        create_intent_conformance_request,
    )
    from .ir_registry import IRFamily

    op = str(candidate.get("operation") or "unknown")
    op_slug = _slug(op)
    goal_id = f"goal:repair-{op_slug}"
    actions = list(candidate.get("actions") or [])
    declarations: list[dict[str, Any]] = [
        {
            "id": goal_id,
            "kind": "goal",
            "grounded": True,
            "operation": op,
            "contract_id": candidate.get("contract_id"),
            "finding_kind": candidate.get("finding_kind"),
        }
    ]
    sequence_ids: list[str] = []
    for action in actions:
        action_id = str(action["action_id"])
        sequence_ids.append(action_id)
        declarations.append(
            {
                "id": action_id,
                "kind": "action",
                "goal_id": goal_id,
                "depends_on": list(action.get("depends_on") or []),
                "grounded": True,
            }
        )
        for effect in action.get("effects") or []:
            declarations.append(
                {
                    "id": str(effect["effect_id"]),
                    "kind": "effect",
                    "action_id": action_id,
                    "operation": effect.get("operation"),
                    "target": effect.get("target"),
                    "grounded": True,
                }
            )
        declarations.append(
            {
                "id": f"precondition:{action_id}",
                "kind": "precondition",
                "action_id": action_id,
                "statement_id": f"statement:mcp-mediated:{op_slug}",
                "grounded": True,
            }
        )
        declarations.append(
            {
                "id": f"verification:{action_id}",
                "kind": "verification",
                "action_id": action_id,
                "evidence_id": f"evidence:reindex:{op_slug}",
                "grounded": True,
            }
        )
    if sequence_ids:
        declarations.append(
            {
                "id": f"flow:seq-{op_slug}",
                "kind": "sequence",
                "sequence": sequence_ids,
                "grounded": True,
            }
        )

    intent = _load_normalized(IRFamily.INTENT, declarations=tuple(declarations))
    formalization = _load_normalized(
        IRFamily.FORMALIZATION,
        declarations=(
            {
                "id": f"statement:mcp-mediated:{op_slug}",
                "kind": "statement",
                "grounded": True,
            },
            {
                "id": f"evidence:reindex:{op_slug}",
                "kind": "statement",
                "grounded": True,
            },
        ),
        formal_views=(
            {
                "view_id": f"view:fol-{op_slug}",
                "view_kind": "first_order",
                "grounded": True,
            },
        ),
    )
    compilation = compile_intent_constraints(intent, formalization)
    status = _status_value(compilation)
    constraint_set = None
    constraint_count = 0
    obligation_ids: list[str] = []
    conformance: dict[str, Any] = {}
    try:
        if getattr(compilation, "status", None) is IntentCompilationStatus.COMPILED:
            constraint_set = compilation.require_constraint_set()
            constraint_count = len(getattr(constraint_set, "constraints", ()) or ())
            obligation_ids = [
                item.obligation_id
                for item in getattr(constraint_set, "proof_obligations", ()) or ()
            ]
            intent_candidate = copy.deepcopy(dict(candidate))
            intent_candidate["intent_root"] = dict(
                getattr(constraint_set, "intent_root", {}) or {}
            )
            intent_candidate["formalization_root"] = dict(
                getattr(constraint_set, "formalization_root", {}) or {}
            )
            intent_candidate["goal_ids"] = [goal_id]
            conf_req = create_intent_conformance_request(
                compilation,
                intent_candidate,
                discharged_obligation_ids=tuple(obligation_ids),
            )
            conformance = {
                "request_available": conf_req is not None,
                "request_type": type(conf_req).__name__,
                "content_id": getattr(conf_req, "content_id", None),
            }
    except Exception as exc:  # noqa: BLE001
        conformance = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "family": "intent_ir",
        "applied": True,
        "available": True,
        "status": status,
        "ok": status in {"compiled", "complete", IntentCompilationStatus.COMPILED.value},
        "grants_execution_authority": False,
        "role": "required_work_constraints",
        "artifact": {
            "root_artifact_id": getattr(intent, "root_artifact_id", None),
            "root_cid_v1": getattr(intent, "root_cid_v1", None),
            "root_supervisor_digest": getattr(intent, "root_supervisor_digest", None),
            "family": IRFamily.INTENT.value,
        },
        "formalization": {
            "root_artifact_id": getattr(formalization, "root_artifact_id", None),
            "root_cid_v1": getattr(formalization, "root_cid_v1", None),
            "root_supervisor_digest": getattr(
                formalization, "root_supervisor_digest", None
            ),
            "family": IRFamily.FORMALIZATION.value,
        },
        "constraint_count": constraint_count,
        "obligation_ids": obligation_ids,
        "compilation": _result_to_dict(compilation),
        "conformance": conformance,
        "logic_applied": [
            "IRRegistry.load+verify",
            "IRAdapterRegistry.normalize(intent_ir)",
            "IRAdapterRegistry.normalize(formalization)",
            "compile_intent_constraints",
            "create_intent_conformance_request",
        ],
        "notes": [
            "IntentIR constraints describe required repair work only.",
            "Intent never authorizes MCP effect execution.",
        ],
    }


def apply_legal_logic(
    candidate: Mapping[str, Any],
    *,
    modality: str = "obligation",
) -> dict[str, Any]:
    """Build LegalIR norms for the surface, normalize, compile applicability."""
    from .ir_registry import IRFamily
    from .legal_constraint_adapter import (
        LegalApplicabilityQuery,
        LegalCompilationStatus,
        compile_legal_constraints,
    )

    op = str(candidate.get("operation") or "unknown")
    op_slug = _slug(op)
    scope = {
        "jurisdiction": str(candidate.get("domain") or "agent_supervisor"),
        "subject": "mcp-contract",
        "principal": "principal:agent",
        "action": "repair",
        "resource": f"resource:contract:{op_slug}",
        "effect": "register_tool",
    }
    declaration = {
        "declaration_id": f"norm:{modality}-mediate-{op_slug}",
        "kind": "norm",
        "modality": modality,
        **scope,
        "effective_from_ms": 1,
        "effective_until_ms": 2_000_000_000_000,
        "source_references": _SOURCE,
        "grounded": True,
        "operation": op,
        "contract_id": candidate.get("contract_id"),
        "finding_kind": candidate.get("finding_kind"),
    }
    # Residual mediation: prohibition of direct non-MCP path when path_class ambiguous
    extra: list[dict[str, Any]] = []
    if "path_class" in str(candidate.get("finding_kind") or ""):
        extra.append(
            {
                "declaration_id": f"norm:prohibit-direct-{op_slug}",
                "kind": "norm",
                "modality": "prohibition",
                **scope,
                "action": "direct_import",
                "effect": "bypass_mcp",
                "effective_from_ms": 1,
                "effective_until_ms": 2_000_000_000_000,
                "source_references": _SOURCE,
                "grounded": True,
            }
        )

    artifact = _load_normalized(
        IRFamily.LEGAL,
        declarations=tuple([declaration, *extra]),
    )
    query = LegalApplicabilityQuery(
        legal_root_artifact_id=artifact.root_artifact_id,
        legal_root_cid_v1=artifact.root_cid_v1,
        legal_root_supervisor_digest=artifact.root_supervisor_digest,
        **scope,
        effective_at_ms=500,
    )
    compilation = compile_legal_constraints(artifact, query)
    status = _status_value(compilation)
    constraints = getattr(compilation, "constraints", ()) or ()
    return {
        "family": "legal_ir",
        "applied": True,
        "available": True,
        "status": status,
        "ok": status
        in {
            "complete",
            "compiled",
            getattr(LegalCompilationStatus.COMPLETE, "value", "complete"),
        },
        "grants_execution_authority": False,
        "role": "applicability_constraints",
        "artifact": {
            "root_artifact_id": artifact.root_artifact_id,
            "root_cid_v1": artifact.root_cid_v1,
            "root_supervisor_digest": artifact.root_supervisor_digest,
            "family": IRFamily.LEGAL.value,
        },
        "modality": modality,
        "constraint_count": len(constraints),
        "compilation": _result_to_dict(compilation),
        "logic_applied": [
            "IRRegistry.load+verify",
            "IRAdapterRegistry.normalize(legal_ir)",
            "compile_legal_constraints",
        ],
        "notes": [
            "LegalIR applicability is context for work item.",
            "Legal permission/obligation is never an execution grant.",
        ],
        "_raw_compilation": compilation,  # internal for plan admission
    }


def apply_security_logic(
    candidate: Mapping[str, Any],
    *,
    decision: str = "allow",
    evaluate: bool = True,
) -> dict[str, Any]:
    """Build SecurityIR policy for the surface, compile, optionally evaluate."""
    from .ir_registry import IRFamily
    from .security_constraint_adapter import (
        SecurityAuthorizationRequest,
        SecurityCompilationStatus,
        compile_security_constraints,
        evaluate_security_authorization,
    )

    op = str(candidate.get("operation") or "unknown")
    op_slug = _slug(op)
    actions = list(candidate.get("actions") or [])
    policies: list[dict[str, Any]] = []
    for action in actions:
        effect = (action.get("effects") or [{}])[0]
        policies.append(
            {
                "declaration_id": f"policy:{action['action_id']}",
                "kind": "policy",
                "decision": decision,
                "principal": "principal:agent",
                "action": str(action.get("action") or "repair"),
                "tool": "tool:mcp-server",
                "target": f"resource:contract:{op_slug}",
                "data_flow": _FLOW,
                "expected_effect": {
                    "operation": effect.get("operation"),
                    "target": effect.get("target"),
                },
                "requested_authority": str(
                    action.get("requested_authority") or "mutation"
                ),
                "source_references": _SOURCE,
                "operation": op,
                "contract_id": candidate.get("contract_id"),
            }
        )
    declarations = (
        {
            "declaration_id": "principal:agent",
            "kind": "principal",
            "source_references": _SOURCE,
        },
        {
            "declaration_id": "tool:mcp-server",
            "kind": "resource",
            "resource_type": "tool",
            "source_references": _SOURCE,
        },
        {
            "declaration_id": f"resource:contract:{op_slug}",
            "kind": "resource",
            "resource_type": "contract",
            "source_references": _SOURCE,
        },
        *policies,
    )
    artifact = _load_normalized(IRFamily.SECURITY, declarations=declarations)
    policy = compile_security_constraints(artifact)
    status = _status_value(policy)
    evaluations: list[dict[str, Any]] = []
    raw_requests: list[Any] = []
    if evaluate and status in {
        "compiled",
        getattr(SecurityCompilationStatus.COMPILED, "value", "compiled"),
    }:
        for action in actions:
            effect = (action.get("effects") or [{}])[0]
            req = SecurityAuthorizationRequest(
                security_root_artifact_id=artifact.root_artifact_id,
                security_root_cid_v1=artifact.root_cid_v1,
                security_root_supervisor_digest=artifact.root_supervisor_digest,
                principal="principal:agent",
                action=str(action.get("action") or "repair"),
                tool="tool:mcp-server",
                target=f"resource:contract:{op_slug}",
                data_flow=_FLOW,
                expected_effect={
                    "operation": effect.get("operation"),
                    "target": effect.get("target"),
                },
                current_state={"contract": "incomplete"},
                requested_authority=str(
                    action.get("requested_authority") or "mutation"
                ),
                evaluated_at_ms=500,
            )
            raw_requests.append(req)
            try:
                receipt = evaluate_security_authorization(policy, req)
                evaluations.append(
                    {
                        "action_id": action.get("action_id"),
                        "outcome": _status_value(receipt)
                        if False
                        else str(
                            getattr(
                                getattr(receipt, "outcome", None),
                                "value",
                                getattr(receipt, "outcome", None),
                            )
                        ),
                        "permitted": bool(getattr(receipt, "permitted", False)),
                        "grants_execution_authority": bool(
                            getattr(receipt, "grants_execution_authority", False)
                        ),
                        "reason_codes": list(
                            getattr(receipt, "reason_codes", ()) or ()
                        ),
                        "receipt": _result_to_dict(receipt),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                evaluations.append(
                    {
                        "action_id": action.get("action_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "permitted": False,
                        "grants_execution_authority": False,
                    }
                )

    return {
        "family": "security_ir",
        "applied": True,
        "available": True,
        "status": status,
        "ok": status
        in {
            "compiled",
            getattr(SecurityCompilationStatus.COMPILED, "value", "compiled"),
        },
        "grants_execution_authority": False,
        "role": "authorization_decision_inputs",
        "artifact": {
            "root_artifact_id": artifact.root_artifact_id,
            "root_cid_v1": artifact.root_cid_v1,
            "root_supervisor_digest": artifact.root_supervisor_digest,
            "family": IRFamily.SECURITY.value,
        },
        "policy_decision": decision,
        "evaluations": evaluations,
        "compilation": _result_to_dict(policy),
        "logic_applied": [
            "IRRegistry.load+verify",
            "IRAdapterRegistry.normalize(security_ir)",
            "compile_security_constraints",
            *(["evaluate_security_authorization"] if evaluate else []),
        ],
        "notes": [
            "SecurityIR evaluation is fail-closed; unknown/deny do not grant.",
            "Compilation and evaluation receipts are not execution permits.",
        ],
        "_raw_policy": policy,
        "_raw_requests": tuple(raw_requests),
        "_raw_artifact": artifact,
    }


def _project_work_surface_uiir_document(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build a minimal closed ui-ux-ir/v1 document for a work surface.

    Descriptor-only: never forges interface CIDs or grants execution.
    """
    import hashlib

    op = str(candidate.get("operation") or "unknown")
    op_slug = _slug(op)
    contract_id = str(candidate.get("contract_id") or f"surface:{op}")
    path = str(candidate.get("path") or f"agent_supervisor/work_surfaces/{op_slug}.py")
    domain = str(candidate.get("domain") or "agent_supervisor")
    requested_risk = str(candidate.get("risk_class") or "low").lower()
    risk_class = (
        requested_risk
        if requested_risk in {"low", "medium", "high", "critical"}
        else "high"
    )
    confirmation_class = str(
        candidate.get("confirmation_class")
        or ("explicit" if risk_class in {"high", "critical"} else "none")
    )
    source_uri = f"workspace://{domain}/{path}"
    content_seed = (
        f"{domain}|{op}|{contract_id}|{path}|{risk_class}|{confirmation_class}"
    ).encode()
    content_sha = hashlib.sha256(content_seed).hexdigest()
    source_ref = f"source:{op_slug}"
    root_id = f"component:{op_slug}:root"
    action_id = f"component:{op_slug}:action"
    return {
        "schema_version": "ui-ux-ir/v1",
        "document_id": f"doc:work-surface:{op_slug}",
        "title": f"Work surface UI for {op}",
        "sources": [
            {
                "ref_id": source_ref,
                "source_uri": source_uri,
                "source_id": op_slug,
                "source_revision": "workspace",
                "content_sha256": content_sha,
                "container_uri": "",
                "container_sha256": "",
                "content_cid": "",
                "license_expression": "",
                "review_status": "machine_extracted",
                "span": None,
            }
        ],
        "components": [
            {
                "component_id": root_id,
                "role": "panel",
                "purpose": f"Surface for operation {op}",
                "accessible_name_ref": "",
                "accessible_description_ref": "",
                "parent_id": "",
                "child_ids": [action_id],
                "modality_binding_ids": [],
                "data_binding_ids": [],
                "program_binding_ids": [],
                "feedback_ids": [],
                "privacy_sensitivity": "none",
                "presentation_classification": "interactive",
                "source_ref_ids": [source_ref],
            },
            {
                "component_id": action_id,
                "role": "button",
                "purpose": f"Invoke {op}",
                "accessible_name_ref": "",
                "accessible_description_ref": "",
                "parent_id": root_id,
                "child_ids": [],
                "modality_binding_ids": [],
                "data_binding_ids": [],
                "program_binding_ids": [],
                "feedback_ids": [],
                "privacy_sensitivity": "none",
                "presentation_classification": "interactive",
                "source_ref_ids": [source_ref],
            },
        ],
        "entry_components": [root_id],
        "terminal_outcomes": [
            {
                "outcome_id": f"outcome:{op_slug}:success",
                "kind": "success",
                "description": f"Mediated completion of {op}",
                "source_ref_ids": [source_ref],
            }
        ],
        "program_bindings": [
            {
                "binding_id": f"program:{op_slug}",
                "target_kind": "mcp_idl_interface_method_schema",
                "target_ref": op,
                "confirmation_class": confirmation_class,
                "risk_class": risk_class,
                "effect_ids": [],
                "precondition_ids": [],
                "verification_ids": [],
                "source_ref_ids": [source_ref],
            }
        ],
        "mcp_idl_bindings": [
            {
                "binding_id": f"mcp-idl:{op_slug}",
                "interface_cid": "",  # never forged
                "method_name": op,
                "argument_schema_ref": "",
                "result_schema_ref": "",
                "source_ref_ids": [source_ref],
            }
        ],
    }


def apply_ui_logic(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project UI/interface intermediate descriptors for the work surface.

    Prefer full ``ipfs_datasets_py.logic.ui_ux_ir`` (UIIRDocument decode +
    identity) when available. Always also apply the supervisor
    interface-contract bridge (Python/JS action contract fragments, in-memory).
    """
    from .interface_contract_codegen import (
        JavaScriptActionContractConfig,
        PythonActionContractConfig,
        render_js_action_contract,
        render_python_action_contract,
    )

    op = str(candidate.get("operation") or "unknown")
    op_slug = _slug(op)
    action_id = f"ui.action.{op_slug}"
    definitions: list[dict[str, str]] = [
        {
            "action": action_id,
            "operation": op,
            "id": action_id,
            "label": f"Repair {op}",
            "phrase": f"Apply work surface repair for {op}",
        }
    ]
    # Intermediate representation of the UI/interface surface
    ui_ir_projection = {
        "schema": "ir-ui-projection@1",
        "family": "ui_ir",
        "ui_ir_cid": None,  # not forged as content CID
        "interface_cid": None,
        "operation": op,
        "contract_id": candidate.get("contract_id"),
        "finding_kind": candidate.get("finding_kind"),
        "nodes": [
            {
                "node_id": f"ui:surface:{op_slug}",
                "kind": "surface",
                "operation": op,
            },
            {
                "node_id": f"ui:action:{op_slug}",
                "kind": "action",
                "action_id": action_id,
                "operation": op,
            },
            {
                "node_id": f"ui:binding:{op_slug}",
                "kind": "interface_binding",
                "action_id": action_id,
                "operation": op,
                "mediation": "mcp_tools_call",
            },
        ],
        "edges": [
            {
                "from": f"ui:surface:{op_slug}",
                "to": f"ui:action:{op_slug}",
                "kind": "exposes",
            },
            {
                "from": f"ui:action:{op_slug}",
                "to": f"ui:binding:{op_slug}",
                "kind": "binds",
            },
        ],
        "grants_execution_authority": False,
    }
    py_cfg = PythonActionContractConfig(
        contract_name="IR_UI_ACTION_CONTRACT",
        definitions_name="IR_UI_DEFINITIONS",
        ids_name="IR_UI_ACTION_IDS",
        operations_name="IR_UI_OPERATIONS",
        docstring=f"Generated UI/interface IR projection for work surface op {op}.",
    )
    js_cfg = JavaScriptActionContractConfig(
        contract_name="irUiActionContract",
        ids_name="irUiActionIds",
        ids_set_name="irUiActionIdSet",
        action_by_id_name="irUiActionById",
        operation_by_id_name="irUiOperationById",
        validator_function_name="isScaUiIrActionId",
    )
    py_src = render_python_action_contract(
        definitions, contract="ir_ui", config=py_cfg
    )
    js_src = render_js_action_contract(
        definitions, contract="ir_ui", config=js_cfg
    )

    full_doc: dict[str, Any] = {
        "available": False,
        "ok": False,
    }
    logic_applied = [
        "project_ui_ir_nodes",
        "render_python_action_contract",
        "render_js_action_contract",
    ]
    formalization: dict[str, Any] = {"available": False}
    multi_projection: dict[str, Any] = {"available": False}
    try:
        from ipfs_datasets_py.logic.ui_ux_ir import (
            UI_UX_IR_SCHEMA_VERSION,
            compile_ui_formalization,
            decode_ui_ir,
            evaluate_semantic_roundtrip,
            project_ui_document,
            ui_ir_identity,
            ui_ir_sha256,
        )

        wire = _project_work_surface_uiir_document(candidate)
        document = decode_ui_ir(wire)
        identity = ui_ir_identity(document)
        digest = ui_ir_sha256(document)
        # Digest is content identity for the declaration — not interface_cid.
        ui_ir_projection["ui_ir_digest"] = digest
        ui_ir_projection["ui_ir_schema_version"] = UI_UX_IR_SCHEMA_VERSION

        try:
            formal_art = compile_ui_formalization(document)
            roundtrip = evaluate_semantic_roundtrip(document, artifact=formal_art)
            formalization = {
                "available": True,
                "ok": True,
                "artifact_id": formal_art.artifact_id,
                "view_ids": [v.view_id for v in formal_art.views],
                "coverage_summary": formal_art.coverage_summary(),
                "roundtrip_passed": bool(roundtrip.passed),
                "result_authority": formal_art.result_authority.value,
                "grants_execution_authority": False,
            }
            logic_applied.append("compile_ui_formalization")
            logic_applied.append("evaluate_semantic_roundtrip")
        except Exception as formal_exc:  # noqa: BLE001
            formalization = {
                "available": False,
                "ok": False,
                "error": f"{type(formal_exc).__name__}: {formal_exc}",
                "grants_execution_authority": False,
            }

        try:
            multi_projection = project_ui_document(
                document, targets=("web", "mobile", "glasses")
            )
            logic_applied.append("project_ui_document(web,mobile,glasses)")
        except Exception as proj_exc:  # noqa: BLE001
            multi_projection = {
                "available": False,
                "passed": False,
                "error": f"{type(proj_exc).__name__}: {proj_exc}",
                "grants_execution_authority": False,
            }

        full_doc = {
            "available": True,
            "ok": True,
            "status": "decoded",
            "schema_version": UI_UX_IR_SCHEMA_VERSION,
            "document_id": document.document_id,
            "digest": digest,
            "identity": identity,
            "component_count": len(document.components),
            "program_binding_count": len(document.program_bindings),
            "mcp_idl_binding_count": len(document.mcp_idl_bindings),
            "formalization": formalization,
            "multi_target_projection": {
                "passed": bool(multi_projection.get("passed")),
                "targets": multi_projection.get("targets"),
                "errors": multi_projection.get("errors"),
                "web_nodes": len(
                    ((multi_projection.get("projections") or {}).get("web") or {}).get(
                        "nodes"
                    )
                    or []
                ),
                "mobile_nodes": len(
                    (
                        (multi_projection.get("projections") or {}).get("mobile") or {}
                    ).get("nodes")
                    or []
                ),
                "glasses_status": (
                    (multi_projection.get("projections") or {}).get("glasses") or {}
                ).get("status"),
            },
            "grants_execution_authority": False,
            "package": "ipfs_datasets_py.logic.ui_ux_ir",
            "typescript_peer": "swissknife/src/services/mcp/ui-ux-ir-codec.ts",
        }
        logic_applied.extend(
            [
                "project_work_surface_uiir_document",
                "ipfs_datasets_py.logic.ui_ux_ir.decode_ui_ir",
                "ipfs_datasets_py.logic.ui_ux_ir.ui_ir_sha256",
            ]
        )
    except Exception as exc:  # noqa: BLE001
        full_doc = {
            "available": False,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "grants_execution_authority": False,
            "formalization": formalization,
            "multi_target_projection": multi_projection,
        }

    bridge_only = not bool(full_doc.get("ok"))
    return {
        "family": "ui_ir",
        "applied": True,
        "available": True,
        "status": "decoded" if full_doc.get("ok") else "projected",
        "ok": True,
        "grants_execution_authority": False,
        "role": "interface_descriptor_surface",
        "bridge_only": bridge_only,
        "full_ui_ux_ir": bool(full_doc.get("ok")),
        "ui_ux_ir": full_doc,
        "formalization": formalization,
        "multi_target_projection": multi_projection,
        "projection": ui_ir_projection,
        "action_definitions": definitions,
        "rendered": {
            "python_bytes": len(py_src.encode("utf-8")),
            "javascript_bytes": len(js_src.encode("utf-8")),
            "python_preview": py_src[:240],
            "javascript_preview": js_src[:240],
        },
        "logic_applied": logic_applied,
        "notes": [
            "UI IR projection is descriptor-only intermediate representation.",
            "Never equate ui_ir_digest / ui_ir_cid and interface_cid without identity profile.",
            "Full package: ipfs_datasets_py.logic.ui_ux_ir (Python authority).",
            "TypeScript peer: swissknife ui-ux-ir-codec (cross-language identity).",
            "No execution authority is granted from UI IR alone.",
        ],
    }


def apply_plan_admission(
    candidate: Mapping[str, Any],
    *,
    intent_result: Mapping[str, Any],
    legal_result: Mapping[str, Any],
    security_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Optionally compile plan admission over applied IR constraints.

    Residual SCA default is **not** to grant execution. This path demonstrates
    that IR constraints bind into :class:`IRConstraintCompiler` domains. Any
    admission without a real Security grant source remains fail-closed.
    """
    from .intent_constraint_adapter import (
        IntentCompilationStatus,
        compile_intent_constraints,
        create_intent_conformance_request,
    )
    from .ir_constraint_compiler import (
        ActionDomainBinding,
        AdmissionAuthority,
        PlanAdmissionRequest,
        RootBinding,
        compile_plan_admission,
    )
    from .ir_registry import IRFamily

    # Rebuild structured objects needed by admission (raw payloads not always
    # serializable). Prefer re-applying lightweight compile path.
    try:
        # Use re-application of intent for a live IntentConformanceRequest
        intent_live = apply_intent_logic(candidate)
        # Legal/security already hold raw objects when called in-process
        legal_raw = legal_result.get("_raw_compilation")
        security_policy = security_result.get("_raw_policy")
        security_requests = security_result.get("_raw_requests") or ()
        security_artifact = security_result.get("_raw_artifact")

        if legal_raw is None or security_policy is None:
            return {
                "applied": False,
                "ok": False,
                "status": "skipped",
                "reason": "missing_raw_legal_or_security_compilation",
            }

        # Reconstruct from apply_intent path again for conf request object
        op = str(candidate.get("operation") or "unknown")
        op_slug = _slug(op)
        goal_id = f"goal:repair-{op_slug}"
        # Minimal: call create from recompiled intent in apply_intent_logic path
        # We need the actual request object — re-run compile_intent path internals
        intent_art = _load_normalized(
            IRFamily.INTENT,
            declarations=tuple(
                [
                    {"id": goal_id, "kind": "goal", "grounded": True},
                    *[
                        {
                            "id": a["action_id"],
                            "kind": "action",
                            "goal_id": goal_id,
                            "depends_on": list(a.get("depends_on") or []),
                            "grounded": True,
                        }
                        for a in (candidate.get("actions") or [])
                    ],
                    *[
                        {
                            "id": e["effect_id"],
                            "kind": "effect",
                            "action_id": a["action_id"],
                            "operation": e.get("operation"),
                            "target": e.get("target"),
                            "grounded": True,
                        }
                        for a in (candidate.get("actions") or [])
                        for e in (a.get("effects") or [])
                    ],
                ]
            ),
        )
        formal = _load_normalized(IRFamily.FORMALIZATION)
        compilation = compile_intent_constraints(intent_art, formal)
        if getattr(compilation, "status", None) is not IntentCompilationStatus.COMPILED:
            return {
                "applied": True,
                "ok": False,
                "status": _status_value(compilation),
                "reason": "intent_not_compiled_for_admission",
            }
        constraints = compilation.require_constraint_set()
        intent_candidate = copy.deepcopy(dict(candidate))
        intent_candidate["intent_root"] = dict(constraints.intent_root)
        intent_candidate["formalization_root"] = dict(constraints.formalization_root)
        intent_candidate["goal_ids"] = [goal_id]
        intent_request = create_intent_conformance_request(
            compilation,
            intent_candidate,
            discharged_obligation_ids=tuple(
                item.obligation_id for item in constraints.proof_obligations
            ),
        )

        action_bindings = tuple(
            ActionDomainBinding(
                action_id=str(action["action_id"]),
                legal_result_ids=(getattr(legal_raw, "content_id", ""),),
                security_request_ids=(
                    (getattr(security_requests[i], "content_id", ""),)
                    if i < len(security_requests)
                    else ()
                ),
            )
            for i, action in enumerate(candidate.get("actions") or [])
        )
        tree = f"tree:ir:{_slug(op)}"
        # Fail-closed authority: empty grant sources → admission must not invent
        request = PlanAdmissionRequest(
            candidate_plan=dict(candidate),
            repository_tree_id=tree,
            intent_request=intent_request,
            legal_results=(legal_raw,),
            security_policy=security_policy,
            security_requests=tuple(security_requests),
            action_bindings=action_bindings,
            authority=AdmissionAuthority(
                principal="principal:agent",
                requested_authority="mutation",
                grant_principal="principal:agent",
                granted_authorities=(),
                grant_source_ids=(),  # intentionally empty — fail-closed
            ),
            root_bindings=(
                RootBinding(
                    "intent",
                    intent_art.root_supervisor_digest,
                    intent_art.root_supervisor_digest,
                ),
                RootBinding(
                    "legal",
                    getattr(legal_raw, "legal_root_supervisor_digest", "legal"),
                    getattr(legal_raw, "legal_root_supervisor_digest", "legal"),
                ),
                RootBinding(
                    "security",
                    getattr(security_artifact, "root_supervisor_digest", "security"),
                    getattr(security_artifact, "root_supervisor_digest", "security"),
                ),
                RootBinding("program", tree, tree),
            ),
        )
        receipt = compile_plan_admission(request)
        admitted = bool(getattr(receipt, "admitted", False) or getattr(receipt, "permitted", False))
        grant_ids = list(getattr(receipt, "security_grant_ids", ()) or ())
        return {
            "applied": True,
            "ok": True,  # compiler ran; fail-closed denial is success of logic
            "status": _status_value(receipt) or ("admitted" if admitted else "rejected"),
            "admitted": admitted,
            "security_grant_ids": grant_ids,
            "grants_execution_authority": bool(grant_ids) and admitted,
            "receipt": _result_to_dict(receipt),
            "logic_applied": [
                "IRConstraintCompiler.compile_plan_admission",
                "AdmissionDomain.intent/legal/security",
            ],
            "notes": [
                "Plan admission exercised IR domains over intermediate constraints.",
                "Empty grant_source_ids keep work surface fail-closed (no execution).",
            ],
            "intent_live_ok": bool(intent_live.get("ok")),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "applied": True,
            "ok": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "grants_execution_authority": False,
        }


def apply_logic_to_ir(
    surface: IrWorkSurface | Mapping[str, Any] | None = None,
    *,
    operation: str = "",
    contract_id: str = "",
    finding_kind: str = "work_item",
    finding_id: str = "",
    path: str = "",
    symbol: str = "",
    domain: str = "agent_supervisor",
    goal_id: str = "",
    program_id: str = "",
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    consumer: str = "generic",
) -> dict[str, Any]:
    """Apply selected IR logic families to one work surface's IR projection.

    Accepts a generic :class:`IrWorkSurface` or field kwargs. Domain defaults to
    ``agent_supervisor`` (not SCA). Consumers: planner, doctor, symbolic_repair,
    contract_repair, sca, generic.
    """
    if isinstance(policy, Mapping):
        policy = IrLogicApplyPolicy.from_mapping(policy)
    elif policy is None:
        policy = IrLogicApplyPolicy()

    if surface is not None:
        if not isinstance(surface, IrWorkSurface):
            surface = IrWorkSurface.from_mapping(surface)
        operation = operation or surface.operation
        contract_id = contract_id or surface.contract_id
        finding_kind = finding_kind if finding_kind != "work_item" else surface.kind
        finding_id = finding_id or surface.finding_id
        path = path or surface.path
        symbol = symbol or surface.symbol
        domain = domain if domain != "agent_supervisor" else surface.domain
        goal_id = goal_id or surface.goal_id
        program_id = program_id or surface.program_id
        consumer = consumer if consumer != "generic" else surface.consumer

    op = operation or (
        contract_id.split(":", 1)[-1] if contract_id else "unknown"
    )
    plan = dict(candidate) if candidate is not None else project_candidate_plan(
        operation=op,
        contract_id=contract_id or f"surface:{op}",
        finding_kind=finding_kind,
        finding_id=finding_id,
        domain=domain,
        goal_id=goal_id,
        program_id=program_id,
    )
    if path:
        plan["path"] = path
    if symbol:
        plan["symbol"] = symbol
    plan["consumer"] = consumer
    plan["domain"] = domain

    families_out: dict[str, Any] = {}
    errors: list[str] = []

    if "intent_ir" in policy.families:
        try:
            families_out["intent_ir"] = apply_intent_logic(plan)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"intent_ir: {type(exc).__name__}: {exc}")
            families_out["intent_ir"] = {
                "family": "intent_ir",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    if "legal_ir" in policy.families:
        try:
            families_out["legal_ir"] = apply_legal_logic(
                plan, modality=policy.legal_modality
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"legal_ir: {type(exc).__name__}: {exc}")
            families_out["legal_ir"] = {
                "family": "legal_ir",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    if "security_ir" in policy.families:
        try:
            families_out["security_ir"] = apply_security_logic(
                plan,
                decision=policy.security_decision,
                evaluate=policy.evaluate_security,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"security_ir: {type(exc).__name__}: {exc}")
            families_out["security_ir"] = {
                "family": "security_ir",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    if "ui_ir" in policy.families:
        try:
            families_out["ui_ir"] = apply_ui_logic(plan)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ui_ir: {type(exc).__name__}: {exc}")
            families_out["ui_ir"] = {
                "family": "ui_ir",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    # Structural IR: AST / knowledge graph / vector index
    structural_want = [f for f in policy.families if f in STRUCTURAL_IR_FAMILIES]
    if structural_want:
        try:
            from .ir_structural_application import apply_structural_logic

            structural = apply_structural_logic(
                operation=op,
                contract_id=str(plan.get("contract_id") or contract_id or ""),
                finding_kind=str(plan.get("finding_kind") or finding_kind or ""),
                path=str(plan.get("path") or ""),
                symbol=str(plan.get("symbol") or ""),
                candidate=plan,
                families=structural_want,
                domain=str(plan.get("domain") or domain or "agent_supervisor"),
            )
            for name, doc in (structural.get("families") or {}).items():
                families_out[name] = doc
            errors.extend(list(structural.get("errors") or []))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"structural_ir: {type(exc).__name__}: {exc}")
            for name in structural_want:
                families_out[name] = {
                    "family": name,
                    "applied": False,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "grants_execution_authority": False,
                }

    admission: dict[str, Any] = {"applied": False, "status": "skipped"}
    if policy.include_plan_admission and all(
        k in families_out for k in ("intent_ir", "legal_ir", "security_ir")
    ):
        admission = apply_plan_admission(
            plan,
            intent_result=families_out["intent_ir"],
            legal_result=families_out["legal_ir"],
            security_result=families_out["security_ir"],
        )

    family_ok = {
        name: bool(doc.get("ok"))
        for name, doc in families_out.items()
        if name in policy.families
    }
    no_false_grants = all(
        not bool(doc.get("grants_execution_authority"))
        for doc in families_out.values()
    ) and not bool(admission.get("grants_execution_authority"))

    passed = bool(family_ok) and all(family_ok.values()) and no_false_grants
    if errors and policy.fail_closed_on_unsupported:
        passed = False
    if not policy.fail_closed_on_unsupported and family_ok:
        # Partial: pass if required core shared+structural families applied
        core = [
            f
            for f in (
                "intent_ir",
                "legal_ir",
                "security_ir",
                "ast",
                "knowledge_graph",
                "vector_index",
            )
            if f in family_ok
        ]
        if core:
            passed = all(family_ok[f] for f in core) and no_false_grants

    # Strip non-serializable raw handles from public report
    public_families: dict[str, Any] = {}
    for name, doc in families_out.items():
        public = {k: v for k, v in doc.items() if not k.startswith("_raw")}
        public_families[name] = public

    return {
        "schema": IR_LOGIC_APPLY_SCHEMA,
        "interface": IR_LOGIC_APPLICATION_INTERFACE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "operation": op,
        "contract_id": plan.get("contract_id"),
        "finding_kind": plan.get("finding_kind"),
        "finding_id": plan.get("finding_id"),
        "candidate_plan": {
            "plan_id": plan.get("plan_id"),
            "action_ids": [a.get("action_id") for a in (plan.get("actions") or [])],
            "effect_ids": [e.get("effect_id") for e in (plan.get("effects") or [])],
        },
        "policy": policy.to_dict(),
        "families": public_families,
        "family_ok": family_ok,
        "plan_admission": {
            k: v for k, v in admission.items() if not k.startswith("_raw")
        },
        "gates": {
            "all_selected_families_ok": all(family_ok.values()) if family_ok else False,
            "no_false_execution_grants": no_false_grants,
            "logic_applied_to_ir": any(
                bool(d.get("applied")) for d in public_families.values()
            ),
        },
        "errors": errors,
        "notes": [
            "Logic was applied to intermediate representations via shared-IR adapters.",
            "Intent/Legal/Security compile on verified normalized IR nodes.",
            "UI IR is projected as interface descriptors (bridge-only).",
            "AST / knowledge_graph / vector_index are structural intermediate IR.",
            "No family grants execution authority from work surface application alone.",
        ],
        "consumers": ["planner", "doctor", "symbolic_repair", "contract_repair", "sca", "generic"],
    }


def apply_logic_to_surfaces(
    surfaces: Sequence[Mapping[str, Any] | IrWorkSurface],
    *,
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
    domain: str = "agent_supervisor",
    consumer: str = "generic",
) -> dict[str, Any]:
    """Batch-apply IR logic to arbitrary work surfaces (domain-agnostic)."""
    if isinstance(policy, Mapping):
        policy = IrLogicApplyPolicy.from_mapping(policy)
    elif policy is None:
        policy = IrLogicApplyPolicy()

    rows: list[dict[str, Any]] = []
    for item in list(surfaces)[: policy.max_surfaces]:
        if isinstance(item, IrWorkSurface):
            surface = item
            if domain and surface.domain == "agent_supervisor":
                surface = IrWorkSurface(
                    **{
                        **surface.to_dict(),
                        "domain": domain,
                        "consumer": consumer or surface.consumer,
                    }
                )
            rows.append(
                apply_logic_to_ir(
                    surface,
                    policy=policy,
                    consumer=consumer or surface.consumer,
                    domain=surface.domain or domain,
                )
            )
            continue
        raw = dict(item)
        if domain and not raw.get("domain"):
            raw["domain"] = domain
        if consumer and not raw.get("consumer"):
            raw["consumer"] = consumer
        surface = IrWorkSurface.from_mapping(raw)
        rows.append(
            apply_logic_to_ir(
                surface,
                policy=policy,
                consumer=surface.consumer or consumer,
                domain=surface.domain or domain,
            )
        )

    passed = bool(rows) and all(r.get("passed") for r in rows)
    return {
        "schema": IR_LOGIC_BATCH_SCHEMA,
        "interface": IR_LOGIC_APPLICATION_INTERFACE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "selected_count": len(rows),
        "domain": domain,
        "consumer": consumer,
        "policy": policy.to_dict(),
        "rows": rows,
        "summary": {
            "intent_ok": sum(
                1
                for r in rows
                if (r.get("family_ok") or {}).get("intent_ir")
            ),
            "legal_ok": sum(
                1 for r in rows if (r.get("family_ok") or {}).get("legal_ir")
            ),
            "security_ok": sum(
                1 for r in rows if (r.get("family_ok") or {}).get("security_ir")
            ),
            "ui_ok": sum(1 for r in rows if (r.get("family_ok") or {}).get("ui_ir")),
            "ast_ok": sum(1 for r in rows if (r.get("family_ok") or {}).get("ast")),
            "knowledge_graph_ok": sum(
                1
                for r in rows
                if (r.get("family_ok") or {}).get("knowledge_graph")
            ),
            "vector_index_ok": sum(
                1 for r in rows if (r.get("family_ok") or {}).get("vector_index")
            ),
        },
        "completion_authoritative": False,
        "notes": [
            "Batch IR logic application over domain-agnostic work surfaces.",
            "Each row materializes shared + structural intermediate IR.",
        ],
    }


# Backward-compatible alias (historical SCA naming)
apply_logic_to_findings = apply_logic_to_surfaces


def load_apply_policy_from_supervisor_profile(
    profile_path: str | None = None,
) -> IrLogicApplyPolicy:
    from pathlib import Path

    if profile_path is None:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[4]
            / "config"
            / "swissknife_symbolic_contract_assurance_supervisor.json",
            Path.cwd()
            / "config"
            / "swissknife_symbolic_contract_assurance_supervisor.json",
        ]
        for c in candidates:
            if c.is_file():
                profile_path = str(c)
                break
    if not profile_path or not Path(profile_path).is_file():
        return IrLogicApplyPolicy()
    doc = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    raw = (
        doc.get("irLogicApplyPolicy")
        or doc.get("ir_logic_apply_policy")
        or (doc.get("irIntegrationPolicy") or {}).get("logicApply")
        or (doc.get("symbolicRepairPolicy") or {}).get("irLogicApply")
        or {}
    )
    return IrLogicApplyPolicy.from_mapping(raw)


# ---------------------------------------------------------------------------
# DCR-035 required-stage gate (separate from advisory IR application paths)
# ---------------------------------------------------------------------------

IR_LOGIC_REQUIRED_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-required-gate@1"
)
IR_LOGIC_REQUIRED_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-required-stage-receipt@1"
)
REQUIRED_IR_LOGIC_STAGES: Final[tuple[str, ...]] = (
    "diagnose",
    "plan",
    "admit",
    "apply",
    "complete",
)
REQUIRED_IR_LOGIC_IDENTITIES: Final[tuple[str, ...]] = (
    "dcr030",
    "dcr031",
    "dcr032",
    "dcr033",
    "dcr034",
)


class IrLogicRequiredGateDisposition(str, Enum):
    """Closed outcomes; this gate never itself grants runtime authority."""

    PASSING = "passing"
    INTEGRATION_PENDING = "integration_pending"
    REJECTED = "rejected"


@dataclass(frozen=True)
class IrLogicRequiredStageReceipt:
    """One typed, non-authoritative receipt for a DCR-035 lifecycle stage."""

    stage: str
    identity_cids: Mapping[str, str]
    surface_cids: tuple[str, ...]
    outcome: str = "passed"
    bridge_only: bool = False
    default_true: bool = False
    swallowed_exception: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", str(self.stage or "").strip())
        object.__setattr__(
            self,
            "identity_cids",
            {
                str(key).strip(): str(value).strip()
                for key, value in dict(self.identity_cids or {}).items()
            },
        )
        object.__setattr__(
            self,
            "surface_cids",
            tuple(str(item).strip() for item in self.surface_cids if str(item).strip()),
        )
        object.__setattr__(self, "outcome", str(self.outcome or "").strip().lower())
        for name in ("bridge_only", "default_true", "swallowed_exception"):
            if type(getattr(self, name)) is not bool:
                raise IrLogicApplicationError(f"{name} must be boolean")
        for name in ("model_call_count", "provider_call_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise IrLogicApplicationError(f"{name} must be a non-negative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IrLogicRequiredStageReceipt":
        if not isinstance(value, Mapping):
            raise IrLogicApplicationError("stage receipt must be a mapping")
        return cls(
            stage=value.get("stage", ""),
            identity_cids=value.get("identity_cids", {}),
            surface_cids=tuple(value.get("surface_cids") or ()),
            outcome=value.get("outcome", ""),
            bridge_only=value.get("bridge_only", False),
            default_true=value.get("default_true", False),
            swallowed_exception=value.get("swallowed_exception", False),
            model_call_count=value.get("model_call_count", 0),
            provider_call_count=value.get("provider_call_count", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": IR_LOGIC_REQUIRED_RECEIPT_SCHEMA,
            "stage": self.stage,
            "identity_cids": dict(sorted(self.identity_cids.items())),
            "surface_cids": list(self.surface_cids),
            "outcome": self.outcome,
            "bridge_only": self.bridge_only,
            "default_true": self.default_true,
            "swallowed_exception": self.swallowed_exception,
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
        }
        payload["receipt_id"] = "ir-logic-stage:sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


@dataclass(frozen=True)
class IrLogicRequiredGateResult:
    """Typed result that can advance a review workflow but not execute it."""

    disposition: IrLogicRequiredGateDisposition
    reason_codes: tuple[str, ...]
    required_identity_cids: Mapping[str, str]
    receipt_ids: tuple[str, ...]
    model_call_count: int = 0
    provider_call_count: int = 0
    execution_authorized: bool = False
    completion_authorized: bool = False

    @property
    def passing(self) -> bool:
        return self.disposition is IrLogicRequiredGateDisposition.PASSING

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_LOGIC_REQUIRED_GATE_SCHEMA,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "required_identity_cids": dict(sorted(self.required_identity_cids.items())),
            "receipt_ids": list(self.receipt_ids),
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
            "execution_authorized": self.execution_authorized,
            "completion_authorized": self.completion_authorized,
        }


def evaluate_required_ir_logic_gate(
    stage_receipts: Sequence[IrLogicRequiredStageReceipt | Mapping[str, Any]],
    *,
    required_identity_cids: Mapping[str, str],
) -> IrLogicRequiredGateResult:
    """Fail closed on any absent, bridge-only, or non-passing stage receipt.

    The function is deliberately pure: it imports no provider, executes no
    model, and never turns a passing intermediate gate into execution or
    completion authority.
    """

    expected = {
        key: str(required_identity_cids.get(key, "") or "").strip()
        for key in REQUIRED_IR_LOGIC_IDENTITIES
    }
    reasons: list[str] = []
    missing_expected = tuple(key for key, value in expected.items() if not value)
    if missing_expected:
        reasons.extend(f"missing_required_identity_{key}" for key in missing_expected)

    normalized: list[IrLogicRequiredStageReceipt] = []
    for value in stage_receipts:
        try:
            receipt = (
                value
                if isinstance(value, IrLogicRequiredStageReceipt)
                else IrLogicRequiredStageReceipt.from_mapping(value)
            )
        except Exception:  # noqa: BLE001 - recorded as a failed gate, never swallowed
            reasons.append("stage_receipt_conversion_error")
            continue
        normalized.append(receipt)

    stages = tuple(item.stage for item in normalized)
    if stages != REQUIRED_IR_LOGIC_STAGES:
        missing = sorted(set(REQUIRED_IR_LOGIC_STAGES).difference(stages))
        extra = sorted(set(stages).difference(REQUIRED_IR_LOGIC_STAGES))
        reasons.extend(f"required_stage_missing_{stage}" for stage in missing)
        reasons.extend(f"unknown_or_duplicate_stage_{stage}" for stage in extra)
        if len(stages) != len(set(stages)) or not missing and not extra:
            reasons.append("required_stage_order_or_duplicate_invalid")

    receipt_ids: list[str] = []
    for receipt in normalized:
        receipt_payload = receipt.to_dict()
        receipt_ids.append(str(receipt_payload["receipt_id"]))
        if receipt.outcome != "passed":
            reasons.append(f"stage_{receipt.stage}_outcome_{receipt.outcome or 'missing'}")
        if not receipt.surface_cids:
            reasons.append(f"stage_{receipt.stage}_empty_surface")
        if receipt.bridge_only:
            reasons.append(f"stage_{receipt.stage}_bridge_only_ui")
        if receipt.default_true:
            reasons.append(f"stage_{receipt.stage}_default_true_flag")
        if receipt.swallowed_exception:
            reasons.append(f"stage_{receipt.stage}_swallowed_exception")
        if receipt.model_call_count or receipt.provider_call_count:
            reasons.append(f"stage_{receipt.stage}_nonzero_model_or_provider_calls")
        if dict(receipt.identity_cids) != expected:
            missing = tuple(
                key for key, value in expected.items() if receipt.identity_cids.get(key) != value
            )
            reasons.extend(
                f"stage_{receipt.stage}_identity_binding_mismatch_{key}" for key in missing
            )
            if set(receipt.identity_cids).difference(expected):
                reasons.append(f"stage_{receipt.stage}_unknown_identity_binding")

    pending = any(
        code.endswith("_dcr033") or code.endswith("_dcr034")
        for code in reasons
    )
    disposition = (
        IrLogicRequiredGateDisposition.PASSING
        if not reasons
        else (
            IrLogicRequiredGateDisposition.INTEGRATION_PENDING
            if pending
            else IrLogicRequiredGateDisposition.REJECTED
        )
    )
    return IrLogicRequiredGateResult(
        disposition=disposition,
        reason_codes=tuple(sorted(set(reasons))),
        required_identity_cids=expected,
        receipt_ids=tuple(receipt_ids),
    )


__all__ = [
    "DEFAULT_APPLY_FAMILIES",
    "SHARED_IR_FAMILIES",
    "STRUCTURAL_IR_FAMILIES",
    "IR_LOGIC_APPLICATION_INTERFACE",
    "IR_LOGIC_APPLY_SCHEMA",
    "IR_LOGIC_BATCH_SCHEMA",
    "IR_LOGIC_REQUIRED_GATE_SCHEMA",
    "IR_LOGIC_REQUIRED_RECEIPT_SCHEMA",
    "SCA_IR_LOGIC_APPLICATOR_INTERFACE",
    "SCA_IR_LOGIC_APPLY_SCHEMA",
    "SCA_IR_LOGIC_BATCH_SCHEMA",
    "IrLogicApplyPolicy",
    "IrLogicApplicationError",
    "IrLogicRequiredGateDisposition",
    "IrLogicRequiredGateResult",
    "IrLogicRequiredStageReceipt",
    "IrWorkSurface",
    "ScaIrLogicApplicatorError",
    "apply_intent_logic",
    "apply_legal_logic",
    "apply_logic_to_findings",
    "apply_logic_to_ir",
    "apply_logic_to_surfaces",
    "apply_plan_admission",
    "apply_security_logic",
    "apply_ui_logic",
    "evaluate_required_ir_logic_gate",
    "load_apply_policy_from_supervisor_profile",
    "project_candidate_plan",
    "REQUIRED_IR_LOGIC_IDENTITIES",
    "REQUIRED_IR_LOGIC_STAGES",
]
