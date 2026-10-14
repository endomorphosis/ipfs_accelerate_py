"""Owner-backed materials for the existing hard planning-admission gates.

Pass :class:`OwnerBackedPlanningAdmissionBuilder` as ``ir_request_builder`` to
``PlanningAnalysisFactory``. A deployment must supply both a material resolver
and an independent owner verifier. There is deliberately no default provider,
fixture, trust policy, grant, proof discharge, or admission decision here.

The verifier is a trust boundary supplied by the deployment: it must actually
authenticate owner provenance, artifact custody, current source roots, review
scope and authority grants. Hashes, enum labels and this module's typed records
cannot establish those facts. The returned request still goes through the
normal prompt and IR checkers. Construction is never Lean admission.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .planning_analysis_factory import PlanningAnalysisAdmissionError
from ..planning.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
    PlanCompilationResult,
    prompt_goal_graph_to_formal_input,
)
from ..planning.plan_admission_service import (
    PlanAdmissionService,
    PlanAdmissionServiceReceipt,
    PlanAdmissionServiceRequest,
)
from ..proof.formal_verification_contracts import content_identity
from ..prompt.prompt_plan_admission import (
    PromptPlanAdmissionPolicy,
    PromptPlanAdmissionRequest,
    _ir_binding_findings,
)
from ..prompt.prompt_workflow import (
    DirectoryScanReceipt,
    PromptGoalGraph,
    PromptWorkflowRequest,
)


MAX_MATERIAL_BYTES = 4 * 1024 * 1024
MAX_REFERENCES = 1024
MAX_ISSUES = 256


def _text(value: Any, label: str, *, maximum: int = 8192) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or "\x00" in value or len(value.encode("utf-8")) > maximum):
        raise ValueError(f"{label} must be bounded nonempty text")
    return value


def _ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            or not values or len(values) > MAX_REFERENCES):
        raise ValueError(f"{label} must be a bounded nonempty sequence")
    result = tuple(_text(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return tuple(sorted(result))


def _bounded(value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_MATERIAL_BYTES:
        raise ValueError("owner admission materials exceed the byte bound")


@dataclass(frozen=True)
class AdmissionMaterialIssue:
    """Bounded blocker; does not contain artifact bodies or grant authority."""

    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        for name, maximum in (("code", 128), ("path", 1024), ("message", 1024)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum=maximum))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class AdmissionMaterialsError(PlanningAnalysisAdmissionError):
    """Preserve detailed missing-material reasons through the existing hook."""

    def __init__(self, issues: Sequence[AdmissionMaterialIssue], *,
                 service_receipt: PlanAdmissionServiceReceipt | None = None) -> None:
        if (isinstance(issues, (str, bytes)) or not issues
                or len(issues) > MAX_ISSUES
                or any(not isinstance(issue, AdmissionMaterialIssue) for issue in issues)):
            raise ValueError("bounded typed admission material issues are required")
        self.issues = tuple(issues)
        if service_receipt is not None and not isinstance(service_receipt, PlanAdmissionServiceReceipt):
            raise TypeError("service_receipt must be a native PlanAdmissionServiceReceipt")
        self.service_receipt = service_receipt
        super().__init__(
            "; ".join(f"{issue.code} at {issue.path}: {issue.message}" for issue in self.issues),
            reason_code="owner_admission_materials_unavailable",
        )


def _fail(code: str, path: str, message: str) -> None:
    raise AdmissionMaterialsError((AdmissionMaterialIssue(code, path, message),))


@dataclass(frozen=True)
class PlanningAdmissionRunBindings:
    """Independently selected execution/policy identities, never an allow grant."""

    execution_plan_id: str
    policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_plan_id", _text(self.execution_plan_id, "execution_plan_id"))
        object.__setattr__(self, "policy_ids", _ids(self.policy_ids, "policy_ids"))


@dataclass(frozen=True)
class PlanningAdmissionMaterialContext:
    """Exact source/plan context against which an owner verifies materials."""

    workflow: PromptWorkflowRequest
    scan: DirectoryScanReceipt
    graph: PromptGoalGraph
    formal: PlanCompilationResult
    repository_tree_id: str
    evidence_bundle_id: str
    run_bindings: PlanningAdmissionRunBindings
    policy: PromptPlanAdmissionPolicy

    def identity_payload(self) -> dict[str, Any]:
        projection = self.formal.admission_projection
        assert projection is not None
        return {
            "schema": "planning-owner-material-context@1",
            "workflow_id": self.workflow.request_cid,
            "scan_id": self.scan.scan_cid,
            "graph_id": self.graph.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "formal_plan_id": self.formal.plan_id,
            "formal_source_identity": self.formal.source_identity,
            "formal_projection_id": projection.projection_id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "execution_plan_id": self.run_bindings.execution_plan_id,
            "policy_ids": list(self.run_bindings.policy_ids),
            "prompt_policy_id": self.policy.policy_id,
            "workflow_roots": {
                "intent": self.workflow.intent_ir_root,
                "legal": self.workflow.legal_ir_root,
                "security": self.workflow.security_ir_root,
                "policy": self.workflow.policy_root,
                "program": self.repository_tree_id,
            },
        }

    @property
    def context_id(self) -> str:
        return content_identity(self.identity_payload())


@dataclass(frozen=True)
class OwnerAdmissionMaterialEnvelope:
    """Resolver output. Provenance references are claims until verified outside."""

    materials: PlanAdmissionServiceRequest
    owner_id: str
    provenance_reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.materials, PlanAdmissionServiceRequest):
            raise TypeError("materials must be PlanAdmissionServiceRequest")
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "provenance_reference_ids",
                           _ids(self.provenance_reference_ids, "provenance_reference_ids"))
        _bounded(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": "planning-owner-material-envelope@1",
            "owner_id": self.owner_id,
            "provenance_reference_ids": list(self.provenance_reference_ids),
            "materials": self.materials.to_dict(),
        }

    @property
    def envelope_id(self) -> str:
        return content_identity(self.identity_payload())


@dataclass(frozen=True)
class OwnerMaterialVerification:
    """Externally produced provenance decision, not an execution/proof receipt."""

    accepted: bool
    context_id: str
    envelope_id: str
    owner_id: str
    verifier_id: str
    verified_reference_ids: tuple[str, ...]
    issues: tuple[AdmissionMaterialIssue, ...] = ()

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be boolean")
        for name in ("context_id", "envelope_id", "owner_id", "verifier_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "verified_reference_ids",
                           _ids(self.verified_reference_ids, "verified_reference_ids"))
        if len(self.issues) > MAX_ISSUES or any(
                not isinstance(issue, AdmissionMaterialIssue) for issue in self.issues):
            raise TypeError("issues must contain bounded typed material issues")
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.accepted and self.issues:
            raise ValueError("accepted owner verification cannot have blockers")


class OwnerAdmissionMaterialVerifier(Protocol):
    """Deployments implement this using real independent owner/trust custody.

    Authenticate every envelope reference and its source/review/grant scope;
    verify currentness against the complete context. Neither self-asserted
    ``reviewed`` labels nor content hashes alone suffice. A refusal must not
    substitute new declarations, grant sources or discharged obligations.
    """

    def verify_owner_materials(
        self, context: PlanningAdmissionMaterialContext,
        envelope: OwnerAdmissionMaterialEnvelope,
    ) -> OwnerMaterialVerification: ...


@dataclass(frozen=True)
class PlanningAdmissionBuiltRequest:
    """Full source-bound native planning receipt; not Lean/execution admission.

    Keeping the result on the call avoids shared mutable diagnostic state when
    several planning lanes use the same builder concurrently.
    """

    request: PromptPlanAdmissionRequest
    context_id: str
    envelope_id: str
    owner_verification: OwnerMaterialVerification
    service_receipt: PlanAdmissionServiceReceipt


class OwnerBackedPlanningAdmissionBuilder:
    """Callable adapter for ``PlanningAnalysisFactory.ir_request_builder``.

    No owner materials are inferred from ``analysis_view``: its optional
    advisory analysis cannot authorize the candidate that it helped produce.
    This adapter performs binding checks only; normal native gates retain
    all authority over proof, semantics, policy and admission.
    """

    def __init__(
        self, *,
        material_resolver: Callable[[PlanningAdmissionMaterialContext], OwnerAdmissionMaterialEnvelope | None],
        owner_verifier: OwnerAdmissionMaterialVerifier,
        run_bindings: PlanningAdmissionRunBindings,
        policy: PromptPlanAdmissionPolicy | None = None,
    ) -> None:
        if not callable(material_resolver):
            raise TypeError("an explicit material_resolver is required")
        if not callable(getattr(owner_verifier, "verify_owner_materials", None)):
            raise TypeError("an explicit independent owner_verifier is required")
        if not isinstance(run_bindings, PlanningAdmissionRunBindings):
            raise TypeError("run_bindings must be PlanningAdmissionRunBindings")
        if policy is not None and not isinstance(policy, PromptPlanAdmissionPolicy):
            raise TypeError("policy must be PromptPlanAdmissionPolicy")
        self._resolver = material_resolver
        self._verifier = owner_verifier
        self._run = run_bindings
        self._policy = policy or PromptPlanAdmissionPolicy()

    def __call__(
        self, *, request: Any, scan: Any, graph: Any,
        repository_tree_id: str, analysis_view: Any = None,
    ) -> PromptPlanAdmissionRequest:
        return self.build_with_receipt(
            request=request, scan=scan, graph=graph,
            repository_tree_id=repository_tree_id, analysis_view=analysis_view,
        ).request

    def build_with_receipt(
        self, *, request: Any, scan: Any, graph: Any,
        repository_tree_id: str, analysis_view: Any = None,
    ) -> PlanningAdmissionBuiltRequest:
        if not isinstance(request, PromptWorkflowRequest):
            _fail("missing_workflow", "$.workflow", "current native workflow request is required")
        if not isinstance(scan, DirectoryScanReceipt):
            _fail("missing_scan", "$.scan", "current native directory scan is required")
        if not isinstance(graph, PromptGoalGraph):
            _fail("missing_graph", "$.graph", "exact native candidate graph is required")
        if (graph.request_cid != request.request_cid or scan.request_cid != request.request_cid
                or graph.scan_cid != scan.scan_cid or graph.program_root != request.program_root):
            _fail("source_binding_mismatch", "$.graph", "workflow, scan and graph source identities differ")
        current_tree = scan.dirty_worktree_root or scan.repository_root_cid
        if repository_tree_id != current_tree:
            _fail("stale_repository_tree", "$.repository_tree_id", "tree differs from the current native scan")
        if (scan.repository_root_cid != request.repository_root_cid
                or scan.repository_root != request.repository_root
                or scan.directory != request.directory
                or scan.program_root != request.program_root
                or scan.scanner_policy_cid != request.scan_policy.content_id):
            _fail("stale_scan", "$.scan", "repository or scanner policy differs from the workflow")
        required_policies = {self._policy.policy_id, request.policy_root, *graph.policy_roots}
        if not required_policies.issubset(self._run.policy_ids):
            _fail("missing_policy_binding", "$.policy_ids", "execution bindings omit current workflow or admission policies")
        try:
            source = prompt_goal_graph_to_formal_input(graph, repository_tree_id=repository_tree_id)
            formal = FormalPlanCompiler().compile(source)
        except (TypeError, ValueError) as exc:
            _fail("formal_construction_failed", "$.formal", f"native formal construction failed: {type(exc).__name__}")
        if formal.status is not CompilationStatus.COMPILED or formal.admission_projection is None:
            _fail("formal_compilation_failed", "$.formal", "native compiler did not produce an exact admission projection")
        # This aggregate binds the current observations, without upgrading
        # their review status, source authority, or any semantic assertion.
        evidence_id = content_identity({
            "namespace": "planning-owner-material-evidence-bundle@1",
            "evidence": [item.to_dict() for item in graph.evidence],
        })
        context = PlanningAdmissionMaterialContext(
            request, scan, graph, formal, repository_tree_id, evidence_id,
            self._run, self._policy,
        )
        try:
            envelope = self._resolver(context)
        except AdmissionMaterialsError:
            raise
        except Exception as exc:
            _fail("owner_resolver_failed", "$.materials", f"independent resolver failed: {type(exc).__name__}")
        if envelope is None:
            raise AdmissionMaterialsError(tuple(
                AdmissionMaterialIssue("missing_owner_material", f"$.materials.{name}",
                                       "independently owned current material is unavailable")
                for name in ("intent_request", "legal_results", "security_policy",
                             "authority", "root_bindings", "proof_results", "validation_results")
            ))
        if not isinstance(envelope, OwnerAdmissionMaterialEnvelope):
            _fail("invalid_owner_material", "$.materials", "resolver must return a typed owner material envelope")
        context_id, envelope_id = context.context_id, envelope.envelope_id
        try:
            verification = self._verifier.verify_owner_materials(context, envelope)
        except AdmissionMaterialsError:
            raise
        except Exception as exc:
            _fail("owner_verification_failed", "$.owner_verification", f"owner verifier failed: {type(exc).__name__}")
        if context.context_id != context_id or envelope.envelope_id != envelope_id:
            _fail("materials_changed_during_verification", "$.materials", "source or material bytes changed during owner verification")
        if not isinstance(verification, OwnerMaterialVerification):
            _fail("invalid_owner_verification", "$.owner_verification", "verifier must return an exact typed provenance decision")
        if (verification.context_id != context_id or verification.envelope_id != envelope_id
                or verification.owner_id != envelope.owner_id
                or verification.verified_reference_ids != envelope.provenance_reference_ids):
            _fail("owner_verification_replay", "$.owner_verification", "owner verification does not cover this exact context and material envelope")
        if not verification.accepted:
            raise AdmissionMaterialsError(verification.issues or (
                AdmissionMaterialIssue("owner_verification_rejected", "$.owner_verification",
                                       "independent owner provenance verification refused these materials"),
            ))
        materials = envelope.materials
        expected = {
            "repository_tree_id": repository_tree_id,
            "formal_plan_id": formal.plan_id,
            "formal_source_identity": formal.source_identity,
            "evidence_bundle_id": evidence_id,
            "execution_plan_id": self._run.execution_plan_id,
            "policy_ids": self._run.policy_ids,
        }
        issues = [AdmissionMaterialIssue(
            "material_binding_mismatch", f"$.materials.{name}",
            "owner material differs from the current independently bound context",
        ) for name, value in expected.items() if getattr(materials, name) != value]
        for name in ("intent_effects", "code_effects"):
            if formal.admission_projection.effects and not getattr(materials, name):
                issues.append(AdmissionMaterialIssue(
                    "missing_independent_effect_stream", f"$.materials.{name}",
                    "independent owner must supply the effect stream; candidate fallback is insufficient",
                ))
        for index, binding in enumerate(materials.root_bindings):
            if not binding.current or not binding.authority_accepted:
                issues.append(AdmissionMaterialIssue(
                    "stale_or_untrusted_root", f"$.materials.root_bindings[{index}]",
                    "root currentness or source authority is not accepted",
                ))
        if issues:
            raise AdmissionMaterialsError(issues[:MAX_ISSUES])
        try:
            service = PlanAdmissionService()
            ir_request = service.construct_request(materials)
            # Reuse the native exact join checker, including proof and
            # validation coverage. Never reconstruct or auto-discharge them.
            findings = _ir_binding_findings(graph, formal, ir_request, self._policy, request)
        except (TypeError, ValueError) as exc:
            _fail("native_material_construction_failed", "$.materials", f"native material construction failed: {type(exc).__name__}")
        if findings:
            raise AdmissionMaterialsError(tuple(
                AdmissionMaterialIssue(str(item.code), item.path, item.message)
                for item in findings[:MAX_ISSUES]
            ))
        try:
            service_receipt = service.admit(materials)
        except (TypeError, ValueError) as exc:
            _fail("native_service_failed", "$.materials", f"native service rejected malformed materials: {type(exc).__name__}")
        if not service_receipt.admitted:
            issues = tuple(
                AdmissionMaterialIssue(str(code), f"$.service.{stage.stage.value}",
                                       stage.message or "native admission stage refused these materials")
                for stage in service_receipt.stage_results if not stage.passed
                for code in stage.reason_codes
            )
            raise AdmissionMaterialsError(issues[:MAX_ISSUES] or (
                AdmissionMaterialIssue("native_service_rejected", "$.service",
                                       "native admission service refused these materials"),
            ), service_receipt=service_receipt)
        if (service_receipt.request_id != materials.request_id
                or service_receipt.ir_request_id != ir_request.request_id
                or context.context_id != context_id or envelope.envelope_id != envelope_id):
            _fail("native_service_binding_mismatch", "$.service", "native receipt or materials differ from the verified source-bound request")
        prompt_request = PromptPlanAdmissionRequest(
            graph=graph, repository_tree_id=repository_tree_id, ir_request=ir_request,
            workflow_request=request, scan_receipt=scan, policy=self._policy,
        )
        return PlanningAdmissionBuiltRequest(
            prompt_request, context_id, envelope_id, verification, service_receipt,
        )


__all__ = [
    "AdmissionMaterialIssue", "AdmissionMaterialsError",
    "PlanningAdmissionRunBindings", "PlanningAdmissionMaterialContext",
    "OwnerAdmissionMaterialEnvelope", "OwnerMaterialVerification",
    "OwnerAdmissionMaterialVerifier", "OwnerBackedPlanningAdmissionBuilder",
    "PlanningAdmissionBuiltRequest",
]
