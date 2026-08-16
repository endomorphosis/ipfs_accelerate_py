"""DCR-046 static security-operator previews; never execute or write target code.

Profile C and D obligations are currently draft/unsupported in the MCP++
profile registry.  A structurally complete preview is therefore useful review
data only and remains integration-pending until DCR-070/DCR-072 integration.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ...autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    NetworkMode,
)
from ...proof.formal_verification_contracts import canonical_json_bytes, content_identity
from ...proof.ir_logic_application import IrLogicRequiredGateResult
from ...proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)
from .registry import OperatorDescriptor, OperatorRegistry

SECURITY_REPAIR_PREVIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-046-security-repair-preview@1"
)
SECURITY_REPAIR_ACTIVATION: Final = "integration_pending_dcr070_dcr072"
_ACTIONS: Final[dict[str, tuple[str, McpObligationFamily, McpObligationFragment]]] = {
    "restore_authorization_binding": (
        "bind_authorization",
        McpObligationFamily.PROFILE_C,
        McpObligationFragment.DELEGATION,
    ),
    "annotate_effect": (
        "annotate_effect",
        McpObligationFamily.PROFILE_C,
        McpObligationFragment.DELEGATION,
    ),
    "gate_policy_before_effect": (
        "guard_effect",
        McpObligationFamily.PROFILE_D,
        McpObligationFragment.POLICY,
    ),
}
_CLOSED_EFFECT_ANNOTATIONS: Final[frozenset[str]] = frozenset(
    {"reviewed_effect_annotation", "policy_before_effect"}
)


class SecurityRepairStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PREVIEWED = "previewed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class SecurityRepairError(ValueError):
    """A security preview attempted to treat unreviewed data as authority."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SecurityRepairError(f"{field} must be non-empty canonical text")
    if any(character.isspace() for character in value) or "\x00" in value:
        raise SecurityRepairError(f"{field} must be an opaque identifier")
    return value


def _span(source: bytes, node: ast.AST) -> tuple[int, int]:
    if not all(hasattr(node, field) for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
        raise SecurityRepairError("security anchor has no exact source span")
    starts = [0, *(index + 1 for index, item in enumerate(source) if item == 10)]
    try:
        return (
            starts[node.lineno - 1] + node.col_offset,  # type: ignore[attr-defined]
            starts[node.end_lineno - 1] + node.end_col_offset,  # type: ignore[attr-defined]
        )
    except (AttributeError, IndexError) as exc:
        raise SecurityRepairError("security anchor span is outside source") from exc


def security_ast_span_identity(source: bytes, node: ast.AST) -> dict[str, object]:
    """Return exact byte evidence for a reviewed structural AST anchor."""

    start, end = _span(source, node)
    return {
        "node_type": type(node).__name__,
        "start": start,
        "end": end,
        "sha256": _sha256(source[start:end]),
    }


@dataclass(frozen=True)
class SecurityRepairPreview:
    """Reversible static bytes plus non-authorizing dependency bindings."""

    status: SecurityRepairStatus
    reason_codes: tuple[str, ...]
    request_cid: str = ""
    forward_cid: str = ""
    inverse_cid: str = ""
    before_digest: str = ""
    after_digest: str = ""
    after_bytes: bytes = b""

    def to_dict(self) -> dict[str, object]:
        body = {
            "schema": SECURITY_REPAIR_PREVIEW_SCHEMA,
            "authoritative": False,
            "activation_status": SECURITY_REPAIR_ACTIVATION,
            "execution_authorized": False,
            "completion_authorized": False,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "request_cid": self.request_cid,
            "forward_cid": self.forward_cid,
            "inverse_cid": self.inverse_cid,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "after_base64": base64.b64encode(self.after_bytes).decode("ascii"),
            "model_call_count": 0,
            "provider_call_count": 0,
        }
        return {**body, "preview_cid": content_identity(body)}


def _rejected(reason: str, *, status: SecurityRepairStatus = SecurityRepairStatus.REJECTED) -> SecurityRepairPreview:
    return SecurityRepairPreview(status=status, reason_codes=(reason,))


def _checked_capability(
    receipt: object,
    evidence: object,
) -> CapabilityReceipt:
    if not isinstance(receipt, CapabilityReceipt):
        raise SecurityRepairError("DCR-004 capability receipt must be typed")
    if (
        not receipt.capability_id.startswith("ipfs_datasets_py.logic.")
        or receipt.status is not CapabilityStatus.AVAILABLE
        or not receipt.available
        or not receipt.content_digest.startswith("module:sha256:")
        or receipt.distribution_version != receipt.expected_version
        or receipt.missing_symbols
        or receipt.reason_codes
        or not receipt.initialized
        or not receipt.reconstructed
        or not receipt.self_test_passed
        or receipt.network_mode is not NetworkMode.OFFLINE
    ):
        raise SecurityRepairError("DCR-004 security/deontic capability is not current offline evidence")
    if receipt.to_dict().get("receipt_id") != receipt.receipt_id:
        raise SecurityRepairError("DCR-004 capability receipt identity does not recompute")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise SecurityRepairError("DCR-004 evidence receipts must be an exact sequence")
    evidence_by_kind: dict[str, CapabilityEvidenceReceipt] = {}
    for item in evidence:
        if not isinstance(item, CapabilityEvidenceReceipt):
            raise SecurityRepairError("DCR-004 evidence receipt must be typed")
        # ``receipt_id`` is recomputed by the typed record; ``verifies`` also
        # binds its kind, source digest and source version exactly.
        if item.to_dict().get("receipt_id") != item.receipt_id:
            raise SecurityRepairError("DCR-004 evidence receipt identity does not recompute")
        if item.evidence_kind in evidence_by_kind:
            raise SecurityRepairError("DCR-004 evidence kinds must be unique")
        evidence_by_kind[item.evidence_kind] = item
    if set(evidence_by_kind) != {"initialization", "reconstruction", "self_test"}:
        raise SecurityRepairError("DCR-004 initialization/reconstruction/self-test receipts required")
    for kind, item in evidence_by_kind.items():
        if not item.verifies(
            evidence_id=receipt.capability_id,
            evidence_kind=kind,
            subject_id=receipt.capability_id,
            subject_digest=receipt.content_digest,
            subject_version=receipt.distribution_version,
        ):
            raise SecurityRepairError("DCR-004 evidence receipt is stale, unverified, or networked")
    return receipt


def _checked_obligation(
    value: object, action: str, *, authorization: str, policy: str, annotation: str
) -> McpGraphContractObligation:
    if not isinstance(value, McpGraphContractObligation):
        raise SecurityRepairError("DCR-031 obligation must be typed")
    _target, family, fragment = _ACTIONS[action]
    if (
        value.family is not family
        or value.fragment is not fragment
        or value.backend is not McpObligationBackend.LOGIC_IR_CANDIDATE
        or value.disposition is not McpObligationDisposition.OPEN
        or not value.graph_cid
        or not value.candidate_cid
        or not value.input_cids
        or not value.schema_bindings
    ):
        raise SecurityRepairError("DCR-031 obligation is draft, unsupported, or unbound")
    if authorization not in value.cid_bindings or policy not in value.cid_bindings:
        raise SecurityRepairError("authorization or policy binding is not in the DCR-031 obligation")
    expected_effect_semantics = (
        "policy_before_effect"
        if family is McpObligationFamily.PROFILE_D
        else "delegation_bound_authorization"
    )
    if value.effect_semantics != expected_effect_semantics:
        raise SecurityRepairError("DCR-031 obligation effect semantics are not reviewed and exact")
    if annotation not in _CLOSED_EFFECT_ANNOTATIONS:
        raise SecurityRepairError("effect annotation is not in the closed reviewed vocabulary")
    if action == "gate_policy_before_effect" and annotation != "policy_before_effect":
        raise SecurityRepairError("policy gate must be policy-before-effect")
    if action != "gate_policy_before_effect" and annotation != "reviewed_effect_annotation":
        raise SecurityRepairError("delegation preview requires the reviewed effect annotation")
    return value


def _checked_gate(
    value: object, obligation: McpGraphContractObligation
) -> IrLogicRequiredGateResult:
    if not isinstance(value, IrLogicRequiredGateResult):
        raise SecurityRepairError("DCR-035 gate must be typed")
    if not value.passing or value.reason_codes or value.model_call_count or value.provider_call_count:
        raise SecurityRepairError("DCR-035 gate is not exact passing zero-call evidence")
    expected = {"dcr030", "dcr031", "dcr032", "dcr033", "dcr034"}
    if set(value.required_identity_cids) != expected or any(
        not item for item in value.required_identity_cids.values()
    ):
        raise SecurityRepairError("DCR-035 gate is missing required identity bindings")
    if value.required_identity_cids["dcr031"] != content_identity(obligation.to_dict()):
        raise SecurityRepairError("DCR-035 gate is stale for this DCR-031 obligation")
    if value.required_identity_cids["dcr030"] != obligation.candidate_cid:
        raise SecurityRepairError("DCR-035 gate is stale for this DCR-030 candidate")
    return value


def _replacement(action: str, *, effect_id: str, authorization: str, policy: str, annotation: str) -> bytes:
    if action == "restore_authorization_binding":
        return f"bind_authorization(effect_id={effect_id!r}, authorization={authorization!r})".encode()
    if action == "annotate_effect":
        return f"annotate_effect(effect_id={effect_id!r}, annotation={annotation!r})".encode()
    return f"guard_effect(effect_id={effect_id!r}, policy={policy!r})".encode()


def build_security_repair_preview(
    request: Mapping[str, Any], *, registry: OperatorRegistry
) -> SecurityRepairPreview:
    """Create a closed structural preview with no execution, write, or provider route."""

    try:
        required = {
            "action", "operator_id", "descriptor_id", "registry_cid", "owner_root",
            "relative_path", "source_bytes", "source_digest", "anchor", "effect_id",
            "authorization_binding", "policy_binding", "effect_annotation", "obligation",
            "capability_receipt", "capability_evidence_receipts", "dcr035_gate",
        }
        if not isinstance(request, Mapping) or set(request) != required:
            raise SecurityRepairError("security preview request fields are closed")
        source = request["source_bytes"]
        if not isinstance(source, bytes) or not source or request["source_digest"] != _sha256(source):
            raise SecurityRepairError("exact source bytes and digest are required")
        action = _text(request["action"], "action")
        if action not in _ACTIONS:
            raise SecurityRepairError("security action is not closed")
        effect_id = _text(request["effect_id"], "effect_id")
        authorization = _text(request["authorization_binding"], "authorization_binding")
        policy = _text(request["policy_binding"], "policy_binding")
        annotation = _text(request["effect_annotation"], "effect_annotation")
        report = registry.report()
        if request["registry_cid"] != report["registry_cid"]:
            raise SecurityRepairError("DCR-040 registry CID is stale")
        descriptors = {item.operator_id: item for item in registry.enumerate()}
        descriptor = descriptors.get(_text(request["operator_id"], "operator_id"))
        if (
            not isinstance(descriptor, OperatorDescriptor)
            or request["descriptor_id"] != descriptor.descriptor_id
            or request["owner_root"] != descriptor.owner_root
            or request["relative_path"] not in descriptor.write_scope
        ):
            raise SecurityRepairError("DCR-040 descriptor membership, owner, or scope is invalid")
        obligation = _checked_obligation(
            request["obligation"], action, authorization=authorization, policy=policy, annotation=annotation
        )
        capability = _checked_capability(
            request["capability_receipt"], request["capability_evidence_receipts"]
        )
        gate = _checked_gate(request["dcr035_gate"], obligation)
        target_name, _family, _fragment = _ACTIONS[action]
        tree = ast.parse(source, filename=str(request["relative_path"]))
        anchors = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == target_name
        ]
        if len(anchors) != 1:
            return _rejected("dynamic_or_ambiguous_security_anchor")
        anchor = anchors[0]
        if not isinstance(request["anchor"], Mapping) or dict(request["anchor"]) != security_ast_span_identity(source, anchor):
            raise SecurityRepairError("security AST anchor is stale or not exact")
        if any(keyword.arg is None or not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, str) for keyword in anchor.keywords):
            return _rejected("dynamic_security_anchor_arguments")
        if action == "gate_policy_before_effect":
            anchor_start, _anchor_end = _span(source, anchor)
            effect_before_gate = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "perform_effect"
                and _span(source, node)[0] < anchor_start
                for node in ast.walk(tree)
            )
            if effect_before_gate:
                return _rejected("post_effect_policy_check")
        replacement = _replacement(
            action, effect_id=effect_id, authorization=authorization, policy=policy, annotation=annotation
        )
        start, end = _span(source, anchor)
        after = source[:start] + replacement + source[end:]
        if after == source:
            return _rejected("preview_requires_effectful_closed_normalization")
        body = {
            key: value for key, value in request.items()
            if key not in {"source_bytes", "obligation", "capability_receipt", "capability_evidence_receipts", "dcr035_gate"}
        }
        request_cid = content_identity(
            {
                **body,
                "source_base64": base64.b64encode(source).decode("ascii"),
                "obligation": obligation.to_dict(),
                "capability_receipt_id": capability.receipt_id,
                "capability_evidence_receipt_ids": sorted(
                    item.receipt_id for item in request["capability_evidence_receipts"]
                ),
                "dcr035_gate": gate.to_dict(),
            }
        )
        return SecurityRepairPreview(
            status=SecurityRepairStatus.PREVIEWED,
            reason_codes=("profile_c_d_runtime_integration_pending",),
            request_cid=request_cid,
            forward_cid=content_identity({"after": _sha256(after), "action": action}),
            inverse_cid=content_identity({"before": _sha256(source), "action": action}),
            before_digest=_sha256(source),
            after_digest=_sha256(after),
            after_bytes=after,
        )
    except (SecurityRepairError, SyntaxError, TypeError, ValueError) as exc:
        return _rejected(str(exc))


def canonical_security_preview_bytes(preview: SecurityRepairPreview) -> bytes:
    if not isinstance(preview, SecurityRepairPreview):
        raise SecurityRepairError("preview must be typed")
    return canonical_json_bytes(preview.to_dict())


__all__ = [
    "SECURITY_REPAIR_ACTIVATION",
    "SECURITY_REPAIR_PREVIEW_SCHEMA",
    "SecurityRepairError",
    "SecurityRepairPreview",
    "SecurityRepairStatus",
    "build_security_repair_preview",
    "canonical_security_preview_bytes",
    "security_ast_span_identity",
]
