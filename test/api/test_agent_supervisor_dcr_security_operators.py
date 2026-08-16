"""DCR-046 security preview tests; all inputs are typed local fixtures."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.security_repairs import (
    SECURITY_REPAIR_ACTIVATION,
    SecurityRepairStatus,
    build_security_repair_preview,
    canonical_security_preview_bytes,
    security_ast_span_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    IrLogicRequiredStageReceipt,
    evaluate_required_ir_logic_gate,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "security.bind.authorization",
            "kind": "replace_exact_bytes",
            "input_schema": {
                "type": "object",
                "required": ["anchor", "relative_path", "source_digest"],
                "properties": {
                    "anchor": "cid",
                    "relative_path": "path",
                    "source_digest": "sha256",
                },
                "additional_properties": False,
            },
            "owner_root": "swissknife",
            "write_scope": ["security.py"],
            "before_predicates": ["unique_anchor"],
            "after_predicates": ["policy_before_effect"],
            "applicability_proofs": ["static_ast"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["pytest", "security.py"]],
        }
    )
    return OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _capability() -> tuple[CapabilityReceipt, tuple[CapabilityEvidenceReceipt, ...]]:
    receipt = CapabilityReceipt(
        capability_id="ipfs_datasets_py.logic.security_deontic",
        status=CapabilityStatus.AVAILABLE,
        origin="/reviewed/security_deontic.py",
        distribution="ipfs-datasets-py",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest="module:sha256:" + "a" * 64,
        symbols=("INITIALIZED", "RECONSTRUCTION_READY", "policy_before_effect"),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=receipt.capability_id,
            evidence_kind=kind,
            subject_id=receipt.capability_id,
            subject_digest=receipt.content_digest,
            subject_version=receipt.distribution_version,
            transcript_digest="transcript:sha256:" + character * 64,
            passed=True,
            network_mode=NetworkMode.OFFLINE,
        )
        for kind, character in (("initialization", "b"), ("reconstruction", "c"), ("self_test", "d"))
    )
    return receipt, evidence


def _gate(obligation: McpGraphContractObligation):
    identities = {
        "dcr030": obligation.candidate_cid,
        "dcr031": content_identity(obligation.to_dict()),
        "dcr032": "cid:dcr032",
        "dcr033": "cid:dcr033",
        "dcr034": "cid:dcr034",
    }
    return evaluate_required_ir_logic_gate(
        tuple(
            IrLogicRequiredStageReceipt(
                stage=stage,
                identity_cids=identities,
                surface_cids=(f"cid:{stage}",),
            )
            for stage in ("diagnose", "plan", "admit", "apply", "complete")
        ),
        required_identity_cids=identities,
    )


def _obligation(action: str = "restore_authorization_binding") -> McpGraphContractObligation:
    family = McpObligationFamily.PROFILE_D if action == "gate_policy_before_effect" else McpObligationFamily.PROFILE_C
    fragment = McpObligationFragment.POLICY if family is McpObligationFamily.PROFILE_D else McpObligationFragment.DELEGATION
    return McpGraphContractObligation(
        obligation_id="dcr031-security-obligation",
        family=family,
        fragment=fragment,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid="cid:graph",
        candidate_cid="cid:candidate",
        input_cids=("cid:input",),
        cid_bindings=("cid:delegation", "cid:policy"),
        schema_bindings=("cid:security-schema",),
        effect_semantics=(
            "policy_before_effect"
            if family is McpObligationFamily.PROFILE_D
            else "delegation_bound_authorization"
        ),
    )


def _request(action: str = "restore_authorization_binding") -> tuple[dict[str, object], OperatorRegistry]:
    registry = _registry()
    target = {
        "restore_authorization_binding": "bind_authorization",
        "annotate_effect": "annotate_effect",
        "gate_policy_before_effect": "guard_effect",
    }[action]
    source = (
        f'{target}(effect_id="effect.send", '
        f'{"authorization" if action == "restore_authorization_binding" else "annotation" if action == "annotate_effect" else "policy"}="cid:old")\n'
    ).encode()
    call = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))
    capability, evidence = _capability()
    obligation = _obligation(action)
    descriptor = registry.enumerate()[0]
    return {
        "action": action,
        "operator_id": descriptor.operator_id,
        "descriptor_id": descriptor.descriptor_id,
        "registry_cid": registry.report()["registry_cid"],
        "owner_root": "swissknife",
        "relative_path": "security.py",
        "source_bytes": source,
        "source_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
        "anchor": security_ast_span_identity(source, call),
        "effect_id": "effect.send",
        "authorization_binding": "cid:delegation",
        "policy_binding": "cid:policy",
        "effect_annotation": (
            "policy_before_effect" if action == "gate_policy_before_effect" else "reviewed_effect_annotation"
        ),
        "obligation": obligation,
        "capability_receipt": capability,
        "capability_evidence_receipts": evidence,
        "dcr035_gate": _gate(obligation),
    }, registry


def test_typed_profile_c_preview_is_structural_reversible_and_still_pending() -> None:
    request, registry = _request()
    preview = build_security_repair_preview(request, registry=registry)

    assert preview.status is SecurityRepairStatus.PREVIEWED
    assert b"cid:delegation" in preview.after_bytes
    assert preview.forward_cid and preview.inverse_cid
    assert preview.to_dict()["activation_status"] == SECURITY_REPAIR_ACTIVATION
    assert preview.to_dict()["execution_authorized"] is False
    assert canonical_security_preview_bytes(preview) == canonical_security_preview_bytes(preview)


def test_draft_or_mapping_obligation_and_stale_capability_evidence_are_rejected() -> None:
    request, registry = _request()
    request["obligation"] = {"server_assertion": True}
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED

    request, registry = _request()
    stale = list(request["capability_evidence_receipts"])
    stale[0] = CapabilityEvidenceReceipt(
        evidence_id="ipfs_datasets_py.logic.security_deontic",
        evidence_kind="initialization",
        subject_id="ipfs_datasets_py.logic.security_deontic",
        subject_digest="module:sha256:" + "e" * 64,
        subject_version="1.0",
        transcript_digest="transcript:sha256:" + "f" * 64,
        passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    request["capability_evidence_receipts"] = tuple(stale)
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED

    request, registry = _request()
    receipt = request["capability_receipt"]
    assert isinstance(receipt, CapabilityReceipt)
    request["capability_receipt"] = replace(receipt, distribution_version="9.9")
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED


@pytest.mark.parametrize("action", ("annotate_effect", "gate_policy_before_effect"))
def test_closed_effect_and_policy_structural_previews_are_non_authorizing(action: str) -> None:
    request, registry = _request(action)
    preview = build_security_repair_preview(request, registry=registry)
    assert preview.status is SecurityRepairStatus.PREVIEWED
    assert preview.to_dict()["provider_call_count"] == preview.to_dict()["model_call_count"] == 0


def test_dynamic_anchor_post_effect_shape_and_nonpassing_gate_are_rejected() -> None:
    request, registry = _request("gate_policy_before_effect")
    source = b'guard_effect(effect_id=dynamic_effect(), policy="cid:old")\n'
    call = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))
    request["source_bytes"] = source
    request["source_digest"] = "sha256:" + hashlib.sha256(source).hexdigest()
    request["anchor"] = security_ast_span_identity(source, call)
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED

    request, registry = _request("gate_policy_before_effect")
    source = b'perform_effect(effect_id="effect.send")\nguard_effect(effect_id="effect.send", policy="cid:old")\n'
    guard = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)][1]
    request["source_bytes"] = source
    request["source_digest"] = "sha256:" + hashlib.sha256(source).hexdigest()
    request["anchor"] = security_ast_span_identity(source, guard)
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED

    request, registry = _request()
    request["dcr035_gate"] = object()
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED

    request, registry = _request()
    gate = request["dcr035_gate"]
    request["dcr035_gate"] = replace(
        gate, required_identity_cids={**gate.required_identity_cids, "dcr031": "cid:stale"}
    )
    assert build_security_repair_preview(request, registry=registry).status is SecurityRepairStatus.REJECTED
