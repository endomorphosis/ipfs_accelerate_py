"""Focused DCR-043 tests: previews are static and evidence-bound only."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from typing import Any

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.dispatch_repairs import (
    DispatchPreviewStatus,
    build_dispatch_repair_preview,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry_repairs import (
    ast_span_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "dispatch.cec-prove",
            "kind": "replace_unique_registration",
            "input_schema": {
                "type": "object",
                "required": ["source_digest", "relative_path", "semantic_target"],
                "properties": {
                    "source_digest": "sha256",
                    "relative_path": "path",
                    "semantic_target": "symbol",
                },
                "additional_properties": False,
            },
            "owner_root": "ipfs_accelerate",
            "write_scope": ["dispatch.py"],
            "before_predicates": ["static_dispatch"],
            "after_predicates": ["equivalent_dispatch"],
            "applicability_proofs": ["dcr031_target"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "before_bytes"},
            "validation_commands": [["pytest", "test_dispatch.py"]],
        }
    )
    return OperatorRegistry(
        [descriptor], reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _anchor(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    return ast_span_identity(source, call)


def _capability() -> tuple[CapabilityReceipt, tuple[CapabilityEvidenceReceipt, ...]]:
    digest = "module:sha256:" + "a" * 64
    receipt = CapabilityReceipt(
        capability_id="ipfs_datasets_py.logic",
        status=CapabilityStatus.AVAILABLE,
        origin="/fixture/ipfs_datasets_py/logic_tools.py",
        distribution="ipfs-datasets-py",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest=digest,
        symbols=("cec_prove",),
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
            subject_digest=digest,
            subject_version="1.0",
            transcript_digest=f"transcript:sha256:{kind * 16}"[:82],
            passed=True,
        )
        for kind in ("initialization", "reconstruction", "self_test")
    )
    return receipt, evidence


def _obligation() -> McpGraphContractObligation:
    graph_cid, candidate_cid, input_cid = "sha256:graph", "sha256:candidate", "sha256:input"
    return McpGraphContractObligation(
        obligation_id="sha256:obligation",
        family=McpObligationFamily.REGISTRY_DISPATCH,
        fragment=McpObligationFragment.REGISTRY,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid=graph_cid,
        candidate_cid=candidate_cid,
        input_cids=(input_cid,),
        cid_bindings=(graph_cid, candidate_cid, input_cid),
        schema_bindings=("mcp-graph-obligation@1",),
        effect_semantics="ipfs_datasets.logic.cec_prove.v1",
    )


def _request(source: bytes) -> dict[str, Any]:
    capability, evidence = _capability()
    return {
        "operator_id": "dispatch.cec-prove",
        "action": "expose_cec_prove_dispatch",
        "owner_root": "ipfs_accelerate",
        "relative_path": "dispatch.py",
        "source_bytes": source,
        "source_digest": _digest(source),
        "anchor": _anchor(source),
        "payload": {
            "operation": "logic.cec-prove",
            "dispatcher_api": "dispatcher.register",
            "handler_symbol": "cec_prove",
            "semantic_target": "ipfs_datasets.logic.cec_prove.v1",
        },
        "handler_semantic": {
            "handler_id": "ipfs_datasets.logic_tools.cec_prove",
            "module": "ipfs_datasets_py.logic_tools",
            "symbol": "cec_prove",
            "semantic_cid": "sha256:handler-semantics",
            "existing": True,
        },
        "capability_receipt": capability,
        "capability_evidence_receipts": evidence,
        "obligation": _obligation(),
    }


def test_static_cec_prove_preview_has_inverse_and_equivalence_predicates() -> None:
    source = (
        b"from ipfs_datasets_py.logic_tools import cec_prove\n"
        b"dispatcher.register('health', cec_prove)\n"
    )
    registry = _registry()
    preview = build_dispatch_repair_preview(
        _request(source), registry=registry, manifest_cid=registry.report()["registry_cid"]
    )

    assert preview.status is DispatchPreviewStatus.PREVIEWED
    assert b"dispatcher.register('logic.cec-prove', cec_prove)" in preview.after_bytes
    assert preview.forward_diff and preview.inverse_diff
    result = preview.to_dict()
    assert result["execution_authorized"] is False
    assert result["activation_status"] == "integration_pending_dcr035_dcr040_dcr070_dcr072"
    assert result["equivalence_predicates"][0]["transport"] == "loopback_mcp"


def test_reviewed_static_registration_action_uses_the_same_closed_evidence() -> None:
    source = (
        b"from ipfs_datasets_py.logic_tools import cec_prove\n"
        b"dispatcher.register('health', cec_prove)\n"
    )
    registry = _registry()
    request = _request(source)
    request["action"] = "register_static_dispatch"

    assert (
        build_dispatch_repair_preview(
            request, registry=registry, manifest_cid=registry.report()["registry_cid"]
        ).status
        is DispatchPreviewStatus.PREVIEWED
    )


def test_dynamic_or_synthesized_handler_abstains_without_preview() -> None:
    dynamic = (
        b"from ipfs_datasets_py.logic_tools import cec_prove\n"
        b"dispatcher.register(operation_name, cec_prove)\n"
    )
    synthesized = (
        b"def cec_prove(value):\n    return value\n\ndispatcher.register('health', cec_prove)\n"
    )
    registry = _registry()
    manifest_cid = registry.report()["registry_cid"]

    assert (
        build_dispatch_repair_preview(
            _request(dynamic), registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.ABSTAINED
    )
    assert (
        build_dispatch_repair_preview(
            _request(synthesized), registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.ABSTAINED
    )


def test_stale_capability_target_and_manifest_are_rejected() -> None:
    source = (
        b"from ipfs_datasets_py.logic_tools import cec_prove\n"
        b"dispatcher.register('health', cec_prove)\n"
    )
    registry = _registry()
    manifest_cid = registry.report()["registry_cid"]
    no_capability = _request(source)
    no_capability["capability_receipt"] = {"available": True}
    stale_typed = _request(source)
    stale_typed["capability_receipt"] = replace(
        stale_typed["capability_receipt"], content_digest="module:sha256:" + "b" * 64
    )
    target_drift = _request(source)
    target_drift["obligation"] = replace(_obligation(), effect_semantics="different.target")

    assert (
        build_dispatch_repair_preview(
            no_capability, registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.REJECTED
    )
    assert (
        build_dispatch_repair_preview(
            target_drift, registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.REJECTED
    )
    assert (
        build_dispatch_repair_preview(
            stale_typed, registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.REJECTED
    )
    assert (
        build_dispatch_repair_preview(
            _request(source), registry=registry, manifest_cid="sha256:stale"
        ).status
        is DispatchPreviewStatus.REJECTED
    )
    provider_route = _request(source)
    provider_route["payload"] = {**provider_route["payload"], "operation": "provider.route"}
    assert (
        build_dispatch_repair_preview(
            provider_route, registry=registry, manifest_cid=manifest_cid
        ).status
        is DispatchPreviewStatus.REJECTED
    )
