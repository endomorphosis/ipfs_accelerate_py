"""Focused DCR-045 UI projection operator previews; no code is applied."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_desktop_expectations import (
    DESKTOP_EXPECTATIONS_INTERFACE,
    DESKTOP_EXPECTATIONS_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_FOREST_PORTABLE_SCHEMA,
    DCR_FOREST_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.ui_projection_repairs import (
    UiProjectionPreviewStatus,
    UiProjectionRepairRequest,
    preview_ui_projection_repair,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_integration import (
    DatasetsLogicIrDisposition,
    DatasetsLogicIrResult,
)


def _descriptor_registry() -> tuple[OperatorDescriptor, OperatorRegistry]:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "ui.projection.replace",
            "kind": "replace_exact_bytes",
            "owner_root": "swissknife",
            "write_scope": ["ui/fixture.ts"],
            "before_predicates": ["projection_exact"],
            "after_predicates": ["projection_exact"],
            "applicability_proofs": ["dcr021"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["python", "-m", "py_compile", "fixture.py"]],
        }
    )
    return descriptor, OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _request() -> UiProjectionRepairRequest:
    portable = {
        "schema": DCR_FOREST_PORTABLE_SCHEMA,
        "root_policy": {"path": "roots", "sha256": "sha256:roots"},
        "config_roots": [],
        "exclusions": [],
        "roots": [],
    }
    forest = {
        "schema": DCR_FOREST_SCHEMA,
        "interface": "DeterministicRepairForest@1",
        "authoritative": True,
        "portable": portable,
        "portable_identity": content_identity(portable),
    }
    projection = {
        "operation": "desktop.open",
        "request_schema": "OpenRequest",
        "result_schema": "OpenResult",
        "effect": "open_window",
        "security": "policy:open",
        "transport": "http",
    }
    effective = {
        "operation": "desktop.open",
        "request": "OpenRequest",
        "result": "OpenResult",
        "transport": "http",
        "authority_class": "reviewed_declaration",
        "source_span": {"root": "swissknife", "path": "ui/fixture.ts", "sha256": "sha256:source"},
    }
    desktop_body = {
        "schema": DESKTOP_EXPECTATIONS_SCHEMA,
        "interface": DESKTOP_EXPECTATIONS_INTERFACE,
        "authoritative": False,
        "scan_mode": "static_source_only",
        "roots": ["swissknife", "mcp-plus-plus"],
        "consumers": [],
        "evidence": [effective],
        "effective_expectations": [effective],
        "blockers": [],
    }
    desktop = {
        **desktop_body,
        "identity": "sha256:"
        + hashlib.sha256(
            json.dumps(desktop_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    graph_body = {
        "schema": MCP_CONTRACT_GRAPH_SCHEMA,
        "interface": MCP_CONTRACT_GRAPH_INTERFACE,
        "authoritative": False,
        "nodes": [],
        "edges": [],
        "blockers": [],
    }
    graph = {
        **graph_body,
        "graph_cid": content_identity(graph_body),
        "canonical_bytes": canonical_json_bytes(graph_body).decode(),
    }
    candidate = DatasetsLogicIrResult(
        DatasetsLogicIrDisposition.NORMALIZED,
        ("candidate_context_only",),
        tuple(sorted((forest["portable_identity"], graph["graph_cid"]))),
        module_binding={
            "module": "fixture",
            "origin": "/fixture",
            "version": "1",
            "content_digest": "module:sha256:fixture",
        },
        normalized_ir={
            "integration_status": "candidate_context_only",
            "input_schema": "ipfs_accelerate_py/agent-supervisor/datasets-logic-ir-input@1",
        },
    )
    descriptor, registry = _descriptor_registry()
    source = b"const answer = BAD_DESCRIPTOR;\n"
    start, end = (
        source.index(b"BAD_DESCRIPTOR"),
        source.index(b"BAD_DESCRIPTOR") + len(b"BAD_DESCRIPTOR"),
    )
    return UiProjectionRepairRequest(
        "ui",
        source,
        "sha256:" + hashlib.sha256(source).hexdigest(),
        "swissknife",
        "ui/fixture.ts",
        start,
        end,
        "sha256:" + hashlib.sha256(source[start:end]).hexdigest(),
        b"GOOD_DESCRIPTOR",
        "literal_exact_span",
        projection,
        dict(projection),
        content_identity(projection),
        content_identity(projection),
        forest,
        desktop,
        graph,
        candidate,
        descriptor,
        registry,
        descriptor.descriptor_id,
        registry.report()["registry_cid"],
    )


def test_projection_preview_is_reversible_non_authoritative_and_pending() -> None:
    preview = preview_ui_projection_repair(_request())
    assert preview.status is UiProjectionPreviewStatus.PREVIEWED
    assert preview.after_bytes == b"const answer = GOOD_DESCRIPTOR;\n"
    payload = preview.to_dict()
    assert payload["server_truth_created"] is False
    assert payload["implementation_authorized"] is False
    assert payload["activation_status"] == "integration_pending_dcr035_dcr040_dcr070_dcr072"


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, anchor_kind="dynamic"),
        lambda value: replace(
            value, reverse_projection={**value.reverse_projection, "security": "weakened"}
        ),
        lambda value: replace(value, model_call_count=1),
        lambda value: replace(
            value,
            logic_candidate=replace(
                value.logic_candidate, disposition=DatasetsLogicIrDisposition.INTEGRATION_PENDING
            ),
        ),
        lambda value: replace(
            value,
            desktop_expectations={**value.desktop_expectations, "blockers": [{"kind": "blocked"}]},
        ),
        lambda value: replace(value, reviewed_descriptor_cid="bafy-forged"),
    ],
)
def test_ambiguous_weakened_stale_or_unreviewed_inputs_never_preview(change) -> None:
    preview = preview_ui_projection_repair(change(_request()))
    assert preview.status in {
        UiProjectionPreviewStatus.ABSTAINED,
        UiProjectionPreviewStatus.REJECTED,
    }
    assert not preview.forward_cid


def test_generated_fixture_authority_and_descriptor_only_success_abstain() -> None:
    request = _request()
    generated = dict(request.desktop_expectations)
    effective = dict(generated["effective_expectations"][0])
    effective["authority_class"] = "generated"
    generated["effective_expectations"] = [effective]
    body = dict(generated)
    body.pop("identity")
    generated["identity"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    preview = preview_ui_projection_repair(replace(request, desktop_expectations=generated))
    assert preview.status is UiProjectionPreviewStatus.ABSTAINED
