"""Focused DCR-030 tests for the deterministic datasets LogicIR facade."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import ipfs_accelerate_py.agent_supervisor.proof.ir_integration as integration
import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_mismatch import (
    analyze_mcp_contract_mismatches,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    McpObservationTranscript,
    ObservationStatus,
    RequiredMcpObservation,
    build_mcp_observation_epoch,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_FOREST_PORTABLE_SCHEMA,
    DCR_FOREST_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilities,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)


def _dcr024_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph_body = {
        "schema": MCP_CONTRACT_GRAPH_SCHEMA,
        "interface": MCP_CONTRACT_GRAPH_INTERFACE,
        "authoritative": False,
        "nodes": [],
        "edges": [{"id": "edge:one", "relation": "expects_descriptor"}],
        "blockers": [],
    }
    graph = {
        **graph_body,
        "graph_cid": content_identity(graph_body),
        "canonical_bytes": canonical_json_bytes(graph_body).decode("utf-8"),
    }
    semantic_roots = {"descriptor": "bafy-descriptor"}
    portable = {
        "schema": DCR_FOREST_PORTABLE_SCHEMA,
        "root_policy": {"path": "config/roots.json", "sha256": "sha256:roots"},
        "config_roots": [],
        "exclusions": [],
        "roots": [],
    }
    forest_cid = content_identity(portable)
    forest = {
        "schema": DCR_FOREST_SCHEMA,
        "interface": "DeterministicRepairForest@1",
        "authoritative": False,
        "portable": portable,
        "portable_identity": forest_cid,
        "host": {"schema": DCR_FOREST_SCHEMA + "/host", "roots": {}},
    }
    snapshot_roots = {"forest": forest_cid}
    required = RequiredMcpObservation(
        service_role="fixture-service",
        edge_id="edge:one",
        package="fixture-package",
        operation="fixture.operation",
        direction="request",
        schema="fixture-schema",
        profile="fixture-profile",
        transport="mcp",
    )
    transcript = McpObservationTranscript(
        status=ObservationStatus.OBSERVED,
        failure=None,
        service_role="fixture-service",
        transport="mcp",
        operation="fixture.operation",
        endpoint="http://127.0.0.1:8765",
        request_bytes=b'{"jsonrpc":"2.0"}',
        response_bytes=b'{"jsonrpc":"2.0","result":{}}',
        graph_cid=graph["graph_cid"],
        runtime_receipt_id="bafy-runtime-receipt",
        process_witness_cid="bafy-process-witness",
        template_cid="bafy-template",
    )
    epoch = build_mcp_observation_epoch(
        graph_cid=graph["graph_cid"],
        semantic_roots=semantic_roots,
        snapshot_roots=snapshot_roots,
        required_observations=(required,),
        receipts=(transcript,),
    )
    return (
        analyze_mcp_contract_mismatches(
            graph=graph,
            semantic_roots=semantic_roots,
            snapshot_roots=snapshot_roots,
            transcript=epoch,
        ),
        graph,
        {"forest": forest_cid, "payload": forest},
    )


def _evidence() -> tuple[integration.DatasetsLogicIrEvidence, ...]:
    report, graph, forest = _dcr024_report()
    source = b"def deterministic_logic_input(): pass\n"
    source_digest = "sha256:" + hashlib.sha256(source).hexdigest()
    return (
        integration.DatasetsLogicIrEvidence(
            kind=integration.DatasetsLogicIrEvidenceKind.SOURCE_BYTES,
            cid=content_identity(
                {
                    "schema": integration.DATASETS_LOGIC_IR_INPUT_SCHEMA + "/source-bytes",
                    "sha256": source_digest,
                }
            ),
            payload=source,
        ),
        integration.DatasetsLogicIrEvidence(
            kind=integration.DatasetsLogicIrEvidenceKind.FOREST,
            cid=forest["forest"],
            payload=forest["payload"],
        ),
        integration.DatasetsLogicIrEvidence(
            kind=integration.DatasetsLogicIrEvidenceKind.GRAPH,
            cid=graph["graph_cid"],
            payload=graph,
        ),
        integration.DatasetsLogicIrEvidence(
            kind=integration.DatasetsLogicIrEvidenceKind.FINDING,
            cid=report["findings_cid"],
            payload=report,
        ),
    )


@pytest.mark.parametrize("kind", ("source", "forest", "graph"))
def test_facade_rejects_forged_local_evidence_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    module_origin = tmp_path / "identity.py"
    module_origin.write_text("def compute_identity(): pass\n", encoding="utf-8")
    capabilities, capability_evidence = _capabilities(module_origin)
    monkeypatch.setattr(
        integration,
        "_invoke_datasets_logic_identity",
        lambda _payload: {
            "cid": "baguqeerafacade",
            "digest": "sha256:facade",
            "profile": "ir-canonical-identity-v1",
            "logic_ir_interface": "LogicIR@1",
        },
    )
    evidence = list(_evidence())
    evidence_kind = "source_bytes" if kind == "source" else kind
    index = {
        item.kind.value: position for position, item in enumerate(evidence)
    }[evidence_kind]
    item = evidence[index]
    if kind == "source":
        evidence[index] = integration.DatasetsLogicIrEvidence(
            kind=item.kind, cid="bafy-forged-source", payload=item.payload
        )
    else:
        payload = dict(item.payload)
        field = "portable_identity" if kind == "forest" else "graph_cid"
        payload[field] = "bafy-forged-identity"
        evidence[index] = integration.DatasetsLogicIrEvidence(
            kind=item.kind, cid=item.cid, payload=payload
        )
    result = integration.normalize_datasets_logic_ir(
        tuple(evidence),
        module_origin=module_origin,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )
    assert result.disposition is integration.DatasetsLogicIrDisposition.INTEGRATION_PENDING
    assert any(
        code.endswith(("identity_invalid", "cid_or_digest_invalid"))
        for code in result.reason_codes
    )


def _capabilities(
    module_origin: Path,
) -> tuple[DeterministicRepairCapabilities, tuple[CapabilityEvidenceReceipt, ...]]:
    digest = "module:sha256:" + hashlib.sha256(module_origin.read_bytes()).hexdigest()
    receipt = CapabilityReceipt(
        capability_id=integration.DATASETS_LOGIC_IDENTITY_MODULE,
        status=CapabilityStatus.AVAILABLE,
        origin=str(module_origin.resolve()),
        distribution="ipfs-datasets-py",
        expected_version="1.0.0",
        distribution_version="1.0.0",
        content_digest=digest,
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    inventory = DeterministicRepairCapabilities(
        modules=(receipt,), toolchains=(), network_mode=NetworkMode.OFFLINE
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=integration.DATASETS_LOGIC_IDENTITY_MODULE,
            evidence_kind=kind,
            subject_id=integration.DATASETS_LOGIC_IDENTITY_MODULE,
            subject_digest=digest,
            subject_version="1.0.0",
            transcript_digest="transcript:sha256:"
            + hashlib.sha256(kind.encode("utf-8")).hexdigest(),
            passed=True,
        )
        for kind in ("initialization", "reconstruction", "self_test")
    )
    return inventory, evidence


def test_facade_is_pending_without_current_capability_receipts(tmp_path: Path) -> None:
    module_origin = tmp_path / "identity.py"
    module_origin.write_text("def compute_identity(): pass\n", encoding="utf-8")

    result = integration.normalize_datasets_logic_ir(_evidence(), module_origin=module_origin)

    assert result.disposition is integration.DatasetsLogicIrDisposition.INTEGRATION_PENDING
    assert "dcr004_capability_inventory_missing" in result.reason_codes
    assert result.model_call_count == 0
    assert result.mutation_authorized is False
    assert not result.normalized_ir


def test_facade_normalizes_only_candidate_context_with_exact_current_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_origin = tmp_path / "identity.py"
    module_origin.write_text("def compute_identity(): pass\n", encoding="utf-8")
    capabilities, capability_evidence = _capabilities(module_origin)
    seen: dict[str, Any] = {}

    def identity(payload: dict[str, Any]) -> dict[str, str]:
        seen.update(payload)
        return {
            "cid": "baguqeerafacade",
            "digest": "sha256:facade",
            "profile": "ir-canonical-identity-v1",
            "logic_ir_interface": "LogicIR@1",
            "candidate": True,
        }

    monkeypatch.setattr(integration, "_invoke_datasets_logic_identity", identity)
    result = integration.normalize_datasets_logic_ir(
        _evidence(),
        module_origin=module_origin,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )

    assert result.disposition is integration.DatasetsLogicIrDisposition.NORMALIZED
    assert result.normalized_ir["integration_status"] == "candidate_context_only"
    assert result.mutation_authorized is False
    assert result.model_call_count == 0
    assert seen["schema"] == integration.DATASETS_LOGIC_IR_INPUT_SCHEMA
    assert result.module_binding["origin"] == str(module_origin.resolve())

    module_origin.write_text("def compute_identity(): return 'changed'\n", encoding="utf-8")
    stale = integration.normalize_datasets_logic_ir(
        _evidence(),
        module_origin=module_origin,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )
    assert stale.disposition is integration.DatasetsLogicIrDisposition.INTEGRATION_PENDING
    assert "datasets_logic_module_binding_mismatch" in stale.reason_codes


def test_facade_rejects_untyped_or_fixture_like_evidence(tmp_path: Path) -> None:
    module_origin = tmp_path / "identity.py"
    module_origin.write_text("def compute_identity(): pass\n", encoding="utf-8")
    unsupported = integration.normalize_datasets_logic_ir((object(),), module_origin=module_origin)
    assert unsupported.disposition is integration.DatasetsLogicIrDisposition.UNSUPPORTED
    assert unsupported.mutation_authorized is False

    with pytest.raises(ValueError, match="unsupported DCR-030 evidence kind"):
        integration.DatasetsLogicIrEvidence(
            kind="fixture_bridge",
            cid="bafy-fixture",
            payload={"fixture": True},
        )
