"""Focused DCR-001..004 artifact materialization tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilities,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.deterministic_artifacts import (
    DeterministicArtifactError,
    materialize_deterministic_repair_artifacts,
    verify_deterministic_repair_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
AUTHORITY_CONFIG = REPOSITORY_ROOT / "config" / "deterministic_contract_repair_authority.json"
ROOT_CONFIG = REPOSITORY_ROOT / "config" / "deterministic_contract_repair_roots.json"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for relative in (
        "swissknife",
        "Mcp-Plus-Plus",
        "external/ipfs_accelerate",
        "external/ipfs_datasets",
        "external/ipfs_kit",
    ):
        (workspace / relative).mkdir(parents=True)
    return workspace


def _inventory() -> tuple[DeterministicRepairCapabilities, tuple[CapabilityEvidenceReceipt, ...]]:
    digest = "module:sha256:" + hashlib.sha256(b"fixture-capability").hexdigest()
    module = CapabilityReceipt(
        capability_id="fixture.logic",
        status=CapabilityStatus.AVAILABLE,
        origin="/reviewed/fixture.py",
        distribution="fixture-distribution",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest=digest,
        symbols=("ExactLogic",),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    inventory = DeterministicRepairCapabilities(
        modules=(module,), toolchains=(), network_mode=NetworkMode.OFFLINE
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=module.capability_id,
            evidence_kind=kind,
            subject_id=module.capability_id,
            subject_digest=module.content_digest,
            subject_version=module.distribution_version,
            transcript_digest="transcript:sha256:"
            + hashlib.sha256(kind.encode("utf-8")).hexdigest(),
            passed=True,
        )
        for kind in ("initialization", "reconstruction", "self_test")
    )
    return inventory, evidence


def test_materializer_is_canonical_non_executable_and_tamper_evident(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = tmp_path / "artifacts"
    output.mkdir()
    inventory, evidence = _inventory()

    outputs = materialize_deterministic_repair_artifacts(
        authority_policy_path=AUTHORITY_CONFIG,
        root_policy_path=ROOT_CONFIG,
        workspace_root=workspace,
        output_dir=output,
        capabilities=inventory,
    )
    first_bytes = {name: path.read_bytes() for name, path in outputs.items()}
    assert set(outputs) == {
        "no-llm-policy.json",
        "disposition-schema.json",
        "root-policy.json",
        "capabilities.json",
    }
    capabilities = json.loads(first_bytes["capabilities.json"])
    assert capabilities["capabilities"]["availability"] == "unavailable"
    assert capabilities["capabilities"]["authoritative"] is False
    assert "self_test_passed" not in capabilities["capabilities"]

    materialize_deterministic_repair_artifacts(
        authority_policy_path=AUTHORITY_CONFIG,
        root_policy_path=ROOT_CONFIG,
        workspace_root=workspace,
        output_dir=output,
        capabilities=inventory,
    )
    assert first_bytes == {name: path.read_bytes() for name, path in outputs.items()}

    available_outputs = materialize_deterministic_repair_artifacts(
        authority_policy_path=AUTHORITY_CONFIG,
        root_policy_path=ROOT_CONFIG,
        workspace_root=workspace,
        output_dir=output,
        capabilities=inventory,
        capability_evidence=evidence,
    )
    capability_artifact = json.loads(available_outputs["capabilities.json"].read_bytes())
    assert capability_artifact["capabilities"]["availability"] == "available"
    assert capability_artifact["capabilities"]["authoritative"] is True

    disposition = json.loads(available_outputs["disposition-schema.json"].read_bytes())
    assert disposition["non_executable"] is True
    assert disposition["public_dispositions"] == [
        "proved_valid",
        "refuted_repairable",
        "repaired_pending_validation",
        "abstain_review",
        "defer_capability",
        "rejected",
        "completed",
    ]
    assert disposition["authority_stages"] == [
        "observed",
        "derived",
        "admitted",
        "mutated",
        "post_edit_validated",
        "reproved",
        "published",
    ]
    assert (
        disposition["source_authority_policy"]["sha256"]
        == hashlib.sha256(AUTHORITY_CONFIG.read_bytes()).hexdigest()
    )
    assert (
        disposition["source_root_policy"]["sha256"]
        == hashlib.sha256(ROOT_CONFIG.read_bytes()).hexdigest()
    )

    verify_deterministic_repair_artifacts(
        authority_policy_path=AUTHORITY_CONFIG,
        root_policy_path=ROOT_CONFIG,
        workspace_root=workspace,
        output_dir=output,
        capabilities=inventory,
        capability_evidence=evidence,
    )
    (output / "no-llm-policy.json").write_bytes(b"{}\n")
    with pytest.raises(DeterministicArtifactError, match="bytes do not match"):
        verify_deterministic_repair_artifacts(
            authority_policy_path=AUTHORITY_CONFIG,
            root_policy_path=ROOT_CONFIG,
            workspace_root=workspace,
            output_dir=output,
            capabilities=inventory,
            capability_evidence=evidence,
        )


def test_materializer_rejects_tampered_reviewed_config_and_implicit_output(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = tmp_path / "artifacts"
    output.mkdir()
    authority = tmp_path / "tampered-authority.json"
    authority.write_text(
        AUTHORITY_CONFIG.read_text(encoding="utf-8").replace(
            '"llm_call_budget": 0', '"llm_call_budget": 1'
        ),
        encoding="utf-8",
    )
    with pytest.raises(DeterministicArtifactError, match="reviewed policy"):
        materialize_deterministic_repair_artifacts(
            authority_policy_path=authority,
            root_policy_path=ROOT_CONFIG,
            workspace_root=workspace,
            output_dir=output,
        )
    with pytest.raises(DeterministicArtifactError, match="output_dir"):
        materialize_deterministic_repair_artifacts(
            authority_policy_path=AUTHORITY_CONFIG,
            root_policy_path=ROOT_CONFIG,
            workspace_root=workspace,
            output_dir=tmp_path / "not-created",
        )
