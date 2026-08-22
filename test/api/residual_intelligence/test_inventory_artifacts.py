from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PROGRAM_ID,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.rights import (
    TrainingCorpusAdmission,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DecodeStatus,
    decode_structured_output,
    grammar_for,
)

from .helpers import split_fixture

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs" / "architecture" / "residual_intelligence_inventory"
BENCHMARK = ROOT / "benchmarks" / "agent_supervisor" / "residual_intelligence"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_baseline_and_prerequisites_bind_exact_start_tree() -> None:
    baseline = _json(INVENTORY / "baseline.json")
    matrix = _json(INVENTORY / "prerequisite_matrix.json")
    source = baseline["source"]
    assert isinstance(source, dict)
    assert source["commit"] == matrix["source_revision"]
    assert source["tree"] == matrix["source_tree"]
    assert baseline["program_identifier"] == PROGRAM_ID
    findings = matrix["findings"]
    assert isinstance(findings, list)
    assert {item["authority"] for item in findings} == {
        "SemanticCompressionHarness",
        "SemanticCompressionGovernor",
        "AdversarialAssuranceEngine",
        "IncrementalVerificationPlanner",
        "IncrementalProofSealer",
        "AdaptivePlanner",
        "ContextCompiler",
        "SupervisorControlService",
        "AutonomousMetaController",
        "ProofCarryingProcedureCompiler",
        "ProofCarryingArchitectureRefactorer",
        "ProofGroundedIRLearningFabric",
        "LearningCheckpointBinding",
    }
    assert matrix["global_blocker"] is None


def test_pgir_gate_preserves_no_go_without_checkpoint_or_promotion() -> None:
    gate = _json(INVENTORY / "pgir_training_gate.json")
    assert gate["decision"] == "training_unavailable"
    assert gate["training_attempted"] is False
    assert gate["checkpoint_created"] is False
    assert gate["promotion_attempted"] is False
    pgir = gate["pgir"]
    assert isinstance(pgir, dict)
    assert pgir["decision"] == "no_go"
    assert pgir["training_admitted_rows"] == 0
    assert pgir["candidate_checkpoint"] is None
    assert pgir["publication_authorized"] is False


def test_inventory_does_not_invent_trajectory_rows() -> None:
    inventory = _json(INVENTORY / "residual_model_call_inventory.json")
    assert inventory["trajectory_observation_count"] == 0
    assert inventory["training_examples_created"] == 0
    assert len(inventory["required_join_fields"]) == 18
    for surface in inventory["surfaces"]:
        assert surface["authoritative"] is False


def test_corpus_construction_is_implemented_without_fake_admission() -> None:
    construction = _json(INVENTORY / "corpus_construction.json")
    first_party = construction["first_party_trajectory_corpus"]
    synthetic = construction["synthetic_and_adversarial_corpus"]
    assert isinstance(first_party, dict)
    assert isinstance(synthetic, dict)
    assert first_party["trajectory_rows_training_admitted"] == 0
    assert synthetic["example_count"] == 4
    assert synthetic["training_admitted"] is False
    assert construction["overall_training_availability"] == "training_unavailable"
    assert construction["weights_created"] is False
    assert construction["checkpoint_created"] is False


def test_benchmark_manifest_uses_exact_closed_taxonomy_without_qualification() -> None:
    manifest = _json(BENCHMARK / "manifest.json")
    assert manifest["status"] == "staged_not_qualified"
    assert set(manifest["task_families"]) == {family.value for family in ResidualTaskFamily}
    assert manifest["training_admission"] == "training_unavailable"
    assert manifest["promotion_evidence"] is False


def test_stored_compact_fixture_decodes_through_current_grammar() -> None:
    fixture = _json(BENCHMARK / "tranche1_contract_cases.json")
    valid = fixture["cases"][0]
    result = decode_structured_output(
        json.dumps(valid["payload"]), grammar_for(valid["task_family"])
    )
    assert result.status is DecodeStatus.VALID


def test_stored_fixture_admission_round_trips_but_cannot_train() -> None:
    payload = _json(BENCHMARK / "synthetic_training_admission.json")
    admission = TrainingCorpusAdmission.from_dict(payload)
    assert admission.to_dict() == payload
    assert admission.can_train is False


def test_stored_semantic_split_matches_current_compiler_identity() -> None:
    stored = _json(BENCHMARK / "synthetic_split_manifest.json")
    _, compiled = split_fixture()
    assert compiled.to_dict() == stored
    assert compiled.leakage_audit().passed is True
