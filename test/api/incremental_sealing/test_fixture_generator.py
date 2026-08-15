"""IPS-045: deterministic fixture repository/history generator."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util

_GEN = Path(__file__).resolve().parents[2] / "fixtures/incremental_proof_sealer/generate_fixture_history.py"
_SPEC = importlib.util.spec_from_file_location("ips_fixture_generator", _GEN)
assert _SPEC is not None and _SPEC.loader is not None
_GEN_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GEN_MOD)
EVIDENCE_SUBSET = _GEN_MOD.EVIDENCE_SUBSET
REQUIRED_SCENARIO_KINDS = _GEN_MOD.REQUIRED_SCENARIO_KINDS
generate_corpus = _GEN_MOD.generate_corpus
render_manifest = _GEN_MOD.render_manifest
write_manifest = _GEN_MOD.write_manifest


def test_evidence_subset() -> None:
    assert EVIDENCE_SUBSET == "ips/fixture-corpus@1"


def test_two_generations_are_byte_identical() -> None:
    first = render_manifest()
    second = render_manifest()
    assert first == second
    assert generate_corpus() == generate_corpus()


def test_required_scenario_kinds_have_provenance_and_closures() -> None:
    corpus = generate_corpus()
    kinds = [item["kind"] for item in corpus["scenarios"]]
    assert tuple(kinds) == REQUIRED_SCENARIO_KINDS
    parent = corpus["genesis_parent"]
    for item in corpus["scenarios"]:
        assert item["parent"] == parent
        assert item["changed_artifact_provenance"]["byte_stable"] is True
        assert item["expected_direct_unit_closure"]
        assert item["expected_transitive_unit_closure"]
        assert "required" in item["full_fallback_decision"]
        assert item["scenario_cid"].startswith("sha256:")
        parent = item["scenario_cid"]


def test_simulated_proving_is_rejection_only() -> None:
    corpus = generate_corpus()
    simulated = [item for item in corpus["scenarios"] if item["simulated_proving"] != "absent"]
    assert simulated
    for item in simulated:
        assert item["simulated_proving"] == "rejection_only"
        assert item["production_success"] is False


def test_checked_in_manifest_matches_generator(tmp_path: Path) -> None:
    generated = tmp_path / "fixture_manifest.json"
    write_manifest(generated)
    committed = Path("test/fixtures/incremental_proof_sealer/fixture_manifest.json")
    assert committed.is_file()
    assert generated.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")
    payload = json.loads(committed.read_text(encoding="utf-8"))
    assert payload["schema"].endswith("fixture-manifest@1")
    assert payload["corpus_cid"].startswith("sha256:")
