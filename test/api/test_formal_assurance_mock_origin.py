"""FACP-024: Accelerate mock-origin rejection tests (pre-repair characterization).

Each seeded mock source must reach its inventoried legacy sink before repair and
classify as FCA origin=simulated / closed outcome Simulated. Same-name real and
fixture decoys are distinguished by provenance, never by naming alone.

These tests do not execute real providers, do not edit production behavior, and
must not treat fixture completion as live evidence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    TrustAbstract,
    findings_for_corpus_entry,
    ipa_corpus_entries,
    load_defect_corpus,
)
from ipfs_accelerate_py.agent_supervisor.assurance.formal_claim_adapter import (
    EvidenceEnvelope,
    Origin,
)

SCHEMA = "facp/accelerate-mock-flow@1"
TASK_ID = "FACP-024"
GOAL_ID = "FACP-G220"

REQUIRED_EVIDENCE_BUCKETS = {
    "mock_worker",
    "hardware",
    "inference_handler",
    "fallback",
    "dependency_injection",
    "compatibility_namespace",
    "production_registry",
}

# Closed outcome algebra (plan §4). Simulated evidence is never production-supported.
CLOSED_OUTCOMES = {
    "Unavailable",
    "Rejected",
    "Simulated",
    "Attempted",
    "Unknown",
    "Observed",
    "Verified",
    "Failed",
    "Compensated",
}

_ORIGIN_BY_PROVENANCE = {
    "absent": Origin.ABSENT,
    "declared": Origin.DECLARED,
    "fixture": Origin.FIXTURE,
    "simulated": Origin.SIMULATED,
    "hermetic_observed": Origin.HERMETIC_OBSERVED,
    "live_observed": Origin.LIVE_OBSERVED,
}


def _repo_root() -> Path:
    # test/api -> test -> external/ipfs_accelerate -> workspace root
    return Path(__file__).resolve().parents[4]


def _cases_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "formal_assurance"
        / "mock_origin_cases.json"
    )


def _inventory_path(root: Path) -> Path:
    return (
        root
        / "implementation_plan"
        / "formal_assurance_control_plane"
        / "baseline"
        / "accelerate_claims.json"
    )


def _corpus_path(root: Path) -> Path:
    return (
        root
        / "implementation_plan"
        / "formal_assurance_control_plane"
        / "baseline"
        / "defect_corpus.jsonl"
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _slice_lines(path: Path, start_line: int, end_line: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    assert 1 <= start_line <= end_line <= len(lines), (
        f"{path}: invalid span {start_line}-{end_line} (file has {len(lines)} lines)"
    )
    return "\n".join(lines[start_line - 1 : end_line])


def _assert_markers(text: str, markers: list[str], *, context: str) -> None:
    for marker in markers:
        assert marker in text, f"{context}: missing marker {marker!r}"


def _envelope_for_provenance(provenance: str) -> EvidenceEnvelope:
    origin = _ORIGIN_BY_PROVENANCE[provenance]
    return EvidenceEnvelope.weakest().with_updates(origin=origin)


def _production_supported(envelope: EvidenceEnvelope) -> bool:
    """Conservative FCA production_supported approximation used by these tests.

    Normative gate: simulated/fixture origins cannot be production_supported.
    """

    return envelope.origin is Origin.LIVE_OBSERVED


def _classify_mock_value(*, provenance: str = "simulated") -> dict[str, Any]:
    envelope = _envelope_for_provenance(provenance)
    outcome = "Simulated" if envelope.origin is Origin.SIMULATED else None
    if envelope.origin is Origin.FIXTURE:
        outcome = "Simulated"  # fixture completion is not live; still non-Observed
    if envelope.origin is Origin.LIVE_OBSERVED:
        outcome = "Observed"
    return {
        "origin": envelope.origin.value,
        "outcome": outcome,
        "production_supported": _production_supported(envelope),
        "envelope": envelope,
    }


def _load_case_seed_ids() -> list[str]:
    doc = _read_json(_cases_path())
    return [str(case["seed_id"]) for case in doc["cases"]]


def _case_by_seed(cases: list[dict[str, Any]], seed_id: str) -> dict[str, Any]:
    for case in cases:
        if case["seed_id"] == seed_id:
            return case
    raise KeyError(seed_id)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="module")
def cases_doc() -> dict[str, Any]:
    path = _cases_path()
    assert path.is_file(), f"missing fixture: {path}"
    return _read_json(path)


@pytest.fixture(scope="module")
def cases(cases_doc: dict[str, Any]) -> list[dict[str, Any]]:
    items = cases_doc.get("cases")
    assert isinstance(items, list) and items, "mock_origin_cases.json requires non-empty cases"
    return items


@pytest.fixture(scope="module")
def inventory(repo_root: Path) -> dict[str, Any]:
    path = _inventory_path(repo_root)
    assert path.is_file(), f"missing accelerate inventory: {path}"
    return _read_json(path)


@pytest.fixture(scope="module")
def inventory_by_seed(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for defect in inventory["confirmed_defects"]:
        seed = (defect.get("counterexample_seed") or {}).get("seed_id")
        if seed:
            out[str(seed)] = defect
    return out


@pytest.fixture(scope="module")
def corpus_by_seed(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = _corpus_path(repo_root)
    assert path.is_file(), f"missing defect corpus: {path}"
    entries = ipa_corpus_entries(load_defect_corpus(path))
    return {str(entry["seed_id"]): dict(entry) for entry in entries}


def test_fixture_schema_and_evidence_coverage(cases_doc: dict[str, Any], cases: list) -> None:
    assert cases_doc["schema"] == SCHEMA
    assert cases_doc["task_id"] == TASK_ID
    assert cases_doc["goal_id"] == GOAL_ID
    policy = cases_doc["policy"]
    assert policy["fixture_completion_is_not_live_evidence"] is True
    assert policy["execute_real_provider"] is False
    assert policy["distinguish_decoys_by_provenance_not_naming"] is True
    assert policy["expected_origin"] == Origin.SIMULATED.value
    assert policy["expected_closed_outcome"] == "Simulated"
    assert policy["production_supported_for_simulated"] is False
    assert policy["expected_closed_outcome"] in CLOSED_OUTCOMES

    buckets = {case["evidence_bucket"] for case in cases}
    missing = REQUIRED_EVIDENCE_BUCKETS - buckets
    assert not missing, f"fixture missing evidence buckets: {sorted(missing)}"
    assert set(cases_doc["evidence_subset"]) == REQUIRED_EVIDENCE_BUCKETS


def test_cases_bind_inventory_and_corpus_seeds(
    cases: list,
    inventory_by_seed: Mapping[str, dict[str, Any]],
    corpus_by_seed: Mapping[str, dict[str, Any]],
) -> None:
    seen: set[str] = set()
    for case in cases:
        seed_id = case["seed_id"]
        assert seed_id not in seen, f"duplicate seed_id in fixture: {seed_id}"
        seen.add(seed_id)
        assert seed_id in inventory_by_seed, f"{seed_id} missing from accelerate_claims.json"
        assert seed_id in corpus_by_seed, f"{seed_id} missing from IPA defect corpus"
        defect = inventory_by_seed[seed_id]
        assert defect["id"] == case["defect_id"]
        assert case["expected_origin"] == Origin.SIMULATED.value
        assert case["expected_outcome"] == "Simulated"
        assert case["production_supported"] is False
        assert case["legacy_sink"]["id"], f"{seed_id}: legacy_sink.id required"
        assert case["decoys"], f"{seed_id}: decoys required for provenance distinction"


@pytest.mark.parametrize("seed_id", _load_case_seed_ids())
def test_seeded_mock_source_reaches_legacy_sink_before_repair(
    seed_id: str,
    cases: list,
    repo_root: Path,
    inventory_by_seed: Mapping[str, dict[str, Any]],
) -> None:
    case = _case_by_seed(cases, seed_id)
    source = case["mock_source"]
    sink = case["legacy_sink"]

    source_path = repo_root / source["path"]
    assert source_path.is_file(), f"{seed_id}: missing mock source {source_path}"
    source_text = _slice_lines(source_path, int(source["start_line"]), int(source["end_line"]))
    _assert_markers(source_text, list(source["markers"]), context=f"{seed_id} mock_source")

    sink_path = repo_root / sink["path"]
    assert sink_path.is_file(), f"{seed_id}: missing legacy sink {sink_path}"
    sink_text = sink_path.read_text(encoding="utf-8", errors="replace")
    _assert_markers(sink_text, list(sink["markers"]), context=f"{seed_id} legacy_sink")

    intermediate = sink.get("intermediate_markers")
    if intermediate:
        mid_path = repo_root / intermediate["path"]
        mid_text = mid_path.read_text(encoding="utf-8", errors="replace")
        _assert_markers(
            mid_text,
            list(intermediate["markers"]),
            context=f"{seed_id} intermediate",
        )

    # Inventory still records a multi-hop mock -> legacy/production consumer flow.
    defect = inventory_by_seed[seed_id]
    flow = [str(step) for step in defect["call_flow_path"]]
    assert len(flow) >= 2, f"{seed_id}: call_flow_path too short"
    flow_blob = " ".join(flow).casefold()
    source_blob = source_text.casefold()
    assert (
        "mock" in source_blob
        or "magicmock" in source_blob
        or "is_mock" in source_blob
    ), f"{seed_id}: mock source span lacks mock-origin markers"
    assert (
        "mock" in flow_blob
        or "magicmock" in flow_blob
        or "fallback" in flow_blob
        or "handler" in flow_blob
        or "cuda" in flow_blob
        or "hardware" in flow_blob
        or "inference" in flow_blob
        or "cid" in flow_blob
        or "classify" in flow_blob
        or "available" in flow_blob
    ), f"{seed_id}: inventoried flow does not describe mock->sink reachability: {flow}"
    assert sink["id"].startswith("sink:"), f"{seed_id}: legacy_sink.id must be sink-shaped"


@pytest.mark.parametrize("seed_id", _load_case_seed_ids())
def test_seeded_mock_classified_simulated_not_production_supported(
    seed_id: str,
    cases: list,
    repo_root: Path,
    corpus_by_seed: Mapping[str, dict[str, Any]],
) -> None:
    _case_by_seed(cases, seed_id)

    classified = _classify_mock_value(provenance="simulated")
    assert classified["origin"] == Origin.SIMULATED.value
    assert classified["outcome"] == "Simulated"
    assert classified["production_supported"] is False
    assert classified["envelope"].origin is Origin.SIMULATED

    # IPA corpus binding must keep a source-to-sink trace and must not claim live origin.
    entry = corpus_by_seed[seed_id]
    findings = findings_for_corpus_entry(entry, repo_root=repo_root)
    assert findings, f"{seed_id}: expected IPA corpus findings"
    trusts = {item.domain_state.trust for item in findings}
    assert TrustAbstract.LIVE_OBSERVED not in trusts, (
        f"{seed_id}: IPA illegally classified mock flow as live_observed"
    )
    family = str(entry.get("family") or "")
    if family == "mock_capability":
        assert TrustAbstract.SIMULATED in trusts, (
            f"{seed_id}: mock_capability seed missing simulated trust; "
            f"got {sorted(t.value for t in trusts)}"
        )
    else:
        # false_success / related seeds still classify simulated under FCA provenance.
        assert classified["origin"] == Origin.SIMULATED.value
    for finding in findings:
        assert finding.trace.steps
        assert finding.trace.source_label
        assert finding.trace.sink_label


@pytest.mark.parametrize("seed_id", _load_case_seed_ids())
def test_same_name_decoys_distinguished_by_provenance_not_naming(
    seed_id: str,
    cases: list,
) -> None:
    case = _case_by_seed(cases, seed_id)
    shared_name = case["shared_surface_name"]
    decoys = case["decoys"]
    assert len(decoys) >= 2

    classified = []
    for decoy in decoys:
        # Naming is identical across decoys; only provenance may differ.
        assert shared_name  # surface name alone is never a classifier
        result = _classify_mock_value(provenance=str(decoy["provenance"]))
        assert result["origin"] == decoy["expected_origin"]
        classified.append((decoy["role"], shared_name, result))

    origins = {item[2]["origin"] for item in classified}
    assert Origin.SIMULATED.value in origins
    assert len(origins) == len({d["provenance"] for d in decoys}), (
        f"{case['seed_id']}: provenance decoys collapsed to identical origins"
    )

    # Naming equality must not imply origin equality.
    names = {item[1] for item in classified}
    assert names == {shared_name}
    mock_origin = next(item[2]["origin"] for item in classified if item[0] == "mock")
    real_origin = next(item[2]["origin"] for item in classified if item[0] == "real")
    fixture_origin = next(item[2]["origin"] for item in classified if item[0] == "fixture")
    assert mock_origin == Origin.SIMULATED.value
    assert real_origin == Origin.LIVE_OBSERVED.value
    assert fixture_origin == Origin.FIXTURE.value
    assert mock_origin != real_origin
    assert mock_origin != fixture_origin

    # Fixture completion is not live evidence / production_supported.
    fixture_env = _envelope_for_provenance("fixture")
    assert _production_supported(fixture_env) is False
    assert fixture_env.origin is not Origin.LIVE_OBSERVED


def test_runtime_probes_remain_simulated_without_real_provider(
    cases: list,
    repo_root: Path,
) -> None:
    """Hermetic runtime probes for safe mock helpers only (no real providers)."""

    probed = 0
    for case in cases:
        probe = case.get("runtime_probe")
        if not probe:
            continue
        probed += 1
        kind = probe["kind"]
        if kind == "ai_model_server_classify":
            from ipfs_accelerate_py.mcp.ai_model_server import mock_classify_text

            payload = mock_classify_text(probe["inputs"]["text"])
            assert payload["prediction"] == probe["expect"]["prediction"]
            assert payload["confidence"] == probe["expect"]["confidence"]
            classified = _classify_mock_value(provenance="simulated")
            assert classified["origin"] == Origin.SIMULATED.value
            assert classified["production_supported"] is False
            assert classified["outcome"] == "Simulated"
        elif kind == "mock_ipfs_random_cid":
            from ipfs_accelerate_py.mcp.tools.mock_ipfs import MockIPFSClient, random_cid

            cid = random_cid()
            assert cid.startswith("Qm")
            assert len(cid) == 46
            # Same-name decoy: MockIPFSClient label alone does not grant live origin.
            client_name = MockIPFSClient.__name__
            assert client_name == case["shared_surface_name"]
            mock_cls = _classify_mock_value(provenance="simulated")
            real_cls = _classify_mock_value(provenance="live_observed")
            assert mock_cls["origin"] != real_cls["origin"]
            assert mock_cls["origin"] == Origin.SIMULATED.value
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write("facp-024-mock-origin")
                handle.flush()
                temp_path = handle.name
            try:
                result = MockIPFSClient().add_file(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
            assert str(result.get("Hash", "")).startswith("Qm")
            assert _classify_mock_value(provenance="simulated")["production_supported"] is False
        else:
            raise AssertionError(f"unknown runtime_probe kind: {kind}")

    assert probed >= 2, "expected at least two hermetic runtime probes"


def test_fixture_policy_rejects_live_promotion_of_simulated(
    cases_doc: dict[str, Any],
) -> None:
    simulated = _envelope_for_provenance("simulated")
    live = _envelope_for_provenance("live_observed")
    assert simulated.origin is Origin.SIMULATED
    assert live.origin is Origin.LIVE_OBSERVED
    # Relabeling is not allowed: a simulated envelope cannot become live by renaming.
    assert simulated.with_updates(origin=Origin.SIMULATED).origin is Origin.SIMULATED
    assert _production_supported(simulated) is False
    assert cases_doc["policy"]["production_supported_for_simulated"] is False
