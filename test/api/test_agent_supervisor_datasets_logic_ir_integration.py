"""DCR-030: normalize observed contracts into real datasets logic IR.

Acceptance
----------
* No fixture-derived or bridge-only artifact substitutes for required
  production input.
* Every normalized row binds original bytes and forest CID.
* Evidence includes input roots, adapter versions, module origins, family
  availability, and normalization diagnostics.
* Observed AST, contract graph, KG, UI IR, SecurityIR, and deterministic
  vector evidence inject into the datasets provider registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_integration import (
    CONTRACT_VERSION,
    DATASETS_LOGIC_FACADE_INTERFACE,
    DCR_ARTIFACT_PATH,
    DCR_TASK_ID,
    DatasetsLogicFacade,
    DatasetsProviderRegistry,
    EvidenceFamily,
    FACADE_ADAPTER_VERSION,
    FamilyAvailability,
    INJECTED_EVIDENCE_FAMILIES,
    IRIntegrationError,
    IR_INPUT_ENVELOPE_INTERFACE,
    IR_INPUT_ENVELOPE_SCHEMA,
    IR_NORMALIZATION_EVIDENCE_TERM,
    IR_NORMALIZATION_RESULT_INTERFACE,
    InputAuthority,
    NORMALIZED_IR_ROW_INTERFACE,
    NORMALIZED_IR_ROW_SCHEMA,
    NormalizedIRRow,
    ProductionInputSubstitutionError,
    REQUIRED_PRODUCTION_FAMILIES,
    build_envelope_from_bytes,
    canonical_ir_cid,
    collect_production_envelopes,
    load_ir_input,
    materialize_ir_input,
    normalize_contract_evidence,
    probe_family_availability,
    write_ir_input,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (
            candidate / "config" / "deterministic_contract_repair_services.json"
        ).is_file() or (
            candidate
            / "data/agent_supervisor/deterministic_contract_repair/forest.json"
        ).is_file():
            return candidate
    return here.parents[4]


_REPO_ROOT = _repo_root()
_FOREST = (
    _REPO_ROOT
    / "data/agent_supervisor/deterministic_contract_repair/forest.json"
)
_GRAPH = (
    _REPO_ROOT
    / "data/agent_supervisor/deterministic_contract_repair/mcp_contract_graph.json"
)
_FINDINGS = (
    _REPO_ROOT
    / "data/agent_supervisor/deterministic_contract_repair/"
    "mcp_contract_mismatch_findings.json"
)
_TRANSCRIPT = (
    _REPO_ROOT
    / "data/agent_supervisor/deterministic_contract_repair/mcp-live-transcript.json"
)


def _production_available() -> bool:
    return all(path.is_file() for path in (_FOREST, _GRAPH, _FINDINGS, _TRANSCRIPT))


def _forest_cid() -> str:
    payload = json.loads(_FOREST.read_text(encoding="utf-8"))
    for key in ("forest_id", "forest_cid"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    local = payload.get("local") or {}
    if isinstance(local, dict):
        value = local.get("forest_id") or local.get("portable_forest_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = _FOREST.read_bytes()
    return content_identity(
        {
            "profile": "dcr-original-bytes-v1",
            "digest": "sha256:"
            + __import__("hashlib").sha256(raw).hexdigest(),
            "byte_length": len(raw),
        }
    )


def _synthetic_production_envelopes(
    forest_cid: str = "sha256:" + "ab" * 32,
) -> tuple[IRInputEnvelope, ...]:
    """Build a complete production envelope set with retained original bytes.

    Used when hermetic unit tests need full coverage without depending on
    workspace DCR artifacts.  Authority remains ``production`` with real
    retained bytes — not fixture or bridge-only.
    """

    families = (
        EvidenceFamily.FOREST.value,
        EvidenceFamily.CONTRACT_GRAPH.value,
        EvidenceFamily.MISMATCH_FINDINGS.value,
        EvidenceFamily.LIVE_TRANSCRIPT.value,
        EvidenceFamily.OBSERVED_AST.value,
        EvidenceFamily.KNOWLEDGE_GRAPH.value,
        EvidenceFamily.UI_IR.value,
        EvidenceFamily.SECURITY_IR.value,
        EvidenceFamily.DETERMINISTIC_VECTOR.value,
    )
    envelopes: list[IRInputEnvelope] = []
    for family in families:
        payload = {
            "family": family,
            "epoch": forest_cid,
            "marker": f"production-{family}",
            "values": [1, 2, 3],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        envelopes.append(
            build_envelope_from_bytes(
                family=family,
                input_root=f"synthetic/{family}.json",
                original_bytes=raw,
                forest_cid=forest_cid,
                authority=InputAuthority.PRODUCTION,
                module_origin="ipfs_datasets_py.logic.ir_core",
                source_path=f"synthetic/{family}.json",
                projection={"family": family, "epoch": forest_cid},
            )
        )
    return tuple(envelopes)


# ---------------------------------------------------------------------------
# Interfaces / symbols
# ---------------------------------------------------------------------------


def test_interfaces_and_symbols() -> None:
    assert DATASETS_LOGIC_FACADE_INTERFACE == "DatasetsLogicFacade@1"
    assert IR_INPUT_ENVELOPE_INTERFACE == "IRInputEnvelope@1"
    assert IR_NORMALIZATION_RESULT_INTERFACE == "IRNormalizationResult@1"
    assert NORMALIZED_IR_ROW_INTERFACE == "NormalizedIRRow@1"
    assert IR_NORMALIZATION_EVIDENCE_TERM == "dcr/ir-normalization@1"
    assert CONTRACT_VERSION == 1
    assert DCR_TASK_ID == "DCR-030"
    assert DCR_ARTIFACT_PATH.endswith("ir-input.json")
    assert callable(normalize_contract_evidence)
    assert callable(DatasetsLogicFacade)
    assert set(INJECTED_EVIDENCE_FAMILIES) == {
        "observed_ast",
        "contract_graph",
        "knowledge_graph",
        "ui_ir",
        "security_ir",
        "deterministic_vector",
    }
    assert "forest" in REQUIRED_PRODUCTION_FAMILIES
    assert "contract_graph" in REQUIRED_PRODUCTION_FAMILIES


# ---------------------------------------------------------------------------
# Row binding invariants
# ---------------------------------------------------------------------------


def test_every_normalized_row_binds_original_bytes_and_forest_cid() -> None:
    forest = "sha256:" + "cd" * 32
    result = normalize_contract_evidence(
        _synthetic_production_envelopes(forest),
        require_production=True,
        inject=True,
    )
    assert result.forest_cid == forest
    assert result.rows
    assert result.verifies_cid() is True
    assert result.result_cid == canonical_ir_cid(result._root_payload())
    assert result.canonical_digest.startswith("sha256:")
    assert result.result_cid != result.canonical_digest
    assert result.authoritative is False
    assert result.completion_authorized is False
    assert result.model_calls == 0
    assert result.evidence_term == IR_NORMALIZATION_EVIDENCE_TERM

    for row in result.rows:
        assert row.original_bytes_digest.startswith("sha256:")
        assert row.original_bytes_cid
        assert not row.original_bytes_cid.startswith("sha256:")
        assert row.forest_cid == forest
        assert row.adapter_version == FACADE_ADAPTER_VERSION
        assert row.input_root
        assert row.schema == NORMALIZED_IR_ROW_SCHEMA
        assert row.interface == NORMALIZED_IR_ROW_INTERFACE
        assert row.row_id == content_identity(row._identity_payload())


def test_envelope_rejects_empty_original_bytes() -> None:
    with pytest.raises(IRIntegrationError, match="original_bytes"):
        build_envelope_from_bytes(
            family="contract_graph",
            input_root="x",
            original_bytes=b"",
            forest_cid="sha256:" + "11" * 32,
        )


def test_row_rejects_digest_as_cid() -> None:
    with pytest.raises(IRIntegrationError, match="pseudo-CID"):
        NormalizedIRRow(
            family="contract_graph",
            input_root="root",
            original_bytes_digest="sha256:" + "aa" * 32,
            original_bytes_cid="sha256:" + "bb" * 32,
            forest_cid="sha256:" + "cc" * 32,
            adapter_version=FACADE_ADAPTER_VERSION,
            module_origin="m",
            family_available=True,
            authority=InputAuthority.PRODUCTION,
        )


# ---------------------------------------------------------------------------
# Fixture / bridge-only substitution rejection
# ---------------------------------------------------------------------------


def test_fixture_cannot_substitute_for_required_production_input() -> None:
    forest = "sha256:" + "ef" * 32
    production = list(_synthetic_production_envelopes(forest))
    # Replace contract_graph with a fixture-authority envelope.
    fixture_raw = b'{"fixture":true,"family":"contract_graph"}'
    production = [
        env
        if env.family != "contract_graph"
        else build_envelope_from_bytes(
            family="contract_graph",
            input_root="fixtures/contract_graph.json",
            original_bytes=fixture_raw,
            forest_cid=forest,
            authority=InputAuthority.FIXTURE,
            module_origin="tests.fixtures",
        )
        for env in production
    ]
    with pytest.raises(ProductionInputSubstitutionError) as excinfo:
        normalize_contract_evidence(
            production,
            require_production=True,
        )
    assert excinfo.value.reason_code == "production_input_substitution_forbidden"
    assert excinfo.value.details["family"] == "contract_graph"
    assert excinfo.value.details["authority"] == "fixture"


def test_bridge_only_cannot_substitute_for_required_production_input() -> None:
    forest = "sha256:" + "12" * 32
    production = list(_synthetic_production_envelopes(forest))
    production = [
        env
        if env.family != "security_ir"
        else build_envelope_from_bytes(
            family="security_ir",
            input_root="bridge/security",
            original_bytes=b'{"bridge":true}',
            forest_cid=forest,
            authority=InputAuthority.BRIDGE_ONLY,
            module_origin="bridge.only",
        )
        for env in production
    ]
    with pytest.raises(ProductionInputSubstitutionError) as excinfo:
        normalize_contract_evidence(production, require_production=True)
    assert excinfo.value.details["authority"] == "bridge_only"
    assert excinfo.value.details["family"] == "security_ir"


def test_fixture_allowed_when_require_production_false() -> None:
    forest = "sha256:" + "34" * 32
    raw = b'{"fixture":true}'
    envelope = build_envelope_from_bytes(
        family="contract_graph",
        input_root="fixtures/graph.json",
        original_bytes=raw,
        forest_cid=forest,
        authority=InputAuthority.FIXTURE,
    )
    result = normalize_contract_evidence(
        (envelope,),
        require_production=False,
        inject=False,
    )
    assert len(result.rows) == 1
    assert result.rows[0].authority is InputAuthority.FIXTURE
    assert result.rows[0].original_bytes_digest.startswith("sha256:")
    assert result.rows[0].forest_cid == forest


def test_missing_required_production_family_fails_closed() -> None:
    forest = "sha256:" + "56" * 32
    # Only forest — missing the rest of REQUIRED_PRODUCTION_FAMILIES.
    envelope = build_envelope_from_bytes(
        family="forest",
        input_root="forest.json",
        original_bytes=b'{"forest":true}',
        forest_cid=forest,
        authority=InputAuthority.PRODUCTION,
    )
    with pytest.raises(IRIntegrationError) as excinfo:
        normalize_contract_evidence((envelope,), require_production=True)
    assert excinfo.value.reason_code == "missing_production_input"
    assert "contract_graph" in excinfo.value.details["missing_families"]


def test_mixed_forest_cids_fail_closed() -> None:
    a = build_envelope_from_bytes(
        family="forest",
        input_root="a",
        original_bytes=b'{"a":1}',
        forest_cid="sha256:" + "aa" * 32,
    )
    b = build_envelope_from_bytes(
        family="contract_graph",
        input_root="b",
        original_bytes=b'{"b":1}',
        forest_cid="sha256:" + "bb" * 32,
    )
    with pytest.raises(IRIntegrationError) as excinfo:
        normalize_contract_evidence(
            (a, b),
            require_production=False,
        )
    assert excinfo.value.reason_code == "forest_cid_mismatch"


# ---------------------------------------------------------------------------
# Evidence subset / diagnostics / registry injection
# ---------------------------------------------------------------------------


def test_evidence_subset_and_registry_injection() -> None:
    forest = "sha256:" + "78" * 32
    registry = DatasetsProviderRegistry()
    result = normalize_contract_evidence(
        _synthetic_production_envelopes(forest),
        registry=registry,
        require_production=True,
        inject=True,
    )
    assert result.input_roots
    assert result.adapter_versions
    assert FACADE_ADAPTER_VERSION in result.adapter_versions
    assert result.module_origins
    assert result.family_availability
    assert all(isinstance(item, FamilyAvailability) for item in result.family_availability)
    assert result.diagnostics

    injected_families = {entry.family for entry in result.registry_entries}
    assert injected_families == set(INJECTED_EVIDENCE_FAMILIES)
    assert set(registry.families()) == set(INJECTED_EVIDENCE_FAMILIES)
    for entry in registry.entries():
        assert entry.original_bytes_cid
        assert entry.forest_cid == forest
        assert entry.row_id
        assert entry.adapter_version
    assert registry.to_dict()["grants_execution_authority"] is False
    assert registry.to_dict()["grants_proof_authority"] is False


def test_facade_capability_receipt_is_non_authoritative() -> None:
    facade = DatasetsLogicFacade()
    receipt = facade.capability_receipt()
    assert receipt["interface"] == DATASETS_LOGIC_FACADE_INTERFACE
    assert receipt["authoritative"] is False
    assert receipt["completion_authorized"] is False
    assert receipt["grants_execution_authority"] is False
    assert receipt["grants_proof_authority"] is False
    assert receipt["model_calls"] == 0
    assert "family_availability" in receipt
    families = facade.discover_families()
    assert families
    assert all(item.module for item in families)


def test_facade_normalize_round_trip(tmp_path: Path) -> None:
    forest = "sha256:" + "9a" * 32
    facade = DatasetsLogicFacade(registry=DatasetsProviderRegistry())
    result = facade.normalize(
        _synthetic_production_envelopes(forest),
        require_production=True,
        inject=True,
    )
    assert facade.last_result is result
    assert result.verifies_cid()

    out = tmp_path / "ir-input.json"
    written = write_ir_input(out, result=result)
    assert written == out
    loaded = load_ir_input(out)
    assert loaded.result_cid == result.result_cid
    assert loaded.forest_cid == forest
    assert len(loaded.rows) == len(result.rows)
    for row in loaded.rows:
        assert row.original_bytes_digest
        assert row.original_bytes_cid
        assert row.forest_cid == forest


def test_artifact_bytes_bound() -> None:
    forest = "sha256:" + "bc" * 32
    result = normalize_contract_evidence(
        _synthetic_production_envelopes(forest),
        require_production=True,
    )
    raw = result.to_artifact_bytes()
    assert len(raw) < 1_048_576
    payload = json.loads(raw.decode("utf-8"))
    assert payload["schema"].endswith("ir-integration-artifact@1")
    assert payload["task_id"] == DCR_TASK_ID
    assert payload["authoritative"] is False
    assert payload["normalization"]["result_cid"] == result.result_cid


# ---------------------------------------------------------------------------
# Production path against committed DCR artifacts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _production_available(),
    reason="committed DCR production artifacts not present in this checkout",
)
def test_production_materialization_binds_forest_and_bytes() -> None:
    forest = _forest_cid()
    envelopes = collect_production_envelopes(repo_root=_REPO_ROOT)
    families = {env.family for env in envelopes}
    for required in REQUIRED_PRODUCTION_FAMILIES:
        assert required in families, f"missing production family {required}"
    for env in envelopes:
        assert env.authority is InputAuthority.PRODUCTION
        assert env.original_bytes
        assert env.forest_cid == forest
        assert env.schema == IR_INPUT_ENVELOPE_SCHEMA

    result = materialize_ir_input(repo_root=_REPO_ROOT)
    assert result.forest_cid == forest
    assert result.verifies_cid() is True
    assert result.evidence_term == IR_NORMALIZATION_EVIDENCE_TERM
    assert result.input_roots
    assert result.adapter_versions
    assert result.module_origins
    assert result.family_availability
    assert result.diagnostics
    assert result.model_calls == 0
    assert result.authoritative is False

    for row in result.rows:
        assert row.original_bytes_digest.startswith("sha256:")
        assert row.original_bytes_cid
        assert row.forest_cid == forest

    injected = {entry.family for entry in result.registry_entries}
    assert injected == set(INJECTED_EVIDENCE_FAMILIES)

    # Graph original bytes digest must match the committed file.
    graph_raw = _GRAPH.read_bytes()
    graph_rows = [row for row in result.rows if row.family == "contract_graph"]
    assert graph_rows
    import hashlib

    expected_digest = "sha256:" + hashlib.sha256(graph_raw).hexdigest()
    assert graph_rows[0].original_bytes_digest == expected_digest


@pytest.mark.skipif(
    not _production_available(),
    reason="committed DCR production artifacts not present in this checkout",
)
def test_production_artifact_write_and_load(tmp_path: Path) -> None:
    result = materialize_ir_input(repo_root=_REPO_ROOT)
    out = tmp_path / "nested" / "ir-input.json"
    write_ir_input(out, result=result, repo_root=_REPO_ROOT)
    loaded = load_ir_input(out)
    assert loaded.result_cid == result.result_cid
    assert loaded.verifies_cid() is True
    assert all(row.forest_cid == result.forest_cid for row in loaded.rows)


@pytest.mark.skipif(
    not _production_available(),
    reason="committed DCR production artifacts not present in this checkout",
)
def test_facade_end_to_end_production() -> None:
    facade = DatasetsLogicFacade(repo_root=_REPO_ROOT)
    result = facade.normalize(require_production=True, inject=True)
    assert result.forest_cid == _forest_cid()
    assert set(facade.registry.families()) == set(INJECTED_EVIDENCE_FAMILIES)
    receipt = facade.capability_receipt()
    assert receipt["provider_id"] == facade.provider_id
    assert facade.to_dict()["last_result_cid"] == result.result_cid


def test_probe_family_availability_is_side_effect_free() -> None:
    first = probe_family_availability()
    second = probe_family_availability()
    assert [item.to_dict() for item in first] == [
        item.to_dict() for item in second
    ]
    assert all(item.family and item.module for item in first)


def test_artifact_round_trip_and_cid_tamper_detection(tmp_path: Path) -> None:
    """Written artifacts reload with the same CID; tampered CIDs fail closed."""

    forest = "sha256:" + "de" * 32
    result = normalize_contract_evidence(
        _synthetic_production_envelopes(forest),
        require_production=True,
    )
    out = tmp_path / "ir-input.json"
    write_ir_input(out, result=result)

    loaded = load_ir_input(out)
    assert loaded.result_cid == result.result_cid
    assert all(row.forest_cid == forest for row in loaded.rows)

    # Tampering the stored result_cid is not enough to pass if body drifts —
    # drop a required binding from a row and recompute nothing.
    corrupt = json.loads(out.read_text(encoding="utf-8"))
    corrupt["normalization"]["rows"][0]["forest_cid"] = "sha256:" + "00" * 32
    out.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(IRIntegrationError) as excinfo:
        load_ir_input(out)
    assert excinfo.value.reason_code in {
        "forest_cid_mismatch",
        "cid_reconstruction_failed",
    }
