"""PCCE-022: closed runtime modes and fail-closed admission/promotion policy."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from ipfs_accelerate_py.proof_context.compatibility import FROZEN_MATRIX
from ipfs_accelerate_py.proof_context.policy import (
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_VERSION,
    DECISION_TABLE,
    ERRORS,
    ERROR_TAXONOMY_CONTENT_ID,
    FORBIDDEN_EVIDENCE,
    LIVE_MODES,
    MODES,
    PCCE_006_CONTENT_ID,
    POLICY,
    POLICY_CID,
    POLICY_DESCRIPTOR,
    POLICY_RESULT_SCHEMA,
    POLICY_SCHEMA,
    PROVENANCES,
    QUALITY_CLASSES,
    SCHEMA,
    SIMULATION_WATERMARK,
    STATUSES,
    STATUS_TAXONOMY_CONTENT_ID,
    PolicyError,
    PolicyResult,
    admit_cid,
    admit_evidence,
    admit_mode,
    apply_simulation_watermark,
    decision_for,
    decision_table,
    environment_promotion_keys,
    evaluation_quality_claims,
    frozen_taxonomy,
    inspect_simulation_watermark,
    mint_policy_cid,
    policy_cid,
    policy_descriptor,
    promote,
    require_admitted,
)

VALID_CID = "bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4"
SEAL_CID = "b" + "c" * 58


def _evidence(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provenance": "live",
        "status": "succeeded",
        "artifact_cid": VALID_CID,
        "seal_cid": SEAL_CID,
        "sealed": True,
        "signature": "sig-v0.1",
        "signature_required": True,
        "self_approved": False,
        "parents": [],
    }
    payload.update(overrides)
    return payload


def _assert_four_modes(result: PolicyResult | Mapping[str, object]) -> None:
    payload = result.to_mapping() if isinstance(result, PolicyResult) else result
    assert tuple(payload["closed_modes"]) == MODES
    assert set(payload["closed_modes"]) == {"production", "supervised", "evaluation", "simulation"}
    assert payload["mode"] in payload["closed_modes"]
    assert tuple(payload["forbidden_evidence"]) == FORBIDDEN_EVIDENCE
    per_mode = payload["per_mode"]
    assert tuple(per_mode) == MODES
    for mode in MODES:
        snapshot = per_mode[mode]
        assert snapshot["mode"] == mode
        assert "admitted" in snapshot
        assert "accepted" in snapshot
        assert snapshot["quality_class"] in QUALITY_CLASSES
        assert snapshot["promotion_admitted"] is False


def test_closed_modes_and_taxonomies_are_frozen() -> None:
    assert MODES == ("production", "supervised", "evaluation", "simulation")
    assert LIVE_MODES == frozenset({"production", "supervised"})
    assert PROVENANCES == ("live", "replayed", "simulated")
    assert FORBIDDEN_EVIDENCE == (
        "simulated",
        "replayed",
        "stale",
        "invalid",
        "unavailable",
        "pseudo-cid",
        "unsigned-required",
        "unsealed",
    )
    assert STATUSES == (
        "succeeded",
        "rejected",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "context_insufficient",
        "model_escalation_required",
        "human_review_required",
        "unavailable",
        "timeout",
        "cancelled",
        "invalid",
        "stale",
        "simulated",
        "infrastructure_failure",
        "partial_effect",
        "repair_required",
    )
    assert "simulated_promoted" in ERRORS
    assert "pseudo_cid" in ERRORS
    assert SCHEMA.endswith("v0.1")
    assert POLICY_SCHEMA.endswith("/policy")
    assert CONTRACT_VERSION == "0.1"
    with pytest.raises(TypeError):
        MODES[0] = "shadow"  # type: ignore[index]
    with pytest.raises(TypeError):
        POLICY["modes"] = ("shadow",)  # type: ignore[index]
    with pytest.raises(TypeError):
        POLICY_DESCRIPTOR["cid"] = "mutated"  # type: ignore[index]


def test_unknown_mode_provenance_and_status_fail_closed() -> None:
    with pytest.raises(PolicyError) as exc:
        admit_mode("shadow")
    assert exc.value.reason == "unknown_field"
    with pytest.raises(PolicyError) as exc:
        admit_evidence("production", _evidence(provenance="recorded"))
    assert exc.value.reason == "unknown_field"
    with pytest.raises(PolicyError) as exc:
        admit_evidence("production", _evidence(status="passed_anyway"))
    assert exc.value.reason == "unknown_field"
    with pytest.raises(PolicyError) as exc:
        admit_evidence("debug", _evidence())
    assert exc.value.reason == "unknown_field"


def test_every_result_enumerates_all_four_modes() -> None:
    results = [
        admit_evidence("production", _evidence()),
        admit_evidence("supervised", _evidence()),
        admit_evidence("evaluation", _evidence(provenance="replayed")),
        admit_evidence("simulation", _evidence()),
        admit_evidence("production", _evidence(provenance="simulated")),
        promote(_evidence(), source_mode="simulation", target_mode="production"),
    ]
    for result in results:
        _assert_four_modes(result)
        assert result.policy_cid == POLICY_CID
        assert result.schema == POLICY_RESULT_SCHEMA


@pytest.mark.parametrize("mode", ("production", "supervised"))
@pytest.mark.parametrize(
    ("override", "defect"),
    (
        ({"provenance": "simulated"}, "simulated"),
        ({"status": "simulated"}, "simulated"),
        ({"provenance": "replayed"}, "replayed"),
        ({"status": "stale"}, "stale"),
        ({"stale": True}, "stale"),
        ({"status": "invalid"}, "invalid"),
        ({"status": "unavailable"}, "unavailable"),
        ({"available": False}, "unavailable"),
        ({"artifact_cid": "sha256:deadbeef"}, "pseudo-cid"),
        ({"artifact_cid": "QmNotACid"}, "pseudo-cid"),
        ({"artifact_cid": "urn:example:1"}, "pseudo-cid"),
        ({"signature": None, "signed": False, "signature_required": True}, "unsigned-required"),
        ({"sealed": False, "seal_cid": None}, "unsealed"),
        ({"unsealed": True}, "unsealed"),
    ),
)
def test_production_and_supervised_reject_forbidden_evidence(
    mode: str,
    override: dict[str, object],
    defect: str,
) -> None:
    result = admit_evidence(mode, _evidence(**override))
    _assert_four_modes(result)
    assert result.admitted is False
    assert result.accepted is False
    assert defect in result.reasons or result.quality_class in {"simulated", "replayed"}
    assert result.per_mode["production"]["accepted"] is False
    assert result.per_mode["supervised"]["accepted"] is False
    if defect == "simulated":
        assert result.error == "simulated_promoted"
        assert result.status == "simulated"
        assert result.watermark == SIMULATION_WATERMARK
    if defect == "replayed":
        assert result.quality_class == "replayed"
        assert result.error == "boundary_violation"


@pytest.mark.parametrize("mode", ("production", "supervised"))
def test_live_sealed_signed_evidence_is_accepted(mode: str) -> None:
    result = admit_evidence(mode, _evidence())
    _assert_four_modes(result)
    assert result.admitted is True
    assert result.accepted is True
    assert result.quality_class == "live"
    assert result.status == "succeeded"
    assert result.error is None
    assert result.watermark is None
    assert result.per_mode[mode]["accepted"] is True


def test_evaluation_separates_quality_claims_by_provenance() -> None:
    live = admit_evidence("evaluation", _evidence())
    replayed = admit_evidence("evaluation", _evidence(provenance="replayed"))
    simulated = admit_evidence("evaluation", _evidence(provenance="simulated"))
    for result in (live, replayed, simulated):
        _assert_four_modes(result)
        assert result.admitted is True
        assert result.accepted is False
        assert result.quality_claim == result.quality_class
    assert live.quality_class == "live"
    assert replayed.quality_class == "replayed"
    assert simulated.quality_class == "simulated"
    assert simulated.watermark == SIMULATION_WATERMARK
    assert simulated.status == "simulated"
    claims = evaluation_quality_claims(_evidence(provenance="replayed"))
    assert tuple(claims) == QUALITY_CLASSES
    assert claims["replayed"] is not None
    assert claims["live"] is None
    assert claims["simulated"] is None


def test_evaluation_rejects_replay_as_live_quality() -> None:
    with pytest.raises(PolicyError) as exc:
        admit_evidence(
            "evaluation",
            _evidence(provenance="replayed", quality_claim="live"),
        )
    assert exc.value.reason == "identity_inconsistent"
    mixed = admit_evidence(
        "evaluation",
        _evidence(provenance="live", parents=[{"provenance": "replayed"}]),
    )
    _assert_four_modes(mixed)
    assert mixed.quality_class == "replayed"
    assert mixed.quality_claim == "replayed"
    assert mixed.per_mode["production"]["admitted"] is False
    with pytest.raises(PolicyError) as merged:
        evaluation_quality_claims({"quality": 0.9, "provenance": "live", "status": "succeeded"})
    assert merged.value.reason == "boundary_violation"
    with pytest.raises(PolicyError) as aggregated:
        evaluation_quality_claims(
            {
                "provenance": "live",
                "status": "succeeded",
                "quality_claims": {"live": 1, "replayed": 1, "merged": 2},
            }
        )
    assert aggregated.value.reason == "boundary_violation"


def test_simulation_is_watermarked_transitively() -> None:
    child = admit_evidence("simulation", _evidence())
    _assert_four_modes(child)
    assert child.watermark == SIMULATION_WATERMARK
    assert child.provenance == "simulated"
    assert child.quality_class == "simulated"
    assert child.status == "simulated"
    assert child.accepted is False
    assert child.per_mode["production"]["admitted"] is False
    parent = apply_simulation_watermark(_evidence())
    assert parent["watermark"] == SIMULATION_WATERMARK
    assert parent["provenance"] == "simulated"
    assert parent["status"] == "simulated"
    nested = admit_evidence(
        "evaluation",
        _evidence(parents=[dict(parent), {"provenance": "live", "status": "succeeded"}]),
    )
    _assert_four_modes(nested)
    assert inspect_simulation_watermark(nested.to_mapping()) is True
    assert nested.watermark == SIMULATION_WATERMARK
    assert nested.quality_class == "simulated"
    assert nested.per_mode["production"]["accepted"] is False
    assert nested.per_mode["supervised"]["admitted"] is False


def test_simulation_has_no_direct_promotion_path() -> None:
    simulated = _evidence(provenance="simulated")
    for target in MODES:
        result = promote(simulated, source_mode="simulation", target_mode=target)
        _assert_four_modes(result)
        assert result.promotion_admitted is False
        assert result.accepted is False
        assert result.error == "simulated_promoted"
    live = _evidence()
    for target in ("production", "supervised", "evaluation"):
        result = promote(live, source_mode="simulation", target_mode=target)
        _assert_four_modes(result)
        assert result.promotion_admitted is False
        assert result.admitted is False
        assert result.error == "simulated_promoted"
    cross = promote(live, source_mode="evaluation", target_mode="production")
    _assert_four_modes(cross)
    assert cross.promotion_admitted is False
    assert cross.admitted is False
    assert cross.error == "boundary_violation"
    identity = promote(live, source_mode="production", target_mode="production")
    _assert_four_modes(identity)
    assert identity.promotion_admitted is False
    assert identity.admitted is True
    assert identity.accepted is True


def test_environment_variables_cannot_promote_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in environment_promotion_keys():
        monkeypatch.setenv(key, "production")
    monkeypatch.setenv("PCCE_MODE", "production")
    result = admit_evidence("simulation", _evidence())
    _assert_four_modes(result)
    assert result.mode == "simulation"
    assert result.accepted is False
    assert result.watermark == SIMULATION_WATERMARK
    assert result.per_mode["production"]["admitted"] is False
    promoted = promote(
        _evidence(provenance="simulated"),
        source_mode="simulation",
        target_mode="production",
    )
    assert promoted.admitted is False
    assert promoted.promotion_admitted is False
    source = Path(inspect.getfile(admit_evidence)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr in {"getenv", "environ"}:
            names.add(node.attr)
    assert "os.environ" not in names
    assert "getenv" not in names
    assert "os.getenv" not in source
    for key in environment_promotion_keys():
        assert key in source


def test_adapter_cannot_approve_its_own_patch() -> None:
    for mode in MODES:
        self_approved = admit_evidence(mode, _evidence(self_approved=True))
        _assert_four_modes(self_approved)
        assert self_approved.admitted is False
        assert self_approved.accepted is False
        assert self_approved.error == "boundary_violation"
        same_actor = admit_evidence(
            mode,
            _evidence(adapter_id="adapter-1", approver_id="adapter-1"),
        )
        assert same_actor.admitted is False
        assert same_actor.error == "boundary_violation"


def test_decision_table_is_complete_and_matches_admission() -> None:
    table = decision_table()
    assert table is DECISION_TABLE
    expected = {
        (mode, provenance, evidence_class)
        for mode in MODES
        for provenance in PROVENANCES
        for evidence_class in ("clean", *FORBIDDEN_EVIDENCE)
    }
    observed = {(row["mode"], row["provenance"], row["evidence_class"]) for row in table}
    assert observed == expected
    assert len(table) == 4 * 3 * 9
    for row in table:
        assert row["promotion_admitted"] is False
        assert row["mode"] in MODES
        if row["mode"] in LIVE_MODES:
            if row["evidence_class"] != "clean" or row["provenance"] != "live":
                assert row["admitted"] is False
                assert row["accepted"] is False
            else:
                assert row["admitted"] is True
                assert row["accepted"] is True
        if row["mode"] == "simulation":
            assert row["accepted"] is False
            assert row["quality_class"] == "simulated"
        if row["mode"] == "evaluation" and row["evidence_class"] == "clean":
            assert row["admitted"] is True
            assert row["accepted"] is False
            assert row["quality_class"] == (
                "simulated" if row["provenance"] == "simulated" else row["provenance"]
            )
        reconstructed = decision_for(row["mode"], row["provenance"], row["evidence_class"])
        assert reconstructed["admitted"] is row["admitted"]
        assert reconstructed["accepted"] is row["accepted"]


def test_decision_table_negative_fixtures_align_with_admit_evidence() -> None:
    fixtures = {
        "clean": _evidence(),
        "simulated": _evidence(provenance="simulated"),
        "replayed": _evidence(provenance="replayed"),
        "stale": _evidence(status="stale"),
        "invalid": _evidence(status="invalid"),
        "unavailable": _evidence(status="unavailable"),
        "pseudo-cid": _evidence(artifact_cid="sha256:deadbeef"),
        "unsigned-required": _evidence(signature=None, signed=False, signature_required=True),
        "unsealed": _evidence(sealed=False, seal_cid=None),
    }
    for mode in MODES:
        for evidence_class, payload in fixtures.items():
            provenance = str(payload.get("provenance", "live"))
            row = decision_for(mode, provenance, evidence_class)
            result = admit_evidence(mode, payload)
            _assert_four_modes(result)
            assert result.admitted is row["admitted"]
            assert result.accepted is row["accepted"]
            if mode in LIVE_MODES and evidence_class in FORBIDDEN_EVIDENCE:
                assert result.admitted is False
                assert result.accepted is False


def test_policy_descriptor_cid_and_frozen_taxonomy() -> None:
    descriptor = policy_descriptor()
    assert descriptor is POLICY
    assert descriptor["schema"] == POLICY_SCHEMA
    assert descriptor["cid"] == POLICY_CID
    assert policy_cid() == POLICY_CID
    assert POLICY_CID.startswith("b")
    admit_cid(POLICY_CID)
    body = {key: value for key, value in descriptor.items() if key != "cid"}
    assert mint_policy_cid(body) == POLICY_CID
    taxonomy = frozen_taxonomy()
    assert taxonomy["pcce_006_content_id"] == PCCE_006_CONTENT_ID
    assert taxonomy["pcce_006_content_id"] == (
        "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
    )
    assert taxonomy["compatibility_matrix_content_id"] == COMPATIBILITY_MATRIX_CONTENT_ID
    assert taxonomy["compatibility_matrix_content_id"] == FROZEN_MATRIX["content_id"]
    assert taxonomy["status_taxonomy_content_id"] == STATUS_TAXONOMY_CONTENT_ID
    assert taxonomy["error_taxonomy_content_id"] == ERROR_TAXONOMY_CONTENT_ID
    assert taxonomy["statuses"] == STATUSES
    assert taxonomy["errors"] == ERRORS
    assert descriptor["promotion_paths"] == ()
    digest = hashlib.sha256(POLICY_CID.encode("utf-8")).hexdigest()
    assert len(digest) == 64


def test_pseudo_cid_and_mocks_are_rejected_in_live_modes() -> None:
    with pytest.raises(PolicyError) as exc:
        admit_cid("sha256:abc")
    assert exc.value.reason == "pseudo_cid"
    with pytest.raises(PolicyError):
        admit_cid("not-a-cid")
    with pytest.raises(PolicyError) as mock_exc:
        admit_evidence("production", Mock())
    assert mock_exc.value.reason == "boundary_violation"
    with pytest.raises(PolicyError):
        admit_evidence("supervised", Mock())
    evaluation = admit_evidence("evaluation", _evidence())
    assert evaluation.admitted is True


def test_require_admitted_raises_for_forbidden_evidence() -> None:
    ok = require_admitted("production", _evidence())
    assert ok.accepted is True
    with pytest.raises(PolicyError) as exc:
        require_admitted("production", _evidence(sealed=False, seal_cid=None))
    assert exc.value.reason == "boundary_violation"
    with pytest.raises(PolicyError) as simulated:
        require_admitted("supervised", _evidence(provenance="simulated"))
    assert simulated.value.reason == "simulated_promoted"


def test_unavailable_and_invalid_are_never_success() -> None:
    for mode in MODES:
        unavailable = admit_evidence(mode, _evidence(status="unavailable"))
        invalid = admit_evidence(mode, _evidence(status="invalid"))
        _assert_four_modes(unavailable)
        _assert_four_modes(invalid)
        assert unavailable.accepted is False
        assert invalid.accepted is False
        assert unavailable.status != "succeeded"
        assert invalid.status != "succeeded"


def test_mode_mismatch_and_malformed_evidence_fail_closed() -> None:
    with pytest.raises(PolicyError) as exc:
        admit_evidence("production", _evidence(mode="evaluation"))
    assert exc.value.reason == "identity_inconsistent"
    with pytest.raises(PolicyError) as missing:
        admit_evidence("production", None)
    assert missing.value.reason == "malformed"
    with pytest.raises(PolicyError):
        admit_evidence("production", object())


def test_policy_result_rejects_simulated_success_and_non_live_acceptance() -> None:
    with pytest.raises(PolicyError):
        PolicyResult(
            schema=POLICY_RESULT_SCHEMA,
            mode="production",
            closed_modes=MODES,
            provenance="simulated",
            quality_class="simulated",
            status="succeeded",
            admitted=False,
            accepted=False,
            promotion_admitted=False,
            error="simulated_promoted",
            reasons=("simulated",),
            watermark=SIMULATION_WATERMARK,
            policy_cid=POLICY_CID,
            forbidden_evidence=FORBIDDEN_EVIDENCE,
            per_mode={name: {"mode": name, "admitted": False, "accepted": False} for name in MODES},
        )
    with pytest.raises(PolicyError):
        PolicyResult(
            schema=POLICY_RESULT_SCHEMA,
            mode="production",
            closed_modes=MODES,
            provenance="replayed",
            quality_class="replayed",
            status="rejected",
            admitted=True,
            accepted=True,
            promotion_admitted=False,
            error=None,
            reasons=(),
            watermark=None,
            policy_cid=POLICY_CID,
            forbidden_evidence=FORBIDDEN_EVIDENCE,
            per_mode={name: {"mode": name, "admitted": False, "accepted": False} for name in MODES},
        )


def test_cold_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    import ipfs_accelerate_py.proof_context.policy as policy

    after = set(tmp_path.rglob("*"))
    assert after == before
    assert policy.POLICY_CID == POLICY_CID
    assert os.getenv("PCCE_MODE") in {None, os.environ.get("PCCE_MODE")}


def test_mapping_proxy_results_are_immutable() -> None:
    result = admit_evidence("production", _evidence())
    payload = result.to_mapping()
    assert isinstance(payload, MappingProxyType)
    with pytest.raises(TypeError):
        payload["mode"] = "simulation"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.per_mode["production"] = {}  # type: ignore[index]
