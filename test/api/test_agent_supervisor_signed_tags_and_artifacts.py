"""PCPR-055: produce Accelerate signed tags and artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.signed_tags_and_artifacts import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    INTENDED_TAG_NAME,
    OutcomeProbe,
    PCPR_055_GOAL_ID,
    PCPR_055_TASK_ID,
    CHECKSUM_KIND,
    TAG_KIND,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    SignedTagsError,
    current_head_static_probes,
    pcpr_055_receipt_promotion,
    platform_signed_catalog,
    qualify_current_head_signed_tags_and_artifacts,
    qualify_signed_tags_and_artifacts,
    render_declared_checksums,
    render_declared_tag_policy,
    verify_signed_tags_and_artifacts_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_055_requirements() -> None:
    assert PCPR_055_TASK_ID == "PCPR-055"
    assert PCPR_055_GOAL_ID == "PCPR-G600"
    assert INTERFACE == "AccelerateSignedTagsAndArtifacts@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/signed-tags-and-artifacts@1"
    assert TAG_KIND == "declared_signed_tag_policy"
    assert CHECKSUM_KIND == "declared_artifact_checksum_manifest"
    assert INTENDED_TAG_NAME == "ipfs_accelerate_py-v0.0.45"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_signed_tags_and_artifacts.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    policy = render_declared_tag_policy()
    assert policy["tag_kind"] == TAG_KIND
    assert policy["intended_tag_name"] == INTENDED_TAG_NAME
    assert policy["signing"]["signed"] is False
    assert policy["signing"]["invented"] is False
    assert policy["live"] is False
    assert policy["published"] is False
    assert policy["release_claim"] is False
    assert policy["closed_release_outcome"] is None
    assert policy["source"]["mutable_main_reference"] is False
    assert policy["commands"]["git_tag_created"] is False
    checksums = render_declared_checksums(tag_policy=policy)
    assert checksums["checksum_kind"] == CHECKSUM_KIND
    assert checksums["hashes_invented"] is False
    assert checksums["signatures_invented"] is False
    assert checksums["artifacts"]["wheel"]["status"] == "unavailable"
    assert checksums["artifacts"]["signed_tag"]["status"] == "unavailable"
    assert checksums["live"] is False
    assert checksums["release_claim"] is False


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_signed_tags_and_artifacts()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.hashes_invented is False
    assert verdict.signatures_invented is False
    assert verdict.live_signed_tag is False
    assert verdict.live_signed_tag_evidence_kind == "unavailable"
    assert verdict.live_artifact_signature is False
    assert verdict.live_artifact_signature_evidence_kind == "unavailable"
    assert verdict.mutable_main_reference is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.blockers == ()
    section = pcpr_055_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_signed_tag"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_signed_tag_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["tag_policy_files_match_generator"].present is True
    assert probes["checksum_files_match_generator"].present is True
    assert probes["pyproject_signed_tags_and_artifacts_table"].present is True
    assert probes["hashes_not_invented"].present is True
    assert probes["signatures_not_invented"].present is True
    assert probes["no_mutable_main_reference"].present is True
    assert probes["named_git_extras_are_not_signed_release"].present is True
    assert probes["lock_sbom_provenance_checksums_measured"].present is True
    assert probes["hashes_invented"].present is False
    assert probes["signatures_invented"].present is False
    assert probes["mutable_main_reference"].present is False
    assert probes["live_signed_tag"].evidence_kind == "unavailable"
    assert probes["live_artifact_signature"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_signed_tags_and_artifacts_files()
    assert verified["ok"] is True
    assert verified["missing"] == []
    catalog = platform_signed_catalog()
    assert catalog["interface"] == "PlatformSignedTagsAndArtifactsCatalog@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_signed_tag"] is False
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_datasets_py"]["tag_policy"]["status"] == "observed"
    assert catalog["components"]["ipfs_kit_py"]["tag_policy"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateSignedTagsAndArtifacts@1"' in pyproject


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(SignedTagsError, match="simulated"):
        qualify_signed_tags_and_artifacts(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=False,
                    evidence_kind="simulated",
                    live=False,
                    simulated_represented_as_live=True,
                    reason="must fail",
                ),
            ),
            tag_policy_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            checksums_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            sbom_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            provenance_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_live_claim_without_measured_live_evidence_is_rejected() -> None:
    with pytest.raises(SignedTagsError, match="measured_live"):
        qualify_signed_tags_and_artifacts(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=True,
                    evidence_kind="measured",
                    live=True,
                    simulated_represented_as_live=False,
                    reason="must fail",
                ),
            ),
            tag_policy_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            checksums_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            sbom_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            provenance_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
