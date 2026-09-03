"""PCPR-056: produce Accelerate portfolio compatibility lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    LOCK_KIND,
    OutcomeProbe,
    PCPR_056_GOAL_ID,
    PCPR_056_TASK_ID,
    PINNED_LOCK_CID,
    PORTFOLIO_VERSION,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    SUPPORTED_COMBINATION_ID,
    PortfolioCompatibilityLockError,
    current_head_static_probes,
    pcpr_056_receipt_promotion,
    platform_compatibility_catalog,
    qualify_current_head_portfolio_compatibility_lock,
    qualify_portfolio_compatibility_lock,
    refuse_lock_remint,
    render_declared_lock,
    verify_portfolio_compatibility_lock_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_056_requirements() -> None:
    assert PCPR_056_TASK_ID == "PCPR-056"
    assert PCPR_056_GOAL_ID == "PCPR-G600"
    assert INTERFACE == "AcceleratePortfolioCompatibilityLock@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/portfolio-compatibility-lock@1"
    assert LOCK_KIND == "declared_portfolio_compatibility_lock"
    assert PORTFOLIO_VERSION == "proof-carrying-platform-0.1.0"
    assert SUPPORTED_COMBINATION_ID == (
        "pcpr.v1.python312.accelerate-datasets-kit.shared-contracts-v1"
    )
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_portfolio_compatibility_lock.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    lock = render_declared_lock()
    assert lock["lock_kind"] == LOCK_KIND
    assert lock["lock"] is True
    assert lock["frozen"] is False
    assert lock["live"] is False
    assert lock["published"] is False
    assert lock["release_claim"] is False
    assert lock["closed_release_outcome"] is None
    assert lock["signing"]["signed"] is False
    assert lock["signing"]["invented"] is False
    assert lock["source"]["mutable_main_reference"] is False
    assert lock["supported_combination"]["lock"] is True
    assert lock["lock_cid"] == PINNED_LOCK_CID
    assert refuse_lock_remint(PINNED_LOCK_CID) == PINNED_LOCK_CID
    with pytest.raises(PortfolioCompatibilityLockError, match="remints"):
        refuse_lock_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_portfolio_compatibility_lock()
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
    assert verdict.live_signed_lock is False
    assert verdict.live_signed_lock_evidence_kind == "unavailable"
    assert verdict.mutable_main_reference is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.lock_cid == PINNED_LOCK_CID
    assert verdict.blockers == ()
    section = pcpr_056_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_signed_lock"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_lock_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["lock_files_match_generator"].present is True
    assert probes["pyproject_portfolio_compatibility_lock_table"].present is True
    assert probes["hashes_not_invented"].present is True
    assert probes["signatures_not_invented"].present is True
    assert probes["no_mutable_main_reference"].present is True
    assert probes["lock_kind_is_declared_portfolio_compatibility_lock"].present is True
    assert probes["supported_combination_is_locked"].present is True
    assert probes["pcpr_043_identities_not_reminted"].present is True
    assert probes["accelerate_artifact_cids_match_pins"].present is True
    assert probes["named_git_extras_are_not_this_lock"].present is True
    assert probes["hashes_invented"].present is False
    assert probes["signatures_invented"].present is False
    assert probes["mutable_main_reference"].present is False
    assert probes["compatibility_identities_reminted"].present is False
    assert probes["live_signed_lock"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_portfolio_compatibility_lock_files()
    assert verified["ok"] is True
    assert verified["missing"] == []
    catalog = platform_compatibility_catalog()
    assert catalog["interface"] == "PlatformPortfolioCompatibilityLock@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_signed_lock"] is False
    assert catalog["lock_cid"] == PINNED_LOCK_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_datasets_py"]["binding"]["status"] == "observed"
    assert catalog["components"]["ipfs_kit_py"]["binding"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AcceleratePortfolioCompatibilityLock@1"' in pyproject


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(PortfolioCompatibilityLockError, match="simulated"):
        qualify_portfolio_compatibility_lock(
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
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_live_claim_without_measured_live_evidence_is_rejected() -> None:
    with pytest.raises(PortfolioCompatibilityLockError, match="measured_live"):
        qualify_portfolio_compatibility_lock(
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
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
