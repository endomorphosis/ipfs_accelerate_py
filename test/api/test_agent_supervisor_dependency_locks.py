"""PCPR-053: produce Accelerate dependency locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.dependency_locks import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    LOCK_KIND,
    NAMED_GIT_EXTRAS,
    OutcomeProbe,
    PCPR_053_GOAL_ID,
    PCPR_053_TASK_ID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    DependencyLockError,
    current_head_static_probes,
    pcpr_053_receipt_promotion,
    platform_lock_catalog,
    qualify_current_head_dependency_locks,
    qualify_dependency_locks,
    render_release_lock,
    scan_requirement_text,
    verify_lock_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_053_requirements() -> None:
    assert PCPR_053_TASK_ID == "PCPR-053"
    assert PCPR_053_GOAL_ID == "PCPR-G600"
    assert INTERFACE == "AccelerateDependencyLocks@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/dependency-locks@1"
    assert LOCK_KIND == "declared_release_profile"
    assert NAMED_GIT_EXTRAS == (
        "libp2p",
        "mcp-p2p",
        "ipfs-transformers",
        "ipfs-model-manager",
    )
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_dependency_locks.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    document = render_release_lock()
    assert document["mutable_git"] is False
    assert document["editable"] is False
    assert document["path_local"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    assert document["hashes"]["invented"] is False
    assert document["hashes"]["status"] == "unavailable"
    assert document["direct_requirement_count"] == 24
    assert document["requirements_authority"] == "requirements.txt"


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_dependency_locks()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.mutable_git_release_lock is False
    assert verdict.hashes_invented is False
    assert verdict.live_resolver_qualified is False
    assert verdict.live_resolver_evidence_kind == "unavailable"
    assert verdict.hash_lock_evidence_kind == "unavailable"
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.blockers == ()
    section = pcpr_053_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_resolver_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_lock_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["release_lock_has_no_vcs"].present is True
    assert probes["release_lock_has_no_editable"].present is True
    assert probes["release_lock_has_no_path"].present is True
    assert probes["release_lock_has_no_mutable_branch"].present is True
    assert probes["mutable_git_in_release_lock"].present is False
    assert probes["editable_in_release_lock"].present is False
    assert probes["path_in_release_lock"].present is False
    assert probes["hashes_not_invented"].present is True
    assert probes["lock_files_match_generator"].present is True
    assert probes["pyproject_dependency_locks_table"].present is True
    assert probes["requirements_authority_agrees"].present is True
    assert probes["lock_kind_is_declared_release_profile"].present is True
    assert probes["named_git_extras_are_not_release_lock"].present is True
    assert probes["inner_requirements_git_is_not_release_lock"].present is True
    assert probes["simulated_results_represented_as_live"].present is False
    assert probes["hashes_represented_as_live"].present is False
    assert probes["live_resolver_represented_as_live"].present is False
    assert probes["closed_release_represented_as_live"].present is False
    assert probes["live_resolver"].evidence_kind == "unavailable"
    assert probes["live_resolver"].present is None
    assert probes["hashed_lock"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_lock_files_match_generator() -> None:
    verified = verify_lock_files()
    assert verified["ok"] is True
    assert verified["missing"] == []
    catalog = platform_lock_catalog()
    assert catalog["interface"] == "PlatformDependencyLockCatalog@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_resolver_qualified"] is False
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_datasets_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_kit_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateDependencyLocks@1"' in pyproject
    assert 'requirements-authority = "requirements.txt"' in pyproject
    requirements = (_PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert scan_requirement_text(requirements)["vcs"] == ()
    inner = (_PACKAGE_ROOT / "ipfs_accelerate_py" / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert scan_requirement_text(inner)["vcs"]


def test_scan_requirement_text_rejects_git_and_editable() -> None:
    hits = scan_requirement_text(
        "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main\n-e ../ipfs_datasets_py\n"
    )
    assert hits["vcs"]
    assert hits["editable"]
    assert hits["mutable"]
    clean = scan_requirement_text("requests>=2.28.0\n")
    assert clean["vcs"] == ()
    assert clean["editable"] == ()
    assert clean["mutable"] == ()


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(DependencyLockError, match="simulated"):
        qualify_dependency_locks(
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
        )


def test_live_claim_without_measured_live_evidence_is_rejected() -> None:
    with pytest.raises(DependencyLockError, match="measured_live"):
        qualify_dependency_locks(
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
        )
