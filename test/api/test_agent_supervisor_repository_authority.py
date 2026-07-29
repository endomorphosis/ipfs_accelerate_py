"""Contract tests for canonical SwissKnife repository authority."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_authority import (
    AuthorityJoinKind,
    AuthorityRootMismatchError,
    AuthoritySource,
    FreshnessWorkKind,
    GitlinkAuthorityError,
    RepositoryAuthorityError,
    ReviewedAuthorityOverride,
    ReviewedEvidenceError,
    SnapshotAuthority,
    bind_authority_reference,
    build_repository_authority,
    dump_snapshot_authority,
    join_authority_bound_references,
    load_snapshot_authority,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _init_repository(root: Path, content: str = "base\n") -> str:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Repository Authority Tests")
    _git(root, "config", "user.email", "authority-tests@example.invalid")
    root.joinpath("README.md").write_text(content, encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", "base")
    return _git(root, "rev-parse", "HEAD")


def _commit(root: Path, content: str, message: str) -> str:
    root.joinpath("README.md").write_text(content, encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _integration_with_gitlink(root: Path, program_commit: str) -> Path:
    integration = root / "integration"
    _init_repository(integration, "integration\n")
    _git(
        integration,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{program_commit},swissknife",
    )
    _git(integration, "commit", "-q", "-m", "record SwissKnife gitlink")
    return integration


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    program = tmp_path / "program"
    base = _init_repository(program)
    integration = _integration_with_gitlink(tmp_path, base)
    return integration, program, base


def _kinds(authority: SnapshotAuthority) -> set[FreshnessWorkKind]:
    return {item.kind for item in authority.freshness_work}


def test_gitlink_is_default_authority_and_checkout_is_independently_bound(
    tmp_path: Path,
) -> None:
    integration, program, gitlink_commit = _fixture(tmp_path)
    checkout_commit = _commit(program, "ahead\n", "checkout advances")

    authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )

    assert authority.authority_source is AuthoritySource.INTEGRATION_GITLINK
    assert authority.program_commit == gitlink_commit
    assert authority.program_commit != checkout_commit
    assert authority.integration_checkout.checkout_cid
    assert authority.swissknife_checkout.checkout_cid
    assert (
        authority.integration_checkout.checkout_cid
        != authority.swissknife_checkout.checkout_cid
    )
    assert _kinds(authority) == {FreshnessWorkKind.CHECKOUT_AHEAD}

    integration_cid = authority.integration_checkout.checkout_cid
    program.joinpath("untracked.txt").write_text("local work\n", encoding="utf-8")
    dirty_authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )
    assert dirty_authority.integration_checkout.checkout_cid == integration_cid
    assert (
        dirty_authority.swissknife_checkout.checkout_cid
        != authority.swissknife_checkout.checkout_cid
    )
    assert _kinds(dirty_authority) == {
        FreshnessWorkKind.CHECKOUT_AHEAD,
        FreshnessWorkKind.CHECKOUT_DIRTY,
    }


def test_divergence_creates_typed_freshness_work(tmp_path: Path) -> None:
    program = tmp_path / "program"
    base = _init_repository(program)
    authority_commit = _commit(program, "authority\n", "authority branch")
    integration = _integration_with_gitlink(tmp_path, authority_commit)
    _git(program, "reset", "--hard", base)
    checkout_commit = _commit(program, "checkout\n", "checkout branch")

    authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )

    assert authority.program_commit == authority_commit
    assert authority.swissknife_checkout.head_commit == checkout_commit
    assert _kinds(authority) == {FreshnessWorkKind.CHECKOUT_DIVERGED}


def test_missing_checkout_creates_typed_freshness_work(tmp_path: Path) -> None:
    program = tmp_path / "program"
    gitlink_commit = _init_repository(program)
    integration = _integration_with_gitlink(tmp_path, gitlink_commit)

    authority = build_repository_authority(integration)

    assert not authority.swissknife_checkout.present
    assert _kinds(authority) == {FreshnessWorkKind.CHECKOUT_MISSING}
    assert authority.freshness_work[0].checkout_commit == ""


def test_reviewed_evidence_can_explicitly_change_program_authority(
    tmp_path: Path,
) -> None:
    integration, program, gitlink_commit = _fixture(tmp_path)
    reviewed_commit = _commit(program, "reviewed\n", "reviewed program")
    override = ReviewedAuthorityOverride(
        program_commit=reviewed_commit,
        supersedes_gitlink_commit=gitlink_commit,
        reviewer="contract-assurance@example.invalid",
        reviewed_at="2026-07-29T00:00:00Z",
        evidence={"review": "SCA-168", "decision": "accept"},
    )

    authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
        reviewed_override=override,
    )

    assert authority.authority_source is AuthoritySource.REVIEWED_EVIDENCE
    assert authority.program_commit == reviewed_commit
    assert authority.reviewed_override == override
    assert authority.freshness_work == ()

    wrong_scope = replace(
        override,
        supersedes_gitlink_commit=reviewed_commit,
        evidence_cid="",
    )
    with pytest.raises(ReviewedEvidenceError) as exc_info:
        build_repository_authority(
            integration,
            swissknife_checkout=program,
            reviewed_override=wrong_scope,
        )
    assert exc_info.value.reason_code == "reviewed_evidence_scope_mismatch"


def test_cache_proof_and_artifact_joins_fail_closed_across_roots(
    tmp_path: Path,
) -> None:
    integration, program, _ = _fixture(tmp_path)
    authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )
    references = tuple(
        bind_authority_reference(
            authority,
            kind=kind,
            reference_id=f"{kind.value}:one",
            payload={"kind": kind.value, "result": "accepted"},
        )
        for kind in AuthorityJoinKind
    )
    assert join_authority_bound_references(authority, references) == references

    integration.joinpath("README.md").write_text(
        "integration changed\n",
        encoding="utf-8",
    )
    foreign_authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )
    assert foreign_authority.authority_root_cid != authority.authority_root_cid

    for reference in references:
        with pytest.raises(AuthorityRootMismatchError) as exc_info:
            join_authority_bound_references(foreign_authority, (reference,))
        assert exc_info.value.reason_code == "authority_root_mismatch"


def test_serialized_authority_revalidates_every_cid_and_freshness_obligation(
    tmp_path: Path,
) -> None:
    integration, program, _ = _fixture(tmp_path)
    _commit(program, "ahead\n", "checkout advances")
    authority = build_repository_authority(
        integration,
        swissknife_checkout=program,
    )
    authority_path = tmp_path / "snapshot-authority.json"
    authority_path.write_text(
        dump_snapshot_authority(authority),
        encoding="utf-8",
    )

    assert load_snapshot_authority(authority_path) == authority

    tampered = json.loads(authority_path.read_text(encoding="utf-8"))
    tampered["swissknife_checkout"]["dirty"] = True
    authority_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RepositoryAuthorityError) as exc_info:
        load_snapshot_authority(authority_path)
    assert exc_info.value.reason_code == "authority_cid_mismatch"

    with pytest.raises(RepositoryAuthorityError) as exc_info:
        replace(authority, freshness_work=(), authority_root_cid="")
    assert exc_info.value.reason_code == "incomplete_freshness_work"


def test_gitlink_record_must_be_a_stage_zero_gitlink(tmp_path: Path) -> None:
    integration = tmp_path / "integration"
    _init_repository(integration)

    with pytest.raises(GitlinkAuthorityError) as exc_info:
        build_repository_authority(integration)
    assert exc_info.value.reason_code == "gitlink_record_missing"


def test_checked_in_snapshot_authority_is_self_validating() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    state_path = repository_root / (
        "data/agent_supervisor/swissknife_contract_assurance/state/"
        "snapshot_authority.json"
    )

    authority = load_snapshot_authority(state_path)

    assert authority.authority_source is AuthoritySource.INTEGRATION_GITLINK
    assert authority.program_commit == authority.integration_gitlink_commit
    assert authority.integration_checkout.checkout_id == "integration"
    assert authority.swissknife_checkout.checkout_id == "swissknife"
    assert authority.authority_root_cid
