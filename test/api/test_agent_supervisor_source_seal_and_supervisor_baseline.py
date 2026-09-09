"""PCPR-000 fail-closed source seal and supervisor baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.source_seal_and_supervisor_baseline import (
    CANONICAL_CONTRACT_CATALOG,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_ACCELERATOR_COMMIT,
    CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID,
    CURRENT_HEAD_DATASETS_COMMIT,
    CURRENT_HEAD_KIT_COMMIT,
    CURRENT_HEAD_POLICY_BUNDLE_CID,
    IMMUTABLE_BOOTSTRAP_INPUTS,
    PCPR_000_GOAL_ID,
    PCPR_000_TASK_ID,
    PLANNING_SOURCE_FOREST,
    POLICY_BASELINE_KEYS,
    SOURCE_AUTHORITY_COUNT,
    SOURCE_AUTHORITY_PATHS,
    SOURCE_REPOSITORIES,
    SOURCE_SEAL_INTERFACE,
    SUPERVISOR_BASELINE_SURFACES,
    BootstrapArtifact,
    PolicyBaseline,
    RepositoryObservation,
    SourceSealError,
    current_head_bootstrap_artifacts,
    current_head_pcpr_000_current_tree_binding,
    current_head_pcpr_000_receipt_promotion,
    current_head_pcpr_000_receipt_sections,
    current_head_policies,
    gitlink_digest,
    pcpr_000_current_tree_binding,
    qualify_current_head_source_seal,
    seal_source_forest_and_supervisor_baseline,
    validate_pcpr_000_outer_receipt,
)


def _clone_source(
    item: RepositoryObservation, **changes: object
) -> RepositoryObservation:
    payload = item.to_mapping()
    payload.update(changes)
    return RepositoryObservation(**payload)


def test_closed_vocabularies_match_pcpr_000_requirements() -> None:
    assert PCPR_000_TASK_ID == "PCPR-000"
    assert PCPR_000_GOAL_ID == "PCPR-G110"
    assert SOURCE_SEAL_INTERFACE == "SourceSealAndSupervisorBaseline@1"
    assert SOURCE_AUTHORITY_COUNT == 3
    assert len(SOURCE_REPOSITORIES) == 3
    assert SOURCE_AUTHORITY_PATHS == {
        "external/ipfs_accelerate",
        "external/ipfs_datasets",
        "external/ipfs_kit",
    }
    assert len(IMMUTABLE_BOOTSTRAP_INPUTS) == 8
    assert len(POLICY_BASELINE_KEYS) == 9
    assert len(CANONICAL_CONTRACT_CATALOG) == 14
    assert len(SUPERVISOR_BASELINE_SURFACES) == 6
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "non_promoted_supervisor_unqualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert "supervisor_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert all(item["disposition"] == "baseline_recorded_not_frozen" for item in CANONICAL_CONTRACT_CATALOG)
    assert all(item["freeze_task"] == "PCPR-002" for item in CANONICAL_CONTRACT_CATALOG)
    assert PLANNING_SOURCE_FOREST["ipfs_accelerate_py"]["commit"] == CURRENT_HEAD_ACCELERATOR_COMMIT
    assert PLANNING_SOURCE_FOREST["ipfs_datasets_py"]["commit"] == CURRENT_HEAD_DATASETS_COMMIT
    assert PLANNING_SOURCE_FOREST["ipfs_kit_py"]["commit"] == CURRENT_HEAD_KIT_COMMIT


def test_clean_gitlink_forest_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_source_seal()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.changed_source_seal_requires_fresh_inventory is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID
    assert verdict.blockers == ()
    assert verdict.portfolio.binding_kind == "isolated_implementation_worktree"
    assert [item.binding_kind for item in verdict.sources] == [
        "gitlink",
        "gitlink",
        "gitlink",
    ]
    assert all(item.planning_pin_match is True for item in verdict.sources)
    assert all(item.source_authority is True for item in verdict.sources)
    assert verdict.portfolio.source_authority is False
    section = current_head_pcpr_000_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False
    assert section["duckdb_or_quack_state_written"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID


def test_current_head_receipt_sections_are_rnd_non_promoted_and_not_a_release() -> None:
    sections = current_head_pcpr_000_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID
    forest = sections["source_forest"]
    assert forest["source_authority_count"] == 3
    assert forest["recursive_sibling_submodules_are_source_authority"] is False
    assert forest["changed_source_seal_requires_fresh_inventory"] is False
    contracts = sections["contract_baseline"]
    assert contracts["frozen"] is False
    assert contracts["freeze_task"] == "PCPR-002"
    assert len(contracts["catalog"]) == 14
    policies = sections["policy_baseline"]
    assert policies["bundle_cid"] == CURRENT_HEAD_POLICY_BUNDLE_CID
    assert policies["markdown_is_bootstrap_only"] is True
    assert policies["ducklake_non_authoritative"] is True
    assert [item["key"] for item in policies["policies"]] == list(POLICY_BASELINE_KEYS)
    baseline = sections["supervisor_baseline"]
    assert baseline["frozen"] is False
    assert baseline["supervisor_promoted"] is False
    assert baseline["duckdb_task_state"] == "unavailable"
    assert baseline["board_projection_is_not_duckdb_authority"] is True
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["recursive_sibling_submodules_are_not_source_authority"] is True


def test_outer_receipt_validator_accepts_generated_clean_forest_receipt() -> None:
    sections = current_head_pcpr_000_receipt_sections()
    payload = {
        "task_id": PCPR_000_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "source_forest": sections["source_forest"],
        "current_tree_binding": current_head_pcpr_000_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
        },
    }
    checked = validate_pcpr_000_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_000_receipt_sections()
    forged = {
        "task_id": PCPR_000_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(SourceSealError, match="closed PCPR release"):
        validate_pcpr_000_outer_receipt(forged)
    forged_status = {
        "task_id": PCPR_000_TASK_ID,
        "status": "non_promoted_supervisor_unqualified",
        "qualification_verdict": sections["qualification_verdict"],
    }
    with pytest.raises(SourceSealError, match="closed PCPR release"):
        validate_pcpr_000_outer_receipt(forged_status)
    forged_verdict = {
        "task_id": PCPR_000_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_unmeasured",
        },
    }
    with pytest.raises(SourceSealError, match="must be null"):
        validate_pcpr_000_outer_receipt(forged_verdict)


def test_duckdb_or_quack_write_is_rejected() -> None:
    with pytest.raises(SourceSealError, match="DuckDB or Quack"):
        seal_source_forest_and_supervisor_baseline(
            portfolio=_portfolio(),
            sources=_sources(),
            bootstrap_artifacts=current_head_bootstrap_artifacts(),
            policies=current_head_policies(),
            duckdb_or_quack_state_written=True,
        )


def test_simulated_clean_cannot_mint_a_seal() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(sources[0], evidence_kind="simulated", clean=True)
    with pytest.raises(SourceSealError, match="simulated cleanliness"):
        seal_source_forest_and_supervisor_baseline(
            portfolio=_portfolio(),
            sources=sources,
            bootstrap_artifacts=current_head_bootstrap_artifacts(),
            policies=current_head_policies(),
        )


def test_gitlink_mismatch_without_isolated_binding_is_typed_blocked() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(
        sources[0],
        commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        gitlink="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        isolated_source_binding="",
    )
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.sources[0].binding_kind == "blocked"
    assert any("gitlink_mismatch" in item for item in verdict.blockers)


def test_isolated_source_binding_admits_gitlink_mismatch() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(
        sources[0],
        commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        tree="cccccccccccccccccccccccccccccccccccccccc",
        gitlink="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        isolated_source_binding="explicit-isolated-source-binding",
    )
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.sources[0].binding_kind == "isolated_source"
    assert verdict.changed_source_seal_requires_fresh_inventory is True
    assert verdict.sources[0].planning_pin_match is False


def test_origin_main_not_ancestor_is_typed_blocked() -> None:
    sources = list(_sources())
    sources[1] = _clone_source(sources[1], origin_main_is_ancestor=False)
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.promotion_status == "typed_blocked"
    assert any("origin_main_not_ancestor" in item for item in verdict.blockers)


def test_missing_repository_is_typed_unavailable() -> None:
    sources = list(_sources())
    sources[2] = RepositoryObservation(
        repository="ipfs_kit_py",
        path="external/ipfs_kit",
        commit=None,
        tree=None,
        origin_main=None,
        origin_main_is_ancestor=None,
        gitlink=None,
        clean=None,
        exact_toplevel=False,
        evidence_kind="unavailable",
        reason="Repository directory is not present.",
    )
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.promotion_status == "typed_unavailable"
    assert verdict.sources[2].binding_kind == "unavailable"
    assert verdict.closed_release_outcome is None


def test_dirty_worktree_is_recorded_and_not_claimed_clean() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(
        sources[0],
        clean=False,
        reason="Nested worktree has uncommitted source-seal files.",
    )
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.sources[0].observation.clean is False
    assert verdict.release_claim is False


def test_origin_main_unavailable_is_not_recorded_as_false_or_zero() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(
        sources[0], origin_main=None, origin_main_is_ancestor=None
    )
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=sources,
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )
    assert verdict.sources[0].observation.origin_main is None
    assert verdict.sources[0].observation.origin_main_is_ancestor is None
    assert any("origin_main_unavailable" in item for item in verdict.blockers)
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.closed_release_outcome is None


def test_recursive_sibling_paths_cannot_become_source_authority() -> None:
    sources = list(_sources())
    sources[0] = _clone_source(sources[0], path="external/meta-wearables-dat-android")
    with pytest.raises(SourceSealError, match="not a PCPR source-authority"):
        seal_source_forest_and_supervisor_baseline(
            portfolio=_portfolio(),
            sources=sources,
            bootstrap_artifacts=current_head_bootstrap_artifacts(),
            policies=current_head_policies(),
        )


def test_undeclared_bootstrap_or_policy_is_rejected() -> None:
    extra_bootstrap = (
        *current_head_bootstrap_artifacts(),
        BootstrapArtifact(
            path="docs/secret.md",
            sha256="0" * 64,
            bytes=1,
            evidence_kind="measured",
        ),
    )
    with pytest.raises(SourceSealError, match="undeclared bootstrap"):
        seal_source_forest_and_supervisor_baseline(
            portfolio=_portfolio(),
            sources=_sources(),
            bootstrap_artifacts=extra_bootstrap,
            policies=current_head_policies(),
        )
    extra_policy = (
        *current_head_policies(),
        PolicyBaseline(
            key="secret_policy",
            content_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            evidence_kind="measured",
        ),
    )
    with pytest.raises(SourceSealError, match="undeclared policy"):
        seal_source_forest_and_supervisor_baseline(
            portfolio=_portfolio(),
            sources=_sources(),
            bootstrap_artifacts=current_head_bootstrap_artifacts(),
            policies=extra_policy,
        )


def test_missing_bootstrap_is_typed_unavailable_not_zero() -> None:
    verdict = seal_source_forest_and_supervisor_baseline(
        portfolio=_portfolio(),
        sources=_sources(),
        bootstrap_artifacts=(),
        policies=current_head_policies(),
    )
    assert all(item.evidence_kind == "unavailable" for item in verdict.bootstrap_artifacts)
    assert all(item.sha256 is None for item in verdict.bootstrap_artifacts)
    assert any(item == "bootstrap:unavailable" for item in verdict.blockers)
    assert verdict.closed_release_outcome is None


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_000_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_pre_change_commit"] == binding["accelerator_gitlink"]
    assert binding["datasets_commit"] == binding["datasets_gitlink"]
    assert binding["kit_commit"] == binding["kit_gitlink"]
    assert binding["accelerator_post_change_commit"] == "pending nested commit after admission"
    assert all(outcome not in binding["outer_subject"] for outcome in CLOSED_RELEASE_OUTCOMES)
    assert "closed_release_outcome" not in binding


def test_current_tree_binding_rejects_non_ancestor_and_gitlink_mismatch() -> None:
    kwargs = dict(current_head_pcpr_000_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["origin_main_is_ancestor"] = False
    with pytest.raises(SourceSealError, match="origin_main_is_ancestor"):
        pcpr_000_current_tree_binding(**kwargs)
    kwargs = dict(current_head_pcpr_000_current_tree_binding())
    kwargs.pop("outer_repository")
    kwargs.pop("owning_repository_for_receipts")
    kwargs.pop("accelerator_post_change_commit")
    kwargs.pop("accelerator_post_change_tree")
    kwargs.pop("evidence_kind")
    kwargs["accelerator_gitlink"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(SourceSealError, match="accelerator_gitlink"):
        pcpr_000_current_tree_binding(**kwargs)


def test_gitlink_digest_is_order_independent() -> None:
    first = (
        {"path": "b", "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        {"path": "a", "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
    )
    second = tuple(reversed(first))
    assert gitlink_digest(first) == gitlink_digest(second)
    assert len(gitlink_digest(first)) == 64


def test_observe_source_forest_on_tmp_repos(tmp_path: Path) -> None:
    git = "/usr/bin/git"
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    _init_git(portfolio, git)
    accelerate = portfolio / "external" / "ipfs_accelerate"
    datasets = portfolio / "external" / "ipfs_datasets"
    kit = portfolio / "external" / "ipfs_kit"
    for path in (accelerate, datasets, kit):
        path.mkdir(parents=True)
        _init_git(path, git)
        (path / "README").write_text("source\n", encoding="utf-8")
        _git(path, git, "add", "README")
        _git(path, git, "commit", "-m", "init")
    _git(portfolio, git, "add", "external")
    # Submodule-style gitlinks are not required for observation of nested roots.
    (portfolio / "NOTE").write_text("outer\n", encoding="utf-8")
    _git(portfolio, git, "add", "NOTE")
    _git(portfolio, git, "commit", "-m", "outer")

    from ipfs_accelerate_py.agent_supervisor.validation.source_seal_and_supervisor_baseline import (
        observe_git_repository,
    )

    observed = observe_git_repository(
        accelerate,
        repository="ipfs_accelerate_py",
        relative="external/ipfs_accelerate",
        git_binary=git,
    )
    assert observed.evidence_kind == "measured"
    assert observed.clean is True
    assert observed.commit is not None
    assert observed.tree is not None
    assert observed.origin_main is None
    assert observed.origin_main_is_ancestor is None

    missing = observe_git_repository(
        tmp_path / "absent",
        repository="ipfs_kit_py",
        relative="external/ipfs_kit",
        git_binary=git,
    )
    assert missing.evidence_kind == "unavailable"
    assert missing.commit is None


def _init_git(path: Path, git: str) -> None:
    _git(path, git, "init")
    _git(path, git, "config", "user.email", "pcpr@example.test")
    _git(path, git, "config", "user.name", "PCPR")


def _git(path: Path, git: str, *args: str) -> None:
    import subprocess

    result = subprocess.run(
        [git, *args], cwd=path, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def _portfolio() -> RepositoryObservation:
    from ipfs_accelerate_py.agent_supervisor.validation.source_seal_and_supervisor_baseline import (
        current_head_portfolio_observation,
    )

    return current_head_portfolio_observation()


def _sources() -> tuple[RepositoryObservation, ...]:
    from ipfs_accelerate_py.agent_supervisor.validation.source_seal_and_supervisor_baseline import (
        current_head_source_observations,
    )

    return current_head_source_observations()
