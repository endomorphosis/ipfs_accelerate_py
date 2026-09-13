"""Real datasets reconstruction; fixture target source is never imported."""

import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state import reconstruction as r
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticStateBundle, SemanticStateModelError,
)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
    ).decode().strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture")
    return dict(expected_commit=git(root, "rev-parse", "HEAD"),
                expected_tree=git(root, "rev-parse", "HEAD^{tree}"),
                repository_id="fixture:reconstruction")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-b", "main")
    (root / "module.py").write_text("def add(a, b):\n    return a + b\n")
    return root, commit(root)


def test_cold_rebuild_and_nomination_use_real_datasets(source):
    root, request = source
    first = r.reconstruct_semantic_state(root, **request)
    second = r.reconstruct_semantic_state(root, **request, nominated_bundle=first.bundle)
    assert first.bundle.blocks == second.bundle.blocks
    assert first.nomination_matched is None
    assert second.nomination_matched is True
    observed = second.observation()
    assert observed["semantic_acceptance_authority"] is False
    assert observed["completion_authority"] is False
    assert observed["block_count"] > 5
    assert observed["opaque_entries"] == []
    assert observed["commit"] == request["expected_commit"]
    assert json.loads(first.bundle.blocks[first.bundle.root.root_cid])["repository_id"] == request["repository_id"]


def test_stale_valid_nomination_rejected_after_real_source_change(source):
    root, request = source
    old = r.reconstruct_semantic_state(root, **request)
    (root / "module.py").write_text("def add(a, b):\n    return a - b\n")
    changed = commit(root)
    with pytest.raises(r.ReconstructionError, match="cold reconstruction"):
        r.reconstruct_semantic_state(root, **changed, nominated_bundle=old.bundle)
    with pytest.raises(r.ReconstructionError, match="requested committed population"):
        r.reconstruct_semantic_state(root, **request)


def test_missing_transitive_block_in_matching_nomination_is_rejected(source):
    root, request = source
    original = r.reconstruct_semantic_state(root, **request).bundle
    blocks = dict(original.blocks)
    del blocks[original.root.symbol_fact_index_cid]
    # Construction checks individual content hashes; full verification must
    # still traverse required references even when the root CID itself matches.
    incomplete = SemanticStateBundle(original.root, blocks)
    with pytest.raises(SemanticStateModelError):
        r.reconstruct_semantic_state(root, **request, nominated_bundle=incomplete)


def test_corrupt_block_cannot_be_presented_as_valid_nomination(source):
    root, request = source
    original = r.reconstruct_semantic_state(root, **request).bundle
    blocks = dict(original.blocks)
    blocks[original.root.symbol_fact_index_cid] = b"{}"
    with pytest.raises(SemanticStateModelError):
        SemanticStateBundle(original.root, blocks)


@pytest.mark.parametrize("mutation", ["dirty", "untracked", "staged", "assume", "skip"])
def test_working_or_hidden_source_rejected(source, mutation):
    root, request = source
    if mutation in {"assume", "skip"}:
        git(root, "update-index", "--assume-unchanged" if mutation == "assume" else "--skip-worktree", "module.py")
    (root / ("extra.py" if mutation == "untracked" else "module.py")).write_text("x = 99\n")
    if mutation == "staged":
        git(root, "add", ".")
    with pytest.raises(r.ReconstructionError, match="clean|hides"):
        r.reconstruct_semantic_state(root, **request)


@pytest.mark.parametrize("limits,reason", [
    (r.ReconstructionLimits(max_total_bytes=1), "total byte"),
    (r.ReconstructionLimits(max_file_bytes=1), "per-file"),
    (r.ReconstructionLimits(max_entries=1), "entry"),
])
def test_budget_refusal_precedes_producer_execution(source, limits, reason, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.semantic_index import committed_snapshot
    root, _ = source
    (root / "second.py").write_text("x = 1\n")
    request = commit(root)
    def forbidden(*args, **kwargs):
        pytest.fail("source acquisition occurred beyond the configured budget")
    monkeypatch.setattr(committed_snapshot, "snapshot_committed_repository", forbidden)
    with pytest.raises(r.ReconstructionBudgetError, match=reason) as refused:
        r.reconstruct_semantic_state(root, **request, limits=limits)
    assert len(refused.value.plan.entries) == 2
    assert refused.value.plan.total_blob_bytes == sum(p.stat().st_size for p in root.glob("*.py"))
    assert refused.value.plan.to_dict()["blob_bytes_acquired"] == 0


def test_source_drift_during_producer_is_rejected(source, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import semantic_state
    root, request = source
    actual = semantic_state.build_semantic_state
    def changed(*args, **kwargs):
        result = actual(*args, **kwargs)
        (root / "module.py").write_text("x = 2\n")
        return result
    monkeypatch.setattr(semantic_state, "build_semantic_state", changed)
    with pytest.raises(r.ReconstructionError, match="clean"):
        r.reconstruct_semantic_state(root, **request)


def test_opaque_source_stays_visible_and_target_is_not_executed(source):
    root, _ = source
    (root / "module.py").write_text("raise RuntimeError('must never execute target')\n")
    (root / "alias.py").symlink_to("module.py")
    request = commit(root)
    result = r.reconstruct_semantic_state(root, **request)
    assert result.observation()["opaque_entries"] == [
        {"path": "alias.py", "reason": "symlink_or_nonregular"},
    ]
    assert result.observation()["completion_authority"] is False


def test_datasets_explicit_complete_population_includes_normally_excluded_source(source, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner
    root, _ = source
    for directory in ("vendor", "coverage", "venv"):
        (root / directory).mkdir()
        (root / directory / "required.py").write_text("def required():\n    return 1\n")
    request = commit(root)
    scanned = []
    real_scan = RepositoryScanner.scan_snapshot
    def observe(self, snapshot, *args, **kwargs):
        scanned.extend(entry.path for entry in snapshot.entries)
        return real_scan(self, snapshot, *args, **kwargs)
    monkeypatch.setattr(RepositoryScanner, "scan_snapshot", observe)
    result = r.reconstruct_semantic_state(root, **request)
    assert scanned == ["coverage/required.py", "module.py", "vendor/required.py", "venv/required.py"]
    assert result.population_cid
    assert result.observation()["population_scope"] == "complete-committed"
    assert result.observation()["completion_authority"] is False


def test_reduced_producer_population_is_still_refused(source, monkeypatch):
    from dataclasses import replace
    from ipfs_datasets_py.logic.software_contracts.semantic_index import committed_snapshot
    root, request = source
    real_snapshot = committed_snapshot.snapshot_committed_repository
    def reduced(*args, **kwargs):
        return replace(real_snapshot(*args, **kwargs), entries=())
    monkeypatch.setattr(committed_snapshot, "snapshot_committed_repository", reduced)
    with pytest.raises(r.ReconstructionError, match="omits committed source"):
        r.reconstruct_semantic_state(root, **request)


def test_gitlink_is_an_explicit_opaque_input_not_nested_acceptance(source):
    root, request = source
    (root / "nested").mkdir()
    git(root, "update-index", "--add", "--cacheinfo",
        "160000," + request["expected_commit"] + ",nested")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "gitlink")
    request.update(expected_commit=git(root, "rev-parse", "HEAD"),
                   expected_tree=git(root, "rev-parse", "HEAD^{tree}"))
    result = r.reconstruct_semantic_state(root, **request)
    assert result.observation()["opaque_entries"] == [
        {"path": "nested", "reason": "symlink_or_nonregular"},
    ]


def test_issuer_shaped_dict_cannot_replace_bundle(source):
    root, request = source
    with pytest.raises(SemanticStateModelError):
        r.reconstruct_semantic_state(root, **request,
                                     nominated_bundle={"issuer": "datasets", "accepted": True})


def test_non_utf8_source_name_preserves_population(source):
    import os
    root, _ = source
    path = os.fsencode(root) + b"/invalid-\xff.py"
    with open(path, "wb") as stream:
        stream.write(b"x = 1\n")
    request = commit(root)
    result = r.reconstruct_semantic_state(root, **request)
    assert len(result.opaque_entries) == 1
    assert result.opaque_entries[0][0].endswith(b"invalid-\xff.py".hex())


def test_repository_identity_and_limits_bind_configuration(source):
    root, request = source
    original = r.reconstruct_semantic_state(root, **request)
    renamed = r.reconstruct_semantic_state(root, **{**request, "repository_id": "fixture:other"})
    bounded = r.reconstruct_semantic_state(root, **request, limits=r.ReconstructionLimits(max_entries=5))
    assert len({x.configuration_digest for x in (original, renamed, bounded)}) == 3
    assert renamed.bundle.root.root_cid != original.bundle.root.root_cid


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "20"])
def test_limits_require_positive_integer(value):
    with pytest.raises(r.ReconstructionError):
        r.ReconstructionLimits(max_total_bytes=value)


def test_subdirectory_cannot_silently_expand_requested_scope(source):
    root, request = source
    sub = root / "pkg"
    sub.mkdir()
    with pytest.raises(r.ReconstructionError, match="exact repository root"):
        r.reconstruct_semantic_state(sub, **request)


def test_filesystem_source_has_no_committed_authority(tmp_path):
    with pytest.raises(r.ReconstructionError, match="failed|unavailable"):
        r.reconstruct_semantic_state(tmp_path, expected_commit="a" * 40,
                                     expected_tree="b" * 40, repository_id="fixture")
