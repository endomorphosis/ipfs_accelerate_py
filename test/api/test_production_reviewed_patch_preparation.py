"""Private Git patch previews and versioned task contracts preserve authority."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import MAX_PROVIDER_RESPONSE_BYTES
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_context_slice import DEFAULT_MAX_SCOPE_PATHS, DEFAULT_MAX_SOURCE_BYTES
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_reviewed_effect import (
    PRODUCTION_TASK_CONTRACT_SCHEMA,
    ProductionReviewedEffectBinding,
    derive_production_reviewed_patch_effects,
    production_task_contract,
    production_task_contract_cid,
    verify_finalized_production_reviewed_effect,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", *args], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _repo(tmp_path: Path, *, object_format: str = "sha1") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--object-format=" + object_format)
    _git(repo, "config", "user.name", "Private Preview Test")
    _git(repo, "config", "user.email", "preview@example.invalid")
    for name, body in {"modify.txt": "one\n", "delete.txt": "obsolete\n", "run.sh": "#!/bin/sh\nexit 0\n"}.items():
        path = repo / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o644)
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    return repo


def _snapshot(repo: Path):
    # Includes the real index, object store, references, worktree bytes and modes.
    # --no-optional-locks keeps this observation from refreshing the index itself.
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    entries = {}
    for path in repo.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            body = os.readlink(path)
        elif stat.S_ISREG(mode):
            body = path.read_bytes()
        else:
            body = None
        entries[path.relative_to(repo).as_posix()] = (mode, body)
    return status, entries


def _new_file_patch(path: str, *, mode: str = "100644", body: str = "new") -> str:
    return (
        f"diff --git a/{path} b/{path}\nnew file mode {mode}\n"
        f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{body}\n"
    )


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_private_preview_reconstructs_all_regular_effects_without_source_mutation(
    tmp_path: Path, object_format: str,
) -> None:
    repo = _repo(tmp_path, object_format=object_format)
    baseline = _git(repo, "rev-parse", "HEAD")
    # Deliberately dirty source bytes and index are not input to the committed preview.
    (repo / "modify.txt").write_text("uncommitted user bytes\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged user bytes\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    before = _snapshot(repo)
    patch = (
        "diff --git a/modify.txt b/modify.txt\n--- a/modify.txt\n+++ b/modify.txt\n"
        "@@ -1 +1 @@\n-one\n+two\n"
        "diff --git a/delete.txt b/delete.txt\ndeleted file mode 100644\n"
        "--- a/delete.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-obsolete\n"
        "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
        + _new_file_patch("created.txt")
    )
    effects = derive_production_reviewed_patch_effects(
        repo_root=repo, baseline_ref=baseline, patch=patch,
    )
    assert effects == {
        "modify.txt": (b"two\n", 0o644), "delete.txt": None,
        "run.sh": (b"#!/bin/sh\nexit 0\n", 0o755),
        "created.txt": (b"new\n", 0o644),
    }
    assert _snapshot(repo) == before


@pytest.mark.parametrize("patch", [
    "", "this is not a patch\n", _new_file_patch("../escape.txt"),
    _new_file_patch(".git/config"), _new_file_patch("link", mode="120000", body="modify.txt"),
])
def test_invalid_unsafe_and_symlink_previews_fail_without_source_mutation(
    tmp_path: Path, patch: str,
) -> None:
    repo = _repo(tmp_path)
    before = _snapshot(repo)
    with pytest.raises(ValueError):
        derive_production_reviewed_patch_effects(repo_root=repo, baseline_ref="HEAD", patch=patch)
    assert _snapshot(repo) == before
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("budget", ["provider_bytes", "effect_paths", "postimage_bytes"])
def test_preview_enforces_real_budgets_without_source_mutation(tmp_path: Path, budget: str) -> None:
    repo = _repo(tmp_path)
    if budget == "provider_bytes":
        patch = "x" * (MAX_PROVIDER_RESPONSE_BYTES + 1)
        reason = "provider response bound"
    elif budget == "effect_paths":
        patch = "".join(_new_file_patch(f"new-{index}.txt") for index in range(DEFAULT_MAX_SCOPE_PATHS + 1))
        reason = "effect path bound"
    else:
        # A compact patch can produce a postimage larger than the byte budget.
        (repo / "large.txt").write_bytes(b"first\n" + b"x" * DEFAULT_MAX_SOURCE_BYTES + b"\n")
        _git(repo, "add", "large.txt")
        _git(repo, "commit", "--quiet", "-m", "large baseline")
        patch = "diff --git a/large.txt b/large.txt\nold mode 100644\nnew mode 100755\n"
        reason = "postimage byte bound"
    before = _snapshot(repo)
    with pytest.raises(ValueError, match=reason):
        derive_production_reviewed_patch_effects(repo_root=repo, baseline_ref="HEAD", patch=patch)
    assert _snapshot(repo) == before


IDENTITY = {
    "canonical_task_key": "default:preview-001", "canonical_task_cid": "cidv1:preview-task",
    "display_task_id": "PREVIEW-001", "board_namespace": "default",
}


def _task(**changes):
    values = dict(
        task_id="PREVIEW-001", title="Bound task", priority="P0", track="review",
        depends_on=(), outputs=("modify.txt",), validation=("python -m py_compile modify.txt",),
        acceptance="exact reviewed bytes", metadata={"Policy": "operator-owned"}, status="ready",
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("status_key", ["Status", "STATUS", " \tStAtUs \t"])
def test_task_contract_v2_omits_only_administrative_status(status_key: str) -> None:
    ready = _task(metadata={"Policy": "operator-owned", status_key: "ready"})
    completed = _task(status="completed", metadata={"Policy": "operator-owned", status_key: "completed"})
    contract = production_task_contract(ready, IDENTITY)
    assert contract["schema"] == PRODUCTION_TASK_CONTRACT_SCHEMA
    assert PRODUCTION_TASK_CONTRACT_SCHEMA.endswith("@2")
    assert contract["metadata"] == {"Policy": "operator-owned"}
    assert contract == production_task_contract(completed, IDENTITY)


@pytest.mark.parametrize("change", [
    {"metadata": {"Policy": "different"}},
    {"metadata": {"Policy": "operator-owned", "Status reason": "changed authority"}},
    {"metadata": {"Policy": "operator-owned", "Completion": "automatic"}},
    {"metadata": {"Policy": "operator-owned", "production_provider_policy": "forged"}},
    {"acceptance": "different acceptance"}, {"validation": ("true",)},
    {"outputs": ("other.txt",)}, {"depends_on": ("PREVIEW-000",)},
])
def test_task_contract_v2_keeps_authority_metadata_and_task_scope_bound(change) -> None:
    assert production_task_contract_cid(_task(**change), IDENTITY) != production_task_contract_cid(_task(), IDENTITY)


def test_old_raw_status_contract_cid_is_rejected_by_real_cold_effect_verifier(tmp_path: Path) -> None:
    # Reuse the frozen native fixture: real owner writes, commits and signs offline.
    from test.api.test_production_independent_review_contract import _signed_fixture

    daemon, task, repo, _key, _calls, _baseline, evidence = _signed_fixture(tmp_path)
    identity = daemon._identity_for_task(task)
    finalized = ProductionReviewedEffectBinding.from_dict(evidence["production_reviewed_effect_binding"])
    kwargs = dict(
        repo_root=repo, task=task, task_identity=identity,
        expected_implementation_commit=evidence["implementation_commit"],
        expected_implementation_tree_id=evidence["implementation_tree_id"],
    )
    assert verify_finalized_production_reviewed_effect(finalized, **kwargs).admitted
    old_contract = production_task_contract(task, identity)
    old_contract.pop("schema")
    old_contract["metadata"] = dict(task.metadata)
    assert any(str(key).strip().casefold() == "status" for key in old_contract["metadata"])
    old_cid = content_identity(old_contract)
    assert old_cid != finalized.task_contract_cid
    candidate = replace(finalized, binding_id="", task_contract_cid=old_cid)
    candidate = replace(candidate, binding_id=content_identity(candidate.unsigned_dict()))
    checked = verify_finalized_production_reviewed_effect(candidate.to_dict(), **kwargs)
    assert not checked.admitted
    assert "reviewed_effect_task_contract_mismatch" in checked.reason_codes
