from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge import (
    protected_recovery_clearance as clearance,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_mutation_lock_path,
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


@dataclass(frozen=True)
class JournalFixture:
    repo: Path
    lock_path: Path
    receipt_dir: Path
    before_head: str
    untrusted_commit: str
    lease_id: str
    review: dict[str, object]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _identified(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = content_identity(result)
    return result


def _journal_fixture(
    tmp_path: Path,
    *,
    protected_path: str = "tasks.todo.md",
) -> JournalFixture:
    repo = tmp_path / "target-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")

    protected = repo / protected_path
    protected.write_text("Status: todo\n", encoding="utf-8")
    _git(repo, "add", f":(top,literal){protected_path}")
    _git(repo, "commit", "-m", "initial protected state")
    before_head = _git(repo, "rev-parse", "HEAD")

    protected.write_text("Status: completed\n", encoding="utf-8")
    _git(repo, "add", f":(top,literal){protected_path}")
    _git(repo, "commit", "-m", "operator status transition")
    untrusted_commit = _git(repo, "rev-parse", "HEAD")

    paths = [protected_path]
    guard = _identified(
        {
            "protected_paths": paths,
            "scopes": [
                {
                    "git_root": str(repo.resolve()),
                    "paths": paths,
                    "before_head": before_head,
                    "before_head_query": {
                        "ok": True,
                        "head": before_head,
                        "unborn": False,
                    },
                }
            ],
            "discovery_errors": [],
        },
        "guard_id",
    )
    intent = _identified(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "supervisor-protected-recovery-intent@1"
            ),
            "operation": "generated_board_update",
            "producer": "fixture-producer",
            "protected_paths": paths,
            "guard_id": guard["guard_id"],
        },
        "intent_id",
    )
    lease_id = content_identity(
        {
            "kind": "fixture-protected-recovery-lease",
            "repo_root": str(repo.resolve()),
            "before_head": before_head,
            "current_head": untrusted_commit,
        }
    )
    metadata = {
        "attempt": 0,
        "branch": "",
        "kind": "merge",
        "lease_id": lease_id,
        "operation": "generated_dirty_repair",
        "owner_script": "implementation_supervisor.py",
        "pid": 2_147_483_647,
        "producer": "fixture-producer",
        "protected_paths": paths,
        "protected_recovery_intent": intent,
        "protected_recovery_owner": "implementation_supervisor",
        "protected_recovery_required": True,
        "protected_release_guard": guard,
        "repo_root": str(repo.resolve()),
        # Exercise hardened journal fields emitted by newer checkout-lock code.
        "worktree_root": str(repo.resolve()),
        "repository_id": checkout_repository_id(repo),
        "state_dir": str((repo / "data" / "state").resolve()),
        "state_path": str((repo / "data" / "state" / "lane.json").resolve()),
        "task_id": "",
    }
    lock_path = checkout_mutation_lock_path(repo)
    lock_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lock_path.chmod(0o600)
    review = clearance.inspect_protected_recovery(repo)
    assert review["eligible"] is True, review
    assert review["reason"] == "protected_generated_history_untrusted"
    assert review["untrusted_commits"] == [untrusted_commit]
    return JournalFixture(
        repo=repo,
        lock_path=lock_path,
        receipt_dir=tmp_path / "receipts",
        before_head=before_head,
        untrusted_commit=untrusted_commit,
        lease_id=lease_id,
        review=review,
    )


def _apply(fixture: JournalFixture, **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "receipt_dir": fixture.receipt_dir,
        "expected_lease_id": fixture.lease_id,
        "expected_review_id": fixture.review["review_id"],
        "expected_lock_sha256": fixture.review["lock_sha256"],
        "approved_commits": [fixture.untrusted_commit],
        "operator_identity": "fixture-operator",
        "operator_note": (
            "Approve the exact reviewed protected-path history only; this "
            "does not assert completion or release authority."
        ),
    }
    arguments.update(overrides)
    return clearance.apply_protected_recovery_clearance(
        fixture.repo,
        **arguments,
    )


def _restore(
    fixture: JournalFixture,
    snapshot_bytes: bytes,
    **overrides: object,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "receipt_dir": fixture.receipt_dir,
        "snapshot_bytes": snapshot_bytes,
        "expected_lease_id": fixture.lease_id,
        "expected_lock_sha256": fixture.review["lock_sha256"],
        "source_review": fixture.review,
        "operator_event_id": "fixture-disappearance-event-0001",
        "approved_commits": [fixture.untrusted_commit],
        "operator_identity": "fixture-operator",
        "operator_note": (
            "Restore only the exact disappeared protected-recovery fence; "
            "this conveys no completion or release authority."
        ),
    }
    arguments.update(overrides)
    return clearance.restore_missing_protected_recovery_fence(
        fixture.repo,
        **arguments,
    )


def _reason(exc_info: pytest.ExceptionInfo[BaseException]) -> str:
    error = exc_info.value
    assert isinstance(error, clearance.ProtectedRecoveryClearanceError)
    return error.reason


def test_exact_apply_releases_lock_and_writes_three_non_authority_receipts(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)

    result = _apply(fixture)

    assert result["released"] is True
    assert result["decision"] == "approved_protected_history_only"
    assert result["authority"] == {
        "completion_authority": False,
        "verification_authority": False,
        "release_authority": False,
        "production_promotion_authority": False,
    }
    assert not fixture.lock_path.exists()
    receipts = sorted(fixture.receipt_dir.glob("*.json"))
    assert len(receipts) == 3
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    assert {payload["schema"] for payload in payloads} == {
        clearance.AUTHORIZATION_SCHEMA,
        clearance.RELEASE_INTENT_SCHEMA,
        clearance.FINAL_RECEIPT_SCHEMA,
    }
    for payload in payloads:
        assert payload["decision"] == "approved_protected_history_only"
        assert not any(payload["authority"].values())
        assert payload["clearance_executor"]["module_sha256"]
        assert payload["clearance_executor"]["implementation_head"]
        assert payload["clearance_executor"]["implementation_tree"]


@pytest.mark.parametrize(
    ("approval_kind", "expected_reason"),
    [
        ("wrong_full", "operator_commit_approval_mismatch"),
        ("abbreviated", "approved_commit_not_full_oid"),
    ],
)
def test_wrong_or_abbreviated_approval_retains_lock(
    tmp_path: Path,
    approval_kind: str,
    expected_reason: str,
) -> None:
    fixture = _journal_fixture(tmp_path)
    approval = (
        fixture.before_head
        if approval_kind == "wrong_full"
        else fixture.untrusted_commit[:12]
    )

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _apply(fixture, approved_commits=[approval])

    assert _reason(exc_info) == expected_reason
    assert fixture.lock_path.exists()
    assert not fixture.receipt_dir.exists() or not list(
        fixture.receipt_dir.glob("*.json")
    )


def test_dirty_repository_refuses_clearance_and_retains_lock(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    (fixture.repo / "unrelated.tmp").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _apply(fixture)

    assert _reason(exc_info) == "repository_dirty"
    assert fixture.lock_path.exists()


@pytest.mark.parametrize("drift", ["digest", "review"])
def test_lock_digest_or_review_drift_retains_lock(
    tmp_path: Path,
    drift: str,
) -> None:
    fixture = _journal_fixture(tmp_path)
    overrides: dict[str, object] = {}
    if drift == "digest":
        overrides["expected_lock_sha256"] = "0" * 64
        expected_reason = "lock_digest_mismatch"
    else:
        unrelated = fixture.repo / "unrelated.txt"
        unrelated.write_text("new clean commit\n", encoding="utf-8")
        _git(fixture.repo, "add", "unrelated.txt")
        _git(fixture.repo, "commit", "-m", "unrelated head movement")
        expected_reason = "review_identity_mismatch"

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _apply(fixture, **overrides)

    assert _reason(exc_info) == expected_reason
    assert fixture.lock_path.exists()


def test_absent_lock_without_receipts_is_not_treated_as_success(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    fixture.lock_path.unlink()

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _apply(fixture)

    assert _reason(exc_info) == "lock_absent_without_authorization"
    assert not fixture.receipt_dir.exists() or not list(
        fixture.receipt_dir.glob("*.json")
    )


def test_rotated_authorized_lease_resumes_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _journal_fixture(tmp_path)
    real_inspect = clearance._inspect_valid_journal
    calls = 0

    def interrupt_after_rotation(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated process interruption")
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(
        clearance,
        "_inspect_valid_journal",
        interrupt_after_rotation,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated process"):
        _apply(fixture)

    rotated = json.loads(fixture.lock_path.read_text(encoding="utf-8"))
    assert rotated["lease_id"] != fixture.lease_id
    assert rotated["protected_recovery_clearance"]["decision"] == (
        "approved_protected_history_only"
    )
    assert len(list(fixture.receipt_dir.glob("*authorization*.json"))) == 1

    monkeypatch.setattr(clearance, "_inspect_valid_journal", real_inspect)
    resumed = _apply(fixture)

    assert resumed["released"] is True
    assert not fixture.lock_path.exists()
    assert len(list(fixture.receipt_dir.glob("*.json"))) == 3


def test_exact_missing_fence_restoration_is_durable_then_normally_cleared(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()

    restored = _restore(fixture, snapshot_bytes)

    assert restored["restored"] is True
    assert restored["restored_exact_snapshot"] is True
    assert restored["authority"] == {
        "completion_authority": False,
        "verification_authority": False,
        "release_authority": False,
        "production_promotion_authority": False,
    }
    assert fixture.lock_path.read_bytes() == snapshot_bytes
    restoration_receipts = sorted(
        fixture.receipt_dir.glob("*restoration-*.json")
    )
    assert len(restoration_receipts) == 2
    assert {
        json.loads(path.read_text(encoding="utf-8"))["schema"]
        for path in restoration_receipts
    } == {
        clearance.RESTORATION_AUTHORIZATION_SCHEMA,
        clearance.RESTORATION_RECEIPT_SCHEMA,
    }

    fresh_review = restored["fresh_review"]
    assert isinstance(fresh_review, dict)
    assert fresh_review["review_id"] != fixture.review["review_id"]
    assert fresh_review["lock_sha256"] == fixture.review["lock_sha256"]
    assert fresh_review["lease_id"] == fixture.lease_id
    released = _apply(
        fixture,
        expected_review_id=fresh_review["review_id"],
    )
    assert released["released"] is True
    assert not fixture.lock_path.exists()
    assert len(list(fixture.receipt_dir.glob("*.json"))) == 5


def test_initial_existing_fence_is_not_restoration_success(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _restore(fixture, snapshot_bytes)

    assert _reason(exc_info) == "restoration_lock_already_present"
    assert fixture.lock_path.read_bytes() == snapshot_bytes
    assert not fixture.receipt_dir.exists() or not list(
        fixture.receipt_dir.glob("*.json")
    )


def test_restoration_rejects_tampered_source_review(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()
    tampered = dict(fixture.review)
    tampered["guard_id"] = "baguqeera-forged"

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _restore(fixture, snapshot_bytes, source_review=tampered)

    assert _reason(exc_info) == "source_review_identity_mismatch"
    assert not fixture.lock_path.exists()
    assert not fixture.receipt_dir.exists() or not list(
        fixture.receipt_dir.glob("*.json")
    )


def test_repeat_disappearance_requires_a_new_operator_event(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()
    _restore(fixture, snapshot_bytes)
    fixture.lock_path.unlink()

    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _restore(fixture, snapshot_bytes)

    assert _reason(exc_info) == (
        "restoration_repeat_absence_requires_new_operator_event"
    )
    assert not fixture.lock_path.exists()


@pytest.mark.parametrize(
    "failure_phase",
    [
        "after_temp_open",
        "during_temp_write",
        "after_temp_fsync",
        "after_publish_before_directory_fsync",
    ],
)
def test_restoration_failure_never_exposes_partial_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()

    def interrupt(phase: str) -> None:
        if phase == failure_phase:
            raise RuntimeError(f"interrupt:{phase}")

    monkeypatch.setattr(clearance, "_restoration_test_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="interrupt:"):
        _restore(fixture, snapshot_bytes)

    if fixture.lock_path.exists():
        assert fixture.lock_path.read_bytes() == snapshot_bytes
        assert fixture.lock_path.stat().st_nlink == 1
    monkeypatch.setattr(
        clearance,
        "_restoration_test_checkpoint",
        lambda _phase: None,
    )
    resumed = _restore(fixture, snapshot_bytes)
    assert resumed["restored"] is True
    assert fixture.lock_path.read_bytes() == snapshot_bytes


def test_restoration_never_overwrites_an_intervening_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()
    intruder = b'{"lease_id":"intervening"}'
    real_link = os.link

    def racing_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(destination) == fixture.lock_path and str(source).endswith(
            ".restoring"
        ):
            fixture.lock_path.write_bytes(intruder)
            fixture.lock_path.chmod(0o600)
            raise FileExistsError("injected publication race")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(clearance.os, "link", racing_link)
    with pytest.raises(clearance.ProtectedRecoveryClearanceError) as exc_info:
        _restore(fixture, snapshot_bytes)

    assert _reason(exc_info) == "restoration_lock_race"
    assert fixture.lock_path.read_bytes() == intruder


def test_restore_cli_reads_only_owner_private_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()
    operator_dir = tmp_path / "operator-input"
    operator_dir.mkdir(mode=0o700)
    snapshot_path = operator_dir / "captured-lock.json"
    review_path = operator_dir / "captured-review.json"
    snapshot_path.write_bytes(snapshot_bytes)
    review_path.write_text(
        json.dumps(fixture.review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_path.chmod(0o600)
    review_path.chmod(0o600)

    returncode = clearance.main(
        [
            "restore",
            "--repo-root",
            str(fixture.repo),
            "--receipt-dir",
            str(fixture.receipt_dir),
            "--snapshot-path",
            str(snapshot_path),
            "--source-review-path",
            str(review_path),
            "--expected-lease-id",
            fixture.lease_id,
            "--expected-lock-sha256",
            str(fixture.review["lock_sha256"]),
            "--operator-event-id",
            "fixture-cli-disappearance-0001",
            "--approve-commit",
            fixture.untrusted_commit,
            "--operator-identity",
            "fixture-operator",
            "--operator-note",
            "Exact fence restoration only; no completion authority.",
        ]
    )

    assert returncode == 0
    output = json.loads(capsys.readouterr().out)
    assert output["restored"] is True
    assert fixture.lock_path.read_bytes() == snapshot_bytes


def test_restore_cli_rejects_symlink_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()
    operator_dir = tmp_path / "operator-input"
    operator_dir.mkdir(mode=0o700)
    actual_snapshot = operator_dir / "actual-lock.json"
    snapshot_link = operator_dir / "captured-lock.json"
    review_path = operator_dir / "captured-review.json"
    actual_snapshot.write_bytes(snapshot_bytes)
    actual_snapshot.chmod(0o600)
    snapshot_link.symlink_to(actual_snapshot)
    review_path.write_text(
        json.dumps(fixture.review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_path.chmod(0o600)

    returncode = clearance.main(
        [
            "restore",
            "--repo-root",
            str(fixture.repo),
            "--receipt-dir",
            str(fixture.receipt_dir),
            "--snapshot-path",
            str(snapshot_link),
            "--source-review-path",
            str(review_path),
            "--expected-lease-id",
            fixture.lease_id,
            "--expected-lock-sha256",
            str(fixture.review["lock_sha256"]),
            "--operator-event-id",
            "fixture-cli-disappearance-0002",
            "--approve-commit",
            fixture.untrusted_commit,
            "--operator-identity",
            "fixture-operator",
            "--operator-note",
            "Exact fence restoration only; no completion authority.",
        ]
    )

    assert returncode == 2
    error = json.loads(capsys.readouterr().err)
    assert error["restored"] is False
    assert error["reason"] == "operator_input_unreadable"
    assert not fixture.lock_path.exists()


def test_restoration_repairs_crash_hardlink_before_final_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()

    def interrupt(phase: str) -> None:
        if phase == "after_temp_fsync":
            raise RuntimeError("stop before publication")

    monkeypatch.setattr(clearance, "_restoration_test_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="stop before publication"):
        _restore(fixture, snapshot_bytes)
    authorization_path = next(
        fixture.receipt_dir.glob("*restoration-authorization*.json")
    )
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    suffix = clearance.hashlib.sha256(
        authorization["restoration_authorization_id"].encode("utf-8")
    ).hexdigest()
    staging = fixture.lock_path.with_name(
        f".{fixture.lock_path.name}.{suffix}.restoring"
    )
    staging.write_bytes(snapshot_bytes)
    staging.chmod(0o600)
    os.link(staging, fixture.lock_path)
    assert fixture.lock_path.stat().st_nlink == 2

    monkeypatch.setattr(
        clearance,
        "_restoration_test_checkpoint",
        lambda _phase: None,
    )
    result = _restore(fixture, snapshot_bytes)

    assert result["restored"] is True
    assert fixture.lock_path.read_bytes() == snapshot_bytes
    assert fixture.lock_path.stat().st_nlink == 1
    assert not staging.exists()


def test_restoration_discards_only_safe_partial_staging_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()

    def interrupt(phase: str) -> None:
        if phase == "after_temp_fsync":
            raise RuntimeError("stop after authorization")

    monkeypatch.setattr(clearance, "_restoration_test_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="stop after authorization"):
        _restore(fixture, snapshot_bytes)
    authorization_path = next(
        fixture.receipt_dir.glob("*restoration-authorization*.json")
    )
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    suffix = clearance.hashlib.sha256(
        authorization["restoration_authorization_id"].encode("utf-8")
    ).hexdigest()
    staging = fixture.lock_path.with_name(
        f".{fixture.lock_path.name}.{suffix}.restoring"
    )
    staging.write_bytes(snapshot_bytes[:37])
    staging.chmod(0o600)

    monkeypatch.setattr(
        clearance,
        "_restoration_test_checkpoint",
        lambda _phase: None,
    )
    result = _restore(fixture, snapshot_bytes)

    assert result["restored"] is True
    assert fixture.lock_path.read_bytes() == snapshot_bytes
    assert fixture.lock_path.stat().st_nlink == 1
    assert not staging.exists()


def test_receipt_publication_repairs_interrupted_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _journal_fixture(tmp_path)
    snapshot_bytes = fixture.lock_path.read_bytes()
    fixture.lock_path.unlink()

    def interrupt(phase: str) -> None:
        if phase == "after_temp_fsync":
            raise RuntimeError("stop after authorization")

    monkeypatch.setattr(clearance, "_restoration_test_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="stop after authorization"):
        _restore(fixture, snapshot_bytes)
    authorization_path = next(
        fixture.receipt_dir.glob("*restoration-authorization*.json")
    )
    pending = authorization_path.with_name(
        f".{authorization_path.name}.crash.pending"
    )
    os.link(authorization_path, pending)
    assert authorization_path.stat().st_nlink == 2

    monkeypatch.setattr(
        clearance,
        "_restoration_test_checkpoint",
        lambda _phase: None,
    )
    result = _restore(fixture, snapshot_bytes)

    assert result["restored"] is True
    assert authorization_path.stat().st_nlink == 1
    assert not pending.exists()


def test_literal_pathspec_magic_filename_is_reviewed(
    tmp_path: Path,
) -> None:
    fixture = _journal_fixture(
        tmp_path,
        protected_path=":(exclude)victim.todo.md",
    )

    assert fixture.review["eligible"] is True
    assert fixture.review["untrusted_commits"] == [
        fixture.untrusted_commit
    ]
    assert fixture.review["history"][0]["changed_protected_paths"] == [
        ":(exclude)victim.todo.md"
    ]


def test_merge_identical_to_in_scope_parent_is_audited_but_not_oversealed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "corridor-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    paths = ["a.todo.md", "b.json"]
    (repo / paths[0]).write_text("Status: todo\n", encoding="utf-8")
    (repo / paths[1]).write_text('{"value": 0}\n', encoding="utf-8")
    _git(repo, "add", *paths)
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "side")
    (repo / paths[1]).write_text('{"value": 1}\n', encoding="utf-8")
    _git(repo, "add", paths[1])
    _git(repo, "commit", "-m", "off-corridor protected change")
    _git(repo, "checkout", "main")
    (repo / paths[0]).write_text("Status: completed\n", encoding="utf-8")
    _git(repo, "add", paths[0])
    _git(repo, "commit", "-m", "review baseline")
    before_head = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "merge",
        "--no-ff",
        "-s",
        "ours",
        "side",
        "-m",
        "audit-only side merge",
    )
    merge_commit = _git(repo, "rev-parse", "HEAD")

    guard = _identified(
        {
            "protected_paths": paths,
            "scopes": [
                {
                    "git_root": str(repo.resolve()),
                    "paths": paths,
                    "before_head": before_head,
                    "before_head_query": {
                        "ok": True,
                        "head": before_head,
                        "unborn": False,
                    },
                }
            ],
            "discovery_errors": [],
        },
        "guard_id",
    )
    intent = _identified(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "supervisor-protected-recovery-intent@1"
            ),
            "operation": "generated_board_update",
            "producer": "fixture-producer",
            "protected_paths": paths,
            "guard_id": guard["guard_id"],
        },
        "intent_id",
    )
    metadata = {
        "attempt": 0,
        "branch": "",
        "kind": "merge",
        "lease_id": content_identity(
            {"kind": "corridor-fixture", "head": merge_commit, "base": base}
        ),
        "operation": "generated_dirty_repair",
        "owner_script": "implementation_supervisor.py",
        "pid": 2_147_483_647,
        "producer": "fixture-producer",
        "protected_paths": paths,
        "protected_recovery_intent": intent,
        "protected_recovery_owner": "implementation_supervisor",
        "protected_recovery_required": True,
        "protected_release_guard": guard,
        "repo_root": str(repo.resolve()),
        "worktree_root": str(repo.resolve()),
        "repository_id": checkout_repository_id(repo),
        "state_dir": str((repo / "data" / "state").resolve()),
        "state_path": str((repo / "data" / "state" / "lane.json").resolve()),
        "task_id": "",
    }
    lock_path = checkout_mutation_lock_path(repo)
    lock_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lock_path.chmod(0o600)

    review = clearance.inspect_protected_recovery(repo)

    assert review["eligible"] is False
    assert review["reason"] == "protected_outputs_clean_history_unchanged"
    assert review["range_oids"] == [merge_commit]
    assert review["history"] == []
    merge_record = review["range_records"][0]
    assert merge_record["commit"] == merge_commit
    assert merge_record["protected_relevant"] is False
    assert any(
        delta["parent_in_scope"] is False and delta["changed"] is True
        for delta in merge_record["protected_parent_deltas"]
    )


def test_merge_history_enumeration_binds_every_parent_and_protected_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "merge-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    paths = ["a.todo.md", "b.json"]
    (repo / paths[0]).write_text("Status: todo\n", encoding="utf-8")
    (repo / paths[1]).write_text('{"value": 0}\n', encoding="utf-8")
    _git(repo, "add", *paths)
    _git(repo, "commit", "-m", "base")
    before_head = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "side")
    (repo / paths[1]).write_text('{"value": 1}\n', encoding="utf-8")
    _git(repo, "add", paths[1])
    _git(repo, "commit", "-m", "side protected change")
    side_commit = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "main")
    (repo / paths[0]).write_text("Status: completed\n", encoding="utf-8")
    _git(repo, "add", paths[0])
    _git(repo, "commit", "-m", "main protected change")
    main_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "merge", "--no-ff", "side", "-m", "merge protected change")
    merge_commit = _git(repo, "rev-parse", "HEAD")

    guard = _identified(
        {
            "protected_paths": paths,
            "scopes": [
                {
                    "git_root": str(repo.resolve()),
                    "paths": paths,
                    "before_head": before_head,
                    "before_head_query": {
                        "ok": True,
                        "head": before_head,
                        "unborn": False,
                    },
                }
            ],
            "discovery_errors": [],
        },
        "guard_id",
    )
    intent = _identified(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "supervisor-protected-recovery-intent@1"
            ),
            "operation": "generated_board_update",
            "producer": "fixture-producer",
            "protected_paths": paths,
            "guard_id": guard["guard_id"],
        },
        "intent_id",
    )
    metadata = {
        "attempt": 0,
        "branch": "",
        "kind": "merge",
        "lease_id": content_identity(
            {"kind": "merge-history-fixture", "head": merge_commit}
        ),
        "operation": "generated_dirty_repair",
        "owner_script": "implementation_supervisor.py",
        "pid": 2_147_483_647,
        "producer": "fixture-producer",
        "protected_paths": paths,
        "protected_recovery_intent": intent,
        "protected_recovery_owner": "implementation_supervisor",
        "protected_recovery_required": True,
        "protected_release_guard": guard,
        "repo_root": str(repo.resolve()),
        "worktree_root": str(repo.resolve()),
        "repository_id": checkout_repository_id(repo),
        "state_dir": str((repo / "data" / "state").resolve()),
        "state_path": str((repo / "data" / "state" / "lane.json").resolve()),
        "task_id": "",
    }
    lock_path = checkout_mutation_lock_path(repo)
    lock_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lock_path.chmod(0o600)

    review = clearance.inspect_protected_recovery(repo)

    assert review["eligible"] is True, review
    assert set(review["untrusted_commits"]) == {
        side_commit,
        main_commit,
        merge_commit,
    }
    records = {item["commit"]: item for item in review["history"]}
    merge_record = records[merge_commit]
    assert len(merge_record["parents"]) == 2
    assert len(merge_record["protected_parent_deltas"]) == 2
    assert set(merge_record["changed_protected_paths"]) == set(paths)
    assert set(review["aggregate_protected_delta"]["changed_protected_paths"]) == (
        set(paths)
    )
