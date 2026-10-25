"""Read-only proof that a retained rescue belongs to a completed older attempt.

This classification never merges, deletes, retires, completes or retries work.
Its only effect is to keep an authenticated obsolete rescue out of merge
preflight. Missing, rotated-away or ambiguous provenance keeps normal behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

from .current_incident_adjudication import _event_rows, _read


def _events(path: Path) -> list[dict]:
    # Native's owned, no-follow, bounded reader and exact canonical hash chain.
    # Do not call _iter_events(): that reader may repair a durable journal.
    return _event_rows(_read(path))


def _worktree_custody(supervisor, daemon, path: Path, branch: str, commit: str) -> bool:
    common = supervisor._verified_git_worktree_common_dir(path)
    if common is None or common != supervisor._verified_git_worktree_common_dir(supervisor.config.repo_root):
        return False
    result = daemon._run_git(["rev-parse", "--absolute-git-dir"], cwd=path)
    if result.returncode != 0:
        return False
    admin = Path(result.stdout.strip())
    if not admin.is_absolute() or admin.resolve(strict=True) != admin:
        return False
    actual_head = daemon._run_git(["rev-parse", "--verify", "HEAD^{commit}"], cwd=path)
    actual_branch = daemon._run_git(["symbolic-ref", "--quiet", "HEAD"], cwd=path)
    if (actual_head.returncode != 0 or actual_head.stdout.strip() != commit
            or actual_branch.returncode != 0 or actual_branch.stdout.strip() != "refs/heads/" + branch):
        return False
    pointer, backlink = _read(path / ".git"), _read(admin / "gitdir")
    if len(pointer) > 4096 or len(backlink) > 4096 or not pointer.startswith(b"gitdir: "):
        return False
    target = Path(pointer.decode().strip().removeprefix("gitdir: "))
    target = target if target.is_absolute() else path / target
    return target.resolve(strict=True) == admin and Path(backlink.decode().strip()).resolve(strict=True) == path / ".git"


def _rescue_original(supervisor, daemon, candidate, records, events):
    branch, commit = candidate["branch"], candidate["head"]
    path, root = Path(candidate["path"]), supervisor.config.worktree_root
    if (not branch.startswith("rescue/worktree/") or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or not path.is_absolute() or path.parent != root or path.resolve(strict=True) != path
            or root.resolve(strict=True) != root or supervisor._git_status_short(path)):
        return None
    matches = [row for row in records if row.get("worktree") == str(path)]
    if (len(matches) != 1 or str(matches[0].get("branch") or "").removeprefix("refs/heads/") != branch
            or matches[0].get("HEAD") != commit or not _worktree_custody(supervisor, daemon, path, branch, commit)):
        return None
    size = daemon._run_git(["cat-file", "-s", commit], cwd=path)
    if size.returncode != 0 or not size.stdout.strip().isdecimal() or int(size.stdout.strip()) > 65536:
        return None
    raw = daemon._run_git(["cat-file", "commit", commit], cwd=path)
    if raw.returncode != 0 or len(raw.stdout.encode()) > 65536:
        return None
    header, separator, body = raw.stdout.partition("\n\n")
    parents = [line[7:] for line in header.splitlines() if line.startswith("parent ")]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if (not separator or len(parents) != 1 or len(lines) != 4
            or not all(any(line.startswith(kind + " ") and " <implementation-supervisor@example.invalid> " in line
                for line in header.splitlines()) for kind in ("author", "committer"))
            or not lines[1].startswith("Original branch: ")
            or lines[2] != "Original HEAD: " + parents[0]
            or lines[3] not in {"Cleanup reason: stale_active_execution_auto_rescue",
                "Cleanup reason: reconciliation_uncommitted_integrated_base"}):
        return None
    original = lines[1].removeprefix("Original branch: ")
    if not original.startswith("implementation/") or lines[0] != "Rescue dirty worktree " + original:
        return None
    proof = [event for event in events if event.get("type") == "dirty_worktree_rescued"
        and event.get("preserved") is True and type(event.get("returncode")) is int
        and event["returncode"] == 0 and event.get("path") == str(path)
        and event.get("target_ref") == candidate["target_ref"]
        and event.get("branch") == original and event.get("head") == parents[0]
        and event.get("rescue_branch") == branch and event.get("rescue_commit") == commit]
    return (original, parents[0], proof[0]["event_id"]) if len(proof) == 1 else None


def _integrated_winner(supervisor, event, task_id, binding, target_ref):
    merge, validation = event.get("merge_result"), event.get("validation_result")
    if (not isinstance(merge, Mapping) or not isinstance(validation, Mapping)
            or validation.get("passed") is not True or type(validation.get("returncode")) is not int
            or validation["returncode"] != 0 or merge.get("merged") is not True
            or merge.get("attempted") is not True or type(merge.get("returncode")) is not int
            or merge["returncode"] != 0 or merge.get("target_branch") != target_ref
            or merge.get("completion_task_cids") != {task_id: binding["canonical_task_cid"]}
            or any(merge.get(key) != value for key, value in binding.items() if key != "board_namespace")):
        return False
    proof = merge.get("integration_commit_proof")
    if not isinstance(proof, Mapping) or proof.get("passed") is not True or proof.get("exact_two_parent_merge") is not True:
        return False
    winner, integrated, base = event.get("implementation_commit"), merge.get("merge_commit"), merge.get("integration_base_commit")
    if (any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (winner, integrated, base))
            or merge.get("implementation_commit") != winner or proof.get("implementation_commit") != winner
            or proof.get("integration_commit") != integrated or proof.get("integration_ref") != integrated
            or proof.get("integration_base_commit") != base or proof.get("target_branch") != target_ref
            or supervisor._git_commit_parents(supervisor.config.repo_root, integrated) != [base, winner]
            or not supervisor._git_ref_is_ancestor(supervisor.config.repo_root, integrated, target_ref)):
        return False
    return True


def completed_retained_candidate(supervisor, candidate: Mapping) -> dict | None:
    """Reprove current canonical completion before attempting merge preflight.

    Branch spelling is never task authority. The original task is recovered
    only from the current campaign's authenticated Native start event. This
    proof grants no completion, retry, provider or lifecycle authority.
    """
    from .implementation_daemon import PortalImplementationDaemon
    from .implementation_supervisor import PortalImplementationSupervisor
    try:
        config = supervisor.config
        if (not isinstance(candidate, Mapping) or set(candidate) - {
                "path", "branch", "head", "target_ref", "nonblocking_main_gitlinks"}
                or not all(isinstance(candidate.get(key), str) and candidate[key]
                    for key in ("path", "branch", "head", "target_ref"))
                or candidate["target_ref"] != (config.merge_target_branch
                    or supervisor._git_current_branch(config.repo_root) or "HEAD")
                or not candidate["branch"].startswith("rescue/worktree/")):
            return None
        # The canonical disabled constructor does not register/save task IDs.
        # Host subclass measured-daemon factories are not needed for this read.
        daemon = PortalImplementationSupervisor._build_worktree_reconciliation_daemon(supervisor)
        if not isinstance(daemon, PortalImplementationDaemon) or daemon.implement is not False:
            return None
        native_path = config.state_dir / f"{config.state_prefix}_events.jsonl"
        native_events, rescue_events = _events(native_path), _events(config.events_path)
        records = supervisor._git_worktree_records(config.repo_root)
        if len(records) > 256:
            return None
        attested = _rescue_original(supervisor, daemon, candidate, records, rescue_events)
        if attested is None:
            return None
        original, baseline, rescue_event_id = attested
        starts = [event for event in native_events if event.get("type") == "implementation_started"
            and event.get("branch") == original and event.get("baseline_ref") == baseline
            and event.get("worktree_path") == candidate["path"]]
        if len(starts) != 1:
            return None
        started = starts[0]
        task_id = started.get("task_id")
        tasks = [task for task in daemon._load_tasks() if task.task_id == task_id]
        if len(tasks) != 1:
            return None
        identity = daemon._identity_for_task(tasks[0])
        binding = {"canonical_task_cid": identity.canonical_task_cid,
            "canonical_task_key": identity.canonical_task_key, "board_namespace": identity.board_namespace}
        if (not all(binding.values()) or any(started.get(key) != value for key, value in binding.items())
                or type(started.get("attempt")) is not int or started["attempt"] < 1):
            return None
        completed = [event for event in native_events if event.get("type") == "implementation_finished"
            and event.get("task_id") == task_id and all(event.get(key) == value for key, value in binding.items())
            and event["sequence"] > started["sequence"]
            and type(event.get("attempt")) is int and event["attempt"] > started["attempt"]
            and type(event.get("returncode")) is int and event["returncode"] == 0
            and _integrated_winner(supervisor, event, task_id, binding, candidate["target_ref"])]
        status_binding = {"task": {"task_id": task_id},
            "completion_task_cids": {task_id: identity.canonical_task_cid}}
        if not completed or PortalImplementationDaemon._completed_task_binding_error(daemon, status_binding):
            return None
        # Repeat custody, current definition/status and winning ancestry checks
        # immediately before the classification. Append-only history is okay.
        current_native = _events(native_path)
        winner = completed[-1]
        if (not any(event == started for event in current_native)
                or not any(event == winner for event in current_native)
                or _rescue_original(supervisor, daemon, candidate,
                    supervisor._git_worktree_records(config.repo_root), _events(config.events_path)) != attested
                or PortalImplementationDaemon._completed_task_binding_error(daemon, status_binding)
                or not _integrated_winner(supervisor, winner, task_id, binding, candidate["target_ref"])):
            return None
        return {"task_id": task_id, **binding, "original_start_event_id": started["event_id"],
            "rescue_event_id": rescue_event_id, "winning_finish_event_id": winner["event_id"],
            "winning_integration_commit": winner["merge_result"]["merge_commit"],
            "completion_authority": False, "new_attempt_authority": False}
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
