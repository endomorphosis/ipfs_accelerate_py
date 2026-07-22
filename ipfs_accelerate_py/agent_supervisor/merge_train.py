"""Single-consumer merge train for autonomous implementation lanes.

The queue is the hand-off boundary between implementation workers and the target
checkout.  This module deliberately owns the *only* dequeue-and-merge critical
section: requests are deduplicated, rebased on the target tip observed inside
that section, and either completed, retried a bounded number of times, or
quarantined by :class:`~.merge_queue.MergeQueue`.

The built-in merger rebases in a detached temporary worktree, then advances the
target with compare-and-swap semantics.  A checked-out target is fast-forwarded
only while its worktree is clean, so its index is never left stale.  A daemon
can supply ``merge_callback`` to reuse a more specialised merger (for example
one that coordinates nested submodules); the callback is still invoked under
the same repo-wide single-consumer lease.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .merge_queue import MergeQueue, MergeRequest


MergeCallback = Callable[[MergeRequest], Mapping[str, Any]]


def _request_value(request: MergeRequest, name: str, *metadata_names: str) -> str:
    """Read a request field while remaining compatible with v1 queue records."""

    value = getattr(request, name, "")
    if value:
        return str(value)
    metadata = getattr(request, "metadata", {})
    if isinstance(metadata, Mapping):
        for candidate in metadata_names:
            value = metadata.get(candidate)
            if value:
                return str(value)
    return ""


def conflict_fingerprint(
    *,
    canonical_task_id: str,
    candidate_commit: str,
    target_commit: str,
    conflict_index: str,
) -> str:
    """Return a stable identity for one concrete rebase conflict.

    ``git ls-files -u`` includes path, mode, object and stage.  Hashing it with
    both tips distinguishes genuinely changed conflicts without incorporating a
    volatile request id or retry counter.
    """

    payload = "\0".join(
        (canonical_task_id, candidate_commit, target_commit, conflict_index)
    )
    return hashlib.sha256(payload.encode("utf-8", errors="surrogateescape")).hexdigest()


class MergeTrain:
    """Consume queued merge candidates serially and durably.

    Args:
        repo_root: Any worktree belonging to the repository whose target ref is
            updated.
        queue: Persistent queue shared by all implementation lanes.
        target_branch: Local branch receiving candidates.
        resolver: Optional callable/object used for rebase conflicts.  Objects
            may expose ``resolve`` and, independently, ``acquire``/``release``
            methods compatible with :class:`MergeResolverRegistry`.
        max_attempts: Last failure count at which a request is quarantined.
        merge_callback: Optional specialised merger.  It receives the claimed
            request and returns a merge-result mapping.
        state_dir: Train receipts/lease/worktrees directory.  Defaults beneath
            the queue directory so independent processes converge on one lease.
    """

    def __init__(
        self,
        repo_root: Path | str,
        queue: MergeQueue,
        *,
        target_branch: str = "main",
        resolver: Any = None,
        max_attempts: int = 3,
        merge_callback: MergeCallback | None = None,
        state_dir: Path | str | None = None,
        git_timeout_seconds: float = 600.0,
        owner_id: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.queue = queue
        self.target_branch = str(target_branch or "main")
        self.resolver = resolver
        self.max_attempts = max(1, int(max_attempts))
        self.merge_callback = merge_callback
        queue_dir = Path(getattr(queue, "queue_dir", self.repo_root / ".merge-queue"))
        self.state_dir = Path(state_dir) if state_dir is not None else queue_dir / "train"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.worktree_dir = self.state_dir / "worktrees"
        self.receipt_dir = self.state_dir / "receipts"
        self.worktree_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.consumer_lock_path = self.state_dir / "consumer.lock"
        self.git_timeout_seconds = max(1.0, float(git_timeout_seconds))
        self.owner_id = owner_id or f"merge-train:{os.getpid()}:{uuid.uuid4().hex}"

    @contextmanager
    def _consumer_lease(self) -> Iterator[bool]:
        """Try to acquire the process-safe, crash-releasing consumer lease."""

        fd = os.open(self.consumer_lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        acquired = False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                yield False
                return
            metadata = json.dumps(
                {"owner_id": self.owner_id, "pid": os.getpid(), "acquired_at": time.time()}
            ).encode("utf-8")
            os.ftruncate(fd, 0)
            os.write(fd, metadata)
            os.fsync(fd)
            yield True
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def run_once(self) -> dict[str, Any] | None:
        """Process at most one request.

        ``None`` means either the queue was empty or another train consumer owns
        the lease.  The latter can be distinguished using :meth:`status`.
        """

        with self._consumer_lease() as acquired:
            if not acquired:
                return None
            request = self._dequeue()
            if request is None:
                return None
            return self._process_claimed(request)

    # Widely useful aliases for supervisors that phrase one iteration as a tick.
    process_next = run_once
    consume_once = run_once

    def drain(self, max_items: int | None = None) -> list[dict[str, Any]]:
        """Drain available work while retaining one consumer lease.

        A single lease for the full batch prevents another process from slipping
        between requests and preserves the queue's priority/age order.
        """

        if max_items is not None and int(max_items) <= 0:
            return []
        results: list[dict[str, Any]] = []
        with self._consumer_lease() as acquired:
            if not acquired:
                return results
            while max_items is None or len(results) < int(max_items):
                request = self._dequeue()
                if request is None:
                    break
                results.append(self._process_claimed(request))
        return results

    run = drain

    def status(self) -> dict[str, Any]:
        queue_status = self.queue.status() if hasattr(self.queue, "status") else {}
        return {
            "owner_id": self.owner_id,
            "target_branch": self.target_branch,
            "state_dir": str(self.state_dir),
            "consumer_lock_path": str(self.consumer_lock_path),
            "queue": queue_status,
        }

    def _process_claimed(self, request: MergeRequest) -> dict[str, Any]:
        started_at = time.time()
        canonical = str(getattr(request, "canonical_identity", "") or "") or _request_value(
            request,
            "canonical_task_id",
            "canonical_task_id",
            "canonical_task_key",
            "task_cid",
        ) or _request_value(request, "task_id")
        candidate = _request_value(
            request,
            "commit_sha",
            "commit_sha",
            "implementation_commit",
            "commit",
        )
        branch = _request_value(request, "branch_name", "branch")

        if not candidate and branch:
            candidate_result = self._git("rev-parse", "--verify", f"{branch}^{{commit}}")
            if candidate_result.returncode == 0:
                candidate = candidate_result.stdout.strip()
        if not canonical or not candidate:
            return self._finish_failure(
                request,
                reason="invalid_merge_request",
                details={"canonical_task_id": canonical, "commit_sha": candidate, "branch": branch},
                started_at=started_at,
                retryable=False,
            )

        verified = self._git("rev-parse", "--verify", f"{candidate}^{{commit}}")
        if verified.returncode != 0:
            return self._finish_failure(
                request,
                reason="candidate_commit_missing",
                details={"commit_sha": candidate, "stderr": verified.stderr[-2000:]},
                started_at=started_at,
                retryable=False,
            )
        candidate = verified.stdout.strip()
        target = self._target_commit()
        if not target:
            return self._finish_failure(
                request,
                reason="target_branch_missing",
                details={"target_branch": self.target_branch},
                started_at=started_at,
            )

        # A callback owns the complete integration lifecycle, including nested
        # repository handoff and taskboard completion.  Root-level receipts or
        # ancestry only prove that the parent commit landed; they cannot safely
        # short-circuit those callback side effects after a daemon restart.
        if self.merge_callback is None:
            dedupe_key = self._dedupe_key(canonical, candidate)
            previous = self._read_receipt(dedupe_key)
            if previous and str(previous.get("status")) in {
                "merged",
                "already_merged",
                "deduplicated",
            }:
                return self._finish_success(
                    request,
                    status="deduplicated",
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                    extra={"duplicate_of": previous.get("request_id", "")},
                )

            if self._is_ancestor(candidate, target):
                return self._finish_success(
                    request,
                    status="already_merged",
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                )

        if self.merge_callback is not None:
            try:
                callback_result = dict(self.merge_callback(request) or {})
            except Exception as exc:  # callbacks are an isolation boundary
                return self._finish_failure(
                    request,
                    reason="merge_callback_exception",
                    details={"exception": f"{type(exc).__name__}: {exc}"},
                    started_at=started_at,
                )
            if callback_result.get("merged") or callback_result.get("already_merged"):
                return self._finish_success(
                    request,
                    status="merged" if callback_result.get("merged") else "already_merged",
                    canonical=canonical,
                    candidate=candidate,
                    target=self._target_commit() or target,
                    started_at=started_at,
                    extra={"merge_result": callback_result},
                )
            callback_reason = str(callback_result.get("reason") or "merge_callback_failed")
            retryable = callback_reason not in {
                "invalid_merge_request",
                "candidate_commit_missing",
                "validation_failed",
                "branch_has_no_changes",
            }
            return self._finish_failure(
                request,
                reason=callback_reason,
                details={"merge_result": callback_result},
                started_at=started_at,
                retryable=retryable,
            )

        result = self._rebase_and_integrate(
            request=request,
            canonical=canonical,
            candidate=candidate,
            target=target,
        )
        if result.get("merged"):
            return self._finish_success(
                request,
                status="merged",
                canonical=canonical,
                candidate=candidate,
                target=str(result.get("target_commit") or target),
                started_at=started_at,
                extra=result,
            )
        return self._finish_failure(
            request,
            reason=str(result.get("reason") or "merge_failed"),
            details=result,
            started_at=started_at,
            retryable=bool(result.get("retryable", True)),
        )

    def _rebase_and_integrate(
        self,
        *,
        request: MergeRequest,
        canonical: str,
        candidate: str,
        target: str,
    ) -> dict[str, Any]:
        workspace = Path(tempfile.mkdtemp(prefix="candidate-", dir=self.worktree_dir))
        added = False
        try:
            add = self._git("worktree", "add", "--detach", str(workspace), candidate)
            if add.returncode != 0:
                return self._command_failure("worktree_add_failed", add)
            added = True

            rebase = self._git("rebase", target, cwd=workspace)
            resolver_result: dict[str, Any] = {}
            if rebase.returncode != 0:
                conflicts = self._git("ls-files", "-u", cwd=workspace).stdout
                fingerprint = conflict_fingerprint(
                    canonical_task_id=canonical,
                    candidate_commit=candidate,
                    target_commit=target,
                    conflict_index=conflicts,
                )
                resolver_result = self._resolve_conflict(
                    request=request,
                    workspace=workspace,
                    fingerprint=fingerprint,
                    candidate=candidate,
                    target=target,
                    conflicts=conflicts,
                )
                if not resolver_result.get("resolved"):
                    self._git("rebase", "--abort", cwd=workspace)
                    return {
                        "merged": False,
                        "retryable": resolver_result.get("retryable", True),
                        "reason": str(resolver_result.get("reason") or "rebase_conflict"),
                        "conflict_fingerprint": fingerprint,
                        "resolver": resolver_result,
                        "stderr": rebase.stderr[-4000:],
                    }
                # A resolver may complete the rebase itself.  Otherwise all
                # conflicts must be staged before the non-interactive continue.
                if self._git("rev-parse", "-q", "--verify", "REBASE_HEAD", cwd=workspace).returncode == 0:
                    continued = self._git(
                        "-c", "core.editor=true", "rebase", "--continue", cwd=workspace
                    )
                    if continued.returncode != 0:
                        self._git("rebase", "--abort", cwd=workspace)
                        return self._command_failure(
                            "resolver_rebase_continue_failed", continued, resolver=resolver_result
                        )

            rebased = self._git("rev-parse", "HEAD", cwd=workspace)
            if rebased.returncode != 0:
                return self._command_failure("rebased_commit_missing", rebased)
            rebased_commit = rebased.stdout.strip()
            # Compare-and-swap is important even under our lease: a human or a
            # different merge mechanism may legitimately advance the branch.
            update = self._advance_target(rebased_commit, expected_target=target)
            if update.returncode != 0:
                update_reason = (
                    "target_worktree_dirty"
                    if "worktree is dirty" in str(update.stderr or "")
                    else "target_advanced"
                )
                return {
                    **self._command_failure(update_reason, update),
                    "retryable": True,
                    "rebased_commit": rebased_commit,
                }
            return {
                "merged": True,
                "rebased": rebased_commit != candidate,
                "candidate_commit": candidate,
                "rebased_commit": rebased_commit,
                "target_commit_before": target,
                "target_commit": rebased_commit,
                "merge_commit": rebased_commit,
                "resolver": resolver_result,
            }
        finally:
            if added:
                self._git("worktree", "remove", "--force", str(workspace))
            shutil.rmtree(workspace, ignore_errors=True)

    def _resolve_conflict(
        self,
        *,
        request: MergeRequest,
        workspace: Path,
        fingerprint: str,
        candidate: str,
        target: str,
        conflicts: str,
    ) -> dict[str, Any]:
        if self.resolver is None:
            return {"resolved": False, "reason": "rebase_conflict", "retryable": True}
        event = {
            "conflict_fingerprint": fingerprint,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": str(getattr(request, "canonical_identity", "") or "")
            or _request_value(request, "canonical_task_id")
            or _request_value(request, "task_id"),
            "branch": _request_value(request, "branch_name"),
            "candidate_commit": candidate,
            "source_commit": candidate,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "workspace": str(workspace),
            "unmerged_index": conflicts,
            "unmerged_paths": sorted(
                {
                    line.split("\t", 1)[1]
                    for line in conflicts.splitlines()
                    if "\t" in line and line.split("\t", 1)[1]
                }
            ),
            "reason": "rebase_conflict",
        }
        claim: Any = None
        acquired = True
        outcome: dict[str, Any] = {
            "resolved": False,
            "reason": "resolver_did_not_run",
        }
        if hasattr(self.resolver, "acquire"):
            claim = self._call_compatible(
                self.resolver.acquire,
                event,
                owner_id=self.owner_id,
                fingerprint=fingerprint,
            )
            acquired = claim is not None and claim is not False
        if not acquired:
            return {"resolved": False, "reason": "resolver_already_active", "retryable": True}
        try:
            callback = getattr(self.resolver, "resolve", None)
            if callback is None and callable(self.resolver):
                callback = self.resolver
            if callback is None:
                outcome = {
                    "resolved": False,
                    "reason": "resolver_not_configured",
                    "retryable": True,
                }
                return outcome
            raw = self._call_compatible(
                callback,
                event,
                request=request,
                workspace=workspace,
                conflict_fingerprint=fingerprint,
                claim=claim,
            )
            if isinstance(raw, Mapping):
                outcome = dict(raw)
                outcome.setdefault(
                    "resolved", bool(outcome.get("applied") or outcome.get("completed"))
                )
                return outcome
            outcome = {"resolved": bool(raw)}
            return outcome
        except Exception as exc:
            outcome = {
                "resolved": False,
                "reason": "resolver_exception",
                "exception": f"{type(exc).__name__}: {exc}",
                "retryable": True,
            }
            return outcome
        finally:
            if claim is not None and hasattr(self.resolver, "release"):
                try:
                    self._call_compatible(
                        self.resolver.release,
                        claim,
                        event=event,
                        owner_id=self.owner_id,
                        succeeded=bool(outcome.get("resolved")),
                        outcome=outcome,
                        error=str(outcome.get("reason") or ""),
                    )
                except Exception:
                    pass

    @staticmethod
    def _call_compatible(callback: Callable[..., Any], positional: Any, **kwargs: Any) -> Any:
        """Invoke callbacks while filtering optional adapter keywords."""

        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(positional, **kwargs)
        accepts_kwargs = any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        filtered = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        positional_parameters = [
            item
            for item in signature.parameters.values()
            if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if positional_parameters:
            return callback(positional, **filtered)
        return callback(**filtered)

    def _finish_success(
        self,
        request: MergeRequest,
        *,
        status: str,
        canonical: str,
        candidate: str,
        target: str,
        started_at: float,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "merged": status == "merged",
            "integrated": True,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": canonical,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "merge_commit": target,
            "started_at": started_at,
            "finished_at": time.time(),
        }
        if extra:
            result.update(extra)
            # Stable public semantics take precedence over callback internals.
            result.update({"status": status, "integrated": True})
        self._write_receipt(self._dedupe_key(canonical, candidate), result)
        self.queue.complete(request)
        return result

    def _finish_failure(
        self,
        request: MergeRequest,
        *,
        reason: str,
        details: Mapping[str, Any],
        started_at: float,
        retryable: bool = True,
    ) -> dict[str, Any]:
        failures = int(getattr(request, "failure_count", 0) or 0) + 1
        exhausted = not retryable or failures >= self.max_attempts
        result = {
            "status": "quarantined" if exhausted else "retrying",
            "merged": False,
            "integrated": False,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": str(getattr(request, "canonical_identity", "") or "")
            or _request_value(request, "canonical_task_id")
            or _request_value(request, "task_id"),
            "commit_sha": _request_value(request, "commit_sha", "implementation_commit", "commit"),
            "target_branch": self.target_branch,
            "reason": reason,
            "failure_count": failures,
            "max_attempts": self.max_attempts,
            "retryable": retryable,
            "started_at": started_at,
            "finished_at": time.time(),
            **dict(details),
        }
        if exhausted:
            quarantine = getattr(self.queue, "quarantine", None)
            if quarantine is not None:
                self._call_queue_failure(quarantine, request, reason, result)
            else:
                self._call_queue_failure(self.queue.fail, request, reason, result, retryable=False)
            self._write_receipt(f"quarantine-{request.request_id}", result)
        else:
            requeue = getattr(self.queue, "requeue")
            self._call_queue_failure(requeue, request, reason, result)
        return result

    @staticmethod
    def _call_queue_failure(
        callback: Callable[..., Any],
        request: MergeRequest,
        reason: str,
        receipt: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        try:
            signature = inspect.signature(callback)
            supported = signature.parameters
            call_kwargs: dict[str, Any] = {}
            if "reason" in supported:
                call_kwargs["reason"] = reason
            if "receipt" in supported:
                call_kwargs["receipt"] = dict(receipt)
            if "details" in supported:
                call_kwargs["details"] = dict(receipt)
            call_kwargs.update({key: value for key, value in kwargs.items() if key in supported})
            callback(request, **call_kwargs)
        except (TypeError, ValueError):
            callback(request, reason)

    def _target_commit(self) -> str:
        result = self._git("rev-parse", "--verify", f"refs/heads/{self.target_branch}^{{commit}}")
        return result.stdout.strip() if result.returncode == 0 else ""

    def _advance_target(
        self,
        rebased_commit: str,
        *,
        expected_target: str,
    ) -> subprocess.CompletedProcess[str]:
        """Fast-forward the target without leaving a checked-out tree stale."""

        target_worktree = self._target_worktree()
        if target_worktree is None:
            return self._git(
                "update-ref",
                f"refs/heads/{self.target_branch}",
                rebased_commit,
                expected_target,
            )
        status = self._git("status", "--porcelain", "--untracked-files=normal", cwd=target_worktree)
        if status.returncode != 0:
            return status
        if status.stdout.strip():
            return subprocess.CompletedProcess(
                ["git", "status", "--porcelain"],
                2,
                stdout=status.stdout,
                stderr=f"target worktree is dirty: {target_worktree}",
            )
        current = self._git("rev-parse", "HEAD", cwd=target_worktree)
        if current.returncode != 0:
            return current
        if current.stdout.strip() != expected_target:
            return subprocess.CompletedProcess(
                ["git", "rev-parse", "HEAD"],
                3,
                stdout=current.stdout,
                stderr="target advanced while candidate was rebased",
            )
        return self._git("merge", "--ff-only", rebased_commit, cwd=target_worktree)

    def _target_worktree(self) -> Path | None:
        """Return the worktree currently holding the target branch, if any."""

        result = self._git("worktree", "list", "--porcelain")
        if result.returncode != 0:
            return None
        path: Path | None = None
        for line in [*result.stdout.splitlines(), ""]:
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ").strip())
            elif line == f"branch refs/heads/{self.target_branch}" and path is not None:
                return path
            elif not line:
                path = None
        return None

    def _dequeue(self) -> MergeRequest | None:
        """Claim with a consumer id when supported by the queue implementation."""

        try:
            signature = inspect.signature(self.queue.dequeue)
            if "consumer_id" in signature.parameters:
                return self.queue.dequeue(consumer_id=self.owner_id)
        except (TypeError, ValueError):
            pass
        return self.queue.dequeue()

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self._git("merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

    @staticmethod
    def _dedupe_key(canonical: str, commit: str) -> str:
        return hashlib.sha256(f"{canonical}\0{commit}".encode("utf-8")).hexdigest()

    def _receipt_path(self, key: str) -> Path:
        safe = "".join(character for character in key if character.isalnum() or character in "-_")
        return self.receipt_dir / f"{safe[:180]}.json"

    def _read_receipt(self, key: str) -> dict[str, Any]:
        try:
            payload = json.loads(self._receipt_path(key).read_text(encoding="utf-8"))
            return dict(payload) if isinstance(payload, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _write_receipt(self, key: str, payload: Mapping[str, Any]) -> None:
        path = self._receipt_path(key)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd or self.repo_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.git_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                ["git", *args],
                124,
                stdout=str(exc.stdout or ""),
                stderr=f"git command timed out after {self.git_timeout_seconds}s: {exc.stderr or ''}",
            )

    @staticmethod
    def _command_failure(
        reason: str,
        command: subprocess.CompletedProcess[str],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "merged": False,
            "retryable": True,
            "reason": reason,
            "returncode": command.returncode,
            "stdout": str(command.stdout or "")[-4000:],
            "stderr": str(command.stderr or "")[-4000:],
            **extra,
        }


__all__ = ["MergeCallback", "MergeTrain", "conflict_fingerprint"]
