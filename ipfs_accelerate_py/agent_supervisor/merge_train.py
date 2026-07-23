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
from typing import Any, Callable, Iterator, Mapping, Sequence

from .formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    InvariantClass,
    PolicySelection,
    RiskLevel,
    default_formal_verification_policy,
)
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
        formal_verification_policy: Optional risk-selected policy applied before
            any callback or built-in target advancement.
        proof_gate: Evidence provider for selected proof requirements.  The
            ``proof_gate_callback`` spelling is retained as an adapter alias.
        proof_cache_dir: Durable exact-selection gate cache shared by retries.
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
        formal_verification_policy: FormalVerificationPolicy | Mapping[str, Any] | None = None,
        proof_gate: Callable[..., Any] | None = None,
        proof_gate_callback: Callable[..., Any] | None = None,
        proof_cache_dir: Path | str | None = None,
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
        if (
            proof_gate is not None
            and proof_gate_callback is not None
            and proof_gate is not proof_gate_callback
        ):
            raise ValueError("proof_gate and proof_gate_callback must refer to the same callback")
        self.proof_gate = proof_gate or proof_gate_callback
        self.proof_gate_callback = self.proof_gate
        if formal_verification_policy is None:
            self.formal_verification_policy = (
                default_formal_verification_policy()
                if self.proof_gate is not None
                else None
            )
        elif isinstance(formal_verification_policy, FormalVerificationPolicy):
            self.formal_verification_policy = formal_verification_policy
        elif isinstance(formal_verification_policy, Mapping):
            self.formal_verification_policy = FormalVerificationPolicy.from_dict(
                formal_verification_policy
            )
        else:
            raise TypeError(
                "formal_verification_policy must be a FormalVerificationPolicy or mapping"
            )
        self.proof_cache_dir = Path(
            proof_cache_dir
            if proof_cache_dir is not None
            else self.state_dir / "proof-gate-cache"
        )
        self.proof_gate_state_dir = self.state_dir / "proof-gates"
        self.proof_gate_pin_dir = self.proof_gate_state_dir / "pins"
        self.proof_gate_attempt_dir = self.proof_gate_state_dir / "attempts"
        if self.formal_verification_policy is not None:
            self.proof_cache_dir.mkdir(parents=True, exist_ok=True)
            self.proof_gate_pin_dir.mkdir(parents=True, exist_ok=True)
            self.proof_gate_attempt_dir.mkdir(parents=True, exist_ok=True)

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
            self._recover_abandoned_claims()
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
            self._recover_abandoned_claims()
            while max_items is None or len(results) < int(max_items):
                request = self._dequeue()
                if request is None:
                    break
                results.append(self._process_claimed(request))
        return results

    run = drain

    def _recover_abandoned_claims(self) -> int:
        recover = getattr(self.queue, "recover_abandoned_train_claims", None)
        if not callable(recover):
            return 0
        return int(recover() or 0)

    def status(self) -> dict[str, Any]:
        queue_status = self.queue.status() if hasattr(self.queue, "status") else {}
        return {
            "owner_id": self.owner_id,
            "target_branch": self.target_branch,
            "state_dir": str(self.state_dir),
            "consumer_lock_path": str(self.consumer_lock_path),
            "proof_gate_enabled": self.formal_verification_policy is not None,
            "proof_policy_id": (
                self.formal_verification_policy.policy_id
                if self.formal_verification_policy is not None
                else ""
            ),
            "proof_cache_dir": str(self.proof_cache_dir),
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

        proof_gate_receipt: dict[str, Any] = {}
        proof_tree_id = ""
        try:
            proof_policy = self._proof_policy_for_request(request)
        except Exception as exc:
            return self._finish_failure(
                request,
                reason="proof_gate_identity_invalid",
                details={"proof_gate_error": f"{type(exc).__name__}: {exc}"},
                started_at=started_at,
                retryable=False,
            )
        if proof_policy is not None:
            tree_result = self._git("rev-parse", "--verify", f"{candidate}^{{tree}}")
            if tree_result.returncode != 0 or not tree_result.stdout.strip():
                return self._finish_failure(
                    request,
                    reason="candidate_tree_missing",
                    details={
                        "commit_sha": candidate,
                        "stderr": tree_result.stderr[-2000:],
                    },
                    started_at=started_at,
                    retryable=False,
                )
            proof_tree_id = f"git-tree:{tree_result.stdout.strip()}"
            gate = self._evaluate_proof_gate(
                request=request,
                candidate=candidate,
                target=target,
                repository_tree_id=proof_tree_id,
                policy=proof_policy,
            )
            proof_gate_receipt = dict(gate.get("receipt") or {})
            if not gate.get("allowed", False):
                return self._finish_failure(
                    request,
                    reason=str(gate.get("reason") or "proof_gate_blocked"),
                    details={
                        "proof_gate": proof_gate_receipt,
                        "proof_gate_cache_hit": bool(gate.get("cache_hit")),
                        "repository_tree_id": proof_tree_id,
                    },
                    started_at=started_at,
                    retryable=bool(gate.get("retryable", True)),
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
                    extra={
                        "duplicate_of": previous.get("request_id", ""),
                        **(
                            {
                                "proof_gate": proof_gate_receipt,
                                "repository_tree_id": proof_tree_id,
                            }
                            if proof_gate_receipt
                            else {}
                        ),
                    },
                )

            if self._is_ancestor(candidate, target):
                return self._finish_success(
                    request,
                    status="already_merged",
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                    extra=(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else None
                    ),
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
                    extra={
                        "merge_result": callback_result,
                        **(
                            {
                                "proof_gate": proof_gate_receipt,
                                "repository_tree_id": proof_tree_id,
                            }
                            if proof_gate_receipt
                            else {}
                        ),
                    },
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
                details={
                    "merge_result": callback_result,
                    **(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {}
                    ),
                },
                started_at=started_at,
                retryable=retryable,
            )

        result = self._rebase_and_integrate(
            request=request,
            canonical=canonical,
            candidate=candidate,
            target=target,
            proof_tree_id=proof_tree_id,
        )
        if result.get("merged"):
            return self._finish_success(
                request,
                status="merged",
                canonical=canonical,
                candidate=candidate,
                target=str(result.get("target_commit") or target),
                started_at=started_at,
                extra={
                    **result,
                    **(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {}
                    ),
                },
            )
        return self._finish_failure(
            request,
            reason=str(result.get("reason") or "merge_failed"),
            details={
                **result,
                **(
                    {
                        "proof_gate": proof_gate_receipt,
                        "repository_tree_id": proof_tree_id,
                    }
                    if proof_gate_receipt
                    else {}
                ),
            },
            started_at=started_at,
            retryable=bool(result.get("retryable", True)),
        )

    @staticmethod
    def _metadata_strings(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(
                item.strip()
                for item in value.replace("\n", ",").split(",")
                if item.strip()
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return tuple(str(item).strip() for item in value if str(item).strip())
        return ()

    @staticmethod
    def _risk_for_priority(priority: str) -> RiskLevel:
        return {
            "P0": RiskLevel.CRITICAL,
            "P1": RiskLevel.HIGH,
            "P2": RiskLevel.MEDIUM,
            "P3": RiskLevel.LOW,
        }.get(str(priority or "").strip().upper(), RiskLevel.MEDIUM)

    @staticmethod
    def _modeled_invariant_hints(path: str) -> tuple[str, ...]:
        """Conservatively classify modeled supervisor surfaces by path.

        These hints are only used when a producer did not provide richer
        symbol-level scope information.  They are deliberately deterministic
        and additive: a broad hint can select a stricter rule, but cannot lower
        a requirement selected by another hint.
        """

        lowered = path.casefold()
        values: set[str] = set()
        if "agent_supervisor/" in lowered:
            values.update(
                {
                    InvariantClass.DATA_INTEGRITY.value,
                    InvariantClass.STATE_TRANSITION.value,
                }
            )
        if "merge" in lowered:
            values.add(InvariantClass.MERGE_IDEMPOTENCE.value)
        if any(token in lowered for token in ("queue", "scheduler", "goal", "task")):
            values.add(InvariantClass.STATE_TRANSITION.value)
        if any(token in lowered for token in ("dag", "graph", "dependency")):
            values.add(InvariantClass.DAG_ACYCLICITY.value)
        if any(token in lowered for token in ("lease", "lock")):
            values.add(InvariantClass.LEASE_SAFETY.value)
        if "cache" in lowered:
            values.add(InvariantClass.CACHE_KEY_COMPLETENESS.value)
        if any(token in lowered for token in ("proof", "evidence", "receipt")):
            values.add(InvariantClass.EVIDENCE_FRESHNESS.value)
        if any(token in lowered for token in ("auth", "permission", "override")):
            values.add(InvariantClass.AUTHORIZATION.value)
        if any(token in lowered for token in ("resource", "worktree", "sandbox")):
            values.add(InvariantClass.RESOURCE_ISOLATION.value)
        return tuple(sorted(values))

    def _changed_scopes(
        self,
        request: MergeRequest,
        *,
        candidate: str,
        target: str,
    ) -> tuple[ChangedScope, ...]:
        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        supplied = metadata.get("proof_changed_scopes")
        if isinstance(supplied, Sequence) and not isinstance(
            supplied, (str, bytes, bytearray)
        ):
            if metadata.get("proof_changed_scopes_complete") is False:
                raise ValueError("proof changed-scope packet is incomplete")
            scopes = []
            for raw in supplied:
                if isinstance(raw, ChangedScope):
                    scopes.append(raw)
                    continue
                if not isinstance(raw, Mapping):
                    raise ValueError("proof_changed_scopes entries must be mappings")
                # Queue producers may use the canonical wire contract or a
                # compact, JSON-friendly spelling.
                if raw.get("schema"):
                    scopes.append(ChangedScope.from_dict(raw))
                else:
                    scopes.append(
                        ChangedScope(
                            path=raw.get("path", ""),
                            ast_scope_ids=tuple(raw.get("ast_scope_ids") or ()),
                            risk=raw.get(
                                "risk", self._risk_for_priority(request.priority)
                            ),
                            invariant_classes=tuple(
                                raw.get("invariant_classes") or ()
                            ),
                            change_kind=raw.get("change_kind", "modified"),
                            metadata=raw.get("metadata") or {},
                        )
                    )
            return tuple(sorted(scopes, key=lambda item: item.scope_id))

        baseline = str(metadata.get("baseline_ref") or "").strip()
        if not baseline:
            merge_base = self._git("merge-base", candidate, target)
            baseline = (
                merge_base.stdout.strip()
                if merge_base.returncode == 0 and merge_base.stdout.strip()
                else target
            )
        diff = self._git(
            "diff",
            "--name-status",
            "--find-renames",
            baseline,
            candidate,
        )
        if diff.returncode != 0:
            raise ValueError(f"could not derive proof changed scopes: {diff.stderr[-1000:]}")

        task_payload = metadata.get("task")
        task_metadata = (
            task_payload.get("metadata")
            if isinstance(task_payload, Mapping)
            and isinstance(task_payload.get("metadata"), Mapping)
            else {}
        )
        ast_scope_ids = self._metadata_strings(
            metadata.get("proof_ast_scope_ids")
            or task_metadata.get("ast symbols")
            or task_metadata.get("ast_symbols")
        )
        explicit_invariants = self._metadata_strings(
            metadata.get("proof_invariant_classes")
            or task_metadata.get("invariant classes")
            or task_metadata.get("invariant_classes")
        )
        risk = self._risk_for_priority(request.priority)
        scopes: list[ChangedScope] = []
        for line in diff.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            status = fields[0]
            path = fields[-1]
            kind = {
                "A": "added",
                "D": "deleted",
                "R": "renamed",
                "C": "copied",
            }.get(status[:1], "modified")
            invariants = tuple(
                sorted(
                    set(explicit_invariants)
                    | set(self._modeled_invariant_hints(path))
                )
            )
            scopes.append(
                ChangedScope(
                    path=path,
                    ast_scope_ids=ast_scope_ids,
                    risk=risk,
                    invariant_classes=invariants,
                    change_kind=kind,
                    metadata={"git_status": status},
                )
            )
        return tuple(sorted(scopes, key=lambda item: item.scope_id))

    @staticmethod
    def _proof_gate_cache_key(selection: PolicySelection) -> str:
        payload = "\0".join(
            (
                selection.policy_id,
                selection.selection_id,
                selection.repository_tree_id,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _proof_policy_for_request(
        self, request: MergeRequest
    ) -> FormalVerificationPolicy | None:
        """Resolve the immutable queue policy without allowing a retry downgrade.

        Queue producers persist the complete policy snapshot beside the
        candidate.  A consumer may additionally have a configured policy, but
        it must be the same snapshot.  This prevents a restarted consumer with
        a missing or weaker configuration from bypassing a gate that already
        timed out or observed an unavailable provider.
        """

        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        raw_policy = metadata.get("formal_verification_policy")
        queued_policy: FormalVerificationPolicy | None = None
        if raw_policy:
            if not isinstance(raw_policy, Mapping):
                raise ValueError("queued formal-verification policy is malformed")
            queued_policy = FormalVerificationPolicy.from_dict(raw_policy)
        configured = self.formal_verification_policy
        if configured is not None and queued_policy is not None:
            if configured.policy_id != queued_policy.policy_id:
                raise ValueError(
                    "configured proof policy does not match the queued policy snapshot"
                )
        return queued_policy or configured

    @staticmethod
    def _default_proof_plan(
        *,
        policy: FormalVerificationPolicy,
        selection: PolicySelection,
    ) -> dict[str, Any]:
        """Describe the exact selected work even when a provider is unavailable."""

        payload: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-proof-plan@1",
            "policy_id": policy.policy_id,
            "selection_id": selection.selection_id,
            "repository_tree_id": selection.repository_tree_id,
            "requirement_ids": [
                item.requirement_id for item in selection.requirements
            ],
            "requirements": [item.to_dict() for item in selection.requirements],
            "fallback_validations": list(selection.fallback_validations),
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        payload["plan_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _pin_proof_selection(
        self,
        request: MergeRequest,
        *,
        selection: PolicySelection,
        changes: Sequence[ChangedScope],
    ) -> dict[str, Any]:
        path = self.proof_gate_pin_dir / f"{request.request_id}.json"
        pin = {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-proof-pin@1",
            "request_id": request.request_id,
            "policy_id": selection.policy_id,
            "selection_id": selection.selection_id,
            "repository_tree_id": selection.repository_tree_id,
            "rollout_mode": selection.rollout_mode.value,
            "changed_scope_ids": [item.scope_id for item in changes],
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("proof gate pin is unreadable") from exc
            for name in (
                "request_id",
                "policy_id",
                "selection_id",
                "repository_tree_id",
                "rollout_mode",
                "changed_scope_ids",
            ):
                if existing.get(name) != pin.get(name):
                    raise ValueError(f"proof gate pin mismatch: {name}")
            return dict(existing)
        self._atomic_json(path, pin)
        return pin

    @staticmethod
    def _gate_receipt_type() -> Any:
        # Kept lazy so older installations which only use the ungated train
        # can still import this module during a rolling deployment.
        from . import formal_verification_policy as policy_module

        receipt_type = getattr(policy_module, "MergeProofGateReceipt", None)
        if receipt_type is None:
            raise RuntimeError("MergeProofGateReceipt is unavailable")
        return receipt_type

    def _read_cached_gate_receipt(
        self, selection: PolicySelection
    ) -> Any | None:
        path = self.proof_cache_dir / f"{self._proof_gate_cache_key(selection)}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt = self._gate_receipt_type().from_dict(payload)
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if (
            receipt.policy_id != selection.policy_id
            or receipt.selection_id != selection.selection_id
            or receipt.repository_tree_id != selection.repository_tree_id
            or not receipt.allowed
            or getattr(receipt, "override_receipt_id", "")
            or not all(
                result.requirement_satisfied
                for result in receipt.decision.results
            )
        ):
            return None
        return receipt

    def _persist_gate_receipt(
        self,
        request: MergeRequest,
        receipt: Any,
        *,
        cache: bool,
    ) -> dict[str, Any]:
        payload = dict(receipt.to_dict())
        attempt = max(1, int(getattr(request, "attempt", 1) or 1))
        attempt_path = (
            self.proof_gate_attempt_dir
            / f"{request.request_id}-attempt-{attempt}.json"
        )
        self._atomic_json(attempt_path, payload)
        if cache:
            selection = PolicySelection.from_dict(payload["selection"])
            cache_path = (
                self.proof_cache_dir
                / f"{self._proof_gate_cache_key(selection)}.json"
            )
            self._atomic_json(cache_path, payload)
        return payload

    def _evaluate_proof_gate(
        self,
        *,
        request: MergeRequest,
        candidate: str,
        target: str,
        repository_tree_id: str,
        policy: FormalVerificationPolicy,
    ) -> dict[str, Any]:
        try:
            changes = self._changed_scopes(
                request, candidate=candidate, target=target
            )
            selection = policy.select(
                changes, repository_tree_id=repository_tree_id
            )
            self._pin_proof_selection(
                request, selection=selection, changes=changes
            )
        except Exception as exc:
            return {
                "allowed": False,
                "retryable": False,
                "reason": "proof_gate_identity_invalid",
                "receipt": {
                    "policy_id": policy.policy_id,
                    "repository_tree_id": repository_tree_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

        cached = self._read_cached_gate_receipt(selection)
        if cached is not None:
            receipt_type = self._gate_receipt_type()
            cached_metadata = dict(getattr(cached, "metadata", {}) or {})
            cached_metadata["reused_receipt_id"] = cached.receipt_id
            receipt = receipt_type.build(
                policy=policy,
                selection=selection,
                repository_tree_id=repository_tree_id,
                proof_plan=getattr(cached, "proof_plan", None) or None,
                outcomes=getattr(cached, "proof_outcomes", ()),
                validations=getattr(cached, "validation_outcomes", ()),
                proof_receipts=getattr(cached, "proof_receipts", ()),
                proof_receipt_ids=getattr(cached, "proof_receipt_ids", ()),
                provider_status=getattr(cached, "provider_status", {}),
                provider_error=getattr(cached, "provider_error", ""),
                cache_status={
                    "status": "hit",
                    "cache_key": self._proof_gate_cache_key(selection),
                    "reused_receipt_id": cached.receipt_id,
                },
                metadata=cached_metadata,
            )
            payload = self._persist_gate_receipt(request, receipt, cache=False)
            return {
                "allowed": True,
                "retryable": False,
                "reason": "proof_gate_cache_hit",
                "receipt": payload,
                "cache_hit": True,
            }

        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        evidence: Any = metadata.get("proof_gate") or metadata.get(
            "proof_gate_evidence"
        )
        provider_error = ""
        callback_failed = False
        if self.proof_gate is not None:
            try:
                evidence = self._call_compatible(
                    self.proof_gate,
                    request,
                    policy=policy,
                    selection=selection,
                    repository_tree_id=repository_tree_id,
                    changed_scopes=changes,
                    changes=changes,
                    cached_evidence=None,
                    candidate_commit=candidate,
                    target_commit=target,
                )
            except Exception as exc:
                callback_failed = True
                provider_error = f"{type(exc).__name__}: {exc}"
                evidence = {
                    "provider_status": {
                        "status": (
                            "timed_out"
                            if isinstance(exc, TimeoutError)
                            else "unavailable"
                        )
                    },
                    "provider_error": provider_error,
                }

        receipt_type = self._gate_receipt_type()
        try:
            if isinstance(evidence, receipt_type):
                receipt = evidence
            elif isinstance(evidence, Mapping) and str(
                evidence.get("schema") or ""
            ).endswith("merge-proof-gate-receipt@1"):
                receipt = receipt_type.from_dict(evidence)
            else:
                packet = dict(evidence) if isinstance(evidence, Mapping) else {}
                for claimed_name, actual in (
                    ("policy_id", policy.policy_id),
                    ("selection_id", selection.selection_id),
                    ("repository_tree_id", repository_tree_id),
                ):
                    claimed = str(packet.get(claimed_name) or "")
                    if claimed and claimed != actual:
                        raise ValueError(
                            f"proof gate {claimed_name} does not match pinned value"
                        )
                receipt = receipt_type.build(
                    policy=policy,
                    selection=selection,
                    repository_tree_id=repository_tree_id,
                    proof_plan=(
                        packet.get("proof_plan")
                        or self._default_proof_plan(
                            policy=policy,
                            selection=selection,
                        )
                    ),
                    outcomes=packet.get(
                        "proof_outcomes", packet.get("outcomes")
                    ),
                    validations=packet.get(
                        "validations", packet.get("validation_outcomes")
                    ),
                    proof_receipts=tuple(packet.get("proof_receipts") or ()),
                    proof_receipt_ids=tuple(
                        packet.get("proof_receipt_ids") or ()
                    ),
                    override=packet.get("override"),
                    provider_status=packet.get("provider_status"),
                    provider_error=str(
                        packet.get("provider_error") or provider_error
                    ),
                    cache_status={
                        "status": "miss",
                        "cache_key": self._proof_gate_cache_key(selection),
                    },
                )
            if (
                receipt.policy_id != policy.policy_id
                or receipt.selection_id != selection.selection_id
                or receipt.repository_tree_id != repository_tree_id
            ):
                raise ValueError("proof gate receipt identity does not match pinned selection")
        except Exception as exc:
            return {
                "allowed": False,
                "retryable": callback_failed,
                "reason": "proof_gate_provider_failed" if callback_failed else "proof_gate_identity_invalid",
                "receipt": {
                    "policy_id": policy.policy_id,
                    "selection_id": selection.selection_id,
                    "repository_tree_id": repository_tree_id,
                    "provider_error": provider_error
                    or f"{type(exc).__name__}: {exc}",
                },
            }

        payload = self._persist_gate_receipt(
            request,
            receipt,
            cache=bool(receipt.allowed)
            and not bool(getattr(receipt, "override_receipt_id", ""))
            and all(
                result.requirement_satisfied
                for result in receipt.decision.results
            ),
        )
        provider_status = payload.get("provider_status")
        status = (
            str(provider_status.get("status") or "")
            if isinstance(provider_status, Mapping)
            else str(provider_status or "")
        ).casefold()
        transient = callback_failed or status in {
            "timed_out",
            "timeout",
            "unavailable",
            "provider_unavailable",
            "error",
            "failed",
        }
        return {
            "allowed": bool(receipt.allowed),
            "retryable": transient or not bool(receipt.allowed),
            "reason": "proof_gate_allowed" if receipt.allowed else "proof_gate_blocked",
            "receipt": payload,
            "cache_hit": False,
        }

    def _rebase_and_integrate(
        self,
        *,
        request: MergeRequest,
        canonical: str,
        candidate: str,
        target: str,
        proof_tree_id: str = "",
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
            if proof_tree_id:
                rebased_tree = self._git(
                    "rev-parse", "--verify", f"{rebased_commit}^{{tree}}", cwd=workspace
                )
                actual_tree_id = (
                    f"git-tree:{rebased_tree.stdout.strip()}"
                    if rebased_tree.returncode == 0 and rebased_tree.stdout.strip()
                    else ""
                )
                if actual_tree_id != proof_tree_id:
                    return {
                        "merged": False,
                        "retryable": True,
                        "reason": "proof_gate_tree_mismatch",
                        "proof_repository_tree_id": proof_tree_id,
                        "integration_repository_tree_id": actual_tree_id,
                        "rebased_commit": rebased_commit,
                        "stderr": rebased_tree.stderr[-2000:],
                    }
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
