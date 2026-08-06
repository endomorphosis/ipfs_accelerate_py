"""Analytical close executor for closed_deterministic dispositions (WPD-022).

Interface: ``AnalyticalCloseExecutor@1``

Applies admitted analytical edits under existing worktree/lease rules without
loading LLM surfaces.  Success requires real byte mutation when the plan
expects writes; fake success without mutation is rejected fail-closed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


ANALYTICAL_CLOSE_EXECUTOR_INTERFACE: Final[str] = "AnalyticalCloseExecutor@1"
ANALYTICAL_CLOSE_EXECUTOR_VERSION: Final[int] = 1
ANALYTICAL_CLOSE_EXECUTOR_EVIDENCE: Final[str] = "wpd/analytical-close-executor@1"
ANALYTICAL_CLOSE_EXECUTOR_PRODUCER: Final[str] = "analytical-close-executor@1"

ANALYTICAL_CLOSE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analytical-close-receipt@1"
)


class AnalyticalCloseExecutorError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete analytical close."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "analytical_close_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "analytical_close_error")


class AnalyticalCloseMutationError(AnalyticalCloseExecutorError):
    """Plan expected writes but no real byte mutation was observed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mutation_required",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class AnalyticalClosePathError(AnalyticalCloseExecutorError, ValueError):
    """Edit path escapes the worktree or is otherwise unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "path_escape",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class AnalyticalEdit:
    """One exact span replacement under a worktree-relative path."""

    path: str
    start: int
    end: int
    replacement: str
    before_hash: str = ""
    expected_after_hash: str = ""
    artifact_id: str = ""

    def __post_init__(self) -> None:
        path = str(self.path or "").strip().replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise AnalyticalClosePathError(
                f"unsafe analytical edit path: {self.path!r}",
                reason_code="unsafe_path",
            )
        if int(self.start) < 0 or int(self.end) < int(self.start):
            raise AnalyticalCloseExecutorError(
                "edit span must satisfy 0 <= start <= end",
                reason_code="invalid_span",
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        replacement = str(self.replacement)
        object.__setattr__(self, "replacement", replacement)
        expected = str(self.expected_after_hash or "").strip() or _sha256_text(
            replacement
        )
        if expected != _sha256_text(replacement):
            raise AnalyticalCloseExecutorError(
                "expected_after_hash must equal sha256 of replacement",
                reason_code="after_hash_mismatch",
            )
        object.__setattr__(self, "expected_after_hash", expected)
        object.__setattr__(
            self, "before_hash", str(self.before_hash or "").strip()
        )
        object.__setattr__(
            self, "artifact_id", str(self.artifact_id or "").strip()
        )


@dataclass(frozen=True)
class AnalyticalClosePlan:
    """Admitted analytical close plan bound to a worktree."""

    edits: tuple[AnalyticalEdit, ...]
    expects_writes: bool = True
    plan_cid: str = ""
    task_cid: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edits", tuple(self.edits or ()))
        if self.expects_writes and not self.edits:
            raise AnalyticalCloseExecutorError(
                "expects_writes requires at least one edit",
                reason_code="empty_write_plan",
            )


@dataclass(frozen=True)
class AnalyticalCloseReceipt:
    """Body-free receipt of an analytical close attempt."""

    applied: bool
    mutated: bool
    bytes_before: int
    bytes_after: int
    paths_touched: tuple[str, ...]
    reason_code: str
    plan_cid: str = ""
    task_cid: str = ""
    edit_count: int = 0
    producer_id: str = ANALYTICAL_CLOSE_EXECUTOR_PRODUCER

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYTICAL_CLOSE_RECEIPT_SCHEMA,
            "contract_version": ANALYTICAL_CLOSE_EXECUTOR_VERSION,
            "applied": self.applied,
            "mutated": self.mutated,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "paths_touched": list(self.paths_touched),
            "reason_code": self.reason_code,
            "plan_cid": self.plan_cid,
            "task_cid": self.task_cid,
            "edit_count": self.edit_count,
            "producer_id": self.producer_id,
        }


@dataclass
class AnalyticalCloseExecutor:
    """Apply admitted analytical edits under a worktree root.

    Never imports LLM client modules.  Roll back file contents on failure when
    a mutation was partially applied.
    """

    worktree_root: Path

    def __post_init__(self) -> None:
        root = Path(self.worktree_root).resolve()
        if not root.is_dir():
            raise AnalyticalCloseExecutorError(
                f"worktree_root is not a directory: {root}",
                reason_code="worktree_missing",
            )
        self.worktree_root = root

    def apply(self, plan: AnalyticalClosePlan | Mapping[str, Any]) -> AnalyticalCloseReceipt:
        """Apply edits and prove mutation when writes are expected."""

        normalized = self._normalize_plan(plan)
        snapshots: dict[str, bytes | None] = {}
        bytes_before = 0
        bytes_after = 0
        paths_touched: list[str] = []

        try:
            # Snapshot + validate before any write.
            for edit in normalized.edits:
                target = self._resolve_path(edit.path)
                if target.exists():
                    before = target.read_bytes()
                else:
                    before = None
                snapshots[edit.path] = before
                if before is not None:
                    bytes_before += len(before)
                    if edit.before_hash:
                        actual = _sha256_bytes(before)
                        if actual != edit.before_hash:
                            raise AnalyticalCloseExecutorError(
                                f"before_hash mismatch for {edit.path}",
                                reason_code="before_hash_mismatch",
                            )
                    text = before.decode("utf-8")
                else:
                    if edit.start != 0 or edit.end != 0:
                        raise AnalyticalCloseExecutorError(
                            f"missing file {edit.path} requires empty span 0:0",
                            reason_code="missing_file",
                        )
                    text = ""
                # Character-based span for closed hermetic shapes.
                if edit.end > len(text):
                    raise AnalyticalCloseExecutorError(
                        f"edit span out of bounds for {edit.path}",
                        reason_code="span_oob",
                    )

            # Apply
            for edit in normalized.edits:
                target = self._resolve_path(edit.path)
                before = snapshots[edit.path]
                text = "" if before is None else before.decode("utf-8")
                new_text = text[: edit.start] + edit.replacement + text[edit.end :]
                target.parent.mkdir(parents=True, exist_ok=True)
                encoded = new_text.encode("utf-8")
                target.write_bytes(encoded)
                bytes_after += len(encoded)
                paths_touched.append(edit.path)

            mutated = False
            for path in paths_touched:
                before = snapshots[path]
                after = self._resolve_path(path).read_bytes()
                if before != after:
                    mutated = True
                    break

            if normalized.expects_writes and not mutated:
                raise AnalyticalCloseMutationError(
                    "analytical close expects writes but no byte mutation occurred",
                    reason_code="fake_success_without_mutation",
                )

            return AnalyticalCloseReceipt(
                applied=True,
                mutated=mutated,
                bytes_before=bytes_before,
                bytes_after=bytes_after,
                paths_touched=tuple(dict.fromkeys(paths_touched)),
                reason_code="analytical_close_applied",
                plan_cid=normalized.plan_cid,
                task_cid=normalized.task_cid,
                edit_count=len(normalized.edits),
            )
        except Exception:
            self._rollback(snapshots)
            raise

    def _normalize_plan(
        self, plan: AnalyticalClosePlan | Mapping[str, Any]
    ) -> AnalyticalClosePlan:
        if isinstance(plan, AnalyticalClosePlan):
            return plan
        if not isinstance(plan, Mapping):
            raise AnalyticalCloseExecutorError(
                "plan must be AnalyticalClosePlan or mapping",
                reason_code="invalid_plan",
            )
        edits_raw = plan.get("edits") or ()
        edits: list[AnalyticalEdit] = []
        for item in edits_raw:
            if isinstance(item, AnalyticalEdit):
                edits.append(item)
            elif isinstance(item, Mapping):
                edits.append(
                    AnalyticalEdit(
                        path=str(item.get("path") or ""),
                        start=int(item.get("start") or 0),
                        end=int(item.get("end") or 0),
                        replacement=str(item.get("replacement") or ""),
                        before_hash=str(item.get("before_hash") or ""),
                        expected_after_hash=str(
                            item.get("expected_after_hash") or ""
                        ),
                        artifact_id=str(item.get("artifact_id") or ""),
                    )
                )
            else:
                raise AnalyticalCloseExecutorError(
                    "edits must be AnalyticalEdit or mappings",
                    reason_code="invalid_edit",
                )
        return AnalyticalClosePlan(
            edits=tuple(edits),
            expects_writes=bool(plan.get("expects_writes", True)),
            plan_cid=str(plan.get("plan_cid") or ""),
            task_cid=str(plan.get("task_cid") or ""),
            notes=tuple(plan.get("notes") or ()),
        )

    def _resolve_path(self, relative: str) -> Path:
        candidate = (self.worktree_root / relative).resolve()
        try:
            candidate.relative_to(self.worktree_root)
        except ValueError as exc:
            raise AnalyticalClosePathError(
                f"path escapes worktree: {relative}",
                reason_code="path_escape",
            ) from exc
        return candidate

    def _rollback(self, snapshots: Mapping[str, bytes | None]) -> None:
        for relative, before in snapshots.items():
            target = self._resolve_path(relative)
            try:
                if before is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(before)
            except OSError:
                # Best-effort rollback; original exception is re-raised.
                continue


def build_analytical_close_executor(worktree_root: Path | str) -> AnalyticalCloseExecutor:
    """Construct an analytical close executor for ``worktree_root``."""

    return AnalyticalCloseExecutor(worktree_root=Path(worktree_root))


__all__ = [
    "ANALYTICAL_CLOSE_EXECUTOR_EVIDENCE",
    "ANALYTICAL_CLOSE_EXECUTOR_INTERFACE",
    "ANALYTICAL_CLOSE_EXECUTOR_PRODUCER",
    "ANALYTICAL_CLOSE_EXECUTOR_VERSION",
    "ANALYTICAL_CLOSE_RECEIPT_SCHEMA",
    "AnalyticalCloseExecutor",
    "AnalyticalCloseExecutorError",
    "AnalyticalCloseMutationError",
    "AnalyticalClosePathError",
    "AnalyticalClosePlan",
    "AnalyticalCloseReceipt",
    "AnalyticalEdit",
    "build_analytical_close_executor",
]
