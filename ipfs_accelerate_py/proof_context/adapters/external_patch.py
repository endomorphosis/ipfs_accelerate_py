"""Fail-closed ingestion of externally authored unified-diff patches (PCCE-034).

This adapter only validates and packages supplied bytes.  It does not read a
repository, invoke a provider, apply a patch, or grant an approval.  The
external origin is retained in a canonical, identity-bound admission record;
the resulting wire records use ``live`` because that is the existing lifecycle
provenance for real, non-replay evidence.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    APPROVAL_AUTHORITY,
    CANONICAL_BRANCH_AUTHORITY,
    AdapterResult,
    CancellationToken,
    admit_adapter_result,
    bind_adapter_request,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    MAX_DECLARED_FILES,
    PATCH_PROPOSAL_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    admit_bounded_patch,
    admit_path_list,
    admit_relative_path,
    assert_declared_scope,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
)

ADAPTER: Final[str] = "ExternalPatchAdapter@0.1"
EXTERNAL_PROVENANCE: Final[str] = "external"
ADMISSION_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/external-patch-admission@1"
_DIFF_HEADER: Final[re.Pattern[str]] = re.compile(r"^diff --git a/(.+) b/(.+)$")
_OLD_HEADER: Final[re.Pattern[str]] = re.compile(r"^--- (.+)$")
_NEW_HEADER: Final[re.Pattern[str]] = re.compile(r"^\+\+\+ (.+)$")


def cid_for_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return the CIDv1 raw/sha2-256 identity of exact bytes."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise MalformedError("artifact identity requires exact bytes")
    digest = hashlib.sha256(bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode("ascii").lower().rstrip("=")


def _cid_for_record(value: Mapping[str, Any]) -> str:
    return cid_for_bytes(wire_canonical_utf8(value).encode("utf-8"))


def _patch_path(raw: str, *, field: str) -> str | None:
    raw = raw.strip()
    if raw in {"/dev/null", "dev/null"}:
        return None
    # Unified headers may contain a timestamp after a tab; filenames may not.
    return admit_relative_path(raw.split("\t", 1)[0].removeprefix("a/").removeprefix("b/"), field=field)


def parse_patch_paths(patch_bytes: bytes | bytearray | memoryview) -> tuple[str, ...]:
    """Parse a strict textual unified diff and return every changed path once.

    Both sides of each ``diff --git`` header are required, and any following
    ``---``/``+++`` file headers must agree.  This deliberately rejects
    binary, opaque, context-free, and malformed patch formats.
    """

    patch = admit_bounded_patch(patch_bytes)
    if not patch:
        raise MalformedError("external patch must not be empty")
    try:
        text = patch.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise MalformedError("external patch must be valid UTF-8 text") from exc
    if "\x00" in text:
        raise MalformedError("external patch must not contain NUL bytes")
    lines = text.splitlines()
    paths: list[str] = []
    seen: set[str] = set()
    current: tuple[str | None, str | None] | None = None
    header_count = 0
    saw_change = False
    for line in lines:
        match = _DIFF_HEADER.match(line)
        if match:
            if current is not None and current[0] is None and current[1] is None:
                raise MalformedError("external patch has an empty diff section")
            left = _patch_path(match.group(1), field="patch_path")
            right = _patch_path(match.group(2), field="patch_path")
            if left is None or right is None:
                raise MalformedError("diff --git headers cannot use /dev/null")
            current = (left, right)
            header_count += 1
            for path in (left, right):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
            continue
        if current is None:
            continue
        old = _OLD_HEADER.match(line)
        new = _NEW_HEADER.match(line)
        if old:
            parsed = _patch_path(old.group(1), field="patch_path")
            if parsed is not None and parsed != current[0]:
                raise BoundaryViolationError("external patch old path disagrees with diff header")
            continue
        if new:
            parsed = _patch_path(new.group(1), field="patch_path")
            if parsed is not None and parsed != current[1]:
                raise BoundaryViolationError("external patch new path disagrees with diff header")
            continue
        if line.startswith(("+", "-", "@@ ")) and not line.startswith(("+++ ", "--- ")):
            saw_change = True
    if header_count == 0 or not paths or not saw_change:
        raise MalformedError("external patch must be a non-empty unified diff")
    return tuple(paths)


@dataclass(frozen=True)
class ExternalPatch:
    """Externally supplied bytes and their explicit repository-relative claim."""

    patch_bytes: bytes
    declared_files: tuple[str, ...]

    def __post_init__(self) -> None:
        patch = admit_bounded_patch(self.patch_bytes)
        declared = admit_path_list(
            self.declared_files,
            field="declared_files",
            min_items=1,
            max_items=MAX_DECLARED_FILES,
        )
        parsed = parse_patch_paths(patch)
        if set(parsed) != set(declared):
            raise BoundaryViolationError(
                "external patch paths and declared files must agree exactly",
                details={"reason": "declared_path_mismatch"},
            )
        object.__setattr__(self, "patch_bytes", patch)
        object.__setattr__(self, "declared_files", declared)

    @property
    def patch_cid(self) -> str:
        return cid_for_bytes(self.patch_bytes)


class ExternalPatchAdapter:
    """Package one immutable external patch under the ordinary adapter contract."""

    def __init__(self, patch: ExternalPatch | bytes, declared_files: Sequence[str] | None = None) -> None:
        if isinstance(patch, ExternalPatch):
            if declared_files is not None:
                raise MalformedError("declared_files is invalid when patch is ExternalPatch")
            self._patch = patch
        else:
            if declared_files is None:
                raise MalformedError("external patch requires declared_files")
            self._patch = ExternalPatch(bytes(patch), tuple(declared_files))

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult:
        if cancellation is not None:
            cancellation.check()
        bind_adapter_request(task, context_pack, route)
        assert_declared_scope(self._patch.declared_files, task.owned_paths, task.declared_files)
        # Re-parse at consumption so a forged/replaced object cannot skip ingestion checks.
        if set(parse_patch_paths(self._patch.patch_bytes)) != set(self._patch.declared_files):
            raise IdentityInconsistentError("external patch path binding drifted")
        admission = {
            "schema": ADMISSION_SCHEMA,
            "adapter": ADAPTER,
            "provenance": EXTERNAL_PROVENANCE,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "pack_cid": context_pack.pack_cid,
            "route_cid": route.decision_cid,
            "patch_cid": self._patch.patch_cid,
            "declared_files": list(self._patch.declared_files),
        }
        response_artifact_cid = _cid_for_record(admission)
        invocation_body = {
            "schema": CODING_AGENT_INVOCATION_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "route_cid": route.decision_cid,
            "provider": route.provider,
            "model": route.model,
            "revision": route.revision or "unspecified",
            "tier": route.tier,
            "token_count": 0,
            "cached_token_count": 0,
            "latency_ms": 0,
            "cost_micros": 0,
            "response_artifact_cid": response_artifact_cid,
            "provenance": "live",
        }
        invocation_cid = _cid_for_record(invocation_body)
        invocation = CodingAgentInvocation.from_mapping({**invocation_body, "invocation_cid": invocation_cid})
        proposal_body = {
            "schema": PATCH_PROPOSAL_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "invocation_cid": invocation_cid,
            "patch_cid": self._patch.patch_cid,
            "declared_files": list(self._patch.declared_files),
            "provenance": "live",
        }
        proposal = PatchProposal.from_mapping({**proposal_body, "proposal_cid": _cid_for_record(proposal_body)})
        result = AdapterResult(
            proposal=proposal,
            invocation=invocation,
            patch_bytes=self._patch.patch_bytes,
            log_bytes=wire_canonical_utf8(admission).encode("utf-8"),
        )
        return admit_adapter_result(task, context_pack, route, result, cancellation=cancellation)


DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType({
    "adapter": ADAPTER,
    "admission_schema": ADMISSION_SCHEMA,
    "external_provenance": EXTERNAL_PROVENANCE,
    "approval_authority": APPROVAL_AUTHORITY,
    "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
    "applies_patch": False,
})


__all__ = [
    "ADAPTER", "ADMISSION_SCHEMA", "DESCRIPTOR", "EXTERNAL_PROVENANCE",
    "ExternalPatch", "ExternalPatchAdapter", "cid_for_bytes", "parse_patch_paths",
]
