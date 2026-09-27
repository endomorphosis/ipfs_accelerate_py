"""Hand CST extraction to the kit-owned VFS outbox and world-root publisher.

The in-memory CST transform is already applied by ``apply_cst_extraction``.
This module stages that result as an outbox nomination and a publication
*request*. It does not write the repository, mutate VFS, or CAS the current
root. An outstanding completion block refuses the handoff. TypeSafe is not
this owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import local
from typing import Any


class CstWorldHandoffError(ValueError):
    """CST result cannot be staged to the world-root owner."""


@dataclass(frozen=True, slots=True)
class CstWorldHandoff:
    applied: bool
    blocked: bool
    reason: str
    outbox_nomination_cid: str
    publication_status: str
    semantic_world_root_cid: str
    world_root_status: str = ""
    writes_repository: bool = False
    cas_completed: bool = False
    changes_current_root: bool = False
    completion_authority: bool = False
    libcst_usable: bool = False
    sources: tuple[tuple[str, str], ...] = ()
    kit_root_cid: str = ""
    kit_generation: int = 0
    vfs_published: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "blocked": self.blocked,
            "reason": self.reason,
            "outbox_nomination_cid": self.outbox_nomination_cid,
            "publication_status": self.publication_status,
            "semantic_world_root_cid": self.semantic_world_root_cid,
            "world_root_status": self.world_root_status,
            "source_paths": [path for path, _text in self.sources],
            "writes_repository": False,
            "cas_completed": False,
            "changes_current_root": False,
            "completion_authority": False,
            "libcst_usable": False,
            "accepted_as_authority": False,
            "kit_root_cid": self.kit_root_cid,
            "kit_generation": self.kit_generation,
            "vfs_published": self.vfs_published,
            "supervisor_accepted": False,
        }


_HANDOFFS = local()


def clear_cst_handoffs() -> None:
    _HANDOFFS.value = ()


def record_cst_handoff(handoff: CstWorldHandoff) -> None:
    current = getattr(_HANDOFFS, "value", ()) or ()
    _HANDOFFS.value = tuple(current) + (handoff,)


def last_cst_handoffs() -> tuple[CstWorldHandoff, ...]:
    value = getattr(_HANDOFFS, "value", ()) or ()
    return tuple(value)


def _source_pairs(result: Any) -> tuple[tuple[str, str], ...]:
    raw = getattr(result, "sources", {}) or {}
    if not isinstance(raw, dict):
        return ()
    return tuple(sorted((str(path), str(text)) for path, text in raw.items()))


def _blocked(reason: str) -> CstWorldHandoff:
    handoff = CstWorldHandoff(
        applied=False,
        blocked=True,
        reason=reason,
        outbox_nomination_cid="",
        publication_status="",
        semantic_world_root_cid="",
    )
    record_cst_handoff(handoff)
    return handoff


def stage_cst_world_root(
    result: Any,
    *,
    expected_generation: int = 1,
    expected_root_cid: str | None = None,
    pre_world_root_cid: str | None = None,
    kit_store: Any | None = None,
    kit_predecessor_cid: str | None = None,
) -> CstWorldHandoff:
    """Stage one CST extraction, and commit it when a kit store is bound.

    Without ``kit_store`` this remains a publication request. With a kit
    coordination store, kit CAS-commits one successor root and publishes
    that root through the post-commit VFS outbox. The git repository is
    not written, and the commit is not task completion.
    """

    from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
        completion_is_blocked,
        last_completion_blocks,
    )

    if completion_is_blocked():
        reason = str(last_completion_blocks().get("reason") or "completion_blocked")
        return _blocked(reason)
    if getattr(result, "mutated", False) is not False:
        raise CstWorldHandoffError("CST handoff cannot mutate the repository")
    if getattr(result, "writes_repository", False) is not False:
        raise CstWorldHandoffError("CST handoff cannot write the repository")
    if getattr(result, "libcst_usable", False) is not False:
        raise CstWorldHandoffError("libcst must not be claimed usable")
    if getattr(result, "can_authorize_completion", False) is not False:
        raise CstWorldHandoffError("CST handoff cannot authorize completion")
    maps = getattr(result, "source_maps", ()) or ()
    if not maps:
        return _blocked("source_map_not_preserved")
    tree_id = str(getattr(result, "tree_id", "") or "")
    artifact_cid = str(getattr(result, "result_cid", "") or "")
    if not artifact_cid:
        raise CstWorldHandoffError("CST result is missing result_cid")

    from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.world_root_adapter import (
        PersistKind,
        VfsOutboxNomination,
        WorldRootAdapterError,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_adapters import (
        OperationalWorldRootPublisher,
    )

    try:
        nomination = VfsOutboxNomination(
            artifact_kind=PersistKind.RECEIPT.value,
            artifact_cid=artifact_cid,
            tree_id=tree_id,
        )
    except WorldRootAdapterError as exc:
        raise CstWorldHandoffError(str(exc)) from exc
    publication = OperationalWorldRootPublisher().request_publication(
        {"semantic_world_root_cid": artifact_cid},
        expected_generation=int(expected_generation),
        expected_root_cid=expected_root_cid,
    )
    if getattr(publication, "changes_current_root", False) is not False:
        raise CstWorldHandoffError("publication must not change the current root")
    world_status = ""
    if pre_world_root_cid:
        from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.world_root_adapter import (
            AdapterStatus,
            persist_through_kit_authorities,
        )

        receipt = persist_through_kit_authorities(
            {
                "tree_id": tree_id,
                "pre_world_root_cid": pre_world_root_cid,
                "current_world_root_cid": pre_world_root_cid,
                "expected_root_generation": int(expected_generation),
                "current_root_generation": int(expected_generation),
                "receipt_cids": [artifact_cid],
                "network": "deny",
                "task_owner": "ipfs_accelerate_py",
            }
        )
        world_status = str(getattr(receipt, "status", "") or "")
        if world_status != AdapterStatus.NOMINATED_PERSIST.value:
            return _blocked(world_status or "world_root_not_nominated")
        if getattr(receipt, "writes_repository", False) is not False:
            raise CstWorldHandoffError("world root cannot write the repository")
    sources = _source_pairs(result)
    kit_root_cid = ""
    kit_generation = 0
    vfs_published = False
    reason = "cst_staged_publication_requested"
    if kit_store is not None:
        from .cst_kit_commit import CstKitCommitError, commit_cst_sources_through_kit

        try:
            committed = commit_cst_sources_through_kit(
                kit_store,
                sources=dict(sources),
                result_cid=artifact_cid,
                pre_world_root_cid=kit_predecessor_cid,
            )
        except CstKitCommitError as exc:
            raise CstWorldHandoffError(str(exc)) from exc
        if not committed.applied:
            return _blocked(committed.reason)
        if committed.supervisor_accepted or committed.completion_authority:
            raise CstWorldHandoffError("kit commit cannot authorize completion")
        if committed.writes_repository:
            raise CstWorldHandoffError("kit commit cannot write the repository")
        kit_root_cid = committed.root_cid
        kit_generation = committed.generation
        vfs_published = committed.vfs_published
        world_status = committed.reason
        reason = committed.reason
    handoff = CstWorldHandoff(
        applied=True,
        blocked=False,
        reason=reason,
        outbox_nomination_cid=nomination.nomination_cid,
        publication_status=str(getattr(publication, "status", "") or ""),
        semantic_world_root_cid=str(
            getattr(publication, "semantic_world_root_cid", "") or artifact_cid
        ),
        world_root_status=world_status,
        sources=sources,
        kit_root_cid=kit_root_cid,
        kit_generation=kit_generation,
        vfs_published=vfs_published,
    )
    record_cst_handoff(handoff)
    return handoff
