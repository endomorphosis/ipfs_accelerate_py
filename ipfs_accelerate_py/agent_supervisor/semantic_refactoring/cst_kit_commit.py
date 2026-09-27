"""Commit a CST result through the kit world-root owner.

Kit compares-and-swaps the current semantic root, then publishes that root
through the post-commit VFS outbox. This module does not write the git
repository, and a durable VFS write is not supervisor acceptance or task
completion. TypeSafe is not this owner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CstKitCommitError(ValueError):
    """CST result cannot be handed to the kit world-root owner."""


@dataclass(frozen=True, slots=True)
class KitCstCommit:
    applied: bool
    reason: str
    root_cid: str = ""
    generation: int = 0
    vfs_published: bool = False
    vfs_path: str = ""
    supervisor_accepted: bool = False
    writes_repository: bool = False
    completion_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "root_cid": self.root_cid,
            "generation": self.generation,
            "vfs_published": self.vfs_published,
            "vfs_path": self.vfs_path,
            "supervisor_accepted": False,
            "writes_repository": False,
            "completion_authority": False,
            "cas_completed": False,
            "accepted_as_authority": False,
        }


def _token(value: str) -> str:
    safe = "".join(char for char in value.lower() if char.isalnum())
    return (safe[:40] or "wave")


def _operation(suffix: str, token: str, generation: int) -> str:
    return f"cst-{suffix}-g{generation}-{token}"[:128]


def _refused(reason: str) -> KitCstCommit:
    return KitCstCommit(applied=False, reason=reason)


def kit_coordination_dir_from_env(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the kit coordination directory for a CST commit, if one is configured.

    ``SAWM_COORDINATION_DIR`` wins. Otherwise ``SAWM_BIND_KIT_STORE`` must be
    true and a supervisor state root must be set. Unset means nominate only.
    """

    env = os.environ if environ is None else environ
    explicit = str(env.get("SAWM_COORDINATION_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    flag = str(env.get("SAWM_BIND_KIT_STORE") or "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return None
    root = str(
        env.get("AGENT_SUPERVISOR_STATE_ROOT") or env.get("AUTONOMY_STATE_ROOT") or ""
    ).strip()
    if not root:
        return None
    return Path(root) / "semantic-world"


_MATERIALIZING_ROLLOUT_MODES = frozenset({"shadow_apply", "guarded", "required"})
_MERGE_NOMINATION_MODES = frozenset({"guarded", "required"})


def rollout_mode_from_env(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get("SPAR_ROLLOUT_MODE") or env.get("SAWM_ROLLOUT_MODE") or "").strip()


def disposable_worktree_root_from_env(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return a disposable worktree directory, never the git repository.

    An explicit ``SAWM_DISPOSABLE_WORKTREE`` wins. ``shadow_apply``,
    ``guarded``, and ``required`` use a directory under the kit coordination
    store. Bootstrap and shadow-plan do not materialize one.
    """

    env = os.environ if environ is None else environ
    explicit = str(env.get("SAWM_DISPOSABLE_WORKTREE") or "").strip()
    if explicit:
        return Path(explicit)
    if rollout_mode_from_env(env) not in _MATERIALIZING_ROLLOUT_MODES:
        return None
    coordination = kit_coordination_dir_from_env(env)
    if coordination is None:
        return None
    return coordination / "disposable-worktree"


def nominate_current_authority_merge(
    coordination_dir: Path | str,
    worktree: Path | str,
    sources: Mapping[str, str],
    *,
    mode: str,
) -> Path:
    """Record a merge nomination. This does not merge or write git.

    Guarded and required modes hand the disposable worktree to the current
    merge owner. The nomination is not a merge request with a branch or
    commit, and it cannot be consumed as one.
    """

    if mode not in _MERGE_NOMINATION_MODES:
        raise CstKitCommitError("merge nomination requires guarded or required mode")
    directory = Path(coordination_dir)
    if (directory / ".git").exists():
        raise CstKitCommitError("merge nomination cannot write the repository")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "spar/current-authority-merge-nomination@1",
        "rollout_mode": mode,
        "merge_authority": "current authority gates",
        "merged": False,
        "self_merge": False,
        "writes_repository": False,
        "promotes_root": False,
        "completion_authority": False,
        "accepted_as_authority": False,
        "worktree": str(Path(worktree)),
        "source_paths": sorted(sources),
    }
    return _write_handoff("nomination", directory, "merge-nomination.json", payload)


def _coordination_directory(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> Path | None:
    if coordination_dir is not None and str(coordination_dir).strip():
        return Path(coordination_dir)
    return kit_coordination_dir_from_env(environ)


def _handoff_key(kind: str, directory: Path) -> str:
    return f"{kind}:{directory.resolve()}"


def _handoff_plane(directory: Path) -> tuple[str | None, str]:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        resolve_handoff_plane,
    )

    return resolve_handoff_plane(directory)


def _legacy_handoff_file(kind: str, directory: Path, filename: str) -> dict | None:
    """Import one old JSON handoff. Returns None when the file is absent."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        save_merge_handoff,
    )

    path = directory / filename
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CstKitCommitError("merge handoff is unreadable") from exc
    if not isinstance(payload, dict):
        raise CstKitCommitError("merge handoff is unreadable")
    target, _source = _handoff_plane(directory)
    if target is not None:
        save_merge_handoff(_handoff_key(kind, directory), payload, target=target)
    return payload


def _control_handoff(kind: str, directory: Path, filename: str) -> dict | None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        load_merge_handoff,
    )

    target, _source = _handoff_plane(directory)
    if target is None:
        return _legacy_handoff_file(kind, directory, filename)
    loaded = load_merge_handoff(_handoff_key(kind, directory), target=target)
    if loaded:
        return loaded
    return _legacy_handoff_file(kind, directory, filename)


def _write_handoff(kind: str, directory: Path, filename: str, payload: dict) -> Path:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        save_merge_handoff,
    )

    path = directory / filename
    target, _source = _handoff_plane(directory)
    if target is None:
        raise CstKitCommitError("merge handoff requires the DuckDB/Quack control plane")
    save_merge_handoff(_handoff_key(kind, directory), payload, target=target)
    return path


def _read_handoff(kind: str, directory: Path, filename: str) -> dict:
    loaded = _control_handoff(kind, directory, filename)
    if not loaded:
        raise CstKitCommitError("merge handoff is unreadable")
    return loaded


def stored_merge_record(path: Path) -> dict:
    """Read a nomination or hold from the control plane, else its JSON file."""

    kind = "hold" if path.name == "merge-owner-hold.json" else "nomination"
    return _read_handoff(kind, path.parent, path.name)


def merge_nomination_outstanding(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> bool:
    """True while a current-authority nomination has not been consumed.

    A file that claims ``merged`` is still outstanding. This adapter does not
    treat that claim as a merge.
    """

    directory = _coordination_directory(environ, coordination_dir)
    if directory is None:
        return False
    try:
        loaded = _control_handoff("nomination", directory, "merge-nomination.json")
    except CstKitCommitError:
        return True
    if not loaded:
        return False
    payload = loaded
    if payload.get("schema") != "spar/current-authority-merge-nomination@1":
        return True
    if payload.get("merged") is not False or payload.get("writes_repository") is not False:
        return True
    return payload.get("consumed") is not True


def spar_merge_hold_outstanding(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> bool:
    """True while the merge owner holds an unreleased, unmerged nomination."""

    directory = _coordination_directory(environ, coordination_dir)
    if directory is None:
        return False
    try:
        loaded = _control_handoff("hold", directory, "merge-owner-hold.json")
    except CstKitCommitError:
        return True
    if not loaded:
        return False
    payload = loaded
    if payload.get("schema") != "spar/current-authority-merge-hold@1":
        return True
    if payload.get("merged") is not False or payload.get("writes_repository") is not False:
        return True
    return payload.get("released") is not True


def merge_handoff_outstanding(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> bool:
    """True until the merge owner has both accepted and released the handoff."""

    return merge_nomination_outstanding(
        environ, coordination_dir
    ) or spar_merge_hold_outstanding(environ, coordination_dir)


def accept_spar_merge_nomination(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> Path:
    """Current merge owner takes custody. This does not merge or write git."""

    if not merge_nomination_outstanding(environ, coordination_dir):
        raise CstKitCommitError("no outstanding merge nomination")
    directory = _coordination_directory(environ, coordination_dir)
    if directory is None:
        raise CstKitCommitError("merge nomination requires a coordination directory")
    nomination = _read_handoff("nomination", directory, "merge-nomination.json")
    if nomination.get("merged") is not False:
        raise CstKitCommitError("merge nomination cannot claim a merge")
    hold = {
        "schema": "spar/current-authority-merge-hold@1",
        "held_by": "current authority gates",
        "rollout_mode": nomination.get("rollout_mode"),
        "worktree": nomination.get("worktree"),
        "source_paths": list(nomination.get("source_paths") or []),
        "merged": False,
        "self_merge": False,
        "writes_repository": False,
        "promotes_root": False,
        "released": False,
        "completion_authority": False,
        "accepted_as_authority": False,
    }
    path = _write_handoff("hold", directory, "merge-owner-hold.json", hold)
    consume_merge_nomination(environ, coordination_dir)
    return path


def release_spar_merge_hold(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> Path:
    """Owner releases custody. This still does not merge."""

    directory = _coordination_directory(environ, coordination_dir)
    if directory is None:
        raise CstKitCommitError("merge hold requires a coordination directory")
    payload = _read_handoff("hold", directory, "merge-owner-hold.json")
    if payload.get("merged") is not False:
        raise CstKitCommitError("merge hold cannot claim a merge")
    payload["released"] = True
    payload["merged"] = False
    payload["self_merge"] = False
    payload["writes_repository"] = False
    payload["promotes_root"] = False
    payload["completion_authority"] = False
    payload["released_by"] = "current authority gates"
    return _write_handoff("hold", directory, "merge-owner-hold.json", payload)


def consume_merge_nomination(
    environ: Mapping[str, str] | None = None,
    coordination_dir: Path | str | None = None,
) -> Path:
    """Let the current merge owner take the nomination. This does not merge."""

    directory = _coordination_directory(environ, coordination_dir)
    if directory is None:
        raise CstKitCommitError("merge nomination requires a coordination directory")
    payload = _read_handoff("nomination", directory, "merge-nomination.json")
    if payload.get("merged") is not False or payload.get("writes_repository") is not False:
        raise CstKitCommitError("merge nomination cannot claim a merge")
    payload["consumed"] = True
    payload["merged"] = False
    payload["self_merge"] = False
    payload["writes_repository"] = False
    payload["promotes_root"] = False
    payload["completion_authority"] = False
    payload["consumed_by"] = "current authority gates"
    return _write_handoff("nomination", directory, "merge-nomination.json", payload)


def materialize_disposable_worktree(
    root: Path | str,
    sources: Mapping[str, str],
) -> Path:
    """Write rewritten sources into an isolated disposable directory.

    The directory must not already be a git checkout. Paths cannot escape it
    or name ``.git``. Nothing here runs git, merges, or promotes a root.
    """

    if not sources:
        raise CstKitCommitError("disposable worktree requires rewritten sources")
    destination = Path(root)
    if destination.exists() and (destination / ".git").exists():
        raise CstKitCommitError("disposable worktree cannot write the repository")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    if (resolved_root / ".git").exists():
        raise CstKitCommitError("disposable worktree cannot write the repository")
    for raw_path, text in sources.items():
        if type(raw_path) is not str or type(text) is not str:
            raise CstKitCommitError("disposable worktree source is invalid")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise CstKitCommitError("disposable worktree path escapes its root")
        target = (resolved_root / relative).resolve()
        if not target.is_relative_to(resolved_root):
            raise CstKitCommitError("disposable worktree path escapes its root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return resolved_root


def open_kit_store_from_env(environ: Mapping[str, str] | None = None) -> Any | None:
    """Open the configured kit coordination store. ``None`` when unset."""

    directory = kit_coordination_dir_from_env(environ)
    if directory is None:
        return None
    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        DurableCoordinationStore,
    )

    directory.mkdir(parents=True, exist_ok=True)
    return DurableCoordinationStore(directory)


def _source_vfs_path(path: str) -> str:
    if type(path) is not str or not path or path.startswith(("/", "\\")):
        raise CstKitCommitError("cst source path is not a relative VFS path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CstKitCommitError("cst source path is not a relative VFS path")
    return "sources/" + path


def _ensure_vfs_parent(vfs: Any, path: str, operation_id: str) -> bool:
    from ipfs_kit_py.core.vfs.contracts import VFSErrorCode, VFSOperationKind
    from ipfs_kit_py.core.vfs.service import make_op

    parts = path.split("/")[:-1]
    acc: list[str] = []
    for index, part in enumerate(parts):
        acc.append(part)
        directory = "/".join(acc)
        stated = vfs.execute(
            make_op(
                VFSOperationKind.STAT,
                operation_id=f"{operation_id}-stat-{index}",
                path=directory,
            )
        )
        if stated.success:
            continue
        created = vfs.execute(
            make_op(
                VFSOperationKind.MKDIR,
                operation_id=f"{operation_id}-mkdir-{index}",
                path=directory,
            )
        )
        if created.success:
            continue
        error = created.result.error
        if error is None or error.code is not VFSErrorCode.ALREADY_EXISTS:
            return False
    return True


def _write_vfs_file(vfs: Any, path: str, payload: bytes, operation_id: str) -> bool:
    from ipfs_kit_py.core.vfs.contracts import VFSOperationKind
    from ipfs_kit_py.core.vfs.service import VFSExecuteRequest, make_op

    if not _ensure_vfs_parent(vfs, path, operation_id):
        return False
    stated = vfs.execute(
        make_op(VFSOperationKind.STAT, operation_id=f"{operation_id}-stat", path=path)
    )
    kind = VFSOperationKind.REPLACE if stated.success else VFSOperationKind.CREATE
    outcome = vfs.execute(
        make_op(kind, operation_id=operation_id, path=path),
        VFSExecuteRequest(payload=payload),
    )
    if not outcome.success:
        return False
    read = vfs.execute(
        make_op(VFSOperationKind.READ, operation_id=f"{operation_id}-read", path=path)
    )
    return bool(read.success and read.data == payload)


def commit_cst_sources_through_kit(
    store: Any,
    *,
    sources: Mapping[str, str],
    result_cid: str,
    pre_world_root_cid: str | None = None,
    workspace: str = "default",
) -> KitCstCommit:
    """Store the rewritten modules, CAS one successor root, then publish VFS.

    ``pre_world_root_cid`` must match the kit current root when it is set.
    An empty kit root accepts the first successor only when no predecessor
    was claimed. The git repository is not written.
    """

    from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
        ProgramGraphNode,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_state.program_identity import (
        SemanticWorldRootIdentity,
    )
    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        DurableCoordinationStore,
    )
    from ipfs_kit_py.mcp_server.mcplusplus.state_root_contracts import RootUpdateStatus
    from ipfs_kit_py.semantic_world_store import (
        SemanticWorldStore,
        publish_vfs_semantic_outbox,
    )
    from ipfs_kit_py.semantic_world_store.graph_history import (
        LogicalProgramGraphStore,
        append_program_graph_snapshot,
    )
    from ipfs_kit_py.semantic_world_store.recovery import (
        SemanticWorldCASAdmissionError,
        SemanticWorldRootRepository,
    )
    from ipfs_kit_py.semantic_world_store.vfs_outbox import SemanticOutboxAdmissionError
    from ipfs_kit_py.semantic_world_store.world_roots import (
        WorldRootAdmissionError,
        build_world_root_manifest,
    )

    if isinstance(store, SemanticWorldStore):
        facade = store
        coordination = facade.store
    elif isinstance(store, DurableCoordinationStore):
        facade = SemanticWorldStore(store)
        coordination = store
    else:
        raise CstKitCommitError(
            "kit store must be a DurableCoordinationStore or SemanticWorldStore"
        )
    if not sources:
        return _refused("cst_sources_missing")
    for path in sources:
        _source_vfs_path(str(path))

    token = _token(str(result_cid or "cst"))
    try:
        current = facade.current_world_root(workspace)
    except SemanticWorldCASAdmissionError:
        return _refused("world_root_cas_corrupt")
    if pre_world_root_cid:
        if current.root_cid != pre_world_root_cid:
            return _refused("world_root_predecessor_mismatch")
    elif current.generation != 0:
        pre_world_root_cid = current.root_cid

    generation = int(current.generation)
    graph = LogicalProgramGraphStore(coordination)
    nodes = []
    for path in sorted(sources):
        text = sources[path]
        if not isinstance(text, str):
            return _refused("cst_sources_missing")
        payload = text.encode("utf-8")
        nodes.append(
            ProgramGraphNode(
                node_kind="source",
                language="python",
                logical_name=str(path),
                source_cid=coordination.put_raw_bytes(payload),
                declaration_cid=coordination.put_raw_bytes(f"decl:{path}".encode("utf-8")),
            )
        )
    binding = graph.put_opaque_subroot(
        f"bind-{token}-g{generation}",
        operation_id=_operation("bind", token, generation),
    ).cid
    sealed = graph.put_opaque_subroot(
        f"seal-{token}-g{generation}",
        operation_id=_operation("seal", token, generation),
    ).cid
    try:
        written = append_program_graph_snapshot(
            graph,
            nodes=nodes,
            edges=[],
            environment_binding_set_cid=binding,
            sealed_binding_cid=sealed,
            operation_id=_operation("snap", token, generation),
        )
        snapshot = graph.get_verified_snapshot(written.cid)
    except Exception as exc:
        raise CstKitCommitError(str(exc)) from exc
    if not written.history_entry_cid:
        return _refused("graph_history_missing")

    def _opaque(label: str) -> str:
        return graph.put_opaque_subroot(
            f"{label}-{token}-g{generation}",
            operation_id=_operation(label, token, generation),
        ).cid

    try:
        identity = SemanticWorldRootIdentity(
            domain_state_cid=_opaque("domain"),
            canonical_program_graph_cid=snapshot.canonical_program_graph_cid,
            program_graph_snapshot_cid=snapshot.program_graph_snapshot_cid,
            semantic_object_index_cid=_opaque("objects"),
            environment_binding_set_cid=snapshot.environment_binding_set_cid,
            policy_cid=_opaque("policy"),
            analysis_limitation_index_cid=_opaque("limits"),
        )
        manifest = build_world_root_manifest(
            SemanticWorldRootRepository(coordination).roots,
            identity=identity,
            graph_history_head_cid=written.history_entry_cid,
            previous_manifest_cid=current.root_cid,
            generation=generation + 1,
            operation_id=_operation("manifest", token, generation),
        )
        cas = facade.compare_and_swap_world_root(
            workspace,
            expected_generation=generation,
            expected_root_cid=current.root_cid,
            new_root_cid=manifest.world_root_manifest_cid,
            operation_id=_operation("cas", token, generation),
        )
    except (WorldRootAdmissionError, SemanticWorldCASAdmissionError) as exc:
        raise CstKitCommitError(str(exc)) from exc
    if cas.supervisor_accepted:
        raise CstKitCommitError("kit CAS must not claim supervisor acceptance")
    if cas.status is not RootUpdateStatus.UPDATED:
        return _refused(f"world_root_cas_{cas.status.value}")
    root_cid = str(cas.after.root_cid or "")
    try:
        published = publish_vfs_semantic_outbox(
            facade.outbox,
            workspace,
            root_cid=root_cid,
            operation_id=_operation("outbox", token, generation),
            cas_result=cas,
        )
    except SemanticOutboxAdmissionError as exc:
        raise CstKitCommitError(str(exc)) from exc
    if published.supervisor_accepted:
        raise CstKitCommitError("VFS outbox must not claim supervisor acceptance")
    if not published.vfs_success or not published.durable_file_mutated:
        return KitCstCommit(
            applied=False,
            reason="vfs_publish_failed",
            root_cid=root_cid,
            generation=int(cas.after.generation),
        )
    published_bytes = facade.outbox.read_published()
    if published_bytes != root_cid.encode("utf-8"):
        return _refused("vfs_publish_failed")
    for index, path in enumerate(sorted(sources)):
        payload = sources[path].encode("utf-8")
        if not _write_vfs_file(
            facade.outbox.vfs,
            _source_vfs_path(path),
            payload,
            _operation(f"src{index}", token, generation),
        ):
            return KitCstCommit(
                applied=False,
                reason="vfs_source_publish_failed",
                root_cid=root_cid,
                generation=int(published.generation),
            )
    return KitCstCommit(
        applied=True,
        reason="kit_root_committed",
        root_cid=root_cid,
        generation=int(published.generation),
        vfs_published=True,
        vfs_path=published.path,
    )
