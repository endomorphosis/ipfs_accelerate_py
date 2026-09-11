"""Ephemeral native authority to launch only an independent-work daemon.

This is deliberately not a restart receipt. The controller owns the Quack
mutation fence and the managed-child launch lock for the entire call. A fresh
native lane owner audits its remaining population; its process-local scope is
consumed before the execution writer closes and the child starts. The child
must independently refresh its own acknowledgement before claiming work.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from . import owner_task_quarantine as local
from ..merge import workspace_quarantine as workspace
from ..task_sources.owner_task_quarantine import require
from ..task_sources.control_plane_contracts import content_identity

_TOKEN = object()


def _source_identity(source: Mapping[str, Any]) -> str:
    require(
        type(source) is dict
        and bool(source.get("repository_root"))
        and bool(source.get("repository_revision"))
        and bool(source.get("control_plane_tree_id"))
        and bool(source.get("source_id"))
        and type(source.get("sources")) is list
        and bool(source["sources"])
        and all(row.get("available") is True for row in source["sources"]),
        "independent_launch_source_unavailable",
    )
    # repository_revision is an observation, not the supervisor's source
    # generation. Match the existing reload watcher: unrelated repository
    # commits must not strand a controller whose control-plane code is intact.
    return content_identity({key: source[key] for key in (
        "repository_root", "control_plane_tree_id", "source_id", "sources",
    )})


class _NativeLaunchScope:
    def __init__(self, token, daemon, binding, imported_source, source_probe):
        require(token is _TOKEN, "independent_launch_native_scope_required")
        self._daemon = daemon
        self._binding = content_identity(binding)
        self._source = _source_identity(imported_source)
        self._source_probe = source_probe
        self._heads = daemon.task_source.intent.owner_task_quarantines()
        self._active = True
        self._deadline = time.monotonic() + 30.0
        self._consumed = False
        self._validate()

    def _validate(self):
        require(self._active and not self._consumed and time.monotonic() < self._deadline,
                "independent_launch_scope_expired")
        daemon = self._daemon
        local.require_owner(daemon)
        require(_source_identity(self._source_probe()) == self._source,
                "independent_launch_loaded_source_changed")
        current = daemon.task_source.intent.owner_task_quarantines()
        require(bool(current) and current == self._heads,
                "independent_launch_owner_or_population_changed")
        local.current(daemon)
        bridge = daemon._database_portal_bridge
        require(bridge is not None, "independent_launch_native_bridge_missing")
        # Every current head names a whole frozen root. Verify native registry
        # evidence for every root without opening any foreign execution writer.
        roots = {}
        for head in current.values():
            require(time.monotonic() < self._deadline, "independent_launch_scope_expired")
            root = head["workspace_root"]
            if root not in roots:
                roots[root] = workspace.verify(bridge.workspace_repository_root, Path(root))
            frozen = roots[root]
            require(frozen["cid"] == head["workspace_custody_cid"]
                    and frozen["fresh_root"] == head["fresh_workspace_root"],
                    "independent_launch_workspace_custody_changed")
        daemon._assert_owner_quarantine_independent_admission()
        require(time.monotonic() < self._deadline, "independent_launch_scope_expired")

    def diagnostic(self):
        self._validate()
        return {
            "reason": "native_independent_custody_prelaunch_admitted",
            "independent_work_admitted": True,
            "safe_to_restart": False,
            "quiesced": False,
            "reconciled": False,
            "completion_authorized": False,
            "owner_task_quarantines": sorted(self._heads),
        }

    def consume(self, binding, start_child: Callable[[], Any]):
        """Consume within the controller's still-held owner and launch locks."""
        require(content_identity(binding) == self._binding,
                "independent_launch_child_binding_changed")
        self._validate()
        self._consumed = True
        # A new native child cannot take the same execution writer until this
        # old process instance has released it. No callback is settled here.
        self._daemon.close()
        return start_child()


@contextmanager
def native_independent_launch_scope(*, daemon, binding, imported_source, source_probe):
    """Audit an actual native owner; caller-supplied diagnostic flags are unused."""
    from .implementation_daemon import DatabaseImplementationDaemon
    require(type(daemon) is DatabaseImplementationDaemon
            and daemon.authority_mode == "quack"
            and daemon._database_portal_bridge is not None,
            "independent_launch_native_owner_required")
    with daemon._lock:
        local.require_owner(daemon)
        require(bool(daemon.task_source.intent.owner_task_quarantines()),
                "independent_launch_central_custody_absent")
        local.refresh(daemon)
        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="native_independent_child_launch", force=True,
        )
        retained = local.current(daemon)
        require(result.get("blocked") is False
                and result.get("repair_batch_pending") is not True
                and result.get("continuation_required") is not True
                and ((bool(retained) and result.get("independent_work_admitted") is True)
                     or (not retained and result.get("reconciled") is True)),
                "independent_launch_remaining_population_unproved")
        scope = _NativeLaunchScope(_TOKEN, daemon, binding, imported_source, source_probe)
        try:
            yield scope
        finally:
            scope._active = False
