"""Ops snapshot of TypeSafe side-channels. Never authority."""

from __future__ import annotations

from typing import Any


def typesafe_ops_snapshot() -> dict[str, Any]:
    """Combine last guardrail, audit reduce, and source-edit lint receipts."""

    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_source_edit_lint,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_guard import (
        last_trace_guardrail,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_reduce import (
        last_audit_reduce,
    )

    return {
        "accepted_as_authority": False,
        "trace_guardrail": last_trace_guardrail(),
        "audit_reduce": last_audit_reduce(),
        "source_edit_lint": last_source_edit_lint(),
    }


def typesafe_ops_snapshot_json() -> str:
    import json

    return json.dumps(typesafe_ops_snapshot(), sort_keys=True, separators=(",", ":"))


__all__ = ["typesafe_ops_snapshot", "typesafe_ops_snapshot_json"]
