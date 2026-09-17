"""Ops snapshot of TypeSafe side-channels. Never authority."""

from __future__ import annotations

from typing import Any


def typesafe_ops_snapshot() -> dict[str, Any]:
    """Combine last guardrail, audit reduce, and source-edit lint receipts."""

    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_view,
        last_source_edit_lint,
        last_static_lint,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_guard import (
        last_trace_guardrail,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_reduce import (
        last_audit_reduce,
    )

    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
        recommend_skip_policy,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor import (
        last_hammer_hint,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
        last_unstall_nomination,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_watchdog import (
        last_watchdog_classification,
    )

    return {
        "accepted_as_authority": False,
        "trace_guardrail": last_trace_guardrail(),
        "audit_reduce": last_audit_reduce(),
        "source_edit_lint": last_source_edit_lint(),
        "static_lint": last_static_lint(),
        "artifact_view": last_artifact_view(),
        "hammer_timeout_hint": last_hammer_hint(),
        "unstall": last_unstall_nomination(),
        "watchdog": last_watchdog_classification(),
        "calibration": recommend_skip_policy(),
    }


def typesafe_ops_snapshot_json() -> str:
    import json

    return json.dumps(typesafe_ops_snapshot(), sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    """Print the snapshot. Never authority. Never fails closed on TypeSafe."""

    del argv
    try:
        print(typesafe_ops_snapshot_json())
    except Exception:
        import json

        print(
            json.dumps(
                {"accepted_as_authority": False, "error": "snapshot_unavailable"},
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "typesafe_ops_snapshot", "typesafe_ops_snapshot_json"]
