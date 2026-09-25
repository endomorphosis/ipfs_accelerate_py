"""Ops snapshot of TypeSafe side-channels. Never authority."""

from __future__ import annotations

from typing import Any


_TEXT_KEYS = (
    "line_text",
    "text",
    "original",
    "parts",
    "ranked",
    "candidates",
)


def _advisory(view: Any) -> dict[str, Any]:
    payload = dict(view) if isinstance(view, dict) else {}
    payload["accepted_as_authority"] = False
    for key in _TEXT_KEYS:
        payload.pop(key, None)
    return payload


def typesafe_ops_snapshot() -> dict[str, Any]:
    """Combine last guardrail, audit reduce, and source-edit lint receipts."""

    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_rank,
        last_artifact_view,
        last_claim_citation,
        last_clause_date,
        last_extracted_span,
        last_line_stitch,
        last_merge_conflict,
        last_parser_triage,
        last_producer_consumer,
        last_refactor_scope,
        last_source_edit_lint,
        last_static_lint,
        last_supporting_line,
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

    recovery = _last_closed_recovery()
    return {
        "accepted_as_authority": False,
        "trace_guardrail": _advisory(last_trace_guardrail()),
        "audit_reduce": _advisory(last_audit_reduce()),
        "source_edit_lint": _advisory(last_source_edit_lint()),
        "static_lint": _advisory(last_static_lint()),
        "refactor_scope": _advisory(last_refactor_scope()),
        "producer_consumer": _advisory(last_producer_consumer()),
        "artifact_view": _advisory(last_artifact_view()),
        "artifact_rank": _advisory(last_artifact_rank()),
        "parser_triage": _advisory(last_parser_triage()),
        "merge_conflict": _advisory(last_merge_conflict()),
        "claim_citation": _advisory(last_claim_citation()),
        "extracted_span": _advisory(last_extracted_span()),
        "clause_date": _advisory(last_clause_date()),
        "supporting_line": _advisory(last_supporting_line()),
        "line_stitch": _advisory(last_line_stitch()),
        "hammer_timeout_hint": _advisory(last_hammer_hint()),
        "unstall": _advisory(last_unstall_nomination()),
        "watchdog": _advisory(last_watchdog_classification()),
        "calibration": recommend_skip_policy(),
        "autoresearch": _advisory(_last_autoresearch()),
        "closed_recovery": _advisory(recovery),
        "recovery_plan_delta_outstanding": recovery.get("outstanding") is True,
        "completion_authority": False,
    }


def _last_closed_recovery() -> dict[str, Any]:
    try:
        from ipfs_accelerate_py.agent_supervisor.autonomy.closed_recovery import (
            last_closed_recovery,
            outstanding_recovery_work,
        )

        payload = last_closed_recovery()
        payload["outstanding"] = outstanding_recovery_work(payload)
        return payload
    except Exception:
        return {"outstanding": False}


def _last_autoresearch() -> dict[str, Any]:
    try:
        from ipfs_datasets_py.logic.integrations.typesafe_autoresearch import (
            last_autoresearch,
        )

        return last_autoresearch()
    except Exception:
        return {}


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
