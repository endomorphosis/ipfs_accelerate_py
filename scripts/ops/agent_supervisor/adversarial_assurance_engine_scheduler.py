#!/usr/bin/env python3
"""Fail-closed supervisor launcher for the Adversarial Assurance Engine.

This is the reviewed AAE binding of the existing multi-lane implementation
supervisor.  It adds no scheduler, task authority, provider, or execution
profile; the implementation remains in the shared supervisor runtime.
"""

from __future__ import annotations

from pathlib import Path


_SOURCE = Path(__file__).with_name("incremental_verification_planner_scheduler.py")
_source = _SOURCE.read_text(encoding="utf-8")
for _old, _new in (
    ("incremental-verification planner", "adversarial-assurance engine"),
    ("incremental-verification-planner", "adversarial-assurance-engine"),
    ("incremental_verification_planner", "adversarial_assurance_engine"),
    ("IVP", "AAE"),
    ("ivp", "aae"),
    (
        "config/agent_supervisor_adversarial_assurance_engine_scheduler.json",
        "config/adversarial_assurance_engine_scheduler.json",
    ),
):
    _source = _source.replace(_old, _new)

# The inherited IVP controller historically bound the common runner arguments
# to the cardinality of one particular profile.  AAE's protected controls may
# legitimately grow before its operator-owned profile is sealed.  Replace that
# brittle length check with equality against the exact arguments rendered from
# the admitted board.  This remains fail-closed and also detects value, order,
# duplication, or omission drift rather than checking only the final count.
_old_guard = (
    "    if len(tracks) != board.max_lanes or len(common) != 59:\n"
)
_new_guard = (
    "    expected_common = common_supervisor_args(board, implement=True)\n"
    "    if (\n"
    "        len(tracks) != board.max_lanes\n"
    "        or tuple(common) != expected_common\n"
    "    ):\n"
)
if _source.count(_old_guard) != 1:
    raise RuntimeError("inherited runner cardinality guard changed unexpectedly")
_source = _source.replace(_old_guard, _new_guard)

# IVP's original profile owns two planning gitlinks.  AAE also owns the scoped
# MCP++ schema/vector gitlink, so its launch preflight must bind all three
# declared repository authorities.  Keep the inherited clean/gitlink/HEAD
# checks and add only the third reviewed source-binding pair.
_old_gitlinks = (
    "    gitlink_specs = (\n"
    "        (\n"
    "            str(source.get(\"ipfs_kit_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_kit_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"ipfs_datasets_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_datasets_planning_revision\") or \"\"),\n"
    "        ),\n"
    "    )\n"
)
_new_gitlinks = (
    "    gitlink_specs = (\n"
    "        (\n"
    "            str(source.get(\"ipfs_kit_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_kit_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"ipfs_datasets_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_datasets_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"mcp_plus_plus_submodule_path\") or \"\"),\n"
    "            str(source.get(\"mcp_plus_plus_planning_revision\") or \"\"),\n"
    "        ),\n"
    "    )\n"
)
if _source.count(_old_gitlinks) != 1:
    raise RuntimeError("inherited source gitlink checks changed unexpectedly")
_source = _source.replace(_old_gitlinks, _new_gitlinks)

# AAE has one intentional, operator-owned release gate.  The ordinary IVP
# status projection treats any blocked task as a campaign-wide failure, which
# would report a false stall while independent pre-runtime work is progressing.
# Preserve the blocked count and expose the exact expected gate, but suppress
# only that one blocker from lifecycle failure.  Terminal detection remains
# unchanged and still requires all tasks completed with zero blocked work.
_old_blocker = (
    '    blockers: list[str] = []\n'
    '    if (counts["blocked_count"] or 0) > 0:\n'
    '        blockers.append(f"blocked_tasks_present:{counts[\'blocked_count\']}")\n'
)
_new_blocker = (
    '    blockers: list[str] = []\n'
    '    blocked_task_ids = {\n'
    '        str(value) for value in (task_payload.get("blocked_task_ids") or ())\n'
    '    }\n'
    '    expected_operator_gate_blocked = (\n'
    '        counts["blocked_count"] == 1\n'
    '        and blocked_task_ids == {"AAE-006"}\n'
    '    )\n'
    '    if (counts["blocked_count"] or 0) > 0 and not expected_operator_gate_blocked:\n'
    '        blockers.append(f"blocked_tasks_present:{counts[\'blocked_count\']}")\n'
)
if _source.count(_old_blocker) != 1:
    raise RuntimeError("inherited lane blocker projection changed unexpectedly")
_source = _source.replace(_old_blocker, _new_blocker)
_return_anchor = '        "blockers": blockers,\n        **counts,\n'
_return_replacement = (
    '        "blockers": blockers,\n'
    '        "expected_operator_gate_blocked": expected_operator_gate_blocked,\n'
    '        **counts,\n'
)
if _source.count(_return_anchor) != 1:
    raise RuntimeError("inherited lane status result changed unexpectedly")
_source = _source.replace(_return_anchor, _return_replacement)

# Execute a source-specialized copy so all constants, type names, lifecycle
# files, task prefix, and status schemas are AAE-bound before any command runs.
exec(compile(_source, str(Path(__file__)), "exec"), globals(), globals())
