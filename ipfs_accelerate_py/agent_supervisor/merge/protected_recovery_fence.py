"""Typed resolution of protected-checkout recovery fences.

A recovery journal is a fail-closed durable fence, not a task-terminal
status.  Peer owners must wait; generated-board dirtiness must be repaired
or operator-cleared; later commits that only touch unrelated protected
*code* must not pin the fence after generated outputs are clean.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

PROTECTED_RECOVERY_FENCE_RESOLUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protected-recovery-fence-resolution@1"
)
SUPERVISOR_RECOVERY_OWNER_SCRIPTS: Final[frozenset[str]] = frozenset(
    {
        "implementation_supervisor.py",
        "implementation_supervisor_entry.py",
    }
)
FENCE_CONTENTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "external_protected_checkout_recovery_required",
        "checkout_mutation_protected_recovery_required",
        "protected_recovery_owner_active",
        "supervisor_protected_recovery_owner_active",
        "protected_recovery_adoption_raced",
        "supervisor_protected_recovery_adoption_raced",
    }
)
_GENERATED_BOARD_SUFFIXES: Final[tuple[str, ...]] = (
    ".todo.md",
    ".objectives.md",
    ".todo.json",
)
_GENERATED_BOARD_PARTS: Final[frozenset[str]] = frozenset({"discovery"})
FENCE_CONTENTION_BACKOFF_SECONDS: Final[int] = 30


def is_supervisor_recovery_owner_script(owner_script: object) -> bool:
    """Return whether ``owner_script`` is a supervisor recovery journal owner."""

    name = Path(str(owner_script or "")).name
    return name in SUPERVISOR_RECOVERY_OWNER_SCRIPTS


def is_protected_recovery_fence_contention(reason: object) -> bool:
    """Return whether ``reason`` is a wait-for-peer fence, not a task defect."""

    return str(reason or "") in FENCE_CONTENTION_REASONS


def is_generated_board_output_path(path: object) -> bool:
    """Return whether ``path`` is a generated board/objectives/discovery output.

    Implementation-protected *code* (operator scripts, validators, receipts)
    can share a recovery journal's path set.  Untrusted later commits to
    those files must not keep a generated-dirty-repair fence pinned once the
    generated board outputs themselves are clean.
    """

    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return False
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or ".." in parsed.parts:
        return False
    lowered = parsed.as_posix().lower()
    if any(part.lower() in _GENERATED_BOARD_PARTS for part in parsed.parts):
        return True
    return lowered.endswith(_GENERATED_BOARD_SUFFIXES)


def generated_board_output_paths(paths: Sequence[object]) -> tuple[str, ...]:
    """Return the generated-board subset of a protected-path journal."""

    selected: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = str(raw or "").strip()
        if not path or path in seen or not is_generated_board_output_path(path):
            continue
        seen.add(path)
        selected.append(path)
    return tuple(selected)
