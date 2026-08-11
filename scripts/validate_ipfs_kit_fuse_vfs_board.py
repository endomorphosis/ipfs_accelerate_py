#!/usr/bin/env python3
"""Fail-closed semantic validator for the IPFS Kit kernel-VFS board."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.todo.md"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json"

NAMESPACE = "ipfs-kit-kernel-vfs-fuse-v1"
BRANCH = "agent/ipfs-kit-fuse-vfs"
ACCELERATOR_ANCESTOR = "ea11293bb996f052d620eae989f5377a956764b1"
IPFS_KIT_REVISION = "69091bf8f11a3ef1fb0e04e11a6d8a4c87f3fa78"

TASK_IDS = (
    "KVFS-000",
    "KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108",
    "KVFS-200", "KVFS-202", "KVFS-201", "KVFS-204", "KVFS-203",
    "KVFS-205", "KVFS-208", "KVFS-210", "KVFS-206",
    "KVFS-303", "KVFS-309", "KVFS-300", "KVFS-301", "KVFS-304",
    "KVFS-400", "KVFS-401", "KVFS-404", "KVFS-403",
    "KVFS-503", "KVFS-500", "KVFS-506", "KVFS-501",
    "KVFS-608", "KVFS-600", "KVFS-601", "KVFS-603",
    "KVFS-703", "KVFS-701", "KVFS-700", "KVFS-702",
    "KVFS-808", "KVFS-800", "KVFS-802", "KVFS-801", "KVFS-811",
)
GOAL_IDS = (
    "KVFS-G000", "KVFS-G100", "KVFS-G200", "KVFS-G300",
    "KVFS-G400", "KVFS-G500", "KVFS-G600", "KVFS-G700", "KVFS-G800",
)
INITIAL_COMPLETED = ("KVFS-000",)
INITIAL_READY = ("KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108")
INITIAL_SHARDS = {
    0: "KVFS-103",
    1: "KVFS-101",
    2: "KVFS-108",
    3: "KVFS-100",
}
TERMINAL_TASK = "KVFS-811"
RETRY_BUDGET_REPAIR_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1"
)
RECONCILIATION_GUARDRAIL_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1"
)
RECONCILIATION_RESOLUTION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.reconciliation-resolution@1"
)
MAX_DISCOVERY_EVIDENCE_BYTES = 1_048_576
MAX_OPERATIONAL_REPAIR_TASKS = 16
FIRST_OPERATIONAL_REPAIR_NUMBER = max(
    int(task_id.rsplit("-", 1)[1]) for task_id in TASK_IDS
) + 1
PERSISTED_OPERATIONAL_STATES = frozenset({"todo", "completed"})
RECONCILIATION_REASONS_BY_KIND = {
    "dirty_backlogged_worktree": frozenset(
        {
            "content_not_in_target",
            "dirty_worktree",
            "empty_status_path",
            "unsupported_status",
        }
    ),
    "main_checkout_dirty": frozenset({"main_checkout_dirty"}),
    "preflight_merge_conflict": frozenset({"preflight_merge_conflict"}),
}
MAX_ACTIVE_OPERATIONAL_RECONCILIATION_TASKS = sum(
    len(reasons) for reasons in RECONCILIATION_REASONS_BY_KIND.values()
)
RECONCILIATION_OUTPUTS = (
    "data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery",
    "docs/architecture/ipfs_kit_fuse_vfs.todo.md",
)
RECONCILIATION_PROFILE = {
    "board namespace": NAMESPACE,
    "goal id": "KVFS-G800",
    "bundle": "ipfs-kit/kernel-vfs/release/terminal",
    "parallel lane": "release-terminal",
    "resource class": "cpu-medium",
}

RETRY_BUDGET_REPAIR_TITLE_RE = re.compile(
    r"^Resolve\s+(?P<kind>validation|implementation|merge)\s+"
    r"retry-budget\s+failure\s+for\s+(?P<source>KVFS-\d{3})$",
    re.IGNORECASE,
)
RETRY_BUDGET_REPAIR_ACCEPTANCE_RE = re.compile(
    r"\b(?:release|remove)\s+(?P<source>KVFS-\d{3})\s+from\s+"
    r"(?:the\s+)?strategy\s+blocked_tasks\b",
    re.IGNORECASE,
)

TASK_DEPENDENCIES = {
    "KVFS-000": (),
    "KVFS-100": ("KVFS-000",),
    "KVFS-101": ("KVFS-000",),
    "KVFS-103": ("KVFS-000",),
    "KVFS-108": ("KVFS-000",),
    "KVFS-200": ("KVFS-100", "KVFS-101"),
    "KVFS-202": ("KVFS-100", "KVFS-101"),
    "KVFS-201": ("KVFS-101", "KVFS-103"),
    "KVFS-204": ("KVFS-101", "KVFS-103"),
    "KVFS-203": ("KVFS-200", "KVFS-201", "KVFS-202", "KVFS-204"),
    "KVFS-205": ("KVFS-200", "KVFS-204"),
    "KVFS-208": ("KVFS-202", "KVFS-204"),
    "KVFS-210": ("KVFS-203",),
    "KVFS-206": ("KVFS-203", "KVFS-205", "KVFS-208", "KVFS-210"),
    "KVFS-303": ("KVFS-100", "KVFS-101"),
    "KVFS-309": ("KVFS-203", "KVFS-205", "KVFS-303"),
    "KVFS-300": ("KVFS-309",),
    "KVFS-301": ("KVFS-208", "KVFS-300", "KVFS-309"),
    "KVFS-304": ("KVFS-301",),
    "KVFS-400": ("KVFS-101", "KVFS-103", "KVFS-200"),
    "KVFS-401": ("KVFS-203", "KVFS-400"),
    "KVFS-404": ("KVFS-301", "KVFS-309", "KVFS-401"),
    "KVFS-403": ("KVFS-304", "KVFS-400", "KVFS-404"),
    "KVFS-503": ("KVFS-100", "KVFS-108"),
    "KVFS-500": ("KVFS-206", "KVFS-300", "KVFS-404", "KVFS-503"),
    "KVFS-506": ("KVFS-301", "KVFS-304", "KVFS-403", "KVFS-500"),
    "KVFS-501": ("KVFS-506",),
    "KVFS-608": ("KVFS-100", "KVFS-108", "KVFS-503"),
    "KVFS-600": ("KVFS-201", "KVFS-202", "KVFS-608"),
    "KVFS-601": ("KVFS-206", "KVFS-300", "KVFS-301", "KVFS-404", "KVFS-600"),
    "KVFS-603": ("KVFS-403", "KVFS-601"),
    "KVFS-703": ("KVFS-503", "KVFS-608"),
    "KVFS-701": ("KVFS-500", "KVFS-703"),
    "KVFS-700": ("KVFS-506", "KVFS-701"),
    "KVFS-702": ("KVFS-500", "KVFS-601", "KVFS-703"),
    "KVFS-808": ("KVFS-500", "KVFS-601"),
    "KVFS-800": ("KVFS-206", "KVFS-301", "KVFS-403", "KVFS-600"),
    "KVFS-802": ("KVFS-506", "KVFS-603", "KVFS-700", "KVFS-800"),
    "KVFS-801": ("KVFS-403", "KVFS-506", "KVFS-603", "KVFS-700"),
    "KVFS-811": (
        "KVFS-501", "KVFS-603", "KVFS-700", "KVFS-702", "KVFS-808",
        "KVFS-802", "KVFS-800", "KVFS-801",
    ),
}

TASK_GROUPS = {
    "KVFS-G100": ("KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108"),
    "KVFS-G200": (
        "KVFS-200", "KVFS-202", "KVFS-201", "KVFS-204", "KVFS-203",
        "KVFS-205", "KVFS-208", "KVFS-210", "KVFS-206",
    ),
    "KVFS-G300": ("KVFS-303", "KVFS-309", "KVFS-300", "KVFS-301", "KVFS-304"),
    "KVFS-G400": ("KVFS-400", "KVFS-401", "KVFS-404", "KVFS-403"),
    "KVFS-G500": ("KVFS-503", "KVFS-500", "KVFS-506", "KVFS-501"),
    "KVFS-G600": ("KVFS-608", "KVFS-600", "KVFS-601", "KVFS-603"),
    "KVFS-G700": ("KVFS-703", "KVFS-701", "KVFS-700", "KVFS-702"),
    "KVFS-G800": ("KVFS-808", "KVFS-800", "KVFS-802", "KVFS-801", "KVFS-811"),
}

GOAL_DEPENDENCIES = {
    "KVFS-G000": (),
    "KVFS-G100": (),
    "KVFS-G200": ("KVFS-G100",),
    "KVFS-G300": ("KVFS-G100",),
    "KVFS-G400": ("KVFS-G100",),
    "KVFS-G500": ("KVFS-G200", "KVFS-G300", "KVFS-G400"),
    "KVFS-G600": ("KVFS-G200", "KVFS-G300", "KVFS-G400"),
    "KVFS-G700": ("KVFS-G500", "KVFS-G600"),
    "KVFS-G800": ("KVFS-G500", "KVFS-G600", "KVFS-G700"),
}

REQUIRED_TASK_FIELDS = (
    "status", "completion", "is schedulable", "review only", "priority",
    "track", "depends on", "goal id", "board namespace", "outputs",
    "validation", "scope paths", "conflict policy", "acceptance",
)
REQUIRED_GOAL_FIELDS = (
    "status", "parent", "depends on", "fib priority", "track", "priority",
    "bundle", "goal", "evidence", "outputs", "validation", "acceptance",
    "gap task", "refinement", "conflict policy",
)
PROTECTED_PATHS = (
    ".gitignore",
    "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md",
    "docs/architecture/ipfs_kit_fuse_vfs.objectives.md",
    "docs/architecture/ipfs_kit_fuse_vfs.todo.md",
    "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json",
    "scripts/validate_ipfs_kit_fuse_vfs_board.py",
    "test/api/test_ipfs_kit_fuse_vfs_board.py",
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _blocks(path: Path, pattern: re.Pattern[str]) -> list[tuple[str, str, dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    matches = list(pattern.finditer(text))
    records: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        prior = ""
        for line in text[match.end():end].splitlines():
            field = re.match(r"^- ([^:]+):\s*(.*)$", line)
            if field:
                prior = field.group(1).strip().lower()
                if prior in fields:
                    raise ValueError(f"{match.group(1)} duplicates field {prior!r}")
                fields[prior] = field.group(2).strip()
            elif prior and line.startswith(("  ", "\t")) and line.strip():
                fields[prior] = f"{fields[prior]} {line.strip()}".strip()
        records.append((match.group(1), match.group(2).strip(), fields))
    return records


def parse_tasks() -> list[tuple[str, str, dict[str, str]]]:
    return _blocks(TODO_PATH, re.compile(r"^## (KVFS-\d{3}) (.+)$", re.MULTILINE))


def parse_goals() -> list[tuple[str, str, dict[str, str]]]:
    return _blocks(OBJECTIVE_PATH, re.compile(r"^## (KVFS-G\d{3}) (.+)$", re.MULTILINE))


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\x00" not in value


def _path_is_within_scope(value: str, scope: str) -> bool:
    """Return whether a safe relative path is owned by a declared scope path."""

    value_path = PurePosixPath(value)
    scope_path = PurePosixPath(scope)
    return value_path == scope_path or value_path.parts[: len(scope_path.parts)] == scope_path.parts


def _git_common_dir(repo_root: Path) -> Path | None:
    """Return the shared Git directory for one live worktree, if available."""

    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "--git-common-dir"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = repo_root / common_dir
    return common_dir.resolve(strict=False)


def _git_toplevel(repo_root: Path) -> Path | None:
    """Return the exact worktree root containing ``repo_root``."""

    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "--show-toplevel"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve(strict=False)


def _supervisor_owned_discovery_path(value: str) -> bool:
    """Bind absolute runtime evidence to this board's Git repository.

    Operational cards are generated in the running supervisor worktree, while
    this validator is also run from isolated repair worktrees.  Requiring the
    exact runtime suffix plus the same Git common directory admits those sibling
    worktrees without trusting an arbitrary absolute path with a matching name.
    """

    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    suffix = PurePosixPath(RECONCILIATION_OUTPUTS[0])
    if (
        not normalized
        or not path.is_absolute()
        or ".." in path.parts
        or "\x00" in normalized
        or len(path.parts) <= len(suffix.parts)
        or path.parent.parts[-len(suffix.parts):] != suffix.parts
    ):
        return False

    discovery_path = Path(normalized)
    if discovery_path.is_symlink() or discovery_path.parent.is_symlink():
        return False
    candidate_repo_root = discovery_path.parent
    for _part in suffix.parts:
        candidate_repo_root = candidate_repo_root.parent
    if discovery_path.parent.resolve(strict=False) != (
        candidate_repo_root.resolve(strict=False) / Path(suffix.as_posix())
    ):
        return False
    candidate_repo_root = candidate_repo_root.resolve(strict=False)
    if _git_toplevel(candidate_repo_root) != candidate_repo_root:
        return False
    expected_common_dir = _git_common_dir(REPO_ROOT)
    candidate_common_dir = _git_common_dir(candidate_repo_root)
    return (
        expected_common_dir is not None
        and candidate_common_dir is not None
        and candidate_common_dir == expected_common_dir
    )


def _read_bounded_regular_file(
    task_id: str,
    path: Path,
    *,
    errors: list[str],
) -> str | None:
    """Read immutable-sized discovery evidence without following a symlink."""

    try:
        before = path.lstat()
    except OSError:
        errors.append(f"{task_id} discovery evidence is unavailable")
        return None
    if not stat.S_ISREG(before.st_mode):
        errors.append(
            f"{task_id} discovery evidence must be a regular non-symlink file"
        )
        return None
    if before.st_size > MAX_DISCOVERY_EVIDENCE_BYTES:
        errors.append(f"{task_id} discovery evidence exceeds 1 MiB")
        return None

    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            payload = handle.read(MAX_DISCOVERY_EVIDENCE_BYTES + 1)
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError:
        errors.append(f"{task_id} discovery evidence is unavailable")
        return None
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        or (
            opened_after.st_size,
            opened_after.st_mtime_ns,
        ) != (opened.st_size, opened.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
    ):
        errors.append(
            f"{task_id} discovery evidence must be a stable regular file"
        )
        return None
    if len(payload) > MAX_DISCOVERY_EVIDENCE_BYTES:
        errors.append(f"{task_id} discovery evidence exceeds 1 MiB")
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{task_id} discovery evidence is not UTF-8")
        return None


def _resolution_receipt_digest(receipt: Mapping[str, object]) -> str:
    """Return the canonical digest of a receipt without its own digest field."""

    payload = {
        str(key): value
        for key, value in receipt.items()
        if str(key) != "receipt_digest"
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _validate_reconciliation_manifest(
    *,
    task_id: str,
    fields: Mapping[str, str],
    discovery_text: str,
    candidate_count: int,
    errors: list[str],
) -> None:
    """Bind the generated card to its machine-readable discovery manifest."""

    matches = re.findall(
        r"^## Machine Readable Manifest\s*\n\s*```json\s*\n(.*?)\n```",
        discovery_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(matches) != 1:
        errors.append(
            f"{task_id} must have one machine-readable reconciliation manifest"
        )
        return
    try:
        manifest = json.loads(matches[0])
    except json.JSONDecodeError:
        errors.append(f"{task_id} reconciliation manifest is malformed")
        return
    if not isinstance(manifest, dict):
        errors.append(f"{task_id} reconciliation manifest must be an object")
        return
    if (
        manifest.get("fingerprint")
        != fields.get("reconciliation fingerprint")
        or manifest.get("kind") != fields.get("reconciliation kind")
        or manifest.get("reason") != fields.get("reconciliation reason")
        or manifest.get("dedupe_key") != fields.get("dedupe key")
        or type(manifest.get("candidate_count")) is not int
        or manifest.get("candidate_count") != candidate_count
    ):
        errors.append(f"{task_id} reconciliation manifest binding mismatch")


def _validate_reconciliation_resolution_receipt(
    *,
    task_id: str,
    fields: Mapping[str, str],
    discovery_text: str,
    candidate_count: int,
    errors: list[str],
) -> None:
    """Require content-addressed postconditions before a guardrail completes."""

    matches = re.findall(
        r"^## Resolution Receipt\s*\n\s*```json\s*\n(.*?)\n```",
        discovery_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(matches) != 1:
        errors.append(
            f"{task_id} must have one machine-readable resolution receipt"
        )
        return
    try:
        receipt = json.loads(matches[0])
    except json.JSONDecodeError:
        errors.append(f"{task_id} resolution receipt is malformed")
        return
    if not isinstance(receipt, dict):
        errors.append(f"{task_id} resolution receipt must be an object")
        return
    if (
        receipt.get("schema") != RECONCILIATION_RESOLUTION_SCHEMA
        or receipt.get("task_id") != task_id
        or receipt.get("reconciliation_fingerprint")
        != fields.get("reconciliation fingerprint")
        or receipt.get("kind") != fields.get("reconciliation kind")
        or receipt.get("reason") != fields.get("reconciliation reason")
        or receipt.get("resolved") is not True
    ):
        errors.append(f"{task_id} resolution receipt binding mismatch")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
        str(receipt.get("resolved_at") or ""),
    ) is None:
        errors.append(f"{task_id} resolution timestamp is invalid")
    if re.fullmatch(
        r"[a-z][a-z0-9_]{2,127}",
        str(receipt.get("resolution_method") or ""),
    ) is None:
        errors.append(f"{task_id} resolution method is invalid")

    postconditions = receipt.get("postconditions")
    if not (
        isinstance(postconditions, dict)
        and type(postconditions.get("candidate_count_before")) is int
        and postconditions.get("candidate_count_before") == candidate_count
        and type(postconditions.get("candidate_count_after")) is int
        and postconditions.get("candidate_count_after") == 0
        and postconditions.get("active_blocker_present_after") is False
        and type(postconditions.get("dirty_worktree_group_count_after")) is int
        and postconditions.get("dirty_worktree_group_count_after") == 0
        and type(postconditions.get("cleanup_skip_count_after")) is int
        and postconditions.get("cleanup_skip_count_after") == 0
    ):
        errors.append(f"{task_id} resolution postconditions are incomplete")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        errors.append(f"{task_id} resolution evidence is empty")

    receipt_digest = str(receipt.get("receipt_digest") or "")
    if receipt_digest != _resolution_receipt_digest(receipt):
        errors.append(f"{task_id} resolution receipt digest mismatch")
    if fields.get("resolution receipt digest") != receipt_digest:
        errors.append(f"{task_id} resolution receipt anchor mismatch")


def _partition_canonical_and_operational_tasks(
    tasks: Iterable[tuple[str, str, dict[str, str]]],
    errors: list[str],
) -> tuple[
    list[tuple[str, str, dict[str, str]]],
    list[tuple[str, str, dict[str, str]]],
]:
    """Keep the sealed program separate from generated recovery appendices."""

    canonical: list[tuple[str, str, dict[str, str]]] = []
    operational: list[tuple[str, str, dict[str, str]]] = []
    appendix_started = False
    all_ids: list[str] = []
    for task in tasks:
        task_id = task[0]
        all_ids.append(task_id)
        if task_id in TASK_IDS:
            if appendix_started:
                errors.append(
                    f"canonical task {task_id} appears after the operational appendix"
                )
            canonical.append(task)
            continue
        appendix_started = True
        operational.append(task)
    if len(all_ids) != len(set(all_ids)):
        errors.append("task IDs are not unique")
    return canonical, operational


def _looks_like_reconciliation_guardrail(
    title: str,
    fields: Mapping[str, str],
) -> bool:
    """Recognize reconciliation-shaped cards before checking their schema."""

    return (
        fields.get("generated by") == RECONCILIATION_GUARDRAIL_SCHEMA
        or fields.get("dedupe key", "").startswith("reconciliation_guardrail:")
        or fields.get("blocked reason") == "operator_reconciliation_required"
        or any(key.startswith("reconciliation ") for key in fields)
        or title.startswith("Resolve dirty main checkout blocking ")
        or title.startswith("Resolve ")
        and (
            " preflight-conflicting backlogged worktree merges" in title
            or " dirty backlogged worktrees blocked by " in title
        )
    )


def _validate_reconciliation_guardrail_task(
    task_id: str,
    title: str,
    fields: Mapping[str, str],
    *,
    errors: list[str],
) -> bool:
    """Validate one bounded, operator-gated reconciliation appendix card."""

    status = fields.get("status", "")
    if (
        fields.get("generated by") != RECONCILIATION_GUARDRAIL_SCHEMA
        or fields.get("canonical board task") != "false"
    ):
        errors.append(f"{task_id} lacks exact reconciliation provenance")

    kind = fields.get("reconciliation kind", "")
    reason = fields.get("reconciliation reason", "")
    kind_is_supported = kind in RECONCILIATION_REASONS_BY_KIND
    if not kind_is_supported:
        errors.append(f"{task_id} has unsupported reconciliation kind {kind!r}")
    elif reason not in RECONCILIATION_REASONS_BY_KIND[kind]:
        errors.append(
            f"{task_id} has unsupported reconciliation reason {reason!r} "
            f"for {kind}"
        )

    fingerprint = fields.get("reconciliation fingerprint", "")
    if (
        re.fullmatch(r"[0-9a-f]{40}", fingerprint) is None
        or fields.get("fingerprint") != fingerprint
    ):
        errors.append(f"{task_id} reconciliation fingerprint mismatch")

    if kind_is_supported:
        expected_dedupe = {
            "main_checkout_dirty": (
                "reconciliation_guardrail:main_checkout_dirty"
            ),
            "preflight_merge_conflict": (
                "reconciliation_guardrail:preflight_merge_conflict"
            ),
            "dirty_backlogged_worktree": (
                f"reconciliation_guardrail:dirty_backlogged_worktree:{reason}"
            ),
        }[kind]
        if fields.get("dedupe key") != expected_dedupe:
            errors.append(f"{task_id} reconciliation dedupe key mismatch")

    if status not in {"blocked", "completed"}:
        errors.append(f"{task_id} has unsafe reconciliation status {status!r}")
    if fields.get("completion") != "manual":
        errors.append(f"{task_id} must use manual reconciliation completion")
    if kind_is_supported:
        expected_priority = (
            "P1"
            if kind != "dirty_backlogged_worktree"
            or reason == "unsupported_status"
            else "P2"
        )
        if fields.get("priority") != expected_priority:
            errors.append(f"{task_id} reconciliation priority mismatch")
    if fields.get("track") != "ops":
        errors.append(f"{task_id} must use the ops track")
    if (
        fields.get("is schedulable") != "false"
        or fields.get("review only") != "true"
        or fields.get("blocked reason")
        != "operator_reconciliation_required"
    ):
        errors.append(f"{task_id} reconciliation authority gate mismatch")

    if _csv(fields.get("depends on", "")):
        errors.append(
            f"{task_id} reconciliation appendix must not alter the sealed DAG"
        )
    if _csv(fields.get("outputs", "")) != RECONCILIATION_OUTPUTS:
        errors.append(f"{task_id} reconciliation output scope mismatch")
    if any(
        field in fields
        for field in ("scope paths", "conflict policy", "graph parents")
    ):
        errors.append(f"{task_id} reconciliation scope authority is unsafe")
    for field, expected in RECONCILIATION_PROFILE.items():
        if fields.get(field) != expected:
            errors.append(f"{task_id} reconciliation {field} mismatch")

    title_match: re.Match[str] | None = None
    candidate_count: int | None = None
    if kind_is_supported:
        title_patterns = {
            "main_checkout_dirty": (
                r"^Resolve dirty main checkout blocking (?P<count>[1-9]\d*) "
                r"worktree merges$"
            ),
            "preflight_merge_conflict": (
                r"^Resolve (?P<count>[1-9]\d*) preflight-conflicting "
                r"backlogged worktree merges$"
            ),
            "dirty_backlogged_worktree": (
                rf"^Resolve (?P<count>[1-9]\d*) dirty backlogged worktrees "
                rf"blocked by {re.escape(reason)}$"
            ),
        }
        title_match = re.fullmatch(title_patterns[kind], title)
        if title_match is None:
            errors.append(f"{task_id} reconciliation title mismatch")
        else:
            candidate_count = int(title_match.group("count"))

    discovery_text = fields.get("reconciliation discovery", "")
    discovery_path = PurePosixPath(discovery_text.replace("\\", "/"))
    expected_name = (
        rf"\d{{4}}-\d{{2}}-\d{{2}}-{task_id.lower()}-"
        rf"reconciliation-{fingerprint[:12]}\.md"
    )
    discovery_is_valid = (
        _supervisor_owned_discovery_path(discovery_text)
        and re.fullmatch(expected_name, discovery_path.name) is not None
    )
    if not discovery_is_valid:
        errors.append(f"{task_id} has invalid reconciliation discovery provenance")

    discovery_evidence: str | None = None
    if discovery_is_valid:
        discovery_evidence = _read_bounded_regular_file(
            task_id,
            Path(discovery_text),
            errors=errors,
        )
    if discovery_evidence is not None and candidate_count is not None:
        _validate_reconciliation_manifest(
            task_id=task_id,
            fields=fields,
            discovery_text=discovery_evidence,
            candidate_count=candidate_count,
            errors=errors,
        )
        if status == "completed":
            _validate_reconciliation_resolution_receipt(
                task_id=task_id,
                fields=fields,
                discovery_text=discovery_evidence,
                candidate_count=candidate_count,
                errors=errors,
            )
    if status == "completed" and not fields.get("resolution receipt digest"):
        errors.append(f"{task_id} resolution receipt anchor mismatch")
    if status != "completed" and fields.get("resolution receipt digest"):
        errors.append(
            f"{task_id} blocked reconciliation has a stale receipt anchor"
        )

    try:
        validation = shlex.split(fields.get("validation", ""))
    except ValueError:
        validation = []
    if validation != ["test", "-f", discovery_text]:
        errors.append(f"{task_id} reconciliation validation is not fail-closed")

    acceptance = fields.get("acceptance", "")
    if title_match is not None:
        candidate_count_text = title_match.group("count")
        acceptance_fragments = (
            discovery_text,
            (
                f"because {candidate_count_text} branch or worktree cleanup "
                "candidates"
            ),
            f"blocked by {reason}",
            "intentionally operator-gated",
            "blocked candidate count decreases",
        )
        if any(fragment not in acceptance for fragment in acceptance_fragments):
            errors.append(
                f"{task_id} reconciliation acceptance/evidence mismatch"
            )
    elif not acceptance:
        errors.append(f"{task_id} has empty reconciliation acceptance")

    return status != "completed"


def _validate_operational_repair_tasks(
    repairs: Iterable[tuple[str, str, dict[str, str]]],
    *,
    canonical_by_id: Mapping[str, dict[str, str]],
    errors: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate bounded operational cards outside the sealed canonical DAG."""

    repair_list = list(repairs)
    if len(repair_list) > MAX_OPERATIONAL_REPAIR_TASKS:
        errors.append("operational appendix exceeds its finite bound")

    operational_ids: list[str] = []
    pending_ids: list[str] = []
    active_source_kinds: set[tuple[str, str]] = set()
    active_reconciliation_count = 0
    reconciliation_dedupe_status: dict[str, str] = {}
    reconciliation_fingerprint_status: dict[str, str] = {}
    for offset, (task_id, title, fields) in enumerate(repair_list):
        operational_ids.append(task_id)
        expected_id = f"KVFS-{FIRST_OPERATIONAL_REPAIR_NUMBER + offset:03d}"
        if task_id != expected_id:
            errors.append(
                "operational appendix IDs must be contiguous: "
                f"expected {expected_id}, got {task_id}"
            )

        if _looks_like_reconciliation_guardrail(title, fields):
            is_pending = _validate_reconciliation_guardrail_task(
                task_id,
                title,
                fields,
                errors=errors,
            )
            if is_pending:
                pending_ids.append(task_id)
                active_reconciliation_count += 1
                if (
                    active_reconciliation_count
                    > MAX_ACTIVE_OPERATIONAL_RECONCILIATION_TASKS
                ):
                    errors.append(
                        "active reconciliation appendix exceeds its finite bound"
                    )

            dedupe_key = fields.get("dedupe key", "")
            fingerprint = fields.get("reconciliation fingerprint", "")
            previous_dedupe_status = reconciliation_dedupe_status.get(dedupe_key)
            if (
                dedupe_key
                and previous_dedupe_status is not None
                and previous_dedupe_status != "completed"
            ):
                errors.append(
                    "concurrent duplicate operational reconciliation task: "
                    f"{dedupe_key}"
                )
            previous_fingerprint_status = (
                reconciliation_fingerprint_status.get(fingerprint)
            )
            if (
                fingerprint
                and previous_fingerprint_status is not None
                and previous_fingerprint_status != "completed"
            ):
                errors.append(
                    "concurrent duplicate operational reconciliation "
                    f"fingerprint: {fingerprint}"
                )
            if dedupe_key:
                reconciliation_dedupe_status[dedupe_key] = fields.get(
                    "status", ""
                )
            if fingerprint:
                reconciliation_fingerprint_status[fingerprint] = fields.get(
                    "status", ""
                )
            continue

        title_match = RETRY_BUDGET_REPAIR_TITLE_RE.fullmatch(title)
        acceptance = fields.get("acceptance", "")
        acceptance_match = RETRY_BUDGET_REPAIR_ACCEPTANCE_RE.search(acceptance)
        source_task_id = fields.get("retry repair source", "")
        failure_kind = fields.get("retry failure kind", "").lower()
        source = canonical_by_id.get(source_task_id)
        if (
            title_match is None
            or acceptance_match is None
            or title_match.group("source").upper() != source_task_id
            or acceptance_match.group("source").upper() != source_task_id
            or title_match.group("kind").lower() != failure_kind
            or source is None
            or failure_kind not in {"validation", "implementation", "merge"}
        ):
            errors.append(f"{task_id} is not a recognized retry repair")
            continue

        if (
            fields.get("generated by") != RETRY_BUDGET_REPAIR_SCHEMA
            or fields.get("canonical board task") != "false"
        ):
            errors.append(f"{task_id} lacks exact operational provenance")
        if fields.get("completion") != "manual":
            errors.append(f"{task_id} must use manual completion")
        if fields.get("priority") != "P1":
            errors.append(f"{task_id} must use P1 priority")
        if fields.get("track") != "ops":
            errors.append(f"{task_id} must use the ops track")

        status = fields.get("status", "")
        if status not in PERSISTED_OPERATIONAL_STATES:
            errors.append(f"{task_id} has invalid operational status {status!r}")
        if status != "completed":
            pending_ids.append(task_id)
            if source.get("status") == "completed":
                errors.append(
                    f"{task_id} is pending after source {source_task_id} completed"
                )

        dependencies = _csv(fields.get("depends on", ""))
        source_dependencies = _csv(source.get("depends on", ""))
        if dependencies != source_dependencies:
            errors.append(
                f"{task_id} dependency scope differs from {source_task_id}"
            )

        repair_outputs = _csv(fields.get("outputs", ""))
        source_outputs = _csv(source.get("outputs", ""))
        if repair_outputs != source_outputs:
            errors.append(
                f"{task_id} output scope differs from source {source_task_id}"
            )
        source_scope = _csv(source.get("scope paths", ""))
        declared_scope = _csv(fields.get("scope paths", ""))
        effective_scope = declared_scope or source_scope
        if declared_scope and declared_scope != source_scope:
            errors.append(
                f"{task_id} declared scope differs from source {source_task_id}"
            )
        if not source_scope or any(
            not any(_path_is_within_scope(output, scope) for scope in effective_scope)
            for output in repair_outputs
        ):
            errors.append(
                f"{task_id} outputs escape source {source_task_id} scope paths"
            )
        for field in ("parallel lane", "conflict policy"):
            if fields.get(field, "") != source.get(field, ""):
                errors.append(
                    f"{task_id} {field} differs from source {source_task_id}"
                )

        discovery_text = fields.get("retry repair discovery", "")
        discovery_path = Path(discovery_text)
        expected_suffix = {
            "validation": "retry-budget",
            "implementation": "implementation-retry-budget",
            "merge": "merge-retry-budget",
        }[failure_kind]
        expected_discovery_name = re.compile(
            rf"\d{{4}}-\d{{2}}-\d{{2}}-{task_id.lower()}-"
            rf"{source_task_id.lower()}-{expected_suffix}\.md"
        )
        retry_discovery_is_valid = (
            bool(discovery_text)
            and discovery_path.is_absolute()
            and _supervisor_owned_discovery_path(discovery_text)
            and expected_discovery_name.fullmatch(discovery_path.name) is not None
            and discovery_text in acceptance
        )
        if not retry_discovery_is_valid:
            errors.append(f"{task_id} has invalid retry discovery provenance")
        else:
            _read_bounded_regular_file(
                task_id,
                discovery_path,
                errors=errors,
            )
        if discovery_text in repair_outputs:
            errors.append(
                f"{task_id} grants write authority to discovery evidence"
            )

        failure_paths = _csv(fields.get("validation failure paths", ""))
        failure_path_authority = fields.get(
            "validation failure path authority", ""
        )
        if failure_paths and failure_path_authority != "diagnostic-read-only":
            errors.append(
                f"{task_id} validation failure paths are not diagnostic-read-only"
            )
        if failure_path_authority and not failure_paths:
            errors.append(
                f"{task_id} declares validation failure authority without paths"
            )
        for path in failure_paths:
            if not _safe_relative(path):
                errors.append(
                    f"{task_id} has unsafe validation failure path {path!r}"
                )
        if not fields.get("validation", ""):
            errors.append(f"{task_id} has no validation command")
        if not acceptance:
            errors.append(f"{task_id} has empty acceptance")

        source_kind = (source_task_id, failure_kind)
        if status != "completed" and source_kind in active_source_kinds:
            errors.append(
                f"{task_id} duplicates an active repair for {source_kind}"
            )
        if status != "completed":
            active_source_kinds.add(source_kind)

    return tuple(operational_ids), tuple(pending_ids)


def _acyclic(nodes: Iterable[str], dependencies: Mapping[str, Iterable[str]], errors: list[str], label: str) -> None:
    state: dict[str, int] = {}
    trail: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            start = trail.index(node) if node in trail else 0
            errors.append(f"{label} dependency cycle: {' -> '.join([*trail[start:], node])}")
            return
        state[node] = 1
        trail.append(node)
        for dependency in dependencies.get(node, ()):
            if dependency in state or dependency in dependencies:
                visit(dependency)
        trail.pop()
        state[node] = 2

    for node in nodes:
        visit(node)


def _shard(task_id: str, lanes: int = 4) -> int:
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % lanes


def _load_config(errors: list[str]) -> dict[str, object]:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"scheduler config unreadable: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append("scheduler config root is not an object")
        return {}
    return payload


def validate() -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, CONFIG_PATH):
        if not path.is_file():
            errors.append(f"missing control file: {path.relative_to(REPO_ROOT)}")

    try:
        parsed_tasks = parse_tasks() if TODO_PATH.is_file() else []
        goals = parse_goals() if OBJECTIVE_PATH.is_file() else []
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"control markdown parse failed: {type(exc).__name__}: {exc}")
        parsed_tasks, goals = [], []

    tasks, operational_repairs = _partition_canonical_and_operational_tasks(
        parsed_tasks,
        errors,
    )
    task_ids = tuple(item[0] for item in tasks)
    goal_ids = tuple(item[0] for item in goals)
    if task_ids != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {list(TASK_IDS)}, got {list(task_ids)}")
    if goal_ids != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {list(GOAL_IDS)}, got {list(goal_ids)}")
    if len(set(goal_ids)) != len(goal_ids):
        errors.append("goal IDs are not unique")

    task_by_id = {task_id: fields for task_id, _title, fields in tasks}
    task_dependencies: dict[str, tuple[str, ...]] = {}
    output_owners: dict[str, list[str]] = {}
    for task_id, _title, fields in tasks:
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in fields]
        if missing:
            errors.append(f"{task_id} missing fields: {missing}")
        status = fields.get("status", "")
        if status not in {"todo", "in_progress", "completed", "blocked"}:
            errors.append(f"{task_id} has invalid status {status!r}")
        if fields.get("completion") not in {"auto", "manual"}:
            errors.append(f"{task_id} has invalid completion")
        if fields.get("is schedulable") not in {"true", "false"}:
            errors.append(f"{task_id} has invalid is schedulable")
        if fields.get("review only") not in {"true", "false"}:
            errors.append(f"{task_id} has invalid review only")
        if fields.get("board namespace") != NAMESPACE:
            errors.append(f"{task_id} board namespace mismatch")
        dependencies = _csv(fields.get("depends on", ""))
        task_dependencies[task_id] = dependencies
        expected_dependencies = TASK_DEPENDENCIES.get(task_id)
        if expected_dependencies is not None and dependencies != expected_dependencies:
            errors.append(
                f"{task_id} dependency mismatch: expected {list(expected_dependencies)}, got {list(dependencies)}"
            )
        for dependency in dependencies:
            if dependency not in TASK_IDS:
                errors.append(f"{task_id} references unknown dependency {dependency}")
        goal_id = fields.get("goal id", "")
        expected_goal = "KVFS-G000" if task_id == "KVFS-000" else next(
            (candidate for candidate, members in TASK_GROUPS.items() if task_id in members), ""
        )
        if goal_id != expected_goal:
            errors.append(f"{task_id} goal mismatch: expected {expected_goal}, got {goal_id}")
        outputs = _csv(fields.get("outputs", ""))
        if not outputs:
            errors.append(f"{task_id} has no outputs")
        scope_paths = _csv(fields.get("scope paths", ""))
        if not scope_paths:
            errors.append(f"{task_id} has no scope paths")
        for scope_path in scope_paths:
            if not _safe_relative(scope_path):
                errors.append(f"{task_id} has unsafe scope path {scope_path!r}")
        for output in outputs:
            if not _safe_relative(output):
                errors.append(f"{task_id} has unsafe output {output!r}")
            if task_id != "KVFS-000" and output in PROTECTED_PATHS:
                errors.append(f"{task_id} writes protected control path {output}")
            if scope_paths and not any(
                _path_is_within_scope(output, scope_path)
                for scope_path in scope_paths
            ):
                errors.append(f"{task_id} output escapes its scope paths: {output}")
            output_owners.setdefault(output, []).append(task_id)
        if task_id in INITIAL_READY:
            native_text = " ".join((fields.get("outputs", ""), fields.get("validation", ""))).lower()
            if any(term in native_text for term in ("live_mount", "live_winfsp", "live_container")):
                errors.append(f"initial task {task_id} requires a native live harness")

    for output, owners in sorted(output_owners.items()):
        if len(owners) > 1:
            errors.append(f"output has multiple owners: {output}: {owners}")
    if set(task_dependencies) == set(TASK_IDS):
        _acyclic(TASK_IDS, task_dependencies, errors, "task")

    operational_task_ids, pending_operational_task_ids = (
        _validate_operational_repair_tasks(
            operational_repairs,
            canonical_by_id=task_by_id,
            errors=errors,
        )
    )

    completed = {task_id for task_id, fields in task_by_id.items() if fields.get("status") == "completed"}
    blocked = {task_id for task_id, fields in task_by_id.items() if fields.get("status") == "blocked"}
    missing_initial_completion = sorted(
        set(INITIAL_COMPLETED) - completed
    )
    if missing_initial_completion:
        errors.append(
            "sealed initial completions regressed: "
            f"{missing_initial_completion}"
        )
    incomplete_dependencies = {
        task_id: sorted(
            dependency
            for dependency in task_dependencies.get(task_id, ())
            if dependency not in completed
        )
        for task_id in sorted(completed)
        if any(
            dependency not in completed
            for dependency in task_dependencies.get(task_id, ())
        )
    }
    if incomplete_dependencies:
        errors.append(
            "completed canonical tasks are not dependency-closed: "
            + json.dumps(incomplete_dependencies, sort_keys=True)
        )
    ready = tuple(
        task_id for task_id in TASK_IDS
        if task_by_id.get(task_id, {}).get("status") == "todo"
        and all(dependency in completed for dependency in task_dependencies.get(task_id, ()))
    )
    initial_shards = {_shard(task_id): task_id for task_id in INITIAL_READY}
    if initial_shards != INITIAL_SHARDS or len(initial_shards) != len(INITIAL_READY):
        errors.append(f"sealed initial strict shard coverage differs: {initial_shards}")
    current_ready_shards: dict[str, list[str]] = {
        str(index): [] for index in range(4)
    }
    for task_id in ready:
        current_ready_shards[str(_shard(task_id))].append(task_id)

    goal_dependencies: dict[str, tuple[str, ...]] = {}
    for goal_id, _title, fields in goals:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in fields]
        if missing:
            errors.append(f"{goal_id} missing fields: {missing}")
        if fields.get("status") not in {
            "active", "provisionally_complete", "verified_complete",
            "analysis_inconclusive", "blocked", "reopened",
        }:
            errors.append(f"{goal_id} has invalid status {fields.get('status')!r}")
        parent = fields.get("parent", "")
        expected_parent = "" if goal_id == "KVFS-G000" else "KVFS-G000"
        if parent != expected_parent:
            errors.append(f"{goal_id} parent mismatch: expected {expected_parent!r}, got {parent!r}")
        dependencies = _csv(fields.get("depends on", ""))
        goal_dependencies[goal_id] = dependencies
        if dependencies != GOAL_DEPENDENCIES.get(goal_id, ()):
            errors.append(f"{goal_id} dependency mismatch: {list(dependencies)}")
        for reference in (*dependencies, *((parent,) if parent else ())):
            if reference not in GOAL_IDS:
                errors.append(f"{goal_id} references unknown goal {reference}")
        evidence = _csv(fields.get("evidence", ""))
        for reference in evidence:
            if reference not in TASK_IDS and reference not in GOAL_IDS:
                errors.append(f"{goal_id} references unknown evidence {reference}")
    if set(goal_dependencies) == set(GOAL_IDS):
        _acyclic(GOAL_IDS, goal_dependencies, errors, "goal")

    config = _load_config(errors)
    exact_config = {
        "schema": "ipfs_accelerate_py.agent_supervisor.ipfs_kit_fuse_vfs.scheduler_config@1",
        "taskboard_path": "docs/architecture/ipfs_kit_fuse_vfs.todo.md",
        "objectives_path": "docs/architecture/ipfs_kit_fuse_vfs.objectives.md",
        "plan_path": "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md",
        "validator_path": "scripts/validate_ipfs_kit_fuse_vfs_board.py",
        "task_prefix": "KVFS-",
        "goal_prefix": "KVFS-G",
        "board_namespace": NAMESPACE,
        "merge_target_branch": BRANCH,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for field, expected in exact_config.items():
        if config.get(field) != expected:
            errors.append(f"scheduler {field} mismatch: expected {expected!r}, got {config.get(field)!r}")
    projection = config.get("initial_projection", {})
    expected_projection = {
        "task_count": 40,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 9,
        "root_goal_id": "KVFS-G000",
    }
    if projection != expected_projection:
        errors.append("scheduler initial projection mismatch")
    source = config.get("source_binding", {})
    for field, expected in {
        "accelerator_required_ancestor": ACCELERATOR_ANCESTOR,
        "accelerator_required_branch": BRANCH,
        "ipfs_kit_submodule_path": "ipfs_kit_py",
        "ipfs_kit_planning_revision": IPFS_KIT_REVISION,
    }.items():
        if not isinstance(source, dict) or source.get(field) != expected:
            errors.append(f"scheduler source_binding.{field} mismatch")
    if config.get("worktree_submodule_paths") != ["ipfs_kit_py"]:
        errors.append("scheduler worktree_submodule_paths mismatch")
    if tuple(config.get("protected_paths", ())) != PROTECTED_PATHS:
        errors.append("scheduler protected_paths mismatch")
    configured_groups = config.get("task_groups", {})
    if not isinstance(configured_groups, dict) or {
        key: tuple(value) if isinstance(value, list) else ()
        for key, value in configured_groups.items()
    } != TASK_GROUPS:
        errors.append("scheduler task_groups mismatch")
    lanes = config.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) != 4:
        errors.append("scheduler must define exactly four lanes")
    else:
        for index, lane in enumerate(lanes):
            expected_task = INITIAL_SHARDS[index]
            if not isinstance(lane, dict) or lane.get("index") != index or lane.get("strict_shard_remainder") != index or lane.get("initial_task_ids") != [expected_task]:
                errors.append(f"scheduler lane {index} mismatch")
    provider = config.get("provider", {})
    provider_seal = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "medium",
        "max_concurrency": 4,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if provider != provider_seal:
        errors.append("scheduler ordered provider seal mismatch")
    runtime = config.get("runtime_paths", {})
    expected_root = "data/agent_supervisor/ipfs_kit_fuse_vfs"
    if not isinstance(runtime, dict) or runtime.get("root") != expected_root:
        errors.append("scheduler runtime root mismatch")
    elif any(
        not isinstance(value, str) or (key not in {"root", "generated_runtime_artifacts_are_completion_authority"} and not value.startswith(expected_root + "/"))
        for key, value in runtime.items()
        if key != "generated_runtime_artifacts_are_completion_authority"
    ):
        errors.append("scheduler runtime paths escape runtime root")
    capability = config.get("native_capability_policy", {})
    required_capability = {
        "doctor_timeout_seconds": 5,
        "mount_readiness_timeout_seconds": 15,
        "integration_case_timeout_seconds": 60,
        "mount_runs_as_bounded_child_process": True,
        "exclusive_mountpoint_and_drive_leases": True,
        "cleanup_finally_and_watchdog_required": True,
        "capability_absence_receipt": "capability_unavailable",
        "capability_absence_may_leave_task_running": False,
        "linux_windows_and_container_certification_independent": True,
    }
    if capability != required_capability:
        errors.append("scheduler native capability anti-stall policy mismatch")

    if PLAN_PATH.is_file():
        plan = PLAN_PATH.read_text(encoding="utf-8").lower()
        required_terms = (
            "canonicalvfsservice", "wal", "generationboundarc", "fusepy",
            "winfsp", "/dev/fuse", "sys_admin", "fsync", "recovery",
            "sha256(task_id)", "capability_unavailable", "rollback",
        )
        for term in required_terms:
            if term not in plan:
                errors.append(f"plan omits required term {term!r}")

    ignore = subprocess.run(
        ("git", "check-ignore", "-q", "data/agent_supervisor/ipfs_kit_fuse_vfs/probe"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignore.returncode != 0:
        errors.append("configured runtime path is not ignored")

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/ipfs-kit-fuse-vfs-board-validation@1",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "task_count": len(tasks),
        "parsed_task_count": len(parsed_tasks),
        "operational_task_count": len(operational_task_ids),
        "operational_task_ids": list(operational_task_ids),
        "pending_operational_task_ids": list(pending_operational_task_ids),
        "goal_count": len(goals),
        "completed_task_ids": sorted(completed),
        "blocked_task_ids": sorted(blocked),
        "ready_task_ids": list(ready),
        "initial_ready_task_ids": list(INITIAL_READY),
        "initial_shards": {
            str(index): task_id for index, task_id in sorted(initial_shards.items())
        },
        "current_ready_shards": current_ready_shards,
        "terminal_task_id": TERMINAL_TASK,
        "board_namespace": NAMESPACE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="validate all sealed board invariants")
    parser.parse_args(argv)
    report = validate()
    json.dump(report, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
