"""Content-addressed repository-forest seal for configured supervisor boards.

The tracked seal deliberately excludes the containing superproject commit and
tree: including either would create an impossible self-referential Git digest.
It instead binds every configured nested root, every protected control except
the seal itself, and the exact validator report observed before launch.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .ordered_provider_authoring import (
    OrderedProviderAuthoringError,
    build_authoring_board_projection,
)

BOOTSTRAP_SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-bootstrap-seal@2"
)
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class BootstrapSealError(ValueError):
    """A configured-board bootstrap seal is absent, stale, or malformed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapSealError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by seal identities."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BootstrapSealError("seal value is not canonical JSON") from exc


def content_id(value: Any) -> str:
    """Return a namespaced SHA-256 identity for canonical JSON bytes."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise BootstrapSealError(f"{field} must be a relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or path.as_posix() in {".", ".."}
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise BootstrapSealError(f"{field} is unsafe")
    return path.as_posix()


def _contained(repo_root: Path, relative: str) -> Path:
    target = repo_root / relative
    try:
        target.resolve(strict=False).relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BootstrapSealError(f"path escapes repository: {relative}") from exc
    return target


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BootstrapSealError(
            f"cannot observe Git state in {cwd}: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise BootstrapSealError(
            f"Git observation failed in {cwd}: {result.stderr.strip()}"
        )
    return result.stdout


def _gitlink(repo_root: Path, relative: str) -> str:
    output = _git(repo_root, "ls-tree", "HEAD", "--", relative)
    match = re.fullmatch(
        rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}\n?",
        output,
    )
    if match is None:
        raise BootstrapSealError(f"configured root is not a Gitlink: {relative}")
    return match.group(1)


def _recursive_gitlinks(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in _git(root, "ls-tree", "-r", "HEAD").splitlines():
        match = re.fullmatch(r"160000 commit ([0-9a-f]{40})\t(.+)", line)
        if match is None:
            continue
        rows.append(
            {
                "path": _safe_relative(match.group(2), field="recursive Gitlink path"),
                "commit": match.group(1),
            }
        )
    return sorted(rows, key=lambda item: item["path"])


def _observe_root(
    repo_root: Path,
    *,
    relative: str,
    planning_revision: str,
) -> dict[str, Any]:
    target = _contained(repo_root, relative)
    if not target.is_dir():
        raise BootstrapSealError(f"configured root is not initialized: {relative}")
    top = Path(_git(target, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != target.resolve():
        raise BootstrapSealError(
            f"configured root is not an exact worktree: {relative}"
        )
    head = _git(target, "rev-parse", "HEAD").strip()
    tree = _git(target, "rev-parse", "HEAD^{tree}").strip()
    dirty = _git(target, "status", "--porcelain=v1", "--untracked-files=all")
    gitlink = _gitlink(repo_root, relative)
    if dirty.strip():
        raise BootstrapSealError(f"configured root is dirty: {relative}")
    if not re.fullmatch(r"[0-9a-f]{40}", planning_revision):
        raise BootstrapSealError(f"planning revision is invalid: {relative}")
    if not (planning_revision == gitlink == head):
        raise BootstrapSealError(
            f"planning revision, Gitlink, and nested HEAD differ: {relative}"
        )
    body: dict[str, Any] = {
        "path": relative,
        "planning_revision": planning_revision,
        "gitlink_commit": gitlink,
        "head_commit": head,
        "tree": tree,
        "recursive_gitlinks": _recursive_gitlinks(target),
    }
    return {**body, "root_id": content_id(body)}


def _planning_revisions(source_binding: Mapping[str, Any]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for key, value in source_binding.items():
        if not key.endswith("_submodule_path") or not isinstance(value, str):
            continue
        prefix = key[: -len("_submodule_path")]
        revision = source_binding.get(f"{prefix}_planning_revision")
        relative = _safe_relative(value, field=key)
        if not isinstance(revision, str):
            raise BootstrapSealError(f"missing planning revision for {relative}")
        if relative in revisions:
            raise BootstrapSealError(f"duplicate configured root: {relative}")
        revisions[relative] = revision.strip()
    return revisions


def _control_records(
    repo_root: Path,
    *,
    protected_paths: Sequence[str],
    seal_path: str,
) -> list[dict[str, str]]:
    normalized = [
        _safe_relative(item, field="protected_paths") for item in protected_paths
    ]
    if len(normalized) != len(set(normalized)):
        raise BootstrapSealError("protected paths contain duplicates")
    records: list[dict[str, str]] = []
    for relative in sorted(item for item in normalized if item != seal_path):
        target = _contained(repo_root, relative)
        if target.is_symlink() or not target.is_file():
            raise BootstrapSealError(
                f"protected control is missing or unsafe: {relative}"
            )
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return records


def build_bootstrap_seal_payload(
    *,
    repo_root: Path | str,
    board_namespace: str,
    source_binding: Mapping[str, Any],
    worktree_submodule_paths: Sequence[str],
    protected_paths: Sequence[str],
    seal_path: str,
    taskboard_path: str,
    task_header_prefix: str,
    validator_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build, but do not write, the exact seal for current reviewed inputs."""

    root = Path(repo_root).resolve()
    normalized_seal = _safe_relative(seal_path, field="bootstrap_seal_path")
    normalized_roots = [
        _safe_relative(item, field="worktree_submodule_paths")
        for item in worktree_submodule_paths
    ]
    if len(normalized_roots) != len(set(normalized_roots)):
        raise BootstrapSealError("configured roots contain duplicates")
    revisions = _planning_revisions(source_binding)
    if set(revisions) != set(normalized_roots):
        raise BootstrapSealError("source binding and configured roots differ")
    if validator_report.get("valid") is not True:
        raise BootstrapSealError("declared validator report is not valid")

    roots = [
        _observe_root(
            root,
            relative=relative,
            planning_revision=revisions[relative],
        )
        for relative in normalized_roots
    ]
    forest_body: dict[str, Any] = {"roots": roots}
    forest = {**forest_body, "forest_id": content_id(forest_body)}

    controls = _control_records(
        root,
        protected_paths=protected_paths,
        seal_path=normalized_seal,
    )
    inventory_body: dict[str, Any] = {
        "forest_id": forest["forest_id"],
        "controls": controls,
    }
    inventory = {**inventory_body, "inventory_id": content_id(inventory_body)}

    validator_report_id = content_id(validator_report)
    baseline_body: dict[str, Any] = {
        "forest_id": forest["forest_id"],
        "inventory_id": inventory["inventory_id"],
        "validator_report_id": validator_report_id,
        "valid": True,
    }
    baseline = {**baseline_body, "baseline_id": content_id(baseline_body)}
    try:
        authoring_board = build_authoring_board_projection(
            taskboard_path=_contained(
                root,
                _safe_relative(taskboard_path, field="taskboard_path"),
            ),
            task_header_prefix=task_header_prefix,
            board_namespace=board_namespace,
        )
    except OrderedProviderAuthoringError as exc:
        raise BootstrapSealError(
            f"authoring taskboard is not sealable: {exc.reason_code}"
        ) from exc
    seal_body: dict[str, Any] = {
        "schema": BOOTSTRAP_SEAL_SCHEMA,
        "board_namespace": board_namespace,
        "forest": forest,
        "inventory": inventory,
        "baseline": baseline,
        "authoring_board": authoring_board,
    }
    return {**seal_body, "seal_id": content_id(seal_body)}


def read_bootstrap_seal(path: Path | str) -> dict[str, Any]:
    """Read one duplicate-key-free JSON seal."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except BootstrapSealError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapSealError("bootstrap seal is unreadable") from exc
    if not isinstance(value, dict):
        raise BootstrapSealError("bootstrap seal must be a JSON object")
    return value


def verify_bootstrap_seal(
    *,
    repo_root: Path | str,
    board_namespace: str,
    source_binding: Mapping[str, Any],
    worktree_submodule_paths: Sequence[str],
    protected_paths: Sequence[str],
    seal_path: str,
    taskboard_path: str,
    task_header_prefix: str,
    validator_report: Mapping[str, Any],
) -> dict[str, str]:
    """Reobserve all bound bytes and reject anything but the exact seal."""

    root = Path(repo_root).resolve()
    normalized_seal = _safe_relative(seal_path, field="bootstrap_seal_path")
    expected = build_bootstrap_seal_payload(
        repo_root=root,
        board_namespace=board_namespace,
        source_binding=source_binding,
        worktree_submodule_paths=worktree_submodule_paths,
        protected_paths=protected_paths,
        seal_path=normalized_seal,
        taskboard_path=taskboard_path,
        task_header_prefix=task_header_prefix,
        validator_report=validator_report,
    )
    actual = read_bootstrap_seal(_contained(root, normalized_seal))
    if actual != expected:
        raise BootstrapSealError(
            "bootstrap seal does not match current forest and controls"
        )
    for field, value in (
        ("seal_id", actual.get("seal_id")),
        ("forest_id", actual.get("forest", {}).get("forest_id")),
        ("inventory_id", actual.get("inventory", {}).get("inventory_id")),
        ("baseline_id", actual.get("baseline", {}).get("baseline_id")),
        (
            "authoring_board_id",
            actual.get("authoring_board", {}).get("authoring_board_id"),
        ),
    ):
        if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
            raise BootstrapSealError(f"{field} is not a content identity")
    return {
        "path": normalized_seal,
        "seal_id": actual["seal_id"],
        "forest_id": actual["forest"]["forest_id"],
        "inventory_id": actual["inventory"]["inventory_id"],
        "baseline_id": actual["baseline"]["baseline_id"],
        "authoring_board_id": actual["authoring_board"]["authoring_board_id"],
    }


__all__ = [
    "BOOTSTRAP_SEAL_SCHEMA",
    "BootstrapSealError",
    "build_bootstrap_seal_payload",
    "canonical_json_bytes",
    "content_id",
    "read_bootstrap_seal",
    "verify_bootstrap_seal",
]
