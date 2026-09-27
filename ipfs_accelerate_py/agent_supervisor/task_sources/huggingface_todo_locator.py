"""Locate Hugging Face autoformal todos and turn them into supervisor work.

The datasets package uploads parquet plus a markdown board. This module is how
ipfs_accelerate_py finds that release, indexes the board, and uses the
time-management fields for scheduling. JSONL is not an input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .persistent_task_queue import PersistentTaskQueue
from .task_identity import canonical_task_identity
from .todo_vector_index import (
    TodoIndexRecord,
    parse_todo_vector_records,
    write_todo_vector_index,
)


LOCATOR_SCHEMA = "ipfs_accelerate_py.agent_supervisor.huggingface_todo_locator/v1"
POINTER_SCHEMA = "ipfs_datasets_py/autoformal-todo-release-pointer/v1"
DEFAULT_TASK_HEADER_PREFIX = "## AFTD-"
DEFAULT_BOARD_NAMESPACE = "uscode-autoformal-todo-v1"

Fetcher = Callable[[str, str, str], bytes]


class HuggingFaceTodoLocatorError(ValueError):
    """Locator payload is missing, mismatched, or not a Hugging Face todo release."""


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HuggingFaceTodoLocatorError(f"{label} is required")
    return text


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_json_source(source: Mapping[str, Any] | str | Path) -> Any:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_dir():
        for name in ("pointer.json", "locator.json"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceTodoLocatorError(f"cannot read Hugging Face todo locator: {path}") from exc


def resolve_huggingface_todo_source(
    source: Mapping[str, Any] | str | Path,
    *,
    package_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Return (locator, local package root). Pointers resolve to both."""

    payload = _read_json_source(source)
    if not isinstance(payload, Mapping):
        raise HuggingFaceTodoLocatorError("locator must be a JSON object")
    schema = str(payload.get("schema") or "")
    if schema == POINTER_SCHEMA:
        root = package_root or payload.get("package_root")
        locator_path = payload.get("locator_path") or (str(Path(root) / "locator.json") if root else "")
        if not locator_path:
            raise HuggingFaceTodoLocatorError("release pointer is missing locator_path/package_root")
        return load_huggingface_todo_locator(locator_path), Path(root) if root else None
    return load_huggingface_todo_locator(payload), Path(package_root) if package_root else None


def load_huggingface_todo_locator(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Accept a mapping, locator.json path, package directory, or release pointer."""

    payload = _read_json_source(source)
    if not isinstance(payload, Mapping):
        raise HuggingFaceTodoLocatorError("locator must be a JSON object")
    schema = str(payload.get("schema") or "")
    if schema == POINTER_SCHEMA:
        locator, _root = resolve_huggingface_todo_source(payload)
        return locator
    if schema != LOCATOR_SCHEMA:
        raise HuggingFaceTodoLocatorError(f"unsupported locator schema: {schema or 'missing'}")
    time_management = payload.get("time_management") or {}
    if time_management and not isinstance(time_management, Mapping):
        raise HuggingFaceTodoLocatorError("time_management must be an object")
    return {
        "board_namespace": str(payload.get("board_namespace") or DEFAULT_BOARD_NAMESPACE),
        "board_path": _text(payload.get("board_path"), "board_path"),
        "dataset_repo_id": _text(payload.get("dataset_repo_id"), "dataset_repo_id"),
        "goal_prefix": str(payload.get("goal_prefix") or "AFTD-G"),
        "release_id": _text(payload.get("release_id"), "release_id"),
        "revision": str(payload.get("revision") or "main"),
        "schema": LOCATOR_SCHEMA,
        "task_header_prefix": str(payload.get("task_header_prefix") or DEFAULT_TASK_HEADER_PREFIX),
        "task_prefix": str(payload.get("task_prefix") or "AFTD-"),
        "time_management": dict(time_management),
        "todos_path": _text(payload.get("todos_path"), "todos_path"),
    }


def materialize_huggingface_todo_board(
    locator: Mapping[str, Any],
    destination: str | Path,
    *,
    package_root: str | Path | None = None,
    fetch: Fetcher | None = None,
) -> Path:
    """Copy the supervisor board locally. Does not write JSONL."""

    resolved = load_huggingface_todo_locator(locator)
    board_path = Path(destination)
    board_path.parent.mkdir(parents=True, exist_ok=True)
    if package_root is not None:
        local = Path(package_root) / "board.md"
        if not local.is_file():
            raise HuggingFaceTodoLocatorError(f"local Hugging Face todo package has no board.md: {local}")
        board_path.write_bytes(local.read_bytes())
        return board_path
    if fetch is None:
        raise HuggingFaceTodoLocatorError("fetch is required when no local package_root is supplied")
    payload = fetch(resolved["dataset_repo_id"], resolved["revision"], resolved["board_path"])
    if not isinstance(payload, (bytes, bytearray)):
        payload = str(payload).encode("utf-8")
    board_path.write_bytes(bytes(payload))
    return board_path


_REQUIRED_BOARD_FIELDS = (
    "Status",
    "Preserve",
    "Replace",
    "Estimated tokens",
    "Estimated validation seconds",
    "Match is not admit",
    "Formalized",
    "Admitted",
    "Failure mode",
)


def assert_autoformal_todo_board(markdown: str) -> None:
    """Refuse JSONL, compiler patches, or tasks without preserve/replace budgets."""

    errors: list[str] = []
    lowered = markdown.casefold()
    if ".jsonl" in lowered:
        errors.append("board must not reference jsonl")
    if "proposed_compiler" in lowered:
        errors.append("board must not propose a compiler patch")
    headers = list(re.finditer(r"^## (AFTD-\S+)\s+", markdown, re.MULTILINE))
    if not headers:
        errors.append("board has no AFTD tasks")
    spans = [match.start() for match in headers] + [len(markdown)]
    for index, match in enumerate(headers):
        block = markdown[spans[index] : spans[index + 1]]
        task_id = match.group(1)
        for field in _REQUIRED_BOARD_FIELDS:
            if f"- {field}:" not in block:
                errors.append(f"{task_id} is missing {field}")
        if "- Formalized: true" in block or "- Admitted: true" in block:
            errors.append(f"{task_id} must not mark formalized or admitted")
        if "- Match is not admit: true" not in block:
            errors.append(f"{task_id} must declare match is not admit")
    if errors:
        raise HuggingFaceTodoLocatorError("; ".join(errors))


def index_huggingface_todo_board(
    board_path: str | Path,
    *,
    repo_root: str | Path,
    locator: Mapping[str, Any],
) -> list[TodoIndexRecord]:
    """Parse the materialized board into supervisor todo-vector records."""

    resolved = load_huggingface_todo_locator(locator)
    return parse_todo_vector_records(
        repo_root=Path(repo_root),
        todo_path=Path(board_path),
        task_header_prefix=resolved["task_header_prefix"],
    )


def time_management_from_records(records: Sequence[TodoIndexRecord]) -> dict[str, Any]:
    """Budget the accelerate scheduler can consume. Locator totals are not authority."""

    items = list(records)
    priority_counts: dict[str, int] = {}
    for record in items:
        key = str(record.priority or "P2")
        priority_counts[key] = priority_counts.get(key, 0) + 1
    return {
        "jsonl_written": False,
        "priority_counts": priority_counts,
        "ready_task_ids": [record.task_id for record in items if record.status in {"todo", "needed", "pending"}],
        "resource_classes": sorted({record.resource_class for record in items if record.resource_class}),
        "task_count": len(items),
        "total_estimated_tokens": sum(int(record.estimated_tokens or 0) for record in items),
        "total_estimated_validation_seconds": sum(
            int(record.estimated_validation_seconds or 0) for record in items
        ),
    }


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def select_within_budget(
    records: Sequence[TodoIndexRecord],
    *,
    max_tokens: int | None = None,
    max_validation_seconds: int | None = None,
) -> list[TodoIndexRecord]:
    """Pick ready todos that fit the remaining token and wall-time budget."""

    ordered = sorted(
        records,
        key=lambda record: (
            _PRIORITY_RANK.get(str(record.priority or "P2").upper(), 9),
            record.task_id,
        ),
    )
    selected: list[TodoIndexRecord] = []
    used_tokens = 0
    used_seconds = 0
    for record in ordered:
        if record.status not in {"todo", "needed", "pending"}:
            continue
        tokens = int(record.estimated_tokens or 0)
        seconds = int(record.estimated_validation_seconds or 0)
        if max_tokens is not None and used_tokens + tokens > max_tokens:
            continue
        if max_validation_seconds is not None and used_seconds + seconds > max_validation_seconds:
            continue
        selected.append(record)
        used_tokens += tokens
        used_seconds += seconds
    return selected


def enqueue_huggingface_todos(
    queue: PersistentTaskQueue,
    records: Sequence[TodoIndexRecord],
    *,
    board_path: str | Path,
    board_namespace: str,
) -> list[str]:
    """Register indexed Hugging Face todos on the persistent supervisor queue."""

    registered: list[str] = []
    source_path = str(Path(board_path))
    for record in records:
        identity = canonical_task_identity(
            {
                "task_id": record.task_id,
                "title": record.title,
                "outputs": list(record.outputs),
                "acceptance": record.acceptance,
                "metadata": {
                    "board namespace": board_namespace,
                    "canonical task cid": record.canonical_task_cid,
                    "canonical task key": record.canonical_task_key,
                },
            },
            board_namespace=board_namespace,
            source_path=source_path,
        )
        queue.register_task(identity, priority=record.priority or "P2", track=record.track or "")
        registered.append(record.task_id)
    return registered


def locate_and_schedule_huggingface_todos(
    source: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    board_destination: str | Path,
    package_root: str | Path | None = None,
    fetch: Fetcher | None = None,
    queue: PersistentTaskQueue | None = None,
    index_path: str | Path | None = None,
    max_tokens: int | None = None,
    max_validation_seconds: int | None = None,
) -> dict[str, Any]:
    """Load a Hugging Face todo locator, index the board, and optionally enqueue."""

    locator, resolved_root = resolve_huggingface_todo_source(source, package_root=package_root)
    package_root = resolved_root if package_root is None else package_root
    board_path = materialize_huggingface_todo_board(
        locator,
        board_destination,
        package_root=package_root,
        fetch=fetch,
    )
    assert_autoformal_todo_board(board_path.read_text(encoding="utf-8"))
    records = index_huggingface_todo_board(
        board_path, repo_root=repo_root, locator=locator
    )
    scheduled = select_within_budget(
        records, max_tokens=max_tokens, max_validation_seconds=max_validation_seconds
    )
    enqueued: list[str] = []
    if queue is not None:
        enqueued = enqueue_huggingface_todos(
            queue,
            scheduled,
            board_path=board_path,
            board_namespace=locator["board_namespace"],
        )
    index_receipt: dict[str, Any] | None = None
    if index_path is not None:
        index_receipt = write_todo_vector_index(
            repo_root=Path(repo_root),
            todo_path=board_path,
            index_path=Path(index_path),
            task_header_prefix=locator["task_header_prefix"],
        )
    budget = time_management_from_records(records)
    return {
        "board_namespace": locator["board_namespace"],
        "board_path": str(board_path),
        "dataset_repo_id": locator["dataset_repo_id"],
        "enqueued_task_ids": enqueued,
        "index_path": str(index_path) if index_path is not None else "",
        "index_receipt": index_receipt,
        "jsonl_written": False,
        "locator": locator,
        "release_id": locator["release_id"],
        "scheduled_task_ids": [record.task_id for record in scheduled],
        "task_count": len(records),
        "time_management": budget,
        "wrote_compiler": False,
    }
