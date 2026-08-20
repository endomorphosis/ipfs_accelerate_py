"""Ingest JSON or Markdown taskboards into DuckDB with Quack health gating.

The supervisor keeps the canonical board in DuckDB so callers can query a
bounded context view instead of loading the full board into a model window.
Malformed text is repaired by deterministic utilities; duplicate JSON keys,
unknown task identities, and live launch are fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .database_task_source import DatabaseTaskSource
from .quack_capabilities import probe_quack_capabilities

TASKBOARD_INGEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-ingest@1"
)
TASKBOARD_REPAIR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-text-repair@1"
)
TASKBOARD_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-context-view@1"
)
KNOWN_STATUSES: Final = frozenset(
    {
        "todo",
        "ready",
        "blocked",
        "queued",
        "proposed",
        "admitted",
        "in_progress",
        "running",
        "completed",
        "failed",
        "cancelled",
        "quarantined",
        "rejected",
        "skipped",
    }
)
STATUS_ALIASES: Final = {
    "to-do": "todo",
    "to do": "todo",
    "not started": "todo",
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "wip": "in_progress",
    "done": "completed",
    "complete": "completed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}
_SMART_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u200b": "",
        "\ufeff": "",
        "\u200c": "",
        "\u200d": "",
    }
)
_HEADING_RE = re.compile(
    r"^(#{2,6})([^\s#])",
    flags=re.MULTILINE,
)
_BULLET_RE = re.compile(r"^(\s*)-([^\s-])", flags=re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CONTROL_IN_STRING = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MAX_CONTEXT_BYTES: Final = 32_000
DEFAULT_CONTEXT_BYTES: Final = 4_096
MAX_OBJECTIVE_CHARS: Final = 240


class TaskboardIngestError(ValueError):
    """Fail-closed ingest or repair error."""


@dataclass(frozen=True)
class RepairAction:
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class RepairResult:
    text: str
    actions: list[RepairAction] = field(default_factory=list)
    encoding: str = "utf-8"

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASKBOARD_REPAIR_SCHEMA,
            "changed": self.changed,
            "encoding": self.encoding,
            "actions": [item.to_dict() for item in self.actions],
            "byte_count": len(self.text.encode("utf-8")),
        }


def _cid(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def repair_malformed_text(raw: str | bytes, *, kind: str = "auto") -> RepairResult:
    """Deterministically repair malformed taskboard text. No model calls."""

    actions: list[RepairAction] = []
    encoding = "utf-8"
    if isinstance(raw, bytes):
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
            actions.append(RepairAction("strip_bom", "removed UTF-8 BOM"))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            encoding = "utf-8-replace"
            actions.append(RepairAction("decode_replace", "replaced invalid UTF-8"))
    else:
        text = str(raw)
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
            actions.append(RepairAction("strip_bom", "removed UTF-8 BOM"))
    if "\r\n" in text or "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        actions.append(RepairAction("normalize_newlines", "converted to LF"))
    translated = text.translate(_SMART_QUOTES)
    if translated != text:
        text = translated
        actions.append(RepairAction("normalize_unicode", "replaced smart quotes and unicode spaces"))
    stripped_lines = [line.rstrip(" \t") for line in text.split("\n")]
    if stripped_lines != text.split("\n"):
        text = "\n".join(stripped_lines)
        actions.append(RepairAction("rstrip_lines", "removed trailing line whitespace"))
    heading = _HEADING_RE.sub(r"\1 \2", text)
    if heading != text:
        text = heading
        actions.append(RepairAction("heading_space", "inserted space after markdown heading marks"))
    bullets = _BULLET_RE.sub(r"\1- \2", text)
    if bullets != text:
        text = bullets
        actions.append(RepairAction("bullet_space", "inserted space after markdown list dashes"))
    detected = kind
    if kind == "auto":
        stripped = text.lstrip()
        detected = "json" if stripped.startswith("{") or stripped.startswith("[") else "markdown"
    if detected == "json" and _TRAILING_COMMA_RE.search(text):
        text = _TRAILING_COMMA_RE.sub(r"\1", text)
        actions.append(RepairAction("json_trailing_comma", "removed trailing commas"))
    if text and not text.endswith("\n"):
        text += "\n"
        actions.append(RepairAction("trailing_newline", "appended final newline"))
    return RepairResult(text=text, actions=actions, encoding=encoding)


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TaskboardIngestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json_taskboard(text: str) -> dict[str, Any]:
    repaired = repair_malformed_text(text, kind="json")
    try:
        payload = json.loads(
            repaired.text, object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise TaskboardIngestError(f"taskboard JSON is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise TaskboardIngestError("taskboard JSON must be an object")
    payload["_repair"] = repaired.to_dict()
    return payload


def parse_markdown_field_board(text: str) -> dict[str, Any]:
    """Parse generator-style Markdown boards (`## ID title` plus `- Field: value`)."""

    repaired = repair_malformed_text(text, kind="markdown")
    body = repaired.text
    tasks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    title = ""
    for line in body.split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current is not None:
                tasks.append(current)
            rest = line[3:].strip()
            parts = rest.split(" ", 1)
            task_id = parts[0].strip()
            current = {
                "stable_task_id": task_id,
                "task_id": task_id,
                "task_alias": task_id,
                "title": parts[1].strip() if len(parts) > 1 else task_id,
            }
            continue
        if current is None or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        field = key.strip().casefold().replace(" ", "_")
        current[field] = value.strip()
        if field == "status":
            current["status"] = normalize_status(value)
        if field == "depends_on":
            current["dependencies"] = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]
        if field == "stable_task_id" and value.strip():
            current["stable_task_id"] = value.strip()
            current["task_id"] = value.strip()
            current["task_alias"] = value.strip()
    if current is not None:
        tasks.append(current)
    return {
        "title": title,
        "tasks": tasks,
        "goals": [],
        "_repair": repaired.to_dict(),
        "source_kind": "markdown_fields",
    }


def normalize_status(value: object) -> str:
    raw = _one_line(value).casefold()
    aliased = STATUS_ALIASES.get(raw, raw.replace(" ", "_"))
    if aliased not in KNOWN_STATUSES:
        raise TaskboardIngestError(f"unknown task status {value!r}")
    return aliased


def validate_taskboard(board: Mapping[str, Any]) -> list[str]:
    """Return blocking errors. Empty means the board may be ingested."""

    errors: list[str] = []
    tasks = board.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["taskboard has no tasks"]
    seen: set[str] = set()
    ids: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            errors.append(f"task[{index}] is not an object")
            continue
        alias = str(
            task.get("stable_task_id")
            or task.get("task_id")
            or task.get("task_alias")
            or ""
        ).strip()
        if not alias:
            errors.append(f"task[{index}] is missing a task id")
            continue
        if any(character in alias for character in "\0\n\r"):
            errors.append(f"{alias}: task id contains a newline")
        if alias in seen:
            errors.append(f"{alias}: duplicate task id")
        seen.add(alias)
        ids.append(alias)
        status = str(task.get("status") or "todo")
        try:
            normalize_status(status)
        except TaskboardIngestError as exc:
            errors.append(f"{alias}: {exc}")
        deps = task.get("dependencies") or task.get("depends_on") or ()
        if isinstance(deps, str):
            deps = [item.strip() for item in deps.split(",") if item.strip()]
        if not isinstance(deps, Sequence) or isinstance(deps, (str, bytes)):
            errors.append(f"{alias}: dependencies are malformed")
    known = set(ids)
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        alias = str(
            task.get("stable_task_id") or task.get("task_id") or task.get("task_alias") or ""
        )
        deps = task.get("dependencies") or task.get("depends_on") or ()
        if isinstance(deps, str):
            deps = [item.strip() for item in deps.split(",") if item.strip()]
        if not isinstance(deps, Sequence) or isinstance(deps, (str, bytes)):
            continue
        for dep in deps:
            dep_id = str(dep).strip()
            if dep_id and dep_id not in known:
                errors.append(f"{alias}: unknown dependency {dep_id}")
    return errors


def load_taskboard(path: Path | str, *, repair: bool = True) -> dict[str, Any]:
    location = Path(path)
    raw = location.read_bytes()
    text = repair_malformed_text(raw).text if repair else raw.decode("utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or location.suffix.casefold() == ".json":
        board = parse_json_taskboard(text)
    else:
        board = parse_markdown_field_board(text)
    board["source_path"] = str(location)
    errors = validate_taskboard(board)
    if errors:
        raise TaskboardIngestError(
            "taskboard is malformed: " + "; ".join(errors[:12])
        )
    return board


def _task_alias(task: Mapping[str, Any]) -> str:
    return str(
        task.get("stable_task_id")
        or task.get("task_id")
        or task.get("task_alias")
        or ""
    ).strip()


def board_to_population(board: Mapping[str, Any]) -> dict[str, Any]:
    tasks_in = list(board.get("tasks") or ())
    goals_in = list(board.get("goals") or ())
    namespace = str(
        board.get("board_namespace") or board.get("title") or "taskboard"
    )
    plan_alias = str(board.get("plan_revision") or board.get("plan_alias") or "plan")
    plan_root = str(board.get("plan_root_cid") or _cid({"namespace": namespace, "plan": plan_alias}))
    tree_id = str(board.get("repository_tree_id") or _cid({"namespace": namespace, "tree": "local"}))
    goal_cids: dict[str, str] = {}
    goals: list[dict[str, Any]] = []
    if not goals_in:
        root_alias = "G-ROOT"
        goal_cids[root_alias] = _cid({"goal": root_alias, "plan": plan_root})
        goals.append(
            {
                "goal_cid": goal_cids[root_alias],
                "goal_alias": root_alias,
                "goal_id": root_alias,
                "title": namespace,
                "ordinal": 1,
                "status": "open",
            }
        )
    for index, goal in enumerate(goals_in, start=1):
        if not isinstance(goal, Mapping):
            continue
        alias = str(goal.get("goal_id") or goal.get("goal_alias") or f"G-{index}")
        cid = str(goal.get("goal_cid") or _cid({"goal": alias, "plan": plan_root}))
        goal_cids[alias] = cid
        goals.append(
            {
                "goal_cid": cid,
                "goal_alias": alias,
                "goal_id": alias,
                "title": str(goal.get("title") or alias),
                "ordinal": int(goal.get("ordinal") or index),
                "status": str(goal.get("status") or "open"),
            }
        )
    default_goal = next(iter(goal_cids.values()))
    declared: dict[str, str] = {}
    for task in tasks_in:
        if not isinstance(task, Mapping):
            continue
        alias = _task_alias(task)
        declared[alias] = str(task.get("task_cid") or _cid({"task": alias, "plan": plan_root}))
    tasks: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks_in, start=1):
        if not isinstance(task, Mapping):
            continue
        alias = _task_alias(task)
        deps_raw = task.get("dependencies") or task.get("depends_on") or ()
        if isinstance(deps_raw, str):
            deps = [item.strip() for item in deps_raw.split(",") if item.strip()]
        else:
            deps = [str(item) for item in deps_raw]
        goal_ref = str(task.get("subgoal_id") or task.get("goal_id") or task.get("parent_goal_id") or "")
        goal_cid = goal_cids.get(goal_ref, default_goal)
        status = normalize_status(task.get("status") or "todo")
        body = dict(task)
        tasks.append(
            {
                **body,
                "task_cid": declared[alias],
                "task_id": alias,
                "task_alias": alias,
                "goal_cid": goal_cid,
                "plan_cid": plan_root,
                "ordinal": ordinal,
                "status": status,
                "title": str(task.get("title") or alias),
                "dependencies": [declared.get(item, item) for item in deps],
                "outputs": [],
                "acceptance": (
                    [str(task.get("acceptance"))]
                    if str(task.get("acceptance") or "").strip()
                    else []
                ),
                "validations": list(task.get("execution_validation") or ()),
            }
        )
    return {
        "schema": TASKBOARD_INGEST_SCHEMA,
        "repository_tree_id": tree_id,
        "plan_root_cid": plan_root,
        "goals": goals,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": plan_alias,
                "goal_cid": default_goal,
                "status": "active",
            }
        ],
        "tasks": tasks,
        "board_namespace": namespace,
    }


def probe_ingest_quack(*, require_quack: bool = True) -> dict[str, Any]:
    report = probe_quack_capabilities(
        allow_network_install=False,
        allow_local_load=True,
        use_cache=True,
    )
    payload = report.to_dict()
    if require_quack and not report.passes_health_check:
        raise TaskboardIngestError(
            "Quack health check failed; refusing taskboard ingest without "
            "DuckDB 1.5.x + local core Quack"
        )
    payload["require_quack"] = require_quack
    payload["network_install_attempted"] = False
    return payload


def ingest_taskboard(
    *,
    board_path: Path | str,
    store_path: Path | str,
    repair: bool = True,
    require_quack: bool = True,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Validate, optionally repair, and ingest a taskboard into DuckDB."""

    quack = probe_ingest_quack(require_quack=require_quack)
    board = load_taskboard(board_path, repair=repair)
    population = board_to_population(board)
    store = Path(store_path)
    store.parent.mkdir(parents=True, exist_ok=True)
    inserted: list[str] = []
    skipped: list[str] = []
    with DatabaseTaskSource(store, install_schema=True) as source:
        existing: set[str] = set()
        if store.is_file() and not replace_existing:
            for item in source.list_tasks(limit=1000).tasks:
                existing.add(item.task_alias)
        to_load = dict(population)
        if existing and not replace_existing:
            kept = [
                task
                for task in population["tasks"]
                if str(task["task_alias"]) not in existing
            ]
            skipped = [
                str(task["task_alias"])
                for task in population["tasks"]
                if str(task["task_alias"]) in existing
            ]
            to_load["tasks"] = kept
        if to_load.get("tasks"):
            receipt = source.materialize(
                to_load,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            )
        else:
            receipt = {"task_count": 0}
        inserted = [
            str(task["task_alias"]) for task in to_load.get("tasks") or ()
        ]
        after = source.list_tasks(limit=1000)
        statuses: dict[str, int] = {}
        for item in after.tasks:
            statuses[item.status] = statuses.get(item.status, 0) + 1
    return {
        "schema": TASKBOARD_INGEST_SCHEMA,
        "process_started": False,
        "configured_board_launch": False,
        "store": str(store),
        "source_path": str(board_path),
        "board_namespace": population["board_namespace"],
        "plan_root_cid": population["plan_root_cid"],
        "inserted_count": len(inserted),
        "skipped_existing_count": len(skipped),
        "task_count": sum(statuses.values()),
        "status_counts": statuses,
        "repair": board.get("_repair") or {},
        "quack": {
            "passes_health_check": quack.get("passes_health_check"),
            "require_quack": require_quack,
            "network_install_attempted": False,
        },
        "materialize_task_count": receipt.get("task_count"),
        "ingest_cid": _cid(
            {
                "store": str(store),
                "source": str(board_path),
                "inserted": inserted,
            }
        ),
    }


def compact_task_view(task: Mapping[str, Any], *, max_objective: int = MAX_OBJECTIVE_CHARS) -> dict[str, Any]:
    alias = str(task.get("task_alias") or task.get("stable_task_id") or "")
    body = task.get("body") if isinstance(task.get("body"), Mapping) else task
    owned = body.get("owned_files") or body.get("execution_owned_files") or ()
    deps = task.get("dependencies") or body.get("dependencies") or ()
    objective = _one_line(body.get("objective") or body.get("title") or "")
    if len(objective) > max_objective:
        objective = objective[: max_objective - 1] + "…"
    validations = body.get("execution_validation") or body.get("validations") or ()
    argv: list[str] = []
    if isinstance(validations, Sequence) and validations:
        first = validations[0]
        if isinstance(first, Mapping):
            argv = [str(part) for part in first.get("argv") or ()]
        elif isinstance(first, Sequence) and not isinstance(first, str):
            argv = [str(part) for part in first]
    return {
        "task_id": alias,
        "status": str(task.get("status") or body.get("status") or ""),
        "title": str(body.get("title") or alias),
        "objective": objective,
        "dependency_count": len(list(deps)) if isinstance(deps, Sequence) else 0,
        "owned_file_count": len(list(owned)) if isinstance(owned, Sequence) else 0,
        "validation_argv": argv[:12],
        "epic": str(body.get("epic") or ""),
        "completion_mode": str(body.get("completion_mode") or body.get("completion") or ""),
    }


def taskboard_context_view(
    store_path: Path | str,
    *,
    ready_only: bool = True,
    task_id: str = "",
    max_bytes: int = DEFAULT_CONTEXT_BYTES,
) -> dict[str, Any]:
    """Return a bounded DuckDB projection for model context."""

    if max_bytes < 256 or max_bytes > MAX_CONTEXT_BYTES:
        raise TaskboardIngestError("max_bytes is outside the context budget")
    selected: list[dict[str, Any]] = []
    truncated = False
    with DatabaseTaskSource(Path(store_path), install_schema=False) as source:
        if task_id:
            item = source.get_task(task_id)
            records = [item] if item is not None else []
        elif ready_only:
            records = list(source.ready_tasks(limit=64).tasks)
        else:
            records = list(source.list_tasks(limit=64).tasks)
        status_counts: dict[str, int] = {}
        all_tasks = source.list_tasks(limit=1000).tasks
        for item in all_tasks:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        payload = {
            "schema": TASKBOARD_CONTEXT_SCHEMA,
            "store": str(store_path),
            "ready_only": ready_only,
            "task_count": len(all_tasks),
            "status_counts": status_counts,
            "selected": selected,
            "truncated": False,
        }
        used = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        for item in records:
            raw = item.to_dict() if hasattr(item, "to_dict") else {
                "task_alias": getattr(item, "task_alias", ""),
                "status": getattr(item, "status", ""),
                "body": getattr(item, "body", {}) if hasattr(item, "body") else {},
            }
            view = compact_task_view(raw)
            encoded = json.dumps(view, separators=(",", ":")).encode("utf-8")
            if used + len(encoded) + 2 > max_bytes:
                truncated = True
                break
            selected.append(view)
            used += len(encoded) + 2
        payload["selected"] = selected
        payload["truncated"] = truncated
        payload["byte_count"] = used
        payload["selected_count"] = len(selected)
        return payload


def write_repaired_text(path: Path | str, raw: str | bytes, *, kind: str = "auto") -> RepairResult:
    result = repair_malformed_text(raw, kind=kind)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.text, encoding="utf-8")
    return result
