"""Canonical Markdown projection for admitted prompt-generated task graphs.

The human-readable fields retain the task grammar consumed by the existing
implementation daemon.  A bounded base64url metadata marker alongside each
task carries the lossless canonical records needed to prove plan identity,
task population, aliases, dependencies, and projection integrity.  Mutable
status remains outside those semantic records and therefore cannot change a
task CID.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..prompt.prompt_workflow import (
    PromptGoalGraph,
    PromptGoalRecord,
    PromptTaskRecord,
    prompt_workflow_cid,
)
from .taskboard_store import (
    MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
    TaskboardMaterializationEntry,
    TaskboardMaterializationPreview,
    TaskboardMaterializationTransactionResult,
    TaskboardMaterializationTransactionState,
    TaskboardSnapshot,
    TaskboardStore,
    TaskboardTaskRecord,
    _has_taskboard_materialization_transaction,
    commit_taskboard_materialization,
    preview_taskboard_materialization,
    taskboard_revision,
)


MARKDOWN_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/markdown-task-source@1"
)
MARKDOWN_TASK_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/markdown-task-record@1"
)
MARKDOWN_TASK_SOURCE_VERSION: Final = 1
DEFAULT_MARKDOWN_TASK_PREFIX: Final = "TASK"
_MARKER_PREFIX: Final = "agent-supervisor-task-source:v1:"
_MARKER_RE = re.compile(
    rf"^[ \t]*<!--[ \t]+{re.escape(_MARKER_PREFIX)}"
    r"(?P<payload>[A-Za-z0-9_-]+)[ \t]+-->[ \t]*$",
    flags=re.MULTILINE,
)
_HEADING_RE = re.compile(
    r"^##[ \t]+(?P<task_id>\S+)(?:[ \t]+(?P<title>[^\n]*))?[ \t]*$",
    flags=re.MULTILINE,
)
_SAFE_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_VOLATILE_FIELDS = frozenset(
    {
        "status",
        "created_at_ms",
        "updated_at_ms",
        "started_at_ms",
        "finished_at_ms",
        "observed_at_ms",
    }
)


class MarkdownTaskSourceError(ValueError):
    """Base class for fail-closed Markdown projection errors."""


class MarkdownTaskSourceIntegrityError(MarkdownTaskSourceError):
    """Markdown bytes do not prove one complete canonical projection."""


class MarkdownTaskSourceConflict(MarkdownTaskSourceError):
    """A compare-and-swap, population, alias, or plan-root fence failed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise MarkdownTaskSourceError(
            "Markdown projection metadata must be canonical JSON"
        ) from exc


def _semantic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic(member)
            for key, member in sorted(value.items())
            if str(key) not in _VOLATILE_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_semantic(member) for member in value]
    return value


def _metadata_marker(value: Mapping[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(_canonical_json_bytes(value)).decode(
        "ascii"
    ).rstrip("=")
    return f"<!-- {_MARKER_PREFIX}{encoded} -->"


def _decode_marker(value: str) -> dict[str, Any]:
    try:
        padding = "=" * ((4 - len(value) % 4) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        decoded = json.loads(raw)
    except (
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task metadata marker is malformed"
        ) from exc
    if not isinstance(decoded, dict):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task metadata marker must contain an object"
        )
    if _canonical_json_bytes(decoded) != raw:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task metadata marker is not byte-canonical"
        )
    return decoded


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _csv(values: Iterable[Any]) -> str:
    selected = [_one_line(item) for item in values if _one_line(item)]
    return ", ".join(selected)


def _record_cid(record: Mapping[str, Any]) -> str:
    semantic_record = {
        key: value for key, value in record.items() if key != "content_id"
    }
    return prompt_workflow_cid(semantic_record)


def _validate_alias(value: str, *, noun: str) -> str:
    selected = str(value or "").strip()
    if not _SAFE_ALIAS_RE.fullmatch(selected):
        raise MarkdownTaskSourceError(
            f"{noun} must match {_SAFE_ALIAS_RE.pattern}"
        )
    return selected


def _validate_goal_alias(value: str) -> str:
    selected = str(value or "").strip()
    if (
        not selected
        or "\n" in selected
        or "\r" in selected
        or "\x00" in selected
        or len(selected.encode("utf-8")) > 512
    ):
        raise MarkdownTaskSourceError(
            "goal alias must be bounded single-line text"
        )
    return selected


def _topological_tasks(graph: PromptGoalGraph) -> tuple[PromptTaskRecord, ...]:
    by_cid = {item.task_cid: item for item in graph.tasks}
    dependencies = {
        item.task_cid: set(item.dependency_task_cids) for item in graph.tasks
    }
    dependents: dict[str, set[str]] = {
        task_cid: set() for task_cid in dependencies
    }
    for task_cid, required in dependencies.items():
        for dependency in required:
            if dependency not in dependents:
                raise MarkdownTaskSourceError(
                    "task dependency graph references an unknown task"
                )
            dependents[dependency].add(task_cid)
    ready = sorted(
        task_cid
        for task_cid, required in dependencies.items()
        if not required
    )
    ordered: list[PromptTaskRecord] = []
    ordered_cids: set[str] = set()
    while ready:
        task_cid = ready.pop(0)
        ordered.append(by_cid[task_cid])
        ordered_cids.add(task_cid)
        for dependent in sorted(dependents[task_cid]):
            dependencies[dependent].discard(task_cid)
            if (
                not dependencies[dependent]
                and dependent not in ready
                and dependent not in ordered_cids
            ):
                ready.append(dependent)
                ready.sort()
    if len(ordered) != len(by_cid):
        raise MarkdownTaskSourceError("task dependency graph contains a cycle")
    return tuple(ordered)


def _projection_identity(
    *,
    plan_root: str,
    revision: int,
    task_aliases: Mapping[str, str],
    goal_aliases: Mapping[str, str],
) -> str:
    payload = {
        "schema": MARKDOWN_TASK_SOURCE_SCHEMA,
        "version": MARKDOWN_TASK_SOURCE_VERSION,
        "revision": revision,
        "plan_root": plan_root,
        "task_aliases": dict(sorted(task_aliases.items())),
        "goal_aliases": dict(sorted(goal_aliases.items())),
    }
    return (
        "markdown-task-source:sha256:"
        + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    )


@dataclass(frozen=True)
class MarkdownTaskProjection:
    """A complete byte-stable task population ready for journaled append."""

    plan_root: str
    projection_id: str
    entries: tuple[TaskboardMaterializationEntry, ...]
    task_cids: tuple[str, ...]
    task_aliases: Mapping[str, str]
    goal_cids: tuple[str, ...]
    goal_aliases: Mapping[str, str]
    board_namespace: str
    revision: int = 1
    schema: str = MARKDOWN_TASK_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MARKDOWN_TASK_SOURCE_SCHEMA:
            raise MarkdownTaskSourceError(
                "unsupported Markdown task-source schema"
            )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise MarkdownTaskSourceError(
                "projection revision must be a positive integer"
            )
        entries = tuple(self.entries)
        if not entries or len(entries) > MAX_TASKBOARD_MATERIALIZATION_ENTRIES:
            raise MarkdownTaskSourceError(
                "Markdown projection task population is outside its bound"
            )
        task_cids = tuple(self.task_cids)
        goal_cids = tuple(self.goal_cids)
        task_aliases = dict(self.task_aliases)
        goal_aliases = dict(self.goal_aliases)
        if (
            len(task_cids) != len(set(task_cids))
            or set(task_aliases) != set(task_cids)
            or len(set(task_aliases.values())) != len(task_aliases)
        ):
            raise MarkdownTaskSourceError(
                "Markdown projection contains duplicate task aliases or CIDs"
            )
        if any(
            alias != task_cid and alias in set(task_cids)
            for task_cid, alias in task_aliases.items()
        ):
            raise MarkdownTaskSourceError(
                "Markdown projection task alias collides with another task CID"
            )
        if (
            len(goal_cids) != len(set(goal_cids))
            or set(goal_aliases) != set(goal_cids)
            or len(set(goal_aliases.values())) != len(goal_aliases)
        ):
            raise MarkdownTaskSourceError(
                "Markdown projection contains duplicate goal aliases or CIDs"
            )
        if any(
            alias != goal_cid and alias in set(goal_cids)
            for goal_cid, alias in goal_aliases.items()
        ):
            raise MarkdownTaskSourceError(
                "Markdown projection goal alias collides with another goal CID"
            )
        if tuple(item.task_id for item in entries) != tuple(
            task_aliases[item] for item in task_cids
        ):
            raise MarkdownTaskSourceError(
                "entry order does not match the declared task population"
            )
        expected = _projection_identity(
            plan_root=self.plan_root,
            revision=self.revision,
            task_aliases=task_aliases,
            goal_aliases=goal_aliases,
        )
        if self.projection_id != expected:
            raise MarkdownTaskSourceError(
                "Markdown projection identity does not match"
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "task_cids", task_cids)
        object.__setattr__(self, "goal_cids", goal_cids)
        object.__setattr__(self, "task_aliases", task_aliases)
        object.__setattr__(self, "goal_aliases", goal_aliases)

    @property
    def rendered_text(self) -> str:
        return "\n\n".join(item.rendered_block for item in self.entries) + "\n"

    @property
    def text(self) -> str:
        return self.rendered_text

    @property
    def markdown(self) -> str:
        return self.rendered_text

    @property
    def canonical_bytes(self) -> bytes:
        return self.rendered_text.encode("utf-8")

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self.task_aliases[item] for item in self.task_cids)

    def to_markdown(self) -> str:
        return self.rendered_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "projection_id": self.projection_id,
            "plan_root": self.plan_root,
            "board_namespace": self.board_namespace,
            "task_cids": list(self.task_cids),
            "task_ids": list(self.task_ids),
            "task_aliases": dict(self.task_aliases),
            "goal_cids": list(self.goal_cids),
            "goal_aliases": dict(self.goal_aliases),
            "markdown_sha256": hashlib.sha256(self.canonical_bytes).hexdigest(),
            "byte_count": len(self.canonical_bytes),
        }

    def __str__(self) -> str:
        return self.rendered_text

    def preview(
        self,
        board_text: str,
        *,
        expected_board_revision: str = "",
    ) -> TaskboardMaterializationPreview:
        return preview_taskboard_materialization(
            board_text,
            self.entries,
            expected_board_revision=expected_board_revision,
        )


def _admitted_graph(value: Any) -> tuple[PromptGoalGraph, str]:
    if isinstance(value, PromptGoalGraph):
        raise MarkdownTaskSourceError(
            "a bare graph is not admission evidence; supply an admitted result"
        )
    admitted = bool(getattr(value, "admitted", False))
    graph = getattr(value, "admitted_graph", None)
    plan_root = str(getattr(value, "plan_root_cid", "") or "")
    task_cids = tuple(getattr(value, "task_cids", ()) or ())
    if not admitted or not isinstance(graph, PromptGoalGraph) or not plan_root:
        raise MarkdownTaskSourceError(
            "Markdown projection requires an admitted prompt plan"
        )
    expected_task_cids = tuple(sorted(item.task_cid for item in graph.tasks))
    if tuple(sorted(task_cids)) != expected_task_cids:
        raise MarkdownTaskSourceError(
            "admission task population does not match the admitted graph"
        )
    receipt = getattr(value, "receipt", None)
    candidate_plan_root = str(
        getattr(receipt, "candidate_plan_cid", "") or ""
    )
    if graph.plan_root_cid != candidate_plan_root:
        raise MarkdownTaskSourceError(
            "admission candidate root does not match the admitted graph"
        )
    topology = tuple(task.task_cid for task in _topological_tasks(graph))
    if tuple(getattr(receipt, "topological_task_cids", ()) or ()) != topology:
        raise MarkdownTaskSourceError(
            "admission topology does not match the admitted graph"
        )
    expected_plan_root = prompt_workflow_cid(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "admitted-prompt-plan@1"
            ),
            "candidate_plan_cid": candidate_plan_root,
            "formal_plan_id": str(
                getattr(receipt, "formal_plan_id", "") or ""
            ),
            "ir_receipt_id": str(
                getattr(receipt, "ir_receipt_id", "") or ""
            ),
            "policy_id": str(getattr(receipt, "policy_id", "") or ""),
            "repository_tree_id": str(
                getattr(receipt, "repository_tree_id", "") or ""
            ),
            "task_cids": list(expected_task_cids),
            "topology_id": str(getattr(receipt, "topology_id", "") or ""),
        }
    )
    if plan_root != expected_plan_root:
        raise MarkdownTaskSourceError(
            "admission plan root does not match its admitted evidence"
        )
    return graph, plan_root


def _render_task_block(
    *,
    task: PromptTaskRecord,
    goal: PromptGoalRecord,
    task_alias: str,
    goal_alias: str,
    dependency_aliases: tuple[str, ...],
    plan_root: str,
    candidate_plan_root: str,
    projection_id: str,
    revision: int,
    board_namespace: str,
    task_population_cids: tuple[str, ...],
    goal_population_cids: tuple[str, ...],
    graph_core: Mapping[str, Any],
    assigned_goal_records: Sequence[Mapping[str, Any]],
) -> str:
    task_record = _semantic(task.to_record())
    metadata = {
        "schema": MARKDOWN_TASK_RECORD_SCHEMA,
        "version": MARKDOWN_TASK_SOURCE_VERSION,
        "projection_schema": MARKDOWN_TASK_SOURCE_SCHEMA,
        "projection_revision": revision,
        "projection_id": projection_id,
        "plan_root": plan_root,
        "candidate_plan_root": candidate_plan_root,
        "board_namespace": board_namespace,
        "task_alias": task_alias,
        "task_cid": task.task_cid,
        "goal_alias": goal_alias,
        "goal_cid": goal.goal_cid,
        "dependency_aliases": list(dependency_aliases),
        "dependency_task_cids": list(task.dependency_task_cids),
        "task_population_cids": list(task_population_cids),
        "goal_population_cids": list(goal_population_cids),
        "graph_core": graph_core,
        "task_record": task_record,
        "goal_records": list(assigned_goal_records),
    }
    validation = [
        shlex.join(item.argv)
        + ("" if item.cwd == "." else f" [cwd={item.cwd}]")
        for item in task.validations
    ]
    acceptance = [item.criterion for item in task.acceptance]
    goal_acceptance = [item.criterion for item in goal.acceptance]
    conflict_policy = (
        task.provenance.get("conflict_policy")
        if isinstance(task.provenance, Mapping)
        else ""
    ) or task.fallback_behavior
    lines = [
        f"## {task_alias} {_one_line(task.objective)}",
        "",
        "- Status: todo",
        f"- Priority: {_one_line(task.priority)}",
        f"- Track: {_one_line(task.track)}",
        f"- Depends on: {_csv(dependency_aliases)}",
        f"- Goal id: {goal_alias}",
        f"- Goal CID: {goal.goal_cid}",
        f"- Task key: {_one_line(task.task_key)}",
        f"- Task CID: {task.task_cid}",
        f"- Plan root: {plan_root}",
        f"- Schema: {MARKDOWN_TASK_SOURCE_SCHEMA}",
        f"- Revision: {revision}",
        f"- Projection ID: {projection_id}",
        f"- Board namespace: {_one_line(board_namespace)}",
        f"- Bundle: {_one_line(task.bundle)}",
        f"- Parallel lane: {_one_line(task.parallel_lane)}",
        f"- Resource class: {_one_line(task.resource_class)}",
        f"- Scope paths: {_csv(task.scope_paths)}",
        f"- Outputs: {_csv(item.path for item in task.outputs)}",
        f"- Output effects: {_csv(item.effect for item in task.outputs)}",
        f"- Validation: {' ; '.join(_one_line(item) for item in validation)}",
        f"- Acceptance: {' ; '.join(_one_line(item) for item in acceptance)}",
        f"- Predicted files: {_csv(task.predicted_files)}",
        f"- Conflict policy: {_one_line(conflict_policy)}",
        f"- Risks: {_csv(task.risks)}",
        f"- Assumptions: {_csv(task.assumptions)}",
        f"- Goal title: {_one_line(goal.title)}",
        f"- Goal objective: {_one_line(goal.objective)}",
        f"- Goal dependencies: {_csv(goal.dependency_goal_cids)}",
        f"- Goal acceptance: {' ; '.join(_one_line(item) for item in goal_acceptance)}",
        _metadata_marker(metadata),
    ]
    return "\n".join(lines)


def project_admitted_plan(
    admission: Any,
    *,
    task_prefix: str = DEFAULT_MARKDOWN_TASK_PREFIX,
    board_namespace: str = "prompt-workflow",
    aliases: Mapping[str, str] | None = None,
    revision: int = 1,
) -> MarkdownTaskProjection:
    """Compile an admitted graph into byte-stable supervisor Markdown blocks."""

    graph, plan_root = _admitted_graph(admission)
    if len(graph.tasks) > MAX_TASKBOARD_MATERIALIZATION_ENTRIES:
        raise MarkdownTaskSourceError(
            "admitted plan exceeds the Markdown materialization epoch bound"
        )
    prefix = _validate_alias(task_prefix, noun="task_prefix").rstrip("-")
    namespace = _one_line(board_namespace)
    if not namespace:
        raise MarkdownTaskSourceError("board_namespace must not be empty")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or revision > (2**63 - 1)
    ):
        raise MarkdownTaskSourceError(
            "revision must be a bounded positive integer"
        )

    ordered_tasks = _topological_tasks(graph)
    if aliases is None:
        task_aliases = {
            task.task_cid: f"{prefix}-{index:03d}"
            for index, task in enumerate(ordered_tasks, start=1)
        }
    else:
        supplied = {str(key): str(value) for key, value in aliases.items()}
        expected = {item.task_cid for item in graph.tasks}
        if set(supplied) != expected:
            raise MarkdownTaskSourceError(
                "task alias mapping must cover the exact admitted population"
            )
        task_aliases = supplied
    for alias in task_aliases.values():
        _validate_alias(alias, noun="task alias")
    if len(set(task_aliases.values())) != len(task_aliases):
        raise MarkdownTaskSourceError("task aliases must be unique")

    goal_aliases = {
        goal.goal_cid: _validate_goal_alias(goal.goal_key)
        for goal in graph.goals
    }
    if len(set(goal_aliases.values())) != len(goal_aliases):
        raise MarkdownTaskSourceError("goal aliases must be unique")
    projection_id = _projection_identity(
        plan_root=plan_root,
        revision=revision,
        task_aliases=task_aliases,
        goal_aliases=goal_aliases,
    )
    graph_semantic = _semantic(graph.to_dict())
    graph_core = {
        key: value
        for key, value in graph_semantic.items()
        if key not in {"goals", "tasks"}
    }
    goal_records = tuple(
        _semantic(goal.to_record()) for goal in graph.goals
    )
    goal_assignments: list[list[Mapping[str, Any]]] = [
        [] for _item in ordered_tasks
    ]
    for index, record in enumerate(goal_records):
        goal_assignments[index % len(goal_assignments)].append(record)
    task_population_cids = tuple(task.task_cid for task in ordered_tasks)
    goal_population_cids = tuple(goal.goal_cid for goal in graph.goals)
    goals = {goal.goal_cid: goal for goal in graph.goals}
    entries: list[TaskboardMaterializationEntry] = []
    for index, task in enumerate(ordered_tasks):
        dependency_aliases = tuple(
            task_aliases[item] for item in task.dependency_task_cids
        )
        goal = goals[task.goal_cid]
        rendered = _render_task_block(
            task=task,
            goal=goal,
            task_alias=task_aliases[task.task_cid],
            goal_alias=goal_aliases[goal.goal_cid],
            dependency_aliases=dependency_aliases,
            plan_root=plan_root,
            candidate_plan_root=graph.plan_root_cid,
            projection_id=projection_id,
            revision=revision,
            board_namespace=namespace,
            task_population_cids=task_population_cids,
            goal_population_cids=goal_population_cids,
            graph_core=graph_core,
            assigned_goal_records=goal_assignments[index],
        )
        entries.append(
            TaskboardMaterializationEntry(
                task_id=task_aliases[task.task_cid],
                goal_id=goal_aliases[goal.goal_cid],
                rendered_block=rendered,
            )
        )
    return MarkdownTaskProjection(
        plan_root=plan_root,
        projection_id=projection_id,
        entries=tuple(entries),
        task_cids=task_population_cids,
        task_aliases=task_aliases,
        goal_cids=goal_population_cids,
        goal_aliases=goal_aliases,
        board_namespace=namespace,
        revision=revision,
    )


render_admitted_plan = project_admitted_plan
render_markdown_task_source = project_admitted_plan


def _block_for_marker(
    text: str,
    marker: re.Match[str],
    headings: Sequence[re.Match[str]],
) -> tuple[re.Match[str], str, int]:
    prior = [heading for heading in headings if heading.start() < marker.start()]
    if not prior:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task metadata marker has no task heading"
        )
    heading = prior[-1]
    following = [
        candidate for candidate in headings if candidate.start() > heading.start()
    ]
    end = following[0].start() if following else len(text)
    block = text[heading.start():end].rstrip()
    if len(_MARKER_RE.findall(block)) != 1:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task block must contain exactly one metadata marker"
        )
    return heading, block, text.count("\n", 0, heading.start()) + 1


def _block_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines()[1:]:
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        normalized = key.strip().casefold()
        if normalized in fields:
            raise MarkdownTaskSourceIntegrityError(
                f"Markdown task block repeats field {key.strip()!r}"
            )
        fields[normalized] = value.strip()
    return fields


def parse_markdown_task_source(
    text: str,
    *,
    path: Path | str = Path("taskboard.md"),
    max_tasks: int = MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
    board_revision: str = "",
) -> TaskboardSnapshot:
    """Parse and independently verify a complete canonical projection."""

    if not isinstance(text, str):
        raise TypeError("Markdown task source must be text")
    if (
        isinstance(max_tasks, bool)
        or not isinstance(max_tasks, int)
        or max_tasks < 1
        or max_tasks > MAX_TASKBOARD_MATERIALIZATION_ENTRIES
    ):
        raise ValueError(
            "max_tasks must be within the Markdown task population bound"
        )
    encoded = text.encode("utf-8")
    if _MARKER_PREFIX in text and not _MARKER_RE.search(text):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task source contains a partial metadata marker"
        )
    markers = tuple(_MARKER_RE.finditer(text))
    schema_line_count = len(
        re.findall(
            rf"^- Schema:[ \t]*{re.escape(MARKDOWN_TASK_SOURCE_SCHEMA)}[ \t]*$",
            text,
            flags=re.MULTILINE,
        )
    )
    if schema_line_count != len(markers):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task source contains a partial canonical render"
        )
    if len(markers) > max_tasks:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task population exceeds its configured bound"
        )
    if not markers:
        return TaskboardSnapshot(
            path=Path(path),
            board_revision=board_revision or taskboard_revision(encoded),
            tasks=(),
            byte_count=len(encoded),
        )
    headings = tuple(_HEADING_RE.finditer(text))
    records: list[TaskboardTaskRecord] = []
    payloads: list[dict[str, Any]] = []
    displayed_fields: list[dict[str, str]] = []
    goal_records: dict[str, Mapping[str, Any]] = {}
    task_records: dict[str, Mapping[str, Any]] = {}
    for marker in markers:
        payload = _decode_marker(marker.group("payload"))
        if payload.get("schema") != MARKDOWN_TASK_RECORD_SCHEMA:
            raise MarkdownTaskSourceIntegrityError(
                "unsupported Markdown task record schema"
            )
        if payload.get("version") != MARKDOWN_TASK_SOURCE_VERSION:
            raise MarkdownTaskSourceIntegrityError(
                "unsupported Markdown task record version"
            )
        required_metadata = {
            "projection_schema",
            "projection_revision",
            "projection_id",
            "plan_root",
            "candidate_plan_root",
            "board_namespace",
            "task_alias",
            "task_cid",
            "goal_alias",
            "goal_cid",
            "dependency_aliases",
            "dependency_task_cids",
            "task_population_cids",
            "goal_population_cids",
            "graph_core",
            "task_record",
            "goal_records",
        }
        missing_metadata = sorted(required_metadata - set(payload))
        if missing_metadata:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task metadata is partial; missing "
                + ", ".join(missing_metadata)
            )
        heading, block, source_line = _block_for_marker(text, marker, headings)
        task_id = str(heading.group("task_id") or "")
        title = str(heading.group("title") or "").strip()
        fields = _block_fields(block)
        required_fields = {
            "status",
            "goal id",
            "goal cid",
            "task cid",
            "plan root",
            "schema",
            "revision",
            "projection id",
        }
        missing = sorted(required_fields - set(fields))
        if missing:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task block is partially rendered; missing "
                + ", ".join(missing)
            )
        expected_pairs = {
            "task_alias": task_id,
            "task_cid": fields["task cid"],
            "goal_alias": fields["goal id"],
            "goal_cid": fields["goal cid"],
            "plan_root": fields["plan root"],
            "projection_id": fields["projection id"],
            "projection_schema": fields["schema"],
        }
        for key, expected in expected_pairs.items():
            if str(payload.get(key) or "") != expected:
                raise MarkdownTaskSourceIntegrityError(
                    f"Markdown field {key} does not match canonical metadata"
                )
        try:
            displayed_revision = int(fields["revision"])
        except ValueError as exc:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown projection revision is malformed"
            ) from exc
        if payload.get("projection_revision") != displayed_revision:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown projection revision metadata does not match"
            )
        task_record = payload.get("task_record")
        if not isinstance(task_record, Mapping):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task semantic record is missing"
            )
        task_cid = str(payload["task_cid"])
        if str(task_record.get("content_id") or "") != task_cid:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task CID does not match its semantic record"
            )
        if _record_cid(task_record) != task_cid:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task semantic record has been altered"
            )
        if (
            str(task_record.get("goal_cid") or "") != str(payload["goal_cid"])
            or list(task_record.get("dependency_task_cids") or ())
            != payload["dependency_task_cids"]
        ):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task ownership or dependency projection has drifted"
            )
        if task_cid in task_records:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown source contains duplicate task CIDs"
            )
        task_records[task_cid] = dict(task_record)
        assigned_goals = payload.get("goal_records")
        if not isinstance(assigned_goals, list):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown goal record assignment is malformed"
            )
        for goal_record in assigned_goals:
            if not isinstance(goal_record, Mapping):
                raise MarkdownTaskSourceIntegrityError(
                    "Markdown goal semantic record is malformed"
                )
            goal_cid = str(goal_record.get("content_id") or "")
            if not goal_cid or _record_cid(goal_record) != goal_cid:
                raise MarkdownTaskSourceIntegrityError(
                    "Markdown goal semantic record has been altered"
                )
            if goal_cid in goal_records:
                raise MarkdownTaskSourceIntegrityError(
                    "Markdown source contains duplicate goal CIDs"
                )
            goal_records[goal_cid] = dict(goal_record)
        dependencies = payload.get("dependency_aliases")
        dependency_cids = payload.get("dependency_task_cids")
        if not isinstance(dependencies, list) or not isinstance(
            dependency_cids, list
        ):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown dependency metadata is malformed"
            )
        records.append(
            TaskboardTaskRecord(
                task_id=task_id,
                task_cid=task_cid,
                goal_id=str(payload["goal_alias"]),
                goal_cid=str(payload["goal_cid"]),
                plan_root=str(payload["plan_root"]),
                title=title,
                status=fields["status"],
                dependency_task_ids=tuple(str(item) for item in dependencies),
                dependency_task_cids=tuple(
                    str(item) for item in dependency_cids
                ),
                schema=str(payload["projection_schema"]),
                projection_revision=displayed_revision,
                board_namespace=str(payload.get("board_namespace") or ""),
                source_line=source_line,
                rendered_block=block,
                metadata=payload,
            )
        )
        payloads.append(payload)
        displayed_fields.append(fields)

    first = payloads[0]
    invariant_fields = (
        "projection_schema",
        "projection_revision",
        "projection_id",
        "plan_root",
        "candidate_plan_root",
        "board_namespace",
        "task_population_cids",
        "goal_population_cids",
        "graph_core",
    )
    for payload in payloads[1:]:
        if any(payload.get(name) != first.get(name) for name in invariant_fields):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task records disagree on projection metadata"
            )
    declared_tasks = tuple(str(item) for item in first["task_population_cids"])
    declared_goals = tuple(str(item) for item in first["goal_population_cids"])
    actual_tasks = tuple(item.task_cid for item in records)
    if declared_tasks != actual_tasks or set(declared_tasks) != set(task_records):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown task population drift detected"
        )
    if set(declared_goals) != set(goal_records) or len(declared_goals) != len(
        goal_records
    ):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown goal population drift detected"
        )
    for payload in payloads:
        goal_cid = str(payload["goal_cid"])
        goal_record = goal_records.get(goal_cid)
        if goal_record is None or str(goal_record.get("goal_key") or "") != str(
            payload["goal_alias"]
        ):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown task goal alias/CID mapping drift detected"
            )
    for record, payload, fields in zip(
        records, payloads, displayed_fields, strict=True
    ):
        task_record = payload["task_record"]
        goal_record = goal_records[str(payload["goal_cid"])]
        validations = [
            shlex.join(tuple(str(item) for item in validation["argv"]))
            + (
                ""
                if str(validation.get("cwd") or ".") == "."
                else f" [cwd={validation['cwd']}]"
            )
            for validation in task_record["validations"]
        ]
        provenance = task_record.get("provenance")
        conflict_policy = (
            provenance.get("conflict_policy")
            if isinstance(provenance, Mapping)
            else ""
        ) or task_record["fallback_behavior"]
        expected_display = {
            "priority": _one_line(task_record["priority"]),
            "track": _one_line(task_record["track"]),
            "depends on": _csv(payload["dependency_aliases"]),
            "goal id": str(payload["goal_alias"]),
            "goal cid": str(payload["goal_cid"]),
            "task key": _one_line(task_record["task_key"]),
            "task cid": str(payload["task_cid"]),
            "plan root": str(payload["plan_root"]),
            "schema": str(payload["projection_schema"]),
            "revision": str(payload["projection_revision"]),
            "projection id": str(payload["projection_id"]),
            "board namespace": _one_line(payload["board_namespace"]),
            "bundle": _one_line(task_record["bundle"]),
            "parallel lane": _one_line(task_record["parallel_lane"]),
            "resource class": _one_line(task_record["resource_class"]),
            "scope paths": _csv(task_record["scope_paths"]),
            "outputs": _csv(
                output["path"] for output in task_record["outputs"]
            ),
            "output effects": _csv(
                output["effect"] for output in task_record["outputs"]
            ),
            "validation": " ; ".join(
                _one_line(validation) for validation in validations
            ),
            "acceptance": " ; ".join(
                _one_line(item["criterion"])
                for item in task_record["acceptance"]
            ),
            "predicted files": _csv(task_record["predicted_files"]),
            "conflict policy": _one_line(conflict_policy),
            "risks": _csv(task_record["risks"]),
            "assumptions": _csv(task_record["assumptions"]),
            "goal title": _one_line(goal_record["title"]),
            "goal objective": _one_line(goal_record["objective"]),
            "goal dependencies": _csv(goal_record["dependency_goal_cids"]),
            "goal acceptance": " ; ".join(
                _one_line(item["criterion"])
                for item in goal_record["acceptance"]
            ),
        }
        if record.title != _one_line(task_record["objective"]) or any(
            fields.get(key) != value for key, value in expected_display.items()
        ):
            raise MarkdownTaskSourceIntegrityError(
                "Markdown human-readable projection has drifted from "
                "canonical metadata"
            )
    alias_by_cid = {item.task_cid: item.task_id for item in records}
    for item in records:
        expected_aliases = tuple(
            alias_by_cid[cid] for cid in item.dependency_task_cids
        )
        if item.dependency_task_ids != expected_aliases:
            raise MarkdownTaskSourceIntegrityError(
                "Markdown dependency alias/CID mapping drift detected"
            )
    graph_core = first.get("graph_core")
    if not isinstance(graph_core, Mapping):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown graph core is malformed"
        )
    reconstructed = {
        **dict(graph_core),
        "goals": [
            goal_records[cid]
            for cid in sorted(goal_records)
        ],
        "tasks": [
            task_records[cid]
            for cid in sorted(task_records)
        ],
    }
    plan_root = str(first["plan_root"])
    candidate_plan_root = str(first["candidate_plan_root"])

    def with_lifecycle(
        record: Mapping[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            **dict(record),
            "status": status,
            "created_at_ms": 0,
            "updated_at_ms": 0,
        }

    try:
        validated_graph = PromptGoalGraph.from_dict(
            {
                **dict(graph_core),
                "goals": [
                    with_lifecycle(goal_records[cid], status="proposed")
                    for cid in sorted(goal_records)
                ],
                "tasks": [
                    with_lifecycle(task_records[cid], status="proposed")
                    for cid in sorted(task_records)
                ],
                "evidence": [
                    with_lifecycle(item, status="admitted")
                    for item in graph_core.get("evidence", ())
                ],
                "status": "proposed",
                "created_at_ms": 0,
                "updated_at_ms": 0,
            }
        )
    except (TypeError, ValueError) as exc:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown records do not form a valid admitted graph"
        ) from exc
    if (
        prompt_workflow_cid(reconstructed) != candidate_plan_root
        or validated_graph.plan_root_cid != candidate_plan_root
    ):
        raise MarkdownTaskSourceIntegrityError(
            "Markdown records do not reconstruct the admitted candidate root"
        )
    task_aliases = {item.task_cid: item.task_id for item in records}
    goal_aliases = {
        str(payload["goal_cid"]): str(payload["goal_alias"])
        for payload in payloads
    }
    for goal_cid in declared_goals:
        goal_aliases.setdefault(
            goal_cid,
            str(goal_records[goal_cid].get("goal_key") or ""),
        )
    expected_projection = _projection_identity(
        plan_root=plan_root,
        revision=int(first["projection_revision"]),
        task_aliases=task_aliases,
        goal_aliases=goal_aliases,
    )
    if first["projection_id"] != expected_projection:
        raise MarkdownTaskSourceIntegrityError(
            "Markdown projection identity does not match its aliases"
        )
    return TaskboardSnapshot(
        path=Path(path),
        board_revision=board_revision or taskboard_revision(encoded),
        tasks=tuple(records),
        byte_count=len(encoded),
        plan_root=plan_root,
        projection_schema=str(first["projection_schema"]),
        projection_revision=int(first["projection_revision"]),
        projection_id=str(first["projection_id"]),
    )


def replace_markdown_task_status(
    text: str,
    *,
    task_id: str,
    expected_status: str,
    new_status: str,
) -> str:
    """Replace exactly one task's mutable status without touching its marker."""

    headings = tuple(_HEADING_RE.finditer(text))
    matches = [item for item in headings if item.group("task_id") == task_id]
    if len(matches) != 1:
        raise MarkdownTaskSourceConflict(
            f"expected exactly one task heading for {task_id!r}"
        )
    heading = matches[0]
    following = [item for item in headings if item.start() > heading.start()]
    end = following[0].start() if following else len(text)
    block = text[heading.start():end]
    pattern = re.compile(
        rf"^- Status:[ \t]*{re.escape(expected_status)}[ \t]*$",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if len(pattern.findall(block)) != 1:
        raise MarkdownTaskSourceConflict(
            "task status no longer matches the compare-and-swap input"
        )
    replaced = pattern.sub(f"- Status: {new_status}", block, count=1)
    return text[:heading.start()] + replaced + text[end:]


@dataclass(frozen=True)
class MarkdownMaterializationResult:
    projection: MarkdownTaskProjection
    snapshot: TaskboardSnapshot
    transaction: TaskboardMaterializationTransactionResult | None = None
    no_op: bool = False
    reason_codes: tuple[str, ...] = ()

    @property
    def committed(self) -> bool:
        return self.no_op or bool(
            self.transaction is not None and self.transaction.committed
        )

    @property
    def changed(self) -> bool:
        return bool(self.transaction is not None and self.transaction.changed)

    @property
    def resumed(self) -> bool:
        return bool(self.transaction is not None and self.transaction.resumed)

    @property
    def write_count(self) -> int:
        return self.transaction.write_count if self.transaction is not None else 0

    @property
    def board_revision(self) -> str:
        return self.snapshot.board_revision


class MarkdownTaskSource:
    """Materialize and operate on one canonical Markdown task source."""

    def __init__(
        self,
        path: Path | str,
        *,
        root: Path | str | None = None,
        journal_path: Path | str | None = None,
        events_path: Path | str | None = None,
        task_prefix: str = DEFAULT_MARKDOWN_TASK_PREFIX,
        board_namespace: str = "prompt-workflow",
        max_bytes: int = 4 * 1024 * 1024,
        max_tasks: int = MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
    ) -> None:
        self.store = TaskboardStore(
            path,
            root=root,
            journal_path=journal_path,
            events_path=events_path,
            max_bytes=max_bytes,
            max_tasks=max_tasks,
        )
        self.task_prefix = _validate_alias(task_prefix, noun="task_prefix").rstrip(
            "-"
        )
        self.board_namespace = _one_line(board_namespace)
        if not self.board_namespace:
            raise MarkdownTaskSourceError("board_namespace must not be empty")
        self._pending: dict[str, TaskboardMaterializationPreview] = {}

    @property
    def path(self) -> Path:
        return self.store.path

    @property
    def journal_path(self) -> Path:
        return self.store.journal_path

    @property
    def events_path(self) -> Path:
        return self.store.events_path

    def project(
        self,
        admission: Any,
        *,
        aliases: Mapping[str, str] | None = None,
        revision: int = 1,
    ) -> MarkdownTaskProjection:
        return project_admitted_plan(
            admission,
            task_prefix=self.task_prefix,
            board_namespace=self.board_namespace,
            aliases=aliases,
            revision=revision,
        )

    def render(self, admission: Any, **kwargs: Any) -> str:
        return self.project(admission, **kwargs).rendered_text

    def snapshot(self) -> TaskboardSnapshot:
        return self.store.snapshot()

    load = snapshot

    def query(self, **kwargs: Any) -> tuple[TaskboardTaskRecord, ...]:
        return self.store.query(**kwargs)

    def get(self, task_id: str) -> TaskboardTaskRecord | None:
        return self.store.get(task_id)

    get_task = get

    def ready_set(self, **kwargs: Any) -> tuple[TaskboardTaskRecord, ...]:
        return self.store.ready_set(**kwargs)

    ready = ready_set

    def compare_and_swap_status(
        self,
        task_id: str,
        **kwargs: Any,
    ) -> Any:
        return self.store.compare_and_swap_status(task_id, **kwargs)

    cas_status = compare_and_swap_status

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.store.append_event(event_type, payload)

    def events(self, cursor: Any, **kwargs: Any) -> Any:
        return self.store.events(cursor, **kwargs)

    def watch(self, **kwargs: Any) -> Any:
        return self.store.watch(**kwargs)

    def check_integrity(self) -> Any:
        return self.store.check_integrity()

    integrity = check_integrity

    @staticmethod
    def _recover_preview(
        text: str,
        projection: MarkdownTaskProjection,
    ) -> TaskboardMaterializationPreview | None:
        first_heading = f"## {projection.entries[0].task_id}"
        positions = [
            match.start()
            for match in re.finditer(
                rf"(?m)^{re.escape(first_heading)}(?=[ \t]|$)", text
            )
        ]
        for position in positions:
            prefix = text[:position]
            bases = {prefix}
            if prefix.endswith("\n"):
                bases.add(prefix[:-1])
            if prefix.endswith("\n\n"):
                bases.add(prefix[:-2])
            for base in bases:
                for count in range(1, len(projection.entries) + 1):
                    preview = preview_taskboard_materialization(
                        base, projection.entries[:count]
                    )
                    if preview.candidate_text != text:
                        continue
                    return projection.preview(base)
        return None

    def materialize(
        self,
        admission_or_projection: Any,
        *,
        aliases: Mapping[str, str] | None = None,
        revision: int = 1,
        expected_board_revision: str = "",
        epoch_id: str = "",
    ) -> MarkdownMaterializationResult:
        projection = (
            admission_or_projection
            if isinstance(admission_or_projection, MarkdownTaskProjection)
            else self.project(
                admission_or_projection,
                aliases=aliases,
                revision=revision,
            )
        )
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            raw = b""
        if len(raw) > self.store.max_bytes:
            raise MarkdownTaskSourceConflict(
                "taskboard exceeds its configured byte bound"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MarkdownTaskSourceConflict("taskboard is not UTF-8") from exc
        current_revision = taskboard_revision(raw)
        if expected_board_revision and expected_board_revision != current_revision:
            raise MarkdownTaskSourceConflict("stale taskboard revision")

        preview = self._pending.get(projection.projection_id)
        if preview is None:
            recovered_preview = self._recover_preview(text, projection)
            if recovered_preview is not None:
                epoch = epoch_id or projection.projection_id
                try:
                    recoverable = _has_taskboard_materialization_transaction(
                        self.journal_path,
                        recovered_preview,
                        epoch,
                    )
                except ValueError as exc:
                    raise MarkdownTaskSourceConflict(str(exc)) from exc
                if recoverable:
                    preview = recovered_preview
        if preview is None:
            try:
                snapshot = parse_markdown_task_source(
                    text,
                    path=self.path,
                    max_tasks=self.store.max_tasks,
                    board_revision=current_revision,
                )
            except MarkdownTaskSourceIntegrityError as exc:
                raise MarkdownTaskSourceConflict(str(exc)) from exc
            if snapshot.tasks:
                if (
                    snapshot.plan_root != projection.plan_root
                    or snapshot.task_cids != projection.task_cids
                    or snapshot.projection_id != projection.projection_id
                ):
                    raise MarkdownTaskSourceConflict(
                        "taskboard task population or plan root drift detected"
                    )
                return MarkdownMaterializationResult(
                    projection=projection,
                    snapshot=snapshot,
                    no_op=True,
                    reason_codes=("identical_replay",),
                )
            preview = projection.preview(
                text,
                expected_board_revision=expected_board_revision,
            )
        if (
            len(preview.candidate_text.encode("utf-8"))
            > self.store.max_bytes
        ):
            raise MarkdownTaskSourceConflict(
                "candidate taskboard exceeds its configured byte bound"
            )
        self._pending[projection.projection_id] = preview
        transaction = commit_taskboard_materialization(
            self.path,
            self.journal_path,
            preview,
            epoch_id=epoch_id or projection.projection_id,
            expected_board_revision=preview.base_board_revision,
        )
        if transaction.state is TaskboardMaterializationTransactionState.BLOCKED:
            raise MarkdownTaskSourceConflict(
                "taskboard materialization blocked: "
                + ", ".join(transaction.reason_codes)
            )
        if not transaction.committed:
            return MarkdownMaterializationResult(
                projection=projection,
                snapshot=TaskboardSnapshot(
                    path=self.path,
                    board_revision=taskboard_revision(
                        self.path.read_bytes() if self.path.exists() else b""
                    ),
                    tasks=(),
                    byte_count=(
                        self.path.stat().st_size if self.path.exists() else 0
                    ),
                ),
                transaction=transaction,
                reason_codes=transaction.reason_codes,
            )
        snapshot = self.store.snapshot()
        if (
            snapshot.plan_root != projection.plan_root
            or snapshot.task_cids != projection.task_cids
            or snapshot.projection_id != projection.projection_id
        ):
            raise MarkdownTaskSourceIntegrityError(
                "committed Markdown task population failed verification"
            )
        self._pending.pop(projection.projection_id, None)
        return MarkdownMaterializationResult(
            projection=projection,
            snapshot=snapshot,
            transaction=transaction,
            no_op=not transaction.changed and transaction.write_count == 0,
            reason_codes=transaction.reason_codes,
        )

    commit = materialize


def materialize_markdown_task_source(
    path: Path | str,
    admission: Any,
    **kwargs: Any,
) -> MarkdownMaterializationResult:
    """Functional facade for one canonical Markdown materialization."""

    source_keys = {
        "root",
        "journal_path",
        "events_path",
        "task_prefix",
        "board_namespace",
        "max_bytes",
        "max_tasks",
    }
    source_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in source_keys
    }
    return MarkdownTaskSource(path, **source_kwargs).materialize(
        admission, **kwargs
    )


MarkdownTaskSourceProjection = MarkdownTaskProjection
MarkdownTaskSourceResult = MarkdownMaterializationResult


__all__ = [
    "DEFAULT_MARKDOWN_TASK_PREFIX",
    "MARKDOWN_TASK_RECORD_SCHEMA",
    "MARKDOWN_TASK_SOURCE_SCHEMA",
    "MARKDOWN_TASK_SOURCE_VERSION",
    "MarkdownMaterializationResult",
    "MarkdownTaskProjection",
    "MarkdownTaskSource",
    "MarkdownTaskSourceConflict",
    "MarkdownTaskSourceError",
    "MarkdownTaskSourceIntegrityError",
    "MarkdownTaskSourceProjection",
    "MarkdownTaskSourceResult",
    "materialize_markdown_task_source",
    "parse_markdown_task_source",
    "project_admitted_plan",
    "render_admitted_plan",
    "render_markdown_task_source",
    "replace_markdown_task_status",
]
