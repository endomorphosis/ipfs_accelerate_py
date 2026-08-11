"""Deterministic post-completion ops refill for drained implementation boards.

When a supervised todo board reaches zero open tasks, programs often still need
a bounded operator/ops follow-on phase (completion-gate validation, PR package
prep, live canary, dry-run publish, handoff receipts). Free-form objective
refill is too open-ended for reviewed boards; this module seeds an explicit
catalog of post-completion task cards once per catalog digest.

The seeder is idempotent: existing task ids and semantic identities are skipped,
and strategy records prevent repeated exhaustive seeds after a successful pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backlog_refinery import (
    effective_open_task_count,
    ensure_task_blocks_present,
    load_strategy,
    normalize_task_id,
    task_block_is_present,
    task_ids_from_todo_text,
    write_json,
)

POST_COMPLETION_OPS_ANALYZER_VERSION = "post-completion-ops-refill/v1"
POST_COMPLETION_OPS_CATALOG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/post-completion-ops-catalog@1"
)
POST_COMPLETION_OPS_STRATEGY_DIGEST_KEY = "post_completion_ops_catalog_digest"
POST_COMPLETION_OPS_STRATEGY_SEEDED_AT_KEY = "post_completion_ops_seeded_at"
POST_COMPLETION_OPS_STRATEGY_SEEDED_IDS_KEY = "post_completion_ops_seeded_task_ids"
POST_COMPLETION_OPS_STRATEGY_TASK_COUNT_KEY = "post_completion_ops_seeded_task_count"
POST_COMPLETION_OPS_STRATEGY_LAST_REASON_KEY = "post_completion_ops_last_reason"


@dataclass(frozen=True)
class PostCompletionOpsTaskSpec:
    """One reviewed post-completion task from a catalog."""

    task_id: str
    title: str
    depends_on: tuple[str, ...] = ()
    goal_id: str = ""
    track: str = "post-completion-ops"
    priority: str = "P0"
    bundle: str = "post-completion-ops"
    parallel_lane: str = ""
    outputs: tuple[str, ...] = ()
    validation: str = ""
    acceptance: str = ""
    effects: str = ""
    preconditions: str = ""
    conflict_policy: str = (
        "Own only the listed outputs for this post-completion ops task; "
        "do not reopen the drained foundation board or bypass human approval gates."
    )
    resource_class: str = "cpu-medium"
    token_class: str = "medium"
    estimated_tokens: int = 8000
    board_namespace: str = ""
    is_schedulable: bool = True
    review_only: bool = False
    shard_hint: int | None = None

    def numeric_id(self) -> int | None:
        match = re.search(r"(\d+)$", self.task_id)
        return int(match.group(1)) if match else None


@dataclass(frozen=True)
class PostCompletionOpsCatalog:
    """Reviewed catalog of post-completion ops tasks."""

    schema: str
    program: str
    task_prefix: str
    trigger: str
    tasks: tuple[PostCompletionOpsTaskSpec, ...]
    source_path: Path | None = None
    notes: str = ""

    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "program": self.program,
            "task_prefix": self.task_prefix,
            "trigger": self.trigger,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "depends_on": list(task.depends_on),
                    "goal_id": task.goal_id,
                    "track": task.track,
                    "priority": task.priority,
                    "bundle": task.bundle,
                    "parallel_lane": task.parallel_lane,
                    "outputs": list(task.outputs),
                    "validation": task.validation,
                    "acceptance": task.acceptance,
                    "effects": task.effects,
                    "preconditions": task.preconditions,
                    "conflict_policy": task.conflict_policy,
                    "resource_class": task.resource_class,
                    "token_class": task.token_class,
                    "estimated_tokens": task.estimated_tokens,
                    "board_namespace": task.board_namespace,
                    "is_schedulable": task.is_schedulable,
                    "review_only": task.review_only,
                    "shard_hint": task.shard_hint,
                }
                for task in self.tasks
            ],
            "notes": self.notes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(encoded).hexdigest()


@dataclass
class PostCompletionOpsRefillResult:
    """Outcome of one post-completion ops seed attempt."""

    reason: str
    catalog_digest: str = ""
    open_task_count: int = 0
    board_task_count: int = 0
    seeded_task_ids: list[str] = field(default_factory=list)
    already_present_task_ids: list[str] = field(default_factory=list)
    expanded_execution_slice_task_ids: list[str] = field(default_factory=list)
    lane_slice_updates: dict[str, list[str]] = field(default_factory=dict)
    board_updated: bool = False
    config_updated: bool = False
    strategy_updated: bool = False
    error: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "catalog_digest": self.catalog_digest,
            "open_task_count": self.open_task_count,
            "board_task_count": self.board_task_count,
            "seeded_task_ids": list(self.seeded_task_ids),
            "already_present_task_ids": list(self.already_present_task_ids),
            "expanded_execution_slice_task_ids": list(
                self.expanded_execution_slice_task_ids
            ),
            "lane_slice_updates": {
                key: list(values) for key, values in self.lane_slice_updates.items()
            },
            "board_updated": self.board_updated,
            "config_updated": self.config_updated,
            "strategy_updated": self.strategy_updated,
            "generated_count": len(self.seeded_task_ids),
            "task_ids": list(self.seeded_task_ids),
            "error": self.error,
            "analyzer_version": POST_COMPLETION_OPS_ANALYZER_VERSION,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return tuple(item for item in parts if item)
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_post_completion_ops_catalog(path: Path) -> PostCompletionOpsCatalog:
    """Load and validate a post-completion ops catalog JSON document."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"post-completion ops catalog must be an object: {path}")
    schema = str(payload.get("schema") or "").strip()
    if schema != POST_COMPLETION_OPS_CATALOG_SCHEMA:
        raise ValueError(
            f"unsupported post-completion ops catalog schema {schema!r}; "
            f"expected {POST_COMPLETION_OPS_CATALOG_SCHEMA!r}"
        )
    task_prefix = str(payload.get("task_prefix") or "").strip()
    if not task_prefix:
        raise ValueError("post-completion ops catalog requires task_prefix")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("post-completion ops catalog requires a non-empty tasks list")
    tasks: list[PostCompletionOpsTaskSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"catalog tasks[{index}] must be an object")
        task_id = normalize_task_id(raw.get("task_id") or raw.get("id") or "")
        title = str(raw.get("title") or "").strip()
        if not task_id or not title:
            raise ValueError(f"catalog tasks[{index}] requires task_id and title")
        if not task_id.startswith(task_prefix):
            raise ValueError(
                f"catalog task {task_id} must use task_prefix {task_prefix}"
            )
        if task_id in seen:
            raise ValueError(f"duplicate catalog task_id: {task_id}")
        seen.add(task_id)
        tasks.append(
            PostCompletionOpsTaskSpec(
                task_id=task_id,
                title=title,
                depends_on=_as_str_tuple(raw.get("depends_on")),
                goal_id=str(raw.get("goal_id") or "").strip(),
                track=str(raw.get("track") or "post-completion-ops").strip(),
                priority=str(raw.get("priority") or "P0").strip(),
                bundle=str(raw.get("bundle") or "post-completion-ops").strip(),
                parallel_lane=str(raw.get("parallel_lane") or "").strip(),
                outputs=_as_str_tuple(raw.get("outputs")),
                validation=str(raw.get("validation") or "").strip(),
                acceptance=str(raw.get("acceptance") or "").strip(),
                effects=str(raw.get("effects") or "").strip(),
                preconditions=str(raw.get("preconditions") or "").strip(),
                conflict_policy=str(
                    raw.get("conflict_policy")
                    or PostCompletionOpsTaskSpec.conflict_policy
                ).strip(),
                resource_class=str(raw.get("resource_class") or "cpu-medium").strip(),
                token_class=str(raw.get("token_class") or "medium").strip(),
                estimated_tokens=int(raw.get("estimated_tokens") or 8000),
                board_namespace=str(raw.get("board_namespace") or "").strip(),
                is_schedulable=bool(raw.get("is_schedulable", True)),
                review_only=bool(raw.get("review_only", False)),
                shard_hint=_as_optional_int(raw.get("shard_hint")),
            )
        )
    return PostCompletionOpsCatalog(
        schema=schema,
        program=str(payload.get("program") or "").strip(),
        task_prefix=task_prefix,
        trigger=str(payload.get("trigger") or "board_drained").strip() or "board_drained",
        tasks=tuple(tasks),
        source_path=path,
        notes=str(payload.get("notes") or "").strip(),
    )


def render_post_completion_task_block(
    task: PostCompletionOpsTaskSpec,
    *,
    board_namespace: str = "",
    allow_concurrent_with: str = "",
) -> str:
    """Render one PATLAW-style markdown task card for the todo board."""

    namespace = task.board_namespace or board_namespace or "post-completion-ops"
    depends = ", ".join(task.depends_on) if task.depends_on else ""
    outputs = ", ".join(task.outputs) if task.outputs else task.task_id
    validation = task.validation or f"test -n {task.task_id}"
    acceptance = (
        task.acceptance
        or "Post-completion ops task produces its declared outputs with fail-closed validation."
    )
    preconditions = (
        task.preconditions
        or "Foundation board is drained or prior post-completion dependencies are complete."
    )
    effects = (
        task.effects
        or "Advances the bounded post-completion ops phase without reopening foundation work."
    )
    lines = [
        f"## {task.task_id} {task.title}",
        "",
        "- Status: pending",
        "- Completion: manual",
        f"- Is schedulable: {'true' if task.is_schedulable else 'false'}",
        f"- Review only: {'true' if task.review_only else 'false'}",
        f"- Priority: {task.priority or 'P0'}",
        f"- Track: {task.track or 'post-completion-ops'}",
        f"- Depends on: {depends}",
        f"- Goal id: {task.goal_id or 'PATLAW-G200'}",
        f"- Outputs: {outputs}",
        f"- Validation: {validation}",
        f"- Board namespace: {namespace}",
        f"- Bundle: {task.bundle or 'post-completion-ops'}",
        f"- Parallel lane: {task.parallel_lane or 'post-completion-ops'}",
        f"- Resource class: {task.resource_class}",
        f"- Token class: {task.token_class}",
        f"- Estimated tokens: {task.estimated_tokens}",
        f"- Predicted files: {outputs}",
        f"- Allow concurrent with: {allow_concurrent_with}",
        f"- Conflict policy: {task.conflict_policy}",
        f"- Preconditions: {preconditions}",
        f"- Effects: {effects}",
        f"- Acceptance: {acceptance}",
        "- Dedupe key: "
        f"post-completion-ops:{task.task_id}:{sha256(task.title.encode('utf-8')).hexdigest()[:16]}",
    ]
    return "\n".join(lines)


def task_shard_index(
    task: PostCompletionOpsTaskSpec,
    *,
    shard_count: int,
) -> int:
    """Assign a catalog task to a stable shard index."""

    count = max(1, int(shard_count))
    if task.shard_hint is not None:
        return int(task.shard_hint) % count
    numeric = task.numeric_id()
    if numeric is not None:
        return numeric % count
    digest = sha256(task.task_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % count


def expand_lane_slices_for_catalog(
    lane_slices: Mapping[str, Sequence[str]] | None,
    catalog: PostCompletionOpsCatalog,
    *,
    shard_count: int,
) -> dict[str, list[str]]:
    """Return lane_slices with catalog task ids inserted into their shards."""

    count = max(1, int(shard_count))
    updated: dict[str, list[str]] = {
        str(index): [] for index in range(count)
    }
    if isinstance(lane_slices, Mapping):
        for key, values in lane_slices.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            bucket = updated.setdefault(str(key), [])
            for item in values:
                text = str(item).strip()
                if text and text not in bucket:
                    bucket.append(text)
    for task in catalog.tasks:
        if not task.is_schedulable:
            continue
        shard = str(task_shard_index(task, shard_count=count))
        bucket = updated.setdefault(shard, [])
        if task.task_id not in bucket:
            bucket.append(task.task_id)
    return updated


def execution_slice_task_ids_for_shard(
    catalog: PostCompletionOpsCatalog,
    *,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    """Catalog task ids owned by one shard."""

    count = max(1, int(shard_count))
    index = int(shard_index) % count
    return [
        task.task_id
        for task in catalog.tasks
        if task.is_schedulable
        and task_shard_index(task, shard_count=count) == index
    ]


def board_is_drained_for_prefix(
    todo_text: str,
    *,
    task_prefix: str,
    state_path: Path | None = None,
) -> tuple[bool, int, int]:
    """Return (drained, open_count, total_task_count) for a board prefix."""

    open_count = effective_open_task_count(
        todo_text, state_path=state_path, task_prefix=task_prefix
    )
    total = len(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    return open_count == 0 and total > 0, open_count, total


def update_config_lane_slices(
    config_path: Path,
    lane_slices: Mapping[str, Sequence[str]],
) -> bool:
    """Rewrite config.lane_slices when the assignment changes."""

    if not config_path.exists():
        return False
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config is not a JSON object: {config_path}")
    current = payload.get("lane_slices")
    normalized = {
        str(key): [str(item) for item in values]
        for key, values in lane_slices.items()
    }
    if current == normalized:
        return False
    payload["lane_slices"] = normalized
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return True


def seed_post_completion_ops(
    *,
    todo_path: Path,
    strategy_path: Path,
    catalog: PostCompletionOpsCatalog,
    state_path: Path | None = None,
    config_path: Path | None = None,
    board_namespace: str = "",
    shard_count: int = 1,
    shard_index: int = 0,
    force: bool = False,
    require_drained: bool = True,
) -> PostCompletionOpsRefillResult:
    """Seed missing catalog tasks onto a drained board and expand lane slices."""

    digest = catalog.digest()
    if not todo_path.exists():
        return PostCompletionOpsRefillResult(
            reason="todo_missing",
            catalog_digest=digest,
            error=f"todo path does not exist: {todo_path}",
        )
    todo_text = todo_path.read_text(encoding="utf-8")
    drained, open_count, total = board_is_drained_for_prefix(
        todo_text,
        task_prefix=catalog.task_prefix,
        state_path=state_path,
    )
    result = PostCompletionOpsRefillResult(
        reason="not_seeded",
        catalog_digest=digest,
        open_task_count=open_count,
        board_task_count=total,
    )
    strategy = load_strategy(strategy_path)
    previous_digest = str(strategy.get(POST_COMPLETION_OPS_STRATEGY_DIGEST_KEY) or "")
    present = [
        task.task_id
        for task in catalog.tasks
        if task_block_is_present(todo_text, task.task_id)
    ]
    missing = [
        task.task_id
        for task in catalog.tasks
        if task.task_id not in present
    ]
    result.already_present_task_ids = present
    result.expanded_execution_slice_task_ids = execution_slice_task_ids_for_shard(
        catalog, shard_count=shard_count, shard_index=shard_index
    )

    # Catalog already fully present: expand slices only and keep strategy in sync.
    if not missing and not force:
        if previous_digest != digest or not strategy.get(
            POST_COMPLETION_OPS_STRATEGY_SEEDED_IDS_KEY
        ):
            strategy[POST_COMPLETION_OPS_STRATEGY_DIGEST_KEY] = digest
            strategy[POST_COMPLETION_OPS_STRATEGY_SEEDED_AT_KEY] = _utc_now()
            strategy[POST_COMPLETION_OPS_STRATEGY_SEEDED_IDS_KEY] = list(present)
            strategy[POST_COMPLETION_OPS_STRATEGY_TASK_COUNT_KEY] = total
            strategy[POST_COMPLETION_OPS_STRATEGY_LAST_REASON_KEY] = "already_seeded"
            write_json(strategy_path, strategy)
            result.strategy_updated = True
        result.reason = "already_seeded"
        return result

    # First-time seed of missing catalog tasks requires a drained foundation
    # board unless the caller opts out or forces.
    if require_drained and not drained and not force:
        result.reason = "board_not_drained"
        return result

    blocks: list[tuple[str, str]] = []
    already_present: list[str] = []
    for task in catalog.tasks:
        if task_block_is_present(todo_text, task.task_id):
            already_present.append(task.task_id)
            continue
        blocks.append(
            (
                task.task_id,
                render_post_completion_task_block(
                    task, board_namespace=board_namespace or catalog.program
                ),
            )
        )
    board_updated = False
    if blocks:
        board_updated = ensure_task_blocks_present(todo_path, blocks)
    seeded_ids = [task_id for task_id, _ in blocks] if board_updated else []
    if blocks and not board_updated:
        # Semantic-duplicate or concurrent writer may have filled the board.
        seeded_ids = []
        refreshed = todo_path.read_text(encoding="utf-8")
        for task_id, _ in blocks:
            if task_block_is_present(refreshed, task_id):
                already_present.append(task_id)

    lane_slices = expand_lane_slices_for_catalog(
        {},
        catalog,
        shard_count=shard_count,
    )
    config_updated = False
    if config_path is not None and config_path.exists():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config_payload = {}
        existing_slices = (
            config_payload.get("lane_slices")
            if isinstance(config_payload, dict)
            else {}
        )
        lane_slices = expand_lane_slices_for_catalog(
            existing_slices if isinstance(existing_slices, Mapping) else {},
            catalog,
            shard_count=shard_count,
        )
        config_updated = update_config_lane_slices(config_path, lane_slices)

    strategy[POST_COMPLETION_OPS_STRATEGY_DIGEST_KEY] = digest
    strategy[POST_COMPLETION_OPS_STRATEGY_SEEDED_AT_KEY] = _utc_now()
    strategy[POST_COMPLETION_OPS_STRATEGY_SEEDED_IDS_KEY] = sorted(
        set(already_present) | set(seeded_ids) | set(task.task_id for task in catalog.tasks)
    )
    strategy[POST_COMPLETION_OPS_STRATEGY_TASK_COUNT_KEY] = len(
        task_ids_from_todo_text(
            todo_path.read_text(encoding="utf-8"),
            task_prefix=catalog.task_prefix,
        )
    )
    strategy[POST_COMPLETION_OPS_STRATEGY_LAST_REASON_KEY] = (
        "seeded" if seeded_ids or config_updated else "already_present"
    )
    write_json(strategy_path, strategy)

    result.seeded_task_ids = seeded_ids
    result.already_present_task_ids = sorted(set(already_present))
    result.expanded_execution_slice_task_ids = execution_slice_task_ids_for_shard(
        catalog, shard_count=shard_count, shard_index=shard_index
    )
    result.lane_slice_updates = lane_slices
    result.board_updated = board_updated
    result.config_updated = config_updated
    result.strategy_updated = True
    result.reason = (
        "seeded"
        if seeded_ids or config_updated
        else "already_present"
    )
    return result
