"""Analysis-driven parallel lane planning for reviewed todo boards.

Turns markdown task boards into conflict-free execution slices by combining:

* predicted_files / outputs / predicted paths (declared work contracts)
* optional Python AST enrichment for those files (imports, symbols, interfaces)
* the core :func:`materialize_task_conflict_graph` coloring engine
* dependency edges (ready-order only; conflicts still force different colors)

This does **not** invent tasks, weaken reviewed-board execution, or treat
packet membership as completion authority.  It only proposes ``lane_slices``
that maximize concurrent width under file/AST conflict constraints.

Formal verification / logic signals may be supplied as optional
``obligation_locality_keys`` or ``proof_cache_key_prefixes`` on tasks; when
present they co-locate prove work on the same tree without merging conflicting
edit surfaces.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.conflict_graph import (
    build_python_ast_blob_record,
    materialize_task_conflict_graph,
)

PARALLEL_LANE_PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/parallel-lane-plan@1"
)

# Task ids look like PAAR-010, PATLAW-002, CBP-015, TASK-A (program-prefix + id).
# Plain section titles ("Objective", "Parallel lanes") are not tasks.
_TASK_HEADER = re.compile(
    r"^##\s+((?:[A-Z][A-Z0-9]*)-(?:\d+|[A-Z][A-Z0-9_-]*))\s+(.+?)\s*$",
    re.M,
)
_FIELD = re.compile(r"^-\s+([^:]+):\s*(.*?)\s*$")
_NON_EXECUTABLE = frozenset(
    {"completed", "cancelled", "canceled", "deferred", "done", "complete"}
)


def _split_csv(value: str) -> list[str]:
    if not value or not str(value).strip():
        return []
    return [
        part.strip()
        for part in re.split(r"[,;\n]", str(value))
        if part.strip()
    ]


def parse_markdown_todo_tasks(text: str) -> list[dict[str, Any]]:
    """Parse ``## TASK-ID Title`` markdown cards into task mappings."""

    blocks = re.split(r"(?=^##\s+\S)", text, flags=re.M)
    tasks: list[dict[str, Any]] = []
    for block in blocks:
        header = _TASK_HEADER.match(block)
        if not header:
            continue
        task_id = header.group(1).strip()
        title = header.group(2).strip()
        fields: dict[str, str] = {}
        for line in block.splitlines()[1:]:
            match = _FIELD.match(line)
            if match:
                fields[match.group(1).strip()] = match.group(2).strip()
        status = str(fields.get("Status") or fields.get("status") or "").strip()
        is_schedulable = str(
            fields.get("Is schedulable") or fields.get("is schedulable") or "true"
        ).strip().casefold()
        if is_schedulable in {"false", "0", "no"}:
            continue
        if status.casefold() in _NON_EXECUTABLE:
            continue
        predicted = _split_csv(
            fields.get("Predicted files")
            or fields.get("predicted files")
            or fields.get("Predicted Files")
            or ""
        )
        outputs = _split_csv(fields.get("Outputs") or fields.get("outputs") or "")
        depends = _split_csv(
            fields.get("Depends on") or fields.get("depends on") or ""
        )
        allow = _split_csv(
            fields.get("Allow concurrent with")
            or fields.get("allow concurrent with")
            or ""
        )
        symbols = _split_csv(
            fields.get("AST query") or fields.get("ast query") or ""
        )
        interfaces = _split_csv(
            fields.get("Interfaces") or fields.get("interfaces") or ""
        )
        task: dict[str, Any] = {
            "task_id": task_id,
            "task_cid": f"task:{task_id}",
            "title": title,
            "status": status or "pending",
            "goal_id": fields.get("Goal id") or fields.get("goal id") or "",
            "bundle": fields.get("Bundle") or fields.get("bundle") or "",
            "parallel_lane": fields.get("Parallel lane")
            or fields.get("parallel lane")
            or "",
            "predicted_files": predicted or list(outputs),
            "outputs": outputs,
            "depends_on": depends,
            "dependencies": depends,
            "allow_concurrent_with": allow,
            "ast_symbols": symbols,
            "interfaces": interfaces,
            "resource_class": fields.get("Resource class")
            or fields.get("resource class")
            or "cpu-small",
            "token_class": fields.get("Token class")
            or fields.get("token class")
            or "medium",
            "estimated_tokens": int(
                re.sub(r"\D", "", fields.get("Estimated tokens") or "0") or "0"
            ),
            "validation": fields.get("Validation") or fields.get("validation") or "",
            "conflict_policy": fields.get("Conflict policy")
            or fields.get("conflict policy")
            or "",
            "metadata": dict(fields),
        }
        tasks.append(task)
    return tasks


def _collect_ast_records(
    tasks: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None,
    max_files: int = 200,
) -> list[Any]:
    """Parse predicted Python files into AST blob records for conflict surfaces."""

    if repo_root is None:
        return []
    root = Path(repo_root)
    seen: set[str] = set()
    records: list[Any] = []
    for task in tasks:
        for rel in list(task.get("predicted_files") or []) + list(
            task.get("outputs") or []
        ):
            path = str(rel or "").replace("\\", "/").lstrip("./")
            if not path.endswith(".py") or path in seen:
                continue
            seen.add(path)
            if len(seen) > max_files:
                break
            full = root / path
            if not full.is_file():
                continue
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                records.append(build_python_ast_blob_record(source))
            except Exception:  # noqa: BLE001 — AST failure must not block planning
                continue
    return records


def _build_concurrency_overrides(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate allow_concurrent_with into non-blocking override pairs."""

    by_id = {str(t.get("task_id")): t for t in tasks}
    overrides: list[dict[str, Any]] = []
    for task in tasks:
        left = str(task.get("task_id") or "")
        for peer in task.get("allow_concurrent_with") or []:
            right = str(peer).strip()
            if not right or right not in by_id:
                continue
            overrides.append(
                {
                    "left": str(task.get("task_cid") or f"task:{left}"),
                    "right": str(
                        by_id[right].get("task_cid") or f"task:{right}"
                    ),
                    "allowed": True,
                    "reason": "board_allow_concurrent_with",
                }
            )
    return overrides


@dataclass(frozen=True)
class ParallelLanePlan:
    """Result of analysis-driven lane planning."""

    lane_slices: dict[str, list[str]]
    max_lanes: int
    task_count: int
    independent_width: int
    blocking_edge_count: int
    dependency_edges: list[tuple[str, str]]
    rationale: list[str]
    task_colors: dict[str, int]
    schema: str = PARALLEL_LANE_PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "max_lanes": self.max_lanes,
            "task_count": self.task_count,
            "independent_width": self.independent_width,
            "blocking_edge_count": self.blocking_edge_count,
            "lane_slices": {
                key: list(value) for key, value in sorted(self.lane_slices.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
            },
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "task_colors": dict(sorted(self.task_colors.items())),
            "rationale": list(self.rationale),
        }


def plan_parallel_lanes(
    tasks: Sequence[Mapping[str, Any]],
    *,
    max_lanes: int = 6,
    repo_root: Path | str | None = None,
    use_ast: bool = True,
    honor_allow_concurrent: bool = True,
) -> ParallelLanePlan:
    """Color ready tasks into at most ``max_lanes`` conflict-free execution slices.

    Tasks that declare file/AST conflicts are never placed in the same lane.
    Dependencies are retained as edges for operator inspection; the scheduler
    still enforces readiness.  When more conflict-free colors are needed than
    ``max_lanes``, colors are folded modulo ``max_lanes`` **only among
    non-adjacent** tasks; residual conflicts force additional virtual capacity
    via the conflict graph's own max_lanes semantics (which creates extra colors
    rather than co-scheduling conflicts).
    """

    if max_lanes < 1:
        raise ValueError("max_lanes must be >= 1")
    material = [dict(task) for task in tasks]
    if not material:
        return ParallelLanePlan(
            lane_slices={str(i): [] for i in range(max_lanes)},
            max_lanes=max_lanes,
            task_count=0,
            independent_width=0,
            blocking_edge_count=0,
            dependency_edges=[],
            rationale=["no schedulable tasks"],
            task_colors={},
        )

    # Ensure every task has a stable cid and predicted files.
    for task in material:
        tid = str(task.get("task_id") or "").strip()
        if not tid:
            raise ValueError("every task requires task_id")
        task.setdefault("task_cid", f"task:{tid}")
        if not task.get("predicted_files"):
            task["predicted_files"] = list(task.get("outputs") or [])

    root = Path(repo_root).expanduser().resolve() if repo_root else None
    ast_records = (
        _collect_ast_records(material, repo_root=root) if use_ast and root else []
    )
    overrides = (
        _build_concurrency_overrides(material) if honor_allow_concurrent else []
    )

    graph = materialize_task_conflict_graph(
        material,
        repo_root=root,
        ast_records=ast_records or None,
        concurrency_overrides=overrides or None,
        max_lanes=max_lanes,
    )

    cid_to_id = {
        str(surface.task_cid): str(surface.task_id)
        for surface in graph.surfaces.values()
    }
    # Prefer board task_id when surfaces used synthetic ids.
    for task in material:
        cid_to_id[str(task["task_cid"])] = str(task["task_id"])

    colors = {
        cid_to_id.get(cid, cid): int(color)
        for cid, color in graph.colors.items()
    }

    # Fold colors into [0, max_lanes) while keeping conflicting pairs apart.
    blocking_pairs: set[frozenset[str]] = set()
    for edge in graph.edges:
        if not edge.blocks_concurrency:
            continue
        left = cid_to_id.get(edge.left_task_cid, edge.left_task_cid)
        right = cid_to_id.get(edge.right_task_cid, edge.right_task_cid)
        blocking_pairs.add(frozenset((left, right)))

    # Greedy recolor into max_lanes with conflict constraints.
    order = sorted(colors.keys())
    # Highest degree first for better packing.
    degree = defaultdict(int)
    for pair in blocking_pairs:
        for node in pair:
            degree[node] += 1
    order.sort(key=lambda tid: (-degree[tid], tid))

    lane_of: dict[str, int] = {}
    for tid in order:
        forbidden = {
            lane_of[peer]
            for pair in blocking_pairs
            if tid in pair
            for peer in pair
            if peer != tid and peer in lane_of
        }
        # Prefer dependency co-location (same lane as a dependency) when free.
        preferred: list[int] = []
        task_map = {str(t["task_id"]): t for t in material}
        for dep in task_map.get(tid, {}).get("depends_on") or []:
            dep_id = str(dep).strip()
            if dep_id in lane_of:
                preferred.append(lane_of[dep_id])
        chosen = None
        for candidate in preferred + list(range(max_lanes)):
            if candidate not in forbidden and 0 <= candidate < max_lanes:
                chosen = candidate
                break
        if chosen is None:
            # Capacity exceeded with conflicts: extend beyond max_lanes rather
            # than co-schedule (fail-safe parallelism).
            chosen = max_lanes
            while chosen in forbidden:
                chosen += 1
        lane_of[tid] = chosen

    # Load-balance into the requested max_lanes without co-scheduling conflicts.
    # Prefer spreading from overloaded lanes into empty/lighter lanes.
    for _ in range(len(lane_of) * 2):
        loads = defaultdict(int)
        for lane in lane_of.values():
            if 0 <= lane < max_lanes:
                loads[lane] += 1
        if not loads:
            break
        heavy = max(range(max_lanes), key=lambda i: loads.get(i, 0))
        light = min(range(max_lanes), key=lambda i: loads.get(i, 0))
        if loads.get(heavy, 0) - loads.get(light, 0) <= 1:
            break
        moved = False
        for tid in sorted(lane_of, key=lambda t: (-degree[t], t)):
            if lane_of[tid] != heavy:
                continue
            forbidden = {
                lane_of[peer]
                for pair in blocking_pairs
                if tid in pair
                for peer in pair
                if peer != tid and peer in lane_of
            }
            if light not in forbidden:
                lane_of[tid] = light
                moved = True
                break
        if not moved:
            break

    used = max((lane for lane in lane_of.values() if lane >= 0), default=-1) + 1
    slice_count = max(max_lanes, used)
    lane_slices: dict[str, list[str]] = {str(i): [] for i in range(slice_count)}
    for tid in sorted(lane_of):
        lane_slices[str(lane_of[tid])].append(tid)
    for key in lane_slices:
        lane_slices[key].sort()

    dep_edges: list[tuple[str, str]] = []
    for task in material:
        tid = str(task["task_id"])
        for dep in task.get("depends_on") or []:
            dep_id = str(dep).strip()
            if dep_id:
                dep_edges.append((dep_id, tid))

    rationale = [
        f"schedulable_tasks={len(material)}",
        f"ast_records={len(ast_records)}",
        f"blocking_edges={sum(1 for e in graph.edges if e.blocks_concurrency)}",
        f"requested_max_lanes={max_lanes}",
        f"emitted_lanes={slice_count}",
        f"independent_width={graph.independent_width}",
        "source=materialize_task_conflict_graph+dependency_colocation+load_balance",
    ]
    if overrides:
        rationale.append(f"allow_concurrent_overrides={len(overrides)}")

    return ParallelLanePlan(
        lane_slices=lane_slices,
        max_lanes=slice_count,
        task_count=len(material),
        independent_width=int(graph.independent_width),
        blocking_edge_count=sum(1 for e in graph.edges if e.blocks_concurrency),
        dependency_edges=sorted(set(dep_edges)),
        rationale=rationale,
        task_colors=lane_of,
    )


def plan_parallel_lanes_from_todo_path(
    todo_path: Path | str,
    *,
    max_lanes: int = 6,
    repo_root: Path | str | None = None,
    use_ast: bool = True,
) -> ParallelLanePlan:
    """Load a markdown todo board and plan parallel lanes."""

    path = Path(todo_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    tasks = parse_markdown_todo_tasks(text)
    root = repo_root
    if root is None:
        # Heuristic: repo root is parent of docs/ when todo lives under docs/
        for parent in path.parents:
            if (parent / "ipfs_datasets_py").is_dir() or (
                parent / "ipfs_accelerate_py"
            ).is_dir():
                root = parent
                break
    return plan_parallel_lanes(
        tasks,
        max_lanes=max_lanes,
        repo_root=root,
        use_ast=use_ast,
    )


def apply_lane_slices_to_config(
    config_path: Path | str,
    plan: ParallelLanePlan,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Merge planned lane_slices into a scheduler config JSON object."""

    path = Path(config_path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["lane_slices"] = plan.lane_slices
    config["max_lanes"] = plan.max_lanes
    config["shard_count"] = plan.max_lanes
    config["parallel_lane_plan"] = {
        "schema": plan.schema,
        "independent_width": plan.independent_width,
        "blocking_edge_count": plan.blocking_edge_count,
        "rationale": plan.rationale,
        "task_count": plan.task_count,
    }
    if write:
        path.write_text(
            json.dumps(config, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan conflict-free parallel lanes from a markdown todo board"
    )
    parser.add_argument("--todo-path", required=True)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--max-lanes", type=int, default=6)
    parser.add_argument("--no-ast", action="store_true")
    parser.add_argument(
        "--apply-config",
        default="",
        help="Optional scheduler config path to update lane_slices",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write --apply-config in place (default: dry print only)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan = plan_parallel_lanes_from_todo_path(
        args.todo_path,
        max_lanes=int(args.max_lanes),
        repo_root=args.repo_root or None,
        use_ast=not bool(args.no_ast),
    )
    if args.apply_config:
        apply_lane_slices_to_config(
            args.apply_config, plan, write=bool(args.write)
        )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"parallel lanes: tasks={plan.task_count} "
            f"lanes={plan.max_lanes} width={plan.independent_width} "
            f"blocking_edges={plan.blocking_edge_count}"
        )
        for lane, ids in sorted(
            plan.lane_slices.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0
        ):
            print(f"  shard {lane}: {', '.join(ids) or '(empty)'}")
        for line in plan.rationale:
            print(f"  # {line}")
    return 0


__all__ = [
    "PARALLEL_LANE_PLAN_SCHEMA",
    "ParallelLanePlan",
    "apply_lane_slices_to_config",
    "parse_markdown_todo_tasks",
    "plan_parallel_lanes",
    "plan_parallel_lanes_from_todo_path",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
