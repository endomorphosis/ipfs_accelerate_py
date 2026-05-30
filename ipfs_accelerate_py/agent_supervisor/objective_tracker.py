"""Objective tracking document helpers for autonomous agent supervisors."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .objective_graph import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    ObjectiveFinding,
    ObjectiveGoal,
    evidence_index,
    goal_graph,
    normalize_field_key,
    objective_heap_schedule,
    parse_goal_heap,
    safe_bundle_key,
    split_terms,
    utc_now,
)


DEFAULT_ULTIMATE_GOAL = (
    "Make this repository satisfy its stated objective with verifiable code, tests, docs, and runtime evidence."
)
DEFAULT_ROOT_EVIDENCE = (
    "objective goal graph",
    "bundle-local todo shards",
    "AST evidence dataset",
    "embedding evidence scan",
    "LLM merge conflict resolver",
)
DEFAULT_GOAL_PREFIX = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_GOAL_PREFIX", "OBJ-G")
DEFAULT_TRACKING_DOCUMENT_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_DOCUMENT_TITLE", "Objective Heap")
DEFAULT_ROOT_GOAL_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_ROOT_TITLE", "Objective outcome")


@dataclass(frozen=True)
class ObjectiveTrackingResult:
    """Summary of objective tracking document mutations."""

    objective_path: Path
    created: bool
    appended_goal_ids: list[str]
    graph_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        if self.graph_path is not None:
            payload["graph_path"] = str(self.graph_path)
        return payload


@dataclass(frozen=True)
class ObjectiveCompletionResult:
    """Summary of objective goals reconciled from repository evidence."""

    objective_path: Path
    completed_goal_ids: list[str]
    active_goal_count: int
    completed_goal_count: int
    completion_evidence: dict[str, dict[str, list[str]]]
    validation_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        return payload


def fibonacci_number(index: int) -> int:
    """Return a one-indexed Fibonacci-ish priority bucket."""

    if index <= 1:
        return 1
    left, right = 1, 1
    for _ in range(2, index + 1):
        left, right = right, left + right
    return right


def fibonacci_priority(depth: int, sibling_index: int = 0) -> int:
    """Return a stable integer priority for a goal depth and sibling order."""

    return fibonacci_number(max(1, depth + 2)) * 1000 + max(0, sibling_index)


def infer_goal_prefix(goals: Sequence[ObjectiveGoal], *, fallback: str = DEFAULT_GOAL_PREFIX) -> str:
    """Infer a numeric goal-id prefix from existing goals."""

    prefixes: dict[str, int] = {}
    for goal in goals:
        match = re.match(r"^(.+?)(\d+)$", goal.goal_id)
        if not match:
            continue
        prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
    if not prefixes:
        return fallback
    return sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def next_goal_id(goals: Sequence[ObjectiveGoal], *, prefix: str | None = None) -> str:
    prefix = prefix or infer_goal_prefix(goals)
    highest = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for goal in goals:
        match = pattern.match(goal.goal_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def render_goal_block(*, goal_id: str, title: str, fields: dict[str, str]) -> str:
    rows = [f"## {goal_id} {title.strip()}", ""]
    for key, value in fields.items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows).rstrip() + "\n"


def rewrite_goal_fields(text: str, updates: Mapping[str, Mapping[str, str]]) -> str:
    """Rewrite selected markdown goal fields without reparsing the whole file."""

    if not updates:
        return text

    lines = text.splitlines()
    rewritten: list[str] = []
    block: list[str] = []
    current_goal_id = ""
    header_pattern = re.compile(r"^##\s+(\S+)\s+.+?\s*$")

    def flush() -> None:
        nonlocal block, current_goal_id
        if not block:
            return
        goal_updates = updates.get(current_goal_id)
        if not goal_updates:
            rewritten.extend(block)
            block = []
            return

        normalized_updates = {normalize_field_key(key): (key, value) for key, value in goal_updates.items()}
        seen_keys: set[str] = set()
        output: list[str] = []
        last_field_index = 0
        for line in block:
            if line.startswith("- ") and ":" in line:
                key, _value = line[2:].split(":", 1)
                normalized = normalize_field_key(key)
                if normalized in normalized_updates:
                    output.append(f"- {normalized_updates[normalized][0]}: {normalized_updates[normalized][1]}")
                    seen_keys.add(normalized)
                else:
                    output.append(line)
                last_field_index = len(output)
            else:
                output.append(line)

        missing_lines = [
            f"- {key}: {value}"
            for normalized, (key, value) in normalized_updates.items()
            if normalized not in seen_keys
        ]
        if missing_lines:
            insert_at = max(1, last_field_index)
            output[insert_at:insert_at] = missing_lines
        rewritten.extend(output)
        block = []

    for line in lines:
        header = header_pattern.match(line)
        if header:
            flush()
            current_goal_id = header.group(1)
            block = [line]
            continue
        if block:
            block.append(line)
        else:
            rewritten.append(line)
    flush()
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rewritten) + suffix


def completion_evidence_summary(evidence: Mapping[str, Sequence[str]]) -> str:
    parts: list[str] = []
    for term, paths in evidence.items():
        compact_paths = ", ".join(str(path) for path in list(paths)[:3])
        if compact_paths:
            parts.append(f"{term} => {compact_paths}")
    return "; ".join(parts)


def split_validation_commands(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def run_goal_validation(
    *,
    repo_root: Path,
    goal: ObjectiveGoal,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Run a goal's validation commands before marking it completed."""

    commands = split_validation_commands(str(goal.fields.get("validation") or ""))
    if not commands:
        return {
            "attempted": False,
            "passed": True,
            "returncode": 0,
            "results": [],
            "reason": "no_commands",
        }
    results: list[dict[str, Any]] = []
    for command in commands:
        started_at = utc_now()
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=repo_root,
                text=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "returncode": 124,
                "timed_out": True,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
            }
            results.append(result)
            return {
                "attempted": True,
                "passed": False,
                "returncode": 124,
                "results": results,
                "failed_command": command,
                "error": "timeout",
            }
        result = {
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        results.append(result)
        if completed.returncode != 0:
            return {
                "attempted": True,
                "passed": False,
                "returncode": completed.returncode,
                "results": results,
                "failed_command": command,
            }
    return {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "results": results,
    }


def reconcile_objective_goal_completion(
    *,
    repo_root: Path,
    objective_path: Path,
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
) -> ObjectiveCompletionResult:
    """Mark active goals complete once all required evidence is present."""

    if not objective_path.exists():
        return ObjectiveCompletionResult(
            objective_path=objective_path,
            completed_goal_ids=[],
            active_goal_count=0,
            completed_goal_count=0,
            completion_evidence={},
            validation_results={},
        )

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    active_goals = [goal for goal in goals if goal.status in {"active", "todo", "open"}]
    terms: list[str] = []
    for goal in active_goals:
        terms.extend(goal.required_evidence)
    evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=terms,
        embedding_min_score=embedding_min_score,
    )

    updates: dict[str, dict[str, str]] = {}
    completed_goal_ids: list[str] = []
    completion_evidence: dict[str, dict[str, list[str]]] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    completed_at = utc_now()
    for goal in active_goals:
        required = goal.required_evidence
        if not required or any(not evidence.get(term) for term in required):
            continue
        validation_result = run_goal_validation(repo_root=repo_root, goal=goal)
        validation_results[goal.goal_id] = validation_result
        if not validation_result.get("passed", False):
            continue
        goal_evidence = {term: list(evidence.get(term, [])) for term in required}
        completed_goal_ids.append(goal.goal_id)
        completion_evidence[goal.goal_id] = goal_evidence
        updates[goal.goal_id] = {
            "Status": "completed",
            "Completed at": completed_at,
            "Completion evidence": completion_evidence_summary(goal_evidence),
            "Completion validation": str(validation_result.get("returncode", 0)),
        }

    if completed_goal_ids:
        objective_path.write_text(rewrite_goal_fields(text, updates), encoding="utf-8")
        goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))

    return ObjectiveCompletionResult(
        objective_path=objective_path,
        completed_goal_ids=completed_goal_ids,
        active_goal_count=sum(1 for goal in goals if goal.status in {"active", "todo", "open"}),
        completed_goal_count=sum(1 for goal in goals if goal.status == "completed"),
        completion_evidence=completion_evidence,
        validation_results=validation_results,
    )


def ensure_objective_tracking_document(
    objective_path: Path,
    *,
    ultimate_goal: str = DEFAULT_ULTIMATE_GOAL,
    root_evidence: Sequence[str] = DEFAULT_ROOT_EVIDENCE,
    root_goal_id: str | None = None,
    goal_prefix: str = DEFAULT_GOAL_PREFIX,
    document_title: str = DEFAULT_TRACKING_DOCUMENT_TITLE,
    root_goal_title: str = DEFAULT_ROOT_GOAL_TITLE,
) -> ObjectiveTrackingResult:
    """Create the objective tracking document if it does not exist."""

    if objective_path.exists():
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    root_goal_id = root_goal_id or f"{goal_prefix}000"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {document_title}",
            "",
            "This document is the supervisor planning state. It is intentionally separate from markdown todo",
            "boards: todos represent executable work, while this heap represents the objective graph used to",
            "decide what work should exist.",
            "",
            render_goal_block(
                goal_id=root_goal_id,
                title=root_goal_title,
                fields={
                    "Status": "active",
                    "Parent": "",
                    "Fib priority": str(fibonacci_priority(0)),
                    "Track": "ops",
                    "Priority": "P0",
                    "Bundle": "objective/ops/root",
                    "Goal": ultimate_goal,
                    "Evidence": ", ".join(root_evidence),
                    "Outputs": "ipfs_accelerate_py/agent_supervisor, docs",
                    "Validation": f"test -f {objective_path.as_posix()}",
                    "Refinement depth": "0",
                    "Conflict policy": "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts",
                    "Gap task": "Refine this root objective into concrete child goals with code, tests, docs, and runtime evidence.",
                },
            ).rstrip(),
            "",
        ]
    )
    objective_path.write_text(text, encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=True, appended_goal_ids=[root_goal_id])


def discover_submodule_paths(repo_root: Path) -> list[str]:
    """Return repo-relative Git submodule paths from .gitmodules."""

    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        return []
    paths: list[str] = []
    for line in gitmodules.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path") or "=" not in stripped:
            continue
        _key, value = stripped.split("=", 1)
        path = value.strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def interoperability_pairs(submodules: Sequence[str], *, focus: Sequence[str] = ()) -> list[tuple[str, str]]:
    paths = [path for path in dict.fromkeys(str(item).strip() for item in submodules) if path]
    focus_paths = [path for path in dict.fromkeys(str(item).strip() for item in focus) if path]
    pairs: list[tuple[str, str]] = []
    if focus_paths:
        for left in focus_paths:
            if left not in paths:
                continue
            for right in paths:
                if right == left:
                    continue
                pair = tuple(sorted((left, right)))
                if pair not in pairs:
                    pairs.append(pair)
        return pairs
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1 :]:
            pairs.append((left, right))
    return pairs


def append_interoperability_goals(
    objective_path: Path,
    *,
    repo_root: Path,
    focus: Sequence[str] = (),
    max_goals: int = 12,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Seed graph goals for cross-submodule integration and interoperability tests."""

    if not objective_path.exists() or max_goals <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    submodules = discover_submodule_paths(repo_root)
    pairs = interoperability_pairs(submodules, focus=focus)
    if not pairs:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    existing_pairs = {
        " ".join(split_terms(str(goal.fields.get("interoperability_pair") or ""))).lower()
        for goal in goals
        if str(goal.fields.get("interoperability_pair") or "").strip()
    }
    graph = goal_graph(goals)
    root_goal_id = sorted(graph.get("roots") or [goals[0].goal_id if goals else ""])[0]
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for left, right in pairs:
        pair_key = " ".join(sorted((left, right))).lower()
        if pair_key in existing_pairs:
            continue
        goal_id = allocate_goal_id()
        safe_left = safe_bundle_key(left).replace("-", "_")
        safe_right = safe_bundle_key(right).replace("-", "_")
        test_path = f"tests/integration/test_{safe_left}_{safe_right}_interop.py"
        doc_path = f"docs/integration/{safe_left}-{safe_right}.md"
        fields = {
            "Status": "active",
            "Parent": root_goal_id,
            "Fib priority": str(fibonacci_priority(1, len(appended_goal_ids))),
            "Track": "interoperability",
            "Priority": "P1",
            "Bundle": f"objective/interoperability/{safe_left}-{safe_right}",
            "Goal kind": "interoperability",
            "Interoperability pair": f"{left}, {right}",
            "Submodules": f"{left}, {right}",
            "Goal": (
                f"Prove `{left}` interoperates with `{right}` through importable contracts, "
                "interface descriptors, runtime handoff behavior, and integration tests."
            ),
            "Evidence": f"{test_path}, {doc_path}, interface contract {left} {right}",
            "Outputs": f"{test_path}, {doc_path}, {left}, {right}",
            "Validation": "python -m pytest tests/integration -q",
            "Refinement depth": "1",
            "Embedding query": f"{left} {right} interoperability integration test interface descriptor",
            "AST query": f"{left}, {right}, interface contract, integration test",
            "Parallel lane": f"objective/interoperability/{safe_left}-{safe_right}",
            "Conflict policy": "keep pair-specific integration edits isolated; use the LLM merge resolver for conflicts",
            "Gap task": (
                f"Create one larger integration work item proving `{left}` and `{right}` can be used together, "
                "including a test, a contract note, and any adapter code needed by the objective."
            ),
        }
        appended_blocks.append(render_goal_block(goal_id=goal_id, title=f"Interoperate {left} with {right}", fields=fields))
        appended_goal_ids.append(goal_id)
        existing_pairs.add(pair_key)
        if len(appended_goal_ids) >= max_goals:
            break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def existing_refinement_keys(goals: Sequence[ObjectiveGoal]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for goal in goals:
        for parent in goal.parent_goal_ids:
            for evidence in goal.required_evidence:
                keys.add((parent, normalize_evidence_key(evidence)))
    return keys


def normalize_evidence_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def refinement_title(parent_title: str, evidence: str) -> str:
    compact = " ".join(evidence.strip().split())
    if len(compact) > 72:
        compact = compact[:69].rstrip() + "..."
    return f"Prove {compact} for {parent_title}"


def refinement_fields(finding: ObjectiveFinding, *, evidence: str, depth: int, sibling_index: int) -> dict[str, str]:
    outputs = ", ".join(finding.outputs) if finding.outputs else "ipfs_accelerate_py/agent_supervisor, docs, tests"
    return {
        "Status": "active",
        "Parent": finding.goal_id,
        "Fib priority": str(fibonacci_priority(depth, sibling_index)),
        "Track": finding.track,
        "Priority": finding.priority,
        "Bundle": finding.bundle_key,
        "Goal": f"Create concrete implementation, tests, docs, or interface descriptors proving `{evidence}`.",
        "Evidence": evidence,
        "Outputs": outputs,
        "Validation": finding.validation,
        "Refinement depth": str(depth),
        "Embedding query": evidence,
        "AST query": evidence,
        "Parallel lane": finding.parallel_lane,
        "Conflict policy": finding.conflict_policy,
        "Gap task": f"Close the missing objective evidence `{evidence}` with a narrow, verifiable change.",
    }


def append_refinement_goals(
    objective_path: Path,
    findings: Sequence[ObjectiveFinding],
    *,
    max_children_per_finding: int = 3,
    max_depth: int = 4,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Append child goals for missing evidence terms that are still too broad."""

    if not objective_path.exists() or max_children_per_finding <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    graph = goal_graph(goals)
    refinement_keys = existing_refinement_keys(goals)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for finding in findings:
        parent_depth = int(graph.get("depths", {}).get(finding.goal_id, finding.graph_depth))
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue
        created_for_parent = 0
        for evidence in finding.missing_evidence:
            key = (finding.goal_id, normalize_evidence_key(evidence))
            if key in refinement_keys:
                continue
            goal_id = allocate_goal_id()
            fields = refinement_fields(
                finding,
                evidence=evidence,
                depth=child_depth,
                sibling_index=created_for_parent,
            )
            appended_blocks.append(
                render_goal_block(
                    goal_id=goal_id,
                    title=refinement_title(finding.title, evidence),
                    fields=fields,
                )
            )
            appended_goal_ids.append(goal_id)
            refinement_keys.add(key)
            created_for_parent += 1
            if created_for_parent >= max_children_per_finding:
                break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def thought_node_id(kind: str, *parts: str) -> str:
    seed = "\0".join(str(part) for part in parts)
    digest = sha1(seed.encode("utf-8")).hexdigest()[:12]
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "thought"
    return f"{safe_kind}:{digest}"


def build_objective_thought_graph(goals: Sequence[ObjectiveGoal]) -> dict[str, Any]:
    """Build typed thought nodes from objective goals for integration planning."""

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, **payload: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, **payload})

    def add_edge(source: str, target: str, kind: str) -> None:
        edges.append({"from": source, "to": target, "kind": kind})

    for goal in goals:
        goal_node = f"goal:{goal.goal_id}"
        outputs = split_terms(str(goal.fields.get("outputs") or ""))
        validation = str(goal.fields.get("validation") or "").strip()
        interop_pair = split_terms(str(goal.fields.get("interoperability_pair") or ""))
        submodules = split_terms(str(goal.fields.get("submodules") or ""))
        add_node(
            goal_node,
            kind="goal",
            goal_id=goal.goal_id,
            title=goal.title,
            status=goal.status,
            track=str(goal.fields.get("track") or "ops"),
            priority=str(goal.fields.get("priority") or "P2"),
            thought=(
                str(goal.fields.get("goal") or goal.title)
                or "Decide what implementation evidence proves this objective."
            ),
        )
        for term in goal.required_evidence:
            evidence_node = thought_node_id("evidence", goal.goal_id, term)
            add_node(
                evidence_node,
                kind="evidence_requirement",
                goal_id=goal.goal_id,
                term=term,
                thought=f"Find or create repository evidence proving `{term}`.",
            )
            add_edge(goal_node, evidence_node, "requires_evidence")
        if validation:
            validation_node = thought_node_id("validation", goal.goal_id, validation)
            add_node(
                validation_node,
                kind="validation_strategy",
                goal_id=goal.goal_id,
                command=validation,
                thought="Run this validation before closing the goal.",
            )
            add_edge(goal_node, validation_node, "validated_by")
        for output in outputs:
            surface_node = thought_node_id("code_surface", goal.goal_id, output)
            add_node(
                surface_node,
                kind="code_surface",
                goal_id=goal.goal_id,
                path=output,
                thought=f"Inspect or modify `{output}` as part of this objective.",
            )
            add_edge(goal_node, surface_node, "touches_surface")
        if interop_pair or submodules:
            pair_values = interop_pair or submodules
            interop_node = thought_node_id("interoperability_pair", goal.goal_id, ",".join(pair_values))
            add_node(
                interop_node,
                kind="interoperability_pair",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Prove these components interoperate through contracts and tests.",
            )
            add_edge(goal_node, interop_node, "targets_interoperability")
            test_node = thought_node_id("test_strategy", goal.goal_id, ",".join(pair_values))
            add_node(
                test_node,
                kind="test_strategy",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Write or update integration tests that exercise the shared runtime boundary.",
            )
            add_edge(interop_node, test_node, "needs_test_strategy")

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_thought_graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": edges,
    }


def write_objective_graph_artifact(
    *,
    objective_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    """Write a JSON graph artifact for the current objective heap."""

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8")) if objective_path.exists() else []
    graph = goal_graph(goals)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_graph",
        "generated_at": utc_now(),
        "objective_path": str(objective_path),
        "goal_count": len(goals),
        "active_goal_count": sum(1 for goal in goals if goal.status in {"active", "todo", "open"}),
        "completed_goal_count": sum(1 for goal in goals if goal.status == "completed"),
        "heap_schedule": [record.to_dict() for record in objective_heap_schedule(goals)],
        "thought_graph": build_objective_thought_graph(goals),
        "goals": [
            {
                "goal_id": goal.goal_id,
                "title": goal.title,
                "status": goal.status,
                "fib_priority": goal.priority[0],
                "parents": goal.parent_goal_ids,
                "evidence": goal.required_evidence,
                "track": goal.fields.get("track", "ops"),
                "bundle": goal.fields.get("bundle", ""),
                "refinement_depth": goal.fields.get("refinement_depth", str(graph["depths"].get(goal.goal_id, 0))),
            }
            for goal in sorted(goals, key=lambda item: item.priority)
        ],
        "graph": graph,
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_root_evidence(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(split_terms(value))
    return terms or list(DEFAULT_ROOT_EVIDENCE)
