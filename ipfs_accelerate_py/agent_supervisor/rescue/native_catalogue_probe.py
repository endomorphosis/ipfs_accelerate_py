"""Observe a bound native catalogue without activating tasks or owning its store.

The catalogue is read afresh in a bounded, read-only DuckDB subprocess. No native
repository/Portal constructor, migration, repair, checkpoint load or provider is
called. Catalogue status is distinct from executor health and completion authority.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any

SCHEMA = "agent-supervisor/native-catalogue-probe@1"
DIAGNOSTIC_SCHEMA = "native-catalogue-diagnostic/v1"
WORKER_ERROR_SCHEMA = "native-catalogue-worker-error/v1"
POPULATION_SCHEMA = "legal-ir-native-task-population/v1"
LINEAGE_POPULATION_SCHEMA = "legal-ir-native-task-population/v2"
BOUNDARY_POPULATION_SCHEMA = "legal-ir-native-catalogue-boundary/v1"
GRAPH_DEPENDENCY_BOUNDARY_SCHEMA = "legal-ir-native-dependency-boundary/v2"
AUTHORED_BOUNDARY_POPULATION_SHA256 = "ae902ebf03aee807ea9518a4b803e9f4a38bd36c3767895abe9c818048f85b89"
BOUNDARY_SEMANTIC_FIELDS = (
    "completion", "resource_stage", "parent_task_cid", "parent_task_alias",
    "obligation", "obligation_kind", "spawn_decision", "resolution_subgoal",
    "resolution_work", "missing_record_subgoal", "execution_packet_required",
    "admitted", "formalized", "gold_or_teacher_admission_granted",
    "constitution_roundtrip_ok", "completion_authority")
GRAPH_RESOLUTION_MARKERS = ("resolution_subgoal", "resolution_work", "missing_record_subgoal")
MAX_POPULATION_BYTES = 2 * 1024 * 1024
MAX_GRAPH_BOUNDARY_BYTES = 4 * 1024 * 1024
MAX_GRAPH_EDGES = 10_000
MAX_GRAPH_DEPTH = 16
MAX_REPORT_BYTES = 512 * 1024
MAX_BODY_BYTES = 262_144
MAX_TASKS = 1_000
MAX_PLAN_ROOTS = 64
MAX_RUNTIME_ROWS = 4_096
READY_STATUSES = frozenset({"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}\Z")


class InvalidCatalogue(ValueError):
    """Closed diagnostic code; never executable report text."""


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidCatalogue("duplicate_json_key")
        result[key] = value
    return result


def _decode(raw):
    def nonfinite(_):
        raise InvalidCatalogue("nonfinite_json")
    return json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=nonfinite)


def _identifier(value):
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise InvalidCatalogue("invalid_binding")
    return value


def _file_bytes(path: Path, maximum: int):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise InvalidCatalogue("invalid_input_file")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > maximum:
            raise InvalidCatalogue("input_too_large")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _file_metadata(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise InvalidCatalogue("invalid_database_file")
    return {"device": info.st_dev, "inode": info.st_ino,
            "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


_PLAN_BODY_BINDINGS = ("repository_tree_id", "continuation_of", "continued_by",
                       "revision_cid", "delta_cid", "idempotency_key")


class _LineagePlan(dict):
    """Internal v2 provenance; a caller's JSON key cannot select this profile."""


class _GraphBoundaryPlan(dict):
    """Closed observational task graph, with the authored scope unchanged."""


def _bound_boundary_document(reference, *, maximum=MAX_POPULATION_BYTES):
    if (type(reference) is not dict or set(reference) != {"path", "bytes", "sha256"}
            or type(reference["path"]) is not str
            or type(reference["bytes"]) is not int
            or not 0 < reference["bytes"] <= maximum
            or type(reference["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None):
        raise InvalidCatalogue("boundary_descriptor_invalid")
    path = Path(reference["path"])
    if (not path.is_absolute() or ".." in path.parts
            or any(part.is_symlink() for part in (path, *path.parents))):
        raise InvalidCatalogue("boundary_descriptor_invalid")
    raw = _file_bytes(path, maximum)
    if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise InvalidCatalogue("boundary_descriptor_changed")
    document = _decode(raw)
    if type(document) is not dict:
        raise InvalidCatalogue("boundary_document_invalid")
    return document


def _graph_literal(value):
    """Canonical JSON preserves literal types, including bool versus number."""
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_GRAPH_DEPTH:
            raise InvalidCatalogue("graph_literal_bound")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise InvalidCatalogue("graph_literal_invalid")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) not in (type(None), str, bool, int, float):
            raise InvalidCatalogue("graph_literal_invalid")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError):
        raise InvalidCatalogue("graph_literal_invalid") from None
    if len(raw) > MAX_BODY_BYTES:
        raise InvalidCatalogue("graph_literal_bound")
    return raw


def _graph_identity(value):
    # Foreign prerequisites may have an absent native association. None and ''
    # are exact observed values, never an inherited goal, plan or objective.
    if value is not None and value != "":
        _identifier(value)


def _graph_resolution_projection(projection):
    required = {"completion", "resource_stage", "parent_task_cid", "parent_task_alias",
                "obligation", "obligation_kind", "admitted", "formalized",
                "gold_or_teacher_admission_granted", "constitution_roundtrip_ok",
                "completion_authority"}
    if (type(projection) is not dict
            or not required <= set(projection) <= set(BOUNDARY_SEMANTIC_FIELDS)
            or projection["completion"] != "manual" or projection["resource_stage"] != "analysis"
            or any(type(projection[key]) is not str or not projection[key] for key in
                   ("obligation", "obligation_kind"))
            or not any(projection.get(key) is True for key in
                       ("resolution_subgoal", "resolution_work", "missing_record_subgoal"))
            or any(projection[key] is not False for key in
                   ("admitted", "formalized", "gold_or_teacher_admission_granted",
                    "constitution_roundtrip_ok", "completion_authority"))):
        raise InvalidCatalogue("graph_semantics_invalid")
    for key in ("parent_task_cid", "parent_task_alias"):
        _identifier(projection[key])
    for key in ("resolution_subgoal", "resolution_work", "missing_record_subgoal",
                "execution_packet_required"):
        if key in projection and type(projection[key]) is not bool:
            raise InvalidCatalogue("graph_semantics_invalid")
    if "spawn_decision" in projection and (type(projection["spawn_decision"]) is not str
                                           or not projection["spawn_decision"]):
        raise InvalidCatalogue("graph_semantics_invalid")
    _graph_literal(projection)


def _graph_resolution_marker(body):
    # Presence, including False, cannot downgrade a resolution record into an
    # identity-only prerequisite. Malformed marked records fail qualification.
    return any(key in body for key in GRAPH_RESOLUTION_MARKERS)


def _graph_topology(selected, roots, observed):
    """Authenticate the whole relation with iterative reachability and DAG checks."""
    core = set(selected)
    scope = core | set(observed)
    if len(scope) > MAX_TASKS:
        raise InvalidCatalogue("graph_node_bound")
    authored = {(cid, dep, "depends_on") for cid, row in selected.items()
                for dep in row.get("dependencies", [])}
    edges = list(authored) + list(roots)
    edges.extend((cid, row["dependency_task_cid"], row["kind"])
                 for cid, task in observed.items() for row in task["dependencies"])
    if len(edges) > MAX_GRAPH_EDGES:
        raise InvalidCatalogue("graph_edge_bound")
    if len(set(edges)) != len(edges) or len({(a, b) for a, b, _ in edges}) != len(edges):
        raise InvalidCatalogue("graph_duplicate_edge")
    adjacency = {cid: [] for cid in scope}
    incoming = dict.fromkeys(scope, 0)
    for source, target, _ in edges:
        if source not in scope or target not in scope:
            raise InvalidCatalogue("graph_not_closed")
        adjacency[source].append(target)
        incoming[target] += 1
    # Minimum outward BFS distance starts at ALL authored tasks. Long paths
    # within the authored DAG do not consume the bounded discovery depth.
    distances = dict.fromkeys(core, 0)
    pending = collections.deque(sorted(core))
    while pending:
        source = pending.popleft()
        for target in adjacency[source]:
            if target not in distances:
                depth = distances[source] + 1
                if depth > MAX_GRAPH_DEPTH:
                    raise InvalidCatalogue("graph_depth_bound")
                distances[target] = depth
                pending.append(target)
    if set(distances) != scope:
        raise InvalidCatalogue("graph_unreachable_node")
    pending = collections.deque(cid for cid in scope if incoming[cid] == 0)
    visited = 0
    while pending:
        source = pending.popleft()
        visited += 1
        for target in adjacency[source]:
            incoming[target] -= 1
            if incoming[target] == 0:
                pending.append(target)
    if visited != len(scope):
        raise InvalidCatalogue("graph_cycle")
    return {"node_count": len(scope), "edge_count": len(edges),
            "maximum_outward_depth": max(distances.values(), default=0)}


def _graph_resolution_lineage(selected, roots, observed):
    """Verify declared resolution origins; do not infer prerequisite lineage."""
    scope = dict(selected)
    scope.update(observed)
    edges = set(roots)
    edges.update((cid, row["dependency_task_cid"], row["kind"])
                 for cid, node in observed.items() for row in node["dependencies"])
    for cid, node in observed.items():
        binding = node["body_binding"]
        if binding["profile"] != "resolution-obligation/v1":
            continue
        projection = binding["projection"]
        parent = scope.get(projection["parent_task_cid"])
        if (parent is None or parent["task_alias"] != projection["parent_task_alias"]
                or any(type(node[key]) is not type(parent[key]) or node[key] != parent[key]
                       for key in ("goal_cid", "plan_cid", "objective_id"))
                or not cid.startswith(parent["task_cid"] + "--")
                or not node["task_alias"].startswith(parent["task_alias"] + "--")
                or (parent["task_cid"], cid, "depends_on") not in edges):
            raise InvalidCatalogue("graph_resolution_lineage_invalid")


def _graph_dependency_boundary(document, selected, goals, arguments, authored_ref):
    fields = {"schema", "board_namespace", "plan_root_cid", "repository_tree_id",
              "authored_population_sha256", "authored_task_count", "authored_goal_count",
              "authored_dependency_count", "root_dependencies", "observed_tasks", "completion_authority"}
    if (set(document) != fields or document["schema"] != GRAPH_DEPENDENCY_BOUNDARY_SCHEMA
            or any(document[key] != arguments[key] for key in
                   ("board_namespace", "plan_root_cid", "repository_tree_id"))
            or document["authored_population_sha256"] != authored_ref["sha256"]
            or authored_ref["sha256"] != AUTHORED_BOUNDARY_POPULATION_SHA256
            or document["completion_authority"] is not False):
        raise InvalidCatalogue("graph_boundary_invalid")
    authored = {(cid, dep) for cid, task in selected.items() for dep in task.get("dependencies", [])}
    for key, actual, required in (("authored_task_count", len(selected), 64),
                                  ("authored_goal_count", len(goals), 34),
                                  ("authored_dependency_count", len(authored), 115)):
        if type(document[key]) is not int or document[key] != actual or actual != required:
            raise InvalidCatalogue("graph_authored_population_changed")
    nodes, root_rows = document["observed_tasks"], document["root_dependencies"]
    if (type(nodes) is not list or len(nodes) + len(selected) > MAX_TASKS
            or type(root_rows) is not list or len(root_rows) > MAX_GRAPH_EDGES):
        raise InvalidCatalogue("graph_bounds")
    fields = {"task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "minimum_revision",
              "namespace_binding", "observation_role", "body_binding", "dependencies"}
    observed = {}
    for node in nodes:
        if type(node) is not dict or set(node) != fields:
            raise InvalidCatalogue("graph_node_invalid")
        cid = _identifier(node["task_cid"])
        for key in ("task_alias", "goal_cid"):
            _identifier(node[key])
        for key in ("plan_cid", "objective_id"):
            _graph_identity(node[key])
        if (cid in selected or cid in observed or type(node["minimum_revision"]) is not int
                or node["minimum_revision"] < 1):
            raise InvalidCatalogue("graph_identity_invalid")
        namespace = node["namespace_binding"]
        if (type(namespace) is not dict or set(namespace) != {"present", "value"}
                or type(namespace["present"]) is not bool
                or not namespace["present"] and namespace["value"] is not None):
            raise InvalidCatalogue("graph_namespace_invalid")
        _graph_literal(namespace["value"])
        binding = node["body_binding"]
        if type(binding) is not dict:
            raise InvalidCatalogue("graph_body_binding_invalid")
        if binding.get("profile") == "resolution-obligation/v1":
            if set(binding) != {"profile", "projection"}:
                raise InvalidCatalogue("graph_body_binding_invalid")
            _graph_resolution_projection(binding["projection"])
        elif binding != {"profile": "literal-prerequisite/v1"}:
            raise InvalidCatalogue("graph_body_binding_invalid")
        dependencies = node["dependencies"]
        if type(dependencies) is not list or len(dependencies) > MAX_GRAPH_EDGES:
            raise InvalidCatalogue("graph_edge_bound")
        for row in dependencies:
            if type(row) is not dict or set(row) != {"dependency_task_cid", "kind"}:
                raise InvalidCatalogue("graph_edge_invalid")
            _identifier(row["dependency_task_cid"])
            _identifier(row["kind"])
        observed[cid] = node
    roots = []
    for row in root_rows:
        if type(row) is not dict or set(row) != {"task_cid", "dependency_task_cid", "kind"}:
            raise InvalidCatalogue("graph_root_invalid")
        for key in row:
            _identifier(row[key])
        if row["task_cid"] not in selected or row["dependency_task_cid"] not in observed:
            raise InvalidCatalogue("graph_root_invalid")
        roots.append((row["task_cid"], row["dependency_task_cid"], row["kind"]))
    root_targets = {target for _, target, _ in roots}
    for cid, node in observed.items():
        if node["observation_role"] != ("root_dependency" if cid in root_targets else "transitive_dependency"):
            raise InvalidCatalogue("graph_role_invalid")
    _graph_resolution_lineage(selected, roots, observed)
    return document, _graph_topology(selected, roots, observed)


def _boundary_population(envelope, arguments, raw):
    if set(envelope) != {"schema", "authored_population_ref", "dependency_boundary_ref"}:
        raise InvalidCatalogue("population_boundary_invalid")
    authored_ref = envelope["authored_population_ref"]
    authored = _bound_boundary_document(authored_ref)
    if (authored_ref["sha256"] != AUTHORED_BOUNDARY_POPULATION_SHA256
            or authored.get("schema") != POPULATION_SCHEMA):
        raise InvalidCatalogue("population_boundary_authored_invalid")
    inner_arguments = dict(arguments, population_path=authored_ref["path"],
                           population_sha256=authored_ref["sha256"])
    selected, goals, original_plan = _population(inner_arguments)
    reference = envelope["dependency_boundary_ref"]
    document = _bound_boundary_document(reference, maximum=MAX_GRAPH_BOUNDARY_BYTES)
    if document.get("schema") != GRAPH_DEPENDENCY_BOUNDARY_SCHEMA:
        raise InvalidCatalogue("dependency_boundary_schema_unsupported")
    boundary, topology = _graph_dependency_boundary(document, selected, goals, arguments, authored_ref)
    plan = _GraphBoundaryPlan(original_plan)
    plan.graph_topology = topology
    plan.boundary = boundary
    plan.authored_population_ref = authored_ref
    plan.dependency_boundary_ref = envelope["dependency_boundary_ref"]
    plan.envelope_ref = {"path": arguments["population_path"], "bytes": len(raw),
                         "sha256": arguments["population_sha256"]}
    return selected, goals, plan


def _lineage_population(population, selected, goals, plan_root, tree):
    """Validate an explicitly pinned diagnostic profile, not a revision grant.

    Native steer retains existing tasks' original plan CIDs. Its continuation
    has the same root goal, stores continuation_of, and adds continued_by to the
    predecessor. None in a declared body binding means literal key absence.
    """
    if set(population) != {"schema", "board_namespace", "plan_root_cid",
                            "repository_tree_id", "tasks", "goals", "plans"}:
        raise InvalidCatalogue("population_lineage_invalid")
    task_fields = {"task_cid", "task_alias", "goal_cid", "plan_cid",
                   "objective_id", "dependencies"}
    goal_fields = {"goal_cid", "goal_alias", "objective_id", "parent_goal_cid"}
    if (any(set(task) != task_fields for task in selected.values())
            or any(set(goal) != goal_fields for goal in goals.values())
            or len({task["task_alias"] for task in selected.values()}) != len(selected)
            or len({goal["goal_alias"] for goal in goals.values()}) != len(goals)):
        raise InvalidCatalogue("population_lineage_invalid")
    plans = population["plans"]
    fields = {"plan_cid", "goal_cid", "status", "revision", "body_sha256",
              *_PLAN_BODY_BINDINGS}
    if type(plans) is not list or not 1 <= len(plans) <= MAX_PLAN_ROOTS:
        raise InvalidCatalogue("population_lineage_invalid")
    by_id = {}
    for plan in plans:
        if type(plan) is not dict or set(plan) != fields:
            raise InvalidCatalogue("population_lineage_invalid")
        for key in ("plan_cid", "goal_cid"):
            _identifier(plan[key])
        if (plan["plan_cid"] in by_id or plan["goal_cid"] not in goals
                or type(plan["revision"]) is not int or plan["revision"] < 1
                or type(plan["body_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", plan["body_sha256"]) is None):
            raise InvalidCatalogue("population_lineage_invalid")
        for key in _PLAN_BODY_BINDINGS:
            if plan[key] is not None:
                _identifier(plan[key])
        by_id[plan["plan_cid"]] = plan
    root_goal = plans[0]["goal_cid"]
    for index, plan in enumerate(plans):
        previous = plans[index - 1]["plan_cid"] if index else None
        following = plans[index + 1]["plan_cid"] if index + 1 < len(plans) else None
        if (plan["continuation_of"] != previous or plan["continued_by"] != following
                or plan["goal_cid"] != root_goal
                or plan["status"] != ("continued" if following else "active")
                or index and any(plan[key] is None for key in
                    ("repository_tree_id", "revision_cid", "delta_cid", "idempotency_key"))):
            raise InvalidCatalogue("population_lineage_invalid")
    if (plans[-1]["plan_cid"] != plan_root
            or plans[-1]["repository_tree_id"] != tree
            or any(task["plan_cid"] not in by_id for task in selected.values())):
        raise InvalidCatalogue("population_lineage_invalid")
    # Goal parent rows are exact native fields. Every declared goal must reach
    # the unchanged native root goal; unrelated roots and cycles are refused.
    for goal in goals.values():
        seen = set()
        current = goal["goal_cid"]
        while current != root_goal:
            if current in seen or current not in goals:
                raise InvalidCatalogue("population_lineage_invalid")
            seen.add(current)
            current = goals[current]["parent_goal_cid"]
        if goals[root_goal]["parent_goal_cid"] != "":
            raise InvalidCatalogue("population_lineage_invalid")
    return _LineagePlan(plan_cid=plan_root, goal_cid=root_goal, lineage=plans)


def _read_plan_lineage(connection, expected, expected_goals):
    plans = expected["lineage"]
    by_id = {plan["plan_cid"]: plan for plan in plans}
    # Exact native scope, including unexpected continuations/forks under the
    # same immutable goal. A selected subset cannot hide an undeclared root.
    goal_rows = _bounded_rows(connection,
        "WITH RECURSIVE selected_goals(goal_cid) AS ("
        "SELECT goal_cid FROM goals WHERE goal_cid = ? UNION "
        "SELECT g.goal_cid FROM goals g JOIN selected_goals s "
        "ON g.parent_goal_cid = s.goal_cid) SELECT goal_cid FROM selected_goals",
        [expected["goal_cid"]], maximum=MAX_TASKS)
    if (len(goal_rows) != len(expected_goals)
            or {row[0] for row in goal_rows} != set(expected_goals)):
        raise InvalidCatalogue("native_goal_mismatch")
    goal_ids = sorted(expected_goals)
    rows = _bounded_rows(connection,
        "SELECT plan_cid, goal_cid, status, revision, body_json FROM plans "
        "WHERE goal_cid IN (" + ",".join("?" for _ in goal_ids) + ") ORDER BY plan_cid",
        goal_ids,
        maximum=MAX_PLAN_ROOTS)
    if len(rows) != len(plans):
        raise InvalidCatalogue("native_plan_lineage_mismatch")
    observed = set()
    for cid, goal, status, revision, raw in rows:
        plan = by_id.get(cid)
        if (plan is None or cid in observed
                or (goal, status, revision) != (plan["goal_cid"], plan["status"], plan["revision"])
                or type(revision) is not int or type(raw) is not str
                or len(raw.encode()) > MAX_BODY_BYTES
                or hashlib.sha256(raw.encode()).hexdigest() != plan["body_sha256"]):
            raise InvalidCatalogue("native_plan_lineage_mismatch")
        observed.add(cid)
        body = _decode(raw)
        if type(body) is not dict:
            raise InvalidCatalogue("native_plan_lineage_mismatch")
        for key in _PLAN_BODY_BINDINGS:
            value = plan[key]
            if (key in body if value is None else key not in body or body[key] != value):
                raise InvalidCatalogue("native_plan_lineage_mismatch")


def _population(arguments):
    namespace = _identifier(arguments["board_namespace"])
    plan_root = _identifier(arguments["plan_root_cid"])
    tree = _identifier(arguments["repository_tree_id"])
    digest = arguments["population_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise InvalidCatalogue("invalid_binding")
    raw = _file_bytes(Path(arguments["population_path"]), MAX_POPULATION_BYTES)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise InvalidCatalogue("population_digest_mismatch")
    population = _decode(raw)
    if type(population) is dict and population.get("schema") == BOUNDARY_POPULATION_SCHEMA:
        return _boundary_population(population, arguments, raw)
    lineage = isinstance(population, dict) and population.get("schema") == LINEAGE_POPULATION_SCHEMA
    if (not isinstance(population, dict)
            or population.get("schema") not in (POPULATION_SCHEMA, LINEAGE_POPULATION_SCHEMA)
            or population.get("board_namespace") != namespace
            or population.get("plan_root_cid") != plan_root
            or population.get("repository_tree_id") != tree):
        raise InvalidCatalogue("population_scope_mismatch")
    tasks, goals = population.get("tasks"), population.get("goals")
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS
            or not isinstance(goals, list) or not 1 <= len(goals) <= MAX_TASKS):
        raise InvalidCatalogue("population_bounds")
    selected = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise InvalidCatalogue("population_task_invalid")
        for key in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id"):
            _identifier(task.get(key))
        if task["task_cid"] in selected or not lineage and task["plan_cid"] != plan_root:
            raise InvalidCatalogue("population_task_invalid")
        selected[task["task_cid"]] = task
    goal_map = {}
    for goal in goals:
        if not isinstance(goal, dict):
            raise InvalidCatalogue("population_goal_invalid")
        for key in ("goal_cid", "goal_alias"):
            _identifier(goal.get(key))
        # Native goals may inherit their objective through the bound parent.
        # DEFAULT '' is an exact stored value, not a wildcard for native equality.
        if goal.get("objective_id") != "":
            _identifier(goal.get("objective_id"))
        if goal["goal_cid"] in goal_map:
            raise InvalidCatalogue("population_goal_invalid")
        goal_map[goal["goal_cid"]] = goal
    if any(task["goal_cid"] not in goal_map for task in selected.values()):
        raise InvalidCatalogue("population_goal_invalid")
    plans = population.get("plans")
    if not lineage and (not isinstance(plans, list) or len(plans) != 1
            or not isinstance(plans[0], dict) or plans[0].get("plan_cid") != plan_root
            or plans[0].get("goal_cid") not in goal_map):
        raise InvalidCatalogue("population_plan_invalid")
    for task in selected.values():
        dependencies = task.get("dependencies", [])
        if (not isinstance(dependencies, list) or len(dependencies) > MAX_TASKS
                or any(not isinstance(cid, str) or cid not in selected for cid in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise InvalidCatalogue("population_dependencies_invalid")
    expected_plan = (_lineage_population(population, selected, goal_map, plan_root, tree)
                     if lineage else plans[0])
    return selected, goal_map, expected_plan


def _bounded_rows(connection, sql, parameters=(), *, maximum=MAX_RUNTIME_ROWS, byte_budget=None):
    cursor = connection.execute(sql, parameters)
    if byte_budget is None:
        rows = cursor.fetchmany(maximum + 1)
    else:
        # V2 shares one finite budget across core and noncore observations.
        # Fetch incrementally; do not materialize hundreds of large bodies
        # before noticing that the closed-read budget has been exceeded.
        rows = []
        while len(rows) <= maximum:
            row = cursor.fetchone()
            if row is None:
                break
            byte_budget[0] -= len(json.dumps(row, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode()) + 1
            if byte_budget[0] < 0:
                raise InvalidCatalogue("native_graph_read_bound")
            rows.append(row)
    if len(rows) > maximum:
        raise InvalidCatalogue("native_row_bound")
    return rows


def _validate_graph_observation(plan, expected, edges, observed_rows, observed_edges):
    """Match the full declared graph; observe foreign prerequisites literally.

    Identity-only prerequisites do not acquire qualified body semantics merely
    because they are referenced. No observed status satisfies a parent here.
    """
    declared = {node["task_cid"]: node for node in plan.boundary["observed_tasks"]}
    roots = {(row["task_cid"], row["dependency_task_cid"], row["kind"])
             for row in plan.boundary["root_dependencies"]}
    authored = {(cid, dep, "depends_on") for cid, row in expected.items()
                for dep in row.get("dependencies", [])}
    wanted = authored | roots
    if set(edges) != wanted or len(edges) != len(wanted):
        raise InvalidCatalogue("native_graph_dependencies_changed")
    wanted_observed = {(cid, dep["dependency_task_cid"], dep["kind"])
                       for cid, node in declared.items() for dep in node["dependencies"]}
    if set(observed_edges) != wanted_observed or len(observed_edges) != len(wanted_observed):
        raise InvalidCatalogue("native_graph_dependencies_changed")
    if len(observed_rows) != len(declared):
        raise InvalidCatalogue("native_graph_population_mismatch")
    # Bound the actual fetched observation as well as the certificate. This is
    # separate from the existing smaller public diagnostic/report bound.
    if len(json.dumps([observed_rows, edges, observed_edges], ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()) > MAX_GRAPH_BOUNDARY_BYTES:
        raise InvalidCatalogue("native_graph_read_bound")
    seen, observations = set(), []
    for cid, alias, goal, plan_cid, objective, status, revision, raw in observed_rows:
        node = declared.get(cid)
        identity = (alias, goal, plan_cid, objective)
        if (node is None or cid in seen or type(revision) is not int
                or revision < node["minimum_revision"]
                or any(type(actual) is not type(node[key]) or actual != node[key]
                       for key, actual in zip(("task_alias", "goal_cid", "plan_cid", "objective_id"), identity))):
            raise InvalidCatalogue("native_graph_identity_changed")
        if type(raw) is not str or len(raw.encode()) > MAX_BODY_BYTES:
            raise InvalidCatalogue("native_body_bound")
        body = _decode(raw)
        namespace = node["namespace_binding"]
        if (type(body) is not dict or ("board_namespace" in body) != namespace["present"]
                or namespace["present"] and _graph_literal(body["board_namespace"]) != _graph_literal(namespace["value"])):
            raise InvalidCatalogue("native_graph_namespace_changed")
        binding = node["body_binding"]
        if binding["profile"] == "literal-prerequisite/v1" and _graph_resolution_marker(body):
            raise InvalidCatalogue("native_graph_resolution_profile_required")
        if binding["profile"] == "resolution-obligation/v1":
            projection = binding["projection"]
            actual = {key: body[key] for key in BOUNDARY_SEMANTIC_FIELDS if key in body}
            if (set(actual) != set(projection)
                    or any(type(actual[key]) is not type(value) or actual[key] != value
                           for key, value in projection.items())):
                raise InvalidCatalogue("native_graph_semantics_changed")
        _identifier(status)
        seen.add(cid)
        observations.append({"task_cid": cid, "task_alias": alias, "goal_cid": goal,
            "plan_cid": plan_cid, "objective_id": objective, "status": status, "revision": revision,
            "namespace_binding": namespace, "observation_role": node["observation_role"],
            "body_binding_profile": binding["profile"]})
    return {"schema": "native-observational-dependency-graph/v1",
            "authored_population_ref": plan.authored_population_ref,
            "dependency_boundary_ref": plan.dependency_boundary_ref,
            "population_envelope_ref": plan.envelope_ref,
            "authored_dependency_count": len(authored),
            "boundary_dependency_count": len(roots),
            "native_outgoing_dependency_count": len(edges),
            "observed_task_count": len(observations),
            "observed_dependency_count": len(observed_edges),
            **plan.graph_topology,
            "observed_status_counts": dict(collections.Counter(row["status"] for row in observations)),
            "observed_tasks": sorted(observations, key=lambda row: row["task_cid"]),
            "prerequisite_body_semantics_qualified": False,
            "completion_authority": False, "dispatch_authority": False,
            "observed_task_completion_established": False}


def _retained_outcomes(body, task_cid, revision):
    """Observe preserved native receipts without granting retry or completion."""
    entries = body.get("retained_execution_outcomes", [])
    if not isinstance(entries, list) or len(entries) > 16:
        raise InvalidCatalogue("native_retained_outcome_invalid")
    result, seen = [], set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {
                "task_cid", "task_revision", "completion_receipt", "sha256"}
                or entry["task_cid"] != task_cid
                or type(entry["task_revision"]) is not int
                or not 0 <= entry["task_revision"] < revision):
            raise InvalidCatalogue("native_retained_outcome_invalid")
        receipt = entry["completion_receipt"]
        if not isinstance(receipt, dict) or not receipt:
            raise InvalidCatalogue("native_retained_outcome_invalid")
        digest = hashlib.sha256(json.dumps(receipt, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if entry["sha256"] != digest or (entry["task_revision"], digest) in seen:
            raise InvalidCatalogue("native_retained_outcome_invalid")
        seen.add((entry["task_revision"], digest))
        attempt, maximum = receipt.get("attempt_number"), receipt.get("max_task_attempts")
        retry, exhausted = receipt.get("automatic_retry_admitted"), receipt.get("attempt_budget_exhausted")
        providers, failure = receipt.get("provider_invocation_count"), receipt.get("failure_kind")
        for value in (attempt, maximum):
            if value is not None and (type(value) is not int or not 1 <= value <= 10000):
                raise InvalidCatalogue("native_retained_outcome_invalid")
        if any(value is not None and type(value) is not bool for value in (retry, exhausted)):
            raise InvalidCatalogue("native_retained_outcome_invalid")
        if maximum is not None and attempt is not None and exhausted is not None and exhausted != (attempt >= maximum):
            raise InvalidCatalogue("native_retained_outcome_invalid")
        if providers is not None and (type(providers) is not int or not 0 <= providers <= 10000):
            raise InvalidCatalogue("native_retained_outcome_invalid")
        if failure is not None:
            _identifier(failure)
        if receipt.get("task_cid", task_cid) != task_cid:
            raise InvalidCatalogue("native_retained_outcome_invalid")
        result.append({"task_revision": entry["task_revision"], "receipt_sha256": digest,
                       "attempt_number": attempt, "max_task_attempts": maximum,
                       "automatic_retry_admitted": retry, "attempt_budget_exhausted": exhausted,
                       "provider_invocation_count": providers, "failure_kind": failure})
    return result


def _read_catalogue(arguments):
    """Worker-only direct SQL, deliberately independent of native constructors."""
    expected, expected_goals, expected_plan = _population(arguments)
    graph_budget = [MAX_GRAPH_BOUNDARY_BYTES] if isinstance(expected_plan, _GraphBoundaryPlan) else None
    database = Path(arguments["database_path"])
    before = {"database": _file_metadata(database),
              "wal": _file_metadata(Path(str(database) + ".wal"))}
    if before["database"] is None:
        raise InvalidCatalogue("database_missing")
    import duckdb

    connection = duckdb.connect(str(database), read_only=True, config={
        "threads": 1, "memory_limit": "128MB", "enable_external_access": False,
        "autoload_known_extensions": False, "autoinstall_known_extensions": False})
    try:
        connection.execute("BEGIN TRANSACTION")
        task_ids = sorted(expected)
        placeholders = ",".join("?" for _ in task_ids)
        cursor = int(connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events").fetchone()[0])
        rows = _bounded_rows(connection,
            "SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, status, revision, body_json "
            f"FROM tasks WHERE task_cid IN ({placeholders}) ORDER BY task_cid", task_ids,
            maximum=MAX_TASKS, byte_budget=graph_budget)
        if len(rows) != len(expected):
            raise InvalidCatalogue("native_population_mismatch")
        tasks = []
        for cid, alias, goal, plan, objective, status, revision, raw in rows:
            original = expected[cid]
            if ((alias, goal, plan, objective) != tuple(original[key] for key in
                    ("task_alias", "goal_cid", "plan_cid", "objective_id"))
                    or type(revision) is not int or revision < 0):
                raise InvalidCatalogue("native_identity_mismatch")
            if not isinstance(raw, str) or len(raw.encode()) > MAX_BODY_BYTES:
                raise InvalidCatalogue("native_body_bound")
            body = _decode(raw)
            if not isinstance(body, dict) or body.get("board_namespace") != arguments["board_namespace"]:
                raise InvalidCatalogue("native_namespace_mismatch")
            _identifier(status)
            receipt = body.get("completion_receipt", {})
            if not isinstance(receipt, dict):
                raise InvalidCatalogue("native_receipt_invalid")
            attempt = receipt.get("attempt_number")
            if attempt is not None and (type(attempt) is not int or not 1 <= attempt <= 10_000):
                raise InvalidCatalogue("native_receipt_invalid")
            retry = receipt.get("automatic_retry_admitted")
            if retry is not None and type(retry) is not bool:
                raise InvalidCatalogue("native_receipt_invalid")
            maximum = receipt.get("max_task_attempts")
            exhausted = receipt.get("attempt_budget_exhausted")
            if maximum is not None and (type(maximum) is not int or not 1 <= maximum <= 10_000):
                raise InvalidCatalogue("native_receipt_invalid")
            if exhausted is not None and type(exhausted) is not bool:
                raise InvalidCatalogue("native_receipt_invalid")
            if maximum is not None and attempt is not None and exhausted is not None:
                if exhausted != (attempt >= maximum):
                    raise InvalidCatalogue("native_receipt_invalid")
            failure = receipt.get("failure_kind")
            if failure is not None:
                _identifier(failure)
            providers = receipt.get("provider_invocation_count")
            if providers is not None and (type(providers) is not int or not 0 <= providers <= 10000):
                raise InvalidCatalogue("native_receipt_invalid")
            retained = _retained_outcomes(body, cid, revision)
            tasks.append({"task_cid": cid, "task_alias": alias, "goal_cid": goal,
                          "plan_cid": plan, "status": status, "revision": revision,
                          "completion_receipt_attempt_number": attempt,
                          "automatic_retry_admitted": retry,
                          "receipt_max_task_attempts": maximum,
                          "receipt_attempt_budget_exhausted": exhausted,
                          "receipt_failure_kind": failure,
                          "receipt_provider_invocation_count": providers,
                          "retained_execution_outcomes": retained})
        if isinstance(expected_plan, _GraphBoundaryPlan):
            count = int(connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE json_extract(body_json, '$.board_namespace') = CAST(? AS JSON)",
                [json.dumps(arguments["board_namespace"])]).fetchone()[0])
            matching_noncore = sum(node["namespace_binding"]["present"]
                and type(node["namespace_binding"]["value"]) is str
                and node["namespace_binding"]["value"] == arguments["board_namespace"]
                for node in expected_plan.boundary["observed_tasks"])
        else:
            count = int(connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE json_extract_string(body_json, '$.board_namespace') = ?",
                [arguments["board_namespace"]]).fetchone()[0])
            matching_noncore = 0
        if count != len(expected) + matching_noncore:
            raise InvalidCatalogue("native_population_mismatch")
        goal_ids = sorted(expected_goals)
        goal_rows = _bounded_rows(connection,
            "SELECT goal_cid, goal_alias, objective_id, parent_goal_cid, status, revision FROM goals "
            "WHERE goal_cid IN (" + ",".join("?" for _ in goal_ids) + ") ORDER BY goal_cid",
            goal_ids, maximum=MAX_TASKS, byte_budget=graph_budget)
        if len(goal_rows) != len(expected_goals):
            raise InvalidCatalogue("native_goal_mismatch")
        for cid, alias, objective, parent, status, revision in goal_rows:
            original = expected_goals[cid]
            if ((alias, objective, parent) != (original["goal_alias"], original["objective_id"],
                                              original.get("parent_goal_cid", ""))
                    or type(revision) is not int or revision < 0):
                raise InvalidCatalogue("native_goal_mismatch")
            _identifier(status)
        if isinstance(expected_plan, _LineagePlan):
            _read_plan_lineage(connection, expected_plan, expected_goals)
        else:
            plan_rows = _bounded_rows(connection,
                "SELECT plan_cid, goal_cid FROM plans WHERE plan_cid = ?", [arguments["plan_root_cid"]], maximum=1,
                byte_budget=graph_budget)
            if plan_rows != [(expected_plan["plan_cid"], expected_plan["goal_cid"])]:
                raise InvalidCatalogue("native_plan_mismatch")
        attempts = _bounded_rows(connection,
            "SELECT task_cid, attempt_id, attempt_number, status FROM task_attempts "
            f"WHERE task_cid IN ({placeholders}) ORDER BY task_cid, attempt_number", task_ids, byte_budget=graph_budget)
        providers = _bounded_rows(connection,
            "SELECT task_cid, attempt_id, status FROM provider_invocations "
            f"WHERE task_cid IN ({placeholders}) ORDER BY task_cid, attempt_id", task_ids, byte_budget=graph_budget)
        claims = _bounded_rows(connection,
            "SELECT task_cid, state, COUNT(*) FROM task_claims "
            f"WHERE task_cid IN ({placeholders}) GROUP BY task_cid, state", task_ids, byte_budget=graph_budget)
        edge_columns = "task_cid, dependency_task_cid, kind" if isinstance(expected_plan, _GraphBoundaryPlan) else "task_cid, dependency_task_cid"
        edges = _bounded_rows(connection,
            "SELECT " + edge_columns + " FROM task_dependencies "
            f"WHERE task_cid IN ({placeholders}) ORDER BY task_cid, dependency_task_cid", task_ids,
            maximum=MAX_GRAPH_EDGES if isinstance(expected_plan, _GraphBoundaryPlan) else MAX_RUNTIME_ROWS,
            byte_budget=graph_budget)
        expected_edges = {(cid, dependency) for cid, task in expected.items()
                          for dependency in task.get("dependencies", [])}
        boundary_observation = None
        if isinstance(expected_plan, _GraphBoundaryPlan):
            observed_ids = sorted(node["task_cid"] for node in expected_plan.boundary["observed_tasks"])
            observed_rows, observed_edges = [], []
            if observed_ids:
                observed_placeholders = ",".join("?" for _ in observed_ids)
                observed_rows = _bounded_rows(connection,
                    "SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, status, revision, body_json "
                    f"FROM tasks WHERE task_cid IN ({observed_placeholders}) ORDER BY task_cid",
                    observed_ids, maximum=MAX_TASKS - len(expected), byte_budget=graph_budget)
                observed_edges = _bounded_rows(connection,
                    "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies "
                    f"WHERE task_cid IN ({observed_placeholders}) ORDER BY task_cid, dependency_task_cid",
                    observed_ids, maximum=MAX_GRAPH_EDGES, byte_budget=graph_budget)
            boundary_observation = _validate_graph_observation(expected_plan, expected,
                                                               edges, observed_rows, observed_edges)
            dependencies = len(expected_edges)
        else:
            if set(edges) != expected_edges or len(edges) != len(expected_edges):
                raise InvalidCatalogue("native_dependencies_mismatch")
            dependencies = len(edges)
        generation = connection.execute(
            "SELECT generation, schema_revision, database_uuid FROM store_generations "
            "ORDER BY generation DESC LIMIT 1").fetchone()
        if (not generation or type(generation[0]) is not int or generation[0] < 1
                or type(generation[1]) is not int or generation[1] < 1):
            raise InvalidCatalogue("native_store_identity_missing")
        _identifier(generation[2])
        closing = int(connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events").fetchone()[0])
        if cursor != closing:
            raise InvalidCatalogue("native_snapshot_changed")
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    after = {"database": _file_metadata(database),
             "wal": _file_metadata(Path(str(database) + ".wal"))}
    if before != after:
        raise InvalidCatalogue("native_store_changed_during_read")
    return {"tasks": tasks, "goal_count": len(goal_rows), "dependency_count": dependencies,
            **({"population_schema": BOUNDARY_POPULATION_SCHEMA,
                "dependency_boundary": boundary_observation} if boundary_observation is not None else {}),
            **({"population_schema": LINEAGE_POPULATION_SCHEMA,
                "plan_lineage": expected_plan["lineage"]} if isinstance(expected_plan, _LineagePlan) else {}),
            "attempt_rows": attempts, "provider_status_rows": providers, "claim_state_rows": claims,
            "event_cursor": cursor, "store_generation": generation[0],
            "store_schema_revision": generation[1], "database_uuid": generation[2],
            "database_and_wal_metadata_unchanged": True,
            "read_finished_at_epoch": time.time()}


def _worker_exception_metadata(error):
    """Bounded exception identity/digest only; never serialize SQL or data text."""
    name = type(error).__name__
    if type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name) is None:
        name = "Exception"
    result = {"schema": WORKER_ERROR_SCHEMA, "exception_class": name}
    try:
        message = str(error)
        if type(message) is str and len(message) <= MAX_BODY_BYTES:
            raw = message.encode()
            if len(raw) <= MAX_BODY_BYTES:
                result["error_text_sha256"] = hashlib.sha256(raw).hexdigest()
    except Exception:
        pass
    return result


def _safe_worker_error(value, returncode, stderr=None):
    """Only whitelisted scalar metadata crosses into an unknown diagnostic."""
    result = {"schema": WORKER_ERROR_SCHEMA}
    if type(returncode) is int and -(2**31) <= returncode < 2**31:
        result["returncode"] = returncode
    if type(value) is dict and value.get("schema") == WORKER_ERROR_SCHEMA:
        name = value.get("exception_class")
        if type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name):
            result["exception_class"] = name
        digest = value.get("error_text_sha256")
        if type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest):
            result["error_text_sha256"] = digest
    if type(stderr) is bytes and 0 < len(stderr) <= MAX_REPORT_BYTES:
        result["stderr_sha256"] = hashlib.sha256(stderr).hexdigest()
    return result


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 ** 2, 1536 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _unknown(board_id, arguments, code, *, worker_error=None):
    return {"schema": SCHEMA, "board_id": board_id, "health": "unknown",
            "complete": False, "completion_candidate": False, "busy": False,
            "progress_token": "", "reason_codes": [code],
            "details": {"completion_authority": False, "native_catalogue": {
                "schema": DIAGNOSTIC_SCHEMA, "board_namespace": arguments.get("board_namespace"),
                "plan_root_cid": arguments.get("plan_root_cid"),
                "repository_tree_id": arguments.get("repository_tree_id"),
                "population_sha256": arguments.get("population_sha256"),
                "probe_observed_at_epoch": time.time(),
                "checked_at_epoch": None, "fresh_read": False,
                "activation_state": "unknown", "completion_authority": False,
                **({"worker_error": worker_error} if worker_error is not None else {})}}}


def observe_native_catalogue(database_path: Path, population_path: Path, *, board_id: str,
        board_namespace: str, plan_root_cid: str, repository_tree_id: str,
        population_sha256: str, max_age_seconds: float = 180, timeout_seconds: float = 15,
        now: float | None = None, executor_profile_ref: dict | None = None):
    """Fresh catalogue observation; no executor or task-activation authority."""
    arguments = {"database_path": str(database_path), "population_path": str(population_path),
                 "board_namespace": board_namespace, "plan_root_cid": plan_root_cid,
                 "repository_tree_id": repository_tree_id, "population_sha256": population_sha256}
    worker_error = None
    try:
        _identifier(board_id)
        for key in ("board_namespace", "plan_root_cid", "repository_tree_id"):
            _identifier(arguments[key])
        for value, maximum in ((max_age_seconds, 3600), (timeout_seconds, 30)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise InvalidCatalogue("invalid_binding")
        if not isinstance(population_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", population_sha256) is None:
            raise InvalidCatalogue("invalid_binding")
        # Parent admission uses bounded nonblocking reads, before allocating a worker.
        _population(arguments)
        env = {key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "LC_ALL")}
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        process = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                  "--worker", json.dumps(arguments, separators=(",", ":"))],
            capture_output=True, timeout=timeout_seconds, env=env, preexec_fn=_limits)
        worker_error = _safe_worker_error(None, process.returncode,
                                          process.stderr if process.returncode != 0 else None)
        if process.returncode != 0 or len(process.stdout) > MAX_REPORT_BYTES:
            raise InvalidCatalogue("native_worker_failed")
        data = _decode(process.stdout)
        if not isinstance(data, dict) or data.get("error"):
            code = data.get("error") if isinstance(data, dict) else "native_worker_invalid"
            if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,80}", code):
                code = "native_worker_invalid"
            return _unknown(board_id, arguments, "native_catalogue_" + code,
                worker_error=_safe_worker_error(data.get("worker_error") if isinstance(data, dict) else None,
                                                process.returncode))
        checked = data["read_finished_at_epoch"]
        observed = time.time() if now is None else now
        if (type(checked) not in (int, float) or not math.isfinite(checked)
                or type(observed) not in (int, float) or not math.isfinite(observed)
                or not 0 <= observed - checked <= max_age_seconds):
            raise InvalidCatalogue("native_read_stale")
        tasks = data["tasks"]
        counts = dict(collections.Counter(task["status"] for task in tasks))
        # A later independent factual completion does not settle older provider actions.
        # Keep those observations, but do not present a completed task's history as
        # the current frontier's blocker. This observer still cannot admit retries,
        # establish an executor, or infer claimability from catalogue status.
        retry_not_admitted, exhausted, external_unknown, retained_unknown = [], [], [], []
        provider_unknown = {row[0] for row in data["provider_status_rows"]
                            if "unknown" in str(row[2]).lower()}
        for task in tasks:
            retained = task.get("retained_execution_outcomes", [])
            current = {"automatic_retry_admitted": task["automatic_retry_admitted"],
                       "attempt_budget_exhausted": task["receipt_attempt_budget_exhausted"],
                       "failure_kind": task["receipt_failure_kind"],
                       "provider_invocation_count": task.get("receipt_provider_invocation_count")}
            outcomes = [current, *retained]
            cid = task["task_cid"]
            if any(outcome["automatic_retry_admitted"] is False for outcome in outcomes):
                retry_not_admitted.append(cid)
            if any(outcome["attempt_budget_exhausted"] is True for outcome in outcomes):
                exhausted.append(cid)
            for outcome in outcomes:
                failure = str(outcome["failure_kind"]).lower()
                if ("unknown" in failure and ("provider" in failure or
                        (outcome.get("provider_invocation_count") or 0) > 0)):
                    provider_unknown.add(cid)
                if "outcome_unknown" in failure and cid not in external_unknown:
                    external_unknown.append(cid)
            if any("unknown" in str(outcome["failure_kind"]).lower() for outcome in retained):
                retained_unknown.append(cid)
        # Only the native completed status separates accepted factual work from
        # unresolved execution history. Other statuses (including cancellation or
        # failure) remain conservative blockers. These lists never authorize reuse
        # or replay of the historical attempt of a completed task.
        nonterminal = {task["task_cid"] for task in tasks if task["status"] != "completed"}
        current_blockers = {
            "retry_not_admitted_task_cids": sorted(nonterminal.intersection(retry_not_admitted)),
            "exhausted_receipt_task_cids": sorted(nonterminal.intersection(exhausted)),
            "provider_outcome_unknown_task_cids": sorted(nonterminal.intersection(provider_unknown)),
            "external_outcome_unknown_task_cids": sorted(nonterminal.intersection(external_unknown)),
        }
        codes = ["native_catalogue_activation_missing"]
        if counts.get("blocked") == len(tasks):
            codes.append("native_catalogue_expected_blocked")
        else:
            codes.append("native_catalogue_executor_unverified")
        if current_blockers["retry_not_admitted_task_cids"]: codes.append("native_catalogue_retry_not_admitted")
        if current_blockers["exhausted_receipt_task_cids"]: codes.append("native_catalogue_attempt_budget_exhausted")
        if current_blockers["provider_outcome_unknown_task_cids"]: codes.append("native_catalogue_provider_outcome_unknown")
        if current_blockers["external_outcome_unknown_task_cids"]: codes.append("native_catalogue_outcome_unknown")
        diagnostic = {"schema": DIAGNOSTIC_SCHEMA, **{key: arguments[key] for key in
            ("board_namespace", "plan_root_cid", "repository_tree_id", "population_sha256")},
            "probe_observed_at_epoch": time.time(),
            "checked_at_epoch": checked, "fresh_read": True, "activation_state": "not_activated",
            "executor_health": "not_established", "completion_authority": False,
            "task_count": len(tasks), "goal_count": data["goal_count"],
            "dependency_count": data["dependency_count"], "counts": counts,
            "ready_status_count": sum(counts.get(status, 0) for status in READY_STATUSES),
            "tasks": tasks, "event_cursor": data["event_cursor"],
            "store_generation": data["store_generation"], "store_schema_revision": data["store_schema_revision"],
            "database_uuid": data["database_uuid"],
            "database_and_wal_metadata_unchanged": data["database_and_wal_metadata_unchanged"],
            "normalized_attempt_rows": data["attempt_rows"],
            "normalized_provider_status_rows": data["provider_status_rows"],
            "normalized_claim_state_rows": data["claim_state_rows"],
            "attempt_history_complete": False,
            "empty_normalized_rows_establish_zero_attempts_or_provider_success": False,
            "provider_outcome_knowledge": "unknown_observed" if provider_unknown else "not_established",
            "external_outcome_knowledge": "unknown_observed" if external_unknown else "not_established",
            "retry_not_admitted_task_cids": retry_not_admitted,
            "exhausted_receipt_task_cids": exhausted,
            "retained_unknown_outcome_task_cids": retained_unknown,
            "current_nonterminal_blockers": current_blockers,
            "historical_outcomes_are_separate_from_current_completion": True}
        if data.get("population_schema") == LINEAGE_POPULATION_SCHEMA:
            diagnostic.update(population_schema=LINEAGE_POPULATION_SCHEMA,
                              plan_lineage=data["plan_lineage"])
        if data.get("population_schema") == BOUNDARY_POPULATION_SCHEMA:
            diagnostic.update(population_schema=BOUNDARY_POPULATION_SCHEMA,
                              dependency_boundary=data["dependency_boundary"])
        if executor_profile_ref is not None:
            # The native catalogue remains an observer. Its opt-in executor
            # signal comes from independent process/source/original-cycle proof.
            try:
                from ipfs_accelerate_py.agent_supervisor.rescue.native_executor_health import verify_bound_executor
                verification = verify_bound_executor(executor_profile_ref, diagnostic,
                                                       now=observed)
            except (ImportError, OSError, ValueError, TypeError, KeyError):
                verification = {"verified": False, "reason": "executor_verifier_unavailable"}
            diagnostic["executor_verification"] = verification
            if verification.get("verified") is True:
                diagnostic.update(activation_state="active", executor_health="healthy",
                                  executor_scope=verification["scope"])
                codes = [code for code in codes if code not in {
                    "native_catalogue_activation_missing", "native_catalogue_executor_unverified"}]
                if diagnostic["ready_status_count"] == 0 and counts.get("blocked", 0):
                    if "native_catalogue_expected_blocked" not in codes:
                        codes.append("native_catalogue_expected_blocked")
                elif not codes:
                    codes.append("native_catalogue_executor_active")
            elif "native_catalogue_executor_unverified" not in codes:
                codes.append("native_catalogue_executor_unverified")
        progress = [data["event_cursor"],
                    [[task["task_cid"], task["status"], task["revision"]] for task in tasks]]
        boundary = data.get("dependency_boundary", {})
        if boundary.get("schema") == "native-observational-dependency-graph/v1":
            progress.append([[task["task_cid"], task["status"], task["revision"]]
                             for task in boundary["observed_tasks"]])
        token = hashlib.sha256(json.dumps(progress, separators=(",", ":")).encode()).hexdigest()
        result = {"schema": SCHEMA, "board_id": board_id, "health": "blocked",
                  "complete": False, "completion_candidate": False, "busy": False,
                  "progress_token": token, "reason_codes": codes,
                  "details": {"completion_authority": False, "native_catalogue": diagnostic,
                              "wait": {"kind": "dependency_or_review", "completion_authority": False}}}
        if len(json.dumps(result).encode()) > MAX_REPORT_BYTES:
            raise InvalidCatalogue("native_report_bound")
        return result
    except subprocess.TimeoutExpired:
        return _unknown(board_id, arguments, "native_catalogue_read_timeout")
    except InvalidCatalogue as error:
        return _unknown(board_id, arguments, "native_catalogue_" + str(error), worker_error=worker_error)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return _unknown(board_id, arguments, "native_catalogue_read_unavailable", worker_error=worker_error)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--worker":
        try:
            value = _read_catalogue(_decode(argv[1]))
        except InvalidCatalogue as error:
            try:
                code = str(error)
            except Exception:
                code = "native_worker_failed"
            if re.fullmatch(r"[a-z_]{1,80}", code) is None:
                code = "native_worker_invalid"
            value = {"error": code, "worker_error": _worker_exception_metadata(error)}
        except Exception as error:
            value = {"error": "read_unavailable", "worker_error": _worker_exception_metadata(error)}
        encoded = json.dumps(value, separators=(",", ":"))
        print(encoded if len(encoded.encode()) <= MAX_REPORT_BYTES else '{"error":"native_report_bound"}')
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--plan-root-cid", required=True)
    parser.add_argument("--repository-tree-id", required=True)
    parser.add_argument("--population-sha256", required=True)
    parser.add_argument("--max-age-seconds", type=float, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=15)
    parser.add_argument("--executor-profile", type=Path)
    parser.add_argument("--executor-profile-bytes", type=int)
    parser.add_argument("--executor-profile-sha256")
    args = parser.parse_args(argv)
    selected = (args.executor_profile, args.executor_profile_bytes, args.executor_profile_sha256)
    if any(value is not None for value in selected) and not all(value is not None for value in selected):
        parser.error("executor health requires the complete path/bytes/SHA descriptor")
    executor_ref = (None if args.executor_profile is None else {
        "path": str(args.executor_profile), "bytes": args.executor_profile_bytes,
        "sha256": args.executor_profile_sha256})
    print(json.dumps(observe_native_catalogue(args.database, args.population,
        board_id=args.board, board_namespace=args.namespace, plan_root_cid=args.plan_root_cid,
        repository_tree_id=args.repository_tree_id, population_sha256=args.population_sha256,
        max_age_seconds=args.max_age_seconds, timeout_seconds=args.timeout_seconds,
        executor_profile_ref=executor_ref), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
