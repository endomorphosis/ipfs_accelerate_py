#!/usr/bin/env python3
"""Validate this authoring contract and supervisor projections without launching work.

The default path uses only the standard library. Native parser qualification is
optional and runs the requested checkout's real parsers in a separate, read-only,
network-disabled Python process. This is structural planning evidence; it does
not establish correctness of future model, graph, or Lean outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any


TASK_ID = re.compile(r"LAC-[0-9]{3}")
GOAL_ID = re.compile(r"LAC-G[0-9]{3}")
NATIVE_MARKER = "NATIVE_PARSER_RESULT="


def load_json(path: Path) -> Any:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def safe_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\\" not in value
        and not any(char in value for char in "\x00\n\r:*?[]")
        and not value.endswith("/")
        and not PurePosixPath(value).is_absolute()
        and not any(part in {".", ".."} for part in value.split("/"))
        and "//" not in value
    )


def graph_order(edges: dict[str, list[str]], label: str, errors: list[str]) -> list[str]:
    pending = {key: set(value) for key, value in edges.items()}
    result: list[str] = []
    while pending:
        ready = sorted(key for key, deps in pending.items() if not deps)
        if not ready:
            errors.append(f"{label} cycle or unresolved reference among: {', '.join(sorted(pending))}")
            break
        result.extend(ready)
        for key in ready:
            del pending[key]
        for deps in pending.values():
            deps.difference_update(ready)
    return result


def field_records(text: str, prefix: str, errors: list[str]) -> list[dict[str, Any]]:
    """Read the documented field grammar; native qualification remains separate."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            current = None
            heading = line[3:].split(" ", 1)
            if heading[0].startswith(prefix):
                current = {"id": heading[0], "title": heading[1] if len(heading) > 1 else "", "fields": {}}
                records.append(current)
        elif current is not None and line.strip().startswith("- ") and ":" in line:
            key, value = line.strip()[2:].split(":", 1)
            key = key.strip().lower()
            if key in current["fields"]:
                errors.append(f"{current['id']}: duplicate Markdown field '{key}' at line {lineno}")
            current["fields"][key] = value.strip()
    return records


def csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def compare_projections(
    task_records: list[dict[str, Any]], goal_records: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]], goals: dict[str, dict[str, Any]],
    errors: list[str], *, source: str,
) -> None:
    for records, expected, label in ((task_records, tasks, "tasks"), (goal_records, goals, "goals")):
        ids = [record["id"] for record in records]
        duplicate = [key for key, count in Counter(ids).items() if count > 1]
        if duplicate:
            errors.append(f"{source} duplicate {label}: {', '.join(duplicate)}")
        missing, extra = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
        if missing or extra:
            errors.append(f"{source} {label} population mismatch: missing={missing}, extra={extra}")
    for record in task_records:
        expected = tasks.get(record["id"])
        if expected is None:
            continue
        fields = record["fields"]
        if record["title"] != expected.get("title"):
            errors.append(f"{source} {record['id']}: title differs from tasks.json")
        if csv(fields.get("depends on", "")) != expected.get("depends_on"):
            errors.append(f"{source} {record['id']}: dependency projection differs from tasks.json")
        if fields.get("goal id") != expected.get("goal_id"):
            errors.append(f"{source} {record['id']}: Goal id differs from tasks.json")
        if fields.get("status", "").lower() != expected.get("status"):
            errors.append(f"{source} {record['id']}: Status differs from tasks.json")
        if csv(fields.get("predicted files", "")) != expected.get("owned_paths"):
            errors.append(f"{source} {record['id']}: Predicted files differs from owned_paths")
        if csv(fields.get("outputs", "")) != expected.get("outputs"):
            errors.append(f"{source} {record['id']}: Outputs differs from tasks.json")
        if fields.get("parallel lane") != expected.get("lane"):
            errors.append(f"{source} {record['id']}: Parallel lane differs from tasks.json")
    for record in goal_records:
        expected = goals.get(record["id"])
        if expected is not None:
            if record["title"] != expected.get("title"):
                errors.append(f"{source} {record['id']}: goal title differs from tasks.json")
            if record["fields"].get("parent", "") != (expected.get("parent") or ""):
                errors.append(f"{source} {record['id']}: Parent differs from tasks.json")


NATIVE_SCRIPT = r'''
import json, os, pathlib, sys
sys.dont_write_bytecode = True
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
def read_only(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & write_flags):
            raise PermissionError("native parser qualification forbids file writes")
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.utime", "subprocess.Popen", "os.system", "socket.connect", "socket.bind", "socket.sendto"}:
        raise PermissionError("native parser qualification forbids effects: " + event)
sys.addaudithook(read_only)
root, tasks_path, goals_path = map(pathlib.Path, sys.argv[1:4])
sys.path.insert(0, str(root))
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
tasks = parse_task_file(tasks_path, "## LAC-")
goals = parse_goal_heap(goals_path.read_text(encoding="utf-8"))
payload = {"tasks": [{"id": t.task_id, "title": t.title, "fields": dict(t.metadata)} for t in tasks],
           "goals": [{"id": g.goal_id, "title": g.title, "fields": dict(g.fields)} for g in goals],
           "implementation_parser": str(pathlib.Path(sys.modules[parse_task_file.__module__].__file__).resolve()),
           "objective_parser": str(pathlib.Path(sys.modules[parse_goal_heap.__module__].__file__).resolve())}
print("NATIVE_PARSER_RESULT=" + json.dumps(payload, sort_keys=True))
'''


def native_parse(accelerate_root: Path, plan_dir: Path) -> dict[str, Any]:
    root = accelerate_root.resolve(strict=True)
    if not (root / "ipfs_accelerate_py" / "agent_supervisor").is_dir():
        raise ValueError("--accelerate-root must contain ipfs_accelerate_py/agent_supervisor")
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "IPFS_ACCEL_SKIP_CORE": "1",
        "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_AUTO_INSTALL": "false",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    })
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", NATIVE_SCRIPT, str(root),
         str(plan_dir / "tasks.todo.md"), str(plan_dir / "goals.objectives.md")],
        env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True,
        timeout=60, check=False,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout)[-4000:].strip()
        raise ValueError(f"native parser failed (exit {proc.returncode}); select a compatible clean checkout with dependencies already installed. No installation or launch was attempted. {detail}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith(NATIVE_MARKER)]
    if len(lines) != 1:
        raise ValueError("native parser did not emit exactly one result record")
    result = json.loads(lines[0][len(NATIVE_MARKER):])
    for key in ("implementation_parser", "objective_parser"):
        if not Path(result[key]).is_relative_to(root):
            raise ValueError(f"native {key} was loaded outside the requested checkout")
    return result


def validate(plan_dir: Path, accelerate_root: Path | None, require_native: bool) -> dict[str, Any]:
    errors: list[str] = []
    files = {name: plan_dir / name for name in ("tasks.json", "tasks.todo.md", "goals.objectives.md")}
    try:
        data = load_json(files["tasks.json"])
        if not isinstance(data, dict):
            raise ValueError("tasks.json root must be an object")
        markdown = {name: files[name].read_text(encoding="utf-8") for name in ("tasks.todo.md", "goals.objectives.md")}
    except (OSError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)]}
    if data.get("schema") != "legal-autoformalization-plan/v1":
        errors.append("tasks.json: unsupported schema")
    if not isinstance(data.get("campaign_id"), str) or not data["campaign_id"].strip():
        errors.append("tasks.json: campaign_id must be nonempty")
    profiles = data.get("resource_profiles")
    if not isinstance(profiles, dict) or not profiles:
        errors.append("resource_profiles must be a nonempty object")
        profiles = {}
    if not isinstance(data.get("policy"), dict) or not data.get("policy"):
        errors.append("policy must be a nonempty object")
    goals: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    for key, pattern, records in (("goals", GOAL_ID, goals), ("tasks", TASK_ID, tasks)):
        items = data.get(key)
        if not isinstance(items, list) or not items:
            errors.append(f"{key} must be a nonempty array")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{key}[{index}] must be an object")
                continue
            identity = item.get("id")
            if not isinstance(identity, str) or not pattern.fullmatch(identity):
                errors.append(f"{key}[{index}] has an invalid id")
                continue
            if identity in records:
                errors.append(f"duplicate {key} id: {identity}")
            records[identity] = item
            if not isinstance(item.get("title"), str) or not item["title"].strip():
                errors.append(f"{identity}: title must be nonempty")
            if not strings(item.get("acceptance")) or not item["acceptance"]:
                errors.append(f"{identity}: acceptance must be a nonempty string array")
    goal_edges: dict[str, list[str]] = {}
    roots: list[str] = []
    for identity, item in goals.items():
        parent = item.get("parent")
        if parent in (None, ""):
            roots.append(identity)
            goal_edges[identity] = []
        elif not isinstance(parent, str) or parent not in goals:
            errors.append(f"{identity}: unknown parent {parent!r}")
            goal_edges[identity] = []
        else:
            goal_edges[identity] = [parent]
    if len(roots) != 1:
        errors.append(f"goal hierarchy must have one root; found {roots}")
    graph_order(goal_edges, "goal hierarchy", errors)
    task_edges: dict[str, list[str]] = {}
    for identity, item in tasks.items():
        if not isinstance(item.get("goal_id"), str) or item["goal_id"] not in goals:
            errors.append(f"{identity}: unknown goal_id")
        for key in ("repository", "lane", "priority", "objective", "resource_profile", "gate"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                errors.append(f"{identity}: {key} must be nonempty")
        if item.get("status") != "todo":
            errors.append(f"{identity}: a new plan must declare status todo, not {item.get('status')!r}")
        profile = item.get("resource_profile")
        if not isinstance(profile, str) or profile not in profiles:
            errors.append(f"{identity}: unknown resource_profile {profile!r}")
        for key in ("depends_on", "existing_owners", "owned_paths", "outputs", "acceptance", "validation", "exclusive_keys", "allowed_effects", "inputs"):
            if not strings(item.get(key)):
                errors.append(f"{identity}: {key} must be a string array")
        for key in ("owned_paths", "outputs", "validation"):
            if not item.get(key):
                errors.append(f"{identity}: {key} must be nonempty")
        for key in ("owned_paths", "outputs"):
            for path in item.get(key, []) if isinstance(item.get(key), list) else []:
                if not safe_path(path):
                    errors.append(f"{identity}: unsafe {key} path {path!r}")
        deps = item.get("depends_on") if strings(item.get("depends_on")) else []
        if len(deps) != len(set(deps)):
            errors.append(f"{identity}: duplicate dependency")
        if identity in deps:
            errors.append(f"{identity}: self dependency")
        for dep in deps:
            if dep not in tasks:
                errors.append(f"{identity}: unknown dependency {dep}")
        task_edges[identity] = list(deps)
    order = graph_order(task_edges, "task dependency graph", errors)
    ancestors: dict[str, set[str]] = {}
    for identity in order:
        inherited: set[str] = set()
        for dep in task_edges[identity]:
            inherited.add(dep)
            inherited.update(ancestors.get(dep, set()))
        ancestors[identity] = inherited
    task_items = list(tasks.items())
    for index, (left_id, left) in enumerate(task_items):
        for right_id, right in task_items[index + 1:]:
            if left.get("repository") != right.get("repository"):
                continue
            if left_id in ancestors.get(right_id, set()) or right_id in ancestors.get(left_id, set()):
                continue
            shared_paths = set(left.get("owned_paths", [])) & set(right.get("owned_paths", []))
            shared_keys = set(left.get("exclusive_keys", [])) & set(right.get("exclusive_keys", []))
            if shared_paths and not shared_keys:
                errors.append(f"{left_id}/{right_id}: unordered shared write paths lack a common exclusion key: {sorted(shared_paths)}")
    parents = {item.get("parent") for item in goals.values() if isinstance(item.get("parent"), str)}
    leaves = set(goals) - parents
    covered = {item.get("goal_id") for item in tasks.values() if isinstance(item.get("goal_id"), str)}
    if leaves - covered:
        errors.append(f"leaf goals without tasks: {sorted(leaves - covered)}")
    compare_projections(
        field_records(markdown["tasks.todo.md"], "LAC-", errors),
        field_records(markdown["goals.objectives.md"], "LAC-G", errors),
        tasks, goals, errors, source="Markdown",
    )
    native: dict[str, Any] = {"requested": bool(accelerate_root), "qualified": False}
    if require_native and accelerate_root is None:
        errors.append("--require-native-parser requires --accelerate-root")
    if accelerate_root is not None:
        try:
            result = native_parse(accelerate_root, plan_dir)
            before = len(errors)
            compare_projections(result["tasks"], result["goals"], tasks, goals, errors, source="native parser")
            native.update({"qualified": len(errors) == before, "accelerate_root": str(accelerate_root.resolve()),
                           "implementation_parser": result["implementation_parser"], "objective_parser": result["objective_parser"]})
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    return {
        "schema": "legal-autoformalization-plan-validation/v1", "valid": not errors,
        "campaign_id": data.get("campaign_id"), "goal_count": len(goals), "task_count": len(tasks),
        "root_goal_ids": roots, "leaf_goal_ids": sorted(leaves),
        "initial_ready_task_ids": sorted(identity for identity, deps in task_edges.items() if not deps),
        "topological_order": order,
        "lanes": dict(sorted(Counter(str(item.get("lane", "")) for item in tasks.values()).items())),
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "native_parser": native, "errors": errors,
        "effects": {"task_imported": False, "daemon_launched": False, "model_run": False, "state_written": False},
        "limits": "Structure and projection consistency only; launch qualification and output correctness remain campaign tasks.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--accelerate-root", type=Path)
    parser.add_argument("--require-native-parser", action="store_true")
    parser.add_argument("--check-all", action="store_true", help="Compatibility alias for configured-board preflight; all structural checks already run by default.")
    args = parser.parse_args()
    result = validate(args.plan_dir.resolve(), args.accelerate_root, args.require_native_parser)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
