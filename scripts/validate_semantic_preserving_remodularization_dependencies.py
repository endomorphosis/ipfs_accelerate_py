#!/usr/bin/env python3
"""Fail-closed source/dependency seal validator for SPAR."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py"
SEAL_PATH = ROOT / "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json"


class ValidationError(RuntimeError):
    pass


def _load_spec() -> Any:
    spec = importlib.util.spec_from_file_location("spar_control_spec_dependency_validator", SPEC_PATH)
    if spec is None or spec.loader is None:
        raise ValidationError("SPAR control specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path.relative_to(ROOT)} must contain one object")
    return value


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=cwd, text=True, capture_output=True, check=False, timeout=60)
    if result.returncode:
        raise ValidationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _acyclic(task_ids: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> bool:
    children = {task: [] for task in task_ids}
    indegree = {task: 0 for task in task_ids}
    for parent, child in edges:
        children[parent].append(child)
        indegree[child] += 1
    ready = sorted(task for task, count in indegree.items() if count == 0)
    observed: list[str] = []
    while ready:
        task = ready.pop(0)
        observed.append(task)
        for child in sorted(children[task]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    return len(observed) == len(task_ids)


def validate() -> dict[str, Any]:
    spec = _load_spec()
    seal = _read_json(SEAL_PATH)
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    check("schema", seal.get("schema") == "spar/dependency-source-seal@1", seal.get("schema"))
    claimed_cid = seal.get("seal_cid")
    unsealed = dict(seal)
    unsealed.pop("seal_cid", None)
    check("seal_cid", claimed_cid == spec.identity(unsealed), {"claimed": claimed_cid, "computed": spec.identity(unsealed)})

    task_ids = tuple(task.task_id for task in spec.TASKS)
    goal_ids = tuple(goal.goal_id for goal in spec.GOALS)
    expected_edges = tuple(sorted((dependency, task.task_id) for task in spec.TASKS for dependency in task.dependencies))
    observed_edges = tuple(tuple(item) for item in seal.get("dependency_edges", ()))
    check("task_sequence", task_ids == tuple(f"SPAR-{index:03d}" for index in range(51)), task_ids)
    check("goal_sequence", goal_ids[0] == "SPAR-G000" and len(set(goal_ids)) == 32, {"root": goal_ids[0], "count": len(goal_ids)})
    check("dependency_edges", observed_edges == expected_edges, {"expected_count": len(expected_edges), "observed_count": len(observed_edges)})
    check("dependency_root", seal.get("dependency_root_cid") == spec.identity(list(expected_edges)), seal.get("dependency_root_cid"))
    check("counts", seal.get("task_count") == 51 and seal.get("goal_count") == 32 and seal.get("dependency_count") == len(expected_edges), {key: seal.get(key) for key in ("task_count", "goal_count", "dependency_count")})
    known = set(task_ids)
    check("dependency_references", all(parent in known and child in known and parent < child for parent, child in expected_edges), "all dependencies are known and precede their consumers")
    check("dependency_acyclic", _acyclic(task_ids, expected_edges), "topological closure")
    roots = sorted(task for task in task_ids if not any(child == task for _, child in expected_edges))
    check("single_dependency_root", roots == ["SPAR-000"], roots)

    control_hashes = seal.get("control_file_sha256")
    if not isinstance(control_hashes, dict):
        check("control_hash_manifest", False, "absent")
    else:
        mismatches: list[str] = []
        for relative, claimed in sorted(control_hashes.items()):
            path = ROOT / relative
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != claimed:
                mismatches.append(relative)
        check("control_hash_manifest", not mismatches, mismatches)

    bindings = seal.get("source_binding") if isinstance(seal.get("source_binding"), dict) else {}
    outer = bindings.get("accelerator") if isinstance(bindings.get("accelerator"), dict) else {}
    check("accelerator_base_object", _git(ROOT, "cat-file", "-t", str(outer.get("commit") or "")) == "commit" and _git(ROOT, "rev-parse", f"{outer.get('commit')}^{{tree}}") == outer.get("tree"), outer)
    for name, relative in (("datasets", "ipfs_datasets_py"), ("kit", "ipfs_kit_py")):
        binding = bindings.get(name) if isinstance(bindings.get(name), dict) else {}
        nested = ROOT / relative
        try:
            clean = not _git(nested, "status", "--porcelain=v1", "--untracked-files=all")
            head = _git(nested, "rev-parse", "HEAD")
            tree = _git(nested, "rev-parse", "HEAD^{tree}")
            row = _git(ROOT, "ls-tree", str(outer.get("commit")), "--", relative).split()
            gitlink = len(row) >= 3 and row[0] == "160000" and row[2] == head
            passed = clean and head == binding.get("commit") and tree == binding.get("tree") and gitlink
            detail = {"clean": clean, "head": head, "tree": tree, "gitlink": gitlink}
        except ValidationError as exc:
            passed, detail = False, type(exc).__name__
        check(f"{name}_binding", passed, detail)

    check("historical_reuse_is_conditional", all("current-store" in rule or "historical" not in rule for rule in seal.get("rules", ())), seal.get("rules"))
    return {"schema": "spar/dependency-validation@1", "valid": not errors, "program": spec.PROGRAM, "task_count": len(task_ids), "goal_count": len(goal_ids), "dependency_count": len(expected_edges), "errors": errors, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args()
    try:
        report = validate()
    except Exception as exc:
        report = {"schema": "spar/dependency-validation@1", "valid": False, "errors": [type(exc).__name__], "checks": []}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
