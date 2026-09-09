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


def _validate_outer_source_binding(
    *,
    root: Path,
    binding: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Bind the current superproject to the sealed planning ancestry."""

    sealed_commit = str(binding.get("commit") or "")
    sealed_tree = str(binding.get("tree") or "")
    try:
        current_commit = _git(root, "rev-parse", "HEAD")
        object_is_commit = _git(root, "cat-file", "-t", sealed_commit) == "commit"
        observed_tree = _git(root, "rev-parse", f"{sealed_commit}^{{tree}}")
        ancestry = subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                sealed_commit,
                current_commit,
            ),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError, ValidationError):
        return False, {
            **binding,
            "current_head": "",
            "planning_object_is_commit": False,
            "planning_tree_exact": False,
            "planning_is_ancestor": False,
        }
    detail = {
        **binding,
        "current_head": current_commit,
        "planning_object_is_commit": object_is_commit,
        "planning_tree_exact": observed_tree == sealed_tree,
        "planning_is_ancestor": ancestry.returncode == 0,
    }
    return bool(
        detail["planning_object_is_commit"]
        and detail["planning_tree_exact"]
        and detail["planning_is_ancestor"]
    ), detail


def _validate_nested_source_binding(
    *,
    root: Path,
    outer_planning_commit: str,
    relative: str,
    binding: dict[str, Any],
    permit_descendants: bool = True,
) -> tuple[bool, dict[str, Any] | str]:
    """Verify immutable planning identity and the current descendant frame.

    The dependency seal binds the SPAR-000 planning commit.  Accepted worker
    merges may advance a nested gitlink, so the live checkout is a separate
    frame: it must be clean, exactly recorded by the current superproject, and
    descend from the sealed planning commit.  Advancing the seal itself would
    silently rewrite the program's source lineage.
    """

    nested = root / relative
    try:
        planning_commit = str(binding.get("commit") or "")
        planning_tree = str(binding.get("tree") or "")
        planning_object_is_commit = (
            _git(nested, "cat-file", "-t", planning_commit) == "commit"
        )
        observed_planning_tree = _git(
            nested,
            "rev-parse",
            f"{planning_commit}^{{tree}}",
        )
        planning_row = _git(
            root,
            "ls-tree",
            outer_planning_commit,
            "--",
            relative,
        )
        planning_gitlink = planning_row == (
            f"160000 commit {planning_commit}\t{relative}"
        )

        clean = not _git(
            nested,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        current_head = _git(nested, "rev-parse", "HEAD")
        current_tree = _git(nested, "rev-parse", "HEAD^{tree}")
        current_outer_head = _git(root, "rev-parse", "HEAD")
        current_row = _git(
            root,
            "ls-tree",
            current_outer_head,
            "--",
            relative,
        )
        current_gitlink = current_row == (
            f"160000 commit {current_head}\t{relative}"
        )
        ancestry = subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                planning_commit,
                current_head,
            ),
            cwd=nested,
            capture_output=True,
            check=False,
            timeout=60,
        )
        planning_is_ancestor = ancestry.returncode == 0
        planning_tree_exact = observed_planning_tree == planning_tree
        passed = bool(
            planning_object_is_commit
            and planning_tree_exact
            and planning_gitlink
            and clean
            and current_head
            and current_tree
            and current_gitlink
            and planning_is_ancestor
            and (permit_descendants or current_head == planning_commit)
        )
        return passed, {
            "clean": clean,
            "planning_commit": planning_commit,
            "planning_tree": observed_planning_tree,
            "planning_tree_exact": planning_tree_exact,
            "planning_gitlink": planning_gitlink,
            "planning_is_ancestor": planning_is_ancestor,
            "advancement_policy": (
                "accepted_descendants"
                if permit_descendants
                else "exact_read_only_pin"
            ),
            "current_is_exact_planning_revision": (
                current_head == planning_commit
            ),
            "current_head": current_head,
            "current_tree": current_tree,
            "current_gitlink": current_gitlink,
        }
    except (OSError, subprocess.SubprocessError, ValidationError) as exc:
        return False, type(exc).__name__


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

    runtime_hashes = seal.get("bootstrap_runtime_file_sha256")
    if not isinstance(runtime_hashes, dict) or not runtime_hashes:
        check("bootstrap_runtime_hash_manifest", False, "absent")
    else:
        mismatches = []
        for relative, claimed in sorted(runtime_hashes.items()):
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != claimed
            ):
                mismatches.append(relative)
        check("bootstrap_runtime_hash_manifest", not mismatches, mismatches)

    bindings = seal.get("source_binding") if isinstance(seal.get("source_binding"), dict) else {}
    outer = bindings.get("accelerator") if isinstance(bindings.get("accelerator"), dict) else {}
    accelerator_base_valid, accelerator_base_detail = (
        _validate_outer_source_binding(root=ROOT, binding=outer)
    )
    check(
        "accelerator_base_object",
        accelerator_base_valid,
        accelerator_base_detail,
    )
    for name, relative, permit_descendants in (
        ("datasets", "ipfs_datasets_py", True),
        ("kit", "ipfs_kit_py", True),
        ("mcp_plus_plus", "ipfs_accelerate_py/mcplusplus", False),
    ):
        binding = bindings.get(name) if isinstance(bindings.get(name), dict) else {}
        passed, detail = _validate_nested_source_binding(
            root=ROOT,
            outer_planning_commit=str(outer.get("commit") or ""),
            relative=relative,
            binding=binding,
            permit_descendants=permit_descendants,
        )
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
