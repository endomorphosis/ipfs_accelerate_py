#!/usr/bin/env python3
"""ASREF multi-lane launch: objective scan + implementation supervisors.

Wires the Agent Supervisor Module Refactor objective heap and todo board into a
multi-lane implementation supervisor configuration (ASREF-G100 / ASREF-010).

Protected architecture files are always fenced via
``--implementation-protected-path`` and are never rewritten by this tool:

- docs/architecture/agent_supervisor_module_refactor.objectives.md
- docs/architecture/agent_supervisor_module_refactor.todo.md
- docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md

Implementation provider (Grok 4.6 or successor) is selected at launch via
``IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER`` / ``--implementation-provider``.
Provider bridges remain in integrations/runtime; package moves must not wait on
provider choice.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (  # noqa: E402
    ImplementationSupervisorTrackConfig,
    build_configured_multi_supervisor_cli_runner,
    utc_run_stamp,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_file,
)


TODO_RELATIVE = Path("docs/architecture/agent_supervisor_module_refactor.todo.md")
OBJECTIVE_RELATIVE = Path(
    "docs/architecture/agent_supervisor_module_refactor.objectives.md"
)
PLAN_RELATIVE = Path("docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md")
BUNDLE_RELATIVE = Path("data/agent_supervisor/bundles/asref")
DISCOVERY_RELATIVE = Path("data/agent_supervisor/discovery/asref")
LAUNCH_RECIPE_RELATIVE = BUNDLE_RELATIVE / "launch_recipe.json"
PROTECTED_PATHS_RELATIVE = BUNDLE_RELATIVE / "protected_paths.json"
EVIDENCE_COVERAGE_RELATIVE = BUNDLE_RELATIVE / "evidence_coverage_asref_g100.md"
ENTRY_RELATIVE = Path(
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)

TASK_PREFIX = "ASREF-"
TASK_HEADER_PREFIX = f"## {TASK_PREFIX}"
PROTECTED_PATHS = (
    OBJECTIVE_RELATIVE.as_posix(),
    TODO_RELATIVE.as_posix(),
    PLAN_RELATIVE.as_posix(),
)
EVIDENCE_TERMS = (
    PLAN_RELATIVE.as_posix(),
    TODO_RELATIVE.as_posix(),
)
DEFAULT_NAMESPACE = "asref-v1"
DEFAULT_LANES = 4
DEFAULT_MERGE_BRANCH = "refactor/agent-supervisor-layout"
DEFAULT_REFILL_OPEN_TASK_THRESHOLD = 3
IMPLEMENTATION_PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
GROK_PROVIDER_ALIASES = frozenset(
    {
        "grok",
        "grok_cli",
        "grok-cli",
        "xai_cli",
        "xai-cli",
        "grok_build",
        "grok-build",
        "grok_build_cli",
        "grok-build-cli",
    }
)


def _runtime_root(namespace: str) -> Path:
    configured = os.environ.get("IPFS_ACCELERATE_AGENT_SUPERVISOR_ROOT", "").strip()
    base = (
        Path(configured).expanduser()
        if configured
        else Path.home()
        / ".local"
        / "share"
        / "ipfs_accelerate_py"
        / "agent-supervisor"
    )
    return base / namespace


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def inspect_board(
    *,
    repo_root: Path = REPO_ROOT,
    lanes: int = DEFAULT_LANES,
    merge_branch: str = DEFAULT_MERGE_BRANCH,
) -> dict[str, Any]:
    """Preflight the ASREF heap, board, protected paths, and bundle seed."""

    todo_path = repo_root / TODO_RELATIVE
    objective_path = repo_root / OBJECTIVE_RELATIVE
    plan_path = repo_root / PLAN_RELATIVE
    bundle_dir = repo_root / BUNDLE_RELATIVE
    errors: list[str] = []
    warnings: list[str] = []

    if not todo_path.is_file():
        errors.append(f"missing todo board: {todo_path}")
    if not objective_path.is_file():
        errors.append(f"missing objective heap: {objective_path}")
    if not plan_path.is_file():
        errors.append(f"missing plan: {plan_path}")
    if not bundle_dir.is_dir():
        errors.append(f"missing bundle dir: {bundle_dir}")

    for relative in PROTECTED_PATHS:
        if not (repo_root / relative).is_file():
            errors.append(f"protected path missing on disk: {relative}")

    seed_files = (
        LAUNCH_RECIPE_RELATIVE,
        PROTECTED_PATHS_RELATIVE,
        EVIDENCE_COVERAGE_RELATIVE,
        BUNDLE_RELATIVE / "lane_matrix.json",
        BUNDLE_RELATIVE / "seed_manifest.json",
        BUNDLE_RELATIVE / "asref-bootstrap.todo.md",
        BUNDLE_RELATIVE / "README.md",
        ENTRY_RELATIVE,
    )
    for relative in seed_files:
        if not (repo_root / relative).is_file():
            warnings.append(f"expected ASREF seed file missing: {relative.as_posix()}")

    protected_payload = _load_json(repo_root / PROTECTED_PATHS_RELATIVE)
    if protected_payload is not None:
        listed = {
            str(item).strip()
            for item in (protected_payload.get("protected_paths") or [])
            if str(item).strip()
        }
        for required in PROTECTED_PATHS:
            if required not in listed:
                errors.append(
                    f"protected_paths.json missing required path: {required}"
                )

    recipe = _load_json(repo_root / LAUNCH_RECIPE_RELATIVE)
    if recipe is not None:
        recipe_protected = {
            str(item).strip()
            for item in (recipe.get("protected_paths") or [])
            if str(item).strip()
        }
        for required in PROTECTED_PATHS:
            if required not in recipe_protected:
                errors.append(
                    f"launch_recipe.json missing protected path: {required}"
                )
        for term in EVIDENCE_TERMS:
            covered = {
                str(item).strip()
                for item in (recipe.get("evidence_terms_covered") or [])
                if str(item).strip()
            }
            if term not in covered:
                warnings.append(
                    f"launch_recipe.json evidence_terms_covered missing {term}"
                )

    branch = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    head_branch = (branch.stdout or "").strip()
    if head_branch and head_branch != merge_branch:
        warnings.append(
            f"repo HEAD is {head_branch!r}; expected {merge_branch!r} for ASREF"
        )

    tasks = (
        parse_task_file(todo_path, TASK_HEADER_PREFIX) if todo_path.is_file() else []
    )
    goals = (
        parse_goal_heap(objective_path.read_text(encoding="utf-8"))
        if objective_path.is_file()
        else []
    )
    open_tasks = [
        task
        for task in tasks
        if str(getattr(task, "status", "")).lower()
        not in {"completed", "done", "closed", "cancelled", "canceled"}
    ]
    completed_ids = {
        task.task_id
        for task in tasks
        if str(getattr(task, "status", "")).lower()
        in {"completed", "done", "closed"}
    }
    task_ids = {task.task_id for task in tasks}
    ready_loose: list[str] = []
    for task in open_tasks:
        deps = tuple(getattr(task, "depends_on", ()) or ())
        if all(dep in completed_ids or dep not in task_ids for dep in deps):
            ready_loose.append(task.task_id)

    if not open_tasks:
        warnings.append("no open ASREF tasks on the board")
    if open_tasks and not ready_loose:
        warnings.append("no ready ASREF tasks (all open tasks blocked)")

    return {
        "schema": "ipfs_accelerate_py.asref.preflight.v1",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "repo_root": str(repo_root),
        "head_branch": head_branch,
        "merge_branch": merge_branch,
        "task_count": len(tasks),
        "open_task_count": len(open_tasks),
        "ready_task_ids": ready_loose,
        "goal_count": len(goals),
        "lanes": lanes,
        "protected_paths": list(PROTECTED_PATHS),
        "evidence_terms_covered": list(EVIDENCE_TERMS),
        "bundle_dir": str(bundle_dir),
        "todo_path": str(todo_path),
        "objective_path": str(objective_path),
        "plan_path": str(plan_path),
        "implementation_provider_env": IMPLEMENTATION_PROVIDER_ENV,
        "implementation_provider": os.environ.get(
            IMPLEMENTATION_PROVIDER_ENV, ""
        ).strip()
        or None,
    }


def verify_evidence(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Prove ASREF-G100 missing evidence terms are wired into launch artifacts."""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for term in EVIDENCE_TERMS:
        path = repo_root / term
        present = path.is_file()
        checks.append(
            {
                "term": term,
                "kind": "path_exists",
                "ok": present,
                "detail": str(path),
            }
        )
        if not present:
            errors.append(f"evidence path missing: {term}")

    recipe = _load_json(repo_root / LAUNCH_RECIPE_RELATIVE) or {}
    covered = {
        str(item).strip()
        for item in (recipe.get("evidence_terms_covered") or [])
        if str(item).strip()
    }
    for term in EVIDENCE_TERMS:
        ok = term in covered
        checks.append(
            {
                "term": term,
                "kind": "launch_recipe_coverage",
                "ok": ok,
                "detail": LAUNCH_RECIPE_RELATIVE.as_posix(),
            }
        )
        if not ok:
            errors.append(f"launch_recipe does not cover evidence term: {term}")

    protected_payload = _load_json(repo_root / PROTECTED_PATHS_RELATIVE) or {}
    protected_listed = {
        str(item).strip()
        for item in (protected_payload.get("protected_paths") or [])
        if str(item).strip()
    }
    for term in PROTECTED_PATHS:
        ok = term in protected_listed
        checks.append(
            {
                "term": term,
                "kind": "protected_paths_list",
                "ok": ok,
                "detail": PROTECTED_PATHS_RELATIVE.as_posix(),
            }
        )
        if not ok:
            errors.append(f"protected_paths.json missing: {term}")

    coverage_doc = repo_root / EVIDENCE_COVERAGE_RELATIVE
    coverage_text = (
        coverage_doc.read_text(encoding="utf-8") if coverage_doc.is_file() else ""
    )
    for term in EVIDENCE_TERMS:
        ok = term in coverage_text
        checks.append(
            {
                "term": term,
                "kind": "evidence_coverage_doc",
                "ok": ok,
                "detail": EVIDENCE_COVERAGE_RELATIVE.as_posix(),
            }
        )
        if not ok:
            errors.append(f"evidence coverage doc missing term: {term}")

    # Launcher source itself must emit protected-path flags for plan + todo.
    launcher_src = Path(__file__).read_text(encoding="utf-8")
    for term in EVIDENCE_TERMS:
        ok = term in launcher_src or term.split("/")[-1] in launcher_src
        checks.append(
            {
                "term": term,
                "kind": "launcher_source_reference",
                "ok": ok,
                "detail": "scripts/ops/agent_supervisor/asref_multi_lane.py",
            }
        )
        if not ok:
            errors.append(f"launcher source does not reference evidence term: {term}")

    return {
        "schema": "ipfs_accelerate_py.asref.evidence_verify.v1",
        "goal_id": "ASREF-G100",
        "task_id": "ASREF-010",
        "ok": not errors,
        "errors": errors,
        "checks": checks,
        "evidence_terms": list(EVIDENCE_TERMS),
        "protected_paths": list(PROTECTED_PATHS),
    }


def _common_args(
    *,
    repo_root: Path,
    runtime_root: Path,
    merge_branch: str,
    enable_objective_refill: bool,
    refill_open_task_threshold: int,
) -> tuple[str, ...]:
    python = sys.executable
    bundle_dir = repo_root / BUNDLE_RELATIVE
    discovery_dir = repo_root / DISCOVERY_RELATIVE
    args: list[str] = [
        "--todo-path",
        str(repo_root / TODO_RELATIVE),
        "--task-prefix",
        TASK_PREFIX,
        "--worktree-root",
        str(runtime_root / "worktrees"),
        "--merge-target-branch",
        merge_branch,
        "--stale-seconds",
        "1800",
        "--check-interval",
        "15",
        "--watchdog-startup-grace-seconds",
        "600",
        "--max-restarts",
        "50",
        "--max-task-attempts",
        "5",
        "--daemon-interval",
        "20",
        "--implementation-timeout",
        "3600",
        "--implementation-log-stall-seconds",
        "1200",
        "--llm-merge-resolver-command",
        f"{python} -m ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback",
        "--llm-merge-resolver-timeout-seconds",
        "1800",
        "--worktree-reconciliation-max-merges",
        "1",
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--implement",
        "--log-level",
        "INFO",
    ]
    for protected in PROTECTED_PATHS:
        args.extend(["--implementation-protected-path", protected])
    if enable_objective_refill:
        discovery_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(repo_root / OBJECTIVE_RELATIVE),
                "--objective-graph-path",
                str(runtime_root / "objective_graph.json"),
                "--objective-bundle-dir",
                str(bundle_dir),
                "--objective-dataset-dir",
                str(runtime_root / "datasets"),
                "--objective-discovery-dir",
                str(discovery_dir),
                "--objective-todo-vector-index-path",
                str(bundle_dir / "todo_vector_index.json"),
                "--objective-scan-min-open-tasks",
                str(refill_open_task_threshold),
                "--objective-scan-max-findings",
                "4",
                "--objective-scan-cooldown-seconds",
                "900",
                "--objective-refill-timeout-seconds",
                "600",
                "--objective-goal-prefix",
                "ASREF-G",
                "--objective-root-goal-id",
                "ASREF-G000",
                "--objective-root-goal-title",
                "Clear agent_supervisor package layout and monorepo root hygiene",
                "--objective-tracking-document-title",
                "Agent Supervisor Module Refactor Objective Heap",
                "--objective-mission-term",
                "agent_supervisor,package,refactor,import,layout,module,readme",
            ]
        )
    return tuple(args)


def _objective_scan_argv(
    *,
    repo_root: Path,
    refine: bool,
    max_findings: int,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon",
        "--objective-path",
        str(repo_root / OBJECTIVE_RELATIVE),
        "--todo-path",
        str(repo_root / TODO_RELATIVE),
        "--discovery-dir",
        str(repo_root / DISCOVERY_RELATIVE),
        "--objective-bundle-dir",
        str(repo_root / BUNDLE_RELATIVE),
        "--max-findings",
        str(max_findings),
        "--no-reconcile-goal-completion",
    ]
    if refine:
        argv.append("--refine-objective-heap")
    return argv


def run_objective_scan(args: argparse.Namespace) -> int:
    report = inspect_board(
        repo_root=REPO_ROOT,
        lanes=args.lanes,
        merge_branch=args.merge_branch,
    )
    # Objective scan only needs heap + board + bundle dir.
    hard_errors = [
        err
        for err in report["errors"]
        if "missing todo board" in err
        or "missing objective heap" in err
        or "missing bundle dir" in err
    ]
    if hard_errors and not args.force:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    (REPO_ROOT / BUNDLE_RELATIVE).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / DISCOVERY_RELATIVE).mkdir(parents=True, exist_ok=True)

    argv = _objective_scan_argv(
        repo_root=REPO_ROOT,
        refine=bool(args.refine_objective_heap),
        max_findings=int(args.max_findings),
    )
    payload = {
        "schema": "ipfs_accelerate_py.asref.objective_scan.v1",
        "argv": argv,
        "bundle_dir": str(REPO_ROOT / BUNDLE_RELATIVE),
        "todo_path": str(REPO_ROOT / TODO_RELATIVE),
        "objective_path": str(REPO_ROOT / OBJECTIVE_RELATIVE),
        "plan_path": str(REPO_ROOT / PLAN_RELATIVE),
        "protected_paths": list(PROTECTED_PATHS),
        "evidence_terms_covered": list(EVIDENCE_TERMS),
        "refine_objective_heap": bool(args.refine_objective_heap),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return int(subprocess.call(argv, cwd=str(REPO_ROOT)))


def launch(args: argparse.Namespace) -> int:
    report = inspect_board(
        repo_root=REPO_ROOT,
        lanes=args.lanes,
        merge_branch=args.merge_branch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"] and not args.force:
        return 2

    provider = str(getattr(args, "implementation_provider", "") or "").strip()
    if provider:
        os.environ[IMPLEMENTATION_PROVIDER_ENV] = provider
        if provider.lower() in GROK_PROVIDER_ALIASES:
            # Preferred ASREF-G100 path; still does not mutate goal/todo text.
            pass

    runtime_root = _runtime_root(args.namespace)
    for sub in (
        "master",
        "state",
        "worktrees",
        "bundles",
        "discovery",
        "datasets",
    ):
        (runtime_root / sub).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / BUNDLE_RELATIVE).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / DISCOVERY_RELATIVE).mkdir(parents=True, exist_ok=True)

    stamp = utc_run_stamp()
    entry = REPO_ROOT / ENTRY_RELATIVE
    if not entry.is_file():
        print(
            json.dumps({"error": f"missing entry script: {entry}"}),
            file=sys.stderr,
        )
        return 2

    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=REPO_ROOT,
        duration_seconds=args.duration_seconds,
        heartbeat_interval_seconds=30,
        supervisor_status_stale_seconds=300,
        stop_grace_seconds=900,
        stamp=stamp,
        master_dir=runtime_root / "master",
        master_log=runtime_root / "master" / f"asref_{stamp}.log",
        master_pid_path=runtime_root / "master" / "asref.pid",
        label="asref-module-refactor",
        python_executable=sys.executable,
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="asref-module-refactor",
                script_path=entry,
                state_dir=runtime_root / "state",
                state_prefix="asref",
            ),
        ),
        common_args=_common_args(
            repo_root=REPO_ROOT,
            runtime_root=runtime_root,
            merge_branch=args.merge_branch,
            enable_objective_refill=args.enable_objective_refill,
            refill_open_task_threshold=args.refill_open_task_threshold,
        ),
        detach=not args.foreground,
    )
    lane_args = [
        "--implementation-supervisor-lanes-per-track",
        str(args.lanes),
    ]
    payload = {
        "schema": "ipfs_accelerate_py.asref.launch.v1",
        "argv": [*runner.args(), *lane_args],
        "runtime_root": str(runtime_root),
        "bundle_dir": str(REPO_ROOT / BUNDLE_RELATIVE),
        "stamp": stamp,
        "master_log": str(runtime_root / "master" / f"asref_{stamp}.log"),
        "master_pid_path": str(runtime_root / "master" / "asref.pid"),
        "protected_paths": list(PROTECTED_PATHS),
        "evidence_terms_covered": list(EVIDENCE_TERMS),
        "implementation_provider": os.environ.get(
            IMPLEMENTATION_PROVIDER_ENV, ""
        ).strip()
        or None,
        "todo_path": str(REPO_ROOT / TODO_RELATIVE),
        "objective_path": str(REPO_ROOT / OBJECTIVE_RELATIVE),
        "plan_path": str(REPO_ROOT / PLAN_RELATIVE),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return int(runner.run(lane_args))


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--merge-branch", default=DEFAULT_MERGE_BRANCH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if preflight reports errors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned argv without starting processes",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ASREF multi-lane preflight, objective scan, and implementation "
            "supervisor launch (Grok 4.6 selectable via provider env/flag)"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser(
        "preflight",
        help="Validate heap, board, protected paths, and bundle seed",
    )
    _add_shared_flags(preflight)

    verify = sub.add_parser(
        "verify-evidence",
        help="Prove ASREF-G100 plan/todo evidence terms are wired",
    )
    _add_shared_flags(verify)

    scan = sub.add_parser(
        "objective-scan",
        help="Run objective daemon against the ASREF heap into the todo board",
    )
    _add_shared_flags(scan)
    scan.add_argument(
        "--refine-objective-heap",
        action="store_true",
        help="Ask the objective daemon to refine broad goals into children",
    )
    scan.add_argument(
        "--max-findings",
        type=int,
        default=12,
        help="Max objective findings to materialize as todos",
    )

    launch_p = sub.add_parser(
        "launch",
        help="Launch multi-lane implementation supervisors for ASREF",
    )
    _add_shared_flags(launch_p)
    launch_p.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
        help="Master supervisor duration (default: inf)",
    )
    launch_p.add_argument(
        "--enable-objective-refill",
        action="store_true",
        help="Enable objective-heap refill into the ASREF board",
    )
    launch_p.add_argument(
        "--refill-open-task-threshold",
        type=int,
        default=DEFAULT_REFILL_OPEN_TASK_THRESHOLD,
    )
    launch_p.add_argument(
        "--foreground",
        action="store_true",
        help="Do not detach the multi-supervisor master",
    )
    launch_p.add_argument(
        "--implementation-provider",
        default=os.environ.get(IMPLEMENTATION_PROVIDER_ENV, ""),
        help=(
            f"Set {IMPLEMENTATION_PROVIDER_ENV} for child daemons "
            "(e.g. grok, goose, codex, auto). Prefer grok for ASREF-G100."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "preflight":
        report = inspect_board(
            repo_root=REPO_ROOT,
            lanes=args.lanes,
            merge_branch=args.merge_branch,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2

    if args.command == "verify-evidence":
        report = verify_evidence(repo_root=REPO_ROOT)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2

    if args.command == "objective-scan":
        return run_objective_scan(args)

    if args.command == "launch":
        return launch(args)

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
