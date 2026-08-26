#!/usr/bin/env python3
"""Materialize truthful PGIR-211 evidence from immutable revisions and live runs."""
from __future__ import annotations

import argparse
import base64
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

SOURCE_ROOT = Path(__file__).resolve().parents[6]
EVIDENCE_DIR = Path(__file__).resolve().parent
SAFE_PYTHONPATH = "/home/barberb/.local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages"
SAFE_SYS_PATH_TAIL = (
    "/home/barberb/.local/lib/python3.12/site-packages",
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
GIT_ENVIRONMENT_CONTROLS = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_COUNT": "0",
    "GIT_TERMINAL_PROMPT": "0",
}
BASE_SUBPROCESS_ENVIRONMENT = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
PYTHON_SUBPROCESS_ENVIRONMENT = {
    **BASE_SUBPROCESS_ENVIRONMENT,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONPATH": SAFE_PYTHONPATH,
}
EXECUTION_ENVIRONMENT_CONTROLS = {
    "environment_mode": "exact minimal environment; no inherited variables",
    "git": GIT_ENVIRONMENT_CONTROLS,
    "curl_configuration": "every curl argv begins /usr/bin/curl --disable",
    "home_and_xdg_variables_present": False,
    "proxy_and_credential_variables_present": False,
}


def require_safe_startup() -> dict[str, Any]:
    if sys.executable != "/usr/bin/python3.12" or sys.flags.no_site != 1 or sys.flags.no_user_site != 1:
        raise RuntimeError("capture requires /usr/bin/python3.12 -S with PYTHONNOUSERSITE=1")
    if os.environ.get("PYTHONPATH") != SAFE_PYTHONPATH or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1" or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("capture startup environment boundary drift")
    if "PYTHONHOME" in os.environ or "PYTEST_ADDOPTS" in os.environ:
        raise RuntimeError("capture inherited prohibited Python/pytest controls")
    if Path(sys.path[0] or os.getcwd()).resolve() != EVIDENCE_DIR or tuple(sys.path[1:]) != SAFE_SYS_PATH_TAIL:
        raise RuntimeError("capture exact sys.path boundary drift")
    meta_path = [f"{finder.__module__}.{finder.__name__}" for finder in sys.meta_path if isinstance(finder, type)]
    if meta_path != ["_frozen_importlib.BuiltinImporter", "_frozen_importlib.FrozenImporter", "_frozen_importlib_external.PathFinder"]:
        raise RuntimeError("capture meta_path contains an injected finder")
    forbidden = [name for name in sys.modules if name in {"site", "sitecustomize", "usercustomize", "aae_mcplusplus_validators_bootstrap"} or name.startswith("__editable__")]
    if forbidden:
        raise RuntimeError("capture loaded a site/bootstrap/editable hook")
    return {
        "executable": sys.executable,
        "no_site": True,
        "no_user_site": True,
        "pythonpath": SAFE_PYTHONPATH,
        "sys_path_tail": list(SAFE_SYS_PATH_TAIL),
        "meta_path": meta_path,
        "forbidden_modules_loaded": [],
        "entrypoint_directory": "data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/integrated-acceptance",
        "shared_package_roots_are_raw_not_pth_processed": True,
        "hermetic": False,
    }


CAPTURE_STARTUP = require_safe_startup()
VERIFIER_PATH = SOURCE_ROOT / "scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py"
SPEC = importlib.util.spec_from_file_location("pgir211_verifier", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import PGIR-211 verifier")
V = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V
SPEC.loader.exec_module(V)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stream_record(data: bytes, *, compress: bool = False) -> dict[str, Any]:
    row = V.identity(data)
    if compress:
        row["gzip_base64"] = base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")
    else:
        row["utf8"] = data.decode("utf-8")
    return row


def execute(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    compress_stdout: bool = False,
    timeout: int = 900,
) -> dict[str, Any]:
    command_environment = dict(env) if env is not None else dict(BASE_SUBPROCESS_ENVIRONMENT)
    expected_environment = PYTHON_SUBPROCESS_ENVIRONMENT if list(argv)[:1] == ["/usr/bin/python3.12"] else BASE_SUBPROCESS_ENVIRONMENT
    if command_environment != expected_environment:
        raise RuntimeError("subprocess environment is not the exact role-specific minimal environment")
    started = utc_now()
    process = subprocess.run(
        list(argv), cwd=cwd, env=command_environment,
        capture_output=True, timeout=timeout, check=False,
    )
    ended = utc_now()
    return {
        "argv": list(argv),
        "cwd": str(cwd.resolve()),
        "started_at_utc": started,
        "ended_at_utc": ended,
        "exit_code": process.returncode,
        "environment_controls": EXECUTION_ENVIRONMENT_CONTROLS,
        "environment": command_environment,
        "stdout": stream_record(process.stdout, compress=compress_stdout),
        "stderr": stream_record(process.stderr),
    }


def require_success(execution: Mapping[str, Any], label: str) -> None:
    if execution["exit_code"] != 0:
        raise RuntimeError(f"{label} failed: {execution['stderr'].get('utf8', '')}")


def raise_primary_or_cleanup(primary: BaseException | None, cleanup_failures: Sequence[str]) -> None:
    if cleanup_failures:
        cleanup_message = "; ".join(cleanup_failures)
        if primary is not None:
            raise RuntimeError(
                f"primary failure ({type(primary).__name__}): {primary}; cleanup failure(s): {cleanup_message}"
            ) from primary
        raise RuntimeError(f"cleanup failure(s): {cleanup_message}")
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)


def worktree_registration(repository: Path, target: Path) -> dict[str, Any]:
    row = execute(
        ["/usr/bin/git", "-C", str(repository), "worktree", "list", "--porcelain", "-z"],
        cwd=repository,
    )
    require_success(row, f"worktree registration query for {target}")
    target_resolved = target.resolve()
    paths = [
        Path(field.removeprefix("worktree ")).resolve()
        for field in row["stdout"]["utf8"].split("\0")
        if field.startswith("worktree ")
    ]
    row.update(
        target_path=str(target_resolved),
        registered=target_resolved in paths,
    )
    return row


def write_json(name: str, value: Mapping[str, Any]) -> None:
    (EVIDENCE_DIR / name).write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def controlled_python_environment() -> tuple[dict[str, str], dict[str, str]]:
    controls = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": SAFE_PYTHONPATH,
    }
    return dict(PYTHON_SUBPROCESS_ENVIRONMENT), controls


def inspect_checkout(path: Path) -> dict[str, Any]:
    nested = path / "ipfs_datasets_py"
    commands = {
        "outer_head": (["/usr/bin/git", "rev-parse", "HEAD"], path),
        "outer_tree": (["/usr/bin/git", "rev-parse", "HEAD^{tree}"], path),
        "outer_gitlink": (["/usr/bin/git", "ls-tree", "HEAD", "ipfs_datasets_py"], path),
        "outer_status": (["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"], path),
        "nested_head": (["/usr/bin/git", "rev-parse", "HEAD"], nested),
        "nested_tree": (["/usr/bin/git", "rev-parse", "HEAD^{tree}"], nested),
        "nested_status": (["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"], nested),
    }
    rows = {name: execute(argv, cwd=cwd) for name, (argv, cwd) in commands.items()}
    for name, row in rows.items():
        require_success(row, f"checkout inspection {name}")
    return rows


@contextmanager
def clean_target_checkout(label: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    parent = Path(tempfile.mkdtemp(prefix=f"pgir211-{label}-"))
    outer = parent / "outer"
    nested = outer / "ipfs_datasets_py"
    source_repositories = {"outer": str(SOURCE_ROOT), "nested": str(SOURCE_ROOT / "ipfs_datasets_py")}
    outer_registered = False
    nested_registered = False
    primary_failure: BaseException | None = None
    checkout: dict[str, Any] = {
        "fresh": True,
        "initialized_submodules": ["ipfs_datasets_py"],
        "recursive_submodule_update": False,
        "repository_forest_complete_for_task": True,
        "removed_after_capture": False,
        "path": str(outer),
        "nested_path": str(nested),
        "source_repositories": source_repositories,
    }
    try:
        checkout["precreation"] = {
            "outer": execute(["/usr/bin/test", "!", "-e", str(outer)], cwd=parent),
            "nested": execute(["/usr/bin/test", "!", "-e", str(nested)], cwd=parent),
        }
        for role, row in checkout["precreation"].items():
            require_success(row, f"{role} precreation absence")
        outer_create = execute(
            ["/usr/bin/git", "-C", str(SOURCE_ROOT), "worktree", "add", "--detach", str(outer), V.TARGET],
            cwd=SOURCE_ROOT,
        )
        checkout["creation"] = {"outer": outer_create}
        checkout["registration_after_creation"] = {
            "outer": worktree_registration(SOURCE_ROOT, outer),
        }
        outer_registered = checkout["registration_after_creation"]["outer"]["registered"]
        if not outer_registered:
            raise RuntimeError("outer target worktree was not registered after creation attempt")
        require_success(outer_create, "outer target worktree creation")
        nested_prepare = execute(["/usr/bin/rmdir", str(nested)], cwd=outer)
        checkout["nested_path_preparation"] = nested_prepare
        require_success(nested_prepare, "nested empty gitlink path removal")
        nested_absent = execute(["/usr/bin/test", "!", "-e", str(nested)], cwd=outer)
        checkout["nested_absence_after_preparation"] = nested_absent
        require_success(nested_absent, "nested post-preparation absence")
        nested_create = execute(
            ["/usr/bin/git", "-C", str(SOURCE_ROOT / "ipfs_datasets_py"), "worktree", "add", "--detach", str(nested), V.CURRENT],
            cwd=SOURCE_ROOT / "ipfs_datasets_py",
        )
        checkout["creation"]["nested"] = nested_create
        checkout["registration_after_creation"]["nested"] = worktree_registration(
            SOURCE_ROOT / "ipfs_datasets_py", nested,
        )
        nested_registered = checkout["registration_after_creation"]["nested"]["registered"]
        if not nested_registered:
            raise RuntimeError("nested target worktree was not registered after creation attempt")
        require_success(nested_create, "nested target worktree creation")
        checkout["before"] = inspect_checkout(outer)
        yield outer, checkout
        checkout["after"] = inspect_checkout(outer)
    except BaseException as exc:
        primary_failure = exc

    cleanup_failures: list[str] = []
    if "after" not in checkout and outer_registered and nested_registered and outer.exists() and nested.exists():
        try:
            checkout["after"] = inspect_checkout(outer)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"final checkout inspection: {exc}")
    removal: dict[str, Any] = {}
    registration_before_removal: dict[str, Any] = {}
    registration_after_removal: dict[str, Any] = {}
    try:
        nested_before_removal = worktree_registration(SOURCE_ROOT / "ipfs_datasets_py", nested)
        registration_before_removal["nested"] = nested_before_removal
        nested_registered = nested_before_removal["registered"]
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        cleanup_failures.append(f"nested registration inspection before removal: {exc}")
    nested_cleanup_attempted = nested_registered or nested.exists()
    if nested_registered:
        try:
            nested_remove = execute(
                ["/usr/bin/git", "-C", str(SOURCE_ROOT / "ipfs_datasets_py"), "worktree", "remove", str(nested)],
                cwd=SOURCE_ROOT / "ipfs_datasets_py",
            )
            removal["nested"] = nested_remove
            if nested_remove["exit_code"] != 0:
                cleanup_failures.append("nested target worktree ordinary removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"nested target worktree ordinary removal execution: {exc}")
    elif nested.exists():
        try:
            nested_remove = execute(["/usr/bin/rm", "-r", "--", str(nested)], cwd=parent)
            removal["nested"] = nested_remove
            if nested_remove["exit_code"] != 0:
                cleanup_failures.append("nested unregistered partial checkout removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"nested unregistered partial checkout removal execution: {exc}")
    if nested_cleanup_attempted:
        try:
            registration_after_removal["nested"] = worktree_registration(SOURCE_ROOT / "ipfs_datasets_py", nested)
            nested_registered = registration_after_removal["nested"]["registered"]
            if nested_registered:
                cleanup_failures.append("nested target worktree remained registered after removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"nested registration inspection after removal: {exc}")
    try:
        outer_before_removal = worktree_registration(SOURCE_ROOT, outer)
        registration_before_removal["outer"] = outer_before_removal
        outer_registered = outer_before_removal["registered"]
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        cleanup_failures.append(f"outer registration inspection before removal: {exc}")
    if outer_registered and not nested_registered and not nested.exists():
        try:
            nested_restoration = execute(["/usr/bin/mkdir", "--", str(nested)], cwd=outer)
            checkout["nested_path_restoration"] = nested_restoration
            if nested_restoration["exit_code"] != 0:
                cleanup_failures.append("nested empty gitlink path restoration")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"nested empty gitlink path restoration execution: {exc}")
    if outer_registered and nested.exists():
        try:
            outer_status_after_restoration = execute(
                ["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"],
                cwd=outer,
            )
            checkout["outer_status_after_nested_restoration"] = outer_status_after_restoration
            if outer_status_after_restoration["exit_code"] != 0 or outer_status_after_restoration["stdout"]["utf8"]:
                cleanup_failures.append("outer checkout was not clean after nested gitlink path restoration")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"outer status after nested gitlink path restoration execution: {exc}")
    outer_cleanup_attempted = outer_registered or outer.exists()
    if outer_registered:
        try:
            outer_remove = execute(
                ["/usr/bin/git", "-C", str(SOURCE_ROOT), "worktree", "remove", str(outer)],
                cwd=SOURCE_ROOT,
            )
            removal["outer"] = outer_remove
            if outer_remove["exit_code"] != 0:
                cleanup_failures.append("outer target worktree ordinary removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"outer target worktree ordinary removal execution: {exc}")
    elif outer.exists():
        try:
            outer_remove = execute(["/usr/bin/rm", "-r", "--", str(outer)], cwd=parent)
            removal["outer"] = outer_remove
            if outer_remove["exit_code"] != 0:
                cleanup_failures.append("outer unregistered partial checkout removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"outer unregistered partial checkout removal execution: {exc}")
    if outer_cleanup_attempted:
        try:
            registration_after_removal["outer"] = worktree_registration(SOURCE_ROOT, outer)
            outer_registered = registration_after_removal["outer"]["registered"]
            if outer_registered:
                cleanup_failures.append("outer target worktree remained registered after removal")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            cleanup_failures.append(f"outer registration inspection after removal: {exc}")
    checkout["removal"] = removal
    checkout["registration_before_removal"] = registration_before_removal
    checkout["registration_after_removal"] = registration_after_removal
    checkout["removed_after_capture"] = not outer.exists() and not nested.exists() and not outer_registered and not nested_registered
    if checkout["removed_after_capture"] and parent.exists():
        try:
            parent.rmdir()
        except OSError as exc:
            cleanup_failures.append(f"temporary checkout parent removal: {exc}")
    elif parent.exists():
        cleanup_failures.append("temporary checkout parent retained because checkout removal was incomplete")
    raise_primary_or_cleanup(primary_failure, cleanup_failures)


def build_historical() -> None:
    started = utc_now()
    campaign = V.git_json(
        V.ROOT, V.TOKENIZER_REVISION,
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json",
    )
    campaign_root = V.record_for(
        "data/agent_supervisor/proof_grounded_ir_learning/freeze/campaign_input_root.json",
        V.ROOT, V.TOKENIZER_REVISION,
    )
    campaign_root.update(root_sha256=campaign["root_sha256"], root_cid=campaign["root_cid"])
    sealed = []
    for path in V.SEALED_PATHS:
        row = V.record_for(path, V.DATASETS, V.SEALED, path)
        row["current_git_blob"] = V.git(V.DATASETS, "rev-parse", f"{V.CURRENT}:{path}")
        sealed.append(row)
    r1_files = []
    for name, _self_field in V.R1_FILES:
        relative = f"data/ir_learning/evaluations/deterministic/{name}"
        row = V.record_for("ipfs_datasets_py/" + relative, V.DATASETS, V.R1_REVISION, relative)
        row["current_git_blob"] = V.git(V.DATASETS, "rev-parse", f"{V.CURRENT}:{relative}")
        r1_files.append(row)
    retirement_files = []
    for name in V.RETIREMENT_FILES:
        relative = f"data/ir_learning/evaluations/deterministic/successor-v1/{name}"
        row = V.record_for("ipfs_datasets_py/" + relative, V.DATASETS, V.CURRENT, relative)
        row.update(task_id="PGIR-204", result_identity="RESULT(PGIR-204)")
        retirement_files.append(row)
    closure: dict[str, Any] = {
        "schema": "proof-grounded-ir-learning/integrated-historical-closure/v2",
        "task_id": "PGIR-211",
        "capture_startup_environment": CAPTURE_STARTUP,
        "sources": V.expected_sources(),
        "captured_at_utc": utc_now(),
        "target": V.target_identity(),
        "predecessor_files": V.expected_predecessor_records(),
        "task_identity_sources": V.expected_task_identity_source_records(),
        "integrated_forest": {
            "target": {"commit": V.TARGET, "tree": V.TARGET_TREE, "gitlink": V.CURRENT},
            "outer_commits": V.expected_outer_forest(),
            "compare_and_swap": [
                {"task_id": task, "old_gitlink": old, "new_gitlink": new, "implementation": implementation, "merge": merge, "completion": completion}
                for task, old, new, implementation, merge, completion in V.CAS
            ],
        },
        "sealed_successor_files": sealed,
        "r1_files": r1_files,
        "retirement_files": retirement_files,
        "campaign_root": campaign_root,
        "campaign_bindings": V.expected_campaign_bindings(campaign),
        "campaign_objective_source": V.record_for(
            "docs/architecture/proof_grounded_ir_learning.todo.md",
            V.ROOT,
            V.CAMPAIGN_OUTER_REVISION,
        ),
    }
    pg208, pg209, pg210 = V.verify_predecessors(closure)
    V.verify_forest(closure, pg208, pg210)
    closure["replay_summary"] = {
        "r1": V.verify_r1(closure),
        "retirement": V.verify_retirement(closure, pg209),
        "campaign": V.verify_campaign(closure),
        "successor": V.verify_successor_population(),
    }
    closure["observed_start_utc"] = started
    closure["observed_end_utc"] = utc_now()
    write_json("historical_closure_receipt.json", closure)


def build_tests() -> None:
    started = utc_now()
    with clean_target_checkout("tests") as (outer, checkout):
        env, controlled_environment = controlled_python_environment()
        runtime = {
            "python": execute(["/usr/bin/python3.12", "-S", "--version"], cwd=outer, env=env),
            "pytest": execute(["/usr/bin/python3.12", "-S", "-m", "pytest", "--version"], cwd=outer, env=env),
            "target_import": execute(["/usr/bin/python3.12", "-S", "-c", V.TARGET_IMPORT_PROBE], cwd=outer / "ipfs_datasets_py", env=env),
        }
        collections = []
        executions = []
        populations = (
            ("focused_34", V.FOCUSED_COLLECT_ARGV, V.FOCUSED_TEST_ARGV, 34),
            ("supplementary_3", V.SUPPLEMENTARY_COLLECT_ARGV, V.SUPPLEMENTARY_TEST_ARGV, 3),
        )
        for role, collect_argv, run_argv, count in populations:
            collection = execute(collect_argv, cwd=outer, env=env)
            require_success(collection, role + " collection")
            node_ids = V.collected_node_ids(collection["stdout"]["utf8"], count, role + " collection", str(outer))
            if len(node_ids) != count:
                raise RuntimeError(f"{role} collected {len(node_ids)}, expected {count}")
            node_id_set = {"count": count, **V.identity("".join(f"{node_id}\n" for node_id in node_ids).encode())}
            collection.update(role=role + "_collection", collected=count, node_ids=node_ids, node_id_set=node_id_set)
            collections.append(collection)
            row = execute(run_argv, cwd=outer, env=env)
            require_success(row, role)
            outcomes = V.strict_passing_outcomes(row["stdout"]["utf8"], count, role, str(outer), run_argv[5:])
            row.update(role=role, collected=count, collection_node_id_set=node_id_set, **outcomes)
            executions.append(row)
    toolchain = V.test_toolchain_identity()
    pytest_configuration = {
        "outer": V.record_for("pytest.ini", V.ROOT, V.TARGET),
        "nested": V.record_for("ipfs_datasets_py/pytest.ini", V.DATASETS, V.CURRENT, "pytest.ini"),
    }
    target = V.target_identity()
    sources = V.expected_sources()
    ended = utc_now()
    receipt = {
        "schema": "proof-grounded-ir-learning/integrated-test-receipt/v2",
        "task_id": "PGIR-211", "target": target, "sources": sources,
        "capture_startup_environment": CAPTURE_STARTUP,
        "observed_start_utc": started, "observed_end_utc": ended,
        "isolated_target_checkout": checkout, "runtime": runtime, "collections": collections, "executions": executions,
        "controlled_environment": controlled_environment,
        "unset_environment": ["PYTHONHOME", "PYTEST_ADDOPTS"],
        "toolchain": toolchain,
        "pytest_configuration": pytest_configuration,
    }
    write_json("test_receipt.json", receipt)


def build_network() -> None:
    started = utc_now()
    rows = []
    with clean_target_checkout("network") as (outer, checkout):
        for frozen in V.expected_release_rows():
            argv = ["/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail-with-body", "--header", "Accept-Encoding: identity", "--write-out", "\n%{http_code}", frozen["url"]]
            execution = execute(argv, cwd=outer, timeout=120)
            require_success(execution, f"network {frozen['release_id']}")
            combined = execution["stdout"]["utf8"].encode()
            if not combined.endswith(b"\n200"):
                raise RuntimeError(f"unexpected HTTP status for {frozen['release_id']}")
            body = combined[:-4]
            document = V.strict_json_bytes(body, frozen["release_id"])
            rows.append({
                **frozen,
                "observed_revision": document.get("sha"),
                "http_status": 200,
                "body": {"utf8": body.decode("utf-8"), **V.identity(body)},
                "canonical_json_identity": V.identity(V.canonical(document)),
                "execution": execution,
            })
    receipt = {
        "schema": "proof-grounded-ir-learning/integrated-network-capture/v2",
        "task_id": "PGIR-211", "target": V.target_identity(), "sources": V.expected_sources(),
        "capture_startup_environment": CAPTURE_STARTUP,
        "observed_start_utc": started, "observed_end_utc": utc_now(),
        "network_execution_required": True, "offline_replay_permitted": False,
        "isolated_target_checkout": checkout, "responses": rows,
        "response_count": len(rows), "all_exact_revision_hashes_matched": all(row["body"]["sha256"] == row["expected_sha256"] and row["observed_revision"] == row["revision"] for row in rows),
    }
    write_json("network_receipt.json", receipt)


def refs_from_execution(execution: Mapping[str, Any]) -> list[dict[str, str]]:
    return V.parse_ls_remote(execution["stdout"]["utf8"].encode(), "capture")


def capture_remote(name: str, url: str, candidates: Sequence[tuple[str, Sequence[str]]], expected_missing: Sequence[str]) -> dict[str, Any]:
    observed_start = utc_now()
    parent = Path(tempfile.mkdtemp(prefix=f"pgir211-portability-{name}-"))
    bare = parent / "bare.git"
    result: dict[str, Any] | None = None
    removal: dict[str, Any] | None = None
    try:
        precreation = execute(["/usr/bin/test", "!", "-e", str(bare)], cwd=parent)
        require_success(precreation, f"{name} absence")
        init = execute(["/usr/bin/git", "init", "--bare", str(bare)], cwd=parent)
        require_success(init, f"{name} init")
        bare_check = execute(["/usr/bin/git", "rev-parse", "--is-bare-repository"], cwd=bare)
        empty_before = execute(["/usr/bin/git", "for-each-ref", "--format=%(objectname)%09%(refname)"], cwd=bare)
        for execution, label in ((bare_check, "bare check"), (empty_before, "empty refs")):
            require_success(execution, f"{name} {label}")
        pre = execute(["/usr/bin/git", "ls-remote", "--refs", "--heads", "--tags", url], cwd=bare, timeout=300)
        fetch = execute(["/usr/bin/git", "fetch", "--no-write-fetch-head", url, "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"], cwd=bare, timeout=900)
        post = execute(["/usr/bin/git", "ls-remote", "--refs", "--heads", "--tags", url], cwd=bare, timeout=300)
        fetched = execute(["/usr/bin/git", "for-each-ref", "--format=%(objectname)%09%(refname)", "refs/remotes/origin", "refs/tags"], cwd=bare)
        rev_list = execute(["/usr/bin/git", "rev-list", "--all"], cwd=bare, compress_stdout=True, timeout=900)
        for execution, label in ((pre, "pre ls-remote"), (fetch, "fetch"), (post, "post ls-remote"), (fetched, "fetched refs"), (rev_list, "rev-list")):
            require_success(execution, f"{name} {label}")
        pre_rows, post_rows, fetched_rows = refs_from_execution(pre), refs_from_execution(post), refs_from_execution(fetched)
        normalized_fetched = [{"oid": row["oid"], "ref": row["ref"].replace("refs/remotes/origin/", "refs/heads/", 1)} for row in fetched_rows]
        normalized_fetched.sort(key=lambda row: row["ref"])
        if not (pre_rows == post_rows == normalized_fetched):
            raise RuntimeError(f"{name} remote_ref_drift")
        rev_bytes = gzip.decompress(base64.b64decode(rev_list["stdout"]["gzip_base64"]))
        rev_oids = set(rev_bytes.decode("ascii").splitlines())
        candidate_rows = []
        reachable = []
        source_repo = SOURCE_ROOT if name == "outer" else SOURCE_ROOT / "ipfs_datasets_py"
        for oid, roles in candidates:
            local_type = V.git(source_repo, "cat-file", "-t", oid)
            object_check = execute(["/usr/bin/git", "cat-file", "-e", f"{oid}^{{commit}}"], cwd=bare)
            remote_reachable = oid in rev_oids
            containment = None
            refs: list[str] = []
            if remote_reachable:
                containment = execute(["/usr/bin/git", "for-each-ref", "--contains", oid, "--format=%(refname)"], cwd=bare)
                require_success(containment, f"{name} containment {oid}")
                refs = containment["stdout"]["utf8"].splitlines()
                reachable.append(oid)
            elif object_check["exit_code"] == 0:
                raise RuntimeError(f"{name} unreachable candidate unexpectedly exists: {oid}")
            ref_data = "".join(f"{ref}\n" for ref in refs).encode()
            candidate_rows.append({
                "oid": oid, "source_roles": list(roles), "local_object_type": local_type,
                "remote_reachable": remote_reachable, "object_check": object_check,
                "containment_execution": containment, "containing_refs": refs,
                "containing_ref_count": len(refs), "containing_ref_set": {"count": len(refs), **V.identity(ref_data)},
                "witness_ref": refs[0] if refs else None,
            })
        missing = sorted(oid for oid, _roles in candidates if oid not in rev_oids)
        if missing != list(expected_missing):
            raise RuntimeError(f"{name} missing population drift: {missing}")
        ref_bytes = V.normalized_ref_bytes(pre_rows)
        rev_list["commit_count"] = len(rev_oids)
        rev_list["reachable_candidate_oids"] = reachable
        result = {
            "repository": name, "remote_url": url, "isolated_bare": True, "temp_repo_fresh": True,
            "bare_repo_path": str(bare), "bare_repo_parent": str(parent),
            "precreation": precreation, "init": init, "bare_check": bare_check,
            "empty_refs_before_fetch": empty_before, "pre_ls_remote": pre, "fetch": fetch,
            "post_ls_remote": post, "fetched_refs": fetched, "rev_list_all": rev_list,
            "normalized_ref_set": {"count": len(pre_rows), **V.identity(ref_bytes)},
            "pre_equals_post_equals_fetched": True, "candidates": candidate_rows,
            "reachable_candidate_oids": reachable, "missing_candidate_oids": missing,
        }
    finally:
        active_failure = sys.exc_info()[1]
        cleanup_failures: list[str] = []
        if bare.exists():
            try:
                removal = execute(["/usr/bin/rm", "-r", "--", str(bare)], cwd=parent)
                if removal["exit_code"] != 0:
                    cleanup_failures.append(f"{name} bare removal")
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                cleanup_failures.append(f"{name} bare removal execution: {exc}")
        elif result is not None:
            cleanup_failures.append(f"{name} bare repository disappeared before recorded removal")
        if not bare.exists() and parent.exists():
            try:
                parent.rmdir()
            except OSError as exc:
                cleanup_failures.append(f"{name} temporary parent removal: {exc}")
        if cleanup_failures:
            raise_primary_or_cleanup(active_failure, cleanup_failures)
    if result is None or removal is None:
        raise RuntimeError(f"{name} portability capture did not complete")
    result.update({
        "removed_after_capture": not bare.exists(),
        "observed_start_utc": observed_start,
        "observed_end_utc": utc_now(),
        "removal": removal,
    })
    return result


def build_portability() -> None:
    started = utc_now()
    outer = capture_remote("outer", V.OUTER_REMOTE, V.OUTER_CANDIDATES, V.OUTER_MISSING)
    nested = capture_remote("nested", V.NESTED_REMOTE, V.NESTED_CANDIDATES, V.NESTED_MISSING)
    no_go = V.portability_no_go_claim(
        outer["missing_candidate_oids"],
        nested["missing_candidate_oids"],
    )
    receipt = {
        "schema": "proof-grounded-ir-learning/integrated-portability-capture/v2",
        "task_id": "PGIR-211", "target": V.target_identity(), "sources": V.expected_sources(),
        "capture_startup_environment": CAPTURE_STARTUP,
        "observed_start_utc": started, "observed_end_utc": utc_now(),
        "repositories": {"outer": outer, "nested": nested},
        **no_go,
    }
    write_json("portability_receipt.json", receipt)


def build_component() -> None:
    started = utc_now()
    with clean_target_checkout("component") as (outer, checkout):
        argv = ["/usr/bin/python3.12", "-S", str(VERIFIER_PATH), "--components-pre-acceptance", "--target-root", str(outer)]
        env, controlled_environment = controlled_python_environment()
        execution = execute(argv, cwd=outer, env=env, timeout=900)
        require_success(execution, "pre-acceptance component verifier")
    inputs = {
        name: {"path": name, **V.identity((EVIDENCE_DIR / name).read_bytes())}
        for name in ("README.md", "capture_evidence.py", "historical_closure_receipt.json", "network_receipt.json", "portability_receipt.json", "test_receipt.json")
    }
    receipt = {
        "schema": "proof-grounded-ir-learning/integrated-component-verification/v2",
        "task_id": "PGIR-211", "target": V.target_identity(), "sources": V.expected_sources(),
        "capture_startup_environment": CAPTURE_STARTUP,
        "observed_start_utc": started, "observed_end_utc": utc_now(),
        "execution_source_location": "prospective PGIR-211 verifier/evidence from SOURCE_ROOT; all repository inputs resolved through --target-root pointing at the fresh clean immutable target checkout",
        "execution_source_absolute_path": str(VERIFIER_PATH),
        "acceptance_artifact_bound": False, "component_inputs": inputs,
        "isolated_target_checkout": checkout, "execution": execution,
        "controlled_environment": controlled_environment,
        "unset_environment": ["PYTHONHOME", "PYTEST_ADDOPTS"],
        "component_verified": True, "pgir_205_execution_authorized": False,
    }
    write_json("component_verification_receipt.json", receipt)


def build_acceptance() -> None:
    component_receipt = V.read_json(EVIDENCE_DIR / "component_verification_receipt.json")
    output = V.strict_json_bytes(
        V.retained_stream_bytes(component_receipt["execution"]["stdout"], "component stdout"),
        "component stdout",
    )
    component_results = {**output["components"], "component_verification": {"component_verified": True}}
    policy_opaque = component_results["historical"]["campaign"]["semantics"]["opaque_policy_revision"]
    measured_adapter = component_results["historical"]["campaign"]["semantics"]["measured_adapter_mismatch"]
    dirty_snapshot = component_results["historical"]["campaign"]["inventories"]["baseline_recursive"]["dirty_snapshot_mismatch"]
    toolchain_mismatches = component_results["tests"]["toolchain_record_mismatches"]
    portability_no_go = V.portability_no_go_claim(V.OUTER_MISSING, V.NESTED_MISSING)
    portability_summary_keys = ("status", "missing_outer_commits", "missing_nested_commits", "pgir_205_execution_authorized")
    closure_names = (
        "README.md", "capture_evidence.py", "component_verification_receipt.json",
        "historical_closure_receipt.json", "network_receipt.json", "portability_receipt.json", "test_receipt.json",
    )
    receipt: dict[str, Any] = {
        "schema": "proof-grounded-ir-learning/successor-integrated-acceptance/v2",
        "task_id": "PGIR-211", "result_identity": "RESULT(PGIR-211)",
        "capture_startup_environment": CAPTURE_STARTUP,
        "decision": "permanent_no_go", "completion_authoritative": False,
        "pgir_205_execution_authorized": False, "containing_commit_claimed": False,
        "acceptance_identity_derivation": "canonical DAG-JSON/SHA-256 of this document after omitting acceptance_sha256 and acceptance_cid; the RESULT(PGIR-211) supersession row uses a field-name pointer and contains no derived self value",
        "target": V.target_identity(), "verifier_source": V.source_identity(VERIFIER_PATH),
        "supersession_chain": [
            {"result_identity": "RESULT(PGIR-208)", "acceptance_cid": V.P208_CID},
            {"result_identity": "RESULT(PGIR-210)", "acceptance_cid": V.P210_CID},
            {"result_identity": "RESULT(PGIR-211)", "self_reference_field": "acceptance_cid"},
        ],
        "predecessor_acceptance_cids": {"PGIR-208": V.P208_CID, "PGIR-209": V.P209_CID, "PGIR-210": V.P210_CID},
        "canonical_closure": {name: {"path": name, **V.identity((EVIDENCE_DIR / name).read_bytes())} for name in closure_names},
        "component_results": component_results,
        "portability_no_go": {key: portability_no_go[key] for key in portability_summary_keys},
        "permanent_no_go_reason_codes": [
            "remote_commit_population_incomplete", "historical_measured_adapter_cid_mismatch",
            "historical_policy_revision_opaque", "historical_dirty_snapshot_cid_unsealed_mismatch",
            "test_toolchain_loaded_record_mismatch", "test_toolchain_unused_console_record_mismatch",
            "zero_rights_admitted_materialized_rows",
            "tokenizer_not_admitted", "current_baseline_retired",
        ],
        "unresolved_links": [policy_opaque],
        "historical_recursive_defects": {"opaque_links": [policy_opaque], "mismatches": [measured_adapter, dirty_snapshot]},
        "execution_environment_defects": {
            "toolchain_integrity_status": "test_toolchain_integrity_no_go",
            "toolchain_record_mismatches": toolchain_mismatches,
            "test_results_authority": "observed_behavior_only",
        },
    }
    projection = dict(receipt)
    receipt["acceptance_sha256"] = "sha256:" + V.hashlib.sha256(V.canonical(projection)).hexdigest()
    receipt["acceptance_cid"] = V.dag_cid(projection)
    write_json("integrated_acceptance.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("historical", "tests", "network", "portability", "component", "acceptance"))
    args = parser.parse_args()
    {
        "historical": build_historical, "tests": build_tests, "network": build_network,
        "portability": build_portability, "component": build_component, "acceptance": build_acceptance,
    }[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
