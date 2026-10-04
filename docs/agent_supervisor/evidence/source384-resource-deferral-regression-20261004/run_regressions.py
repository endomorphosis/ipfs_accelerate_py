"""Capture selected source pins and outcomes for resource-deferral controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent.parent
A = WORKSPACE / ".worktrees/ir-release-accelerate-20261002"
D = WORKSPACE / ".worktrees/ir-pressure-attribution-datasets-20261004"
TESTS = [
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_database_portal_attempt_budget.py",
    "test/api/test_database_bounded_pre_effect_retry.py",
    "test/api/semantic_state/test_source384_task_context_bundle.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py::test_crash_restart_resumes_without_duplicating_provider_or_effect",
    "test/api/test_agent_supervisor_database_implementation_daemon.py::test_implicit_embedded_owner_is_store_scoped_and_restart_stable",
    "test/api/test_agent_supervisor_database_implementation_daemon.py::test_provider_heartbeat_renews_exact_task_claim",
    "test/api/test_agent_supervisor_database_implementation_daemon.py::test_provider_result_is_rejected_after_fenced_takeover",
    "test/api/causal_federation/test_admitted_executor.py::test_typed_daemon_promotes_local_attempt_before_provider",
]
A_PATHS = [*[test.split("::")[0] for test in TESTS],
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
]
D_PATHS = [
    "ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_resources.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py",
]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--select")
    args = parser.parse_args()
    assert args.label.replace("-", "").isalnum()
    target = BASE / args.label
    target.mkdir(exist_ok=False)
    selected = set(A_PATHS) | set(git(A, "diff", "--name-only", "--", "*.py").splitlines())

    def pins():
        return {prefix + "/" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for root, prefix, paths in ((A, "source", selected), (D, "datasets", D_PATHS))
                for name in sorted(paths)}

    environment = {
        "PYTHONPATH": str(A) + ":" + str(D) + ":" + str(WORKSPACE / ".venvs/terminal-bench-harbor/lib/python3.12/site-packages"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH": str(target / "fallback-scheduler.json"),
        "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR": "/tmp/srd-" + args.label + "-private/orchestration",
        "IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB": "/tmp/srd-" + args.label + "-private/seal.duckdb",
        "IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH": "/tmp/srd-" + args.label + "-private/keys.db",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp=/tmp/srd-" + args.label, "--junitxml=" + str(target / "results.xml"), *TESTS]
    if args.select:
        argv += ["-k", args.select]
    before = pins()
    command = dict(argv=argv, cwd=str(A), environment_overrides=environment,
                   source_revisions={"source": git(A, "rev-parse", "HEAD"), "datasets": git(D, "rev-parse", "HEAD")},
                   source_pins=before, selected_source_pins_complete_repository=False,
                   harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (target / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (target / "stdout.log").open("x") as log:
        result = subprocess.run(argv, cwd=A, env={**os.environ, **environment}, stdout=log, stderr=subprocess.STDOUT)
    after = pins()
    receipt = dict(returncode=result.returncode, seconds=time.monotonic() - started,
                   source_pins_unchanged=before == after, after_pins=after)
    xml = target / "results.xml"
    if xml.exists():
        receipt["xml_counts"] = {key: sum(int(suite.get(key, "0")) for suite in ET.parse(xml).getroot().iter("testsuite"))
                                 for key in ("tests", "failures", "errors", "skipped")}
    (target / "exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "after_pins"}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
