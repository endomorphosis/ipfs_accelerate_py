"""Capture selected source pins and outcomes for completed header observation controls."""
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
    "test/api/test_local_completion_bridge.py",
    "test/api/test_header_intent_applicability.py",
    "test/api/test_agent_supervisor_local_planning_admission.py",
    "test/api/semantic_state/test_published_task_context.py",
    "test/api/test_intent_requirement_observation.py",
    "test/api/test_intent_requirement_observation_native.py",
    "test/api/test_router_public_instruction.py",
]
A_PATHS = [*[test.split("::")[0] for test in TESTS],
    "ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/header_intent_applicability.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/intent_requirement_observation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py",
    "ipfs_accelerate_py/agent_supervisor/planning/intent_symbolic_planning.py",
    "ipfs_accelerate_py/agent_supervisor/planning/intent_requirement_adapter.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
]
D_PATHS = [
    "ipfs_datasets_py/logic/software_contracts/codebase_header_context.py",
    "ipfs_datasets_py/logic/security_ir/bounded_header_checker.py",
    "ipfs_datasets_py/logic/security_ir/doctor_header_contracts.py",
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
        "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR": "/tmp/lcho-" + args.label + "-private/orchestration",
        "IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB": "/tmp/lcho-" + args.label + "-private/seal.duckdb",
        "IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH": "/tmp/lcho-" + args.label + "-private/keys.db",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp=/tmp/lcho-" + args.label, "--junitxml=" + str(target / "results.xml"), *TESTS]
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
