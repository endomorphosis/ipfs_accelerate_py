"""Source-only request authoring. This does not import or execute any native SDK."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("handles_only", "isolated_fixture_publication"), default="handles_only")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).absolute().parent
    adapter = here.parents[1] / "adapter-v02"
    datasets = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
    accelerator = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    assert re.fullmatch(r"legal-ir-native-acceptance-fixture-[a-z0-9][a-z0-9-]{0,80}", args.name)
    assert args.run_root.is_absolute() and args.run_root.name == args.name and args.output.is_absolute()
    assert ".." not in args.run_root.parts and os.path.abspath(args.run_root) == str(args.run_root)
    assert Path('/tmp') not in args.run_root.parents
    paths = []
    for location in (adapter, here):
        manifest = json.loads((location / "SOURCE_MANIFEST.json").read_bytes())
        for row in manifest["files"]:
            path = location / row["path"]
            raw = path.read_bytes()
            assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
            paths.append(path)
    native = datasets / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    paths.extend(native / (name + ".py") for name in (
        "existing_resource_scheduler", "resource_scheduler", "parented_phase_resources",
        "validation_run_resources", "autoencoder_daemon_resources"))
    paths.extend(accelerator / name for name in (
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py"))
    request = {
        "schema": "legal-ir-native-fixture-entry@2", "name": args.name, "purpose": "controller_preparation",
        "scope": "synthetic-native-acceptance-regression-only", "run_root": str(args.run_root), "mode": args.mode,
        "scheduler_state_path": "/tmp/ipfs-datasets-resource-scheduler-1000.json",
        "expected_scheduler_policy_sha256": "f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661",
        "ledger_path": str(root / "workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json"),
        "roots": [str(root / "workspace/test-logs"), str(root / "workspace/todo-queues"),
                  str(root / "docs/implementation/reports/evidence"), "/tmp/pytest-of-barberb"],
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dict.fromkeys(paths)},
        "adapter_manifest_sha256": hashlib.sha256((adapter / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
        "regression_manifest_sha256": hashlib.sha256((here / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
    }
    raw = (json.dumps(request, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"request_path": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "native_sdk_imported_or_executed": False, "mode": args.mode}))


if __name__ == "__main__":
    main()
