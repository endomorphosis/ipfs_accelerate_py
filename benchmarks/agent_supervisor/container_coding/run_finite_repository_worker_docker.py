"""Run the authored finite worker crossing in a disposable offline container.

Copy selected repository sources before execution, deploy the existing separate
worker identity, and retain actual container/lifecycle/checker results. Source
copies are sequential observations, not process-origin attestations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[3]
WORKER_ROOT = "/opt/ipfs-supervisor"
_RETAINED_HOST_LEASES = {}


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _remove_container(name):
    """Confirm absence through the live engine, including uncertain creation."""
    for _ in range(3):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        observed = subprocess.run(["docker", "ps", "-a", "-q", "--no-trunc", "--filter", "name=^/" + name + "$"],
            capture_output=True, text=True, timeout=15)
        if observed.returncode == 0 and not observed.stdout.strip():
            return True
    return False


def _snapshot(package, destination):
    """Retain Python and small package resources without Git or user state."""
    destination.mkdir()
    selected = []
    for prefix in ("ipfs_accelerate_py", "benchmarks", "test", "ipfs_datasets_py", "ipfs_kit_py"):
        base = package / prefix
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(package)
            if any(part in {"__pycache__", ".git", ".venv", "node_modules", "workspace", "evidence"}
                   for part in relative.parts):
                continue
            if (path.is_symlink() or not path.is_file()
                    or path.suffix not in {".py", ".json", ".yaml", ".yml", ".toml", ".sql", ".lean", ".lgt"}
                    or path.stat().st_size > 4 * 1024**2):
                continue
            raw = path.read_bytes()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            target.chmod(0o444)
            selected.append({"path": str(path), "relative": relative.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    if not selected:
        raise ValueError("package snapshot contains no selected sources")
    for directory in (destination, *(path for path in destination.rglob("*") if path.is_dir())):
        directory.chmod(0o755)
    return selected


def run(output, *, image, datasets, kit, lean, tests_only=False, test_targets=(), advisory_worker=False):
    if tests_only and advisory_worker:
        raise ValueError("native tests and advisory worker qualification are separate routes")
    if test_targets and not tests_only:
        raise ValueError("test selection requires the native test route")
    from .container_worker_deployment import OWNER_ENTRY, VALIDATION_ENTRY, WORKER_ENTRY
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceLane, ResourceSchedulerConfig,
    )
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh exact output required")
    output.mkdir(mode=0o755)
    selected = {
        "source": _snapshot(SOURCE, output / "source"),
        "datasets": _snapshot(Path(datasets).resolve(strict=True), output / "datasets"),
        "kit": _snapshot(Path(kit).resolve(strict=True), output / "kit"),
    }
    _write(output / "selected-source-snapshot.json", selected)
    deployment = output / "deployment"
    deployment.mkdir()
    for name, content in (("owner-worker", OWNER_ENTRY), ("validation-worker", VALIDATION_ENTRY),
                          ("worker-entry", WORKER_ENTRY)):
        (deployment / name).write_text(content)
    (deployment / "sudoers").write_text(
        "supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n")
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    name = "ipfs-finite-worker-" + str(time.time_ns())
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=output / "host-resource-admission.json", lane_reservations={}, auto_renew_leases=True))
    record = {"schema": "finite-worker-offline-container-execution@1", "image_id": image_id,
        "network": "none", "privileged": False, "cpu_limit": 12, "memory_limit_bytes": 8 * 1024**3,
        "pids_limit": 512, "source_snapshot_scope": "selected sequential Python/resource copies; not atomic checkout or process-origin attestation"}
    record["mode"] = ("native_test_suite" if tests_only else
                      "advisory_worker_successor_fixture" if advisory_worker else "worker_successor_fixture")
    started = time.monotonic()
    container = None
    lease = scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=12, memory_mb=8192,
        child_process_slots=12, timeout=90, request_id="finite-worker-container")
    try:
        record["host_reservation"] = lease.to_dict()
        try:
            record["container_creation_attempted"] = True
            container = subprocess.check_output(["docker", "create", "--network", "none",
                "--cpus", "12", "--memory", "8g", "--pids-limit", "512", "--user", "0",
                "--name", name,
                "-v", str(output / "source") + ":" + WORKER_ROOT + "/source:ro",
                "-v", str(output / "datasets") + ":" + WORKER_ROOT + "/datasets:ro",
                "-v", str(output / "kit") + ":" + WORKER_ROOT + "/kit:ro",
                "-v", str(Path(lean).resolve(strict=True)) + ":/toolchains/lean:ro",
                "--entrypoint", "/bin/sleep", image_id, "1800"], text=True, timeout=30).strip()
            record["container_id"] = container
            subprocess.run(["docker", "start", container], check=True, capture_output=True, timeout=30)
            for filename in ("owner-worker", "validation-worker", "worker-entry"):
                subprocess.run(["docker", "cp", str(deployment / filename),
                    container + ":" + WORKER_ROOT + "/bin/" + filename], check=True, capture_output=True, timeout=30)
            subprocess.run(["docker", "cp", str(deployment / "sudoers"),
                container + ":/etc/sudoers.d/finite-worker"], check=True, capture_output=True, timeout=30)
            setup = (
                "import json,os,pathlib;root=pathlib.Path('/opt/ipfs-supervisor');"
                "[(os.chown(root/'bin'/n,0,0),(root/'bin'/n).chmod(0o755)) for n in ('owner-worker','validation-worker','worker-entry')];"
                "p=pathlib.Path('/etc/sudoers.d/finite-worker');os.chown(p,0,0);p.chmod(0o440);"
                "results=pathlib.Path('/results');results.mkdir();os.chown(results,1000,1000);results.chmod(0o755);"
                "boundary={'schema':'supervisor-container-worker-boundary@1',"
                f"'container_id':{container!r},'image_id':{image_id!r},"
                "'owner_uid':1000,'worker_uid':1001,'single_worker':True,"
                "'namespaces':{k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')},"
                "'allowed_worktree_roots':['/opt/ipfs-supervisor/worktrees'],"
                "'owner_private_paths':['/opt/ipfs-supervisor/state','/results/native/private'],"
                "'validation_repository_roots':['/results/native/repository']};"
                "p=root/'container-boundary.json';p.write_text(json.dumps(boundary,sort_keys=True));p.chmod(0o444)"
            )
            subprocess.run(["docker", "exec", "--user", "0", container,
                "/usr/local/bin/python", "-I", "-c", setup], check=True, capture_output=True, timeout=30)
            if tests_only:
                # Historical native fixtures name their host tools explicitly.
                # Point those paths at the real image Python and mounted Lean;
                # their normal sealing and resource checks still run unchanged.
                aliases = {
                    "/home/barberb/.local/bin/python": WORKER_ROOT + "/venv/bin/python",
                    "/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean": "/toolchains/lean/bin/lean",
                }
                alias_setup = (
                    "import pathlib;"
                    f"aliases={aliases!r};"
                    "[(pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True),"
                    "pathlib.Path(path).symlink_to(target)) for path,target in aliases.items()]"
                )
                subprocess.run(["docker", "exec", "--user", "0", container,
                    "/usr/local/bin/python", "-I", "-c", alias_setup],
                    check=True, capture_output=True, timeout=30)
                record["native_test_tool_aliases"] = aliases
            with (output / "run.stdout").open("w") as stdout, (output / "run.stderr").open("w") as stderr:
                module = ("benchmarks.agent_supervisor.container_coding.finite_repository_native_tests"
                          if tests_only else
                          "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_join_experiment"
                          if advisory_worker else "benchmarks.agent_supervisor.container_coding.finite_repository_worker_experiment")
                arguments = ["/results/tests", *test_targets] if tests_only else ["run", "/results/native"]
                process = subprocess.run(["docker", "exec", "--user", "1000:1000",
                    "-e", "DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean",
                    "-e", "IPFS_ACCELERATE_RUN_LIVE_QUACK=1",
                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
                    "-e", "IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=/results/tests/pytest-seal.duckdb",
                    container, WORKER_ROOT + "/venv/bin/python", "-m",
                    module, *arguments], stdout=stdout, stderr=stderr, timeout=600)
            record["returncode"] = process.returncode
            record["host_lease_held_after_native_exit"] = not lease.released
        finally:
            if container is not None:
                try:
                    copied = subprocess.run(["docker", "cp", container + ":/results/.", str(output)],
                        capture_output=True, text=True, timeout=60)
                    record["container_results_copied"] = copied.returncode == 0
                    handoffs = subprocess.run(["docker", "cp", container + ":" + WORKER_ROOT + "/finite-handoffs",
                        str(output / "handoffs")], capture_output=True, text=True, timeout=30)
                    record["candidate_handoffs_copied"] = handoffs.returncode == 0
                except (OSError, subprocess.TimeoutExpired) as error:
                    record["result_copy_error_type"] = type(error).__name__
            if record.get("container_creation_attempted"):
                record["container_removed"] = _remove_container(name)
                if not record["container_removed"]:
                    # Keep actual cleanup failure visible. A container deletion
                    # failure must not be reported as successful qualification.
                    raise RuntimeError("disposable finite worker container cleanup failed")
            record["elapsed_seconds"] = time.monotonic() - started
            _write(output / "container-execution.json", record)
    finally:
        if not record.get("container_creation_attempted") or record.get("container_removed") is True:
            lease.release()
        else:
            _RETAINED_HOST_LEASES[name] = lease
            record["host_reservation_retained_for_live_container"] = True
            record["retention_scope"] = "current wrapper process only; owner crash recovery is not qualified"
            _write(output / "unsafe-container-cleanup.json", record)
    record["host_resources_after_cleanup"] = scheduler.snapshot()
    changed = []
    for rows in selected.values():
        for item in rows:
            p = Path(item["path"])
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                changed.append(item["path"])
    record["original_selected_source_changes_after_execution"] = changed
    record["runtime_used_retained_source_copies"] = True
    _write(output / "container-execution-final.json", record)
    if (record.get("returncode") != 0 or not record.get("container_removed")
            or not record.get("container_results_copied")):
        raise RuntimeError("finite worker qualification failed; retained stdout/stderr and native evidence identify the stage")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--tests-only", action="store_true")
    route.add_argument("--advisory-worker", action="store_true")
    parser.add_argument("--test-target", action="append", default=[])
    parser.add_argument("--image", default="ipfs-supervisor-finite-worker:20261002")
    parser.add_argument("--datasets", type=Path, default=SOURCE.parent / "ipfs_datasets")
    parser.add_argument("--kit", type=Path, default=SOURCE.parent / "ipfs_kit")
    parser.add_argument("--lean", type=Path, default=Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1"))
    args = parser.parse_args()
    result = run(args.output, image=args.image, datasets=args.datasets, kit=args.kit, lean=args.lean,
                 tests_only=args.tests_only, test_targets=tuple(args.test_target), advisory_worker=args.advisory_worker)
    print(json.dumps({"returncode": result["returncode"], "container_removed": result["container_removed"]}))


if __name__ == "__main__":
    main()
