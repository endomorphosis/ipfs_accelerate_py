"""Run the authored finite worker crossing in a disposable offline proof-query container.

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
import stat
import time

SOURCE = Path(__file__).resolve().parents[3]
WORKER_ROOT = "/opt/ipfs-supervisor"
POST_BIRTH_CASES = {
    "child-post-birth-callback-error": "isolated_after_real_birth_callback_error",
    "child-post-birth-ack-drop": "isolated_real_birth_application_ACK_drop",
}
FIXTURE_CASES = ("all", "positive", "child-late-proof", "child-late-epoch", *POST_BIRTH_CASES)
_RETAINED_HOST_LEASES = {}
_NATIVE_SOLVER_PREFLIGHT = """import json
import pathlib
import shutil
import subprocess
import sys
expected = json.loads(sys.argv[1])
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import _native_executable, _binary_digest
rows = []
for name in ('z3', 'cvc5'):
    discovered = shutil.which(name)
    if discovered is None:
        raise RuntimeError(name + ' native executable unavailable')
    path, launcher = _native_executable(discovered)
    with pathlib.Path(path).open('rb') as stream:
        if stream.read(4) != bytes((127, 69, 76, 70)):
            raise RuntimeError('actual solver ELF required')
    digest = _binary_digest(path)
    if name in expected and (path != '/usr/local/bin/' + name or digest != expected[name]):
        raise RuntimeError(name + ' mounted solver capsule pin differs')
    version = subprocess.check_output([path, '--version'], text=True, timeout=10)
    if not version.strip() or _binary_digest(path) != digest:
        raise RuntimeError(name + ' solver version or executable custody differs')
    rows.append({'name': name, 'discovered_path': discovered,
        'resolved_discovered_path': str(pathlib.Path(discovered).resolve(strict=True)),
        'path': path, 'version': version, 'sha256': digest,
        'launcher_sha256': launcher, 'actual_elf': True})
print(json.dumps(rows))
"""


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



def _solver_capsules(output, supplied):
    """Copy only actual native ELF executables into readonly owned mounts."""
    rows = []
    directory = output / "solver-tools"
    for name, source in supplied.items():
        if source is None:
            continue
        if not rows:
            directory.mkdir(mode=0o755)
        original = Path(source).absolute()
        actual = original.resolve(strict=True)
        before = actual.stat()
        if not stat.S_ISREG(before.st_mode) or not before.st_mode & 0o111:
            raise ValueError(name + " capsule requires a regular executable ELF")
        raw = actual.read_bytes()
        after = actual.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(name + " solver source changed during capture")
        if len(raw) < 20 or raw[:4] != bytes((127, 69, 76, 70)) or raw[5] not in (1, 2):
            raise ValueError(name + " capsule requires actual ELF bytes, not an installer launcher")
        target = directory / name
        target.write_bytes(raw)
        target.chmod(0o555)
        digest = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(name + " copied solver capsule differs")
        rows.append({"name": name, "original_source": str(original), "resolved_actual_source": str(actual),
            "copy": str(target), "mount": "/usr/local/bin/" + name, "sha256": digest, "bytes": len(raw),
            "elf_class": raw[4], "elf_endianness": raw[5],
            "elf_machine": int.from_bytes(raw[18:20], "little" if raw[5] == 1 else "big"),
            "source_stat": {"device": before.st_dev, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns, "mode": stat.S_IMODE(before.st_mode)},
            "readonly_mount": True, "host_version_executed": False})
    _write(output / "solver-tool-capsules.json", {"schema": "finite-proof-query-native-solver-capsules@1",
        "scope": "retained actual ELF bytes; executable availability and versions require native image preflight",
        "tools": rows, "bytes_copied": sum(row["bytes"] for row in rows),
        "model_inference_calls": 0, "training_steps": 0})
    return rows


def _final_retention(output, record, scheduler, selected):
    """Always record available final observations, including early refusals."""
    try:
        record["host_resources_after_cleanup"] = scheduler.snapshot()
        record["host_resource_snapshot_status"] = "observed"
    except Exception as error:
        record["host_resources_after_cleanup"] = None
        record["host_resource_snapshot_status"] = "unknown"
        record["host_resource_snapshot_error"] = {"type": type(error).__name__, "message": str(error)[:4096]}
    changed, unavailable = [], []
    for rows in selected.values():
        for item in rows:
            path = Path(item["path"])
            try:
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception as error:
                unavailable.append({"path": item["path"], "before_sha256": item["sha256"],
                    "status": "unknown", "error_type": type(error).__name__, "message": str(error)[:4096]})
                continue
            if current != item["sha256"]:
                changed.append({"path": item["path"], "before_sha256": item["sha256"], "current_sha256": current})
    record["original_selected_source_changes_after_execution"] = changed
    record["original_selected_source_unavailable_after_execution"] = unavailable
    record["source_comparison_status"] = "unknown" if unavailable else "observed"
    record["runtime_used_retained_source_copies"] = bool(record.get("container_id"))
    record["source_evidence_scope"] = "selected producer pins and readonly package copies; no complete import census or process-origin claim"
    capsule_rows = []
    for item in record.get("solver_tool_capsules", []):
        row = {"name": item["name"], "copy": item["copy"], "expected_sha256": item["sha256"]}
        try:
            row["current_sha256"] = hashlib.sha256(Path(item["copy"]).read_bytes()).hexdigest()
            row["copy_matches"] = row["current_sha256"] == item["sha256"]
        except Exception as error:
            row.update({"copy_matches": False, "error_type": type(error).__name__, "message": str(error)[:4096]})
        capsule_rows.append(row)
    record["solver_tool_capsules_after_execution"] = capsule_rows
    record["solver_capsule_byte_integrity"] = all(row["copy_matches"] for row in capsule_rows)
    record["candidate_handoffs_copied"] = record.get("candidate_handoffs_copied")
    record["container_results_copied"] = record.get("container_results_copied")
    record["qualification_status"] = "completed" if (
        record.get("returncode") == 0 and record.get("container_removed") is True
        and record.get("container_results_copied") is True and record.get("candidate_handoffs_copied") is True
        and not record.get("native_exception") and record["host_resource_snapshot_status"] == "observed"
        and record["solver_capsule_byte_integrity"] is True
        and not record["host_resources_after_cleanup"]["active_lease_count"]
        and not record["host_resources_after_cleanup"]["waiting_request_count"]) else "unqualified"
    _write(output / "container-execution-final.json", record)


def run(output, *, image, datasets, kit, lean, z3=None, cvc5=None, fixture_case="all"):
    if type(fixture_case) is not str or fixture_case not in FIXTURE_CASES:
        raise ValueError("exact finite proof-query fixture selection required")
    tests_only, test_targets, advisory_worker = False, (), False
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
    solver_capsules = _solver_capsules(output, {"z3": z3, "cvc5": cvc5})
    solver_mounts = [arg for item in solver_capsules for arg in ("-v", item["copy"] + ":" + item["mount"] + ":ro")]
    expected_solvers = {item["name"]: item["sha256"] for item in solver_capsules}
    deployment = output / "deployment"
    deployment.mkdir()
    for name, content in (("owner-worker", OWNER_ENTRY), ("validation-worker", VALIDATION_ENTRY),
                          ("worker-entry", WORKER_ENTRY)):
        (deployment / name).write_text(content)
    (deployment / "sudoers").write_text(
        "supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n")
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    name = "ipfs-finite-proof-query-worker-" + str(time.time_ns())
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=output / "host-resource-admission.json", lane_reservations={}, auto_renew_leases=True))
    record = {"schema": "finite-proof-query-worker-offline-container-execution@1", "image_id": image_id,
        "network": "none", "privileged": False, "cpu_limit": 12, "memory_limit_bytes": 8 * 1024**3,
        "pids_limit": 512, "fixture_case": fixture_case, "native_timeout_seconds": 900, "solver_tool_capsules": solver_capsules, "source_snapshot_scope": "selected sequential Python/resource copies; not atomic checkout or process-origin attestation"}
    record["mode"] = ("native_test_suite" if tests_only else
                      "advisory_worker_successor_fixture" if advisory_worker else "proof_query_worker_fixture")
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
                *solver_mounts,
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
                "'owner_private_paths':['/opt/ipfs-supervisor/state','/results/native/positive/private','/results/native/child-late-proof/private','/results/native/child-late-epoch/private','/results/native/child-post-birth-callback-error/private','/results/native/child-post-birth-ack-drop/private'],"
                "'validation_repository_roots':['/results/native/positive/repository','/results/native/child-late-proof/repository','/results/native/child-late-epoch/repository','/results/native/child-post-birth-callback-error/repository','/results/native/child-post-birth-ack-drop/repository']};"
                "p=root/'container-boundary.json';p.write_text(json.dumps(boundary,sort_keys=True));p.chmod(0o444)"
            )
            if fixture_case != "all":
                # The boundary names all authored cases. The selected harness
                # creates its own fresh output; provision the other protected
                # roots before the isolated worker checks their ancestors.
                setup += (
                    ";native=results/'native';native.mkdir(mode=0o755);os.chown(native,1000,1000)"
                    f";case_roots=[native/n for n in {FIXTURE_CASES[1:]!r} if n!={fixture_case!r}]"
                    ";[(p.mkdir(mode=0o755),os.chown(p,1000,1000),"
                    "(p/'private').mkdir(mode=0o700),os.chown(p/'private',1000,1000),"
                    "(p/'repository').mkdir(mode=0o755),os.chown(p/'repository',1000,1000)) for p in case_roots]"
                    f";handoff=root/'finite-handoffs'/{fixture_case!r};"
                    "handoff.mkdir(mode=0o755);os.chown(handoff,1000,1000)"
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
            tools = subprocess.run(["docker", "exec", "--user", "1000:1000", container,
                WORKER_ROOT + "/venv/bin/python", "-c", _NATIVE_SOLVER_PREFLIGHT, json.dumps(expected_solvers)],
                capture_output=True, text=True, timeout=30)
            _write(output / "container-native-solvers.json", {"returncode": tools.returncode,
                "stdout": tools.stdout, "stderr": tools.stderr})
            tools.check_returncode()
            native_solvers = json.loads(tools.stdout)
            if (not isinstance(native_solvers, list) or {row["name"] for row in native_solvers} != {"z3", "cvc5"}
                    or len(native_solvers) != 2 or any(not row["actual_elf"] for row in native_solvers)):
                raise RuntimeError("complete native solver ELF/version observations required")
            record["container_native_solver_pins"] = native_solvers
            record["native_solver_version_observations"] = len(native_solvers)
            with (output / "run.stdout").open("w") as stdout, (output / "run.stderr").open("w") as stderr:
                module = ("benchmarks.agent_supervisor.container_coding.finite_repository_native_tests"
                          if tests_only else
                          "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_join_experiment"
                          if advisory_worker else "benchmarks.agent_supervisor.container_coding.finite_proof_query_worker_experiment")
                arguments = ["/results/tests", *test_targets] if tests_only else ["run-all", "/results/native"]
                if fixture_case != "all":
                    arguments = ["run", "/results/native/" + fixture_case,
                        "--handoff-root", WORKER_ROOT + "/finite-handoffs/" + fixture_case]
                    if fixture_case in POST_BIRTH_CASES:
                        arguments += ["--post-birth-control", POST_BIRTH_CASES[fixture_case]]
                    elif fixture_case != "positive":
                        arguments += ["--child-control", "proof" if fixture_case == "child-late-proof" else "epoch"]
                process = subprocess.run(["docker", "exec", "--user", "1000:1000",
                    "-e", "DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean",
                    "-e", "IPFS_ACCELERATE_RUN_LIVE_QUACK=1",
                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
                    "-e", "IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=/results/tests/pytest-seal.duckdb",
                    container, WORKER_ROOT + "/venv/bin/python", "-m",
                    module, *arguments], stdout=stdout, stderr=stderr, timeout=900)
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
    except BaseException as error:
        record["native_exception"] = {"type": type(error).__name__, "message": str(error)[:4096]}
        raise
    finally:
        try:
            if not record.get("container_creation_attempted") or record.get("container_removed") is True:
                lease.release()
            else:
                _RETAINED_HOST_LEASES[name] = lease
                record["host_reservation_retained_for_live_container"] = True
                record["retention_scope"] = "current wrapper process only; owner crash recovery is not qualified"
                _write(output / "unsafe-container-cleanup.json", record)
        finally:
            _final_retention(output, record, scheduler, selected)
    if (record.get("returncode") != 0 or not record.get("container_removed")
            or not record.get("container_results_copied") or not record.get("candidate_handoffs_copied")
            or record["qualification_status"] != "completed"):
        raise RuntimeError("finite worker qualification failed; retained stdout/stderr and native evidence identify the stage")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image", default="ipfs-supervisor-finite-worker:20261002")
    parser.add_argument("--datasets", type=Path, default=SOURCE.parent / "ipfs_datasets")
    parser.add_argument("--kit", type=Path, default=SOURCE.parent / "ipfs_kit")
    parser.add_argument("--lean", type=Path, default=Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1"))
    parser.add_argument("--z3", type=Path, help="Optional actual ELF to retain and mount readonly at /usr/local/bin/z3")
    parser.add_argument("--cvc5", type=Path, help="Optional actual ELF; installer launcher scripts are refused")
    parser.add_argument("--fixture-case", choices=FIXTURE_CASES, default="all",
        help="Run one genuine fixture or the original full collection; each container retains a fixed 900-second native budget")
    args = parser.parse_args()
    result = run(args.output, image=args.image, datasets=args.datasets, kit=args.kit, lean=args.lean,
                 z3=args.z3, cvc5=args.cvc5, fixture_case=args.fixture_case)
    print(json.dumps({"returncode": result["returncode"], "container_removed": result["container_removed"]}))


if __name__ == "__main__":
    main()
