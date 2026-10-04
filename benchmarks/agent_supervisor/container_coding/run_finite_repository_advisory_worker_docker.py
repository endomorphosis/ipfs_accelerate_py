"""Run the joined advisory-feature worker fixture in an offline container.

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
MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_worker_experiment"
_PREFLIGHT_SCHEMA = "finite-advisory-worker-offline-preflight@1"

# This fixed program checks the deployed image and the mounted selected source.
# It creates no registry, owner, trained model, theorem or candidate. Native
# extensions are loaded explicitly from their signed, platform-pinned cache.
_PREFLIGHT = r'''import hashlib,importlib,importlib.metadata,json,os,pathlib,stat,subprocess,sys
root=pathlib.Path('/opt/ipfs-supervisor')
sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
def require(value,message):
 if not value:raise ValueError(message)
def pin(path):
 path=pathlib.Path(path).resolve(strict=True)
 info=path.stat()
 require(stat.S_ISREG(info.st_mode) and 0<info.st_size<=1024**3,'bounded actual dependency file required')
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
 after=path.stat()
 require((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'dependency changed while reading')
 return {'path':str(path),'bytes':info.st_size,'sha256':digest.hexdigest()}
require(os.getuid()==os.geteuid()==1000,'actual owner UID required')
import torch,numpy,duckdb,_duckdb
require(duckdb.__version__=='1.5.5','actual DuckDB 1.5.5 required')
require(torch.version.cuda is None,'offline CPU Torch build required')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
require((torch.tensor([1.0],dtype=torch.float64)+1.0).tolist()==[2.0],'actual CPU tensor calculation failed')
packages={}
for name,module,native_name in (('torch',torch,'torch._C'),('numpy',numpy,'numpy.core._multiarray_umath'),('duckdb',duckdb,'_duckdb')):
 native=importlib.import_module(native_name)
 distribution=importlib.metadata.distribution(name)
 paths={pathlib.Path(module.__file__).resolve(),pathlib.Path(native.__file__).resolve()}
 metadata=distribution.read_text('METADATA')
 require(type(metadata) is str and bool(metadata),'installed package metadata required')
 base=pathlib.Path(module.__file__).resolve().parent
 for directory in ((base/'lib',) if name=='torch' else ((base.parent/'numpy.libs',) if name=='numpy' else ())):
  if directory.is_dir():paths.update(path.resolve() for path in directory.glob('*.so*') if path.is_file())
 packages[name]={'version':module.__version__,'distribution_version':distribution.version,'metadata_sha256':hashlib.sha256(metadata.encode()).hexdigest(),'selected_files':[pin(path) for path in sorted(paths)]}
from ipfs_datasets_py.ducklake.autoencoder_history import _native_connection
connection,extensions=_native_connection()
try:require(connection.execute('SELECT 1').fetchone()[0]==1,'explicit signed native LOAD failed')
finally:connection.close()
require(set(extensions['extensions'])=={'httpfs','ducklake','quack'},'complete signed native extension family required')
cache=pathlib.Path.home()/'.duckdb/extensions'/('v'+duckdb.__version__)/extensions['platform']
extension_files=[pin(cache/(name+'.duckdb_extension')) for name in extensions['extensions']]
require(all(item['sha256']==extensions['extensions'][pathlib.Path(item['path']).name.removesuffix('.duckdb_extension')] for item in extension_files),'loaded native extension changed')
lean=pathlib.Path('/toolchains/lean/bin/lean')
require(lean.resolve(strict=True)==lean,'mapped exact Lean executable required')
version=subprocess.run([str(lean),'--version'],capture_output=True,text=True,timeout=10)
require(version.returncode==0 and version.stdout.startswith('Lean ') and len((version.stdout+version.stderr).encode())<=16384,'actual mapped Lean version probe failed')
report={'schema':'finite-advisory-worker-offline-preflight@1','status':'completed','owner_uid':os.getuid(),'python':sys.version,'executable':pin(sys.executable),'packages':packages,'torch_device':'cpu','torch_dtype':'float64','tensor_check_passed':True,'extension_load':extensions,'extension_files':extension_files,'lean':{'executable':pin(lean),'version':version.stdout.strip(),'stderr':version.stderr},'namespaces':{key:os.readlink('/proc/self/ns/'+key) for key in ('pid','mnt','net')},'source_snapshot_scope':'selected package entry points and native libraries; not complete transitive dependency attestation','training_steps':0,'proof_authority':False,'execution_authority':False,'completion_authority':False,'production_activated':False}
print(json.dumps(report,sort_keys=True,allow_nan=False))
'''


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _preflight(container, output, image_id):
    """Observe actual offline dependencies under the held container envelope."""
    source = output / "offline-preflight.py"
    source.write_text(_PREFLIGHT)
    raw = source.read_bytes()
    record = {"schema": "finite-advisory-worker-offline-preflight-execution@1",
              "image_id": image_id, "container_id": container,
              "source": {"path": str(source), "bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest()},
              "timeout_seconds": 90, "network": "none", "status": "incomplete"}
    started = time.monotonic()
    try:
        process = subprocess.run(["docker", "exec", "--user", "1000:1000",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
            container, WORKER_ROOT + "/venv/bin/python", "-I", "-B", "-c", _PREFLIGHT],
            capture_output=True, text=True, timeout=90)
        (output / "offline-preflight.stdout").write_text(process.stdout)
        (output / "offline-preflight.stderr").write_text(process.stderr)
        record.update(returncode=process.returncode,
                      stdout_sha256=hashlib.sha256(process.stdout.encode()).hexdigest(),
                      stderr_sha256=hashlib.sha256(process.stderr.encode()).hexdigest())
        if process.returncode != 0:
            raise RuntimeError("offline dependency preflight failed; retained stderr identifies the stage")
        observed = json.loads(process.stdout)
        if (type(observed) is not dict or observed.get("schema") != _PREFLIGHT_SCHEMA
                or observed.get("status") != "completed"
                or type(observed.get("owner_uid")) is not int or observed["owner_uid"] != 1000
                or type(observed.get("training_steps")) is not int or observed["training_steps"] != 0):
            raise RuntimeError("exact completed offline preflight result required")
        record.update(status="completed", observation=observed)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, subprocess.TimeoutExpired):
            for role, value in (("stdout", error.stdout), ("stderr", error.stderr)):
                if value is not None:
                    path = output / ("offline-preflight." + role)
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        _write(output / "offline-preflight.json", record)


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


def run(output, *, image, datasets, kit, lean):
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
    name = "ipfs-finite-advisory-worker-" + str(time.time_ns())
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=output / "host-resource-admission.json", lane_reservations={}, auto_renew_leases=True))
    record = {"schema": "finite-advisory-worker-offline-container-execution@1", "image_id": image_id,
        "network": "none", "privileged": False, "cpu_limit": 12, "memory_limit_bytes": 8 * 1024**3,
        "pids_limit": 512, "source_snapshot_scope": "selected sequential Python/resource copies; not atomic checkout or process-origin attestation"}
    record["mode"] = "joined_advisory_worker_successor_fixture"
    record["native_total_timeout_seconds"] = 900
    record["finite_preparation_deadline_seconds"] = 90
    started = time.monotonic()
    container = None
    lease = None
    try:
        record["host_resources_before_reservation"] = scheduler.snapshot()
        lease = scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=12, memory_mb=8192,
            child_process_slots=12, timeout=90, request_id="finite-advisory-worker-container")
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
            record["offline_preflight"] = _preflight(container, output, image_id)
            with (output / "run.stdout").open("w") as stdout, (output / "run.stderr").open("w") as stderr:
                arguments = ["run", "/results/native"]
                process = subprocess.run(["docker", "exec", "--user", "1000:1000",
                    "-e", "DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean",
                    "-e", "IPFS_ACCELERATE_RUN_LIVE_QUACK=1",
                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
                    container, WORKER_ROOT + "/venv/bin/python", "-m",
                    MODULE, *arguments], stdout=stdout, stderr=stderr, timeout=900)
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
        record["execution_error_type"] = type(error).__name__
        record["execution_error"] = str(error)[:2048]
        record["host_reservation_acquired"] = lease is not None
        record["native_module_launched"] = (output / "run.stdout").exists()
        if isinstance(error, subprocess.TimeoutExpired):
            record["execution_timed_out"] = True
        raise
    finally:
        if lease is not None and (not record.get("container_creation_attempted")
                                  or record.get("container_removed") is True):
            lease.release()
        elif lease is not None:
            _RETAINED_HOST_LEASES[name] = lease
            record["host_reservation_retained_for_live_container"] = True
            record["retention_scope"] = "current wrapper process only; owner crash recovery is not qualified"
            _write(output / "unsafe-container-cleanup.json", record)
        record["elapsed_seconds"] = time.monotonic() - started
        record["host_resources_after_cleanup"] = scheduler.snapshot()
        changed = []
        for rows in selected.values():
            for item in rows:
                p = Path(item["path"])
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                    changed.append(item["path"])
        record["original_selected_source_changes_after_execution"] = changed
        record["runtime_used_retained_source_copies"] = record.get("container_creation_attempted", False)
        record["host_reservation_acquired"] = lease is not None
        record["host_reservation_released"] = lease is not None and lease.released
        record["native_module_launched"] = (output / "run.stdout").exists()
        _write(output / "container-execution-final.json", record)
    if (record.get("returncode") != 0 or not record.get("container_removed")
            or not record.get("container_results_copied")):
        raise RuntimeError("finite worker qualification failed; retained stdout/stderr and native evidence identify the stage")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image", default="ipfs-supervisor-finite-worker:20261002")
    parser.add_argument("--datasets", type=Path, default=SOURCE.parent / "ipfs_datasets")
    parser.add_argument("--kit", type=Path, default=SOURCE.parent / "ipfs_kit")
    parser.add_argument("--lean", type=Path, default=Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1"))
    args = parser.parse_args()
    result = run(args.output, image=args.image, datasets=args.datasets, kit=args.kit, lean=args.lean)
    print(json.dumps({"returncode": result["returncode"], "container_removed": result["container_removed"]}))


if __name__ == "__main__":
    main()
