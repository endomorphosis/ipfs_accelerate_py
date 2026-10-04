"""Execute a fresh signed successor under the existing isolated worker image.

The host reservation attaches to the same explicit pool as scan/CUDA jobs.
The CPU-only container has a separate local accounting pool constrained by its
held host reservation and actual CPU/RAM/PID cgroup. No closed owner is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import run_codebase_inventory_resume_docker as deployment
from . import source_successor_dispatch_fixture as fixture
from . import source_successor_full_scan_fixture as shared

SOURCE = Path(__file__).resolve().parents[3]
MODULE = "benchmarks.agent_supervisor.container_coding.qualify_codebase_signed_successor"
DEFAULT_HOST = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/successor-expansion-resources-20261003-01/configuration.json")
_RETAINED_HOST_LEASES = {}


def _write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _observe_container_limits(container, name):
    template = ('{"Id":{{json .Id}},"Name":{{json .Name}},"HostConfig":{"NanoCpus":{{json .HostConfig.NanoCpus}},'
        '"Memory":{{json .HostConfig.Memory}},"PidsLimit":{{json .HostConfig.PidsLimit}},'
        '"NetworkMode":{{json .HostConfig.NetworkMode}},"Privileged":{{json .HostConfig.Privileged}}},'
        '"Mounts":{{json .Mounts}}}')
    inspected = subprocess.run(["docker", "inspect", "--format", template, container],
        check=True, capture_output=True, text=True, timeout=15)
    if len(inspected.stdout.encode()) > 256 * 1024:
        raise ValueError("bounded actual container inspection required")
    inspection = json.loads(inspected.stdout)
    expected = {"NanoCpus": 12_000_000_000, "Memory": 8 * 1024**3,
        "PidsLimit": 512, "NetworkMode": "none", "Privileged": False}
    if (inspection["Id"] != container or inspection["Name"] != "/" + name
            or fixture.inert.numerical_wire(inspection["HostConfig"]) != fixture.inert.numerical_wire(expected)):
        raise ValueError("actual container limits differ from held host envelope")
    program = ("import json,os,pathlib;root=pathlib.Path('/sys/fs/cgroup');"
        "values={name:(root/name).read_text().strip() for name in ('cpu.max','memory.max','pids.max')};"
        "print(json.dumps({'schema':'signed-successor-container-cgroup-observation@1',"
        "'uid':os.getuid(),'euid':os.geteuid(),'values':values,"
        "'namespaces':{name:os.readlink('/proc/self/ns/'+name) for name in ('pid','mnt','net')}},sort_keys=True))")
    observed = subprocess.run(["docker", "exec", "--user", "1000:1000", container,
        deployment.WORKER_ROOT + "/venv/bin/python", "-I", "-B", "-c", program],
        check=True, capture_output=True, text=True, timeout=15)
    if len(observed.stdout.encode()) > 16_384:
        raise ValueError("bounded actual cgroup observation required")
    cgroup = json.loads(observed.stdout)
    cpu = cgroup["values"]["cpu.max"].split()
    if (type(cgroup["uid"]) is not int or cgroup["uid"] != 1000 or cgroup["euid"] != 1000
            or len(cpu) != 2 or not all(value.isdecimal() for value in cpu)
            or int(cpu[1]) <= 0 or int(cpu[0]) != 12 * int(cpu[1])
            or cgroup["values"]["memory.max"] != str(8 * 1024**3)
            or cgroup["values"]["pids.max"] != "512"):
        raise ValueError("actual CPU RAM PID cgroup differs from declared limits")
    return {"inspection": inspection, "inspection_stdout": inspected.stdout,
        "inspection_stderr": inspected.stderr, "inspection_returncode": inspected.returncode,
        "cgroup": cgroup, "cgroup_stdout": observed.stdout, "cgroup_stderr": observed.stderr,
        "cgroup_returncode": observed.returncode, "process_origin_attested": False}


def run(output, *, source_namespace, source_audit, source_reader_controls, image, datasets, kit, lean,
        host_configuration=DEFAULT_HOST, native_seconds=3000):
    if type(native_seconds) is not int or not 900 <= native_seconds <= 3600:
        raise ValueError("bounded native qualification deadline required")
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh exact signed worker namespace required")
    datasets = Path(datasets).resolve(strict=True)
    sys.path.insert(0, str(datasets))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler
    if Path(resource_scheduler.__file__).resolve() != datasets / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py":
        raise ValueError("exact selected host scheduler producer required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from .container_worker_deployment import OWNER_ENTRY, VALIDATION_ENTRY, WORKER_ENTRY

    host_raw = fixture._read(host_configuration)
    host = fixture.inert.parse(host_raw)
    if host["schema"] != "successor-expansion-host-configuration@1" or host["auto_renew_leases"] is not True:
        raise ValueError("explicit common proof-host authority required")
    scheduler = shared.shared_scheduler(host["state_path"], host["persisted_config"])
    output.mkdir(mode=0o755)
    started = time.monotonic()
    selected = {}
    try:
        selected = {
            "source": deployment._snapshot(SOURCE, output / "source", prefixes=("ipfs_accelerate_py", "benchmarks", "test")),
            "datasets": deployment._snapshot(datasets, output / "datasets", prefixes=("ipfs_datasets_py",)),
            "kit": deployment._snapshot(Path(kit).resolve(strict=True), output / "kit", prefixes=("ipfs_kit_py",)),
        }
        _write(output / "selected-source-snapshot.json", selected)
        deployment._verify_snapshot(selected, output)
        receipt = fixture.stage_closed_successor_dispatch(source_namespace, output / "setup-seed",
            audit_path=source_audit, guard_receipt=source_reader_controls)
        _write(output / "source-setup-seed.json", receipt)
        _write(output / "shared-host-configuration.json", host)
    except BaseException as error:
        _write(output / "staging-failure.json", {"schema": "signed-successor-worker-staging-failure@1",
            "error_type": type(error).__name__, "error": str(error), "native_module_launched": False,
            "container_created": False, "known_actual_setup_epochs": 0, "new_scan_pages": 0,
            "recorded_seconds": time.monotonic() - started})
        raise

    scripts = output / "deployment"
    scripts.mkdir()
    authored_branch = """elif args.model=='successor-authored-format-fixture':
 if not all(instruction_values) or args.semantic_repository is not None or any(residual_values):raise SystemExit('exact successor fixture context required')
 boundary=json.loads(artifact.read_bytes())
 private_access={name:{'read':os.access(name,os.R_OK),'write':os.access(name,os.W_OK),'execute':os.access(name,os.X_OK)} for name in boundary['owner_private_paths']}
 if any(value for checks in private_access.values() for value in checks.values()):raise SystemExit('successor worker can access owner private directory')
 print(json.dumps({'schema':'successor-authored-worker-boundary@1','pid':os.getpid(),'uid':os.getuid(),'euid':os.geteuid(),'gid':os.getgid(),'groups':os.getgroups(),'workspace':str(pathlib.Path.cwd()),'boundary':evidence,'boundary_artifact':str(artifact),'boundary_sha256':digest,'private_access':private_access,'provider_calls':0,'training_steps':0},sort_keys=True),flush=True)
 from benchmarks.agent_supervisor.container_coding.qualify_codebase_signed_successor import run_authored_worker
 raise SystemExit(run_authored_worker(artifact=args.public_instruction_artifact,expected_sha256=args.public_instruction_sha256,task_cid=args.public_instruction_task_cid))
"""
    anchor = "else:\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main"
    if WORKER_ENTRY.count(anchor) != 1:
        raise ValueError("exact deployed worker entry branch required")
    worker = WORKER_ENTRY.replace(anchor, authored_branch + anchor)
    compile(worker, "signed-successor-worker-entry", "exec")
    for name, text in (("owner-worker", OWNER_ENTRY), ("validation-worker", VALIDATION_ENTRY), ("worker-entry", worker)):
        (scripts / name).write_text(text)
    (scripts / "sudoers").write_text("supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n")
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    name = "ipfs-signed-successor-worker-" + str(time.time_ns())
    record = {"schema": "signed-successor-worker-offline-container-execution@1", "image_id": image_id,
        "container_name": name,
        "network": "none", "privileged": False, "cpu_limit": 12, "memory_limit_bytes": 8 * 1024**3,
        "pids_limit": 512, "mode": "fresh_complete_source_model_successor_signed_worker",
        "native_total_timeout_seconds": native_seconds, "native_cooperative_deadline_seconds": native_seconds - 50,
        "container_lifetime_seconds": native_seconds + 180, "inventory_operation_deadline_seconds": 120,
        "signed_admission_operation_deadline_seconds": 180,
        "native_start_deadline_seconds": 30, "staging_seconds": time.monotonic() - started,
        "source_namespace": receipt["source_namespace"], "source_audit": receipt["audit"],
        "inherited_actual_setup_epochs": 2, "inherited_completed_scan_pages": 10,
        "inherited_reference_scan_pages": 1, "new_setup_fitting_epochs": 0, "new_scan_pages_created": 0,
        "host_scheduler_state_path": str(scheduler.state_path), "host_scheduler_configuration": scheduler.config.persisted_dict(),
        "host_configuration_pin": fixture._pin(host_raw), "retained_source_verified_before_execution": True,
        "source_snapshot_scope": "listed sequential Python/resources; no process-origin or atomic checkout attestation",
        "nested_resource_scope": "inner local pool bounded by held host envelope and actual container cgroup",
        "proof_authority": False, "production_activated": False}
    container = lease = None
    try:
        record["host_resources_before_reservation"] = scheduler.snapshot()
        lease = scheduler.acquire(lane=ResourceLane.ORCHESTRATION, cpu_slots=12, memory_mb=8192,
            child_process_slots=12, timeout=90, request_id="signed-successor-worker-container")
        record["host_reservation"] = lease.to_dict()
        try:
            record["container_creation_attempted"] = True
            container = subprocess.check_output(["docker", "create", "--network", "none", "--cpus", "12",
                "--memory", "8g", "--pids-limit", "512", "--user", "0", "--name", name,
                "-v", str(output / "source") + ":" + deployment.WORKER_ROOT + "/source:ro",
                "-v", str(output / "datasets") + ":" + deployment.WORKER_ROOT + "/datasets:ro",
                "-v", str(output / "kit") + ":" + deployment.WORKER_ROOT + "/kit:ro",
                "-v", str(output / "setup-seed") + ":" + deployment.WORKER_ROOT + "/state/transport-seed:ro",
                "-v", str(output / "source-setup-seed.json") + ":" + str(output / "source-setup-seed.json") + ":ro",
                "-v", str(Path(lean).resolve(strict=True)) + ":/toolchains/lean:ro",
                "--entrypoint", "/bin/sleep", image_id, str(native_seconds + 180)], text=True, timeout=30).strip()
            record["container_id"] = container
            subprocess.run(["docker", "start", container], check=True, capture_output=True, timeout=30)
            record["actual_container_limits"] = _observe_container_limits(container, name)
            mounts = record["actual_container_limits"]["inspection"]["Mounts"]
            for local, target in ((output / "source", deployment.WORKER_ROOT + "/source"),
                    (output / "datasets", deployment.WORKER_ROOT + "/datasets"),
                    (output / "kit", deployment.WORKER_ROOT + "/kit"),
                    (output / "setup-seed", deployment.WORKER_ROOT + "/state/transport-seed"),
                    (output / "source-setup-seed.json", str(output / "source-setup-seed.json")),
                    (Path(lean).resolve(strict=True), "/toolchains/lean")):
                matching = [row for row in mounts if row["Destination"] == target]
                if (len(matching) != 1 or matching[0]["Type"] != "bind"
                        or matching[0]["Source"] != str(local) or matching[0]["RW"] is not False):
                    raise ValueError("actual read-only selected mount differs: " + target)
            for filename in ("owner-worker", "validation-worker", "worker-entry"):
                subprocess.run(["docker", "cp", str(scripts / filename), container + ":" + deployment.WORKER_ROOT + "/bin/" + filename],
                    check=True, capture_output=True, timeout=30)
            subprocess.run(["docker", "cp", str(scripts / "sudoers"), container + ":/etc/sudoers.d/finite-worker"],
                check=True, capture_output=True, timeout=30)
            # Rootless Docker maps the host's UID1000 to container root. Read
            # the protected host seed through that root identity, then make an
            # independent container-local copy owned by supervisor UID1000.
            # Its exact absolute locator, bytes, modes and mtimes remain bound.
            transfer_program = """import hashlib,json,os,pathlib,shutil,stat
def inventory(root):
 rows=[];total=0
 for path in [root,*sorted(root.rglob('*'))]:
  info=path.lstat();relative=path.relative_to(root).as_posix()
  if stat.S_ISDIR(info.st_mode):row={'path':relative,'kind':'directory','mode':stat.S_IMODE(info.st_mode)}
  else:
   if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>16*1024**2:raise ValueError('bounded independent regular seed required')
   raw=path.read_bytes();total+=len(raw)
   if total>128*1024**2:raise ValueError('seed byte ceiling exceeded')
   row={'path':relative,'kind':'file','mode':stat.S_IMODE(info.st_mode),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':info.st_mtime_ns}
  rows.append(row)
  if len(rows)>4096:raise ValueError('seed member ceiling exceeded')
 wire=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 return {'members':len(rows),'files':sum(row['kind']=='file' for row in rows),'bytes':total,'inventory_sha256':hashlib.sha256(wire).hexdigest()}
source=pathlib.Path('/opt/ipfs-supervisor/state/transport-seed')
destination=pathlib.Path(DESTINATION)
if destination.exists() or destination.resolve()!=destination:raise ValueError('fresh exact container-local seed required')
before=inventory(source);destination.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(source,destination,copy_function=shutil.copy2)
copied=inventory(destination)
if before!=copied:raise ValueError('container seed bytes or modes differ')
for path in [destination,*destination.rglob('*')]:os.chown(path,1000,1000,follow_symlinks=False)
if inventory(source)!=before or inventory(destination)!=before:raise ValueError('seed changed during private ownership transfer')
print(json.dumps({'schema':'signed-successor-container-seed-transfer@1','source':str(source),'destination':str(destination),'source_before':before,'source_after':inventory(source),'destination_after':inventory(destination),'owner_uid':1000,'owner_gid':1000,'root_mode':stat.S_IMODE(destination.stat().st_mode),'native_owners_opened':False,'proof_authority':False},sort_keys=True),flush=True)
""".replace("DESTINATION", repr(str(output / "setup-seed")))
            transfer = subprocess.run(["docker", "exec", "--user", "0", container, "/usr/local/bin/python",
                "-I", "-B", "-c", transfer_program], check=True, capture_output=True, text=True, timeout=30)
            record["container_seed_transfer"] = {"observation": json.loads(transfer.stdout),
                "stdout": transfer.stdout, "stderr": transfer.stderr, "returncode": transfer.returncode,
                "program_sha256": hashlib.sha256(transfer_program.encode()).hexdigest()}
            access_program = ("import hashlib,json,os,pathlib,stat;"
                f"path=pathlib.Path({str(output / 'setup-seed')!r});"
                "raw=(path/'staged-successor-dispatch.json').read_bytes();"
                "value={'uid':os.getuid(),'euid':os.geteuid(),'seed_uid':path.stat().st_uid,"
                "'seed_gid':path.stat().st_gid,'seed_mode':stat.S_IMODE(path.stat().st_mode),"
                "'manifest_bytes':len(raw),'manifest_sha256':hashlib.sha256(raw).hexdigest()};"
                "assert value['uid']==value['euid']==value['seed_uid']==1000 and value['seed_mode']==448;"
                "print(json.dumps(value,sort_keys=True),flush=True)")
            access = subprocess.run(["docker", "exec", "--user", "1000:1000", container,
                deployment.WORKER_ROOT + "/venv/bin/python", "-I", "-B", "-c", access_program],
                check=True, capture_output=True, text=True, timeout=15)
            record["container_seed_owner_access"] = {"observation": json.loads(access.stdout),
                "stdout": access.stdout, "stderr": access.stderr, "returncode": access.returncode}
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
                f"'owner_private_paths':['/opt/ipfs-supervisor/state','/results/native/private',{str(output / 'setup-seed')!r}],"
                "'validation_repository_roots':['/results/native/repository']};"
                "p=root/'container-boundary.json';p.write_text(json.dumps(boundary,sort_keys=True));p.chmod(0o444)"
            )
            subprocess.run(["docker", "exec", "--user", "0", container, "/usr/local/bin/python", "-I", "-c", setup],
                check=True, capture_output=True, timeout=30)
            subprocess.run(["docker", "cp", container + ":" + deployment.WORKER_ROOT + "/container-boundary.json",
                str(scripts / "container-boundary.json")], check=True, capture_output=True, timeout=30)
            # The independent container-local seed has the same exact locator;
            # the protected host transport remains read-only. Closed owners
            # and the closed source archive are never mounted into Docker.
            record["offline_preflight"] = deployment._preflight(container, output, image_id)
            parent = {"host_scheduler_state_path": str(scheduler.state_path),
                "host_configuration_pin": fixture._pin(host_raw), "host_reservation": lease.to_dict(),
                "container_id": container, "cpu_limit": 12, "memory_limit_bytes": 8 * 1024**3, "pids_limit": 512}
            authority_program = (
                "import sys,pathlib,json,os;root=pathlib.Path('/opt/ipfs-supervisor');"
                "sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')];"
                "from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceSchedulerConfig,GlobalResourceScheduler;"
                "state=pathlib.Path('/results/container-resource-admission.json');"
                "config=ResourceSchedulerConfig.for_proof_host(state_path=state,lane_reservations={},auto_renew_leases=True);"
                "config.total_cpu_slots=min(config.total_cpu_slots,12);config.total_memory_mb=min(config.total_memory_mb,8192);"
                "config.total_child_process_slots=min(config.total_child_process_slots,12);config.total_gpu_memory_mb=None;"
                "config.total_unified_memory_mb=config.total_memory_mb;config.validate();scheduler=GlobalResourceScheduler(config);"
                f"parent={parent!r};"
                "authority={'schema':'successor-dispatch-container-resource-authority@1','state_path':str(state),"
                "'persisted_config':config.persisted_dict(),'auto_renew_leases':True,'lease_ttl_seconds':config.lease_ttl_seconds,"
                "'parent_host_envelope':parent,'namespace':{k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')},"
                "'scope':'local PID accounting inside held host envelope and actual CPU RAM PID cgroup',"
                "'independent_whole_host_pool':False,'shares_host_pid_state':False,'proof_authority':False,"
                "'initial_resources':scheduler.snapshot()};"
                "path=pathlib.Path('/results/container-resource-authority.json');"
                "raw=json.dumps(authority,sort_keys=True,indent=2,allow_nan=False).encode()+b'\\n';"
                "stream=path.open('xb');stream.write(raw);stream.flush();os.fsync(stream.fileno());stream.close();path.chmod(0o444);"
                "print(json.dumps(authority,sort_keys=True),flush=True)"
            )
            authority = subprocess.run(["docker", "exec", "--user", "1000:1000", container,
                deployment.WORKER_ROOT + "/venv/bin/python", "-I", "-B", "-c", authority_program],
                check=True, capture_output=True, text=True, timeout=30)
            record["container_resource_authority"] = json.loads(authority.stdout)
            arguments = ["--output", "/results/native", "--overall-seconds", str(native_seconds - 50),
                "--setup-seed", str(output / "setup-seed"), "--setup-seed-receipt", str(output / "source-setup-seed.json"),
                "--host-configuration", "/results/container-resource-authority.json"]
            with (output / "run.stdout").open("xb") as stdout, (output / "run.stderr").open("xb") as stderr:
                process = subprocess.run(["docker", "exec", "--user", "1000:1000",
                    "-e", "DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean", "-e", "IPFS_ACCELERATE_RUN_LIVE_QUACK=1",
                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    container, deployment.WORKER_ROOT + "/venv/bin/python", "-B", "-m", MODULE, *arguments],
                    stdout=stdout, stderr=stderr, timeout=native_seconds)
            record["returncode"] = process.returncode
            record["host_lease_held_after_native_exit"] = not lease.released
        finally:
            if container is not None:
                try:
                    copied = subprocess.run(["docker", "cp", container + ":/results/.", str(output)],
                        capture_output=True, text=True, timeout=60)
                    record["container_results_copied"] = copied.returncode == 0
                    handoffs = subprocess.run(["docker", "cp", container + ":" + deployment.WORKER_ROOT + "/finite-handoffs",
                        str(output / "handoffs")], capture_output=True, text=True, timeout=30)
                    record["candidate_handoffs_copied"] = handoffs.returncode == 0
                except (OSError, subprocess.TimeoutExpired) as error:
                    record["result_copy_error_type"] = type(error).__name__
            if record.get("container_creation_attempted"):
                record["container_removed"] = deployment._remove_container(name)
                absence = subprocess.run(["docker", "ps", "-a", "-q", "--no-trunc", "--filter", "name=^/" + name + "$"],
                    capture_output=True, text=True, timeout=15)
                record["container_absence_observation"] = {"container_name": name, "container_id": container,
                    "returncode": absence.returncode, "stdout": absence.stdout, "stderr": absence.stderr,
                    "scope": "exact named disposable container absence through the live engine"}
                record["container_removed"] = record["container_removed"] and absence.returncode == 0 and not absence.stdout.strip()
                if not record["container_removed"]:
                    raise RuntimeError("disposable signed successor container cleanup failed")
            record["elapsed_seconds"] = time.monotonic() - started
            _write(output / "container-execution.json", record)
    except BaseException as error:
        record.update(execution_error_type=type(error).__name__, execution_error=str(error)[:2048])
        raise
    finally:
        if lease is not None and (not record.get("container_creation_attempted") or record.get("container_removed") is True):
            lease.release()
        elif lease is not None:
            _RETAINED_HOST_LEASES[name] = lease
            record["host_reservation_retained_for_live_container"] = True
            _write(output / "unsafe-container-cleanup.json", record)
        record["elapsed_seconds"] = time.monotonic() - started
        record["host_resources_after_cleanup"] = scheduler.snapshot()
        if record.get("host_reservation_retained_for_live_container"):
            record["host_owned_resources_after_cleanup"] = {"scope": "live container still reserved", "released": False}
        else:
            record["host_owned_resources_after_cleanup"] = shared.assert_clean(scheduler)
        changed = []
        for rows in selected.values():
            for row in rows:
                path = Path(row["path"])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    changed.append(row["path"])
        record["original_selected_source_changes_after_execution"] = changed
        try:
            deployment._verify_snapshot(selected, output)
            record["retained_source_verified_after_execution"] = True
        except BaseException as error:
            record.update(retained_source_verified_after_execution=False, retained_source_verification_error=str(error))
        record["runtime_used_retained_source_copies"] = record.get("container_creation_attempted", False)
        record["host_reservation_acquired"] = lease is not None
        record["host_reservation_released"] = lease is not None and lease.released
        record["native_module_launched"] = (output / "run.stdout").exists()
        source_reader = fixture.inert.Reader(Path(source_namespace), seconds=120)
        audited = fixture.inert.parse(fixture._read(source_audit, 4 * fixture.inert.MIB))
        record["closed_source_full_archive_preserved_after_execution"] = fixture.inert.same(source_reader.whole_archive(), audited["archive"])
        _write(output / "container-execution-final.json", record)
    if (record.get("returncode") != 0 or not record.get("container_removed") or not record.get("container_results_copied")
            or not record.get("retained_source_verified_after_execution")
            or not record.get("closed_source_full_archive_preserved_after_execution")):
        raise RuntimeError("signed successor qualification failed; retained native evidence identifies the stage")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-namespace", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-reader-controls", type=Path, required=True)
    parser.add_argument("--image", default="ipfs-supervisor-finite-worker:20261002")
    parser.add_argument("--datasets", type=Path, default=SOURCE.parent / "ipfs_datasets")
    parser.add_argument("--kit", type=Path, default=SOURCE.parent / "ipfs_kit")
    parser.add_argument("--lean", type=Path, default=Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1"))
    parser.add_argument("--native-seconds", type=int, default=3000)
    parser.add_argument("--host-configuration", type=Path, default=DEFAULT_HOST)
    args = parser.parse_args()
    result = run(args.output, source_namespace=args.source_namespace, source_audit=args.source_audit,
        source_reader_controls=args.source_reader_controls, image=args.image, datasets=args.datasets, kit=args.kit,
        lean=args.lean, native_seconds=args.native_seconds, host_configuration=args.host_configuration)
    print(json.dumps({key: result.get(key) for key in ("returncode", "container_removed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
