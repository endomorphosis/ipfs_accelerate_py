"""Replay a closed run's public inventory artifact across actual Unix UIDs.

This is a public-reader diagnostic, not a worker, claim, model, publication,
or complete supervisor qualification. Only copied repository bytes change.
The original closed namespace is read sequentially and is never mounted RW.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

SCHEMA = "inventory-public-worker-read-diagnostic@1"
_RETAINED_LEASES = {}
_PROMPT = ("private/launch/state/run/admitted_database_portal_attempts/"
           "5287a14964293069615f087f/implementation-logs/inventory-offset-base-context-capsule.json")
_SELECTED = ("ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_candidate_runner.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_evidence_worker_context.py")
_GIT_ENV = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_COUNT": "0",
            "GIT_TERMINAL_PROMPT": "0"}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _write(path, value):
    path.write_bytes(_wire(value) + b"\n")


def _raw(path, maximum=16 * 1024**2):
    path = Path(path).absolute()
    _need(path.parent.resolve(strict=True) == path.parent, "file ancestor alias refused")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 <= before.st_size <= maximum, "bounded single-link regular file required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mode", "st_nlink", "st_mtime_ns", "st_ctime_ns")
    _need(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key)
              for key in fields), "file changed while reading")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                 "mode": stat.S_IMODE(before.st_mode)}


def _pins(source):
    return {name: _raw(source / name, 4 * 1024**2)[1] for name in _SELECTED}


def _drop(uid, gid):
    def apply():
        os.setgroups([1000])
        os.setgid(gid)
        os.setuid(uid)
    return apply


def _git(root, *args, uid=None):
    process = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-c", "safe.directory=" + str(root),
        "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        env=_GIT_ENV, preexec_fn=None if uid is None else _drop(uid, 1000),
        check=True, capture_output=True, timeout=15)
    return process.stdout.decode().strip()


def _stage(namespace, destination, source):
    # Existing bounded byte transport is stdlib-only and opens no native owners.
    from . import inventory_resume_setup_fixture as transport
    _need(namespace.resolve(strict=True) == namespace and source.resolve(strict=True) == source,
          "exact existing source namespaces required")
    _need(not destination.exists() and destination.parent.resolve(strict=True) == destination.parent,
          "fresh exact diagnostic input directory required")
    closed_raw, closed_pin = _raw(namespace / "container-execution-final.json")
    closed = json.loads(closed_raw)
    _need(closed.get("container_removed") is True and closed.get("host_reservation_released") is True,
          "source qualification container and host lease must be closed")
    context_raw, context_pin = _raw(namespace / "native/public-worker-context.json")
    context = json.loads(context_raw)
    artifact_raw, artifact_pin = _raw(namespace / "native/public-worker-artifact.raw.json", 8 * 1024**2)
    payload = json.loads(artifact_raw)
    prompt_raw, prompt_pin = _raw(namespace / "native" / _PROMPT, 256_000)
    prompt = json.loads(prompt_raw)
    boundary_raw, boundary_pin = _raw(namespace / "native/container-boundary.json", 65_536)
    boundary = json.loads(boundary_raw)
    _need(hashlib.sha256(artifact_raw).hexdigest() == context["sha256"]
          and payload.get("schema") == "supervisor-public-instruction@3"
          and payload.get("task_cid") == context["task_cid"]
          and payload.get("repository") == "/results/native/repository"
          and prompt.get("objective_id") == "INVENTORY-OFFSET"
          and re.fullmatch(r"sha256:[0-9a-f]{64}", boundary["image_id"]) is not None,
          "exact retained public input and image required")
    repository = namespace / "native/repository"
    tree, directories = transport._tree(repository)
    relative = Path(context["artifact"]).relative_to("/results/native/repository")
    actual, actual_pin = _raw(repository / relative, 8 * 1024**2)
    _need(actual == artifact_raw and _git(repository, "rev-parse", "HEAD")
          == payload["manifest"]["payload"]["baseline_commit"], "retained repository/public baseline differs")
    members = []
    for name, fingerprint in sorted(tree.items()):
        _, pin = _raw(repository / name, 128 * 1024**2)
        members.append({"path": name, "bytes": pin["bytes"], "sha256": pin["sha256"],
                        "mode": pin["mode"]})
    destination.mkdir(mode=0o755)
    transport._copy_members(repository, destination / "repository", members, tree)
    for name, raw in (("context.json", context_raw), ("artifact.raw.json", artifact_raw),
                      ("prompt.json", prompt_raw), ("boundary.json", boundary_raw)):
        path = destination / name
        path.write_bytes(raw)
        path.chmod(0o444)
    _need(transport._tree(repository) == (tree, directories), "retained repository changed during staging")
    return {"schema": "inventory-public-worker-read-input@1", "source_namespace": str(namespace),
        "baseline_commit": payload["manifest"]["payload"]["baseline_commit"],
        "image_id": boundary["image_id"], "context": context,
        "manifest_signature": payload["manifest"]["binding"],
        "planning_signature": payload["inventory_plan_admission"]["receipt"]["binding"],
        "source_pins": [closed_pin, context_pin, artifact_pin, prompt_pin, boundary_pin, actual_pin],
        "repository_members": members, "selected_product_sources": _pins(source),
        "preservation_scope": "all consumed source files and full copied repository membership; sequential",
        "private_database_files_supplied": False, "signing_keys_supplied": False}


def _worker(input_dir, workspace, *, refusal=False):
    _need(os.getuid() == os.geteuid() == 1001 and os.getgid() == 1000,
          "actual UID1001/group1000 required")
    sys.path[:0] = ["/opt/ipfs-supervisor/source", "/opt/ipfs-supervisor/datasets", "/opt/ipfs-supervisor/kit"]
    context = json.loads((input_dir / "context.json").read_bytes())
    prompt = (input_dir / "prompt.json").read_text()
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
    attempts = []
    owners = {"CodebaseIndex", "DuckDBASTStore", "AutoencoderRegistry", "CodebaseVerificationCatalog"}

    def guard(frame, event, _argument):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        owner = frame.f_locals.get("self")
        blocked = (module.startswith("ipfs_datasets_py") and
            ((name == "__init__" and type(owner).__name__ in owners)
             or name in {"train_features", "train_codebase_autoencoder", "train_current_codebase_autoencoder"}
             or (name == "train" and "autoencoder" in module)))
        if blocked:
            attempts.append({"module": module, "function": name})
            raise AssertionError("public reader attempted native owners or fitting")

    request = {"artifact": Path(context["artifact"]), "expected_sha256": context["sha256"],
        "task_cid": context["task_cid"], "prompt": prompt, "workspace": workspace}
    sys.setprofile(guard)
    try:
        if refusal:
            try:
                load_public_instruction(**request)
            except (ValueError, subprocess.CalledProcessError) as error:
                _need(not attempts, "native owner/fit attempt during refusal")
                result = {"status": "refused", "error_type": type(error).__name__,
                          "error": str(error)[:1024], "uid": os.getuid(), "gid": os.getgid()}
            else:
                raise AssertionError("mutated public observation was accepted")
        else:
            first, before = load_public_instruction(**request)
            second, after = load_public_instruction(**request)
            _need(first == second and before == after and before["schema"]
                  == "supervisor-public-instruction-inclusion@3", "public replay changed")
            private = [Path("/results/native/private"), Path("/opt/ipfs-supervisor/state")]
            denied = {str(path): {"read": os.access(path, os.R_OK), "write": os.access(path, os.W_OK),
                                  "execute": os.access(path, os.X_OK)} for path in private}
            _need(all(value is False for row in denied.values() for value in row.values()),
                  "worker can access private denial sentinel directories")
            for path in [*[(path / "sentinel") for path in private], Path(context["artifact"])]:
                try:
                    with path.open("ab"):
                        pass
                except PermissionError:
                    pass
                else:
                    raise AssertionError("worker can mutate owner evidence")
            plain = subprocess.run(["/usr/bin/git", "-C", "/results/native/repository", "rev-parse", "HEAD"],
                env=_GIT_ENV, capture_output=True, timeout=10)
            _need(plain.returncode != 0 and b"dubious ownership" in plain.stderr,
                  "unscoped Git ownership guard was disabled")
            result = {"status": "completed", "uid": os.getuid(), "euid": os.geteuid(),
                "gid": os.getgid(), "groups": os.getgroups(), "load_calls": 2,
                "block_sha256": hashlib.sha256(first.encode()).hexdigest(), "raw_receipts": [before, after],
                "private_denial": denied, "private_denial_scope": "owner-only authored empty sentinels",
                "unscoped_git_ownership_refused": True}
    finally:
        sys.setprofile(None)
    result.update(native_owner_attempts=attempts, fitting_attempts=0, provider_calls=0,
                  task_claims=0, edits=0, publications=0, completion_authority=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def _inside(input_dir):
    _need(os.getuid() == os.geteuid() == 0, "diagnostic root must allocate distinct unprivileged children")
    root = Path("/results/native/repository")
    root.parent.mkdir(parents=True)
    root.parent.parent.chmod(0o755)
    root.parent.chmod(0o755)
    shutil.copytree(input_dir / "repository", root)
    for parent, directories, files in os.walk(root, followlinks=False):
        path = Path(parent)
        _need(not path.is_symlink(), "diagnostic copied directory alias")
        os.chown(path, 1000, 1000)
        path.chmod(0o755)
        for name in files:
            item = path / name
            _need(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1,
                  "diagnostic copied file alias")
            os.chown(item, 1000, 1000)
    for path in (Path("/results/native/private"), Path("/opt/ipfs-supervisor/state")):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(mode=0o700, parents=True)
        os.chown(path, 1000, 1000)
        (path / "sentinel").write_text("authored empty private denial sentinel\n")
        os.chown(path / "sentinel", 1000, 1000)
    _git(root, "worktree", "prune", "--expire", "now", uid=1000)
    baseline = _git(root, "rev-parse", "HEAD", uid=1000)
    workspace = Path("/opt/ipfs-supervisor/worktrees/public-reader-diagnostic")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    os.chown(workspace.parent, 1000, 1000)
    workspace.parent.chmod(0o755)
    _git(root, "worktree", "add", "--detach", "-q", str(workspace), baseline, uid=1000)
    environment = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    observed = []

    def child(label, target=workspace, refusal=False):
        argv = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
                "--input", str(input_dir), "--workspace", str(target)]
        if refusal:
            argv.append("--refusal")
        process = subprocess.run(argv, cwd=str(target), env=environment, preexec_fn=_drop(1001, 1000),
            capture_output=True, timeout=35)
        _need(process.returncode == 0, label + " failed: " + process.stderr.decode(errors="replace")[-4096:])
        value = json.loads(process.stdout)
        observed.append({"control": label, "returncode": process.returncode, "observation": value,
            "stdout_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr).hexdigest(),
            "stderr": process.stderr.decode(errors="replace")[-4096:]})

    child("actual_uid1001_two_public_replays_and_denial")
    for label, path in (("allocated_source_mutation_refused", workspace / "calc.py"),
                        ("canonical_source_mutation_refused", root / "README.md")):
        original = path.read_bytes()
        path.write_bytes(original + b"\n# diagnostic-only mutation\n")
        try:
            child(label, refusal=True)
        finally:
            path.write_bytes(original)
    extra = root / "diagnostic_undeclared.py"
    extra.write_text("# diagnostic-only extra tracked source\n")
    os.chown(extra, 1000, 1000)
    _git(root, "add", extra.name, uid=1000)
    try:
        child("extra_tracked_source_refused", refusal=True)
    finally:
        _git(root, "reset", "-q", "--", extra.name, uid=1000)
        extra.unlink()
    foreign = workspace.parent / "foreign-public-reader-diagnostic"
    _git(root, "clone", "--no-hardlinks", "-q", str(root), str(foreign), uid=1000)
    child("foreign_worktree_common_directory_refused", target=foreign, refusal=True)
    child("restored_uid1001_two_public_replays_and_denial")
    final_status = _git(root, "status", "--porcelain", "--untracked-files=all", uid=1000).splitlines()
    _need(_git(root, "rev-parse", "HEAD", uid=1000) == baseline
          and all(row.startswith("?? .runtime/") for row in final_status),
          "diagnostic copied repository was not restored")
    print(json.dumps({"schema": SCHEMA, "status": "completed", "baseline_commit": baseline,
        "observations": observed, "actual_uid1001_children": len(observed), "private_database_files_supplied": False,
        "signing_keys_supplied": False, "native_owners_opened": False, "fitting_attempts": 0,
        "provider_calls": 0, "task_claims": 0, "worker_edits": 0, "publications": 0,
        "mutation_scope": "temporary corruption controls in diagnostic copies only",
        "native_worker_qualified": False, "proof_authority": False, "completion_authority": False},
        sort_keys=True, allow_nan=False), flush=True)


def run(namespace, output, *, source, datasets, kit):
    from . import inventory_resume_setup_fixture as transport
    namespace, source, datasets, kit = [Path(value).absolute() for value in (namespace, source, datasets, kit)]
    output = Path(output).absolute()
    _need(not output.exists() and output.parent.resolve(strict=True) == output.parent,
          "fresh exact output required")
    output.mkdir(mode=0o755)
    staged = _stage(namespace, output / "input", source)
    _write(output / "input-receipt.json", staged)
    helper_raw, helper_pin = _raw(Path(__file__), 1024**2)
    (output / "diagnostic.py").write_bytes(helper_raw)
    (output / "diagnostic.py").chmod(0o444)
    tree_before = transport._tree(namespace / "native/repository")
    sys.path.insert(0, str(datasets))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler
    _need(Path(resource_scheduler.__file__).resolve() == datasets
          / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py",
          "host scheduler came from another datasets generation")
    scheduler = resource_scheduler.GlobalResourceScheduler(resource_scheduler.ResourceSchedulerConfig.for_proof_host(
        state_path=output / "host-resource-admission.json", lane_reservations={}, auto_renew_leases=True))
    record = {"schema": "inventory-public-worker-read-container@1", "image_id": staged["image_id"],
        "helper": helper_pin, "cpu_limit": 1, "memory_limit_bytes": 2 * 1024**3, "pids_limit": 128,
        "network": "none", "privileged": False, "native_worker_qualified": False,
        "source_namespace": str(namespace), "input_receipt_sha256": hashlib.sha256(_wire(staged)).hexdigest()}
    name = "ipfs-public-inventory-reader-" + str(time.time_ns())
    container, lease = None, None
    started = time.monotonic()
    try:
        record["host_resources_before_reservation"] = scheduler.snapshot()
        lease = scheduler.acquire(lane=resource_scheduler.ResourceLane.ORCHESTRATION,
            cpu_slots=1, memory_mb=2048, child_process_slots=1, timeout=60,
            request_id="inventory-public-worker-read-container")
        record["host_reservation"] = lease.to_dict()
        container = subprocess.check_output(["docker", "create", "--network", "none", "--cpus", "1",
            "--memory", "2g", "--pids-limit", "128", "--user", "0", "--name", name,
            "-v", str(source) + ":/opt/ipfs-supervisor/source:ro",
            "-v", str(datasets) + ":/opt/ipfs-supervisor/datasets:ro",
            "-v", str(kit) + ":/opt/ipfs-supervisor/kit:ro",
            "-v", str(output / "input") + ":/diagnostic-input:ro",
            "-v", str(output / "diagnostic.py") + ":/diagnostic.py:ro",
            "--entrypoint", "/bin/sleep", staged["image_id"], "240"], text=True, timeout=30).strip()
        record["container_id"] = container
        subprocess.run(["docker", "start", container], check=True, capture_output=True, timeout=30)
        process = subprocess.run(["docker", "exec", container, "/opt/ipfs-supervisor/venv/bin/python",
            "-I", "-B", "/diagnostic.py", "--inside", "--input", "/diagnostic-input"],
            capture_output=True, timeout=180)
        (output / "run.stdout").write_bytes(process.stdout)
        (output / "run.stderr").write_bytes(process.stderr)
        record.update(returncode=process.returncode, host_lease_held_after_child_exit=not lease.released,
                      stdout_sha256=hashlib.sha256(process.stdout).hexdigest(),
                      stderr_sha256=hashlib.sha256(process.stderr).hexdigest())
        _need(process.returncode == 0, "actual UID diagnostic failed; retained stderr identifies the phase")
        observation = json.loads(process.stdout)
        _need(observation.get("schema") == SCHEMA and observation.get("status") == "completed",
              "exact diagnostic result required")
        _write(output / "result.json", observation)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2048])
        raise
    finally:
        if container is not None:
            removed = subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)
            absent = subprocess.run(["docker", "ps", "-a", "-q", "--no-trunc", "--filter", "name=^/" + name + "$"],
                                    capture_output=True, timeout=30)
            record["container_removed"] = removed.returncode == 0 and absent.returncode == 0 and not absent.stdout.strip()
        else:
            record["container_removed"] = True
        if lease is not None and record["container_removed"]:
            lease.release()
        elif lease is not None:
            _RETAINED_LEASES[name] = lease
        record.update(host_reservation_released=lease is not None and lease.released,
                      host_resources_after_cleanup=scheduler.snapshot(), recorded_seconds=time.monotonic() - started)
        record["source_repository_membership_unchanged"] = transport._tree(namespace / "native/repository") == tree_before
        record["source_consumed_files_unchanged"] = all(_raw(Path(pin["path"]))[1] == pin
                                                      for pin in staged["source_pins"])
        record["selected_product_sources_unchanged"] = _pins(source) == staged["selected_product_sources"]
        record["diagnostic_helper_unchanged"] = _raw(Path(__file__), 1024**2)[1] == helper_pin
        _write(output / "container-execution-final.json", record)
    _need(record["container_removed"] and record["host_reservation_released"]
          and record["source_repository_membership_unchanged"] and record["source_consumed_files_unchanged"]
          and record["selected_product_sources_unchanged"] and record["diagnostic_helper_unchanged"],
          "diagnostic cleanup or consumed-source preservation failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--refusal", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source-namespace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--kit", type=Path)
    args = parser.parse_args()
    if args.worker:
        _worker(args.input, args.workspace, refusal=args.refusal)
    elif args.inside:
        _inside(args.input)
    else:
        _need(all(value is not None for value in (args.source_namespace, args.output, args.source, args.datasets, args.kit)),
              "all selected source paths and fresh output are required")
        print(json.dumps(run(args.source_namespace, args.output, source=args.source,
            datasets=args.datasets, kit=args.kit), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
