"""Actual ignored-but-present source capture control on a fresh tiny Git repo.

This source-only qualification opens one native DuckDB AST/catalog owner. It
never opens a model registry, executes repository code, fits, infers or starts a
worker. Leases use genuine conservative host admission, not kernel/RSS limits.
The retained observations do not attest execution or grant proof authority.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time

SCHEMA = "codebase-source-delta-ignore-native-qualification@1"
REPOSITORY_ID = "qualification:source-delta-ignore"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def patch(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def physical(path):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "bounded regular physical source required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        first = os.fstat(stream.fileno())
        raw = stream.read(65537)
        last = os.fstat(stream.fileno())
    after = path.lstat()
    witness = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
                            row.st_mode, row.st_nlink)
    require(len(raw) <= 65536 and witness(before) == witness(first) == witness(last) == witness(after),
            "physical source changed during observation")
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "source_cid": cid_for_bytes(raw), "regular": True}


def git(repository, *arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
                       GIT_NO_REPLACE_OBJECTS="1", LC_ALL="C")
    result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
        "-C", str(repository), *arguments], env=environment, stdin=subprocess.DEVNULL,
        check=True, capture_output=True, timeout=10)
    return result.stdout


def pins(output, delta):
    paths = {name: Path(importlib.import_module(name).__file__).resolve()
             for name in delta._implementation()["files"]}
    paths["qualification_harness"] = Path(__file__).resolve()
    directory = output / "producers"
    directory.mkdir()
    rows = []
    for name, path in sorted(paths.items()):
        raw = path.read_bytes()
        target = directory / (name + ".py")
        with target.open("xb") as stream:
            stream.write(raw)
        rows.append({"name": name, "path": str(path), "copy": target.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    result = {"schema": "codebase-source-delta-selected-producers@1", "files": rows,
              "execution_attestation": False, "scope": "listed_local_files_only"}
    write(output / "generation-inputs.json", result)
    require(len(rows) == 23, "exact twenty-two product sources plus this harness required")
    return result


def verify_pins(output, selected):
    for row in selected["files"]:
        raw = Path(row["path"]).read_bytes()
        require(raw == (output / row["copy"]).read_bytes() and len(raw) == row["bytes"]
                and hashlib.sha256(raw).hexdigest() == row["sha256"], "selected source changed: " + row["name"])


def qualify(output, *, timeout_seconds=360):
    import duckdb
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseCatalogLimits
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig, ResourceLane)

    require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 600,
            "bounded qualification deadline required")
    output = Path(output).absolute()
    require(output.parent.resolve(strict=True) == output.parent and not output.exists() and not output.is_symlink(),
            "fresh canonical output namespace required")
    output.mkdir()
    started = time.monotonic()
    deadline = started + timeout_seconds
    report = {"schema": SCHEMA, "qualified": False, "pid": os.getpid(),
        "scope": "two_member_ignored_but_physically_present_capture_removal",
        "phases": [], "model_registry_open_attempts": 0, "training_attempts": 0, "inference_attempts": 0,
        "new_fitting_epochs": 0, "repository_code_executed": False, "worker_launched": False,
        "proof_authority": False, "admission_authority": False, "physical_absence_verified": False,
        "numerical_reuse": False, "model_advanced": False, "source_connection_closed": False}
    scheduler = connection = None

    def phase(name, action):
        require(time.monotonic() < deadline, "overall ignore qualification deadline exceeded")
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        begin = time.monotonic()
        try:
            result = action()
        except BaseException:
            row.update(status="failed", elapsed_seconds=time.monotonic() - begin)
            raise
        row.update(status="completed", elapsed_seconds=time.monotonic() - begin)
        return result

    def forbidden(counter, label):
        def call(*args, **kwargs):
            report[counter] += 1
            raise AssertionError("source-only ignore qualification attempted " + label)
        return call

    try:
        selected = pins(output, delta)
        scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
            state_path=output / "resource-admission.json", lane_reservations={}, auto_renew_leases=True))
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        with scheduler.acquire(lane=ResourceLane.SNAPSHOT_EVALUATION, cpu_slots=2, memory_mb=2048,
                child_process_slots=2, timeout=min(30, timeout_seconds)) as parent, ExitStack() as guards:
            targets = ((AutoencoderRegistry, "__init__", "model_registry_open_attempts"),
                (training, "train_current_codebase_features", "training_attempts"),
                (features, "train_projection_features", "training_attempts"),
                (runtimes.SourceBoundCodebaseFeatureRuntime, "train", "training_attempts"),
                (training, "infer_current_codebase_features", "inference_attempts"),
                (features, "infer_projection_features", "inference_attempts"),
                (resume, "_worker", "inference_attempts"))
            report["named_execution_guards"] = []
            for owner, name, counter in targets:
                label = owner.__module__ + "." + owner.__name__ + "." + name if isinstance(owner, type) else owner.__name__ + "." + name
                report["named_execution_guards"].append({"name": label, "counter": counter})
                guards.enter_context(patch(owner, name, forbidden(counter, label)))
            repository = output / "repository"
            repository.mkdir()
            (repository / "normal.py").write_bytes(b"def normal(n: int) -> int:\n    return n + 1\n")
            for args in (("init", "--quiet"), ("config", "user.name", "Source delta ignore qualification"),
                    ("config", "user.email", "source-delta-ignore@example.invalid"), ("add", "normal.py"),
                    ("commit", "--quiet", "--no-verify", "-m", "Authored tracked source")):
                phase("git_" + args[0], lambda args=args: git(repository, *args))
            # Keep both complete captures in the same working acquisition
            # mode. Without this stable tracked edit, hiding the last
            # untracked file switches git-working -> git-clean and correctly
            # changes normal.py's entry metadata despite identical bytes.
            with (repository / "normal.py").open("ab") as stream:
                stream.write(b"# Stable authored working source in both captures.\n")
            ignored = repository / "ignored.py"
            ignored.write_bytes(b"def ignored(n: int) -> int:\n    return n + 1001\n")
            report["physical_presence_before"] = physical(ignored)
            private = output / "private"
            private.mkdir(mode=0o700)
            connection = duckdb.connect(str(private / "source.duckdb"), config={"threads": 1, "memory_limit": "128MB"})
            store = DuckDBASTStore(connection=connection)
            artifacts = ImmutableCAS(private / "source-artifacts")
            index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                catalog=CodebaseCatalog(store, artifacts, limits=CodebaseCatalogLimits(max_entries=16)))
            options = {"parent_lease": parent, "timeout_seconds": 120.0, "memory_mb": 1024}
            preparation = {"repository_id": REPOSITORY_ID, "limits": CodebaseScanLimits(16, 4096), **options}
            previous = phase("publish_previous_source", lambda: index.prepare_current(repository,
                operation_id="ignored-before", expected_head=None, **preparation))
            write(output / "previous-publication-receipt.json", previous.to_dict())
            write(output / "previous-manifest.json", index.load(previous.manifest_cid).to_dict())
            exclude = repository / ".git/info/exclude"
            require(exclude.is_file() and not exclude.is_symlink(), "fresh local exclude file required")
            with exclude.open("ab") as stream:
                stream.write(b"\nignored.py\n")
            current = phase("publish_current_source", lambda: index.prepare_current(repository,
                operation_id="ignored-after", expected_head=previous.head, **preparation))
            write(output / "current-publication-receipt.json", current.to_dict())
            write(output / "current-manifest.json", index.load(current.manifest_cid).to_dict())
            record = phase("build_default_source_delta", lambda: delta.build_current_codebase_source_delta(
                index, repository, previous_head=previous.head, expected_head=current.head, **options))
            value = record.to_dict()
            require(value["coverage"]["previous_entries"] == 2 and value["coverage"]["current_entries"] == 1
                and value["coverage"]["union_entries"] == 2 and value["coverage"]["classifications"] ==
                    {"retained": 1, "changed": 0, "added": 0, "removed": 1}, "exact two-to-one captured union required")
            row = next(item for item in value["ledger"] if item["source_key"] == "raw:" + b"ignored.py".hex())
            require(row["classification"] == "removed" and row["current"] is None
                and row["previous"]["entry"]["source_cid"] == report["physical_presence_before"]["source_cid"]
                and row["source_bytes_comparison"] == row["ast_identity_comparison"] == "unavailable",
                    "ignored source removal/comparability differs")
            require(value["removal_scope"] == "absent_from_current_complete_capture"
                and value["physical_absence_verified"] is False and value["numerical_reuse"] is False
                and value["model_advanced"] is False and len(value["authority"]) == 14
                and all(flag is False for flag in value["authority"].values()), "capture-only authority required")
            write(output / "optimized-source-delta.json", {"artifact_cid": record.artifact_cid, "value": value})
            phase("receive_current_source_delta", lambda: delta.validate_current_codebase_source_delta(record, index,
                repository, **options))
            report["physical_presence_after"] = physical(ignored)
            require(report["physical_presence_before"] == report["physical_presence_after"],
                    "ignored source must remain physically present with exact bytes")
            report.update(receiving_verified=True, coverage=value["coverage"], optimized_cid=record.artifact_cid,
                previous_head=previous.head.to_dict(), current_head=current.head.to_dict(),
                removal_scope=value["removal_scope"], authority=value["authority"], physical_bytes_unchanged=True,
                default_operation_timeout_seconds=120, source_owner_reopens=0,
                current_owner_counts={"source": 1, "model": 0})
            connection.close()
            connection = None
            report["source_connection_closed"] = True
            report["current_owner_counts"] = {"source": 0, "model": 0}
            verify_pins(output, selected)
        resources = scheduler.snapshot()
        require(resources["active_lease_count"] == resources["active_child_lease_count"] == resources["waiting_request_count"] == 0
                and all(amount == 0 for amount in resources["allocated"].values()), "source reservations not drained")
        require(all(report[counter] == 0 for counter in ("model_registry_open_attempts", "training_attempts", "inference_attempts")),
                "source-only execution guards failed")
        report.update(qualified=True, final_resources=resources)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if connection is not None:
            connection.close()
            report["source_connection_closed"] = True
        if scheduler is not None:
            report["final_resources"] = scheduler.snapshot()
        report["recorded_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=360)
    args = parser.parse_args()
    result = qualify(args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps({"qualified": result["qualified"], "recorded_seconds": result["recorded_seconds"],
        "coverage": result["coverage"], "model_registry_open_attempts": result["model_registry_open_attempts"],
        "training_attempts": result["training_attempts"], "inference_attempts": result["inference_attempts"]}))


if __name__ == "__main__":
    main()
