"""Actual 300-member structural successor qualification without model owners.

The fixture explicitly publishes two source generations, then observes their
complete captured inventories. It never imports or executes repository source.
Native observations and the separate archive reader do not attest execution or
grant proof/admission authority. Resource leases are admission accounting, not
kernel enforcement or a process RSS limit.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import qualify_codebase_inventory_resume as authored
from . import qualify_codebase_inventory_scan as base

SCHEMA = "codebase-source-delta-native-qualification@1"
REPOSITORY_ID = "qualification:source-delta"


def _open(output):
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseCatalogLimits
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    private = output / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    authored.require(private.resolve() == private and not private.is_symlink()
                     and private.stat().st_mode & 0o077 == 0, "private source owner required")
    connection = duckdb.connect(str(private / "source.duckdb"),
                               config={"threads": 1, "memory_limit": "128MB"})
    try:
        store = DuckDBASTStore(connection=connection)
        artifacts = ImmutableCAS(private / "source-artifacts")
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
            catalog=CodebaseCatalog(store, artifacts, limits=CodebaseCatalogLimits(max_entries=512)))
    except BaseException:
        connection.close()
        raise
    return index, connection


def _source_owner(index, connection):
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_projection_replay import ASTS_CATALOG_TABLES
    tables = {}
    for name in ASTS_CATALOG_TABLES:
        rows = connection.execute('SELECT * FROM "' + name + '" ORDER BY ALL').fetchall()
        tables[name] = {"rows": len(rows), "sha256": hashlib.sha256(base._wire(rows)).hexdigest()}
    return {"current_head": index.current(REPOSITORY_ID).to_dict(), "tables": tables}


def _pins(output, delta):
    paths = {name: Path(importlib.import_module(name).__file__).resolve()
             for name in delta._implementation()["files"]}
    paths["qualification_harness"] = Path(__file__).resolve()
    paths["authored_repository_fixture"] = Path(authored.__file__).resolve()
    destination = output / "producers"
    destination.mkdir()
    rows = []
    for name, path in sorted(paths.items()):
        raw = path.read_bytes()
        copy = destination / (name + ".py")
        with copy.open("xb") as stream:
            stream.write(raw)
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    record = {"schema": "codebase-source-delta-selected-producers@1", "files": rows,
              "execution_attestation": False, "scope": "listed_local_files_only"}
    base._write(output / "generation-inputs.json", record)
    return record


def _verify_pins(output, selected):
    for row in selected["files"]:
        raw = Path(row["path"]).read_bytes()
        authored.require(raw == (output / row["copy"]).read_bytes()
            and len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "selected source changed: " + row["name"])


def qualify(output, *, timeout_seconds=900):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits, StaleCodebaseError
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, deadline = time.monotonic(), time.monotonic() + timeout_seconds
    report = {"schema": SCHEMA, "qualified": False, "scope": "300_member_structural_source_successor",
              "pid": os.getpid(),
              "phases": [], "controls": [], "model_registry_open_attempts": 0,
              "training_attempts": 0, "inference_attempts": 0,
              "new_fitting_epochs": 0, "source_owner_reopens": 0,
              "repository_code_executed": False, "proof_authority": False,
              "admission_authority": False, "production_default_activated": False,
              "cuda_qualified": False, "384d_qualified": False,
              "physical_absence_verified": False}
    scheduler, connection = None, None

    def remaining():
        seconds = deadline - time.monotonic()
        authored.require(seconds > 0, "overall source qualification deadline exceeded")
        return min(seconds, 120)

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - started
        base._progress(output / "progress.json", report)

    def phase(name, action):
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        progress()
        began = time.monotonic()
        try:
            result = action()
        except BaseException:
            row.update(status="failed", elapsed_seconds=time.monotonic() - began)
            progress()
            raise
        row.update(status="completed", elapsed_seconds=time.monotonic() - began)
        progress()
        return result

    def forbidden(counter):
        def call(*args, **kwargs):
            report[counter] += 1
            raise AssertionError("source-only qualification attempted " + counter)
        return call

    def refuse(name, action, allowed):
        began = time.monotonic()
        try:
            action()
        except allowed as error:
            row = {"name": name, "refused": True, "error_type": type(error).__name__,
                   "error": str(error), "elapsed_seconds": time.monotonic() - began}
            report["controls"].append(row)
            base._assert_clean(scheduler)
            progress()
            return row
        raise AssertionError("control was accepted: " + name)

    try:
        selected = _pins(output, delta)
        scheduler = authored._scheduler(output)
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        repository = output / "repository"
        phase("author_300_member_repository", lambda: authored._sources(repository))
        index, connection = _open(output)
        with ExitStack() as guards:
            guards.enter_context(base._patch(AutoencoderRegistry, "__init__", forbidden("model_registry_open_attempts")))
            for module, name in ((training, "train_current_codebase_features"),
                                 (features, "train_projection_features")):
                guards.enter_context(base._patch(module, name, forbidden("training_attempts")))
            for module, name in ((training, "infer_current_codebase_features"),
                                 (features, "infer_projection_features"), (resume, "_worker")):
                guards.enter_context(base._patch(module, name, forbidden("inference_attempts")))
            preparation = {"repository_id": REPOSITORY_ID, "limits": CodebaseScanLimits(512, 65536),
                           "scheduler": scheduler, "memory_mb": 1024}
            previous = phase("publish_previous_source", lambda: index.prepare_current(repository,
                operation_id="source-delta-initial", expected_head=None, timeout_seconds=remaining(), **preparation))
            base._write(output / "previous-publication-receipt.json", previous.to_dict())
            base._write(output / "previous-manifest.json", index.load(previous.head.manifest_cid).to_dict())
            (repository / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
            (repository / "bulk000.py").unlink()
            (repository / "bulk001.py").rename(repository / "renamed.py")
            (repository / "added.py").write_text("def added(n: int) -> int:\n    return n + 700\n")
            authored._git(repository, "add", "-A")
            authored._git(repository, "commit", "--quiet", "--no-verify", "-m", "Authored source successor")
            current = phase("publish_current_source", lambda: index.prepare_current(repository,
                operation_id="source-delta-successor", expected_head=previous.head,
                timeout_seconds=remaining(), **preparation))
            base._write(output / "current-publication-receipt.json", current.to_dict())
            base._write(output / "current-manifest.json", index.load(current.head.manifest_cid).to_dict())
            before = _source_owner(index, connection)
            original_artifacts = base._files(index.artifacts.root)
            base._write(output / "native-source-owner-before.json", before)
            base._write(output / "source-artifacts-before.json", original_artifacts)

            def build(optimized):
                return delta.build_current_codebase_source_delta(index, repository,
                    previous_head=previous.head, expected_head=current.head, optimized=optimized,
                    scheduler=scheduler, timeout_seconds=remaining())

            optimized = phase("build_default_source_delta", lambda: build(True))
            reference = phase("build_reference_source_delta", lambda: build(False))
            for name, record in (("optimized", optimized), ("reference", reference)):
                base._write(output / (name + "-source-delta.json"),
                            {"artifact_cid": record.artifact_cid, "value": record.to_dict()})
            left, right = optimized.to_dict(), reference.to_dict()
            authored.require(left.pop("optimized") is True and right.pop("optimized") is False and left == right,
                             "default/reference source delta differs")
            coverage = optimized.to_dict()["coverage"]
            authored.require(coverage["previous_entries"] == coverage["current_entries"] == 300
                and coverage["union_entries"] == 302
                and coverage["classifications"] == {"retained": 297, "changed": 1, "added": 2, "removed": 2},
                "complete source change accounting differs")
            report.update(coverage=coverage, previous_head=previous.head.to_dict(), current_head=current.head.to_dict(),
                          optimized_cid=optimized.artifact_cid, reference_cid=reference.artifact_cid,
                          default_reference_equal=True)

            def receive(record=optimized):
                return delta.validate_current_codebase_source_delta(record, index, repository,
                    scheduler=scheduler, timeout_seconds=remaining())

            phase("receive_current_source_delta", receive)
            dirty = repository / "calc.py"
            with base._restore_bytes(dirty):
                dirty.write_text("def increment(n: int) -> int:\n    return n + 999\n")
                phase("refuse_same_head_source_drift", lambda: refuse("same_head_source_drift", receive,
                    (StaleCodebaseError, delta.CodebaseSourceDeltaError)))
            cancelled = threading.Event()
            cancelled.set()
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
            phase("refuse_precancelled_receiving", lambda: refuse("precancelled_receiving",
                lambda: delta.validate_current_codebase_source_delta(optimized, index, repository,
                    scheduler=scheduler, cancel_event=cancelled, timeout_seconds=remaining()), (LeaseCancelledError,)))
            ast_revision = current.head.ast_revision_id
            revision = connection.execute("SELECT revision FROM source_revisions WHERE revision_id=?", [ast_revision]).fetchone()[0]
            try:
                connection.execute("UPDATE source_revisions SET revision=? WHERE revision_id=?", ["tampered-native", ast_revision])
                from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStoreIntegrityError
                phase("refuse_native_sql_drift", lambda: refuse("native_sql_drift", receive,
                    (DuckDBASTStoreIntegrityError, delta.CodebaseSourceDeltaError, StaleCodebaseError)))
            finally:
                connection.execute("UPDATE source_revisions SET revision=? WHERE revision_id=?", [revision, ast_revision])
            old_index = index
            before_owner = {"pid": os.getpid(), "index_id": id(index), "connection_id": id(connection),
                            "current_head": index.current(REPOSITORY_ID).to_dict()}
            connection.close()
            connection = None
            index, connection = _open(output)
            report["source_owner_reopens"] += 1
            after_owner = {"pid": os.getpid(), "index_id": id(index), "connection_id": id(connection),
                           "current_head": index.current(REPOSITORY_ID).to_dict()}
            authored.require(index is not old_index and before_owner["connection_id"] != id(connection),
                             "source owner reopen did not create a fresh native connection")
            report["cold_owner_reopen"] = {"before": before_owner, "after": after_owner,
                "old_connection_closed": True, "new_source_owner": True,
                "same_process": True, "new_process_claimed": False}
            rehydrated = delta.load_codebase_source_delta(index.artifacts, optimized.artifact_cid)
            phase("cold_receive_current_source_delta", lambda: delta.validate_current_codebase_source_delta(
                rehydrated, index, repository, scheduler=scheduler, timeout_seconds=remaining()))
            authored.require(rehydrated._payload == optimized._payload, "cold source delta bytes differ")
            after = _source_owner(index, connection)
            authored.require(before == after, "native source head/relational rows changed during receiving")
            base._write(output / "native-source-owner-after.json", after)
            ending = {row["path"]: row for row in base._files(index.artifacts.root)}
            authored.require(all(ending[row["path"]] == row for row in original_artifacts),
                             "original captured source/AST artifacts changed")
            _verify_pins(output, selected)
            authored.require(all(report[name] == 0 for name in (
                "model_registry_open_attempts", "training_attempts", "inference_attempts")),
                "source-only qualification attempted model work")
            report.update(qualified=True, native_owner_unchanged=True, original_source_artifacts_preserved=True,
                          cold_receiving_verified=True, selected_producers_unchanged=True)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
    finally:
        if connection is not None:
            connection.close()
        if scheduler is not None:
            report["final_resources"] = base._assert_clean(scheduler)
        report["recorded_seconds"] = time.monotonic() - started
        base._write(output / "result.json", report)
        progress()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = qualify(args.output)
    print(json.dumps({"qualified": result["qualified"], "recorded_seconds": result["recorded_seconds"],
                      "error": result.get("error")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
