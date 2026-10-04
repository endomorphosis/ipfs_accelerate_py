"""Closed successor setup transport and bounded fresh scan processes.

Staging uses only guarded standard-library reads. Native imports are lazy and
only reopen the new materialized copy. A chunk uses the caller's exact shared
scheduler path/configuration and executes at most two new inference pages.
Historical prefixes are explicitly excluded; this helper fits nothing.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from unittest.mock import patch

from . import audit_codebase_source_successor as inert

SCHEMA = "source-successor-materialized-setup@1"
CHUNK_SCHEMA = "source-successor-fresh-scan-chunk@1"
REQUEST_SCHEMA = "source-successor-fresh-scan-chunk-request@1"
REPOSITORY_ID = "qualification:source-successor"
MODULE = "benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture"
SOURCE_NAMESPACE = "/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02"
SOURCE_AUDIT_SHA256 = "c096ce302f556af7f1733372de8b54cfb1f56ad23561f7ab78a860e9f0504fea"
SOURCE_AUDIT_BYTES = 482926
SOURCE_GUARD_SHA256 = "8e239c4d915f49568dfda7a0e94f37c14b964f3677f19f881f02bfaaa5ec85cf"
SOURCE_GUARD_BYTES = 9593
MAX_COPIED_FILES = 2048
MAX_COPIED_BYTES = 128 * inert.MIB
EVIDENCE_NAMES = ("previous-manifest.json", "current-manifest.json", "previous-publication-receipt.json",
    "current-publication-receipt.json", "root-training-record.json", "child-training-record.json",
    "source-delta.json", "checkpoint-states-after.json", "owners-before.json", "owners-after-cold.json", "generation-inputs.json")
EXCLUDED_EXPORT_NAMES = ("scan-root.json", "reference-scan-root.json", "prefix-page.json",
    "reference-prefix-page.json", "successor-selection.json", "reference-successor-selection.json")
REQUEST_FIELDS = {"schema", "output", "run_number", "root_cid", "cursor", "version_id", "expected_head",
    "checkpoint_states", "expected_registry_owner_generation", "scheduler_state_path", "scheduler_configuration",
    "timeout_seconds", "max_pages", "materialization_receipt_sha256", "helper_sha256"}


class SuccessorFullScanFixtureError(ValueError):
    """The fixed closed seed or explicit fresh chunk binding differs."""


def need(condition, message):
    if not condition:
        raise SuccessorFullScanFixtureError(message)


def pin(raw):
    return {"bytes": len(raw), "sha256": inert.sha(raw)}


def native_json(value):
    """Normalize documented DuckDB tuple rows to finite inert JSON lists."""
    return inert.parse(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def canonical_path(path, *, exists=True):
    value = Path(path).absolute()
    need(value.resolve(strict=exists) == value and (value.exists() if exists else not value.exists()),
         "canonical fresh/existing absolute path required")
    for item in (value, *value.parents):
        need(not item.is_symlink(), "path traverses a symlink")
    return value


def read_absolute(path, maximum=16 * inert.MIB):
    value = canonical_path(path)
    return inert.Reader(value.parent, seconds=120).raw(value.name, maximum)


def write_exclusive(path, value):
    raw = inert.numerical_wire(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def source_bundle(source, audit_path, guard_receipt):
    """Verify the fixed frozen report, reader controls, result and full tree."""
    source = canonical_path(source)
    need(str(source) == SOURCE_NAMESPACE, "fixed independently audited successor seed required")
    audit_raw = read_absolute(audit_path, 2 * inert.MIB)
    guard_raw = read_absolute(guard_receipt, inert.MIB)
    need(pin(audit_raw) == {"bytes": SOURCE_AUDIT_BYTES, "sha256": SOURCE_AUDIT_SHA256}
         and pin(guard_raw) == {"bytes": SOURCE_GUARD_BYTES, "sha256": SOURCE_GUARD_SHA256},
         "final audited seed and executed reader-control pins differ")
    report, guard = inert.parse(audit_raw), inert.parse(guard_raw)
    reader_raw = read_absolute(Path(inert.__file__))
    need(report["qualified"] is True and report["preserved"] is True and report["errors"] == []
         and report["namespace"] == guard["namespace"] == str(source)
         and guard["qualified"] is True and guard["tests"] == 93
         and all(type(guard[field]) is int and guard[field] == 0 for field in ("failures", "errors", "skipped"))
         and guard["current_pins_unchanged"] is True
         and pin(reader_raw) == {key: guard["reader"][key] for key in ("bytes", "sha256")}
         == {key: report["reader"][key] for key in ("bytes", "sha256")}, "source qualification/reader generation join differs")
    reader = inert.Reader(source, seconds=120)
    archive = reader.whole_archive()
    need(inert.same(archive, report["archive"]), "closed audited archive membership/bytes/modes changed")
    result_raw = reader.raw("result.json")
    need(pin(result_raw) == {key: guard["source_native_result"][key] for key in ("bytes", "sha256")}
         == {key: report["audited_result"][key] for key in ("bytes", "sha256")}, "closed native result generation differs")
    need(guard["source_native_result"]["path"] == str(source / "result.json"), "native result namespace differs")
    producers = reader.json("generation-inputs.json")
    for row in producers["files"]:
        raw = read_absolute(row["path"], 4 * inert.MIB)
        need(pin(raw) == {key: row[key] for key in ("bytes", "sha256")}, "current listed seed producer differs: " + row["name"])
        need(pin(reader.raw(row["copy"], 4 * inert.MIB)) == pin(raw), "retained/current seed producer bytes differ")
    return source, reader, report, guard, archive, {"audit": pin(audit_raw), "guard": pin(guard_raw), "reader": pin(reader_raw), "native_result": pin(result_raw)}


def allowed_member(path):
    return path in {"private/source.duckdb", "private/model.duckdb"} or path.startswith(
        ("private/source-artifacts/", "private/model-artifacts/", "repository/"))


def copy_regular(reader, row, destination):
    """Create independent bytes and preserve the audited file mode/mtime."""
    need(row["kind"] == "file" and type(row["nlink"]) is int and row["nlink"] == 1
         and type(row["bytes"]) is int and 0 <= row["bytes"] <= 16 * inert.MIB,
         "bounded single-link source member required")
    raw = reader.raw(row["path"], 16 * inert.MIB)
    need(pin(raw) == {key: row[key] for key in ("bytes", "sha256")}, "audited source copy bytes differ")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    destination.chmod(row["mode"])
    os.utime(destination, ns=(row["mtime_ns"], row["mtime_ns"]))
    info = destination.lstat()
    source_info = (reader.root / row["path"]).lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and (info.st_dev, info.st_ino) != (source_info.st_dev, source_info.st_ino),
         "materialization requires independent regular bytes")
    copied = inert.Reader(destination.parent, seconds=120).raw(destination.name, 16 * inert.MIB)
    need(pin(copied) == pin(raw) and stat.S_IMODE(info.st_mode) == row["mode"], "materialized bytes/mode differ")
    return {"path": destination.as_posix(), **pin(copied), "mode": row["mode"]}


def stage_closed_successor_setup(source, audit_path, guard_receipt, destination):
    source, reader, report, guard, archive, upstream = source_bundle(source, audit_path, guard_receipt)
    destination = canonical_path(destination, exists=False)
    need(source not in destination.parents and destination not in source.parents, "fresh seed destination overlaps closed source")
    omitted = set()
    excluded = []
    for name in EXCLUDED_EXPORT_NAMES:
        export = reader.json(name)
        wanted = export["artifact_cid"]
        numerical = "prefix-page" in name
        relative = "private/source-artifacts/" + ("source" if numerical else "structured") + "/" + wanted[:4] + "/" + wanted
        omitted.add(relative); excluded.append({"export": name, "path": relative, "cid": wanted})
    files = [row for row in archive["files"] if row["kind"] == "file" and allowed_member(row["path"]) and row["path"] not in omitted]
    symlinks = [row for row in archive["files"] if row["kind"] == "symlink" and allowed_member(row["path"])]
    need(not symlinks, "this fixed successor fixture must have no transported symlink aliases")
    need(len(files) + len(EVIDENCE_NAMES) <= MAX_COPIED_FILES
         and sum(row["bytes"] for row in files) + sum(next(row["bytes"] for row in archive["files"] if row["path"] == name) for name in EVIDENCE_NAMES) <= MAX_COPIED_BYTES,
         "bounded full successor materialization exceeded")
    destination.mkdir(parents=True, mode=0o700)
    copied = []
    try:
        for row in files:
            observed = copy_regular(reader, row, destination / row["path"])
            copied.append({**observed, "path": row["path"], "source_path": row["path"]})
        for name in EVIDENCE_NAMES:
            row = next(row for row in archive["files"] if row["path"] == name)
            observed = copy_regular(reader, row, destination / "seed-evidence" / name)
            copied.append({**observed, "path": "seed-evidence/" + name, "source_path": name})
        (destination / "private").chmod(0o700)
        for name in ("source.duckdb", "model.duckdb"):
            need((destination / "private" / name).stat().st_mode & 0o600 == 0o600, "copied database must retain owner read/write mode")
        for relative in omitted:
            need(not (destination / relative).exists(), "historical scan artifact was inherited")
        need(inert.same(inert.Reader(source, seconds=120).whole_archive(), archive), "closed primary changed during materialization")
        result = {"schema": SCHEMA, "qualified": True, "source_namespace": str(source), "destination": str(destination),
            "source_pins": upstream, "source_archive_inventory_cid": archive["inventory_cid"],
            "copied_members": sorted(copied, key=lambda row: row["path"]), "copied_files": len(copied),
            "copied_bytes": sum(row["bytes"] for row in copied), "excluded_historical_scan_objects": excluded,
            "selected_models": report["selected_model_identities"], "current_head": reader.json("source-delta.json")["value"]["current_head"],
            "checkpoint_states": reader.json("checkpoint-states-after.json"), "inherited_setup_epochs": 2,
            "inherited_scan_pages": 0, "new_fitting_epochs": 0, "new_scan_pages": 0,
            "fresh_native_receiving_required": True, "native_owners_opened": False, "proof_authority": False,
            "source_execution_attested": False, "scan_execution_attested": False}
        write_exclusive(destination / "materialized-successor-setup.json", result)
        return result
    except BaseException as error:
        write_exclusive(destination / "materialization-failure.json", {"schema": SCHEMA, "qualified": False,
            "error_type": type(error).__name__, "error": str(error), "copied_files": len(copied),
            "copied_members": copied, "new_fitting_epochs": 0, "native_owners_opened": False})
        raise


def materialization(output):
    output = canonical_path(output)
    raw = read_absolute(output / "materialized-successor-setup.json", 2 * inert.MIB)
    receipt = inert.parse(raw)
    need(receipt["schema"] == SCHEMA and receipt["qualified"] is True and receipt["destination"] == str(output)
         and receipt["source_namespace"] == SOURCE_NAMESPACE and receipt["new_fitting_epochs"] == receipt["new_scan_pages"] == 0
         and receipt["native_owners_opened"] is False, "materialized setup receipt differs")
    need((output / "private").resolve() == output / "private" and not (output / "private").is_symlink()
         and (output / "private").stat().st_mode & 0o077 == 0, "fresh owner-private directory required")
    return receipt, inert.sha(raw)


def open_materialized_successor(output):
    """Open owners only under a newly received materialization namespace."""
    output = canonical_path(output)
    materialization(output)
    relocate_materialized_source(output)
    relocate_materialized_registry(output)
    from . import qualify_codebase_inventory_resume as native
    return native._open(output)


def relocated_source_observation(before, old_root, new_root):
    """Only the copied structural catalog's local CAS path may change."""
    need(type(before) is dict and set(before) == {"ast", "catalog"}
         and type(before["catalog"]) is dict and set(before["catalog"]) == {"meta", "heads", "operations"},
         "complete copied source/structural catalog observation required")
    result = deepcopy(before); rows = result["catalog"]["meta"]
    need(type(rows) is list and len(rows) == 1 and len(rows[0]) == 5
         and rows[0][0] == 1 and type(rows[0][0]) is int and rows[0][4] == str(old_root),
         "copied source catalog old artifact path differs")
    rows[0][4] = str(new_root)
    return result


def relocate_materialized_source(output):
    """Guard all copied AST/control rows around one NEW-copy CAS path write."""
    output = canonical_path(output); setup, setup_sha = materialization(output)
    receipt_path = output / "copied-source-catalog-relocation.json"
    old_root = str(Path(SOURCE_NAMESPACE) / "private/source-artifacts")
    new_root = str(output / "private/source-artifacts")
    if receipt_path.exists():
        receipt = inert.parse(read_absolute(receipt_path, 2 * inert.MIB))
        need(receipt["schema"] == "source-successor-copied-source-catalog-relocation@1" and receipt["qualified"] is True
             and receipt["output"] == str(output) and receipt["materialization_receipt_sha256"] == setup_sha
             and receipt["old_artifact_root"] == old_root and receipt["new_artifact_root"] == new_root
             and receipt["native_source_publication_performed"] is False and receipt["new_fitting_epochs"] == 0,
             "recorded copied source relocation differs")
        return receipt
    member = next(row for row in setup["copied_members"] if row["path"] == "private/source.duckdb")
    database = output / member["path"]
    need(pin(read_absolute(database, 16 * inert.MIB)) == {key: member[key] for key in ("bytes", "sha256")},
         "copied source DB bytes changed before sole relocation")
    previous = inert.parse(read_absolute(output / "seed-evidence/previous-publication-receipt.json"))
    current = inert.parse(read_absolute(output / "seed-evidence/current-publication-receipt.json"))
    closed_source = inert.parse(read_absolute(output / "seed-evidence/owners-after-cold.json"))["source"]
    import duckdb
    from ipfs_datasets_py.duckdb_control import codebase_catalog as catalog
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_projection_replay import ASTS_CATALOG_TABLES
    connection = duckdb.connect(str(database), config={"threads": 1, "memory_limit": "128MB"})
    try:
        def observation():
            ast = {}
            for name in ASTS_CATALOG_TABLES:
                rows = native_json(connection.execute('SELECT * FROM "' + name + '" ORDER BY ALL').fetchall())
                ast[name] = {"rows": len(rows), "sha256": inert.sha(inert.numerical_wire(rows))}
            control = {name: native_json(connection.execute("SELECT * FROM codebase_control." + name + " ORDER BY ALL").fetchall())
                       for name in ("meta", "heads", "operations")}
            return {"ast": ast, "catalog": control}
        before = observation()
        definitions = native_json(connection.execute("SELECT table_name, sql FROM duckdb_tables() WHERE database_name=current_database() AND schema_name='codebase_control' ORDER BY table_name").fetchall())
        expected_meta = [[1, catalog.SCHEMA, inert.structured(list(catalog._DDL)), inert.structured(definitions), old_root]]
        head = inert.receipt_head(current)
        need(inert.same(head, setup["current_head"]) and inert.same(head, closed_source["current_head"]),
             "copied source publication/head seed differs")
        expected_heads = [[head[field] for field in ("repository_id", "generation", "manifest_cid", "snapshot_cid", "ast_revision_id", "receipt_cid")]]
        operations = [[receipt["operation_id"], receipt["request_cid"], inert.structured(receipt), inert.wire(receipt).decode()]
                      for receipt in (previous, current)]
        operations.sort()
        need(inert.same(before["ast"], closed_source["tables"])
             and inert.same(before["catalog"], {"meta": expected_meta, "heads": expected_heads, "operations": operations}),
             "complete copied AST/catalog/publication rows differ before relocation")
        wanted = relocated_source_observation(before, old_root, new_root)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("UPDATE codebase_control.meta SET artifact_root=? WHERE singleton=1 AND artifact_root=?", [new_root, old_root])
            after = observation()
            need(inert.same(after, wanted), "copied source relocation changed unrelated AST/catalog rows")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    receipt = {"schema": "source-successor-copied-source-catalog-relocation@1", "qualified": True,
        "output": str(output), "materialization_receipt_sha256": setup_sha, "old_artifact_root": old_root,
        "new_artifact_root": new_root, "source_before_sha256": inert.sha(inert.numerical_wire(before)),
        "source_after_sha256": inert.sha(inert.numerical_wire(after)), "ast_tables": before["ast"],
        "catalog_before": before["catalog"], "catalog_after": after["catalog"], "current_head": head,
        "all13_ast_table_rows_preserved": True, "all_catalog_rows_preserved_except_local_artifact_root": True,
        "native_source_publication_performed": False, "new_fitting_epochs": 0, "old_native_owners_opened": False,
        "proof_authority": False}
    write_exclusive(receipt_path, receipt)
    return receipt


def relocated_registry_observation(before, old_root, new_root):
    """Pure exact row expectation for one recorded copied-store relocation."""
    need(type(before) is dict and "meta" in before and type(before["meta"]) is list
         and len(before["meta"]) == 1 and len(before["meta"][0]) == 6,
         "complete copied registry metadata required")
    result = deepcopy(before); row = list(result["meta"][0])
    need(row[4] == str(old_root) and type(row[5]) is int and row[5] > 0,
         "copied registry old artifact path/generation differs")
    row[4] = str(new_root); result["meta"][0] = row
    return result


def closed_copied_registry(output):
    """The copied DB reflects final closure, including the native cold reopen."""
    return inert.parse(read_absolute(Path(output) / "seed-evidence/owners-after-cold.json"))["registry"]


def relocate_materialized_registry(output):
    """Relocate only the copied registry's durable local artifact-store path.

The old native DB is never connected. This recorded fresh-copy transaction
does not open AutoencoderRegistry, advance its owner generation, fit a model,
change model version identities, or mutate the frozen source/model records.
"""
    output = canonical_path(output)
    setup, setup_sha = materialization(output)
    receipt_path = output / "copied-registry-relocation.json"
    expected_old_root = str(Path(SOURCE_NAMESPACE) / "private/model-artifacts")
    expected_new_root = str(output / "private/model-artifacts")
    if receipt_path.exists():
        receipt = inert.parse(read_absolute(receipt_path, 2 * inert.MIB))
        need(receipt["schema"] == "source-successor-copied-registry-relocation@1" and receipt["qualified"] is True
             and receipt["output"] == str(output) and receipt["materialization_receipt_sha256"] == setup_sha
             and receipt["old_artifact_root"] == expected_old_root and receipt["new_artifact_root"] == expected_new_root
             and receipt["new_fitting_epochs"] == 0 and receipt["owner_generation_advanced"] is False,
             "recorded fresh registry relocation differs")
        return receipt
    row = next(row for row in setup["copied_members"] if row["path"] == "private/model.duckdb")
    database = output / row["path"]
    need(pin(read_absolute(database)) == {key: row[key] for key in ("bytes", "sha256")},
         "copied registry bytes changed before sole relocation")
    expected = closed_copied_registry(output)
    wanted = relocated_registry_observation(expected, expected_old_root, expected_new_root)
    import duckdb
    from . import qualify_codebase_inventory_scan as native
    connection = duckdb.connect(str(database), config={"threads": 1, "memory_limit": "128MB"})
    try:
        def observation():
            return {name: connection.execute("SELECT * FROM autoencoder_control." + name + " ORDER BY ALL").fetchall()
                    for name in native.TABLES}
        before = native_json(observation())
        need(inert.observation_same(before, expected), "complete copied registry rows differ before relocation")
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("UPDATE autoencoder_control.meta SET artifact_root=? WHERE singleton=1 AND artifact_root=? AND owner_generation=?",
                               [expected_new_root, expected_old_root, before["meta"][0][5]])
            after = native_json(observation())
            need(inert.observation_same(after, wanted), "copied registry relocation changed unrelated rows")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    receipt = {"schema": "source-successor-copied-registry-relocation@1", "qualified": True,
        "output": str(output), "materialization_receipt_sha256": setup_sha, "old_artifact_root": expected_old_root,
        "new_artifact_root": expected_new_root, "registry_before_sha256": inert.sha(inert.numerical_wire(before)),
        "registry_after_sha256": inert.sha(inert.numerical_wire(after)), "owner_generation_before": before["meta"][0][5],
        "owner_generation_after": after["meta"][0][5], "owner_generation_advanced": False,
        "version_rows_preserved": True, "other_registry_rows_preserved": True, "new_fitting_epochs": 0,
        "old_native_owners_opened": False, "model_artifacts_changed": False, "proof_authority": False}
    write_exclusive(receipt_path, receipt)
    return receipt


def close_materialized_successor(registry, connection):
    from . import qualify_codebase_inventory_resume as native
    native._close(registry, connection)


def checkpoint_states(registry, receipt):
    from . import qualify_codebase_inventory_resume as native
    result = {label: native._state(registry, model["version_id"], include_identity=True)
              for label, model in receipt["selected_models"].items()}
    need(inert.same(result, receipt["checkpoint_states"]), "copied checkpoint weights/Adam/report changed")
    return result


def owners(index, registry, connection):
    from . import qualify_codebase_source_successor as native
    return native_json(native._owners(index, registry, connection))


def owner_generation(observed):
    rows = observed["registry"]["meta"]
    need(type(rows) is list and len(rows) == 1 and len(rows[0]) == 6 and type(rows[0][5]) is int and rows[0][5] > 0,
         "exact current registry owner generation required")
    return rows[0][5]


def expected_after_reopens(observed, count):
    need(type(count) is int and 0 <= count <= 8, "bounded explicit owner reopen count required")
    result = deepcopy(observed); row = list(result["registry"]["meta"][0])
    row[5] = owner_generation(observed) + count
    result["registry"]["meta"][0] = row
    return result


def shared_scheduler(state_path, configuration):
    """Attach to the existing main pool; retain native pressure sampling."""
    state_path = canonical_path(state_path)
    need(configuration["proof_safety_enabled"] is True, "existing shared proof-host scheduler configuration required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig(state_path=state_path, auto_renew_leases=True, **configuration))
    need(scheduler.state_path == state_path and inert.observation_same(scheduler.config.persisted_dict(), configuration),
         "child scheduler detached from the main shared pool")
    with scheduler._locked_state(persist=False) as state:
        need(inert.observation_same(state["config"], configuration), "live shared scheduler configuration differs")
    return scheduler


def validate_chunk_request(request):
    need(type(request) is dict and set(request) == REQUEST_FIELDS and request["schema"] == REQUEST_SCHEMA,
         "closed fresh chunk request required")
    for field, minimum, maximum in (("run_number", 1, 4), ("max_pages", 1, 2), ("expected_registry_owner_generation", 1, 2**63 - 1)):
        need(type(request[field]) is int and minimum <= request[field] <= maximum, "exact bounded chunk integer required: " + field)
    need(type(request["timeout_seconds"]) in {int, float} and math.isfinite(request["timeout_seconds"])
         and 0 < request["timeout_seconds"] <= 1200, "bounded finite chunk deadline required")
    inert.check_cid(request["root_cid"]); inert.head_shape(request["expected_head"])
    need(request["expected_head"]["repository_id"] == REPOSITORY_ID, "fresh chunk source repository differs")
    cursor = request["cursor"]
    need(type(cursor) is dict and set(cursor) == {"schema", "root_cid", "next_offset", "previous_page_cid"}
         and cursor["schema"] == "codebase-inventory-resume-cursor@1" and cursor["root_cid"] == request["root_cid"]
         and type(cursor["next_offset"]) is int and 1 <= cursor["next_offset"] < 300,
         "complete root-bound resumable cursor required")
    inert.check_cid(cursor["previous_page_cid"], source=True)
    need(type(request["version_id"]) is str and request["version_id"].startswith("sha256:")
         and len(request["version_id"]) == 71 and all(c in "0123456789abcdef" for c in request["version_id"][7:]),
         "exact selected version identity required")
    for field in ("materialization_receipt_sha256", "helper_sha256"):
        need(type(request[field]) is str and len(request[field]) == 64 and all(c in "0123456789abcdef" for c in request[field]),
             "exact fresh process input pin required")
    need(type(request["checkpoint_states"]) is dict and set(request["checkpoint_states"]) == {"root", "child"},
         "complete frozen root/child checkpoints required")
    need(type(request["scheduler_configuration"]) is dict and request["scheduler_configuration"].get("proof_safety_enabled") is True,
         "native proof-host shared configuration required")
    canonical_path(request["output"]); canonical_path(request["scheduler_state_path"])
    return request


def assert_clean(scheduler, *, owner_pids=None):
    """Only named scan owners must drain; other host clients may be active."""
    pids = [os.getpid()] if owner_pids is None else list(owner_pids)
    need(pids and all(type(pid) is int and pid > 1 for pid in pids), "explicit scan process owners required")
    # Native snapshot recovers stale owners under the pool's stable lock.
    scheduler.snapshot()
    with scheduler._locked_state(persist=False) as state:
        active = sum(record.get("owner_pid") in pids for record in state["leases"].values())
        waiting = sum(record.get("owner_pid") in pids for record in state["waiters"].values())
        need(active == waiting == 0, "scan process reservations did not drain")
        return {"active_lease_count": active, "waiting_request_count": waiting,
            "owner_pids": sorted(set(pids)), "scope": "named_scan_process_owners_only",
            "global_active_lease_count": len(state["leases"]), "global_waiting_request_count": len(state["waiters"])}


def no_fit(report):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    def refuse(*args, **kwargs):
        report["post_setup_fit_attempt_count"] += 1
        raise SuccessorFullScanFixtureError("fresh scan attempted model fitting")
    stack = ExitStack()
    for module, name in ((training, "train_current_codebase_features"), (features, "train_projection_features"),
                         (runtimes.SourceBoundCodebaseFeatureRuntime, "train")):
        stack.enter_context(patch.object(module, name, refuse))
    return stack


def resume_chunk(request_path):
    request_path = canonical_path(request_path)
    request_raw = read_absolute(request_path, 2 * inert.MIB)
    request = validate_chunk_request(inert.parse(request_raw))
    output = Path(request["output"])
    need(request_path.parent == output and request_path.name == f"successor-resume-request-{request['run_number']:02d}.json",
         "numbered fresh request namespace differs")
    need(inert.sha(read_absolute(Path(__file__))) == request["helper_sha256"], "fresh process helper generation changed")
    setup, setup_sha = materialization(output)
    need(setup_sha == request["materialization_receipt_sha256"]
         and inert.same(setup["current_head"], request["expected_head"])
         and inert.same(setup["checkpoint_states"], request["checkpoint_states"])
         and setup["selected_models"]["child"]["version_id"] == request["version_id"], "fresh chunk full seed input binding differs")
    receipt_path = output / f"successor-fresh-process-run-{request['run_number']:02d}.json"
    need(not receipt_path.exists(), "fresh chunk receipt already exists")
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scanner
    report = {"schema": CHUNK_SCHEMA, "qualified": False, "complete": False, "pid": os.getpid(),
        "run_number": request["run_number"], "root_cid": request["root_cid"], "version_id": request["version_id"],
        "request_cursor": request["cursor"], "request_sha256": inert.sha(request_raw),
        "helper_sha256": request["helper_sha256"], "materialization_receipt_sha256": setup_sha,
        "scheduler_state_path": request["scheduler_state_path"], "scheduler_configuration": request["scheduler_configuration"],
        "max_pages": request["max_pages"], "timeout_seconds": request["timeout_seconds"],
        "post_setup_fit_attempt_count": 0, "pages_created": [], "source_execution_attested": False,
        "proof_authority": False, "scan_execution_attested": False, "new_fitting_epochs": 0,
        "fit_guard_scope": "three named owner-process training APIs plus exact frozen checkpoints and native owner rows"}
    started = time.monotonic(); deadline = started + request["timeout_seconds"]
    scheduler = index = registry = connection = None
    try:
        scheduler = shared_scheduler(request["scheduler_state_path"], request["scheduler_configuration"])
        index, registry, connection = open_materialized_successor(output)
        root = scanner.load_codebase_scan_resume_root(index.artifacts, request["root_cid"])
        need(inert.same(root.to_dict()["head"], request["expected_head"])
             and root.to_dict()["model"]["version_id"] == request["version_id"], "fresh received root source/model differs")
        cursor = scanner.CodebaseScanResumeCursor.from_dict(request["cursor"])
        before = owners(index, registry, connection)
        numerical = checkpoint_states(registry, setup)
        need(owner_generation(before) == request["expected_registry_owner_generation"], "fresh registry owner generation differs")
        report["registry_owner_generation_before"] = owner_generation(before)
        def options():
            remaining = deadline - time.monotonic()
            need(remaining > 0, "bounded fresh process deadline exceeded")
            return {"scheduler": scheduler, "timeout_seconds": min(600.0, remaining),
                    "admission_timeout_seconds": min(30.0, remaining), "memory_mb": 1024}
        with no_fit(report):
            while cursor is not None and len(report["pages_created"]) < request["max_pages"]:
                page = scanner.scan_current_codebase_page(index, output / "repository", root=root, registry=registry, cursor=cursor, **options())
                report["pages_created"].append(page.artifact_cid)
                write_exclusive(output / f"successor-process-{request['run_number']:02d}-page-{len(report['pages_created']):02d}.json",
                    {"artifact_cid": page.artifact_cid, "value": page.to_dict()})
                cursor = page.next_cursor
            need(len(report["pages_created"]) == request["max_pages"], "fresh process produced an incomplete bounded chunk")
            report["next_cursor"] = None if cursor is None else cursor.to_dict()
            report["prefix_tail_cid"] = page.artifact_cid
            if cursor is None:
                completion = scanner.complete_current_codebase_scan(index, output / "repository", root=root, registry=registry, tail_page_cid=page.artifact_cid, **options())
                need(scanner.validate_current_codebase_scan_completion(completion, index, output / "repository", root=root, registry=registry, **options()) is completion,
                     "fresh complete-scan receiving selected another record")
                write_exclusive(output / "successor-scan-completion.json", {"artifact_cid": completion.artifact_cid, "value": completion.to_dict()})
                report.update(complete=True, completion_cid=completion.artifact_cid, coverage=completion.to_dict()["coverage"])
        after = owners(index, registry, connection)
        need(inert.observation_same(before, after) and inert.same(checkpoint_states(registry, setup), numerical), "fresh scan changed source/model owner or weights/Adam")
        report.update(qualified=True, numerical_before=numerical, numerical_after=numerical,
            registry_owner_generation_after=owner_generation(after), source_model_owner_preservation=True,
            final_resources=assert_clean(scheduler))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            close_materialized_successor(registry, connection)
        except BaseException as cleanup_error:
            report.update(qualified=False, owner_cleanup_error=str(cleanup_error))
            raise
        finally:
            try:
                if scheduler is not None:
                    report["final_resources"] = assert_clean(scheduler)
            except BaseException as cleanup_error:
                report.update(qualified=False, resource_cleanup_error=str(cleanup_error))
                raise
            finally:
                report["recorded_seconds"] = time.monotonic() - started
                write_exclusive(receipt_path, report)
    return report


def launch_resume_chunk(output, *, root_cid, cursor, version_id, expected_head, checkpoint_states,
                        expected_registry_owner_generation, scheduler, run_number, timeout_seconds=900.0, max_pages=2):
    """One bounded child; caller closes native owners before invoking it."""
    output = canonical_path(output)
    setup, setup_sha = materialization(output)
    request = {"schema": REQUEST_SCHEMA, "output": str(output), "run_number": run_number,
        "root_cid": root_cid, "cursor": cursor, "version_id": version_id, "expected_head": expected_head,
        "checkpoint_states": checkpoint_states, "expected_registry_owner_generation": expected_registry_owner_generation,
        "scheduler_state_path": str(scheduler.state_path), "scheduler_configuration": scheduler.config.persisted_dict(),
        "timeout_seconds": timeout_seconds, "max_pages": max_pages, "materialization_receipt_sha256": setup_sha,
        "helper_sha256": inert.sha(read_absolute(Path(__file__)))}
    validate_chunk_request(request)
    need(inert.same(setup["current_head"], expected_head) and inert.same(setup["checkpoint_states"], checkpoint_states), "parent launch setup inputs differ")
    request_path = output / f"successor-resume-request-{run_number:02d}.json"
    write_exclusive(request_path, request)
    stdout_path = output / f"successor-fresh-process-{run_number:02d}.stdout"
    stderr_path = output / f"successor-fresh-process-{run_number:02d}.stderr"
    started = time.monotonic()
    child_cwd = canonical_path(Path(__file__).resolve().parents[3])
    returncode = None; child_pid = None; timeout_group_terminated = False
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            child = subprocess.Popen([sys.executable, "-B", "-m", MODULE, "resume", str(request_path)], stdout=stdout, stderr=stderr,
                start_new_session=True, cwd=child_cwd)
            child_pid = child.pid
            try:
                returncode = child.wait(timeout=timeout_seconds + 30)
            except subprocess.TimeoutExpired:
                timeout_group_terminated = True
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)
                returncode = child.returncode
                raise
        need(returncode == 0, "fresh scan child failed; preserve numbered receipt/stdout/stderr")
        result = inert.parse(read_absolute(output / f"successor-fresh-process-run-{run_number:02d}.json", 4 * inert.MIB))
        need(result["schema"] == CHUNK_SCHEMA and result["qualified"] is True and type(result["pid"]) is int
             and result["pid"] > 1 and result["pid"] != os.getpid() and result["run_number"] == run_number
             and result["pid"] == child_pid
             and result["request_cursor"] == cursor and result["root_cid"] == root_cid and result["version_id"] == version_id
             and result["request_sha256"] == inert.sha(read_absolute(request_path))
             and result["helper_sha256"] == request["helper_sha256"] and result["materialization_receipt_sha256"] == setup_sha
             and result["scheduler_state_path"] == str(scheduler.state_path)
             and inert.observation_same(result["scheduler_configuration"], scheduler.config.persisted_dict())
             and type(result["post_setup_fit_attempt_count"]) is int and result["post_setup_fit_attempt_count"] == 0
             and len(result["pages_created"]) == max_pages and len(set(result["pages_created"])) == max_pages
             and result["registry_owner_generation_before"] == result["registry_owner_generation_after"] == expected_registry_owner_generation
             and result["source_model_owner_preservation"] is True and inert.same(result["numerical_before"], checkpoint_states)
             and inert.same(result["numerical_after"], checkpoint_states), "fresh child receipt binding differs")
        need(all(type(result["final_resources"][field]) is int and result["final_resources"][field] == 0
                 for field in ("active_lease_count", "waiting_request_count"))
             and result["final_resources"]["owner_pids"] == [child_pid]
             and result["final_resources"]["scope"] == "named_scan_process_owners_only", "fresh child leaked shared reservations")
        assert_clean(scheduler, owner_pids=[os.getpid(), child_pid])
        return result
    finally:
        write_exclusive(output / f"successor-fresh-process-launch-{run_number:02d}.json", {"schema": "source-successor-fresh-scan-child-launch@1",
            "run_number": run_number, "parent_pid": os.getpid(), "child_pid": child_pid, "returncode": returncode,
            "timeout_group_terminated": timeout_group_terminated,
            "child_working_directory": str(child_cwd),
            "recorded_seconds": time.monotonic() - started, "request_sha256": inert.sha(read_absolute(request_path)),
            "helper_sha256": request["helper_sha256"], "scheduler_state_path": str(scheduler.state_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("resume"); child.add_argument("request", type=Path)
    args = parser.parse_args()
    result = resume_chunk(args.request)
    print(json.dumps({key: result[key] for key in ("qualified", "complete", "pid", "pages_created", "recorded_seconds")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
