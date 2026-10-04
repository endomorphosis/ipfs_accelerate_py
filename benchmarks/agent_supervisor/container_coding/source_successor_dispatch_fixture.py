"""Transport a closed, independently checked full scan to a fresh worker owner.

Staging/materialization are guarded byte copies with no native owner or Git.
Only the copied stores' local artifact paths are relocated before native open.
Weights, saved Adam, source heads, publications and complete scan CAS stay fixed.
The receiver must freshly check all source/model/scan bindings after reopening.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from . import audit_codebase_source_successor as inert
from . import source_successor_full_scan_fixture as transport

SCHEMA = "source-successor-dispatch-staged-setup@1"
MATERIALIZED_SCHEMA = "source-successor-dispatch-materialized-setup@1"
RELOCATION_SCHEMA = "source-successor-dispatch-copied-store-relocations@1"
REPOSITORY_ID = "qualification:source-successor"
MAX_FILES = 2048
MAX_BYTES = 128 * inert.MIB
STAGED_FIELDS = {"schema", "qualified", "source_namespace", "staged_destination", "source_archive_inventory_cid",
    "audit", "reader_controls", "reader", "native_result", "copied_members", "copied_files", "copied_bytes",
    "reader_control_scope", "reader_control_source_namespace",
    "selected_producers", "current_head", "previous_head", "root_cid", "completion_cid", "selection_cid",
    "source_delta_cid", "selected_version_id", "previous_version_id", "checkpoint_states", "inherited_setup_epochs",
    "inherited_scan_pages", "inherited_reference_pages", "new_fitting_epochs", "new_scan_pages", "native_owners_opened",
    "fresh_native_receiving_required", "proof_authority", "source_execution_attested", "scan_execution_attested"}


def need(value, message):
    if not value:
        raise ValueError(message)


def _pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(path, value):
    raw = inert.numerical_wire(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _read(path, maximum=16 * inert.MIB):
    path = transport.canonical_path(path)
    return inert.Reader(path.parent, seconds=120).raw(path.name, maximum)


def _native_json(value):
    # DuckDB returns tuple rows; use the established finite JSON observation
    # boundary so tuples become lists without admitting foreign objects.
    return transport.native_json(value)


def _relative(value):
    need(type(value) is str and value and len(value) <= 8192, "bounded exact relative path required")
    path = Path(value)
    need(not path.is_absolute() and all(part not in {"", ".", ".."} for part in value.split("/"))
         and path.as_posix() == value and "\\" not in value and "\0" not in value,
         "canonical contained relative path required")
    return value


def _absolute_declared(value):
    need(type(value) is str and 0 < len(value) <= 8192 and "\0" not in value, "exact declared namespace required")
    path = Path(value)
    need(path.is_absolute() and path.as_posix() == value and ".." not in path.parts,
         "canonical declared absolute namespace required")
    return path


def _pin_descriptor(value, maximum=4 * inert.MIB):
    need(type(value) is dict and set(value) == {"bytes", "sha256"}, "closed raw pin descriptor required")
    inert.integer(value["bytes"], maximum, 1)
    need(type(value["sha256"]) is str and len(value["sha256"]) == 64
         and all(character in "0123456789abcdef" for character in value["sha256"]), "exact raw SHA pin required")


def _allowed(path):
    if path.endswith((".lock", ".wal")):
        return False
    return path in {"private/source.duckdb", "private/model.duckdb"} or path.startswith(
        ("private/source-artifacts/", "private/model-artifacts/", "repository/"))


def _current_producers(reader, generation):
    rows = []
    for row in generation["files"]:
        name = row["name"]
        if name == "__main__":
            expected_path = Path(__file__).with_name("qualify_codebase_full_successor.py")
        else:
            need(type(name) is str and all(part.isidentifier() for part in name.split(".")),
                 "closed selected producer module name required")
            accelerate = Path(__file__).resolve().parents[3]
            if name.startswith("ipfs_datasets_py."):
                root = accelerate.parent / "ipfs_datasets"
            elif name.startswith(("ipfs_accelerate_py.", "benchmarks.agent_supervisor.container_coding.")):
                root = accelerate
            else:
                raise ValueError("foreign selected producer namespace")
            expected_path = root.joinpath(*name.split(".")).with_suffix(".py")
            if not expected_path.exists():
                expected_path = root.joinpath(*name.split("."), "__init__.py")
        need(str(expected_path.resolve(strict=True)) == row["path"],
             "selected producer checkout path differs: " + name)
        current = _read(row["path"], 4 * inert.MIB)
        frozen = reader.raw(row["copy"], 4 * inert.MIB)
        need(_pin(current) == _pin(frozen) == {key: row[key] for key in ("bytes", "sha256")},
             "current full-scan producer generation differs: " + row["name"])
        rows.append({key: row[key] for key in ("name", "path", "bytes", "sha256")})
    return rows


def stage_closed_successor_dispatch(namespace, destination, *, audit_path, guard_receipt):
    """Receive one complete frozen reader generation and copy independent bytes."""
    namespace = transport.canonical_path(namespace)
    destination = transport.canonical_path(destination, exists=False)
    need(namespace not in destination.parents and destination not in namespace.parents,
         "fresh worker seed overlaps closed full-scan archive")
    audit_raw, guard_raw = _read(audit_path, 4 * inert.MIB), _read(guard_receipt, 4 * inert.MIB)
    audit, controls = inert.parse(audit_raw), inert.parse(guard_raw)
    reader_path = Path(__file__).with_name("audit_codebase_full_successor.py").resolve()
    reader_pin = _pin(_read(reader_path))
    need(audit["schema"] == "codebase-full-successor-independent-audit@1"
         and audit["qualified"] is True and audit["preserved"] is True and audit["errors"] == []
         and audit["complete_scan_qualified"] is True and audit["namespace"] == str(namespace)
         and reader_pin == {key: audit["reader"][key] for key in ("bytes", "sha256")},
         "complete frozen full-scan reader/result generation required")
    need(controls["schema"] == "full-successor-independent-reader-guard-tests@1"
         and controls["qualified"] is True and controls["current_pins_unchanged"] is True
         and all(type(controls[field]) is int and controls[field] == 0 for field in ("failures", "errors", "skipped"))
         and all(controls[field] is False for field in ("native_owners_opened", "sql_executed", "git_executed"))
         and all(type(controls[field]) is int and controls[field] == 0 for field in ("new_fitting_epochs", "new_inference_pages"))
         and type(controls["tests"]) is int and controls["tests"] > 0
         and reader_pin == {key: controls["reader"][key] for key in ("bytes", "sha256")},
         "executed exact full-scan reader controls required")
    reader = inert.Reader(namespace, seconds=120)
    archive = reader.whole_archive()
    need(inert.same(archive, audit["archive"]), "closed full-scan archive changed since independent audit")
    result_raw = reader.raw("result.json", 4 * inert.MIB)
    result = inert.parse(result_raw)
    setup = reader.json("materialized-successor-setup.json", 4 * inert.MIB)
    need(controls["source_namespace"] == setup["source_namespace"] == transport.SOURCE_NAMESPACE,
         "reader controls must describe the inherited seed roles and synthetic custody checks")
    need(_pin(result_raw) == {key: audit["audited_result"][key] for key in ("bytes", "sha256")}
         and result["qualified"] is True and result["complete_scan_qualified"] is True,
         "exact qualified complete native result differs")
    for field, expected in (("new_fitting_epochs", 0), ("inherited_setup_epochs", 2),
                            ("inherited_scan_pages", 0), ("new_default_scan_pages", 10),
                            ("new_reference_scan_pages", 1), ("fresh_process_count", 4)):
        need(type(result[field]) is int and result[field] == expected, "full-scan accounting differs: " + field)
    need(result["current_head"]["repository_id"] == REPOSITORY_ID
         and type(result["coverage"]["inventory_entries"]) is int and result["coverage"]["inventory_entries"] == 300
         and type(result["coverage"]["pages"]) is int and result["coverage"]["pages"] == 10,
         "complete current source namespace/coverage differs")
    generation = reader.json("generation-inputs.json")
    producers = _current_producers(reader, generation)
    native_files = [row for row in archive["files"] if row["kind"] == "file" and _allowed(row["path"])]
    need(not any(row["kind"] == "symlink" and _allowed(row["path"]) for row in archive["files"]),
         "full-scan transport requires regular source/model bytes")
    evidence_names = [row["path"] for row in archive["files"] if row["kind"] == "file"
                      and not row["path"].startswith(("private/", "repository/"))
                      and row["path"] != "progress.json"]
    evidence_names.sort()
    selected = native_files + [next(row for row in archive["files"] if row["path"] == name) for name in evidence_names]
    need(len(selected) <= MAX_FILES and sum(row["bytes"] for row in selected) <= MAX_BYTES,
         "bounded complete worker setup exceeded")
    destination.mkdir(parents=True, mode=0o700)
    copied = []
    for row in selected:
        relative = row["path"] if _allowed(row["path"]) else "closed-full-scan/" + row["path"]
        observed = transport.copy_regular(reader, row, destination / relative)
        copied.append({**observed, "path": relative, "source_path": row["path"]})
    (destination / "private").chmod(0o700)
    for name, raw in (("independent-full-scan-audit.json", audit_raw),
                      ("independent-full-scan-reader-controls.json", guard_raw)):
        with (destination / name).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        (destination / name).chmod(0o444)
    need(inert.same(inert.Reader(namespace, seconds=120).whole_archive(), archive),
         "closed complete full-scan archive changed during staging")
    receipt = {"schema": SCHEMA, "qualified": True, "source_namespace": str(namespace),
        "staged_destination": str(destination), "source_archive_inventory_cid": archive["inventory_cid"],
        "audit": _pin(audit_raw), "reader_controls": _pin(guard_raw), "reader": reader_pin,
        "reader_control_scope": controls["scope"], "reader_control_source_namespace": controls["source_namespace"],
        "native_result": _pin(result_raw), "copied_members": sorted(copied, key=lambda row: row["path"]),
        "copied_files": len(copied), "copied_bytes": sum(row["bytes"] for row in copied),
        "selected_producers": producers, "current_head": result["current_head"],
        "previous_head": result["previous_head"], "root_cid": result["root_cid"],
        "completion_cid": result["completion_cid"], "selection_cid": result["successor_selection_cid"],
        "source_delta_cid": result["source_delta_cid"], "selected_version_id": result["child_version_id"],
        "previous_version_id": result["parent_version_id"], "checkpoint_states": result["checkpoint_states"],
        "inherited_setup_epochs": 2, "inherited_scan_pages": 10, "inherited_reference_pages": 1,
        "new_fitting_epochs": 0, "new_scan_pages": 0, "native_owners_opened": False,
        "fresh_native_receiving_required": True, "proof_authority": False,
        "source_execution_attested": False, "scan_execution_attested": False}
    _write(destination / "staged-successor-dispatch.json", receipt)
    return receipt


def _stage_receipt(value):
    need(type(value) is dict and set(value) == STAGED_FIELDS and value["schema"] == SCHEMA and value["qualified"] is True,
         "qualified staged successor dispatch receipt required")
    inert.head_shape(value["current_head"]); inert.head_shape(value["previous_head"])
    need(value["current_head"]["repository_id"] == value["previous_head"]["repository_id"] == REPOSITORY_ID
         and value["current_head"]["generation"] == value["previous_head"]["generation"] + 1,
         "exact adjacent successor source heads required")
    for field in ("source_namespace", "staged_destination"):
        _absolute_declared(value[field])
    inert.check_cid(value["source_archive_inventory_cid"])
    for field in ("audit", "reader_controls", "reader", "native_result"):
        _pin_descriptor(value[field])
    inert.text(value["reader_control_scope"], 8192)
    need(value["reader_control_source_namespace"] == transport.SOURCE_NAMESPACE,
         "exact inherited source reader-control role required")
    for field in ("root_cid", "completion_cid", "selection_cid", "source_delta_cid"):
        inert.check_cid(value[field])
    for field, expected in (("inherited_setup_epochs", 2), ("inherited_scan_pages", 10),
                            ("inherited_reference_pages", 1), ("new_fitting_epochs", 0), ("new_scan_pages", 0)):
        need(type(value[field]) is int and value[field] == expected, "staged dispatch accounting differs")
    need(value["fresh_native_receiving_required"] is True
         and all(value[field] is False for field in ("native_owners_opened", "proof_authority",
                                                     "source_execution_attested", "scan_execution_attested")),
         "staged dispatch authority differs")
    members = value["copied_members"]
    need(type(members) is list and 0 < len(members) <= MAX_FILES
         and [row["path"] for row in members] == sorted({row["path"] for row in members})
         and type(value["copied_files"]) is int and value["copied_files"] == len(members),
         "complete exact staged member inventory required")
    for row in members:
        need(type(row) is dict and set(row) == {"path", "source_path", "bytes", "sha256", "mode"},
             "closed staged member descriptor required")
        _relative(row["path"])
        _relative(row["source_path"])
        need(_allowed(row["path"]) or row["path"].startswith("closed-full-scan/"), "foreign staged member")
        need(row["source_path"] == (row["path"] if _allowed(row["path"]) else row["path"].removeprefix("closed-full-scan/")),
             "staged source/target member mapping differs")
        inert.integer(row["bytes"], 16 * inert.MIB)
        inert.integer(row["mode"], 0o777)
        need(type(row["sha256"]) is str and len(row["sha256"]) == 64
             and all(letter in "0123456789abcdef" for letter in row["sha256"]), "exact staged member SHA required")
    need(type(value["copied_bytes"]) is int and value["copied_bytes"] == sum(row["bytes"] for row in members) <= MAX_BYTES,
         "staged member byte accounting differs")
    need({"private/source.duckdb", "private/model.duckdb", "closed-full-scan/owners-after-cold.json",
          "closed-full-scan/copied-source-catalog-relocation.json", "closed-full-scan/successor-scan-completion.json",
          "closed-full-scan/successor-selection.json", "closed-full-scan/scan-root.json"}
         <= {row["path"] for row in members}, "complete source/model/scan owner material is missing")
    return value


def materialize_staged_successor_dispatch(seed, output, receipt):
    """Recheck every staged byte before creating an independent native copy."""
    receipt = _stage_receipt(receipt)
    seed = transport.canonical_path(seed)
    need(receipt["staged_destination"] == str(seed), "staged destination differs from exact received seed")
    output = transport.canonical_path(output, exists=False)
    source = Path(receipt["source_namespace"])
    need(seed not in output.parents and output not in seed.parents and source not in output.parents
         and output not in source.parents and output != seed and output != source,
         "fresh native output overlaps a closed source or staged seed")
    staged_raw = _read(seed / "staged-successor-dispatch.json", 4 * inert.MIB)
    need(inert.observation_same(inert.parse(staged_raw), receipt), "staged manifest bytes differ")
    reader = inert.Reader(seed, seconds=120)
    before = reader.whole_archive()
    expected_directories = {"."}
    allowed_files = {row["path"] for row in receipt["copied_members"]} | {
        "staged-successor-dispatch.json", "independent-full-scan-audit.json", "independent-full-scan-reader-controls.json"}
    for name in allowed_files:
        expected_directories.update(parent.as_posix() for parent in Path(name).parents)
    need({row["path"] for row in before["files"] if row["kind"] == "file"} == allowed_files
         and {row["path"] for row in before["files"] if row["kind"] == "directory"} == expected_directories
         and all(row["kind"] in {"file", "directory"} for row in before["files"]),
         "staged full membership contains aliases or foreign files")
    for name, wanted in (("independent-full-scan-audit.json", receipt["audit"]),
                         ("independent-full-scan-reader-controls.json", receipt["reader_controls"])):
        need(_pin(reader.raw(name, 4 * inert.MIB)) == wanted, "staged independent receipt pin differs")
    # Validate the entire staged input before making a new destination.
    for row in receipt["copied_members"]:
        observed = next(item for item in before["files"] if item["path"] == row["path"])
        need({key: observed[key] for key in ("bytes", "sha256", "mode")} == {
            key: row[key] for key in ("bytes", "sha256", "mode")}, "staged bytes/modes differ")
    output.mkdir(mode=0o755)
    for row in receipt["copied_members"]:
        observed = next(item for item in before["files"] if item["path"] == row["path"])
        transport.copy_regular(reader, observed, output / row["path"])
    (output / "private").chmod(0o700)
    need(inert.same(inert.Reader(seed, seconds=120).whole_archive(), before), "staged seed changed during materialization")
    result = {"schema": MATERIALIZED_SCHEMA, "qualified": True, "output": str(output),
        "seed": str(seed), "staged_receipt_sha256": inert.sha(staged_raw), "staged_receipt": receipt,
        "new_fitting_epochs": 0, "new_scan_pages": 0, "native_owners_opened": False,
        "proof_authority": False, "fresh_native_receiving_required": True}
    _write(output / "materialized-successor-dispatch.json", result)
    return result


def _observations(connection, registry=False):
    if registry:
        from .qualify_codebase_inventory_scan import TABLES
        return {name: _native_json(connection.execute("SELECT * FROM autoencoder_control." + name + " ORDER BY ALL").fetchall())
                for name in TABLES}
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_projection_replay import ASTS_CATALOG_TABLES
    ast = {}
    for name in ASTS_CATALOG_TABLES:
        rows = _native_json(connection.execute('SELECT * FROM "' + name + '" ORDER BY ALL').fetchall())
        ast[name] = {"rows": len(rows), "sha256": inert.sha(inert.numerical_wire(rows))}
    return {"ast": ast, "catalog": {name: _native_json(connection.execute(
        "SELECT * FROM codebase_control." + name + " ORDER BY ALL").fetchall()) for name in ("meta", "heads", "operations")}}


def _relocate(output, receipt, *, registry):
    old = Path(receipt["source_namespace"]) / "private" / ("model-artifacts" if registry else "source-artifacts")
    new = output / "private" / old.name
    database = output / "private" / ("model.duckdb" if registry else "source.duckdb")
    member = next(row for row in receipt["copied_members"] if row["path"] == database.relative_to(output).as_posix())
    need(_pin(_read(database)) == {key: member[key] for key in ("bytes", "sha256")},
         "copied dispatch DB changed before its sole local relocation")
    owners = inert.parse(_read(output / "closed-full-scan/owners-after-cold.json"))
    if registry:
        expected = owners["registry"]
        wanted = transport.relocated_registry_observation(expected, old, new)
        table = "autoencoder_control.meta"
    else:
        original = inert.parse(_read(output / "closed-full-scan/copied-source-catalog-relocation.json"))
        expected = {"ast": owners["source"]["tables"], "catalog": original["catalog_after"]}
        wanted = transport.relocated_source_observation(expected, old, new)
        table = "codebase_control.meta"
    import duckdb
    connection = duckdb.connect(str(database), config={"threads": 1, "memory_limit": "128MB"})
    try:
        before = _observations(connection, registry)
        need(inert.observation_same(before, expected), "complete copied dispatch owner rows differ before relocation")
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("UPDATE " + table + " SET artifact_root=? WHERE singleton=1 AND artifact_root=?", [str(new), str(old)])
            after = _observations(connection, registry)
            need(inert.observation_same(after, wanted), "dispatch relocation changed unrelated owner rows")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return {"owner": "registry" if registry else "source", "old_artifact_root": str(old), "new_artifact_root": str(new),
        "before_sha256": inert.sha(inert.numerical_wire(before)), "after_sha256": inert.sha(inert.numerical_wire(after)),
        "before": before, "after": after, "only_local_path_changed": True, "old_native_owners_opened": False,
        "native_publication_performed": False, "fitting_performed": False, "owner_generation_advanced": False}


def open_materialized_successor_dispatch(output):
    output = transport.canonical_path(output)
    materialization_raw = _read(output / "materialized-successor-dispatch.json", 4 * inert.MIB)
    materialization = inert.parse(materialization_raw)
    need(type(materialization) is dict and set(materialization) == {"schema", "qualified", "output", "seed",
         "staged_receipt_sha256", "staged_receipt", "new_fitting_epochs", "new_scan_pages", "native_owners_opened",
         "proof_authority", "fresh_native_receiving_required"}
         and materialization["schema"] == MATERIALIZED_SCHEMA and materialization["qualified"] is True
         and materialization["output"] == str(output) and materialization["native_owners_opened"] is False
         and materialization["proof_authority"] is False and materialization["fresh_native_receiving_required"] is True
         and all(type(materialization[field]) is int and materialization[field] == 0
                 for field in ("new_fitting_epochs", "new_scan_pages")),
         "exact materialized successor dispatch owner required")
    receipt = _stage_receipt(materialization["staged_receipt"])
    _absolute_declared(materialization["seed"])
    need(materialization["seed"] == receipt["staged_destination"]
         and materialization["staged_receipt_sha256"] == inert.sha(inert.numerical_wire(receipt) + b"\n"),
         "materialized staged receipt bytes or seed binding differ")
    path = output / "copied-successor-dispatch-store-relocations.json"
    if not path.exists():
        relocations = [_relocate(output, receipt, registry=False), _relocate(output, receipt, registry=True)]
        _write(path, {"schema": RELOCATION_SCHEMA, "qualified": True, "output": str(output),
            "materialization_sha256": inert.sha(materialization_raw), "relocations": relocations,
            "new_fitting_epochs": 0, "new_scan_pages": 0, "old_native_owners_opened": False,
            "native_owner_generation_advanced": False, "proof_authority": False})
    else:
        relocation = inert.parse(_read(path, 4 * inert.MIB))
        need(relocation["schema"] == RELOCATION_SCHEMA and relocation["qualified"] is True
             and relocation["output"] == str(output) and relocation["materialization_sha256"] == inert.sha(materialization_raw),
             "recorded copied dispatch store relocation differs")
    from .qualify_codebase_inventory_resume import _open
    return _open(output)


__all__ = ["stage_closed_successor_dispatch", "materialize_staged_successor_dispatch", "open_materialized_successor_dispatch"]
