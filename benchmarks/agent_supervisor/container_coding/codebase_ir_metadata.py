"""Bounded experimental metadata storage, with real native restart verification.

This creates its own DuckDB and isolated DuckLake namespaces. Original finite
JSON records retain every field, including candidate and proof labels; storage
integrity does not authenticate those labels or grant execution/proof authority.
No existing source/training catalog, service, or production namespace is opened.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


SCHEMA = "experimental-codebase-ir-metadata-manifest@2"
REPORT_SCHEMA = "experimental-codebase-ir-metadata-readback@2"
ROW_SCHEMA = "experimental-codebase-ir-metadata-row@1"
PACKET_SCHEMA = "experimental-codebase-ir-metadata-row-packet@1"
DEFAULT_FAMILIES = ("ast", "kg", "vectors", "contracts")
LIMITS = {"families": 32, "rows": 65536, "row_bytes": 262144,
          "total_payload_bytes": 64 * 1024**2, "source_bytes": 1048576,
          "manifest_bytes": 16 * 1024**2, "json_depth": 32,
          "json_nodes": 50000, "restart_seconds": 60,
          "history_bytes": 256 * 1024**2,
          "packet_rows": 100, "packet_bytes": 128 * 1024,
          "single_row_packet_bytes": 262144 + 1024}
LAKE_LAYOUT = {"schema": PACKET_SCHEMA, "table": "history.events",
               "encoding": "canonical-row-array-json-in-payload_json-string",
               "typed_vector_columns": False, "native_event_batch_rows": 10,
               "native_event_batch_bytes": 1048576}
QUALIFICATION = {"experimental": True, "production_activated": False,
                 "admitted": False, "proof_authority": False}
_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_TABLE_DDL = (
    "CREATE TABLE metadata_rows (family VARCHAR NOT NULL, row_id VARCHAR NOT NULL, "
    "payload_sha256 VARCHAR NOT NULL, source_snapshot_sha256 VARCHAR NOT NULL, "
    "row_ordinal BIGINT NOT NULL, payload_json VARCHAR NOT NULL, "
    "PRIMARY KEY(family,row_id), UNIQUE(family,row_ordinal))",
    "CREATE TABLE metadata_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
    "manifest_json VARCHAR NOT NULL)",
)


class MetadataError(ValueError):
    """Incomplete, changed, conflicting, or unbounded experiment metadata."""


def _require(condition, message):
    if not condition:
        raise MetadataError(message)


def _plain(value, depth=0, budget=None):
    if budget is None:
        budget = [LIMITS["json_nodes"]]
    budget[0] -= 1
    _require(depth <= LIMITS["json_depth"] and budget[0] >= 0, "JSON structural bound exceeded")
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return
    if kind is float:
        _require(math.isfinite(value), "nonfinite metadata value")
        return
    if kind is dict:
        _require(all(type(key) is str for key in value), "metadata object keys must be strings")
        for item in value.values():
            _plain(item, depth + 1, budget)
        return
    if kind is list:
        for item in value:
            _plain(item, depth + 1, budget)
        return
    raise MetadataError("metadata must contain ordinary finite JSON values")


def _json(value):
    _plain(value)
    return _wire(value)


def _wire(value):
    """Encode known constructed wrappers; original payloads are bounded separately."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise MetadataError("invalid canonical metadata JSON") from exc


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _row_array(values):
    # Bounds apply to each original record; aggregate membership can exceed the
    # per-record node bound without silently dropping any rows.
    return b"[" + b",".join(_wire(row) for row in values) + b"]"


def _row_root(rows):
    raw = b"{" + b",".join(_json(family) + b":" + _row_array(rows[family])
                             for family in sorted(rows)) + b"}"
    return _digest(raw)


def _loads(raw):
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(MetadataError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MetadataError("invalid metadata JSON") from exc
    return value


def _parse(raw):
    value = _loads(raw)
    _require(_json(value) == raw, "metadata JSON is not canonical")
    return value


def _path(value):
    _require(isinstance(value, Path) and value.is_absolute() and value.resolve() == value,
             "canonical absolute output path required")
    return value


def _read(path, maximum):
    _require(path.resolve(strict=True) == path, "artifact path is aliased")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
                 "bounded independent regular artifact required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    _require(len(raw) <= maximum and signature(before) == signature(after) == signature(path.stat()),
             "artifact changed during read")
    return raw


def _write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def _owner(output, *, create=False):
    path = output / "owner.lock"
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "invalid experiment owner lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise MetadataError("experiment metadata already has an owner") from exc
        identity = (info.st_dev, info.st_ino)
        _require(identity == (path.stat().st_dev, path.stat().st_ino), "owner lock changed")
        yield
        _require(identity == (path.stat().st_dev, path.stat().st_ino), "owner lock changed")
    finally:
        os.close(fd)


def _history():
    from ipfs_datasets_py.ducklake import autoencoder_history
    return autoencoder_history


def _connect(path, *, read_only=False):
    import duckdb
    return duckdb.connect(str(path), read_only=read_only,
                          config={"autoinstall_known_extensions": "false",
                                  "autoload_known_extensions": "false",
                                  "enable_external_access": "false", "threads": "1"})


def _catalog_digest(cx):
    tables = cx.execute("SELECT schema_name,table_name,sql FROM duckdb_tables() "
                        "WHERE NOT internal ORDER BY schema_name,table_name").fetchall()
    views = cx.execute("SELECT schema_name,view_name,sql FROM duckdb_views() "
                       "WHERE NOT internal ORDER BY schema_name,view_name").fetchall()
    return _digest(_json({"tables": [list(row) for row in tables],
                          "views": [list(row) for row in views]}))


def _row(family, payload, ordinal, source_digest):
    raw = _json(payload)
    _require(type(payload) is dict and len(raw) <= LIMITS["row_bytes"], "bounded object row required")
    identity = {"schema": ROW_SCHEMA, "family": family,
                "payload_sha256": _digest(raw), "source_snapshot_sha256": source_digest}
    return {**identity, "row_id": _digest(_json(identity)), "row_ordinal": ordinal,
            "payload": json.loads(raw)}


def _families(records, source_digest):
    _require(type(records) is dict and all(type(name) is str and _FAMILY.fullmatch(name) for name in records),
             "invalid metadata family")
    names = sorted(set(records) | set(DEFAULT_FAMILIES))
    _require(len(names) <= LIMITS["families"], "family bound exceeded")
    result, total, byte_count = {}, 0, 0
    for family in names:
        payloads = records.get(family, [])
        _require(type(payloads) is list, "family records must be a list")
        rows, identities, declared = [], set(), {}
        for ordinal, payload in enumerate(payloads):
            row = _row(family, payload, ordinal, source_digest)
            _require(row["row_id"] not in identities, "duplicate metadata row identity")
            identities.add(row["row_id"])
            if "row_id" in payload:
                key = _json(payload["row_id"])
                _require(key not in declared, "conflicting declared row identity")
                declared[key] = row["row_id"]
            rows.append(row)
            total += 1
            byte_count += len(_json(payload))
            _require(total <= LIMITS["rows"] and byte_count <= LIMITS["total_payload_bytes"],
                     "metadata population bound exceeded; no truncation permitted")
        result[family] = rows
    return result


def _packet(values):
    _require(bool(values) and len(values) <= LIMITS["packet_rows"], "invalid packet row count")
    raw = _row_array(values)
    maximum = LIMITS["single_row_packet_bytes"] if len(values) == 1 else LIMITS["packet_bytes"]
    _require(len(raw) <= maximum, "row packet byte bound exceeded")
    body = {"schema": PACKET_SCHEMA, "family": values[0]["family"],
            "source_snapshot_sha256": values[0]["source_snapshot_sha256"],
            "row_count": len(values), "ordered_row_ids": [value["row_id"] for value in values],
            "payload_sha256": _digest(raw), "payload_bytes": len(raw),
            "payload_json": raw.decode("utf-8")}
    return {**body, "packet_id": _digest(_json(body))}


def _packet_rows(packet):
    """Parse exact stored rows, checking complete identity and ordered membership."""
    fields = {"schema", "family", "source_snapshot_sha256", "row_count", "ordered_row_ids",
              "payload_sha256", "payload_bytes", "payload_json", "packet_id"}
    _require(type(packet) is dict and set(packet) == fields and packet["schema"] == PACKET_SCHEMA,
             "invalid row packet fields")
    body = {key: value for key, value in packet.items() if key != "packet_id"}
    _require(_digest(_json(body)) == packet["packet_id"], "row packet identity differs")
    _require(type(packet["family"]) is str and _FAMILY.fullmatch(packet["family"])
             and type(packet["row_count"]) is int and 1 <= packet["row_count"] <= LIMITS["packet_rows"]
             and type(packet["payload_json"]) is str and type(packet["payload_bytes"]) is int,
             "invalid row packet population")
    raw = packet["payload_json"].encode("utf-8")
    maximum = LIMITS["single_row_packet_bytes"] if packet["row_count"] == 1 else LIMITS["packet_bytes"]
    _require(len(raw) == packet["payload_bytes"] and len(raw) <= maximum
             and _digest(raw) == packet["payload_sha256"], "row packet payload differs")
    values = _loads(raw)
    _require(type(values) is list and len(values) == packet["row_count"], "partial row packet")
    rebuilt = []
    fields = {"schema", "family", "payload_sha256", "source_snapshot_sha256",
              "row_id", "row_ordinal", "payload"}
    for value in values:
        _require(type(value) is dict and set(value) == fields and value["schema"] == ROW_SCHEMA
                 and value["family"] == packet["family"]
                 and value["source_snapshot_sha256"] == packet["source_snapshot_sha256"]
                 and type(value["row_ordinal"]) is int and value["row_ordinal"] >= 0,
                 "invalid packet row binding")
        row = _row(packet["family"], value["payload"], value["row_ordinal"], packet["source_snapshot_sha256"])
        _require(_wire(row) == _wire(value), "packet row identity differs")
        rebuilt.append(row)
    _require(type(packet["ordered_row_ids"]) is list
             and packet["ordered_row_ids"] == [value["row_id"] for value in rebuilt]
             and len(set(packet["ordered_row_ids"])) == len(rebuilt), "packet ordered membership differs")
    _require(_row_array(rebuilt) == raw, "row packet JSON is not canonical")
    return rebuilt


def _packets(rows):
    packets = []
    for family in sorted(rows):
        pending = []
        for row in rows[family]:
            if pending and (len(pending) >= LIMITS["packet_rows"]
                            or len(_row_array(pending + [row])) > LIMITS["packet_bytes"]):
                packets.append(_packet(pending))
                pending = []
            pending.append(row)
        if pending:
            packets.append(_packet(pending))
    return packets


def _batches(rows, source, created_at):
    history = _history()
    batches, pending = [], []
    for packet in _packets(rows):
        event = {"event_id": packet["packet_id"], "kind": "codebase_ir_" + packet["family"],
                 "payload": packet, "created_at": created_at, "version": None, "variant": None}
        try:
            history.build_history_batch(source, pending + [event])
        except history.HistoryError:
            _require(bool(pending), "row packet cannot fit the DuckLake batch bounds")
            batches.append(history.build_history_batch(source, pending))
            pending = [event]
            try:
                history.build_history_batch(source, pending)
            except history.HistoryError as exc:
                raise MetadataError("row packet cannot fit the DuckLake batch bounds") from exc
        else:
            pending.append(event)
    if pending:
        batches.append(history.build_history_batch(source, pending))
    return batches


def _report(manifest, manifest_digest):
    return {"schema": REPORT_SCHEMA, "output": manifest["output"],
            "manifest_sha256": manifest_digest,
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "row_count": manifest["row_count"], "row_root_sha256": manifest["row_root_sha256"],
            "family_counts": {name: value["count"] for name, value in manifest["families"].items()},
            "family_digests": {name: value["digest"] for name, value in manifest["families"].items()},
            "family_views": {name: value["view"] for name, value in manifest["families"].items()},
            "exports": {name: value["export"] for name, value in manifest["families"].items()},
            "native_runtime": manifest["native_runtime"], "limits": manifest["limits"],
            "lake_layout": manifest["lake_layout"], "lake_packet_count": manifest["lake_packet_count"],
            "lake_snapshot_ids": [value["receipt"]["snapshot_id"] for value in manifest["batches"]],
            "lake_snapshot_digest": manifest["lake_snapshot_digest"],
            "lake_receipts": [value["receipt"] for value in manifest["batches"]],
            "qualification": dict(QUALIFICATION)}


def hydrate_codebase_ir_metadata(*, records: dict[str, list[dict]], output: Path,
                                 source_snapshot: dict) -> dict:
    """Create a new bounded experiment namespace; failures retain partial evidence.

    Every supplied row is preserved. Duplicate identities and over-bound inputs
    reject before creation. An existing namespace is verified, never overwritten;
    use ``validate_codebase_ir_metadata`` for reopen rather than rehydration.
    """
    output = _path(output)
    _require(not output.exists() and output.parent.is_dir(), "output must be new under an existing parent")
    _require(type(source_snapshot) is dict and bool(source_snapshot), "source snapshot object required")
    source_raw = _json(source_snapshot)
    _require(len(source_raw) <= LIMITS["source_bytes"], "source snapshot byte bound exceeded")
    source_digest = _digest(source_raw)
    rows = _families(records, source_digest)
    created_at = float(time.time())
    source = {"source_id": "codebase-ir-experiment:" + source_digest.removeprefix("sha256:"),
              "database_path": str(output / "metadata.duckdb"), "artifact_root": str(output / "exports")}
    batches = _batches(rows, source, created_at)
    history = _history()
    cx, runtime = history._native_connection()
    cx.close()
    output.mkdir(mode=0o700)
    with _owner(output, create=True):
        (output / "exports").mkdir(mode=0o700)
        (output / "batches").mkdir(mode=0o700)
        families = {}
        for family, values in rows.items():
            raw = b"".join(_wire(row) + b"\n" for row in values)
            relative = "exports/" + family + ".jsonl"
            _write(output / relative, raw)
            families[family] = {"count": len(values), "digest": _digest(_row_array(values)),
                                "view": "family_" + family,
                                "export": {"relative_path": relative, "sha256": _digest(raw), "bytes": len(raw)}}
        with _connect(output / "metadata.duckdb") as catalog:
            catalog.execute("BEGIN")
            try:
                for statement in _TABLE_DDL:
                    catalog.execute(statement)
                for family, values in rows.items():
                    catalog.execute(f"CREATE VIEW family_{family} AS SELECT * FROM metadata_rows WHERE family='{family}'")
                    if values:
                        catalog.executemany("INSERT INTO metadata_rows VALUES (?,?,?,?,?,?)", [
                            [family, row["row_id"], row["payload_sha256"], source_digest,
                             row["row_ordinal"], _json(row["payload"]).decode()] for row in values])
                catalog.execute("COMMIT")
            except BaseException:
                catalog.execute("ROLLBACK")
                raise
            catalog_digest = _catalog_digest(catalog)
        receipts = []
        with history.IsolatedNativeDuckLakeHistory(output / "lake", create=True,
                                                   max_history_bytes=LIMITS["history_bytes"]) as sink:
            for index, batch in enumerate(batches):
                raw = _json(batch)
                relative = f"batches/{index:06d}.json"
                _write(output / relative, raw)
                receipt = sink.append(batch)
                receipts.append({"relative_path": relative, "sha256": _digest(raw),
                                 "bytes": len(raw), "receipt": receipt})
        manifest = {"schema": SCHEMA, "output": str(output), "created_at": created_at,
                    "source_snapshot": json.loads(source_raw), "source_snapshot_sha256": source_digest,
                    "native_runtime": runtime, "limits": dict(LIMITS), "families": families,
                    "row_count": sum(map(len, rows.values())), "row_root_sha256": _row_root(rows),
                    "catalog_schema_sha256": catalog_digest, "batches": receipts,
                    "lake_layout": dict(LAKE_LAYOUT), "lake_packet_count": len(_packets(rows)),
                    "lake_snapshot_digest": _digest(_json([item["receipt"] for item in receipts])),
                    "qualification": dict(QUALIFICATION)}
        raw = _json(manifest)
        _require(len(raw) <= LIMITS["manifest_bytes"], "metadata manifest bound exceeded")
        with _connect(output / "metadata.duckdb") as catalog:
            catalog.execute("INSERT INTO metadata_identity VALUES (1,?)", [raw.decode()])
        _write(output / "manifest.json", raw)
    return validate_codebase_ir_metadata(output=output, fresh_process=True)


def _checked(output):
    raw = _read(output / "manifest.json", LIMITS["manifest_bytes"])
    manifest = _parse(raw)
    fields = {"schema", "output", "created_at", "source_snapshot", "source_snapshot_sha256",
              "native_runtime", "limits", "families", "row_count", "row_root_sha256",
              "catalog_schema_sha256", "batches", "lake_snapshot_digest", "qualification"}
    fields.update({"lake_layout", "lake_packet_count"})
    _require(type(manifest) is dict and set(manifest) == fields and manifest["schema"] == SCHEMA,
             "incomplete or foreign metadata manifest")
    _require(manifest["output"] == str(output) and manifest["limits"] == LIMITS
             and manifest["qualification"] == QUALIFICATION and manifest["lake_layout"] == LAKE_LAYOUT,
             "manifest namespace or contract differs")
    _require(type(manifest["source_snapshot"]) is dict and bool(manifest["source_snapshot"])
             and _digest(_json(manifest["source_snapshot"])) == manifest["source_snapshot_sha256"],
             "source snapshot identity differs")
    _require(type(manifest["created_at"]) is float and math.isfinite(manifest["created_at"]),
             "invalid creation timestamp")
    families = manifest["families"]
    _require(type(families) is dict and set(DEFAULT_FAMILIES).issubset(families)
             and len(families) <= LIMITS["families"]
             and all(_FAMILY.fullmatch(name) for name in families), "invalid family manifest")
    _require(type(manifest["row_count"]) is int and 0 <= manifest["row_count"] <= LIMITS["rows"],
             "invalid manifest row count")
    for family, value in families.items():
        _require(type(value) is dict and set(value) == {"count", "digest", "view", "export"}
                 and type(value["count"]) is int and 0 <= value["count"] <= LIMITS["rows"]
                 and value["view"] == "family_" + family, "invalid family descriptor")
        export = value["export"]
        _require(type(export) is dict and set(export) == {"relative_path", "sha256", "bytes"}
                 and export["relative_path"] == "exports/" + family + ".jsonl"
                 and type(export["bytes"]) is int and 0 <= export["bytes"] <= 2 * LIMITS["total_payload_bytes"],
                 "invalid family export descriptor")
    _require(sum(value["count"] for value in families.values()) == manifest["row_count"],
             "manifest family population differs")
    _read(output / "metadata.duckdb", 256 * 1024**2)
    rows = {family: [] for family in sorted(families)}
    with _connect(output / "metadata.duckdb", read_only=True) as catalog:
        _require(_catalog_digest(catalog) == manifest["catalog_schema_sha256"], "native catalog schema differs")
        _require(catalog.execute("SELECT singleton,manifest_json FROM metadata_identity LIMIT 2").fetchall()
                 == [(1, raw.decode())], "native manifest differs")
        found = catalog.execute("SELECT family,row_id,payload_sha256,source_snapshot_sha256,row_ordinal,payload_json "
                                "FROM metadata_rows ORDER BY family,row_ordinal LIMIT ?",
                                [manifest["row_count"] + 1]).fetchall()
        _require(len(found) == manifest["row_count"], "native row population differs")
        for family, row_id, digest, source_digest, ordinal, payload_json in found:
            _require(family in rows and type(ordinal) is int and ordinal == len(rows[family]),
                     "native family or ordinal differs")
            payload = _parse(payload_json.encode())
            row = _row(family, payload, ordinal, manifest["source_snapshot_sha256"])
            _require((row["row_id"], row["payload_sha256"], row["source_snapshot_sha256"])
                     == (row_id, digest, source_digest), "native row identity or payload differs")
            rows[family].append(row)
        for family in rows:
            _require(catalog.execute(f"SELECT count(*) FROM family_{family}").fetchone()[0]
                     == len(rows[family]), "native family view differs")
    # Reapply all population/declared-identity bounds, not just individual hashes.
    _require(_families({family: [row["payload"] for row in values] for family, values in rows.items()},
                       manifest["source_snapshot_sha256"]) == rows, "native row reconstruction differs")
    _require(_row_root(rows) == manifest["row_root_sha256"], "row root differs")
    for family, values in rows.items():
        value = families[family]
        export = value["export"]
        raw_export = _read(output / export["relative_path"], 2 * LIMITS["total_payload_bytes"])
        expected_export = b"".join(_wire(row) + b"\n" for row in values)
        _require(raw_export == expected_export and len(raw_export) == export["bytes"]
                 and _digest(raw_export) == export["sha256"] and len(values) == value["count"]
                 and _digest(_row_array(values)) == value["digest"], "family export or digest differs")
    history = _history()
    source = {"source_id": "codebase-ir-experiment:" + manifest["source_snapshot_sha256"].removeprefix("sha256:"),
              "database_path": str(output / "metadata.duckdb"), "artifact_root": str(output / "exports")}
    expected_batches = _batches(rows, source, manifest["created_at"])
    packets = _packets(rows)
    _require(type(manifest["lake_packet_count"]) is int and manifest["lake_packet_count"] == len(packets),
             "lake packet population differs")
    _require(type(manifest["batches"]) is list and len(manifest["batches"]) == len(expected_batches),
             "lake batch population differs")
    with history.IsolatedNativeDuckLakeHistory(output / "lake", max_history_bytes=LIMITS["history_bytes"]) as sink:
        _require(sink.identity["runtime"] == manifest["native_runtime"], "native runtime differs")
        for index, (descriptor, batch) in enumerate(zip(manifest["batches"], expected_batches)):
            _require(type(descriptor) is dict and set(descriptor) == {"relative_path", "sha256", "bytes", "receipt"}
                     and descriptor["relative_path"] == f"batches/{index:06d}.json"
                     and type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= history.MAX_BATCH_BYTES,
                     "invalid lake batch descriptor")
            batch_raw = _read(output / descriptor["relative_path"], history.MAX_BATCH_BYTES)
            _require(batch_raw == _json(batch) and len(batch_raw) == descriptor["bytes"]
                     and _digest(batch_raw) == descriptor["sha256"], "lake batch content differs")
            _require(sink.lookup(batch) == descriptor["receipt"], "native lake receipt differs")
        observed = sink._connection.execute("SELECT source_id,event_id,event_json FROM history.events "
                                            "ORDER BY event_id LIMIT ?", [len(packets) + 1]).fetchall()
        _require(len(observed) == len(packets), "native lake packet population differs")
        expected_packets = {packet["packet_id"]: packet for packet in packets}
        recovered = {family: [] for family in rows}
        for source_id, event_id, event_json in observed:
            event = _parse(event_json.encode("utf-8"))
            _require(source_id == source["source_id"] and event_id in expected_packets
                     and event["event_id"] == event_id and event["payload"] == expected_packets[event_id],
                     "native row packet binding differs")
            values = _packet_rows(event["payload"])
            recovered[event["payload"]["family"]].extend(values)
        for family in recovered:
            recovered[family].sort(key=lambda value: value["row_ordinal"])
            _require(_row_array(recovered[family]) == _row_array(rows[family]),
                     "native packet reconstruction differs from catalog/export")
        _require(sum(map(len, recovered.values())) == manifest["row_count"]
                 and sink._connection.execute("SELECT count(*) FROM history.commits").fetchone()[0]
                 == len(expected_batches), "native lake population differs")
    _require(_digest(_json([value["receipt"] for value in manifest["batches"]]))
             == manifest["lake_snapshot_digest"], "lake snapshot receipt digest differs")
    return _report(manifest, _digest(raw))


def _fresh(output, expected):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(Path(value or os.getcwd()).resolve()) for value in sys.path)
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--validate", str(output),
                                "--manifest-sha256", expected["manifest_sha256"]], env=env,
                               capture_output=True, text=True, timeout=LIMITS["restart_seconds"])
    _require(completed.returncode == 0, "fresh-process native metadata readback failed: " + completed.stderr[-2000:])
    try:
        observed = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise MetadataError("fresh-process readback did not return a report") from exc
    _require(observed == expected, "fresh-process readback report differs")
    return {"verified": True, "method": "new_python_process_native_duckdb_and_ducklake_readback",
            "manifest_sha256": observed["manifest_sha256"], "row_root_sha256": observed["row_root_sha256"],
            "row_count": observed["row_count"], "lake_snapshot_digest": observed["lake_snapshot_digest"]}


def validate_codebase_ir_metadata(*, output: Path, expected: dict | None = None,
                                  fresh_process: bool = False) -> dict:
    """Reopen both native stores and verify every record, export, and lake receipt.

    ``expected`` may be the hydration report or a manifest SHA binding. Historical
    metadata labels remain labels: successful storage validation is not a proof.
    """
    output = _path(output)
    _require(type(fresh_process) is bool, "fresh_process must be a boolean")
    try:
        with _owner(output):
            report = _checked(output)
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError("native metadata validation failed: " + str(exc)) from exc
    if expected is not None:
        _require(type(expected) is dict and "manifest_sha256" in expected,
                 "expected manifest identity required")
        for key, value in expected.items():
            if key != "fresh_process_readback":
                _require(key in report and report[key] == value, "expected metadata binding differs: " + key)
    if fresh_process:
        report["fresh_process_readback"] = _fresh(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    print(json.dumps(validate_codebase_ir_metadata(output=arguments.validate,
                     expected={"manifest_sha256": arguments.manifest_sha256}), sort_keys=True))
