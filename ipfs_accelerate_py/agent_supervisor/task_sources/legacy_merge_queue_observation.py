"""Authenticated, bounded observations of the separate Portal merge queue.

Only the launcher binds a queue path. The socket reader cannot nominate paths,
claim work, or retain the writer guard. Effects need fresh independent admission.
"""
from __future__ import annotations

import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Settlement receipts retain their existing SHA-256 JSON identity. Legacy
# DOUBLEs are losslessly encoded as float.hex strings before crossing the
# owner transport, whose control-plane canonical framing forbids floats.
from ..merge.merge_queue import (
    _settlement_canonical_bytes as canonical_json_bytes,
    _settlement_content_id as content_identity,
)

OPERATION = "legacy.merge_queue.task.observe"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/legacy-merge-queue-observation@1"
MAX_POPULATION = 8192
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 12 * 1024 * 1024
MAX_MATCHES = 64
DOUBLE_ENCODING = "ieee754-float64-hex@1"


def _require(condition: bool, message: str) -> None:
    from .typed_state_owner import TypedStateOwnerProtocolError
    if not condition:
        raise TypedStateOwnerProtocolError(message)


def _text(value: Any) -> bool:
    return (type(value) is str and bool(value) and value == value.strip()
            and "\x00" not in value and len(value.encode()) <= 1024)


def queue_identities(directory: Path) -> dict[str, Any]:
    """Capture existing no-follow identities; never create a queue or lock."""
    result = {}
    for name in ("", ".merge_queue.duckdb.lock", "merge_queue.duckdb"):
        details = (directory / name).lstat()
        _require((stat.S_ISDIR(details.st_mode) if not name else stat.S_ISREG(details.st_mode)),
                 "queue binding requires real directory and regular store/lock")
        result[name or "directory"] = {"device": details.st_dev, "inode": details.st_ino}
    return result


def queue_binding(*, queue_dir: Path, target_repository_id: str,
                  target_branch: str, database_scope_cid: str) -> dict[str, Any]:
    _require(isinstance(queue_dir, Path) and queue_dir.is_absolute()
             and queue_dir == queue_dir.resolve(strict=True),
             "queue binding requires a canonical absolute path")
    _require(all(_text(v) for v in (target_repository_id, target_branch, database_scope_cid)),
             "queue binding target or database scope is invalid")
    return {"queue_dir": str(queue_dir), "target_repository_id": target_repository_id,
            "target_branch": target_branch, "database_scope_cid": database_scope_cid,
            "identities": queue_identities(queue_dir)}


def observation_request(owner_identity: Mapping[str, Any], task_cids: Any,
                        *, task_cid: str) -> dict[str, Any]:
    from .typed_state_owner import completion_progress_request
    completion = dict(completion_progress_request(owner_identity, task_cids))
    material = {"schema": SCHEMA + "/request", "task_cid": task_cid,
                "completion_request": completion}
    return validate_request({**material, "request_cid": content_identity(material)})


def validate_request(value: Any) -> dict[str, Any]:
    from .typed_state_owner import _validated_completion_progress_request
    _require(isinstance(value, Mapping) and set(value) == {
        "schema", "task_cid", "completion_request", "request_cid"},
        "queue observation request shape differs")
    material = {k: v for k, v in value.items() if k != "request_cid"}
    _require(value["schema"] == SCHEMA + "/request"
             and content_identity(material) == value["request_cid"],
             "queue observation request seal differs")
    completion = _validated_completion_progress_request(value["completion_request"])
    _require(_text(value["task_cid"]) and value["task_cid"] in completion["task_cids"],
             "queue observation task is outside the requested population")
    return dict(value)


def _bindings(row: Mapping[str, Any]) -> dict[str, str]:
    from ..merge.merge_queue import _strict_settlement_metadata
    metadata = _strict_settlement_metadata(row["metadata_json"])
    bindings = metadata.get("completion_task_cids")
    _require(isinstance(bindings, dict) and bool(bindings)
             and all(_text(k) and _text(v) for k, v in bindings.items())
             and _text(row["task_id"]) and _text(row["canonical_task_id"])
             and bindings.get(row["task_id"]) == row["canonical_task_id"],
             "queue population has missing or conflicting task bindings")
    return bindings


def _matches(row: Mapping[str, Any], alias: str, cid: str) -> bool:
    bindings = row["completion_task_cids"]
    return (row["task_id"] == alias or row["canonical_task_id"] == cid
            or row["canonical_task_key"] in (alias, cid)
            or alias in bindings or cid in bindings.values())


def _read_population(settlement: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Read all rows under the caller's existing writer guard, without setup."""
    import duckdb
    from ..merge.merge_queue import (
        _MERGE_QUEUE_SETTLEMENT_COLUMNS, connect_duckdb_with_policy,
        _settlement_regular_file_identity,
    )
    _require(type(settlement["row_count"]) is int
             and 0 <= settlement["row_count"] <= MAX_POPULATION,
             "queue observation population exceeds bound")
    path = Path(settlement["database"]["path"])
    before = _settlement_regular_file_identity(path, label="database")
    _require(before == {k: settlement["database"][k] for k in before},
             "queue database identity changed before population read")
    columns = _MERGE_QUEUE_SETTLEMENT_COLUMNS["merge_requests"]
    text_columns = [name for name, kind, _ in columns if kind == "VARCHAR"]
    lengths = " + ".join(f"COALESCE(octet_length(encode({name})), 0)" for name in text_columns)
    connection = connect_duckdb_with_policy(duckdb, path, read_only=True)
    try:
        connection.execute("BEGIN TRANSACTION")
        summary = connection.execute(
            f"SELECT count(*), COALESCE(sum({lengths}), 0) FROM merge_requests"
        ).fetchone()
        _require(summary is not None and summary[0] == settlement["row_count"]
                 and 0 <= summary[1] <= MAX_INPUT_BYTES,
                 "queue observation population bytes exceed bound or population changed")
        # Fixed in-tree column identifiers only. Input bytes are bounded before
        # materialization; no large unbounded metadata crosses the owner socket.
        selections, oversized = [], []
        for name, kind, _ in columns:
            if kind == "VARCHAR":
                limit = 65536 if name == "metadata_json" else 4096
                oversized.append(f"COALESCE(octet_length(encode({name})), 0) > {limit}")
                selections.append(f"CASE WHEN octet_length(encode({name})) <= {limit} THEN {name} END")
            else:
                selections.append(name)
        oversize_count = connection.execute(
            f"SELECT count(*) FROM merge_requests WHERE {' OR '.join(oversized)}"
        ).fetchone()
        _require(oversize_count is not None and oversize_count[0] == 0,
                 "queue field exceeds bound, including nullable fields")
        raw = connection.execute(
            f"SELECT {', '.join(selections)} FROM merge_requests ORDER BY request_id LIMIT ?",
            [MAX_POPULATION + 1],
        ).fetchall()
        _require(len(raw) == settlement["row_count"], "queue observation population changed")
        rows, inventory = [], []
        for values in raw:
            row = dict(zip((c[0] for c in columns), values))
            for name, kind, nullable in columns:
                value = row[name]
                _require(value is not None or nullable == "YES", "queue row is missing or exceeds field bound")
                if value is not None and kind == "VARCHAR":
                    _require(type(value) is str, "queue text field differs")
                elif value is not None:
                    _require((type(value) is int if kind in ("INTEGER", "BIGINT")
                              else type(value) is float)
                             and math.isfinite(value) and value >= 0,
                             "queue numeric authority field is invalid")
            _require(_text(row["request_id"]), "queue request identity is invalid")
            bindings = _bindings(row)
            for name, kind, _ in columns:
                if kind == "DOUBLE":
                    row[name] = float(row[name]).hex()
            inventory.append({k: row[k] for k in (
                "request_id", "task_id", "canonical_task_id", "canonical_task_key")}
                | {"completion_task_cids": bindings, "row_cid": content_identity(row)})
            rows.append(row)
        connection.execute("COMMIT")
    finally:
        connection.close()
    _require(_settlement_regular_file_identity(path, label="database") == before,
             "queue database changed during population read")
    return rows, inventory


def capture_observation(binding: Mapping[str, Any], request: Mapping[str, Any],
                        *, capture_control: Any) -> dict[str, Any]:
    """Queue lock precedes owner lock, matching existing queue-to-task guards."""
    from ..merge.merge_queue import hold_merge_queue_settlement
    request = validate_request(request)
    directory = Path(binding["queue_dir"])
    _require(queue_identities(directory) == binding["identities"], "bound queue identity changed")
    with hold_merge_queue_settlement(
        directory, target_repository_id=binding["target_repository_id"],
        target_branch=binding["target_branch"], max_active_ids=1024,
    ) as settlement:
        _require(queue_identities(directory) == binding["identities"], "bound queue identity changed")
        rows, inventory = _read_population(settlement)
        # The queue is constant throughout this owner-side MVCC snapshot. No
        # task lock is held while waiting to acquire the queue writer guard.
        control = capture_control(request["completion_request"])
        facts = control["closeout_facts"]
        task_rows = facts["relations"]["tasks"]
        _require(task_rows["available"] and not task_rows["truncated"],
                 "owner task population is unavailable or truncated")
        aliases = {row["task_cid"]: row["task_alias"] for row in task_rows["rows"]}
        _require(len(aliases) == len(task_rows["rows"])
                 and sorted(aliases) == request["completion_request"]["task_cids"]
                 and all(_text(a) for a in aliases.values())
                 and len(set(aliases.values())) == len(aliases),
                 "owner task alias population differs")
        alias = aliases[request["task_cid"]]
        matches = [row for row, item in zip(rows, inventory)
                   if _matches(item, alias, request["task_cid"])]
        _require(len(matches) <= MAX_MATCHES, "matching queue population exceeds bound")
        material = {"schema": SCHEMA, "request_cid": request["request_cid"],
                    "task_cid": request["task_cid"], "task_alias": alias,
                    "queue_binding": dict(binding),
                    "queue_settlement": dict(settlement) | {
                        "max_updated_at": float(settlement["max_updated_at"]).hex()},
                    "legacy_double_encoding": DOUBLE_ENCODING,
                    "control_snapshot": dict(control), "population": inventory,
                    "population_cid": content_identity(inventory), "matching_rows": matches,
                    "population_complete": True, "guard_retained": False,
                    "claim_authority": False, "retry_authorized": False,
                    "completion_authority": False}
        result = {**material, "observation_cid": content_identity(material)}
        _require(queue_identities(directory) == binding["identities"], "bound queue identity changed")
    # Return only after the guard's final database/WAL identity checks succeed.
    return dict(validate_observation(result, request=request))


def decode_observed_queue_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore finite legacy DOUBLEs, preserving exact NULL and integer fields."""
    from ..merge.merge_queue import _MERGE_QUEUE_SETTLEMENT_COLUMNS
    columns = _MERGE_QUEUE_SETTLEMENT_COLUMNS["merge_requests"]
    _require(isinstance(value, Mapping) and set(value) == {c[0] for c in columns},
             "matching queue row shape differs")
    decoded = dict(value)
    for name, kind, nullable in columns:
        observed = value[name]
        if observed is None:
            _require(nullable == "YES", "matching queue row has unexpected NULL")
        elif kind in ("INTEGER", "BIGINT"):
            _require(type(observed) is int and observed >= 0,
                     "matching queue integer authority field differs")
        elif kind == "DOUBLE":
            _require(type(observed) is str, "matching queue DOUBLE encoding differs")
            try:
                decoded[name] = float.fromhex(observed)
            except ValueError:
                _require(False, "matching queue DOUBLE encoding differs")
            _require(math.isfinite(decoded[name]) and decoded[name] >= 0
                     and decoded[name].hex() == observed, "matching queue DOUBLE encoding differs")
        else:
            _require(type(observed) is str and len(observed.encode()) <= (
                65536 if name == "metadata_json" else 4096), "matching queue text field differs")
    _bindings(decoded)
    return decoded


def validate_observation(value: Any, *, request: Mapping[str, Any]) -> Mapping[str, Any]:
    from .closeout_snapshot import validate_closeout_snapshot
    request = validate_request(request)
    fields = {"schema", "request_cid", "task_cid", "task_alias", "queue_binding",
              "queue_settlement", "control_snapshot", "population", "population_cid",
              "matching_rows", "population_complete", "guard_retained", "claim_authority",
              "retry_authorized", "completion_authority", "observation_cid", "legacy_double_encoding"}
    _require(isinstance(value, Mapping) and set(value) == fields, "queue observation shape differs")
    _require(len(canonical_json_bytes(value)) <= MAX_RESULT_BYTES, "queue observation exceeds byte bound")
    material = {k: v for k, v in value.items() if k != "observation_cid"}
    _require(value["schema"] == SCHEMA and content_identity(material) == value["observation_cid"]
             and value["request_cid"] == request["request_cid"]
             and value["task_cid"] == request["task_cid"]
             and value["legacy_double_encoding"] == DOUBLE_ENCODING
             and value["population_complete"] is True
             and all(value[k] is False for k in ("guard_retained", "claim_authority",
                                                 "retry_authorized", "completion_authority")),
             "queue observation identity, seal or authority differs")
    control = validate_closeout_snapshot(value["control_snapshot"], request=request["completion_request"])
    tasks = control["closeout_facts"]["relations"]["tasks"]
    _require(tasks["available"] and not tasks["truncated"]
             and [row["task_alias"] for row in tasks["rows"]
                  if row["task_cid"] == request["task_cid"]] == [value["task_alias"]],
             "queue observation task alias differs from owner snapshot")
    binding, settlement = value["queue_binding"], value["queue_settlement"]
    _require(isinstance(binding, Mapping) and set(binding) == {
        "queue_dir", "target_repository_id", "target_branch", "database_scope_cid", "identities"}
        and isinstance(settlement, Mapping), "queue observation binding differs")
    settlement_material = {k: v for k, v in settlement.items() if k != "receipt_cid"}
    _require(type(settlement_material.get("max_updated_at")) is str,
             "queue settlement timestamp encoding differs")
    timestamp = float.fromhex(settlement_material["max_updated_at"])
    _require(math.isfinite(timestamp) and timestamp >= 0
             and timestamp.hex() == settlement_material["max_updated_at"],
             "queue settlement timestamp encoding differs")
    settlement_material["max_updated_at"] = timestamp
    _require(content_identity(settlement_material) == settlement.get("receipt_cid")
             and settlement["database"]["path"] == str(Path(binding["queue_dir"]) / "merge_queue.duckdb")
             and {k: settlement["database"][k] for k in ("device", "inode")}
                 == binding["identities"]["merge_queue.duckdb"]
             and settlement["target"]["repository_id"] == binding["target_repository_id"]
             and settlement["target"]["branch"] == binding["target_branch"],
             "queue observation settlement or bound store identity differs")
    population, matches = value["population"], value["matching_rows"]
    _require(type(population) is list and len(population) <= MAX_POPULATION
             and len(population) == value["queue_settlement"]["row_count"]
             and content_identity(population) == value["population_cid"]
             and type(matches) is list and len(matches) <= MAX_MATCHES,
             "queue observation population differs or exceeds bound")
    for item in population:
        _require(isinstance(item, Mapping) and set(item) == {
            "request_id", "task_id", "canonical_task_id", "canonical_task_key",
            "completion_task_cids", "row_cid"}, "queue population entry shape differs")
        bindings = item["completion_task_cids"]
        _require(all(_text(item[k]) for k in ("request_id", "task_id", "canonical_task_id", "row_cid"))
                 and type(item["canonical_task_key"]) is str
                 and isinstance(bindings, Mapping) and bool(bindings)
                 and all(_text(k) and _text(v) for k, v in bindings.items())
                 and bindings.get(item["task_id"]) == item["canonical_task_id"],
                 "queue population task bindings differ")
    for row in matches:
        decode_observed_queue_row(row)
    ids = [item["request_id"] for item in population]
    _require(ids == sorted(set(ids)), "queue observation population is not unique and ordered")
    expected = [item for item in population if _matches(item, value["task_alias"], value["task_cid"])]
    _require([row["request_id"] for row in matches] == [item["request_id"] for item in expected]
             and all(content_identity(row) == item["row_cid"] for row, item in zip(matches, expected)),
             "queue observation omitted or changed matching authority rows")
    return value
