"""Actual isolated native metadata hydration, tamper rejection and restart."""

from copy import deepcopy
import json
from pathlib import Path

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata


def _records():
    return {"ast": [{"row_id": "increment", "node": "FunctionDef", "line": 1,
                     "source_sha256": "a" * 64, "candidate": True, "proof_authority": False}],
            "kg": [{"subject": "increment", "relation": "returns", "object": "n+1"}],
            "vectors": [{"unit": index, "values": [0.5, -0.0, index / 10], "model_sha256": "b" * 64}
                        for index in range(23)],
            "contracts": [{"assumptions": ["pure integer"], "property": "result=n+1",
                           "evidence_kind": "checked_bounded_solver", "proof_authority": True}],
            "frontiers": [{"reason": "dynamic call", "supported": False, "extra": {"unicode": "é"}}]}


@pytest.fixture
def hydrated(tmp_path):
    output = tmp_path / "metadata"
    records = _records()
    source = {"schema": "source-fixture@1", "snapshot_cid": "fixture-snapshot", "dirty": True}
    report = metadata.hydrate_codebase_ir_metadata(records=records, output=output, source_snapshot=source)
    return output, report, records


def test_native_tables_family_views_chunks_and_fresh_process_preserve_all_fields(hydrated):
    output, report, records = hydrated
    assert report["row_count"] == sum(map(len, records.values()))
    assert report["family_counts"] == {name: len(rows) for name, rows in records.items()}
    assert report["fresh_process_readback"]["verified"] is True
    assert report["fresh_process_readback"]["row_count"] == report["row_count"]
    assert len(report["lake_snapshot_ids"]) >= 1
    assert len(set(report["lake_snapshot_ids"])) == len(report["lake_snapshot_ids"])
    assert report["lake_packet_count"] < report["row_count"]
    assert report["lake_layout"]["schema"] == metadata.PACKET_SCHEMA
    assert report["lake_layout"]["typed_vector_columns"] is False
    assert report["qualification"]["proof_authority"] is False
    assert report["qualification"]["production_activated"] is False
    with duckdb.connect(str(output / "metadata.duckdb"), read_only=True) as cx:
        for family, expected_rows in records.items():
            found = cx.execute(f"SELECT payload_json FROM family_{family} ORDER BY row_ordinal").fetchall()
            assert [json.loads(row[0]) for row in found] == expected_rows
            assert [row[0] for row in found] == [metadata._json(value).decode() for value in expected_rows]
        constraints = cx.execute("SELECT constraint_type FROM duckdb_constraints() "
                                 "WHERE table_name='metadata_rows'").fetchall()
        assert ("PRIMARY KEY",) in constraints
        assert ("UNIQUE",) in constraints
    for family, descriptor in report["exports"].items():
        rows = [json.loads(line) for line in (output / descriptor["relative_path"]).read_text().splitlines()]
        assert [row["payload"] for row in rows] == records[family]
        assert len({row["row_id"] for row in rows}) == len(rows)
    assert any((output / "lake/data").rglob("*.parquet"))
    reopened = metadata.validate_codebase_ir_metadata(output=output, expected=report, fresh_process=True)
    assert reopened == report


@pytest.mark.parametrize("mutation", ["duplicate", "conflicting_identity", "nonfinite", "tuple", "bad_family", "too_large"])
def test_invalid_population_rejects_before_namespace_creation(tmp_path, mutation):
    records = _records()
    if mutation == "duplicate":
        records["ast"].append(deepcopy(records["ast"][0]))
    elif mutation == "conflicting_identity":
        changed = deepcopy(records["ast"][0])
        changed["line"] = 2
        records["ast"].append(changed)
    elif mutation == "nonfinite":
        records["vectors"][0]["values"].append(float("nan"))
    elif mutation == "tuple":
        records["ast"][0]["foreign"] = (1, 2)
    elif mutation == "bad_family":
        records["bad;drop_table"] = []
    elif mutation == "too_large":
        records["ast"][0]["extra"] = "x" * metadata.LIMITS["row_bytes"]
    output = tmp_path / "rejected"
    with pytest.raises(metadata.MetadataError):
        metadata.hydrate_codebase_ir_metadata(records=records, output=output, source_snapshot={"snapshot": "fixture"})
    assert not output.exists()


def test_changed_native_row_cannot_reuse_old_identity(hydrated):
    output, report, _ = hydrated
    with duckdb.connect(str(output / "metadata.duckdb")) as cx:
        cx.execute("UPDATE metadata_rows SET payload_json=? WHERE family='ast'", ['{"node":"changed"}'])
    with pytest.raises(metadata.MetadataError, match="native row identity"):
        metadata.validate_codebase_ir_metadata(output=output, expected=report)


def test_changed_export_is_rejected(hydrated):
    output, report, _ = hydrated
    path = output / report["exports"]["ast"]["relative_path"]
    path.write_bytes(b'{}\n')
    with pytest.raises(metadata.MetadataError, match="family export"):
        metadata.validate_codebase_ir_metadata(output=output, expected=report)


def test_partial_manifest_and_changed_manifest_binding_are_rejected(hydrated):
    output, report, _ = hydrated
    path = output / "manifest.json"
    original = path.read_bytes()
    value = json.loads(original)
    del value["batches"]
    path.write_bytes(metadata._json(value))
    with pytest.raises(metadata.MetadataError, match="incomplete"):
        metadata.validate_codebase_ir_metadata(output=output)
    path.write_bytes(original)
    wrong = deepcopy(report)
    wrong["source_snapshot_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(metadata.MetadataError, match="expected metadata binding"):
        metadata.validate_codebase_ir_metadata(output=output, expected=wrong)


def test_changed_lake_record_is_rejected(hydrated):
    output, report, _ = hydrated
    history = metadata._history()
    with history.IsolatedNativeDuckLakeHistory(output / "lake") as sink:
        sink._connection.execute("UPDATE history.events SET event_json=? WHERE event_id=?",
                                 ['{}', report["lake_receipts"][0]["event_ids"][0]])
    with pytest.raises(metadata.MetadataError, match="event identity/content conflict"):
        metadata.validate_codebase_ir_metadata(output=output, expected=report)


def test_exclusive_owner_and_existing_namespace_are_not_overwritten(hydrated):
    output, report, records = hydrated
    with metadata._owner(output):
        with pytest.raises(metadata.MetadataError, match="already has an owner"):
            metadata.validate_codebase_ir_metadata(output=output)
    with pytest.raises(metadata.MetadataError, match="output must be new"):
        metadata.hydrate_codebase_ir_metadata(records=records, output=output, source_snapshot={"snapshot": "fixture"})
    assert metadata.validate_codebase_ir_metadata(output=output, expected=report)["row_count"] == report["row_count"]


def test_near_depth_bound_record_roundtrips_through_string_packet(tmp_path):
    nested = "complete original leaf"
    for _ in range(31):
        nested = {"next": nested}
    payload = {"nested": nested, "proof_authority": False, "candidate": True}
    metadata._plain(payload)
    # The original fits; embedding it as an ordinary event object adds depth.
    with pytest.raises(metadata.MetadataError, match="structural bound"):
        metadata._plain({"payload": payload})
    output = tmp_path / "deep"
    report = metadata.hydrate_codebase_ir_metadata(records={"ast": [payload]}, output=output,
                                                   source_snapshot={"snapshot": "depth-bound-fixture"})
    assert report["row_count"] == report["lake_packet_count"] == 1
    assert report["fresh_process_readback"]["verified"] is True
    history = metadata._history()
    with history.IsolatedNativeDuckLakeHistory(output / "lake") as sink:
        event = json.loads(sink._connection.execute("SELECT event_json FROM history.events").fetchone()[0])
        packet = event["payload"]
        assert type(packet["payload_json"]) is str
        assert metadata._packet_rows(packet)[0]["payload"] == payload
    assert metadata.validate_codebase_ir_metadata(output=output, expected=report)["row_count"] == 1


def test_thousand_rows_use_bounded_packets_with_complete_native_restart(tmp_path):
    records = {"vectors": [{"row_id": str(index), "values": [index / 100, -0.0],
                            "candidate": True, "proof_authority": False} for index in range(1200)]}
    output = tmp_path / "many"
    report = metadata.hydrate_codebase_ir_metadata(records=records, output=output,
                                                   source_snapshot={"snapshot": "scale-fixture"})
    assert report["row_count"] == report["family_counts"]["vectors"] == 1200
    assert report["lake_packet_count"] == 12
    assert len(report["lake_snapshot_ids"]) == 2
    assert report["fresh_process_readback"]["row_count"] == 1200
    history = metadata._history()
    with history.IsolatedNativeDuckLakeHistory(output / "lake") as sink:
        events = sink._connection.execute("SELECT event_json FROM history.events").fetchall()
        assert len(events) == 12
        restored = []
        for (raw,) in events:
            packet = json.loads(raw)["payload"]
            assert packet["row_count"] <= metadata.LIMITS["packet_rows"]
            assert packet["payload_bytes"] <= metadata.LIMITS["packet_bytes"]
            restored.extend(metadata._packet_rows(packet))
        restored.sort(key=lambda row: row["row_ordinal"])
        assert [row["payload"] for row in restored] == records["vectors"]
    assert len(list((output / "lake/data").rglob("*.parquet"))) < 30


@pytest.mark.parametrize("mutation", ["partial", "changed_row", "ordered_ids", "payload_digest"])
def test_self_consistent_packet_envelope_cannot_hide_partial_or_changed_rows(mutation):
    rows = metadata._families({"ast": [{"name": "first", "proof_authority": False},
                                        {"name": "second", "proof_authority": False}]}, "sha256:" + "a" * 64)["ast"]
    packet = metadata._packet(rows)
    assert metadata._packet_rows(packet) == rows
    if mutation == "partial":
        values = rows[:1]
    elif mutation == "changed_row":
        values = deepcopy(rows)
        values[0]["payload"]["proof_authority"] = True
    else:
        values = rows
    raw = metadata._row_array(values)
    packet.update(payload_json=raw.decode(), payload_bytes=len(raw), payload_sha256=metadata._digest(raw))
    if mutation == "ordered_ids":
        packet["ordered_row_ids"].reverse()
    elif mutation == "payload_digest":
        packet["payload_sha256"] = "sha256:" + "0" * 64
    # Even a new, self-consistent outer packet ID cannot bless inner omissions,
    # changed authority labels under an old row identity, or changed membership.
    packet["packet_id"] = metadata._digest(metadata._json({key: value for key, value in packet.items()
                                                          if key != "packet_id"}))
    with pytest.raises(metadata.MetadataError):
        metadata._packet_rows(packet)
