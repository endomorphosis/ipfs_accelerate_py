"""Pure lossless partition and assembly controls; no native-store qualification.

The orchestration fixture replaces only the legacy hydrate/validate callbacks
with JSON-only stand-ins. It never constructs a native owner, database, model,
prover, worker or subprocess. Actual native part and fresh-process evidence is
provided separately by the joined experiment.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import finite_repository_sharded_metadata as shards
from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as native
from benchmarks.agent_supervisor.container_coding.terminal_codebase_supervisor_fixture import (
    bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
)


def _load(output):
    return json.loads((output / "manifest.json").read_bytes())


def _rewrite(output, manifest):
    (output / "manifest.json").write_bytes(native._wire(manifest))


@pytest.fixture
def assembly(tmp_path, monkeypatch):
    """Small deterministic parts, unchanged production constants, JSON callbacks."""
    events = []
    original_partition = shards.partition_codebase_ir_metadata_records
    monkeypatch.setattr(shards, "partition_codebase_ir_metadata_records",
                        lambda *, records: original_partition(records=records, max_rows=2))

    def inspect_part(output):
        manifest = _load(output)
        source = manifest["source_snapshot_sha256"]
        for family, descriptor in manifest["families"].items():
            raw = native._read(output / descriptor["export"]["relative_path"],
                               2 * native.LIMITS["total_payload_bytes"])
            assert len(raw) == descriptor["export"]["bytes"]
            assert native._digest(raw) == descriptor["export"]["sha256"]
            rows = [native._loads(line) for line in raw.splitlines()]
            assert rows == [native._row(family, row["payload"], index, source)
                            for index, row in enumerate(rows)]
            assert descriptor["count"] == len(rows)
            assert descriptor["digest"] == native._digest(native._row_array(rows))
        return native._report(manifest, native._digest(native._read(output / "manifest.json", native.LIMITS["manifest_bytes"])))

    def hydrate(*, records, output, source_snapshot):
        events.append(("hydrate", str(output), deepcopy(records), deepcopy(source_snapshot)))
        source = native._digest(native._json(source_snapshot))
        rows = native._families(records, source)
        output.mkdir()
        (output / "exports").mkdir()
        families = {}
        for family, values in rows.items():
            relative = "exports/" + family + ".jsonl"
            raw = b"".join(native._wire(row) + b"\n" for row in values)
            native._write(output / relative, raw)
            families[family] = {"count": len(values), "digest": native._digest(native._row_array(values)),
                                "view": "metadata_" + family,
                                "export": {"relative_path": relative, "sha256": native._digest(raw), "bytes": len(raw)}}
        manifest = {"schema": native.SCHEMA, "output": str(output), "source_snapshot": source_snapshot,
                    "source_snapshot_sha256": source, "row_count": sum(map(len, rows.values())),
                    "row_root_sha256": native._row_root(rows), "families": families,
                    "native_runtime": {"structural_fixture_only": True}, "limits": dict(native.LIMITS),
                    "lake_layout": "structural_fixture_only", "lake_packet_count": 0,
                    "lake_snapshot_digest": native._digest(b"[]"), "batches": []}
        native._write(output / "manifest.json", native._wire(manifest))
        result = inspect_part(output)
        result["fresh_process_readback"] = {"verified": True, "method": "structural_fixture_only"}
        return result

    def validate(*, output, expected, fresh_process):
        events.append(("validate", str(output), fresh_process))
        result = inspect_part(output)
        assert native._wire(result) == native._wire(expected)
        if fresh_process:
            result["fresh_process_readback"] = {"verified": True, "method": "structural_fixture_only"}
        return result

    monkeypatch.setattr(native, "hydrate_codebase_ir_metadata", hydrate)
    monkeypatch.setattr(native, "validate_codebase_ir_metadata", validate)
    records = {"training": [{"row_id": "train-0", "epochs": 16}, {"row_id": "train-1", "epochs": 16}],
               "ast": [{"symbol": "alpha"}, {"symbol": "beta"}, {"symbol": "gamma"}], "empty": []}
    output = tmp_path / "assembly"
    report = shards.hydrate_sharded_codebase_ir_metadata(records=records, output=output,
                                                        source_snapshot={"producer": "pure-fixture", "generation": 2})
    return SimpleNamespace(output=output, report=report, records=records, events=events,
                           inspect_part=inspect_part, validate=validate, original_partition=original_partition)


def test_caps_preserve_legacy_limits_and_no_new_authority():
    assert shards.LIMITS == {"part_payload_bytes": 32 * 1024**2, "part_rows": 32768, "parts": 16,
                            "total_payload_bytes": 512 * 1024**2, "rows": 524288, "families": 32,
                            "manifest_bytes": 16 * 1024**2, "export_bytes": 256 * 1024**2,
                            "restart_seconds": 300}
    assert native.LIMITS["total_payload_bytes"] == 64 * 1024**2
    assert native.LIMITS["rows"] == 65536 and native.LIMITS["row_bytes"] == 256 * 1024
    assert shards.CHUNK_POLICY["bytes"] == 49152
    assert shards.QUALIFICATION["experimental"] is True
    assert all(value is False for key, value in shards.QUALIFICATION.items() if key != "experimental")


@pytest.mark.parametrize("records", [{}, {"empty": []}, {"ast": []}])
def test_empty_inputs_keep_original_keys_and_one_bounded_part(records):
    plan = shards.partition_codebase_ir_metadata_records(records=records)
    assert plan["input_families"] == sorted(records)
    assert set(plan["records"]) == set(records) | set(native.DEFAULT_FAMILIES)
    assert len(plan["parts"]) == 1 and plan["row_count"] == plan["payload_bytes"] == 0
    assert plan["packaged_sha256"] == native._digest(native._wire(records))
    assert all(span == {"start": 0, "stop": 0, "count": 0}
               for span in plan["parts"][0]["global_family_ranges"].values())


def test_exact_byte_boundary_and_global_ranges_preserve_order():
    records = {"vectors": [{"v": 0}], "ast": [{"n": 0}, {"n": 1}, {"n": 2}]}
    size = len(native._json(records["ast"][0]))
    plan = shards.partition_codebase_ir_metadata_records(records=records, max_payload_bytes=2 * size)
    assert [part["row_count"] for part in plan["parts"]] == [2, 2]
    assert [part["payload_bytes"] for part in plan["parts"]] == [2 * size, 2 * size]
    assert [part["global_family_ranges"]["ast"] for part in plan["parts"]] == [
        {"start": 0, "stop": 2, "count": 2}, {"start": 2, "stop": 3, "count": 1}]
    assert plan == shards.partition_codebase_ir_metadata_records(records=dict(reversed(list(records.items()))),
                                                                 max_payload_bytes=2 * size)
    assert [row for part in plan["parts"] for row in part["records"]["ast"]] == records["ast"]
    records["ast"][0]["n"] = 999
    assert plan["parts"][0]["records"]["ast"][0]["n"] == 0


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, "4", 32 * 1024**2 + 1])
def test_partition_rejects_invalid_or_increased_byte_budget(budget):
    with pytest.raises(ValueError, match="budgets"):
        shards.partition_codebase_ir_metadata_records(records={}, max_payload_bytes=budget)


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, "4", 32769])
def test_partition_rejects_invalid_or_increased_row_budget(budget):
    with pytest.raises(ValueError, match="budgets"):
        shards.partition_codebase_ir_metadata_records(records={}, max_rows=budget)


@pytest.mark.parametrize("records", [[], {1: []}, {"bad family": []}, {"ast": ()}, {"ast": [1]},
                                     {"ast": [{"n": float("nan")}]}, {"ast": [{"n": (1, 2)}]}])
def test_partition_rejects_unbounded_or_nonordinary_input(records):
    with pytest.raises(ValueError):
        shards.partition_codebase_ir_metadata_records(records=records)


def test_partition_refuses_one_oversized_payload_and_too_many_parts():
    with pytest.raises(ValueError, match="one payload"):
        shards.partition_codebase_ir_metadata_records(records={"ast": [{"blob": "x" * native.LIMITS["row_bytes"]}]})
    with pytest.raises(ValueError, match="one payload"):
        shards.partition_codebase_ir_metadata_records(records={"ast": [{"n": 0}]}, max_payload_bytes=1)
    with pytest.raises(ValueError, match="part count"):
        shards.partition_codebase_ir_metadata_records(records={"ast": [{"n": index} for index in range(17)]}, max_rows=1)


def test_global_family_bound_includes_default_families():
    accepted = {"family_" + str(index): [] for index in range(28)}
    assert len(shards.partition_codebase_ir_metadata_records(records=accepted)["records"]) == 32
    accepted["extra"] = []
    with pytest.raises(ValueError, match="family bound"):
        shards.partition_codebase_ir_metadata_records(records=accepted)


@pytest.mark.parametrize("rows", [[{"n": 0}, {"n": 1}, {"n": 0}],
                                  [{"row_id": "same", "n": 0}, {"n": 1}, {"row_id": "same", "n": 2}]])
def test_duplicate_identity_is_rejected_across_part_boundary(rows):
    with pytest.raises(ValueError, match="global.*identity"):
        shards.partition_codebase_ir_metadata_records(records={"ast": rows}, max_rows=1)


@pytest.mark.parametrize("field,cap", [("rows", 2), ("total_payload_bytes", 15)])
def test_global_caps_reject_without_omission_under_reduced_fixture_budget(monkeypatch, field, cap):
    monkeypatch.setattr(shards, "LIMITS", {**shards.LIMITS, field: cap})
    with pytest.raises(ValueError, match="global metadata population"):
        shards.partition_codebase_ir_metadata_records(records={"ast": [{"n": index} for index in range(3)]})


def test_chunked_producer_spans_parts_with_global_descriptor_ordinal():
    producers = {"artifacts": [{"n": 0}, {"content": "x" * 300000}], "empty": []}
    packaged = bound_terminal_codebase_metadata_records(producers)
    plan = shards.partition_codebase_ir_metadata_records(records=packaged, max_rows=2)
    assert len(plan["parts"]) > 2
    assert plan["records"]["artifacts"][1]["original_ordinal"] == 1
    recovered = {family: [row for part in plan["parts"] for row in part["records"][family]]
                 for family in plan["input_families"]}
    assert recovered == packaged
    assert reconstruct_terminal_codebase_metadata_records(recovered) == producers
    assert plan["producer_sha256"] == native._digest(native._wire(producers))


@pytest.mark.parametrize("control", ["missing", "duplicate", "changed", "reordered", "orphan"])
def test_partition_refuses_incomplete_or_altered_chunk_population(control):
    packaged = bound_terminal_codebase_metadata_records({"artifacts": [{"content": "x" * 300000}]})
    chunks = packaged[shards.CHUNK_POLICY["family"]]
    if control == "missing":
        chunks.pop()
    elif control == "duplicate":
        chunks.append(deepcopy(chunks[0]))
    elif control == "changed":
        chunks[0]["base64"] = "AA=="
    elif control == "reordered":
        chunks.reverse()
    else:
        packaged["artifacts"] = []
    with pytest.raises(ValueError):
        shards.partition_codebase_ir_metadata_records(records=packaged, max_rows=2)


@pytest.mark.parametrize("control", ["bool-ordinal", "bool-bytecount", "bool-chunkcount", "oversized-producer", "wrong-chunkcount"])
def test_artifact_descriptors_obey_original_producer_limit_and_exact_types(control):
    packaged = bound_terminal_codebase_metadata_records({"artifacts": [{"content": "x" * 300000}]})
    descriptor = packaged["artifacts"][0]
    if control == "bool-ordinal":
        descriptor["original_ordinal"] = False
    elif control == "bool-bytecount":
        descriptor["payload_bytes"] = True
    elif control == "bool-chunkcount":
        descriptor["chunk_count"] = True
    elif control == "oversized-producer":
        descriptor["payload_bytes"] = 32000001
    else:
        descriptor["chunk_count"] += 1
    with pytest.raises(ValueError, match="bounded producer artifact descriptor"):
        shards.partition_codebase_ir_metadata_records(records=packaged)


def test_assembly_exact_reconstruction_and_native_local_ordinal_reset(assembly):
    result = shards.validate_sharded_codebase_ir_metadata(output=assembly.output, expected=assembly.report,
                                                         fresh_process=False)
    assert "fresh_process_readback" not in result
    assert result["row_count"] == 5 and len(result["parts"]) == 3
    assert result["producer_sha256"] == native._digest(native._wire(assembly.records))
    assert shards.reconstruct_sharded_codebase_ir_metadata_records(output=assembly.output,
                                                                  expected=assembly.report) == assembly.records
    native_gamma = json.loads((assembly.output / "parts/000001/exports/ast.jsonl").read_bytes().splitlines()[0])
    global_gamma = json.loads((assembly.output / "exports/ast.jsonl").read_bytes().splitlines()[2])
    assert native_gamma["row_ordinal"] == 0 and global_gamma["row_ordinal"] == 2
    assert native_gamma["payload"] == global_gamma["payload"] == {"symbol": "gamma"}
    assert native_gamma["source_snapshot_sha256"] != global_gamma["source_snapshot_sha256"]
    assert all(event[2] is False for event in assembly.events if event[0] == "validate")


@pytest.mark.parametrize("control", ["missing-part", "extra-part", "duplicate-part", "part-order",
                                     "global-export", "native-export", "source-bytes", "source-sha",
                                     "native-manifest", "producer-digest", "packaged-digest", "row-root",
                                     "global-range", "range-bool", "row-count-bool", "family-count-bool",
                                     "part-row-count-bool", "native-limits", "increased-limits", "proof-claim",
                                     "unknown-field", "bad-input-families", "missing-family", "nonmapping-families"])
def test_assembly_refuses_missing_changed_or_false_authority_evidence(assembly, control):
    output, manifest = assembly.output, _load(assembly.output)
    if control == "missing-part":
        (output / "parts/000001").rename(output / "hidden-part")
    elif control == "extra-part":
        (output / "parts/unaccounted").mkdir()
    elif control == "duplicate-part":
        manifest["parts"][1] = deepcopy(manifest["parts"][0])
    elif control == "part-order":
        manifest["parts"].reverse()
    elif control in ("global-export", "native-export"):
        path = output / ("exports/ast.jsonl" if control == "global-export" else "parts/000000/exports/ast.jsonl")
        path.write_bytes(path.read_bytes() + b"\n")
    elif control == "source-bytes":
        (output / "source-snapshot.json").write_bytes(b"{}")
    elif control == "source-sha":
        manifest["source_snapshot_sha256"] = native._digest(b"{}")
    elif control == "native-manifest":
        (output / "parts/000000/manifest.json").write_bytes(b"{}")
    elif control in ("producer-digest", "packaged-digest", "row-root"):
        key = {"producer-digest": "producer_sha256", "packaged-digest": "packaged_sha256", "row-root": "row_root_sha256"}[control]
        manifest[key] = native._digest(b"wrong")
    elif control in ("global-range", "range-bool"):
        manifest["parts"][0]["global_family_ranges"]["ast"]["start"] = 1 if control == "global-range" else False
    elif control == "row-count-bool":
        manifest["row_count"] = True
    elif control == "family-count-bool":
        manifest["family_counts"]["kg"] = False
    elif control == "part-row-count-bool":
        manifest["parts"][0]["row_count"] = True
    elif control == "native-limits":
        manifest["native_limits"]["total_payload_bytes"] *= 2
    elif control == "increased-limits":
        manifest["limits"]["part_payload_bytes"] *= 2
    elif control == "proof-claim":
        manifest["qualification"]["proof_authority"] = True
    elif control == "unknown-field":
        manifest["unreviewed_authority"] = True
    elif control == "bad-input-families":
        manifest["input_families"].append([])
    elif control == "missing-family":
        del manifest["exports"]["kg"]
    elif control == "nonmapping-families":
        manifest["family_counts"] = None
    _rewrite(output, manifest)
    with pytest.raises((ValueError, OSError)):
        shards.validate_sharded_codebase_ir_metadata(output=output, fresh_process=False)


@pytest.mark.parametrize("artifact", ["source-snapshot.json", "exports/ast.jsonl", "parts/000000/manifest.json"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_assembly_requires_independent_regular_artifacts(assembly, artifact, alias):
    path = assembly.output / artifact
    external = assembly.output.parent / "external"
    if alias == "symlink":
        path.rename(external)
        path.symlink_to(external)
    else:
        os.link(path, external)
    with pytest.raises(ValueError):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)


def test_native_callback_refusal_is_propagated(assembly, monkeypatch):
    def refuse(**_):
        raise ValueError("legacy native evidence refused")
    monkeypatch.setattr(native, "validate_codebase_ir_metadata", refuse)
    with pytest.raises(ValueError, match="legacy native evidence refused"):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)


def test_late_callback_tamper_of_earlier_artifact_is_rejected(assembly, monkeypatch):
    def mutate(**kwargs):
        report = assembly.validate(**kwargs)
        if kwargs["output"].name == "000002":
            path = assembly.output / "parts/000000/exports/ast.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
        return report
    monkeypatch.setattr(native, "validate_codebase_ir_metadata", mutate)
    with pytest.raises(ValueError, match="artifact bytes"):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)


def test_fresh_global_command_reassembles_all_parts_after_owner_lock_release(assembly, monkeypatch):
    commands = []
    def run(argv, **kwargs):
        commands.append((argv, kwargs))
        assert argv[:4] == [shards.sys.executable, "-B", "-m", shards.MODULE]
        assert argv[4:] == ["--validate", str(assembly.output), "--manifest-sha256", assembly.report["manifest_sha256"]]
        assert kwargs["timeout"] == 300 and kwargs["capture_output"] is True and kwargs["text"] is True
        report = shards.validate_sharded_codebase_ir_metadata(output=assembly.output,
                    expected={"manifest_sha256": argv[-1]}, fresh_process=False)
        return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
    monkeypatch.setattr(shards.subprocess, "run", run)
    result = shards.validate_sharded_codebase_ir_metadata(output=assembly.output, expected=assembly.report,
                                                        fresh_process=True)
    assert len(commands) == 1
    assert result["fresh_process_readback"]["global_new_process_verified"] is True
    assert len(result["fresh_process_readback"]["parts"]) == 3
    assert [event[2] for event in assembly.events if event[0] == "validate"] == [True] * 3 + [False] * 3
    assert result["qualification"] == shards.QUALIFICATION


@pytest.mark.parametrize("control", ["failed", "invalid-json", "incomplete", "bool-alias", "duplicate-field", "late-source-change"])
def test_global_fresh_failure_or_changed_evidence_cannot_verify(assembly, monkeypatch, control):
    def run(*_, **__):
        observed = shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)
        if control == "failed":
            return SimpleNamespace(returncode=1, stdout="", stderr="fresh refused")
        if control == "invalid-json":
            return SimpleNamespace(returncode=0, stdout="broken", stderr="")
        if control == "duplicate-field":
            return SimpleNamespace(returncode=0, stdout='{"row_count":0,' + json.dumps(observed)[1:], stderr="")
        if control == "incomplete":
            del observed["parts"]
        if control == "bool-alias":
            observed["qualification"]["proof_authority"] = 0
        if control == "late-source-change":
            (assembly.output / "source-snapshot.json").write_bytes(b"{}")
        return SimpleNamespace(returncode=0, stdout=json.dumps(observed), stderr="")
    monkeypatch.setattr(shards.subprocess, "run", run)
    with pytest.raises(ValueError):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=True)


@pytest.mark.parametrize("fresh", [1, 0, "false", None])
def test_fresh_option_requires_boolean(assembly, fresh):
    with pytest.raises(ValueError, match="boolean"):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=fresh)


@pytest.mark.parametrize("expected", [{}, {"manifest_sha256": "wrong"}, {"manifest_sha256": "actual", "unknown": 1}])
def test_expected_identity_cannot_be_omitted_or_replaced(assembly, expected):
    if expected.get("manifest_sha256") == "actual":
        expected["manifest_sha256"] = assembly.report["manifest_sha256"]
    with pytest.raises(ValueError):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, expected=expected, fresh_process=False)


def test_invalid_input_is_refused_before_namespace_or_native_callback(tmp_path, monkeypatch):
    def unreachable(**_):
        pytest.fail("native callback must not run")
    monkeypatch.setattr(native, "hydrate_codebase_ir_metadata", unreachable)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        shards.hydrate_sharded_codebase_ir_metadata(records={"ast": [{"blob": "x" * 262144}]},
            output=output, source_snapshot={"producer": "pure"})
    assert not output.exists()


def test_source_producer_attribution_must_match_before_native_work(tmp_path, monkeypatch):
    def unreachable(**_):
        pytest.fail("contradictory producer attribution must not run a native callback")
    monkeypatch.setattr(native, "hydrate_codebase_ir_metadata", unreachable)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="producer attribution"):
        shards.hydrate_sharded_codebase_ir_metadata(records={"ast": [{"n": 0}]}, output=output,
            source_snapshot={"producer_sha256": native._digest(b"wrong")})
    assert not output.exists()


@pytest.mark.parametrize("control", ["missing-manifest", "missing-source", "native-report-none", "native-report-missing-sha"])
def test_public_validation_normalizes_malformed_artifacts(assembly, control):
    manifest = _load(assembly.output)
    if control == "missing-manifest":
        (assembly.output / "manifest.json").unlink()
    elif control == "missing-source":
        (assembly.output / "source-snapshot.json").unlink()
    elif control == "native-report-none":
        manifest["parts"][0]["native_report"] = None
        _rewrite(assembly.output, manifest)
    else:
        del manifest["parts"][0]["native_report"]["manifest_sha256"]
        _rewrite(assembly.output, manifest)
    with pytest.raises(shards.ShardedMetadataError):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)


def test_reconstruction_checks_complete_expected_report(assembly):
    expected = deepcopy(assembly.report)
    expected["qualification"]["proof_authority"] = 0
    with pytest.raises(ValueError, match="expected reconstruction binding"):
        shards.reconstruct_sharded_codebase_ir_metadata_records(output=assembly.output, expected=expected)


def test_cold_global_source_context_keeps_unchanged_native_bound(assembly):
    manifest = _load(assembly.output)
    manifest["source_snapshot"]["oversized"] = "x" * native.LIMITS["source_bytes"]
    manifest["source_snapshot_sha256"] = native._digest(native._json(manifest["source_snapshot"]))
    _rewrite(assembly.output, manifest)
    with pytest.raises(ValueError, match="source snapshot bound"):
        shards.validate_sharded_codebase_ir_metadata(output=assembly.output, fresh_process=False)


def test_cold_rehashed_assembly_cannot_relabel_complete_producer(assembly):
    """An entirely rehashed authored fixture must still obey source attribution."""
    output, manifest = assembly.output, _load(assembly.output)
    manifest["source_snapshot"]["producer_sha256"] = native._digest(b"contradictory producer")
    raw = native._json(manifest["source_snapshot"])
    (output / "source-snapshot.json").write_bytes(raw)
    manifest["source_snapshot_sha256"] = native._digest(raw)
    manifest["source_snapshot_artifact"].update(sha256=native._digest(raw), bytes=len(raw))
    all_payloads = {family: [] for family in manifest["family_counts"]}
    for part in manifest["parts"]:
        directory = output / part["relative_path"]
        local = _load(directory)
        local["source_snapshot"] = shards._part_source(manifest, part)
        local["source_snapshot_sha256"] = native._digest(native._json(local["source_snapshot"]))
        part["source_snapshot_sha256"] = local["source_snapshot_sha256"]
        rows = {}
        for family, family_descriptor in local["families"].items():
            path = directory / family_descriptor["export"]["relative_path"]
            payloads = [json.loads(line)["payload"] for line in path.read_bytes().splitlines()]
            all_payloads[family].extend(payloads)
            rows[family] = [native._row(family, payload, index, local["source_snapshot_sha256"])
                            for index, payload in enumerate(payloads)]
            export_raw = b"".join(native._wire(row) + b"\n" for row in rows[family])
            path.write_bytes(export_raw)
            family_descriptor["export"].update(sha256=native._digest(export_raw), bytes=len(export_raw))
            family_descriptor["digest"] = native._digest(native._row_array(rows[family]))
        local["row_root_sha256"] = native._row_root(rows)
        _rewrite(directory, local)
        part["native_report"] = assembly.inspect_part(directory)
        part_raw = native._wire(local)
        part["native_manifest_sha256"] = native._digest(part_raw)
        part["native_manifest"].update(sha256=native._digest(part_raw), bytes=len(part_raw))
    for family, payloads in all_payloads.items():
        raw = b"".join(native._wire(native._row(family, payload, index, manifest["source_snapshot_sha256"])) + b"\n"
                       for index, payload in enumerate(payloads))
        (output / manifest["exports"][family]["relative_path"]).write_bytes(raw)
        manifest["exports"][family].update(sha256=native._digest(raw), bytes=len(raw))
    manifest["row_root_sha256"] = shards._global_row_root(all_payloads, manifest["source_snapshot_sha256"])
    _rewrite(output, manifest)
    with pytest.raises(ValueError, match="producer attribution"):
        shards.validate_sharded_codebase_ir_metadata(output=output, fresh_process=False)
