"""Pure transport controls with authored byte fixtures; not native qualification."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import inventory_resume_scan_fixture as scan

_NAMES = (
    "logic.software_contracts.codebase_inventory_lineage", "logic.software_contracts.codebase_inventory_projection_replay",
    "logic.software_contracts.codebase_inventory_receiving", "logic.software_contracts.codebase_inventory_resume",
    "logic.software_contracts.codebase_inventory_scan", "logic.software_contracts.codebase_inventory_targets",
    "logic.software_contracts.codebase_ir", "logic.software_contracts.codebase_ir_targets", "logic.software_contracts.codebase_source_training",
    "logic.software_contracts.duckdb_ast_store", "logic.software_contracts.semantic_index.snapshot",
    "optimizers.logic_theorem_optimizer.autoencoder_projection_features", "optimizers.logic_theorem_optimizer.autoencoder_runtime_registry",
    "optimizers.logic_theorem_optimizer.codebase_feature_worker", "optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker",
    "optimizers.logic_theorem_optimizer.codebase_inventory_feature_worker", "optimizers.logic_theorem_optimizer.modal_autoencoder_cuda")


def write(path, value):
    raw = value if type(value) is bytes else scan._wire(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": scan._sha(raw)}


def cas(native, value, codec, role):
    raw = scan._wire(value, ascii=codec == "raw")
    wanted = scan._cid(raw, codec)
    path = "private/source-artifacts/" + scan._cas_path(wanted, codec)
    pin = write(native / path, raw)
    return {"role": role, "path": path, "cid": wanted, "codec": codec, **pin,
            "mode": stat.S_IMODE((native / path).stat().st_mode)}


@pytest.fixture
def authored(tmp_path):
    """Construct inert schema-valid historical observations, not processes."""
    namespace = tmp_path / scan.SOURCE_NAME
    native = namespace / "native"
    head = {"schema": "codebase-head@1", "repository_id": "qualification:inventory-resume", "generation": 1,
        "manifest_cid": scan._structured({"manifest": 1}), "snapshot_cid": scan._structured({"snapshot": 1}),
        "receipt_cid": scan._structured({"receipt": 1})}
    head["ast_revision_id"] = "rev:" + head["repository_id"] + ":snapshot:" + head["snapshot_cid"]
    # Read current producer bytes only; never import any native product module.
    names = ["ipfs_datasets_py." + name for name in _NAMES]
    # The producer set is frozen by the independently retained root's protocol.
    implementation = {}
    for name in names:
        path = scan._current_path(name)
        raw = path.read_bytes()
        implementation[name] = {"path": "datasets/" + name.replace(".", "/") + ".py", "bytes": len(raw), "sha256": scan._sha(raw)}
        write(namespace / implementation[name]["path"], raw)
    states = {}
    for role, epochs in (("root", 1), ("child", 2)):
        saved = {"state": {"completed_epochs": epochs, "adam": [{"step": epochs}] * 4, "latent_width": 8},
                 "report": {"inert_fixture": role}, "feature_space": {"columns": list(range(53))}}
        raw = scan._wire(saved); checksum = scan._sha(raw)
        write(native / "private/model-artifacts" / checksum[:2] / checksum, raw)
        states[role] = {"artifact": {"bytes": len(raw), "sha256": checksum}, "state_sha256": scan._sha(scan._wire(saved["state"])),
            "report_sha256": scan._sha(scan._wire(saved["report"])), "completed_epochs": epochs, "adam_steps": [epochs] * 4,
            "latent_width": 8, "feature_columns": 53}
    child = states["child"]
    model = {"version_id": "sha256:" + "c" * 64, "artifact": child["artifact"],
        "artifact_cid": scan._cid((native / "private/model-artifacts" / child["artifact"]["sha256"][:2] / child["artifact"]["sha256"]).read_bytes(), "raw"),
        "latent_width": 8, "feature_columns": 53, "projection_ids": ["contracts", "program"], "projection_widths": {"contracts": 2, "program": 51},
        "contract_sha256": "a" * 64, "feature_space_sha256": "b" * 64, "state_sha256": child["state_sha256"]}
    members = [{"source_key": "raw:" + f"source{i:03d}.py".encode().hex(), "raw_path_hex": f"source{i:03d}.py".encode().hex(),
        "entry_cid": scan._structured({"entry": i})} for i in range(300)]
    root = {"schema": "codebase-inventory-resume-root@1", "head": head, "head_cid": scan._structured(head), "members": members,
        "membership_cid": scan._structured(members), "model": model, "implementation": {"files": {name: row["sha256"] for name, row in implementation.items()}},
        "optimized": True, "limits": {"max_inventory_entries": 512, "page_entries": 32}, "authority": dict(scan._SCAN_AUTHORITY)}
    root["implementation"]["sha256"] = scan._sha(scan._wire(root["implementation"]["files"]))
    root_row = cas(native, root, "dag-json", "root")
    descriptors, pages, transport_rows = [], [], [root_row]
    previous = None
    for number, start in enumerate(range(0, 300, 32), 1):
        end = min(start + 32, 300)
        dispositions = ["inferred"] * (22 if number <= 9 else 7)
        dispositions += (["deferred_budget"] * (9 if number == 1 else 10) if number <= 9 else ["opaque", "opaque", "parse_failed", "unsupported_target", "unsupported_target"])
        if number == 1:
            dispositions += ["unindexed"]
        entries, rows = [], []
        for offset, disposition in enumerate(dispositions):
            digest = hashlib.sha256(str(start + offset).encode()).hexdigest()
            coverage = [{"projection_id": key, "known_atoms": 1, "unknown_atoms": 0} for key in model["projection_ids"]] if disposition == "inferred" else []
            entry = {"member_index": start + offset, "source_key": members[start + offset]["source_key"], "entry_cid": members[start + offset]["entry_cid"],
                "disposition": disposition, "reason": None if disposition == "inferred" else "inert_fixture", "coverage": coverage,
                "target_sha256": digest if disposition == "inferred" else None, "source_digest": digest if disposition == "inferred" else None,
                "inference_index": len(rows) if disposition == "inferred" else None}
            entries.append(entry)
            if disposition == "inferred":
                rows.append({"source_digest": digest, "latent": [0.25] * 8, "reconstructed_projection_features": {"contracts": [0.0] * 2, "program": [0.5] * 51}})
        counts = dict(sorted(scan.Counter(dispositions).items()))
        page = {"schema": "codebase-inventory-resume-page@1", "root_cid": root_row["cid"], "head_cid": root["head_cid"], "membership_cid": root["membership_cid"],
            "model_artifact_cid": model["artifact_cid"], "start": start, "end": end, "total_entries": 300, "previous_page_cid": previous,
            "page_membership_cid": scan._structured(members[start:end]), "entries": entries, "authority": dict(scan._SCAN_AUTHORITY),
            "coverage": {"inventory_entries": end - start, "inferred_rows": len(rows), "dispositions": counts},
            "worker_receipt": {"returncode": 0, "workspace_cleaned": True, "source_execution_attested": False,
                "worker_sha256": implementation["ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"]["sha256"]},
            "inference": {"schema": "native-projection-feature-inference/v1", "training_executed": False, "decoded_formulas_generated": False,
                **{key: model[key] for key in ("contract_sha256", "state_sha256", "feature_space_sha256")}, "rows": rows,
                "coverage": [item for entry in entries for item in entry["coverage"]]}}
        row = cas(native, page, "raw", f"page-{number:02d}"); transport_rows.append(row); pages.append(page)
        descriptors.append({"page_cid": row["cid"], "start": start, "end": end, "membership_cid": page["page_membership_cid"], "inferred_rows": len(rows), "dispositions": counts})
        previous = row["cid"]
    coverage = {"inventory_entries": 300, "pages": 10, "inferred_rows": 205, "dispositions": {"deferred_budget": 89, "inferred": 205, "opaque": 2, "parse_failed": 1, "unindexed": 1, "unsupported_target": 2}}
    completion = {"schema": "codebase-inventory-resume-completion@1", "root_cid": root_row["cid"], "head_cid": root["head_cid"], "membership_cid": root["membership_cid"], "model_artifact_cid": model["artifact_cid"], "pages": descriptors, "coverage": coverage, "authority": dict(scan._SCAN_AUTHORITY)}
    completion_row = cas(native, completion, "dag-json", "completion"); transport_rows.append(completion_row)
    optout = cas(native, {**root, "optimized": False}, "dag-json", "optout-root"); transport_rows.append(optout)
    reference = cas(native, {**pages[0], "root_cid": optout["cid"]}, "raw", "optout-reference-page"); transport_rows.append(reference)
    equivalence = {"optimized_root_cid": root_row["cid"], "optimized_page_cid": descriptors[0]["page_cid"], "reference_root_cid": optout["cid"], "reference_page_cid": reference["cid"], "entries_coverage_and_inference_exact": True, "throughput_qualified": False}
    state = {key: child[key] for key in ("state_sha256", "completed_epochs", "adam_steps", "latent_width", "feature_columns")}
    runs, observations = [], []
    for number in range(1, 5):
        cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": root_row["cid"], "next_offset": number * 64, "previous_page_cid": descriptors[number * 2 - 1]["page_cid"]}
        next_cursor = {**cursor, "next_offset": (number + 1) * 64, "previous_page_cid": descriptors[number * 2 + 1]["page_cid"]} if number < 4 else None
        run = {"run_number": number, "pid": 100 + number, "qualified": True, "complete": number == 4, "root_cid": root_row["cid"], "completion_cid": completion_row["cid"],
            "registry_owner_generation_before": number + 2, "registry_owner_generation_after": number + 2, "post_setup_fit_attempt_count": 0,
            "source_model_owner_preservation": True, "numerical_before": state, "numerical_after": state, "request_cursor": cursor, "next_cursor": next_cursor,
            "pages_created": [row["page_cid"] for row in descriptors[number * 2:number * 2 + 2]], "final_resources": {"active_lease_count": 0, "waiting_request_count": 0}}
        request = {"cursor": cursor, "run_number": number, "root_cid": root_row["cid"], "version_id": model["version_id"]}
        write(native / f"resume-request-{number:02d}.json", request); write(native / f"fresh-process-run-{number:02d}.json", run)
        stdout = scan._wire({"qualified": True, "recorded_seconds": 1.5}) + b"\n"
        write(native / f"fresh-process-{number:02d}.stdout", stdout); write(native / f"fresh-process-{number:02d}.stderr", b"")
        observations.append({**run, "stdout_sha256": scan._sha(stdout), "stderr_sha256": scan._sha(b"")}); runs.append(run)
    positive_fresh = {"pids": [101, 102, 103, 104], "chunks": observations}
    write(native / "fresh-process-resume.json", {"qualified": True, "complete": True, "process_runs": runs, "pids": positive_fresh["pids"], "coverage": coverage, "root_cid": root_row["cid"], "completion_cid": completion_row["cid"]})
    owners = {"registry": {"meta": [[1, "schema", "policy", "resource", "/results/native/private/model-artifacts", 2]]}, "source_head": head, "model_artifacts": []}
    write(native / "owners-before-scans.json", owners)
    write(native / "owners-after-resume.json", {**owners, "registry": {"meta": [owners["registry"]["meta"][0][:5] + [7]]}})
    seed = {"schema": scan.transport.SCHEMA, "qualified": False, "head": head, "child_version_id": model["version_id"], "checkpoint_states": states}
    write(namespace / "source-setup-seed.json", seed)
    result = {"schema": "codebase-inventory-resume-native-qualification@1", "qualified": False, "error_type": "StructuredIdentityError", "unknown_fitting_epochs": False,
        "phases": [{"name": "bind_inventory_native_runtime", "status": "failed"}], "final_resources": {"active_lease_count": 0, "waiting_request_count": 0},
        "known_actual_setup_epochs": 0, "inherited_actual_setup_epochs": 2, "post_setup_fit_attempt_count": 0, "setup_training_attempts": [],
        "scan_root_cid": root_row["cid"], "completed_scan_cid": completion_row["cid"], "head": head, "selected_version_id": model["version_id"], "scan_coverage": coverage,
        "opt_out_equivalence": equivalence, "registry_owner_reopen_transition": {"schema": "inventory-resume-registry-owner-reopen@1", "before": 2, "after": 7, "reopens": 5, "child_generations": [3, 4, 5, 6], "other_owner_fields_unchanged": True}}
    result_pin = write(native / "result.json", result)
    container = {key: True for key in ("container_removed", "host_reservation_released", "container_results_copied", "retained_source_verified_after_execution", "runtime_used_retained_source_copies", "native_module_launched")}
    container["host_resources_after_cleanup"] = {"active_lease_count": 0, "waiting_request_count": 0, "allocated_child_process_slots": 0, "allocated": {"cpu_slots": 0, "memory_mb": 0}}
    container_pin = write(namespace / "container-execution-final.json", container)
    write(native / "scan-root.json", root); write(native / "scan-completion.json", completion); write(native / "opt-out-equivalence.json", equivalence)
    generation = {"execution_attestation": False, "files": []}
    for name, pin in implementation.items():
        locator = "producers/" + name + ".py"
        write(native / locator, (namespace / pin["path"]).read_bytes())
        generation["files"].append({"name": name, "copy": locator, "bytes": pin["bytes"], "sha256": pin["sha256"]})
    write(native / "generation-inputs.json", generation)
    positive = {"schema": "inventory-resume-positive-completed-scan-observation@1", "verified": True, "overall_native_qualification": False, "native_worker_qualified": False,
        "authority": dict(scan._SCAN_AUTHORITY), "transport_artifacts": transport_rows, "implementation_files": implementation, "source_namespace": str(namespace),
        "source_result": result_pin, "source_container": container_pin, "root_cid": root_row["cid"], "completion_cid": completion_row["cid"], "head": head,
        "membership_cid": root["membership_cid"], "model": model, "coverage": coverage, "ordered_pages": descriptors, "full_checkpoint_states": states,
        "numerical_state": state, "fresh_process_resume": positive_fresh, "opt_out_equivalence": equivalence,
        **{name: False for name in ("current_deployment_freshness_attested", "native_owners_opened_by_auditor", "numerical_execution_independently_reperformed", "process_origin_attested")}}
    historical = []
    for path in namespace.rglob("*"):
        if path.is_file():
            historical.append({"path": path.relative_to(namespace).as_posix(), "kind": "file", **{"bytes": path.stat().st_size, "sha256": scan._sha(path.read_bytes())}, "mode": stat.S_IMODE(path.stat().st_mode), "mtime_ns": path.stat().st_mtime_ns})
    audit = {"schema": "inventory-resume-worker-independent-audit@1", "qualified": False, "namespace": str(namespace), "primary_archive": {"preserved": True, "before": {"files": historical}}, "positive_completed_scan": positive}
    audit_path = tmp_path / "audit.json"; write(audit_path, audit)
    return namespace, audit_path


def test_pure_stage_and_existing_cas_materialization(authored, tmp_path):
    namespace, audit = authored
    receipt = scan.stage_closed_scan(namespace, tmp_path / "seed", audit)
    target = tmp_path / "cas"; target.mkdir(); write(target / "unrelated-existing-baseline", b"baseline")
    result = scan.materialize_staged_scan(tmp_path / "seed", target, receipt)
    assert result["copied_cas_objects"] == 14 and result["new_scan_pages"] == result["new_fitting_epochs"] == result["new_registry_reopens"] == 0
    assert len([path for path in target.rglob("b*") if path.is_file()]) == 14 and (target / "unrelated-existing-baseline").read_bytes() == b"baseline"
    assert result["fresh_native_validation_required"] is True and all(flag is False for flag in result["authority"].values())
    assert all(not any(word in row["path"] for word in ("signed", "intent", "launch", "profile")) for row in receipt["copied_members"])


@pytest.mark.parametrize("mutation", ["unclosed", "worker_qualified", "unknown_fit", "wrong_failure", "authority_zero", "duplicate_page", "wrong_head", "wrong_coverage", "generation", "pid", "source_pin", "impl_pin", "checkpoint_state", "reordered_pages", "source_namespace"])
def test_staging_refuses_changed_historical_observations(authored, tmp_path, mutation):
    namespace, audit_path = authored
    audit = json.loads(audit_path.read_text()); positive = audit["positive_completed_scan"]
    if mutation == "unclosed":
        path = namespace / "container-execution-final.json"; value = json.loads(path.read_text()); value["container_removed"] = False; write(path, value)
    elif mutation == "worker_qualified": positive["native_worker_qualified"] = True
    elif mutation == "unknown_fit":
        path = namespace / "native/result.json"; value = json.loads(path.read_text()); value["unknown_fitting_epochs"] = True; write(path, value)
    elif mutation == "wrong_failure":
        path = namespace / "native/result.json"; value = json.loads(path.read_text()); value["phases"][0]["name"] = "receive_complete"; write(path, value)
    elif mutation == "authority_zero": positive["authority"]["proof_authority"] = 0
    elif mutation == "duplicate_page": positive["transport_artifacts"][2] = copy.deepcopy(positive["transport_artifacts"][1])
    elif mutation == "wrong_head": positive["head"]["generation"] = 2
    elif mutation == "wrong_coverage": positive["coverage"]["inferred_rows"] = 204
    elif mutation == "generation": positive["fresh_process_resume"]["chunks"][1]["registry_owner_generation_after"] = 3
    elif mutation == "pid": positive["fresh_process_resume"]["pids"][1] = positive["fresh_process_resume"]["pids"][0]
    elif mutation == "source_pin": positive["source_result"]["sha256"] = "0" * 64
    elif mutation == "impl_pin": next(iter(positive["implementation_files"].values()))["sha256"] = "0" * 64
    elif mutation == "checkpoint_state": positive["full_checkpoint_states"]["child"]["adam_steps"][0] = 1
    elif mutation == "reordered_pages": positive["transport_artifacts"][1]["role"], positive["transport_artifacts"][2]["role"] = positive["transport_artifacts"][2]["role"], positive["transport_artifacts"][1]["role"]
    else: positive["source_namespace"] = str(tmp_path)
    write(audit_path, audit)
    with pytest.raises((scan.ClosedScanError, scan.transport.ClosedSetupError)):
        scan.stage_closed_scan(namespace, tmp_path / "seed", audit_path)


@pytest.mark.parametrize("mutation", ["extra", "missing", "bytes", "symlink", "hardlink", "receipt_state", "receipt_coverage", "receipt_authority", "receipt_summary", "cas_overwrite"])
def test_materialization_refuses_staged_and_receipt_changes(authored, tmp_path, mutation):
    namespace, audit = authored; seed = tmp_path / "seed"
    receipt = scan.stage_closed_scan(namespace, seed, audit)
    target = tmp_path / "cas"; target.mkdir()
    first = next(row for row in receipt["copied_members"] if row["kind"] == "cas"); path = seed / first["path"]
    if mutation == "extra": write(seed / "unsigned-launch.json", b"extra")
    elif mutation == "missing": path.unlink()
    elif mutation == "bytes": path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "symlink": path.unlink(); path.symlink_to(audit)
    elif mutation == "hardlink": os.link(path, tmp_path / "alias")
    elif mutation == "receipt_state": receipt["checkpoint_states"]["child"]["adam_steps"][0] = 0
    elif mutation == "receipt_coverage": receipt["coverage"]["inferred_rows"] = 204
    elif mutation == "receipt_authority": receipt["authority"]["proof_authority"] = 0
    elif mutation == "receipt_summary": receipt["source_scan_summary"]["fresh_process_resume"]["pids"][0] = 1
    else: write(target / scan._cas_path(first["cid"], first["codec"]), b"old")
    with pytest.raises((scan.ClosedScanError, scan.transport.ClosedSetupError)):
        scan.materialize_staged_scan(seed, target, receipt)


def test_bound_preflight_precedes_cas_bodies(authored, tmp_path, monkeypatch):
    namespace, audit = authored
    monkeypatch.setattr(scan, "MAX_BYTES", 1)
    calls = []
    original = scan.transport._read
    def counted(*args, **kwargs):
        calls.append(args[0]); return original(*args, **kwargs)
    monkeypatch.setattr(scan.transport, "_read", counted)
    with pytest.raises(scan.ClosedScanError, match="before body read"):
        scan.stage_closed_scan(namespace, tmp_path / "seed", audit)
    assert calls == []


def test_late_source_mutation_is_refused(authored, tmp_path, monkeypatch):
    namespace, audit = authored
    original = scan._copy_selected
    def changed(*args, **kwargs):
        original(*args, **kwargs)
        path = namespace / "native/fresh-process-01.stdout"
        path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(scan, "_copy_selected", changed)
    with pytest.raises(scan.ClosedScanError, match="changed after read"):
        scan.stage_closed_scan(namespace, tmp_path / "seed", audit)


def test_destination_overlap_is_refused(authored, tmp_path):
    namespace, audit = authored
    with pytest.raises(scan.ClosedScanError, match="overlap"):
        scan.stage_closed_scan(namespace, namespace / "seed", audit)
