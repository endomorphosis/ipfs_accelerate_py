"""Independent stdlib receiving of a closed complete successor scan.

The frozen earlier reader supplies strict CID, byte and native schema checks.
This additive reader closes the ten-page chain, four fresh-process receipts,
the common scheduler binding and stored evaluation role/source joins. It
opens no native owner or SQL connection and executes no Git, model or solver.
Recorded execution is an observation, not process-origin attestation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

_OLD_READER = Path(__file__).resolve().with_name("audit_codebase_source_successor.py")
_OLD_SHA256 = "ada264cb162b4d20751e36c80d5504348a5439b05d04ad2469ee41450ea0bf14"
if hashlib.sha256(_OLD_READER.read_bytes()).hexdigest() != _OLD_SHA256:
    raise ValueError("frozen earlier stdlib reader generation differs before import")
_spec = importlib.util.spec_from_file_location("frozen_source_successor_stdlib_reader", _OLD_READER)
inert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inert)
SCHEMA = "codebase-full-successor-independent-audit@1"
ROLE_PATHS = {"train": "calc.py", "canary": "canary.py", "tune": "tune.py"}
HELPER_SHA256 = "ee9832a2805bef6ff3db706424286560fd0a59ccdbc66377f11c5d44712e60a0"
EVIDENCE_NAMES = ("previous-manifest.json", "current-manifest.json", "previous-publication-receipt.json",
    "current-publication-receipt.json", "root-training-record.json", "child-training-record.json",
    "source-delta.json", "checkpoint-states-after.json", "owners-before.json", "owners-after-cold.json", "generation-inputs.json")
EXCLUDED_EXPORTS = (("scan-root.json", "default_root"), ("reference-scan-root.json", "reference_root"),
    ("prefix-page.json", "default_prefix_page"), ("reference-prefix-page.json", "reference_prefix_page"),
    ("successor-selection.json", "default_selection"), ("reference-successor-selection.json", "reference_selection"))
MATERIALIZATION_FIELDS = {"schema", "qualified", "source_namespace", "destination", "source_pins",
    "source_archive_inventory_cid", "copied_members", "copied_files", "copied_bytes", "excluded_historical_scan_objects",
    "selected_models", "current_head", "checkpoint_states", "inherited_setup_epochs", "inherited_scan_pages",
    "new_fitting_epochs", "new_scan_pages", "fresh_native_receiving_required", "native_owners_opened", "proof_authority",
    "source_execution_attested", "scan_execution_attested"}
RESULT_FIELDS = set("384d_qualified checkpoint_states child_version_id closed_source_archive_preserved_after_job "
    "cold_receiving_verified complete_scan_qualified completion_cid controls coverage cpu_scan_cuda_visible_devices "
    "cuda_qualified current_head elapsed_seconds_so_far final_resources fresh_process_count fresh_processes "
    "host_configuration_path host_configuration_pin inference_attempts_outside_pages inherited_scan_pages "
    "inherited_setup_epochs kernel_resource_enforcement_claimed materialization model_head_promoted "
    "native_owner_unchanged_except_copy_path_relocation_and_six_opens new_default_scan_pages new_fitting_epochs "
    "new_reference_scan_pages new_scan_pages_created numerical_page_reuse operation_deadline_seconds "
    "opt_out_equivalence original_source_artifacts_preserved owner_open_count parent_version_id phases pid "
    "post_setup_fit_attempt_count previous_head production_default_activated proof_authority qualified "
    "recorded_seconds repository_code_executed root_cid scan_execution_attested scheduler_configuration "
    "scheduler_state_path schema scope selected_producers_unchanged source_delta_cid source_execution_attested "
    "successor_selection_cid worker_dispatch_qualified".split())


def parse_v2_sha1_index(raw):
    """Bounded fixed-fixture Git index decoding; no Git execution.

    Git documents the binary layout at https://git-scm.com/docs/gitformat-index.
    This qualification supports only ordinary SHA-1 v2 indexes with one TREE
    extension. Source/CAS integrity continues to use full SHA256/CID bytes.
    """
    inert.need(type(raw) is bytes and 32 <= len(raw) <= 2 * inert.MIB,
        "bounded exact Git index bytes required")
    signature, version, count = struct.unpack(">4sII", raw[:12])
    inert.need(signature == b"DIRC" and version == 2 and count == 300
        and hashlib.sha1(raw[:-20]).digest() == raw[-20:], "fixed Git v2 SHA-1 index header/checksum differs")
    offset = 12; rows = []
    for _ in range(count):
        inert.need(offset + 63 <= len(raw) - 20, "truncated Git v2 index entry")
        fields = struct.unpack(">10I20sH", raw[offset:offset + 62])
        flags = fields[-1]
        inert.need(flags & 0xF000 == 0 and fields[6] in {0o100644, 0o100755},
            "ordinary stage-zero Git v2 index entries required")
        end = raw.find(b"\0", offset + 62, len(raw) - 20)
        inert.need(end != -1, "unterminated Git v2 index path")
        path = raw[offset + 62:end]
        inert.need(0 < len(path) <= 4094 and flags & 0xFFF == len(path)
            and not path.startswith(b"/") and not path.endswith(b"/")
            and all(part not in {b"", b".", b"..", b".git"} for part in path.split(b"/")),
            "closed ordinary Git v2 index path/flags differ")
        next_offset = offset + ((end - offset + 1 + 7) // 8) * 8
        inert.need(next_offset <= len(raw) - 20 and raw[end:next_offset] == b"\0" * (next_offset - end),
            "Git v2 index entry padding differs")
        rows.append({"path_hex": path.hex(), "ctime_s": fields[0], "ctime_ns": fields[1], "mtime_s": fields[2],
            "mtime_ns": fields[3], "dev": fields[4], "ino": fields[5], "mode": fields[6], "uid": fields[7],
            "gid": fields[8], "size": fields[9], "oid": fields[10].hex(), "flags": flags})
        offset = next_offset
    paths = [bytes.fromhex(row["path_hex"]) for row in rows]
    inert.need(paths == sorted(set(paths)), "Git v2 index paths duplicated/reordered")
    extension = raw[offset:-20]
    inert.need(len(extension) == 35 and extension[:4] == b"TREE"
        and struct.unpack(">I", extension[4:8])[0] == 27 and extension[8:15] == b"\0" + b"300 0\n",
        "fixed Git v2 index TREE extension differs")
    return rows, extension


def verify_copied_git_index(original_raw, current_raw):
    """Allow only copy-related ctime/inode refresh; preserve tracked identities."""
    previous, previous_extension = parse_v2_sha1_index(original_raw)
    current, current_extension = parse_v2_sha1_index(current_raw)
    allowed = {"ctime_s", "ctime_ns", "ino"}
    stable = lambda row: {key: value for key, value in row.items() if key not in allowed}
    inert.need(len(original_raw) == len(current_raw) and previous_extension == current_extension
        and inert.same([stable(row) for row in previous], [stable(row) for row in current]),
        "copied Git index changed tracked identities or fields beyond ctime/inode refresh")
    changed = sum(left != right for left, right in zip(previous, current))
    return {"path": "repository/.git/index", "entries": 300,
        "original": {"bytes": len(original_raw), "sha256": inert.sha(original_raw)},
        "current": {"bytes": len(current_raw), "sha256": inert.sha(current_raw)},
        "changed_stat_cache_entries": changed, "allowed_fields": sorted(allowed),
        "tracked_oid_mode_flags_paths_extensions_preserved": True,
        "git_executed": False, "repository_source_byte_changes_permitted": False}


def verify_materialization(setup, upstream, namespace):
    """Receive the entire copied population against the fixed source inventory."""
    inert.closed(setup, MATERIALIZATION_FIELDS, "complete seed materialization")
    inert.need(setup["schema"] == "source-successor-materialized-setup@1" and setup["qualified"] is True
        and setup["destination"] == str(namespace) and type(setup["inherited_setup_epochs"]) is int
        and setup["inherited_setup_epochs"] == 2
        and all(type(setup[field]) is int and setup[field] == 0 for field in ("inherited_scan_pages", "new_fitting_epochs", "new_scan_pages"))
        and setup["fresh_native_receiving_required"] is True
        and all(setup[field] is False for field in ("native_owners_opened", "proof_authority", "source_execution_attested", "scan_execution_attested")),
        "explicit closed seed transport/accounting differs")
    inert.need(inert.same(setup["selected_models"], upstream["selected_model_identities"])
        and inert.same(setup["current_head"], upstream["source_delta"]["current_head"])
        and setup["source_archive_inventory_cid"] == upstream["archive"]["inventory_cid"], "fixed complete source/model generation differs")
    excluded = []
    for name, key in EXCLUDED_EXPORTS:
        wanted = upstream["native_artifact_cids"][key]
        excluded.append({"export": name, "cid": wanted,
            "path": "private/source-artifacts/" + ("source" if "prefix-page" in name else "structured") + "/" + wanted[:4] + "/" + wanted})
    inert.need(inert.same(setup["excluded_historical_scan_objects"], excluded), "exact six historical scan exclusions differ")
    omitted = {row["path"] for row in excluded}
    originals = {row["path"]: row for row in upstream["archive"]["files"] if row["kind"] == "file"}
    expected = []
    for source_path, row in originals.items():
        allowed = source_path in {"private/source.duckdb", "private/model.duckdb"} or source_path.startswith(
            ("private/source-artifacts/", "private/model-artifacts/", "repository/"))
        if (allowed and source_path not in omitted) or source_path in EVIDENCE_NAMES:
            expected.append({"path": source_path if allowed else "seed-evidence/" + source_path,
                "source_path": source_path, **{key: row[key] for key in ("bytes", "sha256", "mode")}})
    expected.sort(key=lambda row: row["path"])
    ledger = setup["copied_members"]
    inert.need(type(ledger) is list and len(ledger) <= 2048, "bounded complete copied source ledger required")
    for member in ledger:
        inert.closed(member, {"path", "source_path", "bytes", "sha256", "mode"}, "copied immutable source member")
        inert.integer(member["bytes"], 16 * inert.MIB); inert.integer(member["mode"], 0o777)
    inert.need(inert.same(ledger, expected) and type(setup["copied_files"]) is int and setup["copied_files"] == len(expected)
        and type(setup["copied_bytes"]) is int and setup["copied_bytes"] == sum(row["bytes"] for row in expected)
        and setup["copied_bytes"] <= 128 * inert.MIB, "entire copied source/model/evidence population differs")
    return {"copied_files": len(expected), "copied_bytes": setup["copied_bytes"],
        "excluded_historical_scan_objects": excluded, "complete_population_verified": True}


def verify_selected_producers(generation, reader, upstream):
    inert.closed(generation, {"schema", "files", "scope", "execution_attestation"}, "full scan selected producers")
    inert.need(generation["schema"] == "codebase-full-successor-selected-producers@1"
        and generation["scope"] == "listed_local_files_only" and generation["execution_attestation"] is False
        and type(generation["files"]) is list and 1 <= len(generation["files"]) <= 64,
        "bounded new selected producer scope differs")
    producers = {}; names = []
    for row in generation["files"]:
        inert.closed(row, {"name", "path", "copy", "bytes", "sha256"}, "full scan selected producer")
        inert.text(row["name"], 512); inert.integer(row["bytes"], 4 * inert.MIB, 1)
        inert.need(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*", row["name"])
            and type(row["path"]) is str and Path(row["path"]).is_absolute()
            and Path(row["path"]).as_posix() == row["path"]
            and row["copy"] == "producers/" + row["name"] + ".py", "selected producer locator differs")
        source_root = ("/home/barberb/lift_coding/external/ipfs_datasets" if row["name"].startswith("ipfs_datasets_py.") else
            "/home/barberb/lift_coding/external/ipfs_accelerate" if row["name"].startswith("benchmarks.agent_supervisor.container_coding.") else None)
        inert.need(source_root is not None and row["path"] == source_root + "/" + row["name"].replace(".", "/") + ".py",
            "selected producer exact workspace module path differs")
        raw = reader.raw(row["copy"], 4 * inert.MIB)
        inert.need(len(raw) == row["bytes"] and inert.sha(raw) == row["sha256"], "retained full scan producer bytes differ")
        if row["name"] in upstream["selected_producers"]:
            inert.need(row["sha256"] == upstream["selected_producers"][row["name"]], "inherited selected source generation differs")
        names.append(row["name"]); producers[row["name"]] = row["sha256"]
    inert.need(names == sorted(set(names)), "selected producer population duplicated/reordered")
    required = {"ipfs_datasets_py.duckdb_control.autoencoder_registry", "ipfs_datasets_py.duckdb_control.codebase_catalog",
        "ipfs_datasets_py.logic.software_contracts.codebase_resources",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
        "benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture",
        "benchmarks.agent_supervisor.container_coding.qualify_codebase_full_successor"}
    inert.need(required <= set(producers), "complete named native source/catalog/registry/scheduler producers required")
    inert.need(producers["benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture"] == HELPER_SHA256,
        "executed final helper guard generation differs")
    return producers


def verify_shared_host(host, result, original_raw):
    inert.closed(result["host_configuration_pin"], {"bytes", "sha256"}, "shared host source pin")
    inert.integer(result["host_configuration_pin"]["bytes"], inert.MIB, 1)
    inert.need(inert.same(result["host_configuration_pin"], {"bytes": len(original_raw), "sha256": inert.sha(original_raw)})
        and inert.observation_same(host, inert.parse(original_raw)), "shared original host configuration bytes differ")
    inert.closed(host, {"schema", "state_path", "persisted_config", "hardware", "initial_resources", "auto_renew_leases",
        "kernel_enforcement_claimed", "lease_ttl_seconds", "unified_pool_scope"}, "shared host configuration")
    inert.need(host["schema"] == "successor-expansion-host-configuration@1" and host["state_path"] == result["scheduler_state_path"]
        and host["auto_renew_leases"] is True and host["kernel_enforcement_claimed"] is False
        and type(host["lease_ttl_seconds"]) in {int, float} and host["lease_ttl_seconds"] == 120
        and inert.observation_same(host["persisted_config"], result["scheduler_configuration"])
        and host["persisted_config"]["proof_safety_enabled"] is True,
        "full scan/fresh process common native proof-host pool differs")
    config = host["persisted_config"]
    for key, expected in (("total_cpu_slots", 16), ("total_child_process_slots", 16), ("total_memory_mb", 99688),
        ("total_gpu_memory_mb", 99688), ("total_unified_memory_mb", 99688), ("proof_memory_headroom_mb", 24922)):
        inert.need(type(config[key]) is int and config[key] == expected, "exact existing common host capacity differs")
    return {"state_path": host["state_path"], "configuration_pin": result["host_configuration_pin"],
        "shared_pool_bound": True, "kernel_resource_enforcement_claimed": False}


def current_binding(manifest, head, path, *, get_artifact):
    """Reconstruct exact source custody without interpreting source semantics."""
    inert.head_shape(head)
    inert.need(inert.structured(manifest) == head["manifest_cid"]
        and manifest["snapshot"]["snapshot_cid"] == head["snapshot_cid"], "current evaluation complete manifest/head differs")
    entries = [entry for entry in manifest["snapshot"]["entries"] if entry["path"] == path]
    inert.need(len(entries) == 1, "evaluation role path is absent/ambiguous in complete source manifest")
    entry = entries[0]; inert.entry_shape(entry)
    key = "raw:" + entry["raw_path_hex"]
    units = [unit for unit in manifest["units"] if unit["source_key"] == key]
    inert.need(len(units) == 1 and units[0]["entry_cid"] == entry["entry_cid"]
        and units[0]["parse_status"] == "ok" and entry["opaque_reason"] is None, "current evaluation full entry/unit custody differs")
    raw = get_artifact(entry["source_cid"], source=True)
    inert.need(type(raw) is bytes and len(raw) == entry["size_bytes"]
        and inert.cid(raw, source=True) == entry["source_cid"], "current evaluation raw source bytes differ")
    return {"schema": "codebase-ir-feature-source-binding@1", "head": deepcopy(head),
        "repository_id": head["repository_id"], "path": path, "source_key": key,
        "source_revision": "snapshot:" + head["snapshot_cid"], "source_cid": entry["source_cid"],
        "content_sha256": inert.sha(raw), "entry": deepcopy(entry), "unit": deepcopy(units[0]),
        "ast_cid": units[0]["ast_cid"], "authored_contracts": []}


def verify_current_evaluation_bindings(root_checked, child_checked, previous_manifest, current_manifest, *, get_artifact):
    """Close current source role/path and frozen cohort source-byte identities."""
    inert.verify_model_lineage(root_checked, child_checked)
    root, child = root_checked["saved"], child_checked["saved"]
    root_head = root["report"]["codebase_provenance"]["head"]
    def stored_complete(target):
        binding = inert.stored_target_binding(target)
        rows = [row for row in target["validation"] if row.get("validator_id") == "codebase_ir.exact_native_target_replay@1"]
        inert.need(len(rows) == 1 and rows[0]["details"]["authored_contracts"] == [], "exact stored authored contracts required")
        return {**binding, "authored_contracts": []}
    reports = {}
    for label, saved, manifest in (("root", root, previous_manifest), ("child", child, current_manifest)):
        p = saved["report"]["codebase_provenance"]
        selections = [{"contracts": [], "path": path, "role": role} for role, path in ROLE_PATHS.items()]
        selections.sort(key=lambda row: row["path"])
        inert.need(inert.same(p["selections"], selections), "exact current evaluation role/path selection differs")
        expected = [{"role": row["role"], "binding": current_binding(manifest, p["head"], row["path"], get_artifact=get_artifact)}
                    for row in selections if row["role"] != "train"]
        inert.need(inert.same(p["current_evaluation_bindings"], expected), "complete current evaluation source bindings differ")
        for role, field in (("tune", "tuning_targets"), ("canary", "canary_targets"), ("train", "replay_targets")):
            targets = p[field]
            inert.need(len(targets) == 1, "complete single-path frozen role cohort required")
            stored = stored_complete(targets[0])
            parent_binding = current_binding(previous_manifest, root_head, ROLE_PATHS[role], get_artifact=get_artifact)
            inert.need(inert.same(stored, parent_binding), "complete frozen/replay root-head role/source binding differs")
            current = current_binding(manifest, p["head"], ROLE_PATHS[role], get_artifact=get_artifact)
            # The root's frozen evaluation source is unchanged by the successor.
            # Replay is historical training source and is checked separately.
            if role != "train" or label == "root":
                inert.need(stored["source_cid"] == current["source_cid"]
                    and stored["content_sha256"] == current["content_sha256"], "frozen evaluation cohort differs from current role source bytes")
        training = p["training_targets"]
        current_training_binding = current_binding(manifest, p["head"], ROLE_PATHS["train"], get_artifact=get_artifact)
        inert.need(len(training) == (1 if label == "root" else 2)
            and inert.same(stored_complete(training[0]), current_training_binding), "complete current training model-head role/source binding differs")
        if label == "child":
            inert.need(inert.same(stored_complete(training[1]), current_binding(previous_manifest, root_head, ROLE_PATHS["train"], get_artifact=get_artifact)),
                "complete inherited training replay root-head role/source binding differs")
        reports[label] = {"current_evaluation_bindings": expected,
            "current_role_paths": ROLE_PATHS, "stored_training_target_count": len(p["training_targets"])}
    return {"models": reports, "complete_current_evaluation_bindings_verified": True,
        "frozen_cohort_role_source_bytes_verified": True, "targets_independently_relowered": False,
        "source_runtime_semantics_verified": False}


def verify_page(envelope, root, *, start, previous_page_cid):
    inert.closed(envelope, {"artifact_cid", "value"}, "complete page export")
    value, r = envelope["value"], root["value"]
    inert.need(envelope["artifact_cid"] == inert.cid(inert.numerical_wire(value), source=True), "complete page raw CID differs")
    inert.closed(value, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "start", "end",
        "total_entries", "page_membership_cid", "previous_page_cid", "entries", "inference", "worker_receipt", "coverage", "authority"}, "complete page")
    inert.integer(start, 299); end = min(start + 32, 300)
    members = r["members"][start:end]
    inert.need(value["schema"] == "codebase-inventory-resume-page@1" and value["root_cid"] == root["artifact_cid"]
        and value["head_cid"] == r["head_cid"] and value["membership_cid"] == r["membership_cid"]
        and value["model_artifact_cid"] == r["model"]["artifact_cid"]
        and type(value["start"]) is int and value["start"] == start
        and type(value["end"]) is int and value["end"] == end
        and type(value["total_entries"]) is int and value["total_entries"] == 300
        and value["page_membership_cid"] == inert.structured(members)
        and value["previous_page_cid"] == previous_page_cid, "complete ordered page root/source/model/prefix binding differs")
    inert.false_authority(value["authority"])
    entries = value["entries"]
    inert.need(type(entries) is list and len(entries) == len(members), "complete page entry ledger required")
    for ordinal, (entry, member) in enumerate(zip(entries, members), start):
        inert.closed(entry, {"member_index", "source_key", "entry_cid", "disposition", "reason", "target_sha256",
            "source_digest", "coverage", "inference_index"}, "complete page entry")
        inert.need(type(entry["member_index"]) is int and entry["member_index"] == ordinal
            and entry["source_key"] == member["source_key"] and entry["entry_cid"] == member["entry_cid"]
            and entry["disposition"] in {"inferred", "opaque", "unindexed", "parse_failed", "parse_partial", "unsupported_target", "feature_incompatible", "deferred_budget"},
            "complete ordered page member/disposition differs")
        source_disposition = ("opaque" if member["opaque_reason"] is not None else
            {"failed": "parse_failed", "partial": "parse_partial", "unindexed": "unindexed"}.get(member["parse_status"]))
        if source_disposition is None and not member["path"].endswith(".py"):
            source_disposition = "unsupported_target"
        if source_disposition is not None:
            inert.need(entry["disposition"] == source_disposition, "complete page disposition differs from captured source/parse custody")
        inert.need(entry["reason"] is None or (type(entry["reason"]) is str and 0 < len(entry["reason"]) <= 256), "bounded page reason required")
        for field in ("target_sha256", "source_digest"):
            inert.need(entry[field] is None or (type(entry[field]) is str and re.fullmatch(r"[0-9a-f]{64}", entry[field])), "page source/target digest differs")
        inert.need((entry["target_sha256"] is None) == (entry["source_digest"] is None), "page target binding incomplete")
        inert.need(type(entry["coverage"]) is list and len(entry["coverage"]) <= 2, "bounded page projection coverage required")
        ids = []
        for item in entry["coverage"]:
            inert.closed(item, {"projection_id", "known_atoms", "unknown_atoms"}, "page feature coverage")
            inert.text(item["projection_id"], 512); inert.integer(item["known_atoms"]); inert.integer(item["unknown_atoms"])
            ids.append(item["projection_id"])
        inert.need(ids == sorted(set(ids)), "page projection coverage duplicated/reordered")
        inferred = entry["disposition"] == "inferred"
        inert.need(inferred == (entry["inference_index"] is not None), "page inference index/disposition differs")
        if inferred:
            inert.need(entry["target_sha256"] is not None and entry["reason"] is None
                and ids == r["model"]["projection_ids"] and all(row["known_atoms"] > 0 for row in entry["coverage"]), "selected page projection coverage differs")
    counts = dict(sorted(Counter(row["disposition"] for row in entries).items()))
    inferred = [row for row in entries if row["disposition"] == "inferred"]
    inert.need(inert.same(value["coverage"], {"inventory_entries": len(members), "inferred_rows": len(inferred), "dispositions": counts}), "complete page coverage conservation differs")
    model, inference, receipt = r["model"], value["inference"], value["worker_receipt"]
    inert.closed(inference, {"schema", "contract_sha256", "state_sha256", "feature_space_sha256", "rows", "coverage",
        "training_executed", "decoded_formulas_generated", "representation", "qualified", "admitted", "formalized", "promotion_performed"}, "complete page inference")
    inert.need(inference["schema"] == "native-projection-feature-inference/v1"
        and inference["representation"] == "native_compiler_structural_features_not_semantic_text_embeddings"
        and all(inference[field] is False for field in ("training_executed", "decoded_formulas_generated", "qualified", "admitted", "formalized", "promotion_performed"))
        and all(inference[field] == model[field] for field in ("contract_sha256", "state_sha256", "feature_space_sha256"))
        and inert.same(inference["coverage"], [item for entry in inferred for item in entry["coverage"]]), "complete page inference model/coverage/authority differs")
    rows = inference["rows"]
    inert.need(type(rows) is list and len(rows) == len(inferred), "complete numerical row population differs")
    for ordinal, (entry, row) in enumerate(zip(inferred, rows)):
        inert.closed(row, {"source_digest", "latent", "reconstructed_projection_features"}, "complete page numerical row")
        inert.need(type(entry["inference_index"]) is int and entry["inference_index"] == ordinal
            and entry["source_digest"] == row["source_digest"] and type(row["latent"]) is list and len(row["latent"]) == 8
            and type(row["reconstructed_projection_features"]) is dict and set(row["reconstructed_projection_features"]) == set(model["projection_ids"]), "complete numerical source/layout differs")
        numbers = list(row["latent"])
        for projection, block in row["reconstructed_projection_features"].items():
            inert.need(type(block) is list and len(block) == model["projection_widths"][projection], "complete numerical projection width differs")
            numbers.extend(block)
        inert.need(all(type(number) in {int, float} and math.isfinite(number) for number in numbers), "nonfinite/aliased complete numerical value")
    if not inferred:
        inert.need(receipt is None, "zero-row page must have no numerical worker receipt")
        return {"page_cid": envelope["artifact_cid"], "start": start, "end": end,
            "membership_cid": value["page_membership_cid"], "inferred_rows": 0, "dispositions": counts}
    inert.closed(receipt, {"executable_sha256", "worker_sha256", "input_sha256", "output_sha256", "input_bytes", "output_bytes", "elapsed_ms",
        "returncode", "workspace_cleaned", "limits", "memory_enforcement", "source_execution_attested"}, "complete page worker receipt")
    inert.need(type(receipt["returncode"]) is int and receipt["returncode"] == 0 and receipt["workspace_cleaned"] is True
        and receipt["source_execution_attested"] is False and receipt["memory_enforcement"] == "sampled_process_tree_rss_with_possible_overshoot"
        and receipt["worker_sha256"] == r["implementation"]["files"]["ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"], "complete page worker binding/scope differs")
    for field in ("executable_sha256", "worker_sha256", "input_sha256", "output_sha256"):
        inert.need(type(receipt[field]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt[field]), "complete worker SHA differs")
    inert.integer(receipt["elapsed_ms"])
    inert.closed(receipt["limits"], {"resident_memory_bytes", "max_input_bytes", "max_output_bytes"}, "complete worker limits")
    inert.integer(receipt["limits"]["resident_memory_bytes"], 1024 * inert.MIB, 1024 * inert.MIB)
    for field in ("input_bytes", "output_bytes"):
        inert.integer(receipt[field], 32 * inert.MIB, 1)
        inert.need(receipt[field] <= receipt["limits"]["max_" + field] == r["limits"]["max_" + field], "complete worker byte limits differ")
    return {"page_cid": envelope["artifact_cid"], "start": start, "end": end,
        "membership_cid": value["page_membership_cid"], "inferred_rows": len(inferred), "dispositions": counts}


def verify_completion(envelope, root, pages):
    inert.closed(envelope, {"artifact_cid", "value"}, "complete scan export")
    value = envelope["value"]
    inert.need(envelope["artifact_cid"] == inert.structured(value), "complete scan receipt CID differs")
    inert.closed(value, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "pages", "coverage", "authority"}, "complete scan receipt")
    inert.need(value["schema"] == "codebase-inventory-resume-completion@1" and value["root_cid"] == root["artifact_cid"]
        and value["head_cid"] == root["value"]["head_cid"] and value["membership_cid"] == root["value"]["membership_cid"]
        and value["model_artifact_cid"] == root["value"]["model"]["artifact_cid"]
        and type(pages) is list and len(pages) == 10 and len({page["artifact_cid"] for page in pages}) == 10,
        "complete ten-page source/model population differs")
    inert.false_authority(value["authority"])
    previous = None; start = 0; descriptors = []; counts = Counter(); inferred = 0
    for page in pages:
        descriptor = verify_page(page, root, start=start, previous_page_cid=previous)
        descriptors.append(descriptor); counts.update(descriptor["dispositions"]); inferred += descriptor["inferred_rows"]
        previous, start = page["artifact_cid"], descriptor["end"]
    expected = {"inventory_entries": 300, "pages": 10, "inferred_rows": inferred, "dispositions": dict(sorted(counts.items()))}
    inert.need(start == 300 and sum(counts.values()) == 300 and inferred > 0
        and inert.same(value["pages"], descriptors) and inert.same(value["coverage"], expected), "complete ordered scan chain/coverage conservation differs")
    return {"completion_cid": envelope["artifact_cid"], "page_cids": [page["artifact_cid"] for page in pages],
        "coverage": expected, "all300_members_have_explicit_dispositions": True,
        "all300_members_numerically_inferred": inferred == 300, "numerical_execution_independently_reperformed": False}


def verify_opt_out_equivalence(record, default_page, reference_page):
    inert.closed(record, {"entries_coverage_and_inference_exact", "scope", "reference_page_cid", "optimized_page_cid", "throughput_qualified"},
        "full scan opt-out comparison scope")
    expected = {"entries_coverage_and_inference_exact": True, "scope": "first32_ordered_members_only",
        "reference_page_cid": reference_page["artifact_cid"], "optimized_page_cid": default_page["artifact_cid"], "throughput_qualified": False}
    inert.need(inert.same(record, expected) and all(inert.observation_same(default_page["value"][field], reference_page["value"][field])
        for field in ("entries", "coverage", "inference")), "fresh opt-out comparison complete identities/numerical first32/scope differs")
    return expected


def artifact(reader, wanted, *, numerical=False):
    inert.check_cid(wanted, source=numerical)
    raw = reader.cas(wanted, source=numerical) if not numerical else reader.raw(
        "private/source-artifacts/source/" + wanted[:4] + "/" + wanted, 16 * inert.MIB)
    value = inert.parse(raw)
    expected = inert.numerical_wire(value) if numerical else inert.wire(value)
    inert.need(raw == expected and inert.cid(raw, source=numerical) == wanted, "full scan native CAS bytes/canonical identity differ")
    return {"artifact_cid": wanted, "value": value}


def received_export(reader, locator, *, numerical=False):
    envelope = reader.json(locator, 16 * inert.MIB)
    inert.closed(envelope, {"artifact_cid", "value"}, "full scan native export")
    recorded = artifact(reader, envelope["artifact_cid"], numerical=numerical)
    inert.need(inert.observation_same(envelope, recorded), "full scan exported native CAS preimage differs")
    return envelope


def verify_scoped_resources(value, pids):
    inert.closed(value, {"active_lease_count", "waiting_request_count", "owner_pids", "scope",
        "global_active_lease_count", "global_waiting_request_count"}, "scoped host resources")
    inert.need(type(pids) is list and pids == sorted(set(pids)) and all(type(pid) is int and pid > 1 for pid in pids), "complete exact named scan owner PIDs required")
    inert.need(value["scope"] == "named_scan_process_owners_only" and value["owner_pids"] == pids
        and all(type(value[field]) is int and value[field] == 0 for field in ("active_lease_count", "waiting_request_count")),
        "named scan owners retained reservations")
    inert.integer(value["global_active_lease_count"]); inert.integer(value["global_waiting_request_count"])
    return value


def verify_processes(reader, result, root, completion, frozen, helper_sha, helper_path):
    chunks = result["fresh_processes"]
    inert.need(type(chunks) is list and len(chunks) == 4 and type(result["pid"]) is int and result["pid"] > 1,
        "four complete fresh process observations required")
    pids = []; pages = []; cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": root["artifact_cid"],
        "next_offset": 64, "previous_page_cid": completion["value"]["pages"][1]["page_cid"]}
    before = reader.json("owners-before.json")
    generation = before["registry"]["meta"][0][5]
    for number, chunk in enumerate(chunks, 1):
        raw_request = reader.raw(f"successor-resume-request-{number:02d}.json", 2 * inert.MIB)
        request = inert.parse(raw_request)
        stored = reader.json(f"successor-fresh-process-run-{number:02d}.json", 4 * inert.MIB)
        launch = reader.json(f"successor-fresh-process-launch-{number:02d}.json")
        inert.need(inert.observation_same(chunk, stored), "full result fresh chunk receipt differs")
        chunk_fields = {"schema", "qualified", "complete", "pid", "run_number", "root_cid", "version_id", "request_cursor",
            "request_sha256", "helper_sha256", "materialization_receipt_sha256", "scheduler_state_path", "scheduler_configuration",
            "max_pages", "timeout_seconds", "post_setup_fit_attempt_count", "pages_created", "source_execution_attested",
            "proof_authority", "scan_execution_attested", "new_fitting_epochs", "fit_guard_scope", "registry_owner_generation_before",
            "next_cursor", "prefix_tail_cid", "numerical_before", "numerical_after", "registry_owner_generation_after",
            "source_model_owner_preservation", "final_resources", "recorded_seconds"}
        if number == 4:
            chunk_fields.update({"completion_cid", "coverage"})
        inert.closed(chunk, chunk_fields, "complete fresh process receipt")
        inert.closed(request, {"schema", "output", "run_number", "root_cid", "cursor", "version_id", "expected_head",
            "checkpoint_states", "expected_registry_owner_generation", "scheduler_state_path", "scheduler_configuration",
            "timeout_seconds", "max_pages", "materialization_receipt_sha256", "helper_sha256"}, "fresh complete process request")
        inert.need(request["schema"] == "source-successor-fresh-scan-chunk-request@1" and type(request["run_number"]) is int
            and request["run_number"] == number and request["output"] == str(reader.root)
            and type(request["max_pages"]) is int and request["max_pages"] == 2
            and request["root_cid"] == root["artifact_cid"] and inert.same(request["cursor"], cursor)
            and request["version_id"] == root["value"]["model"]["version_id"]
            and inert.same(request["expected_head"], root["value"]["head"])
            and inert.same(request["checkpoint_states"], frozen)
            and type(request["expected_registry_owner_generation"]) is int
            and request["expected_registry_owner_generation"] == generation + number,
            "fresh process request full source/model/prefix/generation differs")
        inert.need(type(request["timeout_seconds"]) in {int, float} and math.isfinite(request["timeout_seconds"])
            and 0 < request["timeout_seconds"] <= 900, "finite recorded fresh chunk deadline required")
        inert.need(request["helper_sha256"] == helper_sha
            and request["materialization_receipt_sha256"] == inert.sha(reader.raw("materialized-successor-setup.json", 2 * inert.MIB))
            and request["scheduler_state_path"] == result["scheduler_state_path"]
            and inert.observation_same(request["scheduler_configuration"], result["scheduler_configuration"]), "fresh process shared host/helper/materialization generation differs")
        pid = chunk["pid"]
        inert.need(type(pid) is int and pid > 1 and pid != result["pid"] and pid not in pids
            and chunk["schema"] == "source-successor-fresh-scan-chunk@1" and chunk["qualified"] is True
            and type(chunk["run_number"]) is int and chunk["run_number"] == number
            and chunk["root_cid"] == request["root_cid"] and chunk["version_id"] == request["version_id"]
            and inert.same(chunk["request_cursor"], cursor) and chunk["request_sha256"] == inert.sha(raw_request)
            and chunk["helper_sha256"] == helper_sha and chunk["materialization_receipt_sha256"] == request["materialization_receipt_sha256"]
            and chunk["scheduler_state_path"] == request["scheduler_state_path"]
            and inert.observation_same(chunk["scheduler_configuration"], request["scheduler_configuration"])
            and chunk["timeout_seconds"] == request["timeout_seconds"]
            and type(chunk["max_pages"]) is int and chunk["max_pages"] == 2,
            "fresh process observed input/PID/full binding differs")
        for field in ("post_setup_fit_attempt_count", "new_fitting_epochs"):
            inert.need(type(chunk[field]) is int and chunk[field] == 0, "fresh process numerical fitting counter differs")
        inert.need(all(chunk[field] is False for field in ("source_execution_attested", "proof_authority", "scan_execution_attested"))
            and chunk["source_model_owner_preservation"] is True
            and inert.same(chunk["numerical_before"], frozen) and inert.same(chunk["numerical_after"], frozen)
            and type(chunk["registry_owner_generation_before"]) is int
            and type(chunk["registry_owner_generation_after"]) is int
            and chunk["registry_owner_generation_before"] == chunk["registry_owner_generation_after"] == generation + number,
            "fresh process frozen owner/state/scope differs")
        inert.need(chunk["fit_guard_scope"] == "three named owner-process training APIs plus exact frozen checkpoints and native owner rows",
            "fresh process fit guard observation scope differs")
        expected_pages = [row["page_cid"] for row in completion["value"]["pages"][number * 2:number * 2 + 2]]
        inert.need(inert.same(chunk["pages_created"], expected_pages) and chunk["prefix_tail_cid"] == expected_pages[-1]
            and chunk["complete"] is (number == 4), "fresh process ordered page population/completion differs")
        for ordinal, wanted in enumerate(expected_pages, 1):
            exported = received_export(reader, f"successor-process-{number:02d}-page-{ordinal:02d}.json", numerical=True)
            inert.need(exported["artifact_cid"] == wanted, "fresh process page export differs from complete chain")
        expected_cursor = None if number == 4 else {"schema": "codebase-inventory-resume-cursor@1", "root_cid": root["artifact_cid"],
            "next_offset": 64 + number * 64, "previous_page_cid": expected_pages[-1]}
        inert.need(inert.same(chunk["next_cursor"], expected_cursor), "fresh process next complete prefix differs")
        if number == 4:
            inert.need(chunk["completion_cid"] == completion["artifact_cid"] and inert.same(chunk["coverage"], completion["value"]["coverage"]), "last fresh process full completion differs")
        verify_scoped_resources(chunk["final_resources"], [pid])
        inert.closed(launch, {"schema", "run_number", "parent_pid", "child_pid", "returncode", "timeout_group_terminated",
            "child_working_directory", "recorded_seconds", "request_sha256", "helper_sha256", "scheduler_state_path"}, "observed fresh process launch")
        inert.need(launch["schema"] == "source-successor-fresh-scan-child-launch@1" and type(launch["returncode"]) is int and launch["returncode"] == 0
            and type(launch["run_number"]) is int and launch["run_number"] == number
            and type(launch["parent_pid"]) is int and launch["parent_pid"] == result["pid"]
            and type(launch["child_pid"]) is int and launch["child_pid"] == pid
            and launch["child_working_directory"] == str(Path(helper_path).parent.parent.parent.parent)
            and launch["timeout_group_terminated"] is False and launch["request_sha256"] == inert.sha(raw_request)
            and launch["helper_sha256"] == helper_sha and launch["scheduler_state_path"] == result["scheduler_state_path"],
            "actual child launch observation differs")
        inert.need(type(chunk["recorded_seconds"]) in {int, float} and math.isfinite(chunk["recorded_seconds"]) and chunk["recorded_seconds"] >= 0
            and type(launch["recorded_seconds"]) in {int, float} and math.isfinite(launch["recorded_seconds"]) and launch["recorded_seconds"] >= chunk["recorded_seconds"],
            "actual fresh process cost observation differs")
        cursor = expected_cursor; pids.append(pid); pages.extend(expected_pages)
    inert.need(len(set(pages)) == 8 and cursor is None, "complete eight fresh process pages required")
    verify_scoped_resources(result["final_resources"], sorted([result["pid"], *pids]))
    return {"pids": pids, "pages_created": pages, "fresh_processes": 4, "source_model_owner_preservation": True,
        "shared_scheduler_state_path": result["scheduler_state_path"], "process_origin_independently_attested": False}


def verify_costs(result):
    inert.closed(result, RESULT_FIELDS, "complete native result")
    inert.need(result["schema"] == "codebase-full-successor-native-qualification@1" and result["qualified"] is True
        and result["complete_scan_qualified"] is True
        and result["scope"] == "complete_300_member_successor_cpu8d_scan_and_first32_opt_out_comparison",
        "a qualified complete native successor scan is required")
    for field, expected in (("new_fitting_epochs", 0), ("inherited_setup_epochs", 2), ("inherited_scan_pages", 0),
        ("new_default_scan_pages", 10), ("new_reference_scan_pages", 1), ("new_scan_pages_created", 11),
        ("post_setup_fit_attempt_count", 0), ("inference_attempts_outside_pages", 0), ("fresh_process_count", 4), ("owner_open_count", 6)):
        inert.need(type(result[field]) is int and result[field] == expected, "exact actual full scan accounting differs: " + field)
    for field in ("worker_dispatch_qualified", "cuda_qualified", "384d_qualified", "proof_authority", "source_execution_attested",
        "scan_execution_attested", "production_default_activated", "repository_code_executed", "kernel_resource_enforcement_claimed",
        "numerical_page_reuse", "model_head_promoted"):
        inert.need(result[field] is False, "complete scan acquired unreviewed authority: " + field)
    for field in ("cold_receiving_verified", "original_source_artifacts_preserved", "selected_producers_unchanged",
        "native_owner_unchanged_except_copy_path_relocation_and_six_opens", "closed_source_archive_preserved_after_job"):
        inert.need(result[field] is True, "complete recorded native receiving closure differs: " + field)
    inert.need(result["cpu_scan_cuda_visible_devices"] == "", "CPU scan qualification must record CUDA visibility disabled")
    for field in ("recorded_seconds", "elapsed_seconds_so_far"):
        inert.need(type(result[field]) in {int, float} and math.isfinite(result[field]) and result[field] >= 0, "finite complete native actual cost required")
    phases = result["phases"]
    inert.need(type(phases) is list and 1 <= len(phases) <= 64, "bounded full native phase ledger required")
    for phase in phases:
        inert.closed(phase, {"name", "status", "elapsed_seconds"}, "complete native phase")
        inert.need(phase["status"] == "completed" and type(phase["elapsed_seconds"]) in {int, float}
            and math.isfinite(phase["elapsed_seconds"]) and phase["elapsed_seconds"] >= 0, "complete native phase actual cost differs")
    inert.need(type(result["controls"]) is list and len(result["controls"]) == 2, "both actual refusal controls required")
    for control in result["controls"]:
        inert.closed(control, {"name", "refused", "error_type", "error", "elapsed_seconds"}, "actual complete scan refusal")
        inert.text(control["error_type"], 256); inert.text(control["error"], 2048)
        inert.need(control["refused"] is True and type(control["elapsed_seconds"]) in {int, float}
            and math.isfinite(control["elapsed_seconds"]) and control["elapsed_seconds"] >= 0, "actual refusal observation differs")
    inert.need([row["name"] for row in result["controls"]] == ["incomplete_prefix_is_unknown", "precancelled_complete_receiving"],
        "both actual complete/incomplete cancellation controls required")
    deadlines = result["operation_deadline_seconds"]
    inert.closed(deadlines, {"default_selection", "reference_selection", "inference_page", "completion_receiving", "fresh_chunk", "overall", "reference_override_is_qualification_only"}, "full scan qualification deadlines")
    inert.need(all(type(deadlines[field]) in {int, float} and deadlines[field] == amount for field, amount in
        (("default_selection", 120), ("reference_selection", 600), ("inference_page", 600), ("completion_receiving", 600), ("fresh_chunk", 900)))
        and type(deadlines["overall"]) in {int, float} and 0 < deadlines["overall"] <= 7200
        and deadlines["reference_override_is_qualification_only"] is True, "explicit bounded full qualification deadlines differ")
    return {"recorded_native_seconds": result["recorded_seconds"], "new_fitting_epochs": 0,
        "inherited_setup_epochs": 2, "new_default_scan_pages": 10, "new_reference_scan_pages": 1,
        "new_scan_pages_created": 11, "fresh_process_count": 4, "owner_open_count": 6,
        "operation_deadline_seconds": deadlines, "actual_controls": result["controls"], "phases": phases}


def verify_owners(before, after, setup, source_relocation, registry_relocation):
    generation = before["registry"]["meta"][0][5]
    expected = deepcopy(before); expected["registry"]["meta"][0][5] += 5
    inert.need(inert.observation_same(after, expected) and type(generation) is int and generation == 3,
        "source/model rows differ beyond four fresh opens plus one cold reopen")
    for receipt, schema, suffix in ((source_relocation, "source-successor-copied-source-catalog-relocation@1", "source-artifacts"),
        (registry_relocation, "source-successor-copied-registry-relocation@1", "model-artifacts")):
        common = {"schema", "qualified", "output", "materialization_receipt_sha256", "old_artifact_root", "new_artifact_root",
            "new_fitting_epochs", "old_native_owners_opened", "proof_authority"}
        extra = ({"source_before_sha256", "source_after_sha256", "ast_tables", "catalog_before", "catalog_after", "current_head",
            "all13_ast_table_rows_preserved", "all_catalog_rows_preserved_except_local_artifact_root", "native_source_publication_performed"}
            if suffix == "source-artifacts" else {"owner_generation_before", "owner_generation_after", "owner_generation_advanced",
                "version_rows_preserved", "other_registry_rows_preserved", "model_artifacts_changed", "registry_before_sha256", "registry_after_sha256"})
        inert.closed(receipt, common | extra, "copied owner relocation receipt")
        inert.need(receipt["schema"] == schema and receipt["qualified"] is True
            and receipt["output"] == setup["destination"] and receipt["new_fitting_epochs"] == 0
            and type(receipt["new_fitting_epochs"]) is int and receipt["old_native_owners_opened"] is False
            and receipt["proof_authority"] is False
            and receipt["old_artifact_root"] == setup["source_namespace"] + "/private/" + suffix
            and receipt["new_artifact_root"] == setup["destination"] + "/private/" + suffix,
            "copied local store relocation binding differs")
    catalog_before, catalog_after = source_relocation["catalog_before"], source_relocation["catalog_after"]
    expected_catalog = deepcopy(catalog_before); expected_catalog["meta"][0][4] = source_relocation["new_artifact_root"]
    inert.need(inert.same(catalog_after, expected_catalog)
        and source_relocation["catalog_before"]["meta"][0][4] == source_relocation["old_artifact_root"]
        and source_relocation["all13_ast_table_rows_preserved"] is True
        and source_relocation["all_catalog_rows_preserved_except_local_artifact_root"] is True
        and source_relocation["native_source_publication_performed"] is False
        and inert.same(source_relocation["ast_tables"], before["source"]["tables"])
        and set(source_relocation["ast_tables"]) == inert.TABLE_NAMES
        and inert.same(source_relocation["current_head"], before["source"]["current_head"]),
        "complete copied source catalog/13 AST rows relocation differs")
    inert.need(source_relocation["source_before_sha256"] == inert.sha(inert.numerical_wire({"ast": source_relocation["ast_tables"], "catalog": catalog_before}))
        and source_relocation["source_after_sha256"] == inert.sha(inert.numerical_wire({"ast": source_relocation["ast_tables"], "catalog": catalog_after})),
        "copied source full observation digest differs")
    inert.need(type(registry_relocation["owner_generation_before"]) is int and type(registry_relocation["owner_generation_after"]) is int
        and registry_relocation["owner_generation_before"] == registry_relocation["owner_generation_after"] == 2
        and registry_relocation["owner_generation_advanced"] is False
        and registry_relocation["version_rows_preserved"] is True
        and registry_relocation["other_registry_rows_preserved"] is True
        and registry_relocation["model_artifacts_changed"] is False
        and before["registry"]["meta"][0][4] == registry_relocation["new_artifact_root"], "copied registry relocation/generation differs")
    relocated_before = deepcopy(before["registry"])
    relocated_before["meta"][0][4] = registry_relocation["old_artifact_root"]
    relocated_before["meta"][0][5] = 2
    relocated_after = deepcopy(relocated_before); relocated_after["meta"][0][4] = registry_relocation["new_artifact_root"]
    inert.need(registry_relocation["registry_before_sha256"] == inert.sha(inert.numerical_wire(relocated_before))
        and registry_relocation["registry_after_sha256"] == inert.sha(inert.numerical_wire(relocated_after)),
        "complete relocated registry row digest differs")
    return {"source_rows_preserved": True, "model_rows_preserved_except_six_explicit_owner_opens": True,
        "copied_source_and_registry_local_paths_relocated": True, "native_before_generation": generation,
        "native_after_generation": after["registry"]["meta"][0][5], "intrinsic_duckdb_bytes_independently_decoded": False}


def audit_closed_archive(namespace, *, seconds=120):
    started = time.monotonic()
    reader = inert.Reader(namespace, seconds=seconds)
    reader_pin = {"bytes": Path(__file__).stat().st_size, "sha256": inert.sha(Path(__file__).read_bytes())}
    report = {"schema": SCHEMA, "namespace": str(reader.root), "qualified": False, "preserved": False,
        "errors": [], "reader": reader_pin, "frozen_reader": {"bytes": 118630, "sha256": _OLD_SHA256},
        "native_owners_opened": False, "sql_executed": False, "git_executed": False,
        "primary_writes": False, "proof_authority": False, "process_origin_attested": False,
        "numerical_execution_independently_reperformed": False, "targets_independently_relowered": False,
        "worker_dispatch_qualified": False, "cuda_qualified": False, "384d_qualified": False}
    before = reader.whole_archive()
    try:
        inert.need(inert.sha(_OLD_READER.read_bytes()) == _OLD_SHA256, "frozen earlier stdlib reader generation differs")
        result_raw = reader.raw("result.json", 4 * inert.MIB); result = inert.parse(result_raw)
        report["audited_result"] = {"path": "result.json", "bytes": len(result_raw), "sha256": inert.sha(result_raw)}
        report["costs"] = verify_costs(result)
        setup_raw = reader.raw("materialized-successor-setup.json", 2 * inert.MIB); setup = inert.parse(setup_raw)
        source_namespace = "/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02"
        upstream_pins = {"audit": {"bytes": 482926, "sha256": "c096ce302f556af7f1733372de8b54cfb1f56ad23561f7ab78a860e9f0504fea"},
            "guard": {"bytes": 9593, "sha256": "8e239c4d915f49568dfda7a0e94f37c14b964f3677f19f881f02bfaaa5ec85cf"},
            "reader": {"bytes": 118630, "sha256": _OLD_SHA256},
            "native_result": {"bytes": 7410, "sha256": "1a2de8246df5e561ae994f5b77578d69921323e3b18d9c3d0da6ff8d32638f45"}}
        inert.need(setup["source_namespace"] == source_namespace and inert.same(setup["source_pins"], upstream_pins), "fixed source audit/native result/reader guard generation differs")
        audit_path = Path(source_namespace).with_name("source-successor-qualification-20261003-02-readonly-audit-20261003-01.json")
        upstream_raw = inert.Reader(audit_path.parent, seconds=120).raw(audit_path.name, 2 * inert.MIB)
        inert.need(len(upstream_raw) == upstream_pins["audit"]["bytes"] and inert.sha(upstream_raw) == upstream_pins["audit"]["sha256"], "fixed retained source audit bytes differ")
        upstream = inert.parse(upstream_raw)
        report["materialization"] = verify_materialization(setup, upstream, reader.root)
        summary_fields = ("source_namespace", "source_pins", "source_archive_inventory_cid", "copied_files", "copied_bytes",
            "inherited_setup_epochs", "inherited_scan_pages", "new_fitting_epochs")
        inert.need(inert.same(result["materialization"], {key: setup[key] for key in summary_fields}), "native result materialization preimage differs")
        source_rows = {row["path"]: row for row in upstream["archive"]["files"] if row["kind"] == "file"}
        inert.need(setup["source_archive_inventory_cid"] == upstream["archive"]["inventory_cid"], "fixed source archive generation differs")
        ledger_paths = []
        for member in setup["copied_members"]:
            inert.closed(member, {"path", "source_path", "bytes", "sha256", "mode"}, "copied immutable source member")
            original = source_rows[member["source_path"]]
            inert.need(all(member[field] == original[field] for field in ("bytes", "sha256", "mode")), "materialization original source member pin differs")
            expected_path = member["source_path"] if member["source_path"].startswith(("private/", "repository/")) else "seed-evidence/" + member["source_path"]
            inert.need(member["path"] == expected_path, "copied original source locator differs")
            ledger_paths.append(member["path"])
            if member["path"] in {"private/source.duckdb", "private/model.duckdb"}:
                continue  # Only new-copy recorded relocations/native opens change DB bytes.
            raw = reader.raw(member["path"], 16 * inert.MIB)
            if member["path"] == "repository/.git/index":
                original_raw = inert.Reader(source_namespace, seconds=120).raw(member["source_path"], 2 * inert.MIB)
                inert.need(len(original_raw) == member["bytes"] and inert.sha(original_raw) == member["sha256"]
                    and (reader.root / member["path"]).stat().st_mode & 0o777 == member["mode"], "copied original Git index pin/mode differs")
                report["copied_git_index"] = verify_copied_git_index(original_raw, raw)
                continue
            inert.need(len(raw) == member["bytes"] and inert.sha(raw) == member["sha256"]
                and (reader.root / member["path"]).stat().st_mode & 0o777 == member["mode"], "copied frozen source/model/evidence bytes or modes changed")
        inert.need(ledger_paths == sorted(set(ledger_paths)) and type(setup["copied_files"]) is int and setup["copied_files"] == len(ledger_paths)
            and type(setup["copied_bytes"]) is int and setup["copied_bytes"] == sum(row["bytes"] for row in setup["copied_members"]), "complete unique copied source ledger/count differs")
        delta = received_export(reader, "source-delta.json")
        previous_manifest = reader.json("seed-evidence/previous-manifest.json")["value"]
        current_manifest = reader.json("seed-evidence/current-manifest.json")["value"]
        source = inert.verify_source_delta(delta, previous_manifest, current_manifest,
            reader.json("seed-evidence/previous-publication-receipt.json"), reader.json("seed-evidence/current-publication-receipt.json"), get_artifact=reader.cas)
        selection = received_export(reader, "successor-selection.json")
        root = received_export(reader, "scan-root.json")
        inert.verify_selection(selection, delta, root, setup["selected_models"]["root"])
        inert.verify_root(root, delta, setup["selected_models"]["child"])
        models = inert.audit_models(reader, selection, reader.json("seed-evidence/root-training-record.json", 16 * inert.MIB),
            reader.json("seed-evidence/child-training-record.json", 16 * inert.MIB), reader.json("checkpoint-states-after.json"))
        inert.need(inert.same(reader.json("checkpoint-states-before.json"), reader.json("checkpoint-states-after.json"))
            and inert.same(reader.json("checkpoint-states-after.json"), setup["checkpoint_states"]), "complete frozen model state changed")
        report["current_evaluation_closure"] = verify_current_evaluation_bindings(models["root"], models["child"], previous_manifest, current_manifest, get_artifact=reader.cas)
        before_owners, after_owners = reader.json("owners-before.json"), reader.json("owners-after-cold.json")
        source_relocation = reader.json("copied-source-catalog-relocation.json")
        registry_relocation = reader.json("copied-registry-relocation.json")
        inert.need(source_relocation["materialization_receipt_sha256"] == registry_relocation["materialization_receipt_sha256"] == inert.sha(setup_raw),
            "native copied owner relocation materialization generation differs")
        report["registry_closure"] = inert.verify_registry_lineage(before_owners["registry"], selection, models)
        report["owners"] = verify_owners(before_owners, after_owners, setup,
            source_relocation, registry_relocation)
        completion = received_export(reader, "successor-scan-completion.json")
        pages = [artifact(reader, row["page_cid"], numerical=True) for row in completion["value"]["pages"]]
        report["complete_scan"] = verify_completion(completion, root, pages)
        inert.need(result["root_cid"] == root["artifact_cid"] and result["completion_cid"] == completion["artifact_cid"]
            and result["source_delta_cid"] == delta["artifact_cid"] and result["successor_selection_cid"] == selection["artifact_cid"]
            and inert.same(result["coverage"], completion["value"]["coverage"])
            and inert.same(result["current_head"], delta["value"]["current_head"])
            and inert.same(result["previous_head"], delta["value"]["previous_head"])
            and inert.same(result["checkpoint_states"], setup["checkpoint_states"])
            and result["parent_version_id"] == setup["selected_models"]["root"]["version_id"]
            and result["child_version_id"] == setup["selected_models"]["child"]["version_id"], "native full result complete source/root/coverage identities differ")
        for ordinal in (1, 2):
            inert.need(inert.observation_same(received_export(reader, f"successor-parent-page-{ordinal:02d}.json", numerical=True), pages[ordinal - 1]), "fresh parent page export differs from complete chain")
        reference_selection = received_export(reader, "reference-successor-selection.json")
        reference_root = received_export(reader, "reference-scan-root.json")
        inert.verify_selection(reference_selection, delta, reference_root, setup["selected_models"]["root"])
        inert.verify_root(reference_root, delta, setup["selected_models"]["child"])
        inert.need(inert.same(reference_root["value"], {**root["value"], "optimized": False}) and root["value"]["optimized"] is True, "fresh opt-out root differs outside declared profile")
        reference = received_export(reader, "reference-prefix-page.json", numerical=True)
        verify_page(reference, reference_root, start=0, previous_page_cid=None)
        report["opt_out_equivalence"] = verify_opt_out_equivalence(result["opt_out_equivalence"], pages[0], reference)
        generation = reader.json("generation-inputs.json")
        producers = verify_selected_producers(generation, reader, upstream)
        inert.implementation_shape(root["value"]["implementation"], producers)
        inert.implementation_shape(selection["value"]["implementation"], producers)
        inert.implementation_shape(delta["value"]["implementation"], producers)
        inert.implementation_shape(reference_root["value"]["implementation"], producers)
        inert.implementation_shape(reference_selection["value"]["implementation"], producers)
        helper_sha = producers["benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture"]
        helper_path = next(row["path"] for row in generation["files"] if row["name"] == "benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture")
        report["processes"] = verify_processes(reader, result, root, completion, setup["checkpoint_states"], helper_sha, helper_path)
        host = reader.json("shared-host-configuration.json")
        host_path = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/successor-expansion-resources-20261003-01/configuration.json")
        inert.need(result["host_configuration_path"] == str(host_path), "existing shared host namespace differs")
        host_raw = inert.Reader(host_path.parent, seconds=120).raw(host_path.name, inert.MIB)
        report["shared_host"] = verify_shared_host(host, result, host_raw)
        cas_before = reader.json("source-artifacts-before.json")
        expected_cas = [{"path": row["path"].removeprefix("private/source-artifacts/"), **{key: row[key] for key in ("bytes", "sha256")}}
            for row in setup["copied_members"] if row["path"].startswith("private/source-artifacts/")]
        inert.need(inert.same(cas_before, expected_cas), "complete original copied CAS population differs")
        report.update(qualified=True, complete_scan_qualified=True, source_delta=source,
            selected_model_identities=setup["selected_models"], selected_producers=producers,
            default_reference_first32_equal=True, reference_scope="one separately executed first32 page; throughput unqualified")
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        after = reader.whole_archive()
        report["preserved"] = inert.same(before, after)
        report["archive"] = after
        if not report["preserved"]:
            report["qualified"] = False
            report["errors"].append({"type": "ArchiveChanged", "message": "primary archive changed during read-only audit"})
        report["recorded_audit_seconds"] = time.monotonic() - started
        report["guarded_read_bytes"] = reader.total_read_bytes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    result = audit_closed_archive(args.namespace, seconds=args.seconds)
    output = args.output.absolute()
    inert.need(not output.exists() and Path(args.namespace).resolve() not in output.parents, "fresh audit output outside primary namespace required")
    with output.open("xb") as stream:
        stream.write(inert.numerical_wire(result) + b"\n")
    output.chmod(0o444)
    copy = output.with_suffix(".reader.py")
    with copy.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    copy.chmod(0o444)
    print(json.dumps({key: result[key] for key in ("qualified", "preserved", "errors", "recorded_audit_seconds")}, sort_keys=True))
    return 0 if result["qualified"] and result["preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
