"""Receive one closed signed-successor Docker archive from ordinary bytes.

This reader joins retained signed/native/resource observations. It never opens
an owner, SQL, Git, Docker, model or network connection, and grants no current
eligibility or proof authority. Original execution is not independently replayed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import time

PINS = {
    "audit_codebase_signed_successor": "b94b9b28191815f421eb3139f95346e344f3e6ada13f0c280683faf33f750512",
    "audit_codebase_source_successor": "ada264cb162b4d20751e36c80d5504348a5439b05d04ad2469ee41450ea0bf14",
    "audit_codebase_inventory_resume_worker": "db0d93f8321ea0a09a8dd22bc7b42b6195c3ce31e7eb638b5c15d9b7cc111076",
    "audit_codebase_full_successor": "4ad068038facd5877811c3de79d09e5e1c704025305656a764c11ebac10a5c51"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_reader(name):
    path = Path(__file__).with_name(name + ".py")
    need(hashlib.sha256(path.read_bytes()).hexdigest() == PINS[name], "frozen independent reader differs: " + name)
    spec = importlib.util.spec_from_file_location("signed_successor_archive_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for _name, _digest in PINS.items():
    need(hashlib.sha256(Path(__file__).with_name(_name + ".py").read_bytes()).hexdigest() == _digest,
         "frozen reader source differs before loading: " + _name)
worker = load_reader("audit_codebase_signed_successor")
source = load_reader("audit_codebase_source_successor")
full = load_reader("audit_codebase_full_successor")
MIB = 1024**2
SCHEMA = "codebase-signed-successor-closed-archive-audit@1"
FULL_AUDIT_PIN = {"bytes": 497646, "sha256": "43b5b045371bc85851657eeab04382e0f844ae22bf7aa4b0556fd9e22a8a73a1"}
FULL_GUARDS_PIN = {"bytes": 1527, "sha256": "06a59f04397157affd33ba86a554455b7d203b9a5c7d418b63e396ff0f112d2e"}
PHASES = ('materialize_independently_audited_complete_successor','open_only_new_copied_native_owners',
    'receive_full_successor_default_pair_120_then30','initialize_private_signed_owner',
    'sign_fresh_full_successor_task_manifest','admit_full_native_graph','receive_signed_full_admission',
    'transactional_native_full_task_install','claim_native_prerequisite','execute_native_prerequisite_public_check',
    'prepare_public_successor_worker_context','bind_successor_native_runtime','native_worker_start',
    'native_worker_stop_and_uid_cleanup','check_published_check_type.py','check_published_check_offset.py',
    'publish_formatted_source_generation_three','published_source_rejects_old_completion')
LIFECYCLE_FIELDS = {'start','stop','worker_launched','residual_task','task_observations',
    'observed_worker_allocations','remaining_processes','bootstrap_errors','native_diagnostics'}
SCOPE_FIELDS = {'schema','profile','admission_cid','current_inventory','native_population','candidate','head','lease',
    'implementation','task_population_preserved','current_facts','removed_task_cids','inventory_features_are_advisory',
    'future_claim_and_fence','resource_scope','freshness_scope','worker_freshness_scope','task_omission_authority',
    'completion_authority','proof_authority','publication_authority','production_activation'}


def same(left, right):
    return source.numerical_wire(left) == source.numerical_wire(right)


def exact_int(value, expected, message):
    need(type(value) is int and value == expected, message)


def finite_seconds(value, maximum, message):
    need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum, message)


def costs(result):
    need(result["schema"] == "codebase-signed-successor-native-qualification@1"
         and result["qualified"] is True and result["native_worker_qualified"] is True
         and result["full_administrator_population_completed"] is True
         and result["native_source_generation_advanced"] is True,
         "complete actual signed successor qualification required")
    for name, count in (("inherited_setup_epochs", 2), ("inherited_scan_pages", 10),
            ("inherited_reference_pages", 1), ("new_fitting_epochs", 0), ("new_scan_pages", 0),
            ("new_reference_pages", 0), ("post_setup_fit_attempt_count", 0), ("inference_attempt_count", 0), ("provider_calls", 0)):
        exact_int(result[name], count, "exact numerical/scan cost differs: " + name)
    for name in ("proof_authority", "source_execution_attested", "scan_execution_attested", "cuda_qualified",
            "384d_qualified", "production_default_activated", "complete_scan_reexecuted_here"):
        need(result[name] is False, "native observation authority expanded: " + name)
    finite_seconds(result["paired_entry_seconds"], 120, "paired entry exceeded native deadline")
    finite_seconds(result["paired_close_seconds"], 30, "paired close exceeded unchanged START30")
    finite_seconds(result["recorded_seconds"], 3550, "bounded recorded native time required")
    need(type(result['phases']) is list and tuple(row['name'] for row in result['phases'])==PHASES,
         'complete ordered native phase population required')
    for row in result["phases"]:
        need(set(row) == {"name", "status", "elapsed_seconds"} and row["status"] == "completed",
             "actual phase is incomplete")
        finite_seconds(row["elapsed_seconds"], 3550, "finite actual phase time required")
    exact_int(result["operation_deadlines"]["native_start_ms"], 30000, "native START deadline changed")
    exact_int(result["operation_deadlines"]["default_receiving"], 120, "standalone default receiving deadline changed")
    need(type(result["operation_deadlines"]["admission"]) is int
         and 0 < result["operation_deadlines"]["admission"] <= 180, "bounded signed metadata deadline required")
    need(type(result['operation_deadlines']['worker_lifetime']) is int
         and 0 < result['operation_deadlines']['worker_lifetime']<=600, 'bounded permitted native fixture lifetime required')
    refusals = result["controls"]
    need(len(refusals) == 1 and refusals[0]["name"] == "published_source_rejects_old_completion"
         and refusals[0]["refused"] is True and refusals[0]["integrity_refusal_claimed"] is True
         and refusals[0]["error_type"] == "StaleCodebaseError"
         and refusals[0]["error"] == "resume catalog head changed",
         "actual stale scan integrity refusal required")
    reference = result["public_receivers_reference_close"]
    exact_int(reference["deadline_seconds"], 30, "public reference deadline changed")
    need(reference["integrity_refusal_claimed"] is False
         and ((reference["completed"] is True and reference["budget_refused"] is False)
              or (reference["completed"] is False and reference["budget_refused"] is True)),
         "reference budget scope differs")
    return {"new_fitting_epochs": 0, "new_scan_pages": 0, "inherited_setup_epochs": 2,
        "inherited_default_pages": 10, "native_seconds": result["recorded_seconds"],
        "paired_entry_seconds": result["paired_entry_seconds"], "paired_close_seconds": result["paired_close_seconds"],
        "reference": reference}


def container_limits(record, seed):
    for name in ("container_removed", "container_results_copied", "host_reservation_released",
            "retained_source_verified_before_execution", "retained_source_verified_after_execution",
            "closed_source_full_archive_preserved_after_execution"):
        need(record[name] is True, "outer resource/custody closure incomplete: " + name)
    exact_int(record["returncode"], 0, "native command failed")
    changes = record["original_selected_source_changes_after_execution"]
    need(type(changes) is list and all(type(path) is str and Path(path).is_absolute() for path in changes),
         "bounded original source change observations required")
    observed = record["actual_container_limits"]
    need(same(source.parse(observed["inspection_stdout"].encode()), observed["inspection"])
         and same(source.parse(observed["cgroup_stdout"].encode()), observed["cgroup"]), "actual engine/cgroup raw joins differ")
    for name in ("inspection_returncode", "cgroup_returncode"):
        exact_int(observed[name], 0, "actual limit probe failed")
    inspection, cgroup = observed["inspection"], observed["cgroup"]
    need(inspection["Id"] == record["container_id"] and inspection["Name"] == "/" + record["container_name"]
         and same(inspection["HostConfig"],
        {"NanoCpus": 12_000_000_000, "Memory": 8 * 1024**3, "PidsLimit": 512,
         "NetworkMode": "none", "Privileged": False}), "actual container HostConfig differs")
    cpu = cgroup["values"]["cpu.max"].split()
    need(len(cpu) == 2 and all(value.isdecimal() for value in cpu) and int(cpu[1]) > 0
         and int(cpu[0]) == 12 * int(cpu[1]) and cgroup["values"]["memory.max"] == str(8 * 1024**3)
         and cgroup["values"]["pids.max"] == "512", "actual CPU RAM PID cgroup differs")
    exact_int(cgroup["uid"], 1000, "actual cgroup probe owner differs")
    exact_int(cgroup["euid"], 1000, "actual cgroup probe effective owner differs")
    need(cgroup["schema"] == "signed-successor-container-cgroup-observation@1", "actual cgroup schema differs")
    absence = record["container_absence_observation"]
    need(absence["container_id"] == record["container_id"] and absence["container_name"] == record["container_name"]
         and absence["stdout"] == "", "exact disposable container still observed")
    exact_int(absence["returncode"], 0, "live engine absence probe failed")
    transfer = record["container_seed_transfer"]
    value = transfer["observation"]
    need(same(source.parse(transfer["stdout"].encode()), value)
         and value["destination"] == seed["staged_destination"]
         and value["source"] == "/opt/ipfs-supervisor/state/transport-seed"
         and same(value["source_before"], value["source_after"]) and same(value["source_before"], value["destination_after"])
         and value["native_owners_opened"] is False and value["proof_authority"] is False,
         "rootless seed ownership transfer changed bound bytes or locator")
    exact_int(value["owner_uid"], 1000, "private seed owner differs")
    exact_int(value["root_mode"], 0o700, "private seed mode differs")
    exact_int(transfer["returncode"], 0, "seed transfer failed")
    access = record["container_seed_owner_access"]
    need(same(source.parse(access["stdout"].encode()), access["observation"]), "owner access raw observation differs")
    for name in ("uid", "euid", "seed_uid", "seed_gid"):
        exact_int(access["observation"][name], 1000, "seed supervisor ownership/access differs")
    exact_int(access["observation"]["seed_mode"], 0o700, "owner private seed mode differs")
    mounted = {row["Destination"]: row for row in inspection["Mounts"]}
    mount = mounted[value["source"]]
    need(mount["Type"] == "bind" and mount["RW"] is False and mount["Source"] == seed["staged_destination"],
         "actual transport seed mount differs")
    owned = record["host_owned_resources_after_cleanup"]
    worker.inventory.resources(owned)
    need(owned["scope"] == "named_scan_process_owners_only"
         and owned["owner_pids"] == [record["host_reservation"]["owner_pid"]],
         "host cleanup must belong to the exact reservation owner")
    # Other participants can legitimately retain leases in the common pool.
    # The retained global observation is reported, not mistaken for ownership.
    return {"container_id": record["container_id"], "actual_limits": inspection["HostConfig"],
        "cgroup": cgroup, "container_absence": absence, "rootless_private_seed_transfer": value,
        "original_selected_source_changes_observed": changes,
        "host_owned_resources_after_cleanup": owned,
        "host_global_resources_observed_after_cleanup": record["host_resources_after_cleanup"],
        "retained_source_copies_verified_separately": True, "process_origin_attested": False}


def local_pool(authority, record, result, boundary):
    need(set(authority) == {"schema", "state_path", "persisted_config", "auto_renew_leases", "lease_ttl_seconds",
         "parent_host_envelope", "namespace", "scope", "independent_whole_host_pool", "shares_host_pid_state",
         "proof_authority", "initial_resources"}
         and authority["schema"] == "successor-dispatch-container-resource-authority@1"
         and authority["state_path"] == result["scheduler_state_path"] == "/results/container-resource-admission.json"
         and same(authority["persisted_config"], result["scheduler_configuration"])
         and authority["auto_renew_leases"] is True
         and all(authority[key] is False for key in ("independent_whole_host_pool", "shares_host_pid_state", "proof_authority")),
         "bounded container-local pool authority differs")
    finite_seconds(authority["lease_ttl_seconds"], 120, "bounded renewable lease TTL required")
    need(authority["lease_ttl_seconds"] > 0, "positive renewable lease TTL required")
    config = authority["persisted_config"]
    for name, maximum in (("total_cpu_slots", 12), ("total_memory_mb", 8192),
            ("total_child_process_slots", 12), ("total_unified_memory_mb", 8192)):
        need(type(config[name]) is int and 0 < config[name] <= maximum, "local pool exceeds held host envelope")
    need(config["total_gpu_memory_mb"] is None and config["total_unified_memory_mb"] == config["total_memory_mb"]
         and config["proof_safety_enabled"] is True and config["lane_reservations"] == {}, "CPU-only safe local pool required")
    parent = authority["parent_host_envelope"]
    need(parent["container_id"] == record["container_id"]
         and parent["host_scheduler_state_path"] == record["host_scheduler_state_path"] != authority["state_path"]
         and same(parent["host_configuration_pin"], record["host_configuration_pin"])
         and same(authority["namespace"], boundary["namespaces"])
         and same(authority["namespace"], record["actual_container_limits"]["cgroup"]["namespaces"]),
         "local pool belongs to another host or namespace")
    lease = parent["host_reservation"]
    need(lease["lease_id"] == record["host_reservation"]["lease_id"]
         and same(lease, record["host_reservation"])
         and lease["released"] is False and lease["cancelled"] is False and lease["requires_gpu"] is False,
         "parent host reservation was not held")
    for name, count in (("cpu_slots", 12), ("memory_mb", 8192), ("child_process_slots", 12)):
        exact_int(lease[name], count, "host lease does not cover local container")
    for name, count in (("cpu_limit", 12), ("memory_limit_bytes", 8 * 1024**3), ("pids_limit", 512)):
        exact_int(parent[name], count, "parent actual cgroup envelope differs")
    worker.inventory.resources(result["final_resources"])
    worker.inventory.resources(result["resource_after_native_owner_close"])
    for resource in (result["final_resources"], result["resource_after_native_owner_close"]):
        for name in ("global_active_lease_count", "global_waiting_request_count"):
            exact_int(resource[name], 0, "container-local resource participant remains")
    return {"host_state": parent["host_scheduler_state_path"], "local_state": authority["state_path"],
        "host_lease_id": lease["lease_id"], "shared_host_pid_state": False}


def verify_public_checks(checks, reader):
    need(type(checks) is list and len(checks) == 2
         and [Path(row["argv"][-1]).name for row in checks] == ["check_type.py", "check_offset.py"],
         "complete ordered original public checks required")
    for row in checks:
        need(set(row)=={'schema','path','argv','cwd','timeout_seconds','returncode','stdout','stderr',
             'stdout_bytes','stderr_bytes','stdout_sha256','stderr_sha256','source_before','source_after'}
             and row["schema"] == "source-successor-published-public-check@1"
             and row["cwd"] == "/results/native/repository" and row["argv"][1] == "-B"
             and len(row["argv"]) == 3 and row["path"] == row["argv"][-1]
             and row["argv"][-1] in {"check_type.py", "check_offset.py"}, "actual published checker invocation differs")
        exact_int(row["timeout_seconds"], 10, "original public check deadline changed")
        exact_int(row["returncode"], 0, "published original public check failed")
        need(len(row['stdout'].encode())+len(row['stderr'].encode())<=65536,
             'combined public output exceeds actual subprocess bound')
        for role in ("stdout", "stderr"):
            raw = row[role].encode("utf-8")
            need(len(raw) <= 65536 and type(row[role + "_bytes"]) is int
                 and row[role + "_bytes"] == len(raw)
                 and row[role + "_sha256"] == hashlib.sha256(raw).hexdigest(), "public-check raw output identity differs")
        need(same(row["source_before"], row["source_after"]), "public check source changed during execution")
        raw = reader.raw("native/repository/" + row["argv"][-1], 65536)
        need(row["source_after"]["bytes"] == len(raw)
             and row["source_after"]["sha256"] == hashlib.sha256(raw).hexdigest(), "retained original public check source differs")
    return checks


def verify_complete_scan(reader, seed, result, states):
    native = source.Reader(reader.root / "native", seconds=120)
    selection, root, completion, delta = [full.received_export(native, name) for name in
        ("successor-selection.json", "scan-root.json", "scan-completion.json", "source-delta.json")]
    need(selection["artifact_cid"] == seed["selection_cid"] == result["selection_cid"]
         and root["artifact_cid"] == seed["root_cid"] == result["root_cid"]
         and completion["artifact_cid"] == seed["completion_cid"] == result["completion_cid"]
         and delta["artifact_cid"] == seed["source_delta_cid"] == result["source_delta_cid"],
         "actual worker consumed another inherited scan or successor")
    source.verify_selection(selection, delta, root, selection["value"]["previous_model"])
    source.verify_root(root, delta, selection["value"]["model"])
    previous = native.json("closed-full-scan/seed-evidence/previous-manifest.json")["value"]
    current = native.json("closed-full-scan/seed-evidence/current-manifest.json")["value"]
    source.verify_source_delta(delta, previous, current,
        native.json("closed-full-scan/seed-evidence/previous-publication-receipt.json"),
        native.json("closed-full-scan/seed-evidence/current-publication-receipt.json"), get_artifact=native.cas)
    models = source.audit_models(native, selection,
        native.json("closed-full-scan/seed-evidence/root-training-record.json", 16 * MIB),
        native.json("closed-full-scan/seed-evidence/child-training-record.json", 16 * MIB), states)
    evaluation = full.verify_current_evaluation_bindings(models["root"], models["child"], previous, current,
                                                       get_artifact=native.cas)
    pages = [full.artifact(native, row["page_cid"], numerical=True) for row in completion["value"]["pages"]]
    checked = full.verify_completion(completion, root, pages)
    need(same(checked["coverage"], result["scan_coverage"]), "actual scan coverage differs")
    return {"complete_scan": checked, "evaluation_roles": evaluation,
        "selected_models": {label: item["checkpoint_state"] for label, item in models.items()},
        "fresh_native_execution_reperformed": False}


def verify_selected_sources(reader, archive):
    selected = reader.json("selected-source-snapshot.json", 16 * MIB)
    need(set(selected) == {"source", "datasets", "kit"}, "complete three-repository selected snapshot required")
    total = 0
    for package, rows in selected.items():
        need(type(rows) is list and 0 < len(rows) <= 20000, "bounded selected source membership required")
        paths = [row["relative"] for row in rows]
        need(len(paths) == len(set(paths)), "duplicate selected source path")
        observed = {row["path"].removeprefix(package + "/") for row in archive["files"]
                    if row["kind"] == "file" and row["path"].startswith(package + "/")}
        need(set(paths) == observed, "retained selected source membership differs")
        for row in rows:
            raw = reader.raw(package + "/" + row["relative"], 4 * MIB)
            need(type(row["bytes"]) is int and row["bytes"] == len(raw)
                 and row["sha256"] == hashlib.sha256(raw).hexdigest()
                 and reader.path(package + "/" + row["relative"]).stat().st_mode & 0o222 == 0,
                 "retained selected producer bytes or mode differ")
            total += len(raw)
    generation = reader.json("native/generation-inputs.json", 4 * MIB)
    for row in generation["files"]:
        path = Path(row["path"])
        mapped = None
        for package in selected:
            mount = Path("/opt/ipfs-supervisor") / package
            if path.is_relative_to(mount):
                mapped = package + "/" + path.relative_to(mount).as_posix()
                break
        need(mapped is not None, "native selected module is outside copied checkouts")
        retained, deployed = reader.raw("native/" + row["copy"], 4 * MIB), reader.raw(mapped, 4 * MIB)
        need(len(retained) == row["bytes"] and retained == deployed
             and hashlib.sha256(retained).hexdigest() == row["sha256"], "native listed producer differs from deployed copy")
    return {"selected_files": sum(len(rows) for rows in selected.values()), "selected_bytes": total,
        "native_listed_producers": len(generation["files"]), "scope": "listed sequential retained source bodies only",
        "complete_transitive_dependency_attestation": False}


def verify_staged_custody(reader, archive, seed, record):
    raw = reader.raw("setup-seed/staged-successor-dispatch.json", 4 * MIB)
    need(same(source.parse(raw), seed) and raw == source.numerical_wire(seed) + b"\n", "canonical staged receipt bytes differ")
    access = record["container_seed_owner_access"]["observation"]
    need(type(access["manifest_bytes"]) is int and access["manifest_bytes"] == len(raw)
         and access["manifest_sha256"] == hashlib.sha256(raw).hexdigest(), "actual supervisor read another seed manifest")
    expected_files = {"staged-successor-dispatch.json", "independent-full-scan-audit.json", "independent-full-scan-reader-controls.json"}
    paths = [row["path"] for row in seed["copied_members"]]
    need(paths == sorted(set(paths)) and type(seed["copied_files"]) is int and seed["copied_files"] == len(paths)
         and type(seed["copied_bytes"]) is int and seed["copied_bytes"] == sum(row["bytes"] for row in seed["copied_members"]),
         "complete ordered staged seed inventory required")
    expected_files.update(paths)
    observed = {row["path"].removeprefix("setup-seed/"): row for row in archive["files"]
                if row["kind"] == "file" and row["path"].startswith("setup-seed/")}
    need(set(observed) == expected_files, "staged seed contains foreign or missing files")
    for row in seed["copied_members"]:
        item = observed[row["path"]]
        need(all(same(item[field], row[field]) for field in ("bytes", "sha256", "mode")), "immutable staged source/model/evidence byte or mode changed")
    audits = {}
    for field, filename, expected in (("audit", "independent-full-scan-audit.json", FULL_AUDIT_PIN),
            ("reader_controls", "independent-full-scan-reader-controls.json", FULL_GUARDS_PIN)):
        body = reader.raw("setup-seed/" + filename, 4 * MIB)
        need(same(seed[field], expected)
             and same(seed[field], {"bytes":len(body), "sha256":hashlib.sha256(body).hexdigest()}), "staged independent evidence raw pin differs")
        audits[field] = source.parse(body)
    audit, guards = audits["audit"], audits["reader_controls"]
    need(audit["qualified"] is True and audit["preserved"] is True and audit["errors"] == []
         and audit["complete_scan_qualified"] is True and audit["namespace"] == seed["source_namespace"]
         and audit["archive"]["inventory_cid"] == seed["source_archive_inventory_cid"]
         and same({key:audit["reader"][key] for key in ("bytes","sha256")}, seed["reader"])
         and seed["reader"]["sha256"] == PINS["audit_codebase_full_successor"], "closed fullscan audit generation differs")
    need(guards["qualified"] is True and guards["current_pins_unchanged"] is True
         and all(type(guards[key]) is int and guards[key] == 0 for key in ("failures","errors","skipped"))
         and guards["tests"] == 79, "exact fullscan reader guard generation differs")
    native_raw = reader.raw("setup-seed/closed-full-scan/result.json", 4 * MIB)
    need(same(seed["native_result"], {"bytes":len(native_raw),"sha256":hashlib.sha256(native_raw).hexdigest()})
         and same(seed["native_result"], {key:audit["audited_result"][key] for key in ("bytes","sha256")}),
         "inherited fullscan native result raw generation differs")
    return {"staged_files":len(observed),"staged_copied_bytes":seed["copied_bytes"],
        "fullscan_inventory_cid":seed["source_archive_inventory_cid"], "fullscan_reader_sha256":seed["reader"]["sha256"],
        "original_closed_namespace_opened":False, "native_execution_reperformed":False}


SCOPE_MODULES = {
    'ipfs_accelerate_py.agent_supervisor.runtime.' + name for name in
    ('codebase_inventory_evidence_admission', 'local_planning_admission', 'candidate_execution',
     'local_completion_bridge', 'router_public_instruction', 'codebase_inventory_evidence_worker_context',
     'codebase_successor_dispatch_admission', 'codebase_successor_dispatch_context', 'codebase_inventory_execution')
} | {'ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime',
     'ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner',
     'ipfs_datasets_py.logic.software_contracts.codebase_inventory_receiving',
     'ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_receiving'}


def verify_execution_scope(scope, binding, reader, seed, prepared):
    admission=reader.json('native/admission.json',16*MIB)
    manifest=reader.json('native/signed-manifest.json',16*MIB)
    received=reader.json('native/current-admission-verification.json',4*MIB)
    need(same(manifest,admission['manifest']) and scope['schema']=='supervisor-codebase-inventory-execution-scope@1'
         and scope['profile']=='codebase-inventory-one-ready-native-worker@1'
         and scope['admission_cid']==worker.structured(admission)
         and same(scope['current_inventory'],received) and same(scope['head'],seed['current_head']),
         'signed execution scope differs from retained current admission')
    for name in ('identity','profile_id'):
        need(binding[name]==manifest['binding'][name]==admission['receipt']['binding'][name],
             'signed execution scope belongs to another author/profile')
    planning=worker.signature(admission['receipt'])
    for name in ('manifest_cid','graph_cid','plan_id','pending_cid','administrator_task_cids'):
        need(same(received[name],planning[name]),'current native admission differs from signed planning: '+name)
    for name in ('root_cid','completion_cid','source_delta_cid'):
        need(received[name]==seed[name],'current native admission differs from scan: '+name)
    need(received['successor_selection_cid']==seed['selection_cid']
         and received['codebase_successor_context_cid']==prepared['codebase_successor_context_cid']
         and received['codebase_inventory_context_cid']==prepared['codebase_inventory_context_cid']
         and same(received['head'],seed['current_head']) and same(received['previous_head'],seed['previous_head']),
         'current native admission differs from selected source/model context')
    candidate=reader.json('native/candidate.json',4*MIB)
    need(set(scope['candidate'])==set(candidate)|{'implementation_command'}
         and all(same(scope['candidate'][name],value) for name,value in candidate.items())
         and scope['candidate']['implementation_command']==shlex.join(candidate['argv']),
         'signed scope differs from literal authored worker command')
    need(set(scope['implementation'])==SCOPE_MODULES,'complete listed execution producers required')
    for name,digest in scope['implementation'].items():
        package='datasets' if name.startswith('ipfs_datasets_py.') else 'source'
        raw=reader.raw(package+'/'+name.replace('.','/')+'.py',4*MIB)
        need(type(digest) is str and hashlib.sha256(raw).hexdigest()==digest,
             'signed execution producer differs from retained source: '+name)
    return {'admission_cid':scope['admission_cid'],'current_admission_joined':True,
            'listed_execution_producers':len(SCOPE_MODULES),'process_origin_attested':False}


def verify_native_population(population, admission, reader, prepared):
    need(set(population)=={'tasks','completed_prerequisites','completion_rows','authority_rows',
         'selected_task_cids','owner_identity','execution_route_policy'},'complete native population observation required')
    graph=admission['graph'];manifest=admission['manifest']['payload'];planning=worker.signature(admission['receipt'])
    expected={row['task_key']:row for row in graph['tasks']}
    need(set(expected)=={'SUCCESSOR-TYPE','SUCCESSOR-FORMAT'},'original two-task graph required')
    ids={name:worker.prompt_task_record_cid(row) for name,row in expected.items()}
    tasks={row['task_alias']:row for row in population['tasks']}
    need(len(population['tasks'])==2 and set(tasks)==set(ids)
         and population['selected_task_cids']==[ids['SUCCESSOR-FORMAT']]==[prepared['task_cid']],
         'native signed scope omitted or selected another task')
    for name,row in tasks.items():
        task=expected[name]
        need(row['task_cid']==ids[name] and row['goal_cid']==task['goal_cid'] and row['plan_cid']==planning['plan_id']
             and same(row['dependencies'],task['dependency_task_cids']) and type(row['revision']) is int
             and ((name=='SUCCESSOR-TYPE' and row['status']=='completed' and row['revision']>=2)
               or (name=='SUCCESSOR-FORMAT' and row['status']=='ready' and row['revision']>=1)),
             'native task status/dependency/identity differs from signed graph')
        envelope=row['body']['local_planning_contract'];contract=worker.signature(envelope)
        wanted={'schema':'supervisor-local-pending-completion@1','task_cid':ids[name],'task_key':name,
            'manifest':admission['manifest'],'manifest_cid':planning['manifest_cid'],'plan_id':planning['plan_id'],
            'graph_cid':planning['graph_cid'],'pending_requirements':planning['pending_requirements'],
            'pending_cid':planning['pending_cid'],'dependencies':task['dependency_task_cids'],
            'task_spec':next(spec for spec in manifest['tasks'] if spec['task_key']==name),
            'planning_receipt_cid':worker.structured(admission['receipt']),'intent_owner_id':'intent-repository:local'}
        need(same(contract,wanted) and row['identity']['local_contract_cid']==worker.structured(envelope)
             and all(envelope['binding'][key]==admission['manifest']['binding'][key] for key in ('identity','profile_id')),
             'native pending task contract differs from complete signed obligations')
        spec=wanted['task_spec']
        relations={
            'outputs':[{'ordinal':n,'path':item['path'],'effect':item} for n,item in enumerate(spec['outputs'])],
            'acceptance':[{'ordinal':n,'criterion':item['criterion'],'evidence_policy':item}
                          for n,item in enumerate(spec['acceptance'])],
            'validations':[{'ordinal':n,'argv':item['argv'],'policy':{key:value for key,value in item.items()
                          if key not in {'argv','validation_commands','command'}}}
                          for n,item in enumerate(spec['validations'])]}
        need(all(same(row[name],value) for name,value in relations.items()),
             'native detached public requirements differ from signed task specification')
    prerequisite=ids['SUCCESSOR-TYPE'];bindings=population['completed_prerequisites'];rows=population['completion_rows']
    need(set(bindings)==set(rows)=={prerequisite} and len(rows[prerequisite])==1,'exact completed prerequisite population required')
    binding,receipt=bindings[prerequisite],rows[prerequisite][0]
    need(len(receipt)==10 and receipt[0]==binding['completion_receipt_cid'] and receipt[1]==prerequisite
         and receipt[2]==expected['SUCCESSOR-TYPE']['goal_cid'] and receipt[8]==binding['completion_evidence_digest']
         and binding['task_cid']==prerequisite and binding['task_alias']=='SUCCESSOR-TYPE'
         and same(binding['task_revision'],tasks['SUCCESSOR-TYPE']['revision']), 'native prerequisite receipt binding differs')
    validation=reader.json('native/prerequisite-validation.json',4*MIB)
    need(validation['passed'] is True and validation['task_cid']==prerequisite and len(validation['results'])==1
         and validation['results'][0]['validation_key']=='public-type'
         and validation['results'][0]['outcome']=='passed','actual prerequisite public check observation required')
    body=source.parse(receipt[9].encode())
    need(body['schema']=='ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1'
         and same(body['revision'],tasks['SUCCESSOR-TYPE']['revision'])
         and body['evidence_digests']==[validation['results'][0]['evidence_digest']]
         and body['receipt']['evidence_digest']==validation['results'][0]['evidence_digest'],
         'completed prerequisite differs from retained actual public type evidence')
    return {'original_signed_tasks':2,'selected_format_task':prepared['task_cid'],
            'type_prerequisite_observation_joined':True,'native_persistence_requeried':False}


def verify_portal_cleanup_identity(reader, allocation, task_cid, population):
    """Join retained Portal identity bytes to the independently signed task.

    The queue's display alias is descriptive. Its canonical identity is rebuilt
    from the admitted task semantics; the immutable attempt projection then
    binds that work to the original Intent task, claim and fenced owner. This
    is historical byte receiving, not a new task claim or native SQL replay.
    """
    tasks = [row for row in population['tasks'] if row['task_cid'] == task_cid]
    need(len(tasks) == 1, 'cleanup requires one exact admitted signed task')
    task = tasks[0]
    alias = task['task_alias']
    need(alias == 'SUCCESSOR-FORMAT' and allocation['task_id'] == alias,
         'cleanup allocation belongs to another task')
    workspace = Path(allocation['workspace_path'])
    need(allocation['schema'] == 'ipfs_accelerate_py/agent-supervisor/worktree-lifecycle-record@1'
         and allocation['repo_root'] == '/results/native/repository'
         and workspace.is_absolute() and workspace.is_relative_to('/opt/ipfs-supervisor/worktrees')
         and workspace != Path('/opt/ipfs-supervisor/worktrees') and '..' not in workspace.parts,
         'observed allocation escaped its exact native repository/worktree roots')
    state = Path(allocation['state_dir'])
    prefix = Path('/results/native/private/launch/state/run/admitted_database_portal_attempts')
    need(state.parent == prefix and re.fullmatch(r'[0-9a-f]{24}', state.name) is not None,
         'cleanup attempt state escaped its exact owner root')
    relative = 'native/' + state.relative_to('/results/native').as_posix()
    raw_binding = reader.raw(relative + '/database-attempt-binding.json', 4 * MIB)
    raw_projection = reader.raw(relative + '/task-projection.runtime.todo.md', 4 * MIB)
    raw_queue = reader.raw(relative + '/task_queue.json', 4 * MIB)
    binding = source.parse(raw_binding)
    fields = {'schema','interface','attempt_id','claim_id','task_cid','task_alias','goal_cid','plan_cid',
        'task_revision','fencing_token','fence_epoch','lease_id','task_body_digest','projection_seed_digest',
        'projection_immutable_digest','authoritative_task_store','projection_authority','binding_id',
        'control_binding_id','control_task_projection_cid','control_expected_revision','control_portal_binding_basis_cid'}
    need(type(binding) is dict and set(binding) == fields
         and binding['schema'] == 'ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@2'
         and binding['interface'] == 'DatabasePortalExecutionBridge@1'
         and binding['authoritative_task_store'] == 'duckdb' and binding['projection_authority'] is False,
         'closed non-authoritative native attempt binding required')
    for name in fields - {'task_revision','fencing_token','fence_epoch','control_expected_revision','projection_authority'}:
        need(type(binding[name]) is str and bool(binding[name]), 'exact attempt binding text required: ' + name)
    for name in ('task_revision','fencing_token','fence_epoch','control_expected_revision'):
        need(type(binding[name]) is int and 0 < binding[name] <= 2**53,
             'exact positive attempt binding integer required: ' + name)
    unsigned = {key: value for key, value in binding.items() if key != 'binding_id'}
    sha = lambda value: 'sha256:' + hashlib.sha256(value).hexdigest()
    need(binding['binding_id'] == sha(worker.wire(unsigned))
         and state.name == hashlib.sha256(binding['attempt_id'].encode()).hexdigest()[:24]
         and binding['task_cid'] == task_cid and binding['task_alias'] == alias
         and binding['goal_cid'] == task['goal_cid'] and binding['plan_cid'] == task['plan_cid']
         and binding['task_revision'] == binding['control_expected_revision'],
         'attempt binding identity or signed task join differs')
    text = raw_projection.decode('utf-8')
    lines = text.splitlines()
    title = task['body']['title']
    need(type(title) is str and title and '\n' not in title and '\r' not in title
         and lines[:4] == ['# Database attempt projection (non-authoritative)', '', '## ' + alias + ' ' + title, '']
         and text.endswith('\n'), 'one exact signed task projection header required')
    metadata = {}
    for line in lines[4:]:
        need(line.startswith('- ') and ':' in line, 'exact single-line task projection required')
        key, value = line[2:].split(':', 1)
        need(value == '' or value.startswith(' '), 'exact task projection separator required')
        value = value[1:] if value else ''
        need(key not in metadata, 'duplicate task projection field')
        metadata[key] = value
    receipt = source.parse(metadata.get('Completion Receipt', '').encode())
    need(type(receipt) is dict and receipt.get('operation') == 'database_attempt_admitted'
         and receipt.get('claim_phase_schema') == 'ipfs_accelerate_py/agent-supervisor/typed-database-attempt-admission@1',
         'native claim-phase projection receipt required')
    for name in ('attempt_number','claimed_from_revision','admitted_from_revision'):
        need(type(receipt.get(name)) is int and 0 < receipt[name] <= 2**53,
             'exact claim-phase revision or attempt required')
    need(receipt['claimed_from_revision'] == task['revision']
         and receipt['admitted_from_revision'] == task['revision'] + 1
         and binding['task_revision'] == receipt['admitted_from_revision'] + 1,
         'claim-time task revisions differ from admitted task')
    for name in ('attempt_id','claim_id','lease_id','fencing_token','fence_epoch'):
        need(same(receipt.get(name), binding[name]), 'projection claim differs from attempt binding: ' + name)
    body = {**task['body'], 'completion_receipt': receipt}
    need(binding['task_body_digest'] == sha(worker.wire(body)),
         'claim-time task body differs from admitted signed contract')
    route_policy = population['execution_route_policy']
    entries = [row for row in route_policy['entries'] if row['task_cid'] == task_cid]
    need(len(entries) == 1, 'one exact signed task execution route required')
    route = {'schema':'ipfs_accelerate_py/agent-supervisor/task-execution-route-binding@1', **entries[0],
        **{name: route_policy[name] for name in ('plan_root_cid','policy_id','repository_tree_id','source_revision')}}
    need(same(receipt.get('execution_route_binding'), route), 'claim projection execution route changed')
    owner = receipt.get('claim_process_attestation')
    need(type(owner) is dict and owner.get('schema') == 'ipfs_accelerate_py/agent-supervisor/typed-database-claim-process@1'
         and type(owner.get('uid')) is int and owner['uid'] == 1000
         and same(allocation['owner'], {name: owner.get(name) for name in ('boot_id','parent_pid','pid','start_time_ticks')}),
         'cleanup allocation owner differs from native claim process')
    outputs = [row['path'] for row in task['outputs']]
    validations = [shlex.join(row['argv']) for row in task['validations']]
    acceptance = ' ; '.join(row['criterion'] for row in task['acceptance'])
    wanted = {'Status':metadata.get('Status'), 'Completion':'auto', 'Priority':task['priority'], 'Track':'implementation',
        'Depends on':'', 'Outputs':', '.join(outputs), 'Validation':' ; '.join(validations), 'Acceptance':acceptance,
        'Database task CID':task_cid, 'Database attempt ID':binding['attempt_id'], 'Database claim ID':binding['claim_id'],
        'Database attempt number':str(receipt['attempt_number']), 'Database lease ID':binding['lease_id'],
        'Database owner session ID':receipt.get('owner_session_id'), 'Database fencing token':str(binding['fencing_token']),
        'Database fence epoch':str(binding['fence_epoch']), 'Database dependency CIDs':', '.join(task['dependencies']),
        'Projection authority':'false', 'Board namespace':'intent',
        'Local planning contract CID':worker.structured(task['body']['local_planning_contract']),
        'Completion Receipt':worker.wire(receipt).decode(), 'Title':title}
    need(metadata.get('Status') in {'ready','in_progress','completed','complete','done'} and same(metadata, wanted),
         'immutable task projection differs from admitted signed task and claim')
    seed = text.replace('- Status: ' + metadata['Status'], '- Status: ready', 1)
    immutable = re.sub(r'(?mi)^-\s*status\s*:\s*.*$', '- Status: <mutable>', text)
    need(binding['projection_seed_digest'] == sha(seed.encode())
         and binding['projection_immutable_digest'] == sha(immutable.encode()),
         'retained projection seed or immutable bytes differ')
    basis = {'schema':'ipfs_accelerate_py/agent-supervisor/database-portal-binding-basis@1',
        **{name: binding[name] for name in ('task_alias','task_revision','goal_cid','plan_cid','task_body_digest','control_task_projection_cid')}}
    control = {'schema':'ipfs_accelerate_py/agent-supervisor/database-control-claim-binding@2',
        **{name: receipt[name] for name in ('attempt_id','claim_id','attempt_number','lease_id','owner_session_id','fencing_token','fence_epoch')},
        'task_cid':task_cid, 'control_expected_status':'in_progress', 'control_expected_revision':binding['task_revision'],
        'control_task_projection_cid':binding['control_task_projection_cid'],
        'database_portal_binding_basis':basis, 'database_portal_binding_basis_cid':worker.structured(basis)}
    need(binding['control_portal_binding_basis_cid'] == worker.structured(basis)
         and binding['control_binding_id'] == worker.structured(control), 'claim-time control binding bytes differ')
    normalize = lambda value: re.sub(r'\s+', ' ', value).strip().casefold()
    material = {'schema':'ipfs_accelerate_py/agent-supervisor/task-identity@1', 'semantic':{
        'title':normalize(title), 'outputs':sorted(set(outputs)),
        'acceptance':[normalize(value) for value in acceptance.split(',') if normalize(value)]}}
    portal_cid, portal_key = worker.structured(material), 'task/v1/' + hashlib.sha256(worker.wire(material)).hexdigest()
    queue = source.parse(raw_queue)
    need(type(queue) is dict and set(queue) == {'schema','aliases','entries','entry_count','updated_at'}
         and queue['schema'] == 'persistent_task_queue_v3' and type(queue['entry_count']) is int and queue['entry_count'] == 1
         and set(queue['entries']) == {portal_cid} and queue['aliases'] == {alias:portal_cid, 'intent::' + alias:portal_cid},
         'native task queue differs from rebuilt portal identity')
    queued = queue['entries'][portal_cid]
    need(queued['canonical_task_cid'] == portal_cid and queued['canonical_task_key'] == portal_key
         and queued['task_id'] == alias and queued['aliases'] == [alias, 'intent::' + alias]
         and queued['track'] == wanted['Track'] and queued['priority'] == wanted['Priority']
         and queued['provenance'] == [{'board_namespace':'intent','display_task_id':alias,
             'source_path':str(state / 'task-projection.runtime.todo.md')}]
         and type(queued['attempt_count']) is int and 1 <= queued['attempt_count'] < 65536,
         'portal queue identity or exact projection provenance differs')
    need(allocation['canonical_task_cid'] == portal_cid
         and type(allocation['attempt']) is int
         and allocation['attempt'] == ((1 << 52) | (receipt['attempt_number'] << 16) | queued['attempt_count']),
         'cleanup allocation belongs to another portal work or attempt')
    record = {'kind':'worktree-lifecycle-record', **{name: allocation[name] for name in
        ('canonical_task_cid','task_id','attempt','workspace_path')}}
    need(allocation['record_id'] == worker.structured(record), 'cleanup allocation record CID differs')
    return {'signed_task_cid':task_cid, 'portal_task_cid':portal_cid, 'portal_task_key':portal_key,
        'attempt_binding_id':binding['binding_id'], 'projection_authority':False,
        'control_task_projection_replayed':False, 'native_persistence_requeried':False,
        'retained_bytes':{name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in
            (('database-attempt-binding.json',raw_binding),('task-projection.runtime.todo.md',raw_projection),('task_queue.json',raw_queue))}}


def verify_fixture_cleanup(rows, allocations, archive, task_cid, *, reader, population):
    need(type(rows) is list and type(allocations) is list and 0 < len(rows) <= len(allocations) <= 16,
         'retained actual allocation and explicit fixture cleanup required')
    expected={row['record_id']:row for row in allocations}
    need(len(expected)==len(allocations),'duplicate observed allocation')
    identities = {key: verify_portal_cleanup_identity(reader, allocation, task_cid, population)
                  for key, allocation in expected.items()}
    seen=set()
    for row in rows:
        need(set(row)=={'allocation','decision','scope'}
             and row['scope']=='explicit owner fixture cleanup after native STOP', 'fixture cleanup scope differs')
        allocation,decision=row['allocation'],row['decision']
        key=allocation['record_id'];need(key in expected and key not in seen and same(allocation,expected[key])
             and allocation['canonical_task_cid']==identities[key]['portal_task_cid'],
             'fixture cleanup differs from actual worker allocation')
        seen.add(key)
        path=Path(allocation['workspace_path'])
        need(path.is_absolute() and path.is_relative_to('/opt/ipfs-supervisor/worktrees')
             and path!=Path('/opt/ipfs-supervisor/worktrees') and '..' not in path.parts,
             'fixture cleanup allocation escaped owned root')
        need(set(decision)=={'disposition','allowed','reason','failure_kind','provider_call_allowed','attempt_consumed','record'}
             and decision['allowed'] is True and decision['disposition'] in {'allow','reclaim_then_allow'}
             and all(type(decision[name]) is bool for name in ('provider_call_allowed','attempt_consumed'))
             and type(decision['reason']) is str and type(decision['failure_kind']) is str,
             'native lifecycle owner did not authorize fixture removal')
        if decision['record'] is not None:
            need(all(same(decision['record'][name],allocation[name]) for name in
                ('record_id','task_id','canonical_task_cid','owner','lease_id','workspace_path','branch','state_dir')),
                'cleanup lifecycle authority belongs to another allocation')
    skipped = set(expected) - seen
    paired = ('canonical_task_cid','task_id','attempt','owner','branch','lease_id','lane_id','repo_root','state_dir','created_at')
    for key in skipped:
        preparing = expected[key]
        need(preparing['state'] == 'preparing' and type(preparing['fence']) is int
             and any(all(same(preparing[name], expected[removed][name]) for name in paired)
                 and type(expected[removed]['fence']) is int and preparing['fence'] < expected[removed]['fence']
                 and preparing['workspace_path'] != expected[removed]['workspace_path'] for removed in seen),
             'unremoved allocation is not a superseded preparing observation')
    need(not any(row['path']=='native/repository/.git/worktrees'
         or row['path'].startswith('native/repository/.git/worktrees/') for row in archive['files']),
         'linked worktree metadata remains in closed archive')
    return {'explicit_fixture_allocations_removed':len(seen),
            'preparing_observations_without_removal_receipt':len(skipped),
            'preparing_physical_absence_verified':False, 'task_identity_bindings':identities,
            'general_crash_recovery_qualified':False}


def verify_preserved_repository(archive, seed, prior):
    entries = prior['snapshot']['entries']
    paths = [row['path'] for row in entries]
    need(len(paths) == 300 and len(set(paths)) == 300 and paths.count('calc.py') == 1,
         'complete original captured repository population required')
    seed_rows = seed['copied_members']
    need(len({row['path'] for row in seed_rows}) == len(seed_rows), 'staged member identity repeated')
    baseline = {row['path']: row for row in seed_rows}
    retained_rows = archive['files']
    need(len({row['path'] for row in retained_rows}) == len(retained_rows), 'retained member identity repeated')
    retained = {row['path']: row for row in retained_rows}
    for path in paths:
        if path == 'calc.py':
            continue
        original = baseline.get('repository/' + path)
        actual = retained.get('native/repository/' + path)
        need(original is not None and actual is not None and actual['kind'] == 'file',
             'unchanged captured repository member missing')
        need(all(same(actual[name], original[name]) for name in ('bytes', 'sha256', 'mode')),
             'other captured repository member changed bytes or mode')
    return {'unchanged_repository_files': 299,
            'opaque_files_byte_pinned': sum(row['kind'] == 'opaque' for row in entries if row['path'] != 'calc.py'),
            'source_semantics_verified': False}


def audit_closed_archive(namespace, *, seconds=300):
    started = time.monotonic()
    reader = source.Reader(namespace, seconds=seconds)
    before = reader.whole_archive()
    report = {"schema": SCHEMA, "namespace": str(reader.root), "qualified": False,
        "preserved": False, "errors": [], "reader": {"bytes": Path(__file__).stat().st_size,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "frozen_readers": PINS, "native_owners_opened": False, "sql_executed": False,
        "git_executed": False, "docker_executed": False, "network_executed": False,
        "numerical_execution_reperformed": False, "targets_independently_relowered": False,
        "proof_authority": False, "source_semantics_verified": False, "process_origin_attested": False}
    try:
        result = reader.json("native/result.json", 8 * MIB)
        record = reader.json("container-execution-final.json", 4 * MIB)
        seed = reader.json("source-setup-seed.json", 4 * MIB)
        report["costs"] = costs(result)
        report["container"] = container_limits(record, seed)
        report["staged_custody"] = verify_staged_custody(reader, before, seed, record)
        boundary = reader.json("native/container-boundary.json")
        authority = reader.json("native/container-resource-authority.json")
        report["resources"] = local_pool(authority, record, result, boundary)
        need(same(seed, reader.json("native/source-dispatch-seed.json", 4 * MIB)), "actual native seed binding differs")
        prepared = reader.json("native/public-worker-context.json")
        actual = worker.verify_authored_worker_receipt(reader.root / "native", prepared=prepared, task_cid=prepared["task_cid"])
        need(same(actual, reader.json("native/authored-worker-receipt-verification.json", 8 * MIB))
             and same(actual, result["authored_worker_receipt"]), "actual signed public worker receipt differs")
        report["signed_public_worker"] = actual
        need(result["start"]["status"] == result["stop"]["status"] == "succeeded"
             and result["bootstrap_errors"] == [] and result["residual_task"]["status"] == "completed"
             and result["native_task_statuses"] == {"SUCCESSOR-TYPE": "completed", "SUCCESSOR-FORMAT": "completed"},
             "native original task population or lifecycle incomplete")
        exact_int(result["remaining_processes"], 0, "native process remained after STOP")
        lifecycle = reader.json("native/native-lifecycle.json", 8 * MIB)
        need(set(lifecycle)==LIFECYCLE_FIELDS and all(same(lifecycle[name], result[name]) for name in lifecycle),
             "complete retained lifecycle observation differs")
        scope_before, scope_after = reader.json("native/execution-scope-before.json", 8 * MIB), reader.json("native/execution-scope-after-stop.json", 8 * MIB)
        need(same(scope_before, scope_after), "native scope changed across START/STOP")
        scope = worker.signature(scope_before)
        need(set(scope)==SCOPE_FIELDS,'closed actual execution scope required')
        report['signed_execution_scope']=verify_execution_scope(scope,scope_before['binding'],reader,seed,prepared)
        report['native_population']=verify_native_population(scope['native_population'],
            reader.json('native/admission.json',16*MIB),reader,prepared)
        need(scope["current_facts"] == scope["removed_task_cids"] == [] and scope["task_population_preserved"] is True
             and scope["inventory_features_are_advisory"] is True, "signed scope omitted original obligations")
        for name in ("task_omission_authority", "completion_authority", "proof_authority", "publication_authority", "production_activation"):
            need(scope[name] is False, "signed execution scope gained authority")
        for name, count in (("cpu_slots", 4), ("memory_mb", 4096), ("child_process_slots", 8)):
            exact_int(scope["lease"][name], count, "actual held worker lease differs")
        need(scope["candidate"]["public_instruction"] == prepared, "worker candidate context changed")
        report['fixture_cleanup']=verify_fixture_cleanup(reader.json('native/owner-fixture-worktree-cleanup.json',4*MIB),
            result['observed_worker_allocations'],before,prepared['task_cid'], reader=reader,
            population=scope['native_population'])
        publication = result["publication"]
        need(publication["changed_paths"] == ["calc.py"] and publication["public_checks_passed"] is True
             and len(publication["parents"]) == 2 and publication["parents"][0] == publication["baseline_commit"],
             "native two-parent formatting merge differs")
        need(publication['baseline_commit']==reader.json('native/signed-manifest.json',16*MIB)['payload']['baseline_commit'],
             'native publication differs from signed baseline')
        need(reader.raw("native/repository/calc.py", 65536) == worker.AFTER, "published authored patch bytes differ")
        checks = reader.json("native/published-public-checks.json")
        need(same(checks, publication["public_checks"]), "published public-check evidence differs")
        report["public_checks"] = verify_public_checks(checks, reader)
        before_states, after_states = reader.json("native/checkpoint-states-before.json", 4 * MIB), reader.json("native/checkpoint-states-after.json", 4 * MIB)
        need(same(before_states, after_states) and same(after_states, seed["checkpoint_states"])
             and same(before_states, result["numerical_before"]) and same(after_states, result["numerical_after"]),
             "parent/child checkpoint or Adam population changed")
        report["scan_and_model_custody"] = verify_complete_scan(reader, seed, result, after_states)
        original_owners = reader.json("native/closed-full-scan/owners-after-cold.json", 4 * MIB)
        owners_before, owners_after = reader.json("native/owners-before.json", 4 * MIB), reader.json("native/owners-after-publication.json", 4 * MIB)
        expected_registry = deepcopy(original_owners["registry"])
        expected_registry["meta"][0][4] = "/results/native/private/model-artifacts"
        expected_registry["meta"][0][5] += 1
        need(same(owners_before["registry"], expected_registry) and same(owners_after["registry"], expected_registry),
             "registry rows changed beyond one explicit native owner open")
        for row in seed["copied_members"]:
            if row["path"].startswith(("private/model-artifacts/", "private/source-artifacts/", "closed-full-scan/")):
                raw = reader.raw("native/" + row["path"], 16 * MIB)
                need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "frozen model/source/scan CAS or evidence changed")
        for row in reader.json("native/source-artifacts-before.json", 4 * MIB):
            raw = reader.raw("native/private/source-artifacts/" + row["path"], 16 * MIB)
            need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "preexisting immutable source artifact changed")
        need(owners_after["source"]["current_head"]["generation"] == seed["current_head"]["generation"] + 1,
             "actual formatted source did not advance generation")
        native_reader = source.Reader(reader.root / "native", seconds=120)
        published = reader.json("native/published-source-receipt.json", 4 * MIB)
        published_head = source.receipt_head(published)
        need(same(published_head, owners_after["source"]["current_head"])
             and same(published["previous_head"], seed["current_head"]), "published source receipt does not join prior captured head")
        manifest = source.parse(native_reader.cas(published_head["manifest_cid"]))
        source.manifest_members(manifest, published_head, get_artifact=native_reader.cas)
        need(manifest['snapshot']['git_commit']==publication['published_commit'],
             'published captured source belongs to another Git commit')
        calc=next(row for row in manifest['snapshot']['entries'] if row['path']=='calc.py')
        need(calc['size_bytes']==len(worker.AFTER)
             and native_reader.cas(calc['source_cid'],source=True)==worker.AFTER,
             'published source manifest does not capture actual authored formatting bytes')
        prior=reader.json('native/closed-full-scan/seed-evidence/current-manifest.json',4*MIB)['value']
        for name in ('mode','repository_id','exclusions','max_entries','max_file_bytes'):
            need(same(manifest['snapshot'][name],prior['snapshot'][name]),'published capture policy changed')
        def source_members(value):
            return {row['path']:{name:row[name] for name in ('source_cid','size_bytes','kind','opaque_reason')}
                    for row in value['snapshot']['entries'] if row['path']!='calc.py'}
        need(same(source_members(manifest),source_members(prior)),
             'published source changed other captured members or their disposition')
        report['preserved_repository'] = verify_preserved_repository(before, seed, prior)
        report["native_observations"] = {"original_tasks_completed": 2, "remaining_processes": 0,
            "registry_generation_before": original_owners["registry"]["meta"][0][5],
            "registry_generation_after": expected_registry["meta"][0][5], "checkpoints_and_saved_adam_unchanged": True,
            "source_generation_after": owners_after["source"]["current_head"]["generation"], "publication": publication}
        report["selected_sources"] = verify_selected_sources(reader, before)
        report["qualified"] = True
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        after = source.Reader(namespace, seconds=seconds).whole_archive()
        report["preserved"] = same(before, after)
        report["archive"] = before
        report["qualified"] = report["qualified"] and report["preserved"]
        report["recorded_seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    namespace = args.namespace.resolve(strict=True)
    output = args.output.absolute()
    need(output.resolve() == output and output != namespace and namespace not in output.parents
         and not output.exists(), "fresh external audit output required")
    report = audit_closed_archive(namespace, seconds=args.seconds)
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    output.chmod(0o444)
    print(json.dumps({key: report[key] for key in ("qualified", "preserved", "errors", "recorded_seconds")}, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
