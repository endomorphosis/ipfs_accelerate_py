"""Bounded transport of one closed completed scan, without native execution.

Only fourteen explicitly reviewed CAS objects and fixed historical observations
are copied.  Their numerical contents remain advisory.  A receiving harness
must reopen its own owners and fully validate the completed chain before use;
historical child processes and registry reopenings are never counted as new
work.  Reads and copies have sequential guards, not an atomic snapshot.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from benchmarks.agent_supervisor.container_coding import inventory_resume_setup_fixture as transport

SCHEMA = "inventory-resume-closed-scan-seed@1"
MATERIALIZED_SCHEMA = "inventory-resume-materialized-scan-seed@1"
MAX_FILES = 128
MAX_BYTES = 128 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
SOURCE_NAME = "inventory-resume-worker-qualification-20261003-09"
_AUTHORITY = {key: False for key in ("native_owners_opened", "training_executed", "source_execution_attested",
    "scan_execution_attested", "proof_authority", "qualification_authority", "execution_authority")}
_SCAN_AUTHORITY = {key: False for key in ("source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority", "admission_authority",
    "authoritative_cache_eligible", "behavioral_satisfaction", "training_executed", "decoded_formulas_generated",
    "repository_code_executed", "source_execution_attested", "scan_execution_attested")}
_PROVENANCE = ("container-execution-final.json", "native/result.json", "source-setup-seed.json",
    "native/generation-inputs.json", "native/scan-root.json", "native/scan-completion.json",
    "native/opt-out-equivalence.json", "native/fresh-process-resume.json", "native/owners-before-scans.json",
    "native/owners-after-resume.json") + tuple("native/" + pattern.format(number) for number in range(1, 5)
        for pattern in ("resume-request-{:02d}.json", "fresh-process-run-{:02d}.json",
                        "fresh-process-{:02d}.stdout", "fresh-process-{:02d}.stderr"))
_RECEIPT_FIELDS = {"schema", "qualified", "source_namespace", "source_audit", "source_scan_summary", "root_cid",
    "completion_cid", "head", "selected_version_id", "coverage", "opt_out_equivalence", "producers",
    "checkpoint_states", "copied_members", "copied_files", "copied_bytes", "inherited_actual_setup_epochs",
    "new_fitting_epochs", "new_scan_pages", "new_registry_reopens", "unknown_fitting_epochs",
    "fresh_native_validation_required", "authority"}


class ClosedScanError(ValueError):
    """The closed historical scan or bounded fresh transport differs."""


def _need(value, message):
    if not value:
        raise ClosedScanError(message)


def _wire(value, *, ascii=True):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ClosedScanError("finite JSON required") from error


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _cid(raw, codec):
    _need(codec in {"raw", "dag-json"}, "closed CID codec required")
    prefix = b"\x01\x55" if codec == "raw" else b"\x01\xa9\x02"
    return "b" + base64.b32encode(prefix + b"\x12\x20" + hashlib.sha256(raw).digest()).decode().lower().rstrip("=")


def _structured(value):
    _strict(value, floats=False)
    return _cid(_wire(value, ascii=False), "dag-json")


def _strict(value, *, floats=True, depth=0):
    _need(depth <= 96, "bounded JSON depth required")
    if type(value) is dict:
        _need(all(type(key) is str for key in value), "string JSON keys required")
        for item in value.values():
            _strict(item, floats=floats, depth=depth + 1)
    elif type(value) is list:
        for item in value:
            _strict(item, floats=floats, depth=depth + 1)
    else:
        _need(value is None or type(value) in {str, bool, int}
              or (floats and type(value) is float and math.isfinite(value)), "finite inert JSON required")


def _closed(value, fields, name):
    _need(type(value) is dict and set(value) == set(fields), "closed " + name + " required")


def _int(value, expected):
    _need(type(value) is int and value == expected, "exact integer differs")


def _false(value, expected):
    _closed(value, expected, "authority")
    _need(all(flag is False for flag in value.values()), "authority must remain false")


class _Reads:
    def __init__(self):
        self.identities = {}
        self.pins = {}

    def reserve(self, paths):
        for path in paths:
            identity = transport._fingerprint(transport._regular(path))
            _need(path not in self.identities or self.identities[path] == identity, "observed file identity changed")
            self.identities[path] = identity
        _need(len(self.identities) <= MAX_FILES and sum(row[4] for row in self.identities.values()) <= MAX_BYTES,
              "aggregate scan transport bound exceeded before body read")

    def raw(self, path, maximum=MAX_JSON_BYTES):
        self.reserve([path])
        raw, pin, _ = transport._read(path, maximum, expected=self.identities[path])
        _need(path not in self.pins or self.pins[path] == pin, "observed file bytes changed")
        self.pins[path] = pin
        return raw

    def json(self, path, maximum=MAX_JSON_BYTES):
        raw = self.raw(path, maximum)
        def pairs(items):
            result = {}
            for key, value in items:
                _need(key not in result, "duplicate JSON key refused")
                result[key] = value
            return result
        try:
            result = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: _need(False, "nonfinite JSON"))
            _strict(result)
            return result
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ClosedScanError("bounded finite JSON required") from error

    def close(self):
        for path, identity in self.identities.items():
            _need(transport._fingerprint(transport._regular(path)) == identity, "source membership changed after read")
            if path in self.pins:
                _, pin, _ = transport._read(path, MAX_BYTES, expected=identity, keep=False)
                _need(pin == self.pins[path], "source bytes changed after read")


def _cas_path(wanted, codec):
    _need(type(wanted) is str and re.fullmatch(r"b[a-z2-7]{58,61}", wanted), "canonical bounded CID required")
    return ("source" if codec == "raw" else "structured") + "/" + wanted[:4] + "/" + wanted


def _current_path(name):
    base = Path(__file__).resolve().parents[4]
    if base == Path("/opt/ipfs-supervisor"):
        return base / "datasets" / Path(*name.split(".")).with_suffix(".py")
    return transport._current_producer_path(name)


def _transport_rows(positive):
    rows = positive.get("transport_artifacts")
    _need(type(rows) is list and len(rows) == 14, "exact fourteen reviewed CAS objects required")
    seen, roles = set(), []
    for row in rows:
        _closed(row, {"role", "path", "cid", "codec", "bytes", "sha256", "mode"}, "transport artifact")
        relative = "private/source-artifacts/" + _cas_path(row["cid"], row["codec"])
        _need(row["path"] == relative and row["cid"] not in seen, "exact unique native CAS locator required")
        _need(type(row["bytes"]) is int and 0 < row["bytes"] <= (16 if row["codec"] == "raw" else 4) * 1024**2,
              "bounded CAS artifact required")
        _need(type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
              and type(row["mode"]) is int and 0 <= row["mode"] <= 0o777, "exact artifact pin required")
        roles.append(row["role"]); seen.add(row["cid"])
        _need(row["codec"] == ("dag-json" if row["role"] in {"root", "completion", "optout-root"} else "raw"),
              "artifact role/codec differs")
    _need(set(roles) == {"root", "completion", "optout-root", "optout-reference-page", *(f"page-{number:02d}" for number in range(1, 11))},
          "closed fourteen CAS roles required")
    return rows


def _read_cas(base, row, reads):
    path = base / transport._relative(row["path"])
    raw = reads.raw(path, (16 if row["codec"] == "raw" else 4) * 1024**2)
    _need(len(raw) == row["bytes"] and _sha(raw) == row["sha256"] and _cid(raw, row["codec"]) == row["cid"],
          "CAS bytes, digest or CID differ")
    value = reads.json(path)
    _need(raw == _wire(value, ascii=row["codec"] == "raw"), "canonical native CAS JSON required")
    if row["codec"] == "dag-json":
        _strict(value, floats=False)
    return value


def _page(page, wanted, root, root_cid, previous, start):
    _closed(page, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "start", "end",
        "total_entries", "page_membership_cid", "previous_page_cid", "entries", "inference", "worker_receipt",
        "coverage", "authority"}, "scan page")
    end = min(start + 32, 300)
    selected = root["members"][start:end]
    _need(page["schema"] == "codebase-inventory-resume-page@1" and page["root_cid"] == root_cid
          and page["head_cid"] == root["head_cid"] and page["membership_cid"] == root["membership_cid"]
          and page["model_artifact_cid"] == root["model"]["artifact_cid"]
          and page["previous_page_cid"] == previous and page["page_membership_cid"] == _structured(selected),
          "page root, model, membership or prefix differs")
    for key, value in (("start", start), ("end", end), ("total_entries", 300)):
        _int(page[key], value)
    entries = page["entries"]
    _need(type(entries) is list and len(entries) == end - start, "complete page ledger required")
    dispositions = Counter()
    inferred = []
    for ordinal, (entry, member) in enumerate(zip(entries, selected)):
        _closed(entry, {"member_index", "source_key", "entry_cid", "disposition", "reason", "target_sha256",
            "source_digest", "coverage", "inference_index"}, "page entry")
        _int(entry["member_index"], start + ordinal)
        _need(entry["source_key"] == member["source_key"] and entry["entry_cid"] == member["entry_cid"],
              "entry membership identity differs")
        _need(entry["disposition"] in {"inferred", "deferred_budget", "opaque", "parse_failed", "unindexed", "unsupported_target"},
              "closed inventory disposition required")
        dispositions[entry["disposition"]] += 1
        if entry["disposition"] == "inferred":
            _int(entry["inference_index"], len(inferred))
            _need(entry["reason"] is None and type(entry["coverage"]) is list and len(entry["coverage"]) == 2,
                  "inferred structural coverage differs")
            for key in ("target_sha256", "source_digest"):
                _need(type(entry[key]) is str and re.fullmatch(r"[0-9a-f]{64}", entry[key]), "target digest required")
            inferred.append(entry)
        else:
            _need(entry["inference_index"] is None, "deferred member claims inference")
    counts = dict(sorted(dispositions.items()))
    _need(page["coverage"] == {"inventory_entries": end - start, "inferred_rows": len(inferred), "dispositions": counts},
          "page coverage differs")
    _false(page["authority"], _SCAN_AUTHORITY)
    inference = page["inference"]
    _need(type(inference) is dict and inference.get("schema") == "native-projection-feature-inference/v1"
          and inference.get("training_executed") is False and inference.get("decoded_formulas_generated") is False,
          "historical numerical representation differs")
    for key in ("contract_sha256", "state_sha256", "feature_space_sha256"):
        _need(inference.get(key) == root["model"][key], "numerical model binding differs")
    _need(inference.get("coverage") == [item for entry in inferred for item in entry["coverage"]]
          and type(inference.get("rows")) is list and len(inference["rows"]) == len(inferred), "numerical coverage differs")
    for entry, row in zip(inferred, inference["rows"]):
        _closed(row, {"source_digest", "latent", "reconstructed_projection_features"}, "numerical row")
        _need(row["source_digest"] == entry["source_digest"] and type(row["latent"]) is list and len(row["latent"]) == 8,
              "numerical source or latent width differs")
        projections = row["reconstructed_projection_features"]
        _closed(projections, root["model"]["projection_ids"], "numerical projections")
        numbers = list(row["latent"])
        for key, block in projections.items():
            _need(type(block) is list and len(block) == root["model"]["projection_widths"][key], "feature width differs")
            numbers.extend(block)
        _need(all(type(value) in {int, float} and math.isfinite(value) for value in numbers), "finite numerical vectors required")
    receipt = page["worker_receipt"]
    _need(type(receipt) is dict and type(receipt.get("returncode")) is int and receipt["returncode"] == 0
          and receipt.get("workspace_cleaned") is True and receipt.get("source_execution_attested") is False
          and receipt.get("worker_sha256") == root["implementation"]["files"][
              "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"],
          "historical worker receipt differs")
    return {"page_cid": wanted, "start": start, "end": end, "membership_cid": page["page_membership_cid"],
            "inferred_rows": len(inferred), "dispositions": counts}


def _scan(base, positive, reads):
    _need(positive.get("schema") == "inventory-resume-positive-completed-scan-observation@1"
          and positive.get("verified") is True and positive.get("overall_native_qualification") is False
          and positive.get("native_worker_qualified") is False, "explicit positive scan with unqualified worker required")
    _false(positive["authority"], _SCAN_AUTHORITY)
    rows = _transport_rows(positive)
    reads.reserve([base / row["path"] for row in rows])
    values = {row["cid"]: _read_cas(base, row, reads) for row in rows}
    roles = {role: [row for row in rows if row["role"] == role] for role in {row["role"] for row in rows}}
    root_cid, completion_cid = roles["root"][0]["cid"], roles["completion"][0]["cid"]
    root, completion = values[root_cid], values[completion_cid]
    _closed(root, {"schema", "head", "head_cid", "membership_cid", "members", "model", "implementation",
                   "limits", "optimized", "authority"}, "scan root")
    _need(root["schema"] == "codebase-inventory-resume-root@1" and root["optimized"] is True
          and root["head_cid"] == _structured(root["head"]) and root["membership_cid"] == _structured(root["members"]),
          "root content identities differ")
    transport._head(root["head"])
    members = root["members"]
    _need(type(members) is list and len(members) == 300 and [row["raw_path_hex"] for row in members]
          == sorted({row["raw_path_hex"] for row in members})
          and all(row["source_key"] == "raw:" + row["raw_path_hex"] for row in members), "exact ordered 300-member root required")
    _need(root["limits"].get("page_entries") == 32 and type(root["limits"]["page_entries"]) is int
          and root["limits"].get("max_inventory_entries") == 512, "reviewed scan profile differs")
    _false(root["authority"], _SCAN_AUTHORITY); _false(completion["authority"], _SCAN_AUTHORITY)
    _need(root["model"]["latent_width"] == 8 and type(root["model"]["latent_width"]) is int
          and root["model"]["feature_columns"] == sum(root["model"]["projection_widths"].values()), "global 8D model layout differs")
    _need(len(root["implementation"]["files"]) == 17
          and root["implementation"]["sha256"] == _sha(_wire(root["implementation"]["files"])), "exact implementation pins differ")
    _closed(completion, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "pages", "coverage", "authority"}, "completion")
    _need(completion["schema"] == "codebase-inventory-resume-completion@1" and completion["root_cid"] == root_cid
          and completion["head_cid"] == root["head_cid"] and completion["membership_cid"] == root["membership_cid"]
          and completion["model_artifact_cid"] == root["model"]["artifact_cid"] and len(completion["pages"]) == 10,
          "completion binding differs")
    previous, start, counts = None, 0, Counter()
    page_cids = [roles[f"page-{number:02d}"][0]["cid"] for number in range(1, 11)]
    _need(len(set(page_cids)) == 10 and page_cids == [row["page_cid"] for row in completion["pages"]], "page membership/order differs")
    for descriptor in completion["pages"]:
        wanted = descriptor["page_cid"]
        expected = _page(values[wanted], wanted, root, root_cid, previous, start)
        _need(_wire(descriptor) == _wire(expected), "completion page descriptor differs")
        previous, start = wanted, expected["end"]
        counts.update(expected["dispositions"])
    coverage = {"inventory_entries": 300, "pages": 10, "inferred_rows": 205,
        "dispositions": {"deferred_budget": 89, "inferred": 205, "opaque": 2, "parse_failed": 1, "unindexed": 1, "unsupported_target": 2}}
    _need(start == 300 and dict(sorted(counts.items())) == coverage["dispositions"]
          and _wire(completion["coverage"]) == _wire(coverage), "complete coverage conservation differs")
    reference_root = values[roles["optout-root"][0]["cid"]]
    _need(reference_root == {**root, "optimized": False}, "opt-out reference root differs")
    reference_page_cid = roles["optout-reference-page"][0]["cid"]
    reference = values[reference_page_cid]
    _page(reference, reference_page_cid, reference_root, roles["optout-root"][0]["cid"], None, 0)
    first = values[completion["pages"][0]["page_cid"]]
    for key in ("entries", "coverage", "inference"):
        _need(_wire(first[key]) == _wire(reference[key]), "optimized numerical reference differs")
    equivalence = positive["opt_out_equivalence"]
    _need(equivalence["optimized_root_cid"] == root_cid and equivalence["optimized_page_cid"] == completion["pages"][0]["page_cid"]
          and equivalence["reference_root_cid"] == roles["optout-root"][0]["cid"]
          and equivalence["reference_page_cid"] == reference_page_cid, "opt-out CID identities differ")
    for key, value in (("root_cid", root_cid), ("completion_cid", completion_cid), ("head", root["head"]),
        ("membership_cid", root["membership_cid"]), ("model", root["model"]), ("coverage", coverage), ("ordered_pages", completion["pages"])):
        _need(_wire(positive.get(key)) == _wire(value), "independent positive scan binding differs: " + key)
    return root, completion, rows


def _history(provenance, root, completion, positive):
    result = provenance["native/result.json"]
    container = provenance["container-execution-final.json"]
    _need(result.get("schema") == "codebase-inventory-resume-native-qualification@1" and result.get("qualified") is False
          and result.get("error_type") == "StructuredIdentityError" and result.get("unknown_fitting_epochs") is False,
          "exact closed unqualified runtime-create result required")
    failed = [row for row in result["phases"] if row.get("status") == "failed"]
    _need(len(failed) == 1 and failed[0]["name"] == "bind_inventory_native_runtime", "only runtime-create failure may be reused")
    for name in ("container_removed", "host_reservation_released", "container_results_copied",
                 "retained_source_verified_after_execution", "runtime_used_retained_source_copies", "native_module_launched"):
        _need(container.get(name) is True, "closed container lifecycle required")
    for record in (result["final_resources"], container["host_resources_after_cleanup"]):
        for key in ("active_lease_count", "waiting_request_count"):
            _int(record[key], 0)
    for key in ("allocated_child_process_slots",):
        _int(container["host_resources_after_cleanup"][key], 0)
    for key in ("cpu_slots", "memory_mb"):
        _int(container["host_resources_after_cleanup"]["allocated"][key], 0)
    for key, expected in (("known_actual_setup_epochs", 0), ("inherited_actual_setup_epochs", 2), ("post_setup_fit_attempt_count", 0)):
        _int(result[key], expected)
    _need(result["setup_training_attempts"] == [] and result["scan_root_cid"] == completion["root_cid"]
          and result["completed_scan_cid"] == positive["completion_cid"] and result["head"] == root["head"]
          and result["selected_version_id"] == root["model"]["version_id"] and result["scan_coverage"] == completion["coverage"],
          "source result identities differ")
    seed = provenance["source-setup-seed.json"]
    _need(seed["schema"] == transport.SCHEMA and seed["qualified"] is False and seed["head"] == root["head"]
          and seed["child_version_id"] == root["model"]["version_id"]
          and seed["checkpoint_states"] == positive["full_checkpoint_states"], "original setup/checkpoint binding differs")
    for key in ("state_sha256", "latent_width", "feature_columns"):
        _need(seed["checkpoint_states"]["child"][key] == root["model"][key], "full child checkpoint state differs")
    _need(seed["checkpoint_states"]["child"]["artifact"] == root["model"]["artifact"], "child checkpoint artifact differs")
    fresh = provenance["native/fresh-process-resume.json"]
    runs = fresh["process_runs"]
    _need(fresh["qualified"] is True and fresh["complete"] is True and len(runs) == 4
          and fresh["coverage"] == completion["coverage"] and fresh["root_cid"] == completion["root_cid"]
          and fresh["completion_cid"] == positive["completion_cid"], "four historical resumed executions required")
    previous = completion["pages"][1]["page_cid"]
    pids = []
    state = positive["numerical_state"]
    _need(_wire(state) == _wire({key: seed["checkpoint_states"]["child"][key] for key in
          ("state_sha256", "completed_epochs", "adam_steps", "latent_width", "feature_columns")}), "historical full numerical state differs")
    for number, run in enumerate(runs, 1):
        request = provenance[f"native/resume-request-{number:02d}.json"]
        _need(_wire(run) == _wire(provenance[f"native/fresh-process-run-{number:02d}.json"]), "historical chunk report differs")
        _int(run["run_number"], number)
        _need(type(run["pid"]) is int and run["pid"] > 1 and run["pid"] not in pids, "distinct historical child PID required")
        pids.append(run["pid"])
        for key in ("registry_owner_generation_before", "registry_owner_generation_after"):
            _int(run[key], number + 2)
        _int(run["post_setup_fit_attempt_count"], 0)
        _need(run["qualified"] is True and run["source_model_owner_preservation"] is True
              and run["numerical_before"] == state == run["numerical_after"], "historical state or owner preservation differs")
        cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": completion["root_cid"],
            "next_offset": (number * 2) * 32, "previous_page_cid": previous}
        _need(request["cursor"] == cursor == run["request_cursor"] and request["root_cid"] == completion["root_cid"]
              and request["version_id"] == root["model"]["version_id"] and request["run_number"] == number,
              "historical cursor or request binding differs")
        selected = [row["page_cid"] for row in completion["pages"][number * 2:number * 2 + 2]]
        _need(run["pages_created"] == selected, "historical child page sequence differs")
        reported = positive["fresh_process_resume"]["chunks"][number - 1]
        for key in ("run_number", "pid", "registry_owner_generation_before", "registry_owner_generation_after", "request_cursor", "pages_created", "next_cursor"):
            _need(_wire(reported[key]) == _wire(run[key]), "independent historical chunk differs")
        previous = selected[-1]
        _need(run["complete"] is (number == 4), "historical completion timing differs")
        if number < 4:
            _need(run["next_cursor"] == {**cursor, "next_offset": (number * 2 + 2) * 32, "previous_page_cid": previous},
                  "historical cursor continuation differs")
        else:
            _need(run["completion_cid"] == positive["completion_cid"], "historical final completion differs")
        for key in ("active_lease_count", "waiting_request_count"):
            _int(run["final_resources"][key], 0)
    _need(fresh["pids"] == pids and positive["fresh_process_resume"]["pids"] == pids, "historical PID ledger differs")
    transition = result["registry_owner_reopen_transition"]
    _need(transition == {"schema": "inventory-resume-registry-owner-reopen@1", "before": 2, "after": 7,
        "reopens": 5, "child_generations": [3, 4, 5, 6], "other_owner_fields_unchanged": True}, "explicit historical owner transition differs")
    before, after = provenance["native/owners-before-scans.json"], provenance["native/owners-after-resume.json"]
    _need(type(before["registry"]["meta"]) is list and len(before["registry"]["meta"]) == 1
          and len(before["registry"]["meta"][0]) == 6, "native registry owner row required")
    before_registry, after_registry = before["registry"], after["registry"]
    original = before_registry["meta"][0]
    final = after_registry["meta"][0]
    _int(original[5], 2); _int(final[5], 7)
    _need(_wire({**after, "registry": {**after_registry, "meta": [final[:5] + [2]]}}) == _wire(before),
          "historical owner fields changed beyond explicit reopens")
    equivalence = provenance["native/opt-out-equivalence.json"]
    _need(equivalence == result["opt_out_equivalence"] and equivalence["entries_coverage_and_inference_exact"] is True
          and equivalence["throughput_qualified"] is False, "historical opt-out observation differs")
    _need(equivalence == positive["opt_out_equivalence"], "independent opt-out observation differs")
    return result, equivalence


def _copy_selected(source_map, destination, rows, reads):
    destination = transport._absolute(destination, exists=destination.exists())
    if destination.exists():
        _need(not any(destination.iterdir()), "fresh empty staging directory required")
    else:
        destination.mkdir(mode=0o700)
    for row in rows:
        target = destination / row["path"]
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _need(target.parent.resolve(strict=True) == target.parent, "copy ancestor alias refused")
        raw = reads.raw(source_map[row["path"]], MAX_JSON_BYTES)
        _need(len(raw) == row["bytes"] and _sha(raw) == row["sha256"], "selected copy bytes differ")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, row["mode"])
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw); stream.flush(); os.fsync(descriptor)
            os.fchmod(descriptor, row["mode"])
        finally:
            os.close(descriptor)
    _need(set(transport._tree(destination)[0]) == {row["path"] for row in rows}, "copied membership differs")


def _summary(positive, result):
    return {"schema": positive["schema"], "root_cid": positive["root_cid"], "completion_cid": positive["completion_cid"],
        "membership_cid": positive["membership_cid"], "numerical_state": positive["numerical_state"],
        "fresh_process_resume": positive["fresh_process_resume"], "registry_owner_reopen_transition": result["registry_owner_reopen_transition"],
        "historical_execution_only": True, "overall_native_qualification": False, "native_worker_qualified": False}


def stage_closed_scan(namespace: Path, destination: Path, audit_path: Path) -> dict:
    """Stage exactly one independently reviewed, closed native09 scan."""
    namespace = transport._absolute(namespace)
    _need(namespace.name == SOURCE_NAME, "exact reviewed source namespace required")
    destination = destination.absolute()
    _need(not destination.is_relative_to(namespace) and not namespace.is_relative_to(destination), "source/destination overlap refused")
    audit_path = audit_path.absolute()
    _need(not audit_path.is_relative_to(namespace), "independent supplemental audit must be a sibling observation")
    reads = _Reads()
    reads.reserve([audit_path] + [namespace / path for path in _PROVENANCE])
    audit = reads.json(audit_path)
    _need(audit.get("schema") == "inventory-resume-worker-independent-audit@1" and audit.get("qualified") is False
          and audit.get("primary_archive", {}).get("preserved") is True, "preserved independent unqualified archive audit required")
    positive = audit["positive_completed_scan"]
    _need(audit.get("namespace") == str(namespace) and positive.get("source_namespace") == str(namespace),
          "independent audit source namespace differs")
    for name in ("current_deployment_freshness_attested", "native_owners_opened_by_auditor",
                 "numerical_execution_independently_reperformed", "process_origin_attested"):
        _need(positive.get(name) is False, "audit authority must remain false")
    root, completion, cas_rows = _scan(namespace / "native", positive, reads)
    provenance = {path: reads.json(namespace / path) for path in _PROVENANCE if not path.endswith((".stdout", ".stderr"))}
    result, equivalence = _history(provenance, root, completion, positive)
    for role, path in (("source_result", "native/result.json"), ("source_container", "container-execution-final.json")):
        _need(positive[role] == reads.pins[namespace / path], "independent closed observation byte pin differs")
    _need(provenance["native/scan-root.json"] == root and provenance["native/scan-completion.json"] == completion,
          "review records differ from exact CAS")
    for number in range(1, 5):
        stdout = reads.raw(namespace / f"native/fresh-process-{number:02d}.stdout", 1024**2)
        stderr = reads.raw(namespace / f"native/fresh-process-{number:02d}.stderr", 1024**2)
        _need(stderr == b"" and stdout.strip().splitlines(), "historical stdout or stderr differs")
        final = json.loads(stdout.strip().splitlines()[-1])
        _need(type(final) is dict and final.get("qualified") is True, "historical child stdout is not positive")
        observed = positive["fresh_process_resume"]["chunks"][number - 1]
        _need(_sha(stdout) == observed["stdout_sha256"] and _sha(stderr) == observed["stderr_sha256"],
              "independent historical stdout/stderr pin differs")
    generation = provenance["native/generation-inputs.json"]
    _need(generation.get("execution_attestation") is False, "producer observation must remain advisory")
    selected = {row["name"]: row for row in generation["files"]}
    pins = positive["implementation_files"]
    _need(type(pins) is dict and set(pins) == set(root["implementation"]["files"]), "complete seventeen implementation pins required")
    implementation_paths = []
    for name in pins:
        _need(name.startswith("ipfs_datasets_py.") and re.fullmatch(r"[a-zA-Z0-9_.]+", name), "closed datasets implementation locator required")
        _closed(pins[name], {"path", "bytes", "sha256"}, "implementation observation")
        expected_path = "datasets/" + name.replace(".", "/") + ".py"
        _need(pins[name]["path"] == expected_path, "exact retained implementation path required")
        implementation_paths.extend([namespace / expected_path, _current_path(name)])
        if name in selected:
            implementation_paths.append(namespace / "native" / selected[name]["copy"])
    reads.reserve(implementation_paths)
    producers = {}
    for name, pin in pins.items():
        raw = reads.raw(namespace / pin["path"], 4 * 1024**2)
        _need(len(raw) == pin["bytes"] and _sha(raw) == pin["sha256"] == root["implementation"]["files"][name]
              and reads.raw(_current_path(name), 4 * 1024**2) == raw, "retained/current implementation differs")
        if name in selected:
            row = selected[name]
            _need(row["bytes"] == pin["bytes"] and row["sha256"] == pin["sha256"]
                  and reads.raw(namespace / "native" / row["copy"], 4 * 1024**2) == raw, "native selected producer differs")
        producers[name] = {"bytes": pin["bytes"], "sha256": pin["sha256"]}
    for name, checksum in transport._PRODUCERS.items():
        _need(producers[name]["sha256"] == checksum, "six original checkpoint producer pins differ")
    for role, state in positive["full_checkpoint_states"].items():
        _need(role in {"root", "child"}, "closed inherited checkpoint roles required")
        artifact = state["artifact"]
        raw = reads.raw(namespace / "native/private/model-artifacts" / artifact["sha256"][:2] / artifact["sha256"])
        saved = reads.json(namespace / "native/private/model-artifacts" / artifact["sha256"][:2] / artifact["sha256"])
        actual = {"artifact": {"bytes": len(raw), "sha256": _sha(raw)}, "state_sha256": _sha(_wire(saved["state"])),
            "report_sha256": _sha(_wire(saved["report"])), "completed_epochs": saved["state"]["completed_epochs"],
            "adam_steps": [item["step"] for item in saved["state"]["adam"]], "latent_width": saved["state"]["latent_width"],
            "feature_columns": len(saved["feature_space"]["columns"])}
        _need(_wire(actual) == _wire(state) and raw == _wire(saved), "full raw inherited checkpoint state differs")
    historical = {row["path"]: row for row in audit["primary_archive"]["before"]["files"] if row["kind"] == "file"}
    for path, pin in reads.pins.items():
        if path.is_relative_to(namespace):
            locator = path.relative_to(namespace).as_posix()
            _need(locator in historical, "observed source file absent from independent preservation inventory")
            previous = historical[locator]
            metadata = path.lstat()
            _need(pin == {key: previous[key] for key in ("bytes", "sha256")}
                  and stat.S_IMODE(metadata.st_mode) == previous["mode"] and metadata.st_mtime_ns == previous["mtime_ns"],
                  "source bytes/mode/mtime differ from independent preservation inventory")
    copied, sources = [], {}
    for row in cas_rows:
        name = "cas/" + row["path"].removeprefix("private/source-artifacts/")
        copied.append({"path": name, "bytes": row["bytes"], "sha256": row["sha256"], "mode": row["mode"],
                       "kind": "cas", "cid": row["cid"], "codec": row["codec"], "role": row["role"]})
        sources[name] = namespace / "native" / row["path"]
    for path in (*_PROVENANCE, "independent-audit.json"):
        source = audit_path if path == "independent-audit.json" else namespace / path
        raw = reads.raw(source)
        name = "provenance/" + path
        copied.append({"path": name, "bytes": len(raw), "sha256": _sha(raw), "mode": stat.S_IMODE(source.lstat().st_mode), "kind": "provenance"})
        sources[name] = source
    copied.sort(key=lambda row: row["path"])
    _need(len(copied) <= MAX_FILES and sum(row["bytes"] for row in copied) <= MAX_BYTES, "aggregate copied scan bound exceeded")
    _copy_selected(sources, destination, copied, reads)
    reads.close()
    summary = _summary(positive, result)
    receipt = {"schema": SCHEMA, "qualified": False, "source_namespace": str(namespace),
        "source_audit": {"path": str(audit_path), **reads.pins[audit_path]}, "source_scan_summary": summary,
        "root_cid": positive["root_cid"], "completion_cid": positive["completion_cid"], "head": root["head"],
        "selected_version_id": root["model"]["version_id"], "coverage": completion["coverage"], "opt_out_equivalence": equivalence,
        "producers": producers, "checkpoint_states": positive["full_checkpoint_states"], "copied_members": copied,
        "copied_files": len(copied), "copied_bytes": sum(row["bytes"] for row in copied), "inherited_actual_setup_epochs": 2,
        "new_fitting_epochs": 0, "new_scan_pages": 0, "new_registry_reopens": 0, "unknown_fitting_epochs": False,
        "fresh_native_validation_required": True, "authority": dict(_AUTHORITY)}
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt):
    _strict(receipt)
    _closed(receipt, _RECEIPT_FIELDS, "scan seed receipt")
    _need(receipt["schema"] == SCHEMA and receipt["qualified"] is False and receipt["unknown_fitting_epochs"] is False
          and receipt["fresh_native_validation_required"] is True, "advisory scan seed receipt required")
    _false(receipt["authority"], _AUTHORITY)
    _closed(receipt["source_audit"], {"path", "bytes", "sha256"}, "source audit pin")
    _need(type(receipt["source_namespace"]) is str and Path(receipt["source_namespace"]).is_absolute()
          and Path(receipt["source_namespace"]).name == SOURCE_NAME
          and type(receipt["source_audit"]["path"]) is str and Path(receipt["source_audit"]["path"]).is_absolute()
          and type(receipt["source_audit"]["bytes"]) is int and 0 < receipt["source_audit"]["bytes"] <= MAX_JSON_BYTES
          and type(receipt["source_audit"]["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt["source_audit"]["sha256"]),
          "exact bounded original audit locator required")
    for key, expected in (("inherited_actual_setup_epochs", 2), ("new_fitting_epochs", 0), ("new_scan_pages", 0), ("new_registry_reopens", 0)):
        _int(receipt[key], expected)
    rows = receipt["copied_members"]
    _need(type(rows) is list and 14 < len(rows) <= MAX_FILES and [row["path"] for row in rows] == sorted({row["path"] for row in rows}),
          "unique ordered bounded seed members required")
    for row in rows:
        fields = {"path", "bytes", "sha256", "mode", "kind"} | ({"cid", "codec", "role"} if row.get("kind") == "cas" else set())
        _closed(row, fields, "copied scan member")
        transport._relative(row["path"])
        _need(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_JSON_BYTES
              and type(row["mode"]) is int and 0 <= row["mode"] <= 0o777
              and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "bounded copied pin required")
        if row["kind"] == "cas":
            _need(row["path"] == "cas/" + _cas_path(row["cid"], row["codec"]), "exact CAS stage locator required")
        else:
            _need(row["kind"] == "provenance" and row["path"] in {"provenance/" + path for path in (*_PROVENANCE, "independent-audit.json")},
                  "unreviewed provenance member refused")
    _need(sum(row["kind"] == "cas" for row in rows) == 14
          and {row["path"] for row in rows if row["kind"] == "provenance"}
            == {"provenance/" + path for path in (*_PROVENANCE, "independent-audit.json")}, "exact staged membership required")
    _int(receipt["copied_files"], len(rows)); _int(receipt["copied_bytes"], sum(row["bytes"] for row in rows))
    _need(receipt["copied_bytes"] <= MAX_BYTES, "aggregate seed byte bound exceeded")


def materialize_staged_scan(seed: Path, index_artifacts_root: Path, receipt: dict) -> dict:
    """Copy reviewed CAS bytes into a fresh baseline CAS; never open owners."""
    _validate_receipt(receipt)
    seed, destination = transport._absolute(seed), transport._absolute(index_artifacts_root)
    _need(not seed.is_relative_to(destination) and not destination.is_relative_to(seed), "seed/CAS overlap refused")
    files, directories = transport._tree(seed)
    _need(set(files) == {row["path"] for row in receipt["copied_members"]}, "seed membership differs")
    reads = _Reads(); reads.reserve([seed / row["path"] for row in receipt["copied_members"]])
    for row in receipt["copied_members"]:
        raw = reads.raw(seed / row["path"])
        _need(len(raw) == row["bytes"] and _sha(raw) == row["sha256"], "staged member bytes differ")
    audit = reads.json(seed / "provenance/independent-audit.json")
    _need(reads.pins[seed / "provenance/independent-audit.json"] == {key: receipt["source_audit"][key] for key in ("bytes", "sha256")},
          "staged independent audit pin differs")
    positive = audit["positive_completed_scan"]
    # The shared scan parser requires native paths.  Use its original locators
    # through a pure read adapter, never filesystem aliases or product imports.
    class StageReads:
        def path(self, path):
            relative = path.relative_to(seed).as_posix()
            return seed / ("cas/" + relative.removeprefix("private/source-artifacts/"))
        def reserve(self, paths):
            reads.reserve([self.path(path) for path in paths])
        def raw(self, path, maximum=MAX_JSON_BYTES):
            return reads.raw(self.path(path), maximum)
        def json(self, path, maximum=MAX_JSON_BYTES):
            return reads.json(self.path(path), maximum)
    root, completion, cas_rows = _scan(seed, positive, StageReads())
    provenance = {path: reads.json(seed / "provenance" / path) for path in _PROVENANCE if not path.endswith((".stdout", ".stderr"))}
    result, equivalence = _history(provenance, root, completion, positive)
    for key, value in (("root_cid", positive["root_cid"]), ("completion_cid", positive["completion_cid"]), ("head", root["head"]),
        ("selected_version_id", root["model"]["version_id"]), ("coverage", completion["coverage"]), ("opt_out_equivalence", equivalence),
        ("checkpoint_states", positive["full_checkpoint_states"])):
        _need(_wire(receipt[key]) == _wire(value), "receipt/staged binding differs: " + key)
    _need(_wire(receipt["source_scan_summary"]) == _wire(_summary(positive, result))
          and receipt["source_namespace"] == positive["source_namespace"],
          "receipt historical execution summary differs")
    pins = positive["implementation_files"]
    _need(set(receipt["producers"]) == set(pins), "receipt complete producer set differs")
    reads.reserve([_current_path(name) for name in pins])
    for name, pin in pins.items():
        raw = reads.raw(_current_path(name), 4 * 1024**2)
        expected = {"bytes": len(raw), "sha256": _sha(raw)}
        _need(expected == receipt["producers"][name] == {key: pin[key] for key in ("bytes", "sha256")}
              and expected["sha256"] == root["implementation"]["files"][name], "current materializing producer differs")
    targets = [destination / _cas_path(row["cid"], row["codec"]) for row in cas_rows]
    _need(all(not path.exists() and not path.is_symlink() for path in targets), "CAS overwrite refused")
    destination_identity = transport._fingerprint(destination.lstat())[:2]
    for row, target in zip(cas_rows, targets):
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _need(target.parent.resolve(strict=True) == target.parent and transport._fingerprint(destination.lstat())[:2] == destination_identity,
              "destination directory alias or replacement refused")
        raw = reads.raw(seed / "cas" / _cas_path(row["cid"], row["codec"]))
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw); stream.flush(); os.fsync(descriptor)
            os.fchmod(descriptor, row["mode"])
        finally:
            os.close(descriptor)
        _, pin, _ = transport._read(target, MAX_JSON_BYTES, keep=False)
        _need(pin == {key: row[key] for key in ("bytes", "sha256")}, "materialized CAS bytes differ")
    reads.close()
    _need(transport._tree(seed) == (files, directories), "seed membership changed after materialization")
    return {"schema": MATERIALIZED_SCHEMA, "qualified": False, "seed_receipt_sha256": _sha(_wire(receipt)),
        "output": str(destination), **{key: receipt[key] for key in ("root_cid", "completion_cid", "head", "selected_version_id",
            "coverage", "opt_out_equivalence", "source_namespace", "source_audit", "source_scan_summary", "producers", "checkpoint_states")},
        "copied_cas_objects": 14, "copied_cas_bytes": sum(row["bytes"] for row in cas_rows),
        "inherited_actual_setup_epochs": 2, "new_fitting_epochs": 0, "new_scan_pages": 0, "new_registry_reopens": 0,
        "unknown_fitting_epochs": False, "fresh_native_validation_required": True, "authority": dict(_AUTHORITY)}
