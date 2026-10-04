"""Bounded read-only joins for a retained resumable-inventory worker run.

No product module, owner database, model, checker, Git command or Docker daemon
is opened or executed. Ed25519 verification uses only the retained public key.
Signatures are historical observations, not current revocation checks. Native
execution receipts and numerical results are checked as retained observations;
this reader does not attest process origin or numerical correctness.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import unicodedata

SCHEMA = "inventory-resume-worker-independent-audit@1"
MAX_FILES = 65_536
MAX_ARCHIVE_BYTES = 8 * 1024**3
MAX_FILE_BYTES = 2 * 1024**3
MAX_READ_BYTES = 64 * 1024**2
MAX_TOTAL_READ_BYTES = 512 * 1024**2
AUTHORITY = frozenset(("source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority", "admission_authority",
    "authoritative_cache_eligible", "behavioral_satisfaction", "training_executed",
    "decoded_formulas_generated", "repository_code_executed", "source_execution_attested",
    "scan_execution_attested"))
SETUP_PRODUCERS = {
    "ipfs_datasets_py.logic.software_contracts.codebase_source_training": "aea5fa8803ae6b6dde7c4e71b81f68a7f95257b3a5d76cea693ee13e66710aa0",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir_targets": "88d619efdc83f8e207b13b05f879c956655a7b5825f5647b46457ca92539c3cf",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry": "ca31983a5f514c7f5cf8b1b9250ef92e1beb13c0a2305381aac575931e79b4ac",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features": "0d93a2e56d0b46cff34f15c4792042bca33f47af00da80e21fa501ddf442de06",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_feature_worker": "91174cb51a0832e4cd691e56f94c0d6b34af1f06e6afec2c0d780d58fa3196db",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_cuda": "954bcdee38c8f55f0d9d57045b9f2648aac7efcdf105b3655f410dea1249f2e1",
}
SETUP_SOURCE_NAME = "inventory-resume-worker-qualification-20261003-06"
SETUP_SOURCE_AUDIT_SHA256 = "8020c5c31f67d0ee8e903c2e97123af6fa9cebf8dbf2a570710457f0035a9091"
SETUP_AUTHORITY = {"native_owners_opened", "training_executed", "source_execution_attested",
                   "proof_authority", "qualification_authority"}
SCAN_SOURCE_NAME = "inventory-resume-worker-qualification-20261003-09"
SCAN_SOURCE_AUDIT_SHA256 = "501330f167706f39bdbde7ac235de2444f2c94cb7a14bfe27f0f5258fb87e940"
SCAN_SEED_AUTHORITY = SETUP_AUTHORITY | {"scan_execution_attested", "execution_authority"}
SCAN_PROVENANCE = ("container-execution-final.json", "native/result.json", "source-setup-seed.json",
    "native/generation-inputs.json", "native/scan-root.json", "native/scan-completion.json",
    "native/opt-out-equivalence.json", "native/fresh-process-resume.json", "native/owners-before-scans.json",
    "native/owners-after-resume.json") + tuple("native/" + pattern.format(number) for number in range(1, 5)
        for pattern in ("resume-request-{:02d}.json", "fresh-process-run-{:02d}.json",
                        "fresh-process-{:02d}.stdout", "fresh-process-{:02d}.stderr"))
VOLATILE = {"status", "created_at_ms", "updated_at_ms", "observed_at_ms", "started_at_ms", "finished_at_ms"}
BEFORE = b"def increment(n: int) -> int:\n    return n + 1\n"
AFTER = b"def increment(n: int) -> int:\n    return n + 2\n"


class InventoryWorkerAuditError(ValueError):
    """A required retained observation does not independently join."""


def need(condition, message):
    if not condition:
        raise InventoryWorkerAuditError(message)


def wire(value, *, ascii=True):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii,
                      allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(value, depth=0):
    need(depth <= 96, "structured identity nesting bound exceeded")
    if type(value) is dict:
        need(all(type(key) is str for key in value), "structured keys must be strings")
        for item in value.values():
            strict(item, depth + 1)
    elif type(value) is list:
        for item in value:
            strict(item, depth + 1)
    else:
        need(type(value) in {str, bool, int, type(None)}, "structured identity rejects floating values")


def cid(raw, codec="raw"):
    need(codec in {"raw", "dag-json"}, "unsupported CID codec")
    prefix = b"\x01\x55" if codec == "raw" else b"\x01\xa9\x02"
    return "b" + base64.b32encode(prefix + b"\x12\x20" + hashlib.sha256(raw).digest()).decode().lower().rstrip("=")


def structured(value):
    strict(value)
    return cid(wire(value, ascii=False), "dag-json")


def semantic(value):
    if type(value) is dict:
        return {key: semantic(item) for key, item in value.items() if key not in VOLATILE}
    if type(value) is list:
        return [semantic(item) for item in value]
    return value


def prompt_task_record_cid(record):
    """Replay the native task identity, omitting only its outer self-ID.

    PromptTaskRecord.to_record adds content_id to its to_dict payload. Embedded
    output/validation/acceptance records retain their own content_id fields in
    the parent identity; removing those would change the native contract.
    """
    need(type(record) is dict and record.get("schema") == "ipfs_accelerate_py/agent-supervisor/prompt-task-record@1"
         and type(record.get("contract_version")) is int and record["contract_version"] == 1
         and type(record.get("content_id")) is str, "exact reviewed task record required")
    wanted = structured(semantic({key: value for key, value in record.items() if key != "content_id"}))
    need(record["content_id"] == wanted, "native task record self identity differs")
    return wanted


def modality_digest(contract):
    """Reviewed float-free modality identity preimage, without product imports."""
    strict(contract)
    def normalized(value):
        if type(value) is str:
            return unicodedata.normalize("NFC", value)
        if type(value) is list:
            return [normalized(item) for item in value]
        if type(value) is dict:
            result = {unicodedata.normalize("NFC", key): normalized(item) for key, item in value.items()}
            need(len(result) == len(value), "modality key normalization collides")
            return result
        return value
    return sha(wire({"canonicalization": "ir-canonical-json-v1", "collection_semantics": {},
        "domain": "autoencoder.modality", "identity_profile": "ir-canonical-identity-v1",
        "payload": normalized(contract), "schema_version": "autoencoder-modality-contract/v1"}, ascii=False))


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def parse(raw):
    def constant(value):
        raise InventoryWorkerAuditError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def false_authority(value):
    need(type(value) is dict and set(value) == AUTHORITY and all(flag is False for flag in value.values()),
         "inventory authority must remain exactly false")


def signature(envelope):
    need(type(envelope) is dict and set(envelope) == {"payload", "binding"}, "closed signed envelope required")
    binding = envelope["binding"]
    need(set(binding) == {"identity", "profile_id", "signature"}, "closed signature binding required")
    did = binding["identity"]
    need(type(did) is str and did.startswith("did:key:z"), "Ed25519 did:key required")
    alphabet, encoded, number = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", did[9:], 0
    for character in encoded:
        need(character in alphabet, "noncanonical public key alphabet")
        number = number * 58 + alphabet.index(character)
    public = b"\0" * (len(encoded) - len(encoded.lstrip("1"))) + number.to_bytes((number.bit_length() + 7) // 8, "big")
    need(len(public) == 34 and public[:2] == b"\xed\x01", "Ed25519 public key codec differs")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(public[2:]).verify(
            base64.b64decode(binding["signature"], validate=True), wire(envelope["payload"]))
    except Exception as error:
        raise InventoryWorkerAuditError("retained public signature does not verify") from error
    return envelope["payload"]


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink)


class Reader:
    def __init__(self, namespace, seconds, *, native_output=False):
        self.root = Path(namespace).absolute()
        need(self.root.resolve(strict=True) == self.root and self.root.is_dir(), "canonical archive root required")
        self.deadline = time.monotonic() + seconds
        self.total_read_bytes = 0
        self.observed = {}
        self.native_output = native_output

    def tick(self):
        need(time.monotonic() < self.deadline, "read-only audit exceeded declared deadline")

    def path(self, locator):
        need(type(locator) is str and locator and "\0" not in locator, "invalid retained locator")
        value = PurePosixPath(locator)
        need(".." not in value.parts, "retained locator escapes namespace")
        mounts = {"/results": self.root.parent if self.native_output else self.root,
            "/opt/ipfs-supervisor/source": self.root / "source",
            "/opt/ipfs-supervisor/datasets": self.root / "datasets", "/opt/ipfs-supervisor/kit": self.root / "kit",
            "/opt/ipfs-supervisor/finite-handoffs": self.root / "handoffs"}
        if value.is_absolute():
            result = Path(locator) if Path(locator).is_relative_to(self.root) else None
            for mount, target in mounts.items():
                if value == PurePosixPath(mount) or value.is_relative_to(mount):
                    result = target / value.relative_to(mount)
                    break
            need(result is not None, "locator is outside retained container mounts: " + locator)
        else:
            result = self.root / value
        for parent in (result, *result.parents):
            need(not parent.is_symlink(), "retained locator traverses an alias: " + str(result))
            if parent == self.root:
                break
        need(result.is_relative_to(self.root), "mapped locator escapes archive")
        return result

    def raw(self, locator, maximum=MAX_READ_BYTES):
        self.tick()
        path = self.path(str(locator))
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
                 "bounded single-link regular artifact required: " + str(path))
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        need(len(raw) == before.st_size <= maximum and identity(before) == identity(after)
             and identity(path.lstat()) == identity(after), "artifact changed during guarded read")
        self.total_read_bytes += len(raw)
        need(self.total_read_bytes <= MAX_TOTAL_READ_BYTES, "aggregate guarded read byte bound exceeded")
        record = {"bytes": len(raw), "sha256": sha(raw)}
        relative = path.relative_to(self.root).as_posix()
        need(relative not in self.observed or self.observed[relative] == record, "artifact changed between observations")
        self.observed[relative] = record
        return raw

    def json(self, locator, maximum=MAX_READ_BYTES):
        return parse(self.raw(locator, maximum))

    def cas(self, wanted, *, raw=False):
        need(type(wanted) is str and re.fullmatch(r"b[a-z2-7]{50,70}", wanted), "canonical CID required")
        locator = "native/private/source-artifacts/" + ("source" if raw else "structured") + "/" + wanted[:4] + "/" + wanted
        payload = self.raw(locator, 16 * 1024**2 if raw else 4 * 1024**2)
        need(cid(payload, "raw" if raw else "dag-json") == wanted, "native source CAS CID differs")
        value = parse(payload)
        need(payload == wire(value, ascii=raw), "native CAS canonical encoding differs")
        if not raw:
            strict(value)
        return value

    def whole_archive(self):
        """Stream every regular byte; retain link metadata without following it."""
        records, total, traversed = [], 0, 0
        for directory, names, files in os.walk(self.root, followlinks=False):
            self.tick()
            names.sort(); files.sort()
            for name in list(names):
                path = Path(directory) / name
                info = path.lstat()
                traversed += 1
                need(traversed <= MAX_FILES * 2, "archive traversal bound exceeded")
                relative = path.relative_to(self.root).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    records.append({"path": relative, "kind": "symlink", "target": os.readlink(path),
                                    "mode": stat.S_IMODE(info.st_mode)})
                    names.remove(name)
                else:
                    need(stat.S_ISDIR(info.st_mode), "archive directory is not inert")
                    records.append({"path": relative, "kind": "directory", "mode": stat.S_IMODE(info.st_mode)})
            for name in files:
                self.tick()
                path = Path(directory) / name
                before = path.lstat()
                relative = path.relative_to(self.root).as_posix()
                traversed += 1
                need(traversed <= MAX_FILES * 2 and len(records) < MAX_FILES, "whole archive entry bound exceeded")
                if stat.S_ISLNK(before.st_mode):
                    records.append({"path": relative, "kind": "symlink", "target": os.readlink(path),
                                    "mode": stat.S_IMODE(before.st_mode)})
                    continue
                need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_FILE_BYTES,
                     "whole archive contains a nonregular or oversized member: " + relative)
                total += before.st_size
                need(total <= MAX_ARCHIVE_BYTES, "whole archive byte bound exceeded")
                digest = hashlib.sha256()
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    need(identity(os.fstat(stream.fileno())) == identity(before), "archive identity changed before streaming")
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        self.tick(); digest.update(block)
                    need(identity(os.fstat(stream.fileno())) == identity(before), "archive changed during streaming")
                need(identity(path.lstat()) == identity(before), "archive changed after streaming")
                records.append({"path": relative, "kind": "file", "bytes": before.st_size,
                    "sha256": digest.hexdigest(), "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink,
                    "mtime_ns": before.st_mtime_ns})
        records.sort(key=lambda item: item["path"])
        return {"files": records, "regular_files": sum(row["kind"] == "file" for row in records),
            "regular_bytes": total, "inventory_cid": structured(records)}


def resources(value):
    need(type(value) is dict and type(value.get("active_lease_count")) is int
         and type(value.get("waiting_request_count")) is int
         and value["active_lease_count"] == value["waiting_request_count"] == 0,
         "actual public scheduler counts must be zero")


def audit_sources(reader, archive, report):
    selected = reader.json("selected-source-snapshot.json")
    need(set(selected) == {"source", "datasets", "kit"}, "three retained package roots required")
    actual_files = {row["path"]: row for row in archive["files"] if row["kind"] == "file"}
    count, total = 0, 0
    for package, rows in selected.items():
        need(type(rows) is list and len(rows) <= 20_000, "selected source count bound exceeded")
        expected = set()
        for row in rows:
            relative = package + "/" + row["relative"]
            path = reader.path(relative)
            need(relative not in expected, "duplicate selected source member")
            expected.add(relative)
            need(relative in actual_files, "selected retained source missing")
            retained = actual_files[relative]
            need(retained["sha256"] == row["sha256"] and retained["bytes"] == row["bytes"] <= 4 * 1024**2
                 and not retained["mode"] & 0o222, "retained source byte/mode pin differs: " + str(path))
            count += 1; total += row["bytes"]
        actual = {name for name in actual_files if name.startswith(package + "/")}
        need(actual == expected, "retained package membership differs")
        need(not any(row["kind"] == "symlink" and row["path"].startswith(package + "/")
                     for row in archive["files"]), "retained package alias")
    producers = reader.json("native/generation-inputs.json")
    producer_map = {}
    for row in producers["files"]:
        raw = reader.raw("native/" + row["copy"])
        original = reader.raw(row["path"])
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"] and raw == original,
             "selected producer/staged byte join differs: " + row["name"])
        need(row["name"] not in producer_map, "duplicate selected producer")
        producer_map[row["name"]] = row["sha256"]
    report["selected_generation"] = {"files": count, "bytes": total, "selected_producers": len(producer_map),
        "producers_cid": structured(producers), "full_transitive_dependency_attestation": False,
        "mutable_working_checkout_qualified": False}
    return producer_map


def positive_seconds(value, message):
    need(type(value) in {int, float} and math.isfinite(value) and value > 0, message)


def owner_reopen_baseline(before, after, reopens):
    """Allow only the native registry's explicitly counted owner generation step."""
    need(type(reopens) is int and reopens >= 1, "exact owner reopen count required")
    original, current = before["registry"]["meta"], after["registry"]["meta"]
    need(type(original) is list and type(current) is list and len(original) == len(current) == 1
         and type(original[0]) is list and type(current[0]) is list
         and len(original[0]) == len(current[0]) == 6
         and type(original[0][0]) is int and type(current[0][0]) is int
         and original[0][0] == current[0][0] == 1
         and type(original[0][5]) is int and original[0][5] >= 1
         and type(current[0][5]) is int and current[0][5] == original[0][5] + reopens,
         "native registry owner generation differs from the explicit reopen count")
    normalized = parse(wire(after))
    normalized["registry"]["meta"][0][5] = original[0][5]
    need(wire(normalized) == wire(before), "owner reopen changed another source/registry/model baseline field")
    return {"before_owner_generation": original[0][5], "after_owner_generation": current[0][5],
        "counted_owner_reopens": reopens, "all_other_owner_baseline_bytes_exact": True}


def audit_fresh_process_resume(reader, result, root, completion, state):
    """Join retained owner-process reports to the complete ordered page chain."""
    fresh = reader.json("native/fresh-process-resume.json", 1024**2)
    prefix = reader.json("native/prefix-pages.json", 64 * 1024)
    root_cid, completion_cid = result["scan_root_cid"], result["completed_scan_cid"]
    descriptors = completion["pages"]
    page_cids = [row["page_cid"] for row in descriptors]
    need(type(prefix) is list and len(prefix) == 2 and prefix == page_cids[:2],
         "fresh-process initial two-page prefix differs")
    need(fresh["qualified"] is True and type(fresh["post_setup_fit_attempt_count"]) is int
         and fresh["post_setup_fit_attempt_count"] == 0
         and type(fresh["pid"]) is int and fresh["pid"] > 1
         and fresh["completion_cid"] == completion_cid and fresh["coverage"] == completion["coverage"]
         and prefix + fresh["pages_created"] == page_cids
         and wire(fresh["numerical_before"]) == wire(fresh["numerical_after"]) == wire(state)
         and fresh["source_model_owner_preservation"] is True,
         "fresh-process durable continuation differs")
    resources(fresh["final_resources"])
    positive_seconds(fresh["recorded_seconds"], "fresh-process elapsed time must be recorded")
    if fresh["schema"] == "inventory-resume-fresh-process@1":
        return {"schema": fresh["schema"], "pids": [fresh["pid"]], "owner_processes": 1,
            "pages_created": fresh["pages_created"], "reported_no_fitting": True,
            "process_origin_attested": False}
    need(fresh["schema"] == "inventory-resume-fresh-process@2" and fresh["complete"] is True
         and fresh["root_cid"] == root_cid and fresh["version_id"] == root["model"]["version_id"],
         "bounded fresh-process aggregate profile differs")
    runs, pids = fresh["process_runs"], fresh["pids"]
    need(type(runs) is list and len(runs) == 4 and type(pids) is list and len(pids) == 4
         and all(type(pid) is int and pid > 1 for pid in pids) and len(set(pids)) == 4
         and pids == [run["pid"] for run in runs] and fresh["pid"] == pids[-1],
         "four distinct recorded owner-process identities required")
    baseline = reader.json("native/owners-before-scans.json", 8 * 1024**2)
    reopened = reader.json("native/owners-after-resume.json", 8 * 1024**2)
    owner_transition = owner_reopen_baseline(baseline, reopened, 5)
    generation = owner_transition["before_owner_generation"]
    declared_transition = {"schema": "inventory-resume-registry-owner-reopen@1",
        "before": generation, "after": generation + 5, "reopens": 5,
        "child_generations": list(range(generation + 1, generation + 5)),
        "other_owner_fields_unchanged": True}
    need(wire(result["registry_owner_reopen_transition"]) == wire(declared_transition)
         and type(fresh["registry_owner_generation_before"]) is int
         and fresh["registry_owner_generation_before"] == generation + 1
         and type(fresh["registry_owner_generation_after"]) is int
         and fresh["registry_owner_generation_after"] == generation + 4,
         "aggregate registry owner lifecycle differs from exact native reopens")
    cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": root_cid,
        "next_offset": descriptors[1]["end"], "previous_page_cid": page_cids[1]}
    pages, elapsed, chunk_observations = [], 0, []
    for number, chunk in enumerate(runs, 1):
        retained = reader.json(f"native/fresh-process-run-{number:02d}.json", 256 * 1024)
        request = reader.json(f"native/resume-request-{number:02d}.json", 64 * 1024)
        need(wire(chunk) == wire(retained), "aggregate child differs from numbered final report")
        need(set(request) == {"root_cid", "cursor", "version_id", "timeout_seconds", "max_pages", "run_number"}
             and request["root_cid"] == root_cid and request["version_id"] == root["model"]["version_id"]
             and type(request["run_number"]) is int and request["run_number"] == number
             and type(request["max_pages"]) is int and request["max_pages"] == 2
             and wire(request["cursor"]) == wire(cursor), "exact bounded numbered child request differs")
        positive_seconds(request["timeout_seconds"], "bounded child request deadline required")
        need(request["timeout_seconds"] <= 420, "child request deadline exceeds fixture bound")
        selected = descriptors[2 + (number - 1) * 2:2 + number * 2]
        expected_pages = [row["page_cid"] for row in selected]
        next_cursor = None if number == 4 else {"schema": "codebase-inventory-resume-cursor@1",
            "root_cid": root_cid, "next_offset": selected[-1]["end"],
            "previous_page_cid": expected_pages[-1]}
        need(chunk["schema"] == "inventory-resume-fresh-process-chunk@1" and chunk["qualified"] is True
             and type(chunk["complete"]) is bool and chunk["complete"] is (number == 4)
             and chunk["pid"] == pids[number - 1]
             and type(chunk["post_setup_fit_attempt_count"]) is int and chunk["post_setup_fit_attempt_count"] == 0
             and chunk["root_cid"] == root_cid and chunk["version_id"] == root["model"]["version_id"]
             and type(chunk["run_number"]) is int and chunk["run_number"] == number
             and type(chunk["max_pages"]) is int and chunk["max_pages"] == 2
             and wire(chunk["request_cursor"]) == wire(cursor)
             and wire(chunk["next_cursor"]) == wire(next_cursor)
             and wire(chunk["timeout_seconds"]) == wire(request["timeout_seconds"])
             and chunk["pages_created"] == expected_pages and len(expected_pages) == 2
             and chunk["prefix_tail_cid"] == expected_pages[-1]
             and wire(chunk["numerical_before"]) == wire(chunk["numerical_after"]) == wire(state)
             and type(chunk["registry_owner_generation_before"]) is int
             and chunk["registry_owner_generation_before"] == generation + number
             and type(chunk["registry_owner_generation_after"]) is int
             and chunk["registry_owner_generation_after"] == generation + number
             and chunk["source_model_owner_preservation"] is True,
             "bounded child cursor/pages/model/no-fitting closure differs")
        resources(chunk["final_resources"])
        positive_seconds(chunk["recorded_seconds"], "child elapsed time must be recorded")
        if number == 4:
            need(chunk["completion_cid"] == completion_cid and chunk["coverage"] == completion["coverage"],
                 "final child complete scan differs")
        else:
            need("completion_cid" not in chunk and "coverage" not in chunk,
                 "partial child must not claim completed scan coverage")
        stdout = reader.raw(f"native/fresh-process-{number:02d}.stdout", 1024**2)
        lines = [line for line in stdout.splitlines() if line.strip()]
        need(lines and wire(parse(lines[-1])) == wire({"qualified": True, "recorded_seconds": chunk["recorded_seconds"]}),
             "actual child CLI stdout does not join final numbered report")
        stderr = reader.raw(f"native/fresh-process-{number:02d}.stderr", 1024**2)
        need(not stderr, "completed child retained stderr requires review")
        pages.extend(expected_pages); elapsed += chunk["recorded_seconds"]
        chunk_observations.append({"run_number": number, "pid": chunk["pid"],
            "registry_owner_generation_before": chunk["registry_owner_generation_before"],
            "registry_owner_generation_after": chunk["registry_owner_generation_after"],
            "request_cursor": cursor, "next_cursor": next_cursor, "pages_created": expected_pages,
            "recorded_seconds": chunk["recorded_seconds"], "stdout_sha256": sha(stdout),
            "stderr_sha256": sha(stderr), "native_final_resources": chunk["final_resources"]})
        cursor = next_cursor
    need(pages == fresh["pages_created"] and len(pages) == 8 and cursor is None
         and fresh["recorded_seconds"] >= elapsed, "bounded aggregate page/time conservation differs")
    return {"schema": fresh["schema"], "pids": pids, "owner_processes": 4,
        "pages_created": pages, "chunks": chunk_observations, "reported_no_fitting": True,
        "registry_owner_reopen_transition": declared_transition,
        "full_native_owner_reopen_baseline": owner_transition,
        "all_chunk_current_source_model_closures_retained": True, "process_origin_attested": False}


def audit_setup_seed(reader, result, root, producers, report):
    """Receive one fixed closed setup through original, staged and native bytes."""
    if "setup_reuse" not in result:
        need(result.get("inherited_actual_setup_epochs", 0) == 0, "undeclared inherited setup epochs")
        return
    seed = reader.json("source-setup-seed.json", 1024**2)
    need(wire(seed) == wire(reader.json("native/source-setup-seed.json", 1024**2)),
         "outer and native setup seed receipts differ")
    need(set(seed) == {"schema", "qualified", "source_namespace", "source_pins", "producer_pins",
        "copied_members", "copied_bytes", "copied_files", "head", "root_version_id", "child_version_id",
        "checkpoint_states", "inherited_actual_setup_epochs", "new_fitting_epochs", "unknown_fitting_epochs", "authority"}
         and seed["schema"] == "inventory-resume-closed-setup-seed@1" and seed["qualified"] is False
         and seed["unknown_fitting_epochs"] is False and set(seed["authority"]) == SETUP_AUTHORITY
         and all(flag is False for flag in seed["authority"].values())
         and type(seed["inherited_actual_setup_epochs"]) is int and seed["inherited_actual_setup_epochs"] == 2
         and type(seed["new_fitting_epochs"]) is int and seed["new_fitting_epochs"] == 0,
         "closed setup seed must remain inert with exact inherited epoch accounting")
    source_path = reader.root.parent / SETUP_SOURCE_NAME
    need(seed["source_namespace"] == str(source_path), "only the fixed independently audited closed06 setup is supported")
    source = Reader(source_path, min(300, reader.deadline - time.monotonic()))
    siblings = Reader(reader.root.parent, min(300, reader.deadline - time.monotonic()))
    audit_name = SETUP_SOURCE_NAME + "-readonly-audit-20261003-01.json"
    audit_raw = siblings.raw(audit_name, 8 * 1024**2)
    need(sha(audit_raw) == SETUP_SOURCE_AUDIT_SHA256, "fixed closed06 independent preservation receipt differs")
    old_audit = parse(audit_raw)
    need(old_audit["qualified"] is False and old_audit["primary_archive"]["preserved"] is True,
         "seed source cannot claim previous native worker qualification")
    historical_files = {row["path"]: row for row in old_audit["primary_archive"]["before"]["files"] if row["kind"] == "file"}
    expected_pins = {"container_final": "container-execution-final.json", "result": "native/result.json",
        "generation": "native/generation-inputs.json", "owners": "native/owners-before-scans.json",
        "captured": "native/captured-artifacts-before-scans.json"}
    need(set(seed["source_pins"]) == set(expected_pins), "complete original setup observation pins required")
    original = {}
    def historical_source(locator, maximum):
        raw = source.raw(locator, maximum)
        metadata = source.path(locator).stat()
        historical = historical_files[locator]
        need(sha(raw) == historical["sha256"] and len(raw) == historical["bytes"]
             and stat.S_IMODE(metadata.st_mode) == historical["mode"]
             and metadata.st_mtime_ns == historical["mtime_ns"],
             "fixed closed source bytes/mode/mtime differ from its independent preservation receipt")
        return raw
    for role, locator in expected_pins.items():
        pin = seed["source_pins"][role]
        raw = historical_source(locator, 8 * 1024**2)
        need(set(pin) == {"path", "bytes", "sha256"} and pin["path"] == locator
             and type(pin["bytes"]) is int and len(raw) == pin["bytes"]
             and sha(raw) == pin["sha256"] == historical_files[locator]["sha256"],
             "original closed setup observation pin differs")
        original[role] = parse(raw)
    previous, container = original["result"], original["container_final"]
    need(previous["qualified"] is False and previous["known_actual_setup_epochs"] == 2
         and previous["unknown_fitting_epochs"] is False and previous["post_setup_fit_attempt_count"] == 0
         and container["container_removed"] is True and container["host_reservation_released"] is True,
         "original closed source setup fitting/lifecycle differs")
    resources(previous["final_resources"]); resources(container["host_resources_after_cleanup"])
    need(wire(seed["head"]) == wire(root["head"]) == wire(previous["head"])
         and seed["child_version_id"] == root["model"]["version_id"] == previous["selected_version_id"]
         and seed["root_version_id"] != seed["child_version_id"], "reused complete source/model selection differs")
    attempts = previous["setup_training_attempts"]
    need(len(attempts) == 2 and [row["name"] for row in attempts] == ["root", "child"]
         and [row["version_id"] for row in attempts] == [seed["root_version_id"], seed["child_version_id"]]
         and all(type(row["actual_completed_epochs"]) is int and row["actual_completed_epochs"] == 1
                 and row["requested_epochs"] == 1 and row["unknown_actual_epochs_on_failure"] is False for row in attempts),
         "original setup epochs did not return for the declared lineage")
    need(set(seed["producer_pins"]) == set(SETUP_PRODUCERS), "exact six frozen setup producers required")
    generation = {row["name"]: row for row in original["generation"]["files"]}
    for name, expected in SETUP_PRODUCERS.items():
        raw = historical_source("datasets/" + name.replace(".", "/") + ".py", 4 * 1024**2)
        pin = seed["producer_pins"][name]
        need(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and len(raw) == pin["bytes"]
             and sha(raw) == pin["sha256"] == expected == producers[name]
             and generation[name]["bytes"] == len(raw) and generation[name]["sha256"] == expected,
             "reused setup producer differs from original or selected generation")
    members = seed["copied_members"]
    need(type(members) is list and 0 < len(members) <= 4096
         and type(seed["copied_files"]) is int and seed["copied_files"] == len(members)
         and type(seed["copied_bytes"]) is int and 0 < seed["copied_bytes"] <= 128 * 1024**2,
         "bounded copied setup membership required")
    baseline = {"private/source-artifacts/" + row["path"]: row for row in original["captured"]}
    baseline.update({"private/model-artifacts/" + row["path"]: row for row in original["owners"]["model_artifacts"]})
    expected_members = set(baseline) | {"private/source.duckdb", "private/model.duckdb", "root.json", "child.json"}
    expected_members.update(path.removeprefix("native/") for path in historical_files if path.startswith("native/repository/"))
    names = [row["path"] for row in members]
    need(names == sorted(expected_members) and len(set(names)) == len(names)
         and seed["copied_bytes"] == sum(row["bytes"] for row in members), "exact original setup member/totals conservation differs")
    actual_stage = set()
    for directory, dirs, filenames in os.walk(reader.path("setup-seed"), followlinks=False):
        reader.tick()
        need(not any((Path(directory) / name).is_symlink() for name in dirs), "setup seed directory alias")
        for name in filenames:
            actual_stage.add((Path(directory) / name).relative_to(reader.path("setup-seed")).as_posix())
            need(len(actual_stage) <= 4096, "staged setup member bound exceeded")
    need(actual_stage == expected_members, "retained staged setup has extra/missing copied members")
    for row in members:
        need(set(row) == {"path", "bytes", "sha256", "mode"} and type(row["bytes"]) is int
             and type(row["mode"]) is int and 0 <= row["mode"] <= 0o777,
             "closed copied setup member required")
        locator = "native/" + row["path"]
        raw = historical_source(locator, 16 * 1024**2)
        staged = reader.raw("setup-seed/" + row["path"], 16 * 1024**2)
        need(raw == staged and len(raw) == row["bytes"]
             and sha(raw) == row["sha256"] == historical_files[locator]["sha256"]
             and stat.S_IMODE(reader.path("setup-seed/" + row["path"]).stat().st_mode) == row["mode"],
             "original/staged setup exact bytes/modes differ")
        if row["path"] in baseline:
            need(row["bytes"] == baseline[row["path"]]["bytes"] and row["sha256"] == baseline[row["path"]]["sha256"],
                 "captured source/model baseline differs from staged setup")
            native = reader.raw(locator, 16 * 1024**2)
            need(native == raw, "retained native original source/model artifact differs from seed")
    materialized = reader.json("native/setup-seed-materialization.json", 1024**2)
    need(wire(materialized) == wire(result["setup_reuse"])
         and materialized["schema"] == "inventory-resume-materialized-setup-seed@1"
         and materialized["qualified"] is False and materialized["seed_receipt_sha256"] == sha(wire(seed))
         and materialized["output"] == "/results/native"
         and materialized["unknown_fitting_epochs"] is False
         and type(materialized["inherited_actual_setup_epochs"]) is int and materialized["inherited_actual_setup_epochs"] == 2
         and type(materialized["new_fitting_epochs"]) is int and materialized["new_fitting_epochs"] == 0
         and set(materialized["authority"]) == SETUP_AUTHORITY
         and all(flag is False for flag in materialized["authority"].values()),
         "native materialization receipt identity/cost/authority differs")
    for key in ("copied_members", "head", "root_version_id", "child_version_id", "checkpoint_states"):
        need(wire(materialized[key]) == wire(seed[key]), "native materialization lost exact seed identity")
    need(set(seed["checkpoint_states"]) == {"root", "child"}, "both inherited full checkpoint states required")
    for role in ("root", "child"):
        raw_record = reader.raw("native/" + role + ".json", 16 * 1024**2)
        need(raw_record == historical_source("native/" + role + ".json", 16 * 1024**2), "original/native training record differs")
        record = parse(raw_record)
        artifact = record["registry_artifact"]
        saved_raw = reader.raw("native/private/model-artifacts/" + artifact["sha256"][:2] + "/" + artifact["sha256"], 16 * 1024**2)
        saved = parse(saved_raw)
        epochs = 1 if role == "root" else 2
        state = {"artifact": artifact, "state_sha256": sha(wire(saved["state"])), "report_sha256": sha(wire(saved["report"])),
            "completed_epochs": saved["state"]["completed_epochs"], "adam_steps": [row["step"] for row in saved["state"]["adam"]],
            "latent_width": saved["state"]["latent_width"], "feature_columns": len(saved["feature_space"]["columns"])}
        need(wire(state) == wire(seed["checkpoint_states"][role]) and state["completed_epochs"] == epochs
             and all(step == epochs for step in state["adam_steps"])
             and record["version_id"] == seed[role + "_version_id"] and wire(record["head"]) == wire(seed["head"])
             and record["parent_version_id"] == (None if role == "root" else seed["root_version_id"])
             and cid(saved_raw) == record["checkpoint_raw_cid"] and saved_raw == wire(saved)
             and record["state_sha256"] == state["state_sha256"]
             and record["feature_space_sha256"] == sha(wire(saved["feature_space"]))
             and record["contract_sha256"] == saved["state"]["contract_sha256"]
             and record["contract_sha256"] == saved["report"]["contract_sha256"]
             and record["contract_sha256"] == modality_digest(saved["contract"])
             and record["report_json"] == wire(saved["report"]).decode(),
             "full inherited native checkpoint/optimizer/report bindings differ")
    owners = reader.json("native/owners-before-scans.json", 8 * 1024**2)
    transition = owner_reopen_baseline(original["owners"], owners, 1)
    need(wire(reader.json("native/captured-artifacts-before-scans.json", 1024**2)) == wire(original["captured"])
         and result["reused_setup_state_native_checked"] is True,
         "fresh native receiving owner/source baseline observation differs from inherited setup")
    report["setup_seed"] = {"source_namespace": str(source_path), "source_audit_sha256": SETUP_SOURCE_AUDIT_SHA256,
        "seed_receipt_sha256": sha(wire(seed)), "copied_files": len(members), "copied_bytes": seed["copied_bytes"],
        "checkpoint_states": seed["checkpoint_states"], "inherited_actual_setup_epochs": 2, "new_fitting_epochs": 0,
        "fixed_source_original_and_staged_bytes_verified": True, "owner_reopen_transition": transition,
        "fresh_native_baseline_preserved_except_explicit_owner_reopen": True,
        "ending_native_database_byte_equality_claimed": False, "native_owners_opened_by_auditor": False,
        "original_recorded_seconds": previous["recorded_seconds"], "original_container_total_seconds": container["elapsed_seconds"],
        "source_guarded_artifact_pins": source.observed}


def audit_failed_resume_chunks(reader, result, producers, preflight, report):
    """Retain genuine partial numerical pages without granting completion."""
    paths = sorted(reader.path("native").glob("fresh-process-run-*.json"))
    if not paths:
        return
    need(len(paths) <= 4 and [path.name for path in paths] == [f"fresh-process-run-{number:02d}.json"
        for number in range(1, len(paths) + 1)], "failed numbered resume reports exceed their declared bounded profile")
    root = reader.json("native/scan-root.json", 4 * 1024**2)
    root_cid = structured(root)
    need(reader.cas(root_cid) == root and root["head"] == result["head"]
         and root["model"]["version_id"] == result["selected_version_id"], "failed child root identity differs")
    prefix = reader.json("native/prefix-pages.json", 64 * 1024)
    need(type(prefix) is list and len(prefix) == 2, "failed bounded resume requires retained initial two pages")
    cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": root_cid,
        "next_offset": 64, "previous_page_cid": prefix[-1]}
    observations = []
    for number, path in enumerate(paths, 1):
        chunk = reader.json("native/" + path.name, 256 * 1024)
        request = reader.json(f"native/resume-request-{number:02d}.json", 64 * 1024)
        need(chunk["schema"] == "inventory-resume-fresh-process-chunk@1"
             and type(chunk["qualified"]) is bool and type(chunk["complete"]) is bool
             and type(chunk["pid"]) is int and chunk["pid"] > 1
             and type(chunk["post_setup_fit_attempt_count"]) is int and chunk["post_setup_fit_attempt_count"] == 0
             and chunk["root_cid"] == request["root_cid"] == root_cid
             and chunk["version_id"] == request["version_id"] == root["model"]["version_id"]
             and type(chunk["run_number"]) is int and chunk["run_number"] == request["run_number"] == number
             and type(chunk["max_pages"]) is int and chunk["max_pages"] == request["max_pages"] == 2
             and wire(chunk["request_cursor"]) == wire(request["cursor"]) == wire(cursor),
             "failed child exact request/cost/root identity differs")
        created = chunk["pages_created"]
        need(type(created) is list and 0 < len(created) <= 2, "failed child must retain a bounded positive page prefix")
        positive = []
        for page_cid in created:
            page = reader.cas(page_cid, raw=True)
            need(page["schema"] == "codebase-inventory-resume-page@1" and page["root_cid"] == root_cid
                 and page["head_cid"] == root["head_cid"] and page["membership_cid"] == root["membership_cid"]
                 and page["model_artifact_cid"] == root["model"]["artifact_cid"]
                 and page["start"] == cursor["next_offset"] and page["end"] == min(300, page["start"] + 32)
                 and page["previous_page_cid"] == cursor["previous_page_cid"], "failed child native page prefix differs")
            false_authority(page["authority"])
            receipt, inference = page["worker_receipt"], page["inference"]
            need(type(receipt) is dict and type(receipt["returncode"]) is int and receipt["returncode"] == 0
                 and receipt["workspace_cleaned"] is True and receipt["source_execution_attested"] is False
                 and receipt["worker_sha256"] == producers["ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"]
                 and receipt["executable_sha256"] == preflight["executable"]["sha256"]
                 and receipt["input_bytes"] > 0 and receipt["output_bytes"] > 0
                 and inference["training_executed"] is False and inference["decoded_formulas_generated"] is False
                 and type(page["coverage"]["inferred_rows"]) is int and page["coverage"]["inferred_rows"] > 0,
                 "failed child partial page lacks genuine positive advisory worker receipt")
            positive.append({"page_cid": page_cid, "start": page["start"], "end": page["end"],
                "inferred_rows": page["coverage"]["inferred_rows"], "worker_receipt": receipt})
            cursor = None if page["end"] == 300 else {"schema": "codebase-inventory-resume-cursor@1",
                "root_cid": root_cid, "next_offset": page["end"], "previous_page_cid": page_cid}
        stdout = reader.raw(f"native/fresh-process-{number:02d}.stdout", 1024**2)
        stderr = reader.raw(f"native/fresh-process-{number:02d}.stderr", 1024**2)
        if chunk["qualified"] is False:
            lines = [line for line in stderr.splitlines() if line.strip()]
            need(lines and wire(parse(lines[-1])) == wire({"qualified": False,
                "error_type": chunk["error_type"], "error": chunk["error"]}), "actual failed child stderr does not join its final report")
        observations.append({"report": chunk, "positive_partial_native_pages": positive,
            "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
            "native_owner_closing_validation_completed": chunk.get("source_model_owner_preservation") is True,
            "numerical_execution_independently_reperformed": False, "completion_qualified": False})
    report["failed_fresh_process_observations"] = observations


def audit_reused_completed_scan(reader, result, root, completion, state, report):
    """Join historical page executions to fresh receiving without inventing reruns."""
    seed = reader.json("source-scan-seed.json", 1024**2)
    expected_fields = {"schema", "qualified", "source_namespace", "source_audit", "source_scan_summary", "root_cid",
        "completion_cid", "head", "selected_version_id", "coverage", "opt_out_equivalence", "producers",
        "checkpoint_states", "copied_members", "copied_files", "copied_bytes", "inherited_actual_setup_epochs",
        "new_fitting_epochs", "new_scan_pages", "new_registry_reopens", "unknown_fitting_epochs",
        "fresh_native_validation_required", "authority"}
    need(set(seed) == expected_fields and seed["schema"] == "inventory-resume-closed-scan-seed@1"
         and seed["qualified"] is False and seed["unknown_fitting_epochs"] is False
         and seed["fresh_native_validation_required"] is True
         and set(seed["authority"]) == SCAN_SEED_AUTHORITY
         and all(value is False for value in seed["authority"].values()), "exact advisory completed scan seed required")
    need(wire(seed) == wire(reader.json("native/source-scan-seed.json", 1024**2)) == wire(result["source_scan_seed"]),
         "host/native/result scan seed differs")
    for name, expected in (("inherited_actual_setup_epochs", 2), ("new_fitting_epochs", 0),
            ("new_scan_pages", 0), ("new_registry_reopens", 0)):
        need(type(seed[name]) is int and seed[name] == expected, "scan seed cost or owner lifecycle differs")
    source_path = reader.root.parent / SCAN_SOURCE_NAME
    need(seed["source_namespace"] == str(source_path), "exact closed09 scan source required")
    source = Reader(source_path, max(0.001, reader.deadline - time.monotonic()))
    siblings = Reader(reader.root.parent, max(0.001, reader.deadline - time.monotonic()))
    audit_name = SCAN_SOURCE_NAME + "-readonly-audit-20261003-02.json"
    audit_raw = siblings.raw(audit_name, 16 * 1024**2)
    need(sha(audit_raw) == SCAN_SOURCE_AUDIT_SHA256
         and seed["source_audit"] == {"path": str(reader.root.parent / audit_name), "bytes": len(audit_raw),
                                     "sha256": sha(audit_raw)}, "fixed positive source09 scan audit differs")
    original_audit = parse(audit_raw)
    source_before = source.whole_archive()
    need(original_audit["qualified"] is False and original_audit["namespace"] == str(source_path)
         and original_audit["primary_archive"]["preserved"] is True
         and wire(source_before) == wire(original_audit["primary_archive"]["before"]),
         "closed09 whole archive changed since positive scan receiving")
    positive = original_audit["positive_completed_scan"]
    need(positive["schema"] == "inventory-resume-positive-completed-scan-observation@1"
         and positive["verified"] is True and positive["overall_native_qualification"] is False
         and positive["native_worker_qualified"] is False and positive["source_namespace"] == str(source_path)
         and positive["native_owners_opened_by_auditor"] is False
         and positive["numerical_execution_independently_reperformed"] is False
         and positive["current_deployment_freshness_attested"] is False and positive["process_origin_attested"] is False,
         "source scan observation cannot grant current execution or freshness authority")
    false_authority(positive["authority"])
    for key, value in (("root_cid", result["scan_root_cid"]), ("completion_cid", result["completed_scan_cid"]),
            ("head", root["head"]), ("coverage", completion["coverage"]), ("membership_cid", root["membership_cid"]),
            ("model", root["model"]), ("numerical_state", state), ("ordered_pages", completion["pages"])):
        need(wire(positive[key]) == wire(value), "source completed scan differs from current retained scan: " + key)
    need(seed["root_cid"] == positive["root_cid"] and seed["completion_cid"] == positive["completion_cid"]
         and wire(seed["head"]) == wire(root["head"]) and seed["selected_version_id"] == root["model"]["version_id"]
         and wire(seed["coverage"]) == wire(completion["coverage"])
         and wire(seed["opt_out_equivalence"]) == wire(positive["opt_out_equivalence"])
         and wire(seed["checkpoint_states"]) == wire(positive["full_checkpoint_states"])
             == wire(report["setup_seed"]["checkpoint_states"]), "seed source/model/full optimizer or reference binding differs")
    expected_summary = {"schema": positive["schema"], "root_cid": positive["root_cid"],
        "completion_cid": positive["completion_cid"], "membership_cid": positive["membership_cid"],
        "numerical_state": positive["numerical_state"], "fresh_process_resume": positive["fresh_process_resume"],
        "registry_owner_reopen_transition": positive["fresh_process_resume"]["registry_owner_reopen_transition"],
        "historical_execution_only": True, "overall_native_qualification": False, "native_worker_qualified": False}
    need(wire(seed["source_scan_summary"]) == wire(expected_summary), "historical scan summary differs")
    expected_producers = {name: {key: pin[key] for key in ("bytes", "sha256")}
        for name, pin in positive["implementation_files"].items()}
    need(wire(seed["producers"]) == wire(expected_producers) and len(expected_producers) == 17,
         "complete seventeen source scan producer pins required")
    for name, pin in positive["implementation_files"].items():
        need(pin["path"] == "datasets/" + name.replace(".", "/") + ".py"
             and pin["sha256"] == root["implementation"]["files"][name], "source producer locator/root pin differs")
        original = source.raw(pin["path"], 4 * 1024**2)
        current = reader.raw(pin["path"], 4 * 1024**2)
        need(original == current and len(current) == pin["bytes"] and sha(current) == pin["sha256"],
             "source and current staged scan producer generation differs")
    for name, checksum in SETUP_PRODUCERS.items():
        need(positive["setup_producer_pins"][name]["sha256"] == expected_producers[name]["sha256"] == checksum,
             "six checkpoint producers changed during scan reuse")
    original_result = source.json("native/result.json", 1024**2)
    original_container = source.json("container-execution-final.json", 1024**2)
    need(source.observed["native/result.json"] == positive["source_result"]
         and source.observed["container-execution-final.json"] == positive["source_container"]
         and original_result["qualified"] is False and original_result["error_type"] == "StructuredIdentityError"
         and original_container["container_removed"] is True and original_container["host_reservation_released"] is True,
         "source failed-result/container byte/lifecycle join differs")
    resources(original_result["final_resources"]); resources(original_container["host_resources_after_cleanup"])
    expected_members = []
    for row in positive["transport_artifacts"]:
        expected_members.append({"path": "cas/" + row["path"].removeprefix("private/source-artifacts/"),
            "bytes": row["bytes"], "sha256": row["sha256"], "mode": row["mode"], "kind": "cas",
            "cid": row["cid"], "codec": row["codec"], "role": row["role"]})
    for locator in (*SCAN_PROVENANCE, "independent-audit.json"):
        raw = audit_raw if locator == "independent-audit.json" else source.raw(locator, 16 * 1024**2)
        path = siblings.path(audit_name) if locator == "independent-audit.json" else source.path(locator)
        expected_members.append({"path": "provenance/" + locator, "bytes": len(raw), "sha256": sha(raw),
            "mode": stat.S_IMODE(path.stat().st_mode), "kind": "provenance"})
    expected_members.sort(key=lambda row: row["path"])
    need(len(expected_members) == 41 and wire(seed["copied_members"]) == wire(expected_members)
         and type(seed["copied_files"]) is int and seed["copied_files"] == 41
         and type(seed["copied_bytes"]) is int and seed["copied_bytes"] == sum(row["bytes"] for row in expected_members)
         and seed["copied_bytes"] <= 128 * 1024**2, "exact fourteen CAS and bounded provenance-only transport required")
    staged = set()
    for directory, names, files in os.walk(reader.path("scan-seed"), followlinks=False):
        reader.tick()
        need(not any((Path(directory) / name).is_symlink() for name in names), "scan seed directory alias")
        for name in files:
            path = Path(directory) / name
            need(not path.is_symlink(), "scan seed file alias")
            staged.add(path.relative_to(reader.path("scan-seed")).as_posix())
            need(len(staged) <= 128, "scan seed member bound exceeded")
    need(staged == {row["path"] for row in expected_members}, "scan seed copied extra or missing history/CAS members")
    for row in expected_members:
        raw = reader.raw("scan-seed/" + row["path"], 16 * 1024**2)
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"]
             and stat.S_IMODE(reader.path("scan-seed/" + row["path"]).stat().st_mode) == row["mode"],
             "host staged CAS or historical provenance byte/mode differs")
        if row["kind"] == "cas":
            locator = "native/private/source-artifacts/" + row["path"].removeprefix("cas/")
            need(reader.raw(locator, 16 * 1024**2) == source.raw(locator, 16 * 1024**2) == raw
                 and cid(raw, row["codec"]) == row["cid"], "original/staged/current CAS codec/bytes differ")
    materialized = reader.json("native/scan-seed-materialization.json", 1024**2)
    expected_materialized = {"schema": "inventory-resume-materialized-scan-seed@1", "qualified": False,
        "seed_receipt_sha256": sha(wire(seed)), "output": "/results/native/private/source-artifacts",
        **{key: seed[key] for key in ("root_cid", "completion_cid", "head", "selected_version_id", "coverage",
             "opt_out_equivalence", "source_namespace", "source_audit", "source_scan_summary", "producers", "checkpoint_states")},
        "copied_cas_objects": 14, "copied_cas_bytes": sum(row["bytes"] for row in expected_members if row["kind"] == "cas"),
        "inherited_actual_setup_epochs": 2, "new_fitting_epochs": 0, "new_scan_pages": 0, "new_registry_reopens": 0,
        "unknown_fitting_epochs": False, "fresh_native_validation_required": True, "authority": seed["authority"]}
    need(wire(materialized) == wire(expected_materialized) == wire(result["scan_reuse"]),
         "native materialized scan receipt/result differs")
    need(type(result["new_scan_pages_created"]) is int and result["new_scan_pages_created"] == 0
         and type(result["inherited_scan_pages"]) is int and result["inherited_scan_pages"] == 10
         and "fresh_process_resume" not in result and "registry_owner_reopen_transition" not in result,
         "current run cannot claim old numerical pages or owner restarts as new work")
    baseline = reader.json("native/owners-before-scans.json", 8 * 1024**2)
    ending = reader.json("native/owners-after-resume.json", 8 * 1024**2)
    need(wire(baseline) == wire(ending) and type(baseline["registry"]["meta"][0][5]) is int
         and baseline["registry"]["meta"][0][5] == 2, "fresh current receiving must preserve exact owner generation and full state")
    phases = [row for row in result["phases"] if row["name"] == "receive_reused_completed_chain"]
    need(len(phases) == 1 and phases[0]["status"] == "completed", "fresh full completed scan receiving was not recorded")
    resources(result["final_resources"])
    source_after = source.whole_archive()
    need(source_before == source_after, "closed09 changed during composed scan receiving audit")
    report["scan_reuse"] = {"schema": "inventory-resume-composed-scan-receiving-audit@1", "source_namespace": str(source_path),
        "source_audit_sha256": sha(audit_raw), "source_result": positive["source_result"],
        "source_container": positive["source_container"], "copied_cas_objects": 14,
        "copied_cas_bytes": expected_materialized["copied_cas_bytes"], "copied_provenance_members": 27,
        "current_owner_generation": 2, "current_full_owner_state_unchanged": True,
        "new_numerical_pages": 0, "new_owner_reopens": 0, "fresh_completed_chain_receiving_recorded": True,
        "historical_source_owner_reopens": positive["fresh_process_resume"]["registry_owner_reopen_transition"],
        "historical_preworker_controls": positive["seven_preworker_controls"],
        "source_archive_preserved": True, "source_archive_inventory_cid": source_after["inventory_cid"],
        "numerical_execution_independently_reperformed": False, "native_owners_opened_by_auditor": False}
    historical = positive["fresh_process_resume"]
    return {"schema": "inventory-resume-inherited-scan-provenance@1", "source_namespace": str(source_path),
        "pids": historical["pids"], "owner_processes": 0, "inherited_owner_processes": 4,
        "pages_created": [], "inherited_pages": historical["pages_created"],
        "source_fresh_process_resume": historical, "reported_no_fitting": True, "process_origin_attested": False}


def audit_scan(reader, result, producers, preflight, report):
    root_cid, completion_cid = result["scan_root_cid"], result["completed_scan_cid"]
    root = reader.cas(root_cid)
    completion = reader.cas(completion_cid)
    need(root == reader.json("native/scan-root.json") and root["schema"] == "codebase-inventory-resume-root@1",
         "review root differs from native CAS")
    need(completion["schema"] == "codebase-inventory-resume-completion@1", "completion schema differs")
    need(root["head"] == result["head"] and root["head_cid"] == structured(root["head"]), "native head identity differs")
    members = root["members"]
    need(len(members) == 300 and root["membership_cid"] == structured(members)
         and [row["raw_path_hex"] for row in members] == sorted({row["raw_path_hex"] for row in members}),
         "complete ordered 300-member root required")
    need(root["limits"]["max_inventory_entries"] == 512 and root["limits"]["page_entries"] == 32,
         "native larger-than-256 profile differs")
    false_authority(root["authority"]); false_authority(completion["authority"])
    model = root["model"]
    need(model["latent_width"] == 8 and model["version_id"] == result["selected_version_id"], "selected 8D model differs")
    checksum = model["artifact"]["sha256"]
    model_raw = reader.raw("native/private/model-artifacts/" + checksum[:2] + "/" + checksum, 16 * 1024**2)
    need(len(model_raw) == model["artifact"]["bytes"] and sha(model_raw) == checksum
         and cid(model_raw) == model["artifact_cid"], "selected model artifact byte identity differs")
    saved = parse(model_raw)
    need(sha(wire(saved["state"])) == model["state_sha256"]
         and sha(wire(saved["feature_space"])) == model["feature_space_sha256"]
         and modality_digest(saved["contract"]) == model["contract_sha256"], "frozen numerical model identity differs")
    state = {"state_sha256": sha(wire(saved["state"])), "completed_epochs": saved["state"]["completed_epochs"],
        "adam_steps": [item["step"] for item in saved["state"]["adam"]],
        "latent_width": saved["state"]["latent_width"], "feature_columns": len(saved["feature_space"]["columns"])}
    need(state == result["numerical_before"] == result["numerical_after"], "retained numerical before/after differs")
    audit_setup_seed(reader, result, root, producers, report)
    need(sha(wire(root["implementation"]["files"])) == root["implementation"]["sha256"], "implementation pin digest differs")
    for name, checksum in root["implementation"]["files"].items():
        retained_sha = producers.get(name)
        if retained_sha is None:
            need(name.startswith("ipfs_datasets_py."), "unreviewed implementation module locator")
            retained_sha = sha(reader.raw("datasets/" + name.replace(".", "/") + ".py"))
        need(retained_sha == checksum, "root implementation is not selected-generation pinned: " + name)
    source_manifest = reader.cas(root["head"]["manifest_cid"])
    snapshot = source_manifest["snapshot"]
    need(structured({key: value for key, value in snapshot.items() if key != "snapshot_cid"}) == snapshot["snapshot_cid"],
         "source snapshot identity differs")
    entries = snapshot["entries"]
    units = {row["source_key"]: row for row in source_manifest["units"]}
    derived = []
    for entry in entries:
        need(structured({key: value for key, value in entry.items() if key != "entry_cid"}) == entry["entry_cid"],
             "snapshot entry identity differs")
        key = "raw:" + entry["raw_path_hex"]
        unit = units[key]
        derived.append({"source_key": key, "path": entry["path"], "raw_path_hex": entry["raw_path_hex"],
            "entry_cid": entry["entry_cid"], "source_cid": entry["source_cid"], "ast_cid": unit["ast_cid"],
            "parse_status": unit["parse_status"], "source_size_bytes": entry["size_bytes"], "opaque_reason": entry["opaque_reason"]})
    need(derived == members, "native structural manifest/full membership join differs")
    common = {"root_cid": root_cid, "head_cid": root["head_cid"], "membership_cid": root["membership_cid"],
              "model_artifact_cid": model["artifact_cid"]}
    need(all(completion[key] == value for key, value in common.items()), "completion root/model binding differs")
    counts, inferred, offset, previous, positive, page_receipts = Counter(), 0, 0, None, 0, []
    worker_sha = producers["ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"]
    need(len(completion["pages"]) == 10, "300 sources require ten 32-entry native pages")
    for descriptor in completion["pages"]:
        page_cid = descriptor["page_cid"]
        page = reader.cas(page_cid, raw=True)
        need(page["schema"] == "codebase-inventory-resume-page@1"
             and all(page[key] == value for key, value in common.items()), "page root/model identity differs")
        end = min(offset + 32, 300)
        selected = members[offset:end]
        need(page["start"] == offset and page["end"] == end and page["total_entries"] == 300
             and page["previous_page_cid"] == previous and page["page_membership_cid"] == structured(selected),
             "ordered native page prefix differs")
        ledger = page["entries"]
        need([row["member_index"] for row in ledger] == list(range(offset, end))
             and [(row["source_key"], row["entry_cid"]) for row in ledger]
                 == [(row["source_key"], row["entry_cid"]) for row in selected], "complete page membership differs")
        disposition_counts = dict(sorted(Counter(row["disposition"] for row in ledger).items()))
        n = disposition_counts.get("inferred", 0)
        coverage = {"inventory_entries": end - offset, "inferred_rows": n, "dispositions": disposition_counts}
        expected = {"page_cid": page_cid, "start": offset, "end": end,
            "membership_cid": page["page_membership_cid"], "inferred_rows": n, "dispositions": disposition_counts}
        need(page["coverage"] == coverage and descriptor == expected, "page coverage/descriptor differs")
        false_authority(page["authority"])
        need(n > 0, "each native page must contain positive inferred rows")
        receipt, inference = page["worker_receipt"], page["inference"]
        need(type(receipt) is dict and type(inference) is dict and type(receipt["returncode"]) is int
             and receipt["returncode"] == 0 and receipt["workspace_cleaned"] is True
             and receipt["source_execution_attested"] is False and receipt["worker_sha256"] == worker_sha
             and receipt["executable_sha256"] == preflight["executable"]["sha256"], "positive native page worker receipt differs")
        need(receipt["memory_enforcement"] == "sampled_process_tree_rss_with_possible_overshoot"
             and 0 < receipt["input_bytes"] <= receipt["limits"]["max_input_bytes"] == root["limits"]["max_input_bytes"]
             and 0 < receipt["output_bytes"] <= receipt["limits"]["max_output_bytes"] == root["limits"]["max_output_bytes"]
             and type(receipt["elapsed_ms"]) is int and receipt["elapsed_ms"] >= 0,
             "native numerical byte/resource receipt differs")
        need(inference["schema"] == "native-projection-feature-inference/v1"
             and inference["training_executed"] is False and inference["decoded_formulas_generated"] is False
             and inference["representation"] == "native_compiler_structural_features_not_semantic_text_embeddings",
             "numerical advisory representation differs")
        for key in ("contract_sha256", "state_sha256", "feature_space_sha256"):
            need(inference[key] == model[key], "numerical page selected model differs")
        selected_inferred = [row for row in ledger if row["disposition"] == "inferred"]
        need(len(inference["rows"]) == n and inference["coverage"] == [item for row in selected_inferred for item in row["coverage"]],
             "numerical rows/coverage differ")
        for ordinal, (entry, row) in enumerate(zip(selected_inferred, inference["rows"])):
            need(entry["inference_index"] == ordinal and entry["source_digest"] == row["source_digest"]
                 and len(row["latent"]) == 8 and set(row["reconstructed_projection_features"]) == set(model["projection_ids"]),
                 "numerical row source/layout differs")
            numbers = list(row["latent"])
            for key, values in row["reconstructed_projection_features"].items():
                need(len(values) == model["projection_widths"][key], "numerical feature width differs")
                numbers.extend(values)
            need(all(type(value) in {int, float} and math.isfinite(value) for value in numbers), "nonfinite numerical row")
        counts.update(disposition_counts); inferred += n; positive += 1
        page_receipts.append({"page_cid": page_cid, "inferred_rows": n, "worker_receipt": receipt})
        offset, previous = end, page_cid
    coverage = {"inventory_entries": 300, "pages": 10, "inferred_rows": inferred, "dispositions": dict(sorted(counts.items()))}
    need(offset == 300 and completion["coverage"] == coverage == result["scan_coverage"], "complete 300-entry coverage conservation differs")
    if "scan_reuse" in result:
        fresh = audit_reused_completed_scan(reader, result, root, completion, state, report)
    else:
        fresh = audit_fresh_process_resume(reader, result, root, completion, state)
    for row in reader.json("native/captured-artifacts-before-scans.json"):
        raw = reader.raw("native/private/source-artifacts/" + row["path"])
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "original captured artifact pin differs")
    for row in reader.json("native/owners-before-scans.json")["model_artifacts"]:
        raw = reader.raw("native/private/model-artifacts/" + row["path"])
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "original model artifact pin differs")
    report["scan"] = {"root_cid": root_cid, "completion_cid": completion_cid, "coverage": coverage,
        "positive_page_workers": positive, "page_receipts": page_receipts,
        "fresh_process_pid": fresh["pids"][-1], "fresh_process_resume": fresh, "numerical_state": state,
        "raw_worker_input_output_not_retained": True, "numerical_execution_independently_reperformed": False}
    if "scan_reuse" in result:
        report["scan"].update(numerical_page_execution_scope="inherited closed09 artifact receipts",
            fresh_process_scope="historical closed09 observations; no new owner processes or numerical pages")
    return root, completion


def verify_authored_worker_receipt(output: Path, *, prepared: dict, task_cid: str) -> dict:
    """Join one real authored-worker stdout line after native STOP.

    This pure receiving gate reads only bounded ordinary files and public-key
    signatures. It grants no completion, freshness, model-owner or private
    evidence-epoch authority. The caller must hold its native lifecycle lease
    until actual STOP/UID cleanup; this function does not perform that cleanup.
    """
    reader = Reader(output, 120, native_output=True)
    need(type(prepared) is dict and prepared.get("task_cid") == task_cid, "exact prepared task descriptor required")
    original = reader.raw(prepared["artifact"], 8 * 1024**2)
    public = parse(original)
    need(sha(original) == prepared["sha256"] and not reader.path(prepared["artifact"]).stat().st_mode & 0o222,
         "original prepared artifact bytes/mode differ")
    need(public.get("schema") == "supervisor-public-instruction@3" and public["task_cid"] == task_cid
         and public["context_cid"] == prepared["context_cid"]
         and public["context_cid"] == structured({key: value for key, value in public.items() if key != "context_cid"}),
         "public inventory artifact identity differs")
    manifest = signature(public["manifest"])
    plan = public["inventory_plan_admission"]
    planning = signature(plan["receipt"])
    need(manifest["schema"] == "supervisor-local-benchmark-manifest@5"
         and planning["schema"] == "supervisor-local-planning-receipt@3"
         and public["manifest"]["binding"]["identity"] == plan["receipt"]["binding"]["identity"] == public["owner_identity"]
         and public["manifest"]["binding"]["profile_id"] == plan["receipt"]["binding"]["profile_id"] == public["owner_profile_id"],
         "exact historical inventory public signature profiles differ")
    context, graph = public["codebase_inventory_context"], plan["graph"]
    scan = context["scan"]
    false_authority(context["authority"]); false_authority(scan["authority"])
    need(context["schema"] == "supervisor-codebase-inventory-context@1" and context["evidence"] is None
         and scan["root_cid"] == structured(scan["root_record"])
         and scan["completion_cid"] == structured(scan["completion_record"])
         and scan["membership_cid"] == structured(scan["members"])
         and len(scan["members"]) == 300 and scan["coverage"]["inventory_entries"] == 300,
         "full authored inventory context identity/300-member closure differs")
    full_cid = structured(context)
    tasks = {task["task_key"]: task for task in graph["tasks"]}
    task_cids = {key: prompt_task_record_cid(task) for key, task in tasks.items()}
    need(set(tasks) == {"INVENTORY-TYPE", "INVENTORY-OFFSET"}
         and task_cids["INVENTORY-OFFSET"] == task_cid
         and tasks["INVENTORY-OFFSET"]["dependency_task_cids"] == [task_cids["INVENTORY-TYPE"]],
         "authored public task population/dependency differs")
    need(planning["manifest_cid"] == structured(public["manifest"]) == public["manifest_cid"] == prepared["manifest_cid"]
         and planning["graph_cid"] == structured(semantic(graph))
         and planning["administrator_task_cids"] == sorted(task_cids.values())
         and planning["pending_cid"] == structured(planning["pending_requirements"])
         and planning["codebase_inventory_context_cid"] == full_cid == prepared["codebase_inventory_context_cid"]
         and planning["current_facts"] == planning["removed_task_cids"] == []
         and planning["runtime_requirements_preserved"] is True,
         "signed public planning/full-context bindings differ")
    compact = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    compact["member_paths"] = [row["path"] for row in scan["members"]]
    need(manifest["codebase_inventory_context"] == {"schema": "supervisor-codebase-inventory-declaration@1",
        "full_context_cid": full_cid, "scan": compact, "evidence": None, "authority": context["authority"]}
        and set(manifest["sources"]) == set(compact["member_paths"]), "full signed source/context declaration differs")
    spec = next(item for item in manifest["tasks"] if item["task_key"] == "INVENTORY-OFFSET")
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = [row for row in scan["members"] if row["path"] in spec["scope_paths"]]
    projection = {"schema": "supervisor-codebase-inventory-worker-context@1", "task_cid": task_cid,
        "task_id": "INVENTORY-OFFSET", "manifest_cid": public["manifest_cid"], "graph_cid": planning["graph_cid"],
        "planning_receipt_cid": structured(plan["receipt"]), "codebase_inventory_context_cid": full_cid,
        "scan": advisory, "evidence": None, "administrator_task_cids": sorted(task_cids.values()), "task_spec": spec,
        "dependency_task_cids": tasks["INVENTORY-OFFSET"]["dependency_task_cids"],
        "pending_requirements": planning["pending_requirements"], "pending_cid": planning["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False, "authority": context["authority"]}
    need(structured(projection) == prepared["inventory_context_cid"], "actual worker projection identity differs")
    launch = reader.path("private/launch")
    need(launch.is_dir(), "native private launch logs unavailable")
    matches, boundaries, searched, entries = [], [], [], 0
    for directory, names, files in os.walk(launch, followlinks=False):
        reader.tick()
        names.sort(); files.sort()
        need(not any((Path(directory) / name).is_symlink() for name in names), "launch log directory alias")
        entries += len(names) + len(files)
        need(entries <= 4096, "bounded native log traversal required")
        if "implementation-logs" not in Path(directory).parts:
            continue
        for name in files:
            if not name.endswith(".log"):
                continue
            need(len(searched) < 64, "native implementation log count exceeds bound")
            relative = (Path(directory) / name).relative_to(reader.root).as_posix()
            raw = reader.raw(relative, 16 * 1024**2)
            pin = {"path": relative, "bytes": len(raw), "sha256": sha(raw)}
            searched.append(pin)
            for line_number, line in enumerate(raw.splitlines(), 1):
                if (b"inventory-resume-authored-native-worker@1" not in line
                        and b"inventory-authored-worker-boundary@1" not in line):
                    continue
                try:
                    receipt = parse(line)
                except (ValueError, UnicodeError):
                    continue
                if type(receipt) is dict and receipt.get("schema") == "inventory-resume-authored-native-worker@1":
                    matches.append({"log": pin, "line": line_number, "receipt": receipt})
                elif type(receipt) is dict and receipt.get("schema") == "inventory-authored-worker-boundary@1":
                    boundaries.append({"log": pin, "line": line_number, "receipt": receipt})
    need(len(matches) == 1, "exactly one actual authored worker stdout receipt required; logs="
         + ",".join(row["path"] for row in searched))
    found, receipt = matches[0], matches[0]["receipt"]
    need(len(boundaries) == 1 and boundaries[0]["log"] == found["log"] and boundaries[0]["line"] < found["line"],
         "one actual deployer-boundary stdout receipt must precede authored stdout in the same log")
    boundary_found = boundaries[0]
    boundary_receipt = boundary_found["receipt"]
    boundary_raw = reader.raw("container-boundary.json", 16_384)
    boundary = parse(boundary_raw)
    need(boundary["schema"] == "supervisor-container-worker-boundary@1"
         and boundary["owner_uid"] == 1000 and boundary["worker_uid"] == 1001 and boundary["single_worker"] is True
         and re.fullmatch(r"[0-9a-f]{64}", boundary["container_id"]) is not None
         and re.fullmatch(r"sha256:[0-9a-f]{64}", boundary["image_id"]) is not None
         and boundary["allowed_worktree_roots"] == ["/opt/ipfs-supervisor/worktrees"],
         "copied root-deployer container boundary differs")
    expected_boundary = {key: boundary[key] for key in ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
    expected_boundary.update(purpose="coding", manifest_sha256=sha(boundary_raw), completion_authority=False)
    need(boundary_receipt["boundary"] == expected_boundary
         and boundary_receipt["boundary_artifact"] == "/opt/ipfs-supervisor/container-boundary.json"
         and boundary_receipt["boundary_sha256"] == sha(boundary_raw)
         and type(boundary_receipt["uid"]) is int and type(boundary_receipt["euid"]) is int
         and boundary_receipt["uid"] == boundary_receipt["euid"] == 1001
         and type(boundary_receipt["pid"]) is int and boundary_receipt["pid"] == receipt["pid"]
         and type(boundary_receipt["gid"]) is int and boundary_receipt["gid"] > 0
         and type(boundary_receipt["groups"]) is list and all(type(item) is int and item > 0 for item in boundary_receipt["groups"])
         and type(boundary_receipt["provider_calls"]) is int and boundary_receipt["provider_calls"] == 0
         and type(boundary_receipt["training_steps"]) is int and boundary_receipt["training_steps"] == 0,
         "actual root-deployer boundary receipt does not join authored worker identity")
    workspace = PurePosixPath(boundary_receipt["workspace"])
    need(workspace.is_absolute() and ".." not in workspace.parts and workspace != PurePosixPath("/opt/ipfs-supervisor/worktrees")
         and workspace.is_relative_to("/opt/ipfs-supervisor/worktrees"), "actual worker workspace is outside allocated root")
    access = boundary_receipt["private_access"]
    need(type(access) is dict and set(access) == set(boundary["owner_private_paths"]) and len(access) == 2
         and all(type(row) is dict and set(row) == {"read", "write", "execute"}
                 and all(value is False for value in row.values()) for row in access.values()),
         "actual worker private-authority denial observations are incomplete")
    need(type(receipt["uid"]) is int and receipt["uid"] == 1001 and type(receipt["pid"]) is int and receipt["pid"] > 1
         and receipt["task_cid"] == task_cid and receipt["status"] == "materialized" and receipt["path"] == "calc.py"
         and receipt["before_sha256"] == sha(BEFORE) and receipt["after_sha256"] == sha(AFTER)
         and type(receipt["provider_calls"]) is int and receipt["provider_calls"] == 0
         and type(receipt["training_steps"]) is int and receipt["training_steps"] == 0
         and receipt["proof_authority"] is False and receipt["completion_authority"] is False
         and receipt["native_completion_recorded_here"] is False, "real authored worker UID/task/patch/accounting differs")
    inclusion = receipt["public_instruction"]
    need(inclusion["schema"] == "supervisor-public-instruction-inclusion@3"
         and inclusion["artifact"] == prepared["artifact"] and inclusion["artifact_sha256"] == prepared["sha256"]
         and inclusion["context_cid"] == prepared["context_cid"] and inclusion["task_cid"] == task_cid
         and inclusion["manifest_cid"] == prepared["manifest_cid"] and inclusion["source_path"] == prepared["source_path"] == "README.md"
         and inclusion["source_sha256"] == prepared["source_sha256"] and inclusion["source_bytes"] == prepared["source_bytes"]
         and inclusion["manifest_signature_verified"] is True and inclusion["source_freshness_verified"] is True
         and inclusion["historical_replay"] is False and inclusion["verbatim_utf8"] is True
         and inclusion["semantic_minification_applied"] is False and type(inclusion["extra_provider_calls"]) is int
         and inclusion["extra_provider_calls"] == 0 and inclusion["completion_authority"] is False
         and inclusion["scope_expansion_authority"] is False, "real public reader inclusion differs")
    inventory = inclusion["codebase_inventory"]
    need(inventory == {"context_cid": prepared["inventory_context_cid"], "codebase_inventory_context_cid": full_cid,
        "root_cid": scan["root_cid"], "completion_cid": scan["completion_cid"], "membership_cid": scan["membership_cid"],
        "planning_receipt_cid": structured(plan["receipt"]), "administrator_task_cids": sorted(task_cids.values()),
        "pending_cid": planning["pending_cid"], "current_facts": [], "removed_task_cids": [],
        "runtime_requirements_preserved": True, "native_inventory_current_verified_here": False,
        "native_persistence_verified_here": False, "authority": context["authority"]},
        "real worker public advisory/full-population inclusion differs")
    return {"schema": "inventory-authored-worker-receipt-verification@1", "verified": True,
        "actual_stdout_observation": found, "original_artifact": {"path": prepared["artifact"],
            "sha256": sha(original), "bytes": len(original)}, "public_signatures_verified": True,
        "actual_boundary_stdout_observation": boundary_found,
        "retained_root_boundary": {"path": "container-boundary.json", "sha256": sha(boundary_raw), "bytes": len(boundary_raw)},
        "context_cid": full_cid, "worker_context_cid": prepared["inventory_context_cid"],
        "task_cid": task_cid, "native_registry_opened": False, "native_inventory_freshness_verified_here": False,
        "private_evidence_epoch_verified_here": False, "process_origin_attested": False,
        "completion_authority": False, "proof_authority": False, "guarded_read_bytes": reader.total_read_bytes}


def audit_public_worker(reader, archive, result, root, completion, report):
    prepared = reader.json("native/public-worker-context.json")
    actual_verified = verify_authored_worker_receipt(reader.root / "native", prepared=prepared, task_cid=prepared["task_cid"])
    retained_verified = reader.json("native/authored-worker-receipt-verification.json")
    need(actual_verified == retained_verified == result["authored_worker_receipt"],
         "independent actual boundary/worker receipt differs from native post-STOP gate")
    original_raw = reader.raw(prepared["artifact"], 8 * 1024**2)
    public = parse(original_raw)
    need(sha(original_raw) == prepared["sha256"] and not reader.path(prepared["artifact"]).stat().st_mode & 0o222,
         "original public instruction byte hash/mode differs")
    need(public == reader.json("native/public-worker-artifact.json"), "public review semantic JSON differs")
    need(original_raw == reader.raw("native/public-worker-artifact.raw.json", 8 * 1024**2),
         "retained exact public artifact byte copy differs")
    need(public["schema"] == "supervisor-public-instruction@3"
         and prepared["task_cid"] == public["task_cid"] == result["residual_task"]["task_cid"], "selected public task differs")
    signed = reader.json("native/signed-manifest.json")
    manifest = signature(signed)
    admission = reader.json("native/admission.json")
    planning = signature(admission["receipt"])
    need(admission["manifest"] == signed == public["manifest"]
         and public["inventory_plan_admission"]["receipt"] == admission["receipt"]
         and public["inventory_plan_admission"]["graph"] == admission["graph"], "full public signed admission differs")
    need(signed["binding"]["identity"] == admission["receipt"]["binding"]["identity"] == public["owner_identity"]
         and signed["binding"]["profile_id"] == admission["receipt"]["binding"]["profile_id"] == public["owner_profile_id"],
         "historical public signer identity differs")
    need(manifest["schema"] == "supervisor-local-benchmark-manifest@5"
         and planning["schema"] == "supervisor-local-planning-receipt@3", "explicit inventory signature profiles required")
    graph = admission["graph"]
    tasks = {task["task_key"]: task for task in graph["tasks"]}
    task_cids = {key: prompt_task_record_cid(task) for key, task in tasks.items()}
    need(set(tasks) == {"INVENTORY-TYPE", "INVENTORY-OFFSET"}
         and tasks["INVENTORY-OFFSET"]["dependency_task_cids"] == [task_cids["INVENTORY-TYPE"]]
         and task_cids["INVENTORY-OFFSET"] == prepared["task_cid"], "full authored native task population differs")
    need(planning["manifest_cid"] == structured(signed) == public["manifest_cid"]
         and planning["graph_cid"] == structured(semantic(graph))
         and planning["administrator_task_cids"] == sorted(task_cids.values())
         and planning["pending_cid"] == structured(planning["pending_requirements"])
         and planning["current_facts"] == [] and planning["removed_task_cids"] == []
         and planning["runtime_requirements_preserved"] is True, "signed native graph/pending population differs")
    context = public["codebase_inventory_context"]
    scan = context["scan"]
    false_authority(context["authority"])
    need(context["schema"] == "supervisor-codebase-inventory-context@1" and context["evidence"] is None
         and scan["root_record"] == root and scan["completion_record"] == completion
         and scan["root_cid"] == result["scan_root_cid"] and scan["completion_cid"] == result["completed_scan_cid"]
         and scan["members"] == root["members"] and scan["coverage"] == completion["coverage"], "full public 300-member scan context differs")
    full_cid = structured(context)
    need(full_cid == prepared["codebase_inventory_context_cid"] == planning["codebase_inventory_context_cid"],
         "full inventory context identity differs")
    declaration = manifest["codebase_inventory_context"]
    compact = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    compact["member_paths"] = [member["path"] for member in scan["members"]]
    expected = {"schema": "supervisor-codebase-inventory-declaration@1", "full_context_cid": full_cid,
                "scan": compact, "evidence": None, "authority": context["authority"]}
    need(declaration == expected and set(manifest["sources"]) == set(compact["member_paths"])
         and len(manifest["sources"]) == 300, "compact signed declaration loses full source closure")
    for path, binding in manifest["sources"].items():
        raw = reader.raw("native/repository/" + path, 64 * 1024 + 1)
        expected_sha = sha(AFTER) if path == "calc.py" else binding["sha256"]
        need(sha(raw) == expected_sha and bool(reader.path("native/repository/" + path).stat().st_mode & 0o111)
             == binding["executable"], "published source differs from signed baseline/authorized patch: " + path)
    spec = next(item for item in manifest["tasks"] if item["task_key"] == "INVENTORY-OFFSET")
    need("README.md" in spec["scope_paths"] and spec["outputs"] == [{"path": "calc.py", "effect": "modify", "media_type": "text/x-python"}],
         "authored selected worker scope differs")
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid", "membership_cid",
        "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = [member for member in scan["members"] if member["path"] in spec["scope_paths"]]
    projection = {"schema": "supervisor-codebase-inventory-worker-context@1", "task_cid": prepared["task_cid"],
        "task_id": "INVENTORY-OFFSET", "manifest_cid": structured(signed), "graph_cid": planning["graph_cid"],
        "planning_receipt_cid": structured(admission["receipt"]), "codebase_inventory_context_cid": full_cid,
        "scan": advisory, "evidence": None, "administrator_task_cids": sorted(task_cids.values()), "task_spec": spec,
        "dependency_task_cids": tasks["INVENTORY-OFFSET"]["dependency_task_cids"],
        "pending_requirements": planning["pending_requirements"], "pending_cid": planning["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False, "authority": context["authority"]}
    need(structured(projection) == prepared["inventory_context_cid"], "rendered worker context digest differs")
    matches, searched = [], []
    for row in archive["files"]:
        if row["kind"] == "file" and row["path"].startswith("native/private/launch/") and row["path"].endswith(".log"):
            if "/implementation-logs/" not in row["path"]:
                continue
            need(len(searched) < 64 and row["bytes"] <= 16 * 1024**2, "bounded actual implementation logs required")
            raw = reader.raw(row["path"], 16 * 1024**2)
            searched.append({"path": row["path"], "bytes": len(raw), "sha256": sha(raw)})
            for line_number, line in enumerate(raw.splitlines(), 1):
                if b"inventory-resume-authored-native-worker@1" not in line:
                    continue
                try:
                    value = parse(line)
                except (ValueError, UnicodeError):
                    continue
                if type(value) is dict and value.get("schema") == "inventory-resume-authored-native-worker@1":
                    matches.append({"log": searched[-1], "line": line_number, "receipt": value})
    need(len(matches) == 1, "expected exactly one actual authored worker stdout receipt; searched implementation_logs: "
         + ",".join(item["path"] for item in searched))
    match, receipt = matches[0], matches[0]["receipt"]
    need(type(receipt["uid"]) is int and receipt["uid"] == 1001 and type(receipt["pid"]) is int and receipt["pid"] > 1
         and receipt["status"] == "materialized" and receipt["task_cid"] == prepared["task_cid"]
         and receipt["path"] == "calc.py" and receipt["before_sha256"] == sha(BEFORE) and receipt["after_sha256"] == sha(AFTER)
         and type(receipt["provider_calls"]) is int and receipt["provider_calls"] == 0
         and type(receipt["training_steps"]) is int and receipt["training_steps"] == 0
         and receipt["proof_authority"] is False and receipt["completion_authority"] is False
         and receipt["native_completion_recorded_here"] is False, "actual authored worker identity/patch/cost differs")
    inclusion = receipt["public_instruction"]
    need(inclusion["schema"] == "supervisor-public-instruction-inclusion@3"
         and inclusion["artifact"] == prepared["artifact"] and inclusion["artifact_sha256"] == prepared["sha256"]
         and inclusion["context_cid"] == prepared["context_cid"] and inclusion["task_cid"] == prepared["task_cid"]
         and inclusion["manifest_cid"] == prepared["manifest_cid"] and inclusion["source_path"] == "README.md"
         and inclusion["source_sha256"] == prepared["source_sha256"]
         and inclusion["source_bytes"] == prepared["source_bytes"]
         and inclusion["manifest_signature_verified"] is True and inclusion["source_freshness_verified"] is True
         and inclusion["historical_replay"] is False and inclusion["semantic_minification_applied"] is False
         and inclusion["verbatim_utf8"] is True and inclusion["extra_provider_calls"] == 0
         and inclusion["completion_authority"] is False and inclusion["scope_expansion_authority"] is False,
         "actual public inclusion receipt does not bind prepared bytes")
    inventory = inclusion["codebase_inventory"]
    need(inventory["context_cid"] == prepared["inventory_context_cid"]
         and inventory["codebase_inventory_context_cid"] == full_cid
         and inventory["root_cid"] == scan["root_cid"] and inventory["completion_cid"] == scan["completion_cid"]
         and inventory["membership_cid"] == scan["membership_cid"]
         and inventory["planning_receipt_cid"] == structured(admission["receipt"])
         and inventory["administrator_task_cids"] == sorted(task_cids.values())
         and inventory["pending_cid"] == planning["pending_cid"]
         and inventory["current_facts"] == inventory["removed_task_cids"] == []
         and inventory["runtime_requirements_preserved"] is True
         and inventory["native_inventory_current_verified_here"] is False
         and inventory["native_persistence_verified_here"] is False, "actual worker inclusion lost native population/context")
    false_authority(inventory["authority"])
    report["worker"] = {"actual_stdout_observation": match, "original_public_artifact_sha256": sha(original_raw),
        "actual_boundary_stdout_observation": actual_verified["actual_boundary_stdout_observation"],
        "post_stop_receipt_gate_independently_replayed": True,
        "review_json_is_semantic_reserialization": True, "full_inventory_context_cid": full_cid,
        "worker_context_cid": prepared["inventory_context_cid"], "historical_public_signatures_verified": True,
        "native_task_cids": task_cids, "current_owner_registry_reopened": False}


def audit_refusal_controls(result, report):
    """Keep historical scan controls separate from current publication checks."""
    historical_names = {"legacy_256_profile_refuses_large_head", "incomplete_prefix_cannot_complete",
        "cursor_offset_forgery", "pre_cancelled_resume", "current_source_drift",
        "durable_page_byte_tamper", "native_projection_row_tamper"}
    current_rows = result["controls"]
    need(type(current_rows) is list, "current native controls require a list")
    controls = {row["name"]: row for row in current_rows}
    need(len(controls) == len(current_rows), "duplicate current native refusal control")
    expected = historical_names | {"published_source_rejects_old_completion"}
    inherited = []
    if "scan_reuse" in result:
        receiving = report["scan_reuse"]
        need(receiving["fresh_completed_chain_receiving_recorded"] is True
             and receiving["source_archive_preserved"] is True
             and receiving["source_audit_sha256"] == SCAN_SOURCE_AUDIT_SHA256,
             "composed controls require independently pinned historical receiving")
        inherited = receiving["historical_preworker_controls"]
        need(type(inherited) is list and len(inherited) == len(historical_names)
             and {row["name"] for row in inherited} == historical_names,
             "seven historical preworker controls required")
        expected = {"published_source_rejects_old_completion"}
    need(set(controls) == expected, "native qualification refusal controls incomplete")
    for row in [*controls.values(), *inherited]:
        need(row["refused"] is True and row["accepted_resource_or_fit_failure"] is False,
             "resource or fitting failure cannot qualify a native semantic refusal")
        resources(row["resources_after"])
    return controls, inherited


def audit_lifecycle(reader, result, container, report):
    need(result["qualified"] is True and type(result["post_setup_fit_attempt_count"]) is int
         and result["post_setup_fit_attempt_count"] == 0 and type(result["known_actual_setup_epochs"]) is int
         and result["unknown_fitting_epochs"] is False,
         "native qualification/fitting cost not complete")
    attempts = result["setup_training_attempts"]
    if "setup_reuse" in result:
        need(result["known_actual_setup_epochs"] == 0 and attempts == []
             and type(result["inherited_actual_setup_epochs"]) is int and result["inherited_actual_setup_epochs"] == 2
             and report["setup_seed"]["new_fitting_epochs"] == 0,
             "reused setup must charge inherited epochs separately from new fitting")
        inherited = 2
    else:
        need(result["known_actual_setup_epochs"] == 2 and result.get("inherited_actual_setup_epochs", 0) == 0
             and len(attempts) == 2 and all(type(row["actual_completed_epochs"]) is int
             and row["requested_epochs"] == row["actual_completed_epochs"] == 1
             and row["unknown_actual_epochs_on_failure"] is False for row in attempts), "actual setup epoch accounting differs")
        inherited = 0
    controls, inherited_controls = audit_refusal_controls(result, report)
    resources(result["final_resources"]); resources(result["resource_after_native_owner_close"])
    need(result["start"]["status"] == "succeeded" and result["stop"]["status"] == "succeeded"
         and type(result["remaining_processes"]) is int and result["remaining_processes"] == 0
         and not result["bootstrap_errors"] and result["residual_task"]["status"] == "completed"
         and result["native_task_statuses"] == {"INVENTORY-TYPE": "completed", "INVENTORY-OFFSET": "completed"},
         "actual native START/completion/STOP/UID cleanup incomplete")
    lifecycle = reader.json("native/native-lifecycle.json")
    need(all(lifecycle[key] == result[key] for key in ("start", "stop", "residual_task", "task_observations",
        "observed_worker_allocations", "remaining_processes", "bootstrap_errors")), "retained native lifecycle differs")
    before = reader.json("native/execution-scope-before.json")
    after = reader.json("native/execution-scope-after-stop.json")
    scope = signature(before)
    need(before == after and scope["schema"] == "supervisor-codebase-inventory-execution-scope@1"
         and scope["profile"] == "codebase-inventory-one-ready-native-worker@1"
         and scope["current_facts"] == scope["removed_task_cids"] == []
         and scope["task_population_preserved"] is True and scope["inventory_features_are_advisory"] is True
         and all(scope[key] is False for key in ("task_omission_authority", "completion_authority", "proof_authority",
             "publication_authority", "production_activation")), "signed native execution scope differs")
    population = scope["native_population"]
    selected = result["residual_task"]["task_cid"]
    need(population["selected_task_cids"] == [selected] and len(population["tasks"]) == 2
         and {row["task_cid"] for row in population["tasks"]} == set(report["worker"]["native_task_cids"].values()),
         "scope loses full original native population")
    ready = next(row for row in population["tasks"] if row["task_cid"] == selected)
    prerequisite = next(row for row in population["tasks"] if row["task_cid"] != selected)
    need(ready["status"] == "ready" and prerequisite["status"] == "completed"
         and prerequisite["task_cid"] in population["completed_prerequisites"]
         and bool(population["completion_rows"][prerequisite["task_cid"]])
         and scope["candidate"]["public_instruction"] == reader.json("native/public-worker-context.json")
         and scope["candidate"]["task_revision"] == ready["revision"]
         and scope["candidate"]["argv"] == reader.json("native/candidate.json")["argv"]
         and scope["lease"]["cpu_slots"] == 4 and scope["lease"]["memory_mb"] == 4096
         and scope["lease"]["child_process_slots"] == 8, "scope ready selection/prerequisite/lease binding differs")
    need(result["publication"]["public_checks_passed"] is True and result["publication"]["changed_paths"] == ["calc.py"]
         and len(result["publication"]["parents"]) == 2
         and result["publication"]["parents"][0] == result["publication"]["baseline_commit"], "actual native publication differs")
    need(result["proof_authority"] is False and result["source_execution_attested"] is False
         and result["scan_execution_attested"] is False and result["production_default_activated"] is False
         and result["cuda_qualified"] is False and result["384d_qualified"] is False,
         "qualification claims expanded beyond current native fixture")
    need(container["returncode"] == 0 and container["container_removed"] is True
         and container["container_results_copied"] is True and container["network"] == "none"
         and container["privileged"] is False and container["host_reservation_released"] is True
         and container["retained_source_verified_before_execution"] is True
         and container["retained_source_verified_after_execution"] is True, "outer Docker/resource/snapshot closure incomplete")
    resources(container["host_resources_after_cleanup"])
    need(type(result["recorded_seconds"]) in {int, float} and result["recorded_seconds"] > 0
         and container["elapsed_seconds"] >= result["recorded_seconds"]
         and container["staging_seconds"] >= 0, "actual recorded cost differs")
    report["lifecycle"] = {"known_actual_setup_epochs": result["known_actual_setup_epochs"],
        "inherited_actual_setup_epochs": inherited, "post_setup_fit_attempt_count": 0,
        "native_recorded_seconds": result["recorded_seconds"], "container_total_seconds": container["elapsed_seconds"],
        "staging_seconds": container["staging_seconds"], "native_final_resources": result["final_resources"],
        "host_final_resources": container["host_resources_after_cleanup"], "native_remaining_processes": 0,
        "container_removed": True, "control_names": sorted(controls), "authority": "retained_observations_only"}
    if "scan_reuse" in result:
        report["lifecycle"].update(qualification_composition="fresh native worker plus inherited closed09 scan execution",
            inherited_control_names=sorted(row["name"] for row in inherited_controls),
            inherited_control_scope="closed09 historical preworker observations; no current rerun")


def run(namespace, report_path, *, seconds=300):
    need(type(seconds) in {int, float} and math.isfinite(seconds) and 0 < seconds <= 600, "bounded audit deadline required")
    reader = Reader(namespace, seconds)
    target = Path(report_path).absolute()
    need(not target.is_relative_to(reader.root) and not target.exists()
         and target.parent.resolve(strict=True) == target.parent, "fresh sibling audit report required")
    started = time.monotonic()
    report = {"schema": SCHEMA, "qualified": False, "namespace": str(reader.root),
        "audit_source_sha256": sha(Path(__file__).read_bytes()), "native_jobs_executed": 0,
        "native_owner_databases_opened": 0, "training_steps": 0, "authority": "none",
        "scope": "bounded retained bytes/signatures/receipts; sequential point-time observations, no process-origin attestation",
        "bounds": {"seconds": seconds, "max_archive_files": MAX_FILES, "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_file_bytes": MAX_FILE_BYTES, "max_read_bytes": MAX_READ_BYTES, "max_total_read_bytes": MAX_TOTAL_READ_BYTES}}
    before = None
    try:
        before = reader.whole_archive()
        result = reader.json("native/result.json")
        container = reader.json("container-execution-final.json")
        preflight_record = reader.json("offline-preflight.json")
        raw_preflight = reader.raw("offline-preflight.stdout")
        preflight = parse(raw_preflight)
        need(preflight_record["status"] == "completed" and preflight_record["returncode"] == 0
             and preflight_record["stdout_sha256"] == sha(raw_preflight)
             and preflight_record["stderr_sha256"] == sha(reader.raw("offline-preflight.stderr"))
             and preflight_record["observation"] == preflight and preflight["owner_uid"] == 1000
             and preflight["training_steps"] == 0 and preflight["torch_device"] == "cpu"
             and preflight["torch_dtype"] == "float64" and preflight["tensor_check_passed"] is True,
             "actual offline dependency preflight differs")
        producers = audit_sources(reader, before, report)
        if result.get("qualified") is not True:
            report["native_failure_observation"] = {"qualified": result.get("qualified"),
                "error_type": result.get("error_type"), "error": result.get("error"),
                "recorded_seconds": result.get("recorded_seconds"),
                "known_actual_setup_epochs": result.get("known_actual_setup_epochs"),
                "inherited_actual_setup_epochs": result.get("inherited_actual_setup_epochs", 0),
                "unknown_fitting_epochs": result.get("unknown_fitting_epochs"),
                "post_setup_fit_attempt_count": result.get("post_setup_fit_attempt_count"),
                "setup_training_attempts": result.get("setup_training_attempts"),
                "phases": result.get("phases"), "final_resources": result.get("final_resources"),
                "container_removed": container.get("container_removed"),
                "host_reservation_released": container.get("host_reservation_released"),
                "host_final_resources": container.get("host_resources_after_cleanup"),
                "container_total_seconds": container.get("elapsed_seconds"),
                "worker_receipt_qualified": False}
            if "scan_reuse" in result:
                # A later native failure omits final numerical summary fields.
                # Supply an explicitly disclosed receiving projection from the
                # retained checkpoint; never rewrite or qualify the native result.
                root = reader.json("native/scan-root.json")
                artifact = root["model"]["artifact"]
                saved = parse(reader.raw("native/private/model-artifacts/" + artifact["sha256"][:2]
                                        + "/" + artifact["sha256"], 16 * 1024**2))
                state = {"state_sha256": sha(wire(saved["state"])), "completed_epochs": saved["state"]["completed_epochs"],
                    "adam_steps": [item["step"] for item in saved["state"]["adam"]],
                    "latent_width": saved["state"]["latent_width"], "feature_columns": len(saved["feature_space"]["columns"])}
                projected = {**result, "numerical_before": state, "numerical_after": state}
                audit_scan(reader, projected, producers, preflight, report)
                report["positive_completed_scan_receiving"] = {"verified": True,
                    "overall_native_qualification": False, "native_worker_qualified": False,
                    "scope": "fresh completed-chain receiving with closed09 historical numerical receipts",
                    "receiving_input_projection": {"numerical_before": state, "numerical_after": state,
                        "origin": "retained raw checkpoint joined to receiving owners and inherited full states",
                        "native_final_summary_fields_present": "numerical_before" in result and "numerical_after" in result},
                    "native_owners_opened_by_auditor": False, "numerical_execution_independently_reperformed": False}
            elif "setup_reuse" in result:
                root = reader.json("native/scan-root.json") if reader.path("native/scan-root.json").exists() else {
                    "head": result["head"], "model": {"version_id": result["selected_version_id"]}}
                audit_setup_seed(reader, result, root, producers, report)
            audit_failed_resume_chunks(reader, result, producers, preflight, report)
            raise InventoryWorkerAuditError("native qualification failed before required complete scan/worker evidence: "
                                            + str(result.get("error_type")) + ": " + str(result.get("error")))
        root, completion = audit_scan(reader, result, producers, preflight, report)
        audit_public_worker(reader, before, result, root, completion, report)
        audit_lifecycle(reader, result, container, report)
        report["qualified"] = True
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error)[:4096])
    finally:
        try:
            after = reader.whole_archive()
            need(before == after, "whole primary archive bytes/modes/membership changed during read-only audit")
            report["primary_archive"] = {"before": before, "ending_inventory_cid": after["inventory_cid"],
                "preserved": True, "passes": 2}
        except BaseException as error:
            report.update(qualified=False, archive_preservation_error=str(error)[:4096])
        report["guarded_read_bytes"] = reader.total_read_bytes
        report["observed_artifact_pins"] = reader.observed
        report["recorded_seconds"] = time.monotonic() - started
        raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        need(len(raw) <= 32 * 1024**2, "derived report byte bound exceeded")
        with target.open("xb") as stream:
            stream.write(raw)
        target.chmod(0o444)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    result = run(args.namespace, args.report, seconds=args.seconds)
    print(json.dumps({"schema": SCHEMA, "qualified": result["qualified"], "report": str(args.report),
        "error": result.get("error"), "archive_preservation_error": result.get("archive_preservation_error")}, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
