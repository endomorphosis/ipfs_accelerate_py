"""Explicitly registered source repairs; never task integration authority.

The operator supplies an independently reviewed admission file AND its digest
through trusted launch configuration. Neither this module nor an admission file
can register itself. The digest is a trust pin, not a signature or evidence that
tests ran. Historical failed operations remain failed. All functions are reads.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

ADMISSION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-source-repair-admission@1"
CONTINUITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-source-repair-continuity@1"
QUALIFICATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-source-repair-qualification@1"
PATH_ENV = "IPFS_ACCELERATE_ASEH_SOURCE_REPAIR_ADMISSION_PATH"
SHA_ENV = "IPFS_ACCELERATE_ASEH_SOURCE_REPAIR_ADMISSION_SHA256"
DENIED = {name: False for name in (
    "task_completion", "queue_settlement", "callback_replay", "source_replay",
    "historical_authorization_rewrite",
)}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_COMMITS = 512
MAX_PATHS = 20000


class SourceRepairDenied(ValueError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SourceRepairDenied(reason)


def _oid(value: Any) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
             "source repair object ID differs")
    return value


def pinned_json(reference: Mapping[str, str]) -> dict[str, Any]:
    """Read one bounded regular, owned, immutable-by-observation JSON file."""
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256"},
             "source repair file reference differs")
    path = Path(reference["path"])
    _require(path.is_absolute() and str(path) == reference["path"]
             and path.resolve() == path, "source repair file path differs")
    _require(re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None,
             "source repair file digest differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                 and not before.st_mode & 0o022 and before.st_nlink == 1
                 and 0 < before.st_size <= MAX_JSON_BYTES,
                 "source repair file custody differs")
        raw = bytearray()
        while len(raw) <= MAX_JSON_BYTES:
            chunk = os.read(fd, min(65536, MAX_JSON_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        def identity(row):
            return (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
                    row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        _require(identity(before) == identity(after) == identity(named)
                 and len(raw) == before.st_size
                 and hashlib.sha256(raw).hexdigest() == reference["sha256"],
                 "source repair file changed or does not match its pin")
    finally:
        os.close(fd)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "source repair JSON has duplicate keys")
            result[key] = value
        return result
    def invalid_constant(value):
        raise SourceRepairDenied("source repair JSON contains a non-finite number")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    _require(isinstance(value, dict), "source repair file is not an object")
    return value


def raw_delta(git_bytes: Callable[..., bytes], base: str, target: str) -> list[dict[str, str]]:
    raw = git_bytes("diff-tree", "--no-commit-id", "--no-renames", "--no-ext-diff",
                    "--no-textconv", "--raw", "--full-index", "-r", "-z",
                    _oid(base), _oid(target))
    _require(len(raw) <= MAX_JSON_BYTES, "source repair delta exceeds bound")
    fields = raw.split(b"\0")
    _require(fields[-1] == b"" and (len(fields) - 1) % 2 == 0,
             "source repair raw delta is malformed")
    rows = []
    for i in range(0, len(fields) - 1, 2):
        meta = fields[i].decode("ascii").split()
        path = fields[i + 1].decode("utf-8", errors="strict")
        _require(len(meta) == 5 and meta[0].startswith(":"), "source repair delta metadata differs")
        _require(path and not path.startswith("/") and all(p not in {"", ".", ".."} for p in path.split("/"))
                 and "\x00" not in path, "source repair delta path differs")
        old_mode, new_mode = meta[0][1:], meta[1]
        _require(old_mode in {"000000", "100644", "100755", "120000", "160000"}
                 and new_mode in {"000000", "100644", "100755", "120000", "160000"}
                 and meta[4] in {"A", "D", "M", "T"}, "source repair delta mode/status differs")
        rows.append({"path": path, "old_mode": old_mode, "new_mode": new_mode,
                     "old_blob": _oid(meta[2]), "new_blob": _oid(meta[3]), "status": meta[4]})
    _require(len(rows) <= MAX_PATHS and len({r["path"] for r in rows}) == len(rows),
             "source repair delta count differs")
    return rows


def history(git: Callable[..., str], git_bytes: Callable[..., bytes], *,
            anchor: str, repair_base: str, target: str) -> dict[str, Any]:
    """Describe every first-parent edge, plus all source-only and aggregate blobs."""
    for head in (anchor, repair_base, target):
        _oid(head)
        _require(git("rev-parse", "--verify", head + "^{commit}") == head,
                 "source repair commit is unavailable")
    _require(anchor != target and repair_base != target, "source repair suffix is empty")
    lines = git("log", "--first-parent", "--reverse", "--format=%H %T %P",
                f"{anchor}..{target}").splitlines()
    _require(0 < len(lines) <= MAX_COMMITS, "source repair history exceeds bound")
    rows = []
    previous = anchor
    linear = repair_base == anchor
    for line in lines:
        fields = line.split()
        _require(len(fields) in {3, 4}, "source repair history parent count differs")
        commit, tree, *parents = map(_oid, fields)
        _require(parents[0] == previous and len(parents) == (1 if linear else 2),
                 "source repair history is not the registered merge-prefix/linear-suffix")
        row = {"commit": commit, "tree": tree, "parents": parents}
        if linear:
            row["delta"] = raw_delta(git_bytes, previous, commit)
            _require(all(r["old_mode"] != "160000" and r["new_mode"] != "160000"
                         for r in row["delta"]), "source-only repair changes a submodule")
        rows.append(row)
        previous = commit
        if commit == repair_base:
            linear = True
    _require(previous == target and linear, "source repair base/target is not on the complete suffix")
    return {"anchor_head": anchor, "anchor_tree": git("rev-parse", anchor + "^{tree}"),
            "repair_base_head": repair_base, "repair_base_tree": git("rev-parse", repair_base + "^{tree}"),
            "target_head": target, "target_tree": git("rev-parse", target + "^{tree}"),
            "commits": rows, "anchor_delta": raw_delta(git_bytes, anchor, target),
            "repair_delta": raw_delta(git_bytes, repair_base, target)}


def registration(environ: Mapping[str, str], *, repository: Path,
                 git: Callable[..., str], git_bytes: Callable[..., bytes]) -> dict[str, Any] | None:
    path, pin = environ.get(PATH_ENV), environ.get(SHA_ENV)
    if path is None and pin is None:
        return None
    _require(bool(path) and bool(pin), "source repair registration requires path and independently pinned digest")
    value = pinned_json({"path": path, "sha256": pin})
    _require(set(value) == {"schema", "repository", "authority", "history", "canonical_prefix", "qualification",
                           "historical_refusal", "denied_authority"}
             and value["schema"] == ADMISSION_SCHEMA and value["repository"] == str(repository.resolve())
             and value["denied_authority"] == DENIED
             and all(v is False for v in value["denied_authority"].values()),
             "source repair registration shape/scope differs")
    authority = pinned_json(value["authority"])
    _require(authority.get("schema") == "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap-repair-historical-live-evidence-revision-closure-transition@1",
             "source repair authority is not the retained R45 receipt")
    h = value["history"]
    _require(isinstance(h, dict), "source repair history is absent")
    actual = history(git, git_bytes, anchor=authority["repair_head"],
                     repair_base=h["repair_base_head"], target=h["target_head"])
    _require(h == actual and h["anchor_tree"] == authority["repair_tree"],
             "source repair full history/tree/blob delta changed")
    qualification = pinned_json(value["qualification"])
    subject = {k: v for k, v in value.items() if k != "qualification"}
    _require(set(qualification) == {"schema", "subject_sha256", "checks", "provenance"}
             and qualification["schema"] == QUALIFICATION_SCHEMA
             and qualification["subject_sha256"] == digest(subject)
             and isinstance(qualification["checks"], dict) and qualification["checks"]
             and all(isinstance(k, str) and k and v is True for k, v in qualification["checks"].items())
             and isinstance(qualification["provenance"], list) and qualification["provenance"],
             "source repair independent qualification is absent or stale")
    for ref in qualification["provenance"]:
        pinned_json(ref)
    _require(isinstance(value["historical_refusal"], dict)
             and set(value["historical_refusal"]) == {"exit", "preflight", "archive", "controller_exit"},
             "source repair historical refusal references differ")
    refusal = {key: pinned_json(ref) for key, ref in value["historical_refusal"].items()}
    # Exact bytes are externally pinned. Recheck semantics without interpreting
    # an archival refusal as either source success or permission to replay it.
    exit_row = refusal["exit"].get("scope", {})
    _require(exit_row.get("phase") == "source_refused_archived_target"
             and exit_row.get("source_result_sha256") is None,
             "source repair historical refusal was rewritten as success")
    _require(exit_row.get("source_invoked") is True
             and exit_row.get("source_replay_authority") is False
             and exit_row.get("native_launch_authority") is False
             and exit_row.get("task_or_queue_settlement_authority") is False,
             "source repair historical refusal authority differs")
    refused_target = exit_row.get("observation", {}).get("source", {}).get("repositories", {}).get(".", {}).get("head")
    previous_source = exit_row.get("observation", {}).get("index_readmission", {}).get("current_source", {})
    _require(previous_source.get("repositories", {}).get(".", {}).get("head") == h["repair_base_head"],
             "source repair archived previous source differs from its repair base")
    _require(refused_target in {r["commit"] for r in h["commits"]},
             "source repair historical refused target is outside this source history")
    _require(refusal["preflight"].get("launch_admitted") is False
             and refusal["preflight"].get("callback_settlement_authority") is False
             and refusal["preflight"].get("returncode") == 1
             and refusal["archive"].get("callback_settlement_authority") is False
             and refusal["controller_exit"].get("normal_exit_code_matches_admitted_refusal") is True
             and refusal["controller_exit"].get("exit_receipt_sha256") == value["historical_refusal"]["exit"]["sha256"],
             "source repair failed preflight/closed controller evidence differs")
    return value


def current_target(reg: Mapping[str, Any], git: Callable[..., str], *,
                   head: str, tree: str) -> None:
    """A later target may add canonical merges, never another linear repair.

    This shape check only selects the route. Materialized admission separately
    requires the ordinary task/queue completion proof for every tail merge.
    """
    h = reg["history"]
    _require(git("rev-parse", _oid(head) + "^{tree}") == _oid(tree),
             "source repair current target tree differs")
    if head == h["target_head"]:
        _require(tree == h["target_tree"], "source repair registered target tree differs")
        return
    previous = h["target_head"]
    lines = git("log", "--first-parent", "--reverse", "--format=%H %T %P",
                f"{previous}..{head}").splitlines()
    _require(0 < len(lines) <= MAX_COMMITS, "source repair canonical tail exceeds bound")
    for line in lines:
        fields = line.split()
        _require(len(fields) == 4 and fields[2] == previous,
                 "source repair has an unregistered linear or unrelated descendant")
        for field in fields:
            _oid(field)
        previous = fields[0]
    _require(previous == head, "source repair canonical tail target differs")


def _mirror_source_repair_continuity(value: dict[str, Any], *, record_kind: str) -> dict[str, Any]:
    """Record a continuity receipt. It is not task or queue completion."""

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(value.get("receipt_cid") or "source_repair_continuity")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind=record_kind,
            record_ref=record_ref,
            subject_kind="receipt_id",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return value


def bind(reg: Mapping[str, Any], *, pin: str, canonical_prefix: Mapping[str, Any] | None,
         witness: Mapping[str, Any], guard: Mapping[str, Any],
         canonical_tail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    h = reg["history"]
    _oid(witness.get("head")); _oid(witness.get("tree"))
    _require(isinstance(guard, Mapping) and bool(guard), "source repair active guard is absent")
    value = {"schema": CONTINUITY_SCHEMA, "base_head": h["anchor_head"],
             "target_head": witness["head"], "target_tree": witness["tree"],
             "registration_sha256": pin, "canonical_prefix": canonical_prefix,
             "canonical_tail": canonical_tail,
             "active_witness": dict(witness), "active_guard_sha256": digest(guard)}
    value["receipt_cid"] = "sha256:" + digest(value)
    return _mirror_source_repair_continuity(value, record_kind="source_repair_continuity_binding")


def validate(value: Mapping[str, Any], reg: Mapping[str, Any], *, pin: str,
             witness: Mapping[str, Any], guard: Mapping[str, Any],
             canonical_validator: Callable[..., Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "source repair continuity is absent")
    prefix = value.get("canonical_prefix")
    h = reg["history"]
    _require(prefix == reg["canonical_prefix"], "source repair reviewed canonical prefix changed")
    if h["repair_base_head"] == h["anchor_head"]:
        _require(prefix is None, "source repair empty canonical prefix differs")
    else:
        _require(isinstance(prefix, Mapping) and prefix.get("schema") ==
                 "ipfs_accelerate_py/agent-supervisor/aseh-canonical-merge-suffix@1",
                 "source repair canonical prefix schema differs")
        canonical_validator(prefix, authorization_candidate_head=h["anchor_head"],
                            authorization_candidate_tree=h["anchor_tree"],
                            active_candidate_head=h["repair_base_head"],
                            active_candidate_tree=h["repair_base_tree"])
    tail = value.get("canonical_tail")
    if witness.get("head") == h["target_head"]:
        _require(witness.get("tree") == h["target_tree"] and tail is None,
                 "source repair exact target/tail differs")
    else:
        _require(isinstance(tail, Mapping) and tail.get("schema") ==
                 "ipfs_accelerate_py/agent-supervisor/aseh-canonical-merge-suffix@1",
                 "source repair canonical tail schema differs")
        canonical_validator(tail, authorization_candidate_head=h["target_head"],
                            authorization_candidate_tree=h["target_tree"],
                            active_candidate_head=witness.get("head"),
                            active_candidate_tree=witness.get("tree"))
    expected = bind(reg, pin=pin, canonical_prefix=prefix, canonical_tail=tail,
                    witness=witness, guard=guard)
    _require(dict(value) == expected, "source repair continuity or active custody changed")
    return _mirror_source_repair_continuity(expected, record_kind="source_repair_continuity_validation")
