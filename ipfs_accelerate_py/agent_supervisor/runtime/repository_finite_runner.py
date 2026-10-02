"""Materialize a separately signed finite-repository candidate in a native worktree."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import stat
import sys

from ..proof.formal_verification_contracts import content_identity
from .doctor_candidate_runner import MAX_BYTES, _directory, _git, _read, _sha, _unique
from .doctor_contract_candidate_runner import _check_parent, _path, _write
from .local_planning_admission import verify_did_key_signature
from .repository_finite_handoff import SCHEMA, SCOPE, FALSE

FIELDS = {"schema", "repository", "baseline_commit", "task_cid", "task_id", "task_revision",
    "manifest_cid", "instruction_sha256", "instruction_path", "permitted_output", "edit", "evidence",
    "scope", "artifact_cid", *FALSE}


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def materialize_finite_candidate(*, artifact: Path, expected_sha256: str, task_cid: str,
        owner_did: str, profile_id: str, prompt: str, workspace: Path) -> dict:
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, info = _read(parent, artifact.name)
    finally:
        os.close(parent)
    if stat.S_IMODE(info.st_mode) & 0o222 or not _digest(expected_sha256) or _sha(raw) != expected_sha256:
        raise ValueError("immutable owner-pinned finite handoff required")
    envelope = json.loads(raw, object_pairs_hook=_unique)
    if type(envelope) is not dict or set(envelope) != {"payload", "binding"}:
        raise ValueError("closed owner-signed finite evidence required")
    payload, binding = envelope["payload"], envelope["binding"]
    if (type(binding) is not dict or set(binding) != {"identity", "profile_id", "signature"}
            or binding["identity"] != owner_did or binding["profile_id"] != profile_id
            or type(owner_did) is not str or not owner_did or type(profile_id) is not str or not profile_id):
        raise ValueError("finite evidence belongs to a different launch owner")
    verify_did_key_signature(identity_did=owner_did, payload=payload, signature=binding["signature"])
    if (type(payload) is not dict or set(payload) != FIELDS or payload["schema"] != SCHEMA
            or payload["artifact_cid"] != content_identity({k: v for k, v in payload.items() if k != "artifact_cid"})
            or payload["task_cid"] != task_cid or type(payload["task_revision"]) is not int
            or payload["task_revision"] < 1 or payload["scope"] != SCOPE
            or not _digest(payload["instruction_sha256"])
            or any(payload[name] is not False for name in FALSE)):
        raise ValueError("finite handoff identity or authority differs")
    evidence = payload["evidence"]
    required = {"query_cid", "source_head", "preview_cid", "selected_operation", "complete_requirements",
                "initial_observation_cid", "candidate_observation_cid", "input_sha256", "scope", "inputs"}
    if type(evidence) is not dict or set(evidence) != required:
        raise ValueError("closed finite evidence bindings required")
    inputs = evidence["inputs"]
    if (evidence["scope"] != SCOPE or not _digest(evidence["input_sha256"])
            or evidence["complete_requirements"] != ["finite-integer-offset-goal", "finite-integer-type-goal"]
            or type(inputs) is not list or not 1 <= len(inputs) <= 32
            or any(type(item) is not int or abs(item) > 2**31 for item in inputs)
            or inputs != sorted(set(inputs))):
        raise ValueError("complete finite instruction and nonempty exact domain required")
    operation = evidence["selected_operation"]
    if (type(operation) is not dict or set(operation) != {"requirement_id", "task_id", "producer_id",
            "path", "function_name", "parameter", "review_ref", "operation"}
            or operation["requirement_id"] != "finite-integer-offset-goal" or operation["operation"] != "update"):
        raise ValueError("exact reviewed offset operation required")
    if type(prompt) is not str or len(prompt.encode()) > 256000:
        raise ValueError("bounded native task prompt required")
    native, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if type(native) is not dict or native.get("objective_id") != payload["task_id"]:
        raise ValueError("native task differs from finite handoff")
    root, canonical = Path(workspace).absolute(), Path(payload["repository"])
    if (root.resolve(strict=True) != root or not canonical.is_absolute()
            or canonical.resolve(strict=True) != canonical or root == canonical or artifact.is_relative_to(root)):
        raise ValueError("separate native allocated worktree required")
    expected = canonical / ".runtime/repository-finite-handoffs" / (expected_sha256 + ".json")
    if artifact != expected:
        raise ValueError("finite artifact is outside its owner-controlled location")
    for location in (canonical / ".runtime", expected.parent, expected):
        metadata = location.lstat()
        kind = stat.S_ISREG if location == expected else stat.S_ISDIR
        if (metadata.st_uid != canonical.stat().st_uid or not kind(metadata.st_mode)
                or metadata.st_mode & 0o022):
            raise ValueError("finite artifact is not owner-controlled")
    if (Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()) != root
            or _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir") !=
               _git(canonical, "rev-parse", "--path-format=absolute", "--git-common-dir")):
        raise ValueError("finite worktree belongs to a foreign repository")
    baseline = payload["baseline_commit"]
    if (type(baseline) is not str or re.fullmatch(r"[a-f0-9]{40}", baseline) is None
            or _git(root, "rev-parse", "HEAD").decode().strip() != baseline
            or _git(canonical, "rev-parse", "HEAD").decode().strip() != baseline):
        raise ValueError("finite baseline drifted")
    output, edit = payload["permitted_output"], payload["edit"]
    if (type(output) is not dict or set(output) != {"path", "effect", "media_type"}
            or output["effect"] != "modify" or type(edit) is not dict
            or set(edit) != {"path", "before_sha256", "after_sha256", "after_bytes_base64"}
            or output["path"] != edit["path"] or operation["path"] != edit["path"] or not _digest(edit["before_sha256"])
            or not _digest(edit["after_sha256"])):
        raise ValueError("single declared finite source modification required")
    relative = _path(edit["path"])
    if payload["instruction_path"] is not None:
        instruction = _path(payload["instruction_path"])
        if instruction == relative:
            raise ValueError("finite instruction cannot be a modified source")
        original_instruction = _git(root, "show", baseline + ":" + str(instruction))
        instruction_parent = _directory(root / instruction.parent)
        try:
            if (_sha(original_instruction) != payload["instruction_sha256"]
                    or _read(instruction_parent, instruction.name)[0] != original_instruction):
                raise ValueError("allocated finite instruction drifted")
        finally:
            os.close(instruction_parent)
    after = base64.b64decode(edit["after_bytes_base64"], validate=True)
    original = _git(root, "show", baseline + ":" + str(relative))
    if (len(after) > MAX_BYTES or _sha(after) != edit["after_sha256"]
            or _sha(original) != edit["before_sha256"] or original == after):
        raise ValueError("finite candidate bytes or preimage differ")
    canonical_parent = _directory(canonical / relative.parent)
    try:
        if _read(canonical_parent, relative.name)[0] != original:
            raise ValueError("canonical finite preimage drifted")
    finally:
        os.close(canonical_parent)
    parent = _directory(root / relative.parent)
    try:
        current, metadata = _read(parent, relative.name)
        if current != original:
            raise ValueError("allocated finite preimage drifted")
        _check_parent(root, relative, parent)
        mode = _write(parent, relative, after, metadata)
        _check_parent(root, relative, parent)
        if _read(parent, relative.name)[0] != after or _git(root, "rev-parse", "HEAD").decode().strip() != baseline:
            raise ValueError("finite candidate changed during materialization")
    finally:
        os.close(parent)
    return dict(schema="native-repository-finite-materialization@1", status="candidate_materialized",
        artifact_cid=payload["artifact_cid"], task_cid=task_cid, baseline_commit=baseline,
        changed_paths=[str(relative)], source_after_sha256=edit["after_sha256"], write_mode=mode,
        evidence=payload["evidence"], provider_calls=0, **FALSE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-cid", required=True)
    parser.add_argument("--owner-did", required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    try:
        result = materialize_finite_candidate(artifact=args.artifact, expected_sha256=args.sha256,
            task_cid=args.task_cid, owner_did=args.owner_did, profile_id=args.profile_id,
            prompt=sys.stdin.buffer.read(256001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps(dict(status="refused", error_type=type(error).__name__, completion_authority=False)), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
