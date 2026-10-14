"""Observe owner-private failed-worker candidate refreshes without exporting text.

Only current-trial router source-session receipts select files. Hashes and sizes
describe retained advisory artifacts; they do not establish provider delivery,
proof acceptance, continuation admission, correctness or token savings.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_checkpoint as checkpoint
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_output_observation import _read, _open_directory, _identity
from .terminal_numerical_retrieval_observation import _need, _sha, _parse, MAX_SESSIONS

SCHEMA = "terminal-owner-candidate-refresh-observation@1"
_BINDINGS = ("prior_session_id", "descriptor_sha256", "task_id", "native_prompt_sha256", "workspace_generation")
_OBSERVATION_FIELDS = {"schema", "status", "reason_code", *_BINDINGS,
    "artifacts_count", "artifact_bytes", "elapsed_seconds", *checkpoint.FLAGS}


def _base(status="unavailable", reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status=status, reason=reason, sessions=[], counts=None,
        observation_only=True, **checkpoint.FLAGS, provider_tokens=None,
        total_token_savings_measured=False, raw_sources_exported=False,
        raw_prompts_exported=False, artifact_delivery_observed=False,
        accepted_index_published=False)


def _flags(value):
    _need(type(value) is dict and all(type(value.get(key)) is type(expected)
        and value[key] == expected for key, expected in checkpoint.FLAGS.items()))


def _hex(value, length=64):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None)
    return value


def _private_read(fd, name, directory, *, limit, with_identity=False):
    before = os.stat(name, dir_fd=fd, follow_symlinks=False)
    _need(stat.S_IMODE(before.st_mode) == 0o400 and before.st_gid == directory.st_gid)
    raw = _read(fd, name, limit=limit, private=True)
    after = os.stat(name, dir_fd=fd, follow_symlinks=False)
    _need(_identity(before) == _identity(after) and before.st_gid == after.st_gid)
    return (raw, after) if with_identity else raw


def _session(fd, directory, row, scope):
    receipt = row["source_view_retrieval"]
    session = _hex(receipt["session_id"], 32)
    name = "candidate-round-" + session
    raw = _private_read(fd, name + ".json", directory, limit=checkpoint.MAX_MANIFEST_BYTES)
    manifest = _parse(raw)
    _need(set(manifest) == {"schema", "context", "artifacts", "observation", *checkpoint.FLAGS}
        and manifest["schema"] == checkpoint.SCHEMA)
    _flags(manifest)
    observation = manifest["observation"]
    _need(type(observation) is dict and set(observation) == _OBSERVATION_FIELDS
        and observation["schema"] == checkpoint.OBSERVATION_SCHEMA
        and observation["status"] == "prepared"
        and observation["reason_code"] == "candidate_context_prepared")
    _flags(observation)
    _need(observation["prior_session_id"] == session
        and observation["task_id"] == receipt["task_id"] == scope["task_alias"])
    for key in ("descriptor_sha256", "native_prompt_sha256"):
        _need(_hex(observation[key]) == _hex(receipt[key]))
    _hex(observation["workspace_generation"])
    _need(type(observation["artifacts_count"]) is int and observation["artifacts_count"] == 3
        and type(observation["artifact_bytes"]) is int
        and 0 < observation["artifact_bytes"] <= checkpoint.MAX_ARTIFACT_BYTES
        and type(observation["elapsed_seconds"]) in (int, float)
        and math.isfinite(observation["elapsed_seconds"]) and observation["elapsed_seconds"] >= 0)
    context = manifest["context"]
    _need(type(context) is dict and context.get("schema") == "supervisor-owner-candidate-round-context@1"
        and all(context.get(key) == observation[key] for key in _BINDINGS)
        and type(context.get("prior_returncode")) is int and 1 <= context["prior_returncode"] <= 255
        and type(context.get("round_number")) is int and context["round_number"] == 2
        and context.get("source_state") == "working-candidate"
        and all(context.get(key) is False for key in ("baseline_evidence_applicable", "proof_authority",
            "execution_authority", "completion_authority", "next_round_admitted", "accepted_index_published"))
        and all(type(context.get(key)) is int and context[key] == 0 for key in (
            "source_obligations_discharged", "provider_calls", "prover_calls", "model_inference_calls", "training_steps")))
    artifacts = manifest["artifacts"]
    _need(type(artifacts) is dict and set(artifacts) == {"summary", "capsules", "formal"}
        and type(context.get("artifacts")) is dict and set(context["artifacts"]) == set(artifacts))
    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        child_info = os.fstat(child)
        _need(child_info.st_uid == directory.st_uid and child_info.st_gid == directory.st_gid
            and stat.S_IMODE(child_info.st_mode) == 0o700
            and _identity(child_info) == _identity(os.stat(name, dir_fd=fd, follow_symlinks=False)))
        pins, observed_files = {}, {}
        total = 0
        for artifact in sorted(artifacts):
            binding = artifacts[artifact]
            _need(type(binding) is dict and set(binding) == {"path", "sha256", "bytes"}
                and binding["path"] == name + "/" + artifact + ".json"
                and type(binding["bytes"]) is int and 0 < binding["bytes"] <= checkpoint.MAX_ARTIFACT_BYTES)
            _hex(binding["sha256"])
            total += binding["bytes"]
            _need(total <= checkpoint.MAX_ARTIFACT_BYTES)
            pin = dict(sha256=binding["sha256"], bytes=binding["bytes"])
            _need(context["artifacts"][artifact] == pin
                and type(context["artifacts"][artifact].get("bytes")) is int)
            body, observed_files[artifact] = _private_read(child, artifact + ".json", child_info,
                limit=binding["bytes"], with_identity=True)
            _need(len(body) == binding["bytes"] and _sha(body) == binding["sha256"])
            pins[artifact] = pin
        # Bracket artifact hashing by the stable manifest, then recheck every
        # file identity. This observation is not a lock or replay authority.
        _need(raw == _private_read(fd, name + ".json", directory, limit=checkpoint.MAX_MANIFEST_BYTES))
        _need(total == observation["artifact_bytes"]
            and _identity(child_info) == _identity(os.fstat(child))
            and _identity(child_info) == _identity(os.stat(name, dir_fd=fd, follow_symlinks=False)))
        for artifact, before in observed_files.items():
            after = os.stat(artifact + ".json", dir_fd=child, follow_symlinks=False)
            _need(_identity(before) == _identity(after) and before.st_gid == after.st_gid)
    finally:
        os.close(child)
    return dict(session_id_sha256=_sha(session.encode()),
        invocation_id_sha256=_sha(row["invocation_id"].encode()),
        descriptor_sha256=observation["descriptor_sha256"],
        native_prompt_sha256=observation["native_prompt_sha256"],
        workspace_generation=observation["workspace_generation"],
        checkpoint_manifest_sha256=_sha(raw), checkpoint_manifest_bytes=len(raw),
        artifacts=pins, artifact_bytes=total, artifacts_count=3,
        elapsed_seconds=observation["elapsed_seconds"], status="prepared", **checkpoint.FLAGS)


def collect(*, parent, scope, report):
    """Join bounded owner checkpoints to this trial's coding source sessions.

    A wrapper can fail after a successful provider receipt, so a prepared
    checkpoint remains observable for that session. Missing evidence for a
    failed selected session is unknown, never zero preparation.
    """
    try:
        _need(type(report) is dict and type(report.get("provider_invocations")) is list
            and len(report["provider_invocations"]) <= 128
            and (report.get("unreceipted_provider_attempt") is None
                or report.get("unreceipted_provider_attempt") is False))
        selected, sessions, invocations = [], set(), set()
        for row in report["provider_invocations"]:
            _need(type(row) is dict)
            if "source_view_retrieval" not in row:
                continue
            _need(row.get("schema") == "router-implementation-invocation@1"
                and row.get("phase") == row.get("purpose") == "coding"
                and type(row.get("invocation_id")) is str and 0 < len(row["invocation_id"]) <= 512
                and row["invocation_id"] not in invocations
                and row.get("status") in {"failed", "provider_returned"})
            receipt = row["source_view_retrieval"]
            _need(type(receipt) is dict and receipt.get("schema") in {
                "worker-source-dispatch-observation@1", "worker-source-dispatch-observation@2",
                "worker-source-dispatch-observation@3", "worker-source-dispatch-observation@4"}
                and all(receipt.get(key) is False for key in (
                    "proof_authority", "execution_authority", "completion_authority")))
            session = _hex(receipt.get("session_id"), 32)
            _need(session not in sessions)
            sessions.add(session); invocations.add(row["invocation_id"])
            selected.append(row)
        if not selected:
            return _base("not_selected", "no_source_session_receipt")
        _need(len(selected) <= MAX_SESSIONS and type(scope) is dict
            and type(scope.get("task_alias")) is str and bool(scope["task_alias"])
            and type(report.get("worker_cleanup_returncode")) is int and report["worker_cleanup_returncode"] == 0)
        owner = os.geteuid()
        _need(owner > 0 and owner == os.getuid())
        parent = Path(parent)
        _protected_directory(parent, owner=owner, group=owner)
        fd = _open_directory(parent)
        try:
            directory = os.fstat(fd)
            _need(_identity(directory) == _identity(parent.lstat()))
            items = []
            for row in selected:
                name = "candidate-round-" + row["source_view_retrieval"]["session_id"] + ".json"
                try:
                    os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    _need(row["status"] == "provider_returned")
                    continue
                items.append(_session(fd, directory, row, scope))
            _protected_directory(parent, owner=owner, group=owner)
            _need(_identity(directory) == _identity(parent.lstat()))
        finally:
            os.close(fd)
        if not items:
            return _base("not_selected", "no_failed_source_session_checkpoint")
        elapsed = sum(item["elapsed_seconds"] for item in items)
        _need(math.isfinite(elapsed))
        result = _base("observed", "matched_owner_candidate_checkpoints")
        result.update(sessions=items, counts=dict(sessions=len(items),
            artifacts=sum(item["artifacts_count"] for item in items),
            artifact_bytes=sum(item["artifact_bytes"] for item in items),
            elapsed_seconds=elapsed),
            artifact_byte_scope="retained advisory artifacts; not delivered model input or tokens")
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError, OverflowError):
        return _base()


__all__ = ["collect"]
