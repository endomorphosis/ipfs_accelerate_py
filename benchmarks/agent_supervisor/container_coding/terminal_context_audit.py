"""Bounded historical context/input audit, without exporting prompt contents.

The supervisor calls this after collecting router receipts and before Harbor
removes the container. Receipt workspaces remain usable as input identities
after workers exit or their worktrees are removed. This is observation only:
neither successful reconstruction nor a provider receipt grants authority or
establishes task correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time


SCHEMA = "terminal-final-context-audit@1"
KINDS = ("semantic-context", "code-retrieval-context", "intent-world-context")
MAX_FILE_BYTES = 2_000_000
MAX_INVENTORY_ENTRIES = 20_000
MAX_CAPSULES = 32
MAX_RECEIPTS = 16
MAX_DEPTH = 32
MAX_REQUEST_BYTES = 65_536
MAX_RESULT_BYTES = 262_144
RECEIPT_FIELDS = (
    "schema", "phase", "invocation_id", "purpose", "workspace",
    "prompt_sha256", "prompt_bytes", "native_prompt_sha256", "native_prompt_bytes",
    "model_prompt_sha256", "model_prompt_bytes", "workspace_advisory_sha256",
    "workspace_advisory_bytes",
)
TRANSLATION_FIELDS = ("router_prompt_sha256", "router_prompt_bytes", "semantic_translation", "doctor_residual_context", "public_instruction")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _base(status="unknown", reason="observation_unavailable") -> dict:
    return {"schema": SCHEMA, "status": status, "reason": reason,
        "any_native_input_verified": None, "any_model_input_verified": None,
        "all_observed_coding_inputs_verified": None,
        "capsules": [], "invocations": [], "matches": [], "read_errors": [],
        "world_scope": "sealed historical capture; claims may supersede its intent revision",
        "input_scope": "router task prompt; provider harness/system context is outside this digest",
        "raw_prompts_exported": False, "provider_calls": 0,
        "execution_authority": False, "completion_authority": False,
        "task_correctness_established": False}


def _open_directory(path: Path) -> int:
    """Open every absolute directory component without following a symlink."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("noncanonical directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_at(root_fd: int, relative: str, limit=MAX_FILE_BYTES) -> bytes:
    """Race-safe, bounded regular-file read under the already opened state."""
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(p in {".", ".."} for p in parts):
        raise ValueError("invalid state-relative file")
    parent = os.dup(root_fd)
    fd = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent)
            os.close(parent)
            parent = next_fd
        # NONBLOCK prevents a substituted FIFO from blocking before fstat.
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
            raise ValueError("nonregular or oversized state file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) > limit or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError("state file changed during observation")
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _json(raw: bytes):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique)


def _identity(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_:./-]{1,512}", value) is None:
        raise ValueError("invalid context identity")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid digest")
    return value


def _capsule_paths(root_fd: int) -> list[str]:
    result, entries = [], 0
    def failed(error):
        raise error
    for directory, dirs, files, _ in os.fwalk(".", topdown=True, follow_symlinks=False,
                                             dir_fd=root_fd, onerror=failed):
        entries += len(dirs) + len(files)
        if entries > MAX_INVENTORY_ENTRIES or len(PurePosixPath(directory).parts) > MAX_DEPTH:
            raise ValueError("state inventory exceeds audit bound")
        dirs.sort()
        for name in sorted(files):
            if name.endswith("-base-context-capsule.json"):
                result.append((PurePosixPath(directory) / name).as_posix())
                if len(result) > MAX_CAPSULES:
                    raise ValueError("capsule inventory exceeds audit bound")
    return result


def _receipt(item: dict, workspace_root: Path) -> dict:
    if not isinstance(item, dict) or item.get("schema") != "router-implementation-invocation@1":
        raise ValueError("invalid router receipt")
    if item.get("phase") != "coding" or item.get("purpose") != "coding":
        raise ValueError("non-coding router receipt")
    result = {key: item[key] for key in RECEIPT_FIELDS}
    result.update({key: item[key] for key in TRANSLATION_FIELDS if key in item})
    _identity(result["invocation_id"])
    location = result["workspace"]
    if not isinstance(location, str) or len(location.encode()) > 4096:
        raise ValueError("invalid workspace identity")
    workspace, root = Path(location), Path(workspace_root)
    if (not workspace.is_absolute() or ".." in workspace.parts
            or str(workspace) != location or not root.is_absolute()
            or workspace == root or not workspace.is_relative_to(root)):
        raise ValueError("workspace outside configured allocation root")
    for field in ("prompt_sha256", "native_prompt_sha256", "model_prompt_sha256", "workspace_advisory_sha256"):
        _digest(result[field])
    for field in ("prompt_bytes", "native_prompt_bytes", "model_prompt_bytes", "workspace_advisory_bytes"):
        if type(result[field]) is not int or not 0 < result[field] <= 262_144:
            raise ValueError("invalid input byte count")
    translated = result.get("semantic_translation") is not None
    instruction = result.get("public_instruction")
    router_bytes = result.get("router_prompt_bytes", result["native_prompt_bytes"])
    if translated or instruction is not None:
        _digest(result["router_prompt_sha256"])
        if type(router_bytes) is not int or not 0 < router_bytes <= 262_144:
            raise ValueError("invalid translated input byte count")
    if translated:
        translation = result["semantic_translation"]
        if (not isinstance(translation, dict)
                or translation.get("schema") not in {"supervisor-semantic-router-encoding@1",
                                                     "supervisor-semantic-router-encoding@2"}
                or translation.get("freshness_checked") is not True):
            raise ValueError("invalid semantic translation receipt")
        if (translation["schema"] == "supervisor-semantic-router-encoding@2"
                and translation.get("transport_schema") != "supervisor-semantic-router-input@2"):
            raise ValueError("invalid semantic transport version")
        if (translation["schema"] == "supervisor-semantic-router-encoding@1"
                and translation.get("transport_schema", "supervisor-semantic-router-input@1")
                != "supervisor-semantic-router-input@1"):
            raise ValueError("invalid semantic transport version")
        _identity(translation["translation_cid"])
    elif instruction is None and (router_bytes != result["native_prompt_bytes"]
          or result.get("router_prompt_sha256", result["native_prompt_sha256"]) != result["native_prompt_sha256"]):
        raise ValueError("unexplained router projection")
    residual = result.get("doctor_residual_context")
    if residual is not None:
        if (not translated or not isinstance(residual, dict)
                or residual.get("schema") != "doctor-residual-router-context@1"
                or residual.get("candidate_only") is not True):
            raise ValueError("invalid Doctor residual receipt")
        for key in ("artifact_sha256", "advisory_sha256"):
            _digest(residual[key])
        _identity(residual["task_cid"])
    if instruction is not None:
        if (not isinstance(instruction, dict)
                or instruction.get("schema") not in {"supervisor-public-instruction-inclusion@1", "supervisor-public-instruction-inclusion@2"}
                or any(instruction.get(k) is not True for k in
                       ("manifest_signature_verified", "verbatim_utf8", "source_freshness_verified"))
                or any(instruction.get(k) is not False for k in
                       ("semantic_minification_applied", "historical_replay", "completion_authority", "scope_expansion_authority"))):
            raise ValueError("invalid public instruction inclusion receipt")
        for key in ("artifact_sha256", "source_sha256", "block_sha256"):
            _digest(instruction[key])
        _identity(instruction["task_cid"])
        if instruction["schema"] == "supervisor-public-instruction-inclusion@2":
            requirements = instruction.get("intent_requirements")
            if (not isinstance(requirements, dict) or requirements.get("admission_graph_verified") is not True
                    or any(requirements.get(key) is not False for key in ("native_persistence_verified_here",
                        "source_semantics_verified", "semantic_alignment_verified", "proof_authority",
                        "execution_authority", "completion_authority"))):
                raise ValueError("invalid task-specific intent requirement receipt")
            for key in ("requirements_context_cid", "contract_cid", "graph_cid", "coverage_cid", "planning_receipt_cid"):
                _identity(requirements[key])
            _digest(requirements["ledger_sha256"])
            for key in ("requirement_ids", "dependency_requirement_ids", "prohibition_requirement_ids"):
                if (not isinstance(requirements.get(key), list) or len(requirements[key]) > 512
                        or any(not isinstance(item, str) for item in requirements[key])
                        or len(set(requirements[key])) != len(requirements[key])):
                    raise ValueError("invalid intent requirement identity population")
                for requirement_id in requirements[key]:
                    _identity(requirement_id)
    if (result["prompt_sha256"] != result["native_prompt_sha256"]
            or result["prompt_bytes"] != result["native_prompt_bytes"]
            or router_bytes + result["workspace_advisory_bytes"] != result["model_prompt_bytes"]):
        raise ValueError("inconsistent router input identities")
    return result


def _model_projection(*, rendered: str, receipt: dict, repository: Path | None):
    """Reconstruct historical provider bytes without granting freshness."""
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    translated = receipt.get("semantic_translation") is not None
    router_prompt, checks = rendered, {}
    if translated:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import replay_semantic_router_prompt_for_audit
        if repository is None:
            raise ValueError("translated receipt requires canonical artifact repository")
        encoded = replay_semantic_router_prompt_for_audit(prompt=rendered, repository=repository,
            transport_schema=receipt["semantic_translation"].get("transport_schema",
                                                               "supervisor-semantic-router-input@1"))
        router_prompt = encoded.provider_prompt
        checks = {
            "semantic_translation_cid": encoded.table.translation_cid == receipt["semantic_translation"]["translation_cid"],
            "semantic_translation_schema": encoded.receipt["schema"] == receipt["semantic_translation"]["schema"],
        }
    residual = receipt.get("doctor_residual_context")
    if residual is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import load_doctor_residual_advisory
        if not translated or repository is None:
            raise ValueError("Doctor residual requires a canonical semantic repository")
        advisory, observed = load_doctor_residual_advisory(artifact=Path(residual["artifact"]),
            expected_sha256=residual["artifact_sha256"], repository=repository,
            task_cid=residual["task_cid"], prompt=rendered, require_current_source=False)
        router_prompt += advisory
        for field in ("context_cid", "native_capsule_id", "advisory_sha256", "advisory_bytes"):
            checks["doctor_residual_" + field] = observed[field] == residual[field]
    instruction = receipt.get("public_instruction")
    if instruction is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
        if repository is None:
            raise ValueError("public instruction requires a canonical artifact repository")
        block, observed = load_public_instruction(artifact=Path(instruction["artifact"]),
            expected_sha256=instruction["artifact_sha256"], repository=repository,
            task_cid=instruction["task_cid"], prompt=rendered, require_current_source=False)
        router_prompt += block
        for field in ("schema", "context_cid", "manifest_cid", "task_id", "source_path", "source_sha256",
                      "source_bytes", "block_sha256", "block_bytes"):
            checks["public_instruction_" + field] = observed[field] == instruction[field]
        if instruction["schema"] == "supervisor-public-instruction-inclusion@2":
            checks["public_instruction_intent_requirements"] = observed.get("intent_requirements") == instruction["intent_requirements"]
    if translated or instruction is not None:
        checks.update(router_prompt_sha256=_sha(router_prompt.encode()) == receipt["router_prompt_sha256"],
            router_prompt_bytes=len(router_prompt.encode()) == receipt["router_prompt_bytes"])
    model, advisory = render_model_prompt(prompt=router_prompt, purpose="coding",
        workspace=Path(receipt["workspace"]), semantic_transport=translated)
    checks.update(model_prompt_sha256=_sha(model.encode()) == receipt["model_prompt_sha256"],
        model_prompt_bytes=len(model.encode()) == receipt["model_prompt_bytes"],
        workspace_advisory_sha256=_sha(advisory.encode()) == receipt["workspace_advisory_sha256"],
        workspace_advisory_bytes=len(advisory.encode()) == receipt["workspace_advisory_bytes"])
    return model, advisory, checks


def _capsule(raw: bytes, *, expected: dict, prepared: dict) -> tuple[dict, str]:
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
        ContextCapsule, canonical_context_json_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule

    capsule = ContextCapsule.from_dict(_json(raw))
    rendered = render_context_capsule(capsule)
    wire = _json(rendered.encode())
    contexts, chunks = {}, {}
    for kind in KINDS:
        refs = sorted((row for row in wire["evidence"] if row["kind"] == kind),
                      key=lambda row: row["reference_id"])
        if not refs or not all(row["metadata"].get("required") is True for row in refs):
            raise ValueError("required context missing")
        text = "".join(row["summary"] for row in refs)
        artifact = "sha256:" + _sha(canonical_context_json_bytes({"text": text}))
        for index, row in enumerate(refs):
            if (row["metadata"].get("chunk_index") != index
                    or row["metadata"].get("chunk_count") != len(refs)
                    or row["metadata"].get("artifact_content_id") != artifact
                    or row["referenced_content_id"] != "sha256:" + _sha(row["summary"].encode())):
                raise ValueError("context chunk identity mismatch")
        contexts[kind] = _json(text.encode())
        chunks[kind] = {"count": len(refs), "bytes": len(text.encode()), "sha256": _sha(text.encode())}
    semantic, retrieval, world = (contexts[kind] for kind in KINDS)
    if semantic.get("schema") == "supervisor-semantic-worker-context@2":
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _validate_program_payload
        _validate_program_payload(semantic)
    query = prepared["query"]
    if not isinstance(query, str) or len(query.encode()) > 32768:
        raise ValueError("invalid public query")
    sources = prepared["manifest"]["payload"]["sources"]
    source_hashes = retrieval["source_sha256"]
    expected_paths = {"bottle.py"}
    if prepared.get("task_profile") is not None:
        from .terminal_task_profile import (PROFILE, validate_task_profile,
            task_profile_bytes, task_profile_index_paths)
        profile = validate_task_profile(prepared["task_profile"], instruction=query)
        if sources.get(PROFILE, {}).get("sha256") != _sha(task_profile_bytes(profile)):
            raise ValueError("retrieval scope profile differs from signed source")
        expected_paths = set(task_profile_index_paths(profile))
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_paths:
        raise ValueError("unexpected retrieval source scope")
    empty = retrieval.get("retrieval_schema") == "supervisor-empty-code-retrieval@1"
    if empty:
        from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import validate_empty_worker_context
        from .terminal_task_profile import INSTRUCTION, PROFILE, SMOKE
        validate_empty_worker_context(retrieval)
        support = {name: {"role": role, "sha256": sources[name]["sha256"]} for name, role in (
            (INSTRUCTION, "instruction"), (PROFILE, "task_profile"), (SMOKE, "structural_smoke"))}
        if (prepared.get("task_profile") is None
                or retrieval["program_paths"] != sorted(expected_paths)
                or retrieval["original_support_hashes"] != support
                or semantic.get("schema") != "supervisor-semantic-worker-context@2"
                or semantic.get("program_paths") != sorted(expected_paths)
                or set(sources) != expected_paths | set(support)
                or any(retrieval[key] != expected.get(key) for key in (
                    "retrieval_schema", "source_population_cid", "disposition", "embedding_calls"))):
            raise ValueError("historical empty population differs from signed task source roles")
    elif retrieval["index_id"] is None or expected.get("retrieval_schema") == "supervisor-empty-code-retrieval@1":
        raise ValueError("historical retrieval lane differs from captured source population")
    tasks = [row for row in world.get("tasks", []) if row.get("task_cid") == expected["task_cid"]]
    checks = {
        "task_alias": semantic["task_id"] == retrieval["task_id"] == wire["objective_id"] == prepared["spec"]["task_key"],
        "semantic_root": semantic["semantic_root_cid"] == expected["semantic_root_cid"] == world["semantic_root_cid"],
        "vector_root": retrieval["index_id"] == expected["index_id"],
        "world_root": world["world_snapshot_cid"] == expected["world_snapshot_cid"],
        "public_query": retrieval["query_text"] == query,
        "source_hashes": all(_digest(digest) == sources[path]["sha256"] for path, digest in source_hashes.items()),
        "semantic_source_hashes": all(semantic["manifest"][path]["sha256"] == digest for path, digest in source_hashes.items()),
        "semantic_input_scope": set(semantic["manifest"]) == set(prepared["worker_inputs"]),
        "semantic_captured_hashes": all(binding["sha256"] == sources[path]["sha256"]
            for path, binding in semantic["manifest"].items()),
        "semantic_program_scope": (semantic.get("program_paths") == sorted(expected_paths)
            if semantic.get("schema") == "supervisor-semantic-worker-context@2" else "program_paths" not in semantic),
        "current_retrieval_at_capture": retrieval["status"] == "current",
        "nomination_only": retrieval.get("nomination_only") is True,
        "no_retrieval_authority": all(retrieval.get(k) is False for k in ("semantic_authority", "execution_authority", "completion_authority")),
        "no_world_authority": all(world.get(k) is False for k in ("execution_authority", "completion_authority")),
        "no_semantic_completion": semantic.get("completion_authority") is False,
        "canonical_task_contract": len(tasks) == 1 and bool(tasks[0].get("body", {}).get("local_planning_contract_cid")),
    }
    result = {"capsule_id": _identity(capsule.capsule_id), "persisted_capsule_sha256": _sha(raw),
        "native_prompt_sha256": _sha(rendered.encode()), "native_prompt_bytes": len(rendered.encode()),
        "context_chunks": chunks, "checks": checks, "all_context_checks_passed": all(checks.values()),
        "semantic_root_cid": _identity(semantic["semantic_root_cid"]),
        "index_id": None if empty else _identity(retrieval["index_id"]), "retrieval_result_id": _identity(retrieval["result_id"]),
        "retrieval_query_id": _identity(retrieval["query_id"]),
        "world_snapshot_cid": _identity(world["world_snapshot_cid"]),
        "plan_projection_cid": _identity(world["plan_projection_cid"]),
        "task_cid": _identity(expected["task_cid"]), "public_query_sha256": _sha(query.encode()),
        "source_sha256": {key: _digest(value) for key, value in source_hashes.items()},
        "intent_freshness_checked": world.get("intent_freshness_checked") is True}
    if empty:
        result.update(retrieval_schema=retrieval["retrieval_schema"],
            source_population_cid=_identity(retrieval["source_population_cid"]),
            disposition=retrieval["disposition"], embedding_calls=0)
    if len(tasks) == 1:
        result["local_contract_cid"] = _identity(tasks[0].get("body", {}).get("local_planning_contract_cid"))
    return result, rendered


def audit_terminal_context(*, state: Path, receipts: list[dict], workspace_root: Path) -> dict:
    """Read historical evidence and compare exact native and model input bytes."""
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt

    result = _base()
    if len(receipts) > MAX_RECEIPTS:
        raise ValueError("receipt inventory exceeds audit bound")
    parsed, seen = [], set()
    result["coding_receipts_seen"] = len(receipts)
    for item in receipts:
        try:
            value = _receipt(item, workspace_root)
            if value["invocation_id"] in seen:
                raise ValueError("duplicate router invocation")
            seen.add(value["invocation_id"])
            parsed.append(value)
        except (ValueError, KeyError, TypeError):
            result["read_errors"].append({"kind": "receipt", "error_type": "InvalidReceipt"})
    result["usable_coding_receipts"] = len(parsed)
    if not parsed:
        result["reason"] = "no_usable_coding_receipt"
        return result
    result["invocations"] = [{"invocation_id": row["invocation_id"],
        "native_input_verified": None, "model_input_verified": None,
        "status": "unverified", "reason": "native_capsule_unavailable"} for row in parsed]
    root_fd = _open_directory(state)
    try:
        expected_raw = _read_at(root_fd, "context-result.json")
        prepared_raw = _read_at(root_fd, "prepared.json")
        expected, prepared = _json(expected_raw), _json(prepared_raw)
        result.update(context_result_sha256=_sha(expected_raw), prepared_state_sha256=_sha(prepared_raw))
        paths = _capsule_paths(root_fd)
        if not paths:
            result["reason"] = "native_capsule_unavailable"
            return result
        for relative in paths:
            try:
                observed, rendered = _capsule(_read_at(root_fd, relative), expected=expected, prepared=prepared)
                result["capsules"].append(observed)
                if not observed["all_context_checks_passed"]:
                    continue
                for receipt in parsed:
                    if (observed["native_prompt_sha256"] != receipt["native_prompt_sha256"]
                            or observed["native_prompt_bytes"] != receipt["native_prompt_bytes"]):
                        continue
                    model, advisory, checks = _model_projection(rendered=rendered, receipt=receipt,
                        repository=Path(prepared["repository"]) if prepared.get("repository") else None)
                    result["matches"].append({"capsule_id": observed["capsule_id"],
                        "invocation_id": receipt["invocation_id"],
                        "workspace_sha256": _sha(receipt["workspace"].encode()),
                        "native_prompt_sha256": observed["native_prompt_sha256"],
                        "native_prompt_bytes": observed["native_prompt_bytes"],
                        "model_prompt_sha256": _sha(model.encode()), "model_prompt_bytes": len(model.encode()),
                        "workspace_advisory_sha256": _sha(advisory.encode()),
                        "workspace_advisory_bytes": len(advisory.encode()),
                        "model_input_checks": checks, "model_input_verified": all(checks.values())})
            except Exception as error:
                result["read_errors"].append({"kind": "capsule", "error_type": type(error).__name__})
    finally:
        os.close(root_fd)
    for invocation in result["invocations"]:
        matching = [row for row in result["matches"] if row["invocation_id"] == invocation["invocation_id"]]
        invocation["native_input_verified"] = bool(matching)
        invocation["model_input_verified"] = any(row["model_input_verified"] for row in matching)
        invocation["status"] = ("verified_exact_model_input" if invocation["model_input_verified"]
            else "verified_native_input_only" if matching else "unverified")
        invocation["reason"] = ("exact_context_and_input_digests_match" if invocation["model_input_verified"]
            else "model_projection_mismatch" if matching else "no_exact_stored_capsule_match")
    result["any_native_input_verified"] = any(row["native_input_verified"] for row in result["invocations"])
    result["any_model_input_verified"] = any(row["model_input_verified"] for row in result["invocations"])
    result["all_observed_coding_inputs_verified"] = (len(parsed) == len(receipts)
        and all(row["model_input_verified"] for row in result["invocations"]))
    result["status"] = ("verified_all_observed_model_inputs" if result["all_observed_coding_inputs_verified"]
        else "verified_some_observed_model_inputs" if result["any_model_input_verified"]
        else "verified_native_input_only" if result["any_native_input_verified"] else "unverified")
    result["reason"] = "per_invocation_context_and_input_comparison"
    return result


def collect_terminal_context_audit(*, state: Path, receipts: list[dict],
        workspace_root: Path, timeout_seconds: float = 3.0) -> dict:
    """Run in a time-bounded child; collection failure cannot affect completion."""
    started = time.monotonic()
    if timeout_seconds < .1:
        return _base(reason="audit_budget_unavailable")
    try:
        rows = [{key: row[key] for key in (*RECEIPT_FIELDS, *TRANSLATION_FIELDS) if key in row}
                for row in receipts if isinstance(row, dict) and row.get("phase") == "coding"]
        request = json.dumps(rows).encode()
        if len(rows) > MAX_RECEIPTS or len(request) > MAX_REQUEST_BYTES:
            return _base(reason="audit_request_exceeds_bound")
        if not rows:
            return _base(reason="no_coding_receipt")
        # The owner may still have the task checkout as cwd. Never prepend
        # that worker-controlled directory to the trusted module search path.
        process = subprocess.run([sys.executable, "-B", "-P", "-m", __name__, "--state", str(state),
            "--workspace-root", str(workspace_root)], input=request, capture_output=True,
            timeout=min(10.0, timeout_seconds), check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if process.returncode or len(process.stdout) > MAX_RESULT_BYTES:
            return _base(reason="audit_process_unavailable")
        result = _json(process.stdout)
        if not isinstance(result, dict) or result.get("schema") != SCHEMA:
            return _base(reason="audit_result_invalid")
        result["seconds"] = time.monotonic() - started
        return result
    except subprocess.TimeoutExpired:
        return _base(reason="audit_timeout")
    except Exception as error:
        result = _base(reason="audit_unavailable")
        result["error_type"] = type(error).__name__
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("audit request exceeds bound")
        receipts = _json(raw)
        if not isinstance(receipts, list):
            raise ValueError("audit receipts must be a list")
        result = audit_terminal_context(state=args.state, receipts=receipts, workspace_root=args.workspace_root)
    except Exception as error:
        result = _base(reason="audit_unavailable")
        result["error_type"] = type(error).__name__
    encoded = json.dumps(result, sort_keys=True)
    if len(encoded.encode()) > MAX_RESULT_BYTES:
        encoded = json.dumps(_base(reason="audit_result_exceeds_bound"))
    print(encoded)


if __name__ == "__main__":
    main()
