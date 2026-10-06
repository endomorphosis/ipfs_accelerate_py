"""Bounded offline transport sizing from cold public task inputs only.

The native envelope is a synthetic public-task fixture. No historical model
input, hidden verifier, reference/candidate solution, rollout or thought is read.
No provider, model, benchmark, task-source execution or training is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from unittest.mock import patch

MAX_INPUT_BYTES = 1_000_000
MAX_MODULE_BYTES = 8_000_000
MAX_PROMPT_BYTES = 262_144
RANK_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
RANK_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
PUBLIC_INPUTS = ("instruction.md", "environment/src/eigen.py", "environment/src/eval.py")
BASE_SCHEMA = "supervisor-semantic-router-input@1"
COMPACT_SCHEMA = "supervisor-semantic-router-input@2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(path, maximum=MAX_INPUT_BYTES):
    path = Path(path).absolute()
    before = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISREG(before.st_mode)
            or before.st_size > maximum):
        raise ValueError("bounded canonical regular file required")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("input changed during bounded read")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


@contextmanager
def offline_network_guard():
    def refused(*args, **kwargs):
        raise RuntimeError("network access forbidden during offline measurement")
    # Preserve the socket type: Trio's import-time type aliases refer to it.
    with patch.object(socket.socket, "connect", refused), patch.object(socket.socket, "connect_ex", refused), \
            patch.object(socket.socket, "sendto", refused), patch.object(socket, "create_connection", refused), \
            patch.object(socket, "getaddrinfo", refused):
        yield


def tokenizer_from_verified_local_ranks(path):
    import tiktoken
    import tiktoken.load
    import tiktoken_ext.openai_public as public

    raw, binding = read_bound(path, MAX_MODULE_BYTES)
    if binding["sha256"] != RANK_SHA256:
        raise ValueError("o200k_base rank SHA256 differs from the standard constructor pin")
    constructor_text = inspect.getsource(public.o200k_base)
    if RANK_SHA256 not in constructor_text or RANK_URL not in constructor_text:
        raise ValueError("installed o200k_base constructor pin differs")
    rank_path = str(Path(path).absolute())

    def local_rank_bytes(blobpath, expected_hash=None):
        if blobpath != rank_path or expected_hash not in (None, RANK_SHA256):
            raise ValueError("unexpected local tokenizer read")
        return raw

    # Use the installed BPE parser without fetching or writing generic caches.
    with patch.object(tiktoken.load, "read_file_cached", local_rank_bytes):
        ranks = tiktoken.load.load_tiktoken_bpe(rank_path, expected_hash=RANK_SHA256)

    def local_loader(blobpath, expected_hash=None):
        if blobpath != RANK_URL or expected_hash != RANK_SHA256:
            raise ValueError("unexpected installed tokenizer constructor dependency")
        return ranks

    with patch.object(public, "load_tiktoken_bpe", local_loader):
        configuration = public.o200k_base()
    tokenizer = tiktoken.Encoding(**configuration)
    source_raw, constructor_binding = read_bound(Path(public.__file__), MAX_MODULE_BYTES)
    return tokenizer, {"name": "o200k_base", "package": "tiktoken",
        "package_version": importlib.metadata.version("tiktoken"),
        "rank_source_url": RANK_URL, "expected_rank_sha256": RANK_SHA256,
        "local_rank_file": binding, "installed_constructor_module": constructor_binding,
        "constructor_text_sha256": sha(constructor_text.encode()),
        "rank_count": len(ranks), "provider_encoding_verified": False,
        "measurement": "local estimated input tokens, ordinary-text encoding only",
        "provider_tokens_observed": False, "network_calls_during_measurement": 0}


def git_identity(root):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, timeout=15).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
        "--untracked-files=no"], timeout=15)
    return {"path": str(root), "head": head, "tracked_status_sha256": sha(status),
            "tracked_status_bytes": len(status), "clean": not status}


def imported_checkout_bindings(roots):
    paths = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).absolute()
        if path.suffix == ".py" and any(path.is_relative_to(root) for root in roots):
            paths.add(path)
    if len(paths) > 2048:
        raise ValueError("imported source inventory exceeds audit bound")
    return [read_bound(path, MAX_MODULE_BYTES)[1] for path in sorted(paths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--public-task", type=Path, required=True)
    parser.add_argument("--tokenizer-ranks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, datasets, task = (path.resolve(strict=True) for path in
                               (args.checkout, args.datasets, args.public_task))
    if task.name != "largest-eigenval":
        raise ValueError("only the reviewed largest-eigenval public fixture is permitted")
    output = args.output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical fixture output directory required")
    sys.path[:0] = [str(checkout), str(datasets)]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    inputs, bodies = [], {}
    for relative in PUBLIC_INPUTS:
        raw, binding = read_bound(task / relative)
        raw.decode("utf-8")
        binding["role"] = "cold original public task input"
        binding["task_relative_path"] = relative
        inputs.append(binding)
        bodies[relative] = raw
    if sum(row["bytes"] for row in inputs) > MAX_INPUT_BYTES:
        raise ValueError("combined public input size exceeds scope bound")
    producer_raw, producer_binding = read_bound(Path(__file__), MAX_MODULE_BYTES)
    checkout_before, datasets_before = git_identity(checkout), git_identity(datasets)
    with offline_network_guard():
        from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
            ContextCompiler, build_text_context_references, render_context_capsule,
        )
        from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
        from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
        from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
        if getattr(codec, "COMPACT_TRANSPORT_SCHEMA", None) != COMPACT_SCHEMA:
            raise ValueError("implemented transport@2 required before offline comparison")
        tokenizer, tokenizer_metadata = tokenizer_from_verified_local_ranks(args.tokenizer_ranks)
        output.mkdir(parents=True)
        repository = output / "repository"
        repository.mkdir()
        for name in ("eigen.py", "eval.py"):
            (repository / name).write_bytes(bodies["environment/src/" + name])
        instruction = bodies["instruction.md"].decode("utf-8")
        # This declaration is deliberately synthetic; it is never relabelled
        # as a native historical planner/admission/implementation request.
        task_id = "public-largest-eigenval-offline-fixture"
        objective = "Inspect the original largest-eigenval public Python inputs."
        artifact_dir = repository / ".fixture/semantic"
        prepared = prepare_semantic_context(repository=repository,
            paths=["eigen.py", "eval.py"], required_raw_paths=["eigen.py", "eval.py"],
            objective=objective, task_id=task_id, output=artifact_dir,
            max_source_bytes=MAX_INPUT_BYTES, max_files=2, max_symbols=256,
            context_input_tokens=32768, max_payload_bytes=MAX_PROMPT_BYTES)
        artifact = artifact_dir / "worker-context.json"
        semantic_raw, semantic_binding = read_bound(artifact, MAX_PROMPT_BYTES)
        semantic = json.loads(semantic_raw)
        references = build_text_context_references(semantic_raw.decode("utf-8"),
            reference_prefix="semantic-context", kind="semantic-context",
            path=artifact.relative_to(repository).as_posix(), repository_id="repository:public-fixture",
            tree_id="tree:public-fixture", required=True)
        compiled = ContextCompiler(ContextBudget(max_input_tokens=65536, max_items=128,
            max_item_bytes=16384, max_serialized_bytes=MAX_PROMPT_BYTES)).compile(
                repository_id="repository:public-fixture", tree_id="tree:public-fixture",
                objective_id=task_id, objective_revision="sha256:" + sha(bodies["instruction.md"]),
                policy_id="policy:public-fixture", policy_revision="sha256:public-fixture",
                caller="supervisor:offline-fixture", stage="implementation",
                goal={"id": task_id, "objective": objective},
                authority={"mode": "candidate_only", "completion_authority": False},
                scope={"allowed_paths": ["eigen.py"], "read_only_evidence_paths": ["eval.py"]},
                acceptance={"criteria": ["Original public task remains the task request; offline sizing grants no task correctness or completion authority."]},
                evidence=references)
        native_prompt = render_context_capsule(compiled.capsule)
        doctor_suffix = ""  # No fabricated Doctor receipt or mathematical status.
        public_suffix = "\nOriginal public task instruction (verbatim):\n" + instruction
        shared_suffix = doctor_suffix + public_suffix
        native_model_prompt, native_workspace_advisory = render_model_prompt(
            prompt=native_prompt + shared_suffix, purpose="coding", workspace=repository,
            semantic_transport=False)
        native_complete = {"native_prompt_bytes": len(native_prompt.encode()),
            "native_prompt_sha256": sha(native_prompt.encode()),
            "complete_model_prompt_bytes": len(native_model_prompt.encode()),
            "complete_model_prompt_sha256": sha(native_model_prompt.encode()),
            "workspace_advisory_bytes": len(native_workspace_advisory.encode()),
            "workspace_advisory_sha256": sha(native_workspace_advisory.encode()),
            "estimated_complete_input_tokens_o200k_base": len(tokenizer.encode(native_model_prompt,
                allowed_special=set(), disallowed_special=()))}
        measurements, complete_prompts = [], {}
        for transport_schema in (BASE_SCHEMA, COMPACT_SCHEMA):
            transported = codec.encode_semantic_router_prompt(prompt=native_prompt,
                repository=repository, transport_schema=transport_schema)
            restored = codec.restore_semantic_router_prompt(provider_prompt=transported.provider_prompt,
                table=transported.table, repository=repository)
            if restored != native_prompt:
                raise ValueError("transport did not restore the exact native fixture")
            router_prompt = transported.provider_prompt + shared_suffix
            model_prompt, workspace_advisory = render_model_prompt(prompt=router_prompt,
                purpose="coding", workspace=repository, semantic_transport=True)
            if len(model_prompt.encode()) > MAX_PROMPT_BYTES:
                raise ValueError("complete fixture prompt exceeds runner byte budget")
            complete_prompts[transport_schema] = model_prompt
            mapping = transported.table.to_dict()
            measurements.append({"transport_schema": transport_schema,
                "native_prompt_bytes": len(native_prompt.encode()), "native_prompt_sha256": sha(native_prompt.encode()),
                "transport_prompt_bytes": len(transported.provider_prompt.encode()),
                "transport_prompt_sha256": sha(transported.provider_prompt.encode()),
                "doctor_suffix_bytes": len(doctor_suffix.encode()),
                "public_instruction_suffix_bytes": len(public_suffix.encode()),
                "shared_suffix_sha256": sha(shared_suffix.encode()),
                "router_prompt_bytes": len(router_prompt.encode()),
                "router_prompt_sha256": sha(router_prompt.encode()),
                "workspace_advisory_bytes": len(workspace_advisory.encode()),
                "workspace_advisory_sha256": sha(workspace_advisory.encode()),
                "complete_model_prompt_bytes": len(model_prompt.encode()),
                "complete_model_prompt_sha256": sha(model_prompt.encode()),
                "estimated_complete_input_tokens_o200k_base": len(tokenizer.encode(model_prompt,
                    allowed_special=set(), disallowed_special=())),
                "estimated_transport_input_tokens_o200k_base": len(tokenizer.encode(transported.provider_prompt,
                    allowed_special=set(), disallowed_special=())),
                "retained_controller_translation_table_bytes": len(transported.table.payload_json.encode()),
                "retained_controller_translation_table_sha256": sha(transported.table.payload_json.encode()),
                "transport_alias_dictionary_present": "translation_table" in json.loads(transported.provider_prompt),
                "native_bound_complete_prompt_bytes_saved": native_complete["complete_model_prompt_bytes"] - len(model_prompt.encode()),
                "identifier_mappings": len(mapping["entries"]),
                "identifier_occurrences": len(mapping["replacement_paths"]),
                "exact_native_roundtrip": True, "freshness_checked": transported.freshness_checked,
                "encoding_receipt": transported.receipt})
        baseline, compact = measurements
        for key in ("native_prompt_sha256", "doctor_suffix_bytes", "public_instruction_suffix_bytes",
                    "shared_suffix_sha256", "workspace_advisory_bytes", "workspace_advisory_sha256"):
            if baseline[key] != compact[key]:
                raise ValueError("complete comparison changed a common prompt component")
        imported = imported_checkout_bindings((checkout, datasets))
        packages = {}
        for name in ("ipfs-accelerate-py", "ipfs-datasets-py", "duckdb", "tiktoken"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "source_checkout_only"
    for item in inputs:
        if read_bound(item["path"])[1]["sha256"] != item["sha256"]:
            raise ValueError("original public task changed during comparison")
    if checkout_before != git_identity(checkout) or datasets_before != git_identity(datasets):
        raise ValueError("checkout identity changed during comparison; rerun after edits settle")
    result = {"schema": "terminal-public-task-transport-offline-sizing@1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_kind": "synthetic envelope from cold original public largest-eigenval task inputs",
        "historical_actual_prompt": False, "public_task": str(task), "public_inputs": inputs,
        "producer": producer_binding, "checkouts": [checkout_before, datasets_before],
        "imported_checkout_source_bindings": imported, "package_versions": packages,
        "tokenizer": tokenizer_metadata, "native_semantic_artifact": semantic_binding,
        "native_semantic_schema": semantic["schema"], "capsule_count": len(semantic["capsules"]),
        "semantic_root_cid": semantic["semantic_root_cid"], "scope_cid": semantic["scope_cid"],
        "native_producer_record_schema": prepared.get("schema"), "measurements": measurements,
        "unencoded_native_fixture_complete_prompt": native_complete,
        "fixture_settings": {"editable_paths": ["eigen.py"], "read_only_evidence_paths": ["eval.py"],
            "semantic_reference_chunk_bytes": inspect.signature(build_text_context_references).parameters["chunk_bytes"].default,
            "semantic_reference_count": len(references),
            "native_semantic_source_count": len(semantic["manifest"]),
            "capsule_population": "cold full scoped producer population; no candidate solution", "doctor_append_selected": False},
        "comparison": {
            "identical_native_prompt_and_append_components": True,
            "complete_prompt_bytes_saved": baseline["complete_model_prompt_bytes"] - compact["complete_model_prompt_bytes"],
            "estimated_complete_input_tokens_saved_o200k_base": baseline["estimated_complete_input_tokens_o200k_base"] - compact["estimated_complete_input_tokens_o200k_base"],
            "both_transports_roundtrip_exactly": True},
        "planning_preserved_experiment_protocol": {
            "required_planning_strategy": "direct or explicit @1 intent_coverage",
            "required_llm_phases": ["planning", "coding"], "required_router": "llm_router",
            "required_common_arm": "full", "required_common_model": "gpt-6.1-sol",
            "required_common_reasoning_effort": "high", "required_common_cli_version": "0.160.0",
            "required_common_resource_profile": "source384-5cpu-16gib-extended@1",
            "required_measurement": "sum complete native cumulative input+output tokens across retained sessions, retries and lookups; cached input stays included",
            "fresh_paired_baseline_required_for_attribution": True,
            "planning_session_observed_in_this_offline_measurement": False,
            "coding_session_observed_in_this_offline_measurement": False},
        "provider_calls": 0, "benchmark_runs": 0, "model_inferences": 0,
        "training_updates": 0, "hidden_verifier_bodies_read": False,
        "reference_or_candidate_solutions_read": False, "rollouts_or_thoughts_read": False,
        "task_source_executed": False, "original_task_inputs_unchanged": True,
        "measured_total_supervisor_token_savings": False,
        "limits": ["This is an offline representation comparison on a synthetic public-task envelope, not the retained historical prompt or a new official trial.",
            "o200k_base counts are local estimates of supplied initial input, not verified gpt-6.1 tokenization or observed provider totals.",
            "Provider harness, later native CLI model/tool turns, retries, cache costs and task quality are outside this fixture.",
            "No Doctor residual appendix was loaded; its omitted length is zero in both arms, and no proof status is invented.",
            "The treatment changes coding transport only; it leaves the retained planning session active in the proposed experiment."]}
    (output / "measurement.json").write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False) + "\n")
    print(json.dumps({"fixture_kind": result["fixture_kind"], "comparison": result["comparison"],
        "provider_calls": 0, "measured_total_supervisor_token_savings": False}, sort_keys=True))


if __name__ == "__main__":
    main()
