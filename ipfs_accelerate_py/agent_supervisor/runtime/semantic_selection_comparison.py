"""Experimental, source-preserving comparison of native capsule nominations.

This control measures four complete rendered representations of one current
source inventory. It permits capsule selection only when both native contexts
still carry every original source byte. Declared dependency diagnostics never
establish semantic completeness, task acceptance, or a production policy.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath

from ..context.context_compiler import _utf8_chunks
from . import semantic_router_translation as codec
from .context_projection_comparison import measure_context_candidates
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter

SCHEMA = "supervisor-semantic-selection-comparison@1"
REFERENCE_CHUNK_BYTES = 6144
_MATCHED_PAYLOAD_FIELDS = (
    "schema", "task_id", "objective", "manifest", "scope_cid", "semantic_root_cid",
    "required_raw_paths", "preparation_bounds", "reconstruction", "refresh_lineage",
    "doctor_snapshot_id", "doctor_manifest_cid", "completion_authority",
)
_CHUNK_METADATA = {"chunk_index", "chunk_count", "artifact_content_id"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference_policy(refs, text):
    _require(tuple(ref.summary for ref in refs) == _utf8_chunks(text, max_bytes=REFERENCE_CHUNK_BYTES),
             "comparison requires matching 6144-byte reference chunk policy")
    policies = []
    for index, ref in enumerate(refs):
        prefix, _, number = ref.reference_id.rpartition(":")
        _require(prefix and number == f"{index + 1:04d}", "semantic reference prefix policy differs")
        policies.append({
            "prefix": prefix, "kind": ref.kind, "tier": ref.tier.value,
            "repository_id": ref.repository_id, "tree_id": ref.tree_id,
            "metadata": {key: value for key, value in ref.metadata.items() if key not in _CHUNK_METADATA},
        })
    _require(all(row == policies[0] for row in policies), "semantic reference policy changes across chunks")
    return policies[0]


def _capsules(payload):
    rows = payload.get("capsules")
    _require(type(rows) is list, "native capsule population required")
    indexed = {}
    for row in rows:
        _require(type(row) is dict and type(row.get("stable_symbol_id")) is str
                 and row["stable_symbol_id"] not in indexed, "duplicate or invalid native capsule identity")
        indexed[row["stable_symbol_id"]] = row
    return indexed


def _admissions(payload, capsules):
    rows = payload.get("admissions")
    _require(type(rows) is list and len(rows) == len(capsules), "native admission population differs")
    ids = {row["capsule_cid"] for row in capsules.values()}
    indexed = {}
    for row in rows:
        _require(type(row) is dict and type(row.get("ref")) is dict,
                 "native admission record required")
        identity = row["ref"].get("capsule_cid")
        _require(type(identity) is str and identity in ids and identity not in indexed,
                 "duplicate or foreign native admission identity")
        indexed[identity] = row
    return indexed


def _producer_index(repository, artifact, payload):
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SortedPairIndex,
        verify_block_bytes,
    )

    blocks = PurePosixPath(artifact).parent / "blocks"

    def identity(cid):
        _require(type(cid) is str and PurePosixPath(cid).name == cid and cid not in {".", ".."},
                 "invalid retained producer block identity")
        raw = codec._read(repository, str(blocks / cid))
        verify_block_bytes(cid, raw)
        return codec._parse(raw.decode())

    root = identity(payload["semantic_root_cid"])
    index_cid = root["capsule_index_cid"]
    index = SortedPairIndex.from_dict({**identity(index_cid), "index_cid": index_cid})
    return index_cid, dict(index.pairs)


def _pack_differences(full, selected):
    """Keep field identities and list deltas, never admission/source prose."""
    comparisons = {}

    def compare(path, left, right):
        row = {"equal": left == right, "full_sha256": _digest(left), "selected_sha256": _digest(right)}
        if type(left) is list and type(right) is list:
            left_ids, right_ids = sorted(_digest(item) for item in left), sorted(_digest(item) for item in right)
            row.update(full_count=len(left), selected_count=len(right),
                       removed_record_sha256=sorted(set(left_ids) - set(right_ids)),
                       added_record_sha256=sorted(set(right_ids) - set(left_ids)))
        comparisons[path] = row

    for field in sorted(set(full) | set(selected)):
        if field == "pack" and type(full.get(field)) is dict and type(selected.get(field)) is dict:
            for inner in sorted(set(full[field]) | set(selected[field])):
                compare("pack.pack." + inner, full[field].get(inner), selected[field].get(inner))
        else:
            compare("pack." + field, full.get(field), selected.get(field))
    return {"changed_fields": [path for path, row in comparisons.items() if not row["equal"]],
            "fields": comparisons, "raw_values_retained": False}


def compare_semantic_selection(*, repository: Path, full_prompt: str, selected_prompt: str,
                               workspace: Path, purpose: str, provider: str, model: str,
                               token_counter: PromptTokenCounter | None = None,
                               measurement_basis: str = "owner_declared_local_tokenizer",
                               max_input_tokens: int | None = None) -> dict:
    """Verify one source-preserving control and measure four complete inputs.

    Native source freshness and root-backed full capsules are checked before and
    after counting. Selection is an advisory lexical nomination, not a lossless
    transformation of the full decision context. The complete local user-message
    counts exclude provider instructions and subsequent calls/tool turns.
    """
    _require(isinstance(repository, Path), "explicit canonical repository Path required")
    repository = repository.resolve(strict=True)
    for prompt in (full_prompt, selected_prompt):
        _require(type(prompt) is str and bool(prompt) and len(prompt.encode()) <= MAX_MODEL_PROMPT_BYTES,
                 "bounded literal compiled prompt required")
    if token_counter is not None:
        _require(isinstance(token_counter, PromptTokenCounter)
                 and token_counter.provider == provider and token_counter.model == model,
                 "token counter differs from pinned provider/model")

    encoded = [codec.encode_semantic_router_prompt(prompt=prompt, repository=repository,
                slot_policy_revision=codec.SLOT_POLICY_V2) for prompt in (full_prompt, selected_prompt)]
    for entry, prompt in zip(encoded, (full_prompt, selected_prompt), strict=False):
        _require(codec.restore_semantic_router_prompt(provider_prompt=entry.provider_prompt,
                 table=entry.table, repository=repository) == prompt, "native alias restoration differs")
    full_wire, full_suffix, full_positions, full_refs, full_text, full_artifact = codec._prompt_parts(full_prompt)
    selected_wire, selected_suffix, selected_positions, selected_refs, selected_text, selected_artifact = (
        codec._prompt_parts(selected_prompt))
    _require({key: value for key, value in full_wire.items() if key != "evidence"}
             == {key: value for key, value in selected_wire.items() if key != "evidence"},
             "compiled immutable core differs")
    _require(full_suffix == selected_suffix, "literal prompt suffix differs")
    for positions in (full_positions, selected_positions):
        _require(positions == list(range(positions[0], positions[0] + len(positions))),
                 "semantic reference block must be contiguous")
    _require(full_positions[0] == selected_positions[0]
             and [row for index, row in enumerate(full_wire["evidence"]) if index not in full_positions]
             == [row for index, row in enumerate(selected_wire["evidence"]) if index not in selected_positions],
             "unrelated evidence or semantic insertion position differs")
    _require(_reference_policy(full_refs, full_text) == _reference_policy(selected_refs, selected_text),
             "semantic reference coverage or rendering policy differs")

    full, selected = json.loads(full_text), json.loads(selected_text)
    _require(full.get("schema") == selected.get("schema") == "supervisor-semantic-worker-context@1",
             "full native worker artifacts required")
    _require("worker_projection" not in full, "full control cannot already be a worker projection")
    _require(all(full.get(key) == selected.get(key) for key in _MATCHED_PAYLOAD_FIELDS),
             "semantic task/source/scope/root or preparation bindings differ")
    manifest = full["manifest"]
    _require(type(full.get("raw_sources")) is dict and type(selected.get("raw_sources")) is dict
             and set(full["raw_sources"]) == set(selected["raw_sources"]) == set(manifest)
             and full["raw_sources"] == selected["raw_sources"],
             "comparison requires every scoped raw source identically embedded")
    full_capsules, selected_capsules = _capsules(full), _capsules(selected)
    _require(all(full_capsules.get(key) == value for key, value in selected_capsules.items()),
             "selected capsule is not an exact full-context subset")
    full_admissions, selected_admissions = _admissions(full, full_capsules), _admissions(selected, selected_capsules)
    _require(all(full_admissions.get(key) == value for key, value in selected_admissions.items()),
             "selected admission is not an exact full-context subset")
    index_cid, index = _producer_index(repository, full_artifact, full)
    _require(index == {key: value["capsule_cid"] for key, value in full_capsules.items()},
             "full capsule population differs from complete producer index")
    selected_index_cid, selected_index = _producer_index(repository, selected_artifact, selected)
    _require(selected_index_cid == index_cid and selected_index == index, "selected producer index differs")
    projection = selected.get("worker_projection")
    _require(type(projection) is dict and projection.get("schema") == "supervisor-semantic-worker-projection@1"
             and projection.get("selector") == "lexical-query-symbol-name@1"
             and type(projection.get("query")) is str and bool(projection["query"])
             and projection.get("semantic_equivalence_claimed") is False
             and projection.get("completion_authority") is False
             and projection.get("capsule_index_cid") == index_cid
             and projection.get("full_capsule_count") == len(full_capsules)
             and projection.get("selected_capsule_count") == len(selected_capsules)
             and projection.get("omitted_capsule_count") == len(full_capsules) - len(selected_capsules)
             and projection.get("raw_source_fetch_required") == {},
             "selected native projection bindings or denominator differ")
    selected_symbols = projection.get("selected_symbols")
    _require(type(selected_symbols) is list and len(selected_symbols) == len(selected_capsules)
             and len({row.get("stable_symbol_id") for row in selected_symbols if type(row) is dict}) == len(selected_symbols)
             and {row.get("stable_symbol_id") for row in selected_symbols if type(row) is dict} == set(selected_capsules)
             and all(row.get("path") == selected_capsules[row["stable_symbol_id"]]["source_slice_path"]
                     for row in selected_symbols), "selected symbol population differs")

    from .semantic_selection_dependency_audit import audit_selected_capsule_dependencies
    dependency_audit = audit_selected_capsule_dependencies(
        full_capsules=full["capsules"], selected_symbol_ids=list(selected_capsules))
    failures, counts, drift = [], {}, []

    def observe(text):
        try:
            count = token_counter.count(text)
        except Exception as error:
            failures.append(type(error).__name__)
            raise
        digest = sha256(text.encode()).hexdigest()
        if digest in counts and counts[digest] != count:
            drift.append(digest)
        counts[digest] = count
        return count

    observed_counter = (replace(token_counter, count_tokens=observe)
                        if token_counter is not None else None)
    arguments = dict(native_prompt=full_prompt, suffix="", purpose=purpose, workspace=workspace,
                     provider=provider, model=model, token_counter=observed_counter,
                     measurement_basis=measurement_basis, max_input_tokens=max_input_tokens)
    full_measurement = measure_context_candidates(projected_prompt=full_prompt,
        alias_prompt=encoded[0].provider_prompt, **arguments)
    selected_measurement = measure_context_candidates(projected_prompt=selected_prompt,
        alias_prompt=encoded[1].provider_prompt, **arguments)
    for measurement in (full_measurement, selected_measurement):
        if "failed" in measurement["measurement_status"].values():
            failures.append("TokenCountFailed")
    rows = {
        "full_native": deepcopy(full_measurement["candidates"]["native"]),
        "full_v2_aliases": deepcopy(full_measurement["candidates"]["alias"]),
        "query_selected_native": deepcopy(selected_measurement["candidates"]["projected"]),
        "query_selected_v2_aliases": deepcopy(selected_measurement["candidates"]["alias"]),
    }
    status = ("failed" if failures or drift else "unavailable" if token_counter is None
              else "estimated" if token_counter.count_kind == "estimate" else "measured")
    if status == "failed":
        for row in rows.values():
            row["input_tokens"] = None
            if "estimated_input_tokens" in row:
                row["estimated_input_tokens"] = None
        if max_input_tokens is not None:
            raise ValueError("comparison token budget requires a working consistent pinned counter")
    # Counter callbacks are external inputs. Recheck source/parent identity after
    # observing them before publishing any current-source comparison.
    for entry, prompt in zip(encoded, (full_prompt, selected_prompt), strict=False):
        _require(codec.restore_semantic_router_prompt(provider_prompt=entry.provider_prompt,
                 table=entry.table, repository=repository) == prompt, "native alias restoration differs after counting")
    baseline = rows["full_native"]
    return {
        "schema": SCHEMA, "provider": provider, "model": model,
        "task_id": full["task_id"], "scope_cid": full["scope_cid"], "semantic_root_cid": full["semantic_root_cid"],
        "measurement_basis": measurement_basis, "count_scope": "rendered_user_prompt_only",
        "measurement_status": status,
        "measurement_error_type": "InconsistentTokenCount" if drift else failures[0] if failures else None,
        "token_measurements_consistent": True if status in {"measured", "estimated"} else False if status == "failed" else None,
        "tokenizer_binding": token_counter.binding() if token_counter is not None else None,
        "token_count_kind": token_counter.count_kind if token_counter is not None else None,
        "tokenizer_compatibility_basis": token_counter.compatibility_basis if token_counter is not None else None,
        "representations": rows,
        "comparisons_to_full_native": {name: {
            "rendered_user_byte_delta": baseline["model_prompt_bytes"] - row["model_prompt_bytes"],
            "rendered_user_content_token_delta": baseline["input_tokens"] - row["input_tokens"] if status == "measured" else None,
            "rendered_user_estimated_content_token_delta": (
                baseline["estimated_input_tokens"] - row["estimated_input_tokens"] if status == "estimated" else None),
        } for name, row in rows.items()},
        "matched_source_inventory": {"source_count": len(manifest), "manifest_sha256": _digest(manifest),
            "all_scoped_raw_sources_identically_embedded": True, "source_freshness_verified": True},
        "matched_compiled_core": True, "matched_literal_suffix": True, "matched_unrelated_evidence": True,
        "matching_reference_chunk_policy": {"chunk_bytes": REFERENCE_CHUNK_BYTES,
            "full_chunks": len(full_refs), "selected_chunks": len(selected_refs), "verified": True},
        "full_capsule_index_verified": True, "capsule_index_cid": index_cid,
        "selection": {"selector": projection["selector"], "query_sha256": sha256(projection["query"].encode()).hexdigest(),
            "full_capsule_count": len(full_capsules), "selected_capsule_count": len(selected_capsules),
            "omitted_capsule_count": len(full_capsules) - len(selected_capsules),
            "selected_capsules_and_admissions_exact_subsets": True, "raw_source_fetch_required_count": 0},
        "pack_decision_differences": _pack_differences(full["pack"], selected["pack"]),
        "dependency_audit": dependency_audit,
        "exact_restoration_targets": {"full_v2_aliases": "full_native", "query_selected_v2_aliases": "query_selected_native"},
        "alias_restoration_verified": True, "provider_compatibility_verified": False,
        "dependency_closure_verified": False, "program_equivalence_verified": False,
        "semantic_equivalence_claimed": False, "task_acceptance_verified": False, "deployment_ready": False,
        "proof_authority": False, "source_substitution_authority": False,
        "runtime_activation": False, "provider_calls": 0, "downloads": 0,
        "whole_session_token_savings": None, "campaign_usage_complete": None,
        "execution_authority": False, "completion_authority": False,
    }
