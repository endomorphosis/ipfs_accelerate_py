"""Replay signed Terminal context selection without changing full task custody.

The historical observation checks a retained declaration and source metadata.
The live helper additionally verifies the complete signed input population and
reads selected originals. Neither observation admits a native allocation,
authorizes source omission, or establishes program semantics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ..core.multiformats_identity import cid_for_dag_json, digest_hex_from_cid, validate_cid
from ..proof.formal_verification_contracts import content_identity
from .terminal_task_profile import (
    INSTRUCTION, PROFILE, SMOKE, SCOPED_SCHEMA,
    task_profile_bytes, task_profile_index_paths, task_profile_semantic_inputs,
    task_profile_semantic_required_paths, task_profile_worker_inputs,
    task_profile_smoke, validate_task_profile,
)

SCHEMA = "terminal-signed-semantic-scope-replay@1"
FRONTIER_SCHEMA = "terminal-signed-semantic-scope-frontier@1"
MAX_FILES = 64
MAX_SOURCE_BYTES = 1_000_000


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def terminal_semantic_scope_observation(profile, manifest_envelope, *, selected_sources=None):
    """Check retained selection metadata without reading the current checkout.

    Signed local source rows contain SHA256 and executable mode, without byte
    lengths or raw CIDs. Those additional values must therefore be supplied by
    a separately retained capture. This checks their consistency and bounds;
    it does not authenticate the envelope signature or establish currentness.
    """
    _need(type(manifest_envelope) is dict and type(manifest_envelope.get("payload")) is dict,
          "signed semantic scope requires its retained manifest envelope")
    manifest = manifest_envelope["payload"]
    sources = manifest.get("sources")
    _need(type(sources) is dict, "retained complete signed source inventory required")
    if PROFILE not in sources:
        _need(profile is None, "unsigned profile cannot override an absent signed declaration")
        return None
    _need(type(profile) is dict, "signed public profile cannot be omitted")
    canonical = validate_task_profile(profile)
    profile_sha = hashlib.sha256(task_profile_bytes(canonical)).hexdigest()
    _need(type(sources[PROFILE]) is dict and sources[PROFILE].get("sha256") == profile_sha,
          "semantic scope declaration differs from signed profile bytes")
    if canonical["schema"] != SCOPED_SCHEMA:
        return None
    _need(_bytes(profile) == _bytes(canonical), "canonical signed semantic scope profile required")
    _need(type(sources) is dict and set(sources) == set(task_profile_worker_inputs(canonical)),
          "signed semantic scope differs from complete public source inventory")
    for binding in sources.values():
        _need(type(binding) is dict and set(binding) == {"sha256", "executable"}
              and _digest(binding["sha256"]) and type(binding["executable"]) is bool,
              "closed signed source binding required")
    _need(canonical["instruction_sha256"] == sources[INSTRUCTION]["sha256"]
          and hashlib.sha256(task_profile_smoke(canonical).encode("utf-8")).hexdigest()
          == sources[SMOKE]["sha256"],
          "retained instruction or structural smoke differs from canonical public contract")
    paths = sorted(task_profile_semantic_inputs(canonical))
    required = sorted(task_profile_semantic_required_paths(canonical))
    program = sorted(task_profile_index_paths(canonical))
    _need(len(paths) <= MAX_FILES and set(required) <= set(paths)
          and set(program) <= set(paths) and set(paths) - set(program) <= set(required),
          "signed semantic selection violates its closed role or file bounds")
    _need(type(selected_sources) is dict and set(selected_sources) == set(paths),
          "retained selected source metadata required for exact semantic scope sizes")
    selected, total = {}, 0
    for name in paths:
        row = selected_sources[name]
        _need(type(row) is dict and set(row) == {
            "sha256", "source_cid", "source_bytes", "executable", "role", "required"},
            "closed selected source metadata required")
        _need(row["sha256"] == sources[name]["sha256"]
              and type(row["executable"]) is bool
              and row["executable"] == sources[name]["executable"]
              and type(row["source_bytes"]) is int and 0 <= row["source_bytes"] <= MAX_SOURCE_BYTES
              and row["role"] == ("program" if name in program else "support")
              and type(row["required"]) is bool and row["required"] == (name in required),
              "selected source metadata differs from signed declaration")
        _need(type(row["source_cid"]) is str and len(row["source_cid"]) <= 128,
              "bounded canonical selected raw source CID required")
        validate_cid(row["source_cid"], codecs=("raw",))
        _need(digest_hex_from_cid(row["source_cid"], codecs=("raw",)) == row["sha256"],
              "selected raw source CID differs from signed SHA256")
        total += row["source_bytes"]
        _need(total <= MAX_SOURCE_BYTES, "signed semantic source population exceeds byte bound")
        selected[name] = dict(row)
    manifest_cid = content_identity(manifest_envelope)
    omitted = [{"path": name, **sources[name]} for name in sorted(set(sources) - set(paths))]
    frontier = {"schema": FRONTIER_SCHEMA, "manifest_cid": manifest_cid, "sources": omitted}
    return dict(schema=SCHEMA, manifest_cid=manifest_cid,
        profile_sha256=profile_sha, declaration_cid=cid_for_dag_json(canonical["semantic_scope"]),
        selected_paths=paths, required_raw_paths=required, program_paths=program,
        selected_sources=selected, selected_count=len(paths), full_count=len(sources),
        frontier_count=len(omitted), frontier_cid=cid_for_dag_json(frontier),
        selected_source_bytes=total, max_files=MAX_FILES, max_source_bytes=MAX_SOURCE_BYTES,
        historical_only=True, source_currentness_verified=False, signature_authority=False,
        source_semantics_verified=False, proof_authority=False, execution_authority=False,
        completion_authority=False, omission_authority=False)


def replay_terminal_semantic_scope(prepared):
    """Verify the complete current signed population before selecting context."""
    from . import local_planning_admission as local
    from .terminal_source_partition import _read, _unique_pairs, terminal_profile_partition
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    _need(type(prepared) is dict, "exact Terminal preparation required")
    frozen = _bytes(prepared)
    manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
    root = Path(prepared["repository"]).absolute()
    _need(root.resolve(strict=True) == root and str(root) == manifest["repository"],
          "semantic scope repository differs from signed input")
    sources = manifest["sources"]
    if PROFILE not in sources:
        _need(prepared.get("task_profile") is None and _bytes(prepared) == frozen,
              "unsigned profile cannot replace an absent signed declaration")
        return None
    raw_profile = _read(root, PROFILE, sources, 65536)
    captured = json.loads(raw_profile, object_pairs_hook=_unique_pairs)
    instruction = _read(root, INSTRUCTION, sources, 32768).decode("utf-8")
    canonical = validate_task_profile(captured, instruction=instruction)
    _need(raw_profile == task_profile_bytes(canonical)
          and _bytes(prepared.get("task_profile")) == _bytes(canonical),
          "prepared public profile differs from its actual signed declaration")
    if canonical["schema"] != SCOPED_SCHEMA:
        _need(_bytes(prepared) == frozen, "signed public preparation changed during replay")
        return None
    partition = terminal_profile_partition(repository=root, manifest=manifest)
    _need(partition is not None, "semantic scope requires the complete signed public partition")
    _need(prepared.get("query") == instruction
          and prepared.get("worker_inputs") == task_profile_worker_inputs(canonical),
          "semantic scope preparation differs from its exact signed public declaration")
    paths = sorted(task_profile_semantic_inputs(canonical))
    required = set(task_profile_semantic_required_paths(canonical))
    program = set(task_profile_index_paths(canonical))
    _need(len(paths) <= MAX_FILES and program <= set(partition.program_paths)
          and {INSTRUCTION, SMOKE, PROFILE} <= set(paths)
          and set(paths) - program <= required,
          "selected semantic scope differs from the complete signed program/support partition")
    selected, total = {}, 0
    for name in paths:
        raw = _read(root, name, sources, MAX_SOURCE_BYTES - total)
        raw.decode("utf-8")
        total += len(raw)
        selected[name] = dict(sha256=sources[name]["sha256"], source_cid=cid_for_bytes(raw),
            source_bytes=len(raw), executable=sources[name]["executable"],
            role="program" if name in program else "support", required=name in required)
    receipt = terminal_semantic_scope_observation(canonical, prepared["manifest"], selected_sources=selected)
    _need(_bytes(prepared) == frozen, "signed semantic scope preparation changed during replay")
    after, _, _ = local._manifest(prepared["manifest"], initial=True)
    _need(_bytes(after) == _bytes(manifest) and _bytes(prepared) == frozen,
          "signed semantic source inventory changed during scope replay")
    return receipt


def validate_terminal_semantic_payload(prepared, semantic, *, required_raw_paths=None):
    """Join a current worker payload to the independently signed exact scope."""
    receipt = replay_terminal_semantic_scope(prepared)
    if receipt is None:
        return None
    _need(type(semantic) is dict and semantic.get("schema") == "supervisor-semantic-worker-context@2"
          and type(semantic.get("manifest")) is dict
          and set(semantic["manifest"]) == set(receipt["selected_paths"])
          and semantic.get("required_raw_paths") == receipt["required_raw_paths"]
          and semantic.get("program_paths") == receipt["program_paths"]
          and semantic.get("completion_authority") is False,
          "semantic payload differs from exact signed selection or required roles")
    if required_raw_paths is not None:
        _need(type(required_raw_paths) in {list, tuple}
              and all(type(name) is str for name in required_raw_paths)
              and len(set(required_raw_paths)) == len(required_raw_paths)
              and sorted(required_raw_paths) == receipt["required_raw_paths"],
              "semantic required raw selection differs from signed scope")
    for name, row in receipt["selected_sources"].items():
        _need(semantic["manifest"][name] == {"sha256": row["sha256"], "source_cid": row["source_cid"]},
              "semantic captured source identity differs from signed originals")
    raw_sources = semantic.get("raw_sources")
    _need(type(raw_sources) is dict and set(raw_sources) <= set(receipt["selected_paths"])
          and set(receipt["required_raw_paths"]) <= set(raw_sources),
          "semantic payload omits signed required originals or adds an unknown source")
    for name, text in raw_sources.items():
        row = receipt["selected_sources"][name]
        _need(type(text) is str and len(text) <= row["source_bytes"],
              "semantic raw source must be bounded exact UTF-8 text")
        raw = text.encode("utf-8")
        _need(len(raw) == row["source_bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
              "semantic raw source bytes differ from signed originals")
    return receipt
