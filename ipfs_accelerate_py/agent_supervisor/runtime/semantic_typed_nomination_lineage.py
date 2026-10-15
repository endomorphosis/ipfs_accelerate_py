"""Verify typed nomination lineage against independent input and native evidence.

The owner retains a bounded immutable sidecar, separate from model input and
compact launch receipts. Historical verification replays retained native facts;
it does not grant current freshness, task coverage or proof authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from . import semantic_router_translation as codec
from .local_prompt_tokenizer import _json_object, _local_path, _local_regular_file, _sha
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_typed_source_context import (
    project_source_bound_typed_links, project_worker_source_bound_typed_links,
    replay_source_bound_typed_links_for_audit,
)

LINEAGE_SCHEMA = "supervisor-semantic-typed-nomination-lineage@1"
BINDING_SCHEMA = "supervisor-semantic-typed-nomination-binding@1"
VERIFICATION_SCHEMA = "supervisor-semantic-typed-nomination-verification@1"
MAX_LINEAGE_BYTES = 2_000_000
BINDING_FIELDS = frozenset({"schema", "derived_nomination_artifact", "derived_nomination_sha256",
                            "lineage_artifact", "lineage_sha256"})


def _encoded(value):
    return codec._json(value).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def owner_typed_graph_root(*, task_id: str, parent_sha256: str) -> str:
    _sha(parent_sha256, "typed owner parent")
    if (type(task_id) is not str or not task_id or task_id != task_id.strip()
            or len(task_id.encode()) > 512 or any(c in task_id for c in ("\x00", "\n", "\r"))):
        raise ValueError("bounded independent typed task required")
    return "owner-typed:" + _digest({"task_id": task_id, "parent_sha256": parent_sha256})


_SCOPED_PROPOSAL_FIELDS = frozenset({
    "graph_scope", "materialized_symbol_ids", "omitted_root_symbol_ids",
    "root_symbol_records", "root_symbol_records_count", "root_symbol_records_sha256",
    "graph_population", "incoming_boundary_relation_rows", "incoming_boundary_relation_count",
    "incoming_boundary_relations_sha256", "graph_sha256", "full_root_graph_materialized",
    "graph_scope_commitment", "graph_scope_commitment_sha256",
})


def _retained_graph_scope(*, source_proposal, projected):
    """Choose replay only from a supported retained source proposal contract.

    The retained header is untrusted input, not verification evidence. This
    bounded structural gate identifies the one deterministic native replay;
    the caller must still compare the complete replayed proposal byte-for-byte.
    Live defaults and printed authority/freshness flags never choose a scope.
    """
    if type(source_proposal) is not dict or type(projected) is not bool:
        raise ValueError("literal retained typed source proposal and parent mode required")
    prefix = ("supervisor-worker-source-bound-typed-context-proposal@" if projected
              else "supervisor-source-bound-typed-context-proposal@")
    schema, adapter = source_proposal.get("schema"), source_proposal.get("adapter_version")
    if type(schema) is not str or type(adapter) is not str:
        raise ValueError("explicit retained typed source schema and adapter required")
    if schema == prefix + "1" and adapter == "native-typed-dependency-proposal@1":
        if _SCOPED_PROPOSAL_FIELDS & set(source_proposal):
            raise ValueError("legacy typed source proposal cannot carry scoped graph fields")
        return "full_root"
    if (schema != prefix + "2" or adapter != "native-typed-dependency-proposal@2"
            or type(source_proposal.get("graph_scope")) is not str
            or source_proposal["graph_scope"] != "task_closure"
            or not _SCOPED_PROPOSAL_FIELDS <= set(source_proposal)):
        raise ValueError("supported exact retained typed graph scope required")
    for field in ("materialized_symbol_ids", "omitted_root_symbol_ids"):
        values = source_proposal[field]
        if (type(values) is not list or len(values) > 1024
                or (field == "materialized_symbol_ids" and not values)
                or any(type(value) is not str or not value or len(value.encode()) > 16_384
                       for value in values)
                or values != sorted(set(values))):
            raise ValueError("bounded distinct ordered retained typed graph population required")
    if set(source_proposal["materialized_symbol_ids"]) & set(source_proposal["omitted_root_symbol_ids"]):
        raise ValueError("retained typed graph populations must remain disjoint")
    for rows_name, count_name, maximum in (
        ("root_symbol_records", "root_symbol_records_count", 1024),
        ("incoming_boundary_relation_rows", "incoming_boundary_relation_count", 8192),
    ):
        rows, count = source_proposal[rows_name], source_proposal[count_name]
        if (type(rows) is not list or any(type(row) is not dict for row in rows)
                or type(count) is not int or not 0 <= count <= maximum or count != len(rows)
                or (rows_name == "root_symbol_records" and not rows)):
            raise ValueError("bounded exact retained typed graph rows and counts required")
    for field in ("root_symbol_records_sha256", "incoming_boundary_relations_sha256",
                  "graph_sha256", "graph_scope_commitment_sha256"):
        if type(source_proposal[field]) is not str:
            raise ValueError("literal retained typed graph commitment required")
        _sha(source_proposal[field], "retained " + field)
    population = source_proposal["graph_population"]
    if type(population) is not dict or set(population) != {
            "symbols", "relations", "referenced_artifact_endpoints"}:
        raise ValueError("closed retained typed graph population commitment required")
    for kind, rows in population.items():
        if type(rows) is not dict or set(rows) != {"full_root", "materialized", "omitted_root"}:
            raise ValueError("closed retained typed graph partition commitment required")
        for descriptor in rows.values():
            if (type(descriptor) is not dict or set(descriptor) != {"count", "sha256"}
                    or type(descriptor["count"]) is not int
                    or not 0 <= descriptor["count"] <= (1024 if kind == "symbols" else 8192)
                    or type(descriptor["sha256"]) is not str):
                raise ValueError("bounded retained typed graph count/hash commitment required")
            _sha(descriptor["sha256"], "retained graph population")
    commitment = source_proposal["graph_scope_commitment"]
    if (type(source_proposal.get("graph")) is not dict
            or type(source_proposal["full_root_graph_materialized"]) is not bool
            or type(commitment) is not dict or set(commitment) != {
                "schema", "root_cid", "graph_root_id", "graph_scope", "materialization_basis",
                "seed_symbol_ids", "materialized_symbol_ids", "omitted_root_symbol_ids", "index_cids",
                "root_symbol_records_count", "root_symbol_records_sha256",
                "full_capsule_identities_sha256", "full_relation_ledger_sha256", "graph_population",
                "incoming_boundary_relation_count", "incoming_boundary_relations_sha256",
                "graph_sha256", "full_root_graph_materialized"}
            or type(commitment.get("schema")) is not str
            or commitment["schema"] != "supervisor-semantic-typed-graph-scope@1"
            or type(commitment.get("graph_scope")) is not str
            or commitment["graph_scope"] != "task_closure"
            or commitment.get("materialization_basis")
            != "all_outgoing_recorded_relations_of_explicit_seed_resolved_symbol_closure"):
        raise ValueError("closed literal retained typed graph scope commitment required")
    # These hashes authenticate the retained object's internal commitments only.
    # Complete deterministic native replay below must establish authenticity.
    for rows_name, digest_name in (
        ("root_symbol_records", "root_symbol_records_sha256"),
        ("incoming_boundary_relation_rows", "incoming_boundary_relations_sha256"),
        ("graph", "graph_sha256"),
        ("graph_scope_commitment", "graph_scope_commitment_sha256"),
    ):
        if _digest(source_proposal[rows_name]) != source_proposal[digest_name]:
            raise ValueError("retained typed graph contents differ from scope commitments")
    return "task_closure"



def _owner_material(*, repository, workspace, artifact, digest, directory, maximum):
    """Read only one owner-controlled digest-addressed immutable object."""
    _sha(digest, "typed owner material")
    if not isinstance(artifact, Path) or not artifact.is_absolute():
        raise ValueError("absolute typed owner material Path required")
    path = _local_path(artifact)
    expected = repository / ".runtime" / directory / (digest + ".json")
    if path != expected or path.is_relative_to(workspace) or path.resolve(strict=True) != path:
        raise ValueError("typed owner material escaped its protected digest namespace")
    for parent in (repository, repository / ".runtime", expected.parent):
        info = parent.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError("owner-controlled typed material directory required")
    with _local_regular_file(path, maximum=maximum) as (fd, before):
        if (before.st_uid != os.geteuid() or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o444):
            raise ValueError("single-link immutable typed owner material required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_mode", "st_nlink")
        if (len(raw) != before.st_size or len(raw) > maximum
                or hashlib.sha256(raw).hexdigest() != digest
                or any(getattr(before, f) != getattr(after, f) for f in fields)):
            raise ValueError("typed owner material changed or differs from independent hash")
    return raw


def _validate_lineage(*, lineage, native_prompt, expected_origin_nomination_sha256,
                      expected_task_id, owner_binding, source_proposal):
    """Reconstruct the transformation; do not accept cached verification flags."""
    from .semantic_decision_owner import build_decision_nomination, MAX_NOMINATION_BYTES
    _sha(expected_origin_nomination_sha256, "independent origin nomination")
    if (type(lineage) is not dict
            or set(lineage) != {"schema", "origin_nomination_text", "typed_nomination"}
            or lineage["schema"] != LINEAGE_SCHEMA):
        raise ValueError("exact typed nomination lineage required")
    origin_text = lineage["origin_nomination_text"]
    if type(origin_text) is not str or not 1 <= len(origin_text.encode()) <= MAX_NOMINATION_BYTES:
        raise ValueError("bounded exact original nomination bytes required")
    origin_raw = origin_text.encode()
    if hashlib.sha256(origin_raw).hexdigest() != expected_origin_nomination_sha256:
        raise ValueError("typed lineage original declaration differs from independent pin")
    origin = _json_object(origin_raw, "typed origin nomination")
    if set(origin) != {"schema", "policy"}:
        raise ValueError("exact original nomination declaration required")
    original = SemanticDecisionPolicy.from_dict(origin["policy"])
    if origin != build_decision_nomination(original) or not original.required_capsule_ids:
        raise ValueError("typed origin nomination revision or explicit seeds differ")
    value = lineage["typed_nomination"]
    if (type(value) is not dict or set(value) != {"schema", "nomination", "launch_policy", "policy", "receipt"}
            or value["schema"] != "supervisor-semantic-typed-owner-nomination@1"):
        raise ValueError("exact retained typed owner result required")
    receipt = value["receipt"]
    if type(receipt) is not dict or receipt.get("schema") != "supervisor-semantic-typed-owner-policy-receipt@1":
        raise ValueError("typed owner transformation receipt required")
    wire, _, _, _, text, artifact = codec._prompt_parts(native_prompt)
    parent = codec._parse(text)
    parent_sha = codec._sha(text)
    if (wire.get("objective_id") != expected_task_id or parent.get("task_id") != expected_task_id
            or parent.get("schema") != "supervisor-semantic-worker-context@1"):
        raise ValueError("typed lineage differs from actual independently reconstructed native input")
    rows = parent.get("capsules")
    if type(rows) is not list or not 1 <= len(rows) <= 1024:
        raise ValueError("bounded actual typed worker inventory required")
    by_symbol, by_capsule = {}, {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("literal native typed capsule required")
        sid, cid = row.get("stable_symbol_id"), row.get("capsule_cid")
        if (type(sid) is not str or type(cid) is not str
                or sid in by_symbol or cid in by_capsule):
            raise ValueError("distinct actual native typed identities required")
        by_symbol[sid], by_capsule[cid] = cid, sid
    if set(by_capsule) != set(original.required_capsule_ids) | set(original.optional_capsule_ids):
        raise ValueError("typed origin does not partition the actual delivered inventory")
    stored_source = receipt.get("source_proposal")
    if (type(stored_source) is not dict or stored_source.get("source_freshness_verified") is not True
            or stored_source.get("source_check_scope") != "opening_and_closing_current_native_verification"):
        raise ValueError("retained opening/closing source proposal required")
    comparable = json.loads(_encoded(stored_source))
    comparable["source_freshness_verified"] = source_proposal["source_freshness_verified"]
    comparable["source_check_scope"] = source_proposal["source_check_scope"]
    if _encoded(comparable) != _encoded(source_proposal):
        raise ValueError("typed lineage graph differs from replayed retained native evidence")
    nominees = source_proposal["nominee_symbol_ids"]
    if (type(nominees) is not list or len(nominees) != len(set(nominees))
            or not set(nominees) <= set(by_symbol)):
        raise ValueError("typed lineage includes missing delivered dependency nominees")
    known = {by_symbol[sid] for sid in nominees}
    added = tuple(cid for cid in original.optional_capsule_ids
                  if cid in known and cid not in original.selected_optional_capsule_ids)
    selected = SemanticDecisionPolicy(
        required_capsule_ids=original.required_capsule_ids, optional_capsule_ids=original.optional_capsule_ids,
        selected_optional_capsule_ids=original.selected_optional_capsule_ids + added,
        task_text_source_path=original.task_text_source_path)
    # Reuse the existing exact @1/@2 material and task-text contract, with no
    # current-source assertion or invented provider envelope.
    from .semantic_context_projection import SemanticProjectionBindings
    from .semantic_decision_packet import build_semantic_decision_packet
    build_semantic_decision_packet(original_text=text, policy=selected,
        bindings=SemanticProjectionBindings(artifact=artifact, expected_sha256=parent_sha,
            task_id=expected_task_id, semantic_root_cid=parent["semantic_root_cid"], scope_cid=parent["scope_cid"],
            repository_id=wire["repository_id"], tree_id=wire["tree_id"]))
    nomination = build_decision_nomination(selected)
    descriptor = {"schema": "supervisor-semantic-decision-launch-policy@" + selected.schema_revision,
                  "task_id": expected_task_id, "parent_artifact": artifact,
                  "parent_sha256": parent_sha, "policy": selected.to_dict()}
    if (value["nomination"] != nomination or value["policy"] != selected.to_dict()
            or value["launch_policy"] != descriptor):
        raise ValueError("typed lineage does not reconstruct the exact derived nomination/launch policy")
    expected = {
        "task_id": expected_task_id, "nomination_artifact": owner_binding["nomination_artifact"],
        "nomination_sha256": expected_origin_nomination_sha256,
        "native_prompt_sha256": codec._sha(native_prompt), "parent_artifact": artifact,
        "parent_sha256": parent_sha, "semantic_root_cid": parent["semantic_root_cid"],
        "graph_root_id": owner_typed_graph_root(task_id=expected_task_id, parent_sha256=parent_sha),
        "original_policy_sha256": _digest(original.to_dict()), "selected_policy_sha256": _digest(selected.to_dict()),
        "added_selected_optional_capsule_ids": list(added),
        "known_nominee_capsule_ids": [by_symbol[sid] for sid in sorted(nominees)],
        "owner_required_partition_preserved": True, "owner_optional_partition_preserved": True,
        "prior_selected_optional_preserved": True, "source_freshness_verified": True,
        "owner_declaration_rechecked": True, "provider_called": False, "descriptor_published": False,
        "projection_aware": "worker_projection" in parent,
        "full_capsule_count": len(source_proposal["full_symbol_ids"]), "delivered_capsule_count": len(rows),
        "source_check_scope": "current_source_wrapper_and_post_builder_native_verification",
        "selection_basis": "recorded_typed_nominees_within_independent_owner_partition",
    }
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("typed lineage source/policy/owner commitments differ")
    for field in ("owner_task_coverage_verified", "effective_task_coverage_verified", "source_substitution_authority",
                  "proof_authority", "execution_authority", "completion_authority", "deployment_ready", "runtime_activation"):
        if receipt.get(field) is not False:
            raise ValueError("typed lineage cannot promote task/proof/execution authority")
    return nomination, descriptor


def load_and_verify_typed_nomination_lineage(
    *, artifact: Path, expected_sha256: str, native_prompt: str, repository: Path,
    workspace: Path, expected_origin_nomination_sha256: str, expected_task_id: str,
    owner_binding: dict, require_current_source: bool = False,
) -> dict:
    """Return redacted commitments only after actual file and native graph replay."""
    if type(require_current_source) is not bool:
        raise ValueError("literal current-source verification choice required")
    if (not isinstance(repository, Path) or not repository.is_absolute()
            or repository.resolve(strict=True) != repository or not repository.is_dir()):
        raise ValueError("canonical existing typed source repository required")
    if (not isinstance(workspace, Path) or not workspace.is_absolute()
            or _local_path(workspace) != workspace or workspace.resolve(strict=False) != workspace):
        raise ValueError("canonical typed workspace identity required")
    if (repository == workspace or repository.is_relative_to(workspace)
            or (require_current_source and not workspace.is_dir())):
        raise ValueError("independent allocated typed workspace required")
    if (type(native_prompt) is not str or not 1 <= len(native_prompt.encode()) <= MAX_MODEL_PROMPT_BYTES
            or type(owner_binding) is not dict or len(_encoded(owner_binding)) > 65_536):
        raise ValueError("bounded actual native input and compact owner receipt required")
    commit = owner_binding.get("typed_nomination")
    if type(commit) is not dict or set(commit) != BINDING_FIELDS or commit.get("schema") != BINDING_SCHEMA:
        raise ValueError("exact compact typed nomination binding required")
    if (owner_binding.get("schema") != "supervisor-semantic-decision-owner-binding@1"
            or owner_binding.get("provider_called") is not False
            or owner_binding.get("completion_authority") is not False):
        raise ValueError("typed material cannot claim provider dispatch or completion")
    for name in ("derived_nomination_artifact", "lineage_artifact"):
        value = commit[name]
        if (type(value) is not str or not 1 <= len(value.encode()) <= 4096
                or any(c in value for c in ("\x00", "\n", "\r")) or not _local_path(value).is_absolute()):
            raise ValueError("bounded canonical typed material path required")
    if (commit["lineage_artifact"] != str(artifact) or commit["lineage_sha256"] != expected_sha256
            or owner_binding.get("task_id") != expected_task_id
            or owner_binding.get("nomination_sha256") != expected_origin_nomination_sha256
            or owner_binding.get("native_prompt_sha256") != codec._sha(native_prompt)
            or type(owner_binding.get("native_prompt_bytes")) is not int
            or owner_binding["native_prompt_bytes"] != len(native_prompt.encode())):
        raise ValueError("typed lineage differs from independent origin/task/native input")
    raw = _owner_material(repository=repository, workspace=workspace, artifact=artifact,
        digest=expected_sha256, directory="semantic-typed-nominations", maximum=MAX_LINEAGE_BYTES)
    lineage = _json_object(raw, "typed nomination lineage")
    if type(lineage.get("typed_nomination")) is not dict or type(lineage.get("origin_nomination_text")) is not str:
        raise ValueError("retained typed result and exact origin bytes required")
    receipt = lineage["typed_nomination"].get("receipt")
    if type(receipt) is not dict:
        raise ValueError("typed lineage receipt required")
    original = _json_object(lineage["origin_nomination_text"].encode(), "typed origin")
    if (len(lineage["origin_nomination_text"].encode()) > 256_000
            or hashlib.sha256(lineage["origin_nomination_text"].encode()).hexdigest() != expected_origin_nomination_sha256
            or set(original) != {"schema", "policy"}):
        raise ValueError("exact independently pinned origin required before native replay")
    policy = SemanticDecisionPolicy.from_dict(original.get("policy"))
    wire, _, _, _, text, parent_artifact = codec._prompt_parts(native_prompt)
    parent = codec._parse(text)
    mapping = {row["capsule_cid"]: row["stable_symbol_id"] for row in parent["capsules"]}
    seeds = tuple(mapping[cid] for cid in policy.required_capsule_ids)
    graph_root = owner_typed_graph_root(task_id=expected_task_id, parent_sha256=codec._sha(text))
    projected = "worker_projection" in parent
    graph_scope = _retained_graph_scope(
        source_proposal=receipt.get("source_proposal"), projected=projected)
    replay = (project_worker_source_bound_typed_links if projected else project_source_bound_typed_links
              ) if require_current_source else replay_source_bound_typed_links_for_audit
    source = replay(repository=repository, artifact=parent_artifact, expected_sha256=codec._sha(text),
        task_id=expected_task_id, expected_semantic_root_cid=parent["semantic_root_cid"],
        graph_root_id=graph_root, seed_symbol_ids=seeds, graph_scope=graph_scope)
    nomination, descriptor = _validate_lineage(lineage=lineage, native_prompt=native_prompt,
        expected_origin_nomination_sha256=expected_origin_nomination_sha256,
        expected_task_id=expected_task_id, owner_binding=owner_binding, source_proposal=source)
    derived_raw = _owner_material(repository=repository, workspace=workspace,
        artifact=Path(commit["derived_nomination_artifact"]), digest=commit["derived_nomination_sha256"],
        directory="semantic-decision-nominations", maximum=256_000)
    launch_raw = _owner_material(repository=repository, workspace=workspace,
        artifact=Path(owner_binding["launch_policy_artifact"]), digest=owner_binding["launch_policy_sha256"],
        directory="semantic-decision-policies", maximum=256_000)
    if (derived_raw != _encoded(nomination) or launch_raw != _encoded(descriptor)
            or owner_binding.get("parent_artifact") != parent_artifact
            or owner_binding.get("parent_sha256") != codec._sha(text)
            or owner_binding.get("policy_sha256") != _digest(descriptor["policy"])):
        raise ValueError("actual published typed nomination/launch differs from reconstructed lineage")
    if _owner_material(repository=repository, workspace=workspace, artifact=artifact,
        digest=expected_sha256, directory="semantic-typed-nominations", maximum=MAX_LINEAGE_BYTES) != raw:
        raise ValueError("typed lineage changed during native graph replay")
    return {"schema": VERIFICATION_SCHEMA, "task_id": expected_task_id,
        "origin_nomination_sha256": expected_origin_nomination_sha256,
        "derived_nomination_sha256": hashlib.sha256(derived_raw).hexdigest(), "lineage_sha256": expected_sha256,
        "native_prompt_sha256": codec._sha(native_prompt), "native_prompt_bytes": len(native_prompt.encode()),
        "parent_artifact": parent_artifact, "parent_sha256": codec._sha(text),
        "policy_sha256": _digest(descriptor["policy"]), "launch_policy_sha256": hashlib.sha256(launch_raw).hexdigest(),
        "owner_binding_sha256": _digest(owner_binding), "source_freshness_revalidated": require_current_source,
        "provider_calls": 0, "proof_authority": False, "execution_authority": False, "completion_authority": False}
