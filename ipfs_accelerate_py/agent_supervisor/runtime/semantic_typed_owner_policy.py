"""Select recorded dependency nominees within an independently held policy.

The owner's required/optional partition stays exact. Known optional nominees
are retained upfront; graph membership never creates requiredness, closes an
unknown binding, or grants task/proof authority. This library boundary neither
publishes a descriptor nor starts a worker/provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from . import semantic_router_translation as codec
from .local_prompt_tokenizer import _local_path, _sha
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES
from .semantic_decision_owner import build_decision_nomination, load_decision_nomination
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_decision_worker import build_decision_launch_policy
from .semantic_typed_source_context import (
    SCOPED_SCHEMA,
    SCOPED_WORKER_SCHEMA,
    _proposal_symbol_records,
    project_source_bound_typed_links,
    project_worker_source_bound_typed_links,
)

if TYPE_CHECKING:
    from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding

SCHEMA = "supervisor-semantic-typed-owner-nomination@1"
RETRIEVAL_SCHEMA = "supervisor-semantic-typed-owner-nomination@2"
RECEIPT_SCHEMA = "supervisor-semantic-typed-owner-policy-receipt@1"
RETRIEVAL_RECEIPT_SCHEMA = "supervisor-semantic-typed-owner-policy-receipt@2"
MAX_REPORT_BYTES = 2_000_000
_TASK_GRAPH_ENV = "IPFS_ACCELERATE_TYPED_TASK_GRAPH"


def _digest(value):
    return hashlib.sha256(codec._json(value).encode("utf-8")).hexdigest()


def _bounded_text(value, name, maximum):
    if (type(value) is not str or not value or value != value.strip()
            or any(char in value for char in ("\x00", "\n", "\r"))
            or len(value.encode()) > maximum):
        raise ValueError("bounded explicit " + name + " required")
    return value


def _canonical_directory(value, name):
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("absolute canonical " + name + " Path required")
    path = _local_path(value)
    if path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError("canonical existing " + name + " directory required")
    return path


@dataclass(frozen=True)
class TypedOwnerNomination:
    """Immutable retained JSON; accessors return detached values."""

    result_json: str

    def to_dict(self) -> dict:
        return json.loads(self.result_json)

    @property
    def nomination(self) -> dict:
        return self.to_dict()["nomination"]

    @property
    def launch_policy(self) -> dict:
        return self.to_dict()["launch_policy"]

    @property
    def policy(self) -> SemanticDecisionPolicy:
        return SemanticDecisionPolicy.from_dict(self.to_dict()["policy"])

    @property
    def receipt(self) -> dict:
        return self.to_dict()["receipt"]


def build_typed_owner_decision_nomination(
    *, native_prompt: str, repository: Path, workspace: Path,
    nomination_artifact: Path, nomination_sha256: str, task_id: str,
    expected_parent_sha256: str, expected_semantic_root_cid: str,
    graph_root_id: str, max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = MAX_REPORT_BYTES,
    retrieval_query_binding: SemanticRetrievalQueryBinding | None = None,
) -> TypedOwnerNomination:
    """Retain known optional dependencies without reclassifying the partition.

    The independently pinned owner nomination supplies a complete partition of
    the actual full worker inventory. Every owner-required capsule supplies an
    explicit graph seed. Recorded outgoing resolved-symbol nominees that the
    owner classified optional are added to upfront optional selection, with
    existing required, optional and selected-optional order preserved.

    A projected native worker is joined to its own complete retained CAS graph.
    Its owner policy partitions only the delivered capsule inventory. Every
    known nominee must already be in that inventory; missing bodies refuse,
    even if their raw source is available. No ranking backfill or native parent
    mutation occurs. Version @2 is permitted only for a projected parent, and
    the existing launch builder validates its exact required text references.
    Bounds, unknown identities, source/parent/policy changes and incomplete
    partitions refuse before any successful result. No files are written.
    """
    for name, value, maximum in (
        ("max_selected_symbols", max_selected_symbols, 32),
        ("max_graph_nodes", max_graph_nodes, 2048),
        ("max_graph_edges", max_graph_edges, 8192),
        ("max_report_bytes", max_report_bytes, MAX_REPORT_BYTES),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("typed owner nomination has invalid " + name)
    _bounded_text(task_id, "owner task key", 512)
    if retrieval_query_binding is not None:
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        if (type(retrieval_query_binding) is not SemanticRetrievalQueryBinding
                or retrieval_query_binding.task_id != task_id):
            raise ValueError("independently bound typed retrieval parent and task required")
    _bounded_text(expected_semantic_root_cid, "semantic root CID", 8192)
    _bounded_text(graph_root_id, "graph root identity", 4096)
    _sha(expected_parent_sha256, "typed owner semantic parent")
    _sha(nomination_sha256, "typed owner independent nomination")
    if (type(native_prompt) is not str or not native_prompt.strip()
            or len(native_prompt.encode()) > MAX_MODEL_PROMPT_BYTES):
        raise ValueError("bounded nonempty native compiled prompt required")
    root = _canonical_directory(repository, "semantic repository")
    work = _canonical_directory(workspace, "allocated workspace")
    if root == work or root.is_relative_to(work):
        raise ValueError("semantic source repository must remain outside allocated candidate")
    if not isinstance(nomination_artifact, Path) or not nomination_artifact.is_absolute():
        raise ValueError("absolute independent owner nomination Path required")
    nomination_path = _local_path(nomination_artifact)
    original_policy = load_decision_nomination(
        artifact=nomination_path, expected_sha256=nomination_sha256, workspace=work,
    )
    if type(original_policy) is not SemanticDecisionPolicy:
        raise ValueError("independent typed semantic owner policy required")
    if retrieval_query_binding is not None and original_policy.task_text_source_path is None:
        raise ValueError("typed retrieval nomination requires its nominated required task text source")
    if not original_policy.required_capsule_ids:
        raise ValueError("typed owner nomination requires nonempty explicit required capsule seeds")
    if len(original_policy.required_capsule_ids) > max_selected_symbols:
        raise ValueError("explicit required capsule seeds exceed typed nomination bound")

    wire, _, _, _, text, artifact = codec._prompt_parts(native_prompt)
    if wire.get("objective_id") != task_id:
        raise ValueError("native compiled prompt differs from independent owner task key")
    if codec._sha(text) != expected_parent_sha256:
        raise ValueError("native semantic parent SHA256 differs from independent expected parent")
    parent = codec._parse(text)
    if (type(parent) is not dict
            or parent.get("schema") != "supervisor-semantic-worker-context@1"):
        raise ValueError("typed owner nomination requires a native worker parent")
    projected = "worker_projection" in parent
    if projected:
        projection = parent["worker_projection"]
        if (type(projection) is not dict
                or projection.get("schema") != "supervisor-semantic-worker-projection@1"
                or projection.get("selector") != "lexical-query-symbol-name@1"
                or type(projection.get("query")) is not str or not projection["query"]
                or projection.get("semantic_equivalence_claimed") is not False
                or projection.get("completion_authority") is not False):
            raise ValueError("typed owner worker projection requires its supported native policy")
    elif original_policy.task_text_source_path is not None:
        raise ValueError("typed owner full-parent bridge does not support @2 task-text policy")
    if parent.get("task_id") != task_id or parent.get("semantic_root_cid") != expected_semantic_root_cid:
        raise ValueError("typed owner native parent task or semantic root differs")
    rows = parent.get("capsules")
    if type(rows) is not list or not 1 <= len(rows) <= 1024:
        raise ValueError("complete bounded native capsule inventory required")
    by_capsule, by_symbol = {}, {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("typed native capsule identity row required")
        capsule_id, symbol_id = row.get("capsule_cid"), row.get("stable_symbol_id")
        _bounded_text(capsule_id, "native capsule identity", 16_384)
        _bounded_text(symbol_id, "native symbol identity", 16_384)
        if capsule_id in by_capsule or symbol_id in by_symbol:
            raise ValueError("duplicate native capsule or symbol identity")
        by_capsule[capsule_id] = symbol_id
        by_symbol[symbol_id] = capsule_id
    if set(by_capsule) != (set(original_policy.required_capsule_ids)
                          | set(original_policy.optional_capsule_ids)):
        raise ValueError("owner policy must completely partition the actual native worker inventory")
    seeds = tuple(by_capsule[cid] for cid in original_policy.required_capsule_ids)
    project = (project_worker_source_bound_typed_links if projected
               else project_source_bound_typed_links)
    graph_scope = "full_root" if os.environ.get(_TASK_GRAPH_ENV) == "0" else "task_closure"
    source_proposal = project(
        repository=root, artifact=artifact, expected_sha256=expected_parent_sha256,
        task_id=task_id, expected_semantic_root_cid=expected_semantic_root_cid,
        graph_root_id=graph_root_id, seed_symbol_ids=seeds,
        max_selected_symbols=max_selected_symbols, max_graph_nodes=max_graph_nodes,
        max_graph_edges=max_graph_edges, max_report_bytes=max_report_bytes,
        graph_scope=graph_scope,
    )
    binding = source_proposal.get("source_binding") if type(source_proposal) is dict else None
    if (type(binding) is not dict
            or binding.get("task_id") != task_id or binding.get("artifact") != artifact
            or binding.get("artifact_sha256") != expected_parent_sha256
            or binding.get("semantic_root_cid") != expected_semantic_root_cid
            or source_proposal.get("graph_root_id") != graph_root_id
            or source_proposal.get("source_freshness_verified") is not True
            or source_proposal.get("owner_task_policy_verified") is not False):
        raise ValueError("current typed source proposal binding differs")
    if graph_scope == "task_closure":
        if source_proposal.get("schema") != (SCOPED_WORKER_SCHEMA if projected else SCOPED_SCHEMA):
            raise ValueError("current typed source proposal scoped version differs")
        _proposal_symbol_records(source_proposal, graph_scope=graph_scope)
    full_ids, nominee_ids = source_proposal.get("full_symbol_ids"), source_proposal.get("nominee_symbol_ids")
    if (type(full_ids) is not list or type(nominee_ids) is not list
            or not 1 <= len(full_ids) <= 1024
            or any(type(sid) is not str for sid in full_ids + nominee_ids)
            or len(full_ids) != len(set(full_ids))
            or not 1 <= len(nominee_ids) <= max_selected_symbols
            or len(nominee_ids) != len(set(nominee_ids))
            or not set(seeds) <= set(nominee_ids)
            or not set(nominee_ids) <= set(full_ids)):
        raise ValueError("typed proposal nominee or full native symbol population differs")
    if projected:
        inventory = source_proposal.get("worker_inventory")
        expected_inventory = [{"stable_symbol_id": row["stable_symbol_id"],
                               "capsule_cid": row["capsule_cid"]} for row in rows]
        if (type(inventory) is not list
                or any(type(row) is not dict or set(row) != {"stable_symbol_id", "capsule_cid"}
                       for row in inventory)
                or inventory != expected_inventory):
            raise ValueError("typed projected source worker inventory differs from actual delivered parent")
        graph = source_proposal.get("graph")
        if type(graph) is not dict or type(graph.get("nodes")) is not list:
            raise ValueError("typed projected source requires its native proposal graph")
        if len(graph["nodes"]) > max_graph_nodes:
            raise ValueError("typed projected source graph population exceeds bound")
        graph_symbols, graph_capsules = {}, set()
        for record in _proposal_symbol_records(source_proposal, graph_scope=graph_scope):
            if type(record) is not dict:
                raise ValueError("typed projected source requires typed symbol metadata")
            sid, cid = record["native_symbol_id"], record.get("capsule_cid")
            _bounded_text(sid, "full graph native symbol identity", 16_384)
            _bounded_text(cid, "full graph native capsule identity", 16_384)
            if sid in graph_symbols or cid in graph_capsules:
                raise ValueError("typed projected source graph has duplicate native symbols or capsules")
            graph_symbols[sid] = cid
            graph_capsules.add(cid)
        if sorted(graph_symbols) != sorted(full_ids):
            raise ValueError("typed projected source full graph symbol population differs")
        if any(graph_symbols.get(sid) != cid for sid, cid in by_symbol.items()):
            raise ValueError("typed projected source worker identities differ from complete retained graph")
        missing = sorted(set(nominee_ids) - set(by_symbol))
        if missing:
            # Native IDs are normally short CIDs. Bound the diagnostic even if
            # an authored/injected projection provides the maximum text size.
            details = ", ".join(missing)
            if len(details.encode()) > 4096:
                details = ("count=" + str(len(missing)) + "; sha256=" + _digest(missing))
            raise ValueError("missing known typed dependencies in actual delivered worker inventory: " + details)
    elif sorted(full_ids) != sorted(by_symbol) or not set(nominee_ids) <= set(by_symbol):
        # Preserve the original complete-parent path and its authored fixtures;
        # only the projected path needs the additional full graph join.
        raise ValueError("typed proposal nominee or full native symbol population differs")
    if any(source_proposal.get(name) is not False for name in (
            "effective_task_coverage_verified", "source_substitution_authority",
            "proof_authority", "execution_authority", "completion_authority",
            "deployment_ready", "runtime_activation")):
        raise ValueError("typed source proposal cannot supply owner task or execution authority")
    nominee_capsule_ids = {by_symbol[sid] for sid in nominee_ids}
    existing_selected = set(original_policy.selected_optional_capsule_ids)
    added = tuple(cid for cid in original_policy.optional_capsule_ids
                  if cid in nominee_capsule_ids and cid not in existing_selected)
    policy = SemanticDecisionPolicy(
        required_capsule_ids=original_policy.required_capsule_ids,
        optional_capsule_ids=original_policy.optional_capsule_ids,
        selected_optional_capsule_ids=original_policy.selected_optional_capsule_ids + added,
        task_text_source_path=original_policy.task_text_source_path,
    )
    nomination = build_decision_nomination(policy)
    launch_policy = build_decision_launch_policy(
        native_prompt=native_prompt, policy=policy, repository=root,
        **({"retrieval_query_binding": retrieval_query_binding} if retrieval_query_binding is not None else {}),
    )
    if (launch_policy.get("task_id") != task_id
            or launch_policy.get("parent_artifact") != artifact
            or launch_policy.get("parent_sha256") != expected_parent_sha256
            or launch_policy.get("policy") != policy.to_dict()):
        raise ValueError("typed owner launch policy native binding differs")

    # Both descriptor builders may interleave with external source changes.
    # Repeat the native producer/current source boundary after their work.
    final, _ = codec._verified_semantic(
        root, artifact=artifact, text=text, task_id=task_id, current=True,
    )
    if codec._json(final) != codec._json(parent) or codec._read(root, artifact) != text.encode("utf-8"):
        raise ValueError("typed owner semantic parent changed during policy construction")
    final_owner = load_decision_nomination(
        artifact=nomination_path, expected_sha256=nomination_sha256, workspace=work,
    )
    if final_owner != original_policy:
        raise ValueError("independent owner declaration changed during policy construction")

    if retrieval_query_binding is not None:
        closing = build_decision_launch_policy(native_prompt=native_prompt, policy=policy,
            repository=root, retrieval_query_binding=retrieval_query_binding)
        if closing != launch_policy:
            raise ValueError("typed retrieval parents changed during policy construction")
    receipt = {
        "schema": RETRIEVAL_RECEIPT_SCHEMA if retrieval_query_binding is not None else RECEIPT_SCHEMA, "task_id": task_id,
        "nomination_artifact": str(nomination_path), "nomination_sha256": nomination_sha256,
        "original_policy_sha256": _digest(original_policy.to_dict()),
        "selected_policy_sha256": _digest(policy.to_dict()),
        "native_prompt_sha256": codec._sha(native_prompt),
        "parent_artifact": artifact, "parent_sha256": expected_parent_sha256,
        "semantic_root_cid": expected_semantic_root_cid, "graph_root_id": graph_root_id,
        "source_proposal": source_proposal,
        "projection_aware": projected,
        "full_capsule_count": len(full_ids), "delivered_capsule_count": len(by_capsule),
        "added_selected_optional_capsule_ids": list(added),
        "known_nominee_capsule_ids": [by_symbol[sid] for sid in sorted(nominee_ids)],
        "owner_required_partition_preserved": policy.required_capsule_ids == original_policy.required_capsule_ids,
        "owner_optional_partition_preserved": policy.optional_capsule_ids == original_policy.optional_capsule_ids,
        "prior_selected_optional_preserved": policy.selected_optional_capsule_ids[:len(original_policy.selected_optional_capsule_ids)]
            == original_policy.selected_optional_capsule_ids,
        "source_freshness_verified": True, "owner_declaration_rechecked": True,
        "source_check_scope": "current_source_wrapper_and_post_builder_native_verification",
        "selection_basis": "recorded_typed_nominees_within_independent_owner_partition",
        "owner_task_coverage_verified": False, "effective_task_coverage_verified": False,
        "source_substitution_authority": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
        "deployment_ready": False, "runtime_activation": False,
        "descriptor_published": False, "provider_called": False,
    }
    if retrieval_query_binding is not None:
        receipt["retrieval_query_binding"] = retrieval_query_binding.to_dict()
    value = {"schema": RETRIEVAL_SCHEMA if retrieval_query_binding is not None else SCHEMA,
        "nomination": nomination, "launch_policy": launch_policy,
        "policy": policy.to_dict(), "receipt": receipt}
    result_json = codec._json(value)
    if len(result_json.encode("utf-8")) > max_report_bytes:
        raise ValueError("typed owner complete result exceeds report byte bound")
    return TypedOwnerNomination(result_json)
