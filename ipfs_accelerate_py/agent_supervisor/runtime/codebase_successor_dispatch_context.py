"""Pure public replay of a source successor and its complete inventory.

These retained records describe a historical direct model child and a complete
scan. Public validation opens no native owner and establishes no currentness,
training, numerical execution, promotion, or proof authority. Signed native
task graphs and pending runtime requirements remain independently binding.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from ..proof.formal_verification_contracts import content_identity
from . import codebase_inventory_evidence_worker_context as inventory

SCHEMA = "supervisor-codebase-successor-context@1"
DECLARATION_SCHEMA = "supervisor-codebase-successor-declaration@1"
WORKER_SCHEMA = "supervisor-codebase-successor-worker-context@1"
MAX_CONTEXT_BYTES = 12 * 1024 * 1024
AUTHORITY_NAMES = inventory.AUTHORITY_NAMES
_CONTEXT_FIELDS = {"schema", "selection", "source_delta", "inventory", "authority"}
_DECLARATION_FIELDS = {"schema", "full_context_cid", "selection_cid", "source_delta_cid",
    "previous_head", "current_head", "previous_membership_cid", "current_membership_cid",
    "previous_model", "model", "root_cid", "completion_cid", "inventory_context_cid", "authority"}
_BASIS_FIELDS = ("variant_id", "contract_sha256", "feature_space_sha256", "latent_width",
                 "feature_columns", "projection_ids", "projection_widths")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _closed(value, fields, name):
    _need(type(value) is dict and set(value) == fields, "closed " + name + " required")


def _wire(value, *, max_bytes=MAX_CONTEXT_BYTES):
    _need(type(max_bytes) is int and max_bytes > 0, "exact successor byte bound required")

    def string_size(item, remaining):
        size = len(item) + 2
        _need(size <= remaining, "successor context byte bound exceeded")
        for character in item:
            code = ord(character)
            if character in {'"', "\\"} or character in "\b\f\n\r\t":
                size += 1
            elif code < 32 or 126 < code <= 65535:
                size += 5
            elif code > 65535:
                size += 11
            _need(size <= remaining, "successor context byte bound exceeded")
        return size

    stack, nodes, size = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _need(nodes <= 1_000_000 and depth <= 48, "bounded successor context required")
        if type(item) is dict:
            _need(all(type(key) is str for key in item), "exact successor context keys required")
            _need(nodes + len(stack) + len(item) <= 1_000_000, "bounded successor context required")
            size += 2 + max(0, len(item) - 1) + len(item)
            for key in item:
                size += string_size(key, max_bytes - size)
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(nodes + len(stack) + len(item) <= 1_000_000, "bounded successor context required")
            size += 2 + max(0, len(item) - 1)
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            _need(item.bit_length() <= 128, "bounded successor integer required")
            size += len(str(item))
        else:
            _need(type(item) in {str, bool, type(None)}, "float-free successor context required")
            size += string_size(item, max_bytes - size) if type(item) is str else (
                4 if item is None or item is True else 5)
        _need(size <= max_bytes, "successor context byte bound exceeded")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode("utf-8")
    _need(len(raw) == size and len(raw) <= max_bytes, "successor context byte bound exceeded")
    return raw


def _guard(value, *, max_bytes=MAX_CONTEXT_BYTES):
    raw = _wire(value, max_bytes=max_bytes)
    return len(raw), hashlib.sha256(raw).digest()


def _same(left, right):
    return _wire(left) == _wire(right)


def _authority(value):
    _closed(value, set(AUTHORITY_NAMES), "successor authority")
    _need(all(flag is False for flag in value.values()), "successor context cannot acquire authority")


def _model_pair(previous, current):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume

    resume._model_shape(previous)
    resume._model_shape(current)
    _need(previous["version_id"] != current["version_id"] and len(current["ancestry"]) >= 2
          and _same(current["ancestry"][1:], previous["ancestry"]), "exact direct successor model child required")
    for field in _BASIS_FIELDS:
        _need(_same(previous[field], current[field]), "successor frozen numerical basis differs: " + field)


def validate_successor_context(value, *, sources=None):
    """Join full typed historical records; never open owners or infer freshness."""
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor import CodebaseSourceDeltaRecord
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_model import CodebaseSuccessorScanRecord

    original, raw = value, _wire(value)
    source_guard = None if sources is None else _guard(sources)
    value = json.loads(raw)
    _closed(value, _CONTEXT_FIELDS, "successor context")
    _need(value["schema"] == SCHEMA, "successor context schema differs")
    _authority(value["authority"])
    for field in ("selection", "source_delta"):
        _closed(value[field], {"artifact_cid", "value"}, "successor " + field + " envelope")
    selection, source = value["selection"], value["source_delta"]
    try:
        selected = CodebaseSuccessorScanRecord.from_dict(selection["artifact_cid"], selection["value"])
        changed = CodebaseSourceDeltaRecord.from_dict(source["artifact_cid"], source["value"])
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise ValueError("invalid typed successor selection or source delta") from error
    _need(type(selected) is CodebaseSuccessorScanRecord and type(changed) is CodebaseSourceDeltaRecord,
          "exact typed successor selection and source delta records required")
    _need(_same(selected.to_dict(), selection["value"]) and _same(changed.to_dict(), source["value"]),
          "typed successor record bytes changed during validation")
    checked_inventory = inventory.validate_inventory_context(value["inventory"], sources=sources)
    _need(_same(checked_inventory, value["inventory"]), "successor inventory changed during validation")
    s, d, scan = selected.to_dict(), changed.to_dict(), checked_inventory["scan"]
    _need(len(scan["members"]) > 0, "successor dispatch requires a nonempty complete inventory")
    _need(s["source_delta_cid"] == changed.artifact_cid,
          "successor selection belongs to a different source delta")
    for field in ("previous_head", "current_head", "previous_membership_cid", "current_membership_cid"):
        _need(_same(s[field], d[field]), "successor source delta binding differs: " + field)
    root = scan["root_record"]
    _need(_same(s["current_head"], scan["head"]) and s["current_membership_cid"] == scan["membership_cid"]
          and _same(s["model"], scan["model"]) and s["root_cid"] == scan["root_cid"]
          and _same(s["scan_limits"], scan["limits"]) and s["optimized"] is root["optimized"],
          "successor complete inventory head/model/root/profile differs")
    members = [row["current"]["member"] for row in d["ledger"] if row["current"] is not None]
    _need(_same(members, scan["members"]), "successor delta current membership differs from complete scan")
    _need(_wire(original) == raw and _wire(value) == raw,
          "successor context input changed during validation")
    if source_guard is not None:
        _need(_guard(sources) == source_guard, "successor signed sources changed during validation")
    return value


def _declaration(context):
    selection = context["selection"]["value"]
    scan = context["inventory"]["scan"]
    return {"schema": DECLARATION_SCHEMA, "full_context_cid": content_identity(context),
        "selection_cid": context["selection"]["artifact_cid"],
        "source_delta_cid": context["source_delta"]["artifact_cid"],
        **{field: selection[field] for field in ("previous_head", "current_head", "previous_membership_cid",
            "current_membership_cid", "previous_model", "model", "root_cid")},
        "completion_cid": scan["completion_cid"],
        "inventory_context_cid": content_identity(context["inventory"]), "authority": dict(context["authority"])}


def successor_declaration(context):
    """Bind all full context bytes in a compact signed historical declaration."""
    return _declaration(validate_successor_context(context))


def validate_successor_declaration(value, *, inventory_declaration, sources=None, context=None):
    """Join a strict signed declaration to the completed inventory declaration."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume

    original, raw = value, _wire(value)
    inventory_guard = _guard(inventory_declaration)
    source_guard = None if sources is None else _guard(sources)
    context_guard = None if context is None else _guard(context)
    value = json.loads(raw)
    _closed(value, _DECLARATION_FIELDS, "successor declaration")
    _need(value["schema"] == DECLARATION_SCHEMA, "successor declaration schema differs")
    _authority(value["authority"])
    for field in ("full_context_cid", "selection_cid", "source_delta_cid", "previous_membership_cid",
                  "current_membership_cid", "root_cid", "completion_cid", "inventory_context_cid"):
        resume._cid(value[field])
    previous, current = resume._head(value["previous_head"]), resume._head(value["current_head"])
    _need(previous.repository_id == current.repository_id and current.generation == previous.generation + 1,
          "immediate successor source heads required")
    _model_pair(value["previous_model"], value["model"])
    checked_inventory = inventory.validate_inventory_declaration(inventory_declaration, sources=sources)
    scan = checked_inventory["scan"]
    _need(scan["coverage"]["inventory_entries"] > 0, "successor dispatch requires a nonempty complete inventory")
    _need(value["inventory_context_cid"] == checked_inventory["full_context_cid"]
          and _same(value["current_head"], scan["head"])
          and value["current_membership_cid"] == scan["membership_cid"]
          and _same(value["model"], scan["model"]) and value["root_cid"] == scan["root_cid"]
          and value["completion_cid"] == scan["completion_cid"],
          "successor signed inventory head/model/root/completion differs")
    if context is not None:
        checked = validate_successor_context(context, sources=sources)
        _need(_same(_declaration(checked), value), "full successor context differs from its signed declaration")
        _need(_same(inventory.inventory_declaration(checked["inventory"]), checked_inventory),
              "full successor inventory differs from its signed declaration")
    _need(_wire(original) == raw and _wire(value) == raw
          and _guard(inventory_declaration) == inventory_guard,
          "successor declaration input changed during validation")
    if context_guard is not None:
        _need(_guard(context) == context_guard, "successor full context changed during validation")
    if source_guard is not None:
        _need(_guard(sources) == source_guard, "successor signed sources changed during validation")
    return value


def _verify_signature(envelope, profile):
    from ..control.profile_authority import verify_did_key_signature

    _closed(envelope, {"payload", "binding"}, "successor signed envelope")
    _need(type(envelope["payload"]) is dict, "exact successor signed payload required")
    binding = envelope["binding"]
    _closed(binding, {"identity", "signature", "profile_id"}, "successor signature binding")
    _need(binding["identity"] == profile.identity_did and binding["profile_id"] == profile.profile_id,
          "successor signature belongs to a different owner")
    _need(type(binding["signature"]) is str and 0 < len(binding["signature"]) <= 512,
          "exact bounded successor signature required")
    verify_did_key_signature(identity_did=profile.identity_did, payload=envelope["payload"],
                             signature=binding["signature"], mirror=False)
    return json.loads(_wire(envelope["payload"]))


def _planning_payload(*arguments):
    from . import local_planning_admission as local
    from .supervisor_meta_index import public_replay_without_metadata

    with public_replay_without_metadata():
        return local._planning_payload(*arguments)


def build_successor_worker_context(*, manifest_envelope, graph, receipt, task_cid,
                                   owner_identity, owner_profile_id, inventory_context, successor_context):
    """Replay signatures and every native task; render descriptive successor refs."""
    from . import local_planning_admission as local

    _need(type(task_cid) is str and 0 < len(task_cid.encode("utf-8")) <= 512,
          "exact bounded successor task identity required")
    _need(all(type(value) is str and 0 < len(value.encode("utf-8")) <= 512
              for value in (owner_identity, owner_profile_id)), "exact bounded public owner identifiers required")
    inputs = (manifest_envelope, graph, receipt, inventory_context, successor_context)
    guards = tuple(_guard(value) for value in inputs)
    profile = SimpleNamespace(identity_did=owner_identity, profile_id=owner_profile_id)
    manifest = _verify_signature(manifest_envelope, profile)
    _need(manifest.get("schema") == local.SUCCESSOR_MANIFEST_SCHEMA,
          "successor worker requires the explicit signed successor manifest profile")
    local._validate_local_manifest_declarations(manifest)
    native = local.PromptGoalGraph.from_dict(graph)
    signed = _verify_signature(receipt, profile)
    expected = local._plain(_planning_payload(native, manifest_envelope, manifest, profile, manifest["sources"]))
    _need(expected["schema"] == local.SUCCESSOR_PLANNING_RECEIPT_SCHEMA
          and _wire(signed, max_bytes=local.MAX_PLANNING_RECEIPT_BYTES)
          == _wire(expected, max_bytes=local.MAX_PLANNING_RECEIPT_BYTES),
          "successor worker signed full planning receipt differs")
    tasks = [task for task in native.tasks if task.task_cid == task_cid]
    _need(len(tasks) == 1, "successor worker must select one signed native task")
    task = tasks[0]
    spec = next(spec for spec in manifest["tasks"] if spec["task_key"] == task.task_key)
    full = validate_successor_context(successor_context, sources=manifest["sources"])
    context = inventory.validate_inventory_context(inventory_context, sources=manifest["sources"])
    _need(_same(context, full["inventory"]), "successor worker inventory context differs")
    inventory.validate_inventory_declaration(manifest["codebase_inventory_context"], sources=manifest["sources"], context=context)
    declaration = validate_successor_declaration(manifest["codebase_successor_context"],
        inventory_declaration=manifest["codebase_inventory_context"], sources=manifest["sources"], context=full)
    scan = context["scan"]
    selected = [member for member in scan["members"] if member["path"] in spec["scope_paths"]]
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid",
        "membership_cid", "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = selected
    result = {"schema": WORKER_SCHEMA, "task_cid": task_cid, "task_id": task.task_key,
        "manifest_cid": content_identity(manifest_envelope), "graph_cid": native.content_id,
        "planning_receipt_cid": content_identity(receipt),
        "codebase_inventory_context_cid": content_identity(context),
        "codebase_successor_context_cid": content_identity(full), "codebase_successor": declaration,
        "scan": advisory, "evidence": context["evidence"],
        "administrator_task_cids": sorted(item.task_cid for item in native.tasks),
        "task_spec": spec, "dependency_task_cids": list(task.dependency_task_cids),
        "pending_requirements": expected["pending_requirements"], "pending_cid": expected["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False,
        "authority": dict(full["authority"])}
    result["context_cid"] = content_identity(result)
    _wire(result)
    _need(tuple(_guard(value) for value in inputs) == guards, "successor worker input changed during public replay")
    return result


__all__ = ["SCHEMA", "DECLARATION_SCHEMA", "WORKER_SCHEMA", "MAX_CONTEXT_BYTES", "AUTHORITY_NAMES",
           "validate_successor_context", "successor_declaration", "validate_successor_declaration",
           "build_successor_worker_context"]
