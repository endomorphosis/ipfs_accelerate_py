"""Pure successor metadata and public signature replay controls.

The historical heads, typed source delta, direct-child selection and completed
scan are authored fixture data. These tests do not qualify native freshness,
open native owners, fit a model, execute inference or run repository code.
"""
from copy import deepcopy
import hashlib
import sys
from types import SimpleNamespace

import pytest

from ipfs_datasets_py.duckdb_control import codebase_catalog as catalog
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as selection
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import SnapshotEntry
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_worker_context as inventory
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_context as worker


@pytest.fixture(autouse=True)
def no_native_or_numerical_work(monkeypatch):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore

    def forbidden(*_, **__):
        pytest.fail("pure metadata replay attempted native owner or numerical work")

    for owner, name in ((AutoencoderRegistry, "__init__"), (catalog.CodebaseCatalog, "__init__"),
            (ImmutableCAS, "__init__"), (DuckDBASTStore, "__init__"), (resume, "_worker"),
            (resume.features, "train_projection_features"), (resume.features, "infer_projection_features"),
            (resume.training, "train_current_codebase_features"), (resume.training, "_worker")):
        monkeypatch.setattr(owner, name, forbidden)


def _false():
    return {name: False for name in inventory.AUTHORITY_NAMES}


def _implementation():
    files = {"authored-metadata-test-producer": "d" * 64}
    return {"files": files, "sha256": resume.features.digest(files),
        "scope": "listed_local_files_only_not_execution_attestation"}


def _publication(number, previous=None, *, label="authored-successor"):
    repository_id = "repository:" + label
    manifest = cid_for_structured({"authored-manifest": label, "generation": number})
    snapshot = cid_for_structured({"authored-snapshot": label, "generation": number})
    receipt = catalog.CodebasePublicationReceipt(operation_id=f"authored-publication-{number}",
        request_cid=cid_for_structured({"schema": catalog.REQUEST_SCHEMA,
            "repository_id": repository_id, "manifest_cid": manifest,
            "expected_head": previous.to_dict() if previous is not None else None}),
        previous_head=previous, repository_id=repository_id, generation=number,
        manifest_cid=manifest, snapshot_cid=snapshot,
        ast_revision_id=f"rev:{repository_id}:snapshot:{snapshot}")
    return receipt


def _model(version, raw, *, parent=None):
    artifact = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    ancestry = [{"version_id": version, "artifact": artifact}]
    if parent is not None:
        ancestry.extend(deepcopy(parent["ancestry"]))
    return {"version_id": version, "variant_id": "variant:authored-successor",
        "artifact": artifact, "artifact_cid": cid_for_bytes(raw), "contract_sha256": "a" * 64,
        "state_sha256": hashlib.sha256(version.encode()).hexdigest(), "feature_space_sha256": "c" * 64,
        "latent_width": 8, "feature_columns": 1, "projection_ids": ["projection:authored"],
        "projection_widths": {"projection:authored": 1}, "ancestry": ancestry}


def _side(path, raw):
    entry = SnapshotEntry(path=path, kind="artifact", size_bytes=len(raw),
        source_cid=cid_for_bytes(raw), acquisition="captured", disposition="filesystem")
    member = {"source_key": entry.source_key, "path": entry.path, "raw_path_hex": entry.raw_path_hex,
        "entry_cid": entry.entry_cid, "source_cid": entry.source_cid, "ast_cid": None,
        "parse_status": "unindexed", "source_size_bytes": len(raw), "opaque_reason": None}
    return {"entry": entry.to_dict(), "member": member}


def _delta_envelope(value):
    record = delta.CodebaseSourceDeltaRecord.from_dict(cid_for_structured(value), value)
    return {"artifact_cid": record.artifact_cid, "value": record.to_dict()}


def _selection_envelope(value):
    record = selection.CodebaseSuccessorScanRecord.from_dict(cid_for_structured(value), value)
    return {"artifact_cid": record.artifact_cid, "value": record.to_dict()}


def _inventory_context(root_body):
    root = resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(root_body), root_body)
    members = root_body["members"]
    pages = []
    if members:
        pages.append({"page_cid": cid_for_bytes(b"authored-completed-unindexed-page"),
            "start": 0, "end": len(members), "membership_cid": root_body["membership_cid"],
            "inferred_rows": 0, "dispositions": {"unindexed": len(members)}})
    complete_body = {"schema": resume.COMPLETION_SCHEMA, "root_cid": root.artifact_cid,
        "head_cid": root_body["head_cid"], "membership_cid": root_body["membership_cid"],
        "model_artifact_cid": root_body["model"]["artifact_cid"], "pages": pages,
        "coverage": {"inventory_entries": len(members), "inferred_rows": 0, "pages": len(pages),
            "dispositions": {"unindexed": len(members)} if members else {}}, "authority": _false()}
    complete = resume.CodebaseScanResumeCompletion.from_dict(cid_for_structured(complete_body), complete_body)
    return {"schema": inventory.SCHEMA, "scan": complete.advisory_refs(root),
        "evidence": None, "authority": _false()}


def authored_successor_context(*, names=("input.py", "keep.py"), optimized=True):
    """Genuine typed authored history, with no CAS or model-owner observations."""
    previous_receipt = _publication(1)
    current_receipt = _publication(2, previous_receipt.head)
    ledger = []
    for path in names:
        old = _side(path, b"VALUE = 1\n")
        current = _side(path, b"VALUE = 2\n" if path == "input.py" else b"VALUE = 1\n")
        ledger.append({"source_key": old["member"]["source_key"],
            "classification": "changed" if path == "input.py" else "retained",
            "previous": old, "current": current,
            "source_bytes_comparison": "different" if path == "input.py" else "equal",
            "ast_identity_comparison": "unavailable"})
    delta_body = {"schema": delta.SCHEMA, "previous_head": previous_receipt.head.to_dict(),
        "current_head": current_receipt.head.to_dict(),
        "previous_publication_receipt": previous_receipt.to_dict(),
        "current_publication_receipt": current_receipt.to_dict(),
        "previous_membership_cid": cid_for_structured([row["previous"]["member"] for row in ledger]),
        "current_membership_cid": cid_for_structured([row["current"]["member"] for row in ledger]),
        "capture_policy": {"max_entries": 2, "max_file_bytes": 4096, "exclusions": []},
        "ledger": ledger, "coverage": delta._coverage(ledger),
        "limits": delta.CodebaseSourceDeltaLimits(max_inventory_entries=2, max_union_entries=4,
            max_file_bytes=4096).to_dict(), "optimized": optimized, "implementation": _implementation(),
        "authority": _false(), "numerical_reuse": False, "model_advanced": False,
        "removal_scope": "absent_from_current_complete_capture", "physical_absence_verified": False}
    source_delta = _delta_envelope(delta_body)
    parent = _model("version:authored-parent", b"authored-parent-checkpoint")
    child = _model("version:authored-child", b"authored-child-checkpoint", parent=parent)
    limits = resume.CodebaseScanResumeLimits(max_inventory_entries=2, page_entries=2, max_pages=2,
        max_inferred_rows=2).to_dict()
    root_body = {"schema": resume.ROOT_SCHEMA, "head": current_receipt.head.to_dict(),
        "head_cid": cid_for_structured(current_receipt.head.to_dict()),
        "members": [row["current"]["member"] for row in ledger],
        "membership_cid": delta_body["current_membership_cid"], "model": child, "limits": limits,
        "optimized": optimized, "implementation": _implementation(), "authority": _false()}
    current_inventory = _inventory_context(root_body)
    selection_body = {"schema": selection.SCHEMA, "source_delta_cid": source_delta["artifact_cid"],
        "previous_head": delta_body["previous_head"], "current_head": delta_body["current_head"],
        "previous_membership_cid": delta_body["previous_membership_cid"],
        "current_membership_cid": delta_body["current_membership_cid"],
        "previous_training_record_cid": cid_for_structured({"authored-parent-training": 1}),
        "training_record_cid": cid_for_structured({"authored-child-training": 2}),
        "previous_model": parent, "model": child, "root_cid": current_inventory["scan"]["root_cid"],
        "scan_limits": limits, "optimized": optimized, "implementation": _implementation(),
        "authority": _false(), "training_performed_here": False, "inference_performed_here": False,
        "numerical_reuse": False, "model_head_promoted": False}
    return {"schema": worker.SCHEMA, "selection": _selection_envelope(selection_body),
        "source_delta": source_delta, "inventory": current_inventory, "authority": _false()}


@pytest.fixture
def successor_context():
    return authored_successor_context()


@pytest.fixture
def sources(successor_context):
    members = successor_context["inventory"]["scan"]["members"]
    return {member["path"]: {"sha256": hashlib.sha256(
        b"VALUE = 2\n" if member["path"] == "input.py" else b"VALUE = 1\n").hexdigest(),
        "executable": False} for member in members}


@pytest.mark.parametrize("optimized", [True, False])
def test_typed_complete_history_is_detached_and_non_numerical(optimized):
    context = authored_successor_context(optimized=optimized)
    original = deepcopy(context)
    validated = worker.validate_successor_context(context)
    assert validated == original and validated is not context
    assert validated["inventory"]["scan"]["coverage"] == {
        "inventory_entries": 2, "inferred_rows": 0, "pages": 1, "dispositions": {"unindexed": 2}}
    assert all(flag is False for flag in validated["authority"].values())
    assert all(validated["selection"]["value"][name] is False for name in selection._FALSE)
    validated["source_delta"]["value"]["ledger"].clear()
    validated["selection"]["value"]["model"]["state_sha256"] = "f" * 64
    assert context == original


@pytest.mark.parametrize("section", [None, "selection", "source_delta", "inventory", "authority"])
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_public_context_requires_closed_shapes(successor_context, section, mutation):
    value = deepcopy(successor_context)
    target = value if section is None else value[section]
    if mutation == "extra":
        target["foreign"] = False
    else:
        target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("section,field", [("selection", "training_performed_here"),
    ("selection", "inference_performed_here"), ("selection", "numerical_reuse"),
    ("selection", "model_head_promoted"), ("source_delta", "numerical_reuse"),
    ("source_delta", "model_advanced"), ("source_delta", "physical_absence_verified")])
def test_false_metadata_flags_reject_integer_aliases(successor_context, section, field):
    value = deepcopy(successor_context)
    value[section]["value"][field] = 0
    value[section]["artifact_cid"] = cid_for_structured(value[section]["value"])
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("section", [None, "selection", "source_delta", "inventory"])
@pytest.mark.parametrize("flag", [0, True])
def test_no_authority_section_can_acquire_authority(successor_context, section, flag):
    value = deepcopy(successor_context)
    target = value if section is None else value[section]
    if section in {"selection", "source_delta"}:
        target = target["value"]
    target["authority"]["proof_authority"] = flag
    if section in {"selection", "source_delta"}:
        value[section]["artifact_cid"] = cid_for_structured(target)
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("field,value", [("optimized", 1), ("current_head", True),
    ("scan_limits", True)])
def test_boolean_aliases_cannot_replace_profile_or_counts(successor_context, field, value):
    context = deepcopy(successor_context)
    body = context["selection"]["value"]
    if field == "current_head":
        body[field]["generation"] = value
    elif field == "scan_limits":
        body[field]["page_entries"] = value
    else:
        body[field] = value
    context["selection"]["artifact_cid"] = cid_for_structured(body)
    with pytest.raises(ValueError):
        worker.validate_successor_context(context)


@pytest.mark.parametrize("foreign", [0.0, float("nan"), object(), (False,), SimpleNamespace(flag=False)])
def test_non_plain_or_float_context_values_are_rejected(successor_context, foreign):
    value = deepcopy(successor_context)
    value["authority"]["proof_authority"] = foreign
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("section", ["selection", "source_delta"])
def test_retained_record_cid_must_match_exact_canonical_body(successor_context, section):
    value = deepcopy(successor_context)
    value[section]["artifact_cid"] = cid_for_structured({"foreign": section})
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("change", ["previous_head", "current_head", "parent", "child", "root",
    "limits", "optimization"])
def test_rehashed_typed_selection_cannot_cross_inventory_bindings(successor_context, change):
    value = deepcopy(successor_context)
    body = value["selection"]["value"]
    if change in {"previous_head", "current_head"}:
        head = deepcopy(body[change])
        snapshot = cid_for_structured({"foreign-selection-snapshot": change})
        head["snapshot_cid"] = snapshot
        head["ast_revision_id"] = f"rev:{head['repository_id']}:snapshot:{snapshot}"
        head["manifest_cid"] = cid_for_structured({"foreign-selection-manifest": change})
        body[change] = catalog.CodebaseHead.from_dict(head).to_dict()
    elif change == "parent":
        parent = _model("version:foreign-parent", b"foreign-parent-checkpoint")
        body["previous_model"] = parent
        body["model"]["ancestry"][1:] = deepcopy(parent["ancestry"])
    elif change == "child":
        body["model"] = _model("version:foreign-child", b"foreign-child-checkpoint",
            parent=body["previous_model"])
    elif change == "root":
        body["root_cid"] = cid_for_structured({"foreign-root": 1})
    elif change == "limits":
        body["scan_limits"]["page_entries"] = 1
    else:
        body["optimized"] = False
    value["selection"] = _selection_envelope(body)
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("change", ["member", "prefix"])
def test_rehashed_delta_members_must_equal_complete_inventory(successor_context, change):
    value = deepcopy(successor_context)
    body = value["source_delta"]["value"]
    if change == "member":
        member = body["ledger"][0]["current"]["member"]
        member.update(ast_cid=cid_for_structured({"foreign-AST": 1}), parse_status="ok")
    else:
        row = body["ledger"][-1]
        row.update(current=None, classification="removed", source_bytes_comparison="unavailable",
            ast_identity_comparison="unavailable")
    body["current_membership_cid"] = cid_for_structured(
        [row["current"]["member"] for row in body["ledger"] if row["current"] is not None])
    body["coverage"] = delta._coverage(body["ledger"])
    value["source_delta"] = _delta_envelope(body)
    selected = value["selection"]["value"]
    selected.update(source_delta_cid=value["source_delta"]["artifact_cid"],
        current_membership_cid=body["current_membership_cid"])
    value["selection"] = _selection_envelope(selected)
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


def test_rehashed_delta_cannot_replace_immediate_publication(successor_context):
    value = deepcopy(successor_context)
    old = _publication(1, label="foreign-successor")
    new = _publication(2, old.head, label="foreign-successor")
    body = value["source_delta"]["value"]
    body.update(previous_head=old.head.to_dict(), current_head=new.head.to_dict(),
        previous_publication_receipt=old.to_dict(), current_publication_receipt=new.to_dict())
    value["source_delta"] = _delta_envelope(body)
    selected = value["selection"]["value"]
    selected["source_delta_cid"] = value["source_delta"]["artifact_cid"]
    value["selection"] = _selection_envelope(selected)
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


def test_empty_but_self_consistent_completed_history_is_not_dispatch_context():
    context = authored_successor_context(names=())
    assert context["inventory"]["scan"]["coverage"]["inventory_entries"] == 0
    with pytest.raises(ValueError):
        worker.validate_successor_context(context)


def test_partial_typed_completion_cannot_stand_for_full_root(successor_context):
    value = deepcopy(successor_context)
    refs = value["inventory"]["scan"]
    complete = refs["completion_record"]
    complete["pages"][0].update(end=1,
        membership_cid=cid_for_structured(refs["members"][:1]), dispositions={"unindexed": 1})
    complete["coverage"] = {"inventory_entries": 1, "inferred_rows": 0, "pages": 1,
        "dispositions": {"unindexed": 1}}
    typed = resume.CodebaseScanResumeCompletion.from_dict(cid_for_structured(complete), complete)
    refs.update(completion_cid=typed.artifact_cid, pages=deepcopy(complete["pages"]),
        coverage=deepcopy(complete["coverage"]))
    with pytest.raises(ValueError):
        worker.validate_successor_context(value)


@pytest.mark.parametrize("change", ["missing", "extra", "replacement"])
def test_signed_sources_must_cover_all_current_members(successor_context, sources, change):
    current = deepcopy(sources)
    current.pop("keep.py")
    if change == "extra":
        current = {**sources, "extra.py": {"sha256": "e" * 64, "executable": False}}
    elif change == "replacement":
        current["foreign.py"] = {"sha256": "e" * 64, "executable": False}
    with pytest.raises(ValueError):
        worker.validate_successor_context(successor_context, sources=current)


def test_declaration_is_exact_detached_projection(successor_context, sources):
    declaration = worker.successor_declaration(successor_context)
    scan = successor_context["inventory"]["scan"]
    assert declaration["full_context_cid"] == content_identity(successor_context)
    assert declaration["selection_cid"] == successor_context["selection"]["artifact_cid"]
    assert declaration["source_delta_cid"] == successor_context["source_delta"]["artifact_cid"]
    assert declaration["root_cid"] == scan["root_cid"]
    assert declaration["completion_cid"] == scan["completion_cid"]
    assert declaration["inventory_context_cid"] == content_identity(successor_context["inventory"])
    compact_inventory = inventory.inventory_declaration(successor_context["inventory"])
    validated = worker.validate_successor_declaration(declaration,
        inventory_declaration=compact_inventory, sources=sources)
    assert validated == declaration and validated is not declaration
    assert worker.validate_successor_declaration(declaration, inventory_declaration=compact_inventory,
        sources=sources, context=successor_context) == declaration
    validated["model"]["state_sha256"] = "f" * 64
    assert declaration["model"] == scan["model"]


@pytest.mark.parametrize("change", ["schema", "full_context_cid", "selection_cid", "source_delta_cid",
    "previous_head", "current_head", "previous_membership_cid", "current_membership_cid",
    "previous_model", "model", "root_cid", "completion_cid", "inventory_context_cid", "authority",
    "extra", "missing"])
def test_declaration_cannot_substitute_full_context_projection(successor_context, change):
    value = worker.successor_declaration(successor_context)
    if change == "extra":
        value["foreign"] = False
    elif change == "missing":
        value.pop("source_delta_cid")
    elif change == "schema":
        value["schema"] = inventory.DECLARATION_SCHEMA
    elif change == "authority":
        value["authority"]["execution_authority"] = 0
    elif change in {"previous_head", "current_head"}:
        value[change]["manifest_cid"] = cid_for_structured({"foreign-head": change})
    elif change in {"previous_model", "model"}:
        value[change]["state_sha256"] = "f" * 64
    else:
        value[change] = cid_for_structured({"foreign-declaration": change})
    with pytest.raises(ValueError):
        worker.validate_successor_declaration(value,
            inventory_declaration=inventory.inventory_declaration(successor_context["inventory"]),
            context=successor_context)


def test_callback_mutating_original_plain_input_is_refused(successor_context, monkeypatch):
    original = selection._shape
    calls = []

    def changed(body):
        original(body)
        calls.append(1)
        successor_context["authority"]["proof_authority"] = True

    monkeypatch.setattr(selection, "_shape", changed)
    with pytest.raises(ValueError):
        worker.validate_successor_context(successor_context)
    assert calls


def test_callback_mutating_signed_sources_after_member_replay_is_refused(
        successor_context, sources, monkeypatch):
    original = inventory.validate_inventory_context
    calls = []

    def changed(context, **arguments):
        checked = original(context, **arguments)
        calls.append(1)
        sources["input.py"]["sha256"] = "f" * 64
        return checked

    monkeypatch.setattr(inventory, "validate_inventory_context", changed)
    with pytest.raises(ValueError, match="sources changed"):
        worker.validate_successor_context(successor_context, sources=sources)
    assert calls


def test_callback_mutating_inventory_declaration_after_replay_is_refused(
        successor_context, monkeypatch):
    declaration = worker.successor_declaration(successor_context)
    compact_inventory = inventory.inventory_declaration(successor_context["inventory"])
    original = inventory.validate_inventory_declaration
    calls = []

    def changed(context, **arguments):
        checked = original(context, **arguments)
        calls.append(1)
        compact_inventory["authority"]["proof_authority"] = True
        return checked

    monkeypatch.setattr(inventory, "validate_inventory_declaration", changed)
    with pytest.raises(ValueError, match="input changed"):
        worker.validate_successor_declaration(declaration, inventory_declaration=compact_inventory)
    assert calls


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_declaration_signed_sources_must_match_complete_population(successor_context, sources, change):
    declaration = worker.successor_declaration(successor_context)
    if change == "missing":
        sources.pop("keep.py")
    else:
        sources["extra.py"] = {"sha256": "e" * 64, "executable": False}
    with pytest.raises(ValueError):
        worker.validate_successor_declaration(declaration, sources=sources,
            inventory_declaration=inventory.inventory_declaration(successor_context["inventory"]))


def test_mapping_subclass_is_not_plain_metadata(successor_context):
    class ForeignMapping(dict):
        pass

    with pytest.raises(ValueError):
        worker.validate_successor_context(ForeignMapping(successor_context))


@pytest.mark.parametrize("change", ["depth", "bytes", "integer"])
def test_public_metadata_is_bounded_before_native_replay(successor_context, change):
    value = deepcopy(successor_context)
    foreign = False
    if change == "depth":
        for _ in range(49):
            foreign = [foreign]
    elif change == "bytes":
        foreign = "x" * worker.MAX_CONTEXT_BYTES
    else:
        foreign = 1 << 129
    value["authority"]["proof_authority"] = foreign
    with pytest.raises(ValueError, match="bound"):
        worker.validate_successor_context(value)


@pytest.fixture
def signed_successor_case(successor_context, sources, monkeypatch):
    """Real in-memory signatures and typed graph; no lifecycle or owner state."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority as authority
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
        PromptTaskRecord, PromptValidationRecord,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta

    def forbidden(*_, **__):
        pytest.fail("public metadata replay attempted private profile or repository observation")

    for name in ("load_local_profile", "_signed", "_manifest", "_sources", "_manifest_observation_git"):
        monkeypatch.setattr(local, name, forbidden)
    # The product may retain existing native mirror hooks. Its real public
    # replay scope must refuse activation even when storage is configured.
    monkeypatch.setenv(meta.ENV_DUCKDB, "/authored/unopened/sentinel-meta.duckdb")
    activation_calls = []

    def forbidden_activation(*_, **__):
        activation_calls.append(1)
        pytest.fail("public replay activated configured metadata storage")

    monkeypatch.setattr(meta.SupervisorMetaIndex, "from_env", classmethod(forbidden_activation))
    monkeypatch.setattr(meta, "_connect", forbidden_activation)
    native_mirror = meta.mirror_work_record
    mirror_calls = []

    def observed_mirror(**arguments):
        mirror_calls.append(deepcopy(arguments))
        return native_mirror(**arguments)

    monkeypatch.setattr(meta, "mirror_work_record", observed_mirror)
    native_verify = authority.verify_did_key_signature
    signature_calls = []

    def public_verify(**arguments):
        # Observe the product's actual argument; never supply purity on its
        # behalf. The unchanged native verifier receives every argument.
        assert arguments.get("mirror") is False
        signature_calls.append(deepcopy(arguments))
        return native_verify(**arguments)

    monkeypatch.setattr(authority, "verify_did_key_signature", public_verify)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    profile = SimpleNamespace(identity_did=authority.ed25519_did_key(private_key.public_key()),
        profile_id="profile:authored-successor-public-replay")

    def sign(payload):
        return {"payload": deepcopy(payload), "binding": {"identity": profile.identity_did,
            "profile_id": profile.profile_id, "signature": authority._signature(private_key, payload)}}

    policy = content_identity(local.LOCAL_POLICY)
    names = ("input.py", "keep.py")
    checks = tuple(PromptValidationRecord(validation_key=f"check-{position}", argv=("python", name),
        policy_cid=policy) for position, name in enumerate(names))
    criteria = tuple(PromptAcceptanceRecord(criterion_key=f"criterion-{position}",
        criterion=f"The independently authored {name} runtime check must pass",
        validation_keys=(checks[position].validation_key,)) for position, name in enumerate(names))
    goal = PromptGoalRecord(goal_key="SUCCESSOR-ALL-TASKS", parent_goal_cid="", dependency_goal_cids=(),
        title="Retain both declared tasks", objective="Repair the independently declared source files",
        rationale="Authored public metadata control", scope_paths=names, acceptance=criteria)
    tasks = []
    for position, (key, name) in enumerate(zip(("SUCCESSOR-FIRST", "SUCCESSOR-SECOND"), names)):
        tasks.append(PromptTaskRecord(task_key=key, goal_cid=goal.goal_cid,
            dependency_task_cids=() if position == 0 else (tasks[0].task_cid,),
            objective=f"Repair {name} and satisfy its declared runtime check",
            rationale="Independent administrator task", scope_paths=(name,),
            outputs=(PromptOutputRecord(path=name, effect="modify", media_type="text/x-python"),),
            validations=(checks[position],), acceptance=(criteria[position],), evidence_cids=(),
            policy_roots=(policy,), predicted_files=(name,)))
    roots = {name: content_identity({"independent-successor-planning-root": name})
        for name in ("request_cid", "scan_cid", "program_root")}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=tuple(tasks), evidence=())
    specs = [{"task_key": task.task_key, "scope_paths": list(task.scope_paths),
        "outputs": [{name: getattr(item, name) for name in ("path", "effect", "media_type")}
            for item in task.outputs],
        "validations": [{name: local._plain(getattr(item, name)) for name in
            ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
            for item in task.validations],
        "acceptance": [{name: local._plain(getattr(item, name)) for name in
            ("criterion_key", "criterion", "evidence_cids", "validation_keys")}
            for item in task.acceptance],
        "dependencies": [] if position == 0 else [tasks[0].task_key]}
        for position, task in enumerate(tasks)]
    payload = {"schema": local.SUCCESSOR_MANIFEST_SCHEMA, "repository": "/authored/successor/repository",
        "repository_cid": content_identity({"authored-successor-repository": 1}),
        "baseline_commit": "f" * 40, "profile_content_id": content_identity({"authored-public-profile": 1}),
        "profile_dir": "/authored/unopened/profile", "lifecycle_dir": "/authored/unopened/lifecycle",
        "sources": deepcopy(sources), "policy": deepcopy(local.LOCAL_POLICY), "planning_roots": roots,
        "tasks": specs, "created_outputs": [],
        "codebase_inventory_context": inventory.inventory_declaration(successor_context["inventory"]),
        "codebase_successor_context": worker.successor_declaration(successor_context)}
    local._validate_local_manifest_declarations(payload)
    manifest = sign(payload)
    cold_catalog = "ipfs_accelerate_py.agent_supervisor.proof.code_property_catalog" not in sys.modules
    with meta.public_replay_without_metadata():
        receipt_payload = local._plain(local._planning_payload(graph, manifest, payload, profile, sources))
    if cold_catalog:
        assert {"seed_code_properties", "code_property_catalog"} <= {
            call["record_kind"] for call in mirror_calls}
    return {"manifest": manifest, "graph": graph.to_dict(), "receipt": sign(receipt_payload),
        "inventory": deepcopy(successor_context["inventory"]), "successor": deepcopy(successor_context),
        "tasks": tasks, "profile": profile, "sign": sign, "signature_calls": signature_calls,
        "mirror_calls": mirror_calls, "activation_calls": activation_calls}


def _build_signed(case, **overrides):
    arguments = {"manifest_envelope": case["manifest"], "graph": case["graph"],
        "receipt": case["receipt"], "task_cid": case["tasks"][0].task_cid,
        "owner_identity": case["profile"].identity_did, "owner_profile_id": case["profile"].profile_id,
        "inventory_context": case["inventory"], "successor_context": case["successor"]}
    arguments.update(overrides)
    return worker.build_successor_worker_context(**arguments)


@pytest.mark.parametrize("position", [0, 1])
def test_real_public_signatures_preserve_entire_native_graph_and_pending_checks(
        signed_successor_case, position):
    case = signed_successor_case
    original = deepcopy({name: case[name] for name in ("manifest", "graph", "receipt", "inventory", "successor")})
    selected = case["tasks"][position]
    previous_mirror_calls = len(case["mirror_calls"])
    result = _build_signed(case, task_cid=selected.task_cid)
    assert len(case["mirror_calls"]) > previous_mirror_calls
    assert case["activation_calls"] == []
    assert len(case["signature_calls"]) == 2
    assert all(call["mirror"] is False for call in case["signature_calls"])
    assert [call["payload"] for call in case["signature_calls"]] == [
        case["manifest"]["payload"], case["receipt"]["payload"]]
    assert all(call["identity_did"] == case["profile"].identity_did for call in case["signature_calls"])
    assert result["schema"] == worker.WORKER_SCHEMA
    assert result["task_id"] == selected.task_key
    assert result["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])
    assert result["dependency_task_cids"] == list(selected.dependency_task_cids)
    assert result["task_spec"] == case["manifest"]["payload"]["tasks"][position]
    assert result["codebase_successor_context_cid"] == content_identity(case["successor"])
    assert result["codebase_inventory_context_cid"] == content_identity(case["inventory"])
    assert result["codebase_successor"] == worker.successor_declaration(case["successor"])
    assert [member["path"] for member in result["scan"]["selected_task_members"]] == list(selected.scope_paths)
    expected = case["receipt"]["payload"]
    assert result["pending_requirements"] == expected["pending_requirements"]
    assert result["pending_cid"] == expected["pending_cid"]
    assert result["pending_requirements"]
    assert all(item["phase"] == "post_execution" and item["required"] is True
        for item in result["pending_requirements"])
    assert result["current_facts"] == result["removed_task_cids"] == []
    assert result["runtime_requirements_preserved"] is True
    assert result["native_inventory_current_verified_here"] is False
    assert result["native_persistence_verified_here"] is False
    assert result["publication_authority"] is result["scope_expansion_authority"] is False
    assert all(flag is False for flag in result["authority"].values())
    unsigned_result = {key: value for key, value in result.items() if key != "context_cid"}
    assert result["context_cid"] == content_identity(unsigned_result)
    result["task_spec"]["scope_paths"].clear()
    result["scan"]["selected_task_members"].clear()
    assert {name: case[name] for name in original} == original


@pytest.mark.parametrize("change", ["task_population", "pending", "facts", "runtime_flag",
    "context", "selection", "source_delta"])
def test_resigned_receipt_cannot_change_native_contract(signed_successor_case, change):
    case = signed_successor_case
    value = deepcopy(case["receipt"]["payload"])
    if change == "task_population":
        value["administrator_task_cids"].pop()
    elif change == "pending":
        value["pending_requirements"][0]["required"] = False
    elif change == "facts":
        value["current_facts"] = [{"authored-foreign-fact": True}]
    elif change == "runtime_flag":
        value["runtime_requirements_preserved"] = 1
    else:
        name = {"context": "codebase_successor_context_cid", "selection": "successor_selection_cid",
            "source_delta": "source_delta_cid"}[change]
        value[name] = cid_for_structured({"foreign-receipt": change})
    with pytest.raises(ValueError):
        _build_signed(case, receipt=case["sign"](value))


@pytest.mark.parametrize("change", ["task_population", "acceptance", "scope", "authority", "old_profile"])
def test_resigned_manifest_cannot_change_native_task_or_successor_contract(signed_successor_case, change):
    case = signed_successor_case
    value = deepcopy(case["manifest"]["payload"])
    if change == "task_population":
        value["tasks"].pop()
    elif change == "acceptance":
        value["tasks"][0]["acceptance"][0]["criterion"] = "Forged acceptance"
    elif change == "scope":
        value["tasks"][0]["scope_paths"].append("keep.py")
    elif change == "authority":
        value["codebase_successor_context"]["authority"]["proof_authority"] = 0
    else:
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local

        value["schema"] = local.INVENTORY_MANIFEST_SCHEMA
        value.pop("codebase_successor_context")
    with pytest.raises(ValueError):
        _build_signed(case, manifest_envelope=case["sign"](value))


@pytest.mark.parametrize("section", ["manifest", "receipt"])
def test_worker_replays_actual_signature_bytes(signed_successor_case, section):
    case = signed_successor_case
    envelope = deepcopy(case[section])
    envelope["binding"]["signature"] = "AAAA"
    argument = "manifest_envelope" if section == "manifest" else "receipt"
    with pytest.raises(ValueError):
        _build_signed(case, **{argument: envelope})


@pytest.mark.parametrize("change", ["owner", "profile", "unknown_task"])
def test_public_replay_requires_exact_owner_and_selected_native_task(signed_successor_case, change):
    case = signed_successor_case
    arguments = {"owner_identity": "did:key:foreign"} if change == "owner" else {
        "owner_profile_id": "profile:foreign"} if change == "profile" else {
        "task_cid": cid_for_structured({"foreign-task": 1})}
    with pytest.raises(ValueError):
        _build_signed(case, **arguments)


def test_worker_refuses_plain_graph_input_mutation_after_full_native_replay(
        signed_successor_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local

    case = signed_successor_case
    original = local._planning_payload
    calls = []

    def changed(*arguments, **options):
        result = original(*arguments, **options)
        calls.append(1)
        case["graph"]["tasks"][1]["objective"] = "Changed during detached public replay"
        return result

    monkeypatch.setattr(local, "_planning_payload", changed)
    with pytest.raises(ValueError, match="input changed"):
        _build_signed(case)
    assert calls


@pytest.mark.parametrize("section", ["context", "declaration"])
def test_late_callback_cannot_mutate_detached_private_metadata(successor_context, monkeypatch, section):
    original_context = deepcopy(successor_context)
    declaration = worker.successor_declaration(successor_context)
    compact_inventory = inventory.inventory_declaration(successor_context["inventory"])
    closed = worker._closed
    private = []

    def capture(value, fields, name):
        closed(value, fields, name)
        if name == "successor " + section:
            private.append(value)

    monkeypatch.setattr(worker, "_closed", capture)
    if section == "context":
        replay = inventory.validate_inventory_context

        def changed(value, **arguments):
            checked = replay(value, **arguments)
            # This field is inert historical metadata, but its CID-bound
            # bytes cannot change after the immutable record was replayed.
            private[-1]["selection"]["value"]["training_record_cid"] = cid_for_structured(
                {"foreign-private-training": 1})
            return checked

        monkeypatch.setattr(inventory, "validate_inventory_context", changed)
        with pytest.raises(ValueError, match="changed"):
            worker.validate_successor_context(successor_context)
    else:
        replay = inventory.validate_inventory_declaration

        def changed(value, **arguments):
            checked = replay(value, **arguments)
            private[-1]["selection_cid"] = cid_for_structured({"foreign-private-selection": 1})
            return checked

        monkeypatch.setattr(inventory, "validate_inventory_declaration", changed)
        with pytest.raises(ValueError, match="changed"):
            worker.validate_successor_declaration(declaration, inventory_declaration=compact_inventory)
    assert private
    assert successor_context == original_context


def test_optional_full_context_is_guarded_across_later_declaration_callbacks(successor_context, monkeypatch):
    declaration = worker.successor_declaration(successor_context)
    compact_inventory = inventory.inventory_declaration(successor_context["inventory"])
    original = inventory.inventory_declaration
    calls = []

    def changed(value):
        checked = original(value)
        calls.append(1)
        successor_context["selection"]["value"]["training_record_cid"] = cid_for_structured(
            {"foreign-full-context-after-validation": 1})
        return checked

    monkeypatch.setattr(inventory, "inventory_declaration", changed)
    with pytest.raises(ValueError, match="changed"):
        worker.validate_successor_declaration(declaration, inventory_declaration=compact_inventory,
            context=successor_context)
    assert calls


@pytest.mark.parametrize("section", ["unicode_value", "unicode_key", "astral_value"])
def test_escaped_json_budget_is_checked_before_json_allocation(successor_context, monkeypatch, section):
    value = deepcopy(successor_context)
    if section == "astral_value":
        value["authority"]["proof_authority"] = "\U0001f680" * (worker.MAX_CONTEXT_BYTES // 12 + 1)
    else:
        oversized = "\u0100" * (worker.MAX_CONTEXT_BYTES // 6 + 1)
        if section == "unicode_key":
            value["authority"][oversized] = False
        else:
            value["authority"]["proof_authority"] = oversized

    def forbidden(*_, **__):
        pytest.fail("oversized public input reached JSON allocation")

    monkeypatch.setattr(worker.json, "dumps", forbidden)
    with pytest.raises(ValueError, match="bound"):
        worker.validate_successor_context(value)
