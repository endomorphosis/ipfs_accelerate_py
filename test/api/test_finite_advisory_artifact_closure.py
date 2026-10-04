"""Real signature and file-custody controls for advisory launch artifact seals.

The preparation fixture replaces native context/lowering/proposal observation
and mandatory-path derivation with authored records. It retains the actual
profile signer, canonical signed envelope, descriptor reads and file/directory
identity checks. These cases do not train, prove, launch, or qualify a native
preparation. Real callback and worker controls belong to the Docker experiment.
All mutations affect independently owned temporary files.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)


@pytest.fixture(scope="module")
def signature_profiles(tmp_path_factory):
    """Actual Ed25519 keys and anchored profiles; no native admission or fit."""
    root = tmp_path_factory.mktemp("advisory-artifact-signatures")
    patch = pytest.MonkeyPatch()
    patch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", root / "account")
    result = []
    try:
        for label in ("owner", "foreign"):
            profile_dir = root / (label + "-profile")
            lifecycle_dir = root / (label + "-lifecycle")
            profile = profile_authority.initialize_local_profile(
                repository_cid=cid_for_structured({"authored_signature_fixture": label}),
                baseline_commit="a" * 40,
                profile_dir=profile_dir,
                lifecycle_dir=lifecycle_dir,
            )
            result.append({"profile": profile,
                           "manifest": {"profile_dir": str(profile_dir),
                                        "lifecycle_dir": str(lifecycle_dir)}})
        yield result
    finally:
        patch.undo()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return path


def _cid(label):
    return cid_for_structured({"authored_advisory_artifact_test": label})


def _context(head, mode):
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as feature
    artifacts = {}
    for role in ("invocation", "record", "checkpoint", "lineage", "inference", "metrics"):
        raw = canonical_dag_json_bytes({"authored_feature_blob": role})
        blob = {"schema": "supervisor-codebase-feature-blob@1", "role": role,
                "relative_path": role + ".json", "sha256": _sha(raw), "size_bytes": len(raw)}
        blob["blob_cid"] = cid_for_structured(blob)
        artifacts[role] = blob
    value = {"schema": feature.SCHEMA, "profile": feature.PROFILE, "mode": mode,
        "head": deepcopy(head), "structural_context_cid": _cid("structural-context"),
        "semantic_state_cid": _cid("semantic-state"), "model_enabled": True,
        "version_id": "sha256:" + "1" * 64, "parent_version_id": None,
        "model_record_cid": _cid("model-record"),
        "candidate_checkpoint_raw_cid": _cid("checkpoint"),
        "feature_space_sha256": "2" * 64, "contract_sha256": "3" * 64,
        "state_sha256": "4" * 64, "latent_width": 8, "parameter_dtype": "float64",
        "actual_training_delta": 16 if mode == "train" else 0,
        "selected_total_epochs": 5, "selected_optimizer_steps": [5, 5, 5, 5],
        "selection_version_cid": _cid("selection"),
        "metrics_scope": "fixed tuning and repeated post-selection canary; structural reconstruction only",
        "representation": "native_compiler_structural_features_not_semantic_text_embeddings",
        "evaluation_scope": "source-cohort transductive diagnostics; no blind structural generalization",
        "receipt_scope": "retained selected process identities and fresh numerical replay; no authenticated process-origin or transitive environment attestation",
        "artifacts": artifacts, "authority": deepcopy(feature._FALSE)}
    value["context_cid"] = cid_for_structured(value)
    return value


@pytest.fixture
def closure_case(tmp_path, monkeypatch, signature_profiles):
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_advisory_artifact_closure as closure
    from benchmarks.agent_supervisor.container_coding import finite_repository_advisory_worker_support as support
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead

    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    repository = root / "repository"
    repository.mkdir(mode=0o700)
    tree = root / "selected-artifacts"
    source = b"def increment(n: int) -> int:\n    return n + 1\n"
    replacement = b"def increment(n: int) -> int:\n    return n + 2\n"
    paths = [_write(tree / name, raw) for name, raw in
             (("checkpoint.json", b'{"authored_checkpoint":1}'),
              ("model.olean", b"authored-not-a-native-proof"),
              ("context.json", b'{"authored_context":1}'))]
    candidate_path = _write(root / "candidate.json", b'{"authored_candidate":1}', 0o444)
    head = CodebaseHead(repository_id="fixture:advisory-closure",
        generation=1, manifest_cid=_cid("manifest"), snapshot_cid=_cid("snapshot"),
        ast_revision_id="rev:fixture:advisory-closure:snapshot:" + _cid("snapshot"),
        receipt_cid=_cid("head-receipt")).to_dict()
    tasks = sorted((_cid("type-task"), _cid("offset-task")))
    candidate = {"artifact": str(candidate_path), "sha256": _sha(candidate_path.read_bytes()),
        "candidate_cid": _cid("candidate"), "finite_admission_cid": _cid("admission"),
        "semantic_context_cid": _cid("semantic-context"), "task_cid": _cid("offset-task"),
        "task_id": "OFFSET-TASK", "task_revision": 2,
        "before_sha256": _sha(source), "after_sha256": _sha(replacement)}
    trained, frozen = _context(head, "train"), _context(head, "frozen")
    bridge = {"schema": support.BRIDGE_SCHEMA, "scope": support.BRIDGE_SCOPE,
        "admission_cid": candidate["finite_admission_cid"],
        "reviewed_candidate_cid": _cid("reviewed"), "generated_result_cid": _cid("generated"),
        "worker_candidate_cid": candidate["candidate_cid"], "head": deepcopy(head),
        "source_cid": cid_for_bytes(source), "replacement_cid": cid_for_bytes(replacement),
        "replacement_sha256": _sha(replacement), "task_cid": candidate["task_cid"],
        "task_revision": candidate["task_revision"], "administrator_task_cids": tasks,
        "model_binding": deepcopy(trained), "model_context_cid": trained["context_cid"],
        "model_version_id": trained["version_id"], "training_steps": 0, "provider_calls": 0,
        **deepcopy(support.CLAIMS)}
    bridge["bridge_cid"] = cid_for_structured(bridge)
    payload = {"head": head, "finite_admission_cid": candidate["finite_admission_cid"],
        "semantic_context_cid": candidate["semantic_context_cid"], "candidate": candidate,
        "administrator_task_cids": tasks,
        "source": {"source_cid": cid_for_bytes(source), "git_commit": "a" * 40,
                   "manifest_cid": head["manifest_cid"], "ast_revision_id": head["ast_revision_id"],
                   "semantic_state_cid": trained["semantic_state_cid"], "custody_cid": _cid("custody")},
        "model": {"training_context": trained, "frozen_context": frozen},
        "lowering": {"result_cid": _cid("lowering-result"), "translation_cid": _cid("translation"),
                     "contract_cid": _cid("lowering-contract"), "tool_policy_cid": _cid("tool-policy"),
                     "process_policy_cid": _cid("process-policy"),
                     "lean_certificate_cid": _cid("lean-certificate"),
                     "lean_olean_cid": cid_for_bytes(paths[1].read_bytes()),
                     "source_cid": cid_for_bytes(source), "status": "model_refuted",
                     "scope": "formal_ast_lowering_correctness"},
        "proposal": {"reviewed_candidate_cid": bridge["reviewed_candidate_cid"],
                     "generated_result_cid": bridge["generated_result_cid"],
                     "worker_candidate_cid": bridge["worker_candidate_cid"],
                     "replacement_cid": bridge["replacement_cid"],
                     "replacement_sha256": bridge["replacement_sha256"], "bridge": bridge}}
    prepared = {"payload": payload, "manifest": deepcopy(signature_profiles[0]["manifest"])}
    calls = []
    def validate(**kwargs):
        calls.append("authored_native_preparation")
        return deepcopy(prepared)
    def mandatory(**kwargs):
        calls.append("authored_mandatory_paths")
        return {"paths": [*paths, candidate_path], "trees": [tree]}
    monkeypatch.setattr(closure, "_validate_preparation", validate)
    monkeypatch.setattr(closure, "_mandatory_paths", mandatory)
    arguments = {"owner": SimpleNamespace(repository=repository, expected_head=head), "registry": object(),
        "admission": {}, "candidate": deepcopy(candidate), "training_context": trained,
        "frozen_context": frozen, "lowering_proof": {}, "contract": {}, "tool_policy": {},
        "reviewed_candidate": {}, "generated_candidate": {}, "bridge": deepcopy(bridge),
        "artifact_paths": [*paths, candidate_path], "output": root / "closure"}
    return {"module": closure, "arguments": arguments, "prepared": prepared, "calls": calls,
            "root": root, "tree": tree, "paths": paths, "candidate_path": candidate_path,
            "signature_profiles": signature_profiles,
            "expected": {key: deepcopy(payload[key]) for key in
                         ("head", "finite_admission_cid", "semantic_context_cid", "candidate",
                          "administrator_task_cids")}}


def _prepare(case, **changes):
    return case["module"].prepare_finite_advisory_artifact_closure(**{**case["arguments"], **changes})


def _require(case, value, **changes):
    return value.require_detached(**{**deepcopy(case["expected"]), **changes})


def _resign_reference(case, reference, payload=None, *, foreign=False):
    """Retain actual re-signed bytes so checksum-only rejection cannot suffice."""
    value = deepcopy(reference)
    payload = deepcopy(value["signed_closure"]["payload"] if payload is None else payload)
    selected = case["signature_profiles"][1 if foreign else 0]
    envelope = local._signed(payload, selected["manifest"])
    assert local._verify_signature(envelope, selected["profile"]) == payload
    raw = canonical_dag_json_bytes(envelope)
    path = _write(case["root"] / "independently-resigned.json", raw)
    pin, observed = case["module"]._read(path)
    assert observed == raw
    value.update(signed_closure=envelope, closure_cid=cid_for_structured(envelope), artifact=pin)
    return value


def test_actual_signed_reference_and_live_typed_custody_are_separate(closure_case):
    case, module = closure_case, closure_case["module"]
    value = _prepare(case)
    assert type(value) is module.FrozenFiniteAdvisoryArtifactClosure
    assert case["calls"] == ["authored_native_preparation", "authored_mandatory_paths"]
    reference = value.material_binding
    payload = module.verify_finite_advisory_artifact_closure_reference(reference)
    assert payload["head"] == case["expected"]["head"]
    assert local._verify_signature(reference["signed_closure"], case["signature_profiles"][0]["profile"]) == payload
    _require(case, value)
    detached = value.material_binding
    detached["authority"][next(iter(detached["authority"]))] = True
    assert value.material_binding == reference


@pytest.mark.parametrize("kind", [
    "in-place", "same-byte-replacement", "write-restore", "mtime-restore", "mode",
    "hardlink", "symlink", "parent-symlink", "added-file", "removed-file", "fifo",
    "directory-mode", "directory-replacement",
])
def test_detached_witness_refuses_artifact_and_selected_namespace_mutation(closure_case, kind):
    case = closure_case
    value = _prepare(case)
    path = case["paths"][0]
    raw, info = path.read_bytes(), path.stat(follow_symlinks=False)
    if kind == "in-place":
        path.write_bytes(raw + b"\n")
    elif kind == "same-byte-replacement":
        replacement = _write(case["root"] / "new-inode", raw, stat.S_IMODE(info.st_mode))
        os.replace(replacement, path)
        assert path.read_bytes() == raw and path.stat().st_ino != info.st_ino
    elif kind in {"write-restore", "mtime-restore"}:
        path.write_bytes(raw + b"\n")
        path.write_bytes(raw)
        if kind == "mtime-restore":
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            assert path.stat().st_mtime_ns == info.st_mtime_ns
        assert path.read_bytes() == raw and path.stat().st_ctime_ns != info.st_ctime_ns
    elif kind == "mode":
        path.chmod(stat.S_IMODE(info.st_mode) ^ 0o040)
    elif kind == "hardlink":
        os.link(path, case["root"] / "outside-selected-tree-link")
        assert path.stat().st_nlink == 2
    elif kind == "symlink":
        independent = _write(case["root"] / "independent-identical-bytes", raw)
        path.unlink()
        path.symlink_to(independent)
    elif kind == "parent-symlink":
        moved = case["root"] / "moved-selected-artifacts"
        case["tree"].rename(moved)
        case["tree"].symlink_to(moved, target_is_directory=True)
    elif kind == "added-file":
        _write(case["tree"] / "unexpected-extra.json", b"{}")
    elif kind == "removed-file":
        path.unlink()
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path, mode=0o600)
    elif kind == "directory-mode":
        case["tree"].chmod(stat.S_IMODE(case["tree"].stat().st_mode) ^ 0o010)
    else:
        moved = case["root"] / "moved-directory"
        case["tree"].rename(moved)
        case["tree"].mkdir(mode=stat.S_IMODE(moved.stat().st_mode))
        for old in moved.iterdir():
            _write(case["tree"] / old.name, old.read_bytes(), stat.S_IMODE(old.stat().st_mode))
    with pytest.raises((ValueError, OSError)):
        _require(case, value)


@pytest.mark.parametrize("field", [
    "head", "finite_admission_cid", "semantic_context_cid", "candidate",
    "administrator_task_cids",
])
def test_live_closure_requires_exact_selected_native_relation(closure_case, field):
    case, altered = closure_case, deepcopy(closure_case["expected"][field])
    value = _prepare(case)
    if field == "head":
        altered["generation"] += 1
    elif field == "candidate":
        altered["task_revision"] += 1
    elif field == "administrator_task_cids":
        altered = altered[:1]
    else:
        altered = _cid("foreign-" + field)
    with pytest.raises(ValueError):
        _require(case, value, **{field: altered})


@pytest.mark.parametrize("field,value", [
    ("task_revision", True), ("task_revision", 2.0), ("task_revision", 3),
    ("task_cid", _cid("foreign-task")), ("candidate_cid", _cid("foreign-candidate")),
    ("finite_admission_cid", _cid("foreign-admission")),
    ("semantic_context_cid", _cid("foreign-context")),
    ("before_sha256", "0" * 64), ("after_sha256", "0" * 64),
    ("artifact", "/tmp/foreign-authored-path"), ("sha256", "0" * 64),
])
def test_live_closure_cannot_borrow_another_candidate_descriptor(closure_case, field, value):
    case = closure_case
    closure = _prepare(case)
    candidate = deepcopy(case["expected"]["candidate"])
    candidate[field] = value
    with pytest.raises(ValueError):
        _require(case, closure, candidate=candidate)


@pytest.mark.parametrize("field", ["schema", "profile", "closure_cid", "artifact", "authority", "signed_closure"])
def test_reference_outer_identity_is_closed(closure_case, field):
    case = closure_case
    reference = _prepare(case).material_binding
    if field in {"artifact", "signed_closure", "authority"}:
        reference[field] = {}
    else:
        reference[field] = "foreign"
    with pytest.raises(ValueError):
        case["module"].verify_finite_advisory_artifact_closure_reference(reference)


def test_reference_rejects_an_unknown_field(closure_case):
    case = closure_case
    reference = _prepare(case).material_binding
    reference["execution_grant"] = True
    with pytest.raises(ValueError):
        case["module"].verify_finite_advisory_artifact_closure_reference(reference)


@pytest.mark.parametrize("kind", ["signature", "signer", "profile", "payload", "envelope-extra"])
def test_actual_signature_cannot_be_replaced_or_retargeted(closure_case, kind):
    case = closure_case
    reference = _prepare(case).material_binding
    envelope = reference["signed_closure"]
    if kind == "signature":
        signature = envelope["binding"]["signature"]
        envelope["binding"]["signature"] = ("A" if signature[0] != "A" else "B") + signature[1:]
    elif kind == "signer":
        foreign = local._signed(envelope["payload"], case["signature_profiles"][1]["manifest"])
        envelope["binding"] = foreign["binding"]
    elif kind == "profile":
        envelope["binding"]["profile_id"] = "foreign-profile"
    elif kind == "payload":
        envelope["payload"]["semantic_context_cid"] = _cid("forged-signed-context")
    else:
        envelope["signature_override"] = True
    with pytest.raises(ValueError):
        case["module"].verify_finite_advisory_artifact_closure_reference(reference)


@pytest.mark.parametrize("relation", [
    "source-cid", "source-manifest", "source-ast", "source-semantic-state", "lowering-source",
    "proposal-reviewed", "proposal-generated", "proposal-worker", "proposal-replacement",
    "proposal-replacement-sha", "bridge-head", "bridge-source", "bridge-task", "bridge-revision",
    "bridge-population", "bridge-admission", "bridge-model-context", "bridge-model-version",
    "model-head", "model-version", "model-feature-basis", "model-state", "model-record",
    "model-checkpoint", "model-contract", "model-training-delta", "model-frozen-delta",
    "model-train-mode", "model-frozen-mode", "model-width", "model-dtype",
])
def test_prepared_material_requires_matching_source_model_proof_and_proposal(closure_case, relation):
    case, payload = closure_case, closure_case["prepared"]["payload"]
    bridge = payload["proposal"]["bridge"]
    model = payload["model"]
    if relation == "source-cid":
        payload["source"]["source_cid"] = cid_for_bytes(b"different source")
    elif relation == "source-manifest":
        payload["source"]["manifest_cid"] = _cid("different-manifest")
    elif relation == "source-ast":
        payload["source"]["ast_revision_id"] = "foreign-ast"
    elif relation == "source-semantic-state":
        payload["source"]["semantic_state_cid"] = _cid("different-semantic-state")
    elif relation == "lowering-source":
        payload["lowering"]["source_cid"] = cid_for_bytes(b"other proved source")
    elif relation.startswith("proposal-"):
        key = {"proposal-reviewed": "reviewed_candidate_cid", "proposal-generated": "generated_result_cid",
               "proposal-worker": "worker_candidate_cid", "proposal-replacement": "replacement_cid",
               "proposal-replacement-sha": "replacement_sha256"}[relation]
        payload["proposal"][key] = "0" * 64 if key.endswith("sha256") else _cid("different-" + key)
    elif relation.startswith("bridge-"):
        key = {"bridge-head": "head", "bridge-source": "source_cid", "bridge-task": "task_cid",
               "bridge-revision": "task_revision", "bridge-population": "administrator_task_cids",
               "bridge-admission": "admission_cid", "bridge-model-context": "model_context_cid",
               "bridge-model-version": "model_version_id"}[relation]
        if key == "head":
            bridge[key]["generation"] += 1
        elif key == "task_revision":
            bridge[key] += 1
        elif key == "administrator_task_cids":
            bridge[key] = bridge[key][:1]
        else:
            bridge[key] = _cid("different-bridge-" + key)
        bridge["bridge_cid"] = cid_for_structured({k: v for k, v in bridge.items() if k != "bridge_cid"})
    else:
        target = model["frozen_context"]
        key, changed = {
            "model-head": ("head", {**target["head"], "generation": 2}),
            "model-version": ("version_id", "sha256:" + "5" * 64),
            "model-feature-basis": ("feature_space_sha256", "5" * 64),
            "model-state": ("state_sha256", "5" * 64),
            "model-record": ("model_record_cid", _cid("different-model-record")),
            "model-checkpoint": ("candidate_checkpoint_raw_cid", _cid("different-checkpoint")),
            "model-contract": ("contract_sha256", "5" * 64),
            "model-training-delta": ("actual_training_delta", 15),
            "model-frozen-delta": ("actual_training_delta", 1),
            "model-train-mode": ("mode", "frozen"),
            "model-frozen-mode": ("mode", "train"),
            "model-width": ("latent_width", 9),
            "model-dtype": ("parameter_dtype", "float32"),
        }[relation]
        if relation in {"model-training-delta", "model-train-mode"}:
            target = model["training_context"]
        target[key] = changed
        target["context_cid"] = cid_for_structured({k: v for k, v in target.items() if k != "context_cid"})
    with pytest.raises(ValueError):
        _prepare(case)


@pytest.mark.parametrize("field", [
    "proof_authority", "execution_authority", "publication_authority", "completion_authority",
    "omission_authority", "production_activated", "convergence_proved", "formal_decoder_available",
])
@pytest.mark.parametrize("alias", [True, 0])
def test_bridge_boolean_aliases_cannot_gain_signed_artifact_authority(closure_case, field, alias):
    case = closure_case
    bridge = case["prepared"]["payload"]["proposal"]["bridge"]
    bridge[field] = alias
    bridge["bridge_cid"] = cid_for_structured({k: v for k, v in bridge.items() if k != "bridge_cid"})
    with pytest.raises(ValueError):
        _prepare(case)


@pytest.mark.parametrize("context", ["training_context", "frozen_context"])
@pytest.mark.parametrize("field,value", [
    ("model_enabled", 1), ("latent_width", True), ("actual_training_delta", False),
    ("authority", {"execution_authority": False}),
])
def test_model_type_aliases_and_partial_authority_are_refused(closure_case, context, field, value):
    case = closure_case
    binding = case["prepared"]["payload"]["model"][context]
    binding[field] = value
    binding["context_cid"] = cid_for_structured({k: v for k, v in binding.items() if k != "context_cid"})
    with pytest.raises(ValueError):
        _prepare(case)


def _scope_case(case, *, advisory=True):
    """An unlaunched seam object; native reservation/lifecycle are not exercised."""
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution as execution
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    scope = object.__new__(execution.FrozenFiniteRepositoryExecutionScope)
    value = _prepare(case) if advisory else None
    payload = {"schema": execution.ADVISORY_SCHEMA if advisory else execution.SCHEMA,
               "profile": execution.ADVISORY_PROFILE if advisory else execution.PROFILE,
               "finite_admission_cid": case["expected"]["finite_admission_cid"],
               "semantic_context_cid": case["expected"]["semantic_context_cid"],
               "candidate": {"descriptor": deepcopy(case["expected"]["candidate"])}}
    if advisory:
        payload["advisory_closure"] = value.material_binding
    envelope = local._signed(payload, case["signature_profiles"][0]["manifest"])
    scope._material = canonical_dag_json_bytes(envelope)
    scope._advisory_closure = value
    scope._owner = SimpleNamespace(expected_head=CodebaseHead.from_dict(case["expected"]["head"]))
    scope._semantic = {"administrator_task_cids": deepcopy(case["expected"]["administrator_task_cids"])}
    scope._spawned = False
    return execution, scope, value


def test_none_seam_preserves_the_original_signed_execution_profile(closure_case):
    execution, scope, _ = _scope_case(closure_case, advisory=False)
    before = scope.to_dict()
    scope._require_advisory_current()
    assert scope.to_dict() == before
    assert "advisory_closure" not in before["payload"]
    assert before["payload"]["schema"] == execution.SCHEMA
    assert before["payload"]["profile"] == execution.PROFILE
    assert execution._pins() == execution._pins(advisory=False)
    module = closure_case["module"].__name__
    assert module not in execution._pins()
    assert module in execution._pins(advisory=True)


@pytest.mark.parametrize("kind", ["removed", "serialized", "substituted", "subclass", "schema-downgrade", "reference-removed"])
def test_signed_new_execution_seam_cannot_drop_or_substitute_live_closure(closure_case, kind):
    case = closure_case
    execution, scope, value = _scope_case(case)
    if kind == "removed":
        scope._advisory_closure = None
    elif kind == "serialized":
        scope._advisory_closure = value.material_binding
    elif kind == "substituted":
        scope._advisory_closure = _prepare(case, output=case["root"] / "foreign-closure")
    elif kind == "subclass":
        class ForeignClosure(type(value)):
            pass
        foreign = object.__new__(ForeignClosure)
        for name in ("_seal", "_reference_bytes", "_signed_bytes"):
            object.__setattr__(foreign, name, getattr(value, name))
        scope._advisory_closure = foreign
    else:
        envelope = scope.to_dict()
        if kind == "schema-downgrade":
            envelope["payload"]["schema"] = execution.SCHEMA
            envelope["payload"]["profile"] = execution.PROFILE
        else:
            envelope["payload"].pop("advisory_closure")
        scope._material = canonical_dag_json_bytes(local._signed(envelope["payload"], case["signature_profiles"][0]["manifest"]))
    with pytest.raises(ValueError):
        scope._require_advisory_current()


@pytest.mark.parametrize("kind", ["wrong-token", "subclass"])
def test_serialized_reference_and_subclasses_cannot_construct_live_capability(closure_case, kind):
    case = closure_case
    value = _prepare(case)
    if kind == "wrong-token":
        with pytest.raises(ValueError, match="factory"):
            type(value)(object(), reference_bytes=value._reference_bytes, signed_bytes=value._signed_bytes,
                        output_directory_bytes=value._output_directory_bytes)
    else:
        class ForeignClosure(type(value)):
            pass
        foreign = object.__new__(ForeignClosure)
        for name in ("_seal", "_reference_bytes", "_signed_bytes"):
            object.__setattr__(foreign, name, getattr(value, name))
        with pytest.raises(ValueError, match="exact live"):
            _require(case, foreign)


@pytest.mark.parametrize("value", [{}, True, "serialized-live-grant", SimpleNamespace()])
def test_execution_refuses_non_typed_closure_before_reading_scheduler_or_acquiring(value, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution as execution
    # Constructor bypass here isolates only the function's first exact-type
    # gate. It must reject before any owner field or native scheduler is used.
    owner = object.__new__(RepositoryPlanPreviewOwner)
    with pytest.raises(ValueError, match="exact live native finite advisory"):
        with execution.reserve_finite_repository_execution(owner=owner, admission={}, candidate={},
                server=None, source=None, output=tmp_path / "refused",
                policy_observer=lambda request: request, advisory_closure=value):
            pytest.fail("non-typed advisory value returned a native scope")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("late_edit", [False, True])
def test_final_spawn_seam_checks_advisory_bytes_after_existing_native_guards(closure_case, late_edit):
    case = closure_case
    _, scope, _ = _scope_case(case)
    runtime = object()
    scope._runtime = runtime
    steps = []
    scope._active = lambda: steps.append("active")
    scope._physical_source = lambda: steps.append("source")
    scope._fence = lambda: steps.append("native-artifacts")
    def detached():
        steps.append("native-rows-and-launcher")
        if late_edit:
            case["paths"][0].write_bytes(b"late callback artifact mutation")
    scope._detached_fence = detached
    advisory = scope._require_advisory_current
    def verify_advisory():
        steps.append("advisory")
        return advisory()
    scope._require_advisory_current = verify_advisory
    if late_edit:
        with pytest.raises(ValueError):
            scope.require_spawn_fence(runtime)
    else:
        scope.require_spawn_fence(runtime)
    assert steps == ["active", "source", "native-artifacts", "native-rows-and-launcher", "advisory"]
    assert scope._spawned is False


def test_shutdown_seam_does_not_recheck_drifted_advisory_or_source_bytes(closure_case, monkeypatch):
    case = closure_case
    _, scope, _ = _scope_case(case)
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
    launcher = _write(case["root"] / "authored-launcher", b"authored-not-a-root-launcher")
    scope._launcher = {"path": str(launcher), "sha256": _sha(launcher.read_bytes())}
    # Root-owned launcher validation and active lease state are explicitly
    # stubbed only for this method-routing unit case; no shutdown occurs.
    monkeypatch.setattr(candidate_execution, "_root_file", lambda path: _sha(path.read_bytes()))
    active_calls = []
    scope._active = lambda **kwargs: active_calls.append(kwargs)
    def forbidden():
        pytest.fail("shutdown entered source or advisory admission")
    scope._require_advisory_current = forbidden
    scope._physical_source = forbidden
    scope.require_prelaunch_current = forbidden
    case["paths"][0].write_bytes(b"advisory changed after actual process birth")
    runtime = SimpleNamespace(manifest={"finite_execution_scope": scope.to_dict(),
                                       "finite_worker_launcher": deepcopy(scope._launcher)},
                              _context_refresh_stopped=lambda: True)
    scope._runtime = runtime
    scope.require_runtime(runtime, stopping=True)
    scope.finish_runtime(runtime)
    assert active_calls == [{"allow_cancelled": True}, {"allow_cancelled": True}]
    assert scope._cleaned is True


@pytest.mark.parametrize("alias", [True, 0])
@pytest.mark.parametrize("field", [
    "execution_authority", "proof_authority", "publication_authority", "completion_authority",
    "omission_authority", "production_activated", "authenticated_process_origin", "atomicity_attested",
    "model_enabled_for_signed_worker", "convergence_proved",
])
def test_even_owner_resigned_payload_cannot_alias_artifact_authority(closure_case, alias, field):
    case = closure_case
    reference = _prepare(case).material_binding
    payload = deepcopy(reference["signed_closure"]["payload"])
    payload["authority"][field] = alias
    resigned = _resign_reference(case, reference, payload)
    with pytest.raises(ValueError, match="closed advisory closure payload"):
        case["module"].verify_finite_advisory_artifact_closure_reference(resigned)


def test_historical_signature_verification_needs_explicit_expected_owner_for_owner_claim(closure_case):
    case = closure_case
    reference = _prepare(case).material_binding
    owner = reference["signed_closure"]["binding"]
    foreign = _resign_reference(case, reference, foreign=True)
    # An independently valid historical signer is not a live launch grant.
    assert case["module"].verify_finite_advisory_artifact_closure_reference(foreign)
    with pytest.raises(ValueError):
        case["module"].verify_finite_advisory_artifact_closure_reference(foreign, expected_binding=owner)
    assert case["module"].verify_finite_advisory_artifact_closure_reference(reference, expected_binding=owner)


@pytest.mark.parametrize("kind", [
    "file-count", "directory-count", "file-bytes", "total-bytes", "inventory-entries",
    "material-bytes", "limits-changed", "unknown-payload-field", "duplicate-file", "duplicate-directory",
    "witness-bool", "size-bool", "missing-model-role", "wrong-model-blob-cid",
])
def test_owner_resigned_record_still_obeys_count_type_and_byte_caps(closure_case, kind):
    case, module = closure_case, closure_case["module"]
    reference = _prepare(case).material_binding
    payload = deepcopy(reference["signed_closure"]["payload"])
    if kind == "file-count":
        original = payload["files"][0]
        payload["files"] = [{**deepcopy(original), "path": str(case["root"] / (f"file-{index:04d}"))}
                            for index in range(module.LIMITS["files"] + 1)]
    elif kind == "directory-count":
        original = payload["directories"][0]
        payload["directories"] = [{**deepcopy(original), "path": str(case["root"] / (f"dir-{index:04d}"))}
                                  for index in range(module.LIMITS["directories"] + 1)]
    elif kind in {"file-bytes", "total-bytes"}:
        if kind == "file-bytes":
            payload["files"][0]["size_bytes"] = module.LIMITS["file_bytes"] + 1
            payload["files"][0]["witness"][6] = payload["files"][0]["size_bytes"]
        else:
            for row in payload["files"]:
                row["size_bytes"] = module.LIMITS["file_bytes"]
                row["witness"][6] = row["size_bytes"]
    elif kind == "inventory-entries":
        payload["directories"][0]["entries"] = [{"name": f"extra-{index:04d}", "kind": "file"}
                                               for index in range(module.LIMITS["inventory_entries"] + 1)]
    elif kind == "material-bytes":
        payload["implementation"]["authored-oversized-source-label"] = "X" * (module.LIMITS["material_bytes"] + 1)
    elif kind == "limits-changed":
        payload["limits"]["file_bytes"] += 1
    elif kind == "unknown-payload-field":
        payload["native_activation"] = True
    elif kind == "duplicate-file":
        payload["files"].append(deepcopy(payload["files"][0]))
    elif kind == "duplicate-directory":
        payload["directories"].append(deepcopy(payload["directories"][0]))
    elif kind == "witness-bool":
        payload["files"][0]["witness"][0] = True
    elif kind == "size-bool":
        payload["files"][0]["size_bytes"] = False
        payload["files"][0]["witness"][6] = False
    elif kind == "missing-model-role":
        payload["model"]["frozen_context"]["artifacts"].pop("lineage")
    else:
        payload["model"]["frozen_context"]["artifacts"]["checkpoint"]["blob_cid"] = _cid("wrong-blob")
    resigned = _resign_reference(case, reference, payload)
    with pytest.raises(ValueError):
        module.verify_finite_advisory_artifact_closure_reference(resigned)


@pytest.mark.parametrize("kind", ["bytes", "same-byte-replacement", "added-output-file"])
def test_closure_file_and_output_inventory_remain_physically_sealed(closure_case, kind):
    case = closure_case
    value = _prepare(case)
    path = Path(value.material_binding["artifact"]["path"])
    if kind == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif kind == "same-byte-replacement":
        original = path.stat().st_ino
        replacement = _write(case["root"] / "new-closure-inode", path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        os.replace(replacement, path)
        assert path.stat().st_ino != original
    else:
        _write(path.parent / "unexpected-output.json", b"{}")
    with pytest.raises(ValueError):
        _require(case, value)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "oversize", "count", "existing-output", "historical-pin-drift"])
def test_factory_refuses_invalid_artifacts_without_publishing_closure(closure_case, kind):
    case, module = closure_case, closure_case["module"]
    changes = {}
    if kind == "symlink":
        target = case["paths"][0]
        copy = _write(case["root"] / "identical-original", target.read_bytes())
        target.unlink()
        target.symlink_to(copy)
    elif kind == "hardlink":
        os.link(case["paths"][0], case["root"] / "artifact-link")
    elif kind == "fifo":
        target = case["paths"][0]
        target.unlink()
        os.mkfifo(target, 0o600)
    elif kind == "oversize":
        with case["paths"][0].open("r+b") as stream:
            stream.truncate(module.LIMITS["file_bytes"] + 1)
    elif kind == "count":
        changes["artifact_paths"] = [case["paths"][0]] * (module.LIMITS["files"] + 1)
    elif kind == "existing-output":
        case["arguments"]["output"].mkdir()
    else:
        changes["artifact_paths"] = [{"path": str(case["paths"][0]), "bytes": case["paths"][0].stat().st_size,
                                     "sha256": "0" * 64}]
    with pytest.raises((ValueError, OSError)):
        _prepare(case, **changes)
    assert not (case["arguments"]["output"] / "closure.json").exists()


def test_previously_absent_registry_companion_cannot_appear_after_seal(closure_case, monkeypatch):
    case = closure_case
    absent = case["root"] / "authored-registry.duckdb.wal"
    original = case["module"]._mandatory_paths
    def mandatory(**kwargs):
        selected = original(**kwargs)
        selected["absent_paths"] = [absent]
        return selected
    monkeypatch.setattr(case["module"], "_mandatory_paths", mandatory)
    value = _prepare(case)
    _write(absent, b"new companion bytes")
    with pytest.raises(ValueError, match="absent registry companion"):
        _require(case, value)


@pytest.mark.parametrize("mutation", ["write-restore", "same-byte-replacement"])
def test_mutation_during_real_descriptor_read_cannot_hide_behind_original_bytes(closure_case, monkeypatch, mutation):
    case = closure_case
    value = _prepare(case)
    target = case["paths"][0]
    raw, inode = target.read_bytes(), target.stat().st_ino
    original = os.read
    hits = []
    def read(descriptor, size):
        block = original(descriptor, size)
        if not hits and block and os.fstat(descriptor).st_ino == inode:
            hits.append(True)
            if mutation == "write-restore":
                target.write_bytes(raw + b"\n")
                target.write_bytes(raw)
            else:
                replacement = _write(case["root"] / "read-race-new-inode", raw,
                                     stat.S_IMODE(target.stat().st_mode))
                os.replace(replacement, target)
        return block
    monkeypatch.setattr(os, "read", read)
    with pytest.raises(ValueError):
        _require(case, value)
    assert hits == [True], "the mutation must follow the actual selected descriptor read"
    assert target.read_bytes() == raw


def _mandatory_expected(case, monkeypatch, *, change=None):
    """Freeze authored validated byte identities before the factory reads them."""
    expected = {str(path): {"bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}
                for path in (*case["paths"], case["candidate_path"])}
    if change is not None:
        change(expected)
    original = case["module"]._mandatory_paths
    def mandatory(**kwargs):
        selected = original(**kwargs)
        selected["expected"] = deepcopy(expected)
        return selected
    monkeypatch.setattr(case["module"], "_mandatory_paths", mandatory)
    return expected


def test_omitting_caller_extras_keeps_every_mandatory_path_and_validated_identity(closure_case, monkeypatch):
    case = closure_case
    expected = _mandatory_expected(case, monkeypatch)
    value = _prepare(case, artifact_paths=[])
    files = value.material_binding["signed_closure"]["payload"]["files"]
    assert set(expected) <= {row["path"] for row in files}
    assert str(Path(case["module"].__file__).absolute()) in {row["path"] for row in files}
    assert {row["path"]: {"bytes": row["size_bytes"], "sha256": row["sha256"]}
            for row in files if row["path"] in expected} == expected
    _require(case, value)


@pytest.mark.parametrize("kind", ["sha", "size", "size-bool", "unknown-field", "unselected-path"])
def test_mandatory_validated_identity_is_closed_and_independent_of_caller_extras(closure_case, monkeypatch, kind):
    case = closure_case
    path = str(case["paths"][0])
    def change(expected):
        if kind == "sha":
            expected[path]["sha256"] = "0" * 64
        elif kind == "size":
            expected[path]["bytes"] += 1
        elif kind == "size-bool":
            expected[path]["bytes"] = False
        elif kind == "unknown-field":
            expected[path]["owner_approved"] = True
        else:
            expected[str(case["root"] / "never-selected-path")] = deepcopy(expected[path])
    _mandatory_expected(case, monkeypatch, change=change)
    with pytest.raises((ValueError, OSError)):
        _prepare(case, artifact_paths=[])
    assert not (case["arguments"]["output"] / "closure.json").exists()


def test_conflicting_mandatory_and_extra_identity_cannot_override_native_selection(closure_case, monkeypatch):
    case = closure_case
    path = case["paths"][0]
    _mandatory_expected(case, monkeypatch, change=lambda values: values[str(path)].update(sha256="0" * 64))
    caller_correct = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}
    with pytest.raises(ValueError, match="mandatory artifact"):
        _prepare(case, artifact_paths=[caller_correct])
    assert not (case["arguments"]["output"] / "closure.json").exists()


def test_byte_mutation_after_preparation_cannot_become_new_mandatory_baseline(closure_case, monkeypatch):
    case = closure_case
    _mandatory_expected(case, monkeypatch)
    original = case["module"]._validate_preparation
    touched = []
    def validate(**kwargs):
        prepared = original(**kwargs)
        touched.append(True)
        case["paths"][0].write_bytes(b"changed after authored native preparation returns")
        return prepared
    monkeypatch.setattr(case["module"], "_validate_preparation", validate)
    with pytest.raises(ValueError, match="mandatory artifact"):
        _prepare(case, artifact_paths=[])
    assert touched == [True]
    assert not (case["arguments"]["output"] / "closure.json").exists()


def test_actual_signer_callback_mutation_after_pin_capture_cannot_return_live_closure(closure_case, monkeypatch):
    case = closure_case
    _mandatory_expected(case, monkeypatch)
    original = local._signed
    signed = []
    def sign(payload, manifest):
        envelope = original(payload, manifest)
        if payload.get("schema") == case["module"].SCHEMA:
            # Real signing already completed; the independent factory must
            # close the previously captured file identities after callbacks.
            signed.append(envelope)
            case["paths"][0].write_bytes(b"mutated after actual Ed25519 seal signing")
        return envelope
    monkeypatch.setattr(local, "_signed", sign)
    with pytest.raises(ValueError):
        _prepare(case, artifact_paths=[])
    assert len(signed) == 1
    assert local._verify_signature(signed[0], case["signature_profiles"][0]["profile"])
