"""Strict role policies over real native context production and source gates.

Role labels are owner declarations. These tests establish bounded selection and
binding preservation, not domain completeness, provider delivery, or savings.
Native scanning, capsule freshness, packing, storage, and reload are unchanged.
"""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
    build_code_symbol_vector_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import (
    build_program_evidence_index,
)
from ipfs_accelerate_py.agent_supervisor.runtime import context_selection_observation
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    prepare_code_retrieval_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.role_context_selection import (
    RoleContextProfile,
    load_role_semantic_context,
    prepare_role_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
    ProgramWorldDatabase,
)

TASK = "ROLE-CONTEXT-001"
OBJECTIVE = "Inspect the requested operation while preserving every public constraint."
INSTRUCTION = "Keep this complete instruction, including its Unicode: café, λ.\n"


class TextSubclass(str):
    pass


class DictSubclass(dict):
    pass


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "code.py").write_text(
        "def alpha(value):\n    return value + 1\n\n"
        "def zeta(value):\n    return value - 1\n",
        encoding="utf-8",
    )
    (root / "instruction.md").write_text(INSTRUCTION, encoding="utf-8")
    return root


def _prepare(root, profile, *, name, program=("code.py",)):
    return prepare_role_semantic_context(
        repository=root,
        paths=["code.py", "instruction.md"],
        required_raw_paths=["instruction.md"],
        program_paths=list(program),
        objective=OBJECTIVE,
        task_id=TASK,
        output=root / ".roles" / name,
        profile=profile,
    )


def _load(root, reference, profile, **overrides):
    return load_role_semantic_context(**{
        "repository": root,
        "artifact": reference["artifact"],
        "expected_sha256": reference["sha256"],
        "task_id": TASK,
        "profile": profile,
        **overrides,
    })


def _assert_native_binding(root, reference, profile):
    assert reference["schema"] == "supervisor-role-context-selection-ref@1"
    assert reference["role"] == profile.role
    assert reference["profile_sha256"] == profile.fingerprint()
    assert reference["completion_authority"] is False
    assert reference["provider_delivery_verified"] is False
    manifest_raw = (root / reference["artifact"]).read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest["schema"] == "supervisor-role-context-selection@1"
    assert _encoded(manifest) == manifest_raw
    assert _sha(manifest_raw) == reference["sha256"]
    assert manifest["profile"] == profile.to_dict()
    assert manifest["profile_sha256"] == profile.fingerprint()
    assert manifest["task_id"] == TASK
    assert all(value is False for value in manifest["authority"].values())
    assert manifest["provider_token_usage"] is manifest["token_savings"] is None
    worker_path = root / manifest["worker"]["artifact"]
    worker_raw = worker_path.read_bytes()
    assert manifest["worker"]["sha256"] == reference["worker_payload_sha256"] == _sha(worker_raw)
    assert manifest["worker"]["bytes"] == len(worker_raw)
    payload = json.loads(worker_raw)
    assert payload["schema"] == "supervisor-semantic-worker-context@2"
    assert payload["raw_sources"]["instruction.md"] == INSTRUCTION
    assert payload["completion_authority"] is False
    assert payload["worker_projection"]["query"] == profile.query
    assert payload["worker_projection"]["max_capsules"] == profile.capsule_limit
    assert payload["worker_projection"]["max_bytes"] == profile.max_bytes
    assert "profile" not in payload and "role" not in payload
    assert manifest["bindings"] == {
        "scope_cid": payload["scope_cid"],
        "semantic_root_cid": payload["semantic_root_cid"],
        "pack_cid": payload["pack"]["pack_cid"],
    }
    selection_ref = manifest["context_selection_observation"]
    assert selection_ref["role"] == profile.role
    if selection_ref["schema"] == "supervisor-context-selection-observation-unavailable@1":
        assert selection_ref == {
            "schema": "supervisor-context-selection-observation-unavailable@1",
            "status": "unavailable", "reason": "observation_resource_bound", "role": profile.role,
        }
        assert not (worker_path.parent / "context-selection.json").exists()
    else:
        selection_raw = (worker_path.parent / selection_ref["artifact"]).read_bytes()
        assert _sha(selection_raw) == selection_ref["sha256"]
        selection = json.loads(selection_raw)
        assert selection["role"] == profile.role
        assert selection["role_basis"] == "owner_declared"
        assert selection["payload"]["sha256"] == _sha(worker_raw)
        assert selection["selection"]["selected_capsule_count"] == len(payload["capsules"])
        assert all(value is False for value in selection["authority"].values())
    records = ProgramWorldDatabase(
        worker_path.parent / "world.duckdb", worker_path.parent / "world-lake"
    ).records_for_decision(task_id=TASK)
    assert records["n"] == 1
    assert records["records"][0]["payload"]["context_selection_observation"] == selection_ref
    assert records["ducklake_authoritative"] is False
    assert _load(root, reference, profile) == worker_raw
    return manifest, payload, worker_raw


def test_profiles_round_trip_are_immutable_and_preserve_exact_policy():
    profile = RoleContextProfile("planning", "inspect λ", capsule_limit=0, max_bytes=8192)
    document = profile.to_dict()
    assert document == {
        "schema": "supervisor-role-context-profile@1", "role": "planning", "query": "inspect λ",
        "policy_id": "role-context@1", "policy_revision": 1, "capsule_limit": 0, "max_bytes": 8192,
        "retrieval_artifact": None, "retrieval_sha256": None,
    }
    assert RoleContextProfile.from_dict(document) == profile
    assert profile.fingerprint() == _sha(_encoded(document))
    assert profile.native_policy() == {
        "worker_query": "inspect λ", "worker_capsule_limit": 0, "worker_max_bytes": 8192,
    }
    with pytest.raises(FrozenInstanceError):
        profile.query = "changed"
    document["query"] = "changed"
    assert profile.query == "inspect λ"
    assert profile.to_dict()["query"] == "inspect λ"
    ranked = RoleContextProfile(
        "review", "inspect", retrieval_artifact="retrieval/index.json", retrieval_sha256="a" * 64,
    )
    assert ranked.native_policy()["worker_retrieval"] == {
        "artifact": "retrieval/index.json", "sha256": "a" * 64,
    }
    assert RoleContextProfile("coding", "λ" * 4096).query == "λ" * 4096
    assert ranked.fingerprint() != profile.fingerprint()


@pytest.mark.parametrize("changes", [
    {"role": "worker"}, {"role": "Coding"}, {"role": None}, {"role": TextSubclass("coding")},
    {"query": ""}, {"query": " \n\t"}, {"query": None}, {"query": b"inspect"},
    {"query": TextSubclass("inspect")}, {"query": "λ" * 4097},
    {"policy_id": ""}, {"policy_id": None}, {"policy_id": TextSubclass("role-context@1")},
    {"policy_revision": True}, {"policy_revision": 0}, {"policy_revision": 1.0},
    {"capsule_limit": True}, {"capsule_limit": -1}, {"capsule_limit": 33},
    {"capsule_limit": 1.0}, {"capsule_limit": float("nan")},
    {"max_bytes": True}, {"max_bytes": 8191}, {"max_bytes": 65537},
    {"max_bytes": 32768.0}, {"max_bytes": float("inf")},
    {"retrieval_artifact": "index.json"}, {"retrieval_sha256": "a" * 64},
    {"retrieval_artifact": "index.json", "retrieval_sha256": "A" * 64},
    {"retrieval_artifact": "index.json", "retrieval_sha256": "g" * 64},
    {"retrieval_artifact": "index.json", "retrieval_sha256": "a" * 63},
    {"retrieval_artifact": "index.json", "retrieval_sha256": b"a" * 64},
])
def test_profile_rejects_noncanonical_types_and_unbounded_policies(changes):
    with pytest.raises(ValueError):
        RoleContextProfile(**{"role": "coding", "query": "inspect", **changes})


@pytest.mark.parametrize("path", ["", ".", "./index.json", "../index.json", "/index.json",
                                  "a/../index.json", "a//index.json", "index.json\x00"])
def test_retrieval_handles_must_be_canonical_relative_paths(path):
    with pytest.raises(ValueError):
        RoleContextProfile("review", "inspect", retrieval_artifact=path, retrieval_sha256="a" * 64)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "schema", "dict-subclass"])
def test_profile_wire_document_has_closed_exact_fields(mutation):
    document = RoleContextProfile("coding", "inspect").to_dict()
    if mutation == "unknown":
        document["omission_authority"] = True
    elif mutation == "missing":
        del document["query"]
    elif mutation == "schema":
        document["schema"] = "supervisor-role-context-profile@2"
    else:
        document = DictSubclass(document)
    with pytest.raises(ValueError):
        RoleContextProfile.from_dict(document)


@pytest.mark.parametrize("field,value", [
    ("role", "unknown"), ("capsule_limit", True), ("policy_revision", False),
    ("retrieval_sha256", "A" * 64),
])
@pytest.mark.parametrize("method", ["to_dict", "fingerprint", "native_policy"])
def test_public_profile_methods_revalidate_frozen_object_bypasses(field, value, method):
    profile = RoleContextProfile("coding", "inspect")
    object.__setattr__(profile, field, value)
    with pytest.raises(ValueError):
        getattr(profile, method)()


def test_native_roles_select_different_symbols_and_keep_exact_source_gates(tmp_path):
    root = _repository(tmp_path)
    coding = RoleContextProfile("coding", "alpha", capsule_limit=1)
    planning = RoleContextProfile("planning", "zeta", capsule_limit=2)
    coding_ref = _prepare(root, coding, name="coding")
    planning_ref = _prepare(root, planning, name="planning")
    coding_manifest, coding_payload, coding_raw = _assert_native_binding(root, coding_ref, coding)
    planning_manifest, planning_payload, planning_raw = _assert_native_binding(root, planning_ref, planning)
    assert coding_ref["artifact"] != planning_ref["artifact"]
    assert coding_raw != planning_raw
    assert coding_payload["scope_cid"] == planning_payload["scope_cid"]
    assert coding_payload["semantic_root_cid"] == planning_payload["semantic_root_cid"]
    assert coding_payload["manifest"] == planning_payload["manifest"]
    assert coding_payload["raw_sources"] == planning_payload["raw_sources"]
    coding_symbols = coding_payload["worker_projection"]["selected_symbols"]
    planning_symbols = planning_payload["worker_projection"]["selected_symbols"]
    assert len(coding_symbols) == 1 and coding_symbols[0]["qualified_name"] == "code.alpha"
    assert len(planning_symbols) == 2 and planning_symbols[0]["qualified_name"] == "code.zeta"

    # The same native policy with its default coding observation label must
    # render identical worker input to the wrapper's planning observation.
    baseline = root / ".roles/default-native"
    default = prepare_semantic_context(
        repository=root, paths=["code.py", "instruction.md"],
        required_raw_paths=["instruction.md"], program_paths=["code.py"],
        objective=OBJECTIVE, task_id=TASK, output=baseline, **planning.native_policy(),
    )
    assert (baseline / "worker-context.json").read_bytes() == planning_raw
    assert default["context_selection_observation"]["role"] == "coding"
    assert planning_manifest["context_selection_observation"]["role"] == "planning"

    for profile in (planning, RoleContextProfile("review", "alpha", capsule_limit=1),
                    RoleContextProfile("coding", "zeta", capsule_limit=1),
                    RoleContextProfile("coding", "alpha", capsule_limit=2),
                    RoleContextProfile("coding", "alpha", capsule_limit=1, policy_revision=2)):
        with pytest.raises(ValueError):
            _load(root, coding_ref, profile)
    with pytest.raises(ValueError):
        _load(root, coding_ref, coding, task_id="FOREIGN-TASK")

    manifest_path = root / coding_ref["artifact"]
    original_manifest = manifest_path.read_bytes()
    for mutation in ("worker-sha", "worker-bytes", "worker-escape", "bindings", "observation",
                     "authority", "profile", "profile-sha", "task", "unknown"):
        forged = json.loads(original_manifest)
        if mutation == "worker-sha":
            forged["worker"]["sha256"] = "0" * 64
        elif mutation == "worker-bytes":
            forged["worker"]["bytes"] += 1
        elif mutation == "worker-escape":
            forged["worker"]["artifact"] = "../outside.json"
        elif mutation == "bindings":
            forged["bindings"]["scope_cid"] = "forged-scope"
        elif mutation == "observation":
            forged["context_selection_observation"]["sha256"] = "0" * 64
        elif mutation == "authority":
            forged["authority"]["completion_authority"] = True
        elif mutation == "profile":
            forged["profile"]["role"] = "review"
        elif mutation == "profile-sha":
            forged["profile_sha256"] = "0" * 64
        elif mutation == "task":
            forged["task_id"] = "FOREIGN-TASK"
        else:
            forged["unknown_authority"] = True
        raw = _encoded(forged)
        try:
            manifest_path.write_bytes(raw)
            with pytest.raises(ValueError):
                _load(root, coding_ref, coding, expected_sha256=_sha(raw))
        finally:
            manifest_path.write_bytes(original_manifest)
        assert (root / coding_manifest["worker"]["artifact"]).read_bytes() == coding_raw

    sidecar_path = manifest_path.parent / "context-selection.json"
    original_sidecar = sidecar_path.read_bytes()
    foreign_sidecar = (root / planning_ref["artifact"]).parent / "context-selection.json"
    for observation in ("foreign-valid-role", "forged-unavailable"):
        forged = json.loads(original_manifest)
        if observation == "foreign-valid-role":
            forged["context_selection_observation"] = planning_manifest["context_selection_observation"]
            sidecar_path.write_bytes(foreign_sidecar.read_bytes())
        else:
            forged["context_selection_observation"] = {
                "schema": "supervisor-context-selection-observation-unavailable@1",
                "status": "unavailable", "reason": "observation_resource_bound", "role": "coding",
            }
            sidecar_path.unlink()
        raw = _encoded(forged)
        try:
            manifest_path.write_bytes(raw)
            with pytest.raises(ValueError):
                _load(root, coding_ref, coding, expected_sha256=_sha(raw))
        finally:
            sidecar_path.write_bytes(original_sidecar)
            manifest_path.write_bytes(original_manifest)

    # Even unchanged, digest-matching bytes cannot be loaded through an escape.
    worker_path = root / coding_manifest["worker"]["artifact"]
    for index, path in enumerate((manifest_path, worker_path, sidecar_path)):
        original = path.read_bytes()
        escaped = tmp_path / f"outside-{index}.json"
        escaped.write_bytes(original)
        try:
            path.unlink()
            path.symlink_to(escaped)
            with pytest.raises(ValueError):
                _load(root, coding_ref, coding)
        finally:
            path.unlink()
            path.write_bytes(original)

    for name, profile, reference in (("coding", coding, coding_ref), ("planning", planning, planning_ref)):
        original = (root / reference["artifact"]).read_bytes()
        with pytest.raises((ValueError, FileExistsError)):
            _prepare(root, profile, name=name)
        assert (root / reference["artifact"]).read_bytes() == original
    (root / "code.py").write_text("def alpha(value):\n    return value + 7\n", encoding="utf-8")
    for reference, profile in ((coding_ref, coding), (planning_ref, planning)):
        with pytest.raises(ValueError):
            _load(root, reference, profile)
    assert (root / coding_manifest["worker"]["artifact"]).read_bytes() == coding_raw
    assert (root / planning_manifest["worker"]["artifact"]).read_bytes() == planning_raw


@pytest.mark.parametrize("observation_bound", [None, 1], ids=["observed", "real-observer-bound"])
def test_empty_program_and_zero_caps_preserve_required_raw_without_invented_coverage(
    tmp_path, monkeypatch, observation_bound,
):
    normal_observation_bound = context_selection_observation.MAX_OBSERVATION_BYTES
    if observation_bound is not None:
        # Exercise the real observer's encoder bound throughout preparation and
        # replay; the native producer, storage and freshness checks stay real.
        monkeypatch.setattr(context_selection_observation, "MAX_OBSERVATION_BYTES", observation_bound)
    root = _repository(tmp_path)
    profile = RoleContextProfile("review", "alpha", capsule_limit=0)
    reference = _prepare(root, profile, name="empty-review", program=())
    manifest, payload, raw = _assert_native_binding(root, reference, profile)
    assert manifest["context_selection_observation"]["schema"] == (
        "supervisor-context-selection-observation-ref@1" if observation_bound is None
        else "supervisor-context-selection-observation-unavailable@1"
    )
    assert payload["program_paths"] == []
    assert payload["capsules"] == payload["admissions"] == []
    assert payload["worker_projection"]["full_capsule_count"] == 0
    assert payload["worker_projection"]["selected_capsule_count"] == 0
    assert payload["reconstruction"]["semantic_acceptance_authority"] is False
    assert payload["raw_sources"] == {
        name: (root / name).read_text(encoding="utf-8") for name in payload["manifest"]
    }
    if observation_bound is not None:
        with monkeypatch.context() as restored:
            restored.setattr(context_selection_observation, "MAX_OBSERVATION_BYTES", normal_observation_bound)
            with pytest.raises(ValueError):
                _load(root, reference, profile)
        (root / "instruction.md").write_text("Changed required instruction.\n", encoding="utf-8")
        with pytest.raises(ValueError):
            _load(root, reference, profile)
        assert (root / manifest["worker"]["artifact"]).read_bytes() == raw


def test_native_role_retrieval_replays_complete_ast_and_refuses_corruption_and_source_drift(tmp_path):
    root = _repository(tmp_path)
    source = (root / "code.py").read_text(encoding="utf-8")
    ast = build_program_evidence_index({"code.py": source}).ast_index
    # Authored 2D vectors establish replay and selection behavior only. They do
    # not measure semantic embedding quality or a learned relevance policy.
    snapshot = build_code_symbol_vector_index(
        ast, forest_id="test:scope", tree_id="test:tree", coverage_id=ast.index_id,
        dimensions=2, model_id="authored-test-vectors", model_revision="1",
        configuration_id="test:2d",
        vectors=lambda row: (1., 0.) if row.symbol == "zeta" else (0., 1.),
    )
    retrieval_path = root / "retrieval.json"
    retrieval = prepare_code_retrieval_context(
        repository=root, task_id=TASK, query_text="alpha", snapshot=snapshot,
        result=snapshot.search((1., 0.), max_results=1), output=retrieval_path,
    )
    profile = RoleContextProfile(
        "review", "alpha", capsule_limit=1,
        retrieval_artifact="retrieval.json", retrieval_sha256=retrieval["sha256"],
    )
    reference = _prepare(root, profile, name="retrieval-review")
    manifest, payload, raw = _assert_native_binding(root, reference, profile)
    projection = payload["worker_projection"]
    assert projection["schema"] == "supervisor-semantic-worker-projection@2"
    assert projection["selector"] == "verified-retrieval-symbol-rank@1"
    assert projection["query"] == "alpha"
    assert len(projection["selected_symbols"]) == 1
    selected = projection["selected_symbols"][0]
    assert selected["qualified_name"] == "code.zeta"
    assert projection["ranked_symbol_ids"] == [selected["stable_symbol_id"]]
    assert projection["retrieval"] == {
        "artifact": "retrieval.json", "sha256": retrieval["sha256"],
    }
    retained_retrieval = retrieval_path.read_bytes()
    try:
        retrieval_path.write_bytes(b"{}")
        with pytest.raises(ValueError):
            _load(root, reference, profile)
    finally:
        retrieval_path.write_bytes(retained_retrieval)
    assert _load(root, reference, profile) == raw
    (root / "code.py").write_text(source.replace("value - 1", "value - 7"), encoding="utf-8")
    with pytest.raises(ValueError):
        _load(root, reference, profile)
    assert (root / manifest["worker"]["artifact"]).read_bytes() == raw
