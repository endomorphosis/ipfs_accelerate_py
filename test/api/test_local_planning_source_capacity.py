"""Reserve declared creations inside the existing signed source limits.

The declaration controls isolate capacity from the independently tested scan
and successor-context validators. The two integration controls use a genuine
temporary Git repository, local owner profile and signatures, without models,
workers, private source owners or task materialization.
"""
from copy import deepcopy
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


CAPACITY_ERROR = "declared creations exceed local source inventory capacity"


def _declaration(schema, baseline_count, creation_count):
    sources = {f"source_{number:04d}.py": {"sha256": "a" * 64, "executable": False}
               for number in range(baseline_count)}
    names = [f"created_{number:02d}.json" for number in range(creation_count)]
    policy = content_identity(local.LOCAL_POLICY)
    task = {"task_key": "CAPACITY", "scope_paths": ["source_0000.py", *names],
        "outputs": [{"path": "source_0000.py", "effect": "modify", "media_type": "text/x-python"},
                    *[{"path": name, "effect": "create", "media_type": "application/json"} for name in names]],
        "validations": [{"validation_key": "public", "argv": [sys.executable, "source_0000.py"],
            "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy}],
        "acceptance": [{"criterion_key": "public-pass", "criterion": "Retain the public check",
            "evidence_cids": [], "validation_keys": ["public"]}], "dependencies": []}
    payload = {"schema": schema, "repository": "/authored/repository", "repository_cid": "authored",
        "baseline_commit": "a" * 40, "profile_content_id": "authored",
        "profile_dir": "/authored/profile", "lifecycle_dir": "/authored/lifecycle",
        "sources": sources, "policy": deepcopy(local.LOCAL_POLICY),
        "planning_roots": {name: content_identity({"authored-capacity": name})
            for name in ("request_cid", "scan_cid", "program_root")}, "tasks": [task]}
    if local.supports_created_outputs(payload):
        payload["created_outputs"] = names
    if schema == local.PLANNER_MANIFEST_SCHEMA:
        payload["planning_inputs"] = {}
    if schema == local.INTENT_MANIFEST_SCHEMA:
        payload["intent_requirements"] = {}
    if schema in local.INVENTORY_MANIFEST_SCHEMAS:
        payload["codebase_inventory_context"] = {"controlled-capacity-only": True}
    if schema == local.SUCCESSOR_MANIFEST_SCHEMA:
        payload["codebase_successor_context"] = {"controlled-capacity-only": True}
    return payload


@pytest.fixture
def controlled_context_validation(monkeypatch):
    """Do not claim completeness/currentness from the capacity-only metadata."""
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_worker_context as inventory
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_context as successor
    monkeypatch.setattr(inventory, "validate_inventory_declaration", lambda value, **_: value)
    monkeypatch.setattr(successor, "validate_successor_declaration", lambda value, **_: value)


@pytest.mark.parametrize("schema", [local.INVENTORY_MANIFEST_SCHEMA, local.SUCCESSOR_MANIFEST_SCHEMA])
@pytest.mark.parametrize("baseline,created,accepted", [
    (1024, 0, True), (1023, 1, True), (1024, 1, False), (992, 32, True), (993, 32, False),
])
def test_inventory_and_successor_reserve_all_declared_creation_slots(
        controlled_context_validation, schema, baseline, created, accepted):
    payload = _declaration(schema, baseline, created)
    before = deepcopy(payload)
    if accepted:
        outputs = local._validate_local_manifest_declarations(payload)
        assert outputs == {"source_0000.py", *payload["created_outputs"]}
    else:
        with pytest.raises(local.LocalPlanningError, match=CAPACITY_ERROR):
            local._validate_local_manifest_declarations(payload)
    assert payload == before  # Capacity must never omit signed members or tasks.


@pytest.mark.parametrize("schema", [local.CREATE_MANIFEST_SCHEMA, local.INTENT_MANIFEST_SCHEMA])
@pytest.mark.parametrize("baseline,created,accepted", [
    (255, 1, True), (256, 1, False), (224, 32, True), (225, 32, False),
])
def test_historical_creation_profiles_keep_their_256_file_limit(schema, baseline, created, accepted):
    payload = _declaration(schema, baseline, created)
    if accepted:
        assert len(local._validate_local_manifest_declarations(payload)) == created + 1
    else:
        with pytest.raises(local.LocalPlanningError, match=CAPACITY_ERROR):
            local._validate_local_manifest_declarations(payload)


@pytest.mark.parametrize("schema", [local.MANIFEST_SCHEMA, local.PLANNER_MANIFEST_SCHEMA])
@pytest.mark.parametrize("baseline", [128, 129])
def test_historical_modify_only_profiles_keep_their_128_file_limit(schema, baseline):
    payload = _declaration(schema, baseline, 0)
    if baseline == 128:
        assert local._validate_local_manifest_declarations(payload) == {"source_0000.py"}
    else:
        with pytest.raises(local.LocalPlanningError, match="inventory bound"):
            local._validate_local_manifest_declarations(payload)


@pytest.mark.parametrize("schema", [local.INVENTORY_MANIFEST_SCHEMA, local.SUCCESSOR_MANIFEST_SCHEMA])
@pytest.mark.parametrize("mutation", ["duplicate_output", "duplicate_declared_name", "modify_disguised_as_create"])
def test_duplicate_or_existing_output_cannot_fake_creation_capacity(
        controlled_context_validation, schema, mutation):
    payload = _declaration(schema, 1023, 1)
    task = payload["tasks"][0]
    if mutation == "duplicate_output":
        task["outputs"].append(deepcopy(task["outputs"][-1]))
    elif mutation == "duplicate_declared_name":
        payload["created_outputs"].append(payload["created_outputs"][0])
    else:
        payload["created_outputs"] = ["source_0000.py"]
        task["scope_paths"] = ["source_0000.py"]
        task["outputs"] = [{"path": "source_0000.py", "effect": "create", "media_type": "text/x-python"}]
    with pytest.raises(local.LocalPlanningError, match="declared-create|absent at baseline"):
        local._validate_local_manifest_declarations(payload)


@pytest.mark.parametrize("schema", [local.INVENTORY_MANIFEST_SCHEMA, local.SUCCESSOR_MANIFEST_SCHEMA])
def test_modification_at_full_capacity_does_not_consume_an_absent_output_slot(controlled_context_validation, schema):
    payload = _declaration(schema, 1024, 0)
    assert local._validate_local_manifest_declarations(payload) == {"source_0000.py"}
    assert len(payload["sources"]) == 1024 and payload["created_outputs"] == []


@pytest.fixture
def large_local_case(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "lifecycle-registry")
    monkeypatch.delenv(meta.ENV_DUCKDB, raising=False)
    monkeypatch.delenv(meta.ENV_DUCKLAKE, raising=False)
    repository = tmp_path / "repository"
    repository.mkdir()
    for number in range(256):
        (repository / f"source_{number:04d}.py").write_text(f"VALUE = {number}\n")
    for arguments in (("init", "-q"), ("add", "."),
            ("-c", "user.name=Capacity controls", "-c", "user.email=capacity@example.invalid",
             "commit", "-qm", "complete independent source inventory")):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True)
    profile_dir, lifecycle_dir = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    payload = _declaration(local.CREATE_MANIFEST_SCHEMA, 256, 1)
    root, head, repository_cid = local._repository(repository)
    profile = local.load_local_profile(repository_cid=repository_cid,
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    payload.update(repository=str(root), repository_cid=repository_cid, baseline_commit=head,
        profile_content_id=profile.content_id, profile_dir=str(profile_dir), lifecycle_dir=str(lifecycle_dir),
        sources=local._sources(root, sorted(payload["sources"]), max_files=256))
    check = PromptValidationRecord(**payload["tasks"][0]["validations"][0])
    acceptance = PromptAcceptanceRecord(**payload["tasks"][0]["acceptance"][0])
    scope = tuple(payload["tasks"][0]["scope_paths"])
    goal = PromptGoalRecord(goal_key="CAPACITY-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Preserve all sources", objective="Create the declared report", rationale="Independent administrator data",
        scope_paths=scope, acceptance=(acceptance,))
    task = PromptTaskRecord(task_key="CAPACITY", goal_cid=goal.goal_cid, dependency_task_cids=(),
        objective=goal.objective, rationale=goal.rationale, scope_paths=scope,
        outputs=tuple(PromptOutputRecord(**item) for item in payload["tasks"][0]["outputs"]),
        validations=(check,), acceptance=(acceptance,), evidence_cids=(),
        policy_roots=(content_identity(local.LOCAL_POLICY),), predicted_files=scope)
    graph = PromptGoalGraph(**payload["planning_roots"], policy_roots=task.policy_roots,
        goals=(goal,), tasks=(task,), evidence=())
    return {"repository": repository, "payload": payload, "profile": profile, "graph": graph}


def test_genuine_manifest_author_refuses_over_capacity_before_signing(large_local_case, monkeypatch):
    case = large_local_case
    payload = case["payload"]
    signed_calls = []
    def forbidden_signature(*_, **__):
        signed_calls.append(True)
        pytest.fail("over-capacity manifest reached owner signing")
    monkeypatch.setattr(local, "_signed", forbidden_signature)
    with pytest.raises(local.LocalPlanningError, match=CAPACITY_ERROR):
        local.author_local_benchmark_manifest(repository=case["repository"], profile_dir=payload["profile_dir"],
            lifecycle_dir=payload["lifecycle_dir"], task_specs=payload["tasks"], planning_roots=payload["planning_roots"])
    assert signed_calls == []


def test_real_owner_resign_cannot_authorize_over_capacity_admission(large_local_case, monkeypatch):
    case = large_local_case
    envelope = local._signed(case["payload"], case["payload"])
    assert local._verify_signature(envelope, case["profile"]) == case["payload"]
    signed_calls = []
    def forbidden_receipt(*_, **__):
        signed_calls.append(True)
        pytest.fail("over-capacity admission reached receipt signing")
    monkeypatch.setattr(local, "_signed", forbidden_receipt)
    with pytest.raises(local.LocalPlanningError, match=CAPACITY_ERROR):
        local.admit_local_benchmark_plan(graph=case["graph"], manifest=envelope)
    assert signed_calls == []
