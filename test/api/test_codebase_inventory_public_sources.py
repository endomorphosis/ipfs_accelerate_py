"""Inventory public Git reads retain the signed source and worktree boundary.

Git's own different-owner test switch exercises its actual ownership refusal;
the separate disposable Docker qualification exercises real UID1000/1001.
These controls perform no training, inference, provider calls or workers.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution import GIT_OWNER_ENV
from test.api.test_codebase_inventory_evidence_admission import inventory_case as inventory_case


def _git(root, *args):
    return subprocess.check_output(["/usr/bin/git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def observation_case(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "answer.py").write_text("ANSWER = 1\n")
    (root / "check.py").write_text("assert ANSWER == 2\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Public inventory controls", "-c", "user.email=control@example.invalid",
         "commit", "-qm", "independent sources")
    manifest = {"schema": local.INVENTORY_MANIFEST_SCHEMA,
        "baseline_commit": _git(root, "rev-parse", "HEAD"), "created_outputs": [],
        "sources": local._sources(root, ["answer.py", "check.py"]),
        "tasks": [{"outputs": [{"path": "answer.py", "effect": "modify"}]}]}
    return root, manifest


def test_actual_git_owner_refusal_requires_only_exact_per_call_public_trust(observation_case, monkeypatch):
    root, manifest = observation_case
    actual = subprocess.check_output
    observations = []
    def different_owner(argv, **options):
        observations.append((list(argv), dict(options["env"])))
        return actual(argv, **{**options, "env": {**options["env"], "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}})
    monkeypatch.setattr(local.subprocess, "check_output", different_owner)
    with pytest.raises(subprocess.CalledProcessError):
        local.observe_local_manifest_sources(root, manifest, initial=True)
    assert local.observe_public_inventory_manifest_sources(root, manifest, initial=True) == manifest["sources"]
    assert len(observations) == 4
    assert "safe.directory=" + str(root) not in observations[0][0]
    for argv, environment in observations[1:]:
        assert argv[:10] == ["/usr/bin/git", "--no-replace-objects", "-c", "safe.directory=" + str(root),
            "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-C", str(root)]
        assert environment == {"PATH": "/usr/bin:/bin", **GIT_OWNER_ENV}
        assert "safe.directory=*" not in argv
    # No local/global configuration is installed by a public read.
    with pytest.raises(subprocess.CalledProcessError):
        actual(["/usr/bin/git", "-C", str(root), "config", "--local", "--get-all", "safe.directory"], text=True)


@pytest.mark.parametrize("args", [
    ("status", "--porcelain"), ("config", "safe.directory", "*"),
    ("ls-files", "--others", "--exclude-standard", "-z"),
    ("ls-tree", "-r", "--name-only", "-z", "HEAD"),
    ("ls-tree", "-r", "--name-only", "-z", "-c safe.directory=*"),
])
def test_public_runner_rejects_unreviewed_git_shapes_before_launch(observation_case, monkeypatch, args):
    def forbidden(*_, **__):
        raise AssertionError("unreviewed Git command launched")
    monkeypatch.setattr(local.subprocess, "check_output", forbidden)
    with pytest.raises(local.LocalPlanningError, match="fixed Git reads"):
        local._manifest_observation_git(observation_case[0], *args, public_readonly=True)


@pytest.mark.parametrize("route", [0, 1, None, "public"])
def test_public_route_requires_exact_boolean(observation_case, monkeypatch, route):
    monkeypatch.setattr(local.subprocess, "check_output", lambda *_, **__: pytest.fail("Git launched"))
    with pytest.raises(local.LocalPlanningError, match="exact public read route"):
        local._manifest_observation_git(observation_case[0], "ls-files", "-z", public_readonly=route)


@pytest.mark.parametrize("mutation", ["schema", "initial_alias", "relative_root", "symlink_root"])
def test_public_observation_requires_inventory_schema_and_exact_root(observation_case, tmp_path, mutation):
    root, manifest = observation_case
    manifest = deepcopy(manifest)
    initial = True
    if mutation == "schema":
        manifest["schema"] = local.INTENT_MANIFEST_SCHEMA
    elif mutation == "initial_alias":
        initial = 1
    elif mutation == "relative_root":
        root = Path(".")
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    with pytest.raises(local.LocalPlanningError, match="explicit schema and canonical root"):
        local.observe_public_inventory_manifest_sources(root, manifest, initial=initial)


@pytest.mark.parametrize("mutation", ["untracked", "ignored", "tracked", "immutable_source", "output_source", "symlink", "missing"])
def test_public_observation_preserves_all_source_membership_checks(observation_case, mutation):
    root, manifest = observation_case
    if mutation in {"untracked", "ignored", "tracked"}:
        (root / "foreign.py").write_text("FOREIGN = 1\n")
        if mutation == "ignored":
            (root / ".git/info/exclude").write_text("foreign.py\n")
        elif mutation == "tracked":
            _git(root, "add", "foreign.py")
    elif mutation in {"immutable_source", "output_source"}:
        (root / ("check.py" if mutation == "immutable_source" else "answer.py")).write_text("pass\n")
    elif mutation == "symlink":
        (root / "answer.py").unlink()
        (root / "answer.py").symlink_to(root / "check.py")
    else:
        (root / "answer.py").unlink()
    with pytest.raises(local.LocalPlanningError):
        local.observe_public_inventory_manifest_sources(root, manifest, initial=True)


def test_public_observation_keeps_declared_create_and_post_execution_rules(observation_case):
    root, manifest = observation_case
    manifest["created_outputs"] = ["report.json"]
    manifest["tasks"][0]["outputs"].append({"path": "report.json", "effect": "create"})
    assert local.observe_public_inventory_manifest_sources(root, manifest, initial=True) == manifest["sources"]
    (root / "answer.py").write_text("ANSWER = 2\n")
    (root / "report.json").write_text('{"answer": 2}\n')
    after = local.observe_public_inventory_manifest_sources(root, manifest)
    assert set(after) == set(manifest["sources"]) | {"report.json"}
    assert after == local.observe_local_manifest_sources(root, manifest)
    with pytest.raises(local.LocalPlanningError, match="absent before admission"):
        local.observe_public_inventory_manifest_sources(root, manifest, initial=True)


def _request(case, tmp_path):
    task = case["tasks"][0]
    source = (case["repository"] / "test_answer.py").read_bytes()
    selected = instruction.prepare_public_instruction_context(repository=case["repository"],
        admission=case["admission"], task_cid=task.task_cid, source_path="test_answer.py",
        expected_source_sha256=hashlib.sha256(source).hexdigest(), codebase_inventory_context=case["context"])
    workspace = tmp_path / "allocated-worker"
    _git(case["repository"], "worktree", "add", "--detach", "-q", str(workspace))
    return {"artifact": Path(selected["artifact"]), "expected_sha256": selected["sha256"],
        "task_cid": task.task_cid, "prompt": json.dumps({"objective_id": task.task_key}), "workspace": workspace}


def test_signed_300_member_router_uses_public_route_for_both_bound_roots(inventory_case, tmp_path, monkeypatch):
    case = inventory_case
    request = _request(case, tmp_path)
    actual = local.observe_public_inventory_manifest_sources
    roots = []
    def observed(root, manifest, **options):
        assert manifest["schema"] == local.INVENTORY_MANIFEST_SCHEMA and len(manifest["sources"]) == 300
        roots.append(root)
        return actual(root, manifest, **options)
    monkeypatch.setattr(local, "observe_public_inventory_manifest_sources", observed)
    monkeypatch.setattr(local, "_git", lambda *_, **__: pytest.fail("public load entered owner-only Git route"))
    block, receipt = instruction.load_public_instruction(**request)
    assert "assert answer() == 2" in block
    assert roots == [case["repository"], request["workspace"]]
    assert receipt["manifest_signature_verified"] is True
    assert receipt["source_freshness_verified"] is True
    assert receipt["completion_authority"] is False
    assert receipt["codebase_inventory"]["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])


@pytest.mark.parametrize("mutation", ["signature", "foreign_workspace"])
def test_router_rejects_unbound_signature_or_worktree_before_public_observation(inventory_case, tmp_path, monkeypatch, mutation):
    request = _request(inventory_case, tmp_path)
    if mutation == "signature":
        payload = json.loads(request["artifact"].read_bytes())
        payload["manifest"]["binding"]["signature"] = "A" * 86 + "=="
        payload["manifest_cid"] = instruction.content_identity(payload["manifest"])
        payload["context_cid"] = instruction.content_identity({k: v for k, v in payload.items() if k != "context_cid"})
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        path = request["artifact"].parent / (hashlib.sha256(raw).hexdigest() + ".json")
        path.write_bytes(raw)
        path.chmod(0o444)
        request.update(artifact=path, expected_sha256=hashlib.sha256(raw).hexdigest())
    else:
        request["workspace"] = inventory_case["repository"]
    monkeypatch.setattr(local, "observe_public_inventory_manifest_sources", lambda *_, **__: pytest.fail("unbound public observation"))
    with pytest.raises(ValueError):
        instruction.load_public_instruction(**request)
