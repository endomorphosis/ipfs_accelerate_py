"""Private Git/JSON fixtures only. No native DB, owner, services or imports.

Operator functions are compiled verbatim from the candidate AST. Import-only
dependencies of the canonical suffix checker are substituted explicitly; its
merge and task predicates run unchanged. Historical one-shot validators are
substituted only in the prequalification composition test below.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "scripts/run_agent_supervisor_efficiency_state_hardening.py"
CONTRACT = BASE / "ipfs_accelerate_py/agent_supervisor/runtime/source_repair_continuity.py"
spec = importlib.util.spec_from_file_location("private_source_repair", CONTRACT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def functions(names, namespace):
    nodes = [copy.deepcopy(FUNCTIONS[name]) for name in names]
    class RemoveImport(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return ast.Pass()
    # Stubs provide contract/checkout_repository_id; no production dependency
    # module or native runtime is imported by these private tests.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *[RemoveImport().visit(node) for node in nodes]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), namespace)


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    def raw(*args):
        return subprocess.check_output(["/usr/bin/git", "--no-optional-locks", "-C", str(repo), *args], env=env, stderr=subprocess.PIPE)
    def git(*args):
        return raw(*args).decode().strip()
    git("init", "-b", "main")
    git("config", "user.name", "private-fixture")
    git("config", "user.email", "private@example.invalid")
    def commit(path, text):
        (repo / path).write_text(text)
        git("add", "--", path)
        git("commit", "-qm", text)
        return git("rev-parse", "HEAD")
    anchor = commit("anchor.txt", "anchor")
    git("checkout", "-qb", "topic")
    commit("merged.txt", "merge candidate")
    git("checkout", "-q", "main")
    git("merge", "--no-ff", "-qm", "canonical merge", "topic")
    repair_base = git("rev-parse", "HEAD")
    middle = commit("repair.py", "first repair")
    target = commit("repair.py", "second repair")
    authority_value = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap-repair-historical-live-evidence-revision-closure-transition@1",
        "repair_head": anchor, "repair_tree": git("rev-parse", anchor + "^{tree}"),
        "receipt_cid": "sha256:" + "a" * 64, "authorization_attempt": {"attempt": "preserved"},
    }
    authority = write_json(tmp_path / "authority.json", authority_value)
    exit_ref = write_json(tmp_path / "historical-exit.json", {"scope": {
        "phase": "source_refused_archived_target", "source_result_sha256": None,
        "source_invoked": True, "source_replay_authority": False, "native_launch_authority": False,
        "task_or_queue_settlement_authority": False,
        "observation": {"source": {"repositories": {".": {"head": target}}},
                        "index_readmission": {"current_source": {"repositories": {".": {"head": repair_base}}}}},
    }})
    preflight = write_json(tmp_path / "historical-preflight.json", {"launch_admitted": False, "callback_settlement_authority": False, "returncode": 1})
    archive = write_json(tmp_path / "archive.json", {"files": [], "callback_settlement_authority": False})
    closed = write_json(tmp_path / "closed.json", {"normal_exit_code_matches_admitted_refusal": True, "exit_receipt_sha256": exit_ref["sha256"]})
    subject = {
        "schema": contract.ADMISSION_SCHEMA, "repository": str(repo), "authority": authority,
        "history": contract.history(git, raw, anchor=anchor, repair_base=repair_base, target=target),
        "historical_refusal": {"exit": exit_ref, "preflight": preflight, "archive": archive, "controller_exit": closed},
        "denied_authority": dict(contract.DENIED),
    }
    ns, _, _ = operator(SimpleNamespace(repo=repo, git=git, raw=raw, target=target, subject=subject))
    subject["canonical_prefix"] = ns["_admit_canonical_merge_suffix"](
        SimpleNamespace(protected_paths=[]), base_head=anchor, target_head=repair_base,
        bootstrap={"integrity": {"task_revisions": {}}},
        integrity={"task_statuses": {}, "task_revisions": {}, "task_cids": {}},
        task_outputs={}, completed_requests=[], admission_mode="sealed_line_descendant")
    provenance = write_json(tmp_path / "review.json", {"private_test_review": True})
    def publish(subject_value=None):
        value = copy.deepcopy(subject_value or subject)
        qualification = write_json(tmp_path / "qualification.json", {"schema": contract.QUALIFICATION_SCHEMA,
            "subject_sha256": contract.digest(value), "checks": {"private_fixture": True}, "provenance": [provenance]})
        value["qualification"] = qualification
        admission = write_json(tmp_path / "admission.json", value)
        monkeypatch.setenv(contract.PATH_ENV, admission["path"])
        monkeypatch.setenv(contract.SHA_ENV, admission["sha256"])
        return value, admission
    value, pin = publish()
    return SimpleNamespace(repo=repo, git=git, raw=raw, anchor=anchor, repair_base=repair_base,
                           middle=middle, target=target, subject=subject, value=value, pin=pin,
                           publish=publish, tmp=tmp_path, authority=authority_value)


def load(f):
    return contract.registration(os.environ, repository=f.repo, git=f.git, git_bytes=f.raw)


def operator(f):
    h = f.subject["history"]
    witness = {"head": f.target, "tree": h["target_tree"], "status_digest": "clean", "epoch": "current"}
    guard = SimpleNamespace(candidate_head=f.target, candidate_tree=h["target_tree"], record={"pid": os.getpid(), "epoch": "held"})
    def guard_check(row, **kwargs):
        if row is not ns["_ASEH_CANDIDATE_GIT_GUARD"] or row is not guard:
            raise ValueError("guard replaced")
    def witness_check(row, **kwargs):
        if row != witness:
            raise ValueError("witness changed")
    ns = {"__builtins__": __builtins__, "Any": Any, "Mapping": Mapping, "Sequence": Sequence,
          "Path": Path, "os": os, "json": json, "re": re, "contract": contract,
          "ROOT": f.repo, "OperatorError": contract.SourceRepairDenied,
          "_git": f.git, "_git_bytes": f.raw, "_canonical_json": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")),
          "_identity": lambda v: "sha256:" + contract.digest(v),
          "_ASEH_CANDIDATE_GIT_GUARD": guard,
          "_ASEH_RETAINED_INDEX_OBSERVATION": None,
          "_validate_candidate_git_guard_health": guard_check,
          "_validate_retained_index_observation": lambda row, **kwargs: dict(row.record),
          "_candidate_authorization_witness": lambda **kwargs: dict(witness),
          "_assert_candidate_authorization_witness": witness_check,
          "checkout_repository_id": lambda root: "private-repository",
          "_git_changed_paths": lambda base, target: tuple(r["path"] for r in contract.raw_delta(f.raw, base, target)),
          "ASEH_SEALED_LINE_DESCENDANT_TASK_ALIAS": "ASEH-SEALED-LINE"}
    names = ["_validated_r45_source_repair_environment", "_r45_source_repair_receipt_identity",
             "_r45_source_repair_registration", "_r45_active_git_custody_record",
             "_validate_r45_source_repair_continuity", "_r45_descendant_parent_shape",
             "_admit_r45_or_canonical_descendant", "_validate_r29_historical_live_effect_continuity",
             "_admit_canonical_merge_suffix", "_sealed_owner_delegation_environment", "preflight"]
    functions(names, ns)
    return ns, witness, guard


def options(f):
    return {"base_head": f.anchor, "target_head": f.target,
            "bootstrap": {"integrity": {"task_revisions": {}}},
            "integrity": {"task_statuses": {}, "task_revisions": {}, "task_cids": {}},
            "task_outputs": {}, "completed_requests": [], "admission_mode": "sealed_line_descendant"}


def admitted(f, ns, witness):
    bundle = {"source_only_repair_selected": True, "transition": f.authority,
              "receipt": f.authority, "active_source_tree": witness["tree"], "active_source_witness": witness}
    return ns["_admit_r45_or_canonical_descendant"](SimpleNamespace(protected_paths=[]), source_repair_bundle=bundle, **options(f))


def test_registered_linear_suffix_uses_actual_canonical_prefix(fixture):
    f = fixture
    assert load(f) == f.value
    ns, witness, guard = operator(f)
    proof = admitted(f, ns, witness)
    assert proof["schema"] == contract.CONTINUITY_SCHEMA
    assert len(proof["canonical_prefix"]["integrations"]) == 1
    assert proof["canonical_prefix"]["target_head"] == f.repair_base
    assert "integrations" not in proof and "task_alias" not in proof and "request_id" not in proof
    assert ns["_r45_descendant_parent_shape"](parents=[f.middle], continuity={"repair_to_current": proof},
        transition=f.authority, candidate_head=f.target, candidate_tree=witness["tree"])


def test_operator_normalized_historical_receipt_still_admits(fixture):
    f = fixture; ns, witness, _ = operator(f)
    normalized = copy.deepcopy(f.authority)
    normalized["candidate_authorization_witness"] = {
        **dict(normalized.get("candidate_authorization_witness") or {}),
        "normalized_by_validator": True,
    }
    bundle = {"source_only_repair_selected": True, "transition": f.authority,
              "receipt": normalized, "active_source_tree": witness["tree"],
              "active_source_witness": witness}
    proof = ns["_admit_r45_or_canonical_descendant"](
        SimpleNamespace(protected_paths=[]), source_repair_bundle=bundle, **options(f))
    assert proof["schema"] == contract.CONTINUITY_SCHEMA
    rewritten = copy.deepcopy(normalized)
    rewritten["receipt_cid"] = "sha256:" + "b" * 64
    bundle["receipt"] = rewritten
    with pytest.raises(ValueError, match="authority/target differs"):
        ns["_admit_r45_or_canonical_descendant"](
            SimpleNamespace(protected_paths=[]), source_repair_bundle=bundle, **options(f))


@pytest.mark.parametrize("mutation", ["missing_pin", "missing_path", "unregistered", "changed_bytes", "qualification_deleted", "qualification_changed", "authority_changed"])
def test_explicit_registration_and_immutable_artifacts_required(fixture, monkeypatch, mutation):
    f = fixture
    if mutation == "missing_pin": monkeypatch.delenv(contract.SHA_ENV)
    elif mutation == "missing_path": monkeypatch.delenv(contract.PATH_ENV)
    elif mutation == "unregistered":
        monkeypatch.delenv(contract.SHA_ENV); monkeypatch.delenv(contract.PATH_ENV)
    elif mutation == "changed_bytes": Path(f.pin["path"]).write_text("{}")
    elif mutation == "qualification_deleted": (f.tmp / "qualification.json").unlink()
    elif mutation == "qualification_changed": (f.tmp / "qualification.json").write_text("{}")
    elif mutation == "authority_changed": (f.tmp / "authority.json").write_text("{}")
    ns, witness, _ = operator(f)
    with pytest.raises((ValueError, OSError)):
        admitted(f, ns, witness)


@pytest.mark.parametrize("mutation", ["failed_check", "empty_checks", "wrong_subject", "missing_provenance", "extra_field"])
def test_repinned_malformed_qualification_cannot_become_authority(fixture, monkeypatch, mutation):
    f = fixture
    path = f.tmp / "qualification.json"
    value = json.loads(path.read_text())
    if mutation == "failed_check": value["checks"]["private_fixture"] = False
    elif mutation == "empty_checks": value["checks"] = {}
    elif mutation == "wrong_subject": value["subject_sha256"] = "0" * 64
    elif mutation == "missing_provenance": value["provenance"] = []
    else: value["launch_authorized"] = True
    registration = copy.deepcopy(f.value)
    registration["qualification"] = write_json(path, value)
    ref = write_json(Path(f.pin["path"]), registration)
    monkeypatch.setenv(contract.SHA_ENV, ref["sha256"])
    with pytest.raises(ValueError, match="qualification"): load(f)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_pinned_json_rejects_ambiguous_shapes(tmp_path, raw):
    p = tmp_path / "ambiguous.json"; p.write_bytes(raw); p.chmod(0o600)
    with pytest.raises(ValueError):
        contract.pinned_json({"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()})


@pytest.mark.parametrize("mutation", ["missing_commit", "commit_parent", "tree", "blob", "mode", "path", "anchor_tree", "task_authority", "invented_request", "foreign_repo"])
def test_even_repinned_forged_history_or_scope_refused(fixture, mutation):
    f = fixture
    value = copy.deepcopy(f.subject)
    if mutation == "missing_commit": del value["history"]["commits"][1]
    elif mutation == "commit_parent": value["history"]["commits"][1]["parents"] = [f.anchor]
    elif mutation == "tree": value["history"]["commits"][1]["tree"] = "1" * 40
    elif mutation in {"blob", "mode", "path"}:
        key = {"blob": "new_blob", "mode": "new_mode", "path": "path"}[mutation]
        value["history"]["commits"][1]["delta"][0][key] = {"blob": "2" * 40, "mode": "100755", "path": "other.py"}[mutation]
    elif mutation == "anchor_tree": value["history"]["anchor_tree"] = "3" * 40
    elif mutation == "task_authority": value["denied_authority"]["task_completion"] = True
    elif mutation == "invented_request": value["request_id"] = "fabricated"
    elif mutation == "foreign_repo": value["repository"] = str(f.tmp)
    f.publish(value)
    with pytest.raises(ValueError): load(f)


@pytest.mark.parametrize("mutation", ["witness", "guard", "pin", "prefix", "prefix_metadata", "nested_source_prefix", "digest", "extra"])
def test_runtime_proof_cannot_be_forged_or_rebound(fixture, mutation):
    f = fixture; ns, witness, guard = operator(f); proof = admitted(f, ns, witness)
    if mutation == "witness": proof["active_witness"]["epoch"] = "stale"
    elif mutation == "guard": proof["active_guard_sha256"] = "0" * 64
    elif mutation == "pin": proof["registration_sha256"] = "0" * 64
    elif mutation == "prefix": proof["canonical_prefix"]["integrations"] = []
    elif mutation == "prefix_metadata":
        proof["canonical_prefix"]["integrations"][0]["request_id"] = "forged-prefix-request"
        prefix = proof["canonical_prefix"]; prefix.pop("receipt_cid", None)
        prefix["receipt_cid"] = "sha256:" + contract.digest(prefix)
    elif mutation == "nested_source_prefix": proof["canonical_prefix"]["schema"] = contract.CONTINUITY_SCHEMA
    elif mutation == "extra": proof["source_replay"] = True
    else: proof["receipt_cid"] = "sha256:" + "0" * 64
    if mutation != "digest":
        proof.pop("receipt_cid", None); proof["receipt_cid"] = "sha256:" + contract.digest(proof)
    with pytest.raises(ValueError):
        ns["_validate_r29_historical_live_effect_continuity"](proof,
            authorization_candidate_head=f.anchor, authorization_candidate_tree=f.subject["history"]["anchor_tree"],
            active_candidate_head=f.target, active_candidate_tree=witness["tree"])


def test_canonical_completion_still_refuses_linear_even_with_registration(fixture):
    f = fixture; ns, witness, _ = operator(f)
    kwargs = options(f); kwargs["admission_mode"] = "canonical_completion"; kwargs["base_head"] = f.repair_base
    with pytest.raises(ValueError, match="non-canonical integration"):
        ns["_admit_r45_or_canonical_descendant"](SimpleNamespace(protected_paths=[]), source_repair_bundle=None, **kwargs)


def test_source_repair_skips_disposable_event_replay_when_registered(fixture):
    f = fixture; ns, _, _ = operator(f)
    functions(["_source_repair_launch_skips_disposable_event_replay"], ns)
    ns["_launch_uses_observational_index_custody"] = lambda: True
    assert ns["_source_repair_launch_skips_disposable_event_replay"]() is True
    ns["_launch_uses_observational_index_custody"] = lambda: False
    assert ns["_source_repair_launch_skips_disposable_event_replay"]() is False


def test_observational_index_custody_binds_without_write_guard(fixture):
    f = fixture; ns, witness, _ = operator(f)
    ns["_ASEH_CANDIDATE_GIT_GUARD"] = None
    observation = SimpleNamespace(
        candidate_head=f.target, candidate_tree=witness["tree"],
        record={"observational": True, "epoch": "retained"},
    )
    ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = observation
    proof = admitted(f, ns, witness)
    assert proof["schema"] == contract.CONTINUITY_SCHEMA
    assert proof["active_guard_sha256"] == contract.digest(observation.record)
    ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = None
    with pytest.raises(ValueError, match="Git custody is absent"):
        admitted(f, ns, witness)


def test_preflight_retains_same_guard_through_late_exact_check(fixture):
    f = fixture; ns, witness, guard = operator(f)
    ns["_ASEH_CANDIDATE_GIT_GUARD"] = None
    ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = None
    seen = []
    observation = SimpleNamespace(
        candidate_head=f.target, candidate_tree=witness["tree"],
        record={"observational": True},
    )
    @contextmanager
    def held(**kwargs):
        assert ns["_ASEH_CANDIDATE_GIT_GUARD"] is None
        assert ns["_ASEH_RETAINED_INDEX_OBSERVATION"] is None
        ns["_ASEH_CANDIDATE_GIT_GUARD"] = guard
        ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = observation
        try: yield observation
        finally:
            ns["_ASEH_CANDIDATE_GIT_GUARD"] = None
            ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = None
            seen.append("released")
    def body(config):
        proof = admitted(f, ns, witness)
        assert ns["_r45_descendant_parent_shape"](parents=[f.middle], continuity={"repair_to_current": proof},
            transition=f.authority, candidate_head=f.target, candidate_tree=witness["tree"])
        seen.append("late_exact")
        return 0, {"private": True}
    ns["_prepared_retained_index_observation"] = held
    ns["_preflight_with_current_source"] = body
    assert ns["preflight"](f.tmp / "private-config.json")[0] == 0
    assert seen == ["late_exact", "released"]


def test_positive_owner_environment_retains_only_verified_registration(fixture, monkeypatch):
    f = fixture; ns, _, _ = operator(f)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    projected = ns["_sealed_owner_delegation_environment"](f.tmp)
    assert projected[contract.PATH_ENV] == f.pin["path"]
    assert projected[contract.SHA_ENV] == f.pin["sha256"]
    assert "UNRELATED_SECRET" not in projected and "PYTHONPATH" not in projected
    scheduler = BASE / "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py"
    tree = ast.parse(scheduler.read_text())
    names = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    sealed_owner = ast.get_source_segment(scheduler.read_text(), names["_run_aseh_sealed_owner"])
    assert contract.PATH_ENV in sealed_owner and contract.SHA_ENV in sealed_owner
    exact = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(f.tmp),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME": str(f.tmp),
        "XDG_CACHE_HOME": str(f.tmp / ".cache" / "xdg"),
        "CUDA_CACHE_PATH": str(f.tmp / ".cache" / "cuda"),
        "CUDA_CACHE_DISABLE": "1",
        contract.PATH_ENV: projected[contract.PATH_ENV],
        contract.SHA_ENV: projected[contract.SHA_ENV],
    }
    assert exact == projected
    # A real clean child receives the exact positive projection. Its independent
    # contract read uses the same retained Git source/artifacts, no native import.
    code = '''import importlib.util,json,os,pathlib,subprocess,sys
spec=importlib.util.spec_from_file_location("c",sys.argv[1]);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
repo=pathlib.Path(sys.argv[2])
def raw(*a): return subprocess.check_output(["/usr/bin/git","--no-optional-locks","-C",str(repo),*a],env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
def git(*a): return raw(*a).decode().strip()
r=c.registration(os.environ,repository=repo,git=git,git_bytes=raw)
print(json.dumps({"target":r["history"]["target_head"],"pin":os.environ[c.SHA_ENV]}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(CONTRACT), str(f.repo)], env=projected, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"target": f.target, "pin": f.pin["sha256"]}
    (f.tmp / "qualification.json").write_text("{}")
    with pytest.raises(ValueError): ns["_sealed_owner_delegation_environment"](f.tmp)




def test_later_canonical_task_tail_requires_real_task_queue_bindings(fixture):
    f = fixture; ns, witness, guard = operator(f)
    repair_target = f.target
    f.git("checkout", "-qb", "completed-task")
    (f.repo / "task.txt").write_text("validated task output")
    f.git("add", "task.txt"); f.git("commit", "-qm", "task candidate")
    candidate = f.git("rev-parse", "HEAD"); candidate_tree = f.git("rev-parse", "HEAD^{tree}")
    f.git("checkout", "-q", "main"); f.git("merge", "--no-ff", "-qm", "complete task", "completed-task")
    f.target = f.git("rev-parse", "HEAD")
    witness.update(head=f.target, tree=f.git("rev-parse", "HEAD^{tree}"))
    guard.candidate_head = witness["head"]; guard.candidate_tree = witness["tree"]
    ns["COMPLETED_STATUSES"] = {"completed"}
    ns["_git_tree_entry"] = lambda commit, path: f.git("ls-tree", commit, "--", path)
    def integrated(root, *, candidate_commit, target_commit, changed_submodule_paths):
        # Private repository has no submodules. Real graph/tree checks occur in
        # the unchanged checker; this imported handoff seam is substituted.
        assert changed_submodule_paths == []
        f.git("merge-base", "--is-ancestor", candidate_commit, target_commit)
        return {"passed": True}
    ns["integrated_candidate_handoff_proof"] = integrated
    bundle = {"source_only_repair_selected": True, "transition": f.authority,
              "receipt": f.authority, "active_source_tree": witness["tree"], "active_source_witness": witness}
    kwargs = options(f)
    board = SimpleNamespace(protected_paths=[], merge_target_branch="main")
    with pytest.raises(ValueError, match="completed queue request"):
        ns["_admit_r45_or_canonical_descendant"](board, source_repair_bundle=bundle, **kwargs)
    alias, cid = "ASEH-PRIVATE-TASK", "private-task-cid"
    metadata = {"schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
                "target_binding_schema": "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1",
                "target_repository_id": "private-repository", "target_branch": "main",
                "completion_task_cids": {alias: cid},
                "validation_proof": {"passed": True, "target_commit": candidate, "target_tree": candidate_tree},
                "candidate_tree": candidate_tree, "repository_tree_id": "git-tree:" + candidate_tree,
                "task": {"outputs": ["task.txt"]}, "baseline_ref": repair_target, "changed_submodule_paths": []}
    request = SimpleNamespace(commit_sha=candidate, request_id="private-completed-request", task_id=alias,
                              canonical_task_id=cid, canonical_task_key=cid, status="completed", metadata=metadata)
    kwargs.update(integrity={"task_statuses": {alias: "completed"}, "task_revisions": {alias: 1}, "task_cids": {alias: cid}},
                  task_outputs={alias: ["task.txt"]}, completed_requests=[request])
    proof = ns["_admit_r45_or_canonical_descendant"](board, source_repair_bundle=bundle, **kwargs)
    assert proof["canonical_tail"]["base_head"] == repair_target
    assert proof["canonical_tail"]["integrations"][0]["request_id"] == request.request_id
    assert proof["registration_sha256"] == f.pin["sha256"]
    assert ns["_validated_r45_source_repair_environment"]()[contract.SHA_ENV] == f.pin["sha256"]
    metadata["validation_proof"]["passed"] = False
    with pytest.raises(ValueError, match="candidate validation"):
        ns["_admit_r45_or_canonical_descendant"](board, source_repair_bundle=bundle, **kwargs)


def test_later_linear_descendant_never_inherits_registration(fixture):
    f = fixture; ns, witness, guard = operator(f)
    (f.repo / "unreviewed.py").write_text("unreviewed")
    f.git("add", "unreviewed.py"); f.git("commit", "-qm", "unreviewed linear advance")
    with pytest.raises(ValueError, match="unregistered linear"):
        ns["_validated_r45_source_repair_environment"]()


def test_prequalify_then_deferred_completion_preserves_historical_tuple(fixture):
    f = fixture; ns, witness, guard = operator(f)
    functions(["_prequalify_r45_historical_live_launch", "_complete_r45_historical_live_prequalification",
               "_r45_launch_historical_live_evidence"], ns)
    required = next(ast.literal_eval(n.value) for n in FUNCTIONS["_complete_r45_historical_live_prequalification"].body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in n.targets))
    authority = copy.deepcopy(f.authority)
    authority.update({name: {"historical": name} for name in required})
    authority.update(candidate_authorization_witness={"historical": "witness"},
                     durable_candidate_witness={"witness_cid": "historical-durable"},
                     candidate_git_guard={"historical": "guard"},
                     historical_live_policy_admission={"policy_admission_cid": "preserved-policy"},
                     historical_live_execution_evidence={"active_policy_cid": "preserved-policy", "authorizing_receipt_cid": None})
    authority_ref = write_json(f.tmp / "authority.json", authority)
    subject = copy.deepcopy(f.subject); subject["authority"] = authority_ref
    f.value, f.pin = f.publish(subject); f.authority = authority
    expected_names = ["r39_validation_dependency_failure_evidence", "approved_validation_runtime_deployment",
        "r40_authorization_attempt", "r40_authorization_failure_evidence", "validation_dependency_directories_contract",
        "r41_bootstrap_failure_evidence", "native_dependency_bootstrap_dispatch_contract", "r42_authorization_attempt",
        "r42_authorization_timeout_failure_evidence", "validation_timeout_policy_contract", "r43_authorization_attempt",
        "r43_validation_failure_evidence", "validation_runtime_binding_contract", "r44_authorization_attempt",
        "r44_historical_live_evidence_admission_failure_evidence"]
    seen = []
    def one_shot(**kwargs):
        if kwargs.get("expected_r45_receipt") != authority:
            raise ValueError("historical receipt changed")
        return tuple(authority[name] for name in expected_names)
    ns.update(_r45_population_requires_policy=lambda value: False,
              _r29_receipt_name_is_absent=lambda path: False,
              _load_exact_r39_receipt_chain=lambda paths: [{"old": "r39"}],
              _secure_runtime_json=lambda path, **kwargs: copy.deepcopy(authority),
              _repair_historical_live_evidence_revision_closure_transition_receipt_id=lambda row: authority["receipt_cid"],
              _validate_repair_historical_live_evidence_revision_closure_transition=lambda row, **kwargs: copy.deepcopy(row),
              _validate_r45_authorization_attempt_record=lambda row, **kwargs: row,
              _assert_r45_one_shot_state=one_shot,
              _bootstrap_receipt_id=lambda value: "bootstrap",
              _admit_r45_historical_live_policy_admission=lambda row, **kwargs: row,
              _r27_historical_live_validation_executor_contract=lambda: {},
              _validate_r45_receipt_historical_live_execution_evidence=lambda row, **kwargs: row,
              _admit_exact_r45_transition_chain=lambda rows: rows,
              STATUS_RECEIPT_MAX_BYTES=16*1024*1024)
    for name in ("ASEH_R31_PUBLISHED_R30_RECEIPT_CID", "ASEH_R37_PUBLISHED_R36_RECEIPT_CID",
                 "ASEH_R39_PUBLISHED_R38_RECEIPT_CID", "ASEH_R40_PUBLISHED_R39_RECEIPT_CID"):
        ns[name] = "historical-" + name
    def live_boundary(**kwargs):
        assert kwargs["authorization_receipt"] == authority
        assert kwargs["authorization_attempt"] == authority["authorization_attempt"]
        assert kwargs["continuity_admission"]["schema"] == contract.CONTINUITY_SCHEMA
        seen.append("current_effect_custody_check_substitute")
    ns["_assert_r45_historical_live_effect_admission"] = live_boundary
    paths = {"repair_historical_live_evidence_revision_closure_transition_receipt": f.tmp / "authority.json"}
    bundle = ns["_prequalify_r45_historical_live_launch"](paths=paths,
        population={"source_head": f.target, "repository_tree_id": witness["tree"]}, bootstrap={})
    assert bundle["source_only_repair_selected"] is True and bundle["historical_live_deferred"] is True
    assert not seen
    proof = ns["_admit_r45_or_canonical_descendant"](SimpleNamespace(protected_paths=[]), source_repair_bundle=bundle, **options(f))
    completed = ns["_complete_r45_historical_live_prequalification"](paths=paths, bundle=bundle, continuity_admission=proof)
    assert completed["historical_live_deferred"] is False
    assert completed["receipt"] == authority and completed["policy_admission"] == authority["historical_live_policy_admission"]
    assert len(seen) == 1
    assert (f.tmp / "authority.json").read_bytes() == Path(authority_ref["path"]).read_bytes()


def literal_constants(namespace):
    """Evaluate only literal/name/container/addition constants, never imports/calls."""
    allowed = (ast.Constant, ast.Name, ast.Load, ast.Tuple, ast.List, ast.Dict, ast.Set,
               ast.BinOp, ast.Add, ast.Subscript, ast.Slice, ast.UnaryOp, ast.USub, ast.Starred)
    assignments = [ast.Assign(targets=[n.target], value=n.value) if isinstance(n, ast.AnnAssign) else n for n in TREE.body]
    pending = [n for n in assignments if isinstance(n, ast.Assign)
               and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)
               and all(isinstance(v, allowed) for v in ast.walk(n.value))]
    for _ in range(len(pending)):
        remaining = []
        for node in pending:
            try:
                value = eval(compile(ast.Expression(node.value), "<literal-source-constants>", "eval"),
                             {"__builtins__": {}}, namespace)
            except (NameError, TypeError, KeyError):
                remaining.append(node); continue
            for target in node.targets: namespace[target.id] = value
        if len(remaining) == len(pending): break
        pending = remaining


def test_actual_preflight_exact_run_and_owner_start_route_composition(fixture):
    """Actual wrappers/body/exact-run/owner projection, private control seams.

    Substitutions: historical chain/receipt validation, materialized native DB
    read, interpreter/native seal branch (None), Git guard acquisition (private
    object). No production DB is opened and no owner process is launched.
    New registration/history/proof validators and both exact-run calls are real.
    """
    f = fixture; ns, witness, guard = operator(f); literal_constants(ns)
    # Restore private fake guard after literal initialization resets globals.
    ns["_ASEH_CANDIDATE_GIT_GUARD"] = guard
    functions(["_assert_exact_run_launch_admission", "_r45_inherited_owner_start_permission_context",
               "_r23_owner_start_permission_context_from_launch_admission", "_preflight_with_current_source"], ns)
    authority = copy.deepcopy(f.authority)
    authority.update({
        "base_head": ns["REPAIR_HISTORICAL_LIVE_EVIDENCE_REVISION_CLOSURE_TRANSITION_BASE_HEAD"],
        "base_tree": ns["REPAIR_HISTORICAL_LIVE_EVIDENCE_REVISION_CLOSURE_TRANSITION_BASE_TREE"],
        "failed_unpublished_r44_head": ns["REPAIR_HISTORICAL_LIVE_EVIDENCE_REVISION_CLOSURE_TRANSITION_BASE_HEAD"],
        "failed_unpublished_r44_tree": ns["REPAIR_HISTORICAL_LIVE_EVIDENCE_REVISION_CLOSURE_TRANSITION_BASE_TREE"],
        "previous_receipt_cid": ns["ASEH_R40_PUBLISHED_R39_RECEIPT_CID"],
        "failed_r44_authorization_attempt_cid": ns["ASEH_R45_FAILED_R44_AUTHORIZATION_ATTEMPT_CID"],
        "r44_historical_live_evidence_admission_failure_evidence_cid": "sha256:" + "c" * 64,
    })
    ref = write_json(f.tmp / "authority.json", authority)
    subject = copy.deepcopy(f.subject); subject["authority"] = ref
    f.value, f.pin = f.publish(subject); f.authority = authority
    r39 = {"receipt_cid": ns["ASEH_R40_PUBLISHED_R39_RECEIPT_CID"]}
    def chain(value):
        if value != [r39, authority]: raise ValueError("historical chain changed")
        return value
    ns.update(_admit_exact_r45_transition_chain=chain,
              _r45_transition_retains_inherited_evidence=lambda value: value == authority,
              _r45_expected_r44_authorization_failure_evidence=lambda: {"evidence_cid": "sha256:" + "c" * 64})
    captured = []
    def materialized(*args):
        proof = admitted(f, ns, witness)
        result = {"runtime_source_head": f.target, "runtime_repository_tree_id": witness["tree"],
            "repair_transition": authority, "repair_transition_chain": [r39, authority],
            "canonical_continuity": {"repair_to_current": proof,
                "published_r39_to_historical_live_evidence_revision_closure": authority},
            "historical_live_authorizing_receipt_cid": authority["receipt_cid"], "projection_matches_events": True,
            "bootstrap_receipt_id": "sha256:" + "1" * 64}
        for key in ("r40_receipt_absent", "r41_receipt_absent", "r42_receipt_absent", "r43_receipt_absent",
                    "r43_attempt_present", "r44_receipt_absent", "r44_attempt_present"):
            result[key] = True
        result["admission_cid"] = ns["_identity"](result)
        captured.append(result)
        return result
    @contextmanager
    def bound(**kwargs): yield
    @contextmanager
    def held(**kwargs):
        ns["_ASEH_CANDIDATE_GIT_GUARD"] = guard
        try: yield guard
        finally: ns["_ASEH_CANDIDATE_GIT_GUARD"] = None
    database = f.tmp / "private-non-database-marker"; database.write_text("existence only")
    board = SimpleNamespace(resolved_database_program=lambda: SimpleNamespace(store_id="data/aseh/control.duckdb"))
    ns.update(_load=lambda path: (board, {}), preflight_configured_board=lambda board: {"valid": True},
              _paths=lambda board: {"bootstrap_receipt": f.tmp / "authority.json", "database": database},
              _bounded_launch_admission=bound, _prepared_candidate_git_guard=held,
              _r14_native_seal_anchor=lambda *args, **kwargs: None,
              _admit_materialized_launch=materialized)
    ns["_prepared_retained_index_observation"] = held
    ns["_ASEH_CANDIDATE_GIT_GUARD"] = None
    ns["_ASEH_RETAINED_INDEX_OBSERVATION"] = None
    code, report = ns["preflight"](f.tmp / "config.json")
    assert code == 0, report
    assert report["sealed_launch_admission"]["admitted"] is True
    assert ns["_ASEH_CANDIDATE_GIT_GUARD"] is None
    # Resume/owner entry independently builds its admission inside its own live
    # guard; it cannot transplant the previous preflight guard epoch.
    guard.record = {"pid": os.getpid(), "epoch": "fresh-owner-admission"}
    ns["_ASEH_CANDIDATE_GIT_GUARD"] = guard
    owner_admission = materialized()
    old_witness = {"head": f.anchor, "tree": f.subject["history"]["anchor_tree"]}
    r30 = {"repair_head": f.anchor, "repair_tree": old_witness["tree"],
           "candidate_authorization_witness": old_witness,
           "durable_candidate_witness": {"head": f.anchor, "tree": old_witness["tree"],
                "authorization_v1_witness_cid": ns["_identity"](old_witness)}}
    r23 = {"schema": ns["REPAIR_SEALED_OWNER_DATABASE_PERMISSION_HARDENING_TRANSITION_SCHEMA"],
           "receipt_cid": ns["ASEH_R24_EXACT_R1_R23_RECEIPT_CIDS"][-1]}
    ns.update(_admit_exact_r30_transition_chain=lambda value: [r23, {}, {}, {}, {}, r30],
              _validate_r30_durable_candidate_witness=lambda row: row,
              _validate_r23_owner_start_permission_context=lambda row, **kwargs: row)
    context = ns["_r23_owner_start_permission_context_from_launch_admission"](
        board=board, launch_admission=owner_admission, candidate_head=f.target,
        candidate_tree=witness["tree"], candidate_authorization_witness=witness)
    assert context["materialized_launch_admission_cid"] == owner_admission["admission_cid"]
    assert captured[0]["canonical_continuity"]["repair_to_current"]["registration_sha256"] == (
        owner_admission["canonical_continuity"]["repair_to_current"]["registration_sha256"])
    with pytest.raises(ValueError, match="custody changed"):
        ns["_assert_exact_run_launch_admission"](captured[0], candidate_head=f.target, candidate_tree=witness["tree"])
    owner_admission["repair_transition"]["failed_r44_authorization_attempt_cid"] = "forged"
    with pytest.raises(ValueError):
        ns["_assert_exact_run_launch_admission"](owner_admission, candidate_head=f.target, candidate_tree=witness["tree"])
