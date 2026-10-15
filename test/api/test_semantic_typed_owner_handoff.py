"""Real owner publication with an explicitly authored typed-graph builder.

The external owner file, immutable publishers, hashes, locks and compact
receipts are real. Native graph/source verification is substituted and grants
no authority; separate native qualification covers that integration.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner as owner
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy as typed_policy
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_nomination_lineage as lineage_api
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy import TypedOwnerNomination

TASK = "TASK-AUTHORED-TYPED-HANDOFF"
ARTIFACT = ".runtime/semantic/worker-context.json"
SOURCE = b"def helper(value):\n    return value + 1\n\ndef target(value):\n    return helper(value)\n"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(raw):
    return sha256(raw).hexdigest()


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    # This retained historical seam deliberately uses the authentic legacy
    # source proposal contract. Scope replay must stay exact even after the
    # owner default changes to task_closure.
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", "0")
    repository, workspace = tmp_path / "source", tmp_path / "candidate"
    repository.mkdir(mode=0o755)
    workspace.mkdir(mode=0o755)
    source_path = repository / "operations.py"
    source_path.write_bytes(SOURCE)
    parent_path = repository / ARTIFACT
    parent_path.parent.mkdir(parents=True, mode=0o755)
    parent_path.parent.chmod(0o755)
    parent_path.parent.parent.chmod(0o755)
    parent = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": TASK,
        "objective": "Preserve the public target and helper behavior.",
        "semantic_root_cid": "authored-semantic-root", "scope_cid": "authored-source-scope",
        "manifest": {"operations.py": {"sha256": _sha(SOURCE), "source_cid": "authored-source-cid"}},
        "capsules": [{"stable_symbol_id": "symbol-target", "capsule_cid": "capsule-target"},
                     {"stable_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}],
        "admissions": [
            {"ref": {"stable_symbol_id": "symbol-target", "capsule_cid": "capsule-target"}},
            {"ref": {"stable_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}},
        ],
        "required_raw_paths": ["operations.py"], "raw_sources": {"operations.py": SOURCE.decode()},
        "preparation_bounds": {"max_capsules": 2, "max_bytes": 8192},
        "pack": {"schema": "ipfs-accelerate.context-pack-result@1", "interface": "ContextPack@1",
                 "pack": {}},
        "doctor_snapshot_id": "authored-doctor-snapshot",
        "doctor_manifest_cid": "authored-doctor-manifest",
        "reconstruction": {"semantic_equivalence_claimed": False, "completion_authority": False},
        "refresh_lineage": None,
        "completion_authority": False,
    }
    parent_path.write_bytes(_encoded(parent))
    policy = SemanticDecisionPolicy(required_capsule_ids=("capsule-target",),
                                    optional_capsule_ids=("capsule-helper",))
    selected = SemanticDecisionPolicy(required_capsule_ids=policy.required_capsule_ids,
        optional_capsule_ids=policy.optional_capsule_ids, selected_optional_capsule_ids=("capsule-helper",))
    nomination_path = tmp_path / "owner declaration café.json"
    # Whitespace and UTF-8 spelling must be retained exactly in lineage, while
    # the standard generated nomination uses the publisher's canonical bytes.
    origin_text = json.dumps(owner.build_decision_nomination(policy), ensure_ascii=False, indent=2) + "\n"
    nomination_path.write_text(origin_text)
    nomination_path.chmod(0o444)
    wire = {"objective_id": TASK, "repository_id": "authored-repository", "tree_id": "authored-tree"}
    prompt = _encoded(wire) + "\nPreserve literal public requirement café.  \n".encode()
    state = SimpleNamespace(repository=repository, workspace=workspace, source_path=source_path,
        parent_path=parent_path, parent=parent, parent_text=parent_path.read_text(),
        nomination_path=nomination_path, origin_text=origin_text, policy=policy, selected=selected,
        prompt=prompt, typed_calls=[], launch_calls=[], verify_calls=[], diagnostic_bytes=0,
        typed_error=None, after_typed=None, replay_calls=[], latest_source_proposal=None)

    def parts(text):
        parsed, end = json.JSONDecoder().raw_decode(text)
        return parsed, text[end:], (), (), state.parent_text, ARTIFACT

    def verified(root, *, artifact, text, task_id, current):
        state.verify_calls.append((root, artifact, task_id, current))
        assert current is True
        raw = owner.codec._read(root, artifact)
        if raw != text.encode() or task_id != TASK:
            raise ValueError("authored current verifier: parent/task changed")
        if owner.codec._read(root, "operations.py") != SOURCE:
            raise ValueError("authored current verifier: source changed")
        return json.loads(raw), ("symbol-helper", "symbol-target")

    def launch(*, native_prompt, policy, repository):
        state.launch_calls.append((native_prompt, policy, repository))
        return {"schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
                "task_id": TASK, "parent_artifact": ARTIFACT,
                "parent_sha256": _sha(state.parent_text.encode()), "policy": policy.to_dict()}

    def typed(**kwargs):
        state.typed_calls.append(kwargs)
        if state.typed_error is not None:
            raise state.typed_error
        original = owner.load_decision_nomination(
            artifact=kwargs["nomination_artifact"], expected_sha256=kwargs["nomination_sha256"],
            workspace=state.workspace)
        assert original == state.policy
        descriptor = launch(native_prompt=kwargs["native_prompt"], policy=state.selected,
                            repository=state.repository)
        proposal = {
            "schema": "supervisor-source-bound-typed-context-proposal@1",
            "adapter_version": "native-typed-dependency-proposal@1",
            "semantic_root_cid": state.parent["semantic_root_cid"],
            "graph_root_id": kwargs["graph_root_id"],
            "full_symbol_ids": ["symbol-helper", "symbol-target"],
            "nominee_symbol_ids": ["symbol-helper", "symbol-target"],
            "graph": {"nodes": [
                {"record": {"native_symbol_id": "symbol-target", "capsule_cid": "capsule-target"}},
                {"record": {"native_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}}],
                "edges": []},
            "unresolved_state_binding_rows": [{"unresolved_target": "global:helper"}],
            "source_freshness_verified": True, "owner_task_policy_verified": False,
            "source_check_scope": "opening_and_closing_current_native_verification",
            "effective_task_coverage_verified": False, "source_substitution_authority": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False,
            "deployment_ready": False, "runtime_activation": False,
        }
        if state.diagnostic_bytes:
            proposal["authored_diagnostic"] = "x" * state.diagnostic_bytes
        state.latest_source_proposal = deepcopy(proposal)
        receipt = {
            "schema": "supervisor-semantic-typed-owner-policy-receipt@1",
            "task_id": TASK, "nomination_artifact": str(state.nomination_path),
            "nomination_sha256": _sha(state.origin_text.encode()),
            "original_policy_sha256": _sha(_encoded(state.policy.to_dict())),
            "selected_policy_sha256": _sha(_encoded(state.selected.to_dict())),
            "native_prompt_sha256": _sha(state.prompt), "parent_artifact": ARTIFACT,
            "parent_sha256": _sha(state.parent_text.encode()),
            "semantic_root_cid": state.parent["semantic_root_cid"],
            "graph_root_id": kwargs["graph_root_id"], "source_proposal": proposal,
            "added_selected_optional_capsule_ids": ["capsule-helper"],
            "known_nominee_capsule_ids": ["capsule-helper", "capsule-target"],
            "owner_required_partition_preserved": True, "owner_optional_partition_preserved": True,
            "prior_selected_optional_preserved": True, "source_freshness_verified": True,
            "owner_declaration_rechecked": True, "owner_task_coverage_verified": False,
            "effective_task_coverage_verified": False, "source_substitution_authority": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False,
            "deployment_ready": False, "runtime_activation": False,
            "provider_called": False, "descriptor_published": False,
            "projection_aware": False, "full_capsule_count": 2, "delivered_capsule_count": 2,
            "source_check_scope": "current_source_wrapper_and_post_builder_native_verification",
            "selection_basis": "recorded_typed_nominees_within_independent_owner_partition",
        }
        value = {"schema": "supervisor-semantic-typed-owner-nomination@1",
                 "nomination": owner.build_decision_nomination(state.selected),
                 "launch_policy": descriptor, "policy": state.selected.to_dict(), "receipt": receipt}
        if state.after_typed is not None:
            state.after_typed()
        return TypedOwnerNomination(_encoded(value).decode())

    monkeypatch.setattr(owner.codec, "_prompt_parts", parts)
    monkeypatch.setattr(owner.codec, "_verified_semantic", verified)
    monkeypatch.setattr(owner, "build_decision_launch_policy", launch)
    monkeypatch.setattr(typed_policy, "build_typed_owner_decision_nomination", typed)
    # Allow either root's lazy module import or a patchable forwarding alias.
    monkeypatch.setattr(owner, "build_typed_owner_decision_nomination", typed, raising=False)
    def current_replay(**kwargs):
        state.replay_calls.append(("current", kwargs))
        verified(kwargs["repository"], artifact=kwargs["artifact"], text=state.parent_text,
                 task_id=kwargs["task_id"], current=True)
        return deepcopy(state.latest_source_proposal)
    def historical_replay(**kwargs):
        state.replay_calls.append(("historical", kwargs))
        value = deepcopy(state.latest_source_proposal)
        value["source_freshness_verified"] = False
        value["source_check_scope"] = "authored_retained_native_replay"
        return value
    monkeypatch.setattr(lineage_api, "project_source_bound_typed_links", current_replay)
    monkeypatch.setattr(lineage_api, "project_worker_source_bound_typed_links", current_replay)
    monkeypatch.setattr(lineage_api, "replay_source_bound_typed_links_for_audit", historical_replay)
    return state


def _arguments(state, mode=None):
    values = ["--model", "fixture-model", "--purpose", "coding", "--semantic-repository", str(state.repository),
        "--semantic-decision-nomination-artifact", str(state.nomination_path),
        "--semantic-decision-nomination-sha256", _sha(state.origin_text.encode()),
        "--semantic-decision-task-id", TASK]
    if mode is not None:
        values += ["--semantic-decision-typed-nomination", mode]
    return values


def _prepare(state, mode="enabled", *, arguments=None):
    return owner.prepare_decision_owner_dispatch(arguments=_arguments(state, mode) if arguments is None else arguments,
        prompt_bytes=state.prompt, workspace=state.workspace)


def _assert_immutable(path, expected_sha):
    assert _sha(path.read_bytes()) == expected_sha
    assert path.name == expected_sha + ".json"
    metadata = path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_nlink == 1


def test_enabled_handoff_commits_origin_derived_lineage_and_final_policy_without_changing_stdin(handoff):
    state = handoff
    result = _prepare(state)
    receipt = result.receipt
    assert len(state.typed_calls) == 1
    assert receipt["nomination_artifact"] == str(state.nomination_path)
    assert receipt["nomination_sha256"] == _sha(state.origin_text.encode())
    binding = receipt["typed_nomination"]
    assert set(binding) == {"schema", "derived_nomination_artifact", "derived_nomination_sha256",
                            "lineage_artifact", "lineage_sha256"}
    assert binding["schema"] == "supervisor-semantic-typed-nomination-binding@1"
    derived, lineage = Path(binding["derived_nomination_artifact"]), Path(binding["lineage_artifact"])
    _assert_immutable(derived, binding["derived_nomination_sha256"])
    _assert_immutable(lineage, binding["lineage_sha256"])
    assert derived.parent == state.repository / ".runtime/semantic-decision-nominations"
    assert lineage.parent == state.repository / ".runtime/semantic-typed-nominations"
    assert json.loads(derived.read_bytes()) == owner.build_decision_nomination(state.selected)
    retained = json.loads(lineage.read_bytes())
    assert set(retained) == {"schema", "origin_nomination_text", "typed_nomination"}
    assert retained["schema"] == "supervisor-semantic-typed-nomination-lineage@1"
    assert retained["origin_nomination_text"] == state.origin_text
    assert retained["typed_nomination"]["policy"] == state.selected.to_dict()
    assert receipt["policy_sha256"] == _sha(_encoded(state.selected.to_dict()))
    launch = Path(receipt["launch_policy_artifact"])
    _assert_immutable(launch, receipt["launch_policy_sha256"])
    assert json.loads(launch.read_bytes())["policy"] == state.selected.to_dict()
    assert result.prompt_bytes == state.prompt
    assert "--semantic-decision-typed-nomination" not in result.worker_arguments
    assert "--semantic-decision-nomination-artifact" not in result.worker_arguments
    assert receipt["provider_called"] is receipt["completion_authority"] is False
    assert len(_encoded(receipt)) <= 65_536
    assert "source_proposal" not in receipt and "global:helper" not in json.dumps(receipt)
    assert state.source_path.read_bytes() == SOURCE
    assert state.nomination_path.read_text() == state.origin_text


@pytest.mark.parametrize("mode", [None, "disabled"])
def test_bare_runtime_and_explicit_optout_keep_legacy_owner_policy(handoff, mode):
    state = handoff
    result = _prepare(state, mode)
    assert state.typed_calls == []
    assert result.receipt["policy_sha256"] == _sha(_encoded(state.policy.to_dict()))
    assert "typed_nomination" not in result.receipt
    assert json.loads(Path(result.receipt["launch_policy_artifact"]).read_bytes())["policy"] == state.policy.to_dict()
    assert not (state.repository / ".runtime/semantic-decision-nominations").exists()
    assert not (state.repository / ".runtime/semantic-typed-nominations").exists()


@pytest.mark.parametrize("damage", ["true", "false", "boolean", "inline", "duplicate", "missing"])
def test_typed_mode_is_one_exact_literal_two_token_binding(handoff, damage):
    state = handoff
    arguments = _arguments(state)
    flag = "--semantic-decision-typed-nomination"
    if damage in {"true", "false"}:
        arguments += [flag, damage]
    elif damage == "boolean":
        arguments += [flag, True]
    elif damage == "inline":
        arguments += [flag + "=enabled"]
    elif damage == "duplicate":
        arguments += [flag, "enabled", flag, "disabled"]
    else:
        arguments += [flag]
    with pytest.raises(ValueError):
        _prepare(state, arguments=arguments)
    assert state.typed_calls == state.launch_calls == []


def test_large_audit_sidecar_does_not_expand_compact_owner_receipt(handoff):
    state = handoff
    state.diagnostic_bytes = 300_000
    result = _prepare(state)
    receipt = result.receipt
    lineage = Path(receipt["typed_nomination"]["lineage_artifact"])
    assert 256_000 < lineage.stat().st_size <= 2_000_000
    assert len(_encoded(receipt)) <= 65_536
    assert "authored_diagnostic" not in json.dumps(receipt)
    assert json.loads(lineage.read_bytes())["typed_nomination"]["receipt"]["source_proposal"]["authored_diagnostic"] == "x" * 300_000


def test_sidecar_over_hard_ceiling_cannot_return_successful_dispatch(handoff):
    state = handoff
    state.diagnostic_bytes = 2_000_000
    with pytest.raises(ValueError):
        _prepare(state)
    assert not list((state.repository / ".runtime").rglob(".pending-*"))


def test_typed_graph_refusal_precedes_any_immutable_dispatch_publication(handoff):
    state = handoff
    state.typed_error = ValueError("authored missing known typed dependency")
    with pytest.raises(ValueError, match="missing known typed dependency"):
        _prepare(state)
    assert not (state.repository / ".runtime/semantic-decision-policies").exists()
    assert not (state.repository / ".runtime/semantic-decision-nominations").exists()
    assert not (state.repository / ".runtime/semantic-typed-nominations").exists()


@pytest.mark.parametrize("target", ["source", "parent", "origin"])
def test_changes_after_real_publication_prevent_worker_dispatch(handoff, monkeypatch, target):
    state = handoff
    actual_link = owner.os.link
    commits = []
    path = {"source": state.source_path, "parent": state.parent_path, "origin": state.nomination_path}[target]
    before = path.read_bytes()
    def link(*args, **kwargs):
        value = actual_link(*args, **kwargs)
        commits.append(args)
        if len(commits) == 1:
            path.chmod(0o600)
            path.write_bytes(before + b"\n# concurrent external change\n")
        return value
    monkeypatch.setattr(owner.os, "link", link)
    with pytest.raises(ValueError):
        _prepare(state)
    assert commits and path.read_bytes() != before
    assert not list((state.repository / ".runtime").rglob(".pending-*"))


def test_partial_publication_failure_retains_audit_files_and_cleans_pending_inode(handoff, monkeypatch):
    state = handoff
    actual_link = owner.os.link
    calls = []
    def link(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise OSError("authored later publication failure")
        return actual_link(*args, **kwargs)
    monkeypatch.setattr(owner.os, "link", link)
    with pytest.raises(OSError, match="later publication failure"):
        _prepare(state)
    assert len(calls) == 2
    retained = list((state.repository / ".runtime/semantic-decision-nominations").glob("*.json"))
    assert len(retained) == 1
    assert retained[0].stat().st_nlink == 1 and stat.S_IMODE(retained[0].stat().st_mode) == 0o444
    assert not list((state.repository / ".runtime").rglob(".pending-*"))
    lock = state.repository / ".runtime/.semantic-decision-publish.lock"
    assert lock.read_bytes() == b"" and stat.S_IMODE(lock.stat().st_mode) == 0o600
    # A later exact retry proves the lock and immutable retained artifact are
    # reusable; the failed attempt itself returned no worker dispatch object.
    monkeypatch.setattr(owner.os, "link", actual_link)
    assert _prepare(state).receipt["provider_called"] is False


def test_exact_typed_repeat_reuses_all_immutable_artifacts_with_new_owner_observation(handoff):
    state = handoff
    first = _prepare(state)
    extension = first.receipt["typed_nomination"]
    paths = [Path(first.receipt["launch_policy_artifact"]),
             Path(extension["derived_nomination_artifact"]), Path(extension["lineage_artifact"])]
    before = {str(path): (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in paths}
    second = _prepare(state)
    assert first.worker_arguments == second.worker_arguments
    assert first.receipt["typed_nomination"] == second.receipt["typed_nomination"]
    assert first.receipt["binding_id"] != second.receipt["binding_id"]
    assert before == {str(path): (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) for path in paths}


@pytest.mark.parametrize("damage", ["bytes", "mode", "hardlink", "symlink"])
def test_preexisting_typed_sidecar_must_remain_exact_owned_immutable_single_link(handoff, tmp_path, damage):
    state = handoff
    first = _prepare(state)
    path = Path(first.receipt["typed_nomination"]["lineage_artifact"])
    if damage == "bytes":
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    elif damage == "mode":
        path.chmod(0o600)
    elif damage == "hardlink":
        (tmp_path / "lineage-alias.json").hardlink_to(path)
    else:
        copy = tmp_path / "other-lineage.json"
        copy.write_bytes(path.read_bytes())
        copy.chmod(0o444)
        path.unlink()
        path.symlink_to(copy)
    with pytest.raises((ValueError, OSError)):
        _prepare(state)


def _verify_lineage(state, result, **changes):
    receipt = result.receipt
    binding = receipt["typed_nomination"]
    arguments = dict(artifact=Path(binding["lineage_artifact"]),
        expected_sha256=binding["lineage_sha256"], native_prompt=state.prompt.decode(),
        repository=state.repository, workspace=state.workspace,
        expected_origin_nomination_sha256=_sha(state.origin_text.encode()), expected_task_id=TASK,
        owner_binding=receipt, require_current_source=False)
    arguments.update(changes)
    return lineage_api.load_and_verify_typed_nomination_lineage(**arguments)


def test_actual_lineage_loader_returns_only_historical_commitments(handoff):
    state = handoff
    result = _prepare(state)
    before = deepcopy(result.receipt)
    verification = _verify_lineage(state, result)
    assert verification["schema"] == "supervisor-semantic-typed-nomination-verification@1"
    assert verification["owner_binding_sha256"] == _sha(_encoded(result.receipt))
    assert verification["origin_nomination_sha256"] == _sha(state.origin_text.encode())
    assert verification["derived_nomination_sha256"] == result.receipt["typed_nomination"]["derived_nomination_sha256"]
    assert verification["source_freshness_revalidated"] is False
    assert type(verification["provider_calls"]) is int and verification["provider_calls"] == 0
    assert verification["execution_authority"] is verification["proof_authority"] is verification["completion_authority"] is False
    assert state.replay_calls[-1][0] == "historical"
    assert "graph" not in verification and "global:helper" not in json.dumps(verification)
    assert result.receipt == before


@pytest.mark.parametrize("target", ["lineage", "derived", "launch"])
@pytest.mark.parametrize("damage", ["missing", "bytes", "mode", "hardlink", "symlink"])
def test_actual_lineage_loader_refuses_unavailable_or_changed_published_material(handoff, tmp_path, target, damage):
    state = handoff
    result = _prepare(state)
    receipt = result.receipt
    binding = receipt["typed_nomination"]
    path = Path({"lineage": binding["lineage_artifact"], "derived": binding["derived_nomination_artifact"],
                 "launch": receipt["launch_policy_artifact"]}[target])
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    elif damage == "mode":
        path.chmod(0o600)
    elif damage == "hardlink":
        (tmp_path / "published-alias.json").hardlink_to(path)
    else:
        copied = tmp_path / "foreign-published.json"
        copied.write_bytes(path.read_bytes())
        copied.chmod(0o444)
        path.unlink()
        path.symlink_to(copied)
    with pytest.raises((ValueError, OSError)):
        _verify_lineage(state, result)


@pytest.mark.parametrize("damage", ["origin_pin", "task", "native_prompt", "lineage_pin", "owner_parent", "owner_policy"])
def test_actual_lineage_loader_requires_independent_origin_and_final_input_bindings(handoff, damage):
    state = handoff
    result = _prepare(state)
    changes = {}
    if damage == "origin_pin":
        changes["expected_origin_nomination_sha256"] = "0" * 64
    elif damage == "task":
        changes["expected_task_id"] = "TASK-FOREIGN"
    elif damage == "native_prompt":
        changes["native_prompt"] = state.prompt.decode() + "Changed literal suffix"
    elif damage == "lineage_pin":
        changes["expected_sha256"] = "0" * 64
    else:
        receipt = result.receipt
        receipt["parent_sha256" if damage == "owner_parent" else "policy_sha256"] = "0" * 64
        changes["owner_binding"] = receipt
    with pytest.raises(ValueError):
        _verify_lineage(state, result, **changes)


@pytest.mark.parametrize("damage", ["extra", "origin_text", "derived_policy", "stored_graph", "self_verified"])
def test_rehashed_lineage_cannot_replace_original_policy_or_replayed_graph(handoff, damage):
    state = handoff
    result = _prepare(state)
    receipt = result.receipt
    old_path = Path(receipt["typed_nomination"]["lineage_artifact"])
    value = json.loads(old_path.read_bytes())
    if damage == "extra":
        value["verified"] = True
    elif damage == "origin_text":
        value["origin_nomination_text"] += " "
    elif damage == "derived_policy":
        value["typed_nomination"]["policy"]["selected_optional_capsule_ids"].clear()
    elif damage == "stored_graph":
        value["typed_nomination"]["receipt"]["source_proposal"]["nominee_symbol_ids"] = ["symbol-target"]
    else:
        value["typed_nomination"]["receipt"]["source_proposal"]["lineage_verified"] = True
    raw = _encoded(value)
    digest = _sha(raw)
    path = old_path.parent / (digest + ".json")
    path.write_bytes(raw)
    path.chmod(0o444)
    receipt["typed_nomination"].update(lineage_artifact=str(path), lineage_sha256=digest)
    with pytest.raises(ValueError):
        _verify_lineage(state, result, artifact=path, expected_sha256=digest, owner_binding=receipt)


def test_historical_lineage_does_not_claim_current_freshness_after_source_changes(handoff):
    state = handoff
    result = _prepare(state)
    state.source_path.write_bytes(SOURCE + b"\n# task changes happened later\n")
    historical = _verify_lineage(state, result)
    assert historical["source_freshness_revalidated"] is False
    with pytest.raises(ValueError, match="source changed"):
        _verify_lineage(state, result, require_current_source=True)
