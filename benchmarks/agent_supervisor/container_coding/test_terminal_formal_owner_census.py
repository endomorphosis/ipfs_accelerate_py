"""Authored host admission exercises existing native owners, never a prover.

The canonical key profiles and kernel acceptance evidence are authored fixture
inputs. A matching signed scope ID joins the caller's admitted identities; it
does not authenticate those identities or establish a real theorem.
"""

from dataclasses import replace
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_live_metadata_census as census
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_live_metadata_census import owner
from benchmarks.agent_supervisor.container_coding.test_terminal_formal_status_census import _repeat_preparation
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorProofCacheGate
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.runtime import formal_status_census as formal
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_scoped_analysis import build_scoped_doctor_analysis
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_agent_supervisor_doctor_proof_cache import _identity, _key, _receipt


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


@pytest.fixture
def host(owner, tmp_path):
    admission = json.loads((owner["state"] / "admission.json").read_bytes())
    scoped = build_scoped_doctor_analysis(repository=owner["root"], admission=admission,
                                         task_cid=owner["task_cid"])
    scope_id = scoped.report["scope_id"]
    assert scoped.snapshot.roots.tree_id == scope_id
    # These full canonical identities are explicitly admitted by the fixture
    # host. Their logical IDs match the independently reconstructed signed scope.
    tree = _identity("authored-tree", scope_id)
    key = _key(forest=_identity("authored-forest", scope_id), tree=tree,
               overlay=_identity("authored-overlay", scope_id), candidate_tree=tree)
    with public_replay_without_metadata():
        gate = DoctorProofCacheGate(tmp_path / "PRIVATE_FORMAL_OWNER.duckdb")
        assert gate.put(key, _receipt(tree_id=scope_id,
                                     metadata={"private_note": "PRIVATE_PROOF_BODY"})).stored
    admitted = census.TerminalFormalStatusOwner(scoped_analysis=scoped, gate=gate, keys=(key,),
        owner_path_sha256=hashlib.sha256(str(gate.db_path).encode()).hexdigest(),
        expected_key_ids=(key.key_id,))
    return {"terminal": owner, "admitted": admitted, "key": key, "gate": gate, "scoped": scoped}


def _open(host, **kwargs):
    return census.prepare_terminal_live_metadata_census(state=host["terminal"]["state"],
        require_source384=False, include_doctor_context=False,
        include_formal_status_census=True, formal_status_owner=host["admitted"], **kwargs)


def _forbid_new_owners(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Terminal census must reuse its admitted native owner")
    monkeypatch.setattr(DoctorProofCacheGate, "__init__", forbidden)
    monkeypatch.setattr(FormalVerificationCache, "__init__", forbidden)


def _forbid_gate_observations(host, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid admission must refuse before consulting the native gate")
    for name in ("lookup", "revalidate_for_render", "revalidate_for_commit",
                 "is_quarantined", "is_tombstoned"):
        monkeypatch.setattr(host["gate"], name, forbidden)


def _assert_closed(host, report):
    nested = report["formal_status_census"]
    exported = json.dumps(report, sort_keys=True)
    for private in (str(host["terminal"]["root"]), str(host["terminal"]["state"]),
                    str(host["gate"].db_path), host["terminal"]["task_cid"],
                    host["scoped"].report["scope_id"], host["key"].key_id,
                    "PRIVATE_PROOF_BODY", "def identity", "authored_reference"):
        assert private not in exported
    assert len(exported.encode()) < 16 * 1024
    for name in formal._AUTHORITIES:
        assert nested[name] is False
    assert nested["atomic_owner_snapshot"] is False
    assert nested["eligibility_at_return_verified"] is False
    assert nested["repeated_status_population"] is None
    assert nested["new_storage_owners_created"] == 0


@pytest.mark.parametrize("population", ["hit", "miss", "empty"])
def test_admitted_native_population_preserves_prompt_and_exports_no_bodies(host, monkeypatch, population):
    if population == "hit":
        with census.prepare_terminal_live_metadata_census(state=host["terminal"]["state"],
                require_source384=False, include_doctor_context=False) as baseline:
            prompt, options = baseline["prompt_artifact"].read_bytes(), baseline["runner_kwargs"].copy()
        _repeat_preparation(host["terminal"])
    elif population == "miss":
        missing = replace(host["key"], goal=_identity("authored-missing-goal"))
        host["admitted"] = replace(host["admitted"], keys=(missing,), expected_key_ids=(missing.key_id,))
    else:
        host["admitted"] = replace(host["admitted"], keys=(), expected_key_ids=())
    _forbid_new_owners(monkeypatch)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert report["schema"] == census.FORMAL_STATUS_SCHEMA
        assert report["formal_status_context_binding_current"] is True
        assert nested["status"] == "measured"
        assert nested["observation_current"] is True
        assert nested["requested_key_count"] == (0 if population == "empty" else 1)
        assert nested["accepted_receipts"] == (1 if population == "hit" else 0)
        if population != "empty":
            assert nested["lookup_dispositions"][population] == 1
            assert nested["render_dispositions"][population] == 1
        else:
            assert nested["gate_lookup_calls"] == nested["gate_render_revalidation_calls"] == 0
        assert observed["metadata"]["provider_calls"] == observed["metadata"]["model_runs"] == 0
        if population == "hit":
            assert observed["prompt_artifact"].read_bytes() == prompt
            assert observed["runner_kwargs"] == options
        _assert_closed(host, report)
    assert report["post_yield_admission_task_revalidated"] is True
    assert report["formal_status_context_binding_current"] is True
    assert report["formal_status_census"] is nested
    assert nested["observation_current"] is True
    assert nested["accepted_receipts"] == (1 if population == "hit" else 0)
    assert report["formal_status_owner_observation_passes"] == 2
    totals = report["formal_status_native_observation_totals"]
    expected_calls = 0 if population == "empty" else 2
    assert totals["gate_lookup_calls"] == totals["gate_render_revalidation_calls"] == expected_calls


@pytest.mark.parametrize("mismatch", ["expected-keys", "scope-tree", "scope-forest", "scope-overlay",
                                      "owner-pin", "scoped-task", "forged-scope"])
def test_wrong_admission_refuses_before_gate_consultation(host, monkeypatch, mismatch):
    admitted = host["admitted"]
    if mismatch == "expected-keys":
        admitted = replace(admitted, expected_key_ids=())
    elif mismatch == "scope-tree":
        tree = _identity("foreign-tree", "FOREIGN_SCOPE")
        key = replace(host["key"], tree=tree, candidate_tree=tree)
        admitted = replace(admitted, keys=(key,), expected_key_ids=(key.key_id,))
    elif mismatch in {"scope-forest", "scope-overlay"}:
        name = mismatch.removeprefix("scope-")
        key = replace(host["key"], **{name: _identity("foreign-" + name, "FOREIGN_SCOPE")})
        admitted = replace(admitted, keys=(key,), expected_key_ids=(key.key_id,))
    elif mismatch == "owner-pin":
        admitted = replace(admitted, owner_path_sha256="0" * 64)
    elif mismatch == "scoped-task":
        admitted = replace(admitted, scoped_analysis=replace(host["scoped"], task_cid="FOREIGN_TASK"))
    else:
        forged = host["scoped"].report
        forged["scope_id"] = "FORGED_SCOPE"
        scoped = replace(host["scoped"], _report_json=json.dumps(forged))
        tree = _identity("forged-tree", "FORGED_SCOPE")
        key = replace(host["key"], forest=_identity("forged-forest", "FORGED_SCOPE"),
            overlay=_identity("forged-overlay", "FORGED_SCOPE"), tree=tree, candidate_tree=tree)
        admitted = replace(admitted, scoped_analysis=scoped, keys=(key,), expected_key_ids=(key.key_id,))
    host["admitted"] = admitted
    _forbid_new_owners(monkeypatch)
    _forbid_gate_observations(host, monkeypatch)
    with _open(host) as observed:
        report = observed["metadata"]
        nested = report["formal_status_census"]
        assert nested["status"] == "refused"
        assert nested["accepted_receipts"] is None
        assert nested["observation_current"] is False
        assert nested["gate_lookup_calls"] == nested["gate_render_revalidation_calls"] == 0
        assert nested["gate_denial_state_checks"] == 0
        _assert_closed(host, report)


@pytest.mark.parametrize("invalid", ["foreign-type", "disabled"])
def test_unsupported_owner_selection_refuses_before_terminal_owner_observation(tmp_path, monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid explicit owner selection must refuse before observing Terminal state")
    monkeypatch.setattr(census, "_owner_snapshot", forbidden)
    selection = object() if invalid == "foreign-type" else census.TerminalFormalStatusOwner(
        scoped_analysis=object(), gate=object(), keys=(), owner_path_sha256="0" * 64,
        expected_key_ids=())
    with pytest.raises(ValueError):
        with census.prepare_terminal_live_metadata_census(state=tmp_path, require_source384=False,
                include_formal_status_census=invalid != "disabled", formal_status_owner=selection):
            pytest.fail("invalid owner selection yielded")


@pytest.mark.parametrize("when", ["before", "after-yield"])
def test_existing_native_quarantine_never_retains_an_accepted_population(host, monkeypatch, when):
    if when == "before":
        host["gate"].quarantine(host["key"])
    _forbid_new_owners(monkeypatch)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        if when == "after-yield":
            assert nested["accepted_receipts"] == 1
            host["gate"].quarantine(host["key"])
        else:
            assert nested["status"] == "refused"
            assert nested["accepted_receipts"] is None
    nested = report["formal_status_census"]
    assert nested["status"] == "refused"
    assert nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is True
    _assert_closed(host, report)


def _mutate(host, mutation):
    terminal = host["terminal"]
    if mutation == "source":
        (terminal["root"] / "source.py").write_text("changed = True\n")
    elif mutation == "task":
        with IntentRepository(terminal["state"] / "intent.duckdb", install_schema=False) as intent:
            intent.cas_task_status(task_cid=terminal["task_cid"],
                expected_revision=terminal["task_before"]["revision"], new_status="in_progress")
    else:
        path = terminal["state"] / "context-result.json"
        value = json.loads(path.read_bytes())
        value["index_id"] = "FOREIGN_CONTEXT"
        prep._write(path, value)


@pytest.mark.parametrize("mutation", ["source", "task"])
def test_native_lookup_cannot_keep_a_positive_observation_after_current_owner_drift(host, monkeypatch, mutation):
    original_lookup, original_census = host["gate"].lookup, formal.census_formal_status
    observations = []
    mutated = False
    def drift_after_lookup(key):
        nonlocal mutated
        result = original_lookup(key)
        if not mutated:
            mutated = True
            _mutate(host, mutation)
        return result
    def capture(**kwargs):
        value = original_census(**kwargs)
        observations.append(value)
        return value
    monkeypatch.setattr(host["gate"], "lookup", drift_after_lookup)
    monkeypatch.setattr(formal, "census_formal_status", capture)
    _forbid_new_owners(monkeypatch)
    with pytest.raises(ValueError):
        with _open(host):
            pass
    assert observations
    for observed in observations:
        assert observed["accepted_receipts"] is None
        assert observed["observation_current"] is False
        assert observed["status"] == "refused"
    assert observations[0]["gate_lookup_calls"] == 1
    assert observations[0]["gate_render_revalidation_calls"] == 0


@pytest.mark.parametrize("mutation", ["source", "task", "context"])
def test_escaped_native_observation_clears_when_terminal_custody_changes(host, monkeypatch, mutation):
    _forbid_new_owners(monkeypatch)
    with pytest.raises(ValueError):
        with _open(host) as observed:
            report = observed["metadata"]
            assert report["formal_status_census"]["accepted_receipts"] == 1
            _mutate(host, mutation)
    nested = report["formal_status_census"]
    assert report["formal_status_context_binding_current"] is False
    assert nested["status"] == "refused"
    assert nested["reason_code"] == "terminal_context_custody_changed"
    assert nested["observation_current"] is False
    assert nested["accepted_receipts"] is None


def test_identical_owner_bytes_at_a_replaced_inode_are_not_the_admitted_native_owner(host, monkeypatch):
    _forbid_new_owners(monkeypatch)
    with _open(host) as observed:
        report = observed["metadata"]
        assert report["formal_status_census"]["accepted_receipts"] == 1
        path = host["gate"].db_path
        retained = path.with_name("fixture-original-owner.duckdb")
        original_inode = path.stat().st_ino
        path.rename(retained)
        path.write_bytes(retained.read_bytes())
        assert path.stat().st_ino != original_inode
    nested = report["formal_status_census"]
    assert nested["status"] == "refused"
    assert nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is True


def test_different_native_cache_object_at_the_same_path_is_not_the_admitted_owner(host, monkeypatch):
    path = host["gate"].db_path
    original_stamp = path.stat().st_ino
    # The fixture prepares another exact object for its existing private test
    # database before census, then prohibits every new owner constructor.
    with public_replay_without_metadata():
        replacement = FormalVerificationCache(path, exact_path=True)
    assert replacement is not host["gate"].authoritative_cache
    assert replacement.db_path == path
    assert path.stat().st_ino == original_stamp
    _forbid_new_owners(monkeypatch)
    with _open(host) as observed:
        report = observed["metadata"]
        nested = report["formal_status_census"]
        assert nested["accepted_receipts"] == 1
        host["gate"]._cache = replacement
    assert report["formal_status_census"] is nested
    assert nested["status"] == "refused"
    assert nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is True
    assert report["post_yield_admission_task_revalidated"] is True
    assert report["formal_status_owner_observation_passes"] == 2
    _assert_closed(host, report)


def test_cleanup_failure_invalidates_an_escaped_accepted_native_observation(host, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    original = engine.run_command
    def fail_cleanup(argv, **kwargs):
        if "worktree" in argv and "remove" in argv:
            raise RuntimeError("authored native-owner cleanup failure")
        return original(argv, **kwargs)
    monkeypatch.setattr(engine, "run_command", fail_cleanup)
    _forbid_new_owners(monkeypatch)
    with pytest.raises(RuntimeError, match="authored native-owner cleanup failure"):
        with _open(host) as observed:
            report = observed["metadata"]
            assert report["formal_status_census"]["accepted_receipts"] == 1
    nested = report["formal_status_census"]
    assert report["formal_status_context_binding_current"] is False
    assert nested["status"] == "refused"
    assert nested["reason_code"] == "terminal_context_custody_changed"
    assert nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    # Undo only this authored test's known allocation before fixture disposal.
    monkeypatch.setattr(engine, "run_command", original)
    if observed["workspace"].exists():
        prep._git(host["terminal"]["root"], "worktree", "remove", "--force", str(observed["workspace"]))
