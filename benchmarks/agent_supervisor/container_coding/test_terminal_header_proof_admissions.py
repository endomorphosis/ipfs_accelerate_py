"""Bound Terminal observations of a native scoped header proof.

The router and pinned prover transcripts are explicitly authored fixture
outputs during setup. Source admission, complete keys, sealed SQLite custody,
the generic DuckDB gate and the bridge are native. Terminal is then forbidden
from running a prover, constructing a proof owner or changing its worker input.
These observations establish no whole-program or task-completion authority.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_live_metadata_census as census
from benchmarks.agent_supervisor.container_coding.local_live_planner import preflight_proposal
from benchmarks.agent_supervisor.container_coding.test_terminal_formal_status_census import _repeat_preparation
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.test_terminal_live_metadata_census import _git
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorExecutableRole,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_cache_state import DoctorCacheState
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorProofCacheGate, DoctorSealedReceiptStore
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import canonical_json, content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as factory
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_cache_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import formal_status_census as formal
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_scoped_analysis import build_scoped_doctor_analysis
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_agent_supervisor_doctor_proof_cache import _identity
from test.api.test_doctor_contract_proof_custody import _profiles
from test.api.test_doctor_header_contracts import PROGRAM, analyze


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _path_pin(gate):
    return hashlib.sha256(str(gate.db_path).encode()).hexdigest()


def _prepare_header_owner(tmp_path, monkeypatch):
    setup = tmp_path / "authored-terminal-header-fixture"
    setup.mkdir(mode=0o700)
    root, state = setup / "repository", setup / "state"
    root.mkdir(mode=0o755)
    # Install the actual header span BEFORE any native source admission.
    (root / "source.py").write_text(PROGRAM)
    (root / "support.py").write_text("def authored_reference(value):\n    return value\n")
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "authored public header fixture")):
        _git(root, *args)
    (root / ".git").chmod(0o755)
    instruction = setup / "instruction.md"
    instruction.write_text("Improve source.py while preserving behavior. This is an authored header fixture.")
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
        "input_paths": ["source.py", "support.py"],
        "outputs": [{"path": "source.py", "effect": "modify", "media_type": "text/x-python"}]}
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
                            task_profile=profile, disable_intent_autoencoder=True)
    state.chmod(0o700)
    initial = prep.initial_context(state=state)
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    proposal = preflight_proposal({"spec": prepared["spec"],
        "evidence": _select_evidence(request, scan, prep._config(root))})
    proposal["root_goal_key"] = "TB-GOAL"
    proposal["goals"][0].update(goal_key="TB-GOAL", title="Authored public header fixture",
        objective="Fulfill the public instruction")
    proposal["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Authored header task", objective="Fulfill the public instruction")
    proposal["tasks"][0].update(task_key=prepared["spec"]["task_key"], goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction",
        predicted_files=[row["path"] for row in prepared["spec"]["outputs"]])
    router_calls = []

    def authored_router(prompt, **kwargs):
        router_calls.append(prompt)
        return {"text": json.dumps(proposal), "observation": {}, "execution_receipt": None}

    _version(monkeypatch)
    planning = prep.plan(state, provider_callable=authored_router)
    assert planning["qualified"] is True and len(router_calls) == 1
    indexed = prep.context(state=state)
    (root / ".runtime").chmod(0o755)
    for name in ("prepared", "admission", "planning-result", "context-result", "initial-context-result"):
        (state / (name + ".json")).chmod(0o600)
    (state / "intent.duckdb").chmod(0o600)
    task_cid = planning["task_cids"][0]
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task_before, watermark = intent.get_task(task_cid), intent.event_watermark()
    return {"root": root, "state": state, "initial": initial, "indexed": indexed,
            "task_cid": task_cid, "task_before": task_before, "watermark": watermark}


@pytest.fixture
def host(tmp_path, monkeypatch):
    terminal = _prepare_header_owner(tmp_path, monkeypatch)
    admission = json.loads((terminal["state"] / "admission.json").read_bytes())
    scoped = build_scoped_doctor_analysis(repository=terminal["root"], admission=admission,
                                         task_cid=terminal["task_cid"])
    candidate = analyze().candidate
    values = {"terminal": terminal, "scoped": scoped, "contract": candidate.contract,
              "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
              "calls": [], "tmp_path": tmp_path}
    for name in ("lean", "z3"):
        path = tmp_path / ("authored-terminal-" + name)
        path.write_text("#!/bin/sh\n# authored inert executable; never run\nexit 99\n")
        path.chmod(0o755)
        values[name] = path
    prepared = factory.prepare_header_contract_proof(scoped=scoped, contract=values["contract"],
        candidate_cid=values["candidate_cid"], lean=values["lean"], z3=values["z3"],
        proof_key_profiles=values["profiles"])
    cfg = json.loads(prepared.kernel_pin.argv[-2])

    def authored_pinned_output(self, pin, request):
        values["calls"].append(pin.role)
        theorem, lowering = request["theorem"], request["lowering"]
        output = {"theorem_cid": content_identity(theorem), "lowering_cid": content_identity(lowering),
            "property_id": theorem["property_id"], "consequence_ref": theorem["consequence_ref"],
            "premise_ids": theorem["premise_ids"]}
        if pin.role is DoctorExecutableRole.SOLVER:
            output.update(status="proved", proof_object=theorem["theorem_body"])
        else:
            native = request["native_receipt"]
            output.update(status="kernel_verified", native_receipt_cid=content_identity(native),
                proof_object_cid=content_identity({"proof_object": native["proof_object"]}),
                kernel_id=cfg["kernel_id"])
        transcript = {"executable_sha256": pin.executable_sha256, "command_id": pin.command_id,
            "stdin_cid": content_identity({"bytes_hex": canonical_json(request).encode().hex()}),
            "stdout_cid": content_identity({"bytes_hex": canonical_json(output).encode().hex()}),
            "stderr_cid": content_identity({"bytes_hex": ""}), "exit_code": 0}
        transcript["transcript_cid"] = content_identity(transcript)
        return output, transcript

    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", authored_pinned_output)
    with public_replay_without_metadata():
        store = DoctorSealedReceiptStore(tmp_path / "PRIVATE_HEADER_PROOF.sqlite",
                                        authority_id="authored-terminal-header-owner")
        hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=store,
            trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
        proof = hammer.verify_authoritative(prepared.theorem, prepared.lowering,
            solver_pin=prepared.solver_pin, kernel_pin=prepared.kernel_pin,
            current_roots=prepared.theorem.roots,
            eligible_consequence_refs=(prepared.theorem.consequence_ref,))
        reference = custody.persist_header_proof_key_custody(prepared=prepared, proof=proof, hammer=hammer)
        gate = DoctorProofCacheGate(tmp_path / "PRIVATE_HEADER_CACHE.duckdb")
        owner_pin = _path_pin(gate)
        header = bridge.import_header_proof_to_cache(prepared=prepared, hammer=hammer,
            custody_ref=reference, expected_key_id=prepared.key.key_id, gate=gate,
            expected_owner_path_sha256=owner_pin)
        replacement_cache = FormalVerificationCache(gate.db_path, exact_path=True)
        replacement_gate = DoctorProofCacheGate(cache=replacement_cache)
        replacement_state = DoctorCacheState(gate.authoritative_cache)
        replacement_store = DoctorSealedReceiptStore(store.path, authority_id=store.authority_id)
    expected = ((prepared.key.key_id, header.receipt.receipt_id),)
    owner = census.TerminalFormalStatusOwner(scoped_analysis=scoped, gate=gate, keys=(prepared.key,),
        owner_path_sha256=owner_pin, expected_key_ids=(prepared.key.key_id,),
        header_proof_admissions=(header,), expected_receipt_ids=expected)
    values.update(prepared=prepared, store=store, hammer=hammer, proof=proof, custody_ref=reference,
        gate=gate, header=header, owner=owner, owner_pin=owner_pin, expected=expected,
        replacement_cache=replacement_cache, replacement_gate=replacement_gate,
        replacement_state=replacement_state, replacement_store=replacement_store)
    return values


def _forbid_terminal_producers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Terminal observation must reuse admitted proof owners without new proof/model calls")
    for cls in (DoctorSealedReceiptStore, DoctorProofCacheGate, FormalVerificationCache, DoctorCacheState):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", forbidden)
    monkeypatch.setattr(bridge, "import_header_proof_to_cache", forbidden)
    for name in ("prepare", "initial_context", "plan", "context"):
        monkeypatch.setattr(prep, name, forbidden)
    for name in ("admit_local_benchmark_plan", "materialize_local_benchmark_plan", "run_local_task_validations"):
        monkeypatch.setattr(local, name, forbidden)


def _open(host):
    return census.prepare_terminal_live_metadata_census(state=host["terminal"]["state"],
        require_source384=False, include_doctor_context=False,
        include_formal_status_census=True, formal_status_owner=host["owner"])


def _assert_closed(host, report):
    nested = report["formal_status_census"]
    exported = json.dumps(report, sort_keys=True)
    for private in (str(host["terminal"]["root"]), str(host["terminal"]["state"]),
                    str(host["gate"].db_path), str(host["store"].path),
                    host["terminal"]["task_cid"], host["scoped"].report["scope_id"],
                    host["prepared"].key.key_id, host["header"].receipt.receipt_id,
                    "clean_payload", "native_custody_receipt_cid", "authored_reference"):
        assert private not in exported
    assert len(exported.encode()) < 16 * 1024
    for name in formal._AUTHORITIES:
        assert nested[name] is False
    assert nested["new_storage_owners_created"] == 0
    assert nested["atomic_owner_snapshot"] is False
    assert report["provider_calls"] == report["model_runs"] == 0


def test_bound_header_observation_two_passes_preserve_native_prompt_and_worker_input(host, monkeypatch):
    terminal = host["terminal"]
    with census.prepare_terminal_live_metadata_census(state=terminal["state"],
            require_source384=False, include_doctor_context=False) as baseline:
        prompt = baseline["prompt_artifact"].read_bytes()
        options = baseline["runner_kwargs"].copy()
    _repeat_preparation(terminal)
    calls = list(host["calls"])
    _forbid_terminal_producers(monkeypatch)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert observed["prompt_artifact"].read_bytes() == prompt
        assert observed["runner_kwargs"] == options
        assert nested["status"] == "measured" and nested["observation_current"] is True
        assert nested["accepted_receipts"] == nested["requested_key_count"] == 1
        assert nested["binding_admission"] == "terminal-signed-task-scoped-header-bridge@1"
        assert nested["header_admission_current_checks"] > 0
        _assert_closed(host, report)
    assert report["formal_status_census"] is nested
    assert nested["status"] == "measured" and nested["accepted_receipts"] == 1
    assert nested["observation_current"] is True
    assert report["formal_status_owner_observation_passes"] == 2
    assert report["formal_status_native_observation_totals"]["gate_lookup_calls"] == 2
    assert report["formal_status_native_observation_totals"]["gate_render_revalidation_calls"] == 2
    assert report["formal_status_native_observation_totals"]["header_admission_current_checks"] >= 2
    assert report["formal_status_context_binding_current"] is True
    assert report["allocated_worktree_removed"] is True
    assert host["calls"] == calls
    with IntentRepository(terminal["state"] / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(terminal["task_cid"]) == terminal["task_before"]
        assert intent.event_watermark() == terminal["watermark"]
    _assert_closed(host, report)


@pytest.mark.parametrize("mismatch", ["missing-receipts", "missing-admissions", "empty-admissions",
    "missing-key", "duplicate-admission", "duplicate-expected", "wrong-receipt", "wrong-key",
    "foreign-handle", "foreign-gate", "foreign-scope"])
def test_invalid_header_population_refuses_before_native_gate_observation(host, monkeypatch, mismatch):
    changes = {}
    if mismatch == "missing-receipts":
        changes["expected_receipt_ids"] = None
    elif mismatch == "missing-admissions":
        changes["header_proof_admissions"] = None
    elif mismatch == "empty-admissions":
        changes["header_proof_admissions"] = ()
    elif mismatch == "missing-key":
        changes.update(keys=(), expected_key_ids=())
    elif mismatch == "duplicate-admission":
        changes["header_proof_admissions"] = (host["header"], host["header"])
    elif mismatch == "duplicate-expected":
        changes["expected_receipt_ids"] = host["expected"] * 2
    elif mismatch == "wrong-receipt":
        changes["expected_receipt_ids"] = ((host["prepared"].key.key_id,
                                          content_identity({"authored": "foreign-receipt"})),)
    elif mismatch == "wrong-key":
        key = replace(host["prepared"].key, goal=_identity("authored-foreign-header-goal"))
        changes.update(keys=(key,), expected_key_ids=(key.key_id,),
            expected_receipt_ids=((key.key_id, host["header"].receipt.receipt_id),))
    elif mismatch == "foreign-handle":
        changes["header_proof_admissions"] = (object(),)
    elif mismatch == "foreign-gate":
        changes["gate"] = host["replacement_gate"]
    else:
        changes["scoped_analysis"] = replace(host["scoped"], task_cid="FOREIGN_SCOPED_TASK")
    host["owner"] = replace(host["owner"], **changes)
    _forbid_terminal_producers(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid header admission must refuse before consulting a generic owner")

    for gate in (host["gate"], host["replacement_gate"]):
        for name in ("lookup", "revalidate_for_render", "revalidate_for_commit", "is_quarantined", "is_tombstoned"):
            monkeypatch.setattr(gate, name, forbidden)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert nested["status"] == "refused"
        assert nested["accepted_receipts"] is None and nested["observation_current"] is False
        assert nested["gate_lookup_calls"] == nested["gate_render_revalidation_calls"] == 0
        _assert_closed(host, report)


@pytest.mark.parametrize("drift", ["source", "task", "executable", "profile", "native-seal", "native-inode",
    "native-object", "emitter", "codec", "quarantine", "generic-state", "generic-state-cache", "generic-cache"])
def test_exit_revalidation_clears_escaped_acceptance_after_proof_binding_drift(host, monkeypatch, drift):
    if drift in {"emitter", "codec"}:
        module = bridge if drift == "emitter" else bridge.receipt_contracts
        copied = host["tmp_path"] / ("fixture-copied-" + drift + ".py")
        copied.write_bytes(Path(module.__file__).read_bytes())
        monkeypatch.setattr(module, "__file__", str(copied))
    _forbid_terminal_producers(monkeypatch)
    raised = None
    try:
        with _open(host) as observed:
            report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
            assert nested["accepted_receipts"] == 1
            if drift == "source":
                (host["terminal"]["root"] / "source.py").write_text(PROGRAM + "\n# changed\n")
            elif drift == "task":
                terminal = host["terminal"]
                with IntentRepository(terminal["state"] / "intent.duckdb", install_schema=False) as intent:
                    intent.cas_task_status(task_cid=terminal["task_cid"],
                        expected_revision=terminal["task_before"]["revision"], new_status="in_progress")
            elif drift == "executable":
                host["lean"].write_text("#!/bin/sh\n# changed proof executable\nexit 98\n")
            elif drift == "profile":
                object.__setattr__(host["prepared"].proof_key_profiles, "policy", host["profiles"].environment)
            elif drift == "native-seal":
                host["store"]._key_path.write_bytes(b"x" * 32)
            elif drift == "native-inode":
                path = host["store"].path
                original = path.with_name("retained-authored-proof.sqlite")
                path.rename(original)
                path.write_bytes(original.read_bytes())
            elif drift == "native-object":
                host["hammer"]._authoritative_store = host["replacement_store"]
            elif drift in {"emitter", "codec"}:
                copied.write_bytes(copied.read_bytes() + b"\n# authored byte drift\n")
            elif drift == "quarantine":
                host["gate"].quarantine(host["prepared"].key)
            elif drift == "generic-state":
                host["gate"]._state = host["replacement_state"]
            elif drift == "generic-state-cache":
                host["gate"]._state._cache = host["replacement_cache"]
            else:
                host["gate"]._cache = host["replacement_cache"]
    except ValueError as exc:
        raised = exc
    assert report["formal_status_census"] is nested
    assert nested["status"] == "refused" and nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is (drift not in {"source", "task"})
    assert (raised is not None) is (drift in {"source", "task"})
    _assert_closed(host, report)


def test_consistently_wrong_generic_receipt_is_refused_despite_complete_key_matches(host, monkeypatch):
    _forbid_terminal_producers(monkeypatch)
    original = host["gate"].lookup
    foreign = replace(host["header"].receipt,
                      metadata={**host["header"].receipt.to_dict()["metadata"], "authored_wrong_receipt": True})
    assert foreign.receipt_id != host["expected"][0][1]

    def wrong_receipt(key, **kwargs):
        result = original(key, **kwargs)
        assert result.hit
        return replace(result, receipt=foreign, audit=replace(result.audit, receipt_id=foreign.receipt_id))

    monkeypatch.setattr(host["gate"], "lookup", wrong_receipt)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert nested["status"] == "refused" and nested["accepted_receipts"] is None
        assert nested["observation_current"] is False
        _assert_closed(host, report)


def test_quarantine_appearing_after_a_positive_lookup_cannot_escape_observation(host, monkeypatch):
    _forbid_terminal_producers(monkeypatch)
    gate = host["gate"]
    original_lookup, original_tombstone = gate.lookup, gate.is_tombstoned
    state = {"positive_returned": False, "revoked": False}
    before_source = (host["terminal"]["root"] / "source.py").read_bytes()

    def observe_lookup(key, **kwargs):
        result = original_lookup(key, **kwargs)
        assert result.hit
        state["positive_returned"] = True
        return result

    def revoke_after_tombstone_read(key):
        result = original_tombstone(key)
        if state["positive_returned"] and not state["revoked"]:
            assert result is False
            state["revoked"] = True
            gate.quarantine(key)
        return result

    monkeypatch.setattr(gate, "lookup", observe_lookup)
    monkeypatch.setattr(gate, "is_tombstoned", revoke_after_tombstone_read)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert nested["status"] == "refused" and nested["accepted_receipts"] is None
        assert nested["observation_current"] is False
        _assert_closed(host, report)
    assert state == {"positive_returned": True, "revoked": True}
    assert (host["terminal"]["root"] / "source.py").read_bytes() == before_source


def test_public_status_flags_cannot_skip_private_exit_revalidation(host, monkeypatch):
    _forbid_terminal_producers(monkeypatch)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert nested["accepted_receipts"] == 1
        nested["completion_authority"] = True
        nested["authored_extra_field"] = "must be discarded"
        nested["status"] = "unmeasured"
        nested["observation_current"] = False
        report["formal_status_host_owner_supplied"] = False
        object.__setattr__(host["prepared"].proof_key_profiles, "policy", host["profiles"].environment)
    assert report["formal_status_census"] is nested
    assert nested["status"] == "refused" and nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is True
    assert nested["completion_authority"] is False
    assert "authored_extra_field" not in nested
    _assert_closed(host, report)


def test_nested_report_replacement_cannot_retain_an_escaped_positive(host, monkeypatch):
    _forbid_terminal_producers(monkeypatch)
    try:
        with _open(host) as observed:
            report, original = observed["metadata"], observed["metadata"]["formal_status_census"]
            assert original["accepted_receipts"] == 1
            original["completion_authority"] = True
            original["authored_extra_field"] = "must be discarded"
            report["formal_status_census"] = dict(original)
            object.__setattr__(host["prepared"].proof_key_profiles, "policy", host["profiles"].environment)
    except ValueError:
        pass
    assert report["formal_status_census"] is original
    assert original["status"] == "refused" and original["observation_current"] is False
    assert original["accepted_receipts"] is None
    assert original["completion_authority"] is False
    assert "authored_extra_field" not in original
    _assert_closed(host, report)


def test_cleanup_refusal_invalidates_private_observation_despite_public_flag_tampering(host, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    original_run = engine.run_command

    def fail_cleanup(argv, **kwargs):
        if "worktree" in argv and "remove" in argv:
            # Inject after the final successful proof observation, so the
            # exceptional cleanup path itself must restore the private body.
            nested["completion_authority"] = True
            nested["authored_extra_field"] = "must be discarded"
            raise RuntimeError("authored bound-header cleanup failure")
        return original_run(argv, **kwargs)

    _forbid_terminal_producers(monkeypatch)
    monkeypatch.setattr(engine, "run_command", fail_cleanup)
    with pytest.raises(RuntimeError, match="authored bound-header cleanup failure"):
        with _open(host) as observed:
            report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
            assert nested["accepted_receipts"] == 1
            nested["completion_authority"] = True
            nested["authored_extra_field"] = "must be discarded"
            nested["status"] = "unmeasured"
            nested["observation_current"] = False
            report["formal_status_host_owner_supplied"] = False
            report["formal_status_context_binding_current"] = False
    assert report["formal_status_census"] is nested
    assert nested["status"] == "refused" and nested["observation_current"] is False
    assert nested["accepted_receipts"] is None
    assert report["formal_status_context_binding_current"] is False
    assert nested["completion_authority"] is False
    assert "authored_extra_field" not in nested
    _assert_closed(host, report)
    # Cleanup only this test's known authored allocation after restoring Git.
    monkeypatch.setattr(engine, "run_command", original_run)
    if observed["workspace"].exists():
        prep._git(host["terminal"]["root"], "worktree", "remove", "--force", str(observed["workspace"]))


def test_raw_terminal_owner_cannot_count_scoped_receipt_without_bound_header_handle(host, monkeypatch):
    host["owner"] = replace(host["owner"], header_proof_admissions=None, expected_receipt_ids=None)
    _forbid_terminal_producers(monkeypatch)
    with _open(host) as observed:
        report, nested = observed["metadata"], observed["metadata"]["formal_status_census"]
        assert nested["status"] == "refused"
        assert nested["accepted_receipts"] is None and nested["observation_current"] is False
        assert nested["gate_lookup_calls"] == 1
        assert nested["gate_render_revalidation_calls"] == 0
        assert report["formal_status_context_binding_current"] is True
        _assert_closed(host, report)
    assert report["formal_status_context_binding_current"] is True
    assert nested["accepted_receipts"] is None
