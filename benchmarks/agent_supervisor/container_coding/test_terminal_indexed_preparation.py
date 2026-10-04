"""Actual scanner/signature contracts without a provider or benchmark oracle."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.local_live_planner import preflight_proposal
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import parse_prompt_goal_graph, _select_evidence
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest, DirectoryScanReceipt
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    (root / "bottle.py").write_text("def application():\n    return 'original'\n")
    for args in (("init", "-q"), ("add", "."), ("-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "upstream")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "bottle.py").write_text("def application():\n    return 'original image dirty bytes'\n")
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Inspect bottle.py, repair problems and write report.jsonl with file_path and cwe_id fields.")
    return root, instruction, tmp_path / "state"


def _proposal_json(prepared):
    # This is explicitly a hand-authored parser fixture, never live output.
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    config = prep._config(Path(prepared["repository"]))
    fixture = preflight_proposal({"spec": prepared["spec"], "evidence": _select_evidence(request, scan, config)})
    fixture["root_goal_key"] = "TB-GOAL"
    fixture["goals"][0].update(goal_key="TB-GOAL", title="Public task", objective="Fulfill public task")
    fixture["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL", title="Repair", objective="Fulfill public task")
    fixture["tasks"][0].update(task_key="TB-CODE-TASK", goal_key="TB-SUBGOAL",
        objective="Fulfill public task", predicted_files=["bottle.py", "report.jsonl"])
    return json.dumps(fixture)


def _proposal_graph(prepared):
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    return parse_prompt_goal_graph(_proposal_json(prepared), request, scan,
        config=prep._config(Path(prepared["repository"])),
        constraint_summaries=prepared["constraints"])


def test_prepare_preserves_original_dirty_bytes_and_signs_absent_report(original):
    root, instruction, state = original
    before = (root / "bottle.py").read_bytes()
    result = prep.prepare(repository=root, instruction=instruction, state=state)
    assert (root / "bottle.py").read_bytes() == before
    assert not (root / "report.jsonl").exists()
    assert result["provider_calls"] == 0 and result["benchmark_success"] is None
    assert result["query"] == instruction.read_text()
    assert (root / prep.INSTRUCTION).read_text() == instruction.read_text()
    assert "original image dirty bytes" in (state / "original-image.diff").read_text()
    assert result["manifest"]["payload"]["schema"] == local.CREATE_MANIFEST_SCHEMA
    assert result["manifest"]["payload"]["created_outputs"] == ["report.jsonl"]
    graph = _proposal_graph(result)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=result["manifest"])
    verified = local.verify_local_benchmark_admission(admission)
    assert all(row["phase"] == "post_execution" for row in verified["receipt"]["pending_requirements"])
    assert verified["receipt"]["completion_authority"] is False
    assert result["provider"] == "codex_cli" and result["model"] == "gpt-6.1-sol"
    assert result["reasoning_effort"] == "high" and result["max_total_agent_seconds"] == 300
    proc = subprocess.run(prep.ARGV, cwd=root, capture_output=True)
    assert proc.returncode != 0


@pytest.mark.parametrize("mutation", ["preseed", "untracked", "symlink"])
def test_prepare_rejects_undeclared_or_preseeded_inputs(original, mutation):
    root, instruction, state = original
    if mutation == "preseed":
        (root / "report.jsonl").write_text("{}\n")
    elif mutation == "untracked":
        (root / "oracle.py").write_text("raise RuntimeError()")
    else:
        (root / "report.jsonl").symlink_to(instruction)
    with pytest.raises(ValueError):
        prep.prepare(repository=root, instruction=instruction, state=state)
    assert not state.exists()


def test_signed_task_cannot_omit_create_or_change_public_check(original):
    result = prep.prepare(repository=original[0], instruction=original[1], state=original[2])
    graph = _proposal_graph(result)
    forged = deepcopy(result["manifest"])
    forged["payload"]["tasks"][0]["validations"][0]["argv"] = ["true"]
    forged = local._signed(forged["payload"], forged["payload"])
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(graph=graph, manifest=forged)


def test_public_smoke_is_structural_and_not_an_oracle(original):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    # Arbitrary syntactically valid CWE IDs pass: this check cannot claim the
    # vulnerability has been identified or repaired.
    (root / "report.jsonl").write_text(json.dumps({"file_path": "/app/bottle.py", "cwe_id": ["cwe-999999"]}) + "\n")
    assert subprocess.run(prep.ARGV, cwd=root, capture_output=True).returncode == 0


def test_authored_fixture_context_uses_real_native_admission_and_full_index(original):
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    graph = _proposal_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    result = prep.context(state=state)
    assert result["provider_calls"] == 0 and result["benchmark_success"] is None
    assert result["indexed_symbols"] == result["native_fact_rows_replayed"] == 1
    assert result["worker_semantic_bytes"] <= 32768
    assert result["learned_embeddings"] is False
    assert result["semantic_root_cid"] and result["world_snapshot_cid"]
    assert result["doctor_repair"]["status"] == "abstained"
    assert result["doctor_repair"]["diagnostics_replayed"] is True
    assert result["doctor_repair"]["automatic_repair_eligible"] is False
    assert set(result["doctor_repair"]["stages"].values()) == {"not_run"}
    assert json.loads((state / "doctor-repair-eligibility.json").read_text()) == result["doctor_repair"]
    timings = result["timings"]
    assert timings["initial_function_imports_and_admission_seconds"] >= 0
    assert timings["initial_phase_included_in_seconds"] is False
    assert timings["final_context_result_persistence_included"] is False
    assert set(timings["nonoverlapping_seconds"]) == {
        "vector_qualification", "persisted_snapshot_reopen",
        "supervised_semantic_world_context", "bundle_persistence",
        "doctor_eligibility", "doctor_report_persistence",
    }
    assert all(value >= 0 for value in timings["nonoverlapping_seconds"].values())
    assert sum(timings["nonoverlapping_seconds"].values()) <= result["seconds"]
    assert timings["nested_timings_overlap_parent"] is True
    assert timings["nested_vector_timings_may_overlap_each_other"] is True
    assert timings["nested_helper_seconds"]["vector_qualification"] == json.loads(
        (root / ".runtime/terminal-vectors/result.json").read_text()
    )["seconds"]
    assert json.loads((state / "context-result.json").read_text())["timings"] == timings
    assert not (root / "report.jsonl").exists()
    assert (root / ".runtime/terminal-context-bundle.json").is_file()


def test_planner_requires_external_worker_isolation_before_any_call(tmp_path):
    with pytest.raises(ValueError, match="isolated worker"):
        prep.plan(tmp_path)


def test_tampered_public_query_cannot_reach_planner(original):
    result = prep.prepare(repository=original[0], instruction=original[1], state=original[2])
    result["query"] = "A hint not in the public instruction"
    prep._write(original[2] / "prepared.json", result)
    calls = []
    with pytest.raises(ValueError, match="signed public input"):
        prep.plan(original[2], provider_callable=lambda *a, **k: calls.append(a))
    assert calls == []
    assert not (original[2] / "planner-invoked.json").exists()


def test_accepted_fixture_admission_survives_native_storage_failure(original, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepositoryBoundsError

    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    native_run = subprocess.run
    def reported_version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.158.0\n", "")
        return native_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", reported_version)
    def failed_storage(**kwargs):
        assert json.loads((state / "admission.json").read_text()) == kwargs["admission"]
        raise IntentRepositoryBoundsError("injected native storage failure")
    monkeypatch.setattr(local, "materialize_local_benchmark_plan", failed_storage)
    # An explicit authored parser fixture, not a live provider qualification.
    result = prep.plan(state, provider_callable=lambda *args, **kwargs: {
        "text": _proposal_json(prepared), "observation": {}, "execution_receipt": None,
    })
    assert result["qualified"] is False
    assert result["failure"]["type"] == "IntentRepositoryBoundsError"
    assert result["provider_receipt"]["outcome"] == "provider"
    saved = json.loads((state / "admission.json").read_text())
    assert local.verify_local_benchmark_admission(saved)["receipt"]["planning_permitted"] is True


@pytest.mark.parametrize("failure", [None, "tamper", "replay", "summary_budget"])
def test_optional_source_unit_advice_reaches_only_bounded_planner_context(original, monkeypatch, failure):
    from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as advice

    root, instruction, state = original
    raw_instruction = instruction.read_text()
    calls = []
    # Explicit authored adapter-boundary fixture; datasets tests exercise the
    # real checkpoint and this test exercises signed planner request delivery.
    def native(text, intent, security):
        calls.append(text)
        return {"schema":"source-document-autoencoder/v1", "report_sha256":"1"*64,
            "candidates":[{"domain":"intent_ir"}], "counts":{"intent_candidates":1},
            "security_regions":[], "intent":{"units":[{"accepted":True,
                "start_char":0,"end_char":len(text),"inference":{"learned":{"frame":{
                    "actor":"maintainer","action":"inspect","object":"bottle.py","modality":"intended"}}}}]}}
    monkeypatch.setattr(advice, "_native", native)
    prepared = prep.prepare(repository=root,instruction=instruction,state=state,
        enable_source_unit_autoencoder=True)
    assert calls == [raw_instruction]
    assert prepared["query"] == raw_instruction
    assert json.loads((state/"intent-advice.json").read_text())["status"] == "disabled"
    if failure == "tamper":
        with (state/"source-unit-advice.json").open("ab") as stream: stream.write(b" ")
    elif failure == "replay":
        def unavailable(*args,**kwargs): raise ValueError("injected unavailable checkpoint")
        monkeypatch.setattr(advice,"_native",unavailable)
    elif failure == "summary_budget":
        def oversize(*args,**kwargs): raise ValueError("injected existing planner byte bound")
        monkeypatch.setattr(advice,"source_unit_planner_summary",oversize)
    native_run = subprocess.run
    def version(argv,*args,**kwargs):
        if argv == ["codex","--version"]:
            return subprocess.CompletedProcess(argv,0,"codex-cli 0.158.0\n","")
        return native_run(argv,*args,**kwargs)
    monkeypatch.setattr(subprocess,"run",version)
    captured=[]
    def fixture_provider(prompt,**kwargs):
        captured.append(prompt)
        return {"text":_proposal_json(prepared),"observation":{},"execution_receipt":None}
    result=prep.plan(state,provider_callable=fixture_provider)
    assert len(captured)==1 and raw_instruction in captured[0]
    assert (root/prep.INSTRUCTION).read_text()==raw_instruction
    assert result["provider_receipt"]["outcome"] == "provider"
    assert result["intent_preplanning"]["status"] == "superseded_by_source_unit_preplanning"
    assert result["intent_preplanning"]["supplied_to_router"] is False
    delivery=result["source_unit_preplanning"]
    assert delivery["supplied_to_router"] is (failure is None)
    assert ("Optional source-unit autoencoder advice" in captured[0]) is (failure is None)
    assert delivery["execution_authority"] is delivery["source_semantics_verified"] is False
    if failure is None:
        assert delivery["inference_replays"]==1 and delivery["replay_seconds"]>=0


def test_intent_ablation_overrides_source_unit_intent_checkpoint_selection(original, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as advice
    root,instruction,state=original
    calls=[]
    def checked_native(text,intent,security):
        calls.append((intent,security))
        return {"candidates":[],"report_sha256":"1"*64}
    monkeypatch.setattr(advice,"_native",checked_native)
    security=instruction.parent/"security-descriptor.json"
    security.write_text('{"schema":"explicit_security_fixture"}')
    prepared=prep.prepare(repository=root,instruction=instruction,state=state,
        disable_intent_autoencoder=True,enable_source_unit_autoencoder=True,
        intent_checkpoint_descriptor=instruction.parent/"intentionally-unreadable-intent.json",
        source_unit_security_decoder_descriptor=security)
    assert calls==[(None,{"schema":"explicit_security_fixture"})]
    assert prepared["source_unit_preplanning"]["status"]=="fail_open_no_candidates"
    assert prepared["query"]==instruction.read_text()


@pytest.mark.parametrize("disable_intent", [False, True])
def test_source_family_flags_reach_preparation_with_intent_ablation_precedence(original, monkeypatch, disable_intent):
    from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as advice

    root, instruction, state = original
    calls = []
    def checked_native(text, intent, security, family_request=None):
        calls.append((text, intent, security, deepcopy(family_request)))
        return {"candidates": [], "report_sha256": "1" * 64}
    monkeypatch.setattr(advice, "_native", checked_native)
    intent = instruction.parent / "intent-descriptor.json"
    if not disable_intent:
        intent.write_text('{"schema":"explicit_intent_fixture"}')
    security = instruction.parent / "security-descriptor.json"
    security.write_text('{"schema":"explicit_security_fixture"}')
    context = instruction.parent / "family-context.json"
    # The disabled Intent path must not open a stale or unavailable Intent-only
    # context, while the independently enabled Security decoder stays selected.
    if not disable_intent:
        context.write_text('{"schema":"explicit_family_context_fixture"}')
    selected = ["higher_order", "dcec"]
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        intent_checkpoint_descriptor=intent, disable_intent_autoencoder=disable_intent,
        enable_source_unit_autoencoder=True, source_unit_security_decoder_descriptor=security,
        source_unit_project_logic_families=True, source_unit_intent_family_context=context,
        source_unit_intent_logic_families=selected)
    assert len(calls) == 1
    text, intent_descriptor, security_descriptor, family_request = calls[0]
    assert text == instruction.read_text() == prepared["query"]
    assert intent_descriptor == (None if disable_intent else {"schema": "explicit_intent_fixture"})
    assert security_descriptor == {"schema": "explicit_security_fixture"}
    assert family_request["requested_families"] == (None if disable_intent else selected)
    assert family_request["context"] == (None if disable_intent else {"schema": "explicit_family_context_fixture"})
    if disable_intent:
        assert family_request["context_pin"] is None
    else:
        assert family_request["context_pin"]["path"] == str(context.absolute())
    sidecar = json.loads((state / "source-unit-advice.json").read_text())
    assert sidecar["schema"] == advice.FAMILY_SCHEMA
    assert sidecar["family_request"] == family_request
    assert sidecar["status"] == "fail_open_no_candidates"
    assert prepared["source_unit_preplanning"]["logic_families_enabled"] is True
    assert prepared["source_unit_preplanning"]["enabled"] is True
    assert json.loads((state / "intent-advice.json").read_text())["status"] == "disabled"
    assert (root / prep.INSTRUCTION).read_text() == instruction.read_text()
    assert not (root / "report.jsonl").exists()
