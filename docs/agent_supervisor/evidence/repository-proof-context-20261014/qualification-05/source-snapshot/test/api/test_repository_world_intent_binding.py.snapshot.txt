"""Native intent observations remain owner-bound through worker dispatch.

The native database, signed local plan, source index and declared checks are
real. Provider calls are authored; these checks establish no live token savings.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextTier
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.repository_intent_world import (
    capture_repository_intent_world,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    RepositoryWorldPacketError, prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _admit, _start
from test.api.test_repository_context_compiler import compile_context, compiler, obligation
from test.api.test_repository_source_index import scheduler


TASK = "Make the public answer check pass without changing its validation source."
EFFECTS = ("inspect_repository", "edit_source", "run_validation")
PATHS = ("answer.py",)


@pytest.fixture
def intent_case(scenario, tmp_path, scheduler):
    admission, tasks = _admit(scenario)
    task_id = tasks["LOCAL-TASK"]
    with RepositorySourceSession(scenario["repository"], tmp_path / "intent-source-state",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        observer = capture_repository_intent_world(session=session, intent=scenario["intent"],
            admission=admission, task_id=task_id, generation=1)
        yield dict(scenario, admission=admission, task_id=task_id, session=session,
                   adapter=session.context_adapter(compiler()), observation=observer)
    assert scheduler.snapshot()["active_lease_count"] == 0


def compiled(case, *, references=None, optional=()):
    references = (case["observation"].reference,) if references is None else references
    return compile_context(case["adapter"], references, query="answer", optional_evidence=optional,
        goal={"intent": TASK}, authority={"mode": "proposal", "allowed_paths": PATHS,
            "world_task_packet": repository_world_task_binding(case["task_id"], TASK, EFFECTS, PATHS)},
        scope={"paths": PATHS}, acceptance={"criteria": ("The declared public check passes.",)})


def prepared(case, compilation=None, **changes):
    arguments = dict(adapter=case["adapter"], compilation=compiled(case) if compilation is None else compilation,
        generation=1, task_id=case["task_id"], task_text=TASK,
        unresolved_obligations=("Inspect the observed requirements and perform the remaining work.",),
        validation_commands=("python test_answer.py",), allowed_effects=EFFECTS, write_paths=PATHS,
        intent_world_observation=case["observation"])
    arguments.update(changes)
    return prepare_repository_world_packet(**arguments)


def test_issued_native_observation_is_required_and_receipt_is_content_free(intent_case):
    case = intent_case
    packet = prepared(case)
    receipt = packet.validate_current(workspace=case["repository"], task_text=TASK)
    assert receipt["intent_world"] == case["observation"].receipt()
    delivered = json.loads(packet.prompt.split("\n", 1)[1])["packet"]["members"]
    references = [row["payload"]["reference"] for row in delivered
                  if row["kind"] == "evidence" and "reference" in row["payload"]]
    assert case["observation"].reference.to_record() in references
    assert receipt["independent_world_owner_freshness"] == "not_revalidated"
    assert receipt["world_projection_status"] == "unavailable"
    assert not receipt["proof_authority"] and not receipt["completion_authority"]
    encoded = json.dumps(receipt)
    assert TASK not in encoded and case["observation"].reference.summary not in encoded
    for key in ("repository", "profile", "lifecycle"):
        assert str(case[key]) not in encoded
    detached = packet.receipt()
    detached["intent_world"].clear()
    assert packet.receipt()["intent_world"] == case["observation"].receipt()


def test_reserved_intent_reference_requires_issued_observer(intent_case):
    with pytest.raises(RepositoryWorldPacketError, match="intent_observation_missing"):
        prepared(intent_case, intent_world_observation=None)


@pytest.mark.parametrize("change", ["summary", "metadata", "missing"])
def test_native_observation_cannot_be_rewritten_or_omitted_from_context(intent_case, change):
    reference = intent_case["observation"].reference
    if change == "summary":
        references = (replace(reference, summary=reference.summary + " altered"),)
    elif change == "metadata":
        references = (replace(reference, metadata={**reference.metadata, "authored_extra": True}),)
    else:
        references = (obligation("preserve-public-check"),)
    context = compiled(intent_case, references=references)
    with pytest.raises(RepositoryWorldPacketError, match="intent_reference_mismatch"):
        prepared(intent_case, context)


def test_native_observation_cannot_be_demoted_to_optional_context(intent_case):
    reference = intent_case["observation"].reference
    optional = replace(reference, tier=ContextTier.EVIDENCE,
                       metadata={**reference.metadata, "required": False})
    context = compiled(intent_case, references=(obligation("preserve-public-check"),), optional=(optional,))
    with pytest.raises(RepositoryWorldPacketError, match="intent_reference_mismatch"):
        prepared(intent_case, context)


def test_unissued_observation_and_replaced_prepared_packet_are_refused(intent_case):
    case = intent_case
    with pytest.raises(ValueError):
        prepared(case, intent_world_observation=replace(case["observation"]))
    packet = prepared(case)
    forged = replace(packet, _intent_world_observation=None)
    with pytest.raises(RepositoryWorldPacketError, match="not_issued"):
        forged.validate_current(workspace=case["repository"], task_text=TASK)
    with pytest.raises(RepositoryWorldPacketError, match="invalid_world_packet_intent_observation"):
        prepared(case, intent_world_observation=object())


def test_native_task_change_invalidates_packet_without_source_change(intent_case):
    case = intent_case
    packet = prepared(case)
    source_tree = case["session"].status_summary()["source_tree_sha256"]
    _start(case, case["task_id"])
    assert case["session"].status_summary()["source_tree_sha256"] == source_tree
    with pytest.raises(ValueError):
        packet.validate_current(workspace=case["repository"], task_text=TASK)


def test_real_declared_check_invalidates_old_packet_and_reaches_successor(intent_case):
    case = intent_case
    _start(case, case["task_id"])
    case["observation"] = capture_repository_intent_world(session=case["session"], intent=case["intent"],
        admission=case["admission"], task_id=case["task_id"], generation=1)
    packet = prepared(case)
    before_tree = case["session"].status_summary()["source_tree_sha256"]
    actual = local.run_local_task_validations(intent=case["intent"], task_cid=case["task_id"],
        attempt_id="world-packet:observed-failure")
    assert actual["passed"] is False
    assert case["session"].status_summary()["source_tree_sha256"] == before_tree
    with pytest.raises(ValueError):
        packet.validate_current(workspace=case["repository"], task_text=TASK)
    current = capture_repository_intent_world(session=case["session"], intent=case["intent"],
        admission=case["admission"], task_id=case["task_id"], generation=1)
    case["observation"] = current
    successor = prepared(case)
    assert "failed" in current.reference.summary
    assert actual["results"][0]["evidence_digest"] in successor.prompt
    assert successor.receipt()["intent_world"] == current.receipt()
    assert successor.receipt()["intent_world"] != packet.receipt()["intent_world"]


def test_runner_rechecks_native_owner_after_provider_initialization(intent_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    case = intent_case
    packet = prepared(case)
    calls = []
    monkeypatch.chdir(case["repository"])
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-provider-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))

    def initialize(*args, **kwargs):
        _start(case, case["task_id"])
        return object()

    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    monkeypatch.setattr(llm_router, "generate_text", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
                   max_output_tokens=128, world_task_packet=packet)
    assert case["intent"].get_task(case["task_id"])["status"] == "in_progress"
    assert calls == []
