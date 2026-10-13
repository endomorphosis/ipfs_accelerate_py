"""Real planning dispatch over authored native source and reviewed Intent.

Providers are stubbed at the existing transport seam. These tests exercise the
native source index and input budgets without executing the inspected program.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_planning_experiment as experiment
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import RequiredContextOverflowError
from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import (
    TaskProposalRouterError, run_task_proposal_router,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchCapacity, ProviderBatchScheduler, ProviderBatchSchedulerConfig,
)
from test.api.test_repository_source_index import scheduler  # noqa: F401


def token_counter(value):
    return max(1, len(value.encode("utf-8")) // 4)


@pytest.fixture
def case(tmp_path, scheduler, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")
    workspace = tmp_path / "source"
    workspace.mkdir()
    for path, content in experiment.authored_sources().items():
        (workspace / path).write_text(content, encoding="utf-8")
    with RepositorySourceSession(workspace, tmp_path / "state", representation="compact",
        index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        config = experiment.make_config(workspace, tmp_path / "artifacts", session, counter=token_counter)
        calls = []

        def provider(prompt, invocation):
            calls.append((prompt, invocation))
            return experiment.scripted_proposal(prompt)

        monkeypatch.setattr("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router", provider)
        yield SimpleNamespace(workspace=workspace, session=session, config=config, calls=calls,
            provider=provider, scheduler=scheduler)
    assert scheduler.snapshot()["active_lease_count"] == 0


def generate(config):
    return run_task_proposal_router(config, generate=True, provider="authored-stub",
        model="authored-stub", max_new_tokens=1024, timeout_seconds=2)


def required_evidence(prompt):
    return {row["reference_id"]: row for row in json.loads(prompt)["evidence"]
        if row.get("metadata", {}).get("required") is True}


def test_real_consumer_preserves_reviewed_intent_and_explicit_unavailable_domains(case):
    payload = generate(case.config)
    assert len(case.calls) == 1
    prompt, invocation = case.calls[0]
    body = json.loads(prompt)
    refs = required_evidence(prompt)
    assert set(refs) == {"planning-required:intent_ir", "planning-required:security_ir", "planning-required:ui_ux_ir"}
    assert json.loads(refs["planning-required:intent_ir"]["summary"]) == case.config.repository_context.intent_contract
    for domain in ("security_ir", "ui_ux_ir"):
        status = json.loads(refs["planning-required:" + domain]["summary"])
        assert status["schema"] == "repository-planning-domain-status@1"
        assert status["status"] == "unavailable"
    assert body["tree_id"] == "sha256:" + case.session.status()["source_tree_sha256"]
    assert body["authority"]["repository_context"]["generation"] == case.session.generation
    assert body["authority"]["proof_authority"] is False
    assert body["authority"]["task"]["allowed_paths"] == ["answer.py"]
    assert payload["repository_tree_id"] == body["tree_id"]
    assert payload["prompt_bytes"] == len(prompt.encode("utf-8"))
    assert payload["prompt_tokens"] == token_counter(prompt)
    assert payload["context_input_tokens"] >= payload["prompt_tokens"]
    assert invocation.repo_root == case.workspace
    proposal = json.loads((case.config.artifact_dir / "rpc-001-proposal.json").read_text())
    assert proposal["repository_tree_id"] == body["tree_id"]
    assert proposal["authority_claims"]["completion_authoritative"] is False
    receipt = (case.config.artifact_dir / "rpc-001-context-receipt.json").read_text()
    assert prompt not in receipt
    assert experiment.authored_sources()["requirements.txt"].strip() not in receipt
    assert (case.workspace / "answer.py").read_text() == experiment.authored_sources()["answer.py"]


def test_dry_run_uses_original_snapshot_and_does_not_dispatch_or_persist(case):
    payload = run_task_proposal_router(case.config, generate=False)
    assert payload["generate"] is False
    assert payload["prompt_bytes"] > 0
    assert payload["prompt_tokens"] > 0
    assert case.calls == []
    assert not case.config.artifact_dir.exists()
    assert case.session.read(generation=case.session.generation, path="answer.py")["generation"] == 1


def test_exact_token_count_is_unknown_without_supplied_tokenizer(case):
    payload = generate(replace(case.config, context_tokenizer=None))
    assert payload.get("prompt_tokens") is None
    assert payload["context_input_tokens"] > 0
    assert payload["prompt_bytes"] == len(case.calls[0][0].encode("utf-8"))


def test_expansion_sends_full_reconstructed_context_and_preserves_required_bodies(case):
    generate(case.config)
    initial = case.calls[-1][0]
    candidates = case.session.context_query(generation=1, query=experiment.QUERY, limit=3)["candidates"]
    selected = next(row for row in candidates if row["path"] == "answer.py")
    expanded = replace(case.config, repository_context=replace(case.config.repository_context,
        expansion_reference_ids=("repository-symbol:" + selected["symbol_id"],)))
    payload = generate(expanded)
    prompt = case.calls[-1][0]
    decoded = json.loads(prompt)
    assert all(key in decoded for key in ("goal", "authority", "scope", "acceptance", "evidence"))
    assert required_evidence(initial) == required_evidence(prompt)
    refs = [row for row in decoded["evidence"] if row.get("metadata", {}).get("repository_capsule_expanded")]
    assert len(refs) == 1 and refs[0]["path"] == "answer.py"
    assert json.loads(refs[0]["summary"])["proof_authority"] is False
    assert payload["prompt_tokens"] == token_counter(prompt)
    assert len(prompt) > len(initial)


@pytest.mark.parametrize("path", ["answer.py", "plan.md", "tasks.todo.md", "requirements.txt"])
def test_mutated_canonical_inputs_refuse_before_provider(case, path):
    selected = case.workspace / path
    selected.write_text(selected.read_text() + "\n# changed outside the session\n")
    with pytest.raises((ValueError, TaskProposalRouterError), match="stale|refresh|source|changed|current"):
        generate(case.config)
    assert case.calls == []
    assert not case.config.artifact_dir.exists()


class MutationRuntime:
    def __init__(self, callback, when, artifact_kind="context_receipt"):
        self.callback, self.when, self.artifact_kind = callback, when, artifact_kind
        self.fired = False

    def _fire(self):
        if not self.fired:
            self.fired = True
            self.callback()

    def route(self, boundary, payload):
        if self.when == "provider_route" and boundary == "task_proposal" and payload.get("operation"):
            self._fire()
        if self.when == "artifact_route" and payload.get("artifact_kind") == self.artifact_kind:
            self._fire()
        return SimpleNamespace(payload=payload, decision_request=SimpleNamespace(expected_effects=()))

    def authorize_mutation(self, decision, callback):
        if self.when == "artifact_authorize" and decision.payload.get("artifact_kind") == self.artifact_kind:
            self._fire()
        return callback()


@pytest.mark.parametrize("mutation", ["source", "board", "plan", "generation"])
def test_decision_runtime_mutation_before_dispatch_is_fenced(case, mutation):
    def mutate():
        if mutation == "generation":
            case.session.refresh(generation=1)
        else:
            path = {"source": "answer.py", "board": "tasks.todo.md", "plan": "plan.md"}[mutation]
            target = case.workspace / path
            target.write_text(target.read_text() + "\n# intervening change\n")
    runtime = MutationRuntime(mutate, "provider_route")
    with pytest.raises((ValueError, TaskProposalRouterError), match="stale|refresh|source|changed|generation|current"):
        generate(replace(case.config, decision_runtime=runtime))
    assert runtime.fired
    assert case.calls == []
    assert not case.config.artifact_dir.exists()


@pytest.mark.parametrize("mutation", ["source", "board", "plan", "generation"])
def test_source_or_generation_change_during_response_refuses_artifacts(case, monkeypatch, mutation):
    def provider(prompt, invocation):
        result = case.provider(prompt, invocation)
        if mutation == "generation":
            case.session.refresh(generation=1)
        else:
            path = {"source": "answer.py", "board": "tasks.todo.md", "plan": "plan.md"}[mutation]
            target = case.workspace / path
            target.write_text(target.read_text() + "\n# change while awaiting response\n")
        return result
    monkeypatch.setattr("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router", provider)
    with pytest.raises((ValueError, TaskProposalRouterError), match="stale|refresh|source|changed|generation|current"):
        generate(case.config)
    assert len(case.calls) == 1
    assert not case.config.artifact_dir.exists()


@pytest.mark.parametrize("when", ["artifact_route", "artifact_authorize"])
@pytest.mark.parametrize("artifact_kind", ["context_receipt", "task_proposal"])
def test_artifact_boundary_rechecks_inside_runtime_callback(case, when, artifact_kind):
    def mutate():
        target = case.workspace / "answer.py"
        target.write_text(target.read_text() + "\n# change after provider response\n")
    runtime = MutationRuntime(mutate, when, artifact_kind)
    with pytest.raises((ValueError, TaskProposalRouterError), match="stale|refresh|source|changed|current"):
        generate(replace(case.config, decision_runtime=runtime))
    assert runtime.fired
    assert len(case.calls) == 1
    assert not (case.config.artifact_dir / "rpc-001-proposal.json").exists()
    if artifact_kind == "context_receipt":
        assert not case.config.artifact_dir.exists()


@pytest.mark.parametrize("location", ["workspace", "session_state", "ancestor"])
def test_artifact_directory_must_be_disjoint_from_source_and_state(case, location):
    destination = {"workspace": case.workspace / "artifacts",
        "session_state": case.session.state_dir / "artifacts", "ancestor": case.workspace.parent}[location]
    with pytest.raises((ValueError, TaskProposalRouterError), match="artifact|outside|disjoint|overlap|state|workspace"):
        generate(replace(case.config, artifact_dir=destination))
    assert case.calls == []


def test_session_for_another_workspace_is_refused(case, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises((ValueError, TaskProposalRouterError), match="workspace|repository|root"):
        generate(replace(case.config, repo_root=other))
    assert case.calls == []


@pytest.mark.parametrize("field", ["task_board_path", "plan_path"])
def test_task_board_and_plan_must_belong_to_snapshot(case, tmp_path, field):
    external = tmp_path / ("external-" + field)
    external.write_text(getattr(case.config, field).read_text())
    with pytest.raises((ValueError, TaskProposalRouterError), match="outside|source|snapshot|workspace|path|repository"):
        generate(replace(case.config, **{field: external}))
    assert case.calls == []


def test_tight_budget_refuses_instead_of_dropping_required_intent_or_status(case):
    with pytest.raises(RequiredContextOverflowError):
        generate(replace(case.config, context_max_input_tokens=1))
    assert case.calls == []
    assert not case.config.artifact_dir.exists()


def test_native_expansion_requires_known_selected_handle(case):
    config = replace(case.config, repository_context=replace(case.config.repository_context,
        expansion_reference_ids=("repository-symbol:unknown",)))
    with pytest.raises(ValueError, match="reference|unknown|owned|handle|selected"):
        generate(config)
    assert case.calls == []


def test_existing_batch_capacity_backoff_is_preserved(case):
    samples, dispatched = [], []

    def capacity(provider_id):
        samples.append(provider_id)
        return ProviderBatchCapacity(provider_id=provider_id, healthy=True,
            max_batch_size=1, max_concurrent_batches=1,
            available_concurrent_batches=int(len(samples) >= 3))

    def dispatch(requests):
        members = tuple(requests)
        dispatched.extend(members)
        assert len(samples) >= 3
        return [experiment.scripted_proposal(item.payload) for item in members]

    with ProviderBatchScheduler(dispatch, capacity_supplier=capacity,
        config=ProviderBatchSchedulerConfig(batch_window_ms=0, admission_retry_ms=1,
            max_parallel_batches=1, max_batch_size=1, provider_limits={"authored-stub": 1})) as batch:
        payload = generate(replace(case.config, provider_batch_scheduler=batch))
        metrics = batch.metrics()
    assert len(samples) >= 3 and len(dispatched) == 1
    assert case.calls == []
    assert payload["provider_batch"]["status"] == "succeeded"
    assert dispatched[0].provenance["prompt_bytes"] == payload["prompt_bytes"]
    assert payload["prompt_tokens"] == token_counter(dispatched[0].payload)
    assert metrics.admission_deferrals >= 2


def test_queued_source_mutation_refuses_response_publication(case):
    samples, dispatched = [], []

    def capacity(provider_id):
        samples.append(provider_id)
        if len(samples) == 1:
            target = case.workspace / "answer.py"
            target.write_text(target.read_text() + "\n# source changed during capacity backoff\n")
        return ProviderBatchCapacity(provider_id=provider_id, healthy=True,
            max_batch_size=1, max_concurrent_batches=1,
            available_concurrent_batches=int(len(samples) >= 3))

    def dispatch(requests):
        members = tuple(requests)
        dispatched.extend(members)
        return [experiment.scripted_proposal(item.payload) for item in members]

    with ProviderBatchScheduler(dispatch, capacity_supplier=capacity,
        config=ProviderBatchSchedulerConfig(batch_window_ms=0, admission_retry_ms=1,
            max_parallel_batches=1, max_batch_size=1, provider_limits={"authored-stub": 1})) as batch:
        with pytest.raises((ValueError, TaskProposalRouterError), match="stale|refresh|source|changed|current"):
            generate(replace(case.config, provider_batch_scheduler=batch))
    # Existing batch admission has no transaction with the source owner. The
    # actual physical dispatch may happen; its stale response cannot publish.
    assert len(samples) >= 3 and len(dispatched) == 1
    assert case.calls == []
    assert not case.config.artifact_dir.exists()


def test_full_mixed_intent_population_reaches_provider_without_filtering(case):
    from test.api.test_repository_planning_evidence import mixed_contract
    source, contract = mixed_contract()
    (case.workspace / "instruction.txt").write_text(source)
    case.session.refresh(generation=1)
    config = replace(case.config, repository_context=replace(case.config.repository_context,
        intent_contract=contract, generation=case.session.generation))
    generate(config)
    reference = required_evidence(case.calls[-1][0])["planning-required:intent_ir"]
    body = json.loads(reference["summary"])
    assert body == contract
    rows = body["ledger"]["requirements"]
    assert any(row["modality"] == "permitted" for row in rows)
    assert any(row["kind"] != "goal" or row["representation"] != "native_statement" for row in rows)
    assert set(reference["metadata"]["coverage_ids"]) == {
        "planning-domain:intent_ir", *(row["requirement_id"] for row in rows)}


@pytest.mark.parametrize("artifact", ["context-receipt", "proposal", "proposal-rejection"])
@pytest.mark.parametrize("destination", ["source", "state"])
def test_artifact_child_symlink_cannot_overwrite_source_or_state(case, monkeypatch, artifact, destination):
    case.config.artifact_dir.mkdir()
    target = case.workspace / "answer.py"
    if destination == "state":
        target = case.session.state_dir / "protected-state.txt"
        target.write_text("Preserve private session state.\n")
    original = target.read_bytes()
    artifact_path = case.config.artifact_dir / ("rpc-001-" + artifact + ".json")
    artifact_path.symlink_to(target)
    if artifact == "proposal-rejection":
        monkeypatch.setattr("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router",
            lambda prompt, invocation: "authored invalid proposal")
    with pytest.raises((ValueError, TaskProposalRouterError, OSError), match="symlink|artifact|unsafe|symbolic|refus"):
        generate(case.config)
    assert target.read_bytes() == original
    assert artifact_path.is_symlink()


def test_task_id_cannot_supply_an_artifact_path(case):
    task_board = case.workspace / "tasks.todo.md"
    task_board.write_text(task_board.read_text().replace("RPC-001", "RPC-../../escaped"))
    case.session.refresh(generation=1)
    config = replace(case.config, repository_context=replace(case.config.repository_context,
        generation=case.session.generation))
    with pytest.raises((ValueError, TaskProposalRouterError), match="artifact|basename|name|path|task"):
        generate(config)
    assert not (case.workspace.parent / "escaped-context-receipt.json").exists()
    assert not (case.workspace.parent / "escaped-proposal.json").exists()


def test_stateless_consumer_benchmark_accounts_complete_prompts(monkeypatch):
    # The benchmark's real source sessions use the outer serialized campaign's
    # scheduler ledger. The callback remains a local stub throughout.
    result = experiment._exercise(token_counter)
    assert result["qualified"]
    assert result["stub_dispatches"] == 5
    for sequence in result["sequences"].values():
        assert sequence["consumer_calls"] == 2
        for field in ("prompt_bytes", "local_tokenizer_proxy_tokens"):
            assert sequence[field] == sum(result["arms"][arm][field] for arm in sequence["components"])
    assert result["checks"]["mandatory_evidence_identical"]
    assert result["checks"]["expanded_populations_match"]
