"""SCA-615 production-wire bounded Grok proposal and independent Codex review.

Acceptance (fail-closed):

* A production model-assisted task invokes only the typed packet route.
* Grok cannot self-review.
* Codex receives only the bounded proposal/evidence slice.
* The final applied patch and merge bind to the admitted review chain.
* Absent/degraded/stale/cross-task receipts remain pending.
* Deterministic-only tasks invoke no model.
* No provider receives the repository corpus.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA,
    PRODUCTION_PROVIDER_ROUTE_INTERFACE,
    PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA,
    SCAEV615ROUTE,
    ImplementationProviderRouter,
    ProductionContractPacket,
    ProductionReceiptDisposition,
    ProviderReason,
    ProviderRole,
    ReviewPresence,
    RouteStatus,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
    build_production_provider_route_evaluation,
    evaluate_production_provider_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    MODEL_ASSISTED_PROVIDER_ROUTE_EVENT,
    PRODUCTION_PROVIDER_ROUTE_BINDING_EVENT,
    PRODUCTION_PROVIDER_ROUTE_EVENT,
    PRODUCTION_PROVIDER_ROUTE_PENDING_EVENT,
    ImplementationRetryDeferred,
    PortalTask,
    TodoImplementationDaemon,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_PATH = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "swissknife_contract_assurance"
    / "evaluation"
    / "production-provider-route.json"
)
# Fall back to workspace-relative path when tests run from the monorepo root.
if not EVALUATION_PATH.exists():
    EVALUATION_PATH = Path(
        "data/agent_supervisor/swissknife_contract_assurance/evaluation/"
        "production-provider-route.json"
    )

SNAPSHOT = "git-commit:sca-615-fixture"
PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "implementation_daemon.py"
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _daemon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TodoImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Production Route Test")
    _git(repo, "config", "user.email", "production-route@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Production provider route tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
    target = repo / PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SCA-",
        implement=True,
        implementation_command="raw-model-command-must-not-run",
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda _kind, _payload, action: action(),
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_ALLOW_RAW_MODEL_COMMAND",
        raising=False,
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE", "1")
    return daemon


def _task(**overrides: Any) -> PortalTask:
    payload = {
        "task_id": "SCA-615",
        "title": "Production-wire bounded Grok proposal and independent Codex review",
        "status": "ready",
        "completion": "manual",
        "priority": "P0",
        "track": "production-provider-routing",
        "outputs": [PATH],
        "validation": [
            "python3 -m pytest external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_production_provider_route.py -q"
        ],
        "acceptance": (
            "A production model-assisted task invokes only the typed packet route"
        ),
        "metadata": {
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
        },
    }
    payload.update(overrides)
    return PortalTask(**payload)


def _events(daemon: TodoImplementationDaemon) -> list[dict[str, Any]]:
    if not daemon.events_path.exists():
        return []
    return [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _accept(proposal):
    return {"accepted": True, "reason_code": f"admitted:{proposal.role.value}"}


def _grok(request):
    assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
    provider_input = request["provider_input"]
    assert "contract_packet" in provider_input
    encoded = json.dumps(provider_input, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    assert "workspace_path" not in encoded
    return {
        "proposal": {
            "declared_paths": [PATH],
            "files": [
                {
                    "path": PATH,
                    "content": "# production-route-applied\n",
                }
            ],
        }
    }


def _codex(request):
    assert request["role"] == ProviderRole.CODEX_REVIEW.value
    provider_input = request["provider_input"]
    assert "contract_packet" not in provider_input
    assert "admitted_implementation_proposal" in provider_input
    assert "evidence_slice" in provider_input
    assert provider_input["admitted_implementation_proposal"][
        "completion_authoritative"
    ] is False
    encoded = json.dumps(provider_input, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    return {"decision": "approve", "findings": []}


def test_production_model_assisted_invokes_only_typed_packet_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    workspace = daemon.repo_root
    writes_via_raw = []

    def forbid_raw(*_args, **_kwargs):
        writes_via_raw.append("raw")
        raise AssertionError("raw model command must not run")

    monkeypatch.setattr(daemon, "_build_implementation_command", forbid_raw)

    result = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=workspace,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert result["raw_model_command_invoked"] is False
    assert result["typed_packet_route_only"] is True
    assert result["returncode"] == 0
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert result["binding"] is not None
    assert result["pending"] is False
    assert writes_via_raw == []

    applied = (workspace / PATH).read_text(encoding="utf-8")
    assert "production-route-applied" in applied

    events = _events(daemon)
    assert any(item.get("type") == PRODUCTION_PROVIDER_ROUTE_EVENT for item in events)
    assert any(
        item.get("type") == MODEL_ASSISTED_PROVIDER_ROUTE_EVENT for item in events
    )
    assert any(
        item.get("type") == PRODUCTION_PROVIDER_ROUTE_BINDING_EVENT for item in events
    )
    production = next(
        item for item in events if item.get("type") == PRODUCTION_PROVIDER_ROUTE_EVENT
    )
    assert production["typed_packet_route_only"] is True
    assert production["raw_model_command_invoked"] is False
    assert production["provider"]
    assert production["packet"]
    assert production["review_chain"]
    assert production["provider_receipt"]


def _admitted_file_proposal(*entries: tuple[str, str]) -> SimpleNamespace:
    paths = [path for path, _content in entries]
    return SimpleNamespace(
        admitted=True,
        payload={
            "proposal": {
                "declared_paths": paths,
                "files": [
                    {"path": path, "content": content}
                    for path, content in entries
                ],
            }
        },
    )


def test_production_writer_rejects_path_aliases_symlinks_and_nested_repositories(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    workspace = daemon.repo_root
    target = workspace / PATH
    original = target.read_bytes()

    writer = daemon._make_production_workspace_writer(
        workspace,
        task=_task(),
        expected_lease_id="lease:writer-paths",
    )
    alias = _admitted_file_proposal((f"./{PATH}", "alias\n"))
    with pytest.raises(RuntimeError, match="canonical relative path"):
        writer(alias, "lease:writer-paths")
    assert target.read_bytes() == original

    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink write target"):
        writer(_admitted_file_proposal((PATH, "escaped\n")), "lease:writer-paths")
    assert outside.read_text(encoding="utf-8") == "outside\n"

    nested_path = "nested/target.py"
    nested = workspace / "nested"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    (nested / "target.py").write_text("nested baseline\n", encoding="utf-8")
    nested_writer = daemon._make_production_workspace_writer(
        workspace,
        task=_task(outputs=[nested_path]),
        expected_lease_id="lease:nested-repo",
    )
    with pytest.raises(RuntimeError, match="nested repository path"):
        nested_writer(
            _admitted_file_proposal((nested_path, "nested replacement\n")),
            "lease:nested-repo",
        )
    assert (nested / "target.py").read_text(encoding="utf-8") == (
        "nested baseline\n"
    )


def test_production_writer_rolls_back_all_files_after_partial_replace_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    workspace = daemon.repo_root
    second_path = "second.py"
    first = workspace / PATH
    second = workspace / second_path
    second.write_text("second baseline\n", encoding="utf-8")
    first_before = first.read_bytes()
    second_before = second.read_bytes()
    writer = daemon._make_production_workspace_writer(
        workspace,
        task=_task(outputs=[PATH, second_path]),
        expected_lease_id="lease:transaction",
    )

    real_replace = os.replace
    failed = False

    def fail_second_once(source, destination):
        nonlocal failed
        if Path(destination) == second and not failed:
            failed = True
            raise OSError("injected second replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_once)
    with pytest.raises(RuntimeError, match="transactional file replacement failed"):
        writer(
            _admitted_file_proposal(
                (PATH, "first replacement\n"),
                (second_path, "second replacement\n"),
            ),
            "lease:transaction",
        )

    assert failed
    assert first.read_bytes() == first_before
    assert second.read_bytes() == second_before
    assert not list(workspace.rglob(".production-provider-write-*"))


def test_production_writer_requires_one_exact_declared_representation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    workspace = daemon.repo_root
    target = workspace / PATH
    original = target.read_bytes()
    writer = daemon._make_production_workspace_writer(
        workspace,
        task=_task(),
        expected_lease_id="lease:representation",
    )
    ambiguous = SimpleNamespace(
        admitted=True,
        payload={
            "proposal": {
                "declared_paths": [PATH],
                "files": [{"path": PATH, "content": "replacement\n"}],
                "patch": f"diff --git a/{PATH} b/{PATH}\n",
            }
        },
    )
    with pytest.raises(RuntimeError, match="both file replacements and a patch"):
        writer(ambiguous, "lease:representation")

    extra_declared = SimpleNamespace(
        admitted=True,
        payload={
            "proposal": {
                "declared_paths": [PATH, "unused.py"],
                "files": [{"path": PATH, "content": "replacement\n"}],
            }
        },
    )
    with pytest.raises(RuntimeError, match="must exactly match replacements"):
        writer(extra_declared, "lease:representation")
    assert target.read_bytes() == original


def test_production_route_forbids_raw_implementation_command(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    with pytest.raises(RuntimeError, match="typed packet route"):
        daemon._build_implementation_command(daemon.repo_root, task=task)


def test_grok_cannot_self_review_on_production_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)

    def same(request):
        if request["role"] == ProviderRole.GROK_IMPLEMENT.value:
            return {
                "proposal": {
                    "patch": "x",
                    "declared_paths": [PATH],
                    "files": [{"path": PATH, "content": "x\n"}],
                }
            }
        return {"decision": "approve"}

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=same,
        codex_provider=same,
        admission_gate=_accept,
    )
    assert result["route_result"].status is RouteStatus.REJECTED
    assert (
        result["route_result"].reason_code
        == ProviderReason.SELF_REVIEW_FORBIDDEN.value
    )
    assert result["binding"] is None
    assert result["pending"] is True
    assert daemon.model_assisted_authoritative_completion_allowed(
        result["route_result"]
    ) is False


def test_codex_receives_only_bounded_proposal_evidence_slice(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}

    def codex(request):
        seen["input"] = request["provider_input"]
        assert set(request["provider_input"]) == {
            "admitted_implementation_proposal",
            "evidence_slice",
        }
        slice_ = request["provider_input"]["evidence_slice"]
        assert "packet_id" in slice_
        assert "scope" in slice_
        assert "acceptance" in slice_
        assert "goal_ids" in slice_
        # Full goal corpus / counterexample bodies must not appear.
        encoded = json.dumps(request["provider_input"], sort_keys=True)
        assert "counterexample" not in encoded
        assert "repository_corpus" not in encoded
        return {"decision": "approve", "findings": []}

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=_grok,
        codex_provider=codex,
        admission_gate=_accept,
    )
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert "admitted_implementation_proposal" in seen["input"]


def test_applied_patch_and_merge_bind_to_admitted_review_chain(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    result = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:sca-615:1",
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )
    binding = result["binding"]
    assert binding is not None
    payload = binding.to_dict()
    assert payload["schema"] == PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA
    assert payload["provider_result_admitted"] is True
    assert payload["review_presence"] == ReviewPresence.INDEPENDENT.value
    assert payload["write_performed"] is True
    assert payload["writer_lease_id"] == "lease:sca-615:1"
    assert payload["receipt_id"]
    assert payload["review_chain_digest"]
    assert payload["selected_proposal_digest"]
    assert payload["completion_authoritative"] is False

    # Merge metadata carries the same admitted review-chain binding.
    assert daemon._last_production_review_chain_binding is binding
    metadata_probe = {
        "task_id": task.task_id,
    }
    # Simulate the enqueue attachment path.
    production_binding = daemon._last_production_review_chain_binding
    assert production_binding.task_id == task.task_id
    metadata_probe["admitted_review_chain_binding"] = production_binding.to_dict()
    assert metadata_probe["admitted_review_chain_binding"]["receipt_id"] == (
        binding.receipt_id
    )


@pytest.mark.parametrize(
    "kind",
    ["absent", "degraded", "stale", "cross_task"],
)
def test_absent_degraded_stale_cross_task_receipts_remain_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()

    if kind == "absent":
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=False,
            grok_provider=_grok,
            admission_gate=_accept,
        )
        assert result["route_result"].review_presence == ReviewPresence.ABSENT.value
        assert result["pending"] is True
        assert result["binding"] is None
        assert result["disposition"] is ProductionReceiptDisposition.PENDING_ABSENT
    elif kind == "degraded":
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=False,
            grok_provider=_grok,
            codex_provider=lambda _request: (_ for _ in ()).throw(
                RuntimeError("review unavailable")
            ),
            admission_gate=_accept,
        )
        assert result["route_result"].review_presence == ReviewPresence.DEGRADED.value
        assert result["pending"] is True
        assert result["binding"] is None
        assert result["disposition"] is ProductionReceiptDisposition.PENDING_DEGRADED
    elif kind == "stale":
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )
        disposition, reason = evaluate_production_provider_receipt(
            result["receipt"],
            expected_task_id=task.task_id,
            expected_snapshot_id="git-commit:other",
            current_snapshot_id="git-commit:other",
        )
        assert disposition is ProductionReceiptDisposition.PENDING_STALE
        assert reason == ProviderReason.RECEIPT_STALE.value
        assert daemon.production_provider_receipt_allows_merge(
            result["receipt"],
            expected_task_id=task.task_id,
            expected_snapshot_id="git-commit:other",
        ) is False
        return
    else:  # cross_task
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )
        disposition, reason = evaluate_production_provider_receipt(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=SNAPSHOT,
        )
        assert disposition is ProductionReceiptDisposition.PENDING_CROSS_TASK
        assert reason == ProviderReason.RECEIPT_CROSS_TASK.value
        assert daemon.production_provider_receipt_allows_merge(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=SNAPSHOT,
        ) is False
        return

    assert daemon.model_assisted_authoritative_completion_allowed(
        result["route_result"]
    ) is False
    pending_events = [
        item
        for item in _events(daemon)
        if item.get("type") == PRODUCTION_PROVIDER_ROUTE_PENDING_EVENT
    ]
    assert pending_events
    assert all(item.get("pending") is True for item in pending_events)
    assert all(item.get("completion_authoritative") is False for item in pending_events)


def test_deterministic_only_tasks_invoke_no_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        task_id="SCA-DET",
        metadata={"Provider role": "deterministic-only"},
    )
    assert daemon._task_uses_typed_local_execution(task) is True
    assert daemon._production_provider_route_enabled(task) is False
    with pytest.raises(RuntimeError, match="deterministic-only"):
        daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )


def test_complete_provider_contract_infers_production_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE",
        raising=False,
    )

    assert daemon._production_provider_route_enabled(_task()) is True


@pytest.mark.parametrize(
    "provider_role",
    [
        "grok-implement",
        "codex-review",
        "grok-implement, codex-review, unknown-provider",
    ],
)
def test_partial_or_unknown_provider_contract_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    provider_role: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE",
        raising=False,
    )
    task = _task(metadata={"Provider role": provider_role})

    with pytest.raises(
        ImplementationRetryDeferred,
        match="provider role must be exactly",
    ):
        daemon._production_provider_route_enabled(task)


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"outputs": []}, "outputs"),
        ({"validation": []}, "validation"),
        ({"acceptance": ""}, "acceptance"),
        (
            {
                "metadata": {
                    "Provider role": "grok-implement, codex-review",
                }
            },
            "positive context budget tokens",
        ),
    ],
)
def test_incomplete_production_contract_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    missing: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE",
        raising=False,
    )

    with pytest.raises(ImplementationRetryDeferred, match=missing):
        daemon._production_provider_route_enabled(_task(**overrides))


def test_complete_contract_cannot_silently_downgrade_when_route_disabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE",
        "0",
    )

    with pytest.raises(
        ImplementationRetryDeferred,
        match="route is disabled",
    ):
        daemon._production_provider_route_enabled(_task())


def test_no_provider_receives_repository_corpus() -> None:
    seen: list[dict[str, Any]] = []

    def grok(request):
        seen.append(request.to_dict())
        assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
        return {
            "proposal": {
                "patch": "bounded",
                "declared_paths": [PATH],
            }
        }

    def codex(request):
        seen.append(request.to_dict())
        assert request["role"] == ProviderRole.CODEX_REVIEW.value
        return {"decision": "approve"}

    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
        validation_commands=["true"],
        acceptance_criteria="no corpus",
    )
    # Ensure broad keys are rejected before any provider call.
    with pytest.raises(Exception):
        build_production_contract_packet(
            task_id="SCA-615",
            snapshot_id=SNAPSHOT,
            write_paths=[PATH],
            extra_goal={"repository_corpus": "entire-tree"},
        )

    # Distinct callables: Grok must never self-review.
    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=_accept,
    )
    result = router.route(packet, current_snapshot_id=SNAPSHOT)
    assert result.status is RouteStatus.SUCCEEDED
    encoded = json.dumps(seen, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    assert "workspace_path" not in encoded
    assert "full_repository" not in encoded
    # Codex path must not include the implementer's full contract packet.
    codex_request = next(
        item for item in seen if item["role"] == ProviderRole.CODEX_REVIEW.value
    )
    assert "contract_packet" not in codex_request["provider_input"]


def test_bind_applied_patch_requires_independent_admitted_review() -> None:
    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
    )
    router = ImplementationProviderRouter(
        grok_provider=_grok,
        admission_gate=_accept,
    )
    fallback = router.route(packet, current_snapshot_id=SNAPSHOT)
    assert bind_applied_patch_to_review_chain(fallback) is None

    full = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda _proposal, _lease: None,
    ).route(
        packet,
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:bound",
    )
    binding = bind_applied_patch_to_review_chain(
        full,
        writer_lease_id="lease:bound",
    )
    assert binding is not None
    assert binding.write_performed is True
    assert binding.writer_lease_id == "lease:bound"


def test_production_evaluation_artifact_exists_and_covers_acceptance() -> None:
    assert EVALUATION_PATH.exists(), f"missing evaluation artifact: {EVALUATION_PATH}"
    payload = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA
    assert payload["interface"] == PRODUCTION_PROVIDER_ROUTE_INTERFACE
    assert SCAEV615ROUTE in payload["evidence"]["requirement_ids"]
    acceptance = payload["acceptance"]
    assert acceptance["typed_packet_route_only"] is True
    assert acceptance["grok_cannot_self_review"] is True
    assert acceptance["codex_bounded_slice_only"] is True
    assert acceptance["apply_merge_bound_to_review_chain"] is True
    assert acceptance["deterministic_only_no_model"] is True
    assert acceptance["no_repository_corpus"] is True
    assert payload["production_route"]["raw_model_command_forbidden"] is True
    assert payload["corpus_isolation"]["provider_receives_repository_corpus"] is False
    assert payload["deterministic_only"]["invokes_no_model"] is True
    case_ids = {item.get("id") for item in payload.get("cases") or []}
    assert "happy-path-admitted" in case_ids
    assert "absent-review-pending" in case_ids
    assert "cross-task-receipt-pending" in case_ids
    assert "stale-receipt-pending" in case_ids


def test_daemon_builds_bounded_production_packet(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    packet = daemon.build_production_contract_packet_for_task(
        _task(),
        snapshot_id=SNAPSHOT,
        attempt=2,
    )
    assert isinstance(packet, ProductionContractPacket)
    assert packet.task_id == "SCA-615"
    assert packet.snapshot_id == SNAPSHOT
    payload = dict(packet.provider_input_payload)
    assert payload["authority"]["completion_authoritative"] is False
    assert PATH in payload["scope"]["write_paths"]
    assert "repository_corpus" not in json.dumps(payload)


def test_build_production_provider_route_evaluation_helper() -> None:
    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
    )
    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda _p, _l: None,
    ).route(
        packet,
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:eval",
    )
    binding = bind_applied_patch_to_review_chain(result, writer_lease_id="lease:eval")
    evaluation = build_production_provider_route_evaluation(
        route_result=result,
        binding=binding,
        deterministic_only_model_calls=0,
        raw_model_command_invoked=False,
        corpus_exposed_to_provider=False,
    )
    assert evaluation["schema"] == PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA
    assert evaluation["acceptance"]["typed_packet_route_only"] is True
    assert evaluation["route_result"]["provider_result_admitted"] is True
    assert evaluation["evaluation_id"]
