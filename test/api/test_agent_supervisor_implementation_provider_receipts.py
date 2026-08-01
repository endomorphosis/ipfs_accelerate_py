"""SCA-228 model-assisted provider receipt and independent review wiring.

Proves that implementation-daemon model-assisted routes record nonempty
provider, packet, review-chain, and provider-receipt fields; that Grok cannot
self-review; that Codex receives only the bounded proposal/evidence slice; and
that absent or degraded review remains explicit and cannot satisfy
authoritative completion.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderReason,
    ProviderRole,
    ReviewPresence,
    RouteStatus,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    MODEL_ASSISTED_PROVIDER_ROUTE_EVENT,
    PortalTask,
    TodoImplementationDaemon,
)


SNAPSHOT = "git-tree:current"
PATH = "external/ipfs_accelerate/ipfs_accelerate_py/mcp/dispatch.py"


@dataclass(frozen=True)
class _Packet:
    packet_id: str = "packet:sca-228"
    snapshot_id: str = SNAPSHOT
    task_id: str = "SCA-228-fixture"
    implementable: bool = True
    payload: Mapping[str, Any] | None = None

    def assert_current(self, current_snapshot_id: str) -> None:
        if current_snapshot_id != self.snapshot_id:
            raise ValueError("stale")

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return self.payload or MappingProxyType(
            {
                "goal": {
                    "contract_ids": ["contract:repo.inspect"],
                    "obligation_ids": ["obligation:arguments"],
                    "counterexample": {
                        "data_label": "untrusted_repository_data",
                        "instruction_authority": False,
                        "value": {"expected": "string", "actual": "integer"},
                    },
                },
                "authority": {
                    "provider_semantic_authority": False,
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                },
                "scope": {
                    "read_paths": [PATH],
                    "write_paths": [PATH],
                },
                "acceptance": {
                    "validation_commands": ["python -m pytest test_contract.py -q"],
                    "reproof_commands": ["python -m proof.recheck obligation:arguments"],
                },
            }
        )


def _git(repo, *arguments: str) -> None:
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
    _git(repo, "config", "user.name", "Provider Receipt Test")
    _git(repo, "config", "user.email", "provider-receipt@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Provider receipt tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
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
        implementation_command="model-command-must-not-run",
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
    return daemon


def _task() -> PortalTask:
    return PortalTask(
        task_id="SCA-228",
        title="Wire bounded Grok implementation and independent Codex review receipts",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider-routing",
        outputs=[
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
        ],
        validation=[
            "python3 -m pytest external/ipfs_accelerate/test/api/test_agent_supervisor_implementation_provider_receipts.py -q"
        ],
        acceptance=(
            "Model-assisted implementation events contain nonempty provider, "
            "packet, review-chain, and provider-receipt fields"
        ),
        metadata={
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
        },
    )


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
    return {
        "proposal": {
            "patch": f"diff --git a/{PATH} b/{PATH}\n",
            "declared_paths": [PATH],
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
    return {"decision": "approve", "findings": []}


def test_daemon_route_event_has_nonempty_provider_packet_review_chain_and_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()

    route_result, event, receipt_path = daemon.route_model_assisted_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        task=task,
        attempt=1,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert route_result.status is RouteStatus.SUCCEEDED
    assert event["provider"] == ProviderRole.GROK_IMPLEMENT.value
    assert event["packet"]["packet_id"] == "packet:sca-228"
    assert event["packet"]["packet_cid"]
    assert int(event["packet"]["packet_bytes"]) > 0
    assert event["review_chain"]
    assert event["review_chain"][0]["role"] == ProviderRole.GROK_IMPLEMENT.value
    assert event["review_chain"][1]["role"] == ProviderRole.CODEX_REVIEW.value
    assert event["provider_receipt"]["receipt_id"]
    assert event["provider_receipt"]["provider"]
    assert event["provider_receipt"]["packet"]["packet_cid"]
    assert event["provider_receipt"]["review_chain"]
    assert event["completion_authoritative"] is False
    assert event["provider_result_admitted"] is True
    assert receipt_path.exists()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["receipt_id"] == event["receipt_id"]
    assert persisted["completion_authoritative"] is False

    events = _events(daemon)
    route_events = [
        item for item in events if item.get("type") == MODEL_ASSISTED_PROVIDER_ROUTE_EVENT
    ]
    assert len(route_events) == 1
    recorded = route_events[0]
    for field in ("provider", "packet", "review_chain", "provider_receipt"):
        assert recorded.get(field), f"{field} must be nonempty"
        assert recorded[field]


def test_daemon_rejects_grok_self_review(
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
                }
            }
        return {"decision": "approve"}

    route_result, event, _path = daemon.route_model_assisted_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        task=_task(),
        attempt=1,
        grok_provider=same,
        codex_provider=same,
        admission_gate=_accept,
    )

    assert route_result.status is RouteStatus.REJECTED
    assert route_result.reason_code == ProviderReason.SELF_REVIEW_FORBIDDEN.value
    assert event["provider_result_admitted"] is False
    assert event["completion_authoritative"] is False
    assert daemon.model_assisted_authoritative_completion_allowed(route_result) is False
    assert daemon.model_assisted_authoritative_completion_allowed(event) is False


def test_daemon_codex_slice_excludes_full_goal_corpus(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}

    def codex(request):
        seen["input"] = request["provider_input"]
        assert "contract_packet" not in request["provider_input"]
        slice_ = request["provider_input"]["evidence_slice"]
        encoded = json.dumps(slice_, sort_keys=True)
        assert "counterexample" not in encoded
        assert "untrusted_repository_data" not in encoded
        assert slice_["goal_ids"]["contract_ids"] == ["contract:repo.inspect"]
        return {"decision": "approve", "findings": []}

    route_result, _event, _path = daemon.route_model_assisted_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        task=_task(),
        attempt=1,
        grok_provider=_grok,
        codex_provider=codex,
        admission_gate=_accept,
    )

    assert route_result.status is RouteStatus.SUCCEEDED
    assert "admitted_implementation_proposal" in seen["input"]
    assert "evidence_slice" in seen["input"]


def test_daemon_absent_and_degraded_review_cannot_authoritatively_complete(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()

    absent, absent_event, _ = daemon.route_model_assisted_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        task=task,
        attempt=1,
        grok_provider=_grok,
        admission_gate=_accept,
    )
    assert absent.status is RouteStatus.FALLBACK
    assert absent.review_presence == ReviewPresence.ABSENT.value
    assert absent.provider_result_admitted is False
    assert absent_event["completion_authoritative"] is False
    assert absent_event["provider_result_admitted"] is False
    assert absent_event["review_chain"][-1]["status"] == "absent"
    assert daemon.model_assisted_authoritative_completion_allowed(absent) is False
    assert daemon.model_assisted_authoritative_completion_allowed(absent_event) is False

    degraded, degraded_event, _ = daemon.route_model_assisted_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        task=task,
        attempt=2,
        grok_provider=_grok,
        codex_provider=lambda _request: (_ for _ in ()).throw(
            RuntimeError("review unavailable")
        ),
        admission_gate=_accept,
    )
    assert degraded.status is RouteStatus.FALLBACK
    assert degraded.review_presence == ReviewPresence.DEGRADED.value
    assert degraded.provider_result_admitted is False
    assert degraded_event["completion_authoritative"] is False
    assert degraded_event["review_chain"][-1]["status"] == "degraded"
    assert daemon.model_assisted_authoritative_completion_allowed(degraded) is False

    authority_denied = [
        item
        for item in _events(daemon)
        if item.get("type") == "model_assisted_provider_authority_denied"
    ]
    assert len(authority_denied) >= 2
    assert all(item.get("completion_authoritative") is False for item in authority_denied)


def test_task_metadata_tracks_independent_codex_review_requirement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    assert daemon._task_declares_independent_codex_review(task) is True
    assert daemon._task_model_assisted_provider_roles(task) == (
        ProviderRole.GROK_IMPLEMENT.value,
        ProviderRole.CODEX_REVIEW.value,
    )
    local = PortalTask(
        task_id="SCA-DET",
        title="local",
        status="ready",
        completion="manual",
        priority="P2",
        track="static-analysis",
        metadata={"Provider role": "deterministic-only"},
    )
    assert daemon._task_declares_independent_codex_review(local) is False
    assert daemon._task_model_assisted_provider_roles(local) == ()
