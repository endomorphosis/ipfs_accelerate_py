"""Adversarial regressions for the production provider trust boundary.

These tests intentionally exercise the points where an otherwise well-formed
model response could cross into a repository write, validation, commit, or
signed acceptance.  Provider output is never itself authority.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ImplementationProviderRouter,
    ProviderBounds,
    ProviderReason,
    ProviderRequest,
    ProviderRole,
    ProviderRoutingError,
    RouteStatus,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PRODUCTION_PROVIDER_ROUTE_EVENT,
    ImplementationRetryDeferred,
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_provider_cli import (
    _native_cli_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import (
    ProductionProviderReviewAuthority,
    trusted_public_key_from_private_path,
    verify_production_provider_review_attestation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    BoundProductionCLIProvider,
    ProductionCLIProviderPolicy,
    build_production_cli_provider_pair,
)

SNAPSHOT = "git-commit:provider-security-fixture"
TARGET_PATH = "src/target.py"


def _packet(*, task_id: str = "SEC-001", path: str = TARGET_PATH):
    return build_production_contract_packet(
        task_id=task_id,
        snapshot_id=SNAPSHOT,
        write_paths=[path],
        validation_commands=[f"python -m py_compile {path}"],
        acceptance_criteria="the exact reviewed candidate passes validation",
    )


def _grok_proposal(*, content: str = "VALUE = 'proposal-a'\n") -> dict[str, Any]:
    return {
        "proposal": {
            "declared_paths": [TARGET_PATH],
            "files": [{"path": TARGET_PATH, "content": content}],
        }
    }


def _admit(proposal: Any) -> dict[str, Any]:
    return {
        "accepted": True,
        "reason_code": f"admitted:{proposal.role.value}",
    }


@pytest.mark.parametrize(
    "review_response",
    [
        pytest.param({}, id="missing"),
        pytest.param({"decision": "APPROVE"}, id="upper-case"),
        pytest.param({"decision": "Approve"}, id="title-case"),
        pytest.param({"decision": " approve "}, id="whitespace-confused"),
        pytest.param({"decision": "approved"}, id="unknown-synonym"),
        pytest.param({"decision": "allow"}, id="unknown-decision"),
    ],
)
def test_unknown_missing_or_case_confused_codex_decision_never_writes(
    review_response: dict[str, Any],
) -> None:
    writes: list[Any] = []
    result = ImplementationProviderRouter(
        grok_provider=lambda _request: _grok_proposal(),
        codex_provider=lambda _request: review_response,
        admission_gate=_admit,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:security:decision",
    )

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.PROVIDER_RESPONSE_MALFORMED.value
    assert result.write_performed is False
    assert writes == []


@pytest.mark.parametrize("decision", ["repair", "replace"])
def test_codex_authored_repair_requires_another_independent_review_before_write(
    decision: str,
) -> None:
    """The reviewer cannot independently approve bytes it authored itself."""

    writes: list[Any] = []

    def codex(_request: ProviderRequest) -> dict[str, Any]:
        return {
            "decision": decision,
            "findings": [],
            "proposal": {
                "declared_paths": [TARGET_PATH],
                "files": [
                    {
                        "path": TARGET_PATH,
                        "content": "VALUE = 'codex-authored-final-bytes'\n",
                    }
                ],
            },
        }

    result = ImplementationProviderRouter(
        grok_provider=lambda _request: _grok_proposal(),
        codex_provider=codex,
        admission_gate=_admit,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:security:self-review",
    )

    assert result.write_performed is False
    assert writes == []
    assert bind_applied_patch_to_review_chain(result) is None


def test_codex_quota_exhaustion_applies_admitted_grok_proposal() -> None:
    """When independent Codex review cannot run for capacity, apply Grok.

    Completions remain non-authoritative; independent review is still required
    for formal merge admission. Repository effects may still land so the
    supervisor can complete implement/validate work while Codex is offline.
    """

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
        ProviderQuotaError,
    )

    writes: list[Any] = []

    def codex(_request: ProviderRequest) -> dict[str, Any]:
        raise ProviderQuotaError(
            "legacy_codex_usage_capacity_exhausted",
            reason_code=ProviderReason.CODEX_QUOTA_EXHAUSTED.value,
        )

    result = ImplementationProviderRouter(
        grok_provider=lambda _request: _grok_proposal(
            content="VALUE = 'capacity-recovery'\n"
        ),
        codex_provider=codex,
        admission_gate=_admit,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:security:codex-quota",
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert (
        result.reason_code == ProviderReason.CODEX_QUOTA_EXHAUSTED.value
    )
    assert result.write_performed is True
    assert result.provider_result_admitted is True
    assert result.completion_authoritative is False
    assert len(writes) == 1
    # Formal merge binding still requires independent review.
    assert bind_applied_patch_to_review_chain(result) is None


def test_codex_quota_recovery_skips_reviewed_effect_and_reaches_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity-recovery writes must not fail on reviewed-effect capture.

    Live wave0 previously admitted Grok, wrote files, then raised
    ``production reviewed effect capture failed`` because capture hard-requires
    Codex approve. That aborted the attempt and cleaned the worktree. Capture
    must be skipped so validation/commit can complete non-authoritatively.
    """

    daemon = _daemon_for_ephemeral_handoff(tmp_path, monkeypatch)
    task = _production_task()
    calls: list[str] = []
    committed: list[str] = []
    queued_requests: list[Any] = []
    policy = daemon.production_provider_policy
    assert isinstance(policy, ProductionCLIProviderPolicy)

    def seed(worktree_path: Path, _branch: str, *, task: Any = None) -> str:
        _git(
            daemon.repo_root,
            "worktree",
            "add",
            "-b",
            _branch,
            str(worktree_path),
            "HEAD",
        )
        return _git(worktree_path, "rev-parse", "HEAD")

    def validate(workspace: Path, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("validation")
        assert (workspace / TARGET_PATH).read_text(encoding="utf-8") == (
            "VALUE = 'capacity-recovery-daemon'\n"
        )
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        }

    real_commit = daemon._commit_worktree_changes

    def commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("commit")
        result = real_commit(*args, **kwargs)
        committed.append(str(result.get("commit") or ""))
        return result

    real_enqueue = daemon._enqueue_validated_worktree

    def enqueue(**kwargs: Any) -> dict[str, Any]:
        calls.append("enqueue")
        assert kwargs["implementation_commit"] == committed[-1]
        assert kwargs["validation_result"]["passed"] is True
        # Capacity recovery has no reviewed-effect binding.
        assert daemon._last_production_reviewed_effect_binding is None
        return real_enqueue(**kwargs)

    real_queue_enqueue = daemon.merge_queue.enqueue

    def queue_enqueue(**kwargs: Any) -> Any:
        request = real_queue_enqueue(**kwargs)
        queued_requests.append(request)
        return request

    def invoke(_prompt: str, config: Any) -> tuple[str, LlmChildResultEnvelope]:
        if config.provider == policy.codex_provider:
            raise RuntimeError(
                "legacy_codex_usage_capacity_exhausted: "
                "You've hit your usage limit"
            )
        encoded = json.dumps(
            _grok_proposal(content="VALUE = 'capacity-recovery-daemon'\n"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return encoded, LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=config.request_id,
            attempt=config.attempt,
            idempotency_key=config.idempotency_key,
            status="ok",
            effective_provider=str(config.provider or ""),
            text_chars=len(encoded),
            exit_code=0,
        )

    grok, codex = build_production_cli_provider_pair(policy, invoker=invoke)

    monkeypatch.setattr(daemon, "_create_seeded_worktree", seed)
    monkeypatch.setattr(
        daemon,
        "_production_landed_task_guard_for_workspace",
        lambda *_args, **_kwargs: {
            "guarded": False,
            "action": "new_implementation_route_allowed",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_require_implementation_protected_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_finalize_implementation_protected_path_fence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_prepare_worktree_for_validation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(daemon, "_run_validation_with_candidate_binding", validate)
    monkeypatch.setattr(daemon, "_commit_worktree_changes", commit)
    monkeypatch.setattr(daemon, "_enqueue_validated_worktree", enqueue)
    monkeypatch.setattr(daemon.merge_queue, "enqueue", queue_enqueue)
    monkeypatch.setattr(
        daemon,
        "_consume_one_merge_candidate",
        lambda: {"status": "deferred", "reason": "test_consumer_disabled"},
    )
    monkeypatch.setattr(
        daemon,
        "_record_failed_attempt_retry_context",
        lambda *_args, **_kwargs: None,
    )
    daemon._production_grok_provider = grok
    daemon._production_codex_provider = codex

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=1,
        started_at=datetime.now(UTC).isoformat(),
        log_path=daemon.state_path.parent / "implementation.log",
        prompt="capacity recovery production route",
    )

    assert result["returncode"] == 0
    assert result["implementation_commit"] == committed[-1]
    assert _git(daemon.repo_root, "cat-file", "-t", committed[-1]) == "commit"
    assert result["merge_result"]["queued"] is True
    assert calls == ["validation", "commit", "enqueue"]
    assert len(queued_requests) == 1
    queued_metadata = queued_requests[0].metadata
    # Non-authoritative capacity recovery: no reviewed-effect / attestation.
    assert "production_reviewed_effect_binding" not in queued_metadata
    assert "provider_review_attestation" not in queued_metadata
    assert daemon._production_capacity_recovery_write_active() is True
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        "production_reviewed_effect_skipped_capacity_recovery" in json.dumps(event)
        for event in events
    ), "expected capacity-recovery skip event"


def test_grok_json_quota_defers_without_review_write_or_attempt_charge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon_for_ephemeral_handoff(tmp_path, monkeypatch)
    task = _production_task()
    policy = daemon.production_provider_policy
    assert isinstance(policy, ProductionCLIProviderPolicy)
    provider_calls: list[str] = []
    writes: list[Any] = []

    def seed(worktree_path: Path, branch: str, *, task: Any = None) -> str:
        _git(
            daemon.repo_root,
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree_path),
            "HEAD",
        )
        return _git(worktree_path, "rev-parse", "HEAD")

    def invoke(_prompt: str, config: Any) -> tuple[str, LlmChildResultEnvelope]:
        provider_calls.append(str(config.provider or ""))
        if config.provider != policy.grok_provider:
            raise AssertionError("Codex review must not run after Grok quota")
        inner = json.dumps(
            {
                "message": (
                    "API error (status 402 Payment Required): "
                    "Grok Build usage balance exhausted"
                ),
                "http_status": 402,
            },
            indent=2,
        )
        stdout = json.dumps(
            {"type": "error", "message": "Internal error: " + inner},
            separators=(",", ":"),
        ).encode("utf-8")
        raise _native_cli_failure(
            ["grok", "--output-format", "json"],
            return_code=1,
            stdout=bytearray(stdout),
            stderr=bytearray(b"ignored untrusted diagnostic"),
        )

    grok, codex = build_production_cli_provider_pair(policy, invoker=invoke)
    monkeypatch.setattr(daemon, "_create_seeded_worktree", seed)
    monkeypatch.setattr(
        daemon,
        "_production_landed_task_guard_for_workspace",
        lambda *_args, **_kwargs: {
            "guarded": False,
            "action": "new_implementation_route_allowed",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_require_implementation_protected_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_finalize_implementation_protected_path_fence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_make_production_workspace_writer",
        lambda *_args, **_kwargs: (
            lambda proposal, lease: writes.append((proposal, lease))
        ),
    )
    daemon._production_grok_provider = grok
    daemon._production_codex_provider = codex

    state = PortalTaskState()
    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at=datetime.now(UTC).isoformat(),
        log_path=daemon.state_path.parent / "grok-quota.log",
        prompt="bounded quota deferral regression",
    )

    assert provider_calls == [policy.grok_provider]
    assert writes == []
    assert result["deferred"] is True
    assert result["reason"] == "provider_capacity_exhausted"
    assert result["attempt_consumed"] is False
    assert result["cleanup_result"]["cleaned"] is True
    assert result["cleanup_result"]["lifecycle_finalize"]["finalized"] is True
    recovered = PortalTaskState.load(daemon.state_path)
    assert recovered.implementation_attempts == {}
    assert recovered.implementation_attempts_by_cid == {}
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    route_event = next(
        event for event in events if event["type"] == PRODUCTION_PROVIDER_ROUTE_EVENT
    )
    assert route_event["reason_code"] == ProviderReason.GROK_QUOTA_EXHAUSTED.value


def _provider_request(role: ProviderRole) -> ProviderRequest:
    prompt = json.dumps(
        {"role": role.value, "task_id": "SEC-001", "provider_input": {}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ProviderRequest(
        role=role,
        packet_id="packet:security",
        snapshot_id=SNAPSHOT,
        task_id="SEC-001",
        payload={},
        bounds=ProviderBounds(),
        prompt=prompt,
        prompt_tokens=16,
    )


class _ChildReceiptWithoutExitCode:
    """Protocol-shaped child receipt with the security-critical field absent."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def to_dict(self) -> dict[str, Any]:
        payload = LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=self._config.request_id,
            attempt=self._config.attempt,
            idempotency_key=self._config.idempotency_key,
            status="ok",
            effective_provider=str(self._config.provider or ""),
            text_chars=1,
            exit_code=0,
        ).to_dict()
        payload.pop("exit_code")
        return payload


def test_missing_child_exit_code_fails_as_an_unbound_execution_receipt() -> None:
    policy = ProductionCLIProviderPolicy()
    provider = BoundProductionCLIProvider(
        policy=policy,
        role=ProviderRole.GROK_IMPLEMENT,
        provider_name=policy.grok_provider,
        model_name=policy.grok_model,
        invoker=lambda _prompt, config: (
            json.dumps(_grok_proposal()),
            _ChildReceiptWithoutExitCode(config),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(RuntimeError, match="execution receipt is not bound"):
        provider(_provider_request(ProviderRole.GROK_IMPLEMENT))


def test_concurrent_shared_review_key_creation_has_one_stable_authority(
    tmp_path: Path,
) -> None:
    """Independent lanes must never observe a partial or divergent key."""

    key_path = tmp_path / "bundle-state" / "shared-review.ed25519"
    release_path = tmp_path / "release-workers"
    repo_root = Path(__file__).resolve().parents[2]
    worker = """
import sys
import time
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import ProductionProviderReviewAuthority
key_path = Path(sys.argv[1])
release_path = Path(sys.argv[2])
deadline = time.monotonic() + 20
while not release_path.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('concurrency release timed out')
    time.sleep(0.001)
authority = ProductionProviderReviewAuthority.load_or_create(key_path)
print(authority.issuer_key_id, flush=True)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repo_root), env.get("PYTHONPATH", "")) if item
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(key_path), str(release_path)],
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(12)
    ]
    release_path.write_text("go\n", encoding="utf-8")

    outcomes = [child.communicate(timeout=30) for child in children]
    failures = [
        {"returncode": child.returncode, "stdout": out, "stderr": err}
        for child, (out, err) in zip(children, outcomes, strict=True)
        if child.returncode != 0
    ]
    assert failures == []
    issuer_ids = {out.strip() for out, _err in outcomes}
    assert len(issuer_ids) == 1
    assert "" not in issuer_ids
    assert key_path.stat().st_size == 32
    assert key_path.stat().st_mode & 0o777 == 0o600
    trusted_id, trusted_key = trusted_public_key_from_private_path(key_path)
    assert trusted_id == next(iter(issuer_ids))
    assert len(trusted_key) == 32


def _daemon_for_ephemeral_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TodoImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Security tasks\n", encoding="utf-8")
    target = repo / TARGET_PATH
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 'baseline'\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Provider Security Test")
    _git(repo, "config", "user.email", "provider-security@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "provider route baseline")
    policy = ProductionCLIProviderPolicy()
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SEC-",
        implement=True,
        use_ephemeral_worktree=True,
        worktree_root=repo / "worktrees",
        production_provider_policy=policy.name,
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_ALLOW_RAW_MODEL_COMMAND",
        raising=False,
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE", "1")
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda _kind, _payload, action: action(),
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    return daemon


def _production_task() -> PortalTask:
    return PortalTask(
        task_id="SEC-001",
        title="Typed production handoff",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider-security",
        outputs=[TARGET_PATH],
        validation=[f"python -m py_compile {TARGET_PATH}"],
        acceptance="the exact provider candidate is validated and committed",
        metadata={
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
        },
    )


def test_typed_production_route_rc0_reaches_validation_and_commit_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon_for_ephemeral_handoff(tmp_path, monkeypatch)
    task = _production_task()
    calls: list[str] = []
    committed: list[str] = []
    queued_requests: list[Any] = []

    def seed(worktree_path: Path, _branch: str, *, task: Any = None) -> str:
        _git(
            daemon.repo_root,
            "worktree",
            "add",
            "-b",
            _branch,
            str(worktree_path),
            "HEAD",
        )
        return _git(worktree_path, "rev-parse", "HEAD")

    def validate(workspace: Path, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("validation")
        assert (workspace / TARGET_PATH).read_text(encoding="utf-8") == ("VALUE = 'proposal-a'\n")
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        }

    real_commit = daemon._commit_worktree_changes

    def commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("commit")
        result = real_commit(*args, **kwargs)
        committed.append(str(result.get("commit") or ""))
        return result

    real_enqueue = daemon._enqueue_validated_worktree

    def enqueue(**kwargs: Any) -> dict[str, Any]:
        calls.append("enqueue")
        assert kwargs["implementation_commit"] == committed[-1]
        assert kwargs["validation_result"]["passed"] is True
        return real_enqueue(**kwargs)

    real_queue_enqueue = daemon.merge_queue.enqueue

    def queue_enqueue(**kwargs: Any) -> Any:
        request = real_queue_enqueue(**kwargs)
        queued_requests.append(request)
        return request

    policy = daemon.production_provider_policy
    assert isinstance(policy, ProductionCLIProviderPolicy)

    def invoke(_prompt: str, config: Any) -> tuple[str, LlmChildResultEnvelope]:
        output = (
            _grok_proposal()
            if config.provider == policy.grok_provider
            else {"decision": "approve", "findings": []}
        )
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":"))
        return encoded, LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=config.request_id,
            attempt=config.attempt,
            idempotency_key=config.idempotency_key,
            status="ok",
            effective_provider=str(config.provider or ""),
            text_chars=len(encoded),
            exit_code=0,
        )

    grok, codex = build_production_cli_provider_pair(policy, invoker=invoke)

    monkeypatch.setattr(daemon, "_create_seeded_worktree", seed)
    monkeypatch.setattr(
        daemon,
        "_production_landed_task_guard_for_workspace",
        lambda *_args, **_kwargs: {
            "guarded": False,
            "action": "new_implementation_route_allowed",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_require_implementation_protected_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_finalize_implementation_protected_path_fence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_prepare_worktree_for_validation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(daemon, "_run_validation_with_candidate_binding", validate)
    monkeypatch.setattr(daemon, "_commit_worktree_changes", commit)
    monkeypatch.setattr(daemon, "_enqueue_validated_worktree", enqueue)
    monkeypatch.setattr(daemon.merge_queue, "enqueue", queue_enqueue)
    monkeypatch.setattr(
        daemon,
        "_consume_one_merge_candidate",
        lambda: {"status": "deferred", "reason": "test_consumer_disabled"},
    )
    monkeypatch.setattr(
        daemon,
        "_record_failed_attempt_retry_context",
        lambda *_args, **_kwargs: None,
    )
    daemon._production_grok_provider = grok
    daemon._production_codex_provider = codex

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=1,
        started_at=datetime.now(UTC).isoformat(),
        log_path=daemon.state_path.parent / "implementation.log",
        prompt="typed production route",
    )

    assert result["returncode"] == 0
    assert result["implementation_commit"] == committed[-1]
    assert _git(daemon.repo_root, "cat-file", "-t", committed[-1]) == "commit"
    assert result["merge_result"]["queued"] is True
    assert calls == ["validation", "commit", "enqueue"]
    assert len(queued_requests) == 1
    queued_metadata = queued_requests[0].metadata
    effect = queued_metadata["production_reviewed_effect_binding"]
    attestation = queued_metadata["provider_review_attestation"]
    assert effect["implementation_commit"] == committed[-1]
    assert effect["implementation_tree_id"] == (
        "git-tree:" + _git(daemon.repo_root, "rev-parse", f"{committed[-1]}^{{tree}}")
    )
    assert attestation["reviewed_effect_binding_cid"] == effect["binding_id"]
    assert queued_metadata["provider_execution_receipt"]["write_performed"] is True


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _make_unrelated_commits(repo: Path) -> tuple[str, str, str, str]:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Provider Security Test")
    _git(repo, "config", "user.email", "provider-security@example.invalid")
    target = repo / TARGET_PATH
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (repo / "unrelated.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "proposal-a")
    target.write_text("VALUE = 'proposal-a'\n", encoding="utf-8")
    _git(repo, "add", TARGET_PATH)
    _git(repo, "commit", "-m", "proposal A")
    commit_a = _git(repo, "rev-parse", "HEAD")
    tree_a = "git-tree:" + _git(repo, "rev-parse", "HEAD^{tree}")

    _git(repo, "checkout", "-b", "unrelated-b", baseline)
    (repo / "unrelated.txt").write_text("unrelated B\n", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-m", "unrelated B")
    commit_b = _git(repo, "rev-parse", "HEAD")
    tree_b = "git-tree:" + _git(repo, "rev-parse", "HEAD^{tree}")
    return commit_a, tree_a, commit_b, tree_b


def _applied_route_for_proposal_a():
    result = ImplementationProviderRouter(
        grok_provider=lambda _request: _grok_proposal(),
        codex_provider=lambda _request: {
            "decision": "approve",
            "findings": [],
        },
        admission_gate=_admit,
        writer=lambda _proposal, _lease: None,
    ).route(
        _packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:security:attestation",
    )
    binding = bind_applied_patch_to_review_chain(result)
    assert result.status is RouteStatus.SUCCEEDED
    assert binding is not None
    return result, binding


def test_proposal_a_cannot_be_attested_or_accepted_for_unrelated_commit_tree_b(
    tmp_path: Path,
) -> None:
    _commit_a, _tree_a, commit_b, tree_b = _make_unrelated_commits(tmp_path / "repo")
    result, binding = _applied_route_for_proposal_a()
    unrelated_binding = replace(binding, implementation_commit=commit_b)
    authority = ProductionProviderReviewAuthority.generate()

    try:
        suspect = authority.issue(
            provider_receipt=result.provider_receipt,
            review_chain_binding=unrelated_binding,
            provider_policy_id=ProductionCLIProviderPolicy().policy_id,
            implementation_commit=commit_b,
            implementation_tree_id=tree_b,
            issued_at_ms=1_800_000_000_000,
            nonce="proposal-commit-substitution-0001",
        )
    except ValueError:
        return

    verification = verify_production_provider_review_attestation(
        suspect,
        trusted_public_keys={
            authority.issuer_key_id: authority.public_key_bytes,
        },
        provider_receipt=result.provider_receipt,
        review_chain_binding=unrelated_binding,
        expected_task_id="SEC-001",
        expected_snapshot_id=SNAPSHOT,
        expected_provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        expected_implementation_commit=commit_b,
        expected_implementation_tree_id=tree_b,
    )

    assert verification.admitted is False, (
        "a proposal/review digest was accepted for a Git commit and tree whose "
        "diff was never bound to that proposal"
    )


def test_existing_file_without_source_or_ast_binding_fails_before_provider_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Provider Context Test")
    _git(repo, "config", "user.email", "provider-context@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Context-bound tasks\n", encoding="utf-8")
    existing = repo / TARGET_PATH
    existing.parent.mkdir(parents=True)
    # Deliberately exceed the whole-file disclosure bound.  With no
    # operator-supplied qualified-symbol hints, the daemon cannot construct a
    # semantically sufficient AST slice and must stop before either provider.
    original = (
        "\n\n".join(
            f"def stable_component_{index:03d}(value: int) -> int:\n    return value + {index}\n"
            for index in range(256)
        )
        + "\n"
    )
    assert len(original.encode("utf-8")) > 8_192
    existing.write_text(original, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "existing implementation")
    head = _git(repo, "rev-parse", "HEAD")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SEC-",
        implement=True,
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_ALLOW_RAW_MODEL_COMMAND",
        raising=False,
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE", "1")
    task = PortalTask(
        task_id="SEC-EXISTING",
        title="Safely modify an existing implementation",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider-security",
        outputs=[TARGET_PATH],
        validation=[f"python -m py_compile {TARGET_PATH}"],
        acceptance="preserve existing behavior while adding the requested change",
        metadata={
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
            # Deliberately no qualified-symbol hints are supplied for the
            # large existing Python source.
        },
    )
    calls: list[str] = []

    def grok(_request: ProviderRequest) -> dict[str, Any]:
        calls.append("grok")
        return _grok_proposal(content="def hallucinated():\n    return None\n")

    def codex(_request: ProviderRequest) -> dict[str, Any]:
        calls.append("codex")
        return {"decision": "approve", "findings": []}

    outcome: dict[str, Any] | None = None
    deferred: Exception | None = None
    try:
        outcome = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=repo,
            snapshot_id=f"git-commit:{head}",
            apply=True,
            grok_provider=grok,
            codex_provider=codex,
            admission_gate=_admit,
        )
    except (ImplementationRetryDeferred, ProviderRoutingError) as exc:
        deferred = exc

    if isinstance(deferred, ProviderRoutingError):
        assert deferred.reason_code == "symbol_scope_required"

    fail_closed = deferred is not None or bool(
        outcome and outcome.get("returncode") != 0 and outcome.get("pending") is True
    )
    assert fail_closed is True
    assert calls == []
    assert existing.read_text(encoding="utf-8") == original
