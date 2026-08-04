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

import copy
import inspect
import json
import os
import subprocess
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    MAX_PROVIDER_JSON_DEPTH,
    MAX_PROVIDER_JSON_ITEMS,
    MAX_PROVIDER_PROMPT_TOKENS,
    PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA,
    PRODUCTION_PROVIDER_ROUTE_INTERFACE,
    PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA,
    SCAEV615ROUTE,
    ImplementationProviderRouter,
    ProductionContractPacket,
    ProductionReceiptDisposition,
    ProviderReason,
    ProviderRole,
    ProviderRoutingError,
    ReviewPresence,
    RouteStatus,
    VerifiedGrokQuotaExhaustion,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
    build_production_provider_route_evaluation,
    evaluate_production_provider_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    MODEL_ASSISTED_PROVIDER_ROUTE_EVENT,
    MODEL_ASSISTED_PROVIDER_REVIEW_PENDING_EVENT,
    PRODUCTION_PROVIDER_ROUTE_BINDING_EVENT,
    PRODUCTION_PROVIDER_ROUTE_EVENT,
    PRODUCTION_PROVIDER_ROUTE_PENDING_EVENT,
    ImplementationRetryDeferred,
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "swissknife_contract_assurance"
    / "evaluation"
    / "production-provider-route.json"
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


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _snapshot(daemon: TodoImplementationDaemon) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=daemon.repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return f"git-commit:{result.stdout.strip()}"


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
            "Context budget tokens": str(MAX_PROVIDER_PROMPT_TOKENS),
        },
    }
    payload.update(overrides)
    return PortalTask(**payload)


def _install_landed_route_preflight_evidence(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_leaf_diverged: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Install one real Git chain and exact durable preflight projections."""

    repo = daemon.repo_root
    implementation_commit = _git_output(repo, "rev-parse", "HEAD^{commit}")
    (repo / PATH).write_text("# denied implementation\n", encoding="utf-8")
    _git(repo, "add", PATH)
    _git(repo, "commit", "-m", "land denied implementation")
    merge_commit = _git_output(repo, "rev-parse", "HEAD^{commit}")
    merge_tree_id = "git-tree:" + _git_output(
        repo,
        "rev-parse",
        f"{merge_commit}^{{tree}}",
    )
    if task_leaf_diverged:
        (repo / PATH).write_text("# later repair\n", encoding="utf-8")
        _git(repo, "add", PATH)
        _git(repo, "commit", "-m", "repair task leaf")
    else:
        (repo / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
        _git(repo, "add", "unrelated.txt")
        _git(repo, "commit", "-m", "advance unrelated leaf")
    baseline_ref = _git_output(repo, "rev-parse", "HEAD^{commit}")
    baseline_tree_id = "git-tree:" + _git_output(
        repo,
        "rev-parse",
        f"{baseline_ref}^{{tree}}",
    )

    identity = daemon._identity_for_task(task)
    task_binding_id = implementation_daemon_module.post_merge_task_binding_id(
        task
    )
    origin_stream_id = implementation_daemon_module._event_stream_binding(
        daemon.events_path
    )[0]
    denial_id = "denial:landed-route-preflight"
    denial = {
        "denial_id": denial_id,
        "correction_origin_stream_id": origin_stream_id,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
        "repository_tree_id": merge_tree_id,
        "review_receipt_id": "review:landed-route-preflight",
        "diff_binding_id": "diff:landed-route-preflight",
        "source_event_id": "event:landed-route-preflight",
        "source_event_sequence": 17,
        "review_attempt": 1,
        "implementation_attempt": 1,
        "source_finding_count": 1,
        "included_finding_count": 1,
        "truncated": False,
    }
    feedback = {
        "feedback_binding_id": "feedback:landed-route-preflight",
        "findings": [{"finding_id": "finding:landed-route-preflight"}],
        "truncated": False,
    }
    authority_material = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "post-merge-correction-dispatch-authority@1"
        ),
        "authority_kind": "review_denial",
        "authority_id": "authority:landed-route-preflight",
        "authorized_attempt": 2,
        "durable_denial_id": denial_id,
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
        "task_binding_id": task_binding_id,
        "target_repository_id": daemon.merge_target_repository_id,
        "target_branch": daemon.resolved_merge_target_branch,
        "durable_authority_head_record_id": "record:ready",
        "durable_authority_head_ordinal": 3,
        "durable_authority_state_id": "state:ready",
        "origin_stream_id": origin_stream_id,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
        "repository_tree_id": merge_tree_id,
        "review_receipt_id": denial["review_receipt_id"],
        "diff_binding_id": denial["diff_binding_id"],
        "source_event_id": denial["source_event_id"],
        "source_event_sequence": denial["source_event_sequence"],
        "review_attempt": denial["review_attempt"],
        "source_implementation_attempt": denial["implementation_attempt"],
    }
    authority = {
        **authority_material,
        "authority_binding_id": implementation_daemon_module.content_identity(
            authority_material
        ),
    }
    durable_authority = {
        "authority_available": True,
        "denial_id": denial_id,
        "authority_kind": authority["authority_kind"],
        "authority_id": authority["authority_id"],
        "authorized_attempt": 2,
        "head_record_id": authority["durable_authority_head_record_id"],
        "head_ordinal": authority["durable_authority_head_ordinal"],
        "authority_state_id": authority["durable_authority_state_id"],
        "origin_stream_id": origin_stream_id,
    }
    landed_guard = {
        "guarded": True,
        "workspace_clean": True,
        "recovery_reason": "recovered_implementation_binding",
        "recovery_source": "strict-ledger",
        "landed_implementation_commit": implementation_commit,
        "landed_merge_commit": merge_commit,
        "landed_repository_tree_id": merge_tree_id,
        "baseline_ref": baseline_ref,
        "repository_tree_id": baseline_tree_id,
    }
    monkeypatch.setattr(
        daemon,
        "_verified_durable_post_merge_denial",
        lambda *_args, **_kwargs: denial,
    )
    monkeypatch.setattr(
        daemon,
        "_verified_complete_post_merge_denial_feedback",
        lambda *_args, **_kwargs: feedback,
    )
    monkeypatch.setattr(
        daemon.merge_queue,
        "verified_post_merge_correction_authority",
        lambda *_args, **_kwargs: durable_authority,
    )
    return authority, denial, feedback, durable_authority, landed_guard


@pytest.mark.parametrize(
    ("failure_class", "expected_reason"),
    [
        ("authority", "authority_invalid"),
        ("durable", "durable_authority_invalid"),
        ("workspace", "landed_workspace_invalid"),
        ("ancestry", "landed_ancestry_invalid"),
        ("git-comparison", "task_leaf_comparison_failed"),
    ],
)
def test_landed_route_preflight_rejects_non_leaf_failures_without_spending_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_class: str,
    expected_reason: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    authority, _denial, _feedback, durable, landed_guard = (
        _install_landed_route_preflight_evidence(
            daemon,
            task,
            monkeypatch,
            task_leaf_diverged=False,
        )
    )
    if failure_class == "authority":
        authority["target_branch"] = "tampered"
    elif failure_class == "durable":
        durable["authority_available"] = False
    elif failure_class == "workspace":
        landed_guard["workspace_clean"] = False
    elif failure_class == "ancestry":
        monkeypatch.setattr(
            daemon,
            "_git_ref_is_ancestor",
            lambda *_args, **_kwargs: False,
        )
    else:
        real_run = subprocess.run

        def fail_leaf_comparison(*args, **kwargs):
            command = args[0]
            if (
                command[:2] == ["git", "--literal-pathspecs"]
                and "diff" in command
            ):
                return subprocess.CompletedProcess(command, 128)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", fail_leaf_comparison)

    result = daemon._post_merge_correction_landed_route_candidate(
        task,
        attempt=2,
        authority=authority,
        landed_guard=landed_guard,
        workspace_path=daemon.repo_root,
    )

    assert result.disposition is (
        implementation_daemon_module
        ._PostMergeCorrectionLandedRouteDisposition.REJECTED
    )
    assert result.reason == expected_reason
    assert not result.candidate
    assert not _events(daemon)
    assert not daemon._sealed_post_merge_correction_routes
    assert not daemon._claimed_post_merge_correction_routes


def test_complete_feedback_authority_rejects_boolean_attempt_fail_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed attempt types return no feedback instead of raising at runtime."""

    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    identity = daemon._identity_for_task(task)
    material = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "post-merge-correction-dispatch-authority@1"
        ),
        "authority_kind": "review_denial",
        "authority_id": "authority:boolean-attempt",
        "authorized_attempt": True,
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
        "task_binding_id": (
            implementation_daemon_module.post_merge_task_binding_id(task)
        ),
        "complete_denial_feedback": {},
    }
    authority = {
        **material,
        "authority_binding_id": implementation_daemon_module.content_identity(
            material
        ),
    }

    assert daemon._validated_complete_denial_feedback_from_authority(
        task,
        attempt=1,
        authority=authority,
    ) == {}


@pytest.mark.parametrize(
    ("task_leaf_diverged", "expected_disposition"),
    [
        (False, "verified"),
        (True, "task_leaf_diverged"),
    ],
)
def test_landed_route_preflight_activates_one_sealed_capability(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    task_leaf_diverged: bool,
    expected_disposition: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    authority, denial, feedback, _durable, landed_guard = (
        _install_landed_route_preflight_evidence(
            daemon,
            task,
            monkeypatch,
            task_leaf_diverged=task_leaf_diverged,
        )
    )
    preflight = daemon._post_merge_correction_landed_route_candidate(
        task,
        attempt=2,
        authority=authority,
        landed_guard=landed_guard,
        workspace_path=daemon.repo_root,
    )
    assert preflight.disposition.value == expected_disposition
    candidate = dict(preflight.candidate)
    identity = daemon._identity_for_task(task)
    started_event = {
        "type": "implementation_started",
        "task_id": task.task_id,
        "attempt": 2,
        "event_id": "event:implementation-started",
        "sequence": 23,
        "post_merge_correction_landed_route_candidate_id": candidate[
            "route_candidate_id"
        ],
        "post_merge_correction_authority": authority,
    }
    consumption = {
        "record_kind": "denial_consumed",
        "record_id": "record:consumed",
        "denial_id": denial["denial_id"],
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
        "task_binding_id": (
            implementation_daemon_module.post_merge_task_binding_id(task)
        ),
        "attempt": 2,
        "parent_record_id": authority["durable_authority_head_record_id"],
        "detail": {
            "authority_kind": authority["authority_kind"],
            "authority_id": authority["authority_id"],
            "started_event_id": started_event["event_id"],
            "started_event_sequence": started_event["sequence"],
        },
    }
    consumed_state = {
        "authority_available": False,
        "complete_feedback_available": True,
        "state": "consumed",
        "head_record_id": consumption["record_id"],
    }
    monkeypatch.setattr(
        daemon.merge_queue,
        "verified_post_merge_correction_chain",
        lambda *_args, **_kwargs: (consumption,),
    )
    monkeypatch.setattr(
        daemon.merge_queue,
        "verified_post_merge_correction_authority",
        lambda *_args, **_kwargs: consumed_state,
    )

    activated = daemon._activate_post_merge_correction_landed_route(
        task,
        attempt=2,
        authority=authority,
        candidate=candidate,
        started_event=started_event,
        landed_guard=landed_guard,
    )

    assert activated["guarded"] is False
    assert activated["invoke_grok_implementation"] is True
    assert activated["invoke_codex_review"] is True
    assert activated["post_merge_correction_route"] == candidate
    assert len(daemon._sealed_post_merge_correction_routes) == 1
    assert not daemon._claimed_post_merge_correction_routes
    entry = next(iter(daemon._sealed_post_merge_correction_routes.values()))
    material = implementation_daemon_module._correction_route_material_snapshot(
        entry
    )
    assert material is not None
    assert material["authority_binding_id"] == authority["authority_binding_id"]
    assert material["consumption_record_id"] == consumption["record_id"]
    assert material["complete_denial_feedback_id"] == feedback[
        "feedback_binding_id"
    ]


def test_ordinary_correction_start_seals_capability_without_landed_candidate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumed ordinary correction cannot reach the private route unsealed."""

    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    authority, denial, feedback, _durable, _landed_guard = (
        _install_landed_route_preflight_evidence(
            daemon,
            task,
            monkeypatch,
            task_leaf_diverged=False,
        )
    )
    identity = daemon._identity_for_task(task)
    started_event = {
        "type": "implementation_started",
        "task_id": task.task_id,
        "attempt": 2,
        "event_id": "event:ordinary-correction-started",
        "sequence": 29,
        "post_merge_correction_authority": authority,
    }
    consumption = {
        "record_kind": "denial_consumed",
        "record_id": "record:ordinary-correction-consumed",
        "denial_id": denial["denial_id"],
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
        "task_binding_id": (
            implementation_daemon_module.post_merge_task_binding_id(task)
        ),
        "attempt": 2,
        "parent_record_id": authority["durable_authority_head_record_id"],
        "detail": {
            "authority_kind": authority["authority_kind"],
            "authority_id": authority["authority_id"],
            "started_event_id": started_event["event_id"],
            "started_event_sequence": started_event["sequence"],
        },
    }
    consumed_state = {
        "authority_available": False,
        "complete_feedback_available": True,
        "state": "consumed",
        "head_record_id": consumption["record_id"],
    }
    monkeypatch.setattr(
        daemon.merge_queue,
        "verified_post_merge_correction_chain",
        lambda *_args, **_kwargs: (consumption,),
    )
    monkeypatch.setattr(
        daemon.merge_queue,
        "verified_post_merge_correction_authority",
        lambda *_args, **_kwargs: consumed_state,
    )

    assert not daemon._seal_post_merge_correction_route_after_start(
        task,
        attempt=2,
        authority=authority,
        started_event=started_event,
        complete_feedback_id="feedback:wrong",
    )
    assert not daemon._sealed_post_merge_correction_routes

    sealed = daemon._seal_post_merge_correction_route_after_start(
        task,
        attempt=2,
        authority=authority,
        started_event=started_event,
        complete_feedback_id=feedback["feedback_binding_id"],
    )

    assert sealed["guarded"] is False
    assert sealed["post_merge_correction_reservation"]["record_id"] == (
        consumption["record_id"]
    )
    assert len(daemon._sealed_post_merge_correction_routes) == 1
    assert not daemon._claimed_post_merge_correction_routes
    assert not daemon._seal_post_merge_correction_route_after_start(
        task,
        attempt=2,
        authority=authority,
        started_event=started_event,
        complete_feedback_id=feedback["feedback_binding_id"],
    )


def _install_claimable_correction_route(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    *,
    attempt: int,
    suffix: str = "primary",
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Install one exact daemon-owned entry without fabricating durable state."""

    identity = daemon._identity_for_task(task)
    denial_id = f"denial:{suffix}"
    authority_material = {
        "schema": "test/post-merge-correction-authority@1",
        "task_id": task.task_id,
        "authorized_attempt": int(attempt),
        "durable_denial_id": denial_id,
        "authority_kind": "review_denial",
        "authority_id": f"authority:{suffix}",
    }
    authority = {
        **authority_material,
        "authority_binding_id": (
            implementation_daemon_module.content_identity(authority_material)
        ),
    }
    feedback = {
        "schema": "test/complete-post-merge-correction-feedback@1",
        "feedback_binding_id": f"feedback:{suffix}",
        "findings": [
            {
                "finding_id": f"finding:{suffix}",
                "message": "repair the exact denied effect",
            }
        ],
        "truncated": False,
    }
    started_event_id = f"event:started:{suffix}"
    started_sequence = 40 + len(suffix)
    consumption_record_id = f"record:consumed:{suffix}"
    capability_material = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "sealed-post-merge-correction-route@1"
        ),
        "task_id": task.task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "task_binding_id": (
            implementation_daemon_module.post_merge_task_binding_id(task)
        ),
        "attempt": int(attempt),
        "authority_binding_id": authority["authority_binding_id"],
        "durable_denial_id": denial_id,
        "complete_denial_feedback_id": feedback["feedback_binding_id"],
        "pre_consumption_head_record_id": f"record:ready:{suffix}",
        "consumption_record_id": consumption_record_id,
        "implementation_started_event_id": started_event_id,
        "implementation_started_event_sequence": started_sequence,
        "authority": authority,
    }
    capability_material["capability_id"] = (
        implementation_daemon_module.content_identity(capability_material)
    )
    capability = (
        implementation_daemon_module
        ._LivePostMergeCorrectionRouteCapability()
    )
    frozen, canonical = (
        implementation_daemon_module._freeze_correction_capability_json(
            capability_material
        )
    )
    entry = (
        implementation_daemon_module
        ._PostMergeCorrectionRouteRegistryEntry(
            capability=capability,
            material=frozen,
            canonical=canonical,
        )
    )
    key = implementation_daemon_module._correction_route_registry_key(
        capability_material
    )
    assert key is not None
    with daemon._post_merge_correction_route_registry_lock:
        daemon._sealed_post_merge_correction_routes[key] = entry
    reservation = {
        "durable_consumption_record_id": consumption_record_id,
        "implementation_started_event_id": started_event_id,
        "implementation_started_event_sequence": started_sequence,
    }
    return key, entry, feedback, reservation


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


@pytest.mark.parametrize(
    "field_name",
    [
        "correction_feedback",
        "post_merge_correction_authority",
        "post_merge_correction_route_capability",
    ],
)
def test_public_packet_route_rejects_all_correction_material(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    key, entry, feedback, _reservation = (
        _install_claimable_correction_route(
            daemon,
            task,
            attempt=2,
        )
    )
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=_snapshot(daemon),
        write_paths=task.outputs,
        read_paths=task.outputs,
    )
    payload = dict(packet.payload)
    payload[field_name] = (
        feedback
        if field_name == "correction_feedback"
        else entry.capability
    )
    injected_packet = ProductionContractPacket(
        packet_id=packet.packet_id,
        snapshot_id=packet.snapshot_id,
        task_id=packet.task_id,
        implementable=packet.implementable,
        payload=payload,
    )

    assert "_correction_route_capability" not in inspect.signature(
        daemon.route_model_assisted_contract_packet
    ).parameters
    with pytest.raises(RuntimeError, match="public.*rejects correction"):
        daemon.route_model_assisted_contract_packet(
            injected_packet,
            current_snapshot_id=packet.snapshot_id,
            task=task,
            attempt=2,
        )
    with daemon._post_merge_correction_route_registry_lock:
        assert daemon._sealed_post_merge_correction_routes.get(key) is entry
        assert daemon._claimed_post_merge_correction_routes == {}

    with pytest.raises(TypeError):
        daemon.route_model_assisted_contract_packet(
            packet,
            current_snapshot_id=packet.snapshot_id,
            task=task,
            attempt=2,
            _correction_route_capability=entry.capability,
        )


def test_stateful_packet_accessor_cannot_swap_correction_bytes_at_router(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=_snapshot(daemon),
        write_paths=task.outputs,
        read_paths=task.outputs,
    )
    clean_payload = dict(packet.payload)
    correction_payload = {
        **clean_payload,
        "correction_feedback": {"feedback_binding_id": "forged"},
    }

    class StatefulPacket:
        packet_id = packet.packet_id
        snapshot_id = packet.snapshot_id
        task_id = packet.task_id
        implementable = True

        def __init__(self):
            self.payload_reads = 0

        def assert_current(self, current_snapshot_id):
            assert current_snapshot_id == self.snapshot_id

        @property
        def provider_input_payload(self):
            self.payload_reads += 1
            # The pre-fix public/core/router read sequence reached the forged
            # fourth value. The fixed boundary routes the first frozen view.
            if self.payload_reads >= 4:
                return correction_payload
            return clean_payload

    stateful = StatefulPacket()
    provider_inputs: list[dict[str, Any]] = []

    def grok(request):
        provider_inputs.append(dict(request["provider_input"]))
        assert "correction_feedback" not in request["provider_input"]
        return _grok(request)

    result, _event, _receipt = daemon.route_model_assisted_contract_packet(
        stateful,
        current_snapshot_id=packet.snapshot_id,
        task=task,
        attempt=1,
        grok_provider=grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert stateful.payload_reads == 1
    assert provider_inputs
    assert all("correction_feedback" not in value for value in provider_inputs)


def test_stateful_packet_with_first_read_correction_invokes_no_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=_snapshot(daemon),
        write_paths=task.outputs,
        read_paths=task.outputs,
    )
    clean_payload = dict(packet.payload)
    correction_payload = {
        **clean_payload,
        "correction_feedback": {"feedback_binding_id": "forged"},
    }

    class StatefulPacket:
        packet_id = packet.packet_id
        snapshot_id = packet.snapshot_id
        task_id = packet.task_id
        implementable = True

        def __init__(self):
            self.payload_reads = 0

        def assert_current(self, current_snapshot_id):
            assert current_snapshot_id == self.snapshot_id

        @property
        def provider_input_payload(self):
            self.payload_reads += 1
            return correction_payload if self.payload_reads == 1 else clean_payload

    stateful = StatefulPacket()
    provider_calls: list[str] = []

    with pytest.raises(RuntimeError, match="public.*rejects correction"):
        daemon.route_model_assisted_contract_packet(
            stateful,
            current_snapshot_id=packet.snapshot_id,
            task=task,
            attempt=1,
            grok_provider=lambda _request: provider_calls.append("grok"),
            codex_provider=lambda _request: provider_calls.append("codex"),
            admission_gate=_accept,
        )

    assert stateful.payload_reads == 1
    assert provider_calls == []


def test_materialized_packet_preserves_nested_generic_mappings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=_snapshot(daemon),
        write_paths=task.outputs,
        read_paths=task.outputs,
        extra_goal={
            "nested_mapping": MappingProxyType(
                {"value": "preserved"}
            )
        },
    )
    materialized = (
        implementation_daemon_module
        ._materialize_model_assisted_contract_packet(
            packet,
            current_snapshot_id=packet.snapshot_id,
        )
    )
    assert materialized.payload["goal"]["nested_mapping"]["value"] == (
        "preserved"
    )

    result, _event, _receipt = daemon.route_model_assisted_contract_packet(
        packet,
        current_snapshot_id=packet.snapshot_id,
        task=task,
        attempt=1,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert result.status is RouteStatus.SUCCEEDED


@pytest.mark.parametrize(
    "invalid_kind",
    ["non_string_key", "cycle", "depth", "items", "memoryview"],
)
def test_materializer_translates_invalid_json_edges_without_provider_call(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    if invalid_kind == "non_string_key":
        invalid_value: Any = {"nested": {1: "not-a-string-key"}}
    elif invalid_kind == "cycle":
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle
        invalid_value = {"nested": cycle}
    elif invalid_kind == "depth":
        invalid_value = "leaf"
        for _index in range(MAX_PROVIDER_JSON_DEPTH + 1):
            invalid_value = {"nested": invalid_value}
    elif invalid_kind == "items":
        invalid_value = {
            "values": list(range(MAX_PROVIDER_JSON_ITEMS))
        }
    else:
        invalid_value = {"opaque": memoryview(b"not-json")}
    packet = ProductionContractPacket(
        packet_id=f"packet:invalid:{invalid_kind}",
        snapshot_id=_snapshot(daemon),
        task_id=task.task_id,
        payload=invalid_value,
    )
    provider_calls: list[str] = []

    with pytest.raises(ProviderRoutingError) as captured:
        daemon.route_model_assisted_contract_packet(
            packet,
            current_snapshot_id=packet.snapshot_id,
            task=task,
            attempt=1,
            grok_provider=lambda _request: provider_calls.append("grok"),
            codex_provider=lambda _request: provider_calls.append("codex"),
            admission_gate=_accept,
        )

    assert captured.value.reason_code == ProviderReason.PACKET_MALFORMED.value
    assert provider_calls == []


def test_constructed_copied_and_unregistered_correction_tokens_have_no_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    _key, entry, _feedback, _reservation = (
        _install_claimable_correction_route(
            daemon,
            task,
            attempt=2,
        )
    )
    token_type = type(entry.capability)

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(entry.capability)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(entry.capability)

    forged_candidates = (
        token_type(),
        object.__new__(token_type),
        dict(entry.material),
    )
    for forged in forged_candidates:
        with pytest.raises(
            RuntimeError,
            match="capability is no longer current",
        ):
            daemon._run_production_model_assisted_route_core(
                task,
                attempt=2,
                workspace_path=daemon.repo_root,
                snapshot_id=_snapshot(daemon),
                apply=False,
                _correction_route_capability=forged,
            )


def test_internal_correction_route_is_one_shot_and_rereads_durable_bindings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    key, entry, feedback, reservation = _install_claimable_correction_route(
        daemon,
        task,
        attempt=2,
    )
    denial_id = str(entry.material["durable_denial_id"])
    feedback_reads: list[str] = []
    reservation_reads: list[int] = []

    def read_feedback(bound_task, bound_denial_id):
        assert bound_task is task
        assert bound_denial_id == denial_id
        feedback_reads.append(bound_denial_id)
        return dict(feedback)

    def read_reservation(bound_task, *, attempt, authority):
        assert bound_task is task
        assert attempt == 2
        assert authority["durable_denial_id"] == denial_id
        reservation_reads.append(attempt)
        return dict(reservation)

    monkeypatch.setattr(
        daemon,
        "_verified_complete_post_merge_denial_feedback",
        read_feedback,
    )
    monkeypatch.setattr(
        daemon,
        "_verified_post_merge_correction_route_reservation",
        read_reservation,
    )

    result = daemon._run_production_post_merge_correction_route(
        task,
        attempt=2,
        workspace_path=daemon.repo_root,
        snapshot_id=_snapshot(daemon),
        apply=False,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert result["returncode"] == 0
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert feedback_reads == [denial_id, denial_id]
    assert reservation_reads == [2, 2]
    assert entry.capability._burned is True
    with daemon._post_merge_correction_route_registry_lock:
        assert key not in daemon._sealed_post_merge_correction_routes
        assert key not in daemon._claimed_post_merge_correction_routes

    with pytest.raises(RuntimeError, match="unambiguous sealed capability"):
        daemon._run_production_post_merge_correction_route(
            task,
            attempt=2,
        )
    with pytest.raises(RuntimeError, match="capability is no longer current"):
        daemon._run_production_model_assisted_route_core(
            task,
            attempt=2,
            workspace_path=daemon.repo_root,
            snapshot_id=_snapshot(daemon),
            apply=False,
            _correction_route_capability=entry.capability,
        )
    assert feedback_reads == [denial_id, denial_id]
    assert reservation_reads == [2, 2]


def test_ambiguous_correction_claim_is_non_destructive(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    first_key, first_entry, _feedback, _reservation = (
        _install_claimable_correction_route(
            daemon,
            task,
            attempt=2,
            suffix="first",
        )
    )
    second_key, second_entry, _feedback, _reservation = (
        _install_claimable_correction_route(
            daemon,
            task,
            attempt=2,
            suffix="second",
        )
    )
    before = dict(daemon._sealed_post_merge_correction_routes)
    monkeypatch.setattr(
        daemon,
        "_run_production_model_assisted_route_core",
        lambda *_args, **_kwargs: pytest.fail(
            "ambiguous correction authority reached the route core"
        ),
    )

    with pytest.raises(RuntimeError, match="unambiguous sealed capability"):
        daemon._run_production_post_merge_correction_route(
            task,
            attempt=2,
        )

    with daemon._post_merge_correction_route_registry_lock:
        assert daemon._sealed_post_merge_correction_routes == before
        assert daemon._claimed_post_merge_correction_routes == {}
        assert daemon._sealed_post_merge_correction_routes[first_key] is first_entry
        assert daemon._sealed_post_merge_correction_routes[second_key] is second_entry
    assert first_entry.capability._burned is False
    assert second_entry.capability._burned is False


def test_concurrent_lower_boundary_has_one_exact_claim_winner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    key, entry, feedback, reservation = _install_claimable_correction_route(
        daemon,
        task,
        attempt=2,
    )
    with daemon._post_merge_correction_route_registry_lock:
        assert daemon._sealed_post_merge_correction_routes.pop(key) is entry
        daemon._claimed_post_merge_correction_routes[key] = entry
    feedback_reads: list[str] = []
    reservation_reads: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_verified_complete_post_merge_denial_feedback",
        lambda _task, denial_id: (
            feedback_reads.append(denial_id) or dict(feedback)
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_verified_post_merge_correction_route_reservation",
        lambda _task, *, attempt, authority: (
            reservation_reads.append(attempt) or dict(reservation)
        ),
    )
    barrier = threading.Barrier(3)
    outcomes: list[Any] = []

    def consume() -> None:
        barrier.wait()
        outcomes.append(
            daemon._fresh_post_merge_correction_capability_bindings(
                task,
                attempt=2,
                capability=entry.capability,
                consume=True,
            )
        )

    threads = [threading.Thread(target=consume) for _index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert len([outcome for outcome in outcomes if outcome is not None]) == 1
    assert len([outcome for outcome in outcomes if outcome is None]) == 1
    assert len(feedback_reads) == 1
    assert reservation_reads == [2]
    assert entry.capability._burned is True
    with daemon._post_merge_correction_route_registry_lock:
        assert key not in daemon._claimed_post_merge_correction_routes


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
        snapshot_id=_snapshot(daemon),
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


def test_verified_grok_quota_persists_terra_candidate_without_write_or_self_review(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    baseline = (daemon.repo_root / PATH).read_bytes()
    calls: list[str] = []

    def grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    def terra(request):
        calls.append("terra")
        assert request.role is ProviderRole.CODEX_QUOTA_IMPLEMENT
        return {
            "proposal": {
                "declared_paths": [PATH],
                "files": [
                    {
                        "path": PATH,
                        "content": "# terra-pending-non-codex-review\n",
                    }
                ],
            }
        }

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=3,
        workspace_path=daemon.repo_root,
        snapshot_id=_snapshot(daemon),
        apply=True,
        grok_provider=grok,
        codex_implementation_fallback_provider=terra,
        codex_provider=lambda _request: calls.append("codex-review"),
        admission_gate=_accept,
    )

    route = result["route_result"]
    assert calls == ["grok", "terra"]
    assert route.status is RouteStatus.DEFERRED
    assert route.reason_code == ProviderReason.NON_CODEX_REVIEW_REQUIRED.value
    assert route.review_presence == ReviewPresence.ABSENT.value
    assert route.provider_result_admitted is False
    assert route.write_performed is False
    assert result["returncode"] == 1
    assert result["pending"] is True
    assert result["binding"] is None
    assert result["reviewed_effect_binding"] is None
    assert (daemon.repo_root / PATH).read_bytes() == baseline

    event = result["event"]
    assert event["provider"] == ProviderRole.CODEX_QUOTA_IMPLEMENT.value
    assert event["required_review_role"] == ProviderRole.NON_CODEX_REVIEW.value
    proposal_path = Path(event["pending_proposal_path"])
    assert proposal_path.is_file()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["artifact_id"] == event["pending_proposal_artifact_id"]
    assert proposal["provider_result_admitted"] is False
    assert proposal["write_performed"] is False
    assert proposal["proposal"]["role"] == ProviderRole.CODEX_QUOTA_IMPLEMENT.value
    assert proposal["required_review_role"] == ProviderRole.NON_CODEX_REVIEW.value


def test_terra_candidate_latches_waiting_without_consuming_or_reinvoking(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    daemon._register_task_identities([task])
    calls: list[str] = []

    def grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    def terra(_request):
        calls.append("terra")
        return {
            "proposal": {
                "declared_paths": [PATH],
                "files": [{"path": PATH, "content": "# pending\n"}],
            }
        }

    daemon._production_grok_provider = grok
    daemon._production_codex_implementation_fallback_provider = terra
    daemon._production_codex_provider = lambda _request: calls.append(
        "codex-review"
    )
    daemon._production_admission_gate = _accept
    state = PortalTaskState()

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-08-03T00:00:00+00:00",
        log_path=daemon.implementation_log_dir / "terra-latch.log",
        prompt="implement the bounded packet",
    )

    assert calls == ["grok", "terra"]
    assert result["reason"] == "provider_review_pending"
    assert result["attempt_consumed"] is False
    assert result["provider_call_allowed"] is False
    persisted = PortalTaskState.load(daemon.state_path)
    canonical_cid = daemon._canonical_ref(task)
    assert persisted.implementation_attempts_by_cid.get(canonical_cid, 0) == 0
    assert persisted.pending_provider_reviews[canonical_cid]["artifact_id"] == (
        result["pending_proposal_artifact_id"]
    )

    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(
        daemon,
        "_run_implementation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("latched candidate must not invoke a provider")
        ),
    )
    follow_up = daemon.run_once()

    assert calls == ["grok", "terra"]
    assert follow_up["implementation_result"] is None
    assert follow_up["pending_provider_review_task_ids"] == [task.task_id]
    assert PortalTaskState.load(daemon.state_path).task_statuses[task.task_id] == (
        "waiting"
    )
    assert any(
        event.get("type") == MODEL_ASSISTED_PROVIDER_REVIEW_PENDING_EVENT
        for event in _events(daemon)
    )


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


def _admitted_patch_proposal(
    patch: str,
    *declared_paths: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        admitted=True,
        payload={
            "proposal": {
                "declared_paths": list(declared_paths),
                "patch": patch,
            }
        },
    )


def _install_direct_submodule(
    tmp_path: Path,
    daemon: TodoImplementationDaemon,
) -> tuple[str, str, Path]:
    root = "external/ipfs_datasets"
    inner_path = f"{root}/logic/ui.py"
    source = tmp_path / "ipfs-datasets-source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "Production Route Test")
    _git(source, "config", "user.email", "production-route@example.invalid")
    source_target = source / "logic" / "ui.py"
    source_target.parent.mkdir(parents=True)
    source_target.write_text("# child baseline\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "child baseline")

    _git(
        daemon.repo_root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(source),
        root,
    )
    _git(daemon.repo_root, "commit", "-am", "add datasets submodule")
    daemon.worktree_submodule_paths = (root,)
    return root, inner_path, daemon.repo_root / root


def _mixed_outer_child_patch(inner_path: str) -> str:
    return (
        f"diff --git a/{PATH} b/{PATH}\n"
        f"--- a/{PATH}\n"
        f"+++ b/{PATH}\n"
        "@@ -1 +1 @@\n"
        "-# baseline\n"
        "+# outer patched\n"
        f"diff --git a/{inner_path} b/{inner_path}\n"
        f"--- a/{inner_path}\n"
        f"+++ b/{inner_path}\n"
        "@@ -1 +1 @@\n"
        "-# child baseline\n"
        "+# child patched\n"
    )


def test_production_writer_supports_transactional_outer_and_submodule_files(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    _root, inner_path, child = _install_direct_submodule(tmp_path, daemon)
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[PATH, inner_path]),
        expected_lease_id="lease:mixed-files",
    )

    writer(
        _admitted_file_proposal(
            (PATH, "# outer replacement\n"),
            (inner_path, "# child replacement\n"),
        ),
        "lease:mixed-files",
    )

    assert (daemon.repo_root / PATH).read_text(encoding="utf-8") == (
        "# outer replacement\n"
    )
    assert (child / "logic" / "ui.py").read_text(encoding="utf-8") == (
        "# child replacement\n"
    )


def test_production_writer_applies_mixed_patch_through_synthetic_flat_tree(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    _root, inner_path, child = _install_direct_submodule(tmp_path, daemon)
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[PATH, inner_path]),
        expected_lease_id="lease:mixed-patch",
    )
    real_run = subprocess.run
    apply_worktrees: list[Path] = []

    def observe_git_apply(*args, **kwargs):
        command = args[0]
        if command[:2] == ["git", "apply"]:
            apply_worktrees.append(Path(kwargs["cwd"]))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", observe_git_apply)

    writer(
        _admitted_patch_proposal(
            _mixed_outer_child_patch(inner_path),
            PATH,
            inner_path,
        ),
        "lease:mixed-patch",
    )

    assert (daemon.repo_root / PATH).read_text(encoding="utf-8") == (
        "# outer patched\n"
    )
    assert (child / "logic" / "ui.py").read_text(encoding="utf-8") == (
        "# child patched\n"
    )
    assert len(apply_worktrees) == 3
    assert daemon.repo_root not in apply_worktrees
    assert child not in apply_worktrees


def test_production_writer_revalidates_exact_gitlink_and_child_head(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    root, inner_path, child = _install_direct_submodule(tmp_path, daemon)
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[inner_path]),
        expected_lease_id="lease:head-fence",
    )

    (child / "logic" / "ui.py").write_text("# next child\n", encoding="utf-8")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "unrecorded child head")

    with pytest.raises(RuntimeError, match="HEAD does not match outer gitlink"):
        writer(
            _admitted_file_proposal((inner_path, "# forbidden\n")),
            "lease:head-fence",
        )
    assert (child / "logic" / "ui.py").read_text(encoding="utf-8") == (
        "# next child\n"
    )

    root_writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[root]),
        expected_lease_id="lease:root-fence",
    )
    with pytest.raises(RuntimeError, match="root itself is not a writable file"):
        root_writer(
            _admitted_file_proposal((root, "# forbidden\n")),
            "lease:root-fence",
        )


def test_production_writer_rejects_repository_below_registered_submodule(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    root, _inner_path, child = _install_direct_submodule(tmp_path, daemon)
    nested_path = f"{root}/logic/nested/target.py"
    nested = child / "logic" / "nested"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    (nested / "target.py").write_text("nested baseline\n", encoding="utf-8")
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[nested_path]),
        expected_lease_id="lease:deeper-repo",
    )

    with pytest.raises(RuntimeError, match="nested repository path"):
        writer(
            _admitted_file_proposal((nested_path, "nested replacement\n")),
            "lease:deeper-repo",
        )
    assert (nested / "target.py").read_text(encoding="utf-8") == (
        "nested baseline\n"
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "target.py"
    outside_target.write_text("outside baseline\n", encoding="utf-8")
    link = child / "logic" / "link"
    link.symlink_to(outside, target_is_directory=True)
    symlink_path = f"{root}/logic/link/target.py"
    symlink_writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[symlink_path]),
        expected_lease_id="lease:child-symlink",
    )
    with pytest.raises(RuntimeError, match="symlink path component"):
        symlink_writer(
            _admitted_file_proposal((symlink_path, "escaped\n")),
            "lease:child-symlink",
        )
    assert outside_target.read_text(encoding="utf-8") == "outside baseline\n"


def test_production_writer_requires_mode_160000_at_registered_boundary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    root = "ordinary-child"
    inner_path = f"{root}/target.py"
    child = daemon.repo_root / root
    child.mkdir()
    (child / "target.py").write_text("ordinary baseline\n", encoding="utf-8")
    _git(daemon.repo_root, "add", root)
    _git(daemon.repo_root, "commit", "-m", "ordinary directory")
    _git(child, "init")
    _git(child, "config", "user.name", "Production Route Test")
    _git(child, "config", "user.email", "production-route@example.invalid")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "standalone child")
    daemon.worktree_submodule_paths = (root,)
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[inner_path]),
        expected_lease_id="lease:not-gitlink",
    )

    with pytest.raises(RuntimeError, match="not an exact HEAD gitlink"):
        writer(
            _admitted_file_proposal((inner_path, "forbidden\n")),
            "lease:not-gitlink",
        )
    assert (child / "target.py").read_text(encoding="utf-8") == (
        "ordinary baseline\n"
    )


def test_production_writer_rolls_back_mixed_patch_materialization_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    _root, inner_path, child = _install_direct_submodule(tmp_path, daemon)
    outer_target = daemon.repo_root / PATH
    child_target = child / "logic" / "ui.py"
    outer_before = outer_target.read_bytes()
    child_before = child_target.read_bytes()
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[PATH, inner_path]),
        expected_lease_id="lease:mixed-patch-rollback",
    )

    real_replace = os.replace
    failed = False

    def fail_child_once(source, destination):
        nonlocal failed
        if Path(destination) == child_target and not failed:
            failed = True
            raise OSError("injected child patch replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_child_once)
    with pytest.raises(
        RuntimeError,
        match="transactional patch materialization failed",
    ):
        writer(
            _admitted_patch_proposal(
                _mixed_outer_child_patch(inner_path),
                PATH,
                inner_path,
            ),
            "lease:mixed-patch-rollback",
        )

    assert failed
    assert outer_target.read_bytes() == outer_before
    assert child_target.read_bytes() == child_before
    assert not list(daemon.repo_root.rglob(".production-provider-write-*"))


def test_production_writer_rejects_target_tamper_after_synthetic_patch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    _root, inner_path, child = _install_direct_submodule(tmp_path, daemon)
    outer_target = daemon.repo_root / PATH
    child_target = child / "logic" / "ui.py"
    outer_before = outer_target.read_bytes()
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[PATH, inner_path]),
        expected_lease_id="lease:patch-tamper",
    )
    real_run = subprocess.run
    tampered = False

    def tamper_after_synthetic_apply(*args, **kwargs):
        nonlocal tampered
        result = real_run(*args, **kwargs)
        command = args[0]
        if (
            command[:2] == ["git", "apply"]
            and "--check" not in command
            and "--numstat" not in command
            and not tampered
        ):
            child_target.write_text("# concurrent tamper\n", encoding="utf-8")
            tampered = True
        return result

    monkeypatch.setattr(subprocess, "run", tamper_after_synthetic_apply)
    with pytest.raises(RuntimeError, match="write baseline changed"):
        writer(
            _admitted_patch_proposal(
                _mixed_outer_child_patch(inner_path),
                PATH,
                inner_path,
            ),
            "lease:patch-tamper",
        )

    assert tampered
    assert outer_target.read_bytes() == outer_before
    assert child_target.read_text(encoding="utf-8") == "# concurrent tamper\n"


def test_production_writer_rollback_prunes_new_submodule_directories(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    root, _inner_path, child = _install_direct_submodule(tmp_path, daemon)
    new_path = f"{root}/generated/nested/new.py"
    new_target = daemon.repo_root / new_path
    outer_target = daemon.repo_root / PATH
    outer_before = outer_target.read_bytes()
    writer = daemon._make_production_workspace_writer(
        daemon.repo_root,
        task=_task(outputs=[new_path, PATH]),
        expected_lease_id="lease:new-directory-rollback",
    )
    real_replace = os.replace
    failed = False

    def fail_outer_once(source, destination):
        nonlocal failed
        if Path(destination) == outer_target and not failed:
            failed = True
            raise OSError("injected outer replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_outer_once)
    with pytest.raises(RuntimeError, match="transactional file replacement failed"):
        writer(
            _admitted_file_proposal(
                (new_path, "# generated\n"),
                (PATH, "# outer replacement\n"),
            ),
            "lease:new-directory-rollback",
        )

    assert failed
    assert outer_target.read_bytes() == outer_before
    assert not new_target.exists()
    assert not (child / "generated").exists()
    assert not list(daemon.repo_root.rglob(".production-provider-write-*"))


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
        return {"decision": "approve", "findings": []}

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=_snapshot(daemon),
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
        assert "context_slice" in slice_
        assert slice_["context_slice"]["manifest_cid"].startswith("b")
        assert request.prompt_tokens <= MAX_PROVIDER_PROMPT_TOKENS
        # Full goal corpus / counterexample bodies must not appear.
        encoded = json.dumps(request["provider_input"], sort_keys=True)
        assert "counterexample" not in encoded
        assert "repository_corpus" not in encoded
        return {"decision": "approve", "findings": []}

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=_snapshot(daemon),
        apply=True,
        grok_provider=_grok,
        codex_provider=codex,
        admission_gate=_accept,
    )
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert "admitted_implementation_proposal" in seen["input"]
    assert all(
        attempt.prompt_tokens <= MAX_PROVIDER_PROMPT_TOKENS
        for attempt in result["route_result"].attempts
    )


def test_caller_packet_without_context_fails_before_any_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    snapshot = _snapshot(daemon)
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=snapshot,
        write_paths=task.outputs,
        read_paths=task.outputs,
    )
    calls: list[str] = []

    with pytest.raises(ProviderRoutingError) as captured:
        daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=snapshot,
            packet=packet,
            apply=False,
            grok_provider=lambda _request: calls.append("grok"),
            codex_provider=lambda _request: calls.append("codex"),
            admission_gate=_accept,
        )

    assert captured.value.reason_code == "context_manifest_missing"
    assert calls == []


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
        snapshot_id=_snapshot(daemon),
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
            snapshot_id=_snapshot(daemon),
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
            snapshot_id=_snapshot(daemon),
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
            snapshot_id=_snapshot(daemon),
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
            snapshot_id=_snapshot(daemon),
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )
        disposition, reason = evaluate_production_provider_receipt(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=result["snapshot_id"],
        )
        assert disposition is ProductionReceiptDisposition.PENDING_CROSS_TASK
        assert reason == ProviderReason.RECEIPT_CROSS_TASK.value
        assert daemon.production_provider_receipt_allows_merge(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=result["snapshot_id"],
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
            snapshot_id=_snapshot(daemon),
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
        return {"decision": "approve", "findings": []}

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
    snapshot = _snapshot(daemon)
    packet = daemon.build_production_contract_packet_for_task(
        _task(),
        snapshot_id=snapshot,
        attempt=2,
    )
    assert isinstance(packet, ProductionContractPacket)
    assert packet.task_id == "SCA-615"
    assert packet.snapshot_id == snapshot
    payload = dict(packet.provider_input_payload)
    assert payload["authority"]["completion_authoritative"] is False
    assert PATH in payload["scope"]["write_paths"]
    assert payload["scope"]["read_paths"] == [PATH]
    assert payload["context_slice"]["repository_binding"]["snapshot_id"] == snapshot
    assert payload["context_slice"]["scope"]["effect_paths"] == [PATH]
    assert payload["context_slice"]["scope"]["read_paths"] == [PATH]
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
