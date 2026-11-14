"""Focused tests for the sealed state-laws trusted acquisition boundary."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.trusted_actions import (
    state_laws_live_cohort_v1 as action_module,
)
from ipfs_accelerate_py.agent_supervisor.trusted_actions import (
    state_laws_offline_validation_v1 as offline_validation,
)
from ipfs_accelerate_py.agent_supervisor.trusted_actions.state_laws_live_cohort_v1 import (
    _live_action_progress_marker,
    EVIDENCE_ENVIRONMENT_VARIABLE,
    PRE_ACTION_VALIDATION_HANDLER,
    PRE_ACTION_VALIDATION_SCHEMA,
    RECEIPT_ENVIRONMENT_VARIABLE,
    RECEIPT_SHA256_ENVIRONMENT_VARIABLE,
    SANDBOX_CAPABILITY,
    SANDBOX_MODE,
    SANDBOX_POLICY_PATH,
    SOURCE_PROJECTION_MODE,
    STATE_LAWS_TASK_COHORTS,
    TRUSTED_PREVALIDATION_ACTION_POLICY_SCHEMA,
    TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA,
    WRAPPER_PATH,
    execute_state_laws_live_cohort_action,
    parse_state_laws_live_cohort_policy,
    pre_action_validation_command,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    STATE_LAWS_TRUSTED_ACTION_RECEIPT_ENV,
    STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV,
    ValidationRuntimeError,
    build_validation_environment,
    validation_shell_command,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
    build_validation_cache_key,
)

ACCELERATOR_ROOT = Path(action_module.__file__).resolve().parents[3]

_COHORT_A_LEDGER_HOSTS = {
    "AL": "alison.legislature.state.al.us",
    "AK": "akleg.gov",
    "AZ": "azleg.gov",
    "AR": "arkleg.state.ar.us",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "HOME": "/nonexistent/state-laws-projection-test",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _git_blob(content: str) -> str:
    raw = content.encode("utf-8", errors="surrogatepass")
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw,
        usedforsecurity=False,
    ).hexdigest()


def _policy_payload() -> dict[str, Any]:
    wrapper = ACCELERATOR_ROOT / WRAPPER_PATH
    sandbox_policy = ACCELERATOR_ROOT / SANDBOX_POLICY_PATH
    return {
        "schema": TRUSTED_PREVALIDATION_ACTION_POLICY_SCHEMA,
        "handler": "state_laws_live_cohort_v1",
        "phase": "accepted_proposal_before_hermetic_validation",
        "required_for_task_completion": True,
        "receipt_schema": TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA,
        "task_cohorts": dict(STATE_LAWS_TASK_COHORTS),
        "runner_path": action_module.RUNNER_PATH,
        "evidence_root_environment_variable": EVIDENCE_ENVIRONMENT_VARIABLE,
        "validation_receipt_environment_variable": RECEIPT_ENVIRONMENT_VARIABLE,
        "validation_receipt_sha256_environment_variable": (
            RECEIPT_SHA256_ENVIRONMENT_VARIABLE
        ),
        "evidence_root": "workspace/state-laws-evidence",
        "receipt_root": "workspace/state-laws-evidence/action-receipts",
        "max_parallel": 2,
        "action_timeout_seconds": 43200,
        "progress_timeout_seconds": 1800,
        "trusted_static_pre_action_gate": (
            action_module.TRUSTED_STATIC_PRE_ACTION_GATE
        ),
        "adapter_evidence_ownership": action_module.ADAPTER_EVIDENCE_OWNERSHIP,
        "adapter_contract_catalog_binding_required": True,
        "adapter_contract_closed_mode_schema_required": True,
        "adapter_contract_reviewed_policy_digest_required": True,
        "source_closure_policy_registry_path": (
            action_module.SOURCE_CLOSURE_POLICY_REGISTRY_PATH
        ),
        "source_closure_policy_registry_schema": (
            action_module.SOURCE_CLOSURE_POLICY_REGISTRY_SCHEMA
        ),
        "adapter_evidence_api_signature_and_dataflow_required": True,
        "source_projection": {
            "mode": SOURCE_PROJECTION_MODE,
            "shared_paths": [
                action_module.OFFICIAL_SOURCE_CATALOG_PATH,
                action_module.SOURCE_CLOSURE_POLICY_REGISTRY_PATH,
                action_module.RUNNER_PATH,
                "shared/policy.py",
            ],
            "cohort_paths": {
                cohort: [f"cohorts/{cohort.lower()}_adapter.py"]
                for cohort in STATE_LAWS_TASK_COHORTS.values()
            },
            "unrelated_cohort_paths_invalidate": False,
            "shared_path_changes_invalidate": True,
            "path_sha256_map_required": True,
            "digest_required": True,
        },
        "pre_action_validation": {
            "schema": PRE_ACTION_VALIDATION_SCHEMA,
            "handler": PRE_ACTION_VALIDATION_HANDLER,
            "network": "disabled",
            "proposal_bound": True,
        },
        "receipt_bindings": list(action_module._REQUIRED_RECEIPT_BINDINGS),
        "retry": {
            "max_attempts": 3,
            "resume_exact_fingerprint_only": True,
        },
        "sandbox": {
            "mode": SANDBOX_MODE,
            "required_capability": SANDBOX_CAPABILITY,
            "wrapper_path": WRAPPER_PATH,
            "wrapper_sha256": _sha256(wrapper),
            "policy_path": SANDBOX_POLICY_PATH,
            "policy_sha256": _sha256(sandbox_policy),
            "unavailable_is_failure": True,
        },
    }


def _workspace(repo: Path) -> Path:
    workspace = repo / "candidate"
    for relative in (
        action_module.OFFICIAL_SOURCE_CATALOG_PATH,
        action_module.SOURCE_CLOSURE_POLICY_REGISTRY_PATH,
        action_module.RUNNER_PATH,
        "shared/policy.py",
        *(f"cohorts/{cohort.lower()}_adapter.py" for cohort in "ABCDEFGHIJKLM"),
    ):
        _write(workspace / relative, f"# {relative}\n")
    return workspace


def _unit_source_projection_builder(
    *,
    workspace_path: Path,
    cohort: str,
    policy: Any,
    **_kwargs: Any,
) -> tuple[dict[str, str], str, dict[str, bytes]]:
    projection = {
        relative: _sha256(workspace_path / relative)
        for relative in (
            *policy.source_projection.shared_paths,
            *policy.source_projection.cohort_paths[cohort],
        )
    }
    digest = action_module._content_id(
        {
            "schema": "state-laws-reindex-live-source-projection-v1",
            "cohort_id": cohort,
            "files": dict(sorted(projection.items())),
        }
    )
    return dict(sorted(projection.items())), digest, {}


@dataclass(frozen=True)
class _DiffEntry:
    old_path: str
    new_path: str
    content: str

    def to_dict(self, *, include_sources: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "content": self.content,
        }
        if include_sources:
            result["old_source"] = ""
            result["new_source"] = self.content
        return result


def _proposal_validation(
    *,
    serial: int,
    entries: tuple[_DiffEntry, ...],
    baseline_id: str = "1" * 40,
) -> SimpleNamespace:
    proposal = SimpleNamespace(
        proposal_id=f"proposal-{serial}",
        diff_digest=f"diff-{serial}",
        repository_tree_id=f"tree-{serial}",
        baseline_id=baseline_id,
        candidate_diff=entries,
        changed_paths=tuple(entry.new_path or entry.old_path for entry in entries),
    )
    return SimpleNamespace(
        accepted=True,
        proposal=proposal,
        receipt=SimpleNamespace(receipt_id=f"proposal-receipt-{serial}"),
    )


def _task(cohort: str = "A") -> SimpleNamespace:
    task_id = f"SLR-{9 + ord(cohort) - ord('A'):03d}"
    return SimpleNamespace(
        task_id=task_id,
        metadata={
            "Trusted prevalidation action": "state_laws_live_cohort_v1",
            "Trusted action cohort": cohort,
            "Trusted action required": "true",
            "Trusted action receipt schema": (
                TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA
            ),
            "Trusted action source projection": SOURCE_PROJECTION_MODE,
            "Trusted static pre action gate": (
                action_module.TRUSTED_STATIC_PRE_ACTION_GATE
            ),
            "Adapter evidence ownership": action_module.ADAPTER_EVIDENCE_OWNERSHIP,
            "Pre-action validation": pre_action_validation_command(cohort),
            "Validation materialized outputs": (
                f"docs/reports/state_laws_reindex/cohort_{cohort.lower()}.json"
            ),
        },
    )


def _accepted_projection_case(
    tmp_path: Path,
    *,
    unsafe_shared_mode: bool = False,
) -> tuple[
    Path,
    Any,
    SimpleNamespace,
    CandidateDiffEntry,
    CandidateDiffEntry,
]:
    workspace = tmp_path / "accepted-candidate"
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    runner_source = "# sealed runner baseline\n"
    shared_source = "POLICY = 'baseline'\n"
    adapter_before = "VALUE = 'before'\n"
    adapter_after = "VALUE = 'after'\n"
    test_after = "def test_candidate():\n    assert True\n"
    _write(
        workspace / action_module.OFFICIAL_SOURCE_CATALOG_PATH,
        '{"catalog":"sealed-test"}\n',
    )
    _write(
        workspace / action_module.SOURCE_CLOSURE_POLICY_REGISTRY_PATH,
        '{"registry":"sealed-test"}\n',
    )
    _write(workspace / action_module.RUNNER_PATH, runner_source)
    shared_path = workspace / "shared/policy.py"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    if unsafe_shared_mode:
        shared_path.symlink_to("policy-target.py")
    else:
        _write(shared_path, shared_source)
    _write(workspace / "cohorts/a_adapter.py", adapter_before)
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.name", "State Laws Test")
    _git(workspace, "config", "user.email", "state-laws@example.invalid")
    _git(workspace, "add", "--all")
    _git(workspace, "commit", "--quiet", "-m", "sealed baseline")
    baseline = _git(workspace, "rev-parse", "HEAD")
    if unsafe_shared_mode:
        shared_path.unlink()
        _write(shared_path, shared_source)
    _write(workspace / "cohorts/a_adapter.py", adapter_after)
    test_path = "tests/integration/legal_scrapers/test_state_laws_cohort_a.py"
    _write(workspace / test_path, test_after)
    adapter_entry = CandidateDiffEntry(
        old_path="cohorts/a_adapter.py",
        new_path="cohorts/a_adapter.py",
        change_kind=DiffChangeKind.MODIFY,
        before_source=adapter_before,
        after_source=adapter_after,
        before_blob_id=_git(
            workspace,
            "rev-parse",
            f"{baseline}:cohorts/a_adapter.py",
        ),
        after_blob_id=_git_blob(adapter_after),
    )
    test_entry = CandidateDiffEntry(
        old_path="",
        new_path=test_path,
        change_kind=DiffChangeKind.ADD,
        before_source=None,
        after_source=test_after,
        before_blob_id="",
        after_blob_id=_git_blob(test_after),
    )
    proposal = SimpleNamespace(
        proposal_id="accepted-projection-1",
        diff_digest="diff-accepted-projection-1",
        repository_tree_id="tree-accepted-projection-1",
        baseline_id=baseline,
        candidate_diff=(adapter_entry, test_entry),
        changed_paths=(adapter_entry.new_path, test_entry.new_path),
    )
    validation = SimpleNamespace(
        accepted=True,
        proposal=proposal,
        receipt=SimpleNamespace(receipt_id="accepted-projection-receipt-1"),
    )
    return workspace, policy, validation, adapter_entry, test_entry


def _proposal_with_entries(
    validation: SimpleNamespace,
    entries: tuple[CandidateDiffEntry, ...],
) -> SimpleNamespace:
    proposal = validation.proposal
    return SimpleNamespace(
        accepted=True,
        proposal=SimpleNamespace(
            proposal_id=proposal.proposal_id,
            diff_digest=proposal.diff_digest,
            repository_tree_id=proposal.repository_tree_id,
            baseline_id=proposal.baseline_id,
            candidate_diff=entries,
            changed_paths=tuple(entry.new_path or entry.old_path for entry in entries),
        ),
        receipt=validation.receipt,
    )


def _bwrap_probe(payload: dict[str, Any]) -> dict[str, object]:
    return offline_validation.probe_bwrap_network_none(
        workspace_path=".",
        repository_tree_id="tree-1",
        environment={},
        runtime_builder=lambda **_kwargs: SimpleNamespace(runtime_id="probe-runtime"),
        process_runner=lambda _runtime: payload,
    )


def _complete_bwrap_receipt(**overrides: Any) -> dict[str, Any]:
    receipt = {
        "returncode": 75,
        "output": offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE,
        "output_truncated": False,
        "timed_out": False,
        "cancelled": False,
        "infrastructure_failure": True,
        "error": "hermetic_isolation_unavailable",
        "runtime_id": "probe-runtime",
    }
    receipt.update(overrides)
    return receipt


@pytest.mark.parametrize(
    "output",
    [
        offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE,
        offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE + "\n",
    ],
)
def test_bwrap_probe_accepts_only_observed_loopback_envelope(
    output: str,
) -> None:
    result = _bwrap_probe(_complete_bwrap_receipt(output=output))

    assert result["process_receipt_complete"] is True
    assert result["fallback_eligible"] is True
    assert result["supported"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", " " + offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE),
        ("output", offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE + "\n\n"),
        ("output", offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE + "\r\n"),
        ("output", offline_validation._EXACT_BWRAP_LOOPBACK_FAILURE + " trailing"),
        ("returncode", 74),
        ("returncode", True),
        ("error", "resource_admission_failed"),
        ("infrastructure_failure", False),
        ("timed_out", True),
        ("cancelled", True),
        ("output_truncated", True),
        ("runtime_id", "other-runtime"),
    ],
)
def test_bwrap_probe_rejects_near_misses(field: str, value: Any) -> None:
    result = _bwrap_probe(_complete_bwrap_receipt(**{field: value}))

    assert result["fallback_eligible"] is False
    assert result["supported"] is False


@pytest.mark.parametrize(
    "missing_field",
    [
        "returncode",
        "output",
        "output_truncated",
        "timed_out",
        "cancelled",
        "infrastructure_failure",
        "error",
        "runtime_id",
    ],
)
def test_bwrap_probe_rejects_partial_process_receipts(
    missing_field: str,
) -> None:
    receipt = _complete_bwrap_receipt()
    del receipt[missing_field]

    result = _bwrap_probe(receipt)

    assert result["process_receipt_complete"] is False
    assert result["fallback_eligible"] is False
    assert result["supported"] is False


def test_bwrap_probe_accepts_only_complete_clean_success() -> None:
    result = _bwrap_probe(
        _complete_bwrap_receipt(
            returncode=0,
            output="",
            infrastructure_failure=False,
            error="",
        )
    )

    assert result["process_receipt_complete"] is True
    assert result["supported"] is True
    assert result["fallback_eligible"] is False


def _offline_scheduler_result(
    *,
    mode: str,
    gate_passed: bool = True,
    returncode: int = 0,
) -> dict[str, Any]:
    phase_ids = ["sha256:" + character * 64 for character in "abc"]
    receipt = {
        "schema": offline_validation.OFFLINE_VALIDATION_SCHEMA,
        "receipt_id": "d" * 64,
        "topology": offline_validation.OFFLINE_VALIDATION_TOPOLOGY,
        "mode": mode,
        "official_catalog_sha256": "e" * 64,
        "source_closure_policy_registry_path": (
            "data/legal/state_laws/source_closure_policy_registry.json"
        ),
        "source_closure_policy_registry_schema": (
            "state-laws-source-closure-policy-registry-v1"
        ),
        "source_closure_policy_registry_sha256": "5" * 64,
        "cleanup_watchdog_assets": {
            "sandbox_runtime_sha256": "1" * 64,
            "proxy_script_sha256": "2" * 64,
        },
        "environment_digest": "3" * 64,
        "source_contract_gate": {
            "passed": gate_passed,
            "official_catalog_sha256": "e" * 64,
        },
        "cleanup": {
            "complete": True,
            "remaining_owned_container_count": 0,
        },
        "attempts": (
            [
                {
                    "attempt_number": number,
                    "returncode": 0,
                    "child_returncode": 0,
                    "container": {"container_id": phase_ids[number]},
                    "network_boundary": {
                        "denied_probes": [
                            "public",
                            "metadata",
                            "public_ipv6",
                        ]
                    },
                }
                for number in range(1, 3)
            ]
            if mode == "gate_and_pytest"
            else []
        ),
        "phase_container_ids": (
            phase_ids if mode == "gate_and_pytest" else phase_ids[:1]
        ),
        "returncode": returncode,
        "infrastructure_failure": False,
        "timed_out": False,
        "cancelled": False,
        "output_truncated": False,
        "classification": "passed" if returncode == 0 else "deterministic_failure",
        "output_sha256": "4" * 64,
        "output_bytes": 0,
    }
    return {
        "command": pre_action_validation_command("A"),
        "returncode": returncode,
        "output": "",
        "infrastructure_failure": False,
        "runtime_id": "d" * 64,
        "state_laws_offline_validation": receipt,
    }


def _preaction_daemon(tmp_path: Path) -> PortalImplementationDaemon:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.implementation_timeout = 60
    daemon.trusted_prevalidation_action_policy = parse_state_laws_live_cohort_policy(
        _policy_payload()
    )
    daemon._canonical_ref = lambda _task: "cid-slr-009"
    return daemon


class _UnitOfflineScheduler:
    def __init__(self, **kwargs: Any) -> None:
        self.runner = kwargs["runner"]

    def run_validated(
        self,
        _proposal: Any,
        commands: tuple[Any, ...],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        item = dict(self.runner(spec=commands[0], timeout_seconds=60))
        item.update(
            {
                "stage": "targeted",
                "resource_cost": 1,
                "validation_result_digest": "f" * 64,
                "resource_admission": {
                    "admitted": True,
                    "resource_cost": 1,
                },
                "resource_lease": {
                    "lease_id": "lease-1",
                    "released": True,
                },
            }
        )
        returncode = int(item.get("returncode") or 0)
        return {
            "attempted": True,
            "passed": returncode == 0,
            "returncode": returncode,
            "results": [item],
        }


def test_supported_bwrap_uses_only_sealed_docker_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _preaction_daemon(tmp_path)
    generic_calls: list[bool] = []
    offline_calls: list[dict[str, Any]] = []

    class Scheduler:
        def run_validated(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            generic_calls.append(True)
            pytest.fail("generic Bubblewrap must never execute candidate pytest")

    daemon.trusted_prevalidation_validation_scheduler = Scheduler()
    monkeypatch.setattr(
        daemon_module,
        "probe_bwrap_network_none",
        lambda **_kwargs: {
            "supported": True,
            "fallback_eligible": False,
            "returncode": 0,
        },
    )

    def offline(**kwargs: Any) -> dict[str, Any]:
        offline_calls.append(kwargs)
        assert "gate_only" not in kwargs
        return _offline_scheduler_result(mode="gate_and_pytest")

    monkeypatch.setattr(
        daemon_module,
        "execute_state_laws_offline_validation",
        offline,
    )
    monkeypatch.setattr(
        daemon_module,
        "ValidationScheduler",
        _UnitOfflineScheduler,
    )
    result = daemon._run_trusted_pre_action_validation(
        tmp_path,
        _task(),
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
        ),
    )

    assert len(offline_calls) == 1
    assert generic_calls == []
    assert result["passed"] is True
    assert result["fallback_used"] is False
    assert result["fallback_eligible"] is False
    assert result["offline_validation_used"] is True
    assert result["topology"] == offline_validation.OFFLINE_VALIDATION_TOPOLOGY
    assert result["static_gate_passed"] is True


def test_exact_bwrap_failure_runs_one_gate_and_two_pytest_containers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _preaction_daemon(tmp_path)
    daemon.trusted_prevalidation_validation_scheduler = object()
    offline_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        daemon_module,
        "probe_bwrap_network_none",
        lambda **_kwargs: {
            "supported": False,
            "fallback_eligible": True,
            "returncode": 75,
        },
    )

    def offline(**kwargs: Any) -> dict[str, Any]:
        offline_calls.append(kwargs)
        assert "gate_only" not in kwargs
        return _offline_scheduler_result(mode="gate_and_pytest")

    monkeypatch.setattr(
        daemon_module,
        "execute_state_laws_offline_validation",
        offline,
    )

    monkeypatch.setattr(
        daemon_module,
        "ValidationScheduler",
        _UnitOfflineScheduler,
    )
    result = daemon._run_trusted_pre_action_validation(
        tmp_path,
        _task(),
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
        ),
    )

    assert len(offline_calls) == 1
    assert result["passed"] is True
    assert result["fallback_used"] is True
    assert result["offline_validation_used"] is True
    assert result["static_gate_passed"] is True
    assert result["topology"] == offline_validation.OFFLINE_VALIDATION_TOPOLOGY
    compact = result["results"][0]
    assert compact["resource_admission"]["resource_cost"] == 1
    assert compact["resource_lease"]["released"] is True
    assert compact["validation_result_digest"] == "f" * 64
    assert len(compact["state_laws_offline_validation"]["phase_container_ids"]) == 3
    compact_offline = compact["state_laws_offline_validation"]
    assert compact_offline["cleanup_watchdog_assets"] == {
        "sandbox_runtime_sha256": "1" * 64,
        "proxy_script_sha256": "2" * 64,
    }
    assert compact_offline["environment_digest"] == "3" * 64
    assert compact_offline["timed_out"] is False
    assert compact_offline["cancelled"] is False
    assert compact_offline["output_truncated"] is False
    assert compact_offline["classification"] == "passed"
    assert compact_offline["output_sha256"] == "4" * 64
    assert compact_offline["source_closure_policy_registry_path"] == (
        "data/legal/state_laws/source_closure_policy_registry.json"
    )
    assert compact_offline["source_closure_policy_registry_schema"] == (
        "state-laws-source-closure-policy-registry-v1"
    )
    assert compact_offline["source_closure_policy_registry_sha256"] == "5" * 64


def test_unrecognized_bwrap_probe_never_dispatches_candidate_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _preaction_daemon(tmp_path)
    generic_calls: list[bool] = []
    offline_calls: list[bool] = []

    class Scheduler:
        def run_validated(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            generic_calls.append(True)
            pytest.fail("unrecognized probe must not execute candidate pytest")

    daemon.trusted_prevalidation_validation_scheduler = Scheduler()
    monkeypatch.setattr(
        daemon_module,
        "probe_bwrap_network_none",
        lambda **_kwargs: {
            "supported": False,
            "fallback_eligible": False,
            "returncode": 75,
            "output": "bwrap: unrecognized isolation failure\n",
            "runtime_id": "probe-runtime",
        },
    )
    monkeypatch.setattr(
        daemon_module,
        "execute_state_laws_offline_validation",
        lambda **_kwargs: offline_calls.append(True),
    )

    result = daemon._run_trusted_pre_action_validation(
        tmp_path,
        _task(),
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
        ),
    )

    assert result["passed"] is False
    assert result["returncode"] == 75
    assert result["offline_validation_used"] is False
    assert result["topology"] == "probe_failed_closed"
    assert result["isolation_probe"]["runtime_id"] == "probe-runtime"
    assert generic_calls == []
    assert offline_calls == []


def test_supported_bwrap_static_gate_failure_never_reaches_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _preaction_daemon(tmp_path)
    generic_calls: list[bool] = []
    action_calls: list[bool] = []
    events: list[str] = []

    class Scheduler:
        def run_validated(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            generic_calls.append(True)
            return {}

    daemon.trusted_prevalidation_validation_scheduler = Scheduler()
    monkeypatch.setattr(
        daemon_module,
        "probe_bwrap_network_none",
        lambda **_kwargs: {
            "supported": True,
            "fallback_eligible": False,
            "returncode": 0,
        },
    )
    monkeypatch.setattr(
        daemon_module,
        "execute_state_laws_offline_validation",
        lambda **_kwargs: _offline_scheduler_result(
            mode="gate_and_pytest",
            gate_passed=False,
            returncode=1,
        ),
    )
    monkeypatch.setattr(
        daemon_module,
        "ValidationScheduler",
        _UnitOfflineScheduler,
    )
    monkeypatch.setattr(
        daemon,
        "_run_trusted_prevalidation_action",
        lambda *_args, **_kwargs: action_calls.append(True),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda name, _payload: events.append(name),
    )
    proposal = _proposal_validation(
        serial=1,
        entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
    )
    precheck, action_result = daemon._run_trusted_prevalidation_sequence(
        tmp_path,
        _task(),
        tmp_path / "action.log",
        state=None,
        proposal_validation=proposal,
    )

    assert precheck["passed"] is False
    assert precheck["static_gate_passed"] is False
    assert precheck["offline_validation_used"] is True
    assert generic_calls == []
    assert action_calls == []
    assert action_result is None
    assert events == ["trusted_pre_action_validation_blocked"]


def _materialize_evidence(
    *,
    repo: Path,
    workspace: Path,
    policy: Any,
    cohort: str = "A",
    ledger_hosts: dict[str, str] | None = None,
) -> None:
    projection, source_digest, _accepted_sources = _unit_source_projection_builder(
        workspace_path=workspace,
        cohort=cohort,
        policy=policy,
    )
    run_id = "a" * 64
    cohort_root = repo / policy.evidence_root / f"cohort_{cohort}"
    manifest_path = cohort_root / run_id / "live_run_manifest.json"
    run_root = manifest_path.parent
    selected_hosts = dict(ledger_hosts or _COHORT_A_LEDGER_HOSTS)
    artifacts: dict[str, dict[str, Any]] = {}
    for state in action_module.COHORT_JURISDICTIONS[cohort]:
        host = selected_hosts[state]
        relative = f"corpus/evidence/STATE-{state}-source-units.jsonl"
        ledger_path = run_root / relative
        ledger = {
            "schema": action_module.SOURCE_UNIT_SCHEMA,
            "state_code": state,
            "source_url": f"https://{host}/codes/{state.lower()}",
            "provenance": {
                "schema": action_module.SOURCE_UNIT_PROVENANCE_SCHEMA,
                "provider": "requests_direct",
                "acquisition_mode": "live_official",
                "requested_url": f"https://{host}/codes/{state.lower()}",
                "final_url": f"https://{host}/codes/{state.lower()}/chapter-1",
                "live_official": True,
            },
        }
        _write(
            ledger_path,
            json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
        )
        artifacts[relative] = {
            "sha256": _sha256(ledger_path),
            "size": ledger_path.stat().st_size,
        }
    manifest = {
        "schema": action_module.LIVE_RUN_SCHEMA,
        "mode": action_module.LIVE_MODE,
        "status": "candidate_complete",
        "run_id": run_id,
        "cohort_id": cohort,
        "jurisdictions": list(action_module.COHORT_JURISDICTIONS[cohort]),
        "run_fingerprint": run_id,
        "source_tree_digest": source_digest,
        "source_projection": {
            "schema": "state-laws-reindex-live-source-projection-v1",
            "files": dict(sorted(projection.items())),
        },
        "refresh": {"returncode": 0},
        "corpus_root": action_module.LIVE_CORPUS_ROOT,
        "artifacts": dict(sorted(artifacts.items())),
    }
    _write(
        manifest_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
    )
    pointer = {
        "schema": action_module.CURRENT_RUN_SCHEMA,
        "run_id": run_id,
        "manifest_path": f"{run_id}/live_run_manifest.json",
        "manifest_sha256": _sha256(manifest_path),
    }
    _write(
        cohort_root / "current_run.json",
        json.dumps(pointer, sort_keys=True, separators=(",", ":")),
    )


def _materialize_sandbox_receipt(
    *,
    argv: list[str],
    policy: Any,
    cohort: str = "A",
    acquisition_hosts: list[str] | None = None,
) -> None:
    receipt_path = Path(argv[argv.index("--sandbox-receipt") + 1])
    sandbox_policy = json.loads(
        (ACCELERATOR_ROOT / SANDBOX_POLICY_PATH).read_text(encoding="utf-8")
    )
    audit_path = receipt_path.with_name("proxy-audit.jsonl")
    _write(audit_path, "producer-shaped audit fixture\n")
    exact_hosts = sandbox_policy["network"]["cohort_exact_hosts"][cohort]
    baseline_commit = argv[argv.index("--baseline-id") + 1]
    baseline_snapshot = {
        "baseline_commit": baseline_commit,
        "baseline_regular_files_digest": "f" * 64,
        "baseline_regular_file_count": 100,
        "baseline_regular_file_bytes": 4096,
        "baseline_excluded_nonregular_count": 4,
        "baseline_excluded_nonregular_digest": "0" * 64,
    }
    acquisition_metrics = {
        host: {
            "request_count": 1,
            "upstream_bytes": 128,
            "downstream_bytes": 256,
        }
        for host in (acquisition_hosts or list(_COHORT_A_LEDGER_HOSTS.values()))
    }

    def image_receipt(name: str) -> dict[str, Any]:
        image = sandbox_policy["docker"][name]
        return {
            "image_id": image["image_id"],
            "os": image["os"],
            "architecture": image["architecture"],
            "dependency_manifest_mode": image["dependency_manifest_mode"],
            "dependency_manifest_sha256": image["dependency_manifest_sha256"],
            "labels": dict(sorted(image["required_labels"].items())),
        }

    def namespace(mode: str, digest: str) -> dict[str, Any]:
        if mode == "none_shared_forwarder":
            resource_name = "runner"
            mounts = [
                "/workspace",
                f"/evidence/cohort_{cohort}",
                "/opt/ipfs-accelerate/state_laws_live_cohort_container_v1.py",
                "/etc/chromium/policies/managed",
            ]
        elif mode == "none":
            resource_name = "forwarder"
            mounts = [
                "/opt/ipfs-accelerate/state_laws_official_egress_proxy_v1.py",
                "/run/state-laws-proxy",
            ]
        else:
            resource_name = "proxy"
            mounts = [
                "/opt/ipfs-accelerate/state_laws_official_egress_proxy_v1.py",
                "/run/audit",
                "/run/policy/allowlist.json",
                "/run/state-laws-proxy",
            ]
        resource = sandbox_policy["resources"][resource_name]
        return {
            "container_id": f"sha256:{hashlib.sha256(mode.encode()).hexdigest()}",
            "network_mode": mode,
            "network_namespace_sha256": digest,
            "published_ports": False,
            "read_only_root": True,
            "candidate_user": "1000:1000",
            "capabilities": "none",
            "no_new_privileges": True,
            "mount_destinations": sorted(mounts),
            "nofile": resource["nofile"],
            "pids_limit": resource["pids"],
        }

    assets = sandbox_policy["assets"]
    shared_namespace = "b" * 64
    allowed_metrics = {
        host: dict(metrics) for host, metrics in acquisition_metrics.items()
    }
    preflight_host = exact_hosts[0]
    preflight_metrics = allowed_metrics.setdefault(
        preflight_host,
        {"request_count": 0, "upstream_bytes": 0, "downstream_bytes": 0},
    )
    preflight_metrics["request_count"] += 1
    preflight_metrics["upstream_bytes"] += 64
    preflight_metrics["downstream_bytes"] += 128
    proxy_audit = {
        "sha256": _sha256(audit_path),
        "size": audit_path.stat().st_size,
        "event_count": len(acquisition_metrics) + 3,
        "allowed_count": len(acquisition_metrics) + 1,
        "rejected_count": 0,
        "failed_count": 0,
        "ready_count": 1,
        "terminal_count": 1,
        "terminal_status": "signal_stop",
        "first_event": "proxy_ready",
        "last_event": "proxy_terminal",
        "allowed_hosts": sorted(allowed_metrics),
        "allowed_host_metrics": dict(sorted(allowed_metrics.items())),
        "preflight_allowed_count": 1,
        "acquisition_allowed_count": len(acquisition_metrics),
        "acquisition_host_metrics": acquisition_metrics,
        "acquisition_host_metrics_digest": action_module._content_id(
            acquisition_metrics
        ),
    }
    cleanup_token = "9" * 24
    cleanup_runtime_root = sandbox_policy["process"]["runtime_root"]
    cleanup_lease_root = (
        f"{cleanup_runtime_root}/ipfs-accelerate-slr-action-{cleanup_token}"
    )
    cleanup_properties = [
        "Type=exec",
        "KillMode=control-group",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "RestrictAddressFamilies=AF_UNIX",
        "MemoryMax=134217728",
        "TasksMax=32",
        "UMask=0077",
        f"ReadWritePaths={cleanup_lease_root}",
        f"WorkingDirectory={cleanup_lease_root}",
    ]
    smoke_raw = {
        "schema": "state-laws-playwright-smoke-v1",
        "cohort": cohort,
        "attempts": [
            {
                "requested_url": f"https://{exact_hosts[0]}/",
                "status": "response",
                "final_url": f"https://{exact_hosts[0]}/",
                "http_status": 200,
            }
        ],
        "successful_requested_url": f"https://{exact_hosts[0]}/",
        "successful_final_url": f"https://{exact_hosts[0]}/",
        "http_status": 200,
    }
    smoke_bytes = action_module._canonical_bytes(smoke_raw) + b"\n"
    playwright_smoke = {
        **smoke_raw,
        "sha256": hashlib.sha256(smoke_bytes).hexdigest(),
        "size": len(smoke_bytes),
    }
    receipt = {
        "schema": action_module.SANDBOX_RECEIPT_SCHEMA,
        "handler": "state_laws_live_cohort_v1",
        "status": "completed",
        "cohort": cohort,
        "policy_sha256": policy.sandbox.policy_sha256,
        "runner_image": image_receipt("runner_image"),
        "proxy_image": image_receipt("proxy_image"),
        "proxy_script_sha256": assets["proxy_script_sha256"],
        "container_launcher_sha256": assets["container_launcher_sha256"],
        "dockerfile_sha256": assets["dockerfile_sha256"],
        "proxy_dockerfile_sha256": assets["proxy_dockerfile_sha256"],
        "chromium_policy_sha256": assets["chromium_policy_sha256"],
        "sandbox_runtime_sha256": assets["sandbox_runtime_sha256"],
        "docker_client_sha256": sandbox_policy["docker"]["client_sha256"],
        "candidate_environment_digest": sandbox_policy["process"][
            "candidate_environment_digest"
        ],
        "catalog_sha256": sandbox_policy["network"]["catalog_sha256"],
        "allowlist_digest": action_module._content_id(
            {
                "schema": action_module.SANDBOX_ALLOWLIST_SCHEMA,
                "cohort": cohort,
                "hostname_match": "exact",
                "domains": exact_hosts,
            }
        ),
        "allowlist_exact_hosts": exact_hosts,
        "source_projection_digest": argv[argv.index("--source-projection-digest") + 1],
        **baseline_snapshot,
        "candidate_workspace_snapshot_digest": action_module._content_id(
            {
                **baseline_snapshot,
                "source_projection_digest": argv[
                    argv.index("--source-projection-digest") + 1
                ],
            }
        ),
        "git_client_sha256": sandbox_policy["process"]["git_client_sha256"],
        "playwright_instrumentation_digest": "d" * 64,
        "playwright_launch_count": 8,
        "playwright_helper_contract": "playwright_launch_kwargs@1",
        "playwright_smoke": playwright_smoke,
        "preflight_runner_namespace": namespace(
            "none_shared_forwarder", shared_namespace
        ),
        "runner_namespace": namespace("none_shared_forwarder", shared_namespace),
        "forwarder_namespace": namespace("none", shared_namespace),
        "proxy_namespace": namespace("bridge", "e" * 64),
        "transport": "isolated_forwarder_loopback_to_readonly_unix_socket_v1",
        "authority_claim_scope": (
            "exact_host_sni_resolved_public_peer_timing_and_byte_counts_only"
        ),
        "trusted_response_content_binding": False,
        "proxy_orderly_stop": True,
        "forwarder_orderly_stop": True,
        "proxy_audit_path": str(audit_path),
        "proxy_audit": proxy_audit,
        "runner_returncode": 0,
        "cleanup": {
            "runner_verification_removed": True,
            "proxy_verification_removed": True,
            "runner_removed": True,
            "forwarder_removed": True,
            "proxy_removed": True,
            "remaining_owned_container_count": 0,
            "complete": True,
        },
        "cleanup_service": {
            "unit": f"ipfs-accelerate-slr-cleanup-{cleanup_token}.service",
            "main_pid": 424243,
            "main_pid_starttime": "123456",
            "control_group_sha256": "8" * 64,
            "properties_digest": action_module._content_id(cleanup_properties),
            "systemd_run_sha256": sandbox_policy["process"]["systemd_run_sha256"],
            "systemctl_sha256": sandbox_policy["process"]["systemctl_sha256"],
            "cleanup_python_sha256": sandbox_policy["process"]["cleanup_python_sha256"],
            "cleanup_env_sha256": sandbox_policy["process"]["cleanup_env_sha256"],
            "runtime_root": cleanup_runtime_root,
            "independent_cgroup": True,
        },
        "startup_sweep_removed_container_ids_sha256": [],
        "startup_sweep_removed_lease_ids_sha256": [],
        "cleanup_claim_scope": (
            "independent_user_transient_service_plus_dead_owner_exact_label_sweep"
        ),
        "started_at": "2026-08-10T00:00:00Z",
        "finished_at": "2026-08-10T00:01:00Z",
        "failure": "",
    }
    _write(
        receipt_path,
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
    )


class _CompletedProcess:
    pid = 424242

    def poll(self) -> int:
        return 0

    def wait(self) -> int:
        return 0


def test_policy_is_closed_and_seals_all_thirteen_cohorts() -> None:
    payload = _policy_payload()

    policy = parse_state_laws_live_cohort_policy(payload)

    assert policy.cohort_for_task("SLR-009") == "A"
    assert policy.cohort_for_task("SLR-021") == "M"
    assert policy.cohort_for_task("SLR-022") is None
    assert policy.payload["validation_receipt_sha256_environment_variable"] == (
        STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV
    )
    hostile = dict(payload)
    hostile["command"] = "curl https://example.invalid"
    with pytest.raises(action_module.TrustedPrevalidationActionError):
        parse_state_laws_live_cohort_policy(hostile)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("trusted_static_pre_action_gate", "untrusted_gate"),
        ("adapter_evidence_ownership", "shared_adapter_ownership"),
        ("adapter_contract_catalog_binding_required", False),
        ("adapter_contract_closed_mode_schema_required", False),
        ("adapter_contract_reviewed_policy_digest_required", False),
        ("adapter_evidence_api_signature_and_dataflow_required", False),
        ("source_closure_policy_registry_path", "data/untrusted.json"),
        ("source_closure_policy_registry_schema", "untrusted-registry-v1"),
    ],
)
def test_policy_requires_exact_static_gate_and_reviewed_registry(
    field: str,
    invalid: object,
) -> None:
    payload = _policy_payload()
    payload[field] = invalid

    with pytest.raises(action_module.TrustedPrevalidationActionError):
        parse_state_laws_live_cohort_policy(payload)

    missing = _policy_payload()
    del missing[field]
    with pytest.raises(action_module.TrustedPrevalidationActionError):
        parse_state_laws_live_cohort_policy(missing)


def test_policy_requires_catalog_and_registry_as_first_shared_sources() -> None:
    payload = _policy_payload()
    source = dict(payload["source_projection"])
    source["shared_paths"] = list(source["shared_paths"])[1:]
    payload["source_projection"] = source

    with pytest.raises(
        action_module.TrustedPrevalidationActionError,
        match="must begin with the sealed official catalog",
    ):
        parse_state_laws_live_cohort_policy(payload)


def test_accepted_projection_materializes_exact_baseline_and_diff(
    tmp_path: Path,
) -> None:
    workspace, policy, validation, adapter_entry, test_entry = (
        _accepted_projection_case(tmp_path)
    )
    action_binding = action_module._proposal_binding(validation, cohort="A")

    projection, digest, accepted_sources = (
        action_module.accepted_proposal_source_projection(
            workspace_path=workspace,
            cohort="A",
            policy=policy,
            proposal_validation=validation,
            proposal_action_diff_digest=action_binding["proposal_action_diff_digest"],
            required_changed_paths=(test_entry.new_path,),
            allowed_extra_changed_paths=(test_entry.new_path,),
        )
    )

    assert (
        projection[adapter_entry.new_path]
        == hashlib.sha256(adapter_entry.after_source.encode("utf-8")).hexdigest()
    )
    assert accepted_sources == {
        adapter_entry.new_path: adapter_entry.after_source.encode("utf-8"),
        test_entry.new_path: test_entry.after_source.encode("utf-8"),
    }
    assert digest == action_module._content_id(
        {
            "schema": "state-laws-reindex-live-source-projection-v1",
            "cohort_id": "A",
            "files": projection,
        }
    )


@pytest.mark.parametrize("mutation", ["unchanged_shared", "changed_adapter"])
def test_accepted_projection_workspace_mutation_blocks_live_action_before_spawn(
    tmp_path: Path,
    mutation: str,
) -> None:
    workspace, policy, validation, _adapter_entry, _test_entry = (
        _accepted_projection_case(tmp_path)
    )
    target = (
        workspace / "shared/policy.py"
        if mutation == "unchanged_shared"
        else workspace / "cohorts/a_adapter.py"
    )
    target.write_text("MUTATED AFTER ACCEPTANCE\n", encoding="utf-8")
    sandbox_calls: list[bool] = []
    process_calls: list[bool] = []

    with pytest.raises(
        action_module.TrustedPrevalidationActionError,
        match="workspace bytes differ from accepted proposal",
    ):
        execute_state_laws_live_cohort_action(
            policy=policy,
            repo_root=tmp_path,
            workspace_path=workspace,
            task=_task(),
            canonical_task_cid="cid-slr-009",
            proposal_validation=validation,
            attempt=1,
            log_path=tmp_path / "mutation.log",
            sandbox_capability=lambda **_kwargs: sandbox_calls.append(True),
            popen=lambda *_args, **_kwargs: process_calls.append(True),
        )

    assert sandbox_calls == []
    assert process_calls == []


@pytest.mark.parametrize(
    "case",
    [
        "add_at_existing",
        "modify_at_absent",
        "type_change",
        "missing_required_test",
        "unsafe_baseline_mode",
    ],
)
def test_accepted_projection_rejects_ambiguous_or_unsafe_diff(
    tmp_path: Path,
    case: str,
) -> None:
    workspace, policy, validation, adapter_entry, test_entry = (
        _accepted_projection_case(
            tmp_path,
            unsafe_shared_mode=case == "unsafe_baseline_mode",
        )
    )
    if case == "add_at_existing":
        entries = (
            replace(
                adapter_entry,
                old_path="",
                change_kind=DiffChangeKind.ADD,
                before_source=None,
                before_blob_id="",
            ),
            test_entry,
        )
    elif case == "modify_at_absent":
        entries = (
            adapter_entry,
            replace(
                test_entry,
                old_path=test_entry.new_path,
                change_kind=DiffChangeKind.MODIFY,
                before_source="not in baseline\n",
                before_blob_id=_git_blob("not in baseline\n"),
            ),
        )
    elif case == "type_change":
        entries = (
            replace(adapter_entry, change_kind=DiffChangeKind.TYPE_CHANGE),
            test_entry,
        )
    elif case == "missing_required_test":
        entries = (adapter_entry,)
    else:
        entries = (adapter_entry, test_entry)
    validation = _proposal_with_entries(validation, entries)
    binding = action_module._proposal_binding(validation, cohort="A")

    with pytest.raises(action_module.TrustedPrevalidationActionError):
        action_module.accepted_proposal_source_projection(
            workspace_path=workspace,
            cohort="A",
            policy=policy,
            proposal_validation=validation,
            proposal_action_diff_digest=binding["proposal_action_diff_digest"],
            required_changed_paths=(test_entry.new_path,),
            allowed_extra_changed_paths=(test_entry.new_path,),
        )


def test_offline_overlay_mutation_is_deterministic_before_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, policy, validation, _adapter_entry, _test_entry = (
        _accepted_projection_case(tmp_path)
    )
    (workspace / "shared/policy.py").write_text(
        "MUTATED AFTER ACCEPTANCE\n",
        encoding="utf-8",
    )
    sandbox_calls: list[bool] = []
    monkeypatch.setattr(
        action_module,
        "_sandbox_capability",
        lambda **_kwargs: sandbox_calls.append(True),
    )

    result = offline_validation.execute_state_laws_offline_validation(
        policy=policy,
        repo_root=tmp_path,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=validation,
        command=pre_action_validation_command("A"),
        timeout_seconds=30,
    )

    assert result["returncode"] == 1
    assert result["classification"] == "deterministic_failure"
    receipt = result["state_laws_offline_validation"]
    assert receipt["mode"] == "proposal_overlay_refusal"
    assert receipt["phase_container_ids"] == []
    assert receipt["cleanup"]["remaining_owned_container_count"] == 0
    assert sandbox_calls == []


def test_unavailable_sandbox_fails_before_child_launch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    launches: list[Any] = []

    result = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
        ),
        attempt=1,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        sandbox_capability=lambda **kwargs: (_ for _ in ()).throw(
            action_module.TrustedPrevalidationActionError(
                "sealed Docker acquisition capability unavailable"
            )
        ),
        popen=lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    assert result["passed"] is False
    assert result["attempted"] is False
    assert result["reason"] == "sandbox_unavailable"
    assert result["blocker"] == "sealed Docker acquisition capability unavailable"
    assert launches == []
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["completion_authoritative"] is False


def test_cohort_lock_blocks_same_cohort_but_not_another_cohort(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    receipt_root = repo / policy.receipt_root
    receipt_root.mkdir(parents=True)
    lock_path = receipt_root / "cohort_A.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        same = execute_state_laws_live_cohort_action(
            policy=policy,
            repo_root=repo,
            workspace_path=workspace,
            task=_task("A"),
            canonical_task_cid="cid-slr-009",
            proposal_validation=_proposal_validation(
                serial=1,
                entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
            ),
            attempt=1,
            log_path=repo / "a.log",
            source_projection_builder=_unit_source_projection_builder,
        )
        assert same["reason"] == "trusted_prevalidation_action_cohort_lease_busy"

        different = execute_state_laws_live_cohort_action(
            policy=policy,
            repo_root=repo,
            workspace_path=workspace,
            task=_task("B"),
            canonical_task_cid="cid-slr-010",
            proposal_validation=_proposal_validation(
                serial=2,
                entries=(_DiffEntry("", "cohorts/b_adapter.py", "adapter"),),
            ),
            attempt=1,
            log_path=repo / "b.log",
            source_projection_builder=_unit_source_projection_builder,
            sandbox_capability=lambda **kwargs: (_ for _ in ()).throw(
                action_module.TrustedPrevalidationActionError("expected test stop")
            ),
        )
        assert different["reason"] == "sandbox_unavailable"
    finally:
        os.close(descriptor)


def test_action_uses_fixed_runner_argv_and_exact_scrubbed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    captured: dict[str, Any] = {}
    monkeypatch.setenv("HF_TOKEN", "must-not-cross")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-cross.invalid")
    monkeypatch.setenv("STATE_SCRAPER_MAX_STATUTES", "1")

    def fake_popen(argv: list[str], **kwargs: Any) -> _CompletedProcess:
        captured["argv"] = list(argv)
        captured["environment"] = dict(kwargs["env"])
        _materialize_evidence(
            repo=repo,
            workspace=workspace,
            policy=policy,
        )
        _materialize_sandbox_receipt(argv=argv, policy=policy)
        return _CompletedProcess()

    result = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(_DiffEntry("", "cohorts/a_adapter.py", "adapter"),),
        ),
        attempt=1,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        sandbox_capability=lambda **kwargs: (
            ACCELERATOR_ROOT / WRAPPER_PATH,
            ACCELERATOR_ROOT / SANDBOX_POLICY_PATH,
        ),
        popen=fake_popen,
    )

    assert result["passed"] is True
    assert result["runner_argv"] == [
        str(Path(sys.executable).resolve()),
        str((workspace / action_module.RUNNER_PATH).resolve()),
        "--cohort",
        "A",
        "--execute-live",
        "--max-parallel",
        "2",
    ]
    assert captured["environment"] == action_module._exact_environment(
        repo / policy.evidence_root
    )
    assert not {
        "HF_TOKEN",
        "HTTPS_PROXY",
        "STATE_SCRAPER_MAX_STATUTES",
    } & set(captured["environment"])
    assert set(result["scrubbed_environment"]) == set(
        action_module.SCRUBBED_ENVIRONMENT_NAMES
    )
    assert result["evidence_ledger_live_official_hosts"] == sorted(
        _COHORT_A_LEDGER_HOSTS.values()
    )
    assert result["evidence_ledger_live_official_hosts_digest"] == (
        action_module._content_id(sorted(_COHORT_A_LEDGER_HOSTS.values()))
    )


@pytest.mark.parametrize("mutation", ["checksum", "unauthorized_host", "extra_field"])
def test_sandbox_playwright_smoke_tampering_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    _projection, source_digest, _accepted_sources = _unit_source_projection_builder(
        workspace_path=workspace,
        cohort="A",
        policy=policy,
    )
    baseline = "1" * 40
    receipt_path = repo / "action" / "sandbox-receipt.json"
    argv = [
        "sealed-sandbox",
        "--sandbox-receipt",
        str(receipt_path),
        "--source-projection-digest",
        source_digest,
        "--baseline-id",
        baseline,
    ]
    _materialize_sandbox_receipt(argv=argv, policy=policy)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    smoke = payload["playwright_smoke"]
    if mutation == "checksum":
        smoke["sha256"] = "0" * 64
    elif mutation == "unauthorized_host":
        hostile = "https://evil.invalid/"
        smoke["attempts"][0]["requested_url"] = hostile
        smoke["attempts"][0]["final_url"] = hostile
        smoke["successful_requested_url"] = hostile
        smoke["successful_final_url"] = hostile
        raw = {
            name: value
            for name, value in smoke.items()
            if name not in {"sha256", "size"}
        }
        body = action_module._canonical_bytes(raw) + b"\n"
        smoke["sha256"] = hashlib.sha256(body).hexdigest()
        smoke["size"] = len(body)
    else:
        smoke["provider_note"] = "ignored fields must not be accepted"
    receipt_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(
        action_module.TrustedPrevalidationActionError,
        match="Playwright",
    ):
        action_module._sandbox_receipt_binding(
            sandbox_receipt_path=receipt_path,
            sandbox_policy_path=ACCELERATOR_ROOT / SANDBOX_POLICY_PATH,
            expected_policy_sha256=policy.sandbox.policy_sha256,
            cohort="A",
            source_projection_digest=source_digest,
            baseline_commit=baseline,
        )


@pytest.mark.parametrize("mutation", ["absent", "zero_downstream"])
def test_live_evidence_rejects_ledger_host_without_acquisition_traffic(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    _materialize_evidence(repo=repo, workspace=workspace, policy=policy)
    projection, source_digest, _accepted_sources = _unit_source_projection_builder(
        workspace_path=workspace,
        cohort="A",
        policy=policy,
    )
    metrics = {
        host: {
            "request_count": 1,
            "upstream_bytes": 128,
            "downstream_bytes": 256,
        }
        for host in _COHORT_A_LEDGER_HOSTS.values()
    }
    target = _COHORT_A_LEDGER_HOSTS["AR"]
    if mutation == "absent":
        del metrics[target]
    else:
        metrics[target]["downstream_bytes"] = 0

    with pytest.raises(
        action_module.TrustedPrevalidationActionError,
        match="ledger host is absent from acquisition traffic",
    ):
        action_module._live_evidence_binding(
            evidence_root=repo / policy.evidence_root,
            cohort="A",
            expected_source_projection=projection,
            expected_source_digest=source_digest,
            acquisition_host_metrics=metrics,
        )


def test_provider_materialized_report_blocks_first_acquisition(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    _write(
        workspace / "docs/reports/state_laws_reindex/cohort_a.json",
        "provider placeholder\n",
    )
    launches: list[bool] = []

    result = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(
            serial=1,
            entries=(
                _DiffEntry("", "cohorts/a_adapter.py", "adapter"),
                _DiffEntry(
                    "",
                    "docs/reports/state_laws_reindex/cohort_a.json",
                    "provider placeholder",
                ),
            ),
        ),
        attempt=1,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        sandbox_capability=lambda **kwargs: launches.append(True),
    )

    assert result["passed"] is False
    assert result["attempted"] is False
    assert result["reason"] == ("validation_materialized_output_present_before_action")
    assert launches == []


def test_report_only_reproposal_reuses_and_nonreport_diff_does_not(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    workspace = _workspace(repo)
    policy = parse_state_laws_live_cohort_policy(_policy_payload())
    adapter = _DiffEntry("", "cohorts/a_adapter.py", "adapter")

    def fake_popen(argv: list[str], **kwargs: Any) -> _CompletedProcess:
        del kwargs
        _materialize_evidence(repo=repo, workspace=workspace, policy=policy)
        _materialize_sandbox_receipt(argv=argv, policy=policy)
        return _CompletedProcess()

    first = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(serial=1, entries=(adapter,)),
        attempt=1,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        sandbox_capability=lambda **kwargs: (
            ACCELERATOR_ROOT / WRAPPER_PATH,
            ACCELERATOR_ROOT / SANDBOX_POLICY_PATH,
        ),
        popen=fake_popen,
    )
    assert first["passed"] is True
    first_receipt = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert set(first_receipt) == action_module._TERMINAL_RECEIPT_FIELDS
    assert first_receipt["reused_from_receipt_id"] == ""
    assert first_receipt["rebound_at"] is None

    report = _DiffEntry(
        "",
        "docs/reports/state_laws_reindex/cohort_a.json",
        "certified report",
    )
    _write(
        workspace / "docs/reports/state_laws_reindex/cohort_a.json",
        "certified report\n",
    )
    reused = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(
            serial=2,
            entries=(adapter, report),
        ),
        attempt=2,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        popen=lambda *args, **kwargs: pytest.fail("receipt was not reused"),
    )
    assert reused["passed"] is True
    assert reused["reused"] is True
    assert reused["repository_tree_id"] == "tree-2"
    assert reused["reused_from_receipt_id"] == first["receipt_id"]
    rebound_receipt = json.loads(
        Path(reused["receipt_path"]).read_text(encoding="utf-8")
    )
    assert set(rebound_receipt) == action_module._TERMINAL_RECEIPT_FIELDS
    assert rebound_receipt["reason"] == ("trusted_prevalidation_action_receipt_reused")
    assert (
        reused["candidate_source_fingerprint"]
        == (first["candidate_source_fingerprint"])
    )

    (workspace / "docs/reports/state_laws_reindex/cohort_a.json").unlink()
    nonreport = _DiffEntry("", "docs/other.json", "meaningful change")
    rejected = execute_state_laws_live_cohort_action(
        policy=policy,
        repo_root=repo,
        workspace_path=workspace,
        task=_task(),
        canonical_task_cid="cid-slr-009",
        proposal_validation=_proposal_validation(
            serial=3,
            entries=(adapter, nonreport),
        ),
        attempt=3,
        log_path=repo / "action.log",
        source_projection_builder=_unit_source_projection_builder,
        sandbox_capability=lambda **kwargs: (_ for _ in ()).throw(
            action_module.TrustedPrevalidationActionError(
                "new non-report proposal requires a new acquisition"
            )
        ),
    )
    assert rejected["passed"] is False
    assert rejected["reason"] == "sandbox_unavailable"
    assert rejected["receipt_path"] != first["receipt_path"]


def test_failed_hermetic_precheck_never_invokes_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = object.__new__(PortalImplementationDaemon)
    task = SimpleNamespace(task_id="SLR-009")
    events: list[tuple[str, Any]] = []
    launches: list[bool] = []
    monkeypatch.setattr(
        daemon,
        "_run_trusted_pre_action_validation",
        lambda *args, **kwargs: {
            "required": True,
            "attempted": True,
            "passed": False,
            "reason": "pytest_failed",
            "command": pre_action_validation_command("A"),
        },
    )
    monkeypatch.setattr(
        daemon,
        "_run_trusted_prevalidation_action",
        lambda *args, **kwargs: launches.append(True),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda name, payload: events.append((name, payload)),
    )

    precheck, action = daemon._run_trusted_prevalidation_sequence(
        tmp_path,
        task,
        tmp_path / "action.log",
        state=None,
        proposal_validation=SimpleNamespace(accepted=True),
    )

    assert precheck["passed"] is False
    assert action is None
    assert launches == []
    assert events[0][0] == "trusted_pre_action_validation_blocked"


def test_receipt_bytes_are_bound_into_environment_and_cache_identity(
    tmp_path: Path,
) -> None:
    receipt_path = (tmp_path / "receipt.json").resolve()
    receipt_path.write_bytes(b'{"serial":1}\n')
    first_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    first_environment = build_validation_environment(
        {
            STATE_LAWS_TRUSTED_ACTION_RECEIPT_ENV: str(receipt_path),
            STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV: first_sha,
        }
    )
    first_key = build_validation_cache_key(
        target_commit="candidate-tree",
        command="python -m pytest -q",
        environment=first_environment,
        dependency_state="same-candidate",
    )

    receipt_path.write_bytes(b'{"serial":2}\n')
    second_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValidationRuntimeError, match="SHA-256 mismatch"):
        build_validation_environment(
            {
                STATE_LAWS_TRUSTED_ACTION_RECEIPT_ENV: str(receipt_path),
                STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV: first_sha,
            }
        )
    second_environment = build_validation_environment(
        {
            STATE_LAWS_TRUSTED_ACTION_RECEIPT_ENV: str(receipt_path),
            STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV: second_sha,
        }
    )
    second_key = build_validation_cache_key(
        target_commit="candidate-tree",
        command="python -m pytest -q",
        environment=second_environment,
        dependency_state="same-candidate",
    )

    assert (
        first_environment[STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV]
        != (second_environment[STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV])
    )
    assert first_key.digest != second_key.digest
    for variable in (
        STATE_LAWS_TRUSTED_ACTION_RECEIPT_ENV,
        STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256_ENV,
    ):
        with pytest.raises(ValidationRuntimeError, match="may not override"):
            validation_shell_command(f"{variable}=hostile python -V")


def test_live_action_progress_marker_tracks_nested_corpus_writes(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "impl.log"
    log_path.write_bytes(b"start\n")
    cohort_root = tmp_path / "cohort_A"
    run_dir = cohort_root / "b9dd9a92b5eeebad0987788361d599673d88384ea1565a3dd25982bd59d797d9"
    progress = run_dir / "corpus" / "state_refresh_progress.json"
    archived = (
        cohort_root
        / "archived"
        / "old-run"
        / "corpus"
        / "state_refresh_progress.json"
    )
    progress.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    progress.write_text('{"status":"running","completed_count":0}\n', encoding="utf-8")
    archived.write_text('{"status":"failed"}\n', encoding="utf-8")

    first = _live_action_progress_marker(log_path=log_path, cohort_root=cohort_root)
    second = _live_action_progress_marker(log_path=log_path, cohort_root=cohort_root)
    assert first == second

    archived.write_text('{"status":"failed","ignored":true}\n', encoding="utf-8")
    after_archive = _live_action_progress_marker(
        log_path=log_path, cohort_root=cohort_root
    )
    assert after_archive == first

    progress.write_text(
        '{"status":"running","completed_count":0,"updated_at":"later"}\n',
        encoding="utf-8",
    )
    after_progress = _live_action_progress_marker(
        log_path=log_path, cohort_root=cohort_root
    )
    assert after_progress != first
    assert after_progress[3] >= first[3]
    assert after_progress[4] > first[4]
