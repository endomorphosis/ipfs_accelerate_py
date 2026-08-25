"""Tests for deterministic same-attempt implementation auto-rescue."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.diagnostics import (
    summarize_test_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    build_inline_provider_rescue_prompt,
    derive_materialize_commands,
    is_undeclared_helper_path,
    plan_automatic_implementation_rescue,
)

ROOT = Path(__file__).resolve().parents[2]
VRIF_MATERIALIZER = "scripts/materialize_vrif_frozen_benchmark.py"
VRIF_DECLARED_OUTPUTS = (
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
    "test/api/residual_intelligence/test_benchmark.py",
)
VRIF_OUTPUTS = (
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
    "test/api/residual_intelligence/test_benchmark.py",
)
VRIF_VALIDATION = (
    "python3 -m pytest -q test/api/residual_intelligence/test_benchmark.py"
)
VRIF_CHANGED_PATHS = (
    "test/api/residual_intelligence/test_benchmark.py",
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
)


def test_vrif_owner_recovery_constants_remain_exactly_sealed() -> None:
    assert (
        implementation_daemon.VRIF_BENCHMARK_RECOVERY_DECLARED_OUTPUTS
        == VRIF_DECLARED_OUTPUTS
    )
    assert implementation_daemon.VRIF_BENCHMARK_RECOVERY_OUTPUTS == VRIF_OUTPUTS
    assert implementation_daemon.VRIF_BENCHMARK_RECOVERY_VALIDATION == VRIF_VALIDATION


def test_vrif_owner_environment_is_exact_and_disables_replace_objects() -> None:
    assert PortalImplementationDaemon._vrif_benchmark_owner_environment() == {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
    }


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _vrif_clean_repository(tmp_path: Path) -> tuple[Path, str, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "vrif-rescue@example.invalid")
    _git(workspace, "config", "user.name", "VRIF rescue fixture")
    trusted_code_paths = tuple(
        dict.fromkeys(
            (
                VRIF_MATERIALIZER,
                *implementation_daemon.VRIF_BENCHMARK_RECOVERY_TRUSTED_CODE_PATHS,
                *implementation_daemon.VRIF_BENCHMARK_RECOVERY_PRIVATE_PACKAGE_INITS,
            )
        )
    )
    for relative in trusted_code_paths:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    data_paths = tuple(implementation_daemon.VRIF_BENCHMARK_RECOVERY_DATA_PATHS)
    for relative in data_paths:
        data_path = workspace / relative
        if data_path.exists():
            continue
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(f"baseline input for {relative}\n", encoding="utf-8")
    for relative in VRIF_OUTPUTS:
        output = workspace / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"baseline placeholder for {relative}\n", encoding="utf-8")
    _git(workspace, "add", *trusted_code_paths, *data_paths, *VRIF_OUTPUTS)
    _git(workspace, "commit", "-qm", "trusted VRIF materializer baseline")
    baseline = _git(workspace, "rev-parse", "HEAD")
    tree = _git(workspace, "rev-parse", "HEAD^{tree}")
    return workspace, baseline, tree


def _vrif_task(**changes: object) -> PortalTask:
    task = PortalTask(
        task_id="VRIF-030",
        title="Build frozen paired benchmark",
        status="in_progress",
        completion="auto",
        priority="P1",
        track="implementation",
        outputs=list(VRIF_DECLARED_OUTPUTS),
        validation=[VRIF_VALIDATION],
        canonical_task_cid="baguqeera-vrif-030-fixture",
    )
    return replace(task, **changes)


def test_vrif_owner_gate_reserves_provider_free_empty_candidate(
    tmp_path: Path,
) -> None:
    daemon = PortalImplementationDaemon(
        todo_path=ROOT
        / "docs/architecture/agent_supervisor_residual_intelligence.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=ROOT,
        implement=True,
    )

    reserved = daemon._evaluate_pre_implementation_provider_gate(
        task=_vrif_task(),
        attempt=1,
        worktree_path=ROOT,
    )
    ordinary = daemon._evaluate_pre_implementation_provider_gate(
        task=_vrif_task(
            task_id="VRIF-999",
            outputs=["ordinary.txt"],
            validation=["python3 -m py_compile ordinary.py"],
            canonical_task_cid="baguqeera-ordinary-fixture",
        ),
        attempt=1,
        worktree_path=ROOT,
    )

    assert reserved["disposition"] == "abstain_review"
    assert reserved["reason_code"] == "no_analytical_close"
    assert reserved["skip_provider"] is True
    assert reserved["provider_authorized"] is False
    assert reserved["owner_recovery_reserved"] is True
    assert ordinary["reason_code"] == "no_analytical_close_provider_dispatched"
    assert ordinary["skip_provider"] is False
    assert ordinary["provider_authorized"] is True
    assert ordinary["owner_recovery_reserved"] is False


def _vrif_exact_empty_patch_result(baseline: str) -> dict[str, object]:
    proposal_id = "1" * 64
    policy_id = "2" * 64
    receipt_id = "3" * 64
    canonical_task_cid = str(_vrif_task().canonical_task_cid)
    findings = [
        ["empty_patch", "patch", "candidate diff contains no file changes", ""],
        [
            "missing_required_field",
            "structure",
            "structured proposal requires operations",
            "",
        ],
        [
            "missing_required_field",
            "structure",
            "structured proposal requires patch_text",
            "",
        ],
    ]
    return {
        "attempted": False,
        "passed": False,
        "returncode": implementation_daemon.PROPOSAL_VALIDATION_FAILURE_RETURN_CODE,
        "results": [],
        "reason": "no_change_completion_not_allowed",
        "proposal_gate": {
            "accepted": False,
            "attempted": True,
            "changed_paths": [],
            "completion_authoritative": False,
            "policy_id": policy_id,
            "proof_authoritative": False,
            "proposal_id": proposal_id,
            "reason": "empty_patch_reserved_for_no_change_gate",
            "reason_codes": ["empty_patch", "missing_required_field"],
            "receipt_id": receipt_id,
            "repository_tree_id": baseline,
        },
        "no_change_policy_gate": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor/"
                "no-change-candidate-policy-gate@1"
            ),
            "accepted": False,
            "attempted": True,
            "actual_findings": findings,
            "baseline_id": baseline,
            "candidate_fingerprint": (
                "sha256:"
                + implementation_daemon.VRIF_BENCHMARK_RECOVERY_EMPTY_DIFF_DIGEST
            ),
            "canonical_task_cid": canonical_task_cid,
            "changed_paths": [],
            "completion_authoritative": False,
            "completion_mode": "",
            "context_id": canonical_task_cid,
            "diff_digest": (
                implementation_daemon.VRIF_BENCHMARK_RECOVERY_EMPTY_DIFF_DIGEST
            ),
            "expected_findings": copy.deepcopy(findings),
            "objective_id": canonical_task_cid,
            "policy_id": policy_id,
            "proof_authoritative": False,
            "proposal_accepted": False,
            "proposal_collection_error": "",
            "proposal_id": proposal_id,
            "proposal_receipt_id": receipt_id,
            "reason": "no_change_completion_not_allowed",
            "repository_tree_id": baseline,
            "task_id": "VRIF-030",
        },
        "failure_review": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "implementation-failure-review@1"
            ),
            "task_id": "VRIF-030",
            "accepted": False,
            "decision": "guide_rescue",
            "finding_codes": ["empty_patch", "missing_required_field"],
            "reason_codes": ["generic_implementation_failure"],
            "changed_paths": [],
            "denied_paths": [],
            "out_of_scope_paths": [],
            "missing_expected_outputs": [],
            "failed_commands": [],
            "expected_outputs": list(VRIF_OUTPUTS),
            "completion_authoritative": False,
            "proof_authoritative": False,
        },
    }


def test_vrif_owner_materialize_argv_requires_exact_clean_gate_and_repository(
    tmp_path: Path,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)

    argv = PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
        workspace_path=workspace,
        task=_vrif_task(),
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
    )

    assert argv == (
        sys.executable,
        "-I",
        "-S",
        "-B",
        VRIF_MATERIALIZER,
        "--repo-root",
        str(workspace.resolve()),
        "--baseline-commit",
        baseline,
        "--write",
    )


@pytest.mark.parametrize(
    ("task_changes", "gate_mutation"),
    [
        ({"task_id": "OTHER-030"}, None),
        ({"outputs": list(VRIF_DECLARED_OUTPUTS[:-1])}, None),
        ({"outputs": [*VRIF_DECLARED_OUTPUTS, "extra.json"]}, None),
        ({"outputs": list(reversed(VRIF_DECLARED_OUTPUTS))}, None),
        ({"validation": [VRIF_VALIDATION.replace("python3", "python")]}, None),
        ({"validation": [VRIF_VALIDATION, "true"]}, None),
        ({}, "returncode"),
        ({}, "extra_reason_code"),
        ({}, "missing_finding"),
        ({}, "findings_disagree"),
        ({}, "wrong_baseline"),
        ({}, "canonical_cid_mismatch"),
        ({}, "reject_review"),
    ],
)
def test_vrif_owner_materialize_argv_rejects_nonexact_task_or_gate(
    tmp_path: Path,
    task_changes: dict[str, object],
    gate_mutation: str | None,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    gate = _vrif_exact_empty_patch_result(baseline)
    proposal_gate = gate["proposal_gate"]
    no_change_gate = gate["no_change_policy_gate"]
    assert isinstance(proposal_gate, dict)
    assert isinstance(no_change_gate, dict)
    if gate_mutation == "returncode":
        gate["returncode"] = 0
    elif gate_mutation == "extra_reason_code":
        proposal_gate["reason_codes"] = [
            "empty_patch",
            "missing_required_field",
            "scope_expansion_denied",
        ]
    elif gate_mutation == "missing_finding":
        no_change_gate["actual_findings"] = no_change_gate["actual_findings"][:-1]
    elif gate_mutation == "findings_disagree":
        no_change_gate["expected_findings"] = []
    elif gate_mutation == "wrong_baseline":
        no_change_gate["baseline_id"] = "f" * 40
    elif gate_mutation == "canonical_cid_mismatch":
        no_change_gate["canonical_task_cid"] = "baguqeera-foreign-task"
    elif gate_mutation == "reject_review":
        failure_review = gate["failure_review"]
        assert isinstance(failure_review, dict)
        failure_review["decision"] = "reject"

    assert (
        PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
            workspace_path=workspace,
            task=_vrif_task(**task_changes),
            baseline_ref=baseline,
            validation_result=gate,
        )
        == ()
    )


@pytest.mark.parametrize(
    "defect", ["tracked", "untracked", "symlink", "parent_symlink", "altered"]
)
def test_vrif_owner_materialize_argv_rejects_dirty_or_untrusted_workspace(
    tmp_path: Path,
    defect: str,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    materializer = workspace / VRIF_MATERIALIZER
    if defect == "tracked":
        materializer.write_bytes(materializer.read_bytes() + b"\n# changed\n")
    elif defect == "untracked":
        (workspace / "untracked.txt").write_text("unexpected\n", encoding="utf-8")
    elif defect == "symlink":
        materializer.unlink()
        materializer.symlink_to("../untracked-materializer.py")
    elif defect == "parent_symlink":
        scripts = workspace / "scripts"
        external_scripts = tmp_path / "external-scripts"
        scripts.rename(external_scripts)
        scripts.symlink_to(external_scripts, target_is_directory=True)
    else:
        _git(workspace, "update-index", "--assume-unchanged", VRIF_MATERIALIZER)
        materializer.write_bytes(materializer.read_bytes() + b"\n# concealed drift\n")

    assert (
        PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
            workspace_path=workspace,
            task=_vrif_task(),
            baseline_ref=baseline,
            validation_result=_vrif_exact_empty_patch_result(baseline),
        )
        == ()
    )


@pytest.mark.parametrize(
    "config_key",
    [
        "filter.vrif.clean",
        "filter.vrif.process",
        "diff.vrif.command",
        "diff.vrif.textconv",
        "core.fsmonitor",
        "core.hooksPath",
    ],
)
def test_vrif_owner_authorization_rejects_dangerous_local_git_config_without_execution(
    tmp_path: Path,
    config_key: str,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    marker = tmp_path / "dangerous-git-config-executed"
    helper = tmp_path / "dangerous-git-helper"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed\\n', encoding='utf-8')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    config_value = str(helper)
    if config_key == "core.hooksPath":
        hooks = tmp_path / "dangerous-hooks"
        hooks.mkdir()
        for hook_name in ("post-checkout", "pre-commit"):
            hook = hooks / hook_name
            hook.write_bytes(helper.read_bytes())
            hook.chmod(0o755)
        config_value = str(hooks)
    _git(workspace, "config", "--local", config_key, config_value)

    argv = PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
        workspace_path=workspace,
        task=_vrif_task(),
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
    )

    assert argv == ()
    assert not marker.exists()


def _sha256_identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _vrif_materializer_receipt(
    *,
    baseline: str,
    tree: str,
    payloads: dict[str, bytes],
    changed_paths: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "vrif-frozen-benchmark-materialization@1"
        ),
        "mode": "write",
        "baseline_commit": baseline,
        "baseline_tree": tree,
        "changed_paths": list(changed_paths or VRIF_CHANGED_PATHS),
        "case_count": 96,
        "case_root": "sha256:" + "4" * 64,
        "binding_set_id": "sha256:" + "5" * 64,
        "freeze_id": "sha256:" + "6" * 64,
        "output_identities": {
            path: _sha256_identity(payload) for path, payload in payloads.items()
        },
    }


def _vrif_runner_daemon(monkeypatch: pytest.MonkeyPatch) -> PortalImplementationDaemon:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.implementation_timeout = 60
    monkeypatch.setattr(daemon, "_record_event", lambda *_args, **_kwargs: None)
    return daemon


def test_vrif_owner_materializer_runner_is_shell_free_isolated_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, baseline, tree = _vrif_clean_repository(tmp_path)
    task = _vrif_task()
    gate = _vrif_exact_empty_patch_result(baseline)
    argv = PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
        workspace_path=workspace,
        task=task,
        baseline_ref=baseline,
        validation_result=gate,
    )
    assert argv
    payloads = {path: f"owner materialized {path}\n".encode() for path in VRIF_OUTPUTS}
    receipt = _vrif_materializer_receipt(
        baseline=baseline,
        tree=tree,
        payloads=payloads,
    )
    captured: dict[str, object] = {}
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        command_tuple = tuple(str(item) for item in command)
        if (
            command_tuple[:4] == (sys.executable, "-I", "-S", "-B")
            and command_tuple[5:] == argv[5:]
            and command_tuple[4] != VRIF_MATERIALIZER
        ):
            captured.update(kwargs)
            for path, payload in payloads.items():
                (workspace / path).write_bytes(payload)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                stderr="",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(implementation_daemon.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/attacker-controlled-pythonpath")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker-controlled-venv")
    daemon = _vrif_runner_daemon(monkeypatch)

    result = daemon._run_vrif_benchmark_owner_materializer(
        workspace_path=workspace,
        task=task,
        attempt=1,
        baseline_ref=baseline,
        argv=argv,
        log_path=tmp_path / "implementation.log",
    )

    assert result["attempted"] is True
    assert result["passed"] is True
    assert result["returncode"] == 0
    assert set(result["changed_paths"]) == set(VRIF_OUTPUTS)
    assert captured["shell"] is False
    assert captured["cwd"] == workspace
    assert captured["text"] is True
    assert captured["capture_output"] is True
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert all(
        result["output_identities"][path] == _sha256_identity(payloads[path])
        for path in VRIF_OUTPUTS
    )


@pytest.mark.parametrize(
    "defect",
    [
        "nonzero",
        "extra_field",
        "duplicate_json",
        "extra_stdout",
        "wrong_tree",
        "partial_paths",
        "wrong_hash",
        "branch_switch",
        "trusted_code_mutation",
        "data_mutation",
        "index_flag_mutation",
    ],
)
def test_vrif_owner_materializer_runner_rejects_malformed_or_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    workspace, baseline, tree = _vrif_clean_repository(tmp_path)
    task = _vrif_task()
    argv = PortalImplementationDaemon._vrif_benchmark_owner_materialize_argv(
        workspace_path=workspace,
        task=task,
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
    )
    assert argv
    payloads = {path: f"owner materialized {path}\n".encode() for path in VRIF_OUTPUTS}
    receipt = _vrif_materializer_receipt(
        baseline=baseline,
        tree=tree,
        payloads=payloads,
    )
    if defect == "extra_field":
        receipt["unexpected"] = True
    elif defect == "wrong_tree":
        receipt["baseline_tree"] = "f" * 40
    elif defect == "partial_paths":
        receipt["changed_paths"] = list(VRIF_CHANGED_PATHS[:-1])
        output_identities = receipt["output_identities"]
        assert isinstance(output_identities, dict)
        output_identities.pop(VRIF_CHANGED_PATHS[-1])
    elif defect == "wrong_hash":
        output_identities = receipt["output_identities"]
        assert isinstance(output_identities, dict)
        output_identities[VRIF_OUTPUTS[0]] = "sha256:" + "f" * 64

    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        command_tuple = tuple(str(item) for item in command)
        if not (
            command_tuple[:4] == (sys.executable, "-I", "-S", "-B")
            and command_tuple[5:] == argv[5:]
            and command_tuple[4] != VRIF_MATERIALIZER
        ):
            return real_run(command, *args, **kwargs)
        if defect == "nonzero":
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="failed")
        written = VRIF_OUTPUTS[:-1] if defect == "partial_paths" else VRIF_OUTPUTS
        for path in written:
            (workspace / path).write_bytes(payloads[path])
        if defect == "branch_switch":
            switched = real_run(
                ["git", "switch", "-qc", "post-materialization-drift"],
                cwd=workspace,
                capture_output=True,
                check=False,
                text=True,
            )
            assert switched.returncode == 0, switched.stderr
        elif defect == "trusted_code_mutation":
            trusted_path = (
                workspace
                / implementation_daemon.VRIF_BENCHMARK_RECOVERY_TRUSTED_CODE_PATHS[
                    1
                ]
            )
            trusted_path.write_bytes(trusted_path.read_bytes() + b"\n# post-run drift\n")
        elif defect == "data_mutation":
            data_path = (
                workspace
                / implementation_daemon.VRIF_BENCHMARK_RECOVERY_DATA_PATHS[0]
            )
            data_path.write_bytes(data_path.read_bytes() + b"\npost-run drift\n")
        elif defect == "index_flag_mutation":
            flag_path = (
                implementation_daemon.VRIF_BENCHMARK_RECOVERY_TRUSTED_CODE_PATHS[1]
            )
            flagged = real_run(
                ["git", "update-index", "--assume-unchanged", "--", flag_path],
                cwd=workspace,
                capture_output=True,
                check=False,
                text=True,
            )
            assert flagged.returncode == 0, flagged.stderr
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        if defect == "duplicate_json":
            encoded = encoded.replace(
                '"mode":"write"',
                '"mode":"write","mode":"write"',
            )
        elif defect == "extra_stdout":
            encoded = "untrusted preamble\n" + encoded
        return subprocess.CompletedProcess(command, 0, stdout=encoded + "\n", stderr="")

    monkeypatch.setattr(implementation_daemon.subprocess, "run", fake_run)
    daemon = _vrif_runner_daemon(monkeypatch)

    result = daemon._run_vrif_benchmark_owner_materializer(
        workspace_path=workspace,
        task=task,
        attempt=1,
        baseline_ref=baseline,
        argv=argv,
        log_path=tmp_path / "implementation.log",
    )

    assert result["attempted"] is True
    assert result["passed"] is False


def _vrif_accepted_proposal(baseline: str) -> SimpleNamespace:
    return SimpleNamespace(
        accepted=True,
        findings=(),
        policy=SimpleNamespace(policy_id="4" * 64),
        proposal=SimpleNamespace(
            changed_paths=VRIF_OUTPUTS,
            proposal_id="5" * 64,
            repository_tree_id=baseline,
        ),
        receipt=SimpleNamespace(receipt_id="6" * 64),
    )


def _vrif_recovery_daemon(monkeypatch: pytest.MonkeyPatch) -> PortalImplementationDaemon:
    daemon = object.__new__(PortalImplementationDaemon)
    monkeypatch.setattr(daemon, "_record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        daemon,
        "_sanitize_failed_validation_result",
        lambda result: dict(result),
    )
    return daemon


def test_vrif_owner_recovery_stages_exact_outputs_and_runs_uncached_bound_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, baseline, tree = _vrif_clean_repository(tmp_path)
    branch = _git(workspace, "branch", "--show-current")
    task = _vrif_task()
    payloads = {path: f"recovered {path}\n".encode() for path in VRIF_OUTPUTS}
    receipt = _vrif_materializer_receipt(
        baseline=baseline,
        tree=tree,
        payloads=payloads,
    )
    materializer_calls: list[tuple[str, ...]] = []
    validation_calls: list[dict[str, object]] = []
    binding_calls: list[dict[str, object]] = []
    git_calls: list[tuple[str, ...]] = []
    proposal = _vrif_accepted_proposal(baseline)
    daemon = _vrif_recovery_daemon(monkeypatch)
    real_run = subprocess.run

    def materialize(**kwargs):
        materializer_calls.append(tuple(kwargs["argv"]))
        for path, payload in payloads.items():
            (workspace / path).write_bytes(payload)
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "reason": "vrif_benchmark_owner_materialized",
            "changed_paths": list(VRIF_OUTPUTS),
            "receipt": receipt,
            "output_identities": dict(receipt["output_identities"]),
        }

    def validate_commands(*args, **kwargs):
        validation_calls.append(dict(kwargs))
        assert args == (workspace, task, tmp_path / "implementation.log")
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [{"command": VRIF_VALIDATION, "passed": True}],
        }

    def verify_binding(*args, **kwargs):
        binding_calls.append(dict(kwargs))
        assert args == (workspace, task)
        return dict(kwargs["validation_result"])

    def record_run(command, *args, **kwargs):
        command_tuple = tuple(str(item) for item in command)
        if command_tuple and Path(command_tuple[0]).name == "git":
            git_calls.append(command_tuple)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_run_vrif_benchmark_owner_materializer",
        materialize,
    )
    monkeypatch.setattr(
        daemon,
        "_stage_declared_candidate_outputs",
        lambda *_args, **_kwargs: pytest.fail(
            "owner recovery must not use generic staging"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_staged_worktree_paths",
        lambda *_args, **_kwargs: pytest.fail(
            "owner recovery must not use the bare staged-path helper"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *_args, **_kwargs: proposal,
    )
    monkeypatch.setattr(daemon, "_run_validation_commands", validate_commands)
    monkeypatch.setattr(
        daemon,
        "_verify_post_validation_candidate_binding",
        verify_binding,
    )
    monkeypatch.setattr(
        daemon,
        "_automatic_implementation_rescue",
        lambda **_kwargs: pytest.fail("owner recovery must not fall through"),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail("owner recovery must not invoke a provider"),
    )
    monkeypatch.setattr(implementation_daemon.subprocess, "run", record_run)

    result = daemon._run_vrif_benchmark_owner_recovery_after_review(
        task=task,
        attempt=1,
        workspace_path=workspace,
        branch_name=branch,
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
        log_path=tmp_path / "implementation.log",
        state=None,
    )

    assert result is not None
    assert result["passed"] is True
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"] == {
        "succeeded": True,
        "owner_recovery": True,
        "provider_passes": 0,
        "materializer_attempted": True,
        "changed_paths": list(VRIF_OUTPUTS),
    }
    assert len(materializer_calls) == 1
    assert _git(workspace, "diff", "--cached", "--name-only").splitlines() == list(
        VRIF_OUTPUTS
    )
    hash_calls = [call for call in git_calls if "hash-object" in call]
    update_calls = [call for call in git_calls if "update-index" in call]
    assert len(hash_calls) == len(VRIF_OUTPUTS)
    assert all(
        call[0] == implementation_daemon.VRIF_BENCHMARK_RECOVERY_GIT
        for call in (*hash_calls, *update_calls)
    )
    assert all("-w" in call and "--no-filters" in call for call in hash_calls)
    assert all("--stdin" in call for call in hash_calls)
    assert len(update_calls) == len(VRIF_OUTPUTS)
    assert all("--add" in call and "--cacheinfo" in call for call in update_calls)
    assert all(
        any(path in " ".join(call) for call in update_calls) for path in VRIF_OUTPUTS
    )
    assert not any("add" in call for call in git_calls)
    assert len(validation_calls) == 1
    assert validation_calls[0]["force_uncached"] is True
    assert validation_calls[0]["proposal_validation"] is proposal
    assert len(binding_calls) == 1
    assert binding_calls[0]["baseline_ref"] == baseline
    assert binding_calls[0]["proposal_validation"] is proposal


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("materializer", "vrif_benchmark_owner_materialization_failed"),
        ("gate", "vrif_benchmark_owner_materializer_not_authorized"),
        ("validation", "vrif_benchmark_owner_contract_mismatch"),
        ("output", "vrif_benchmark_owner_contract_mismatch"),
    ],
)
def test_vrif_owner_recovery_failure_is_terminal_without_generic_or_provider_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    branch = _git(workspace, "branch", "--show-current")
    task = _vrif_task()
    gate = _vrif_exact_empty_patch_result(baseline)
    if mutation == "gate":
        proposal_gate = gate["proposal_gate"]
        assert isinstance(proposal_gate, dict)
        proposal_gate["reason_codes"] = ["empty_patch"]
    elif mutation == "validation":
        task = _vrif_task(validation=["python3 -m pytest -q unrelated.py"])
    elif mutation == "output":
        task = _vrif_task(outputs=list(VRIF_DECLARED_OUTPUTS[:-1]))

    daemon = _vrif_recovery_daemon(monkeypatch)
    monkeypatch.setattr(
        daemon,
        "_run_vrif_benchmark_owner_materializer",
        lambda **_kwargs: {
            "attempted": True,
            "passed": False,
            "returncode": 9,
            "reason": "synthetic_materializer_failure",
            "changed_paths": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_run_auto_rescue_materialize_commands",
        lambda **_kwargs: pytest.fail("generic materialization must be unreachable"),
    )
    monkeypatch.setattr(
        daemon,
        "_automatic_implementation_rescue",
        lambda **_kwargs: pytest.fail("generic auto-rescue must be unreachable"),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail("provider rescue must be unreachable"),
    )

    result = daemon._run_vrif_benchmark_owner_recovery_after_review(
        task=task,
        attempt=2,
        workspace_path=workspace,
        branch_name=branch,
        baseline_ref=baseline,
        validation_result=gate,
        log_path=tmp_path / "implementation.log",
        state=None,
    )

    assert result is not None
    assert result["passed"] is False
    assert result["reason"] == expected_reason
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["owner_recovery"] is True
    assert result["auto_rescue"]["provider_passes"] == 0


def test_vrif_owner_recovery_rejects_partial_staging_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, baseline, tree = _vrif_clean_repository(tmp_path)
    task = _vrif_task()
    payloads = {path: f"recovered {path}\n".encode() for path in VRIF_OUTPUTS}
    receipt = _vrif_materializer_receipt(
        baseline=baseline,
        tree=tree,
        payloads=payloads,
    )
    daemon = _vrif_recovery_daemon(monkeypatch)

    def materialize(**_kwargs):
        for path, payload in payloads.items():
            (workspace / path).write_bytes(payload)
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "reason": "vrif_benchmark_owner_materialized",
            "changed_paths": list(VRIF_OUTPUTS),
            "receipt": receipt,
            "output_identities": dict(receipt["output_identities"]),
        }

    def stage_partial(*_args, **_kwargs) -> tuple[str, ...]:
        _git(workspace, "add", "--", *VRIF_OUTPUTS[:-1])
        return VRIF_OUTPUTS[:-1]

    monkeypatch.setattr(
        daemon,
        "_run_vrif_benchmark_owner_materializer",
        materialize,
    )
    monkeypatch.setattr(
        daemon,
        "_stage_vrif_benchmark_owner_outputs",
        stage_partial,
    )
    monkeypatch.setattr(
        daemon,
        "_stage_declared_candidate_outputs",
        lambda *_args, **_kwargs: pytest.fail(
            "owner recovery must not use generic staging"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *_args, **_kwargs: pytest.fail(
            "partial staging must fail before proposal admission"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        lambda *_args, **_kwargs: pytest.fail(
            "partial staging must fail before validation"
        ),
    )

    result = daemon._run_vrif_benchmark_owner_recovery_after_review(
        task=task,
        attempt=1,
        workspace_path=workspace,
        branch_name=_git(workspace, "branch", "--show-current"),
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
        log_path=tmp_path / "implementation.log",
        state=None,
    )

    assert result is not None
    assert result["passed"] is False
    assert result["reason"] == "vrif_benchmark_owner_staging_failed"
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["provider_passes"] == 0


def _write_vrif_commit_candidate(
    workspace: Path,
    paths: tuple[str, ...] = VRIF_OUTPUTS,
) -> None:
    for relative_path in paths:
        (workspace / relative_path).write_text(
            f"sealed commit candidate for {relative_path}\n",
            encoding="utf-8",
        )


def test_commit_vrif_owner_changes_commits_only_exact_outputs_without_hooks(
    tmp_path: Path,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = object.__new__(PortalImplementationDaemon)
    marker = tmp_path / "pre-commit-hook-ran"
    pre_commit = workspace / ".git" / "hooks" / "pre-commit"
    pre_commit.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran\\n', encoding='utf-8')\n"
        "raise SystemExit(73)\n",
        encoding="utf-8",
    )
    pre_commit.chmod(0o755)
    _write_vrif_commit_candidate(workspace)
    assert daemon._stage_vrif_benchmark_owner_outputs(
        workspace,
        baseline_ref=baseline,
    ) == VRIF_OUTPUTS

    result = daemon._commit_vrif_benchmark_owner_changes(
        workspace,
        task=_vrif_task(),
        attempt=7,
        baseline_ref=baseline,
    )

    assert result["committed"] is True
    assert result["reason"] == "vrif_benchmark_owner_committed"
    commit = str(result["commit"])
    assert _git(workspace, "rev-list", "--parents", "-n", "1", commit).split() == [
        commit,
        baseline,
    ]
    changed_paths = _git(
        workspace,
        "diff",
        "--name-only",
        baseline,
        commit,
        "--",
    ).splitlines()
    assert changed_paths == list(VRIF_OUTPUTS)
    tree_entries = _git(
        workspace,
        "ls-tree",
        commit,
        "--",
        *VRIF_OUTPUTS,
    ).splitlines()
    assert len(tree_entries) == len(VRIF_OUTPUTS)
    assert all(entry.startswith("100644 blob ") for entry in tree_entries)
    assert {entry.split("\t", 1)[1] for entry in tree_entries} == set(VRIF_OUTPUTS)
    assert _git(workspace, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert not marker.exists()


def test_commit_vrif_owner_changes_rejects_malformed_task(
    tmp_path: Path,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = object.__new__(PortalImplementationDaemon)
    _write_vrif_commit_candidate(workspace)
    assert daemon._stage_vrif_benchmark_owner_outputs(
        workspace,
        baseline_ref=baseline,
    ) == VRIF_OUTPUTS

    result = daemon._commit_vrif_benchmark_owner_changes(
        workspace,
        task=_vrif_task(outputs=list(VRIF_OUTPUTS[:-1])),
        attempt=1,
        baseline_ref=baseline,
    )

    assert result == {
        "committed": False,
        "reason": "vrif_benchmark_owner_commit_not_authorized",
    }
    assert _git(workspace, "rev-parse", "HEAD") == baseline


def test_commit_vrif_owner_changes_rejects_partial_output_candidate(
    tmp_path: Path,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = object.__new__(PortalImplementationDaemon)
    _write_vrif_commit_candidate(workspace, VRIF_OUTPUTS[:-1])
    assert daemon._stage_vrif_benchmark_owner_outputs(
        workspace,
        baseline_ref=baseline,
    ) == VRIF_OUTPUTS

    result = daemon._commit_vrif_benchmark_owner_changes(
        workspace,
        task=_vrif_task(),
        attempt=1,
        baseline_ref=baseline,
    )

    assert result["committed"] is False
    assert result["reason"] == "vrif_benchmark_owner_commit_not_authorized"
    assert _git(workspace, "rev-parse", "HEAD") == baseline
    assert _git(workspace, "diff", "--cached", "--name-only").splitlines() == list(
        VRIF_OUTPUTS[:-1]
    )


def test_commit_vrif_owner_changes_rejects_residual_unstaged_path(
    tmp_path: Path,
) -> None:
    workspace, _baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = object.__new__(PortalImplementationDaemon)
    residual = workspace / "tracked-residual.txt"
    residual.write_text("baseline\n", encoding="utf-8")
    _git(workspace, "add", "--", residual.name)
    _git(workspace, "commit", "-qm", "add tracked residual fixture")
    baseline = _git(workspace, "rev-parse", "HEAD")
    _write_vrif_commit_candidate(workspace)
    assert daemon._stage_vrif_benchmark_owner_outputs(
        workspace,
        baseline_ref=baseline,
    ) == VRIF_OUTPUTS
    residual.write_text("unstaged drift\n", encoding="utf-8")

    result = daemon._commit_vrif_benchmark_owner_changes(
        workspace,
        task=_vrif_task(),
        attempt=1,
        baseline_ref=baseline,
    )

    assert result["committed"] is False
    assert result["reason"] == "vrif_benchmark_owner_commit_not_authorized"
    assert _git(workspace, "rev-parse", "HEAD") == baseline
    assert residual.name in _git(workspace, "diff", "--name-only").splitlines()


def test_commit_vrif_owner_changes_rejects_existing_executable_output_mode(
    tmp_path: Path,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = object.__new__(PortalImplementationDaemon)
    _write_vrif_commit_candidate(workspace)
    executable_output = workspace / VRIF_OUTPUTS[0]
    executable_output.chmod(0o755)
    _git(workspace, "add", "--", *VRIF_OUTPUTS)
    _git(workspace, "commit", "-qm", "candidate with executable output")
    candidate = _git(workspace, "rev-parse", "HEAD")
    assert _git(workspace, "rev-parse", "HEAD^1") == baseline
    assert _git(
        workspace,
        "diff",
        "--name-only",
        baseline,
        candidate,
        "--",
    ).splitlines() == list(VRIF_OUTPUTS)
    executable_entry = _git(
        workspace,
        "ls-tree",
        candidate,
        "--",
        VRIF_OUTPUTS[0],
    )
    assert executable_entry.startswith("100755 blob ")

    result = daemon._commit_vrif_benchmark_owner_changes(
        workspace,
        task=_vrif_task(),
        attempt=1,
        baseline_ref=baseline,
    )

    assert result["committed"] is False
    assert result["reason"] == "vrif_benchmark_owner_commit_not_authorized"
    assert _git(workspace, "rev-parse", "HEAD") == candidate
    assert _git(workspace, "status", "--porcelain=v1") == ""


def test_vrif_owner_recovery_ordinary_task_remains_outside_reserved_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, baseline, _tree = _vrif_clean_repository(tmp_path)
    daemon = _vrif_recovery_daemon(monkeypatch)

    result = daemon._run_vrif_benchmark_owner_recovery_after_review(
        task=_vrif_task(task_id="ORDINARY-030"),
        attempt=1,
        workspace_path=workspace,
        branch_name=_git(workspace, "branch", "--show-current"),
        baseline_ref=baseline,
        validation_result=_vrif_exact_empty_patch_result(baseline),
        log_path=tmp_path / "implementation.log",
        state=None,
    )

    assert result is None


def test_summarize_test_failure_prefers_assertion_over_banner() -> None:
    output = """
============================= test session starts ==============================
collected 1 item

test/api/test_foo.py F                                                   [100%]

=================================== FAILURES ===================================
_______________________________ test_provider_surfaces _________________________

    def test_provider_surfaces():
>       assert surfaces["count"] >= 1
E       AssertionError: assert 0 >= 1

test/api/test_foo.py:12: AssertionError
=========================== short test summary info ============================
FAILED test/api/test_foo.py::test_provider_surfaces - AssertionError: assert 0 >= 1
============================== 1 failed in 0.12s ===============================
"""
    summary = summarize_test_failure(output)
    assert "test/api/test_foo.py::test_provider_surfaces" in summary["failed_tests"]
    head = summary["failure_head"]
    assert "AssertionError" in head
    assert "short test summary info" not in head
    assert "assert 0 >= 1" in head


def test_summarize_test_failure_quiet_mode_still_extracts_failed_node() -> None:
    output = """
F                                                                        [100%]
=========================== short test summary info ============================
FAILED external/ipfs_accelerate/test/api/test_agent_supervisor_dcr_provider_surface_health.py::test_codec - AssertionError: missing providers
"""
    summary = summarize_test_failure(output)
    assert any("test_codec" in item for item in summary["failed_tests"])
    assert "AssertionError" in summary["failure_head"]
    assert "missing providers" in summary["failure_head"]


def test_plan_stage_and_revalidate_for_empty_patch_with_dirty_outputs() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"],
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["proposal_gate_failed", "empty_or_no_change"],
                "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"],
            },
        },
        expected_outputs=(
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/analysis/provider_surface_health.py",
            "data/agent_supervisor/deterministic_contract_repair/provider-surfaces.json",
        ),
        expected_outputs_present_on_disk=True,
        dirty_in_scope_paths=(
            "data/agent_supervisor/deterministic_contract_repair/provider-surfaces.json",
        ),
    )
    assert plan.action is AutoRescueAction.STAGE_AND_REVALIDATE
    assert plan.reason == "stage_declared_outputs_and_revalidate"


def test_plan_inline_provider_rescue_for_validation_command_failed() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": [
                "python3 -m pytest -q external/ipfs_accelerate/test/api/test_foo.py"
            ],
            "failure_head": "E   AssertionError: missing providers",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
                "failed_commands": [
                    "python3 -m pytest -q external/ipfs_accelerate/test/api/test_foo.py"
                ],
                "next_attempt_prompt_addendum": "Re-run and fix the pytest command.",
            },
        },
        expected_outputs=("external/ipfs_accelerate/test/api/test_foo.py",),
        expected_outputs_present_on_disk=True,
        allow_provider_rescue=True,
    )
    assert plan.action is AutoRescueAction.INLINE_PROVIDER_RESCUE
    assert "validation" in plan.reason


def test_plan_stage_after_proposal_accept_when_outputs_incomplete() -> None:
    """PTR-style: proposal accepted, residual review says outputs incomplete."""

    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["cargo test --locked --manifest-path ..."],
            "proposal_gate": {"accepted": True},
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "large_or_undeclared_refactor",
                ],
                "missing_expected_outputs": [
                    "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
                    "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/WIRE_FORMAT.md",
                ],
                "failed_commands": ["cargo test --locked --manifest-path ..."],
            },
        },
        expected_outputs=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/WIRE_FORMAT.md",
        ),
        expected_outputs_present_on_disk=True,
        dirty_in_scope_paths=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
        ),
        allow_provider_rescue=True,
    )
    assert plan.action is AutoRescueAction.STAGE_AND_REVALIDATE
    assert plan.reason == "stage_declared_outputs_and_revalidate"

    after_stage = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["cargo test --locked --manifest-path ..."],
            "proposal_gate": {"accepted": True},
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["incomplete_expected_outputs"],
                "failed_commands": ["cargo test --locked --manifest-path ..."],
            },
        },
        expected_outputs=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
        ),
        expected_outputs_present_on_disk=True,
        stage_rescue_used=True,
        allow_provider_rescue=True,
    )
    assert after_stage.action is AutoRescueAction.INLINE_PROVIDER_RESCUE


def test_derive_materialize_commands_from_validate_cli() -> None:
    commands = derive_materialize_commands(
        (
            "PYTHONPATH=external/ipfs_accelerate python3 -m "
            "external.ipfs_accelerate.ipfs_accelerate_py.agent_supervisor.analysis."
            "deterministic_desktop_expectations validate --workspace . "
            "--artifact data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        )
    )
    assert commands
    assert any(" materialize " in command for command in commands)
    assert all(" validate " not in command for command in commands)


def test_plan_materialize_when_expected_artifact_missing() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "proposal_gate_failed",
                ],
                "finding_codes": ["expected_output_ignored_or_unstaged"],
                "missing_expected_outputs": [
                    "data/agent_supervisor/deterministic_contract_repair/"
                    "desktop-expectations.json"
                ],
            },
        },
        expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "analysis/deterministic_desktop_expectations.py",
        ),
        validation_commands=(
            "python3 -m pkg.mod validate --workspace . --artifact "
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        missing_expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        expected_outputs_present_on_disk=False,
    )
    assert plan.action is AutoRescueAction.MATERIALIZE_AND_STAGE
    assert plan.materialize_commands
    assert "desktop-expectations.json" in " ".join(plan.missing_expected_outputs)


def test_plan_provider_rescue_after_stage_for_residual_incomplete() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "proposal_gate_failed",
                ],
                "finding_codes": ["expected_output_ignored_or_unstaged"],
                "missing_expected_outputs": [
                    "data/agent_supervisor/deterministic_contract_repair/"
                    "desktop-expectations.json"
                ],
            },
        },
        expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        stage_rescue_used=True,
        materialize_rescue_used=True,
        allow_provider_rescue=True,
        expected_outputs_present_on_disk=False,
        missing_expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
    )
    assert plan.action is AutoRescueAction.INLINE_PROVIDER_RESCUE
    assert "residual" in plan.reason or "incomplete" in plan.reason


def test_is_undeclared_helper_path_recognizes_scratch_files() -> None:
    expected = ("swissknife/src/services/gui-optimizer/cli.ts",)
    assert is_undeclared_helper_path("tmp-vgo-062-write-evidence.mts", expected)
    assert is_undeclared_helper_path("swissknife/_run_registry.py", expected)
    assert is_undeclared_helper_path("DELETE_ME_helper.py", expected)
    assert is_undeclared_helper_path("scripts/vgo060-selfcheck.py", expected)
    assert not is_undeclared_helper_path(
        "swissknife/src/services/gui-optimizer/cli.ts",
        expected,
    )
    assert not is_undeclared_helper_path(
        "swissknife/src/services/gui-optimizer/targets/agent-supervisor.ts",
        expected,
    )


def test_plan_strips_helper_only_scope_denials_when_outputs_exist() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "scope_adjudication_failed",
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "finding_codes": ["path_outside_scope"],
                "denied_paths": [
                    "tmp-vgo-062-write-evidence.mts",
                    "swissknife/_run_selfcheck.py",
                ],
            },
        },
        expected_outputs=(
            "swissknife/src/services/gui-optimizer/targets/agent-supervisor.ts",
            "swissknife/test/unit/services/gui-optimizer/agent-supervisor-baseline.test.ts",
        ),
        expected_outputs_present_on_disk=True,
    )
    assert plan.action is AutoRescueAction.STRIP_DENIED_HELPERS
    assert plan.reason == "strip_undeclared_helper_paths"
    assert plan.denied_helper_paths == (
        "swissknife/_run_selfcheck.py",
        "tmp-vgo-062-write-evidence.mts",
    )

    mixed = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "finding_codes": ["path_outside_scope"],
                "denied_paths": [
                    "tmp-vgo-060-helper.py",
                    "swissknife/src/unrelated/secret.ts",
                ],
            },
        },
        expected_outputs=("scripts/gui-opt",),
        expected_outputs_present_on_disk=True,
    )
    assert mixed.action is AutoRescueAction.NONE
    assert mixed.reason in {"hard_deny_or_reject", "hard_deny_reason_codes"}

    after_strip = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "denied_paths": ["tmp-vgo-062-write-evidence.mts"],
            },
        },
        expected_outputs=("scripts/gui-opt",),
        expected_outputs_present_on_disk=True,
        strip_helpers_used=True,
    )
    assert after_strip.action is AutoRescueAction.NONE


def test_plan_refuses_hard_deny_and_exhausted_budget() -> None:
    hard = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["hard_deny_findings"],
                "finding_codes": ["secret_change_forbidden"],
            },
        },
        expected_outputs_present_on_disk=True,
    )
    assert hard.action is AutoRescueAction.NONE

    exhausted = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "validation_command_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
            },
        },
        expected_outputs_present_on_disk=True,
        stage_rescue_used=True,
        materialize_rescue_used=True,
        provider_rescue_passes_used=1,
        already_auto_rescued=True,
    )
    assert exhausted.action is AutoRescueAction.NONE


def test_inline_provider_rescue_prompt_includes_failure_evidence() -> None:
    prompt = build_inline_provider_rescue_prompt(
        base_prompt="Implement DCR-013 outputs.",
        validation_result={
            "next_attempt_prompt_addendum": "Prior attempt failure review (guide_rescue).",
            "failed_commands": ["python3 -m pytest -q test_foo.py"],
            "failed_tests": ["test_foo.py::test_codec"],
            "failure_head": "E   AssertionError: missing providers",
            "failure_review": {
                "decision": "guide_rescue",
                "failed_commands": ["python3 -m pytest -q test_foo.py"],
            },
        },
    )
    assert "Automatic same-attempt validation rescue" in prompt
    assert "Prior attempt failure review" in prompt
    assert "test_foo.py::test_codec" in prompt
    assert "AssertionError: missing providers" in prompt
    assert prompt.startswith("Implement DCR-013 outputs.")


def _inline_rescue_test_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PortalImplementationDaemon, list[tuple[str, dict[str, object]]]]:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.implementation_timeout = 60
    daemon.implementation_max_timeout = 60
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        daemon,
        "_expected_outputs_present_on_disk",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        daemon,
        "_dirty_in_scope_declared_output_paths",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda name, payload: events.append((name, dict(payload))),
    )
    monkeypatch.setattr(
        daemon,
        "_ensure_implementation_checkpoint_dir",
        lambda _task: tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_process_environment",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_stage_declared_candidate_outputs",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        daemon,
        "_apply_implementation_failure_review",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )
    return daemon, events


def _run_inline_rescue(
    daemon: PortalImplementationDaemon,
    tmp_path: Path,
    command: list[str],
) -> dict[str, object]:
    task = PortalTask(
        task_id="RESCUE-001",
        title="repair validation",
        status="in_progress",
        completion="validation passes",
        priority="high",
        track="test",
        outputs=["result.txt"],
    )
    return daemon._automatic_implementation_rescue(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        branch_name="agent/rescue-001",
        baseline_ref="a" * 40,
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["python3 -m pytest -q test_result.py"],
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
            },
        },
        log_path=tmp_path / "implementation.log",
        state=None,
        command=command,
        base_prompt="repair the implementation",
    )


def test_inline_provider_rescue_refuses_prompt_bound_control_plane_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    accepted_path = "/proc/self/fd/71"
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=71,
        executable_path=accepted_path,
    )
    daemon._scoped_recovery_control_plane_launches = {
        "unrelated": SimpleNamespace(
            descriptor=72,
            executable_path="/proc/self/fd/72",
        )
    }
    command = [sys.executable, "-I", accepted_path, "--workspace", str(tmp_path)]
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail(
            "prompt-bound control-plane command must not run inline rescue"
        ),
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["provider_passes"] == 0


def test_inline_provider_rescue_fails_closed_on_ambiguous_sealed_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    accepted_path = "/proc/self/fd/71"
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=71,
        executable_path=accepted_path,
    )
    daemon._scoped_recovery_control_plane_launches = {
        "ambiguous": SimpleNamespace(
            descriptor=72,
            executable_path=accepted_path,
        )
    }
    command = [sys.executable, "-I", accepted_path, "--workspace", str(tmp_path)]
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail(
            "ambiguous control-plane authority must not launch a provider"
        ),
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert not any(name.endswith("provider_started") for name, _payload in events)


def test_inline_provider_rescue_keeps_unsealed_command_without_pass_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    command = ["/opt/providers/grok", "--model", "grok-4.6"]
    calls: list[dict[str, object]] = []

    def fake_stream(run_command, **kwargs):
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess(run_command, 0)

    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        fake_stream,
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is True
    assert len(calls) == 1
    assert "pass_fds" not in calls[0]
