"""Acceptance tests for the pytest verification adapter (IVP-005)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_forest import (
    AuthorityMode,
    LocalLocator,
    PortableGitClosure,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    RepositoryIdentity,
)
from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    CapabilitySnapshot,
    LockIdentity,
    ToolIdentity,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    PhaseOutcome,
    TestExecutionKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.verification.adapters.pytest_adapter import (
    PYTEST_ADAPTER_EVIDENCE,
    PYTEST_PHASE_REPORT_SCHEMA,
    PYTEST_VERIFICATION_ADAPTER_INTERFACE,
    PYTEST_VERIFICATION_ADAPTER_SCHEMA,
    PytestAdvisoryPolicy,
    PytestNodePhaseAccounting,
    PytestRunMode,
    PytestVerificationAdapter,
    PytestVerificationAdapterError,
    PytestVerificationRequest,
    build_pytest_argv,
    create_pytest_verification_adapter,
    encode_phase_report,
    parse_phase_report,
    project_terminal_status,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    PROOF_OBLIGATION_NOT_APPLICABLE_CID,
    TerminalStatus,
    TestReceipt,
    VerificationIdentityCompiler,
    VerificationReceiptKind,
    VerificationReceiptKey,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessRunner,
    VerificationRunDisposition,
    VerificationRunResult,
    VerificationSandboxIdentity,
    VerificationStreamArtifact,
    build_closed_sandbox,
    build_hermetic_environment,
)


# ---------------------------------------------------------------------------
# Identity / receipt-key helpers (compact recipe over the contracts compiler)
# ---------------------------------------------------------------------------

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)
SELECTOR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/verification-selector-argv@1"


def _structured_cid(schema: str, value: object) -> str:
    return content_identity({"schema": schema, "value": value})


def _artifact(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


def _repository_forest() -> RepositoryForest:
    alias = "ipfs_accelerate_py"
    descriptor = RepositoryDescriptor(
        identity=RepositoryIdentity(logical_name=alias),
        portable_closure=PortableGitClosure(
            commit="abcdef0123456789abcdef0123456789abcdef01",
            tree="0123456789abcdef0123456789abcdef01234567",
        ),
        local_locator=LocalLocator(
            alias=alias,
            root_path="/fixture/ipfs_accelerate_py",
            resolved_root_path="/fixture/ipfs_accelerate_py",
            local_repository_binding_id="fixture-binding:ipfs-accelerate",
        ),
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    return RepositoryForest(
        descriptors=(descriptor,),
        sole_write_alias=alias,
        policy_cid=_artifact("repository-forest-policy"),
    )


def _host() -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=16,
        available_worker_capacity=16,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=("cpu-validation", "cpu-small"),
    )


def _sandbox(tmp_path: Path) -> VerificationSandboxIdentity:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return build_closed_sandbox(source_root=source, artifact_root=artifacts)


def _compile_test_key(
    *,
    selector_argv: Sequence[str] | tuple[str, ...],
    python_executable: str,
    tool_version: str = "9.1.1",
    fixture_bytes: bytes = b"fixture-payload-v1\n",
    configuration_bytes: bytes = b"[pytest]\naddopts = -q\n",
) -> VerificationReceiptKey:
    repository_forest = _repository_forest()
    descriptor = repository_forest.write_descriptor()
    capability_name = "verification-tool"
    executable_bytes = b"reviewed-launcher:pytest"
    executable_sha256 = "sha256:" + hashlib.sha256(executable_bytes).hexdigest()
    dependency_lock_bytes = b"package==1.2.3 --hash=sha256:abcd\n"
    dependency_lock_path = "requirements.lock"
    dependency_lock_identity = LockIdentity(
        path=dependency_lock_path,
        identity="sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest(),
    )
    capability_snapshot = CapabilitySnapshot(
        tool_identities={capability_name: executable_sha256},
        lock_identities={dependency_lock_path: dependency_lock_identity.identity},
        environment_names=("LANG", "LC_ALL"),
        read_paths=("/fixture/ipfs_accelerate_py",),
        write_paths=("/fixture/artifacts",),
    )
    tool_identity = ToolIdentity(
        name=capability_name,
        kind="executable",
        locator=python_executable.rsplit("/", 1)[-1],
        version="launcher-fixture-1",
        identity=executable_sha256,
        roles=("verification",),
    )
    observed_environment = {
        "sandbox_schema": "hermetic-sandbox@1",
        "sandbox_policy": {
            "schema": "hermetic-sandbox-policy@1",
            "network": "deny",
            "auto_install": "deny",
            "home_cache": "deny",
            "auth_material": "deny",
        },
        "filesystem_policy": {
            "schema": "verification-filesystem-policy@1",
            "source": "read_only",
            "artifacts": "private_writable",
        },
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "x86_64",
            "libc": "glibc-2.39",
        },
        "interpreter": {
            "schema": "verification-interpreter@1",
            "implementation": "cpython",
            "version": "3.12.3",
            "abi": "cp312",
        },
        "toolchain": {
            "schema": "verification-toolchain@1",
            "name": "locked-python",
            "revision": "fixture-1",
        },
        "dependency_distribution": {
            "schema": "verification-dependency-distribution@1",
            "entries": (f"pytest=={tool_version}",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }
    version_probe_argv = (python_executable, "-m", "pytest", "--version")
    tool_version_probe_output = f"pytest {tool_version}\n".encode()
    expected_environment = {
        **observed_environment,
        "network_policy": NETWORK_POLICY_DENY_ALL,
        "tool_name": "pytest",
        "tool_version": tool_version,
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": python_executable,
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            TOOL_EXECUTABLE_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
        ),
        "tool_version_probe_argv": version_probe_argv,
        "tool_version_probe_output_cid": cid_for_bytes(tool_version_probe_output),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": PYTEST_VERIFICATION_ADAPTER_SCHEMA,
        "capability_environment_names": tuple(sorted(capability_snapshot.environment_names)),
        "capability_read_paths": tuple(sorted(capability_snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(capability_snapshot.write_paths)),
        "capability_lock_identities": dict(
            sorted(capability_snapshot.lock_identities.items())
        ),
        "selected_dependency_lock_path": dependency_lock_path,
        "selected_dependency_lock_identity": dependency_lock_identity.to_dict(),
    }
    claimed_environment_cid = _structured_cid(ENVIRONMENT_SCHEMA, expected_environment)
    semantic = {"symbols": ["example.calculate@2"], "edge_root": "sha256:semantic-edges"}
    semantic_cid = _structured_cid(
        "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1",
        semantic,
    )
    tree_observation = {
        "repository_forest_cid": repository_forest.forest_id,
        "git_commit_id": descriptor.commit,
        "git_tree_id": descriptor.tree,
        "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
        "dirty_overlay_cid": descriptor.dirty_overlay_digest,
        "dirty": descriptor.dirty,
        "repository_alias": descriptor.alias,
        "repository_id": descriptor.repository_id,
        "descriptor_cid": descriptor.descriptor_cid,
        "base_repository_tree_id": "git-tree:base",
    }
    claimed_tree_cid = _structured_cid(TREE_SCHEMA, tree_observation)
    return VerificationIdentityCompiler().compile_key(
        repository_forest=repository_forest,
        repository_alias="ipfs_accelerate_py",
        claimed_repository_tree_cid=claimed_tree_cid,
        patch_base_tree_id="git-tree:base",
        repository_state_tree_id="git-tree:base",
        invalidation_plan_tree_id="git-tree:base",
        context_pack_tree_id="git-tree:base",
        observed_semantic_state=semantic,
        repository_state_semantic_root_cid=semantic_cid,
        invalidation_plan_semantic_root_cid=semantic_cid,
        context_pack_semantic_root_cid=semantic_cid,
        affected_symbol_versions=(
            {
                "symbol": "example.calculate",
                "version": 2,
                "source_cid": _artifact("source-v2"),
            },
        ),
        observed_environment=observed_environment,
        capability_snapshot=capability_snapshot,
        tool_capability_name=capability_name,
        tool_identity=tool_identity,
        resolved_tool_executable=python_executable,
        tool_executable_bytes=executable_bytes,
        tool_version_probe_argv=version_probe_argv,
        tool_version_probe_output_bytes=tool_version_probe_output,
        claimed_environment_cid=claimed_environment_cid,
        dependency_lock_path=dependency_lock_path,
        dependency_lock_identity=dependency_lock_identity,
        dependency_lock_bytes=dependency_lock_bytes,
        selector_argv=tuple(selector_argv),
        proof_obligation=None,
        tool_name="pytest",
        tool_version=tool_version,
        configuration_bytes=configuration_bytes,
        fixture_data_bytes=(fixture_bytes,),
        network_policy=NETWORK_POLICY_DENY_ALL,
        receipt_schema_version=1,
        receipt_kind=VerificationReceiptKind.TEST,
        adapter_schema=PYTEST_VERIFICATION_ADAPTER_SCHEMA,
        proof_backend_binding=None,
    )


def _stream(data: bytes = b"") -> VerificationStreamArtifact:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return VerificationStreamArtifact(
        digest=digest,
        cid=cid_for_bytes(data),
        truncated=False,
        byte_count=len(data),
        captured_byte_count=len(data),
        preview=data.decode("utf-8", errors="replace")[:4096],
    )


def _run_result(
    *,
    argv: Sequence[str],
    cwd: str,
    environment: Mapping[str, str],
    sandbox: VerificationSandboxIdentity,
    terminal_status: TerminalStatus = TerminalStatus.PASSED,
    disposition: VerificationRunDisposition = VerificationRunDisposition.COMPLETED,
    exit_code: int | None = 0,
    timed_out: bool = False,
    cancelled: bool = False,
    unavailable: bool = False,
    process_started: bool = True,
    publication_allowed: bool = True,
    stdout: bytes = b"",
    stderr: bytes = b"",
    reason_codes: tuple[str, ...] = (),
    duration_ms: int = 42,
) -> VerificationRunResult:
    return VerificationRunResult(
        terminal_status=terminal_status,
        disposition=disposition,
        exit_code=exit_code,
        duration_ms=duration_ms,
        command_argv=tuple(argv),
        executable=str(argv[0]),
        cwd=cwd,
        environment=dict(environment),
        sandbox=sandbox.to_dict(),
        network_policy=NETWORK_POLICY_DENY_ALL,
        timeout_seconds=30.0,
        stdout=_stream(stdout),
        stderr=_stream(stderr),
        process_started=process_started,
        publication_allowed=publication_allowed,
        timed_out=timed_out,
        cancelled=cancelled,
        unavailable=unavailable,
        reason_codes=reason_codes,
        reason=reason_codes[0] if reason_codes else "",
    )


class _FakeRunner:
    """Injectable process runner seam for adapter unit tests."""

    def __init__(self, result: VerificationRunResult | None = None) -> None:
        self.result = result
        self.calls: list[VerificationCommand] = []
        self.cancellation: VerificationCancellation | None = None

    def run(
        self,
        command: VerificationCommand,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> VerificationRunResult:
        self.calls.append(command)
        self.cancellation = cancellation
        assert self.result is not None
        return self.result


NODE_A = "test/api/test_example.py::test_alpha"
NODE_B = "test/api/test_example.py::test_beta"
NODE_SKIP = "test/api/test_example.py::test_skip_me"


def _passing_items(*nodeids: str) -> list[dict[str, Any]]:
    return [
        {
            "nodeid": nodeid,
            "setup": "pass",
            "call": "pass",
            "teardown": "pass",
            "markers": [],
            "wasxfail": False,
            "duration_ms": 3,
        }
        for nodeid in nodeids
    ]


def _request(
    tmp_path: Path,
    *,
    mode: PytestRunMode = PytestRunMode.SELECTED_NODES,
    node_ids: Sequence[str] = (NODE_A,),
    suite_paths: Sequence[str] = (),
    config_args: Sequence[str] = (),
    extra_pytest_args: Sequence[str] = (),
    advisory: PytestAdvisoryPolicy | None = None,
    simulated: bool = False,
    injected_phase_report: Any = None,
    existing_bridge: bool = False,
    python_executable: str | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[PytestVerificationRequest, tuple[str, ...], VerificationSandboxIdentity]:
    python = python_executable or "/usr/bin/python3.12"
    box = _sandbox(tmp_path)
    argv = build_pytest_argv(
        python_executable=python,
        mode=mode,
        node_ids=node_ids,
        suite_paths=suite_paths,
        config_args=config_args,
        extra_pytest_args=extra_pytest_args,
    )
    key = _compile_test_key(selector_argv=argv, python_executable=python)
    bridge_receipt = None
    bridge_key = None
    if existing_bridge:
        observed_tree = key.repository_tree_observation
        bridge_key = TestExecutionKey(
            locator_cid=_artifact("test-locator"),
            repository_forest_cid=observed_tree["repository_forest_cid"],
            git_commit_id=observed_tree["git_commit_id"],
            git_tree_id=observed_tree["git_tree_id"],
            gitlink_state_cid=observed_tree["gitlink_state_cid"],
            dirty_overlay_cid=_artifact("pytest-dirty-overlay"),
            fixture_cids=(_artifact("pytest-fixture-closure"),),
            static_trace_root_cid=_artifact("pytest-static-trace"),
            runtime_trace_root_cid=_artifact("pytest-runtime-trace"),
            pytest_version=key.tool_version,
            command_semantics_cid=_artifact("pytest-command-semantics"),
            config_cid=_artifact("pytest-config"),
            dependency_lock_cid=_artifact("pytest-lock-closure"),
            environment_cid=_artifact("pytest-allowlisted-environment"),
            policy_cid=_artifact("pytest-policy"),
            components={"repository_descriptor": observed_tree["descriptor_cid"]},
        )
        bridge_receipt = TestPassReceipt(
            execution_key_cid=bridge_key.execution_key_id,
            locator_cid=bridge_key.locator_cid,
            setup_outcome=PhaseOutcome.PASS,
            call_outcome=PhaseOutcome.PASS,
            teardown_outcome=PhaseOutcome.PASS,
            static_trace_root_cid=bridge_key.static_trace_root_cid,
            runtime_trace_root_cid=bridge_key.runtime_trace_root_cid,
            completeness_receipt_cid=_artifact("pytest-runtime-completeness"),
            dependency_forest_cid=bridge_key.repository_forest_cid,
            policy_cid=bridge_key.policy_cid,
            admitted=True,
        )
    request = PytestVerificationRequest(
        receipt_key=key,
        mode=mode,
        python_executable=python,
        sandbox=box,
        cwd=str(box.source_root),
        timeout_seconds=timeout_seconds,
        node_ids=node_ids,
        suite_paths=suite_paths,
        config_args=config_args,
        extra_pytest_args=extra_pytest_args,
        environment=build_hermetic_environment(
            path=os.environ.get("PATH", "/usr/bin:/bin")
        ),
        advisory=advisory or PytestAdvisoryPolicy(),
        existing_test_pass_receipt=bridge_receipt,
        existing_test_execution_key=bridge_key,
        simulated=simulated,
        injected_phase_report=injected_phase_report,
        lane_id=f"pytest-test:{tmp_path.name}",
    )
    return request, argv, box


# ---------------------------------------------------------------------------
# Argv / interface
# ---------------------------------------------------------------------------


def test_explicit_python_m_pytest_argv_and_reproducible_list(tmp_path: Path) -> None:
    request, argv, _box = _request(
        tmp_path,
        node_ids=(NODE_A, NODE_B),
        config_args=("-c", "pytest.ini"),
    )
    adapter = create_pytest_verification_adapter(_FakeRunner())
    built = adapter.build_argv(request)
    assert built == argv
    assert built[0] == request.python_executable
    assert built[1:3] == ("-m", "pytest")
    assert "-c" in built and "pytest.ini" in built
    assert NODE_A in built and NODE_B in built
    assert "shell" not in " ".join(built).lower()
    # Reproducible: identical requests yield identical argv lists.
    again = adapter.build_argv(request)
    assert again == built
    assert list(again) == list(built)


def test_full_suite_oracle_mode_argv(tmp_path: Path) -> None:
    request, argv, _ = _request(
        tmp_path,
        mode=PytestRunMode.FULL_SUITE_ORACLE,
        node_ids=(),
        suite_paths=("test/api",),
    )
    assert request.mode is PytestRunMode.FULL_SUITE_ORACLE
    assert argv[1:3] == ("-m", "pytest")
    assert "test/api" in argv
    assert NODE_A not in argv


def test_selected_nodes_requires_node_ids() -> None:
    with pytest.raises(PytestVerificationAdapterError) as excinfo:
        build_pytest_argv(
            python_executable="/usr/bin/python3",
            mode=PytestRunMode.SELECTED_NODES,
            node_ids=(),
        )
    assert excinfo.value.reason_code == "empty_selector"


def test_adapter_interface_and_evidence_constants() -> None:
    adapter = create_pytest_verification_adapter()
    assert adapter.interface == PYTEST_VERIFICATION_ADAPTER_INTERFACE
    assert adapter.schema == PYTEST_VERIFICATION_ADAPTER_SCHEMA
    assert adapter.evidence == PYTEST_ADAPTER_EVIDENCE
    assert PYTEST_ADAPTER_EVIDENCE == "ivp/pytest-adapter@1"


# ---------------------------------------------------------------------------
# Bindings: selector / config / environment / fixtures
# ---------------------------------------------------------------------------


def test_selector_config_environment_fixture_bindings(tmp_path: Path) -> None:
    report = encode_phase_report(items=_passing_items(NODE_A), collected=1)
    request, argv, _ = _request(
        tmp_path,
        config_args=("-o", "xfail_strict=true"),
        injected_phase_report=report,
    )
    key = request.receipt_key
    assert key.tool_name == "pytest"
    assert key.adapter_schema == PYTEST_VERIFICATION_ADAPTER_SCHEMA
    assert key.configuration_cid  # configuration bytes bound
    assert key.fixture_data_cids  # fixture bytes bound
    assert key.environment_cid
    assert key.selector_cid == _structured_cid(SELECTOR_SCHEMA, {"argv": list(argv)})
    assert key.environment_observation["tool_name"] == "pytest"
    assert key.environment_observation["network_policy"] == NETWORK_POLICY_DENY_ALL

    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.receipt is not None
    assert result.receipt.execution.command_argv == argv
    assert result.receipt.key.selector_cid == key.selector_cid
    assert result.receipt.key.fixture_data_cids == key.fixture_data_cids
    assert result.receipt.key.configuration_cid == key.configuration_cid
    assert result.receipt.key.environment_cid == key.environment_cid


def test_selector_mismatch_is_rejected(tmp_path: Path) -> None:
    request, _argv, box = _request(tmp_path, node_ids=(NODE_A,))
    # Corrupt the request by swapping in a key compiled for a different selector.
    other_argv = build_pytest_argv(
        python_executable=request.python_executable,
        mode=PytestRunMode.SELECTED_NODES,
        node_ids=(NODE_B,),
    )
    other_key = _compile_test_key(
        selector_argv=other_argv,
        python_executable=request.python_executable,
    )
    bad = PytestVerificationRequest(
        receipt_key=other_key,
        mode=PytestRunMode.SELECTED_NODES,
        python_executable=request.python_executable,
        sandbox=box,
        cwd=str(box.source_root),
        timeout_seconds=30.0,
        node_ids=(NODE_A,),
        environment=dict(request.environment),
        injected_phase_report=encode_phase_report(items=_passing_items(NODE_A)),
    )
    with pytest.raises(PytestVerificationAdapterError) as excinfo:
        PytestVerificationAdapter(_FakeRunner()).execute(bad)
    assert excinfo.value.reason_code == "selector_binding_mismatch"


# ---------------------------------------------------------------------------
# Timeout / cancel / unavailable preservation
# ---------------------------------------------------------------------------


def test_timeout_is_preserved_and_cannot_pass(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            terminal_status=TerminalStatus.TIMEOUT,
            disposition=VerificationRunDisposition.TIMEOUT,
            exit_code=None,
            timed_out=True,
            publication_allowed=False,
            reason_codes=("timeout",),
        )
    )
    result = PytestVerificationAdapter(fake).execute(request)
    assert fake.calls and fake.calls[0].argv[1:3] == ("-m", "pytest")
    assert result.terminal_status is TerminalStatus.TIMEOUT
    assert not result.production_admissible
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.TIMEOUT
    assert "timeout" in result.reason_codes


def test_cancellation_is_preserved(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    cancel = VerificationCancellation(cancellation_id="cancel:pytest-test")
    cancel.cancel(reason="operator-abort")
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            terminal_status=TerminalStatus.CANCELLED,
            disposition=VerificationRunDisposition.CANCELLED,
            exit_code=None,
            cancelled=True,
            process_started=False,
            publication_allowed=False,
            reason_codes=("cancelled_before_spawn",),
        )
    )
    result = PytestVerificationAdapter(fake).execute(request, cancellation=cancel)
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert not result.production_admissible
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.CANCELLED


def test_unavailable_executable_is_preserved(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            terminal_status=TerminalStatus.UNAVAILABLE,
            disposition=VerificationRunDisposition.UNAVAILABLE,
            exit_code=None,
            unavailable=True,
            process_started=False,
            publication_allowed=False,
            reason_codes=("executable_missing",),
        )
    )
    result = PytestVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert not result.production_admissible
    assert "unavailable" in result.reason_codes or "executable_missing" in result.reason_codes


# ---------------------------------------------------------------------------
# Setup / call / teardown accounting + status projection
# ---------------------------------------------------------------------------


def test_setup_call_teardown_pass_projects_passed(tmp_path: Path) -> None:
    report = encode_phase_report(items=_passing_items(NODE_A, NODE_B), collected=2)
    request, argv, _ = _request(
        tmp_path,
        node_ids=(NODE_A, NODE_B),
        injected_phase_report=report,
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.production_admissible
    assert result.ok
    assert result.collected_count == 2
    assert len(result.phase_accounting) == 2
    for item in result.phase_accounting:
        assert item.all_phases_pass
        assert item.setup is PhaseOutcome.PASS
        assert item.call is PhaseOutcome.PASS
        assert item.teardown is PhaseOutcome.PASS
    assert result.command_argv == argv
    assert result.phase_report_cid
    assert result.artifact_cids
    assert result.receipt is not None
    assert result.receipt.terminal_success


def test_setup_failure_cannot_pass(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_A,
            "setup": "error",
            "call": "not_run",
            "teardown": "not_run",
            "markers": [],
        }
    ]
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert not result.production_admissible
    assert any(code.startswith("setup_failure:") for code in result.reason_codes)


def test_teardown_failure_cannot_pass(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_A,
            "setup": "pass",
            "call": "pass",
            "teardown": "error",
            "markers": [],
        }
    ]
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert not result.production_admissible
    assert any(code.startswith("teardown_failure:") for code in result.reason_codes)


def test_unexpected_xpass_cannot_pass(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_A,
            "setup": "pass",
            "call": "xpass",
            "teardown": "pass",
            "markers": ["xfail"],
            "wasxfail": True,
        }
    ]
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert not result.production_admissible
    assert any("xpass" in code for code in result.reason_codes)


def test_collection_failure_cannot_pass(tmp_path: Path) -> None:
    report = encode_phase_report(
        items=[],
        collected=0,
        collection_errors=("ImportError while collecting test_example.py",),
    )
    request, _, _ = _request(tmp_path, injected_phase_report=report)
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert "collection_failure" in result.reason_codes
    assert not result.production_admissible


# ---------------------------------------------------------------------------
# Empty collection / usage / malformed -> invalid
# ---------------------------------------------------------------------------


def test_empty_collection_is_invalid(tmp_path: Path) -> None:
    report = encode_phase_report(items=[], collected=0)
    request, _, _ = _request(tmp_path, injected_phase_report=report)
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "empty_collection" in result.reason_codes
    assert not result.production_admissible


def test_usage_error_is_invalid(tmp_path: Path) -> None:
    report = encode_phase_report(items=[], collected=0, usage_error=True)
    request, _, _ = _request(tmp_path, injected_phase_report=report)
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "usage_error" in result.reason_codes


def test_malformed_output_is_invalid(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path, injected_phase_report="not-json-at-all {{{")
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "malformed_output" in result.reason_codes


def test_wrong_schema_report_is_malformed_invalid(tmp_path: Path) -> None:
    request, _, _ = _request(
        tmp_path,
        injected_phase_report={
            "schema": "other-report@1",
            "collected": 1,
            "items": _passing_items(NODE_A),
        },
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "malformed_output" in result.reason_codes


# ---------------------------------------------------------------------------
# Required skip/xfail -> not_modeled unless predeclared advisory
# ---------------------------------------------------------------------------


def test_required_skip_is_not_modeled(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_SKIP,
            "setup": "skip",
            "call": "not_run",
            "teardown": "not_run",
            "markers": ["skip"],
        }
    ]
    request, _, _ = _request(
        tmp_path,
        node_ids=(NODE_SKIP,),
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.NOT_MODELED
    assert any("required_skip_or_xfail" in code for code in result.reason_codes)
    assert not result.production_admissible


def test_required_xfail_is_not_modeled(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_A,
            "setup": "pass",
            "call": "xfail",
            "teardown": "pass",
            "markers": ["xfail"],
            "wasxfail": True,
        }
    ]
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.NOT_MODELED
    assert not result.production_admissible


def test_predeclared_advisory_skip_does_not_block_other_passes(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_SKIP,
            "setup": "skip",
            "call": "not_run",
            "teardown": "not_run",
            "markers": ["skip", "advisory"],
        },
        {
            "nodeid": NODE_A,
            "setup": "pass",
            "call": "pass",
            "teardown": "pass",
            "markers": [],
        },
    ]
    request, _, _ = _request(
        tmp_path,
        node_ids=(NODE_SKIP, NODE_A),
        advisory=PytestAdvisoryPolicy(node_ids=(NODE_SKIP,), markers=("advisory",)),
        injected_phase_report=encode_phase_report(items=items, collected=2),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.production_admissible
    assert any(item.advisory for item in result.phase_accounting if item.nodeid == NODE_SKIP)


def test_advisory_only_skips_without_required_pass_is_not_modeled(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_SKIP,
            "setup": "skip",
            "call": "not_run",
            "teardown": "not_run",
            "markers": ["skip"],
        }
    ]
    request, _, _ = _request(
        tmp_path,
        node_ids=(NODE_SKIP,),
        advisory=PytestAdvisoryPolicy(node_ids=(NODE_SKIP,)),
        injected_phase_report=encode_phase_report(items=items, collected=1),
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.NOT_MODELED
    assert "no_required_pass" in result.reason_codes
    assert not result.production_admissible


# ---------------------------------------------------------------------------
# Existing-test-receipt projection
# ---------------------------------------------------------------------------


def test_authoritative_existing_test_receipt_projection(tmp_path: Path) -> None:
    report = encode_phase_report(items=_passing_items(NODE_A), collected=1)
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=report,
        existing_bridge=True,
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.receipt is not None
    assert result.receipt.test_pass_receipt is not None
    assert result.receipt.test_execution_key is not None
    assert result.receipt.status is TerminalStatus.PASSED
    assert result.receipt.terminal_success
    # Round-trip preserves authoritative projection.
    restored = TestReceipt.from_dict(result.receipt.to_record())
    assert restored.status is TerminalStatus.PASSED
    assert restored.test_pass_receipt is not None
    assert restored.test_pass_receipt.all_phases_pass
    assert restored.test_pass_receipt.admitted
    key_cid = request.existing_test_execution_key.execution_key_id  # type: ignore[union-attr]
    receipt_cid = request.existing_test_pass_receipt.receipt_id  # type: ignore[union-attr]
    assert key_cid in result.receipt.execution.artifact_cids
    assert receipt_cid in result.receipt.execution.artifact_cids


def test_existing_bridge_not_attached_on_failure(tmp_path: Path) -> None:
    items = [
        {
            "nodeid": NODE_A,
            "setup": "pass",
            "call": "fail",
            "teardown": "pass",
            "markers": [],
        }
    ]
    request, _, _ = _request(
        tmp_path,
        injected_phase_report=encode_phase_report(items=items, collected=1),
        existing_bridge=True,
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert result.receipt is not None
    assert result.receipt.test_pass_receipt is None
    assert result.receipt.test_execution_key is None


# ---------------------------------------------------------------------------
# Artifact references + simulated cannot satisfy production
# ---------------------------------------------------------------------------


def test_artifact_references_and_argv_retained(tmp_path: Path) -> None:
    report = encode_phase_report(items=_passing_items(NODE_A), collected=1)
    request, argv, box = _request(tmp_path)
    stdout = json.dumps(report).encode()
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            stdout=stdout,
            stderr=b"",
            exit_code=0,
        )
    )
    # Also drop phase report file under artifact root.
    report_path = Path(box.artifact_root) / request.phase_report_relpath
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = PytestVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.command_argv == tuple(argv)
    assert result.phase_report_cid
    assert result.artifact_cids
    assert result.receipt is not None
    assert result.receipt.execution.stdout_artifact_cid
    assert result.receipt.execution.stderr_artifact_cid
    assert set(result.phase_report_cid for _ in [1]).issubset(set(result.artifact_cids))
    payload = result.to_dict()
    assert payload["command_argv"] == list(argv)
    assert payload["artifact_cids"]
    assert payload["production_admissible"] is True


def test_simulated_mode_cannot_satisfy_production(tmp_path: Path) -> None:
    report = encode_phase_report(items=_passing_items(NODE_A), collected=1)
    request, _, _ = _request(
        tmp_path,
        simulated=True,
        injected_phase_report=report,
    )
    result = PytestVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.SIMULATED
    assert result.simulated is True
    assert not result.production_admissible
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.SIMULATED
    assert "simulated_mode" in result.reason_codes


def test_runner_invoked_with_shell_false_and_hermetic_env(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    report = encode_phase_report(items=_passing_items(NODE_A), collected=1)
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            stdout=json.dumps(report).encode(),
        )
    )
    # Force real report load path via file.
    (Path(box.artifact_root) / request.phase_report_relpath).write_text(
        json.dumps(report), encoding="utf-8"
    )
    result = PytestVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert len(fake.calls) == 1
    command = fake.calls[0]
    assert command.argv[1:3] == ("-m", "pytest")
    assert command.network_policy == NETWORK_POLICY_DENY_ALL
    assert "PIP_INDEX_URL" not in command.environment
    assert command.environment.get("PYTHONHASHSEED") == "0"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_phase_report_and_project_status_helpers() -> None:
    report = encode_phase_report(
        items=[
            PytestNodePhaseAccounting(
                nodeid=NODE_A,
                setup=PhaseOutcome.PASS,
                call=PhaseOutcome.PASS,
                teardown=PhaseOutcome.PASS,
            )
        ],
        collected=1,
    )
    assert report["schema"] == PYTEST_PHASE_REPORT_SCHEMA
    items, collected, errors, usage, malformed = parse_phase_report(report)
    assert collected == 1 and not errors and not usage and not malformed
    assert items[0].all_phases_pass
    status, reasons = project_terminal_status(
        run_result=None,
        phase_items=items,
        collected=collected,
        collection_errors=errors,
        usage_error=usage,
        malformed=malformed,
        mode=PytestRunMode.SELECTED_NODES,
        selected_node_ids=(NODE_A,),
        simulated=False,
    )
    assert status is TerminalStatus.PASSED
    assert "all_required_phases_passed" in reasons


def test_proof_obligation_not_applicable_constant_still_importable() -> None:
    # Sanity: adapter module coexists with contracts package exports.
    assert PROOF_OBLIGATION_NOT_APPLICABLE_CID
    assert EligibilityClass.REPOSITORY_FOREST_BOUND.value == "repository_forest_bound"
