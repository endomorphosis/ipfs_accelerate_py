"""Acceptance tests for the mypy verification adapter (IVP-006)."""

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
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.verification.adapters.mypy_adapter import (
    MYPY_ADAPTER_EVIDENCE,
    MYPY_DIAGNOSTICS_SCHEMA,
    MYPY_VERIFICATION_ADAPTER_INTERFACE,
    MYPY_VERIFICATION_ADAPTER_SCHEMA,
    MypyDiagnostic,
    MypyInvocation,
    MypyRunMode,
    MypyVerificationAdapter,
    MypyVerificationAdapterError,
    MypyVerificationRequest,
    build_mypy_argv,
    create_mypy_verification_adapter,
    encode_diagnostics_report,
    parse_diagnostics_report,
    project_terminal_status,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    TerminalStatus,
    TypeCheckReceipt,
    VerificationIdentityCompiler,
    VerificationReceiptKind,
    VerificationReceiptKey,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    VerificationCancellation,
    VerificationCommand,
    VerificationRunDisposition,
    VerificationRunResult,
    VerificationSandboxIdentity,
    VerificationStreamArtifact,
    build_closed_sandbox,
    build_hermetic_environment,
)


# ---------------------------------------------------------------------------
# Identity / receipt-key helpers
# ---------------------------------------------------------------------------

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)
SELECTOR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/verification-selector-argv@1"

DEFAULT_MYPY = "/usr/bin/mypy"
DEFAULT_PYTHON = "/usr/bin/python3.12"
PATH_A = "pkg/module_a.py"
PATH_B = "pkg/module_b.py"
MODULE_A = "pkg.module_a"


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


def _sandbox(tmp_path: Path) -> VerificationSandboxIdentity:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return build_closed_sandbox(source_root=source, artifact_root=artifacts)


def _compile_type_check_key(
    *,
    selector_argv: Sequence[str] | tuple[str, ...],
    resolved_tool_executable: str,
    tool_version: str = "1.20.2",
    configuration_bytes: bytes = b"[mypy]\nstrict = True\n",
    fixture_bytes: bytes = b"mypy-fixture-payload-v1\n",
    executable_bytes: bytes = b"reviewed-launcher:mypy",
    tool_version_probe_output: bytes | None = None,
    tool_version_probe_argv: Sequence[str] | None = None,
) -> VerificationReceiptKey:
    repository_forest = _repository_forest()
    descriptor = repository_forest.write_descriptor()
    capability_name = "verification-tool"
    executable_sha256 = "sha256:" + hashlib.sha256(executable_bytes).hexdigest()
    dependency_lock_bytes = b"mypy==1.20.2 --hash=sha256:abcd\n"
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
        locator=resolved_tool_executable.rsplit("/", 1)[-1],
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
            "entries": (f"mypy=={tool_version}",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }
    if tool_version_probe_argv is not None:
        version_probe_argv = tuple(tool_version_probe_argv)
    elif (
        len(selector_argv) >= 3
        and selector_argv[1] == "-m"
        and selector_argv[2] == "mypy"
    ):
        version_probe_argv = (
            resolved_tool_executable,
            "-m",
            "mypy",
            "--version",
        )
    else:
        version_probe_argv = (resolved_tool_executable, "--version")
    if tool_version_probe_output is None:
        tool_version_probe_output = f"mypy {tool_version} (compiled: yes)\n".encode()
    expected_environment = {
        **observed_environment,
        "network_policy": NETWORK_POLICY_DENY_ALL,
        "tool_name": "mypy",
        "tool_version": tool_version,
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": resolved_tool_executable,
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            TOOL_EXECUTABLE_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
        ),
        "tool_version_probe_argv": version_probe_argv,
        "tool_version_probe_output_cid": cid_for_bytes(tool_version_probe_output),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": MYPY_VERIFICATION_ADAPTER_SCHEMA,
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
        resolved_tool_executable=resolved_tool_executable,
        tool_executable_bytes=executable_bytes,
        tool_version_probe_argv=version_probe_argv,
        tool_version_probe_output_bytes=tool_version_probe_output,
        claimed_environment_cid=claimed_environment_cid,
        dependency_lock_path=dependency_lock_path,
        dependency_lock_identity=dependency_lock_identity,
        dependency_lock_bytes=dependency_lock_bytes,
        selector_argv=tuple(selector_argv),
        proof_obligation=None,
        tool_name="mypy",
        tool_version=tool_version,
        configuration_bytes=configuration_bytes,
        fixture_data_bytes=(fixture_bytes,),
        network_policy=NETWORK_POLICY_DENY_ALL,
        receipt_schema_version=1,
        receipt_kind=VerificationReceiptKind.TYPE_CHECK,
        adapter_schema=MYPY_VERIFICATION_ADAPTER_SCHEMA,
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


def _request(
    tmp_path: Path,
    *,
    mode: MypyRunMode = MypyRunMode.SELECTED_TARGETS,
    invocation: MypyInvocation = MypyInvocation.DIRECT,
    paths: Sequence[str] = (PATH_A,),
    modules: Sequence[str] = (),
    packages: Sequence[str] = (),
    config_args: Sequence[str] = (),
    extra_mypy_args: Sequence[str] = (),
    simulated: bool = False,
    injected_diagnostics: Any = None,
    mypy_executable: str | None = None,
    python_executable: str | None = None,
    tool_version: str = "1.20.2",
    configuration_bytes: bytes = b"[mypy]\nstrict = True\n",
    executable_bytes: bytes = b"reviewed-launcher:mypy",
    timeout_seconds: float = 30.0,
) -> tuple[MypyVerificationRequest, tuple[str, ...], VerificationSandboxIdentity]:
    mypy = mypy_executable or DEFAULT_MYPY
    python = python_executable or DEFAULT_PYTHON
    box = _sandbox(tmp_path)
    cache_dir = str(Path(box.artifact_root) / ".mypy_cache")
    argv = build_mypy_argv(
        invocation=invocation,
        mypy_executable=mypy,
        python_executable=python,
        mode=mode,
        paths=paths,
        modules=modules,
        packages=packages,
        config_args=config_args,
        extra_mypy_args=extra_mypy_args,
        cache_dir=cache_dir,
    )
    resolved = mypy if invocation is MypyInvocation.DIRECT else python
    key = _compile_type_check_key(
        selector_argv=argv,
        resolved_tool_executable=resolved,
        tool_version=tool_version,
        configuration_bytes=configuration_bytes,
        executable_bytes=executable_bytes,
    )
    request = MypyVerificationRequest(
        receipt_key=key,
        mode=mode,
        sandbox=box,
        cwd=str(box.source_root),
        timeout_seconds=timeout_seconds,
        mypy_executable=mypy if invocation is MypyInvocation.DIRECT else "",
        python_executable=python if invocation is MypyInvocation.PYTHON_MODULE else "",
        invocation=invocation,
        paths=paths,
        modules=modules,
        packages=packages,
        config_args=config_args,
        extra_mypy_args=extra_mypy_args,
        environment=build_hermetic_environment(
            path=os.environ.get("PATH", "/usr/bin:/bin")
        ),
        simulated=simulated,
        injected_diagnostics=injected_diagnostics,
        lane_id=f"mypy-test:{tmp_path.name}",
    )
    return request, argv, box


# ---------------------------------------------------------------------------
# Argv / interface
# ---------------------------------------------------------------------------


def test_explicit_argv_and_reproducible_list(tmp_path: Path) -> None:
    request, argv, box = _request(
        tmp_path,
        paths=(PATH_A, PATH_B),
        config_args=("--config-file", "mypy.ini"),
    )
    adapter = create_mypy_verification_adapter(_FakeRunner())
    built = adapter.build_argv(request)
    assert built == argv
    assert built[0] == request.mypy_executable
    assert "--config-file" in built and "mypy.ini" in built
    assert PATH_A in built and PATH_B in built
    assert "--no-incremental" in built
    assert "--cache-dir" in built
    assert str(Path(box.artifact_root) / ".mypy_cache") in built
    assert "shell" not in " ".join(built).lower()
    again = adapter.build_argv(request)
    assert again == built
    assert list(again) == list(built)


def test_python_module_invocation_argv(tmp_path: Path) -> None:
    request, argv, _ = _request(
        tmp_path,
        invocation=MypyInvocation.PYTHON_MODULE,
        modules=(MODULE_A,),
        paths=(),
    )
    assert request.invocation is MypyInvocation.PYTHON_MODULE
    assert argv[0] == request.python_executable
    assert argv[1:3] == ("-m", "mypy")
    assert "-m" in argv[3:] and MODULE_A in argv


def test_package_or_module_mode_argv(tmp_path: Path) -> None:
    request, argv, _ = _request(
        tmp_path,
        mode=MypyRunMode.PACKAGE_OR_MODULE,
        paths=(),
        packages=("pkg",),
        modules=(),
    )
    assert request.mode is MypyRunMode.PACKAGE_OR_MODULE
    assert "-p" in argv and "pkg" in argv


def test_empty_selector_is_rejected() -> None:
    with pytest.raises(MypyVerificationAdapterError) as excinfo:
        build_mypy_argv(
            mypy_executable=DEFAULT_MYPY,
            paths=(),
            modules=(),
            packages=(),
        )
    assert excinfo.value.reason_code == "empty_selector"


def test_adapter_interface_and_evidence_constants() -> None:
    adapter = create_mypy_verification_adapter()
    assert adapter.interface == MYPY_VERIFICATION_ADAPTER_INTERFACE
    assert adapter.schema == MYPY_VERIFICATION_ADAPTER_SCHEMA
    assert adapter.evidence == MYPY_ADAPTER_EVIDENCE
    assert MYPY_ADAPTER_EVIDENCE == "ivp/mypy-adapter@1"


# ---------------------------------------------------------------------------
# Observed deterministic environment + bindings
# ---------------------------------------------------------------------------


def test_explicit_argv_and_observed_deterministic_environment(tmp_path: Path) -> None:
    report = encode_diagnostics_report(items=(), exit_code=0, checked_files=1, error_count=0)
    request, argv, _ = _request(
        tmp_path,
        config_args=("--strict",),
        injected_diagnostics=report,
    )
    key = request.receipt_key
    assert key.tool_name == "mypy"
    assert key.adapter_schema == MYPY_VERIFICATION_ADAPTER_SCHEMA
    assert key.receipt_kind is VerificationReceiptKind.TYPE_CHECK
    assert key.configuration_cid
    assert key.environment_cid
    assert key.selector_cid == _structured_cid(SELECTOR_SCHEMA, {"argv": list(argv)})
    env = key.environment_observation
    assert env["tool_name"] == "mypy"
    assert env["network_policy"] == NETWORK_POLICY_DENY_ALL
    assert env["environment_values"]["LANG"] == "C.UTF-8"
    assert env["environment_values"]["LC_ALL"] == "C.UTF-8"
    assert env["sandbox_policy"]["network"] == "deny"
    assert env["sandbox_policy"]["auto_install"] == "deny"

    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.receipt is not None
    assert result.receipt.execution.command_argv == argv
    assert result.receipt.key.selector_cid == key.selector_cid
    assert result.receipt.key.configuration_cid == key.configuration_cid
    assert result.receipt.key.environment_cid == key.environment_cid
    assert result.command_argv == argv


def test_selector_mismatch_is_rejected(tmp_path: Path) -> None:
    request, _argv, box = _request(tmp_path, paths=(PATH_A,))
    other_argv = build_mypy_argv(
        mypy_executable=request.mypy_executable,
        paths=(PATH_B,),
        cache_dir=str(Path(box.artifact_root) / ".mypy_cache"),
    )
    other_key = _compile_type_check_key(
        selector_argv=other_argv,
        resolved_tool_executable=request.mypy_executable,
    )
    bad = MypyVerificationRequest(
        receipt_key=other_key,
        mode=MypyRunMode.SELECTED_TARGETS,
        sandbox=box,
        cwd=str(box.source_root),
        timeout_seconds=30.0,
        mypy_executable=request.mypy_executable,
        paths=(PATH_A,),
        environment=dict(request.environment),
        injected_diagnostics=encode_diagnostics_report(exit_code=0),
    )
    with pytest.raises(MypyVerificationAdapterError) as excinfo:
        MypyVerificationAdapter(_FakeRunner()).execute(bad)
    assert excinfo.value.reason_code == "selector_binding_mismatch"


# ---------------------------------------------------------------------------
# Tool executable / version / config mutations change keys
# ---------------------------------------------------------------------------


def test_tool_executable_version_and_config_mutations_change_keys(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    cache_dir = str(Path(box.artifact_root) / ".mypy_cache")
    argv = build_mypy_argv(
        mypy_executable=DEFAULT_MYPY,
        paths=(PATH_A,),
        cache_dir=cache_dir,
    )
    base = _compile_type_check_key(
        selector_argv=argv,
        resolved_tool_executable=DEFAULT_MYPY,
        tool_version="1.20.2",
        configuration_bytes=b"[mypy]\nstrict = True\n",
        executable_bytes=b"reviewed-launcher:mypy-v1",
    )
    version_mutated = _compile_type_check_key(
        selector_argv=argv,
        resolved_tool_executable=DEFAULT_MYPY,
        tool_version="1.21.0",
        configuration_bytes=b"[mypy]\nstrict = True\n",
        executable_bytes=b"reviewed-launcher:mypy-v1",
        tool_version_probe_output=b"mypy 1.21.0 (compiled: yes)\n",
    )
    config_mutated = _compile_type_check_key(
        selector_argv=argv,
        resolved_tool_executable=DEFAULT_MYPY,
        tool_version="1.20.2",
        configuration_bytes=b"[mypy]\nstrict = False\ndisallow_untyped_defs = True\n",
        executable_bytes=b"reviewed-launcher:mypy-v1",
    )
    exe_mutated = _compile_type_check_key(
        selector_argv=build_mypy_argv(
            mypy_executable="/opt/tools/mypy",
            paths=(PATH_A,),
            cache_dir=cache_dir,
        ),
        resolved_tool_executable="/opt/tools/mypy",
        tool_version="1.20.2",
        configuration_bytes=b"[mypy]\nstrict = True\n",
        executable_bytes=b"reviewed-launcher:mypy-alternate",
    )
    assert base.key_id != version_mutated.key_id
    assert base.tool_version != version_mutated.tool_version
    assert base.key_id != config_mutated.key_id
    assert base.configuration_cid != config_mutated.configuration_cid
    assert base.key_id != exe_mutated.key_id
    assert base.environment_cid != exe_mutated.environment_cid
    assert (
        base.environment_observation["tool_executable_cid"]
        != exe_mutated.environment_observation["tool_executable_cid"]
    )


# ---------------------------------------------------------------------------
# Pass / fail / timeout / unavailable / cancelled map losslessly
# ---------------------------------------------------------------------------


def test_pass_maps_to_passed_type_check_receipt(tmp_path: Path) -> None:
    report = encode_diagnostics_report(
        items=(),
        exit_code=0,
        checked_files=1,
        error_count=0,
    )
    request, argv, _ = _request(tmp_path, injected_diagnostics=report)
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.production_admissible
    assert result.ok
    assert result.command_argv == argv
    assert result.receipt is not None
    assert isinstance(result.receipt, TypeCheckReceipt)
    assert result.receipt.terminal_success
    assert result.receipt.status is TerminalStatus.PASSED
    restored = TypeCheckReceipt.from_dict(result.receipt.to_record())
    assert restored.status is TerminalStatus.PASSED


def test_fail_maps_to_failed_with_diagnostics(tmp_path: Path) -> None:
    items = [
        {
            "path": PATH_A,
            "line": 10,
            "column": 5,
            "severity": "error",
            "message": "Name 'x' is not defined",
            "error_code": "name-defined",
        }
    ]
    report = encode_diagnostics_report(
        items=items,
        exit_code=1,
        checked_files=1,
        error_count=1,
    )
    request, _, _ = _request(tmp_path, injected_diagnostics=report)
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert not result.production_admissible
    assert not result.ok
    assert result.error_count == 1
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].error_code == "name-defined"
    assert "type_check_failed" in result.reason_codes
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.FAILED


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
    result = MypyVerificationAdapter(fake).execute(request)
    assert fake.calls and fake.calls[0].argv[0] == request.mypy_executable
    assert result.terminal_status is TerminalStatus.TIMEOUT
    assert not result.production_admissible
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.TIMEOUT
    assert "timeout" in result.reason_codes


def test_cancellation_is_preserved(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    cancel = VerificationCancellation(cancellation_id="cancel:mypy-test")
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
    result = MypyVerificationAdapter(fake).execute(request, cancellation=cancel)
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert not result.production_admissible
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.CANCELLED


def test_missing_mypy_remains_unavailable(tmp_path: Path) -> None:
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
    result = MypyVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert not result.production_admissible
    assert not result.ok
    assert "unavailable" in result.reason_codes or "executable_missing" in result.reason_codes
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Usage / malformed output is invalid
# ---------------------------------------------------------------------------


def test_usage_error_is_invalid(tmp_path: Path) -> None:
    report = encode_diagnostics_report(
        items=(),
        exit_code=2,
        checked_files=0,
        error_count=0,
        usage_error=True,
    )
    request, _, _ = _request(tmp_path, injected_diagnostics=report)
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "usage_error" in result.reason_codes
    assert not result.production_admissible


def test_mypy_exit_code_2_is_invalid(tmp_path: Path) -> None:
    report = encode_diagnostics_report(
        items=(),
        exit_code=2,
        checked_files=0,
        error_count=0,
    )
    request, _, _ = _request(tmp_path, injected_diagnostics=report)
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "usage_error" in result.reason_codes or "mypy_exit_2" in result.reason_codes


def test_malformed_output_is_invalid(tmp_path: Path) -> None:
    request, _, _ = _request(
        tmp_path,
        injected_diagnostics=b"\xff\xfe not-valid-utf8-or-json {{{",
    )
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "malformed_output" in result.reason_codes


def test_wrong_schema_report_is_malformed_invalid(tmp_path: Path) -> None:
    request, _, _ = _request(
        tmp_path,
        injected_diagnostics={
            "schema": "other-report@1",
            "exit_code": 0,
            "items": [],
        },
    )
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "malformed_output" in result.reason_codes


def test_usage_text_from_free_form_is_invalid(tmp_path: Path) -> None:
    text = "usage: mypy [-h] [-v] [-V] [options] [files ...]\n"
    request, _, _ = _request(tmp_path, injected_diagnostics=text)
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.INVALID
    assert "usage_error" in result.reason_codes


# ---------------------------------------------------------------------------
# Bounded diagnostics are artifacts
# ---------------------------------------------------------------------------


def test_bounded_diagnostics_are_artifacts(tmp_path: Path) -> None:
    items = [
        MypyDiagnostic(
            path=PATH_A,
            line=3,
            column=1,
            severity="error",
            message="Incompatible types in assignment",
            error_code="assignment",
        ),
        MypyDiagnostic(
            path=PATH_A,
            line=4,
            severity="note",
            message="Expression has type 'int'",
        ),
    ]
    report = encode_diagnostics_report(
        items=items,
        exit_code=1,
        checked_files=1,
        error_count=1,
    )
    request, argv, box = _request(tmp_path)
    stdout = (
        f"{PATH_A}:3:1: error: Incompatible types in assignment  [assignment]\n"
        f"{PATH_A}:4: note: Expression has type 'int'\n"
        "Found 1 error in 1 file (checked 1 source file)\n"
    ).encode()
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            terminal_status=TerminalStatus.FAILED,
            disposition=VerificationRunDisposition.COMPLETED,
            exit_code=1,
            stdout=stdout,
            stderr=b"",
        )
    )
    # Persist structured diagnostics under artifact root as well.
    report_path = Path(box.artifact_root) / request.diagnostics_relpath
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = MypyVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert result.diagnostics_cid
    assert result.artifact_cids
    assert result.diagnostics_cid in result.artifact_cids
    assert result.receipt is not None
    assert result.receipt.execution.stdout_artifact_cid
    assert result.receipt.execution.stderr_artifact_cid
    assert set(result.diagnostics_cid for _ in [1]).issubset(set(result.artifact_cids))
    assert result.receipt.artifact_cids
    payload = result.to_dict()
    assert payload["command_argv"] == list(argv)
    assert payload["artifact_cids"]
    assert payload["diagnostics_cid"]
    assert payload["production_admissible"] is False


def test_free_form_mypy_text_parsed_into_diagnostics(tmp_path: Path) -> None:
    text = (
        f"{PATH_A}:10: error: Name 'x' is not defined  [name-defined]\n"
        "Found 1 error in 1 file (checked 1 source file)\n"
    )
    items, exit_code, error_count, checked, usage, malformed = parse_diagnostics_report(
        text, fallback_exit_code=1
    )
    assert not usage and not malformed
    assert exit_code == 1
    assert error_count == 1
    assert checked == 1
    assert len(items) == 1
    assert items[0].path == PATH_A
    assert items[0].error_code == "name-defined"

    request, argv, box = _request(tmp_path)
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            exit_code=1,
            stdout=text.encode(),
            terminal_status=TerminalStatus.FAILED,
        )
    )
    result = MypyVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.FAILED
    assert result.error_count >= 1
    assert any(item.is_error for item in result.diagnostics)


# ---------------------------------------------------------------------------
# Simulated cannot satisfy production + hermetic runner
# ---------------------------------------------------------------------------


def test_simulated_mode_cannot_satisfy_production(tmp_path: Path) -> None:
    report = encode_diagnostics_report(exit_code=0, checked_files=1, error_count=0)
    request, _, _ = _request(
        tmp_path,
        simulated=True,
        injected_diagnostics=report,
    )
    result = MypyVerificationAdapter(_FakeRunner()).execute(request)
    assert result.terminal_status is TerminalStatus.SIMULATED
    assert result.simulated is True
    assert not result.production_admissible
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.SIMULATED
    assert "simulated_mode" in result.reason_codes


def test_runner_invoked_with_hermetic_env_and_no_shell(tmp_path: Path) -> None:
    request, argv, box = _request(tmp_path)
    report = encode_diagnostics_report(exit_code=0, checked_files=1, error_count=0)
    fake = _FakeRunner(
        _run_result(
            argv=argv,
            cwd=str(box.source_root),
            environment=dict(request.environment),
            sandbox=box,
            stdout=json.dumps(report).encode(),
        )
    )
    (Path(box.artifact_root) / request.diagnostics_relpath).write_text(
        json.dumps(report), encoding="utf-8"
    )
    result = MypyVerificationAdapter(fake).execute(request)
    assert result.terminal_status is TerminalStatus.PASSED
    assert len(fake.calls) == 1
    command = fake.calls[0]
    assert command.argv[0] == request.mypy_executable
    assert command.network_policy == NETWORK_POLICY_DENY_ALL
    assert "PIP_INDEX_URL" not in command.environment
    assert command.environment.get("PYTHONHASHSEED") == "0"
    assert command.environment.get("MYPY_CACHE_DIR")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_and_project_helpers() -> None:
    report = encode_diagnostics_report(
        items=[
            MypyDiagnostic(
                path=PATH_A,
                line=1,
                severity="error",
                message="boom",
                error_code="misc",
            )
        ],
        exit_code=1,
        checked_files=1,
        error_count=1,
    )
    assert report["schema"] == MYPY_DIAGNOSTICS_SCHEMA
    items, exit_code, errors, checked, usage, malformed = parse_diagnostics_report(report)
    assert exit_code == 1 and errors == 1 and checked == 1
    assert not usage and not malformed
    assert items[0].is_error
    status, reasons = project_terminal_status(
        run_result=None,
        diagnostics=items,
        exit_code=exit_code,
        error_count=errors,
        usage_error=usage,
        malformed=malformed,
        simulated=False,
    )
    assert status is TerminalStatus.FAILED
    assert "type_check_failed" in reasons

    pass_status, pass_reasons = project_terminal_status(
        run_result=None,
        diagnostics=(),
        exit_code=0,
        error_count=0,
        usage_error=False,
        malformed=False,
        simulated=False,
    )
    assert pass_status is TerminalStatus.PASSED
    assert "type_check_passed" in pass_reasons


def test_host_resource_snapshot_still_importable() -> None:
    # Sanity: fixture dependencies remain importable in this test module.
    host = HostResourceSnapshot(
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
    assert host.worker_limit == 16
