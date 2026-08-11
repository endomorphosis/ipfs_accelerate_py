"""Acceptance tests for Z3 and existing proof-assistant adapters (IVP-007)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

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
    AssuranceLevel,
    AttemptStatus,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofAttempt,
    ProofEvidence,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ProofReceipt as FormalProofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.verification.adapters.prover_adapters import (
    PROOF_ASSISTANT_ADAPTER_INTERFACE,
    PROOF_ASSISTANT_ADAPTER_SCHEMA,
    PROVER_ADAPTER_EVIDENCE,
    Z3_VERIFICATION_ADAPTER_INTERFACE,
    Z3_VERIFICATION_ADAPTER_SCHEMA,
    ExistingProofAssistantAdapter,
    ProofAssistantVerificationRequest,
    ProofAuthoritySource,
    ProverVerificationAdapterError,
    RegistryKernelAdmission,
    Z3SolverOutcome,
    Z3VerificationAdapter,
    Z3VerificationRequest,
    build_z3_argv,
    create_existing_proof_assistant_adapter,
    create_z3_verification_adapter,
    parse_z3_solver_outcome,
    source_contains_incomplete_or_unsafe_proof,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    TerminalStatus,
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
# Compact identity recipes
# ---------------------------------------------------------------------------

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)


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
        resource_classes=("cpu-validation", "cpu-proof", "cpu-small"),
    )


def _sandbox(tmp_path: Path) -> VerificationSandboxIdentity:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return build_closed_sandbox(source_root=source, artifact_root=artifacts)


def _compile_proof_key(
    *,
    tool_name: str,
    tool_version: str,
    selector_argv: Sequence[str],
    adapter_schema: str,
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
    attempt_stage: ProofStage = ProofStage.SOLVE,
    translator_id: str = "translator:python-smt@1.2.0",
    solver_id: str = "solver:z3@4.13.3",
    kernel_id: str = "kernel:reviewed-z3-result@1",
    statement: str = "not (x >= 0 implies result >= 0)",
) -> VerificationReceiptKey:
    repository_forest = _repository_forest()
    descriptor = repository_forest.write_descriptor()
    capability_name = "verification-tool"
    executable_bytes = f"reviewed-launcher:{tool_name}".encode()
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
        locator=selector_argv[0].rsplit("/", 1)[-1],
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
            "name": "locked-prover",
            "revision": "fixture-1",
        },
        "dependency_distribution": {
            "schema": "verification-dependency-distribution@1",
            "entries": (f"{tool_name}=={tool_version}",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }
    invocation_prefix = (
        selector_argv[:3]
        if len(selector_argv) >= 3 and selector_argv[1] == "-m"
        else selector_argv[:1]
    )
    version_probe_argv = (*invocation_prefix, "--version")
    tool_version_probe_output = f"{tool_name} {tool_version}\n".encode()
    tool_executable_cid = _structured_cid(
        TOOL_EXECUTABLE_SCHEMA,
        {"capability_name": capability_name, "sha256": executable_sha256},
    )
    expected_environment = {
        **observed_environment,
        "network_policy": NETWORK_POLICY_DENY_ALL,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": selector_argv[0],
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": tool_executable_cid,
        "tool_version_probe_argv": version_probe_argv,
        "tool_version_probe_output_cid": cid_for_bytes(tool_version_probe_output),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": adapter_schema,
        "capability_environment_names": tuple(
            sorted(capability_snapshot.environment_names)
        ),
        "capability_read_paths": tuple(sorted(capability_snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(capability_snapshot.write_paths)),
        "capability_lock_identities": dict(
            sorted(capability_snapshot.lock_identities.items())
        ),
        "selected_dependency_lock_path": dependency_lock_path,
        "selected_dependency_lock_identity": dependency_lock_identity.to_dict(),
    }
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
    semantic = {
        "symbols": ["example.calculate@2"],
        "edge_root": "sha256:semantic-edges",
    }
    proof_obligation = CodeProofObligation(
        repository_id=descriptor.repository_id,
        repository_tree_id=descriptor.tree,
        ast_scope_ids=("scope:example.calculate",),
        statement=statement,
        premise_ids=("premise:contract",),
        template_id="python-contract-to-smtlib2",
        template_version="1.2.0",
        template_semantic_hash="sha256:" + "a" * 64,
        required_assurance=required_assurance,
    )
    proof_backend_binding = {
        "plan_id": _artifact("formal-plan"),
        "step_id": "proof-step:adapter",
        "attempt_stage": attempt_stage.value,
        "attempt_provider_id": "provider:fixture",
        "provider_id": "provider:fixture",
        "repository_tree_identity_kind": "git_tree",
        "repository_tree_identity": descriptor.tree,
        "translator_id": translator_id,
        "solver_id": solver_id,
        "kernel_id": kernel_id,
        "toolchain_id": "toolchain:locked@1",
        "policy_id": "policy:proof@1",
        "theorem_registry_id": "registry:fixture@1",
        "ast_scope_ids": ("scope:example.calculate",),
        "premise_ids": ("premise:contract",),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "tool_executable_cid": tool_executable_cid,
    }
    return VerificationIdentityCompiler().compile_key(
        repository_forest=repository_forest,
        repository_alias=repository_forest.sole_write_alias,
        claimed_repository_tree_cid=_structured_cid(TREE_SCHEMA, tree_observation),
        patch_base_tree_id="git-tree:base",
        repository_state_tree_id="git-tree:base",
        invalidation_plan_tree_id="git-tree:base",
        context_pack_tree_id="git-tree:base",
        observed_semantic_state=semantic,
        repository_state_semantic_root_cid=_structured_cid(
            "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1", semantic
        ),
        invalidation_plan_semantic_root_cid=_structured_cid(
            "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1", semantic
        ),
        context_pack_semantic_root_cid=_structured_cid(
            "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1", semantic
        ),
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
        resolved_tool_executable=selector_argv[0],
        tool_executable_bytes=executable_bytes,
        tool_version_probe_argv=version_probe_argv,
        tool_version_probe_output_bytes=tool_version_probe_output,
        claimed_environment_cid=_structured_cid(
            ENVIRONMENT_SCHEMA, expected_environment
        ),
        dependency_lock_path=dependency_lock_path,
        dependency_lock_identity=dependency_lock_identity,
        dependency_lock_bytes=dependency_lock_bytes,
        selector_argv=tuple(selector_argv),
        proof_obligation=proof_obligation,
        tool_name=tool_name,
        tool_version=tool_version,
        configuration_bytes=b"[tool]\nstrict = true\n",
        fixture_data_bytes=(b"fixture-one\n",),
        network_policy=NETWORK_POLICY_DENY_ALL,
        receipt_schema_version=1,
        receipt_kind=VerificationReceiptKind.PROOF,
        adapter_schema=adapter_schema,
        proof_backend_binding=proof_backend_binding,
    )


def _z3_key(
    *,
    executable: str = "/usr/bin/z3",
    version: str = "4.13.3",
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
) -> VerificationReceiptKey:
    return _compile_proof_key(
        tool_name="z3",
        tool_version=version,
        selector_argv=(executable, "-smt2", "obligation.smt2"),
        adapter_schema=Z3_VERIFICATION_ADAPTER_SCHEMA,
        required_assurance=required_assurance,
        attempt_stage=ProofStage.SOLVE,
        translator_id="translator:python-smt@1.2.0",
        solver_id=f"solver:z3@{version}",
        kernel_id="kernel:reviewed-z3-result@1",
    )


def _lean_key(
    *,
    executable: str = "/usr/bin/lean",
    version: str = "4.0.0",
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> VerificationReceiptKey:
    return _compile_proof_key(
        tool_name="lean",
        tool_version=version,
        selector_argv=(executable, "kernel_probe.lean"),
        adapter_schema=PROOF_ASSISTANT_ADAPTER_SCHEMA,
        required_assurance=required_assurance,
        attempt_stage=ProofStage.KERNEL_VERIFY,
        translator_id="translator:lean-native@1",
        solver_id="solver:none@0",
        kernel_id="kernel:lean4@1",
        statement="example : True := by trivial",
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=30_000,
        cpu_time_ms=20_000,
        memory_bytes=512 * 1024 * 1024,
        disk_bytes=64 * 1024 * 1024,
        max_processes=4,
        max_premises=32,
        max_output_bytes=1_000_000,
        model_token_limit=0,
        provider_quota=1,
        network_allowed=False,
    )


def _operational_lean_admission(
    *,
    executable: str = "/usr/bin/lean",
    version: str = "4.0.0",
) -> RegistryKernelAdmission:
    return RegistryKernelAdmission(
        prover_id="lean",
        offline=True,
        admitted=True,
        authority_capability="lean_kernel_check",
        smoke_tested=True,
        versioned=True,
        executable_path=executable,
        executable_version=version,
        fixture_id="lean-kernel-smoke@1",
        registry_entry_id="registry:lean@1",
    )


def _empty_stream() -> VerificationStreamArtifact:
    empty = cid_for_bytes(b"")
    digest = "sha256:" + hashlib.sha256(b"").hexdigest()
    return VerificationStreamArtifact(
        digest=digest,
        cid=empty,
        truncated=False,
        byte_count=0,
        captured_byte_count=0,
        preview="",
    )


class _FakeRunner:
    """Injected process runner for deterministic adapter tests."""

    def __init__(
        self,
        result: VerificationRunResult | None = None,
        *,
        raise_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.raise_error = raise_error
        self.calls: list[tuple[VerificationCommand, VerificationCancellation | None]] = []

    def run(
        self,
        command: VerificationCommand,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> VerificationRunResult:
        self.calls.append((command, cancellation))
        if self.raise_error is not None:
            raise self.raise_error
        assert self.result is not None
        # Echo command argv into the result for observation binding.
        return replace(
            self.result,
            command_argv=tuple(command.argv),
            executable=command.argv[0] if command.argv else "",
        )


def _run_result(
    *,
    disposition: VerificationRunDisposition = VerificationRunDisposition.COMPLETED,
    exit_code: int | None = 0,
    stdout: str = "unsat\n",
    timed_out: bool = False,
    cancelled: bool = False,
    unavailable: bool = False,
    publication_allowed: bool | None = None,
    process_started: bool = True,
    reason_codes: tuple[str, ...] = (),
) -> VerificationRunResult:
    if publication_allowed is None:
        publication_allowed = not (timed_out or cancelled or unavailable)
    terminal = {
        VerificationRunDisposition.COMPLETED: TerminalStatus.PASSED,
        VerificationRunDisposition.FAILED: TerminalStatus.FAILED,
        VerificationRunDisposition.TIMEOUT: TerminalStatus.TIMEOUT,
        VerificationRunDisposition.CANCELLED: TerminalStatus.CANCELLED,
        VerificationRunDisposition.UNAVAILABLE: TerminalStatus.UNAVAILABLE,
    }[disposition]
    stdout_bytes = stdout.encode("utf-8")
    stream = VerificationStreamArtifact(
        digest="sha256:" + hashlib.sha256(stdout_bytes).hexdigest(),
        cid=cid_for_bytes(stdout_bytes),
        truncated=False,
        byte_count=len(stdout_bytes),
        captured_byte_count=len(stdout_bytes),
        preview=stdout[:4096],
    )
    return VerificationRunResult(
        terminal_status=terminal,
        disposition=disposition,
        exit_code=exit_code,
        duration_ms=12,
        command_argv=("/usr/bin/z3", "-smt2", "obligation.smt2"),
        executable="/usr/bin/z3",
        cwd="/tmp",
        environment={},
        sandbox={},
        network_policy=NETWORK_POLICY_DENY_ALL,
        timeout_seconds=5.0,
        stdout=stream,
        stderr=_empty_stream(),
        process_started=process_started,
        publication_allowed=publication_allowed,
        timed_out=timed_out,
        cancelled=cancelled,
        unavailable=unavailable,
        reason_codes=reason_codes,
        reason=reason_codes[0] if reason_codes else "",
    )


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------


def test_adapter_interface_and_evidence_constants() -> None:
    assert Z3_VERIFICATION_ADAPTER_INTERFACE == "Z3VerificationAdapter@1"
    assert Z3_VERIFICATION_ADAPTER_SCHEMA == "z3-verification-adapter@1"
    assert PROOF_ASSISTANT_ADAPTER_INTERFACE == "ExistingProofAssistantAdapter@1"
    assert PROOF_ASSISTANT_ADAPTER_SCHEMA == "existing-proof-assistant-adapter@1"
    assert PROVER_ADAPTER_EVIDENCE == "ivp/prover-adapter@1"
    adapter = create_z3_verification_adapter()
    assert adapter.interface == Z3_VERIFICATION_ADAPTER_INTERFACE
    assert adapter.evidence == PROVER_ADAPTER_EVIDENCE
    pa = create_existing_proof_assistant_adapter()
    assert pa.interface == PROOF_ASSISTANT_ADAPTER_INTERFACE


def test_parse_z3_outcome_and_bare_text() -> None:
    assert parse_z3_solver_outcome("unsat\n") is Z3SolverOutcome.UNSAT
    assert parse_z3_solver_outcome("sat\n") is Z3SolverOutcome.SAT
    assert parse_z3_solver_outcome("unknown\n") is Z3SolverOutcome.UNKNOWN
    # Prose containing the word unsat is not a structured outcome.
    assert (
        parse_z3_solver_outcome("the formula is unsat under these axioms\n")
        is Z3SolverOutcome.MALFORMED
    )
    assert parse_z3_solver_outcome("") is Z3SolverOutcome.MALFORMED


def test_build_z3_argv_is_explicit() -> None:
    argv = build_z3_argv(
        z3_executable="/usr/bin/z3",
        smtlib_relpath="obligation.smt2",
        extra_z3_args=("-T:5",),
    )
    assert argv == ("/usr/bin/z3", "-smt2", "-T:5", "obligation.smt2")


def test_incomplete_proof_detection() -> None:
    bad, reason = source_contains_incomplete_or_unsafe_proof("theorem t : True := sorry")
    assert bad
    assert "sorry" in reason or "incomplete" in reason
    bad2, _ = source_contains_incomplete_or_unsafe_proof("theorem t : True := by admit")
    assert bad2
    bad3, _ = source_contains_incomplete_or_unsafe_proof("unsafe def x := 1")
    assert bad3
    ok, reason_ok = source_contains_incomplete_or_unsafe_proof(
        "theorem t : True := by trivial"
    )
    assert not ok
    assert reason_ok == ""


# ---------------------------------------------------------------------------
# Z3 adapter
# ---------------------------------------------------------------------------


def test_z3_unsat_proves_with_bound_obligation_and_assurance(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    # Use injection path (no live runner needed).
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        smtlib_payload="(assert false)\n(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNSAT,
        injected_stdout="unsat\n",
        injected_exit_code=0,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.PROVED
    assert result.ok
    assert result.production_admissible
    assert result.solver_outcome is Z3SolverOutcome.UNSAT
    assert result.authority_source is ProofAuthoritySource.CURRENT_DIRECT_EXECUTION
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.PROVED
    assert result.formal_proof_receipt is not None
    assert result.formal_proof_receipt.obligation_id == key.proof_obligation_cid
    assert result.formal_proof_receipt.translator_id == "translator:python-smt@1.2.0"
    assert result.formal_proof_receipt.solver_id == "solver:z3@4.13.3"
    assert result.command_argv[0] == "/usr/bin/z3"
    assert "-smt2" in result.command_argv
    assert result.solver_report_cid
    assert result.artifact_cids


def test_z3_sat_disproves_with_verified_counterexample(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        smtlib_payload="(assert true)\n(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.SAT,
        injected_stdout="sat\n",
        counterexample_verified=True,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.DISPROVED
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt.status is TerminalStatus.DISPROVED
    assert result.formal_proof_receipt is not None
    assert result.formal_proof_receipt.authoritative_verdict is ProofVerdict.DISPROVED


def test_z3_unknown_stays_unknown(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNKNOWN,
        injected_stdout="unknown\n",
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.UNKNOWN
    assert not result.production_admissible
    assert not result.ok


def test_bare_solver_text_cannot_prove(tmp_path: Path) -> None:
    """Prose mentioning unsat without structured binding cannot prove."""
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    # Mismatched translator_id breaks the obligation/tool binding.
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:WRONG@9",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNSAT,
        injected_stdout="unsat\n",
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status in {
        TerminalStatus.UNKNOWN,
        TerminalStatus.INVALID,
    }
    assert result.terminal_status is not TerminalStatus.PROVED
    assert not result.production_admissible
    assert "bare_solver_text_not_authority" in result.reason_codes or any(
        "mismatch" in code for code in result.reason_codes
    )


def test_absent_z3_is_unavailable_never_proves(tmp_path: Path) -> None:
    key = _z3_key(executable="/usr/bin/z3")
    sandbox = _sandbox(tmp_path)
    fake = _FakeRunner(
        _run_result(
            disposition=VerificationRunDisposition.UNAVAILABLE,
            exit_code=None,
            unavailable=True,
            process_started=False,
            publication_allowed=False,
            reason_codes=("executable_missing",),
            stdout="",
        )
    )
    adapter = Z3VerificationAdapter(process_runner=fake)  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=2.0,
        smtlib_payload="(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        environment=build_hermetic_environment(),
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert result.terminal_status is not TerminalStatus.PROVED
    assert not result.production_admissible
    assert result.publication_allowed is False
    assert fake.calls


def test_z3_timeout_never_proves(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    fake = _FakeRunner(
        _run_result(
            disposition=VerificationRunDisposition.TIMEOUT,
            exit_code=None,
            timed_out=True,
            publication_allowed=False,
            reason_codes=("timeout",),
            stdout="",
        )
    )
    adapter = Z3VerificationAdapter(process_runner=fake)  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=0.05,
        smtlib_payload="(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        environment=build_hermetic_environment(),
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.TIMEOUT
    assert result.terminal_status is not TerminalStatus.PROVED
    assert result.formal_proof_receipt is None
    assert result.publication_allowed is False


def test_z3_cancellation_fences_late_output(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    cancel = VerificationCancellation(cancellation_id="cancel:z3-test")
    fake = _FakeRunner(
        _run_result(
            disposition=VerificationRunDisposition.CANCELLED,
            exit_code=None,
            cancelled=True,
            publication_allowed=False,
            reason_codes=("cancelled",),
            stdout="unsat\n",  # late success text must not publish
        )
    )
    adapter = Z3VerificationAdapter(process_runner=fake)  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        smtlib_payload="(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        environment=build_hermetic_environment(),
        resource_budget=_budget(),
    )
    # Pre-cancel so token is observed; fake still returns cancelled disposition.
    cancel.cancel(cancellation_id="cancel:z3-test", reason="operator-abort")
    result = adapter.execute(request, cancellation=cancel)
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.publication_allowed is False
    assert result.terminal_status is not TerminalStatus.PROVED
    assert fake.calls


def test_z3_existing_authoritative_evidence_projects_proved(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    backend = key.proof_backend_binding
    assert backend is not None
    evidence = ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=_artifact("solver-result"),
        subject_id=key.proof_obligation_cid,
        verifier_id=backend["solver_id"],
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata={"solver_outcome": "unsat", "counterexample_verified": False},
    )
    attempt = ProofAttempt(
        plan_id=backend["plan_id"],
        step_id=backend["step_id"],
        obligation_id=key.proof_obligation_cid,
        repository_tree_id=backend["repository_tree_identity"],
        provider_id=backend["attempt_provider_id"],
        stage=ProofStage(backend["attempt_stage"]),
        status=AttemptStatus.SUCCEEDED,
        evidence=(evidence,),
        input_ids=(key.key_id,),
        output_ids=(evidence.evidence_id,),
    )
    formal = FormalProofReceipt(
        obligation_id=key.proof_obligation_cid,
        plan_id=backend["plan_id"],
        attempt_id=attempt.attempt_id,
        repository_id=backend["repository_id"],
        repository_tree_id=backend["repository_tree_identity"],
        ast_scope_ids=tuple(backend["ast_scope_ids"]),
        premise_ids=tuple(backend["premise_ids"]),
        translator_id=backend["translator_id"],
        solver_id=backend["solver_id"],
        kernel_id=backend["kernel_id"],
        toolchain_id=backend["toolchain_id"],
        theorem_registry_id=backend["theorem_registry_id"],
        policy_id=backend["policy_id"],
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        provider_id=backend["provider_id"],
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=EvidenceFreshness.CURRENT,
    )
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id=backend["translator_id"],
        translator_version=key.tool_version,
        existing_formal_proof_receipt=formal,
        existing_proof_attempt=attempt,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.PROVED
    assert (
        result.authority_source
        is ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE
    )
    assert result.ok


def test_z3_solver_checked_insufficient_for_kernel_required(tmp_path: Path) -> None:
    key = _z3_key(required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNSAT,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    # Solver evidence cannot satisfy kernel_verified.
    assert result.terminal_status is TerminalStatus.UNKNOWN
    assert not result.production_admissible


def test_z3_simulated_cannot_satisfy_production(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        simulated=True,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.SIMULATED
    assert not result.production_admissible
    assert result.publication_allowed is False


def test_z3_live_runner_with_fake_unsat(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    fake = _FakeRunner(_run_result(stdout="unsat\n", exit_code=0))
    adapter = Z3VerificationAdapter(process_runner=fake)  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        smtlib_payload="(assert false)\n(check-sat)\n",
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        environment=build_hermetic_environment(),
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.PROVED
    assert fake.calls
    command, _ = fake.calls[0]
    assert command.argv[0] == "/usr/bin/z3"
    assert command.argv[1] == "-smt2"
    # SMT file was written into the sandbox artifact root.
    smt_path = Path(sandbox.artifact_root) / "obligation.smt2"
    assert smt_path.is_file()
    assert "check-sat" in smt_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existing proof-assistant adapter
# ---------------------------------------------------------------------------


def test_proof_assistant_unavailable_without_registry_admission(
    tmp_path: Path,
) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=None,
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert not result.production_admissible


def test_proof_assistant_rejects_non_offline_admission(tmp_path: Path) -> None:
    with pytest.raises(ProverVerificationAdapterError) as excinfo:
        RegistryKernelAdmission(
            prover_id="lean",
            offline=False,
            admitted=True,
            authority_capability="lean_kernel_check",
            smoke_tested=True,
            executable_path="/usr/bin/lean",
        )
    assert excinfo.value.reason_code == "kernel_not_offline"


def test_proof_assistant_rejects_unknown_prover() -> None:
    with pytest.raises(ProverVerificationAdapterError) as excinfo:
        RegistryKernelAdmission(
            prover_id="vampire",
            offline=True,
            admitted=True,
            authority_capability="first_order_theorem",
            smoke_tested=True,
            executable_path="/usr/bin/vampire",
        )
    assert excinfo.value.reason_code == "kernel_not_registry_admitted"


def test_proof_assistant_sorry_cannot_prove(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=_operational_lean_admission(),
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := sorry",
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is not TerminalStatus.PROVED
    assert result.terminal_status is TerminalStatus.UNKNOWN
    assert not result.production_admissible
    assert any("sorry" in c or "draft" in c or "incomplete" in c for c in result.reason_codes)


def test_proof_assistant_model_draft_cannot_prove(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=_operational_lean_admission(),
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        model_generated_draft=True,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is not TerminalStatus.PROVED
    assert not result.production_admissible


def test_proof_assistant_admit_unsafe_cannot_prove(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    for source in (
        "Theorem t : True. admit.",
        "unsafe def boom := 1",
    ):
        request = ProofAssistantVerificationRequest(
            receipt_key=key,
            sandbox=sandbox,
            cwd=str(sandbox.source_root),
            timeout_seconds=5.0,
            registry_admission=_operational_lean_admission(),
            kernel_executable="/usr/bin/lean",
            checked_source=source,
            resource_budget=_budget(),
        )
        result = adapter.execute(request)
        assert result.terminal_status is not TerminalStatus.PROVED


def test_proof_assistant_kernel_accept_proves(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=_operational_lean_admission(),
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        injected_kernel_accepted=True,
        injected_stdout='{"severity":"information"}\n',
        injected_exit_code=0,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.PROVED
    assert result.ok
    assert (
        result.authority_source is ProofAuthoritySource.CURRENT_DIRECT_EXECUTION
    )
    assert result.formal_proof_receipt is not None
    assert (
        result.formal_proof_receipt.authoritative_assurance
        is AssuranceLevel.KERNEL_VERIFIED
    )


def test_proof_assistant_kernel_reject_does_not_prove(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=_operational_lean_admission(),
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        injected_kernel_accepted=False,
        injected_failure_code="kernel_rejected",
        injected_exit_code=1,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is not TerminalStatus.PROVED
    assert result.terminal_status is TerminalStatus.UNKNOWN


def test_proof_assistant_existing_evidence_projects_status(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    backend = key.proof_backend_binding
    assert backend is not None
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=_artifact("kernel-result"),
        subject_id=key.proof_obligation_cid,
        verifier_id=backend["kernel_id"],
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )
    attempt = ProofAttempt(
        plan_id=backend["plan_id"],
        step_id=backend["step_id"],
        obligation_id=key.proof_obligation_cid,
        repository_tree_id=backend["repository_tree_identity"],
        provider_id=backend["attempt_provider_id"],
        stage=ProofStage(backend["attempt_stage"]),
        status=AttemptStatus.SUCCEEDED,
        evidence=(evidence,),
        input_ids=(key.key_id,),
        output_ids=(evidence.evidence_id,),
    )
    formal = FormalProofReceipt(
        obligation_id=key.proof_obligation_cid,
        plan_id=backend["plan_id"],
        attempt_id=attempt.attempt_id,
        repository_id=backend["repository_id"],
        repository_tree_id=backend["repository_tree_identity"],
        ast_scope_ids=tuple(backend["ast_scope_ids"]),
        premise_ids=tuple(backend["premise_ids"]),
        translator_id=backend["translator_id"],
        solver_id=backend["solver_id"],
        kernel_id=backend["kernel_id"],
        toolchain_id=backend["toolchain_id"],
        theorem_registry_id=backend["theorem_registry_id"],
        policy_id=backend["policy_id"],
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        provider_id=backend["provider_id"],
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=EvidenceFreshness.CURRENT,
    )
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        # No operational registry probe — existing evidence is still authoritative.
        registry_admission=None,
        kernel_executable="/usr/bin/lean",
        existing_formal_proof_receipt=formal,
        existing_proof_attempt=attempt,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.PROVED
    assert (
        result.authority_source
        is ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE
    )
    assert result.ok


def test_proof_assistant_cancellation_fences(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    cancel = VerificationCancellation(cancellation_id="cancel:lean")
    fake = _FakeRunner(
        _run_result(
            disposition=VerificationRunDisposition.CANCELLED,
            cancelled=True,
            publication_allowed=False,
            reason_codes=("cancelled",),
            stdout="",
            exit_code=None,
        )
    )
    adapter = ExistingProofAssistantAdapter(process_runner=fake)  # type: ignore[arg-type]
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=_operational_lean_admission(),
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        environment=build_hermetic_environment(),
        resource_budget=_budget(),
    )
    cancel.cancel(cancellation_id="cancel:lean", reason="stop")
    result = adapter.execute(request, cancellation=cancel)
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.publication_allowed is False
    assert fake.calls


def test_proof_assistant_non_smoke_tested_unavailable(tmp_path: Path) -> None:
    key = _lean_key()
    sandbox = _sandbox(tmp_path)
    admission = RegistryKernelAdmission(
        prover_id="lean",
        offline=True,
        admitted=True,
        authority_capability="lean_kernel_check",
        smoke_tested=False,
        versioned=True,
        executable_path="/usr/bin/lean",
        executable_version="4.0.0",
    )
    assert not admission.operational
    adapter = ExistingProofAssistantAdapter(
        process_runner=_FakeRunner()  # type: ignore[arg-type]
    )
    request = ProofAssistantVerificationRequest(
        receipt_key=key,
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        registry_admission=admission,
        kernel_executable="/usr/bin/lean",
        checked_source="theorem t : True := by trivial",
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE


def test_receipt_round_trip_for_z3_proved(tmp_path: Path) -> None:
    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNSAT,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    assert result.receipt is not None
    from ipfs_accelerate_py.agent_supervisor.verification.contracts import ProofReceipt

    rebuilt = ProofReceipt.from_dict(result.receipt.to_record())
    assert rebuilt.status is TerminalStatus.PROVED
    assert rebuilt.receipt_id == result.receipt.receipt_id


def test_result_to_dict_is_json_safe(tmp_path: Path) -> None:
    import json

    key = _z3_key()
    sandbox = _sandbox(tmp_path)
    adapter = Z3VerificationAdapter(process_runner=_FakeRunner())  # type: ignore[arg-type]
    request = Z3VerificationRequest(
        receipt_key=key,
        z3_executable="/usr/bin/z3",
        sandbox=sandbox,
        cwd=str(sandbox.source_root),
        timeout_seconds=5.0,
        translator_id="translator:python-smt@1.2.0",
        translator_version="4.13.3",
        injected_solver_outcome=Z3SolverOutcome.UNKNOWN,
        resource_budget=_budget(),
    )
    result = adapter.execute(request)
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["evidence"] == [
        PROVER_ADAPTER_EVIDENCE,
        "ivp/process-runner@1",
        "ivp/process-tree-cancellation@1",
    ]
