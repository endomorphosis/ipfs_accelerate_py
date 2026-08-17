from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_forest import (
    AuthorityMode,
    ForestPolicy,
    ForestRootSpec,
    LocalLocator,
    PortableGitClosure,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    RepositoryIdentity,
    build_repository_forest,
    freeze_repository_forest,
    replay_repository_forest,
)
from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    AnalysisExecutionProfile,
    CapabilitySnapshot,
    LockIdentity,
    ToolIdentity,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
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
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    PhaseOutcome,
    TestExecutionKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    PROOF_OBLIGATION_NOT_APPLICABLE_CID,
    CacheReuseDecision,
    CacheReuseDisposition,
    CounterexampleReceipt,
    DirectExecutionObservation,
    ModelRoute,
    ModelRouteDecision,
    ProofReceipt,
    StaticAnalysisReceipt,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationBoundsError,
    VerificationBundle,
    VerificationCommitment,
    VerificationContractError,
    VerificationIdentityCompiler,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceipt,
    VerificationReceiptKey,
    VerificationReceiptKind,
    VerificationSummary,
    aggregate_terminal_status,
    build_verification_commitment,
)

TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
SEMANTIC_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)
OBLIGATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/verification-proof-obligation@1"
)


def _structured_cid(schema: str, value: object) -> str:
    return content_identity({"schema": schema, "value": value})


def _artifact(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


def _repository_forest(
    *,
    commit: str = "abcdef0123456789abcdef0123456789abcdef01",
    tree: str = "0123456789abcdef0123456789abcdef01234567",
) -> RepositoryForest:
    alias = "ipfs_accelerate_py"
    descriptor = RepositoryDescriptor(
        identity=RepositoryIdentity(logical_name=alias),
        portable_closure=PortableGitClosure(commit=commit, tree=tree),
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


def _expected_environment(values: dict[str, object]) -> dict[str, object]:
    snapshot = values["capability_snapshot"]
    assert isinstance(snapshot, CapabilitySnapshot)
    tool_identity = values["tool_identity"]
    assert isinstance(tool_identity, ToolIdentity)
    capability_name = values["tool_capability_name"]
    assert isinstance(capability_name, str)
    executable_sha256 = snapshot.tool_identities[capability_name]
    return {
        **values["observed_environment"],  # type: ignore[dict-item]
        "network_policy": values["network_policy"],
        "tool_name": values["tool_name"],
        "tool_version": values["tool_version"],
        "tool_capability_name": capability_name,
        "tool_launcher_identity": tool_identity.to_dict(),
        "resolved_tool_executable": values["resolved_tool_executable"],
        "tool_executable_sha256": executable_sha256,
        "tool_executable_cid": _structured_cid(
            TOOL_EXECUTABLE_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
        ),
        "tool_version_probe_argv": values["tool_version_probe_argv"],
        "tool_version_probe_output_cid": cid_for_bytes(
            values["tool_version_probe_output_bytes"]  # type: ignore[arg-type]
        ),
        "tool_inventory_schema": "observed-tool-inventory@1",
        "adapter_schema": values["adapter_schema"],
        "capability_environment_names": tuple(sorted(snapshot.environment_names)),
        "capability_read_paths": tuple(sorted(snapshot.read_paths)),
        "capability_write_paths": tuple(sorted(snapshot.write_paths)),
        "capability_lock_identities": dict(sorted(snapshot.lock_identities.items())),
        "selected_dependency_lock_path": values["dependency_lock_path"],
        "selected_dependency_lock_identity": values[
            "dependency_lock_identity"
        ].to_dict(),  # type: ignore[union-attr]
    }


def _compiler_kwargs(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
) -> dict[str, object]:
    tool_name, tool_version, selector_argv, adapter_schema = {
        VerificationReceiptKind.STATIC_ANALYSIS: (
            "ruff",
            "0.12.11",
            ("/usr/bin/ruff", "check", "src/example.py"),
            "ruff-verification-adapter@1",
        ),
        VerificationReceiptKind.TYPE_CHECK: (
            "mypy",
            "1.18.2",
            ("/usr/bin/python3.12", "-m", "mypy", "src/example.py"),
            "mypy-verification-adapter@1",
        ),
        VerificationReceiptKind.TEST: (
            "pytest",
            "9.1.1",
            ("/usr/bin/python3.12", "-m", "pytest", "src/example.py"),
            "pytest-verification-adapter@1",
        ),
        VerificationReceiptKind.PROOF: (
            "z3",
            "4.13.3",
            ("/usr/bin/z3", "-smt2", "obligation.smt2"),
            "z3-verification-adapter@1",
        ),
    }[kind]
    repository_forest = _repository_forest()
    descriptor = repository_forest.write_descriptor()
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
    sandbox_environment = {
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
            "entries": ("mypy==1.18.2",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }
    capability_name = "verification-tool"
    executable_bytes = ("reviewed-launcher:" + tool_name).encode()
    executable_sha256 = "sha256:" + hashlib.sha256(executable_bytes).hexdigest()
    dependency_lock_bytes = b"package==1.2.3 --hash=sha256:abcd\n"
    dependency_lock_path = "requirements.lock"
    dependency_lock_identity = LockIdentity(
        path=dependency_lock_path,
        identity="sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest(),
    )
    capability_snapshot = CapabilitySnapshot(
        tool_identities={capability_name: executable_sha256},
        lock_identities={
            dependency_lock_path: (
                "sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest()
            )
        },
        environment_names=("LANG", "LC_ALL"),
        read_paths=("/workspace/source",),
        write_paths=("/workspace/artifacts",),
    )
    tool_identity = ToolIdentity(
        name=capability_name,
        kind="executable",
        locator=selector_argv[0].rsplit("/", 1)[-1],
        version="launcher-fixture-1",
        identity=executable_sha256,
        roles=("verification",),
    )
    invocation_prefix = (
        selector_argv[:3]
        if len(selector_argv) >= 3 and selector_argv[1] == "-m"
        else selector_argv[:1]
    )
    version_probe_argv = (*invocation_prefix, "--version")
    version_probe_output = f"{tool_name} {tool_version}\n".encode()
    proof_obligation = None
    proof_backend_binding = None
    if kind is VerificationReceiptKind.PROOF:
        proof_obligation = CodeProofObligation(
            repository_id=descriptor.repository_id,
            repository_tree_id=descriptor.tree,
            ast_scope_ids=("scope:example.calculate",),
            statement="not (x >= 0 implies result >= 0)",
            premise_ids=("premise:contract",),
            template_id="python-contract-to-smtlib2",
            template_version="1.2.0",
            template_semantic_hash="sha256:" + "a" * 64,
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
        )
        proof_backend_binding = {
            "plan_id": _artifact("formal-plan"),
            "step_id": "proof-step:z3",
            "attempt_stage": ProofStage.SOLVE.value,
            "attempt_provider_id": "provider:fixture",
            "provider_id": "provider:fixture",
            "repository_tree_identity_kind": "git_tree",
            "repository_tree_identity": descriptor.tree,
            "translator_id": "translator:python-smt@1",
            "solver_id": "solver:z3@4.13.3",
            "kernel_id": "kernel:reviewed-z3-result@1",
            "toolchain_id": "toolchain:locked@1",
            "policy_id": "policy:proof@1",
            "theorem_registry_id": "registry:fixture@1",
            "ast_scope_ids": ("scope:example.calculate",),
            "premise_ids": ("premise:contract",),
            "tool_name": tool_name,
            "tool_version": tool_version,
            "tool_executable_cid": _structured_cid(
                TOOL_EXECUTABLE_SCHEMA,
                {
                    "capability_name": capability_name,
                    "sha256": executable_sha256,
                },
            ),
        }
    values: dict[str, object] = {
        "repository_forest": repository_forest,
        "repository_alias": repository_forest.sole_write_alias,
        "claimed_repository_tree_cid": _structured_cid(
            TREE_SCHEMA, tree_observation
        ),
        "patch_base_tree_id": "git-tree:base",
        "repository_state_tree_id": "git-tree:base",
        "invalidation_plan_tree_id": "git-tree:base",
        "context_pack_tree_id": "git-tree:base",
        "observed_semantic_state": semantic,
        "repository_state_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "invalidation_plan_semantic_root_cid": _structured_cid(
            SEMANTIC_SCHEMA, semantic
        ),
        "context_pack_semantic_root_cid": _structured_cid(SEMANTIC_SCHEMA, semantic),
        "affected_symbol_versions": (
            {
                "symbol": "example.calculate",
                "version": 2,
                "source_cid": _artifact("source-v2"),
            },
        ),
        "observed_environment": sandbox_environment,
        "capability_snapshot": capability_snapshot,
        "tool_capability_name": capability_name,
        "tool_identity": tool_identity,
        "resolved_tool_executable": selector_argv[0],
        "tool_executable_bytes": executable_bytes,
        "tool_version_probe_argv": version_probe_argv,
        "tool_version_probe_output_bytes": version_probe_output,
        "claimed_environment_cid": "",
        "dependency_lock_path": dependency_lock_path,
        "dependency_lock_identity": dependency_lock_identity,
        "dependency_lock_bytes": dependency_lock_bytes,
        "selector_argv": selector_argv,
        "proof_obligation": proof_obligation,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "configuration_bytes": b"[tool]\nstrict = true\n",
        "fixture_data_bytes": (b"fixture-one\n", b"fixture-two\n"),
        "network_policy": "deny_all",
        "receipt_schema_version": 1,
        "receipt_kind": kind,
        "adapter_schema": adapter_schema,
        "proof_backend_binding": proof_backend_binding,
    }
    values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(values),
    )
    return values


def _key(
    kind: VerificationReceiptKind = VerificationReceiptKind.TYPE_CHECK,
    **changes: object,
) -> VerificationReceiptKey:
    values = _compiler_kwargs(kind)
    values.update(changes)
    if "dependency_lock_bytes" in changes and "capability_snapshot" not in changes:
        snapshot = values["capability_snapshot"]
        assert isinstance(snapshot, CapabilitySnapshot)
        lock_bytes = values["dependency_lock_bytes"]
        assert isinstance(lock_bytes, bytes)
        lock_path = values["dependency_lock_path"]
        assert isinstance(lock_path, str)
        values["capability_snapshot"] = replace(
            snapshot,
            lock_identities={
                **dict(snapshot.lock_identities),
                lock_path: "sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
            },
        )
        if "dependency_lock_identity" not in changes:
            values["dependency_lock_identity"] = LockIdentity(
                path=lock_path,
                identity="sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
            )
        if "claimed_environment_cid" not in changes:
            values["claimed_environment_cid"] = _structured_cid(
                ENVIRONMENT_SCHEMA,
                _expected_environment(values),
            )
    if {"tool_name", "tool_version", "adapter_schema"}.intersection(
        changes
    ) and "claimed_environment_cid" not in changes:
        values["tool_version_probe_output_bytes"] = (
            f"{values['tool_name']} {values['tool_version']}\n".encode()
        )
        values["claimed_environment_cid"] = _structured_cid(
            ENVIRONMENT_SCHEMA,
            _expected_environment(values),
        )
    return VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def _observation(
    key: VerificationReceiptKey,
    status: TerminalStatus = TerminalStatus.PASSED,
    *,
    label: str = "run",
    repository_tree_observation: object | None = None,
    environment_observation: object | None = None,
    additional_artifact_cids: tuple[str, ...] = (),
    command_argv: tuple[str, ...] | None = None,
) -> DirectExecutionObservation:
    default_command_argv = {
        VerificationReceiptKind.STATIC_ANALYSIS: (
            "/usr/bin/ruff",
            "check",
            "src/example.py",
        ),
        VerificationReceiptKind.TYPE_CHECK: (
            "/usr/bin/python3.12",
            "-m",
            "mypy",
            "src/example.py",
        ),
        VerificationReceiptKind.TEST: (
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "src/example.py",
        ),
        VerificationReceiptKind.PROOF: (
            "/usr/bin/z3",
            "-smt2",
            "obligation.smt2",
        ),
    }[key.receipt_kind]
    environment = dict(key.environment_observation)
    return DirectExecutionObservation(
        receipt_key_cid=key.key_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        repository_tree_observation=(
            repository_tree_observation
            if repository_tree_observation is not None
            else key.repository_tree_observation
        ),
        environment_observation=(
            environment_observation
            if environment_observation is not None
            else environment
        ),
        terminal_status=status,
        command_argv=(
            command_argv if command_argv is not None else default_command_argv
        ),
        duration_ms=125,
        exit_code=(
            0
            if status
            in {
                TerminalStatus.PASSED,
                TerminalStatus.PROVED,
                TerminalStatus.DISPROVED,
            }
            else 1
        ),
        stdout_artifact_cid=_artifact(f"{label}-stdout"),
        stderr_artifact_cid=_artifact(f"{label}-stderr"),
        artifact_cids=(_artifact(f"{label}-report"), *additional_artifact_cids),
        reason_codes=(f"{label}_observed",),
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
        model_token_limit=4_096,
        provider_quota=1,
        network_allowed=False,
    )


def _formal_result(
    key: VerificationReceiptKey,
    evidence: tuple[ProofEvidence, ...],
    *,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> tuple[ProofAttempt, FormalProofReceipt]:
    backend = key.proof_backend_binding
    assert backend is not None
    attempt = ProofAttempt(
        plan_id=backend["plan_id"],
        step_id=backend["step_id"],
        obligation_id=key.proof_obligation_cid,
        repository_tree_id=backend["repository_tree_identity"],
        provider_id=backend["provider_id"],
        stage=ProofStage(backend["attempt_stage"]),
        status=AttemptStatus.SUCCEEDED,
        evidence=evidence,
        input_ids=(key.key_id,),
        output_ids=tuple(item.evidence_id for item in evidence),
    )
    formal = FormalProofReceipt(
        obligation_id=key.proof_obligation_cid,
        plan_id=backend["plan_id"],
        attempt_id=attempt.attempt_id,
        repository_id=backend["repository_id"],
        repository_tree_id=backend["repository_tree_identity"],
        ast_scope_ids=backend["ast_scope_ids"],
        premise_ids=backend["premise_ids"],
        translator_id=backend["translator_id"],
        solver_id=backend["solver_id"],
        kernel_id=backend["kernel_id"],
        toolchain_id=backend["toolchain_id"],
        theorem_registry_id=backend["theorem_registry_id"],
        policy_id=backend["policy_id"],
        resource_budget=_budget(),
        verdict=verdict,
        evidence=evidence,
        provider_id=backend["provider_id"],
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=freshness,
    )
    return attempt, formal


def _solver_evidence(
    key: VerificationReceiptKey,
    *,
    accepted: bool = True,
    simulated: bool = False,
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED if accepted else EvidenceVerdict.REJECTED,
        artifact_id=_artifact("solver-result"),
        subject_id=key.proof_obligation_cid,
        verifier_id="solver:z3@4.13.3",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=simulated,
        metadata={"counterexample_verified": not accepted},
    )


def _route(*, human: bool = False) -> ModelRouteDecision:
    route = ModelRoute.HUMAN_REVIEW_REQUIRED if human else ModelRoute.SMALL_LOCAL_MODEL
    return ModelRouteDecision(
        route=route,
        considered_routes=(ModelRoute.SMALL_LOCAL_MODEL, route) if human else (route,),
        decisive_reason_codes=(
            "unresolved_authority" if human else "localized_exact_counterexample",
        ),
        required_capabilities=("bounded_context",),
        context_token_estimate=2_048,
        policy_cid=_artifact("route-policy"),
    )


def test_terminal_status_vocabulary_is_exact_and_closed() -> None:
    expected = {
        "passed",
        "failed",
        "proved",
        "disproved",
        "unknown",
        "timeout",
        "unavailable",
        "not_modeled",
        "stale",
        "invalid",
        "cancelled",
        "simulated",
    }
    assert {item.value for item in TerminalStatus} == expected
    assert all(item.terminal for item in TerminalStatus)
    assert {item for item in TerminalStatus if item.successful} == {
        TerminalStatus.PASSED,
        TerminalStatus.PROVED,
    }
    with pytest.raises(ValueError):
        TerminalStatus("PASSED")
    with pytest.raises(ValueError):
        TerminalStatus("timed_out")


def test_compiler_binds_exact_observed_target_separately_from_equal_base_roots() -> (
    None
):
    values = _compiler_kwargs()
    key = VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    assert values["patch_base_tree_id"] == values["repository_state_tree_id"]
    assert values["patch_base_tree_id"] == values["invalidation_plan_tree_id"]
    assert values["patch_base_tree_id"] == values["context_pack_tree_id"]
    assert key.repository_tree_cid == values["claimed_repository_tree_cid"]
    assert key.repository_tree_cid != values["patch_base_tree_id"]
    assert key.proof_obligation_cid == PROOF_OBLIGATION_NOT_APPLICABLE_CID
    assert VerificationReceiptKey.from_dict(key.to_record()) == key


@pytest.mark.parametrize(
    "field",
    (
        "repository_state_tree_id",
        "invalidation_plan_tree_id",
        "context_pack_tree_id",
    ),
)
def test_compiler_rejects_any_base_root_mismatch(field: str) -> None:
    values = _compiler_kwargs()
    values[field] = "git-tree:different"
    with pytest.raises(VerificationIdentityError, match="base trees disagree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    (
        "repository_state_semantic_root_cid",
        "invalidation_plan_semantic_root_cid",
        "context_pack_semantic_root_cid",
    ),
)
def test_compiler_rejects_any_semantic_root_mismatch(field: str) -> None:
    values = _compiler_kwargs()
    values[field] = _artifact("wrong-semantic-root")
    with pytest.raises(VerificationIdentityError, match="semantic roots disagree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_compiler_rejects_target_and_environment_claim_mismatch() -> None:
    values = _compiler_kwargs()
    values["claimed_repository_tree_cid"] = _artifact("wrong-tree")
    with pytest.raises(VerificationIdentityError, match="patched tree"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["claimed_environment_cid"] = _artifact("wrong-environment")
    with pytest.raises(VerificationIdentityError, match="effective environment"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_compiler_rejects_environment_network_policy_mismatch() -> None:
    values = _compiler_kwargs()
    values["capability_snapshot"] = CapabilitySnapshot(
        tool_identities=dict(
            values["capability_snapshot"].tool_identities  # type: ignore[union-attr]
        ),
        network_enabled=True,
    )
    with pytest.raises(VerificationIdentityError, match="hermetic"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_compiler_requires_closed_repository_and_observed_tool_bindings() -> None:
    values = _compiler_kwargs()
    values["repository_forest"] = {"self_asserted": True}
    with pytest.raises(VerificationContractError, match="RepositoryForest"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["repository_alias"] = "unbound_repository"
    with pytest.raises(VerificationIdentityError, match="descriptor binding"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["observed_environment"] = {"opaque": "caller-value"}
    with pytest.raises(VerificationContractError, match="sandbox field set"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["tool_version"] = "caller-claimed-version"
    with pytest.raises(VerificationIdentityError, match="probe output"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["capability_snapshot"] = CapabilitySnapshot(
        unavailable_tools=(str(values["tool_capability_name"]),),
    )
    with pytest.raises(VerificationIdentityError, match="unavailable"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs(VerificationReceiptKind.TEST)
    values["tool_name"] = "mypy"
    values["tool_version"] = "1.18.2"
    values["tool_version_probe_output_bytes"] = b"mypy 1.18.2\n"
    values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(values),
    )
    with pytest.raises(VerificationIdentityError, match="selected Python module"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_observed_executable_and_version_probe_change_exact_key_identity() -> None:
    baseline = _key()
    values = _compiler_kwargs()
    capability_name = str(values["tool_capability_name"])
    changed_executable = b"reviewed-launcher:mypy:changed"
    changed_sha256 = "sha256:" + hashlib.sha256(changed_executable).hexdigest()
    original_snapshot = values["capability_snapshot"]
    assert isinstance(original_snapshot, CapabilitySnapshot)
    values["capability_snapshot"] = replace(
        original_snapshot,
        tool_identities={capability_name: changed_sha256},
    )
    values["tool_identity"] = replace(
        values["tool_identity"],  # type: ignore[arg-type]
        identity=changed_sha256,
    )
    values["tool_executable_bytes"] = changed_executable
    values["tool_version_probe_output_bytes"] = b"mypy 1.18.2 changed-probe-output\n"
    values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(values),
    )
    changed = VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]
    assert changed.environment_cid != baseline.environment_cid
    assert changed.key_id != baseline.key_id

    mismatched = _compiler_kwargs()
    mismatched["tool_executable_bytes"] = changed_executable
    with pytest.raises(VerificationIdentityError, match="executable bytes"):
        VerificationIdentityCompiler().compile_key(**mismatched)  # type: ignore[arg-type]

    mismatched = _compiler_kwargs()
    mismatched["dependency_lock_bytes"] = b"unobserved dependency lock\n"
    with pytest.raises(VerificationIdentityError, match="dependency lock bytes"):
        VerificationIdentityCompiler().compile_key(**mismatched)  # type: ignore[arg-type]

    mismatched = _compiler_kwargs()
    mismatched["dependency_lock_identity"] = LockIdentity(
        path="requirements.lock",
        identity="sha256:" + hashlib.sha256(b"reviewed-other-lock").hexdigest(),
    )
    with pytest.raises(VerificationIdentityError, match="reviewed lock identity"):
        VerificationIdentityCompiler().compile_key(**mismatched)  # type: ignore[arg-type]


def test_live_git_forest_and_capability_observations_compile_exact_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    for argv in (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Fixture"),
        ("git", "config", "user.email", "fixture@example.invalid"),
    ):
        subprocess.run(argv, cwd=repository, check=True)
    (repository / "README.md").write_text("# fixture\n", encoding="utf-8")
    lock_bytes = b"locked\n"
    (repository / "fixture.lock").write_bytes(lock_bytes)
    subprocess.run(
        ("git", "add", "README.md", "fixture.lock"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "commit", "-qm", "fixture"), cwd=repository, check=True
    )

    forest = build_repository_forest(
        ForestPolicy(
            roots=(
                ForestRootSpec(
                    alias="repo",
                    root_path=repository,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    ),
                ),
            ),
            sole_write_alias="repo",
        )
    )
    replayed = replay_repository_forest(freeze_repository_forest(forest))
    assert replayed.forest_id == forest.forest_id
    descriptor = replayed.write_descriptor()

    executable = tmp_path / "fixture-checker"
    executable_bytes = b"#!/bin/sh\nprintf 'fixture-checker 1.2.3\\n'\n"
    executable.write_bytes(executable_bytes)
    executable.chmod(0o755)
    executable_sha256 = "sha256:" + hashlib.sha256(executable_bytes).hexdigest()
    tool_identity = ToolIdentity(
        name="fixture-tool",
        kind="executable",
        locator=str(executable),
        version="1.2.3",
        identity=executable_sha256,
        roles=(
            "verification",
            "node",
            "parser",
            "proof",
            "python",
            "solver",
            "typescript",
        ),
    )
    profile_path = (
        Path(__file__).resolve().parents[2]
        / "data/datasets_contract_analysis/policy/analyzer-profile-v1.json"
    )
    base_profile = AnalysisExecutionProfile.load(profile_path)
    live_lock_identity = LockIdentity(
        path="fixture.lock",
        identity="sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
    )
    profile = replace(
        base_profile,
        tools=(tool_identity,),
        locks=(live_lock_identity,),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.contract_analysis."
        "execution_profile.shutil.which",
        lambda locator: str(executable) if locator == str(executable) else None,
    )
    environment_values = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    artifacts = repository / "data/datasets_contract_analysis/runtime"
    artifacts.mkdir(parents=True)
    snapshot = CapabilitySnapshot.observe(
        profile,
        repository_root=repository,
        environment=environment_values,
        read_paths=(str(repository),),
        write_paths=(str(artifacts),),
    )
    assert snapshot.tool_identities["fixture-tool"] == executable_sha256
    assert profile.validate(snapshot, repository_root=repository).ok
    probe_argv = (str(executable), "--version")
    probe_output = subprocess.run(
        probe_argv,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout

    values = _compiler_kwargs()
    base_tree = descriptor.tree
    tree_observation = {
        "repository_forest_cid": forest.forest_id,
        "git_commit_id": descriptor.commit,
        "git_tree_id": descriptor.tree,
        "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
        "dirty_overlay_cid": descriptor.dirty_overlay_digest,
        "dirty": descriptor.dirty,
        "repository_alias": descriptor.alias,
        "repository_id": descriptor.repository_id,
        "descriptor_cid": descriptor.descriptor_cid,
        "base_repository_tree_id": base_tree,
    }
    live_sandbox = dict(values["observed_environment"])  # type: ignore[arg-type]
    live_sandbox["interpreter"] = {
        "schema": "verification-interpreter@1",
        "implementation": "posix-script",
        "version": "fixture-1",
        "abi": "none",
    }
    live_sandbox["toolchain"] = {
        "schema": "verification-toolchain@1",
        "name": "fixture-checker",
        "revision": "1.2.3",
    }
    live_sandbox["dependency_distribution"] = {
        "schema": "verification-dependency-distribution@1",
        "entries": (),
    }
    values.update(
        repository_forest=forest,
        repository_alias="repo",
        claimed_repository_tree_cid=_structured_cid(TREE_SCHEMA, tree_observation),
        patch_base_tree_id=base_tree,
        repository_state_tree_id=base_tree,
        invalidation_plan_tree_id=base_tree,
        context_pack_tree_id=base_tree,
        observed_environment=live_sandbox,
        capability_snapshot=snapshot,
        tool_capability_name="fixture-tool",
        tool_identity=tool_identity,
        resolved_tool_executable=str(executable),
        tool_executable_bytes=executable_bytes,
        selector_argv=(str(executable), "check", "README.md"),
        tool_version_probe_argv=probe_argv,
        tool_version_probe_output_bytes=probe_output,
        tool_name="fixture-checker",
        tool_version="1.2.3",
        adapter_schema="fixture-checker-adapter@1",
        dependency_lock_path="fixture.lock",
        dependency_lock_identity=live_lock_identity,
        dependency_lock_bytes=lock_bytes,
    )
    values["claimed_environment_cid"] = _structured_cid(
        ENVIRONMENT_SCHEMA,
        _expected_environment(values),
    )
    key = VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]
    assert key.repository_tree_observation["repository_forest_cid"] == forest.forest_id
    assert key.environment_observation["tool_executable_sha256"] == executable_sha256

    mismatched = _compiler_kwargs()
    mismatched["resolved_tool_executable"] = "/opt/unreviewed/python3.12"
    with pytest.raises(VerificationIdentityError, match="resolved executable"):
        VerificationIdentityCompiler().compile_key(**mismatched)  # type: ignore[arg-type]


def test_every_key_input_mutation_changes_identity() -> None:
    base = _key()
    mutations: list[VerificationReceiptKey] = []

    changed_forest = _repository_forest(
        commit="fedcba9876543210fedcba9876543210fedcba98",
        tree="fedcba9876543210fedcba9876543210fedcba98",
    )
    changed_descriptor = changed_forest.write_descriptor()
    tree = {
        "repository_forest_cid": changed_forest.forest_id,
        "git_commit_id": changed_descriptor.commit,
        "git_tree_id": changed_descriptor.tree,
        "gitlink_state_cid": (
            changed_descriptor.portable_closure.gitlink_closure_cid
        ),
        "dirty_overlay_cid": changed_descriptor.dirty_overlay_digest,
        "dirty": changed_descriptor.dirty,
        "repository_alias": changed_descriptor.alias,
        "repository_id": changed_descriptor.repository_id,
        "descriptor_cid": changed_descriptor.descriptor_cid,
        "base_repository_tree_id": "git-tree:base",
    }
    mutations.append(
        _key(
            repository_forest=changed_forest,
            claimed_repository_tree_cid=_structured_cid(TREE_SCHEMA, tree),
        )
    )
    semantic = {"symbols": ["different"], "edge_root": "different"}
    semantic_cid = _structured_cid(SEMANTIC_SCHEMA, semantic)
    mutations.append(
        _key(
            observed_semantic_state=semantic,
            repository_state_semantic_root_cid=semantic_cid,
            invalidation_plan_semantic_root_cid=semantic_cid,
            context_pack_semantic_root_cid=semantic_cid,
        )
    )
    mutations.append(
        _key(
            affected_symbol_versions=(
                {"symbol": "other", "version": 9, "source_cid": _artifact("other")},
            )
        )
    )
    environment_values = _compiler_kwargs()
    environment = {
        **environment_values["observed_environment"],  # type: ignore[dict-item]
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "aarch64",
            "libc": "glibc-2.39",
        },
    }
    environment_values["observed_environment"] = environment
    mutations.append(
        _key(
            observed_environment=environment,
            claimed_environment_cid=_structured_cid(
                ENVIRONMENT_SCHEMA,
                _expected_environment(environment_values),
            ),
        )
    )
    capability_values = _compiler_kwargs()
    snapshot = capability_values["capability_snapshot"]
    assert isinstance(snapshot, CapabilitySnapshot)
    changed_snapshot = replace(
        snapshot,
        read_paths=("/workspace/other-source",),
        write_paths=("/workspace/other-artifacts",),
    )
    capability_values["capability_snapshot"] = changed_snapshot
    mutations.append(
        _key(
            capability_snapshot=changed_snapshot,
            claimed_environment_cid=_structured_cid(
                ENVIRONMENT_SCHEMA,
                _expected_environment(capability_values),
            ),
        )
    )
    mutations.extend(
        (
            _key(dependency_lock_bytes=b"different lock"),
            _key(selector_argv=("/usr/bin/python3.12", "-m", "mypy", "other.py")),
            _key(
                tool_name="pyright",
                tool_version="1.1.407",
                selector_argv=(
                    "/usr/bin/python3.12",
                    "-m",
                    "pyright",
                    "src/example.py",
                ),
                tool_version_probe_argv=(
                    "/usr/bin/python3.12",
                    "-m",
                    "pyright",
                    "--version",
                ),
                adapter_schema="pyright-verification-adapter@1",
            ),
            _key(tool_version="1.19.0"),
            _key(configuration_bytes=b"different config"),
            _key(fixture_data_bytes=(b"different fixture",)),
            _key(
                network_policy="loopback_only",
                claimed_environment_cid=_structured_cid(
                    ENVIRONMENT_SCHEMA,
                    _expected_environment(
                        {
                            **_compiler_kwargs(),
                            "network_policy": "loopback_only",
                        }
                    ),
                ),
            ),
            _key(receipt_schema_version=2),
            _key(adapter_schema="mypy-verification-adapter@2"),
        )
    )

    assert len({item.key_id for item in mutations}) == len(mutations)
    assert all(item.key_id != base.key_id for item in mutations)

    proof_a = _key(VerificationReceiptKind.PROOF)
    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    obligation = values["proof_obligation"]
    assert isinstance(obligation, CodeProofObligation)
    obligation = replace(obligation, statement="alternate normalized obligation")
    proof_b = _key(VerificationReceiptKind.PROOF, proof_obligation=obligation)
    assert proof_b.proof_obligation_cid != proof_a.proof_obligation_cid
    assert proof_b.key_id != proof_a.key_id


def test_key_canonicalizes_set_like_inputs_but_preserves_selector_order() -> None:
    values = _compiler_kwargs()
    symbols = values["affected_symbol_versions"]
    values["affected_symbol_versions"] = tuple(reversed(symbols))  # type: ignore[arg-type]
    fixtures = values["fixture_data_bytes"]
    values["fixture_data_bytes"] = tuple(reversed(fixtures))  # type: ignore[arg-type]
    assert VerificationIdentityCompiler().compile_key(**values).key_id == _key().key_id  # type: ignore[arg-type]

    values = _compiler_kwargs()
    selector = values["selector_argv"]
    values["selector_argv"] = (*selector, "--strict")  # type: ignore[misc]
    ordered = VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]
    values["selector_argv"] = (*selector[:3], "--strict", *selector[3:])  # type: ignore[index]
    assert VerificationIdentityCompiler().compile_key(**values).key_id != ordered.key_id  # type: ignore[arg-type]


def test_proof_applicability_and_translation_are_fail_closed() -> None:
    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    values["proof_obligation"] = None
    with pytest.raises(VerificationIdentityError, match="CodeProofObligation"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    values["proof_obligation"] = {"schema": "wrong@1", "contract_version": 1}
    with pytest.raises(VerificationContractError, match="exact schema"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    values = _compiler_kwargs()
    values["proof_obligation"] = _compiler_kwargs(VerificationReceiptKind.PROOF)[
        "proof_obligation"
    ]
    with pytest.raises(VerificationIdentityError, match="non-proof"):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_wrong_schema_interface_version_unknown_field_and_forged_id_reject() -> None:
    key = _key()
    for field, value in (
        ("schema", "wrong@1"),
        ("interface", "Wrong@1"),
        ("contract_version", 2),
        ("contract_version", True),
    ):
        payload = key.to_record()
        payload[field] = value
        with pytest.raises(VerificationContractError):
            VerificationReceiptKey.from_dict(payload)

    payload = key.to_record()
    payload["unexpected"] = "field"
    with pytest.raises(VerificationContractError, match="unsupported fields"):
        VerificationReceiptKey.from_dict(payload)

    payload = key.to_record()
    payload["key_id"] = _artifact("forged")
    with pytest.raises(VerificationIdentityError, match="does not match"):
        VerificationReceiptKey.from_dict(payload)


@pytest.mark.parametrize(
    "bad_value",
    (
        {"ratio": 0.5},
        {"secret": "do-not-hash"},
        {"nested": {"private_witness": "proof"}},
        {"authorization": "Bearer token"},
    ),
)
def test_identity_inputs_reject_floats_secrets_and_witnesses(
    bad_value: dict[str, object],
) -> None:
    values = _compiler_kwargs()
    values["observed_semantic_state"] = bad_value
    values["repository_state_semantic_root_cid"] = _artifact("claim")
    values["invalidation_plan_semantic_root_cid"] = _artifact("claim")
    values["context_pack_semantic_root_cid"] = _artifact("claim")
    with pytest.raises(VerificationContractError):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]


def test_oversized_identity_bytes_and_text_reject() -> None:
    values = _compiler_kwargs()
    values["dependency_lock_bytes"] = b"x" * (16 * 1_048_576 + 1)
    with pytest.raises(VerificationBoundsError):
        VerificationIdentityCompiler().compile_key(**values)  # type: ignore[arg-type]

    with pytest.raises(VerificationBoundsError):
        replace(_key(), tool_version="x" * 9_000)


@pytest.mark.parametrize(
    "status",
    (
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
        TerminalStatus.CANCELLED,
        TerminalStatus.INVALID,
    ),
)
def test_direct_receipts_preserve_nonaccepting_terminal_status(
    status: TerminalStatus,
) -> None:
    key = _key()
    receipt = TypeCheckReceipt(key=key, execution=_observation(key, status))
    assert receipt.status is status
    assert receipt.terminal_success is False
    assert TypeCheckReceipt.from_dict(receipt.to_record()) == receipt


def test_direct_execution_must_bind_key_tree_environment_and_argv() -> None:
    key = _key()
    other = _key(tool_version="other")
    with pytest.raises(VerificationIdentityError, match="receipt key"):
        TypeCheckReceipt(key=key, execution=_observation(other))

    observation = _observation(key)
    with pytest.raises(VerificationIdentityError, match="environment"):
        TypeCheckReceipt(
            key=key,
            execution=replace(observation, environment_cid=_artifact("other-env")),
        )

    with pytest.raises(VerificationContractError, match="command_argv"):
        replace(observation, command_argv=())

    different_argv = replace(
        observation,
        command_argv=("/usr/bin/python3.12", "-m", "mypy", "other.py"),
    )
    with pytest.raises(VerificationIdentityError, match="selector"):
        TypeCheckReceipt(key=key, execution=different_argv)


def test_conclusive_direct_execution_requires_zero_exit_and_persisted_streams() -> None:
    observation = _observation(_key())
    for changes in (
        {"exit_code": 1},
        {"exit_code": None},
        {"stdout_artifact_cid": ""},
        {"stderr_artifact_cid": ""},
    ):
        with pytest.raises(VerificationContractError):
            replace(observation, **changes)


def test_selector_argv_preserves_whitespace_duplicates_and_empty_arguments() -> None:
    exact = (
        "/usr/bin/python3.12",
        "-m",
        "mypy",
        "--flag",
        "--flag",
        " value ",
        "",
    )
    key = _key(selector_argv=exact)
    receipt = TypeCheckReceipt(
        key,
        _observation(key, command_argv=exact),
    )
    assert receipt.terminal_success
    with pytest.raises(VerificationIdentityError, match="selector"):
        TypeCheckReceipt(
            key,
            _observation(
                key,
                command_argv=(
                    "/usr/bin/python3.12",
                    "-m",
                    "mypy",
                    "--flag",
                    " value ",
                    "",
                ),
            ),
        )


def test_nonproof_receipts_reject_proof_statuses() -> None:
    key = _key()
    with pytest.raises(VerificationContractError, match="non-proof"):
        TypeCheckReceipt(
            key=key,
            execution=_observation(key, TerminalStatus.PROVED),
        )


def test_test_pass_projection_comes_from_full_existing_receipt() -> None:
    key = _key(VerificationReceiptKind.TEST)
    observed_tree = key.repository_tree_observation
    source_key = TestExecutionKey(
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
    existing = TestPassReceipt(
        execution_key_cid=source_key.execution_key_id,
        locator_cid=source_key.locator_cid,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=source_key.static_trace_root_cid,
        runtime_trace_root_cid=source_key.runtime_trace_root_cid,
        completeness_receipt_cid=_artifact("pytest-runtime-completeness"),
        dependency_forest_cid=source_key.repository_forest_cid,
        policy_cid=source_key.policy_cid,
        admitted=True,
    )
    receipt = TestReceipt(
        key=key,
        execution=_observation(
            key,
            additional_artifact_cids=(source_key.execution_key_id, existing.receipt_id),
        ),
        test_pass_receipt=existing,
        test_execution_key=source_key,
    )
    assert receipt.status is TerminalStatus.PASSED
    assert receipt.terminal_success
    assert TestReceipt.from_dict(receipt.to_record()) == receipt

    non_reusable_key = replace(
        source_key,
        eligibility_class=EligibilityClass.NON_REUSABLE,
        components={
            **dict(source_key.components),
            "non_reusable_reason": "opaque_dynamic_dependency",
        },
    )
    non_reusable_pass = replace(
        existing,
        execution_key_cid=non_reusable_key.execution_key_id,
    )
    current_only = TestReceipt(
        key=key,
        execution=_observation(
            key,
            label="non-reusable-current-run",
            additional_artifact_cids=(
                non_reusable_key.execution_key_id,
                non_reusable_pass.receipt_id,
            ),
        ),
        test_pass_receipt=non_reusable_pass,
        test_execution_key=non_reusable_key,
    )
    assert current_only.status is TerminalStatus.PASSED
    with pytest.raises(VerificationContractError, match="non-reusable"):
        CacheReuseDecision(
            key_cid=key.key_id,
            disposition=CacheReuseDisposition.REUSED,
            reason_codes=("forged_non_reusable_test_hit",),
            candidate_receipt=current_only,
        )

    forged = receipt.to_record()
    forged["status"] = TerminalStatus.SIMULATED.value
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        TestReceipt.from_dict(forged)

    not_admitted = replace(existing, admitted=False)
    rejected = TestReceipt(
        key=key,
        execution=_observation(
            key,
            additional_artifact_cids=(
                source_key.execution_key_id,
                not_admitted.receipt_id,
            ),
        ),
        test_pass_receipt=not_admitted,
        test_execution_key=source_key,
    )
    assert rejected.status is TerminalStatus.INVALID
    assert not rejected.terminal_success

    distinct_domain_key = replace(
        source_key,
        environment_cid=_artifact("different-domain-env"),
        dependency_lock_cid=_artifact("different-domain-lock"),
        command_semantics_cid=_artifact("different-domain-command"),
        config_cid=_artifact("different-domain-config"),
        fixture_cids=(_artifact("different-domain-fixture"),),
        dirty_overlay_cid=_artifact("different-domain-dirty-overlay"),
    )
    distinct_domain_receipt = replace(
        existing,
        execution_key_cid=distinct_domain_key.execution_key_id,
    )
    assert TestReceipt(
        key=key,
        execution=_observation(
            key,
            additional_artifact_cids=(
                distinct_domain_key.execution_key_id,
                distinct_domain_receipt.receipt_id,
            ),
        ),
        test_pass_receipt=distinct_domain_receipt,
        test_execution_key=distinct_domain_key,
    ).terminal_success

    for field in (
        "dependency_forest_cid",
        "static_trace_root_cid",
        "runtime_trace_root_cid",
        "policy_cid",
    ):
        mismatched_receipt = replace(existing, **{field: _artifact(f"wrong-{field}")})
        with pytest.raises(VerificationIdentityError, match="forest, trace, and policy"):
            TestReceipt(
                key=key,
                execution=_observation(
                    key,
                    additional_artifact_cids=(
                        source_key.execution_key_id,
                        mismatched_receipt.receipt_id,
                    ),
                ),
                test_pass_receipt=mismatched_receipt,
                test_execution_key=source_key,
            )

    mismatched_tree_key = replace(
        source_key,
        git_tree_id="ffffffffffffffffffffffffffffffffffffffff",
    )
    mismatched_tree_receipt = replace(
        existing, execution_key_cid=mismatched_tree_key.execution_key_id
    )
    with pytest.raises(VerificationIdentityError, match="does not match"):
        TestReceipt(
            key=key,
            execution=_observation(
                key,
                additional_artifact_cids=(
                    mismatched_tree_key.execution_key_id,
                    mismatched_tree_receipt.receipt_id,
                ),
            ),
            test_pass_receipt=mismatched_tree_receipt,
            test_execution_key=mismatched_tree_key,
        )

    mismatched_descriptor_key = replace(
        source_key,
        components={"repository_descriptor": _artifact("foreign-descriptor")},
    )
    mismatched_descriptor_receipt = replace(
        existing,
        execution_key_cid=mismatched_descriptor_key.execution_key_id,
    )
    with pytest.raises(VerificationIdentityError, match="write descriptor"):
        TestReceipt(
            key=key,
            execution=_observation(
                key,
                additional_artifact_cids=(
                    mismatched_descriptor_key.execution_key_id,
                    mismatched_descriptor_receipt.receipt_id,
                ),
            ),
            test_pass_receipt=mismatched_descriptor_receipt,
            test_execution_key=mismatched_descriptor_key,
        )


def test_existing_test_bridge_requires_exact_upstream_headers() -> None:
    key = _key(VerificationReceiptKind.TEST)
    observed_tree = key.repository_tree_observation
    source_key = TestExecutionKey(
        locator_cid=_artifact("strict-test-locator"),
        repository_forest_cid=observed_tree["repository_forest_cid"],
        git_commit_id=observed_tree["git_commit_id"],
        git_tree_id=observed_tree["git_tree_id"],
        gitlink_state_cid=observed_tree["gitlink_state_cid"],
        dirty_overlay_cid=_artifact("strict-pytest-dirty-overlay"),
        fixture_cids=(_artifact("strict-pytest-fixture-closure"),),
        static_trace_root_cid=_artifact("strict-pytest-static-trace"),
        runtime_trace_root_cid=_artifact("strict-pytest-runtime-trace"),
        pytest_version=key.tool_version,
        command_semantics_cid=_artifact("strict-pytest-command-semantics"),
        config_cid=_artifact("strict-pytest-config"),
        dependency_lock_cid=_artifact("strict-pytest-lock-closure"),
        environment_cid=_artifact("strict-pytest-environment"),
        policy_cid=_artifact("strict-pytest-policy"),
        components={"repository_descriptor": observed_tree["descriptor_cid"]},
    )
    source_receipt = TestPassReceipt(
        execution_key_cid=source_key.execution_key_id,
        locator_cid=source_key.locator_cid,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=source_key.static_trace_root_cid,
        runtime_trace_root_cid=source_key.runtime_trace_root_cid,
        completeness_receipt_cid=_artifact("strict-pytest-completeness"),
        dependency_forest_cid=source_key.repository_forest_cid,
        policy_cid=source_key.policy_cid,
        admitted=True,
    )
    wrapped = TestReceipt(
        key,
        _observation(
            key,
            additional_artifact_cids=(
                source_key.execution_key_id,
                source_receipt.receipt_id,
            ),
        ),
        source_receipt,
        source_key,
    )
    for nested_name in ("test_execution_key", "test_pass_receipt"):
        for field, replacement in (
            ("schema", None),
            ("contract_version", None),
            ("contract_version", True),
            ("interface", None),
        ):
            payload = json.loads(json.dumps(wrapped.to_record()))
            nested = payload[nested_name]
            if replacement is None:
                nested.pop(field, None)
            else:
                nested[field] = replacement
            with pytest.raises(VerificationContractError, match="exact"):
                TestReceipt.from_dict(payload)


def test_existing_test_bridge_snapshots_mutable_upstream_mappings() -> None:
    key = _key(VerificationReceiptKind.TEST)
    observed_tree = key.repository_tree_observation
    source_components = {
        "repository_descriptor": observed_tree["descriptor_cid"],
        "test_module_cid": _artifact("mutable-module"),
    }
    source_key_metadata = {"nested": {"value": "before"}}
    source_receipt_metadata = {"nested": {"value": "before"}}
    source_key = TestExecutionKey(
        locator_cid=_artifact("snapshot-test-locator"),
        repository_forest_cid=observed_tree["repository_forest_cid"],
        git_commit_id=observed_tree["git_commit_id"],
        git_tree_id=observed_tree["git_tree_id"],
        gitlink_state_cid=observed_tree["gitlink_state_cid"],
        dirty_overlay_cid=_artifact("snapshot-pytest-dirty-overlay"),
        fixture_cids=(_artifact("snapshot-pytest-fixture-closure"),),
        static_trace_root_cid=_artifact("snapshot-pytest-static-trace"),
        runtime_trace_root_cid=_artifact("snapshot-pytest-runtime-trace"),
        pytest_version=key.tool_version,
        command_semantics_cid=_artifact("snapshot-pytest-command-semantics"),
        config_cid=_artifact("snapshot-pytest-config"),
        dependency_lock_cid=_artifact("snapshot-pytest-lock-closure"),
        environment_cid=_artifact("snapshot-pytest-environment"),
        policy_cid=_artifact("snapshot-pytest-policy"),
        components=source_components,
        metadata=source_key_metadata,
    )
    source_receipt = TestPassReceipt(
        execution_key_cid=source_key.execution_key_id,
        locator_cid=source_key.locator_cid,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=source_key.static_trace_root_cid,
        runtime_trace_root_cid=source_key.runtime_trace_root_cid,
        completeness_receipt_cid=_artifact("snapshot-pytest-completeness"),
        dependency_forest_cid=source_key.repository_forest_cid,
        policy_cid=source_key.policy_cid,
        admitted=True,
        metadata=source_receipt_metadata,
    )
    wrapped = TestReceipt(
        key,
        _observation(
            key,
            additional_artifact_cids=(
                source_key.execution_key_id,
                source_receipt.receipt_id,
            ),
        ),
        source_receipt,
        source_key,
    )
    before = wrapped.receipt_id
    source_components["test_module_cid"] = _artifact("mutated-module")
    source_key_metadata["nested"]["value"] = "after"
    source_receipt_metadata["nested"]["value"] = "after"
    assert wrapped.receipt_id == before
    assert wrapped.test_execution_key is not None
    assert wrapped.test_pass_receipt is not None
    with pytest.raises(TypeError):
        wrapped.test_execution_key.components["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        wrapped.test_pass_receipt.metadata["new"] = "mutation"  # type: ignore[index]


def test_receipt_success_is_not_an_independent_constructor_field() -> None:
    for receipt_type in (
        StaticAnalysisReceipt,
        TypeCheckReceipt,
        TestReceipt,
        ProofReceipt,
    ):
        assert "status" not in inspect.signature(receipt_type).parameters
        assert "passed" not in inspect.signature(receipt_type).parameters
        assert "proved" not in inspect.signature(receipt_type).parameters


def test_formal_proof_success_uses_existing_authoritative_assurance() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    attempt, formal = _formal_result(key, (_solver_evidence(key),))
    assert formal.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED
    receipt = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.PROVED,
            additional_artifact_cids=(attempt.attempt_id, formal.receipt_id),
        ),
        formal_proof_receipt=formal,
        proof_attempt=attempt,
    )
    assert receipt.status is TerminalStatus.PROVED
    assert receipt.terminal_success
    assert ProofReceipt.from_dict(receipt.to_record()) == receipt

    producer_receipt_only = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.PROVED,
            label="producer-receipt-only",
            additional_artifact_cids=(formal.receipt_id,),
        ),
        formal_proof_receipt=formal,
    )
    assert producer_receipt_only.status is TerminalStatus.PROVED
    assert producer_receipt_only.proof_attempt is None
    assert ProofReceipt.from_dict(producer_receipt_only.to_record()) == (
        producer_receipt_only
    )
    with pytest.raises(VerificationIdentityError, match="formal proof"):
        ProofReceipt(
            key=key,
            execution=_observation(
                key,
                TerminalStatus.PROVED,
                label="missing-formal-artifact",
            ),
            formal_proof_receipt=formal,
        )

    kernel_values = _compiler_kwargs(VerificationReceiptKind.PROOF)
    kernel_obligation = kernel_values["proof_obligation"]
    assert isinstance(kernel_obligation, CodeProofObligation)
    kernel_values["proof_obligation"] = replace(
        kernel_obligation,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    kernel_key = VerificationIdentityCompiler().compile_key(  # type: ignore[arg-type]
        **kernel_values
    )
    kernel_backend = kernel_key.proof_backend_binding
    assert kernel_backend is not None
    assert kernel_backend["required_assurance"] == "kernel_verified"
    solver_attempt, solver_formal = _formal_result(
        kernel_key,
        (_solver_evidence(kernel_key),),
    )
    insufficient = ProofReceipt(
        key=kernel_key,
        execution=_observation(
            kernel_key,
            TerminalStatus.UNKNOWN,
            label="kernel-required-solver-only",
            additional_artifact_cids=(
                solver_attempt.attempt_id,
                solver_formal.receipt_id,
            ),
        ),
        formal_proof_receipt=solver_formal,
        proof_attempt=solver_attempt,
    )
    assert insufficient.status is TerminalStatus.UNKNOWN
    assert not insufficient.terminal_success
    with pytest.raises(VerificationIdentityError, match="conflicts"):
        ProofReceipt(
            key=kernel_key,
            execution=_observation(
                kernel_key,
                TerminalStatus.PROVED,
                label="forged-kernel-assurance",
                additional_artifact_cids=(
                    solver_attempt.attempt_id,
                    solver_formal.receipt_id,
                ),
            ),
            formal_proof_receipt=solver_formal,
            proof_attempt=solver_attempt,
        )


def test_authoritative_disproof_dominates_declared_proof_and_accepted_evidence() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    attempt, conflicting = _formal_result(
        key,
        (
            _solver_evidence(key),
            _solver_evidence(key, accepted=False),
        ),
        verdict=ProofVerdict.PROVED,
    )
    assert conflicting.verdict is ProofVerdict.PROVED
    assert conflicting.authoritative_verdict is ProofVerdict.DISPROVED
    receipt = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.DISPROVED,
            label="conflicting-authoritative-evidence",
            additional_artifact_cids=(attempt.attempt_id, conflicting.receipt_id),
        ),
        formal_proof_receipt=conflicting,
        proof_attempt=attempt,
    )
    assert receipt.status is TerminalStatus.DISPROVED
    assert not receipt.terminal_success
    assert ProofReceipt.from_dict(receipt.to_record()) == receipt

    with pytest.raises(VerificationIdentityError, match="formal proof"):
        ProofReceipt(
            key=key,
            execution=_observation(
                key,
                TerminalStatus.PROVED,
                label="forged-conflicting-proof",
                additional_artifact_cids=(attempt.attempt_id, conflicting.receipt_id),
            ),
            formal_proof_receipt=conflicting,
            proof_attempt=attempt,
        )


@pytest.mark.parametrize(
    ("failure_code", "expected_status"),
    (
        ("binding_mismatch", TerminalStatus.INVALID),
        ("kernel_unavailable", TerminalStatus.UNAVAILABLE),
    ),
)
def test_authoritative_kernel_failure_dominates_weaker_solver_success(
    failure_code: str,
    expected_status: TerminalStatus,
) -> None:
    key = _key(VerificationReceiptKind.PROOF)
    backend = key.proof_backend_binding
    assert backend is not None
    kernel_failure = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ERROR,
        artifact_id=_artifact(f"kernel-{failure_code}"),
        subject_id=key.proof_obligation_cid,
        verifier_id=backend["kernel_id"],
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        metadata={"failure_code": failure_code},
    )
    attempt, conflicting = _formal_result(
        key,
        (_solver_evidence(key), kernel_failure),
        verdict=ProofVerdict.PROVED,
    )
    receipt = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            expected_status,
            label=f"authoritative-{failure_code}",
            additional_artifact_cids=(attempt.attempt_id, conflicting.receipt_id),
        ),
        formal_proof_receipt=conflicting,
        proof_attempt=attempt,
    )
    assert receipt.status is expected_status
    assert not receipt.terminal_success


def test_provider_claim_simulation_staleness_and_counterexample_project_safely() -> (
    None
):
    key = _key(VerificationReceiptKind.PROOF)
    provider_only = ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.PROVIDER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=_artifact("provider-candidate"),
        subject_id=key.proof_obligation_cid,
        verifier_id="provider:untrusted",
        freshness=EvidenceFreshness.CURRENT,
        independent=False,
    )
    claimed_attempt, claimed_formal = _formal_result(key, (provider_only,))
    claimed = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.UNKNOWN,
            additional_artifact_cids=(
                claimed_attempt.attempt_id,
                claimed_formal.receipt_id,
            ),
        ),
        formal_proof_receipt=claimed_formal,
        proof_attempt=claimed_attempt,
    )
    assert claimed.status is TerminalStatus.UNKNOWN
    assert not claimed.terminal_success

    simulated_attempt, simulated_formal = _formal_result(
        key, (_solver_evidence(key, simulated=True),)
    )
    simulated = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.SIMULATED,
            additional_artifact_cids=(
                simulated_attempt.attempt_id,
                simulated_formal.receipt_id,
            ),
        ),
        formal_proof_receipt=simulated_formal,
        proof_attempt=simulated_attempt,
    )
    assert simulated.status is TerminalStatus.SIMULATED
    assert not simulated.terminal_success

    stale_attempt, stale_formal = _formal_result(
        key,
        (_solver_evidence(key),),
        freshness=EvidenceFreshness.STALE,
    )
    stale = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.STALE,
            additional_artifact_cids=(
                stale_attempt.attempt_id,
                stale_formal.receipt_id,
            ),
        ),
        formal_proof_receipt=stale_formal,
        proof_attempt=stale_attempt,
    )
    assert stale.status is TerminalStatus.STALE

    disproved_attempt, disproved_formal = _formal_result(
        key,
        (_solver_evidence(key, accepted=False),),
        verdict=ProofVerdict.DISPROVED,
    )
    disproved = ProofReceipt(
        key=key,
        execution=_observation(
            key,
            TerminalStatus.DISPROVED,
            additional_artifact_cids=(
                disproved_attempt.attempt_id,
                disproved_formal.receipt_id,
            ),
        ),
        formal_proof_receipt=disproved_formal,
        proof_attempt=disproved_attempt,
    )
    assert disproved.status is TerminalStatus.DISPROVED
    assert not disproved.terminal_success


@pytest.mark.parametrize(
    "status",
    (
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
        TerminalStatus.CANCELLED,
        TerminalStatus.UNKNOWN,
    ),
)
def test_nonconclusive_execution_cannot_be_upgraded_by_formal_evidence(
    status: TerminalStatus,
) -> None:
    key = _key(VerificationReceiptKind.PROOF)
    attempt, formal = _formal_result(key, (_solver_evidence(key),))
    with pytest.raises(VerificationIdentityError, match="conflicts"):
        ProofReceipt(
            key,
            _observation(
                key,
                status,
                label=f"mixed-{status.value}",
                additional_artifact_cids=(attempt.attempt_id, formal.receipt_id),
            ),
            formal,
            attempt,
        )


def test_formal_bridge_binds_backend_and_exact_attempt() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    attempt, formal = _formal_result(key, (_solver_evidence(key),))

    formal_mutations = (
        {"repository_id": "repository:foreign"},
        {"plan_id": _artifact("other-plan")},
        {"translator_id": "translator:other@1"},
        {"solver_id": "solver:other@1"},
        {"kernel_id": "kernel:other@1"},
        {"toolchain_id": "toolchain:other@1"},
        {"policy_id": "policy:other@1"},
        {"theorem_registry_id": "registry:other@1"},
        {"ast_scope_ids": ("scope:other",)},
        {"premise_ids": ("premise:other",)},
        {"provider_id": "provider:other"},
        {"repository_tree_id": "f" * 40},
    )
    for index, changes in enumerate(formal_mutations):
        changed = replace(formal, **changes)
        with pytest.raises(VerificationIdentityError):
            ProofReceipt(
                key,
                _observation(
                    key,
                    TerminalStatus.PROVED,
                    label=f"formal-binding-{index}",
                    additional_artifact_cids=(attempt.attempt_id, changed.receipt_id),
                ),
                changed,
                attempt,
            )

    attempt_mutations = (
        {"plan_id": _artifact("other-attempt-plan")},
        {"step_id": "proof-step:other"},
        {"provider_id": "provider:other"},
        {"repository_tree_id": "e" * 40},
        {"stage": ProofStage.KERNEL_VERIFY},
        {"status": AttemptStatus.FAILED},
    )
    for index, changes in enumerate(attempt_mutations):
        changed_attempt = replace(attempt, **changes)
        changed_formal = replace(formal, attempt_id=changed_attempt.attempt_id)
        with pytest.raises(VerificationIdentityError):
            ProofReceipt(
                key,
                _observation(
                    key,
                    TerminalStatus.PROVED,
                    label=f"attempt-binding-{index}",
                    additional_artifact_cids=(
                        changed_attempt.attempt_id,
                        changed_formal.receipt_id,
                    ),
                ),
                changed_formal,
                changed_attempt,
            )

    producer_shaped_attempt = replace(
        attempt,
        evidence=(),
        input_ids=(_artifact("producer-owned-input"),),
        output_ids=(_artifact("producer-owned-output"),),
    )
    producer_shaped_formal = replace(
        formal,
        attempt_id=producer_shaped_attempt.attempt_id,
    )
    assert ProofReceipt(
        key,
        _observation(
            key,
            TerminalStatus.PROVED,
            label="producer-shaped-attempt",
            additional_artifact_cids=(
                producer_shaped_attempt.attempt_id,
                producer_shaped_formal.receipt_id,
            ),
        ),
        producer_shaped_formal,
        producer_shaped_attempt,
    ).terminal_success


def test_formal_bridge_requires_exact_upstream_headers() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    attempt, formal = _formal_result(key, (_solver_evidence(key),))
    wrapped = ProofReceipt(
        key,
        _observation(
            key,
            TerminalStatus.PROVED,
            additional_artifact_cids=(attempt.attempt_id, formal.receipt_id),
        ),
        formal,
        attempt,
    )
    for nested_name in ("formal_proof_receipt", "proof_attempt"):
        for field, replacement in (
            ("schema", None),
            ("contract_version", None),
            ("contract_version", True),
        ):
            payload = json.loads(json.dumps(wrapped.to_record()))
            nested = payload[nested_name]
            if replacement is None:
                nested.pop(field, None)
            else:
                nested[field] = replacement
            with pytest.raises(VerificationContractError, match="exact"):
                ProofReceipt.from_dict(payload)


def test_formal_bridge_snapshots_mutable_attempt_receipt_and_evidence() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    evidence_metadata = {
        "counterexample_verified": False,
        "nested": {"value": "before"},
    }
    evidence = ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=_artifact("snapshot-solver-result"),
        subject_id=key.proof_obligation_cid,
        verifier_id="solver:z3@4.13.3",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        metadata=evidence_metadata,
    )
    attempt, formal = _formal_result(key, (evidence,))
    attempt_usage = {"cpu_ms": 10}
    attempt_metadata = {"nested": {"value": "before"}}
    formal_usage = {"wall_ms": 20}
    formal_metadata = {"nested": {"value": "before"}}
    attempt = replace(
        attempt,
        resource_usage=attempt_usage,
        metadata=attempt_metadata,
    )
    formal = replace(
        formal,
        attempt_id=attempt.attempt_id,
        resource_usage=formal_usage,
        metadata=formal_metadata,
    )
    wrapped = ProofReceipt(
        key,
        _observation(
            key,
            TerminalStatus.PROVED,
            label="snapshot-proof",
            additional_artifact_cids=(attempt.attempt_id, formal.receipt_id),
        ),
        formal,
        attempt,
    )
    before = wrapped.receipt_id
    evidence_metadata["nested"]["value"] = "after"
    attempt_usage["cpu_ms"] = 99
    attempt_metadata["nested"]["value"] = "after"
    formal_usage["wall_ms"] = 99
    formal_metadata["nested"]["value"] = "after"
    assert wrapped.receipt_id == before
    assert wrapped.formal_proof_receipt is not None
    assert wrapped.proof_attempt is not None
    with pytest.raises(TypeError):
        wrapped.formal_proof_receipt.metadata["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        wrapped.proof_attempt.resource_usage["new"] = 1  # type: ignore[index]


def test_proof_direct_observation_cannot_mint_proved_or_failed() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    for status in (
        TerminalStatus.PROVED,
        TerminalStatus.DISPROVED,
        TerminalStatus.PASSED,
        TerminalStatus.FAILED,
    ):
        with pytest.raises(VerificationContractError, match="conclusive proof"):
            ProofReceipt(key=key, execution=_observation(key, status))


def test_cache_decision_reuses_only_successful_receipt_statuses() -> None:
    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key))
    decision = CacheReuseDecision(
        key_cid=key.key_id,
        disposition=CacheReuseDisposition.REUSED,
        reason_codes=("exact_current_production_receipt",),
        candidate_receipt=receipt,
    )
    assert decision.reusable
    assert CacheReuseDecision.from_dict(decision.to_record()) == decision

    for status in (
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
        TerminalStatus.STALE,
        TerminalStatus.UNKNOWN,
    ):
        with pytest.raises(VerificationContractError, match="successful terminal"):
            rejected = TypeCheckReceipt(key, _observation(key, status))
            replace(decision, candidate_receipt=rejected)

    payload = decision.to_record()
    payload["reusable"] = 1
    with pytest.raises(VerificationContractError, match="boolean"):
        CacheReuseDecision.from_dict(payload)

    other_key = _key(receipt_schema_version=2)
    with pytest.raises(VerificationIdentityError, match="exact key"):
        CacheReuseDecision(
            key_cid=other_key.key_id,
            disposition=CacheReuseDisposition.REUSED,
            reason_codes=("forged_cross_key_reuse",),
            candidate_receipt=receipt,
        )

    forged = decision.to_record()
    forged["candidate_status"] = TerminalStatus.SIMULATED.value
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        CacheReuseDecision.from_dict(forged)


def test_model_route_is_provider_neutral_and_boolean_projection_is_strict() -> None:
    decision = _route()
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert ModelRouteDecision.from_dict(decision.to_record()) == decision
    parameters = inspect.signature(ModelRouteDecision).parameters
    assert not {"provider", "vendor", "model_id"} & set(parameters)

    payload = decision.to_record()
    payload["provider"] = "vendor-specific"
    with pytest.raises(VerificationContractError, match="unsupported fields"):
        ModelRouteDecision.from_dict(payload)

    payload = decision.to_record()
    payload["requires_human_review"] = 0
    with pytest.raises(VerificationContractError, match="boolean"):
        ModelRouteDecision.from_dict(payload)


def _plan(
    key: VerificationReceiptKey,
    *additional_keys: VerificationReceiptKey,
) -> VerificationPlan:
    keys = (key, *additional_keys)
    return VerificationPlan(
        repository_tree_cid=key.repository_tree_cid,
        semantic_state_root_cid=key.semantic_state_root_cid,
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        required_receipt_keys=keys,
        cache_reuse_decisions=tuple(
            CacheReuseDecision(
                key_cid=item.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            )
            for item in keys
        ),
        affected_tests=(),
        fallback_tests=(),
        required_static_checks=(),
        required_type_checks=("src/example.py",),
        affected_proof_obligation_cids=tuple(
            sorted(
                {
                    item.proof_obligation_cid
                    for item in keys
                    if item.receipt_kind is VerificationReceiptKind.PROOF
                }
            )
        ),
        full_suite_receipt_key_cids=(),
        full_suite_required=False,
        full_suite_reason_codes=(),
        human_review_required=False,
        human_review_reason_codes=(),
        expected_cpu_millis=1_000,
        expected_memory_bytes=256 * 1024 * 1024,
        expected_processes=1,
        expected_proof_slots=sum(
            item.receipt_kind is VerificationReceiptKind.PROOF for item in keys
        ),
        expected_artifact_bytes=1_000_000,
        step_timeouts_ms={"type-check": 30_000},
        max_execution_time_ms=60_000,
        dependency_dag={"type-check": ()},
        acceptance_criteria=("all required current checks pass",),
        policy_cid=_artifact("verification-policy"),
    )


def test_verification_plan_round_trip_order_and_fail_closed_dag() -> None:
    key = _key()
    plan = _plan(key)
    assert plan.execution_order == ("type-check",)
    assert VerificationPlan.from_dict(plan.to_record()) == plan

    with pytest.raises(VerificationContractError, match="cycle"):
        replace(
            plan,
            dependency_dag={"a": ("b",), "b": ("a",)},
            step_timeouts_ms={"a": 1, "b": 1},
        )
    full_suite_key = _key(VerificationReceiptKind.TEST)
    with pytest.raises(VerificationContractError, match="reason_codes"):
        replace(
            _plan(full_suite_key),
            full_suite_required=True,
            full_suite_receipt_key_cids=(full_suite_key.key_id,),
            full_suite_reason_codes=(),
        )
    with pytest.raises(VerificationContractError, match="boolean"):
        replace(plan, human_review_required=1)  # type: ignore[arg-type]

    with pytest.raises(VerificationIdentityError, match="cover required"):
        replace(plan, cache_reuse_decisions=())

    with pytest.raises(VerificationIdentityError, match="proof obligations"):
        replace(plan, affected_proof_obligation_cids=(_artifact("unbound-proof"),))

    with pytest.raises(VerificationContractError, match="one decision per key"):
        replace(
            plan,
            cache_reuse_decisions=(
                plan.cache_reuse_decisions[0],
                CacheReuseDecision(
                    key_cid=key.key_id,
                    disposition=CacheReuseDisposition.MISSING,
                    reason_codes=("duplicate",),
                ),
            ),
        )


def test_counterexample_is_compact_typed_and_round_trips() -> None:
    key = _key()
    failed = TypeCheckReceipt(key, _observation(key, TerminalStatus.FAILED))
    counterexample = CounterexampleReceipt(
        failed_key_cid=key.key_id,
        failed_receipt_cid=failed.receipt_id,
        failed_selector=key.selector_cid,
        failure_identity_cid=_artifact("failure-identity"),
        relevant_symbol_version_cids=key.affected_symbol_version_cids,
        minimized_traceback=("example.py:12: expected str, observed int",),
        relevant_assertion="result must be a string",
        relevant_input={"state": "present", "value": {"argument_type": "int"}},
        expected_output={"state": "present", "value": "str"},
        observed_output={"state": "present", "value": "int"},
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 10,
                "end_line": 13,
                "artifact_cid": _artifact("source-span"),
                "symbol": "example.calculate",
            },
        ),
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        reproduction_argv=failed.execution.command_argv,
        artifact_cids=(_artifact("bounded-diagnostic"),),
        minimized=True,
        reason_codes=("deterministic_slice_preserved_failure",),
    )
    assert CounterexampleReceipt.from_dict(counterexample.to_record()) == counterexample
    assert len(counterexample.canonical_bytes()) < 262_144

    with pytest.raises(VerificationContractError, match="private or witness"):
        replace(
            counterexample,
            relevant_input={"state": "present", "value": {"secret": "token"}},
        )
    with pytest.raises(VerificationBoundsError):
        replace(counterexample, minimized_traceback=("x" * 3_000,))


def test_bundle_rejects_counterexample_identity_poisoning() -> None:
    key = _key()
    failed = TypeCheckReceipt(
        key, _observation(key, TerminalStatus.FAILED, label="counterexample-failure")
    )
    counterexample = CounterexampleReceipt(
        failed_key_cid=key.key_id,
        failed_receipt_cid=failed.receipt_id,
        failed_selector=key.selector_cid,
        failure_identity_cid=_artifact("failure-identity"),
        relevant_symbol_version_cids=key.affected_symbol_version_cids,
        minimized_traceback=("example.py:12: expected str, observed int",),
        relevant_assertion="result must be a string",
        relevant_input={"state": "present", "value": {"argument_type": "int"}},
        expected_output={"state": "present", "value": "str"},
        observed_output={"state": "present", "value": "int"},
        source_spans=(),
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        reproduction_argv=("/usr/bin/python3.12", "-m", "mypy", "src/example.py"),
        artifact_cids=(_artifact("bounded-diagnostic"),),
        minimized=True,
    )

    bundle = VerificationBundle(
        verification_plan=_plan(key),
        receipts=(failed,),
        reused_receipt_cids=(),
        executed_receipt_cids=(failed.receipt_id,),
        counterexamples=(counterexample,),
        unresolved_requirement_ids=(),
        human_review_required=False,
    )
    assert bundle.counterexamples == (counterexample,)

    mutations = (
        replace(counterexample, failed_key_cid=_artifact("foreign-key")),
        replace(counterexample, failed_selector=_artifact("foreign-selector")),
        replace(counterexample, environment_cid=_artifact("foreign-environment")),
        replace(counterexample, dependency_lock_cid=_artifact("foreign-lock")),
        replace(
            counterexample,
            relevant_symbol_version_cids=(_artifact("foreign-symbol"),),
        ),
        replace(
            counterexample,
            failed_obligation_cid=_artifact("foreign-obligation"),
        ),
        replace(
            counterexample,
            reproduction_argv=("/usr/bin/curl", "https://invalid.example/payload"),
        ),
    )
    for poisoned in mutations:
        with pytest.raises(VerificationIdentityError):
            replace(bundle, counterexamples=(poisoned,))


def _bundle(
    key: VerificationReceiptKey,
    receipt: VerificationReceipt,
) -> VerificationBundle:
    return VerificationBundle(
        verification_plan=_plan(key),
        receipts=(receipt,),
        reused_receipt_cids=(),
        executed_receipt_cids=(receipt.receipt_id,),
        counterexamples=(),
        unresolved_requirement_ids=(),
        human_review_required=False,
    )


def test_bundle_binds_objects_ids_status_cardinality_tree_and_environment() -> None:
    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key))
    bundle = _bundle(key, receipt)
    assert bundle.structurally_complete
    assert VerificationBundle.from_dict(bundle.to_record()) == bundle

    with pytest.raises(VerificationContractError, match="plan-approved cache hit"):
        replace(
            bundle,
            reused_receipt_cids=(receipt.receipt_id,),
            executed_receipt_cids=(),
        )

    reuse_decision = CacheReuseDecision(
        key_cid=key.key_id,
        disposition=CacheReuseDisposition.REUSED,
        reason_codes=("exact_current_production_receipt",),
        candidate_receipt=receipt,
    )
    reuse_plan = replace(
        _plan(key),
        cache_reuse_decisions=(reuse_decision,),
    )
    reused_bundle = VerificationBundle(
        verification_plan=reuse_plan,
        receipts=(receipt,),
        reused_receipt_cids=(receipt.receipt_id,),
        executed_receipt_cids=(),
        counterexamples=(),
        unresolved_requirement_ids=(),
        human_review_required=False,
    )
    assert reused_bundle.structurally_complete

    rerun_receipt = TypeCheckReceipt(
        key,
        _observation(key, label="independent-rerun"),
    )
    with pytest.raises(VerificationContractError, match="exact plan-approved"):
        VerificationBundle(
            verification_plan=reuse_plan,
            receipts=(rerun_receipt,),
            reused_receipt_cids=(rerun_receipt.receipt_id,),
            executed_receipt_cids=(),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )

    with pytest.raises(VerificationContractError, match="conflicts"):
        VerificationBundle(
            verification_plan=reuse_plan,
            receipts=(receipt,),
            reused_receipt_cids=(),
            executed_receipt_cids=(receipt.receipt_id,),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )

    stale_decision = CacheReuseDecision(
        key_cid=key.key_id,
        disposition=CacheReuseDisposition.STALE,
        reason_codes=("candidate_tombstoned",),
        candidate_receipt=receipt,
    )
    stale_plan = replace(
        _plan(key),
        cache_reuse_decisions=(stale_decision,),
    )
    with pytest.raises(VerificationContractError, match="rejected cache candidate"):
        VerificationBundle(
            verification_plan=stale_plan,
            receipts=(receipt,),
            reused_receipt_cids=(),
            executed_receipt_cids=(receipt.receipt_id,),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )

    fresh_after_stale = TypeCheckReceipt(
        key,
        _observation(key, label="fresh-after-stale-candidate"),
    )
    assert VerificationBundle(
        verification_plan=stale_plan,
        receipts=(fresh_after_stale,),
        reused_receipt_cids=(),
        executed_receipt_cids=(fresh_after_stale.receipt_id,),
        counterexamples=(),
        unresolved_requirement_ids=(),
        human_review_required=False,
    ).structurally_complete

    failed = TypeCheckReceipt(
        key, _observation(key, TerminalStatus.FAILED, label="failed")
    )
    assert not _bundle(key, failed).structurally_complete
    with pytest.raises(VerificationContractError, match="reused bundle receipts"):
        replace(
            _bundle(key, failed),
            reused_receipt_cids=(failed.receipt_id,),
            executed_receipt_cids=(),
        )

    other_key = _key(tool_version="other")
    other_receipt = TypeCheckReceipt(other_key, _observation(other_key))
    with pytest.raises(VerificationIdentityError, match="required check set"):
        replace(
            bundle,
            receipts=(other_receipt,),
            executed_receipt_cids=(other_receipt.receipt_id,),
        )

    with pytest.raises(VerificationContractError, match="one result per key"):
        replace(
            bundle,
            receipts=(receipt, failed),
            executed_receipt_cids=(receipt.receipt_id, failed.receipt_id),
        )

    env_values = _compiler_kwargs()
    environment = {
        **env_values["observed_environment"],  # type: ignore[dict-item]
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "aarch64",
            "libc": "glibc-2.39",
        },
    }
    mixed_key = _key(
        observed_environment=environment,
        claimed_environment_cid=_structured_cid(
            ENVIRONMENT_SCHEMA,
            _expected_environment(
                {**env_values, "observed_environment": environment}
            ),
        ),
    )
    mixed_receipt = TypeCheckReceipt(
        mixed_key,
        _observation(mixed_key),
    )
    with pytest.raises(VerificationIdentityError, match="plan identities"):
        _plan(key, mixed_key)
    with pytest.raises(VerificationIdentityError, match="required check set"):
        VerificationBundle(
            verification_plan=_plan(key),
            receipts=(receipt, mixed_receipt),
            reused_receipt_cids=(),
            executed_receipt_cids=(receipt.receipt_id, mixed_receipt.receipt_id),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )


def test_missing_bundle_key_must_be_explicitly_unresolved() -> None:
    key = _key()
    with pytest.raises(VerificationContractError, match="unresolved requirements"):
        VerificationBundle(
            verification_plan=_plan(key),
            receipts=(),
            reused_receipt_cids=(),
            executed_receipt_cids=(),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )


def test_bundle_cannot_downgrade_review_or_designated_full_suite() -> None:
    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key, label="review-required"))
    review_plan = replace(
        _plan(key),
        human_review_required=True,
        human_review_reason_codes=("policy_authority_unresolved",),
    )
    with pytest.raises(VerificationContractError, match="human review"):
        VerificationBundle(
            verification_plan=review_plan,
            receipts=(receipt,),
            reused_receipt_cids=(),
            executed_receipt_cids=(receipt.receipt_id,),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )

    full_key = _key(VerificationReceiptKind.TEST)
    full_plan = replace(
        _plan(full_key),
        full_suite_required=True,
        full_suite_receipt_key_cids=(full_key.key_id,),
        full_suite_reason_codes=("selector_uncertain",),
    )
    pending = VerificationBundle(
        verification_plan=full_plan,
        receipts=(),
        reused_receipt_cids=(),
        executed_receipt_cids=(),
        counterexamples=(),
        unresolved_requirement_ids=(full_key.key_id,),
        human_review_required=False,
    )
    assert pending.mandatory_fallback_pending
    full_receipt = TestReceipt(full_key, _observation(full_key, label="full-suite"))
    completed = VerificationBundle(
        verification_plan=full_plan,
        receipts=(full_receipt,),
        reused_receipt_cids=(),
        executed_receipt_cids=(full_receipt.receipt_id,),
        counterexamples=(),
        unresolved_requirement_ids=(),
        human_review_required=False,
    )
    assert not completed.mandatory_fallback_pending


def test_unresolved_proof_count_is_distinct_and_inconclusive_results_remain_adverse() -> None:
    first = _key(VerificationReceiptKind.PROOF)
    second = _key(VerificationReceiptKind.PROOF, receipt_schema_version=2)
    assert first.proof_obligation_cid == second.proof_obligation_cid
    timed_out = ProofReceipt(
        first,
        _observation(first, TerminalStatus.TIMEOUT, label="proof-timeout"),
    )
    bundle = VerificationBundle(
        verification_plan=_plan(first, second),
        receipts=(timed_out,),
        reused_receipt_cids=(),
        executed_receipt_cids=(timed_out.receipt_id,),
        counterexamples=(),
        unresolved_requirement_ids=(second.key_id,),
        human_review_required=False,
    )
    assert bundle.unresolved_proof_obligation_cids == (
        first.proof_obligation_cid,
    )
    assert bundle.unresolved_obligation_count == 1
    commitment = build_verification_commitment(bundle)
    assert commitment.unresolved_obligation_count == 1
    assert commitment.aggregate_terminal_status is TerminalStatus.TIMEOUT

    type_key = _key()
    missing_type = VerificationBundle(
        verification_plan=_plan(type_key),
        receipts=(),
        reused_receipt_cids=(),
        executed_receipt_cids=(),
        counterexamples=(),
        unresolved_requirement_ids=(type_key.key_id,),
        human_review_required=False,
    )
    type_commitment = build_verification_commitment(missing_type)
    assert type_commitment.unresolved_obligation_count == 0
    assert type_commitment.aggregate_terminal_status is TerminalStatus.UNKNOWN


def test_summary_round_trip_and_route_flag_consistency() -> None:
    key = _key()
    summary = VerificationSummary(
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        changed_symbol_version_cids=key.affected_symbol_version_cids,
        dependency_cone_symbols=("example.calculate",),
        selected_tests=("test/test_example.py::test_calculate",),
        reused_check_key_cids=(),
        executed_check_key_cids=(key.key_id,),
        failure_receipt_cids=(),
        counterexample_cids=(),
        unresolved_obligation_cids=(),
        full_suite_pending=False,
        human_review_required=False,
        verification_wall_time_ms=125,
        reused_time_saved_ms=0,
        counterexample_context_tokens=0,
        aggregate_terminal_status=TerminalStatus.PASSED,
        model_route_decision=_route(),
        policy_cid=_artifact("summary-policy"),
    )
    assert VerificationSummary.from_dict(summary.to_record()) == summary
    with pytest.raises(VerificationContractError, match="human-review"):
        replace(summary, human_review_required=True)


def test_all_terminal_statuses_round_trip_in_commitment_leaves() -> None:
    for status in TerminalStatus:
        if status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}:
            key = _key(VerificationReceiptKind.PROOF)
            disproved = status is TerminalStatus.DISPROVED
            attempt, formal = _formal_result(
                key,
                (_solver_evidence(key, accepted=not disproved),),
                verdict=(ProofVerdict.DISPROVED if disproved else ProofVerdict.PROVED),
            )
            receipt: VerificationReceipt = ProofReceipt(
                key=key,
                execution=_observation(
                    key,
                    status,
                    label=f"commitment-{status.value}",
                    additional_artifact_cids=(attempt.attempt_id, formal.receipt_id),
                ),
                formal_proof_receipt=formal,
                proof_attempt=attempt,
            )
        else:
            key = _key()
            receipt = TypeCheckReceipt(
                key,
                _observation(key, status, label=f"commitment-{status.value}"),
            )
        commitment = build_verification_commitment(_bundle(key, receipt))
        assert VerificationCommitment.from_dict(commitment.to_record()) == commitment
        assert commitment.aggregate_terminal_status is status


def test_commitment_is_deterministic_sensitive_and_exact_membership_bound() -> None:
    first_key = _key()
    second_key = _key(receipt_schema_version=2)
    first_receipt = TypeCheckReceipt(first_key, _observation(first_key, label="first"))
    second_receipt = TypeCheckReceipt(
        second_key, _observation(second_key, label="second")
    )

    def bundle_for(
        receipts: tuple[VerificationReceipt, ...],
    ) -> VerificationBundle:
        return VerificationBundle(
            verification_plan=_plan(first_key, second_key),
            receipts=receipts,
            reused_receipt_cids=(),
            executed_receipt_cids=tuple(item.receipt_id for item in receipts),
            counterexamples=(),
            unresolved_requirement_ids=(),
            human_review_required=False,
        )

    forward = build_verification_commitment(bundle_for((first_receipt, second_receipt)))
    reverse = build_verification_commitment(bundle_for((second_receipt, first_receipt)))
    assert forward.merkle_root == reverse.merkle_root
    assert forward.commitment_id == reverse.commitment_id
    assert forward.aggregate_terminal_status is TerminalStatus.PASSED
    assert VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF is False

    changed_second = TypeCheckReceipt(
        second_key,
        _observation(second_key, TerminalStatus.FAILED, label="changed-second"),
    )
    changed = build_verification_commitment(bundle_for((first_receipt, changed_second)))
    assert changed.merkle_root != forward.merkle_root
    assert changed.commitment_id != forward.commitment_id
    assert changed.required_check_set_cid == forward.required_check_set_cid

    narrower = build_verification_commitment(_bundle(first_key, first_receipt))
    assert narrower.required_check_set_cid != forward.required_check_set_cid

    forged = forward.to_record()
    forged["verification_bundle"]["verification_plan"] = _plan(
        first_key
    ).to_record()
    with pytest.raises(VerificationContractError):
        VerificationCommitment.from_dict(forged)

    forged = forward.to_record()
    forged["admitted_leaves"][0]["receipt_cid"] = _artifact("foreign-receipt")
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        VerificationCommitment.from_dict(forged)

    forged = forward.to_record()
    forged["public_statement"]["environment_cid"] = _artifact("foreign-environment")
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        VerificationCommitment.from_dict(forged)

    forged = forward.to_record()
    forged["verification_bundle"]["receipts"][0]["status"] = (
        TerminalStatus.SIMULATED.value
    )
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        VerificationCommitment.from_dict(forged)


def test_commitment_cannot_claim_pass_for_empty_or_unresolved_membership() -> None:
    key = _key(VerificationReceiptKind.PROOF)
    unresolved_bundle = VerificationBundle(
        verification_plan=_plan(key),
        receipts=(),
        reused_receipt_cids=(),
        executed_receipt_cids=(),
        counterexamples=(),
        unresolved_requirement_ids=(key.key_id,),
        human_review_required=False,
    )
    commitment = build_verification_commitment(unresolved_bundle)
    assert commitment.admitted_leaves == ()
    assert commitment.unresolved_obligation_count == 1
    assert commitment.aggregate_terminal_status is TerminalStatus.UNKNOWN
    forged = commitment.to_record()
    forged["aggregate_terminal_status"] = TerminalStatus.PASSED.value
    with pytest.raises(VerificationIdentityError, match="derived projection"):
        VerificationCommitment.from_dict(forged)
    assert tuple(inspect.signature(VerificationCommitment).parameters) == (
        "verification_bundle",
    )


def test_aggregate_status_precedence_is_fail_closed() -> None:
    assert (
        aggregate_terminal_status((TerminalStatus.PASSED, TerminalStatus.PROVED))
        is TerminalStatus.PASSED
    )
    assert aggregate_terminal_status((TerminalStatus.PROVED,)) is TerminalStatus.PROVED
    assert (
        aggregate_terminal_status((TerminalStatus.PASSED, TerminalStatus.TIMEOUT))
        is TerminalStatus.TIMEOUT
    )
    assert (
        aggregate_terminal_status((TerminalStatus.FAILED, TerminalStatus.INVALID))
        is TerminalStatus.INVALID
    )
    assert (
        aggregate_terminal_status((), unresolved_obligation_count=0)
        is TerminalStatus.UNKNOWN
    )


def test_canonical_json_round_trip_is_deterministic_and_nested_values_are_frozen() -> (
    None
):
    decision = ModelRouteDecision(
        route=ModelRoute.SMALL_LOCAL_MODEL,
        considered_routes=(ModelRoute.SMALL_LOCAL_MODEL,),
        decisive_reason_codes=("localized_exact_counterexample",),
        required_capabilities=("exact_contracts", "bounded_context"),
        context_token_estimate=1_000,
        policy_cid=_artifact("policy"),
    )
    payload = decision.to_dict()
    reordered = {key: payload[key] for key in reversed(tuple(payload))}
    assert ModelRouteDecision.from_dict(reordered).content_id == decision.content_id
    assert json.loads(decision.to_json())["route"] == "small_local_model"

    key = _key()
    receipt = TypeCheckReceipt(key, _observation(key))
    commitment = build_verification_commitment(_bundle(key, receipt))
    with pytest.raises(TypeError):
        commitment.public_statement["new"] = "mutation"  # type: ignore[index]


def test_package_import_surface_is_complete() -> None:
    from ipfs_accelerate_py.agent_supervisor import verification

    for name in (
        "StaticAnalysisReceipt",
        "TypeCheckReceipt",
        "TestReceipt",
        "ProofReceipt",
        "CounterexampleReceipt",
        "VerificationBundle",
        "VerificationSummary",
        "CacheReuseDecision",
        "ModelRouteDecision",
        "VerificationPlan",
        "VerificationCommitment",
        "VerificationReceiptKey",
        "VerificationIdentityCompiler",
        "build_verification_commitment",
    ):
        assert name in verification.__all__
        assert getattr(verification, name) is not None


def test_raw_identity_helper_uses_real_cid_not_pseudo_hash() -> None:
    key = _key()
    assert key.dependency_lock_cid == cid_for_bytes(
        _compiler_kwargs()["dependency_lock_bytes"]  # type: ignore[arg-type]
    )
    assert key.dependency_lock_cid.startswith("bafkrei")
