"""Deterministic invalidation mutation population for proof-backed test reuse.

PTR-091: every admitted identity and dependency class that binds
``TestExecutionKey@1`` must change the exact execution context when mutated.
Locator-index candidates remain hints; they cannot authorize a skip when the
current execution key, policy, circuit, or verifying key no longer match the
immutable receipt/certificate.

This module is a pure fixture corpus.  It never calls a prover, network, or
user cache root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    EligibilityClass,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
    TestProofCache,
)

NOW_MS: Final = 10_000
BASELINE_CREATED_MS: Final = 9_000
BASELINE_EXPIRES_MS: Final = 11_000

MUTATION_CORPUS_INTERFACE: Final = "ProofReuseMutationCorpus@1"


class MutationTarget(str, Enum):
    """Where a mutation is applied."""

    EXECUTION_KEY = "execution_key"
    POLICY = "policy"


@dataclass(frozen=True)
class MutationSpec:
    """One externally distinguishable invalidation case.

    Attributes:
        name: Stable case id used in pytest parametrization.
        category: Evidence-class label from the PTR-091 evidence subset.
        target: Whether the current execution key or current policy is mutated.
        field: Attribute name on the execution key or policy mapping.
        value: Replacement value for ``field``.
        expected_reason: Bounded ``ReuseReasonCode`` after re-admission.
        description: Human-readable intent (not authority).
    """

    name: str
    category: str
    target: MutationTarget
    field: str
    value: Any
    expected_reason: ReuseReasonCode
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.category or not self.field:
            raise ValueError("mutation name, category, and field are required")
        if not isinstance(self.target, MutationTarget):
            object.__setattr__(self, "target", MutationTarget(self.target))
        if not isinstance(self.expected_reason, ReuseReasonCode):
            object.__setattr__(
                self,
                "expected_reason",
                ReuseReasonCode(self.expected_reason),
            )


# Evidence subset from PTR-091: test/import/indirect dependency/fixture/
# conftest/hook/parameter/lock/environment/hardware/data/dynamic import/
# dirty tree/policy/circuit/key mutations.
INVALIDATION_MUTATIONS: Final[tuple[MutationSpec, ...]] = (
    MutationSpec(
        name="test_module",
        category="test",
        target=MutationTarget.EXECUTION_KEY,
        field="test_module_cid",
        value="cid:test-module-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Test module source identity changed",
    ),
    MutationSpec(
        name="test_function",
        category="test",
        target=MutationTarget.EXECUTION_KEY,
        field="test_function_cid",
        value="cid:test-function-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Test function body identity changed",
    ),
    MutationSpec(
        name="test_ast",
        category="test",
        target=MutationTarget.EXECUTION_KEY,
        field="test_ast_cid",
        value="cid:test-ast-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Test AST identity changed",
    ),
    MutationSpec(
        name="import_static_trace",
        category="import",
        target=MutationTarget.EXECUTION_KEY,
        field="static_trace_root_cid",
        value="cid:static-trace-import-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Direct import closure root changed",
    ),
    MutationSpec(
        name="indirect_dependency",
        category="indirect_dependency",
        target=MutationTarget.EXECUTION_KEY,
        field="components",
        value={
            "direct_import": "cid:import-a",
            "indirect_dependency": "cid:indirect-mutated",
        },
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Transitive dependency identity changed",
    ),
    MutationSpec(
        name="fixture_definition",
        category="fixture",
        target=MutationTarget.EXECUTION_KEY,
        field="fixture_cids",
        value=("cid:fixture-mutated",),
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Fixture definition or value adapter changed",
    ),
    MutationSpec(
        name="conftest_closure",
        category="conftest",
        target=MutationTarget.EXECUTION_KEY,
        field="conftest_closure_cid",
        value="cid:conftest-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Conftest hierarchy identity changed",
    ),
    MutationSpec(
        name="hook_plugin",
        category="hook",
        target=MutationTarget.EXECUTION_KEY,
        field="hook_plugin_cids",
        value=("cid:hook-mutated",),
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Pytest hook/plugin code identity changed",
    ),
    MutationSpec(
        name="parameter_source",
        category="parameter",
        target=MutationTarget.EXECUTION_KEY,
        field="parameter_source_cid",
        value="cid:parameter-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Canonical parameter identity changed",
    ),
    MutationSpec(
        name="dependency_lock",
        category="lock",
        target=MutationTarget.EXECUTION_KEY,
        field="dependency_lock_cid",
        value="cid:lock-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Dependency lock fingerprint changed",
    ),
    MutationSpec(
        name="installed_distributions",
        category="lock",
        target=MutationTarget.EXECUTION_KEY,
        field="installed_distributions_cid",
        value="cid:distributions-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Installed distribution fingerprint changed",
    ),
    MutationSpec(
        name="environment",
        category="environment",
        target=MutationTarget.EXECUTION_KEY,
        field="environment_cid",
        value="cid:environment-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Allowlisted environment identity changed",
    ),
    MutationSpec(
        name="hardware_capability",
        category="hardware",
        target=MutationTarget.EXECUTION_KEY,
        field="hardware_capability_cid",
        value="cid:hardware-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Hardware/capability identity changed",
    ),
    MutationSpec(
        name="external_data_snapshot",
        category="data",
        target=MutationTarget.EXECUTION_KEY,
        field="external_snapshot_cids",
        value=("cid:data-snapshot-mutated",),
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="External data snapshot identity changed",
    ),
    MutationSpec(
        name="dynamic_import_frontier",
        category="dynamic_import",
        target=MutationTarget.EXECUTION_KEY,
        field="static_unknown_frontier",
        value=("dynamic_import:mutated",),
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Dynamic-import frontier identity changed",
    ),
    MutationSpec(
        name="dynamic_import_non_reusable",
        category="dynamic_import",
        target=MutationTarget.EXECUTION_KEY,
        field="eligibility_class",
        value=EligibilityClass.NON_REUSABLE,
        expected_reason=ReuseReasonCode.ELIGIBILITY_DENIED,
        description="Dynamic/unresolved import forces non_reusable class",
    ),
    MutationSpec(
        name="dirty_overlay",
        category="dirty_tree",
        target=MutationTarget.EXECUTION_KEY,
        field="dirty_overlay_cid",
        value="cid:dirty-overlay-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Dirty working-tree overlay identity changed",
    ),
    MutationSpec(
        name="git_tree",
        category="dirty_tree",
        target=MutationTarget.EXECUTION_KEY,
        field="git_tree_id",
        value="tree:mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Git tree identity changed",
    ),
    MutationSpec(
        name="repository_forest",
        category="dirty_tree",
        target=MutationTarget.EXECUTION_KEY,
        field="repository_forest_cid",
        value="cid:repository-forest-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Admitted repository-forest binding changed",
    ),
    MutationSpec(
        name="config",
        category="config",
        target=MutationTarget.EXECUTION_KEY,
        field="config_cid",
        value="cid:config-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Pytest/config identity changed",
    ),
    MutationSpec(
        name="policy_binding",
        category="policy",
        target=MutationTarget.EXECUTION_KEY,
        field="policy_cid",
        value="cid:policy-mutated",
        expected_reason=ReuseReasonCode.POLICY_MISMATCH,
        description="Reuse policy binding on execution key changed",
    ),
    MutationSpec(
        name="tracer_schema",
        category="policy",
        target=MutationTarget.EXECUTION_KEY,
        field="tracer_schema_cid",
        value="cid:tracer-schema-mutated",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        description="Tracer schema identity changed",
    ),
    MutationSpec(
        name="circuit",
        category="circuit",
        target=MutationTarget.POLICY,
        field="circuit_cid",
        value="cid:circuit-mutated",
        expected_reason=ReuseReasonCode.POLICY_MISMATCH,
        description="Current admitted circuit identity changed",
    ),
    MutationSpec(
        name="verifying_key",
        category="key",
        target=MutationTarget.POLICY,
        field="verifying_key_cid",
        value="cid:verifying-key-mutated",
        expected_reason=ReuseReasonCode.POLICY_MISMATCH,
        description="Current verifying-key identity changed",
    ),
    MutationSpec(
        name="policy_requirements",
        category="policy",
        target=MutationTarget.POLICY,
        field="policy_cid",
        value="cid:policy-requirements-mutated",
        expected_reason=ReuseReasonCode.POLICY_MISMATCH,
        description="Current policy requirements identity changed",
    ),
)


REQUIRED_MUTATION_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "test",
        "import",
        "indirect_dependency",
        "fixture",
        "conftest",
        "hook",
        "parameter",
        "lock",
        "environment",
        "hardware",
        "data",
        "dynamic_import",
        "dirty_tree",
        "policy",
        "circuit",
        "key",
    }
)


@dataclass
class StaleSkipTracker:
    """Authoritative accumulator for false/stale skip admissions.

    The tracker is the single source of truth for the population-level
    ``stale_skip_count`` asserted by PTR-091.  Only ``SKIP`` / ``proof_cache_hit``
    outcomes under a mutated current identity count as stale.
    """

    __test__: ClassVar[bool] = False

    stale_skip_count: int = 0
    executed_count: int = 0
    decisions: list[tuple[str, str, str]] = field(default_factory=list)

    def record(
        self,
        *,
        case: str,
        decision: ReuseDecision,
        expected_run: bool = True,
    ) -> None:
        action = (
            decision.action.value
            if isinstance(decision.action, ReuseAction)
            else str(decision.action)
        )
        reason = (
            decision.reason_code.value
            if isinstance(decision.reason_code, ReuseReasonCode)
            else str(decision.reason_code)
        )
        self.decisions.append((case, action, reason))
        is_skip = decision.action is ReuseAction.SKIP or (
            decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
        )
        if expected_run and is_skip:
            self.stale_skip_count += 1
        elif decision.action is ReuseAction.RUN:
            self.executed_count += 1

    @property
    def authoritative_stale_skip_count(self) -> int:
        return self.stale_skip_count


def assert_no_stale_proof_skip(
    decision: ReuseDecision,
    *,
    tracker: StaleSkipTracker | None = None,
    case: str = "",
    expected_reason: ReuseReasonCode | None = None,
) -> None:
    """Fail closed if a mutated context still authorizes a proof-backed skip."""

    if tracker is not None:
        tracker.record(case=case or "anonymous", decision=decision, expected_run=True)

    if decision.action is ReuseAction.SKIP:
        raise AssertionError(
            f"stale proof skip for case={case!r}: reason={decision.reason_code!r} "
            f"certificate_cid={decision.certificate_cid!r}"
        )
    if decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT:
        raise AssertionError(
            f"stale proof_cache_hit under RUN semantics for case={case!r}"
        )
    if decision.action is not ReuseAction.RUN:
        raise AssertionError(
            f"expected RUN after mutation case={case!r}, got {decision.action!r}"
        )
    if expected_reason is not None and decision.reason_code is not expected_reason:
        raise AssertionError(
            f"case={case!r}: expected reason {expected_reason!r}, "
            f"got {decision.reason_code!r}"
        )
    if decision.certificate_cid or decision.receipt_cid:
        # RUN may carry empty CIDs only; non-empty hit fields would smuggle skip.
        if decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT:
            raise AssertionError(
                f"case={case!r}: hit fields present on non-authorizing decision"
            )


def baseline_locator(
    *,
    node_id: str = "test/api/test_mutation_subject.py::test_subject",
) -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:proof-reuse-mutations",
        package_identity="package:ipfs-accelerate-py",
        node_id=node_id,
    )


def baseline_execution_key(locator: TestLocatorKey) -> TestExecutionKey:
    """Full identity surface so each mutation class has a bound baseline field."""

    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        git_commit_id="commit:baseline",
        git_tree_id="tree:baseline",
        gitlink_state_cid="cid:gitlink-baseline",
        dirty_overlay_cid="cid:dirty-overlay-clean",
        test_module_cid="cid:test-module",
        test_class_cid="cid:test-class",
        test_function_cid="cid:test-function",
        decorator_cids=("cid:decorator",),
        parameter_source_cid="cid:parameter-source",
        test_ast_cid="cid:test-ast",
        fixture_cids=("cid:fixture",),
        conftest_closure_cid="cid:conftest",
        hook_plugin_cids=("cid:hook",),
        static_trace_root_cid="cid:static-trace",
        static_unknown_frontier=(),
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        pytest_version="8.3.2",
        python_version="3.12.4",
        plugin_versions_cid="cid:plugin-versions",
        command_semantics_cid="cid:command-semantics",
        config_cid="cid:config",
        markers=("proof_reuse",),
        dependency_lock_cid="cid:dependency-lock",
        installed_distributions_cid="cid:installed-distributions",
        environment_cid="cid:environment",
        platform_cid="cid:platform",
        interpreter_abi_cid="cid:interpreter-abi",
        hardware_capability_cid="cid:hardware",
        external_snapshot_cids=("cid:data-snapshot",),
        policy_cid="cid:policy",
        canonicalization_schema_cid="cid:canonicalization",
        tracer_schema_cid="cid:tracer-schema",
        certificate_schema_cid="cid:certificate-schema",
        eligibility_class=EligibilityClass.REPOSITORY_FOREST_BOUND,
        components={
            "direct_import": "cid:import-a",
            "indirect_dependency": "cid:indirect-a",
        },
    )


def baseline_policy(**changes: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "policy_cid": "cid:policy",
        "statement_cid": "cid:statement",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:verifying-key",
        "proof_system_id": "groth16",
        "trusted_issuer_ids": ("issuer:trusted",),
        "allowed_epochs": ("epoch:7",),
        "revoked_issuer_ids": (),
        "revoked_receipt_cids": (),
        "revoked_certificate_cids": (),
    }
    policy.update(changes)
    return policy


def baseline_receipt(
    locator: TestLocatorKey,
    execution_key: TestExecutionKey,
) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid="cid:completeness-receipt",
        dependency_forest_cid=execution_key.repository_forest_cid,
        issuer_key_id="key:issuer",
        policy_cid=execution_key.policy_cid,
    )


def baseline_certificate(
    receipt: TestPassReceipt,
    execution_key: TestExecutionKey,
    *,
    circuit_cid: str = "cid:circuit",
    verifying_key_cid: str = "cid:verifying-key",
) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid=circuit_cid,
        verifying_key_cid=verifying_key_cid,
        proof_artifact_cid="cid:proof",
        issuer_id="issuer:trusted",
        epoch="epoch:7",
        proof_system_id="groth16",
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        public_inputs={
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": execution_key.execution_key_id,
            "policy_cid": execution_key.policy_cid,
            "statement_cid": "cid:statement",
            "circuit_cid": circuit_cid,
            "verifying_key_cid": verifying_key_cid,
            "proof_system_id": "groth16",
            "issuer_id": "issuer:trusted",
            "issuer_key_id": "key:issuer",
            "epoch": "epoch:7",
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
    )


def baseline_candidate(
    receipt: TestPassReceipt,
    certificate: TestProofCertificate,
) -> dict[str, Any]:
    return TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=BASELINE_CREATED_MS,
        expires_at_ms=BASELINE_EXPIRES_MS,
    )


@dataclass(frozen=True)
class BaselineArtifacts:
    """Immutable warm-reuse baseline under a single locator."""

    locator: TestLocatorKey
    execution_key: TestExecutionKey
    receipt: TestPassReceipt
    certificate: TestProofCertificate
    candidate: dict[str, Any]
    policy: dict[str, Any]

    @property
    def execution_key_cid(self) -> str:
        return self.execution_key.execution_key_id


def build_baseline_artifacts(
    *,
    node_id: str = "test/api/test_mutation_subject.py::test_subject",
) -> BaselineArtifacts:
    locator = baseline_locator(node_id=node_id)
    execution_key = baseline_execution_key(locator)
    receipt = baseline_receipt(locator, execution_key)
    certificate = baseline_certificate(receipt, execution_key)
    candidate = baseline_candidate(receipt, certificate)
    return BaselineArtifacts(
        locator=locator,
        execution_key=execution_key,
        receipt=receipt,
        certificate=certificate,
        candidate=candidate,
        policy=baseline_policy(),
    )


def apply_mutation(
    baseline: BaselineArtifacts,
    mutation: MutationSpec,
) -> tuple[TestExecutionKey, dict[str, Any]]:
    """Return the current (execution_key, policy) after applying ``mutation``."""

    if mutation.target is MutationTarget.EXECUTION_KEY:
        if not hasattr(baseline.execution_key, mutation.field):
            raise AttributeError(
                f"execution key has no field {mutation.field!r} for {mutation.name}"
            )
        current_key = replace(
            baseline.execution_key,
            **{mutation.field: mutation.value},
        )
        return current_key, dict(baseline.policy)

    if mutation.target is MutationTarget.POLICY:
        current_policy = dict(baseline.policy)
        current_policy[mutation.field] = mutation.value
        return baseline.execution_key, current_policy

    raise ValueError(f"unsupported mutation target {mutation.target!r}")


def mutation_changes_execution_context(
    baseline: BaselineArtifacts,
    mutation: MutationSpec,
) -> bool:
    """True when the mutation changes exact current identity or policy authority."""

    current_key, current_policy = apply_mutation(baseline, mutation)
    if mutation.target is MutationTarget.EXECUTION_KEY:
        return current_key.execution_key_id != baseline.execution_key.execution_key_id
    return current_policy != baseline.policy


class ProofReuseMutationCorpus:
    """Closed population of invalidation mutations for PTR-091.

    The corpus is intentionally static and deterministic so CI can treat the
    population as a completeness gate rather than a sampling experiment.
    """

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = MUTATION_CORPUS_INTERFACE

    def __init__(
        self,
        mutations: Sequence[MutationSpec] | None = None,
    ) -> None:
        items = tuple(mutations if mutations is not None else INVALIDATION_MUTATIONS)
        if not items:
            raise ValueError("mutation corpus cannot be empty")
        names = [item.name for item in items]
        if len(names) != len(set(names)):
            raise ValueError("mutation names must be unique")
        self._mutations = items

    @property
    def mutations(self) -> tuple[MutationSpec, ...]:
        return self._mutations

    def __iter__(self):
        return iter(self._mutations)

    def __len__(self) -> int:
        return len(self._mutations)

    def by_name(self, name: str) -> MutationSpec:
        for mutation in self._mutations:
            if mutation.name == name:
                return mutation
        raise KeyError(name)

    def categories(self) -> frozenset[str]:
        return frozenset(item.category for item in self._mutations)

    def missing_required_categories(self) -> frozenset[str]:
        return REQUIRED_MUTATION_CATEGORIES - self.categories()

    def ensure_complete(self) -> None:
        missing = self.missing_required_categories()
        if missing:
            raise AssertionError(
                f"mutation corpus missing required categories: {sorted(missing)}"
            )

    def apply(
        self,
        baseline: BaselineArtifacts,
        mutation: MutationSpec | str,
    ) -> tuple[TestExecutionKey, dict[str, Any]]:
        if isinstance(mutation, str):
            mutation = self.by_name(mutation)
        return apply_mutation(baseline, mutation)

    def evaluate_population(
        self,
        *,
        verifier: Callable[..., Any] | None = None,
        tracker: StaleSkipTracker | None = None,
        execute: Callable[[str], None] | None = None,
    ) -> StaleSkipTracker:
        """Run every mutation against a warm baseline candidate and execute.

        Returns the authoritative stale-skip tracker (zero expected).
        """

        tracker = tracker if tracker is not None else StaleSkipTracker()
        baseline = build_baseline_artifacts()
        verify = verifier if verifier is not None else (lambda *_args, **_kwargs: True)

        for mutation in self._mutations:
            current_key, current_policy = apply_mutation(baseline, mutation)
            assert mutation_changes_execution_context(baseline, mutation), mutation.name

            cache = TestProofCache(
                current_policy=current_policy,
                verifier=verify,
                clock=lambda: NOW_MS,
            )
            # Locator index returns the *baseline* candidate (stale identity).
            result = cache.lookup(
                baseline.locator,
                current_key,
                candidates=(baseline.candidate,),
                now_ms=NOW_MS,
            )
            assert_no_stale_proof_skip(
                result.decision,
                tracker=tracker,
                case=mutation.name,
                expected_reason=mutation.expected_reason,
            )
            if execute is not None:
                execute(mutation.name)
            else:
                # Default: model real test body execution after RUN.
                pass

        return tracker


def unrelated_locator_candidate(
    *,
    baseline: BaselineArtifacts | None = None,
    other_node_id: str = "test/api/test_other.py::test_other",
) -> tuple[TestLocatorKey, TestExecutionKey, dict[str, Any]]:
    """Build a valid candidate under a *different* locator/execution key.

    Used to prove locator-index pollution cannot override current identity.
    """

    del baseline  # reserved for call-site symmetry
    other = build_baseline_artifacts(node_id=other_node_id)
    return other.locator, other.execution_key, other.candidate


__all__ = [
    "BASELINE_CREATED_MS",
    "BASELINE_EXPIRES_MS",
    "INVALIDATION_MUTATIONS",
    "MUTATION_CORPUS_INTERFACE",
    "NOW_MS",
    "REQUIRED_MUTATION_CATEGORIES",
    "BaselineArtifacts",
    "MutationSpec",
    "MutationTarget",
    "ProofReuseMutationCorpus",
    "StaleSkipTracker",
    "apply_mutation",
    "assert_no_stale_proof_skip",
    "baseline_candidate",
    "baseline_certificate",
    "baseline_execution_key",
    "baseline_locator",
    "baseline_policy",
    "baseline_receipt",
    "build_baseline_artifacts",
    "mutation_changes_execution_context",
    "unrelated_locator_candidate",
]
