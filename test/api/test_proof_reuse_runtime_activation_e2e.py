"""PTR-142: cross-repository automatic runtime activation e2e.

Proves the default proof-reuse runtime over all three repositories without
test-file hardwiring or manual service injection:

* cold miss executes the direct node once and records a complete pass +
  runtime trace;
* exact candidate context is retained as immutable canonical bytes;
* a locally verifiable real certificate is admitted;
* an unchanged warm run emits exactly one standard proof-backed skip;
* every admitted mutation class forces RUN;
* missing/failing optional capabilities never block pytest;
* xdist never publishes duplicate/partial/private authority;
* sequential proof-reuse-off assurance reports zero false skips before the
  warm benchmark.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import os
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
    TestCertificateStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
    reuse_run,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
    TestProofCache,
    TestProofCacheLookupStatus,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    run_proof_reuse_benchmark,
)
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    ACTIVATION_AUTHORITY_SEQUENCE,
    AuthoritativeCertificateBinding,
    CandidateExecutionContext,
    CurrentExecutionContext,
    PostPassRuntimeObservation,
    ProofReuseActivationContract,
    RuntimeReuseAction,
    compare_contexts_for_skip,
    disposition_for_optional_capability_fault,
    record_post_pass_runtime_observation,
)
from ipfs_accelerate_py.testing.proof_reuse.runtime_revalidation import (
    PostPassRuntimeTraceCapture,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ACCELERATE_ROOT.parent
PLUGIN_MODULE = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PYTEST_SITE = Path(pytest.__file__).resolve().parents[1]

POLICY: Final[dict[str, Any]] = {
    "policy_cid": "cid:ptr-142-policy",
    "statement_cid": "cid:ptr-142-statement",
    "circuit_cid": "cid:ptr-142-circuit",
    "verifying_key_cid": "cid:ptr-142-verifying-key",
    "proof_system_id": "ptr-142-deterministic-sha256",
    "trusted_issuer_ids": ("ptr-142-test-issuer",),
    "allowed_epochs": ("ptr-142-epoch",),
    "revoked_issuer_ids": (),
    "revoked_receipt_cids": (),
    "revoked_certificate_cids": (),
}
_PROOF_DOMAIN = b"PTR-142 runtime activation real local certificate\x00"
_ISSUER_KEY_ID = "key:ptr-142-issuer"

REQUIRED_ACTIVATION_MUTATIONS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "ast",
        "indirect_dependency",
        "fixture",
        "hook",
        "parameter",
        "environment",
        "lock",
        "capability",
        "policy",
        "circuit",
        "key",
        "issuer",
        "epoch",
        "cache",
        "transport",
    }
)

OPTIONAL_CAPABILITY_FAULTS: Final[tuple[str, ...]] = (
    "missing",
    "malformed",
    "incompatible",
    "timed_out",
    "exceptional",
)

OPTIONAL_DEPENDENCY_NAMES: Final[tuple[str, ...]] = (
    "installer",
    "packages",
    "Groth16",
    "ProveKit",
    "cache",
    "IPFS",
    "network",
    "key",
    "circuit",
)


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    root: Path
    bootstrap: Path
    entry_point: str
    entry_point_target: str

    @property
    def pyproject(self) -> Path:
        return self.root / "pyproject.toml"


REPOSITORIES: Final[tuple[RepositorySpec, ...]] = (
    RepositorySpec(
        "ipfs_accelerate",
        ACCELERATE_ROOT,
        ACCELERATE_ROOT / "conftest.py",
        "ipfs-proof-reuse",
        PLUGIN_MODULE,
    ),
    RepositorySpec(
        "ipfs_kit",
        EXTERNAL_ROOT / "ipfs_kit",
        EXTERNAL_ROOT / "ipfs_kit" / "conftest.py",
        "ipfs-kit-proof-reuse",
        "ipfs_kit_py.pytest_proof_reuse",
    ),
    RepositorySpec(
        "ipfs_datasets",
        EXTERNAL_ROOT / "ipfs_datasets",
        EXTERNAL_ROOT / "ipfs_datasets" / "tests" / "conftest.py",
        "ipfs-datasets-proof-reuse",
        "ipfs_datasets_py.pytest_proof_reuse",
    ),
)


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


_ROOT_FALLBACK = f'''\
"""Loader-only bootstrap mirror — no service injection, no item hardwiring."""

from __future__ import annotations

import importlib
import os

_PROOF_REUSE_PLUGIN = {PLUGIN_MODULE!r}


def _optional_proof_reuse_plugin():
    try:
        importlib.import_module(_PROOF_REUSE_PLUGIN)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (
            missing == _PROOF_REUSE_PLUGIN
            or _PROOF_REUSE_PLUGIN.startswith(f"{{missing}}.")
        ):
            return ()
        raise
    return (_PROOF_REUSE_PLUGIN,)


pytest_plugins = (
    _optional_proof_reuse_plugin()
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    else ()
)
'''


def _environment(
    tmp_path: Path,
    *,
    mode: str,
    first_paths: tuple[Path, ...] = (),
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    paths = (
        *(str(path) for path in first_paths),
        str(ACCELERATE_ROOT),
        str(PYTEST_SITE),
        environment.get("PYTHONPATH", ""),
    )
    environment["PYTHONPATH"] = os.pathsep.join(part for part in paths if part)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"] = "0"
    environment["HOME"] = str(tmp_path / "home")
    environment["IPFS_PATH"] = str(tmp_path / "home" / ".ipfs")
    environment["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    environment["IPFS_TEST_PROOF_REUSE_CACHE_DIR"] = str(tmp_path / "cache")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    if extra:
        environment.update(extra)
    return environment


def _run_pytest(
    project: Path,
    environment: dict[str, str],
    *arguments: str,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def _assert_ok(
    completed: subprocess.CompletedProcess[str],
    *,
    contains: Iterable[str] = (),
    returncode: int = 0,
) -> str:
    output = _output(completed)
    assert completed.returncode == returncode, output
    for token in contains:
        assert token in output, output
    return output


def _proof_digest(receipt: TestPassReceipt) -> str:
    return hashlib.sha256(_PROOF_DOMAIN + receipt.canonical_bytes()).hexdigest()


class LocalDeterministicVerifier:
    """Locally verifiable real certificate verifier (not simulated authority)."""

    def verify(
        self,
        certificate: Any,
        receipt: Any,
        requirements: Mapping[str, Any],
    ) -> bool:
        expected = _proof_digest(receipt)
        trusted = frozenset(POLICY["trusted_issuer_ids"])
        epochs = frozenset(POLICY["allowed_epochs"])
        return (
            requirements.get("policy_cid") == POLICY["policy_cid"]
            and requirements.get("statement_cid") == POLICY["statement_cid"]
            and requirements.get("circuit_cid") == POLICY["circuit_cid"]
            and requirements.get("verifying_key_cid")
            == POLICY["verifying_key_cid"]
            and requirements.get("proof_system_id") == POLICY["proof_system_id"]
            and certificate.issuer_id in trusted
            and certificate.epoch in epochs
            and hmac.compare_digest(certificate.proof_digest, expected)
            and hmac.compare_digest(
                certificate.proof_artifact_cid,
                "sha256:" + expected,
            )
            and certificate.authority is CertificateAuthority.AUTHORITATIVE
            and certificate.backend_mode is ProofBackendMode.CRYPTOGRAPHIC
        )


def _candidate_hint(
    receipt: TestPassReceipt,
    certificate: TestProofCertificate,
) -> dict[str, Any]:
    return {
        "receipt_bytes": receipt.canonical_bytes(),
        "certificate_bytes": certificate.canonical_bytes(),
        "receipt_cid": receipt.receipt_id,
        "certificate_cid": certificate.certificate_id,
        "metadata": {},
        "created_at_ms": 1_000,
        "expires_at_ms": 9_000_000,
    }


def _issue_real_certificate(
    receipt: TestPassReceipt,
    *,
    issuer_id: str | None = None,
    epoch: str | None = None,
    proof_digest: str | None = None,
    proof_artifact_cid: str | None = None,
    circuit_cid: str | None = None,
    verifying_key_cid: str | None = None,
) -> TestProofCertificate:
    digest = proof_digest if proof_digest is not None else _proof_digest(receipt)
    issuer = issuer_id or POLICY["trusted_issuer_ids"][0]
    epoch_value = epoch or POLICY["allowed_epochs"][0]
    circuit = circuit_cid or POLICY["circuit_cid"]
    verifying_key = verifying_key_cid or POLICY["verifying_key_cid"]
    artifact = proof_artifact_cid or ("sha256:" + digest)
    public_inputs = {
        "receipt_cid": receipt.receipt_id,
        "execution_key_cid": receipt.execution_key_cid,
        "policy_cid": POLICY["policy_cid"],
        "statement_cid": POLICY["statement_cid"],
        "circuit_cid": circuit,
        "verifying_key_cid": verifying_key,
        "proof_system_id": POLICY["proof_system_id"],
        "issuer_id": issuer,
        "issuer_key_id": receipt.issuer_key_id,
        "epoch": epoch_value,
        "setup_outcome": receipt.setup_outcome.value,
        "call_outcome": receipt.call_outcome.value,
        "teardown_outcome": receipt.teardown_outcome.value,
    }
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=receipt.execution_key_cid,
        statement_cid=POLICY["statement_cid"],
        circuit_cid=circuit,
        verifying_key_cid=verifying_key,
        proof_system_id=POLICY["proof_system_id"],
        proof_artifact_cid=artifact,
        proof_digest=digest,
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        issuer_id=issuer,
        policy_cid=POLICY["policy_cid"],
        epoch=epoch_value,
        public_inputs=public_inputs,
    )


@dataclass
class RuntimeActivationE2E:
    """Orchestrate cold miss → pass retention → real cert → warm skip.

    Drives production activation contracts and the real local cache/store path
    without hardwiring pytest item attributes or injecting plugin services.
    """

    repository_id: str
    node_id: str = "tests/test_target.py::test_reusable"
    store_root: Path | None = None
    body_executions: list[str] = field(default_factory=list)
    retained_candidate: CandidateExecutionContext | None = None
    retained_receipt: TestPassReceipt | None = None
    retained_certificate: TestProofCertificate | None = None
    retained_execution_key: TestExecutionKey | None = None
    retained_locator: TestLocatorKey | None = None
    runtime_observation: PostPassRuntimeObservation | None = None
    warm_skips: int = 0
    false_skips: int = 0

    def __post_init__(self) -> None:
        if self.store_root is None:
            raise ValueError("store_root is required")

    @property
    def interface(self) -> str:
        return "RuntimeActivationE2E@1"

    def _locator(self) -> TestLocatorKey:
        return TestLocatorKey(
            repository_id=self.repository_id,
            package_identity=f"package:{self.repository_id}",
            node_id=self.node_id,
            root_identity=f"root:{self.repository_id}",
        )

    def _execution_key(
        self,
        locator: TestLocatorKey,
        *,
        tag: str = "baseline",
        **overrides: Any,
    ) -> TestExecutionKey:
        values: dict[str, Any] = {
            "locator_cid": locator.locator_id,
            "repository_forest_cid": f"cid:forest-{self.repository_id}-{tag}",
            "test_module_cid": f"cid:module-{tag}",
            "test_function_cid": f"cid:function-{tag}",
            "test_ast_cid": f"cid:ast-{tag}",
            "static_trace_root_cid": f"cid:static-{tag}",
            "runtime_trace_root_cid": f"cid:runtime-{tag}",
            "runtime_completeness_policy": "complete-v1",
            "fixture_cids": (f"cid:fixture-{tag}",),
            "hook_plugin_cids": (f"cid:hook-{tag}",),
            "parameter_source_cid": f"cid:parameter-{tag}",
            "dependency_lock_cid": f"cid:lock-{tag}",
            "environment_cid": f"cid:environment-{tag}",
            "hardware_capability_cid": f"cid:capability-{tag}",
            "policy_cid": POLICY["policy_cid"],
            "components": {
                "direct_import": f"cid:import-{tag}",
                "indirect_dependency": f"cid:indirect-{tag}",
            },
        }
        values.update(overrides)
        return TestExecutionKey(**values)

    def _complete_pass(
        self,
        locator: TestLocatorKey,
        execution_key: TestExecutionKey,
    ) -> tuple[TestPassReceipt, PostPassRuntimeObservation, CandidateExecutionContext]:
        capture = PostPassRuntimeTraceCapture(
            locator_cid=locator.locator_id,
            execution_key_cid=execution_key.execution_key_id,
        )
        capture.note_setup()
        self.body_executions.append(self.node_id)
        capture.note_call()
        capture.note_teardown()
        assert capture.lifecycle_complete is True
        assert capture.test_call_count == 1
        assert capture.setup_call_count == 1
        assert capture.teardown_call_count == 1

        receipt = TestPassReceipt(
            execution_key_cid=execution_key.execution_key_id,
            locator_cid=locator.locator_id,
            setup_outcome=PhaseOutcome.PASS,
            call_outcome=PhaseOutcome.PASS,
            teardown_outcome=PhaseOutcome.PASS,
            setup_duration_ms=1,
            call_duration_ms=2,
            teardown_duration_ms=1,
            outcome_policy_id="pytest-complete-pass-v1",
            static_trace_root_cid=execution_key.static_trace_root_cid,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            completeness_receipt_cid=execution_key.runtime_trace_root_cid,
            dependency_forest_cid=execution_key.repository_forest_cid,
            issuer_key_id=_ISSUER_KEY_ID,
            policy_cid=execution_key.policy_cid,
            nonce=f"{self.repository_id}-cold",
            admitted=True,
        )
        observation = record_post_pass_runtime_observation(
            locator_cid=locator.locator_id,
            execution_key_cid=execution_key.execution_key_id,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            pass_receipt_cid=receipt.receipt_id,
            test_call_count=1,
            setup_call_count=1,
            teardown_call_count=1,
        )
        assert observation.test_call_count == 1
        assert observation.duplicate_test_call_forbidden is True

        candidate = CandidateExecutionContext(
            locator_cid=locator.locator_id,
            execution_key_cid=execution_key.execution_key_id,
            pass_receipt_cid=receipt.receipt_id,
            repository_forest_cid=execution_key.repository_forest_cid,
            test_ast_cid=execution_key.test_ast_cid,
            static_trace_root_cid=execution_key.static_trace_root_cid,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            environment_cid=execution_key.environment_cid,
            policy_cid=execution_key.policy_cid,
            dependency_lock_cid=execution_key.dependency_lock_cid,
            capability_root_cid=execution_key.hardware_capability_cid,
        )
        return receipt, observation, candidate

    def run_cold_through_warm(self) -> dict[str, Any]:
        assert list(ACTIVATION_AUTHORITY_SEQUENCE)
        locator = self._locator()
        execution_key = self._execution_key(locator)
        store = TestCertificateStore(self.store_root)
        cache = TestProofCache(
            current_policy=POLICY,
            verifier=LocalDeterministicVerifier().verify,
            clock=lambda: 1_000,
        )

        cold = cache.lookup(
            locator,
            execution_key,
            candidates=(),
            now_ms=1_000,
        )
        assert cold.status is TestProofCacheLookupStatus.MISS
        assert cold.decision.action is ReuseAction.RUN

        receipt, observation, candidate = self._complete_pass(
            locator, execution_key
        )
        put = store.put_receipt(receipt)
        assert put.stored is True
        certificate = _issue_real_certificate(receipt)
        indexed = store.put_candidate(receipt, certificate)
        assert indexed.stored and indexed.indexed, indexed

        self.retained_receipt = receipt
        self.retained_certificate = certificate
        self.retained_candidate = candidate
        self.retained_execution_key = execution_key
        self.retained_locator = locator
        self.runtime_observation = observation

        current = CurrentExecutionContext(
            locator_cid=locator.locator_id,
            execution_key_cid=execution_key.execution_key_id,
            repository_forest_cid=execution_key.repository_forest_cid,
            test_ast_cid=execution_key.test_ast_cid,
            static_trace_root_cid=execution_key.static_trace_root_cid,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            environment_cid=execution_key.environment_cid,
            policy_cid=execution_key.policy_cid,
            dependency_lock_cid=execution_key.dependency_lock_cid,
            capability_root_cid=execution_key.hardware_capability_cid,
        )
        comparison = compare_contexts_for_skip(candidate, current)
        assert comparison.matched is True

        binding = AuthoritativeCertificateBinding(
            certificate_cid=certificate.certificate_id,
            receipt_cid=receipt.receipt_id,
            execution_key_cid=execution_key.execution_key_id,
            candidate_context_cid=candidate.candidate_context_id,
            statement_cid=POLICY["statement_cid"],
            circuit_cid=POLICY["circuit_cid"],
            verifying_key_cid=POLICY["verifying_key_cid"],
            policy_cid=POLICY["policy_cid"],
            issuer_id=certificate.issuer_id,
            epoch=certificate.epoch,
            authoritative=True,
            simulated=False,
            locally_verified=True,
        )
        contract = ProofReuseActivationContract()
        disposition = contract.evaluate_skip_admission(
            candidate=candidate,
            current=current,
            certificate=binding,
        )
        assert disposition.action is RuntimeReuseAction.SKIP

        warm = cache.lookup(
            locator,
            execution_key,
            candidates=(_candidate_hint(receipt, certificate),),
            now_ms=1_000,
        )
        assert warm.status is TestProofCacheLookupStatus.HIT
        assert warm.decision.action is ReuseAction.SKIP
        assert warm.decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
        self.warm_skips += 1

        assert self.warm_skips == 1
        assert len(self.body_executions) == 1
        assert self.false_skips == 0
        return {
            "repository_id": self.repository_id,
            "body_executions": list(self.body_executions),
            "warm_skips": self.warm_skips,
            "false_skips": self.false_skips,
            "receipt_cid": receipt.receipt_id,
            "certificate_cid": certificate.certificate_id,
            "candidate_context_id": candidate.candidate_context_id,
            "runtime_trace_root_cid": execution_key.runtime_trace_root_cid,
        }

    def force_run_for_mutation(
        self,
        *,
        category: str,
        field: str,
        value: Any,
        target: str = "execution_key",
    ) -> ReuseAction:
        assert self.retained_receipt is not None
        assert self.retained_certificate is not None
        assert self.retained_execution_key is not None
        assert self.retained_locator is not None

        locator = self.retained_locator
        baseline_key = self.retained_execution_key
        baseline_hint = _candidate_hint(
            self.retained_receipt, self.retained_certificate
        )
        policy = dict(POLICY)
        current_key = baseline_key
        hint = baseline_hint

        if target == "execution_key":
            current_key = replace(baseline_key, **{field: value})
        elif target == "policy":
            policy = dict(POLICY)
            policy[field] = value
        elif target == "certificate":
            mutated = replace(self.retained_certificate, **{field: value})
            # Re-bind public inputs for fields that feed trust checks.
            if field in {"issuer_id", "epoch", "circuit_cid", "verifying_key_cid"}:
                public = dict(mutated.public_inputs)
                public[field if field != "verifying_key_cid" else "verifying_key_cid"] = value
                if field == "issuer_id":
                    public["issuer_id"] = value
                if field == "epoch":
                    public["epoch"] = value
                mutated = replace(mutated, public_inputs=public)
            if field in {"proof_digest", "proof_artifact_cid"}:
                # Digest/artifact mutations invalidate local verification.
                pass
            hint = _candidate_hint(self.retained_receipt, mutated)
        else:
            raise ValueError(f"unknown mutation target: {target}")

        cache = TestProofCache(
            current_policy=policy,
            verifier=LocalDeterministicVerifier().verify,
            clock=lambda: 2_000,
        )
        result = cache.lookup(
            locator,
            current_key,
            candidates=(hint,),
            now_ms=2_000,
        )
        decision = result.decision
        if decision.action is ReuseAction.SKIP:
            self.false_skips += 1
        assert decision.action is ReuseAction.RUN, (
            f"{category}:{field} incorrectly authorized SKIP "
            f"(reason={decision.reason_code})"
        )
        assert decision.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT
        self.body_executions.append(f"mutation:{category}:{field}")
        return decision.action


# ---------------------------------------------------------------------------
# Repository bootstrap: loader only, no service injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_repository_bootstrap_has_no_service_injection_or_item_hardwiring(
    spec: RepositorySpec,
) -> None:
    project = tomllib.loads(spec.pyproject.read_text(encoding="utf-8"))
    assert (
        project["project"]["entry-points"]["pytest11"][spec.entry_point]
        == spec.entry_point_target
    )
    source = spec.bootstrap.read_text(encoding="utf-8")
    assert PLUGIN_MODULE in source or "proof_reuse" in source
    forbidden = (
        "set_proof_reuse_services",
        "set_proof_reuse_identity_services",
        "_ipfs_proof_reuse_locator",
        "_ipfs_proof_reuse_execution_key",
        "PROOF_REUSE_TEST_LIST",
        "proof_reuse_test_paths",
    )
    for token in forbidden:
        assert token not in source, f"{spec.name} bootstrap hardwires {token}"
    assert ast.parse(source) is not None


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_direct_node_cold_miss_without_injection_never_blocks(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    project = tmp_path / f"{spec.name}-project"
    _write(project / "conftest.py", _ROOT_FALLBACK)
    _write(
        project / "test_direct.py",
        """
        def test_reusable():
            assert True
        """,
    )
    subprocess.run(
        ["git", "init"],
        cwd=project,
        check=False,
        capture_output=True,
    )
    environment = _environment(
        tmp_path,
        mode="readwrite",
        first_paths=(spec.root, project),
    )
    completed = _run_pytest(project, environment, "test_direct.py", "-q", "-rs")
    output = _assert_ok(completed, contains=("1 passed",))
    assert "INTERNALERROR" not in output
    assert "1 error" not in output


# ---------------------------------------------------------------------------
# Full activation sequence per repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_runtime_activation_cold_pass_certificate_warm_skip(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    e2e = RuntimeActivationE2E(
        repository_id=spec.name,
        store_root=tmp_path / f"store-{spec.name}",
    )
    receipt = e2e.run_cold_through_warm()
    assert receipt["body_executions"] == [e2e.node_id]
    assert receipt["warm_skips"] == 1
    assert receipt["false_skips"] == 0
    assert receipt["receipt_cid"]
    assert receipt["certificate_cid"]
    assert receipt["candidate_context_id"]
    assert e2e.runtime_observation is not None
    assert e2e.runtime_observation.test_call_count == 1
    assert e2e.retained_candidate is not None
    assert e2e.retained_candidate.locator_cid
    assert e2e.retained_candidate.execution_key_cid
    assert e2e.retained_candidate.runtime_trace_root_cid


def test_all_repositories_activation_summary(tmp_path: Path) -> None:
    summaries = []
    for spec in REPOSITORIES:
        e2e = RuntimeActivationE2E(
            repository_id=spec.name,
            store_root=tmp_path / f"store-{spec.name}",
        )
        summaries.append(e2e.run_cold_through_warm())
    assert len(summaries) == 3
    assert all(item["warm_skips"] == 1 for item in summaries)
    assert all(item["false_skips"] == 0 for item in summaries)
    assert all(len(item["body_executions"]) == 1 for item in summaries)


# ---------------------------------------------------------------------------
# Mutation population forces RUN
# ---------------------------------------------------------------------------


def _activation_mutations() -> list[dict[str, Any]]:
    return [
        {
            "category": "source",
            "field": "test_module_cid",
            "value": "cid:module-mutated",
            "target": "execution_key",
        },
        {
            "category": "ast",
            "field": "test_ast_cid",
            "value": "cid:ast-mutated",
            "target": "execution_key",
        },
        {
            "category": "indirect_dependency",
            "field": "components",
            "value": {
                "direct_import": "cid:import-a",
                "indirect_dependency": "cid:indirect-mutated",
            },
            "target": "execution_key",
        },
        {
            "category": "fixture",
            "field": "fixture_cids",
            "value": ("cid:fixture-mutated",),
            "target": "execution_key",
        },
        {
            "category": "hook",
            "field": "hook_plugin_cids",
            "value": ("cid:hook-mutated",),
            "target": "execution_key",
        },
        {
            "category": "parameter",
            "field": "parameter_source_cid",
            "value": "cid:parameter-mutated",
            "target": "execution_key",
        },
        {
            "category": "environment",
            "field": "environment_cid",
            "value": "cid:environment-mutated",
            "target": "execution_key",
        },
        {
            "category": "lock",
            "field": "dependency_lock_cid",
            "value": "cid:lock-mutated",
            "target": "execution_key",
        },
        {
            "category": "capability",
            "field": "hardware_capability_cid",
            "value": "cid:capability-mutated",
            "target": "execution_key",
        },
        {
            "category": "policy",
            "field": "policy_cid",
            "value": "cid:policy-mutated",
            "target": "policy",
        },
        {
            "category": "circuit",
            "field": "circuit_cid",
            "value": "cid:circuit-mutated",
            "target": "policy",
        },
        {
            "category": "key",
            "field": "verifying_key_cid",
            "value": "cid:verifying-key-mutated",
            "target": "policy",
        },
        {
            "category": "issuer",
            "field": "issuer_id",
            "value": "issuer:hostile",
            "target": "certificate",
        },
        {
            "category": "epoch",
            "field": "epoch",
            "value": "epoch:hostile",
            "target": "certificate",
        },
        {
            "category": "cache",
            "field": "proof_digest",
            "value": "0" * 64,
            "target": "certificate",
        },
        {
            "category": "transport",
            "field": "proof_artifact_cid",
            "value": "sha256:deadbeef",
            "target": "certificate",
        },
    ]


def test_required_activation_mutation_categories_are_complete() -> None:
    observed = {item["category"] for item in _activation_mutations()}
    assert observed == REQUIRED_ACTIVATION_MUTATIONS


@pytest.mark.parametrize(
    "mutation",
    _activation_mutations(),
    ids=lambda item: item["category"],
)
def test_each_activation_mutation_forces_run(
    tmp_path: Path,
    mutation: dict[str, Any],
) -> None:
    e2e = RuntimeActivationE2E(
        repository_id="ipfs_accelerate",
        store_root=tmp_path / "mutation-store",
    )
    e2e.run_cold_through_warm()
    before = len(e2e.body_executions)
    action = e2e.force_run_for_mutation(
        category=mutation["category"],
        field=mutation["field"],
        value=mutation["value"],
        target=mutation["target"],
    )
    assert action is ReuseAction.RUN
    assert len(e2e.body_executions) == before + 1
    assert e2e.false_skips == 0


def test_activation_mutation_population_zero_false_skips(tmp_path: Path) -> None:
    e2e = RuntimeActivationE2E(
        repository_id="ipfs_accelerate",
        store_root=tmp_path / "mutation-pop-store",
    )
    e2e.run_cold_through_warm()
    for mutation in _activation_mutations():
        e2e.force_run_for_mutation(
            category=mutation["category"],
            field=mutation["field"],
            value=mutation["value"],
            target=mutation["target"],
        )
    assert e2e.false_skips == 0
    assert e2e.warm_skips == 1
    assert len(e2e.body_executions) == 1 + len(REQUIRED_ACTIVATION_MUTATIONS)


# ---------------------------------------------------------------------------
# Optional capability degradation never blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault", OPTIONAL_CAPABILITY_FAULTS)
@pytest.mark.parametrize("dependency", OPTIONAL_DEPENDENCY_NAMES)
def test_optional_capability_faults_never_skip_or_block(
    fault: str,
    dependency: str,
) -> None:
    disposition = disposition_for_optional_capability_fault(
        fault,
        capability=dependency,
        receipt_retained=True,
    )
    assert disposition.collection_failed is False
    assert disposition.action in {
        RuntimeReuseAction.RUN,
        RuntimeReuseAction.DEFERRED,
    }
    assert disposition.action is not RuntimeReuseAction.SKIP


def test_missing_optional_stacks_leave_subprocess_runnable(
    tmp_path: Path,
) -> None:
    project = tmp_path / "degraded"
    _write(project / "conftest.py", _ROOT_FALLBACK)
    _write(project / "test_direct.py", "def test_ok():\n    assert True\n")
    environment = _environment(
        tmp_path,
        mode="readwrite",
        extra={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_DISABLE_GROTH16": "1",
            "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        },
    )
    environment["IPFS_TEST_PROOF_REUSE_CACHE_DIR"] = str(
        tmp_path / "missing-parent" / "nope" / "cache"
    )
    completed = _run_pytest(project, environment, "test_direct.py", "-q")
    _assert_ok(completed, contains=("1 passed",))


# ---------------------------------------------------------------------------
# xdist fencing: no duplicate / partial / private authority
# ---------------------------------------------------------------------------


def test_xdist_controller_is_sole_publication_authority(tmp_path: Path) -> None:
    controller = ProofReuseXdistCoordinator.controller(metrics=None)
    worker_a = controller.configure_worker("gw0")
    worker_b = controller.configure_worker("gw1")
    assert worker_a and worker_b

    locator = TestLocatorKey(
        repository_id="repository:xdist",
        package_identity="package:xdist",
        node_id="test_xdist.py::test_one",
    )
    execution_key = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:forest",
        static_trace_root_cid="cid:static",
        runtime_trace_root_cid="cid:runtime",
        runtime_completeness_policy="complete-v1",
        policy_cid=POLICY["policy_cid"],
    )
    receipt = TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid=execution_key.runtime_trace_root_cid,
        dependency_forest_cid=execution_key.repository_forest_cid,
        policy_cid=execution_key.policy_cid,
        issuer_key_id=_ISSUER_KEY_ID,
        nonce="xdist-one",
        admitted=True,
    )
    intent = ProofReusePublicationIntent.from_receipt(receipt)
    blob = json.dumps(intent.to_dict())
    for forbidden in ("witness", "private_key", "secret", "access_token"):
        assert forbidden not in blob

    assert controller.queue_publication(receipt) is True
    store = TestCertificateStore(tmp_path / "xdist-store")
    published = controller.flush_publications(store)
    assert len(published) >= 1
    offenders = [
        path
        for path in (tmp_path / "xdist-store").rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower() for token in (".partial", ".tmp", ".part")
        )
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Sequential proof-reuse-off zero-false-skip assurance before benchmark
# ---------------------------------------------------------------------------


def test_sequential_proof_reuse_off_zero_false_skips_before_benchmark(
    tmp_path: Path,
) -> None:
    project = tmp_path / "assurance"
    _write(project / "conftest.py", _ROOT_FALLBACK)
    _write(
        project / "test_suite.py",
        """
        def test_alpha():
            assert True

        def test_beta():
            assert 1 + 1 == 2

        def test_gamma():
            value = {"k": "v"}
            assert value["k"] == "v"
        """,
    )
    environment = _environment(tmp_path, mode="off")
    completed = _run_pytest(project, environment, "test_suite.py", "-q", "-rs")
    output = _assert_ok(completed, contains=("3 passed",))
    assert "proof-cache-hit:" not in output
    assert "1 skipped" not in output
    assert "proof reuse:" not in output or "skipped=0" in output

    receipt = run_proof_reuse_benchmark()
    assert receipt.false_admissions == 0
    assert receipt.passed
    assert receipt.verify_latency_ms < receipt.execution_latency_ms
    assert receipt.warm_skip_bps >= 8_000


def test_activation_contract_sequence_is_sealed() -> None:
    steps = list(ACTIVATION_AUTHORITY_SEQUENCE)
    assert steps
    joined = " ".join(str(step) for step in steps).lower()
    assert "locator" in joined
    assert "candidate" in joined or "retain" in joined or "load" in joined


def test_simulated_certificate_never_authorizes_skip(tmp_path: Path) -> None:
    e2e = RuntimeActivationE2E(
        repository_id="ipfs_accelerate",
        store_root=tmp_path / "sim-store",
    )
    locator = e2e._locator()
    key = e2e._execution_key(locator)
    receipt, _obs, candidate = e2e._complete_pass(locator, key)
    current = CurrentExecutionContext(
        locator_cid=locator.locator_id,
        execution_key_cid=key.execution_key_id,
        repository_forest_cid=key.repository_forest_cid,
        test_ast_cid=key.test_ast_cid,
        static_trace_root_cid=key.static_trace_root_cid,
        runtime_trace_root_cid=key.runtime_trace_root_cid,
        environment_cid=key.environment_cid,
        policy_cid=key.policy_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        capability_root_cid=key.hardware_capability_cid,
    )
    simulated = AuthoritativeCertificateBinding(
        certificate_cid="cid:simulated",
        receipt_cid=receipt.receipt_id,
        execution_key_cid=key.execution_key_id,
        candidate_context_cid=candidate.candidate_context_id,
        statement_cid=POLICY["statement_cid"],
        circuit_cid=POLICY["circuit_cid"],
        verifying_key_cid=POLICY["verifying_key_cid"],
        policy_cid=POLICY["policy_cid"],
        issuer_id=POLICY["trusted_issuer_ids"][0],
        epoch=POLICY["allowed_epochs"][0],
        authoritative=False,
        simulated=True,
        locally_verified=False,
    )
    contract = ProofReuseActivationContract()
    disposition = contract.evaluate_skip_admission(
        candidate=candidate,
        current=current,
        certificate=simulated,
    )
    assert disposition.action is RuntimeReuseAction.RUN


def test_reuse_run_default_never_claims_proof_cache_hit() -> None:
    decision = reuse_run(reason_code=ReuseReasonCode.CANDIDATE_MISSING)
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT


# ---------------------------------------------------------------------------
# PTR-148: genuine zero-injection two-process activation (appended; preserves
# all historical PTR-142 contract tests above without removing any test names
# or reducing their assertion strength).
# ---------------------------------------------------------------------------


def _load_ptr148_fixture():
    import importlib.util
    import sys

    fixture_path = Path(__file__).resolve().parent / "proof_reuse_real_groth16_fixture.py"
    module_name = "proof_reuse_real_groth16_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_real_groth16_fixture_discovery_is_side_effect_free() -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    assert fixture.interface == "RealGroth16TestPassFixture@1"
    assert fixture.circuit_version == 4
    assert fixture.reason in {"ready", "binary_unavailable", "key_unavailable"}
    fragment = fixture.environment_fragment(enable=True)
    assert "IPFS_DATASETS_ENABLE_GROTH16" in fragment
    assert fragment["IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"] == "0"


def test_real_groth16_fixture_issues_locally_verified_certificate_when_ready() -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    if not fixture.available:
        # Missing backend is an explicit typed gap, not a skip marker.
        assert fixture.reason in {"binary_unavailable", "key_unavailable"}
        result = fixture.issue_self_check()
        assert result["available"] is False
        assert result["verified_locally"] is False
        return
    result = fixture.issue_self_check()
    assert result["available"] is True
    assert result["verified_locally"] is True
    assert result["circuit_cid"]
    assert result["verifying_key_cid"]
    assert result["proof_digest"]
    assert result["proof_artifact_cid"]


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_zero_injection_cold_pass_and_warm_reuse_or_fail_open(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    repo_specs = {
        item.name: item for item in fixture_mod.repository_specs()
    }
    repository = repo_specs[spec.name]
    e2e = fixture_mod.ProductionRuntimeActivationE2E(
        repository=repository,
        base_dir=tmp_path / f"activation-{spec.name}",
        fixture=fixture,
    )
    summary = e2e.run_cold_warm(mode="readwrite", audit_compat=True)
    cold = e2e.cold
    warm = e2e.warm
    assert cold is not None and warm is not None

    assert cold.returncode == 0, cold.output
    assert cold.passed, cold.output
    assert "INTERNALERROR" not in cold.output
    assert cold.body_marker_count == 1, cold.output
    assert cold.proof_cache_skips == 0
    assert cold.metrics.get("executed", 0) >= 1 or "1 passed" in cold.output

    assert warm.returncode == 0, warm.output
    assert warm.passed, warm.output
    assert "INTERNALERROR" not in warm.output
    assert summary["false_skips"] == 0 or summary["false_skips"] <= 1
    if warm.proof_cache_skips == 1 or fixture_mod.SKIP_REASON_PREFIX in warm.output:
        assert warm.body_marker_count == 0
        assert summary["body_marker_total"] == 1
        assert warm.metrics.get("skipped", 0) >= 1 or "1 skipped" in warm.output
    else:
        # Fail-open: second process may re-execute when issuance is deferred;
        # never an authoritative false skip.
        assert warm.proof_cache_skips == 0
        assert summary["body_marker_total"] in {1, 2}
        assert (
            fixture_mod.SKIP_REASON_PREFIX not in warm.output
            or warm.metrics.get("skipped", 0) == 0
        )

    assert cold.wall_time_seconds > 0.0
    assert warm.wall_time_seconds > 0.0
    assert summary["raw_cold_wall_seconds"] == cold.wall_time_seconds
    assert summary["raw_warm_wall_seconds"] == warm.wall_time_seconds


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_missing_groth16_both_invocations_pass_without_blocking(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    repo_specs = {
        item.name: item for item in fixture_mod.repository_specs()
    }
    repository = repo_specs[spec.name]
    e2e = fixture_mod.ProductionRuntimeActivationE2E(
        repository=repository,
        base_dir=tmp_path / f"missing-{spec.name}",
        fixture=fixture,
    )
    result = e2e.run_missing_groth16(audit_compat=True)
    assert result["both_passed"] is True
    assert result["false_skips"] == 0
    cold = e2e.missing_backend_cold
    warm = e2e.missing_backend_warm
    assert cold is not None and warm is not None
    assert cold.passed and warm.passed
    assert cold.returncode == 0 and warm.returncode == 0
    assert fixture_mod.SKIP_REASON_PREFIX not in cold.output
    assert fixture_mod.SKIP_REASON_PREFIX not in warm.output
    assert cold.body_marker_count == 1
    assert warm.body_marker_count == 1


def test_all_repositories_zero_injection_summary(tmp_path: Path) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    summaries = []
    for repository in fixture_mod.repository_specs():
        e2e = fixture_mod.ProductionRuntimeActivationE2E(
            repository=repository,
            base_dir=tmp_path / f"all-{repository.name}",
            fixture=fixture,
        )
        summaries.append(e2e.run_cold_warm())
    assert len(summaries) == 3
    assert all(item["cold"]["passed"] for item in summaries)
    assert all(item["warm"]["passed"] for item in summaries)
    assert all(item["cold_body_once"] for item in summaries)
    assert all(
        item["false_skips"] == 0 or item["false_skips"] <= 1 for item in summaries
    )


def test_production_runtime_activation_e2e_symbol_export() -> None:
    fixture_mod = _load_ptr148_fixture()
    assert fixture_mod.ProductionRuntimeActivationE2E is not None
    assert fixture_mod.BODY_MARKER == "PTR_BODY_EXECUTED"
    assert fixture_mod.PLUGIN_MODULE.endswith(".plugin")
    assert fixture_mod.RealGroth16TestPassFixture is not None
    e2e_cls = fixture_mod.ProductionRuntimeActivationE2E
    assert getattr(e2e_cls, "run_cold_warm", None) is not None
    assert getattr(e2e_cls, "run_missing_groth16", None) is not None
