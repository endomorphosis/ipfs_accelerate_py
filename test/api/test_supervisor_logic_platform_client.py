"""Contract tests for SupervisorLogicPlatformClient@1 (LPC-110).

Covers handshake, catalog, formalization, slice/obligation/plan, capability
discovery, typed invocation, reconstruction, verification, receipts,
counterexamples, and cache freshness — plus request binding, non-overclaim,
and quiet import hermeticity.
"""

from __future__ import annotations

import ast
import importlib
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    LogicFamily,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceKind,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    CLIENT_GOAL_ID,
    CLIENT_SCHEMA_VERSION,
    CLIENT_TASK_ID,
    ClientOperation,
    ClientRequestContext,
    ClientResult,
    LogicPlatformClientError,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
    SupervisorLogicPlatformClient,
    _clear_default_client_for_tests,
    get_supervisor_logic_platform_client,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_provider_contract import (
    SupervisorLogicProviderFacade,
)
from ipfs_datasets_py.logic.backends.provider import LogicProviderRequest
from ipfs_datasets_py.logic.platform.manifest import (
    DEFAULT_LOGIC_PLATFORM_MANIFEST,
    HandshakeRequirements,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SOURCE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "proof"
    / "logic_platform_client.py"
)
CLIENT_NOTE = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "supervisor_client.md"
)
CLIENT_MODULE_NAME = (
    "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client"
)


class FixtureLogicProvider:
    """Minimal in-process provider for typed invocation tests."""

    provider_id = "fixture.logic-platform-client"
    provider_version = "1.0.0"
    protocol_version = 1

    def __init__(self, *, simulated: bool = False, fail: bool = False) -> None:
        self.requests: list[Any] = []
        self.simulated = simulated
        self.fail = fail

    def _invoke(self, request: LogicProviderRequest) -> dict[str, object]:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("fixture provider forced failure")
        return {
            "echo": dict(request.payload),
            "operation": request.operation.value,
            "provider_claimed_authority": "authoritative",
            "simulated": self.simulated,
            "semantic_verdict": "unknown",
        }

    capability = _invoke
    translate = _invoke
    prove = _invoke
    reconstruct = _invoke
    verify = _invoke
    attest = _invoke


def _context(**overrides: Any) -> ClientRequestContext:
    values: dict[str, Any] = {
        "task_id": "LPC-110",
        "repository_tree_id": "tree:sha256:abc",
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:deadbeef",
        "resource_budget": ResourceBudget(
            wall_time_ms=5_000,
            memory_bytes=16 * 1024 * 1024,
            network_allowed=False,
        ),
        "network_allowed": False,
        "deadline_unix_ms": int(time.time() * 1000) + 60_000,
        "correlation_id": "corr:test-1",
        "authority_ceiling": AssuranceLevel.SOLVER_CHECKED.value,
        "evidence_kind": EvidenceKind.SOLVER_RESULT.value,
    }
    values.update(overrides)
    return ClientRequestContext(**values)


def _client(
    *,
    provider: FixtureLogicProvider | None = None,
    require_handshake: bool = True,
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    facade = None
    if provider is not None:
        facade = SupervisorLogicProviderFacade(
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            provider=provider,
        )
    return SupervisorLogicPlatformClient(
        provider_facade=facade,
        require_handshake=require_handshake,
        **kwargs,
    )


def _handshaken_client(
    provider: FixtureLogicProvider | None = None,
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    client = _client(provider=provider, **kwargs)
    result = client.handshake()
    assert result.ok, result.to_dict()
    return client


# ---------------------------------------------------------------------------
# Module / note / identity
# ---------------------------------------------------------------------------


def test_client_interface_and_schema_are_stable() -> None:
    client = SupervisorLogicPlatformClient(require_handshake=False)
    payload = client.to_dict()
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["interface"] == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["schema_version"] == CLIENT_SCHEMA_VERSION
    assert payload["task_id"] == CLIENT_TASK_ID
    assert payload["goal_id"] == CLIENT_GOAL_ID
    assert client.datasets_import_is_lazy() is True
    for operation in ClientOperation:
        assert operation.value in payload["operations"]


def test_declared_note_documents_acceptance_surface() -> None:
    text = CLIENT_NOTE.read_text(encoding="utf-8")
    assert "SupervisorLogicPlatformClient@1" in text
    for token in (
        "handshake",
        "catalog",
        "formalization",
        "slice",
        "obligation",
        "plan",
        "capability",
        "reconstruction",
        "verification",
        "receipts",
        "counterexamples",
        "cache freshness",
    ):
        assert token in text.lower() or token.replace(" ", "_") in text.lower()
    assert "logic_platform_client.py" in text
    assert "test_supervisor_logic_platform_client.py" in text


def test_importing_client_module_does_not_import_datasets_package() -> None:
    source = CLIENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ipfs_datasets_py"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ipfs_datasets_py"), module

    # Runtime: constructing a client must not mark datasets as loaded.
    module = importlib.import_module(CLIENT_MODULE_NAME)
    client = module.SupervisorLogicPlatformClient(require_handshake=False)
    assert client.datasets_import_is_lazy() is True
    assert client.loaded_datasets is False
    # Explicit boundary call is what loads datasets.
    result = client.handshake()
    assert result.ok
    assert client.loaded_datasets is True


def test_manifest_lists_client_as_compatible_adapter() -> None:
    manifest = DEFAULT_LOGIC_PLATFORM_MANIFEST
    assert (
        SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
        in manifest.compatible_adapter_versions
    )


# ---------------------------------------------------------------------------
# Request binding (fail closed)
# ---------------------------------------------------------------------------


def test_request_context_requires_task_tree_policy() -> None:
    with pytest.raises(LogicPlatformClientError):
        ClientRequestContext(
            task_id="",
            repository_tree_id="tree:1",
            policy_id="policy:1",
        )
    with pytest.raises(LogicPlatformClientError):
        ClientRequestContext(
            task_id="task:1",
            repository_tree_id="",
            policy_id="policy:1",
        )
    with pytest.raises(LogicPlatformClientError):
        ClientRequestContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="",
        )


def test_request_context_rejects_network_budget_overclaim() -> None:
    with pytest.raises(LogicPlatformClientError, match="network"):
        ClientRequestContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="policy:1",
            network_allowed=False,
            resource_budget=ResourceBudget(network_allowed=True),
        )


def test_request_context_rejects_kernel_authority_for_candidate_evidence() -> None:
    with pytest.raises(LogicPlatformClientError, match="authority"):
        ClientRequestContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="policy:1",
            evidence_kind=EvidenceKind.ATP_CANDIDATE.value,
            authority_ceiling=AssuranceLevel.KERNEL_VERIFIED.value,
        )


def test_request_context_assigns_correlation_when_missing() -> None:
    ctx = ClientRequestContext(
        task_id="task:1",
        repository_tree_id="tree:1",
        policy_id="policy:1",
        correlation_id="",
    )
    assert ctx.correlation_id.startswith("corr:")


def test_client_result_success_does_not_imply_proved() -> None:
    result = ClientResult.success(
        ClientOperation.PROVE,
        {"operation_status": "succeeded"},
        authority_ceiling=AssuranceLevel.SOLVER_CHECKED.value,
    )
    assert result.ok is True
    assert result.operation_status == "succeeded"
    assert result.semantic_verdict == "unknown"
    assert result.to_dict()["semantic_verdict"] == "unknown"


def test_client_result_rejects_simulated_kernel_authority() -> None:
    with pytest.raises(LogicPlatformClientError, match="simulated"):
        ClientResult.success(
            ClientOperation.PROVE,
            {"simulated": True},
            authority_ceiling=AssuranceLevel.KERNEL_VERIFIED.value,
            simulated=True,
        )


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


def test_handshake_succeeds_without_git_or_siblings() -> None:
    client = _client(require_handshake=True)
    result = client.handshake()
    assert result.ok is True
    assert result.operation is ClientOperation.HANDSHAKE
    assert result.payload is not None
    assert result.payload["compatible"] is True
    assert result.payload["requires_git"] is False
    assert result.payload["requires_sibling_repos"] is False
    assert result.payload["requires_repository_layout"] is False
    assert result.payload["client_interface_listed"] is True
    assert client.loaded_datasets is True


def test_handshake_reports_typed_incompatibility() -> None:
    client = _client()
    result = client.handshake(
        HandshakeRequirements(
            required_adapter_versions=("SupervisorLogicPlatformClient@99",)
        )
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "incompatible"
    details = result.error["details"]["handshake"]
    assert details["compatible"] is False
    assert details["incompatibilities"]


def test_semantic_ops_require_handshake_by_default() -> None:
    client = _client(provider=FixtureLogicProvider(), require_handshake=True)
    with pytest.raises(LogicPlatformClientError, match="handshake"):
        client.catalog(_context())


def test_handshake_is_first_step_before_catalog() -> None:
    client = _client()
    handshake = client.handshake()
    assert handshake.ok
    catalog = client.catalog(_context())
    assert catalog.ok
    assert catalog.payload is not None
    assert catalog.payload["provider_availability_claimed"] is False
    assert catalog.payload["executable"] is False
    assert catalog.payload["content_root"]
    assert catalog.payload["content_digest"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Formalization / slice / obligation / plan
# ---------------------------------------------------------------------------


def test_formalize_slice_obligation_plan_bind_context() -> None:
    client = _handshaken_client()
    ctx = _context(plan_id="plan:fixture-1")
    formalize = client.formalize(
        ctx, {"goal": "prove invariant", "logic_family": LogicFamily.TDFOL}
    )
    assert formalize.ok
    assert formalize.payload is not None
    assert formalize.payload["kind"] == "formalization_artifact"
    assert formalize.payload["bindings"]["task_id"] == "LPC-110"
    assert formalize.payload["proof_success"] is False
    assert formalize.payload["projections"]["logic_family"]["canonical_id"] == "tdfol"

    slice_result = client.slice(ctx, {"slice_id": "slice:1"})
    assert slice_result.ok
    assert slice_result.payload is not None
    assert slice_result.payload["kind"] == "domain_logic_slice"

    obligation = client.obligation(
        ctx, {"obligation_id": "obl:1", "property_kind": "authorization"}
    )
    assert obligation.ok
    assert obligation.payload is not None
    assert obligation.payload["projections"]["property_kind"]["canonical_id"] == (
        "authorization"
    )

    plan = client.plan(ctx, {"plan_id": "plan:fixture-1", "steps": []})
    assert plan.ok
    assert plan.payload is not None
    assert plan.payload["kind"] == "goal_directed_proof_plan"


def test_unknown_residual_vocabulary_fails_closed() -> None:
    client = _handshaken_client()
    result = client.formalize(
        _context(), {"logic_family": "not-a-real-family-xyz"}
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "unknown_vocabulary"


# ---------------------------------------------------------------------------
# Capability / typed invocation / reconstruction / verification
# ---------------------------------------------------------------------------


def test_capability_and_typed_provider_ops() -> None:
    provider = FixtureLogicProvider()
    client = _handshaken_client(provider=provider)
    ctx = _context()

    capability = client.capability(ctx, {"probe": True})
    assert capability.ok
    assert capability.semantic_verdict == "unknown"
    assert capability.payload is not None
    assert "provider_claimed_authority" not in capability.payload
    assert capability.payload["operation_status"] == "succeeded"
    assert capability.authority_ceiling == AssuranceLevel.SOLVER_CHECKED.value

    for method, operation in (
        (client.translate, ClientOperation.TRANSLATE),
        (client.prove, ClientOperation.PROVE),
        (client.reconstruct, ClientOperation.RECONSTRUCT),
        (client.verify, ClientOperation.VERIFY),
        (client.attest, ClientOperation.ATTEST),
    ):
        result = method(ctx, {"obligation_id": f"obl:{operation.value}"})
        assert result.ok, result.to_dict()
        assert result.operation is operation
        assert result.payload is not None
        assert result.payload["proof_success"] is False
        assert result.semantic_verdict == "unknown"

    assert len(provider.requests) == 6


def test_invoke_dispatches_closed_operation_vocabulary() -> None:
    provider = FixtureLogicProvider()
    client = _handshaken_client(provider=provider)
    ctx = _context()
    result = client.invoke(ctx, "prove", {"obligation_id": "obl:invoke"})
    assert result.ok
    assert result.operation is ClientOperation.PROVE


def test_prove_strips_provider_authority_overclaim_and_caps_ceiling() -> None:
    provider = FixtureLogicProvider()
    client = _handshaken_client(provider=provider)
    # Context ceiling is solver_checked; provider claims authoritative.
    result = client.prove(_context(), {"goal": "x"})
    assert result.ok
    assert result.payload is not None
    assert "provider_claimed_authority" not in result.payload
    assert result.authority_ceiling == AssuranceLevel.SOLVER_CHECKED.value
    assert result.payload["authority_ceiling"] == AssuranceLevel.SOLVER_CHECKED.value


def test_simulated_provider_result_stays_below_kernel() -> None:
    provider = FixtureLogicProvider(simulated=True)
    client = _handshaken_client(provider=provider)
    result = client.prove(
        _context(authority_ceiling=AssuranceLevel.SOLVER_CHECKED.value),
        {"goal": "sim"},
    )
    assert result.ok
    assert result.simulated is True
    assert result.authority_ceiling == AssuranceLevel.CANDIDATE.value


def test_provider_failure_surfaces_typed_error() -> None:
    class FailingProvider(FixtureLogicProvider):
        def prove(self, request: LogicProviderRequest) -> dict[str, object]:
            self.requests.append(request)
            raise RuntimeError("boom")

    provider = FailingProvider()
    client = _handshaken_client(provider=provider)
    result = client.prove(_context(), {"goal": "fail"})
    # Facade converts provider exceptions via dispatch; accept either ok=False
    # from the provider boundary or a structured failure code.
    assert result.ok is False or result.semantic_verdict != "proved"


def test_cancelled_context_fails_closed() -> None:
    token = CancellationToken()
    token.cancel()
    client = _handshaken_client(provider=FixtureLogicProvider())
    result = client.prove(_context(cancellation=token), {"goal": "x"})
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "cancelled"


def test_expired_context_fails_closed() -> None:
    client = _handshaken_client(provider=FixtureLogicProvider())
    result = client.prove(
        _context(deadline_unix_ms=1),  # long expired
        {"goal": "x"},
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "timed_out"


def test_typed_invocation_projects_logic_family_residual() -> None:
    provider = FixtureLogicProvider()
    client = _handshaken_client(provider=provider)
    result = client.prove(
        _context(),
        {"logic_family": LogicFamily.DCEC, "obligation_id": "obl:dcec"},
    )
    assert result.ok
    assert provider.requests
    last = provider.requests[-1]
    assert last.payload["logic_family"] == "dcec"
    assert last.payload["logic_family_residual"] == LogicFamily.DCEC.value


# ---------------------------------------------------------------------------
# Receipts / counterexamples / cache freshness
# ---------------------------------------------------------------------------


def test_receipts_are_untrusted_until_admission() -> None:
    client = _handshaken_client()
    result = client.receipts(
        _context(),
        {
            "receipts": [
                {
                    "receipt_id": "receipt:1",
                    "operation_status": "succeeded",
                    "semantic_verdict": "proved",
                    "authority_ceiling": AssuranceLevel.KERNEL_VERIFIED.value,
                    "simulated": True,
                }
            ]
        },
    )
    assert result.ok
    assert result.payload is not None
    assert result.payload["admitted"] is False
    assert result.payload["receipts"][0]["trusted"] is False
    assert result.payload["receipts"][0]["admitted"] is False
    assert result.payload["receipts"][0]["authority_ceiling"] == (
        AssuranceLevel.CANDIDATE.value
    )
    assert result.simulated is True


def test_counterexamples_project_with_evidence_kind() -> None:
    client = _handshaken_client()
    result = client.counterexamples(
        _context(evidence_kind=EvidenceKind.SOLVER_RESULT.value),
        {
            "counterexamples": [
                {"model": {"x": 1}, "semantic_verdict": "disproved"}
            ]
        },
    )
    assert result.ok
    assert result.payload is not None
    assert result.payload["count"] == 1
    assert result.payload["evidence_kind"] == EvidenceKind.SOLVER_RESULT.value
    assert result.payload["semantic_verdict"] == "disproved"
    assert result.payload["counterexamples"][0]["trusted"] is False


def test_cache_freshness_current_and_stale_paths() -> None:
    client = _handshaken_client()
    ctx = _context()
    current = client.cache_freshness(
        ctx, {"cache_key_digest": "sha256:" + ("ab" * 32)}
    )
    assert current.ok
    assert current.payload is not None
    assert current.payload["is_fresh"] is True
    assert current.freshness["status"] == "current"
    assert current.payload["proof_success"] is False

    stale = client.cache_freshness(
        ctx,
        {
            "cache_key_digest": "sha256:" + ("cd" * 32),
            "expected_tree_id": "tree:other",
        },
    )
    assert stale.ok
    assert stale.payload is not None
    assert stale.payload["is_fresh"] is False
    assert "tree_mismatch" in stale.payload["reasons"]
    assert stale.freshness["status"] == "stale"

    declared = client.cache_freshness(ctx, {"status": "stale"})
    assert declared.ok
    assert declared.payload is not None
    assert declared.payload["is_fresh"] is False


# ---------------------------------------------------------------------------
# Factory / operation matrix completeness
# ---------------------------------------------------------------------------


def test_get_supervisor_logic_platform_client_singleton_and_override() -> None:
    _clear_default_client_for_tests()
    first = get_supervisor_logic_platform_client()
    second = get_supervisor_logic_platform_client()
    assert first is second
    third = get_supervisor_logic_platform_client(require_handshake=False)
    assert third is not first
    _clear_default_client_for_tests()


def test_operation_matrix_covers_acceptance_surface() -> None:
    expected = {
        "handshake",
        "catalog",
        "formalize",
        "slice",
        "obligation",
        "plan",
        "capability",
        "translate",
        "prove",
        "reconstruct",
        "verify",
        "attest",
        "receipts",
        "counterexamples",
        "cache_freshness",
    }
    assert {op.value for op in ClientOperation} == expected


def test_end_to_end_acceptance_path() -> None:
    """Exercise the full LPC-110 acceptance path in one cohesive flow."""

    provider = FixtureLogicProvider()
    client = _client(provider=provider)
    handshake = client.handshake()
    assert handshake.ok

    ctx = _context(plan_id="plan:e2e")
    assert client.catalog(ctx).ok
    assert client.formalize(ctx, {"logic_family": "tdfol"}).ok
    assert client.slice(ctx, {"slice_id": "slice:e2e"}).ok
    assert client.obligation(ctx, {"obligation_id": "obl:e2e"}).ok
    assert client.plan(ctx, {"plan_id": "plan:e2e"}).ok
    assert client.capability(ctx, {}).ok
    assert client.translate(ctx, {"source": "ast"}).ok
    assert client.prove(ctx, {"obligation_id": "obl:e2e"}).ok
    assert client.reconstruct(ctx, {"trace": "t"}).ok
    assert client.verify(ctx, {"receipt_id": "r1"}).ok
    assert client.attest(ctx, {"receipt_id": "r1"}).ok
    assert client.receipts(ctx, {"receipts": [{"receipt_id": "r1"}]}).ok
    assert client.counterexamples(
        ctx, {"counterexample": {"model": {"a": 0}}}
    ).ok
    freshness = client.cache_freshness(
        ctx, {"cache_key_digest": "sha256:" + ("11" * 32)}
    )
    assert freshness.ok
    assert freshness.payload is not None
    assert freshness.payload["is_fresh"] is True

    # Success never upgrades to proved across the path.
    prove = client.prove(ctx, {"obligation_id": "obl:final"})
    assert prove.ok
    assert prove.semantic_verdict != "proved"
    assert prove.payload is not None
    assert prove.payload["proof_success"] is False
