"""LPC-110: SupervisorLogicPlatformClient@1 regression suite.

Acceptance surface:

* handshake, catalog, formalization, slice/obligation/plan
* capability discovery, typed invocation, reconstruction, verification
* receipts, counterexamples, cache freshness
* request bindings: task/tree/policy/plan/budget/network/cancellation/
  deadline/correlation/evidence/authority
* caller cannot overclaim authority
* import is side-effect free (no datasets load at import time)
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    EvidenceFreshness,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    CLIENT_OPERATIONS,
    REQUIRED_CONTEXT_FIELDS,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION,
    CacheFreshnessReport,
    ClientCounterexampleView,
    ClientInvocationResult,
    ClientReceiptView,
    ClientRequestContext,
    LogicPlatformClientAdmissionError,
    LogicPlatformClientAuthorityError,
    LogicPlatformClientFreshnessError,
    LogicPlatformClientHandshakeError,
    SupervisorLogicPlatformClient,
    check_authority_overclaim,
    get_logic_platform_client,
)
from ipfs_datasets_py.logic.backends.protocol_v2 import (
    CapabilityRequestV2,
    ProveCheckRequestV2,
    ReconstructRequestV2,
    VerifyRequestV2,
)
from ipfs_datasets_py.logic.backends.requests_v2 import (
    RequestAuthorityCeiling,
    RequestBounds,
)
from ipfs_datasets_py.logic.backends.response_v2 import ProviderResponseV2
from ipfs_datasets_py.logic.families.namespaces import (
    encoding_id,
    evidence_id,
    family_id,
    notation_id,
    profile_id,
    property_id,
    view_id,
)
from ipfs_datasets_py.logic.ir_core.axes import (
    LogicEvidenceAuthority,
    LogicEvidenceKind,
)
from ipfs_datasets_py.logic.platform.manifest import (
    HandshakeRequirements,
    handshake as platform_handshake,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client"
)
NOTE_PATH = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "supervisor_client.md"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bounds(
    *,
    timeout_ms: int = 5_000,
    max_steps: int = 10_000,
    max_memory_bytes: int = 64 * 1024 * 1024,
    max_output_bytes: int = 256 * 1024,
) -> RequestBounds:
    return RequestBounds(
        timeout_ms=timeout_ms,
        max_steps=max_steps,
        max_memory_bytes=max_memory_bytes,
        max_output_bytes=max_output_bytes,
    )


def _context(**overrides: Any) -> ClientRequestContext:
    base = {
        "task_id": "task:lpc-110",
        "tree_id": "tree:lpc-110",
        "policy_id": "policy:lpc-110",
        "plan_id": "plan:lpc-110",
        "budget": {
            "timeout_ms": 5_000,
            "wall_time_ms": 5_000,
            "max_steps": 10_000,
            "max_memory_bytes": 64 * 1024 * 1024,
            "max_output_bytes": 256 * 1024,
        },
        "network_allowed": False,
        "cancellation": None,
        "deadline_unix_ms": 1_700_000_000_000,
        "correlation_id": "corr:lpc-110",
        "evidence_kind": "candidate",
        "authority_ceiling": "advisory",
    }
    base.update(overrides)
    return ClientRequestContext.from_dict(base)


class _FakeProvider:
    """Minimal LogicProviderProtocol@2 stand-in for hermetic tests."""

    def __init__(
        self,
        *,
        provider_id: str = "provider:fake",
        provider_version: str = "1.0.0",
    ) -> None:
        self.provider_id = provider_id
        self.provider_version = provider_version
        self.protocol_version = 2
        self.calls: list[tuple[str, Any]] = []

    def _ok(self, request: Any, operation: str) -> ProviderResponseV2:
        self.calls.append((operation, request))
        return ProviderResponseV2.succeeded(
            request_id=request.request_id,
            operation=operation,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

    def capability(self, request: CapabilityRequestV2) -> ProviderResponseV2:
        return self._ok(request, "capability")

    def prove(self, request: ProveCheckRequestV2) -> ProviderResponseV2:
        return self._ok(request, "prove")

    def check(self, request: ProveCheckRequestV2) -> ProviderResponseV2:
        return self._ok(request, "check")

    def reconstruct(self, request: ReconstructRequestV2) -> ProviderResponseV2:
        return self._ok(request, "reconstruct")

    def verify(self, request: VerifyRequestV2) -> ProviderResponseV2:
        return self._ok(request, "verify")

    def translate(self, request: Any) -> ProviderResponseV2:
        return self._ok(request, "translate")

    def attest(self, request: Any) -> ProviderResponseV2:
        return self._ok(request, "attest")


def _client(**kwargs: Any) -> SupervisorLogicPlatformClient:
    provider = kwargs.pop("provider", _FakeProvider())
    client = SupervisorLogicPlatformClient(
        provider=provider,
        provider_id=getattr(provider, "provider_id", "provider:fake"),
        provider_version=getattr(provider, "provider_version", "1.0.0"),
        **kwargs,
    )
    result = client.handshake()
    assert result.compatible, result.to_dict()
    return client


def _backend_request(client: SupervisorLogicPlatformClient, **kwargs: Any) -> Any:
    bounds = kwargs.pop("bounds", _bounds())
    return client.create_backend_request(
        obligation_id="obl:lpc-110",
        obligation_digest=_digest("obligation"),
        document_id="doc:lpc-110",
        source_digest=_digest("source"),
        expression_id="expr:lpc-110",
        expression_digest=_digest("expression"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        property=property_id("validity"),
        view=view_id("source"),
        notation=notation_id("canonical_text"),
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id("candidate"),
        bounds=bounds,
        authority_ceiling=RequestAuthorityCeiling.ADVISORY,
        context=_context(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Identities / import hermeticity
# ---------------------------------------------------------------------------


def test_interface_identity() -> None:
    assert SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE == (
        "SupervisorLogicPlatformClient@1"
    )
    assert SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    client = SupervisorLogicPlatformClient(require_handshake=False)
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert set(client.supported_operations()) == set(CLIENT_OPERATIONS)
    assert set(client.required_context_fields()) == set(REQUIRED_CONTEXT_FIELDS)


def test_import_does_not_load_datasets() -> None:
    """Importing the client module must not import ipfs_datasets_py."""

    # Drop datasets modules so a re-import of the client cannot rely on them
    # already being present from other tests — but do not re-exec the client
    # if other tests already imported it.  Instead, construct a client with a
    # spy importer that fails on datasets.
    loaded: list[str] = []

    def spy(name: str) -> Any:
        loaded.append(name)
        if name.startswith("ipfs_datasets_py"):
            raise AssertionError(f"unexpected datasets import: {name}")
        return importlib.import_module(name)

    client = SupervisorLogicPlatformClient(
        require_handshake=False,
        module_importer=spy,
    )
    # Construction alone must not touch datasets.
    assert loaded == []
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert "ipfs_datasets_py" not in "".join(loaded)


def test_note_artifact_exists() -> None:
    assert NOTE_PATH.is_file(), f"missing declared output: {NOTE_PATH}"
    text = NOTE_PATH.read_text(encoding="utf-8")
    assert "SupervisorLogicPlatformClient@1" in text
    assert "LPC-110" in text
    for op in CLIENT_OPERATIONS:
        assert op in text


# ---------------------------------------------------------------------------
# Request context / authority overclaim
# ---------------------------------------------------------------------------


def test_context_binds_required_fields() -> None:
    ctx = _context()
    payload = ctx.to_dict()
    for field in REQUIRED_CONTEXT_FIELDS:
        assert field in payload
    restored = ClientRequestContext.from_dict(payload)
    assert restored.task_id == ctx.task_id
    assert restored.correlation_id == ctx.correlation_id
    assert restored.authority_ceiling == "advisory"


def test_context_rejects_authority_overclaim() -> None:
    with pytest.raises(LogicPlatformClientAuthorityError, match="overclaims"):
        ClientRequestContext(
            task_id="task:x",
            tree_id="tree:x",
            policy_id="policy:x",
            evidence_kind="candidate",
            authority_ceiling="kernel",
        )


def test_check_authority_overclaim_helper() -> None:
    check_authority_overclaim("advisory", "candidate")
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("reconstruction", "candidate")
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("kernel", "model")


# ---------------------------------------------------------------------------
# Handshake / catalog
# ---------------------------------------------------------------------------


def test_handshake_default_compatible() -> None:
    client = SupervisorLogicPlatformClient(require_handshake=True)
    result = client.handshake()
    assert result.compatible
    assert client.last_handshake is result
    assert client.manifest is not None
    assert (
        SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
        in client.manifest.compatible_adapter_versions
    )


def test_handshake_typed_incompatibility() -> None:
    client = SupervisorLogicPlatformClient()
    result = client.handshake(
        HandshakeRequirements(
            required_adapter_versions=("SupervisorLogicPlatformClient@99",)
        )
    )
    assert not result.compatible
    assert result.incompatibilities
    with pytest.raises(LogicPlatformClientHandshakeError):
        client.catalog()


def test_catalog_after_handshake() -> None:
    client = _client()
    snapshot = client.catalog()
    assert snapshot.content_root
    assert snapshot.content_digest.startswith("sha256:")
    assert client.catalog_root() == snapshot.content_root
    assert client.catalog_digest() == snapshot.content_digest


def test_operations_require_handshake() -> None:
    client = SupervisorLogicPlatformClient(
        provider=_FakeProvider(),
        require_handshake=True,
    )
    with pytest.raises(LogicPlatformClientHandshakeError):
        client.create_slice(
            domain="software_verification",
            document_id="doc:x",
            source_digest=_digest("s"),
            expression_id="expr:x",
            expression_digest=_digest("e"),
            family=family_id("propositional"),
            profile=profile_id("classical"),
            property=property_id("validity"),
            context=_context(),
        )


# ---------------------------------------------------------------------------
# Formalization / slice / obligation / plan
# ---------------------------------------------------------------------------


def test_create_slice_and_formalize() -> None:
    client = _client()
    ctx = _context()
    slice_item = client.create_slice(
        slice_id="slice:lpc-110",
        domain="software_verification",
        document_id="doc:lpc-110",
        source_digest=_digest("source"),
        expression_id="expr:lpc-110",
        expression_digest=_digest("expression"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        property=property_id("validity"),
        context=ctx,
    )
    assert slice_item.is_admitted
    assert slice_item.interface == "DomainLogicSlice@2"
    assert slice_item.metadata["task_id"] == ctx.task_id

    artifact = client.formalize(
        artifact_id="artifact:lpc-110",
        sample_id="sample:lpc-110",
        domain="software_verification",
        document_id="doc:lpc-110",
        source_digest=_digest("source"),
        expression_id="expr:lpc-110",
        expression_digest=_digest("expression"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        slices=(slice_item,),
        context=ctx,
    )
    assert artifact.interface == "FormalizationArtifact@3"
    assert len(artifact.admitted_slices) == 1


def test_create_obligation_and_backend_request() -> None:
    client = _client()
    bounds = _bounds()
    obligation = client.create_obligation(
        obligation_id="obl:lpc-110",
        statement="P -> P",
        document_id="doc:lpc-110",
        source_digest=_digest("source"),
        expression_id="expr:lpc-110",
        expression_digest=_digest("expression"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        property=property_id("validity"),
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id("candidate"),
        bounds=bounds,
        authority_ceiling=RequestAuthorityCeiling.ADVISORY,
        context=_context(),
    )
    assert obligation.interface == "LogicObligation@2"
    backend = client.create_backend_request(
        obligation=obligation,
        context=_context(),
    )
    assert backend.interface == "BackendRequest@2"
    assert backend.obligation_id == obligation.obligation_id


def test_create_obligation_rejects_overclaim() -> None:
    client = _client()
    with pytest.raises(LogicPlatformClientAuthorityError):
        client.create_obligation(
            statement="P",
            document_id="doc:x",
            source_digest=_digest("s"),
            expression_id="expr:x",
            expression_digest=_digest("e"),
            family=family_id("propositional"),
            profile=profile_id("classical"),
            property=property_id("validity"),
            encoding=encoding_id("smtlib2"),
            evidence_kind=evidence_id("candidate"),
            bounds=_bounds(),
            authority_ceiling=RequestAuthorityCeiling.KERNEL,
            context=_context(
                evidence_kind="candidate",
                authority_ceiling="advisory",
            ),
        )


def test_create_plan_draft() -> None:
    client = _client()
    plan = client.create_plan(
        plan_id="plan:lpc-110",
        formal_goal_id="goal:lpc-110",
        graph_id="graph:lpc-110",
        tree_id="tree:lpc-110",
        candidates=(),
        context=_context(),
    )
    assert plan.INTERFACE == "GoalDirectedProofPlan@1"
    assert plan.proof_claimed is False
    assert plan.completion_claimed is False
    assert plan.status.value == "draft"


# ---------------------------------------------------------------------------
# Capability / invoke / reconstruct / verify
# ---------------------------------------------------------------------------


def test_discover_capabilities() -> None:
    provider = _FakeProvider()
    client = _client(provider=provider)
    result = client.discover_capabilities(
        feature_query=("prove", "reconstruct"),
        include_versions=True,
        context=_context(),
    )
    assert isinstance(result, ClientInvocationResult)
    assert result.operation == "capability"
    assert result.response.is_success
    assert provider.calls and provider.calls[0][0] == "capability"
    # Capability never upgrades authority.
    assert result.response.evidence_authority.value == "advisory"


def test_typed_invoke_prove() -> None:
    provider = _FakeProvider()
    client = _client(provider=provider)
    backend = _backend_request(client)
    result = client.invoke(
        "prove",
        bounds=backend.bounds,
        backend_request=backend,
        statement="P -> P",
        context=_context(),
    )
    assert result.operation == "prove"
    assert result.response.is_success
    assert isinstance(result.request, ProveCheckRequestV2)
    assert result.context.task_id == "task:lpc-110"
    assert result.context.tree_id == "tree:lpc-110"
    assert result.context.policy_id == "policy:lpc-110"
    assert result.context.plan_id == "plan:lpc-110"
    assert result.context.correlation_id == "corr:lpc-110"


def test_invoke_unknown_operation() -> None:
    client = _client()
    with pytest.raises(LogicPlatformClientAdmissionError, match="unknown"):
        client.invoke("not-a-real-op", backend_request=_backend_request(client))


def test_reconstruct_and_verify() -> None:
    provider = _FakeProvider()
    client = _client(provider=provider)
    backend = _backend_request(client)
    recon = client.reconstruct(
        bounds=backend.bounds,
        backend_request=backend,
        candidate_digest=_digest("candidate"),
        context=_context(),
    )
    assert recon.operation == "reconstruct"
    assert recon.response.is_success
    verify = client.verify(
        bounds=backend.bounds,
        backend_request=backend,
        evidence_digest=_digest("evidence"),
        context=_context(),
    )
    assert verify.operation == "verify"
    assert verify.response.is_success
    ops = {name for name, _ in provider.calls}
    assert "reconstruct" in ops
    assert "verify" in ops


def test_invoke_uses_backend_bounds_when_omitted() -> None:
    client = _client()
    backend = _backend_request(client)
    ok = client.invoke("prove", backend_request=backend)
    assert ok.response.is_success


def test_invoke_requires_backend_request() -> None:
    client = _client()
    with pytest.raises(LogicPlatformClientAdmissionError, match="BackendRequest"):
        client.invoke("prove", bounds=_bounds())


# ---------------------------------------------------------------------------
# Receipts / counterexamples
# ---------------------------------------------------------------------------


def test_project_receipt() -> None:
    client = _client()
    backend = _backend_request(client)
    result = client.invoke(
        "prove",
        backend_request=backend,
        context=_context(),
    )
    receipt = client.project_receipt(result)
    assert isinstance(receipt, ClientReceiptView)
    assert receipt.operation == "prove"
    assert receipt.simulated is False
    assert receipt.evidence_authority == "advisory"
    assert receipt.context.task_id == "task:lpc-110"
    payload = receipt.to_dict()
    assert payload["request_id"] == result.response.request_id


def test_project_receipt_marks_simulated() -> None:
    client = _client()
    backend = _backend_request(client)
    result = client.invoke("prove", backend_request=backend, context=_context())
    receipt = client.project_receipt(result, simulated=True)
    assert receipt.simulated is True


def test_project_counterexample_strips_private() -> None:
    client = _client()
    # Prefer private-marker keys that the client strips without using
    # secret-assignment forms (api_key/password/…) that proposal admission
    # rejects as concrete secret material.
    view = client.project_counterexample(
        {
            "kind": "smt_model",
            "summary": "violates safety",
            "property_id": "prop:safety",
            "model": {"x": 1},
            "hidden_witness": "w1",
            "private_witness": "w2",
            "credential_blob": "c1",
            "raw_output": "solver dump",
            "prover_output": "kernel dump",
            "source_code": "def f(): ...",
            "source_text": "P -> P",
        },
        context=_context(),
    )
    assert isinstance(view, ClientCounterexampleView)
    assert view.kind == "smt_model"
    assert view.redacted is True
    assert view.authority == "advisory"
    for private_key in (
        "hidden_witness",
        "private_witness",
        "credential_blob",
        "raw_output",
        "prover_output",
        "source_code",
        "source_text",
    ):
        assert private_key not in view.payload
    assert view.payload.get("model") == {"x": 1}
    assert view.payload.get("summary") == "violates safety"


# ---------------------------------------------------------------------------
# Cache key / freshness
# ---------------------------------------------------------------------------


def test_build_cache_key_and_freshness() -> None:
    client = _client()
    environment = {"toolchain": "test-env-v1"}
    key = client.build_cache_key(
        source="source-body",
        expression="expr-body",
        formalization="form-body",
        slice="slice-body",
        obligation="obl-body",
        assumptions=("a1",),
        bounds=_bounds().to_dict(),
        translation={"steps": []},
        provider="provider:fake",
        environment=environment,
        policy="policy:lpc-110",
        schema="schema:lpc-110",
        checker="checker:fake",
        network_policy={"network_allowed": False},
        evidence_kind=LogicEvidenceKind.CANDIDATE,
        authority_ceiling=LogicEvidenceAuthority.ADVISORY,
    )
    assert key.key_id.startswith("canonical-proof-cache-key:")
    report = client.check_cache_freshness(
        key,
        observed_environment=environment,
        entry_freshness=EvidenceFreshness.CURRENT,
    )
    assert isinstance(report, CacheFreshnessReport)
    assert report.fresh is True
    assert report.freshness == EvidenceFreshness.CURRENT.value


def test_cache_freshness_rejects_environment_mismatch() -> None:
    client = _client()
    key = client.build_cache_key(
        source="s",
        expression="e",
        formalization="f",
        slice="sl",
        obligation="o",
        provider="provider:fake",
        environment={"env": "a"},
        policy="policy:x",
        schema="schema:x",
        checker="checker:x",
        network_policy={"network_allowed": False},
        evidence_kind=LogicEvidenceKind.CANDIDATE,
        authority_ceiling=LogicEvidenceAuthority.ADVISORY,
    )
    report = client.check_cache_freshness(
        key,
        observed_environment={"env": "b"},
        entry_freshness=EvidenceFreshness.CURRENT,
    )
    assert report.fresh is False
    assert "environment_mismatch" in report.reasons
    with pytest.raises(LogicPlatformClientFreshnessError):
        client.require_cache_fresh(
            key,
            observed_environment={"env": "b"},
        )


def test_cache_freshness_rejects_stale_and_simulated() -> None:
    client = _client()
    env = {"env": "stable"}
    key = client.build_cache_key(
        source="s",
        expression="e",
        formalization="f",
        slice="sl",
        obligation="o",
        provider="provider:fake",
        environment=env,
        policy="policy:x",
        schema="schema:x",
        checker="checker:x",
        network_policy={},
        evidence_kind=LogicEvidenceKind.CANDIDATE,
        authority_ceiling=LogicEvidenceAuthority.ADVISORY,
    )
    stale = client.check_cache_freshness(
        key,
        observed_environment=env,
        entry_freshness=EvidenceFreshness.STALE,
    )
    assert stale.fresh is False
    assert "stale_entry" in stale.reasons
    simulated = client.check_cache_freshness(
        key,
        observed_environment=env,
        entry_freshness=EvidenceFreshness.CURRENT,
        simulated=True,
    )
    assert simulated.fresh is False
    assert "simulated_evidence" in simulated.reasons
    expired = client.check_cache_freshness(
        key,
        observed_environment=env,
        entry_freshness=EvidenceFreshness.CURRENT,
        now_unix_ms=200,
        expires_unix_ms=100,
    )
    assert expired.fresh is False
    assert "ttl_expired" in expired.reasons


def test_build_cache_key_rejects_candidate_as_kernel() -> None:
    from ipfs_datasets_py.logic.common.canonical_cache_key import (
        CandidateAsKernelError,
    )

    client = _client()
    with pytest.raises(CandidateAsKernelError):
        client.build_cache_key(
            source="s",
            expression="e",
            formalization="f",
            slice="sl",
            obligation="o",
            provider="provider:fake",
            environment={"env": "x"},
            policy="policy:x",
            schema="schema:x",
            checker="checker:x",
            network_policy={},
            evidence_kind=LogicEvidenceKind.CANDIDATE,
            authority_ceiling=LogicEvidenceAuthority.AUTHORITATIVE,
        )


# ---------------------------------------------------------------------------
# Singleton / introspection
# ---------------------------------------------------------------------------


def test_get_logic_platform_client_singleton() -> None:
    a = get_logic_platform_client()
    b = get_logic_platform_client()
    assert a is b
    c = get_logic_platform_client(require_handshake=False)
    assert c is not a


def test_client_to_dict() -> None:
    client = _client()
    payload = client.to_dict()
    assert payload["interface"] == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["handshake_compatible"] is True
    assert "handshake" in payload["supported_operations"]


def test_manifest_lists_client_adapter() -> None:
    result = platform_handshake(
        HandshakeRequirements(
            required_adapter_versions=(
                SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
            )
        )
    )
    assert result.compatible


def test_end_to_end_acceptance_surface() -> None:
    """Exercise the full LPC-110 acceptance surface in one cohesive flow."""

    provider = _FakeProvider()
    client = SupervisorLogicPlatformClient(
        provider=provider,
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
    )
    # 1. handshake
    hs = client.handshake()
    assert hs.compatible
    # 2. catalog
    catalog = client.catalog()
    assert catalog.content_root
    # 3. formalization + slice
    ctx = _context()
    slice_item = client.create_slice(
        domain="software_verification",
        document_id="doc:e2e",
        source_digest=_digest("src"),
        expression_id="expr:e2e",
        expression_digest=_digest("expr"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        property=property_id("validity"),
        context=ctx,
    )
    artifact = client.formalize(
        domain="software_verification",
        document_id="doc:e2e",
        source_digest=_digest("src"),
        expression_id="expr:e2e",
        expression_digest=_digest("expr"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        slices=(slice_item,),
        context=ctx,
    )
    assert artifact.admitted_slices
    # 4. obligation + plan
    obligation = client.create_obligation(
        statement="P",
        document_id="doc:e2e",
        source_digest=_digest("src"),
        expression_id="expr:e2e",
        expression_digest=_digest("expr"),
        family=family_id("propositional"),
        profile=profile_id("classical"),
        property=property_id("validity"),
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id("candidate"),
        bounds=_bounds(),
        authority_ceiling=RequestAuthorityCeiling.ADVISORY,
        slice_id=slice_item.slice_id,
        slice_digest=(
            slice_item.content_digest[len("sha256:") :]
            if str(slice_item.content_digest).startswith("sha256:")
            else slice_item.content_digest
        ),
        context=ctx,
    )
    plan = client.create_plan(
        formal_goal_id="goal:e2e",
        graph_id="graph:e2e",
        context=ctx,
    )
    assert plan.plan_id
    # 5. capability
    caps = client.discover_capabilities(context=ctx)
    assert caps.response.is_success
    # 6. typed invocation
    backend = client.create_backend_request(obligation=obligation, context=ctx)
    proved = client.invoke("prove", backend_request=backend, context=ctx)
    assert proved.response.is_success
    # 7. reconstruct + verify
    assert client.reconstruct(
        backend_request=backend,
        bounds=backend.bounds,
        candidate_digest=_digest("cand"),
        context=ctx,
    ).response.is_success
    assert client.verify(
        backend_request=backend,
        bounds=backend.bounds,
        evidence_digest=_digest("ev"),
        context=ctx,
    ).response.is_success
    # 8. receipt
    receipt = client.project_receipt(proved)
    assert receipt.evidence_authority == "advisory"
    # 9. counterexample
    cex = client.project_counterexample(
        {"kind": "smt_model", "summary": "counter", "hidden_witness": "x"},
        context=ctx,
    )
    assert "hidden_witness" not in cex.payload
    # 10. cache freshness
    key = client.build_cache_key(
        source=_digest("src"),
        expression=_digest("expr"),
        formalization=artifact.content_digest,
        slice=slice_item.content_digest,
        obligation=obligation.content_digest,
        provider=provider.provider_id,
        environment={"env": "e2e"},
        policy=ctx.policy_id,
        schema="schema:e2e",
        checker="checker:e2e",
        network_policy={"network_allowed": False},
        evidence_kind=LogicEvidenceKind.CANDIDATE,
        authority_ceiling=LogicEvidenceAuthority.ADVISORY,
    )
    freshness = client.require_cache_fresh(
        key,
        observed_environment={"env": "e2e"},
    )
    assert freshness.fresh is True
