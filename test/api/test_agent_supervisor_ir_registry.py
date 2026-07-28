from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    PinnedArtifactRef,
    ReferenceAuthority,
    canonical_artifact_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IR_CAPABILITY_SCHEMA,
    IR_REGISTRY_VERSION,
    IRCapability,
    IRDeclaredAuthority,
    IRFailureCode,
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IROperation,
    IRRegistry,
    IRRegistryBounds,
    IRRegistryError,
    IRReviewState,
    IRSchemaSupport,
    IRTrustState,
    SUPPORTED_IR_SCHEMAS,
    create_default_ir_registry,
    deterministic_ir_fixture,
    normalize_ir_family,
)


def _request(
    reference: PinnedArtifactRef,
    family: IRFamily | str,
    **changes: object,
) -> IRLoadRequest:
    values: dict[str, object] = {
        "reference": reference,
        "family": family,
        "producer_configuration_id": "configuration:deterministic@1",
    }
    values.update(changes)
    return IRLoadRequest(**values)


def _registered_fixture(
    family: IRFamily | str = IRFamily.INTENT,
    **fixture_values: object,
) -> tuple[IRRegistry, PinnedArtifactRef, bytes]:
    reference, encoded = deterministic_ir_fixture(family, **fixture_values)
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    return registry, reference, encoded


def _reference_for_payload(
    payload: object,
    *,
    family: IRFamily = IRFamily.INTENT,
    reference_schema: str | None = None,
    reference_version: str = "1",
    producer_id: str = "producer:deterministic-ir-fixture",
    authority: ReferenceAuthority = ReferenceAuthority.VERIFIED,
    artifact_id: str = "fixture:custom",
) -> tuple[PinnedArtifactRef, bytes]:
    encoded = canonical_artifact_bytes(payload)
    schema = reference_schema or next(
        item.schema for item in SUPPORTED_IR_SCHEMAS if item.family is family
    )
    return (
        PinnedArtifactRef.from_canonical_bytes(
            encoded,
            artifact_id=artifact_id,
            artifact_kind=family.value,
            artifact_schema=schema,
            artifact_schema_version=reference_version,
            producer_id=producer_id,
            authority=authority,
        ),
        encoded,
    )


def _support_for(reference: PinnedArtifactRef, family: IRFamily) -> IRSchemaSupport:
    return next(
        item
        for item in SUPPORTED_IR_SCHEMAS
        if item.matches(reference, family)
    )


def test_default_capability_discovery_is_complete_canonical_and_side_effect_free() -> None:
    registry = create_default_ir_registry(include_optional_ipfs_datasets=True)

    capabilities = registry.discover_capabilities()
    assert [item.provider_id for item in capabilities] == [
        "supervisor-local-ir",
        "ipfs-datasets-ir",
    ]
    assert capabilities[0].remote is False
    assert capabilities[1].remote is True
    assert all(set(item.families) == set(IRFamily) for item in capabilities)
    assert all(set(item.operations) == set(IROperation) for item in capabilities)
    assert {item.family for item in registry.supported_schemas()} == set(IRFamily)
    assert registry.supported_schemas("intent") == registry.supported_schemas(
        IRFamily.INTENT
    )

    record = capabilities[1].to_dict()
    assert record["schema"] == IR_CAPABILITY_SCHEMA
    assert record["registry_version"] == IR_REGISTRY_VERSION
    assert record["lazy"] is True
    assert record["families"] == sorted(item.value for item in IRFamily)
    assert record["operations"] == sorted(item.value for item in IROperation)
    assert registry.discover_capabilities("legal", "load") == capabilities


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("core", IRFamily.IR_CORE),
        ("shared_ir_core", IRFamily.IR_CORE),
        ("formal", IRFamily.FORMALIZATION),
        ("intentir", IRFamily.INTENT),
        ("legal-ir", IRFamily.LEGAL),
        ("securityir", IRFamily.SECURITY),
    ],
)
def test_family_aliases_are_explicit_and_unknown_families_are_rejected(
    spelling: str, expected: IRFamily
) -> None:
    assert normalize_ir_family(spelling) is expected

    with pytest.raises(IRRegistryError, match="unsupported IR family"):
        normalize_ir_family("ambient-policy")


@pytest.mark.parametrize("family", tuple(IRFamily))
def test_exact_local_artifact_verifies_for_every_supported_family(
    family: IRFamily,
) -> None:
    declarations = ({"declaration_id": f"{family.value}:declaration:1"},)
    formal_views = ({"view_id": f"{family.value}:view:1", "logic": "tdfol"},)
    claims = ({"claim_id": f"{family.value}:claim:1"},)
    assumptions = ({"assumption_id": f"{family.value}:assumption:1"},)
    obligations = ({"obligation_id": f"{family.value}:obligation:1"},)
    registry, reference, encoded = _registered_fixture(
        family,
        declarations=declarations,
        formal_views=formal_views,
        claims=claims,
        assumptions=assumptions,
        obligations=obligations,
    )

    result = registry.load(_request(reference, family))
    artifact = result.require_artifact()

    assert result.status is IRLoadStatus.VERIFIED
    assert result.successful is True
    assert result.fail_closed is False
    assert artifact.reference == reference
    assert artifact.root_reference == reference
    assert artifact.family is family
    assert artifact.canonical_bytes == encoded
    assert artifact.producer_configuration_id == "configuration:deterministic@1"
    assert artifact.provenance == (
        MappingProxyType(
            {
                "source_id": "source:deterministic-fixture",
                "span_id": "span:0",
            }
        ),
    )
    assert artifact.review_state is IRReviewState.REVIEWED
    assert artifact.trust_state is IRTrustState.TRUSTED
    assert artifact.declared_authority is IRDeclaredAuthority.VERIFIED
    assert artifact.provider_id == "supervisor-local-ir"
    assert artifact.body_cached is False
    assert reference.verify_canonical_bytes(artifact.canonical_bytes)
    with pytest.raises(TypeError):
        bool(result)
    with pytest.raises(TypeError):
        artifact.payload["freshness"] = "stale"  # type: ignore[index]


def test_local_path_is_loaded_lazily_and_reverified_after_replacement(
    tmp_path: Path,
) -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.LEGAL)
    path = tmp_path / "legal-ir.json"
    path.write_bytes(encoded)
    registry = IRRegistry()
    registry.register_local_path(reference, path.resolve())

    assert registry.load(_request(reference, IRFamily.LEGAL)).successful

    changed = encoded.replace(b'"fresh"', b'"stale"')
    assert len(changed) == len(encoded)
    path.write_bytes(changed)
    result = registry.load(_request(reference, IRFamily.LEGAL))

    assert result.status is IRLoadStatus.QUARANTINED
    assert result.failure is not None
    assert "CIDv1" in result.failure.reason


def test_local_registration_rejects_unpinned_bytes_and_relative_paths(
    tmp_path: Path,
) -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.SECURITY)
    registry = IRRegistry()

    with pytest.raises(IRRegistryError, match="do not match pinned"):
        registry.register_local_artifact(reference, encoded + b" ")
    with pytest.raises(IRRegistryError, match="must be absolute"):
        registry.register_local_path(reference, Path("relative.json"))

    absent = tmp_path / "absent.json"
    registry.register_local_path(reference, absent.resolve())
    result = registry.load(_request(reference, IRFamily.SECURITY))
    assert result.status is IRLoadStatus.UNAVAILABLE


def test_schema_version_and_family_are_verified_against_reference_and_request() -> None:
    registry, reference, _ = _registered_fixture(IRFamily.INTENT)

    wrong_family = registry.load(_request(reference, IRFamily.LEGAL))
    assert wrong_family.status is IRLoadStatus.UNSUPPORTED

    payload = json.loads(
        deterministic_ir_fixture(IRFamily.INTENT)[1].decode("utf-8")
    )
    payload["schema"] = "intent-ir@future"
    mismatched_reference, encoded = _reference_for_payload(payload)
    mismatch_registry = IRRegistry()
    mismatch_registry.register_local_artifact(mismatched_reference, encoded)
    mismatch = mismatch_registry.load(
        _request(mismatched_reference, IRFamily.INTENT)
    )
    assert mismatch.status is IRLoadStatus.UNSUPPORTED

    unsupported_reference, unsupported_bytes = deterministic_ir_fixture(
        IRFamily.INTENT,
        schema="intent-ir@2",
        schema_version="2",
    )
    unsupported_registry = IRRegistry()
    unsupported_registry.register_local_artifact(
        unsupported_reference, unsupported_bytes
    )
    unsupported = unsupported_registry.load(
        _request(unsupported_reference, IRFamily.INTENT)
    )
    assert unsupported.status is IRLoadStatus.UNSUPPORTED
    assert unsupported.fail_closed


def test_missing_provider_and_missing_artifact_are_typed_distinctly() -> None:
    reference, _ = deterministic_ir_fixture(IRFamily.FORMALIZATION)
    registry = IRRegistry()

    missing = registry.load(_request(reference, IRFamily.FORMALIZATION))
    assert missing.status is IRLoadStatus.UNAVAILABLE
    assert missing.failure is not None
    assert missing.failure.code is IRFailureCode.UNAVAILABLE
    assert missing.failure.details == ("supervisor-local-ir: FileNotFoundError",)

    unknown_provider = registry.load(
        _request(
            reference,
            IRFamily.FORMALIZATION,
            provider_id="provider:not-declared",
        )
    )
    assert unknown_provider.status is IRLoadStatus.UNSUPPORTED
    assert unknown_provider.failure is not None
    assert unknown_provider.failure.provider_id == "provider:not-declared"


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"provenance": []}, IRLoadStatus.PARTIAL),
        ({"trust": None}, IRLoadStatus.PARTIAL),
        ({"review": None}, IRLoadStatus.PARTIAL),
        ({"authority": None}, IRLoadStatus.PARTIAL),
        ({"partial": True}, IRLoadStatus.PARTIAL),
        ({"truncated": True}, IRLoadStatus.PARTIAL),
        ({"freshness": "expired"}, IRLoadStatus.STALE),
        ({"trust": {"state": "quarantined"}}, IRLoadStatus.QUARANTINED),
        ({"trust": {"state": "untrusted"}}, IRLoadStatus.QUARANTINED),
        ({"review": {"status": "rejected"}}, IRLoadStatus.QUARANTINED),
        ({"review": {"status": "unreviewed"}}, IRLoadStatus.QUARANTINED),
        ({"ambiguities": ["two applicable declarations"]}, IRLoadStatus.AMBIGUOUS),
        (
            {"contradictions": ["allow and deny the same effect"]},
            IRLoadStatus.CONTRADICTION,
        ),
    ],
)
def test_every_declared_failure_state_is_typed_and_required_inputs_fail_closed(
    updates: dict[str, object],
    expected: IRLoadStatus,
) -> None:
    registry, reference, _ = _registered_fixture(
        IRFamily.LEGAL, updates=updates
    )
    result = registry.load(_request(reference, IRFamily.LEGAL))

    assert result.status is expected
    assert result.artifact is None
    assert result.failure is not None
    assert result.failure.code.value == expected.value
    assert result.failure.required is True
    assert result.fail_closed is True
    with pytest.raises(IRRegistryError, match="failed closed"):
        result.require_artifact()
    with pytest.raises(TypeError):
        bool(result)


def test_optional_failures_remain_typed_but_do_not_claim_fail_closed() -> None:
    reference, _ = deterministic_ir_fixture(IRFamily.IR_CORE)
    result = IRRegistry().load(
        _request(reference, IRFamily.IR_CORE, required=False)
    )

    assert result.status is IRLoadStatus.UNAVAILABLE
    assert result.successful is False
    assert result.fail_closed is False
    assert result.failure is not None
    assert result.failure.required is False


def test_producer_configuration_provenance_and_authority_are_exact() -> None:
    registry, reference, _ = _registered_fixture(IRFamily.SECURITY)
    stale = registry.load(
        _request(
            reference,
            IRFamily.SECURITY,
            producer_configuration_id="configuration:other@2",
        )
    )
    assert stale.status is IRLoadStatus.STALE

    wrong_producer_registry, wrong_producer_ref, _ = _registered_fixture(
        IRFamily.SECURITY,
        producer_id="producer:payload",
    )
    forged_ref = replace(
        wrong_producer_ref,
        producer_id="producer:pinned",
    )
    producer_result = wrong_producer_registry.load(
        _request(forged_ref, IRFamily.SECURITY)
    )
    assert producer_result.status is IRLoadStatus.QUARANTINED

    authority_registry, authority_ref, authority_bytes = _registered_fixture(
        IRFamily.SECURITY,
        authority=IRDeclaredAuthority.VERIFIED,
        reference_authority=ReferenceAuthority.ADVISORY,
    )
    assert authority_ref.verify_canonical_bytes(authority_bytes)
    authority_result = authority_registry.load(
        _request(authority_ref, IRFamily.SECURITY)
    )
    assert authority_result.status is IRLoadStatus.QUARANTINED
    assert authority_result.failure is not None
    assert "authority exceeds" in authority_result.failure.reason


def test_canonical_bytes_cid_and_digest_are_reverified_on_every_provider_load() -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.INTENT)
    alternate = encoded.replace(
        b'"span_id":"span:0"', b'"span_id":"span:1"'
    )
    assert len(alternate) == len(encoded)
    assert canonical_artifact_bytes(json.loads(alternate)) == alternate
    registry = IRRegistry()
    registry.register_provider(
        IRCapability(
            provider_id="provider:corrupt",
            provider_version="1",
            capability_revision="corrupt@1",
            schemas=(_support_for(reference, IRFamily.INTENT),),
            operations=(IROperation.LOAD,),
        ),
        loader=lambda _request: alternate,
    )

    result = registry.load(
        _request(
            reference,
            IRFamily.INTENT,
            provider_id="provider:corrupt",
        )
    )

    assert result.status is IRLoadStatus.QUARANTINED
    assert result.failure is not None
    assert "canonical bytes, CIDv1, or supervisor digest" in result.failure.reason


def test_root_membership_requires_both_pinned_root_identities() -> None:
    root_reference, _ = deterministic_ir_fixture(
        IRFamily.LEGAL,
        artifact_id="fixture:legal-root",
    )
    registry, child_reference, _ = _registered_fixture(
        IRFamily.LEGAL,
        artifact_id="fixture:legal-child",
        root_reference=root_reference,
    )
    verified = registry.load(
        _request(
            child_reference,
            IRFamily.LEGAL,
            root_reference=root_reference,
        )
    )
    assert verified.status is IRLoadStatus.VERIFIED
    assert verified.require_artifact().root_reference == root_reference

    missing_registry, missing_ref, _ = _registered_fixture(
        IRFamily.LEGAL,
        artifact_id="fixture:no-membership",
    )
    missing = missing_registry.load(
        _request(missing_ref, IRFamily.LEGAL, root_reference=root_reference)
    )
    assert missing.status is IRLoadStatus.PARTIAL

    other_root, _ = deterministic_ir_fixture(
        IRFamily.LEGAL,
        artifact_id="fixture:other-root",
        updates={"claims": [{"claim_id": "other"}]},
    )
    stale = registry.load(
        _request(
            child_reference,
            IRFamily.LEGAL,
            root_reference=other_root,
        )
    )
    assert stale.status is IRLoadStatus.STALE


@pytest.mark.parametrize(
    ("bounds", "expected_fragment"),
    [
        (IRRegistryBounds(max_artifact_bytes=64), "pre-load bound"),
        (IRRegistryBounds(max_items=4), "structure is outside bounds"),
        (IRRegistryBounds(max_depth=2), "structure is outside bounds"),
        (IRRegistryBounds(max_text_bytes=8), "structure is outside bounds"),
    ],
)
def test_request_byte_count_depth_and_text_bounds_are_typed(
    bounds: IRRegistryBounds,
    expected_fragment: str,
) -> None:
    registry, reference, _ = _registered_fixture(IRFamily.IR_CORE)
    result = registry.load(
        _request(reference, IRFamily.IR_CORE, bounds=bounds)
    )

    assert result.status is IRLoadStatus.BOUNDS
    assert result.failure is not None
    assert expected_fragment in result.failure.reason


def test_registry_global_byte_bound_cannot_be_broadened_by_request() -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.INTENT)
    registry = IRRegistry(bounds=IRRegistryBounds(max_artifact_bytes=64))
    registry.register_local_artifact(reference, encoded)

    result = registry.load(
        _request(
            reference,
            IRFamily.INTENT,
            bounds=IRRegistryBounds(max_artifact_bytes=len(encoded) + 1),
        )
    )
    assert result.status is IRLoadStatus.BOUNDS


def test_lazy_provider_factory_is_not_called_by_capability_or_schema_discovery() -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.INTENT)
    support = _support_for(reference, IRFamily.INTENT)
    calls: list[str] = []

    def factory():
        calls.append("factory")

        def load(request: IRLoadRequest) -> bytes:
            calls.append(f"load:{request.reference.artifact_id}")
            return encoded

        return load

    registry = IRRegistry()
    registry.register_lazy_provider(
        IRCapability(
            provider_id="provider:lazy",
            provider_version="1",
            capability_revision="lazy@1",
            schemas=(support,),
            operations=(IROperation.LOAD,),
            remote=True,
        ),
        factory=factory,
    )

    assert [item.provider_id for item in registry.discover_capabilities()] == [
        "supervisor-local-ir",
        "provider:lazy",
    ]
    assert registry.supported_schemas(IRFamily.INTENT)
    assert calls == []

    request = _request(
        reference,
        IRFamily.INTENT,
        provider_id="provider:lazy",
    )
    assert registry.load(request).status is IRLoadStatus.VERIFIED
    assert registry.load(request).status is IRLoadStatus.VERIFIED
    assert calls == [
        "factory",
        "load:fixture:intent_ir",
        "load:fixture:intent_ir",
    ]


def test_optional_module_is_imported_only_by_explicit_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.FORMALIZATION)
    support = _support_for(reference, IRFamily.FORMALIZATION)
    imported: list[str] = []

    class Provider:
        def load_ir_artifact(self, _request: IRLoadRequest) -> bytes:
            return encoded

    module = type("FixtureModule", (), {"provider": Provider})()

    def import_module(name: str):
        imported.append(name)
        assert name == "fixture_optional_ir_provider"
        return module

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.ir_registry.importlib.import_module",
        import_module,
    )
    registry = IRRegistry()
    registry.register_optional_module(
        IRCapability(
            provider_id="provider:optional-module",
            provider_version="1",
            capability_revision="module@1",
            schemas=(support,),
            operations=(IROperation.LOAD,),
            remote=True,
        ),
        module_name="fixture_optional_ir_provider",
        attribute="provider",
    )

    registry.discover_capabilities(IRFamily.FORMALIZATION)
    registry.supported_schemas(IRFamily.FORMALIZATION)
    assert imported == []

    result = registry.load(
        _request(
            reference,
            IRFamily.FORMALIZATION,
            provider_id="provider:optional-module",
        )
    )
    assert result.status is IRLoadStatus.VERIFIED
    assert imported == ["fixture_optional_ir_provider"]


def test_async_provider_is_typed_unavailable_for_sync_and_supported_async() -> None:
    reference, encoded = deterministic_ir_fixture(IRFamily.SECURITY)
    support = _support_for(reference, IRFamily.SECURITY)

    async def load(_request: IRLoadRequest) -> bytes:
        return encoded

    registry = IRRegistry()
    registry.register_lazy_provider(
        IRCapability(
            provider_id="provider:async",
            provider_version="1",
            capability_revision="async@1",
            schemas=(support,),
            operations=(IROperation.LOAD,),
            remote=True,
        ),
        factory=lambda: load,
    )
    request = _request(
        reference,
        IRFamily.SECURITY,
        provider_id="provider:async",
    )

    synchronous = registry.load(request)
    asynchronous = asyncio.run(registry.load_async(request))

    assert synchronous.status is IRLoadStatus.UNAVAILABLE
    assert synchronous.failure is not None
    assert synchronous.failure.details == (
        "provider:async: asynchronous loader",
    )
    assert asynchronous.status is IRLoadStatus.VERIFIED


def test_analysis_transport_only_locates_then_exact_bytes_are_reverified() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.analysis_transport import (
        AnalysisTransportStatus,
    )

    reference, encoded = deterministic_ir_fixture(IRFamily.SECURITY)
    transport_requests: list[object] = []
    resolver_calls: list[MappingProxyType] = []

    class Transport:
        async def dispatch(self, request: object) -> object:
            transport_requests.append(request)
            return SimpleNamespace(
                status=AnalysisTransportStatus.COMPLETED,
                truncated=False,
                evidence_references=(
                    MappingProxyType(
                        {
                            "artifact_id": reference.artifact_id,
                            "cid": reference.cid_v1,
                            "digest": reference.supervisor_digest,
                            "uri": "ipfs://exact-fixture",
                        }
                    ),
                ),
            )

    def resolve(
        location: MappingProxyType, request: IRLoadRequest
    ) -> bytes:
        resolver_calls.append(location)
        assert request.reference == reference
        return encoded

    registry = IRRegistry()
    registry.register_analysis_transport(
        IRCapability(
            provider_id="provider:analysis-transport",
            provider_version="1",
            capability_revision="transport@1",
            remote=True,
        ),
        transport=Transport(),
        resolver=resolve,
    )
    result = asyncio.run(
        registry.load_async(
            _request(
                reference,
                IRFamily.SECURITY,
                provider_id="provider:analysis-transport",
            )
        )
    )

    assert result.status is IRLoadStatus.VERIFIED
    assert result.require_artifact().declared_authority is (
        IRDeclaredAuthority.VERIFIED
    )
    assert len(transport_requests) == 1
    assert len(resolver_calls) == 1
    assert "content" not in resolver_calls[0]
    assert "body" not in resolver_calls[0]


def test_malformed_registry_contracts_raise_instead_of_becoming_provider_outcomes() -> None:
    reference, _ = deterministic_ir_fixture(IRFamily.INTENT)

    with pytest.raises(IRRegistryError, match="must be an integer"):
        IRRegistryBounds(max_items=True)
    with pytest.raises(IRRegistryError, match="non-empty and unique"):
        IRCapability(
            provider_id="provider",
            provider_version="1",
            capability_revision="1",
            schemas=(),
        )
    with pytest.raises(IRRegistryError, match="must be a PinnedArtifactRef"):
        IRLoadRequest(reference={}, family=IRFamily.INTENT)  # type: ignore[arg-type]
    with pytest.raises(IRRegistryError, match="must be IRLoadRequest"):
        IRRegistry().load(reference)  # type: ignore[arg-type]


def test_fresh_interpreter_import_and_discovery_import_no_optional_provider_and_start_no_process() -> None:
    script = r'''
import builtins
import importlib
import json
from pathlib import Path
import sys
import types

events = []
def audit(event, args):
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
    }:
        events.append(event)
        raise AssertionError("process start during import/discovery: " + event)
sys.addaudithook(audit)

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
        raise AssertionError("optional provider imported: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

# Import the module in its real package namespace while deliberately avoiding
# the repository's unrelated eager package initializer.  ASI-125 owns the
# registry module, not the legacy aggregate exports in agent_supervisor.__init__.
root = Path.cwd()
top_package = types.ModuleType("ipfs_accelerate_py")
top_package.__path__ = [str(root / "ipfs_accelerate_py")]
supervisor_package = types.ModuleType("ipfs_accelerate_py.agent_supervisor")
supervisor_package.__path__ = [
    str(root / "ipfs_accelerate_py" / "agent_supervisor")
]
sys.modules["ipfs_accelerate_py"] = top_package
sys.modules["ipfs_accelerate_py.agent_supervisor"] = supervisor_package
module = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.proof.ir_registry"
)

registry = module.create_default_ir_registry(
    include_optional_ipfs_datasets=True
)
capabilities = registry.discover_capabilities()
schemas = registry.supported_schemas()
print(json.dumps({
    "providers": [item.provider_id for item in capabilities],
    "families": sorted({family.value for item in capabilities for family in item.families}),
    "operations": sorted({operation.value for item in capabilities for operation in item.operations}),
    "schema_count": len(schemas),
    "optional_loaded": any(
        name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.")
        for name in sys.modules
    ),
    "process_events": events,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)

    assert report["providers"] == [
        "supervisor-local-ir",
        "ipfs-datasets-ir",
    ]
    assert report["families"] == sorted(item.value for item in IRFamily)
    assert report["operations"] == sorted(item.value for item in IROperation)
    assert report["schema_count"] == len(SUPPORTED_IR_SCHEMAS)
    assert report["optional_loaded"] is False
    assert report["process_events"] == []
