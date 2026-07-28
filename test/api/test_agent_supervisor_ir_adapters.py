from __future__ import annotations

import importlib
import subprocess
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import (
    IRAdapterBounds,
    IRAdapterError,
    IRAdapterRegistry,
    IRAdapterStatus,
    IRNodeKind,
    NormalizedResultAuthority,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRDeclaredAuthority,
    IRFailure,
    IRFailureCode,
    IRFamily,
    IRLoadRequest,
    IRLoadResult,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)


_DECLARATION_AUTHORITY = {
    IRFamily.IR_CORE: NormalizedResultAuthority.VERIFIED_INPUT,
    IRFamily.FORMALIZATION: NormalizedResultAuthority.VERIFIED_INPUT,
    IRFamily.INTENT: NormalizedResultAuthority.DESCRIPTIVE_INPUT,
    IRFamily.LEGAL: NormalizedResultAuthority.CONSTRAINT_INPUT,
    IRFamily.SECURITY: NormalizedResultAuthority.POLICY_INPUT,
}


def _verified_load(
    family: IRFamily,
    *,
    declarations: tuple[dict[str, object], ...] = (),
    formal_views: tuple[dict[str, object], ...] = (),
    claims: tuple[dict[str, object], ...] = (),
    assumptions: tuple[dict[str, object], ...] = (),
    obligations: tuple[dict[str, object], ...] = (),
    updates: dict[str, object] | None = None,
) -> IRLoadResult:
    reference, encoded = deterministic_ir_fixture(
        family,
        declarations=declarations,
        formal_views=formal_views,
        claims=claims,
        assumptions=assumptions,
        obligations=obligations,
        updates=updates,
    )
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    result = registry.load(IRLoadRequest(reference=reference, family=family))
    assert result.status is IRLoadStatus.VERIFIED
    return result


@pytest.mark.parametrize("family", tuple(IRFamily))
def test_every_ir_family_normalizes_all_shared_sections_with_bounded_authority(
    family: IRFamily,
) -> None:
    loaded = _verified_load(
        family,
        declarations=(
            {
                "declaration_id": "declaration:z",
                "kind": "policy" if family is IRFamily.SECURITY else "statement",
                "grounded": True,
            },
            {
                "declaration_id": "declaration:a",
                "kind": "statement",
                "grounded": True,
            },
        ),
        formal_views=(
            {
                "view_id": "view:one",
                "view_kind": "first_order",
                "grounded": True,
            },
        ),
        claims=(
            {
                "claim_id": "claim:one",
                "claim_kind": "assertion",
                "grounded": True,
            },
        ),
        assumptions=(
            {
                "assumption_id": "assumption:one",
                "kind": "environment",
                "grounded": True,
            },
        ),
        obligations=(
            {
                "obligation_id": "obligation:one",
                "kind": "proof",
                "grounded": True,
            },
        ),
        updates={
            "result_authority": [
                {
                    "result_id": "result-authority:one",
                    "kind": "producer_declaration",
                    "grounded": True,
                }
            ]
        },
    )

    result = IRAdapterRegistry().normalize(loaded)
    normalized = result.require_artifact()

    assert result.status is IRAdapterStatus.NORMALIZED
    assert normalized.family is family
    assert [item.node_id for item in normalized.declarations] == [
        "declaration:a",
        "declaration:z",
    ]
    assert tuple(item.node_kind for item in normalized.nodes) == (
        IRNodeKind.DECLARATION,
        IRNodeKind.DECLARATION,
        IRNodeKind.FORMAL_VIEW,
        IRNodeKind.CLAIM,
        IRNodeKind.ASSUMPTION,
        IRNodeKind.OBLIGATION,
        IRNodeKind.RESULT_AUTHORITY,
    )
    assert all(
        item.result_authority is _DECLARATION_AUTHORITY[family]
        for item in normalized.declarations
    )
    assert normalized.formal_views[0].result_authority is (
        NormalizedResultAuthority.PROPOSAL_ONLY
    )
    assert normalized.assumptions[0].result_authority is (
        NormalizedResultAuthority.CONTEXT_ONLY
    )
    assert normalized.result_authority[0].result_authority is (
        NormalizedResultAuthority.VERIFIED_INPUT
    )
    assert normalized.proof_obligations == normalized.obligations
    assert not normalized.grants_execution_authority
    assert not normalized.source_corpus_copied
    assert all(not item.result_authority.grants_execution for item in normalized.nodes)
    assert normalized.source_cid_v1 == loaded.require_artifact().reference.cid_v1
    assert normalized.root_supervisor_digest == (
        loaded.require_artifact().root_reference.supervisor_digest
    )


@pytest.mark.parametrize("code", tuple(IRFailureCode))
def test_registry_failures_pass_through_as_typed_fail_closed_adapter_results(
    code: IRFailureCode,
) -> None:
    reference, _ = deterministic_ir_fixture(IRFamily.INTENT)
    request = IRLoadRequest(
        reference=reference,
        family=IRFamily.INTENT,
        required=True,
    )
    load_result = IRLoadResult(
        status=IRLoadStatus(code.value),
        request=request,
        failure=IRFailure(
            code=code,
            reason=f"fixture {code.value}",
            required=True,
            artifact_id=reference.artifact_id,
            provider_id="provider:fixture",
        ),
    )

    result = IRAdapterRegistry().normalize(load_result)

    assert result.status.value == code.value
    assert result.failure is load_result.failure
    assert result.fail_closed
    with pytest.raises(IRAdapterError, match=code.value):
        result.require_artifact()
    with pytest.raises(TypeError, match="no truth value"):
        bool(result)

    optional = IRAdapterRegistry().normalize(load_result, required=False)
    assert optional.failure is not None
    assert optional.failure.code is code
    assert optional.failure.required is False
    assert not optional.fail_closed


def test_normalization_keeps_only_compact_references_and_never_copies_corpus() -> None:
    marker = "SECRET-CORPUS-MUST-NOT-LEAK"
    loaded = _verified_load(
        IRFamily.LEGAL,
        declarations=(
            {
                "declaration_id": "norm:one",
                "kind": "obligation",
                "text": marker,
                "source_text": marker,
                "corpus": [marker],
                "graph": {"body": marker},
                "metadata": {
                    "body": marker,
                    "safe_label": "compact",
                    "nested": {"raw_output": marker, "revision": "revision:1"},
                },
                "source_references": [
                    {
                        "source_id": "source:statute",
                        "span_id": "span:10-20",
                        "digest": "sha256:" + "a" * 64,
                        "summary": marker,
                        "content": marker,
                    }
                ],
                "provenance": [
                    {
                        "record_id": "provenance:one",
                        "producer_id": "producer:reviewed-law",
                        "configuration_id": "configuration:law@1",
                        "source_text": marker,
                    }
                ],
            },
        ),
    )

    normalized = IRAdapterRegistry().normalize(loaded).require_artifact()
    node = normalized.declarations[0]
    serialized = normalized.canonical_bytes.decode("utf-8")

    assert marker not in serialized
    assert node.attributes["metadata"]["safe_label"] == "compact"
    assert "body" not in node.attributes["metadata"]
    assert "raw_output" not in node.attributes["metadata"]["nested"]
    assert dict(node.source_references[0]) == {
        "digest": "sha256:" + "a" * 64,
        "source_id": "source:statute",
        "span_id": "span:10-20",
    }
    assert dict(node.provenance_references[0]) == {
        "configuration_id": "configuration:law@1",
        "producer_id": "producer:reviewed-law",
        "record_id": "provenance:one",
    }
    assert normalized.to_dict()["source_corpus_copied"] is False
    assert normalized.to_dict()["grants_execution_authority"] is False


def test_identical_duplicate_ids_are_ambiguous_and_conflicting_ids_contradict() -> None:
    duplicate = {"declaration_id": "declaration:duplicate", "kind": "goal"}
    ambiguous = IRAdapterRegistry().normalize(
        _verified_load(
            IRFamily.INTENT,
            declarations=(duplicate, duplicate),
        )
    )
    assert ambiguous.status is IRAdapterStatus.AMBIGUOUS
    assert ambiguous.failure is not None
    assert ambiguous.failure.code is IRFailureCode.AMBIGUOUS
    assert ambiguous.fail_closed

    contradiction = IRAdapterRegistry().normalize(
        _verified_load(
            IRFamily.INTENT,
            declarations=(
                duplicate,
                {
                    "declaration_id": "declaration:duplicate",
                    "kind": "invariant",
                },
            ),
        )
    )
    assert contradiction.status is IRAdapterStatus.CONTRADICTION
    assert contradiction.failure is not None
    assert contradiction.failure.code is IRFailureCode.CONTRADICTION
    assert contradiction.fail_closed


def test_malformed_sections_missing_ids_and_every_adapter_bound_fail_closed() -> None:
    malformed = IRAdapterRegistry().normalize(
        _verified_load(IRFamily.SECURITY, updates={"claims": {"claim_id": "bad"}})
    )
    assert malformed.status is IRAdapterStatus.PARTIAL

    missing_id = IRAdapterRegistry().normalize(
        _verified_load(IRFamily.SECURITY, claims=({"kind": "threat"},))
    )
    assert missing_id.status is IRAdapterStatus.PARTIAL

    two_nodes = _verified_load(
        IRFamily.SECURITY,
        declarations=(
            {"declaration_id": "declaration:one", "kind": "asset"},
            {"declaration_id": "declaration:two", "kind": "resource"},
        ),
    )
    node_count = IRAdapterRegistry(
        bounds=IRAdapterBounds(max_nodes=1)
    ).normalize(two_nodes)
    assert node_count.status is IRAdapterStatus.BOUNDS

    large_node = _verified_load(
        IRFamily.SECURITY,
        declarations=(
            {
                "declaration_id": "declaration:large",
                "kind": "policy",
                "label": "x" * 1024,
            },
        ),
    )
    node_bytes = IRAdapterRegistry(
        bounds=IRAdapterBounds(max_node_bytes=256)
    ).normalize(large_node)
    assert node_bytes.status is IRAdapterStatus.BOUNDS

    too_many_references = _verified_load(
        IRFamily.SECURITY,
        declarations=(
            {
                "declaration_id": "declaration:references",
                "kind": "policy",
                "source_references": ("source:one", "source:two"),
            },
        ),
    )
    reference_count = IRAdapterRegistry(
        bounds=IRAdapterBounds(max_references_per_node=1)
    ).normalize(too_many_references)
    assert reference_count.status is IRAdapterStatus.BOUNDS

    normalized_bytes = IRAdapterRegistry(
        bounds=IRAdapterBounds(max_normalized_bytes=256)
    ).normalize(large_node)
    assert normalized_bytes.status is IRAdapterStatus.BOUNDS
    assert all(
        item.failure is not None and item.failure.required and item.fail_closed
        for item in (node_count, node_bytes, reference_count, normalized_bytes)
    )


def test_family_mismatch_is_typed_unsupported_not_an_authority_conversion() -> None:
    verified = _verified_load(IRFamily.INTENT).require_artifact()
    from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import LegalIRAdapter

    result = LegalIRAdapter().normalize(verified)

    assert result.status is IRAdapterStatus.UNSUPPORTED
    assert result.failure is not None
    assert result.failure.code is IRFailureCode.UNSUPPORTED
    assert result.fail_closed


def test_capability_discovery_covers_all_families_without_import_or_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    processes: list[object] = []
    threads: list[object] = []

    def forbidden_import(name: str, package: str | None = None) -> object:
        imported.append(name)
        raise AssertionError(f"capability discovery imported {name}")

    def forbidden_process(*args: object, **kwargs: object) -> object:
        processes.append((args, kwargs))
        raise AssertionError("capability discovery started a process")

    def forbidden_thread(thread: threading.Thread) -> None:
        threads.append(thread)
        raise AssertionError("capability discovery started a thread")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    monkeypatch.setattr(threading.Thread, "start", forbidden_thread)

    registry = IRAdapterRegistry()
    capabilities = registry.discover_capabilities()

    assert {item.family for item in capabilities} == set(IRFamily)
    assert len(capabilities) == len(IRFamily)
    assert all(
        set(item.operations)
        == {
            "normalize_declarations",
            "normalize_formal_views",
            "normalize_claims",
            "normalize_assumptions",
            "normalize_obligations",
            "normalize_result_authority",
        }
        for item in capabilities
    )
    assert all(item.to_dict()["lazy"] is True for item in capabilities)
    assert all(
        item.to_dict()["grants_execution_authority"] is False
        for item in capabilities
    )
    assert imported == []
    assert processes == []
    assert threads == []
