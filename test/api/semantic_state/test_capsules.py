"""SCH-006 capsule admission and durable-index tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import (
    ADMISSION_CONSERVATIVE,
    ADMISSION_EXACT,
    ADMISSION_RAW,
    CAPSULE_ADMISSION_INTERFACE,
    CAPSULE_ADMISSION_SCHEMA,
    CONFIDENCE_VALUES,
    CapsuleAdmission,
    CapsuleAdmissionError,
    CapsuleCache,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    admit_capsule,
    capsule_may_substitute,
    retrieve_opaque_source,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    SemanticCapsuleRef,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


@dataclass
class FakeCapsule:
    """Minimal datasets capsule identity surface (facts intentionally unused)."""

    capsule_cid: str
    stable_symbol_id: str
    version_cid: str
    source_cid: str
    confidence: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    docstring_hint: str | None = None
    # Authoritative-looking facts that admission must never copy.
    signature: Mapping[str, Any] = field(default_factory=lambda: {"name": "secret"})
    effects: tuple[str, ...] = ("IO",)


@dataclass
class FakeFreshness:
    capsule_cid: str
    freshness: str
    admission: str
    caveats: tuple[str, ...] = ()
    assessment_cid: str | None = None


class MemoryDurablePort:
    """Hermetic DurableSemanticStatePort double."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        assert codec == "dag-json"
        body = dict(artifact)
        self._objects[expected_cid] = body
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def has(self, cid: str) -> bool:
        return cid in self._objects

    def get_bytes(self, cid: str) -> bytes:
        return repr(self._objects[cid]).encode("utf-8")


def _capsule(
    *,
    label: str = "sym",
    confidence: str = "exact",
    metadata: Mapping[str, Any] | None = None,
    docstring_hint: str | None = None,
) -> FakeCapsule:
    return FakeCapsule(
        capsule_cid=_cid(f"capsule-{label}"),
        stable_symbol_id=f"symbol.{label}",
        version_cid=_cid(f"version-{label}"),
        source_cid=_cid(f"source-{label}"),
        confidence=confidence,
        metadata=dict(metadata or {}),
        docstring_hint=docstring_hint,
    )


def test_exact_capsule_substitutes_without_raw_source() -> None:
    capsule = _capsule(confidence="exact")
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        validity_bindings=[_cid("policy"), _cid("dep")],
    )
    assert admission.admission == ADMISSION_EXACT
    assert admission.freshness == FRESHNESS_FRESH
    assert admission.may_substitute is True
    assert admission.requires_raw_source is False
    assert admission.ref.raw_source_required is False
    assert admission.ref.confidence == "exact"
    assert admission.ref.capsule_cid == capsule.capsule_cid
    assert admission.ref.validity_bindings == tuple(
        sorted((_cid("dep"), _cid("policy")))
    )
    assert capsule_may_substitute(admission) is True
    # Facts must not appear on the admission record.
    payload = admission.to_dict()
    assert "signature" not in payload
    assert "effects" not in payload
    assert "signature" not in payload["ref"]


def test_conservative_capsule_substitutes_with_visible_caveats() -> None:
    capsule = _capsule(confidence="conservative")
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
    )
    assert admission.admission == ADMISSION_CONSERVATIVE
    assert any("conservative" in item for item in admission.caveats)
    assert admission.may_substitute is True
    assert capsule_may_substitute(admission) is True


@pytest.mark.parametrize("confidence", ["heuristic", "opaque"])
def test_heuristic_and_opaque_force_raw_source(confidence: str) -> None:
    capsule = _capsule(confidence=confidence)
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
    )
    assert admission.admission == ADMISSION_RAW
    assert admission.requires_raw_source is True
    assert admission.ref.raw_source_required is True
    assert capsule_may_substitute(admission) is False
    assert any(confidence in item for item in admission.caveats)


def test_stale_assessment_forces_raw_even_for_exact() -> None:
    capsule = _capsule(confidence="exact")
    assessment = FakeFreshness(
        capsule_cid=capsule.capsule_cid,
        freshness=FRESHNESS_STALE,
        admission=ADMISSION_RAW,
        caveats=("capsule_cid_mismatch",),
        assessment_cid=_cid("assessment"),
    )
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        assessment=assessment,
    )
    assert admission.admission == ADMISSION_RAW
    assert admission.freshness == FRESHNESS_STALE
    assert admission.assessment_cid == _cid("assessment")
    assert capsule_may_substitute(admission) is False


def test_unknown_freshness_forces_raw() -> None:
    capsule = _capsule(confidence="exact")
    assessment = FakeFreshness(
        capsule_cid=capsule.capsule_cid,
        freshness=FRESHNESS_UNKNOWN,
        admission=ADMISSION_RAW,
        caveats=("capsule_index_missing",),
    )
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        assessment=assessment,
    )
    assert admission.admission == ADMISSION_RAW
    assert admission.freshness == FRESHNESS_UNKNOWN


def test_producer_exact_assessment_is_honored() -> None:
    capsule = _capsule(confidence="exact")
    assessment = FakeFreshness(
        capsule_cid=capsule.capsule_cid,
        freshness=FRESHNESS_FRESH,
        admission=ADMISSION_EXACT,
        caveats=(),
        assessment_cid=_cid("assess-exact"),
    )
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        assessment=assessment,
    )
    assert admission.admission == ADMISSION_EXACT
    assert capsule_may_substitute(admission) is True


def test_llm_summary_cannot_raise_confidence() -> None:
    capsule = _capsule(
        confidence="heuristic",
        metadata={
            "llm_summary": "looks exact to me",
            "raised_confidence": "exact",
        },
    )
    with pytest.raises(CapsuleAdmissionError, match="cannot raise"):
        admit_capsule(capsule, semantic_state_root_cid=_cid("root"))


def test_docstring_hint_does_not_block_exact_but_is_non_authoritative() -> None:
    capsule = _capsule(
        confidence="exact",
        docstring_hint="public helper",
        metadata={"docstring": "hint only"},
    )
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
    )
    assert admission.admission == ADMISSION_EXACT
    assert "heuristic_hints_non_authoritative" in admission.caveats
    # Hint text is never copied into the admission artifact.
    blob = str(admission.to_dict())
    assert "public helper" not in blob
    assert "hint only" not in blob


def test_force_raw_source_overrides_exact() -> None:
    capsule = _capsule(confidence="exact")
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        force_raw_source=True,
    )
    assert admission.admission == ADMISSION_RAW
    assert "force_raw_source" in admission.caveats


def test_admission_round_trip_and_content_cid_stable() -> None:
    capsule = _capsule(confidence="conservative", label="rt")
    first = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root-rt"),
        validity_bindings=[_cid("b1")],
    )
    second = CapsuleAdmission.from_dict(first.to_dict())
    assert second.to_dict() == first.to_dict()
    assert second.content_cid() == first.content_cid()
    assert second.content_cid() == cid_for_payload(first.to_dict())
    assert first.to_dict()["schema"] == CAPSULE_ADMISSION_SCHEMA
    assert first.to_dict()["interface"] == CAPSULE_ADMISSION_INTERFACE


def test_identical_inputs_yield_identical_admission() -> None:
    capsule = _capsule(confidence="exact", label="det")
    kwargs = {
        "semantic_state_root_cid": _cid("root-det"),
        "validity_bindings": [_cid("z"), _cid("a")],
    }
    a = admit_capsule(capsule, **kwargs)
    b = admit_capsule(capsule, **kwargs)
    assert a.to_dict() == b.to_dict()
    assert a.content_cid() == b.content_cid()
    # Bindings are sorted uniquely.
    assert a.ref.validity_bindings == tuple(sorted((_cid("a"), _cid("z"))))


def test_mapping_capsule_is_admitted() -> None:
    capsule = {
        "capsule_cid": _cid("map-cap"),
        "stable_symbol_id": "mod.fn",
        "version_cid": _cid("map-ver"),
        "source_cid": _cid("map-src"),
        "confidence": "exact",
    }
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
    )
    assert admission.ref.stable_symbol_id == "mod.fn"
    assert admission.admission == ADMISSION_EXACT


def test_missing_source_cid_fails_closed() -> None:
    capsule = FakeCapsule(
        capsule_cid=_cid("c"),
        stable_symbol_id="s",
        version_cid=_cid("v"),
        source_cid="",
        confidence="exact",
    )
    with pytest.raises(CapsuleAdmissionError, match="source_cid"):
        admit_capsule(capsule, semantic_state_root_cid=_cid("root"))


def test_invalid_confidence_fails_closed() -> None:
    capsule = _capsule(confidence="pretty_sure")
    with pytest.raises(CapsuleAdmissionError, match="confidence"):
        admit_capsule(capsule, semantic_state_root_cid=_cid("root"))


def test_capsule_cache_stores_and_loads_admission() -> None:
    port = MemoryDurablePort()
    cache = CapsuleCache(port)
    capsule = _capsule(confidence="exact", label="cache")
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
    )
    cid = cache.store_admission(admission)
    assert cache.has(cid)
    loaded = cache.get_admission(cid)
    assert loaded.to_dict() == admission.to_dict()
    assert loaded.content_cid() == cid


def test_capsule_cache_stores_envelope_under_producer_cid_key() -> None:
    port = MemoryDurablePort()
    cache = CapsuleCache(port)
    capsule_cid = _cid("producer-capsule")
    envelope = {
        "capsule_cid": capsule_cid,
        "stable_symbol_id": "mod.x",
        "confidence": "exact",
        # Fact-looking field remains only inside the envelope, not re-derived.
        "signature": {"name": "x"},
    }
    wrapper_cid = cache.store_capsule_envelope(
        capsule_cid=capsule_cid,
        envelope=envelope,
    )
    loaded = cache.get_capsule_envelope(wrapper_cid)
    assert loaded["capsule_cid"] == capsule_cid
    assert loaded["signature"] == {"name": "x"}


def test_capsule_cache_rejects_envelope_cid_mismatch() -> None:
    port = MemoryDurablePort()
    cache = CapsuleCache(port)
    with pytest.raises(CapsuleAdmissionError, match="does not match"):
        cache.store_capsule_envelope(
            capsule_cid=_cid("expected"),
            envelope={"capsule_cid": _cid("other"), "x": 1},
        )


def test_retrieve_opaque_source_uses_provider_tree_binding() -> None:
    calls: list[tuple[Any, ...]] = []

    class Provider:
        def read_required_source(
            self,
            semantic_index: Any,
            symbol_id: str,
            *,
            expected_producer_state_cid: str,
            read_source_blob: Any = None,
        ) -> dict[str, Any]:
            calls.append(
                (semantic_index, symbol_id, expected_producer_state_cid, read_source_blob)
            )
            return {
                "source_cid": _cid("src"),
                "bytes": b"def f():\n    return 1\n",
                "from_scanned_tree": True,
            }

    provider = Provider()
    index = object()
    blob_reader = lambda path: b"unused"
    result = retrieve_opaque_source(
        provider,
        index,
        "mod.f",
        expected_producer_state_cid=_cid("producer-state"),
        read_source_blob=blob_reader,
    )
    assert result["from_scanned_tree"] is True
    assert calls[0][1] == "mod.f"
    assert calls[0][2] == _cid("producer-state")
    assert calls[0][3] is blob_reader


def test_retrieve_opaque_source_requires_provider() -> None:
    with pytest.raises(CapsuleAdmissionError, match="provider"):
        retrieve_opaque_source(
            None,
            object(),
            "s",
            expected_producer_state_cid=_cid("state"),
        )


def test_semantic_capsule_ref_remains_admission_only() -> None:
    ref = SemanticCapsuleRef(
        capsule_cid=_cid("c"),
        semantic_state_root_cid=_cid("r"),
        stable_symbol_id="s",
        version_cid=_cid("v"),
        source_cid=_cid("src"),
        confidence="opaque",
        validity_bindings=(),
        raw_source_required=True,
    )
    assert set(ref.to_dict()) == {
        "capsule_cid",
        "semantic_state_root_cid",
        "stable_symbol_id",
        "version_cid",
        "source_cid",
        "confidence",
        "validity_bindings",
        "raw_source_required",
    }


def test_closed_confidence_vocabulary() -> None:
    assert CONFIDENCE_VALUES == frozenset(
        {"exact", "conservative", "heuristic", "opaque"}
    )


def test_producer_raw_cannot_be_relaxed_by_local_exact_defaults() -> None:
    """Even with exact confidence, producer raw admission wins."""
    capsule = _capsule(confidence="exact")
    assessment = FakeFreshness(
        capsule_cid=capsule.capsule_cid,
        freshness=FRESHNESS_FRESH,
        admission=ADMISSION_RAW,
        caveats=("obligation:raw_source_requirement",),
    )
    admission = admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        assessment=assessment,
    )
    assert admission.admission == ADMISSION_RAW
    assert capsule_may_substitute(admission) is False
