"""Cross-repository counterexample contract alignment (FVT-002 / FVT-G007).

Proves datasets PublicCounterexampleBoundary@1 / CounterexampleEnvelope@2
delegates semantic identity to the mature supervisor normalizer without
creating a second identity, and that public projections stay aligned across
repository surfaces.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    FORMAL_COUNTEREXAMPLE_SCHEMA,
    CounterexampleKind,
    CounterexampleValidationError,
    FormalCounterexample,
    normalize_counterexample,
    normalize_smt_model,
)
from ipfs_datasets_py.logic.software_verification.counterexamples.contracts import (
    COUNTEREXAMPLE_ENVELOPE_SCHEMA,
    PUBLIC_COUNTEREXAMPLE_BOUNDARY_INTERFACE,
    CounterexampleBoundaryError,
    CounterexampleEnvelope,
    PublicCounterexampleBoundary,
    project_public_counterexample,
)
from ipfs_datasets_py.logic.verification_api import (
    VerificationAuthority,
    VerificationStatus,
    explain_counterexample,
    get_verification_api,
)


def _shared_raw_witness() -> dict[str, Any]:
    return {
        "model": {
            "owners": 2,
            "public_state": "double-owner",
            "hidden_witness": "NEVER-CROSS-REPO",
            "credential": "NEVER-CROSS-REPO-CRED",
            "note": "Authorization: Bearer crossrepotokenvalue0001",
        },
        "stdout": "unbounded prover output",
        "source_excerpt": "unrelated source text",
        "raw_output": "raw dump",
        "violated_property": "obligation:exclusive-lease",
        "assumption_ids": ["assumption:lease"],
        "finite_bounds": {"timeout_ms": 250, "max_steps": 16},
        "provider_id": "provider:z3",
        "tool_id": "solver.z3",
        "tree_id": "tree:cross-repo@1",
        "ast_scope_id": "symbol:normalize",
        "source_ref_ids": ["source:formal_counterexamples.py"],
        "span_ids": ["span:normalize"],
        "plan_id": "plan:288",
        "task_id": "task:288",
        "summary": "exclusive lease violated",
    }


def test_datasets_envelope_reuses_supervisor_semantic_identity() -> None:
    raw = _shared_raw_witness()
    formal = normalize_smt_model(
        raw,
        violated_property="obligation:exclusive-lease",
        bindings={
            "task_id": "task:288",
            "plan_id": "plan:288",
            "tree_id": "tree:cross-repo@1",
            "provider_id": "provider:z3",
            "assumption_id": "assumption:lease",
            "ast_scope_id": "symbol:normalize",
        },
    )
    envelope = project_public_counterexample(raw)

    assert formal.schema == FORMAL_COUNTEREXAMPLE_SCHEMA or formal.to_dict()[
        "schema"
    ] == FORMAL_COUNTEREXAMPLE_SCHEMA
    assert envelope.schema == COUNTEREXAMPLE_ENVELOPE_SCHEMA
    assert envelope.counterexample_id == formal.semantic_id
    assert envelope.semantic_id == formal.semantic_id
    assert envelope.kind == formal.kind.value
    assert envelope.violated_property == formal.violated_property
    assert envelope.property_class == formal.property_class
    assert dict(envelope.payload) == dict(formal.payload)
    assert tuple(envelope.assumptions) == tuple(formal.assumption_ids)
    assert dict(envelope.bounds) == dict(formal.finite_bounds)


def test_projecting_pre_normalized_formal_counterexample_preserves_identity() -> None:
    formal = normalize_counterexample(_shared_raw_witness())
    envelope = project_public_counterexample(formal)
    again = project_public_counterexample(formal.to_dict())

    assert envelope.counterexample_id == formal.semantic_id
    assert again.counterexample_id == formal.semantic_id
    assert envelope.content_id == again.content_id


def test_secret_changes_do_not_fork_cross_repository_identity() -> None:
    left_raw = _shared_raw_witness()
    right_raw = _shared_raw_witness()
    right_raw["model"] = {
        **right_raw["model"],
        "hidden_witness": "OTHER-SECRET",
        "credential": "OTHER-CRED",
    }

    supervisor_left = normalize_counterexample(left_raw)
    supervisor_right = normalize_counterexample(right_raw)
    datasets_left = project_public_counterexample(left_raw)
    datasets_right = project_public_counterexample(right_raw)

    assert supervisor_left.semantic_id == supervisor_right.semantic_id
    assert datasets_left.counterexample_id == datasets_right.counterexample_id
    assert datasets_left.counterexample_id == supervisor_left.semantic_id


def test_public_surfaces_never_reintroduce_private_channels() -> None:
    formal = normalize_counterexample(_shared_raw_witness())
    envelope = project_public_counterexample(_shared_raw_witness())
    api = get_verification_api(reset=True)
    response = api.explain_counterexample(_shared_raw_witness())

    surfaces = [
        formal.to_json().lower(),
        formal.to_capsule_dict(),
        envelope.to_dict(),
        envelope.to_witness_dict(),
        response.to_dict(),
    ]
    for surface in surfaces:
        encoded = (
            surface
            if isinstance(surface, str)
            else json.dumps(surface, sort_keys=True).lower()
        )
        if not isinstance(encoded, str):
            encoded = json.dumps(encoded, sort_keys=True).lower()
        assert "never-cross-repo" not in encoded
        assert "hidden_witness" not in encoded
        assert "credential" not in encoded
        assert "stdout" not in encoded
        assert "source_excerpt" not in encoded
        assert "raw_output" not in encoded
        assert "crossrepotokenvalue0001" not in encoded
        assert "unrelated source text" not in encoded


def test_verification_api_module_entry_matches_boundary_projection() -> None:
    raw = _shared_raw_witness()
    envelope = project_public_counterexample(raw)
    response = explain_counterexample(raw)

    assert response.status is VerificationStatus.SUCCEEDED
    assert response.result["counterexample_id"] == envelope.counterexample_id
    assert response.result["kind"] == envelope.kind
    assert response.result["boundary"] == PUBLIC_COUNTEREXAMPLE_BOUNDARY_INTERFACE
    assert response.authority is VerificationAuthority.BOUNDED
    assert response.result["authority"] == envelope.authority.value
    assert "raw" not in response.result
    assert response.witnesses[0]["counterexample_id"] == envelope.counterexample_id


def test_forged_supervisor_or_datasets_identity_fails_closed() -> None:
    formal = normalize_counterexample(_shared_raw_witness())
    with pytest.raises(CounterexampleValidationError, match="identity"):
        FormalCounterexample.from_dict(
            {**formal.to_dict(), "counterexample_id": "forged"}
        )

    envelope = project_public_counterexample(_shared_raw_witness())
    with pytest.raises(CounterexampleBoundaryError, match="identity"):
        CounterexampleEnvelope.from_dict(
            {**envelope.to_dict(), "counterexample_id": "forged"}
        )


def test_boundary_adapter_and_kinds_cover_required_families() -> None:
    boundary = PublicCounterexampleBoundary()
    cases = [
        (
            {"model": {"x": 1}, "violated_property": "p-model"},
            CounterexampleKind.SMT_MODEL,
        ),
        (
            {"unsat_core": ["a", "b"], "violated_property": "p-core"},
            CounterexampleKind.SMT_UNSAT_CORE,
        ),
        (
            {
                "trace": [{"state": 0}, {"state": 1}],
                "invariant": "Inv",
                "violated_property": "p-trace",
            },
            CounterexampleKind.TLA_TRACE,
        ),
        (
            {
                "failure_code": "statement_mismatch",
                "violated_property": "p-kernel",
            },
            CounterexampleKind.KERNEL_ERROR,
        ),
    ]
    for raw, expected_kind in cases:
        formal = normalize_counterexample(raw)
        envelope = boundary.project(raw)
        assert formal.kind is expected_kind
        assert envelope.kind == expected_kind.value
        assert envelope.counterexample_id == formal.semantic_id
        # Required public fields always present.
        public = envelope.to_public_dict()
        for key in (
            "kind",
            "property_class",
            "violated_property",
            "source_map",
            "tool",
            "assumptions",
            "bounds",
            "authority",
            "private_artifacts",
        ):
            assert key in public


def test_retained_private_artifact_metadata_never_embeds_raw_bytes() -> None:
    store = {
        "raw_output": {
            "digest": "sha256:" + "ef" * 32,
            "retention_policy_id": "policy:private-counterexample-store@1",
            "byte_size": 128,
            "media_type": "application/octet-stream",
        }
    }
    envelope = project_public_counterexample(
        _shared_raw_witness(), private_store=store
    )
    formal = normalize_counterexample(_shared_raw_witness())

    refs = {item.channel: item for item in envelope.private_artifacts}
    assert "provider_artifact" in refs
    assert refs["provider_artifact"].retained is True
    assert refs["provider_artifact"].digest == "sha256:" + "ef" * 32
    encoded = json.dumps(envelope.to_dict()).lower()
    assert "raw dump" not in encoded
    assert "raw_output" not in encoded
    # Supervisor side remains free of the raw channel name values in payload.
    assert "raw dump" not in formal.to_json().lower()
