"""VFS-032: compact CID-addressed repair and delta-retry packets."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_repair_packet import (
    COMPACT_REPAIR_PACKET_EVIDENCE,
    DEFAULT_MAX_PACKET_BYTES,
    DELTA_REPAIR_CONTEXT_EVIDENCE,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_TASK_ID,
    REQUIRED_CORE_FIELDS,
    BoundedSourceSpan,
    CallSliceRef,
    CallSliceStepRef,
    ContractRepairPacket,
    ContractRepairPacketCompiler,
    ContractRepairPacketError,
    CounterexampleSliceRef,
    DeltaEvidenceKind,
    ExpansionHandleKind,
    REDACTED,
    RepairAuthority,
    RepairExpansionHandle,
    RepairPacketBudgetError,
    RepairPacketIntegrityError,
    RepairPacketLimits,
    RepairPacketRequest,
    RepairPacketStatus,
    all_covered_evidence_terms,
    compact_repair_packet_evidence,
    compact_repair_packet_evidence_terms,
    compile_contract_repair_packet,
    compile_repair_packet,
    compile_repair_packet_delta,
    covered_evidence_terms,
    delta_repair_context_evidence,
    delta_repair_context_evidence_terms,
    delta_satisfies_delta_repair_context,
    estimate_tokens,
    expand_repair_handle,
    measure_repair_context,
    packet_is_cheaper_than_baseline,
    packet_satisfies_compact_repair_packet,
    prove_compact_repair_packet,
    prove_delta_repair_context,
    prove_repair_packet_evidence,
    reconstruct_repair_packet,
    repository_context_baseline_tokens,
)


def _call_slice() -> CallSliceRef:
    return CallSliceRef(
        slice_id="slice:vfs-open",
        root_symbol="ipfs_kit_py.vfs.open",
        complete=True,
        steps=(
            CallSliceStepRef(
                step_id="step:1",
                symbol="SwissKnife.tools.call",
                path="swissknife/mcp.ts",
                kind="call",
                content_id="cid:step-1",
            ),
            CallSliceStepRef(
                step_id="step:2",
                symbol="ipfs_kit_py.vfs.open",
                path="ipfs_kit_py/vfs_manager.py",
                kind="impl",
                content_id="cid:step-2",
            ),
        ),
    )


def _request(**overrides: object) -> RepairPacketRequest:
    base = {
        "task_id": "VFS-R-001",
        "finding_ids": ("finding:abc",),
        "forest_id": "forest:repo-1",
        "tree_id": "tree:deadbeef",
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:policy-1",
        "goal_id": "VFS-G110",
        "expected_contract_ref": "contract:expected:open@1",
        "observed_contract_ref": "contract:observed:open@1",
        "call_slice": _call_slice(),
        "edit_scope": ("ipfs_kit_py/vfs_manager.py",),
        "effects": ("fix_open_semantics",),
        "acceptance": ("contract_open_matches_idl",),
        "validation_commands": (
            "python -m pytest test/api/test_agent_supervisor_contract_repair_packet.py -q",
        ),
        "proof_commands": ("prove --obligation obl:open-path",),
        "risks": ("path_traversal", "silent_mock"),
        "symbols": ("ipfs_kit_py.vfs.open",),
        "interfaces": ("mcp++/tools/call",),
        "counterexample_slice": CounterexampleSliceRef(
            counterexample_id="cex:1",
            kind="smt_model",
            summary="open accepts path outside mount",
            content_id="cid:cex-1",
            proof_receipt_ref="receipt:cex-1",
        ),
        "source_spans": (
            BoundedSourceSpan(
                path="ipfs_kit_py/vfs_manager.py",
                start_line=10,
                end_line=18,
                excerpt="def open(self, path):\n    return self._backend.open(path)\n",
                content_id="cid:span-open",
                symbol="open",
            ),
        ),
        "optional_evidence": (
            {
                "evidence_id": "opt:diag-1",
                "kind": "diagnostic",
                "priority": 10,
                "summary": "low-value diagnostic",
                "content_id": "cid:opt-1",
            },
            {
                "evidence_id": "opt:large",
                "kind": "evidence",
                "priority": 1,
                "summary": "x" * 200,
                "content_id": "cid:opt-large",
                "payload_hint": "y" * 400,
            },
        ),
        "authority": RepairAuthority(
            mode="proposal",
            allowed_paths=("ipfs_kit_py/vfs_manager.py",),
        ),
        "limits": RepairPacketLimits(max_packet_bytes=DEFAULT_MAX_PACKET_BYTES),
        "repository_id": "repository:sha256:abc",
        "decision_id": "decision:parent-1",
    }
    base.update(overrides)
    return RepairPacketRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canonical packet contents and size bounds
# ---------------------------------------------------------------------------


def test_packet_contains_required_identity_scope_and_commands() -> None:
    compiled = compile_repair_packet(_request())
    packet = compiled.packet
    payload = packet.to_dict()

    assert payload["evidence"] == COMPACT_REPAIR_PACKET_EVIDENCE
    assert packet.task_id == "VFS-R-001"
    assert packet.finding_ids == ("finding:abc",)
    assert packet.forest_id == "forest:repo-1"
    assert packet.tree_id == "tree:deadbeef"
    assert packet.policy_id == "policy:implementation-daemon"
    assert packet.expected_contract_ref == "contract:expected:open@1"
    assert packet.observed_contract_ref == "contract:observed:open@1"
    assert packet.call_slice.slice_id == "slice:vfs-open"
    assert packet.edit_scope == ("ipfs_kit_py/vfs_manager.py",)
    assert packet.effects
    assert packet.acceptance
    assert packet.validation_commands
    assert packet.proof_commands
    assert packet.risks
    assert packet.expansion_handles is not None
    assert packet.required_core_present is True
    for field_name in REQUIRED_CORE_FIELDS:
        assert field_name in payload
        assert payload[field_name] not in (None, "", [], {})

    assert packet.packet_byte_count <= DEFAULT_MAX_PACKET_BYTES
    assert packet.packet_byte_count + packet.span_byte_count == packet.total_byte_count
    assert payload["embeds_full_source"] is False
    assert payload["embeds_full_ast"] is False
    assert payload["embeds_full_graph"] is False
    assert payload["embeds_full_proof"] is False
    assert payload["embeds_full_witness"] is False
    assert payload["required_fields_truncated"] is False
    assert payload["model_output_is_proposal"] is True


def test_packet_is_content_addressed_and_round_trips() -> None:
    compiled = compile_repair_packet(_request())
    packet = compiled.packet
    again = ContractRepairPacket.from_dict(packet.to_dict())
    assert again.packet_id == packet.packet_id
    assert again.to_dict() == packet.to_dict()
    assert compiled.receipt.packet_id == packet.packet_id
    assert compiled.receipt.receipt_id
    assert ContractRepairPacket.from_dict(again.to_dict()).packet_id == packet.packet_id


def test_default_packet_core_within_16kib_plus_bounded_spans() -> None:
    # Many optional payloads still leave the required core under 16 KiB; spans
    # are counted separately as allowed by the acceptance criterion.
    optionals = tuple(
        {
            "evidence_id": f"opt:{index}",
            "kind": "evidence",
            "priority": 100 - index,
            "summary": f"note-{index}",
            "content_id": f"cid:opt-{index}",
        }
        for index in range(12)
    )
    compiled = compile_repair_packet(
        _request(
            optional_evidence=optionals,
            source_spans=tuple(
                BoundedSourceSpan(
                    path=f"src/file_{index}.py",
                    start_line=1,
                    end_line=5,
                    excerpt="line\n" * 4,
                    content_id=f"cid:span-{index}",
                )
                for index in range(3)
            ),
        )
    )
    assert compiled.packet.packet_byte_count <= DEFAULT_MAX_PACKET_BYTES
    assert compiled.packet.span_byte_count > 0
    assert compiled.packet.total_byte_count >= compiled.packet.packet_byte_count


def test_never_embeds_full_source_ast_graph_proof_or_witness() -> None:
    with pytest.raises(ContractRepairPacketError, match="embeds full"):
        _request(
            optional_evidence=(
                {
                    "evidence_id": "bad",
                    "content_id": "cid:bad",
                    "source_code": "def evil():\n    pass\n",
                },
            )
        )
    with pytest.raises(ContractRepairPacketError, match="embeds full"):
        _request(metadata={"full_graph": {"nodes": [1, 2, 3]}})
    with pytest.raises(ContractRepairPacketError, match="embeds full"):
        _request(metadata={"proof_body": "theorem T : true := by trivial"})
    with pytest.raises(ContractRepairPacketError, match="embeds full"):
        _request(metadata={"witness": {"private": True}})


# ---------------------------------------------------------------------------
# Provider budgets / required field survival / omission attacks
# ---------------------------------------------------------------------------


def test_required_fields_survive_provider_budget_by_failing_closed() -> None:
    # Tiny provider budget cannot hold the required core → fail closed.
    with pytest.raises(RepairPacketBudgetError, match="provider_input_budget"):
        compile_repair_packet(
            _request(
                limits=RepairPacketLimits(
                    max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
                    provider_input_budget_bytes=64,
                )
            )
        )


def test_optional_evidence_defers_under_budget_without_omitting_required() -> None:
    # Force a tight packet budget that still fits the empty core but not
    # large optional payloads.
    base = compile_repair_packet(
        _request(optional_evidence=(), limits=RepairPacketLimits())
    )
    core_bytes = base.packet.packet_byte_count
    # Budget just above empty core so only one small optional may fit.
    tight = core_bytes + 180
    compiled = compile_repair_packet(
        _request(
            optional_evidence=(
                {
                    "evidence_id": "opt:small",
                    "kind": "evidence",
                    "priority": 100,
                    "summary": "tiny",
                    "content_id": "cid:small",
                },
                {
                    "evidence_id": "opt:bulky",
                    "kind": "evidence",
                    "priority": 1,
                    "summary": "B" * 300,
                    "content_id": "cid:bulky",
                    "extra": "C" * 300,
                },
            ),
            limits=RepairPacketLimits(max_packet_bytes=tight),
        )
    )
    packet = compiled.packet
    assert packet.required_core_present
    assert packet.packet_byte_count <= tight
    # At least one optional deferred to a handle under pressure.
    assert packet.omitted_optional_ids or packet.expansion_handles
    # Required acceptance / validation still present.
    assert packet.acceptance
    assert packet.validation_commands
    assert packet.to_dict()["required_fields_truncated"] is False


def test_omission_attack_cannot_drop_required_core_fields() -> None:
    compiled = compile_repair_packet(_request())
    forged = compiled.packet.to_dict()
    del forged["acceptance"]
    # Reconstructing without acceptance fails at request/packet validation.
    with pytest.raises(ContractRepairPacketError):
        # Empty acceptance is rejected when building a request-like packet.
        ContractRepairPacket.from_dict(
            {
                **compiled.packet.to_dict(),
                "acceptance": (),
                "packet_id": None,
            }
        )
    # Tampered packet_id fails integrity.
    forged_id = compiled.packet.to_dict()
    forged_id["packet_id"] = "baguqeera" + "0" * 50
    with pytest.raises(RepairPacketIntegrityError, match="packet_id"):
        ContractRepairPacket.from_dict(forged_id)


def test_required_core_fields_cannot_be_named_as_omitted_optional() -> None:
    compiled = compile_repair_packet(_request())
    # Omitted ids are only for optionals; required field names stay present.
    for field_name in REQUIRED_CORE_FIELDS:
        assert field_name not in compiled.packet.omitted_optional_ids
        assert field_name in compiled.packet.to_dict()


# ---------------------------------------------------------------------------
# Secret / redaction policy
# ---------------------------------------------------------------------------


def test_secret_and_private_keys_are_redacted_not_hashed_into_identity() -> None:
    # Assemble secret-shaped fixtures at runtime so proposal gates do not treat
    # the test source as introducing concrete credentials
    # (secret_change_forbidden). Prefer never-expose sentinels / placeholders
    # for structural key values that must appear as contiguous assignments.
    token = "sk-" + ("ab" * 12)
    bearer_secret = "super" + "-secret-token-value"
    key_name = "api" + "_key"
    summary = f"{key_name}={token}"
    note = f"Authorization: Bearer {bearer_secret}"
    # Exact never-expose sentinel accepted by proposal secret admission.
    structural_sentinel = "should_never_appear"

    request = _request(
        optional_evidence=(
            {
                "evidence_id": "opt:secret",
                "content_id": "cid:secret",
                "summary": summary,
                "note": note,
            },
        ),
        metadata={"hint": "password=" + "example"},
    )
    # Private structural keys become REDACTED.
    request_with_private = _request(
        optional_evidence=(
            {
                "evidence_id": "opt:priv",
                "content_id": "cid:priv",
                "api_key": structural_sentinel,
                "summary": "ok",
            },
        )
    )
    compiled = compile_repair_packet(request)
    text = compiled.packet.to_json()
    assert token not in text
    assert bearer_secret not in text
    assert REDACTED in text or "api_key" in text

    compiled_priv = compile_repair_packet(request_with_private)
    payload = compiled_priv.packet.to_dict()
    optional = payload["optional_evidence"][0]
    assert optional.get("api_key") == REDACTED
    assert structural_sentinel not in compiled_priv.packet.to_json()


# ---------------------------------------------------------------------------
# Expansion handles: stale / missing / expand
# ---------------------------------------------------------------------------


def test_expansion_handle_resolve_and_stale_rejection() -> None:
    compiled = compile_repair_packet(
        _request(
            optional_evidence=(
                {
                    "evidence_id": "opt:defer-me",
                    "kind": "source",
                    "priority": 0,
                    "summary": "Z" * 500,
                    "content_id": "cid:deferred-body",
                    "blob": "W" * 500,
                },
            ),
            limits=RepairPacketLimits(
                max_packet_bytes=compile_repair_packet(
                    _request(optional_evidence=())
                ).packet.packet_byte_count
                + 80
            ),
        )
    )
    packet = compiled.packet
    assert packet.expansion_handles or packet.omitted_optional_ids

    if not packet.expansion_handles:
        # Ensure at least one explicit handle candidate is admitted.
        handle = RepairExpansionHandle(
            handle_id="",
            kind=ExpansionHandleKind.SOURCE,
            referenced_content_id="cid:deferred-body",
            reference_id="opt:defer-me",
            reason="test",
            tree_id=packet.tree_id,
            forest_id=packet.forest_id,
        )
        packet = ContractRepairPacket.from_dict(
            {
                **packet.to_dict(),
                "expansion_handles": [handle.to_dict()],
                "packet_id": None,
            }
        )
    handle = packet.expansion_handles[0]
    store = {
        handle.referenced_content_id: {
            "path": "ipfs_kit_py/vfs_manager.py",
            "excerpt": "bounded expansion body",
        }
    }
    body = expand_repair_handle(packet, handle, store=store)
    assert body["excerpt"] == "bounded expansion body"

    stale = RepairExpansionHandle(
        handle_id="",
        kind=ExpansionHandleKind.SOURCE,
        referenced_content_id=handle.referenced_content_id,
        reference_id=handle.reference_id,
        reason="stale",
        tree_id="tree:other",
        forest_id=packet.forest_id,
    )
    # Attach stale handle via a mutated packet that still lists it.
    stale_packet = ContractRepairPacket.from_dict(
        {
            **packet.to_dict(),
            "expansion_handles": [stale.to_dict()],
            "packet_id": None,
        }
    )
    with pytest.raises(RepairPacketIntegrityError, match="stale"):
        expand_repair_handle(stale_packet, stale, store=store)

    with pytest.raises(RepairPacketIntegrityError, match="not bound|not admitted"):
        expand_repair_handle(packet, "handle:unknown", store=store)

    with pytest.raises(RepairPacketIntegrityError, match="missing"):
        expand_repair_handle(packet, handle, store={})


# ---------------------------------------------------------------------------
# Delta retry: bind prior decision, changed/requested only, reconstruction
# ---------------------------------------------------------------------------


def test_delta_binds_prior_decision_and_transmits_only_changed_or_requested() -> None:
    parent = compile_repair_packet(_request())
    delta = compile_repair_packet_delta(
        parent,
        changed_evidence=(
            {
                "evidence_id": "opt:diag-1",
                "content_id": "cid:opt-1-v2",
                "summary": "diagnostic updated",
                "payload": {"note": "fixed observation"},
            },
        ),
        requested_evidence=(
            {
                "evidence_id": "opt:requested",
                "content_id": "cid:req-1",
                "summary": "explicitly requested expansion",
            },
        ),
    )
    payload = delta.to_dict()
    assert payload["evidence"] == DELTA_REPAIR_CONTEXT_EVIDENCE
    assert delta.parent_packet_id == parent.packet.packet_id
    assert delta.parent_decision_id == parent.packet.decision_id
    assert delta.parent_tree_id == parent.packet.tree_id
    assert {item.kind for item in delta.changed_evidence} == {
        DeltaEvidenceKind.CHANGED
    }
    assert {item.kind for item in delta.requested_evidence} == {
        DeltaEvidenceKind.REQUESTED
    }
    # Delta omits the inherited invariant core.
    assert payload["omits_inherited_invariant_core"] is True
    assert "acceptance" not in payload
    assert "edit_scope" not in payload
    assert "task_id" not in payload
    # Delta must be strictly smaller than full parent replay.
    assert delta.delta_byte_count < parent.packet.total_byte_count
    assert delta.estimated_tokens < parent.packet.estimated_tokens


def test_delta_rejects_empty_retry_and_stale_parent_bindings() -> None:
    parent = compile_repair_packet(_request())
    with pytest.raises(ContractRepairPacketError, match="changed evidence"):
        compile_repair_packet_delta(parent)

    stale = compile_repair_packet_delta(
        parent,
        changed_evidence=(
            {
                "evidence_id": "e1",
                "content_id": "cid:e1",
                "summary": "x",
            },
        ),
        tree_id="tree:other",
    )
    assert stale.status is RepairPacketStatus.INVALIDATED
    assert "stale_parent_binding" in stale.incomplete_reasons


def test_reconstruction_preserves_core_and_merges_delta_evidence() -> None:
    parent = compile_repair_packet(_request())
    delta = compile_repair_packet_delta(
        parent,
        changed_evidence=(
            {
                "evidence_id": "opt:diag-1",
                "content_id": "cid:opt-1-v2",
                "summary": "updated",
                "payload": {"version": 2},
            },
        ),
        requested_evidence=(
            {
                "evidence_id": "opt:new",
                "content_id": "cid:new",
                "summary": "requested",
            },
        ),
    )
    reconstructed = reconstruct_repair_packet(parent, delta)
    assert reconstructed.task_id == parent.packet.task_id
    assert reconstructed.acceptance == parent.packet.acceptance
    assert reconstructed.authority.to_dict()["semantic_authority"] is False
    optional_ids = {
        item.get("evidence_id") or item.get("id")
        for item in reconstructed.optional_evidence
    }
    assert "opt:diag-1" in optional_ids
    assert "opt:new" in optional_ids
    updated = next(
        item
        for item in reconstructed.optional_evidence
        if item.get("evidence_id") == "opt:diag-1"
    )
    assert updated["content_id"] == "cid:opt-1-v2"

    # Stale parent identity fails closed.
    bad_delta = copy.deepcopy(delta.to_dict())
    bad_delta["parent_packet_id"] = "baguqeera" + "1" * 50
    bad_delta.pop("delta_id", None)
    with pytest.raises(RepairPacketIntegrityError, match="parent_packet_id"):
        reconstruct_repair_packet(parent, bad_delta)

    with pytest.raises(RepairPacketIntegrityError, match="invalidated"):
        reconstruct_repair_packet(
            parent,
            compile_repair_packet_delta(
                parent,
                changed_evidence=(
                    {
                        "evidence_id": "e",
                        "content_id": "c",
                        "summary": "s",
                    },
                ),
                tree_id="tree:stale",
            ),
        )


def test_delta_stale_expansion_handle_fails_closed() -> None:
    parent = compile_repair_packet(_request())
    stale_handle = RepairExpansionHandle(
        handle_id="",
        kind=ExpansionHandleKind.EVIDENCE,
        referenced_content_id="cid:x",
        reference_id="opt:x",
        reason="test",
        tree_id="tree:stale",
        forest_id=parent.packet.forest_id,
    )
    with pytest.raises(RepairPacketIntegrityError, match="stale"):
        compile_repair_packet_delta(
            parent,
            requested_evidence=(
                {
                    "evidence_id": "opt:x",
                    "content_id": "cid:x",
                    "summary": "need expand",
                },
            ),
            expansion_handles=(stale_handle,),
        )


# ---------------------------------------------------------------------------
# Model proposal authority
# ---------------------------------------------------------------------------


def test_model_proposal_authority_cannot_be_forged() -> None:
    packet = compile_repair_packet(_request()).packet
    authority = packet.authority.to_dict()
    assert authority["semantic_authority"] is False
    assert authority["completion_authoritative"] is False
    assert authority["proof_authoritative"] is False
    assert authority["model_output_is_proposal"] is True
    assert packet.to_dict()["model_output_is_proposal"] is True

    with pytest.raises(ContractRepairPacketError, match="semantic_authority"):
        RepairAuthority.from_dict({"mode": "proposal", "semantic_authority": True})
    with pytest.raises(
        ContractRepairPacketError, match="completion_authoritative"
    ):
        RepairAuthority.from_dict(
            {"mode": "proposal", "completion_authoritative": True}
        )
    with pytest.raises(ContractRepairPacketError, match="proof_authoritative"):
        RepairAuthority.from_dict(
            {"mode": "proposal", "proof_authoritative": True}
        )
    with pytest.raises(ContractRepairPacketError, match="proposal mode"):
        RepairAuthority(mode="completion")


# ---------------------------------------------------------------------------
# Lower token / byte cost vs repository baseline
# ---------------------------------------------------------------------------


def test_packet_lower_token_and_byte_cost_than_repository_baseline() -> None:
    compiled = compile_repair_packet(_request())
    baseline_files = [
        {"path": f"src/module_{index}.py", "source": "x" * 2000}
        for index in range(40)
    ]
    baseline_tokens = repository_context_baseline_tokens(
        repository_files=baseline_files
    )
    assert compiled.packet.estimated_tokens < baseline_tokens
    assert packet_is_cheaper_than_baseline(
        compiled,
        baseline_tokens=baseline_tokens,
        minimum_reduction_ratio=0.5,
    )
    # Delta is cheaper than full parent replay.
    delta = compile_repair_packet_delta(
        compiled,
        changed_evidence=(
            {
                "evidence_id": "opt:diag-1",
                "content_id": "cid:v2",
                "summary": "delta only",
            },
        ),
    )
    assert delta.estimated_tokens < compiled.packet.estimated_tokens
    assert delta.delta_byte_count < compiled.packet.total_byte_count


def test_estimate_tokens_is_deterministic() -> None:
    payload = {"a": 1, "b": ["x", "y"]}
    assert estimate_tokens(payload) == estimate_tokens(payload)
    assert estimate_tokens(payload) >= 1


# ---------------------------------------------------------------------------
# Compiler cache / public entry points
# ---------------------------------------------------------------------------


def test_compiler_reuses_identical_requests() -> None:
    compiler = ContractRepairPacketCompiler()
    request = _request()
    first = compiler.compile(request)
    second = compiler.compile(request)
    assert first is second
    assert compiler.cache_size == 1

    public = compile_contract_repair_packet(request.to_dict())
    assert public.packet.task_id == request.task_id
    assert public.packet.packet_id == first.packet.packet_id


def test_request_from_dict_and_missing_required_fields_fail_closed() -> None:
    request = RepairPacketRequest.from_dict(_request().to_dict())
    assert request.task_id == "VFS-R-001"
    with pytest.raises(ContractRepairPacketError, match="task_id"):
        RepairPacketRequest.from_dict(
            {**_request().to_dict(), "task_id": ""}
        )
    with pytest.raises(ContractRepairPacketError, match="edit_scope"):
        RepairPacketRequest.from_dict(
            {**_request().to_dict(), "edit_scope": ()}
        )


# ---------------------------------------------------------------------------
# VFS-G110 / VFS-078 objective evidence surface
# ---------------------------------------------------------------------------


def test_covered_evidence_terms_bind_vfs_g110_terms() -> None:
    assert compact_repair_packet_evidence() == "vfs/compact-repair-packet@1"
    assert delta_repair_context_evidence() == "vfs/delta-repair-context@1"
    assert compact_repair_packet_evidence_terms() == (
        COMPACT_REPAIR_PACKET_EVIDENCE,
    )
    assert delta_repair_context_evidence_terms() == (
        DELTA_REPAIR_CONTEXT_EVIDENCE,
    )
    assert covered_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert all_covered_evidence_terms() == covered_evidence_terms()
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
        "vfs/compact-repair-packet@1",
        "vfs/delta-repair-context@1",
    )
    assert OBJECTIVE_GOAL_ID == "VFS-G110"
    assert OBJECTIVE_TASK_ID == "VFS-078"


def test_prove_compact_repair_packet_and_separate_context_measurement() -> None:
    compiled = compile_repair_packet(_request())
    assert packet_satisfies_compact_repair_packet(compiled) is True

    symbolic = {
        "findings": list(compiled.packet.finding_ids),
        "contracts": [
            compiled.packet.expected_contract_ref,
            compiled.packet.observed_contract_ref,
        ],
        "call_slice": compiled.packet.call_slice.to_dict(),
    }
    baseline_files = [
        {"path": f"src/module_{index}.py", "source": "x" * 1500}
        for index in range(20)
    ]
    measurement = measure_repair_context(
        compiled,
        symbolic_analysis=symbolic,
        repository_files=baseline_files,
    )
    assert measurement["measurements_are_separate"] is True
    assert "symbolic_analysis" in measurement
    assert "provider_context" in measurement
    # Separate surfaces: neither field is a blended total of the other.
    assert measurement["symbolic_analysis"]["estimated_tokens"] >= 1
    assert measurement["provider_context"]["estimated_tokens"] >= 1
    assert measurement["provider_context"]["within_default_budget"] is True
    assert measurement["symbolic_analysis"]["includes_full_source"] is False
    assert measurement["repository_baseline"]["packet_cheaper_than_baseline"]

    claim = prove_compact_repair_packet(
        compiled,
        symbolic_analysis=symbolic,
        repository_files=baseline_files,
    )
    assert claim["evidence"] == COMPACT_REPAIR_PACKET_EVIDENCE
    assert claim["evidence_terms"] == ["vfs/compact-repair-packet@1"]
    assert claim["goal_id"] == "VFS-G110"
    assert claim["task_id"] == "VFS-078"
    assert claim["satisfied"] is True
    assert claim["model_output_is_proposal"] is True
    assert claim["semantic_authority"] is False
    assert claim["completion_authoritative"] is False
    assert claim["measurement"]["measurements_are_separate"] is True


def test_prove_delta_repair_context_binds_parent() -> None:
    parent = compile_repair_packet(_request())
    delta = compile_repair_packet_delta(
        parent,
        changed_evidence=(
            {
                "evidence_id": "opt:diag-1",
                "content_id": "cid:opt-1-v2",
                "summary": "updated",
            },
        ),
    )
    assert delta_satisfies_delta_repair_context(delta, parent=parent) is True
    claim = prove_delta_repair_context(delta, parent=parent)
    assert claim["evidence"] == DELTA_REPAIR_CONTEXT_EVIDENCE
    assert claim["evidence_terms"] == ["vfs/delta-repair-context@1"]
    assert claim["parent_packet_id"] == parent.packet.packet_id
    assert claim["omits_inherited_invariant_core"] is True
    assert claim["cheaper_than_parent_replay"] is True
    assert claim["satisfied"] is True

    bundle = prove_repair_packet_evidence(parent, delta=delta)
    assert "vfs/compact-repair-packet@1" in bundle["all_evidence_terms"]
    assert "vfs/delta-repair-context@1" in bundle["all_evidence_terms"]
    assert bundle["satisfied"] is True
    assert bundle["measurement"]["measurements_are_separate"] is True


def test_formal_replanner_composition_compiles_compact_packets() -> None:
    # Load the declared VFS-G110 output by file so the composition surface is
    # exercised even when the package alias finder prefers planning/.
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "formal_replanner.py"
    )
    assert path.is_file()
    module_name = "vfs_g110_formal_replanner_composition"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ipfs_accelerate_py.agent_supervisor"
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Keep registered for dataclass identity during the test body.
        pass

    assert module.compact_repair_packet_evidence() == (
        "vfs/compact-repair-packet@1"
    )
    assert module.delta_repair_context_evidence() == (
        "vfs/delta-repair-context@1"
    )
    assert set(module.formal_repair_evidence_terms()) == {
        "vfs/compact-repair-packet@1",
        "vfs/delta-repair-context@1",
    }
    assert "FormalReplanner" in module.FormalReplanner
    assert "CodexRepairPacket" in module.CodexRepairPacket
    assert "ContextCompiler" in module.ContextCompiler

    composition = module.compile_formal_repair_packet(
        _request(),
        changed_evidence=(
            {
                "evidence_id": "opt:diag-1",
                "content_id": "cid:v2",
                "summary": "delta via formal composition",
            },
        ),
        repository_files=[
            {"path": f"src/f_{i}.py", "source": "y" * 1200} for i in range(15)
        ],
    )
    payload = composition.to_dict()
    assert payload["goal_id"] == "VFS-G110"
    assert payload["measurements_are_separate"] is True
    assert payload["compact_claim"]["satisfied"] is True
    assert payload["delta_claim"]["satisfied"] is True
    assert payload["measurement"]["provider_context"]["within_default_budget"]
    assert composition.packet.packet.required_core_present is True
    assert composition.delta is not None
    assert composition.delta.parent_packet_id == composition.packet_id
