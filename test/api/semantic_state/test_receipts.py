"""SCH-009 MCP++ receipts and freshness-admission tests."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import (
    canonicalize_artifact,
    compute_artifact_cid,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    VerificationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.receipts import (
    ADMISSION_ADMITTED,
    ADMISSION_INCOMPLETE,
    ADMISSION_SIMULATED,
    ADMISSION_STALE,
    ADMISSION_UNAVAILABLE,
    ADAPTER_ID,
    CompiledReceipt,
    FRESHNESS_ADMISSION_INTERFACE,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    PROOF_STATUS_PASSED,
    PROOF_STATUS_UNAVAILABLE,
    PROVIDER_MODE_PRODUCTION,
    PROVIDER_MODE_SIMULATED,
    RECEIPT_INTERFACE,
    RECEIPT_SCHEMA,
    ReceiptAdmission,
    ReceiptBindings,
    ReceiptCompiler,
    ReceiptError,
    ReceiptFreshnessPolicy,
    SimulatedReceiptError,
    StaleReceiptError,
    UnavailableProofError,
    admit_receipt,
    build_receipt_index,
    compile_verification_receipt,
    receipt_may_promote_root,
    receipt_may_verify,
    receipts_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import (
    SemanticStateWireCodec,
    cid_for_payload,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


class MemoryDurablePort:
    """Hermetic DurableSemanticStatePort double."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self.put_order: list[str] = []

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
        self.put_order.append(expected_cid)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def has(self, cid: str) -> bool:
        return cid in self._objects


def _bindings(**overrides: Any) -> ReceiptBindings:
    payload: dict[str, Any] = {
        "pre_tree_cid": _cid("pre-tree"),
        "post_tree_cid": _cid("post-tree"),
        "datasets_state_cid": _cid("datasets-state"),
        "datasets_semantic_state_root_cid": _cid("datasets-root"),
        "capsule_index_cid": _cid("capsule-index"),
        "delta_cid": _cid("delta"),
        "selection_cid": _cid("selection"),
        "previous_semantic_state_root_cid": _cid("prev-root"),
        "current_semantic_state_root_cid": _cid("curr-root"),
        "command_identity": "sch-cmd:verify:1",
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
        "provider_mode": PROVIDER_MODE_PRODUCTION,
        "proof_outcomes": [
            {"proof_id": "proof.a", "status": PROOF_STATUS_PASSED},
        ],
        "output_artifact_cids": [_cid("out-a"), _cid("out-b")],
        "event_parent_cid": _cid("event-parent"),
    }
    payload.update(overrides)
    return ReceiptBindings.from_dict(payload)


def _current_from(bindings: ReceiptBindings) -> dict[str, Any]:
    data = bindings.to_dict()
    # Drop non-world fields that assess treats specially when present.
    return {
        "pre_tree_cid": data["pre_tree_cid"],
        "post_tree_cid": data["post_tree_cid"],
        "datasets_state_cid": data["datasets_state_cid"],
        "datasets_semantic_state_root_cid": data["datasets_semantic_state_root_cid"],
        "capsule_index_cid": data["capsule_index_cid"],
        "delta_cid": data["delta_cid"],
        "selection_cid": data["selection_cid"],
        "previous_semantic_state_root_cid": data["previous_semantic_state_root_cid"],
        "current_semantic_state_root_cid": data["current_semantic_state_root_cid"],
        "command_identity": data["command_identity"],
        "toolchain_cid": data["toolchain_cid"],
        "dependency_lock_cid": data["dependency_lock_cid"],
        "config_cid": data["config_cid"],
        "policy_cid": data["policy_cid"],
        "interface_cid": data["interface_cid"],
        "provider_mode": data["provider_mode"],
    }


# ---------------------------------------------------------------------------
# Descriptor / authority
# ---------------------------------------------------------------------------


def test_descriptor_pins_interfaces_and_forbids() -> None:
    descriptor = receipts_descriptor()
    assert descriptor["interface"] == RECEIPT_INTERFACE
    assert descriptor["freshness_interface"] == FRESHNESS_ADMISSION_INTERFACE
    assert descriptor["adapter_id"] == ADAPTER_ID
    forbids = set(descriptor["forbids"])
    assert "stale_acceptance" in forbids
    assert "simulation_acceptance" in forbids
    assert "unavailable_proof_as_passed" in forbids
    assert "pseudo_cid" in forbids


# ---------------------------------------------------------------------------
# Compile, rehash, closed schema
# ---------------------------------------------------------------------------


def test_compile_receipt_rehashes_with_real_cidv1() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings,
        exit_code=0,
        stages_passed=True,
        store=False,
    )
    assert receipt.schema == RECEIPT_SCHEMA
    assert receipt.interface == RECEIPT_INTERFACE
    assert receipt.freshness == FRESHNESS_FRESH
    assert receipt.acceptance_eligible is True
    assert receipt.simulated is False
    assert receipt.output_cid.startswith("b")
    assert receipt.receipt_cid.startswith("b")
    # Real Kubo CIDv1, not the pseudo SHA-256 label.
    body = receipt.body_dict()
    assert receipt.output_cid == cid_for_bytes(canonicalize_artifact(body))
    assert receipt.output_cid == cid_for_payload(body)
    assert receipt.output_cid != compute_artifact_cid(body)
    # Closed rehash round-trip.
    again = CompiledReceipt.from_dict(receipt.to_dict())
    assert again.receipt_cid == receipt.receipt_cid
    assert again.output_cid == receipt.output_cid


def test_closed_schema_rejects_unknown_fields() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    payload = receipt.to_dict()
    payload["wall_clock_ms"] = 12
    with pytest.raises(HarnessError, match="fields must be exactly"):
        CompiledReceipt.from_dict(payload)


def test_tampered_body_fails_rehash() -> None:
    receipt = compile_verification_receipt(
        _bindings(), exit_code=0, stages_passed=True, store=False
    )
    payload = receipt.to_dict()
    payload["exit_code"] = 1
    with pytest.raises(ReceiptError, match="rehash|does not match"):
        CompiledReceipt.from_dict(payload)


def test_store_before_reference_order() -> None:
    port = MemoryDurablePort()
    compiler = ReceiptCompiler(durable=port)
    receipt = compiler.compile(
        _bindings(),
        exit_code=0,
        stages_passed=True,
        store=True,
    )
    assert port.has(receipt.output_cid)
    assert port.has(receipt.receipt_cid)
    # Body (output) stored before or at same logical step as envelope.
    assert receipt.output_cid in port.put_order
    assert receipt.receipt_cid in port.put_order
    assert port.put_order.index(receipt.output_cid) < port.put_order.index(
        receipt.receipt_cid
    )
    loaded = compiler.load(receipt.receipt_cid)
    assert loaded.receipt_cid == receipt.receipt_cid
    assert loaded.acceptance_eligible is True


def test_projects_to_verification_receipt_contract() -> None:
    receipt = compile_verification_receipt(
        _bindings(), exit_code=0, stages_passed=True, store=False
    )
    projected = receipt.to_verification_receipt()
    assert isinstance(projected, VerificationReceipt)
    assert projected.tree_cid == receipt.bindings.post_tree_cid
    assert projected.policy_cid == receipt.bindings.policy_cid
    assert projected.acceptance_eligible is True
    assert projected.simulated is False


def test_mcp_execution_receipt_round_trip() -> None:
    receipt = compile_verification_receipt(
        _bindings(), exit_code=0, stages_passed=True, store=False
    )
    envelope = receipt.as_mcp_execution_receipt()
    codec = SemanticStateWireCodec()
    body = codec.decode_execution_receipt(envelope)
    restored = CompiledReceipt.from_dict(body)
    assert restored.receipt_cid == receipt.receipt_cid


# ---------------------------------------------------------------------------
# Freshness: any bound input change stales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "pre_tree_cid",
        "post_tree_cid",
        "datasets_state_cid",
        "datasets_semantic_state_root_cid",
        "capsule_index_cid",
        "delta_cid",
        "selection_cid",
        "current_semantic_state_root_cid",
        "toolchain_cid",
        "dependency_lock_cid",
        "config_cid",
        "policy_cid",
        "interface_cid",
    ],
)
def test_any_bound_input_change_stales_receipt(field: str) -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    current = _current_from(bindings)
    current[field] = _cid(f"changed-{field}")
    admission = admit_receipt(receipt, current=current)
    assert admission.admission == ADMISSION_STALE
    assert admission.freshness == FRESHNESS_STALE
    assert f"stale:{field}" in admission.stale_obligations
    assert admission.can_verify is False
    assert admission.can_promote_root is False
    assert receipt_may_verify(admission) is False
    assert receipt_may_promote_root(admission) is False


def test_fresh_matching_bindings_admit() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    admission = admit_receipt(receipt, current=_current_from(bindings))
    assert admission.admission == ADMISSION_ADMITTED
    assert admission.freshness == FRESHNESS_FRESH
    assert admission.stale_obligations == ()
    assert admission.can_verify is True
    assert admission.can_promote_root is True
    assert receipt_may_verify(admission) is True
    assert receipt_may_promote_root(admission) is True


# ---------------------------------------------------------------------------
# Policy / interface invalidation obligations
# ---------------------------------------------------------------------------


def test_policy_change_invalidates_decisions_and_security_admission() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    current = _current_from(bindings)
    current["policy_cid"] = _cid("new-policy")
    admission = admit_receipt(receipt, current=current)
    assert admission.admission == ADMISSION_STALE
    assert "stale:policy_cid" in admission.stale_obligations
    assert "obligation:policy_decision" in admission.stale_obligations
    assert "obligation:security_admission" in admission.stale_obligations
    assert "policy_invalidates_decisions" in admission.reason_codes


def test_interface_change_invalidates_descriptions_and_adapters() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    current = _current_from(bindings)
    current["interface_cid"] = _cid("new-interface")
    admission = admit_receipt(receipt, current=current)
    assert admission.admission == ADMISSION_STALE
    assert "stale:interface_cid" in admission.stale_obligations
    assert "obligation:interface_description" in admission.stale_obligations
    assert "obligation:client_adapter" in admission.stale_obligations
    assert "interface_invalidates_adapters" in admission.reason_codes


def test_dependency_lock_and_config_stale_emit_verification_obligations() -> None:
    policy = ReceiptFreshnessPolicy()
    bindings = _bindings()
    current = _current_from(bindings)
    current["dependency_lock_cid"] = _cid("new-lock")
    freshness, obs, _ = policy.assess(bindings, current=current)
    assert freshness == FRESHNESS_STALE
    assert "obligation:verification_receipt" in obs
    assert "obligation:dependent_summary" in obs

    current = _current_from(bindings)
    current["config_cid"] = _cid("new-config")
    freshness, obs, _ = policy.assess(bindings, current=current)
    assert freshness == FRESHNESS_STALE
    assert "obligation:bound_test_receipt" in obs


def test_stale_obligations_are_sorted_unique() -> None:
    bindings = _bindings()
    current = _current_from(bindings)
    current["policy_cid"] = _cid("p2")
    current["interface_cid"] = _cid("i2")
    _, obs, _ = ReceiptFreshnessPolicy().assess(bindings, current=current)
    assert obs == tuple(sorted(obs))
    assert len(obs) == len(set(obs))


# ---------------------------------------------------------------------------
# Unavailable proof is explicit
# ---------------------------------------------------------------------------


def test_unavailable_proof_is_explicit_and_blocks_promotion() -> None:
    bindings = _bindings(
        proof_outcomes=[
            {"proof_id": "proof.a", "status": PROOF_STATUS_UNAVAILABLE},
        ]
    )
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    assert receipt.unavailable_proof is True
    assert receipt.acceptance_eligible is False
    assert "unavailable_proof_explicit" in receipt.reason_codes

    admission = admit_receipt(receipt, current=_current_from(bindings))
    assert admission.admission == ADMISSION_UNAVAILABLE
    assert admission.unavailable_proof is True
    assert admission.can_verify is False
    assert admission.can_promote_root is False
    assert receipt_may_promote_root(admission) is False

    with pytest.raises(UnavailableProofError):
        admit_receipt(
            receipt,
            current=_current_from(bindings),
            raise_on_reject=True,
        )


def test_cannot_construct_acceptance_eligible_with_unavailable_proof() -> None:
    bindings = _bindings(
        proof_outcomes=[
            {"proof_id": "p", "status": PROOF_STATUS_UNAVAILABLE},
        ]
    )
    body = {
        "schema": RECEIPT_SCHEMA,
        "interface": RECEIPT_INTERFACE,
        "bindings": bindings.to_dict(),
        "exit_code": 0,
        "stages_passed": True,
        "simulated": False,
        "fresh": True,
        "acceptance_eligible": True,
        "freshness": FRESHNESS_FRESH,
        "unavailable_proof": True,
        "reason_codes": [],
    }
    # Constructor invariants fire via from_dict after rehash fields are added.
    with pytest.raises(ReceiptError, match="unavailable proof"):
        # Build with matching CIDs then fail invariant.
        output_cid = cid_for_payload(body)
        receipt_cid = cid_for_payload({"output_cid": output_cid, "result": body})
        full = dict(body)
        full["output_cid"] = output_cid
        full["receipt_cid"] = receipt_cid
        CompiledReceipt.from_dict(full)


# ---------------------------------------------------------------------------
# Simulation and stale cannot satisfy verification / promotion
# ---------------------------------------------------------------------------


def test_simulation_receipt_cannot_verify_or_promote() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings,
        exit_code=0,
        stages_passed=True,
        simulated=True,
        store=False,
    )
    assert receipt.simulated is True
    assert receipt.acceptance_eligible is False
    admission = admit_receipt(receipt, current=_current_from(bindings))
    assert admission.admission == ADMISSION_SIMULATED
    assert admission.can_verify is False
    assert admission.can_promote_root is False
    assert receipt_may_verify(admission) is False
    assert receipt_may_promote_root(admission) is False
    with pytest.raises(SimulatedReceiptError):
        admit_receipt(
            receipt, current=_current_from(bindings), raise_on_reject=True
        )


def test_provider_mode_simulated_forces_simulation() -> None:
    bindings = _bindings(provider_mode=PROVIDER_MODE_SIMULATED)
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    assert receipt.simulated is True
    assert receipt.acceptance_eligible is False


def test_stale_receipt_cannot_satisfy_verification_or_promotion() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    current = _current_from(bindings)
    current["post_tree_cid"] = _cid("other-tree")
    with pytest.raises(StaleReceiptError) as excinfo:
        admit_receipt(receipt, current=current, raise_on_reject=True)
    assert "stale:post_tree_cid" in excinfo.value.stale_obligations
    assert receipt_may_verify(
        admit_receipt(receipt, current=current)
    ) is False


def test_incomplete_stages_block_admission() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=1, stages_passed=False, store=False
    )
    assert receipt.acceptance_eligible is False
    admission = admit_receipt(receipt, current=_current_from(bindings))
    assert admission.admission == ADMISSION_INCOMPLETE
    assert admission.can_promote_root is False


def test_event_parent_not_current_stales() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    admission = admit_receipt(
        receipt,
        current=_current_from(bindings),
        event_parent_current=False,
    )
    assert admission.admission == ADMISSION_STALE
    assert "stale:event_parent" in admission.stale_obligations


def test_missing_output_artifacts_stale() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    admission = admit_receipt(
        receipt,
        current=_current_from(bindings),
        output_artifacts_present=False,
    )
    assert admission.admission == ADMISSION_STALE
    assert "stale:output_artifacts" in admission.stale_obligations


def test_require_stored_before_admission_reference() -> None:
    port = MemoryDurablePort()
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, durable=port, store=True
    )
    admission = admit_receipt(
        receipt,
        current=_current_from(bindings),
        require_stored=True,
        durable=port,
    )
    assert admission.admission == ADMISSION_ADMITTED

    # Distinct bindings so content/CID is not already present from the first store.
    other = _bindings(command_identity="sch-cmd:verify:unstored")
    unstored = compile_verification_receipt(
        other, exit_code=0, stages_passed=True, store=False
    )
    with pytest.raises(ReceiptError, match="stored before"):
        admit_receipt(
            unstored,
            current=_current_from(other),
            require_stored=True,
            durable=port,
        )


def test_receipt_index_is_sorted_content_addressed() -> None:
    a = _cid("r-a")
    b = _cid("r-b")
    index = build_receipt_index([b, a, b])
    assert index["receipt_cids"] == sorted({a, b})
    assert index["index_cid"] == cid_for_payload(
        {"schema": index["schema"], "receipt_cids": index["receipt_cids"]}
    )


def test_admission_record_closed_schema() -> None:
    bindings = _bindings()
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    admission = admit_receipt(receipt, current=_current_from(bindings))
    payload = admission.to_dict()
    restored = ReceiptAdmission.from_dict(payload)
    assert restored.receipt_cid == admission.receipt_cid
    payload["extra"] = True
    with pytest.raises(HarnessError, match="fields must be exactly"):
        ReceiptAdmission.from_dict(payload)


def test_forged_cid_rejected_in_bindings() -> None:
    with pytest.raises(HarnessError, match="forged|CIDv1|base32"):
        _bindings(pre_tree_cid="sim:local")
    with pytest.raises(HarnessError, match="forged|CIDv1|base32"):
        _bindings(policy_cid="cidv1-sha256-" + "ab" * 32)
