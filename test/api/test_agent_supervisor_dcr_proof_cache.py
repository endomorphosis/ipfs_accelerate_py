"""DCR-034 proof-cache contract tests; no prover or network is executed."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    RepairAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.proof.dcr_proof_cache import (
    DcrProofCache,
    DcrProofCacheBinding,
    DcrProofCacheDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_reconstruction import (
    KernelReconstructionDisposition,
    KernelReconstructionResult,
    KernelReconstructionRoots,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    DeterministicProverDisposition,
    DeterministicProverResources,
    DeterministicProverRoute,
)


def _id(value: str) -> str:
    return content_identity({"fixture": value})


def _binding(*, epoch: str = "epoch:one") -> DcrProofCacheBinding:
    obligation = McpGraphContractObligation(
        obligation_id=_id("obligation"),
        family=McpObligationFamily.JSONRPC_BASELINE,
        fragment=McpObligationFragment.JSONRPC,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid=_id("graph"),
        candidate_cid=_id("candidate"),
        input_cids=tuple(sorted((_id("graph"), _id("input")))),
    )
    route = DeterministicProverRoute(
        obligation_id=obligation.obligation_id,
        obligation_cid=content_identity(obligation.to_dict()),
        backend_id="fixture-local-backend",
        disposition=DeterministicProverDisposition.ROUTED,
        reason_codes=(),
        resources=DeterministicProverResources(
            seed=1, max_steps=10, max_memory_bytes=4096
        ),
        capability_receipt_ids=("capability:module", "capability:toolchain"),
        evidence_receipt_ids=("evidence:init", "evidence:reconstruct"),
        integration_pending=False,
    )
    authority = RepairAuthorityRoots(
        repository_id="repository:fixture",
        repository_forest_cid=_id("forest"),
        git_tree_id=_id("tree"),
        policy_root=_id("repair-policy"),
        rpr_plan_cid=_id("rpr-plan"),
        rpr_packet_cid=_id("rpr-packet"),
    )
    roots = KernelReconstructionRoots(
        authority_roots=authority,
        graph_cid=obligation.graph_cid,
        live_transcript_cid=_id("transcript"),
    )
    receipt = KernelReconstructionResult(
        KernelReconstructionDisposition.RECONSTRUCTED,
        (),
        request_cid=_id("request"),
        proof_cid=_id("proof"),
        certificate_cid=_id("certificate"),
        roots=roots,
    )
    return DcrProofCacheBinding(
        dcr030_input_cids=tuple(sorted((_id("input"), _id("source-input")))),
        dcr030_source_root=_id("source"),
        dcr030_forest_root=authority.repository_forest_cid,
        dcr031_obligation=obligation,
        dcr032_route=route,
        dcr032_toolchain_id="fixture-toolchain",
        dcr033_receipt=receipt,
        dcr033_kernel_binding_cid=_id("kernel-binding"),
        policy_root=_id("policy"),
        runtime_root=_id("runtime"),
        transcript_root=roots.live_transcript_cid,
        dependency_roots=(_id("dependency-one"), _id("dependency-two")),
        epoch_cid=epoch,
    )


def test_cache_hit_requires_equal_cold_receipt_and_never_is_proof_authority() -> None:
    cache = DcrProofCache()
    binding = _binding()

    stored = cache.put(binding)
    hit = cache.lookup(binding, cold_receipt=binding.dcr033_receipt)

    assert stored.disposition is DcrProofCacheDisposition.MISS
    assert hit.disposition is DcrProofCacheDisposition.HIT
    assert hit.cached_receipt == binding.dcr033_receipt.to_dict()
    assert hit.proof_authorized is False
    assert hit.model_call_count == hit.provider_call_count == 0

    mismatched = replace(binding.dcr033_receipt, proof_cid=_id("other-proof"))
    assert (
        cache.lookup(binding, cold_receipt=mismatched).disposition
        is DcrProofCacheDisposition.MISS
    )


def test_changed_root_cross_epoch_and_dependency_invalidation_fail_closed() -> None:
    cache = DcrProofCache()
    binding = _binding()
    cache.put(binding)

    changed_root = replace(binding, runtime_root=_id("changed-runtime"))
    assert cache.lookup(changed_root).disposition is DcrProofCacheDisposition.MISS

    cross_epoch = replace(binding, epoch_cid="epoch:two")
    assert (
        cache.lookup(cross_epoch).disposition
        is DcrProofCacheDisposition.CROSS_EPOCH_REJECTED
    )

    invalidated = cache.invalidate_dependencies((binding.dependency_roots[0],))
    assert invalidated == (binding.key_cid,)
    assert cache.lookup(binding).disposition is DcrProofCacheDisposition.INVALIDATED


def test_only_typed_reconstructed_or_replayable_counterexample_receipts_store() -> None:
    cache = DcrProofCache()
    binding = _binding()
    invalid = replace(
        binding,
        dcr033_receipt=replace(
            binding.dcr033_receipt,
            disposition=KernelReconstructionDisposition.INVALID,
        ),
    )
    missing_root = replace(binding, dcr030_source_root="")

    assert cache.put(invalid).disposition is DcrProofCacheDisposition.REJECTED
    assert cache.put(missing_root).disposition is DcrProofCacheDisposition.REJECTED

    counterexample = replace(
        binding.dcr033_receipt,
        disposition=KernelReconstructionDisposition.REFUTED,
        proof_cid="",
        certificate_cid="",
        counterexample_cid=_id("counterexample"),
        counterexample_bytes=b"replayable-fixture",
    )
    refuted = replace(binding, dcr033_receipt=counterexample)
    assert cache.put(refuted).disposition is DcrProofCacheDisposition.MISS
    assert (
        cache.lookup(refuted, cold_receipt=counterexample).disposition
        is DcrProofCacheDisposition.HIT
    )
