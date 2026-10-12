"""Admit a freshly replayed scoped header lemma to an existing Doctor cache.

The receipt proves the exact local lemma under its retained assumptions. It
cannot authorize task completion, source mutation, execution or whole-program
claims. Host-admitted profiles and a path digest are expected inputs, not
authentication. Existing native and cache owners may be written; sequential
source/owner checks are not an atomic lease or proof of resource enforcement.
No owner, provider or model is constructed by this adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from ..planning.deterministic_doctor_tactician import DoctorTacticianPlanReceipt
from ..proof import formal_verification_contracts as receipt_contracts
from ..proof.doctor_cache_state import DoctorCacheState
from ..proof.doctor_proof_cache import (
    DoctorCacheAuditReceipt, DoctorCacheDisposition, DoctorCacheLookupResult,
    DoctorCacheStage, DoctorCacheStoreResult, DoctorProofCacheGate, DoctorProofCacheKey,
)
from ..proof.formal_verification_cache import FormalVerificationCache
from ..proof.formal_verification_contracts import (
    AssuranceLevel, EvidenceAuthority, EvidenceFreshness, EvidenceKind, EvidenceVerdict,
    LocalProofScope, ProofEvidence, ProofReceipt, ProofVerdict, ScopedProofReceipt,
    canonical_json, content_identity,
)
from . import doctor_contract_proof as proof_factory
from . import doctor_contract_proof_custody as custody
from .doctor_contract_key_custody import _native_pin_config, restore_header_source_premise
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "supervisor-doctor-header-cache-bridge@1"
MAX_RECEIPT_BYTES = 256 * 1024


class HeaderProofCacheBridgeError(ValueError):
    """Current source, original custody or an admitted owner refused reuse.

    A refusal may follow writes to the existing cache owner. It never returns
    an accepted handle or stage result after a failed final check.
    """

    native_gate_bookkeeping_may_write_existing_owner = True
    atomic_owner_snapshot = False


def _require(condition, message):
    if not condition:
        raise HeaderProofCacheBridgeError(message)


def _json(value):
    body = canonical_json(value)
    _require(len(body.encode("utf-8")) <= MAX_RECEIPT_BYTES,
             "bounded public header receipt required")
    return body


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _implementation_bindings():
    """Current emitter/codec byte pins, not policy or proof authentication."""
    return {"bridge_sha256": hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest(),
        "receipt_contracts_sha256": hashlib.sha256(
            Path(receipt_contracts.__file__).resolve(strict=True).read_bytes()).hexdigest()}


def header_proof_implementation_bindings():
    """Current emitter/codec byte provenance; this confers no proof authority."""
    return dict(_implementation_bindings())


class _CacheFence:
    def __init__(self, gate, expected_path_sha256):
        _require(type(gate) is DoctorProofCacheGate
                 and type(gate.authoritative_cache) is FormalVerificationCache
                 and type(gate._state) is DoctorCacheState,
                 "existing exact Doctor gate and formal cache required")
        _require(type(expected_path_sha256) is str and len(expected_path_sha256) == 64
                 and all(c in "0123456789abcdef" for c in expected_path_sha256),
                 "independently expected canonical cache path digest required")
        self.gate, self.cache, self.state = gate, gate.authoritative_cache, gate._state
        self.path = gate.db_path
        _require(type(self.path) is type(Path()) and self.path.is_absolute()
                 and self.path.suffix == ".duckdb" and _sha(str(self.path)) == expected_path_sha256,
                 "cache owner differs from independently expected path")
        self.owner = custody._file_identity(self.path)
        self.check()

    def check(self):
        _require(self.gate.authoritative_cache is self.cache and self.gate._state is self.state
                 and self.state._cache is self.cache and self.gate.db_path == self.path
                 and self.cache.db_path == self.path
                 and custody._file_identity(self.path) == self.owner,
                 "original generic cache owner changed")


def _native_recipe_join(prepared, proof):
    """Review the actual header adapter's namespaces and canonical transcripts."""
    key, theorem = prepared.key, prepared.theorem
    _native_pin_config(proof_module=proof_factory, solver_pin=prepared.solver_pin,
        kernel_pin=prepared.kernel_pin, source=theorem.theorem_body,
        toolchain_preimage=prepared.toolchain_preimage)
    cfg = json.loads(prepared.kernel_pin.argv[4])
    native, kernel = proof.native_execution, proof.kernel_replay
    _require(kernel.kernel_id == cfg["kernel_id"]
             and key.kernel.logical_id == prepared.kernel_pin.verifier_id
             and key.solver.logical_id == prepared.solver_pin.verifier_id
             and kernel.executable_pin_cid == prepared.kernel_pin.content_id
             and native.executable_pin_cid == prepared.solver_pin.content_id
             and native.proof_object == theorem.theorem_body,
             "native kernel identity or proof differs from the reviewed header recipe")
    common = {"theorem_cid": theorem.content_id, "lowering_cid": prepared.lowering.content_id,
        "property_id": theorem.property_id, "consequence_ref": theorem.consequence_ref,
        "premise_ids": list(theorem.premise_ids)}
    inputs = (
        {"interface": "DoctorNativeSolverRequest@1", "roots": theorem.roots.to_dict(),
         "theorem": theorem.to_dict(), "lowering": prepared.lowering.to_dict()},
        {"interface": "DoctorKernelReplayRequest@1", "roots": theorem.roots.to_dict(),
         "theorem": theorem.to_dict(), "lowering": prepared.lowering.to_dict(),
         "native_receipt": native.to_dict()},
    )
    outputs = (
        {**common, "status": "proved", "proof_object": theorem.theorem_body},
        {**common, "status": "kernel_verified", "native_receipt_cid": native.content_id,
         "proof_object_cid": native.proof_object_cid, "kernel_id": cfg["kernel_id"]},
    )
    for receipt, pin, request, output in zip(
            (native, kernel), (prepared.solver_pin, prepared.kernel_pin), inputs, outputs):
        transcript = {name: getattr(receipt, name) for name in (
            "executable_sha256", "command_id", "stdin_cid", "stdout_cid", "stderr_cid", "exit_code")}
        _require(receipt.executed and receipt.exit_code == 0
                 and receipt.executable_sha256 == pin.executable_sha256
                 and receipt.command_id == pin.command_id
                 and receipt.stdin_cid == content_identity({"bytes_hex": canonical_json(request).encode().hex()})
                 and receipt.stdout_cid == content_identity({"bytes_hex": canonical_json(output).encode().hex()})
                 and receipt.transcript_cid == content_identity(transcript),
                 "sealed native header transcript differs from the actual pinned recipe")
    _require(kernel.kernel_verified, "independent native kernel replay required")
    contract = prepared.contract
    _require(len(key.premises) == 1
             and key.premises[0].logical_id == content_identity(contract)
             and theorem.premise_ids == (key.premises[0].logical_id,)
             and restore_header_source_premise(json.loads(key.premises[0].canonical_bytes),
                                               expected_contract=contract) == contract
             and json.loads(key.goal.canonical_bytes) == theorem.to_dict()
             and key.goal.logical_id == theorem.theorem_id
             and key.ast.logical_id == prepared.scoped_analysis.snapshot.roots.ast_root_id
             and key.ast.logical_id == prepared.scoped_analysis.diagnostic_snapshot.ast_index.index_id
             and theorem.roots.index_id == prepared.scoped_analysis.snapshot.roots.index_id
             and theorem.roots.repository_id == prepared.scoped_analysis.snapshot.roots.repository_id
             and key.budget.to_dict() == prepared.bounds.to_resource_budget().to_dict(),
             "local theorem, original source premise, AST, repository or budget differs")


def _boundary(prepared):
    """Retain real open frontiers and, when present, recompiled SecurityIR IDs."""
    scoped, contract = prepared.scoped_analysis, prepared.contract
    proof_factory._replay_bound_candidate(scoped=scoped, contract=contract,
                                          candidate_cid=prepared.candidate_cid)
    frontiers = set(scoped.report["open_frontiers"]) | set(contract["open_frontiers"])
    residual_ids = ()
    if "security_ir_declaration_cid" in contract:
        _verified, spec, _paths = proof_factory.scoped_analysis._selection(
            scoped.repository, json.loads(scoped._admission_json), scoped.task_cid)
        analyses = []
        protocol = proof_factory.header_contracts.WsgiHeaderProtocolContract(**contract["protocol"])
        for output in spec["outputs"]:
            path = output["path"]
            if output["effect"] != "modify" or not path.endswith(".py"):
                continue
            analysis = proof_factory.header_contracts.analyze_http_header_contracts(
                scoped.sources[path].decode("utf-8"), protocol=protocol)
            analyses.append({"path": path, "status": analysis.status,
                "reason_codes": list(analysis.reason_codes), "source_sha256": analysis.source_sha256,
                "contracts": list(analysis.contracts), "open_frontiers": list(analysis.open_frontiers)})
        compiled = proof_factory.security_ir_bridge.compile_header_security_ir(scoped=scoped, analyses=analyses)
        _require(compiled.report["declaration_cid"] == contract["security_ir_declaration_cid"]
                 and compiled.report["artifact_cid"] == contract["security_ir_formalization_cid"],
                 "SecurityIR residuals differ from independently recompiled declarations")
        residual_ids = tuple(sorted(item.obligation_id for item in compiled.artifact.proof_obligations))
        frontiers.update(compiled.report["open_frontiers"])
    _require(all(type(item) is str and 0 < len(item.encode()) <= 4096 for item in frontiers)
             and len(frontiers) <= 256, "bounded native open frontier strings required")
    return tuple(sorted(frontiers)), residual_ids


def _receipt(prepared, proof, custody_ref):
    implementation_bindings = _implementation_bindings()
    key, theorem = prepared.key, prepared.theorem
    _native_recipe_join(prepared, proof)
    frontiers, residual_ids = _boundary(prepared)
    scope = LocalProofScope(obligation_id=theorem.theorem_id, theorem_cid=theorem.content_id,
        property_id=theorem.property_id, consequence_ref=theorem.consequence_ref,
        proof_scope=proof_factory.PROOF_SCOPE, premise_ids=theorem.premise_ids,
        assumption_ids=theorem.assumption_ids, residual_obligation_ids=residual_ids)
    evidence = ProofEvidence(kind=EvidenceKind.KERNEL_VERIFICATION, authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED, artifact_id=proof.kernel_store_ref.receipt_cid,
        subject_id=theorem.theorem_id, verifier_id=prepared.kernel_pin.verifier_id,
        freshness=EvidenceFreshness.CURRENT, independent=True, simulated=False,
        metadata={"schema": "doctor-header-kernel-evidence@1", "theorem_cid": theorem.content_id,
            "lowering_cid": prepared.lowering.content_id, "native_receipt_cid": proof.native_execution.content_id,
            "native_kernel_id": proof.kernel_replay.kernel_id, "kernel_pin_cid": prepared.kernel_pin.content_id,
            "kernel_command_id": prepared.kernel_pin.command_id, "kernel_entry_cid": proof.kernel_store_ref.entry_cid})
    receipt = ScopedProofReceipt(scope=scope, obligation_id=theorem.theorem_id,
        plan_id=DoctorTacticianPlanReceipt.from_dict(prepared.report["tactician"]).content_id,
        attempt_id=proof.native_execution.content_id,
        repository_id=theorem.roots.repository_id, repository_tree_id=key.tree.logical_id,
        ast_scope_ids=(key.ast.logical_id,), premise_ids=theorem.premise_ids,
        translator_id=key.translation.logical_id, solver_id=key.solver.logical_id,
        kernel_id=key.kernel.logical_id, toolchain_id=key.toolchain.logical_id,
        theorem_registry_id=key.registry.logical_id, policy_id=key.policy.logical_id,
        resource_budget=key.budget, verdict=ProofVerdict.PROVED, evidence=(evidence,),
        provider_id=proof.producer_id, freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id=proof.kernel_store_ref.receipt_cid,
        metadata={"schema": SCHEMA, "full_key_id": key.key_id, "native_proof_cid": proof.content_id,
            "native_authority_entry_cid": proof.authority_store_ref.entry_cid,
            "native_custody_receipt_cid": custody_ref.receipt_cid,
            "generic_receipt_implementations": implementation_bindings,
            "source_premise_profile_cid": key.premises[0].cid, "open_frontiers": list(frontiers),
            "security_ir_obligations_discharged": False, "completion_authority": False,
            "execution_authority": False, "mutation_authority": False, "whole_program_proved": False,
            "resource_enforcement_verified": False})
    body = _json(receipt.to_dict())
    restored = ProofReceipt.from_dict(json.loads(body))
    _require(type(restored) is ScopedProofReceipt and _json(restored.to_dict()) == body
             and receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
             and receipt.authoritative_verdict is ProofVerdict.PROVED
             and not receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED),
             "typed conditional local receipt did not retain its assurance and completion boundary")
    _require(_implementation_bindings() == implementation_bindings,
             "current generic header receipt emitter or codec changed during reconstruction")
    return receipt


class _AdmissionGuard:
    def __init__(self, prepared, hammer, custody_ref, expected_key_id, gate, expected_owner_path_sha256):
        self.cache_fence = _CacheFence(gate, expected_owner_path_sha256)
        self.prepared, self.hammer, self.gate = prepared, hammer, gate
        self.body, self.store = custody._prepared_inputs(prepared, hammer, expected_key_id)
        self.native_fence = custody._OwnerFence(hammer, self.store)
        self.reference = custody._reference(custody_ref, custody.HeaderContractProofKeyCustody.SCHEMA)
        self.expected_key_id = expected_key_id
        self.key_json = canonical_json(prepared.key.to_dict())
        retained = self.store.reload(self.reference, custody.HeaderContractProofKeyCustody)
        self.native_fence.check()
        _require(retained.preparation_json == self.body, "original native custody preparation differs")
        self.proof_reference = retained.proof_ref
        self.proof = None
        self.receipt_json = None
        self._operation_active = False
        self.denial()

    def denial(self):
        for check in (self.gate.is_quarantined, self.gate.is_tombstoned):
            value = check(self.prepared.key)
            _require(type(value) is bool and not value, "native Doctor key is quarantined or tombstoned")
            self.cache_fence.check()
            self.native_fence.check()
        # The public boolean projections consult separate snapshots. A later
        # tombstone consultation can also see a newly visible quarantine.
        # Preserve both denial flags from one final native transaction.
        state = self.gate._state.check(self.prepared.key)
        _require(type(state) is dict and not state,
                 "combined native Doctor denial state refused local proof")
        self.cache_fence.check()
        self.native_fence.check()

    def _current_native(self):
        body, store = custody._prepared_inputs(self.prepared, self.hammer, self.expected_key_id)
        _require(store is self.store and body == self.body
                 and canonical_json(self.prepared.key.to_dict()) == self.key_json,
                 "current independent source, profiles or complete key changed")
        retained = self.store.reload(self.reference, custody.HeaderContractProofKeyCustody)
        self.native_fence.check()
        _require(retained.preparation_json == self.body
                 and retained.proof_ref.to_dict() == self.proof_reference.to_dict(),
                 "original native proof custody changed")
        custody._proof_binding(self.prepared, self.proof, self.store, self.native_fence)
        _require(self.proof.authority_store_ref.to_dict() == self.proof_reference.to_dict(),
                 "freshly replayed proof is not the original sealed proof")
        if self.receipt_json is not None:
            _require(_json(_receipt(self.prepared, self.proof, self.reference).to_dict()) == self.receipt_json,
                     "current local scope or deterministic public receipt changed")
        custody._still_current(self.prepared, self.hammer, self.body, self.store, self.native_fence)

    @contextmanager
    def _operation(self):
        _require(not self._operation_active, "recursive header proof admission operation refused")
        self._operation_active = True
        try:
            yield
        finally:
            self._operation_active = False

    def _check(self):
        self.cache_fence.check()
        self.native_fence.check()
        self._current_native()
        self.denial()
        # Source and original custody are rechecked after the final generic
        # owner consultation. These sequential fences still provide no lease.
        self._current_native()
        self.cache_fence.check()
        self.native_fence.check()

    def check(self):
        with self._operation():
            self._check()

    def result(self, result, stage):
        _require(type(result) is DoctorCacheLookupResult and type(result.key) is DoctorProofCacheKey
                 and canonical_json(result.key.to_dict()) == self.key_json
                 and result.disposition is DoctorCacheDisposition.HIT and result.diagnostic is False
                 and type(result.receipt) is ScopedProofReceipt
                 and _json(result.receipt.to_dict()) == self.receipt_json,
                 "current native cache hit differs from the exact scoped receipt")
        audit = result.audit
        _require(type(audit) is DoctorCacheAuditReceipt and audit.stage is stage
                 and audit.disposition is DoctorCacheDisposition.HIT
                 and audit.authoritative is True and audit.reconstructed is True
                 and audit.key_id == self.expected_key_id
                 and audit.semantic_root_ids == self.prepared.key.semantic_root_ids
                 and audit.receipt_id == result.receipt.receipt_id
                 and audit.assurance == result.receipt.authoritative_assurance.value
                 and not DoctorProofCacheGate._receipt_reasons(self.prepared.key, result.receipt)
                 and result.receipt.authoritative_verdict is ProofVerdict.PROVED
                 and not result.receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED),
                 "native cache audit or local theorem boundary differs")
        return result

    def revalidate(self, stage):
        try:
            with self._operation(), public_replay_without_metadata():
                self._check()
                result = (self.gate.revalidate_for_render(self.prepared.key) if stage is DoctorCacheStage.RENDER
                          else self.gate.revalidate_for_commit(self.prepared.key))
                self.cache_fence.check()
                result = self.result(result, stage)
                self._check()
                return result
        except HeaderProofCacheBridgeError:
            raise
        except Exception as exc:
            raise HeaderProofCacheBridgeError("current scoped header cache reuse refused") from exc


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class HeaderProofCacheAdmission:
    """An immutable historical admission with freshly checked stage methods."""

    _guard: _AdmissionGuard = field(repr=False)

    def __init__(self):
        raise HeaderProofCacheBridgeError("use import_header_proof_to_cache with admitted existing owners")

    @property
    def key(self):
        return DoctorProofCacheKey.from_dict(json.loads(self._guard.key_json))

    @property
    def receipt(self):
        receipt = ProofReceipt.from_dict(json.loads(self._guard.receipt_json))
        # Freeze detached public copies after their native constructors have
        # normalized mappings. Retained canonical bytes remain unchanged.
        object.__setattr__(receipt, "metadata", _freeze(receipt.metadata))
        object.__setattr__(receipt, "resource_usage", _freeze(receipt.resource_usage))
        for item in receipt.evidence:
            object.__setattr__(item, "metadata", _freeze(item.metadata))
        return receipt

    accepted = property(lambda self: True)
    stored = property(lambda self: True)
    completion_authority = property(lambda self: False)
    execution_authority = property(lambda self: False)
    mutation_authority = property(lambda self: False)
    whole_program_authority = property(lambda self: False)
    whole_program_proved = property(lambda self: False)
    resource_enforcement_verified = property(lambda self: False)
    native_gate_bookkeeping_may_write_existing_owner = property(lambda self: True)
    atomic_owner_snapshot = property(lambda self: False)
    new_storage_owners_created = property(lambda self: 0)

    def assert_bound(self, *, gate, scoped_analysis, expected_key_id,
                     expected_receipt_id, expected_owner_path_sha256):
        """Check exact host object and canonical identity joins without storage I/O.

        Expected identities are independently admitted by the host. This method
        does not certify currentness; use ``assert_current`` or a stage method.
        """
        guard = self._guard
        _require(type(self) is HeaderProofCacheAdmission
                 and gate is guard.gate and gate is guard.cache_fence.gate
                 and scoped_analysis is guard.prepared.scoped_analysis
                 and type(expected_key_id) is str and expected_key_id == guard.expected_key_id
                 and canonical_json(guard.prepared.key.to_dict()) == guard.key_json
                 and guard.prepared.key.key_id == expected_key_id
                 and type(expected_owner_path_sha256) is str
                 and _sha(str(guard.cache_fence.path)) == expected_owner_path_sha256
                 and type(expected_receipt_id) is str and guard.receipt_json is not None,
                 "header proof admission differs from exact independently admitted host inputs")
        receipt = ProofReceipt.from_dict(json.loads(guard.receipt_json))
        _require(type(receipt) is ScopedProofReceipt and receipt.receipt_id == expected_receipt_id,
                 "header proof admission differs from independently expected generic receipt")

    def assert_current(self):
        """Recheck source, native custody and both owners without kernel execution."""
        try:
            with public_replay_without_metadata():
                self._guard.check()
        except HeaderProofCacheBridgeError:
            raise
        except Exception as exc:
            raise HeaderProofCacheBridgeError("current scoped header admission refused") from exc

    def assert_cache_binding(self, *, gate, expected_key_id, expected_receipt_id,
                             expected_owner_path_sha256):
        """Pure cache joins; the Terminal host separately joins the signed analysis."""
        self.assert_bound(gate=gate, scoped_analysis=self._guard.prepared.scoped_analysis,
            expected_key_id=expected_key_id, expected_receipt_id=expected_receipt_id,
            expected_owner_path_sha256=expected_owner_path_sha256)

    def inventory_binding(self, *, hammer, gate, scoped_analysis, expected_key_id,
                          expected_receipt_id, expected_native_owner_path_sha256,
                          expected_cache_owner_path_sha256):
        """Snapshot original custody for an independently admitted sealed inventory.

        This checks current source and both original owners without a kernel call.
        Returned identifiers are custody inputs, never a replacement for replay.
        """
        with public_replay_without_metadata():
            self.assert_bound(gate=gate, scoped_analysis=scoped_analysis,
                expected_key_id=expected_key_id, expected_receipt_id=expected_receipt_id,
                expected_owner_path_sha256=expected_cache_owner_path_sha256)
            guard = self._guard
            _require(hammer is guard.hammer and hammer._authoritative_store is guard.store
                     and type(expected_native_owner_path_sha256) is str
                     and _sha(str(guard.native_fence.path)) == expected_native_owner_path_sha256,
                     "original native proof owner differs from independently admitted inventory")
            guard.check()
            receipt = ProofReceipt.from_dict(json.loads(guard.receipt_json))
            result = {"key_id": expected_key_id, "generic_receipt_id": expected_receipt_id,
                "custody_ref": guard.reference.to_dict(), "task_cid": scoped_analysis.task_cid,
                "repository_id": guard.prepared.theorem.roots.repository_id,
                "scope_id": scoped_analysis.report["scope_id"],
                "native_owner_path_sha256": expected_native_owner_path_sha256,
                "cache_owner_path_sha256": expected_cache_owner_path_sha256,
                "generic_receipt_implementations": dict(
                    receipt.metadata["generic_receipt_implementations"])}
            guard.check()
            _require(result["generic_receipt_implementations"] == _implementation_bindings(),
                     "current receipt implementations differ from original inventory binding")
            return result

    def revalidate_for_render(self):
        return self._guard.revalidate(DoctorCacheStage.RENDER)

    def revalidate_for_commit(self):
        return self._guard.revalidate(DoctorCacheStage.COMMIT)


def import_header_proof_to_cache(*, prepared, hammer, custody_ref, expected_key_id,
                                 gate, expected_owner_path_sha256,
                                 expected_receipt_id=None) -> HeaderProofCacheAdmission:
    """Replay the actual pinned kernel, then store only the exact local theorem.

    The caller independently supplies current preparation, both instantiated
    owners, the original custody reference and expected full key/path identities.
    A cached or serialized acceptance flag cannot replace native replay. Stage
    checks subsequently reload original custody/lineage and current source, but
    do not rerun the kernel. Their checks are sequential and may write existing
    native gate bookkeeping; no failed final check returns an accepted handle.
    An optional independent generic receipt identity is checked after fresh
    kernel replay and before the positive cache write. Earlier native bookkeeping
    and executable effects remain possible.
    """
    try:
        with public_replay_without_metadata():
            _require(type(expected_key_id) is str and expected_key_id,
                     "independently expected full native key identity required")
            _require(expected_receipt_id is None or (
                type(expected_receipt_id) is str and expected_receipt_id
                and expected_receipt_id == expected_receipt_id.strip()
                and len(expected_receipt_id.encode("utf-8")) <= 1024
                and all(ord(char) >= 32 for char in expected_receipt_id)),
                "bounded independently expected generic receipt identity required")
            guard = _AdmissionGuard(prepared, hammer, custody_ref, expected_key_id,
                                    gate, expected_owner_path_sha256)
            guard.proof = custody.replay_header_proof_key_custody(prepared=prepared, hammer=hammer,
                custody_ref=guard.reference, expected_key_id=expected_key_id)
            guard.check()
            receipt = _receipt(prepared, guard.proof, guard.reference)
            _require(expected_receipt_id is None or receipt.receipt_id == expected_receipt_id,
                     "fresh local receipt differs from independently expected generic identity")
            guard.receipt_json = _json(receipt.to_dict())
            guard.check()
            stored = gate.put(prepared.key, receipt, complete=True)
            guard.cache_fence.check()
            _require(type(stored) is DoctorCacheStoreResult and stored.stored is True
                     and type(stored.key) is DoctorProofCacheKey
                     and canonical_json(stored.key.to_dict()) == guard.key_json
                     and type(stored.receipt) is ScopedProofReceipt
                     and _json(stored.receipt.to_dict()) == guard.receipt_json,
                     "existing native Doctor gate refused the exact scoped receipt")
            guard.check()
            result = guard.result(gate.lookup(prepared.key), DoctorCacheStage.LOOKUP)
            guard.check()
            _require(result.receipt.receipt_id == receipt.receipt_id, "stored local receipt identity changed")
            handle = object.__new__(HeaderProofCacheAdmission)
            object.__setattr__(handle, "_guard", guard)
            return handle
    except HeaderProofCacheBridgeError:
        raise
    except Exception as exc:
        raise HeaderProofCacheBridgeError("current native header cache admission refused; existing owners may have been written") from exc


__all__ = ["HeaderProofCacheBridgeError", "HeaderProofCacheAdmission", "import_header_proof_to_cache",
           "header_proof_implementation_bindings"]
