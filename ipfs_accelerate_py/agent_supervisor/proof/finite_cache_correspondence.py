"""Owner-derived cache identities for the closed finite integer-offset profile.

This preparation compiles exact current captured source; it does not run that
source, a solver or Lean. Both key representations derive from the same native
source/contract/domain/tool material. Checked syntax correspondence is distinct
from proving the requested postcondition. The existing identity bridge continues
to reject positive receipt admission, including for apparently matching goals.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time

from .canonical_cache_key_bridge import (
    bridge_canonical_proof_cache_key, unbridge_canonical_proof_cache_key,
)
from .formal_verification_cache import ProofCacheKey


SCHEMA = "finite-integer-cache-correspondence/v1"
PROFILE = "python-integer-offset-finite@1"
MAX_REPORT_BYTES = 2 * 1024 * 1024
FALSE = dict(proof_authority=False, receipt_admission_supported=False,
             execution_authority=False, completion_authority=False,
             source_execution_performed=False, checker_execution_performed=False,
             contract_satisfaction_checked=False, training_executed=False, inference_executed=False)

# Reviewed first-party source/IR/VC/key owners. External runtimes remain the
# native tool policy's explicit trusted dependency scope, not an attestation.
PRODUCERS = (
    __name__,
    "ipfs_accelerate_py.agent_supervisor.proof.canonical_cache_key_bridge",
    "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache",
    "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts",
    "ipfs_datasets_py.duckdb_control.codebase_catalog",
    *("ipfs_datasets_py.logic.software_contracts." + name for name in (
        "codebase_integer_profile", "codebase_finite_integer_observation", "codebase_ir",
        "codebase_resources", "cache", "content", "ast_ir", "duckdb_ast_store", "duckdb_ingest",
        "python_frontend", "semantic_index.snapshot")),
    *("ipfs_datasets_py.logic.software_verification." + name for name in (
        "codebase_pipeline", "codebase_source_adapters", "contracts", "program", "vc", "ir",
        "properties", "receipts", "translations")),
    *("ipfs_datasets_py.logic.ir_core." + name for name in (
        "axes", "identity", "canonical", "claims", "artifacts", "diagnostics", "provenance", "protocols")),
    "ipfs_datasets_py.logic.common.canonical_cache_key",
    "ipfs_datasets_py.logic.families.models",
    "ipfs_datasets_py.logic.backends.smt.compiler",
    "ipfs_datasets_py.logic.backends.smt.differential",
)


class FiniteCacheCorrespondenceError(ValueError):
    """The selected native materials or their key correspondence differ."""


def _require(value, reason):
    if not value:
        raise FiniteCacheCorrespondenceError(reason)


def _raw(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise FiniteCacheCorrespondenceError("bounded inert correspondence JSON required") from error
    _require(len(raw) <= MAX_REPORT_BYTES, "correspondence report exceeds byte budget")
    return raw


def _copy(value):
    return json.loads(_raw(value))


def _pins():
    return {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in PRODUCERS}


def _mapping():
    locations = {
        "source": ["candidate_tree"], "expression": ["obligation.program"],
        "formalization": ["obligation.formalization"], "slice": ["obligation.slice"],
        "obligation": ["obligation.native_request"], "assumptions": ["premises"],
        "bounds": ["obligation.finite_domain"], "translation": ["translator"],
        "provider": ["solver.provider_id"], "environment": ["toolchain"],
        "policy": ["policy.native_policy"], "schema": ["theorem_registry.schema_inventory"],
        "checker": ["kernel.checker_id"], "network_policy": ["policy.network_policy"],
        "evidence_kind": ["policy.evidence_kind"], "authority_ceiling": ["policy.authority_ceiling"],
    }
    return [{"canonical_dimension": name, "material": name,
             "encoding": "literal" if name in {"provider", "checker", "evidence_kind", "authority_ceiling"} else "canonical_JSON_sha256",
             "execution_fields": paths} for name, paths in locations.items()]


def prepare_finite_cache_correspondence(
    *, index, repository, expected_head, contract, inputs, tool_policy,
    observation_limits=None, scheduler=None, parent_lease=None, cancel_event=None,
    timeout_seconds=60,
):
    """Derive both keys from a freshly observed native source owner.

    Source dependencies are closed only for the guarded single-function fragment
    (no imports, calls or global value reads). Python caller/type assumptions and
    the tool policy's un-attested transitive runtime scope remain explicit.
    ``observation_limits`` declares the future finite-observer operation ceiling;
    it never substitutes for the semantic finite domain or an execution receipt.
    Model mode is fixed to off; learned dependency profiles are unsupported here.
    """
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseHead
    from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey, content_digest
    from ipfs_datasets_py.logic.ir_core.axes import LogicEvidenceAuthority, LogicEvidenceKind
    from ipfs_datasets_py.logic.software_contracts import codebase_integer_profile as scalar
    from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as finite
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    _require(type(index) is RepositoryCodebaseIndex and type(index.catalog) is CodebaseCatalog
             and type(index.artifacts) is ImmutableCAS and type(expected_head) is CodebaseHead
             and type(contract) is scalar.IntegerOffsetContract,
             "exact native source owner, head and integer contract required")
    _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 300, "bounded correspondence deadline required")
    domain = finite.build_finite_integer_domain(inputs)
    policy_input = _copy(tool_policy)
    operational = {"timeout_seconds": 60, "memory_mb": 1024} if observation_limits is None else _copy(observation_limits)
    _require(type(operational) is dict and set(operational) == {"timeout_seconds", "memory_mb"}
             and type(operational["timeout_seconds"]) is int and 1 <= operational["timeout_seconds"] <= 300
             and type(operational["memory_mb"]) is int and 1024 <= operational["memory_mb"] <= 16384,
             "closed bounded native observation operation limits required")
    contract = scalar.IntegerOffsetContract.from_dict(contract.to_dict())
    expected_head = CodebaseHead.from_dict(expected_head.to_dict())
    repository = Path(repository).resolve(strict=True)
    deadline = time.monotonic() + timeout_seconds
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease,
            cancel_event=cancel_event, timeout_seconds=min(30., timeout_seconds), memory_mb=512) as lease:
        signal = lease.combined_cancellation_signal(cancel_event)
        def remaining():
            if signal.is_set():
                raise LeaseCancelledError("finite cache correspondence cancelled")
            duration = deadline - time.monotonic()
            if duration <= 0:
                raise LeaseTimeoutError("finite cache correspondence expired")
            return duration
        def observe():
            duration = remaining()
            return index.observe_current(repository, expected_head=expected_head,
                parent_lease=lease, cancel_event=signal, timeout_seconds=duration,
                admission_timeout_seconds=min(30., duration), memory_mb=512)
        pins = _pins()
        policy = finite._policy(policy_input, remaining)
        compiler_python = finite._tool(Path(sys.executable), remaining)
        observed = observe()
        entry = next((item for item in observed.manifest.snapshot.entries if item.path == contract.path), None)
        _require(entry is not None and not entry.is_opaque, "source is absent or opaque in current captured inventory")
        source = index.artifacts.get_bytes(entry.source_cid)
        compiled = scalar.compile_integer_offset(source, contract, revision="snapshot:" + expected_head.snapshot_cid)
        _require(compiled.source_cid == entry.source_cid, "compiled source differs from captured source")
        lowered = compiled.pipeline.obligation_results[0]
        premises = [dict(kind="declared_runtime_assumption", text=text) for text in scalar.ASSUMPTIONS]
        premises += [dict(kind="native_source_model_equation", assertion=item.to_dict())
                     for item in lowered.smt_obligation.assumptions]
        premises = sorted(premises, key=_raw)
        model = {"mode": "off", "model_dependencies": [], "reason": "deterministic_native_integer_profile"}
        source_material = dict(head=expected_head.to_dict(), path=contract.path, source_cid=entry.source_cid,
            source_sha256=hashlib.sha256(source).hexdigest(), source_bytes=len(source),
            code_dependencies={"scope": "guarded_single_function_no_imports_calls_or_global_value_reads",
                               "dependency_cids": [], "annotation_and_runtime_assumptions": list(scalar.ASSUMPTIONS)},
            model=model)
        translation = dict(profile=scalar.PROFILE, producers=pins,
            compiled_cid=compiled.cid, source_binding=compiled.pipeline.bindings.source.to_dict(),
            model=model, scope="native_source_to_ProgramIR_VC_SMT_correspondence; no solver result")
        provider = "finite-observer:sha256:" + pins[finite.__name__]
        checker = "lean-binary:sha256:" + policy["lean"]["sha256"]
        environment = dict(native_environment=policy["environment"], python=policy["python"], lean=policy["lean"],
            dependency_scope=policy["dependency_scope"], host_platform=platform.platform(),
            compiler_python={"version": sys.version, "cache_tag": sys.implementation.cache_tag,
                             "binary": compiler_python}, producers=pins)
        network = {"operation": "preparation_only_no_target_or_checker_execution",
                   "future_observer_network": "no_requested_network; isolation_not_attested",
                   "native_tool_environment": policy["environment"]}
        materials = dict(source=source_material, expression=compiled.pipeline.program.to_dict(),
            formalization={"compiled": compiled.to_dict(), "vc": lowered.vc_obligation.to_dict(),
                           "scope": "conditional_integer_model_restricted_to_declared_finite_domain"},
            slice={"path": contract.path, "function": contract.function_name, "parameter": contract.parameter},
            obligation={"profile": PROFILE, "contract": contract.to_dict(), "finite_domain": domain,
                        "clauses": ["exact_builtin_int_result", "result_equals_parameter_plus_contract_offset"]},
            assumptions=premises, bounds=domain, translation=translation, provider=provider,
            environment=environment,
            policy={"native_tools": policy, "operation_ceiling": operational, "model": model,
                    "trust_scope": "selected_first_party_producer_files_and_native_binary_bytes; external_runtime_scope_is_assumed"},
            schema={"correspondence": SCHEMA, "profile": PROFILE, "compiler": scalar.COMPILED_SCHEMA,
                    "contract": scalar.CONTRACT_SCHEMA, "domain": finite.DOMAIN_SCHEMA, "model": model},
            checker=checker, network_policy=network,
            evidence_kind=LogicEvidenceKind.DECLARATION.value, authority_ceiling=LogicEvidenceAuthority.NONE.value)
        canonical = CanonicalProofCacheKey.build(**materials, source_cid=entry.source_cid)
        execution = ProofCacheKey(
            obligation={"schema": "finite-integer-key-obligation/v1", "native_request": materials["obligation"],
                        "program": materials["expression"], "formalization": materials["formalization"],
                        "slice": materials["slice"], "finite_domain": domain},
            premises=tuple(premises), translator=translation,
            solver={"provider_id": provider, "role": "declared_finite_observer", "python": policy["python"]},
            kernel={"checker_id": checker, "role": "declared_recorded_table_checker", "lean": policy["lean"]},
            toolchain=environment, theorem_registry={"schema_inventory": materials["schema"], "registered_proofs": []},
            policy={"native_policy": materials["policy"], "network_policy": network,
                    "evidence_kind": materials["evidence_kind"], "authority_ceiling": materials["authority_ceiling"]},
            resource_budget={"operation_ceiling": operational, "native_process_limits": policy["process_limits"],
                             "scope": "declared_execution_limits; not_semantic_input_bounds_or_measured_usage"},
            candidate_tree=source_material)
        bridged = bridge_canonical_proof_cache_key(canonical, execution_key=execution)
        # Recheck native source, selected executable bytes and producing code
        # after deriving every identity; no report can replace these owner reads.
        observe()
        _require(finite._policy(policy_input, remaining) == policy
                 and finite._tool(Path(sys.executable), remaining) == compiler_python and _pins() == pins,
                 "native tools or producing code changed during correspondence")
        remaining()
        report = dict(schema=SCHEMA, profile=PROFILE, materials=materials, field_correspondence=_mapping(),
            canonical_key=canonical.to_dict(), canonical_key_id=canonical.key_id,
            execution_key=execution.to_dict(), execution_key_id=execution.key_id,
            bridged_key=bridged.to_dict(), bridged_key_id=bridged.key_id,
            source_observed_before_and_after=True, model_mode="off", **FALSE)
        report["report_sha256"] = content_digest(report)
        return _copy(report)


def verify_finite_cache_correspondence(report, **owner_inputs):
    """Reobserve/recompile, compare the closed report, then recover native keys.

    Returns ``(CanonicalProofCacheKey, ProofCacheKey, bridged ProofCacheKey)``.
    This identity verification does not verify an execution or proof receipt.
    """
    from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey
    value = _copy(report)
    expected = prepare_finite_cache_correspondence(**owner_inputs)
    _require(_raw(value) == _raw(expected), "finite owner-derived correspondence does not replay exactly")
    semantic = CanonicalProofCacheKey.from_dict(expected["canonical_key"])
    execution = ProofCacheKey.from_dict(expected["execution_key"])
    bridged = ProofCacheKey.from_dict(expected["bridged_key"])
    semantic, execution = unbridge_canonical_proof_cache_key(
        bridged, request=semantic, expected_execution_key=execution)
    return semantic, execution, bridged


__all__ = ["SCHEMA", "PROFILE", "PRODUCERS", "FiniteCacheCorrespondenceError",
           "prepare_finite_cache_correspondence", "verify_finite_cache_correspondence"]
