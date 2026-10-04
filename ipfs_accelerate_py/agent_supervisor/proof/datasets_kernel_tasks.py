"""Closed native Lean checks under the datasets/supervisor shared lease.

This profile checks only ``forall n : Nat, n + k = k + n`` for a bounded
literal k. It accepts neither arbitrary source nor a supplied execution result.
Kernel authority is limited to that generated theorem; it establishes no
correspondence to repository code, natural language intent, or runtime behavior.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from pathlib import Path
from dataclasses import replace

from .multi_prover_resources import ProverResourceRequest, ProverTask, ProverTaskFailure

PROFILE = "lean-nat-add-comm@1"
SCHEMA = "datasets-supervisor-lean-observation@1"
_MIB = 1024 * 1024


class _Cancellation:
    def __init__(self, context):
        self.context = context

    def is_set(self):
        return self.context.cancelled or self.context.remaining_seconds == 0


def _profile_source(offset):
    name = f"profile_nat_add_comm_{offset}"
    statement = f"forall n : Nat, n + {offset} = {offset} + n"
    source = (f"import Init\n\ntheorem {name} (n : Nat) : n + {offset} = {offset} + n := by\n"
              f"  exact Nat.add_comm n {offset}\n")
    return name, statement, source


def make_datasets_lean_nat_task(
    *, task_id: str, offset: int, dependencies=(), timeout_seconds: float = 30,
    memory_mb: int = 512,
) -> ProverTask:
    """Create one native Lean Nat-addition task without running or installing it.

    The sampled native process-tree RSS cap is ``memory_mb``; address space
    has a separate finite floor of 4 GiB for the Lean runtime. The supervisor
    reserves an additional 128 MiB for orchestration and bounded capture. Probes and checking share
    one task deadline, one CPU/process slot, and the actual bridge child.
    Native unavailability, malformed evidence or rejection blocks dependents.
    Execution-bypassing deterministic caching is deliberately disabled.
    """
    if type(task_id) is not str or not task_id or task_id != task_id.strip() or len(task_id.encode()) > 1024:
        raise ValueError("task_id must be a bounded nonempty trimmed string")
    if type(offset) is not int or not 0 <= offset <= 65535:
        raise ValueError("offset must be an exact integer in [0, 65535]")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300):
        raise ValueError("timeout_seconds must be finite in (0, 300]")
    if type(memory_mb) is not int or not 128 <= memory_mb <= 4096:
        raise ValueError("memory_mb must be an integer in [128, 4096]")
    if (type(dependencies) not in (tuple, list) or len(dependencies) > 64
            or any(type(item) is not str or not item or len(item.encode()) > 1024 for item in dependencies)):
        raise ValueError("bounded explicit dependency identifiers required")
    name, statement, source = _profile_source(offset)

    def run(context):
        from .datasets_prover_resources import DatasetsChildResourceLease
        from ipfs_datasets_py.logic.backends.kernel.lean import (
            LEAN_KERNEL_BACKEND_VERSION, LeanKernelBackend, LeanKernelOutcome, evaluate_lean_kernel_output,
            instrument_lean_source_for_axioms, extract_generated_proof,
        )
        from ipfs_datasets_py.logic.backends.kernel.wasm import (
            CapabilityPlane, KernelCapabilityState, KernelSourceTreeBinding, KernelToolchainBinding, content_digest,
        )
        from ipfs_datasets_py.logic.backends.process import BoundedToolRunner, ToolRunLimits, ToolRunRequest
        from ipfs_datasets_py.logic.backends.results import ResultAuthority, ResultStatus, TheoremResult
        from ipfs_datasets_py.logic.ir_core.claims import FrozenMap, stable_digest
        from ipfs_datasets_py.logic.families.models import EvidenceAuthority
        from ipfs_datasets_py.logic.ir_core.protocols import BackendRequest, ExecutionBounds, QueryKind

        if not isinstance(context.lease, DatasetsChildResourceLease):
            raise TypeError("native Lean task requires a datasets-backed supervisor child")
        native_grant = context.lease.datasets_parent_lease
        if (native_grant.memory_mb < 128 + memory_mb or native_grant.cpu_slots < 1
                or native_grant.child_process_slots < 1 or context.lease.request.thread_slots < 1):
            raise ProverTaskFailure("native Lean task grant cannot contain declared limits",
                                   reasons=("lean_resource_envelope",))
        signal = _Cancellation(context)
        if signal.is_set():
            raise ProverTaskFailure("Lean task cancelled before launch", reasons=("cancelled",))
        payload = {
            "schema": SCHEMA, "profile": PROFILE, "task_id": task_id, "offset": offset,
            "statement": statement, "domain": "Lean.Nat", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "kernel_accepted": False, "kernel_authority": False,
            "source_semantics_verified": False, "behavior_authority": False,
            "execution_authority": False, "completion_authority": False,
            "datasets_parent_lease_id": context.lease.datasets_parent_lease.lease_id,
            "resource_limits": {"resident_memory_bytes": memory_mb * _MIB,
                "address_space_bytes": max(4096, memory_mb) * _MIB,
                "rss_guard": "sampled process-tree RSS; not a kernel aggregate cgroup ceiling"},
        }
        executable = shutil.which("lean")
        if not executable:
            payload["status"] = "unavailable"
            raise ProverTaskFailure("native Lean executable unavailable", result=payload,
                                   reasons=("lean_unavailable",))
        # A private runner owns both probe and kernel execution. No caller can
        # replace it with an accepted-result fixture. Every launch gets the
        # same cooperative cancellation, remaining deadline and finite limits.
        native_checks = []
        class _LeasedRunner(BoundedToolRunner):
            def run(self, request, **kwargs):
                if not isinstance(request, ToolRunRequest) or request.argv[0] != executable:
                    raise ValueError("closed Lean runner requires its resolved native executable")
                kernel = request.argv == (executable, "--json", "{workspace}/Main.lean")
                if kernel:
                    if dict(request.input_files) != {"Main.lean": instrument_lean_source_for_axioms(source, name)}:
                        raise ValueError("native Lean request changed the closed generated source")
                elif request.argv != (executable, "--version"):
                    raise ValueError("closed Lean runner accepts only version or exact kernel commands")
                remaining = context.remaining_seconds
                if signal.is_set() or remaining is not None and remaining <= 0:
                    raise ProverTaskFailure("Lean task deadline or cancellation", reasons=("cancelled",))
                wall = min(request.limits.timeout_seconds, remaining) if remaining is not None else request.limits.timeout_seconds
                argv = request.argv
                argv = (argv[0], "-j1", *argv[1:])
                limits = replace(request.limits, timeout_seconds=wall,
                    cpu_seconds=max(.001, wall), memory_bytes=max(4096, memory_mb) * _MIB,
                    resident_memory_bytes=memory_mb * _MIB)
                bounded = replace(request, argv=argv, limits=limits,
                    environment={**dict(request.environment), **context.lease.child_environment(),
                        "LEAN_NUM_THREADS": "1", "LEAN_STACK_SIZE_KB": "8192"})
                kwargs["cancellation"] = signal
                outcome = super().run(bounded, **kwargs)
                if kernel:
                    native_checks.append(outcome)
                return outcome

        environment = {key: value for key, value in os.environ.items()
                       if key in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "HOME", "ELAN_HOME"}}
        # The bounded runner isolates HOME. Keep an existing elan installation
        # discoverable without changing that workspace isolation.
        environment.setdefault("ELAN_HOME", str(Path.home() / ".elan"))
        runner = _LeasedRunner(base_environment=environment)
        probe = runner.run(ToolRunRequest(argv=(executable, "--version"),
            limits=ToolRunLimits(timeout_seconds=min(5.0, timeout_seconds), max_output_bytes=4096)))
        version = probe.stdout.strip()
        if (not probe.ok or probe.output_truncated or not re.fullmatch(r"Lean \(version [^\r\n]+\)", version)):
            payload.update(status="unavailable", probe=probe.to_dict())
            raise ProverTaskFailure("native Lean version probe did not complete", result=payload,
                                   reasons=("lean_probe_failed",))
        remaining = context.remaining_seconds
        if signal.is_set() or remaining is not None and remaining <= 0:
            raise ProverTaskFailure("Lean task stopped after version probe", result=payload, reasons=("cancelled",))
        wall = min(timeout_seconds, remaining) if remaining is not None else timeout_seconds
        binding = stable_digest({"profile": PROFILE, "offset": offset, "statement": statement})
        request = BackendRequest(request_id=task_id, claim_id=f"{PROFILE}:{offset}",
            declaration_id=name, claim_digest=binding, obligation_id=f"{task_id}:nat-add-comm",
            obligation_digest=stable_digest({"profile_binding": binding, "source": source}),
            assumption_ids=(), logic_family="lean", query_kind=QueryKind.THEOREM_PROOF,
            bounds=ExecutionBounds(timeout_ms=max(1, math.floor(wall * 1000)),
                max_steps=100_000, max_memory_bytes=memory_mb * _MIB, max_output_bytes=32768),
            payload=FrozenMap({"encoding": "lean4", "source": source}), requested_backend_id="lean")
        native = KernelCapabilityState.available_native(kernel_id="lean", executable=executable, version=version)
        backend = LeanKernelBackend(executable=executable, backend_version=version,
                                    runner=runner, native_probe=lambda: native)
        outcome = backend.run(request, cancellation=signal, plane=CapabilityPlane.NATIVE)
        if not isinstance(outcome, LeanKernelOutcome):
            raise ProverTaskFailure("unexpected native Lean outcome type", result=payload,
                                   reasons=("lean_binding_mismatch",))
        receipt, result = outcome.receipt, outcome.result
        expected_tree = KernelSourceTreeBinding.from_files({"Main.lean": source}, primary_path="Main.lean")
        generated_proof = extract_generated_proof(source)
        expected_toolchain = KernelToolchainBinding(
            toolchain_id=f"toolchain:lean:native:{version}", kernel_id="lean",
            plane=CapabilityPlane.NATIVE, executable=executable, version=version,
            command_template="{lean} --json {source_file}")
        expected_witness = {
            "receipt_id": receipt.receipt_id, "theorem_name": name,
            "theorem_digest": content_digest(source),
            "generated_proof_digest": content_digest(generated_proof),
            "source_tree": expected_tree.to_dict(), "toolchain": expected_toolchain.to_dict(),
            "translation": None,
            "axiom_report": receipt.axiom_report.to_dict() if receipt.axiom_report is not None else None,
        }
        bindings_match = (
            outcome.request_digest == request.digest == receipt.request_digest
            and outcome.source_binding == receipt.source_binding
            and receipt.source_binding.source_digest == content_digest(source)
            and receipt.source_tree == expected_tree
            and receipt.theorem_name == name and receipt.theorem_digest == content_digest(source)
            and receipt.imports == ("Init",) and receipt.translation is None
            and receipt.generated_proof == generated_proof
            and receipt.generated_proof_digest == content_digest(generated_proof)
            and receipt.toolchain == expected_toolchain
            and outcome.interface_version == LEAN_KERNEL_BACKEND_VERSION
            and outcome.capability.native == native
            and receipt.plane is CapabilityPlane.NATIVE
            and result.bounds == request.bounds and not result.assumptions
            and result.backend_id == "lean" and result.backend_version == version
            and result.result_id == f"result:lean:{request.digest[:24]}"
            and stable_digest(result.witness.to_dict()) == stable_digest(expected_witness)
            and stable_digest(result.metadata.to_dict()) == stable_digest({
                "adapter_interface": LEAN_KERNEL_BACKEND_VERSION,
                "capability": outcome.capability.to_dict(), "kernel_receipt": receipt.to_dict(),
                "source_binding": receipt.source_binding.to_dict(),
            })
        )
        axiom = receipt.axiom_report
        live_accepted, live_axiom = False, None
        if len(native_checks) == 1:
            observed = native_checks[0]
            live_accepted, live_axiom, _ = evaluate_lean_kernel_output(observed, declaration=name)
            live_accepted = bool(live_accepted and observed.ok and observed.workspace_cleaned
                                and not observed.output_truncated and not observed.workspace_limit_exceeded)
        accepted = (live_accepted and live_axiom == axiom and bindings_match and receipt.accepted and isinstance(result, TheoremResult)
            and result.status is ResultStatus.PROVED and result.authority is ResultAuthority.THEOREM
            and result.translation_ceiling is EvidenceAuthority.INDEPENDENTLY_CHECKABLE
            and axiom is not None and axiom.declaration == name and not axiom.contains_sorry_ax
            and not axiom.axioms and not result.exceeded_bounds and not signal.is_set())
        payload.update(status=result.status.value, request_digest=request.digest,
            kernel_accepted=accepted, kernel_authority=accepted,
            kernel_receipt=receipt.to_dict(), typed_result=result.to_dict(),
            bindings_match=bindings_match, native_version=version,
            native_checks=len(native_checks), native_execution_observed=len(native_checks) == 1,
            native_output_accepted=live_accepted,
            kernel_scope="Only the generated Nat theorem under the installed trusted Lean Init imports; no repository/source correspondence.")
        if not accepted:
            raise ProverTaskFailure("native Lean did not accept the exact generated theorem", result=payload,
                reasons=("lean_kernel_not_accepted" if bindings_match else "lean_binding_mismatch",))
        return payload

    return ProverTask(task_id=task_id,
        resources=ProverResourceRequest.for_family(task_id, "itp_kernel", cpu_slots=1,
            process_slots=1, thread_slots=1, memory_bytes=(128 + memory_mb) * _MIB),
        runner=run, dependencies=tuple(dependencies), timeout_ms=math.ceil(timeout_seconds * 1000))


__all__ = ["PROFILE", "SCHEMA", "make_datasets_lean_nat_task"]
