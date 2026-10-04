"""Closed Isabelle/HOL Nat theorem checks under the shared datasets owner.

Only a generated addition-commutativity theorem is accepted. Kernel authority
concerns that theorem under the installed trusted HOL image; there is no
repository, intent, cross-family translation, or runtime-behavior claim.
"""
from __future__ import annotations

import hashlib
import math
from ipfs_datasets_py.logic.backends.installers import isabelle_profile as _native_profile

from .multi_prover_resources import ProverResourceRequest, ProverTask, ProverTaskFailure

PROFILE = "isabelle-nat-add-comm@1"
SCHEMA = "datasets-supervisor-isabelle-observation@1"
_MIB = 1024 * 1024
_PROCESS_SLOTS = _native_profile.PROCESS_SLOTS
_CPU_SLOTS = _native_profile.CPU_SLOTS


class _Cancellation:
    def __init__(self, context):
        self.context = context

    def is_set(self):
        return self.context.cancelled or self.context.remaining_seconds == 0


def _read_bounded(path, limit, signal):
    def checkpoint():
        if signal.is_set():
            raise ProverTaskFailure("Isabelle preparation stopped", reasons=("cancelled",))
    return _native_profile.regular_bytes(path, limit, checkpoint)


def _runtime(signal):
    def checkpoint():
        if signal.is_set():
            raise ProverTaskFailure("Isabelle preparation stopped", reasons=("cancelled",))
    return _native_profile.resolve_runtime(checkpoint=checkpoint)


def _profile_source(offset):
    name = f"profile_nat_add_comm_{offset}"
    statement = f"forall n : Isabelle.HOL.nat, n + {offset} = {offset} + n"
    source = (f'theory ProfileNatAddComm\nimports Main\nbegin\nlemma {name}: "(n::nat) + {offset} = {offset} + n"\n'
              '  by (rule add.commute)\nend\n')
    return name, statement, source


def _settings(memory_mb):
    return _native_profile.private_settings(memory_mb)


def make_datasets_isabelle_nat_task(
    *, task_id: str, offset: int, dependencies=(), timeout_seconds: float = 120,
    memory_mb: int = 2048,
) -> ProverTask:
    """Check one generated Isabelle/HOL theorem with fresh native evidence.

    The default task reserves 2304 MiB, three CPU/thread slots and twelve process
    slots for the JVM, Poly/ML and launcher helpers. Callers must provide a
    shared parent large enough to contain this envelope. Native tree RSS is
    sampled at ``memory_mb``; per-process address space is capped at 32 GiB
    because the Poly/ML object runtime reserves large sparse mappings.
    No installer or execution-bypassing cache is invoked.
    """
    if type(task_id) is not str or not task_id or task_id != task_id.strip() or len(task_id.encode()) > 1024:
        raise ValueError("task_id must be a bounded nonempty trimmed string")
    if type(offset) is not int or not 0 <= offset <= 65535:
        raise ValueError("offset must be an exact integer in [0, 65535]")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300):
        raise ValueError("timeout_seconds must be finite in (0, 300]")
    if type(memory_mb) is not int or not 1024 <= memory_mb <= 4096:
        raise ValueError("memory_mb must be an integer in [1024, 4096]")
    if (type(dependencies) not in (tuple, list) or len(dependencies) > 64
            or any(type(item) is not str or not item or len(item.encode()) > 1024 for item in dependencies)):
        raise ValueError("bounded explicit dependency identifiers required")
    name, statement, source = _profile_source(offset)

    def run(context):
        from .datasets_prover_resources import DatasetsChildResourceLease
        from ipfs_datasets_py.logic.backends.kernel import isabelle
        from ipfs_datasets_py.logic.backends.kernel.wasm import (
            CapabilityPlane, KernelCapabilityState, KernelSourceTreeBinding, KernelToolchainBinding, content_digest,
        )
        from ipfs_datasets_py.logic.backends.process import BoundedToolRunner, ToolRunLimits, ToolRunRequest
        from ipfs_datasets_py.logic.backends.results import ResultStatus
        from ipfs_datasets_py.logic.external_provers.isabelle_runtime import add_kernel_audit, theory_command
        from ipfs_datasets_py.logic.ir_core.claims import FrozenMap, stable_digest
        from ipfs_datasets_py.logic.ir_core.protocols import BackendRequest, ExecutionBounds, QueryKind

        if not isinstance(context.lease, DatasetsChildResourceLease):
            raise TypeError("native Isabelle task requires a datasets-backed supervisor child")
        grant = context.lease.datasets_parent_lease
        if (grant.memory_mb < memory_mb + 256 or grant.cpu_slots < _CPU_SLOTS
                or grant.child_process_slots < _PROCESS_SLOTS or context.lease.request.thread_slots < _CPU_SLOTS):
            raise ProverTaskFailure("native Isabelle task grant cannot contain declared limits",
                                   reasons=("isabelle_resource_envelope",))
        signal = _Cancellation(context)
        payload = {"schema": SCHEMA, "profile": PROFILE, "task_id": task_id, "offset": offset,
            "statement": statement, "domain": "Isabelle.HOL.nat", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "kernel_accepted": False, "kernel_authority": False, "bindings_match": False,
            "native_checks": 0, "source_semantics_verified": False, "cross_family_correspondence_verified": False,
            "behavior_authority": False, "execution_authority": False, "completion_authority": False,
            "datasets_parent_lease_id": grant.lease_id,
            "resource_limits": {"resident_memory_bytes": memory_mb * _MIB,
                "address_space_bytes": 32 * 1024 * _MIB, "process_slots_reserved": _PROCESS_SLOTS,
                "jvm_active_processors": 1, "ml_threads": 1, "ml_gc_threads": 1,
                "thread_scope": "computation/GC parallelism; JVM/ML service threads are not a three-OS-thread ceiling",
                "rss_guard": "sampled process-tree RSS; not a kernel aggregate cgroup ceiling",
                "process_guard": "scheduler reservation for trusted launcher helpers; no kernel process-count ceiling"}}
        try:
            executable, runtime = _runtime(signal)
        except (OSError, ValueError, TypeError) as exc:
            payload.update(status="unavailable", reason=str(exc)[:1024])
            raise ProverTaskFailure("supported installed Isabelle runtime unavailable", result=payload,
                                   reasons=("isabelle_unavailable",)) from exc
        payload["native_runtime"] = runtime
        version = runtime["version"]
        settings = _settings(memory_mb)
        settings_files = {".isabelle/etc/settings": settings, f".isabelle/{version}/etc/settings": settings}
        audited_source = add_kernel_audit(source, name)
        command = tuple(theory_command(executable, "ProfileNatAddComm", "{workspace}"))
        native_checks = []
        payload["native_phases"] = []

        class _LeasedRunner(BoundedToolRunner):
            def run(self, request, **kwargs):
                if not isinstance(request, ToolRunRequest):
                    raise ValueError("closed Isabelle runner requires typed requests")
                kernel = request.argv == command
                if kernel:
                    expected_paths = {"ProfileNatAddComm.thy", *settings_files}
                    if set(request.input_files) != expected_paths or request.input_files["ProfileNatAddComm.thy"] != audited_source:
                        raise ValueError("native Isabelle request changed the closed generated theory")
                elif request.argv != (executable, "version"):
                    raise ValueError("closed Isabelle runner accepts only version and exact kernel commands")
                def remaining():
                    duration = context.remaining_seconds
                    if signal.is_set() or duration is not None and duration <= 0:
                        raise ProverTaskFailure("Isabelle task stopped before native launch", reasons=("cancelled",))
                    return request.limits.timeout_seconds if duration is None else duration
                observed, record = _native_profile.run_admitted_phase(request,
                    parent_lease=grant, version=version, memory_mb=memory_mb,
                    remaining=remaining, cancellation=signal, run=super().run,
                    phase="kernel" if kernel else "version", environment=context.lease.child_environment())
                payload["native_phases"].append(record)
                if kernel:
                    native_checks.append(observed)
                return observed

        runner = _LeasedRunner(base_environment={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
        probe = runner.run(ToolRunRequest(argv=(executable, "version"),
            limits=ToolRunLimits(timeout_seconds=min(15, timeout_seconds), max_output_bytes=4096)))
        if (not probe.ok or probe.error or probe.output_truncated or probe.workspace_limit_exceeded
                or not probe.workspace_cleaned or probe.command != (executable, "version")
                or probe.stdout.strip() != version):
            payload.update(status="unavailable", probe=probe.to_dict())
            raise ProverTaskFailure("Isabelle bounded version probe did not complete", result=payload,
                                   reasons=("isabelle_probe_failed",))
        remaining = context.remaining_seconds
        if signal.is_set() or remaining is not None and remaining <= 0:
            raise ProverTaskFailure("Isabelle task stopped after probe", result=payload, reasons=("cancelled",))
        wall = min(timeout_seconds, remaining) if remaining is not None else timeout_seconds
        profile_binding = stable_digest({"profile": PROFILE, "offset": offset, "statement": statement})
        request = BackendRequest(request_id=task_id, claim_id=f"{PROFILE}:{offset}", declaration_id=name,
            claim_digest=profile_binding, obligation_id=f"{task_id}:nat-add-comm",
            obligation_digest=stable_digest({"profile_binding": profile_binding, "source": source}),
            assumption_ids=(), logic_family="isabelle", query_kind=QueryKind.THEOREM_PROOF,
            bounds=ExecutionBounds(timeout_ms=max(1, math.floor(wall * 1000)), max_steps=100_000,
                max_memory_bytes=memory_mb * _MIB, max_output_bytes=32768),
            payload=FrozenMap({"encoding": "isabelle", "source": source, "path": "ProfileNatAddComm.thy"}),
            requested_backend_id="isabelle")
        native = KernelCapabilityState.available_native(kernel_id="isabelle", executable=executable, version=version)
        backend = isabelle.IsabelleKernelBackend(executable=executable, backend_version=version,
            runner=runner, native_probe=lambda: native)
        # The supplied native capability avoids frontend discovery; provide the
        # already-probed identifier so the backend writes versioned user settings.
        backend._runtime_identifier = version
        outcome = backend.run(request, cancellation=signal, plane=CapabilityPlane.NATIVE)
        payload["native_checks"] = len(native_checks)
        if not isinstance(outcome, isabelle.IsabelleKernelOutcome) or len(native_checks) != 1:
            raise ProverTaskFailure("Isabelle returned no fresh typed native result", result=payload,
                                   reasons=("isabelle_binding_mismatch",))
        observed = native_checks[0]
        live_accepted, axiom, diagnostics = isabelle.evaluate_isabelle_kernel_output(observed, declaration=name, source=source)
        live_accepted = bool(live_accepted and observed.ok and not observed.error
                            and observed.command == command and observed.workspace_cleaned
                            and not observed.workspace_limit_exceeded and not observed.output_truncated)
        path = isabelle.correct_isabelle_path_metadata(source, caller_path="ProfileNatAddComm.thy", session_dir=".")
        binding = isabelle.IsabelleSourceBinding.bind(request, source, path_metadata=path)
        tree = KernelSourceTreeBinding.from_files({path.theory_path: source}, primary_path=path.theory_path)
        proof = isabelle.extract_generated_proof(source)
        toolchain = KernelToolchainBinding(toolchain_id=f"toolchain:isabelle:native:{version}",
            kernel_id="isabelle", plane=CapabilityPlane.NATIVE, executable=executable, version=version,
            command_template=path.command_template, metadata=FrozenMap({"theory_name": path.theory_name,
                "theory_path": path.theory_path, "session_dir": path.session_dir, "path_metadata_corrected": False}))
        expected_receipt = isabelle.IsabelleKernelReceipt(request_digest=request.digest, source_binding=binding,
            theorem_name=name, theorem_digest=content_digest(source), imports=("Main",), generated_proof=proof,
            generated_proof_digest=content_digest(proof), toolchain=toolchain, source_tree=tree, path_metadata=path,
            translation=None, axiom_report=axiom, plane=CapabilityPlane.NATIVE, accepted=live_accepted,
            authority_disposition=isabelle.IsabelleAuthorityDisposition.REJECT, diagnostics=diagnostics)
        expected_result = backend._build_result(request=request, binding=binding,
            status=ResultStatus.PROVED if live_accepted else ResultStatus.ERROR,
            usage=isabelle._usage_from_process(observed), receipt=expected_receipt, capability=outcome.capability,
            reason="" if live_accepted else next(iter(diagnostics), "isabelle kernel rejected the proof"),
            diagnostics=diagnostics)
        bindings_match = (outcome.request_digest == request.digest
            and outcome.interface_version == isabelle.ISABELLE_KERNEL_BACKEND_VERSION
            and stable_digest(outcome.source_binding.to_dict()) == stable_digest(binding.to_dict())
            and stable_digest(outcome.capability.native.to_dict()) == stable_digest(native.to_dict())
            and stable_digest(outcome.receipt.to_dict()) == stable_digest(expected_receipt.to_dict())
            and stable_digest(outcome.result.to_dict()) == stable_digest(expected_result.to_dict()))
        # Runtime dependencies are explicitly trusted; detect changes to the
        # selected launcher/configuration files around this invocation.
        _, after_runtime = _runtime(signal)
        runtime_unchanged = stable_digest(runtime) == stable_digest(after_runtime)
        accepted = bool(live_accepted and bindings_match and runtime_unchanged and axiom is not None
            and axiom.declaration == name and not axiom.contains_sorry
            and not axiom.contains_unreviewed_axiomatization and not axiom.residual_axioms and not signal.is_set())
        payload.update(status=outcome.result.status.value, request_digest=request.digest,
            kernel_accepted=accepted, kernel_authority=accepted, bindings_match=bindings_match,
            kernel_receipt=outcome.receipt.to_dict(), typed_result=outcome.result.to_dict(),
            native_execution_observed=True, native_output_accepted=live_accepted, runtime_unchanged=runtime_unchanged,
            kernel_scope="Only the generated Isabelle/HOL Nat theorem under installed trusted Main/HOL/Pure imports; no source or cross-family correspondence.")
        if not accepted:
            raise ProverTaskFailure("Isabelle did not accept the exact generated theorem", result=payload,
                reasons=("isabelle_kernel_not_accepted" if bindings_match else "isabelle_binding_mismatch",))
        return payload

    return ProverTask(task_id=task_id,
        resources=ProverResourceRequest.for_family(task_id, "itp_kernel", cpu_slots=_CPU_SLOTS,
            process_slots=_PROCESS_SLOTS, thread_slots=_CPU_SLOTS, memory_bytes=(256 + memory_mb) * _MIB),
        runner=run, dependencies=tuple(dependencies), timeout_ms=math.ceil(timeout_seconds * 1000))


__all__ = ["PROFILE", "SCHEMA", "make_datasets_isabelle_nat_task"]
