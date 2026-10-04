"""Closed finite TLC counter observations under the shared proof resource owner.

Only the generated count=0 / count'=count+1 model is accepted. Checking is
finite and safety-only: it grants no kernel, source, runtime, or completion
claim. Missing tools, incomplete exploration and counterexamples block tasks
which depend on the requested invariant holding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import replace
from pathlib import Path

from .multi_prover_resources import ProverResourceRequest, ProverTask, ProverTaskFailure

PROFILE = "tlc-finite-counter@1"
SCHEMA = "datasets-supervisor-tlc-observation@1"
_MIB = 1024 * 1024
_TLC = "managed-tlc-pinned"


class _Cancellation:
    def __init__(self, context):
        self.context = context

    def is_set(self):
        return self.context.cancelled or self.context.remaining_seconds == 0


def _artifacts(counter_bound, invariant_max):
    from ipfs_datasets_py.logic.backends.tla.compiler import TLACompiler, TLACompileBounds
    from ipfs_datasets_py.logic.software_verification.state import (
        Boundedness, FiniteDomainBound, PredicateRole, StatePredicate, StateSchema,
        StateTypeKind, StateVariable,
    )
    from ipfs_datasets_py.logic.software_verification.transitions import (
        Action, ActionFrame, StateTransitionIR, TransitionKind, TransitionRelation,
    )
    variable = StateVariable("var:count", "count", StateTypeKind.INTEGER, Boundedness.FINITE,
        domain_bound=FiniteDomainBound("bound:count", lower=0, upper=counter_bound))
    predicates = (
        StatePredicate("pred:init", PredicateRole.INITIAL, "count = 0",
            expression={"var:count": 0}, subject_variable_ids=("var:count",)),
        StatePredicate("pred:guard", PredicateRole.GUARD, f"count <= {counter_bound - 1}",
            subject_variable_ids=("var:count",)),
        StatePredicate("pred:next", PredicateRole.NEXT, "count' = count + 1",
            subject_variable_ids=("var:count",)),
        StatePredicate("pred:invariant", PredicateRole.INVARIANT, f"count <= {invariant_max}",
            subject_variable_ids=("var:count",)),
    )
    document = StateTransitionIR(schema=StateSchema(variables=(variable,)), predicates=predicates,
        actions=(Action("action:increment", "Increment", ActionFrame(reads=("var:count",), writes=("var:count",)),
                        guard_predicate_id="pred:guard", next_predicate_id="pred:next"),),
        transitions=(TransitionRelation("rel:next", TransitionKind.ACTION, "Closed bounded increment",
                                        action_ids=("action:increment",), allows_stutter=True),),
        metadata={"profile": PROFILE, "counter_bound": counter_bound, "invariant_max": invariant_max})
    bounds = TLACompileBounds(max_steps=counter_bound, max_variables=1, max_actions=1,
        max_predicates=4, max_integer_span=64, default_integer_lower=0, default_integer_upper=counter_bound)
    original = TLACompiler(bounds=bounds).compile(document, module_name="BoundedCounter")
    # The generic compiler adds BoundedProgress. This closed profile deliberately
    # checks only invariants. Terminal deadlock at the finite step boundary is
    # expected; no deadlock freedom or liveness property is requested.
    config = ("SPECIFICATION Spec\n" + "".join(f"INVARIANT {item}\n" for item in original.safety_properties)
              + "CHECK_DEADLOCK FALSE\n")
    artifact = replace(original, tlc_config_text=config, liveness_properties=(),
        fairness_limitations=("Closed safety-only profile; no liveness, fairness, or deadlock-freedom claim.",),
        source_map=tuple(item for item in original.source_map if item.role != "liveness"))
    return artifact


def _read_bounded(path, limit, signal):
    if signal.is_set():
        raise ProverTaskFailure("TLC task cancelled during preparation", reasons=("cancelled",))
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("native runtime input must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
    finally:
        os.close(descriptor)
    if len(payload) > limit:
        raise ValueError("native runtime input exceeds byte bound")
    if signal.is_set():
        raise ProverTaskFailure("TLC task cancelled during preparation", reasons=("cancelled",))
    return payload


def _runtime(signal):
    from ipfs_datasets_py.logic.backends.installers import state_model
    root = state_model.expand_user_local_root()
    jar = root / "tlc" / state_model.TLC_VERSION / "tla2tools.jar"
    jar_bytes = _read_bounded(jar, 16 * _MIB, signal)
    if hashlib.sha256(jar_bytes).hexdigest() != state_model.TLC_SHA256:
        raise ValueError("TLC JAR differs from the reviewed native pin")
    # The exact verified JAR bytes are copied into each private workspace;
    # a mutable managed launcher never supplies execution or acceptance.
    manifest = json.loads(_read_bounded(root / "manifests" / "tlc.json", 65536, signal))
    java_name = manifest.get("java_executable")
    if type(java_name) is not str or not java_name:
        raise ValueError("installed TLC manifest must select its Java runtime")
    java = Path(java_name).expanduser().resolve(strict=True)
    java_bytes = _read_bounded(java, 32 * _MIB, signal)
    if not java_bytes.startswith(b"\x7fELF") or not os.access(java, os.X_OK):
        raise ValueError("selected Java must be an installed executable ELF binary")
    identity = {"tlc_version": state_model.TLC_VERSION, "tlc_jar_sha256": state_model.TLC_SHA256,
        "java_executable_sha256": hashlib.sha256(java_bytes).hexdigest(),
        "java_executable": str(java), "tlc_jar": str(jar.resolve()),
        "dependency_scope": "Exact pinned TLC JAR and selected Java executable; transitive JVM/native libraries are trusted, not attested."}
    return java, jar_bytes, identity


def make_datasets_tlc_counter_task(
    *, task_id: str, counter_bound: int, invariant_max: int, dependencies=(),
    timeout_seconds: float = 30, memory_mb: int = 512,
) -> ProverTask:
    """Check the invariant count<=invariant_max over count=0..counter_bound.

    A positive task needs complete TLC exploration of exactly counter_bound+1
    distinct states. A counterexample is retained as a failed invariant task.
    Both native probes and the checker run under the actual bridge child;
    this adapter never invokes an installer or skips execution via a cache.
    """
    if type(task_id) is not str or not task_id or task_id != task_id.strip() or len(task_id.encode()) > 1024:
        raise ValueError("task_id must be a bounded nonempty trimmed string")
    if type(counter_bound) is not int or not 1 <= counter_bound <= 64:
        raise ValueError("counter_bound must be an exact integer in [1, 64]")
    if type(invariant_max) is not int or not 0 <= invariant_max <= counter_bound:
        raise ValueError("invariant_max must be an exact integer in [0, counter_bound]")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300):
        raise ValueError("timeout_seconds must be finite in (0, 300]")
    if type(memory_mb) is not int or not 256 <= memory_mb <= 4096:
        raise ValueError("memory_mb must be an integer in [256, 4096]")
    if (type(dependencies) not in (tuple, list) or len(dependencies) > 64
            or any(type(item) is not str or not item or len(item.encode()) > 1024 for item in dependencies)):
        raise ValueError("bounded explicit dependency identifiers required")
    artifact = _artifacts(counter_bound, invariant_max)

    def run(context):
        from .datasets_prover_resources import DatasetsChildResourceLease
        from ipfs_datasets_py.logic.backends.installers.state_model import java_major_version, TLC_MIN_JAVA_MAJOR
        from ipfs_datasets_py.logic.backends.process import BoundedToolRunner, ToolRunLimits, ToolRunRequest, ToolRuntime
        from ipfs_datasets_py.logic.backends.tla import runners
        from ipfs_datasets_py.logic.ir_core.claims import FrozenMap, stable_digest
        from ipfs_datasets_py.logic.ir_core.protocols import BackendRequest, ExecutionBounds, QueryKind

        if not isinstance(context.lease, DatasetsChildResourceLease):
            raise TypeError("native TLC task requires a datasets-backed supervisor child")
        native_grant = context.lease.datasets_parent_lease
        if (native_grant.memory_mb < 128 + memory_mb or native_grant.cpu_slots < 1
                or native_grant.child_process_slots < 1 or context.lease.request.thread_slots < 1):
            raise ProverTaskFailure("native TLC task grant cannot contain declared limits", reasons=("tlc_resource_envelope",))
        signal = _Cancellation(context)
        payload = {"schema": SCHEMA, "profile": PROFILE, "task_id": task_id,
            "counter_bound": counter_bound, "invariant_max": invariant_max, "max_steps": counter_bound,
            "expected_distinct_states": counter_bound + 1, "model_check_passed": False,
            "bindings_match": False, "native_checks": 0, "bounded": True,
            "proof_authority": False, "kernel_authority": False, "source_semantics_verified": False,
            "behavior_authority": False, "execution_authority": False, "completion_authority": False,
            "scope": "Only this finite generated safety model; terminal deadlock checking disabled; no liveness or fairness claim.",
            "profile_transform": "TLACompiler state projection; safety-only configuration; CHECK_DEADLOCK FALSE; no requested BoundedProgress property.",
            "artifact_digest": artifact.artifact_digest, "model_digest": artifact.model_digest,
            "configuration_digest": artifact.tlc_config_digest,
            "datasets_parent_lease_id": native_grant.lease_id,
            "resource_limits": {"resident_memory_bytes": memory_mb * _MIB,
                "address_space_bytes": max(4096, memory_mb * 4) * _MIB,
                "tlc_workers": 1, "rss_guard": "sampled process-tree RSS; not a kernel aggregate cgroup ceiling"}}
        try:
            java, jar_bytes, runtime_identity = _runtime(signal)
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            payload.update(status="unavailable", reason=str(exc)[:1024])
            raise ProverTaskFailure("reviewed installed TLC/Java runtime unavailable", result=payload,
                                   reasons=("tlc_unavailable",)) from exc
        payload["native_runtime"] = runtime_identity
        check_argv = (_TLC, "-config", "BoundedCounter.cfg", "BoundedCounter.tla")
        check_files = {"BoundedCounter.tla": artifact.model_text, "BoundedCounter.cfg": artifact.tlc_config_text}
        native_checks, help_checks = [], []
        jvm_options = ("-Djava.io.tmpdir=.", "-XX:ActiveProcessorCount=1", "-XX:+UseSerialGC", "-Xms16m",
            f"-Xmx{min(128, memory_mb // 4)}m", "-Xss1m", "-XX:MaxMetaspaceSize=128m",
            "-XX:ReservedCodeCacheSize=64m", "-XX:-UsePerfData")

        class _LeasedRunner(BoundedToolRunner):
            def run(self, request, **kwargs):
                if not isinstance(request, ToolRunRequest):
                    raise ValueError("closed TLC runner requires typed native requests")
                kind = "check" if request.argv == check_argv else "help" if request.argv == (_TLC, "-help") else "java" if request.argv == (str(java), "-version") else ""
                if not kind or (kind == "check" and dict(request.input_files) != check_files):
                    raise ValueError("native TLC request changed the closed generated command/model/config")
                remaining = context.remaining_seconds
                if signal.is_set() or remaining is not None and remaining <= 0:
                    raise ProverTaskFailure("TLC task deadline or cancellation", reasons=("cancelled",))
                wall = min(request.limits.timeout_seconds, remaining) if remaining is not None else request.limits.timeout_seconds
                argv = (str(java), *jvm_options, "-version") if kind == "java" else (
                    str(java), *jvm_options, "-cp", "{workspace}/tla2tools.jar", "tlc2.TLC",
                    *(("-workers", "1", "-fpmem", "0.0625", "-seed", "1", *request.argv[1:]) if kind == "check" else ("-help",)))
                files = {**dict(request.input_files), **({"tla2tools.jar": jar_bytes} if kind != "java" else {})}
                bounded = replace(request, argv=argv, input_files=files, runtime=ToolRuntime.NATIVE,
                    limits=replace(request.limits, timeout_seconds=wall, cpu_seconds=max(.001, wall),
                        memory_bytes=max(4096, memory_mb * 4) * _MIB, resident_memory_bytes=memory_mb * _MIB,
                        max_input_bytes=16 * _MIB, max_workspace_bytes=32 * _MIB),
                    environment={key: value for key, value in context.lease.child_environment().items()
                        if key not in {"JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"}})
                kwargs["cancellation"] = signal
                result = super().run(bounded, **kwargs)
                if kind == "check":
                    native_checks.append(result)
                elif kind == "help":
                    help_checks.append(result)
                return result

        runner = _LeasedRunner(base_environment={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
        java_probe = runner.run(ToolRunRequest(argv=(str(java), "-version"),
            limits=ToolRunLimits(timeout_seconds=min(5, timeout_seconds), max_output_bytes=4096)))
        banner = "\n".join(part for part in (java_probe.stdout, java_probe.stderr) if part).strip()
        major = java_major_version(banner)
        if not java_probe.ok or java_probe.output_truncated or major is None or major < TLC_MIN_JAVA_MAJOR:
            payload.update(status="unavailable", java_probe=java_probe.to_dict())
            raise ProverTaskFailure("selected Java runtime did not pass its bounded probe", result=payload,
                                   reasons=("tlc_java_unavailable",))
        runtime_identity.update(java_banner=banner, java_major=major)
        remaining = context.remaining_seconds
        if signal.is_set() or remaining is not None and remaining <= 0:
            raise ProverTaskFailure("TLC task stopped after Java probe", result=payload, reasons=("cancelled",))
        wall = min(timeout_seconds, remaining) if remaining is not None else timeout_seconds
        profile_binding = stable_digest({"profile": PROFILE, "counter_bound": counter_bound,
                                          "invariant_max": invariant_max, "artifact_digest": artifact.artifact_digest})
        request = BackendRequest(request_id=task_id, claim_id=f"{PROFILE}:{counter_bound}:{invariant_max}",
            declaration_id="BoundedCounter", claim_digest=profile_binding,
            obligation_id=f"{task_id}:counter-invariant", obligation_digest=profile_binding,
            assumption_ids=(), logic_family="state_transition", query_kind=QueryKind.SATISFIABILITY,
            bounds=ExecutionBounds(timeout_ms=max(1, math.floor(wall * 1000)), max_steps=counter_bound,
                max_memory_bytes=memory_mb * _MIB, max_output_bytes=32768),
            payload=FrozenMap({"profile": PROFILE, "artifact_digest": artifact.artifact_digest}), requested_backend_id="tlc")
        backend = runners.TLCBackend(runner=runner, executable=_TLC, jvm_probe=lambda: True,
                                     lazy_install=False)
        check_started = time.monotonic()
        outcome = backend.check(artifact, request=request, cancellation=signal)
        check_elapsed_ms = (time.monotonic() - check_started) * 1000
        if not isinstance(outcome, runners.ModelCheckOutcome) or len(native_checks) != 1 or len(help_checks) != 1:
            raise ProverTaskFailure("TLC returned no fresh native result", result=payload, reasons=("tlc_binding_mismatch",))
        process, help_result = native_checks[0], help_checks[0]
        help_text = (help_result.stdout or help_result.stderr).strip()[:512]
        help_valid = (help_result.returncode in (0, 1) and not help_result.cancelled and not help_result.timed_out
            and not help_result.resource_exhausted and not help_result.output_truncated
            and not help_result.error and "TLC" in help_text and "Version" in help_text
            and "provides model checking" in help_text)
        # Whole-check timing includes setup and the bounded version probe. It is
        # operational telemetry; independently bound it to this observed call.
        elapsed = outcome.receipt.elapsed_ms
        timing_valid = (type(elapsed) is int
            and max(0, (process.elapsed_seconds + help_result.elapsed_seconds) * 1000 - 2) <= elapsed
            and elapsed <= check_elapsed_ms + 2)
        combined = "\n".join(part for part in (process.stdout, process.stderr) if part)
        live_status, reason = backend._classify(process, combined)
        trace = None
        if live_status is runners.ModelCheckOutcomeStatus.COUNTEREXAMPLE:
            supplemental = backend._counterexample_from_outputs(process.output_files)
            trace = runners.parse_counterexample_trace(supplemental or combined)
            if supplemental:
                trace = replace(trace, source="checker_counterexample_file")
            trace = runners.replay_counterexample(trace, artifact.source_map)
        expected_receipt = runners.ModelCheckReceipt(tool=runners.ModelCheckerTool.TLC, status=live_status,
            artifact_digest=artifact.artifact_digest, model_digest=artifact.model_digest,
            configuration_digest=artifact.tlc_config_digest, configuration_text=artifact.tlc_config_text,
            executable=_TLC, tool_version=help_text, command=check_argv,
            checked_safety_properties=artifact.safety_properties if live_status not in {
                runners.ModelCheckOutcomeStatus.UNAVAILABLE, runners.ModelCheckOutcomeStatus.ERROR,
                runners.ModelCheckOutcomeStatus.MALFORMED} else (), checked_liveness_properties=(),
            fairness_limitations=artifact.fairness_limitations + backend.capability.limitations,
            capability=backend.capability, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr,
            elapsed_ms=elapsed, timeout_seconds=request.bounds.timeout_ms / 1000,
            output_truncated=process.output_truncated, reason=reason, counterexample=trace, jvm_available=True)
        expected_result = backend._result_from_receipt(expected_receipt, request=request, bounds=request.bounds)
        bindings_match = (timing_valid and outcome.request_digest == request.digest and outcome.interface_version == runners.TLC_BACKEND_VERSION
            and outcome.artifacts is not None and stable_digest(outcome.artifacts.to_dict()) == stable_digest(artifact.to_dict())
            and stable_digest(outcome.receipt.to_dict()) == stable_digest(expected_receipt.to_dict())
            and stable_digest(outcome.result.to_dict()) == stable_digest(expected_result.to_dict()))
        # Telemetry only: solver classification and trace parsing remain in the
        # existing typed backend. Complete exploration of this closed profile
        # must visit every count from zero through the explicit bound.
        state_rows = re.findall(r"([0-9]+) states generated, ([0-9]+) distinct states found, ([0-9]+) states left on queue", combined)
        distinct = int(state_rows[-1][1]) if state_rows else None
        queue_empty = bool(state_rows and state_rows[-1][2] == "0")
        passed = (bindings_match and help_valid and live_status is runners.ModelCheckOutcomeStatus.PASSED
            and process.ok and process.workspace_cleaned and not process.workspace_limit_exceeded
            and not process.output_truncated and distinct == counter_bound + 1 and queue_empty and not signal.is_set())
        # Hash the selected native executable again; shared libraries remain a
        # disclosed trusted runtime dependency, not an asserted immutable image.
        runtime_unchanged = hashlib.sha256(_read_bounded(java, 32 * _MIB, signal)).hexdigest() == runtime_identity["java_executable_sha256"]
        passed = bool(passed and runtime_unchanged)
        payload.update(status=live_status.value, model_check_passed=passed, bindings_match=bindings_match,
            native_checks=len(native_checks), distinct_states=distinct, queue_empty=queue_empty,
            request_digest=request.digest, native_runtime=runtime_identity,
            model_check_receipt=outcome.receipt.to_dict(), typed_result=outcome.result.to_dict(),
            runtime_unchanged=runtime_unchanged)
        if not passed:
            raise ProverTaskFailure("TLC did not establish the exact finite counter invariant", result=payload,
                reasons=("tlc_invariant_not_established" if bindings_match else "tlc_binding_mismatch",))
        return payload

    return ProverTask(task_id=task_id,
        resources=ProverResourceRequest.for_family(task_id, "jvm_model_checking", cpu_slots=1,
            process_slots=1, thread_slots=1, memory_bytes=(128 + memory_mb) * _MIB),
        runner=run, dependencies=tuple(dependencies), timeout_ms=math.ceil(timeout_seconds * 1000))


__all__ = ["PROFILE", "SCHEMA", "make_datasets_tlc_counter_task"]
