from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler import (
    AdmissionDecision,
    AdmissionKind,
    ArtifactBindings,
    ArtifactState,
    BudgetReservation,
    ConditionOperator,
    EffectClass,
    ExecutionMode,
    FailureTransition,
    IdempotencyClass,
    InMemoryCheckpointStore,
    InMemoryIdempotencyStore,
    InterpreterAdmissionPorts,
    IsolationReservation,
    OperationResult,
    ProcedureAuthorityEnvelope,
    ProcedureCertificate,
    ProcedureCompilerCapabilityError,
    ProcedureEffect,
    ProcedureInterpreter,
    ProcedureInterpreterError,
    ProcedureInvocation,
    ProcedureLocal,
    ProcedureObservation,
    ProcedureOutcomeStatus,
    ProcedureParameter,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    ProofCarryingProcedureCompiler,
    RetryPolicy,
    RiskClass,
    RuntimeCost,
    RuntimeFailureCode,
    RuntimeIdentity,
    StepOperation,
    TraceState,
    TrustedOperation,
    TrustedOperationCatalog,
    ValueType,
    validate_procedure_spec,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.interpreter import (
    CheckpointPhase,
)


class _Clock:
    def __init__(self) -> None:
        self.monotonic = 0

    def now_ms(self) -> int:
        return 100

    def monotonic_ms(self) -> int:
        return self.monotonic

    def wait_ms(self, milliseconds: int) -> None:
        self.monotonic += milliseconds


class _Admissions:
    def __init__(self, rejected: AdmissionKind | None = None) -> None:
        self.rejected = rejected
        self.calls: list[AdmissionKind] = []

    def admit(self, request):
        self.calls.append(request.kind)
        admitted = request.kind is not self.rejected
        return AdmissionDecision(
            admitted=admitted,
            kind=request.kind,
            receipt_cids=("admission-{}".format(request.kind.value),) if admitted else (),
            reason_code="test_rejection" if not admitted else "",
            observed_at_ms=100,
            predicate_value=True if admitted else False,
        )


class _Budgets:
    def __init__(self) -> None:
        self.releases = 0

    def reserve(self, request):
        return BudgetReservation(
            reservation_id="reservation-1",
            token_limit=request.token_limit,
            resource_limit=request.resource_limit,
            time_limit_ms=request.time_limit_ms,
            receipt_cid="budget-receipt-1",
        )

    def release(self, reservation, *, consumed):
        self.releases += 1


class _Isolation:
    def __init__(self, *, fail: bool = False, mismatched: bool = False) -> None:
        self.fail = fail
        self.mismatched = mismatched
        self.releases = 0
        self.compensations = 0

    def acquire(self, request):
        if self.fail:
            raise RuntimeError("external lease owner unavailable")
        return IsolationReservation(
            reservation_id="isolation-1",
            lease_id=request.lease_id,
            fencing_token=request.fencing_token + (1 if self.mismatched else 0),
            scope_paths=request.scope_paths,
            worktree_id="worktree-1" if request.worktree_required else "",
            read_only=request.read_only,
            receipt_cid="isolation-receipt-1",
        )

    def compensate(self, reservation, *, reason_code):
        self.compensations += 1
        return "isolation-compensation-1"

    def release(self, reservation):
        self.releases += 1


class _InterruptingCheckpointStore(InMemoryCheckpointStore):
    def __init__(self, phase: CheckpointPhase) -> None:
        super().__init__()
        self.phase = phase
        self.interrupted = False

    def save(self, checkpoint):
        super().save(checkpoint)
        if checkpoint.phase is self.phase and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt


def _bindings(*, tree: str = "tree-1") -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repository-1",
        repository_commit="commit-1",
        tree_id=tree,
        objective_id="PCPC-G000",
        task_id="PCPC-005",
        contract_revision="contracts-1",
        policy_revision="policy-1",
        environment_id="environment-1",
    )


def _certificate(spec: ProcedureSpec) -> ProcedureCertificate:
    return ProcedureCertificate(
        bindings=spec.bindings,
        procedure_cid=spec.content_id,
        procedure_version=spec.version,
        task_family_cid=spec.task_family_id,
        source_episode_cids=("episode-1",),
        specification_cids=("specification-1",),
        counterexample_set_cid="counterexamples-1",
        operation_catalog_revision="catalog-1",
        effect_policy_revision="effects-1",
        authority_policy_revision="policy-1",
        verification_policy_revision="verification-1",
        repository_families=("repository-family-1",),
        supported_language_classes=("python",),
        supported_framework_classes=("stdlib",),
        risk_ceiling=RiskClass.REPOSITORY_WRITE,
        proof_receipt_cids=("proof-1",),
        test_receipt_cids=("test-1",),
        adversarial_assurance_cids=("assurance-1",),
        held_out_evaluation_cid="held-out-1",
        shadow_evaluation_cid="shadow-1",
        known_limitations=(),
        issuer="test-issuer",
        signature="independently-verified-test-signature",
        issued_at_ms=1,
        expires_at_ms=10_000,
        state=ArtifactState.VERIFIED,
    )


def _invocation(
    spec: ProcedureSpec,
    certificate: ProcedureCertificate,
    *,
    effectful: bool,
    fencing_token: int = 7,
) -> ProcedureInvocation:
    return ProcedureInvocation(
        bindings=spec.bindings,
        procedure_cid=spec.content_id,
        certificate_cid=certificate.content_id,
        registry_revision="registry-1",
        parameters={},
        requested_scope=("src",),
        authority_receipt_cids=("authority-1",),
        idempotency_key="invocation-key-1",
        dry_run=not effectful,
        requested_at_ms=10,
        lease_id="lease-1" if effectful else "",
        fencing_token=fencing_token if effectful else 0,
    )


def _runtime(*, tree: str = "tree-1", effectful: bool = False, fence: int = 7):
    return RuntimeIdentity(
        repository_id="repository-1",
        repository_commit="commit-1",
        tree_id=tree,
        objective_id="PCPC-G000",
        task_id="PCPC-005",
        contract_revision="contracts-1",
        policy_revision="policy-1",
        environment_id="environment-1",
        registry_revision="registry-1",
        operation_catalog_revision="catalog-1",
        now_ms=100,
        active_lease_id="lease-1" if effectful else "",
        fencing_token=fence if effectful else 0,
    )


def _read_spec() -> ProcedureSpec:
    spec = ProcedureSpec(
        bindings=_bindings(),
        name="deterministic-read-validation",
        version=ProcedureVersion(major=1),
        task_family_id="family-read-validation",
        entry_step_id="validate",
        parameters=(),
        locals=(ProcedureLocal("ok", ValueType.BOOLEAN),),
        preconditions=(
            ProcedurePrecondition(
                "current-tree",
                "binding:tree_id",
                ConditionOperator.CURRENT,
                evidence_producer="tree-admission",
                evidence_type="tree-receipt",
            ),
        ),
        declared_reads=("src",),
        declared_effects=(ProcedureEffect("validation-effect", EffectClass.VALIDATION),),
        steps=(
            ProcedureStep(
                "validate",
                StepOperation.CHECK_POSTCONDITION,
                "check-postcondition-v1",
                output_bindings={"ok": "local:ok"},
                declared_effect_ids=("validation-effect",),
                required_authority_ids=("run-validation",),
                timeout_ms=5_000,
                evidence_outputs=("validated",),
            ),
        ),
        postconditions=(
            ProcedurePostcondition(
                "accepted-postcondition",
                "local:ok",
                ConditionOperator.EQUALS,
                True,
                "postcondition-admission",
                "postcondition-receipt",
            ),
        ),
        observations=(
            ProcedureObservation(
                "validated",
                "check-postcondition-v1",
                "local:ok",
                ConditionOperator.EQUALS,
                True,
                "validation-receipt",
            ),
        ),
        validation=ProcedureValidationPlan(("validate",), ("validated",)),
        authority=ProcedureAuthorityEnvelope(
            "policy-1",
            ("run-validation",),
            (),
            (StepOperation.CHECK_POSTCONDITION,),
            RiskClass.OBSERVATION_ONLY,
        ),
        resources=ProcedureResourceEnvelope(
            wall_time_ms=10_000,
            cpu_time_ms=10_000,
            memory_bytes=1_000_000,
            disk_bytes=1_000_000,
            model_token_limit=0,
            model_call_limit=0,
        ),
        terminal_step_ids=("validate",),
        scope_paths=("src",),
        provenance_cids=("source-1",),
        state=ArtifactState.SHADOW,
    )
    return validate_procedure_spec(spec)


def _write_spec(*, rollback: bool = False, retry: bool = False) -> ProcedureSpec:
    patch_failure = (
        FailureTransition.ROLLBACK
        if rollback
        else (FailureTransition.RETRY if retry else FailureTransition.ABORT)
    )
    retry_policy = (
        RetryPolicy(
            max_attempts=2,
            retryable_failure_codes=("provider_unavailable",),
            requires_new_evidence=False,
        )
        if retry
        else RetryPolicy()
    )
    effects = [
        ProcedureEffect(
            "write-effect",
            EffectClass.REPOSITORY_WRITE,
            targets=("src/file.py",),
            reversible=True,
        ),
        ProcedureEffect("validation-effect", EffectClass.VALIDATION),
    ]
    steps = [
        ProcedureStep(
            "patch",
            StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
            "patch-template-v1",
            output_bindings={"patched": "local:patched"},
            declared_effect_ids=("write-effect",),
            required_authority_ids=("write-worktree",),
            timeout_ms=5_000,
            retry_policy=retry_policy,
            idempotency=IdempotencyClass.NEVER_REPLAY_UNKNOWN,
            failure_transition=patch_failure,
            failure_target="rollback-plan" if rollback else "",
            evidence_outputs=("patch-observation",),
            next_step_id="validate",
        ),
        ProcedureStep(
            "validate",
            StepOperation.CHECK_POSTCONDITION,
            "check-postcondition-v1",
            output_bindings={"ok": "local:ok"},
            declared_effect_ids=("validation-effect",),
            required_authority_ids=("run-validation",),
            timeout_ms=5_000,
            evidence_outputs=("validated",),
        ),
    ]
    rollback_plans = ()
    if rollback:
        effects.append(
            ProcedureEffect(
                "rollback-effect",
                EffectClass.ROLLBACK,
                targets=("src/file.py",),
                reversible=True,
            )
        )
        steps.append(
            ProcedureStep(
                "rollback-step",
                StepOperation.ROLLBACK,
                "rollback-v1",
                output_bindings={"restored": "local:restored"},
                declared_effect_ids=("rollback-effect",),
                required_authority_ids=("write-worktree",),
                timeout_ms=5_000,
                evidence_outputs=("rollback-observation",),
            )
        )
        rollback_plans = (
            ProcedureRollback(
                "rollback-plan",
                ("write-effect",),
                ("rollback-step",),
                ("rollback-observation",),
                "pre-patch-tree-1",
            ),
        )
    spec = ProcedureSpec(
        bindings=_bindings(),
        name="bounded-write-procedure",
        version=ProcedureVersion(major=1),
        task_family_id="family-bounded-write",
        entry_step_id="patch",
        locals=(
            ProcedureLocal("patched", ValueType.BOOLEAN),
            ProcedureLocal("ok", ValueType.BOOLEAN),
            ProcedureLocal("restored", ValueType.BOOLEAN),
        ),
        preconditions=(
            ProcedurePrecondition(
                "current-tree",
                "binding:tree_id",
                ConditionOperator.CURRENT,
                evidence_producer="tree-admission",
                evidence_type="tree-receipt",
            ),
        ),
        declared_reads=("src",),
        declared_effects=tuple(effects),
        steps=tuple(steps),
        postconditions=(
            ProcedurePostcondition(
                "accepted-postcondition",
                "local:ok",
                ConditionOperator.EQUALS,
                True,
                "postcondition-admission",
                "postcondition-receipt",
            ),
        ),
        observations=(
            ProcedureObservation(
                "patch-observation",
                "patch-template-v1",
                "local:patched",
                ConditionOperator.EQUALS,
                True,
                "write-receipt",
            ),
            ProcedureObservation(
                "validated",
                "check-postcondition-v1",
                "local:ok",
                ConditionOperator.EQUALS,
                True,
                "validation-receipt",
            ),
            ProcedureObservation(
                "rollback-observation",
                "rollback-v1",
                "local:restored",
                ConditionOperator.EQUALS,
                True,
                "rollback-receipt",
            ),
        ),
        validation=ProcedureValidationPlan(("validate",), ("validated",)),
        rollback=rollback_plans,
        authority=ProcedureAuthorityEnvelope(
            "policy-1",
            ("write-worktree", "run-validation"),
            (),
            (
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                StepOperation.CHECK_POSTCONDITION,
                StepOperation.ROLLBACK,
            ),
            RiskClass.REPOSITORY_WRITE,
        ),
        resources=ProcedureResourceEnvelope(
            wall_time_ms=10_000,
            cpu_time_ms=10_000,
            memory_bytes=1_000_000,
            disk_bytes=1_000_000,
            model_token_limit=0,
            model_call_limit=0,
        ),
        terminal_step_ids=("validate",),
        scope_paths=("src",),
        provenance_cids=("source-1",),
        state=ArtifactState.CANDIDATE,
    )
    return validate_procedure_spec(spec)


def _interpreter(
    operations,
    admissions=None,
    checkpoints=None,
    idempotency=None,
    isolation=None,
    budgets=None,
    clock=None,
):
    admission = admissions or _Admissions()
    ports = InterpreterAdmissionPorts(admission, admission, admission, admission)
    return ProcedureInterpreter(
        operation_catalog=TrustedOperationCatalog("catalog-1", operations),
        admissions=ports,
        isolation=isolation or _Isolation(),
        budget_reservations=budgets or _Budgets(),
        checkpoints=checkpoints or InMemoryCheckpointStore(),
        idempotency=idempotency or InMemoryIdempotencyStore(),
        clock=clock or _Clock(),
    )


def _read_operation(handler):
    return TrustedOperation(
        StepOperation.CHECK_POSTCONDITION,
        "check-postcondition-v1",
        handler,
        allowed_effect_ids=("validation-effect",),
        read_only=True,
        output_types={"ok": ValueType.BOOLEAN},
    )


def _read_success(_request):
    return OperationResult(
        True,
        outputs={"ok": True},
        observed_effect_ids=("validation-effect",),
        evidence_cids=("operation-validation-1",),
        cost=RuntimeCost(resource_units=1, elapsed_ms=2),
    )


def test_deterministic_success_and_idempotent_terminal_replay():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    admissions = _Admissions()
    interpreter = _interpreter((_read_operation(handler),), admissions=admissions)
    first = interpreter.execute(
        spec,
        invocation,
        certificate,
        _runtime(),
        mode=ExecutionMode.SHADOW,
    )
    replay = interpreter.execute(
        spec,
        invocation,
        certificate,
        _runtime(),
        mode=ExecutionMode.SHADOW,
    )

    assert first.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert replay.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert first.trace.state is TraceState.COMPLETE
    # Historical success is never trusted as current completion.  Recovery
    # re-admits the checkpoint, authority, validation, and postcondition while
    # preserving operation single-flight.
    assert replay.trace.state is TraceState.RECOVERED
    assert replay.resumed is True
    assert calls == 1
    assert AdmissionKind.CHECKPOINT in admissions.calls
    assert admissions.calls.count(AdmissionKind.VALIDATION) == 2
    assert admissions.calls.count(AdmissionKind.POSTCONDITION) == 2
    assert first.failure is None
    assert first.receipt.outcome_cid == first.outcome.content_id


def test_certificate_current_binding_precondition_and_allowlist_fail_closed():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),))
    with pytest.raises(ProcedureInterpreterError) as missing:
        interpreter.execute(spec, invocation, None, _runtime(), mode=ExecutionMode.SHADOW)
    assert missing.value.reason_code == RuntimeFailureCode.CERTIFICATE_REQUIRED.value

    with pytest.raises(ProcedureInterpreterError) as stale:
        interpreter.execute(
            spec, invocation, certificate, _runtime(tree="other-tree"), mode=ExecutionMode.SHADOW
        )
    assert stale.value.reason_code == RuntimeFailureCode.BINDING_MISMATCH.value

    rejected = _Admissions(AdmissionKind.PRECONDITION)
    rejecting = _interpreter((_read_operation(handler),), admissions=rejected)
    with pytest.raises(ProcedureInterpreterError) as precondition:
        rejecting.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert precondition.value.reason_code == "test_rejection"

    no_operation = _interpreter(())
    with pytest.raises(ProcedureInterpreterError) as unknown:
        no_operation.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert unknown.value.reason_code == RuntimeFailureCode.UNKNOWN_OPERATION.value
    assert calls == 0


def test_authority_rejection_prevents_dispatch():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    admission = _Admissions(AdmissionKind.AUTHORITY)
    interpreter = _interpreter((_read_operation(handler),), admissions=admission)
    with pytest.raises(ProcedureInterpreterError):
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert calls == 0


def test_scope_escape_is_typed_and_validation_is_not_run():
    spec = _write_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    validation_calls = 0

    def patch(_request):
        return OperationResult(
            True,
            outputs={"patched": True},
            observed_effect_ids=("write-effect",),
            changed_paths=("outside/file.py",),
            evidence_cids=("write-receipt-1",),
        )

    def validate(_request):
        nonlocal validation_calls
        validation_calls += 1
        return _read_success(_request)

    isolation = _Isolation()
    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(validate),
        ),
        isolation=isolation,
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.outcome.status is ProcedureOutcomeStatus.ROLLED_BACK
    assert result.failure.failure_code == RuntimeFailureCode.SCOPE_ESCAPE.value
    assert result.outcome.rollback_receipt_cids == ("isolation-compensation-1",)
    assert "isolation-compensation-1" in result.receipt.admitted_evidence_cids
    assert isolation.compensations == 1
    assert validation_calls == 0


def test_started_unobserved_is_unknown_and_never_blindly_retried():
    spec = _write_spec(retry=True)
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    calls = 0

    def patch(_request):
        nonlocal calls
        calls += 1
        return OperationResult(
            False,
            failure_code="provider_unavailable",
            retryable=True,
            external_outcome_observed=False,
        )

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                maximum_retries=1,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        )
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.outcome.status is ProcedureOutcomeStatus.INCOMPLETE
    assert result.failure.failure_code == RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value
    assert result.trace.state is TraceState.INTERRUPTED
    assert calls == 1


def test_restart_from_started_checkpoint_never_dispatches_again():
    spec = _write_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    checkpoints = InMemoryCheckpointStore()
    calls = 0

    def interrupted(_request):
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                interrupted,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        ),
        checkpoints=checkpoints,
    )
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    recovered = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert recovered.failure.failure_code == RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value
    assert recovered.resumed is True
    assert calls == 1


def test_rollback_requires_observed_compensation_and_external_admission():
    spec = _write_spec(rollback=True)
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)

    def patch(_request):
        return OperationResult(
            False,
            observed_effect_ids=("write-effect",),
            changed_paths=("src/file.py",),
            evidence_cids=("write-failure-receipt",),
            failure_code="patch_failed",
        )

    def rollback(_request):
        return OperationResult(
            True,
            outputs={"restored": True},
            observed_effect_ids=("rollback-effect",),
            changed_paths=("src/file.py",),
            evidence_cids=("rollback-operation-receipt",),
        )

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
            TrustedOperation(
                StepOperation.ROLLBACK,
                "rollback-v1",
                rollback,
                allowed_effect_ids=("rollback-effect",),
                read_only=False,
                output_types={"restored": ValueType.BOOLEAN},
            ),
        )
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.outcome.status is ProcedureOutcomeStatus.ROLLED_BACK
    assert result.outcome.rollback_receipt_cids


def test_stale_fencing_is_rejected_before_mutation():
    spec = _write_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True, fencing_token=6)
    called = False

    def patch(_request):
        nonlocal called
        called = True
        raise AssertionError("must not dispatch")

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        )
    )
    with pytest.raises(ProcedureInterpreterError) as stale:
        interpreter.execute(spec, invocation, certificate, _runtime(effectful=True, fence=7))
    assert stale.value.reason_code == RuntimeFailureCode.STALE_FENCING.value
    assert called is False


@pytest.mark.parametrize("mismatched", [False, True])
def test_external_isolation_acquisition_failure_is_typed(mismatched):
    spec = _write_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    isolation = _Isolation(fail=not mismatched, mismatched=mismatched)
    calls = 0

    def patch(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("isolation failure must prevent dispatch")

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        ),
        isolation=isolation,
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.outcome.status is ProcedureOutcomeStatus.REFUSED
    assert result.failure.failure_code == RuntimeFailureCode.ISOLATION_ACQUISITION_FAILED.value
    assert calls == 0


def test_concurrent_same_invocation_is_single_flight():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            interpreter.execute,
            spec,
            invocation,
            certificate,
            _runtime(),
            mode=ExecutionMode.SHADOW,
        )
        assert entered.wait(timeout=3)
        second = pool.submit(
            interpreter.execute,
            spec,
            invocation,
            certificate,
            _runtime(),
            mode=ExecutionMode.SHADOW,
        )
        time.sleep(0.02)
        release.set()
        one = first.result(timeout=3)
        two = second.result(timeout=3)
    assert calls == 1
    assert one.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert two.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert sorted((one.resumed, two.resumed)) == [False, True]


def test_runtime_facade_refuses_synthesis_and_promotion():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    compiler = ProofCarryingProcedureCompiler(_interpreter((_read_operation(_read_success),)))
    result = compiler.invoke(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert result.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    with pytest.raises(ProcedureCompilerCapabilityError):
        compiler.synthesize(spec)
    with pytest.raises(ProcedureCompilerCapabilityError):
        compiler.promote(spec)


def test_runtime_values_reject_floats_before_dispatch():
    with pytest.raises(ProcedureInterpreterError, match="floating-point"):
        OperationResult(True, outputs={"confidence": 0.5})


def test_forged_terminal_checkpoint_is_structurally_rejected():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    checkpoints = InMemoryCheckpointStore()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),), checkpoints=checkpoints)
    interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    terminal = checkpoints.load(invocation.content_id)
    assert terminal is not None
    checkpoints.save(replace(terminal, executed_step_ids=()))

    with pytest.raises(ProcedureInterpreterError) as forged:
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert forged.value.reason_code == RuntimeFailureCode.CHECKPOINT_INVALID.value
    assert calls == 1


def test_recovery_requires_independent_checkpoint_admission():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    admissions = _Admissions()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),), admissions=admissions)
    interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    admissions.rejected = AdmissionKind.CHECKPOINT
    with pytest.raises(ProcedureInterpreterError):
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert calls == 1


def test_terminal_success_recovery_readmits_current_validation_and_postconditions():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    admissions = _Admissions()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),), admissions=admissions)
    interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    admissions.rejected = AdmissionKind.VALIDATION
    recovered = interpreter.execute(
        spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW
    )
    assert recovered.outcome.status is ProcedureOutcomeStatus.FAILED
    assert recovered.failure.failure_code == RuntimeFailureCode.VALIDATION_FAILED.value
    assert calls == 1


def test_successful_observed_checkpoint_resumes_after_step_without_redispatch():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    checkpoints = _InterruptingCheckpointStore(CheckpointPhase.STEP_OBSERVED)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),), checkpoints=checkpoints)
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    recovered = interpreter.execute(
        spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW
    )
    assert recovered.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert recovered.resumed is True
    assert calls == 1


def test_failed_observed_checkpoint_terminates_without_retry_dispatch():
    spec = _write_spec(retry=True)
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    checkpoints = _InterruptingCheckpointStore(CheckpointPhase.STEP_OBSERVED)
    calls = 0

    def patch(_request):
        nonlocal calls
        calls += 1
        return OperationResult(
            False,
            failure_code="provider_unavailable",
            retryable=True,
            external_outcome_observed=True,
            new_evidence=True,
        )

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                maximum_retries=1,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        ),
        checkpoints=checkpoints,
    )
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    recovered = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert recovered.outcome.status is ProcedureOutcomeStatus.FAILED
    assert recovered.failure.failure_code == RuntimeFailureCode.OPERATION_FAILED.value
    assert calls == 1


def test_rolling_back_checkpoint_never_redispatches_compensation():
    spec = _write_spec(rollback=True)
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    checkpoints = _InterruptingCheckpointStore(CheckpointPhase.ROLLING_BACK)
    isolation = _Isolation()
    rollback_calls = 0

    def patch(_request):
        return OperationResult(
            False,
            observed_effect_ids=("write-effect",),
            changed_paths=("src/file.py",),
            evidence_cids=("write-failure-receipt",),
            failure_code="patch_failed",
        )

    def rollback(_request):
        nonlocal rollback_calls
        rollback_calls += 1
        raise AssertionError("ambiguous rollback must not dispatch")

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
            TrustedOperation(
                StepOperation.ROLLBACK,
                "rollback-v1",
                rollback,
                allowed_effect_ids=("rollback-effect",),
                read_only=False,
                output_types={"restored": ValueType.BOOLEAN},
            ),
        ),
        checkpoints=checkpoints,
        isolation=isolation,
    )
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    recovered = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert rollback_calls == 0
    assert recovered.outcome.status is ProcedureOutcomeStatus.ROLLED_BACK
    assert recovered.failure.failure_code == RuntimeFailureCode.ROLLBACK_FAILED.value
    assert "isolation-compensation-1" in recovered.outcome.rollback_receipt_cids


def test_changed_path_without_effect_identity_is_compensated_before_receipt():
    spec = _write_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    isolation = _Isolation()

    def patch(_request):
        return OperationResult(
            True,
            outputs={"patched": True},
            changed_paths=("src/file.py",),
            evidence_cids=("changed-path-receipt",),
        )

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        ),
        isolation=isolation,
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.failure.failure_code == RuntimeFailureCode.EFFECT_VIOLATION.value
    assert result.outcome.status is ProcedureOutcomeStatus.ROLLED_BACK
    assert result.outcome.rollback_receipt_cids == ("isolation-compensation-1",)
    assert "isolation-compensation-1" in result.receipt.admitted_evidence_cids
    assert isolation.compensations == 1


@pytest.mark.parametrize("source", ["literal:/tmp/escape", "literal:../escape"])
def test_operation_literal_path_escape_is_rejected_before_handler(source):
    base = _read_spec()
    step = replace(base.steps[0], input_bindings={"target": source})
    spec = validate_procedure_spec(replace(base, steps=(step,)))
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    operation = replace(
        _read_operation(handler),
        input_types={"target": ValueType.RELATIVE_PATH},
    )
    interpreter = _interpreter((operation,))
    with pytest.raises(ProcedureInterpreterError) as unsafe:
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert unsafe.value.reason_code == RuntimeFailureCode.SCOPE_ESCAPE.value
    assert calls == 0


def test_parameter_and_output_values_enforce_types_and_path_safety():
    base = _read_spec()
    parameter = ProcedureParameter("target", ValueType.RELATIVE_PATH)
    step = replace(base.steps[0], input_bindings={"target": "parameter:target"})
    spec = validate_procedure_spec(replace(base, parameters=(parameter,), steps=(step,)))
    certificate = _certificate(spec)
    invocation = replace(
        _invocation(spec, certificate, effectful=False),
        parameters={"target": 42},
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    operation = replace(
        _read_operation(handler),
        input_types={"target": ValueType.RELATIVE_PATH},
    )
    interpreter = _interpreter((operation,))
    with pytest.raises(ProcedureInterpreterError) as unsafe:
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert unsafe.value.reason_code == RuntimeFailureCode.SCOPE_ESCAPE.value
    assert calls == 0

    safe_invocation = replace(invocation, parameters={"target": "src/file.py"})

    def unsafe_output(_request):
        return OperationResult(
            True,
            outputs={"ok": "/tmp/escape"},
            observed_effect_ids=("validation-effect",),
            evidence_cids=("operation-validation-1",),
        )

    unsafe_operation = replace(operation, handler=unsafe_output)
    result = _interpreter((unsafe_operation,)).execute(
        spec,
        safe_invocation,
        certificate,
        _runtime(),
        mode=ExecutionMode.SHADOW,
    )
    assert result.failure.failure_code == RuntimeFailureCode.SCOPE_ESCAPE.value


@pytest.mark.parametrize("unsafe_key", ["/etc/passwd", "../outside"])
def test_structured_mapping_key_path_escape_is_rejected_before_handler(unsafe_key):
    base = _read_spec()
    parameter = ProcedureParameter("config", ValueType.STRUCTURED)
    step = replace(base.steps[0], input_bindings={"config": "parameter:config"})
    spec = validate_procedure_spec(replace(base, parameters=(parameter,), steps=(step,)))
    certificate = _certificate(spec)
    invocation = replace(
        _invocation(spec, certificate, effectful=False),
        parameters={"config": {unsafe_key: True}},
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    operation = replace(
        _read_operation(handler),
        input_types={"config": ValueType.STRUCTURED},
    )
    interpreter = _interpreter((operation,))
    with pytest.raises(ProcedureInterpreterError) as unsafe:
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert unsafe.value.reason_code == RuntimeFailureCode.SCOPE_ESCAPE.value
    assert calls == 0


def test_two_interpreters_share_store_owned_single_flight_lock():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    checkpoints = InMemoryCheckpointStore()
    idempotency = InMemoryIdempotencyStore()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return _read_success(request)

    operation = _read_operation(handler)
    first_interpreter = _interpreter((operation,), checkpoints=checkpoints, idempotency=idempotency)
    second_interpreter = _interpreter(
        (operation,), checkpoints=checkpoints, idempotency=idempotency
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            first_interpreter.execute,
            spec,
            invocation,
            certificate,
            _runtime(),
            mode=ExecutionMode.SHADOW,
        )
        assert entered.wait(timeout=3)
        second = pool.submit(
            second_interpreter.execute,
            spec,
            invocation,
            certificate,
            _runtime(),
            mode=ExecutionMode.SHADOW,
        )
        time.sleep(0.02)
        release.set()
        one = first.result(timeout=3)
        two = second.result(timeout=3)
    assert calls == 1
    assert one.outcome.status is ProcedureOutcomeStatus.SUCCEEDED
    assert two.outcome.status is ProcedureOutcomeStatus.SUCCEEDED


def test_certificate_risk_ceiling_cannot_understate_procedure_authority():
    spec = _write_spec()
    certificate = replace(_certificate(spec), risk_ceiling=RiskClass.OBSERVATION_ONLY)
    invocation = _invocation(spec, certificate, effectful=True)
    called = False

    def patch(_request):
        nonlocal called
        called = True
        raise AssertionError("risk mismatch must prevent dispatch")

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        )
    )
    with pytest.raises(ProcedureInterpreterError) as risk:
        interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert risk.value.reason_code == RuntimeFailureCode.CERTIFICATE_REJECTED.value
    assert called is False


def test_retry_backoff_exhausts_wall_budget_before_second_dispatch():
    base = _write_spec(retry=True)
    retry = replace(base.steps[0].retry_policy, backoff_ms=10)
    patch_step = replace(base.steps[0], retry_policy=retry, timeout_ms=10)
    validation_step = replace(base.steps[1], timeout_ms=10)
    resources = replace(base.resources, wall_time_ms=10)
    spec = validate_procedure_spec(
        replace(base, steps=(patch_step, validation_step), resources=resources)
    )
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=True)
    clock = _Clock()
    calls = 0

    def patch(_request):
        nonlocal calls
        calls += 1
        return OperationResult(
            False,
            failure_code="provider_unavailable",
            retryable=True,
            external_outcome_observed=True,
        )

    interpreter = _interpreter(
        (
            TrustedOperation(
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                "patch-template-v1",
                patch,
                allowed_effect_ids=("write-effect",),
                read_only=False,
                maximum_retries=1,
                output_types={"patched": ValueType.BOOLEAN},
            ),
            _read_operation(_read_success),
        ),
        clock=clock,
    )
    result = interpreter.execute(spec, invocation, certificate, _runtime(effectful=True))
    assert result.failure.failure_code == RuntimeFailureCode.TIME_BUDGET_EXHAUSTED.value
    assert calls == 1


def test_ready_recovery_readmits_authority_and_preconditions_before_dispatch():
    spec = _read_spec()
    certificate = _certificate(spec)
    invocation = _invocation(spec, certificate, effectful=False)
    checkpoints = _InterruptingCheckpointStore(CheckpointPhase.READY)
    admissions = _Admissions()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter(
        (_read_operation(handler),),
        checkpoints=checkpoints,
        admissions=admissions,
    )
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    admissions.rejected = AdmissionKind.PRECONDITION
    with pytest.raises(ProcedureInterpreterError):
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert calls == 0


def test_segment_glob_does_not_cross_repository_path_boundaries():
    base = _read_spec()
    spec = validate_procedure_spec(replace(base, declared_reads=("src/*",), scope_paths=("src/*",)))
    certificate = _certificate(spec)
    invocation = replace(
        _invocation(spec, certificate, effectful=False),
        requested_scope=("src/nested/file.py",),
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _read_success(request)

    interpreter = _interpreter((_read_operation(handler),))
    with pytest.raises(ProcedureInterpreterError) as scope:
        interpreter.execute(spec, invocation, certificate, _runtime(), mode=ExecutionMode.SHADOW)
    assert scope.value.reason_code == RuntimeFailureCode.SCOPE_ESCAPE.value
    assert calls == 0
