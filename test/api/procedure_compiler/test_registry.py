from __future__ import annotations

import importlib.util
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.certificate import (
    CurrentCertificateContext,
    ProcedureCertificateVerifier,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactState,
    ProcedureVersion,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.registry import (
    DRIFT_ACTOR_ID,
    EMPTY_REVISION_ID,
    InMemoryProcedureRegistryStore,
    ProcedureRegistry,
    ProcedureRegistryError,
    ProcedureRegistryRevision,
    RegistryAuthorization,
    RegistryAuthorizationError,
    RegistryCAS,
    RegistryCASError,
    RegistryCASOutcome,
    RegistryCorruptionError,
    RegistryFilter,
    RegistryLifecycleState,
    RegistryOperation,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.verifier import ProcedureVerifier


def _load_certificate_helpers():
    path = Path(__file__).with_name("test_certificate.py")
    spec = importlib.util.spec_from_file_location("_pcpc018_certificate_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load certificate test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_certificate_helpers()
candidate_for = _helpers.candidate_for
evidence_for = _helpers.evidence_for
issuer_for = _helpers.issuer_for
keyring = _helpers.keyring
policy_for = _helpers.policy_for
trust_policy = _helpers.trust_policy
valid_spec = _helpers.valid_spec


ACTOR_ID = "registry-operator@1"
PROCEDURE_ID = "focused-validation-procedure"


def issue_for(spec=None, *, now_ms: int = 100):
    spec = spec or valid_spec()
    candidate = candidate_for(spec)
    evidence = evidence_for(spec)
    policy = policy_for(spec)
    verification = ProcedureVerifier().verify(candidate, evidence, policy, now_ms=now_ms)
    certificate = issuer_for().issue(
        candidate, verification, evidence, policy, now_ms=now_ms
    )
    context = CurrentCertificateContext.from_policy(policy, now_ms=now_ms)
    return spec, candidate, certificate, context


def make_registry(context: CurrentCertificateContext, store=None):
    return ProcedureRegistry(
        ProcedureCertificateVerifier(trust_policy(), keyring()),
        lambda: context,
        store or InMemoryProcedureRegistryStore(),
    )


def auth(
    operation: RegistryOperation,
    procedure_cid: str,
    *,
    expected_old: str = EMPTY_REVISION_ID,
    target_revision: str = EMPTY_REVISION_ID,
    actor_id: str = ACTOR_ID,
    decision_cid: str = "",
    now_ms: int = 100,
) -> RegistryAuthorization:
    return RegistryAuthorization(
        actor_id=actor_id,
        decision_cid=decision_cid or "{}-decision".format(operation.value),
        operation=operation,
        target_procedure_cid=procedure_cid,
        expected_old_revision_id=expected_old,
        target_revision_id=target_revision,
        granted=True,
        issued_at_ms=now_ms,
    )


def register_spec(registry: ProcedureRegistry, spec, certificate, **changes):
    values = {
        "procedure_id": spec.name,
        "procedure_cid": spec.content_id,
        "certificate": certificate,
        "authorization": auth(RegistryOperation.REGISTER, spec.content_id),
        "capability_ids": spec.authority.required_capability_ids,
        "initial_state": RegistryLifecycleState.CANDIDATE,
        "expected_old_revision_id": EMPTY_REVISION_ID,
        "now_ms": 100,
    }
    values.update(changes)
    if "authorization" not in changes:
        values["authorization"] = auth(
            RegistryOperation.REGISTER,
            spec.content_id,
            expected_old=values["expected_old_revision_id"],
        )
    return registry.register(**values)


def promote_head(registry: ProcedureRegistry, mutation, spec, **changes):
    values = {
        "procedure_id": spec.name,
        "target_procedure_cid": spec.content_id,
        "expected_old_revision_id": mutation.revision.revision_id,
        "rollback_target_revision_id": EMPTY_REVISION_ID,
        "now_ms": 100,
    }
    values.update(changes)
    values["authorization"] = changes.get(
        "authorization",
        auth(
            RegistryOperation.PROMOTE,
            spec.content_id,
            expected_old=values["expected_old_revision_id"],
            target_revision=mutation.revision.revision_id,
        ),
    )
    return registry.promote(**values)


def test_closed_lifecycle_and_lookup_are_deterministic() -> None:
    spec, _candidate, certificate, context = issue_for()
    store = InMemoryProcedureRegistryStore()
    registry = make_registry(context, store)

    registered = register_spec(registry, spec, certificate)
    assert registered.cas.accepted
    assert registered.cas.outcome is RegistryCASOutcome.COMMITTED
    assert registered.revision.state is RegistryLifecycleState.CANDIDATE
    assert registered.revision.procedure_cid == spec.content_id
    assert "steps" not in registered.revision.to_dict()
    assert store.get_certificate(certificate.content_id) is not None

    advanced = registry.advance(
        procedure_id=spec.name,
        next_state=RegistryLifecycleState.SHADOW,
        authorization=auth(
            RegistryOperation.ADVANCE,
            spec.content_id,
            expected_old=registered.revision.revision_id,
            target_revision=registered.revision.revision_id,
        ),
        expected_old_revision_id=registered.revision.revision_id,
    )
    assert advanced.revision.state is RegistryLifecycleState.SHADOW
    promoted = promote_head(registry, advanced, spec)
    assert promoted.cas.accepted
    assert promoted.cas.stale is False
    assert promoted.revision.state is RegistryLifecycleState.PROMOTED
    assert promoted.cas.target_procedure_cid == spec.content_id
    assert promoted.cas.expected_old_revision_id == advanced.revision.revision_id
    assert promoted.cas.rollback_target_revision_id == EMPTY_REVISION_ID
    assert promoted.certificate_admission is not None
    assert promoted.certificate_admission.grants_promotion is False

    exact = registry.lookup_exact(spec.content_id, bindings=spec.bindings)
    family = registry.lookup_family(spec.task_family_id)
    chosen = registry.choose_version(spec.name)
    again = registry.filter(
        RegistryFilter(
            procedure_id=spec.name,
            task_family_cid=spec.task_family_id,
            environment_id=spec.bindings.environment_id,
            capability_ids=spec.authority.required_capability_ids,
            max_risk=RiskClass.REPOSITORY_WRITE,
            language_classes=("python",),
            version=spec.version,
        )
    )
    assert exact is not None
    assert exact.revision_id == promoted.revision.revision_id
    assert family == (promoted.revision,)
    assert chosen == promoted.revision
    assert again == (promoted.revision,)
    assert registry.lookup_family(spec.task_family_id) == family
    assert registry.status()["procedure_count"] == 1
    assert registry.get(spec.name).state is RegistryLifecycleState.PROMOTED
    assert len(store.events()) >= 3


def test_family_capability_risk_and_environment_filters() -> None:
    spec_a, _cand_a, cert_a, context = issue_for()
    spec_b = replace(
        valid_spec(name="other-procedure"),
        task_family_id="OTHER_FAMILY",
        version=ProcedureVersion(major=2),
        authority=replace(
            valid_spec().authority,
            required_capability_ids=("capability.other",),
            risk_ceiling=RiskClass.REPOSITORY_WRITE,
        ),
    )
    _spec_b, _cand_b, cert_b, _context_b = issue_for(spec_b)
    registry = make_registry(context)
    registered_a = register_spec(registry, spec_a, cert_a)
    promote_head(registry, registered_a, spec_a)
    registered_b = register_spec(registry, spec_b, cert_b)
    promote_head(registry, registered_b, spec_b)

    family_a = registry.lookup_family(spec_a.task_family_id)
    assert [item.procedure_id for item in family_a] == [spec_a.name]
    family_b = registry.lookup_family(spec_b.task_family_id)
    assert [item.procedure_id for item in family_b] == [spec_b.name]

    by_capability = registry.filter(
        RegistryFilter(capability_ids=("capability.tests",))
    )
    assert [item.procedure_id for item in by_capability] == [spec_a.name]
    by_risk = registry.filter(RegistryFilter(max_risk=RiskClass.OBSERVATION_ONLY))
    assert [item.procedure_id for item in by_risk] == [spec_a.name]
    by_env = registry.filter(
        RegistryFilter(environment_id=spec_a.bindings.environment_id)
    )
    assert [item.procedure_id for item in by_env] == [spec_a.name, spec_b.name]
    assert registry.filter(RegistryFilter(environment_id="other-environment")) == ()
    by_language = registry.filter(RegistryFilter(language_classes=("python",)))
    assert [item.procedure_id for item in by_language] == [spec_a.name, spec_b.name]
    missing = registry.filter(RegistryFilter(language_classes=("rust",)))
    assert missing == ()
    exact_b = registry.lookup_exact(spec_b.content_id, bindings=spec_b.bindings)
    assert exact_b is not None and exact_b.procedure_id == spec_b.name
    assert (
        registry.lookup_exact(
            spec_b.content_id,
            bindings=replace(spec_a.bindings, tree_id="tree-other"),
        )
        is None
    )


def test_version_choice_selects_current_promoted_head() -> None:
    spec_v1, _cand_v1, cert_v1, context = issue_for()
    spec_v2 = replace(valid_spec(), version=ProcedureVersion(major=2))
    _spec_v2, _cand_v2, cert_v2, _context_v2 = issue_for(spec_v2)
    registry = make_registry(context)

    first = register_spec(registry, spec_v1, cert_v1)
    promoted_v1 = promote_head(registry, first, spec_v1)
    assert registry.choose_version(spec_v1.name).version.major == 1

    second = register_spec(
        registry,
        spec_v2,
        cert_v2,
        expected_old_revision_id=promoted_v1.revision.revision_id,
    )
    assert second.revision.state is RegistryLifecycleState.CANDIDATE
    assert registry.get(spec_v1.name).state is RegistryLifecycleState.PROMOTED
    promoted_v2 = registry.promote(
        procedure_id=spec_v1.name,
        target_procedure_cid=spec_v2.content_id,
        authorization=auth(
            RegistryOperation.PROMOTE,
            spec_v2.content_id,
            expected_old=promoted_v1.revision.revision_id,
            target_revision=second.revision.revision_id,
        ),
        expected_old_revision_id=promoted_v1.revision.revision_id,
        rollback_target_revision_id=promoted_v1.revision.revision_id,
    )
    assert promoted_v2.revision.version.major == 2
    assert registry.choose_version(spec_v1.name).revision_id == promoted_v2.revision.revision_id
    assert registry.choose_version(spec_v1.name, spec_v2.version).version.major == 2
    assert registry.choose_version(spec_v1.name, spec_v1.version) is None
    historical = registry.choose_version(
        spec_v1.name, spec_v1.version, usable_only=False
    )
    assert historical is not None
    assert historical.procedure_cid == spec_v1.content_id


def test_procedure_cannot_promote_itself_or_use_its_certificate() -> None:
    spec, _candidate, certificate, context = issue_for()
    registry = make_registry(context)
    registered = register_spec(registry, spec, certificate)

    with pytest.raises(ProcedureRegistryError, match="cannot register itself as promoted"):
        register_spec(
            registry,
            spec,
            certificate,
            initial_state=RegistryLifecycleState.PROMOTED,
        )
    with pytest.raises(RegistryAuthorizationError, match="cannot promote or mutate itself"):
        promote_head(
            registry,
            registered,
            spec,
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
                target_revision=registered.revision.revision_id,
                actor_id=spec.content_id,
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="cannot promote or mutate itself"):
        promote_head(
            registry,
            registered,
            spec,
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
                target_revision=registered.revision.revision_id,
                actor_id=spec.name,
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="cannot promote or mutate itself"):
        promote_head(
            registry,
            registered,
            spec,
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
                target_revision=registered.revision.revision_id,
                actor_id="self",
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="cannot authorize registry mutation"):
        promote_head(
            registry,
            registered,
            spec,
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
                target_revision=registered.revision.revision_id,
                decision_cid=certificate.content_id,
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="exact target revision"):
        registry.promote(
            procedure_id=spec.name,
            target_procedure_cid=spec.content_id,
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
            ),
            expected_old_revision_id=registered.revision.revision_id,
        )
    with pytest.raises(RegistryAuthorizationError):
        RegistryAuthorization(
            actor_id=ACTOR_ID,
            decision_cid="denied",
            operation=RegistryOperation.PROMOTE,
            target_procedure_cid=spec.content_id,
            granted=False,
        )
    assert registry.get(spec.name).state is RegistryLifecycleState.CANDIDATE


def test_expected_old_cas_is_required_for_promote_rollback_and_revoke() -> None:
    spec, _candidate, certificate, context = issue_for()
    registry = make_registry(context)
    registered = register_spec(registry, spec, certificate)

    with pytest.raises(RegistryCASError, match="expected-old") as stale_promote:
        promote_head(
            registry,
            registered,
            spec,
            expected_old_revision_id="stale-head-revision",
            authorization=auth(
                RegistryOperation.PROMOTE,
                spec.content_id,
                expected_old="stale-head-revision",
                target_revision=registered.revision.revision_id,
            ),
        )
    cas = stale_promote.value.cas
    assert isinstance(cas, RegistryCAS)
    assert cas.accepted is False
    assert cas.stale is True
    assert cas.observed_revision_id == registered.revision.revision_id
    assert cas.expected_old_revision_id == "stale-head-revision"
    assert cas.target_procedure_cid == spec.content_id

    promoted = promote_head(registry, registered, spec)
    with pytest.raises(RegistryAuthorizationError, match="expected-old"):
        registry.revoke(
            procedure_id=spec.name,
            target_procedure_cid=spec.content_id,
            authorization=auth(
                RegistryOperation.REVOKE,
                spec.content_id,
                expected_old=registered.revision.revision_id,
                target_revision=promoted.revision.revision_id,
            ),
            expected_old_revision_id=promoted.revision.revision_id,
        )
    revoked = registry.revoke(
        procedure_id=spec.name,
        target_procedure_cid=spec.content_id,
        authorization=auth(
            RegistryOperation.REVOKE,
            spec.content_id,
            expected_old=promoted.revision.revision_id,
            target_revision=promoted.revision.revision_id,
        ),
        expected_old_revision_id=promoted.revision.revision_id,
    )
    assert revoked.cas.accepted
    assert revoked.revision.state is RegistryLifecycleState.REVOKED
    assert revoked.cas.target_procedure_cid == spec.content_id
    assert registry.lookup_exact(spec.content_id) is None
    assert registry.get(spec.name).state is RegistryLifecycleState.REVOKED


def test_rollback_restores_exact_recorded_target() -> None:
    spec_v1, _cand_v1, cert_v1, context = issue_for()
    spec_v2 = replace(valid_spec(), version=ProcedureVersion(major=2))
    _spec_v2, _cand_v2, cert_v2, _context_v2 = issue_for(spec_v2)
    registry = make_registry(context)
    first = register_spec(registry, spec_v1, cert_v1)
    promoted_v1 = promote_head(registry, first, spec_v1)
    second = register_spec(
        registry,
        spec_v2,
        cert_v2,
        expected_old_revision_id=promoted_v1.revision.revision_id,
    )
    promoted_v2 = registry.promote(
        procedure_id=spec_v1.name,
        target_procedure_cid=spec_v2.content_id,
        authorization=auth(
            RegistryOperation.PROMOTE,
            spec_v2.content_id,
            expected_old=promoted_v1.revision.revision_id,
            target_revision=second.revision.revision_id,
        ),
        expected_old_revision_id=promoted_v1.revision.revision_id,
        rollback_target_revision_id=promoted_v1.revision.revision_id,
    )
    assert promoted_v2.revision.rollback_target_revision_id == promoted_v1.revision.revision_id

    with pytest.raises(ProcedureRegistryError, match="exact recorded target"):
        registry.rollback(
            procedure_id=spec_v1.name,
            target_revision_id=second.revision.revision_id,
            authorization=auth(
                RegistryOperation.ROLLBACK,
                spec_v2.content_id,
                expected_old=promoted_v2.revision.revision_id,
                target_revision=second.revision.revision_id,
            ),
            expected_old_revision_id=promoted_v2.revision.revision_id,
        )

    rolled = registry.rollback(
        procedure_id=spec_v1.name,
        target_revision_id=promoted_v1.revision.revision_id,
        authorization=auth(
            RegistryOperation.ROLLBACK,
            spec_v1.content_id,
            expected_old=promoted_v2.revision.revision_id,
            target_revision=promoted_v1.revision.revision_id,
        ),
        expected_old_revision_id=promoted_v2.revision.revision_id,
    )
    assert rolled.cas.accepted
    assert rolled.cas.operation is RegistryOperation.ROLLBACK
    assert rolled.revision.procedure_cid == spec_v1.content_id
    assert rolled.revision.state is RegistryLifecycleState.PROMOTED
    assert registry.lookup_exact(spec_v1.content_id) is not None
    assert registry.lookup_exact(spec_v2.content_id) is None
    history = registry.history(spec_v1.name)
    assert [item.operation for item in history] == [
        RegistryOperation.REGISTER,
        RegistryOperation.PROMOTE,
        RegistryOperation.REGISTER,
        RegistryOperation.PROMOTE,
        RegistryOperation.ROLLBACK,
    ]


def test_stale_certificate_is_not_usable_and_demotes() -> None:
    spec, _candidate, certificate, context = issue_for(now_ms=100)
    stale_context = replace(context, now_ms=certificate.expires_at_ms + 1)
    current = {"value": context}

    def provider():
        return current["value"]

    registry = ProcedureRegistry(
        ProcedureCertificateVerifier(trust_policy(), keyring()),
        provider,
    )
    registered = register_spec(registry, spec, certificate)
    promoted = promote_head(registry, registered, spec)
    assert registry.lookup_exact(spec.content_id) is not None

    current["value"] = stale_context
    assert registry.lookup_exact(spec.content_id) is None
    demoted = registry.get(spec.name)
    assert demoted.state is RegistryLifecycleState.STALE
    assert demoted.operation is RegistryOperation.DEMOTE
    assert demoted.actor_id == DRIFT_ACTOR_ID
    assert demoted.expected_old_revision_id == promoted.revision.revision_id
    assert registry.lookup_exact(spec.content_id) is None

    spec_two = valid_spec(name="second-procedure")
    _spec_two, _cand_two, cert_two, _ctx_two = issue_for(spec_two, now_ms=100)
    with pytest.raises(ProcedureRegistryError, match="not independently current"):
        register_spec(registry, spec_two, cert_two)


def test_corruption_is_quarantined_and_recovered_from_intact_history() -> None:
    spec, _candidate, certificate, context = issue_for()
    store = InMemoryProcedureRegistryStore()
    registry = make_registry(context, store)
    registered = register_spec(registry, spec, certificate)
    promoted = promote_head(registry, registered, spec)
    store.corrupt_revision_payload(
        promoted.revision.revision_id, "actor_id", "tampered-actor"
    )
    with pytest.raises(RegistryCorruptionError, match="canonical content"):
        registry.get(spec.name, demote_stale=False)
    assert store.quarantined()
    recovered = registry.recover(spec.name)
    assert recovered.revision_id == registered.revision.revision_id
    assert recovered.state is RegistryLifecycleState.CANDIDATE
    assert registry.get(spec.name, demote_stale=False).revision_id == registered.revision.revision_id

    store.drop_head(spec.name)
    restored = registry.recover(spec.name)
    assert restored.revision_id == registered.revision.revision_id


def test_concurrent_writers_only_one_cas_commits() -> None:
    spec, _candidate, certificate, context = issue_for()
    registry = make_registry(context)
    registered = register_spec(registry, spec, certificate)
    barrier = threading.Barrier(2)
    committed: list[ProcedureRegistryRevision] = []
    conflicts: list[RegistryCAS] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            mutation = promote_head(registry, registered, spec)
            with lock:
                committed.append(mutation.revision)
        except RegistryCASError as exc:
            assert exc.cas is not None
            with lock:
                conflicts.append(exc.cas)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(committed) == 1
    assert len(conflicts) == 1
    assert committed[0].state is RegistryLifecycleState.PROMOTED
    assert conflicts[0].stale is True
    assert conflicts[0].accepted is False
    assert conflicts[0].observed_revision_id == committed[0].revision_id
    assert registry.get(spec.name).revision_id == committed[0].revision_id


def test_revision_identity_and_generic_receipts_are_content_addressed() -> None:
    spec, _candidate, certificate, context = issue_for()
    registry = make_registry(context)
    registered = register_spec(registry, spec, certificate)
    promoted = promote_head(registry, registered, spec)
    round_trip = ProcedureRegistryRevision.from_dict(promoted.revision.to_dict())
    assert round_trip == promoted.revision
    artifact = promoted.revision.to_artifact()
    assert artifact.subject_cid == spec.content_id
    assert artifact.state is ArtifactState.PROMOTED
    receipt = promoted.cas.to_artifact(spec.bindings, created_at_ms=100)
    assert receipt.facts["accepted"] is True
    assert receipt.facts["expected_old_revision_id"] == registered.revision.revision_id
    assert promoted.cas.new_revision_id == promoted.revision.revision_id
    events = registry.store.events()
    assert events[-1]["receipt"]["facts"]["accepted"] is True
    with pytest.raises(ProcedureRegistryError, match="closed lifecycle"):
        registry.advance(
            procedure_id=spec.name,
            next_state=RegistryLifecycleState.CANDIDATE,
            authorization=auth(
                RegistryOperation.ADVANCE,
                spec.content_id,
                expected_old=promoted.revision.revision_id,
                target_revision=promoted.revision.revision_id,
            ),
            expected_old_revision_id=promoted.revision.revision_id,
        )
