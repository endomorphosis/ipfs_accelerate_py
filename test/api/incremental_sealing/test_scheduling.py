"""IPS-033: proof-work scheduling and resource admission."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.scheduling import (
    EVIDENCE_SUBSET,
    PRIORITY_ORDER,
    RESOURCE_EVIDENCE_SUBSET,
    AdmissionVerdict,
    ProofResourcePolicy,
    ProofWorkItem,
    ProofWorkScheduler,
    SchedulingError,
    WorkClass,
    build_proof_schedule,
)


def _item(work_id: str, work_class: WorkClass, **kwargs: object) -> ProofWorkItem:
    payload = {
        "work_id": work_id,
        "unit_id": work_id.replace("w:", "unit/"),
        "work_class": work_class,
    }
    payload.update(kwargs)
    return ProofWorkItem(**payload)  # type: ignore[arg-type]


def test_evidence_subsets_and_priority_order() -> None:
    assert EVIDENCE_SUBSET == "ips/proof-schedule@1"
    assert RESOURCE_EVIDENCE_SUBSET == "ips/resource-admission@1"
    assert PRIORITY_ORDER[0] == "invalidation_check"
    assert PRIORITY_ORDER[-1] == "full_fallback"


def test_priority_and_fan_in_are_deterministic() -> None:
    items = (
        _item("w:exp", WorkClass.EXPENSIVE_DIRECT),
        _item("w:inv", WorkClass.INVALIDATION_CHECK),
        _item("w:cache", WorkClass.CACHE_VERIFICATION),
        _item("w:small", WorkClass.SMALL_INDEPENDENT),
        _item("w:crit", WorkClass.CRITICAL_PATH),
        _item("w:full", WorkClass.FULL_FALLBACK),
    )
    first = build_proof_schedule(items)
    second = build_proof_schedule(tuple(reversed(items)))
    assert [slot.item.work_id for slot in first] == [
        slot.item.work_id for slot in second
    ]
    assert [slot.item.work_id for slot in first] == [
        "w:inv",
        "w:cache",
        "w:small",
        "w:crit",
        "w:exp",
        "w:full",
    ]


def test_dependencies_preserve_publication_order() -> None:
    items = (
        _item("w:child", WorkClass.CRITICAL_PATH, depends_on=("w:parent",), publication_order=2),
        _item("w:parent", WorkClass.SMALL_INDEPENDENT, publication_order=1),
        _item("w:sib", WorkClass.SMALL_INDEPENDENT, publication_order=0),
    )
    schedule = build_proof_schedule(items)
    ids = [slot.item.work_id for slot in schedule]
    assert ids.index("w:parent") < ids.index("w:child")
    assert schedule[0].item.work_id == "w:sib"
    waves = {slot.item.work_id: slot.wave for slot in schedule}
    assert waves["w:child"] > waves["w:parent"]


def test_oversubscribed_work_waits_then_admits_after_release() -> None:
    policy = ProofResourcePolicy(max_cpu=2, max_memory_mb=128, max_parallel=1)
    scheduler = ProofWorkScheduler(policy)
    first = _item("w:a", WorkClass.SMALL_INDEPENDENT, cpu=2, memory_mb=64)
    second = _item("w:b", WorkClass.SMALL_INDEPENDENT, cpu=2, memory_mb=64)
    assert scheduler.admit(first) is AdmissionVerdict.ADMITTED
    assert scheduler.admit(second) is AdmissionVerdict.WAIT
    scheduler.release("w:a")
    assert scheduler.admit(second) is AdmissionVerdict.ADMITTED


def test_missing_gpu_or_simulated_gpu_is_unavailable() -> None:
    scheduler = ProofWorkScheduler(ProofResourcePolicy(max_gpu=0))
    real = _item("w:gpu", WorkClass.EXPENSIVE_DIRECT, gpu=1)
    fake = _item("w:sim", WorkClass.EXPENSIVE_DIRECT, gpu=1, simulated_gpu=True)
    assert scheduler.admit(real) is AdmissionVerdict.UNAVAILABLE
    assert scheduler.admit(fake) is AdmissionVerdict.UNAVAILABLE


def test_independent_units_share_a_wave() -> None:
    items = (
        _item("w:a", WorkClass.SMALL_INDEPENDENT),
        _item("w:b", WorkClass.SMALL_INDEPENDENT),
    )
    schedule = build_proof_schedule(items)
    assert schedule[0].wave == schedule[1].wave == 0


def test_cycle_fails_closed() -> None:
    items = (
        _item("w:a", WorkClass.SMALL_INDEPENDENT, depends_on=("w:b",)),
        _item("w:b", WorkClass.SMALL_INDEPENDENT, depends_on=("w:a",)),
    )
    with pytest.raises(SchedulingError, match="cycle"):
        build_proof_schedule(items)
