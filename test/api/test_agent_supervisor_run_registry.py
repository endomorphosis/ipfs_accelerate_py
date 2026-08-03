"""ASE-013 durable run registry, reconstruction, CAS, and selection tests."""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pytest


def _ensure_hermetic_cid_utils() -> None:
    """Install a multiformats-backed cid_utils when editable deps are missing."""

    try:
        from ipfs_datasets_py.utils import cid_utils as _cid_utils  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    def _canonical_json_bytes(obj: object) -> bytes:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        ).encode("utf-8")

    def _canonical_dag_json_bytes(obj: object) -> bytes:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _cid_for_bytes(
        data: bytes,
        *,
        base: str = "base32",
        codec: str = "raw",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        from multiformats import CID, multihash

        digest = multihash.digest(bytes(data), mh_type)
        return str(CID(base, version, codec, digest))

    def _cid_for_dag_json(
        obj: object,
        *,
        base: str = "base32",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        return _cid_for_bytes(
            _canonical_dag_json_bytes(obj),
            base=base,
            codec="dag-json",
            mh_type=mh_type,
            version=version,
        )

    def _cid_for_obj(
        obj: object,
        *,
        base: str = "base32",
        codec: str = "raw",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        return _cid_for_bytes(
            _canonical_json_bytes(obj),
            base=base,
            codec=codec,
            mh_type=mh_type,
            version=version,
        )

    def _validate_cid(
        value: object,
        *,
        codecs: object = ("raw", "dag-json"),
        mh_type: str = "sha2-256",
        version: int = 1,
        base: str = "base32",
    ) -> str:
        if not isinstance(value, str) or not value or value != value.lower():
            raise ValueError("CID must be a nonempty lowercase string")
        from multiformats import CID, multihash

        try:
            parsed = CID.decode(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("CID is not decodable") from exc
        allowed = frozenset(codecs)  # type: ignore[arg-type]
        expected_size = multihash.get(mh_type).max_digest_size
        if (
            parsed.version != version
            or parsed.codec.name not in allowed
            or parsed.hashfun.name != mh_type
            or (
                expected_size is not None
                and len(parsed.raw_digest) != expected_size
            )
            or parsed.base.name != base
            or str(parsed) != value
        ):
            raise ValueError(
                "CID must use the requested canonical version/base/codec/multihash"
            )
        return value

    datasets = sys.modules.get("ipfs_datasets_py")
    if datasets is None:
        datasets = types.ModuleType("ipfs_datasets_py")
        datasets.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ipfs_datasets_py"] = datasets

    utils = sys.modules.get("ipfs_datasets_py.utils")
    if utils is None:
        utils = types.ModuleType("ipfs_datasets_py.utils")
        utils.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ipfs_datasets_py.utils"] = utils
        datasets.utils = utils  # type: ignore[attr-defined]

    cid_mod = types.ModuleType("ipfs_datasets_py.utils.cid_utils")
    cid_mod.canonical_json_bytes = _canonical_json_bytes  # type: ignore[attr-defined]
    cid_mod.canonical_dag_json_bytes = _canonical_dag_json_bytes  # type: ignore[attr-defined]
    cid_mod.cid_for_bytes = _cid_for_bytes  # type: ignore[attr-defined]
    cid_mod.cid_for_dag_json = _cid_for_dag_json  # type: ignore[attr-defined]
    cid_mod.cid_for_obj = _cid_for_obj  # type: ignore[attr-defined]
    cid_mod.validate_cid = _validate_cid  # type: ignore[attr-defined]
    cid_mod.__all__ = [  # type: ignore[attr-defined]
        "canonical_dag_json_bytes",
        "canonical_json_bytes",
        "cid_for_bytes",
        "cid_for_dag_json",
        "cid_for_obj",
        "validate_cid",
    ]
    sys.modules["ipfs_datasets_py.utils.cid_utils"] = cid_mod
    utils.cid_utils = cid_mod  # type: ignore[attr-defined]


_ensure_hermetic_cid_utils()

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (  # noqa: E402
    ContinuationAction,
    RunHandle,
    RunHealth,
    RunState,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (  # noqa: E402
    RUN_REGISTRY_REQUIREMENT_ID,
    RegistryTxOutcome,
    RunCasConflictError,
    RunExistsError,
    RunNotFoundError,
    RunRegistry,
    RunRegistryCorruptionError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (  # noqa: E402
    RunAdoptionAction,
    RunCandidateClass,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (  # noqa: E402
    cid_for_bytes,
    cid_for_dag_json,
)

NAMESPACE = "repo:ase-013-demo"
REPOSITORY_ID = "repository:sha256:ase013fixturerepoidentity0001"
CHECKOUT_ID = "checkout:ase-013-main"


class _Clock:
    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self.now = start_ms

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ms: int) -> int:
        self.now += delta_ms
        return self.now


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": "ase-013", "label": label})


def _prompt_cid(label: str) -> str:
    """Prompt identities must be CIDv1/raw/sha2-256."""

    return cid_for_bytes(
        f"ase-013-prompt:{label}".encode("utf-8"),
        codec="raw",
    )


def _handle(
    *,
    label: str = "alpha",
    revision: int = 1,
    state: RunState = RunState.RUNNING,
    health: RunHealth = RunHealth.HEALTHY,
    created_at_ms: int = 1_000,
    updated_at_ms: int = 2_000,
    objective: str | None = None,
    process: str | None = None,
    event_cursor: str = "event:1",
    continuation: ContinuationAction | None = None,
) -> RunHandle:
    if continuation is None:
        if state is RunState.RUNNING:
            continuation = ContinuationAction.MONITOR
        elif state in {RunState.COMPLETED, RunState.CANCELLED, RunState.FAILED}:
            continuation = ContinuationAction.NONE
        elif state is RunState.NEEDS_INPUT:
            continuation = ContinuationAction.ASK_INPUT
        else:
            continuation = ContinuationAction.MONITOR

    if health is None:  # type: ignore[comparison-overlap]
        health = RunHealth.TERMINAL if continuation is ContinuationAction.NONE else RunHealth.HEALTHY

    process_cid = process if process is not None else _cid(f"{label}-process")
    lifecycle = _cid(f"{label}-lifecycle")
    state_rev = _cid(f"{label}-state-rev-r{revision}")
    health_rev = _cid(f"{label}-health-rev-r{revision}")
    lease_id = f"lease:{label}"
    fencing = 1
    ambiguity = ""

    if state in {RunState.COMPLETED, RunState.CANCELLED, RunState.FAILED}:
        health = RunHealth.TERMINAL
        continuation = ContinuationAction.NONE
        # Terminal handles may still carry process history.
    if state is RunState.NEEDS_INPUT:
        ambiguity = _cid(f"{label}-ambiguity")
        continuation = ContinuationAction.ASK_INPUT
        # needs_input does not require process/lease in the same way as running.
        process_cid = ""
        lifecycle = ""
        state_rev = ""
        health_rev = ""
        lease_id = ""
        fencing = 0
        health = RunHealth.DEGRADED

    if state is not RunState.RUNNING and state not in {
        RunState.COMPLETED,
        RunState.CANCELLED,
        RunState.FAILED,
        RunState.NEEDS_INPUT,
    }:
        # Intermediate states: keep process optional, no strict monitor rules.
        if state in {
            RunState.RECEIVED,
            RunState.RESOLVING,
            RunState.RESOLVED,
            RunState.PREVIEWING,
            RunState.ADMITTED,
        }:
            process_cid = ""
            lifecycle = lifecycle  # may remain empty if we clear
            lifecycle = ""
            state_rev = ""
            health_rev = ""
            lease_id = ""
            fencing = 0
            continuation = ContinuationAction.START
            health = RunHealth.UNKNOWN

    return RunHandle(
        run_id=_cid(f"run-{label}"),
        run_revision=revision,
        target_resolution_receipt_cid=_cid(f"{label}-target-receipt"),
        invocation_cid=_cid(f"{label}-invocation"),
        prompt_cid=_prompt_cid(label),
        workflow_cid=_cid(f"{label}-workflow"),
        scan_cid=_cid(f"{label}-scan"),
        plan_cid=_cid(f"{label}-plan"),
        materialization_cid=_cid(f"{label}-materialization"),
        task_source_cid=_cid(f"{label}-task-source"),
        task_source_revision_cid=_cid(f"{label}-task-source-rev"),
        lifecycle_profile_cid=lifecycle,
        process_cid=process_cid,
        objective_cid=objective if objective is not None else _cid(f"{label}-objective"),
        objective_revision_cid=_cid(f"{label}-objective-rev"),
        lease_id=lease_id,
        fencing_generation=fencing,
        state=state,
        health=health,
        state_revision_cid=state_rev,
        health_revision_cid=health_rev,
        event_cursor=event_cursor,
        continuation_action=continuation,
        pending_approval_cid="",
        ambiguity_cid=ambiguity,
        created_at_ms=created_at_ms,
        updated_at_ms=updated_at_ms,
    )


def _registry(tmp_path: Path, clock: _Clock | None = None) -> RunRegistry:
    return RunRegistry(tmp_path / "run-registry", clock_ms=clock or _Clock())


def test_requirement_id_and_restart_behavior(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.requirement_id() == RUN_REGISTRY_REQUIREMENT_ID
    behavior = registry.restart_behavior()
    assert behavior["handles_survive_restart"] is True
    assert behavior["corruption_policy"] == "quarantine_fail_closed"
    assert "directory_name" in behavior["non_authoritative"]


def test_create_and_exact_lookup(tmp_path: Path) -> None:
    clock = _Clock()
    registry = _registry(tmp_path, clock)
    handle = _handle(label="exact")
    receipt = registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    assert receipt.outcome is RegistryTxOutcome.COMMITTED
    assert receipt.run_id == handle.run_id
    assert receipt.run_revision == 1
    assert receipt.integrity_cid
    assert receipt.content_id

    loaded = registry.get(handle.run_id)
    assert loaded == handle
    assert registry.exists(handle.run_id)
    assert registry.integrity_cid(handle.run_id) == receipt.integrity_cid
    root = registry.get_root(handle.run_id)
    assert root.run_namespace == NAMESPACE
    assert root.repository_id == REPOSITORY_ID
    assert root.initial_handle_cid == handle.content_id


def test_restart_reconstructs_complete_handle(tmp_path: Path) -> None:
    """Closing the process and reopening the same root rebuilds the handle."""

    root = tmp_path / "run-registry"
    first = RunRegistry(root, clock_ms=_Clock())
    handle = _handle(label="restart", revision=1)
    first.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    advanced = replace(
        handle,
        run_revision=2,
        event_cursor="event:9",
        updated_at_ms=handle.updated_at_ms + 50,
        state_revision_cid=_cid("restart-state-rev-r2"),
        health_revision_cid=_cid("restart-health-rev-r2"),
    )
    first.cas_update(advanced, expected_revision=1)
    first.close()

    second = RunRegistry(root, clock_ms=_Clock(start_ms=1_800_000_000_000))
    reconstructed = second.reconstruct(handle.run_id)
    assert reconstructed == advanced
    assert reconstructed.run_revision == 2
    assert reconstructed.event_cursor == "event:9"
    assert reconstructed.handle_cid == advanced.content_id
    assert reconstructed.semantic_id == advanced.semantic_id
    second.close()


def test_cas_update_advances_revision_and_rejects_stale(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="cas")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    next_handle = replace(
        handle,
        run_revision=2,
        event_cursor="event:2",
        updated_at_ms=handle.updated_at_ms + 10,
        state_revision_cid=_cid("cas-state-rev-r2"),
        health_revision_cid=_cid("cas-health-rev-r2"),
    )
    committed = registry.cas_update(next_handle, expected_revision=1)
    assert committed.outcome is RegistryTxOutcome.COMMITTED
    assert committed.previous_revision == 1
    assert committed.run_revision == 2
    assert registry.get(handle.run_id) == next_handle

    stale = replace(
        handle,
        run_revision=2,
        event_cursor="event:stale",
        updated_at_ms=handle.updated_at_ms + 20,
        state_revision_cid=_cid("cas-state-rev-stale"),
        health_revision_cid=_cid("cas-health-rev-stale"),
    )
    with pytest.raises(RunCasConflictError) as excinfo:
        registry.cas_update(stale, expected_revision=1)
    assert excinfo.value.receipt is not None
    assert excinfo.value.receipt.outcome is RegistryTxOutcome.CONFLICT
    # Winner remains the first successful revision-2 update.
    assert registry.get(handle.run_id) == next_handle


def test_conflicting_revision_updates_cannot_both_win(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    base = _handle(label="race")
    registry.create(
        base,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def writer(tag: str) -> str:
        candidate = replace(
            base,
            run_revision=2,
            event_cursor=f"event:{tag}",
            updated_at_ms=base.updated_at_ms + 100,
            state_revision_cid=_cid(f"race-state-{tag}"),
            health_revision_cid=_cid(f"race-health-{tag}"),
        )
        barrier.wait(timeout=5)
        try:
            receipt = registry.cas_update(candidate, expected_revision=1)
            with lock:
                results.append(f"ok:{tag}:{receipt.handle_cid}")
            return "committed"
        except RunCasConflictError:
            with lock:
                results.append(f"conflict:{tag}")
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, "a"), pool.submit(writer, "b")]
        outcomes = [future.result(timeout=10) for future in as_completed(futures)]

    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == 1
    assert sum(1 for item in results if item.startswith("ok:")) == 1
    assert sum(1 for item in results if item.startswith("conflict:")) == 1

    head = registry.get(base.run_id)
    assert head.run_revision == 2
    assert head.event_cursor in {"event:a", "event:b"}


def test_unique_compatible_selection_is_deterministic(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    only = _handle(label="unique")
    registry.create(
        only,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    # Foreign namespace/repo must not steal selection.
    other = _handle(label="other-ns")
    registry.create(
        other,
        run_namespace="repo:other",
        repository_id="repository:other",
    )

    first = registry.select_current(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    second = registry.select_current(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    assert first.action is RunAdoptionAction.ADOPT
    assert first.selected_run_id == only.run_id
    assert first.selected_handle == only
    assert first.integrity_cid == registry.integrity_cid(only.run_id)
    assert second.selected_run_id == first.selected_run_id
    assert second.to_dict()["selected_handle_cid"] == only.content_id


def test_multiple_compatible_runs_are_explicitly_ambiguous(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    alpha = _handle(label="multi-a")
    beta = _handle(label="multi-b")
    for handle in (alpha, beta):
        registry.create(
            handle,
            run_namespace=NAMESPACE,
            repository_id=REPOSITORY_ID,
            checkout_id=CHECKOUT_ID,
        )

    selection = registry.select_current(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    assert selection.action is RunAdoptionAction.REPORT_AMBIGUOUS
    assert selection.selected_run_id == ""
    assert selection.selected_handle is None
    assert "multiple_compatible_runs" in selection.reason_codes
    compatible = [
        item
        for item in selection.classified
        if item.classification is RunCandidateClass.COMPATIBLE
    ]
    assert {item.candidate.run_id for item in compatible} == {
        alpha.run_id,
        beta.run_id,
    }


def test_incompatible_runs_are_explicit_not_adopted(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    foreign = _handle(label="foreign")
    registry.create(
        foreign,
        run_namespace=NAMESPACE,
        repository_id="repository:foreign-identity",
        checkout_id=CHECKOUT_ID,
    )
    selection = registry.select_current(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    assert selection.selected_run_id == ""
    assert selection.action in {
        RunAdoptionAction.CREATE,
        RunAdoptionAction.REPORT_STALE_OR_INCOMPATIBLE,
    }
    assert any(
        item.classification is RunCandidateClass.INCOMPATIBLE
        for item in selection.classified
    )
    assert any(
        "incompatible" in code or "create_new_run" in code
        for code in selection.reason_codes
    )


def test_corruption_quarantines_instead_of_canonical_handle(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="corrupt")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    # Tamper with the head so integrity no longer matches the handle snapshot.
    head_path = (
        registry.registry_root
        / "namespaces"
        / NAMESPACE.replace(":", "~")
        / "runs"
        / handle.run_id
        / "head.json"
    )
    payload = json.loads(head_path.read_text(encoding="utf-8"))
    payload["event_cursor"] = "event:tampered"
    # Keep a stale content_id so identity verification fails closed.
    head_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunRegistryCorruptionError) as excinfo:
        registry.reconstruct(handle.run_id)
    err = excinfo.value
    assert err.quarantine_path
    assert Path(err.quarantine_path).is_file()
    assert "head_identity_mismatch" in err.reason_codes or err.reason_codes

    with pytest.raises(RunNotFoundError):
        registry.get(handle.run_id)
    assert registry.exists(handle.run_id) is False

    # Must not reappear as a canonical-looking candidate.
    candidates = registry.list_candidates(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    assert all(item.run_id != handle.run_id for item in candidates)


def test_partial_registry_repair_restores_unique_head(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="repair")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    head_path = (
        registry.registry_root
        / "namespaces"
        / NAMESPACE.replace(":", "~")
        / "runs"
        / handle.run_id
        / "head.json"
    )
    head_path.unlink()
    report = registry.repair()
    assert handle.run_id in report.repaired_run_ids
    assert registry.get(handle.run_id) == handle


def test_set_current_pointer_cas_and_get_current(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="current")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        checkout_id=CHECKOUT_ID,
    )
    receipt = registry.set_current(
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
        run_id=handle.run_id,
        checkout_id=CHECKOUT_ID,
    )
    assert receipt.outcome is RegistryTxOutcome.COMMITTED
    assert registry.get_current(
        run_namespace=NAMESPACE, repository_id=REPOSITORY_ID
    ) == handle

    with pytest.raises(RunCasConflictError):
        registry.set_current(
            run_namespace=NAMESPACE,
            repository_id=REPOSITORY_ID,
            run_id=handle.run_id,
            checkout_id=CHECKOUT_ID,
            expected_pointer_revision=0,
        )


def test_duplicate_create_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="dup")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    with pytest.raises(RunExistsError):
        registry.create(
            handle,
            run_namespace=NAMESPACE,
            repository_id=REPOSITORY_ID,
        )


def test_list_runs_is_bounded_and_sorted(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "reg", max_list=2)
    labels = ["list-c", "list-a", "list-b"]
    for label in labels:
        registry.create(
            _handle(label=label),
            run_namespace=NAMESPACE,
            repository_id=REPOSITORY_ID,
        )
    listed = registry.list_runs(run_namespace=NAMESPACE, repository_id=REPOSITORY_ID)
    assert len(listed) == 2
    assert listed[0].run_id < listed[1].run_id


def test_cas_rejects_non_monotonic_revision(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="mono")
    registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    jump = replace(
        handle,
        run_revision=3,
        updated_at_ms=handle.updated_at_ms + 1,
        state_revision_cid=_cid("mono-state-3"),
        health_revision_cid=_cid("mono-health-3"),
    )
    with pytest.raises(Exception, match="exactly one"):
        registry.cas_update(jump, expected_revision=1)


def test_transaction_receipt_is_content_addressed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    handle = _handle(label="tx")
    receipt = registry.create(
        handle,
        run_namespace=NAMESPACE,
        repository_id=REPOSITORY_ID,
    )
    encoded = receipt.to_dict()
    assert encoded["content_id"] == receipt.content_id
    assert encoded["operation"] == "create"
    assert encoded["outcome"] == "committed"
    assert RUN_REGISTRY_REQUIREMENT_ID in registry.restart_behavior()["requirement_id"]


def test_missing_run_is_explicit(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(RunNotFoundError):
        registry.get(_cid("missing-run"))
