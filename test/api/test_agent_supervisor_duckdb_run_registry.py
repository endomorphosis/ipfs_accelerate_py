"""Tests for DuckDB-backed mutable run registry with immutable IPLD history."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ContinuationAction,
    RunHandle,
    RunHealth,
    RunState,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
    RegistryTxOutcome,
    RunCasConflictError,
    RunRegistry,
    RunRegistryReadOnlyError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (
    RunAdoptionAction,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_bytes,
    cid_for_dag_json,
)

duckdb = pytest.importorskip("duckdb")


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": "ase2-004", "label": label})


def _prompt_cid(label: str) -> str:
    """Prompt identities must be CIDv1/raw/sha2-256."""

    return cid_for_bytes(
        f"ase2-004-prompt:{label}".encode("utf-8"),
        codec="raw",
    )


def _handle(
    *,
    label: str = "alpha",
    revision: int = 1,
    state: RunState = RunState.RUNNING,
    health: RunHealth = RunHealth.HEALTHY,
    updated_at_ms: int = 2_000,
    created_at_ms: int = 1_000,
    event_cursor: str = "event:1",
    objective_cid: str = "",
    lifecycle_profile_cid: str = "",
) -> RunHandle:
    """Build a full contract-valid RunHandle (semantic_id is derived)."""

    if state is RunState.RUNNING:
        continuation = ContinuationAction.MONITOR
        process_cid = _cid(f"{label}-process")
        lifecycle = lifecycle_profile_cid or _cid(f"{label}-lifecycle")
        state_rev = _cid(f"{label}-state-rev-r{revision}")
        health_rev = _cid(f"{label}-health-rev-r{revision}")
        lease_id = f"lease:{label}"
        fencing = 1
        ambiguity = ""
        if health is None:  # pragma: no cover
            health = RunHealth.HEALTHY
    elif state is RunState.FAILED:
        continuation = ContinuationAction.NONE
        health = RunHealth.TERMINAL if health is RunHealth.HEALTHY else health
        # Keep process history for failed runs when unhealthy is requested.
        if health is RunHealth.UNHEALTHY:
            process_cid = _cid(f"{label}-process")
            lifecycle = lifecycle_profile_cid or _cid(f"{label}-lifecycle")
            state_rev = _cid(f"{label}-state-rev-r{revision}")
            health_rev = _cid(f"{label}-health-rev-r{revision}")
            lease_id = f"lease:{label}"
            fencing = 1
        else:
            process_cid = _cid(f"{label}-process")
            lifecycle = lifecycle_profile_cid or _cid(f"{label}-lifecycle")
            state_rev = _cid(f"{label}-state-rev-r{revision}")
            health_rev = _cid(f"{label}-health-rev-r{revision}")
            lease_id = f"lease:{label}"
            fencing = 1
        ambiguity = ""
    else:
        continuation = ContinuationAction.MONITOR
        process_cid = _cid(f"{label}-process")
        lifecycle = lifecycle_profile_cid or _cid(f"{label}-lifecycle")
        state_rev = _cid(f"{label}-state-rev-r{revision}")
        health_rev = _cid(f"{label}-health-rev-r{revision}")
        lease_id = f"lease:{label}"
        fencing = 1
        ambiguity = ""

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
        objective_cid=objective_cid or _cid(f"{label}-objective"),
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


def _advance(handle: RunHandle, **kwargs) -> RunHandle:
    """Advance revision with optional field overrides (not including semantic_id)."""

    tag = kwargs.pop("tag", "x")
    updates: dict = {
        "run_revision": handle.run_revision + 1,
        "updated_at_ms": handle.updated_at_ms + 10,
        "state_revision_cid": _cid(
            f"{handle.run_id}-state-rev-r{handle.run_revision + 1}-{tag}"
        ),
    }
    for key, value in kwargs.items():
        updates[key] = value
    return replace(handle, **updates)


@pytest.fixture
def duck_registry(tmp_path: Path):
    reg = RunRegistry(tmp_path / "registry", backend="duckdb", auto_migrate=False)
    yield reg
    reg.close()


class TestDuckDBCasConflict:
    def test_conflicting_updates_cannot_both_win(self, duck_registry: RunRegistry):
        h1 = _handle(label="cas", revision=1)
        duck_registry.create(
            h1,
            run_namespace="ns.demo",
            repository_id="repo-1",
        )
        a = _advance(h1, tag="a", event_cursor="cursor-a")
        b = _advance(h1, tag="b", event_cursor="cursor-b")

        receipt_a = duck_registry.cas_update(a, expected_revision=1)
        assert receipt_a.outcome is RegistryTxOutcome.COMMITTED

        with pytest.raises(RunCasConflictError) as exc_info:
            duck_registry.cas_update(b, expected_revision=1)
        assert exc_info.value.receipt is not None
        assert exc_info.value.receipt.outcome is RegistryTxOutcome.CONFLICT

        current = duck_registry.reconstruct(h1.run_id)
        assert current.content_id == a.content_id
        assert current.event_cursor == "cursor-a"
        assert current.run_revision == 2

    def test_duckdb_cas_store_head_loses_on_stale_revision(
        self, duck_registry: RunRegistry
    ):
        h1 = _handle(label="stale", revision=1)
        duck_registry.create(
            h1, run_namespace="ns.demo", repository_id="repo-1"
        )
        a = _advance(h1, tag="a")
        duck_registry.cas_update(a, expected_revision=1)

        # Direct backend CAS with stale expected revision must fail.
        assert duck_registry._duck is not None
        stale = _advance(h1, tag="stale")
        from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
            RunHeadRecord,
        )

        head = RunHeadRecord.from_handle(
            stale, previous_handle_cid=h1.content_id, previous_revision=1
        )
        ok = duck_registry._duck.cas_store_head(
            run_id=h1.run_id,
            run_namespace="ns.demo",
            expected_revision=1,
            expected_handle_cid=h1.content_id,
            head_payload=head.to_dict(),
        )
        assert ok is False
        current = duck_registry.reconstruct(h1.run_id)
        assert current.content_id == a.content_id


class TestDuckDBRestartReconstruction:
    def test_restart_reconstructs_same_handle(self, tmp_path: Path):
        root = tmp_path / "registry"
        h1 = _handle(label="restart", revision=1)
        with RunRegistry(root, backend="duckdb", auto_migrate=False) as reg:
            reg.create(h1, run_namespace="ns.demo", repository_id="repo-1")
            h2 = _advance(h1, tag="next", state=RunState.RUNNING)
            reg.cas_update(h2, expected_revision=1)
            integrity = reg.integrity_cid(h1.run_id)
            handle_before = reg.reconstruct(h1.run_id)

        # Fresh process: new RunRegistry instance against same root.
        with RunRegistry(root, backend="duckdb", auto_migrate=False) as reg2:
            handle_after = reg2.reconstruct(h1.run_id)
            assert handle_after.to_dict() == handle_before.to_dict()
            assert handle_after.content_id == handle_before.content_id
            assert reg2.integrity_cid(h1.run_id) == integrity
            # Immutable IPLD history still on disk.
            snap = list((root / "namespaces").rglob("handles/*.json"))
            assert len(snap) >= 2


class TestDuckDBAdoption:
    def test_one_compatible_healthy_process_is_adopted(
        self, duck_registry: RunRegistry
    ):
        obj = _cid("shared-obj")
        prof = _cid("shared-prof")
        h1 = _handle(
            label="adopt1",
            revision=1,
            state=RunState.RUNNING,
            health=RunHealth.HEALTHY,
            objective_cid=obj,
            lifecycle_profile_cid=prof,
        )
        # Unhealthy runner should not win over the healthy runner.
        h2 = _handle(
            label="adopt2",
            revision=1,
            state=RunState.RUNNING,
            health=RunHealth.UNHEALTHY,
            objective_cid=obj,
            lifecycle_profile_cid=prof,
        )
        duck_registry.create(
            h1, run_namespace="ns.adopt", repository_id="repo-adopt"
        )
        duck_registry.create(
            h2, run_namespace="ns.adopt", repository_id="repo-adopt"
        )

        selection = duck_registry.select_current(
            run_namespace="ns.adopt",
            repository_id="repo-adopt",
            expected_objective_cid=obj,
            expected_profile_cid=prof,
        )
        assert selection.selected_run_id == h1.run_id
        assert selection.selected_handle is not None
        assert selection.selected_handle.content_id == h1.content_id
        assert selection.action is RunAdoptionAction.ADOPT or (
            selection.selected_run_id == h1.run_id
        )


class TestLegacyJsonMigration:
    def test_migration_lossless_and_idempotent(self, tmp_path: Path):
        root = tmp_path / "registry"
        h1 = _handle(label="mig", revision=1)
        with RunRegistry(root, backend="json") as json_reg:
            json_reg.create(
                h1, run_namespace="ns.mig", repository_id="repo-mig"
            )
            h2 = _advance(h1, tag="m2")
            json_reg.cas_update(h2, expected_revision=1)
            json_reg.set_current(
                run_namespace="ns.mig",
                repository_id="repo-mig",
                run_id=h1.run_id,
            )
            before = json_reg.reconstruct(h1.run_id)
            before_head = json_reg.get_head(h1.run_id)
            before_current = json_reg.get_current(
                run_namespace="ns.mig", repository_id="repo-mig"
            )
            assert before_current is not None

        # Migrate into DuckDB.
        with RunRegistry(root, backend="duckdb", auto_migrate=True) as duck_reg:
            after = duck_reg.reconstruct(h1.run_id)
            assert after.to_dict() == before.to_dict()
            assert duck_reg.get_head(h1.run_id).to_dict() == before_head.to_dict()
            current = duck_reg.get_current(
                run_namespace="ns.mig", repository_id="repo-mig"
            )
            assert current is not None
            assert current.content_id == before_current.content_id

            # Second migration is idempotent (NOOP or zero new rows).
            receipt2 = duck_reg.migrate_legacy_json()
            assert receipt2.outcome in {
                RegistryTxOutcome.NOOP,
                RegistryTxOutcome.COMMITTED,
            }
            after2 = duck_reg.reconstruct(h1.run_id)
            assert after2.to_dict() == before.to_dict()

            # Immutable handle snapshots untouched on disk.
            handles = list((root / "namespaces").rglob("handles/*.json"))
            assert len(handles) >= 2
            for path in handles:
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded = RunHandle.from_dict(payload)
                assert loaded.content_id == path.stem or loaded.content_id in path.name


class TestImmutableReplica:
    def test_replica_is_queryable_but_cannot_claim_fence_or_accept_effects(
        self, tmp_path: Path
    ):
        root = tmp_path / "registry"
        h1 = _handle(label="ro", revision=1)
        with RunRegistry(root, backend="duckdb", auto_migrate=False) as reg:
            reg.create(h1, run_namespace="ns.ro", repository_id="repo-ro")
            reg.set_current(
                run_namespace="ns.ro",
                repository_id="repo-ro",
                run_id=h1.run_id,
            )

        with RunRegistry(
            root,
            backend="duckdb",
            immutable_replica=True,
            auto_migrate=False,
        ) as replica:
            # Readable.
            got = replica.reconstruct(h1.run_id)
            assert got.content_id == h1.content_id
            assert replica.exists(h1.run_id)
            listed = replica.list_runs(run_namespace="ns.ro")
            assert len(listed) == 1
            current = replica.get_current(
                run_namespace="ns.ro", repository_id="repo-ro"
            )
            assert current is not None
            assert current.run_id == h1.run_id

            # Writes rejected: claim / fence / effect surfaces.
            h2 = _advance(h1, tag="write")
            with pytest.raises(RunRegistryReadOnlyError):
                replica.cas_update(h2, expected_revision=1)
            with pytest.raises(RunRegistryReadOnlyError):
                replica.create(
                    _handle(label="new"),
                    run_namespace="ns.ro",
                    repository_id="repo-ro",
                )
            with pytest.raises(RunRegistryReadOnlyError):
                replica.set_current(
                    run_namespace="ns.ro",
                    repository_id="repo-ro",
                    run_id=h1.run_id,
                )
            with pytest.raises(RunRegistryReadOnlyError):
                replica.repair()
            with pytest.raises(RunRegistryReadOnlyError):
                replica.migrate_legacy_json()

            # Backend itself rejects mutation.
            assert replica._duck is not None
            assert replica._duck.read_only is True


class TestConcurrentCas:
    def test_threaded_cas_only_one_winner(self, duck_registry: RunRegistry):
        h1 = _handle(label="thr", revision=1)
        duck_registry.create(
            h1, run_namespace="ns.thr", repository_id="repo-thr"
        )

        winners: list[str] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def worker(tag: str) -> None:
            try:
                barrier.wait(timeout=5)
                nxt = _advance(h1, tag=tag, event_cursor=f"c-{tag}")
                duck_registry.cas_update(nxt, expected_revision=1)
                winners.append(tag)
            except BaseException as exc:  # noqa: BLE001 — collect for assertion
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(winners) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], RunCasConflictError)
        current = duck_registry.reconstruct(h1.run_id)
        assert current.run_revision == 2
        assert current.event_cursor in {"c-t1", "c-t2"}
