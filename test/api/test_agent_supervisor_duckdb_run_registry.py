"""Tests for DuckDB-backed mutable run registry with immutable IPLD history."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    RunHandle,
    RunHealth,
    RunState,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
    RunCasConflictError,
    RunRegistry,
    RunRegistryReadOnlyError,
    RegistryTxOutcome,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (
    RunAdoptionAction,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_dag_json

duckdb = pytest.importorskip("duckdb")


def _cid(label: str) -> str:
    return cid_for_dag_json({"label": label, "v": 1})


def _handle(
    *,
    run_id: str | None = None,
    revision: int = 1,
    state: RunState = RunState.RUNNING,
    health: RunHealth = RunHealth.HEALTHY,
    updated_at_ms: int = 1_000,
    created_at_ms: int = 1_000,
    semantic_suffix: str = "a",
    event_cursor: str = "",
    objective_cid: str = "",
    lifecycle_profile_cid: str = "",
) -> RunHandle:
    rid = run_id or _cid(f"run-{revision}-{semantic_suffix}")
    return RunHandle(
        run_id=rid,
        run_revision=revision,
        semantic_id=_cid(f"sem-{rid}-{revision}-{semantic_suffix}"),
        state=state,
        health=health,
        target_resolution_receipt_cid=_cid(f"receipt-{rid}"),
        invocation_cid=_cid(f"inv-{rid}"),
        prompt_cid=_cid(f"prompt-{rid}"),
        objective_cid=objective_cid or _cid(f"obj-{rid}"),
        lifecycle_profile_cid=lifecycle_profile_cid or _cid(f"prof-{rid}"),
        event_cursor=event_cursor,
        artifact_cursor="",
        continuation_cursor="",
        created_at_ms=created_at_ms,
        updated_at_ms=updated_at_ms,
        state_revision_cid=_cid(f"state-rev-{rid}-{revision}"),
    )


def _advance(handle: RunHandle, **kwargs) -> RunHandle:
    payload = handle.to_dict()
    payload["run_revision"] = handle.run_revision + 1
    payload["semantic_id"] = _cid(
        f"sem-{handle.run_id}-{handle.run_revision + 1}-{kwargs.get('tag', 'x')}"
    )
    payload["updated_at_ms"] = handle.updated_at_ms + 10
    payload["state_revision_cid"] = _cid(
        f"state-rev-{handle.run_id}-{handle.run_revision + 1}"
    )
    for key, value in kwargs.items():
        if key == "tag":
            continue
        if key == "state" and not isinstance(value, str):
            payload[key] = value.value
        elif key == "health" and not isinstance(value, str):
            payload[key] = value.value
        else:
            payload[key] = value
    # Drop content_id so from_dict recomputes.
    payload.pop("content_id", None)
    return RunHandle.from_dict(payload)


@pytest.fixture
def duck_registry(tmp_path: Path):
    reg = RunRegistry(tmp_path / "registry", backend="duckdb", auto_migrate=False)
    yield reg
    reg.close()


class TestDuckDBCasConflict:
    def test_conflicting_updates_cannot_both_win(self, duck_registry: RunRegistry):
        h1 = _handle(revision=1)
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
        h1 = _handle(revision=1)
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
        h1 = _handle(revision=1)
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
            revision=1,
            state=RunState.RUNNING,
            health=RunHealth.HEALTHY,
            objective_cid=obj,
            lifecycle_profile_cid=prof,
            semantic_suffix="adopt1",
        )
        # Terminal / unhealthy should not win over the healthy runner.
        h2 = _handle(
            revision=1,
            state=RunState.FAILED,
            health=RunHealth.UNHEALTHY,
            objective_cid=obj,
            lifecycle_profile_cid=prof,
            semantic_suffix="adopt2",
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
        assert selection.action in {
            RunAdoptionAction.ADOPT,
            RunAdoptionAction.ATTACH,
            RunAdoptionAction.RESUME,
        } or selection.selected_run_id == h1.run_id
        assert selection.selected_run_id == h1.run_id
        assert selection.selected_handle is not None
        assert selection.selected_handle.content_id == h1.content_id


class TestLegacyJsonMigration:
    def test_migration_lossless_and_idempotent(self, tmp_path: Path):
        root = tmp_path / "registry"
        h1 = _handle(revision=1, semantic_suffix="mig")
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
        h1 = _handle(revision=1, semantic_suffix="ro")
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
                    _handle(semantic_suffix="new"),
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
        h1 = _handle(revision=1, semantic_suffix=" thr")
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
