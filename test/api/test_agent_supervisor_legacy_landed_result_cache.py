"""Signed, fenced legacy landed leaf-cache regressions."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.verified_ipld_backend import (
    InMemoryConformantBackend,
    VerifiedIPLDBackend,
    VerifiedIPLDError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_result_cache as cache_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_review as legacy,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_attestation import (
    legacy_landed_review_key_id,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_result_cache import (
    LegacyLandedLeafCacheError,
    LegacyLandedLeafCacheKey,
    LegacyLandedLeafResultCache,
    verify_legacy_landed_leaf_cache_record,
)

HEAD = "a" * 40
TREE = "b" * 40


def _private_key_file(path: Path) -> str:
    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    path.write_bytes(raw)
    path.chmod(0o600)
    return legacy_landed_review_key_id(private.public_key().public_bytes_raw())


def _policy_payload(
    issuer_key_id: str,
    *,
    current_head: str = HEAD,
    current_tree_id: str = TREE,
) -> dict[str, Any]:
    template = copy.deepcopy(legacy.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE)
    body = {
        "schema": legacy.LEGACY_LANDED_REVIEW_POLICY_SCHEMA,
        "interface": legacy.LEGACY_LANDED_REVIEW_POLICY_INTERFACE,
        "enabled": True,
        "issuer_key_id": issuer_key_id,
        "current_head": current_head,
        "current_tree_id": current_tree_id,
        "max_leaf_tokens": template["max_leaf_tokens"],
        "providers": template["providers"],
        "tasks": template["tasks"],
        "historical_provider": "unverified",
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    return {**body, "policy_id": content_identity(body)}


def _binding(task: legacy.LegacyTaskPolicy) -> legacy.LegacyRepositoryBinding:
    blobs = tuple(
        (
            path,
            legacy._GitBlob(  # noqa: SLF001 - exact immutable test fixture
                True,
                "100644",
                f"{index + 1:040x}",
                f"def reviewed_{index}():\n    return {task.task_id!r}\n".encode(),
            ),
        )
        for index, path in enumerate(task.paths)
    )
    return legacy.LegacyRepositoryBinding(
        task=task,
        current_head=HEAD,
        current_tree_id=TREE,
        historical_diff=(
            b"diff --git a/source.py b/source.py\n"
            b"--- a/source.py\n+++ b/source.py\n"
            b"@@ -1 +1 @@\n-old\n+new\n"
        ),
        current_blobs=blobs,
    )


def _cache_fixture(
    tmp_path: Path,
) -> tuple[
    LegacyLandedLeafResultCache,
    legacy.LegacyLandedReviewPolicy,
    legacy.LegacyTaskPolicy,
    dict[str, Any],
    Path,
]:
    key_path = tmp_path / "legacy-review.key"
    issuer = _private_key_file(key_path)
    policy = legacy.parse_legacy_landed_review_policy(_policy_payload(issuer))
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    cache = LegacyLandedLeafResultCache(
        tmp_path / "cache.duckdb",
        policy=policy,
        operator_key_path=key_path,
    )
    return cache, policy, task, manifest, key_path


class _ApprovingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, request: legacy.LegacyLeafReviewRequest
    ) -> legacy.LegacyProviderObservation:
        self.calls += 1
        leaf = request.payload["leaf"]
        return legacy.LegacyProviderObservation(
            observation_id=(
                f"{request.role}:observation:{os.getpid()}:{self.calls}"
            ),
            requested_provider=request.provider,
            requested_model=request.model,
            effective_provider=request.provider,
            effective_model=request.model,
            provider_chain=(request.provider,),
            fallback_used=False,
            supervisor_observed=True,
            response={
                "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
                "decision": "approve",
                "manifest_id": request.payload["manifest_id"],
                "leaf_id": leaf["leaf_id"],
                "findings": [],
            },
        )


class _SharedApprovingProvider:
    def __init__(self, calls: Any) -> None:
        self.calls = calls

    def __call__(
        self, request: legacy.LegacyLeafReviewRequest
    ) -> legacy.LegacyProviderObservation:
        with self.calls.get_lock():
            self.calls.value += 1
            call = int(self.calls.value)
        leaf = request.payload["leaf"]
        return legacy.LegacyProviderObservation(
            observation_id=f"shared:{os.getpid()}:{call}",
            requested_provider=request.provider,
            requested_model=request.model,
            effective_provider=request.provider,
            effective_model=request.model,
            provider_chain=(request.provider,),
            fallback_used=False,
            supervisor_observed=True,
            response={
                "schema": legacy.LEGACY_LANDED_LEAF_DECISION_SCHEMA,
                "decision": "approve",
                "manifest_id": request.payload["manifest_id"],
                "leaf_id": leaf["leaf_id"],
                "findings": [],
            },
        )


def _review_process(
    database: str,
    key_path: str,
    policy_payload: dict[str, Any],
    manifest: dict[str, Any],
    calls: Any,
    queue: Any,
    index: int,
) -> None:
    try:
        policy = legacy.parse_legacy_landed_review_policy(policy_payload)
        task = policy.task("ASE-005")
        cache = LegacyLandedLeafResultCache(
            database,
            policy=policy,
            operator_key_path=key_path,
        )
        result = cache.review_leaf(
            task=task,
            manifest=manifest,
            leaf=manifest["leaves"][0],
            provider=policy.grok,
            invoker=_SharedApprovingProvider(calls),
            review_run_id=f"legacy-review:{index:048x}",
            wait_timeout_seconds=30,
        )
        queue.put(("ok", result.cache_hit, result.receipt["review_run_id"]))
    except BaseException as exc:  # pragma: no cover - reported to parent
        queue.put(("error", type(exc).__name__, str(exc)))


def _key_for_first_leaf(
    cache: LegacyLandedLeafResultCache,
    policy: legacy.LegacyLandedReviewPolicy,
    task: legacy.LegacyTaskPolicy,
    manifest: dict[str, Any],
    *,
    provider: legacy.LegacyProviderPolicy | None = None,
) -> LegacyLandedLeafCacheKey:
    selected = provider or policy.grok
    request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=selected,
    )
    return cache.key_for(
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=selected,
        request=request,
    )


def test_cold_then_restart_warm_rebinds_signed_evidence_without_provider(
    tmp_path: Path,
) -> None:
    cache, policy, task, manifest, key_path = _cache_fixture(tmp_path)
    provider = _ApprovingProvider()
    cold = cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=provider,
        review_run_id="legacy-review:" + "1" * 48,
    )
    restarted = LegacyLandedLeafResultCache(
        cache.path,
        policy=policy,
        operator_key_path=key_path,
    )

    def forbidden(_request: legacy.LegacyLeafReviewRequest) -> Any:
        raise AssertionError("warm cache invoked the provider")

    warm = restarted.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=forbidden,
        review_run_id="legacy-review:" + "2" * 48,
    )

    assert cold.cache_hit is False
    assert warm.cache_hit is True
    assert provider.calls == 1
    assert warm.receipt["review_run_id"] != cold.receipt["review_run_id"]
    assert warm.receipt["provider_evidence_source"] == "signed_cache"
    assert warm.receipt["provider_invoked_in_current_run"] is False
    record = warm.receipt["provider_evidence_cache_record"]
    verification = verify_legacy_landed_leaf_cache_record(
        record,
        expected_key=_key_for_first_leaf(cache, policy, task, manifest),
        trusted_public_keys=cache.trusted_public_keys,
    )
    assert verification.verified is True
    assert record["validation_cached"] is False
    assert record["completion_authoritative"] is False
    assert record["proof_authoritative"] is False


def test_cache_key_binds_every_review_dimension_and_is_closed(
    tmp_path: Path,
) -> None:
    cache, policy, task, manifest, _key_path = _cache_fixture(tmp_path)
    key = _key_for_first_leaf(cache, policy, task, manifest)
    mutations: dict[str, Any] = {
        "policy_id": key.policy_id + "x",
        "task_id": key.task_id + "x",
        "canonical_task_key": key.canonical_task_key + "x",
        "canonical_task_cid": key.canonical_task_cid + "x",
        "manifest_id": key.manifest_id + "x",
        "manifest_merkle_root": key.manifest_merkle_root + "x",
        "leaf_index": key.leaf_index + 1,
        "leaf_id": key.leaf_id + "x",
        "request_id": key.request_id + "x",
        "request_cid": key.request_cid + "x",
        "role": key.role + "x",
        "provider": key.provider + "x",
        "model": key.model + "x",
        "current_head": "c" * 40,
        "current_tree_id": "d" * 40,
    }
    identities = {
        replace(key, **{field: value}).key_id
        for field, value in mutations.items()
    }
    assert key.key_id not in identities
    assert len(identities) == len(mutations)

    extra = key.to_dict()
    extra["caller_override"] = True
    with pytest.raises(ValueError, match="shape"):
        LegacyLandedLeafCacheKey.from_dict(extra)
    missing = key.to_dict()
    missing.pop("request_cid")
    with pytest.raises(ValueError, match="shape"):
        LegacyLandedLeafCacheKey.from_dict(missing)

    for foreign in (
        replace(key, policy_id=key.policy_id + "x"),
        replace(key, current_head="c" * 40),
        replace(key, current_tree_id="d" * 40),
        replace(key, task_id="ASE-999"),
        replace(key, canonical_task_cid=key.canonical_task_cid + "x"),
        replace(key, provider="another-provider"),
    ):
        with pytest.raises(LegacyLandedLeafCacheError, match="policy|binding"):
            cache.lookup(foreign)
        with pytest.raises(LegacyLandedLeafCacheError, match="policy|binding"):
            cache.acquire(foreign)


def test_stale_fencing_token_cannot_publish_after_takeover(tmp_path: Path) -> None:
    now = [1.0]
    key_path = tmp_path / "legacy-review.key"
    issuer = _private_key_file(key_path)
    policy = legacy.parse_legacy_landed_review_policy(_policy_payload(issuer))
    task = policy.task("ASE-005")
    manifest = legacy.build_legacy_landed_byte_manifest(policy, _binding(task))
    cache = LegacyLandedLeafResultCache(
        tmp_path / "cache.duckdb",
        policy=policy,
        operator_key_path=key_path,
        clock=lambda: now[0],
    )
    key = _key_for_first_leaf(cache, policy, task, manifest)
    first = cache.acquire(key, owner_id="first", lease_seconds=1)
    now[0] = 3.0
    second = cache.acquire(key, owner_id="second", lease_seconds=1)
    request = legacy._leaf_review_request(  # noqa: SLF001
        policy=policy,
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
    )
    receipt = legacy._review_one_leaf(  # noqa: SLF001
        request=request,
        provider=policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "3" * 48,
    )

    with pytest.raises(LegacyLandedLeafCacheError, match="fenced"):
        cache.put(key, receipt, lease=first)
    assert cache.put(key, receipt, lease=second).key == key


def test_signed_row_tampering_is_poison_not_a_cache_miss(tmp_path: Path) -> None:
    cache, policy, task, manifest, _key_path = _cache_fixture(tmp_path)
    cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "4" * 48,
    )
    key = _key_for_first_leaf(cache, policy, task, manifest)
    connection = open_duckdb_connection(cache.path)
    try:
        row = connection.execute(
            "SELECT record_json FROM legacy_landed_leaf_records WHERE key_id=?",
            (key.key_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row["record_json"]))
        payload["signature"] = "AAAA"
        connection.execute(
            "UPDATE legacy_landed_leaf_records SET record_json=? WHERE key_id=?",
            (
                canonical_json_bytes(payload).decode("ascii"),
                key.key_id,
            ),
        )
    finally:
        connection.close()

    with pytest.raises(LegacyLandedLeafCacheError, match="poisoned"):
        cache.lookup(key)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="four-process cache lease regression requires fork",
)
def test_four_processes_same_key_perform_exactly_one_provider_call(
    tmp_path: Path,
) -> None:
    cache, policy, task, manifest, key_path = _cache_fixture(tmp_path)
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    queue = context.Queue()
    payload = _policy_payload(policy.issuer_key_id)
    processes = [
        context.Process(
            target=_review_process,
            args=(
                str(cache.path),
                str(key_path),
                payload,
                manifest,
                calls,
                queue,
                index + 1,
            ),
        )
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _process in processes]

    assert all(item[0] == "ok" for item in results), results
    assert calls.value == 1
    assert sum(item[1] is False for item in results) == 1
    assert len({item[2] for item in results}) == 4
    assert len(cache.records()) == 1


class _InsertDuringRawPutBackend(InMemoryConformantBackend):
    def __init__(self, hook: Any) -> None:
        super().__init__()
        self._hook = hook
        self._triggered = False

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        cid = super().block_put(data, codec=codec)
        if codec == "raw" and not self._triggered:
            self._triggered = True
            self._hook()
        return cid


def test_snapshot_inventory_is_exact_during_concurrent_insert_and_tamper_fails(
    tmp_path: Path,
) -> None:
    cache, policy, task, manifest, key_path = _cache_fixture(tmp_path)
    first_provider = _ApprovingProvider()
    cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=first_provider,
        review_run_id="legacy-review:" + "5" * 48,
    )

    def concurrent_insert() -> None:
        cache.review_leaf(
            task=task,
            manifest=manifest,
            leaf=manifest["leaves"][0],
            provider=policy.codex,
            invoker=_ApprovingProvider(),
            review_run_id="legacy-review:" + "6" * 48,
        )

    storage = _InsertDuringRawPutBackend(concurrent_insert)
    backend = VerifiedIPLDBackend(backend=storage)
    snapshot = cache.export_snapshot(tmp_path / "snapshots", backend=backend)
    assert snapshot.row_count == 1
    assert len(cache.records()) == 2
    assert snapshot.parquet_cid in storage._pins  # noqa: SLF001
    assert snapshot.manifest_cid in storage._pins  # noqa: SLF001

    imported = LegacyLandedLeafResultCache(
        tmp_path / "imported.duckdb",
        policy=policy,
        operator_key_path=key_path,
    )
    assert imported.import_snapshot(snapshot.manifest_cid, backend=backend) == 1
    assert len(imported.records()) == 1
    assert imported.import_snapshot(snapshot.manifest_cid, backend=backend) == 0

    version_one = dict(snapshot.manifest)
    version_one["schema"] = (
        cache_module.LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA_V1
    )
    version_one.pop("replication_pin_requested")
    version_one_cid = backend.put_dag_json(version_one).cid
    version_one_cache = LegacyLandedLeafResultCache(
        tmp_path / "version-one.duckdb",
        policy=policy,
        operator_key_path=key_path,
    )
    assert version_one_cache.import_snapshot(
        version_one_cid, backend=backend
    ) == 1

    storage._blocks[snapshot.parquet_cid] = b"tampered"  # noqa: SLF001
    empty = LegacyLandedLeafResultCache(
        tmp_path / "empty.duckdb",
        policy=policy,
        operator_key_path=key_path,
    )
    with pytest.raises(VerifiedIPLDError):
        empty.import_snapshot(snapshot.manifest_cid, backend=backend)
    assert empty.records() == ()


def test_import_rejects_foreign_policy_rows_and_bounded_duplicate_inventory(
    tmp_path: Path,
) -> None:
    cache, old_policy, task, manifest, key_path = _cache_fixture(tmp_path)
    cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=old_policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "9" * 48,
    )
    storage = InMemoryConformantBackend()
    backend = VerifiedIPLDBackend(backend=storage)
    snapshot = cache.export_snapshot(tmp_path / "foreign-snapshot", backend=backend)

    current_policy = legacy.parse_legacy_landed_review_policy(
        _policy_payload(
            old_policy.issuer_key_id,
            current_head="c" * 40,
            current_tree_id="d" * 40,
        )
    )
    current = LegacyLandedLeafResultCache(
        tmp_path / "current.duckdb",
        policy=current_policy,
        operator_key_path=key_path,
    )
    relabelled = {
        **snapshot.manifest,
        "policy_id": current_policy.policy_id,
        "current_head": current_policy.current_head,
        "current_tree_id": current_policy.current_tree_id,
    }
    relabelled_cid = backend.put_dag_json(relabelled).cid
    with pytest.raises(LegacyLandedLeafCacheError, match="pinned policy"):
        current.import_snapshot(relabelled_cid, backend=backend)
    assert current.records() == ()

    duplicate = {
        **snapshot.manifest,
        "row_count": 2,
        "ordered_key_ids": [
            snapshot.manifest["ordered_key_ids"][0],
            snapshot.manifest["ordered_key_ids"][0],
        ],
        "ordered_record_ids": [
            snapshot.manifest["ordered_record_ids"][0],
            snapshot.manifest["ordered_record_ids"][0],
        ],
    }
    duplicate_cid = backend.put_dag_json(duplicate).cid
    with pytest.raises(LegacyLandedLeafCacheError, match="inventory bounds"):
        cache.import_snapshot(duplicate_cid, backend=backend)

    oversized = {
        **snapshot.manifest,
        "parquet_byte_length": cache_module.MAX_SNAPSHOT_PARQUET_BYTES + 1,
    }
    oversized_cid = backend.put_dag_json(oversized).cid
    with pytest.raises(LegacyLandedLeafCacheError, match="inventory bounds"):
        cache.import_snapshot(oversized_cid, backend=backend)


def test_snapshot_bound_fails_explicitly_and_existing_cid_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, policy, task, manifest, _key_path = _cache_fixture(tmp_path)
    cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "7" * 48,
    )
    backend = VerifiedIPLDBackend(backend=InMemoryConformantBackend())
    snapshot = cache.export_snapshot(tmp_path / "snapshots", backend=backend)
    snapshot.parquet_path.write_bytes(b"not-the-content-addressed-parquet")

    with pytest.raises(LegacyLandedLeafCacheError, match="differs from its CID"):
        cache.export_snapshot(tmp_path / "snapshots", backend=backend)
    assert snapshot.parquet_path.read_bytes() == b"not-the-content-addressed-parquet"

    monkeypatch.setattr(cache_module, "MAX_SNAPSHOT_RECORDS", 0)
    with pytest.raises(LegacyLandedLeafCacheError, match="bound exceeded"):
        cache.export_snapshot(tmp_path / "bounded", backend=backend)


def test_export_enforces_parquet_byte_bound_before_backend_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, policy, task, manifest, _key_path = _cache_fixture(tmp_path)
    cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "a" * 48,
    )
    storage = InMemoryConformantBackend()
    backend = VerifiedIPLDBackend(backend=storage)
    monkeypatch.setattr(cache_module, "MAX_SNAPSHOT_PARQUET_BYTES", 1)

    with pytest.raises(LegacyLandedLeafCacheError, match="byte bound"):
        cache.export_snapshot(tmp_path / "too-large", backend=backend)
    assert storage._blocks == {}  # noqa: SLF001


def test_cache_never_replays_validation_receipts(tmp_path: Path) -> None:
    cache, policy, task, manifest, _key_path = _cache_fixture(tmp_path)
    result = cache.review_leaf(
        task=task,
        manifest=manifest,
        leaf=manifest["leaves"][0],
        provider=policy.grok,
        invoker=_ApprovingProvider(),
        review_run_id="legacy-review:" + "8" * 48,
    )
    record = result.receipt
    assert "validation_receipts" not in record
    assert "attestation" not in record
    assert "review_aggregate" not in record
    assert record["completion_authoritative"] is False
    assert record["proof_authoritative"] is False

    validation = subprocess.CompletedProcess(["true"], 0, b"", b"")
    assert validation.returncode == 0
