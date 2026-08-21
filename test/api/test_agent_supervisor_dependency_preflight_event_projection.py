"""Bounded, proof-carrying event persistence for dependency preflight."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (
    BoundedArtifactStore,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    MAX_RECEIPT_BYTES,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    ValidationProjectDependencyPreflightDeferred,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bind_database_portal_execution_from_args,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    project_dependency_preflight as preflight_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PREFLIGHT_EVENT_PROJECTION_SCHEMA,
    canonical_project_dependency_preflight_receipt_bytes,
    project_dependency_preflight_error_receipt,
)


def _daemon(tmp_path: Path) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        dependency_preflight_artifact_store_path=tmp_path / "preflight-artifacts",
    )


def _reseal(receipt: dict[str, object]) -> dict[str, object]:
    receipt.pop("receipt_id", None)
    receipt.pop("retry_fingerprint", None)
    receipt["receipt_id"] = preflight_module._content_sha256(receipt)
    receipt["retry_fingerprint"] = preflight_module._retry_fingerprint(receipt)
    return receipt


def _oversized_receipt(tmp_path: Path) -> dict[str, object]:
    receipt = project_dependency_preflight_error_receipt(
        tmp_path,
        ["python -m pytest -q"],
        RuntimeError("fixture dependency failure"),
    )
    receipt["probe"] = {
        "projects": [
            {
                "marker_skipped": [
                    {
                        "name": f"distribution-{index}",
                        "marker": "x" * 384,
                        "requirement": f"distribution-{index}>=1",
                    }
                    for index in range(3_200)
                ],
                "observed": [
                    {
                        "name": f"observed-{index}",
                        "installed_version": "1.0.0",
                    }
                    for index in range(512)
                ],
            }
        ]
    }
    return _reseal(receipt)


def _projection_at(event: dict[str, object], *path: str) -> dict[str, object]:
    value: object = event
    for field in path:
        assert isinstance(value, dict)
        value = value[field]
    assert isinstance(value, dict)
    return value


def test_all_nested_preflight_receipts_use_one_bounded_verified_artifact(
    tmp_path,
) -> None:
    daemon = _daemon(tmp_path)
    receipt = _oversized_receipt(tmp_path)
    canonical = canonical_project_dependency_preflight_receipt_bytes(receipt)
    assert len(canonical) > 1_000_000

    try:
        daemon._record_event(
            "implementation_finished",
            {
                "task_id": "LGCVF-080",
                "dependency_preflight": receipt,
                "exception_result": {"dependency_preflight": receipt},
                "workspace_setup": {
                    "validation_project_dependency_preflight": receipt,
                },
            },
        )

        encoded_event = daemon.events_path.read_bytes().splitlines()[-1]
        assert len(encoded_event) < MAX_RECEIPT_BYTES
        event = json.loads(encoded_event)
        projections = (
            _projection_at(event, "dependency_preflight"),
            _projection_at(
                event,
                "exception_result",
                "dependency_preflight",
            ),
            _projection_at(
                event,
                "workspace_setup",
                "validation_project_dependency_preflight",
            ),
        )
        assert {
            projection["schema"] for projection in projections
        } == {PROJECT_DEPENDENCY_PREFLIGHT_EVENT_PROJECTION_SCHEMA}
        references = {
            json.dumps(
                projection["full_receipt_artifact"],
                sort_keys=True,
            )
            for projection in projections
        }
        assert len(references) == 1
        reference = projections[0]["full_receipt_artifact"]
        assert isinstance(reference, dict)
        assert daemon._dependency_preflight_artifact_store is not None
        assert (
            daemon._dependency_preflight_artifact_store.read_blob(reference)
            == canonical
        )
        assert daemon._dependency_preflight_artifact_store.usage()["blob_count"] == 1
    finally:
        daemon.close_event_runtime()


def test_event_boundary_rejects_forged_preflight_without_append(
    tmp_path,
) -> None:
    daemon = _daemon(tmp_path)
    receipt = _oversized_receipt(tmp_path)
    receipt["receipt_id"] = "0" * 64

    try:
        with pytest.raises(ValueError, match="receipt identity mismatch"):
            daemon._record_event(
                "validation_project_dependency_preflight_failed",
                {"dependency_preflight": receipt},
            )
        assert not daemon.events_path.exists()
    finally:
        daemon.close_event_runtime()


def test_event_boundary_rejects_caller_supplied_compact_projection(
    tmp_path,
) -> None:
    daemon = _daemon(tmp_path)
    forged_projection = {
        "schema": PROJECT_DEPENDENCY_PREFLIGHT_EVENT_PROJECTION_SCHEMA,
        "receipt_id": "forged",
        "passed": True,
        "completion_authority": True,
        "full_receipt_artifact": {
            "schema": "ipfs_accelerate_py.agent_supervisor.bounded-blob-reference@1",
            "artifact_id": "blob:sha256:" + "0" * 64,
            "digest": "sha256:" + "0" * 64,
            "size_bytes": 1,
            "kind": "validation_project_dependency_preflight_receipt",
            "media_type": "application/json",
        },
    }

    try:
        with pytest.raises(
            RuntimeError,
            match="caller-supplied dependency preflight event projection",
        ):
            daemon._record_event(
                "implementation_finished",
                {"nested": {"dependency_preflight": forged_projection}},
            )
        assert not daemon.events_path.exists()
    finally:
        daemon.close_event_runtime()


def test_preflight_artifact_failure_blocks_dispatch_with_inline_error_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    candidate = project_dependency_preflight_error_receipt(
        tmp_path,
        ["python -m pytest -q"],
        RuntimeError("fixture dependency failure"),
    )
    candidate["passed"] = True
    candidate["reason"] = "fixture_dependencies_satisfied"
    _reseal(candidate)
    task = PortalTask(
        task_id="LGCVF-080",
        title="Bound dependency preflight event",
        status="todo",
        completion="manual",
        priority="P0",
        track="runtime",
        validation=["python -m pytest -q"],
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "preflight_validation_project_dependencies",
        lambda *_args, **_kwargs: dict(candidate),
    )

    def reject_artifact_write(*_args, **_kwargs):
        raise OSError("fixture CAS unavailable")

    monkeypatch.setattr(BoundedArtifactStore, "put_blob", reject_artifact_write)

    try:
        with pytest.raises(
            ValidationProjectDependencyPreflightDeferred
        ) as raised:
            daemon._require_validation_project_dependency_preflight(
                workspace_path=tmp_path,
                task=task,
                attempt=1,
            )

        assert raised.value.receipt["passed"] is False
        assert raised.value.receipt["reason"] == (
            "project_dependency_preflight_infrastructure_error"
        )
        event = json.loads(daemon.events_path.read_text(encoding="utf-8"))
        projection = event["dependency_preflight"]
        assert projection["schema"] == (
            PROJECT_DEPENDENCY_PREFLIGHT_EVENT_PROJECTION_SCHEMA
        )
        assert projection["inline_receipt"] == raised.value.receipt
        assert "full_receipt_artifact" not in projection
        assert projection["completion_authority"] is False
    finally:
        daemon.close_event_runtime()


def test_database_portal_attempts_share_dependency_preflight_artifact_store(
    tmp_path,
) -> None:
    captured_callbacks: dict[str, object] = {}

    class BindingDaemon:
        task_source = object()

        @staticmethod
        def bind_execution_callbacks(**callbacks: object) -> None:
            captured_callbacks.update(callbacks)

    class CapturingPortal:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "canonical-board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "lgcvf",
            "--implement",
            "--once",
        ]
    )
    bridge = bind_database_portal_execution_from_args(
        BindingDaemon(),
        args,
        repo_root=tmp_path,
        portal_daemon_class=CapturingPortal,
    )
    assert bridge is not None
    portal = bridge.portal_factory(
        SimpleNamespace(
            task_projection=tmp_path / "attempt" / "task.md",
            state=tmp_path / "attempt" / "state.json",
            strategy=tmp_path / "attempt" / "strategy.json",
            events=tmp_path / "attempt" / "events.jsonl",
            implementation_logs=tmp_path / "attempt" / "logs",
        ),
        "LGCVF-080",
    )

    assert isinstance(portal, CapturingPortal)
    assert portal.kwargs["dependency_preflight_artifact_store_path"] == (
        tmp_path
        / "state"
        / "lgcvf_database_portal_attempts"
        / "dependency-preflight-artifacts"
    )
    assert set(captured_callbacks) == {
        "provider_fn",
        "effect_fn",
        "validation_fn",
    }
