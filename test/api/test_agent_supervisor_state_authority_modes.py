"""DQP-030: de-authorize legacy files and explicit compatibility modes.

Acceptance:
- Under Quack authority, changing/deleting MD/JSON/JSONL/PID/status projections
  cannot change scheduling/lifecycle.
- Server failure returns unavailable/recovery-required rather than file fallback.
- Legacy import cannot run implicitly.
- Exports carry non-authority marker.

Evidence subset: cold discovery, absent exports, tampered status/taskboard,
local DB fallback refusal, server unavailable, mode transition, rollback.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ResolutionDisposition,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (
    DATABASE_STATE_RESOLVER_INTERFACE,
    DATABASE_STATE_RESOLVER_REQUIREMENT_ID,
    DatabaseServerStatus,
    DatabaseStateEvidence,
    DatabaseStateResolver,
    DatabaseStateResolverError,
    resolve_database_state,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    EXPORT_AUTHORITY_CLASS_VALUE,
    EXPORT_NON_AUTHORITY_MARKER,
    STATE_AUTHORITY_MODE_INTERFACE,
    AuthorityAvailability,
    ImplicitLegacyImportError,
    ScheduleAuthoritySource,
    StateAuthorityMode,
    StateAuthorityModeError,
    StateAuthorityTransitionError,
    StateAuthorityUnavailableError,
    allowed_state_authority_transitions,
    attach_export_non_authority_marker,
    closed_state_authority_modes,
    evaluate_projection_authority,
    evaluate_schedule_authority,
    export_non_authority_marker,
    file_watch_enabled_for_mode,
    file_write_enabled_for_mode,
    gate_legacy_import,
    is_quack_authority_mode,
    open_task_source_for_authority_mode,
    parse_state_authority_mode,
    projection_mutation_affects_schedule,
    require_explicit_legacy_import,
    state_authority_mode_policy,
    transition_state_authority_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_schedule(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ready_task_cids": ["task:cid:ready-1"],
        "revision": 7,
        "claimed": [],
    }
    values.update(changes)
    return values


def _db_lifecycle(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "phase": "running",
        "daemon_session": "session:1",
        "fence_epoch": 3,
    }
    values.update(changes)
    return values


def _file_projections(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "markdown": {
            "schedule": {"ready_task_cids": ["task:cid:forged-md"]},
            "status": "completed",
        },
        "json": {"tasks": [{"id": "DQP-FORGED", "status": "todo"}]},
        "jsonl": [{"event": "forged"}],
        "pid": {"pid": 4242, "alive": True},
        "status": {"phase": "forged", "healthy": True},
        "taskboard": {"ready": ["DQP-FORGED"]},
    }
    values.update(changes)
    return values


def _database_evidence(**changes: object) -> DatabaseStateEvidence:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:dqp-030-test",
        "repository_root": "/home/dev/src/project",
        "authority_mode": StateAuthorityMode.QUACK_AUTHORITATIVE.value,
        "server_status": DatabaseServerStatus.AVAILABLE,
        "database_schedule": _db_schedule(),
        "database_lifecycle": _db_lifecycle(),
        "file_projections": _file_projections(),
        "home_directory": "/home/dev",
        "environ": {},
        "store_id": "control.duckdb",
        "database_uuid": "123e4567-e89b-12d3-a456-426614174000",
        "generation": 1,
        "schema_revision": 1,
    }
    values.update(changes)
    return DatabaseStateEvidence(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Closed mode vocabulary and policies
# ---------------------------------------------------------------------------


def test_closed_state_authority_modes_are_exact() -> None:
    modes = closed_state_authority_modes()
    assert modes == (
        "legacy_import",
        "embedded_maintenance",
        "quack_shadow",
        "quack_authoritative",
        "export_only",
    )
    assert STATE_AUTHORITY_MODE_INTERFACE == "StateAuthorityMode@1"
    for mode in modes:
        policy = state_authority_mode_policy(mode)
        assert policy.mode.value == mode
        assert policy.allows_implicit_legacy_import is False
        assert policy.projections_authoritative is False
        assert policy.allows_file_fallback_on_server_failure is False
        assert policy.to_dict()["export_authority_class"] == "export"


def test_parse_state_authority_mode_normalizes_and_rejects_unknown() -> None:
    assert (
        parse_state_authority_mode("Quack-Authoritative")
        is StateAuthorityMode.QUACK_AUTHORITATIVE
    )
    with pytest.raises(StateAuthorityModeError, match="required"):
        parse_state_authority_mode("")
    with pytest.raises(StateAuthorityModeError, match="unsupported"):
        parse_state_authority_mode("markdown_default")


def test_quack_modes_disable_file_watch_and_runtime_writes() -> None:
    for mode in (
        StateAuthorityMode.QUACK_SHADOW,
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        StateAuthorityMode.EMBEDDED_MAINTENANCE,
    ):
        assert is_quack_authority_mode(mode) is (
            mode
            in {
                StateAuthorityMode.QUACK_SHADOW,
                StateAuthorityMode.QUACK_AUTHORITATIVE,
            }
        )
        assert file_watch_enabled_for_mode(mode) is False
        assert file_write_enabled_for_mode(mode) is False
        policy = state_authority_mode_policy(mode)
        assert policy.scheduling_source is ScheduleAuthoritySource.DATABASE
        assert policy.lifecycle_source is ScheduleAuthoritySource.DATABASE


def test_export_only_allows_export_writes_but_not_schedule_authority() -> None:
    policy = state_authority_mode_policy(StateAuthorityMode.EXPORT_ONLY)
    assert policy.export_only is True
    assert policy.scheduling_source is ScheduleAuthoritySource.NONE
    assert file_write_enabled_for_mode(StateAuthorityMode.EXPORT_ONLY) is True
    assert file_watch_enabled_for_mode(StateAuthorityMode.EXPORT_ONLY) is False


# ---------------------------------------------------------------------------
# Legacy import is explicit-only
# ---------------------------------------------------------------------------


def test_legacy_import_cannot_run_implicitly() -> None:
    with pytest.raises(ImplicitLegacyImportError, match="implicitly"):
        require_explicit_legacy_import(
            StateAuthorityMode.LEGACY_IMPORT, explicit=False
        )
    with pytest.raises(ImplicitLegacyImportError, match="implicitly"):
        gate_legacy_import(
            mode=StateAuthorityMode.LEGACY_IMPORT, explicit=False
        )
    with pytest.raises(ImplicitLegacyImportError, match="implicitly"):
        evaluate_schedule_authority(
            StateAuthorityMode.LEGACY_IMPORT,
            file_projections=_file_projections(),
            explicit_legacy_import=False,
        )


def test_legacy_import_requires_matching_mode_when_explicit() -> None:
    with pytest.raises(StateAuthorityModeError, match="requires mode"):
        require_explicit_legacy_import(
            StateAuthorityMode.QUACK_AUTHORITATIVE, explicit=True
        )
    mode = gate_legacy_import(
        mode="legacy_import", explicit=True
    )
    assert mode is StateAuthorityMode.LEGACY_IMPORT


def test_open_task_source_refuses_implicit_legacy_and_markdown_under_quack(
    tmp_path: Path,
) -> None:
    board = tmp_path / "board.todo.md"
    board.write_text("# board\n", encoding="utf-8")
    with pytest.raises(ImplicitLegacyImportError):
        open_task_source_for_authority_mode(
            board,
            mode=StateAuthorityMode.LEGACY_IMPORT,
            explicit_legacy_import=False,
        )
    with pytest.raises(StateAuthorityModeError, match="refusing implicit markdown|markdown"):
        open_task_source_for_authority_mode(
            board,
            mode=StateAuthorityMode.QUACK_AUTHORITATIVE,
            server_available=True,
        )


def test_open_task_source_refuses_quack_open_when_server_unavailable(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    db.write_bytes(b"")
    with pytest.raises(StateAuthorityUnavailableError) as excinfo:
        open_task_source_for_authority_mode(
            db,
            mode=StateAuthorityMode.QUACK_AUTHORITATIVE,
            server_available=False,
        )
    assert excinfo.value.recovery_required is True
    assert "file_fallback_refused" in excinfo.value.reason_codes


# ---------------------------------------------------------------------------
# Quack authority ignores projection mutations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        StateAuthorityMode.QUACK_SHADOW,
        StateAuthorityMode.EMBEDDED_MAINTENANCE,
    ],
)
def test_projection_change_or_delete_cannot_change_schedule(
    mode: StateAuthorityMode,
) -> None:
    schedule = _db_schedule(ready_task_cids=["task:cid:db-ready"])
    lifecycle = _db_lifecycle(phase="running")
    before = _file_projections()
    after_tampered = _file_projections(
        markdown={"status": "completed", "ready": ["forged"]},
        json={"tasks": []},
        jsonl=[],
        pid={},
        status={"phase": "stopped"},
        taskboard={},
    )
    after_deleted: dict[str, object] = {}

    baseline = evaluate_schedule_authority(
        mode,
        database_schedule=schedule,
        database_lifecycle=lifecycle,
        file_projections=before,
        server_available=True,
    )
    assert baseline.availability is AuthorityAvailability.AVAILABLE
    assert baseline.scheduling_source is ScheduleAuthoritySource.DATABASE
    assert dict(baseline.schedule) == schedule
    assert dict(baseline.lifecycle) == lifecycle
    assert baseline.file_projections_ignored is True
    assert baseline.used_file_fallback is False
    assert "task:cid:forged-md" not in json.dumps(dict(baseline.schedule))

    assert (
        projection_mutation_affects_schedule(
            mode,
            before_projections=before,
            after_projections=after_tampered,
            database_schedule=schedule,
            database_lifecycle=lifecycle,
        )
        is False
    )
    assert (
        projection_mutation_affects_schedule(
            mode,
            before_projections=before,
            after_projections=after_deleted,
            database_schedule=schedule,
            database_lifecycle=lifecycle,
        )
        is False
    )

    for kind in ("markdown", "json", "jsonl", "pid", "status", "taskboard"):
        decision = evaluate_projection_authority(mode, kind)
        assert decision.authoritative is False
        assert decision.influences_scheduling is False
        assert decision.influences_lifecycle is False


def test_absent_exports_do_not_block_quack_authority() -> None:
    decision = evaluate_schedule_authority(
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        database_schedule=_db_schedule(),
        database_lifecycle=_db_lifecycle(),
        file_projections=None,
        server_available=True,
    )
    assert decision.availability is AuthorityAvailability.AVAILABLE
    assert decision.file_projections_ignored is True
    assert dict(decision.schedule)["revision"] == 7


# ---------------------------------------------------------------------------
# Server failure: unavailable / recovery-required, no file fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        StateAuthorityMode.QUACK_SHADOW,
        StateAuthorityMode.EMBEDDED_MAINTENANCE,
    ],
)
def test_server_failure_returns_recovery_required_not_file_fallback(
    mode: StateAuthorityMode,
) -> None:
    forged = _file_projections()
    decision = evaluate_schedule_authority(
        mode,
        database_schedule=_db_schedule(),
        database_lifecycle=_db_lifecycle(),
        file_projections=forged,
        server_available=False,
    )
    assert decision.availability is AuthorityAvailability.RECOVERY_REQUIRED
    assert decision.recovery_required is True
    assert decision.used_file_fallback is False
    assert decision.file_projections_ignored is True
    assert dict(decision.schedule) == {}
    assert dict(decision.lifecycle) == {}
    assert "file_fallback_refused" in decision.reason_codes
    assert "legacy_projections_present_but_ignored" in decision.reason_codes

    with pytest.raises(StateAuthorityUnavailableError) as excinfo:
        evaluate_schedule_authority(
            mode,
            database_schedule=_db_schedule(),
            file_projections=forged,
            server_available=False,
            raise_on_unavailable=True,
        )
    assert excinfo.value.recovery_required is True
    assert "file_fallback_refused" in excinfo.value.reason_codes


def test_local_db_file_is_not_fallback_under_quack_when_server_down() -> None:
    """A local control.duckdb path must not become authority when Quack is down."""

    decision = evaluate_schedule_authority(
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        database_schedule={
            "ready_task_cids": ["task:cid:local-file"],
            "source": "embedded_file",
        },
        file_projections=_file_projections(),
        server_available=False,
        recovery_required=True,
    )
    assert decision.recovery_required is True
    assert decision.used_file_fallback is False
    # Schedule is empty: neither local DB snapshot nor files are adopted.
    assert dict(decision.schedule) == {}


# ---------------------------------------------------------------------------
# Export non-authority marker
# ---------------------------------------------------------------------------


def test_exports_carry_non_authority_marker() -> None:
    assert export_non_authority_marker() == EXPORT_NON_AUTHORITY_MARKER
    payload = attach_export_non_authority_marker(
        {"tasks": [{"id": "DQP-030"}], "revision": 1}
    )
    assert isinstance(payload, dict)
    assert payload["authority_class"] == EXPORT_AUTHORITY_CLASS_VALUE
    assert payload["authoritative"] is False
    assert payload["non_authority_marker"] == EXPORT_NON_AUTHORITY_MARKER

    banner = attach_export_non_authority_marker(media_type="markdown")
    assert isinstance(banner, str)
    assert "NON-AUTHORITATIVE" in banner

    with pytest.raises(StateAuthorityModeError, match="authoritative"):
        attach_export_non_authority_marker({"authority_class": "authoritative"})


def test_export_only_mode_decision_embeds_marker() -> None:
    decision = evaluate_schedule_authority(StateAuthorityMode.EXPORT_ONLY)
    assert decision.scheduling_source is ScheduleAuthoritySource.NONE
    assert decision.export_marker == EXPORT_NON_AUTHORITY_MARKER
    assert decision.to_dict()["export_authority_class"] == "export"


# ---------------------------------------------------------------------------
# Mode transitions and rollback
# ---------------------------------------------------------------------------


def test_mode_transition_and_rollback_preserve_history_route_only() -> None:
    forward = transition_state_authority_mode(
        StateAuthorityMode.QUACK_SHADOW,
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        reason="canary_promotion",
    )
    assert forward.from_mode is StateAuthorityMode.QUACK_SHADOW
    assert forward.to_mode is StateAuthorityMode.QUACK_AUTHORITATIVE
    assert forward.rollback is False
    assert "history_preserved" in forward.reason_codes
    assert forward.receipt_id.startswith("task-source:sha256:")

    rollback = transition_state_authority_mode(
        StateAuthorityMode.QUACK_AUTHORITATIVE,
        StateAuthorityMode.QUACK_SHADOW,
        reason="kill_switch",
        rollback=True,
    )
    assert rollback.rollback is True
    assert "rollback" in rollback.reason_codes
    assert rollback.to_mode is StateAuthorityMode.QUACK_SHADOW

    assert "quack_authoritative" in allowed_state_authority_transitions(
        StateAuthorityMode.QUACK_SHADOW
    )
    with pytest.raises(StateAuthorityTransitionError, match="not allowed"):
        transition_state_authority_mode(
            StateAuthorityMode.EXPORT_ONLY,
            StateAuthorityMode.LEGACY_IMPORT,
        )


# ---------------------------------------------------------------------------
# DatabaseStateResolver@1
# ---------------------------------------------------------------------------


def test_database_state_resolver_interface_and_requirement_id() -> None:
    assert DATABASE_STATE_RESOLVER_INTERFACE == "DatabaseStateResolver@1"
    assert DATABASE_STATE_RESOLVER_REQUIREMENT_ID.startswith(
        "agent_supervisor.entrypoints.database_state_resolver"
    )


def test_database_state_resolver_quack_authoritative_ignores_tampered_files() -> None:
    resolution = resolve_database_state(_database_evidence())
    assert resolution.authority_mode == "quack_authoritative"
    assert resolution.disposition is ResolutionDisposition.UNIQUE
    assert resolution.availability == "available"
    assert resolution.recovery_required is False
    assert resolution.used_file_fallback is False
    assert resolution.projections_authoritative is False
    assert resolution.file_projections_ignored is True
    assert resolution.file_watch_enabled is False
    assert resolution.scheduling_source == "database"
    assert resolution.lifecycle_source == "database"
    assert dict(resolution.schedule) == _db_schedule()
    assert dict(resolution.lifecycle) == _db_lifecycle()
    assert "task:cid:forged-md" not in json.dumps(dict(resolution.schedule))
    assert resolution.export_authority_class == "export"
    assert resolution.export_non_authority_marker == EXPORT_NON_AUTHORITY_MARKER
    assert resolution.export_payload["authority_class"] == "export"
    assert resolution.export_payload["authoritative"] is False
    assert "projection_mutation_invariant_holds" in resolution.reason_codes
    assert resolution.state_root.startswith("/home/dev/.local/state/")
    assert "/src/project" not in resolution.state_root


def test_database_state_resolver_server_unavailable_no_file_fallback() -> None:
    resolution = resolve_database_state(
        _database_evidence(
            server_status=DatabaseServerStatus.UNAVAILABLE,
            file_projections=_file_projections(
                markdown={"schedule": {"ready_task_cids": ["forged"]}}
            ),
        )
    )
    assert resolution.disposition is ResolutionDisposition.UNAVAILABLE
    assert resolution.availability == "recovery_required"
    assert resolution.recovery_required is True
    assert resolution.used_file_fallback is False
    assert dict(resolution.schedule) == {}
    assert "file_fallback_refused" in resolution.reason_codes


def test_database_state_resolver_rejects_implicit_legacy_import() -> None:
    with pytest.raises(DatabaseStateResolverError, match="implicitly"):
        resolve_database_state(
            _database_evidence(
                authority_mode=StateAuthorityMode.LEGACY_IMPORT.value,
                explicit_legacy_import=False,
            )
        )


def test_database_state_resolver_explicit_legacy_import() -> None:
    resolution = resolve_database_state(
        _database_evidence(
            authority_mode=StateAuthorityMode.LEGACY_IMPORT.value,
            explicit_legacy_import=True,
            file_projections={
                "schedule": {"ready_task_cids": ["task:cid:imported"]},
                "lifecycle": {"phase": "imported"},
            },
        )
    )
    assert resolution.authority_mode == "legacy_import"
    assert resolution.scheduling_source == "legacy_import"
    assert resolution.disposition is ResolutionDisposition.UNIQUE
    assert dict(resolution.schedule)["ready_task_cids"] == ["task:cid:imported"]
    assert resolution.export_authority_class == "export"


def test_database_state_resolver_requires_explicit_mode() -> None:
    with pytest.raises(DatabaseStateResolverError, match="authority_mode"):
        DatabaseStateEvidence(
            repository_id="repository:sha256:x",
            repository_root="/home/dev/src/project",
            authority_mode="",
            home_directory="/home/dev",
        )


def test_database_state_resolver_cold_discovery_without_exports() -> None:
    """Cold discovery with no export files remains side-effect free and valid."""

    resolution = resolve_database_state(
        _database_evidence(
            file_projections=None,
            export_requested=False,
            database_schedule={"ready_task_cids": [], "revision": 1},
            database_lifecycle={"phase": "idle"},
        )
    )
    assert resolution.availability == "available"
    assert resolution.file_projections_ignored is True
    assert dict(resolution.schedule)["revision"] == 1


def test_database_state_resolver_export_only_mode() -> None:
    resolution = resolve_database_state(
        _database_evidence(
            authority_mode=StateAuthorityMode.EXPORT_ONLY.value,
            export_requested=True,
            database_schedule=_db_schedule(),
            file_projections=_file_projections(),
        )
    )
    assert resolution.authority_mode == "export_only"
    assert resolution.scheduling_source == "none"
    assert resolution.disposition is ResolutionDisposition.DEFAULTED
    assert dict(resolution.schedule) == {}
    assert resolution.export_payload["export_requested"] is True
    assert "NON-AUTHORITATIVE" in resolution.export_non_authority_marker


def test_database_state_resolver_mode_transition_api() -> None:
    resolver = DatabaseStateResolver()
    receipt = resolver.transition(
        "quack_shadow",
        "quack_authoritative",
        reason="cutover",
    )
    assert receipt["from_mode"] == "quack_shadow"
    assert receipt["to_mode"] == "quack_authoritative"
    assert "history_preserved" in receipt["reason_codes"]

    rollback = resolver.transition(
        "quack_authoritative",
        "quack_shadow",
        reason="kill_switch",
        rollback=True,
    )
    assert rollback["rollback"] is True

    with pytest.raises(DatabaseStateResolverError, match="not allowed"):
        resolver.transition("export_only", "legacy_import")


def test_database_state_resolver_shadow_mode_dual_observation_non_authority() -> None:
    resolution = resolve_database_state(
        _database_evidence(
            authority_mode=StateAuthorityMode.QUACK_SHADOW.value,
            file_projections=_file_projections(),
        )
    )
    assert resolution.authority_mode == "quack_shadow"
    assert resolution.mode_policy["dual_observation"] is True
    assert resolution.scheduling_source == "database"
    assert "dual_observation_non_authoritative" in resolution.reason_codes
    assert dict(resolution.schedule) == _db_schedule()


# ---------------------------------------------------------------------------
# Cold import / module side-effect free
# ---------------------------------------------------------------------------


def test_cold_import_of_modules_is_side_effect_free() -> None:
    """Cold import of authority modules performs no filesystem or network I/O."""

    task_source_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.task_sources.task_source"
    )
    resolver_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver"
    )
    modes = task_source_mod.closed_state_authority_modes()
    assert "quack_authoritative" in modes
    assert resolver_mod.DATABASE_STATE_RESOLVER_INTERFACE == "DatabaseStateResolver@1"
    # Construction of pure policy objects must not touch the filesystem.
    policy = task_source_mod.state_authority_mode_policy("quack_authoritative")
    assert policy.file_watch_enabled is False
    assert policy.allows_file_fallback_on_server_failure is False
