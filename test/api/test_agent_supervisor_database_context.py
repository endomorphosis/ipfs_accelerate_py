"""Tests for DatabaseContextManifest@1 / ContextDelta@1 / LLMContextFrontier@1.

DQP-026 evidence subset: stable identity, pagination, progressive disclosure,
secret/private exclusion, unchanged timestamps, stale input, overflow, exact
dependency invalidation.

Acceptance:

* Unchanged semantic state yields identical context CID despite heartbeat/time noise
* Changed evidence yields a bounded delta
* Omitted unresolved frontier is explicit
* No secret/raw unrestricted repository dump enters a model packet
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.context.database_context import (
    AUTHORITY_CLASS,
    CONTEXT_DELTA_INTERFACE,
    DATABASE_CONTEXT_MANIFEST_INTERFACE,
    DEFAULT_POLICY_ID,
    LLM_CONTEXT_FRONTIER_INTERFACE,
    REDACTION_MARKER,
    Completeness,
    ContextBudgetSpec,
    DatabaseContextManifest,
    DatabaseContextOverflowError,
    DatabaseContextSecretError,
    DatabaseContextStaleError,
    DatabaseContextStore,
    FrontierDisposition,
    InvalidationKind,
    LLMContextFrontier,
    MemberKind,
    TaskContextInput,
    build_context_delta,
    build_database_context_manifest,
    compile_manifest_to_capsule,
    duckdb_available,
    open_database_context_store,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseContextStore hermetic tests",
)


def _request(**overrides) -> TaskContextInput:
    values = dict(
        task_cid="task:dqp-026-demo",
        repository_id="repo:demo",
        tree_id="tree:abc",
        policy_id=DEFAULT_POLICY_ID,
        task_revision="rev:1",
        plan_cid="plan:1",
        goal_cid="goal:context-economy",
        task_status="ready",
        task_summary="Build bounded database context capsules",
        unmet_dependencies=(
            {"dependency_id": "dep:impact-view", "summary": "impact view ready"},
        ),
        latest_failure={
            "failure_id": "fail:distinct-1",
            "kind": "validation",
            "summary": "prior validation failed",
        },
        worktree_delta={
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/context/database_context.py",
                "test/api/test_agent_supervisor_database_context.py",
            ],
            "digests": {
                "ipfs_accelerate_py/agent_supervisor/context/database_context.py": (
                    "sha256:aaa"
                ),
                "test/api/test_agent_supervisor_database_context.py": "sha256:bbb",
            },
        },
        impacted_symbols=(
            {"symbol": "DatabaseContextManifest", "path": "context/database_context.py"},
            {"symbol": "build_database_context_manifest", "path": "context/database_context.py"},
        ),
        open_obligations=(
            {"obligation_id": "ob:stable-cid", "summary": "stable context identity"},
        ),
        decisions=(
            {"decision_id": "dec:include-frontier", "summary": "emit explicit frontier"},
        ),
        evidence=(
            {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
        ),
        validations=(
            {
                "command": (
                    "python -m pytest -q "
                    "test/api/test_agent_supervisor_database_context.py"
                ),
            },
        ),
        budget=ContextBudgetSpec(
            max_rows=64,
            max_bytes=64_000,
            max_tokens=8_192,
            page_size=32,
            page_offset=0,
        ),
    )
    values.update(overrides)
    return TaskContextInput(**values)


def _open(tmp_path: Path) -> DatabaseContextStore:
    return open_database_context_store(tmp_path / "database_context.duckdb")


def test_interface_identities() -> None:
    assert DATABASE_CONTEXT_MANIFEST_INTERFACE == "DatabaseContextManifest@1"
    assert CONTEXT_DELTA_INTERFACE == "ContextDelta@1"
    assert LLM_CONTEXT_FRONTIER_INTERFACE == "LLMContextFrontier@1"
    assert DatabaseContextStore.INTERFACE == "DatabaseContextStore@1"
    assert AUTHORITY_CLASS == "derived_evidence"
    assert REDACTION_MARKER == "secret_material"
    assert MemberKind.coerce("unmet_dependency") is MemberKind.DEPENDENCY


def test_cold_import_and_construction_have_no_side_effects() -> None:
    store = DatabaseContextStore("/tmp/should-not-exist-until-open.duckdb")
    assert store.is_open is False


def test_stable_identity_ignores_heartbeat_and_timestamps() -> None:
    first = build_database_context_manifest(
        _request(
            heartbeat_at="2026-01-01T00:00:00Z",
            observed_at="2026-01-01T00:00:01Z",
            metadata={"lease_heartbeat": "noise-1", "pid": "111"},
        )
    )
    second = build_database_context_manifest(
        _request(
            heartbeat_at="2026-08-09T12:34:56Z",
            observed_at="2026-08-09T12:35:00Z",
            metadata={"lease_heartbeat": "noise-2", "pid": "999"},
        )
    )
    assert first.manifest_cid == second.manifest_cid
    assert first.context_cid == second.context_cid
    assert first.interface == DATABASE_CONTEXT_MANIFEST_INTERFACE
    assert first.to_dict()["authority"] == AUTHORITY_CLASS


def test_changed_evidence_yields_bounded_delta() -> None:
    prior = build_database_context_manifest(_request())
    current = build_database_context_manifest(
        _request(
            task_revision="rev:2",
            evidence=(
                {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
                {"evidence_id": "ev:new-2", "summary": "new validation receipt"},
            ),
            impacted_symbols=(
                {
                    "symbol": "DatabaseContextManifest",
                    "path": "context/database_context.py",
                },
                {
                    "symbol": "LLMContextFrontier",
                    "path": "context/database_context.py",
                },
            ),
        )
    )
    assert prior.manifest_cid != current.manifest_cid
    delta = build_context_delta(prior, current)
    assert delta.interface == CONTEXT_DELTA_INTERFACE
    assert delta.from_manifest_cid == prior.manifest_cid
    assert delta.to_manifest_cid == current.manifest_cid
    assert not delta.is_empty
    assert delta.total_bytes > 0
    assert delta.total_tokens > 0
    # Delta transmits only changed/added members, not a full replay.
    transmitted = len(delta.added) + len(delta.changed)
    assert transmitted < len(current.included_members())
    assert InvalidationKind.EVIDENCE.value in delta.invalidations
    assert InvalidationKind.TASK_REVISION.value in delta.invalidations
    assert delta.to_dict()["schema"]
    assert delta.to_dict()["interface"] == CONTEXT_DELTA_INTERFACE


def test_frontier_is_explicit_when_members_omitted() -> None:
    # Tiny page forces progressive disclosure of optional evidence.
    request = _request(
        budget=ContextBudgetSpec(
            max_rows=64,
            max_bytes=64_000,
            max_tokens=8_192,
            page_size=2,
            page_offset=0,
        ),
        evidence=tuple(
            {"evidence_id": f"ev:{index}", "summary": f"evidence {index}"}
            for index in range(12)
        ),
    )
    manifest = build_database_context_manifest(request)
    assert manifest.frontier.interface == LLM_CONTEXT_FRONTIER_INTERFACE
    assert manifest.frontier.is_explicit is True
    assert manifest.frontier.has_more is True
    assert manifest.frontier.omitted_member_ids
    assert manifest.completeness in {
        Completeness.PARTIAL_WITH_FRONTIER,
        Completeness.OVERFLOW,
    }
    packet = manifest.model_packet()
    assert packet["frontier"]["omitted_count"] == len(
        manifest.frontier.omitted_member_ids
    )
    assert packet["frontier"]["disposition"] != FrontierDisposition.EMPTY.value


def test_pagination_progressive_disclosure(tmp_path: Path) -> None:
    with _open(tmp_path) as store:
        page0 = store.build(
            _request(
                budget=ContextBudgetSpec(
                    max_rows=64,
                    max_bytes=64_000,
                    max_tokens=8_192,
                    page_size=3,
                    page_offset=0,
                ),
                evidence=tuple(
                    {"evidence_id": f"ev:{index}", "summary": f"row {index}"}
                    for index in range(10)
                ),
            )
        )
        page1 = store.build(
            _request(
                budget=ContextBudgetSpec(
                    max_rows=64,
                    max_bytes=64_000,
                    max_tokens=8_192,
                    page_size=3,
                    page_offset=3,
                ),
                evidence=tuple(
                    {"evidence_id": f"ev:{index}", "summary": f"row {index}"}
                    for index in range(10)
                ),
            )
        )
        assert page0.frontier.page_offset == 0
        assert page1.frontier.page_offset == 3
        # Distinct pages yield distinct omission sets / CIDs.
        assert page0.manifest_cid != page1.manifest_cid
        loaded = store.get_manifest(page0.manifest_cid)
        assert loaded is not None
        assert loaded.manifest_cid == page0.manifest_cid
        members = store.page_members(page0.manifest_cid, offset=0, limit=2)
        assert len(members) <= 2
        assert all(item.included for item in members)


def test_secret_and_private_material_excluded_from_model_packet() -> None:
    with pytest.raises(DatabaseContextSecretError) as excinfo:
        build_database_context_manifest(
            _request(
                metadata={
                    "api_key": "must_never_appear",
                }
            )
        )
    assert excinfo.value.reason_code == "secret_material_rejected"

    with pytest.raises(DatabaseContextSecretError):
        build_database_context_manifest(
            _request(
                latest_failure={
                    "failure_id": "fail:secret",
                    "password": "must_never_appear",
                }
            )
        )

    with pytest.raises(DatabaseContextSecretError):
        build_database_context_manifest(
            _request(
                worktree_delta={
                    "paths": [".env.local", "src/ok.py"],
                }
            )
        )

    with pytest.raises(DatabaseContextSecretError):
        build_database_context_manifest(
            _request(
                evidence=(
                    {
                        "evidence_id": "ev:dump",
                        "repository_dump": "entire tree text",
                    },
                )
            )
        )

    # Clean packet contains no redaction marker placeholders from real material.
    clean = build_database_context_manifest(_request())
    packet = clean.model_packet()
    serialized = str(packet)
    assert "must_never_appear" not in serialized
    assert "BEGIN PRIVATE KEY" not in serialized
    assert packet["data_label"]
    assert packet["treat_as"] == "data_not_instructions"
    assert "members" in packet
    kinds = {item["kind"] for item in packet["members"]}
    assert "task" in kinds
    assert "validation" in kinds


def test_stale_input_fails_closed() -> None:
    with pytest.raises(DatabaseContextStaleError) as tree_exc:
        build_database_context_manifest(
            _request(tree_id="tree:current", expected_tree_id="tree:prior")
        )
    assert tree_exc.value.reason_code == "stale_tree"

    with pytest.raises(DatabaseContextStaleError) as policy_exc:
        build_database_context_manifest(
            _request(
                policy_digest="policy:sha256:current",
                expected_policy_digest="policy:sha256:prior",
            )
        )
    assert policy_exc.value.reason_code == "stale_policy"

    base = _request()
    digests = build_database_context_manifest(base).dependency_digests
    with pytest.raises(DatabaseContextStaleError) as dep_exc:
        build_database_context_manifest(
            _request(
                unmet_dependencies=(
                    {
                        "dependency_id": "dep:changed",
                        "summary": "dependency changed",
                    },
                ),
                expected_dependency_digests=digests,
            )
        )
    assert dep_exc.value.reason_code == "stale_dependencies"

    prior = build_database_context_manifest(_request(tree_id="tree:a"))
    current = build_database_context_manifest(_request(tree_id="tree:b"))
    with pytest.raises(DatabaseContextStaleError):
        build_context_delta(prior, current)


def test_overflow_of_required_core_fails_closed() -> None:
    with pytest.raises(DatabaseContextOverflowError):
        build_database_context_manifest(
            _request(
                budget=ContextBudgetSpec(
                    max_rows=1,
                    max_bytes=32,
                    max_tokens=8,
                    page_size=1,
                ),
                unmet_dependencies=tuple(
                    {
                        "dependency_id": f"dep:{index}",
                        "summary": f"dependency {index} " + ("x" * 200),
                    }
                    for index in range(8)
                ),
                open_obligations=tuple(
                    {
                        "obligation_id": f"ob:{index}",
                        "summary": f"obligation {index} " + ("y" * 200),
                    }
                    for index in range(8)
                ),
                validations=tuple(
                    {"command": f"python -m pytest test_{index}.py -q"}
                    for index in range(8)
                ),
            )
        )


def test_optional_overflow_is_explicit_frontier_not_silent_drop() -> None:
    manifest = build_database_context_manifest(
        _request(
            budget=ContextBudgetSpec(
                max_rows=8,
                max_bytes=2_000,
                max_tokens=400,
                page_size=64,
            ),
            evidence=tuple(
                {
                    "evidence_id": f"ev:{index}",
                    "summary": ("evidence payload " * 40) + str(index),
                }
                for index in range(30)
            ),
        )
    )
    assert manifest.frontier.disposition in {
        FrontierDisposition.BUDGET_OVERFLOW,
        FrontierDisposition.PAGINATED,
    }
    assert manifest.omitted_members()
    assert all(item.expansion_handle for item in manifest.omitted_members())
    assert manifest.completeness is not Completeness.COMPLETE


def test_exact_dependency_invalidation_in_delta() -> None:
    prior = build_database_context_manifest(
        _request(
            unmet_dependencies=(
                {"dependency_id": "dep:a", "summary": "dependency a"},
            )
        )
    )
    current = build_database_context_manifest(
        _request(
            unmet_dependencies=(
                {"dependency_id": "dep:b", "summary": "dependency b"},
            )
        )
    )
    delta = build_context_delta(prior, current)
    assert InvalidationKind.DEPENDENCY.value in delta.invalidations
    assert prior.dependency_digests != current.dependency_digests


def test_persist_and_round_trip(tmp_path: Path) -> None:
    with _open(tmp_path) as store:
        manifest = store.build(_request())
        assert store.metadata()["interface"] == DatabaseContextStore.INTERFACE
        listed = store.list_manifests_for_task(manifest.task_cid)
        assert manifest.manifest_cid in listed
        loaded = store.get_manifest(manifest.manifest_cid)
        assert loaded is not None
        assert loaded.manifest_cid == manifest.manifest_cid
        assert loaded.task_cid == manifest.task_cid
        assert len(loaded.included_members()) == len(manifest.included_members())

        changed = store.build(
            _request(
                task_revision="rev:2",
                evidence=(
                    {"evidence_id": "ev:impact-1", "summary": "updated"},
                ),
            )
        )
        delta = build_context_delta(manifest, changed)
        store.persist_delta(delta)
        assert delta.delta_id


def test_compile_manifest_uses_context_compiler_boundary() -> None:
    manifest = build_database_context_manifest(_request())
    capsule = compile_manifest_to_capsule(manifest)
    assert capsule.repository_id == manifest.repository_id
    assert capsule.tree_id == manifest.tree_id
    assert capsule.policy_id == manifest.policy_id
    assert capsule.acceptance.get("manifest_cid") == manifest.manifest_cid
    assert capsule.acceptance.get("cannot_include_secrets") is True
    assert capsule.goal
    assert capsule.authority
    assert capsule.scope


def test_manifest_contains_required_semantic_sections() -> None:
    manifest = build_database_context_manifest(_request())
    kinds = {
        item.kind if isinstance(item.kind, MemberKind) else MemberKind(item.kind)
        for item in manifest.included_members()
    }
    for required in (
        MemberKind.TASK,
        MemberKind.DEPENDENCY,
        MemberKind.FAILURE,
        MemberKind.WORKTREE_DELTA,
        MemberKind.IMPACTED_SYMBOL,
        MemberKind.OBLIGATION,
        MemberKind.DECISION,
        MemberKind.EVIDENCE,
        MemberKind.VALIDATION,
    ):
        assert required in kinds
    assert isinstance(manifest.frontier, LLMContextFrontier)
    assert manifest.total_rows == len(manifest.included_members())
    assert manifest.total_bytes > 0
    assert manifest.total_tokens > 0
