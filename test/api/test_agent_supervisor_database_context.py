"""Tests for DatabaseContextManifest@1 / ContextDelta@1 / LLMContextFrontier@1 (DQP-026).

Evidence subset: stable identity, pagination, progressive disclosure,
secret/private exclusion, unchanged timestamps, stale input, overflow,
exact dependency invalidation.

Acceptance:

* Unchanged semantic state yields identical context CID despite heartbeat/time noise
* Changed evidence yields a bounded delta
* Omitted unresolved frontier is explicit
* No secret/raw unrestricted repository dump enters a model packet
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.context.database_context import (
    AUTHORITY_CLASS,
    CONTEXT_DELTA_INTERFACE,
    DATABASE_CONTEXT_MANIFEST_INTERFACE,
    LLM_CONTEXT_FRONTIER_INTERFACE,
    NOISE_FIELD_NAMES,
    REQUIRED_MEMBER_KINDS,
    UNTRUSTED_DATA_LABEL,
    ContextDelta,
    ContextMemberKind,
    DatabaseContextBudget,
    DatabaseContextBoundsError,
    DatabaseContextError,
    DatabaseContextInvalidationError,
    DatabaseContextManifest,
    DatabaseContextRequest,
    DatabaseContextSecretError,
    DatabaseContextStaleError,
    FrontierDisposition,
    FrontierKind,
    LLMContextFrontier,
    assert_dependency_freshness,
    compare_and_delta,
    compile_database_context,
    estimate_tokens,
    is_noise_field,
    model_packet_from_manifest,
    page_members,
    project_to_context_compiler_inputs,
    strip_noise,
)


def _budget(**overrides) -> DatabaseContextBudget:
    values = dict(
        max_rows=64,
        max_bytes=32_000,
        max_tokens=8_000,
        max_item_bytes=2_048,
        max_text_bytes=1_024,
        page_size=8,
        overflow_behavior="frontier",
    )
    values.update(overrides)
    return DatabaseContextBudget(**values)


def _request(**overrides) -> DatabaseContextRequest:
    values: dict = dict(
        task_cid="task:dqp-026-demo",
        repository_id="repo:demo",
        tree_id="tree:abc123",
        schema_revision=1,
        policy_digest="policy:sha256:demo",
        goal_cid="goal:dqp-g050",
        plan_cid="plan:demo-1",
        task_revision=3,
        snapshot_id="snapshot:1",
        parser_id="python-ast@test",
        task={
            "task_cid": "task:dqp-026-demo",
            "task_alias": "DQP-026",
            "title": "Build bounded database context capsules",
            "status": "ready",
            "revision": 3,
            "goal_cid": "goal:dqp-g050",
            "plan_cid": "plan:demo-1",
            # Noise that must not affect CID:
            "heartbeat_at": "2026-08-09T12:00:00Z",
            "heartbeat_at_ms": 1_723_200_000_000,
            "lease_expires_at_ms": 1_723_200_060_000,
            "observed_at": "2026-08-09T12:00:01Z",
        },
        unmet_dependencies=(
            {
                "dependency_id": "dep:dqp-012",
                "status": "unmet",
                "summary": "objectives migration",
                "created_at": "noise",
            },
            {
                "dependency_id": "dep:dqp-023",
                "status": "unmet",
                "summary": "impact graph",
            },
        ),
        latest_failure={
            "signature_id": "fail:sig-1",
            "failure_kind": "validation_failed",
            "summary": "pytest exit 1",
            "signature_digest": "sha256:fail1",
            "last_seen_at": "should-be-stripped",
        },
        worktree_delta=(
            {
                "path": "ipfs_accelerate_py/agent_supervisor/context/database_context.py",
                "change": "modify",
                "before_digest": "sha256:before1",
                "after_digest": "sha256:after1",
            },
            {
                "path": "test/api/test_agent_supervisor_database_context.py",
                "change": "add",
                "after_digest": "sha256:after2",
            },
        ),
        impacted_symbols=(
            {
                "symbol": "compile_database_context",
                "path": "ipfs_accelerate_py/agent_supervisor/context/database_context.py",
                "disposition": "must_repair",
            },
            {
                "symbol": "DatabaseContextManifest",
                "path": "ipfs_accelerate_py/agent_supervisor/context/database_context.py",
            },
        ),
        open_obligations=(
            {
                "obligation_id": "obl:stable-cid",
                "summary": "stable identity under noise",
            },
            {
                "obligation_id": "obl:no-secret-dump",
                "summary": "no secret in model packet",
            },
        ),
        decisions=(
            {
                "decision_id": "dec:use-frontier",
                "summary": "omit overflow via frontier",
            },
        ),
        evidence=(
            {
                "evidence_id": "ev:impact-1",
                "digest": "sha256:impact1",
                "summary": "impact closure receipt",
            },
        ),
        validations=(
            "python -m pytest -q test/api/test_agent_supervisor_database_context.py",
            {
                "validation_id": "val:typecheck",
                "command": "python -m compileall ipfs_accelerate_py/agent_supervisor/context/database_context.py",
            },
        ),
        unresolved_frontier=(
            {
                "frontier_id": "frontier:dynamic-1",
                "kind": "dynamic_call",
                "disposition": "unresolved",
                "reason": "getattr dispatch unresolved",
                "blocks_automatic_repair": True,
                "heartbeat_at_ms": 999,
            },
        ),
        budget=_budget(),
        heartbeat_at="2026-08-09T12:00:00Z",
        heartbeat_at_ms=1_723_200_000_000,
        observed_at_ms=1_723_200_001_000,
        lease_expires_at_ms=1_723_200_060_000,
        compiled_at="2026-08-09T12:00:02Z",
        metadata={"request_note": "demo", "polled_at": "noise"},
    )
    values.update(overrides)
    return DatabaseContextRequest(**values)


def test_interface_identities() -> None:
    assert DATABASE_CONTEXT_MANIFEST_INTERFACE == "DatabaseContextManifest@1"
    assert CONTEXT_DELTA_INTERFACE == "ContextDelta@1"
    assert LLM_CONTEXT_FRONTIER_INTERFACE == "LLMContextFrontier@1"
    assert DatabaseContextManifest.INTERFACE == DATABASE_CONTEXT_MANIFEST_INTERFACE
    assert ContextDelta.INTERFACE == CONTEXT_DELTA_INTERFACE
    assert LLMContextFrontier.INTERFACE == LLM_CONTEXT_FRONTIER_INTERFACE
    assert AUTHORITY_CLASS == "derived_evidence"
    assert set(REQUIRED_MEMBER_KINDS) >= {
        "task",
        "unmet_dependency",
        "latest_failure",
        "worktree_delta",
        "impacted_symbol",
        "open_obligation",
        "decision",
        "evidence",
        "validation",
    }
    assert is_noise_field("heartbeat_at_ms")
    assert is_noise_field("lease_expires_at")
    assert not is_noise_field("task_cid")
    assert "heartbeat_at" in NOISE_FIELD_NAMES


def test_unchanged_semantic_state_identical_cid_despite_heartbeat_noise() -> None:
    first = compile_database_context(_request())
    second = compile_database_context(
        _request(
            heartbeat_at="2026-08-09T18:00:00Z",
            heartbeat_at_ms=1_723_221_600_000,
            observed_at="later",
            observed_at_ms=1_723_221_601_000,
            lease_expires_at_ms=1_723_221_660_000,
            compiled_at="2026-08-09T18:00:05Z",
            task={
                **dict(_request().task),
                "heartbeat_at": "later-heartbeat",
                "heartbeat_at_ms": 42,
                "lease_expires_at_ms": 99,
                "observed_at": "later-observed",
                "created_at": "noise-created",
            },
            unmet_dependencies=(
                {
                    "dependency_id": "dep:dqp-012",
                    "status": "unmet",
                    "summary": "objectives migration",
                    "created_at": "different-noise",
                    "polled_at": "x",
                },
                {
                    "dependency_id": "dep:dqp-023",
                    "status": "unmet",
                    "summary": "impact graph",
                    "updated_at_ms": 12345,
                },
            ),
            metadata={"request_note": "demo", "server_time": "noise"},
        )
    )

    assert first.manifest_cid == second.manifest_cid
    assert first.content_id == second.content_id
    assert first.capsule_id == second.capsule_id
    assert first.to_dict()["manifest_cid"] == second.to_dict()["manifest_cid"]

    # Round-trip preserves identity.
    restored = DatabaseContextManifest.from_dict(first.to_dict())
    assert restored.manifest_cid == first.manifest_cid

    delta = compare_and_delta(first, second)
    assert delta.from_manifest_cid == delta.to_manifest_cid
    assert delta.added == ()
    assert delta.changed == ()
    assert delta.removed == ()
    assert delta.byte_size >= 0
    assert delta.bounded is True


def test_changed_evidence_yields_bounded_delta() -> None:
    prior = compile_database_context(_request())
    current = compile_database_context(
        _request(
            evidence=(
                {
                    "evidence_id": "ev:impact-1",
                    "digest": "sha256:impact1-changed",
                    "summary": "impact closure receipt refreshed",
                },
                {
                    "evidence_id": "ev:new",
                    "digest": "sha256:new",
                    "summary": "new evidence",
                },
            ),
            decisions=(),  # remove prior decision
        )
    )

    assert prior.manifest_cid != current.manifest_cid
    delta = compare_and_delta(prior, current)

    assert delta.interface == CONTEXT_DELTA_INTERFACE
    assert delta.from_manifest_cid == prior.manifest_cid
    assert delta.to_manifest_cid == current.manifest_cid
    assert delta.bounded is True
    assert delta.byte_size < current.byte_size
    assert delta.token_estimate < current.token_estimate or delta.unchanged_count > 0

    changed_ids = {item.member_id for item in delta.changed}
    added_ids = {item.member_id for item in delta.added}
    assert any("ev:impact-1" in mid or mid.endswith("ev:impact-1") for mid in changed_ids) or any(
        "ev:impact-1" in mid for mid in added_ids | changed_ids
    )
    assert any("ev:new" in mid for mid in added_ids)
    assert any("dec:use-frontier" in mid for mid in delta.removed)
    assert delta.unchanged_count >= 1

    restored = ContextDelta.from_dict(delta.to_dict())
    assert restored.delta_id == delta.delta_id


def test_omitted_unresolved_frontier_is_explicit() -> None:
    manifest = compile_database_context(_request())
    assert manifest.frontier.interface == LLM_CONTEXT_FRONTIER_INTERFACE
    assert manifest.frontier.entries
    assert manifest.frontier.complete is False
    assert manifest.frontier.unresolved_count >= 1
    assert manifest.frontier.blocks_automatic_repair is True

    dynamic = next(
        item
        for item in manifest.frontier.entries
        if item.kind is FrontierKind.DYNAMIC_CALL
        or item.frontier_id.endswith("dynamic-1")
    )
    assert dynamic.disposition is FrontierDisposition.UNRESOLVED
    assert dynamic.blocks_automatic_repair is True
    assert "unresolved" in dynamic.reason or dynamic.reason

    packet = model_packet_from_manifest(manifest)
    frontier = packet.frontier
    assert frontier["complete"] is False
    assert frontier["unresolved_count"] >= 1
    assert frontier["entries"]
    assert any(
        entry.get("kind") == FrontierKind.DYNAMIC_CALL.value
        or "dynamic" in entry.get("frontier_id", "")
        for entry in frontier["entries"]
    )


def test_budget_overflow_records_explicit_frontier() -> None:
    # Force optional overflow via tiny row budget while keeping required members small.
    many_symbols = tuple(
        {
            "symbol": f"sym_{index}",
            "path": f"src/mod_{index}.py",
            "summary": f"symbol {index}",
        }
        for index in range(40)
    )
    manifest = compile_database_context(
        _request(
            impacted_symbols=many_symbols,
            decisions=(),
            evidence=(),
            worktree_delta=(),
            budget=_budget(max_rows=12, page_size=4, overflow_behavior="frontier"),
        )
    )
    assert manifest.truncated is True
    assert manifest.frontier.omitted_count >= 1
    dispositions = {item.disposition for item in manifest.frontier.entries}
    assert (
        FrontierDisposition.OMITTED_BUDGET in dispositions
        or FrontierDisposition.OMITTED_PAGINATION in dispositions
    )
    # Required task still present.
    assert manifest.members_by_kind(ContextMemberKind.TASK)
    assert all(item.disclosed for item in manifest.members)


def test_pagination_and_progressive_disclosure() -> None:
    symbols = tuple(
        {"symbol": f"s{index}", "path": f"a/b_{index}.py", "summary": f"s{index}"}
        for index in range(20)
    )
    page0 = compile_database_context(
        _request(
            impacted_symbols=symbols,
            decisions=(),
            evidence=(),
            worktree_delta=(),
            page=0,
            budget=_budget(page_size=5, max_rows=64),
        )
    )
    page1 = compile_database_context(
        _request(
            impacted_symbols=symbols,
            decisions=(),
            evidence=(),
            worktree_delta=(),
            page=1,
            budget=_budget(page_size=5, max_rows=64),
        )
    )
    ids0 = set(page0.member_ids(kind=ContextMemberKind.IMPACTED_SYMBOL))
    ids1 = set(page1.member_ids(kind=ContextMemberKind.IMPACTED_SYMBOL))
    assert ids0
    assert ids1
    assert ids0.isdisjoint(ids1) or page0.frontier.next_page_token

    # page_members helper
    slice0, token = page_members(page0, page=0, page_size=2)
    assert len(slice0) <= 2
    if token:
        assert token.startswith("page:")

    # Progressive disclosure for oversized member payloads.
    huge = {
        "evidence_id": "ev:huge",
        "summary": "oversized",
        "note": "x" * 3_000,
    }
    progressive = compile_database_context(
        _request(
            evidence=(huge,),
            decisions=(),
            budget=_budget(max_item_bytes=512, max_text_bytes=400),
        )
    )
    huge_members = [
        item
        for item in progressive.members
        if item.kind is ContextMemberKind.EVIDENCE
    ]
    assert huge_members
    assert huge_members[0].payload.get("progressive") is True
    assert any(
        item.disposition is FrontierDisposition.PROGRESSIVE
        for item in progressive.frontier.entries
    )


def test_model_packet_excludes_secrets_and_repository_dumps() -> None:
    manifest = compile_database_context(_request())
    packet = model_packet_from_manifest(manifest)
    payload = packet.provider_payload()

    assert payload["manifest_cid"] == manifest.manifest_cid
    assert payload["data_label"] == UNTRUSTED_DATA_LABEL
    assert payload["treat_as"] == "data_not_instructions"
    assert payload["authority"]["completion_authoritative"] is False
    assert payload["authority"]["write_authority"] is False
    assert payload["authority"]["model_output_is_nomination_only"] is True
    assert "python -m pytest" in " ".join(payload["validation_commands"])
    assert payload["open_obligation_ids"]
    assert payload["impacted_symbols"]

    serialized = str(dict(payload)).casefold()
    for banned in (
        "source_body",
        "repository_dump",
        "private_key",
        "api_key",
        "password=",
        "-----begin",
    ):
        assert banned not in serialized

    # Reject secret-bearing input fail-closed.
    with pytest.raises(DatabaseContextSecretError):
        compile_database_context(
            _request(
                evidence=(
                    {
                        "evidence_id": "ev:secret",
                        "api_key": "sk-super-secret-value",
                        "summary": "leak",
                    },
                )
            )
        )

    with pytest.raises(DatabaseContextSecretError):
        compile_database_context(
            _request(
                decisions=(
                    {
                        "decision_id": "dec:dump",
                        "repository_dump": "entire tree " + ("A" * 100),
                    },
                )
            )
        )

    with pytest.raises(DatabaseContextSecretError):
        compile_database_context(
            _request(
                latest_failure={
                    "signature_id": "fail:secret",
                    "summary": "Authorization: Bearer supersecrettokenvalue",
                }
            )
        )


def test_stale_input_and_exact_dependency_invalidation() -> None:
    with pytest.raises(DatabaseContextStaleError):
        compile_database_context(
            _request(
                expected_roots={
                    "tree_id": "tree:other",
                    "task_cid": "task:dqp-026-demo",
                    "policy_digest": "policy:sha256:demo",
                    "schema_revision": "1",
                }
            )
        )

    manifest = compile_database_context(_request())
    assert_dependency_freshness(
        manifest,
        tree_id="tree:abc123",
        policy_digest="policy:sha256:demo",
        schema_revision=1,
        task_cid="task:dqp-026-demo",
        snapshot_id="snapshot:1",
    )
    with pytest.raises(DatabaseContextInvalidationError):
        assert_dependency_freshness(manifest, tree_id="tree:drifted")

    prior = manifest
    drifted = compile_database_context(_request(tree_id="tree:drifted"))
    delta = compare_and_delta(prior, drifted)
    assert "tree_id" in delta.invalidated_roots
    assert delta.from_manifest_cid != delta.to_manifest_cid


def test_strip_noise_and_fail_closed_overflow() -> None:
    noisy = {
        "task_cid": "task:1",
        "heartbeat_at_ms": 1,
        "nested": {"created_at": "x", "value": 2, "polled_at": "y"},
        "items": [{"updated_at": "z", "id": "a"}],
    }
    clean = strip_noise(noisy)
    assert clean == {
        "items": [{"id": "a"}],
        "nested": {"value": 2},
        "task_cid": "task:1",
    }

    with pytest.raises(DatabaseContextBoundsError):
        compile_database_context(
            _request(
                impacted_symbols=tuple(
                    {"symbol": f"s{i}", "path": f"p/{i}.py"} for i in range(30)
                ),
                decisions=(),
                evidence=(),
                worktree_delta=(),
                budget=_budget(
                    max_rows=8, page_size=50, overflow_behavior="fail_closed"
                ),
            )
        )


def test_capsule_contains_required_semantic_sections() -> None:
    manifest = compile_database_context(_request())
    kinds = {item.kind for item in manifest.members}
    for kind in (
        ContextMemberKind.TASK,
        ContextMemberKind.UNMET_DEPENDENCY,
        ContextMemberKind.LATEST_FAILURE,
        ContextMemberKind.WORKTREE_DELTA,
        ContextMemberKind.IMPACTED_SYMBOL,
        ContextMemberKind.OPEN_OBLIGATION,
        ContextMemberKind.DECISION,
        ContextMemberKind.EVIDENCE,
        ContextMemberKind.VALIDATION,
    ):
        assert kind in kinds

    assert manifest.task_cid == "task:dqp-026-demo"
    assert manifest.repository_id == "repo:demo"
    assert manifest.tree_id == "tree:abc123"
    assert manifest.schema_revision == 1
    assert manifest.policy_digest == "policy:sha256:demo"
    assert manifest.row_count == len(manifest.members)
    assert manifest.byte_size > 0
    assert manifest.token_estimate == estimate_tokens(manifest.byte_size)
    assert manifest.semantic_roots["task_cid"] == manifest.task_cid
    assert manifest.semantic_roots["tree_id"] == manifest.tree_id

    # Project into ContextCompiler boundary without inventing authority.
    projection = project_to_context_compiler_inputs(manifest)
    assert projection["manifest_cid"] == manifest.manifest_cid
    assert projection["goal"]["task_cid"] == manifest.task_cid
    assert projection["authority"]["policy_digest"] == manifest.policy_digest
    assert projection["acceptance"]["validation_commands"]
    assert projection["evidence"]
    assert projection["frontier"]["interface"] == LLM_CONTEXT_FRONTIER_INTERFACE


def test_member_kind_coercion_and_frontier_round_trip() -> None:
    assert ContextMemberKind.coerce("symbols") is ContextMemberKind.IMPACTED_SYMBOL
    assert ContextMemberKind.coerce("validation_command") is ContextMemberKind.VALIDATION
    assert FrontierKind.coerce("overflow") is FrontierKind.BUDGET_OVERFLOW
    assert FrontierDisposition.coerce("omitted_budget") is FrontierDisposition.OMITTED_BUDGET

    frontier = LLMContextFrontier(
        entries=(
            {
                "frontier_id": "frontier:a",
                "kind": "unresolved_symbol",
                "disposition": "unresolved",
                "reason": "open",
                "blocks_automatic_repair": True,
            },
            {
                "frontier_id": "frontier:b",
                "kind": "pagination",
                "disposition": "omitted_pagination",
                "reason": "page",
            },
        )
    )
    assert frontier.unresolved_count == 1
    assert frontier.omitted_count == 1
    assert frontier.complete is False
    page = frontier.page_slice(0, page_size=1)
    assert len(page.entries) == 1
    assert page.next_page_token == "page:1"

    restored = LLMContextFrontier.from_dict(frontier.to_dict())
    assert restored.frontier_cid == frontier.frontier_cid


def test_delta_byte_limit_and_identical_noise_only_change() -> None:
    prior = compile_database_context(_request())
    # Only noise changes on request surface.
    current = compile_database_context(
        _request(heartbeat_at_ms=prior.token_estimate + 99_999)
    )
    assert prior.manifest_cid == current.manifest_cid
    delta = compare_and_delta(prior, current)
    assert not delta.added and not delta.changed and not delta.removed

    # Changing a validation command yields a delta that respects max bounds.
    changed = compile_database_context(
        _request(
            validations=(
                "python -m pytest -q test/api/test_agent_supervisor_database_context.py --maxfail=1",
            )
        )
    )
    delta2 = compare_and_delta(prior, changed, max_delta_bytes=prior.byte_size)
    assert delta2.byte_size <= prior.byte_size
    assert delta2.to_manifest_cid == changed.manifest_cid


def test_malformed_request_fail_closed() -> None:
    with pytest.raises(DatabaseContextError):
        DatabaseContextRequest(
            task_cid="task:x",
            repository_id="repo:x",
            tree_id="tree:x",
            schema_revision=1,
            policy_digest="policy:x",
            task={},  # empty
        )
    with pytest.raises(DatabaseContextError):
        ContextMemberKind.coerce("not-a-kind")
