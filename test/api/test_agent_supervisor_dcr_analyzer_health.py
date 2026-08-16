"""DCR-012 exact parser-accounting tests."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_analyzer_health import (
    ANALYZER_HEALTH_INTERFACE,
    REPOSITORY_INDEX_INTERFACE,
    MandatoryParserState,
    ParserDisposition,
    TrackedPathParserRecord,
    build_analyzer_health,
)


def record(path: str, disposition: ParserDisposition, **kwargs: str) -> TrackedPathParserRecord:
    defaults = {"source_digest": f"sha256:{path}"}
    if disposition is ParserDisposition.PARSED:
        defaults["parser_id"] = "python-ast@1"
    else:
        defaults["reason_code"] = "typed-reason"
    if disposition is ParserDisposition.EXCLUDED_BY_REVIEWED_POLICY:
        defaults["reviewed_policy_id"] = "policy:reviewed@1"
    defaults.update(kwargs)
    return TrackedPathParserRecord(path, disposition, **defaults)


def test_all_tracked_paths_have_one_stable_content_addressed_disposition() -> None:
    rows = (
        record("a.py", ParserDisposition.PARSED),
        record("b.txt", ParserDisposition.EXCLUDED_BY_REVIEWED_POLICY),
    )
    first = build_analyzer_health(
        forest_identity="forest:current",
        tracked_paths=("b.txt", "a.py"),
        records=rows,
        mandatory_parser_states={"python-ast@1": MandatoryParserState.AVAILABLE},
    )
    second = build_analyzer_health(
        forest_identity="forest:current",
        tracked_paths=("a.py", "b.txt"),
        records=tuple(reversed(rows)),
        mandatory_parser_states={"python-ast@1": "available"},
    )
    assert ANALYZER_HEALTH_INTERFACE == "AnalyzerHealth@1"
    assert REPOSITORY_INDEX_INTERFACE == "RepositoryIndex@1"
    assert first.receipt_id == second.receipt_id
    assert first.safe_for_completion
    assert first.to_dict()["completion_authoritative"] is False


def test_missing_or_duplicate_path_accounting_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="every tracked path"):
        build_analyzer_health(
            forest_identity="forest:current",
            tracked_paths=("a.py", "b.py"),
            records=(record("a.py", ParserDisposition.PARSED),),
            mandatory_parser_states={},
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_analyzer_health(
            forest_identity="forest:current",
            tracked_paths=("a.py",),
            records=(
                record("a.py", ParserDisposition.PARSED),
                record("a.py", ParserDisposition.PARSER_FAILURE),
            ),
            mandatory_parser_states={},
        )


@pytest.mark.parametrize(
    ("disposition", "state", "baseline", "blocker"),
    (
        (
            ParserDisposition.PARSER_FAILURE,
            MandatoryParserState.AVAILABLE,
            0,
            "parser_failure:a.py",
        ),
        (
            ParserDisposition.PARSED,
            MandatoryParserState.MISSING,
            0,
            "mandatory_parser_missing:python-ast@1",
        ),
        (
            ParserDisposition.PARSED,
            MandatoryParserState.AMBIGUOUS,
            0,
            "mandatory_parser_ambiguous:python-ast@1",
        ),
        (ParserDisposition.PARSED, MandatoryParserState.AVAILABLE, 22, "stale_22_failure_baseline"),
        (
            ParserDisposition.TYPED_UNSUPPORTED,
            MandatoryParserState.AVAILABLE,
            0,
            "typed_unsupported:a.py",
        ),
    ),
)
def test_all_parser_or_baseline_blockers_are_typed_and_fail_closed(
    disposition: ParserDisposition, state: MandatoryParserState, baseline: int, blocker: str
) -> None:
    health = build_analyzer_health(
        forest_identity="forest:current",
        tracked_paths=("a.py",),
        records=(record("a.py", disposition),),
        mandatory_parser_states={"python-ast@1": state},
        stored_baseline_parser_failures=baseline,
    )
    assert not health.safe_for_completion
    assert blocker in health.blockers


def test_reviewed_exclusion_requires_policy_and_is_not_a_failure_cap() -> None:
    with pytest.raises(ValueError, match="reviewed_policy_id"):
        TrackedPathParserRecord(
            "a.generated",
            ParserDisposition.EXCLUDED_BY_REVIEWED_POLICY,
            reason_code="generated",
        )
    health = build_analyzer_health(
        forest_identity="forest:current",
        tracked_paths=("a.generated",),
        records=(record("a.generated", ParserDisposition.EXCLUDED_BY_REVIEWED_POLICY),),
        mandatory_parser_states={},
    )
    assert health.safe_for_completion
