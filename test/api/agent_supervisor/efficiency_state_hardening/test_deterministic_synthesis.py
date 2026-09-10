"""Fixtures for ASEH-054's closed exact deterministic transform grammar."""

from __future__ import annotations

import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
    DeterministicDoctorSynthesizer,
    materialize_exact_transform,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    ExactTransformKind,
    ExactTransformRejectionReason,
    ExactTransformRequest,
    exact_transform_specs,
)


def _request(
    kind: ExactTransformKind,
    source: str,
    *,
    parameters: dict[str, str] | None = None,
    field_mapping: dict[str, str] | None = None,
    path: str = "pkg/fixture.py",
) -> ExactTransformRequest:
    return ExactTransformRequest(
        kind=kind,
        path=path,
        span_text=source,
        before_hash="sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
        allowed_paths=(path,),
        parameters=parameters or {},
        field_mapping=field_mapping or {},
    )


@pytest.mark.parametrize(
    ("kind", "source", "parameters", "field_mapping", "expected"),
    (
        (
            ExactTransformKind.TYPED_ERROR,
            'raise LegacyError("bad")',
            {"old_error": "LegacyError", "new_error": "TypedError"},
            {},
            'raise TypedError("bad")',
        ),
        (
            ExactTransformKind.SCHEMA,
            '{"name":"Ada","user":"u1"}',
            {},
            {"user": "user_id"},
            '{"name":"Ada","user_id":"u1"}',
        ),
        (
            ExactTransformKind.IMPORT,
            "import os\n",
            {"module": "pkg.errors", "name": "TypedError"},
            {},
            "import os\nfrom pkg.errors import TypedError\n",
        ),
        (
            ExactTransformKind.ADAPTER,
            "payload",
            {"adapter": "PayloadAdapter", "expression": "payload"},
            {},
            "PayloadAdapter(payload)",
        ),
        (
            ExactTransformKind.RENAME,
            "old_name",
            {"old_name": "old_name", "new_name": "new_name"},
            {},
            "new_name",
        ),
        (
            ExactTransformKind.VECTOR,
            '["old","keep"]',
            {},
            {"old": "new"},
            '["new","keep"]',
        ),
        (
            ExactTransformKind.WRAPPER,
            "payload",
            {"wrapper": "freeze", "expression": "payload"},
            {},
            "freeze(payload)",
        ),
        (
            ExactTransformKind.FORMAT,
            "answer = 42  ",
            {},
            {},
            "answer = 42\n",
        ),
    ),
)
def test_each_allowlisted_transform_is_deterministic_nonempty_and_idempotent(
    kind: ExactTransformKind,
    source: str,
    parameters: dict[str, str],
    field_mapping: dict[str, str],
    expected: str,
) -> None:
    request = _request(kind, source, parameters=parameters, field_mapping=field_mapping)
    first = materialize_exact_transform(request)
    second = DeterministicDoctorSynthesizer.synthesize_exact(request)

    assert first.admitted and second.admitted
    assert first.replacement == second.replacement == expected
    assert first.replacement
    assert first.before_hash == request.before_hash
    assert first.after_hash == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    assert first.idempotent is True
    assert first.preconditions and first.postconditions

    repeated = _request(
        kind,
        first.replacement,
        parameters=parameters,
        field_mapping=field_mapping,
    )
    replay = materialize_exact_transform(repeated)
    assert replay.admitted
    assert replay.replacement == first.replacement
    assert replay.changed is False


def test_allowlist_has_all_declared_families_with_explicit_contracts() -> None:
    specs = exact_transform_specs()
    assert tuple(spec.kind for spec in specs) == tuple(ExactTransformKind)
    assert all(spec.preconditions and spec.postconditions for spec in specs)


def test_out_of_scope_path_and_stale_span_reject_at_construction() -> None:
    with pytest.raises(Exception, match="path_out_of_scope"):
        ExactTransformRequest(
            kind=ExactTransformKind.RENAME,
            path="pkg/outside.py",
            span_text="old",
            before_hash="sha256:" + hashlib.sha256(b"old").hexdigest(),
            allowed_paths=("pkg/in_scope.py",),
            parameters={"old_name": "old", "new_name": "new"},
        )
    with pytest.raises(Exception, match="stale_span"):
        ExactTransformRequest(
            kind=ExactTransformKind.RENAME,
            path="pkg/fixture.py",
            span_text="old",
            before_hash="sha256:" + hashlib.sha256(b"other").hexdigest(),
            allowed_paths=("pkg/fixture.py",),
            parameters={"old_name": "old", "new_name": "new"},
        )


@pytest.mark.parametrize(
    ("kind", "source", "parameters", "field_mapping", "reason"),
    (
        (
            ExactTransformKind.TYPED_ERROR,
            "raise LegacyError(*args)",
            {"old_error": "LegacyError", "new_error": "TypedError"},
            {},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.SCHEMA,
            '{"name":"Ada"}',
            {},
            {"missing": "present"},
            ExactTransformRejectionReason.NON_TOTAL_MAPPING,
        ),
        (
            ExactTransformKind.IMPORT,
            "value = 1",
            {"module": "pkg.errors", "name": "TypedError"},
            {},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.ADAPTER,
            "make_payload()",
            {"adapter": "PayloadAdapter", "expression": "payload"},
            {},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.RENAME,
            "old.name",
            {"old_name": "old", "new_name": "new"},
            {},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.VECTOR,
            '["old","old"]',
            {},
            {"old": "new"},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.WRAPPER,
            "payload + other",
            {"wrapper": "freeze", "expression": "payload"},
            {},
            ExactTransformRejectionReason.UNSUPPORTED_GRAMMAR,
        ),
        (
            ExactTransformKind.FORMAT,
            "",
            {},
            {},
            ExactTransformRejectionReason.EMPTY_INPUT,
        ),
    ),
)
def test_outside_each_closed_grammar_returns_typed_empty_rejection(
    kind: ExactTransformKind,
    source: str,
    parameters: dict[str, str],
    field_mapping: dict[str, str],
    reason: ExactTransformRejectionReason,
) -> None:
    result = materialize_exact_transform(
        _request(kind, source, parameters=parameters, field_mapping=field_mapping)
    )
    assert not result.admitted
    assert result.replacement == ""
    assert result.rejection_reason is reason
