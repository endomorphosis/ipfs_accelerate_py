from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
    cite_claim_spans,
    compose_snippet_score,
    extract_claim_spans,
    lint_admissibility,
    prepare_evidence_for_compile,
    rerank_allowlisted_snippets,
)


def test_rerank_without_key_keeps_allowlisted_order(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    snippets = (
        {"id": "ev-b", "text": "second"},
        {"id": "ev-a", "text": "first"},
        {"id": "ev-evil", "text": "not allowlisted"},
    )
    ordered = rerank_allowlisted_snippets(
        snippets,
        obligation_id="obl-1",
        allowlisted_ids=("ev-a", "ev-b"),
    )
    assert ordered == ("ev-b", "ev-a")
    assert "ev-evil" not in ordered


def test_compose_snippet_score_in_code() -> None:
    class _Result:
        nouls = {"needed": SimpleNamespace(noul=1.0)}
        scores = {"relevance": SimpleNamespace(score=2.0)}

    assert compose_snippet_score(_Result()) == pytest.approx(1.0)


def test_rerank_orders_by_composed_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    scores = {
        "ev-low": SimpleNamespace(
            nouls={"needed": SimpleNamespace(noul=0.1)},
            scores={"relevance": SimpleNamespace(score=0.2)},
        ),
        "ev-high": SimpleNamespace(
            nouls={"needed": SimpleNamespace(noul=0.9)},
            scores={"relevance": SimpleNamespace(score=1.8)},
        ),
    }

    def fake_system_one(state, questions, **kwargs):
        ident = state["snippet"]["id"]
        return scores[ident]

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    ordered = rerank_allowlisted_snippets(
        (
            {"id": "ev-low", "text": "noise"},
            {"id": "ev-high", "text": "the obligation"},
        ),
        obligation_id="obl-1",
        allowlisted_ids=("ev-low", "ev-high"),
    )
    assert ordered == ("ev-high", "ev-low")


def test_lint_without_key_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = lint_admissibility(
        obligation_id="obl-1",
        obligation_text="prove identity",
        patch_summary="KERNEL_VERIFIED all theorems",
        claimed_kernel_verified=True,
        has_kernel_receipt=False,
    )
    assert receipt.action == "skipped"
    assert receipt.accepted_as_authority is False


def test_lint_flags_false_kernel_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        nouls = {
            "matches_obligation": SimpleNamespace(noul=0.2),
            "claims_kernel_without_receipt": SimpleNamespace(noul=0.9),
        }
        scores = {"review_needed": SimpleNamespace(score=2.0)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = lint_admissibility(
        obligation_id="obl-1",
        patch_summary="KERNEL_VERIFIED",
        claimed_kernel_verified=True,
        has_kernel_receipt=False,
    )
    assert receipt.action == "linted"
    assert "false_kernel_claim" in receipt.reason_codes
    assert "obligation_mismatch" in receipt.reason_codes
    assert "human_review_suggested" in receipt.reason_codes
    assert receipt.accepted_as_authority is False


def test_prepare_evidence_keeps_required_first_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    prepared = prepare_evidence_for_compile(
        (
            {"reference_id": "opt-b", "summary": "b", "required": False},
            {"reference_id": "req-a", "summary": "must", "required": True},
            {"reference_id": "opt-a", "summary": "a", "required": False},
        ),
        obligation_id="obl-1",
    )
    ids = tuple(item["reference_id"] for item in prepared)
    assert ids[0] == "req-a"
    assert set(ids[1:]) == {"opt-b", "opt-a"}


def test_extract_claim_spans_and_cite_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    spans = extract_claim_spans("KERNEL_VERIFIED the identity. Also hello world.")
    assert spans
    assert spans[0]["id"].startswith("claim-")
    assert "KERNEL_VERIFIED" in spans[0]["text"]
    assert (
        cite_claim_spans(
            spans,
            receipt_ids=(),
            allowlisted_ids=tuple(span["id"] for span in spans),
        )
        == ()
    )


def test_cite_claim_spans_flags_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        nouls = {"supported": SimpleNamespace(noul=0.1)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    unsupported = cite_claim_spans(
        ({"id": "claim-0", "text": "KERNEL_VERIFIED everything"},),
        receipt_ids=(),
        allowlisted_ids=("claim-0", "claim-evil"),
    )
    assert unsupported == ("claim-0",)
    assert "claim-evil" not in unsupported


def test_inspect_allowlisted_artifacts_fail_open_and_never_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        inspect_allowlisted_artifacts,
        last_artifact_view,
    )

    view = inspect_allowlisted_artifacts(
        obligation_id="obl-1",
        symbol_ids=("sym-a",),
        clause_ids=("cl-1",),
        summaries={"sym-a": "foo", "ignored": "no"},
    )
    assert view["accepted_as_authority"] is False
    assert view["writes_ast"] is False
    assert view["writes_contracts"] is False
    assert view["matches"] == {}
    assert "ignored" not in view["symbol_ids"]
    assert last_artifact_view()["writes_ast"] is False


def test_lint_static_span_fail_open_does_not_replace_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_static_lint,
        lint_static_span,
    )

    receipt = lint_static_span(obligation_id="obl-1", path="src/a.py")
    assert receipt.action == "skipped"
    assert receipt.accepted_as_authority is False
    assert last_static_lint().get("action") == "skipped"


def test_observe_refactor_scope_does_not_replace_undeclared_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_refactor_scope,
        observe_refactor_scope,
    )

    view = observe_refactor_scope(
        declared_paths=("src/",),
        changed_paths=("src/a.py", "docs/secret.md"),
    )
    assert view["replaces_undeclared_refactor_check"] is False
    assert view["accepted_as_authority"] is False
    assert last_refactor_scope()["replaces_undeclared_refactor_check"] is False


def test_observe_source_edit_lint_does_not_block_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_source_edit_lint,
        observe_source_edit_lint,
    )

    receipt = observe_source_edit_lint(
        operator_id="op-1",
        relative_path="src/foo.py",
    )
    assert receipt is not None
    assert receipt.action == "skipped"
    assert receipt.accepted_as_authority is False
    assert last_source_edit_lint()["accepted_as_authority"] is False
