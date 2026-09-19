from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
    assemble_date_parts,
    check_claim_citation,
    cite_claim_spans,
    compose_pairwise_order,
    compose_snippet_score,
    extract_claim_spans,
    extract_clause_date,
    find_supporting_line,
    last_claim_citation,
    last_clause_date,
    last_extracted_span,
    last_line_stitch,
    last_supporting_line,
    lint_admissibility,
    observe_claim_citation,
    observe_clause_date,
    pick_extracted_span,
    stitch_hard_wrapped_lines,
    observe_merge_conflict_paths,
    pairwise_questions,
    prepare_evidence_for_compile,
    producer_consumer_questions,
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

    calls: list[int] = []

    seen_questions: list[dict] = []

    def fake_system_one(state, questions, **kwargs):
        calls.append(1)
        seen_questions.append(questions)
        assert "needed_ev-low" in questions
        assert "needed_ev-high" in questions
        return SimpleNamespace(
            nouls={
                "needed_ev-low": scores["ev-low"].nouls["needed"],
                "needed_ev-high": scores["ev-high"].nouls["needed"],
            },
            scores={
                "relevance_ev-low": scores["ev-low"].scores["relevance"],
                "relevance_ev-high": scores["ev-high"].scores["relevance"],
            },
        )

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
    assert calls == [1]
    assert "better_ev-low_than_ev-high" in seen_questions[0]
    assert ordered == ("ev-high", "ev-low")


def test_compose_pairwise_order_allowlisted_only() -> None:
    class _N:
        def __init__(self, noul: float) -> None:
            self.noul = noul

    ordered = compose_pairwise_order(
        ("a", "b", "c"),
        nouls={"better_a_than_b": _N(0.9), "better_a_than_c": _N(0.8), "better_b_than_c": _N(0.2)},
        independent={"a": 0.1, "b": 0.9, "c": 0.5},
    )
    assert ordered[0] == "a"
    assert "invented" not in ordered
    assert set(ordered) == {"a", "b", "c"}
    questions = pairwise_questions(
        ("a", "b"),
        left_path="snippets.{id}.text",
        right_path="snippets.{id}.text",
    )
    assert "better_a_than_b" in questions
    assert "better_a_than_invented" not in questions


def test_producer_and_citation_nouls_have_structured_criteria() -> None:
    producer = producer_consumer_questions()["producer_matches"].to_dict()
    assert producer["criteria"]["true"]["what"]
    assert producer["criteria"]["false"]["examples"]
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        citation_questions,
    )

    citation = citation_questions()["supported"].to_dict()
    assert citation["criteria"]["true"]["what"]
    assert "KERNEL_VERIFIED" in citation["criteria"]["false"]["examples"][0]
    field = producer["instructions"]["field"]
    assert field["name"] == "producer"
    assert field["type"] == "string"


def test_observe_merge_conflict_paths_does_not_fence_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    view = observe_merge_conflict_paths(
        declared_paths=("src/",),
        conflict_paths=("src/a.py", "docs/secret.md"),
    )
    assert view["replaces_consumer_fence"] is False
    assert view["writes_merge"] is False
    assert view["accepted_as_authority"] is False
    assert view["replaces_undeclared_refactor_check"] is False


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

    calls: list[int] = []

    class _Result:
        nouls = {
            "supported_claim-0": SimpleNamespace(noul=0.1),
            "supported_claim-1": SimpleNamespace(noul=0.9),
        }

    def fake_system_one(_state, questions, **_kwargs):
        calls.append(1)
        assert "supported_claim-0" in questions
        assert "supported_claim-1" in questions
        return _Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    unsupported = cite_claim_spans(
        (
            {"id": "claim-0", "text": "KERNEL_VERIFIED everything"},
            {"id": "claim-1", "text": "KERNEL_VERIFIED with a receipt"},
        ),
        receipt_ids=("receipt-1",),
        allowlisted_ids=("claim-0", "claim-1", "claim-evil"),
    )
    assert calls == [1]
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


def test_inspect_allowlisted_artifacts_one_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    calls: list[int] = []

    class _Result:
        nouls = {
            "matches_sym-a": SimpleNamespace(noul=0.8),
            "matches_cl-1": SimpleNamespace(noul=0.2),
        }

    def fake_system_one(_state, questions, **_kwargs):
        calls.append(1)
        assert "matches_sym-a" in questions
        assert "matches_cl-1" in questions
        return _Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        inspect_allowlisted_artifacts,
    )

    view = inspect_allowlisted_artifacts(
        obligation_id="obl-1",
        symbol_ids=("sym-a",),
        clause_ids=("cl-1",),
        summaries={"sym-a": "foo", "cl-1": "bar"},
    )
    assert calls == [1]
    assert view["matches"]["sym-a"] == pytest.approx(0.8)
    assert view["writes_ast"] is False


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


def test_rank_allowlisted_artifacts_fail_open_keeps_existing_ids(
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
        last_artifact_rank,
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(
        ("node-b", "node-a", "node-evil-not-used"),
        obligation_id="obl-1",
    )
    assert ranked == ("node-b", "node-a", "node-evil-not-used")
    view = last_artifact_rank()
    assert view["invents_ids"] is False
    assert view["replaces_undeclared_refactor_check"] is False
    assert view["replaces_boundary_cost_ranking"] is False
    assert view["accepted_as_authority"] is False
    assert view.get("suggested") == ""
    assert view.get("admits_candidate") is False
    assert "node-invented" not in view["ranked_ids"]


def test_rank_allowlisted_artifacts_orders_existing_ids_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.inspect_allowlisted_artifacts",
        lambda **_kwargs: {
            "matches": {
                "node-low": 0.1,
                "node-high": 0.9,
                "node-invented": 1.0,
            }
        },
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_rank,
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(
        ("node-low", "node-high"),
        obligation_id="obl-1",
    )
    assert ranked == ("node-high", "node-low")
    assert "node-invented" not in ranked
    view = last_artifact_rank()
    assert view["invents_ids"] is False
    assert view["replaces_boundary_cost_ranking"] is False
    assert "node-invented" not in view["matches"]


def test_observe_parser_failure_clusters_never_excludes_protected(
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
        last_parser_triage,
        observe_parser_failure_clusters,
    )

    view = observe_parser_failure_clusters(
        (
            {
                "cluster_id": "cl-mcp",
                "path_family": "mcp/runtime",
                "protected": True,
                "protected_member_count": 2,
            },
        )
    )
    assert view["weakens_thresholds"] is False
    assert view["excludes_mcp_surface"] is False
    assert view["protected_contract_surface"] is True
    assert view["accepted_as_authority"] is False
    assert last_parser_triage()["excludes_mcp_surface"] is False


def test_observe_parser_failure_clusters_one_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    calls: list[int] = []

    class _Result:
        nouls = {
            "fixture_or_generated_cl-a": SimpleNamespace(noul=0.7),
            "fixture_or_generated_cl-b": SimpleNamespace(noul=0.1),
        }

    def fake_system_one(_state, questions, **_kwargs):
        calls.append(1)
        assert "fixture_or_generated_cl-a" in questions
        assert "fixture_or_generated_cl-b" in questions
        return _Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        observe_parser_failure_clusters,
    )

    view = observe_parser_failure_clusters(
        (
            {"cluster_id": "cl-a", "path_family": "fixtures/", "protected": False},
            {"cluster_id": "cl-b", "path_family": "src/", "protected": False},
        )
    )
    assert calls == [1]
    assert view["fixture_like"]["cl-a"] == pytest.approx(0.7)
    assert view["excludes_mcp_surface"] is False
    assert view["weakens_thresholds"] is False


def test_rank_allowlisted_artifacts_drops_ineligible_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.inspect_allowlisted_artifacts",
        lambda **_kwargs: {"matches": {"elig-b": 0.9, "elig-a": 0.1, "ineligible": 1.0}},
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(("elig-a", "elig-b"), obligation_id="repair")
    assert ranked == ("elig-b", "elig-a")
    assert "ineligible" not in ranked


def test_lint_producer_consumer_fail_open_does_not_rewrite(
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
        last_producer_consumer,
        lint_producer_consumer,
    )

    view = lint_producer_consumer(
        artifact_id="blob:1",
        claimed_producer="regex-heuristic",
        admitted_producer="typescript-compiler-api",
        claimed_producer_version="v0",
        admitted_producer_version="typescript-ast-extractor@2",
        claimed_consumer_id="other",
        admitted_consumer_id="owner",
    )
    assert view["accepted_as_authority"] is False
    assert view["rewrites_producer_id"] is False
    assert view["replaces_protocol_error"] is False
    assert view["replaces_consumer_fence"] is False
    assert view["claimed_producer"] == "regex-heuristic"
    assert view["admitted_producer"] == "typescript-compiler-api"
    assert view["claimed_consumer_id"] == "other"
    assert view["admitted_consumer_id"] == "owner"
    assert last_producer_consumer()["rewrites_producer_id"] is False


def test_lint_producer_consumer_flags_mismatch_without_rewriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        nouls = {
            "producer_matches": SimpleNamespace(noul=0.1),
            "producer_version_matches": SimpleNamespace(noul=0.2),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        lint_producer_consumer,
    )

    view = lint_producer_consumer(
        artifact_id="blob:1",
        claimed_producer="regex-heuristic",
        admitted_producer="typescript-compiler-api",
        claimed_producer_version="v0",
        admitted_producer_version="typescript-ast-extractor@2",
    )
    assert view["producer_matches"] == pytest.approx(0.1)
    assert "producer_mismatch" in view["reason_codes"]
    assert "producer_version_mismatch" in view["reason_codes"]
    assert view["rewrites_producer_id"] is False
    assert view["claimed_producer"] == "regex-heuristic"
    assert view["admitted_producer"] == "typescript-compiler-api"


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


def _clear_typesafe_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_claim_citation_without_key_skips_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: called.append(1),
    )
    view = check_claim_citation(
        "All humans are mortal. Socrates is a human.",
        "All humans are mortal",
        quote="All humans are mortal.",
    )
    assert called == []
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert view["verdict"] == ""
    assert view["status"] == "found"
    assert last_claim_citation()["accepted_as_authority"] is False


def test_claim_citation_missing_quote_fabricated_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: called.append(1) or (_ for _ in ()).throw(
            AssertionError("missing quote must not call system_one")
        ),
    )
    view = check_claim_citation(
        "All humans are mortal.",
        "Validators must reject future iat claims.",
        quote="iat claims in the future MUST be rejected.",
    )
    assert called == []
    assert view["verdict"] == "fabricated"
    assert view["status"] == "missing"
    assert view["auto"] is True
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert last_claim_citation()["verdict"] == "fabricated"


def test_claim_citation_supports_high_conf_still_not_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "relation": SimpleNamespace(choice="supports", confidence=0.93),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = check_claim_citation(
        "All humans are mortal. Socrates is a human.",
        "All humans are mortal",
        quote="All humans are mortal.",
    )
    assert view["verdict"] == "supports"
    assert view["verdict"] != "verified"
    assert view["auto"] is True
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False


def test_observe_claim_citation_does_not_admit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    view = observe_claim_citation(
        "The agency shall provide notice.",
        "obligation(agency, provide_notice)",
    )
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert view["verdict"] != "verified"


def test_pick_extracted_span_fail_open_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_typesafe_keys(monkeypatch)
    view = pick_extracted_span(
        "write src/a.py",
        ("src/a.py", "src/b.py"),
    )
    assert view["pick"] == ""
    assert view["candidates"] == ["src/a.py", "src/b.py"]
    assert view["invents_span"] is False
    assert view["accepted_as_authority"] is False
    assert last_extracted_span()["pick"] == ""


def test_pick_extracted_span_empty_candidates_skips_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: called.append(1),
    )
    view = pick_extracted_span("write src/a.py", ())
    assert called == []
    assert view["pick"] == ""
    assert view["reason_codes"] == ["no_candidates"]


def test_pick_extracted_span_copies_allowlisted_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"pick": SimpleNamespace(choice="src/a.py", confidence=0.97)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = pick_extracted_span("write src/a.py", ("src/a.py", "src/b.py"))
    assert view["pick"] == "src/a.py"
    assert view["invents_span"] is False
    assert view["accepted_as_authority"] is False


def test_pick_extracted_span_unknown_choice_becomes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"pick": SimpleNamespace(choice="/etc/passwd", confidence=0.99)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = pick_extracted_span("write src/a.py", ("src/a.py", "src/b.py"))
    assert view["pick"] == "none"
    assert view["invents_span"] is False
    assert "/etc/passwd" not in view["candidates"]


def test_merge_conflict_pick_does_not_rewrite_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    view = observe_merge_conflict_paths(
        declared_paths=("src/",),
        conflict_paths=("src/a.py", "docs/secret.md"),
    )
    assert view["conflict_paths"] == ["src/a.py", "docs/secret.md"]
    assert view["typesafe_pick"]["invents_span"] is False
    assert view["typesafe_pick"]["pick"] == ""
    assert view["writes_merge"] is False


_TODAY = date(2026, 7, 30)


def _date_parts(**overrides: dict) -> dict:
    base = {
        "mode": {"choice": "none", "confidence": 0.9},
        "month": {"choice": "none", "confidence": 0.9},
        "day": {"choice": "none", "confidence": 0.9},
        "year": {"choice": "none", "confidence": 0.9},
        "day_anchor": {"choice": "none", "confidence": 0.9},
        "weekday": {"choice": "none", "confidence": 0.9},
        "week_offset": {"choice": "none", "confidence": 0.9},
    }
    base.update(overrides)
    return base


def test_assemble_absolute_date_uses_stated_year() -> None:
    view = assemble_date_parts(
        _date_parts(
            mode={"choice": "absolute", "confidence": 0.97},
            month={"choice": "January", "confidence": 0.99},
            day={"choice": "1", "confidence": 0.99},
            year={"choice": "2025", "confidence": 0.97},
        ),
        today=_TODAY,
    )
    assert view["date"] == "2025-01-01"
    assert view["incomplete"] is False
    assert view["needs_review"] is False


def test_assemble_absolute_date_infers_year_in_code() -> None:
    view = assemble_date_parts(
        _date_parts(
            mode={"choice": "absolute", "confidence": 0.95},
            month={"choice": "August", "confidence": 0.95},
            day={"choice": "14", "confidence": 0.95},
            year={"choice": "none", "confidence": 0.9},
        ),
        today=_TODAY,
    )
    assert view["date"] == "2026-08-14"


def test_assemble_relative_tomorrow_in_code() -> None:
    view = assemble_date_parts(
        _date_parts(
            mode={"choice": "relative", "confidence": 0.94},
            day_anchor={"choice": "tomorrow", "confidence": 0.94},
        ),
        today=_TODAY,
    )
    assert view["date"] == "2026-07-31"


def test_assemble_next_thursday_in_code() -> None:
    view = assemble_date_parts(
        _date_parts(
            mode={"choice": "relative", "confidence": 0.92},
            day_anchor={"choice": "weekday", "confidence": 0.92},
            weekday={"choice": "Thursday", "confidence": 0.92},
            week_offset={"choice": "next", "confidence": 0.92},
        ),
        today=_TODAY,
    )
    assert view["date"] == "2026-08-06"


def test_assemble_incomplete_absolute_flags_review() -> None:
    view = assemble_date_parts(
        _date_parts(
            mode={"choice": "absolute", "confidence": 0.46},
            month={"choice": "none", "confidence": 0.46},
            day={"choice": "none", "confidence": 0.4},
        ),
        today=_TODAY,
    )
    assert view["date"] == ""
    assert view["incomplete"] is True
    assert view["needs_review"] is True


def test_extract_clause_date_fail_open_skips_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: called.append(1),
    )
    view = extract_clause_date(
        "This agreement expires December 31, 2027.",
        today=_TODAY,
    )
    assert called == []
    assert view["date"] == ""
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert last_clause_date()["date"] == ""


def test_extract_clause_date_assembles_stubbed_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "mode": SimpleNamespace(choice="absolute", confidence=0.97),
            "month": SimpleNamespace(choice="January", confidence=0.99),
            "day": SimpleNamespace(choice="1", confidence=0.99),
            "year": SimpleNamespace(choice="2025", confidence=0.97),
            "day_anchor": SimpleNamespace(choice="none", confidence=0.9),
            "weekday": SimpleNamespace(choice="none", confidence=0.9),
            "week_offset": SimpleNamespace(choice="none", confidence=0.9),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = extract_clause_date(
        "This agreement is effective January 1, 2025.",
        today=_TODAY,
    )
    assert view["date"] == "2025-01-01"
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert view["incomplete"] is False


def test_observe_clause_date_does_not_admit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    view = observe_clause_date(
        "Please return the signed form by August 14.",
        today=_TODAY,
    )
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert view["date"] == ""


_SOURCE_LINES = (
    "You own Your Content.\n"
    "GitHub may suspend or terminate access.\n"
    "You must be age 13 or older."
)


def test_find_supporting_line_fail_open_skips_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: called.append(1),
    )
    view = find_supporting_line(_SOURCE_LINES, "who owns uploaded code?")
    assert called == []
    assert view["line_id"] == ""
    assert view["invents_ids"] is False
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert last_supporting_line()["line_id"] == ""


def test_find_supporting_line_copies_source_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "where": SimpleNamespace(
                choice="L000",
                confidence=0.95,
                probabilities={"L000": 0.95, "L001": 0.03, "L002": 0.02},
            )
        }
        nouls = {"exists": SimpleNamespace(noul=0.98)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = find_supporting_line(_SOURCE_LINES, "who owns uploaded code?")
    assert view["line_id"] == "L000"
    assert view["line_text"] == "You own Your Content."
    assert view["verdict"] == "answered"
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False
    assert view["invents_ids"] is False
    assert view["ranked"][0]["id"] == "L000"


def test_find_supporting_line_low_exists_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "where": SimpleNamespace(
                choice="L001",
                confidence=0.86,
                probabilities={"L001": 0.86, "L000": 0.14},
            )
        }
        nouls = {"exists": SimpleNamespace(noul=0.14)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = find_supporting_line(_SOURCE_LINES, "must disputes go to arbitration?")
    assert view["verdict"] == "absent"
    assert view["line_id"] == "L001"
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False


def test_find_supporting_line_drops_unknown_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "where": SimpleNamespace(
                choice="L999",
                confidence=0.99,
                probabilities={"L999": 1.0},
            )
        }
        nouls = {"exists": SimpleNamespace(noul=0.9)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    view = find_supporting_line(_SOURCE_LINES, "who owns uploaded code?")
    assert view["line_id"] == ""
    assert view["line_text"] == ""
    assert view["invents_ids"] is False
    assert view["verdict"] == "answered"


def _stub_artifact_rank_and_confirm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    matches: dict[str, float],
    fits: dict[str, float],
    winner: str,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.inspect_allowlisted_artifacts",
        lambda **_kwargs: {"matches": dict(matches), "pair_nouls": {}},
    )

    class _ConfirmResult:
        nouls = {
            f"fits::{ident}": SimpleNamespace(noul=value)
            for ident, value in fits.items()
        }
        choices = {"which": SimpleNamespace(choice=winner, confidence=0.9)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _ConfirmResult(),
    )


def test_rank_confirm_rejects_all_when_fits_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_artifact_rank_and_confirm(
        monkeypatch,
        matches={"cand-a": 0.8, "cand-b": 0.2},
        fits={"cand-a": 0.2, "cand-b": 0.1},
        winner="cand-a",
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_rank,
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(("cand-a", "cand-b"), obligation_id="obl-1")
    assert ranked == ("cand-a", "cand-b")
    view = last_artifact_rank()
    assert view["suggested"] == ""
    assert view["fits"]["cand-a"] == pytest.approx(0.2)
    assert view["invents_ids"] is False
    assert view["admits_candidate"] is False


def test_rank_confirm_suggests_shortlist_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_artifact_rank_and_confirm(
        monkeypatch,
        matches={"cand-edit": 0.7, "cand-author": 0.3},
        fits={"cand-edit": 0.73, "cand-author": 0.38},
        winner="cand-author",
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_rank,
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(
        ("cand-edit", "cand-author"), obligation_id="obl-1"
    )
    assert ranked == ("cand-edit", "cand-author")
    view = last_artifact_rank()
    assert view["suggested"] == "cand-author"
    assert view["suggested"] in ranked
    assert view["accepted_as_authority"] is False
    assert view["admits_candidate"] is False


def test_rank_confirm_drops_unknown_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_artifact_rank_and_confirm(
        monkeypatch,
        matches={"cand-a": 0.9, "cand-b": 0.1},
        fits={"cand-a": 0.8, "cand-b": 0.4},
        winner="cand-invented",
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        last_artifact_rank,
        rank_allowlisted_artifacts,
    )

    ranked = rank_allowlisted_artifacts(("cand-a", "cand-b"), obligation_id="obl-1")
    view = last_artifact_rank()
    assert ranked == ("cand-a", "cand-b")
    assert view["suggested"] == ""
    assert "cand-invented" not in view["shortlist"]


def test_stitch_hard_wrapped_lines_fail_open_keeps_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_typesafe_keys(monkeypatch)
    original = "All humans are\nmortal."
    view = stitch_hard_wrapped_lines(original)
    assert view["text"] == original
    assert view["generates_text"] is False
    assert view["generates_markup"] is False
    assert view["original"] == original
    assert view["blocks"] == []
    assert last_line_stitch()["accepted_as_authority"] is False


def test_stitch_merge_uses_only_input_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        nouls = {"L001": SimpleNamespace(noul=0.8)}
        choices = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    original = "All humans are\nmortal."
    view = stitch_hard_wrapped_lines(original)
    assert view["generates_text"] is False
    assert view["generates_markup"] is False
    assert view["text"] == "All humans are mortal."
    compact = view["text"].replace(" ", "")
    source = original.replace("\n", "").replace(" ", "")
    assert compact == source


def test_stitch_classify_labels_without_generating_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _JoinResult:
        nouls = {"L001": SimpleNamespace(noul=0.1)}
        choices = {}

    class _ClassifyResult:
        nouls = {
            "step_B000": SimpleNamespace(noul=0.1),
            "step_B001": SimpleNamespace(noul=0.1),
        }
        choices = {
            "type_B000": SimpleNamespace(choice="heading", confidence=0.99),
            "hlevel_B000": SimpleNamespace(choice="title", confidence=0.9),
            "callout_B000": SimpleNamespace(choice="note", confidence=0.2),
            "type_B001": SimpleNamespace(choice="paragraph", confidence=0.95),
            "hlevel_B001": SimpleNamespace(choice="section", confidence=0.2),
            "callout_B001": SimpleNamespace(choice="note", confidence=0.2),
        }

    def _system_one(_state, questions, **_kwargs):
        if any(str(key).startswith("type_") for key in questions):
            return _ClassifyResult()
        return _JoinResult()

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        _system_one,
    )
    original = "Migration memo\nHi everyone, the cutover is next week."
    view = stitch_hard_wrapped_lines(original)
    assert view["generates_text"] is False
    assert view["generates_markup"] is False
    assert "#" not in view["text"]
    texts = [block["text"] for block in view["blocks"]]
    compact_out = "".join(texts).replace(" ", "")
    compact_in = original.replace("\n", "").replace(" ", "")
    assert compact_out == compact_in
    assert view["blocks"][0]["type"] == "heading"
    assert view["blocks"][0]["hlevel"] == "title"
    assert view["blocks"][1]["type"] == "paragraph"
    assert view["accepted_as_authority"] is False
    assert view["kernel_verified"] is False


def test_stitch_classify_unknown_type_falls_back_to_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _JoinResult:
        nouls = {"L001": SimpleNamespace(noul=0.1)}
        choices = {}

    class _ClassifyResult:
        nouls = {
            "step_B000": SimpleNamespace(noul=0.0),
            "step_B001": SimpleNamespace(noul=0.0),
        }
        choices = {
            "type_B000": SimpleNamespace(choice="slideshow", confidence=0.99),
            "type_B001": SimpleNamespace(choice="paragraph", confidence=0.9),
        }

    def _system_one(_state, questions, **_kwargs):
        if any(str(key).startswith("type_") for key in questions):
            return _ClassifyResult()
        return _JoinResult()

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        _system_one,
    )
    view = stitch_hard_wrapped_lines("Title line\nBody sentence here.")
    assert view["blocks"][0]["type"] == "paragraph"
    assert view["generates_markup"] is False
