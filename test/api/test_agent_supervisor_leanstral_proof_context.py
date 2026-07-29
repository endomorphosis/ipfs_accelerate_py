"""Fixed-theorem prompt contracts for the Leanstral proof provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (
    write_code_evidence_graph_artifact,
)
from ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph import (
    build_code_evidence_graph,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
    LeanstralProofProvider,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_context import (
    LEANSTRAL_PROOF_OUTPUT_SCHEMA,
    ContextEntry,
    ContextTrust,
    FixedTheoremIdentity,
    LeanstralPromptLimits,
    LeanstralProofContext,
    ProofContextBudgetError,
    ProofContextBuilder,
    ProofContextError,
    ProofContextLimits,
    ProofContextQuery,
    ProofContextTarget,
    build_leanstral_proof_context,
    estimate_context_tokens,
)

SYMBOL = "agent_supervisor.fixed_scope"


def _capsule(tmp_path: Path, obligation_id: str = "goal-1"):
    graph = build_code_evidence_graph(
        task_records=[
            {
                "task_id": "REF-262",
                "canonical_task_cid": "task-ref-262",
                "title": "Fixed theorem prompts",
            }
        ],
        ast_records=[
            {
                "scope_id": "scope-fixed",
                "kind": "qualified_symbol",
                "qualified_symbol": SYMBOL,
                "path": "agent_supervisor/fixed_scope.lean",
            }
        ],
        obligations=[
            {
                "obligation_id": "premise-1",
                "task_id": "REF-262",
                "ast_scope_ids": ["scope-fixed"],
                "statement": "h : P",
            },
            {
                "obligation_id": obligation_id,
                "task_id": "REF-262",
                "ast_scope_ids": ["scope-fixed"],
                "premise_ids": ["premise-1"],
                "assumptions": ["P"],
                "conclusion": "P",
                "statement": "theorem fixed (h : P) : P",
                "template_id": "identity-proof",
                "template_version": "1",
            },
        ],
        proof_records=[
            {
                "receipt_id": "receipt-premise",
                "task_id": "REF-262",
                "obligation_id": "premise-1",
                "verdict": "proved",
                "freshness": "current",
                "authoritative_assurance": "kernel_verified",
            }
        ],
        validation_records=[
            {
                "validation_receipt_id": f"failure-{obligation_id}",
                "task_id": "REF-262",
                "obligation_ids": [obligation_id],
                "status": "failed",
                "failure_reason": "simp did not close the final goal",
                "contradiction_id": f"failure-{obligation_id}",
                "contradiction": "the tactic left P as an open goal",
            }
        ],
    )
    path = tmp_path / f"{obligation_id}.json"
    write_code_evidence_graph_artifact(path, graph)
    return ProofContextBuilder(path).build(
        ProofContextQuery(
            task_id="REF-262",
            symbols=(SYMBOL,),
            obligation_ids=(obligation_id, "premise-1"),
            receipt_ids=("receipt-premise",),
            contradiction_ids=(f"failure-{obligation_id}",),
        ),
        target=ProofContextTarget.LEANSTRAL,
        limits=ProofContextLimits(max_bytes=24_000, max_tokens=6_000),
    )


def _theorem(obligation_id: str = "goal-1", theorem_id: str = "Fixed.identity"):
    return FixedTheoremIdentity(
        theorem_id=theorem_id,
        obligation_id=obligation_id,
        declaration_name="Fixed.identity",
        assumptions=("P",),
        conclusion="P",
        template_id="identity-proof",
        template_version="1",
        source_scope=(SYMBOL,),
        allowed_premise_ids=("premise-1",),
        canonical_source_digest="sha256:fixed-source",
    )


def _request(capsule, theorem, **payload):
    values = {
        "context_capsule": capsule.to_dict(),
        "fixed_theorem": theorem.to_dict(),
        "resource_class": "model",
    }
    values.update(payload)
    return ProviderRequest(
        request_id=f"request-{theorem.obligation_id}",
        operation="prove",
        payload=values,
        resource_budget=ResourceBudget(
            wall_time_ms=5_000,
            max_premises=4,
            max_output_bytes=8_192,
            model_token_limit=512,
        ),
    )


def test_prompt_contains_fixed_identity_evidence_failures_and_schema(
    tmp_path: Path,
) -> None:
    capsule = _capsule(tmp_path)
    theorem = _theorem()
    context = build_leanstral_proof_context(capsule, theorem)
    prompt = json.loads(context.to_prompt())

    assert prompt["fixed_theorem"]["identity_digest"] == theorem.identity_digest
    assert prompt["fixed_theorem"]["assumptions"] == ["P"]
    assert prompt["fixed_theorem"]["conclusion"] == "P"
    assert prompt["fixed_theorem"]["template_id"] == "identity-proof"
    assert prompt["fixed_theorem"]["source_scope"] == [SYMBOL]
    assert [item["premise_id"] for item in prompt["allowed_premises"]] == ["premise-1"]
    assert prompt["trusted_prior_receipts"] == [
        {
            "assurance": "kernel_verified",
            "checked_evidence": True,
            "obligation_id": "premise-1",
            "receipt_id": "receipt-premise",
            "repository_tree_id": "",
            "trust": ContextTrust.TRUSTED_FACT.value,
            "verdict": "proved",
        }
    ]
    assert prompt["compact_failures"][0]["failure_id"] == "failure-goal-1"
    assert prompt["output_schema"]["schema"] == LEANSTRAL_PROOF_OUTPUT_SCHEMA
    assert set(prompt["immutable_constraints"]["may_propose"]) == {
        "proof_text",
        "decomposition",
    }
    assert context.prompt_bytes <= context.limits.max_bytes
    assert context.prompt_tokens <= context.limits.max_tokens


def test_semantic_stage_context_reaches_final_prompt_without_authority(
    tmp_path: Path,
) -> None:
    capsule = _capsule(tmp_path)
    semantic_context = {
        "schema": "example.semantic-stage-context@1",
        "entities": ["Ada", "archivist"],
        "normalized_predicates": ["trained(Ada)"],
        # These labels remain visible as untrusted model input but must not be
        # promoted into the typed compact-failure lane.
        "status": "failed",
        "failure_reason": "untrusted semantic-stage label",
    }
    duplicate_late = ContextEntry(
        trust=ContextTrust.UNTRUSTED_SUGGESTION,
        kind="semantic_stage_context",
        record_id="semantic-a",
        fields=semantic_context,
    )
    duplicate_early = replace(duplicate_late)
    capsule = replace(
        capsule,
        untrusted_suggestions=(
            *capsule.untrusted_suggestions,
            duplicate_late,
            duplicate_early,
        ),
    )

    context = build_leanstral_proof_context(capsule, _theorem())
    prompt = json.loads(context.to_prompt())

    # Exact duplicate records are collapsed without losing distinct provenance.
    assert [item["hint_id"] for item in prompt["untrusted_semantic_hints"]] == [
        "semantic-a"
    ]
    hint = prompt["untrusted_semantic_hints"][0]
    assert hint["semantic_context"] == semantic_context
    assert hint["trust"] == ContextTrust.UNTRUSTED_SUGGESTION.value
    assert hint["checked_evidence"] is False
    assert hint["authoritative"] is False
    assert hint["usable_as_premise"] is False
    assert hint["usable_as_proof_evidence"] is False
    assert hint["usable_as_failure_evidence"] is False
    assert prompt["semantic_hint_policy"]["semantic_guidance_only"] is True
    assert all(
        item["failure_id"] != "semantic-a"
        for item in prompt["compact_failures"]
    )
    assert context.omitted["untrusted_semantic_hints"] == 0
    assert (
        context.prompt_sha256
        == hashlib.sha256(context.to_prompt().encode("utf-8")).hexdigest()
    )
    assert LeanstralProofContext.from_json(context.to_prompt()) == context

    changed_entry = replace(
        duplicate_early,
        fields={**semantic_context, "entities": ["Grace"]},
    )
    changed = build_leanstral_proof_context(
        replace(
            capsule,
            untrusted_suggestions=(
                *tuple(
                    item
                    for item in capsule.untrusted_suggestions
                    if item.kind != "semantic_stage_context"
                ),
                changed_entry,
            ),
        ),
        _theorem(),
    )
    assert changed.context_id != context.context_id
    assert changed.prompt_sha256 != context.prompt_sha256


def test_semantic_hint_count_and_byte_limits_fail_closed(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    hints = tuple(
        ContextEntry(
            trust=ContextTrust.UNTRUSTED_SUGGESTION,
            kind="semantic_stage_context",
            record_id=f"semantic-{index}",
            fields={"candidate": character * 96},
        )
        for index, character in enumerate(("a", "b"))
    )
    capsule = replace(
        capsule,
        untrusted_suggestions=(*capsule.untrusted_suggestions, *hints),
    )

    with pytest.raises(ProofContextBudgetError, match="cannot fit"):
        build_leanstral_proof_context(
            capsule,
            _theorem(),
            limits=LeanstralPromptLimits(
                max_bytes=24_000,
                max_tokens=6_000,
                max_semantic_hints=1,
            ),
        )
    with pytest.raises(ProofContextBudgetError, match="cannot fit"):
        build_leanstral_proof_context(
            replace(
                capsule,
                untrusted_suggestions=(
                    *tuple(
                        item
                        for item in capsule.untrusted_suggestions
                        if item.kind != "semantic_stage_context"
                    ),
                    hints[0],
                ),
            ),
            _theorem(),
            limits=LeanstralPromptLimits(
                max_bytes=24_000,
                max_tokens=6_000,
                max_semantic_hint_bytes=128,
            ),
        )


def test_realistic_ten_kib_semantic_hint_is_fully_included(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    hint = ContextEntry(
        trust=ContextTrust.UNTRUSTED_SUGGESTION,
        kind="semantic_stage_context",
        record_id="semantic-realistic-size",
        fields={"candidate_ir": "x" * 10_240},
    )
    capsule = replace(
        capsule,
        untrusted_suggestions=(*capsule.untrusted_suggestions, hint),
    )

    context = build_leanstral_proof_context(capsule, _theorem())

    assert len(context.untrusted_semantic_hints) == 1
    assert context.untrusted_semantic_hints[0].byte_count > 10_002
    assert context.omitted["untrusted_semantic_hints"] == 0
    assert json.loads(context.to_prompt())["untrusted_semantic_hints"][0][
        "semantic_context"
    ] == hint.fields


def test_legacy_v1_prompt_without_semantic_hints_deserializes(
    tmp_path: Path,
) -> None:
    context = build_leanstral_proof_context(_capsule(tmp_path), _theorem())
    legacy = context.to_dict()
    legacy["schema"] = (
        "ipfs_accelerate_py.agent_supervisor.leanstral-proof-context@1"
    )
    legacy["instruction"] = (
        "Prove the fixed Lean theorem or decompose it into proof subgoals. "
        "Use only allowed_premises and trusted_prior_receipts as checked "
        "evidence. Compact failures and reusable drafts are diagnostic "
        "hints, not checked evidence. Return only JSON matching output_schema."
    )
    legacy.pop("untrusted_semantic_hints")
    legacy.pop("semantic_hint_policy")
    legacy["limits"]["schema"] = (
        "ipfs_accelerate_py.agent_supervisor.leanstral-prompt-limits@1"
    )
    for name in (
        "max_semantic_hints",
        "max_semantic_hint_bytes",
        "max_semantic_hint_bytes_total",
    ):
        legacy["limits"].pop(name)

    restored = LeanstralProofContext.from_dict(legacy)

    assert restored.untrusted_semantic_hints == ()
    assert restored.to_dict()["schema"].endswith("@2")


def test_semantic_hint_strict_json_and_identity_collisions_fail_closed(
    tmp_path: Path,
) -> None:
    capsule = _capsule(tmp_path)
    not_finite = ContextEntry(
        trust=ContextTrust.UNTRUSTED_SUGGESTION,
        kind="semantic_stage_context",
        record_id="semantic-nan",
        fields={"confidence": float("nan")},
    )
    with pytest.raises(ProofContextError, match="non-finite"):
        build_leanstral_proof_context(
            replace(
                capsule,
                untrusted_suggestions=(
                    *capsule.untrusted_suggestions,
                    not_finite,
                ),
            ),
            _theorem(),
        )

    finite_float = replace(
        not_finite,
        record_id="semantic-float",
        fields={"confidence": 0.8},
    )
    with pytest.raises(ProofContextError, match="string-tagged"):
        build_leanstral_proof_context(
            replace(
                capsule,
                untrusted_suggestions=(
                    *capsule.untrusted_suggestions,
                    finite_float,
                ),
            ),
            _theorem(),
        )

    collisions = (
        ContextEntry(
            trust=ContextTrust.UNTRUSTED_SUGGESTION,
            kind="semantic_stage_context",
            record_id="semantic-collision",
            fields={"candidate": "first"},
        ),
        ContextEntry(
            trust=ContextTrust.UNTRUSTED_SUGGESTION,
            kind="semantic_stage_context",
            record_id="semantic-collision",
            fields={"candidate": "second"},
        ),
    )
    with pytest.raises(ProofContextError, match="conflicting"):
        build_leanstral_proof_context(
            replace(
                capsule,
                untrusted_suggestions=(
                    *capsule.untrusted_suggestions,
                    *collisions,
                ),
            ),
            _theorem(),
        )


def test_provider_builds_prompt_and_accepts_only_bound_output(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    theorem = _theorem()
    calls = []

    def generate(prompt, **kwargs):
        calls.append((json.loads(prompt), kwargs))
        return json.dumps(
            {
                "schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
                "theorem_id": theorem.theorem_id,
                "proposal_kind": "proof",
                "proof_text": "by exact premise_1",
            }
        )

    result = LeanstralProofProvider(llm_generate=generate).prove(
        _request(capsule, theorem)
    )

    assert calls[0][0]["fixed_theorem"]["theorem_id"] == theorem.theorem_id
    assert calls[0][1]["max_new_tokens"] == 512
    assert result["proof_text"] == "by exact premise_1"
    assert result["proposal_kind"] == "proof"
    assert result["theorem_id"] == theorem.theorem_id
    assert result["theorem_equivalence_key"] == theorem.equivalence_key
    assert result["context_capsule_id"] == capsule.capsule_id
    assert result["verified"] is False
    assert result["kernel_checked"] is False
    assert result["prompt_tokens"] <= capsule.limits.max_tokens
    assert result["response_tokens"] <= result["token_budget"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"assumptions": ["False"]},
        {"conclusion": "False"},
        {"template_id": "different-template"},
        {"source_scope": ["unrelated.scope"]},
        {"allowed_premises": ["invented"]},
    ],
)
def test_response_cannot_mutate_fixed_theorem_fields(
    tmp_path: Path, mutation: dict
) -> None:
    capsule = _capsule(tmp_path)
    theorem = _theorem()
    response = {
        "schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
        "theorem_id": theorem.theorem_id,
        "proposal_kind": "proof",
        "proof_text": "by exact premise_1",
        **mutation,
    }
    provider = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: json.dumps(response)
    )

    result = dispatch_provider_request(provider, _request(capsule, theorem))

    assert result.ok is False
    assert result.error.code is ProviderFailureCode.MALFORMED_RESPONSE


def test_decomposition_is_schema_checked_and_stays_unverified(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    theorem = _theorem()
    response = json.dumps(
        {
            "schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
            "theorem_id": theorem.theorem_id,
            "proposal_kind": "decomposition",
            "decomposition": [
                {"subgoal_id": "s1", "statement": "P", "depends_on": []},
                {"subgoal_id": "s2", "statement": "P", "depends_on": ["s1"]},
            ],
        }
    )

    result = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: response
    ).prove(_request(capsule, theorem))

    assert result["proposal_kind"] == "decomposition"
    assert [item["subgoal_id"] for item in result["decomposition"]] == ["s1", "s2"]
    assert result["assurance"] == "unverified"
    assert result["proof_success"] is False


def test_equivalent_task_reuses_draft_only_as_untrusted_hint(tmp_path: Path) -> None:
    first_capsule = _capsule(tmp_path, "goal-1")
    second_capsule = _capsule(tmp_path, "goal-equivalent")
    first_theorem = _theorem("goal-1", "Fixed.identity.one")
    second_theorem = _theorem("goal-equivalent", "Fixed.identity.two")
    prompts = []

    def generate(prompt, **_kwargs):
        parsed = json.loads(prompt)
        prompts.append(parsed)
        return json.dumps(
            {
                "schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
                "theorem_id": parsed["fixed_theorem"]["theorem_id"],
                "proposal_kind": "proof",
                "proof_text": "by exact premise_1",
            }
        )

    provider = LeanstralProofProvider(llm_generate=generate)
    first = provider.prove(_request(first_capsule, first_theorem))
    second = provider.prove(_request(second_capsule, second_theorem))

    assert first_theorem.equivalence_key == second_theorem.equivalence_key
    reused = prompts[1]["reusable_untrusted_drafts"]
    assert [item["artifact_id"] for item in reused] == [first["artifact_id"]]
    assert reused[0]["trust"] == ContextTrust.UNTRUSTED_SUGGESTION.value
    assert reused[0]["checked_evidence"] is False
    assert reused[0]["reusable_as_evidence"] is False
    assert second["reused_artifact_ids"] == [first["artifact_id"]]
    assert prompts[1]["trusted_prior_receipts"][0]["receipt_id"] == "receipt-premise"


def test_prompt_and_response_budgets_fail_closed(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    theorem = _theorem()
    with pytest.raises(ProofContextBudgetError, match="byte budget"):
        build_leanstral_proof_context(
            capsule,
            theorem,
            limits=LeanstralPromptLimits(
                max_bytes=512,
                max_tokens=6_000,
            ),
        )

    oversized = "x" * 3_000
    provider = LeanstralProofProvider(llm_generate=lambda *_args, **_kwargs: oversized)
    response = dispatch_provider_request(
        provider,
        _request(
            capsule,
            theorem,
            prompt_limits={
                "max_bytes": capsule.limits.max_bytes,
                "max_tokens": capsule.limits.max_tokens,
            },
        ),
    )
    assert response.ok is False
    assert response.error.code in {
        ProviderFailureCode.RESOURCE_EXHAUSTED,
        ProviderFailureCode.MALFORMED_RESPONSE,
    }
    assert estimate_context_tokens(oversized) > 512
