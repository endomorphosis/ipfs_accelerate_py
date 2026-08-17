"""SCH-006 assurance-aware ContextPack compilation tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import (
    ADMISSION_CONSERVATIVE,
    ADMISSION_EXACT,
    ADMISSION_RAW,
    CapsuleAdmission,
    admit_capsule,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    CONTEXT_PACK_INTERFACE,
    TOKEN_ESTIMATOR_VERSION,
    ContextCoveragePolicy,
    ContextPackError,
    ContextPacker,
    ContextTokenEstimate,
    pack_context,
    project_admission_to_reference,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    ContextPack,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_context_slice import (
    build_production_context_slice,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _admit(
    *,
    label: str,
    confidence: str,
    assessment_admission: str | None = None,
    freshness: str = "fresh",
) -> CapsuleAdmission:
    capsule = {
        "capsule_cid": _cid(f"cap-{label}"),
        "stable_symbol_id": f"dep.{label}",
        "version_cid": _cid(f"ver-{label}"),
        "source_cid": _cid(f"src-{label}"),
        "confidence": confidence,
    }
    assessment = None
    if assessment_admission is not None:
        assessment = {
            "freshness": freshness,
            "admission": assessment_admission,
            "caveats": (
                ("confidence:conservative",)
                if assessment_admission == ADMISSION_CONSERVATIVE
                else ()
            ),
            "assessment_cid": _cid(f"assess-{label}"),
        }
    return admit_capsule(
        capsule,
        semantic_state_root_cid=_cid("root"),
        assessment=assessment,
    )


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    payload = {
        "objective": "repair greeting helper",
        "target_source_cid": _cid("target"),
        "surrounding_source_cid": _cid("surrounding"),
        "test_source_cid": _cid("tests"),
        "delta_cid": _cid("delta"),
        "obligation_cids": [_cid("obl-a")],
        "counterexample_cids": [_cid("cex-1")],
        "interface_cids": [_cid("iface")],
        "assumptions": ["tests remain hermetic"],
        "dependency_admissions": [
            _admit(label="exact", confidence="exact", assessment_admission=ADMISSION_EXACT),
            _admit(
                label="cons",
                confidence="conservative",
                assessment_admission=ADMISSION_CONSERVATIVE,
            ),
        ],
        "budget": ContextBudget(
            max_input_tokens=50_000,
            reserved_output_tokens=256,
            reserved_tool_tokens=64,
        ),
    }
    payload.update(overrides)
    return payload


def test_pack_contains_all_required_context_pack_fields() -> None:
    result = pack_context(**_base_kwargs())
    pack = result.pack
    assert isinstance(pack, ContextPack)
    body = pack.to_dict()
    required = {
        "objective",
        "target_source_cid",
        "surrounding_source_cid",
        "test_source_cid",
        "dependency_capsule_cids",
        "obligation_cids",
        "counterexample_cids",
        "delta_cid",
        "interface_cids",
        "assumptions",
        "exclusions",
        "token_totals",
        "estimator_version",
        "risk",
        "route",
        "escalation_recommendation",
    }
    assert set(body) == required
    # Round-trip through closed contract.
    assert ContextPack.from_dict(body).to_dict() == body
    assert result.pack_cid == cid_for_payload(body)
    assert pack.estimator_version == TOKEN_ESTIMATOR_VERSION
    assert pack.objective == "repair greeting helper"


def test_exact_and_conservative_capsules_included_with_visible_caveats() -> None:
    result = pack_context(**_base_kwargs())
    pack = result.pack
    exact_cid = _cid("cap-exact")
    cons_cid = _cid("cap-cons")
    assert exact_cid in pack.dependency_capsule_cids
    assert cons_cid in pack.dependency_capsule_cids
    assert pack.dependency_capsule_cids == tuple(sorted(pack.dependency_capsule_cids))
    # Conservative caveats are visible assumptions.
    assert any("conservative" in item for item in pack.assumptions)
    capsule_refs = [
        item for item in result.references if item.kind == "dependency_capsule"
    ]
    assert len(capsule_refs) == 2
    assert all(item.tier is ContextTier.EVIDENCE for item in capsule_refs)
    assert any(
        item.metadata.get("admission") == ADMISSION_CONSERVATIVE
        for item in capsule_refs
    )


def test_heuristic_capsule_forces_raw_source_not_substitution() -> None:
    heuristic = _admit(label="heur", confidence="heuristic")
    result = pack_context(
        **_base_kwargs(
            dependency_admissions=[heuristic],
        )
    )
    pack = result.pack
    assert pack.dependency_capsule_cids == ()
    assert heuristic.ref.source_cid
    raw_refs = [
        item for item in result.references if item.kind == "raw_dependency_source"
    ]
    assert len(raw_refs) == 1
    assert raw_refs[0].tier is ContextTier.INVARIANT
    assert raw_refs[0].referenced_content_id == heuristic.ref.source_cid
    assert raw_refs[0].required is True
    assert any("retrieve_raw_source" in item for item in pack.exclusions)
    assert any(d.startswith("raw_source:") for d in result.decisions)


def test_opaque_capsule_raw_source_from_scanned_tree_identity() -> None:
    opaque = _admit(label="opq", confidence="opaque")
    result = pack_context(**_base_kwargs(dependency_admissions=[opaque]))
    assert opaque.ref.capsule_cid not in result.pack.dependency_capsule_cids
    raw = [r for r in result.references if r.kind == "raw_dependency_source"][0]
    # Opaque path binds the producer source CID (exact scanned tree identity).
    assert raw.referenced_content_id == opaque.ref.source_cid
    assert raw.metadata["raw_source_required"] is True
    assert raw.metadata["datasets_authority"] is True


def test_exact_targets_never_compressed() -> None:
    result = pack_context(**_base_kwargs())
    for kind in ("target_source", "surrounding_source", "test_source"):
        refs = [item for item in result.references if item.kind == kind]
        assert len(refs) == 1
        assert refs[0].tier is ContextTier.INVARIANT
        assert refs[0].required is True
        assert refs[0].metadata.get("never_compress") is True
    assert result.pack.target_source_cid == _cid("target")
    assert result.pack.surrounding_source_cid == _cid("surrounding")
    assert result.pack.test_source_cid == _cid("tests")


def test_llm_summary_cannot_satisfy_proof_or_coverage() -> None:
    """Suggestions never replace required source coverage."""
    result = pack_context(**_base_kwargs())
    # No suggestion-tier reference may be required.
    for item in result.references:
        if item.tier is ContextTier.SUGGESTION:
            assert item.required is False
    # Required coverage kinds remain invariant source CIDs, not summaries.
    required_kinds = {
        item.kind
        for item in result.references
        if item.required and item.tier is ContextTier.INVARIANT
    }
    assert "target_source" in required_kinds
    assert "summary" not in required_kinds
    assert "llm_summary" not in required_kinds


def test_budget_failure_recommends_escalation_without_truncating_required() -> None:
    tight = ContextBudget(
        max_input_tokens=8,
        reserved_output_tokens=0,
        reserved_tool_tokens=0,
        max_items=16,
        max_item_bytes=4096,
        max_serialized_bytes=65_536,
        max_depth=8,
        max_text_bytes=4096,
    )
    result = pack_context(**_base_kwargs(budget=tight))
    assert result.budget_exceeded is True
    assert result.pack.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert "budget_failure" in result.pack.escalation_recommendation
    assert "escalate" in result.pack.escalation_recommendation
    # Required exact CIDs remain present (no silent truncation).
    assert result.pack.target_source_cid == _cid("target")
    assert result.pack.surrounding_source_cid == _cid("surrounding")
    assert result.pack.test_source_cid == _cid("tests")
    assert result.pack.delta_cid == _cid("delta")
    assert any("budget_exceeded" in item for item in result.decisions)
    # Coverage is reported unsatisfied under budget failure.
    assert result.coverage_satisfied is False


def test_identical_inputs_yield_identical_pack_cid_and_tokens() -> None:
    kwargs = _base_kwargs()
    a = pack_context(**kwargs)
    b = pack_context(**kwargs)
    assert a.pack.to_dict() == b.pack.to_dict()
    assert a.pack_cid == b.pack_cid
    assert a.token_estimate.to_dict() == b.token_estimate.to_dict()
    assert a.decisions == b.decisions
    assert [r.to_dict() for r in a.references] == [r.to_dict() for r in b.references]


def test_token_totals_match_estimator_and_category_sum() -> None:
    result = pack_context(**_base_kwargs())
    totals = result.pack.token_totals
    assert totals == result.token_estimate.totals
    assert result.token_estimate.total == sum(totals.values())
    assert result.token_estimate.estimator_version == TOKEN_ESTIMATOR_VERSION
    for key in (
        "target_source",
        "surrounding_source",
        "test_source",
        "dependency_capsules",
        "obligations",
        "counterexamples",
        "delta",
        "interfaces",
        "assumptions",
    ):
        assert key in totals
        assert totals[key] >= 1


def test_exclusions_require_explanations() -> None:
    with pytest.raises(ContextPackError, match="explain"):
        pack_context(
            **_base_kwargs(
                exclusions=["mystery"],
            )
        )


def test_project_admission_to_reference_tiers() -> None:
    exact = _admit(
        label="proj-e",
        confidence="exact",
        assessment_admission=ADMISSION_EXACT,
    )
    raw = _admit(label="proj-h", confidence="heuristic")
    exact_ref = project_admission_to_reference(exact)
    raw_ref = project_admission_to_reference(raw)
    assert exact_ref.tier is ContextTier.EVIDENCE
    assert exact_ref.kind == "dependency_capsule"
    assert raw_ref.tier is ContextTier.INVARIANT
    assert raw_ref.kind == "raw_dependency_source"
    assert raw_ref.required is True


def test_context_packer_class_matches_pack_context() -> None:
    kwargs = _base_kwargs()
    budget = kwargs.pop("budget")
    packer = ContextPacker(budget=budget)
    via_class = packer.pack(**kwargs)
    via_fn = pack_context(budget=budget, **kwargs)
    assert via_class.pack_cid == via_fn.pack_cid
    assert via_class.pack.to_dict() == via_fn.pack.to_dict()


def test_coverage_policy_defaults() -> None:
    policy = ContextCoveragePolicy()
    assert "target_source" in policy.required_kinds
    assert "target_source" in policy.never_compress_kinds
    assert policy.allow_capsule_substitution is True
    body = policy.to_dict()
    assert body["schema"]


def test_raw_source_regions_included_as_invariant() -> None:
    result = pack_context(
        **_base_kwargs(
            dependency_admissions=[],
            raw_source_regions=[
                {
                    "source_cid": _cid("opaque-region"),
                    "reason": "missing_capsule",
                    "path": "pkg/mod.py",
                }
            ],
        )
    )
    refs = [r for r in result.references if r.kind == "raw_source_region"]
    assert len(refs) == 1
    assert refs[0].tier is ContextTier.INVARIANT
    assert refs[0].referenced_content_id == _cid("opaque-region")
    assert refs[0].path == "pkg/mod.py"


def test_stale_admission_excluded_from_dependency_capsules() -> None:
    stale = _admit(
        label="stale",
        confidence="exact",
        assessment_admission=ADMISSION_RAW,
        freshness="stale",
    )
    result = pack_context(**_base_kwargs(dependency_admissions=[stale]))
    assert result.pack.dependency_capsule_cids == ()
    assert any("stale" in item or "raw_source" in item for item in result.pack.exclusions)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _mini_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Context Pack Test")
    _git(repo, "config", "user.email", "context-pack@example.invalid")
    target = repo / "src" / "greeting.py"
    target.parent.mkdir()
    target.write_text(
        "def greet(name: str) -> str:\n    return f\"hello {name}\"\n",
        encoding="utf-8",
    )
    test_path = repo / "tests" / "test_greeting.py"
    test_path.parent.mkdir()
    test_path.write_text(
        "from src.greeting import greet\n\ndef test_greet():\n    assert greet('a')\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo


def test_production_source_coverage_proof_present(tmp_path: Path) -> None:
    repo = _mini_repo(tmp_path)
    task = {
        "task_id": "SCH-006-CTX",
        "title": "pack production slice",
        "acceptance": "slice covers declared paths",
        "outputs": ["src/greeting.py"],
    }
    manifest = build_production_context_slice(
        repo_root=repo,
        task_id=task["task_id"],
        task_payload=task,
        read_paths=["src/greeting.py", "tests/test_greeting.py"],
        effect_paths=["src/greeting.py"],
    )
    result = pack_context(
        **_base_kwargs(production_slice=manifest),
    )
    assert result.production_slice is manifest
    assert result.production_slice_cid == manifest.manifest_cid
    # Production slice carries exact source-coverage records.
    payload = manifest.to_dict()
    assert payload["schema"]
    assert payload["sources"]
    assert payload["scope"]["read_paths"]
    assert "authority" in payload
    assert payload["authority"]["provider_may_read_undeclared_paths"] is False
    assert any(d.startswith("production_slice:") for d in result.decisions)


def test_context_token_estimate_rejects_inconsistent_total() -> None:
    with pytest.raises(ContextPackError, match="does not match"):
        ContextTokenEstimate(
            totals={"a": 1, "b": 2},
            estimator_version="v",
            total=9,
        )


def test_pack_result_interface_marker() -> None:
    result = pack_context(**_base_kwargs())
    assert result.to_dict()["interface"] == CONTEXT_PACK_INTERFACE
    assert result.to_dict()["pack_cid"] == result.pack_cid
