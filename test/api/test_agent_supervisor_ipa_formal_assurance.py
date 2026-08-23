"""FACP-042: IPA product domains, hermetic Souffle fallback, and CEGAR traces."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    BUNDLE,
    HERMETIC_EVALUATOR_ID,
    IPA_CORPUS_FAMILIES,
    SCHEMA,
    STABLE_RULE_IDS,
    TASK_ID,
    CapabilityDisposition,
    DatalogAtom,
    FindingDisposition,
    HermeticReferenceEvaluator,
    IpaError,
    IpaRuleId,
    SouffleStatus,
    SpuriousPathRefinement,
    analyze_python_source,
    analyze_seeded_corpus,
    analyze_tree,
    analyze_typescript_source,
    analyze_with_capability,
    default_ipa_datalog_rules,
    findings_for_corpus_entry,
    ipa_corpus_entries,
    load_defect_corpus,
    mark_imprecise,
    probe_souffle_capability,
    refine_spurious_paths,
)


def _repo_root() -> Path:
    # test/api -> test -> external/ipfs_accelerate -> workspace root
    return Path(__file__).resolve().parents[4]


def _default_corpus_path() -> Path:
    return (
        _repo_root()
        / "implementation_plan"
        / "formal_assurance_control_plane"
        / "baseline"
        / "defect_corpus.jsonl"
    )


SEED_SOURCE = '''\
"""Seeded IPA defect fixture covering every stable rule ID."""

import os
import subprocess
import hashlib
from unittest.mock import MagicMock

# import-time mutation (ipa.rule.import_effect)
os.environ["IPFS_DATASETS_AUTO_INSTALL"] = "true"
subprocess.run(["echo", "bootstrap"], check=False)


def create_mock_handler():
    # mock-to-production (ipa.rule.mock_to_production_flow)
    worker = MagicMock()
    worker.is_available = True
    return {"capability": True, "available": True, "mock": True}


def add_endpoint(name: str):
    api_available = True
    handler = create_mock_handler()
    return {"success": True, "available": True, "handler": handler}


def store_to_ipfs(payload: bytes) -> dict:
    # pseudo-CID (ipa.rule.pseudo_cid_construction)
    digest = hashlib.sha256(payload).hexdigest()
    mock_cid = f"Qm{digest[:44]}"
    return {"success": True, "cid": mock_cid}


def fragile_upload(payload: bytes) -> dict:
    try:
        raise RuntimeError("backend down")
    except Exception:
        # exception swallowing (ipa.rule.exception_swallowing)
        pass
    return {"success": True, "cid": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"}
'''


TS_SEED_SOURCE = """
export const AUTO = true;
process.env.IPFS_AUTO_INSTALL = "true";

export function mockCapability() {
  return { available: true, capability: createMock() };
}

export function fakeCid() {
  return { cid: "bafypseudo0001", success: true };
}
"""


def _assert_source_to_sink(finding) -> None:
    assert finding.rule_id in STABLE_RULE_IDS
    assert finding.trace.steps
    assert len(finding.trace.steps) >= 2
    assert finding.trace.source_label
    assert finding.trace.sink_label
    assert finding.source_span.start_line >= 1
    assert finding.sink_span.start_line >= 1
    kinds = {step.kind for step in finding.trace.steps}
    assert "source" in kinds
    assert "sink" in kinds or "rule" in kinds


def test_stable_rule_ids_cover_ipa_evidence_subset() -> None:
    expected = {
        IpaRuleId.IMPORT_EFFECT.value,
        IpaRuleId.MOCK_TO_PRODUCTION.value,
        IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value,
        IpaRuleId.EXCEPTION_SWALLOWING.value,
        IpaRuleId.PSEUDO_CID.value,
    }
    assert expected <= STABLE_RULE_IDS
    assert IPA_CORPUS_FAMILIES == {
        "import_effect",
        "mock_capability",
        "false_success",
        "pseudo_cid",
    }


def test_analyzer_finds_every_seeded_defect_with_trace_and_rule_id(
    tmp_path: Path,
) -> None:
    module = tmp_path / "pkg" / "seeded_ipa.py"
    module.parent.mkdir(parents=True)
    module.write_text(SEED_SOURCE, encoding="utf-8")

    capability = probe_souffle_capability(force_absent=True)
    report = analyze_tree(
        tmp_path,
        relative_paths=["pkg/seeded_ipa.py"],
        souffle_capability=capability,
    )

    assert report.schema == SCHEMA
    assert report.to_dict()["task_id"] == TASK_ID
    assert report.to_dict()["bundle"] == BUNDLE
    assert report.souffle_capability is not None
    assert report.souffle_capability.available is False
    assert report.analysis_backend == "hermetic_reference_evaluator"

    active = report.active_findings
    rule_ids = {item.rule_id for item in active}
    for rule in (
        IpaRuleId.IMPORT_EFFECT,
        IpaRuleId.MOCK_TO_PRODUCTION,
        IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
        IpaRuleId.EXCEPTION_SWALLOWING,
        IpaRuleId.PSEUDO_CID,
    ):
        assert rule.value in rule_ids, f"missing rule {rule.value} in {sorted(rule_ids)}"

    for finding in active:
        _assert_source_to_sink(finding)
        assert finding.domain_state.to_dict()
        assert finding.family


def test_naming_alone_is_not_a_defect() -> None:
    source = (
        "def demo(success, available, cid):\n"
        "    helper = success\n"
        "    return available\n"
    )
    findings = analyze_python_source(source, path="naming_only.py")
    assert findings == ()


def test_unavailable_souffle_yields_typed_capability_with_hermetic_evaluator() -> None:
    capability = probe_souffle_capability(force_absent=True)
    assert capability.status is SouffleStatus.ABSENT
    assert capability.available is False
    assert capability.disposition is CapabilityDisposition.TYPED_CAPABILITY_GAP
    assert capability.reference_evaluator == HERMETIC_EVALUATOR_ID
    assert "skip_analysis" in capability.prohibited_compensation
    assert "auto_install" in capability.prohibited_compensation

    report = analyze_with_capability(
        SEED_SOURCE,
        path="seeded_ipa.py",
        souffle_capability=capability,
    )
    assert report.souffle_capability is not None
    assert report.souffle_capability.available is False
    assert report.analysis_backend == "hermetic_reference_evaluator"
    assert report.active_findings, "analysis must not be skipped when Souffle is absent"
    assert IpaRuleId.IMPORT_EFFECT.value in report.rule_ids


def test_hermetic_reference_evaluator_derives_ipa_violations() -> None:
    evaluator = HermeticReferenceEvaluator()
    facts = (
        DatalogAtom("MockSource", ("mock1",)),
        DatalogAtom("FlowsTo", ("mock1", "sink1")),
        DatalogAtom("LiveSink", ("sink1",)),
        DatalogAtom("RawHash", ("hash1",)),
        DatalogAtom("FlowsTo", ("hash1", "cid1")),
        DatalogAtom("CidSink", ("cid1",)),
        DatalogAtom("SuccessClaim", ("ok1",)),
        DatalogAtom("FlowsTo", ("ok1", "eff1")),
        DatalogAtom("UnobservedEffect", ("eff1",)),
        DatalogAtom("ModuleTopLevel", ("mod1",)),
        DatalogAtom("EffectfulCall", ("mod1", "subprocess.run")),
        DatalogAtom("SwallowedException", ("ex1",)),
        DatalogAtom("FlowsTo", ("ex1", "ok2")),
        DatalogAtom("SuccessClaim", ("ok2",)),
    )
    result = evaluator.evaluate(facts, default_ipa_datalog_rules())
    assert result.evaluator_id == HERMETIC_EVALUATOR_ID
    violations = result.facts("IpaViolation")
    rule_ids = {row[2] for row in violations if len(row) >= 3}
    assert IpaRuleId.MOCK_TO_PRODUCTION.value in rule_ids
    assert IpaRuleId.PSEUDO_CID.value in rule_ids
    assert IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value in rule_ids
    assert IpaRuleId.IMPORT_EFFECT.value in rule_ids
    assert IpaRuleId.EXCEPTION_SWALLOWING.value in rule_ids


def test_spurious_paths_refine_without_suppressing_seeds() -> None:
    findings = analyze_python_source(SEED_SOURCE, path="seeded_ipa.py")
    assert findings

    seed = findings_for_corpus_entry(
        {
            "seed_id": "seed:test-mock-flow",
            "defect_id": "defect:test-mock-flow",
            "family": "mock_capability",
            "title": "seeded mock flow",
            "roadmap_seed": True,
            "expected_illegal_promotion": "simulated -> live",
            "source_spans": [
                {
                    "path": "seeded_ipa.py",
                    "start_line": 12,
                    "end_line": 16,
                    "symbol": "create_mock_handler",
                    "excerpt": "MagicMock",
                }
            ],
            "call_flow_path": ["create_mock_handler", "add_endpoint"],
        }
    )
    assert len(seed) == 1
    assert seed[0].is_corpus_seed
    _assert_source_to_sink(seed[0])

    # Mark one non-seed finding as imprecise/spurious.
    non_seed = next(item for item in findings if not item.is_corpus_seed)
    imprecise = mark_imprecise(non_seed, note="imprecise dynamic dispatch")
    combined = seed + (imprecise,)

    refinements = (
        SpuriousPathRefinement(
            refinement_id="cegar:drop-imprecise",
            finding_id=imprecise.finding_id,
            reason="callee resolved to pure helper",
            constraint="CalleePure(imprecise)",
        ),
        SpuriousPathRefinement(
            refinement_id="cegar:illegal-seed-suppress",
            finding_id=seed[0].finding_id,
            reason="attempt to suppress corpus seed",
            constraint="ShouldNotApply",
        ),
    )
    refined, refined_ids = refine_spurious_paths(combined, refinements)
    by_id = {item.finding_id: item for item in refined}

    assert imprecise.finding_id in refined_ids
    assert by_id[imprecise.finding_id].disposition is FindingDisposition.REFINED_AWAY
    assert by_id[seed[0].finding_id].disposition is FindingDisposition.CORPUS_BOUND
    assert by_id[seed[0].finding_id].is_corpus_seed
    assert "refused: corpus seed" in by_id[seed[0].finding_id].refinement_note


def test_analyze_seeded_corpus_binds_every_ipa_seed_with_rule_and_trace() -> None:
    corpus_path = _default_corpus_path()
    if not corpus_path.is_file():
        pytest.skip(f"defect corpus unavailable: {corpus_path}")

    capability = probe_souffle_capability(force_absent=True)
    report = analyze_seeded_corpus(
        corpus_path=corpus_path,
        repo_root=_repo_root(),
        souffle_capability=capability,
    )

    entries = ipa_corpus_entries(load_defect_corpus(corpus_path))
    assert entries, "expected IPA-relevant corpus seeds"
    expected_ids = {str(entry["seed_id"]) for entry in entries}
    assert expected_ids <= set(report.corpus_seed_ids_bound)

    active_by_seed: dict[str, list] = {}
    for finding in report.active_findings:
        if not finding.corpus_seed_id:
            continue
        assert finding.disposition is not FindingDisposition.REFINED_AWAY
        _assert_source_to_sink(finding)
        assert finding.rule_id in STABLE_RULE_IDS
        active_by_seed.setdefault(finding.corpus_seed_id, []).append(finding)

    missing = sorted(expected_ids - set(active_by_seed))
    assert not missing, f"IPA analyzer missed seeded defects: {missing[:20]}"

    # Family -> rule mapping holds for corpus-bound findings.
    for entry in entries:
        seed_id = str(entry["seed_id"])
        family = str(entry["family"])
        bound = active_by_seed[seed_id]
        assert any(item.family == family or item.rule.family == family for item in bound)

    assert report.souffle_capability is not None
    assert report.souffle_capability.available is False
    assert report.analysis_backend == "hermetic_reference_evaluator"


def test_corpus_refinement_cannot_drop_seeds() -> None:
    corpus_path = _default_corpus_path()
    if not corpus_path.is_file():
        pytest.skip(f"defect corpus unavailable: {corpus_path}")

    entries = ipa_corpus_entries(load_defect_corpus(corpus_path))
    sample = entries[0]
    findings = findings_for_corpus_entry(sample)
    aggressive = tuple(
        SpuriousPathRefinement(
            refinement_id=f"cegar:drop-{item.finding_id}",
            finding_id=item.finding_id,
            reason="illegal seed suppression",
        )
        for item in findings
    )
    refined, refined_ids = refine_spurious_paths(findings, aggressive)
    assert refined_ids == ()
    assert all(item.disposition is FindingDisposition.CORPUS_BOUND for item in refined)
    assert all(item.is_corpus_seed for item in refined)


def test_typescript_seed_patterns_emit_stable_rules() -> None:
    findings = analyze_typescript_source(TS_SEED_SOURCE, path="seed.ts")
    rule_ids = {item.rule_id for item in findings}
    assert IpaRuleId.IMPORT_EFFECT.value in rule_ids
    assert IpaRuleId.PSEUDO_CID.value in rule_ids or IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value in rule_ids
    for finding in findings:
        _assert_source_to_sink(finding)


def test_product_domain_join_is_monotonic() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
        EffectAbstract,
        IdentityAbstract,
        ProductDomainState,
        ResultAbstract,
        TrustAbstract,
    )

    left = ProductDomainState(
        effect=EffectAbstract.PURE,
        trust=TrustAbstract.SIMULATED,
        result=ResultAbstract.SUCCESS_CLAIMED,
        identity=IdentityAbstract.RAW_HASH,
    )
    right = ProductDomainState(
        effect=EffectAbstract.NETWORK,
        trust=TrustAbstract.LIVE_OBSERVED,
        result=ResultAbstract.VERIFIED,
        identity=IdentityAbstract.PSEUDO_CID,
    )
    joined = left.join(right)
    assert joined.effect is EffectAbstract.NETWORK
    assert joined.trust is TrustAbstract.LIVE_OBSERVED
    assert joined.result is ResultAbstract.VERIFIED
    assert joined.identity is IdentityAbstract.PSEUDO_CID


def test_non_ipa_corpus_family_rejected() -> None:
    with pytest.raises(IpaError):
        findings_for_corpus_entry(
            {
                "seed_id": "seed:browser",
                "family": "browser_authority",
                "source_spans": [
                    {
                        "path": "x.ts",
                        "start_line": 1,
                        "end_line": 1,
                        "excerpt": "allow",
                    }
                ],
            }
        )
