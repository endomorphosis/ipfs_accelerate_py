"""FACP-051: Bounded counterexample-guided repair."""

from __future__ import annotations

import importlib

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    IpaRuleId,
    analyze_python_source,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.formal_assurance_cegis import (
    BENCHMARK_SCHEMA,
    BUNDLE,
    CERTIFICATE_SCHEMA,
    CEGIS_EVIDENCE,
    FAMILY_TO_TRANSFORM,
    GOAL_ID,
    GRAMMAR_SCHEMA,
    INTERFACE,
    PATCH_CERTIFICATE_EVIDENCE,
    REPAIR_GRAMMAR_EVIDENCE,
    RESULT_SCHEMA,
    SCHEMA,
    STABLE_FAMILY_IDS,
    STABLE_TRANSFORM_IDS,
    TASK_ID,
    TOOLCHAIN_ID,
    CegisAbstentionReason,
    CegisBudget,
    CegisDisposition,
    CegisRepairRequest,
    CounterexampleRecord,
    FormalAssuranceCegis,
    RepairFamily,
    RepairTransformId,
    default_cegis,
    list_repair_grammar,
    run_bounded_cegis,
    run_mutation_benchmark,
    select_repair_family,
)

UNRELATED = "keep-me-bytes-xyz-facp051"

ADMITTED = frozenset(
    {
        "seeded/",
        "fixtures/",
        "external/ipfs_accelerate/",
        "external/ipfs_datasets/",
    }
)


EXPLICIT_INIT_SEED = f'''\
import os
UNRELATED = "{UNRELATED}"

os.environ["IPFS_DATASETS_AUTO_INSTALL"] = "true"

def public_api():
    return UNRELATED
'''

TYPED_UNAVAILABLE_SEED = f'''\
UNRELATED = "{UNRELATED}"

def probe_status():
    return {{"success": True, "detail": UNRELATED}}
'''

SIMULATION_SEED = f'''\
UNRELATED = "{UNRELATED}"
from unittest.mock import MagicMock

def create_mock_handler():
    worker = MagicMock()
    return {{"available": True, "mock": True, "detail": UNRELATED}}
'''

CANONICAL_CID_SEED = f'''\
import hashlib
UNRELATED = "{UNRELATED}"

def store_to_ipfs(payload: bytes) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    cid = f"Qm{{digest[:44]}}"
    return {{"cid": cid, "detail": UNRELATED}}
'''

CRITICAL_ERROR_SEED = f'''\
UNRELATED = "{UNRELATED}"

def fragile_upload(payload: bytes) -> dict:
    try:
        raise RuntimeError("backend down")
    except Exception:
        pass
    return {{"ok_marker": UNRELATED, "success": True}}
'''

BROWSER_AUTHORITY_SEED = f'''\
# seeded browser gateway consent default
UNRELATED = "{UNRELATED}"
consent = invocation.consent ?? 'granted'
default_consent = 'granted'
'''

MUTABLE_DEPENDENCY_SEED = f'''\
# install requirements
UNRELATED = "{UNRELATED}"
ipfs_kit_py @ git+https://github.com/endomorphosis/ipfs_kit_py.git@main
datasets @ git+https://github.com/endomorphosis/ipfs_datasets_py.git@master
release_admissible=true
'''

STALE_PROOF_SEED = f'''\
# historical proof receipt reused as live
UNRELATED = "{UNRELATED}"
reuse_stale_proof=true
historical_receipt_as_live=true
{{"status": "live", "proof_id": "proof:stale"}}
'''

LEASE_RECOVERY_SEED = f'''\
# lease/fence recovery gap
UNRELATED = "{UNRELATED}"
blind_retry=true
allow_blind_retry
unknown_irreversible_effect=true
'''

LICENSE_CONFLICT_SEED = f'''\
# package vs repository license conflict
UNRELATED = "{UNRELATED}"
license = {{text = "MIT"}}
repository_license = "AGPL-3.0"
license_conflict_clearance=false
'''


def _finding_for_rule(source: str, path: str, rule: IpaRuleId):
    findings = analyze_python_source(source, path=path)
    matched = [item for item in findings if item.rule_id == rule.value]
    assert matched, f"expected {rule.value} in {[f.rule_id for f in findings]}"
    primary = [item for item in matched if "hermetic datalog" not in item.message]
    return (primary or matched)[0]


def _cx(
    family: RepairFamily,
    path: str,
    *,
    abstract_markers: tuple[str, ...] = (),
    model_markers: tuple[str, ...] = (),
    test_markers: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> CounterexampleRecord:
    return CounterexampleRecord(
        counterexample_id=f"cx:{family.value}:{path}",
        family=family,
        path=path,
        witness=family.value,
        abstract_markers=abstract_markers,
        model_markers=model_markers,
        test_markers=test_markers,
        metadata=metadata or {},
    )


def test_evidence_envelope_and_closed_grammar() -> None:
    assert TASK_ID == "FACP-051"
    assert GOAL_ID == "FACP-G710"
    assert BUNDLE == "facp/synthesis/repair"
    assert SCHEMA == "facp/cegis-repair@1"
    assert GRAMMAR_SCHEMA == "facp/repair-grammar@1"
    assert CERTIFICATE_SCHEMA == "facp/patch-certificate@1"
    assert RESULT_SCHEMA == "facp/cegis-repair-result@1"
    assert CEGIS_EVIDENCE == "facp/cegis-repair@1"
    assert REPAIR_GRAMMAR_EVIDENCE == "facp/repair-grammar@1"
    assert PATCH_CERTIFICATE_EVIDENCE == "facp/patch-certificate@1"
    assert INTERFACE == "FormalAssuranceCegis@1"
    assert TOOLCHAIN_ID.endswith("formal-assurance-cegis/v1")
    grammar = list_repair_grammar()
    assert len(grammar) == len(RepairFamily)
    assert {row.family.value for row in grammar} == STABLE_FAMILY_IDS
    assert {row.transform_id.value for row in grammar} <= STABLE_TRANSFORM_IDS
    assert STABLE_FAMILY_IDS == {
        "false_success",
        "mock_capability",
        "pseudo_cid",
        "import_effect",
        "exception_swallowing",
        "browser_authority",
        "mutable_dependency",
        "stale_proof",
        "missing_lease_recovery",
        "license_conflict",
    }
    license_row = next(row for row in grammar if row.family is RepairFamily.LICENSE_CONFLICT)
    assert license_row.may_certify is False
    for family, transform in FAMILY_TO_TRANSFORM.items():
        assert select_repair_family(family.value) is family
        assert select_repair_family(transform.value) is family


def test_cold_import_is_hermetic() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.autonomous_repair.formal_assurance_cegis"
    )
    assert module.TASK_ID == "FACP-051"
    assert callable(module.run_bounded_cegis)
    assert callable(module.default_cegis)
    # Deterministic-first: no provider router imports in module source.
    source_path = module.__file__
    assert source_path
    text = open(source_path, encoding="utf-8").read()
    assert "openai" not in text.casefold()
    assert "anthropic" not in text.casefold()
    assert "llm_router" not in text


@pytest.mark.parametrize(
    ("seed", "family", "rule", "path"),
    [
        (EXPLICIT_INIT_SEED, RepairFamily.IMPORT_EFFECT, IpaRuleId.IMPORT_EFFECT, "seeded/explicit_init.py"),
        (
            TYPED_UNAVAILABLE_SEED,
            RepairFamily.FALSE_SUCCESS,
            IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
            "seeded/typed_unavailable.py",
        ),
        (
            SIMULATION_SEED,
            RepairFamily.MOCK_CAPABILITY,
            IpaRuleId.MOCK_TO_PRODUCTION,
            "seeded/simulation_evidence.py",
        ),
        (CANONICAL_CID_SEED, RepairFamily.PSEUDO_CID, IpaRuleId.PSEUDO_CID, "seeded/canonical_cid.py"),
        (
            CRITICAL_ERROR_SEED,
            RepairFamily.EXCEPTION_SWALLOWING,
            IpaRuleId.EXCEPTION_SWALLOWING,
            "seeded/critical_error.py",
        ),
    ],
)
def test_ipa_seeded_corpus_produces_patch_certificate(
    seed: str,
    family: RepairFamily,
    rule: IpaRuleId,
    path: str,
) -> None:
    finding = _finding_for_rule(seed, path, rule)
    request = CegisRepairRequest(
        source=seed,
        counterexample=_cx(family, path),
        finding=finding,
        admitted_paths=ADMITTED,
        budget=CegisBudget(max_iterations=2),
    )
    result = run_bounded_cegis(request)
    assert result.disposition is CegisDisposition.CERTIFIED
    assert result.certificate is not None
    cert = result.certificate
    assert cert.schema == CERTIFICATE_SCHEMA
    assert cert.family is family
    assert cert.transform_id is FAMILY_TO_TRANSFORM[family]
    assert cert.before_hash != cert.after_hash
    assert cert.edits
    assert cert.grants_write_authority is False
    assert cert.mutation_gate.admitted
    assert cert.parent_capsule_cids
    assert cert.patch_capsule_cid
    assert cert.affected_capsule_cids
    assert cert.obligation_ids
    assert UNRELATED in cert.edits[0].after_text or UNRELATED in result.candidates_tried[-1].after_source

    # Original counterexample disappears; no new abstract IPA findings.
    after = result.candidates_tried[-1].after_source
    after_findings = analyze_python_source(after, path=path)
    assert rule.value not in {item.rule_id for item in after_findings}
    before_rules = {item.rule_id for item in analyze_python_source(seed, path=path)}
    after_rules = {item.rule_id for item in after_findings}
    assert after_rules - before_rules == set()

    payload = result.to_dict()
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["certified"] is True
    assert payload["certificate"]["certificate_id"] == cert.certificate_id
    assert payload["grants_write_authority"] is False


@pytest.mark.parametrize(
    ("seed", "family", "path", "markers"),
    [
        (
            BROWSER_AUTHORITY_SEED,
            RepairFamily.BROWSER_AUTHORITY,
            "seeded/browser_authority.ts",
            ("granted",),
        ),
        (
            MUTABLE_DEPENDENCY_SEED,
            RepairFamily.MUTABLE_DEPENDENCY,
            "seeded/mutable_deps.txt",
            ("@main", "@master"),
        ),
        (
            STALE_PROOF_SEED,
            RepairFamily.STALE_PROOF,
            "seeded/stale_proof.json",
            ("reuse_stale_proof", "historical_receipt_as_live", '"status": "live"'),
        ),
        (
            LEASE_RECOVERY_SEED,
            RepairFamily.MISSING_LEASE_RECOVERY,
            "seeded/lease_recovery.cfg",
            ("blind_retry=true", "allow_blind_retry"),
        ),
    ],
)
def test_declaration_seeded_corpus_produces_patch_certificate(
    seed: str,
    family: RepairFamily,
    path: str,
    markers: tuple[str, ...],
) -> None:
    request = CegisRepairRequest(
        source=seed,
        counterexample=_cx(family, path, abstract_markers=markers),
        admitted_paths=ADMITTED,
    )
    result = run_bounded_cegis(request)
    assert result.disposition is CegisDisposition.CERTIFIED, result.reasons
    assert result.certificate is not None
    after = result.candidates_tried[-1].after_source
    for marker in markers:
        assert marker not in after
    assert UNRELATED in after
    # No new abstract/model/test counterexamples via gate results.
    assert all(gate["verdict"] == "pass" for gate in result.to_dict()["gate_results"])


def test_license_conflict_typed_abstention() -> None:
    path = "seeded/license_conflict.toml"
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=LICENSE_CONFLICT_SEED,
            counterexample=_cx(RepairFamily.LICENSE_CONFLICT, path),
            admitted_paths=ADMITTED,
        )
    )
    assert result.disposition is CegisDisposition.ABSTAINED
    assert result.certificate is None
    assert CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value in result.reasons
    assert result.grants_write_authority is False
    assert "residual:unresolved_human_legal_review" in result.residual_risks


def test_path_outside_allowlist_rejected() -> None:
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, "seeded/typed_unavailable.py", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=TYPED_UNAVAILABLE_SEED,
            counterexample=_cx(RepairFamily.FALSE_SUCCESS, "vendor/untrusted/module.py"),
            finding=finding,
            admitted_paths=ADMITTED,
        )
    )
    assert result.disposition is CegisDisposition.REJECTED
    assert CegisAbstentionReason.PATH_NOT_ADMITTED.value in result.reasons
    assert result.certificate is None


def test_scope_escape_attack_fails() -> None:
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, "seeded/typed_unavailable.py", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=TYPED_UNAVAILABLE_SEED,
            counterexample=_cx(RepairFamily.FALSE_SUCCESS, "seeded/typed_unavailable.py"),
            finding=finding,
            admitted_paths=ADMITTED,
            proposal_hint={
                "extra_paths": ["../../../etc/passwd"],
                "import_additions": ["subprocess"],
                "new_dependencies": ["evil"],
            },
        )
    )
    assert result.disposition is CegisDisposition.REJECTED
    assert CegisAbstentionReason.SCOPE_ESCAPE.value in result.reasons
    assert result.certificate is None


def test_authority_claim_attack_fails() -> None:
    finding = _finding_for_rule(
        CANONICAL_CID_SEED, "seeded/canonical_cid.py", IpaRuleId.PSEUDO_CID
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=CANONICAL_CID_SEED,
            counterexample=_cx(RepairFamily.PSEUDO_CID, "seeded/canonical_cid.py"),
            finding=finding,
            admitted_paths=ADMITTED,
            proposal_hint={
                "write_authority": True,
                "promote_patch": True,
                "grants_proof_authority": True,
            },
        )
    )
    assert result.disposition is CegisDisposition.REJECTED
    assert CegisAbstentionReason.AUTHORITY_CLAIM.value in result.reasons
    assert result.certificate is None


def test_obligation_waiver_attack_fails() -> None:
    finding = _finding_for_rule(
        SIMULATION_SEED, "seeded/simulation_evidence.py", IpaRuleId.MOCK_TO_PRODUCTION
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=SIMULATION_SEED,
            counterexample=_cx(RepairFamily.MOCK_CAPABILITY, "seeded/simulation_evidence.py"),
            finding=finding,
            admitted_paths=ADMITTED,
            proposal_hint={"waive_obligation": True, "force_admit": True, "skip_proof": True},
        )
    )
    assert result.disposition is CegisDisposition.REJECTED
    assert CegisAbstentionReason.OBLIGATION_WAIVER.value in result.reasons


def test_grammar_expansion_attack_fails() -> None:
    finding = _finding_for_rule(
        EXPLICIT_INIT_SEED, "seeded/explicit_init.py", IpaRuleId.IMPORT_EFFECT
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=EXPLICIT_INIT_SEED,
            counterexample=_cx(RepairFamily.IMPORT_EFFECT, "seeded/explicit_init.py"),
            finding=finding,
            admitted_paths=ADMITTED,
            proposal_hint={"grammar_expansion": True, "new_transform": "freeform_llm_edit"},
        )
    )
    assert result.disposition in {CegisDisposition.REJECTED, CegisDisposition.ABSTAINED}
    assert (
        CegisAbstentionReason.TRANSFORM_OUTSIDE_GRAMMAR.value in result.reasons
        or CegisAbstentionReason.SCOPE_ESCAPE.value in result.reasons
    )


def test_wrong_transform_for_family_abstains() -> None:
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, "seeded/typed_unavailable.py", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=TYPED_UNAVAILABLE_SEED,
            counterexample=_cx(RepairFamily.FALSE_SUCCESS, "seeded/typed_unavailable.py"),
            finding=finding,
            admitted_paths=ADMITTED,
            transform_id=RepairTransformId.CANONICAL_CID,
        )
    )
    assert result.disposition is CegisDisposition.ABSTAINED
    assert CegisAbstentionReason.PRECONDITION_MISMATCH.value in result.reasons


def test_new_abstract_counterexample_is_rejected() -> None:
    # Craft a browser seed that "repairs" granted but introduces a fresh default-granted.
    # The grammar renderer removes granted; instead force a post-condition via markers
    # that remain after a no-op-like precondition mismatch path using stale markers.
    path = "seeded/browser_bad.py"
    seed = "UNRELATED = 'x'\n# no consent default present\n"
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=seed,
            counterexample=_cx(
                RepairFamily.BROWSER_AUTHORITY,
                path,
                abstract_markers=("granted",),
            ),
            admitted_paths=ADMITTED,
        )
    )
    assert result.certificate is None
    assert result.disposition is CegisDisposition.ABSTAINED
    assert CegisAbstentionReason.PRECONDITION_MISMATCH.value in result.reasons


def test_stale_proof_cache_hit_force_fails_proof_gate() -> None:
    path = "seeded/stale_proof_force.json"
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=STALE_PROOF_SEED,
            counterexample=_cx(
                RepairFamily.STALE_PROOF,
                path,
                abstract_markers=(
                    "reuse_stale_proof",
                    "historical_receipt_as_live",
                    '"status": "live"',
                ),
                metadata={"force_stale_cache_hit": True},
            ),
            admitted_paths=ADMITTED,
        )
    )
    assert result.certificate is None
    assert CegisAbstentionReason.STALE_PROOF_REUSE.value in result.reasons


def test_isolated_transaction_does_not_grant_write_authority() -> None:
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, "seeded/typed_unavailable.py", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=TYPED_UNAVAILABLE_SEED,
            counterexample=_cx(RepairFamily.FALSE_SUCCESS, "seeded/typed_unavailable.py"),
            finding=finding,
            admitted_paths=ADMITTED,
        )
    )
    assert result.certified
    assert result.transaction is not None
    assert result.transaction.committed
    assert result.transaction.overlay
    assert result.grants_write_authority is False
    assert result.certificate is not None
    assert result.certificate.grants_write_authority is False


def test_mutation_benchmark_kills_scope_and_authority_attacks() -> None:
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, "seeded/typed_unavailable.py", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    request = CegisRepairRequest(
        source=TYPED_UNAVAILABLE_SEED,
        counterexample=_cx(RepairFamily.FALSE_SUCCESS, "seeded/typed_unavailable.py"),
        finding=finding,
        admitted_paths=ADMITTED,
    )
    report = run_mutation_benchmark(request)
    assert report.schema == BENCHMARK_SCHEMA
    assert report.survived == 0
    assert report.killed == len(report.cases)
    assert report.score == 1.0
    assert {case.mutant_id for case in report.cases} >= {
        "mutant:obligation-waiver",
        "mutant:scope-escape",
        "mutant:authority-claim",
        "mutant:path-not-admitted",
        "mutant:grammar-expansion",
    }


def test_facade_repair_and_benchmark() -> None:
    engine = default_cegis()
    assert isinstance(engine, FormalAssuranceCegis)
    finding = _finding_for_rule(
        CANONICAL_CID_SEED, "seeded/canonical_cid.py", IpaRuleId.PSEUDO_CID
    )
    request = CegisRepairRequest(
        source=CANONICAL_CID_SEED,
        counterexample=_cx(RepairFamily.PSEUDO_CID, "seeded/canonical_cid.py"),
        finding=finding,
        admitted_paths=ADMITTED,
    )
    result = engine.repair(request)
    assert result.certified
    report = engine.benchmark(request)
    assert report.survived == 0
    assert len(engine.grammar()) == len(RepairFamily)


def test_certificate_content_id_is_stable() -> None:
    finding = _finding_for_rule(
        EXPLICIT_INIT_SEED, "seeded/explicit_init.py", IpaRuleId.IMPORT_EFFECT
    )
    request = CegisRepairRequest(
        source=EXPLICIT_INIT_SEED,
        counterexample=_cx(RepairFamily.IMPORT_EFFECT, "seeded/explicit_init.py"),
        finding=finding,
        admitted_paths=ADMITTED,
    )
    first = run_bounded_cegis(request)
    second = run_bounded_cegis(request)
    assert first.certified and second.certified
    assert first.certificate is not None and second.certificate is not None
    assert first.certificate.certificate_id == second.certificate.certificate_id
    assert first.certificate.after_hash == second.certificate.after_hash


def test_lease_recovery_trace_gate_requires_no_blind_retry() -> None:
    result = run_bounded_cegis(
        CegisRepairRequest(
            source=LEASE_RECOVERY_SEED,
            counterexample=_cx(
                RepairFamily.MISSING_LEASE_RECOVERY,
                "seeded/lease_recovery.cfg",
                abstract_markers=("blind_retry=true", "allow_blind_retry"),
            ),
            admitted_paths=ADMITTED,
        )
    )
    assert result.certified
    after = result.candidates_tried[-1].after_source
    assert "no_blind_unknown_retry=true" in after
    assert "REQUIRE_LEASE_FENCE_RECOVERY=true" in after
    assert "blind_retry=true" not in after
