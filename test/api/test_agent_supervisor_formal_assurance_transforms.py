"""FACP-043: Bounded IPA repair transforms and mutation gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    IpaRuleId,
    analyze_python_source,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.formal_assurance_transforms import (
    BUNDLE,
    DETERMINISTIC_REPAIRS_EVIDENCE,
    EVIDENCE_ID,
    GOAL_ID,
    INTERFACE,
    SCHEMA,
    STABLE_TRANSFORM_IDS,
    TASK_ID,
    TRANSFORM_TO_RULE,
    IpaRepairAbstentionReason,
    IpaRepairDisposition,
    IpaRepairTransformId,
    apply_ipa_repair,
    apply_ipa_repair_idempotent,
    default_admitted_paths,
    evaluate_mutation_gate,
    list_transform_grammar,
    path_is_admitted,
    select_transform,
)

UNRELATED = "keep-me-bytes-xyz-facp043"

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

AMBIGUOUS_IMPORT_SEED = f'''\
import os
import subprocess
UNRELATED = "{UNRELATED}"
os.environ["IPFS_DATASETS_AUTO_INSTALL"] = "true"
subprocess.run(["echo", "bootstrap"], check=False)
'''


def _finding_for_rule(source: str, path: str, rule: IpaRuleId):
    findings = analyze_python_source(source, path=path)
    matched = [item for item in findings if item.rule_id == rule.value]
    assert matched, f"expected {rule.value} in {[f.rule_id for f in findings]}"
    # Prefer a non-datalog-derived finding when available.
    primary = [item for item in matched if "hermetic datalog" not in item.message]
    return (primary or matched)[0]


def test_contract_constants_and_closed_grammar() -> None:
    assert TASK_ID == "FACP-043"
    assert GOAL_ID == "FACP-G410"
    assert BUNDLE == "facp/static/repairs"
    assert SCHEMA == "facp/ipa-repair@1"
    assert EVIDENCE_ID == "facp/ipa-repair@1"
    assert DETERMINISTIC_REPAIRS_EVIDENCE == "facp/deterministic-repairs@1"
    assert INTERFACE == "FormalAssuranceIpaTransforms@1"
    assert STABLE_TRANSFORM_IDS == {
        "explicit_init",
        "typed_unavailable",
        "simulation_evidence",
        "canonical_cid",
        "critical_error_propagation",
    }
    grammar = list_transform_grammar()
    assert len(grammar) == 5
    assert {row["transform_id"] for row in grammar} == STABLE_TRANSFORM_IDS
    for transform_id, rule in TRANSFORM_TO_RULE.items():
        assert select_transform(rule.value) is transform_id


@pytest.mark.parametrize(
    ("seed", "rule", "transform", "path"),
    [
        (EXPLICIT_INIT_SEED, IpaRuleId.IMPORT_EFFECT, IpaRepairTransformId.EXPLICIT_INIT, "seeded/explicit_init.py"),
        (TYPED_UNAVAILABLE_SEED, IpaRuleId.SUCCESS_WITHOUT_OBSERVATION, IpaRepairTransformId.TYPED_UNAVAILABLE, "seeded/typed_unavailable.py"),
        (SIMULATION_SEED, IpaRuleId.MOCK_TO_PRODUCTION, IpaRepairTransformId.SIMULATION_EVIDENCE, "seeded/simulation_evidence.py"),
        (CANONICAL_CID_SEED, IpaRuleId.PSEUDO_CID, IpaRepairTransformId.CANONICAL_CID, "seeded/canonical_cid.py"),
        (CRITICAL_ERROR_SEED, IpaRuleId.EXCEPTION_SWALLOWING, IpaRepairTransformId.CRITICAL_ERROR_PROPAGATION, "seeded/critical_error.py"),
    ],
)
def test_each_transform_removes_seeded_finding_and_preserves_unrelated_bytes(
    seed: str,
    rule: IpaRuleId,
    transform: IpaRepairTransformId,
    path: str,
) -> None:
    finding = _finding_for_rule(seed, path, rule)
    assert select_transform(finding) is transform

    receipt = apply_ipa_repair(
        seed,
        finding,
        path=path,
        transform_id=transform,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.APPLIED
    assert receipt.edits
    assert receipt.before_hash != receipt.after_hash
    assert receipt.mutation_gate is not None and receipt.mutation_gate.admitted
    assert receipt.reanalysis is not None
    assert receipt.reanalysis.target_rule_eliminated
    assert rule.value not in receipt.reanalysis.after_rule_ids
    assert receipt.reanalysis.new_rule_ids == ()
    assert UNRELATED in receipt.after_source
    # Unrelated assignment line remains byte-identical.
    before_marker = f'UNRELATED = "{UNRELATED}"'
    assert before_marker in seed
    assert before_marker in receipt.after_source
    assert receipt.public_compat_preserved

    # Reanalysis independently confirms the seeded rule is gone.
    after_findings = analyze_python_source(receipt.after_source, path=path)
    assert rule.value not in {item.rule_id for item in after_findings}


@pytest.mark.parametrize(
    ("seed", "rule", "path"),
    [
        (EXPLICIT_INIT_SEED, IpaRuleId.IMPORT_EFFECT, "seeded/explicit_init.py"),
        (TYPED_UNAVAILABLE_SEED, IpaRuleId.SUCCESS_WITHOUT_OBSERVATION, "seeded/typed_unavailable.py"),
        (SIMULATION_SEED, IpaRuleId.MOCK_TO_PRODUCTION, "seeded/simulation_evidence.py"),
        (CANONICAL_CID_SEED, IpaRuleId.PSEUDO_CID, "seeded/canonical_cid.py"),
        (CRITICAL_ERROR_SEED, IpaRuleId.EXCEPTION_SWALLOWING, "seeded/critical_error.py"),
    ],
)
def test_transforms_are_deterministic_and_idempotent(
    seed: str,
    rule: IpaRuleId,
    path: str,
) -> None:
    finding = _finding_for_rule(seed, path, rule)
    first_a = apply_ipa_repair(seed, finding, path=path, admitted_paths=ADMITTED)
    first_b = apply_ipa_repair(seed, finding, path=path, admitted_paths=ADMITTED)
    assert first_a.disposition is IpaRepairDisposition.APPLIED
    assert first_a.after_source == first_b.after_source
    assert first_a.after_hash == first_b.after_hash

    first, second = apply_ipa_repair_idempotent(
        seed,
        finding,
        path=path,
        admitted_paths=ADMITTED,
    )
    assert first.disposition is IpaRepairDisposition.APPLIED
    assert second.disposition is IpaRepairDisposition.NOOP
    assert second.after_source == first.after_source
    assert second.before_hash == second.after_hash


def test_ambiguous_target_returns_typed_abstention() -> None:
    path = "seeded/ambiguous_import.py"
    findings = analyze_python_source(AMBIGUOUS_IMPORT_SEED, path=path)
    import_findings = [
        item for item in findings if item.rule_id == IpaRuleId.IMPORT_EFFECT.value
    ]
    assert len(import_findings) >= 2
    # Point the finding at a span that overlaps both effect sites without a unique node.
    finding = import_findings[0]
    # Craft an ambiguous span covering both module-level effects.
    from dataclasses import replace

    broad = replace(
        finding,
        source_span=replace(
            finding.source_span,
            start_line=min(item.source_span.start_line for item in import_findings),
            end_line=max(item.source_span.end_line for item in import_findings),
        ),
    )
    receipt = apply_ipa_repair(
        AMBIGUOUS_IMPORT_SEED,
        broad,
        path=path,
        transform_id=IpaRepairTransformId.EXPLICIT_INIT,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.ABSTAINED
    assert receipt.edits == ()
    assert IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value in receipt.reasons
    assert receipt.after_source in {"", AMBIGUOUS_IMPORT_SEED} or receipt.after_hash in {
        "",
        receipt.before_hash,
    }


def test_path_outside_admitted_allowlist_abstains() -> None:
    path = "vendor/untrusted/module.py"
    finding = _finding_for_rule(EXPLICIT_INIT_SEED, "seeded/explicit_init.py", IpaRuleId.IMPORT_EFFECT)
    receipt = apply_ipa_repair(
        EXPLICIT_INIT_SEED,
        finding,
        path=path,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.ABSTAINED
    assert IpaRepairAbstentionReason.PATH_NOT_ADMITTED.value in receipt.reasons
    assert receipt.edits == ()
    assert not path_is_admitted(path, ADMITTED)


def test_wrong_transform_for_finding_abstains() -> None:
    path = "seeded/typed_unavailable.py"
    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, path, IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    receipt = apply_ipa_repair(
        TYPED_UNAVAILABLE_SEED,
        finding,
        path=path,
        transform_id=IpaRepairTransformId.CANONICAL_CID,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.ABSTAINED
    assert IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value in receipt.reasons
    assert receipt.edits == ()


def test_mutation_gate_requires_byte_mutation_and_reanalysis() -> None:
    path = "seeded/typed_unavailable.py"
    decision = evaluate_mutation_gate(
        path,
        before_source=TYPED_UNAVAILABLE_SEED,
        after_source=TYPED_UNAVAILABLE_SEED,
        target_rule_id=IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value,
        admitted_paths=ADMITTED,
        allow_idempotent_noop=False,
    )
    assert not decision.admitted
    assert IpaRepairAbstentionReason.NO_BYTE_CHANGE.value in decision.reasons

    finding = _finding_for_rule(
        TYPED_UNAVAILABLE_SEED, path, IpaRuleId.SUCCESS_WITHOUT_OBSERVATION
    )
    receipt = apply_ipa_repair(
        TYPED_UNAVAILABLE_SEED,
        finding,
        path=path,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.APPLIED
    assert receipt.mutation_gate is not None
    assert receipt.mutation_gate.byte_mutated
    assert receipt.mutation_gate.reanalyzed


def test_default_admitted_paths_cover_datasets_and_accelerate() -> None:
    defaults = default_admitted_paths()
    assert path_is_admitted(
        "external/ipfs_accelerate/ipfs_accelerate_py/ipfs_accelerate.py", defaults
    )
    assert path_is_admitted(
        "external/ipfs_datasets/ipfs_datasets_py/assurance/initialization.py",
        defaults,
    )
    assert path_is_admitted("seeded/fixture.py", defaults)
    assert not path_is_admitted("../../../etc/passwd", defaults)


def test_receipt_to_dict_round_trip_fields() -> None:
    path = "seeded/explicit_init.py"
    finding = _finding_for_rule(EXPLICIT_INIT_SEED, path, IpaRuleId.IMPORT_EFFECT)
    receipt = apply_ipa_repair(
        EXPLICIT_INIT_SEED,
        finding,
        path=path,
        admitted_paths=ADMITTED,
    )
    payload = receipt.to_dict()
    assert payload["schema"] == SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["disposition"] == "applied"
    assert payload["transform_id"] == "explicit_init"
    assert payload["edits"]
    assert payload["reanalysis"]["target_rule_eliminated"] is True
    assert payload["mutation_gate"]["disposition"] == "admitted"


def test_exact_write_path_is_bound_on_applied_edit() -> None:
    path = "seeded/canonical_cid.py"
    finding = _finding_for_rule(CANONICAL_CID_SEED, path, IpaRuleId.PSEUDO_CID)
    receipt = apply_ipa_repair(
        CANONICAL_CID_SEED,
        finding,
        path=path,
        admitted_paths=ADMITTED,
    )
    assert receipt.disposition is IpaRepairDisposition.APPLIED
    assert receipt.path == path
    assert receipt.edits[0].path == path
    assert "mint_content_identity" in receipt.after_source
