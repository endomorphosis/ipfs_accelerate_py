"""ASEH qualification writer used by 070-073 without recaiming ASEH-013."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.validation.paired_qualification_cli import (
    main as qualification_main,
    run_qualification,
)


def test_hermetic_qualification_writes_non_live_receipts(tmp_path: Path) -> None:
    output = tmp_path / "hermetic.json"
    qualification_output = tmp_path / "hermetic_qualification.json"
    ran = run_qualification(
        cohort="hermetic",
        output=output,
        qualification_output=qualification_output,
        minimum_tasks=60,
        allow_honest_nonpromotion=True,
        validator_command=(
            "python3",
            "benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py",
            "--cohort",
            "hermetic",
            "--minimum-tasks",
            "60",
            "--output",
            str(output),
            "--qualification-output",
            str(qualification_output),
            "--allow-honest-nonpromotion",
        ),
    )
    results = json.loads(output.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_output.read_text(encoding="utf-8"))
    assert ran["pair_count"] >= 60
    assert results["live"] is False
    assert results["authority"] is False
    assert results["hermetic_sufficient_for_production_promotion"] is False
    assert qualification["live"] is False
    assert qualification["production_qualified"] is False
    assert qualification["promotion_authorized"] is False
    assert qualification["disposition"] in {
        "evidence_qualified",
        "insufficient_evidence",
        "safety_or_quality_failed",
    }
    assert qualification["pair_count"]["truth_state"] == "measured"
    assert qualification["pair_count"]["value"] == ran["pair_count"]
    assert "identity" in qualification
    assert "identity" in results


def test_historical_cohort_is_honest_insufficient_evidence(tmp_path: Path) -> None:
    output = tmp_path / "historical.json"
    qualification_output = tmp_path / "historical_qualification.json"
    status = qualification_main(
        [
            "--cohort",
            "historical",
            "--minimum-tasks",
            "20",
            "--output",
            str(output),
            "--qualification-output",
            str(qualification_output),
            "--allow-honest-nonpromotion",
        ]
    )
    assert status == 0
    qualification = json.loads(qualification_output.read_text(encoding="utf-8"))
    assert qualification["disposition"] == "insufficient_evidence"
    assert qualification["live"] is False
    assert qualification["qualification"] is False
    assert "historical_corpus_unavailable_on_current_tree" in qualification["reasons"]


def test_canary_without_shadow_is_not_admitted(tmp_path: Path) -> None:
    output = tmp_path / "canary.json"
    qualification_output = tmp_path / "canary_qualification.json"
    status = qualification_main(
        [
            "--cohort",
            "canary",
            "--output",
            str(output),
            "--qualification-output",
            str(qualification_output),
            "--allow-not-admitted",
        ]
    )
    assert status == 0
    qualification = json.loads(qualification_output.read_text(encoding="utf-8"))
    assert qualification["disposition"] == "not_admitted"
    assert "zero_candidate_mutation" in qualification["reasons"]
    assert qualification["live"] is False


def test_013_default_parser_still_rejects_qualification_flags() -> None:
    import importlib.util
    import sys

    harness_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "agent_supervisor"
        / "efficiency_state_hardening"
        / "paired_harness.py"
    )
    spec = importlib.util.spec_from_file_location("aseh_013_parser_probe", harness_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parser = module._build_parser()
    flags = {action.option_strings[0] for action in parser._actions if action.option_strings}
    assert "--write" in flags
    assert "--check" in flags
    assert "--cohort" not in flags
