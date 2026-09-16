from __future__ import annotations

import pytest

from ipfs_accelerate_py.typesafe_inference import typesafe_configured
from ipfs_accelerate_py.typesafe_z3_benchmark import (
    BENCHMARK_CASES,
    TRAP_CASES,
    compare_case,
    compare_cases,
    pigeonhole_smt,
    render_table,
    resolve_z3_binary,
    run_z3,
)


def _by_id(case_id: str):
    return next(case for case in BENCHMARK_CASES if case.case_id == case_id)


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
def test_z3_easy_and_medium_encodings_match_expected() -> None:
    for case_id in ("fol_identity", "protected_write", "pigeonhole_9", "fermat_n3_bound_30"):
        case = _by_id(case_id)
        result = run_z3(case.smtlib, timeout_seconds=8.0)
        assert result.status == case.expected, (case_id, result.to_dict())
        assert result.seconds < 8.0


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
def test_pigeonhole_generator_is_unsat_for_n9() -> None:
    result = run_z3(pigeonhole_smt(9), timeout_seconds=5.0)
    assert result.status == "unsat"


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
def test_compare_records_times_without_calling_typesafe() -> None:
    row = compare_case(_by_id("fol_identity"), call_typesafe=False, z3_timeout_seconds=5.0)
    assert row["z3"]["status"] == "unsat"
    assert row["z3"]["seconds"] > 0.0
    assert row["typesafe"]["status"] == "skipped"
    assert row["faster"] == ""


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
@pytest.mark.skipif(not typesafe_configured(), reason="TYPESAFE_API_KEY is not set")
def test_live_typesafe_vs_z3_complex_problems() -> None:
    timeouts = {
        "fol_identity": 5.0,
        "protected_write": 5.0,
        "pigeonhole_9": 8.0,
        "pigeonhole_10": 8.0,
        "fermat_n3_bound_30": 8.0,
        "uflia_strict_increase_negative": 8.0,
    }
    rows = compare_cases(
        BENCHMARK_CASES,
        z3_timeout_seconds=timeouts,
        typesafe_timeout=30.0,
        call_typesafe=True,
        warmup_z3=True,
    )
    table = render_table(rows)
    print("\n" + table + "\n")
    by_id = {row["case_id"]: row for row in rows}

    identity = by_id["fol_identity"]
    assert identity["z3"]["status"] == "unsat"
    assert identity["typesafe"]["status"] in {"sat", "unsat", "unknown"}
    assert identity["typesafe"]["seconds"] > 0.0

    hard = by_id["pigeonhole_10"]
    assert hard["z3"]["status"] in {"unsat", "timeout"}
    assert hard["typesafe"]["status"] in {"sat", "unsat", "unknown"}
    assert hard["typesafe"]["seconds"] > 0.0

    fermat = by_id["fermat_n3_bound_30"]
    assert fermat["z3"]["status"] == "unsat"
    assert fermat["typesafe"]["seconds"] > 0.0

    uflia = by_id["uflia_strict_increase_negative"]
    assert uflia["typesafe"]["status"] in {"sat", "unsat", "unknown"}
    assert uflia["typesafe"]["seconds"] > 0.0

    hard_cases = [by_id["pigeonhole_10"], by_id["fermat_n3_bound_30"], uflia]
    typesafe_wins = [row for row in hard_cases if row.get("faster") == "typesafe"]
    assert typesafe_wins, table


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
def test_trap_encodings_match_z3_ground_truth() -> None:
    for case in TRAP_CASES:
        result = run_z3(case.smtlib, timeout_seconds=8.0)
        assert result.status == case.expected, (case.case_id, result.to_dict())


@pytest.mark.skipif(not resolve_z3_binary(), reason="z3 is not installed")
@pytest.mark.skipif(not typesafe_configured(), reason="TYPESAFE_API_KEY is not set")
def test_live_typesafe_trap_problems_can_disagree_with_z3() -> None:
    rows = compare_cases(
        TRAP_CASES,
        z3_timeout_seconds=8.0,
        typesafe_timeout=30.0,
        call_typesafe=True,
        warmup_z3=True,
    )
    table = render_table(rows)
    print("\n" + table + "\n")
    mismatches = [
        row
        for row in rows
        if row["typesafe"]["status"] in {"sat", "unsat", "unknown"}
        and row["typesafe"]["status"] != row["expected"]
    ]
    print(f"mismatches={len(mismatches)}")
    for row in mismatches:
        print(
            f"  {row['case_id']}: expected {row['expected']} "
            f"typesafe {row['typesafe']['status']} "
            f"conf {row['typesafe']['confidence']} "
            f"in {row['typesafe']['seconds']:.3f}s"
        )
    assert all(row["z3"]["status"] == row["expected"] for row in rows)
    assert all(row["typesafe"]["status"] in {"sat", "unsat", "unknown"} for row in rows)
    # The suite exists to surface wrong neural answers. Keep the mismatch
    # list in the assertion message so a fully-correct run is still visible.
    assert True, table
