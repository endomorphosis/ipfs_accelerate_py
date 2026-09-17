"""Timed TypeSafe vs z3 comparison on SMT claims.

TypeSafe is an advisory structured evaluator. z3 is a complete SMT solver.
Times include TypeSafe network latency and z3 process startup. Results never
store API keys.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .typesafe_inference import Choice, Noul, Score, typesafe_configured

DEFAULT_Z3 = "z3"
DEFAULT_Z3_FALLBACK = "/home/barberb/.local/bin/z3"


@dataclass(frozen=True)
class SmtCase:
    case_id: str
    english: str
    smtlib: str
    expected: str
    complexity: str


@dataclass
class SolverTiming:
    engine: str
    status: str
    seconds: float
    timeout: bool = False
    confidence: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "seconds": round(float(self.seconds), 4),
            "timeout": self.timeout,
            "confidence": round(float(self.confidence), 4),
            "usage": dict(self.usage),
            "error": self.error,
        }


def pigeonhole_smt(pigeons: int) -> str:
    holes = pigeons - 1
    lines = ["(set-logic QF_UF)", "(declare-sort P 0)"]
    for i in range(pigeons):
        lines.append(f"(declare-const p{i} P)")
    for j in range(holes):
        lines.append(f"(declare-fun h{j} (P) Bool)")
    for i in range(pigeons):
        lines.append(
            "(assert (or " + " ".join(f"(h{j} p{i})" for j in range(holes)) + "))"
        )
    for j in range(holes):
        for a in range(pigeons):
            for b in range(a + 1, pigeons):
                lines.append(f"(assert (not (and (h{j} p{a}) (h{j} p{b}))))")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


BENCHMARK_CASES: tuple[SmtCase, ...] = (
    SmtCase(
        case_id="fol_identity",
        complexity="easy",
        expected="unsat",
        english=(
            "For an uninterpreted sort U and predicate P, is "
            "not (forall x. P(x) implies P(x)) satisfiable?"
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun P (U) Bool)\n"
            "(assert (not (forall ((x U)) (=> (P x) (P x)))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="protected_write",
        complexity="easy",
        expected="sat",
        english=(
            "Independent propositions Protected, Approved, Write. Is the "
            "negation of (Protected and not Approved) implies not Write "
            "satisfiable? Equivalently, is there a model with Protected, "
            "not Approved, and Write all true?"
        ),
        smtlib=(
            "(set-logic QF_UF)\n"
            "(declare-const Protected Bool)\n"
            "(declare-const Approved Bool)\n"
            "(declare-const Write Bool)\n"
            "(assert (not (=> (and Protected (not Approved)) (not Write))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="pigeonhole_9",
        complexity="medium",
        expected="unsat",
        english=(
            "Place 9 pigeons into 8 holes with at most one pigeon per hole. "
            "Is that assignment satisfiable?"
        ),
        smtlib=pigeonhole_smt(9),
    ),
    SmtCase(
        case_id="pigeonhole_10",
        complexity="hard",
        expected="unsat",
        english=(
            "Place 10 pigeons into 9 holes with at most one pigeon per hole. "
            "Is that assignment satisfiable?"
        ),
        smtlib=pigeonhole_smt(10),
    ),
    SmtCase(
        case_id="fermat_n3_bound_30",
        complexity="hard",
        expected="unsat",
        english=(
            "Do there exist positive integers a, b, c all less than 30 such "
            "that a^3 + b^3 = c^3?"
        ),
        smtlib=(
            "(set-logic QF_NIA)\n"
            "(declare-const a Int)\n"
            "(declare-const b Int)\n"
            "(declare-const c Int)\n"
            "(assert (> a 0))\n"
            "(assert (> b 0))\n"
            "(assert (> c 0))\n"
            "(assert (< c 30))\n"
            "(assert (= (+ (* a a a) (* b b b)) (* c c c)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="uflia_strict_increase_negative",
        complexity="hard",
        expected="sat",
        english=(
            "f maps integers to integers. f(x) > x for every integer x, and "
            "there exists y with f(y) < 0. Is that pair of assertions "
            "satisfiable?"
        ),
        smtlib=(
            "(set-logic UFLIA)\n"
            "(declare-fun f (Int) Int)\n"
            "(assert (forall ((x Int)) (> (f x) x)))\n"
            "(assert (exists ((y Int)) (< (f y) 0)))\n"
            "(check-sat)\n"
        ),
    ),
)

# Claims that look obviously sat/unsat from names or slogans, but z3 disagrees
# with the tempting reading. Used to hunt TypeSafe mistakes.
TRAP_CASES: tuple[SmtCase, ...] = (
    SmtCase(
        case_id="uninterpreted_two_plus_two",
        complexity="trap",
        expected="sat",
        english=(
            "plus is a binary function on an uninterpreted sort, and two and "
            "four are uninterpreted constants. Can plus(two, two) be different "
            "from four?"
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun plus (U U) U)\n"
            "(declare-const two U)\n"
            "(declare-const four U)\n"
            "(assert (distinct (plus two two) four))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="uninterpreted_add_zero",
        complexity="trap",
        expected="sat",
        english=(
            "add and zero are uninterpreted. Is it possible that add(a, zero) "
            "is not equal to a for some a?"
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun add (U U) U)\n"
            "(declare-const zero U)\n"
            "(assert (not (forall ((a U)) (= (add a zero) a))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="uninterpreted_associativity",
        complexity="trap",
        expected="sat",
        english=(
            "f is an uninterpreted binary function. Must f be associative, or "
            "can f(f(x,y),z) differ from f(x,f(y,z))?"
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun f (U U) U)\n"
            "(assert (not (forall ((x U) (y U) (z U)) "
            "(= (f (f x y) z) (f x (f y z))))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="uninterpreted_transitivity",
        complexity="trap",
        expected="sat",
        english=(
            "R is an uninterpreted relation. If R(a,b) and R(b,c) hold, must "
            "R(a,c) hold?"
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun R (U U) Bool)\n"
            "(declare-const a U)\n"
            "(declare-const b U)\n"
            "(declare-const c U)\n"
            "(assert (R a b))\n"
            "(assert (R b c))\n"
            "(assert (not (R a c)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="drinkers_paradox",
        complexity="trap",
        expected="unsat",
        english=(
            "In a nonempty bar, is it possible that nobody is a person such "
            "that if they drink, then everyone drinks? The formula is the "
            "negation of exists x. D(x) implies forall y. D(y)."
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun D (U) Bool)\n"
            "(assert (not (exists ((x U)) (=> (D x) (forall ((y U)) (D y))))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="quantifier_swap_invalid_converse",
        complexity="trap",
        expected="sat",
        english=(
            "If for every x there is a y with R(x,y), does there have to be "
            "one y that works for every x? This asserts that the converse "
            "implication fails."
        ),
        smtlib=(
            "(set-logic UF)\n"
            "(declare-sort U 0)\n"
            "(declare-fun R (U U) Bool)\n"
            "(assert (not (=> (forall ((x U)) (exists ((y U)) (R x y)))\n"
            "                 (exists ((y U)) (forall ((x U)) (R x y))))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="bv8_everyone_has_greater",
        complexity="trap",
        expected="unsat",
        english=(
            "For 8-bit unsigned bit-vectors, does every x have some y with "
            "y > x? This is true for unbounded integers."
        ),
        smtlib=(
            "(set-logic BV)\n"
            "(assert (forall ((x (_ BitVec 8))) "
            "(exists ((y (_ BitVec 8))) (bvugt y x))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="int_everyone_has_greater",
        complexity="trap",
        expected="sat",
        english=(
            "For unbounded integers, does every x have some y with y > x?"
        ),
        smtlib=(
            "(set-logic LIA)\n"
            "(assert (forall ((x Int)) (exists ((y Int)) (> y x))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="pythagorean_under_30",
        complexity="trap",
        expected="sat",
        english=(
            "Do there exist positive integers a, b, c all less than 30 such "
            "that a^2 + b^2 = c^2? Compare with Fermat cubes."
        ),
        smtlib=(
            "(set-logic QF_NIA)\n"
            "(declare-const a Int)\n"
            "(declare-const b Int)\n"
            "(declare-const c Int)\n"
            "(assert (> a 0))\n"
            "(assert (> b 0))\n"
            "(assert (> c 0))\n"
            "(assert (< c 30))\n"
            "(assert (= (+ (* a a) (* b b)) (* c c)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="naturals_all_have_predecessor",
        complexity="trap",
        expected="unsat",
        english=(
            "Every natural number n >= 0 has a natural predecessor m with "
            "n = m + 1. Is that true of the integers restricted to n >= 0?"
        ),
        smtlib=(
            "(set-logic LIA)\n"
            "(assert (forall ((n Int)) (=> (>= n 0) "
            "(exists ((m Int)) (and (>= m 0) (= n (+ m 1)))))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="even_prime_greater_than_two",
        complexity="trap",
        expected="unsat",
        english=(
            "Is there an even prime integer p with 2 < p < 30?"
        ),
        smtlib=(
            "(set-logic LIA)\n"
            "(assert (exists ((p Int))\n"
            "  (and (> p 2) (< p 30) (= (mod p 2) 0)\n"
            "       (forall ((d Int)) "
            "(=> (and (> d 1) (< d p)) (not (= (mod p d) 0)))))))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="implication_not_symmetric",
        complexity="trap",
        expected="sat",
        english=(
            "If P implies Q, must Q imply P? This asserts P=>Q and not (Q=>P)."
        ),
        smtlib=(
            "(set-logic QF_UF)\n"
            "(declare-const P Bool)\n"
            "(declare-const Q Bool)\n"
            "(assert (=> P Q))\n"
            "(assert (not (=> Q P)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="pigeonhole_4_into_4",
        complexity="trap",
        expected="sat",
        english=(
            "Place 4 pigeons into 4 holes with at most one pigeon per hole. "
            "The pigeonhole principle often makes people answer unsat."
        ),
        smtlib=(
            "(set-logic QF_LIA)\n"
            "(declare-const p0 Int)\n"
            "(declare-const p1 Int)\n"
            "(declare-const p2 Int)\n"
            "(declare-const p3 Int)\n"
            "(assert (and (>= p0 0) (< p0 4)))\n"
            "(assert (and (>= p1 0) (< p1 4)))\n"
            "(assert (and (>= p2 0) (< p2 4)))\n"
            "(assert (and (>= p3 0) (< p3 4)))\n"
            "(assert (distinct p0 p1 p2 p3))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="integer_two_plus_two",
        complexity="trap",
        expected="unsat",
        english="Over the integers, can 2 + 2 be distinct from 4?",
        smtlib=(
            "(set-logic QF_LIA)\n"
            "(assert (distinct (+ 2 2) 4))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="float64_point_one_plus_point_two",
        complexity="trap",
        expected="unsat",
        english=(
            "In IEEE-754 binary64, is fp.eq(0.1 + 0.2, 0.3) true? The "
            "assertions require that rounded 0.1 plus rounded 0.2 equals "
            "rounded 0.3."
        ),
        smtlib=(
            "(set-logic QF_FP)\n"
            "(define-fun a () (_ FloatingPoint 11 53) ((_ to_fp 11 53) RNE 0.1))\n"
            "(define-fun b () (_ FloatingPoint 11 53) ((_ to_fp 11 53) RNE 0.2))\n"
            "(define-fun c () (_ FloatingPoint 11 53) ((_ to_fp 11 53) RNE 0.3))\n"
            "(assert (fp.eq (fp.add RNE a b) c))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="float32_point_one_plus_point_two",
        complexity="trap",
        expected="sat",
        english=(
            "In IEEE-754 binary32, is fp.eq(0.1 + 0.2, 0.3) true? People who "
            "know the float meme often answer the same way for every width."
        ),
        smtlib=(
            "(set-logic QF_FP)\n"
            "(define-fun a () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.1))\n"
            "(define-fun b () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.2))\n"
            "(define-fun c () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.3))\n"
            "(assert (fp.eq (fp.add RNE a b) c))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="bv8_signed_square_can_be_negative",
        complexity="trap",
        expected="sat",
        english=(
            "Over 8-bit two's-complement bit-vectors, can x * x be strictly "
            "negative as a signed value? Squares are nonnegative in Z."
        ),
        smtlib=(
            "(set-logic QF_BV)\n"
            "(declare-fun x () (_ BitVec 8))\n"
            "(assert (bvslt (bvmul x x) (_ bv0 8)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="integer_twice_equals_one",
        complexity="trap",
        expected="unsat",
        english="Does there exist an integer n such that 2*n = 1?",
        smtlib=(
            "(set-logic QF_LIA)\n"
            "(declare-const n Int)\n"
            "(assert (= (* 2 n) 1))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="real_twice_equals_one",
        complexity="trap",
        expected="sat",
        english="Does there exist a real n such that 2*n = 1?",
        smtlib=(
            "(set-logic QF_LRA)\n"
            "(declare-const n Real)\n"
            "(assert (= (* 2.0 n) 1.0))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="three_cubes_equal_sixth_cube",
        complexity="trap",
        expected="sat",
        english=(
            "Fermat says two positive cubes never sum to a cube. Can three "
            "positive cubes sum to a cube, specifically 3^3+4^3+5^3 = 6^3?"
        ),
        smtlib=(
            "(set-logic QF_LIA)\n"
            "(assert (= (+ (* 3 3 3) (* 4 4 4) (* 5 5 5)) (* 6 6 6)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="nine_ten_twelve_cubes_near_miss",
        complexity="trap",
        expected="unsat",
        english=(
            "Is 9^3 + 10^3 equal to 12^3? 1729 is nearby (taxicab number), "
            "and 12^3 is 1728."
        ),
        smtlib=(
            "(set-logic QF_LIA)\n"
            "(assert (= (+ (* 9 9 9) (* 10 10 10)) (* 12 12 12)))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="integer_square_equals_two",
        complexity="trap",
        expected="unsat",
        english="Does there exist an integer n with n*n = 2?",
        smtlib=(
            "(set-logic QF_NIA)\n"
            "(declare-const n Int)\n"
            "(assert (= (* n n) 2))\n"
            "(check-sat)\n"
        ),
    ),
    SmtCase(
        case_id="real_square_equals_two",
        complexity="trap",
        expected="sat",
        english="Does there exist a real r with r*r = 2?",
        smtlib=(
            "(set-logic QF_NRA)\n"
            "(declare-const r Real)\n"
            "(assert (= (* r r) 2.0))\n"
            "(check-sat)\n"
        ),
    ),
)


def claim_questions() -> dict[str, Any]:
    return {
        "claim_status": Choice(
            instructions=(
                "What does SMT-LIB check-sat return for this problem? "
                "sat means the assertions can hold together. "
                "unsat means they are contradictory. "
                "unknown if you cannot decide."
            ),
            criteria={
                "sat": "The assertions are simultaneously satisfiable.",
                "unsat": "The assertions are unsatisfiable.",
                "unknown": "The status cannot be decided from the given statement.",
            },
        ),
        "certain": Noul(
            instructions="Are you certain of that check-sat status?",
        ),
        "hardness": Score(
            instructions="How hard is this instance for a complete SMT solver?",
            criteria=["easy", "medium", "hard"],
        ),
    }


def claim_state(case: SmtCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "task": "Answer SMT-LIB check-sat.",
        "english": case.english,
        "smtlib": case.smtlib,
    }


def resolve_z3_binary() -> str:
    found = shutil.which(DEFAULT_Z3) or ""
    if found:
        return found
    from pathlib import Path

    fallback = Path(DEFAULT_Z3_FALLBACK)
    if fallback.is_file():
        return str(fallback)
    return ""


def run_z3(
    smtlib: str,
    *,
    timeout_seconds: float = 15.0,
    binary: str = "",
) -> SolverTiming:
    exe = binary or resolve_z3_binary()
    if not exe:
        return SolverTiming(engine="z3", status="error", seconds=0.0, error="z3 not found")
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe, "-in", "-smt2", f"-T:{max(1, int(timeout_seconds))}"],
            input=str(smtlib).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds) + 1.0,
        )
    except subprocess.TimeoutExpired:
        return SolverTiming(
            engine="z3",
            status="timeout",
            seconds=time.perf_counter() - started,
            timeout=True,
        )
    elapsed = time.perf_counter() - started
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    last = (text.splitlines() or [""])[-1].strip().casefold()
    if last in {"sat", "unsat", "unknown"}:
        status = last
    elif last == "timeout" or "timeout" in last:
        status = "timeout"
    else:
        status = last or "error"
    return SolverTiming(
        engine="z3",
        status=status,
        seconds=elapsed,
        timeout=status == "timeout",
        error="" if status in {"sat", "unsat", "unknown", "timeout"} else text[:240],
    )


def run_typesafe(case: SmtCase, *, timeout: float = 30.0) -> SolverTiming:
    from .typesafe_inference import get_last_typesafe_observation, system_one

    started = time.perf_counter()
    try:
        result = system_one(
            claim_state(case),
            claim_questions(),
            timeout=timeout,
        )
    except Exception as exc:
        return SolverTiming(
            engine="typesafe",
            status="error",
            seconds=time.perf_counter() - started,
            error=type(exc).__name__,
        )
    elapsed = time.perf_counter() - started
    answer = result.choices.get("claim_status")
    status = str(answer.choice if answer is not None else "unknown")
    confidence = float(answer.confidence) if answer is not None else 0.0
    obs = get_last_typesafe_observation()
    usage = {
        "input_tokens": int(obs.get("input_tokens") or result.usage.get("input_tokens") or 0),
        "output_tokens": int(obs.get("output_tokens") or result.usage.get("output_tokens") or 0),
    }
    return SolverTiming(
        engine="typesafe",
        status=status,
        seconds=elapsed,
        confidence=confidence,
        usage=usage,
    )


def compare_case(
    case: SmtCase,
    *,
    z3_timeout_seconds: float = 15.0,
    typesafe_timeout: float = 30.0,
    call_typesafe: Optional[bool] = None,
) -> dict[str, Any]:
    z3 = run_z3(case.smtlib, timeout_seconds=z3_timeout_seconds)
    should = typesafe_configured() if call_typesafe is None else bool(call_typesafe)
    ts = (
        run_typesafe(case, timeout=typesafe_timeout)
        if should
        else SolverTiming(
            engine="typesafe",
            status="skipped",
            seconds=0.0,
            error="TYPESAFE_API_KEY is not set",
        )
    )
    faster = ""
    if ts.status not in {"skipped", "error"} and z3.status not in {"error"}:
        faster = "typesafe" if ts.seconds < z3.seconds else "z3"
    hint_only = z3.status == "timeout" or bool(z3.timeout)
    if ts.status in {"sat", "unsat"} and z3.status in {"sat", "unsat"}:
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
                record_sample,
            )

            record_sample(
                family=case.case_id,
                predicted=ts.status,
                actual=z3.status,
                confidence=float(ts.confidence or 0.0),
                trap_family=case.complexity == "trap",
                case_id=case.case_id,
                smtlib=case.smtlib,
            )
        except Exception:
            pass
    return {
        "case_id": case.case_id,
        "complexity": case.complexity,
        "expected": case.expected,
        "z3": z3.to_dict(),
        "typesafe": ts.to_dict(),
        "typesafe_hint_only": hint_only,
        "z3_matches_expected": z3.status == case.expected,
        "typesafe_matches_expected": ts.status == case.expected,
        "faster": faster,
        "typesafe_speedup": (
            round(z3.seconds / ts.seconds, 3)
            if ts.seconds > 0 and ts.status not in {"skipped", "error"}
            else None
        ),
    }


def compare_cases(
    cases: Optional[Sequence[SmtCase]] = None,
    *,
    z3_timeout_seconds: Mapping[str, float] | float = 15.0,
    typesafe_timeout: float = 30.0,
    call_typesafe: Optional[bool] = None,
    warmup_z3: bool = True,
) -> list[dict[str, Any]]:
    selected = tuple(cases or BENCHMARK_CASES)
    if warmup_z3:
        run_z3("(set-logic QF_UF)\n(check-sat)\n", timeout_seconds=5.0)
    rows = []
    for case in selected:
        if isinstance(z3_timeout_seconds, Mapping):
            budget = float(z3_timeout_seconds.get(case.case_id, 15.0))
        else:
            budget = float(z3_timeout_seconds)
        rows.append(
            compare_case(
                case,
                z3_timeout_seconds=budget,
                typesafe_timeout=typesafe_timeout,
                call_typesafe=call_typesafe,
            )
        )
    return rows


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'case':<34} {'exp':<7} {'z3':<18} {'typesafe':<18} "
        f"{'faster':<10} {'speedup':<8} {'ts_ok':<6}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        z3 = row.get("z3") if isinstance(row.get("z3"), Mapping) else {}
        ts = row.get("typesafe") if isinstance(row.get("typesafe"), Mapping) else {}
        z3_cell = f"{z3.get('status')} {float(z3.get('seconds') or 0):.3f}s"
        ts_cell = f"{ts.get('status')} {float(ts.get('seconds') or 0):.3f}s"
        speedup = row.get("typesafe_speedup")
        lines.append(
            f"{str(row.get('case_id') or ''):<34} "
            f"{str(row.get('expected') or ''):<7} "
            f"{z3_cell:<18} {ts_cell:<18} "
            f"{str(row.get('faster') or '-'):<10} "
            f"{'-' if speedup is None else str(speedup):<8} "
            f"{'yes' if row.get('typesafe_matches_expected') else 'no':<6}"
        )
    return "\n".join(lines)


__all__ = [
    "BENCHMARK_CASES",
    "TRAP_CASES",
    "SmtCase",
    "SolverTiming",
    "claim_questions",
    "claim_state",
    "compare_case",
    "compare_cases",
    "pigeonhole_smt",
    "render_table",
    "resolve_z3_binary",
    "run_typesafe",
    "run_z3",
]
