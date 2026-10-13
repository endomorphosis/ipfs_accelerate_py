"""Execute authored examples only; this is not a benchmark/source verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys


def run_case(name, source, expected_events, expected_exception=None, *, optimize=0):
    events = []

    class Value:
        def __repr__(self):
            events.append("value.repr")
            if "repr_error" in name:
                raise LookupError("authored conversion failure")
            return "V"

        def __str__(self):
            events.append("value.str")
            return "V"

        def __format__(self, spec):
            events.append("value.format:" + spec)
            if "format_error" in name:
                raise ArithmeticError("authored formatting failure")
            return "V"

    class Spec:
        def __format__(self, spec):
            events.append("spec.format:" + spec)
            return ">4"

    class Condition:
        def __bool__(self):
            events.append("condition.bool")
            if "bool_error" in name:
                raise ValueError("authored ambiguous truth value")
            return "true" in name

    def value():
        events.append("value")
        return Value()

    def spec():
        events.append("spec")
        if "spec_error" in name:
            raise RuntimeError("authored format spec failure")
        return Spec()

    def condition():
        events.append("condition")
        return Condition()

    def then():
        events.append("then")
        return 1

    def otherwise():
        events.append("else")
        return 2

    def later():
        events.append("later")
        return 3

    namespace = dict(value=value, spec=spec, condition=condition, then=then,
                     otherwise=otherwise, later=later)
    error = None
    try:
        exec(compile(source, "<authored-evaluation-probe>", "exec", optimize=optimize), namespace)
    except BaseException as exc:
        error = type(exc).__name__
    passed = events == expected_events and error == expected_exception
    row = dict(name=name, source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
               optimize=optimize, events=events, expected_events=expected_events,
               exception=error, expected_exception=expected_exception, passed=passed)
    if not passed:
        raise AssertionError(row)
    return row


def main():
    cases = [
        ("format_no_conversion", 'result = f"{value():{spec()}}"',
         ["value", "spec", "spec.format:", "value.format:>4"]),
        ("format_repr", 'result = f"{value()!r:{spec()}}"',
         ["value", "spec", "spec.format:", "value.repr"]),
        ("format_str", 'result = f"{value()!s:{spec()}}"',
         ["value", "spec", "spec.format:", "value.str"]),
        ("format_ascii", 'result = f"{value()!a:{spec()}}"',
         ["value", "spec", "spec.format:", "value.repr"]),
        ("format_spec_error", 'result = f"{value()!r:{spec()}} {later()}"',
         ["value", "spec"], "RuntimeError"),
        ("format_repr_error", 'result = f"{value()!r:{spec()}} {later()}"',
         ["value", "spec", "spec.format:", "value.repr"], "LookupError"),
        ("format_error", 'result = f"{value():{spec()}} {later()}"',
         ["value", "spec", "spec.format:", "value.format:>4"], "ArithmeticError"),
        ("format_left_to_right", 'result = f"{value()} {later()}"',
         ["value", "value.format:", "later"]),
        ("conditional_true", "result = then() if condition() else otherwise()",
         ["condition", "condition.bool", "then"]),
        ("conditional_false", "result = then() if condition() else otherwise()",
         ["condition", "condition.bool", "else"]),
        ("conditional_bool_error", "result = then() if condition() else otherwise()",
         ["condition", "condition.bool"], "ValueError"),
        ("assert_true", 'assert condition(), f"{value()!r:{spec()}}"',
         ["condition", "condition.bool"]),
        ("assert_false", 'assert condition(), f"{value()!r:{spec()}}"',
         ["condition", "condition.bool", "value", "spec", "spec.format:", "value.repr"], "AssertionError"),
        ("assert_bool_error", 'assert condition(), f"{value()!r:{spec()}}"',
         ["condition", "condition.bool"], "ValueError"),
        ("assert_format_error", 'assert condition(), f"{value():{spec()}}"',
         ["condition", "condition.bool", "value", "spec", "spec.format:", "value.format:>4"], "ArithmeticError"),
    ]
    rows = [run_case(*case) for case in cases]
    rows.append(run_case("assert_optimized_false", 'assert condition(), f"{value()!r:{spec()}}"',
                         [], optimize=1))
    rows.append(run_case("assert_optimized_bool_error", 'assert condition(), f"{value()!r:{spec()}}"',
                         [], optimize=2))
    report = dict(schema="authored-cpython-evaluation-observation/v1", python=sys.version,
                  implementation=platform.python_implementation(), executed_authored_examples_only=True,
                  benchmark_source_executed=False, provider_calls=0, solver_calls=0,
                  source_semantics_proven=False, numerical_semantics_proven=False,
                  production_ir_conformance_test=False, cases=rows, passed=len(rows),
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  references=[
                      "https://docs.python.org/3.12/reference/expressions.html#conditional-expressions",
                      "https://docs.python.org/3.12/reference/lexical_analysis.html#f-strings",
                      "https://docs.python.org/3.12/reference/simple_stmts.html#the-assert-statement"])
    target = Path(__file__).with_name("authored-evaluation-observation.json")
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(dict(path=str(target), passed=report["passed"], benchmark_source_executed=False)))


if __name__ == "__main__":
    main()
