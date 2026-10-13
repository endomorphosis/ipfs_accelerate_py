"""Bounded authored CPython observations, never target/benchmark execution."""
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys


def case(name, source, expected, exception=None):
    events = []

    class Stream:
        def __init__(self, label, values):
            self.label, self.values, self.index = label, values, 0

        def __iter__(self):
            events.append("iter:" + self.label)
            if name == "iter_stop_iteration":
                raise StopIteration("authored iter failure")
            return self

        def __next__(self):
            events.append(f"next:{self.label}:{self.index}")
            if name == "next_other_exception":
                raise RuntimeError("authored advance failure")
            if self.index == len(self.values):
                raise StopIteration
            value = self.values[self.index]
            self.index += 1
            return value

    def stream(label, *values):
        events.append("expression:" + label)
        return Stream(label, values)

    def mark(label):
        events.append(label)

    decisions = iter([False] if name == "while_zero" else [True, False])
    class Condition:
        def __bool__(self):
            events.append("truthiness")
            if name == "while_truthiness_exception":
                raise ValueError("authored truthiness failure")
            return next(decisions)

    def condition():
        events.append("condition")
        return Condition()

    error = None
    parsed = False
    try:
        ast.parse(source)
        parsed = True
        exec(compile(source, "<authored-loop-protocol>", "exec"),
             dict(stream=stream, mark=mark, condition=condition))
    except BaseException as exc:
        error = type(exc).__name__
    if events != expected or error != exception:
        raise AssertionError(dict(name=name, actual=events, expected=expected, exception=error))
    return dict(name=name, source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                ast_parse_succeeded=parsed, events=events, exception=error, passed=True)


def main():
    for_source = "for item in stream('x'{values}):\n    {body}\nelse:\n    mark('else')\n"
    rows = [
        case("zero_iteration", for_source.format(values="", body="mark('body')"),
             ["expression:x", "iter:x", "next:x:0", "else"]),
        case("normal_exhaustion", for_source.format(values=", 1", body="mark('body')"),
             ["expression:x", "iter:x", "next:x:0", "body", "next:x:1", "else"]),
        case("continue", for_source.format(values=", 1", body="mark('body'); continue"),
             ["expression:x", "iter:x", "next:x:0", "body", "next:x:1", "else"]),
        case("break", for_source.format(values=", 1, 2", body="mark('body'); break"),
             ["expression:x", "iter:x", "next:x:0", "body"]),
        case("iter_stop_iteration", for_source.format(values=", 1", body="mark('body')"),
             ["expression:x", "iter:x"], "StopIteration"),
        case("next_other_exception", for_source.format(values=", 1", body="mark('body')"),
             ["expression:x", "iter:x", "next:x:0"], "RuntimeError"),
        case("body_stop_iteration", for_source.format(values=", 1", body="mark('body'); raise StopIteration"),
             ["expression:x", "iter:x", "next:x:0", "body"], "StopIteration"),
        case("target_binding_exception", "for left, right in stream('x', [1]):\n    mark('body')\nelse:\n    mark('else')\n",
             ["expression:x", "iter:x", "next:x:0"], "ValueError"),
        case("nested_activation", "for outer in stream('outer', 1, 2):\n    for inner in stream('inner'):\n        mark('body')\n    else:\n        mark('inner_else')\nelse:\n    mark('outer_else')\n",
             ["expression:outer", "iter:outer", "next:outer:0", "expression:inner", "iter:inner", "next:inner:0", "inner_else", "next:outer:1", "expression:inner", "iter:inner", "next:inner:0", "inner_else", "next:outer:2", "outer_else"]),
        case("inner_else_breaks_outer", "for outer in stream('outer', 1, 2):\n    for inner in stream('inner'):\n        mark('body')\n    else:\n        mark('inner_else'); break\nelse:\n    mark('outer_else')\n",
             ["expression:outer", "iter:outer", "next:outer:0", "expression:inner", "iter:inner", "next:inner:0", "inner_else"]),
        case("inner_else_continues_outer", "for outer in stream('outer', 1):\n    for inner in stream('inner'):\n        mark('body')\n    else:\n        mark('inner_else'); continue\n    mark('unreached')\nelse:\n    mark('outer_else')\n",
             ["expression:outer", "iter:outer", "next:outer:0", "expression:inner", "iter:inner", "next:inner:0", "inner_else", "next:outer:1", "outer_else"]),
        case("while_zero", "while condition():\n    mark('body')\nelse:\n    mark('else')\n",
             ["condition", "truthiness", "else"]),
        case("while_continue", "while condition():\n    mark('body'); continue\nelse:\n    mark('else')\n",
             ["condition", "truthiness", "body", "condition", "truthiness", "else"]),
        case("while_truthiness_exception", "while condition():\n    mark('body')\nelse:\n    mark('else')\n",
             ["condition", "truthiness"], "ValueError"),
        case("for_else_invalid_break", "for item in stream('x'):\n    pass\nelse:\n    break\n", [], "SyntaxError"),
        case("while_else_invalid_continue", "while condition():\n    pass\nelse:\n    continue\n", [], "SyntaxError"),
    ]
    report = dict(schema="authored-python-loop-protocol-observation@1", implementation=platform.python_implementation(),
                  python=sys.version, passed=len(rows), cases=rows,
                  executed_authored_examples_only=True, benchmark_source_executed=False,
                  source_semantics_proven=False, numerical_semantics_proven=False,
                  source_obligations_discharged=0, provider_calls=0, solver_calls=0,
                  references=["https://docs.python.org/3.12/reference/compound_stmts.html#the-for-statement",
                              "https://docs.python.org/3.12/reference/simple_stmts.html#the-break-statement",
                              "https://docs.python.org/3.12/library/stdtypes.html#iterator-types"],
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = Path(__file__).with_name("authored-loop-protocol-observation.json")
    with target.open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(dict(path=str(target), passed=len(rows), benchmark_source_executed=False)))


if __name__ == "__main__":
    main()
