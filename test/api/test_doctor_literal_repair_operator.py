"""Safety boundaries of the closed historical literal repair operator."""

import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks/agent_supervisor/container_coding"
sys.path.insert(0, str(DIRECTORY))
try:
    spec = importlib.util.spec_from_file_location(
        "doctor_literal_operator", DIRECTORY / "doctor_literal_repair.py"
    )
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
finally:
    sys.path.remove(str(DIRECTORY))


def test_literal_resolution_never_executes_donor():
    donor = "raise RuntimeError('must never execute')\nA = frozenset({'one'})\nB = frozenset({*A, 'two'})\n"
    assert operator.literal_exports(donor, ["B"]) == {"B": frozenset({"one", "two"})}


@pytest.mark.parametrize(
    "donor",
    [
        "A = __import__('os').getcwd()",
        "A = B\nB = A",
        "A = 'first'\nA = 'second'",
        "A = frozenset('abc')",
        "A = frozenset({1})",
    ],
)
def test_rejects_unsafe_or_ambiguous_literals(donor):
    with pytest.raises(ValueError):
        operator.literal_exports(donor, ["A"])


def test_restoration_preserves_existing_source_and_rejects_repeat():
    first, second = operator.NAMES
    donor = f"{first} = 'schema@2'\n{second} = frozenset({{'a'}})\n"
    caller = f"from .implementation_daemon import {first}, {second}"
    before = b"def existing():\n    return 42\n"
    after, values, _ = operator.propose(before, caller, donor)
    assert after.startswith(before)
    assert values == {first: "schema@2", second: frozenset({"a"})}
    with pytest.raises(ValueError, match="already bound"):
        operator.propose(after, caller, donor)
    with pytest.raises(ValueError, match="does not require"):
        operator.propose(before, "", donor)


@pytest.mark.parametrize(
    "binding", ["{name} = None", "import example as {name}", "def {name}(): pass"]
)
def test_existing_bindings_cannot_be_overwritten(binding):
    caller = "from .implementation_daemon import " + ", ".join(operator.NAMES)
    with pytest.raises(ValueError, match="already bound"):
        operator.propose(binding.format(name=operator.NAMES[0]).encode(), caller, "")
