"""Checked, lazy forwarding tests for the datasets compositional surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    COMPOSITIONAL_VERIFICATION_OPERATIONS,
    EXPECTED_ABSTRACT_ANALYSIS_INTERFACE,
    EXPECTED_ABSTRACT_ANALYSIS_SCHEMA,
    EXPECTED_ASSUME_GUARANTEE_RECEIPT_SCHEMA,
    EXPECTED_COMPOSITIONAL_CONTRACT_SCHEMA,
    EXPECTED_INCREMENTAL_SMT_INTERFACE,
    EXPECTED_INCREMENTAL_SMT_SCHEMA,
    EXPECTED_INCREMENTAL_VERIFICATION_INTERFACE,
    EXPECTED_INCREMENTAL_VERIFICATION_RECEIPT_SCHEMA,
    EXPECTED_INTERPOLATION_RECEIPT_SCHEMA,
    EXPECTED_LOGIC_VERIFICATION_API_INTERFACE,
    IpfsDatasetsSemanticStateProvider,
    SemanticStateAdapterError,
    SemanticStateUnavailable,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _cid(char: str) -> str:
    return "b" + char * 58


class _FakeVerificationAPI:
    interface = EXPECTED_LOGIC_VERIFICATION_API_INTERFACE

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.results = {
            "analyze_abstract_state": SimpleNamespace(
                schema_version=EXPECTED_ABSTRACT_ANALYSIS_SCHEMA,
                INTERFACE=EXPECTED_ABSTRACT_ANALYSIS_INTERFACE,
                analysis_id=_cid("a"),
            ),
            "compile_component_contract": SimpleNamespace(
                schema=EXPECTED_COMPOSITIONAL_CONTRACT_SCHEMA,
                cid=_cid("c"),
            ),
            "discharge_assume_guarantee": SimpleNamespace(
                schema=EXPECTED_ASSUME_GUARANTEE_RECEIPT_SCHEMA,
                receipt_cid=_cid("d"),
            ),
            "plan_incremental_verification": SimpleNamespace(
                schema=EXPECTED_INCREMENTAL_VERIFICATION_RECEIPT_SCHEMA,
                receipt_cid=_cid("e"),
                identity_payload=lambda: {"interface": EXPECTED_INCREMENTAL_VERIFICATION_INTERFACE},
            ),
            "open_incremental_smt_session": SimpleNamespace(
                interface=EXPECTED_INCREMENTAL_SMT_INTERFACE,
                fingerprint=SimpleNamespace(schema=EXPECTED_INCREMENTAL_SMT_SCHEMA),
                add_named_assertion=lambda *args, **kwargs: None,
                push=lambda: None,
                pop=lambda: None,
                check=lambda: None,
                close=lambda: None,
            ),
            "compute_and_validate_interpolant": SimpleNamespace(
                schema=EXPECTED_INTERPOLATION_RECEIPT_SCHEMA,
                receipt_cid=_cid("f"),
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "compositional_verification_operations": list(COMPOSITIONAL_VERIFICATION_OPERATIONS),
        }

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, kwargs))
        return self.results[name]

    def analyze_abstract_state(self, source: str, **kwargs: Any) -> Any:
        return self._call("analyze_abstract_state", source, **kwargs)

    def compile_component_contract(self, contract: Any, **bindings: Any) -> Any:
        return self._call("compile_component_contract", contract, **bindings)

    def discharge_assume_guarantee(self, graph: Any, **kwargs: Any) -> Any:
        return self._call("discharge_assume_guarantee", graph, **kwargs)

    def plan_incremental_verification(
        self, previous_state: Any, current_state: Any, **kwargs: Any
    ) -> Any:
        return self._call("plan_incremental_verification", previous_state, current_state, **kwargs)

    def open_incremental_smt_session(self, **kwargs: Any) -> Any:
        return self._call("open_incremental_smt_session", **kwargs)

    def compute_and_validate_interpolant(
        self, partition_a: Any, partition_b: Any, **kwargs: Any
    ) -> Any:
        return self._call("compute_and_validate_interpolant", partition_a, partition_b, **kwargs)


def test_all_six_operations_forward_without_translating_results() -> None:
    api = _FakeVerificationAPI()
    provider = IpfsDatasetsSemanticStateProvider(verification_api=api)

    observed = (
        provider.analyze_abstract_state("def f(): return 1", source_uri="x.py"),
        provider.compile_component_contract(object(), source_root=_cid("g")),
        provider.discharge_assume_guarantee(object(), max_obligations=2),
        provider.plan_incremental_verification("previous", "current", policy="p"),
        provider.open_incremental_smt_session(session_id="s"),
        provider.compute_and_validate_interpolant("a", "b", theory="QF_LIA"),
    )

    assert observed == tuple(api.results[name] for name in COMPOSITIONAL_VERIFICATION_OPERATIONS)
    assert [call[0] for call in api.calls] == list(COMPOSITIONAL_VERIFICATION_OPERATIONS)
    assert api.calls[0][1] == ("def f(): return 1",)
    assert api.calls[3][1] == ("previous", "current")
    assert api.calls[5][1] == ("a", "b")


def test_verification_api_is_not_resolved_until_operation_call() -> None:
    class _WrongInterface(_FakeVerificationAPI):
        interface = "LogicVerificationAPI@999"

    provider = IpfsDatasetsSemanticStateProvider(verification_api=_WrongInterface())
    # Construction and ordinary object inspection do not touch the extension.
    assert provider._capability is None
    with pytest.raises(SemanticStateUnavailable, match="interface_mismatch"):
        provider.analyze_abstract_state("x = 1")


def test_wrong_datasets_result_schema_fails_closed() -> None:
    api = _FakeVerificationAPI()
    api.results["compile_component_contract"] = SimpleNamespace(
        schema="compositional-contract/v999",
        cid=_cid("c"),
    )
    provider = IpfsDatasetsSemanticStateProvider(verification_api=api)
    with pytest.raises(SemanticStateAdapterError, match="schema"):
        provider.compile_component_contract(object())


def test_current_datasets_analysis_surface_is_lazy_and_checked() -> None:
    provider = IpfsDatasetsSemanticStateProvider()
    result = provider.analyze_abstract_state(
        "def successor(value: int) -> int:\n    return value + 1\n",
        source_uri="fixture.py",
    )
    assert result.INTERFACE == EXPECTED_ABSTRACT_ANALYSIS_INTERFACE
    assert result.schema_version == EXPECTED_ABSTRACT_ANALYSIS_SCHEMA
    assert result.analysis_id.startswith("b")
    # The additive operation does not force the independent semantic-state API.
    assert provider._capability is None


def test_adapter_cold_import_loads_no_datasets_solver_or_model_module() -> None:
    code = f"""
import json
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter
prefixes = (
    "ipfs_datasets_py",
    "z3",
    "cvc5",
    "openai",
    "anthropic",
    "transformers",
)
print(json.dumps(sorted(name for name in sys.modules if name.startswith(prefixes))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []
