"""DCR-080 deterministic daemon composition contracts."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.pre_implementation_provider_gate import (
    assert_provider_dispatch_allowed,
    evaluate_provider_gate,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_composition import (
    DeterministicRepairCompositionRoot,
)


class _Service:
    def __init__(self, method: str, result: Any) -> None:
        self.method = method
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        if name != self.method:
            raise AttributeError(name)

        def call(**kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return self.result

        return call


def _root(*, publisher: Any = None, validation: Any = None) -> tuple[DeterministicRepairCompositionRoot, list[_Service]]:
    services = [
        _Service("inspect", {"receipt_cid": "doctor-receipt"}),
        _Service("plan", {"receipt_cid": "planner-receipt"}),
        _Service("prove", {"receipt_cid": "logic-receipt"}),
        _Service("repair", {"receipt_cid": "repair-receipt"}),
        _Service("validate", validation if validation is not None else {"receipt_cid": "validation-receipt"}),
    ]
    root = DeterministicRepairCompositionRoot(
        doctor=services[0], planner=services[1], logic=services[2], repair=services[3],
        validator=services[4], publisher=publisher,
    )
    return root, services


def test_close_requires_repair_and_publication_receipts() -> None:
    publisher = _Service("publish", {"receipt_cid": "publication-receipt"})
    root, services = _root(publisher=publisher)

    result = root.run(task_id="DCR-080")

    assert result.disposition == "closed_deterministic"
    assert result.service_receipts["repair"] == "repair-receipt"
    assert result.service_receipts["publication"] == "publication-receipt"
    assert all(service.calls for service in services)
    assert publisher.calls
    assert result.receipt_cid


def test_proved_valid_observation_can_close_without_publication() -> None:
    root, _ = _root(validation={"receipt_cid": "proof-observation", "proved_valid": True})

    result = root.run(task_id="DCR-080")

    assert result.disposition == "closed_deterministic"
    assert result.proved_valid is True
    assert "publication" not in result.service_receipts


def test_missing_service_receipt_abstains_and_never_dispatches_provider() -> None:
    root, _ = _root(publisher=_Service("publish", {}))

    result = root.run(task_id="DCR-080")
    gate = evaluate_provider_gate(
        task_id="DCR-080", service_receipt_ids=tuple(result.service_receipts.values()),
        reason_codes=result.reason_codes,
    )

    assert result.disposition == "abstain"
    assert "missing_publication_service_receipt" in result.reason_codes
    assert gate.provider_authorized is False
    assert gate.skip_provider is True
    with pytest.raises(PermissionError, match="forbidden"):
        assert_provider_dispatch_allowed(gate)
