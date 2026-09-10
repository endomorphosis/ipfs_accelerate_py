"""Cross-adapter objective-submission identity parity (DOEP-017 primary).

Proves that the Python client, CLI client, MCP, and MCP++ adapters all mint the
same ObjectiveMaterializationReceipt identities for one SupervisorObjectiveIntent
by delegating to the single canonical intent_service.submit_objective seam.
This module is evidence only: it creates no competing planner, objective store,
auth subsystem, or DuckDB writer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    AuthorityAdmission,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.cli import (
    CANONICAL_CLI_CLIENT,
    CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT,
    submit_objective_cli,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    CANONICAL_PYTHON_CLIENT,
    CANONICAL_PYTHON_CLIENT_ENTRYPOINT,
    submit_objective as python_submit_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
    CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT,
    CANONICAL_OBJECTIVE_SUBMISSION_SERVICE,
    ObjectiveSubmissionPolicyError,
    submit_objective as service_submit_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.objective_resolver import (
    content_addressed_prompt_objective,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    prompt_entrypoints as mcp_adapters,
)
from ipfs_datasets_py.logic.intent_ir.schema import (
    OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
    OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA,
    ObjectiveMaterializationReceipt,
    SupervisorObjectiveIntent,
    SupervisorObjectiveSubmitterKind,
    idea_text_sha256,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]

ADAPTER_SURFACE_PATHS = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "intent_service.py",
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "facade.py",
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "cli.py",
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "mcp_server"
    / "tools"
    / "agent_supervisor_tools"
    / "prompt_entrypoints.py",
)

CANONICAL_SUBMISSION_DELEGATE = (
    "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service.submit_objective"
)
CROSS_ADAPTER_IDENTITY_INTERFACE = "CrossAdapterObjectiveIdentityParity@1"
ADAPTER_TRANSPORTS = (
    "canonical",
    "python",
    "cli",
    "mcp",
    "mcpplusplus",
)


def _minimal_intent(**overrides: Any) -> SupervisorObjectiveIntent:
    idea = "Prove one bounded idea yields identical objective identities across adapters."
    values: dict[str, Any] = {
        "intent_id": "doep.objective.intent.cross-adapter-parity",
        "idea_text": idea,
        "idea_sha256": idea_text_sha256(idea),
        "submitter_kind": SupervisorObjectiveSubmitterKind.HUMAN,
        "caller": "caller:cross-adapter-parity",
        "repository_id": "repository:sha256:cross-adapter-parity",
        "board_namespace": "agent-supervisor-direct-objective-and-event-driven-planning-v1",
        "title_hint": "Cross-adapter identity parity objective",
        "tags": ("direct-objective", "doep", "cross-adapter-identity"),
    }
    values.update(overrides)
    return SupervisorObjectiveIntent(**values)


def _admission(*, delegated: bool = False) -> AuthorityAdmission:
    """Host-shaped DOEP-016 admission record for MCP transport gates."""

    return AuthorityAdmission(
        resolution=SimpleNamespace(authorized=True),
        delegation=object() if delegated else None,
    )


def _receipt_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise AssertionError(f"unreadable receipt payload: {type(value)!r}")
    assert isinstance(payload, dict)
    return payload


def collect_adapter_receipts(
    intent: SupervisorObjectiveIntent | Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Submit one intent through every thin adapter and the canonical service."""

    if intent is None:
        intent = _minimal_intent()
    intent_payload = intent.to_dict() if hasattr(intent, "to_dict") else dict(intent)
    typed_intent = (
        intent
        if isinstance(intent, SupervisorObjectiveIntent)
        else SupervisorObjectiveIntent.from_dict(intent_payload)
    )

    canonical = _receipt_mapping(service_submit_objective(typed_intent))
    python_receipt = _receipt_mapping(python_submit_objective(typed_intent))
    cli_receipt = _receipt_mapping(submit_objective_cli(intent_payload))
    mcp_receipt = _receipt_mapping(
        mcp_adapters.submit_objective_mcp(
            typed_intent,
            admission=_admission(),
        )
    )
    mcpplusplus_receipt = _receipt_mapping(
        mcp_adapters.submit_objective_mcpplusplus(
            typed_intent,
            admission=_admission(delegated=True),
        )
    )
    return {
        "canonical": canonical,
        "python": python_receipt,
        "cli": cli_receipt,
        "mcp": mcp_receipt,
        "mcpplusplus": mcpplusplus_receipt,
    }


def assert_cross_adapter_identity_parity(
    receipts: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    intent: SupervisorObjectiveIntent | None = None,
) -> dict[str, dict[str, Any]]:
    """Assert all adapter receipts share one materialization identity binding."""

    if intent is None:
        intent = _minimal_intent()
    if receipts is None:
        receipts = collect_adapter_receipts(intent)
    assert set(receipts) == set(ADAPTER_TRANSPORTS)

    canonical = dict(receipts["canonical"])
    assert canonical["schema"] == OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA
    assert canonical["intent_id"] == intent.intent_id
    assert canonical["objective_id"] == intent.intent_id
    assert canonical.get("is_completion_authority") in (False, None)
    assert set(canonical).isdisjoint(OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS)

    prompt_cid = cid_for_dag_json(
        {"schema": intent.schema, "intent": intent.to_dict()}
    )
    expected_objective, expected_revision, _pending = content_addressed_prompt_objective(
        prompt_cid,
        repository_id=intent.repository_id,
    )
    assert canonical["objective_cid"] == expected_objective
    assert canonical["objective_revision_cid"] == expected_revision
    assert ObjectiveMaterializationReceipt.from_dict(canonical).receipt_id == (
        canonical["receipt_id"]
    )

    for transport, payload in receipts.items():
        assert dict(payload) == canonical, (
            f"{transport} receipt diverged from canonical objective identity"
        )
    return {name: dict(payload) for name, payload in receipts.items()}


def adapter_surfaces_delegate_to_canonical_service() -> dict[str, str]:
    """Confirm each transport remains a thin delegate over one submission callable."""

    facade_source = ADAPTER_SURFACE_PATHS[1].read_text(encoding="utf-8")
    cli_source = ADAPTER_SURFACE_PATHS[2].read_text(encoding="utf-8")
    mcp_source = ADAPTER_SURFACE_PATHS[3].read_text(encoding="utf-8")

    assert CANONICAL_OBJECTIVE_SUBMISSION_SERVICE == "SupervisorIntentService@1"
    assert CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT == "submit_objective"
    assert CANONICAL_PYTHON_CLIENT == "Supervisor@1"
    assert CANONICAL_PYTHON_CLIENT_ENTRYPOINT == "submit_objective"
    assert CANONICAL_CLI_CLIENT == "SupervisorCLI@1"
    assert CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT == "submit_objective_cli"
    assert (
        mcp_adapters.CANONICAL_MCP_OBJECTIVE_SUBMISSION_ADAPTER
        == "ObjectiveSubmissionMCPAdapter@1"
    )

    assert "from .intent_service import submit_objective" in facade_source
    assert "class PythonClient" not in facade_source
    assert "class DirectObjectiveClient" not in facade_source
    assert "class CanonicalPythonClient" not in facade_source

    assert "from .intent_service import submit_objective" in cli_source
    assert "class ObjectiveSubmissionService" not in cli_source
    assert "class CLIObjectiveStore" not in cli_source

    assert (
        "from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import"
        in mcp_source
    )
    assert "submit_objective as _submit_objective" in mcp_source
    assert "class ObjectiveSubmissionService" not in mcp_source
    assert "class MCPObjectiveStore" not in mcp_source
    assert "FormalPlanCompiler" not in mcp_source

    discovery = mcp_adapters.objective_submission_mcp_discovery_manifest()
    assert discovery["submission_delegate"] == CANONICAL_SUBMISSION_DELEGATE
    assert discovery["completion_authority"] is False
    assert discovery["callers_supply_authoritative_policy"] is False

    return {
        "interface": CROSS_ADAPTER_IDENTITY_INTERFACE,
        "canonical_service": CANONICAL_OBJECTIVE_SUBMISSION_SERVICE,
        "submission_delegate": CANONICAL_SUBMISSION_DELEGATE,
        "python_client": CANONICAL_PYTHON_CLIENT,
        "cli_client": CANONICAL_CLI_CLIENT,
        "mcp_adapter": mcp_adapters.CANONICAL_MCP_OBJECTIVE_SUBMISSION_ADAPTER,
    }


def test_cross_adapter_receipts_share_one_materialization_identity() -> None:
    intent = _minimal_intent()
    receipts = assert_cross_adapter_identity_parity(intent=intent)
    assert receipts["python"]["receipt_id"] == receipts["cli"]["receipt_id"]
    assert receipts["mcp"]["intent_sha256"] == receipts["mcpplusplus"]["intent_sha256"]
    assert receipts["canonical"]["objective_cid"] == receipts["mcp"]["objective_cid"]


def test_identity_is_deterministic_and_authority_gated_only_at_mcp_boundary() -> None:
    intent = _minimal_intent()
    first = assert_cross_adapter_identity_parity(intent=intent)
    second = assert_cross_adapter_identity_parity(intent=intent)
    assert first == second

    with pytest.raises(mcp_adapters.ObjectiveSubmissionMCPError):
        mcp_adapters.submit_objective_mcp(intent, admission=None)
    with pytest.raises(mcp_adapters.ObjectiveSubmissionMCPError):
        mcp_adapters.submit_objective_mcpplusplus(
            intent,
            admission=_admission(delegated=False),
        )
    with pytest.raises(ObjectiveSubmissionPolicyError):
        service_submit_objective(intent, policy_id="policy:forbidden")


def test_adapters_remain_thin_delegates_without_a_competing_subsystem() -> None:
    surfaces = adapter_surfaces_delegate_to_canonical_service()
    assert surfaces["interface"] == CROSS_ADAPTER_IDENTITY_INTERFACE
    assert surfaces["submission_delegate"] == CANONICAL_SUBMISSION_DELEGATE
    for path in ADAPTER_SURFACE_PATHS:
        assert path.is_file()
