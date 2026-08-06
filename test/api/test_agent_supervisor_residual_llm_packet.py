"""Fail-closed coverage for ResidualLlmPacket@1 (WPD-002)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import (
    DEFAULT_MAX_CAPSULE_BYTES,
    DEFAULT_MAX_PACKET_BYTES,
    DEFAULT_MAX_PACKET_TOKENS,
    REQUIRED_CORE_FIELDS,
    RESIDUAL_LLM_PACKET_EVIDENCE,
    RESIDUAL_LLM_PACKET_INTERFACE,
    RESIDUAL_LLM_PACKET_SCHEMA,
    ResidualLlmPacket,
    ResidualLlmPacketBudgetError,
    ResidualLlmPacketError,
    ResidualLlmPacketLimits,
    ResidualLlmPacketReason,
    estimate_tokens,
    packet_satisfies_residual_llm_contract,
    residual_llm_packet_from_codex,
    seal_residual_llm_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _capsule(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/counterexample-context-capsule@1",
        "target_ids": ["symbol:target"],
        "counterexamples": [
            {
                "counterexample_id": "cex:wpd-002",
                "kind": "generic_failure",
                "summary": "focused residual repair required",
                "violated_property": "acceptance must hold",
            }
        ],
        "nodes": [],
        "edges": [],
        "usage": {
            "counterexamples": 1,
            "graph_nodes": 0,
            "graph_edges": 0,
            "encoded_bytes": 128,
            "omitted_counterexamples": 0,
        },
        "limits": {"max_bytes": 4096},
        "minimized": True,
        "redacted": True,
        "contains_private_material": False,
        "contains_raw_prover_output": False,
        "contains_source": False,
    }
    base.update(overrides)
    return base


def _seal(**overrides: object) -> ResidualLlmPacket:
    base: dict[str, object] = {
        "task_id": "WPD-002",
        "repository_id": "repository:sha256:wpd-002",
        "tree_id": "tree:wpd-002",
        "write_paths": (
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/residual_llm_packet.py",
        ),
        "obligation_ids": ("obligation:residual-bounds",),
        "counterexample_capsule": _capsule(),
        "validation_commands": (
            "python3 -m pytest external/ipfs_accelerate/test/api/test_agent_supervisor_residual_llm_packet.py -q",
        ),
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:policy-wpd-002",
        "forest_id": "forest:wpd-002",
        "acceptance_ids": ("wpd/residual-llm-packet@1",),
        "authority_roots": {
            "repository_id": "repository:sha256:wpd-002",
            "tree_id": "tree:wpd-002",
        },
    }
    base.update(overrides)
    return seal_residual_llm_packet(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path: required fields, bounds, content identity
# ---------------------------------------------------------------------------


def test_interface_and_evidence_constants() -> None:
    assert RESIDUAL_LLM_PACKET_INTERFACE == "ResidualLlmPacket@1"
    assert RESIDUAL_LLM_PACKET_EVIDENCE == "wpd/residual-llm-packet@1"
    # Align with CodexRepairPacket / ReplanLimits default prompt budgets.
    assert DEFAULT_MAX_PACKET_BYTES == 24_576
    assert DEFAULT_MAX_PACKET_TOKENS == 6_144
    assert DEFAULT_MAX_CAPSULE_BYTES == 16_384
    assert set(REQUIRED_CORE_FIELDS) == {
        "task_id",
        "repository_id",
        "tree_id",
        "write_paths",
        "obligation_ids",
        "counterexample_capsule",
        "validation_commands",
    }


def test_seal_requires_exact_paths_obligations_capsule_and_commands() -> None:
    packet = _seal()
    record = packet.to_dict()

    assert record["schema"] == RESIDUAL_LLM_PACKET_SCHEMA
    assert record["interface"] == RESIDUAL_LLM_PACKET_INTERFACE
    assert record["evidence"] == RESIDUAL_LLM_PACKET_EVIDENCE
    assert record["write_paths"] == [
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/residual_llm_packet.py"
    ]
    assert record["obligation_ids"] == ["obligation:residual-bounds"]
    assert record["validation_commands"]
    assert record["counterexample_capsule"]["counterexamples"]
    assert record["semantic_authority"] is False
    assert record["write_authority"] is False
    assert record["completion_authority"] is False
    assert record["nomination_only"] is True
    assert record["contains_source_body"] is False
    assert record["contains_secrets"] is False
    assert record["limits"]["max_bytes"] == DEFAULT_MAX_PACKET_BYTES
    assert record["limits"]["max_tokens"] == DEFAULT_MAX_PACKET_TOKENS
    assert record["limits"]["max_capsule_bytes"] == DEFAULT_MAX_CAPSULE_BYTES
    assert packet.byte_size <= packet.max_bytes
    assert packet.estimated_tokens <= packet.max_tokens
    assert packet.byte_size <= DEFAULT_MAX_PACKET_BYTES
    assert packet.max_capsule_bytes <= packet.max_bytes


def test_identity_is_content_addressed_and_deterministic() -> None:
    first = _seal()
    second = _seal()
    assert first.packet_id == second.packet_id
    assert first.content_id == first.packet_id
    assert first.packet_id == content_identity(first._identity_payload())

    # Round-trip preserves identity.
    restored = ResidualLlmPacket.from_dict(first.to_dict())
    assert restored.packet_id == first.packet_id
    assert restored.to_dict() == first.to_dict()
    assert packet_satisfies_residual_llm_contract(first) is True


def test_identity_changes_when_required_fields_change() -> None:
    base = _seal()
    altered = _seal(obligation_ids=("obligation:other",))
    assert base.packet_id != altered.packet_id


# ---------------------------------------------------------------------------
# Required-field fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("write_paths", (), ResidualLlmPacketReason.MISSING_WRITE_PATHS),
        ("obligation_ids", (), ResidualLlmPacketReason.MISSING_OBLIGATIONS),
        (
            "validation_commands",
            (),
            ResidualLlmPacketReason.MISSING_VALIDATION_COMMANDS,
        ),
        (
            "counterexample_capsule",
            None,
            ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        ),
        (
            "counterexample_capsule",
            {},
            ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        ),
        (
            "counterexample_capsule",
            {"summary": "no cex"},
            ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        ),
    ],
)
def test_required_fields_fail_closed(
    field: str, value: object, reason: ResidualLlmPacketReason
) -> None:
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(**{field: value})
    assert excinfo.value.reason_code == reason.value


# ---------------------------------------------------------------------------
# Path exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../secrets.env",
        "/etc/passwd",
        "pkg/*.py",
        "pkg/./mod.py",
        "pkg//mod.py",
        "pkg/",
        ".",
    ],
)
def test_non_exact_write_paths_rejected(bad_path: str) -> None:
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(write_paths=(bad_path,))
    assert excinfo.value.reason_code == ResidualLlmPacketReason.PATH_NOT_EXACT.value


def test_blank_write_path_rejected() -> None:
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(write_paths=("",))
    assert excinfo.value.reason_code in {
        ResidualLlmPacketReason.MALFORMED.value,
        ResidualLlmPacketReason.PATH_NOT_EXACT.value,
        ResidualLlmPacketReason.MISSING_WRITE_PATHS.value,
    }


# ---------------------------------------------------------------------------
# Secrets and unbounded source dumps
# ---------------------------------------------------------------------------


def test_secret_keys_in_capsule_rejected() -> None:
    capsule = _capsule()
    capsule["api_key"] = "must-never-enter-the-prompt"
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(counterexample_capsule=capsule)
    assert excinfo.value.reason_code == ResidualLlmPacketReason.SECRET_MATERIAL.value


def test_source_body_dump_rejected() -> None:
    capsule = _capsule()
    capsule["source_body"] = "def huge():\n    pass\n" * 20
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(counterexample_capsule=capsule)
    assert excinfo.value.reason_code == ResidualLlmPacketReason.FORBIDDEN_BODY.value


def test_nested_forbidden_body_rejected() -> None:
    capsule = _capsule()
    capsule["extra"] = {"full_source": "print('nope')"}
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(counterexample_capsule=capsule)
    assert excinfo.value.reason_code == ResidualLlmPacketReason.FORBIDDEN_BODY.value


def test_secret_value_marker_rejected() -> None:
    capsule = _capsule()
    capsule["note"] = "Authorization: Bearer sk-live-secret-token"
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        _seal(counterexample_capsule=capsule)
    assert excinfo.value.reason_code == ResidualLlmPacketReason.SECRET_MATERIAL.value


def test_sealed_packet_excludes_secret_and_source_markers() -> None:
    packet = _seal()
    encoded = packet.to_json().casefold()
    for marker in (
        '"api_key"',
        '"source_body"',
        '"full_source"',
        '"private_witness"',
        '"password"',
        '"raw_output"',
        '"source_code"',
    ):
        assert marker not in encoded
    assert '"contains_source_body":false' in encoded
    assert '"contains_secrets":false' in encoded


# ---------------------------------------------------------------------------
# Size bounds
# ---------------------------------------------------------------------------


def test_packet_budget_enforced() -> None:
    huge_commands = tuple(f"python3 -m pytest test_{i}.py -q" for i in range(40))
    with pytest.raises(ResidualLlmPacketBudgetError) as excinfo:
        _seal(
            validation_commands=huge_commands,
            limits=ResidualLlmPacketLimits(
                max_bytes=1024,
                max_tokens=256,
                max_capsule_bytes=1024,
            ),
        )
    assert excinfo.value.reason_code == ResidualLlmPacketReason.OVER_BUDGET.value


def test_capsule_budget_enforced() -> None:
    capsule = _capsule()
    capsule["counterexamples"] = [
        {
            "counterexample_id": f"cex:{i}",
            "kind": "generic_failure",
            "summary": "x" * 200,
            "violated_property": "y" * 200,
        }
        for i in range(30)
    ]
    with pytest.raises(ResidualLlmPacketBudgetError) as excinfo:
        seal_residual_llm_packet(
            task_id="WPD-002",
            repository_id="repository:sha256:wpd-002",
            tree_id="tree:wpd-002",
            write_paths=("pkg/mod.py",),
            obligation_ids=("obligation:a",),
            counterexample_capsule=capsule,
            validation_commands=("pytest -q",),
            limits=ResidualLlmPacketLimits(
                max_bytes=24_576,
                max_tokens=8_192,
                max_capsule_bytes=1024,
            ),
        )
    assert excinfo.value.reason_code == ResidualLlmPacketReason.OVER_BUDGET.value


def test_estimate_tokens_is_deterministic() -> None:
    assert estimate_tokens(0) == 0
    assert estimate_tokens(1) == 1
    assert estimate_tokens(3) == 1
    assert estimate_tokens(4) == 2


# ---------------------------------------------------------------------------
# Authority hard-zeros
# ---------------------------------------------------------------------------


def test_authority_claims_rejected() -> None:
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        ResidualLlmPacket(
            task_id="WPD-002",
            repository_id="repository:sha256:wpd-002",
            tree_id="tree:wpd-002",
            write_paths=("pkg/mod.py",),
            obligation_ids=("obligation:a",),
            counterexample_capsule=_capsule(),
            validation_commands=("pytest -q",),
            write_authority=True,
        )
    assert excinfo.value.reason_code == ResidualLlmPacketReason.AUTHORITY_CLAIM.value


def test_forged_packet_id_rejected() -> None:
    packet = _seal()
    payload = packet.to_dict()
    payload["packet_id"] = "b" + "a" * 58
    with pytest.raises(ResidualLlmPacketError):
        ResidualLlmPacket.from_dict(payload)


# ---------------------------------------------------------------------------
# CodexRepairPacket alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeTransition:
    counterexample_id: str = "cex:wpd-002"

    def to_dict(self) -> dict[str, str]:
        return {"counterexample_id": self.counterexample_id}


@dataclass(frozen=True)
class _FakeCodexPacket:
    counterexample_capsule: object
    max_bytes: int = 8_192
    max_tokens: int = 2_048
    transition: _FakeTransition = _FakeTransition()

    def to_dict(self) -> dict[str, object]:
        capsule = self.counterexample_capsule
        if hasattr(capsule, "to_dict"):
            capsule_payload = capsule.to_dict()  # type: ignore[union-attr]
        else:
            capsule_payload = capsule
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/codex-repair-packet@1",
            "transition": self.transition.to_dict(),
            "counterexample_capsule": capsule_payload,
            "limits": {
                "max_bytes": self.max_bytes,
                "max_tokens": self.max_tokens,
            },
        }


def test_seal_from_codex_packet_inherits_capsule_and_refs() -> None:
    codex = _FakeCodexPacket(counterexample_capsule=MappingProxyType(_capsule()))
    packet = residual_llm_packet_from_codex(
        codex,
        task_id="WPD-002",
        repository_id="repository:sha256:wpd-002",
        tree_id="tree:wpd-002",
        write_paths=("pkg/mod.py",),
        obligation_ids=("obligation:a",),
        validation_commands=("pytest -q",),
    )
    assert packet.counterexample_capsule["counterexamples"]
    assert packet.codex_packet_ref
    assert packet.transition_ref
    assert packet.max_bytes == 8_192
    assert packet.max_tokens == 2_048
    # Codex secrets still cannot enter through the capsule.
    dirty = deepcopy(_capsule())
    dirty["access_token"] = "leak"
    with pytest.raises(ResidualLlmPacketError) as excinfo:
        residual_llm_packet_from_codex(
            _FakeCodexPacket(counterexample_capsule=dirty),
            task_id="WPD-002",
            repository_id="repository:sha256:wpd-002",
            tree_id="tree:wpd-002",
            write_paths=("pkg/mod.py",),
            obligation_ids=("obligation:a",),
            validation_commands=("pytest -q",),
        )
    assert excinfo.value.reason_code == ResidualLlmPacketReason.SECRET_MATERIAL.value


def test_limits_from_dict_round_trip() -> None:
    limits = ResidualLlmPacketLimits(
        max_bytes=4096, max_tokens=1024, max_capsule_bytes=2048
    )
    restored = ResidualLlmPacketLimits.from_dict(limits.to_dict())
    assert restored.max_bytes == 4096
    assert restored.max_tokens == 1024
    assert restored.max_capsule_bytes == 2048


def test_direct_construction_enforces_capsule_budget() -> None:
    capsule = _capsule()
    capsule["counterexamples"] = [
        {
            "counterexample_id": f"cex:{i}",
            "kind": "generic_failure",
            "summary": "x" * 200,
            "violated_property": "y" * 200,
        }
        for i in range(30)
    ]
    with pytest.raises(ResidualLlmPacketBudgetError) as excinfo:
        ResidualLlmPacket(
            task_id="WPD-002",
            repository_id="repository:sha256:wpd-002",
            tree_id="tree:wpd-002",
            write_paths=("pkg/mod.py",),
            obligation_ids=("obligation:a",),
            counterexample_capsule=capsule,
            validation_commands=("pytest -q",),
            max_bytes=24_576,
            max_tokens=6_144,
            max_capsule_bytes=1024,
        )
    assert excinfo.value.reason_code == ResidualLlmPacketReason.OVER_BUDGET.value
