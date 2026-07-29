"""Schema and safety-boundary checks for the contract-repair fixture corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "agent_supervisor" / "contract_repair"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
MANIFEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/contract-repair-fixture-manifest@1"
ARTIFACT_ROLES = frozenset({"source", "spec", "test", "history", "index", "proof"})
EXPECTATION_SOURCES = frozenset({"reviewed_spec", "test"})
REQUIRED_SCENARIOS = frozenset(
    {
        "pure_rename", "module_move", "alias", "re_export", "registration",
        "signature_drift", "vector_nearest_incompatible_decoy", "adapter_required",
        "declaration_without_implementation", "unique_new_site", "multiple_site_abstention",
        "dynamic_dispatch", "reflection", "ffi", "ownership_lifetime_unsupported",
        "stale_roots", "read_only_target", "dependency_cycle", "tombstone",
    }
)


def _canonical_content_id(content: Mapping[str, Any]) -> str:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContractRepairFixture:
    fixture_id: str
    scenario: str
    expected: Mapping[str, Any]
    authority: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractRepairFixture":
        return cls(
            fixture_id=str(value["id"]), scenario=str(value["scenario"]),
            expected=value["expected"], authority=value["authority"], artifacts=value["artifacts"],
        )


@dataclass(frozen=True)
class ContractRepairFixtureManifest:
    schema: str
    corpus_id: str
    fixtures: tuple[ContractRepairFixture, ...]

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "ContractRepairFixtureManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema=str(payload["schema"]), corpus_id=str(payload["corpus_id"]),
            fixtures=tuple(ContractRepairFixture.from_dict(case) for case in payload["cases"]),
        )


@pytest.fixture(scope="module")
def manifest() -> ContractRepairFixtureManifest:
    return ContractRepairFixtureManifest.load()


def test_manifest_has_a_content_addressed_recipe_for_every_required_scenario(
    manifest: ContractRepairFixtureManifest,
) -> None:
    assert manifest.schema == MANIFEST_SCHEMA
    assert manifest.corpus_id == "contract-repair-adversarial-v1"
    assert {fixture.scenario for fixture in manifest.fixtures} == REQUIRED_SCENARIOS
    assert len({fixture.fixture_id for fixture in manifest.fixtures}) == len(manifest.fixtures)

    for fixture in manifest.fixtures:
        assert set(fixture.artifacts) == ARTIFACT_ROLES
        assert fixture.expected["trace_disposition"]
        assert fixture.expected["nomination"]
        assert fixture.expected["admission"]
        assert fixture.expected["automated_write"] in {"never", "only_after_target_decision"}
        assert fixture.expected["reason_codes"]
        for role, artifact in fixture.artifacts.items():
            content = artifact["content"]
            assert artifact["content_id"] == _canonical_content_id(content), (fixture.fixture_id, role)
            assert isinstance(content["path"], str) and not content["path"].startswith("/")


def test_expectations_never_grant_authority_to_vectors_or_implementations(
    manifest: ContractRepairFixtureManifest,
) -> None:
    for fixture in manifest.fixtures:
        authority = fixture.authority
        assert set(authority["expectation_sources"]) <= EXPECTATION_SOURCES
        assert authority["expectation_sources"]
        assert authority["implementation_observation_authoritative"] is False
        assert authority["vector_score_authoritative"] is False
        assert authority["requires_independent_proof"] is True
        assert fixture.artifacts["index"]["content"]["semantic_authority"] is False
        assert "implementation_observation" not in authority["expectation_sources"]
        assert "vector_score" not in authority["expectation_sources"]


def test_adversarial_cases_fail_closed_until_their_declared_gate_is_satisfied(
    manifest: ContractRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    for scenario in {
        "signature_drift", "vector_nearest_incompatible_decoy", "declaration_without_implementation",
        "multiple_site_abstention", "dynamic_dispatch", "reflection", "ffi",
        "ownership_lifetime_unsupported", "stale_roots", "read_only_target", "dependency_cycle", "tombstone",
    }:
        assert fixtures[scenario].expected["admission"] == "abstain"
        assert fixtures[scenario].expected["automated_write"] == "never"

    assert fixtures["vector_nearest_incompatible_decoy"].artifacts["index"]["content"]["vector_score"] > 0.99
    assert fixtures["vector_nearest_incompatible_decoy"].expected["admission"] == "abstain"
    assert fixtures["unique_new_site"].expected["nomination"] == "unique_new_implementation_site"
    assert fixtures["adapter_required"].expected["trace_disposition"] == "adapter_required"
