"""Frozen benchmark and adversarial-corpus contract tests for PCPC-029."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = ROOT / "benchmarks" / "agent_supervisor" / "procedure_compiler"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"
RECIPES_PATH = BENCHMARK_ROOT / "case_recipes.fixture"
PARTITIONS = ("synthesis", "development", "held_out", "negative", "boundary", "adversarial")
REQUIRED_COVERAGE = {"recurring", "recovery", "unknown", "unsafe", "transfer"}
SCHEMA = "ipfs_accelerate_py/agent-supervisor/procedure-compiler-benchmark-case@1"


class BenchmarkManifestError(ValueError):
    """The reviewed benchmark declaration is incomplete or unsafe."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkManifestError(f"{path.name} must contain an object")
    return value


@dataclass(frozen=True)
class ProcedureBenchmarkManifest:
    """Closed, synthetic benchmark declaration with content-addressed cases."""

    payload: Mapping[str, Any]
    recipes: Mapping[str, Any]

    @classmethod
    def load(cls) -> "ProcedureBenchmarkManifest":
        return cls(payload=_load(MANIFEST_PATH), recipes=_load(RECIPES_PATH))

    def cases(self) -> tuple[dict[str, Any], ...]:
        partitions = self.recipes.get("partitions")
        families = self.recipes.get("families")
        if partitions != list(PARTITIONS) or not isinstance(families, list):
            raise BenchmarkManifestError("recipes must use the frozen partition vocabulary")
        result: list[dict[str, Any]] = []
        for family_recipe in families:
            if not isinstance(family_recipe, dict):
                raise BenchmarkManifestError("family recipe must be an object")
            for partition in PARTITIONS:
                expected = "refuse" if partition == "adversarial" else family_recipe.get("expected")
                case = {
                    "schema": SCHEMA,
                    "case_id": f"pcpc-v1/{family_recipe.get('family')}/{partition}",
                    "family": family_recipe.get("family"),
                    "partition": partition,
                    "coverage": family_recipe.get("coverage"),
                    "operation": family_recipe.get("operation"),
                    "expected_decision": expected,
                    "synthetic": True,
                    "input_digest": _sha256(
                        {"family": family_recipe.get("family"), "partition": partition, "version": 1}
                    ),
                }
                if partition == "adversarial":
                    case["adversarial"] = {
                        "attack_class": "fixture-shortcut",
                        "mandatory_decision": "refuse",
                        "requires_refusal_reason": True,
                    }
                case["content_sha256"] = _sha256(case)
                result.append(case)
        return tuple(result)


@dataclass(frozen=True)
class BenchmarkPartitionValidator:
    """Fail-closed validator for partition isolation and corpus safety."""

    manifest: ProcedureBenchmarkManifest

    def validate(self) -> tuple[dict[str, Any], ...]:
        manifest = self.manifest.payload
        recipes = self.manifest.recipes
        cases = self.manifest.cases()
        if manifest.get("schema") != "ipfs_accelerate_py/agent-supervisor/procedure-compiler-benchmark-manifest@1":
            raise BenchmarkManifestError("unsupported benchmark manifest schema")
        if manifest.get("status") != "qualified_frozen" or manifest.get("frozen") is not True:
            raise BenchmarkManifestError("benchmark must be qualified and frozen")
        if manifest.get("case_corpus_qualified") is not True or manifest.get("partition_coverage_established") is not True:
            raise BenchmarkManifestError("case corpus is not qualified")
        if manifest.get("partitions") != list(PARTITIONS):
            raise BenchmarkManifestError("manifest partition vocabulary changed")
        if manifest.get("held_out_disjoint") is not True or manifest.get("synthesis_can_read_held_out") is not False:
            raise BenchmarkManifestError("held-out isolation declaration is unsafe")
        access = manifest.get("partition_access")
        if not isinstance(access, dict) or "held_out" in set(access.get("synthesis", ())) | set(access.get("development", ())):
            raise BenchmarkManifestError("held-out cases are visible to training")
        if access.get("held_out") != ["held_out"]:
            raise BenchmarkManifestError("held-out evaluator access is not isolated")
        if manifest.get("large_bodies_in_git") or manifest.get("private_prompts_included") or manifest.get("chain_of_thought_included"):
            raise BenchmarkManifestError("corpus contains prohibited material")
        if recipes.get("schema") != manifest.get("corpus_schema") or recipes.get("synthetic_only") is not True:
            raise BenchmarkManifestError("recipe corpus schema or privacy declaration is invalid")
        family_names = [row.get("family") for row in recipes.get("families", ()) if isinstance(row, dict)]
        if family_names != manifest.get("task_families") or len(set(family_names)) != len(family_names):
            raise BenchmarkManifestError("manifest and recipes disagree about task families")
        if {row.get("coverage") for row in recipes.get("families", ()) if isinstance(row, dict)} != REQUIRED_COVERAGE:
            raise BenchmarkManifestError("required recurring/recovery/unknown/unsafe/transfer coverage is absent")
        if len(cases) != len(family_names) * len(PARTITIONS) or len(cases) != manifest.get("corpus_case_count"):
            raise BenchmarkManifestError("case count is incomplete")
        bounds = manifest.get("bounds")
        if not isinstance(bounds, dict) or not (len(cases) <= bounds.get("max_case_count", 0) and RECIPES_PATH.stat().st_size <= bounds.get("max_recipe_bytes", 0)):
            raise BenchmarkManifestError("corpus exceeds its declared privacy/size bounds")
        case_ids = [case["case_id"] for case in cases]
        digests = [case["content_sha256"] for case in cases]
        if len(case_ids) != len(set(case_ids)) or len(digests) != len(set(digests)):
            raise BenchmarkManifestError("case identities are not unique")
        if any(case["content_sha256"] != _sha256({key: value for key, value in case.items() if key != "content_sha256"}) for case in cases):
            raise BenchmarkManifestError("case content identity mismatch")
        for partition in PARTITIONS:
            rows = [case for case in cases if case["partition"] == partition]
            if len(rows) != len(family_names) or {case["family"] for case in rows} != set(family_names):
                raise BenchmarkManifestError(f"{partition} partition is incomplete")
        held_out = {case["case_id"] for case in cases if case["partition"] == "held_out"}
        observed = {case["case_id"] for case in cases if case["partition"] != "held_out"}
        if held_out & observed:
            raise BenchmarkManifestError("held-out identities overlap another partition")
        adversarial = [case for case in cases if case["partition"] == "adversarial"]
        if not all(case.get("expected_decision") == "refuse" and case.get("adversarial", {}).get("mandatory_decision") == "refuse" and case["adversarial"].get("requires_refusal_reason") is True for case in adversarial):
            raise BenchmarkManifestError("adversarial cases do not require refusal")
        counts = {partition: sum(case["partition"] == partition for case in cases) for partition in PARTITIONS}
        if counts != manifest.get("partition_case_counts"):
            raise BenchmarkManifestError("partition counts do not bind the recipe corpus")
        refs = manifest.get("case_manifest_refs")
        if not isinstance(refs, list) or refs != [{"path": "case_recipes.fixture", "sha256": hashlib.sha256(RECIPES_PATH.read_bytes()).hexdigest()}]:
            raise BenchmarkManifestError("recipe file reference is not content-addressed")
        if manifest.get("corpus_sha256") != _sha256(list(cases)):
            raise BenchmarkManifestError("expanded corpus identity mismatch")
        return cases


def test_frozen_manifest_expands_complete_disjoint_corpus() -> None:
    cases = BenchmarkPartitionValidator(ProcedureBenchmarkManifest.load()).validate()
    assert len(cases) == 138
    assert {case["coverage"] for case in cases} == REQUIRED_COVERAGE
    assert {case["partition"] for case in cases} == set(PARTITIONS)


@pytest.mark.parametrize("mutator", [
    lambda manifest, recipes: manifest["partition_access"].__setitem__("synthesis", ["held_out"]),
    lambda manifest, recipes: manifest.__setitem__("synthesis_can_read_held_out", True),
    lambda manifest, recipes: recipes["families"].pop(),
    lambda manifest, recipes: recipes["families"].__setitem__(0, {**recipes["families"][0], "coverage": "other"}),
])
def test_validator_rejects_held_out_and_coverage_regressions(mutator: Any) -> None:
    source = ProcedureBenchmarkManifest.load()
    manifest, recipes = copy.deepcopy(source.payload), copy.deepcopy(source.recipes)
    mutator(manifest, recipes)
    with pytest.raises(BenchmarkManifestError):
        BenchmarkPartitionValidator(ProcedureBenchmarkManifest(manifest, recipes)).validate()
