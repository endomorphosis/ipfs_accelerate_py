"""Board/config consistency tests for the PCPC supervisor program."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / "scripts/validate_agent_supervisor_procedure_compiler_board.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("pcpc_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(report: dict[str, object], name: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return next(item for item in checks if isinstance(item, dict) and item.get("name") == name)


def _load_benchmark(module) -> dict[str, object]:
    payload = json.loads(module.BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _isolated_benchmark_state(
    module,
    tmp_path: Path,
    monkeypatch,
    manifest: dict[str, object],
    *,
    recipes: dict[str, object] | None = None,
    copy_recipe: bool = True,
) -> tuple[bool, dict[str, object]]:
    source_recipe_path = module.BENCHMARK_RECIPE_PATH
    recipe_path = tmp_path / "case_recipes.fixture"
    if recipes is not None:
        recipe_path.write_text(
            json.dumps(recipes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif copy_recipe:
        recipe_path.write_bytes(source_recipe_path.read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "BENCHMARK_MANIFEST", manifest_path)
    monkeypatch.setattr(module, "BENCHMARK_RECIPE_PATH", recipe_path)
    return module._benchmark_frozen_state()


def test_board_validator_accepts_sealed_program() -> None:
    report = _validator_module().validate_program()
    assert report["valid"] is True, json.dumps(report["errors"], indent=2)
    assert report["task_count"] == 32
    assert report["goal_count"] == 5
    assert report["blocked_task_ids"] == []
    assert report["ready_task_ids"] == ["PCPC-009", "PCPC-011", "PCPC-013"]
    assert _check(report, "self_contained_normative_vocabulary")["passed"] is True
    assert _check(report, "task_parallel_lanes")["passed"] is True
    assert _check(report, "concurrency_dependency_safety")["passed"] is True
    benchmark = _check(report, "benchmark_frozen_state")
    assert benchmark["passed"] is True
    assert benchmark["detail"]["accepted_state"] == "qualified_frozen"


def test_board_validator_accepts_exact_safe_benchmark_scaffold(
    tmp_path, monkeypatch
) -> None:
    module = _validator_module()
    qualified = _load_benchmark(module)
    scaffold = {
        key: deepcopy(qualified[key]) for key in module.BENCHMARK_SCAFFOLD_FIELDS
    }
    scaffold.update(
        {
            "status": "scaffold_only",
            "frozen_scope": "family_and_partition_vocabulary",
            "case_corpus_qualified": False,
            "partition_coverage_established": False,
            "partition_case_counts": deepcopy(module.BENCHMARK_ZERO_PARTITION_COUNTS),
            "case_manifest_refs": [],
            "pcpc_029_obligation": (
                "populate every task family with disjoint synthesis, development, "
                "held_out, negative, boundary, and adversarial cases before qualification"
            ),
            "qualification_blocker": (
                "PCPC-029 has not populated or independently qualified the case corpus"
            ),
        }
    )

    passed, detail = _isolated_benchmark_state(
        module, tmp_path, monkeypatch, scaffold, copy_recipe=False
    )

    assert passed is True
    assert detail["accepted_state"] == "scaffold_only"


@pytest.mark.parametrize(
    "mutation",
    (
        "qualification_flag",
        "partition_vocabulary",
        "partition_count",
        "duplicate_family",
        "private_material",
        "held_out_access",
        "case_count",
        "recipe_reference",
        "corpus_identity",
        "recipe_bound",
        "privacy_policy",
        "partial_manifest",
    ),
)
def test_board_validator_rejects_malformed_qualified_benchmark_declarations(
    mutation: str, tmp_path, monkeypatch
) -> None:
    module = _validator_module()
    manifest = deepcopy(_load_benchmark(module))
    if mutation == "qualification_flag":
        manifest["case_corpus_qualified"] = False
    elif mutation == "partition_vocabulary":
        manifest["partitions"] = [*module.BENCHMARK_PARTITIONS, "training"]
    elif mutation == "partition_count":
        manifest["partition_case_counts"]["held_out"] = 22
    elif mutation == "duplicate_family":
        manifest["task_families"][-1] = manifest["task_families"][0]
    elif mutation == "private_material":
        manifest["private_prompts_included"] = True
    elif mutation == "held_out_access":
        manifest["partition_access"]["synthesis"].append("held_out")
    elif mutation == "case_count":
        manifest["corpus_case_count"] = 137
    elif mutation == "recipe_reference":
        manifest["case_manifest_refs"][0]["sha256"] = "0" * 64
    elif mutation == "corpus_identity":
        manifest["corpus_sha256"] = "0" * 64
    elif mutation == "recipe_bound":
        manifest["bounds"]["max_recipe_bytes"] = 1
    elif mutation == "privacy_policy":
        manifest["privacy_policy"]["forbidden_data"].remove("private prompts")
    elif mutation == "partial_manifest":
        manifest.pop("corpus_schema")
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(f"unknown benchmark mutation: {mutation}")

    passed, detail = _isolated_benchmark_state(
        module, tmp_path, monkeypatch, manifest
    )

    assert passed is False
    assert detail["accepted_state"] is None


@pytest.mark.parametrize(
    "mutation",
    ("synthetic_flag", "recipe_partitions", "missing_family"),
)
def test_board_validator_rejects_coordinated_recipe_corruption(
    mutation: str, tmp_path, monkeypatch
) -> None:
    module = _validator_module()
    manifest = deepcopy(_load_benchmark(module))
    recipes = json.loads(module.BENCHMARK_RECIPE_PATH.read_text(encoding="utf-8"))
    if mutation == "synthetic_flag":
        recipes["synthetic_only"] = False
    elif mutation == "recipe_partitions":
        recipes["partitions"] = list(reversed(recipes["partitions"]))
    elif mutation == "missing_family":
        recipes["families"].pop()
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(f"unknown recipe mutation: {mutation}")
    recipe_bytes = (json.dumps(recipes, indent=2, sort_keys=True) + "\n").encode("utf-8")
    recipe_digest = hashlib.sha256(recipe_bytes).hexdigest()
    manifest["case_manifest_refs"][0]["sha256"] = recipe_digest
    monkeypatch.setattr(module, "BENCHMARK_RECIPE_SHA256", recipe_digest)

    passed, detail = _isolated_benchmark_state(
        module,
        tmp_path,
        monkeypatch,
        manifest,
        recipes=recipes,
    )

    assert passed is False
    assert detail["accepted_state"] is None


def test_board_validator_rejects_lane_metadata_drift(tmp_path, monkeypatch) -> None:
    module = _validator_module()
    original = module.TODO_PATH.read_text(encoding="utf-8")
    corrupted = original.replace(
        "- Parallel lane: pcpc-lane-1", "- Parallel lane: pcpc-lane-0", 1
    )
    todo_path = tmp_path / "todo.md"
    todo_path.write_text(corrupted, encoding="utf-8")
    monkeypatch.setattr(module, "TODO_PATH", todo_path)
    monkeypatch.setattr(module, "REPO_ROOT", Path("/"))

    report = module.validate_program()
    check = _check(report, "task_parallel_lanes")
    assert report["valid"] is False
    assert check["passed"] is False
    assert check["detail"]["PCPC-000"] == {
        "expected": "pcpc-lane-1",
        "observed": "pcpc-lane-0",
    }


def test_board_validator_rejects_concurrency_with_transitive_dependent(
    tmp_path, monkeypatch
) -> None:
    module = _validator_module()
    original = module.TODO_PATH.read_text(encoding="utf-8")
    corrupted = original.replace(
        "- Allow concurrent with:\n", "- Allow concurrent with: PCPC-031\n", 1
    )
    todo_path = tmp_path / "todo.md"
    todo_path.write_text(corrupted, encoding="utf-8")
    monkeypatch.setattr(module, "TODO_PATH", todo_path)
    monkeypatch.setattr(module, "REPO_ROOT", Path("/"))

    report = module.validate_program()
    check = _check(report, "concurrency_dependency_safety")
    assert report["valid"] is False
    assert check["passed"] is False
    assert check["detail"] == [
        {
            "task_id": "PCPC-000",
            "peer": "PCPC-031",
            "relation": "peer_depends_on_task",
        }
    ]


def test_ducklake_is_explicitly_non_authoritative() -> None:
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["ducklake_projection_program"]["scheduling_prerequisite"] is False
    assert config["ducklake_projection_program"]["extension_files_sha256"] == {
        "ducklake.duckdb_extension": (
            "d0b57c8e261b89a1ae367c7224f0857cfde72ab6cf2609f188e0de9b897b1088"
        ),
        "ducklake.duckdb_extension.info": (
            "14c3385450437fee5570ff21b53de687536a75b4590e33f351887df194ef9393"
        ),
    }
    assert config["ducklake_projection_program"]["extension_install_policy"] == "forbidden"
    assert config["ducklake_projection_program"]["network_access"] is False


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("authority", True),
        ("scheduling_prerequisite", True),
        ("extension_install_policy", "allowed"),
        ("network_access", True),
        ("unknown_normative_field", False),
    ),
)
def test_board_validator_rejects_unsafe_ducklake_projection_config(
    mutation: str,
    value: object,
) -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    corrupted = deepcopy(config)
    corrupted["ducklake_projection_program"][mutation] = value
    assert module._ducklake_projection_is_valid(
        corrupted["ducklake_projection_program"],
        owner_extension_hashes=corrupted["quack_owner_isolation"][
            "extension_files_sha256"
        ],
    ) is False


def test_board_validator_rejects_ducklake_extension_digest_drift() -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_projection_program"]["extension_files_sha256"][
        "ducklake.duckdb_extension"
    ] = "0" * 64
    assert module._ducklake_projection_is_valid(
        config["ducklake_projection_program"],
        owner_extension_hashes=config["quack_owner_isolation"]["extension_files_sha256"],
    ) is False


@pytest.mark.parametrize(
    ("catalog_path", "data_path"),
    (
        (
            "docs/redteam.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            ".",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
        ),
    ),
)
def test_board_validator_rejects_non_exact_or_overlapping_ducklake_paths(
    catalog_path: str, data_path: str
) -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_projection_program"]["catalog_path"] = catalog_path
    config["ducklake_projection_program"]["data_path"] = data_path

    assert module._ducklake_projection_is_valid(
        config["ducklake_projection_program"],
        owner_extension_hashes=config["quack_owner_isolation"]["extension_files_sha256"],
    ) is False
