"""ASEH-001 sealed baseline, companion manifests, and mutation detection."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)

ROOT = Path(__file__).resolve().parents[4]
INVENTORY = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory"
BENCHMARKS = ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening"
REQUIREMENTS_PATH = (
    ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
)
SCHEDULER_PATH = ROOT / "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
TODO_PATH = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md"
BOOTSTRAP_PATH = INVENTORY / "bootstrap_baseline.json"
BASELINE_PATH = INVENTORY / "sealed_baseline.json"
PROVIDER_PATH = BENCHMARKS / "provider_model_config.json"
PRICE_PATH = BENCHMARKS / "price_snapshot.json"
ENVIRONMENT_PATH = BENCHMARKS / "environment_identity.json"
MANIFEST_PATH = BENCHMARKS / "sealed_input_manifest.json"

ACCELERATE_PLANNING_COMMIT = "755f45475cc2d13dacd8b330036c1d597afeddde"
ACCELERATE_PLANNING_TREE = "729da9f8293ecfa046a0136381a3d3808f9ed140"
DATASETS_COMMIT = "209dbe2765593fbc6efe8e9281c34f2e8f6e37a6"
DATASETS_TREE = "95f54df34585d0b736706fd90c83f55954489ad9"
KIT_COMMIT = "ba5508d940fb5b23a6d0d9b2084f5195cd26a671"
KIT_TREE = "7c71efa93c4e4124d12fa05515868df3a5344b2e"
DATASETS_GITLINK_AT_PLANNING = "66a02063496fd200f2372b3083e376f1978c6be1"
KIT_GITLINK_AT_PLANNING = "2564aea1ae35061f2165872aff91e8a40801ab7e"
REQUIREMENTS_DIGEST = (
    "sha256:734a1c016bbb4e3e896c612c95a7144d6a8846b921784e75a41d90a5f2568e96"
)
CANONICAL_PYTHON = "/usr/bin/python3.12"
CANONICAL_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
PACKAGE_VERSION = "0.0.45"
UNAVAILABLE = "unavailable"
POLICY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
DIGEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-benchmark-code-digest@1"

REQUIRED_FIELD_MAP = {
    "all_repository_commits_and_trees": "repositories",
    "supervisor_policy_cid_or_equivalent_immutable_identity": "supervisor_policy",
    "provider_and_model_configuration": "provider_and_model_configuration",
    "task_corpus": "task_corpus",
    "acceptance_tests": "acceptance_tests",
    "cost_accounting_method": "cost_accounting_method",
    "benchmark_code_digest": "benchmark_code_digest",
    "environment_identity": "environment_identity",
}

ARTIFACT_PATHS = (
    BASELINE_PATH,
    PROVIDER_PATH,
    PRICE_PATH,
    ENVIRONMENT_PATH,
    MANIFEST_PATH,
)

ARTIFACT_RELATIVE_PATHS = {
    BASELINE_PATH: (
        "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/"
        "sealed_baseline.json"
    ),
    PROVIDER_PATH: (
        "benchmarks/agent_supervisor/efficiency_state_hardening/"
        "provider_model_config.json"
    ),
    PRICE_PATH: (
        "benchmarks/agent_supervisor/efficiency_state_hardening/"
        "price_snapshot.json"
    ),
    ENVIRONMENT_PATH: (
        "benchmarks/agent_supervisor/efficiency_state_hardening/"
        "environment_identity.json"
    ),
    MANIFEST_PATH: (
        "benchmarks/agent_supervisor/efficiency_state_hardening/"
        "sealed_input_manifest.json"
    ),
}

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_METRIC_KEYS = {
    "input",
    "output",
    "cached_input",
    "reasoning",
    "revision",
    "endpoint",
    "tokenizer_revision",
    "charge",
    "usd",
    "price",
    "tokens",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    canonical = _canonical(payload)
    assert raw == canonical or raw + "\n" == canonical
    return payload


def _body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "identity"}


def _identity_of(payload: Mapping[str, Any]) -> str:
    return content_identity(_body(payload) if "identity" in payload else dict(payload))


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _git_ok(*args: str, cwd: Path | None = None) -> bool:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _peel_tree(spec: str, *, cwd: Path | None = None) -> str:
    peeled = _git("rev-parse", spec, cwd=cwd)
    kind = _git("cat-file", "-t", peeled, cwd=cwd)
    if kind == "tree":
        return peeled
    if kind == "commit":
        return _git("rev-parse", f"{peeled}^{{tree}}", cwd=cwd)
    raise AssertionError(f"{spec!r} peels to {kind}, not a commit or tree")


def _gitlink_at(commit: str, path: str) -> str:
    raw = _git("ls-tree", commit, path)
    mode, kind, rest = raw.split(None, 2)
    sha, name = rest.split("\t", 1)
    assert mode == "160000", path
    assert kind == "commit", path
    assert name == path
    assert SHA1_RE.fullmatch(sha)
    return sha


def _assert_cid(value: Any, *, field: str) -> None:
    assert isinstance(value, str) and CID_RE.fullmatch(value), field


def _assert_digest(value: Any, *, field: str) -> None:
    assert isinstance(value, str) and DIGEST_RE.fullmatch(value), field


def _assert_sha1(value: Any, *, field: str) -> None:
    assert isinstance(value, str) and SHA1_RE.fullmatch(value), field


def _assert_unavailable(value: Any, *, field: str) -> None:
    if isinstance(value, str):
        assert value == UNAVAILABLE, field
        return
    assert isinstance(value, dict), field
    assert value.get("status") == UNAVAILABLE, field
    assert value.get("value", UNAVAILABLE) == UNAVAILABLE, field


def _is_numeric_zero(value: Any) -> bool:
    return type(value) in (int, float) and value == 0


def _walk_unavailable(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        unavailable = (
            value.get("status") == UNAVAILABLE
            or value.get("availability") == UNAVAILABLE
        )
        for key, child in value.items():
            if key in _METRIC_KEYS and _is_numeric_zero(child):
                raise AssertionError(f"{path}.{key} recorded numeric zero")
            if unavailable and _is_numeric_zero(child):
                raise AssertionError(f"{path}.{key} recorded numeric zero")
            _walk_unavailable(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_unavailable(child, path=f"{path}[{index}]")


def _assert_required_fields(baseline: dict[str, Any]) -> None:
    for requirement, key in REQUIRED_FIELD_MAP.items():
        assert key in baseline, requirement


def _assert_bound_artifact(
    bound: Mapping[str, Any],
    path: Path,
    *,
    relative: str,
) -> dict[str, Any]:
    payload = _load(path)
    assert bound["path"] == relative
    live = _identity_of(payload)
    _assert_cid(live, field=relative)
    if "identity" in bound:
        assert bound["identity"] == live
    if "identity" in payload:
        assert payload["identity"] == live
    if "sha256" in bound:
        assert bound["sha256"] == _sha256_file(path)
    return payload


def _todo_task_ids() -> list[str]:
    return re.findall(r"^## (ASEH-[0-9]+)\b", TODO_PATH.read_text(encoding="utf-8"), re.M)


def _policy_identity(scheduler: Mapping[str, Any]) -> str:
    return content_identity(
        {
            "authority_policy": scheduler["authority_policy"],
            "plan_revision": "ASEH-PLAN-R1",
            "program_id": scheduler["program_identifier"],
            "requirements_digest": REQUIREMENTS_DIGEST,
            "scheduler_digest": _sha256_file(SCHEDULER_PATH),
            "schema": POLICY_SCHEMA,
        }
    )


def _benchmark_digest(paths: list[str]) -> str:
    return content_identity(
        {
            "files": [
                {"path": relative, "sha256": _sha256_file(ROOT / relative)}
                for relative in paths
            ],
            "schema": DIGEST_SCHEMA,
        }
    )


def test_artifacts_are_canonical_json_with_stable_identity() -> None:
    for path in ARTIFACT_PATHS:
        payload = _load(path)
        assert payload["task_id"] == "ASEH-001"
        assert payload["program_id"] == (
            "agent-supervisor-efficiency-and-state-hardening-v1"
        )
        assert payload["authority"] is False
        cid = _identity_of(payload)
        _assert_cid(cid, field=str(path))
        assert payload["identity"] == cid


def test_every_sealed_artifact_has_an_exact_external_binding() -> None:
    baseline = _load(BASELINE_PATH)
    manifest = _load(MANIFEST_PATH)
    manifest_outputs = {item["path"]: item for item in manifest["sealed_outputs"]}
    baseline_bindings = {
        PROVIDER_PATH: baseline["provider_and_model_configuration"],
        PRICE_PATH: baseline["price_snapshot"],
        ENVIRONMENT_PATH: baseline["environment_identity"],
        MANIFEST_PATH: baseline["sealed_input_manifest"],
    }
    for path, relative in ARTIFACT_RELATIVE_PATHS.items():
        payload = _load(path)
        live_identity = _identity_of(payload)
        live_sha256 = _sha256_file(path)
        if path == BASELINE_PATH:
            # Its top-level CID seals its body; self-referential file hashes
            # would be mathematically unsatisfiable.
            assert payload["identity"] == live_identity
            continue
        bound = baseline_bindings[path]
        required_binding = {
            "identity": live_identity,
            "path": relative,
            "sha256": live_sha256,
        }
        assert {key: bound[key] for key in required_binding} == required_binding
        if path == PRICE_PATH:
            assert bound["captured_at"] == payload["captured_at"]
        if path not in {BASELINE_PATH, MANIFEST_PATH}:
            assert manifest_outputs[relative]["identity"] == live_identity
            assert manifest_outputs[relative]["sha256"] == live_sha256


def test_required_baseline_fields_are_bound() -> None:
    requirements = _load_json(REQUIREMENTS_PATH)
    baseline = _load(BASELINE_PATH)
    required = requirements["sealed_baseline_required_fields"]
    assert required == list(REQUIRED_FIELD_MAP)
    assert len(required) == 8
    _assert_required_fields(baseline)
    assert "resource_limits" in baseline
    assert "price_snapshot" in baseline
    assert baseline["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/aseh-sealed-baseline@1"
    )
    assert baseline["plan_revision"] == "ASEH-PLAN-R1"
    assert baseline["goal_id"] == "ASEH-G010"
    assert baseline["candidate_behavior_permitted"] is False
    assert baseline["qualification"] is False
    assert UTC_RE.fullmatch(baseline["sealed_at"])


def test_three_repository_commits_and_trees() -> None:
    baseline = _load(BASELINE_PATH)
    bootstrap = _load_json(BOOTSTRAP_PATH)
    scheduler = _load_json(SCHEDULER_PATH)
    repositories = baseline["repositories"]
    accelerate = repositories["ipfs_accelerate_py"]
    datasets = repositories["ipfs_datasets_py"]
    kit = repositories["ipfs_kit_py"]

    assert accelerate["planning_commit"] == ACCELERATE_PLANNING_COMMIT
    assert accelerate["planning_tree"] == ACCELERATE_PLANNING_TREE
    assert datasets["commit"] == DATASETS_COMMIT
    assert datasets["tree"] == DATASETS_TREE
    assert kit["commit"] == KIT_COMMIT
    assert kit["tree"] == KIT_TREE
    assert accelerate["planning_commit"] == bootstrap["repositories"][
        "ipfs_accelerate_py"
    ]["commit"]
    assert accelerate["planning_tree"] == bootstrap["repositories"][
        "ipfs_accelerate_py"
    ]["tree"]
    assert accelerate["planning_commit"] == scheduler["source_binding"][
        "accelerator_planning_revision"
    ]
    assert datasets["commit"] == scheduler["source_binding"]["datasets_planning_revision"]
    assert kit["commit"] == scheduler["source_binding"]["kit_planning_revision"]

    assert _peel_tree(ACCELERATE_PLANNING_COMMIT) == ACCELERATE_PLANNING_TREE
    assert _git_ok("merge-base", "--is-ancestor", ACCELERATE_PLANNING_COMMIT, "HEAD")
    assert _gitlink_at(ACCELERATE_PLANNING_COMMIT, "ipfs_datasets_py") == (
        DATASETS_GITLINK_AT_PLANNING
    )
    assert _gitlink_at(ACCELERATE_PLANNING_COMMIT, "ipfs_kit_py") == (
        KIT_GITLINK_AT_PLANNING
    )
    assert accelerate["baseline_gitlinks"] == {
        "ipfs_datasets_py": DATASETS_GITLINK_AT_PLANNING,
        "ipfs_kit_py": KIT_GITLINK_AT_PLANNING,
    }
    assert accelerate["campaign_gitlinks"] == {
        "ipfs_datasets_py": DATASETS_COMMIT,
        "ipfs_kit_py": KIT_COMMIT,
    }

    datasets_root = ROOT / "ipfs_datasets_py"
    kit_root = ROOT / "ipfs_kit_py"
    assert _peel_tree(DATASETS_COMMIT, cwd=datasets_root) == DATASETS_TREE
    assert _peel_tree(KIT_COMMIT, cwd=kit_root) == KIT_TREE

    envelope = baseline["implementation_envelope_tree_id"]
    _assert_sha1(envelope["value"], field="implementation_envelope_tree_id")
    assert envelope["authority"] is False
    assert envelope["claim_is_current_head"] is False
    if _git_ok("cat-file", "-e", envelope["value"]):
        envelope_kind = _git("cat-file", "-t", envelope["value"])
        assert envelope_kind in {"commit", "tree"}

    forest = baseline["bootstrap_forest"]
    _assert_unavailable(forest["commit"], field="bootstrap_forest.commit")
    _assert_unavailable(forest["tree"], field="bootstrap_forest.tree")
    assert forest["planning_commit_is_required_ancestor"] is True
    assert forest["claim_is_runtime_or_completion_evidence"] is False

    for name in ("ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"):
        sealed = repositories[name]
        observed = bootstrap["repositories"][name]
        assert sealed["role"] == observed["role"]
        assert sealed["remote"] == observed["remote"]
        assert sealed["selected_ref"] == observed["selected_ref"]


def test_supervisor_policy_identity_binds_protected_inputs() -> None:
    baseline = _load(BASELINE_PATH)
    scheduler = _load_json(SCHEDULER_PATH)
    policy = baseline["supervisor_policy"]
    expected = _policy_identity(scheduler)
    _assert_cid(expected, field="computed_supervisor_policy_identity")
    if "identity" in policy:
        assert policy["identity"] == expected
    _assert_digest(policy["requirements_digest"], field="requirements_digest")
    assert policy["requirements_digest"] == REQUIREMENTS_DIGEST
    assert policy["requirements_digest"] == scheduler["requirements_digest"]
    assert policy["requirements_digest"] == _sha256_file(REQUIREMENTS_PATH)
    if "scheduler_digest" in policy:
        _assert_digest(policy["scheduler_digest"], field="scheduler_digest")
        assert policy["scheduler_digest"] == _sha256_file(SCHEDULER_PATH)
    assert policy["path"] == (
        "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
    )
    assert policy["requirements_path"] == (
        "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
    )
    assert policy["plan_revision"] == "ASEH-PLAN-R1"
    assert policy["schema"] == POLICY_SCHEMA


def test_provider_model_config_reconciles() -> None:
    baseline = _load(BASELINE_PATH)
    scheduler = _load_json(SCHEDULER_PATH)
    provider = _assert_bound_artifact(
        baseline["provider_and_model_configuration"],
        PROVIDER_PATH,
        relative=(
            "benchmarks/agent_supervisor/efficiency_state_hardening/"
            "provider_model_config.json"
        ),
    )
    assert provider["source"] == (
        "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
    )
    configured = scheduler["provider"]
    by_slot = {item["slot"]: item for item in provider["providers"]}
    assert by_slot["primary"]["provider_id"] == configured["primary_provider_id"]
    assert by_slot["primary"]["model_id"] == configured["primary_model_id"]
    assert by_slot["fallback"]["provider_id"] == configured["fallback_provider_id"]
    assert by_slot["fallback"]["model_id"] == configured["fallback_model_id"]
    assert provider["max_concurrency"] == configured["max_concurrency"]
    for item in provider["providers"]:
        _assert_unavailable(item["revision"], field=f"{item['slot']}.revision")
        _assert_unavailable(item["endpoint"], field=f"{item['slot']}.endpoint")
        _assert_unavailable(
            item["tokenizer_revision"], field=f"{item['slot']}.tokenizer_revision"
        )
        assert item["revision_availability"] == UNAVAILABLE


def test_timestamped_price_snapshot_keeps_unavailable_prices() -> None:
    baseline = _load(BASELINE_PATH)
    price = _assert_bound_artifact(
        baseline["price_snapshot"],
        PRICE_PATH,
        relative=(
            "benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json"
        ),
    )
    assert UTC_RE.fullmatch(price["captured_at"])
    assert price["captured_at"] == baseline["sealed_at"]
    assert price["availability"] == UNAVAILABLE
    assert price["unknown_cost_recorded_as_zero"] is False
    assert price["estimates_overwrite_measurements"] is False
    assert price["currency"] == "USD"
    for item in price["prices"]:
        for field in ("input", "output", "cached_input", "reasoning"):
            _assert_unavailable(item[field], field=f"{item['model_id']}.{field}")


def test_environment_identity_binds_canonical_validation_environment() -> None:
    baseline = _load(BASELINE_PATH)
    bootstrap = _load_json(BOOTSTRAP_PATH)
    environment = _assert_bound_artifact(
        baseline["environment_identity"],
        ENVIRONMENT_PATH,
        relative=(
            "benchmarks/agent_supervisor/efficiency_state_hardening/"
            "environment_identity.json"
        ),
    )
    canonical = environment["canonical_validation_environment"]
    assert canonical["python_interpreter"] == CANONICAL_PYTHON
    assert canonical["python_major_minor"] == [3, 12]
    assert canonical["path"] == CANONICAL_PATH
    assert canonical["home_directory_name_prefix"] == "ipfs-accelerate-validation-home-"
    assert canonical["xdg_cache"] == "$HOME/.cache"
    assert canonical["xdg_config"] == "$HOME/.config"
    assert canonical["xdg_data"] == "$HOME/.local/share"
    assert canonical["xdg_state"] == "$HOME/.local/state"
    _assert_unavailable(
        canonical["operator_profile_state"], field="operator_profile_state"
    )
    assert sys.version_info[:2] == (3, 12)
    assert environment["package_version"] == PACKAGE_VERSION
    assert environment["capability_is_authority"] is False
    assert environment["host_capability_observation"] == bootstrap[
        "host_capability_observation"
    ]
    assert environment["host_capability_observation"]["capability_is_authority"] is False


def test_task_and_acceptance_identities_and_unavailable_corpora() -> None:
    baseline = _load(BASELINE_PATH)
    manifest = _load(MANIFEST_PATH)
    corpus = baseline["task_corpus"]
    acceptance = baseline["acceptance_tests"]
    expected_ids = _todo_task_ids()
    corpus_identity = _identity_of(corpus)
    acceptance_identity = _identity_of(acceptance)
    _assert_cid(corpus_identity, field="task_corpus.identity")
    _assert_cid(acceptance_identity, field="acceptance_tests.identity")
    if "identity" in corpus:
        assert corpus["identity"] == corpus_identity
    if "identity" in acceptance:
        assert acceptance["identity"] == acceptance_identity
    if "task_corpus_identity" in manifest:
        assert manifest["task_corpus_identity"] == corpus_identity
    if "acceptance_input_identity" in manifest:
        assert manifest["acceptance_input_identity"] == acceptance_identity
    assert corpus["board_path"] == (
        "docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md"
    )
    assert corpus["task_count"] == 40
    assert corpus["task_ids"] == expected_ids
    assert expected_ids[0] == "ASEH-000"
    assert expected_ids[-1] == "ASEH-075"
    assert "ASEH-001" in expected_ids
    assert len(expected_ids) == 40
    assert len(set(expected_ids)) == 40
    for name in (
        "hermetic_development_corpus",
        "historical_replay_corpus",
        "live_shadow_canary_cohort",
    ):
        _assert_unavailable(corpus[name], field=name)
    assert acceptance["validation_command"] == [
        "python3",
        "-m",
        "pytest",
        "-q",
        "test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py",
    ]
    assert acceptance["path"] == (
        "test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py"
    )
    if "sha256" in acceptance:
        assert acceptance["sha256"] == _sha256_file(ROOT / acceptance["path"])
    _assert_unavailable(
        acceptance["hermetic_acceptance_vectors"],
        field="hermetic_acceptance_vectors",
    )
    _assert_unavailable(
        acceptance["historical_acceptance_vectors"],
        field="historical_acceptance_vectors",
    )
    _assert_unavailable(
        acceptance["live_acceptance_vectors"], field="live_acceptance_vectors"
    )


def test_cost_method_benchmark_digest_and_resource_limits() -> None:
    baseline = _load(BASELINE_PATH)
    scheduler = _load_json(SCHEDULER_PATH)
    method = baseline["cost_accounting_method"]
    digest = baseline["benchmark_code_digest"]
    limits = baseline["resource_limits"]
    assert method["unknown_cost_recorded_as_zero"] is False
    assert method["estimates_overwrite_measurements"] is False
    assert method["provider_reported_charge_preferred"] is True
    assert "price_snapshot_identity_and_timestamp" in method["fields"]
    assert "audit_and_verification_overhead" in method["fields"]
    assert digest["paired_harness"]["status"] == UNAVAILABLE
    file_paths = [
        item["path"] if isinstance(item, dict) else item for item in digest["files"]
    ]
    live = _benchmark_digest(file_paths)
    _assert_cid(live, field="computed_benchmark_code_digest")
    if "digest" in digest:
        assert digest["digest"] == live
    for item in digest["files"]:
        if isinstance(item, dict) and "sha256" in item:
            assert item["sha256"] == _sha256_file(ROOT / item["path"])
    assert limits["max_lanes"] == scheduler["max_lanes"]
    assert limits["max_concurrency"] == scheduler["provider"]["max_concurrency"]
    assert limits["implementation_timeout_seconds"] == (
        scheduler["implementation_timeout_seconds"]
    )
    assert limits["implementation_max_timeout_seconds"] == (
        scheduler["implementation_max_timeout_seconds"]
    )
    assert limits["max_task_attempts"] == scheduler["max_task_attempts"]
    assert limits["implementation_retry_budget"] == scheduler["implementation_retry_budget"]
    assert limits["validation_retry_budget"] == scheduler["validation_retry_budget"]
    assert limits["merge_retry_budget"] == scheduler["merge_retry_budget"]


def test_manifests_reconcile_to_baseline() -> None:
    baseline = _load(BASELINE_PATH)
    manifest = _assert_bound_artifact(
        baseline["sealed_input_manifest"],
        MANIFEST_PATH,
        relative=(
            "benchmarks/agent_supervisor/efficiency_state_hardening/"
            "sealed_input_manifest.json"
        ),
    )
    by_path = {item["path"]: item for item in manifest["sealed_outputs"]}
    for relative, path, key in (
        (
            "benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json",
            PROVIDER_PATH,
            "provider_and_model_configuration",
        ),
        (
            "benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json",
            PRICE_PATH,
            "price_snapshot",
        ),
        (
            "benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json",
            ENVIRONMENT_PATH,
            "environment_identity",
        ),
    ):
        payload = _load(path)
        live = _identity_of(payload)
        _assert_cid(live, field=relative)
        item = by_path[relative]
        if "identity" in item:
            assert item["identity"] == live
        if "sha256" in item:
            assert item["sha256"] == _sha256_file(path)
        if "identity" in baseline[key]:
            assert baseline[key]["identity"] == live
        if "sha256" in baseline[key] and "sha256" in item:
            assert baseline[key]["sha256"] == item["sha256"]
    protected = {item["path"]: item for item in manifest["protected_inputs"]}
    requirements_item = protected[
        "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
    ]
    if "sha256" in requirements_item:
        assert requirements_item["sha256"] == REQUIREMENTS_DIGEST
    for item in manifest["protected_inputs"]:
        live_digest = _sha256_file(ROOT / item["path"])
        _assert_digest(live_digest, field=item["path"])
        if "sha256" in item:
            assert item["sha256"] == live_digest
    for item in manifest["unavailable_inputs"]:
        assert item["status"] == UNAVAILABLE


def test_unavailable_fields_are_not_numeric_zero() -> None:
    for path in ARTIFACT_PATHS:
        _walk_unavailable(_load(path), path=path.name)
    price = _load(PRICE_PATH)
    assert price["availability"] == UNAVAILABLE
    assert all(
        item[field] == UNAVAILABLE
        for item in price["prices"]
        for field in ("input", "output", "cached_input", "reasoning")
    )


def test_mutation_changes_content_identity() -> None:
    baseline = _body(_load(BASELINE_PATH))
    original = content_identity(baseline)
    mutated = copy.deepcopy(baseline)
    mutated["repositories"]["ipfs_accelerate_py"]["planning_commit"] = "0" * 40
    assert content_identity(mutated) != original
    price_body = _body(_load(PRICE_PATH))
    price_mutated = copy.deepcopy(price_body)
    price_mutated["prices"][0]["input"] = 0
    assert content_identity(price_mutated) != content_identity(price_body)
    provider_body = _body(_load(PROVIDER_PATH))
    provider_mutated = copy.deepcopy(provider_body)
    provider_mutated["providers"][0]["revision"] = "synthetic-latest"
    assert content_identity(provider_mutated) != content_identity(provider_body)


def test_post_seal_file_mutation_fails_canonical_and_digest_checks() -> None:
    for path in ARTIFACT_PATHS:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        mutated = copy.deepcopy(payload)
        mutated["task_id"] = "ASEH-999"
        assert raw != _canonical(mutated)
        assert _sha256_file(path) != (
            "sha256:" + hashlib.sha256(_canonical(mutated).encode("utf-8")).hexdigest()
        )
        assert _identity_of(mutated) != _identity_of(payload)


def test_missing_required_field_is_rejected() -> None:
    baseline = _body(_load(BASELINE_PATH))
    for key in REQUIRED_FIELD_MAP.values():
        mutated = copy.deepcopy(baseline)
        del mutated[key]
        try:
            _assert_required_fields(mutated)
        except AssertionError:
            continue
        raise AssertionError(f"{key} deletion was not rejected")


def test_synthetic_head_cannot_replace_planning_commit() -> None:
    baseline = _body(_load(BASELINE_PATH))
    head = _git("rev-parse", "HEAD")
    _assert_sha1(head, field="HEAD")
    assert (
        baseline["repositories"]["ipfs_accelerate_py"]["planning_commit"]
        == ACCELERATE_PLANNING_COMMIT
    )
    synthetic = copy.deepcopy(baseline)
    synthetic["repositories"]["ipfs_accelerate_py"]["planning_commit"] = head
    if head != ACCELERATE_PLANNING_COMMIT:
        assert content_identity(synthetic) != content_identity(baseline)
        try:
            assert (
                synthetic["repositories"]["ipfs_accelerate_py"]["planning_commit"]
                == ACCELERATE_PLANNING_COMMIT
            )
        except AssertionError:
            return
        raise AssertionError("synthetic HEAD was accepted as planning commit")


def test_zero_price_is_rejected_as_unavailable_replacement() -> None:
    price = _body(_load(PRICE_PATH))
    mutated = copy.deepcopy(price)
    mutated["prices"][0]["input"] = 0
    mutated["availability"] = "measured"
    try:
        _walk_unavailable(mutated, path="price_snapshot")
    except AssertionError:
        return
    raise AssertionError("numeric zero was accepted in place of unavailable")
