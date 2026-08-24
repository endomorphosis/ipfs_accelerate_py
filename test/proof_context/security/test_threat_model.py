"""Static validator for the frozen PCCE-070 threat-model register.

This module parses documentation and frozen source text only. It does not run
adversarial payloads, providers, repository tests, or mutable runtime paths.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCUMENT = ROOT / "docs/security/proof-context-v0.1-threat-model.md"

SOURCE_REPOSITORY = "endomorphosis/ipfs_accelerate_py"
SOURCE_COMMIT = "78af7999a8190256798a78b4aa51a9ad9c1f0e58"
SOURCE_TREE = "9d0b786ac7f60db0dd787b5df813fd2d50f8d04f"

EXPECTED_REGISTER_SHA256 = "505d63a745c6c6ee463712272fd5a932f4f56dda51c8f96e5261e16ac93105ff"
EXPECTED_REGISTER_CID = "bafkreicqlvr2orogy3xemnyse4x5lkjs6t2w3wsrzd4w4utb4fvmsmif74"
EXPECTED_REGISTER_SIZE = 63920

REQUIRED_THREATS = {
    "TH-001": "source_prompt_injection",
    "TH-002": "malicious_tests_fixtures",
    "TH-003": "untrusted_agents_patches",
    "TH-004": "scope_path_escape",
    "TH-005": "process_escape",
    "TH-006": "network_escape",
    "TH-007": "secret_escape",
    "TH-008": "evidence_forgery",
    "TH-009": "evidence_replay_poisoning",
    "TH-010": "benchmark_leakage",
    "TH-011": "provider_disclosure",
    "TH-012": "concurrent_mutation",
    "TH-013": "interruption",
    "TH-014": "compromised_adapter",
}
REQUIRED_TASKS = tuple(f"PCCE-{number:03d}" for number in range(71, 77))
TASK_REPOSITORIES = {
    "PCCE-071": "endomorphosis/ipfs_accelerate_py",
    "PCCE-072": "endomorphosis/ipfs_kit_py",
    "PCCE-073": "endomorphosis/ipfs_accelerate_py",
    "PCCE-074": "endomorphosis/ipfs_datasets_py",
    "PCCE-075": "endomorphosis/ipfs_accelerate_py",
    "PCCE-076": "cross-repository",
}
TASK_ACCEPTANCE_TESTS = {
    "PCCE-071": ["external/ipfs_accelerate/test/proof_context/security/test_sandbox.py"],
    "PCCE-072": ["external/ipfs_kit/tests/proof_context/test_trust_admission.py"],
    "PCCE-073": [
        "external/ipfs_accelerate/test/proof_context/security/test_adversarial_patch_and_agent.py"
    ],
    "PCCE-074": ["external/ipfs_datasets/tests/proof_context/benchmarks/test_isolation.py"],
    "PCCE-075": [
        "external/ipfs_accelerate/test/proof_context/security/test_adversarial_concurrency.py"
    ],
    "PCCE-076": [
        "external/ipfs_accelerate/test/proof_context/security/test_threat_model.py",
        "external/ipfs_accelerate/test/proof_context/security/test_sandbox.py",
        "external/ipfs_kit/tests/proof_context/test_trust_admission.py",
        "external/ipfs_accelerate/test/proof_context/security/test_adversarial_patch_and_agent.py",
        "external/ipfs_datasets/tests/proof_context/benchmarks/test_isolation.py",
        "external/ipfs_accelerate/test/proof_context/security/test_adversarial_concurrency.py",
    ],
}
CONTROL_CLASSES = frozenset({"preventive", "detective", "recovery"})
CONTROL_STATUSES = frozenset(
    {
        "observed_tested_limited",
        "observed_partial",
        "planned",
        "absent_no_go",
    }
)
SEVERITIES = frozenset({"none", "low", "medium", "high", "critical"})

TOP_FIELDS = frozenset(
    {
        "schema",
        "model_id",
        "version",
        "board_namespace",
        "task_id",
        "status",
        "source_snapshot",
        "scope",
        "assets",
        "actors",
        "entry_points",
        "trust_boundaries",
        "controls",
        "threats",
        "residual_risks",
        "task_mappings",
        "qualification_policy",
        "change_control",
        "generation",
        "review",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "repository",
        "commit",
        "tree",
        "reviewed_paths",
        "board_contract_path",
        "board_contract_lines",
    }
)
SCOPE_FIELDS = frozenset({"in_scope", "out_of_scope", "assumptions"})
ASSET_FIELDS = frozenset({"id", "name", "description", "classification"})
ACTOR_FIELDS = frozenset({"id", "name", "description", "trust"})
ENTRY_FIELDS = frozenset({"id", "name", "description", "owner"})
BOUNDARY_FIELDS = frozenset({"id", "name", "from", "to", "data", "current_status"})
ANCHOR_FIELDS = frozenset({"repository", "commit", "path", "symbol", "line"})
TEST_ANCHOR_FIELDS = ANCHOR_FIELDS | {"test_name"}
CONTROL_FIELDS = frozenset(
    {
        "id",
        "title",
        "classes",
        "status",
        "owner_repository",
        "owner_task",
        "claims",
        "code_anchors",
        "test_anchors",
        "planned_paths",
        "limitations",
        "platforms",
        "fail_closed_on_unavailable",
        "qualification_credit",
    }
)
THREAT_FIELDS = frozenset(
    {
        "id",
        "title",
        "category",
        "severity",
        "assets",
        "actors",
        "entry_points",
        "trust_boundaries",
        "preconditions",
        "attack_summary",
        "impact",
        "controls",
        "code_owner",
        "test_owner",
        "current_disposition",
        "planned_tasks",
        "residual_risks",
        "fail_closed",
    }
)
IMPACT_FIELDS = frozenset({"confidentiality", "integrity", "availability", "description"})
THREAT_CONTROL_FIELDS = frozenset({"preventive", "detective", "recovery"})
FAIL_CLOSED_FIELDS = frozenset({"required", "current", "unavailable_result"})
RISK_FIELDS = frozenset(
    {"id", "threats", "severity", "description", "status", "owner_task", "release_blocking"}
)
TASK_FIELDS = frozenset(
    {
        "task_id",
        "role",
        "repository",
        "threats",
        "controls",
        "acceptance_tests",
        "qualification_effect",
    }
)
QUALIFICATION_FIELDS = frozenset(
    {
        "current_qualification",
        "effective_control_rule",
        "high_or_critical_open_result",
        "missing_or_unavailable_result",
        "planned_control_credit",
        "release_gate_task",
    }
)
CHANGE_FIELDS = frozenset({"frozen_before_tasks", "newly_found_threat_policy", "supersedes"})
GENERATION_FIELDS = frozenset(
    {
        "canonicalization",
        "encoding",
        "array_order",
        "artifact_newline",
        "cid_codec",
        "cid_version",
        "multihash",
        "self_reference",
    }
)
REVIEW_FIELDS = frozenset(
    {"analysis_role", "prepared_by", "required_reviewer_role", "review_status"}
)

FORBIDDEN_BODY_KEYS = frozenset(
    {
        "api_key",
        "answer_body",
        "body",
        "credential",
        "credentials",
        "evaluation_data",
        "fixture_body",
        "future_patch",
        "gold_labels",
        "hidden_body",
        "hidden_prompt",
        "payload",
        "private_key",
        "secret",
        "secret_value",
        "test_body",
        "token_value",
    }
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:sk|ghp|github_pat)[-_][a-z0-9_]{12,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"Bearer\s+[A-Za-z0-9._~-]{8,}"
)
ID_PATTERNS = {
    "assets": re.compile(r"AS-\d{2}"),
    "actors": re.compile(r"AC-\d{2}"),
    "entry_points": re.compile(r"EP-\d{2}"),
    "trust_boundaries": re.compile(r"TB-\d{2}"),
    "controls": re.compile(r"(?:OC-\d{2}|PC-\d{3})"),
    "threats": re.compile(r"TH-\d{3}"),
    "residual_risks": re.compile(r"RR-\d{3}"),
}


def _closed(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    assert isinstance(value, Mapping), f"{label} must be an object"
    assert set(value) == fields, f"{label} has a non-closed field set"
    return value


def _assert_text_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    for field in fields:
        assert isinstance(value[field], str) and value[field], f"{label}.{field} must be text"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    admitted: dict[str, Any] = {}
    for key, value in pairs:
        if key in admitted:
            raise ValueError(f"duplicate JSON key {key!r}")
        admitted[key] = value
    return admitted


def extract_register(markdown: str | None = None) -> Mapping[str, Any]:
    """Extract the authoritative closed JSON register from the Markdown."""

    text = DOCUMENT.read_text(encoding="utf-8") if markdown is None else markdown
    match = re.search(
        r"<!-- machine-register:start -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- machine-register:end -->",
        text,
        re.DOTALL,
    )
    assert match is not None, "machine register block is missing"
    value = json.loads(match.group(1), object_pairs_hook=_reject_duplicate_keys)
    assert isinstance(value, dict)
    return value


def _canonicalize(value: Any) -> str:
    if isinstance(value, float):
        raise AssertionError("floats are forbidden in the threat-model register")
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        assert all(isinstance(key, str) for key in value)
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                + ":"
                + _canonicalize(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_canonicalize(item) for item in value) + "]"
    raise AssertionError(f"unsupported register value {type(value).__name__}")


def canonical_register_bytes(register: Mapping[str, Any] | None = None) -> bytes:
    """Return the exact no-newline outer register projection."""

    source = extract_register() if register is None else register
    return _canonicalize(source).encode("utf-8")


def raw_cid_for_bytes(value: bytes) -> str:
    digest = hashlib.sha256(value).digest()
    raw = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def register_identity(register: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    value = canonical_register_bytes(register)
    return {
        "sha256": hashlib.sha256(value).hexdigest(),
        "cid": raw_cid_for_bytes(value),
        "size": len(value),
    }


def _ordered_ids(
    register: Mapping[str, Any], section: str, fields: frozenset[str]
) -> tuple[str, ...]:
    values = register[section]
    assert isinstance(values, list) and values, f"{section} must be a nonempty array"
    ids: list[str] = []
    for index, value in enumerate(values):
        item = _closed(value, fields, f"{section}[{index}]")
        identifier = item["id"]
        assert isinstance(identifier, str) and ID_PATTERNS[section].fullmatch(identifier)
        ids.append(identifier)
    assert ids == sorted(ids), f"{section} IDs must be sorted"
    assert len(ids) == len(set(ids)), f"{section} IDs must be unique"
    return tuple(ids)


def _assert_strings(value: Any, label: str, *, allow_empty: bool = False) -> None:
    assert isinstance(value, list), f"{label} must be an array"
    assert allow_empty or value, f"{label} must not be empty"
    assert all(isinstance(item, str) and item for item in value), f"{label} has invalid text"
    assert len(value) == len(set(value)), f"{label} has duplicate values"


def _assert_sorted_references(value: Any, allowed: set[str], label: str) -> None:
    _assert_strings(value, label)
    assert value == sorted(value), f"{label} must be sorted"
    assert set(value) <= allowed, f"{label} has a dangling reference"


def _assert_relative_path(value: str, label: str) -> None:
    assert isinstance(value, str) and value, f"{label} must be a path"
    path = Path(value)
    assert not path.is_absolute(), f"{label} must be repository-relative"
    assert ".." not in path.parts, f"{label} must not traverse"
    assert "\x00" not in value, f"{label} contains NUL"


def _walk_safety(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, float):
        raise AssertionError(f"float at {'.'.join(path)}")
    if isinstance(value, str):
        assert not SECRET_VALUE_PATTERN.search(value), f"secret-like value at {'.'.join(path)}"
        assert len(value) <= 4096, f"unbounded string at {'.'.join(path)}"
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert isinstance(key, str)
            normalized = key.lower().replace("-", "_")
            assert normalized not in FORBIDDEN_BODY_KEYS, f"forbidden body field {key!r}"
            assert normalized not in {"timestamp", "created_at", "updated_at", "reviewed_at"}
            _walk_safety(item, (*path, key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        assert len(value) <= 4096, f"unbounded array at {'.'.join(path)}"
        for index, item in enumerate(value):
            _walk_safety(item, (*path, str(index)))
        return
    assert value is None or isinstance(value, (bool, int)), f"unsupported value at {'.'.join(path)}"


def validate_register(register: Mapping[str, Any]) -> None:
    """Validate the strict closed register without executing runtime content."""

    model = _closed(register, TOP_FIELDS, "register")
    assert model["schema"] == "pcce/proof-context/v0.1/threat-model@1"
    assert model["model_id"] == "pcce/proof-context/v0.1/security-threat-model"
    assert model["version"] == "1"
    assert model["board_namespace"] == "proof-carrying-context-engine-v0.1"
    assert model["task_id"] == "PCCE-070"
    assert model["status"] == "security_unqualified_no_go"

    source = _closed(model["source_snapshot"], SOURCE_FIELDS, "source_snapshot")
    assert source["repository"] == SOURCE_REPOSITORY
    assert source["commit"] == SOURCE_COMMIT
    assert source["tree"] == SOURCE_TREE
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", source["tree"])
    _assert_strings(source["reviewed_paths"], "reviewed_paths")
    assert source["reviewed_paths"] == sorted(source["reviewed_paths"])
    for index, path in enumerate(source["reviewed_paths"]):
        _assert_relative_path(path, f"reviewed_paths[{index}]")
    _assert_relative_path(source["board_contract_path"], "board_contract_path")
    assert source["board_contract_lines"] == "1971-2207"

    scope = _closed(model["scope"], SCOPE_FIELDS, "scope")
    _assert_strings(scope["in_scope"], "scope.in_scope")
    _assert_strings(scope["out_of_scope"], "scope.out_of_scope")
    _assert_strings(scope["assumptions"], "scope.assumptions")
    assert scope["in_scope"] == sorted(REQUIRED_THREATS.values())
    assert scope["out_of_scope"] == sorted(scope["out_of_scope"])
    assert scope["assumptions"] == sorted(scope["assumptions"])
    assert "documentation is not a sandbox" in scope["assumptions"]

    asset_ids = set(_ordered_ids(model, "assets", ASSET_FIELDS))
    actor_ids = set(_ordered_ids(model, "actors", ACTOR_FIELDS))
    entry_ids = set(_ordered_ids(model, "entry_points", ENTRY_FIELDS))
    boundary_ids = set(_ordered_ids(model, "trust_boundaries", BOUNDARY_FIELDS))
    control_ids = set(_ordered_ids(model, "controls", CONTROL_FIELDS))
    threat_ids = set(_ordered_ids(model, "threats", THREAT_FIELDS))
    risk_ids = set(_ordered_ids(model, "residual_risks", RISK_FIELDS))

    assert asset_ids == {f"AS-{number:02d}" for number in range(1, 9)}
    assert actor_ids == {f"AC-{number:02d}" for number in range(1, 11)}
    assert entry_ids == {f"EP-{number:02d}" for number in range(1, 13)}
    assert boundary_ids == {f"TB-{number:02d}" for number in range(1, 12)}
    assert control_ids == {
        *(f"OC-{number:02d}" for number in range(1, 11)),
        *(f"PC-{number:03d}" for number in range(71, 77)),
    }
    assert threat_ids == set(REQUIRED_THREATS)
    assert risk_ids == {f"RR-{number:03d}" for number in range(1, 11)}

    for item in model["assets"]:
        _assert_text_fields(item, ("name", "description", "classification"), item["id"])
        assert item["classification"] in {
            "availability-critical",
            "confidential-critical",
            "confidential-sensitive",
            "integrity-critical",
        }
    for item in model["actors"]:
        _assert_text_fields(item, ("name", "description", "trust"), item["id"])
    for item in model["entry_points"]:
        _assert_text_fields(item, ("name", "description", "owner"), item["id"])
    for item in model["trust_boundaries"]:
        _assert_text_fields(item, ("name", "from", "to", "data", "current_status"), item["id"])
        assert item["current_status"] in {
            "absent_no_go",
            "command_adapter_only_no_go",
            "partial_no_go",
        }

    reviewed_paths = set(source["reviewed_paths"])
    control_classes: dict[str, set[str]] = {}
    for item in model["controls"]:
        identifier = item["id"]
        _assert_text_fields(item, ("title", "owner_repository", "owner_task"), identifier)
        _assert_strings(item["classes"], f"{identifier}.classes")
        assert set(item["classes"]) <= CONTROL_CLASSES
        control_classes[identifier] = set(item["classes"])
        assert item["status"] in CONTROL_STATUSES
        _assert_strings(item["claims"], f"{identifier}.claims")
        _assert_strings(item["limitations"], f"{identifier}.limitations")
        _assert_strings(item["platforms"], f"{identifier}.platforms")
        assert item["fail_closed_on_unavailable"] is True
        assert item["qualification_credit"] is False
        assert isinstance(item["code_anchors"], list)
        assert isinstance(item["test_anchors"], list)
        assert isinstance(item["planned_paths"], list)
        if item["status"].startswith("observed_"):
            assert item["code_anchors"] and item["test_anchors"]
            assert item["planned_paths"] == []
            assert item["owner_repository"] == SOURCE_REPOSITORY
            assert item["owner_task"] == "existing-runtime"
        elif item["status"] == "planned":
            assert item["code_anchors"] == [] and item["test_anchors"] == []
            assert item["planned_paths"]
            assert item["owner_task"] == identifier.replace("PC-", "PCCE-")
        for index, anchor in enumerate(item["code_anchors"]):
            admitted = _closed(anchor, ANCHOR_FIELDS, f"{identifier}.code_anchors[{index}]")
            assert admitted["repository"] == SOURCE_REPOSITORY
            assert admitted["commit"] == SOURCE_COMMIT
            assert admitted["path"] in reviewed_paths
            assert isinstance(admitted["symbol"], str) and admitted["symbol"]
            assert isinstance(admitted["line"], int) and admitted["line"] > 0
        for index, anchor in enumerate(item["test_anchors"]):
            admitted = _closed(anchor, TEST_ANCHOR_FIELDS, f"{identifier}.test_anchors[{index}]")
            assert admitted["repository"] == SOURCE_REPOSITORY
            assert admitted["commit"] == SOURCE_COMMIT
            assert admitted["path"] in reviewed_paths
            assert admitted["test_name"] == admitted["symbol"]
            assert admitted["test_name"].startswith("test_")
            assert isinstance(admitted["line"], int) and admitted["line"] > 0
        for index, path in enumerate(item["planned_paths"]):
            _assert_relative_path(path, f"{identifier}.planned_paths[{index}]")

    threat_by_id = {item["id"]: item for item in model["threats"]}
    assert {identifier: item["category"] for identifier, item in threat_by_id.items()} == (
        REQUIRED_THREATS
    )
    for identifier, item in threat_by_id.items():
        assert item["severity"] == "critical"
        _assert_sorted_references(item["assets"], asset_ids, f"{identifier}.assets")
        _assert_sorted_references(item["actors"], actor_ids, f"{identifier}.actors")
        _assert_sorted_references(item["entry_points"], entry_ids, f"{identifier}.entry_points")
        _assert_sorted_references(
            item["trust_boundaries"], boundary_ids, f"{identifier}.trust_boundaries"
        )
        _assert_strings(item["preconditions"], f"{identifier}.preconditions")
        assert isinstance(item["attack_summary"], str) and item["attack_summary"]
        admitted_impact = _closed(item["impact"], IMPACT_FIELDS, f"{identifier}.impact")
        assert admitted_impact["confidentiality"] in SEVERITIES
        assert admitted_impact["integrity"] in SEVERITIES
        assert admitted_impact["availability"] in SEVERITIES
        assert isinstance(admitted_impact["description"], str) and admitted_impact["description"]
        controls = _closed(item["controls"], THREAT_CONTROL_FIELDS, f"{identifier}.controls")
        for control_class in sorted(CONTROL_CLASSES):
            _assert_sorted_references(
                controls[control_class], control_ids, f"{identifier}.controls.{control_class}"
            )
            assert all(
                control_class in control_classes[control_id]
                for control_id in controls[control_class]
            )
        assert "PC-076" in controls["detective"]
        assert isinstance(item["code_owner"], str) and item["code_owner"]
        assert isinstance(item["test_owner"], str) and item["test_owner"]
        assert item["current_disposition"] in {"partial_no_go", "absent_no_go"}
        _assert_sorted_references(
            item["planned_tasks"], set(REQUIRED_TASKS), f"{identifier}.planned_tasks"
        )
        assert "PCCE-076" in item["planned_tasks"]
        _assert_sorted_references(item["residual_risks"], risk_ids, f"{identifier}.residual_risks")
        fail_closed = _closed(item["fail_closed"], FAIL_CLOSED_FIELDS, f"{identifier}.fail_closed")
        assert fail_closed["required"] is True
        assert fail_closed["current"] in {"partial", "absent"}
        assert fail_closed["unavailable_result"] == "no_go"

    assert set().union(*(set(item["assets"]) for item in model["threats"])) == asset_ids
    assert set().union(*(set(item["actors"]) for item in model["threats"])) == actor_ids
    assert set().union(*(set(item["entry_points"]) for item in model["threats"])) == entry_ids
    assert (
        set().union(*(set(item["trust_boundaries"]) for item in model["threats"])) == boundary_ids
    )
    assert (
        set().union(
            *(set().union(*map(set, item["controls"].values())) for item in model["threats"])
        )
        == control_ids
    )

    risk_by_id = {item["id"]: item for item in model["residual_risks"]}
    for item in model["residual_risks"]:
        identifier = item["id"]
        _assert_sorted_references(item["threats"], threat_ids, f"{identifier}.threats")
        assert item["severity"] in {"high", "critical"}
        assert isinstance(item["description"], str) and item["description"]
        assert item["status"] == "open_no_go"
        owner_tasks = item["owner_task"].split("/")
        assert owner_tasks == sorted(owner_tasks)
        assert set(owner_tasks) <= set(REQUIRED_TASKS)
        assert item["release_blocking"] is True
        assert all(
            identifier in threat_by_id[threat_id]["residual_risks"] for threat_id in item["threats"]
        )
    for threat_id, item in threat_by_id.items():
        assert all(
            threat_id in risk_by_id[risk_id]["threats"] for risk_id in item["residual_risks"]
        )

    task_mappings = model["task_mappings"]
    assert isinstance(task_mappings, list) and task_mappings
    assert [item["task_id"] for item in task_mappings] == list(REQUIRED_TASKS)
    task_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(task_mappings):
        mapping = _closed(item, TASK_FIELDS, f"task_mappings[{index}]")
        task_id = mapping["task_id"]
        task_by_id[task_id] = mapping
        assert mapping["repository"] == TASK_REPOSITORIES[task_id]
        _assert_sorted_references(mapping["threats"], threat_ids, f"{task_id}.threats")
        assert mapping["controls"] == [task_id.replace("PCCE-", "PC-")]
        _assert_strings(mapping["acceptance_tests"], f"{task_id}.acceptance_tests")
        assert mapping["acceptance_tests"] == TASK_ACCEPTANCE_TESTS[task_id]
        for path in mapping["acceptance_tests"]:
            _assert_relative_path(path, f"{task_id}.acceptance_tests")
        assert isinstance(mapping["role"], str) and mapping["role"]
        assert isinstance(mapping["repository"], str) and mapping["repository"]
        assert isinstance(mapping["qualification_effect"], str) and mapping["qualification_effect"]
    assert task_by_id["PCCE-076"]["role"] == ("evidence-only security audit and qualification gate")
    assert set(task_by_id["PCCE-076"]["threats"]) == threat_ids
    for control in model["controls"]:
        if control["status"] == "planned":
            assert control["owner_repository"] == TASK_REPOSITORIES[control["owner_task"]]
    for threat_id, item in threat_by_id.items():
        mapped_tasks = sorted(
            task_id for task_id, mapping in task_by_id.items() if threat_id in mapping["threats"]
        )
        assert mapped_tasks == item["planned_tasks"]

    qualification = _closed(
        model["qualification_policy"], QUALIFICATION_FIELDS, "qualification_policy"
    )
    assert qualification["current_qualification"] == "no_go"
    assert qualification["high_or_critical_open_result"] == "no_go"
    assert qualification["missing_or_unavailable_result"] == "no_go"
    assert qualification["planned_control_credit"] is False
    assert qualification["release_gate_task"] == "PCCE-076"

    change = _closed(model["change_control"], CHANGE_FIELDS, "change_control")
    assert change["frozen_before_tasks"] == list(REQUIRED_TASKS[:-1])
    assert change["newly_found_threat_policy"] == "append_versioned_delta"
    assert change["supersedes"] == []

    generation = _closed(model["generation"], GENERATION_FIELDS, "generation")
    assert generation == {
        "canonicalization": "RFC8785-admitted-subset",
        "encoding": "UTF-8",
        "array_order": "stable-id-and-explicit-reference-order",
        "artifact_newline": "none",
        "cid_codec": "raw",
        "cid_version": 1,
        "multihash": "sha2-256",
        "self_reference": "receipt-binds-register",
    }

    review = _closed(model["review"], REVIEW_FIELDS, "review")
    assert review["analysis_role"] == "PCCE-070 supervised threat analysis"
    assert review["prepared_by"] == ("ipfs_accelerate_py-agent-supervisor/PCCE-070-implementation")
    assert review["required_reviewer_role"] == "independent-security-reviewer"
    assert review["review_status"] == "pending_outer_integration"
    _walk_safety(model)


def _marked_section(markdown: str, marker: str) -> str:
    match = re.search(
        rf"<!-- {re.escape(marker)}:start -->(.*?)<!-- {re.escape(marker)}:end -->",
        markdown,
        re.DOTALL,
    )
    assert match is not None, f"Markdown section {marker!r} is missing"
    return match.group(1)


def _markdown_ids(markdown: str, marker: str, pattern: str) -> set[str]:
    return set(re.findall(rf"`({pattern})`", _marked_section(markdown, marker)))


def _git_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    return environment


@cache
def _git_show(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=ROOT,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _definitions(source: str) -> Mapping[str, int]:
    definitions: dict[str, int] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            name = ".".join((*self.stack, node.name))
            definitions[name] = node.lineno
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            name = ".".join((*self.stack, node.name))
            definitions[name] = node.lineno
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(ast.parse(source))
    return definitions


def test_register_schema_references_and_no_go_policy_are_closed() -> None:
    validate_register(extract_register())


def test_markdown_and_register_inventories_are_bijective() -> None:
    markdown = DOCUMENT.read_text(encoding="utf-8")
    register = extract_register(markdown)
    expected = {
        "assets": {item["id"] for item in register["assets"]},
        "actors": {item["id"] for item in register["actors"]},
        "entry-points": {item["id"] for item in register["entry_points"]},
        "trust-boundaries": {item["id"] for item in register["trust_boundaries"]},
        "controls": {item["id"] for item in register["controls"]},
        "threats": {item["id"] for item in register["threats"]},
        "residual-risks": {item["id"] for item in register["residual_risks"]},
        "task-mappings": {item["task_id"] for item in register["task_mappings"]},
    }
    patterns = {
        "assets": r"AS-\d{2}",
        "actors": r"AC-\d{2}",
        "entry-points": r"EP-\d{2}",
        "trust-boundaries": r"TB-\d{2}",
        "controls": r"(?:OC-\d{2}|PC-\d{3})",
        "threats": r"TH-\d{3}",
        "residual-risks": r"RR-\d{3}",
        "task-mappings": r"PCCE-\d{3}",
    }
    for marker, ids in expected.items():
        assert _markdown_ids(markdown, marker, patterns[marker]) == ids
    mermaid = re.search(r"```mermaid(.*?)```", markdown, re.DOTALL)
    assert mermaid is not None
    assert set(re.findall(r"TB-\d{2}", mermaid.group(1))) == expected["trust-boundaries"]
    assert "security unqualified / no-go" in markdown
    assert "documentation as a sandbox" in markdown


def test_observed_code_and_test_anchors_exist_at_frozen_source() -> None:
    tree = subprocess.run(
        ["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"],
        cwd=ROOT,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert tree == SOURCE_TREE
    register = extract_register()
    for control in register["controls"]:
        for anchor in (*control["code_anchors"], *control["test_anchors"]):
            definitions = _definitions(_git_show(anchor["path"]))
            assert definitions.get(anchor["symbol"]) == anchor["line"], anchor


def test_canonical_projection_and_raw_cid_are_frozen() -> None:
    register = extract_register()
    first = canonical_register_bytes(register)
    second = canonical_register_bytes(copy.deepcopy(register))
    assert first == second
    assert not first.endswith(b"\n")
    assert b" " not in first[:1]
    assert register_identity(register) == {
        "sha256": EXPECTED_REGISTER_SHA256,
        "cid": EXPECTED_REGISTER_CID,
        "size": EXPECTED_REGISTER_SIZE,
    }
    assert raw_cid_for_bytes(b"") == ("bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku")
    encoded_cid = EXPECTED_REGISTER_CID[1:].upper()
    decoded = base64.b32decode(encoded_cid + "=" * (-len(encoded_cid) % 8))
    assert decoded[:4] == b"\x01\x55\x12\x20"
    assert decoded[4:] == hashlib.sha256(first).digest()


def test_outer_projection_matches_when_integrated() -> None:
    if ROOT.name != "ipfs_accelerate" or ROOT.parent.name != "external":
        return
    outer = ROOT.parent.parent
    projection = outer / "artifacts/proof_carrying_context_engine/security/threat_model.json"
    assert projection.is_file(), "outer PCCE-070 register is missing"
    assert projection.read_bytes() == canonical_register_bytes()


def _drop_required_threat(value: dict[str, Any]) -> None:
    value["threats"].pop()


def _duplicate_identifier(value: dict[str, Any]) -> None:
    duplicate = copy.deepcopy(value["assets"][-1])
    duplicate["id"] = value["assets"][0]["id"]
    value["assets"].append(duplicate)


def _add_unknown_field(value: dict[str, Any]) -> None:
    value["unexpected"] = True


def _add_dangling_control(value: dict[str, Any]) -> None:
    value["threats"][0]["controls"]["detective"].append("PC-999")


def _promote_planned_control(value: dict[str, Any]) -> None:
    planned = next(item for item in value["controls"] if item["id"] == "PC-071")
    planned["qualification_credit"] = True


def _pass_unavailable(value: dict[str, Any]) -> None:
    value["threats"][0]["fail_closed"]["unavailable_result"] = "pass"


def _add_float(value: dict[str, Any]) -> None:
    value["generation"]["cid_version"] = 1.5


def _add_hidden_body(value: dict[str, Any]) -> None:
    value["threats"][0]["hidden_prompt"] = "synthetic denied body"


def _add_absolute_path(value: dict[str, Any]) -> None:
    planned = next(item for item in value["controls"] if item["id"] == "PC-071")
    planned["planned_paths"][0] = "/tmp/not-admitted"


def _drift_source_commit(value: dict[str, Any]) -> None:
    value["source_snapshot"]["commit"] = "0" * 40


def _unsort_inventory(value: dict[str, Any]) -> None:
    value["actors"].reverse()


def _add_secret_like_value(value: dict[str, Any]) -> None:
    value["scope"]["assumptions"].append("Bearer synthetic-token-value")


def _claim_unobserved_control(value: dict[str, Any]) -> None:
    planned = next(item for item in value["controls"] if item["id"] == "PC-072")
    planned["status"] = "observed_tested_limited"


def _give_gate_runtime_role(value: dict[str, Any]) -> None:
    gate = next(item for item in value["task_mappings"] if item["task_id"] == "PCCE-076")
    gate["role"] = "runtime repair"


def _remove_recovery_mapping(value: dict[str, Any]) -> None:
    value["threats"][0]["controls"]["recovery"] = []


MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("required-threat-omission", _drop_required_threat),
    ("duplicate-id", _duplicate_identifier),
    ("unknown-field", _add_unknown_field),
    ("dangling-control", _add_dangling_control),
    ("planned-credit", _promote_planned_control),
    ("unavailable-pass", _pass_unavailable),
    ("float", _add_float),
    ("hidden-body", _add_hidden_body),
    ("absolute-path", _add_absolute_path),
    ("source-drift", _drift_source_commit),
    ("unsorted-inventory", _unsort_inventory),
    ("secret-like-value", _add_secret_like_value),
    ("unobserved-effective", _claim_unobserved_control),
    ("gate-repair", _give_gate_runtime_role),
    ("missing-recovery", _remove_recovery_mapping),
)


@pytest.mark.parametrize(("name", "mutation"), MUTATIONS, ids=[item[0] for item in MUTATIONS])
def test_register_mutations_fail_closed(
    name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    candidate = copy.deepcopy(dict(extract_register()))
    mutation(candidate)
    with pytest.raises((AssertionError, ValueError), match=".+"):
        validate_register(candidate)
    assert name
