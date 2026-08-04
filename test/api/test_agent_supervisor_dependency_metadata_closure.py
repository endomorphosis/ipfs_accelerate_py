from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import (
    project_dependency_preflight as preflight_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    _evaluate_dependency_payload,
)


class _MetadataInventory:
    def __init__(
        self,
        *,
        versions: dict[str, object],
        requirements: dict[str, object] | None = None,
    ) -> None:
        self.versions = dict(versions)
        self.requirements = dict(requirements or {})
        self.version_calls: list[str] = []
        self.requirement_calls: list[str] = []

    def version(self, name: str) -> str:
        self.version_calls.append(name)
        if name not in self.versions:
            raise importlib.metadata.PackageNotFoundError(name)
        value = self.versions[name]
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]

    def requires(self, name: str) -> Any:
        self.requirement_calls.append(name)
        value = self.requirements.get(name, [])
        if isinstance(value, BaseException):
            raise value
        return value


def _payload(*requirements: object) -> dict[str, object]:
    return {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "projects": [
            {
                "root": "ipfs_kit_py",
                "project_name_sha256": "b" * 64,
                "pyproject_sha256": "a" * 64,
                "requirements": list(requirements),
                "requirement_marker_extras": [""] * len(requirements),
                "requires_python": "",
            }
        ],
    }


def _evaluate(
    inventory: _MetadataInventory,
    *requirements: object,
) -> dict[str, Any]:
    return _evaluate_dependency_payload(
        _payload(*requirements),
        version_getter=inventory.version,
        requires_getter=inventory.requires,
    )


@pytest.mark.parametrize("backend_present", [False, True])
def test_requested_eth_hash_extra_checks_pycryptodome(
    backend_present: bool,
) -> None:
    versions = {"eth-hash": "0.8.0"}
    if backend_present:
        versions["pycryptodome"] = "03.023.0"
    inventory = _MetadataInventory(
        versions=versions,
        requirements={
            "eth-hash": [
                "pycryptodome<4,>=3.6.6; extra == 'pycryptodome'"
            ],
            "pycryptodome": [],
        },
    )

    result = _evaluate(
        inventory,
        "Eth_Hash[PyCryptoDome]>=0.3.3",
    )
    project = result["projects"][0]

    assert result["passed"] is backend_present
    assert inventory.version_calls[:1] == ["eth-hash"]
    assert inventory.requirement_calls.count("eth-hash") == 1
    if backend_present:
        observed = {
            item["name"]: item["installed_version"]
            for item in project["observed"]
        }
        assert observed["pycryptodome"] == "3.23.0"
        child = next(
            item
            for item in project["observed"]
            if item["name"] == "pycryptodome"
        )
        assert child["marker_extra"] == "pycryptodome"
        assert project["missing"] == []
    else:
        assert project["missing"][0]["name"] == "pycryptodome"
        assert project["missing"][0]["parent_name"] == "eth-hash"
        assert project["missing"][0]["depth"] == 1
        assert project["missing"][0]["marker_extra"] == "pycryptodome"


def test_nested_incompatible_dependency_is_reported_with_parent() -> None:
    inventory = _MetadataInventory(
        versions={"root-package": "1.0", "nested-package": "1.5"},
        requirements={
            "root-package": ["Nested_Package>=2"],
            "nested-package": [],
        },
    )

    result = _evaluate(inventory, "Root.Package>=1")
    incompatible = result["projects"][0]["incompatible"]

    assert result["passed"] is False
    assert incompatible[0]["name"] == "nested-package"
    assert incompatible[0]["parent_name"] == "root-package"
    assert incompatible[0]["installed_version"] == "1.5"
    assert incompatible[0]["depth"] == 1
    assert inventory.version_calls == ["root-package", "nested-package"]


def test_base_markers_and_every_requested_extra_are_evaluated() -> None:
    inventory = _MetadataInventory(
        versions={
            "root-package": "1",
            "base-dependency": "1",
            "feature-dependency": "1",
            "other-dependency": "1",
        },
        requirements={
            "root-package": [
                "base-dependency>=1; python_version >= '3'",
                "feature-dependency>=1; extra == 'feature'",
                "other-dependency>=1; extra == 'other'",
                "never-installed>=1; python_version < '2'",
                "unrequested-dependency>=1; extra == 'absent'",
            ],
            "base-dependency": [],
            "feature-dependency": [],
            "other-dependency": [],
        },
    )

    result = _evaluate(
        inventory,
        "Root_Package[Other,Feature]>=1",
    )
    project = result["projects"][0]
    observed_names = {item["name"] for item in project["observed"]}

    assert result["passed"] is True
    assert observed_names == {
        "root-package",
        "base-dependency",
        "feature-dependency",
        "other-dependency",
    }
    assert "never-installed" not in inventory.version_calls
    assert "unrequested-dependency" not in inventory.version_calls
    assert project["dependency_closure"]["expanded_context_count"] >= 3
    assert any(
        item["name"] == "feature-dependency"
        and item["marker_extra"] == ""
        for item in project["marker_skipped"]
    )


def test_dependency_cycle_is_detected_without_failing_or_recursing() -> None:
    inventory = _MetadataInventory(
        versions={"package-a": "1", "package-b": "1"},
        requirements={
            "package-a": ["package-b>=1"],
            "package-b": ["package-a>=1"],
        },
    )

    result = _evaluate(inventory, "package-a>=1")
    closure = result["projects"][0]["dependency_closure"]

    assert result["passed"] is True
    assert closure["node_count"] == 2
    assert closure["edge_count"] == 3
    assert closure["cycle_count"] == 1
    assert closure["cycles"][0]["path"] == [
        "package-a",
        "package-b",
        "package-a",
    ]
    assert inventory.requirement_calls == ["package-a", "package-b"]


def test_extra_activating_cycle_remains_subject_to_depth_bound(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        preflight_module,
        "MAX_DEPENDENCY_CLOSURE_DEPTH",
        1,
    )
    inventory = _MetadataInventory(
        versions={"package-a": "1", "package-b": "1"},
        requirements={
            "package-a": ["package-b[next]>=1"],
            "package-b": ["package-a[again]>=1"],
        },
    )

    result = _evaluate(inventory, "package-a>=1")
    project = result["projects"][0]

    assert result["passed"] is False
    assert any(
        item["kind"] == "dependency_closure_bound"
        and item["bound"] == "depth"
        for item in project["invalid"]
    )
    assert project["dependency_closure"]["stopped_on_bound"] is True


@pytest.mark.parametrize(
    (
        "constant_name",
        "maximum",
        "top_requirement",
        "versions",
        "metadata",
        "expected_bound",
        "secret_text",
    ),
    [
        (
            "MAX_DEPENDENCY_CLOSURE_NODES",
            1,
            "root-package",
            {"root-package": "1", "child-package": "1"},
            {"root-package": ["child-package"]},
            "nodes",
            "",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_EDGES",
            1,
            "root-package",
            {"root-package": "1", "child-package": "1"},
            {"root-package": ["child-package"]},
            "edges",
            "",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_DEPTH",
            0,
            "root-package",
            {"root-package": "1", "child-package": "1"},
            {"root-package": ["child-package"]},
            "depth",
            "",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_REQUIREMENTS",
            1,
            "root-package",
            {"root-package": "1", "child-package": "1"},
            {"root-package": ["child-package"]},
            "requirements",
            "",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_CONTEXTS",
            1,
            "root-package[feature]",
            {"root-package": "1"},
            {"root-package": []},
            "contexts",
            "",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES",
            3,
            "root-package-secret",
            {"root-package-secret": "1"},
            {},
            "requirement_bytes",
            "root-package-secret",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_METADATA_TEXT_BYTES",
            3,
            "root-package",
            {"root-package": "1", "child-secret": "1"},
            {"root-package": ["child-secret"]},
            "metadata_text_bytes",
            "child-secret",
        ),
        (
            "MAX_DEPENDENCY_CLOSURE_INSTALLED_VERSION_BYTES",
            4,
            "root-package",
            {"root-package": "1.0+secret-version"},
            {},
            "installed_version_bytes",
            "secret-version",
        ),
    ],
    ids=(
        "nodes",
        "edges",
        "depth",
        "requirements",
        "contexts",
        "requirement-bytes",
        "metadata-text-bytes",
        "installed-version-bytes",
    ),
)
def test_dependency_closure_enforces_every_bound_without_raw_text(
    monkeypatch,
    constant_name: str,
    maximum: int,
    top_requirement: str,
    versions: dict[str, object],
    metadata: dict[str, object],
    expected_bound: str,
    secret_text: str,
) -> None:
    monkeypatch.setattr(preflight_module, constant_name, maximum)
    inventory = _MetadataInventory(
        versions=versions,
        requirements=metadata,
    )

    result = _evaluate(inventory, top_requirement)
    project = result["projects"][0]
    bound_failure = next(
        item
        for item in project["invalid"]
        if item["kind"] == "dependency_closure_bound"
    )

    assert result["passed"] is False
    assert bound_failure["bound"] == expected_bound
    assert bound_failure["maximum"] == maximum
    assert bound_failure["observed"] > maximum
    assert project["dependency_closure"]["stopped_on_bound"] is True
    if secret_text:
        assert secret_text not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize(
    "metadata_value",
    [
        RuntimeError("metadata-exception-secret"),
        "metadata-type-secret",
        [{"metadata-entry-secret": True}],
        ["this is not a valid requirement !!! metadata-parse-secret"],
        ["root-package\ud800metadata-surrogate-secret"],
    ],
    ids=(
        "unavailable",
        "invalid-container",
        "invalid-entry",
        "invalid-pep508",
        "invalid-unicode",
    ),
)
def test_unavailable_or_malformed_metadata_fails_closed_without_raw_text(
    metadata_value: object,
) -> None:
    inventory = _MetadataInventory(
        versions={"root-package": "1"},
        requirements={"root-package": metadata_value},
    )

    result = _evaluate(inventory, "root-package")
    serialized = json.dumps(result, sort_keys=True)

    assert result["passed"] is False
    assert result["projects"][0]["invalid"]
    assert "metadata-exception-secret" not in serialized
    assert "metadata-type-secret" not in serialized
    assert "metadata-entry-secret" not in serialized
    assert "metadata-parse-secret" not in serialized
    assert "metadata-surrogate-secret" not in serialized


def test_transitive_direct_reference_is_hash_only_and_never_fetched() -> None:
    secret_url = (
        "https://user:transitive-secret@example.invalid/pkg.whl"
        "?signature=never-persist"
    )
    inventory = _MetadataInventory(
        versions={"root-package": "1"},
        requirements={
            "root-package": [f"private-package @ {secret_url}"],
        },
    )

    result = _evaluate(inventory, "root-package")
    invalid = result["projects"][0]["invalid"][0]
    serialized = json.dumps(result, sort_keys=True)

    assert result["passed"] is False
    assert invalid["kind"] == "direct_reference_unverifiable"
    assert invalid["name"] == "private-package"
    assert invalid["direct_reference_sha256"]
    assert "private-package" not in inventory.version_calls
    assert "transitive-secret" not in serialized
    assert "never-persist" not in serialized
    assert secret_url not in serialized


@pytest.mark.parametrize(
    ("installed_version", "expected_passed", "expected_version"),
    [
        ("01.002.000", True, "1.2.0"),
        ("not-a-version-secret", False, ""),
    ],
)
def test_installed_versions_are_canonical_or_hash_only(
    installed_version: str,
    expected_passed: bool,
    expected_version: str,
) -> None:
    inventory = _MetadataInventory(
        versions={"root-package": installed_version},
        requirements={"root-package": []},
    )

    result = _evaluate(inventory, "root-package>=1")
    project = result["projects"][0]
    serialized = json.dumps(result, sort_keys=True)

    assert result["passed"] is expected_passed
    if expected_passed:
        assert project["observed"][0]["installed_version"] == expected_version
    else:
        invalid = project["invalid"][0]
        assert invalid["kind"] == "installed_version"
        assert invalid["installed_version_sha256"]
        assert "not-a-version-secret" not in serialized


def test_metadata_order_does_not_change_closure_receipt() -> None:
    versions = {
        "root-package": "1",
        "alpha-package": "1",
        "zeta-package": "1",
    }
    first = _MetadataInventory(
        versions=versions,
        requirements={
            "root-package": ["zeta-package", "alpha-package"],
            "alpha-package": [],
            "zeta-package": [],
        },
    )
    second = _MetadataInventory(
        versions=versions,
        requirements={
            "root-package": ["alpha-package", "zeta-package"],
            "alpha-package": [],
            "zeta-package": [],
        },
    )

    first_result = _evaluate(first, "Root_Package")
    second_result = _evaluate(second, "Root_Package")

    assert first_result == second_result
    assert first.version_calls == second.version_calls
    assert first.requirement_calls == second.requirement_calls
