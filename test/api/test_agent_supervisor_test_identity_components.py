"""PTR-011 identity-component compilation and privacy tests."""

from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.test_identity_components import (
    TEST_IDENTITY_COMPONENTS_INTERFACE,
    TestIdentityComponentError,
    TestIdentityComponents,
    UnsupportedPytestParameter,
    canonicalize_pytest_parameter,
    collect_dependency_identity,
    collect_environment_identity,
    collect_fixture_hook_identity,
)


def _runtime_facts() -> dict[str, dict[str, object]]:
    return {
        "interpreter_facts": {
            "implementation": "cpython",
            "version": [3, 12, 4, "final", 0],
            "cache_tag": "cpython-312",
            "abi_flags": "",
            "byteorder": "little",
            "pointer_bits": 64,
        },
        "platform_facts": {
            "system": "linux",
            "release": "6.8",
            "machine": "x86_64",
            "python_compiler": "GCC 13.2",
            "libc": ["glibc", "2.39"],
        },
        "hardware_facts": {
            "architecture": "x86_64",
            "cpu_count": 8,
            "accelerator_backend": "cuda",
            "accelerator_count": 1,
            "accelerator_architectures": ["sm_80"],
        },
    }


def _compile(**changes: object) -> TestIdentityComponents:
    values: dict[str, object] = {
        "parameter_value": {"model": "tiny", "shards": (1, 2)},
        "parameter_id": "tiny-two-shards",
        "fixtures": [
            {
                "name": "model",
                "scope": "session",
                "definition": "def model(): return build_model()\n",
                "value_adapter_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3udm4nw2v4xkmms",
                "dependencies": ["tmp_path_factory"],
            }
        ],
        "conftests": [
            {
                "path": "test/conftest.py",
                "content": "def pytest_configure(config): pass\n",
            }
        ],
        "hooks": [
            {
                "name": "pytest_collection_modifyitems",
                "implementation": "def pytest_collection_modifyitems(items): pass\n",
                "distribution": "pytest",
                "version": "8.3.2",
                "order": 1,
            }
        ],
        "plugins": [
            {
                "name": "proof-reuse",
                "implementation": "def pytest_configure(config): pass\n",
                "distribution": "ipfs-accelerate-py",
                "version": "1.0.0",
            }
        ],
        "lock_files": {"uv.lock": b"version = 1\n"},
        "installed_distributions": {"pytest": "8.3.2", "pluggy": "1.5.0"},
        "environment": {
            "CI": "true",
            "PYTHONHASHSEED": "7",
            "SECRET_TOKEN": "must-not-enter-identity",
        },
        "environment_allowlist": ("CI", "PYTHONHASHSEED"),
        "capability_facts": {
            "cuda": {"available": True, "runtime": "12.4"}
        },
        "capability_allowlist": ("cuda",),
        **_runtime_facts(),
    }
    values.update(changes)
    return TestIdentityComponents.compile(**values)  # type: ignore[arg-type]


def test_parameters_are_type_preserving_and_order_canonical() -> None:
    first = canonicalize_pytest_parameter(
        {"z": {3, 1, 2}, "a": [True, None, b"\x00"]}
    )
    second = canonicalize_pytest_parameter(
        {"a": [True, None, b"\x00"], "z": {2, 3, 1}}
    )

    assert first == second
    assert canonicalize_pytest_parameter([1, 2]) != canonicalize_pytest_parameter(
        (1, 2)
    )
    assert canonicalize_pytest_parameter(True) != canonicalize_pytest_parameter(1)


@pytest.mark.parametrize(
    "value, reason",
    [
        (1.25, "unsupported_pytest_parameter_float"),
        (object(), "unsupported_pytest_parameter_type"),
        ({1: "not-a-string-key"}, "pytest_parameter_mapping_key_not_string"),
    ],
)
def test_unsupported_parameters_are_rejected_without_repr(
    value: object, reason: str
) -> None:
    with pytest.raises(UnsupportedPytestParameter, match=reason):
        canonicalize_pytest_parameter(value)


def test_parameter_subclasses_are_not_treated_as_reviewed_builtin_types() -> None:
    class CustomString(str):
        pass

    with pytest.raises(
        UnsupportedPytestParameter, match="unsupported_pytest_parameter_type"
    ):
        canonicalize_pytest_parameter(CustomString("looks-safe"))
    with pytest.raises(
        UnsupportedPytestParameter, match="pytest_parameter_mapping_key_not_string"
    ):
        canonicalize_pytest_parameter({CustomString("key"): "value"})


def test_compile_marks_unsupported_parameter_explicitly_non_reusable() -> None:
    compiled = _compile(parameter_value=object())

    assert compiled.reusable is False
    assert compiled.non_reusable_reasons == (
        "unsupported_pytest_parameter_type",
    )
    assert "object at 0x" not in json.dumps(compiled.to_dict())


def test_fixture_scope_definition_value_and_conftest_are_bound() -> None:
    base = _compile()
    scope = _compile(
        fixtures=[
            {
                "name": "model",
                "scope": "function",
                "definition": "def model(): return build_model()\n",
                "value_adapter_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3udm4nw2v4xkmms",
                "dependencies": ["tmp_path_factory"],
            }
        ]
    )
    definition = _compile(
        fixtures=[
            {
                "name": "model",
                "scope": "session",
                "definition": "def model(): return build_other_model()\n",
                "value_adapter_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3udm4nw2v4xkmms",
                "dependencies": ["tmp_path_factory"],
            }
        ]
    )
    conftest = _compile(
        conftests=[
            {
                "path": "test/conftest.py",
                "content": "def pytest_configure(config): config.changed = True\n",
            }
        ]
    )

    assert base.fixture_cids != scope.fixture_cids
    assert base.fixture_cids != definition.fixture_cids
    assert base.conftest_closure_cid != conftest.conftest_closure_cid


def test_fixture_values_are_hashed_not_retained() -> None:
    identity = collect_fixture_hook_identity(
        fixtures=[
            {
                "name": "credential_fixture",
                "scope": "function",
                "definition": "def credential_fixture(): ...\n",
                "value": "private-fixture-value",
            }
        ]
    )

    assert "private-fixture-value" not in repr(identity)
    assert identity.fixture_cids[0].startswith("b")


def test_uncontrolled_or_unsupported_fixture_values_disable_reuse() -> None:
    uncontrolled = _compile(
        fixtures=[
            {
                "name": "dynamic",
                "scope": "function",
                "definition": "def dynamic(): ...\n",
            }
        ]
    )
    unsupported = _compile(
        fixtures=[
            {
                "name": "dynamic",
                "scope": "function",
                "definition": "def dynamic(): ...\n",
                "value": object(),
            }
        ]
    )

    assert uncontrolled.non_reusable_reasons == ("uncontrolled_fixture_value",)
    assert unsupported.non_reusable_reasons == ("unsupported_fixture_value",)
    assert uncontrolled.reusable is unsupported.reusable is False


def test_hooks_plugins_registration_order_and_versions_are_bound() -> None:
    base = _compile()
    changed_order = _compile(
        hooks=[
            {
                "name": "pytest_collection_modifyitems",
                "implementation": "def pytest_collection_modifyitems(items): pass\n",
                "distribution": "pytest",
                "version": "8.3.2",
                "order": 2,
            }
        ]
    )
    changed_version = _compile(
        plugins=[
            {
                "name": "proof-reuse",
                "implementation": "def pytest_configure(config): pass\n",
                "distribution": "ipfs-accelerate-py",
                "version": "1.0.1",
            }
        ]
    )

    assert base.hook_plugin_cids != changed_order.hook_plugin_cids
    assert base.hook_plugin_cids != changed_version.hook_plugin_cids


def test_locks_and_distributions_are_normalized_and_bound() -> None:
    first = collect_dependency_identity(
        lock_files={"requirements.lock": "pytest==8.3.2\n"},
        installed_distributions={"pytest": "8.3.2", "IPFS_Accelerate.Py": "1.0"},
    )
    same = collect_dependency_identity(
        lock_files={"requirements.lock": b"pytest==8.3.2\n"},
        installed_distributions=[("ipfs-accelerate-py", "1.0"), ("pytest", "8.3.2")],
    )
    changed_lock = collect_dependency_identity(
        lock_files={"requirements.lock": "pytest==8.3.3\n"},
        installed_distributions={"pytest": "8.3.2", "ipfs-accelerate-py": "1.0"},
    )
    changed_distribution = collect_dependency_identity(
        lock_files={"requirements.lock": "pytest==8.3.2\n"},
        installed_distributions={"pytest": "8.3.3", "ipfs-accelerate-py": "1.0"},
    )

    assert first == same
    assert first.dependency_lock_cid != changed_lock.dependency_lock_cid
    assert (
        first.installed_distributions_cid
        != changed_distribution.installed_distributions_cid
    )


def test_lock_paths_reject_absolute_traversal_and_non_lock_files() -> None:
    with pytest.raises(TestIdentityComponentError, match="absolute path|traversal"):
        collect_dependency_identity(lock_files={"/home/alice/uv.lock": b"x"})
    with pytest.raises(TestIdentityComponentError, match="allowlisted"):
        collect_dependency_identity(lock_files={"requirements.txt": b"x"})


def test_environment_uses_fixed_allowlist_and_ignores_secrets() -> None:
    facts = _runtime_facts()
    with_secret_a = collect_environment_identity(
        environment={"CI": "true", "API_TOKEN": "one"},
        environment_allowlist=("CI",),
        **facts,
    )
    with_secret_b = collect_environment_identity(
        environment={"CI": "true", "API_TOKEN": "two"},
        environment_allowlist=("CI",),
        **facts,
    )
    changed_allowed = collect_environment_identity(
        environment={"CI": "false", "API_TOKEN": "one"},
        environment_allowlist=("CI",),
        **facts,
    )

    assert with_secret_a == with_secret_b
    assert with_secret_a.environment_cid != changed_allowed.environment_cid
    assert "API_TOKEN" not in repr(with_secret_a)
    assert "true" not in repr(with_secret_a)
    with pytest.raises(TestIdentityComponentError, match="non-reviewed"):
        collect_environment_identity(
            environment={"API_TOKEN": "one"},
            environment_allowlist=("API_TOKEN",),
            **facts,
        )


def test_unset_allowlisted_environment_differs_from_set_empty() -> None:
    facts = _runtime_facts()
    unset = collect_environment_identity(
        environment={},
        environment_allowlist=("CI",),
        **facts,
    )
    empty = collect_environment_identity(
        environment={"CI": ""},
        environment_allowlist=("CI",),
        **facts,
    )

    assert unset.environment_cid != empty.environment_cid


def test_empty_device_selection_is_valid_and_bound_as_set() -> None:
    facts = _runtime_facts()
    unset = collect_environment_identity(
        environment={},
        environment_allowlist=("CUDA_VISIBLE_DEVICES",),
        **facts,
    )
    empty = collect_environment_identity(
        environment={"CUDA_VISIBLE_DEVICES": ""},
        environment_allowlist=("CUDA_VISIBLE_DEVICES",),
        **facts,
    )

    assert unset.environment_cid != empty.environment_cid


def test_duplicate_hook_or_plugin_identity_is_rejected() -> None:
    hook = {
        "name": "pytest_collection_modifyitems",
        "implementation": "def pytest_collection_modifyitems(items): pass\n",
        "distribution": "pytest",
        "version": "8.3.2",
    }

    with pytest.raises(TestIdentityComponentError, match="duplicate hook/plugin"):
        collect_fixture_hook_identity(hooks=[hook, hook])


def test_interpreter_platform_hardware_and_capabilities_are_bound() -> None:
    base = _compile()
    python_changed = _compile(
        interpreter_facts={
            **_runtime_facts()["interpreter_facts"],
            "cache_tag": "cpython-313",
        }
    )
    platform_changed = _compile(
        platform_facts={
            **_runtime_facts()["platform_facts"],
            "machine": "aarch64",
        }
    )
    hardware_changed = _compile(
        hardware_facts={
            **_runtime_facts()["hardware_facts"],
            "accelerator_architectures": ["sm_90"],
        }
    )
    capability_changed = _compile(
        capability_facts={"cuda": {"available": True, "runtime": "12.5"}}
    )

    assert base.interpreter_abi_cid != python_changed.interpreter_abi_cid
    assert base.platform_cid != platform_changed.platform_cid
    assert base.hardware_capability_cid != hardware_changed.hardware_capability_cid
    assert (
        base.hardware_capability_cid
        != capability_changed.hardware_capability_cid
    )


def test_runtime_fact_allowlists_require_complete_reviewed_profiles() -> None:
    facts = _runtime_facts()
    incomplete_interpreter = {
        key: value
        for key, value in facts["interpreter_facts"].items()
        if key != "cache_tag"
    }

    with pytest.raises(TestIdentityComponentError, match="missing a required"):
        collect_environment_identity(
            environment={},
            environment_allowlist=(),
            interpreter_facts=incomplete_interpreter,
            platform_facts=facts["platform_facts"],
            hardware_facts=facts["hardware_facts"],
        )


def test_capabilities_are_fail_closed_to_explicit_allowlist() -> None:
    facts = _runtime_facts()
    with pytest.raises(TestIdentityComponentError, match="non-allowlisted"):
        collect_environment_identity(
            environment={},
            environment_allowlist=(),
            capability_facts={"cuda": {"available": True}},
            capability_allowlist=(),
            **facts,
        )

    absent = collect_environment_identity(
        environment={},
        environment_allowlist=(),
        capability_facts={},
        capability_allowlist=("cuda",),
        **facts,
    )
    present = collect_environment_identity(
        environment={},
        environment_allowlist=(),
        capability_facts={"cuda": {"available": False}},
        capability_allowlist=("cuda",),
        **facts,
    )
    assert absent.hardware_capability_cid != present.hardware_capability_cid


def test_component_bundle_is_stable_and_maps_to_execution_key_fields() -> None:
    first = _compile()
    second = _compile(
        installed_distributions=[("pluggy", "1.5.0"), ("pytest", "8.3.2")]
    )

    assert first.to_dict()["interface"] == TEST_IDENTITY_COMPONENTS_INTERFACE
    assert first.component_root_cid == second.component_root_cid
    fields = first.execution_key_fields()
    assert fields["parameter_source_cid"] == first.parameter_cid
    assert fields["fixture_cids"] == first.fixture_cids
    assert fields["environment_cid"] == first.environment_cid
    assert fields["components"]["parameter"] == first.parameter_cid
    assert fields["components"]["identity_components"] == first.component_root_cid


def test_every_required_component_mutation_changes_bundle_root() -> None:
    base = _compile()
    mutations = [
        _compile(parameter_value={"model": "other", "shards": (1, 2)}),
        _compile(lock_files={"uv.lock": b"version = 2\n"}),
        _compile(installed_distributions={"pytest": "8.3.3", "pluggy": "1.5.0"}),
        _compile(environment={"CI": "false", "PYTHONHASHSEED": "7"}),
        _compile(
            hardware_facts={
                **_runtime_facts()["hardware_facts"],
                "cpu_count": 16,
            }
        ),
        _compile(capability_facts={"cuda": {"available": False, "runtime": "12.4"}}),
    ]

    assert all(
        changed.component_root_cid != base.component_root_cid
        for changed in mutations
    )
