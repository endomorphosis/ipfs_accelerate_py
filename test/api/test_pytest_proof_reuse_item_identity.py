"""Fail-closed automatic item-identity assembly regressions."""

from __future__ import annotations

import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    CidSupportStatus,
    TestExecutionIdentityCompiler,
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_reuse_eligibility import (
    TestReuseEligibilityPolicy,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
    RuntimeTestDependencyTracer,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    ForestPolicy,
    ForestRootSpec,
    RepositoryAuthority,
    build_repository_forest,
)
from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
    ITEM_EXECUTION_KEY_ATTRIBUTE,
    ITEM_IDENTITY_RESULT_ATTRIBUTE,
    CurrentInputCompleteness,
    CurrentItemComponentInputs,
    CurrentItemPolicyInputs,
    CurrentRuntimeTraceEvidence,
    ItemIdentityAssemblyReason,
    ItemIdentityAssemblyServices,
    assemble_and_attach_item_identity,
)


class _Item:
    def __init__(
        self,
        path: Path,
        *,
        parameter: Any = None,
        parameterized: bool = False,
        fixture_names: tuple[str, ...] = (),
    ) -> None:
        self.path = path
        self.nodeid = f"{path.name}::test_direct"
        self.name = "test_direct"
        self.originalname = "test_direct"
        self.fixturenames = fixture_names
        self.cls = None
        self.user_properties: list[Any] = []
        self._markers: list[Any] = []
        if parameterized:
            self.nodeid += "[case]"
            self.callspec = SimpleNamespace(id="case", params={"value": parameter})

    def iter_markers(self, name: str | None = None):
        if name is None:
            return iter(self._markers)
        return (marker for marker in self._markers if marker.name == name)

    def get_closest_marker(self, name: str):
        return next(
            (marker for marker in reversed(self._markers) if marker.name == name),
            None,
        )


def _git(root: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "proof-reuse@example.invalid")
    _git(root, "config", "user.name", "Proof Reuse Test")
    test_path = root / "test_direct.py"
    test_path.write_text(
        "def test_direct():\n    value = 1\n    assert value == 1\n",
        encoding="utf-8",
    )
    _git(root, "add", "test_direct.py")
    _git(root, "commit", "-qm", "fixture")
    return root, test_path


def _forest(root: Path):
    return build_repository_forest(
        ForestPolicy(
            roots=(
                ForestRootSpec(
                    alias="repo",
                    root_path=root,
                    authority=RepositoryAuthority(mode="read_write"),
                ),
            ),
            sole_write_alias="repo",
        )
    )


def _index(root: Path):
    records = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        records.append(
            (
                path.relative_to(root).as_posix(),
                build_python_ast_blob_record(source),
            )
        )
    return build_analysis_ast_index(records)


def _interpreter_facts() -> dict[str, Any]:
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:5]),
        "cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
    }


def _platform_facts() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "system": platform.system().lower(),
        "release": platform.release(),
        "machine": platform.machine().lower(),
        "python_compiler": platform.python_compiler(),
        "libc": [libc_name, libc_version],
    }


def _hardware_facts() -> dict[str, Any]:
    return {
        "architecture": platform.machine().lower(),
        "cpu_count": os.cpu_count() or 0,
        "accelerator_backend": "none",
        "accelerator_count": 0,
        "accelerator_architectures": [],
    }


def _component_inputs(
    *,
    fixtures: tuple[dict[str, Any], ...] = (),
) -> CurrentItemComponentInputs:
    implementation = mint_content_identity(
        {"schema": "test/pytest-plugin@1", "name": "pytest"}
    )
    environment = (
        {"PYTHONUTF8": os.environ["PYTHONUTF8"]}
        if "PYTHONUTF8" in os.environ
        else {}
    )
    return CurrentItemComponentInputs(
        completeness=CurrentInputCompleteness.EXACT_CURRENT,
        fixtures=fixtures,
        plugins=(
            {
                "name": "pytest",
                "implementation_cid": implementation.cid,
                "distribution": "pytest",
                "version": pytest.__version__,
                "registered": True,
                "order": 0,
            },
        ),
        installed_distributions=(("pytest", pytest.__version__),),
        environment=environment,
        environment_allowlist=("PYTHONUTF8",),
        interpreter_facts=_interpreter_facts(),
        platform_facts=_platform_facts(),
        hardware_facts=_hardware_facts(),
    )


def _policy_inputs() -> CurrentItemPolicyInputs:
    def cid(label: str) -> str:
        return mint_content_identity(
            {"schema": "test/proof-reuse-identity@1", "label": label}
        ).cid

    policy_identity = mint_content_identity(
        {"schema": "test/proof-reuse-policy@1", "revision": 1}
    )
    return CurrentItemPolicyInputs(
        completeness=CurrentInputCompleteness.EXACT_CURRENT,
        policy_identity=policy_identity,
        verification_policy={
            "policy_cid": policy_identity.cid,
            "statement_cid": cid("statement"),
            "circuit_cid": cid("circuit"),
            "verifying_key_cid": cid("verifying-key"),
            "proof_system_id": "groth16",
            "trusted_issuer_ids": ("issuer:test",),
            "allowed_epochs": ("epoch:1",),
        },
        reuse_policy=TestReuseEligibilityPolicy(),
        command_semantics={
            "schema": "test/pytest-command@1",
            "selection": "exact-node",
        },
        pytest_config={"schema": "test/pytest-config@1", "root": "repository"},
        plugin_versions={
            "schema": "test/pytest-plugins@1",
            "pytest": pytest.__version__,
        },
        runtime_completeness_policy={
            "schema": "test/runtime-completeness@1",
            "require_complete": True,
        },
        canonicalization_schema={
            "schema": "test/canonicalization@1",
            "profile": "dag-json",
        },
        tracer_schema={
            "schema": "test/tracer-schema@1",
            "static": 1,
            "runtime": 1,
        },
        certificate_schema={
            "schema": "test/certificate-schema@1",
            "version": 1,
        },
    )


def _services(
    root: Path,
    *,
    index_provider=None,
    component_inputs: CurrentItemComponentInputs | None = None,
    runtime_mode: str = "bound",
    compiler: Any = None,
) -> ItemIdentityAssemblyServices:
    policy = _policy_inputs()

    def runtime_provider(item, facts, descriptor, static, components, policies):
        del item, descriptor
        tracer = RuntimeTestDependencyTracer(
            allowed_roots={"repo": root},
            capture_code_objects=False,
        )
        with tracer:
            pass
        assert tracer.result is not None
        if runtime_mode == "raw":
            return tracer.result
        runtime_policy_cid = policies.verified_identities()["runtime_policy"].cid
        return CurrentRuntimeTraceEvidence.bind_fresh_preflight(
            trace=tracer.result,
            node_id=facts.node_id,
            repository_forest_cid=_forest(root).forest_id,
            static_trace_root_cid=static.trace_cid,
            identity_components_cid=components.component_root_cid,
            runtime_completeness_policy_cid=runtime_policy_cid,
        )

    return ItemIdentityAssemblyServices(
        repository_forest_provider=lambda item: _forest(root),
        analysis_index_provider=(
            index_provider
            if index_provider is not None
            else lambda item, descriptor: _index(root)
        ),
        component_inputs_provider=(
            lambda item, facts, descriptor, static: (
                component_inputs or _component_inputs()
            )
        ),
        policy_inputs_provider=(
            lambda item, facts, descriptor, static, components: policy
        ),
        runtime_evidence_provider=(
            None if runtime_mode == "missing" else runtime_provider
        ),
        identity_compiler=compiler,
    )


def test_direct_node_assembles_without_per_test_attributes_or_registry(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)

    result = assemble_and_attach_item_identity(item, _services(root))

    assert result.reason is ItemIdentityAssemblyReason.ADMITTED_FOR_LOOKUP
    assert result.admitted_for_lookup is True
    assert result.action == "RUN"
    assert result.authorizes_skip is False
    assert getattr(item, ITEM_IDENTITY_RESULT_ATTRIBUTE) is result
    assert getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE).test_function_cid.startswith(
        "b"
    )
    assert result.lookup_request.item is item


def test_current_source_mutation_rejects_stale_ast_index_and_changes_identity(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    cached_index = _index(root)
    item = _Item(path)
    before = assemble_and_attach_item_identity(
        item,
        _services(root, index_provider=lambda item, descriptor: cached_index),
    )
    assert before.admitted_for_lookup
    before_key = before.execution_artifact.execution_cid

    path.write_text(
        "def test_direct():\n    value = 2\n    assert value == 2\n",
        encoding="utf-8",
    )
    stale_item = _Item(path)
    stale = assemble_and_attach_item_identity(
        stale_item,
        _services(root, index_provider=lambda item, descriptor: cached_index),
    )
    assert stale.reason is ItemIdentityAssemblyReason.STATIC_TRACE_INCOMPLETE
    assert not hasattr(stale_item, ITEM_EXECUTION_KEY_ATTRIBUTE)

    current_item = _Item(path)
    current = assemble_and_attach_item_identity(current_item, _services(root))
    assert current.admitted_for_lookup
    assert current.execution_artifact.execution_cid != before_key


@pytest.mark.parametrize(
    ("item_factory", "components"),
    (
        (
            lambda path: _Item(path, parameter=object(), parameterized=True),
            _component_inputs(),
        ),
        (
            lambda path: _Item(path, fixture_names=("database",)),
            _component_inputs(
                fixtures=(
                    {
                        "name": "database",
                        "scope": "function",
                        "definition": "def database(): pass",
                    },
                )
            ),
        ),
    ),
)
def test_unsupported_parameter_or_uncontrolled_fixture_runs(
    tmp_path: Path,
    item_factory,
    components: CurrentItemComponentInputs,
) -> None:
    root, path = _repository(tmp_path)
    item = item_factory(path)

    result = assemble_and_attach_item_identity(
        item,
        _services(root, component_inputs=components),
    )

    assert result.reason is ItemIdentityAssemblyReason.COMPONENTS_NON_REUSABLE
    assert result.action == "RUN"
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)


def test_missing_or_raw_runtime_evidence_is_not_current_and_runs(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)

    missing = assemble_and_attach_item_identity(
        _Item(path), _services(root, runtime_mode="missing")
    )
    raw = assemble_and_attach_item_identity(
        _Item(path), _services(root, runtime_mode="raw")
    )

    assert missing.reason is ItemIdentityAssemblyReason.PROVIDER_UNAVAILABLE
    assert missing.stage == "runtime_evidence"
    assert raw.reason is ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_NOT_CURRENT
    assert missing.action == raw.action == "RUN"
    assert not missing.authorizes_skip and not raw.authorizes_skip


def test_missing_cid_capability_never_attaches_lookup_or_skip(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    compiler = TestExecutionIdentityCompiler(
        cid_probe=lambda: CidSupportStatus.MISSING
    )
    item = _Item(path)

    result = assemble_and_attach_item_identity(
        item, _services(root, compiler=compiler)
    )

    assert result.reason is ItemIdentityAssemblyReason.IDENTITY_COMPILER_REJECTED
    assert result.action == "RUN"
    assert result.lookup_request is None
    assert result.authorizes_skip is False
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)


def test_item_identity_module_import_is_cold_and_does_not_require_multiformats(
    tmp_path: Path,
) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "multiformats" or name.startswith("multiformats."):
        raise ModuleNotFoundError("blocked optional dependency", name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[2]))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import ipfs_accelerate_py.testing.proof_reuse.item_identity; "
                "assert not any(n == 'multiformats' or "
                "n.startswith('multiformats.') for n in sys.modules)"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
