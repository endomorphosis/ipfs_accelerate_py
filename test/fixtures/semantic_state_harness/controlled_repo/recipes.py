"""Compact base-tree and mutation recipes for the controlled fixture.

Target modules are textual recipes only. Scanners and tests must read bytes;
they must never import or execute these modules from the fixture package path.
"""

from __future__ import annotations

from .mutation_case import (
    ChangedSymbolOracle,
    ConfidenceOracle,
    FixtureOracle,
    InvalidationOracle,
    MerkleOracle,
    MutationCase,
    PathOperation,
    ReceiptFreshnessOracle,
)

# ---------------------------------------------------------------------------
# Stable symbol / test / proof identities (fixture authority, not scan output)
# ---------------------------------------------------------------------------

SYM_CORE_ADD = "sch_fixture.core:add"
SYM_CORE_MULTIPLY = "sch_fixture.core:multiply"
SYM_CORE_PROCESS = "sch_fixture.core:process"
SYM_API_FETCH = "sch_fixture.api:fetch_value"
SYM_API_CALL_CORE = "sch_fixture.api:call_core"
SYM_SCHEMA_USER = "sch_fixture.schema:UserRecord"
SYM_SCHEMA_DUMP = "sch_fixture.schema:dump_user"
SYM_SEC_AUTHORIZE = "sch_fixture.security:authorize"
SYM_ADAPTER_CLIENT = "sch_fixture.adapters:McpClientAdapter"
SYM_DYNAMIC_LOAD = "sch_fixture.dynamic_loader:load_plugin"
SYM_NATIVE_BRIDGE = "sch_fixture.native_bridge:native_hash"
SYM_GENERATED_BIND = "sch_fixture.generated.bindings:generated_constant"
SYM_POLICY = "policy.admission:AdmissionPolicy"
SYM_LOCK = "deps.lockfile:LockedDependencySet"
SYM_IFACE = "interfaces.mcp_client:McpClientDescriptor"
SYM_PYTEST_CFG = "pytest.ini:pytest_config"
SYM_FIXTURE_SAMPLE = "tests.conftest:sample_record"
SYM_RENAMED = "sch_fixture.core:renamed_process"
SYM_DELETED = "sch_fixture.core:legacy_helper"

TEST_CORE_ADD = "tests/test_core.py::test_add"
TEST_CORE_PROCESS = "tests/test_core.py::test_process"
TEST_API_FETCH = "tests/test_api.py::test_fetch_value"
TEST_API_CALL = "tests/test_api.py::test_call_core"
TEST_SCHEMA = "tests/test_schema.py::test_user_roundtrip"
TEST_SECURITY = "tests/test_security.py::test_authorize_allows"
TEST_ADAPTER = "tests/test_adapters.py::test_adapter_ping"
TEST_DYNAMIC = "tests/test_dynamic.py::test_load_plugin_name"
TEST_NATIVE = "tests/test_native.py::test_native_marker"
TEST_GENERATED = "tests/test_generated.py::test_generated_constant"

PROOF_CORE_ADD = "proof:sch_fixture.core:add"
PROOF_API_SIG = "proof:sch_fixture.api:fetch_value.signature"
PROOF_SCHEMA = "proof:sch_fixture.schema:UserRecord"
PROOF_SECURITY = "proof:sch_fixture.security:authorize.effects"
PROOF_OPAQUE = "proof:sch_fixture.native_bridge:native_hash.raw"

FULL_SUITE = tuple(
    sorted(
        {
            TEST_CORE_ADD,
            TEST_CORE_PROCESS,
            TEST_API_FETCH,
            TEST_API_CALL,
            TEST_SCHEMA,
            TEST_SECURITY,
            TEST_ADAPTER,
            TEST_DYNAMIC,
            TEST_NATIVE,
            TEST_GENERATED,
        }
    )
)

# Marker bytes used only by the post-scan source-race case. These must never
# appear in an admitted context pack for that case.
SOURCE_RACE_MARKER = b"SOURCE_RACE_BYTES_MUST_NOT_ENTER_PACK_v1\n"
SOURCE_RACE_PATH = "src/sch_fixture/core.py"


def _base_core() -> str:
    return '''\
"""Pure numeric helpers for the controlled fixture."""

from __future__ import annotations


def add(left: int, right: int) -> int:
    """Return the sum of two integers."""
    return left + right


def multiply(left: int, right: int) -> int:
    """Return the product of two integers."""
    return left * right


def process(value: int) -> int:
    """Apply the default processing pipeline."""
    return add(value, 1)


def legacy_helper(value: int) -> int:
    """Legacy helper retained for delete/rename cases."""
    return value
'''


def _base_api() -> str:
    return '''\
"""Public API surface that calls into core helpers."""

from __future__ import annotations

from sch_fixture.core import add, process


def fetch_value(key: str, default: int = 0) -> int:
    """Fetch a named value; default is the identity seed."""
    if key == "seed":
        return default
    return process(default)


def call_core(left: int, right: int) -> int:
    """Cross-module call into core.add."""
    return add(left, right)
'''


def _base_schema() -> str:
    return '''\
"""Dataclass schema used by serialization edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    score: int


def dump_user(record: UserRecord) -> Mapping[str, Any]:
    """Serialize a user record to a plain mapping."""
    return asdict(record)
'''


def _base_security() -> str:
    return '''\
"""Side-effect / security boundary helpers."""

from __future__ import annotations


def authorize(role: str, action: str) -> bool:
    """Return True when role may perform action."""
    if role == "admin":
        return True
    return action == "read"
'''


def _base_adapters() -> str:
    return '''\
"""MCP interface client adapter (syntactic fixture only)."""

from __future__ import annotations

from typing import Any, Mapping


class McpClientAdapter:
    """Thin client adapter bound to the interface descriptor."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def ping(self) -> Mapping[str, Any]:
        return {"ok": True, "endpoint": self.endpoint}
'''


def _base_dynamic() -> str:
    return '''\
"""Dynamic import site represented syntactically for scanners.

The harness must not import or execute this module during analysis.
"""

from __future__ import annotations

from typing import Any


def load_plugin(module_name: str) -> Any:
    """Dynamic import placeholder; scanners treat this as opaque/heuristic."""
    # Syntactic dynamic import only — fixture scans must not execute this.
    return __import__(module_name)
'''


def _base_native() -> str:
    return '''\
"""Opaque native dependency boundary (syntactic; never loaded).

Represented for confidence/raw-source oracles. No native extension is shipped.
"""

from __future__ import annotations


def native_hash(payload: bytes) -> str:
    """Opaque native hash boundary — body is intentionally non-analyzable."""
    # NOTE: real native code would live out-of-tree; this is a marker body.
    raise RuntimeError("native_hash is opaque fixture surface")
'''


def _base_generated() -> str:
    return '''\
"""Generated bindings (marker file for generated-file mutations)."""

from __future__ import annotations

# sch-fixture-generated: do-not-edit
generated_constant: int = 42
'''


def _base_init() -> str:
    return '''\
"""sch_fixture controlled package."""

from __future__ import annotations

__all__ = ["api", "core", "schema", "security"]
'''


def _base_generated_init() -> str:
    return '''\
"""Generated package marker."""
'''


def _base_conftest() -> str:
    return '''\
"""Pytest fixtures for the controlled repository."""

from __future__ import annotations

import pytest

from sch_fixture.schema import UserRecord


@pytest.fixture
def sample_record() -> UserRecord:
    return UserRecord(user_id="u-1", score=3)
'''


def _base_test_core() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.core import add, process


def test_add() -> None:
    assert add(2, 3) == 5


def test_process() -> None:
    assert process(4) == 5
'''


def _base_test_api() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.api import call_core, fetch_value


def test_fetch_value() -> None:
    assert fetch_value("seed", default=7) == 7


def test_call_core() -> None:
    assert call_core(2, 3) == 5
'''


def _base_test_schema() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.schema import UserRecord, dump_user


def test_user_roundtrip(sample_record: UserRecord) -> None:
    payload = dump_user(sample_record)
    assert payload["user_id"] == "u-1"
    assert payload["score"] == 3
'''


def _base_test_security() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.security import authorize


def test_authorize_allows() -> None:
    assert authorize("admin", "write") is True
    assert authorize("user", "read") is True
'''


def _base_test_adapters() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.adapters import McpClientAdapter


def test_adapter_ping() -> None:
    assert McpClientAdapter("local").ping()["ok"] is True
'''


def _base_test_dynamic() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.dynamic_loader import load_plugin


def test_load_plugin_name() -> None:
    # Tests may execute only inside a fenced worktree; scanners must not.
    assert callable(load_plugin)
'''


def _base_test_native() -> str:
    return '''\
from __future__ import annotations

from sch_fixture import native_bridge


def test_native_marker() -> None:
    assert hasattr(native_bridge, "native_hash")
'''


def _base_test_generated() -> str:
    return '''\
from __future__ import annotations

from sch_fixture.generated.bindings import generated_constant


def test_generated_constant() -> None:
    assert generated_constant == 42
'''


def base_tree_files() -> dict[str, str]:
    """Return the deterministic base repository path -> text mapping."""

    return {
        "pyproject.toml": (
            "[project]\n"
            'name = "sch-fixture"\n'
            'version = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
            "dependencies = []\n\n"
            "[build-system]\n"
            'requires = ["setuptools>=68"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[tool.setuptools.packages.find]\n"
            'where = ["src"]\n'
        ),
        "pytest.ini": (
            "[pytest]\n"
            "testpaths = tests\n"
            "pythonpath = src\n"
            "addopts = -q\n"
        ),
        "requirements.txt": "pytest==8.3.4\n",
        "requirements.lock": (
            "# sch-fixture lockfile v1\n"
            "pytest==8.3.4 \\\n"
            "    --hash=sha256:schfixturelockhash0001\n"
        ),
        "policy/admission.json": (
            '{\n'
            '  "schema": "sch-fixture/policy.admission@1",\n'
            '  "mode": "enforce",\n'
            '  "allow_simulation": false\n'
            '}\n'
        ),
        "interfaces/mcp_client.json": (
            '{\n'
            '  "schema": "sch-fixture/interfaces.mcp_client@1",\n'
            '  "operations": ["ping", "fetch_value"],\n'
            '  "version": "1"\n'
            '}\n'
        ),
        "src/sch_fixture/__init__.py": _base_init(),
        "src/sch_fixture/core.py": _base_core(),
        "src/sch_fixture/api.py": _base_api(),
        "src/sch_fixture/schema.py": _base_schema(),
        "src/sch_fixture/security.py": _base_security(),
        "src/sch_fixture/adapters.py": _base_adapters(),
        "src/sch_fixture/dynamic_loader.py": _base_dynamic(),
        "src/sch_fixture/native_bridge.py": _base_native(),
        "src/sch_fixture/generated/__init__.py": _base_generated_init(),
        "src/sch_fixture/generated/bindings.py": _base_generated(),
        "tests/conftest.py": _base_conftest(),
        "tests/test_core.py": _base_test_core(),
        "tests/test_api.py": _base_test_api(),
        "tests/test_schema.py": _base_test_schema(),
        "tests/test_security.py": _base_test_security(),
        "tests/test_adapters.py": _base_test_adapters(),
        "tests/test_dynamic.py": _base_test_dynamic(),
        "tests/test_native.py": _base_test_native(),
        "tests/test_generated.py": _base_test_generated(),
    }


def _oracle(
    *,
    symbols: list[str],
    primary: str,
    kinds: list[str],
    merkle_nodes: list[str],
    merkle_paths: list[str],
    root_changes: bool,
    invalidation: list[str],
    selected_tests: list[str],
    proofs: list[str],
    full_tests: list[str] | None = None,
    fallback: str = "none",
    false_negatives: int = 0,
    freshness: str = "current",
    accepts_stale: bool = False,
    binds_tree: bool = True,
    binds_config: bool = True,
    freshness_reasons: list[str] | None = None,
    confidence: str = "exact",
    raw_required: bool = False,
    raw_symbols: list[str] | None = None,
    confidence_reasons: list[str] | None = None,
) -> FixtureOracle:
    return FixtureOracle(
        changed_symbol=ChangedSymbolOracle(
            symbol_ids=tuple(sorted(symbols)),
            primary_symbol_id=primary,
            change_kinds=tuple(sorted(kinds)),
        ),
        merkle=MerkleOracle(
            changed_node_ids=tuple(sorted(merkle_nodes)),
            affected_path_ids=tuple(sorted(merkle_paths)),
            root_changes=root_changes,
        ),
        invalidation=InvalidationOracle(
            invalidation_symbol_ids=tuple(sorted(invalidation)),
            selected_test_node_ids=tuple(sorted(selected_tests)),
            proof_obligation_ids=tuple(sorted(proofs)),
            full_suite_test_node_ids=tuple(sorted(full_tests or list(FULL_SUITE))),
            fallback=fallback,
            expected_false_negatives=false_negatives,
        ),
        receipt_freshness=ReceiptFreshnessOracle(
            disposition=freshness,
            accepts_stale_receipt=accepts_stale,
            binds_tree_cid=binds_tree,
            binds_config_cid=binds_config,
            reason_codes=tuple(sorted(freshness_reasons or ["tree_bound"])),
        ),
        confidence=ConfidenceOracle(
            confidence=confidence,
            raw_source_required=raw_required,
            raw_source_symbol_ids=tuple(sorted(raw_symbols or [])),
            reason_codes=tuple(
                sorted(
                    confidence_reasons
                    or (
                        ["raw_source_required"]
                        if raw_required
                        else ["exact_static_body"]
                    )
                )
            ),
        ),
    )


def _replace(path: str, content: str) -> PathOperation:
    return PathOperation(op="replace", path=path, content=content)


def _add(path: str, content: str) -> PathOperation:
    return PathOperation(op="add", path=path, content=content)


def _delete(path: str) -> PathOperation:
    return PathOperation(op="delete", path=path)


def _rename(from_path: str, to_path: str) -> PathOperation:
    return PathOperation(op="rename", path=to_path, from_path=from_path)


def mutation_cases() -> tuple[MutationCase, ...]:
    """Closed catalogue of base/mutated cases with independent oracles."""

    cases: list[MutationCase] = []

    # 1. Local function body change
    body_core = _base_core().replace(
        "return left + right",
        "return left + right + 0  # body-only change",
        1,
    )
    cases.append(
        MutationCase(
            case_id="local_function_body",
            category="local_function_body",
            description="Local body change to core.add without signature drift",
            operations=(_replace("src/sch_fixture/core.py", body_core),),
            oracle=_oracle(
                symbols=[SYM_CORE_ADD],
                primary=SYM_CORE_ADD,
                kinds=["body"],
                merkle_nodes=["node:sch_fixture.core:add", "node:src/sch_fixture/core.py"],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_ADD],
                selected_tests=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 2. Public signature change
    sig_api = _base_api().replace(
        "def fetch_value(key: str, default: int = 0) -> int:",
        "def fetch_value(key: str, default: int = 0, *, strict: bool = False) -> int:",
        1,
    ).replace(
        "if key == \"seed\":\n        return default\n    return process(default)\n",
        "if strict and key not in {\"seed\", \"live\"}:\n"
        "        raise KeyError(key)\n"
        "    if key == \"seed\":\n"
        "        return default\n"
        "    return process(default)\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="public_signature",
            category="public_signature",
            description="Public signature change on api.fetch_value",
            operations=(_replace("src/sch_fixture/api.py", sig_api),),
            oracle=_oracle(
                symbols=[SYM_API_FETCH],
                primary=SYM_API_FETCH,
                kinds=["signature"],
                merkle_nodes=[
                    "node:sch_fixture.api:fetch_value",
                    "node:src/sch_fixture/api.py",
                ],
                merkle_paths=["src/sch_fixture/api.py"],
                root_changes=True,
                invalidation=[SYM_API_FETCH, SYM_ADAPTER_CLIENT],
                selected_tests=[TEST_API_FETCH, TEST_ADAPTER],
                proofs=[PROOF_API_SIG],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 3. Cross-module call change
    cross_api = _base_api().replace(
        "from sch_fixture.core import add, process",
        "from sch_fixture.core import multiply, process",
        1,
    ).replace(
        "return add(left, right)",
        "return multiply(left, right)",
        1,
    )
    cases.append(
        MutationCase(
            case_id="cross_module_call",
            category="cross_module_call",
            description="api.call_core switches from add to multiply",
            operations=(_replace("src/sch_fixture/api.py", cross_api),),
            oracle=_oracle(
                symbols=[SYM_API_CALL_CORE],
                primary=SYM_API_CALL_CORE,
                kinds=["call", "body"],
                merkle_nodes=[
                    "node:sch_fixture.api:call_core",
                    "node:src/sch_fixture/api.py",
                ],
                merkle_paths=["src/sch_fixture/api.py"],
                root_changes=True,
                invalidation=[SYM_API_CALL_CORE, SYM_CORE_MULTIPLY],
                selected_tests=[TEST_API_CALL],
                proofs=[],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 4. Dataclass / schema change
    schema_mut = _base_schema().replace(
        "user_id: str\n    score: int\n",
        "user_id: str\n    score: int\n    active: bool = True\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="dataclass_schema",
            category="dataclass_schema",
            description="UserRecord schema gains an active field",
            operations=(_replace("src/sch_fixture/schema.py", schema_mut),),
            oracle=_oracle(
                symbols=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP],
                primary=SYM_SCHEMA_USER,
                kinds=["schema", "signature"],
                merkle_nodes=[
                    "node:sch_fixture.schema:UserRecord",
                    "node:src/sch_fixture/schema.py",
                ],
                merkle_paths=["src/sch_fixture/schema.py"],
                root_changes=True,
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP, SYM_FIXTURE_SAMPLE],
                selected_tests=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 5. Exception behavior change
    exc_api = _base_api().replace(
        "if key == \"seed\":\n        return default\n    return process(default)\n",
        "if key == \"seed\":\n        return default\n"
        "    if key == \"missing\":\n"
        "        raise KeyError(key)\n"
        "    return process(default)\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="exception_behavior",
            category="exception_behavior",
            description="fetch_value raises KeyError for missing keys",
            operations=(_replace("src/sch_fixture/api.py", exc_api),),
            oracle=_oracle(
                symbols=[SYM_API_FETCH],
                primary=SYM_API_FETCH,
                kinds=["exception", "body"],
                merkle_nodes=[
                    "node:sch_fixture.api:fetch_value",
                    "node:src/sch_fixture/api.py",
                ],
                merkle_paths=["src/sch_fixture/api.py"],
                root_changes=True,
                invalidation=[SYM_API_FETCH],
                selected_tests=[TEST_API_FETCH],
                proofs=[PROOF_API_SIG],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 6. Side-effect / security policy change
    sec_mut = _base_security().replace(
        "if role == \"admin\":\n        return True\n    return action == \"read\"\n",
        "if role == \"admin\":\n        return True\n"
        "    if role == \"auditor\":\n"
        "        return action in {\"read\", \"audit\"}\n"
        "    return action == \"read\"\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="side_effect_security",
            category="side_effect_security",
            description="authorize gains auditor side-effect policy branch",
            operations=(_replace("src/sch_fixture/security.py", sec_mut),),
            oracle=_oracle(
                symbols=[SYM_SEC_AUTHORIZE],
                primary=SYM_SEC_AUTHORIZE,
                kinds=["effect", "body"],
                merkle_nodes=[
                    "node:sch_fixture.security:authorize",
                    "node:src/sch_fixture/security.py",
                ],
                merkle_paths=["src/sch_fixture/security.py"],
                root_changes=True,
                invalidation=[SYM_SEC_AUTHORIZE],
                selected_tests=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 7. Fixture dependency change
    fixture_mut = _base_conftest().replace(
        'return UserRecord(user_id="u-1", score=3)\n',
        'return UserRecord(user_id="u-1", score=9)\n',
        1,
    )
    cases.append(
        MutationCase(
            case_id="fixture_dependency",
            category="fixture_dependency",
            description="sample_record fixture value changes",
            operations=(_replace("tests/conftest.py", fixture_mut),),
            oracle=_oracle(
                symbols=[SYM_FIXTURE_SAMPLE],
                primary=SYM_FIXTURE_SAMPLE,
                kinds=["fixture"],
                merkle_nodes=["node:tests.conftest:sample_record", "node:tests/conftest.py"],
                merkle_paths=["tests/conftest.py"],
                root_changes=True,
                invalidation=[SYM_FIXTURE_SAMPLE, SYM_SCHEMA_USER],
                selected_tests=[TEST_SCHEMA],
                proofs=[],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 8. Pytest configuration change
    pytest_mut = (
        "[pytest]\n"
        "testpaths = tests\n"
        "pythonpath = src\n"
        "addopts = -q --strict-markers\n"
        "markers =\n"
        "    slow: optional slow tests\n"
    )
    cases.append(
        MutationCase(
            case_id="pytest_configuration",
            category="pytest_configuration",
            description="pytest.ini gains strict markers configuration",
            operations=(_replace("pytest.ini", pytest_mut),),
            oracle=_oracle(
                symbols=[SYM_PYTEST_CFG],
                primary=SYM_PYTEST_CFG,
                kinds=["configuration"],
                merkle_nodes=["node:pytest.ini", "node:pytest.ini:pytest_config"],
                merkle_paths=["pytest.ini"],
                root_changes=True,
                invalidation=[SYM_PYTEST_CFG],
                selected_tests=list(FULL_SUITE),
                proofs=[],
                fallback="full_pytest",
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 9. Dependency / lockfile change
    lock_mut = (
        "# sch-fixture lockfile v1\n"
        "pytest==8.3.5 \\\n"
        "    --hash=sha256:schfixturelockhash0002\n"
    )
    cases.append(
        MutationCase(
            case_id="dependency_lockfile",
            category="dependency_lockfile",
            description="Lockfile pin for pytest advances",
            operations=(
                _replace("requirements.lock", lock_mut),
                _replace("requirements.txt", "pytest==8.3.5\n"),
            ),
            oracle=_oracle(
                symbols=[SYM_LOCK],
                primary=SYM_LOCK,
                kinds=["dependency"],
                merkle_nodes=[
                    "node:requirements.lock",
                    "node:deps.lockfile:LockedDependencySet",
                ],
                merkle_paths=["requirements.lock", "requirements.txt"],
                root_changes=True,
                invalidation=[SYM_LOCK],
                selected_tests=list(FULL_SUITE),
                proofs=[],
                fallback="full_pytest",
                freshness="current",
                freshness_reasons=["dependency_bound", "tree_bound"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 10. Policy change
    policy_mut = (
        "{\n"
        '  "schema": "sch-fixture/policy.admission@1",\n'
        '  "mode": "enforce",\n'
        '  "allow_simulation": false,\n'
        '  "require_fresh_receipts": true\n'
        "}\n"
    )
    cases.append(
        MutationCase(
            case_id="policy_change",
            category="policy",
            description="Admission policy requires fresh receipts",
            operations=(_replace("policy/admission.json", policy_mut),),
            oracle=_oracle(
                symbols=[SYM_POLICY],
                primary=SYM_POLICY,
                kinds=["policy"],
                merkle_nodes=["node:policy/admission.json", "node:policy.admission"],
                merkle_paths=["policy/admission.json"],
                root_changes=True,
                invalidation=[SYM_POLICY],
                selected_tests=[],
                proofs=[],
                full_tests=list(FULL_SUITE),
                freshness="current",
                freshness_reasons=["policy_bound", "tree_bound"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 11. MCP interface / client adapter change
    iface_mut = (
        "{\n"
        '  "schema": "sch-fixture/interfaces.mcp_client@1",\n'
        '  "operations": ["ping", "fetch_value", "explain"],\n'
        '  "version": "2"\n'
        "}\n"
    )
    adapter_mut = _base_adapters().replace(
        "def ping(self) -> Mapping[str, Any]:\n"
        "        return {\"ok\": True, \"endpoint\": self.endpoint}\n",
        "def ping(self) -> Mapping[str, Any]:\n"
        "        return {\"ok\": True, \"endpoint\": self.endpoint}\n\n"
        "    def explain(self, symbol_id: str) -> Mapping[str, Any]:\n"
        "        return {\"symbol_id\": symbol_id, \"endpoint\": self.endpoint}\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="mcp_interface_client_adapter",
            category="mcp_interface_client_adapter",
            description="MCP interface descriptor and client adapter gain explain",
            operations=(
                _replace("interfaces/mcp_client.json", iface_mut),
                _replace("src/sch_fixture/adapters.py", adapter_mut),
            ),
            oracle=_oracle(
                symbols=[SYM_IFACE, SYM_ADAPTER_CLIENT],
                primary=SYM_IFACE,
                kinds=["interface", "signature"],
                merkle_nodes=[
                    "node:interfaces/mcp_client.json",
                    "node:sch_fixture.adapters:McpClientAdapter",
                    "node:src/sch_fixture/adapters.py",
                ],
                merkle_paths=[
                    "interfaces/mcp_client.json",
                    "src/sch_fixture/adapters.py",
                ],
                root_changes=True,
                invalidation=[SYM_IFACE, SYM_ADAPTER_CLIENT],
                selected_tests=[TEST_ADAPTER],
                proofs=[],
                freshness_reasons=["interface_bound", "tree_bound"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 12. Dynamic import
    dynamic_mut = _base_dynamic().replace(
        "return __import__(module_name)\n",
        "import importlib\n\n    return importlib.import_module(module_name)\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="dynamic_import",
            category="dynamic_import",
            description="load_plugin switches dynamic import implementation",
            operations=(_replace("src/sch_fixture/dynamic_loader.py", dynamic_mut),),
            oracle=_oracle(
                symbols=[SYM_DYNAMIC_LOAD],
                primary=SYM_DYNAMIC_LOAD,
                kinds=["body", "dynamic"],
                merkle_nodes=[
                    "node:sch_fixture.dynamic_loader:load_plugin",
                    "node:src/sch_fixture/dynamic_loader.py",
                ],
                merkle_paths=["src/sch_fixture/dynamic_loader.py"],
                root_changes=True,
                invalidation=[SYM_DYNAMIC_LOAD],
                selected_tests=[TEST_DYNAMIC],
                proofs=[],
                confidence="heuristic",
                raw_required=True,
                raw_symbols=[SYM_DYNAMIC_LOAD],
                confidence_reasons=["dynamic_import", "raw_source_required"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 13. Monkey patch surface (test-level patch target change)
    monkey_test = (
        "from __future__ import annotations\n\n"
        "from sch_fixture.core import add\n\n\n"
        "def test_add(monkeypatch) -> None:\n"
        "    monkeypatch.setattr(\"sch_fixture.core.add\", lambda a, b: a + b + 1)\n"
        "    assert add(2, 3) == 6\n\n\n"
        "def test_process() -> None:\n"
        "    from sch_fixture.core import process\n\n"
        "    assert process(4) == 5\n"
    )
    cases.append(
        MutationCase(
            case_id="monkey_patch",
            category="monkey_patch",
            description="test_add applies a monkeypatch to core.add",
            operations=(_replace("tests/test_core.py", monkey_test),),
            oracle=_oracle(
                symbols=[SYM_CORE_ADD],
                primary=SYM_CORE_ADD,
                kinds=["test", "monkeypatch"],
                merkle_nodes=["node:tests/test_core.py", "node:sch_fixture.core:add"],
                merkle_paths=["tests/test_core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_ADD],
                selected_tests=[TEST_CORE_ADD],
                proofs=[],
                confidence="heuristic",
                raw_required=True,
                raw_symbols=[SYM_CORE_ADD],
                confidence_reasons=["monkeypatch", "raw_source_required"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 14. Opaque native dependency
    native_mut = _base_native().replace(
        "raise RuntimeError(\"native_hash is opaque fixture surface\")\n",
        "raise RuntimeError(\"native_hash is opaque fixture surface v2\")\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="opaque_native",
            category="opaque_native",
            description="Opaque native bridge body marker changes",
            operations=(_replace("src/sch_fixture/native_bridge.py", native_mut),),
            oracle=_oracle(
                symbols=[SYM_NATIVE_BRIDGE],
                primary=SYM_NATIVE_BRIDGE,
                kinds=["body", "opaque"],
                merkle_nodes=[
                    "node:sch_fixture.native_bridge:native_hash",
                    "node:src/sch_fixture/native_bridge.py",
                ],
                merkle_paths=["src/sch_fixture/native_bridge.py"],
                root_changes=True,
                invalidation=[SYM_NATIVE_BRIDGE],
                selected_tests=[TEST_NATIVE],
                proofs=[PROOF_OPAQUE],
                confidence="opaque",
                raw_required=True,
                raw_symbols=[SYM_NATIVE_BRIDGE],
                confidence_reasons=["opaque_native", "raw_source_required"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 15. Unrelated formatting change (bounded)
    fmt_core = _base_core().replace(
        "def multiply(left: int, right: int) -> int:\n"
        "    \"\"\"Return the product of two integers.\"\"\"\n"
        "    return left * right\n",
        "def multiply(left: int, right: int) -> int:\n"
        "    \"\"\"Return the product of two integers.\"\"\"\n"
        "    return left * right  # formatted\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="unrelated_formatting",
            category="unrelated_formatting",
            description="Whitespace/comment-only formatting on multiply remains bounded",
            operations=(_replace("src/sch_fixture/core.py", fmt_core),),
            oracle=_oracle(
                symbols=[SYM_CORE_MULTIPLY],
                primary=SYM_CORE_MULTIPLY,
                kinds=["formatting"],
                merkle_nodes=[
                    "node:sch_fixture.core:multiply",
                    "node:src/sch_fixture/core.py",
                ],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[],
                selected_tests=[],
                proofs=[],
                full_tests=list(FULL_SUITE),
                confidence="exact",
                confidence_reasons=["formatting_only"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 16. Deleted symbol
    delete_core = _base_core().replace(
        "\n\ndef legacy_helper(value: int) -> int:\n"
        "    \"\"\"Legacy helper retained for delete/rename cases.\"\"\"\n"
        "    return value\n",
        "\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="deleted_symbol",
            category="deleted_symbol",
            description="legacy_helper is deleted from core",
            operations=(_replace("src/sch_fixture/core.py", delete_core),),
            oracle=_oracle(
                symbols=[SYM_DELETED],
                primary=SYM_DELETED,
                kinds=["delete"],
                merkle_nodes=[
                    "node:sch_fixture.core:legacy_helper",
                    "node:src/sch_fixture/core.py",
                ],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_DELETED],
                selected_tests=[TEST_CORE_PROCESS],
                proofs=[],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 17. Renamed symbol
    rename_core = _base_core().replace(
        "def process(value: int) -> int:",
        "def renamed_process(value: int) -> int:",
        1,
    )
    rename_api = _base_api().replace(
        "from sch_fixture.core import add, process",
        "from sch_fixture.core import add, renamed_process as process",
        1,
    )
    cases.append(
        MutationCase(
            case_id="renamed_symbol",
            category="renamed_symbol",
            description="process is renamed to renamed_process with import alias",
            operations=(
                _replace("src/sch_fixture/core.py", rename_core),
                _replace("src/sch_fixture/api.py", rename_api),
            ),
            oracle=_oracle(
                symbols=[SYM_CORE_PROCESS, SYM_RENAMED, SYM_API_FETCH],
                primary=SYM_CORE_PROCESS,
                kinds=["rename", "signature"],
                merkle_nodes=[
                    "node:sch_fixture.core:process",
                    "node:sch_fixture.core:renamed_process",
                    "node:src/sch_fixture/core.py",
                    "node:src/sch_fixture/api.py",
                ],
                merkle_paths=["src/sch_fixture/api.py", "src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_PROCESS, SYM_RENAMED, SYM_API_FETCH],
                selected_tests=[TEST_CORE_PROCESS, TEST_API_FETCH],
                proofs=[],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 18. Generated file
    gen_mut = _base_generated().replace(
        "generated_constant: int = 42\n",
        "generated_constant: int = 43\n",
        1,
    )
    cases.append(
        MutationCase(
            case_id="generated_file",
            category="generated_file",
            description="Generated bindings constant advances",
            operations=(
                _replace("src/sch_fixture/generated/bindings.py", gen_mut),
            ),
            oracle=_oracle(
                symbols=[SYM_GENERATED_BIND],
                primary=SYM_GENERATED_BIND,
                kinds=["generated", "body"],
                merkle_nodes=[
                    "node:sch_fixture.generated.bindings:generated_constant",
                    "node:src/sch_fixture/generated/bindings.py",
                ],
                merkle_paths=["src/sch_fixture/generated/bindings.py"],
                root_changes=True,
                invalidation=[SYM_GENERATED_BIND],
                selected_tests=[TEST_GENERATED],
                proofs=[],
                confidence="conservative",
                confidence_reasons=["generated_file"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
        )
    )

    # 19. Stale receipt (harness scenario; tiny source touch for tree binding)
    cases.append(
        MutationCase(
            case_id="stale_receipt",
            category="stale_receipt",
            description="Receipt forged against a prior tree must be rejected",
            operations=(
                _replace(
                    "src/sch_fixture/core.py",
                    _base_core().replace(
                        "return left + right",
                        "return left + right  # stale-receipt tree bump",
                        1,
                    ),
                ),
            ),
            oracle=_oracle(
                symbols=[SYM_CORE_ADD],
                primary=SYM_CORE_ADD,
                kinds=["body"],
                merkle_nodes=[
                    "node:sch_fixture.core:add",
                    "node:src/sch_fixture/core.py",
                ],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_ADD],
                selected_tests=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                freshness="reject_stale",
                accepts_stale=False,
                freshness_reasons=["stale_receipt", "tree_mismatch"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
            harness_scenario="stale_receipt",
        )
    )

    # 20. Failed / ABA CAS
    cases.append(
        MutationCase(
            case_id="failed_aba_cas",
            category="failed_aba_cas",
            description="ABA CAS writer must lose against a newer generation",
            operations=(
                _replace(
                    "src/sch_fixture/core.py",
                    _base_core().replace(
                        "return left * right",
                        "return left * right  # cas generation bump",
                        1,
                    ),
                ),
            ),
            oracle=_oracle(
                symbols=[SYM_CORE_MULTIPLY],
                primary=SYM_CORE_MULTIPLY,
                kinds=["body"],
                merkle_nodes=[
                    "node:sch_fixture.core:multiply",
                    "node:src/sch_fixture/core.py",
                ],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_MULTIPLY],
                selected_tests=[TEST_CORE_ADD],
                proofs=[],
                freshness="cas_reject",
                accepts_stale=False,
                freshness_reasons=["aba_cas", "generation_mismatch"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
            harness_scenario="failed_aba_cas",
        )
    )

    # 21. Interrupted state transition
    cases.append(
        MutationCase(
            case_id="interrupted_state_transition",
            category="interrupted_state_transition",
            description="Interrupted write leaves current root unchanged",
            operations=(
                _replace(
                    "src/sch_fixture/api.py",
                    _base_api().replace(
                        "return process(default)",
                        "return process(default)  # interrupted candidate",
                        1,
                    ),
                ),
            ),
            oracle=_oracle(
                symbols=[SYM_API_FETCH],
                primary=SYM_API_FETCH,
                kinds=["body"],
                merkle_nodes=[
                    "node:sch_fixture.api:fetch_value",
                    "node:src/sch_fixture/api.py",
                ],
                merkle_paths=["src/sch_fixture/api.py"],
                root_changes=True,
                invalidation=[SYM_API_FETCH],
                selected_tests=[TEST_API_FETCH],
                proofs=[],
                freshness="interrupted",
                accepts_stale=False,
                freshness_reasons=["interrupted_write", "root_unchanged"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
            harness_scenario="interrupted_state_transition",
        )
    )

    # 22. Concurrent watchers / writers
    cases.append(
        MutationCase(
            case_id="concurrent_watchers_writers",
            category="concurrent_watchers_writers",
            description="Concurrent watchers only request rescans; writers fence CAS",
            operations=(
                _replace(
                    "src/sch_fixture/security.py",
                    _base_security().replace(
                        "return action == \"read\"",
                        "return action == \"read\"  # concurrent fence",
                        1,
                    ),
                ),
            ),
            oracle=_oracle(
                symbols=[SYM_SEC_AUTHORIZE],
                primary=SYM_SEC_AUTHORIZE,
                kinds=["body"],
                merkle_nodes=[
                    "node:sch_fixture.security:authorize",
                    "node:src/sch_fixture/security.py",
                ],
                merkle_paths=["src/sch_fixture/security.py"],
                root_changes=True,
                invalidation=[SYM_SEC_AUTHORIZE],
                selected_tests=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                freshness="concurrent_fence",
                accepts_stale=False,
                freshness_reasons=["concurrent_writer", "lease_fence"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(),
            harness_scenario="concurrent_watchers_writers",
        )
    )

    # 23. Post-scan source race — race bytes must never enter a pack
    race_core = _base_core() + "\n# " + SOURCE_RACE_MARKER.decode("ascii")
    cases.append(
        MutationCase(
            case_id="post_scan_source_race",
            category="post_scan_source_race",
            description=(
                "Filesystem bytes diverge after scan; race payload must not enter packs"
            ),
            operations=(_replace(SOURCE_RACE_PATH, race_core),),
            oracle=_oracle(
                symbols=[SYM_CORE_ADD],
                primary=SYM_CORE_ADD,
                kinds=["source_race"],
                merkle_nodes=["node:src/sch_fixture/core.py"],
                merkle_paths=["src/sch_fixture/core.py"],
                root_changes=True,
                invalidation=[SYM_CORE_ADD],
                selected_tests=[TEST_CORE_ADD],
                proofs=[],
                freshness="require_rescan",
                accepts_stale=False,
                freshness_reasons=["source_race", "require_rescan"],
                confidence="opaque",
                raw_required=True,
                raw_symbols=[SYM_CORE_ADD],
                confidence_reasons=["source_race", "raw_source_required"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=(SOURCE_RACE_PATH,),
            harness_scenario="post_scan_source_race",
        )
    )

    # 24. Out-of-scope model patch
    oos = _add(
        "docs/out_of_scope.md",
        "# Out of scope patch target\n\nThis path is not admitted for model edits.\n",
    )
    cases.append(
        MutationCase(
            case_id="out_of_scope_patch",
            category="out_of_scope_patch",
            description="Model patch touches an out-of-scope documentation path",
            operations=(oos,),
            oracle=_oracle(
                symbols=["docs.out_of_scope:document"],
                primary="docs.out_of_scope:document",
                kinds=["out_of_scope"],
                merkle_nodes=["node:docs/out_of_scope.md"],
                merkle_paths=["docs/out_of_scope.md"],
                root_changes=True,
                invalidation=[],
                selected_tests=[],
                proofs=[],
                full_tests=list(FULL_SUITE),
                freshness="reject_stale",
                accepts_stale=False,
                freshness_reasons=["out_of_scope_path"],
                confidence="exact",
                confidence_reasons=["out_of_scope"],
            ),
            source_race_bytes_forbidden=True,
            change_is_bounded=True,
            pack_excluded_paths=("docs/out_of_scope.md",),
            harness_scenario="out_of_scope_patch",
        )
    )

    # Stable ordering by case_id for determinism.
    cases.sort(key=lambda item: item.case_id)
    return tuple(cases)


REQUIRED_CATEGORIES: tuple[str, ...] = tuple(
    sorted(
        {
            "local_function_body",
            "public_signature",
            "cross_module_call",
            "dataclass_schema",
            "exception_behavior",
            "side_effect_security",
            "fixture_dependency",
            "pytest_configuration",
            "dependency_lockfile",
            "policy",
            "mcp_interface_client_adapter",
            "dynamic_import",
            "monkey_patch",
            "opaque_native",
            "unrelated_formatting",
            "deleted_symbol",
            "renamed_symbol",
            "generated_file",
            "stale_receipt",
            "failed_aba_cas",
            "interrupted_state_transition",
            "concurrent_watchers_writers",
            "post_scan_source_race",
            "out_of_scope_patch",
        }
    )
)
