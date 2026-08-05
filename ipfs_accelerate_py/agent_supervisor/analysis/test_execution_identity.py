"""Locator and execution identity compiler (PTR-010).

Produces content-addressed locator and execution-key artifacts for
proof-backed test reuse.  Missing CID support returns typed non-reusable
artifacts rather than digest-as-CID fallbacks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    TestExecutionKey,
    TestLocatorKey,
)

TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE: Final = "TestExecutionIdentityCompiler@1"
CONTENT_IDENTITY_INTERFACE: Final = "ContentIdentity@1"
_CID_RE: Final = re.compile(r"^b[a-z2-7]{20,}$")
_DIGEST_RE: Final = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
_MAX_NODE_ID: Final = 1024


class CidSupportStatus(str, Enum):
    """Availability of real multiformat CIDv1 minting."""

    AVAILABLE = "available"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TestExecutionIdentityError(ValueError):
    """Typed failure for locator/execution identity construction."""


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """Retained canonical preimage with a verified CIDv1 (never digest-as-CID)."""

    cid: str
    canonical_bytes: bytes
    digest: str = ""
    profile: str = "strict-dag-json-v1"
    interface: str = CONTENT_IDENTITY_INTERFACE
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, (bytes, bytearray)):
            raise TestExecutionIdentityError("canonical_bytes must be bytes")
        object.__setattr__(self, "canonical_bytes", bytes(self.canonical_bytes))
        cid = str(self.cid or "").strip()
        if not cid or not _CID_RE.fullmatch(cid):
            raise TestExecutionIdentityError("cid must be a lowercase base32 CIDv1")
        if _DIGEST_RE.fullmatch(cid):
            raise TestExecutionIdentityError("digest-shaped values cannot be labeled as CID")
        object.__setattr__(self, "cid", cid)
        if not self.digest:
            digest = "sha256:" + hashlib.sha256(self.canonical_bytes).hexdigest()
            object.__setattr__(self, "digest", digest)

    def verify(self) -> None:
        """Rehash retained bytes and require the stored CID to match."""

        expected = mint_content_identity_bytes(self.canonical_bytes)
        if expected.cid != self.cid:
            raise TestExecutionIdentityError("content identity CID does not match retained bytes")
        if expected.digest != self.digest:
            raise TestExecutionIdentityError(
                "content identity digest does not match retained bytes"
            )


def reject_pseudo_cid(value: str, *, field_name: str = "cid") -> str:
    """Reject digest-shaped or empty strings that must never be treated as CIDs."""

    text = str(value or "").strip()
    if not text:
        raise TestExecutionIdentityError(f"{field_name} is required")
    if _DIGEST_RE.fullmatch(text) and not _CID_RE.fullmatch(text):
        raise TestExecutionIdentityError(f"{field_name} is digest-shaped, not a CIDv1")
    if not _CID_RE.fullmatch(text):
        raise TestExecutionIdentityError(f"{field_name} is not a valid CIDv1")
    return text


def normalize_pytest_node_id(node_id: Any) -> str:
    """Normalize a pytest node id for identity binding."""

    text = str(node_id or "").strip()
    if not text:
        raise TestExecutionIdentityError("node_id is required")
    if len(text) > _MAX_NODE_ID:
        raise TestExecutionIdentityError("node_id exceeds bounded length")
    # Collapse accidental double separators introduced by path joins.
    return text.replace("\\", "/")


def _probe_cid_support() -> CidSupportStatus:
    try:
        from ipfs_accelerate_py.agent_supervisor.analysis import content_identity_bridge as bridge

        if not bridge.multiformats_available():
            return CidSupportStatus.MISSING
        # Exercise a tiny mint to detect incompatible providers.
        identity = bridge.identify_strict_artifact({"ptr": "probe", "n": 1})
        if not identity.cid or not _CID_RE.fullmatch(identity.cid):
            return CidSupportStatus.INCOMPATIBLE
        return CidSupportStatus.AVAILABLE
    except Exception:
        return CidSupportStatus.MISSING


def mint_content_identity_bytes(data: bytes | bytearray | memoryview) -> ContentIdentity:
    """Mint identity for already-canonical bytes under the strict artifact profile."""

    retained = bytes(data)
    if not retained:
        raise TestExecutionIdentityError("canonical bytes must be nonempty")
    try:
        from ipfs_accelerate_py.agent_supervisor.analysis import content_identity_bridge as bridge

        identity = bridge.identify_strict_artifact_bytes(retained)
        return ContentIdentity(
            cid=identity.cid,
            canonical_bytes=identity.canonical_bytes,
            digest=identity.digest,
            profile=identity.profile,
            reason_codes=tuple(identity.reason_codes),
        )
    except Exception as exc:
        # Fail closed: never mint a digest-as-CID fallback for authority use.
        raise TestExecutionIdentityError(
            f"CID mint unavailable: {type(exc).__name__}"
        ) from exc


def mint_content_identity(value: Any) -> ContentIdentity:
    """Mint a retained ContentIdentity for a DAG-JSON-compatible value."""

    try:
        from ipfs_accelerate_py.agent_supervisor.analysis import content_identity_bridge as bridge

        identity = bridge.identify_strict_artifact(value)
        return ContentIdentity(
            cid=identity.cid,
            canonical_bytes=identity.canonical_bytes,
            digest=identity.digest,
            profile=identity.profile,
            reason_codes=tuple(identity.reason_codes),
        )
    except Exception:
        # Prefer bridge; if unavailable attempt only when multiformats/datasets
        # path failed for non-import reasons after canonicalization.
        try:
            retained = canonical_json_bytes(value)
            return mint_content_identity_bytes(retained)
        except TestExecutionIdentityError:
            raise
        except Exception as exc:
            raise TestExecutionIdentityError(
                f"content identity mint failed: {type(exc).__name__}"
            ) from exc


@dataclass(frozen=True, slots=True)
class LocatorCompileResult:
    """Result of :meth:`TestExecutionIdentityCompiler.compile_locator`."""

    reusable: bool
    locator: TestLocatorKey | None = None
    locator_cid: str = ""
    reason_code: str = ""
    content_identity: ContentIdentity | None = None

    @property
    def interface(self) -> str:
        return TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE


@dataclass(frozen=True, slots=True)
class ExecutionKeyCompileResult:
    """Result of :meth:`TestExecutionIdentityCompiler.compile_execution_key`."""

    reusable: bool
    execution_key: TestExecutionKey | None = None
    execution_key_cid: str = ""
    reason_code: str = ""
    content_identity: ContentIdentity | None = None

    @property
    def interface(self) -> str:
        return TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE


class TestExecutionIdentityCompiler:
    """Compile exact locator and execution-key artifacts for one pytest item."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE

    def __init__(
        self,
        *,
        cid_probe: Callable[[], CidSupportStatus] | None = None,
    ) -> None:
        self._cid_probe = cid_probe or _probe_cid_support

    def cid_support(self) -> CidSupportStatus:
        try:
            status = self._cid_probe()
        except Exception:
            return CidSupportStatus.UNKNOWN
        if isinstance(status, CidSupportStatus):
            return status
        try:
            return CidSupportStatus(str(status))
        except Exception:
            return CidSupportStatus.UNKNOWN

    def compile_locator(
        self,
        *,
        repository_id: str,
        package_identity: str,
        node_id: str,
        collection_schema_version: str = "1",
        parameter_id: str = "",
        parameter_values_cid: str = "",
        non_reusable_reason: str = "",
        selection_semantics: str = "exact_node",
        root_identity: str = "",
        metadata: Mapping[str, Any] | None = None,
        **_extra: Any,
    ) -> LocatorCompileResult:
        """Compile a stable :class:`TestLocatorKey` identity."""

        if self.cid_support() is not CidSupportStatus.AVAILABLE:
            return LocatorCompileResult(
                reusable=False,
                reason_code="cid_support_unavailable",
            )
        try:
            locator = TestLocatorKey(
                repository_id=str(repository_id or ""),
                package_identity=str(package_identity or ""),
                root_identity=str(root_identity or package_identity or ""),
                node_id=normalize_pytest_node_id(node_id),
                collection_schema_version=str(collection_schema_version or "1"),
                parameter_id=str(parameter_id or ""),
                parameter_values_cid=str(parameter_values_cid or ""),
                non_reusable_reason=str(non_reusable_reason or ""),
                selection_semantics=str(selection_semantics or "exact_node"),
                metadata=dict(metadata or {}),
            )
            identity = mint_content_identity(locator.to_dict())
        except Exception as exc:
            return LocatorCompileResult(
                reusable=False,
                reason_code=f"locator_compile_failed:{type(exc).__name__}"[:96],
            )
        return LocatorCompileResult(
            reusable=True,
            locator=locator,
            locator_cid=identity.cid,
            content_identity=identity,
        )

    def compile_execution_key(
        self,
        *,
        locator_cid: str,
        repository_forest_cid: str,
        git_commit_id: str = "",
        git_tree_id: str = "",
        gitlink_state_cid: str = "",
        dirty_overlay_cid: str = "",
        test_module_cid: str = "",
        test_class_cid: str = "",
        test_function_cid: str = "",
        decorator_cids: Sequence[str] = (),
        parameter_source_cid: str = "",
        test_ast_cid: str = "",
        fixture_cids: Sequence[str] = (),
        conftest_closure_cid: str = "",
        hook_plugin_cids: Sequence[str] = (),
        static_trace_root_cid: str = "",
        static_unknown_frontier: Sequence[str] = (),
        runtime_trace_root_cid: str = "",
        runtime_completeness_policy: str = "",
        pytest_version: str = "",
        python_version: str = "",
        plugin_versions_cid: str = "",
        command_semantics_cid: str = "",
        config_cid: str = "",
        markers: Sequence[str] = (),
        dependency_lock_cid: str = "",
        installed_distributions_cid: str = "",
        environment_cid: str = "",
        platform_cid: str = "",
        interpreter_abi_cid: str = "",
        hardware_capability_cid: str = "",
        external_snapshot_cids: Sequence[str] = (),
        policy_cid: str = "",
        canonicalization_schema_cid: str = "",
        tracer_schema_cid: str = "",
        certificate_schema_cid: str = "",
        eligibility_class: EligibilityClass | str = EligibilityClass.REPOSITORY_FOREST_BOUND,
        components: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_extra: Any,
    ) -> ExecutionKeyCompileResult:
        """Compile a strict :class:`TestExecutionKey` identity."""

        if self.cid_support() is not CidSupportStatus.AVAILABLE:
            return ExecutionKeyCompileResult(
                reusable=False,
                reason_code="cid_support_unavailable",
            )
        try:
            reject_pseudo_cid(locator_cid, field_name="locator_cid")
            reject_pseudo_cid(repository_forest_cid, field_name="repository_forest_cid")
            if isinstance(eligibility_class, str):
                eligibility = EligibilityClass(eligibility_class)
            else:
                eligibility = eligibility_class
            execution_key = TestExecutionKey(
                locator_cid=locator_cid,
                repository_forest_cid=repository_forest_cid,
                git_commit_id=str(git_commit_id or ""),
                git_tree_id=str(git_tree_id or ""),
                gitlink_state_cid=str(gitlink_state_cid or ""),
                dirty_overlay_cid=str(dirty_overlay_cid or ""),
                test_module_cid=str(test_module_cid or ""),
                test_class_cid=str(test_class_cid or ""),
                test_function_cid=str(test_function_cid or ""),
                decorator_cids=tuple(str(item) for item in decorator_cids or ()),
                parameter_source_cid=str(parameter_source_cid or ""),
                test_ast_cid=str(test_ast_cid or ""),
                fixture_cids=tuple(str(item) for item in fixture_cids or ()),
                conftest_closure_cid=str(conftest_closure_cid or ""),
                hook_plugin_cids=tuple(str(item) for item in hook_plugin_cids or ()),
                static_trace_root_cid=str(static_trace_root_cid or ""),
                static_unknown_frontier=tuple(
                    str(item) for item in static_unknown_frontier or ()
                ),
                runtime_trace_root_cid=str(runtime_trace_root_cid or ""),
                runtime_completeness_policy=str(runtime_completeness_policy or ""),
                pytest_version=str(pytest_version or "")[:128],
                python_version=str(python_version or "")[:64],
                plugin_versions_cid=str(plugin_versions_cid or ""),
                command_semantics_cid=str(command_semantics_cid or ""),
                config_cid=str(config_cid or ""),
                markers=tuple(str(item) for item in markers or ()),
                dependency_lock_cid=str(dependency_lock_cid or ""),
                installed_distributions_cid=str(installed_distributions_cid or ""),
                environment_cid=str(environment_cid or ""),
                platform_cid=str(platform_cid or ""),
                interpreter_abi_cid=str(interpreter_abi_cid or ""),
                hardware_capability_cid=str(hardware_capability_cid or ""),
                external_snapshot_cids=tuple(
                    str(item) for item in external_snapshot_cids or ()
                ),
                policy_cid=str(policy_cid or ""),
                canonicalization_schema_cid=str(canonicalization_schema_cid or ""),
                tracer_schema_cid=str(tracer_schema_cid or ""),
                certificate_schema_cid=str(certificate_schema_cid or ""),
                eligibility_class=eligibility,
                components=dict(components or {}),
                metadata=dict(metadata or {}),
            )
            identity = mint_content_identity(execution_key.to_dict())
        except Exception as exc:
            return ExecutionKeyCompileResult(
                reusable=False,
                reason_code=f"execution_key_compile_failed:{type(exc).__name__}"[:96],
            )
        return ExecutionKeyCompileResult(
            reusable=True,
            execution_key=execution_key,
            execution_key_cid=identity.cid,
            content_identity=identity,
        )


# Compatibility aliases used by plan docs / older call sites.
compile_test_locator = TestExecutionIdentityCompiler.compile_locator
compile_test_execution_key = TestExecutionIdentityCompiler.compile_execution_key

__all__ = (
    "CONTENT_IDENTITY_INTERFACE",
    "CidSupportStatus",
    "ContentIdentity",
    "ExecutionKeyCompileResult",
    "LocatorCompileResult",
    "TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE",
    "TestExecutionIdentityCompiler",
    "TestExecutionIdentityError",
    "mint_content_identity",
    "mint_content_identity_bytes",
    "normalize_pytest_node_id",
    "reject_pseudo_cid",
)
