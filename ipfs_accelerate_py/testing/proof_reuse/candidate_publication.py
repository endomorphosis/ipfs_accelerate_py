"""Assemble the final cold-pass candidate publication envelope (PTR-146).

After one ordinary pytest setup/call/teardown lifecycle produces a complete
observed runtime trace, this module:

1. compiles a **new** final :class:`TestExecutionKey` that binds that trace;
2. finalizes one admitted :class:`TestPassReceipt` over the final key + trace;
3. retains the candidate descriptor and every required canonical component.

Skipped, xfailed, failed, incomplete, uncontrolled, overflowed, or exceptional
traces publish nothing authoritative.  The envelope itself never authorizes
``SKIP`` — skip authority remains on the certificate path.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from ...agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ...agent_supervisor.proof.test_candidate_context_store import (
    REQUIRED_COMPONENT_KEYS,
)
from ...agent_supervisor.proof.test_execution_contracts import (
    TestExecutionKey,
    TestPassReceipt,
)
from .activation_contracts import CandidateExecutionContext
from .receipt import (
    ReceiptCaptureResult,
    TestPassReceiptCollector,
    evaluate_complete_pass,
    finalize_test_pass_receipt,
)


CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE: Final = "CandidatePublicationEnvelope@1"
CANDIDATE_PUBLICATION_ENVELOPE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/candidate-publication-envelope@1"
)
COMPLETED_EXECUTION_IDENTITY_INTERFACE: Final = "CompletedExecutionIdentity@1"
COMPLETED_EXECUTION_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/completed-execution-identity@1"
)

ITEM_CANDIDATE_PUBLICATION_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_candidate_publication"
)
ITEM_COMPLETED_IDENTITY_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_completed_execution_identity"
)

# Component keys required by TestCandidateContextStore@1.
_REQUIRED = tuple(REQUIRED_COMPONENT_KEYS)


def _bounded_text(value: Any, *, max_chars: int = 256) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    text = value.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _cid_of(value: Any, *, attrs: tuple[str, ...] = ("content_id", "cid")) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _bounded_text(value)
    for name in attrs:
        attr = getattr(value, name, None)
        if isinstance(attr, str) and attr:
            return _bounded_text(attr)
    if isinstance(value, Mapping):
        for name in attrs:
            attr = value.get(name)
            if isinstance(attr, str) and attr:
                return _bounded_text(attr)
    return ""


def _canonical_bytes_of(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    method = getattr(value, "canonical_bytes", None)
    if callable(method):
        try:
            data = method()
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
        except Exception:
            return None
    if isinstance(method, (bytes, bytearray)):
        return bytes(method)
    retained = getattr(value, "retained_canonical_bytes", None)
    if isinstance(retained, (bytes, bytearray)):
        return bytes(retained)
    if isinstance(value, Mapping):
        try:
            return canonical_json_bytes(dict(value))
        except Exception:
            return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            if isinstance(payload, Mapping):
                return canonical_json_bytes(dict(payload))
        except Exception:
            return None
    return None


def _trace_root_cid(runtime_trace: Any) -> str:
    if runtime_trace is None:
        return ""
    for attr in ("trace_cid", "root_cid", "cid", "content_id"):
        value = getattr(runtime_trace, attr, None)
        if isinstance(value, str) and value:
            return _bounded_text(value)
    if isinstance(runtime_trace, Mapping):
        for key in ("trace_cid", "root_cid", "cid", "content_id"):
            value = runtime_trace.get(key)
            if isinstance(value, str) and value:
                return _bounded_text(value)
    return ""


def _trace_is_complete(runtime_trace: Any) -> bool:
    if runtime_trace is None:
        return False
    complete = getattr(runtime_trace, "complete", None)
    if isinstance(complete, bool):
        return complete
    completeness = getattr(runtime_trace, "completeness", None)
    if completeness is not None:
        nested = getattr(completeness, "complete", None)
        if isinstance(nested, bool):
            return nested
        if isinstance(completeness, str):
            return completeness.lower() == "complete"
    if isinstance(runtime_trace, Mapping):
        if isinstance(runtime_trace.get("complete"), bool):
            return bool(runtime_trace["complete"])
        nested_map = runtime_trace.get("completeness")
        if isinstance(nested_map, Mapping) and isinstance(
            nested_map.get("complete"), bool
        ):
            return bool(nested_map["complete"])
    return False


def _trace_reasons(runtime_trace: Any) -> tuple[str, ...]:
    if runtime_trace is None:
        return ()
    reasons = getattr(runtime_trace, "completeness_reasons", None)
    if reasons is None and isinstance(runtime_trace, Mapping):
        reasons = runtime_trace.get("completeness_reasons")
        nested = runtime_trace.get("completeness")
        if reasons is None and isinstance(nested, Mapping):
            reasons = nested.get("reasons")
    if reasons is None:
        completeness = getattr(runtime_trace, "completeness", None)
        reasons = getattr(completeness, "reasons", None) if completeness is not None else None
    if not reasons:
        return ()
    result: list[str] = []
    try:
        for item in reasons:
            text = _bounded_text(item, max_chars=64)
            if text:
                result.append(text)
            if len(result) >= 64:
                break
    except Exception:
        return ()
    return tuple(result)


_NON_AUTHORITATIVE_MARKERS: Final = frozenset(
    {
        "overflow",
        "instrumentation_failure",
        "unsupported_event",
        "private_event",
        "concurrent_trace",
        "uncontrolled",
        "exceptional",
        "exception",
        "incomplete",
    }
)


def _trace_has_authority(runtime_trace: Any) -> bool:
    if not _trace_is_complete(runtime_trace):
        return False
    reasons = {r.lower() for r in _trace_reasons(runtime_trace)}
    if reasons & _NON_AUTHORITATIVE_MARKERS:
        return False
    for reason in reasons:
        if any(marker in reason for marker in _NON_AUTHORITATIVE_MARKERS):
            return False
    root = _trace_root_cid(runtime_trace)
    return bool(root)


def _component_label_bytes(label: str, *, cid_hint: str = "") -> bytes:
    payload: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/testing/proof-reuse/candidate-component@1",
        "label": label,
    }
    if cid_hint:
        payload["cid_hint"] = cid_hint
    return canonical_json_bytes(payload)


@dataclass(frozen=True, slots=True)
class CompletedExecutionIdentity:
    """Final execution identity compiled after a complete cold runtime trace.

    The key is newly compiled: its ``runtime_trace_root_cid`` is the observed
    cold-pass trace CID, not a collection-time placeholder.
    """

    __test__: ClassVar[bool] = False

    execution_key: TestExecutionKey
    execution_key_cid: str
    runtime_trace: Any
    runtime_trace_root_cid: str
    locator_cid: str
    static_trace_root_cid: str = ""
    repository_forest_cid: str = ""
    policy_cid: str = ""
    retained_execution_key_bytes: bytes = b""
    retained_runtime_trace_bytes: bytes = b""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_key, TestExecutionKey):
            raise TypeError("execution_key must be TestExecutionKey")
        if self.execution_key.runtime_trace_root_cid != self.runtime_trace_root_cid:
            raise ValueError("execution key does not bind the observed runtime trace")
        if self.execution_key_cid != self.execution_key.execution_key_id:
            raise ValueError("execution_key_cid does not match compiled key identity")

    @property
    def interface(self) -> str:
        return COMPLETED_EXECUTION_IDENTITY_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def admitted(self) -> bool:
        return bool(self.execution_key_cid and self.runtime_trace_root_cid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPLETED_EXECUTION_IDENTITY_SCHEMA,
            "interface": COMPLETED_EXECUTION_IDENTITY_INTERFACE,
            "execution_key_cid": self.execution_key_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "locator_cid": self.locator_cid,
            "static_trace_root_cid": self.static_trace_root_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "policy_cid": self.policy_cid,
            "admitted": self.admitted,
            "may_authorize_skip": False,
            "diagnostics": dict(self.diagnostics),
        }


def build_completed_execution_identity(
    *,
    locator_cid: str,
    runtime_trace: Any,
    repository_forest_cid: str = "",
    static_trace_root_cid: str = "",
    policy_cid: str = "",
    environment_cid: str = "",
    test_ast_cid: str = "",
    dependency_lock_cid: str = "",
    installed_distributions_cid: str = "",
    platform_cid: str = "",
    capability_root_cid: str = "",
    runtime_completeness_policy: str = "complete-v1",
    seed_execution_key: TestExecutionKey | Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> CompletedExecutionIdentity | None:
    """Compile a final execution key that binds the complete observed trace.

    Returns ``None`` when the trace is incomplete or otherwise non-authoritative.
    Never raises into the pytest outcome path.
    """

    try:
        if not _trace_has_authority(runtime_trace):
            return None
        runtime_cid = _trace_root_cid(runtime_trace)
        if not locator_cid or not runtime_cid:
            return None

        fields: dict[str, Any] = {}
        if isinstance(seed_execution_key, TestExecutionKey):
            fields.update(
                {
                    name: getattr(seed_execution_key, name)
                    for name in (
                        "repository_forest_cid",
                        "git_commit_id",
                        "git_tree_id",
                        "gitlink_state_cid",
                        "dirty_overlay_cid",
                        "test_module_cid",
                        "test_class_cid",
                        "test_function_cid",
                        "decorator_cids",
                        "parameter_source_cid",
                        "test_ast_cid",
                        "fixture_cids",
                        "conftest_closure_cid",
                        "hook_plugin_cids",
                        "static_trace_root_cid",
                        "static_unknown_frontier",
                        "runtime_completeness_policy",
                        "pytest_version",
                        "python_version",
                        "plugin_versions_cid",
                        "command_semantics_cid",
                        "config_cid",
                        "markers",
                        "dependency_lock_cid",
                        "installed_distributions_cid",
                        "environment_cid",
                        "platform_cid",
                        "interpreter_abi_cid",
                        "hardware_capability_cid",
                        "external_snapshot_cids",
                        "policy_cid",
                        "canonicalization_schema_cid",
                        "tracer_schema_cid",
                        "certificate_schema_cid",
                        "eligibility_class",
                        "components",
                        "metadata",
                    )
                }
            )
        elif isinstance(seed_execution_key, Mapping):
            fields.update(dict(seed_execution_key))

        if repository_forest_cid:
            fields["repository_forest_cid"] = repository_forest_cid
        if static_trace_root_cid:
            fields["static_trace_root_cid"] = static_trace_root_cid
        if policy_cid:
            fields["policy_cid"] = policy_cid
        if environment_cid:
            fields["environment_cid"] = environment_cid
        if test_ast_cid:
            fields["test_ast_cid"] = test_ast_cid
        if dependency_lock_cid:
            fields["dependency_lock_cid"] = dependency_lock_cid
        if installed_distributions_cid:
            fields["installed_distributions_cid"] = installed_distributions_cid
        if platform_cid:
            fields["platform_cid"] = platform_cid
        if capability_root_cid:
            fields["hardware_capability_cid"] = capability_root_cid
        if runtime_completeness_policy:
            fields["runtime_completeness_policy"] = runtime_completeness_policy
        if extra_fields:
            fields.update(dict(extra_fields))

        # Final key always rebinds the freshly observed runtime trace CID.
        fields["locator_cid"] = locator_cid
        fields["runtime_trace_root_cid"] = runtime_cid
        if not fields.get("repository_forest_cid"):
            fields["repository_forest_cid"] = repository_forest_cid or "cid:repository-forest"
        # Drop non-constructor keys if a raw mapping was supplied.
        allowed = {
            "locator_cid",
            "repository_forest_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "dirty_overlay_cid",
            "test_module_cid",
            "test_class_cid",
            "test_function_cid",
            "decorator_cids",
            "parameter_source_cid",
            "test_ast_cid",
            "fixture_cids",
            "conftest_closure_cid",
            "hook_plugin_cids",
            "static_trace_root_cid",
            "static_unknown_frontier",
            "runtime_trace_root_cid",
            "runtime_completeness_policy",
            "pytest_version",
            "python_version",
            "plugin_versions_cid",
            "command_semantics_cid",
            "config_cid",
            "markers",
            "dependency_lock_cid",
            "installed_distributions_cid",
            "environment_cid",
            "platform_cid",
            "interpreter_abi_cid",
            "hardware_capability_cid",
            "external_snapshot_cids",
            "policy_cid",
            "canonicalization_schema_cid",
            "tracer_schema_cid",
            "certificate_schema_cid",
            "eligibility_class",
            "components",
            "metadata",
        }
        constructor_fields = {k: v for k, v in fields.items() if k in allowed}
        key = TestExecutionKey(**constructor_fields)
        key_bytes = key.canonical_bytes()
        trace_bytes = _canonical_bytes_of(runtime_trace) or b""
        return CompletedExecutionIdentity(
            execution_key=key,
            execution_key_cid=key.execution_key_id,
            runtime_trace=runtime_trace,
            runtime_trace_root_cid=runtime_cid,
            locator_cid=locator_cid,
            static_trace_root_cid=str(key.static_trace_root_cid or ""),
            repository_forest_cid=str(key.repository_forest_cid or ""),
            policy_cid=str(key.policy_cid or ""),
            retained_execution_key_bytes=key_bytes,
            retained_runtime_trace_bytes=trace_bytes,
            diagnostics={
                "compiled_after_trace": True,
                "runtime_trace_complete": True,
            },
        )
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class CandidatePublicationEnvelope:
    """Canonical cold-pass candidate descriptor + retained component bytes.

    Publication is non-authoritative for skip: ``may_authorize_skip`` is always
    false.  ``authoritative`` is true only when every required component is
    present and the receipt/identity/trace trio is complete.
    """

    __test__: ClassVar[bool] = False

    descriptor: CandidateExecutionContext
    component_bytes: Mapping[str, bytes]
    component_cids: Mapping[str, str]
    execution_key: TestExecutionKey
    receipt: TestPassReceipt
    runtime_trace: Any
    retained_descriptor_bytes: bytes
    authoritative: bool
    reason_code: str = "ok"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def candidate_context_cid(self) -> str:
        return self.descriptor.candidate_context_id

    @property
    def execution_key_cid(self) -> str:
        return self.execution_key.execution_key_id

    @property
    def receipt_cid(self) -> str:
        return self.receipt.receipt_id

    @property
    def runtime_trace_root_cid(self) -> str:
        return self.descriptor.runtime_trace_root_cid

    @property
    def locator_cid(self) -> str:
        return self.descriptor.locator_cid

    def required_components_present(self) -> bool:
        return all(name in self.component_bytes for name in _REQUIRED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_PUBLICATION_ENVELOPE_SCHEMA,
            "interface": CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE,
            "authoritative": self.authoritative,
            "reason_code": self.reason_code,
            "may_authorize_skip": False,
            "candidate_context_cid": self.candidate_context_cid,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "pass_receipt_cid": self.receipt_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "component_cids": dict(self.component_cids),
            "required_components": list(_REQUIRED),
            "required_components_present": self.required_components_present(),
            "diagnostics": dict(self.diagnostics),
        }


def _empty_publication(
    *,
    reason_code: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    """Publication failures yield ``None`` — nothing authoritative."""

    del reason_code, diagnostics
    return None


def assemble_candidate_publication(
    *,
    completed_identity: CompletedExecutionIdentity,
    receipt: TestPassReceipt,
    component_bytes: Mapping[str, bytes] | None = None,
    environment_cid: str = "",
    test_ast_cid: str = "",
    retained_at_ms: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CandidatePublicationEnvelope | None:
    """Build the candidate publication envelope for one admitted cold pass.

    Requires a completed identity (final key + complete trace) and an admitted
    receipt that binds that key and trace.  Missing required components or any
    identity mismatch returns ``None``.
    """

    try:
        if completed_identity is None or receipt is None:
            return _empty_publication(reason_code="missing_inputs")
        if not completed_identity.admitted or not receipt.admitted:
            return _empty_publication(reason_code="not_admitted")
        if receipt.execution_key_cid != completed_identity.execution_key_cid:
            return _empty_publication(reason_code="receipt_key_mismatch")
        if receipt.runtime_trace_root_cid != completed_identity.runtime_trace_root_cid:
            return _empty_publication(reason_code="receipt_trace_mismatch")
        if receipt.locator_cid != completed_identity.locator_cid:
            return _empty_publication(reason_code="receipt_locator_mismatch")

        key = completed_identity.execution_key
        runtime_trace = completed_identity.runtime_trace
        components: dict[str, bytes] = dict(component_bytes or {})

        # Always retain the freshly compiled key and observed runtime trace.
        components["execution_key"] = (
            completed_identity.retained_execution_key_bytes or key.canonical_bytes()
        )
        components["runtime_trace"] = (
            completed_identity.retained_runtime_trace_bytes
            or _canonical_bytes_of(runtime_trace)
            or b""
        )
        components["pass_receipt"] = receipt.canonical_bytes()

        # Fill remaining required components from the key / supplied bytes.
        def _ensure(name: str, cid: str, payload_label: str) -> None:
            if name in components and components[name]:
                return
            if cid:
                components[name] = _component_label_bytes(payload_label, cid_hint=cid)
            else:
                components[name] = _component_label_bytes(payload_label)

        forest_cid = key.repository_forest_cid or completed_identity.repository_forest_cid
        static_cid = key.static_trace_root_cid or completed_identity.static_trace_root_cid
        policy = key.policy_cid or completed_identity.policy_cid
        env_cid = environment_cid or key.environment_cid
        ast_cid = test_ast_cid or key.test_ast_cid

        _ensure("repository_forest", forest_cid, "repository_forest")
        _ensure("static_trace", static_cid, "static_trace")
        _ensure("policy", policy, "policy")
        _ensure("environment", env_cid, "environment")

        missing = [name for name in _REQUIRED if not components.get(name)]
        if missing:
            return _empty_publication(
                reason_code="component_missing",
                diagnostics={"missing": missing},
            )

        import hashlib
        import json

        resolved_cids: dict[str, str] = {}
        for name, data in components.items():
            try:
                # Prefer dag-json rehash when bytes are canonical JSON objects.
                resolved_cids[name] = content_identity(
                    json.loads(data.decode("utf-8"))
                )
            except Exception:
                # Fall back to a content identity over a labeled wrapper so
                # non-JSON instrumented payloads remain addressable.
                resolved_cids[name] = content_identity(
                    {
                        "schema": (
                            "ipfs_accelerate_py/testing/proof-reuse/"
                            "raw-component-bytes@1"
                        ),
                        "component": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "byte_length": len(data),
                    }
                )

        # For typed contract components, prefer the object's own content id.
        resolved_cids["execution_key"] = key.execution_key_id
        resolved_cids["pass_receipt"] = receipt.receipt_id
        runtime_cid = completed_identity.runtime_trace_root_cid
        if runtime_cid:
            resolved_cids["runtime_trace"] = runtime_cid
        if forest_cid:
            resolved_cids["repository_forest"] = forest_cid
        if static_cid:
            resolved_cids["static_trace"] = static_cid
        if policy:
            resolved_cids["policy"] = policy
        if env_cid:
            resolved_cids["environment"] = env_cid

        now_ms = int(retained_at_ms if retained_at_ms is not None else time.time() * 1000)
        descriptor = CandidateExecutionContext(
            locator_cid=completed_identity.locator_cid,
            execution_key_cid=key.execution_key_id,
            pass_receipt_cid=receipt.receipt_id,
            repository_forest_cid=forest_cid or resolved_cids.get("repository_forest", ""),
            test_ast_cid=ast_cid or resolved_cids.get("test_ast", content_identity({"label": "test_ast"})),
            static_trace_root_cid=static_cid or resolved_cids["static_trace"],
            runtime_trace_root_cid=runtime_cid,
            environment_cid=env_cid or resolved_cids["environment"],
            policy_cid=policy or resolved_cids["policy"],
            dependency_lock_cid=key.dependency_lock_cid,
            installed_distributions_cid=key.installed_distributions_cid,
            platform_cid=key.platform_cid,
            capability_root_cid=key.hardware_capability_cid,
            component_cids=dict(resolved_cids),
            external_snapshot_cids=tuple(key.external_snapshot_cids or ()),
            retained_at_ms=max(0, now_ms),
            metadata={
                **dict(metadata or {}),
                "publication_interface": CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE,
                "completed_identity_interface": COMPLETED_EXECUTION_IDENTITY_INTERFACE,
                "may_authorize_skip": False,
            },
        )
        descriptor_bytes = descriptor.canonical_bytes()
        return CandidatePublicationEnvelope(
            descriptor=descriptor,
            component_bytes=dict(components),
            component_cids=dict(resolved_cids),
            execution_key=key,
            receipt=receipt,
            runtime_trace=runtime_trace,
            retained_descriptor_bytes=descriptor_bytes,
            authoritative=True,
            reason_code="ok",
            diagnostics={
                "required_components": list(_REQUIRED),
                "component_count": len(components),
            },
        )
    except Exception:
        return None


def finalize_cold_pass_publication(
    *,
    collector: TestPassReceiptCollector | Mapping[str, Any] | None,
    runtime_trace: Any,
    locator: Any = None,
    locator_cid: str = "",
    seed_execution_key: TestExecutionKey | Mapping[str, Any] | None = None,
    component_bytes: Mapping[str, bytes] | None = None,
    repository_forest_cid: str = "",
    static_trace_root_cid: str = "",
    policy_cid: str = "",
    environment_cid: str = "",
    test_ast_cid: str = "",
    require_runtime_trace: bool = True,
    item: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[
    ReceiptCaptureResult,
    CompletedExecutionIdentity | None,
    CandidatePublicationEnvelope | None,
]:
    """End-to-end cold-pass finalization after teardown.

    Steps:

    1. Evaluate complete-pass eligibility (setup/call/teardown each PASS).
    2. Compile the final execution key only when the observed trace is complete.
    3. Finalize an admitted receipt bound to that key and trace.
    4. Assemble the candidate publication envelope with every required component.

    Non-passing phases, incomplete traces, and tracing faults yield a
    non-admitted receipt result and ``None`` publication.  Never raises.
    """

    try:
        resolved_locator = locator_cid or _cid_of(
            locator, attrs=("locator_id", "content_id", "cid")
        )

        # Early reject: incomplete / uncontrolled / overflowed / exceptional
        # traces publish nothing authoritative.
        if require_runtime_trace and not _trace_has_authority(runtime_trace):
            phases = None
            if isinstance(collector, TestPassReceiptCollector):
                eligible, disqualifiers = collector.evaluate(
                    runtime_trace=runtime_trace,
                    require_runtime_trace=True,
                )
                phase_outcomes = collector.phase_outcomes()
            elif isinstance(collector, Mapping):
                eligible, disqualifiers = evaluate_complete_pass(
                    collector,
                    runtime_trace=runtime_trace,
                    require_runtime_trace=True,
                )
                phase_outcomes = {
                    k: (v.value if hasattr(v, "value") else str(v))
                    for k, v in collector.items()
                }
            else:
                eligible, disqualifiers = False, ("incomplete_trace",)
                phase_outcomes = {}
            del eligible
            result = ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                disqualifying_states=tuple(disqualifiers) or ("incomplete_trace",),
                phase_outcomes=phase_outcomes,
                store_reason="not_eligible",
                diagnostics={"stage": "cold_pass", "reason": "trace_not_authoritative"},
            )
            if item is not None:
                try:
                    from .receipt import ITEM_RECEIPT_RESULT_ATTRIBUTE

                    setattr(item, ITEM_RECEIPT_RESULT_ATTRIBUTE, result)
                except Exception:
                    pass
            return result, None, None

        completed = build_completed_execution_identity(
            locator_cid=resolved_locator,
            runtime_trace=runtime_trace,
            repository_forest_cid=repository_forest_cid,
            static_trace_root_cid=static_trace_root_cid,
            policy_cid=policy_cid,
            environment_cid=environment_cid,
            test_ast_cid=test_ast_cid,
            seed_execution_key=seed_execution_key,
        )
        if completed is None:
            result = finalize_test_pass_receipt(
                collector,
                locator=locator,
                locator_cid=resolved_locator,
                runtime_trace=runtime_trace,
                writes_receipts=False,
                require_runtime_trace=require_runtime_trace,
                item=item,
            )
            return result, None, None

        receipt_result = finalize_test_pass_receipt(
            collector,
            locator=locator,
            locator_cid=completed.locator_cid,
            execution_key=completed.execution_key,
            execution_key_cid=completed.execution_key_cid,
            runtime_trace=runtime_trace,
            runtime_trace_root_cid=completed.runtime_trace_root_cid,
            static_trace_root_cid=completed.static_trace_root_cid,
            dependency_forest_cid=completed.repository_forest_cid,
            policy_cid=completed.policy_cid,
            writes_receipts=False,
            require_runtime_trace=True,
            item=item,
            metadata=metadata,
        )
        if not receipt_result.admitted or receipt_result.receipt is None:
            return receipt_result, completed, None

        # Bind final key onto the item when present.
        if item is not None:
            try:
                setattr(item, ITEM_COMPLETED_IDENTITY_ATTRIBUTE, completed)
                setattr(
                    item,
                    "_ipfs_proof_reuse_execution_key",
                    completed.execution_key,
                )
            except Exception:
                pass

        envelope = assemble_candidate_publication(
            completed_identity=completed,
            receipt=receipt_result.receipt,
            component_bytes=component_bytes,
            environment_cid=environment_cid,
            test_ast_cid=test_ast_cid,
            metadata=metadata,
        )
        if envelope is not None and item is not None:
            try:
                setattr(item, ITEM_CANDIDATE_PUBLICATION_ATTRIBUTE, envelope)
            except Exception:
                pass
        return receipt_result, completed, envelope
    except Exception as exc:
        return (
            ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                store_reason="finalize_error",
                diagnostics={
                    "stage": "cold_pass",
                    "error_type": type(exc).__name__,
                },
            ),
            None,
            None,
        )


def publication_is_authoritative(value: Any) -> bool:
    """Return whether a publication object retains skip-adjacent authority bits.

    Candidate envelopes are never skip authority themselves; this reports only
    whether the cold-pass unit was complete enough to retain for later issuance.
    """

    if value is None:
        return False
    if isinstance(value, CandidatePublicationEnvelope):
        return value.authoritative and value.required_components_present()
    authoritative = getattr(value, "authoritative", None)
    if isinstance(authoritative, bool):
        return authoritative
    if isinstance(value, Mapping):
        return bool(value.get("authoritative"))
    return False


__all__ = [
    "CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE",
    "CANDIDATE_PUBLICATION_ENVELOPE_SCHEMA",
    "COMPLETED_EXECUTION_IDENTITY_INTERFACE",
    "COMPLETED_EXECUTION_IDENTITY_SCHEMA",
    "CandidatePublicationEnvelope",
    "CompletedExecutionIdentity",
    "ITEM_CANDIDATE_PUBLICATION_ATTRIBUTE",
    "ITEM_COMPLETED_IDENTITY_ATTRIBUTE",
    "assemble_candidate_publication",
    "build_completed_execution_identity",
    "finalize_cold_pass_publication",
    "publication_is_authoritative",
]
