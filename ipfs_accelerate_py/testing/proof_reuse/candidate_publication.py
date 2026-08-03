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
CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_INTERFACE: Final = (
    "ControllerOwnedV2VerificationContext@1"
)
CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/"
    "controller-owned-v2-verification-context@1"
)

ITEM_CANDIDATE_PUBLICATION_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_candidate_publication"
)
ITEM_COMPLETED_IDENTITY_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_completed_execution_identity"
)
ITEM_CONTROLLER_V2_CONTEXT_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_controller_v2_context"
)

# Component keys required by TestCandidateContextStore@1.
_REQUIRED = tuple(REQUIRED_COMPONENT_KEYS)

# Exact V2 expected-public-input pins the controller must reconstruct.
# Certificate metadata must never fill a missing pin.
REQUIRED_CONTROLLER_V2_PIN_FIELDS: Final = (
    "receipt_cid",
    "execution_key_cid",
    "candidate_context_cid",
    "policy_cid",
    "statement_cid",
    "circuit_cid",
    "verifying_key_cid",
    "issuer_id",
    "epoch",
    "backend_id",
)
OPTIONAL_CONTROLLER_V2_PIN_FIELDS: Final = (
    "proof_system_id",
    "locator_cid",
    "content_profile",
    "statement_digest",
    "statement_version",
    "statement_interface",
)
# Per-field and aggregate bounds for controller-owned public handoff bytes.
MAX_CONTROLLER_V2_PIN_CHARS: Final = 256
MAX_CONTROLLER_V2_RETAINED_BYTES: Final = 2 * 1024 * 1024
MAX_CONTROLLER_V2_CONTEXT_BYTES: Final = 4 * 1024 * 1024
MAX_CONTROLLER_V2_COMPONENT_BYTES: Final = 2 * 1024 * 1024


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

    def retained_component(self, name: str) -> bytes | None:
        """Return exact retained component bytes when present and non-empty."""

        data = self.component_bytes.get(name)
        if isinstance(data, (bytes, bytearray)) and data:
            return bytes(data)
        return None

    def controller_owned_v2_pins(
        self,
        *,
        statement_cid: str = "",
        circuit_cid: str = "",
        verifying_key_cid: str = "",
        issuer_id: str = "",
        epoch: str = "",
        backend_id: str = "",
        proof_system_id: str = "",
        statement_digest: str = "",
        content_profile: str = "",
    ) -> "ControllerOwnedV2VerificationContext":
        """Project retained cold-pass components into controller-owned V2 pins.

        Statement/circuit/key/issuer/epoch/backend pins are supplied by the
        controller issuance path, never by an attached certificate.  Missing
        optional pins leave the context incomplete (receipt-only / DEFERRED).
        """

        policy_cid = _bounded_text(
            self.descriptor.policy_cid or self.component_cids.get("policy", ""),
            max_chars=MAX_CONTROLLER_V2_PIN_CHARS,
        )
        retained_receipt = self.retained_component("pass_receipt") or b""
        retained_key = self.retained_component("execution_key") or b""
        retained_descriptor = (
            self.retained_descriptor_bytes
            if isinstance(self.retained_descriptor_bytes, (bytes, bytearray))
            else b""
        )
        return ControllerOwnedV2VerificationContext(
            receipt_cid=_bounded_text(
                self.receipt_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            execution_key_cid=_bounded_text(
                self.execution_key_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            candidate_context_cid=_bounded_text(
                self.candidate_context_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            policy_cid=policy_cid,
            statement_cid=_bounded_text(
                statement_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            circuit_cid=_bounded_text(
                circuit_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            verifying_key_cid=_bounded_text(
                verifying_key_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            issuer_id=_bounded_text(issuer_id, max_chars=MAX_CONTROLLER_V2_PIN_CHARS),
            epoch=_bounded_text(epoch, max_chars=MAX_CONTROLLER_V2_PIN_CHARS),
            backend_id=_bounded_text(
                backend_id, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            proof_system_id=_bounded_text(
                proof_system_id, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            locator_cid=_bounded_text(
                self.locator_cid, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            statement_digest=_bounded_text(
                statement_digest, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            content_profile=_bounded_text(
                content_profile, max_chars=MAX_CONTROLLER_V2_PIN_CHARS
            ),
            retained_receipt_bytes=bytes(retained_receipt),
            retained_candidate_context_bytes=bytes(retained_descriptor),
            retained_execution_key_bytes=bytes(retained_key),
            component_cids=dict(self.component_cids),
            source="candidate_publication_envelope",
        )

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


def _pin_text(value: Any, *, max_chars: int = MAX_CONTROLLER_V2_PIN_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    text = value.strip()
    if not text or len(text) > max_chars:
        return ""
    if any(ord(character) < 32 for character in text):
        return ""
    return text


def _bounded_retained_bytes(
    value: Any,
    *,
    max_bytes: int = MAX_CONTROLLER_V2_RETAINED_BYTES,
) -> bytes | None:
    """Return exact retained public bytes, or ``None`` when oversized/malformed.

    Oversized payloads are never truncated: callers treat ``None`` as a hard
    rejection so xdist transport cannot silently drop required retained material.
    """

    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return b""
        if len(text) % 2 or len(text) > max_bytes * 2:
            return None
        try:
            data = bytes.fromhex(text)
        except ValueError:
            return None
    else:
        return None
    if len(data) > max_bytes:
        return None
    return data


def rehash_controller_owned_public_bytes(
    data: bytes | bytearray,
    *,
    max_bytes: int = MAX_CONTROLLER_V2_RETAINED_BYTES,
) -> str:
    """Rehash retained public bytes before controller use.

    Prefer exact DAG-JSON re-canonicalization when the payload is a JSON
    object/array; otherwise fall back to a labeled content identity over the
    raw digest so non-JSON instrumented blobs remain addressable.
    """

    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("retained public bytes must be bytes")
    payload = bytes(data)
    if not payload:
        raise ValueError("retained public bytes must be nonempty")
    if len(payload) > max_bytes:
        raise ValueError("retained public bytes exceed bound")
    try:
        from .activation_contracts import rehash_retained_canonical_bytes

        return rehash_retained_canonical_bytes(payload)
    except Exception:
        import hashlib

        return content_identity(
            {
                "schema": (
                    "ipfs_accelerate_py/testing/proof-reuse/"
                    "raw-controller-context-bytes@1"
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
            }
        )


@dataclass(frozen=True, slots=True)
class ControllerOwnedV2VerificationContext:
    """Bounded controller-owned expected V2 verification pins (PTR-154).

    Workers and serial nodes may propose only public pin strings and retained
    public bytes.  The controller reconstructs this object, rehashes retained
    bytes, and refuses certificate fields as fillers for missing pins.  The
    context never grants publication or skip authority by itself.
    """

    __test__: ClassVar[bool] = False

    receipt_cid: str = ""
    execution_key_cid: str = ""
    candidate_context_cid: str = ""
    policy_cid: str = ""
    statement_cid: str = ""
    circuit_cid: str = ""
    verifying_key_cid: str = ""
    issuer_id: str = ""
    epoch: str = ""
    backend_id: str = ""
    proof_system_id: str = ""
    locator_cid: str = ""
    statement_digest: str = ""
    content_profile: str = ""
    retained_receipt_bytes: bytes = b""
    retained_candidate_context_bytes: bytes = b""
    retained_execution_key_bytes: bytes = b""
    component_cids: Mapping[str, str] = field(default_factory=dict)
    receipt_bytes_cid: str = ""
    candidate_context_bytes_cid: str = ""
    execution_key_bytes_cid: str = ""
    source: str = ""
    reason_code: str = "ok"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def may_publish_candidate(self) -> bool:
        return False

    def pin_value(self, name: str) -> str:
        return _pin_text(getattr(self, name, ""))

    def missing_required_pins(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name in REQUIRED_CONTROLLER_V2_PIN_FIELDS:
            if not self.pin_value(name):
                missing.append(name)
        return tuple(missing)

    @property
    def is_complete(self) -> bool:
        return not self.missing_required_pins()

    def retained_receipt_bytes_hex(self) -> str:
        return self.retained_receipt_bytes.hex() if self.retained_receipt_bytes else ""

    def retained_candidate_context_bytes_hex(self) -> str:
        return (
            self.retained_candidate_context_bytes.hex()
            if self.retained_candidate_context_bytes
            else ""
        )

    def retained_execution_key_bytes_hex(self) -> str:
        return (
            self.retained_execution_key_bytes.hex()
            if self.retained_execution_key_bytes
            else ""
        )

    def aggregate_byte_length(self) -> int:
        return (
            len(self.retained_receipt_bytes)
            + len(self.retained_candidate_context_bytes)
            + len(self.retained_execution_key_bytes)
        )

    def rehash_retained_bytes(self) -> "ControllerOwnedV2VerificationContext":
        """Return a copy with CID rehash of every retained public blob."""

        receipt_cid = self.receipt_bytes_cid
        candidate_cid = self.candidate_context_bytes_cid
        key_cid = self.execution_key_bytes_cid
        diagnostics = dict(self.diagnostics)
        if self.retained_receipt_bytes:
            receipt_cid = rehash_controller_owned_public_bytes(
                self.retained_receipt_bytes
            )
        if self.retained_candidate_context_bytes:
            candidate_cid = rehash_controller_owned_public_bytes(
                self.retained_candidate_context_bytes
            )
        if self.retained_execution_key_bytes:
            key_cid = rehash_controller_owned_public_bytes(
                self.retained_execution_key_bytes
            )
        if self.candidate_context_cid and candidate_cid:
            # When both a declared pin and retained bytes exist, require match
            # only when the declared pin equals a content identity of those
            # bytes or the retained-bytes CID is recorded separately.
            diagnostics["candidate_context_bytes_rehashed"] = True
        return ControllerOwnedV2VerificationContext(
            receipt_cid=self.receipt_cid,
            execution_key_cid=self.execution_key_cid,
            candidate_context_cid=self.candidate_context_cid,
            policy_cid=self.policy_cid,
            statement_cid=self.statement_cid,
            circuit_cid=self.circuit_cid,
            verifying_key_cid=self.verifying_key_cid,
            issuer_id=self.issuer_id,
            epoch=self.epoch,
            backend_id=self.backend_id,
            proof_system_id=self.proof_system_id,
            locator_cid=self.locator_cid,
            statement_digest=self.statement_digest,
            content_profile=self.content_profile,
            retained_receipt_bytes=self.retained_receipt_bytes,
            retained_candidate_context_bytes=self.retained_candidate_context_bytes,
            retained_execution_key_bytes=self.retained_execution_key_bytes,
            component_cids=dict(self.component_cids),
            receipt_bytes_cid=receipt_cid,
            candidate_context_bytes_cid=candidate_cid,
            execution_key_bytes_cid=key_cid,
            source=self.source,
            reason_code=self.reason_code,
            diagnostics=diagnostics,
        )

    def to_public_mapping(self) -> dict[str, Any]:
        """Public-only transport projection (no nested witness bodies)."""

        payload: dict[str, Any] = {
            "interface": self.interface,
            "schema": CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_SCHEMA,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "policy_cid": self.policy_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "backend_id": self.backend_id,
            "proof_system_id": self.proof_system_id,
            "locator_cid": self.locator_cid,
            "statement_digest": self.statement_digest,
            "content_profile": self.content_profile,
            "may_authorize_skip": False,
            "may_publish_candidate": False,
            "is_complete": self.is_complete,
            "reason_code": self.reason_code,
            "source": self.source,
        }
        if self.retained_receipt_bytes:
            payload["retained_receipt_bytes_hex"] = self.retained_receipt_bytes_hex()
        if self.retained_candidate_context_bytes:
            payload["retained_candidate_context_bytes_hex"] = (
                self.retained_candidate_context_bytes_hex()
            )
        if self.retained_execution_key_bytes:
            payload["retained_execution_key_bytes_hex"] = (
                self.retained_execution_key_bytes_hex()
            )
        if self.receipt_bytes_cid:
            payload["receipt_bytes_cid"] = self.receipt_bytes_cid
        if self.candidate_context_bytes_cid:
            payload["candidate_context_bytes_cid"] = self.candidate_context_bytes_cid
        if self.execution_key_bytes_cid:
            payload["execution_key_bytes_cid"] = self.execution_key_bytes_cid
        if self.component_cids:
            # Flatten only scalar CID strings for transport safety.
            for name, cid in self.component_cids.items():
                text = _pin_text(cid)
                if text:
                    payload[f"component_cid:{name}"] = text
        return {key: value for key, value in payload.items() if value not in ("", None)}

    def to_dict(self) -> dict[str, Any]:
        return self.to_public_mapping()

    def to_deferred_public_mapping(self) -> dict[str, Any]:
        """Project into the deferred-issuance public envelope shape."""

        payload = {
            "interface": "DeferredIssuanceEnvelope@1",
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "policy_cid": self.policy_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "backend_id": self.backend_id,
            "proof_system_id": self.proof_system_id,
            "locator_cid": self.locator_cid,
            "statement_digest": self.statement_digest,
            "content_profile": self.content_profile,
            "retained_receipt_bytes_hex": self.retained_receipt_bytes_hex(),
            "retained_candidate_context_bytes_hex": (
                self.retained_candidate_context_bytes_hex()
            ),
        }
        return {key: value for key, value in payload.items() if value not in ("", None)}

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        certificate: Mapping[str, Any] | None = None,
        rehash: bool = True,
    ) -> "ControllerOwnedV2VerificationContext | None":
        """Parse public pins; certificate is never used to fill missing values."""

        del certificate  # Explicitly unused: certificates cannot fill pins.
        if value is None:
            return None
        if isinstance(value, cls):
            context = value
        else:
            if hasattr(value, "to_dict") and callable(value.to_dict):
                try:
                    value = value.to_dict()
                except Exception:
                    return None
            if not isinstance(value, Mapping):
                return None
            retained_receipt = _bounded_retained_bytes(
                value.get("retained_receipt_bytes_hex")
                or value.get("retained_receipt_bytes")
            )
            retained_candidate = _bounded_retained_bytes(
                value.get("retained_candidate_context_bytes_hex")
                or value.get("retained_candidate_context_bytes")
            )
            retained_key = _bounded_retained_bytes(
                value.get("retained_execution_key_bytes_hex")
                or value.get("retained_execution_key_bytes")
            )
            if (
                retained_receipt is None
                or retained_candidate is None
                or retained_key is None
            ):
                return None
            component_cids: dict[str, str] = {}
            raw_components = value.get("component_cids")
            if isinstance(raw_components, Mapping):
                for name, cid in raw_components.items():
                    text = _pin_text(cid)
                    if text:
                        component_cids[str(name)[:64]] = text
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if key.startswith("component_cid:"):
                    text = _pin_text(raw_value)
                    if text:
                        component_cids[key[len("component_cid:") :][:64]] = text
            context = cls(
                receipt_cid=_pin_text(value.get("receipt_cid")),
                execution_key_cid=_pin_text(value.get("execution_key_cid")),
                candidate_context_cid=_pin_text(value.get("candidate_context_cid")),
                policy_cid=_pin_text(value.get("policy_cid")),
                statement_cid=_pin_text(value.get("statement_cid")),
                circuit_cid=_pin_text(value.get("circuit_cid")),
                verifying_key_cid=_pin_text(value.get("verifying_key_cid")),
                issuer_id=_pin_text(value.get("issuer_id")),
                epoch=_pin_text(value.get("epoch")),
                backend_id=_pin_text(value.get("backend_id")),
                proof_system_id=_pin_text(value.get("proof_system_id")),
                locator_cid=_pin_text(value.get("locator_cid")),
                statement_digest=_pin_text(value.get("statement_digest")),
                content_profile=_pin_text(value.get("content_profile")),
                retained_receipt_bytes=retained_receipt,
                retained_candidate_context_bytes=retained_candidate,
                retained_execution_key_bytes=retained_key,
                component_cids=component_cids,
                receipt_bytes_cid=_pin_text(value.get("receipt_bytes_cid")),
                candidate_context_bytes_cid=_pin_text(
                    value.get("candidate_context_bytes_cid")
                ),
                execution_key_bytes_cid=_pin_text(
                    value.get("execution_key_bytes_cid")
                ),
                source=_pin_text(value.get("source") or "public_mapping"),
                reason_code=_pin_text(value.get("reason_code") or "ok") or "ok",
            )
        if context.aggregate_byte_length() > MAX_CONTROLLER_V2_CONTEXT_BYTES:
            return None
        if not rehash:
            return context
        try:
            return context.rehash_retained_bytes()
        except Exception:
            return None

    @classmethod
    def from_deferred_envelope(
        cls,
        envelope: Any,
        *,
        retained_receipt_bytes: bytes | bytearray | None = None,
        retained_candidate_context_bytes: bytes | bytearray | None = None,
        retained_execution_key_bytes: bytes | bytearray | None = None,
        certificate: Mapping[str, Any] | None = None,
        rehash: bool = True,
    ) -> "ControllerOwnedV2VerificationContext | None":
        """Build from a public deferred envelope without certificate fill-in."""

        del certificate
        try:
            if hasattr(envelope, "to_dict") and callable(envelope.to_dict):
                payload = dict(envelope.to_dict())
            elif isinstance(envelope, Mapping):
                payload = dict(envelope)
            else:
                return None
            if retained_receipt_bytes is not None:
                payload["retained_receipt_bytes_hex"] = bytes(
                    retained_receipt_bytes
                ).hex()
            if retained_candidate_context_bytes is not None:
                payload["retained_candidate_context_bytes_hex"] = bytes(
                    retained_candidate_context_bytes
                ).hex()
            if retained_execution_key_bytes is not None:
                payload["retained_execution_key_bytes_hex"] = bytes(
                    retained_execution_key_bytes
                ).hex()
            payload.setdefault("source", "deferred_envelope")
            return cls.from_mapping(payload, rehash=rehash)
        except Exception:
            return None

    @classmethod
    def from_candidate_publication(
        cls,
        envelope: "CandidatePublicationEnvelope",
        **pins: Any,
    ) -> "ControllerOwnedV2VerificationContext | None":
        if not isinstance(envelope, CandidatePublicationEnvelope):
            return None
        try:
            context = envelope.controller_owned_v2_pins(
                statement_cid=str(pins.get("statement_cid") or ""),
                circuit_cid=str(pins.get("circuit_cid") or ""),
                verifying_key_cid=str(pins.get("verifying_key_cid") or ""),
                issuer_id=str(pins.get("issuer_id") or ""),
                epoch=str(pins.get("epoch") or ""),
                backend_id=str(pins.get("backend_id") or ""),
                proof_system_id=str(pins.get("proof_system_id") or ""),
                statement_digest=str(pins.get("statement_digest") or ""),
                content_profile=str(pins.get("content_profile") or ""),
            )
            return admit_controller_owned_v2_context(context)[0]
        except Exception:
            return None


def admit_controller_owned_v2_context(
    context: ControllerOwnedV2VerificationContext | Mapping[str, Any] | None,
    *,
    require_complete: bool = False,
    require_retained_bytes: bool = False,
    certificate: Mapping[str, Any] | None = None,
    expected_pins: Mapping[str, str] | None = None,
) -> tuple[ControllerOwnedV2VerificationContext | None, str]:
    """Admit a controller-owned context after size/rehash/pin checks.

    * ``certificate`` is accepted only to prove it is ignored for fill-in.
    * Stale/substituted pins relative to *expected_pins* are rejected.
    * Incomplete contexts are admitted only when ``require_complete`` is false
      (receipt-only / DEFERRED paths still need the partial public envelope).
    """

    del certificate
    try:
        if isinstance(context, ControllerOwnedV2VerificationContext):
            admitted = ControllerOwnedV2VerificationContext.from_mapping(
                context.to_public_mapping(),
                rehash=True,
            )
        else:
            admitted = ControllerOwnedV2VerificationContext.from_mapping(
                context,
                rehash=True,
            )
        if admitted is None:
            return None, "controller_context_invalid"
        if admitted.aggregate_byte_length() > MAX_CONTROLLER_V2_CONTEXT_BYTES:
            return None, "controller_context_oversized"
        if require_retained_bytes and not (
            admitted.retained_receipt_bytes or admitted.retained_candidate_context_bytes
        ):
            return None, "controller_context_retained_bytes_missing"
        if require_complete and not admitted.is_complete:
            return None, "controller_context_incomplete:" + ",".join(
                admitted.missing_required_pins()
            )
        if expected_pins:
            for name, expected in expected_pins.items():
                actual = admitted.pin_value(str(name))
                expected_text = _pin_text(expected)
                if expected_text and actual and actual != expected_text:
                    return None, f"controller_context_pin_mismatch:{name}"
                if expected_text and not actual:
                    return None, f"controller_context_pin_missing:{name}"
        return admitted, ""
    except Exception:
        return None, "controller_context_exception"


def reconstruct_controller_owned_v2_context(
    source: Any,
    *,
    retained_receipt_bytes: bytes | bytearray | None = None,
    retained_candidate_context_bytes: bytes | bytearray | None = None,
    retained_execution_key_bytes: bytes | bytearray | None = None,
    certificate: Mapping[str, Any] | None = None,
    require_complete: bool = False,
) -> tuple[ControllerOwnedV2VerificationContext | None, str]:
    """Controller-side reconstruction used by serial and xdist paths alike.

    Certificate fields never fill missing expected pins.  Missing, malformed,
    oversized, stale or substituted context returns a typed miss reason so the
    caller retains receipt-only RUN/DEFERRED behavior.
    """

    try:
        if isinstance(source, CandidatePublicationEnvelope):
            base = source.controller_owned_v2_pins()
            payload = base.to_public_mapping()
        elif isinstance(source, ControllerOwnedV2VerificationContext):
            payload = source.to_public_mapping()
        elif hasattr(source, "to_dict") and callable(source.to_dict):
            payload = dict(source.to_dict())
        elif isinstance(source, Mapping):
            payload = dict(source)
        else:
            return None, "controller_context_unsupported_source"
        # Explicitly refuse certificate fill-in even when supplied.
        if certificate is not None and isinstance(certificate, Mapping):
            for name in REQUIRED_CONTROLLER_V2_PIN_FIELDS:
                if not _pin_text(payload.get(name)) and _pin_text(certificate.get(name)):
                    # Leave the pin empty so incompleteness is preserved.
                    payload.pop(name, None)
        if retained_receipt_bytes is not None:
            bounded = _bounded_retained_bytes(bytes(retained_receipt_bytes))
            if bounded is None:
                return None, "retained_receipt_bytes_oversized"
            payload["retained_receipt_bytes_hex"] = bounded.hex()
        if retained_candidate_context_bytes is not None:
            bounded = _bounded_retained_bytes(bytes(retained_candidate_context_bytes))
            if bounded is None:
                return None, "retained_candidate_context_bytes_oversized"
            payload["retained_candidate_context_bytes_hex"] = bounded.hex()
        if retained_execution_key_bytes is not None:
            bounded = _bounded_retained_bytes(bytes(retained_execution_key_bytes))
            if bounded is None:
                return None, "retained_execution_key_bytes_oversized"
            payload["retained_execution_key_bytes_hex"] = bounded.hex()
        payload.setdefault("source", "controller_reconstruction")
        return admit_controller_owned_v2_context(
            payload,
            require_complete=require_complete,
            certificate=certificate,
        )
    except Exception:
        return None, "controller_context_reconstruction_exception"


__all__ = [
    "CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE",
    "CANDIDATE_PUBLICATION_ENVELOPE_SCHEMA",
    "COMPLETED_EXECUTION_IDENTITY_INTERFACE",
    "COMPLETED_EXECUTION_IDENTITY_SCHEMA",
    "CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_INTERFACE",
    "CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_SCHEMA",
    "CandidatePublicationEnvelope",
    "CompletedExecutionIdentity",
    "ControllerOwnedV2VerificationContext",
    "ITEM_CANDIDATE_PUBLICATION_ATTRIBUTE",
    "ITEM_COMPLETED_IDENTITY_ATTRIBUTE",
    "ITEM_CONTROLLER_V2_CONTEXT_ATTRIBUTE",
    "MAX_CONTROLLER_V2_COMPONENT_BYTES",
    "MAX_CONTROLLER_V2_CONTEXT_BYTES",
    "MAX_CONTROLLER_V2_PIN_CHARS",
    "MAX_CONTROLLER_V2_RETAINED_BYTES",
    "OPTIONAL_CONTROLLER_V2_PIN_FIELDS",
    "REQUIRED_CONTROLLER_V2_PIN_FIELDS",
    "admit_controller_owned_v2_context",
    "assemble_candidate_publication",
    "build_completed_execution_identity",
    "finalize_cold_pass_publication",
    "publication_is_authoritative",
    "reconstruct_controller_owned_v2_context",
    "rehash_controller_owned_public_bytes",
]
