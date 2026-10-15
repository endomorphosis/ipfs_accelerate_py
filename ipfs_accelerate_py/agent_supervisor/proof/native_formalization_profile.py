"""Lossless native Intent/Security observations, never operational admission.

Accept concrete objects from the pinned datasets tree. Native serializers and
validators supply the payloads and identities; this module does not compile,
invent predicates, lower expressions, contact a provider, or authenticate an
owner. Legacy expressions remain complete JSON, including text and numbers.
The profile is deliberately separate from the shared normalized IR contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

SCHEMA = "native-legacy-formalization-profile/v1"
STATUS = "unsupported_for_operational_admission"
MAX_BYTES = 2 * 1024 * 1024
MAX_DEPTH = 64
NATIVE_ROOT = Path("/home/barberb/lift_coding/external/ipfs_datasets")
SOURCE_SCOPE = "current pinned native validation/serialization sources; not artifact creation history or source authenticity"
AUTHORITY = {
    "owner_verification_required": True,
    "owner_authenticated": False,
    "source_semantics_verified": False,
    "execution_authority": False,
    "proof_authority": False,
    "qualified": False,
    "admitted": False,
    "Lean_admitted": False,
    "formalized": False,
    "roundtrip_ok": False,
    "Constitution_formalized": False,
}
_MODULES = (
    "ipfs_datasets_py.logic.intent_ir.schema",
    "ipfs_datasets_py.logic.intent_ir.decoder",
    "ipfs_datasets_py.logic.intent_ir.canonicalize",
    "ipfs_datasets_py.logic.security_ir.model",
    "ipfs_datasets_py.logic.formalization.compiler",
    "ipfs_datasets_py.logic.formalization.views",
    "ipfs_datasets_py.logic.ir_core.identity",
)
_PIN_NAMES = frozenset((*_MODULES, "ipfs_datasets_py", "ipfs_datasets_py.logic",
    "ipfs_datasets_py.logic.intent_ir", "ipfs_datasets_py.logic.security_ir",
    "ipfs_datasets_py.logic.formalization", "ipfs_datasets_py.logic.formalization.samples",
    "ipfs_datasets_py.logic.ir_core", "ipfs_datasets_py.logic.syntax_core"))


class NativeFormalizationProfileError(ValueError):
    """Malformed, drifted, unsupported or inconsistent native observation."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise NativeFormalizationProfileError(message)


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_types(value: Any, depth: int = 0) -> None:
    _need(depth <= MAX_DEPTH, "profile exceeds JSON nesting bound")
    if type(value) is dict:
        _need(all(type(key) is str for key in value), "JSON object keys must be strings")
        for child in value.values():
            _json_types(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _json_types(child, depth + 1)
    else:
        _need(type(value) in (str, int, float, bool, type(None)), "exact JSON values required")
        if type(value) is float:
            _need(math.isfinite(value), "nonfinite JSON number refused")


def _encoded(value: Any) -> bytes:
    _json_types(value)
    try:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise NativeFormalizationProfileError("bounded finite JSON required") from exc
    _need(len(body) <= MAX_BYTES, "profile exceeds 2 MiB byte bound")
    return body


def _decoded(body: bytes | str) -> dict[str, Any]:
    _need(type(body) in (bytes, str), "JSON bytes or text required")
    try:
        wire = body.encode("utf-8") if type(body) is str else body
    except UnicodeError as exc:
        raise NativeFormalizationProfileError("UTF-8 JSON required") from exc
    _need(len(wire) <= MAX_BYTES, "profile exceeds 2 MiB byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key refused")
            result[key] = value
        return result

    def nonfinite(value):
        raise NativeFormalizationProfileError("nonfinite JSON number refused")

    try:
        value = json.loads(wire, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, NativeFormalizationProfileError):
            raise
        raise NativeFormalizationProfileError("bounded finite UTF-8 JSON required") from exc
    _need(type(value) is dict, "profile must be a JSON object")
    _json_types(value)
    return value


def _guard_loaded() -> dict[str, Path]:
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
            _need(module is not None, "datasets import is incomplete")
            origin = getattr(module, "__file__", None)
            if origin:
                path = Path(origin).resolve()
                _need(path.is_relative_to(NATIVE_ROOT.resolve()), "datasets module escaped pinned native tree")
                _need(path.is_file(), "loaded native module source unavailable")
                paths[name] = path
            for root in getattr(module, "__path__", ()):
                _need(Path(root).resolve().is_relative_to(NATIVE_ROOT.resolve()),
                      "datasets namespace escaped pinned native tree")
    return paths


def _source_pins() -> dict[str, dict[str, Any]]:
    return {name: {"path": str(path), "bytes": len(body), "sha256": _sha(body)}
            for name, path in sorted(_guard_loaded().items())
            if name in _PIN_NAMES or name.startswith(("ipfs_datasets_py.logic.ir_core.",
                                                      "ipfs_datasets_py.logic.syntax_core."))
            for body in (path.read_bytes(),)}


def _native_api():
    _guard_loaded()
    try:
        spec = importlib.util.find_spec("ipfs_datasets_py")
        _need(spec is not None and spec.origin is not None
              and Path(spec.origin).resolve() == (NATIVE_ROOT / "ipfs_datasets_py/__init__.py").resolve(),
              "pinned datasets package must be selected before calling this API")
        modules = {name: importlib.import_module(name) for name in _MODULES}
    except (ImportError, ValueError) as exc:
        if isinstance(exc, NativeFormalizationProfileError):
            raise
        raise NativeFormalizationProfileError("pinned native serializers unavailable") from exc
    _guard_loaded()
    return modules


def _byte_ref(body: bytes, scope: str) -> dict[str, Any]:
    _need(type(body) is bytes and len(body) <= MAX_BYTES, "native bytes exceed 2 MiB bound")
    return {"bytes": len(body), "sha256": _sha(body), "scope": scope}


def _gaps(domain, payload, artifact, artifact_wire):
    gaps = []

    def add(code, path, message, **extra):
        gaps.append({"code": code, "path": path, "message": message, **extra})

    add("independent_owner_verifier_required", "/owner_verification",
        "Native hashes, review labels and serialization do not authenticate source custody or authority.")
    add("no_kernel_admission", "/lake_build",
        "This observation performs no source-locked lake build <Lib>.")
    add("formalization_semantics_not_verified", "/native_formalization",
        "Native compile declarations and syntactic identities are not semantic or solver results.")
    for index, formula in enumerate(artifact.formulas):
        path = f"/native_formalization/formulas/{index}"
        if not formula.is_typed:
            add("legacy_formula_expression", path + "/expression",
                "Complete legacy JSON/text is retained; its labels do not establish TypedExpression@1.",
                formula_id=formula.formula_id, view_id=formula.view_id)
        if formula.opaque:
            add("opaque_formula", path + "/opaque", "Native opaque semantics remain unsupported.",
                formula_id=formula.formula_id, view_id=formula.view_id)
    for index, wire in enumerate(artifact_wire["diagnostics"]["diagnostics"]):
        if wire["code"] == "ir.feature.unsupported" or wire["severity"] in ("error", "fatal"):
            add("native_compiler_diagnostic", f"/native_formalization/diagnostics/diagnostics/{index}",
                "The complete native diagnostic remains part of this observation.",
                native_code=wire["code"], native_field_path=wire["location"]["field_path"])
    for index, source in enumerate(payload["sources"]):
        path = f"/native_declaration/sources/{index}"
        add("source_custody_not_authenticated", path,
            "The original source descriptor and review status are retained without an owner decision.")
        if not source.get("content_sha256"):
            add("original_source_digest_missing", path + "/content_sha256",
                "A descriptor digest cannot substitute for original source-content bytes.")
        if domain == "security" and not source["revision"]:
            add("original_source_revision_missing", path + "/revision", "No source version is inferred.")
    for index, source in enumerate(artifact_wire["source_map"]["sources"]):
        if source["metadata"].get("content_binding") == "security_source_descriptor":
            add("source_descriptor_is_not_original_content", f"/native_formalization/source_map/sources/{index}",
                "The native adapter explicitly hashed a source descriptor, not original source bytes.")
    if domain == "intent":
        for index, statement in enumerate(payload["statements"]):
            path = f"/native_declaration/statements/{index}"
            if not statement["predicate"]:
                add("native_predicate_missing", path + "/predicate", "No predicate is invented from normalized text.")
            if statement["grounding"] == "inferred":
                add("inferred_native_statement", path + "/grounding", "Native inference is not upgraded to grounded authority.")
        for index, action in enumerate(payload["actions"]):
            add("independent_action_effect_binding_required", f"/native_declaration/actions/{index}",
                "Native action and statement links do not supply independent candidate/effect-stream bindings.",
                action_id=action["action_id"])
    else:
        for index, policy in enumerate(payload["policies"]):
            add("native_policy_is_not_operational_grant", f"/native_declaration/policies/{index}",
                "Native effect/reference lists are preserved; no exact runtime authorization scope or grant is produced.",
                policy_id=policy["policy_id"])
        for index, machine in enumerate(payload["state_machines"]):
            path = f"/native_declaration/state_machines/{index}"
            add("independent_runtime_state_binding_required", path,
                "Initial state is a declaration, not current state, resource or state-version evidence.",
                state_machine_id=machine["state_machine_id"])
            for number, transition in enumerate(machine["transitions"]):
                for field in ("guard", "effect"):
                    if transition[field]:
                        add("native_transition_semantics_not_lowered", f"{path}/transitions/{number}/{field}",
                            "Native transition text is preserved; no interpreted guard/effect is fabricated.")
        for index, extension in enumerate(payload["extensions"]):
            add("native_extension_semantics_not_lowered", f"/native_declaration/extensions/{index}",
                "Optional and required extension payloads are both retained without a semantic lowering.",
                extension_id=extension["extension_id"])
    return gaps


def _capture(declaration, artifact):
    api = _native_api()
    before = _source_pins()
    try:
        intent = api[_MODULES[0]]
        decoder = api[_MODULES[1]]
        canonical = api[_MODULES[2]]
        security = api[_MODULES[3]]
        formal = api[_MODULES[4]]
        _need(type(artifact) is formal.FormalizationArtifact, "actual pinned FormalizationArtifact required")
        if type(declaration) is intent.IntentIRDocument:
            domain, native_id = "intent", declaration.document_id
            declaration_wire = declaration.to_dict()
            checked = decoder.decode_intent_ir(declaration_wire)
            declaration_bytes = canonical.canonical_intent_ir_bytes(declaration)
            declaration_identity_bytes = declaration_bytes
            native_digest = canonical.intent_ir_sha256(declaration)
            declaration_identity = {"digest": native_digest, "schema_version": declaration.schema_version,
                                    "scope": "canonical_intent_ir_bytes; native API supplies no CID"}
        elif type(declaration) is security.SecurityIR:
            domain, native_id = "security", declaration.declaration_id
            declaration_wire = declaration.to_dict()
            checked = security.SecurityIR.from_dict(declaration_wire)
            declaration_bytes = declaration.canonical_bytes()
            declaration_identity_bytes = declaration.identity.canonical_bytes
            native_digest = declaration.digest
            declaration_identity = declaration.identity.to_dict()
        else:
            raise NativeFormalizationProfileError("actual pinned IntentIRDocument or SecurityIR required")
        _encoded(declaration_wire)
        _need(_encoded(checked.to_dict()) == _encoded(declaration_wire), "native declaration readback repaired fields")
        artifact.validate()
        artifact_wire = artifact.to_dict()
        _encoded(artifact_wire)
        reparsed = formal.FormalizationArtifact.from_dict(artifact_wire)
        _need(_encoded(reparsed.to_dict()) == _encoded(artifact_wire), "native formalization readback repaired fields")
        _need(artifact.domain == domain and artifact.declaration_id == native_id
              and artifact.declaration_digest == native_digest,
              "native domain/declaration identity join mismatch")
        artifact_json = artifact.to_json().encode("utf-8")
        artifact_identity_bytes = artifact.canonical_bytes()
        payload = {
            "schema": SCHEMA, "status": STATUS, "domain": domain,
            "native_declaration": declaration_wire,
            "native_formalization": artifact_wire,
            "byte_bindings": {
                "native_declaration_json": _byte_ref(declaration_bytes, "actual native canonical declaration JSON"),
                "native_declaration_identity": declaration_identity,
                "native_declaration_identity_preimage": _byte_ref(declaration_identity_bytes, "actual native identity input; may differ from declaration JSON"),
                "native_formalization_json": _byte_ref(artifact_json, "actual FormalizationArtifact.to_json transport"),
                "native_formalization_identity": artifact.identity.to_dict(),
                "native_formalization_identity_preimage": _byte_ref(artifact_identity_bytes, "actual FormalizationArtifact canonical identity preimage, not to_json bytes"),
            },
            "serializer_source_pins": before,
            "serializer_source_pin_scope": SOURCE_SCOPE,
            "unsupported_frontier": _gaps(domain, declaration_wire, artifact, artifact_wire),
            "authority": dict(AUTHORITY),
        }
        transport = _encoded(payload)
        return transport, declaration_bytes, declaration_identity_bytes, artifact_json, artifact_identity_bytes
    except (TypeError, ValueError, KeyError, RecursionError) as exc:
        if isinstance(exc, NativeFormalizationProfileError):
            raise
        raise NativeFormalizationProfileError("native validation or serialization refused") from exc

    finally:
        _need(_source_pins() == before, "pinned serializer source closure changed during capture")


@dataclass(frozen=True, init=False)
class NativeFormalizationProfile:
    """Bounded complete observation; native objects are required at creation.

    Source pins observe current native serialization code, not the original
    compiler invocation. Returned dictionaries are detached. Every profile
    stays unsupported for operational admission, including actual typed ASTs.
    """

    _transport_bytes: bytes
    native_declaration_bytes: bytes
    native_declaration_identity_preimage: bytes
    native_formalization_json_bytes: bytes
    native_formalization_identity_preimage: bytes

    def __init__(self, declaration: Any, artifact: Any) -> None:
        values = _capture(declaration, artifact)
        for name, value in zip(self.__dataclass_fields__, values):
            object.__setattr__(self, name, value)

    @property
    def status(self) -> str:
        return STATUS

    @property
    def transport_bytes(self) -> bytes:
        return self._transport_bytes

    @property
    def transport_sha256(self) -> str:
        return _sha(self._transport_bytes)

    def to_dict(self) -> dict[str, Any]:
        return _decoded(self._transport_bytes)


def capture_native_formalization_profile(declaration: Any, artifact: Any) -> NativeFormalizationProfile:
    """Preserve real native data; never invoke the compiler or owner verifier."""
    return NativeFormalizationProfile(declaration, artifact)


def decode_native_formalization_profile(body: bytes | str, *, expected_sha256: str | None = None) -> NativeFormalizationProfile:
    """Validate exact profile readback; no defaults or repaired identities.

    Current pinned serializer source hashes must match the recorded snapshot.
    An optional caller-held transport hash binds the precise supplied bytes.
    Neither that hash nor successful readback authenticates an owner.
    """
    payload = _decoded(body)
    wire = body.encode("utf-8") if type(body) is str else body
    if expected_sha256 is not None:
        _need(type(expected_sha256) is str and expected_sha256 == _sha(wire), "transport identity mismatch")
    _need(set(payload) == {"schema", "status", "domain", "native_declaration", "native_formalization",
          "byte_bindings", "serializer_source_pins", "serializer_source_pin_scope", "unsupported_frontier", "authority"},
          "closed native profile required")
    _need(payload["schema"] == SCHEMA and payload["status"] == STATUS and payload["authority"] == AUTHORITY,
          "native profile cannot upgrade status or authority")
    api = _native_api()
    try:
        if payload["domain"] == "intent":
            declaration = api[_MODULES[1]].decode_intent_ir(payload["native_declaration"])
        elif payload["domain"] == "security":
            declaration = api[_MODULES[3]].SecurityIR.from_dict(payload["native_declaration"])
        else:
            raise NativeFormalizationProfileError("unsupported native profile domain")
        artifact = api[_MODULES[4]].FormalizationArtifact.from_dict(payload["native_formalization"])
        result = NativeFormalizationProfile(declaration, artifact)
    except (TypeError, ValueError, KeyError, RecursionError) as exc:
        if isinstance(exc, NativeFormalizationProfileError):
            raise
        raise NativeFormalizationProfileError("native profile readback refused") from exc
    _need(result.to_dict() == payload and result.transport_bytes == wire,
          "native profile readback repaired fields, identities, source pins or encoding")
    return result


__all__ = ["SCHEMA", "STATUS", "MAX_BYTES", "NativeFormalizationProfileError",
           "NativeFormalizationProfile", "capture_native_formalization_profile", "decode_native_formalization_profile"]
