"""Complete public Doctor keys for the admitted local header proof recipe.

This producer reuses signed scoped source and actual proof preparation inputs.
Explicit profile identities are admitted by the host, never inferred from
diagnostic placeholders. Retained canonical bytes bind identity, not truth or
isolation. The budget records requested limits, not evidence of enforcement.
No owner, model, prover or positive-proof receipt is created here.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from ..analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE, STRICT_ARTIFACT_PROFILE, identify_logic_ir,
    identify_strict_artifact,
)
from ..proof.doctor_proof_cache import DoctorIdentityBinding, DoctorProofCacheKey, _contains_private_material
from ..proof.deterministic_doctor_hammer import (
    DoctorExactLoweringReceipt, DoctorExecutableRole, DoctorHammerBounds,
    DoctorPinnedExecutable, DoctorReviewedTheorem,
)
from ..proof.formal_verification_contracts import AssuranceLevel, canonical_json, content_identity
from .formal_status_census import MAX_KEY_BYTES, _keys
from .supervisor_meta_index import public_replay_without_metadata

_VERSIONED_SCHEMA = re.compile(r".+@(?:[1-9][0-9]*)(?:\.[0-9]+)*\Z")
HEADER_SOURCE_PREMISE_SCHEMA = "reviewed-header-source-premise@1"
_SOURCE_CORRESPONDENCE = "witnesses->ast_source_observations@1"
_FLOW_CLAIM = "partial reaching-store projection; additional header sources unproved"


class DoctorContractKeyCustodyError(ValueError):
    """A complete local header proof key lost its admitted input binding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DoctorContractKeyCustodyError(message)


def _json(value) -> str:
    rendered = canonical_json(value)
    _require(len(rendered.encode()) <= MAX_KEY_BYTES, "proof input exceeds the native key bound")
    return rendered


def _profile(value) -> DoctorIdentityBinding:
    _require(type(value) is DoctorIdentityBinding, "exact admitted profile identity required")
    _require(type(value.canonical_bytes) is bytes
             and len(value.canonical_bytes) <= MAX_KEY_BYTES // 2,
             "bounded retained profile bytes required")
    _require(type(value.cid_version) is int and value.cid_version == 1,
             "exact native CID version required")
    for name in ("logical_id", "profile", "cid", "digest", "multibase", "multicodec",
                 "multihash", "domain", "artifact_schema"):
        field = getattr(value, name)
        _require(type(field) is str and len(field.encode()) <= 4096,
                 "bounded native profile fields required")
    body = _json(value.to_dict())
    restored = DoctorIdentityBinding.from_dict(json.loads(body))
    _require(_json(restored.to_dict()) == body and not restored.contains_private_material,
             "public canonical profile roundtrip required")
    payload = json.loads(restored.canonical_bytes,
                         parse_float=Decimal if restored.profile == LOGIC_IR_PROFILE else float)
    if restored.profile == STRICT_ARTIFACT_PROFILE:
        identified = identify_strict_artifact(payload)
        declared = payload
    else:
        _require(restored.profile == LOGIC_IR_PROFILE and type(payload) is dict
                 and set(payload) == {"canonicalization", "collection_semantics", "domain",
                                      "identity_profile", "payload", "schema_version"}
                 and payload["identity_profile"] == LOGIC_IR_PROFILE
                 and payload["domain"] == restored.domain
                 and payload["schema_version"] == restored.artifact_schema,
                 "exact declared logic profile envelope required")
        identified = identify_logic_ir(payload["payload"], domain=restored.domain,
            schema_version=restored.artifact_schema,
            collection_schema=payload["collection_semantics"])
        declared = payload["payload"]
    _require(identified.canonical_bytes == restored.canonical_bytes
             and identified.cid == restored.cid and identified.profile == restored.profile,
             "canonical profile preimage differs")
    _require(type(declared) is dict and declared
             and type(declared.get("schema")) is str
             and _VERSIONED_SCHEMA.fullmatch(declared["schema"]) is not None,
             "concrete versioned proof profile declaration required")
    _require(not (type(declared) is dict and set(declared) == {"repository_id", "tree_id", "prefix"}),
             "diagnostic placeholder is not an admitted proof profile")
    _require(not (type(declared) is dict
                  and declared.get("mode") == "signed-task-scoped-diagnostics-only@1"),
             "diagnostic policy is not a reusable proof policy")
    return restored


@dataclass(frozen=True, slots=True)
class HeaderContractProofKeyProfiles:
    """Explicit host-admitted canonical profiles; this type grants no authority."""

    corpus: DoctorIdentityBinding
    registry: DoctorIdentityBinding
    policy: DoctorIdentityBinding
    sandbox: DoctorIdentityBinding
    environment: DoctorIdentityBinding

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _profile(getattr(self, name)))


def _artifact(value, *, logical_id: str, expected_cid: str | None = None) -> DoctorIdentityBinding:
    rendered = _json(value)
    identity = identify_strict_artifact(json.loads(rendered))
    _require(identity.canonical_bytes == rendered.encode()
             and (expected_cid is None or identity.cid == expected_cid),
             "native artifact preimage differs from its original identity")
    return DoctorIdentityBinding.from_identity(identity, logical_id=logical_id)


def _typed(value, expected_type):
    _require(type(value) is expected_type, "exact native proof preparation input required")
    body = _json(value.to_dict())
    restored = expected_type.from_dict(json.loads(body))
    _require(_json(restored.to_dict()) == body, "native proof preparation did not roundtrip")
    return restored


def _source_observations(observations, *, protocol, normalizers, maximum_line):
    """Validate only the recognizer's closed public AST observation vocabulary."""
    _require(type(observations) is list and len(observations) == 1,
             "one exact public header AST observation required")
    row = observations[0]
    fields = {"callback_parameter", "callback_calls", "flow_claim", "property_class",
        "property_symbol", "property_line", "store_attribute", "setters",
        "field_name_normalizer", "field_value_normalizer"}
    _require(type(row) is dict and set(row) == fields,
             "closed public header AST observation required")

    def identifier(value):
        return type(value) is str and len(value.encode()) <= 4096 and value.isidentifier()

    def line(value):
        return type(value) is int and 1 <= value <= maximum_line

    _require(type(protocol) is dict and set(protocol) == {"review_ref", "callback_parameter"}
             and row["callback_parameter"] == protocol["callback_parameter"]
             and row["flow_claim"] == _FLOW_CLAIM
             and all(identifier(row[name]) for name in (
                 "callback_parameter", "property_class", "property_symbol", "store_attribute",
                 "field_name_normalizer", "field_value_normalizer"))
             and line(row["property_line"]),
             "public header AST observation differs from the reviewed protocol")
    _require(type(normalizers) is list and len(normalizers) == 2,
             "exact observed header normalizers required")
    roles = {}
    for normalizer in normalizers:
        _require(type(normalizer) is dict and set(normalizer) == {
            "role", "symbol", "conversion", "ast_sha256", "already_guarded"}
            and normalizer["role"] in {"field_name", "field_value"}
            and normalizer["role"] not in roles
            and identifier(normalizer["symbol"]) and identifier(normalizer["conversion"])
            and type(normalizer["already_guarded"]) is bool
            and type(normalizer["ast_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", normalizer["ast_sha256"]) is not None,
            "closed source-derived header normalizer required")
        roles[normalizer["role"]] = normalizer["symbol"]
    _require(all(row[role + "_normalizer"] == symbol for role, symbol in roles.items()),
             "public AST observation differs from exact normalizer roles")
    for name, keys in (("callback_calls", {"function", "line", "property"}),
                       ("setters", {"function", "line"})):
        rows = row[name]
        _require(type(rows) is list and 1 <= len(rows) <= maximum_line,
                 "bounded public AST relation population required")
        for relation in rows:
            _require(type(relation) is dict and set(relation) == keys
                     and identifier(relation["function"]) and line(relation["line"])
                     and (name != "callback_calls"
                          or relation["property"] == row["property_symbol"]),
                     "public AST relation differs from exact source and property scope")


def restore_header_source_premise(projected: dict, *, expected_contract: dict) -> dict:
    """Invert exact field correspondence; restoration itself grants no admission.

The projection has a distinct artifact identity. Its retained original CID is
checked against the complete restored native contract and the independently
expected contract; neither a digest nor restoration authenticates source.
"""
    _require(type(projected) is dict and set(projected) == {
        "schema", "original_contract_cid", "contract_fields", "ast_source_observations",
        "field_correspondence"} and projected["schema"] == HEADER_SOURCE_PREMISE_SCHEMA
        and projected["field_correspondence"] == _SOURCE_CORRESPONDENCE
        and type(projected["contract_fields"]) is dict
        and "witnesses" not in projected["contract_fields"]
        and type(expected_contract) is dict,
        "closed reviewed source-premise projection required")
    body = json.loads(_json(projected))
    _require(not _contains_private_material(body),
             "private material cannot enter a public source-premise projection")
    restored = dict(body["contract_fields"], witnesses=body["ast_source_observations"])
    _require(content_identity(restored) == body["original_contract_cid"]
             and _json(restored) == _json(expected_contract),
             "public source premise does not restore the exact original contract")
    return restored


def project_header_source_premise(*, scoped, contract: dict, candidate_cid: str) -> dict:
    """Project verified public AST observations without proof-witness conversion.

The native header recognizer calls these observations ``witnesses``. This
reviewed source schema retains every observation visibly after exact source
replay; arbitrary private witnesses cannot enter the closed vocabulary.
"""
    from .doctor_contract_proof import _replay_bound_candidate
    from ..analysis.doctor_header_contracts import CONTRACT_SCHEMA, OPERATOR_ID

    _require(type(contract) is dict, "exact native header contract required")
    contract = json.loads(_json(contract))
    with public_replay_without_metadata():
        replayed, path = _replay_bound_candidate(
            scoped=scoped, contract=contract, candidate_cid=candidate_cid)
        _require(replayed == contract and contract["schema"] == CONTRACT_SCHEMA
                 and contract["operator_id"] == OPERATOR_ID
                 and contract["source_sha256"] == hashlib.sha256(scoped.sources[path]).hexdigest(),
                 "public header premise differs from exact native source replay")
        maximum_line = len(scoped.sources[path].decode("utf-8").split("\n"))
        _source_observations(contract["witnesses"], protocol=contract["protocol"],
            normalizers=contract["normalizers"], maximum_line=maximum_line)
        projected = {"schema": HEADER_SOURCE_PREMISE_SCHEMA,
            "original_contract_cid": content_identity(contract),
            "contract_fields": {name: value for name, value in contract.items() if name != "witnesses"},
            "ast_source_observations": contract["witnesses"],
            "field_correspondence": _SOURCE_CORRESPONDENCE}
        restore_header_source_premise(projected, expected_contract=contract)
        scoped.assert_current()
    return json.loads(_json(projected))


def _native_pin_config(*, proof_module, solver_pin, kernel_pin, source, toolchain_preimage):
    """Join both retained commands to the actual recipe and current leaf bytes."""
    pins = (solver_pin, kernel_pin)
    package_root = str(Path(proof_module.__file__).resolve().parents[3])
    executable = Path(sys.executable).resolve(strict=True)
    prefix = ("-I", "-c", proof_module._PROVER_ADAPTER, package_root)
    for pin, role in zip(pins, ("solver", "kernel")):
        _require(type(pin.argv) is tuple and len(pin.argv) == 6
                 and pin.argv[:4] == prefix and pin.argv[5] == role
                 and pin.verifier_id == "verifier:" + role + ":header-guard"
                 and pin.executable_path == str(executable),
                 "native header prover command differs from the reviewed recipe")
    cfg = json.loads(solver_pin.argv[4])
    _require(type(cfg) is dict and set(cfg) == {
        "lean", "z3", "lean_sha256", "z3_sha256", "source_sha256", "smt",
        "kernel_id", "expected_axioms"}, "closed native prover configuration required")
    leaves = {}
    digests = {}
    for role in ("lean", "z3"):
        _require(type(cfg[role]) is str and cfg[role], "canonical prover leaf path required")
        path = Path(cfg[role])
        _require(path.is_absolute() and path.resolve(strict=True) == path
                 and path.is_file() and os.access(path, os.R_OK | os.X_OK),
                 "canonical existing readable prover leaf required")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == cfg[role + "_sha256"] and digest == toolchain_preimage[role],
                 "current prover leaf bytes differ from the command and toolchain")
        leaves[role], digests[role] = path, digest
    expected_cfg = proof_module._header_contract_prover_config(
        lean=leaves["lean"], z3=leaves["z3"], source=source,
        lean_sha256=digests["lean"], z3_sha256=digests["z3"])
    _require(canonical_json(cfg) == canonical_json(expected_cfg)
             and all(pin.argv[4] == json.dumps(expected_cfg) for pin in pins),
             "native prover configuration differs from the exact header recipe")
    executable_digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    _require(all(pin.executable_sha256 == executable_digest for pin in pins),
             "current Python executable bytes differ from native proof pins")
    return tuple((leaves[role], digests[role]) for role in ("lean", "z3")), executable, executable_digest


def build_header_contract_proof_key(
    *, scoped, contract: dict, candidate_cid: str, theorem: DoctorReviewedTheorem,
    lowering: DoctorExactLoweringReceipt, solver_pin: DoctorPinnedExecutable,
    kernel_pin: DoctorPinnedExecutable, bounds: DoctorHammerBounds,
    translator_preimage: dict, toolchain_preimage: dict,
    profiles: HeaderContractProofKeyProfiles,
) -> DoctorProofCacheKey:
    """Produce one bounded full key without storage, execution or proof reuse.

The native goal is the exact local reviewed theorem, not the Terminal task or
whole-program goal. Import-graph and scoped-source bindings keep their original
limited meaning. The already admitted host supplies every missing proof profile.
"""
    _require(type(profiles) is HeaderContractProofKeyProfiles, "explicit admitted proof profiles required")
    # Repeat validation: a frozen object is not authentication, and callers can
    # otherwise manufacture instances without running the dataclass constructor.
    admitted = {name: _profile(getattr(profiles, name)) for name in profiles.__dataclass_fields__}
    _require(type(contract) is dict and type(translator_preimage) is dict
             and type(toolchain_preimage) is dict,
             "exact canonical contract and proof profile preimages required")
    contract = json.loads(_json(contract))
    translator_preimage = json.loads(_json(translator_preimage))
    toolchain_preimage = json.loads(_json(toolchain_preimage))
    _require(type(candidate_cid) is str and 0 < len(candidate_cid.encode()) <= 4096,
             "bounded native candidate identity required")
    with public_replay_without_metadata():
        theorem = _typed(theorem, DoctorReviewedTheorem)
        lowering = _typed(lowering, DoctorExactLoweringReceipt)
        solver_pin = _typed(solver_pin, DoctorPinnedExecutable)
        kernel_pin = _typed(kernel_pin, DoctorPinnedExecutable)
        bounds = _typed(bounds, DoctorHammerBounds)
        lowering.verify_theorem(theorem)
        return _build(scoped=scoped, contract=contract, candidate_cid=candidate_cid,
            theorem=theorem, lowering=lowering, solver_pin=solver_pin,
            kernel_pin=kernel_pin, bounds=bounds, translator_preimage=translator_preimage,
            toolchain_preimage=toolchain_preimage, admitted=admitted)


def _build(*, scoped, contract, candidate_cid, theorem, lowering, solver_pin,
           kernel_pin, bounds, translator_preimage, toolchain_preimage, admitted):
    from . import doctor_contract_proof as proof_module
    from .doctor_scoped_analysis import ScopedDoctorAnalysis, _selection
    from .doctor_contract_proof import (
        LEAN_SOURCE, OPERATOR, _implementation_bindings, _replay_bound_candidate,
    )
    from .doctor_task_workflow import _PROVER_ADAPTER
    from ..analysis.analysis_ast_index import ANALYSIS_AST_INDEX_SCHEMA
    from ..analysis.doctor_repository_diagnostics import DoctorEvidenceSnapshot, QuerySurface

    _require(type(scoped) is ScopedDoctorAnalysis, "exact native scoped source required")
    scoped.assert_current()
    replayed_contract, _path = _replay_bound_candidate(
        scoped=scoped, contract=contract, candidate_cid=candidate_cid)
    _require(replayed_contract == contract, "exact admitted header contract required")
    verified, spec, paths = _selection(scoped.repository, json.loads(scoped._admission_json), scoped.task_cid)
    manifest, observed = verified["manifest"], scoped.report
    scope = {"schema": "supervisor-doctor-signed-task-scope@1",
        "manifest_cid": verified["receipt"]["manifest_cid"], "task_cid": scoped.task_cid,
        "task_spec_cid": content_identity(spec), "source_tree_id": verified["current_source_tree_id"],
        "sources": {name: manifest["sources"][name] for name in paths}}
    scope_id = content_identity(scope)
    roots, source_roots = theorem.roots, scoped.snapshot.roots
    _require(scope_id == observed["scope_id"]
             and roots.repository_id == manifest["repository_cid"]
             and source_roots.repository_id == manifest["repository_cid"]
             and all(getattr(root, name) == scope_id for root in (roots, source_roots)
                     for name in ("forest_id", "tree_id", "overlay_id")),
             "proof roots differ from current signed source scope")
    for name in ("corpus", "policy", "environment"):
        _require(admitted[name].logical_id == getattr(roots, name + "_id"),
                 "proof profile differs from the reviewed theorem roots")
    for name, original in (("corpus", source_roots.corpus_id),
                           ("registry", source_roots.operator_registry_id),
                           ("sandbox", source_roots.sandbox_id),
                           ("environment", source_roots.environment_id)):
        _require(admitted[name].logical_id != original,
                 "scoped diagnostic placeholder cannot supply a proof profile")

    diagnostic = scoped.diagnostic_snapshot
    _require(type(diagnostic) is DoctorEvidenceSnapshot, "exact native diagnostic snapshot required")
    index = diagnostic.ast_index
    _require(index.paths == paths and source_roots.ast_root_id == index.index_id
             and diagnostic.authority_roots.ast_index_id == index.index_id
             and roots.index_id == source_roots.index_id,
             "native AST population differs from signed source scope")
    for row in index.path_records:
        digest = "sha256:" + manifest["sources"][row.path]["sha256"]
        _require(row.ast_record.source_sha256 == digest and row.ast_record.blob_identity == digest,
                 "native AST source bytes differ from signed admission")
    ast_preimage = {"schema": ANALYSIS_AST_INDEX_SCHEMA, "schema_version": index.schema_version,
        "path_records": [row.to_dict() for row in index.path_records]}
    _require(index.index_id == "analysis-ast-index:sha256:" + hashlib.sha256(_json(ast_preimage).encode()).hexdigest(),
             "native AST canonical preimage differs")
    graph_preimage = {"imports": [hit.to_dict()
        for hit in diagnostic.query_index.get(QuerySurface.IMPORTS.value, ())]}
    graph_id = "doctor-import-graph:sha256:" + hashlib.sha256(_json(graph_preimage).encode()).hexdigest()
    _require(graph_id == diagnostic.authority_roots.import_graph_id
             and graph_id == source_roots.graph_id and graph_id == roots.graph_id,
             "native import graph preimage differs")

    contract_id = content_identity(contract)
    consequence = content_identity({"operator": OPERATOR, "contract": contract_id,
        "candidate": candidate_cid, "analysis": observed["analysis_cid"]})
    source = ("-- theorem:" + OPERATOR + " property:reject-control-preserve-safe "
              "claim:local-header-guard " + consequence + "\n" + LEAN_SOURCE)
    implementation_bindings = _implementation_bindings()
    _require(translator_preimage == {"operator": OPERATOR, "implementations": implementation_bindings},
             "translator preimage differs from current reviewed implementation bytes")
    translator_id = content_identity(translator_preimage)
    _require(theorem.translator_id == "translator:" + translator_id
             and theorem.toolchain_id == content_identity(toolchain_preimage)
             and set(toolchain_preimage) == {"lean", "z3", "adapter", "translator"}
             and toolchain_preimage["translator"] == translator_id
             and toolchain_preimage["adapter"] == hashlib.sha256(_PROVER_ADAPTER.encode()).hexdigest(),
             "translator or toolchain canonical preimage differs")
    _require(theorem.theorem_id == "theorem:" + OPERATOR
             and theorem.property_id == "property:reject-control-preserve-safe"
             and theorem.claim_id == "claim:local-header-guard" and theorem.body_format == "lean4"
             and theorem.consequence_ref == consequence and theorem.theorem_body == source
             and theorem.premise_ids == (contract_id,)
             and theorem.assumption_ids == tuple(sorted((
                 "assumption:reviewed-exact-python-ast-translation", "assumption:declared-wsgi-protocol")))
             and theorem.review_receipt_id == content_identity({"operator": OPERATOR,
                 "implementations": implementation_bindings, "contract": contract_id})
             and lowering.logic_ir_statement == source and lowering.native_statement == source,
             "reviewed theorem differs from the exact admitted local header recipe")
    _require(solver_pin.role is DoctorExecutableRole.SOLVER
             and kernel_pin.role is DoctorExecutableRole.KERNEL
             and solver_pin.command_id != kernel_pin.command_id
             and solver_pin.verifier_id != kernel_pin.verifier_id
             and all(pin.toolchain_id == theorem.toolchain_id
                     and pin.environment_id == roots.environment_id for pin in (solver_pin, kernel_pin)),
             "native executable pins differ from the reviewed proof profile")
    leaves, executable, executable_digest = _native_pin_config(
        proof_module=proof_module, solver_pin=solver_pin, kernel_pin=kernel_pin,
        source=source, toolchain_preimage=toolchain_preimage)

    scoped.assert_current()
    scope_binding = _artifact(scope, logical_id=scope_id, expected_cid=scope_id)
    premise = project_header_source_premise(scoped=scoped, contract=contract, candidate_cid=candidate_cid)
    _require(content_identity(premise) != contract_id,
             "source projection must retain its distinct versioned artifact identity")
    key = DoctorProofCacheKey(forest=scope_binding, tree=scope_binding, overlay=scope_binding,
        candidate_tree=scope_binding,
        ast=_artifact(ast_preimage, logical_id=index.index_id),
        graph=_artifact(graph_preimage, logical_id=graph_id),
        goal=_artifact(theorem.to_dict(), logical_id=theorem.theorem_id, expected_cid=theorem.content_id),
        premises=(_artifact(premise, logical_id=contract_id, expected_cid=content_identity(premise)),),
        translation=_artifact(translator_preimage, logical_id=theorem.translator_id, expected_cid=translator_id),
        toolchain=_artifact(toolchain_preimage, logical_id=theorem.toolchain_id, expected_cid=theorem.toolchain_id),
        solver=_artifact(solver_pin.to_dict(), logical_id=solver_pin.verifier_id, expected_cid=solver_pin.content_id),
        kernel=_artifact(kernel_pin.to_dict(), logical_id=kernel_pin.verifier_id, expected_cid=kernel_pin.content_id),
        corpus=admitted["corpus"], registry=admitted["registry"], policy=admitted["policy"],
        sandbox=admitted["sandbox"], environment=admitted["environment"],
        budget=bounds.to_resource_budget(), required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    _keys((key,), None)
    scoped.assert_current()
    _require(_implementation_bindings() == implementation_bindings,
             "reviewed implementation bytes changed during key preparation")
    _require(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in leaves)
             and "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest() == executable_digest,
             "pinned executable bytes changed during key preparation")
    return key
