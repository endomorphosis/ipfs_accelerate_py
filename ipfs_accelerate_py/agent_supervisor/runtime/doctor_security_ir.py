"""Source-grounded native SecurityIR declarations without language-model calls.

The reviewed header AST recognizer supplies observed bindings and the explicit
protocol supplies expectations. Native SecurityIR compilation declares formal
claims and obligations; it does not establish their truth. Proof observations
remain separate and must bind this declaration rather than becoming features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..analysis.doctor_header_contracts import WsgiHeaderProtocolContract, analyze_http_header_contracts
from ..proof.formal_verification_contracts import canonical_json, content_identity
from .doctor_scoped_analysis import ScopedDoctorAnalysis

if TYPE_CHECKING:
    from ipfs_datasets_py.logic.security_ir.model import SecurityIR
    from ipfs_datasets_py.logic.formalization.samples import FormalizationSample
    from ipfs_datasets_py.logic.formalization.compiler import FormalizationArtifact


def _id(kind: str, value: str) -> str:
    return kind + ":" + hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class CompiledHeaderSecurityIR:
    declaration: SecurityIR
    sample: FormalizationSample
    artifact: FormalizationArtifact
    _report_json: str = field(repr=False)

    @property
    def report(self) -> dict:
        return json.loads(self._report_json)


def compile_header_security_ir(*, scoped: ScopedDoctorAnalysis, analyses: Sequence[dict],
                               output: Path | None = None) -> CompiledHeaderSecurityIR:
    """Replay native AST contracts and compile the real declaration vocabulary.

    This handles only recognized local HTTP header normalizers. Unsupported
    Python remains a frontier. No proof result, synthesized source, solver
    verdict, runtime trace or acceptance state enters the declaration.
    """
    from ipfs_datasets_py.logic.security_ir.model import (
        SecurityIR, SecuritySource, Principal, Asset, Resource, Channel, Policy, PolicyEffect,
        ThreatAssumption, SecurityClaim, StateMachine, StateTransition,
    )
    from ipfs_datasets_py.logic.security_ir.formalization_adapter import SecurityIRFormalizationAdapter

    if type(scoped) is not ScopedDoctorAnalysis or not isinstance(analyses, (list, tuple)) or len(analyses) > 64:
        raise ValueError("bounded native scoped analyses required")
    scoped.assert_current()
    observed = scoped.report
    sources = tuple(SecuritySource(source_id=_id("source", path), uri="repository-source:" + path,
        revision=observed["source_tree_id"], content_sha256=observed["source_hashes"][path],
        review_status="unreviewed", attributes={"repository_relative_path": path,
            "scope_id": observed["scope_id"], "source_kind": "inert_python_or_task_input"})
        for path in sorted(scoped.sources))
    assets, resources, policies, assumptions, claims, machines = [], [], [], [], [], []
    supported_paths, unsupported, contract_ids, frontiers, seen = [], [], [], set(), set()
    principals = (Principal("principal:reviewed-wsgi-application", kind="declared-protocol-role"),
                  Principal("principal:reviewed-wsgi-server", kind="declared-protocol-role"))
    channel = Channel(channel_id="channel:reviewed-wsgi-response-headers", protocol="HTTP-response-headers/WSGI",
                      source_node_id=principals[0].principal_id, target_node_id=principals[1].principal_id,
                      source_ids=tuple(source.source_id for source in sources))
    for analysis in analyses:
        if (not isinstance(analysis, dict) or set(analysis) != {
                "path", "status", "reason_codes", "source_sha256", "contracts", "open_frontiers"}
                or analysis["status"] not in {"candidate", "already_satisfied", "unsupported"}):
            raise ValueError("closed native header analysis record required")
        path = analysis["path"]
        if (path in seen or path not in scoped.sources or not path.endswith(".py")
                or analysis["source_sha256"] != observed["source_hashes"][path]):
            raise ValueError("SecurityIR source scope or digest differs")
        seen.add(path)
        if not isinstance(analysis["contracts"], list) or len(analysis["contracts"]) > 1:
            raise ValueError("one reviewed local contract per source required")
        if not analysis["contracts"]:
            if analysis["status"] != "unsupported":
                raise ValueError("supported header analysis requires a declaration")
            unsupported.append({"path": path, "reason_codes": list(analysis["reason_codes"])})
            frontiers.add("unsupported_python_security_semantics")
            continue
        contract = analysis["contracts"][0]
        protocol = WsgiHeaderProtocolContract(**contract["protocol"])
        replay = analyze_http_header_contracts(scoped.sources[path].decode("utf-8"), protocol=protocol)
        if (list(replay.contracts) != analysis["contracts"] or replay.status != analysis["status"]
                or list(replay.reason_codes) != analysis["reason_codes"]
                or list(replay.open_frontiers) != analysis["open_frontiers"]):
            raise ValueError("SecurityIR contract differs from exact native AST replay")
        supported_paths.append(path)
        source_ids = (_id("source", path),)
        contract_id = content_identity(contract)
        contract_ids.append(contract_id)
        frontiers.update(contract["open_frontiers"])
        assumption_ids = []
        for assumption in contract["assumptions"]:
            identifier = _id("assumption", path + "\0" + assumption)
            assumption_ids.append(identifier)
            assumptions.append(ThreatAssumption(identifier, "Assume " + assumption + ".",
                source_ids=source_ids, attributes={"protocol_review_ref": protocol.review_ref,
                    "declared_premise": True}))
        policy_id = _id("policy", path)
        resource_ids = []
        # Preserve source witnesses as structural metadata, not executable code
        # or evidence that the desired guard already holds in current source.
        witness_metadata = {"ast_witnesses": contract["witnesses"],
            "scope_id": observed["scope_id"], "unresolved_frontiers": contract["open_frontiers"],
            "protocol_review_ref": protocol.review_ref}
        for normalizer in contract["normalizers"]:
            key = path + "\0" + normalizer["role"] + "\0" + normalizer["symbol"]
            asset_id, resource_id = _id("asset", key), _id("resource", key)
            resource_ids.append(resource_id)
            attributes = {**witness_metadata, "role": normalizer["role"],
                "conversion_symbol": normalizer["conversion"], "ast_sha256": normalizer["ast_sha256"]}
            assets.append(Asset(asset_id, kind="python-normalizer-function", symbol=normalizer["symbol"],
                                source_ids=source_ids, attributes=attributes))
            resources.append(Resource(resource_id, kind="converted-header-input", asset_ids=(asset_id,),
                                      source_ids=source_ids))
            for obligation, statement in (
                ("reject-controls", "The normalizer must reject a converted ordinary string containing NUL, LF or CR by raising ValueError."),
                ("preserve-safe", "The normalizer must preserve its original normalization result for converted ordinary strings without NUL, LF or CR."),
            ):
                claims.append(SecurityClaim(_id("claim", key + "\0" + obligation), statement,
                    domain="http-header-local-contract", assumption_ids=tuple(assumption_ids),
                    policy_ids=(policy_id,), source_ids=source_ids,
                    attributes={**attributes, "claim_kind": "expected_local_behavior", "obligation": obligation}))
        policies.append(Policy(policy_id, name="Require the reviewed local header control guard",
            effect=PolicyEffect.REQUIRE, resource_ids=tuple(resource_ids), channel_ids=(channel.channel_id,),
            source_ids=source_ids, attributes={"forbidden_codepoints": contract["forbidden_codepoints"],
                "error_type": "ValueError", "normative_reference": contract["normative_reference"],
                "expected_behavior_only": True}))
        machines.append(StateMachine(_id("state-machine", path), states=("converted", "rejected", "accepted"),
            initial_state="converted", source_ids=source_ids,
            transitions=(
                StateTransition("converted", "rejected", "validate-header",
                    guard="contains_any(converted_string, [0, 10, 13])", effect="raise ValueError"),
                StateTransition("converted", "accepted", "validate-header",
                    guard="not contains_any(converted_string, [0, 10, 13])", effect="original normalization result"),
            ), attributes={"semantics": "expected local guard behavior; not observed execution",
                           "unresolved_frontiers": contract["open_frontiers"]}))
    unmodeled_paths = sorted(set(scoped.sources) - set(supported_paths))
    if unmodeled_paths:
        frontiers.add("scoped_sources_without_security_semantics")
    declaration = SecurityIR(declaration_id=_id("doctor-header-security", observed["analysis_cid"]),
        sources=sources, principals=principals, channels=(channel,), assets=tuple(assets), resources=tuple(resources),
        policies=tuple(policies), assumptions=tuple(assumptions), claims=tuple(claims),
        state_machines=tuple(machines))
    adapter = SecurityIRFormalizationAdapter()
    sample = adapter.adapt_sample(declaration)
    artifact = adapter.compile(sample, adapter.default_config(sample))
    report = {"schema": "supervisor-doctor-security-ir-compilation@1",
        "status": "compiled_local_declarations" if supported_paths else "unsupported",
        "analysis_cid": observed["analysis_cid"], "source_tree_id": observed["source_tree_id"],
        "scope_id": observed["scope_id"], "source_hashes": observed["source_hashes"],
        "declaration_cid": declaration.cid, "sample_cid": sample.identity.cid, "artifact_cid": artifact.identity.cid,
        "declaration_schema": declaration.schema_version, "producer": adapter.producer_id,
        "contract_cids": sorted(contract_ids), "modeled_paths": sorted(supported_paths),
        "unmodeled_paths": unmodeled_paths, "unsupported_analyses": unsupported,
        "omitted_source_count": observed["omitted_source_count"], "open_frontiers": sorted(frontiers),
        "source_count": len(sources), "normalizer_count": len(assets), "claim_count": len(claims),
        "formula_count": len(artifact.formulas), "obligation_count": len(artifact.proof_obligations),
        "provider_calls": 0, "solver_executed": False, "proof_created": False,
        "whole_program_proved": False, "mutation_authority": False, "completion_authority": False}
    scoped.assert_current()
    if output is not None:
        output = Path(output).absolute()
        if output.resolve() != output or output.is_relative_to(scoped.repository) or output.exists():
            raise ValueError("SecurityIR artifacts require a fresh external non-symlink directory")
        output.mkdir(parents=True, mode=0o700)
        artifacts = {}
        for name, payload in (("declaration.json", declaration.to_dict()),
                              ("sample.json", sample.to_dict()), ("formalization.json", artifact.to_dict())):
            raw = canonical_json(payload).encode()
            (output / name).write_bytes(raw)
            artifacts[name] = {"path": str(output / name), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        report["artifacts"] = artifacts
        scoped.assert_current()
    report["compilation_cid"] = content_identity(report)
    if output is not None:
        (output / "result.json").write_text(canonical_json(report) + "\n")
    return CompiledHeaderSecurityIR(declaration, sample, artifact, canonical_json(report))
