"""Explicit composition of native Doctor planning, proof, synthesis and impact.

This boundary accepts reviewed, typed operator inputs, never a serialized claim
that a proof succeeded. It executes the pinned hammer and independent kernel,
then passes the resulting sealed authority to the native synthesizer. Impact
must be recomputed from a current program graph. No stage result grants task
completion, and preview never writes the checkout.

The theorem's stated scope is retained. A theorem about byte replacement is
not silently promoted to whole-program correctness. Expectation extraction and
theorem review belong to the closed operator that prepares these inputs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding,
    DoctorAuthorityRoots,
    DoctorEvidenceSnapshot,
    DoctorOperatorKind,
)
from ..analysis.deterministic_doctor_impact import (
    DeterministicDoctorImpactAnalyzer,
    DoctorImpactRequest,
    DoctorPlanCompilationRequest,
    compile_deterministic_doctor_plan,
)
from ..planning.deterministic_doctor_synthesis import (
    DeterministicDoctorSynthesizer,
    DoctorSynthesisRequest,
)
from ..planning.deterministic_doctor_tactician import DeterministicDoctorTactician
from ..proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer,
    DoctorExactLoweringReceipt,
    DoctorPinnedExecutable,
    DoctorReviewedTheorem,
)
from ..proof.formal_verification_contracts import content_identity
from .doctor_source_partition import DoctorSourcePartition
from .doctor_alias_contract import ImportedAliasRepair, ImportedAliasContractError


class DoctorCompositionError(ValueError):
    """A composition input is stale, unbound or attempts to skip a native gate."""


def _signature(function: ast.FunctionDef) -> inspect.Signature:
    args = function.args
    if args.vararg or args.kwarg or args.posonlyargs:
        raise DoctorCompositionError("variadic or positional-only targets need another validator")
    parameters = []
    for index, arg in enumerate(args.args):
        default = None if index >= len(args.args) - len(args.defaults) else inspect.Parameter.empty
        parameters.append(
            inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None,
            )
        )
    return inspect.Signature(parameters)


def _inert_function_module(file_text: str) -> tuple[ast.Module, list[ast.FunctionDef]]:
    """The module-shape gate shared by live composition and eligibility review."""
    tree = ast.parse(file_text)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(definitions) != len(tree.body) or any(node.decorator_list for node in definitions):
        raise DoctorCompositionError("keyword composition requires an inert function-only module")
    return tree, definitions


def _keyword_rename_contract(request: DoctorSynthesisRequest, subject: str) -> str:
    """Validate the currently supported closed operator's exact AST placement.

    Mutation composition currently admits only a keyword rename on a direct
    call to a unique module-local function. Other operator families need their
    own exact impact binding and candidate validator before using this route.
    """
    proposal = request.proposal
    if proposal.kind is not DoctorOperatorKind.EXACT_RENAME:
        raise DoctorCompositionError("operator has no bound composition validator")
    tree, definitions = _inert_function_module(request.file_text)
    declared = [node for node in definitions if node.name == subject]
    if len(declared) != 1:
        raise DoctorCompositionError("impact subject is not one exact local declaration")
    if any(
        isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Import, ast.ImportFrom))
        or (isinstance(node, ast.FunctionDef) and node not in definitions)
        or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == subject)
        or (isinstance(node, ast.arg) and node.arg == subject)
        for node in ast.walk(tree)
    ):
        raise DoctorCompositionError("declaration has unsupported rebinding or dynamic scope")
    lines = request.file_text.splitlines(keepends=True)
    site = proposal.edit_site
    matches = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            offset = len("".join(lines[: keyword.lineno - 1])) + len(
                lines[keyword.lineno - 1].encode()[: keyword.col_offset].decode()
            )
            if offset == site.span_start and offset + len(keyword.arg) == site.span_end:
                matches.append((call, keyword))
    if len(matches) != 1:
        raise DoctorCompositionError("edit site is not one exact keyword identifier")
    call, keyword = matches[0]
    if not isinstance(call.func, ast.Name) or call.func.id != subject:
        raise DoctorCompositionError("impact subject differs from the edited call target")
    if keyword.arg != proposal.previous_parameter_name or request.span_text != keyword.arg:
        raise DoctorCompositionError("keyword preimage differs from the proposed rename")
    if any(item.arg is None for item in call.keywords) or any(
        isinstance(arg, ast.Starred) for arg in call.args
    ):
        raise DoctorCompositionError("expanded call arguments require separate binding evidence")
    signature = _signature(declared[0])
    before_keywords = {item.arg: None for item in call.keywords}
    if len(before_keywords) != len(call.keywords):
        raise DoctorCompositionError("duplicate keyword arguments are not admitted")
    try:
        signature.bind(*([None] * len(call.args)), **before_keywords)
    except TypeError:
        pass
    else:
        raise DoctorCompositionError("original call already satisfies the declared contract")
    keyword.arg = proposal.parameter_name
    after_keywords = {item.arg: None for item in call.keywords}
    if len(after_keywords) != len(call.keywords):
        raise DoctorCompositionError("rename introduces a duplicate keyword")
    try:
        signature.bind(*([None] * len(call.args)), **after_keywords)
    except TypeError as exc:
        raise DoctorCompositionError(
            "renamed keyword does not satisfy the declared signature"
        ) from exc
    after = (
        request.file_text[: site.span_start]
        + proposal.parameter_name
        + request.file_text[site.span_end :]
    )
    if ast.dump(tree) != ast.dump(ast.parse(after)):
        raise DoctorCompositionError("rename changes syntax outside the selected keyword")
    return after


def operator_contract_reconstruction(request, subject, *, alias_contract=None):
    """Reconstruct one closed typed contract; serialized proof claims are ignored."""
    if alias_contract is None:
        return _keyword_rename_contract(request, subject)
    if type(alias_contract) is not ImportedAliasRepair:
        raise DoctorCompositionError("exact imported-alias contract required")
    try:
        return alias_contract.reconstruct(request, subject)
    except ImportedAliasContractError as error:
        raise DoctorCompositionError("imported alias contract does not reconstruct") from error



def assess_doctor_repair_eligibility(
    *,
    repository: Path,
    admission: Mapping[str, Any],
    diagnostic_artifact: Path,
    paths: Sequence[str],
) -> dict[str, Any]:
    """Assess the closed repair operator against real, independently bound input.

    This is a no-effect eligibility observation, not a plan or proof receipt.
    It recomputes the nominated native diagnostics and reads the independently
    signed output scope. A compatible source shape alone cannot authorize a
    repair: callers still need explicit reviewed typed operator, theorem,
    lowering, toolchain and impact inputs through ``DoctorCompositionInputs``.
    """
    from .local_planning_admission import verify_local_benchmark_admission
    from ..analysis.doctor_repository_diagnostics import (
        DoctorAuthorityRoots as DiagnosticRoots, DoctorSourceUnit, diagnose_repository,
    )
    from ..analysis.doctor_contract_adapters import materialize_runtime_diagnostics
    from ..semantic_state.wire import cid_for_payload
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    root = Path(repository).resolve(strict=True)
    verified = verify_local_benchmark_admission(admission, initial=True)
    manifest = verified["manifest"]
    if root != Path(manifest["repository"]):
        raise DoctorCompositionError("repair eligibility repository differs from signed admission")
    if isinstance(paths, (str, bytes)):
        raise DoctorCompositionError("repair eligibility requires an exact source path sequence")
    selected = sorted(paths)
    if (not selected or len(selected) > 256 or len(set(selected)) != len(selected)
            or not set(selected).issubset(manifest["sources"])):
        raise DoctorCompositionError("repair eligibility source scope differs from signed inputs")
    sources = {}
    for name in selected:
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != name
                or path.resolve(strict=True) != path or not path.is_file()):
            raise DoctorCompositionError("repair eligibility source escapes repository")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sources"][name]["sha256"]:
            raise DoctorCompositionError("repair eligibility source changed during assessment")
        sources[name] = raw
    scope_sources = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "source_cid": cid_for_bytes(raw)}
        for name, raw in sources.items()
    }
    scope_cid = cid_for_payload({"schema": "supervisor-source-scope@1", "sources": scope_sources})
    repository_id = cid_for_payload({"repository": str(root)})
    roots = DiagnosticRoots(
        repository_id=repository_id, forest_id=scope_cid, tree_id=scope_cid,
        overlay_id=scope_cid, file_root_id=scope_cid, blob_root_id=scope_cid,
        config_id=content_identity({"paths": selected}),
        policy_id=content_identity({"mode": "context_preparation_only"}),
    )
    diagnostic = diagnose_repository([
        DoctorSourceUnit(path=name, source_bytes=raw, blob_identity=scope_sources[name]["source_cid"])
        for name, raw in sources.items()
    ], authority_roots=roots)
    snapshot, findings, bridge_cid = materialize_runtime_diagnostics(
        diagnostic, require_repository_id=repository_id,
    )
    expected = {"snapshot": snapshot.to_dict(), "findings": [item.to_dict() for item in findings],
                "manifest_cid": bridge_cid}
    artifact = Path(diagnostic_artifact).absolute()
    if (artifact.resolve(strict=True) != artifact or not artifact.is_relative_to(root)
            or not artifact.is_file() or artifact.stat().st_size > 4_000_000):
        raise DoctorCompositionError("repair eligibility diagnostic artifact is outside its bound")
    raw_artifact = artifact.read_bytes()
    if len(raw_artifact) > 4_000_000:
        raise DoctorCompositionError("repair eligibility diagnostic artifact exceeds bound")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DoctorCompositionError("duplicate diagnostic artifact field")
            result[key] = value
        return result

    nominated = json.loads(raw_artifact, object_pairs_hook=unique)
    if content_identity(nominated) != content_identity(expected):
        raise DoctorCompositionError("repair eligibility diagnostics differ from native source replay")
    allowed_outputs = sorted(
        ({"task_key": task["task_key"], **output}
         for task in manifest["tasks"] for output in task["outputs"]),
        key=lambda item: (item["task_key"], item["path"]),
    )
    assessments = []
    for output in allowed_outputs:
        name = output["path"]
        reasons = []
        if output["effect"] != "modify":
            reasons.append("operator_requires_existing_source_modification")
        elif name not in sources:
            reasons.append("output_not_in_verified_diagnostic_scope")
        elif PurePosixPath(name).suffix != ".py":
            reasons.append("operator_requires_python_source")
        else:
            try:
                _inert_function_module(sources[name].decode("utf-8"))
            except (DoctorCompositionError, SyntaxError, UnicodeError):
                reasons.append("unsupported_module_shape")
        assessments.append({"task_key": output["task_key"], "path": name,
                            "module_shape_eligible": not reasons,
                            "reason_codes": reasons})
    # Re-read exact files and the signed native admission before publishing the
    # observation. No status, checkout, source or task contract is modified.
    if any((root / name).read_bytes() != raw for name, raw in sources.items()):
        raise DoctorCompositionError("repair eligibility source changed during assessment")
    current = verify_local_benchmark_admission(admission, initial=True)
    if current["current_source_tree_id"] != verified["current_source_tree_id"]:
        raise DoctorCompositionError("repair eligibility source tree changed during assessment")
    report = {
        "schema": "supervisor-doctor-repair-eligibility@1", "status": "abstained",
        "automatic_repair_eligible": False, "diagnostics_replayed": True,
        "supported_operator": "exact_keyword_rename_in_inert_function_module",
        "manifest_cid": verified["receipt"]["manifest_cid"],
        "source_tree_id": verified["current_source_tree_id"], "scope_cid": scope_cid,
        "source_hashes": {name: item["sha256"] for name, item in scope_sources.items()},
        "diagnostic_artifact_sha256": hashlib.sha256(raw_artifact).hexdigest(),
        "diagnostic_manifest_cid": bridge_cid, "diagnostic_snapshot_id": snapshot.snapshot_id,
        "findings_count": len(findings), "finding_ids": [item.finding_id for item in findings],
        "allowed_outputs": allowed_outputs, "output_assessments": assessments,
        "reason_codes": sorted({"reviewed_typed_operator_and_proof_inputs_unavailable",
                                 *(reason for item in assessments for reason in item["reason_codes"])}),
        "stages": {name: "not_run" for name in ("planning", "proof", "synthesis", "impact", "transaction")},
        "source_edits": 0, "provider_calls": 0, "execution_authority": False,
        "completion_authority": False,
    }
    return {**report, "report_cid": content_identity(report)}


_SOURCE_ROOTS = (
    "repository_id",
    "forest_id",
    "tree_id",
    "overlay_id",
    "file_root_id",
    "ast_root_id",
    "corpus_id",
    "index_id",
)
_PROOF_ROOTS = (
    "repository_id",
    "forest_id",
    "tree_id",
    "overlay_id",
    "graph_id",
    "corpus_id",
    "index_id",
    "model_id",
    "translator_id",
    "toolchain_id",
    "policy_id",
    "environment_id",
)


def composition_snapshot(
    observed: DoctorEvidenceSnapshot,
    roots: DoctorAuthorityRoots,
) -> DoctorEvidenceSnapshot:
    """Bind actual stage roots while preserving every observed source root.

    Diagnostic adapters have placeholders for operational roots. Replacing
    those requires explicit typed inputs and creates a new snapshot identity;
    it cannot relabel a different checkout as the observed one.
    """
    if any(getattr(observed.roots, key) != getattr(roots, key) for key in _SOURCE_ROOTS):
        raise DoctorCompositionError("composition source roots differ from observed snapshot")
    return replace(
        observed,
        roots=roots,
        snapshot_id=content_identity(
            {
                "schema": "doctor-composition-snapshot@1",
                "observed": observed.content_id,
                "stage_roots": roots.content_id,
            }
        ),
    )


def operator_consequence_ref(
    request: DoctorSynthesisRequest,
    expected_after_hash: str,
    expected_behavior_refs: tuple[str, ...],
) -> str:
    """Bind the reviewed consequence to one exact proposed edit and contract.

    Proof receipt IDs and the admission bit are excluded to avoid circular
    identities. Bodies are represented by hashes, including the full preimage.
    """
    proposal = replace(request.proposal, proof_admitted=False, proof_refs=())
    return content_identity(
        {
            "schema": "doctor-reviewed-operator-consequence@1",
            "proposal": proposal.content_id,
            "span_sha256": hashlib.sha256(request.span_text.encode()).hexdigest(),
            "file_sha256": hashlib.sha256(request.file_text.encode()).hexdigest(),
            "expression_sha256": hashlib.sha256(request.expression_text.encode()).hexdigest(),
            "fields": dict(request.field_mappings)
            if isinstance(request.field_mappings, Mapping)
            else [item.to_dict() for item in request.field_mappings],
            "artifact_sha256": hashlib.sha256(request.verified_artifact_bytes).hexdigest()
            if request.verified_artifact_bytes is not None
            else "",
            "expected_after_hash": expected_after_hash,
            "expected_behavior_refs": list(expected_behavior_refs),
            "value_ref": request.value_ref,
            "placement_ref": request.placement_ref,
        }
    )


@dataclass(frozen=True)
class DoctorCompositionInputs:
    """Trusted local composition inputs; deliberately not JSON-deserializable."""

    evidence_id: str
    snapshot: DoctorEvidenceSnapshot
    finding: DeterministicDoctorFinding
    synthesis: DoctorSynthesisRequest
    expected_after_hash: str
    theorem: DoctorReviewedTheorem
    lowering: DoctorExactLoweringReceipt
    solver_pin: DoctorPinnedExecutable
    kernel_pin: DoctorPinnedExecutable
    hammer: DeterministicDoctorHammer
    impact: DoctorImpactRequest
    program_graph: Any
    source_hashes: Mapping[str, str]
    proof_scope: str
    candidates: tuple[Any, ...] = ()
    worktree_adapter: Any = None
    target_ref: str = ""
    base_ref: str = ""
    source_partition: DoctorSourcePartition | None = None
    alias_contract: ImportedAliasRepair | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("snapshot", DoctorEvidenceSnapshot),
            ("finding", DeterministicDoctorFinding),
            ("synthesis", DoctorSynthesisRequest),
            ("theorem", DoctorReviewedTheorem),
            ("lowering", DoctorExactLoweringReceipt),
            ("solver_pin", DoctorPinnedExecutable),
            ("kernel_pin", DoctorPinnedExecutable),
            ("hammer", DeterministicDoctorHammer),
            ("impact", DoctorImpactRequest),
        ):
            if type(getattr(self, name)) is not cls:
                raise DoctorCompositionError(f"{name} must be a concrete {cls.__name__}")
        roots = self.snapshot.roots
        if not self.evidence_id or not self.proof_scope.strip():
            raise DoctorCompositionError("observed evidence identity and proof scope are required")
        if self.finding.snapshot_id != self.snapshot.snapshot_id:
            raise DoctorCompositionError("finding is not bound to the composition snapshot")
        if any(
            item != roots
            for item in (
                self.finding.roots,
                self.synthesis.roots,
                self.impact.roots,
            )
        ):
            raise DoctorCompositionError("stage roots do not match")
        if any(getattr(self.theorem.roots, key) != getattr(roots, key) for key in _PROOF_ROOTS):
            raise DoctorCompositionError("theorem roots do not match")
        if not self.finding.expected_behavior_refs:
            raise DoctorCompositionError("an independent expected contract is required")
        expected = operator_consequence_ref(
            self.synthesis,
            self.expected_after_hash,
            self.finding.expected_behavior_refs,
        )
        if self.theorem.consequence_ref != expected:
            raise DoctorCompositionError("theorem is not bound to the exact operator consequence")
        if not set(self.finding.expected_behavior_refs) <= set(self.theorem.premise_ids):
            raise DoctorCompositionError("theorem omits the independent contract premises")
        self.lowering.verify_theorem(self.theorem)
        if self.synthesis.proof_receipt is not None or not self.synthesis.require_proof_receipt:
            raise DoctorCompositionError("proof authority must be produced by this composition")
        if not self.synthesis.require_idempotent_replay or self.synthesis.already_applied:
            raise DoctorCompositionError("composition requires a fresh replay-checked synthesis")
        if self.impact.impact_closure is not None:
            raise DoctorCompositionError("impact closure must be freshly computed from the graph")
        if self.impact.base_delta is not None or self.impact.candidate_delta is not None:
            raise DoctorCompositionError("composition derives its own exact operator delta")
        operator_contract_reconstruction(self.synthesis, self.impact.subject_symbol_id,
                                         alias_contract=self.alias_contract)
        if getattr(self.program_graph, "graph_id", None) != roots.graph_id:
            raise DoctorCompositionError("program graph does not match the graph root")
        graph_roots = getattr(self.program_graph, "roots", None)
        if graph_roots is None or any(
            getattr(graph_roots, key) != getattr(roots, key)
            for key in ("forest_id", "tree_id", "overlay_id")
        ):
            raise DoctorCompositionError("impact needs the current dependency-graph contract")
        hashes = dict(self.source_hashes)
        if not hashes or len(hashes) > 256:
            raise DoctorCompositionError("source preimages must contain 1 to 256 paths")
        support = set()
        program_paths = set(hashes)
        if self.source_partition is not None:
            if type(self.source_partition) is not DoctorSourcePartition or self.worktree_adapter is None:
                raise DoctorCompositionError("typed replayable source partition and adapter required")
            self.source_partition.assert_current(self.worktree_adapter.repository_root)
            support = {path for path, _, _ in self.source_partition.support_hashes}
            program_paths = set(self.source_partition.program_paths)
            if support & program_paths or support | program_paths != set(hashes):
                raise DoctorCompositionError("partition must cover the complete source ledger")
        if self.alias_contract is not None:
            contract = self.alias_contract.to_dict()
            if (contract["source_hashes"] != {name: hashes[name] for name in sorted(program_paths)}
                    or self.alias_contract.contract_id not in self.finding.expected_behavior_refs):
                raise DoctorCompositionError("alias contract omits independently bound program sources or premises")
        for path, digest in hashes.items():
            pure = PurePosixPath(path)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or str(pure) != path
                or (pure.suffix != ".py" and path not in support)
            ):
                raise DoctorCompositionError("source path escapes the checkout")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise DoctorCompositionError("source hashes must be SHA256 hex digests")
        path = self.synthesis.proposal.edit_site.path
        full_text = self.synthesis.file_text
        if path not in program_paths or hashlib.sha256(full_text.encode()).hexdigest() != hashes[path]:
            raise DoctorCompositionError("synthesis full preimage is not in the source ledger")
        site = self.synthesis.proposal.edit_site
        if full_text[site.span_start : site.span_end] != self.synthesis.span_text:
            raise DoctorCompositionError("edit span differs from the full source preimage")
        object.__setattr__(self, "source_hashes", MappingProxyType(hashes))
        if set(graph_roots.included_roots) != program_paths:
            raise DoctorCompositionError("graph coverage must name every admitted program input")
        if bool(self.target_ref) != (self.worktree_adapter is not None):
            raise DoctorCompositionError(
                "transaction needs both an adapter and explicit target ref"
            )

    def assert_current(self, checkout_root: Path, evidence: Any) -> None:
        if evidence.evidence_id != self.evidence_id:
            raise DoctorCompositionError("observed evidence changed; regenerate the composition")
        expected = composition_snapshot(evidence.snapshot, self.snapshot.roots)
        if expected.content_id != self.snapshot.content_id:
            raise DoctorCompositionError("composition snapshot changed")
        if "diagnostic_source_bound_reached" in evidence.notes:
            raise DoctorCompositionError("bounded diagnostic coverage cannot authorize composition")
        program_paths = set(self.source_hashes)
        support_kinds = {}
        if self.source_partition is not None:
            self.source_partition.assert_current(checkout_root)
            program_paths = set(self.source_partition.program_paths)
            support_kinds = {
                name: {"instruction": "text_reference", "task_profile": "structured_data",
                       "structural_smoke": "semantic_ast"}[role]
                for name, role, _ in self.source_partition.support_hashes
            }
        if any(row.coverage_kind != support_kinds.get(row.path, "semantic_ast")
               for row in evidence.source_inventory):
            raise DoctorCompositionError(
                "opaque or non-semantic admitted inputs require frontier handling"
            )
        inventory = {
            row.path: row.content_digest.removeprefix("sha256:")
            for row in evidence.source_inventory
        }
        # A graph's omitted input is a stale-evidence risk even if the edited
        # file did not change. Bind the complete native diagnostic input set.
        if len(inventory) != len(evidence.source_inventory) or set(inventory) != set(self.source_hashes):
            raise DoctorCompositionError("source ledger omits admitted inventory inputs")
        for path, digest in self.source_hashes.items():
            file = checkout_root / path
            if inventory.get(path) != digest:
                raise DoctorCompositionError("source ledger differs from runtime inventory")
            if file.is_symlink() or not file.resolve().is_relative_to(checkout_root):
                raise DoctorCompositionError("source path crosses a symlink boundary")
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise DoctorCompositionError("source preimage changed; regenerate the composition")
        from ..analysis.program_dependency_graph import PathSource, ProgramDependencyGraph

        # Rebuild through the native deterministic source producer. Caller
        # completeness flags, nominated edges and custom graph annotations
        # cannot close an impact frontier through this composition route.
        rebuilt = ProgramDependencyGraph(self.program_graph.roots).build(
            [
                PathSource(
                    path=path,
                    source=(checkout_root / path).read_text(encoding="utf-8"),
                    language="python",
                )
                for path in sorted(program_paths)
            ]
        )
        if rebuilt.graph_id != self.program_graph.graph_id:
            raise DoctorCompositionError("dependency graph does not replay from the source ledger")


@dataclass(frozen=True)
class DoctorCompositionResult:
    inputs: DoctorCompositionInputs
    tactician: Any
    proof: Any = None
    synthesis: Any = None
    impact: Any = None
    compilation: Any = None
    stages: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return bool(
            self.compilation is not None
            and self.compilation.may_mutate
            and self.synthesis is not None
            and self.synthesis.mutation_capable
        )

    @property
    def plan(self) -> Any:
        return self.compilation.plan if self.compilation is not None else None


def composition_step_validator(composed: DoctorCompositionResult) -> Any:
    """Build the closed operator's actual static validator for live worktrees.

    It reconstructs the edit from the preimage, compares the complete candidate
    tree scope, parses candidate syntax and independently checks every direct
    local call against its declaration. It executes no target code. This is a
    local signature fixed point, not a general task-completion certificate.
    """
    from ..planning.deterministic_doctor_transaction import (
        DoctorStepApplyResult,
        DoctorStepDisposition,
    )

    inputs = composed.inputs

    def validate(session: Any, plan: Any, step: Any) -> DoctorStepApplyResult:
        try:
            if plan != composed.plan or step not in plan.steps:
                raise DoctorCompositionError("validator plan or step mismatch")
            expected = operator_contract_reconstruction(inputs.synthesis, inputs.impact.subject_symbol_id,
                                                        alias_contract=inputs.alias_contract)
            path = inputs.synthesis.proposal.edit_site.path
            candidate = (session.worktree_root / path).read_bytes()
            if candidate != expected.encode():
                raise DoctorCompositionError("candidate differs from the reconstructed exact edit")
            for other, digest in inputs.source_hashes.items():
                if (
                    other != path
                    and hashlib.sha256((session.worktree_root / other).read_bytes()).hexdigest()
                    != digest
                ):
                    raise DoctorCompositionError("candidate changed another admitted input")
            if inputs.alias_contract is not None:
                inputs.alias_contract.validate_candidate(candidate.decode("utf-8"))
            else:
                tree = ast.parse(candidate)
                definitions = {
                    node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                    if (
                        not isinstance(call.func, ast.Name)
                        or call.func.id not in definitions
                        or any(item.arg is None for item in call.keywords)
                        or any(isinstance(arg, ast.Starred) for arg in call.args)
                    ):
                        raise DoctorCompositionError("candidate contains an unsupported call frontier")
                    keywords = {item.arg: None for item in call.keywords}
                    if len(keywords) != len(call.keywords):
                        raise DoctorCompositionError("candidate has duplicate keywords")
                    _signature(definitions[call.func.id]).bind(*([None] * len(call.args)), **keywords)
            receipts = tuple(
                content_identity(
                    {
                        "schema": "doctor-static-candidate-validation@1",
                        "step": step.content_id,
                        "obligation": ref,
                        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
                        "checks": [
                            "exact_reconstruction",
                            "unrelated_inputs_unchanged",
                            "ast_parse",
                            "imported_alias_signature_residual_free" if inputs.alias_contract is not None
                            else "local_signature_residual_free",
                        ],
                        "task_completion_authorized": False,
                    }
                )
                for ref in (step.validation_refs or (step.step_id,))
            )
            return DoctorStepApplyResult(
                disposition=DoctorStepDisposition.PASSED,
                diagnostic_refs=receipts,
                static_replay=True,
            )
        except (DoctorCompositionError, ImportedAliasContractError, SyntaxError, TypeError, OSError) as exc:
            return DoctorStepApplyResult(
                disposition=DoctorStepDisposition.FAILED,
                reason_codes=("composition_candidate_validation_failed",),
                diagnostic_refs=(
                    content_identity(
                        {"failure": type(exc).__name__, "detail": str(exc), "step": step.step_id}
                    ),
                ),
                static_replay=True,
            )

    return validate


def _single_file_execution_plan(compilation: Any, path: str, file_text: str) -> Any:
    """Give one whole-file edit one owner while retaining all consumer coverage.

    Native impact compilation emits a step per consumer SCC. Several SCCs
    can occupy the same file; applying a whole-file replacement once per SCC
    would violate its preimage. Group those writes atomically, preserving all
    validation dependencies and the original impact receipt.
    """
    if not compilation.may_mutate:
        return compilation
    plan = compilation.plan
    writers = tuple(step for step in plan.steps if step.write_paths)
    if not writers or any(step.write_paths != (path,) for step in writers):
        raise DoctorCompositionError("single-file composition cannot cover the compiled write set")
    writer_ids = {step.step_id for step in writers}
    if any(set(step.dependency_step_ids) - writer_ids for step in writers):
        raise DoctorCompositionError("cannot coalesce writes across an intervening read-only gate")
    step_id = content_identity(
        {"schema": "doctor-atomic-file-step@1", "plan": plan.content_id, "path": path}
    )
    # Synthesis binds a source span; the durable worktree protocol binds a
    # complete file. Preserve the original span in the synthesis receipt and
    # give the execution plan the exact whole-file preimage that it checks.
    site = replace(
        plan.edit_sites[0],
        before_hash="sha256:" + hashlib.sha256(file_text.encode()).hexdigest(),
        span_start=0,
        span_end=len(file_text),
        artifact_id="sha256:" + hashlib.sha256(file_text.encode()).hexdigest(),
    )
    merged = replace(
        writers[0],
        step_id=step_id,
        dependency_step_ids=(),
        consumer_ids=tuple(sorted({cid for step in writers for cid in step.consumer_ids})),
        edit_site_refs=(site.content_id,),
        validation_refs=tuple(sorted({ref for step in writers for ref in step.validation_refs})),
    )
    readers = tuple(
        replace(
            step,
            dependency_step_ids=tuple(
                dict.fromkeys(
                    step_id if dep in writer_ids else dep for dep in step.dependency_step_ids
                )
            ),
        )
        for step in plan.steps
        if not step.write_paths
    )
    updated = replace(
        plan,
        steps=(merged, *readers),
        edit_sites=(site,),
        plan_id=content_identity(
            {
                "schema": "doctor-atomic-file-plan@1",
                "compiled_plan": plan.content_id,
                "steps": [step.content_id for step in (merged, *readers)],
                "whole_file_site": site.content_id,
            }
        ),
    )
    return replace(
        compilation,
        plan=updated,
        plan_id=updated.plan_id,
        scc_step_ids=(step_id,),
        producer_id="doctor-repair-composition@1",
        reason_codes=(
            *compilation.reason_codes,
            "single_file_atomic_group",
            "whole_file_preimage_binding",
        ),
    )


def compose_doctor_repair(inputs: DoctorCompositionInputs) -> DoctorCompositionResult:
    """Execute the native gate chain, stopping at its first abstention."""
    if type(inputs) is not DoctorCompositionInputs:
        raise DoctorCompositionError("composition requires typed inputs")
    stages: dict[str, Mapping[str, Any]] = {}

    def record(stage: str, receipt: Any) -> None:
        stages[stage] = {
            "status": "executed",
            "receipt_id": receipt.content_id,
            "disposition": receipt.disposition.value,
            "reason_codes": list(receipt.reason_codes),
        }

    plan = DeterministicDoctorTactician().plan_finding(
        inputs.finding,
        snapshot=inputs.snapshot,
        current_roots=inputs.snapshot.roots,
        candidates=inputs.candidates,
    )
    record("tactician", plan)
    result = DoctorCompositionResult(inputs, plan)
    if not plan.is_planned:
        return replace(result, stages=stages)
    proof = inputs.hammer.verify_authoritative(
        inputs.theorem,
        inputs.lowering,
        solver_pin=inputs.solver_pin,
        kernel_pin=inputs.kernel_pin,
        current_roots=inputs.theorem.roots,
        eligible_consequence_refs=(inputs.theorem.consequence_ref,),
    )
    if proof.mutation_capable:
        proof = inputs.hammer.reverify_authoritative(proof, current_roots=inputs.theorem.roots)
    record("proof", proof)
    stages["proof"] = {
        **stages["proof"],
        "scope": inputs.proof_scope,
        "mutation_capable": proof.mutation_capable,
    }
    result = replace(result, proof=proof)
    if not proof.mutation_capable:
        return replace(result, stages=stages)
    proposal = replace(
        inputs.synthesis.proposal, proof_admitted=True, proof_refs=(proof.content_id,)
    )
    request = replace(
        inputs.synthesis,
        proposal=proposal,
        proof_receipt=proof,
        selected_consequence_ref=proof.selected_consequence_ref,
        finding_id=inputs.finding.finding_id,
        plan_receipt_id=plan.receipt_id,
        proof_receipt_id=proof.receipt_id,
    )
    synthesis = DeterministicDoctorSynthesizer(roots=request.roots).synthesize(request)
    record("synthesis_preview", synthesis)
    result = replace(result, synthesis=synthesis)
    if not synthesis.mutation_capable:
        return replace(result, stages=stages)
    overlay = synthesis.overlay
    if overlay.after_hash != inputs.expected_after_hash:
        raise DoctorCompositionError("rendered overlay differs from the reviewed consequence")
    impact_request = replace(
        inputs.impact,
        overlay_id=overlay.overlay_id,
        overlay_path=overlay.path,
        overlay_patch_cid=overlay.patch_cid,
        overlay_before_hash=overlay.before_hash,
        overlay_after_hash=overlay.after_hash,
        proof_refs=(proof.content_id,),
        require_authoritative_closure=True,
        current_graph_cid=inputs.snapshot.roots.graph_id,
        current_index_cid=inputs.snapshot.roots.index_id,
        current_ast_cid=inputs.snapshot.roots.ast_root_id,
    )
    impact = DeterministicDoctorImpactAnalyzer().analyze(
        impact_request,
        program_graph=inputs.program_graph,
    )
    stages["impact"] = {
        "status": "executed",
        "receipt_id": impact.content_id,
        "mutation_admissible": impact.mutation_admissible,
        "reason_codes": list(impact.reason_codes),
    }
    compilation = compile_deterministic_doctor_plan(
        DoctorPlanCompilationRequest(
            roots=inputs.snapshot.roots,
            closure=impact,
            snapshot_id=inputs.snapshot.snapshot_id,
            finding_ids=(inputs.finding.finding_id,),
            selected_operator_id=proposal.operator_id,
            target_ref=impact_request.subject_symbol_id,
            value_source_ref=request.value_ref,
            placement_ref=request.placement_ref,
            proof_refs=(proof.content_id,),
            edit_sites=(proposal.edit_site,),
            permitted_read_paths=tuple(inputs.source_hashes),
            permitted_write_paths=(overlay.path,),
            forbidden_paths=overlay.forbidden_paths,
            lease_id=inputs.snapshot.roots.lease_id,
            tactician_plan_ref=plan.receipt_id,
            operator_ids=(proposal.operator_id,),
            invalidation_refs=inputs.snapshot.invalidation_refs,
        )
    )
    compilation = _single_file_execution_plan(compilation, overlay.path, request.file_text)
    record("plan_compilation", compilation)
    return replace(result, impact=impact, compilation=compilation, stages=stages)
