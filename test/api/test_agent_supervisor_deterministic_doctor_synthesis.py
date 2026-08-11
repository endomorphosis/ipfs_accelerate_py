"""Fail-closed coverage for proof-admitted doctor overlay synthesis (LPR-036)."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.analytical_change_transforms import (
    FieldMapping,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
    DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
    PRODUCER_ID,
    DeterministicDoctorSynthesizer,
    DoctorAnalyticalOverlay,
    DoctorSynthesisAuthorityError,
    DoctorSynthesisDisposition,
    DoctorSynthesisReason,
    DoctorSynthesisReceipt,
    DoctorSynthesisRequest,
    create_deterministic_doctor_synthesizer,
    materialize_proof_admitted_overlay,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    DoctorOperatorProposal,
    DoctorRepairOperatorRegistry,
    build_default_doctor_operator_registry,
    make_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def registry(auth: DoctorAuthorityRoots | None = None) -> DoctorRepairOperatorRegistry:
    return build_default_doctor_operator_registry(auth or roots())


def mapping(
    *,
    disposition: SynthesisDisposition = SynthesisDisposition.UNIQUE_PROVED,
    expression_ref: str = "expr:ctx",
    proved: tuple[str, ...] | None = None,
    repository_id: str = "repository:fixture",
    tree_id: str = "tree:fixture",
) -> ValueMappingProof:
    if proved is None:
        if disposition is SynthesisDisposition.UNIQUE_PROVED:
            proved = ("candidate:ctx",)
        elif disposition is SynthesisDisposition.AMBIGUOUS:
            proved = ("candidate:a", "candidate:b")
        else:
            proved = ()
    return ValueMappingProof(
        requirement_id="missing:context",
        consumer_id="consumer:one",
        disposition=disposition,
        facet_results=(),
        proved_candidate_ids=proved,
        refuted_candidate_ids=()
        if disposition is not SynthesisDisposition.REFUTED
        else ("candidate:bad",),
        expression_ref=expression_ref,
        type_ref="type:Context",
        repository_id=repository_id,
        tree_id=tree_id,
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        reason_codes=(
            ("unique_source",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ("non_unique",)
        ),
    )


def proof_receipt(
    *,
    admitted: bool = True,
    unique: bool = True,
    consequence: str = "consequence:unique-repair",
    eligible: tuple[str, ...] | None = None,
    finding_id: str = "finding:one",
    plan_receipt_id: str = "plan:one",
    receipt_id: str = "proof:one",
    llm_invocation_count: int = 0,
    model_provider_call_count: int = 0,
    write_authority: bool = False,
    repository_id: str = "repository:fixture",
    tree_id: str = "tree:fixture",
) -> dict[str, object]:
    if eligible is None:
        eligible = (consequence,) if unique else (consequence, "consequence:other")
    return {
        "disposition": "admitted" if admitted else "abstained",
        "uniqueness_satisfied": unique and admitted,
        "selected_consequence_ref": consequence if admitted else "",
        "eligible_consequence_refs": list(eligible),
        "finding_id": finding_id,
        "plan_receipt_id": plan_receipt_id,
        "receipt_id": receipt_id,
        "llm_invocation_count": llm_invocation_count,
        "model_provider_call_count": model_provider_call_count,
        "write_authority": write_authority,
        "roots": {
            "repository_id": repository_id,
            "tree_id": tree_id,
        },
    }


def repair_decision(
    path: str = "pkg/caller.py",
    *,
    admitted: bool = True,
) -> RepairTargetDecision:
    repair_roots = AuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        config_id="config:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    candidate = RepairCandidate(
        repair_roots,
        "trace:one",
        RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan(path, 0, 12, "blob:one"),
        (EvidenceReference("candidate", "candidate:one", producer_id="test"),),
    )
    candidates = (candidate,)
    return RepairTargetDecision(
        roots=repair_roots,
        candidates=candidates,
        candidate_set_id=candidate_set_identity(candidates),
        disposition=DecisionDisposition.ADMITTED if admitted else DecisionDisposition.ABSTAINED,
        strategy=RepairStrategy.NEW_IMPLEMENTATION,
        selected_candidate_id=candidate.content_id if admitted else "",
        permitted_read_paths=(path,) if admitted else (),
        permitted_write_paths=(path,) if admitted else (),
        evidence_refs=(EvidenceReference("authority", "authority:one", producer_id="test"),),
        proof_refs=(EvidenceReference("proof", "proof:one", producer_id="test"),),
        invalidation_refs=("tree:fixture",),
    )


def propose_add_argument(
    reg: DoctorRepairOperatorRegistry,
    source: str = "process(event)",
    *,
    path: str = "pkg/caller.py",
    proof_admitted: bool = True,
    **kwargs: object,
) -> DoctorOperatorProposal:
    site = make_edit_site(path, source)
    defaults: dict[str, object] = {
        "obligation_refs": ("obligation:one",),
        "proof_refs": ("proof:one",),
        "value_source_refs": ("value:ctx",),
        "expression_ref": "expr:ctx",
        "parameter_name": "context",
        "proof_admitted": proof_admitted,
    }
    defaults.update(kwargs)
    return reg.propose(DoctorOperatorKind.ADD_ARGUMENT, site, **defaults)  # type: ignore[arg-type]


_PROOF_DEFAULT = object()


def make_request(
    *,
    source: str = "process(event)",
    path: str = "pkg/caller.py",
    proof_admitted: bool = True,
    proof: object = _PROOF_DEFAULT,
    file_text: str | None = None,
    require_proof_receipt: bool = True,
    **kwargs: object,
) -> DoctorSynthesisRequest:
    auth = roots()
    reg = registry(auth)
    proposal = propose_add_argument(
        reg, source, path=path, proof_admitted=proof_admitted
    )
    if proof is _PROOF_DEFAULT:
        proof_value: object | None = proof_receipt() if require_proof_receipt else None
    else:
        proof_value = proof  # may be None when caller forces missing receipt
    values: dict[str, object] = {
        "roots": auth,
        "proposal": proposal,
        "span_text": source,
        "expression_text": "ctx",
        "value_mapping": mapping(),
        "decision": repair_decision(path=path),
        "proof_receipt": proof_value,
        "selected_consequence_ref": "consequence:unique-repair",
        "value_ref": "value:ctx",
        "placement_ref": f"placement:{path}:0:{len(source)}",
        "finding_id": "finding:one",
        "plan_receipt_id": "plan:one",
        "proof_receipt_id": "proof:one",
        "require_proof_receipt": require_proof_receipt,
    }
    if file_text is not None:
        values["file_text"] = file_text
    values.update(kwargs)
    return DoctorSynthesisRequest(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface / authority surface
# ---------------------------------------------------------------------------


def test_interface_and_factory() -> None:
    auth = roots()
    synth = create_deterministic_doctor_synthesizer(auth)
    assert synth.INTERFACE == DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE
    assert DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE == "DeterministicDoctorSynthesizer@1"
    assert synth.registry.registry_id == registry(auth).registry_id


def test_module_does_not_import_provider_or_llm_surfaces() -> None:
    source_path = Path(
        "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    forbidden = (
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "change_propagation_provider_router",
        "todo_daemon",
        "integrations",
    )
    joined = " ".join(imported)
    for marker in forbidden:
        assert marker not in joined
    # Runtime import graph must not load provider modules as a side effect.
    before = set(sys.modules)
    importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis"
    )
    added = set(sys.modules) - before
    for name in added:
        lowered = name.lower()
        assert "llm_router" not in lowered
        assert "openai" not in lowered
        assert "anthropic" not in lowered


def test_monkeypatched_llm_routes_that_raise_remain_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis must succeed even when LLM routes would raise if called."""

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("llm route must never be called")

    # Install synthetic provider modules that raise on any attribute access use.
    fake_llm = types.ModuleType("ipfs_accelerate_py.agent_supervisor.llm_router")
    fake_llm.complete = _boom  # type: ignore[attr-defined]
    fake_provider = types.ModuleType(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.change_propagation_provider_router"
    )
    fake_provider.route = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_llm.__name__, fake_llm)
    monkeypatch.setitem(sys.modules, fake_provider.__name__, fake_provider)

    request = make_request()
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert receipt.admitted
    assert receipt.provider_invoked is False
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.write_performed is False
    # LLM modules must still raise if someone actually calls them.
    with pytest.raises(RuntimeError, match="never be called"):
        fake_llm.complete("prompt")


# ---------------------------------------------------------------------------
# Happy path: unique proof-admitted overlay
# ---------------------------------------------------------------------------


def test_synthesize_add_argument_overlay_with_patch_cid_and_replay() -> None:
    source = "process(event)"
    file_text = "def caller():\n    return process(event)\n"
    request = make_request(source=source, file_text=file_text)
    synth = DeterministicDoctorSynthesizer(registry=registry())
    receipt = synth.synthesize(request)

    assert receipt.disposition is DoctorSynthesisDisposition.SUPPORTED
    assert receipt.admitted
    assert receipt.overlay is not None
    assert receipt.overlay.path == "pkg/caller.py"
    assert receipt.overlay.replacement == "process(event, context=ctx)"
    assert receipt.overlay.write_authority is False
    assert receipt.overlay.semantic_authority is False
    assert receipt.overlay.source_write_count == 0
    assert receipt.patch_cid
    assert receipt.byte_equivalent_replay is True
    assert receipt.write_performed is False
    assert receipt.provider_invoked is False
    assert DoctorSynthesisReason.RENDERED.value in receipt.reason_codes

    # Exact before / after hashes.
    assert (
        receipt.before_hash
        == "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    )
    assert (
        receipt.after_hash
        == "sha256:"
        + hashlib.sha256(receipt.overlay.replacement.encode()).hexdigest()
    )
    assert receipt.overlay.after_hash == receipt.after_hash

    # Simulation parsed without writing.
    assert receipt.simulation is not None
    assert receipt.simulation.parse_ok is True
    assert receipt.simulation.wrote_target is False

    # Input identities recomputed.
    assert "proposal" in receipt.input_identities
    assert "operator_registry" in receipt.input_identities
    assert "span_before_hash" in receipt.input_identities

    # Round-trip overlay.
    restored = DoctorAnalyticalOverlay.from_dict(receipt.overlay.to_dict())
    assert restored.patch_cid == receipt.overlay.patch_cid
    assert restored.replacement == receipt.overlay.replacement


def test_patch_cid_is_stable_across_runs() -> None:
    request = make_request()
    reg = registry()
    first = materialize_proof_admitted_overlay(request, registry=reg)
    second = materialize_proof_admitted_overlay(request, registry=reg)
    assert first.admitted and second.admitted
    assert first.patch_cid == second.patch_cid
    assert first.overlay is not None and second.overlay is not None
    assert first.overlay.replacement == second.overlay.replacement
    assert first.replay_identity == second.replay_identity


def test_full_file_simulation_applies_span_without_writing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "caller.py"
    original = "def caller():\n    return process(event)\n"
    target.write_text(original, encoding="utf-8")
    before = target.read_text(encoding="utf-8")

    request = make_request(source="process(event)", file_text=original)
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert receipt.admitted
    # Target file untouched.
    assert target.read_text(encoding="utf-8") == before
    assert receipt.simulation is not None
    assert receipt.simulation.wrote_target is False
    # Simulated after text is valid Python.
    simulated = original.replace("process(event)", receipt.overlay.replacement)  # type: ignore[union-attr]
    ast.parse(simulated)


def test_exact_rename_synthesis() -> None:
    auth = roots()
    reg = registry(auth)
    source = "old_name"
    site = make_edit_site("pkg/symbols.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.EXACT_RENAME,
        site,
        obligation_refs=("obligation:rename",),
        proof_refs=("proof:rename",),
        parameter_name="new_name",
        previous_parameter_name="old_name",
        proof_admitted=True,
    )
    request = DoctorSynthesisRequest(
        roots=auth,
        proposal=proposal,
        span_text=source,
        proof_receipt=proof_receipt(),
        selected_consequence_ref="consequence:unique-repair",
        placement_ref="placement:pkg/symbols.py:0:8",
    )
    receipt = DeterministicDoctorSynthesizer(registry=reg).synthesize(request)
    assert receipt.admitted
    assert receipt.overlay is not None
    assert receipt.overlay.replacement == "new_name"


def test_schema_projection_synthesis() -> None:
    auth = roots()
    reg = registry(auth)
    source = '{"tenant":"t1","name":"n"}'
    site = make_edit_site("pkg/schema.json", source)
    proposal = reg.propose(
        DoctorOperatorKind.SCHEMA_PROJECTION,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        field_mapping_refs=("map:tenant->tenant_id", "map:name->display_name"),
        proof_admitted=True,
    )
    request = DoctorSynthesisRequest(
        roots=auth,
        proposal=proposal,
        span_text=source,
        field_mappings=(
            FieldMapping("tenant", "tenant_id"),
            FieldMapping("name", "display_name"),
        ),
        proof_receipt=proof_receipt(),
        selected_consequence_ref="consequence:unique-repair",
    )
    receipt = DeterministicDoctorSynthesizer(registry=reg).synthesize(request)
    assert receipt.admitted
    assert receipt.overlay is not None
    assert receipt.overlay.replacement == '{"display_name":"n","tenant_id":"t1"}'


# ---------------------------------------------------------------------------
# Fail-closed abstention matrix
# ---------------------------------------------------------------------------


def test_without_proof_admission_abstains_with_no_overlay() -> None:
    request = make_request(proof_admitted=False)
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert receipt.disposition is DoctorSynthesisDisposition.ABSTAIN
    assert not receipt.admitted
    assert receipt.overlay is None
    assert DoctorSynthesisReason.PROOF_NOT_ADMITTED.value in receipt.reason_codes
    assert DoctorSynthesisReason.NO_PARTIAL_OVERLAY.value in receipt.reason_codes
    assert receipt.write_performed is False


def test_missing_proof_receipt_abstains() -> None:
    request = make_request(proof=None, require_proof_receipt=True)
    # proof=None with require_proof_receipt True
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert DoctorSynthesisReason.PROOF_RECEIPT_REQUIRED.value in receipt.reason_codes


def test_non_unique_proof_abstains() -> None:
    request = make_request(
        proof=proof_receipt(
            unique=False,
            eligible=("consequence:a", "consequence:b"),
            consequence="consequence:a",
        )
    )
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert DoctorSynthesisReason.PROOF_NOT_UNIQUE.value in receipt.reason_codes


def test_consequence_mismatch_abstains() -> None:
    request = make_request(
        proof=proof_receipt(consequence="consequence:unique-repair"),
        selected_consequence_ref="consequence:other",
    )
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert DoctorSynthesisReason.CONSEQUENCE_MISMATCH.value in receipt.reason_codes
    assert receipt.overlay is None


def test_stale_span_abstains() -> None:
    request = make_request(source="process(event)")
    # Corrupt span text after proposal bound to original before_hash.
    object.__setattr__(request, "span_text", "process(other)")
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert DoctorSynthesisReason.STALE_SPAN.value in receipt.reason_codes


def test_unproved_value_abstains() -> None:
    request = make_request(
        value_mapping=mapping(
            disposition=SynthesisDisposition.AMBIGUOUS,
            proved=("candidate:a", "candidate:b"),
        )
    )
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert (
        DoctorSynthesisReason.UNPROVED_VALUE.value in receipt.reason_codes
        or DoctorSynthesisReason.TARGET_NOT_UNIQUE.value in receipt.reason_codes
        or DoctorSynthesisReason.RENDER_FAILED.value in receipt.reason_codes
    )


def test_value_ref_mismatch_abstains() -> None:
    request = make_request(value_ref="value:wrong")
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert DoctorSynthesisReason.VALUE_MISMATCH.value in receipt.reason_codes
    assert receipt.overlay is None


def test_extra_paths_rejected_at_request_construction() -> None:
    with pytest.raises(DoctorSynthesisAuthorityError, match="extra_paths"):
        make_request(extra_paths=("pkg/other.py",))


def test_extra_imports_rejected_at_request_construction() -> None:
    with pytest.raises(DoctorSynthesisAuthorityError, match="extra imports"):
        make_request(extra_imports=("import evil",))


def test_new_dependency_outside_allowlist_abstains() -> None:
    auth = roots()
    reg = registry(auth)
    source = "import os\n"
    site = make_edit_site("pkg/mod.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.ADD_IMPORT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        import_module="external.vendor",
        import_name="Thing",
        allowed_dependency_paths=("pkg/",),
        proof_admitted=True,
    )
    request = DoctorSynthesisRequest(
        roots=auth,
        proposal=proposal,
        span_text=source,
        proof_receipt=proof_receipt(),
        selected_consequence_ref="consequence:unique-repair",
    )
    receipt = DeterministicDoctorSynthesizer(registry=reg).synthesize(request)
    assert not receipt.admitted
    assert receipt.overlay is None
    assert (
        DoctorSynthesisReason.EXTRA_DEPENDENCY.value in receipt.reason_codes
        or DoctorSynthesisReason.RENDER_FAILED.value in receipt.reason_codes
    )


def test_unsupported_ast_shape_splat_abstains() -> None:
    request = make_request(source="process(event, *rest)")
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert any(
        code
        in {
            DoctorSynthesisReason.UNSUPPORTED_AST_SHAPE.value,
            DoctorSynthesisReason.RENDER_FAILED.value,
        }
        for code in receipt.reason_codes
    )


def test_invented_expression_is_semantics_outside_consequence() -> None:
    request = make_request(expression_text="make_context()")
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None
    assert any(
        code
        in {
            DoctorSynthesisReason.SEMANTICS_OUTSIDE_CONSEQUENCE.value,
            DoctorSynthesisReason.RENDER_FAILED.value,
        }
        for code in receipt.reason_codes
    )


def test_proof_with_llm_invocations_abstains() -> None:
    request = make_request(proof=proof_receipt(llm_invocation_count=1))
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL.value in receipt.reason_codes
    assert receipt.overlay is None


def test_proof_claiming_write_authority_abstains() -> None:
    request = make_request(proof=proof_receipt(write_authority=True))
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert DoctorSynthesisReason.WRITE_ATTEMPTED.value in receipt.reason_codes
    assert receipt.overlay is None


def test_root_mismatch_between_proof_and_request_abstains() -> None:
    request = make_request(proof=proof_receipt(tree_id="tree:other"))
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert DoctorSynthesisReason.ROOT_MISMATCH.value in receipt.reason_codes
    assert receipt.overlay is None


def test_unauthorized_write_path_abstains() -> None:
    request = make_request(
        path="pkg/other.py",
        decision=repair_decision(path="pkg/caller.py", admitted=True),
    )
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert not receipt.admitted
    assert receipt.overlay is None


def test_failed_render_never_returns_partial_overlay() -> None:
    request = make_request(proof_admitted=False)
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    payload = receipt.to_dict()
    assert payload["overlay_id"] == ""
    assert receipt.overlay is None
    # Constructing a supported receipt without overlay must fail.
    with pytest.raises(Exception):
        DoctorSynthesisReceipt(
            disposition=DoctorSynthesisDisposition.SUPPORTED,
            reason_codes=(DoctorSynthesisReason.RENDERED.value,),
            roots=roots(),
            overlay=None,
            patch_cid="cid:x",
            byte_equivalent_replay=True,
        )


def test_overlay_cannot_claim_write_authority() -> None:
    request = make_request()
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    assert receipt.admitted and receipt.overlay is not None
    with pytest.raises(DoctorSynthesisAuthorityError):
        DoctorAnalyticalOverlay(
            roots=receipt.overlay.roots,
            overlay_id=receipt.overlay.overlay_id,
            path=receipt.overlay.path,
            before_hash=receipt.overlay.before_hash,
            after_hash=receipt.overlay.after_hash,
            span_start=receipt.overlay.span_start,
            span_end=receipt.overlay.span_end,
            replacement=receipt.overlay.replacement,
            patch_cid=receipt.overlay.patch_cid,
            operator_id=receipt.overlay.operator_id,
            operator_kind=receipt.overlay.operator_kind,
            write_authority=True,
        )


def test_receipt_cannot_claim_provider_invocation() -> None:
    request = make_request()
    receipt = materialize_proof_admitted_overlay(request, registry=registry())
    with pytest.raises(DoctorSynthesisAuthorityError):
        DoctorSynthesisReceipt(
            disposition=DoctorSynthesisDisposition.SUPPORTED,
            reason_codes=(DoctorSynthesisReason.RENDERED.value,),
            roots=receipt.roots,
            patch_cid=receipt.patch_cid,
            overlay=receipt.overlay,
            byte_equivalent_replay=True,
            provider_invoked=True,
        )


# ---------------------------------------------------------------------------
# Evidence subset / body-free receipt discipline
# ---------------------------------------------------------------------------


def test_receipt_exposes_evidence_subset_without_write() -> None:
    receipt = materialize_proof_admitted_overlay(make_request(), registry=registry())
    assert receipt.admitted
    payload = receipt.to_dict()
    assert payload["finding_id"] == "finding:one"
    assert payload["plan_receipt_id"] == "plan:one"
    assert payload["proof_receipt_id"] == "proof:one"
    assert payload["operator_id"]
    assert payload["selected_consequence_ref"] == "consequence:unique-repair"
    assert payload["value_ref"] == "value:ctx"
    assert payload["placement_ref"]
    assert payload["before_hash"]
    assert payload["after_hash"]
    assert payload["patch_cid"]
    assert payload["write_performed"] is False
    assert payload["write_authority"] is False
    assert payload["provider_invoked"] is False
    assert receipt.producer_id == PRODUCER_ID


def test_symbols_exported() -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis"
    )
    assert hasattr(mod, "DeterministicDoctorSynthesizer")
    assert hasattr(mod, "DoctorAnalyticalOverlay")
    assert hasattr(mod, "DoctorSynthesisReceipt")
    assert inspect.isclass(mod.DeterministicDoctorSynthesizer)
    assert inspect.isclass(mod.DoctorAnalyticalOverlay)
    assert inspect.isclass(mod.DoctorSynthesisReceipt)
