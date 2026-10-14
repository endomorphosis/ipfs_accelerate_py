"""Live local-proof/context joins over real retained repository and cache owners.

The existing authored fixture supplies fake pinned solver/kernel transcripts.
Git, signed manifests, native admission, SQLite/DuckDB/CAS, the repository
session/index and context compiler are real. No target source, Lean, Z3, model,
provider or worker is executed by this suite. A conditional local receipt never
becomes a whole-program, execution, completion or omission-safety claim.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, ContextExpansionCancelled, EvidenceExpansionRequest,
    EvidenceValuePolicy,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget, ContextCapsule, ContextReference, ContextTier,
    canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as factory
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime.repository_proof_context import (
    RepositoryProofContextAdapter,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_proof_source_binding import (
    bind_repository_proof_source,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from test.api.test_agent_supervisor_local_planning_admission import scenario as _scenario  # noqa: F401
from test.api.test_doctor_header_cache_bridge import (
    authored, _forbid_owners, _import, _prepare,  # noqa: F401
)
from test.api.test_doctor_header_contracts import analyze
from test.api.test_repository_source_index import scheduler  # noqa: F401


CONDITIONS = "repository-proof-conditions:"
PROOF = "repository-proof:"
QUESTION = "Which definitions implement the reviewed local header contract?"


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    # Importing an authored fixture does not import its module's autouse fixture.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


@pytest.fixture
def scenario(_scenario):
    # The later authored fixture signs a fresh manifest after _prepare changes
    # answer.py. This original is signed but outside the theorem/task scope.
    repository = _scenario["repository"]
    (repository / "omitted.py").write_text("def parked(value):\n    return value + 7\n")
    for arguments in (("add", "omitted.py"), ("-c", "user.name=Proof context fixture",
            "-c", "user.email=proof-context@example.invalid", "commit", "-qm", "signed omitted original")):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True)
    yield _scenario


def compiler(*, optional_items=None, tokenizer=None):
    return ContextCompiler(ContextBudget(max_input_tokens=100_000,
        reserved_output_tokens=0, reserved_tool_tokens=0, max_items=64,
        max_item_bytes=65_536, max_text_bytes=65_536, max_serialized_bytes=262_144),
        tokenizer=tokenizer or (lambda text: max(1, len(text.encode("utf-8")) // 16)),
        value_policy=EvidenceValuePolicy(max_optional_items=optional_items))


def required_references():
    result = []
    for family in ("intent_ir", "security_ir", "ui_ux_ir"):
        body = "  Preserve the authored " + family + " requirement exactly.\n"
        result.append(ContextReference(reference_id="required:" + family,
            kind=family + "-requirement", tier=ContextTier.INVARIANT,
            referenced_content_id="sha256:" + hashlib.sha256(body.encode()).hexdigest(),
            summary=body, metadata={"required": True, "coverage_ids": ("coverage:" + family,)}))
    return tuple(result)


@contextmanager
def context_case(values, tmp_path, scheduler, *, admission=None, optional_items=None):
    admitted = _import(values) if admission is None else admission
    scoped = values["scoped"]
    pins = dict(expected_manifest_cid=scoped.report["manifest_cid"],
        expected_task_cid=scoped.task_cid, expected_scope_id=scoped.report["scope_id"])
    tokenizer_control = dict(armed=False, action=None, calls=[])

    def tokenizer(text):
        result = max(1, len(text.encode("utf-8")) // 16)
        if tokenizer_control["armed"] and not tokenizer_control["calls"]:
            tokenizer_control["calls"].append(True)
            tokenizer_control["action"]()
        return result

    with RepositorySourceSession(values["repository"], tmp_path / "proof-context-session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        adapter = session.context_adapter(compiler(optional_items=optional_items, tokenizer=tokenizer))
        source = bind_repository_proof_source(session=session, scoped_analysis=scoped,
            generation=1, **pins)
        options = dict(context_adapter=adapter, gate=values["gate"], admission=admitted,
            source_binding=source, scoped_analysis=scoped, expected_key=values["prepared"].key,
            expected_receipt_id=admitted.receipt.receipt_id,
            expected_cache_owner_path_sha256=values["owner_pin"], generation=1, **pins)
        bridge = RepositoryProofContextAdapter(**options)
        case = SimpleNamespace(values=values, session=session, adapter=adapter,
            bridge=bridge, source=source, admission=admitted, options=options,
            required=required_references(), calls_before=tuple(values["calls"]),
            tokenizer_control=tokenizer_control)
        try:
            yield case
        finally:
            # Refusals can leave source/cache deliberately stale. Retiring the
            # wrapper owns only its local parents, not these caller-owned stores.
            if not bridge._closed:
                bridge.close()
            adapter.close()
    assert scheduler.snapshot()["active_lease_count"] == 0


@pytest.fixture
def proof_context(authored, tmp_path, scheduler):
    with context_case(authored, tmp_path, scheduler) as case:
        yield case


def compile_context(case, **changes):
    options = dict(generation=1, query="clean_label clean_payload", unresolved_question=QUESTION,
        required_evidence=case.required,
        required_reference_ids=tuple(ref.reference_id for ref in case.required),
        required_coverage_ids=tuple(value for ref in case.required for value in ref.coverage_ids),
        objective_id="intent:local-header-context", objective_revision="sha256:authored-intent",
        policy_id="policy:authored-local-review", policy_revision="sha256:authored-policy",
        goal={"intent": "Review the conditional local header proof and remaining obligations."},
        authority={"mode": "proposal", "allowed_paths": ("answer.py",)},
        scope={"paths": ("answer.py",)},
        acceptance={"criteria": ("Retain all local conditions and unresolved obligations.",)})
    options.update(changes)
    return case.bridge.compile(**options)


def references(parent):
    return {ref.reference_id: ref for ref in (*parent.evidence, *parent.expansion_references)}


def condition(case, parent):
    refs = references(parent)
    required = [ref for ref in parent.evidence if ref.reference_id.startswith(CONDITIONS)]
    assert len(required) == 1
    ref = required[0]
    body = json.loads(ref.summary)
    receipt = case.admission.receipt
    assert ref.required and ref.tier is ContextTier.INVARIANT
    assert body["local_scope"] == receipt.scope.to_dict()
    assert body["public_theorem"] == json.loads(case.admission.key.goal.canonical_bytes)
    assert body["public_source_premises"] == [json.loads(item.canonical_bytes) for item in case.admission.key.premises]
    assert set(body["open_frontiers"]) == set(receipt.metadata["open_frontiers"])
    assert body["security_ir_obligations_discharged"] is False
    assert receipt.scope.assumption_ids and body["local_scope"]["assumption_ids"]
    assert body["local_scope"]["residual_obligation_ids"] == list(receipt.scope.residual_obligation_ids)
    assert receipt.satisfies_completion() is False
    for name in ("whole_program_proved", "completion_authority", "execution_authority", "mutation_authority"):
        assert body["local_scope"][name] is False
    assert all(parent.authority[name] is False for name in
        ("proof_authority", "execution_authority", "completion_authority"))
    for original in case.required:
        assert refs[original.reference_id].canonical_bytes() == original.canonical_bytes()
    for proof in refs.values():
        if proof.reference_id.startswith(PROOF):
            assert not proof.required and proof.metadata["digest_only"] is True
            assert json.loads(proof.summary)["conditions_reference_id"] == ref.reference_id
    return ref


def native_handle(parent):
    values = [ref for ref in references(parent).values()
              if ref.reference_id.startswith("repository-symbol:")]
    assert values
    return values[0]


def mutate_source(case, name="answer.py"):
    path = case.values["repository"] / name
    path.write_bytes(path.read_bytes() + b"\n# source changed after proof context capture\n")


def deny_cache(case, denial):
    gate = case.values["gate"]
    key = case.admission.key
    if denial == "quarantine":
        gate.quarantine(key)
    elif denial == "tombstone":
        gate.invalidate_semantic_root(root_field="tree", root_cid=key.tree.cid)
    else:
        connection = gate._cache._connect()
        try:
            key_id = key.to_formal_key().key_id
            if denial == "miss":
                connection.execute("DELETE FROM proof_cache_entries WHERE key_id=?", (key_id,))
                connection.commit()
            elif denial == "poison":
                connection.execute("UPDATE proof_cache_entries SET entry_json=? WHERE key_id=?",
                                   ('{"authored_poison":true}', key_id))
                connection.commit()
            else:
                row = connection.execute("SELECT expires_at_ms FROM proof_cache_entries WHERE key_id=?",
                                         (key_id,)).fetchone()
                assert row is not None
                expires = int(row[0])
                gate._cache._clock = lambda: expires / 1000 + 1
        finally:
            connection.close()


def test_compile_render_reuses_owners_without_reexecuting_authored_provers(proof_context, monkeypatch):
    case = proof_context
    _forbid_owners(monkeypatch)
    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned",
        lambda *_args, **_kwargs: pytest.fail("context reuse must not execute a prover"))
    parent = compile_context(case).capsule
    condition(case, parent)
    observed = case.source.observation()
    assert observed["omitted_source_count"] == 1
    assert observed["git_tree_equivalence_verified"] is False
    assert observed["scope_id"] == case.options["expected_scope_id"]
    assert parent.tree_id == "sha256:" + observed["source_tree_sha256"]
    assert case.admission.receipt.repository_tree_id != parent.tree_id
    assert case.bridge.render(parent, generation=1) == canonical_context_json_bytes(parent.provider_input_payload)
    assert case.bridge.validate_current(parent, generation=1) is None
    assert tuple(case.values["calls"]) == case.calls_before
    case.bridge.close()
    with pytest.raises(ValueError):
        case.bridge.render(parent, generation=1)
    assert case.session.status_summary()["source_current"] is True
    assert case.admission.revalidate_for_render().hit
    assert case.adapter.validate_current(parent, generation=1) is None


def test_budget_omits_optional_digest_without_omitting_complete_conditions(authored, tmp_path, scheduler):
    with context_case(authored, tmp_path, scheduler, optional_items=1) as case:
        # One optional slot is valid. A short, explicitly higher-value advisory
        # competes for it through the compiler's real measured cost policy.
        preferred = ContextReference(reference_id="optional:review", kind="review-advisory",
            summary="Review.", metadata={"expected_decision_change_bps": 10_000,
                "uncertainty_bps": 10_000, "uncertainty_reduction_bps": 10_000})
        parent = compile_context(case, optional_evidence=(preferred,)).capsule
        required = condition(case, parent)
        assert preferred in parent.evidence
        assert not any(ref.reference_id.startswith(PROOF) for ref in parent.evidence)
        assert any(ref.reference_id.startswith(PROOF) for ref in parent.expansion_references)
        assert required in parent.evidence
        assert case.bridge.render(parent, generation=1) == canonical_context_json_bytes(parent.provider_input_payload)


def test_native_expansion_and_retry_preserve_conditions_and_every_caller_requirement(proof_context):
    case = proof_context
    parent = compile_context(case).capsule
    original = condition(case, parent).canonical_bytes()
    handle = native_handle(parent)
    expanded = case.bridge.expand_for_question(parent,
        EvidenceExpansionRequest(QUESTION, (handle.reference_id,)), generation=1).reconstructed_capsule
    diagnostic = ContextReference(reference_id="diagnostic:local-proof-retry", kind="diagnostic",
        summary="The local proof still leaves the whole-program review unresolved.")
    retried = case.bridge.retry(expanded, generation=1, evidence=(diagnostic,)).reconstructed_capsule
    for capsule in (expanded, retried):
        assert condition(case, capsule).canonical_bytes() == original
        assert case.bridge.render(capsule, generation=1) == canonical_context_json_bytes(capsule.provider_input_payload)
    assert diagnostic in retried.evidence
    assert tuple(case.values["calls"]) == case.calls_before


def test_independent_security_ir_residuals_survive_projection(authored, tmp_path, scheduler):
    analysis = analyze()
    compiled = factory.security_ir_bridge.compile_header_security_ir(scoped=authored["scoped"],
        analyses=[dict(path="answer.py", status=analysis.status,
            reason_codes=list(analysis.reason_codes), source_sha256=analysis.source_sha256,
            contracts=list(analysis.contracts), open_frontiers=list(analysis.open_frontiers))])
    contract = dict(authored["contract"], security_ir_declaration_cid=compiled.report["declaration_cid"],
                    security_ir_formalization_cid=compiled.report["artifact_cid"])
    prepared = _prepare(authored, contract=contract)
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=authored["store"],
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    # The imported fixture's explicitly authored _run_pinned remains installed.
    with public_replay_without_metadata():
        proof = hammer.verify_authoritative(prepared.theorem, prepared.lowering,
            solver_pin=prepared.solver_pin, kernel_pin=prepared.kernel_pin,
            current_roots=prepared.theorem.roots,
            eligible_consequence_refs=(prepared.theorem.consequence_ref,))
        custody_ref = custody.persist_header_proof_key_custody(prepared=prepared, proof=proof, hammer=hammer)
    admission = _import(authored, prepared=prepared, hammer=hammer, custody_ref=custody_ref,
                        expected_key_id=prepared.key.key_id)
    security_values = dict(authored, prepared=prepared, hammer=hammer, custody_ref=custody_ref)
    with context_case(security_values, tmp_path, scheduler, admission=admission) as case:
        parent = compile_context(case).capsule
        condition(case, parent)
        expected = tuple(sorted(item.obligation_id for item in compiled.artifact.proof_obligations))
        assert expected and admission.receipt.scope.residual_obligation_ids == expected


@pytest.mark.parametrize("forgery", ["key", "serialized_admission", "serialized_source", "manifest", "task", "owner_pin"])
def test_constructor_requires_original_native_inputs_and_independent_pins(proof_context, forgery):
    case = proof_context
    options = dict(case.options)
    if forgery == "key":
        options["expected_key"] = replace(case.admission.key, policy=case.values["profiles"].environment)
    elif forgery == "serialized_admission":
        options["admission"] = {"accepted": True, "key": case.admission.key.to_dict(),
                                "receipt": case.admission.receipt.to_dict()}
    elif forgery == "serialized_source":
        options["source_binding"] = case.source.observation()
    elif forgery == "manifest":
        options["expected_manifest_cid"] = content_identity({"authored": "different-manifest"})
    elif forgery == "task":
        options["expected_task_cid"] = content_identity({"authored": "different-task"})
    else:
        options["expected_cache_owner_path_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        RepositoryProofContextAdapter(**options)
    assert tuple(case.values["calls"]) == case.calls_before


@pytest.mark.parametrize("drift", ["selected", "omitted", "untracked", "empty_directory"])
def test_complete_source_drift_refuses_render_of_an_issued_positive(proof_context, drift):
    case = proof_context
    parent = compile_context(case).capsule
    if drift in {"selected", "omitted"}:
        mutate_source(case, "answer.py" if drift == "selected" else "omitted.py")
    elif drift == "untracked":
        (case.values["repository"] / "new.py").write_text("def extra():\n    return 9\n")
    else:
        (case.values["repository"] / "extra-empty-directory").mkdir()
    with pytest.raises(ValueError):
        case.bridge.render(parent, generation=1)


def test_same_source_refresh_cannot_reuse_the_old_binding_or_parent(proof_context):
    case = proof_context
    parent = compile_context(case).capsule
    case.session.refresh(generation=1)
    for generation in (1, 2):
        with pytest.raises(ValueError):
            case.bridge.validate_current(parent, generation=generation)
    with pytest.raises(ValueError):
        RepositoryProofContextAdapter(**case.options)


@pytest.mark.parametrize("denial", ["expired_clock", "miss", "poison", "quarantine", "tombstone"])
def test_real_cache_denial_after_compile_refuses_current_render(proof_context, denial):
    case = proof_context
    parent = compile_context(case).capsule
    deny_cache(case, denial)
    with pytest.raises(ValueError):
        case.bridge.render(parent, generation=1)
    assert tuple(case.values["calls"]) == case.calls_before


@pytest.mark.parametrize("drift", ["source", "cache", "owner"])
def test_tokenizer_callback_cannot_publish_changed_source_cache_or_owner(proof_context, drift):
    case = proof_context

    def action():
        if drift == "source":
            mutate_source(case, "omitted.py")
        elif drift == "cache":
            case.values["gate"].quarantine(case.admission.key)
        else:
            case.values["gate"]._cache = case.values["replacement_cache"]

    case.tokenizer_control.update(armed=True, action=action)
    with pytest.raises(ValueError):
        compile_context(case)
    assert case.tokenizer_control["calls"] == [True] and not case.bridge._parents


@pytest.mark.parametrize("changed", ["authority", "required_reference"])
def test_native_cache_callback_cannot_mutate_the_original_caller_compile_inputs(
        proof_context, monkeypatch, changed):
    case = proof_context
    authority = {"mode": "proposal", "allowed_paths": ("answer.py",)}
    original = case.values["gate"].lookup
    called = []

    def lookup(*args, **kwargs):
        result = original(*args, **kwargs)
        if not called:
            called.append(True)
            if changed == "authority":
                authority["mode"] = "caller-mutated-execution"
            else:
                object.__setattr__(case.required[0], "summary", "Caller waived the original requirement.")
        return result

    monkeypatch.setattr(case.values["gate"], "lookup", lookup)
    with pytest.raises(ValueError):
        compile_context(case, authority=authority)
    assert called == [True] and not case.bridge._parents


def test_original_tokenizer_cannot_mutate_budget_estimator_or_value_policy_in_place(
        authored, tmp_path, scheduler):
    # Each independent control gets fresh session/index/compiler objects and
    # its own closing refusal. Keep the installed tokenizer identity unchanged.
    for changed in ("budget", "estimator", "value_policy"):
        private = tmp_path / changed
        private.mkdir()
        with context_case(authored, private, scheduler) as case:
            def action():
                selected = case.adapter.compiler
                if changed == "budget":
                    object.__setattr__(selected.budget, "max_items", 63)
                elif changed == "estimator":
                    selected.estimator._chars_per_token += 1
                else:
                    object.__setattr__(selected.value_policy, "max_optional_items", 1)

            case.tokenizer_control.update(armed=True, action=action)
            with pytest.raises(ValueError):
                compile_context(case)
            assert case.tokenizer_control["calls"] == [True] and not case.bridge._parents


@pytest.mark.parametrize("drift", ["source", "cache"])
def test_compiler_return_callback_cannot_publish_a_stale_parent(proof_context, monkeypatch, drift):
    case = proof_context
    original = case.adapter.compiler.compile
    called = []

    def compiler_callback(**options):
        result = original(**options)
        called.append(True)
        mutate_source(case, "omitted.py") if drift == "source" else deny_cache(case, "quarantine")
        return result

    monkeypatch.setattr(case.adapter.compiler, "compile", compiler_callback)
    with pytest.raises(ValueError):
        compile_context(case)
    assert called == [True] and not case.bridge._parents


def test_compiler_callback_reentry_refuses_without_retiring_the_outer_context(proof_context, monkeypatch):
    case = proof_context
    original = case.adapter.compiler.compile
    called = []

    def compiler_callback(**options):
        called.append(True)
        with pytest.raises(ValueError):
            compile_context(case)
        with pytest.raises(ValueError):
            case.bridge.close()
        return original(**options)

    monkeypatch.setattr(case.adapter.compiler, "compile", compiler_callback)
    parent = compile_context(case).capsule
    assert called == [True] and not case.bridge._closed
    condition(case, parent)
    assert case.bridge.render(parent, generation=1) == canonical_context_json_bytes(parent.provider_input_payload)


@pytest.mark.parametrize("forgery", ["equal_copy", "changed_copy", "other_wrapper"])
def test_parent_must_be_the_original_object_issued_by_this_wrapper(proof_context, forgery):
    case = proof_context
    parent = compile_context(case).capsule
    duplicate = ContextCapsule.from_dict(parent.to_dict())
    assert duplicate is not parent and duplicate.canonical_bytes() == parent.canonical_bytes()
    if forgery == "changed_copy":
        duplicate = replace(duplicate, goal={"intent": "Claim whole-program completion."})
    other = None
    target = case.bridge
    if forgery == "other_wrapper":
        other = target = RepositoryProofContextAdapter(**case.options)
        duplicate = parent
    try:
        with pytest.raises(ValueError):
            target.render(duplicate, generation=1)
        assert case.bridge.render(parent, generation=1) == canonical_context_json_bytes(parent.provider_input_payload)
    finally:
        if other is not None:
            other.close()


@pytest.mark.parametrize("replacement", ["conditions", "proof_promotion"])
def test_retry_cannot_replace_conditions_or_promote_the_optional_local_proof(proof_context, replacement):
    case = proof_context
    parent = compile_context(case).capsule
    original = condition(case, parent)
    if replacement == "conditions":
        changed = replace(original, summary="All remaining conditions are waived.")
    else:
        proof = next(ref for ref in references(parent).values() if ref.reference_id.startswith(PROOF))
        changed = replace(proof, tier=ContextTier.INVARIANT, metadata=dict(proof.metadata, required=True),
                          summary="The whole task is complete.")
    with pytest.raises(ValueError):
        case.bridge.retry(parent, generation=1, evidence=(changed,))
    assert condition(case, parent).canonical_bytes() == original.canonical_bytes()


def test_cancellation_callback_source_change_cannot_return_an_expanded_context(proof_context):
    case = proof_context
    parent = compile_context(case).capsule
    handle = native_handle(parent)
    called = []

    def cancel():
        if not called:
            called.append(True)
            mutate_source(case, "omitted.py")
        return False

    with pytest.raises((ValueError, ContextExpansionCancelled)):
        case.bridge.expand_for_question(parent,
            EvidenceExpansionRequest(QUESTION, (handle.reference_id,)), generation=1, cancelled=cancel)
    assert called == [True]
    with pytest.raises(ValueError):
        case.bridge.render(parent, generation=1)


def test_final_render_cache_clock_callback_cannot_escape_the_closing_source_fence(proof_context, monkeypatch):
    case = proof_context
    parent = compile_context(case).capsule
    original_validate = case.adapter.validate_current
    original_clock = case.values["gate"]._cache._clock
    validations, writes = [], []

    def validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        validations.append(True)
        return result

    def clock():
        result = original_clock()
        # render has serialized its provider input before its second adapter
        # fence. Mutate only during the following, final native cache check.
        if len(validations) >= 2 and not writes:
            writes.append(True)
            mutate_source(case, "omitted.py")
        return result

    monkeypatch.setattr(case.adapter, "validate_current", validate)
    monkeypatch.setattr(case.values["gate"]._cache, "_clock", clock)
    with pytest.raises(ValueError):
        case.bridge.render(parent, generation=1)
    assert len(validations) == 2 and writes == [True]
    assert tuple(case.values["calls"]) == case.calls_before


@pytest.mark.parametrize("reserved", ["conditions", "proof"])
def test_caller_references_cannot_enter_the_projection_namespaces(proof_context, reserved):
    case = proof_context
    if reserved == "conditions":
        ref = replace(case.required[0], reference_id=CONDITIONS + "caller-authored")
        required = (*case.required, ref)
        options = dict(required_evidence=required,
            required_reference_ids=tuple(item.reference_id for item in required))
    else:
        ref = ContextReference(reference_id=PROOF + "caller-authored", kind="proof-hint",
                               summary="An unadmitted caller claim.")
        options = dict(optional_evidence=(ref,))
    with pytest.raises(ValueError):
        compile_context(case, **options)
    assert not case.bridge._parents
