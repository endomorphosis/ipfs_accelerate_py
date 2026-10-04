"""Independent qualification of explicit mathematical lookup and runtime residuals."""
from dataclasses import replace
import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import conditional_codebase_evidence as matcher
from ipfs_datasets_py.logic.intent_ir.schema import (
    IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
    ReviewStatus, SourceRef, SourceSpan, StatementKind,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec
from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain

TEXT = "Reviewed mathematical increment clause.\nReturn an exact Python int at runtime."
STATEMENT = "mathematical-goal"
PATH = "counter.py"
VIEW = "repository:conditional-intent"
FALSE_FIELDS = (
    "semantic_alignment_verified", "source_semantics_verified", "runtime_behavior_verified",
    "kernel_checked", "proof_authority", "execution_authority", "completion_authority",
    "mutation_authority", "admission_authority", "authoritative_cache_eligible", "behavioral_satisfaction",
)


def request(contract=None, domain=None):
    contract = contract or ContractSpec("increment", postconditions=("result == n + 1",))
    domain = domain or RequestedInputDomain("increment", predicates=("n >= 0",))
    digest = hashlib.sha256(TEXT.encode()).hexdigest()
    clause_end = TEXT.index("\n")
    sources = tuple(SourceRef(
        ref_id="source:" + role, source_uri="intent:" + role,
        source_id="conditional:request", source_revision="authored:1", content_sha256=digest,
        review_status=ReviewStatus.HUMAN_REVIEWED, span=SourceSpan(start, end),
    ) for role, start, end in (("mathematics", 0, clause_end), ("runtime", clause_end + 1, len(TEXT))))
    statements = (
        IntentStatement(
            statement_id=STATEMENT, kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
            normalized_text=TEXT[:clause_end], source_ref_ids=(sources[0].ref_id,),
            predicate=matcher.PREDICATE,
            arguments=(PATH, contract.function_name, cid_for_structured(contract.to_dict()),
                       cid_for_structured(domain.to_dict()), matcher.PROFILE),
            grounding=NodeGrounding.GROUNDED, review_status=ReviewStatus.HUMAN_REVIEWED,
        ),
        IntentStatement(
            statement_id="runtime-goal", kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
            normalized_text=TEXT[clause_end + 1:], source_ref_ids=(sources[1].ref_id,),
            predicate="runtime_exact_integer", arguments=(PATH, "increment"),
            grounding=NodeGrounding.GROUNDED, review_status=ReviewStatus.HUMAN_REVIEWED,
        ),
    )
    native = IntentIRDocument(document_id="intent:conditional", title="Authored mathematical input domain",
        intent_kind=IntentKind.DECLARATIVE, sources=sources, statements=statements)
    native.validate()
    return native, contract, domain


def prepare(document=None, contract=None, domain=None, **kwargs):
    native, owned_contract, owned_domain = request(contract, domain)
    return matcher.prepare_conditional_codebase_query(
        intent_document=native if document is None else document, source_text=TEXT,
        path=PATH, contract=owned_contract, domain=owned_domain, statement_id=STATEMENT, **kwargs,
    )


def test_exact_reviewed_native_math_retains_complete_authored_ledger_and_residual_ids():
    native, contract, domain = request()
    query = prepare()
    assert query["supported"] and query["reasons"] == []
    assert query["contract_cid"] == cid_for_structured(contract.to_dict())
    assert query["domain_cid"] == cid_for_structured(domain.to_dict())
    assert json.loads(query["intent_document_json"]) == native.to_dict()
    assert query["intent_source_text"] == TEXT
    assert [row["original_text"] for row in query["intent_source_ledger"]] == TEXT.split("\n")
    assert query["requirement_ids"] == [STATEMENT, "runtime-goal"]
    assert query["review_custody_verified"] is query["free_text_semantics_verified"] is False
    assert all(query[name] is False for name in FALSE_FIELDS)
    assert query["producer"]["module"] == matcher.__name__
    assert len(query["producer"]["sha256"]) == 64
    assert prepare(native.to_dict()) == query
    query["statement"]["arguments"][0] = "tampered.py"
    assert json.loads(prepare()["intent_document_json"]) == native.to_dict()


@pytest.mark.parametrize("change", [
    {"predicate": "repair", "arguments": ("agent", PATH)},
    {"review_status": ReviewStatus.UNREVIEWED},
    {"review_status": ReviewStatus.TRUSTED_FIXTURE},
    {"grounding": NodeGrounding.INFERRED},
    {"modality": IntentModality.PROHIBITED},
    {"arguments": (PATH, "increment", "wrong-contract", "wrong-domain", matcher.PROFILE)},
])
def test_generic_or_unreviewed_or_misaligned_atom_remains_unknown(change):
    native, _, _ = request()
    native = replace(native, statements=(replace(native.statements[0], **change), native.statements[1]))
    query = prepare(native)
    assert not query["supported"] and query["reasons"]
    assert query["requirement_ids"] == [STATEMENT, "runtime-goal"]
    assert all(query[name] is False for name in FALSE_FIELDS)


def test_unreviewed_source_is_not_replaced_by_a_statement_review_label():
    native, _, _ = request()
    native = replace(native, sources=(replace(native.sources[0], review_status=ReviewStatus.UNREVIEWED), native.sources[1]))
    assert not prepare(native)["supported"]


@pytest.mark.parametrize("changes", [
    {"content_sha256": "0" * 64},
    {"span": SourceSpan(0, 0)},
    {"span": SourceSpan(0, len(TEXT) + 1)},
])
def test_every_source_reference_is_rebound_to_full_original_bytes(changes):
    native, _, _ = request()
    native = replace(native, sources=(native.sources[0], replace(native.sources[1], **changes)))
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(native)


def test_complete_supplied_source_ledger_must_equal_native_inventory():
    native, _, _ = request()
    identities = [item.to_dict() for item in sorted(native.sources, key=lambda item: item.ref_id)]
    assert prepare(source_identity=identities)["supported"]
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(source_identity=identities[:1])


def test_arbitrary_conversion_hook_cannot_execute_during_native_decoding():
    class Payload:
        def to_dict(self):
            pytest.fail("non-native conversion hook executed")
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(Payload())


def test_explicit_function_domain_binding_and_unsafe_mutated_contract_are_revalidated():
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(domain=RequestedInputDomain("other", predicates=("True",)))
    _, contract, _ = request()
    object.__setattr__(contract, "postconditions", ())
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(contract=contract)


def test_oversized_inert_inputs_reject_before_native_decoder(monkeypatch):
    from ipfs_datasets_py.logic.intent_ir import decoder
    monkeypatch.setattr(decoder, "decode_intent_ir", lambda *args: pytest.fail("oversized input reached decoder"))
    native, _, _ = request()
    payload = native.to_dict()
    payload["title"] = "x" * (matcher.MAX_JSON_BYTES + 1)
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError):
        prepare(payload)


@pytest.fixture
def native_current(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig

    repository = tmp_path / "source"
    repository.mkdir()
    (repository / PATH).write_text("def increment(n: int) -> int:\n    return n\n", encoding="utf-8")
    connection = duckdb.connect(str(tmp_path / "owner.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=connection)
    cas = ImmutableCAS(tmp_path / "artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
                                   catalog=CodebaseCatalog(store, cas))
    owner = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "admission.json", proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        lane_reservations={}, auto_renew_leases=False, poll_interval_seconds=.005,
    ))
    head = index.prepare_current(repository, repository_id=VIEW, operation_id="initial",
                                 expected_head=None, scheduler=owner).head
    catalog = CodebaseVerificationCatalog(index)
    yield {"index": index, "catalog": catalog, "repository": repository,
           "repository_id": VIEW, "expected_head": head, "scheduler": owner}
    assert owner.snapshot()["active_lease_count"] == owner.snapshot()["waiting_request_count"] == 0
    connection.close()


def publish(native, contract, domain, operation_id="conditional-evidence", *, bounds=None):
    for executable in ("z3", "cvc5"):
        if shutil.which(executable) is None:
            pytest.skip(f"real {executable} executable unavailable")
    from ipfs_datasets_py.logic.software_contracts.codebase_verification import verify_current_codebase_unit
    from ipfs_datasets_py.logic.software_contracts.codebase_applicability import verify_current_codebase_applicability
    verification = verify_current_codebase_unit(native["index"], native["repository"],
        expected_head=native["expected_head"], path=PATH, contracts=[contract], scheduler=native["scheduler"], bounds=bounds)
    applicability = verify_current_codebase_applicability(native["index"], native["repository"],
        expected_head=native["expected_head"], verification_cid=verification.artifact_cid,
        domains=[domain], scheduler=native["scheduler"])
    projection = native["catalog"].publish(native["repository"], expected_head=native["expected_head"],
        verification_cid=verification.artifact_cid, applicability_cid=applicability.artifact_cid,
        operation_id=operation_id, scheduler=native["scheduler"])
    return verification, applicability, projection


def match(native, contract=None, domain=None, document=None, **changes):
    document_default, contract, domain = request(contract, domain)
    arguments = dict(native, intent_document=document_default if document is None else document,
        source_text=TEXT, path=PATH, contract=contract, domain=domain, statement_id=STATEMENT)
    arguments.update(changes)
    return matcher.match_conditional_codebase_intent(**arguments)


def assert_runtime_residuals(result):
    assert result["current_facts"] == result["current_behavioral_facts"] == []
    assert result["eligible_requirements"] == result["behavioral_satisfied_requirements"] == []
    assert result["runtime_refutations"] == result["removed_task_ids"] == []
    assert [row["statement_id"] for row in result["residual_requirements"]] == [STATEMENT, "runtime-goal"]
    assert all(row["status"] == "runtime_behavior_unresolved" for row in result["residual_requirements"])
    assert all(result[name] is False for name in FALSE_FIELDS)
    assert result["recorded_execution_attested"] is False
    assert "indexed_query_page" in result
    without_id = {key: value for key, value in result.items() if key != "match_cid"}
    assert cid_for_structured(without_id) == result["match_cid"]


@pytest.mark.parametrize("domain_predicates,expected", [
    (("n > 0",), "recorded_conditional_proved"),
    (("n < 0",), "recorded_conditional_refuted"),
    (("n > 0", "n < 0"), "recorded_conditional_vacuous"),
])
def test_real_requested_domain_classification_is_independent_of_parent_countermodel(native_current, domain_predicates, expected):
    contract = ContractSpec("increment", postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=domain_predicates)
    verification, _, projection = publish(native_current, contract, domain)
    assert verification.conditional_disproved
    result = match(native_current, contract, domain)
    assert result["status"] == expected
    assert result["evidence"]["parent_contract_status"] == "recorded_conditional_refuted"
    assert result["evidence"]["projection_cid"] == projection.projection_cid
    assert len(result["evidence"]["recorded_domain_checks"]["check_obligation_ids"]) == 4
    assert result["head"] == native_current["expected_head"].to_dict()
    assert result["current_root_id"] == native_current["expected_head"].snapshot_cid
    assert_runtime_residuals(result)


@pytest.mark.parametrize("premises,domain_predicates,expected", [
    (("n > 0", "n < 0"), ("True",), "recorded_conditional_vacuous"),
    (("n > 0",), ("True",), "recorded_conditional_domain_not_covered"),
])
def test_recorded_premise_consistency_and_request_coverage_gate_model_matching(native_current, premises, domain_predicates, expected):
    contract = ContractSpec("increment", preconditions=premises, postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=domain_predicates)
    verification, _, _ = publish(native_current, contract, domain)
    assert verification.conditional_proved
    result = match(native_current, contract, domain)
    assert result["status"] == expected
    assert_runtime_residuals(result)


def test_missing_projection_is_unknown_and_does_not_launch_a_solver(native_current, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_verification
    monkeypatch.setattr(codebase_verification, "_native_backends", lambda *args: pytest.fail("lookup launched prover"))
    result = match(native_current)
    assert result["status"] == "unknown" and result["evidence"] is None
    assert "no_exact_current_indexed_evidence" in result["reasons"]
    assert_runtime_residuals(result)


def test_typed_exact_query_uses_one_bounded_page_and_commits_empty_inventory(native_current, monkeypatch):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationSelector
    original = CodebaseVerificationCatalog.query_current
    observed = []

    def query(owner, *args, **kwargs):
        page = original(owner, *args, **kwargs)
        observed.append((kwargs, page))
        return page

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", query)
    monkeypatch.setattr(CodebaseVerificationCatalog, "lookup_current",
        lambda *args, **kwargs: pytest.fail("typed matcher used legacy candidate lookup"))
    result = match(native_current)
    assert len(observed) == 1
    options, page = observed[0]
    assert options["page_size"] == 2 and options["cursor"] is None
    assert type(options["selector"]) is CodebaseVerificationSelector
    selector = options["selector"]
    assert selector.path == PATH and selector.contract_id == result["query"]["contract"]["contract_id"]
    assert selector.expected_contract_cid == result["query"]["contract_cid"]
    assert selector.requested_domain_id == result["query"]["domain"]["domain_id"]
    assert selector.requested_domain_cid == result["query"]["domain_cid"]
    assert selector.verification_cid is None and selector.canonical_key_id is None
    receipt = result["indexed_query_page"]
    assert receipt == page.to_dict() and receipt["complete"] is True
    assert receipt["entries"] == [] and receipt["next_cursor"] is None
    assert receipt["start_cursor"] is None
    assert receipt["selector_cid"] == selector.cid
    assert receipt["head"] == native_current["expected_head"].to_dict()
    assert cid_for_structured({key: value for key, value in receipt.items() if key != "page_cid"}) == receipt["page_cid"]
    assert result["status"] == "unknown" and result["evidence"] is None
    assert_runtime_residuals(result)


@pytest.mark.parametrize("tamper", ["page_type", "selector", "head", "entries_type"])
def test_wrong_native_page_binding_cannot_be_consumed(native_current, monkeypatch, tamper):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    original = CodebaseVerificationCatalog.query_current

    def wrong_page(owner, *args, **kwargs):
        page = original(owner, *args, **kwargs)
        if tamper == "page_type":
            return page.to_dict()
        if tamper == "selector":
            object.__setattr__(page, "selector", replace(page.selector, path="other.py"))
        elif tamper == "head":
            object.__setattr__(page, "head", replace(page.head, generation=page.head.generation + 1))
        elif tamper == "entries_type":
            object.__setattr__(page, "entries", [])
        return page

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", wrong_page)
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError, match="native selector/head"):
        match(native_current)


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong-page@1"), ("head_cid", "wrong-head"),
    ("selector_cid", "wrong-selector"), ("page_cid", "wrong-page"),
    ("start_cursor", {"wrong": "tail"}),
])
def test_page_receipt_identity_is_checked_independently(native_current, monkeypatch, field, value):
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationQueryPage
    original = CodebaseVerificationQueryPage.to_dict

    def wrong_receipt(page):
        return {**original(page), field: value}

    monkeypatch.setattr(CodebaseVerificationQueryPage, "to_dict", wrong_receipt)
    with pytest.raises(matcher.ConditionalCodebaseEvidenceError, match="receipt lost"):
        match(native_current)


def test_unreviewed_native_goal_retains_residuals_without_catalog_query(native_current, monkeypatch):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", lambda *args, **kwargs: pytest.fail("unreviewed goal queried evidence"))
    native, contract, domain = request()
    native = replace(native, statements=(replace(native.statements[0], review_status=ReviewStatus.UNREVIEWED), native.statements[1]))
    result = match(native_current, contract, domain, native)
    assert result["status"] == "unknown" and result["evidence"] is None
    assert result["indexed_query_page"] is None
    assert_runtime_residuals(result)


def test_current_lookup_rejects_source_drift_and_never_publishes_a_successor(native_current):
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
    (native_current["repository"] / PATH).write_text("def increment(n: int) -> int:\n    return n + 1\n")
    with pytest.raises(StaleCodebaseError):
        match(native_current)
    assert native_current["index"].current(VIEW) == native_current["expected_head"]


def test_cancelled_lookup_does_not_query_evidence_or_leave_a_lease(native_current, monkeypatch):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", lambda *args, **kwargs: pytest.fail("cancelled lookup queried"))
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(LeaseCancelledError):
        match(native_current, cancel_event=cancelled)


def test_public_lookup_has_no_result_checker_or_authority_injection():
    import inspect
    parameters = inspect.signature(matcher.match_conditional_codebase_intent).parameters
    assert not {"evidence", "record", "result", "checker", "valid", "trusted", "admitted"} & parameters.keys()


def test_logical_domain_id_does_not_alias_different_requested_predicates(native_current):
    contract = ContractSpec("increment", postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=("n > 0",), domain_id="domain:owner-selection")
    publish(native_current, contract, domain)
    changed = RequestedInputDomain("increment", predicates=("n < 0",), domain_id=domain.domain_id)
    result = match(native_current, contract, changed)
    assert result["status"] == "unknown" and result["evidence"] is None
    assert "no_exact_current_indexed_evidence" in result["reasons"]
    assert_runtime_residuals(result)


def test_multiple_exact_recorded_runs_require_an_explicit_selector(native_current, monkeypatch):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.logic.ir_core.protocols import ExecutionBounds
    contract = ContractSpec("increment", postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=("n > 0",))
    first, _, projection = publish(native_current, contract, domain, "first")
    second, _, _ = publish(native_current, contract, domain, "second", bounds=ExecutionBounds(
        timeout_ms=4000, max_steps=100_000, max_memory_bytes=128 * 1024 * 1024, max_output_bytes=256 * 1024,
    ))
    assert first.artifact_cid != second.artifact_cid
    # The new per-page cap is independent of the old whole-candidate cap.
    native_current["catalog"].limits = replace(native_current["catalog"].limits, max_lookup_results=1)
    original_query = CodebaseVerificationCatalog.query_current
    calls = []

    def incomplete_first_page(owner, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs["page_size"] == 2 and kwargs["cursor"] is None
        return original_query(owner, *args, **{**kwargs, "page_size": 1})

    with monkeypatch.context() as scoped:
        scoped.setattr(CodebaseVerificationCatalog, "query_current", incomplete_first_page)
        incomplete = match(native_current, contract, domain)
    assert len(calls) == 1
    assert incomplete["status"] == "unknown" and incomplete["evidence"] is None
    assert "incomplete_exact_current_evidence_query" in incomplete["reasons"]
    assert len(incomplete["indexed_query_page"]["entries"]) == 1
    assert incomplete["indexed_query_page"]["complete"] is False
    assert incomplete["indexed_query_page"]["next_cursor"] is not None
    assert_runtime_residuals(incomplete)

    def terminal_tail(owner, *args, **kwargs):
        assert kwargs["cursor"] is None
        first_page = original_query(owner, *args, **{**kwargs, "page_size": 1})
        assert first_page.next_cursor is not None
        tail = original_query(owner, *args, **{**kwargs, "page_size": 1, "cursor": first_page.next_cursor})
        assert tail.complete and len(tail.entries) == 1 and tail.start_cursor is not None
        return tail

    with monkeypatch.context() as scoped:
        scoped.setattr(CodebaseVerificationCatalog, "query_current", terminal_tail)
        with pytest.raises(matcher.ConditionalCodebaseEvidenceError, match="native selector/head"):
            match(native_current, contract, domain)
    monkeypatch.setattr(CodebaseVerificationCatalog, "lookup_current",
        lambda *args, **kwargs: pytest.fail("typed query consulted legacy lookup cap"))
    ambiguous = match(native_current, contract, domain)
    assert ambiguous["status"] == "unknown" and ambiguous["evidence"] is None
    assert "ambiguous_exact_current_evidence_requires_explicit_selector" in ambiguous["reasons"]
    selected = match(native_current, contract, domain, verification_cid=first.artifact_cid)
    assert selected["status"] == "recorded_conditional_proved"
    assert selected["evidence"]["projection_cid"] == projection.projection_cid
    receipt = selected["indexed_query_page"]
    assert receipt["complete"] is True and receipt["next_cursor"] is None
    assert receipt["selector"]["verification_cid"] == first.artifact_cid
    assert [row["projection_cid"] for row in receipt["entries"]] == [projection.projection_cid]
    assert receipt["entries"][0]["contract_cid"] == cid_for_structured(contract.to_dict())
    assert receipt["entries"][0]["domain_cid"] == cid_for_structured(domain.to_dict())
    key_id = projection.to_dict()["contracts"][0]["canonical_keys"][0]["key_id"]
    key_selected = match(native_current, contract, domain, verification_cid=first.artifact_cid,
                         expected_key_id=key_id)
    assert key_selected["status"] == "recorded_conditional_proved"
    assert key_selected["indexed_query_page"]["selector"]["canonical_key_id"] == key_id
    assert key_id in key_selected["indexed_query_page"]["entries"][0]["canonical_key_ids"]
    assert_runtime_residuals(key_selected)
    assert_runtime_residuals(ambiguous)
    assert_runtime_residuals(selected)


def test_exact_index_lookup_replays_without_solver_or_version_processes(native_current, monkeypatch):
    contract = ContractSpec("increment", postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=("n > 0",))
    verification, _, _ = publish(native_current, contract, domain)
    original = subprocess.run

    def git_observation_only(command, *args, **kwargs):
        if Path(str(command[0])).name != "git":
            pytest.fail("lookup attempted a solver or version-probe process")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_observation_only)
    result = match(native_current, contract, domain, verification_cid=verification.artifact_cid)
    assert result["status"] == "recorded_conditional_proved"
    assert_runtime_residuals(result)


def test_source_edit_after_index_lookup_withholds_outer_match_result(native_current, monkeypatch):
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
    contract = ContractSpec("increment", postconditions=("result > 0",))
    domain = RequestedInputDomain("increment", predicates=("n > 0",))
    publish(native_current, contract, domain)
    original = CodebaseVerificationCatalog.query_current

    def lookup_then_edit(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        (native_current["repository"] / PATH).write_text("def increment(n: int) -> int:\n    return n + 1\n")
        return result

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", lookup_then_edit)
    with pytest.raises(StaleCodebaseError):
        match(native_current, contract, domain)
    assert native_current["index"].current(VIEW) == native_current["expected_head"]
