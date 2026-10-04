"""Exact conditional proof queries joined to owned finite planning observations.

Indexed solver records describe a mathematical model and remain historical.
Only the existing owner-run finite observer supplies bounded planning facts.
The private preview is application wiring after independent finite admission;
it is not a public API for caller-provided facts or a new execution grant.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import (
    CodebaseVerificationCatalog, CodebaseVerificationProjection,
)
from ipfs_datasets_py.duckdb_control.codebase_verification_queries import (
    CodebaseVerificationQueryEntry, CodebaseVerificationQueryPage,
    CodebaseVerificationQueryRequest, CodebaseVerificationSelector,
)
from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey
from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain
from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec

from ..prompt.plan_create_service import PlanCreateMode, freeze_plan_create_input_snapshot
from . import conditional_codebase_evidence as conditional
from . import finite_integer_codebase as matcher
from . import finite_integer_plan_preview as adapter
from .finite_integer_plan_service import FiniteIntegerPlanCreateService
from .plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

SCHEMA = "supervisor-finite-proof-query-plan-preview@1"
PROFILE = "finite-integer-indexed-conditional-context@1"
CLOSURE_SCHEMA = "supervisor-finite-proof-query-closure@1"
MATERIAL_KEY = "finite_proof_query_closure"
MAX_BYTES = 4 * 1024 * 1024
_REQUIREMENTS = frozenset({matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID})
_AUTHORITY = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "kernel_checked", "proof_authority", "code_proof_authority", "execution_authority",
    "completion_authority", "mutation_authority", "omission_authority",
    "admission_authority", "authoritative_cache_eligible", "behavioral_satisfaction",
    "production_admitted", "production_activation", "worker_launched",
)}
_MODEL_OFF = {"schema": "finite-proof-query-model-selection@1", "mode": "model_off",
              "model_artifact_cid": None, "codebook_cid": None}
_CLOSURE_FIELDS = frozenset({
    "schema", "profile", "head", "head_cid", "current_root_id", "finite_query",
    "contract", "contract_cid", "domain", "domain_cid", "domain_bridge",
    "model_selection", "discovery_query", "exact_query", "projection_cid", "projection",
    "verification_cid", "verification", "applicability_cid", "applicability",
    "canonical_key_membership", "conditional_evidence_summary", "conditional_status",
    "historical_execution_attested", "current_facts", "removed_task_ids", "producer",
    "authority", "closure_cid",
})


class FiniteProofQueryJoinError(ValueError):
    """The exact native source, query, proof or owned finite match differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FiniteProofQueryJoinError(message)


def _wire(value: Any) -> bytes:
    """Validate bounded inert, float-free native material before serialization."""
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= 150_000 and depth <= 48, "bounded exact proof-query material required")
        if type(item) is dict:
            _need(len(item) <= 150_000 - count and all(type(key) is str for key in item),
                  "bounded exact string proof-query keys required")
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(len(item) <= 150_000 - count, "bounded exact proof-query arrays required")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int:
            _need(item.bit_length() <= 128, "bounded exact proof-query integers required")
        else:
            _need(type(item) in {bool, type(None)}, "exact inert proof-query scalars required")
        _need(text_bytes <= MAX_BYTES, "proof-query text bound exceeded")
    try:
        raw = canonical_dag_json_bytes(value)
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise FiniteProofQueryJoinError("canonical exact proof-query JSON required") from error
    _need(len(raw) <= MAX_BYTES, "complete proof-query material exceeds four MiB")
    return raw


def _json(value: Any) -> Any:
    return json.loads(_wire(value))


def _producer() -> dict[str, str]:
    return {"module": __name__, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "selected_source_identity_only_not_execution_attestation"}


def derive_finite_proof_spec(*, intent_document, source_text: str):
    """Derive the entire authored offset contract and exact finite input model.

    The requested input list and its SMT predicate use different native schemas.
    Their explicit canonical bridge is retained in the captured closure.
    """
    query = matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    _need(query["supported"] is True and set(query["requirement_ids"]) == _REQUIREMENTS
          and len(query["requirement_ids"]) == 2, "complete supported two-clause finite instruction required")
    finite = IntegerOffsetContract.from_dict(query["contract"])
    sign = "+" if finite.offset >= 0 else "-"
    contract = ContractSpec(finite.function_name, preconditions=(),
        postconditions=(f"result == {finite.parameter} {sign} {abs(finite.offset)}",),
        contract_id="finite-offset-proof:" + query["query_cid"])
    domain = RequestedInputDomain(finite.function_name,
        predicates=(" or ".join(f"{finite.parameter} == {value}" for value in query["domain_inputs"]),),
        domain_id="finite-input-proof:" + query["domain_cid"])
    _wire(contract.to_dict())
    _wire(domain.to_dict())
    return contract, domain


def _controls(owner, remaining):
    return {"scheduler": owner.scheduler, "parent_lease": owner.parent_lease,
        "cancel_event": owner.cancel_event, "admission_timeout_seconds": min(30.0, remaining()),
        "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}


def _deadline(owner, *, request=None):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )
    seconds = owner.timeout_seconds
    if request is not None:
        seconds = min(seconds, request.budget.max_latency_ms / 1000)
    deadline = time.monotonic() + seconds
    def remaining():
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("finite proof-query join cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("finite proof-query join deadline exceeded")
        return left
    return remaining


def _page(page, *, selector, head):
    _need(type(page) is CodebaseVerificationQueryPage and type(page.selector) is CodebaseVerificationSelector
          and page.selector.to_dict() == selector.to_dict() and page.head.to_dict() == head.to_dict()
          and type(page.entries) is tuple and len(page.entries) == 1
          and type(page.entries[0]) is CodebaseVerificationQueryEntry
          and page.complete is True and page.start_cursor is None and page.next_cursor is None,
          "complete exact singleton native proof-query page required")
    page.__post_init__()
    body = _json(page.to_dict())
    _need(body["page_cid"] == cid_for_structured({key: value for key, value in body.items() if key != "page_cid"})
          and body["complete"] is True and body["start_cursor"] is None and body["next_cursor"] is None
          and body["authority"]["historical_conditional_evidence"] is True
          and all(body["authority"][name] is False for name in (
              "kernel_checked", "source_runtime_semantics_verified", "behavioral_satisfaction",
              "authoritative_cache_eligible", "admission_authority", "completion_authority")),
          "native proof-query page identity or authority differs")
    return body


def _key_membership(projection, verification, applicability, contract_id):
    selected = [item for item in projection["contracts"] if item["contract_id"] == contract_id]
    _need(len(selected) == 1, "complete unique selected contract projection required")
    selected = selected[0]
    obligations = verification["pipeline_result"]["obligation_results"]
    _need(len(obligations) == len(verification["canonical_keys"]) > 0
          and all(item["vc_obligation"]["parent_contract_id"] == contract_id for item in obligations),
          "complete selected verification obligation/key population required")
    expected_verification = []
    for item, key in zip(obligations, verification["canonical_keys"]):
        native = CanonicalProofCacheKey.from_dict(key)
        _need(_wire(native.to_dict()) == _wire(key), "complete canonical verification key differs")
        expected_verification.append({"key_id": native.key_id, "key": native.to_dict(),
                                     "obligation_id": item["vc_obligation"]["obligation_id"]})
    checks, keys = applicability["checks"], applicability["canonical_keys"]
    _need(len(checks) == len(keys) == 4, "all four applicability obligations and keys required")
    expected_applicability = []
    for check, key in zip(checks, keys):
        native = CanonicalProofCacheKey.from_dict(key)
        _need(_wire(native.to_dict()) == _wire(key) and check["parent_contract_id"] == contract_id,
              "complete canonical applicability key or selected contract differs")
        expected_applicability.append({"key_id": native.key_id, "key": native.to_dict(),
            "kind": check["kind"], "obligation_id": check["obligation"]["smt_obligation"]["obligation_id"]})
    _need(_wire(selected["canonical_keys"]) == _wire(expected_verification)
          and _wire(selected["applicability_keys"]) == _wire(expected_applicability),
          "normalized proof-key membership omits or changes a native obligation")
    membership = ([{"phase": "verification", **item} for item in expected_verification]
                  + [{"phase": "applicability", **item} for item in expected_applicability])
    _need(len({item["key_id"] for item in membership}) == len(membership),
          "complete proof-key membership contains an ambiguous duplicate")
    return membership


def _require_frozen_proof_inventory(*, owner, verification_catalog, closure, checkpoint):
    """Close native inventory and immutable proof bytes after observer callbacks.

    This private receiving fence invokes no source observer, policy observer,
    solver, fitting or inference. Only the owner's budget/cancellation checkpoint
    runs during the bounded native inventory validation and immutable reads.
    It does not attest process origin or lock the working checkout.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner and type(verification_catalog) is CodebaseVerificationCatalog
          and verification_catalog.index is owner.index
          and verification_catalog.store is owner.index.ingestor.store
          and verification_catalog.artifacts is owner.index.artifacts
          and type(owner.index.artifacts) is ImmutableCAS and callable(checkpoint),
          "same exact native source/CAS/proof owner and budget checkpoint required")
    value = _json(closure)
    head = owner.expected_head
    head_cid = cid_for_structured(head.to_dict())
    _need(type(value) is dict and set(value) == _CLOSURE_FIELDS
          and value["schema"] == CLOSURE_SCHEMA and value["profile"] == PROFILE
          and value["closure_cid"] == cid_for_structured(
              {key: item for key, item in value.items() if key != "closure_cid"})
          and _wire(value["head"]) == _wire(head.to_dict()) and value["head_cid"] == head_cid
          and value["current_root_id"] == head.snapshot_cid
          and _wire(value["model_selection"]) == _wire(_MODEL_OFF)
          and _wire(value["authority"]) == _wire(_AUTHORITY),
          "complete frozen proof closure or exact source/authority identity differs")
    pages = []
    for phase in ("discovery_query", "exact_query"):
        wrapper = value[phase]
        _need(type(wrapper) is dict and set(wrapper) == {"request", "page"},
              "complete frozen native query request/page required")
        request = CodebaseVerificationQueryRequest.from_dict(wrapper["request"])
        page = wrapper["page"]
        _need(type(page) is dict and request.page_size == 2 and request.cursor is None
              and page.get("schema") == "codebase-verification-query-page@1"
              and page.get("page_cid") == cid_for_structured(
                  {key: item for key, item in page.items() if key != "page_cid"})
              and _wire(page.get("head")) == _wire(head.to_dict()) and page.get("head_cid") == head_cid
              and _wire(page.get("selector")) == _wire(request.selector.to_dict())
              and page.get("selector_cid") == request.selector.cid
              and page.get("complete") is True and page.get("start_cursor") is None
              and page.get("next_cursor") is None and type(page.get("epoch")) is int
              and type(page.get("entries")) is list and len(page["entries"]) == 1,
              "frozen proof query lost its exact complete source/selector/page binding")
        pages.append(page)
    _need(pages[0]["inventory_cid"] == pages[1]["inventory_cid"]
          and pages[0]["epoch"] == pages[1]["epoch"]
          and _wire(pages[0]["entries"]) == _wire(pages[1]["entries"]),
          "frozen proof-query discovery and exact inventory bindings differ")
    expected_keys = _key_membership(value["projection"], value["verification"],
        value["applicability"], value["contract"]["contract_id"])
    _need(_wire(value["canonical_key_membership"]) == _wire(expected_keys),
          "frozen complete proof-key membership differs")
    checkpoint()
    with verification_catalog.store._lock, verification_catalog.store._transaction():
        verification_catalog._ensure_owner()
        verification_catalog._check_schema()
        verification_catalog._check_counts()
        verification_catalog._catalog._check_schema()
        _need(verification_catalog._catalog._current(head.repository_id) == head,
              "source head changed after the final proof-query callback")
        inventory = verification_catalog._queries.validate_inventory(
            expected_head=head, checkpoint=checkpoint)
        _need(inventory.head_cid == head_cid
              and inventory.inventory_cid == pages[0]["inventory_cid"]
              and inventory.epoch == pages[0]["epoch"],
              "proof-query inventory changed after the final callback")
        for identity, body in ((value["projection_cid"], value["projection"]),
                               (value["verification_cid"], value["verification"]),
                               (value["applicability_cid"], value["applicability"])):
            checkpoint()
            _need(cid_for_structured(body) == identity
                  and _wire(owner.index.artifacts.get(identity)) == _wire(body),
                  "retained proof closure bytes changed after the final callback")
        source_binding = value["verification"]["source_binding"]
        source = owner.index.artifacts.get_bytes(source_binding["entry"]["source_cid"])
        _need(_wire(source_binding["head"]) == _wire(head.to_dict())
              and hashlib.sha256(source).hexdigest() == source_binding["content_sha256"],
              "retained captured source changed at the final proof-query boundary")
        checkpoint()


def capture_current_finite_proof_query(*, owner, verification_catalog, intent_document, source_text: str):
    """Capture exact current query records and complete historical proof bodies.

    This performs no solver execution, training, inference or publication. It
    cannot turn recorded conditional evidence into a current behavioral fact.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner and owner.memory_mb >= 512
          and type(verification_catalog) is CodebaseVerificationCatalog
          and verification_catalog.index is owner.index
          and verification_catalog.store is owner.index.ingestor.store
          and verification_catalog.artifacts is owner.index.artifacts
          and type(owner.index.artifacts) is ImmutableCAS,
          "same exact native source/CAS/proof owner and bounded reservation required")
    query = _json(matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text))
    contract, domain = derive_finite_proof_spec(intent_document=intent_document, source_text=source_text)
    head, remaining, producer = owner.expected_head, _deadline(owner), _producer()
    contract_cid, domain_cid = cid_for_structured(contract.to_dict()), cid_for_structured(domain.to_dict())
    selector = CodebaseVerificationSelector(path=query["contract"]["path"], contract_id=contract.contract_id,
        expected_contract_cid=contract_cid, requested_domain_id=domain.domain_id, requested_domain_cid=domain_cid)
    with structural_codebase_context(owner.index, owner.repository,
            repository_id=head.repository_id, expected_head=head, **_controls(owner, remaining)):
        first = verification_catalog.query_current(owner.repository, expected_head=head, selector=selector,
            page_size=2, cursor=None, **_controls(owner, remaining))
        first_body = _page(first, selector=selector, head=head)
        projection = first.entries[0].projection
        _need(type(projection) is CodebaseVerificationProjection and projection.applicability is not None,
              "native linked verification and complete applicability projection required")
        projected = _json(projection.to_dict())
        verification = _json(projection.verification.to_dict())
        applicability = _json(projection.applicability.to_dict())
        _need(verification["requested_contracts"] == [contract.to_dict()]
              and applicability["requested_domains"] == [domain.to_dict()]
              and verification["execution_stage"] == "native_smt",
              "entire indexed authored contract/domain or native execution profile differs")
        request = {"path": query["contract"]["path"], "contract": contract.to_dict(),
            "contract_cid": contract_cid, "domain": domain.to_dict(), "domain_cid": domain_cid}
        status, reasons, summary = conditional._indexed_evidence(projection, head, request)
        classes = [item["differential"]["classification"] for item in applicability["checks"]]
        _need(classes[:3] == ["agree_satisfiable", "agree_satisfiable", "agree_proved"]
              and classes[3:] in (["agree_proved"], ["agree_disproved"])
              and status in {"recorded_conditional_proved", "recorded_conditional_refuted"} and not reasons,
              "satisfiable premises/domain, full domain coverage and conclusive conditional property required")
        membership = _key_membership(projected, verification, applicability, contract.contract_id)
        pinned_selector = CodebaseVerificationSelector(path=selector.path, contract_id=contract.contract_id,
            expected_contract_cid=contract_cid, verification_cid=projection.verification.artifact_cid,
            canonical_key_id=membership[0]["key_id"], requested_domain_id=domain.domain_id,
            requested_domain_cid=domain_cid)
        exact = verification_catalog.query_current(owner.repository, expected_head=head, selector=pinned_selector,
            page_size=2, cursor=None, **_controls(owner, remaining))
        exact_body = _page(exact, selector=pinned_selector, head=head)
        _need(first.inventory_cid == exact.inventory_cid and first.epoch == exact.epoch
              and first.head == exact.head and first.entries[0].entry_id == exact.entries[0].entry_id
              and _wire(exact.entries[0].projection.to_dict()) == _wire(projected)
              and _wire(exact.entries[0].projection.verification.to_dict()) == _wire(verification)
              and _wire(exact.entries[0].projection.applicability.to_dict()) == _wire(applicability),
              "source generation, inventory epoch or proof closure changed between exact queries")
        bridge = {"schema": "finite-input-to-conditional-domain-bridge@1",
            "finite_query_cid": query["query_cid"], "finite_domain_cid": query["domain_cid"],
            "domain_inputs": query["domain_inputs"], "parameter": query["contract"]["parameter"],
            "requested_domain_cid": domain_cid, "predicates": list(domain.predicates)}
        result = {"schema": CLOSURE_SCHEMA, "profile": PROFILE, "head": head.to_dict(),
            "head_cid": cid_for_structured(head.to_dict()), "current_root_id": head.snapshot_cid,
            "finite_query": query, "contract": contract.to_dict(), "contract_cid": contract_cid,
            "domain": domain.to_dict(), "domain_cid": domain_cid, "domain_bridge": bridge,
            "model_selection": dict(_MODEL_OFF),
            "discovery_query": {"request": CodebaseVerificationQueryRequest(selector, 2).to_dict(), "page": first_body},
            "exact_query": {"request": CodebaseVerificationQueryRequest(pinned_selector, 2).to_dict(), "page": exact_body},
            "projection_cid": projection.projection_cid, "projection": projected,
            "verification_cid": projection.verification.artifact_cid, "verification": verification,
            "applicability_cid": projection.applicability.artifact_cid, "applicability": applicability,
            "canonical_key_membership": membership, "conditional_evidence_summary": summary,
            "conditional_status": status, "historical_execution_attested": False,
            "current_facts": [], "removed_task_ids": [], "producer": producer, "authority": dict(_AUTHORITY)}
        result["closure_cid"] = cid_for_structured(result)
        result = _json(result)
        remaining()
    # The final source observation is another cooperative callback boundary.
    # Close with native inventory and immutable bytes, without another source
    # observer that could change those records after their final comparison.
    _require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
        closure=result, checkpoint=remaining)
    _need(_producer() == producer, "proof-query producer source changed during current capture")
    remaining()
    return result


def _preview_owned_finite_proof_join(*, owner, request, intent_document, source_text,
        operation_catalog, match, verification_catalog, policy_observer):
    """Private orchestration for an independently validated owner-run match.

    Only the finite admission owner calls this after its independent evidence
    reconstruction. No public wrapper accepts an arbitrary match or facts.
    Frozen results contain no elapsed times or timestamps and replay exactly.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner and owner.memory_mb >= 1024
          and type(request) is PlanCreateRequest and type(intent_document) is IntentIRDocument
          and type(operation_catalog) is adapter.FiniteIntegerOperationCatalog
          and callable(policy_observer) and request.budget.max_model_calls == 0
          and request.budget.max_tasks >= 2 and request.budget.max_goals >= 2,
          "exact owned finite request, complete operation catalog and model-off budget required")
    operation_catalog.__post_init__()
    owned = _json(match)
    match_bytes = _wire(owned)
    query = matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    finite_contract = IntegerOffsetContract.from_dict(query["contract"])
    _need(owned.get("query") == query and owned.get("head") == owner.expected_head.to_dict()
          and owned.get("current_root_id") == owner.expected_head.snapshot_cid
          and owned.get("match_cid") == cid_for_structured({key: value for key, value in owned.items() if key != "match_cid"})
          and owned.get("removed_task_ids") == [] and all(owned.get(name) is False for name in matcher._AUTHORITY),
          "independently validated exact owned finite match required")
    observation = owned.get("observation")
    _need(type(observation) is dict and observation.get("status") == "observed", "fresh complete finite observation required")
    output = Path(observation["output"])
    _need(output.is_absolute() and output.resolve(strict=True) == output and output.is_dir() and not output.is_symlink(),
          "original canonical owner observation directory required")
    _need(request.repository_root == str(owner.repository)
          and request.repository_id == owner.expected_head.repository_id
          and request.roots.repository_root_cid == owner.expected_head.snapshot_cid
          and request.roots.dirty_worktree_root == owner.expected_head.snapshot_cid
          and request.prompt_source_cid == adapter.finite_integer_prompt_cid(source_text)
          and request.roots.intent_ir_root == adapter.finite_integer_intent_cid(intent_document)
          and request.roots.capability_catalog_root == operation_catalog.cid
          and request.scope_paths == (finite_contract.path,)
          and all((item.path, item.function_name, item.parameter) ==
                  (finite_contract.path, finite_contract.function_name, finite_contract.parameter)
                  for item in operation_catalog.operations), "exact request/source/intent/operation roots required")
    remaining, producer = _deadline(owner, request=request), _producer()
    closure = capture_current_finite_proof_query(owner=replace(owner, timeout_seconds=remaining()),
        verification_catalog=verification_catalog, intent_document=intent_document, source_text=source_text)
    closure_bytes = _wire(closure)
    _need(closure["verification"]["source_binding"]["entry"]["source_cid"] == owned["source_cid"]
          and closure["domain_bridge"]["finite_domain_cid"] == owned["domain_cid"]
          and closure["domain_bridge"]["domain_inputs"] == owned["domain_inputs"]
          and (closure["applicability"]["checks"][3]["differential"]["classification"] == "agree_proved")
              is observation["offset_clause_satisfied"],
          "indexed exact source/domain/property differs from fresh finite observations")

    def require_current():
        _need(_wire(owned) == match_bytes and _producer() == producer,
              "owned finite match or proof-join producer changed during planning")
        matcher._check_observation(observation=owned["observation"], index=owner.index,
            head=owner.expected_head, contract=finite_contract, inputs=query["domain_inputs"],
            tool_policy=observation["tool_policy"], output=output)
        current = capture_current_finite_proof_query(owner=replace(owner, timeout_seconds=remaining()),
            verification_catalog=verification_catalog, intent_document=intent_document, source_text=source_text)
        _need(_wire(current) == closure_bytes, "complete current proof-query closure changed during planning")
        matcher._check_observation(observation=owned["observation"], index=owner.index,
            head=owner.expected_head, contract=finite_contract, inputs=query["domain_inputs"],
            tool_policy=observation["tool_policy"], output=output)
        _require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
            closure=closure, checkpoint=remaining)
        remaining()

    with structural_codebase_context(owner.index, owner.repository,
            repository_id=owner.expected_head.repository_id, expected_head=owner.expected_head,
            **_controls(owner, remaining)) as context:
        _need(request.roots.program_root == context.semantic_state_cid
              and owned["structural_context"] == context.to_dict(), "exact native structural program root required")
        require_current()
        materials, bindings = adapter._materials(owned, operation_catalog, context)
        materials = replace(materials, extra={**materials.extra,
            MATERIAL_KEY: closure, MATERIAL_KEY + "_cid": closure["closure_cid"],
            "finite_proof_query_profile": PROFILE})
        snapshot = freeze_plan_create_input_snapshot(request, materials=materials)

        def observe_roots(value):
            _need(value == request, "service changed the exact finite proof-query request")
            with structural_codebase_context(owner.index, owner.repository,
                    repository_id=owner.expected_head.repository_id, expected_head=owner.expected_head,
                    **_controls(owner, remaining)) as current:
                _need(current == context, "current source context changed during finite proof-query planning")
                require_current()
                roots = policy_observer(request)
                _need(type(roots) is PlanAuthorityRoots, "complete independently observed policy roots required")
                roots.require_current(request.roots)
                require_current()
            require_current()
            return roots

        service = FiniteIntegerPlanCreateService(root_observer=observe_roots, operation_bindings=bindings)
        preview = service.preview_create(request, mode=PlanCreateMode.DETERMINISTIC, materials=materials)
        _need(preview.input_snapshot_cid == snapshot.snapshot_cid and preview.read_only is True
              and preview.admitted is False and not preview.wrote_effects,
              "exact frozen read-only finite proof-query preview required")
        stages = {item.stage.value: item for item in preview.stage_results}
        _need(all(name in stages and stages[name].passed
                  for name in ("scan", "query", "evidence", "obligation", "candidate", "critique")),
              "finite proof-query native planning stage failed closed")
        diagnostics = service.diagnostics
        require_current()
    require_current()
    def record(value):
        return value.to_dict() if callable(getattr(value, "to_dict", None)) else value
    candidate = record(diagnostics["candidate_plan"])
    _need(type(candidate) is dict and type(candidate.get("tasks")) is list,
          "native finite candidate and explicit selected task list required")
    ledger = [{**row, "predicate_id": owned["typed_intent"]["metadata"]["requirement_predicate_ids"][row["statement_id"]],
        "fact_ids": [fact["fact_id"] for fact in owned["current_facts"]
                     if fact["predicate"]["predicate_id"] == row["predicate_id"]]}
        for row in owned["clause_results"]]
    _need(len(ledger) == 2 and {row["statement_id"] for row in ledger} == _REQUIREMENTS,
          "complete two-clause fact/residual requirement ledger required")
    result = {"schema": SCHEMA, "profile": PROFILE, "proof_query_closure": closure,
        "proof_query_closure_cid": closure["closure_cid"], "match": owned,
        "input_snapshot": snapshot.to_dict(), "preview": preview.to_dict(),
        "operation_catalog": operation_catalog.to_dict(), "operation_catalog_cid": operation_catalog.cid,
        "requirement_ledger": ledger, "obligation_graph": record(diagnostics["obligation_graph"]),
        "portfolio": record(diagnostics["portfolio"]), "candidate_plan": candidate,
        "critique": record(diagnostics["critique"]), "critic_evidence": record(diagnostics["critic_evidence"]),
        "execution_plan": record(diagnostics["execution_plan"]),
        "planner_status": "selected" if candidate["tasks"] else "already_complete_in_finite_domain",
        "declared_task_requirement_ids": {item.task_id: item.requirement_id for item in operation_catalog.operations},
        "selected_task_ids": [item["task_id"] for item in candidate["tasks"]],
        "current_facts_count": len(owned["current_facts"]), "removed_task_ids": [],
        "scope": "explicit_finite_domain_only_with_historical_conditional_proof_context",
        "model_selection": dict(_MODEL_OFF), "model_calls": 0, "training_steps": 0,
        "solver_calls": 0, "producer": producer, "authority": dict(_AUTHORITY), **_AUTHORITY}
    result["result_cid"] = cid_for_structured(result)
    result = _json(result)
    _require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
        closure=result["proof_query_closure"], checkpoint=remaining)
    _need(_producer() == producer, "proof-join producer source changed before returning frozen material")
    remaining()
    return result


__all__ = ["SCHEMA", "PROFILE", "CLOSURE_SCHEMA", "MATERIAL_KEY", "MAX_BYTES",
           "FiniteProofQueryJoinError", "derive_finite_proof_spec", "capture_current_finite_proof_query"]
