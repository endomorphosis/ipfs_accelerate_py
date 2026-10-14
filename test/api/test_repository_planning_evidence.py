"""Complete native requirement ledgers and scoped security stay source-bound.

Only authored source and native deterministic owners are used. No fixture
source, provider, solver, container, or external publication is executed.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import repository_planning_evidence as evidence
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
    build_intent_requirement_contract, validate_intent_requirement_contract,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from test.api.test_repository_source_index import scheduler  # noqa: F401
from test.api.test_intent_plan_coverage import _contract, _ledger, _spec
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401


def mixed_contract():
    source, ledger = _ledger(("required", "permitted"), ("answer", "logs"), native_context=True)
    specs = [_spec(row["requirement_id"]) for row in ledger["requirements"]
             if row["kind"] == "goal" and row["representation"] == "native_statement"
             and row["compound"] is None and row["modality"] == "required"]
    return source, build_intent_requirement_contract(source_path="instruction.txt",
        ledger=ledger, requirements=specs, source_text=source)


@pytest.fixture
def current(tmp_path, scheduler):
    source, contract = mixed_contract()
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "instruction.txt").write_text(source)
    (repository / "answer.py").write_text("def answer():\n    return 1\n")
    with RepositorySourceSession(repository, tmp_path / "session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        yield repository, session, contract
    assert scheduler.snapshot()["active_lease_count"] == 0


def build(current, **arguments):
    _repository, session, contract = current
    return evidence.build_repository_planning_evidence(session=session, generation=1,
        intent_contract=arguments.pop("intent_contract", contract), **arguments)


def test_complete_ledger_retains_nonmandatory_and_unsupported_rows(current):
    repository, session, contract = current
    refs = build(current)
    assert len(refs) == 3
    intent, security, ui = refs
    assert json.loads(intent.summary) == contract
    rows = contract["ledger"]["requirements"]
    assert any(row["modality"] == "permitted" for row in rows)
    assert any(row["kind"] != "goal" or row["representation"] != "native_statement" for row in rows)
    assert set(intent.coverage_ids) == {"planning-domain:intent_ir", *(row["requirement_id"] for row in rows)}
    status = session.status_summary()
    for ref in refs:
        assert ref.required
        assert ref.repository_id == status["datasets_head"]["repository_id"]
        assert ref.tree_id == "sha256:" + status["source_tree_sha256"]
        assert ref.byte_count == len(ref.summary.encode())
        assert ref.referenced_content_id == "sha256:" + hashlib.sha256(ref.summary.encode()).hexdigest()
        assert all(ref.metadata[key] is False for key in evidence._FALSE)
    assert intent.metadata["source_sha256"] == hashlib.sha256((repository / "instruction.txt").read_bytes()).hexdigest()
    for ref, domain in ((security, "security_ir"), (ui, "ui_ux_ir")):
        body = json.loads(ref.summary)
        assert body["status"] == "unavailable"
        assert body["domain"] == domain
        assert body["source_semantics_verified"] is False
        assert all(body[key] is False for key in evidence._FALSE)


@pytest.mark.parametrize("mutation", ["grounding", "ledger_row", "source_unit", "path"])
def test_incomplete_or_forged_required_contract_never_becomes_context(current, mutation):
    contract = deepcopy(current[2])
    if mutation == "grounding":
        # Same exact source text, with a separately reviewed supported native
        # interpretation. The mixed control-flow ledger is wholly unsupported
        # and legitimately has no task groundings to remove.
        contract = _contract(("required", "permitted"), ("answer", "logs"))
        assert contract["requirements"]
        contract["requirements"] = []
    elif mutation == "ledger_row":
        contract["ledger"]["requirements"].pop()
    elif mutation == "source_unit":
        contract["ledger"]["source_units"][0]["text"] = "replacement"
    else:
        contract["source_path"] = "answer.py"
    with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="invalid_planning_intent_contract"):
        build(current, intent_contract=contract)
    assert current[1].status_summary()["source_current"] is True


@pytest.mark.parametrize("bound", ["rows", "body", "values", "depth"])
def test_overbound_required_population_is_refused_without_truncation(current, monkeypatch, bound):
    setting, maximum = {
        "rows": ("MAX_REQUIREMENT_ROWS", 1), "body": ("MAX_BODY_BYTES", 100),
        "values": ("MAX_JSON_VALUES", 3), "depth": ("MAX_JSON_DEPTH", 2),
    }[bound]
    original = deepcopy(current[2])
    monkeypatch.setattr(evidence, setting, maximum)
    with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="limit"):
        build(current)
    assert current[2] == original
    assert current[1].status_summary()["source_current"] is True


@pytest.mark.parametrize("drift", ["source", "generation", "during_validation"])
def test_required_domain_evidence_cannot_cross_currentness_fences(current, monkeypatch, drift):
    repository, session, _contract = current
    if drift == "generation":
        session.refresh(generation=1)
    elif drift == "source":
        (repository / "answer.py").write_text("def answer(): return 2\n")
    else:
        def changed(contract, **arguments):
            result = validate_intent_requirement_contract(contract, **arguments)
            (repository / "answer.py").write_text("def answer(): return 3\n")
            return result
        monkeypatch.setattr(evidence, "validate_intent_requirement_contract", changed)
    with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="stale|binding"):
        build(current)


def test_shared_deadline_includes_native_intent_validation(current, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(evidence, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def slow(contract, **arguments):
        result = validate_intent_requirement_contract(contract, **arguments)
        clock[0] = 31.0
        return result

    monkeypatch.setattr(evidence, "validate_intent_requirement_contract", slow)
    with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="timeout"):
        build(current)
    assert current[1].status_summary()["source_current"] is True


def test_resource_cancellation_refuses_before_source_read_and_recovers(current, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    @contextmanager
    def cancelled(**arguments):
        assert arguments["memory_mb"] == 256
        assert arguments["timeout_seconds"] == 10.0
        yield SimpleNamespace(cancelled=True)

    with monkeypatch.context() as scoped:
        scoped.setattr(codebase_resources, "acquire_codebase_resources", cancelled)
        scoped.setattr(current[1], "read", lambda **_kw: pytest.fail("cancelled lease must precede source read"))
        with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="cancelled"):
            build(current)
    assert len(build(current)) == 3


@pytest.mark.parametrize("arguments", [{"security_scope": object()}, {"security_analyses": []}])
def test_partial_security_owner_inputs_are_not_downgraded_to_unavailable(current, arguments):
    with pytest.raises(evidence.RepositoryPlanningEvidenceError, match="incomplete_planning_security"):
        build(current, **arguments)


def security_case(scenario, tmp_path, source):
    from test.api.test_doctor_task_workflow import _prepare
    from test.api.test_doctor_header_contracts import PROTOCOL
    from test.api.test_intent_requirement_admission import _reviewed_contract
    from ipfs_accelerate_py.agent_supervisor.analysis.doctor_header_contracts import analyze_http_header_contracts
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_scoped_analysis import build_scoped_doctor_analysis

    inputs = _prepare(scenario, tmp_path, text=source, external_consumer=True)
    scope = build_scoped_doctor_analysis(repository=scenario["repository"],
        admission=inputs["admission"], task_cid=inputs["task_cid"])
    analysis = analyze_http_header_contracts(source, protocol=PROTOCOL)
    rows = [{"path": "answer.py", "status": analysis.status,
        "reason_codes": list(analysis.reason_codes), "source_sha256": analysis.source_sha256,
        "contracts": list(analysis.contracts), "open_frontiers": list(analysis.open_frontiers)}]
    return scope, rows, _reviewed_contract(scenario, confidence=0.75)


@pytest.mark.parametrize("supported", [True, False])
def test_full_native_security_report_preserves_unmodeled_sources_and_no_proof(scenario, tmp_path, scheduler, supported):
    from test.api.test_doctor_header_contracts import PROGRAM
    from test.api.test_doctor_task_workflow import SOURCE
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_security_ir import compile_header_security_ir

    scope, analyses, contract = security_case(scenario, tmp_path, PROGRAM if supported else SOURCE)
    expected = compile_header_security_ir(scoped=scope, analyses=analyses).report
    with RepositorySourceSession(scenario["repository"], tmp_path / "evidence-session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        intent, security, ui = evidence.build_repository_planning_evidence(
            session=session, generation=1, intent_contract=contract,
            security_scope=scope, security_analyses=analyses)
        assert json.loads(intent.summary) == contract
        native = json.loads(intent.summary)["ledger"]["source_report"]["candidates"][0]["candidate_intent_ir"]
        assert native["statements"][0]["confidence"] == 0.75
        actual = json.loads(security.summary)
        assert actual == expected
        assert actual["unmodeled_paths"]
        assert "test_answer.py" in actual["unmodeled_paths"]
        assert actual["open_frontiers"]
        assert actual["status"] == ("compiled_local_declarations" if supported else "unsupported")
        assert all(actual[key] is False for key in ("proof_created", "whole_program_proved", "solver_executed"))
        if not supported:
            assert actual["unsupported_analyses"]
            assert actual["claim_count"] == 0
        assert json.loads(ui.summary)["status"] == "unavailable"


@pytest.mark.parametrize("drift", ["stale_scope", "foreign_scope", "forged_analysis"])
def test_saved_or_foreign_security_evidence_cannot_bypass_native_currentness(scenario, tmp_path, scheduler, drift):
    from test.api.test_doctor_header_contracts import PROGRAM

    scope, analyses, contract = security_case(scenario, tmp_path, PROGRAM)
    repository = scenario["repository"]
    if drift == "stale_scope":
        (repository / "answer.py").write_text(PROGRAM + "# external source change\n")
    elif drift == "foreign_scope":
        repository = tmp_path / "other-repository"
        repository.mkdir()
        for path in ("answer.py", "test_answer.py"):
            (repository / path).write_bytes((scenario["repository"] / path).read_bytes())
    else:
        analyses[0]["contracts"][0]["forbidden_codepoints"] = [10]
    with RepositorySourceSession(repository, tmp_path / "evidence-session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        with pytest.raises(ValueError):
            evidence.build_repository_planning_evidence(session=session, generation=1,
                intent_contract=contract, security_scope=scope, security_analyses=analyses)
        assert session.status_summary()["source_current"] is True
