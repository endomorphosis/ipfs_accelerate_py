"""Independent acceptance boundaries for the reviewed integer repository profile.

Native tools and durable captured source establish the positive claims. Injected
failures below are adversarial controls, not evidence of native tool execution.
"""
from copy import deepcopy
from dataclasses import replace
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision import _index
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as match
from ipfs_datasets_py.logic.backends.codebase_process import BoundedToolRunner, ToolRunLimits, run_bounded_stdin_tool
from ipfs_datasets_py.logic.software_contracts import codebase_integer_profile as profile
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as observation
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_verification import codebase_source_adapters as adapter
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import get_global_resource_scheduler


SOURCE = b"def increment(n: int) -> int:\n    return n + 1\n"
DOMAIN = [-2, -1, 0, 1, 2]
CONTRACT = profile.IntegerOffsetContract("calc.py", "increment", "n", 2)
LEAN = Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean")


def _text(domain=DOMAIN, path="calc.py"):
    tail = " for inputs " + json.dumps(domain, separators=(",", ":")) + "."
    prefix = f"Under python-integer-offset-finite@1, {path}::increment(n) must return "
    return prefix + "an exact int" + tail + "\n" + prefix + "n + 2" + tail


def _git(root, *args):
    return subprocess.check_output(["/usr/bin/git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def captured(tmp_path):
    repository = tmp_path / "repository"; repository.mkdir()
    (repository / "calc.py").write_bytes(SOURCE)
    (repository / "decoy.py").write_bytes(SOURCE.replace(b"n + 1", b"n + 2"))
    for args in (("init", "-q"), ("config", "user.name", "Profile acceptance"),
                 ("config", "user.email", "qualification@example.invalid"),
                 ("add", "."), ("commit", "-qm", "Exact authored sources")):
        _git(repository, *args)
    scheduler = get_global_resource_scheduler()
    database = tmp_path / "source.duckdb"
    cx = duckdb.connect(str(database), config={"threads": 1, "memory_limit": "64MB"})
    index = _index(cx, tmp_path / "artifacts")
    head = index.prepare_current(repository, repository_id="repository:profile-acceptance",
        operation_id="initial", expected_head=None, scheduler=scheduler).head
    # All tests consume a reopened durable owner, not an in-memory surrogate.
    cx.close()
    cx = duckdb.connect(str(database), config={"threads": 1, "memory_limit": "64MB"})
    index = _index(cx, tmp_path / "artifacts")
    assert index.current(head.repository_id) == head
    tools = observation.seal_finite_integer_tools(python_executable=Path(sys.executable).resolve(), lean_executable=LEAN)
    context = dict(root=tmp_path, repository=repository, index=index, head=head,
                   scheduler=scheduler, tools=tools, sequence=0)
    yield context
    assert not [row for row in scheduler.active_leases() if row["owner_pid"] == os.getpid()]
    cx.close()


def _save(context, name, value):
    (context["root"] / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _source(context):
    manifest = context["index"].load(context["head"].manifest_cid)
    entry = next(row for row in manifest.snapshot.entries if row.path == "calc.py")
    return context["index"].artifacts.get_bytes(entry.source_cid)


def _capture(context, source):
    (context["repository"] / "calc.py").write_bytes(source)
    context["sequence"] += 1
    context["head"] = context["index"].prepare_current(context["repository"],
        repository_id=context["head"].repository_id, operation_id="capture:" + str(context["sequence"]),
        expected_head=context["head"], scheduler=context["scheduler"]).head
    assert _source(context) == source


def _compiled(context):
    return profile.compile_integer_offset(_source(context), CONTRACT,
        revision="snapshot:" + context["head"].snapshot_cid)


def _match(context, *, text=None, **changes):
    context["sequence"] += 1
    text = _text() if text is None else text
    args = dict(index=context["index"], repository=context["repository"],
        repository_id=context["head"].repository_id, expected_head=context["head"],
        intent_document=match.build_finite_integer_intent(text), source_text=text,
        output=context["root"] / ("observation-" + str(context["sequence"])),
        tool_policy=context["tools"], scheduler=context["scheduler"], timeout_seconds=30)
    args.update(changes)
    result = match.match_finite_integer_intent(**args)
    _save(context, "match-" + str(context["sequence"]), result)
    return result


def _native_smt(context, solver, text):
    discovered = shutil.which(solver)
    assert discovered, f"native {solver} required; no skip or stand-in"
    executable, _ = profile._native_executable(discovered)
    args = ["-in", "-smt2"] if solver == "z3" else ["--lang=smt2"]
    with context["scheduler"].acquire("validation", cpu_slots=1, memory_mb=256,
            child_process_slots=1, timeout=30, request_id="acceptance:premise-and-domain"):
        result = run_bounded_stdin_tool([executable, *args], text, runner=BoundedToolRunner(),
            limits=ToolRunLimits(timeout_seconds=10, cpu_seconds=10,
                memory_bytes=256 * 1024**2, resident_memory_bytes=256 * 1024**2,
                max_input_bytes=65536, max_output_bytes=65536, max_workspace_bytes=65536))
    assert result.returncode == 0 and not result.error and not result.timed_out
    assert result.workspace_cleaned and not result.resource_exhausted
    return dict(executable=executable, executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                script=text, result=result.to_dict())


def test_real_premise_satisfiability_total_integer_domain_and_source_applicable_counterexample(captured):
    compiled = _compiled(captured)
    obligation = compiled.pipeline.obligation_results[0].smt_obligation
    assert len(obligation.assumptions) == 1 and obligation.assumptions[0].name == "body_return_0"
    assert not compiled.pipeline.obligation_results[0].vc_obligation.assumption_expression_ids
    assert list(compiled.to_dict()["assumptions"]) == list(profile.ASSUMPTIONS)
    body = obligation.assumptions[0].formula.render()
    goal = obligation.goal.render()
    parameters = [row.name for row in obligation.functions if row.name.startswith("n_")]
    results = [row.name for row in obligation.functions if row.name.startswith("result_")]
    assert len(parameters) == len(results) == 1
    parameter, result = parameters[0], results[0]
    declarations = "\n".join(f"(declare-const {row.name} Int)" for row in obligation.functions)
    scripts = {
        "premise_sat": f"(set-logic ALL)\n{declarations}\n(assert {body})\n(check-sat)\n",
        "complete_integer_domain": f"(set-logic ALL)\n(assert (not (forall (({parameter} Int)) (exists (({result} Int)) {body}))))\n(check-sat)\n",
        "source_bound_witness_zero": f"(set-logic ALL)\n{declarations}\n(assert {body})\n(assert (= {parameter} 0))\n(assert (= {result} 1))\n(assert (not {goal}))\n(check-sat)\n",
    }
    records = {}
    for solver in ("z3", "cvc5"):
        records[solver] = {name: _native_smt(captured, solver, script) for name, script in scripts.items()}
        assert {name: row["result"]["stdout"].strip() for name, row in records[solver].items()} == {
            "premise_sat": "sat", "complete_integer_domain": "unsat", "source_bound_witness_zero": "sat"}
    _save(captured, "premise-domain-native-checks", records)
    with captured["scheduler"].acquire("orchestration", cpu_slots=1, memory_mb=1024,
            child_process_slots=1, timeout=30, request_id="acceptance:conditional-smt") as parent:
        checked = profile.execute_integer_offset(compiled, parent_lease=parent)
    assert checked["status"] == "refuted" and checked["evidence_kind"] == "conditional_smt"
    assert checked["kernel_checked"] is checked["model_checked_against_runtime"] is checked["behavior_authority"] is False
    _save(captured, "native-conditional-smt", checked)
    matched = _match(captured)
    witness = next(row for row in matched["finite_counterexamples"] if row["input"] == 0)
    assert witness["observed_output"] == 1 and witness["expected_output"] == 2
    assert matched["observation"]["kernel_checked_model_table"] is True
    assert matched["source_semantics_verified"] is matched["proof_authority"] is False
    assert matched["source_cid"] == compiled.source_cid == cid_for_bytes(SOURCE)
    fact = matched["current_facts"][0]
    required = {matched["query"]["query_cid"], cid_for_structured(captured["head"].to_dict()),
        captured["head"].snapshot_cid, compiled.source_cid, matched["domain_cid"],
        matched["observation"]["trace_cid"], matched["observation_cid"],
        cid_for_structured(matched["observation"]["lean_certificate"]),
        compiled.cid, matched["observation"]["tool_policy_cid"]}
    assert required <= set(fact["provenance_refs"])
    # Same symbol and desired implementation in another file cannot erase this witness.
    decoy = _match(captured, text=_text(path="decoy.py"))
    assert decoy["residual_clause_ids"] == [] and decoy["source_cid"] != matched["source_cid"]
    assert set(row["fact_id"] for row in decoy["current_facts"]).isdisjoint(row["fact_id"] for row in matched["current_facts"])


@pytest.mark.parametrize("kind", ["contradictory", "narrow_precondition", "changed_literal", "unexplained_bool_refinement", "wrong_revision"])
def test_independent_native_translation_corruption_cannot_pass_closed_model_admission(captured, monkeypatch, kind):
    run = profile.SourceToVerificationPipeline.run
    def corrupt(*args, **kwargs):
        result = run(*args, **kwargs)
        row = result.obligation_results[0]
        obligation = row.smt_obligation
        if kind == "wrong_revision":
            return replace(result, bindings=replace(result.bindings, source=replace(result.bindings.source, source_revision="foreign:old")))
        if kind == "unexplained_bool_refinement":
            from ipfs_datasets_py.logic.backends.smt.compiler import BOOL_SORT
            obligation = replace(obligation, functions=(replace(obligation.functions[0], range=BOOL_SORT), *obligation.functions[1:]))
        else:
            assumption = obligation.assumptions[0]
            if kind == "contradictory":
                extra = replace(assumption, name="contradictory", formula=profile.term_eq(profile.term_int(1), profile.term_int(0)))
                obligation = replace(obligation, assumptions=(*obligation.assumptions, extra))
            elif kind == "narrow_precondition":
                parameter = next(x.name for x in obligation.functions if x.name.startswith("n_"))
                extra = replace(assumption, name="only_zero", formula=profile.term_eq(profile.term_symbol(parameter), profile.term_int(0)))
                obligation = replace(obligation, assumptions=(*obligation.assumptions, extra))
            else:
                left, right = assumption.formula.arguments
                changed_literal = replace(right, arguments=(right.arguments[0], profile.term_int(999)))
                obligation = replace(obligation, assumptions=(replace(assumption,
                    formula=profile.term_eq(left, changed_literal)),))
        return replace(result, obligation_results=(replace(row, smt_obligation=obligation),))
    monkeypatch.setattr(profile.SourceToVerificationPipeline, "run", corrupt)
    with pytest.raises(profile.IntegerProfileError):
        _compiled(captured)


@pytest.mark.parametrize("domain", [[False], [0.0], [], [0, 0], [2**31 + 1]])
def test_request_domain_cannot_silently_refine_python_values_to_int(captured, domain):
    with pytest.raises(match.FiniteIntegerIntentError):
        _match(captured, text=_text(domain))
    assert not list(captured["root"].glob("observation-*"))


def test_missing_contract_cannot_create_a_source_theorem(captured):
    with pytest.raises(profile.IntegerProfileError, match="exact IntegerOffsetContract"):
        profile.compile_integer_offset(_source(captured), None,
            revision="snapshot:" + captured["head"].snapshot_cid)


@pytest.mark.parametrize("selector", ["missing.py::increment(n)", "calc.py::other(n)"])
def test_wrong_exact_selector_keeps_both_requirements_unresolved(captured, selector, monkeypatch):
    monkeypatch.setattr(observation.BoundedToolRunner, "run", lambda *a, **k: pytest.fail("unmatched selector executed"))
    result = _match(captured, text=_text().replace("calc.py::increment(n)", selector))
    assert result["current_facts"] == result["finite_counterexamples"] == []
    assert result["residual_clause_ids"] == sorted([match.TYPE_STATEMENT_ID, match.OFFSET_STATEMENT_ID])


def test_actual_python_annotations_do_not_enforce_the_reviewed_input_assumption(captured):
    compiled = _compiled(captured)
    assert "annotations do not enforce" in compiled.to_dict()["assumptions"][0]
    script = compiled.source.decode() + "\nimport json\nprint(json.dumps([[type(n).__name__, type(increment(n)).__name__, increment(n)] for n in [True, 0.5]]))\n"
    with captured["scheduler"].acquire("validation", cpu_slots=1, memory_mb=256,
            child_process_slots=1, timeout=30, request_id="acceptance:annotation-runtime-boundary"):
        result = run_bounded_stdin_tool([str(Path(sys.executable).resolve()), "-I", "-S", "-c",
            "import sys; exec(sys.stdin.read())"], script, runner=BoundedToolRunner(),
            limits=ToolRunLimits(timeout_seconds=10, cpu_seconds=10, memory_bytes=256 * 1024**2,
                resident_memory_bytes=256 * 1024**2, max_input_bytes=65536,
                max_output_bytes=65536, max_workspace_bytes=65536))
    assert result.returncode == 0 and result.workspace_cleaned
    assert json.loads(result.stdout) == [["bool", "int", 2], ["float", "float", 1.5]]
    for domain in ([True], [0.5]):
        with pytest.raises(match.FiniteIntegerIntentError):
            match.build_finite_integer_intent(_text(domain))
    _save(captured, "annotation-runtime-boundary", dict(source_cid=compiled.source_cid,
        assumptions=compiled.to_dict()["assumptions"], actual_native_process=result.to_dict(),
        outside_domain_satisfaction_authorized=False))


@pytest.mark.parametrize("source", [
    "# café 😀\ndef increment(n: int) -> int:\n    return n + 1\n",
    "def incrément(n: int) -> int:\n    return n + 1\n",
    "def increment(entrée: int) -> int:\n    return entrée + 1\n",
    "def increment(n: int) -> int:\n    return 'é😀'\n",
    "def increment(n: int) -> int:\n    return n + 1\n".replace("\n", "\r\n"),
    "def increment(n: int) -> int:\n    return n + 1\n".replace("\n", "\r"),
    "def increment(n: int) -> int:\n    return '''é\u0085\u2028😀\nsecond line'''\n",
    "def increment(n):\n    return n + 1\n",
    "def increment(n: float) -> float:\n    return n + 1.0\n",
    "def increment(n: Any) -> Any:\n    return n + 1\n",
    "def increment(n: bool) -> bool:\n    return n + 1\n",
    "def increment(n: int) -> int:\n    return n / 2\n",
    "def increment(n: int) -> int:\n    return n // 2\n",
    "def increment(n: int) -> int:\n    return n ** 2\n",
    "def increment(n: int) -> int:\n    return abs(n)\n",
])
def test_exact_captured_utf8_spans_types_and_operations_join_semantic_refusal(captured, source, monkeypatch):
    raw = source.encode()
    unchanged_commit = _git(captured["repository"], "rev-parse", "HEAD")
    previous = captured["head"]
    _capture(captured, raw)
    assert _git(captured["repository"], "rev-parse", "HEAD") == unchanged_commit
    assert captured["head"] != previous and _source(captured) == raw
    adapted = adapter.adapt_source_to_software_verification(source, path="calc.py",
        revision="snapshot:" + captured["head"].snapshot_cid,
        include_supervisor_evidence=False, preserve_type_annotations=True)
    assert adapted.program is not None
    assert adapted.program.sources[0].source_revision == "snapshot:" + captured["head"].snapshot_cid
    assert adapted.program.sources[0].content_sha256 == hashlib.sha256(raw).hexdigest()
    for node in ast.walk(ast.parse(source)):
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            start, end, *_ = adapter._ast_byte_span(node, source=source, offsets=adapter._line_byte_offsets(source))
            assert raw[start:end] == segment.encode()
    _save(captured, "captured-typed-model", adapted.to_dict())
    monkeypatch.setattr(observation.BoundedToolRunner, "run", lambda *a, **k: pytest.fail("unsupported source launched native execution"))
    result = _match(captured)
    assert result["current_facts"] == [] and result["finite_counterexamples"] == []
    assert result["residual_clause_ids"] == sorted([match.TYPE_STATEMENT_ID, match.OFFSET_STATEMENT_ID])
    assert result["observation"]["status"] == "unsupported"
    assert result["observation"]["python_process"] is result["observation"]["lean_certificate"] is None
    assert result["observation"]["source_cid"] == cid_for_bytes(raw)


def test_dirty_same_head_stales_countermodel_and_successor_rechecks_exact_bindings(captured):
    initial = _match(captured)
    old_head = captured["head"]
    original_commit = _git(captured["repository"], "rev-parse", "HEAD")
    (captured["repository"] / "calc.py").write_bytes(SOURCE.replace(b"n + 1", b"n + 2"))
    with pytest.raises(StaleCodebaseError):
        _match(captured)
    _capture(captured, SOURCE.replace(b"n + 1", b"n + 2"))
    assert _git(captured["repository"], "rev-parse", "HEAD") == original_commit
    successor = _match(captured)
    assert successor["finite_counterexamples"] == successor["residual_clause_ids"] == []
    assert len(successor["current_facts"]) == 2
    assert old_head != captured["head"] and initial["source_cid"] != successor["source_cid"]
    with pytest.raises(observation.FiniteIntegerObservationError):
        observation.validate_finite_integer_observation(initial["observation"], expected_head=captured["head"],
            contract=CONTRACT, inputs=DOMAIN, tool_policy=captured["tools"])


@pytest.mark.parametrize("damage", ["marker_only", "caller_authority", "wrong_model_source", "omitted_domain_row"])
def test_caller_receipts_cannot_replace_fresh_owner_execution(captured, monkeypatch, damage):
    actual = _match(captured)
    fake = deepcopy(actual["observation"])
    if damage == "marker_only": fake = {"status": "observed", "kernel_checked_model_table": True, "certificate": "proved"}
    elif damage == "caller_authority": fake["proof_authority"] = True
    elif damage == "wrong_model_source": fake["source_cid"] = cid_for_bytes(SOURCE.replace(b"n + 1", b"n + 2"))
    else: fake["observations"] = fake["observations"][:-1]
    monkeypatch.setattr(observation, "observe_finite_integer_source", lambda **kwargs: fake)
    with pytest.raises((ValueError, KeyError)):
        _match(captured)


@pytest.mark.parametrize("verdict", ["unknown", "disagreement", "timeout"])
def test_native_checked_model_failure_classification_never_becomes_kernel_evidence(captured, monkeypatch, verdict):
    compiled = _compiled(captured)
    native = profile.run_bounded_stdin_tool
    calls = []
    def observed(command, text, **kwargs):
        result = native(command, text, **kwargs)
        calls.append(result.to_dict())
        if "(check-sat)" in text and "(get-model)" not in text:
            if verdict == "unknown": return replace(result, stdout="unknown\n")
            if verdict == "timeout": return replace(result, timed_out=True)
            if "--lang=smt2" in command: return replace(result, stdout="unsat\n")
        return result
    monkeypatch.setattr(profile, "run_bounded_stdin_tool", observed)
    with captured["scheduler"].acquire("orchestration", cpu_slots=1, memory_mb=1024,
            child_process_slots=1, timeout=30, request_id="acceptance:adversarial-result-classification") as parent:
        result = profile.execute_integer_offset(compiled, parent_lease=parent)
    assert len(calls) >= 4 and result["status"] == verdict
    assert result["kernel_checked"] is result["behavior_authority"] is result["model_checked_against_runtime"] is False
    _save(captured, "injected-failure-control", dict(control=verdict, injected_classification=True,
        actual_native_processes=calls, classification=result))
