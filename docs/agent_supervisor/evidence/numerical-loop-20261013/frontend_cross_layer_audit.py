"""Independent authored-source data audit; never execute the audited programs."""
from pathlib import Path
import hashlib
import json
from ipfs_datasets_py.logic.software_verification import numerical_source_v3 as source
from ipfs_datasets_py.logic.software_verification import numerical_logic_v3 as logic

ROOT = Path(__file__).resolve().parent
CASES = {
    "for_protocol": "def f(items):\n    for item in acquire(items):\n        inspect(item)\n    else:\n        finished(items)\n    return item\n",
    "nested_activation": "def f(groups):\n    for group in groups:\n        for item in acquire(group):\n            inspect(item[0])\n",
    "inner_else_continue": "def f(groups):\n    for group in groups:\n        for item in group:\n            inspect(item)\n        else:\n            continue\n    return groups\n",
    "inner_else_break": "def f(groups):\n    for group in groups:\n        for item in group:\n            inspect(item)\n        else:\n            break\n    return groups\n",
    "while_else_outer_continue": "def f(groups):\n    for group in groups:\n        while ready(group):\n            inspect(group)\n        else:\n            continue\n",
    "loop_in_exhausted_else": "def f(groups):\n    for group in groups:\n        inspect(group)\n    else:\n        for item in groups:\n            if skip(item):\n                continue\n            inspect(item)\n",
    "return_both_paths": "def f(items):\n    for item in items:\n        return inspect(item)\n    else:\n        return empty()\n",
    "assertion_ifexp_format": "def f(items, flag):\n    for item in items:\n        assert valid(item) if flag else fallback(item), f'bad {item!r:{width()}}'\n",
    "test_ifexp": "def f(flag):\n    while left() if flag else right():\n        inspect(flag)\n",
    "unsupported_try": "def f(items):\n    try:\n        for item in items:\n            inspect(item)\n    finally:\n        clean()\n",
    "unsupported_target": "def f(items):\n    for left, right in items:\n        inspect(left)\n",
    "unsupported_guarded_rhs": "def f(items, flag):\n    for item in items:\n        owner.field = left() if flag else right()\n",
}


def audit(name, text):
    structural = source.adapt_numerical_source(text, path="authored.py")
    report = logic.formalize_numerical_source(text, path="authored.py")
    assert source.verify_numerical_source(structural, text, path="authored.py").to_dict() == structural.to_dict()
    assert logic.validate_numerical_logic_report(report, text, path="authored.py") == report
    program = structural.program
    expressions = {e.expression_id: e for e in program.expressions}
    commands = {c.command_id: c for c in program.commands}
    regions = {r["region_id"]: r for r in structural.control_regions}
    checks = ["exact_source_and_logic_replay"]
    for region in regions.values():
        function = next(f for f in program.functions if f.function_id == region["function_id"])
        edges = {e.edge_id: e for e in function.cfg.edges}
        if region["kind"] == "for_statement":
            normal = [e for e in edges.values() if e.source_block_id == region["header_block_id"] and e.kind.value == "normal"]
            assert len(normal) == 1 and normal[0].target_block_id == region["target_block_id"]
            assert edges[region["initialization_edge_id"]].target_block_id == region["header_block_id"]
            assert edges[region["exhaustion_edge_id"]].target_block_id == region["else_entry_block_id"]
            assert expressions[region["iterator_expression_id"]].operand_ids == (region["iterable_expression_id"],)
            assert expressions[region["advance_expression_id"]].operand_ids != (region["iterator_expression_id"],)
            assert region["implicit_iterator_close_claimed"] is False
            assert region["target_may_be_unbound_after_zero_iterations"] is True
        for row in structural.command_contexts:
            if row["function_id"] != function.function_id:
                continue
            for edge in (edges[key] for key in row["edge_ids"]):
                if edge.exception_type == "StopIteration":
                    assert edge.source_block_id in {r["header_block_id"] for r in regions.values() if r["kind"] == "for_statement"}
            if row["control_transfer"] in {"continue", "break"}:
                active = [s["region_id"] for s in row["loop_contexts"] if s["phase"] != "else"]
                assert row["target_region_id"] == active[-1]
                assert not commands[row["command_id"]].expression_ids
                assert len(row["edge_ids"]) == 1
    checks += ["iterator_protocol_phase_ownership", "nearest_active_loop_transfers", "no_implicit_close_or_definite_assignment"]
    for row in report["pending_obligations"]:
        assert row["status"] == "pending" and row["proof_authority"] is False
        assert all(row[key] is False for key in logic._CONTROL_FALSE)
        assert row["evaluation_contexts"]
        for context in row["evaluation_contexts"]:
            assert context["source_sha256"] == report["source_sha256"]
            assert context["guard_conditions"][0]["kind"] == "cfg_block_visit"
            assert context["function_id"] in row["function_ids"]
        if "proof_scope" in row:
            assert row["evaluation_context_role"] == "source_anchor_only"
            assert row["logical_guard_scope"] == "unresolved_proof_scope_applicability"
            scope = row["proof_scope"]
            function = next(f for f in program.functions if f.function_id == scope["function_id"])
            assert set(scope["relevant_edge_ids"]) <= {e.edge_id for e in function.cfg.edges}
            assert scope["applicability_verified"] is False
    if name == "assertion_ifexp_format":
        width = next(r for r in report["pending_obligations"] if r["roles"].get("qualified_callee") == "width")
        for context in width["evaluation_contexts"]:
            guards = context["guard_conditions"]
            assert any(g["kind"] == "assertions_enabled" for g in guards)
            assert any(g["kind"] == "truthiness_result" and g["value"] is False for g in guards)
    if name.startswith("unsupported"):
        assert structural.status == "partial" and not report["compound_contracts"]
    checks += ["guarded_effects_and_loop_contexts_preserved", "proof_scope_distinct_from_source_anchor", "no_proof_or_runtime_authority"]
    return dict(name=name, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                structural_status=structural.status, checks=checks, passed=True,
                control_regions=len(regions), pending_obligations=len(report["pending_obligations"]))


rows = [audit(name, text) for name, text in CASES.items()]
protocol_path = ROOT / "authored-loop-protocol-observation.json"
protocol = json.loads(protocol_path.read_text())
result = dict(schema="numerical-loop-cross-layer-audit@1", passed=all(r["passed"] for r in rows),
    authored_structural_cases=rows,
    authored_cpython_protocol_probe=dict(path=protocol_path.name,
        sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        cases=len(protocol["cases"]), passed=sum(c["passed"] for c in protocol["cases"])),
    producers={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (source,logic)},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    benchmark_source_executed=False, audited_authored_source_executed=False,
    provider_calls=0, solver_calls=0,
    limitations=["Structural source replay is not Python/NumPy semantic equivalence or an induction/termination proof.",
      "Only the separate retained CPython protocol probe executes its authored bounded fixtures.",
      "Direct structural validators are supplemented by mandatory exact source replay at pager/link admission.",
      "Try/finally, context managers, async and complex for targets remain explicit unsupported fences.",
      "No state interpretation for temporal/TLA families is inferred from the CFG."])
(ROOT / "frontend-cross-layer-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print(json.dumps(dict(passed=result["passed"], authored_cases=len(rows), cpython_protocol_cases=len(protocol["cases"]))))
