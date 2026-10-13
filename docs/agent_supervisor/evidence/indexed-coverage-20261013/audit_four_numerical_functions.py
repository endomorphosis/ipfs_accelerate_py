"""Content-free replay of admitted public numerical source coverage.

No task source execution, neural inference, solver, or hidden verifier access.
Run with the reviewed datasets checkout on PYTHONPATH.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from ipfs_datasets_py.logic.formalization.autoencoder import source_function_units
from ipfs_datasets_py.logic.formalization.autoencoder.security import source_program_binding_384
from ipfs_datasets_py.logic.software_contracts import source_unit_numerical_links
from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_source

ROOT = Path('/home/barberb/lift_coding')
PUBLIC = ROOT / '.benchmarks/terminal-bench-2/largest-eigenval/environment/src'
OUTPUT = ROOT / 'artifacts/supervisor-indexed-coverage-20261013/four-function-coverage-audit.json'
PRIOR = ROOT / ('artifacts/source-context-substitution-20261013/live-trials/codex-legacy-01/'
               'jobs/supervisor-full-largest-eigenval/largest-eigenval__C4Kyfvm/agent/supervisor-result.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    owner = Path(numerical_source.__file__).resolve().parents[3]
    historical = json.loads(PRIOR.read_bytes())['initial_context']['source384_context']
    summary = historical['summary']
    files = []
    for name in ('eigen.py', 'eval.py'):
        raw = (PUBLIC / name).read_bytes()
        inv = source_function_units.extract_function_units(source_bytes=raw,
            source_sha256=sha(raw), source_path=name)
        native = numerical_logic.formalize_numerical_source(raw.decode(), path=name)
        numerical_logic.validate_numerical_logic_report(native, raw.decode(), path=name)
        linked = source_unit_numerical_links.bind_function_unit_logic(source_bytes=raw,
            unit_inventory=inv, numerical_report=native)
        source_unit_numerical_links.validate_function_unit_logic_links(linked,
            source_bytes=raw, unit_inventory=inv, numerical_report=native)
        units = {row['unit_id']: row for row in inv['units']}
        tree = ast.parse(raw.decode())
        nodes = {node.name: node for node in tree.body if type(node) is ast.FunctionDef}
        rows = []
        for link in linked['links']:
            unit = units[link['unit_id']]
            node = nodes[unit['qualified_name']]
            rows.append({key: link[key] for key in ('unit_id', 'start_byte', 'end_byte',
                'raw_span_sha256', 'normalized_body_sha256', 'status', 'structural_status',
                'unsupported_constructs', 'scalar_guard_status', 'scalar_guard_reason',
                'pending_obligation_count', 'compound_contract_count',
                'assertion_requirement_count', 'source_obligations_discharged')} | dict(
                    qualified_name=unit['qualified_name'],
                    ast_node_counts=dict(sorted(Counter(type(child).__name__ for child in ast.walk(node)).items())),
                    guard_observation='current deterministic guard replay, not a rerun of historical neural inference'))
        files.append(dict(path=name, source_sha256=sha(raw), source_bytes=len(raw),
            source_scope='admitted public task program', functions=rows,
            module_diagnostics=sorted({r['code'] for r in native['structural']['diagnostics'] if not r['function']}),
            pending_obligation_count=len(native['pending_obligations']),
            obligation_kinds=dict(sorted(Counter(r['kind'] for r in native['pending_obligations']).items())),
            report_sha256=native['report_sha256'], exact_replay_passed=True,
            projection_counts={key: len(native['projections'][key]) for key in ('first_order', 'smt')},
            lean_source_sha256=native['projections']['lean']['source_sha256'],
            lean_status=native['projections']['lean']['status']))
    samples = [{key: row[key] for key in ('path', 'qualified_name', 'status',
        'source_validation_status', 'source_validation_reason', 'candidate_sha256') if key in row}
        for row in summary['candidate_samples']]
    report = dict(schema='four-numerical-function-coverage-audit@1',
        datasets_commit=subprocess.check_output(['git', '-C', str(owner), 'rev-parse', 'HEAD'], text=True).strip(),
        producer_sha256={Path(module.__file__).name: sha(Path(module.__file__).read_bytes()) for module in
            (source_function_units, source_program_binding_384, source_unit_numerical_links, numerical_logic, numerical_source)},
        historical_decoder=dict(report_sha256=sha(PRIOR.read_bytes()),
            checkpoint_sha256=historical['checkpoint_sha256'], source_validation_scope=summary['source_validation_scope'],
            candidate_origin=summary['candidate_origin'], coverage=summary['coverage'],
            source_aligned_candidate_count=summary['source_aligned_candidate_count'],
            candidate_samples=samples, omitted_candidates=summary['omitted_candidates'],
            sampled_candidate_content='two retained candidates are scalar >= expressions unrelated to the numerical program; not repaired or promoted',
            neural_inference_rerun=False),
        files=files,
        totals=dict(functions=4, matched_function_spans=4, structurally_complete_functions=2,
            learned_source_aligned_candidates=0, compound_contracts=2, assertion_requirements=4,
            pending_obligations=45, source_obligations_discharged=0),
        profile=numerical_logic.describe_numerical_logic_profile(),
        distinctions=dict(
            learned_candidate='actual pinned SecurityIR checkpoint output, validated by an intentionally narrow two-int-parameter source contract; numerical candidates remain unsupported',
            deterministic_ast_facts='source-hash and exact-byte-bound ProgramIR, numerical pattern roles and assertion requirements, replayed without model calls',
            proof_discharge='none: FOL predicates, abstract SMT constraints, and Lean definitions/conditional lemmas do not establish NumPy runtime identity, floating-point refinement, reachability, or speed'),
        findings=[
            'The four decoder failures are an architecture/profile mismatch, not four missing installations or syntax errors.',
            'The first, reference and speed functions fail the two-parameter scalar guard; the assertion helper first fails the ASCII executable-token guard because its failure string contains Unicode. Removing that lexical restriction still leaves its one-argument numerical body unsupported.',
            'Both selected-eigenpair functions are already structurally complete, with 16 pending obligations each. Further scalar guard relaxation cannot improve their learned candidates soundly.',
            'The assertion helper already contributes all four public assertions and 13 pending obligations. Its sole structural gap is a deferred formatted failure message; completing that expression does not prove the assertions.',
            'The timing helper has two unlowered For bodies, two conditional expressions, and formatted output. This is substantive missing source coverage and explains its zero current obligations. Loop bodies must not be flattened into unconditional straight-line commands.',
            'ProgramIR conditional expressions and shared scalar IfExp grammar exist, but their typed semantics do not establish Python object truthiness or NumPy/timing semantics. Any numerical reuse must preserve guarded branch evaluation explicitly.',
            'The released numpy-unary-selection-384@1 head admits only a direct argmax/argmin return in one plain function. It is not a checkpoint for eig-abs-argmax-tuple compositions.',
            'No automatic projection to temporal, transition, deontic, separation, or concurrency families is justified by these source shapes alone.'
        ],
        recommended_order=[
            dict(priority=1, action='Add separately versioned numerical source grammar for structured conditional expressions and formatted strings, retaining exact expression graphs, conditional evaluation and unknown effects.',
                expected_effect='meaningful expression fidelity; assertion helper can become structurally complete only after its complete failure-message structure is represented',
                boundary='Keep Source384 candidate statuses and proof counters unchanged; do not widen existing scalar profiles or reinterpret imported names as trusted libraries.'),
            dict(priority=2, action='Add reviewed structured For-loop body/iterator/target/else representation with CFG or guarded region ownership, then attach child-call obligations with that region context.',
                expected_effect='Represent the timing helper calls and data dependencies presently hidden behind loop fences; no performance theorem or unconditional effect assertion.',
                boundary='Require explicit handling/refusal of break, continue, return, nested loops, iterator exceptions, and for-else before claiming complete coverage.'),
            dict(priority=3, action='Reuse shared structured 384D training/decoder infrastructure for a separately versioned compositional numerical grammar, starting with authored eig-selection and tuple/index fixtures.',
                expected_effect='A future candidate head can express actual numerical source structure and be independently compared with the whole-file deterministic ProgramIR.',
                boundary='Do not substitute deterministic targets for predictions, do not train on benchmark labels/hidden inputs, and retain exact checkpoint/holdout provenance.'),
            dict(priority=4, action='Define a reviewed numerical semantic boundary, explicit runtime/library/dtype/shape/tolerance witnesses, and source-to-model refinement before trying to discharge the 45 obligations.',
                expected_effect='A path from structural coverage to real source-bound proof obligations rather than additional vacuous compilation.',
                boundary='Reuse the separate source-AST/target-correspondence discipline in codebase_integer_lowering_lean; do not reuse its integer semantics for floating point or assume NumPy correctness.')
        ],
        test_requirements=[
            'Retain existing scalar mismatch/unsupported tests and exact learned candidate bytes; no unsupported candidate acquires qualified/proved status.',
            'Use authored fixtures for new grammar with independent exact UTF-8/CRLF byte spans and source replay mutation rejection.',
            'Conditional expressions: preserve test-before-selected-branch semantics, branch order and object-valued truthiness; never execute both branches or claim Bool typing from syntax.',
            'Formatted strings: retain conversions !s/!r/!a, format specs, nested/dynamic spec dependencies, Unicode, repeated operands and source spans; failure messages are evaluated only on failed assertions.',
            'Loop regions: demonstrate bodies remain conditional/repeated, body calls have enclosing-region provenance, and break/continue/for-else/early-return cases are modeled or explicitly refused.',
            'Retain unknown library effects, stale-source/import-shadowing rejection and conservative unresolved runtime identity.',
            'Replay ProgramIR bridge, FOL and SMT projections and run lake build on generated Lean syntax, with source obligations still pending unless a separately scoped theorem is actually checked.',
            'Run admitted public-source replay after production tests; report separate learned, structural, syntax-build and discharged-proof coverage counts.'
        ],
        activity=dict(source_executed=False, model_calls=0, training_runs=0, solver_calls=0,
            lake_build_executed=False, production_edits=False, hidden_benchmark_inputs_read=False),
        limitations=['Only two historical candidate samples were retained; guard reasons for the other two are current pure-source replay observations.',
            'This audit did not rerun inference or Lake; prior native compilation must be cited from its original evidence separately.',
            'Structural completeness is not source semantic correctness, and zero current obligations for a fenced function is missing coverage, not safety.'])
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(path=str(OUTPUT), sha256=sha(OUTPUT.read_bytes()), totals=report['totals'])))


if __name__ == '__main__':
    main()
