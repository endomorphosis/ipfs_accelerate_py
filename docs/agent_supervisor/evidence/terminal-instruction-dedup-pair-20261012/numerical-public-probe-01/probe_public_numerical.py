"""Bounded offline projection probe of already-public, hash-pinned task inputs.

Run only after the timed comparison container has closed and root releases it.
This artifact calls existing APIs; it does not modify production or task code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

W = Path('/home/barberb/lift_coding')
R = W / 'artifacts/supervisor-ranked-successor-20261012'
P = W / '.worktrees/supervisor-comparison-contracts-20261011'
D = W / '.worktrees/census-tree-pin-datasets-20261012'
A = R / 'current-comparison-archive-05'
NAMES = ('eigen.py', 'eval.py')
HASHES = {'eigen.py': 'bbd5be97fb11e9ed46fd5f1fb9d407efb19b265b549d3c36e2cfba923efe3d9d',
          'eval.py': '70a12a80b9caa392b1f1f0439de853932c5ac61f1a2274b5de1ce93aebd4e139'}
LAKE = Path('/home/barberb/.elan/toolchains/leanprover--lean4---v4.29.1/bin/lake')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def environment(output):
    env = json.loads((A / 'host-execution-environment.json').read_text())['environment_overrides']
    env = dict(env, IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(output / 'orchestration'),
               IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(output / 'seal.duckdb'))
    os.environ.update(env)
    os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKDB', None)
    sys.path[:0] = env['PYTHONPATH'].split(':')
    # This probe needs no network, model/provider client, or downloaded assets.
    def forbid_network(*args, **kwargs):
        raise RuntimeError('network is forbidden in this offline projection probe')
    socket.socket.connect = forbid_network
    socket.create_connection = forbid_network
    return env


def pins():
    diagnostic = json.loads((R / 'full-coding-comparison/publication-staging/diagnostics/public-source-formalization-gaps.json').read_text())
    rows = {str((P if v['tree'] == 'source' else D) / v['path']): v['sha256']
            for v in diagnostic['source_references'].values()}
    archive = json.loads((A / 'bundle-05/manifest.json').read_text())
    inventory = {r['path']: r['sha256'] for r in archive['files']}
    extra = [
        ('datasets', 'ipfs_datasets_py/logic/formalization/autoencoder/native_family_lake_v4.py'),
        ('datasets', 'ipfs_datasets_py/logic/software_contracts/source_unit_numerical_links.py'),
        ('datasets', 'ipfs_datasets_py/logic/software_verification/program.py'),
        ('datasets', 'ipfs_datasets_py/logic/software_verification/syntax_bridge.py'),
        ('datasets', 'ipfs_datasets_py/logic/backends/smt/compiler.py'),
        ('source', 'ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_database.py'),
        ('source', 'ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py'),
    ]
    for tree, relative in extra:
        rows[str((P if tree == 'source' else D) / relative)] = inventory[tree + '/' + relative]
    for path, expected in rows.items():
        if sha(path) != expected:
            raise ValueError('qualified production bytes changed: ' + path)
    return rows


def cold_validate(output):
    environment(output)
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
    expected = json.loads((output / 'captured-context.json').read_text())
    started = time.monotonic()
    actual = native.validate_numerical_context(repository=output / 'public-inputs', expected_receipt=expected)
    result = dict(exact_receipt_replayed=actual == expected,
                  ducklake_verified=actual['hydration']['ducklake_verified'],
                  seconds=time.monotonic() - started,
                  provider_calls=0, model_inference_calls=0, source_executed=False)
    write(output / 'cold-replay.json', result)
    return 0 if result['exact_receipt_replayed'] and result['ducklake_verified'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-after-explicit-container-closure', action='store_true')
    parser.add_argument('--cold-validate', action='store_true')
    args = parser.parse_args()
    if not args.run_after_explicit_container_closure:
        parser.error('root must explicitly release this probe after the timed container closes')
    output = args.output.absolute()
    if args.cold_validate:
        return cold_validate(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    outcome = dict(schema='terminal-public-numerical-projection-probe@1', complete=False,
        provider_calls=0, neural_inference_calls=0, training_steps=0, source_executed=False,
        downloads=0, production_mutations=False, source_obligations_discharged=0,
        source_semantics_verified=False, numerical_correctness_proved=False,
        task_performance_measured=False, learned_coverage_established=False,
        scope='Existing deterministic projections, real native lakes, generated-module compilation and abstract SMT checks only.')
    before = None
    try:
        before = pins()
        env = environment(output)
        write(output / 'command.json', dict(argv=sys.argv, interpreter=sys.executable,
            runner_sha256=sha(__file__), environment_overrides=env, source_before=before,
            public_input_hashes=HASHES, lake=str(LAKE), release='explicit root release after timed container closure'))
        repository = output / 'public-inputs'
        repository.mkdir()
        for name in NAMES:
            original = W / '.benchmarks/terminal-bench-2/largest-eigenval/environment/src' / name
            if sha(original) != HASHES[name]:
                raise ValueError('public source differs from retained baseline input')
            (repository / name).write_bytes(original.read_bytes())
        from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
        from ipfs_datasets_py.logic.software_verification import numerical_logic as logic
        from ipfs_datasets_py.logic.formalization.autoencoder.native_family_lake_v4 import _execute
        import z3
        if Path(native.__file__).resolve() != P / 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py':
            raise ValueError('wrong native consumer root')
        if Path(logic.__file__).resolve() != D / 'ipfs_datasets_py/logic/software_verification/numerical_logic.py':
            raise ValueError('wrong numerical producer root')
        phase = time.monotonic()
        receipt = native.prepare_numerical_context(repository=repository,
            source_hashes=HASHES, output=output / 'native-context', profile=native.PROFILE)
        write(output / 'captured-context.json', receipt)
        outcome['projection_and_hydration_seconds'] = time.monotonic() - phase
        outcome['native_summary'] = receipt['summary']
        outcome['hydration'] = receipt['hydration']
        phase = time.monotonic()
        assert native.validate_numerical_context(repository=repository, expected_receipt=receipt) == receipt
        outcome['same_process_replay_seconds'] = time.monotonic() - phase
        cold = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
            '--output', str(output), '--run-after-explicit-container-closure', '--cold-validate'],
            capture_output=True, text=True, timeout=180)
        (output / 'cold.stdout').write_text(cold.stdout)
        (output / 'cold.stderr').write_text(cold.stderr)
        outcome['cold_process_returncode'] = cold.returncode
        if cold.returncode != 0:
            raise RuntimeError('cold native DuckLake replay failed')
        outcome['cold_replay'] = json.loads((output / 'cold-replay.json').read_text())
        index = json.loads((output / 'native-context/reports.json').read_text())
        per_file = []
        for name in NAMES:
            report_path = output / 'native-context' / index[name]['artifact']
            report = json.loads(report_path.read_text())
            assert sha(report_path) == index[name]['sha256']
            assert logic.validate_numerical_logic_report(report, (repository / name).read_text(), path=name) == report
            folder = output / ('checks-' + name.removesuffix('.py'))
            folder.mkdir()
            artifact = report['projections']['lean']
            if not artifact or artifact['source_obligations_discharged'] != 0:
                raise ValueError('explicit pending numerical Lean artifact required')
            source = artifact['source']
            if 'sorry' in source or 'axiom ' in source:
                raise ValueError('generated source has an unreviewed proof placeholder')
            (folder / (artifact['library'] + '.lean')).write_text(source)
            phase = time.monotonic()
            lean = _execute(source, artifact['library'], str(LAKE), 60)
            write(folder / 'lake-check.json', lean)
            lean_seconds = time.monotonic() - phase
            if lean['status'] != 'passed' or lean.get('backend_executed') is not True:
                raise RuntimeError('generated numerical Lean module did not compile')
            smt_rows = []
            for number, row in enumerate(report['projections']['smt']):
                script = row['compilation']['script']['source']
                script_file = folder / f'obligation-{number:03d}.smt2'
                script_file.write_text(script)
                solver = z3.Solver();solver.set(timeout=5000)
                phase = time.monotonic();solver.from_string(script);checked = solver.check()
                smt_rows.append(dict(index=number, script_sha256=sha(script_file), result=str(checked),
                    seconds=time.monotonic()-phase, source_witnesses_provided=False,
                    source_obligations_discharged=0))
                if checked == z3.unknown:
                    smt_rows[-1]['reason_unknown'] = solver.reason_unknown()
            write(folder / 'smt-checks.json', smt_rows)
            per_file.append(dict(path=name, source_sha256=HASHES[name], report_sha256=index[name]['sha256'],
                summary=report['summary'], unsupported_constructs=report['structural']['diagnostics'],
                compound_contract_count=len(report['compound_contracts']),
                assertion_requirement_count=len(report['assertion_requirements']),
                obligation_kind_counts=dict(Counter(row['kind'] for row in report['pending_obligations'])),
                unsupported_families=report['unsupported_families'],
                lean_status=lean['status'], lean_seconds=lean_seconds,
                lean_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                lean_conditional_lemmas=artifact['conditional_lemmas'],
                smt_check_count=len(smt_rows), smt_results=dict(Counter(row['result'] for row in smt_rows)),
                native_program_roundtrip_exact=(report['projections']['program']['bridge']['status'] == 'ok'
                    and report['projections']['program']['bridge']['preservation'] == 'exact'
                    and not report['projections']['program']['bridge']['losses']
                    and not report['projections']['program']['bridge']['unsupported'])))
            if name == 'eigen.py':
                # Same meaningful negative as the existing native numerical test:
                # removing extent>0 invalidates the last-index lemma at extent=0.
                needle = '(nonempty : 0 < extent)'
                negative_source = source.replace(needle, '')
                assert needle in source and source != negative_source
                (folder / 'negative-missing-nonempty-premise.lean').write_text(negative_source)
                negative = _execute(negative_source, artifact['library'], str(LAKE), 60)
                write(folder / 'negative-lake-check.json', negative)
                if negative['status'] != 'failed' or negative.get('backend_executed') is not True:
                    raise RuntimeError('native Lean did not reject the authored missing-premise control')
                outcome['authored_negative_missing_premise_rejected'] = True
        outcome['per_file'] = per_file
        outcome['tool_versions'] = dict(z3=z3.get_version_string(), z3_module_sha256=sha(z3.__file__),
            lake_executable_sha256=sha(LAKE), lean_toolchain=lean['toolchain'])
        outcome['complete'] = True
        outcome['limitations'] = [
            'ProgramIR acceptance and Lake compilation establish structural/syntax checks only.',
            'Conditional Lean lemmas retain unproved premises for actual source values and calls.',
            'SMT checks the generated abstract obligation queries without source premises; satisfiable counterexamples do not prove source correctness.',
            'NumPy, floating/complex behavior, external calls, effects and performance remain unresolved.',
            'Unsupported loop bodies and deferred assertion message semantics remain explicit fences.',
            'No target program, checkpoint, embedding model, provider, hidden evaluator or solution was executed.']
    except Exception as error:
        outcome['error_type'] = type(error).__name__
        outcome['error'] = str(error)
        (output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        outcome['seconds'] = time.monotonic() - started
        if before is not None:
            after = {path:sha(path) for path in before}
            outcome['source_after'] = after
            outcome['source_unchanged'] = before == after
            outcome['complete'] = outcome['complete'] and outcome['source_unchanged']
        write(output / 'qualification.json', outcome)
        print(json.dumps({key:outcome.get(key) for key in ['complete','seconds','error_type','source_unchanged']}))
    return 0 if outcome['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
