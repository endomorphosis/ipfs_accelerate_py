"""Authored closed import-alias repairs through real Doctor proof and worktrees."""
from copy import deepcopy
from dataclasses import replace
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import _provers
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_alias_contract import (
    ImportedAliasContractError, ImportedAliasRepair, OPERATOR, discover_imported_alias_repair,
    read_imported_alias_sources,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import DoctorCompositionError
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy

SOURCE = 'from helpers import transform as normalize\n\ndef answer(value):\n    return transform(value)\n'
DONOR = 'def transform(value):\n    return value\n'
AFTER = SOURCE.replace('return transform(value)', 'return normalize(value)')


def _discover(source=SOURCE, donor=DONOR, **extra):
    return discover_imported_alias_repair(sources={'answer.py': source, 'helpers.py': donor, **extra}, path='answer.py')


def _prepare(scenario, tmp_path, *, source=SOURCE, donor=DONOR):
    root = scenario['repository']
    (root / 'answer.py').write_text(source)
    (root / 'helpers.py').write_text(donor)
    subprocess.run(['git', '-C', str(root), 'rm', '-q', 'test_answer.py'], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 'answer.py', 'helpers.py'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Doctor fixture', '-c',
        'user.email=doctor@example.invalid', 'commit', '-qm', 'public imported alias mismatch'], check=True)
    baseline = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    profile, lifecycle = tmp_path / 'alias-profile', tmp_path / 'alias-lifecycle'
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    paths = ['answer.py', 'helpers.py']
    roots = {'request_cid': scenario['graph'].request_cid,
        'scan_cid': local.content_identity({'sources': local._sources(root, paths)}),
        'program_root': local.content_identity({'git_tree': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD^{tree}'], text=True).strip()})}
    graph = replace(scenario['graph'], **roots)
    validation = replace(graph.tasks[0].validations[0], argv=(
        'python3', '-B', '-c', 'from answer import answer; assert answer(2) == 2'))
    goal = replace(graph.goals[0], scope_paths=tuple(paths))
    task = replace(graph.tasks[0], goal_cid=goal.goal_cid, scope_paths=tuple(paths), validations=(validation,))
    graph = replace(graph, goals=(goal,), tasks=(task,))
    specs = deepcopy(scenario['manifest']['payload']['tasks'])
    specs[0]['scope_paths'] = paths
    specs[0]['validations'][0]['argv'] = list(validation.argv)
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    local.materialize_local_benchmark_plan(admission=admission, intent=scenario['intent'])
    runtime = DeterministicDoctorRuntime(checkout_root=root, index_root=tmp_path / 'alias-index',
        policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO))
    candidate_ref = 'refs/heads/doctor/imported-alias'
    subprocess.run(['git', '-C', str(root), 'update-ref', candidate_ref, baseline], check=True)
    return dict(runtime=runtime, intent=scenario['intent'], admission=admission,
        task_cid=task.task_cid, state_root=tmp_path / 'alias-state',
        solver_executable=Path(shutil.which('z3') or '/unavailable/z3'),
        kernel_executable=Path('/unavailable/lean'), candidate_ref=candidate_ref)


def test_exact_source_derived_alias_contract_and_no_general_import_repair():
    contract = _discover()
    assert contract.to_dict()['previous'] == 'transform'
    assert contract.to_dict()['replacement'] == 'normalize'
    assert contract.to_dict()['subject'] == 'answer'
    assert set(contract.to_dict()['source_hashes']) == {'answer.py', 'helpers.py'}
    assert contract.to_dict()['target_binding'] == contract.to_dict()['bindings']['normalize']
    contract.validate_candidate(AFTER)
    assert _discover(AFTER) is None
    with pytest.raises(ImportedAliasContractError):
        contract.validate_candidate(AFTER.replace('normalize(value)', 'normalize(3)'))
    forged = contract.to_dict()
    forged['target_binding'] = 'invented'
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(forged), json.dumps(contract.sources()))


def test_alias_identifier_and_module_binding_grammar_bounds():
    alias64 = 'n' * 64
    bounded = SOURCE.replace('normalize', alias64)
    assert _discover(bounded).to_dict()['replacement'] == alias64
    with pytest.raises(ImportedAliasContractError):
        _discover(SOURCE.replace('normalize', alias64 + 'n'))
    with pytest.raises(ImportedAliasContractError):
        _discover(SOURCE.replace('transform', 't' * 65), DONOR.replace('transform', 't' * 65))
    extra = ''.join(f'\ndef other{index}(value):\n    return value\n' for index in range(14))
    assert len(_discover(SOURCE + extra).to_dict()['bindings']) == 16
    with pytest.raises(ImportedAliasContractError):
        _discover(SOURCE + extra + '\ndef overflow(value):\n    return value\n')


@pytest.mark.parametrize('source,donor', [
    (SOURCE.replace('as normalize', 'as normalize, transform as alternate'), DONOR),
    (SOURCE.replace('from helpers', 'from missing'), DONOR),
    (SOURCE.replace('from helpers', 'from .helpers'), DONOR),
    (SOURCE.replace('transform as normalize', '*'), DONOR),
    (SOURCE.replace('def answer(value)', 'def answer(normalize)'), DONOR),
    (SOURCE.replace('def answer(value)', 'def answer(transform)'), DONOR),
    (SOURCE.replace('    return', '    normalize = value\n    return'), DONOR),
    (SOURCE.replace('transform(value)', 'transform((normalize := value))'), DONOR),
    (SOURCE.replace('transform(value)', 'transform([normalize for normalize in value])'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(value.member)'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(*value)'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(**value)'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(normalize(value))'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(value + 1)'), DONOR),
    (SOURCE.replace('transform(value)', 'transform(other=value)'), DONOR),
    (SOURCE, DONOR.replace('value):', 'value=print(1)):')),
    (SOURCE, DONOR.replace('value):', 'value: print(1)):')),
    (SOURCE, DONOR.replace('):', ') -> print(1):')),
    (SOURCE, '@print(1)\n' + DONOR),
    (SOURCE, DONOR + '\ntransform = None\n'),
    (SOURCE, 'import os\n' + DONOR),
    (SOURCE, DONOR.replace('return value', 'return print(value)')),
    (SOURCE, DONOR.replace('return value', 'return transform(value)')),
    (SOURCE.replace('transform', 'len'), DONOR.replace('transform', 'len')),
    (SOURCE.replace('def answer(value)', 'def answer(value=[])'), DONOR),
    (SOURCE.replace('    return transform(value)', '    try:\n        return transform(value)\n    except Exception as normalize:\n        return value'), DONOR),
    (SOURCE.replace('    return transform(value)', '    with value as normalize:\n        return transform(value)'), DONOR),
    (SOURCE.replace('    return transform(value)', '    match value:\n        case normalize:\n            return transform(value)'), DONOR),
])
def test_unsupported_alias_grammar_abstains(source, donor):
    with pytest.raises(ImportedAliasContractError):
        _discover(source, donor)


@pytest.mark.parametrize('module', ['sys', 'os', 'builtins', 'json', 'site'])
def test_stdlib_or_builtin_module_cannot_establish_local_donor(module):
    with pytest.raises(ImportedAliasContractError):
        discover_imported_alias_repair(sources={'answer.py': SOURCE.replace('helpers', module),
            module + '.py': DONOR}, path='answer.py')


@pytest.mark.parametrize('name', ['__name__', '__file__', '__builtins__', '__annotations__'])
def test_implicit_module_global_is_not_an_unbound_original(name):
    with pytest.raises(ImportedAliasContractError):
        _discover(SOURCE.replace('transform', name), DONOR.replace('transform', name))


@pytest.mark.parametrize('shape', ['oversized', 'symlink', 'fifo', 'wrong_hash'])
def test_population_loader_refuses_unbounded_or_unbound_files(tmp_path, shape):
    path = tmp_path / 'helpers.py'
    if shape == 'symlink':
        target = tmp_path / 'donor.txt'
        target.write_text(DONOR)
        path.symlink_to(target)
    elif shape == 'fifo':
        os.mkfifo(path)
    elif shape == 'oversized':
        with path.open('wb') as stream:
            stream.truncate(1_000_001)
    else:
        path.write_text(DONOR + '\n')
    with pytest.raises((ImportedAliasContractError, OSError)):
        read_imported_alias_sources(repository=tmp_path,
            source_hashes={'helpers.py': hashlib.sha256(DONOR.encode()).hexdigest()})


def test_real_lean_and_z3_reject_an_incorrect_alias_target(tmp_path):
    paths = _provers({})
    contract = _discover()
    projection = contract.formal_projection('test:incorrect-target-negative-control')
    target = json.dumps(contract.to_dict()['target_binding'])
    alias = json.dumps(contract.to_dict()['replacement'])
    proof = tmp_path / 'IncorrectAlias.lean'
    proof.write_text(projection['lean'].replace('= some ' + target, '= some "wrong-target"'))
    lean = subprocess.run([str(paths['kernel_executable']), str(proof)],
                          capture_output=True, text=True, timeout=30)
    assert lean.returncode != 0 and 'error' in lean.stdout + lean.stderr
    smt = projection['smt'].replace('(lookupBinding ' + alias + ') ' + target,
                                    '(lookupBinding ' + alias + ') "wrong-target"')
    z3 = subprocess.run([shutil.which('z3'), '-in'], input=smt,
                        capture_output=True, text=True, timeout=15)
    assert z3.returncode == 0 and z3.stdout.strip() == 'sat'


def test_no_unsigned_unresolved_extra_source_is_ignored():
    with pytest.raises(ImportedAliasContractError):
        _discover(**{'consumer.py': 'def other(value):\n    return missing(value)\n'})


def test_currentness_rejects_changed_or_oversized_donor(tmp_path):
    contract = _discover()
    for name, text in contract.sources().items():
        (tmp_path / name).write_text(text)
    contract.assert_current(tmp_path)
    (tmp_path / 'helpers.py').write_text(DONOR + ' ' * 1_000_001)
    with pytest.raises(ImportedAliasContractError):
        contract.assert_current(tmp_path)


def test_real_provers_native_impact_transaction_and_allocated_candidate(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    task_before = scenario['intent'].get_task(inputs['task_cid'])
    prepared = prepare_doctor_task_repair(**inputs)
    assert prepared.report['status'] == 'prepared', prepared.report.get('reason_codes')
    assert prepared.report['operator'] == OPERATOR
    assert prepared.inputs.alias_contract is not None
    assert set(prepared.inputs.program_graph.roots.included_roots) == {'answer.py', 'helpers.py'}
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'candidate_ready', json.dumps(result.get('stages'), sort_keys=True)
    composition = prepared.runtime.composition_result
    assert composition.proof.mutation_capable and composition.synthesis.authoritative_proof
    assert composition.impact.mutation_admissible and result['transaction']['committed']
    handoff = result['handoff']
    assert handoff['operator'] == OPERATOR and handoff['provider_calls'] == 0
    assert 'Not whole-program' in handoff['proof_scope']
    assert base64.b64decode(handoff['edits'][0]['after_bytes_base64']).decode() == AFTER
    assert not handoff['completion_authority'] and not handoff['publication_authority']
    root = scenario['repository']
    allocation = tmp_path / 'allocated-alias-candidate'
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', '-q', str(allocation),
                    handoff['base_commit']], check=True)
    materialize_doctor_candidate(artifact=Path(result['handoff_path']), expected_sha256=result['handoff_sha256'],
        task_cid=inputs['task_cid'], prompt=json.dumps({'objective_id': handoff['task_id']}), workspace=allocation)
    subprocess.run(['python3', '-B', '-c', 'from answer import answer; assert answer(2) == 2'],
                   cwd=allocation, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert (root / 'answer.py').read_text() == SOURCE and (root / 'helpers.py').read_text() == DONOR
    assert scenario['intent'].get_task(inputs['task_cid']) == task_before


def test_donor_drift_refuses_native_execution_before_proof(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    prepared = prepare_doctor_task_repair(**inputs)
    assert prepared.inputs is not None
    (scenario['repository'] / 'helpers.py').write_text(DONOR.replace('return value', 'return 7'))
    with pytest.raises((DoctorCompositionError, local.LocalPlanningError)):
        execute_doctor_task_repair(prepared)
    assert prepared.runtime.composition_result is None
    assert not (inputs['state_root'] / 'candidate-handoff.json').exists()


def test_unsupported_donor_stays_native_residual_before_proof(scenario, tmp_path):
    inputs = _prepare(scenario, tmp_path, donor=DONOR + '\ntransform = 42\n')
    result = execute_doctor_task_repair(prepare_doctor_task_repair(**inputs))
    assert result['status'] == 'residual' and result['plan_refill']['successors']
    assert result['provider_calls'] == 0 and not result['completion_authority']
    assert not inputs['state_root'].exists()


def test_actual_failed_prover_does_not_create_alias_candidate(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    inputs['solver_executable'] = Path('/usr/bin/false')
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual'
    assert not result['stages']['proof']['mutation_capable']
    assert prepared.runtime.composition_transaction is None
    assert not (inputs['state_root'] / 'candidate-handoff.json').exists()


def test_exact_native_prover_config_overflow_returns_residual_before_pin_or_state(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    # Keep the real Lean executable bytes. Its deliberately long selected path
    # makes the actual serialized prover configuration exceed the native argv
    # bound. No process should run or proof store be created for this request.
    directory = tmp_path.joinpath(*['p' * 190 for _ in range(19)])
    directory.mkdir(parents=True)
    selected = directory / 'lean'
    shutil.copy2(inputs['kernel_executable'], selected)
    inputs['kernel_executable'] = selected
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert prepared.inputs is None and result['status'] == 'residual'
    assert result['reason_codes'] == ['operator_proof_bounds_exceeded']
    assert result['operator'] == OPERATOR and result['plan_refill']['successors']
    assert not inputs['state_root'].exists()
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
