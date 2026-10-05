"""Closed alias grammar and real proof controls for concrete Python signatures."""
import json
import shutil
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.doctor_alias_contract import (
    ImportedAliasContractError, ImportedAliasRepair, discover_imported_alias_repair,
)
from test.api.test_doctor_task_workflow import _provers


SOURCE = '''"""Authored caller documentation."""
from helpers import transform as normalize

def answer(value, /, enabled=True):
    """Use the declared normalization helper."""
    return transform(value, enabled=enabled)
'''
DONOR = '''"""Authored helper documentation."""
def transform(value, /, scale=2, *, enabled=True):
    """Preserve the disabled value."""
    return value * scale if enabled else value
'''
AFTER = SOURCE.replace('return transform(', 'return normalize(')


def _contract(source=SOURCE, donor=DONOR):
    return discover_imported_alias_repair(
        sources={'answer.py': source, 'helpers.py': donor}, path='answer.py')


def test_expanded_grammar_retains_exact_signature_and_docstrings():
    contract = _contract()
    assert contract.to_dict()['call_binding'] == {
        'positional_parameters': ['value', 'scale'],
        'keyword_parameters': ['scale', 'enabled'],
        'required_parameters': ['value'],
        'positional_argument_count': 1,
        'keyword_arguments': ['enabled'],
    }
    contract.validate_candidate(AFTER)
    assert _contract(AFTER) is None
    for after in (AFTER.replace('enabled=enabled', 'enabled=False'),
                  AFTER.replace('Authored caller documentation.', 'Different caller documentation.')):
        with pytest.raises(ImportedAliasContractError):
            contract.validate_candidate(after)
    forged = contract.to_dict()
    forged['call_binding']['required_parameters'] = []
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(forged), json.dumps(contract.sources()))


@pytest.mark.parametrize('default', ['None', 'False', '0', '1.5', '-1', '+2.5', '"text"', 'b"bytes"'])
def test_immutable_literal_defaults_are_admitted_without_execution(default):
    donor = f'def transform(value={default}):\n    return value\n'
    source = 'from helpers import transform as normalize\ndef answer():\n    return transform()\n'
    assert _contract(source, donor).to_dict()['call_binding']['required_parameters'] == []


@pytest.mark.parametrize('expression', [
    'value + scale', '-value', 'value < scale', 'value and scale',
    'value if enabled else scale',
])
def test_return_expression_structure_is_preserved_not_executed(expression):
    donor = DONOR.replace('value * scale if enabled else value', expression)
    contract = _contract(donor=donor)
    assert contract.sources()['helpers.py'] == donor
    contract.validate_candidate(AFTER)


@pytest.mark.parametrize('call,signature', [
    ('transform(value=value)', 'value, /, scale=2, *, enabled=True'),
    ('transform(value, value, scale=value)', 'value, /, scale=2, *, enabled=True'),
    ('transform()', 'value, /, scale=2, *, enabled=True'),
    ('transform(value, unknown=enabled)', 'value, /, scale=2, *, enabled=True'),
    ('transform(value, value, value)', 'value, /, scale=2, *, enabled=True'),
    ('transform(value)', 'value, /, scale=2, *, enabled'),
    ('transform(value, enabled=enabled)', 'value, /, scale=2, *rest, enabled=True'),
    ('transform(value, enabled=enabled)', 'value, /, scale=2, enabled=True, **rest'),
])
def test_python_binding_errors_remain_residual(call, signature):
    source = SOURCE.replace('transform(value, enabled=enabled)', call)
    donor = DONOR.replace('value, /, scale=2, *, enabled=True', signature)
    with pytest.raises(ImportedAliasContractError):
        _contract(source, donor)


@pytest.mark.parametrize('replacement', ['print(1)', '[]', '{}', 'object()', 'enabled'])
def test_defaults_cannot_run_code_or_introduce_mutable_global_state(replacement):
    with pytest.raises(ImportedAliasContractError):
        _contract(donor=DONOR.replace('scale=2', 'scale=' + replacement))


@pytest.mark.parametrize('expression', [
    'value.member', 'value[0]', 'abs(value)', '[value for value in scale]',
    '(enabled := value)', 'value * unknown',
])
def test_unclosed_return_grammar_is_not_relabelled_supported(expression):
    with pytest.raises(ImportedAliasContractError):
        _contract(donor=DONOR.replace('value * scale if enabled else value', expression))


def test_signature_projection_executes_real_lean_and_z3_and_rejects_false_claim(tmp_path):
    provers = _provers({})
    projection = _contract().formal_projection('qualification:expanded-alias-binding')
    source = tmp_path / 'Binding.lean'
    source.write_text(projection['lean'])
    run = subprocess.run([str(provers['kernel_executable']), str(source)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert set(run.stdout.strip().splitlines()) == set(projection['expected_axioms'])
    run = subprocess.run([shutil.which('z3'), '-in'], input=projection['smt'],
                         capture_output=True, text=True, timeout=15)
    assert run.returncode == 0 and run.stdout.strip() == 'unsat'
    # A supplied keyword must be allowed by the declared signature. Independently
    # falsify its parameter set in each native language, not the proof result.
    source.write_text(projection['lean'].replace(
        'def keywordParameters : List String := ["scale", "enabled"]',
        'def keywordParameters : List String := ["scale"]'))
    run = subprocess.run([str(provers['kernel_executable']), str(source)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
    smt = projection['smt'].replace('(= "enabled" "enabled")', '(= "enabled" "unbound")')
    run = subprocess.run([shutil.which('z3'), '-in'], input=smt,
                         capture_output=True, text=True, timeout=15)
    assert run.returncode == 0 and run.stdout.strip() == 'sat'
