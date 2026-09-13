"""The independent runner must never overwrite historical cohort outputs."""
import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[4] / 'scripts/qualify_aseh_closeout_readonly.py'
_SPEC = importlib.util.spec_from_file_location('aseh_readonly_runner', _PATH)
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize('alias,cohort', list(RUNNER.COHORTS.items()))
def test_prospective_cohort_paths_preserve_source_command(alias, cohort, tmp_path):
    command = ('python3 ' + RUNNER.HARNESS + ' --cohort ' + cohort
               + ' --output old/results.json --qualification-output old/qualification.json')
    if alias == 'ASEH-073':
        command += ' --require-shadow-receipt old/shadow.json --allow-not-admitted'
    else:
        command += ' --allow-honest-nonpromotion'
    task = {'task_alias': alias, 'body': {'validation': command}}
    original = copy.deepcopy(task)
    result = RUNNER.rewritten_command(task, tmp_path)
    assert task == original
    assert result[result.index('--output') + 1] == str(tmp_path / (alias + '-results.json'))
    assert result[result.index('--qualification-output') + 1] == str(tmp_path / (alias + '-qualification.json'))
    if alias == 'ASEH-073':
        assert result[result.index('--require-shadow-receipt') + 1] == str(tmp_path / 'ASEH-072-results.json')


@pytest.mark.parametrize('command', [
    'python3 -m pytest -q ../outside.py',
    'python3 -m pytest -q /tmp/outside.py',
    'python3 -m pytest -q tests/ok.py --disable-warnings',
    'bash -c anything',
])
def test_closed_validator_surface(command, tmp_path):
    with pytest.raises(ValueError):
        RUNNER.rewritten_command({'task_alias': 'ASEH-000', 'body': {'validation': command}}, tmp_path)


def test_pytest_validator_preserves_exact_arguments(tmp_path):
    task = {'task_alias': 'ASEH-000', 'body': {'validation': 'python3 -m pytest -q tests/example.py'}}
    assert RUNNER.rewritten_command(task, tmp_path) == ['python3', '-m', 'pytest', '-q', 'tests/example.py']
