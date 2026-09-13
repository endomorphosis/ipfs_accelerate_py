"""The independent runner must never overwrite historical cohort outputs."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

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


def test_write_guard_honors_dir_fds_and_symlink_entry_semantics(tmp_path):
    script = '''
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('runner', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base = pathlib.Path(sys.argv[2]); protected = base/'protected'; scratch = base/'scratch'
protected.mkdir(); scratch.mkdir(); (protected/'record').write_text('original')
os.chdir(protected)
fd = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY)
module.install_readonly_guard([protected])
os.mkdir('reservation', dir_fd=fd)
w = os.open('output', os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd)
os.write(w, b'temporary'); os.close(w)
os.rename('output', 'renamed', src_dir_fd=fd, dst_dir_fd=fd)
os.symlink(str(protected/'record'), 'link', dir_fd=fd)
assert (scratch/'link').read_text() == 'original'
for path in [protected/'record', scratch/'link']:
    try: path.write_text('forbidden')
    except PermissionError: pass
    else: raise AssertionError('protected bytes were writable')
try: os.open('record', os.O_WRONLY)
except PermissionError: pass
else: raise AssertionError('cwd write escaped guard')
assert (protected/'record').read_text() == 'original'
os.unlink('link', dir_fd=fd); os.unlink('renamed', dir_fd=fd)
os.rmdir('reservation', dir_fd=fd); os.close(fd)
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(_PATH), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
