"""One frozen eight-turn variant; shared verified archive, serial one-shot launch."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

A = Path(__file__).resolve().parent
R = json.loads((A / 'recipe.json').read_text())
W = Path(R['cwd'])

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def check_sources():
    assert sha(A / 'environment.json') == R['environment_sha256']
    for row in R['shared_input_bindings']:
        assert binding(row['path']) == row, row['path']
    for row in R['roots'].values():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=row['path'], text=True).strip() == row['commit']
        assert not subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=row['path'], text=True)
    for path, digest in R['host_source_sha256'].items():
        assert sha(path) == digest, path

def environment():
    value = json.loads((A / 'environment.json').read_text())
    env = dict(os.environ)
    for key in value['scrub_inherited_names']:
        env.pop(key, None)
    env.update(value['environment_overrides'])
    return env

def binding(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)

def verify():
    # Reuse the byte-bound complete inventory verification; do not copy or expand the archive.
    basis = R['parent_verification']
    assert binding(basis['path']) == basis
    verified = json.loads(Path(basis['path']).read_text())
    assert verified['exit_code'] == 0 and verified['all_archive_members_match'] is True
    assert verified['source_pins_unchanged'] is True
    for key in ('archive', 'manifest'):
        assert binding(verified[key]['path']) == verified[key]
    manifest = json.loads(Path(verified['manifest']['path']).read_text())
    assert manifest['archive_sha256'] == verified['archive']['sha256']
    env = environment()
    os.environ.clear()
    os.environ.update(env)
    sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_coding_policy import require_doctor_coding_archive
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
    from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import require_worker_capability
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import require_runtime_provider_profile
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import require_grok_coding_archive
    require_doctor_coding_archive(manifest, 2, 'doctor-coding-two-attempts@1')
    require_source_view_archive(manifest, 'identity@1', 'source384-5cpu-20gib-formalization@1')
    require_worker_capability(manifest, 'source384-5cpu-20gib-formalization@1')
    require_runtime_provider_profile(manifest, 'grok-4.7-cli-1.0.46@1')
    require_grok_coding_archive(manifest, 8)
    return dict(archive=verified['archive'], manifest=verified['manifest'],
        all_archive_members_match=True, complete_inventory_verification_reused=basis,
        archive_hash_rechecked=True, archive_copied=False, archive_expanded=False,
        current_doctor_policy_sources_match=True, current_grok_turn_policy_sources_match=True,
        provider_calls=0)

def run(step):
    assert step in ('verify', 'prepare', 'execute', 'collect')
    check_sources()
    assert not (A / (step + '-start.json')).exists(), 'immutable step already started'
    if step in ('prepare', 'execute', 'collect'):
        verified = json.loads((A / 'verify-result.json').read_text())
        for key in ('archive', 'manifest'):
            assert binding(verified[key]['path']) == verified[key]
    if step == 'execute':
        previous_execute = Path(R['serial_predecessor_execute_result'])
        previous_collect = previous_execute.with_name('collect-result.json')
        previous_receipt = Path(R['matched_parent']['output']) / 'receipt.json'
        assert previous_execute.is_file() and previous_collect.is_file() and previous_receipt.is_file(), 'serial predecessor execution/collection has not finished'
        assert json.loads(previous_execute.read_text())['source_pins_unchanged'] is True
        assert json.loads(previous_collect.read_text())['exit_code'] == 0
        clearance_path = A / 'serial-predecessor-clearance.json'
        clearance = json.loads(clearance_path.read_text())
        assert clearance['schema'] == 'doctor-turns8-serial-predecessor-clearance@1'
        for key, path in (('execute_result', previous_execute), ('collect_result', previous_collect), ('receipt', previous_receipt)):
            assert clearance[key] == binding(path)
        assert clearance['container_cleanup_observed'] is True
        assert clearance['remaining_trial_container_ids'] == []
        assert clearance['observed_after_collect'] is True
        frozen = json.loads((A / 'prepared-lock-02.json').read_text())
        for row in frozen['files']:
            assert binding(row['path']) == row, row['path']
        reporting_path = A / 'reporting-freeze.json'
        reporting = json.loads(reporting_path.read_text())
        for row in reporting['files']:
            assert binding(row['path']) == row
        readiness_path = Path(R['readiness']['path'])
        readiness = json.loads(readiness_path.read_text())
        assert readiness['provider_generation_calls'] == 0 and readiness['host_auth_unchanged'] is True
        assert readiness['container_and_tmpfs_removed'] is True and readiness['host_temporary_auth_copy_removed'] is True
        assert readiness['observation']['status'] == 'ready' and readiness['observation']['selected_model_observed'] is True
        out = Path(R['output'])
        assert not (out / 'invocation.json').exists(), 'no replay or automatic retry'
        prep, config = out / 'preparation.json', out / 'config.json'
        prepared = json.loads(prep.read_text())
        assert prepared['prepared'] and prepared['config_sha256'] == sha(config)
        assert prepared['candidate_rounds'] == 2 and prepared['doctor_coding_policy'] == 'doctor-coding-two-attempts@1'
        assert prepared['grok_coding_max_turns'] == 8
        assert json.loads(config.read_text())['agents'][0]['kwargs']['grok_coding_max_turns'] == 8
        assert R['limits']['coding_deadline_seconds'] == 1800
        assert prepared['archive_sha256'] == verified['archive']['sha256']
        write(A / 'launch-lock.json', dict(preparation=binding(prep), configuration=binding(config),
            runtime=verified, recipe=binding(A / 'recipe.json'), runner=binding(Path(__file__)),
            reporting=binding(reporting_path), readiness=binding(readiness_path),
            serial_clearance=binding(clearance_path),
            conditional_second_attempt=True, automatic_retries=0, provider_fallback=False))
    if step == 'collect':
        lock = json.loads((A / 'launch-lock.json').read_text())
        for key in ('preparation', 'configuration', 'recipe', 'runner', 'reporting', 'readiness', 'serial_clearance'):
            assert binding(lock[key]['path']) == lock[key]
        for row in json.loads(Path(lock['reporting']['path']).read_text())['files']:
            assert binding(row['path']) == row
    free = shutil.disk_usage(A).free
    if step == 'execute':
        assert free >= R['minimum_free_bytes'], 'physical free-space prerequisite'
    write(A / (step + '-start.json'), dict(step=step, started_at=datetime.now(timezone.utc).isoformat(),
        runner_sha256=sha(Path(__file__)), recipe_sha256=sha(A / 'recipe.json'), free_bytes=free,
        argv=R['commands'].get(step), cwd=str(W)))
    print(json.dumps(dict(step=step, status='started')), flush=True)
    before = time.monotonic()
    if step == 'verify':
        result, code = verify(), 0
    else:
        with (A / (step + '.stdout')).open('xb') as out, (A / (step + '.stderr')).open('xb') as err:
            process = subprocess.run(R['commands'][step], cwd=W, env=environment(), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err)
        code = process.returncode
        result = dict(stdout=binding(A / (step + '.stdout')), stderr=binding(A / (step + '.stderr')))
    check_sources()
    result.update(step=step, exit_code=code, seconds=time.monotonic()-before, source_pins_unchanged=True)
    write(A / (step + '-result.json'), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
