"""The real module entrypoint must retain exact native callback identities."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_runtime as runtime_module


def test_restarted_candidate_cli_preserves_the_original_publication_witness(tmp_path):
    plan = tmp_path / 'plan.json'
    runtime_module.atomic_json(plan, {'tasks': [{'id': 'WWL-002', 'repository': 'accelerate'}]})
    config = runtime_module.CampaignConfig(site_root=tmp_path / 'site',
        framework_root=Path(__file__).resolve().parents[2], state_root=tmp_path / 'state',
        plan_path=plan, config_path=tmp_path / 'config.json')
    request = config.state_root / 'candidate.json'
    runtime_module.atomic_json(request, {'run_id': config.run_id, 'task_id': 'WWL-002'})
    pending = config.state_root / 'control/pending-candidate-continuation.json'
    runtime_module.atomic_json(pending, {'schema': 'wwl.pending-candidate-continuation@1',
        'run_id': config.run_id, 'request_path': str(request),
        'request_sha256': hashlib.sha256(request.read_bytes()).hexdigest()})
    before, contents = pending.stat(), pending.read_bytes()
    with patch.object(runtime_module, 'config_from_args', return_value=config), \
            patch.object(runtime_module, 'wait_for_storage', return_value={'blocked': False}), \
            patch.object(runtime_module, 'bootstrap', return_value={'admitted_plan_revision': 'fixture', 'receipt_id': 'fixture'}), \
            patch.object(runtime_module.CampaignRuntime, 'run', return_value={}), \
            patch.object(runtime_module.signal, 'signal'), \
            patch('ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_migration._registered_stopped_fence',
                return_value={'owner_alive': False, 'main_pid': 0, 'cgroup_empty': True}):
        assert runtime_module.main(['continue-candidate', '--request', str(request)]) == 1
    after = pending.stat()
    assert pending.read_bytes() == contents
    assert (before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_mtime_ns, after.st_ctime_ns)


def test_real_module_cli_delegates_to_canonical_main(tmp_path):
    module = runtime_module.MODULE
    # An authored startup observer replaces only canonical main. The real
    # `python -m` loader still executes the runtime as __main__. Without the
    # delegation, it uses a second CampaignRuntime class and ignores this main.
    (tmp_path / 'sitecustomize.py').write_text(
        'import importlib,json,sys\n'
        f'runtime = importlib.import_module({module!r})\n'
        'def probe_main(argv=None):\n'
        '    print(json.dumps({"canonical_entrypoint":True,"runtime_module":runtime.CampaignRuntime.__module__,"config_module":runtime.CampaignConfig.__module__,"argv":sys.argv[1:]}))\n'
        '    return 0\n'
        'runtime.main = probe_main\n')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(tmp_path) + os.pathsep + str(Path(__file__).resolve().parents[2])
    process = subprocess.run([sys.executable,'-B','-m',module,'--help'],
        cwd=tmp_path,env=environment,capture_output=True,text=True,timeout=45)
    assert process.returncode == 0, process.stderr
    records = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
    assert records == [{'canonical_entrypoint':True,'runtime_module':module,'config_module':module,'argv':['--help']}]


@pytest.mark.parametrize('retained_before_pass',[False,True])
def test_pending_ready_candidate_cannot_silently_spin_without_native_selection(tmp_path,retained_before_pass):
    plan = tmp_path / 'plan.json'
    runtime_module.atomic_json(plan,{'tasks':[{'id':'WWL-002','repository':'accelerate'}]})
    config = runtime_module.CampaignConfig(site_root=tmp_path / 'site',framework_root=Path(__file__).resolve().parents[2],
        state_root=tmp_path / 'state',plan_path=plan,config_path=tmp_path / 'config.json')
    runtime = runtime_module.CampaignRuntime(config,{'admitted_plan_revision':'fixture','receipt_id':'fixture'})
    runtime._candidate_request_path = config.state_root / 'candidate.json'
    runtime._cached = {'ready_task_ids':['WWL-002'],'active_task_id':'','stage':'selection'}
    calls = [0]
    def running():
        calls[0] += 1
        return [object()] if calls[0] == 1 and retained_before_pass else []
    runtime._daemon = SimpleNamespace(list_running_attempts=running,run_once=lambda:{
        'active_task_id':'','implementation_result':None,'selection_idle_reason':'no_ready_tasks','unchanged':True,'write_count':0})
    with patch.object(runtime,'capture'),patch.object(runtime,'_read_candidate_request',return_value={'task_id':'WWL-002'}),\
            patch.object(runtime_module,'git',return_value='fixture-source'):
        runtime._run_pass()
    assert runtime._passes == 1
    assert runtime._last_failure['error_class'] == 'RuntimeError'
    assert runtime._last_failure['reason'] == 'pending_native_candidate_ready_but_native_selection_is_idle'
    assert runtime._last_failure['task_id'] == 'WWL-002'
    assert runtime._last_failure['effects_unknown'] is retained_before_pass
    assert runtime._last_failure['provider_callback_entered'] is False
    assert runtime._last_successful_pass == {}
    assert runtime._last_failure['source_frames']
