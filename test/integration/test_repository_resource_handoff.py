"""Actual cross-process child admission beneath one live native parent."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryResourceBridge, RepositoryResourceBudget, RepositoryPhaseDemand,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_handoff import (
    write_repository_resource_grant, delegated_repository_phase,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourceScheduler, ResourcePolicy


def reservation(tmp_path):
    return RepositoryResourceBridge(ResourceScheduler(ResourcePolicy(max_lanes=4))).reserve(
        repository_id='repository:delegation',workspace=tmp_path,
        budget=RepositoryResourceBudget(cpu_slots=2,memory_mb=2048,process_slots=2,wall_time_ms=180000))


def grant(parent,tmp_path):
    directory=tmp_path/'private-resource';directory.mkdir(mode=0o700)
    return write_repository_resource_grant(envelope=parent,directory=directory,task_cid='task:delegation',
        demand=RepositoryPhaseDemand('validation',memory_mb=512))


def arguments(value):
    return dict(artifact=value['artifact'],expected_sha256=value['sha256'],
        repository_id='repository:delegation',task_cid='task:delegation')


def test_real_subprocess_uses_parent_and_preserves_sibling_validation_capacity(tmp_path):
    with reservation(tmp_path) as parent:
        saved=grant(parent,tmp_path)
        request=tmp_path/'request.json';request.write_text(json.dumps(arguments(saved)))
        script='''
import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_handoff import delegated_repository_phase
with delegated_repository_phase(**json.loads(Path(sys.argv[1]).read_text())) as phase:
    with phase.native.acquire_child(memory_mb=128,child_process_slots=1,timeout=30) as child:
        assert child.parent_lease_id==phase.native.lease_id
    result=phase.receipt()
result['released']=phase.native.released
print(json.dumps(result))
'''
        with parent.phase(RepositoryPhaseDemand('validation',memory_mb=512)):
            result=subprocess.run([sys.executable,'-B','-c',script,str(request)],
                capture_output=True,text=True,timeout=150)
            assert result.returncode==0,result.stderr[-2048:]
        receipt=json.loads(result.stdout)
        assert receipt['parent_lease_id']==parent.native.lease_id
        assert receipt['lease']['parent_lease_id']==parent.native.lease_id
        assert receipt['released'] and receipt['private_token_disclosed'] is False
        assert 'lease_key' not in result.stdout and 'parent_token' not in result.stdout
        owned=[r for r in parent.shared.active_leases() if r['owner_pid']==os.getpid()]
        assert [r['lease_id'] for r in owned]==[parent.native.lease_id]
    assert parent.receipt()['closed'] and parent.receipt()['owned_active_root_count']==0


@pytest.mark.parametrize('damage',['task','repository','digest','mode','hardlink','directory_mode'])
def test_private_grant_identity_and_storage_refusals(tmp_path,damage):
    with reservation(tmp_path) as parent:
        saved=grant(parent,tmp_path);args=arguments(saved)
        path=Path(saved['artifact'])
        if damage=='task':args['task_cid']='task:foreign'
        elif damage=='repository':args['repository_id']='repository:foreign'
        elif damage=='digest':args['expected_sha256']='0'*64
        elif damage=='mode':path.chmod(0o600)
        elif damage=='hardlink':os.link(path,path.with_suffix('.alias'))
        else:path.parent.chmod(0o750)
        with pytest.raises(ValueError):
            with delegated_repository_phase(**args):pytest.fail('invalid private grant admitted')
        assert parent.receipt()['owned_active_root_count']==1


def test_released_parent_cannot_be_replayed_by_fresh_worker(tmp_path):
    with reservation(tmp_path) as parent:saved=grant(parent,tmp_path)
    with pytest.raises(ValueError,match='no longer live'):
        with delegated_repository_phase(**arguments(saved)):pytest.fail('released parent admitted')


def test_parent_cancellation_propagates_to_actual_delegated_phase(tmp_path):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    with pytest.raises(LeaseCancelledError):
        with reservation(tmp_path) as parent:
            saved=grant(parent,tmp_path)
            with delegated_repository_phase(**arguments(saved)) as phase:
                parent.cancel()
                assert phase.native.cancelled
                phase.remaining()
    assert parent.receipt()['closed'] and parent.receipt()['owned_active_root_count']==0
