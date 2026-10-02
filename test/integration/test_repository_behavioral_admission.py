"""Actual native v2 preparation, cold owner replay and signed worker fencing."""
from copy import deepcopy
import json
import os
import uuid
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as gate
from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_handoff as finite
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
from test.integration.test_terminal_codebase_semantic_index import prepared
from test.api.test_finite_integer_codebase import finite_tools, finite_git
from test.api.test_finite_integer_plan_preview import preview_arguments


@pytest.fixture(scope='module')
def admitted(tmp_path_factory,finite_tools):
    retained=os.environ.get('RPI_BEHAVIORAL_QUALIFICATION_FIXTURE')
    if retained:
        root=Path(retained).resolve(strict=True)
        saved=json.loads((root/'native-test-fixture.json').read_text())
        assert saved['schema']=='behavioral-native-test-fixture@1'
        with IntentRepository(root/'intent.duckdb') as intent:
            yield dict(root=root,repository=root/'repository',intent=intent,
                declared=saved['declared'],result=saved['result'],controls=saved['controls'])
        return
    root=tmp_path_factory.mktemp('actual-behavioral-admission')
    generator=prepared.__wrapped__(root)
    index,repository,head,descriptor=next(generator)
    index.artifacts.root.chmod(0o700)
    catalog=IntentCodebaseCatalog(index)
    catalog.publish(repository,expected_head=head,manifest_cid=descriptor['manifest_cid'],operation_id='behavioral-admission')
    cache=FiniteCheckedCache(FormalVerificationCache(root/'proof'),index.artifacts)
    Path(catalog.catalog._database_path).chmod(0o600)
    cache.cache.path.chmod(0o600)
    options=preview_arguments(dict(index=index,expected_head=head,repository=repository,scheduler=None,output=root/'unused'),finite_tools)
    options.pop('output')
    (repository/'.runtime').mkdir(mode=0o755)
    with IntentRepository(root/'intent.duckdb') as intent:
        declared=prepare_local_task(repository=repository,state=root/'policy',intent=intent,
            scope_paths=['calc.py','unsupported.py','README.md','.gitignore'],output_path='calc.py',
            objective=options['source_text'],validation_argv=['python3','-B','-c','from calc import increment; assert increment(0)==2'])
        result=gate.prepare_behavioral_repository_handoff(catalog=catalog,checked_cache=cache,
            semantic_manifest_cid=descriptor['manifest_cid'],**options,admission=declared['admission'],
            intent=intent,task_cid=declared['task_cid'],state=root/'preparation')
        index.catalog.store._connection.close()
        binding=result['signed_evidence']['binding']
        controls=dict(artifact=result['repository_admission_path'],expected_sha256=result['repository_admission_sha256'],
            task_cid=declared['task_cid'],owner_did=binding['identity'],profile_id=binding['profile_id'])
        # Retain actual preparation for independent cold replay after resource
        # refusal; all live gate checks still execute on every reused fixture.
        (root/'native-test-fixture.json').write_text(json.dumps(dict(
            schema='behavioral-native-test-fixture@1',declared=declared,result=result,controls=controls)))
        yield dict(root=root,repository=repository,intent=intent,declared=declared,result=result,controls=controls)
    try:next(generator)
    except StopIteration:pass


def worker_options(admitted,name):
    workspace=admitted['root']/(name+'-'+uuid.uuid4().hex)
    finite_git(admitted['repository'],'worktree','add','--detach',str(workspace),'HEAD')
    value=admitted['result'];binding=value['signed_evidence']['binding']
    return dict(repository_admission=value['repository_admission_path'],
        repository_admission_sha256=value['repository_admission_sha256'],artifact=value['handoff_path'],
        expected_sha256=value['handoff_sha256'],task_cid=admitted['declared']['task_cid'],
        owner_did=binding['identity'],profile_id=binding['profile_id'],
        public_context=value['public_context']['artifact'],public_context_sha256=value['public_context']['sha256'],
        prompt=json.dumps(dict(objective_id=admitted['declared']['task_id'],revision=42)),workspace=workspace)


def test_actual_v2_gate_reopens_native_owners_and_worker_without_private_keys(admitted):
    result=admitted['result']
    assert result['status']=='candidate_ready'
    assert result['behavioral_preview']['repository_proof_snapshot']['schema']=='repository-proof-planning-snapshot@2'
    options=worker_options(admitted,'allocated-success')
    private=admitted['root']/'policy';hidden=admitted['root']/'retained-private-policy'
    private.rename(hidden)
    try:materialized=runner.materialize_behavioral_candidate(**options)
    finally:hidden.rename(private)
    assert materialized['status']=='candidate_materialized'
    verified=materialized['repository_evidence_fence']['before']
    assert verified['fresh_native_checkers'] and not verified['private_owner_keys_used']
    assert verified['native_task_population']==[admitted['declared']['task_cid']]
    assert materialized['repository_evidence_fence']['source_observed_before_and_after']
    assert (options['workspace']/'calc.py').read_text().endswith('return n + 2\n')
    assert (admitted['repository']/'calc.py').read_text().endswith('return n + 1\n')
    assert admitted['intent'].get_task(admitted['declared']['task_cid'])['status']=='ready'


@pytest.mark.parametrize('change',['authority','model','snapshot_fact','snapshot_requirement','retained_blob','match_meaning'])
def test_resigned_mutations_cannot_invent_current_proof_or_planning_meaning(admitted,change,monkeypatch):
    # Artifact-integrity control only: the fixture's actual checked match is
    # injected to reach the corrupted artifact without scheduling another
    # checker. Live before/after source and worker fences are tested separately.
    monkeypatch.setattr(gate,'match_behavioral_intent',
        lambda **kwargs: deepcopy(admitted['result']['behavioral_preview']['behavioral_match']))
    signed=json.loads(Path(admitted['controls']['artifact']).read_text())
    value=deepcopy(signed['payload'])
    if change=='authority':value['execution_authority']=True
    elif change=='model':value['proof_snapshot']['model']['enabled']=True
    elif change=='snapshot_fact':value['proof_snapshot']['current_facts']=[]
    elif change=='snapshot_requirement':value['proof_snapshot']['requirements']=[]
    elif change=='retained_blob':value['behavioral_initial']['artifacts_base64']['lean_olean']='YQ=='
    else:value['match_semantics']['residuals']=[]
    snapshot=value['proof_snapshot'];snapshot['snapshot_cid']=cid_for_structured({k:v for k,v in snapshot.items() if k!='snapshot_cid'})
    value['artifact_cid']=content_identity({k:v for k,v in value.items() if k!='artifact_cid'})
    verified=local.verify_local_benchmark_admission(admitted['declared']['admission'])
    path,digest=finite._public_artifact(admitted['repository'],local._signed(value,verified['manifest']))
    with pytest.raises(ValueError):
        gate.verify_behavioral_repository_handoff(**{**admitted['controls'],'artifact':path,'expected_sha256':digest})


def test_missing_native_source_database_is_not_recreated(admitted):
    value=json.loads(Path(admitted['controls']['artifact']).read_text())['payload']
    path=Path(value['owner_roots']['source_database']['path']);hidden=path.with_name('hidden-existing.duckdb')
    path.rename(hidden)
    try:
        with pytest.raises((ValueError,OSError)):
            gate.verify_behavioral_repository_handoff(**admitted['controls'])
        assert not path.exists()
    finally:hidden.rename(path)


def test_dirty_dependency_before_dispatch_refuses_signed_snapshot(admitted):
    path=admitted['repository']/'README.md';before=path.read_bytes()
    path.write_bytes(before+b'Changed after signed planning.\n')
    try:
        with pytest.raises(ValueError):gate.verify_behavioral_repository_handoff(**admitted['controls'])
    finally:path.write_bytes(before)


def test_dependency_change_during_private_write_fails_worker_without_completion(admitted,monkeypatch):
    options=worker_options(admitted,'allocated-raced')
    actual=runner.materialize_with_public_context
    path=admitted['repository']/'README.md';before=path.read_bytes()
    def race(**kwargs):
        result=actual(**kwargs)
        path.write_bytes(before+b'Changed during private candidate write.\n')
        return result
    monkeypatch.setattr(runner,'materialize_with_public_context',race)
    try:
        with pytest.raises(ValueError):runner.materialize_behavioral_candidate(**options)
    finally:path.write_bytes(before)
    assert admitted['intent'].get_task(admitted['declared']['task_cid'])['status']=='ready'
    assert (admitted['repository']/'calc.py').read_text().endswith('return n + 1\n')
