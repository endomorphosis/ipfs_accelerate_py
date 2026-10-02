"""Actual independent inventory/proof arms under one native host envelope."""
from dataclasses import replace
import json
import os
from pathlib import Path
import importlib.util

import pytest

from benchmarks.agent_supervisor.container_coding import repository_benchmark_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryResourceBridge, RepositoryResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourceScheduler, ResourcePolicy
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import CodebaseScanPolicy
from test.integration.test_terminal_codebase_semantic_index import prepared
from test.api.test_finite_integer_codebase import finite_tools


@pytest.mark.parametrize('proof', [False, True])
def test_native_model_off_arms_keep_complete_inventory_and_independent_proof_population(prepared,finite_tools,tmp_path,proof):
    index,repository,head,_=prepared
    contract=IntegerOffsetContract('calc.py','increment','n',2)
    selection=prep.RepositoryPreparationSelection(CodebaseScanPolicy(),
        proof_contracts=(contract,) if proof else (), proof_inputs=((-2,-1,0,1,2),) if proof else ())
    supervisor=ResourceScheduler(ResourcePolicy(max_lanes=8))
    bridge=RepositoryResourceBridge(supervisor)
    with bridge.reserve(repository_id=head.repository_id,workspace=tmp_path,
            budget=RepositoryResourceBudget(memory_mb=4096,wall_time_ms=180000)) as envelope:
        result=prep.prepare_repository_benchmark(index=index,repository=repository,
            repository_id=head.repository_id,expected_head=head,operation_id='arm:'+str(proof),
            selection=selection,envelope=envelope,
            checked_cache=FiniteCheckedCache(FormalVerificationCache(tmp_path/'proof'),index.artifacts) if proof else None,
            tool_policy=finite_tools if proof else None)
        assert result['qualified']
        assert result['complete_inventory']['inventory_entries']==4
        assert result['selection']['training_selections']==[]
        assert result['training'] is result['inference'] is None
        assert result['model']=={'enabled':False,'identity':'explicit-model-off@1'}
        assert len(result['checked_proofs'])==int(proof)
        if proof:assert result['checked_proofs'][0]['status']=='refuted'
        assert result['resource_receipt']['owned_active_root_count']==1
        assert all(s['status']=='completed' for s in result['stages'])
        assert result['model_promotion_performed'] is result['execution_authority'] is result['benchmark_score'] is False
    assert envelope.receipt()['closed'] and envelope.receipt()['owned_active_root_count']==0
    assert not supervisor.active_leases
    output=os.environ.get('RPI_PREPARATION_EVIDENCE')
    if output:
        path=Path(output);path.mkdir(parents=True,exist_ok=True)
        (path/('proof.json' if proof else 'inventory.json')).write_text(json.dumps(dict(result=result,closed=envelope.receipt()),indent=2,sort_keys=True)+'\n')


@pytest.mark.parametrize('changes', [
    dict(model_policy='implicit'),dict(training_selections=(('calc.py','train','one'),)),
    dict(inference_paths=('calc.py',)),dict(model_policy='required_training',inference_paths=('calc.py',)),
    dict(proof_contracts=(IntegerOffsetContract('calc.py','increment','n',2),)),
    dict(proof_inputs=((0,),)),
])
def test_closed_arm_selection_refuses_implicit_work(changes):
    with pytest.raises(ValueError):prep.RepositoryPreparationSelection(CodebaseScanPolicy(),**changes)


def test_phase_budget_refuses_boolean_and_unbounded_values():
    for kwargs in [dict(phase_seconds=True),dict(phase_seconds=float('inf')),dict(memory_mb=128),dict(max_proof_contracts=17)]:
        with pytest.raises(ValueError):prep.PreparationBudget(**kwargs)


def test_no_regressed_child_can_be_selected():
    original={'parent_holdout':{'count':3,'exact_targets':3},
        'child_holdout':{'count':3,'exact_targets':0,'valid_candidates':3}}
    assert not prep._retained(original)
    original['child_holdout']['exact_targets']=3
    assert prep._retained(original)
    original['child_holdout']['valid_candidates']=2
    assert not prep._retained(original)


# Reuse the real nine-source native fixture and explicit checkpoint selection.
# There is no synthetic checkpoint fallback in the acceptance run.
_fixture_path = Path(__import__('ipfs_datasets_py').__file__).parent.parent / 'tests/integration/logic/software_contracts/test_codebase_training_lifecycle.py'
_fixture_spec = importlib.util.spec_from_file_location('preparation_source384_fixture', _fixture_path)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
current = _fixture_module.current


@pytest.mark.parametrize('policy', ['pinned_parent','optional_training','required_training'])
def test_actual_parent_inference_and_failed_training_policy_are_distinct(current,policy,monkeypatch):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import RUN_LIFECYCLE_SCHEMA
    from ipfs_datasets_py.logic.software_contracts import codebase_training_lifecycle as lifecycle
    rows=tuple((v['path'],v['role'],v['group_id']) for v in current.selections)
    selected=prep.RepositoryPreparationSelection(CodebaseScanPolicy(),
        training_selections=rows if policy!='pinned_parent' else (),
        inference_paths=('holdout_0.py','holdout_1.py','holdout_2.py'),model_policy=policy)
    model=prep.SourceModelSelection(current.registry,current.parent,'main',current.model_head,
        os.environ['CODEBASE384_EMBEDDING_SNAPSHOT'], lifecycle_policy=dict(
            schema=RUN_LIFECYCLE_SCHEMA,max_attempts=1,wall_time_seconds=90.,memory_bytes=4096*1024**2,
            max_input_bytes=32*1024**2,max_samples=9,optimizer_steps=0,head_refits=1,
            max_checkpoint_bytes=32*1024**2,expected_head=current.model_head))
    if policy=='pinned_parent':
        monkeypatch.setattr(lifecycle,'execute_job',lambda *a,**k:pytest.fail('pinned-parent arm trained'))
    supervisor=ResourceScheduler(ResourcePolicy(max_lanes=8))
    with RepositoryResourceBridge(supervisor).reserve(repository_id=current.head.repository_id,
            workspace=current.root,budget=RepositoryResourceBudget(memory_mb=4096,wall_time_ms=240000)) as envelope:
        arguments=dict(index=current.index,repository=current.repo,repository_id=current.head.repository_id,
            expected_head=current.index.current(current.head.repository_id),operation_id='prepare:'+policy,
            selection=selected,envelope=envelope,model=model)
        if policy=='required_training':
            with pytest.raises(prep.RequiredTrainingQualificationError) as caught:
                prep.prepare_repository_benchmark(**arguments)
            result=caught.value.report
            assert not result['qualified'] and result['inference'] is None
            assert result['training']['training_executed']
        else:
            result=prep.prepare_repository_benchmark(**arguments)
            assert result['qualified'] and not result['inference']['training_executed']
            assert result['inference']['provenance_kind']=='pinned_shared_parent_inference'
            assert len(result['inference']['inference']['rows'])==3
            assert all(v['source_contract']['status']=='qualified' for v in result['inference']['inference']['rows'])
            if policy=='optional_training':
                assert result['model']['fallback_reason']=='child_unqualified_or_incomplete'
                assert result['training']['training_executed']
            else:assert result['training'] is None
        assert current.registry.resolve_head(current.variant,'main')==current.model_head
        assert result['evaluation_population_use']==dict(training_holdout='development_retention_gate',
            holdout_used_for_runtime_selection=policy!='pinned_parent',untouched_final_benchmark_test=False)
    current.results.append(result)
    assert envelope.receipt()['closed'] and not supervisor.active_leases
    output=os.environ.get('RPI_PREPARATION_EVIDENCE')
    if output:
        (Path(output)/(policy+'.json')).write_text(json.dumps(dict(result=result,closed=envelope.receipt()),indent=2,sort_keys=True)+'\n')


def test_cancelled_parent_refuses_scan_without_advancing_source(prepared,tmp_path):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    index,repository,head,_=prepared
    supervisor=ResourceScheduler(ResourcePolicy(max_lanes=8))
    with pytest.raises(LeaseCancelledError):
        with RepositoryResourceBridge(supervisor).reserve(repository_id=head.repository_id,
                workspace=tmp_path,budget=RepositoryResourceBudget(memory_mb=4096,wall_time_ms=180000)) as envelope:
            envelope.cancel()
            prep.prepare_repository_benchmark(index=index,repository=repository,
                repository_id=head.repository_id,expected_head=head,operation_id='cancelled-preparation',
                selection=prep.RepositoryPreparationSelection(CodebaseScanPolicy()),envelope=envelope)
    assert index.current(head.repository_id)==head
    assert not supervisor.active_leases and envelope.receipt()['closed']
