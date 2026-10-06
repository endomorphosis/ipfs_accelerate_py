"""Bind learned retrieval and native usage to the exact completed trial10."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = A.parent.parent / '.worktrees/grok-recovery-20261006'
HEAD = 'b966623d3de7457b4139e7d9235a5976f275dd46'
REVISION = '17e1f347d17fe144873b1201da91788898c639cd'


def reference(path):
    raw = path.read_bytes()
    return {'path':str(path.relative_to(A)), 'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest()}


def write(name, value):
    with (A/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def main():
    index_path = A/'indexed-path-observation-10.json'
    index = json.loads(index_path.read_text())
    summary_path = A/'live-trial-summary-10.json'
    summary = json.loads(summary_path.read_text())['trials'][0]
    vector_path = A/'grok-container/learned-vector-runtime-observation-10.json'
    vector = json.loads(vector_path.read_text())
    manifest_path = A/'grok-bundle-09/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert summary['selection_binding']['source_revisions']['source'] == HEAD
    assert manifest['model_snapshot_revision'] == vector['model_revision'] == REVISION
    assert vector['status'] == 'qualified' and vector['ducklake_status'] == 'projected'
    assert vector['canary']['disposition'] == 'passed' and vector['canary']['vector_lane'] == 'enabled'
    assert index['actual_index']['learned_embeddings'] is True
    assert index['admitted_context']['learned_embeddings'] is True
    assert index['admitted_context']['same_initial_index_reference'] is True
    assert index['admitted_context']['new_embedding_calls'] == 0
    config = manifest['source384']['config']
    assert config['embedding_revision'] == REVISION
    assert config['checkpoint_sha256'] == index['source384']['checkpoint_sha256']
    assert index['source384']['neural_inference_replayed'] is False
    inventory = {row['path']:row for row in manifest['files']}
    source_hashes = {}
    for name in (
        'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
        'benchmarks/agent_supervisor/container_coding/terminal_program_population.py',
        'benchmarks/agent_supervisor/container_coding/learned_vector_preflight.py',
        'benchmarks/agent_supervisor/container_coding/terminal_retrieval_selection.py',
        'ipfs_accelerate_py/agent_supervisor/runtime/local_learned_embedding.py',
        'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
    ):
        raw = subprocess.check_output(['git','-C',str(P),'show',HEAD+':'+name])
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == inventory['source/'+name]['sha256']
        source_hashes[name] = digest
    write('retrieval-mode-observation-10.json',{
        'schema':'grok-retrieval-source384-distinction@2','trial_name':'grok-tune-mjcf-10',
        'source_head':HEAD,'source_sha256':source_hashes,'archive_manifest':reference(manifest_path),
        'indexed_observation':reference(index_path),'learned_runtime_observation':reference(vector_path),
        'retrieval_backend':'local-safetensors-symbols@1','retrieval_model':'thenlper/gte-small',
        'retrieval_model_revision':REVISION,'retrieval_device':'cpu','retrieval_dimensions':384,
        'embedding_input':'qualified symbol names only','indexed_symbols':4,'full_capsules':11,
        'initial_and_admitted_learned_embeddings':True,'same_initial_index_reused':True,
        'admitted_context_new_embedding_calls':0,'local_model_calls':vector['local_model_calls'],
        'local_model_texts':vector['local_model_texts'],'remote_embedding_calls':0,
        'embedding_canary_passed':True,'native_fact_rows_replayed':4,
        'vector_and_metadata_persistence':'DuckDB_with_DuckLake_metadata_projection',
        'ducklake_status':'projected','database_contents_independently_reopened_by_this_observer':False,
        'persistence_basis':'source_bound_runtime_qualification_and_reopen_path_completed',
        'source384_embedding_model':'thenlper/gte-small','source384_embedding_revision':REVISION,
        'source384_checkpoint_sha256':config['checkpoint_sha256'],'source384_fresh_inference':True,
        'source384_inference_python_files':index['source384']['counts']['inference_python_files'],
        'source384_training_steps':0,'source384_is_separate_from_retrieval_index':True,
        'nomination_only':True,'semantic_equivalence_claimed':False,'proof_authority':False,
        'full_system_qualified':False,'completion_authority':False,
        'raw_bodies_or_credentials_exported':False,
    })
    calls = summary['router_invocations']
    assert len(calls) == 2 and [row['phase'] for row in calls] == ['planning','coding']
    assert all(row['native_provider_outcome']['reason_code'] == 'end_turn' for row in calls)
    rows = []
    for call in calls:
        usage = call['native_usage']['usage']
        assert usage['input_tokens'] + usage['cached_input_tokens'] + usage['output_tokens'] == usage['total_tokens']
        rows.append({'phase':call['phase'],'model':call['model'],'provider':call['provider'],
            'seconds':call['seconds'],'timeout_seconds':call['timeout_seconds'],
            'native_tokens':usage,'native_envelope_observed':True})
    total = summary['native_usage']
    assert sum(row['native_tokens']['total_tokens'] for row in rows) == total['total_tokens']
    write('performance-observation-10.json',{
        'schema':'terminal-grok-observed-performance@1','trial_name':'grok-tune-mjcf-10',
        'source_head':HEAD,'live_summary':reference(summary_path),'official_reward':1.0,
        'native_task_status':'completed','driver_task_completed':False,'native_custody_closed':False,
        'provider_profile':'grok-4.7-cli-1.0.46@1','resource_profile':'source384-5cpu-20gib-coding600@1',
        'actual_resource_observation':reference(A/'grok-container/resource-observation-10.json'),
        'provider_invocations':rows,'native_observed_total_tokens':total['total_tokens'],
        'native_observed_input_tokens_excluding_cache':total['input_tokens'],
        'native_observed_cached_input_tokens':total['cached_input_tokens'],
        'native_observed_output_tokens':total['output_tokens'],
        'reasoning_tokens_are_subset_of_output':True,'cache_included_in_input':False,
        'all_selected_native_usage_envelopes_observed':True,'billing_total_verified':False,
        'dollar_cost':None,'benchmark_durations_seconds':summary['trial']['durations_seconds'],
        'matched_baseline_in_this_trial':False,'token_or_speed_advantage_claimed':False,
        'native_final_turn_is_task_completion_authority':False,
        'raw_model_task_verifier_or_credential_bodies_exported':False,
    })
    print(json.dumps({'learned_retrieval_qualified':True,'source384_fresh_inference':True,
        'official_reward':1.0,'native_tokens':total['total_tokens'],'native_custody_closed':False}))


if __name__ == '__main__':
    main()
