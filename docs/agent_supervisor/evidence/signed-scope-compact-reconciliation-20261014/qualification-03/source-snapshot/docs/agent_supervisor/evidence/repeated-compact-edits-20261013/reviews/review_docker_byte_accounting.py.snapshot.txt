"""Review retained counters and hashes; never export source or provider text."""
import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parent
R = A / 'docker/run-01'
Q = Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    assert len(raw) <= 1024 * 1024, 'receipt exceeds bound'
    return json.loads(raw)


def binding(path):
    raw = path.read_bytes()
    return {'sha256': sha(raw), 'bytes': len(raw)}


def main():
    release_path = A / 'docker/release-02.json'
    release = read(release_path)
    result = read(R / 'qualification.json')
    probe = read(R / 'results/probe-qualification.json')
    observation = read(R / 'results/session-observation.json')
    worker = read(R / 'private-fixture-ipc/worker-result.json')
    audits = list((R / 'private-worker-observations').glob('*audit.json'))
    assert len(audits) == 1
    audit = read(audits[0])
    assert all(result[key] is True for key in ('complete', 'source_unchanged', 'runner_unchanged', 'container_removed'))
    assert result['owner_returncode'] == 0 and result['probe'] == probe
    assert all(sha(Path(path).read_bytes()) == digest for path, digest in release['source_bindings'].items())
    assert all(sha(Path(path).read_bytes()) == digest for path, digest in release['runner_bindings'].items())
    assert all(sha((R / 'runner-snapshot' / Path(path).name).read_bytes()) == digest for path, digest in release['runner_bindings'].items())
    assert probe['owner_uid'] == 1000 and probe['worker_uid'] == 1001 and probe['retained_owner_inode'] is True
    assert probe['retained_owner_uid'] == 1000 and probe['retained_source_mode'] == 0o660
    assert all(probe[key] is True for key in ('owner_private_read_denied', 'worker_no_new_privileges', 'worker_capabilities_zero', 'canonical_unchanged', 'support_input_unchanged', 'descriptor_removed', 'edit_validation_pending'))
    assert probe['validation_returncode'] == 0 and probe['validation_worker_uid'] == 1001 and probe['remaining_worker_pids'] == []
    assert worker['confirmed_edit_count'] == probe['confirmed_edit_count'] == 3
    steps, calls = worker['edit_steps'], worker['cli_calls']
    assert len(steps) == 3 and len(calls) == probe['cli_call_count'] == 7
    assert len({worker['initial_generation'], *(step['result']['workspace_generation'] for step in steps)}) == 4
    assert len({step['result']['proposal_sha256'] for step in steps}) == 3
    for index, step in enumerate(steps):
        confirmed = step['result']
        assert step['ordinal'] == index + 1
        assert confirmed['ok'] is True and confirmed['status'] == 'applied_validation_pending'
        assert confirmed['validation_pending'] is True and confirmed['automatic_retry_performed'] is False
        assert confirmed['atomic_application_claimed'] is False and confirmed['completion_authority'] is False
        assert confirmed['owner_postimage_verified'] is True and confirmed['write_mode'] == 'preserved_owner_inode'
        assert confirmed['workspace_generation'] != step['before_generation']
        for key in ('source', 'view', 'manifest'):
            assert step['refreshed_' + key + '_sha256'] == confirmed[key + '_sha256']
            assert step['before_' + key + '_sha256'] != confirmed[key + '_sha256']
            if index:
                assert step['before_' + key + '_sha256'] == steps[index - 1]['result'][key + '_sha256']
        if index:
            assert step['before_generation'] == steps[index - 1]['result']['workspace_generation']
        rendered = wire(confirmed) + b'\n'
        assert calls[index * 2 + 1]['stdout_bytes'] == len(rendered)
        assert calls[index * 2 + 1]['stdout_sha256'] == sha(rendered)
    assert worker['final_generation'] == steps[-1]['result']['workspace_generation'] == probe['final_generation_sha256']
    assert worker['source_hash'] == steps[-1]['result']['source_sha256'] == probe['candidate_sha256']
    assert all(type(call['stdout_bytes']) is int and call['stdout_bytes'] > 0 and call['returncode'] == 0 and call['stderr_bytes'] == 0 for call in calls)
    cli_bytes = sum(call['stdout_bytes'] for call in calls)
    assert cli_bytes == probe['cli_stdout_bytes'] and probe['cli_stderr_bytes'] == 0
    assert audit['accounting_complete'] is True and audit['service_stopped'] is True
    requests = audit['requests']
    assert len(requests) == audit['accepted_requests'] == observation['counts']['accepted_requests'] == 10
    assert all(row['delivered'] is True and row['status'] == 'ok' and row['owner_writes_performed'] is False for row in requests)
    assert sum(row['source_edit_operation'] == 'prepare' for row in requests) == 3
    assert sum(row['source_edit_operation'] == 'confirm' for row in requests) == 3
    response_bytes = sum(row['response_bytes'] for row in requests)
    assert response_bytes == audit['response_bytes'] == observation['counts']['delivered_response_bytes'] == observation['counts']['response_bytes']
    source_bytes = sum(row['returned_source_bytes'] for row in requests)
    assert source_bytes == observation['counts']['delivered_source_bytes']
    assert observation['source_edit_mode'] == 'mapped-span@2' and observation['status'] == 'observed'
    assert observation['source_edit_counts'] == dict(prepared_mapping_pages=3, delivered_mapping_pages=3, prepared_confirmation_pages=3, delivered_confirmation_pages=3)
    assert observation['sessions'][0]['audit_sha256'] == sha(audits[0].read_bytes())
    framing_path = Q / 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py'
    assert release['source_bindings'][str(framing_path)] == sha(framing_path.read_bytes())
    assert 'channel.sendall(len(raw).to_bytes(4, "big") + raw)' in framing_path.read_text()
    assert probe['original_expansion_requests'] == worker['original_source_expansion_requests'] == 0
    assert probe['external_model_calls'] == probe['provider_calls'] == probe['source_obligations_discharged'] == 0
    assert not probe['official_verifier_executed'] and not probe['provider_token_usage_measured'] and not probe['full_supervisor_start_qualified']
    assert result['native_provider_tokens'] is None and result['official_terminal_bench_score'] is None
    output = {
        'schema': 'repeated-compact-edit-byte-accounting-review@1', 'status': 'passed',
        'confirmed_edits': 3, 'distinct_working_generations': 4, 'fresh_source_view_manifest_pins_each_edit': True,
        'same_actual_owner_uid_and_inode_retained': True, 'actual_worker_uid': 1001,
        'separate_authored_validation_returncode': 0, 'owner_private_read_denied': True,
        'model_visible_cli_output_observation': {
            'authored_cli_calls': 7, 'source_reads': 4, 'edit_invocations': 3,
            'stdout_bytes_including_actual_cli_newlines': cli_bytes,
            'stdout_newline_bytes': 7, 'canonical_json_bytes_excluding_cli_newlines': cli_bytes - 7,
            'source_read_stdout_bytes_including_newlines': sum(calls[index]['stdout_bytes'] for index in (0, 2, 4, 6)),
            'edit_result_stdout_bytes_including_newlines': sum(calls[index]['stdout_bytes'] for index in (1, 3, 5)),
            'edit_result_rendered_bytes_and_hashes_replayed_exactly': True, 'stderr_bytes': 0,
            'scope': 'Actual captured subprocess stdout in an authored provider hook; these are not provider token measurements.'},
        'private_owner_response_transport': {
            'successful_delivered_responses': 10, 'prepare_pages': 3, 'confirmation_pages': 3, 'source_pages': 4,
            'canonical_json_response_body_bytes': response_bytes, 'derived_frame_prefix_bytes': 40,
            'derived_framed_response_bytes': response_bytes + 40,
            'delivered_source_slice_bytes_including_repeated_reads': source_bytes,
            'scope': 'Owner-to-client responses only; ten four-byte prefixes follow the pinned framing implementation. Prepare/confirm owner frames stay private to the client.'},
        'original_source_expansion_requests': 0, 'full_exchange_bytes': None, 'provider_tokens': None,
        'official_benchmark_reward': None, 'source_obligations_discharged': 0,
        'external_model_calls': 0, 'full_supervisor_start_qualified': False,
        'complete_session_savings_measured': False, 'comparison_baseline_present': False,
        'non_additive_scopes': 'CLI results and private owner responses are two representations of overlapping operations; do not add them as model exposure.',
        'excluded_from_cli_observation': ['CLI command arguments and edit JSON stdin', 'initial model prompt and edit guidance', 'provider tool framing and conversation history', 'subsequent model turns', 'separate authored smoke command/output'],
        'unmeasured_transport': ['Request body byte counts are not retained by the session audit; total bidirectional wire bytes are unknown.'],
        'bindings': {name: binding(path) for name, path in {
            'release': release_path, 'docker_result': R / 'qualification.json', 'probe_result': R / 'results/probe-qualification.json',
            'session_observation': R / 'results/session-observation.json', 'private_worker_metadata': R / 'private-fixture-ipc/worker-result.json',
            'private_audit': audits[0], 'framing_source': framing_path, 'review_helper': Path(__file__).resolve()}.items()},
    }
    path = A / 'docker-byte-accounting-review.json'
    with path.open('x') as handle:
        json.dump(output, handle, indent=2, sort_keys=True); handle.write('\n')
    print(json.dumps({'status': output['status'], 'cli_stdout_bytes': cli_bytes, 'owner_response_body_bytes': response_bytes,
        'owner_framed_response_bytes': response_bytes + 40, 'path': str(path), 'sha256': sha(path.read_bytes())}, sort_keys=True))


if __name__ == '__main__':
    main()
