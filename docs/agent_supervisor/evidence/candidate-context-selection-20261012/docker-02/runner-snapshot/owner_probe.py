"""UID1000 owner: public-source preflight and actual isolated worker launcher.

The model call is an authored hook. Source bytes, native context, broker, codec,
CLI, Unix identities and cleanup are real. This does not measure token savings.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path('/opt/ipfs-supervisor')
PRIVATE = ROOT / 'state'
CONTROL = ROOT / 'fixture-ipc'
sys.path[:0] = [str(ROOT / name) for name in ('source', 'datasets', 'kit')]


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_bytes(wire(value))


def data(page):
    return page['content'].encode() if page['encoding'] == 'utf-8' else base64.b64decode(page['content'], validate=True)


def main():
    assert os.getuid() == os.geteuid() == os.getgid() == 1000
    outcome = dict(schema='source-view-worker-docker-probe@1', complete=False,
        provider_calls=0, model_calls=0, solver_calls=0, native_provider_tokens=None,
        paid_token_savings_measured=False, source_obligations_discharged=0,
        source_semantics_verified=False, official_verifier_executed=False,
        full_live_supervisor_deployment_qualified=False, authored_planning=True,
        intent_autoencoder_disabled=True, learned_source384_inference=False,
        source_view_profile='python-comments@1', edits_mapped=False)
    process = None
    started = time.monotonic()
    try:
        for name in ('OPENAI_API_KEY', 'XAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'):
            assert not os.environ.get(name)
        import duckdb
        con = duckdb.connect(':memory:')
        platform = con.execute('PRAGMA platform').fetchone()[0]
        extension_dir = Path.home() / '.duckdb/extensions' / ('v' + duckdb.__version__) / platform
        extension_dir.mkdir(parents=True, exist_ok=True)
        for name in ('ducklake', 'httpfs', 'quack'):
            source = Path('/extensions') / (name + '.duckdb_extension')
            target = extension_dir / source.name
            if target.exists():
                assert sha(target) == sha(source)
            else:
                shutil.copyfile(source, target)
            con.execute('LOAD ' + name)
        con.close()
        repository = ROOT / 'repository'
        assert repository.is_dir() and not any(repository.iterdir())
        for name in ('eigen.py', 'eval.py'):
            shutil.copyfile(Path('/permitted') / name, repository / name)
        expected_hashes = {name: sha(repository / name) for name in ('eigen.py', 'eval.py')}
        for argv in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Offline Qualification',
                     '-c', 'user.email=qualification@example.invalid', 'commit', '-qm', 'permitted public input']):
            subprocess.run(['git', '-C', str(repository), *argv], check=True, capture_output=True)
        from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as preflight
        from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
        state = PRIVATE / 'preflight'
        result = preflight.qualify(repository=repository, instruction=Path('/permitted/instruction.md'),
            state=state, task_profile=json.loads(Path('/permitted/profile.json').read_bytes()),
            disable_intent_autoencoder=True)
        assert result['qualified'] and result['provider_calls'] == 0
        private_paths = [str(path) for path in sorted(state.rglob('*.duckdb')) if path.is_file()][:3]
        assert private_paths, 'real private owner databases required'
        write(CONTROL / 'private-paths.json', private_paths)
        workspace = ROOT / 'worktrees/task'
        subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach', '-q', str(workspace)],
                       check=True, capture_output=True)
        prompt = (state / 'worker-prompt.txt').read_bytes()
        originals = {name: (repository / name).read_bytes() for name in expected_hashes}
        expected_views = {name: build_source_view(raw, profile='python-comments@1')
                          for name, raw in originals.items()}
        argv = [str(ROOT / 'bin/owner-worker'), '--provider', 'codex_cli', '--model', 'source-view-ipc-authored',
                '--timeout', '300', '--max-output-tokens', '128', '--purpose', 'coding',
                '--semantic-repository', str(repository), '--coding-reply-mode', 'ordinary-completion@1',
                '--source-view-profile', 'python-comments@1']
        write(PRIVATE / 'owner-launch.json', dict(argv=argv, prompt_sha256=hashlib.sha256(prompt).hexdigest(),
              prompt_bytes=len(prompt), task_cid=result['task_cid']))
        process = subprocess.Popen(argv, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=(PRIVATE / 'owner-entry.stderr').open('w'), text=True)
        process.stdin.write(prompt.decode()); process.stdin.close()
        negative_peers, descriptor, lines = [], None, []
        for line in process.stdout:
            lines.append(line)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or 'fixture_event' not in event:
                continue
            name = event['fixture_event']
            if name == 'worker_ready':
                from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
                descriptor = json.loads(Path(event['descriptor']).read_bytes())
                request = dict(schema=client.REQUEST_SCHEMA, session_id=descriptor['session_id'],
                    request_id=uuid.uuid4().hex, query=dict(path='eigen.py', view='compact', cursor=None,
                        max_bytes=4096, expected_source_sha256=None, expected_view_sha256=None))
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                    channel.settimeout(30); channel.connect(descriptor['socket_path'])
                    client.send_frame(channel, request)
                    refused = client.receive_frame(channel)
                assert refused['reason_code'] == 'peer_refused' and refused['page'] is None
                negative_peers.append(dict(case='wrong_peer_uid1000', reason_code=refused['reason_code']))
                proc = subprocess.run(['/usr/bin/sudo', '-n', '-u', 'benchmarkworker', '--',
                    str(ROOT / 'bin/probe-client'), '--source-view-descriptor', event['descriptor'],
                    '--source-view-descriptor-sha256', event['descriptor_sha256'], '--path', 'eigen.py', '--raw-response'],
                    cwd=workspace, capture_output=True, text=True, timeout=30)
                refused = json.loads(proc.stdout)
                assert proc.returncode == 2 and refused['reason_code'] == 'peer_refused' and refused['page'] is None
                negative_peers.append(dict(case='unrelated_peer_uid1001', reason_code=refused['reason_code']))
            elif name in ('mutate_canonical', 'mutate_allocated', 'restore_canonical', 'restore_allocated'):
                action, scope = name.split('_')
                path = (repository if scope == 'canonical' else workspace) / 'eigen.py'
                path.write_bytes(originals['eigen.py'] + (b'\n# authored stale-source probe\n' if action == 'mutate' else b''))
            else:
                raise AssertionError('unknown fixture event')
            (CONTROL / (name + '.ack')).write_text('ack')
        code = process.wait(timeout=30)
        (PRIVATE / 'owner-entry.stdout').write_text(''.join(lines))
        assert code == 0, 'owner entry failed; private diagnostic retained'
        worker = json.loads((CONTROL / 'worker-result.json').read_bytes())
        router = json.loads((CONTROL / 'router-observation.json').read_bytes())
        assert worker['uid'] == 1001 and len(worker['private_read_denials']) == len(private_paths)
        assert descriptor is not None and router['descriptor_environment_restored']
        per_file, envelope_bytes = {}, []
        for path, original in originals.items():
            by_view = {}
            for view, expected in (('original', original), ('compact', expected_views[path].view)):
                responses = worker['pages'][path][view]
                pages = [response['page'] for response in responses]
                actual = b''.join(data(page) for page in pages)
                assert actual == expected and pages[0]['offset'] == 0 and pages[-1]['exhausted']
                assert all(left['end_offset'] == right['offset'] for left, right in zip(pages, pages[1:]))
                assert pages[-1]['end_offset'] == len(expected)
                assert all(page['source_sha256'] == expected_hashes[path] for page in pages)
                assert all(page['view_sha256'] == hashlib.sha256(expected_views[path].view).hexdigest() for page in pages)
                sizes = [len(wire(response)) for response in responses]
                assert all(size <= worker['selected_page_budget'] for size in sizes)
                envelope_bytes.extend(sizes)
                by_view[view] = dict(bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest(),
                    pages=len(pages), complete_envelope_bytes=sizes,
                    effective_profile=pages[0]['effective_profile'], fallback_reason=pages[0]['fallback_reason'])
            per_file[path] = by_view
        working = worker['working_pages']
        checks = worker['working_checks']
        edited_source = originals['eigen.py'] + b'\n# authored working candidate edit\n'
        edited_view = build_source_view(edited_source, profile='python-comments@1')
        assert checks['actual_worker_source_edit'] and checks['actual_worker_source_restore']
        assert len({checks[key] for key in ('initial_generation', 'edited_generation', 'restored_generation')}) == 3
        expected_working = (
            ('edited_compact', edited_view.view, 'eigen.py', edited_source, checks['edited_generation']),
            ('edited_original', edited_source, 'eigen.py', edited_source, checks['edited_generation']),
            ('restored_compact', expected_views['eigen.py'].view, 'eigen.py', originals['eigen.py'], checks['restored_generation']),
        )
        for key, expected, name, source, generation in expected_working:
            page = working[key]['page']
            assert data(page) == expected and page['offset'] == 0 and page['exhausted']
            assert page['source_sha256'] == hashlib.sha256(source).hexdigest()
            assert page['workspace_generation'] == generation
            assert page['baseline_source_sha256'] == expected_hashes[name]
        initial_page = working['initial_eval']['page']
        assert expected_views['eval.py'].view.startswith(data(initial_page))
        assert initial_page['workspace_generation'] == checks['initial_generation']
        assert initial_page['next_cursor'] is not None and initial_page['exhausted'] is False
        restored_pages = [response['page'] for response in working['restored_eval']]
        assert b''.join(data(page) for page in restored_pages) == expected_views['eval.py'].view
        assert restored_pages[0]['offset'] == 0 and restored_pages[-1]['exhausted']
        assert all(left['end_offset'] == right['offset'] for left, right in zip(restored_pages, restored_pages[1:]))
        all_working = [working[key] for key in ('initial_eval', 'edited_compact', 'edited_original', 'restored_compact')]
        all_working += working['restored_eval']
        for response in all_working:
            page = response['page']
            assert len(wire(response)) <= worker['selected_page_budget']
            assert page['schema'] == 'supervisor-worker-source-view-page@2'
            assert page['source_state'] == 'working-candidate' and page['baseline_evidence_applicable'] is False
            assert page['canonical_source_current'] is True and page['workspace_source_current'] is True
            assert page['source_check_profile'] == 'owner-working-source-candidate@1'
            assert page['baseline_source_sha256'] == expected_hashes[page['path']]
            assert all(page[key] is False for key in ('proof_authority', 'completion_authority',
                'execution_authority', 'mutation_authority', 'omission_authority', 'semantic_equivalence_claimed'))
        working_summary = dict(checks, candidate_primary_page_count=len(all_working),
            candidate_primary_envelope_bytes=sum(len(wire(value)) for value in all_working),
            edited_source_bytes=len(edited_source), edited_source_sha256=hashlib.sha256(edited_source).hexdigest(),
            edited_compact_bytes=len(edited_view.view), edited_compact_sha256=hashlib.sha256(edited_view.view).hexdigest(),
            exact_edited_original_expansion_verified=True, exact_restored_eval_pagination_verified=True,
            baseline_and_candidate_envelopes_distinguished=True, all_candidate_false_authority_flags_verified=True)
        assert not Path(descriptor['socket_path']).exists()
        assert not Path(worker['descriptor_path']).exists()
        assert all(sha(repository / name) == digest == sha(workspace / name) for name, digest in expected_hashes.items())
        from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import collect
        collector_exceptions = []
        def trace_collector(frame, event, arg):
            if (event == 'exception' and issubclass(arg[0], (ValueError, OSError, KeyError, TypeError))
                    and Path(frame.f_code.co_filename).name in {
                        'terminal_source_view_observation.py', 'terminal_numerical_retrieval_observation.py',
                        'terminal_output_observation.py', 'source_formalization_context.py'}):
                collector_exceptions.append(dict(file=Path(frame.f_code.co_filename).name,
                    line=frame.f_lineno, error_type=arg[0].__name__))
            return trace_collector
        sys.settrace(trace_collector)
        try:
            observation = collect(parent=ROOT / 'worker-observations',
                scope=dict(task_cid=result['task_cid'], task_alias=descriptor['task_id']),
                report=dict(worker_cleanup_returncode=0, source_view_profile='python-comments@1',
                    provider_invocations=[dict(router['receipt'], phase='coding')]))
        finally:
            sys.settrace(None)
        write(Path('/results/collector-diagnostic.json'), dict(status=observation['status'],
            reason=observation['reason'], exceptions=collector_exceptions))
        assert observation['status'] == 'observed', observation['reason']
        assert observation['counts']['accepted_requests'] == worker['all_cli_calls'] + len(negative_peers)
        assert observation['counts']['delivered_source_bytes'] == worker['all_cli_returned_source_bytes']
        assert observation['candidate_context_counts']['delivered_bytes'] == worker['candidate_context_returned_bytes'] > 0
        from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_artifacts import build_candidate_artifacts
        # Owner independently rebuilds immutable artifacts from exact retained permitted inputs.
        # This qualification replay is outside the measured source broker session.
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import encode_semantic_router_prompt
        table = encode_semantic_router_prompt(prompt=prompt.decode(), repository=repository).table.to_dict()
        payload = json.loads((repository / table['semantic_artifact']).read_bytes())
        selected = tuple(payload.get('program_paths', sorted(table['source_manifest'])))
        captured = {path: (workspace / path).read_bytes() for path in table['source_manifest']}
        expected = build_candidate_artifacts(sources=captured, program_paths=selected,
            workspace_generation=checks['restored_generation'])
        for artifact, responses in worker['candidate_context_pages'].items():
            page = responses[0]['page']
            assert page['artifact_sha256'] == hashlib.sha256(expected[artifact]).hexdigest()
            assert page['artifact_bytes'] == len(expected[artifact])
            assert expected[artifact].startswith(b''.join(data(row['page']) for row in responses))
            assert page['baseline_evidence_applicable'] is page['proof_authority'] is False
            if artifact == 'summary':
                assert b''.join(data(row['page']) for row in responses) == expected[artifact]
        from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_selection import select_candidate_artifact
        selection_evidence = {}
        selection_bases = {}
        for label, value in worker['selective_results'].items():
            generation = value['generation']
            if generation not in selection_bases:
                selection_bases[generation] = build_candidate_artifacts(sources=captured,
                    program_paths=selected,workspace_generation=generation)
            parents = selection_bases[generation]
            selected_raw = select_candidate_artifact(artifacts=parents,sources=captured,
                artifact=value['artifact'],selection=value['selection'])
            assert b''.join(data(row['page']) for row in value['pages']) == selected_raw
            selected_value = json.loads(selected_raw)
            selection_evidence[label] = dict(artifact=value['artifact'],selection=value['selection'],
                artifact_bytes=len(selected_raw),wire_bytes=value['wire_bytes'],pages=len(value['pages']),
                artifact_sha256=hashlib.sha256(selected_raw).hexdigest(),counts=selected_value['counts'],
                baseline_evidence_applicable=selected_value['baseline_evidence_applicable'],
                exact_owner_replay=True, full_parent_artifact_bytes={key:len(raw) for key,raw in parents.items()})
        write(Path('/results/selection-observation.json'),dict(schema='candidate-selective-worker-observation@1',
            queries=selection_evidence,default_model_projection=worker['selective_projection'], initial_discovery_and_subset_wire_bytes=sum(selection_evidence[key]['wire_bytes']
                for key in ('catalog','capsules','obligations')), complete_subset_rows_verified=True,
            source_obligations_discharged=0, provider_tokens=None,total_token_savings_measured=False,
            scope='Complete exact subsets versus parent artifact sizes; different information coverage, not lossless compression or a matched model benchmark.'))
        write(Path('/results/candidate-artifact-observation.json'), dict(
            artifact_bytes={key: len(value) for key, value in expected.items()},
            artifact_sha256={key: hashlib.sha256(value).hexdigest() for key, value in expected.items()},
            exact_owner_replay=True, candidate_counts=observation['candidate_context_counts'],
            stale_refusals=worker['candidate_context_refusals'], generation_rebuilt=True))
        write(Path('/results/session-observation.json'), observation)
        outcome.update(complete=True, owner_uid=1000, worker_uid=1001, public_source_hashes=expected_hashes,
            per_file=per_file, retrieved_source_bytes=sum(len(v) for v in originals.values()),
            retrieved_compact_bytes=sum(len(v.view) for v in expected_views.values()),
            primary_complete_response_envelope_utf8_bytes=sum(envelope_bytes),
            primary_page_count=len(envelope_bytes), max_primary_envelope_bytes=max(envelope_bytes),
            private_read_denial_count=len(private_paths), negative_checks=negative_peers + worker['negatives'],
            broker_stopped=True, descriptor_and_socket_removed=True, source_restored=True,
            authored_router_calls=len(router['authored_generate_calls']),
            all_cli_calls=worker['all_cli_calls'], all_cli_printed_json_bytes=worker['all_cli_printed_json_bytes'],
            raw_cli_envelope_bytes=worker['raw_cli_envelope_bytes'],
            default_cli_projection_bytes=worker['default_cli_projection_bytes'],
            default_projection_checks=worker['default_projection_checks'],
            working_source_refresh=working_summary, candidate_context_counts=observation['candidate_context_counts'],
            all_cli_returned_source_bytes=worker['all_cli_returned_source_bytes'],
            collection_status=observation['status'], broker_counts=observation['counts'])
        write(Path('/results/router-observation.json'), router)
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error)[:500])
        (PRIVATE / 'probe-failure.txt').write_text(traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        outcome['seconds'] = time.monotonic() - started
        write(Path('/results/probe-qualification.json'), outcome)
        print(json.dumps({key: outcome.get(key) for key in ('complete', 'error_type', 'seconds', 'retrieved_source_bytes')}), flush=True)
    return 0 if outcome['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
