"""UID1000 fixture: real preparation, OWNER_ENTRY, router and retrieval IPC."""
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
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_bytes(wire(value))


def main():
    assert os.getuid() == os.geteuid() == os.getgid() == 1000
    outcome = dict(schema='numerical-worker-docker-probe@1', complete=False,
        provider_calls=0, model_calls=0, solver_calls=0, native_provider_tokens=None,
        paid_token_savings_measured=False, source_obligations_discharged=0,
        source_semantics_verified=False, official_verifier_executed=False,
        full_live_supervisor_deployment_qualified=False, authored_planning=True,
        intent_autoencoder_disabled=True, learned_source384_inference=False)
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
                assert sha(target) == sha(source), 'cached native extension differs'
            else:
                shutil.copyfile(source, target)
            con.execute('LOAD ' + name)
        con.close()
        repository = ROOT / 'repository'
        assert repository.is_dir() and not any(repository.iterdir())
        expected_hashes = {}
        for name in ('eigen.py', 'eval.py'):
            shutil.copyfile(Path('/permitted') / name, repository / name)
            expected_hashes[name] = sha(repository / name)
        for argv in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Offline Qualification',
                     '-c', 'user.email=qualification@example.invalid', 'commit', '-qm', 'permitted public input']):
            subprocess.run(['git', '-C', str(repository), *argv], check=True, capture_output=True)
        from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as preflight
        from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
        state = PRIVATE / 'preflight'
        result = preflight.qualify(repository=repository, instruction=Path('/permitted/instruction.md'),
            state=state, task_profile=json.loads(Path('/permitted/profile.json').read_bytes()),
            numerical_formalization_profile='python-numerical-control-structure@3', disable_intent_autoencoder=True)
        assert result['qualified'] and result['provider_calls'] == 0
        formal = result['context']['numerical_formalization']
        assert formal['profile'] == 'python-numerical-control-structure@3'
        assert formal['summary']['pending_obligation_count'] == 105
        assert formal['summary']['structurally_complete_functions'] == 4
        report_dir = Path(formal['output'])
        report_index = json.loads((report_dir / 'reports.json').read_bytes())
        reports = {path: json.loads((report_dir / row['artifact']).read_bytes()) for path, row in report_index.items()}
        expected_rows = {path: report['pending_obligations'] for path, report in reports.items()}
        expected_count = sum(map(len, expected_rows.values()))
        assert expected_count == formal['summary']['pending_obligation_count']
        paths = [report_dir / 'reports.json', report_dir / 'receipt.json',
                 report_dir / 'report-0000.json', report_dir / 'metadata-lake/metadata.ducklake',
                 report_dir / 'world-lake/metadata.ducklake']
        assert all(path.is_file() for path in paths)
        write(CONTROL / 'private-paths.json', [str(path) for path in paths])
        workspace = ROOT / 'worktrees/task'
        subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach', '-q', str(workspace)],
                       check=True, capture_output=True)
        prompt = (state / 'worker-prompt.txt').read_bytes()
        originals = {'canonical': (repository / 'eigen.py').read_bytes(),
                     'allocated': (workspace / 'eigen.py').read_bytes()}
        argv = [str(ROOT / 'bin/owner-worker'), '--provider', 'codex_cli', '--model', 'numerical-ipc-authored',
                '--timeout', '300', '--max-output-tokens', '128', '--purpose', 'coding',
                '--semantic-repository', str(repository), '--coding-reply-mode', 'ordinary-completion@1']
        write(PRIVATE / 'owner-launch.json', dict(argv=argv, prompt_sha256=hashlib.sha256(prompt).hexdigest(),
              prompt_bytes=len(prompt), context_receipt_sha256=formal['receipt_sha256']))
        process = subprocess.Popen(argv, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=(PRIVATE / 'owner-entry.stderr').open('w'), text=True)
        process.stdin.write(prompt.decode()); process.stdin.close()
        negative_peers = []
        descriptor = None
        lines = []
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
                from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as client
                descriptor = json.loads(Path(event['descriptor']).read_bytes())
                request = dict(schema=client.REQUEST_SCHEMA, session_id=descriptor['session_id'],
                    request_id=uuid.uuid4().hex,
                    query=dict(path='eigen.py', function_id=None, kind=None, cursor=None, limit=4, max_bytes=8192))
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                    channel.settimeout(30); channel.connect(descriptor['socket_path'])
                    client.send_frame(channel, request)
                    refused = client.receive_frame(channel)
                assert refused['reason_code'] == 'peer_refused' and refused['page'] is None
                negative_peers.append(dict(case='wrong_peer_uid1000', reason_code=refused['reason_code']))
                proc = subprocess.run(['/usr/bin/sudo', '-n', '-u', 'benchmarkworker', '--',
                    str(ROOT / 'bin/probe-client'), '--numerical-obligation-descriptor', event['descriptor'],
                    '--numerical-obligation-descriptor-sha256', event['descriptor_sha256'], '--path', 'eigen.py'],
                    cwd=workspace, capture_output=True, text=True, timeout=30)
                refused = json.loads(proc.stdout)
                assert proc.returncode == 2 and refused['reason_code'] == 'peer_refused' and refused['page'] is None
                negative_peers.append(dict(case='unrelated_peer_uid1001', reason_code=refused['reason_code']))
            elif name in ('mutate_canonical', 'mutate_allocated', 'restore_canonical', 'restore_allocated'):
                action, scope = name.split('_')
                path = (repository if scope == 'canonical' else workspace) / 'eigen.py'
                path.write_bytes(originals[scope] + (b'\n# authored stale-source probe\n' if action == 'mutate' else b''))
            else:
                raise AssertionError('unknown fixture event')
            (CONTROL / (name + '.ack')).write_text('ack')
        code = process.wait(timeout=30)
        (PRIVATE / 'owner-entry.stdout').write_text(''.join(lines))
        assert code == 0, 'owner entry failed; private diagnostic retained'
        worker = json.loads((CONTROL / 'worker-result.json').read_bytes())
        router = json.loads((CONTROL / 'router-observation.json').read_bytes())
        assert worker['uid'] == 1001 and len(worker['private_read_denials']) == len(paths)
        assert descriptor is not None
        assert router['descriptor_environment_restored'] is True
        per_file, envelope_bytes = {}, []
        for path, expected in expected_rows.items():
            responses = worker['pages'][path]
            actual = [row for response in responses for row in response['page']['rows']]
            assert actual == expected
            assert len({row['obligation_id'] for row in actual}) == len(expected)
            sizes = [len(wire(response)) for response in responses]
            assert all(size <= 8192 for size in sizes)
            envelope_bytes += sizes
            per_file[path] = dict(expected_rows=len(expected), retrieved_rows=len(actual),
                page_count=len(responses), complete_envelope_bytes=sizes,
                rows_sha256=hashlib.sha256(wire(actual)).hexdigest())
        filter_checks = []
        for selection in worker['filters']:
            path, function, kind = selection['path'], selection['function_id'], selection['kind']
            expected = [row for row in expected_rows[path]
                        if (function is None or function in row['function_ids'])
                        and (kind is None or row['kind'] == kind)]
            actual = [row for response in selection['responses'] for row in response['page']['rows']]
            assert actual == expected
            filter_checks.append(dict(path=path, function_id=function, kind=kind,
                                      matched_rows=len(actual), page_count=len(selection['responses'])))
        audits = list((ROOT / 'worker-observations').glob('numerical-*-audit.json'))
        assert len(audits) == 1
        audit = json.loads(audits[0].read_bytes())
        assert audit['service_stopped'] is True and audit['retrieval_provider_calls'] == 0
        assert not Path(descriptor['socket_path']).exists()
        assert not list((ROOT / 'worker-observations').glob('numerical-*.sock'))
        assert not (ROOT / 'worker-observations' / ('numerical-' + descriptor['session_id'] + '.json')).exists()
        assert all(sha(repository / name) == digest == sha(workspace / name) for name, digest in expected_hashes.items())
        from benchmarks.agent_supervisor.container_coding.terminal_numerical_retrieval_observation import collect
        observation = collect(parent=ROOT/'worker-observations',
            scope=dict(task_cid=descriptor['task_cid'], task_alias=descriptor['task_id']),
            report=dict(worker_cleanup_returncode=0,
                        provider_invocations=[dict(router['receipt'], phase='coding')]))
        assert observation['status'] == 'observed', observation['reason']
        assert observation['counts']['accepted_requests'] == audit['accepted_requests']
        assert observation['counts']['response_bytes'] == audit['response_bytes']
        write(Path('/results/session-observation.json'), observation)
        write(Path('/results/broker-audit.json'), audit)
        write(Path('/results/router-observation.json'), router)
        outcome.update(complete=True, owner_uid=1000, worker_uid=1001, public_source_hashes=expected_hashes,
            pending_obligations=expected_count, exact_retrieved_obligations=expected_count, per_file=per_file,
            complete_response_envelope_utf8_bytes=sum(envelope_bytes), max_envelope_bytes=max(envelope_bytes),
            page_count=len(envelope_bytes), private_read_denial_count=len(paths),
            filter_checks=filter_checks, collection_status=observation['status'],
            negative_checks=negative_peers + worker['negatives'], broker_stopped=True,
            descriptor_and_socket_removed=True, source_restored=True,
            native_ducklake_hydration=formal['hydration'], authored_router_calls=len(router['authored_generate_calls']),
            broker_requests=audit['accepted_requests'], broker_response_bytes=audit['response_bytes'])
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error)[:500])
        (PRIVATE / 'probe-failure.txt').write_text(traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        outcome['seconds'] = time.monotonic() - started
        write(Path('/results/probe-qualification.json'), outcome)
        print(json.dumps({key: outcome.get(key) for key in ('complete', 'error_type', 'seconds', 'exact_retrieved_obligations')}), flush=True)
    return 0 if outcome['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
