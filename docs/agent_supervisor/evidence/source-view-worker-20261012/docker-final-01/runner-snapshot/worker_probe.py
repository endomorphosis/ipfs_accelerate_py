"""Authored generation hook; real router, worker identity and source-view CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/opt/ipfs-supervisor')
CONTROL = ROOT / 'fixture-ipc'
CLIENT = ROOT / 'source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py'
ENV = 'IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR'


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def signal_owner(event, **details):
    print(json.dumps(dict(fixture_event=event, **details), sort_keys=True), flush=True)
    target = CONTROL / (event + '.ack')
    deadline = time.monotonic() + 30
    while not target.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('owner acknowledgement unavailable: ' + event)
        time.sleep(.05)


def install():
    assert os.getuid() == os.geteuid() == 1001
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    intelligence_index.discover_available_providers = lambda: ['codex_cli']
    intelligence_index.select_efficient_route = lambda **kwargs: SimpleNamespace(
        provider='codex_cli', model_name='source-view-ipc-authored', catalog_revision='authored-fixture')
    llm_router.get_llm_provider = lambda *args, **kwargs: object()
    calls = []

    def generate(prompt, **kwargs):
        assert not calls
        assert kwargs['allow_local_fallback'] is False and kwargs['allow_cross_provider_fallback'] is False
        descriptor, digest = os.environ[ENV], os.environ[ENV + '_SHA256']
        assert 'worker_source_client.py' in prompt
        calls.append(dict(prompt_utf8_bytes=len(prompt.encode()),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            descriptor_sha256=digest, authored_provider_response=True))
        status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        assert status['NoNewPrivs'].strip() == '1'
        assert all(int(status[name].strip(), 16) == 0 for name in ('CapEff', 'CapPrm', 'CapAmb'))
        denied = []
        for name in json.loads((CONTROL / 'private-paths.json').read_text()):
            try:
                with Path(name).open('rb'):
                    pass
            except PermissionError:
                denied.append(name)
            else:
                raise AssertionError('worker could read private owner database')
        page_budget, all_sizes, all_source_sizes = 4096, [], []
        raw_sizes, projection_sizes = [], []

        def query(path='eigen.py', view='compact', raw_response=True, **options):
            argv = [sys.executable, '-I', '-B', str(CLIENT), '--path', path,
                    '--view', view, '--max-bytes', str(page_budget)]
            if raw_response:
                argv.append('--raw-response')
            for key, value in options.items():
                if value is not None:
                    argv += ['--' + key.replace('_', '-'), json.dumps(value) if key == 'cursor' else str(value)]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            assert len(proc.stdout.encode()) <= page_budget + 1
            response = json.loads(proc.stdout)
            assert proc.returncode == (0 if response['ok'] else 2), proc.stderr
            assert len(wire(response)) <= page_budget
            all_sizes.append(len(wire(response)))
            (raw_sizes if raw_response else projection_sizes).append(len(wire(response)))
            page = response['page'] if raw_response else response
            all_source_sizes.append(page['end_offset'] - page['offset'] if response['ok'] else 0)
            return response

        signal_owner('worker_ready', descriptor=descriptor, descriptor_sha256=digest)
        all_pages, saved_cursor = {}, None
        for path in ('eigen.py', 'eval.py'):
            representations, pins = {}, {}
            for view in ('compact', 'original'):
                pages, cursor = [], None
                for _ in range(64):
                    response = query(path, view, cursor=cursor, **pins)
                    assert response['ok'], response['reason_code']
                    pages.append(response)
                    page = response['page']
                    pins = dict(expected_source_sha256=page['source_sha256'],
                                expected_view_sha256=page['view_sha256'])
                    cursor = page['next_cursor']
                    if cursor is not None and saved_cursor is None:
                        saved_cursor = (path, view, cursor, pins.copy())
                    if cursor is None:
                        break
                else:
                    raise AssertionError('bounded source pagination did not terminate')
                representations[view] = pages
            all_pages[path] = representations
        projections = []
        for path in ('eigen.py', 'eval.py'):
            raw_first = all_pages[path]['compact'][0]
            projected = query(path, raw_response=False)
            expected_fields = {'ok', 'path', 'view', 'source_sha256', 'view_sha256', 'offset', 'end_offset',
                               'exhausted', 'next_cursor', 'encoding', 'content'}
            assert set(projected) == expected_fields and projected['ok'] is True
            for key in expected_fields - {'ok', 'next_cursor'}:
                assert projected[key] == raw_first['page'][key]
            if projected['next_cursor'] is not None:
                assert {k: v for k, v in projected['next_cursor'].items() if k != 'token'} == {
                    k: v for k, v in raw_first['page']['next_cursor'].items() if k != 'token'}
            else:
                assert raw_first['page']['next_cursor'] is None
            assert len(wire(projected)) < len(wire(raw_first))
            projections.append(dict(path=path, original_envelope_bytes=len(wire(raw_first)),
                default_model_json_bytes=len(wire(projected)), exact_source_pins_and_slice=True,
                request_cursor_binding_preserved=True))
        default_refusal = query('not-selected.py', raw_response=False)
        assert set(default_refusal) == {'ok', 'reason_code'} and default_refusal['ok'] is False
        negatives = []
        for label, options in (
            ('outside_source_scope', {'path': 'not-selected.py'}),
            ('wrong_expansion_source_pin', {'view': 'original',
                'expected_source_sha256': '0' * 64,
                'expected_view_sha256': all_pages['eigen.py']['compact'][0]['page']['view_sha256']}),
        ):
            response = query(**options)
            assert response['ok'] is False and response['page'] is None
            negatives.append(dict(case=label, reason_code=response['reason_code']))
        assert saved_cursor is not None, 'public-source fixture must exercise a real continuation'
        path, view, cursor, pins = saved_cursor
        for label in ('forged_cursor', 'consumed_cursor', 'crossed_path_cursor', 'crossed_view_cursor'):
            # A fresh unconsumed cursor isolates each query-binding refusal;
            # only the replay case intentionally reuses a consumed cursor.
            fresh = query(path, view)['page']['next_cursor']
            assert fresh is not None
            options = dict(path=path, view=view, cursor=fresh, **pins)
            if label == 'forged_cursor':
                options['cursor'] = json.loads(json.dumps(fresh))
                options['cursor']['request_sha256'] = '0' * 64
            elif label == 'consumed_cursor':
                options['cursor'] = cursor
            elif label == 'crossed_path_cursor':
                options = dict(path='eval.py' if path == 'eigen.py' else 'eigen.py', view=view, cursor=fresh)
            else:
                options['view'] = 'original' if view == 'compact' else 'compact'
            response = query(**options)
            assert response['ok'] is False and response['page'] is None
            negatives.append(dict(case=label, reason_code=response['reason_code']))
        for scope in ('canonical', 'allocated'):
            signal_owner('mutate_' + scope)
            response = query()
            assert response['ok'] is False and response['page'] is None
            negatives.append(dict(case='stale_' + scope, reason_code=response['reason_code']))
            signal_owner('restore_' + scope)
        assert query()['ok']
        result = dict(schema='authored-source-view-worker-probe@1', uid=os.getuid(), gid=os.getgid(),
            groups=os.getgroups(), workspace=str(Path.cwd()), no_new_privileges=True,
            worker_effective_capabilities=0, descriptor_path=descriptor,
            private_read_denials=denied, pages=all_pages, negatives=negatives,
            provider_calls=0, model_calls=0, native_provider_tokens=None,
            paid_token_savings_measured=False, source_obligations_discharged=0,
            selected_page_budget=page_budget, all_cli_calls=len(all_sizes), all_cli_printed_json_bytes=sum(all_sizes),
            raw_cli_envelope_bytes=sum(raw_sizes), default_cli_projection_bytes=sum(projection_sizes),
            default_projection_checks=projections,
            all_cli_returned_source_bytes=sum(all_source_sizes))
        (CONTROL / 'worker-result.json').write_bytes(wire(result))
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        schema = wire(kwargs['codex_output_schema'])
        set_last_cli_observation('codex_cli', dict(exit_code=0, authored_offline_fixture=True,
            codex_output_schema_sha256=hashlib.sha256(schema).hexdigest(), codex_output_schema_bytes=len(schema)))
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

    llm_router.generate_text = generate
    actual_run = router.run

    def capture_run(*args, **kwargs):
        output, receipt = actual_run(*args, **kwargs)
        assert len(calls) == 1 and ENV not in os.environ and ENV + '_SHA256' not in os.environ
        retained = {key: value for key, value in receipt.items() if key in (
            'schema', 'source_view_retrieval', 'semantic_translation',
            'completion_authority', 'usage', 'provider', 'model', 'model_prompt_bytes',
            'model_prompt_sha256', 'invocation_id', 'purpose', 'workspace', 'external_container_boundary', 'status')}
        (CONTROL / 'router-observation.json').write_bytes(wire(dict(schema='authored-source-view-router-probe@1',
            authored_generate_calls=calls, descriptor_environment_restored=True, receipt=retained,
            provider_calls=0, model_calls=0, native_provider_tokens=None)))
        return output, receipt

    router.run = capture_run
