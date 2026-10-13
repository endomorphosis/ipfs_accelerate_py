"""Authored provider response; real router, UID boundary, CLI and broker.

This file is an offline qualification harness, never model/provider evidence.
"""
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
CLIENT = ROOT / 'source/ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py'


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def signal_owner(event, **details):
    print(json.dumps(dict(fixture_event=event, **details), sort_keys=True), flush=True)
    target = CONTROL / (event + '.ack')
    deadline = time.monotonic() + 30
    while not target.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('owner fixture acknowledgement unavailable: ' + event)
        time.sleep(.05)


def install():
    assert os.getuid() == os.geteuid() == 1001
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    intelligence_index.discover_available_providers = lambda: ['codex_cli']
    intelligence_index.select_efficient_route = lambda **kwargs: SimpleNamespace(
        provider='codex_cli', model_name='numerical-ipc-authored', catalog_revision='authored-fixture')
    llm_router.get_llm_provider = lambda *args, **kwargs: object()
    calls = []

    def generate(prompt, **kwargs):
        assert not calls
        assert kwargs['allow_local_fallback'] is False
        assert kwargs['allow_cross_provider_fallback'] is False
        descriptor = os.environ['IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR']
        digest = os.environ['IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR_SHA256']
        assert 'numerical_obligation_client.py' in prompt
        calls.append({'prompt_utf8_bytes': len(prompt.encode()),
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'descriptor_sha256': digest, 'authored_provider_response': True})
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
                raise AssertionError('worker could read a private owner artifact')

        query_observations = []
        def query(path='eigen.py', **options):
            argv = [sys.executable, '-I', '-B', str(CLIENT), '--path', path,
                    '--limit', '4', '--max-bytes', '8192']
            for key, value in options.items():
                if value is not None:
                    argv += ['--' + key.replace('_', '-'), json.dumps(value) if key == 'cursor' else str(value)]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            assert len(proc.stdout.encode()) <= 8193
            query_observations.append(dict(path=path, filtered=bool(options.get('function_id') or options.get('kind')), returncode=proc.returncode, response_bytes=len(proc.stdout.encode()), stderr=proc.stderr[-1000:]))
            (CONTROL / 'worker-result.json').write_bytes(wire(dict(schema='authored-query-progress@1', complete=False, queries=query_observations)))
            if not proc.stdout:
                raise RuntimeError('authored client transport failed; private query diagnostics retained')
            response = json.loads(proc.stdout)
            assert proc.returncode == (0 if response['ok'] else 2)
            assert len(wire(response)) <= 8192
            return response

        signal_owner('worker_ready', descriptor=descriptor, descriptor_sha256=digest)
        all_pages = {}
        for path in ('eigen.py', 'eval.py'):
            pages = []
            cursor = None
            for _ in range(257):
                response = query(path, cursor=cursor)
                assert response['ok'], response['reason_code']
                pages.append(response)
                cursor = response['page']['next_cursor']
                if cursor is None:
                    break
            else:
                raise AssertionError('pagination did not finish')
            all_pages[path] = pages
        first = all_pages['eigen.py'][0]['page']
        cursor = first['next_cursor']
        assert cursor is not None
        forged = json.loads(json.dumps(cursor))
        forged['context_receipt_sha256'] = '0' * 64
        negatives = []
        for label, options in (
            ('forged_cursor', {'cursor': forged}),
            ('crossed_path_cursor', {'path': 'eval.py', 'cursor': cursor}),
            ('crossed_filter_cursor', {'cursor': cursor, 'kind': 'matrix_shape'}),
        ):
            response = query(**options)
            assert response['ok'] is False
            negatives.append({'case': label, 'reason_code': response['reason_code']})
        filters = []
        for scope in ('canonical', 'allocated'):
            signal_owner('mutate_' + scope)
            response = query()
            assert response['ok'] is False and response['reason_code'] == 'stale_context'
            negatives.append({'case': 'stale_' + scope, 'reason_code': response['reason_code']})
            signal_owner('restore_' + scope)
        assert query()['ok']
        result = dict(schema='authored-numerical-worker-probe@1', uid=os.getuid(), gid=os.getgid(),
                      groups=os.getgroups(), workspace=str(Path.cwd()), no_new_privileges=True,
                      worker_effective_capabilities=0, private_read_denials=denied, pages=all_pages,
                      filters=filters, negatives=negatives, provider_calls=0, model_calls=0,
                      native_provider_tokens=None, paid_token_savings_measured=False,
                      source_obligations_discharged=0)
        (CONTROL / 'worker-result.json').write_bytes(wire(result))
        # Only fixture adapter metadata; no fabricated token/cost observations.
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        schema = wire(kwargs['codex_output_schema'])
        set_last_cli_observation('codex_cli', dict(exit_code=0, authored_offline_fixture=True,
            codex_output_schema_sha256=hashlib.sha256(schema).hexdigest(),
            codex_output_schema_bytes=len(schema)))
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

    llm_router.generate_text = generate
    actual_run = router.run

    def capture_run(*args, **kwargs):
        output, receipt = actual_run(*args, **kwargs)
        assert len(calls) == 1
        assert 'IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR' not in os.environ
        assert 'IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR_SHA256' not in os.environ
        retained = {key: value for key, value in receipt.items()
                    if key in ('schema', 'source_formalization_context', 'numerical_obligation_retrieval',
                               'numerical_obligation_session', 'numerical_obligation_broker',
                               'completion_authority', 'usage', 'provider', 'model', 'model_prompt_bytes',
                               'model_prompt_sha256', 'invocation_id', 'purpose', 'workspace',
                               'external_container_boundary', 'status')}
        (CONTROL / 'router-observation.json').write_bytes(wire(dict(
            schema='authored-numerical-router-probe@1', authored_generate_calls=calls,
            descriptor_environment_restored=True, receipt=retained,
            provider_calls=0, model_calls=0, native_provider_tokens=None)))
        return output, receipt

    router.run = capture_run
