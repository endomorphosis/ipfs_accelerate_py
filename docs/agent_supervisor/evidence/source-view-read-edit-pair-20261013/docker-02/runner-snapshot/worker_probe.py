"""Authored matched source-read/edit trace through the actual isolated router hook.

Default CLI stdout/stderr are retained verbatim, including terminal newlines.
There is no provider request, model decision, token estimate or proof claim.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path('/opt/ipfs-supervisor')
CONTROL = ROOT / 'fixture-ipc'
CLIENT = ROOT / 'source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py'
ENV = 'IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR'


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def content(page):
    return page['content'].encode() if page['encoding'] == 'utf-8' else base64.b64decode(page['content'], validate=True)


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
        profile = json.loads((CONTROL / 'profile.json').read_bytes())['profile']
        assert 'worker_source_client.py' in prompt
        calls.append(dict(prompt_utf8_bytes=len(prompt.encode()), prompt_sha256=sha(prompt.encode()), authored_provider_response=True))
        status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        assert status['NoNewPrivs'].strip() == '1'
        assert all(int(status[name].strip(), 16) == 0 for name in ('CapEff', 'CapPrm', 'CapAmb'))
        denied = []
        for name in json.loads((CONTROL / 'private-paths.json').read_bytes()):
            try:
                with Path(name).open('rb'):
                    pass
            except PermissionError:
                denied.append(name)
            else:
                raise AssertionError('worker could read private owner database')
        records, source_reads = [], []

        def invoke(phase, kind, path=None, view=None, page_index=0, **options):
            client = CLIENT if kind == 'source' else CLIENT.with_name('worker_candidate_context_client.py')
            argv = [sys.executable, '-I', '-B', str(client), '--max-bytes', '8192']
            if kind == 'source':
                argv += ['--path', path, '--view', view, '--working']
            else:
                argv += ['--select', '--artifact', 'catalog']
            for key, value in options.items():
                if value is not None:
                    argv += ['--' + key.replace('_', '-'), json.dumps(value) if key == 'cursor' else str(value)]
            assert '--raw-response' not in argv and len(records) < 50
            proc = subprocess.run(argv, capture_output=True, timeout=45)
            row = dict(index=len(records), operation_id=phase + ':' + (path or kind), page_index=page_index, phase=phase, kind=kind, path=path, view=view, argv=argv,
                returncode=proc.returncode, stdout=proc.stdout.decode('utf-8'), stderr=proc.stderr.decode('utf-8'),
                stdout_utf8_bytes=len(proc.stdout), stderr_utf8_bytes=len(proc.stderr),
                stdout_sha256=sha(proc.stdout), stderr_sha256=sha(proc.stderr))
            records.append(row)
            assert len(proc.stdout) <= 8193 and not proc.stderr, row
            response = json.loads(proc.stdout)
            assert proc.returncode == (0 if response['ok'] else 2)
            return response

        def read(phase, path, view='compact', generation=None, pins=None):
            pages, cursor = [], None
            options = dict(pins or {})
            for page_index in range(16):
                page = invoke(phase, 'source', path, view, page_index=page_index, cursor=cursor,
                    expected_workspace_generation=generation, **options)
                assert page['ok'], page
                pages.append(page)
                generation = page['workspace_generation']
                options = dict(expected_source_sha256=page['source_sha256'], expected_view_sha256=page['view_sha256'])
                cursor = page['next_cursor']
                if cursor is None:
                    break
            else:
                raise AssertionError('source pagination exceeded bound')
            assembled = b''.join(content(page) for page in pages)
            source_reads.append(dict(operation_id=phase + ':' + path, phase=phase, path=path, view=view,
                assembled_sha256=sha(assembled), assembled_bytes=len(assembled), page_count=len(pages),
                **{key: pages[0][key] for key in ('source_sha256', 'view_sha256', 'workspace_generation')}))
            return assembled, pages[0]

        _, bootstrap = read('bootstrap', 'eigen.py')
        initial_generation = bootstrap['workspace_generation']
        cursor, catalog_pages = None, []
        for page_index in range(16):
            page = invoke('discovery', 'catalog', page_index=page_index, cursor=cursor, expected_workspace_generation=initial_generation)
            assert page['ok']
            catalog_pages.append(content(page))
            cursor = page['next_cursor']
            if cursor is None:
                break
        else:
            raise AssertionError('catalog pagination exceeded bound')
        catalog = json.loads(b''.join(catalog_pages))
        assert {'eigen.py', 'eval.py'} <= {row['path'] for row in catalog['rows']}
        baseline = {}
        for path in ('eigen.py', 'eval.py'):
            _, baseline[path] = read('baseline_read', path, generation=initial_generation)
        pins = {name: baseline['eigen.py'][name] for name in ('source_sha256', 'view_sha256')}
        original, _ = read('pre_edit_expansion', 'eigen.py', view='original', generation=initial_generation,
            pins={'expected_' + key: value for key, value in pins.items()})
        target = Path.cwd() / 'eigen.py'
        assert target.read_bytes() == original and original.count(b'idx') == 3
        sibling = (Path.cwd() / 'eval.py').read_bytes()
        candidate = original.replace(b'idx', b'i')
        target.write_bytes(candidate)
        assert target.read_bytes() == candidate and (Path.cwd() / 'eval.py').read_bytes() == sibling
        refusal = invoke('stale_generation', 'source', 'eval.py', 'compact', expected_workspace_generation=initial_generation)
        assert not refusal['ok']
        _, fresh = read('edited_refresh', 'eigen.py')
        edited_generation = fresh['workspace_generation']
        assert edited_generation != initial_generation
        read('edited_read', 'eval.py', generation=edited_generation)
        expanded, _ = read('post_edit_expansion', 'eigen.py', view='original', generation=edited_generation,
            pins={'expected_source_sha256': fresh['source_sha256'], 'expected_view_sha256': fresh['view_sha256']})
        assert expanded == candidate
        edited_hashes = {'eigen.py': sha(target.read_bytes()), 'eval.py': sha(sibling)}
        target.write_bytes(original)
        restored_refusal = invoke('restore_stale_generation', 'source', 'eigen.py', 'compact',
            expected_workspace_generation=edited_generation)
        assert not restored_refusal['ok']
        _, restored = read('restored_refresh', 'eigen.py')
        assert len({initial_generation, edited_generation, restored['workspace_generation']}) == 3
        assert target.read_bytes() == original and (Path.cwd() / 'eval.py').read_bytes() == sibling
        result = dict(schema='authored-source-read-edit-trace@1', profile=profile,
            uid=os.getuid(), gid=os.getgid(), workspace=str(Path.cwd()), descriptor_path=descriptor,
            descriptor_sha256=digest, task_alias=json.loads(Path(descriptor).read_bytes())['task_id'], private_read_denials=denied, calls=records, source_reads=source_reads, page_budget=8192,
            source_edit=dict(operation='replace_all_exact_bytes', path='eigen.py', preimage='idx', replacement='i',
                expected_occurrences=3, before_sha256=sha(original), after_sha256=sha(candidate)),
            prompt=dict(text=prompt, utf8_bytes=len(prompt.encode()), sha256=sha(prompt.encode())),
            source_hashes={'eigen.py': sha(original), 'eval.py': sha(sibling)},
            edited_source_hashes=edited_hashes,
            restored_source_hashes={'eigen.py': sha(target.read_bytes()), 'eval.py': sha((Path.cwd() / 'eval.py').read_bytes())},
            generations=dict(baseline=initial_generation, edited=edited_generation, restored=restored['workspace_generation']),
            no_new_privileges=True, worker_effective_capabilities=0, edits_mapped=False,
            catalog_step_optional=True, input_paths_known_from_manifest_and_fixture=True,
            actual_original_source_edit=True, original_source_restored=True, full_program_behavior_verified=False,
            provider_calls=0, model_calls=0, source_obligations_discharged=0, provider_tokens=None,
            total_token_savings_measured=False,
            limitations='Authored fixed read/edit sequence, not a model decision or official Terminal-Bench run; original inline semantic source remains in router prompt.')
        (CONTROL / 'worker-result.json').write_bytes(wire(result))
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        schema = wire(kwargs['codex_output_schema'])
        set_last_cli_observation('codex_cli', dict(exit_code=0, authored_offline_fixture=True,
            codex_output_schema_sha256=sha(schema), codex_output_schema_bytes=len(schema)))
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

    llm_router.generate_text = generate
    actual_run = router.run

    def capture_run(*args, **kwargs):
        output, receipt = actual_run(*args, **kwargs)
        assert len(calls) == 1 and ENV not in os.environ and ENV + '_SHA256' not in os.environ
        retained = {key: value for key, value in receipt.items() if key in (
            'schema', 'source_view_retrieval', 'semantic_translation', 'completion_authority', 'usage',
            'provider', 'model', 'model_prompt_bytes', 'model_prompt_sha256', 'invocation_id', 'purpose',
            'workspace', 'external_container_boundary', 'status')}
        (CONTROL / 'router-observation.json').write_bytes(wire(dict(schema='authored-source-view-router-probe@1',
            authored_generate_calls=calls, descriptor_environment_restored=True, receipt=retained,
            provider_calls=0, model_calls=0, native_provider_tokens=None)))
        return output, receipt

    router.run = capture_run
