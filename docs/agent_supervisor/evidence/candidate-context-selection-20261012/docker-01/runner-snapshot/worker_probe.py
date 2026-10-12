"""Authored generation hook; real router, worker identity and source-view CLI."""
import hashlib
import base64
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

        def query(path='eigen.py', view='compact', raw_response=True, working=False, **options):
            argv = [sys.executable, '-I', '-B', str(CLIENT), '--path', path,
                    '--view', view, '--max-bytes', str(page_budget)]
            if raw_response:
                argv.append('--raw-response')
            if working:
                argv.append('--working')
            for key, value in options.items():
                if value is not None:
                    argv += ['--' + key.replace('_', '-'), json.dumps(value) if key == 'cursor' else str(value)]
            assert len(all_sizes) < 60, 'fixture must preserve the bounded owner request budget'
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
        # Read-only candidate generations are observed after actual worker edits.
        # Public source read/write here is an authored fixture action, not a model
        # decision or token measurement. The canonical repository stays owner-only.
        initial_working = query('eval.py', working=True)
        assert initial_working['ok'] and initial_working['page']['next_cursor'] is not None
        initial_page = initial_working['page']
        initial_generation = initial_page['workspace_generation']
        initial_cursor = initial_page['next_cursor']
        candidate_path = Path.cwd() / 'eigen.py'
        candidate_original = candidate_path.read_bytes()
        candidate_edited = candidate_original + b'\n# authored working candidate edit\n'
        candidate_path.write_bytes(candidate_edited)
        assert candidate_path.read_bytes() == candidate_edited
        response = query()
        assert not response['ok'] and response['reason_code'] == 'stale_context'
        negatives.append(dict(case='baseline_refuses_worker_edit', reason_code=response['reason_code']))
        response = query('eval.py', working=True, cursor=initial_cursor,
            expected_workspace_generation=initial_generation)
        assert not response['ok']
        negatives.append(dict(case='working_old_generation_after_edit', reason_code=response['reason_code']))
        edited_compact = query(working=True)
        assert edited_compact['ok']
        edited_page = edited_compact['page']
        edited_generation = edited_page['workspace_generation']
        assert edited_generation != initial_generation
        edited_pins = dict(expected_source_sha256=edited_page['source_sha256'],
            expected_view_sha256=edited_page['view_sha256'], expected_workspace_generation=edited_generation)
        edited_original = query(view='original', working=True, **edited_pins)
        assert edited_original['ok'] and edited_original['page']['exhausted']
        assert edited_page['source_sha256'] == hashlib.sha256(candidate_edited).hexdigest()
        projected_candidate = query(working=True, raw_response=False)
        assert projected_candidate['ok'] and projected_candidate['workspace_generation'] == edited_generation
        candidate_labels = {'source_state': 'working-candidate', 'baseline_evidence_applicable': False,
            'baseline_source_sha256': hashlib.sha256(candidate_original).hexdigest()}
        assert all(projected_candidate.get(key) == value for key, value in candidate_labels.items())
        assert all(projected_candidate[key] == edited_page[key] for key in
            ('source_sha256', 'view_sha256', 'content', 'encoding', 'offset', 'end_offset'))
        assert len(wire(projected_candidate)) < len(wire(edited_compact))
        for label, options in (
            ('working_old_cursor_new_generation', dict(path='eval.py', cursor=initial_cursor,
                expected_workspace_generation=edited_generation)),
            ('working_old_generation_after_refresh', dict(expected_workspace_generation=initial_generation)),
            ('working_old_source_expansion_pin', dict(view='original',
                expected_source_sha256=hashlib.sha256(candidate_original).hexdigest(),
                expected_view_sha256=edited_page['view_sha256'], expected_workspace_generation=edited_generation)),
        ):
            response = query(working=True, **options)
            assert not response['ok']
            negatives.append(dict(case=label, reason_code=response['reason_code']))
        candidate_path.write_bytes(candidate_original)
        assert candidate_path.read_bytes() == candidate_original
        # Observe the changed bytes through a pinned read before recapturing A:
        # this must evict B and must not resurrect generation A's issued cursor.
        response = query(working=True, expected_workspace_generation=edited_generation)
        assert not response['ok']
        negatives.append(dict(case='working_generation_stale_after_worker_restore', reason_code=response['reason_code']))
        restored_compact = query(working=True)
        assert restored_compact['ok']
        restored_generation = restored_compact['page']['workspace_generation']
        assert len({initial_generation, edited_generation, restored_generation}) == 3
        assert restored_compact['page']['source_sha256'] == hashlib.sha256(candidate_original).hexdigest()
        response = query('eval.py', working=True, cursor=initial_cursor,
            expected_workspace_generation=restored_generation)
        assert not response['ok']
        negatives.append(dict(case='observed_aba_cursor_not_resurrected', reason_code=response['reason_code']))
        restored_eval, cursor, pins = [], None, {}
        for _ in range(8):
            response = query('eval.py', working=True, cursor=cursor,
                expected_workspace_generation=restored_generation, **pins)
            assert response['ok'] and response['page']['workspace_generation'] == restored_generation
            restored_eval.append(response)
            page = response['page']
            pins = dict(expected_source_sha256=page['source_sha256'], expected_view_sha256=page['view_sha256'])
            cursor = page['next_cursor']
            if cursor is None:
                break
        else:
            raise AssertionError('bounded working-source pagination did not terminate')
        working_pages = dict(initial_eval=initial_working, edited_compact=edited_compact,
            edited_original=edited_original, restored_compact=restored_compact, restored_eval=restored_eval)
        working_checks = dict(actual_worker_source_edit=True, actual_worker_source_restore=True,
            initial_generation=initial_generation, edited_generation=edited_generation,
            restored_generation=restored_generation, observed_aba_generation_distinct=True,
            whole_admitted_scope_change_invalidates_untouched_file_cursor=True,
            default_candidate_labels_preserved=True, default_candidate_json_bytes=len(wire(projected_candidate)),
            edited_compact_envelope_bytes=len(wire(edited_compact)),
            candidate_default_exact_pins_and_slice=True, private_source_body_exports=False,
            baseline_evidence_applicable=False, accepted_source_published=False, proof_refresh_performed=False)
        # Exercise the separate formal/capsule reader through the actual child CLI.
        candidate_pages, candidate_sizes, candidate_refusals = {}, [], []
        candidate_client = CLIENT.with_name('worker_candidate_context_client.py')
        def candidate_query(generation, artifact='summary', cursor=None, selection=None):
            argv = [sys.executable, '-I', '-B', str(candidate_client), '--artifact', artifact,
                '--expected-workspace-generation', generation, '--max-bytes', '8192', '--raw-response']
            if selection is not None:
                argv += ['--select']
                for key, value in selection.items():
                    if value is not None:
                        argv += ['--' + key.replace('_', '-'), value]
            if cursor is not None:
                argv += ['--cursor', json.dumps(cursor)]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            response = json.loads(proc.stdout)
            assert proc.returncode == (0 if response['ok'] else 2), proc.stderr
            assert len(wire(response)) <= 8192
            all_sizes.append(len(wire(response)))
            raw_sizes.append(len(wire(response)))
            all_source_sizes.append(0)
            candidate_sizes.append(response['page']['end_offset'] - response['page']['offset'] if response['ok'] else 0)
            return response
        refused = candidate_query(edited_generation)
        assert not refused['ok']
        candidate_refusals.append(refused['reason_code'])
        for artifact in ('summary', 'capsules', 'formal'):
            response = candidate_query(restored_generation, artifact)
            assert response['ok'], response
            candidate_pages[artifact] = [response]
            if artifact == 'summary':
                while response['page']['next_cursor'] is not None:
                    assert len(candidate_pages[artifact]) < 8
                    response = candidate_query(restored_generation, artifact, response['page']['next_cursor'])
                    assert response['ok'], response
                    candidate_pages[artifact].append(response)
        stale_cursor = candidate_pages['formal'][0]['page']['next_cursor']
        assert stale_cursor is not None
        candidate_path.write_bytes(candidate_original + b'\n# invalidate formal artifacts\n')
        refused = candidate_query(restored_generation, 'formal', stale_cursor)
        assert not refused['ok']
        candidate_refusals.append(refused['reason_code'])
        candidate_path.write_bytes(candidate_original)
        new_generation = query(working=True)['page']['workspace_generation']
        assert new_generation != restored_generation
        rebuilt = candidate_query(new_generation)
        assert rebuilt['ok'] and rebuilt['page']['artifact_sha256'] != candidate_pages['summary'][0]['page']['artifact_sha256']
        # Selected catalog and exact rows, read completely through the real CLI.
        selective_results = {}
        def selected(generation, artifact, **filters):
            selection = dict(path=None, symbol_id=None, function_id=None, kind=None)
            selection.update(filters)
            response = candidate_query(generation, artifact, selection=selection)
            pages = []
            for _ in range(12):
                assert response['ok'], response
                pages.append(response)
                assert response['page']['selection'] == selection
                if response['page']['next_cursor'] is None:
                    break
                response = candidate_query(generation, artifact, cursor=response['page']['next_cursor'], selection=selection)
            else:
                raise AssertionError('selected artifact pagination did not finish')
            raw = b''.join(row['page']['content'].encode() if row['page']['encoding'] == 'utf-8' else base64.b64decode(row['page']['content']) for row in pages)
            assert hashlib.sha256(raw).hexdigest() == pages[0]['page']['artifact_sha256']
            assert len(raw) == pages[0]['page']['artifact_bytes']
            return dict(generation=generation,artifact=artifact,selection=selection,pages=pages,
                payload_bytes=len(raw),wire_bytes=sum(len(wire(row)) for row in pages)), json.loads(raw)
        before_calls = len(all_sizes)
        record, catalog = selected(new_generation, 'catalog', path='eigen.py')
        selective_results['catalog'] = record
        row, = catalog['rows']
        function_id = row['function_coverage'][0]['function_id']
        symbol_id = row['capsules'][-1]['stable_symbol_id']
        # A specific native pending kind is discovered, not inferred to be discharged.
        kind = sorted(row['obligation_kinds'])[0]
        record, obligations = selected(new_generation, 'obligations', path='eigen.py', function_id=function_id,kind=kind)
        selective_results['obligations'] = record
        assert obligations['counts']['matched_count'] > 0
        assert obligations['counts']['filtered_out_count'] > 0 and obligations['counts']['out_of_path_count'] > 0
        assert all(item['kind'] == kind and function_id in item['function_ids'] for item in obligations['rows'])
        record, capsules = selected(new_generation, 'capsules', path='eigen.py',symbol_id=symbol_id)
        selective_results['capsules'] = record
        assert len(capsules['rows']) == 1 and capsules['rows'][0]['stable_symbol_id'] == symbol_id
        assert capsules['counts']['filtered_out_count'] > 0 and capsules['counts']['out_of_path_count'] > 0
        # A source edit invalidates selected context as well as full artifacts.
        candidate_path.write_bytes(candidate_original.replace(b'np.argmax', b'np.argmin'))
        refused = candidate_query(new_generation, 'obligations', selection=selective_results['obligations']['selection'])
        assert not refused['ok'] and refused['reason_code'] == 'stale_context'
        candidate_refusals.append(refused['reason_code'])
        candidate_path.write_bytes(candidate_original)
        selected_refreshed_generation = query(working=True)['page']['workspace_generation']
        assert selected_refreshed_generation != new_generation
        record, refreshed_catalog = selected(selected_refreshed_generation, 'catalog',path='eigen.py')
        selective_results['refreshed_catalog'] = record
        assert record['pages'][0]['page']['artifact_sha256'] != selective_results['catalog']['pages'][0]['page']['artifact_sha256']
        assert len(all_sizes) < 62, 'selection probe must preserve session request budget'
        for scope in ('canonical', 'allocated'):
            signal_owner('mutate_' + scope)
            response = query()
            assert response['ok'] is False and response['page'] is None
            negatives.append(dict(case='stale_' + scope, reason_code=response['reason_code']))
            if scope == 'canonical':
                response = query(working=True)
                assert not response['ok'] and response['reason_code'] == 'stale_context'
                negatives.append(dict(case='working_refuses_canonical_drift', reason_code=response['reason_code']))
            signal_owner('restore_' + scope)
        assert query()['ok']
        result = dict(schema='authored-source-view-worker-probe@1', uid=os.getuid(), gid=os.getgid(),
            groups=os.getgroups(), workspace=str(Path.cwd()), no_new_privileges=True,
            worker_effective_capabilities=0, descriptor_path=descriptor,
            private_read_denials=denied, pages=all_pages, negatives=negatives,
            working_pages=working_pages, working_checks=working_checks,
            candidate_context_pages=candidate_pages, candidate_context_refusals=candidate_refusals,
            candidate_context_returned_bytes=sum(candidate_sizes), candidate_context_rebuilt=rebuilt,
            selective_results=selective_results, selective_function_id=function_id,selective_symbol_id=symbol_id,
            selective_kind=kind,
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
