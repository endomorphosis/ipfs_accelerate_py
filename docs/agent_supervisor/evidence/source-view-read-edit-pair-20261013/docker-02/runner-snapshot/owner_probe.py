"""Native owner qualification of the matched isolated source trace.

Uses the existing preflight, generated launcher, broker and output collector.
Replays public bytes independently; never treats authored edit as a proof.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/opt/ipfs-supervisor')
PRIVATE = ROOT / 'state'
CONTROL = ROOT / 'fixture-ipc'
sys.path[:0] = [str(ROOT / name) for name in ('source', 'datasets', 'kit')]


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    Path(path).write_bytes(wire(value))


def data(page):
    return page['content'].encode() if page['encoding'] == 'utf-8' else base64.b64decode(page['content'], validate=True)


def qualify_arm(profile, *, result, prompt, originals, expected_hashes, state, workspace, repository, private_paths):
    from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
    label = 'exact' if profile == 'identity@1' else 'compact'
    output = Path('/results') / label
    output.mkdir()
    owner_state = PRIVATE / label
    owner_state.mkdir(mode=0o700)
    write(CONTROL / 'profile.json', dict(profile=profile))
    outcome = dict(schema='source-view-read-edit-owner-qualification@1', complete=False, profile=profile,
        provider_calls=0, model_calls=0, provider_tokens=None, total_token_savings_measured=False,
        source_obligations_discharged=0, edits_mapped=False)
    argv = [str(ROOT / 'bin/owner-worker'), '--provider', 'codex_cli', '--model', 'source-view-ipc-authored',
        '--timeout', '300', '--max-output-tokens', '128', '--purpose', 'coding', '--semantic-repository', str(repository),
        '--coding-reply-mode', 'ordinary-completion@1', '--source-view-profile', profile]
    write(owner_state / 'owner-launch.json', dict(argv=argv, prompt_sha256=sha(prompt), prompt_bytes=len(prompt), task_cid=result['task_cid']))
    completed = subprocess.run(argv, cwd=workspace, input=prompt, capture_output=True, timeout=360)
    (owner_state / 'owner-entry.stdout').write_bytes(completed.stdout)
    (owner_state / 'owner-entry.stderr').write_bytes(completed.stderr)
    assert completed.returncode == 0, 'owner entry failed; private diagnostic retained'
    worker = json.loads((CONTROL / 'worker-result.json').read_bytes())
    router = json.loads((CONTROL / 'router-observation.json').read_bytes())
    assert worker['profile'] == profile and worker['uid'] == 1001
    assert len(worker['private_read_denials']) == len(private_paths)
    assert router['descriptor_environment_restored'] and not Path(worker['descriptor_path']).exists()
    assert all((repository / name).read_bytes() == raw == (workspace / name).read_bytes() for name, raw in originals.items())
    edited = {**originals, 'eigen.py': originals['eigen.py'].replace(b'idx', b'i')}
    assert worker['source_hashes'] == worker['restored_source_hashes'] == expected_hashes
    assert worker['edited_source_hashes'] == {name: sha(raw) for name, raw in edited.items()}
    expected_edit = dict(operation='replace_all_exact_bytes', path='eigen.py', preimage='idx', replacement='i',
        expected_occurrences=3, before_sha256=sha(originals['eigen.py']), after_sha256=sha(edited['eigen.py']))
    assert worker['source_edit'] == expected_edit
    grouped, source_bytes, catalog_bytes = {}, 0, 0
    for index, row in enumerate(worker['calls']):
        assert row['index'] == index and '--raw-response' not in row['argv']
        for stream in ('stdout', 'stderr'):
            raw = row[stream].encode()
            assert len(raw) == row[stream + '_utf8_bytes'] and sha(raw) == row[stream + '_sha256']
        page = json.loads(row['stdout'])
        assert row['returncode'] == (0 if page['ok'] else 2)
        assert not row['stderr']
        grouped.setdefault(row['operation_id'], []).append((row, page))
        if page['ok']:
            if row['kind'] == 'source': source_bytes += len(data(page))
            else: catalog_bytes += len(data(page))
    replay = []
    for read in worker['source_reads']:
        rows = grouped[read['operation_id']]
        assert [row['page_index'] for row, page in rows] == list(range(len(rows)))
        pages = [page for row, page in rows]
        assert all(page['ok'] for page in pages)
        assert pages[0]['offset'] == 0 and pages[-1]['exhausted']
        assert all(left['end_offset'] == right['offset'] for left, right in zip(pages, pages[1:]))
        raw = b''.join(data(page) for page in pages)
        population = edited if read['phase'] in {'edited_refresh', 'edited_read', 'post_edit_expansion'} else originals
        source = population[read['path']]
        view = build_source_view(source, profile=profile).view
        expected = source if read['view'] == 'original' else view
        assert raw == expected
        assert len(raw) == read['assembled_bytes'] and sha(raw) == read['assembled_sha256']
        assert len(rows) == read['page_count']
        assert all(page['source_sha256'] == read['source_sha256'] == sha(source)
            and page['view_sha256'] == read['view_sha256'] == sha(view)
            and page['workspace_generation'] == read['workspace_generation'] for page in pages)
        replay.append(dict(**read, complete=True, exact_owner_replay=True))
    # Independently rebuild the complete optional catalog from native owner inputs.
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import encode_semantic_router_prompt
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_artifacts import build_candidate_artifacts
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_selection import select_candidate_artifact
    table = encode_semantic_router_prompt(prompt=prompt.decode(), repository=repository).table.to_dict()
    semantic = json.loads((repository / table['semantic_artifact']).read_bytes())
    program_paths = tuple(semantic.get('program_paths', sorted(table['source_manifest'])))
    captured = {path: (workspace / path).read_bytes() for path in table['source_manifest']}
    assert all(sha(raw) == table['source_manifest'][path]['sha256'] for path, raw in captured.items())
    artifacts = build_candidate_artifacts(sources=captured, program_paths=program_paths,
        workspace_generation=worker['generations']['baseline'])
    expected_catalog = select_candidate_artifact(artifacts=artifacts, sources=captured, artifact='catalog',
        selection=dict(path=None, symbol_id=None, function_id=None, kind=None))
    catalog_rows = grouped['discovery:catalog']
    assert [row['page_index'] for row, page in catalog_rows] == list(range(len(catalog_rows)))
    catalog_pages = [page for row, page in catalog_rows]
    assert all(page['ok'] for page in catalog_pages)
    assert catalog_pages[0]['offset'] == 0 and catalog_pages[-1]['exhausted']
    assert all(left['end_offset'] == right['offset'] for left, right in zip(catalog_pages, catalog_pages[1:]))
    actual_catalog = b''.join(data(page) for page in catalog_pages)
    assert actual_catalog == expected_catalog
    catalog_replay = dict(operation_id='discovery:catalog', assembled_sha256=sha(expected_catalog),
        assembled_bytes=len(expected_catalog), complete=True, exact_owner_replay=True)
    # Native launcher cleanup must complete before another arm is admitted.
    remaining_worker_pids = []
    for status_file in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in status_file.read_text().splitlines() if ':' in line)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields['Uid'].split()[0]) == 1001:
            remaining_worker_pids.append(int(status_file.parent.name))
    assert not remaining_worker_pids
    assert not subprocess.check_output(['git', '-C', str(workspace), 'status', '--porcelain', '--untracked-files=all'])
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import collect
    observation = collect(parent=ROOT / 'worker-observations', scope=dict(task_cid=result['task_cid'], task_alias=worker['task_alias']),
        report=dict(worker_cleanup_returncode=0, source_view_profile=profile,
            provider_invocations=[dict(router['receipt'], phase='coding')]))
    assert observation['status'] == 'observed', observation['reason']
    assert observation['counts']['accepted_requests'] == len(worker['calls'])
    assert observation['counts']['delivered_source_bytes'] == source_bytes
    assert observation['candidate_context_counts']['delivered_bytes'] == catalog_bytes > 0
    write(output / 'session-observation.json', observation)
    write(output / 'worker-result.json', worker)
    write(output / 'router-observation.json', router)
    outcome.update(complete=True, owner_uid=1000, worker_uid=1001, public_source_hashes=expected_hashes,
        edited_source_hashes=worker['edited_source_hashes'], restored_source_hashes=worker['restored_source_hashes'],
        source_edit=expected_edit, source_read_replays=replay, catalog_replay=catalog_replay,
        admitted_source_hashes={path: sha(raw) for path, raw in captured.items()},
        catalog_step_optional=True, paths_known_from_inline_manifest_and_authored_fixture=True,
        remaining_worker_pids=remaining_worker_pids, workspace_git_clean=True, private_read_denial_count=len(private_paths),
        broker_stopped=True, descriptor_removed=True, source_restored=True,
        owner_input_prompt_sha256=sha(prompt), owner_input_prompt_bytes=len(prompt),
        cli_calls=len(worker['calls']), actual_cli_stdout_utf8_bytes=sum(row['stdout_utf8_bytes'] for row in worker['calls']),
        actual_cli_stderr_utf8_bytes=sum(row['stderr_utf8_bytes'] for row in worker['calls']),
        actual_default_cli_newlines_included=True, native_session_accounting_verified=True)
    write(output / 'owner-qualification.json', outcome)
    return outcome


def main():
    assert os.getuid() == os.geteuid() == os.getgid() == 1000
    outcome = dict(schema='source-view-read-edit-owner-qualification@1', complete=False,
        provider_calls=0, model_calls=0, provider_tokens=None, total_token_savings_measured=False,
        source_obligations_discharged=0, authored_planning=True, intent_autoencoder_disabled=True,
        learned_source384_inference=False, edits_mapped=False, official_verifier_executed=False,
        full_live_supervisor_deployment_qualified=False)
    started = time.monotonic()
    try:
        assert not any(os.environ.get(name) for name in ('OPENAI_API_KEY', 'XAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'))
        import duckdb
        con = duckdb.connect(':memory:')
        platform = con.execute('PRAGMA platform').fetchone()[0]
        extension_dir = Path.home() / '.duckdb/extensions' / ('v' + duckdb.__version__) / platform
        extension_dir.mkdir(parents=True, exist_ok=True)
        for name in ('ducklake', 'httpfs', 'quack'):
            source = Path('/extensions') / (name + '.duckdb_extension')
            target = extension_dir / source.name
            if target.exists():
                assert sha(target.read_bytes()) == sha(source.read_bytes())
            else:
                shutil.copyfile(source, target)
            con.execute('LOAD ' + name)
        con.close()
        repository = ROOT / 'repository'
        assert repository.is_dir() and not any(repository.iterdir())
        originals = {name: (Path('/permitted') / name).read_bytes() for name in ('eigen.py', 'eval.py')}
        for name, raw in originals.items():
            (repository / name).write_bytes(raw)
        expected_hashes = {name: sha(raw) for name, raw in originals.items()}
        for argv in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Offline Qualification',
                     '-c', 'user.email=qualification@example.invalid', 'commit', '-qm', 'permitted public input']):
            subprocess.run(['git', '-C', str(repository), *argv], check=True, capture_output=True,
                env={**os.environ, 'GIT_AUTHOR_DATE':'2026-10-13T00:00:00Z', 'GIT_COMMITTER_DATE':'2026-10-13T00:00:00Z'})
        from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as preflight
        from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
        state = PRIVATE / 'preflight'
        result = preflight.qualify(repository=repository, instruction=Path('/permitted/instruction.md'), state=state,
            task_profile=json.loads(Path('/permitted/profile.json').read_bytes()), disable_intent_autoencoder=True)
        assert result['qualified'] and result['provider_calls'] == 0
        private_paths = [str(path) for path in sorted(state.rglob('*.duckdb')) if path.is_file()][:3]
        assert private_paths
        write(CONTROL / 'private-paths.json', private_paths)
        workspace = ROOT / 'worktrees/task'
        subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach', '-q', str(workspace)], check=True, capture_output=True)
        prompt = (state / 'worker-prompt.txt').read_bytes()
        (Path('/results') / 'owner-input-prompt.txt').write_bytes(prompt)
        arms = {}
        for profile in ('identity@1', 'python-comments@1'):
            arms[profile] = qualify_arm(profile, result=result, prompt=prompt, originals=originals,
                expected_hashes=expected_hashes, state=state, workspace=workspace,
                repository=repository, private_paths=private_paths)
        assert all(arm['complete'] for arm in arms.values())
        outcome.update(complete=True, arms=arms, owner_input_prompt_sha256=sha(prompt),
            owner_input_prompt_bytes=len(prompt), public_source_hashes=expected_hashes,
            shared_preflight=True, separate_sequential_worker_sessions=True,
            execution_order=['identity@1','python-comments@1'], timing_comparison_claimed=False)
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error)[:500])
        (PRIVATE / 'probe-failure.txt').write_text(traceback.format_exc())
    finally:
        outcome['seconds'] = time.monotonic() - started
        write(Path('/results/probe-qualification.json'), outcome)
        print(json.dumps({key: outcome.get(key) for key in ('complete', 'error_type', 'seconds', 'actual_cli_stdout_utf8_bytes')}), flush=True)
    return 0 if outcome['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
