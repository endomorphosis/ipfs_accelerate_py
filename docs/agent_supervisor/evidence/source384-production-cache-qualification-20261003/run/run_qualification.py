"""Ordinary pinned production qualification; no diagnostic hook injection."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
WORK = Path('/home/barberb/lift_coding')
A = WORK / '.worktrees/ir-release-accelerate-20261002'
D = WORK / '.worktrees/ir-release-datasets-20261002'
TASK = WORK / '.benchmarks/terminal-bench-2/fix-code-vulnerability'
POLICY = 'source384-native-aarch64-dontneed@1'
PROFILE = 'source384-5cpu-12gib@1'
PREVIOUS = WORK / 'artifacts/source384-full-supervisor-preflight-20261003'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(raw)
    return digest.hexdigest()


def write(name, value):
    with (BASE / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def live_pins(frozen):
    result = {}
    for name in frozen:
        prefix, relative = name.split('/', 1)
        result[name] = sha({'source': A, 'datasets': D}[prefix] / relative)
    return result


def containers():
    result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=ipfs-native-deploy-',
        '--format', '{{.ID}} {{.Names}} {{.Status}}'], capture_output=True, text=True, timeout=15)
    return dict(returncode=result.returncode, rows=result.stdout.splitlines())


def component_review(admission):
    review = admission.get('component_review')
    if type(review) is not dict:
        raise RuntimeError('explicit component review scopes required')
    for key in ('passed_scopes', 'resource_blocked_scopes'):
        values = review.get(key)
        if type(values) is not list or any(type(value) is not str or not value.strip() for value in values):
            raise RuntimeError('explicit passed and resource-blocked component scopes required')
    if (not review['passed_scopes']
            or review.get('unresolved_code_or_test_packaging_failures') != []):
        raise RuntimeError('reviewed passing controls and no unresolved code or test-packaging failures required')
    return review


async def qualify(archive, selection):
    from benchmarks.agent_supervisor.container_coding.terminal_deployment import qualify_original_container
    return await asyncio.wait_for(qualify_original_container(
        task_dir=TASK, archive_dir=archive, output=BASE / 'qualification-01',
        install_codex=True, auth_json=Path('/home/barberb/.codex/auth.json'),
        keep_container=False, resource_profile=PROFILE, source384_context=True,
        setup_cache_selection=selection), timeout=2300)


def main():
    if (BASE / 'qualification-01').exists() or (BASE / 'qualification-command.json').exists():
        raise RuntimeError('fresh qualification output required; preserve prior attempts')
    admission = read(BASE / 'run-admission.json')
    for key in ('root_go_received', 'current_producer_component_review_complete', 'fresh_archive_audited'):
        if admission.get(key) is not True:
            raise RuntimeError('component scope review and archive audit must precede execution')
    reviewed_components = component_review(admission)
    audit_path = BASE / 'build-audit.json'
    if admission['build_audit_sha256'] != sha(audit_path):
        raise RuntimeError('execution admission does not bind the actual audited archive')
    audit = read(audit_path)
    archive = BASE / 'bundle'
    manifest = read(archive / 'manifest.json')
    selection = read(archive / 'setup-cache-selection.json')
    if (sha(archive / 'manifest.json') != audit['manifest_sha256']
            or sha(archive / 'runtime.tar.gz') != audit['archive_sha256']
            or manifest['archive_sha256'] != audit['archive_sha256']
            or selection != audit['setup_cache_selection']
            or selection['policy'] != POLICY
            or selection['manifest_sha256'] != audit['manifest_sha256']):
        raise RuntimeError('production archive or explicit cache selection changed')
    frozen = read(BASE / 'frozen-production-pins.json')
    before = live_pins(frozen)
    inventory = {row['path']: row for row in manifest['files']}
    if before != frozen or any(inventory[name]['sha256'] != digest for name, digest in frozen.items()):
        raise RuntimeError('live or archived production owner differs from final freeze')
    recipe_path = PREVIOUS / 'prepare-command.json'
    recipe = read(recipe_path)
    public_inputs = {name: sha(TASK / name) for name in ('instruction.md', 'task.toml', 'environment/Dockerfile')}
    previous_qualifier = WORK / 'artifacts/source384-docker-inference-publication-20261003/run_qualification.py'
    write('qualification-command.json', dict(
        argv=[sys.executable, '-B', str(Path(__file__).resolve())], cwd=str(Path.cwd()),
        env={key: os.environ.get(key) for key in recipe['env']},
        inherited_environment_recipe=str(recipe_path), inherited_environment_recipe_sha256=sha(recipe_path),
        preceding_ordinary_qualifier=str(previous_qualifier), preceding_ordinary_qualifier_sha256=sha(previous_qualifier),
        script_sha256=sha(Path(__file__)), build_audit_sha256=sha(audit_path),
        admission_sha256=sha(BASE / 'run-admission.json'), component_review=reviewed_components,
        selected_archive_sha256=audit['archive_sha256'], selected_manifest_sha256=audit['manifest_sha256'],
        resource_profile=PROFILE, cache_policy=POLICY, diagnostic_only=False,
        source384_native_seconds=90, qualification_inner_seconds=270, qualification_exec_seconds=300,
        qualification_outer_seconds=2300, setup_operation_timeouts_unchanged=True,
        qualification_aggregate_setup_timeout=None, later_full_harbor_setup_seconds=1800,
        provider_calls=0, official_verifier_executed=False, benchmark_result=False,
        credential_contents_recorded=False, runtime_hooks_installed=False))
    write('qualification-before-pins.json', before)
    write('qualification-public-task-pins.json', public_inputs)
    original_containers = containers()
    write('qualification-containers-before.json', original_containers)
    begun = time.monotonic()
    code = 0
    print(json.dumps(dict(phase='ordinary_production_qualification_started', at=time.time())), flush=True)
    try:
        result = asyncio.run(qualify(archive, selection))
        if (result.get('qualified') is not True or result['source384'].get('qualified') is not True
                or result['setup_cache'].get('completed') is not True
                or result['setup_cache'].get('selection') != selection):
            raise RuntimeError('ordinary production qualification did not pass')
        write('qualification-result.json', result)
    except BaseException as exc:
        code = 1
        traceback.print_exc()
        write('qualification-failure.json', dict(error_type=type(exc).__name__, error=str(exc)[:2048]))
    finally:
        after = live_pins(frozen)
        public_after = {name: sha(TASK / name) for name in public_inputs}
        final_containers = containers()
        write('qualification-after-pins.json', after)
        write('qualification-containers-after.json', final_containers)
        cleanup_verified = (original_containers['returncode'] == final_containers['returncode'] == 0
            and {row.split()[0] for row in final_containers['rows']}
                <= {row.split()[0] for row in original_containers['rows']})
        if before != after or public_inputs != public_after or not cleanup_verified:
            code = 1
        write('qualification-exit.json', dict(returncode=code, seconds=time.monotonic()-begun,
            source_pins_unchanged=before == after, public_task_inputs_unchanged=public_inputs == public_after,
            cleanup_verified=cleanup_verified, diagnostic_only=False,
            provider_calls=0, official_verifier_executed=False, benchmark_result=False))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
