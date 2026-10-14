"""Stream-check a newly built archive and all selected production capabilities."""
import json
from pathlib import Path
import sys
import tarfile
import time
from campaign_common import A, Q, binding, need, now, read, recipe, sha, write_new


def verify():
    recipe_value = recipe()
    directory = Path(recipe_value['runtime_directory'])
    manifest_path = directory / 'manifest.json'
    archive = directory / 'runtime.tar.gz'
    manifest = read(manifest_path)
    need(sha(archive) == manifest['archive_sha256'], 'runtime digest differs')
    rows = {row['path']: row for row in manifest['files']}
    need(len(rows) == len(manifest['files']), 'duplicate archive inventory members')
    before = time.monotonic(); seen = set(); total = 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            need(member.name in rows and member.name not in seen and member.isfile(), 'unexpected archive member')
            row = rows[member.name]
            need(member.size == row['bytes'] and member.mode == row['mode'], 'archive member metadata differs')
            import hashlib
            with stream.extractfile(member) as content:
                need(hashlib.file_digest(content, 'sha256').hexdigest() == row['sha256'], 'archive member bytes differ')
            seen.add(member.name); total += member.size
    need(seen == set(rows), 'archive inventory incomplete')
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
    from benchmarks.agent_supervisor.container_coding.terminal_source_edit_policy import require_source_edit_archive
    from benchmarks.agent_supervisor.container_coding.terminal_capsule_selection_policy import require_capsule_budget_archive
    from benchmarks.agent_supervisor.container_coding.terminal_numerical_policy import require_numerical_archive
    from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import require_worker_capability
    from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import require_semantic_metadata_archive
    from benchmarks.agent_supervisor.container_coding.terminal_deployment import validate_source384_binding, validate_torch_cpu_wheel_binding
    from benchmarks.agent_supervisor.container_coding.terminal_grok_deployment import validate_grok_binding
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import require_runtime_provider_profile
    require_source_view_archive(manifest, 'identity@1', 'source384-5cpu-20gib-formalization@1')
    require_source_edit_archive(manifest, 'mapped-span@2')
    require_capsule_budget_archive(manifest, 'one-capsule@1')
    require_semantic_metadata_archive(manifest, 'legacy')
    require_numerical_archive(manifest, 'python-numerical-control-structure@3')
    require_worker_capability(manifest, 'source384-5cpu-20gib-formalization@1')
    require_runtime_provider_profile(manifest, 'grok-4.7-cli-1.0.46@1')
    need(validate_source384_binding(manifest) is not None, 'Source384 missing')
    need(validate_torch_cpu_wheel_binding(manifest) is not None, 'Torch wheel missing')
    need(validate_grok_binding(manifest, required=True) is not None, 'Grok missing')
    result = dict(schema='capsule-budget-runtime-verification@1', observed_at_utc=now(),
        archive=binding(archive), manifest=binding(manifest_path),
        all_archive_members_match_manifest=True, current_source_capabilities_match=True,
        file_count=len(seen), uncompressed_bytes=total, elapsed_seconds=time.monotonic()-before,
        source_head='recorded_separately_by_bundle_and_freeze', provider_calls=0, model_calls=0,
        credentials_inspected=False, verifier_or_solution_contents_read=False)
    write_new(A / 'live-preparation/verification-result.json', result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    verify()
