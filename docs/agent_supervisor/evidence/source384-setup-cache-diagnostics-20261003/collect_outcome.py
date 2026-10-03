"""Read completed diagnostic artifacts; export selected metadata, never source/model payloads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read(path, bound=1_000_000):
    if not path.exists():
        return None, None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= bound:
            raise ValueError('bounded regular JSON receipt required: ' + path.name)
        body = stream.read(bound + 1)
        after = os.fstat(stream.fileno())
    if len(body) != info.st_size or (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('receipt changed while collecting: ' + path.name)
    value = json.loads(body)
    if type(value) is not dict:
        raise ValueError('object receipt required')
    return value, {'bytes': len(body), 'sha256': digest(body)}


def scalar_fields(value, names):
    if value is None:
        return None
    result = {}
    for name in names:
        if name not in value:
            continue
        child = value[name]
        if child is not None and type(child) not in (str, int, float, bool):
            raise ValueError('scalar metadata field required: ' + name)
        if isinstance(child, str) and len(child) > 1024:
            raise ValueError('metadata string exceeds bound')
        if type(child) is float and not math.isfinite(child):
            raise ValueError('finite metadata required')
        result[name] = child
    return result


def coverage_counts(value):
    result = scalar_fields(value, ('files', 'functions', 'selected_units'))
    for name in ('candidate_statuses', 'file_dispositions', 'unit_dispositions'):
        rows = value[name]
        if type(rows) is not dict or len(rows) > 64:
            raise ValueError('bounded coverage counts required')
        if any(not re.fullmatch('[a-z_]{1,80}', key) or type(count) is not int or count < 0
               for key, count in rows.items()):
            raise ValueError('closed count metadata required')
        result[name] = rows
    return result


def collect(base):
    pins = {}

    def receipt(relative, bound=1_000_000):
        value, pin = read(base / relative, bound)
        if pin is not None:
            pins[relative] = pin
        return value

    finished = receipt('exit.json')
    if finished is None:
        raise ValueError('completed run exit.json required; never infer a pending outcome')
    scope = receipt('scope.json') or {}
    provenance = receipt('host-archive-provenance.json')
    qualified = receipt('docker-01/qualification.json') or {}
    context = receipt('docker-01/source384-context.json') or {}
    phase = receipt('docker-01/source384-result.json') or {}
    outer_failure = receipt('failure.json') or {}
    deployment = receipt('docker-01/deployment/deployment.json')
    archive_cache = receipt('cache-advice.json')
    archive_status = receipt('cache-advice-status.json')
    native_cache = receipt('codex-cache-advice.json')
    native_status = receipt('codex-cache-advice-status.json')
    native_exposure = receipt('docker-01/worker-boundary/installation/native-codex-binary.log')
    resources = receipt('docker-01/resources.json')
    admission = receipt('docker-01/admission-estimate.json')
    cleanup = receipt('containers-after.json')
    verdict = receipt('verdict.json') or {}
    samples = {}
    resource_names = ('cpu_slots', 'total_memory_mb', 'available_memory_mb',
                      'memory_stall_percent', 'cpu_stall_percent', 'io_stall_percent')
    for name in ('cache-resources-before.json', 'cache-resources-after.json',
                 'codex-cache-resources-before.json', 'codex-cache-resources-after.json',
                 'detailed-resources.json'):
        sample = receipt('docker-01/' + name)
        if sample is not None:
            samples[name] = scalar_fields(sample.get('host'), resource_names)

    preservation = None
    if deployment is not None:
        original, retained = deployment['original_inputs'], deployment['retained_inputs']
        preservation = {
            'original_count': len(original['files']),
            'original_inventory_sha256': digest(canonical(original)),
            'retained_inventory_sha256': digest(canonical(retained)),
            'original_inputs_equal_across_deployment': original == retained,
            'reported_preserved': deployment['task_source_preserved'],
            'scope': 'Full original input inventory compared across deployment; no post-agent full-tree claim.',
        }
        if 'signed_source_hashes' in context:
            signed = context['signed_source_hashes']
            preservation.update(signed_input_count=len(signed),
                original_hashes_match_signed_context=all(signed.get(name) == row['sha256']
                    for name, row in original['files'].items()),
                signed_source_hashes_sha256=digest(canonical(signed)))

    inference = {'verified': False, 'model_loads': None, 'coverage': None,
                 'native_worker_executed': None, 'raw_export_publicly_included': False}
    native, native_pin = read(base / 'docker-01/native-inference.json', 16_000_000)
    if native is not None:
        pins['docker-01/native-inference.json'] = {**native_pin, 'payload_excluded': True}
        report = native['report']
        if native_pin['sha256'] != context.get('inference_sha256'):
            raise ValueError('native export differs from context pin')
        if report['key'] != context.get('native_inference_key'):
            raise ValueError('native inference key differs from context')
        body = canonical(report)
        if native['artifact'] != {'bytes': len(body), 'sha256': digest(body)}:
            raise ValueError('canonical published native report differs from descriptor')
        if report['coverage'] != context.get('coverage') or report['worker_receipt'] != context.get('native_worker_receipt'):
            raise ValueError('native coverage/worker receipt differs from context')
        if digest(canonical(report['output'])) != report['worker_receipt']['output_sha256']:
            raise ValueError('worker output does not match its receipt')
        false_flags = ('claim_proved', 'proof_authority', 'execution_authority',
            'completion_authority', 'promotion_performed', 'training_executed', 'training_labels_used',
            'source_executed', 'source_semantics_verified', 'whole_file_semantics_verified')
        if any(native.get(key) is not False or report.get(key) is not False or report['output'].get(key) is not False
               for key in false_flags):
            raise ValueError('unreviewed authority or semantics assertion')
        if native.get('inference_executed') is not True or native.get('native_worker_executed') is not True:
            raise ValueError('native inference execution not established')
        if type(report['output']['model_loads']) is not int or report['output']['model_loads'] < 1:
            raise ValueError('positive observed model-load count required')
        inference = {
            'verified': True, 'model_loads': report['output']['model_loads'],
            'coverage': coverage_counts(report['coverage']), 'native_worker_executed': True,
            'checkpoint_sha256': report['key']['original_checkpoint_sha256'],
            'native_key_sha256': digest(canonical(report['key'])),
            'published_report': native['artifact'], 'raw_export': native_pin,
            'raw_export_publicly_included': False,
            'worker_receipt': scalar_fields(report['worker_receipt'], ('returncode', 'elapsed_ms',
                'input_sha256', 'output_sha256', 'device', 'memory_mb', 'memory_enforcement',
                'provider_calls', 'workspace_cleaned', 'parent_lease_id', 'child_lease_id')),
            'authority_and_semantics_flags': {key: False for key in false_flags},
        }

    cache_fields = ('schema', 'selected_files', 'selected_bytes', 'advised_files', 'advised_bytes',
        'body_reads', 'body_read_bytes', 'body_writes', 'metadata_unchanged', 'seconds',
        'freed_bytes_claimed', 'global_drop_caches', 'cgroup_writes', 'advice_is_best_effort',
        'archive_sha256', 'manifest_sha256', 'post_boundary_receipt_sha256')
    return {
        'schema': 'source-free-cache-diagnostic-outcome@1', 'artifact_directory': str(base),
        'diagnostic_only': True, 'production_qualification_claimed': False,
        'raw_qualifier_passed': qualified.get('qualified'),
        'native_source384_phase_passed': phase.get('qualified'),
        'process': scalar_fields(finished, ('returncode', 'seconds', 'source_pins_unchanged')),
        'error': scalar_fields(phase if phase else outer_failure, ('error_type', 'error_phase')),
        'resource_profile': scope.get('resource_profile'),
        'actual_cgroups': scalar_fields(resources, ('cpu_max', 'memory_max', 'detected_cpu_slots',
            'detected_total_memory_mb', 'available_memory_mb')),
        'resource_samples': samples,
        'failure_resources': scalar_fields(phase.get('failure_resources'), resource_names),
        'admission_estimate': scalar_fields(admission, ('live_available_memory_mb',
            'derived_default_headroom_mb', 'requested_parent_memory_mb', 'minimum_available_memory_mb',
            'exact_admission_decision', 'actual_native_admission_required')),
        'reported_failure_boundary': verdict.get('failed_boundary', verdict.get('outcome')),
        'host_archive_generation': scalar_fields(provenance, ('archive_sha256', 'archive_not_rebuilt',
            'actual_host_deployment_source_sha256', 'archive_deployment_source_sha256',
            'archived_deployment_module_not_used_to_install_python', 'selected_source384_owners_unchanged')),
        'archive_cache': scalar_fields(archive_cache, cache_fields),
        'archive_cache_error_count': len(archive_cache['errors']) if archive_cache else None,
        'archive_cache_status': scalar_fields(archive_status, ('returncode', 'candidate_sha256')),
        'provider_binary_cache': scalar_fields(native_cache, cache_fields),
        'provider_binary_cache_error_count': len(native_cache['errors']) if native_cache else None,
        'provider_binary_cache_status': scalar_fields(native_status, ('returncode', 'candidate_sha256', 'post_boundary_receipt_sha256')),
        'provider_binary_exposure_present': native_exposure is not None,
        'input_preservation': preservation, 'inference': inference,
        'timings_seconds': {**scalar_fields(phase, ('prepare_seconds', 'source384_seconds',
            'initial_context_seconds', 'context_seconds', 'warm_observation_seconds', 'seconds', 'phase_seconds')),
            'deployment': deployment.get('seconds') if deployment else None},
        'deadlines_seconds': scalar_fields(scope, ('qualification_outer_seconds', 'qualification_exec_seconds',
            'qualification_inner_seconds', 'source384_native_seconds')),
        'provider_calls_reported': finished.get('provider_calls'), 'token_counters': None,
        'official_verifier_executed': finished.get('official_verifier_executed'),
        'cleanup': {'receipt_present': cleanup is not None,
            'no_matching_containers': cleanup is not None and cleanup.get('returncode') == 0 and cleanup.get('rows') == [],
            'scope': 'Retained docker-ps filter observation; not a global process inventory.'},
        'receipt_pins': pins, 'benchmark_result': False, 'advantage_claimed': False,
        'payload_policy': 'No raw source, embeddings, weights, native report bodies, credentials or verifier inputs copied. Model/coverage claims require exact retained-native-export verification.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh derived receipt path required')
    value = collect(args.base.resolve(strict=True))
    args.output.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
    print(json.dumps({'output': str(args.output), 'raw_qualifier_passed': value['raw_qualifier_passed'],
                      'inference_verified': value['inference']['verified']}))
