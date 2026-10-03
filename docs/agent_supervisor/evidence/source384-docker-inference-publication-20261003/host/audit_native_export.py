"""Verify the retained native export; publish metadata, never its source payload."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def walk(value, path=()):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + (str(index),))


def main():
    raw = (BASE / 'docker-01/native-inference.json').read_bytes()
    native = json.loads(raw)
    context = json.loads((BASE / 'docker-01/source384-context.json').read_bytes())
    report = native['report']
    assert sha(raw) == context['inference_sha256']
    assert report['key'] == context['native_inference_key']
    artifact = canonical(report)
    assert native['artifact'] == dict(bytes=len(artifact), sha256=sha(artifact))
    assert native['inference_executed'] is True and native['native_worker_executed'] is True
    assert report['output']['model_loads'] == 1
    assert report['coverage'] == context['coverage']
    false_flags = ('claim_proved', 'proof_authority', 'execution_authority',
                   'completion_authority', 'promotion_performed', 'training_executed',
                   'training_labels_used', 'source_executed', 'source_semantics_verified',
                   'whole_file_semantics_verified')
    assert all(native[key] is False and report[key] is False for key in false_flags)
    source_fields = [(path, value) for path, value in walk(native)
                     if path and path[-1] in {'source_text', 'normalized_source_text'}]
    assert source_fields and all(type(value) is str for _, value in source_fields)
    source_values = {value for _, value in source_fields if len(value) >= 16}
    public_checks = []
    for relative in ('result.json', 'docker-01/source384-context.json',
                     'docker-01/source384-result.json', 'docker-01/qualification.json',
                     'docker-01/deployment/deployment.json'):
        payload = (BASE / relative).read_bytes()
        value = json.loads(payload)
        leaves = list(walk(value))
        assert not any(path and path[-1] in {'source_text', 'normalized_source_text'}
                       for path, _ in leaves)
        assert not any(isinstance(item, str) and item in source_values for _, item in leaves)
        assert not any(isinstance(item, list) and len(item) >= 32
                       and all(type(x) in (int, float) for x in item) for _, item in leaves)
        public_checks.append(dict(path=relative, sha256=sha(payload), bytes=len(payload),
                                  raw_source_fields=0, complete_normalized_source_matches=0,
                                  numerical_parameter_arrays=0))
    verification = dict(
        schema='source384-retained-native-export-verification@1',
        scope='Metadata verified against the retained original export; this is not that export.',
        original_export=dict(bytes=len(raw), sha256=sha(raw), publicly_included=False),
        original_export_digest_matches_qualified_receipt=True,
        native_key_matches_qualified_receipt=True,
        canonical_registry_report_digest_verified=True,
        native_registry_artifact=native['artifact'],
        checkpoint_sha256=report['key']['original_checkpoint_sha256'],
        model_loads=report['output']['model_loads'],
        native_worker_executed=True, inference_executed=True,
        worker_receipt=report['worker_receipt'], coverage=report['coverage'],
        authority_and_semantics_flags={key:native[key] for key in false_flags},
        source_payload_excluded=True, benchmark_result=False,
    )
    audit = dict(
        schema='source384-native-export-publication-audit@1',
        disposition='retain_original_locally_exclude_from_public_package',
        original_export=verification['original_export'],
        reason='Native preparation embeds normalized benchmark source and selected source text.',
        raw_source_field_count=len(source_fields),
        raw_source_field_locations=['report.preparation.files[*].extraction.units[*].normalized_source_text',
                                    'report.preparation.selected_inputs[*].source_text'],
        public_receipt_checks=public_checks,
        method='Inspect structured fields; reject raw source fields, exact normalized-source string leaves of at least16 characters and numerical parameter arrays. Manually inspect two typed candidate summaries.',
        limits='Public receipts expose source paths, symbol names, hashes and generated typed candidates; full source/activation export remains local. This is not a general secret scanner.',
        audit_script_sha256=sha(Path(__file__).read_bytes()),
    )
    for filename, value in [('native-inference-verification.json', verification),
                            ('native-inference-public-audit.json', audit)]:
        (BASE / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(native_sha256=sha(raw), model_loads=1,
                          raw_source_fields=len(source_fields), raw_export_excluded=True)))


if __name__ == '__main__':
    main()
