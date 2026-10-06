"""Retain scoped metadata/scripts; keep weights, stores and full record dumps local."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from custody import capture, read, require, write

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/barberb/lift_coding/.worktrees/contextual-state-registration-accelerate-20261006')
DESTINATION = ROOT / 'docs/agent_supervisor/evidence/contextual-state-registration-20261006'
EXCLUDED_NAMES = {'model-manager-before.duckdb', 'model-manager-before-records.json',
    'model-manager-after-records.json', 'hf-legal384-repository-info.json',
    'hf-legal768-repository-info.json', 'selected-state.json'}
EXCLUDED_DIRS = {'__pycache__', 'remote-download', 'mirror-download'}


def main():
    # These are distinct finite observations with different authority scopes.
    registration = json.loads((OUT / 'model-manager-registration.json').read_bytes())
    require(registration['completed'] is True and registration['before_count'] == 668
            and registration['after_count'] == 670, 'completed two-state persisted run required')
    require(json.loads((OUT / 'review/registration-postwrite-review.json').read_bytes())['completed'] is True,
            'independent postwrite review required')
    mirrors = json.loads((OUT / 'dimension-mirrors/publication-summary.json').read_bytes())
    require(mirrors['completed'] is True, 'two complete dimension mirrors required')
    tests = ET.parse(OUT / 'native-metadata-tests.xml')
    cases = tests.findall('.//testcase')
    require(len(cases) == 158 and len({(row.get('classname'), row.get('name')) for row in cases}) == 158
            and not tests.findall('.//failure') and not tests.findall('.//error')
            and not tests.findall('.//skipped'), '158 distinct passing native controls required')
    qualification = {'schema': 'contextual-state-recovery-qualification/v1', 'completed': True,
        'source_base': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'datasets_metadata_owner': '3b3b994407b2fcfef955ce5153afc2eea01e4eff',
        'native_control_tests': {'passed': 158, 'failed': 0, 'errors': 0, 'skipped': 0,
            'files': ['tests/unit/logic/formalization/autoencoder/test_ir_model_manager_import.py',
                      'tests/unit/logic/formalization/autoencoder/test_ir_decoder_profile_inventory.py'],
            'command': 'env IPFS_DATASETS_PY_MINIMAL_IMPORTS=1 IPFS_DATASETS_AUTO_INSTALL=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/unit/logic/formalization/autoencoder/test_ir_model_manager_import.py tests/unit/logic/formalization/autoencoder/test_ir_decoder_profile_inventory.py -q -o addopts= --junitxml=<retained XML>'},
        'original_native_fragment_routes_resolved': 2, 'original_declared_lanes_preserved': 12,
        'strong_contextual_states_registered': 2, 'selected_manager_before_count': 668,
        'selected_manager_after_count': 670, 'prior_records_and_activity_exactly_preserved': True,
        'schema_and_indexes_unchanged': True, 'supervisor_nominated_and_byte_authenticated': 2,
        'original_contextual_source_cache_custody_recovered': True,
        'dimension_repositories_mirrored': 2, 'new_files_per_repository': 3,
        'prior_HF_paths_preserved': {'384D': 502, '768D': 142},
        'fresh_mirror_downloads_match_original_states': 2,
        'historical_semantic_replay_scores_not_rerun': True, 'current_live_service_process_refreshed': False,
        'diagnostics_retained': ['review/registration-draft-review-01.json',
            'asset-survey/development-diagnostics.json', 'observe_registered_contextual_states-diagnostic-01.py',
            'supervisor-registered-state-observation.log'],
        'supervisor_observation_diagnostic': 'Initial read-only probe expected checkpoint_bytes_authenticated inside authority; corrected to the existing top-level API field. No product code or guards changed.',
        'runtime_ready': False, 'model_quality_qualified': False, 'source_text_reconstruction_qualified': False,
        'fresh_holdout_qualified': False, 'long_context_8192_qualified': False,
        'distillation_executed_in_this_increment': False, 'model_inference_executed_in_this_increment': False,
        'training_executed_in_this_increment': False, 'proof_authority': False,
        'scope': 'Original asset custody, native metadata recovery, genuine persisted registration, public dimension mirrors and a concrete next-gate plan. Current production contextual inference remains unqualified.'}
    write(OUT / 'qualification.json', qualification)
    receipts = []
    for path in sorted(OUT.rglob('*')):
        relative = path.relative_to(OUT)
        if not path.is_file() or path.name in EXCLUDED_NAMES or set(relative.parts) & EXCLUDED_DIRS:
            continue
        if path.suffix not in {'.py', '.json', '.log', '.xml', '.md'}:
            continue
        pin = capture(path)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'retained evidence path already exists')
        with target.open('xb') as handle:
            handle.write(read(pin))
        retained = capture(target)
        require((retained['bytes'], retained['sha256']) == (pin['bytes'], pin['sha256']),
                'retained evidence bytes differ')
        receipts.append({'relative_path': relative.as_posix(), 'source_pin': pin,
                         'retained_pin': retained})
    manifest = {'schema': 'contextual-state-recovery-evidence-retention/v1', 'completed': True,
        'retained_count': len(receipts), 'files': receipts,
        'weights_databases_and_complete_baseline_record_dumps_excluded': True,
        'source_artifacts_preserved': True, 'model_loaded': False, 'proof_authority': False}
    manifest_pin = write(DESTINATION / 'retention-manifest.json', manifest)
    print(json.dumps({'manifest_pin': manifest_pin, 'retained_count': len(receipts)}))


if __name__ == '__main__':
    main()
