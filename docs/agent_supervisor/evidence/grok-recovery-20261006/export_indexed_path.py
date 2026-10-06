"""Closed counts and reference digests for one completed indexed trial."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
REASONS = {
    'unsupported_or_incomplete_source_inventory', 'unsupported_output_effect_or_language',
    'unsupported_module_or_signature_shape', 'ambiguous_supported_repairs',
    'no_supported_keyword_mismatch', 'required_local_prover_unavailable',
    'unsupported_binding_scope', 'doctor_analysis_secret_screen_refused',
    'native_doctor_planning_not_admitted', 'native_doctor_gate_abstained',
    'native_doctor_transaction_not_admitted', 'native_doctor_transaction_rejected',
    'ambiguous_header_candidates', 'no_supported_header_candidate',
    'local_operator_does_not_cover_declared_outputs', 'operator_proof_bounds_exceeded',
    'doctor_task_data_contract_unavailable',
}
GAPS = {
    'program_and_support_partition_not_observed', 'empty_program_context_required',
    'semantic_source_coverage_incomplete', 'task_data_semantic_contract_unavailable',
    'task_behavior_contract_not_selected', 'generic_operator_output_coverage_missing',
    'single_declared_task_no_parallel_decomposition', 'local_prover_unavailable',
    'local_contract_proof_not_reported',
}
ENUMS = {'residual', 'candidate_ready', 'model_router', 'doctor_candidate',
    'doctor_contract_candidate', 'available', 'unavailable', 'not_reported',
    'complete_hash_match', 'scoped_hash_match', 'incomplete',
    'not_established_by_this_observation', 'reviewed_local_header_contract', 'not_selected',
    'closed_local_keyword_rename', 'closed_imported_alias_call', 'unrecognized',
    'named_structural_check_not_verified', 'not_classified', 'executable_present',
    'not_executable', 'not_established', 'reviewed_local_operator_only',
    'single_declared_task', 'multiple_declared_tasks', 'not_observed_here'}
PHASES = {'empty_native_world_capture', 'final_evidence_verification',
    'persisted_snapshot_reopen', 'retrieval_persistence',
    'semantic_build_reconstruction_and_hydration', 'source384_capture_and_inference',
    'vector_qualification'}
BOOLS = set('canonical_tasks_created learned_embeddings neural_inference_replayed '
    'completion_authority execution_authority formalization_authority model_promotion_performed '
    'nomination_only proof_authority initial_indexes_reused native_fact_rows_replayed '
    'semantic_equivalence_claimed publication_authority derived_runtime_admitted '
    'named_structural_check validation_semantics_verified checks_executed_by_assessment '
    'whole_task_behavior_verified candidate_ready_reported native_stage_reported receipt_reported '
    'local_contract_proof_reported independently_reverified_by_assessment whole_program_verified '
    'executed_by_assessment presence_establishes_proof intent_requirement_artifact_declared'.split())
INTS = set('full_capsules indexed_symbols world_task_count training_steps excluded_source_files '
    'harness_support_files inference_python_files omitted_candidates program_source_files provider_calls '
    'source_files task_data_files worker_capsules worker_semantic_bytes new_embedding_calls '
    'residual_successors residual_work_proposals signed_input_count program_input_count '
    'harness_support_count task_data_input_count validation_count non_python_output_count '
    'other_reason_count declared_task_count assessment_provider_calls'.split())


def mapping(value):
    return value if type(value) is dict else {}


def atom(value):
    if type(value) is bool or value is None:
        return value
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 10**12:
        return value
    if type(value) is str and value in ENUMS:
        return value
    return None


def pick(value, names):
    value = mapping(value)
    output = {}
    for name in names.split():
        item = value.get(name)
        if name in BOOLS:
            output[name] = item if type(item) is bool else None
        elif name in INTS:
            output[name] = item if type(item) is int and 0 <= item <= 10**12 else None
        elif name == 'seconds':
            output[name] = item if type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 10**12 else None
        else:
            output[name] = item if type(item) is str and item in ENUMS else None
    return output


def digest(value):
    return value if type(value) is str and re.fullmatch('[0-9a-f]{64}', value) else None


def reference(value):
    if type(value) is not str or not 1 <= len(value.encode()) <= 512:
        return None
    return {'bytes': len(value.encode()), 'value_sha256': hashlib.sha256(value.encode()).hexdigest()}


def project(report):
    initial = mapping(report.get('initial_context'))
    context = mapping(report.get('context'))
    source = mapping(initial.get('source384_context'))
    summary = mapping(source.get('summary'))
    doctor = mapping(report.get('doctor_dispatch'))
    capability = mapping(doctor.get('symbolic_capabilities'))
    residual = mapping(doctor.get('residual_context'))
    reasons = doctor.get('reason_codes')
    reasons = reasons if type(reasons) is list and len(reasons) <= 128 else []
    gaps = capability.get('gap_codes')
    gaps = gaps if type(gaps) is list and len(gaps) <= 128 else []
    inventory = mapping(capability.get('inventory'))
    return {
        'actual_index': {**pick(initial, 'canonical_tasks_created full_capsules indexed_symbols learned_embeddings seconds world_task_count'),
            'descriptor_sha256': digest(mapping(initial.get('descriptor')).get('sha256')),
            'phases': {key: atom(value) for key, value in mapping(initial.get('nonoverlapping_seconds')).items() if key in PHASES},
            'reference_hashes': {key: reference(initial.get(key)) for key in ('index_id', 'semantic_root_cid', 'world_snapshot_cid')}},
        'source384': {**pick(source, 'seconds neural_inference_replayed training_steps'),
            **{key: digest(source.get(key)) for key in ('checkpoint_sha256', 'inference_sha256')},
            'counts': pick(summary, 'excluded_source_files harness_support_files inference_python_files omitted_candidates program_source_files provider_calls source_files task_data_files training_steps'),
            'authority': pick(summary, 'completion_authority execution_authority formalization_authority model_promotion_performed nomination_only proof_authority')},
        'admitted_context': {**pick(context, 'initial_indexes_reused indexed_symbols full_capsules worker_capsules worker_semantic_bytes learned_embeddings native_fact_rows_replayed new_embedding_calls nomination_only provider_calls seconds semantic_equivalence_claimed'),
            'same_initial_index_reference': (context.get('index_id') == initial.get('index_id')
                if reference(context.get('index_id')) is not None and reference(initial.get('index_id')) is not None else None),
            'reference_hashes': {key: reference(context.get(key)) for key in ('index_id', 'semantic_root_cid', 'world_snapshot_cid')}},
        'doctor': {**pick(doctor, 'status route analysis_status provider_calls residual_successors residual_work_proposals completion_authority publication_authority'),
            'known_reason_codes': sorted({item for item in reasons if type(item) is str and item in REASONS}),
            'unexported_reason_count': sum(type(item) is not str or item not in REASONS for item in reasons),
            'analysis_reference': reference(doctor.get('analysis_observation_cid')),
            'residual_context': {**pick(residual, 'derived_runtime_admitted provider_calls completion_authority'),
                'sha256': digest(residual.get('sha256')),
                'reference_hashes': {key: reference(residual.get(key)) for key in ('context_cid', 'native_capsule_id')}},
            'symbolic_capabilities': {
                'inventory': {**pick(inventory, 'signed_input_count doctor_hash_binding analysis_status semantic_coverage'),
                    'partition': pick(inventory.get('partition'), 'program_input_count harness_support_count task_data_input_count')},
                'contracts': pick(capability.get('contracts'), 'named_structural_check validation_semantics_verified validation_scope validation_count checks_executed_by_assessment task_behavior_contract whole_task_behavior_verified'),
                'operators': pick(capability.get('operators'), 'selected_workflow non_python_output_count candidate_ready_reported other_reason_count'),
                'proof': pick(capability.get('proof'), 'native_stage_reported receipt_reported local_contract_proof_reported independently_reverified_by_assessment whole_program_verified scope'),
                'provers': {**pick(capability.get('provers'), 'executed_by_assessment presence_establishes_proof'),
                    'presence': pick(mapping(capability.get('provers')).get('presence'), 'lean z3')},
                'planning': pick(capability.get('planning'), 'declared_task_count intent_requirement_artifact_declared symbolic_selection_execution decomposition parallel_execution'),
                'gap_codes': sorted({item for item in gaps if type(item) is str and item in GAPS}),
                'unexported_gap_count': sum(type(item) is not str or item not in GAPS for item in gaps),
                'authority': pick(capability, 'assessment_provider_calls benchmark_solvability proof_authority execution_authority publication_authority completion_authority')}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9]{2}', args.attempt):
        parser.error('two-digit completed trial required')
    spec = importlib.util.spec_from_file_location('closed_live', A / 'qualification/inspect_live_trials.py')
    live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
    trial = 'grok-tune-mjcf-' + args.attempt
    paths = list((A / trial / 'jobs/supervisor-full-tune-mjcf').glob('tune-mjcf__*/agent/supervisor-result.json'))
    if len(paths) != 1:
        raise SystemExit('one exact completed driver report required')
    report, identity = live.read_metadata(A, paths[0], max_bytes=8_388_608)
    output = dict(schema='grok-recovery-indexed-path-observation@2', trial_name=trial,
        retained_driver_report=identity, raw_source_model_formula_verifier_or_credential_data_exported=False,
        proof_or_completion_inferred_from_index=False, **project(report))
    target = A / ('indexed-path-observation-' + args.attempt + '.json')
    with target.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
