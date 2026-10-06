"""Explicit genuine ModelManager registration into the authorized selected store.

Run only after registration-preparation.json and independent review exist.
Native DuckDB connections independently read persisted rows. After its complete
initial load is verified, this isolated manager's public in-memory dictionary is
restricted to the two target IDs so its existing save API touches only those
records. Unrelated records and schema must match exactly after both closes.
There is no inference, optimizer update, serving binding or source embedding call.
"""
import datetime
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from custody import capture, read, require, source, witness, write

BASE = Path('/home/barberb/lift_coding')
OUT = Path(__file__).resolve().parent
DATASETS = BASE / '.worktrees/decoder-profile-recovery-datasets-20261006'
ACCELERATE = BASE / '.worktrees/contextual-state-registration-accelerate-20261006'
STORE = BASE / 'external/ipfs_accelerate/model_manager.duckdb'
RELATIVE = 'ipfs_datasets_py/logic/formalization/autoencoder/ir_model_manager_import.py'
JSON_FIELDS = {'inputs', 'outputs', 'huggingface_config', 'supported_backends',
    'hardware_requirements', 'performance_metrics', 'tags', 'repository_structure', 'serving_config'}


def decode(connection, model_id=None):
    query = 'SELECT * FROM model_metadata ORDER BY model_id' if model_id is None else 'SELECT * FROM model_metadata WHERE model_id = ?'
    cursor = connection.execute(query, [] if model_id is None else [model_id])
    columns = [item[0] for item in cursor.description]
    rows = []
    for row in cursor.fetchall():
        value = dict(zip(columns, row))
        for key in JSON_FIELDS:
            if value.get(key) is not None:
                value[key] = json.loads(value[key])
        for key, item in value.items():
            if isinstance(item, datetime.datetime):
                value[key] = item.isoformat()
        rows.append(value)
    if model_id is None:
        return rows
    require(len(rows) <= 1, 'unique persisted model_id required')
    return rows[0] if rows else None


def schema_snapshot(connection):
    tables = connection.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema','pg_catalog') ORDER BY 1,2").fetchall()
    require(tables == [('main', 'model_metadata')], 'review requires the selected existing single model_metadata table')
    columns = connection.execute("SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema NOT IN ('information_schema','pg_catalog') ORDER BY 1,2,4").fetchall()
    indexes = connection.execute('SELECT schema_name, table_name, index_name, is_unique, is_primary, expressions, sql FROM duckdb_indexes() ORDER BY 1,2,3').fetchall()
    return {'tables': [list(row) for row in tables], 'columns': [list(row) for row in columns],
            'indexes': [list(row) for row in indexes]}


def catalog_targets(manager, expected):
    snapshot = manager.catalog.snapshot()
    cursor, matches = None, []
    for _ in range(100):
        page = manager.list_catalog_models(limit=100, cursor=cursor, snapshot=snapshot)
        for item in page.items:
            for provenance in item.provenance:
                if provenance.source == 'model-manager.persistent' and provenance.source_record_id in expected:
                    matches.append({'source_record_id': provenance.source_record_id, 'canonical_model_id': item.model_id})
        cursor = page.next_cursor
        if cursor is None:
            break
    require(cursor is None and {row['source_record_id'] for row in matches} == expected,
            'genuine manager catalog does not expose both original checkpoint records')
    return {'catalog_revision': snapshot.revision, 'matching_bindings': matches}


def backup_database():
    before = STORE.lstat()
    original = capture(STORE, maximum=4 * 1024**3)
    destination = OUT / 'model-manager-before.duckdb'
    require(witness(STORE.lstat()) == witness(before), 'database changed before backup')
    with STORE.open('rb') as input_handle, destination.open('xb') as output_handle:
        require(witness(os.fstat(input_handle.fileno())) == witness(before), 'backup input changed')
        count = 0
        while True:
            chunk = input_handle.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            require(count <= original['bytes'], 'database grew during backup')
            output_handle.write(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
        require(count == original['bytes'] and witness(os.fstat(input_handle.fileno())) == witness(before),
                'database changed during backup')
    require(witness(STORE.lstat()) == witness(before), 'backup endpoint changed')
    backup = capture(destination, maximum=4 * 1024**3)
    require((backup['bytes'], backup['sha256']) == (original['bytes'], original['sha256']),
            'database backup differs from original')
    return original, backup, before


def main():
    phase = 'preflight'
    manager, reload_manager, mutated = None, None, False
    for key, value in {'IPFS_ACCEL_SKIP_CORE': '1', 'IPFS_DATASETS_ENABLED': '0',
        'IPFS_ACCEL_AUTO_INSTALL': '0', 'IPFS_KIT_DISABLE': '1', 'STORAGE_FORCE_LOCAL': '1',
        'IPFS_DATASETS_PY_MINIMAL_IMPORTS': '1', 'IPFS_DATASETS_AUTO_INSTALL': '0',
        'IPFS_KIT_AUTO_INSTALL_DEPS': '0', 'ENABLE_IPFS_MODEL_STORAGE': '0',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ACCELERATE), str(DATASETS)]
    prepare_pin = capture(OUT / 'registration-preparation.json')
    preparation = json.loads(read(prepare_pin))
    require(preparation['completed'] is True and preparation['model_count'] == 2,
            'reviewed two-state preparation required')
    review_pin = capture(OUT / 'review/registration-driver-review.json')
    review = json.loads(read(review_pin))
    require(review['approved'] is True, 'independent final driver review required')
    require(review['reviewed_driver_pin'] == capture(Path(__file__).resolve())
            and review['reviewed_builder_pin'] == capture(OUT / 'build_registration_plan.py')
            and review['reviewed_custody_pin'] == capture(OUT / 'custody.py')
            and review['reviewed_preparation_pin'] == prepare_pin, 'reviewed source/preparation bytes differ')
    owners = [source(DATASETS, RELATIVE)]
    owner_paths = subprocess.check_output(['git', '-C', str(ACCELERATE), 'ls-files',
        'ipfs_accelerate_py/model_catalog/*.py', 'ipfs_accelerate_py/model_catalog/**/*.py'], text=True).splitlines()
    owners.extend(source(ACCELERATE, relative) for relative in ['ipfs_accelerate_py/model_manager.py', *owner_paths])
    require(owners[0] == preparation['importer_owner'], 'prepared native importer owner changed')
    spec = importlib.util.spec_from_file_location('registration_native_importer', DATASETS / RELATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    plan_pin = preparation['import_plan_pin']
    releases = [preparation['publication_receipt_pin']]
    plan, _, _, _ = native._prepare(plan_pin, releases, native.MAX_REFERENCE_BYTES)
    survey = json.loads(read(preparation['contextual_custody_survey_pin']))
    require(survey['file_pins_before'] == survey['file_pins_after'] and survey['bound_files_unchanged'] is True,
            'completed unchanged original contextual custody required')
    publication = json.loads(read(preparation['publication_receipt_pin']))
    asset_pins = survey['file_pins_before'] + [prepare_pin, preparation['contextual_custody_survey_pin'],
        publication['historical_append_publication_pin'],
        *(row['fresh_download_pin'] for row in publication['fresh_remote_verifications'])]
    for pin in asset_pins:
        require(capture(pin['path']) == pin, 'original contextual input custody changed before import')
    require(not Path(str(STORE) + '.wal').exists(), 'active WAL requires a separate coherent native snapshot')
    import duckdb
    connection = duckdb.connect(str(STORE), read_only=True)
    try:
        baseline = decode(connection)
        baseline_schema = schema_snapshot(connection)
    finally:
        connection.close()
    baseline_by_id = {row['model_id']: row for row in baseline}
    expected = {record['model_metadata']['model_id'] for record in plan['models']}
    require(not (expected & set(baseline_by_id)), 'planned states already present; review explicit idempotent path instead')
    before, backup, before_stat = backup_database()
    baseline_pin = write(OUT / 'model-manager-before-records.json', baseline)
    before_pin = write(OUT / 'model-manager-before.json', {'selected_storage': str(STORE),
        'database_pin': before, 'backup_pin': backup, 'baseline_records_pin': baseline_pin,
        'baseline_model_count': len(baseline), 'source_owners': owners,
        'expected_new_model_ids': sorted(expected), 'baseline_schema': baseline_schema,
        'in_memory_instance_save_population_restricted_to_two_target_ids': True,
        'original_checkpoint_corpus_vector_codec_files_must_remain_unchanged': True})
    refused_optional_imports, refused_execution_events = [], []
    optional_modules = {'ipfs_accelerate_py.common.storage_wrapper', 'ipfs_accelerate_py.ipfs_kit_integration',
        'ipfs_accelerate_py.datasets_integration', 'ipfs_accelerate_py.model_manager_graphrag',
        'torch', 'transformers', 'sentence_transformers'}
    allowed_git = {('git', '-C', str(root), 'rev-parse', 'HEAD') for root in (DATASETS, ACCELERATE)}
    allowed_git.update(('git', '-C', str(DATASETS if index == 0 else ACCELERATE),
                        'show', owner['head'] + ':' + owner['relative_path'])
                       for index, owner in enumerate(owners))

    class OptionalFence(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + '.') for name in optional_modules):
                refused_optional_imports.append(fullname)
                raise ImportError('isolated metadata registration disables optional owner: ' + fullname)
            return None

    def execution_fence(event, args):
        if event in {'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'os.system', 'os.exec'}:
            refused_execution_events.append(event)
            raise RuntimeError('registration forbids provider/network/extra-process execution: ' + event)
        if event == 'subprocess.Popen':
            require(args[0] == 'git' and tuple(args[1]) in allowed_git,
                    'only exact reviewed read-only source Git checks allowed')

    require(not any(name in sys.modules for name in optional_modules), 'optional owner unexpectedly imported before fence')
    sys.meta_path.insert(0, OptionalFence())
    sys.addaudithook(execution_fence)
    # The genuine source's optional ImportError handlers refer to logger before
    # its normal assignment. Seed only that stdlib name in the real module
    # object; execute the exact committed source, which later sets its logger.
    manager_spec = importlib.util.spec_from_file_location('ipfs_accelerate_py.model_manager',
        ACCELERATE / 'ipfs_accelerate_py/model_manager.py')
    manager_module = importlib.util.module_from_spec(manager_spec)
    manager_module.logger = logging.getLogger('ipfs_accelerate_model_manager')
    sys.modules[manager_spec.name] = manager_module
    manager_spec.loader.exec_module(manager_module)
    ModelManager = manager_module.ModelManager
    require(Path(sys.modules[ModelManager.__module__].__file__).resolve() == ACCELERATE / 'ipfs_accelerate_py/model_manager.py',
            'genuine manager imported from wrong source tree')
    queries = []

    def open_manager():
        return ModelManager(storage_path=str(STORE), use_database=True, enable_ipfs=False,
                            project_legacy_models=True, usage_service=None)

    def readback(model_id):
        reopened = duckdb.connect(str(STORE))
        try:
            result = decode(reopened, model_id)
        finally:
            reopened.close()
        queries.append({'model_id': model_id, 'fresh_native_connection': True,
            'manager_models_not_used': True, 'shared_native_engine_cache_possible': True})
        return result

    def check_records(actual):
        by_id = {row['model_id']: row for row in actual}
        require(set(by_id) == set(baseline_by_id) | expected, 'unexpected persisted population change')
        for key, row in baseline_by_id.items():
            require(by_id[key] == row, 'unrelated persisted metadata/activity changed: ' + key)
        for record in plan['models']:
            require(native._matches(by_id[record['model_metadata']['model_id']], record['model_metadata']),
                    'new persisted contextual metadata differs')
        return by_id

    try:
        require(witness(STORE.lstat()) == witness(before_stat)
                and not Path(str(STORE) + '.wal').exists(), 'database changed before genuine construction')
        phase = 'genuine_manager_construction'
        write(OUT / 'model-manager-construction-started.json', {'before_receipt_pin': before_pin,
            'independent_review_pin': review_pin, 'genuine_manager_source': owners[1]})
        mutated = True
        manager = open_manager()
        require(manager.use_database is True and str(manager.storage_path) == str(STORE), 'genuine DuckDB store differs')
        require(set(manager.models) == set(baseline_by_id), 'genuine initial population differs')
        for key, row in baseline_by_id.items():
            require(native._matches(manager.models[key], row), 'genuine loaded metadata differs before API registration')
        require(all(getattr(manager, name) is None for name in
            ('_ipfs_backend', '_datasets_manager', '_filesystem_handler', '_provenance_logger',
             '_artifact_storage', '_storage_wrapper', '_knowledge_graph')), 'optional registration side-effect owner active')
        require(schema_snapshot(manager.con) == baseline_schema, 'constructor changed selected existing schema')
        # This instance is owned by this driver. Removing entries from its
        # in-memory dictionary does not call remove_model and does not delete
        # persisted records. Existing save methods INSERT OR REPLACE only the
        # entries remaining here; genuine add_model/factory methods stay intact.
        with manager._model_lock:
            manager.models = {key: value for key, value in manager.models.items() if key in expected}
        phase = 'genuine_api_registration'
        result = native.import_ir_model_manager_records(plan_pin, release_receipts=releases,
                                                        manager=manager, readback=readback)
        write(OUT / 'model-manager-api-import-result.json', result)
        require(result['registered_count'] == 2 and result['persisted_metadata_matched_count'] == 2,
                'two original states not registered and verified')
        require(all(manager.get_model(key).serving_config is None for key in expected), 'serving config introduced')
        active_catalog = catalog_targets(manager, expected)
        phase = 'genuine_close_then_cold_native_readback'
        manager.close()
        manager = None
        cold = duckdb.connect(str(STORE), read_only=True)
        try:
            rows = decode(cold)
            require(schema_snapshot(cold) == baseline_schema, 'schema changed after registration')
        finally:
            cold.close()
        check_records(rows)
        phase = 'cold_genuine_manager_reload'
        reload_manager = open_manager()
        require(set(reload_manager.models) == set(baseline_by_id) | expected, 'cold genuine population differs')
        for record in plan['models']:
            require(native._matches(reload_manager.get_model(record['model_metadata']['model_id']), record['model_metadata']),
                    'cold genuine API readback differs')
        cold_catalog = catalog_targets(reload_manager, expected)
        with reload_manager._model_lock:
            reload_manager.models = {key: value for key, value in reload_manager.models.items() if key in expected}
        reload_manager.close()
        reload_manager = None
        phase = 'final_native_readback_after_both_genuine_closes'
        cold = duckdb.connect(str(STORE), read_only=True)
        try:
            final_rows = decode(cold)
            require(schema_snapshot(cold) == baseline_schema, 'schema changed after cold genuine reload')
        finally:
            cold.close()
        check_records(final_rows)
        require(not Path(str(STORE) + '.wal').exists(), 'WAL remains after controlled close')
        for pin in asset_pins:
            require(capture(pin['path']) == pin, 'original contextual asset changed during registration')
        require(owners == [source(DATASETS, RELATIVE)]
                + [source(ACCELERATE, owner['relative_path']) for owner in owners[1:]], 'source owners changed')
        native._prepare(plan_pin, releases, native.MAX_REFERENCE_BYTES)
        final_records_pin = write(OUT / 'model-manager-after-records.json', final_rows)
        summary = {'schema': 'original-contextual-state-model-manager-registration/v1', 'completed': True,
            'selected_storage': str(STORE), 'before_receipt_pin': before_pin, 'preparation_pin': prepare_pin,
            'independent_review_pin': review_pin, 'after_database_pin': capture(STORE, maximum=4 * 1024**3),
            'after_records_pin': final_records_pin, 'before_count': len(baseline), 'after_count': len(final_rows),
            'new_checkpoint_count': 2, 'new_model_ids': sorted(expected), 'native_api_result': result,
            'persisted_native_queries': queries, 'cold_native_readback_verified': True,
            'cold_genuine_manager_reload_verified': True, 'active_manager_catalog': active_catalog,
            'cold_manager_catalog': cold_catalog, 'unrelated_stable_metadata_preserved': True,
            'unrelated_activity_timestamps_preserved': True, 'unrelated_persisted_records_exactly_equal': True,
            'schema_and_indexes_unchanged': True, 'genuine_instance_save_population_restricted_to_two_target_ids': True,
            'active_instance_catalog_scope': 'two registered candidates only; cold reloaded catalog initially includes the entire persisted population',
            'optional_owner_import_refusals': refused_optional_imports,
            'network_or_execution_refusals': refused_execution_events,
            'optional_provenance_storage_graphrag_owners_disabled': True,
            'genuine_source_optional_import_logger_bootstrap': 'stdlib logger seeded before executing the exact committed module; actual source and all API methods unchanged',
            'protected_original_file_count': len(asset_pins),
            'original_checkpoint_and_cache_bytes_unchanged': True,
            'native_schema_profile_format_unknown_preserved': True, 'current_external_service_process_refreshed': False,
            'source_scope': 'Selected genuine API and model-catalog Git objects pinned before/after; not a guarded full eager-import closure.',
            'cooperative_endpoint_scope': True, 'runtime_ready': False, 'teacher_qualified': False,
            'proof_authority': False, 'model_loaded_for_inference': False, 'training_executed': False,
            'new_embeddings_generated': False, 'new_huggingface_upload': False}
        summary_pin = write(OUT / 'model-manager-registration.json', summary)
        print(json.dumps({'result_pin': summary_pin, 'registered_count': 2,
                          'before_count': len(baseline), 'after_count': len(final_rows)}))
    except BaseException as error:
        write(OUT / 'model-manager-driver-failure.json', {'phase': phase,
            'exception_type': type(error).__name__, 'message': str(error),
            'database_may_have_changed': mutated, 'no_automatic_rollback_or_retry': True})
        raise
    finally:
        if manager is not None:
            if manager.use_database is True:
                with manager._model_lock:
                    manager.models = {key: value for key, value in manager.models.items() if key in expected}
                manager.close()
        if reload_manager is not None:
            if reload_manager.use_database is True:
                with reload_manager._model_lock:
                    reload_manager.models = {key: value for key, value in reload_manager.models.items() if key in expected}
                reload_manager.close()


if __name__ == '__main__':
    main()
