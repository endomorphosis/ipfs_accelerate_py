"""External diagnostic wrappers; never import this as a production owner."""
import functools as _diag_functools
import json as _diag_json
import math as _diag_math
import pathlib as _diag_pathlib
import sys as _diag_sys
import time as _diag_time

_DIAG_MAX_EVENTS = 4096
_DIAG_MAX_BYTES = 1024 * 1024
_DIAG_EVENTS_PATH = '/opt/ipfs-supervisor/state/source384-phase-events.jsonl'
_DIAG_SUMMARY_PATH = '/opt/ipfs-supervisor/state/source384-phase-summary.json'
_diag_state = None
_DIAG_COLD_TARGETS = (
    ('ipfs_datasets_py.logic.software_contracts.semantic_index.scanner', 'RepositoryScanner', 'scan_snapshot', 'scanner.scan_snapshot'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ingest', 'DuckDBASTIngestor', 'ingest_snapshot', 'ingestor.ingest_snapshot'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ingest', 'DuckDBASTIngestor', '_ingest_sources', 'ingestor._ingest_sources'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ingest', 'DuckDBASTIngestor', '_parse_or_reuse', 'ingestor._parse_or_reuse'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ingest', 'CountingFrontend', 'extract_from_source', 'frontend.extract_from_source'),
    ('ipfs_datasets_py.logic.software_contracts.cache', 'ImmutableCAS', 'put', 'cas.put'),
    ('ipfs_datasets_py.logic.software_contracts.cache', 'ImmutableCAS', 'put_bytes', 'cas.put_bytes'),
    ('ipfs_datasets_py.logic.software_contracts.cache', 'ImmutableCAS', 'get', 'cas.get'),
    ('ipfs_datasets_py.logic.software_contracts.cache', 'ImmutableCAS', 'get_bytes', 'cas.get_bytes'),
    ('ipfs_datasets_py.logic.software_contracts.cache', 'ImmutableCAS', '_publish', 'cas._publish'),
    ('ipfs_datasets_py.duckdb_control.codebase_catalog', 'CodebaseCatalog', 'publish', 'catalog.publish'),
    ('ipfs_datasets_py.duckdb_control.codebase_catalog', 'CodebaseCatalog', '_validate_candidate', 'catalog._validate_candidate'),
    ('ipfs_datasets_py.duckdb_control.codebase_catalog', 'CodebaseCatalog', '_read_artifact', 'catalog._read_artifact'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ast_store', 'DuckDBASTStore', 'apply_batch', 'ast_store.apply_batch'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ast_store', 'DuckDBASTStore', '_validate_projection', 'ast_store._validate_projection'),
    ('ipfs_datasets_py.logic.software_contracts.duckdb_ast_store', 'DuckDBASTStore', '_persist_projection', 'ast_store._persist_projection'),
)


def _diag_emit(value):
    state = _diag_state
    raw = (_diag_json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if state['events'] >= _DIAG_MAX_EVENTS or state['bytes'] + len(raw) > _DIAG_MAX_BYTES:
        state['dropped'] += 1
        return
    state['stream'].write(raw.decode())
    state['events'] += 1
    state['bytes'] += len(raw)


def _diag_snapshot(*, completed=False):
    # These modules are already loaded by the production probe's _pins().
    index = _diag_sys.modules['ipfs_datasets_py.logic.software_contracts.codebase_ir']
    content = _diag_sys.modules['ipfs_datasets_py.logic.software_contracts.content']
    key = index._manifest_producer_key()
    with index._MANIFEST_MEMO_LOCK:
        stats = dict(index._MANIFEST_MEMO_STATS)
        assert set(stats) == {'hits', 'misses', 'evictions', 'bypasses'}
        assert all(type(value) is int and value >= 0 for value in stats.values())
        stats.update(entries=len(index._MANIFEST_MEMO), retained_bytes=index._MANIFEST_MEMO_SIZE)
    layout_info=content._native_registry_layout.cache_info()._asdict()
    layout_recognized=None
    # Never initialize the lazy native getter for the diagnostic. Only inspect
    # its cached result after context ended and only if production populated it.
    if completed and layout_info['currsize']:
        layout_recognized=content._native_registry_layout() is not None
    return dict(manifest_producer_recognized=key is not None,
        manifest_producer_unchanged=key == _diag_state['producer'],
        manifest_memo=stats,
        cid_native_layout_cache=layout_info,
        cid_preexisting_cached_layout_recognized=layout_recognized,
        cid_layout_recognition_is_not_live_fast_path_proof=True,
        cid_encode_memo=content._memo_encode_digest.cache_info()._asdict(),
        cid_validate_memo=content._memo_validate_cid.cache_info()._asdict())


def _diag_install_units():
    state = _diag_state
    if state['units_installed']:
        return
    # Do not import ahead of the native 90-second timer. The production consumer
    # imports source_units before its first prepare_current invocation.
    module = _diag_sys.modules.get('ipfs_datasets_py.logic.software_contracts.codebase_source_units_384')
    if module is None:
        raise RuntimeError('diagnostic source-unit owner was not imported by production')
    _diag_wrap(module, '_context', 'source_units._context')
    _diag_wrap(module, '_worker', 'source_units._worker')
    state['units_installed'] = True
    state['after_units_install'] = _diag_snapshot()
    assert state['after_units_install']['manifest_producer_recognized']
    assert state['after_units_install']['manifest_producer_unchanged']
    _diag_emit(dict(event='units_wrapped', at_seconds=_diag_time.monotonic()-state['started']))


def _diag_call(original, label, args, kwargs):
    state = _diag_state
    state['calls'] += 1
    call_id = state['calls']
    entered = _diag_time.monotonic()
    parent = state['stack'][-1] if state['stack'] else None
    record = dict(event='enter', stage=label, call_id=call_id, parent_call_id=parent,
        depth=len(state['stack']), at_seconds=entered-state['started'])
    if label == 'source_units._worker':
        record['timeout_scope'] = 'worker_entry_before_child_admission_and_subprocess_allocation'
    # Never capture argument contents, repository paths, sources or formulas.
    for key in ('timeout', 'timeout_seconds', 'memory_mb'):
        value = kwargs.get(key)
        if type(value) in (int, float) and _diag_math.isfinite(value):
            record[key] = value
    _diag_emit(record)
    state['stack'].append(call_id)
    error_type = None
    try:
        return original(*args, **kwargs)
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        elapsed = _diag_time.monotonic()-entered
        assert state['stack'].pop() == call_id
        _diag_emit(dict(event='exit', stage=label, call_id=call_id,
            parent_call_id=parent, seconds=elapsed, error_type=error_type,
            at_seconds=_diag_time.monotonic()-state['started']))


def _diag_callback(original, label):
    @_diag_functools.wraps(original)
    def timed(*args, **kwargs):
        return _diag_call(original, label, args, kwargs)
    return timed


def _diag_wrap(owner, name, label, *, install_units=False):
    descriptor = owner.__dict__[name]
    descriptor_type = type(descriptor) if isinstance(descriptor, (staticmethod, classmethod)) else None
    original = descriptor.__func__ if descriptor_type is not None else descriptor
    if not callable(original):
        raise TypeError('diagnostic target is not callable')

    @_diag_functools.wraps(original)
    def timed(*args, **kwargs):
        if install_units:
            _diag_install_units()
        if label == 'ingestor.ingest_snapshot':
            # These per-call wrappers expire with the call; no callback or
            # publication argument is stored in global diagnostic state.
            kwargs = dict(kwargs)
            for name, callback_label in (('before_publish', 'ingestor.seal_manifest'),
                    ('publish_batch', 'ingestor.publication_callback')):
                callback = kwargs.get(name)
                if callable(callback):
                    kwargs[name] = _diag_callback(callback, callback_label)
        return _diag_call(original, label, args, kwargs)

    setattr(owner, name, descriptor_type(timed) if descriptor_type is not None else timed)
    _diag_state['restore'].append((owner, name, descriptor))


def _diag_install():
    global _diag_state
    assert _diag_state is None
    index = _diag_sys.modules['ipfs_datasets_py.logic.software_contracts.codebase_ir']
    store = _diag_sys.modules['ipfs_datasets_py.logic.software_contracts.duckdb_ast_store']
    producer = index._manifest_producer_key()
    assert producer is not None, 'native producer guard unavailable before diagnostic wrappers'
    stream = _diag_pathlib.Path(_DIAG_EVENTS_PATH).open('x', buffering=1)
    _diag_state = dict(started=_diag_time.monotonic(), producer=producer,
        stream=stream, events=0, bytes=0, dropped=0, calls=0, stack=[],
        restore=[], units_installed=False)
    _diag_state['before_wrappers'] = _diag_snapshot()
    _diag_wrap(index.RepositoryCodebaseIndex, 'prepare_current', 'index.prepare_current', install_units=True)
    _diag_wrap(index.RepositoryCodebaseIndex, 'observe_current', 'index.observe_current')
    _diag_wrap(store.DuckDBASTStore, '_rebuild', 'ast_store._rebuild')
    _diag_wrap(store.DuckDBASTStore, '_load', 'ast_store._load')
    for module_name, class_name, method, label in _DIAG_COLD_TARGETS:
        # The unmodified probe's existing _pins() already imported these owners.
        # Resolve only existing modules so instrumentation cannot hoist imports.
        module = _diag_sys.modules[module_name]
        _diag_wrap(getattr(module, class_name), method, label)
    _diag_state['after_wrappers'] = _diag_snapshot()
    assert _diag_state['after_wrappers']['manifest_producer_recognized']
    assert _diag_state['after_wrappers']['manifest_producer_unchanged']
    _diag_emit(dict(event='installed', at_seconds=_diag_time.monotonic()-_diag_state['started']))


def _diag_finish():
    state = _diag_state
    if state is None:
        return
    try:
        after = _diag_snapshot(completed=True)
        summary = dict(schema='source384-stage-diagnostic@1', diagnostic_only=True,
            production_qualification_claimed=False,
            scope='In-memory method timing wrappers; production source files and timeout arguments unchanged.',
            before_wrappers=state.get('before_wrappers'), after_wrappers=state.get('after_wrappers'),
            after_units_install=state.get('after_units_install'), after_context=after,
            events=state['events'], event_bytes=state['bytes'], dropped_events=state['dropped'],
            calls=state['calls'], stack_empty=not state['stack'],
            max_events=_DIAG_MAX_EVENTS, max_event_bytes=_DIAG_MAX_BYTES,
            seconds=_diag_time.monotonic()-state['started'],
            source_unit_wrappers_installed=state['units_installed'])
        summary['additional_cold_method_targets'] = [row[3] for row in _DIAG_COLD_TARGETS]
        raw=_diag_json.dumps(summary,sort_keys=True,allow_nan=False).encode()
        if len(raw)>32768:
            raise ValueError('diagnostic summary exceeded bound')
        with _diag_pathlib.Path(_DIAG_SUMMARY_PATH).open('xb') as stream:
            stream.write(raw)
    finally:
        try:
            state['stream'].close()
        finally:
            for owner, name, descriptor in reversed(state['restore']):
                setattr(owner, name, descriptor)
