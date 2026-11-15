"""Focused test SOURCE for isolated continuation boundaries; NOT EXECUTED.

Temporary fake unit/profile inputs below exercise local control flow only.
They cannot qualify historical custody, an original preflight issuer, Native,
Main identity or the future independent current-health consumer.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


class LocalData:
    """Explicit test double: never represents a genuine unit/owner identity."""
    @staticmethod
    def _number(value):
        return type(value) in (int, float) and math.isfinite(value)

    @staticmethod
    def descriptor(path, **_):
        raw = Path(path).read_bytes()
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    @staticmethod
    def _document(ref):
        return json.loads(Path(ref['path']).read_bytes())


def test_clock_does_not_renew_at_check_or_accept_expired_future_or_nonfinite(monkeypatch):
    monkeypatch.setattr(executor, 'health_owner', lambda: LocalData())
    profile = {'max_age_seconds': 300}
    value = dict(recorded_at=1000, expires_at=1300)
    executor._closed_capacity_clock(value, profile, now=1299.999)
    assert value == dict(recorded_at=1000, expires_at=1300)
    for now in (999, 1300, 1301, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            executor._closed_capacity_clock(value, profile, now=now)
    with pytest.raises(ValueError):
        executor._closed_capacity_clock(dict(recorded_at=1000, expires_at=1301), profile, now=1001)


def test_incident_historical_pins_include_actual200_old37_and_aeb4():
    pins = executor.CLOSED_CAPACITY_INCIDENT_EVIDENCE
    assert pins['failed_stdout_ref'] == (3217, '0d7b1fe831f9b71a3ab2460fe068693830d4fa53d99b5cb6db077b7710e1cc9b')
    assert pins['prior_health_receipt_ref'] == (12498, '2d7591b0f4e3212791b88e958e0db2976d1a1366c4673e52b30c220f8d8482cf')
    assert pins['cleanup_observation_ref'] == (10632, 'aeb4dfb49e6120842061a55db940359643792113437c54adf81a2462e66b152f')


@pytest.fixture
def claim_case(tmp_path, monkeypatch):
    root = tmp_path/'proof'; root.mkdir(mode=0o700)
    lock_path = root/'scientific-executor.lock'; lock_path.write_bytes(b''); lock_path.chmod(0o600)
    profile = dict(proof_root=str(root), receipt_path=str(root/'new-health.json'),
        unit={'argv': [], 'name': 'fake-test-unit.service'})
    grant_path, ack_path = executor._closed_capacity_paths(profile)
    profile['unit']['argv'] = ['python', executor.CLOSED_CAPACITY_FLAG, str(grant_path)]
    value = dict(prior_main_identity={'pid': 101}, prior_health_receipt_ref={'old': '37'},
        global_hold_ref={'old': 'global-held'}, failed_cycle_hold_ref={'old': '200-held'},
        prior_reconciliation_ref={'old': 'catalogue-consumed'}, cleanup_observation_ref={'old': 'aeb4'},
        capacity_preflight_ref={'fresh': 'closed'}, recorded_at=1000, expires_at=1300)
    grant_path.write_text(json.dumps(value)); grant_path.chmod(0o400)
    reference = LocalData.descriptor(grant_path)
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, reference['sha256'])
    monkeypatch.setattr(executor, 'health_owner', lambda: LocalData())
    identity = dict(pid=202, birth_ticks=444, invocation_id='a'*32)
    monkeypatch.setattr(executor, 'require_main_process', lambda *_: identity)
    monkeypatch.setattr(executor, '_validate_closed_capacity_continuation', lambda *_: (200, []))
    monkeypatch.setattr(executor, '_closed_capacity_final_freshness', lambda *_: None)
    return root, profile, reference, ack_path, identity, lock_path


def test_consumption_exclusive_new_Main_bound_and_never_reassigned(claim_case):
    root, profile, reference, ack_path, identity, lock_path = claim_case
    with lock_path.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
            {}, identity, executor_lock=lock) == (200, [])
        original = ack_path.read_bytes(); ack = json.loads(original)
        assert ack['consuming_main_identity'] == identity
        assert ack['spent_cycle_count'] == 200 and ack['next_cycle_number'] == 201
        assert ack['next_health_sequence'] == 1 and ack['generation_number'] == ack['max_generations'] == 2
        assert ack['initial_consecutive_closed_negatives'] == 1 and ack['max_consecutive_closed_negatives'] == 3
        assert ack['state'] == 'claimed_before_first_original_owner'
        assert ack['old_health_republished'] is ack['health_credit_for_failed_cycle'] is False
        # Even a caller bypassing the validator cannot replace the O_EXCL claim.
        with pytest.raises(FileExistsError):
            executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
                {}, identity, executor_lock=lock)
        assert ack_path.read_bytes() == original


def test_partial_claim_stays_spent_instead_of_automatic_reissue(claim_case):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    ack_path.write_bytes(b'{partial-unknown')
    original = ack_path.read_bytes()
    with lock_path.open('rb') as lock:
        with pytest.raises(FileExistsError):
            executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
                {}, identity, executor_lock=lock)
    assert ack_path.read_bytes() == original


def test_wrong_lock_inode_and_wrong_actual_Main_refuse_before_claim(claim_case, tmp_path):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    other = tmp_path/'other.lock'; other.write_bytes(b''); other.chmod(0o600)
    with other.open('rb') as lock:
        with pytest.raises(ValueError, match='executor_lock_not_retained'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='not_real_distinct_Main'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, {'pid': 303}, executor_lock=lock)
    assert not ack_path.exists()


def test_wrong_environment_pin_and_writable_grant_refuse(claim_case, monkeypatch):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, '0'*64)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='explicit_unit_grant'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    assert not ack_path.exists()
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, reference['sha256'])
    Path(reference['path']).chmod(0o600)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='explicit_unit_grant'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    assert not ack_path.exists()


@pytest.fixture
def finite_loop(tmp_path, monkeypatch):
    root = tmp_path/'proof'; root.mkdir(mode=0o700)
    profile = dict(proof_root=str(root), receipt_path=str(root/'new-unissued-health.json'))
    config_ref = {'path': 'fake-config'}; profile_ref = {'path': 'fake-profile'}
    recovery_ref = {'path': 'fake-capacity-grant'}
    config = dict(max_cycles=202, max_proof_bytes=2**30, poll_seconds=1)
    data = {config_ref['path']: config, recovery_ref['path']: {'schema': executor.CLOSED_CAPACITY_SCHEMA}}
    class FakeHealth:
        _path = staticmethod(Path)
        _document = staticmethod(lambda ref: data[ref['path']])
        _profile = staticmethod(lambda _: profile)
    class Stop:
        is_set = staticmethod(lambda: False)
        wait = staticmethod(lambda _: False)
    monkeypatch.setattr(executor, 'health_owner', lambda: FakeHealth())
    monkeypatch.setattr(executor, 'load_configuration', lambda _: (config, profile_ref, profile))
    monkeypatch.setattr(executor, 'require_main_process', lambda *_: {'fake': 'Main'})
    monkeypatch.setattr(executor, 'catalogue_parser_reference', lambda _: {'fake': 'parser'})
    monkeypatch.setattr(executor, 'load_explicit_catalogue_parser', lambda _: object())
    monkeypatch.setattr(executor, 'reconcile_closed_capacity_continuation', lambda *_, **__: (200, []))
    monkeypatch.setattr(executor, 'check_proof_budget', lambda *_, **__: None)
    monkeypatch.setattr(executor, 'observe_native', lambda *_, **__: {'fake': 'native'})
    monkeypatch.setattr(executor, 'select_template', lambda *_: ({}, 'genuine_scoped_idle'))
    holds = []; failures = []
    monkeypatch.setattr(executor, 'persistent_hold', lambda *args: holds.append(args))
    monkeypatch.setattr(executor, '_recovery_failure', lambda *args: failures.append(args))
    return config_ref, recovery_ref, config, Stop(), holds, failures


@pytest.mark.parametrize('errors', [
    (executor.CapacityAdmissionDeferred, executor.CapacityAdmissionDeferred),
    (executor.SourceAdmissionDeferred, executor.CapacityAdmissionDeferred),
    (executor.CatalogueRaceDeferred, executor.SourceAdmissionDeferred),
])
def test_original200_counts_toward_shared_three_negative_bound(finite_loop, monkeypatch, errors):
    config_ref, recovery_ref, _, stop, holds, failures = finite_loop
    attempts = []
    def cycle(*args, **options):
        attempts.append((options['cycle_number'], args[5], options['capacity_continuation_ref']))
        raise errors[len(attempts)-1]('/fake/cycle/negative.json')
    monkeypatch.setattr(executor, 'run_cycle', cycle)
    result = executor.serve(config_ref, stop, recovery_ref=recovery_ref, catalogue_parser_ref={'fake': 'parser'})
    assert result == executor.HOLD_EXIT
    assert attempts == [(201, 1, recovery_ref), (202, 1, recovery_ref)]
    assert holds[-1][-1] == 'closed_negative_consecutive_defer_bound'
    assert failures[-1][1:3] == (202, 1)


def test_only_positive_cycle_advances_new_health_and_resets_negative_counter(finite_loop, monkeypatch):
    config_ref, recovery_ref, config, stop, holds, _ = finite_loop
    config['max_cycles'] = 203
    attempts = []
    def cycle(*args, **options):
        attempts.append((options['cycle_number'], args[5]))
        if options['cycle_number'] == 202:
            raise executor.CapacityAdmissionDeferred('/fake/cycle/negative.json')
    monkeypatch.setattr(executor, 'run_cycle', cycle)
    assert executor.serve(config_ref, stop, recovery_ref=recovery_ref,
        catalogue_parser_ref={'fake': 'parser'}) == executor.HOLD_EXIT
    assert attempts == [(201, 1), (202, 2), (203, 2)]
    assert holds[-1][-1] == 'finite_cycle_budget_exhausted'


# Additive, UNEXECUTED source-generation reproduction controls. These exercise
# the existing metadata parsers against local DATA only. The historical reader
# double used below supplies explicitly modeled preconditions, not old custody.
# No source-generation authority, issuer, manager launch or healthy receipt is
# supplied by these tests. The current continuation remains deliberately closed.
import copy as _source_generation_copy
import sys as _source_generation_sys
from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as _source_generation_health


def _source_generation_put_DATA(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + '\n').encode()
    path.write_bytes(raw)
    path.chmod(0o400)
    return _source_generation_health.descriptor(path)


@pytest.fixture
def source_generation_profile_DATA(tmp_path):
    health = _source_generation_health
    root = tmp_path / 'source-generation-DATA'
    root.mkdir(mode=0o700)
    sources = []
    roles = {}
    for name in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'):
        roles[name] = _source_generation_put_DATA(root / 'source' / (name + '.py'),
            ('# Modeled ordinary source role: ' + name + '\n').encode())
        sources.append(roles[name])
    interpreter = str(Path(_source_generation_sys.executable).resolve())
    sources.append(health.installed_tool_descriptor(interpreter))
    old_pins = []
    for name in ('autoencoder_daemon_resources.py', 'autoencoder_resource_recovery.py',
                 'parented_phase_resources.py', 'run_owned_validation.py'):
        pin = _source_generation_put_DATA(root / 'source' / name, b'# prior generation DATA\n')
        sources.append(pin)
        old_pins.append(pin)
    prior = dict(schema=health.PROFILE_SCHEMA,
        catalogue=dict(board_namespace='local-source-generation-DATA', plan_root_cid='fixture:plan',
            repository_tree_id='fixture:tree', population_sha256='a' * 64,
            database_uuid='fixture:database', store_generation=19),
        unit=dict(name='local-source-generation-DATA.service',
            argv=[interpreter, '-B', roles['main_entrypoint']['path']],
            interpreter=interpreter, working_directory=str(root),
            environment={'PYTHONDONTWRITEBYTECODE': '1'}),
        receipt_path=str(root / 'proof' / 'old-health.json'), proof_root=str(root / 'proof'),
        source_descriptors=sources, task_scope=['fixture:quiet-task'],
        max_age_seconds=300, max_cycle_seconds=300,
        cycle_contract=dict(plan_schema='fixture:unchanged', plan_operation='fixture:quiet'), **roles)
    original_profile_ref = _source_generation_put_DATA(root / 'original-profile.json', prior)
    original_profile_bytes = Path(original_profile_ref['path']).read_bytes()
    refreshed = _source_generation_copy.deepcopy(prior)
    for pin in old_pins:
        path = Path(pin['path']); path.chmod(0o600)
        new_pin = _source_generation_put_DATA(path, b'# current generation DATA\n')
        index = next(i for i, selected in enumerate(refreshed['source_descriptors'])
                     if selected['path'] == pin['path'])
        refreshed['source_descriptors'][index] = new_pin
    refreshed['receipt_path'] = str(root / 'proof' / 'new-UNISSUED-health.json')
    current_ref = _source_generation_put_DATA(root / 'current-profile-DATA.json', refreshed)
    return dict(root=root, prior=prior, current=refreshed, old_pins=old_pins,
        original_profile_ref=original_profile_ref, original_profile_bytes=original_profile_bytes,
        current_ref=current_ref)


def test_source_generation_finite_refreshed_DATA_parses_but_original_bytes_stay_historical(source_generation_profile_DATA):
    case = source_generation_profile_DATA
    with pytest.raises(ValueError, match='descriptor_changed'):
        _source_generation_health._profile(case['original_profile_ref'])
    assert _source_generation_health._profile(case['current_ref']) == case['current']
    assert Path(case['original_profile_ref']['path']).read_bytes() == case['original_profile_bytes']
    for field in ('task_scope', 'catalogue', 'max_age_seconds', 'max_cycle_seconds',
                  'unit', 'cycle_contract', 'main_entrypoint', 'health_module', 'owner_source', 'driver_source'):
        assert case['prior'][field] == case['current'][field]
    assert not Path(case['current']['receipt_path']).exists()


@pytest.mark.parametrize('source_index', range(4))
def test_source_generation_each_retained_stale_pin_refuses_actual_profile_parser(source_generation_profile_DATA, source_index):
    case = source_generation_profile_DATA
    profile = _source_generation_copy.deepcopy(case['current'])
    old = case['old_pins'][source_index]
    index = next(i for i, row in enumerate(profile['source_descriptors']) if row['path'] == old['path'])
    profile['source_descriptors'][index] = old
    reference = _source_generation_put_DATA(case['root'] / 'stale-one-DATA.json', profile)
    with pytest.raises(ValueError, match='descriptor_changed'):
        _source_generation_health._profile(reference)
    assert not Path(profile['receipt_path']).exists()


@pytest.mark.parametrize('old_duplicate', [False, True])
def test_source_generation_duplicate_current_or_old_paths_never_bypass_actual_source_parser(source_generation_profile_DATA, old_duplicate):
    case = source_generation_profile_DATA
    profile = _source_generation_copy.deepcopy(case['current'])
    pin = (case['old_pins'][0] if old_duplicate else
           next(row for row in profile['source_descriptors'] if row['path'] == case['old_pins'][0]['path']))
    profile['source_descriptors'].append(pin)
    reference = _source_generation_put_DATA(case['root'] / 'duplicate-DATA.json', profile)
    with pytest.raises(ValueError, match='descriptor_changed|duplicate_source'):
        _source_generation_health._profile(reference)


@pytest.fixture
def source_generation_legacy_history_DATA(source_generation_profile_DATA, monkeypatch):
    case = source_generation_profile_DATA
    health = _source_generation_health
    current = case['current']; prior = case['prior']
    value = dict.fromkeys(executor.CLOSED_CAPACITY_HISTORY_KEYS)
    value.update(schema=executor.CLOSED_CAPACITY_SCHEMA,
        authorization='continue_distinct_physical_cycle_after_closed_preentry_capacity_refusal',
        profile_ref=case['current_ref'], spent_cycle_count=200, prior_last_health_sequence=37,
        generation_number=2, max_generations=2)
    for field, (size, digest) in executor.CLOSED_CAPACITY_INCIDENT_EVIDENCE.items():
        value[field] = dict(path='/explicit-modeled-historical-DATA/' + field,
                            bytes=size, sha256=digest)
    value['prior_profile_snapshot_ref'] = {**value['prior_profile_ref'],
        'path': '/explicit-modeled-historical-DATA/prior-profile-snapshot'}
    documents = {value['prior_profile_snapshot_ref']['path']: prior,
                 value['prior_config_snapshot_ref']['path']: {'execution_templates': []}}
    historical_refs = list(value[field] for field in executor.CLOSED_CAPACITY_INCIDENT_EVIDENCE)
    class ModeledHistoricalReader:
        # Only the current local profile uses the genuine source parser.
        # These few historical metadata preconditions are explicit unit DATA.
        _keys = staticmethod(health._keys)
        _profile = staticmethod(health._profile)
        @staticmethod
        def _document(ref):
            return documents[ref['path']]
        @staticmethod
        def _bound(ref):
            assert ref in historical_refs
            return b'explicit modeled historical precondition, not original custody'
    def forbidden(*_, **__):
        raise AssertionError('source-generation refusal must precede original custody/census/action')
    monkeypatch.setattr(executor, 'health_owner', lambda: ModeledHistoricalReader())
    monkeypatch.setattr(executor, '_closed_capacity_negative', forbidden)
    monkeypatch.setattr(executor, 'catalogue_race_source_inventory', forbidden)
    monkeypatch.setattr(executor, 'check_proof_budget', forbidden)
    return value, current, case['current_ref']


def test_source_generation_existing_continuation_still_refuses_finite_source_replacement(source_generation_legacy_history_DATA):
    value, profile, reference = source_generation_legacy_history_DATA
    with pytest.raises(ValueError, match='closed_capacity_source_removed_or_unit_authority_changed'):
        executor._closed_capacity_history(value, reference, profile, {})
    assert value['spent_cycle_count'] == 200
    assert value['generation_number'] == value['max_generations'] == 2
    assert value['prior_last_health_sequence'] == 37
    assert not Path(profile['receipt_path']).exists()


@pytest.mark.parametrize('key,replacement', [
    ('spent_cycle_count', 199), ('spent_cycle_count', 201),
    ('spent_cycle_count', 200.0), ('generation_number', 3),
])
def test_source_generation_existing_counters_cannot_be_reset_to_make_replacement_eligible(source_generation_legacy_history_DATA, key, replacement):
    value, profile, reference = source_generation_legacy_history_DATA
    value[key] = replacement
    with pytest.raises(ValueError, match='closed_capacity_counter_or_health_epoch_reset'):
        executor._closed_capacity_history(value, reference, profile, {})
    assert not Path(profile['receipt_path']).exists()


@pytest.mark.parametrize('unknown', ['source_generation_authorization_ref', 'current_storage_limit_bytes', 'allow_source_refresh'])
def test_source_generation_unissued_extra_field_is_not_authority_in_legacy_schema(monkeypatch, unknown):
    class ShapeOnly:
        _keys = staticmethod(_source_generation_health._keys)
        @staticmethod
        def _profile(*_):
            raise AssertionError('unknown authority fields must refuse before any profile read')
    monkeypatch.setattr(executor, 'health_owner', lambda: ShapeOnly())
    value = dict.fromkeys(executor.CLOSED_CAPACITY_HISTORY_KEYS)
    value[unknown] = None
    with pytest.raises(ValueError, match='closed_capacity_history_shape'):
        executor._closed_capacity_history(value, {}, {}, {})


@pytest.mark.parametrize('live_cap,accepted', [(220_000_000_000, True), (250_000_000_000, False)])
def test_source_generation_current250_preflight_cannot_silently_borrow_old220_policy(tmp_path, monkeypatch, live_cap, accepted):
    health = _source_generation_health
    resource = _source_generation_put_DATA(tmp_path / 'autoencoder_daemon_resources.py',
        b'# Local resource source DATA; no owner or grants\n')
    ledger_ref = dict(path='/modeled-local-ledger-DATA', bytes=1, sha256='e' * 64)
    preflight = dict(schema='original-owner-closed-capacity-preflight/v1',
        checked_at=executor.time.time(), max_age_seconds=60, ledger_ref=ledger_ref,
        accounting=dict(observed_apparent_bytes=0, outstanding_full_reservations_bytes=0,
            additional_requested_bytes=executor.REUSE_CYCLE_RESERVE_BYTES,
            charged_bytes=executor.REUSE_CYCLE_RESERVE_BYTES, limit_bytes=live_cap,
            minimum_filesystem_free_bytes=executor.REUSE_CYCLE_RESERVE_BYTES,
            scope='named_roots_only_apparent_bytes'), roots=[], resource_source_ref=resource,
        source_READ_closed=True, ledger_lock_closed=True, reservation_created=False,
        native_grant_created=False, admission_authority=False)
    reference = _source_generation_put_DATA(tmp_path / 'closed-preflight-DATA.json', preflight)
    monkeypatch.setattr(executor, 'health_owner', lambda: health)
    args = ({'capacity_preflight_ref': reference}, {'source_descriptors': [resource]},
            {'source_sha256': {resource['path']: resource['sha256']}}, ledger_ref,
            {'roots': [], 'reservations': {}})
    if accepted:
        result = executor._closed_capacity_preflight(*args)
        assert result['admission_authority'] is result['native_grant_created'] is result['reservation_created'] is False
    else:
        with pytest.raises(ValueError, match='closed_capacity_preflight_cap_not_restored'):
            executor._closed_capacity_preflight(*args)
    assert executor.GLOBAL_RETAINED_BYTES == 220_000_000_000


# Full positive source-version contract below uses genuine local-file readers.
# Every original closure receipt is explicitly FICTIONAL unit DATA. No unit,
# Source lease, Native grant, original process or model/legal result is claimed.
import base64 as _source_generation_base64
import stat as _source_generation_stat


def _source_generation_canonical_DATA(path, value):
    raw = value if type(value) is bytes else executor._source_generation_canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw); path.chmod(0o400)
    return _source_generation_health.descriptor(path)


def _source_generation_replace_DATA(path, value):
    path.chmod(0o600)
    return _source_generation_canonical_DATA(path, value)


@pytest.fixture(scope='module')
def immutable_source_generation_SDK_control_DATA(tmp_path_factory):
    """One immutable incoming-source population inside this original attempt.

    Only ordinary byte copies are shared. Policies, SDK witnesses and every
    fictional invocation stay private to their function-scoped case.
    """
    import stat
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    put = _source_generation_canonical_DATA
    base = Path(tmp_path_factory.getbasetemp())
    root = tmp_path_factory.mktemp('source-generation-immutable-SDK', numbered=False)
    assert root.parent == base and root.resolve(strict=True) == root
    assert not any(path.is_symlink() for path in (root, *root.parents))
    source_root = Path(owner.__file__).parents[3]
    sources = []
    for relative in ('ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py'):
        sources.append(put(root / relative, (source_root / relative).read_bytes()))
    sources.append(put(root / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py',
                       Path(engine.__file__).read_bytes()))
    owning = put(root / 'tests/fictional_owning_tests.py',
                 b'def test_source_generation_seven_roles():\n    pass\n')
    manifest = put(root / 'source-manifest.json', dict(schema=owner.CONTROL_SCHEMA, root=str(root),
        files=sorted(sources + [owning], key=lambda pin: pin['path']), namespace_profiles=[]))
    pins = sources + [owning, manifest]
    assert len(pins) == 9 and len({pin['path'] for pin in pins}) == 9
    assert all(root in Path(pin['path']).parents and len(os.fsencode(pin['path'])) <= 512 for pin in pins)
    assert sum(pin['bytes'] for pin in pins) <= 768 * 1024
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_nlink',
              'st_size', 'st_mtime_ns', 'st_ctime_ns')
    def snapshot():
        population = list(root.rglob('*'))
        assert not any(path.is_symlink() for path in population)
        assert {str(path) for path in population if path.is_file()} == {pin['path'] for pin in pins}
        snapshots = []
        for pin in pins:
            path = Path(pin['path'])
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                before = os.fstat(stream.fileno())
                assert stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
                assert stat.S_IMODE(before.st_mode) == 0o400 and before.st_size == pin['bytes']
                raw = stream.read()
                after = os.fstat(stream.fileno())
            identity = tuple(getattr(before, name) for name in fields)
            assert identity == tuple(getattr(after, name) for name in fields)
            assert identity == tuple(getattr(path.lstat(), name) for name in fields)
            assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256']
            snapshots.append(identity)
        return tuple(snapshots)
    original = snapshot()
    # No mutable descriptor object or policy/capability is shared across cases.
    keys = ('path', 'bytes', 'sha256')
    bundle = (str(root), tuple(tuple(pin[key] for key in keys) for pin in sources),
              tuple(owning[key] for key in keys), tuple(manifest[key] for key in keys))
    yield bundle
    assert snapshot() == original  # Verify retained files; never delete them.


@pytest.fixture
def complete_source_generation_DATA(tmp_path, monkeypatch, immutable_source_generation_SDK_control_DATA):
    """Real parsers over fictional immutable files, without custody substitutes."""
    h = _source_generation_health; put = _source_generation_canonical_DATA
    root = tmp_path/'fictional-source-generation'; root.mkdir(mode=0o700)
    datasets = root/'datasets'
    named = (datasets/'workspace/logs', datasets/'workspace/queues', datasets/'docs/evidence', root/'pytest')
    for path in named: path.mkdir(parents=True)
    proof = named[0]/'same-proof'; proof.mkdir(mode=0o700)
    (proof/'scientific-executor.lock').write_bytes(b''); (proof/'scientific-executor.lock').chmod(0o600)
    monkeypatch.setattr(executor, 'ORIGINAL_NAMED_ROOTS', named)
    ledger = named[0]/'ledger.json'; monkeypatch.setattr(executor, 'ORIGINAL_RESOURCE_LEDGER', ledger)
    oldpackage, package = root/'old-package', root/'new-package'
    oldroles = {role: put(oldpackage/'ipfs_accelerate_py/agent_supervisor/rescue'/filename,
        ('# fictional old '+role+'\n').encode()) for role, filename in (
        ('main_entrypoint', 'scientific_executor.py'), ('health_module', 'native_executor_health.py'),
        ('owner_source', 'original_embedded_observation_owner.py'), ('driver_source', 'native_embedded_observation_driver.py'))}
    oldparser = put(oldpackage/'ipfs_accelerate_py/agent_supervisor/rescue/native_catalogue_probe.py', b'# old parser DATA\n')
    relative_sources = ('ipfs_datasets_py/logic/backends/process.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py',
        'scripts/ops/legal_ir/run_owned_validation.py')
    pins = []; captures = []; raw_old = []
    for index, relative in enumerate(relative_sources):
        raw = ('# fictional old R role '+str(index)+'\n').encode()
        pin = put(datasets/relative, raw); pins.append(pin); raw_old.append(raw)
        info = Path(pin['path']).lstat()
        identity = {key: getattr(info, key) for key in ('st_dev','st_ino','st_mode','st_uid','st_gid',
                                                     'st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
        captures.append(dict(source_ref=pin, identity=identity,
                             raw_base64=_source_generation_base64.b64encode(raw).decode()))
    capture_ref = put(root/'original-source-capture-DATA.json',
                     dict(schema='resource-census-canonical-source-snapshot/v1', files=captures))
    original_profile_path = root/'prior-profile.json'; current_profile_path = root/'current-profile.json'
    interpreter = str(Path(_source_generation_sys.executable).resolve())
    installed = h.installed_tool_descriptor(interpreter)
    template_links = dict(execution_request='attempt/execution-request.json', result='attempt/result.json', origins='attempt/origins.json')
    def template(seed, sources):
        plan = dict(schema='fictional-original-plan/v1', operation='unchanged_idle', task_scope=['fixture:quiet-task'],
            native_metadata_cli=dict(source_sha256={pin['path']:pin['sha256'] for pin in sources},
                project_import_plan={'source_pins':sources}, attempt_directory=str(seed/'attempt')))
        plan_ref = put(seed/'plan.json', plan)
        request = dict(schema='legal-ir-original-native-control-request/v1', purpose='native_control_metadata',
            argv=[interpreter,'-B',oldroles['driver_source']['path'],'--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256']],
            source_sha256={**{pin['path']:pin['sha256'] for pin in sources}, plan_ref['path']:plan_ref['sha256']},
            cpu_slots=1,memory_mb=1536,child_process_slots=8,storage_bytes=384*1024**2,timeout_seconds=300,
            resource_ledger=str(ledger), source_roots=[str(datasets), str(oldpackage)],
            attempt_directory=str(seed/'attempt'), request_id='fictional-unissued:'+str(root))
        request_ref = put(seed/'request.json', request)
        return put(seed/'template.json',dict(schema=executor.TEMPLATE_SCHEMA,kind='genuine_scoped_idle',
                    seed_root=str(seed),request_ref=request_ref,plan_ref=plan_ref,links=template_links))
    old_template = template(proof/'old-unissued-template', pins)
    identity = dict(board_namespace='fictional:board', plan_root_cid='fictional:plan', repository_tree_id='fictional:tree',
                    population_sha256='a'*64, database_uuid='fictional:database', store_generation=19)
    catalogue = {key:identity[key] for key in ('board_namespace','plan_root_cid','repository_tree_id','population_sha256')}
    catalogue.update(database_path=str(root/'control.duckdb'),population_path=str(root/'population.json'),
                     board_id='legal-ir-scientific',max_age_seconds=300,timeout_seconds=5)
    def configuration(path, profile_path, idle):
        return put(path,dict(schema=executor.CONFIG_SCHEMA,profile_path=str(profile_path),catalogue=catalogue,
            idle_template_ref=idle,execution_templates=[],poll_seconds=60,max_cycles=10000,max_proof_bytes=1024**3))
    oldconfig = configuration(root/'old-config.json',original_profile_path,old_template)
    oldargv = [interpreter,'-B',oldroles['main_entrypoint']['path'],'serve','--config',oldconfig['path'],
        '--config-sha256',oldconfig['sha256'],'--known-closed-catalogue-race-recovery',str(root/'old-consumed-grant.json'),
        '--catalogue-parser-source',oldparser['path'],'--catalogue-parser-sha256',oldparser['sha256']]
    prior = dict(schema=h.PROFILE_SCHEMA,catalogue=identity,
        unit=dict(name='fictional-source-generation.service',argv=oldargv,interpreter=interpreter,
            working_directory=str(datasets),environment={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(oldpackage)}),
        receipt_path=str(proof/'old-health.json'),proof_root=str(proof),source_descriptors=[*oldroles.values(),oldparser,installed,*pins,oldconfig],
        task_scope=['fixture:quiet-task'],max_age_seconds=300,max_cycle_seconds=300,
        cycle_contract=dict(plan_schema='fictional-original-plan/v1',plan_operation='unchanged_idle'),**oldroles)
    prior_ref=put(original_profile_path,prior); snapshot=put(root/'prior-profile-frozen-snapshot.json',Path(prior_ref['path']).read_bytes())
    after=[]; replacements=[]; witness_rows=[]
    for index,(pin,raw) in enumerate(zip(pins,raw_old)):
        current = b'MAX_STORAGE_BYTES = 250_000_000_000\n' if index==1 else ('# fictional CURRENT R role '+str(index)+'\n').encode()
        if index==2:current=b'def inspect_validation_reservation(): pass\ndef inspect_validation_disposition(): pass\n'
        if index==4:current=b'def inspect_owned_validation_productive_native_closed_v1(): pass\ndef inspect_owned_validation_no_child_native_closed_v5(): pass\n'
        refreshed=_source_generation_replace_DATA(Path(pin['path']),current); after.append(refreshed)
        slot=prior['source_descriptors'].index(pin); replacements.append(dict(slot=slot,before_ref=pin,after_ref=refreshed))
        captured=put(root/'captured-old'/str(index),raw)
        witness_rows.append(dict(source_ref=pin,captured_source_ref=captured,source_capture_ref=capture_ref))
    stage=_source_generation_copy.deepcopy(prior)
    for row in replacements:stage['source_descriptors'][row['slot']]=row['after_ref']
    stage_ref=put(root/'stage-five-current.json',stage)
    preparation=put(root/'preparation-spec.json',dict(schema='agent-supervisor/scientific-idle-source-preparation@1',
        prior_profile_ref=prior_ref,prior_profile_snapshot_ref=snapshot,profile_ref=stage_ref,replacements=replacements))
    witness=put(root/'historical-source-witness.json',dict(schema='agent-supervisor/historical-project-source-witness@1',
        purpose='closed_cycle_and_receipt_evidence_only',profile_ref=prior_ref,profile_snapshot_ref=snapshot,sources=witness_rows))
    packagepins=[]
    for relative in ('ipfs_accelerate_py/__init__.py','ipfs_accelerate_py/agent_supervisor/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py','ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py','ipfs_accelerate_py/agent_supervisor/rescue/native_catalogue_probe.py'):
        packagepins.append(put(package/relative,('# fictional CURRENT package '+relative+'\n').encode()))
    new_template=template(proof/'new-unissued-template',after)
    config_ref=configuration(root/'current-config.json',current_profile_path,new_template)
    profile=_source_generation_copy.deepcopy(stage)
    profile.update(main_entrypoint=packagepins[3],health_module=packagepins[4],receipt_path=str(proof/'new-health-UNISSUED.json'))
    profile['source_descriptors'] += packagepins+[config_ref]
    argv=list(oldargv);argv[2]=packagepins[3]['path'];argv[argv.index('--config')+1]=config_ref['path']
    argv[argv.index('--config-sha256')+1]=config_ref['sha256']
    argv[argv.index('--known-closed-catalogue-race-recovery')]=executor.CLOSED_CAPACITY_FLAG
    argv[argv.index(executor.CLOSED_CAPACITY_FLAG)+1]=str(executor._closed_capacity_paths(profile)[0])
    argv[argv.index('--catalogue-parser-source')+1]=packagepins[5]['path']
    argv[argv.index('--catalogue-parser-sha256')+1]=packagepins[5]['sha256']
    profile['unit']['argv']=argv;profile['unit']['environment']['PYTHONPATH']=str(package)+os.pathsep+str(oldpackage)
    profile_ref=put(current_profile_path,profile)
    policy_evidence=put(root/'FICTIONAL-operator-policy-approval-evidence.json',
        {'explicitly_fictional_unit_DATA':True,'message':'This is not an actual user authorization.'})
    policy=put(root/'policy-approval-DATA.json',dict(schema='agent-supervisor/scientific-current-storage-policy-approval@1',
        authorization='explicit_existing_user_250GB_policy_source_generation_only',limit_bytes=250_000_000_000,
        prior_limit_bytes=220_000_000_000,ledger_path=str(ledger),named_roots=[str(path) for path in named],
        resource_source_ref=after[1],authorization_evidence_ref=policy_evidence,
        execution_budget_created=False,hold_removal_authorized=False))
    transition=put(root/'template-transition.json',dict(schema='agent-supervisor/scientific-idle-source-template-transition@1',
        prior_template_ref=old_template,template_ref=new_template,replacements=replacements,runtime_activation_or_task_authority=False))
    # Build the genuine owned-artifact-reader input layout from fictional DATA.
    subjects=[packagepins[3],packagepins[4],*after]
    allsubjects={pin['path']:pin for pin in subjects+packagepins}
    selected=['FICTIONAL::test_source_generation_joins']
    scope=put(root/'qualification-scope.json',dict(expected_cases=1,expected_phases=3,selected_checks=selected,
        selector_contract={'pytest_selectors':['FICTIONAL owning test']},sources=allsubjects))
    attempt=proof/'fictional-qualification-attempt'; attempt.mkdir(mode=0o700)
    request=dict(schema='owned-validation-request/v1',attempt_directory=str(attempt),declared_external_writable_paths=[],
        source_sha256={path:pin['sha256'] for path,pin in allsubjects.items()},selected_checks=['fictional-selected-check'])
    request.update({f'fictional_unchanged_field_{i}':i for i in range(12)})
    assert len(request)==17
    request_ref=put(root/'qualification-request.json',request)
    plan_ref=put(root/'qualification-plan.json',dict(request=request_ref,scope=scope))
    observation=dict(schema='scientific-executor-focused-observation/v1',passed=True,error=None,guard_denials=[],
        case_count=1,phase_count=3,selected_checks=selected,pytest_selected_checks=['FICTIONAL owning test'],
        JUnit=dict(tests=1,failures=0,errors=0,skipped=0),
        phases=[dict(nodeid=selected[0],when=when,outcome='passed') for when in ('setup','call','teardown')],
        raw_compiled_sources={pin['path']:pin['sha256'] for pin in subjects},executed_raw_paths=[pin['path'] for pin in subjects])
    observation_ref=put(attempt/'exports/focused-qualification-result.json',observation)
    put(attempt/'exports/test-results.xml',b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    result=dict(schema='owned-validation-result/v1',completed=True,returncode=0,request_sha256=request_ref['sha256'],
                executed_checks=request['selected_checks'])
    put(attempt/'logs/result.json',result);put(attempt/'logs/execution.json',result)
    rows=[]
    for path in sorted(attempt.rglob('*')):
        if path.is_file():
            info=path.lstat();pin=h.descriptor(path)
            rows.append(dict(path=str(path.relative_to(attempt)),bytes=pin['bytes'],sha256=pin['sha256'],device=info.st_dev,inode=info.st_ino))
    reservation_id='f'*32
    durability_ref=put(attempt/'logs/validation-durability.json',dict(schema='owned-validation-durability/v1',
        request_sha256=request_ref['sha256'],reservation_id=reservation_id,files=rows))
    final_bytes=sum(row['bytes'] for row in rows)+durability_ref['bytes'];info=attempt.lstat()
    record=dict(status='released',artifacts_durable_asserted=True,attempt_exceeded_reservation=False,
        validation_request_sha256=request_ref['sha256'],external_charges={},final_external_charged_bytes=0,
        attempt_directory=dict(path=str(attempt),device=info.st_dev,inode=info.st_ino),reservation_id=reservation_id,
        final_attempt_bytes=final_bytes,final_total_charged_bytes=final_bytes,storage_bytes=16*1024**2)
    terminal=dict(result=str(attempt/'logs/result.json'),reservation=dict(status='released',record=record,
        resource_lease={'released':True},process_observation=dict(live_processes=0,tracking_error='',out_of_group_live_identities=[])))
    stdout=put(root/'fictional-original-owner.stdout',terminal)
    closed=dict(plan=plan_ref,request=request_ref,returncode=0,failure=None,cleanup_errors=[],received_signals=[],
        owner_terminal=terminal,stdout=stdout)
    for key in ('completed','child_reaped','source_READ_closed','source_custody_closed',
                'original_native_compute_closed','original_CLI_exit_observed','source_pins_unchanged','installed_tools_unchanged'):closed[key]=True
    closed_ref=put(root/'fictional-original-root-closeout.json',closed)
    qualification=put(root/'source-qualification.json',dict(schema='scientific-source-generation-original-qualification/v1',
        original_root_closeout=closed_ref,plan=plan_ref,request=request_ref,scope=scope,observation=observation_ref,
        required_raw_subjects=subjects,source_inventory_subjects=packagepins,runtime_activation_or_task_authority=False))
    inventory=put(root/'source-inventory.json',dict(schema='legal-ir-scientific-source-inventory/v1',source_root=str(package),
        source_commit='a'*40,source_tree='b'*40,file_count=len(packagepins),files=[dict(pin,mode=0o400) for pin in packagepins]))
    operation=put(root/'source-operation-plan.json',{'fictional_source_operation_plan':True})
    commitment=put(root/'source-commitment.json',dict(schema='agent-supervisor/scientific-producer-source-commitment@3',
        source_root=str(package),source_commit='a'*40,source_tree='b'*40,source_inventory=inventory,
        source_operation_plan=operation,source_qualification=qualification,catalogue_parser=packagepins[5],
        main_entrypoint=packagepins[3],health_module=packagepins[4],runtime_activation_or_task_authority=False))
    spec=put(root/'technical-source-spec.json',dict(schema=executor.SOURCE_GENERATION_SPEC_SCHEMA,
        authorization='approve_technical_source_generation_without_execution_budget',profile_ref=profile_ref,
        prior_profile_ref=prior_ref,prior_profile_snapshot_ref=snapshot,preparation_spec_ref=preparation,
        historical_source_witness_ref=witness,source_commitment_ref=commitment,template_transition_ref=transition,
        storage_policy_approval_ref=policy))
    monkeypatch.setattr(executor,'health_owner',lambda:h)
    manager=dict(Id=profile['unit']['name'],MainPID='0',ActiveState='inactive')
    # Manager identity is the ONLY platform boundary modeled for source issue.
    monkeypatch.setattr(h,'_manager',lambda _:dict(manager))
    case = dict(root=root,proof=proof,named=named,profile=profile,profile_ref=profile_ref,
        prior=prior,prior_ref=prior_ref,snapshot=snapshot,config=h._document(config_ref),spec=spec,
        replacements=replacements,policy=policy,qualification=qualification,commitment=commitment,
        source_inventory=inventory,source_subjects=subjects,packagepins=packagepins,
        manager=manager,ledger=ledger,transition=transition,request=h._document(h._document(new_template)['request_ref']))
    return _source_generation_upgrade_current_v5_DATA(case,monkeypatch,immutable_source_generation_SDK_control_DATA)


def test_source_generation_positive_genuine_file_review_and_once_only_technical_issue(complete_source_generation_DATA):
    case=complete_source_generation_DATA;h=_source_generation_health
    # These checks use genuine profile, template, source, provenance, original
    # artifact, durability and JUnit readers over the explicitly fictional files.
    assert executor.load_configuration(next(pin for pin in case['profile']['source_descriptors']
        if pin['path']==case['profile']['unit']['argv'][case['profile']['unit']['argv'].index('--config')+1]))[1]==case['profile_ref']
    output=executor._source_generation_path(case['profile'])
    reference=executor.authorize_closed_capacity_source_generation(case['spec'],output)
    value=h._document(reference)
    assert value['spent_cycle_count']==200 and value['next_cycle_number']==201
    assert value['remaining_cycles']==9800 and value['generation_number']==value['max_generations']==2
    assert value['max_cycles']==10000 and value['max_proof_bytes']==2**30
    assert value['dispatch_authority'] is value['execution_budget_created'] is value['hold_removal_authorized'] is False
    context=executor._source_generation_context(reference,case['profile_ref'],case['profile'],case['config'])
    try:
        assert executor._source_generation_record(context)['policy']['limit_bytes']==250_000_000_000
    finally:executor._close_source_generation_context(context)
    frozen=output.read_bytes()
    with pytest.raises(ValueError,match='issue_slot_spent'):
        executor.authorize_closed_capacity_source_generation(case['spec'],output)
    assert output.read_bytes()==frozen
    assert not any(path.exists() for path in executor._closed_capacity_paths(case['profile']))
    assert not Path(case['profile']['receipt_path']).exists()


@pytest.mark.parametrize('field,replacement',[
    ('source_commitment_ref',None),('historical_source_witness_ref',None),('storage_policy_approval_ref',None),
    ('unknown_refresh_waiver',True)])
def test_source_generation_unknown_or_missing_control_never_issues(complete_source_generation_DATA,field,replacement):
    case=complete_source_generation_DATA;value=_source_generation_health._document(case['spec'])
    if replacement is None:value.pop(field)
    else:value[field]=replacement
    ref=_source_generation_canonical_DATA(case['root']/('bad-spec-'+field+'.json'),value)
    with pytest.raises(ValueError,match='source_generation_spec_shape'):
        executor.authorize_closed_capacity_source_generation(ref,executor._source_generation_path(case['profile']))
    assert not executor._source_generation_path(case['profile']).exists()


@pytest.mark.parametrize('field,replacement',[
    ('limit_bytes',220_000_000_000),('limit_bytes',250_000_000_000.0),('prior_limit_bytes',250_000_000_000),
    ('execution_budget_created',True),('hold_removal_authorized',True),('authorization_evidence_ref',None)])
def test_source_generation_policy_is_explicit_and_not_an_execution_budget(complete_source_generation_DATA,field,replacement):
    case=complete_source_generation_DATA;value=_source_generation_health._document(case['policy']);value[field]=replacement
    ref=_source_generation_canonical_DATA(case['root']/('bad-policy-'+field+'.json'),value)
    with pytest.raises((ValueError,TypeError)):
        executor._source_generation_storage_policy(ref,case['profile'],case['replacements'])
    assert executor.GLOBAL_RETAINED_BYTES==220_000_000_000


@pytest.mark.parametrize('mutation',['unexecuted_main','wrong_raw_R','missing_package_dependency','not_closed_source',
                                     'missing_durable_file','changed_JUnit','bool_cases'])
def test_source_generation_qualification_actual_reader_refuses_wrong_generation_or_unclosed_DATA(complete_source_generation_DATA,mutation):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    value=h._document(case['legacy_qualification']);root=h._document(value['original_root_closeout'])
    if mutation=='unexecuted_main':
        observation=h._document(value['observation']);observation['executed_raw_paths'].remove(case['profile']['main_entrypoint']['path'])
        value['observation']=put(case['root']/'bad-execution-observation.json',observation)
    elif mutation=='wrong_raw_R':value['required_raw_subjects'][-1]=value['required_raw_subjects'][0]
    elif mutation=='missing_package_dependency':value['source_inventory_subjects'].pop()
    elif mutation=='not_closed_source':
        root['source_READ_closed']=False;value['original_root_closeout']=put(case['root']/'bad-open-root.json',root)
    elif mutation=='missing_durable_file':
        attempt=Path(root['owner_terminal']['reservation']['record']['attempt_directory']['path']);(attempt/'exports/extra-unmanifested.txt').write_bytes(b'x')
    elif mutation=='changed_JUnit':
        path=Path(root['owner_terminal']['reservation']['record']['attempt_directory']['path'])/'exports/test-results.xml'
        _source_generation_replace_DATA(path,b'<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0"/></testsuites>')
    else:
        scope=h._document(value['scope']);scope['expected_cases']=True;value['scope']=put(case['root']/'bad-scope.json',scope)
        plan=h._document(value['plan']);plan['scope']=value['scope'];value['plan']=put(case['root']/'bad-plan-with-typed-scope.json',plan)
        root['plan']=value['plan'];value['original_root_closeout']=put(case['root']/'bad-root-with-typed-scope.json',root)
    ref=put(case['root']/('bad-qualification-'+mutation+'.json'),value)
    with pytest.raises(ValueError):
        executor._source_generation_v1_qualification(ref,case['profile'],{pin['path']:pin for pin in case['packagepins']})


@pytest.mark.parametrize('mutation',['scope','sixth_source','environment','budget','dispatch','wrong_config_sha'])
def test_source_generation_coherent_profile_denies_unlisted_scope_changes(complete_source_generation_DATA,mutation):
    case=complete_source_generation_DATA;h=_source_generation_health
    profile=_source_generation_copy.deepcopy(case['profile']);config=_source_generation_copy.deepcopy(case['config'])
    if mutation=='scope':profile['task_scope'].append('fictional:new-task')
    elif mutation=='sixth_source':profile['source_descriptors'].append(_source_generation_canonical_DATA(case['root']/'sixth.py',b'# unlisted\n'))
    elif mutation=='environment':profile['unit']['environment']['OMP_NUM_THREADS']='2'
    elif mutation=='budget':config['max_cycles']=10001
    elif mutation=='dispatch':config['execution_templates']=[dict(task_cid='fixture:quiet-task',expected_revision=1,template_ref={})]
    else:profile['unit']['argv'][profile['unit']['argv'].index('--config-sha256')+1]='0'*64
    if mutation in ('budget','dispatch'):
        argv=profile['unit']['argv'];path=Path(argv[argv.index('--config')+1])
        refreshed=_source_generation_replace_DATA(path,config)
        profile['source_descriptors']=[refreshed if pin['path']==str(path) else pin for pin in profile['source_descriptors']]
        argv[argv.index('--config-sha256')+1]=refreshed['sha256']
    # Keep config.profile_path coherent so each mutation reaches its own gate.
    profile_ref=_source_generation_replace_DATA(Path(case['profile_ref']['path']),profile)
    spec=h._document(case['spec']);spec['profile_ref']=profile_ref
    ref=_source_generation_canonical_DATA(case['root']/('bad-review-spec-'+mutation+'.json'),spec)
    with pytest.raises(ValueError):executor._source_generation_review(ref,profile_ref,profile,config)


@pytest.mark.parametrize('value',[None,{},True,250_000_000_000])
def test_source_generation_private_context_rejects_unregistered_policy_overrides(value):
    with pytest.raises(ValueError,match='context_not_authenticated'):executor._source_generation_record(value)


def test_source_generation_original_issue_clock_is_inherited_by_existing_capacity_authorizer(complete_source_generation_DATA,monkeypatch):
    case=complete_source_generation_DATA;h=_source_generation_health
    approval=executor.authorize_closed_capacity_source_generation(case['spec'],executor._source_generation_path(case['profile']))
    approval_value=h._document(approval);spec=dict.fromkeys(executor.CLOSED_CAPACITY_V2_KEYS-{'recorded_at','expires_at','manager_before'})
    spec.update(schema=executor.CLOSED_CAPACITY_V2_SCHEMA,profile_ref=case['profile_ref'],source_generation_authorization_ref=approval)
    reference=_source_generation_canonical_DATA(case['root']/'FICTIONAL-capacity-spec.json',spec)
    seen=[]
    # Original 1..200 history, current admission and Native are separate
    # modeled platform preconditions HERE, never claimed by this issue test.
    monkeypatch.setattr(executor,'_validate_closed_capacity_continuation',lambda value,*_:seen.append(dict(value)))
    monkeypatch.setattr(executor,'_closed_capacity_final_freshness',lambda *_:None)
    issued=executor.authorize_closed_capacity_continuation(reference,executor._closed_capacity_paths(case['profile'])[0])
    value=h._document(issued)
    assert value['recorded_at']==approval_value['recorded_at'] and value['expires_at']==approval_value['expires_at']
    assert seen[0]['source_generation_authorization_ref']==approval
    assert executor.GLOBAL_RETAINED_BYTES==220_000_000_000


def test_source_generation_v2_uses_same_ack_slot_and_original201_loop(claim_case,monkeypatch):
    _,profile,reference,ack_path,identity,lock_path=claim_case
    value=LocalData._document(reference);value['schema']=executor.CLOSED_CAPACITY_V2_SCHEMA
    value['source_generation_authorization_ref']={'fictional':'technical-source-only-approval'}
    path=Path(reference['path']);path.chmod(0o600);path.write_text(json.dumps(value));path.chmod(0o400)
    reference=LocalData.descriptor(path);monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV,reference['sha256'])
    # Explicit claim-case platform DATA; current source review is tested above.
    monkeypatch.setattr(executor,'health_owner',lambda:LocalData())
    monkeypatch.setattr(executor,'require_main_process',lambda *_:identity)
    monkeypatch.setattr(executor,'_validate_closed_capacity_continuation',lambda *_:(200,[]))
    monkeypatch.setattr(executor,'_closed_capacity_final_freshness',lambda *_:None)
    with lock_path.open('rb') as lock:
        assert executor.reconcile_closed_capacity_continuation(reference,{},profile,{},identity,executor_lock=lock)==(200,[])
    ack=LocalData._document(LocalData.descriptor(ack_path))
    assert ack['schema']==executor.CLOSED_CAPACITY_V2_ACK_SCHEMA
    assert ack['source_generation_authorization_ref']==value['source_generation_authorization_ref']
    assert ack['remaining_cycle_budget_before_first_owner']==9800
    assert ack['next_cycle_number']==201 and ack['generation_number']==ack['max_generations']==2
    assert ack['execution_budget_created'] is False


@pytest.mark.parametrize('closed_schema',['agent-supervisor/scientific-closed-capacity-continuation@1',
                                           'agent-supervisor/scientific-closed-capacity-continuation@2'])
def test_source_generation_schema_dispatch_keeps_legacy_keys_closed(closed_schema):
    value={'schema':closed_schema};keys=executor._closed_capacity_schema_keys(value)
    if closed_schema==executor.CLOSED_CAPACITY_SCHEMA:assert keys==executor.CLOSED_CAPACITY_KEYS
    else:assert keys==executor.CLOSED_CAPACITY_KEYS|{'source_generation_authorization_ref'}



def test_source_generation_current250_ledger_and_preflight_require_authenticated_context(complete_source_generation_DATA):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    approval=executor.authorize_closed_capacity_source_generation(case['spec'],executor._source_generation_path(case['profile']))
    context=executor._source_generation_context(approval,case['profile_ref'],case['profile'],case['config'])
    try:
        roots=[]
        for path in case['named']:
            info=path.lstat();roots.append(dict(path=str(path),device=info.st_dev,inode=info.st_ino))
        ledger=dict(schema='daemon-resource-reservations-v1',roots=roots,limit_bytes=250_000_000_000,reservations={})
        put(case['ledger'],ledger)
        failed=_source_generation_copy.deepcopy(case['request'])
        for row in case['replacements']:failed['source_sha256'][row['before_ref']['path']]=row['before_ref']['sha256']
        terminal={'reservation':dict(roots=roots,reservation_id='e'*32)}
        ledger_ref,parsed=executor._closed_capacity_ledger(failed,terminal,source_generation=context)
        assert parsed['limit_bytes']==250_000_000_000
        with pytest.raises(ValueError,match='unknown_or_present_parented_record'):
            executor._closed_capacity_ledger(failed,terminal)
        pre=dict(schema='original-owner-closed-capacity-preflight/v1',checked_at=executor.time.time(),max_age_seconds=60,
            ledger_ref=ledger_ref,roots=roots,resource_source_ref=case['replacements'][1]['after_ref'],
            source_READ_closed=True,ledger_lock_closed=True,reservation_created=False,native_grant_created=False,admission_authority=False,
            accounting=dict(observed_apparent_bytes=1234,outstanding_full_reservations_bytes=0,
                additional_requested_bytes=executor.REUSE_CYCLE_RESERVE_BYTES,
                charged_bytes=1234+executor.REUSE_CYCLE_RESERVE_BYTES,limit_bytes=250_000_000_000,
                minimum_filesystem_free_bytes=4*executor.REUSE_CYCLE_RESERVE_BYTES,scope='named_roots_only_apparent_bytes'))
        ref=put(case['root']/'fictional-original-preflight.json',pre)
        assert executor._closed_capacity_preflight({'capacity_preflight_ref':ref},case['profile'],failed,ledger_ref,parsed,
            source_generation=context)['accounting']['limit_bytes']==250_000_000_000
        assert executor.GLOBAL_RETAINED_BYTES==220_000_000_000
    finally:executor._close_source_generation_context(context)


def test_source_generation_expired_approval_is_never_renewed_or_reissued(complete_source_generation_DATA,monkeypatch):
    case=complete_source_generation_DATA;h=_source_generation_health
    path=executor._source_generation_path(case['profile'])
    reference=executor.authorize_closed_capacity_source_generation(case['spec'],path)
    raw=path.read_bytes();value=h._document(reference)
    monkeypatch.setattr(executor.time,'time',lambda:value['expires_at']+1)
    with pytest.raises(ValueError,match='expired_or_future'):
        executor._source_generation_context(reference,case['profile_ref'],case['profile'],case['config'])
    with pytest.raises(ValueError,match='issue_slot_spent'):
        executor.authorize_closed_capacity_source_generation(case['spec'],path)
    assert path.read_bytes()==raw
    assert not any(path.exists() for path in executor._closed_capacity_paths(case['profile']))


def test_source_generation_partial_approval_stays_spent(complete_source_generation_DATA):
    case=complete_source_generation_DATA;path=executor._source_generation_path(case['profile'])
    path.write_bytes(b'{unknown-partial-issue');path.chmod(0o400);raw=path.read_bytes()
    with pytest.raises(ValueError,match='issue_slot_spent'):
        executor.authorize_closed_capacity_source_generation(case['spec'],path)
    assert path.read_bytes()==raw


@pytest.mark.parametrize('closed_schema',['agent-supervisor/scientific-closed-capacity-continuation@1',
                                           'agent-supervisor/scientific-closed-capacity-continuation@2'])
def test_source_generation_both_schemas_reuse_original_serve_201_and_remaining_budget(finite_loop,monkeypatch,closed_schema):
    config_ref,recovery_ref,_,stop,holds,_=finite_loop
    h=executor.health_owner();original=h._document
    def document(ref):
        value=original(ref)
        return dict(value,schema=closed_schema) if ref==recovery_ref else value
    monkeypatch.setattr(type(h),'_document',staticmethod(document))
    attempted=[]
    monkeypatch.setattr(executor,'run_cycle',lambda *args,**options:attempted.append((options['cycle_number'],args[5])))
    assert executor.serve(config_ref,stop,recovery_ref=recovery_ref,catalogue_parser_ref={'fake':'parser'})==executor.HOLD_EXIT
    assert attempted==[(201,1),(202,2)]
    assert holds[-1][-1]=='finite_cycle_budget_exhausted'


@pytest.mark.parametrize('location',['request','plan_map','plan_descriptor','added_native_authority'])
def test_source_generation_unissued_template_projection_refuses_sixth_or_stale_source(complete_source_generation_DATA,location):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    certificate=h._document(case['transition']);template=h._document(certificate['template_ref'])
    request=h._document(template['request_ref']);plan=h._document(template['plan_ref'])
    row=case['replacements'][0];path=row['after_ref']['path']
    if location=='request':request['source_sha256'][path]=row['before_ref']['sha256']
    elif location=='plan_map':plan['native_metadata_cli']['source_sha256'][path]=row['before_ref']['sha256']
    elif location=='plan_descriptor':
        pin=next(pin for pin in plan['native_metadata_cli']['project_import_plan']['source_pins'] if pin['path']==path)
        pin['bytes']=float(pin['bytes'])
    else:request['new_native_authority']=True
    # Keep the new seed fixed and refresh the real physical plan/request
    # bindings, so the intended projection gate sees this changed payload.
    plan_ref=_source_generation_replace_DATA(Path(template['plan_ref']['path']),plan)
    request['source_sha256'][plan_ref['path']]=plan_ref['sha256']
    request['argv'][request['argv'].index('--plan-sha256')+1]=plan_ref['sha256']
    request_ref=_source_generation_replace_DATA(Path(template['request_ref']['path']),request)
    template.update(plan_ref=plan_ref,request_ref=request_ref)
    template_ref=_source_generation_replace_DATA(Path(certificate['template_ref']['path']),template)
    certificate['template_ref']=template_ref;config=dict(case['config'],idle_template_ref=template_ref)
    reference=put(case['root']/('bad-template-transition-'+location+'.json'),certificate)
    with pytest.raises(ValueError):
        executor._source_generation_template_transition(reference,case['replacements'],case['prior'],case['profile'],
            executor.health_owner()._document(next(pin for pin in case['prior']['source_descriptors']
                if pin['path']==case['prior']['unit']['argv'][case['prior']['unit']['argv'].index('--config')+1])),config)


def test_source_generation_cli_dispatch_is_explicit_and_separate_from_service(monkeypatch):
    seen=[]
    monkeypatch.setattr(executor,'_source_generation_authorize_main',lambda argv:seen.append(argv) or 0)
    assert executor.main(['authorize-closed-capacity-source-generation','--spec','fictional-input'])==0
    assert seen==[['authorize-closed-capacity-source-generation','--spec','fictional-input']]



def _source_generation_upgrade_current_v5_DATA(case,monkeypatch,immutable_source_generation_SDK_control_DATA):
    """Fictional original current-v5 SDK records; genuine SDK readers, no mocks.

    Every execution/Source/Native statement in these files is unit DATA only.
    We call existing metadata validators, never original_runner_wait, a process
    constructor, scheduler, service, Source lease or storage reservation.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    import pytest as actual_pytest
    h=_source_generation_health;put=_source_generation_canonical_DATA
    root=case['root']/'fictional-current-v5-SDK';root.mkdir()
    invocation=root/'invocation';invocation.mkdir()
    control_name, frozen_sources, frozen_owning, frozen_manifest = immutable_source_generation_SDK_control_DATA
    control = Path(control_name)
    keys = ('path', 'bytes', 'sha256')
    control_sources = [dict(zip(keys, values)) for values in frozen_sources]
    owning = dict(zip(keys, frozen_owning))
    control_manifest = dict(zip(keys, frozen_manifest))
    node = owning['path'] + '::test_source_generation_seven_roles'
    incoming_engine = next(pin for pin in control_sources if pin['path'].endswith('/todo_daemon/engine.py'))
    incoming_owner = next(pin for pin in control_sources if pin['path'].endswith('/repair_candidate_owners.py'))
    engine_ref = incoming_engine
    datasets=case['named'][0].parents[1];directory=datasets/'ipfs_datasets_py/optimizers/logic_theorem_optimizer'
    required=set(owner._v5_lifecycle_owner_paths(datasets))
    required.update(str(path) for path in (directory/'autoencoder_source_lease.py',directory/'autoencoder_resource_recovery.py',
        directory/'runtime_telemetry.py',directory/'validation_pytest.py',directory/'__init__.py',
        datasets/'ipfs_datasets_py/optimizers/__init__.py',datasets/'ipfs_datasets_py/__init__.py',
        datasets/'ipfs_datasets_py/router_deps.py',datasets/'ipfs_datasets_py/ipfs_backend_router.py',
        datasets/'ipfs_datasets_py/logic/__init__.py',datasets/'ipfs_datasets_py/logic/backends/process.py'))
    dependencies=[]
    for path in sorted(required):
        path=Path(path)
        if not path.exists():put(path,b'# explicit fictional original SDK dependency DATA\n')
        dependencies.append(h.descriptor(path))
    subjects=case['source_subjects'];package=case['packagepins'];parent_cli=subjects[-1]
    child_subjects=[pin for pin in subjects if pin['path']!=parent_cli['path']]
    candidate_root=Path(case['profile']['main_entrypoint']['path']).parents[3]
    pytest_ref=owner.descriptor(Path(actual_pytest.__file__).absolute())
    pytest_support=owner.descriptor(Path(actual_pytest.__file__).parent.parent/'_pytest/python.py')
    original_controls={pin['path']:pin for pin in [*control_sources,parent_cli,
        *[pin for pin in dependencies if Path(pin['path']).name in ('autoencoder_source_lease.py',
              'autoencoder_resource_recovery.py','validation_run_resources.py')]]}
    imports={pin['path']:pin for pin in [*dependencies,*subjects,pytest_support]}
    validation=dict(ordinary_scratch_profile={'schema':'owned-validation-ordinary-scratch-live/v1'},
        trusted_tests=[owning],selected_checks=[node],pytest_module=pytest_ref,import_sources=list(imports.values()),
        attempt_root=str(case['proof']),cpu_slots=1,memory_mb=1536,storage_bytes=16*1024**2,
        timeout_seconds=300,child_process_slots=4,scheduler_state_path=str(root/'scheduler.json'),
        expected_scheduler_policy_sha256='f'*64,roots=[str(path) for path in case['named']],resource_ledger=str(case['ledger']))
    policy=dict(schema=owner.POLICY_SCHEMA,validation=validation,python=str(Path(_source_generation_sys.executable).absolute()),
        datasets_root=str(datasets),source_descriptors=list(original_controls.values()),adoption={},
        control_release=dict(root=str(control),manifest=control_manifest))
    manifest={str(Path(pin['path']).relative_to(candidate_root)):
        dict(bytes=pin['bytes'],sha256=pin['sha256'],executable=False) for pin in package}
    commitment=owner.digest(dict(base_commit='0'*40,manifest=manifest))
    frozen=dict(stage='candidate_ready',validation_passed=False,runtime_adopted=False,
        binding=dict(candidate_root=str(candidate_root),namespace='ptr_lane_0',board_id=owner.REPAIR_PROFILE,attempt_id='e'*32),
        manifest=manifest,changed_paths=[str(Path(pin['path']).relative_to(candidate_root)) for pin in subjects[:2]],
        source_commitment=commitment)
    frozen['freeze_sha256']=owner.digest(frozen)
    freeze_ref=put(root/'frozen-candidate.json',frozen)
    policy_ref=put(root/'policy.json',policy)
    # Existing metadata parser is used only within the real explicit archived
    # SDK source capability. The default current-owner reader remains strict.
    sdk_source=owner.authenticate_archived_v5_sdk_source(policy_ref)
    try:assert owner.archived_v5_sdk_source_data(sdk_source)['producer_ref']==incoming_owner
    finally:owner.close_archived_v5_sdk_source(sdk_source)
    plan=owner.validation_plan(policy,policy_ref,frozen,freeze_ref,candidate_root,invocation)
    plan_ref=put(invocation/'validation-plan.json',plan)
    request=owner.validation_execution_request(policy,plan,plan_ref)
    local_request=put(invocation/'execution-request.json',request)
    original_path=owner.validation_control_directory(policy,frozen)/'execution-request.json'
    request_ref=put(original_path,Path(local_request['path']).read_bytes())
    put(invocation/'execution-request-publication.json',dict(schema=owner.CONTROL_PUBLICATION_SCHEMA,
        request=request_ref,invocation_copy=local_request,plan=plan_ref,freeze=freeze_ref,owner_policy=policy_ref,
        source_commitment=frozen['source_commitment']))
    assert owner.original_execution_request(invocation)==request_ref
    attempt=Path(request['attempt_directory']);attempt.mkdir(mode=0o700,parents=True)
    for name in ('logs','exports'):(attempt/name).mkdir()
    report=dict(schema=owner.WORK_SCHEMA,passed=True,source_READ_closed=True,source_snapshot_stable=True,
        cleanup_errors=[],error=None,junit_error=None,guard_denials=[],plan_sha256=plan_ref['sha256'],
        full_source_manifest_sha256=owner.digest(plan['source_pins']),full_source_file_count=len(plan['source_pins']),
        pytest_module_origin=policy['validation']['pytest_module'],startup=dict(python=policy['python'],isolated=1,no_bytecode=True),
        executed_source_buffers={pin['path']:pin['sha256'] for pin in child_subjects+[owning]},
        phases=[dict(nodeid=node,when=when,outcome='passed')for when in ('setup','call','teardown')],collected_cases=[node])
    put(attempt/'exports/trusted-validation.json',report)
    put(attempt/'exports/test-results.xml',b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="fictional_owning_tests" name="test_source_generation_seven_roles"/></testsuite></testsuites>')
    result=dict(schema='owned-validation-result/v1',completed=True,returncode=0,
        request_sha256=request_ref['sha256'],executed_checks=[node])
    result_ref=put(attempt/'logs/result.json',result);put(attempt/'logs/execution.json',result)
    rows=[]
    for path in sorted(attempt.rglob('*')):
        if path.is_file():
            pin=h.descriptor(path);info=path.lstat()
            rows.append(dict(path=str(path.relative_to(attempt)),bytes=pin['bytes'],sha256=pin['sha256'],device=info.st_dev,inode=info.st_ino))
    rid='9'*32;lease='8'*32;pid=987654321;birth='12345'
    seal=put(attempt/'logs/validation-durability.json',dict(schema='owned-validation-durability/v1',
        request_sha256=request_ref['sha256'],reservation_id=rid,result_sha256=result_ref['sha256'],files=rows))
    total=sum(row['bytes']for row in rows)+seal['bytes'];info=attempt.lstat()
    record=dict(status='released',artifacts_durable_asserted=True,validation_request_sha256=request_ref['sha256'],
        owner_pid=pid,owner_birth=birth,native_lease_id=lease,reservation_id=rid,
        attempt_directory=dict(path=str(attempt),device=info.st_dev,inode=info.st_ino),
        final_attempt_bytes=total,final_total_charged_bytes=total,storage_bytes=16*1024**2,
        final_accounting={'limit_bytes':250_000_000_000})
    terminal=dict(result=result_ref['path'],reservation=dict(status='released',record=record,cleanup_error=None,ledger_path=str(case['ledger']),
        resource_lease=dict(released=True,lease_id=lease,owner_pid=pid),storage_limit_bytes=250_000_000_000,
        process_observation=dict(live_processes=0,out_of_group_live_identities=[],tracking_error='')))
    terminal_ref=put(root/'original-terminal.json',terminal)
    runner_identity=dict(pid=pid,birth=birth);deadline=1000.0
    observed=dict(schema='command-execution-observation/v1',completion_kind='completed',process_pid=pid,
        process_returncode=0,leader_waited=True,output_capture='complete',work_deadline_monotonic=970.0,
        cleanup_deadline_monotonic=980.0,observed_at_monotonic=710.0)
    execution=dict(schema='agent-supervisor/v5-runner-execution-observation@1',request=request_ref,engine_source=engine_ref,
        exact_command_result=True,exact_execution_observation=True,execution_observation=observed,
        public_returncode=0,actual_process_pid=pid,actual_process_returncode=0,actual_Popen_returned=True,
        spawn_attempted=True,process_count=1,runner_identity=runner_identity,identity_observation_error=None,
        engine_returned=True,engine_started_at_monotonic=700.0,engine_finished_at_monotonic=720.0,
        spawn_returned_at_monotonic=702.0,identity_observed_at_monotonic=704.0,
        authority_deadline_monotonic=deadline,work_deadline_monotonic=970.0,cleanup_deadline_monotonic=980.0,
        requested_nominal_window_seconds=300,window_scope='fictional-original-window',
        source_closed=False,native_closed=False,storage_release_authorized=False,retry_authorized=False)
    execution_ref=put(invocation/'runner-execution-observation.json',execution)
    original_argv=[policy['python'],'-I','-B',parent_cli['path'],'--request',request_ref['path'],
        '--expected-request-sha256',request_ref['sha256'],'--resource-ledger',validation['resource_ledger']]
    wait=dict(request=request_ref,original_process_owner_returned=True,returncode=0,stdout=json.dumps(terminal),argv=original_argv,
        actual_Popen_returned=True,actual_process_waited=True,actual_process_returncode=0,
        runner_identity=runner_identity,engine_source=engine_ref,execution_observation=execution_ref)
    wait_ref=put(invocation/'runner-wait.json',wait)
    summary=dict(cases=1,phases=3,reservation_id=rid,native_lease_id=lease,natural_phase_counter_verified=True)
    custody=dict(schema='agent-supervisor/v5-validation-parent-source-custody@2',closure_kind='existing_verified_success',
        storage_status='released',source_closed=True,native_closed=True,storage_release_authorized=False,retry_authorized=False,
        original_wait=wait_ref,verified_checks=summary,source_receipts=[dict(schema_version='autoencoder-source-lease/v1',canonical_root=source_root,cooperative=True,
            lock_path=str(Path(source_root)/'workspace/control-plane/source-leases/source.lock'),mode='read',waited_seconds=0.0)
            for source_root in (policy['datasets_root'],str(candidate_root))],
        authority_deadline_monotonic=deadline,execution_observation=execution_ref,
        source_exit_finished_at_monotonic=730.0,work_deadline_monotonic=970.0,cleanup_deadline_monotonic=980.0)
    parent=put(invocation/'parent-source-closeout.json',dict(schema='agent-supervisor/v5-validation-parent-source-closeout@2',
        request=request_ref,original_wait=wait_ref,producer=incoming_owner,parent_source_custody=custody))
    qualification=put(root/'current-v5-source-qualification.json',dict(
        schema='agent-supervisor/scientific-source-generation-current-v5-qualification@1',qualification_profile='closed_v5_current_Source',
        qualification_owner_source_ref=incoming_owner,policy_ref=policy_ref,plan_ref=plan_ref,request_ref=request_ref,
        terminal_ref=terminal_ref,parent_source_closeout_ref=parent,invocation_root=str(invocation),owning_test_root=str(control),
        required_raw_subjects=subjects,source_inventory_subjects=package,owning_test_sources=[owning],
        expected_function_case_counts={node:1},runtime_activation_or_task_authority=False))
    case['legacy_qualification']=case['qualification'];case['qualification']=qualification
    commitment=h._document(case['commitment']);commitment['source_qualification']=qualification
    case['commitment']=_source_generation_replace_DATA(Path(case['commitment']['path']),commitment)
    spec=h._document(case['spec']);spec['source_commitment_ref']=case['commitment']
    case['spec']=_source_generation_replace_DATA(Path(case['spec']['path']),spec)
    case['v5']=dict(root=root,invocation=invocation,attempt=attempt,qualification=qualification,request=request_ref,
        plan=plan_ref,policy=policy_ref,parent=parent,terminal=terminal_ref,owning=owning,wait=wait_ref,report=report)
    return case


def test_source_generation_current_v5_uses_actual_existing_SDK_metadata_readers(complete_source_generation_DATA):
    case=complete_source_generation_DATA
    accepted=executor._source_generation_qualification(case['qualification'],case['profile'],
        {pin['path']:pin for pin in case['packagepins']})
    assert accepted['qualification_profile']=='closed_v5_current_Source'
    assert _source_generation_health._document(accepted['request_ref'])['schema']=='owned-validation-request/v5'
    assert not executor._source_generation_path(case['profile']).exists()


@pytest.mark.parametrize('mutation',['old_v1','missing_raw_body','wrong_function_count','bool_returncode','Source_open','unsealed_file'])
def test_source_generation_current_v5_denies_old_or_partial_qualification(complete_source_generation_DATA,mutation):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    value=h._document(case['qualification']);v5=case['v5']
    if mutation=='old_v1':reference=case['legacy_qualification']
    else:
        if mutation=='wrong_function_count':value['expected_function_case_counts'][next(iter(value['expected_function_case_counts']))]=2
        elif mutation=='Source_open':
            parent=h._document(value['parent_source_closeout_ref']);parent['parent_source_custody']['source_closed']=False
            value['parent_source_closeout_ref']=_source_generation_replace_DATA(Path(value['parent_source_closeout_ref']['path']),parent)
        elif mutation=='unsealed_file':(v5['attempt']/'exports/unsealed.txt').write_bytes(b'x')
        elif mutation=='bool_returncode':
            path=v5['attempt']/'logs/result.json';result=json.loads(path.read_bytes());result['returncode']=False
            _source_generation_replace_DATA(path,result)
        else:
            path=v5['attempt']/'exports/trusted-validation.json';report=json.loads(path.read_bytes())
            report['executed_source_buffers'].pop(case['source_subjects'][0]['path']);_source_generation_replace_DATA(path,report)
        reference=put(case['root']/('bad-current-v5-'+mutation+'.json'),value)
    with pytest.raises(ValueError):
        executor._source_generation_qualification(reference,case['profile'],{pin['path']:pin for pin in case['packagepins']})



@pytest.mark.parametrize('mutation',['unknown_request','wrong_outputs','bool_budget','open_Source','extra_Source_key','partial_owning_functions'])
def test_source_generation_current_v5_closed_request_and_Source_profile_denials(complete_source_generation_DATA,mutation):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    value=h._document(case['qualification']);v5=case['v5']
    if mutation in ('unknown_request','wrong_outputs','bool_budget'):
        request=h._document(value['request_ref'])
        if mutation=='unknown_request':request['allow_extra_native']=True
        elif mutation=='wrong_outputs':request['outputs']['db']=str(case['root']/'outside-attempt.db')
        else:request['cpu_slots']=True
        value['request_ref']=put(case['root']/('bad-v5-request-'+mutation+'.json'),request)
    elif mutation in ('open_Source','extra_Source_key'):
        parent=h._document(value['parent_source_closeout_ref']);receipt=parent['parent_source_custody']['source_receipts'][0]
        if mutation=='open_Source':receipt['mode']='write'
        else:receipt['completion_authority']=True
        value['parent_source_closeout_ref']=_source_generation_replace_DATA(Path(value['parent_source_closeout_ref']['path']),parent)
    else:
        # Add another declared function to the already pinned maintained test
        # and coherently update the qualification Source pins, plan, report and
        # owning descriptor. The full SDK validation would require re-sealing
        # all artifacts too; the exact owning-function inventory is separately
        # checked without treating a reported passed count as that inventory.
        counts=dict(value['expected_function_case_counts']);counts['unused-test-file::test_missing']=1
        value['expected_function_case_counts']=counts
    reference=put(case['root']/('bad-current-v5-shape-'+mutation+'.json'),value)
    with pytest.raises(ValueError):
        executor._source_generation_qualification(reference,case['profile'],{pin['path']:pin for pin in case['packagepins']})




def test_closed_capacity_v2_is_recognized_but_requires_opaque_source_generation_authority(monkeypatch):
    """Minimal RED1 on unchanged6e4: old reader rejects the new typed shape.

    This control grants nothing: the new reader must recognize its exact shape
    and refuse missing opaque authority before any profile, census or lifecycle.
    """
    monkeypatch.setattr(executor,'health_owner',lambda:_source_generation_health)
    value=dict.fromkeys(executor.CLOSED_CAPACITY_HISTORY_KEYS)
    value.update(schema='agent-supervisor/scientific-closed-capacity-continuation@2',
                 source_generation_authorization_ref={'fictional':'unissued-not-a-capability'})
    with pytest.raises(ValueError,match='closed_capacity_grant_authority'):
        executor._closed_capacity_history(value,{}, {}, {})



@pytest.mark.parametrize('mutation',['missing_bytes','float_bytes','boolean_bytes','extra_key'])
def test_source_generation_current_v5_policy_ref_retains_exact_descriptor_contract(complete_source_generation_DATA,mutation):
    case=complete_source_generation_DATA;h=_source_generation_health
    value=h._document(case['qualification']);offered=dict(value['policy_ref'])
    if mutation=='missing_bytes':offered.pop('bytes')
    elif mutation=='float_bytes':offered['bytes']=float(offered['bytes'])
    elif mutation=='boolean_bytes':offered['bytes']=True
    else:offered['unexpected_authority']=True
    value['policy_ref']=offered
    reference=_source_generation_canonical_DATA(case['root']/('malformed-policy-ref-'+mutation+'.json'),value)
    with pytest.raises(ValueError):
        executor._source_generation_qualification(reference,case['profile'],{pin['path']:pin for pin in case['packagepins']})




def test_source_generation_unissued_template_rejects_only_opaque_request_id_change(complete_source_generation_DATA):
    case=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    oldargv=case['prior']['unit']['argv']
    oldconfig=h._document(next(pin for pin in case['prior']['source_descriptors']
        if pin['path']==oldargv[oldargv.index('--config')+1]))
    # First prove this same complete fixture reaches the real positive reader.
    valid=executor._source_generation_template_transition(case['transition'],case['replacements'],
        case['prior'],case['profile'],oldconfig,case['config'])
    assert valid==case['request']
    assert valid['request_id']=='fictional-unissued:'+str(case['root'])
    certificate=h._document(case['transition']);template=h._document(certificate['template_ref'])
    request=h._document(template['request_ref']);request['request_id']+=':changed-opaque-id-only'
    assert {key:value for key,value in request.items() if key!='request_id'}=={
        key:value for key,value in valid.items() if key!='request_id'}
    # Refresh every consumed descriptor join. The intended rejection must be
    # native-request equality, rather than stale request/template/config bytes.
    request_ref=_source_generation_replace_DATA(Path(template['request_ref']['path']),request)
    template['request_ref']=request_ref
    template_ref=_source_generation_replace_DATA(Path(certificate['template_ref']['path']),template)
    certificate['template_ref']=template_ref
    transition_ref=put(case['root']/'changed-request-id-transition.json',certificate)
    config=_source_generation_copy.deepcopy(case['config']);config['idle_template_ref']=template_ref
    profile=_source_generation_copy.deepcopy(case['profile']);argv=profile['unit']['argv']
    config_path=argv[argv.index('--config')+1]
    config_ref=_source_generation_replace_DATA(Path(config_path),config)
    index=next(index for index,pin in enumerate(profile['source_descriptors']) if pin['path']==config_path)
    profile['source_descriptors'][index]=config_ref
    argv[argv.index('--config-sha256')+1]=config_ref['sha256']
    profile_ref=_source_generation_replace_DATA(Path(case['profile_ref']['path']),profile)
    spec=h._document(case['spec']);spec.update(profile_ref=profile_ref,template_transition_ref=transition_ref)
    spec_ref=put(case['root']/'changed-request-id-spec.json',spec)
    assert h._document(transition_ref)['template_ref']==h._document(config_ref)['idle_template_ref']==template_ref
    assert h._document(template_ref)['request_ref']==request_ref
    assert h._document(spec_ref)['profile_ref']==profile_ref
    assert h._document(profile_ref)['unit']['argv'][argv.index('--config-sha256')+1]==config_ref['sha256']
    with pytest.raises(executor.ExecutorHold,match='^source_generation_template_native_request_changed$'):
        executor._source_generation_template_transition(transition_ref,case['replacements'],
            case['prior'],profile,oldconfig,config)
    assert not executor._source_generation_path(profile).exists()
    assert not any(path.exists() for path in executor._closed_capacity_paths(profile))
    assert not Path(profile['receipt_path']).exists()


# Explicit immutable-source mode below preserves every preceding 88 control.
# The Source/Native, process and wait facts below are FICTIONAL unit DATA only.
@pytest.fixture(scope='module')
def immutable_source_generation_A_qualification_DATA(tmp_path_factory):
    """One immutable incoming-source population inside this original attempt.

    The seven SDK control files, three fictional role bodies and owning file
    form eleven source pins plus one excluded manifest. Only byte copies are shared. Policies, SDK witnesses and every
    fictional invocation stay private to their function-scoped case.
    """
    import stat
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    put = _source_generation_canonical_DATA
    base = Path(tmp_path_factory.getbasetemp())
    root = tmp_path_factory.mktemp('source-generation-immutable-A', numbered=False)
    assert root.parent == base and root.resolve(strict=True) == root
    assert not any(path.is_symlink() for path in (root, *root.parents))
    source_root = Path(owner.__file__).parents[3]
    sources = []
    for relative in ('ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py'):
        sources.append(put(root / relative, (source_root / relative).read_bytes()))
    sources.append(put(root / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py',
                       Path(engine.__file__).read_bytes()))
    # Fictional legal-ir bodies are SOURCE DATA, never executable deployment claims.
    for name in ('scientific_executor.py', 'native_executor_health.py', 'native_catalogue_probe.py'):
        sources.append(put(root / 'ipfs_accelerate_py/agent_supervisor/rescue' / name,
                           ('# fictional immutable qualification body: ' + name + '\n').encode()))
    owning = put(root / 'tests/fictional_owning_tests.py',
                 b'def test_source_generation_seven_roles():\n    pass\n')
    manifest = put(root / 'source-manifest.json', dict(schema=owner.CONTROL_SCHEMA, root=str(root),
        files=sorted(sources + [owning], key=lambda pin: pin['path']), namespace_profiles=[]))
    pins = sources + [owning, manifest]
    assert len(pins) == 12 and len({pin['path'] for pin in pins}) == 12
    assert all(root in Path(pin['path']).parents and len(os.fsencode(pin['path'])) <= 512 for pin in pins)
    assert sum(pin['bytes'] for pin in pins) <= 768 * 1024
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_nlink',
              'st_size', 'st_mtime_ns', 'st_ctime_ns')
    def snapshot():
        population = list(root.rglob('*'))
        assert not any(path.is_symlink() for path in population)
        assert {str(path) for path in population if path.is_file()} == {pin['path'] for pin in pins}
        snapshots = []
        for pin in pins:
            path = Path(pin['path'])
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                before = os.fstat(stream.fileno())
                assert stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
                assert stat.S_IMODE(before.st_mode) == 0o400 and before.st_size == pin['bytes']
                raw = stream.read()
                after = os.fstat(stream.fileno())
            identity = tuple(getattr(before, name) for name in fields)
            assert identity == tuple(getattr(after, name) for name in fields)
            assert identity == tuple(getattr(path.lstat(), name) for name in fields)
            assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256']
            snapshots.append(identity)
        return tuple(snapshots)
    original = snapshot()
    # No mutable descriptor object or policy/capability is shared across cases.
    keys = ('path', 'bytes', 'sha256')
    bundle = (str(root), tuple(tuple(pin[key] for key in keys) for pin in sources),
              tuple(owning[key] for key in keys), tuple(manifest[key] for key in keys))
    yield bundle
    assert snapshot() == original  # Verify retained files; never delete them.


def _source_generation_immutable_qualification_v5_DATA(case, immutable_source_generation_A_qualification_DATA):
    """Fictional original current-v5 SDK records; genuine SDK readers, no mocks.

    Every execution/Source/Native statement in these files is unit DATA only.
    We call existing metadata validators, never original_runner_wait, a process
    constructor, scheduler, service, Source lease or storage reservation.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    import pytest as actual_pytest
    h=_source_generation_health;put=_source_generation_canonical_DATA
    root=case['root']/'immutable-original-v5-DATA';root.mkdir()
    invocation=root/'invocation';invocation.mkdir()
    control_name, frozen_sources, frozen_owning, frozen_manifest = immutable_source_generation_A_qualification_DATA
    control = Path(control_name)
    keys = ('path', 'bytes', 'sha256')
    control_sources = [dict(zip(keys, values)) for values in frozen_sources]
    owning = dict(zip(keys, frozen_owning))
    control_manifest = dict(zip(keys, frozen_manifest))
    node = owning['path'] + '::test_source_generation_seven_roles'
    incoming_engine = next(pin for pin in control_sources if pin['path'].endswith('/todo_daemon/engine.py'))
    incoming_owner = next(pin for pin in control_sources if pin['path'].endswith('/repair_candidate_owners.py'))
    engine_ref = incoming_engine
    datasets=case['named'][0].parents[1];directory=datasets/'ipfs_datasets_py/optimizers/logic_theorem_optimizer'
    required=set(owner._v5_lifecycle_owner_paths(datasets))
    required.update(str(path) for path in (directory/'autoencoder_source_lease.py',directory/'autoencoder_resource_recovery.py',
        directory/'runtime_telemetry.py',directory/'validation_pytest.py',directory/'__init__.py',
        datasets/'ipfs_datasets_py/optimizers/__init__.py',datasets/'ipfs_datasets_py/__init__.py',
        datasets/'ipfs_datasets_py/router_deps.py',datasets/'ipfs_datasets_py/ipfs_backend_router.py',
        datasets/'ipfs_datasets_py/logic/__init__.py',datasets/'ipfs_datasets_py/logic/backends/process.py'))
    dependencies=[]
    for path in sorted(required):
        path=Path(path)
        if not path.exists():put(path,b'# explicit fictional original SDK dependency DATA\n')
        dependencies.append(h.descriptor(path))
    subjects=case['source_subjects'];package=case['packagepins'];parent_cli=subjects[-1]
    child_subjects=[pin for pin in subjects if pin['path']!=parent_cli['path']]
    candidate_root=Path(case['profile']['main_entrypoint']['path']).parents[3]
    pytest_ref=owner.descriptor(Path(actual_pytest.__file__).absolute())
    pytest_support=owner.descriptor(Path(actual_pytest.__file__).parent.parent/'_pytest/python.py')
    original_controls={pin['path']:pin for pin in [*control_sources,parent_cli,
        *[pin for pin in dependencies if Path(pin['path']).name in ('autoencoder_source_lease.py',
              'autoencoder_resource_recovery.py','validation_run_resources.py')]]}
    imports={pin['path']:pin for pin in [*dependencies,*subjects,pytest_support]}
    validation=dict(ordinary_scratch_profile={'schema':'owned-validation-ordinary-scratch-live/v1'},
        trusted_tests=[owning],selected_checks=[node],pytest_module=pytest_ref,import_sources=list(imports.values()),
        attempt_root=str(case['proof']),cpu_slots=1,memory_mb=1536,storage_bytes=16*1024**2,
        timeout_seconds=300,child_process_slots=4,scheduler_state_path=str(root/'scheduler.json'),
        expected_scheduler_policy_sha256='f'*64,roots=[str(path) for path in case['named']],resource_ledger=str(case['ledger']))
    policy=dict(schema=owner.POLICY_SCHEMA,validation=validation,python=str(Path(_source_generation_sys.executable).absolute()),
        datasets_root=str(datasets),source_descriptors=list(original_controls.values()),adoption={},
        control_release=dict(root=str(control),manifest=control_manifest))
    manifest={str(Path(pin['path']).relative_to(candidate_root)):
        dict(bytes=pin['bytes'],sha256=pin['sha256'],executable=False) for pin in package}
    frozen=dict(schema='agent-supervisor/immutable-source-snapshot@1',
        binding=dict(namespace='ptr_lane_0',board_id=owner.REPAIR_PROFILE,attempt_id='d'*32),
        source_root=str(candidate_root),manifest=manifest,
        source_commitment=owner.digest(dict(source_root=str(candidate_root),manifest=manifest)))
    freeze_ref=put(root/'immutable-source-snapshot.json',frozen)
    validation['source_qualification']=dict(schema='agent-supervisor/immutable-source-qualification@1',
        snapshot=freeze_ref,required_source_bodies=sorted(subjects[:2],key=lambda pin:pin['path']))
    policy_ref=put(root/'policy.json',policy)
    # Existing metadata parser is used only within the real explicit archived
    # SDK source capability. The default current-owner reader remains strict.
    sdk_source=owner.authenticate_archived_v5_sdk_source(policy_ref)
    try:assert owner.archived_v5_sdk_source_data(sdk_source)['producer_ref']==incoming_owner
    finally:owner.close_archived_v5_sdk_source(sdk_source)
    plan=owner.validation_plan(policy,policy_ref,frozen,freeze_ref,candidate_root,invocation)
    plan_ref=put(invocation/'validation-plan.json',plan)
    request=owner.validation_execution_request(policy,plan,plan_ref)
    local_request=put(invocation/'execution-request.json',request)
    original_path=owner.validation_control_directory(policy,frozen)/'execution-request.json'
    request_ref=put(original_path,Path(local_request['path']).read_bytes())
    put(invocation/'execution-request-publication.json',dict(schema=owner.CONTROL_PUBLICATION_SCHEMA,
        request=request_ref,invocation_copy=local_request,plan=plan_ref,freeze=freeze_ref,owner_policy=policy_ref,
        source_commitment=frozen['source_commitment']))
    assert owner.original_execution_request(invocation)==request_ref
    attempt=Path(request['attempt_directory']);attempt.mkdir(mode=0o700,parents=True)
    for name in ('logs','exports'):(attempt/name).mkdir()
    report=dict(schema=owner.WORK_SCHEMA,passed=True,source_READ_closed=True,source_snapshot_stable=True,
        cleanup_errors=[],error=None,junit_error=None,guard_denials=[],plan_sha256=plan_ref['sha256'],
        full_source_manifest_sha256=owner.digest(plan['source_pins']),full_source_file_count=len(plan['source_pins']),
        pytest_module_origin=policy['validation']['pytest_module'],startup=dict(python=policy['python'],isolated=1,no_bytecode=True),
        executed_source_buffers={pin['path']:pin['sha256'] for pin in child_subjects+[owning]},
        phases=[dict(nodeid=node,when=when,outcome='passed')for when in ('setup','call','teardown')],collected_cases=[node],
        source_qualification=validation['source_qualification'], required_source_bodies=plan['required_source_bodies'],
        freeze=freeze_ref,source_commitment=frozen['source_commitment'],semantic_negative=False,negative_profile=None,
        raw_compiled_sources={pin['path']:pin['sha256'] for pin in child_subjects+[owning]})
    put(attempt/'exports/trusted-validation.json',report)
    put(attempt/'exports/test-results.xml',b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="fictional_owning_tests" name="test_source_generation_seven_roles"/></testsuite></testsuites>')
    result=dict(schema='owned-validation-result/v1',completed=True,returncode=0,
        request_sha256=request_ref['sha256'],executed_checks=[node])
    result_ref=put(attempt/'logs/result.json',result);put(attempt/'logs/execution.json',result)
    rows=[]
    for path in sorted(attempt.rglob('*')):
        if path.is_file():
            pin=h.descriptor(path);info=path.lstat()
            rows.append(dict(path=str(path.relative_to(attempt)),bytes=pin['bytes'],sha256=pin['sha256'],device=info.st_dev,inode=info.st_ino))
    rid='9'*32;lease='8'*32;pid=987654321;birth='12345'
    seal=put(attempt/'logs/validation-durability.json',dict(schema='owned-validation-durability/v1',
        request_sha256=request_ref['sha256'],reservation_id=rid,result_sha256=result_ref['sha256'],files=rows))
    total=sum(row['bytes']for row in rows)+seal['bytes'];info=attempt.lstat()
    record=dict(status='released',artifacts_durable_asserted=True,validation_request_sha256=request_ref['sha256'],
        owner_pid=pid,owner_birth=birth,native_lease_id=lease,reservation_id=rid,
        attempt_directory=dict(path=str(attempt),device=info.st_dev,inode=info.st_ino),
        final_attempt_bytes=total,final_total_charged_bytes=total,storage_bytes=16*1024**2,
        final_accounting={'limit_bytes':250_000_000_000})
    terminal=dict(result=result_ref['path'],reservation=dict(status='released',record=record,cleanup_error=None,ledger_path=str(case['ledger']),
        resource_lease=dict(released=True,lease_id=lease,owner_pid=pid),storage_limit_bytes=250_000_000_000,
        process_observation=dict(live_processes=0,out_of_group_live_identities=[],tracking_error='')))
    terminal_ref=put(root/'original-terminal.json',terminal)
    runner_identity=dict(pid=pid,birth=birth);deadline=1000.0
    observed=dict(schema='command-execution-observation/v1',completion_kind='completed',process_pid=pid,
        process_returncode=0,leader_waited=True,output_capture='complete',work_deadline_monotonic=970.0,
        cleanup_deadline_monotonic=980.0,observed_at_monotonic=710.0)
    execution=dict(schema='agent-supervisor/v5-runner-execution-observation@1',request=request_ref,engine_source=engine_ref,
        exact_command_result=True,exact_execution_observation=True,execution_observation=observed,
        public_returncode=0,actual_process_pid=pid,actual_process_returncode=0,actual_Popen_returned=True,
        spawn_attempted=True,process_count=1,runner_identity=runner_identity,identity_observation_error=None,
        engine_returned=True,engine_started_at_monotonic=700.0,engine_finished_at_monotonic=720.0,
        spawn_returned_at_monotonic=702.0,identity_observed_at_monotonic=704.0,
        authority_deadline_monotonic=deadline,work_deadline_monotonic=970.0,cleanup_deadline_monotonic=980.0,
        requested_nominal_window_seconds=300,window_scope='fictional-original-window',
        source_closed=False,native_closed=False,storage_release_authorized=False,retry_authorized=False)
    execution_ref=put(invocation/'runner-execution-observation.json',execution)
    original_argv=[policy['python'],'-I','-B',parent_cli['path'],'--request',request_ref['path'],
        '--expected-request-sha256',request_ref['sha256'],'--resource-ledger',validation['resource_ledger']]
    wait=dict(request=request_ref,original_process_owner_returned=True,returncode=0,stdout=json.dumps(terminal),argv=original_argv,
        actual_Popen_returned=True,actual_process_waited=True,actual_process_returncode=0,
        runner_identity=runner_identity,engine_source=engine_ref,execution_observation=execution_ref)
    wait_ref=put(invocation/'runner-wait.json',wait)
    # Genuine saved-DATA verifier supplies the exact summary; no custody API is invoked.
    summary=owner.verify_validation_evidence(policy,plan,request_ref,terminal,invocation)
    custody=dict(schema='agent-supervisor/v5-validation-parent-source-custody@2',closure_kind='existing_verified_success',
        storage_status='released',source_closed=True,native_closed=True,storage_release_authorized=False,retry_authorized=False,
        original_wait=wait_ref,verified_checks=summary,source_receipts=[dict(schema_version='autoencoder-source-lease/v1',canonical_root=source_root,cooperative=True,
            lock_path=str(Path(source_root)/'workspace/control-plane/source-leases/source.lock'),mode='read',waited_seconds=0.0)
            for source_root in (policy['datasets_root'],str(candidate_root))],
        authority_deadline_monotonic=deadline,execution_observation=execution_ref,
        source_exit_finished_at_monotonic=730.0,work_deadline_monotonic=970.0,cleanup_deadline_monotonic=980.0)
    parent=put(invocation/'parent-source-closeout.json',dict(schema='agent-supervisor/v5-validation-parent-source-closeout@2',
        request=request_ref,original_wait=wait_ref,producer=incoming_owner,parent_source_custody=custody))
    qualification=put(root/'current-v5-source-qualification.json',dict(
        schema='agent-supervisor/scientific-source-generation-current-v5-qualification@1',qualification_profile='closed_v5_current_immutable_Source',
        qualification_owner_source_ref=incoming_owner,policy_ref=policy_ref,plan_ref=plan_ref,request_ref=request_ref,
        terminal_ref=terminal_ref,parent_source_closeout_ref=parent,invocation_root=str(invocation),owning_test_root=str(control),
        required_raw_subjects=subjects,source_inventory_subjects=package,owning_test_sources=[owning],
        expected_function_case_counts={node:1},runtime_activation_or_task_authority=False))
    case['qualification']=qualification
    case['v5']=dict(root=root,invocation=invocation,attempt=attempt,qualification=qualification,request=request_ref,
        plan=plan_ref,policy=policy_ref,parent=parent,terminal=terminal_ref,owning=owning,wait=wait_ref,report=report)
    return case


@pytest.fixture
def immutable_source_generation_qualification_DATA(complete_source_generation_DATA,
        immutable_source_generation_A_qualification_DATA):
    """Fresh private evidence over a separately sealed immutable A population.

    The original complete case, all shared source files and its original proof
    records are retained untouched. This fixture models no new live operation.
    """
    original=complete_source_generation_DATA;h=_source_generation_health;put=_source_generation_canonical_DATA
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    case=_source_generation_copy.deepcopy(original)
    root=original['root']/'immutable-mode-DATA';root.mkdir()
    control_name,frozen_sources,frozen_owning,_=immutable_source_generation_A_qualification_DATA
    control=Path(control_name);keys=('path','bytes','sha256')
    package=[dict(zip(keys,row))for row in frozen_sources]+[dict(zip(keys,frozen_owning))]
    by_name={Path(pin['path']).name:pin for pin in package}
    profile=_source_generation_copy.deepcopy(original['profile'])
    oldpackage=Path(profile['main_entrypoint']['path']).parents[3]
    profile.update(main_entrypoint=by_name['scientific_executor.py'],health_module=by_name['native_executor_health.py'])
    profile['source_descriptors']=[pin for pin in profile['source_descriptors']
        if oldpackage not in Path(pin['path']).parents]+package
    argv=profile['unit']['argv'];argv[2]=profile['main_entrypoint']['path']
    argv[argv.index('--catalogue-parser-source')+1]=by_name['native_catalogue_probe.py']['path']
    argv[argv.index('--catalogue-parser-sha256')+1]=by_name['native_catalogue_probe.py']['sha256']
    profile['unit']['environment']['PYTHONPATH']=str(control)+os.pathsep+str(original['root']/'old-package')
    # A fresh profile/config pair preserves the original complete fixture files.
    config=_source_generation_copy.deepcopy(original['config']);config['profile_path']=str(root/'profile.json')
    config_ref=put(root/'config.json',config)
    oldconfig_path=argv[argv.index('--config')+1]
    profile['source_descriptors']=[pin for pin in profile['source_descriptors'] if pin['path']!=oldconfig_path]+[config_ref]
    argv[argv.index('--config')+1]=config_ref['path'];argv[argv.index('--config-sha256')+1]=config_ref['sha256']
    case.update(root=root,profile=profile,profile_ref=put(root/'profile.json',profile),config=config,
        packagepins=package,source_subjects=[profile['main_entrypoint'],profile['health_module']]+original['source_subjects'][2:])
    case=_source_generation_immutable_qualification_v5_DATA(case,immutable_source_generation_A_qualification_DATA)
    previous=h._document(original['commitment'])
    inventory=put(root/'source-inventory.json',dict(schema='legal-ir-scientific-source-inventory/v1',source_root=str(control),
        source_commit=previous['source_commit'],source_tree=previous['source_tree'],file_count=len(package),
        files=[dict(pin,mode=0o400)for pin in package]))
    commitment=dict(previous,source_root=str(control),source_inventory=inventory,
        source_qualification=case['qualification'],main_entrypoint=profile['main_entrypoint'],
        health_module=profile['health_module'],catalogue_parser=by_name['native_catalogue_probe.py'])
    case.update(commitment=put(root/'source-commitment.json',commitment),source_inventory=inventory,
                original_complete_case=original)
    assert owner.checked_source_qualification(h._document(case['v5']['policy']))['root']==control
    return case


def test_source_generation_immutable_mode_rechecks_same_sealed_Main_health_and_original_metadata(
        immutable_source_generation_qualification_DATA):
    case=immutable_source_generation_qualification_DATA
    source,selected=executor._source_generation_source_inventory(case['commitment'],case['profile'])
    accepted=executor._source_generation_qualification(case['qualification'],case['profile'],selected)
    policy=_source_generation_health._document(accepted['policy_ref'])
    plan=_source_generation_health._document(accepted['plan_ref'])
    assert accepted['qualification_profile']=='closed_v5_current_immutable_Source'
    assert source['source_root']==policy['control_release']['root']==plan['candidate_root']
    assert plan['required_candidate_bodies']==[]
    assert plan['required_source_bodies']==sorted(case['profile'][role]['path']for role in ('main_entrypoint','health_module'))
    assert accepted['runtime_activation_or_task_authority'] is False
    assert not executor._source_generation_path(case['profile']).exists()


@pytest.mark.parametrize('mismatch',['immutable_enum_with_repair_policy','repair_enum_with_immutable_policy'])
def test_source_generation_immutable_enum_is_exactly_paired_with_policy(
        immutable_source_generation_qualification_DATA,mismatch):
    case=immutable_source_generation_qualification_DATA;h=_source_generation_health
    target=case['original_complete_case']if mismatch=='immutable_enum_with_repair_policy'else case
    value=h._document(target['qualification'])
    value['qualification_profile']=('closed_v5_current_immutable_Source'if target is not case else'closed_v5_current_Source')
    reference=_source_generation_canonical_DATA(case['root']/('enum-mismatch-'+mismatch+'.json'),value)
    with pytest.raises(ValueError,match='source_generation_v5_qualification_profile_policy_mismatch'):
        executor._source_generation_qualification(reference,target['profile'],{pin['path']:pin for pin in target['packagepins']})
    assert not executor._source_generation_path(target['profile']).exists()


@pytest.mark.parametrize('missing_role',['main_entrypoint','health_module'])
def test_source_generation_immutable_mode_requires_both_exact_role_bodies(
        immutable_source_generation_qualification_DATA,missing_role):
    case=immutable_source_generation_qualification_DATA;h=_source_generation_health
    value=h._document(case['qualification']);policy=h._document(value['policy_ref'])
    profile=policy['validation']['source_qualification']
    profile['required_source_bodies']=[pin for pin in profile['required_source_bodies']
        if pin['path']!=case['profile'][missing_role]['path']]
    value['policy_ref']=_source_generation_canonical_DATA(case['root']/('missing-'+missing_role+'-policy.json'),policy)
    reference=_source_generation_canonical_DATA(case['root']/('missing-'+missing_role+'-qualification.json'),value)
    # The genuine SDK policy reader must reject this before accepting any plan.
    with pytest.raises(ValueError,match='immutable qualification requires exact Main and health source bodies'):
        executor._source_generation_qualification(reference,case['profile'],{pin['path']:pin for pin in case['packagepins']})
    assert not executor._source_generation_path(case['profile']).exists()


def test_source_generation_qualification_receipt_never_becomes_source_generation_approval(
        immutable_source_generation_qualification_DATA):
    case=immutable_source_generation_qualification_DATA;h=_source_generation_health
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owner
    v5=case['v5'];policy=h._document(v5['policy']);context=owner.checked_source_qualification(policy)
    terminal=h._document(v5['terminal']);plan=h._document(v5['plan'])
    summary=owner.verify_validation_evidence(policy,plan,v5['request'],terminal,v5['invocation'])
    receipt=dict(schema=owner.SOURCE_QUALIFICATION_SCHEMA,snapshot=context['snapshot_ref'],
        source_qualification=context['profile'],source_commitment=context['snapshot']['source_commitment'],passed=True,
        source_closed=True,native_closed=True,processes_closed=True,storage_closed=True,cleanup_errors=[],
        original_receipt_reference=v5['terminal']['path'],original_receipt_descriptor=v5['terminal'],
        verified_checks=summary,parent_source_closeout=v5['parent'],repair_authority=False,
        adoption_authority=False,activation_authority=False,completion_authority=False)
    reference=_source_generation_canonical_DATA(case['root']/'source-only-qualification-receipt.json',receipt)
    # The positive qualification shape supplies no technical-generation grant.
    with pytest.raises(ValueError,match='source_generation_approval_shape'):
        executor._source_generation_context(reference,case['profile_ref'],case['profile'],case['config'],live_clock=False)
    assert not executor._source_generation_path(case['profile']).exists()



def test_source_generation_qualification_AST_metadata_parse_uses_bound_bytes_without_physical_origin(
        complete_source_generation_DATA, monkeypatch):
    """Full review AST metadata inspection grants no raw execution provenance."""
    import ast
    case=complete_source_generation_DATA;h=_source_generation_health
    qualification=h._document(case['qualification'])
    assert len(qualification['owning_test_sources'])==1
    owning=qualification['owning_test_sources'][0]
    resource=next(row['after_ref'] for row in case['replacements']
        if Path(row['after_ref']['path']).name=='autoencoder_daemon_resources.py')
    subjects=[owning,resource]
    sources={h._bound(pin):pin for pin in subjects}
    assert len(sources)==2 and all(type(raw) is bytes for raw in sources)
    assert all(h.descriptor(Path(pin['path']))==pin for pin in subjects)
    original_parse=ast.parse;parsed=[]
    def metadata_parse(source, filename='<unknown>', mode='exec', **options):
        if source in sources:
            pin=sources[source]
            assert type(source) is bytes and filename=='<unknown>' and mode=='exec'
            assert hashlib.sha256(source).hexdigest()==pin['sha256']
            parsed.append(pin['path'])
        return original_parse(source, filename=filename, mode=mode, **options)
    monkeypatch.setattr(ast,'parse',metadata_parse)
    review=executor._source_generation_review(case['spec'],case['profile_ref'],
        case['profile'],case['config'])
    try:
        assert review['source']['source_qualification']==case['qualification']
        assert review['policy']==h._document(case['policy'])
        assert parsed==[pin['path'] for pin in subjects]
        assert review['policy']['execution_budget_created'] is False
        assert review['policy']['hold_removal_authorized'] is False
    finally:
        h.close_historical_source_witness(review['witness'])
    assert not executor._source_generation_path(case['profile']).exists()


@pytest.fixture
def physical_R_fifteen_preparation_DATA(source_generation_profile_DATA, monkeypatch):
    """Fictional constructor DATA, not a live import or Source qualification."""
    h = _source_generation_health; put = _source_generation_canonical_DATA
    root = source_generation_profile_DATA['root']/'explicit-fifteen-preparation-DATA'
    prior = _source_generation_copy.deepcopy(source_generation_profile_DATA['current'])
    stage = _source_generation_copy.deepcopy(prior)
    canonical, physical = root/'canonical-datasets', root/'physical-datasets'
    named = (canonical/'workspace/logs', canonical/'workspace/queues', canonical/'docs/evidence', root/'pytest')
    for path in named: path.mkdir(parents=True)
    monkeypatch.setattr(executor, 'ORIGINAL_NAMED_ROOTS', named)
    monkeypatch.setattr(executor, 'health_owner', lambda: h)
    binding = dict(physical_root=str(physical), canonical_root=str(canonical))
    changed = ('ipfs_datasets_py/logic/backends/process.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py',
        'scripts/ops/legal_ir/run_owned_validation.py')
    replacements = []
    for index, relative in enumerate(changed):
        before = put(canonical/relative, ('# fictional prior changed role '+str(index)+'\n').encode())
        after = put(physical/relative, ('# fictional current changed role '+str(index)+'\n').encode())
        slot = len(prior['source_descriptors'])
        prior['source_descriptors'].append(before); stage['source_descriptors'].append(after)
        replacements.append(dict(slot=slot, before_ref=before, after_ref=after))
    # Independent population: do not copy the implementation's role constant.
    relatives = ('ipfs_datasets_py/__init__.py', 'ipfs_datasets_py/ipfs_backend_router.py',
        'ipfs_datasets_py/logic/__init__.py', 'ipfs_datasets_py/optimizers/__init__.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/runtime_telemetry.py',
        'ipfs_datasets_py/router_deps.py')
    relocations = []
    for index, relative in enumerate(relatives):
        raw = ('# fictional unchanged dependency '+str(index)+'\n').encode()
        before = put(canonical/relative, raw); after = put(physical/relative, raw)
        slot = len(prior['source_descriptors'])
        prior['source_descriptors'].append(before); stage['source_descriptors'].append(after)
        relocations.append(dict(slot=slot, before_ref=before, after_ref=after))
    oldref = put(root/'prior-profile.json', prior)
    snapshot = put(root/'prior-profile-snapshot.json', prior)
    namespace_path = physical/'ipfs_datasets_py/logic/backends'
    info = namespace_path.lstat()
    profile = dict(schema='native-embedded-physical-R-source@1', source_root_binding=binding,
        source_descriptors=sorted([row['after_ref'] for row in replacements+relocations],
                                  key=lambda pin: pin['path']),
        normal_namespace_profiles=[dict(fullname='ipfs_datasets_py.logic.backends',
            path=str(namespace_path), device=info.st_dev, inode=info.st_ino, initializer_absent=True)])
    value = dict(schema='agent-supervisor/scientific-idle-source-preparation@3',
        prior_profile_ref=oldref, prior_profile_snapshot_ref=snapshot,
        profile_ref=put(root/'stage-profile.json', stage), source_relocations=relocations,
        physical_source_profile=profile, replacements=replacements, source_root_binding=binding)
    return dict(root=root, value=value, spec=put(root/'preparation.json', value),
                relocations=relocations, physical_source_profile=profile)


def test_source_generation_fifteen_preparation_is_only_constructor_DATA(physical_R_fifteen_preparation_DATA,
        monkeypatch):
    case = physical_R_fifteen_preparation_DATA
    def forbidden(*_, **__): raise AssertionError('constructor DATA cannot launch a process')
    monkeypatch.setattr(executor.subprocess, 'Popen', forbidden)
    comparison = executor.prepare_idle_source_replacement(case['spec'])
    assert comparison['schema'] == 'agent-supervisor/scientific-idle-source-comparison@3'
    assert comparison['changed_source_count'] == 5 and comparison['relocated_source_count'] == 15
    assert comparison['source_relocations'] == case['relocations']
    assert comparison['physical_source_profile'] == case['physical_source_profile']
    assert comparison['raw_inverse_verified'] is True
    assert comparison['namespace_metadata_is_compile_evidence'] is False
    assert comparison['native_driver_template_transition_unresolved'] is True
    assert comparison['runtime_qualified'] is comparison['activation_ready'] is comparison['authority_granted'] is False


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'reordered', 'changed-payload',
    'canonical-after', 'wrong-role', 'boolean-slot', 'crossed-slot', 'wrong-size'])
def test_source_generation_fifteen_preparation_rejects_relocation_mutations_DATA(
        physical_R_fifteen_preparation_DATA, fault):
    case = physical_R_fifteen_preparation_DATA; h = _source_generation_health
    value = _source_generation_copy.deepcopy(case['value']); rows = value['source_relocations']
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows[-1] = _source_generation_copy.deepcopy(rows[0])
    elif fault == 'reordered': rows.reverse()
    elif fault == 'changed-payload': rows[0]['after_ref']['sha256'] = '1'*64
    elif fault == 'canonical-after': rows[0]['after_ref']['path'] = rows[0]['before_ref']['path']
    elif fault == 'wrong-role': rows[0]['before_ref']['path'] += '.unlisted'
    elif fault == 'boolean-slot': rows[0]['slot'] = True
    elif fault == 'crossed-slot': rows[0]['slot'] = value['replacements'][0]['slot']
    else: rows[0]['after_ref']['bytes'] += 1
    ref = _source_generation_canonical_DATA(case['root']/('bad-relocation-'+fault+'.json'), value)
    with pytest.raises(ValueError): executor.prepare_idle_source_replacement(ref)
    assert not executor._source_generation_path(h._profile(value['profile_ref'])).exists()


@pytest.mark.parametrize('fault', ['profile-schema', 'profile-root', 'profile-order', 'profile-omission',
    'namespace-name', 'namespace-path', 'boolean-device', 'zero-inode', 'initializer-present', 'extra-namespace'])
def test_source_generation_fifteen_preparation_rejects_profile_namespace_mutations_DATA(
        physical_R_fifteen_preparation_DATA, fault):
    case = physical_R_fifteen_preparation_DATA
    value = _source_generation_copy.deepcopy(case['value']); profile = value['physical_source_profile']
    namespace = profile['normal_namespace_profiles'][0]
    if fault == 'profile-schema': profile['schema'] = 'native-embedded-physical-R-source@0'
    elif fault == 'profile-root': profile['source_root_binding']['physical_root'] += '-unbound'
    elif fault == 'profile-order': profile['source_descriptors'].reverse()
    elif fault == 'profile-omission': profile['source_descriptors'].pop()
    elif fault == 'namespace-name': namespace['fullname'] = 'ipfs_datasets_py.logic'
    elif fault == 'namespace-path': namespace['path'] += '/nested'
    elif fault == 'boolean-device': namespace['device'] = True
    elif fault == 'zero-inode': namespace['inode'] = 0
    elif fault == 'initializer-present': namespace['initializer_absent'] = False
    else: profile['normal_namespace_profiles'].append(dict(namespace))
    ref = _source_generation_canonical_DATA(case['root']/('bad-profile-'+fault+'.json'), value)
    with pytest.raises(ValueError): executor.prepare_idle_source_replacement(ref)


def test_source_generation_fifteen_preparation_does_not_inherit_five_file_activation_DATA(
        physical_R_fifteen_preparation_DATA):
    case = physical_R_fifteen_preparation_DATA; h = _source_generation_health
    value = {key: case['spec'] for key in executor.SOURCE_GENERATION_SPEC_KEYS}
    value.update(schema=executor.SOURCE_GENERATION_SPEC_SCHEMA,
        authorization='approve_technical_source_generation_without_execution_budget',
        profile_ref=case['value']['profile_ref'], preparation_spec_ref=case['spec'])
    ref = _source_generation_canonical_DATA(case['root']/'unqualified-full-review.json', value)
    with pytest.raises(ValueError, match='source_generation_physical_CLI4_coherent_transition_unqualified'):
        executor._source_generation_review(ref, value['profile_ref'], h._profile(value['profile_ref']), {})


def test_source_generation_fifteen_preparation_preserves_exact_legacy_one_contract_DATA(
        physical_R_fifteen_preparation_DATA, monkeypatch):
    """The same-path five-row @1 result keeps its exact historical field set."""
    case = physical_R_fifteen_preparation_DATA; h = _source_generation_health
    value = case['value']; prior = h._document(value['prior_profile_snapshot_ref'])
    stage = _source_generation_copy.deepcopy(prior); replacements = []
    for row in value['replacements']:
        after = _source_generation_canonical_DATA(Path(row['before_ref']['path']),
                                                  h._bound(row['after_ref']))
        stage['source_descriptors'][row['slot']] = after
        replacements.append(dict(slot=row['slot'], before_ref=row['before_ref'], after_ref=after))
    spec = dict(schema='agent-supervisor/scientific-idle-source-preparation@1',
        prior_profile_ref=value['prior_profile_ref'],
        prior_profile_snapshot_ref=value['prior_profile_snapshot_ref'],
        profile_ref=_source_generation_canonical_DATA(case['root']/'legacy-stage-profile.json', stage),
        replacements=replacements)
    reference = _source_generation_canonical_DATA(case['root']/'legacy-preparation.json', spec)
    def forbidden(*_, **__): raise AssertionError('preparation cannot issue or launch authority')
    monkeypatch.setattr(executor.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(h, '_manager', forbidden)
    comparison = executor.prepare_idle_source_replacement(reference)
    assert comparison == dict(schema='agent-supervisor/scientific-idle-source-comparison@1',
        spec_ref=reference, prior_profile_ref=spec['prior_profile_ref'],
        prior_profile_snapshot_ref=spec['prior_profile_snapshot_ref'], profile_ref=spec['profile_ref'],
        replacements=replacements, source_count=len(stage['source_descriptors']),
        changed_source_count=5, raw_inverse_verified=True, current_source_profile_validated=True,
        coherent_handover_validated=False, config_template_history_unresolved=True,
        runtime_qualified=False, activation_ready=False, dispatch_ready=False,
        hold_removal_authorized=False, authority_granted=False)
    assert not executor._source_generation_path(stage).exists()


@pytest.mark.parametrize('fault', ['legacy-root', 'legacy-relocations', 'legacy-physical-profile',
    'unsupported-two', 'unknown-four', 'physical-missing-root',
    'physical-missing-relocations', 'physical-missing-profile'])
def test_source_generation_fifteen_preparation_keeps_schema_routes_closed_DATA(
        physical_R_fifteen_preparation_DATA, fault):
    case = physical_R_fifteen_preparation_DATA
    value = _source_generation_copy.deepcopy(case['value'])
    extras = {'source_root_binding', 'source_relocations', 'physical_source_profile'}
    if fault.startswith('legacy-') or fault in ('unsupported-two', 'unknown-four'):
        legacy = {key: item for key, item in value.items() if key not in extras}
        if fault.startswith('legacy-'):
            legacy['schema'] = 'agent-supervisor/scientific-idle-source-preparation@1'
            key = {'legacy-root': 'source_root_binding', 'legacy-relocations': 'source_relocations',
                   'legacy-physical-profile': 'physical_source_profile'}[fault]
            legacy[key] = value[key]
        else:
            legacy['schema'] = 'agent-supervisor/scientific-idle-source-preparation@' + (
                '2' if fault == 'unsupported-two' else '4')
        value = legacy
    else:
        value.pop({'physical-missing-root': 'source_root_binding',
                   'physical-missing-relocations': 'source_relocations',
                   'physical-missing-profile': 'physical_source_profile'}[fault])
    reference = _source_generation_canonical_DATA(case['root']/('schema-refusal-'+fault+'.json'), value)
    with pytest.raises(ValueError): executor.prepare_idle_source_replacement(reference)
    assert not executor._source_generation_path(_source_generation_health._profile(case['value']['profile_ref'])).exists()
