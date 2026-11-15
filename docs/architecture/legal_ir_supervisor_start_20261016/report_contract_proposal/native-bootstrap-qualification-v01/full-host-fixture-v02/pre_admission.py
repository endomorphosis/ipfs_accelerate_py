"""Recognize only the original untouched, pre-write capacity refusal.

This is a cleanup classification, never admission. No owner/lease is created,
released, reconstructed or modified here. Unknown outcomes return false.
"""
import ast
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import time

from fixture_io import canonical, need, read, strict

RESOURCE_SHA = 'b9c8ca9f889366df93fd82887b7d6fcbfda9edee12a58a5667b485fe2c8a54ed'
NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
ORIGINAL = Path('/home/barberb/lift_coding/external/ipfs_datasets')
LEDGER = ORIGINAL / 'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'


def _source_lines(resources):
    path = canonical(resources.__file__)
    need(resources.__name__ == NATIVE + 'autoencoder_daemon_resources'
         and resources.__spec__.origin == str(path), 'exact_original_resource_module_required')
    raw = read(path, RESOURCE_SHA)
    tree = ast.parse(raw)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DaemonResourceReservation']
    need(len(classes) == 1, 'original_resource_class_required')
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    calls = [n for n in ast.walk(methods['__enter__']) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'self'
        and n.value.func.attr == '_account'
        and len(n.value.args) == 1 and isinstance(n.value.args[0], ast.Name)
        and n.value.args[0].id == 'ledger'
        and len(n.value.keywords) == 1 and n.value.keywords[0].arg == 'additional'
        and isinstance(n.value.keywords[0].value, ast.Attribute)
        and isinstance(n.value.keywords[0].value.value, ast.Name)
        and n.value.keywords[0].value.value.id == 'self'
        and n.value.keywords[0].value.attr == 'storage_bytes']
    raises = [n for n in ast.walk(methods['_account']) if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Name) and n.exc.id == 'error']
    need(len(calls) == len(raises) == 1, 'exact_original_prewrite_account_call_and_raise_required')
    return calls[0].lineno, raises[0].lineno


def _selected_row_absent(owner):
    """Readonly OS APIs in the original existing flock domain; no private keys.

    The original SDK exposes no standalone public ledger-inspection method.
    This bounded reader observes the selected UUID under the same flock. It
    neither invokes a hidden cleanup API nor infers a productive grant.
    """
    path = canonical(owner.ledger_path)
    need(path == LEDGER, 'exact_original_ledger_required')
    lock = canonical(owner.lock_path)
    need(lock == path.with_name(path.name + '.lock'), 'exact_original_lock_domain_required')
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and before.st_nlink == 1 and before.st_mode & 0o077 == 0,
             'existing_owned_private_original_lock_required')
        until = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    return False
                time.sleep(.05)
        current = lock.lstat()
        need((before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
             'original_lock_inode_changed')
        value = strict(read(path, maximum=8 * 1024**2))
        need(type(value) is dict and set(value) == {'schema', 'roots', 'limit_bytes', 'reservations'}
             and value['schema'] == 'daemon-resource-reservations-v1'
             and value['roots'] == owner.root_identities
             and value['limit_bytes'] == 160000000000
             and type(value['reservations']) is dict and len(value['reservations']) <= 4096,
             'original_strict_ledger_scope_required')
        return owner.reservation_id not in value['reservations']
    finally:
        os.close(descriptor)  # Also drops only this readonly observation lock.


def known_unwritten_capacity_refusal(owner, cause, *, resources_module, parented_module, scheduler):
    try:
        resources, parented = resources_module, parented_module
        need(type(owner) is parented.ParentedPhaseResourceReservation
             and type(cause) is resources.DaemonResourceError
             and str(cause) == 'storage reservation capacity exceeded',
             'exact_original_capacity_refusal_required')
        enter_line, raise_line = _source_lines(resources)
        account_code = resources.DaemonResourceReservation._account.__code__
        enter_code = resources.DaemonResourceReservation.__enter__.__code__
        outer_code = parented.ParentedPhaseResourceReservation.__enter__.__code__
        frames = []
        cursor = cause.__traceback__
        while cursor is not None:
            frames.append(cursor)
            need(len(frames) <= 32, 'bounded_original_refusal_traceback_required')
            cursor = cursor.tb_next
        account = [tb for tb in frames if tb.tb_frame.f_code is account_code
            and tb.tb_frame.f_globals is vars(resources) and tb.tb_lineno == raise_line
            and tb.tb_frame.f_locals.get('self') is owner]
        enter = [tb for tb in frames if tb.tb_frame.f_code is enter_code
            and tb.tb_frame.f_globals is vars(resources) and tb.tb_lineno == enter_line
            and tb.tb_frame.f_locals.get('self') is owner]
        outer = [tb for tb in frames if tb.tb_frame.f_code is outer_code
            and tb.tb_frame.f_globals is vars(parented) and tb.tb_frame.f_locals.get('self') is owner]
        need(len(account) == len(enter) == len(outer) == 1 and frames[-1] is account[0],
             'actual_original_prewrite_call_stack_required')
        local = account[0].tb_frame.f_locals
        need(local.get('error') is cause and local.get('additional') == owner.storage_bytes
             and local.get('ledger') is enter[0].tb_frame.f_locals.get('ledger'),
             'actual_initial_account_objects_required')
        public = owner.to_dict()
        need(public['status'] == 'not_entered' and public['record'] is None
             and public['resource_lease'] is None and owner._entered is False
             and owner._record is None and owner._lease is None
             and owner._child is None and owner._child_history == []
             and owner._owner_pid == os.getpid(), 'no_original_record_child_or_native_grant_required')
        request_id = 'daemon:' + owner.reservation_id  # Original fixed source spelling.
        need(not any(row.get('request_id') == request_id for row in scheduler.active_leases()),
             'selected_original_native_grant_must_be_absent')
        need(_selected_row_absent(owner), 'selected_original_disk_row_must_be_absent')
        return True
    except BaseException:
        return False  # Unknown write/admission/corruption remains retained.
