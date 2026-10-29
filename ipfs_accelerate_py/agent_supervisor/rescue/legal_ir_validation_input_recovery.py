#!/usr/bin/python3
"""Missing-only repair of a fixed, reviewed 384/768 input pinset. Stdlib only.

No network, model execution, executable installation, board or fleet authority.
The manifest cannot grant authority: its complete value must equal literals
compiled into this reviewed file. initialize/recover default to read-only.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import uuid

SCHEMA = 'literal-validation-input-mirror/v1'
STATE_SCHEMA = 'literal-validation-input-recovery-state/v1'
AUTHORITY = 'protected-replay-born-alive-width-5124ce3433930bf627681bb4fe1742e913f2d3fbbc507141a324744f570e6c4f'
DEFAULT_MIRROR = '/home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1/data/legal-ir-autoencoder-v1/validation-input-mirror'
MAX_PINS, MAX_FILE, MAX_TOTAL = 64, 1024**3, 2 * 1024**3
MAX_JSON, CHUNK = 65536, 1024**2
COOLDOWN, WINDOW, ATTEMPTS = 300, 3600, 4
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
LITERAL_PINS = ({'bytes': 374277662,
  'path': '/home/barberb/.cache/huggingface/hub/datasets--justicedao--ipfs_uscode/blobs/4d26df1e3814279e4b4df3af0e454b4f64fc89a81879db926989862b6ad7d8b8',
  'sha256': '4d26df1e3814279e4b4df3af0e454b4f64fc89a81879db926989862b6ad7d8b8'},
 {'bytes': 394,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/42046bd361a033a24264dc41de1fe6cafd68072f',
  'sha256': 'e1790949631401af1bfb6c9c7aeec7fcf612e274d73579d99f704faea40c8ba7'},
 {'bytes': 57,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/4eca68d85ecd3034cf4174d8a4033a75344ea62d',
  'sha256': '948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529'},
 {'bytes': 66746168,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2',
  'sha256': '9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2'},
 {'bytes': 125,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/a8b3208c2884c4efb86e49300fdd3dc877220cdf',
  'sha256': 'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'},
 {'bytes': 583,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/bcc565b3b20a79e17f048e97717303f26ea833e5',
  'sha256': '00273cdfb3e05c7304d180270bdef8adae45023e7bdfc396338bdd815943dc59'},
 {'bytes': 711661,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/c17ed520ed8438736732a54957a69306b8822215',
  'sha256': 'da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0'},
 {'bytes': 190,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/d1514c3162bbe87b343f565fadc62e6c06f04f03',
  'sha256': '4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23'},
 {'bytes': 385,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/db4787d654117aedec30099876bfa6431eb39827',
  'sha256': '73348a3d8f8ae4cc5f851dc6b98be3ea819390b814e83656d5eebbadf75fe43e'},
 {'bytes': 231508,
  'path': '/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/blobs/fb140275c155a9c7c5a3b3e0e77a9e839594a938',
  'sha256': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'},
 {'bytes': 7127,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/code/configuration.py',
  'sha256': '3411088045ffb8a9a0aa9936eae275896b39983a2ee5b08f091b44e6289e4fe4'},
 {'bytes': 59023,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/code/modeling.py',
  'sha256': '374670b416fcc82f081c9cd28b5fd61c2bd91bbe18eb4798fcc48a81f9c250a0'},
 {'bytes': 1429,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/model/config.json',
  'sha256': '711bdc81365fc25d30533cf05b9fdf588e5ba01f18540fbbb1307d787597a313'},
 {'bytes': 610753338,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/model/model.safetensors',
  'sha256': 'f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0'},
 {'bytes': 964,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/model/special_tokens_map.json',
  'sha256': '8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835'},
 {'bytes': 17082756,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/model/tokenizer.json',
  'sha256': 'f59925fcb90c92b894cb93e51bb9b4a6105c5c249fe54ce1c704420ac39b81af'},
 {'bytes': 1149,
  'path': '/home/barberb/lift_coding/artifacts/autoformalization-alignment-20261003/richer-encoder-assets/model/tokenizer_config.json',
  'sha256': '24cebbf2ef20fc317256e03e52ac7b2ca326586f946a8427ecac036332bf0933'},
 {'bytes': 614534,
  'path': '/home/barberb/lift_coding/artifacts/source-reconstruction-v2-20261001/run-01/legal_ir/raw_ce-1729-checkpoint.json',
  'sha256': '6e3f4d731d798aa2732afc37bd74fec34da3267a323844975d2dab78f59f9c61'},
 {'bytes': 1470,
  'path': '/home/barberb/lift_coding/external/ipfs_datasets/configs/autoencoders/gte_multilingual_local_assets_v1.json',
  'sha256': '8beb874aa7b06599346173fde12e95f9f926b3028942d5014cdd2f99c4385166'},
 {'bytes': 1510086,
  'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/decoder-four-width-20261004/training-r1/results/384/preprocessing.json',
  'sha256': 'de96c5f0a90a1472e346f3732ace42f0b4af4b3915026695fb705ae809663d8f'},
 {'bytes': 2864122,
  'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/decoder-four-width-20261004/training-r1/results/768/preprocessing.json',
  'sha256': 'f4acb7b6b20a38c226a509e331d638fa667c929bd5b8970be44048d549f072ae'},
 {'bytes': 2799098,
  'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/decoder-selected-continuation-20261004/training-r1/results/384-selected-followup-lr0001-1729/selected-state.json',
  'sha256': '0ac5c21656187d1d8040a02bcc0b4716a8093db17f05adc9a22e99d5b6b2cc00'},
 {'bytes': 4781613,
  'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/decoder-selected-continuation-20261004/training-r1/results/768-selected-followup-lr0001-1729/selected-state.json',
  'sha256': '8892a3261c0750a6247ba5400069ed18cad3354426e2095ed1d295287e3559b4'})


class Denied(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Denied(reason)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def path_parts(path):
    require(type(path) is str and path.startswith('/') and path != '/', 'absolute_file_path_required')
    require(not any(ord(c) < 32 or ord(c) == 127 for c in path), 'control_character_path')
    parts = path.split('/')[1:]
    require(all(p and p not in ('.', '..') for p in parts), 'canonical_path_required')
    return parts


def normalize_pins(pins):
    require(type(pins) in (tuple, list) and 0 < len(pins) <= MAX_PINS, 'pin_count')
    result, seen = [], set()
    for row in pins:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'pin_fields')
        path_parts(row['path'])
        require(row['path'] not in seen, 'duplicate_pin_path')
        require(type(row['bytes']) is int and 0 < row['bytes'] <= MAX_FILE, 'pin_byte_bound')
        require(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'pin_digest')
        seen.add(row['path'])
        result.append(dict(row))
    require(sum(r['bytes'] for r in result) <= MAX_TOTAL, 'pin_total_bound')
    return sorted(result, key=lambda r: r['path'])


def identity(info):
    return info.st_dev, info.st_ino


def custody(info):
    return identity(info) + (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


class Directory:
    """Every ancestor is opened without following symlinks; hold exact inode."""
    def __init__(self, path, *, create=False, private=False):
        self.path = path
        parts = path_parts(path)
        fd = os.open('/', DIR_FLAGS)
        self.chain = [identity(os.fstat(fd))]
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                following = os.open(part, DIR_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = following
                self.chain.append(identity(os.fstat(fd)))
            if private:
                info = os.fstat(fd)
                require(info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                        'private_directory_owner_or_mode')
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    def assert_bound(self):
        with Directory(self.path) as current:
            require(current.chain == self.chain, 'directory_custody_changed')


def parent(path):
    parts = path_parts(path)
    return '/' + '/'.join(parts[:-1]), parts[-1]


def ordinary(info):
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'ordinary_unaliased_file_required')


def lstat_at(directory, name):
    try:
        return os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def check_deadline(deadline):
    require(deadline is None or time.monotonic() < deadline, 'run_deadline_exceeded')


def checked_read(directory, name, expected=None, *, maximum=MAX_JSON, output=None, deadline=None):
    """Bounded read/hash through held descriptors; never block opening a FIFO."""
    fd = os.open(name, READ_FLAGS, dir_fd=directory.fd)
    try:
        before = os.fstat(fd)
        ordinary(before)
        size = expected['bytes'] if expected else maximum
        require(before.st_size == size if expected else before.st_size <= maximum, 'input_size_mismatch')
        digest, total, chunks = hashlib.sha256(), 0, []
        while True:
            check_deadline(deadline)
            payload = os.read(fd, min(CHUNK, size + 1 - total))
            if not payload:
                break
            total += len(payload)
            require(total <= size, 'input_exceeds_declared_size')
            digest.update(payload)
            if output is not None:
                view = memoryview(payload)
                while view:
                    view = view[os.write(output, view):]
            elif expected is None:
                chunks.append(payload)
        require(custody(os.fstat(fd)) == custody(before), 'input_changed_during_read')
        after_path = os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
        require(custody(after_path) == custody(before), 'input_path_rebound')
        directory.assert_bound()
        if expected:
            require(total == expected['bytes'] and digest.hexdigest() == expected['sha256'], 'input_digest_mismatch')
        return b''.join(chunks)
    finally:
        os.close(fd)


def strict_json(payload):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    try:
        return json.loads(payload.decode('utf-8'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Denied('nonfinite_json')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise Denied('invalid_json') from exc


def remove_owned(directory, name, ino):
    """Only remove the staging/published inode we created, never a replacement."""
    current = lstat_at(directory, name)
    if current is not None and identity(current) == ino:
        os.unlink(name, dir_fd=directory.fd)
        os.fsync(directory.fd)


def publish(directory, name, *, payload=None, source=None, expected=None, deadline=None, replace=False):
    """Anonymous stage + atomic no-clobber link; no SIGKILL hardlink orphan.

    Linux O_TMPFILE and /proc/self/fd linkat are required and fail closed if
    unavailable. Only mutable retry state uses named staging + atomic replace.
    """
    directory.assert_bound()
    old = lstat_at(directory, name)
    if old is not None:
        ordinary(old)
        require(replace, 'destination_present')
    temp = '.input-recovery-' + uuid.uuid4().hex if replace else None
    if replace:
        fd = os.open(temp, CREATE_FLAGS, 0o600, dir_fd=directory.fd)
    else:
        require(hasattr(os, 'O_TMPFILE'), 'anonymous_staging_unavailable')
        fd = os.open('.', os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o600, dir_fd=directory.fd)
    ino = identity(os.fstat(fd))
    published = False
    try:
        if source is not None:
            checked_read(source[0], source[1], expected, output=fd, deadline=deadline)
        else:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        os.fchmod(fd, 0o600 if replace else 0o444)
        os.fsync(fd)
        directory.assert_bound()
        require(os.fstat(fd).st_nlink == (1 if replace else 0), 'staging_inode_aliased')
        if replace:
            require(identity(os.stat(temp, dir_fd=directory.fd, follow_symlinks=False)) == ino,
                    'staging_inode_rebound')
        now = lstat_at(directory, name)
        if replace:
            require((now is None and old is None) or
                    (now is not None and old is not None and custody(now) == custody(old)),
                    'state_changed_before_replace')
            os.replace(temp, name, src_dir_fd=directory.fd, dst_dir_fd=directory.fd)
            published = True
        else:
            # The source is this process's still-held anonymous FD, not an
            # arbitrary symlink. linkat follows that procfs FD atomically.
            os.link('/proc/self/fd/' + str(fd), name, dst_dir_fd=directory.fd,
                    follow_symlinks=True)
            published = True
        os.fsync(directory.fd)
        directory.assert_bound()
        require(identity(os.stat(name, dir_fd=directory.fd, follow_symlinks=False)) == ino,
                'published_inode_rebound')
    except BaseException:
        # Never unlink a canonical destination during rollback. POSIX has no
        # inode-conditional unlink; a peer may replace a published pathname.
        # Exact published bytes or a committed retry reservation remain safe
        # after a post-link/fsync error. The next run is idempotent.
        raise
    finally:
        os.close(fd)
        if temp is not None:
            remove_owned(directory, temp, ino)


class Recovery:
    def __init__(self, mirror=DEFAULT_MIRROR, pins=None):
        # pins is a Python-only fixture seam, never CLI/config authority.
        self.pins = normalize_pins(LITERAL_PINS if pins is None else pins)
        path_parts(mirror)
        self.mirror = mirror
        require(all(not (r['path'] == mirror or r['path'].startswith(mirror + '/') or
                         mirror.startswith(r['path'] + '/')) for r in self.pins), 'mirror_overlaps_input')
        self.manifest = {'schema': SCHEMA, 'authority': AUTHORITY, 'pins': self.pins}

    def load_manifest(self):
        with Directory(self.mirror, private=True) as root:
            actual = strict_json(checked_read(root, 'manifest.json'))
            require(type(actual) is dict and set(actual) == {'schema', 'authority', 'pins'}, 'manifest_fields')
            normalize_pins(actual['pins'])
            require(actual == self.manifest, 'manifest_not_exact_literal_authority')

    def initialize(self, apply=False, *, deadline_seconds=120):
        require(type(deadline_seconds) is int and 0 < deadline_seconds <= 120, 'run_time_budget')
        deadline = time.monotonic() + deadline_seconds
        report = {'action': 'initialize', 'apply': apply, 'pins': len(self.pins), 'objects': []}
        if not apply:
            for row in self.pins:
                path, name = parent(row['path'])
                with Directory(path) as source:
                    checked_read(source, name, row, deadline=deadline)
            report['status'] = 'sources_verified_no_writes'
            return report
        with Directory(self.mirror, create=True, private=True) as root:
            require(lstat_at(root, 'manifest.json') is None, 'mirror_already_initialized')
        with Directory(self.mirror + '/objects', create=True, private=True) as objects:
            for row in self.pins:
                path, name = parent(row['path'])
                with Directory(path) as source:
                    # Always verify the canonical source, even when deduplicating.
                    checked_read(source, name, row, deadline=deadline)
                    if lstat_at(objects, row['sha256']) is None:
                        try:
                            publish(objects, row['sha256'], source=(source, name), expected=row, deadline=deadline)
                        except FileExistsError:
                            pass  # Concurrent initializer may publish identical content.
                    checked_read(objects, row['sha256'], row, deadline=deadline)
                report['objects'].append(row['sha256'])
        with Directory(self.mirror, private=True) as root:
            try:
                publish(root, 'manifest.json', payload=wire(self.manifest))
            except FileExistsError:
                self.load_manifest()
        report['status'] = 'verified_snapshot_initialized'
        return report

    def load_state(self, state):
        if lstat_at(state, 'attempts.json') is None:
            return {'schema': STATE_SCHEMA, 'attempts': {}}
        value = strict_json(checked_read(state, 'attempts.json'))
        require(type(value) is dict and set(value) == {'schema', 'attempts'} and
                value['schema'] == STATE_SCHEMA and type(value['attempts']) is dict, 'state_schema')
        allowed = {r['path'] for r in self.pins}
        require(set(value['attempts']) <= allowed, 'state_unknown_path')
        for attempts in value['attempts'].values():
            require(type(attempts) is list and len(attempts) <= ATTEMPTS and
                    all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in attempts),
                    'state_attempts')
        return value

    def recover(self, apply=False, *, budget_bytes=MAX_FILE, max_restores=8, deadline_seconds=90, now=None):
        require(type(budget_bytes) is int and 0 < budget_bytes <= MAX_TOTAL, 'run_byte_budget')
        require(type(max_restores) is int and 0 < max_restores <= MAX_PINS, 'run_file_budget')
        require(type(deadline_seconds) is int and 0 < deadline_seconds <= 120, 'run_time_budget')
        self.load_manifest()  # No filesystem mutations before complete authentication.
        if not apply:
            return self._recover(None, None, False, budget_bytes, max_restores, deadline_seconds, now)
        with Directory(self.mirror + '/state', create=True, private=True) as state:
            lock = os.open('recovery.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                           0o600, dir_fd=state.fd)
            try:
                ordinary(os.fstat(lock))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return {'action': 'recover', 'apply': True, 'status': 'busy', 'results': []}
                state.assert_bound()
                require(identity(os.fstat(lock)) == identity(os.stat('recovery.lock', dir_fd=state.fd,
                         follow_symlinks=False)), 'lock_inode_rebound')
                attempts = self.load_state(state)
                return self._recover(state, attempts, True, budget_bytes, max_restores, deadline_seconds, now)
            finally:
                os.close(lock)

    def _recover(self, state, attempts, apply, budget, maximum, duration, now):
        stamp = time.time() if now is None else now
        require(type(stamp) in (float, int) and math.isfinite(stamp) and stamp >= 0, 'invalid_clock')
        deadline = time.monotonic() + duration
        report = {'action': 'recover', 'apply': apply, 'status': 'complete', 'results': [],
                  'restored_bytes': 0, 'restored_files': 0,
                  'attempted_bytes': 0, 'attempted_files': 0,
                  'present_files_sha_verified': False}
        with Directory(self.mirror + '/objects', private=True) as objects:
            for row in self.pins:
                item = {'path': row['path']}
                report['results'].append(item)
                try:
                    check_deadline(deadline)
                    source_path, name = parent(row['path'])
                    with Directory(source_path) as destination:
                        info = lstat_at(destination, name)
                        if info is not None:
                            ordinary(info)
                            item['status'] = 'present_untouched'
                            continue
                        if apply:
                            prior = [t for t in attempts['attempts'].get(row['path'], []) if t >= stamp - WINDOW]
                            if len(prior) >= ATTEMPTS or (prior and max(prior) > stamp - COOLDOWN):
                                item['status'] = 'cooldown_deferred'
                                continue
                        if report['attempted_files'] >= maximum or report['attempted_bytes'] + row['bytes'] > budget:
                            item['status'] = 'run_budget_deferred'
                            continue
                        report['attempted_files'] += 1
                        report['attempted_bytes'] += row['bytes']
                        if not apply:
                            checked_read(objects, row['sha256'], row, deadline=deadline)
                            item['status'] = 'verified_missing_restore_available'
                            continue
                        attempts['attempts'][row['path']] = prior + [stamp]
                        # Reserve BEFORE backup I/O or publishing. Corrupt bytes,
                        # contention, or a crash all consume a bounded attempt.
                        publish(state, 'attempts.json', payload=wire(attempts), replace=True)
                        # Copy+hash once into anonymous staging; only verified
                        # exact bytes may reach the atomic no-clobber link.
                        publish(destination, name, source=(objects, row['sha256']), expected=row, deadline=deadline)
                        report['restored_files'] += 1
                        report['restored_bytes'] += row['bytes']
                        item['status'] = 'restored_exact_missing_input'
                except (Denied, OSError) as exc:
                    item['status'] = 'refused'
                    item['reason'] = str(exc) if isinstance(exc, Denied) else errno.errorcode.get(exc.errno, 'OSERROR')
        unresolved = [item['path'] for item in report['results']
                      if item['status'] not in ('present_untouched', 'restored_exact_missing_input')]
        report['unresolved_paths'] = unresolved
        report['unresolved_inputs'] = len(unresolved)
        if unresolved:
            report['status'] = 'attention_required'
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'recover'))
    parser.add_argument('--mirror', default=DEFAULT_MIRROR)
    parser.add_argument('--apply', action='store_true', help='perform only the reviewed local filesystem operation')
    parser.add_argument('--budget-bytes', type=int, default=MAX_FILE)
    parser.add_argument('--max-restores', type=int, default=8)
    parser.add_argument('--deadline-seconds', type=int, default=90)
    args = parser.parse_args(argv)
    try:
        recovery = Recovery(args.mirror)
        if args.action == 'initialize':
            result = recovery.initialize(args.apply, deadline_seconds=args.deadline_seconds)
        else:
            result = recovery.recover(args.apply, budget_bytes=args.budget_bytes,
                                      max_restores=args.max_restores, deadline_seconds=args.deadline_seconds)
        print(wire(result).decode(), end='')
        return 1 if result.get('status') == 'attention_required' else 0
    except (Denied, OSError) as exc:
        print(wire({'status': 'refused', 'reason': str(exc) if isinstance(exc, Denied)
                    else errno.errorcode.get(exc.errno, 'OSERROR')}).decode(), end='')
        return 2


if __name__ == '__main__':
    sys.exit(main())
