"""Synchronize a clean detached submodule to its already accepted parent gitlink.

This is checkout maintenance, never task/callback acceptance. The caller supplies
an exact current parent and records durable phases outside the checkout. Existing
Git locks are refused unless an explicitly authorized operator preserves one via
prepare_index_lock; ordinary maintenance never removes an incumbent Git lock.
"""
from __future__ import annotations
import contextlib
import hashlib
import os
from pathlib import Path
import select
import stat
import subprocess
import time

ENV = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

class Refused(RuntimeError):
    pass

def require(condition, reason):
    if not condition:
        raise Refused(reason)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def identity(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns]

@contextlib.contextmanager
def directory(path):
    """Walk the absolute namespace with pinned no-follow directory handles."""
    path = Path(path).absolute()
    require('..' not in path.parts, 'parent_traversal')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                            os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)

def read_regular(path, limit=32 * 1024 * 1024):
    path = Path(path)
    with directory(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK |
                     os.O_CLOEXEC, dir_fd=parent)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
                    'regular_file_bound')
            chunks, size = [], 0
            while True:
                chunk = os.read(fd, min(1024 * 1024, limit + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                require(size <= limit, 'file_grew_over_bound')
            require(identity(before) == identity(os.fstat(fd)) ==
                    identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)),
                    'file_identity_changed')
            return b''.join(chunks), identity(before)
        finally:
            os.close(fd)

def git(repo, *args, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], env=ENV,
                            capture_output=True, timeout=60)
    if check:
        require(result.returncode == 0, 'git_' + args[0] + '_refused')
    return result

def git_text(repo, *args):
    return git(repo, *args).stdout.decode().strip()

def git_index_path(repo):
    result = Path(git_text(repo, 'rev-parse', '--git-path', 'index'))
    return result if result.is_absolute() else (repo / result).resolve()

def _git_transaction_line(process, command, expected):
    process.stdin.write((command + '\n').encode())
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], 60)
    require(bool(ready), 'git_ref_transaction_timeout')
    require(process.stdout.readline().decode().strip() == expected,
            'git_ref_transaction_' + command + '_refused')

def write_external(path, data):
    """Exclusive durable artifact creation within an already admitted directory."""
    with directory(path.parent) as parent:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent)


def detached_at(repo, expected):
    result = git(repo, 'symbolic-ref', '-q', 'HEAD', check=False)
    return result.returncode == 1 and git_text(repo, 'rev-parse', 'HEAD') == expected


def _advance_detached_head(repo, *, expected_head, target, expected_index_sha256,
                       archive, label, phase, revalidate, prepare_index_lock=None):
    """Explicit FF: prepared expected-old ref transaction plus an owned index.

    Git owns/refuses competing detached HEAD locks. Our index.lock excludes
    other index writers. A private index keeps canonical staging unchanged until
    the exact target tree is verified. No uncertain lock or partial state is
    guessed away; the journal identifies the last completed effect for review.
    """
    require(detached_at(repo, expected_head), 'detached_head_required')
    with directory(archive) as archive_fd:
        require(not Path(os.readlink(f'/proc/self/fd/{archive_fd}')).is_relative_to(repo.resolve()),
                'archive_inside_checkout')
    revalidate()
    git(repo, 'merge-base', '--is-ancestor', expected_head, target)
    index = git_index_path(repo)
    lock_path = index.with_name(index.name + '.lock')
    process = None
    committed = False
    index_published = False
    index_publication_durable = False
    effects_started = False
    lock_owned = None
    alternate = archive / (label + '-prepared.index')
    with directory(index.parent) as parent:
        lock_fd = None
        try:
            if prepare_index_lock is not None:
                require(detached_at(repo, expected_head),
                        'git_ref_changed_before_quarantine')
                require(digest(read_regular(index)[0]) == expected_index_sha256,
                        'git_index_changed_before_quarantine')
                process = subprocess.Popen(['git', '-C', str(repo), 'update-ref', '--stdin',
                                            '-m', 'accepted submodule fast-forward'],
                                           env=ENV, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           bufsize=0)
                _git_transaction_line(process, 'start', 'start: ok')
                process.stdin.write(f'option no-deref\nupdate HEAD {target} {expected_head}\n'.encode())
                process.stdin.flush()
                _git_transaction_line(process, 'prepare', 'prepare: ok')
                phase(label + '_ref_prepared_before_lock_quarantine', repository=str(repo),
                      old=expected_head, target=target)
                require(detached_at(repo, expected_head), 'git_head_changed_before_quarantine')
                revalidate()
                prepare_index_lock(lock_path, archive, phase)
            lock_fd = os.open(lock_path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                              os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            lock_owned = identity(os.fstat(lock_fd))[:2]
            require(detached_at(repo, expected_head),
                    'git_ref_changed_before_prepare')
            raw, _ = read_regular(index)
            require(digest(raw) == expected_index_sha256, 'git_index_changed_before_prepare')
            write_external(archive / (label + '-original.index'), raw)
            write_external(alternate, raw)
            if process is None:
                process = subprocess.Popen(['git', '-C', str(repo), 'update-ref', '--stdin',
                                            '-m', 'accepted submodule fast-forward'],
                                           env=ENV, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           bufsize=0)
                _git_transaction_line(process, 'start', 'start: ok')
                process.stdin.write(f'option no-deref\nupdate HEAD {target} {expected_head}\n'.encode())
                process.stdin.flush()
                _git_transaction_line(process, 'prepare', 'prepare: ok')
            require(detached_at(repo, expected_head),
                    'git_ref_changed_after_prepare')
            phase(label + '_ref_prepared', repository=str(repo), old=expected_head,
                  target=target, alternate_index=str(alternate))
            env = dict(ENV, GIT_INDEX_FILE=str(alternate))
            command = ['read-tree', '-m', '-u', expected_head, target]
            revalidate()
            effects_started = True
            result = subprocess.run(['git', '-C', str(repo), '-c',
                                     'submodule.recurse=false', *command], env=env,
                                    capture_output=True, timeout=120)
            require(result.returncode == 0, 'git_private_index_transition_refused')
            result = subprocess.run(['git', '-C', str(repo), 'write-tree'], env=env,
                                    capture_output=True, timeout=60)
            require(result.returncode == 0 and result.stdout.decode().strip() ==
                    git_text(repo, 'rev-parse', target + '^{tree}'),
                    'prepared_index_not_exact_target')
            require(digest(read_regular(index)[0]) == expected_index_sha256,
                    'canonical_index_changed_during_prepare')
            require(detached_at(repo, expected_head),
                    'git_ref_changed_during_prepare')
            revalidate()
            prepared, _ = read_regular(alternate)
            index_temporary = index.name + '.accepted-sync-' + str(time.time_ns())
            prepared_fd = os.open(index_temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                  os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            with os.fdopen(prepared_fd, 'wb') as stream:
                stream.write(prepared)
                stream.flush()
                os.fsync(stream.fileno())
            require(identity(os.stat(lock_path.name, dir_fd=parent,
                                     follow_symlinks=False))[:2] == lock_owned,
                    'owned_index_lock_replaced')
            # Keep index.lock present across BOTH index and ref publication.
            os.replace(index_temporary, index.name, src_dir_fd=parent, dst_dir_fd=parent)
            index_published = True
            os.fsync(parent)
            index_publication_durable = True
            phase(label + '_index_published', index_sha256=digest(prepared))
            _git_transaction_line(process, 'commit', 'commit: ok')
            committed = True
            phase(label + '_ref_committed', head=target)
            require(detached_at(repo, target),
                    'git_final_ref_mismatch')
        finally:
            if process is not None:
                if not committed and process.poll() is None:
                    try:
                        _git_transaction_line(process, 'abort', 'abort: ok')
                    except (OSError, Refused):
                        pass
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Do not guess-release locks of a still-running Git process.
                    phase(label + '_git_cleanup_uncertain', git_pid=process.pid)
                if process.stdout is not None:
                    process.stdout.close()
            if lock_fd is not None:
                os.close(lock_fd)
            if lock_owned is not None and (committed or not effects_started):
                # Only a proven owned lock is removed. Worktree/index effects
                # remain recorded; no rollback is inferred from an exception.
                try:
                    if identity(os.stat(lock_path.name, dir_fd=parent,
                                        follow_symlinks=False))[:2] == lock_owned:
                        os.unlink(lock_path.name, dir_fd=parent)
                        os.fsync(parent)
                except FileNotFoundError:
                    pass
            if not committed:
                phase(label + '_incomplete_preserved', index_published=index_published,
                      index_publication_durable=index_publication_durable,
                      owned_index_lock_retained=effects_started, index_lock=str(lock_path),
                      alternate_index=str(alternate), automatic_rollback=False)

def accepted_submodule_snapshot(repo_root, relative_path, *, expected_parent, expected_old):
    """Observe accepted Git authority only; no provider/task result is admitted."""
    repo_root = Path(repo_root).absolute()
    relative = Path(relative_path)
    require(not relative.is_absolute() and '..' not in relative.parts and
            str(relative) not in ('', '.'), 'submodule_path_invalid')
    submodule = repo_root / relative
    with directory(submodule):
        pass
    require(git_text(repo_root, 'rev-parse', 'HEAD') == expected_parent,
            'accepted_parent_changed')
    row = git_text(repo_root, 'ls-tree', expected_parent, '--', str(relative)).split()
    require(len(row) == 4 and row[:2] == ['160000', 'commit'] and
            row[3] == str(relative), 'accepted_gitlink_required')
    target = row[2]
    require(detached_at(submodule, expected_old), 'detached_source_changed')
    require(not git_text(submodule, 'status', '--porcelain=v1', '--untracked-files=all'),
            'submodule_checkout_dirty')
    git(submodule, 'merge-base', '--is-ancestor', expected_old, target)
    # Parent staging and unrelated source must remain exactly accepted. The one
    # admitted dirty status is the checkout/gitlink mismatch being corrected.
    status = git_text(repo_root, 'status', '--porcelain=v1', '--untracked-files=all')
    require(status in ('', 'M ' + str(relative)), 'parent_checkout_dirty')
    require(git_text(repo_root, 'diff', '--cached', '--name-only') == '',
            'parent_staging_changed')
    index = git_index_path(submodule)
    return {'parent': expected_parent, 'relative_path': str(relative), 'old': expected_old,
            'target': target, 'parent_index_sha256': digest(read_regular(git_index_path(repo_root))[0]),
            'index_sha256': digest(read_regular(index)[0]),
            'head_sha256': digest(read_regular(Path(git_text(submodule, 'rev-parse', '--absolute-git-dir')) / 'HEAD')[0])}


def synchronize_accepted_submodule(repo_root, relative_path, *, expected_parent,
                                   expected_old, archive, phase, custody_guard,
                                   prepare_index_lock=None):
    """Native maintenance entry under the canonical main-checkout merge lease.

    This is an internal trusted-native-caller API: archive must be a private,
    external durable directory; phase must durably record transitions; and the
    caller must already hold its configured native owner/source admission locks.
    No caller-filled object is a task or callback authority proof.

    custody_guard must freshly deny current work in the canonical submodule;
    isolated provider worktrees are separate. This entry never steals leases or
    quarantines a Git lock by default. Callers retain their own native owner and
    source admission locks. A failure after worktree effects preserves staging
    custody and explicit journal evidence; automatic rollback is forbidden.
    """
    from .checkout_lock import (acquire_checkout_mutation_lease,
                                checkout_lock_metadata, checkout_mutation_lock_path,
                                read_checkout_mutation_lease, release_checkout_mutation_lease)
    repo_root, archive = Path(repo_root), Path(archive)
    initial = accepted_submodule_snapshot(repo_root, relative_path,
                                         expected_parent=expected_parent,
                                         expected_old=expected_old)
    custody_guard()
    metadata = checkout_lock_metadata(kind='merge', repo_root=repo_root,
                                      extra={'operation': 'accepted-submodule-sync'})
    lease, _, _, _ = acquire_checkout_mutation_lease(
        checkout_mutation_lock_path(repo_root), metadata, owner_active=lambda _: True)
    require(lease is not None, 'native_checkout_lease_contended')
    try:
        def revalidate():
            current = read_checkout_mutation_lease(lease.lock_path)
            require(current is not None and current.lease_id == lease.lease_id and
                    (current.device, current.inode) == (lease.device, lease.inode),
                    'native_checkout_lease_changed')
            require(git_text(repo_root, 'rev-parse', 'HEAD') == expected_parent and
                    digest(read_regular(git_index_path(repo_root))[0]) == initial['parent_index_sha256'],
                    'accepted_parent_or_index_changed')
            custody_guard()

        require(accepted_submodule_snapshot(repo_root, relative_path,
                                           expected_parent=expected_parent,
                                           expected_old=expected_old) == initial,
                'accepted_source_changed_under_lease')
        phase('native_checkout_lease_acquired', lease_id=lease.lease_id,
              snapshot=initial, callback_closure_asserted=False)
        if initial['old'] != initial['target']:
            _advance_detached_head(repo_root / relative_path, expected_head=expected_old,
                                   target=initial['target'], expected_index_sha256=initial['index_sha256'],
                                   archive=archive, label='submodule', phase=phase,
                                   revalidate=revalidate, prepare_index_lock=prepare_index_lock)
        revalidate()
        final = accepted_submodule_snapshot(repo_root, relative_path,
                                           expected_parent=expected_parent,
                                           expected_old=initial['target'])
        phase('accepted_submodule_synchronized', snapshot=final,
              callback_closure_asserted=False)
        return final
    finally:
        require(release_checkout_mutation_lease(lease), 'native_checkout_lease_release_uncertain')
