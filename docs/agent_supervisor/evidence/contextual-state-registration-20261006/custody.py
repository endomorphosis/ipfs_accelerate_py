"""Bounded cooperative file custody for the retained-state registration driver."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def witness(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def capture(path, *, maximum=128 * 1024 * 1024, retain=False):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path,
            'canonical existing absolute file required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and 0 < before.st_size <= maximum, 'bounded single-link regular file required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    digest, pieces, total = hashlib.sha256(), [], 0
    try:
        require(witness(os.fstat(fd)) == witness(before), 'descriptor changed at open')
        while True:
            raw = os.read(fd, 1024 * 1024)
            if not raw:
                break
            total += len(raw)
            require(total <= maximum, 'file exceeds byte cap during read')
            digest.update(raw)
            if retain:
                pieces.append(raw)
        require(total == before.st_size and witness(os.fstat(fd)) == witness(before),
                'descriptor changed during read')
    finally:
        os.close(fd)
    require(witness(path.lstat()) == witness(before)
            and path.resolve(strict=True) == path, 'path changed during read')
    pin = {'path': str(path), 'bytes': total, 'sha256': digest.hexdigest()}
    return (pin, b''.join(pieces)) if retain else pin


def read(pin):
    actual, raw = capture(pin['path'], retain=True)
    require(actual == pin, 'file differs from exact pin')
    return raw


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return capture(path)


def source(root, relative):
    root = Path(root)
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    pin, raw = capture(root / relative, retain=True)
    committed = subprocess.check_output(['git', '-C', str(root), 'show', head + ':' + relative])
    require(raw == committed, 'source differs from selected committed Git object')
    return {'head': head, 'relative_path': relative, 'pin': pin}
