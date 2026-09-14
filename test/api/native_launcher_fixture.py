"""Disposable process tree, delayed launch and blocked unknown-child fixture."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

root, role, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert str(root).startswith('/tmp/') and root.name.startswith('test_')
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

if role == 'master':
    children = [subprocess.Popen([sys.executable, '-I', '-S', __file__, str(root), str(i), mode,
                                 '--state-dir',str(root),'--state-prefix',f'fixture{i}',
                                 '--task-shard-index',str(i)],
                                 start_new_session=True) for i in range(4)]
    for i,child in enumerate(children):(root/f'fixture{i}_supervisor.pid').write_text(str(child.pid))
    (root/'roster.json').write_text(json.dumps({'master':os.getpid(), 'wrappers':[p.pid for p in children]}))
    try:
        while True: time.sleep(.01)
    finally:
        for child in children: child.wait(timeout=5)
        (root/'master-closed').touch()
elif role == 'daemon':
    (root/'daemon-ready').touch()
    try:
        while True: time.sleep(.01)
    finally:
        (root/'daemon-closed').touch()
elif role == 'pipe':
    (root/'daemon-ready').touch()
    os.write(1, b'x' * (2 * 1024 * 1024))
else:
    index = int(role)
    descriptor = None
    children = []
    try:
        if index == 2:
            path=(root/'.fixture2_supervisor.lock.update.lock' if (root/'native-fences').exists()
                  else root/'lane2.lock')
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        (root/f'ready-{index}').touch()
        while True:
            if index == 0 and mode != 'never' and (root/'request-peer-progress').exists():
                (root/'peer-progress').touch()
            if index == 2 and descriptor is not None and (root/'peer-progress').exists():
                child = subprocess.Popen([sys.executable, '-I', '-S', __file__, str(root),
                                          'pipe' if mode == 'pipe' else 'daemon', mode],
                                         start_new_session=True,
                                         stdout=subprocess.PIPE if mode == 'pipe' else None)
                children.append(child)
                deadline=time.monotonic()+3
                while not (root/'daemon-ready').exists():
                    assert child.poll() is None and time.monotonic()<deadline
                    time.sleep(.005)
                (root/'daemon.json').write_text(json.dumps({'pid':child.pid, 'parent':os.getpid()}))
                os.close(descriptor)
                descriptor = None
            time.sleep(.01)
    finally:
        if descriptor is not None: os.close(descriptor)
        for child in children: child.wait(timeout=5)
        (root/f'wrapper-{index}-closed').touch()
