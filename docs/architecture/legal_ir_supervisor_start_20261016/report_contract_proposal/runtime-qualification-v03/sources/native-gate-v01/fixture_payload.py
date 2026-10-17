"""Explicit non-model payload: no credentials, provider import or network."""
import os
from pathlib import Path
import sys
import time

if len(sys.argv) != 2 or sys.argv[1] != "positive":
    raise SystemExit(64)
effect = Path("/workspace/fixture-started")
descriptor = os.open(effect, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(b"non-model-native-gate-fixture\n")
    stream.flush()
    os.fsync(stream.fileno())
descriptor = os.open(effect.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.write(2, b"non-model-native-gate-fixture:stderr-open\n")
# Exercise monitored stderr waiting, then EOF while the process remains alive
# so the actual timed provider wait must also check original native authority.
time.sleep(1.5)
os.close(2)
time.sleep(1.5)
