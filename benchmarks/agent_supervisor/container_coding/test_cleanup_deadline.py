"""Exercise real POSIX alarms across the work-to-STOP transition."""
import subprocess
import sys


def test_work_alarm_is_replaced_and_cleanup_stays_bounded():
    script = '''
import signal
import time
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _arm_cleanup_deadline

class Deadline(Exception):
    pass
def expired(signum, frame):
    raise Deadline()
signal.signal(signal.SIGALRM, expired)
signal.setitimer(signal.ITIMER_REAL, .02)
try:
    # The two-second reporting reserve leaves a real 0.3-second STOP window.
    _arm_cleanup_deadline(time.monotonic() + 2.3)
    time.sleep(.06)
    print("stop-passed-work-deadline", flush=True)
    try:
        time.sleep(2)
    except Deadline:
        print("cleanup-deadline-enforced", flush=True)
    else:
        raise AssertionError("cleanup was unbounded")
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["stop-passed-work-deadline", "cleanup-deadline-enforced"]


def test_cleanup_gets_its_window_after_work_deadline_has_already_fired():
    script = '''
import signal
import time
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _arm_cleanup_deadline

class Deadline(Exception):
    pass
def expired(signum, frame):
    signal.setitimer(signal.ITIMER_REAL, 0)
    raise Deadline()
signal.signal(signal.SIGALRM, expired)
signal.setitimer(signal.ITIMER_REAL, .02)
try:
    try:
        time.sleep(2)
    finally:
        _arm_cleanup_deadline(time.monotonic() + 3)
        time.sleep(.05)
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 1
        print("stop-finished-during-unwind", flush=True)
except Deadline:
    print("original-work-timeout-retained", flush=True)
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["stop-finished-during-unwind", "original-work-timeout-retained"]
