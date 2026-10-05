"""Real provider-free replacement of a birth-bound native daemon."""
import signal
import time

from test.integration.test_admitted_benchmark_runtime import admitted  # noqa: F401


def test_native_replacement_receives_fresh_birth_grant_and_stops(admitted):
    runtime, owner, prepared = admitted
    original_task = owner.source.get_task(prepared["task_cid"])
    started = runtime.start()
    assert started.succeeded
    original = runtime.bootstrap_receipts[0]
    tree = runtime.process.snapshot(runtime.profile)
    child = next(member for member in tree.members if member.pid == original["pid"])
    assert runtime.process.identity_alive(child)
    try:
        # Only the independently authenticated disposable fixture daemon.
        runtime.process._signal_exact(child, signal.SIGTERM)
        deadline = time.monotonic() + 40
        while len(runtime.bootstrap_receipts) < 2 and not runtime.bootstrap_errors:
            assert time.monotonic() < deadline, runtime.startup_diagnostics()
            time.sleep(.1)
        assert not runtime.bootstrap_errors, runtime.startup_diagnostics()
        replacement = runtime.bootstrap_receipts[1]
        assert replacement["process_birth_id"] != original["process_birth_id"]
        assert replacement["pid"] != original["pid"]
        assert owner.source.get_task(prepared["task_cid"]) == original_task
        assert runtime.stop().succeeded
        assert not runtime.process.snapshot(runtime.profile).members
        assert all(process.poll() is not None for process in runtime._children)
    finally:
        # A failed assertion must not strand a provider-free test supervisor.
        if runtime.process.snapshot(runtime.profile).members:
            runtime.stop()
        for process in runtime._children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
