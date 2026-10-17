"""Execute actual fixed gate source over in-memory stdin/exec controls only.

No operating-system process, provider, native owner or descriptor is simulated
as genuine. Imports are scoped to this code's local builtins, never global
sys.modules. Exec is a visible inert capture and cannot start any command.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
path = Path(__file__).parent / "proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py"
spec = importlib.util.spec_from_file_location("ordinary_gate_memory_control", path)
H = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = H
spec.loader.exec_module(H)


class InertExecObserved(Exception):
    pass


def execute_gate_memory(stdin, argv, *, readiness=True):
    pending, reads, executions = bytearray(stdin), [], []
    def read(descriptor, maximum):
        assert descriptor == 0 and maximum == 1
        reads.append(maximum)
        raw = bytes(pending[:maximum])
        del pending[:maximum]
        return raw
    def execv(executable, arguments):
        executions.append((executable, list(arguments)))
        raise InertExecObserved()
    controlled = {
        "os": SimpleNamespace(read=read, execv=execv),
        "select": SimpleNamespace(select=lambda *args: ([0], [], []) if readiness else ([], [], [])),
        "sys": SimpleNamespace(argv=argv),
        "time": SimpleNamespace(monotonic=lambda: 0.0),
    }
    def scoped_import(name, *args, **kwargs):
        assert name in controlled, "Gate imports unexpected authority"
        return controlled[name]
    builtins = {"__import__": scoped_import, "bytearray": bytearray, "len": len, "bytes": bytes, "SystemExit": SystemExit}
    result = "unknown"
    try:
        exec(compile(H._START_GATE, "<actual-ordinary-start-gate>", "exec"), {"__builtins__": builtins})
    except InertExecObserved:
        result = "exec_observed_in_memory"
    except SystemExit as error:
        result = error.code
    return result, bytes(pending), reads, executions


class GateMemoryControls(unittest.TestCase):
    def test_exact_marker_execs_exact_original_entrypoint_and_tail(self):
        argv = ["-c", "/usr/bin/env", "-i", "FIXTURE=1", "/inert/model", "-"]
        result, remaining, reads, executions = execute_gate_memory(H._START_MARKER, argv)
        self.assertEqual(result, "exec_observed_in_memory")
        self.assertEqual(executions, [(argv[1], argv[1:])])
        self.assertEqual(remaining, b"")
        self.assertEqual(reads, [1] * len(H._START_MARKER))

    def test_following_prompt_bytes_are_not_buffered_or_consumed(self):
        prompt = b"inert provider prompt bytes\nsecond line\n"
        result, remaining, _reads, _executions = execute_gate_memory(H._START_MARKER + prompt, ["-c", "/inert/entry", "-I"])
        self.assertEqual(result, "exec_observed_in_memory")
        self.assertEqual(remaining, prompt)

    def test_wrong_full_marker_never_execs(self):
        wrong = b"X" + H._START_MARKER[1:]
        result, _remaining, _reads, executions = execute_gate_memory(wrong, ["-c", "/inert/entry"])
        self.assertEqual(result, 125)
        self.assertEqual(executions, [])

    def test_partial_marker_eof_never_execs(self):
        result, _remaining, _reads, executions = execute_gate_memory(H._START_MARKER[:5], ["-c", "/inert/entry"])
        self.assertEqual(result, 125)
        self.assertEqual(executions, [])

    def test_empty_eof_never_execs(self):
        result, _remaining, _reads, executions = execute_gate_memory(b"", ["-c", "/inert/entry"])
        self.assertEqual(result, 125)
        self.assertEqual(executions, [])

    def test_no_readiness_never_reads_or_execs(self):
        result, remaining, reads, executions = execute_gate_memory(H._START_MARKER, ["-c", "/inert/entry"], readiness=False)
        self.assertEqual(result, 125)
        self.assertEqual(remaining, H._START_MARKER)
        self.assertEqual(reads, [])
        self.assertEqual(executions, [])



# These marker-writer controls patch only os.write in a short local scope.
# The inert stream exposes no actual descriptor and cannot execute a process.
from unittest.mock import patch

class PrimaryMarkerWriterControls(unittest.TestCase):
    def test_partial_writes_complete_marker_before_primary_eof(self):
        events, received = [], bytearray()
        stream = SimpleNamespace(fileno=lambda: 987654, close=lambda: events.append("primary_eof"))
        def write(descriptor, raw):
            self.assertEqual(descriptor, 987654)
            count = min(3, len(raw))
            received.extend(raw[:count])
            events.append("marker_write")
            return count
        with patch.object(H.os, "write", side_effect=write):
            H._release_primary_start_and_close_stdin(stream)
        self.assertEqual(bytes(received), H._START_MARKER)
        self.assertEqual(events[-1], "primary_eof")
        self.assertEqual(events.count("primary_eof"), 1)

    def test_zero_write_refuses_without_claiming_start_or_eof(self):
        events = []
        stream = SimpleNamespace(fileno=lambda: 987654, close=lambda: events.append("primary_eof"))
        with patch.object(H.os, "write", return_value=0):
            with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "actual_provider_start_gate_write_failed"):
                H._release_primary_start_and_close_stdin(stream)
        self.assertEqual(events, [])

    def test_write_failure_propagates_and_never_claims_primary_eof(self):
        events = []
        stream = SimpleNamespace(fileno=lambda: 987654, close=lambda: events.append("primary_eof"))
        with patch.object(H.os, "write", side_effect=BrokenPipeError("inert controlled failure")):
            with self.assertRaises(BrokenPipeError):
                H._release_primary_start_and_close_stdin(stream)
        self.assertEqual(events, [])

class MainThreadIOControls(unittest.TestCase):
    """Private I/O primitives over inert memory, not native/process admission."""
    def test_no_context_stderr_preserves_original_read(self):
        events = []
        process = SimpleNamespace(stderr=SimpleNamespace(read=lambda count: events.append(count) or b"default"))
        self.assertEqual(H.ordinary_stderr_chunk(process, 65536), b"default")
        self.assertEqual(events, [65536])

    def test_no_context_wait_preserves_original_wait(self):
        process = SimpleNamespace(wait=lambda: 7)
        self.assertEqual(H.ordinary_provider_wait(process), 7)

    def test_guard_runs_before_ready_read(self):
        events = []
        stream = SimpleNamespace(fileno=lambda: 987654)
        with patch.object(H.select, "select", side_effect=lambda *args: events.append(("select", args[3])) or ([987654], [], [])), \
             patch.object(H.os, "read", side_effect=lambda *args: events.append(("read", args[1])) or b"chunk"):
            self.assertEqual(H._read_periodically_guarded(stream, lambda: events.append("guard"), 65536), b"chunk")
        self.assertEqual(events, ["guard", ("select", 0.5), ("read", 65536)])

    def test_select_timeout_checks_guard_again_before_read(self):
        events, readiness = [], iter([([], [], []), ([987654], [], [])])
        stream = SimpleNamespace(fileno=lambda: 987654)
        with patch.object(H.select, "select", side_effect=lambda *args: next(readiness)), \
             patch.object(H.os, "read", return_value=b"chunk"):
            self.assertEqual(H._read_periodically_guarded(stream, lambda: events.append("guard"), 65536), b"chunk")
        self.assertEqual(events, ["guard", "guard"])

    def test_guard_failure_performs_zero_reads_or_waits(self):
        def refuse():
            raise H.OrdinaryHandoffRefusal("inert-deadline-refusal")
        stream = SimpleNamespace(fileno=lambda: 987654)
        with patch.object(H.select, "select", side_effect=AssertionError("no read readiness on guard failure")), \
             patch.object(H.os, "read", side_effect=AssertionError("no read on guard failure")):
            with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "inert-deadline-refusal"):
                H._read_periodically_guarded(stream, refuse, 65536)
        process = SimpleNamespace(wait=lambda **kwargs: self.fail("no wait on guard failure"))
        with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "inert-deadline-refusal"):
            H._wait_periodically_guarded(process, refuse)

    def test_timed_wait_checks_guard_between_timeouts(self):
        events = []
        def wait(*, timeout):
            events.append(("wait", timeout))
            if len(events) == 2:
                raise H.subprocess.TimeoutExpired(["inert"], timeout)
            return 0
        self.assertEqual(H._wait_periodically_guarded(SimpleNamespace(wait=wait), lambda: events.append("guard")), 0)
        self.assertEqual(events, ["guard", ("wait", 0.5), "guard", ("wait", 0.5)])

if __name__ == "__main__":
    unittest.main()
