"""Normal owning regressions for objective-refill timeout enforcement.

Use the maintained supervisor class and declared error directly. The fixture
allocates the real class without running its service initializer, then replaces
only its signal/threading globals with the existing fakes. unittest cleanup
restores both globals after every test, including failed assertions.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as _implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    ObjectiveRefillTimeoutError,
    PortalImplementationSupervisor,
)


class FakeThreading:
    def __init__(self, *, main=True):
        self.main_token = object()
        self.worker_token = object()
        self.main = main

    def main_thread(self):
        return self.main_token

    def current_thread(self):
        return self.main_token if self.main else self.worker_token


class FakeSignal:
    SIGALRM = object()
    ITIMER_REAL = object()

    def __init__(self, *, pending=False):
        self.old_handler = object()
        self.handler = self.old_handler
        self.timer = (7.5, 2.0) if pending else (0.0, 0.0)
        self.initial_timer = self.timer
        self.events = []

    def getitimer(self, which):
        assert which is self.ITIMER_REAL
        self.events.append(('getitimer', which))
        return self.timer

    def getsignal(self, which):
        assert which is self.SIGALRM
        self.events.append(('getsignal', which))
        return self.handler

    def signal(self, which, handler):
        assert which is self.SIGALRM
        self.events.append(('signal', which, handler))
        old, self.handler = self.handler, handler
        return old

    def setitimer(self, which, seconds):
        assert which is self.ITIMER_REAL
        self.events.append(('setitimer', which, seconds))
        old, self.timer = self.timer, (float(seconds), 0.0)
        return old


class MissingSignalAttribute:
    def __init__(self, fake, missing):
        self.fake = fake
        self.missing = missing

    def __getattr__(self, name):
        if name == self.missing:
            raise AttributeError(name)
        return getattr(self.fake, name)


class TimeoutContractMixin:
    """Provider returns (subject with config, ObjectiveRefillTimeoutError type)."""

    def build_subject(self, *, signal, threading, timeout):
        raise NotImplementedError

    def harness(self, *, timeout=1.25, main=True, missing=None, pending=False):
        fake = FakeSignal(pending=pending)
        exposed = MissingSignalAttribute(fake, missing) if missing else fake
        subject, timeout_error = self.build_subject(
            signal=exposed, threading=FakeThreading(main=main), timeout=timeout
        )
        return subject, fake, timeout_error

    def assert_cleaned(self, fake):
        self.assertIs(fake.handler, fake.old_handler)
        self.assertEqual(fake.timer, (0.0, 0.0))
        self.assertEqual(
            [event[2] for event in fake.events if event[0] == 'setitimer'],
            [1.25, 0.0],
        )
        self.assertEqual(fake.events[-2], ('setitimer', fake.ITIMER_REAL, 0.0))
        self.assertEqual(fake.events[-1], ('signal', fake.SIGALRM, fake.old_handler))

    def assert_refused(self, *, main=True, missing=None, pending=False):
        subject, fake, _ = self.harness(main=main, missing=missing, pending=pending)
        calls = []
        args = object()
        with self.assertRaises(RuntimeError):
            subject._run_objective_refill_with_timeout(lambda value: calls.append(value), args)
        self.assertEqual(calls, [], 'a positive objective budget must refuse before callback')
        self.assertIs(fake.handler, fake.old_handler)
        self.assertEqual(fake.timer, fake.initial_timer)
        self.assertFalse(any(event[0] in {'signal', 'setitimer'} for event in fake.events))

    def test_objective_refuses_positive_timeout_off_main_thread(self):
        self.assert_refused(main=False)

    def test_objective_refuses_positive_timeout_without_alarm_api(self):
        for missing in ('setitimer', 'SIGALRM'):
            with self.subTest(missing=missing):
                self.assert_refused(missing=missing)

    def test_objective_refuses_positive_timeout_when_timer_is_pending(self):
        self.assert_refused(pending=True)

    def test_supported_objective_preserves_return_identity_and_cleans_up(self):
        subject, fake, _ = self.harness()
        args, result = object(), object()
        calls = []

        def callback(value):
            calls.append(value)
            self.assertIsNot(fake.handler, fake.old_handler)
            self.assertEqual(fake.timer, (1.25, 0.0))
            return result

        self.assertIs(subject._run_objective_refill_with_timeout(callback, args), result)
        self.assertEqual(calls, [args])
        self.assert_cleaned(fake)

    def test_supported_objective_preserves_callback_exception_identity_and_cleanup(self):
        subject, fake, _ = self.harness()
        failure = LookupError('exact callback failure')
        calls = []

        def callback(value):
            calls.append(value)
            raise failure

        args = object()
        with self.assertRaises(LookupError) as raised:
            subject._run_objective_refill_with_timeout(callback, args)
        self.assertIs(raised.exception, failure)
        self.assertEqual(calls, [args])
        self.assert_cleaned(fake)

    def test_supported_objective_alarm_uses_declared_error_and_cleans_up(self):
        subject, fake, timeout_error = self.harness()

        def callback(_args):
            fake.handler(fake.SIGALRM, None)

        with self.assertRaises(timeout_error) as raised:
            subject._run_objective_refill_with_timeout(callback, object())
        self.assertIs(type(raised.exception), timeout_error)
        self.assertEqual(str(raised.exception), 'objective refill exceeded 1.250s')
        self.assert_cleaned(fake)

    def test_disabled_objective_budget_keeps_direct_callback(self):
        contexts = ({'main': False}, {'missing': 'setitimer'}, {'missing': 'SIGALRM'}, {'pending': True})
        for timeout in (0.0, -1.0, None):
            for context in contexts:
                with self.subTest(timeout=timeout, context=context):
                    subject, fake, _ = self.harness(timeout=timeout, **context)
                    args, result = object(), object()
                    calls = []

                    def callback(value):
                        calls.append(value)
                        return result

                    self.assertIs(subject._run_objective_refill_with_timeout(callback, args), result)
                    self.assertEqual(calls, [args])
                    self.assertEqual(fake.events, [])
                    self.assertEqual(fake.timer, fake.initial_timer)
                    self.assertIs(fake.handler, fake.old_handler)

    def test_generic_default_keeps_positive_budget_fallback(self):
        contexts = ({'main': False}, {'missing': 'setitimer'}, {'missing': 'SIGALRM'}, {'pending': True})
        for context in contexts:
            with self.subTest(context=context):
                subject, fake, _ = self.harness(**context)
                result = object()
                calls = []

                def callback():
                    calls.append('called')
                    return result

                self.assertIs(subject._run_supervisor_call_with_timeout(
                    phase='unrelated caller', timeout_seconds=1.25,
                    timeout_error=TimeoutError, callback=callback,
                ), result)
                self.assertEqual(calls, ['called'])
                self.assertFalse(any(event[0] in {'signal', 'setitimer'} for event in fake.events))
                self.assertEqual(fake.timer, fake.initial_timer)
                self.assertIs(fake.handler, fake.old_handler)

    def test_generic_supported_budget_preserves_return_and_cleanup(self):
        subject, fake, _ = self.harness()
        result = object()
        self.assertIs(subject._run_supervisor_call_with_timeout(
            phase='unrelated caller', timeout_seconds=1.25,
            timeout_error=TimeoutError, callback=lambda: result,
        ), result)
        self.assert_cleaned(fake)

    def test_generic_supported_budget_preserves_custom_timeout_type(self):
        subject, fake, _ = self.harness()

        class CustomTimeout(TimeoutError):
            pass

        def callback():
            fake.handler(fake.SIGALRM, None)

        with self.assertRaises(CustomTimeout) as raised:
            subject._run_supervisor_call_with_timeout(
                phase='unrelated caller', timeout_seconds=1.25,
                timeout_error=CustomTimeout, callback=callback,
            )
        self.assertIs(type(raised.exception), CustomTimeout)
        self.assertEqual(str(raised.exception), 'unrelated caller exceeded 1.250s')
        self.assert_cleaned(fake)



class TestPortalImplementationSupervisorObjectiveRefillTimeout(
    TimeoutContractMixin, unittest.TestCase
):
    """Exercise ordinary maintained methods with fake alarm/thread state."""

    def build_subject(self, *, signal, threading, timeout):
        self.enterContext(patch.object(_implementation_supervisor, "signal", signal))
        self.enterContext(patch.object(_implementation_supervisor, "threading", threading))
        subject = object.__new__(PortalImplementationSupervisor)
        subject.config = SimpleNamespace(objective_refill_timeout_seconds=timeout)
        return subject, ObjectiveRefillTimeoutError
