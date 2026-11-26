"""Existing-thread handoff contract: fictional RPC, retained receipt, no live daemon."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import fcntl
except ImportError:
    fcntl = None
from ipfs_accelerate_py.cli_integrations import openai_codex_cli_integration as mod


class _ContractSocket:
    def __init__(self):
        self.deadlines = []

    def settimeout(self, seconds):
        if not 0 < seconds <= 60:
            raise AssertionError("socket operation lost its bounded deadline")
        self.deadlines.append(seconds)


class _ContractConnection:
    def __init__(self, receipt, *, can_accept=True, current_turn="turn-1", steer_timeout=False):
        self.receipt = receipt
        self.can_accept = can_accept
        self.current_turn = current_turn
        self.steer_timeout = steer_timeout
        self.socket = _ContractSocket()
        self.messages = []
        self.pre_steer_receipt = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        return False

    def send(self, raw):
        request = json.loads(raw)
        self.messages.append(request)
        if request["method"] == "turn/steer":
            # Persisted ambiguity must precede the potentially delivered operation.
            self.pre_steer_receipt = json.loads(self.receipt.read_text())
            if self.pre_steer_receipt["delivery_state"] != "delivery_unknown":
                raise AssertionError("steer happened before durable no-replay reservation")

    def recv(self, timeout):
        if not 0 < timeout <= 60:
            raise AssertionError("receive lost its bounded deadline")
        request = self.messages[-1]
        method = request["method"]
        if method == "turn/steer" and self.steer_timeout:
            raise TimeoutError("fixture: steer may have arrived but reply was lost")
        results = {
            "initialize": {},
            "thread/read": {"thread": {"id": "thread-1", "canAcceptDirectInput": self.can_accept}},
            "thread/turns/list": {"data": [{"id": self.current_turn, "status": "inProgress"}]},
            "turn/steer": {"turnId": "turn-1"},
        }
        return json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": results[method]})

    @property
    def steers(self):
        return [message for message in self.messages if message["method"] == "turn/steer"]


class ThreadHandoffContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codex-main-handoff-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.receipt = self.root / "message.json"
        self.kwargs = dict(socket_path=str(self.root / "daemon.sock"), thread_id="thread-1",
                           expected_turn_id="turn-1", message="bounded legal-work handoff fixture",
                           receipt_path=str(self.receipt), timeout_seconds=2.0, max_attempts=3)

    def call(self, **changes):
        return mod.send_codex_thread_message(**{**self.kwargs, **changes})

    def assert_no_authority(self, result):
        for field in ("processing_confirmed", "execution_granted", "claim_granted", "completion_eligible"):
            self.assertIs(result[field], False, field)

    def test_ack_is_enqueue_only_and_accepted_receipt_does_not_replay(self):
        connection = _ContractConnection(self.receipt)
        with patch.object(mod, "_open_codex_app_connection", return_value=connection) as open_:
            result = self.call()
        self.assertEqual(result["delivery_state"], "accepted")
        self.assertIs(result["enqueue_acknowledged"], True)
        self.assert_no_authority(result)
        self.assertEqual(len(connection.steers), 1)
        self.assertEqual(connection.steers[0]["params"]["expectedTurnId"], "turn-1")
        self.assertEqual(connection.steers[0]["params"]["threadId"], "thread-1")
        self.assertEqual(connection.steers[0]["params"]["input"][0]["text"], self.kwargs["message"])
        self.assertEqual(connection.pre_steer_receipt["delivery_state"], "delivery_unknown")
        self.assertTrue(connection.closed)
        open_.assert_called_once()
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("accepted delivery replayed")) as retry:
            repeated = self.call()
        retry.assert_not_called()
        self.assertEqual(repeated["delivery_state"], "accepted")
        self.assertEqual(repeated["attempts"], result["attempts"])
        self.assert_no_authority(repeated)

    def test_stale_turn_is_refused_before_steering(self):
        connection = _ContractConnection(self.receipt, current_turn="turn-new")
        with patch.object(mod, "_open_codex_app_connection", return_value=connection):
            result = self.call()
        self.assertEqual((result["delivery_state"], result["reason"]), ("refused", "target_turn_changed"))
        self.assertEqual(connection.steers, [])
        self.assertIs(result["enqueue_acknowledged"], False)
        self.assert_no_authority(result)

    def test_target_not_accepting_direct_input_is_never_steered(self):
        connection = _ContractConnection(self.receipt, can_accept=False)
        with patch.object(mod, "_open_codex_app_connection", return_value=connection):
            result = self.call()
        self.assertEqual((result["delivery_state"], result["reason"]), ("refused", "target_not_accepting_input"))
        self.assertEqual(connection.steers, [])
        self.assertIs(result["enqueue_acknowledged"], False)
        self.assert_no_authority(result)

    def test_lost_steer_ack_is_unknown_and_second_call_makes_no_rpc(self):
        connection = _ContractConnection(self.receipt, steer_timeout=True)
        with patch.object(mod, "_open_codex_app_connection", return_value=connection) as first:
            result = self.call()
        first.assert_called_once()
        self.assertEqual(len(connection.steers), 1)
        self.assertEqual(result["delivery_state"], "delivery_unknown")
        self.assertIs(result["retryable"], False)
        self.assertIs(result["enqueue_acknowledged"], False)
        self.assert_no_authority(result)
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("ambiguous delivery replayed")) as second:
            repeated = self.call()
        second.assert_not_called()
        self.assertEqual(repeated["delivery_state"], "delivery_unknown")
        self.assertEqual(repeated["attempts"], result["attempts"])
        self.assert_no_authority(repeated)

    def test_prior_unknown_delivery_never_connects(self):
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("prior uncertain channel replayed")) as open_:
            result = self.call(prior_delivery_state="unknown")
        open_.assert_not_called()
        self.assertEqual(result["delivery_state"], "delivery_unknown")
        self.assertEqual(result["attempts"], 0)
        self.assertIs(result["retryable"], False)
        self.assert_no_authority(result)

    def test_receipt_cannot_be_reused_for_different_payload(self):
        connection = _ContractConnection(self.receipt)
        with patch.object(mod, "_open_codex_app_connection", return_value=connection):
            accepted = self.call()
        self.assertEqual(accepted["delivery_state"], "accepted")
        original = self.receipt.read_bytes()
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("different payload reused receipt")) as open_:
            result = self.call(message="a different fictional handoff")
        open_.assert_not_called()
        self.assertEqual((result["delivery_state"], result["reason"]), ("refused", "receipt_identity_mismatch"))
        self.assertEqual(self.receipt.read_bytes(), original)
        self.assert_no_authority(result)


"""Public-import failure controls; fictional RPC only."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from ipfs_accelerate_py.cli_integrations import openai_codex_cli_integration as mod

class _FailureSocket:

    def settimeout(self, seconds):
        assert 0 < seconds <= 60

class _FailureConnection:

    def __init__(self, *, ack=None, notifications=0):
        self.socket, self.messages = (_FailureSocket(), [])
        self.ack = ack
        self.notifications = notifications

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def send(self, data):
        self.messages.append(json.loads(data))

    def recv(self, timeout):
        assert 0 < timeout <= 60
        if self.notifications:
            self.notifications -= 1
            return '{"method":"notice"}'
        call = self.messages[-1]
        method = call['method']
        values = {'initialize': {}, 'thread/read': {'thread': {'id': 'thread-1', 'canAcceptDirectInput': True}}, 'thread/turns/list': {'data': [{'id': 'turn-1', 'status': 'inProgress'}]}, 'turn/steer': {'turnId': 'turn-1'} if self.ack is None else self.ack}
        return json.dumps({'id': call['id'], 'result': values[method]})

class TransportFailureControls(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='codex-handoff-controls-')
        self.root = Path(self.temp.name)
        self.kw = dict(socket_path=str(self.root / 'daemon.sock'), thread_id='thread-1', expected_turn_id='turn-1', message='bounded fixture handoff', receipt_path=str(self.root / 'message.json'))

    def tearDown(self):
        self.temp.cleanup()

    def call(self, **changes):
        return mod.send_codex_thread_message(**{**self.kw, **changes})

    def test_persisted_attempt_cap_stops_future_connects(self):
        with patch.object(mod, '_open_codex_app_connection', side_effect=OSError('offline')) as open_:
            for expected in (1, 2, 3):
                result = self.call()
                self.assertEqual((result['delivery_state'], result['attempts']), ('not_sent', expected))
            result = self.call()
            self.assertEqual((result['delivery_state'], result['reason']), ('refused', 'attempt_budget_exhausted'))
            self.assertEqual(open_.call_count, 3)

    def test_restart_cannot_raise_persisted_attempt_budget(self):
        with patch.object(mod, '_open_codex_app_connection', side_effect=OSError) as open_:
            self.call(max_attempts=1)
            result = self.call(max_attempts=3)
            self.assertEqual(result['reason'], 'receipt_identity_mismatch')
            self.assertEqual(open_.call_count, 1)

    def test_receipt_cannot_retarget_thread(self):
        with patch.object(mod, '_open_codex_app_connection', side_effect=OSError) as open_:
            self.call()
            self.assertEqual(self.call(thread_id='other')['reason'], 'receipt_identity_mismatch')
            self.assertEqual(open_.call_count, 1)

    def test_invalid_inputs_do_not_create_receipts(self):
        for change in ({'timeout_seconds': True}, {'timeout_seconds': float('nan')}, {'timeout_seconds': 61}, {'max_attempts': True}, {'max_attempts': 9}, {'message': '\ud800'}, {'message': 'x' * 1001}, {'expected_turn_id': 'turn\nother'}, {'prior_delivery_state': 'accepted'}):
            with self.subTest(change=repr(change)):
                self.assertEqual(self.call(**change)['delivery_state'], 'refused')
                self.assertFalse(Path(self.kw['receipt_path']).exists())

    def test_nonprivate_directory_refuses_without_connection(self):
        self.root.chmod(493)
        with patch.object(mod, '_open_codex_app_connection') as open_:
            self.assertEqual(self.call()['reason'], 'receipt_directory_not_private')
            open_.assert_not_called()

    def test_symlink_receipt_never_replaces_target(self):
        target = self.root / 'target.json'
        target.write_text('original')
        Path(self.kw['receipt_path']).symlink_to(target)
        with patch.object(mod, '_open_codex_app_connection') as open_:
            self.assertEqual(self.call()['delivery_state'], 'refused')
            self.assertEqual(target.read_text(), 'original')
            open_.assert_not_called()

    def test_fifo_lock_refuses_without_waiting(self):
        os.mkfifo(self.kw['receipt_path'] + '.lock')
        with patch.object(mod, '_open_codex_app_connection') as open_:
            self.assertEqual(self.call()['reason'], 'receipt_lock_invalid')
            open_.assert_not_called()

    def test_live_receipt_lock_never_taken_over(self):
        fd = os.open(self.kw['receipt_path'] + '.lock', os.O_CREAT | os.O_RDWR, 384)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(mod, '_open_codex_app_connection') as open_:
                self.assertEqual(self.call()['delivery_state'], 'in_progress')
                open_.assert_not_called()
        finally:
            os.close(fd)

    def test_failed_initial_receipt_write_cleans_stage_and_sends_nothing(self):
        with patch.object(mod.os, 'write', side_effect=OSError('fixture disk full')), patch.object(mod, '_open_codex_app_connection') as open_:
            self.assertEqual(self.call()['delivery_state'], 'refused')
            open_.assert_not_called()
        self.assertEqual([p.name for p in self.root.iterdir()], ['message.json.lock'])

    def test_ack_persistence_failure_keeps_durable_unknown(self):
        connection = _FailureConnection()
        replace = mod.os.replace
        count = [0]

        def replacing(*args, **kwargs):
            count[0] += 1
            if count[0] == 3:
                raise OSError('fixture acknowledgement persistence failure')
            return replace(*args, **kwargs)
        with patch.object(mod, '_open_codex_app_connection', return_value=connection), patch.object(mod.os, 'replace', side_effect=replacing):
            result = self.call()
        self.assertEqual(result['delivery_state'], 'delivery_unknown')
        self.assertEqual(json.loads(Path(self.kw['receipt_path']).read_text())['delivery_state'], 'delivery_unknown')
        self.assertEqual(sum((m.get('method') == 'turn/steer' for m in connection.messages)), 1)

    def test_notification_flood_is_bounded_before_steer(self):
        connection = _FailureConnection(notifications=40)
        with patch.object(mod, '_open_codex_app_connection', return_value=connection):
            self.assertEqual(self.call()['delivery_state'], 'not_sent')
        self.assertFalse(any((m.get('method') == 'turn/steer' for m in connection.messages)))

    def test_wrong_steer_ack_stays_unknown(self):
        connection = _FailureConnection(ack={'turnId': 'other'})
        with patch.object(mod, '_open_codex_app_connection', return_value=connection):
            result = self.call()
        self.assertEqual(result['delivery_state'], 'delivery_unknown')
        self.assertFalse(result['enqueue_acknowledged'])
        self.assertFalse(result['completion_eligible'])

    def test_regular_file_endpoint_never_connects(self):
        Path(self.kw['socket_path']).write_text('not a socket')
        self.assertEqual(self.call()['delivery_state'], 'not_sent')

    def test_oversized_endpoint_refuses_before_receipt_creation(self):
        with patch.object(mod, '_open_codex_app_connection', side_effect=OSError) as open_:
            result = self.call(socket_path='/' + 's' * 4096)
            self.assertEqual(result['delivery_state'], 'refused')
            open_.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_namespace_replaced_during_preflight_never_steers(self):
        connection = _FailureConnection()
        recv = connection.recv
        moved = self.root.with_name(self.root.name + '-old')
        self.addCleanup(shutil.rmtree, moved, True)

        def receiving(timeout):
            if connection.messages[-1].get('method') == 'thread/turns/list':
                self.root.rename(moved)
                self.root.mkdir(mode=448)
            return recv(timeout)
        connection.recv = receiving
        with patch.object(mod, '_open_codex_app_connection', return_value=connection):
            self.assertNotEqual(self.call()['delivery_state'], 'accepted')
        self.assertFalse(any((m.get('method') == 'turn/steer' for m in connection.messages)))

    def test_class_delegates_without_generation(self):
        instance = object.__new__(mod.OpenAICodexCLIIntegration)
        with patch.object(mod, 'send_codex_thread_message', return_value={'fixture': True}) as send:
            self.assertEqual(instance.send_message_to_thread(**self.kw), {'fixture': True})
            send.assert_called_once_with(**self.kw)


# Unix transport is optional; generation remains importable elsewhere.
ThreadHandoffContractTests = unittest.skipIf(fcntl is None, "Unix handoff requires fcntl")(ThreadHandoffContractTests)
TransportFailureControls = unittest.skipIf(fcntl is None, "Unix handoff requires fcntl")(TransportFailureControls)


# Public handoff entry contract: mocked existing transport, no live socket.
import argparse
import io
from contextlib import ExitStack


class _HandoffCLIOutput:
    def __init__(self, *, write_error=None, flush_error=None, short_write=False, closed=False):
        self.parts = []
        self.write_calls = 0
        self.flushes = 0
        self.write_error = write_error
        self.flush_error = flush_error
        self.short_write = short_write
        self.closed_stream = io.StringIO() if closed else None
        if self.closed_stream is not None:
            self.closed_stream.close()

    def write(self, text):
        self.write_calls += 1
        if self.closed_stream is not None:
            return self.closed_stream.write(text)
        if self.write_error is not None:
            raise self.write_error
        if self.short_write:
            self.parts.append(text[:-1])
            return len(text) - 1
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.flushes += 1
        if self.closed_stream is not None:
            self.closed_stream.flush()
        if self.flush_error is not None:
            raise self.flush_error

    def getvalue(self):
        return ''.join(self.parts)


class HandoffCLIEntryContractTests(unittest.TestCase):
    authority_fields = (
        'processing_confirmed', 'completion_eligible',
        'execution_granted', 'claim_granted',
    )

    def argv(self, *, prior='not_attempted', message='fixture handoff'):
        return [
            'handoff', '--socket-path', '/fixture/daemon.sock',
            '--thread-id', 'thread-1', '--expected-turn-id', 'turn-1',
            '--message', message, '--receipt-path', '/fixture/message.json',
            '--prior-delivery-state', prior,
        ]

    def valid_result(self, state='accepted', **extra):
        return {
            'delivery_state': state,
            'reason': 'fixture_' + state,
            'enqueue_acknowledged': state == 'accepted',
            'retryable': state == 'not_sent',
            **{field: False for field in self.authority_fields},
            **extra,
        }

    def decode_output(self, output):
        text = output.getvalue()
        self.assertTrue(text.endswith('\n'))
        self.assertEqual(len(text.splitlines()), 1)
        self.assertEqual(output.flushes, 1)

        def reject_nonjson(value):
            raise AssertionError('non-JSON numeric token: ' + value)

        return json.loads(text, parse_constant=reject_nonjson)

    def invoke(self, result, *, argv=None):
        output = _HandoffCLIOutput()
        with patch.object(mod, 'send_codex_thread_message', return_value=result) as send:
            with patch('sys.stdout', output):
                rc = mod.codex_thread_handoff_main(self.argv() if argv is None else argv)
        return rc, self.decode_output(output), send

    def assert_safe_unknown(self, rc, result, reason):
        self.assertEqual(rc, 3)
        self.assertEqual(result['delivery_state'], 'delivery_unknown')
        self.assertEqual(result['reason'], reason)
        self.assertIs(result['enqueue_acknowledged'], False)
        self.assertIs(result['retryable'], False)
        for field in self.authority_fields:
            self.assertIs(result[field], False, field)

    def test_explicit_unicode_arguments_are_forwarded_once_without_rewriting(self):
        message = 'Résumé 世界 ⚖️\nsecond line'
        result = self.valid_result(fixture_detail='Δ Unicode preserved')
        args = self.argv(prior='not_sent', message=message) + [
            '--timeout-seconds', '2.75', '--max-attempts', '2',
        ]
        rc, emitted, send = self.invoke(result, argv=args)
        self.assertEqual(rc, 0)
        self.assertEqual(emitted, result)
        send.assert_called_once_with(
            socket_path='/fixture/daemon.sock', thread_id='thread-1',
            expected_turn_id='turn-1', message=message,
            receipt_path='/fixture/message.json', prior_delivery_state='not_sent',
            timeout_seconds=2.75, max_attempts=2,
        )

    def test_optional_defaults_are_forwarded_to_existing_helper_once(self):
        rc, emitted, send = self.invoke(self.valid_result())
        self.assertEqual(rc, 0)
        self.assertEqual(emitted['delivery_state'], 'accepted')
        send.assert_called_once_with(
            socket_path='/fixture/daemon.sock', thread_id='thread-1',
            expected_turn_id='turn-1', message='fixture handoff',
            receipt_path='/fixture/message.json', prior_delivery_state='not_attempted',
            timeout_seconds=15.0, max_attempts=3,
        )

    def test_all_valid_delivery_states_preserve_results_and_exit_codes(self):
        for state, expected_rc in (
            ('accepted', 0), ('not_sent', 1), ('refused', 2),
            ('delivery_unknown', 3), ('in_progress', 4),
        ):
            with self.subTest(state=state):
                result = self.valid_result(state, fixture_detail={'retained': [1, 2]})
                rc, emitted, send = self.invoke(result)
                self.assertEqual(rc, expected_rc)
                self.assertEqual(emitted, result)
                send.assert_called_once()

    def test_each_required_argument_is_rejected_before_helper(self):
        required = (
            '--socket-path', '--thread-id', '--expected-turn-id',
            '--message', '--receipt-path', '--prior-delivery-state',
        )
        for option in required:
            with self.subTest(option=option):
                args = self.argv()
                index = args.index(option)
                del args[index:index + 2]
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(mod, 'send_codex_thread_message') as send:
                    with patch('sys.stdout', stdout), patch('sys.stderr', stderr):
                        with self.assertRaises(SystemExit) as stopped:
                            mod.codex_thread_handoff_main(args)
                self.assertEqual(stopped.exception.code, 2)
                send.assert_not_called()
                self.assertEqual(stdout.getvalue(), '')
                self.assertTrue(stderr.getvalue())

    def test_bad_flags_prior_choice_and_positional_are_argparse_failures(self):
        bad_vectors = [
            self.argv() + ['--unknown-flag'],
            self.argv() + ['--timeout-seconds', 'not-a-number'],
            self.argv() + ['--max-attempts', '1.5'],
            self.argv() + ['--message-file', '/fixture/body.txt'],
            self.argv() + ['--stdin'],
            self.argv(prior='accepted'),
            ['replace'] + self.argv()[1:],
            self.argv()[1:],
        ]
        for args in bad_vectors:
            with self.subTest(args=args):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(mod, 'send_codex_thread_message') as send:
                    with patch('sys.stdout', stdout), patch('sys.stderr', stderr):
                        with self.assertRaises(SystemExit) as stopped:
                            mod.codex_thread_handoff_main(args)
                self.assertEqual(stopped.exception.code, 2)
                send.assert_not_called()
                self.assertEqual(stdout.getvalue(), '')
                self.assertTrue(stderr.getvalue())

    def test_prior_unknown_is_forwarded_once_and_never_retried(self):
        result = self.valid_result('delivery_unknown', reason='prior_delivery_unknown')
        rc, emitted, send = self.invoke(result, argv=self.argv(prior='unknown'))
        self.assertEqual((rc, emitted), (3, result))
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs['prior_delivery_state'], 'unknown')

    def test_helper_exception_is_redacted_without_calling_hostile_str(self):
        class HostileError(Exception):
            def __str__(self):
                raise AssertionError('exception text must never be rendered')

        output = _HandoffCLIOutput()
        failure = HostileError('private transport details must not escape')
        with patch.object(mod, 'send_codex_thread_message', side_effect=failure) as send:
            with patch('sys.stdout', output):
                rc = mod.codex_thread_handoff_main(self.argv())
        emitted = self.decode_output(output)
        self.assert_safe_unknown(rc, emitted, 'cli_transport_exception')
        send.assert_called_once()
        self.assertNotIn('private transport details', output.getvalue())
        self.assertNotIn('HostileError', output.getvalue())

    def test_malformed_and_nonjson_results_become_safe_unknown(self):
        malformed = [
            None, [], 'accepted', 1,
            {}, self.valid_result('unrecognized_state'),
            self.valid_result(detail=object()),
            self.valid_result(detail=float('nan')),
            self.valid_result(detail=float('inf')),
        ]
        for index, result in enumerate(malformed):
            with self.subTest(index=index):
                rc, emitted, send = self.invoke(result)
                self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                send.assert_called_once()

    def test_authority_grants_and_nonfalse_authority_fields_are_rejected(self):
        for field in self.authority_fields:
            for invalid in (True, 0, None, 'false'):
                with self.subTest(field=field, invalid=invalid):
                    result = self.valid_result()
                    result[field] = invalid
                    rc, emitted, send = self.invoke(result)
                    self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                    send.assert_called_once()
            with self.subTest(field=field, invalid='missing'):
                result = self.valid_result()
                del result[field]
                rc, emitted, send = self.invoke(result)
                self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                send.assert_called_once()

    def test_invalid_ack_types_and_state_ack_mismatches_are_rejected(self):
        for invalid in (None, 1, 'true'):
            with self.subTest(invalid=invalid):
                rc, emitted, send = self.invoke(self.valid_result(enqueue_acknowledged=invalid))
                self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                send.assert_called_once()
        missing = self.valid_result()
        del missing['enqueue_acknowledged']
        rc, emitted, send = self.invoke(missing)
        self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
        send.assert_called_once()
        for state in ('accepted', 'not_sent', 'refused', 'delivery_unknown', 'in_progress'):
            with self.subTest(state=state):
                result = self.valid_result(state, enqueue_acknowledged=state != 'accepted')
                rc, emitted, send = self.invoke(result)
                self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                send.assert_called_once()

    def test_entry_never_constructs_generator_connects_or_opens_files(self):
        result = self.valid_result()
        output = _HandoffCLIOutput()
        with ExitStack() as stack:
            send = stack.enter_context(patch.object(mod, 'send_codex_thread_message', return_value=result))
            constructor = stack.enter_context(patch.object(mod, 'OpenAICodexCLIIntegration', side_effect=AssertionError('generation constructor used')))
            connect = stack.enter_context(patch.object(mod, '_open_codex_app_connection', side_effect=AssertionError('entry opened connection')))
            opening = stack.enter_context(patch('builtins.open', side_effect=AssertionError('entry opened a file')))
            fd_opening = stack.enter_context(patch.object(mod.os, 'open', side_effect=AssertionError('entry opened an fd')))
            stack.enter_context(patch.object(Path, 'open', side_effect=AssertionError('entry opened a Path')))
            stack.enter_context(patch('sys.stdout', output))
            rc = mod.codex_thread_handoff_main(self.argv())
        self.assertEqual((rc, self.decode_output(output)), (0, result))
        send.assert_called_once()
        constructor.assert_not_called()
        connect.assert_not_called()
        opening.assert_not_called()
        fd_opening.assert_not_called()

    def test_output_failures_return_unknown_code_without_second_write_or_replay(self):
        cases = (
            ('write', {'write_error': OSError('fixture write failure')}),
            ('flush', {'flush_error': OSError('fixture flush failure')}),
            ('short_write', {'short_write': True}),
            ('closed_stream', {'closed': True}),
        )
        for phase, kwargs in cases:
            with self.subTest(phase=phase):
                output = _HandoffCLIOutput(**kwargs)
                with patch.object(mod, 'send_codex_thread_message', return_value=self.valid_result()) as send:
                    with patch('sys.stdout', output):
                        rc = mod.codex_thread_handoff_main(self.argv())
                self.assertEqual(rc, 3)
                send.assert_called_once()
                self.assertEqual(output.write_calls, 1, 'output failure must never trigger a second write')
                self.assertLessEqual(output.flushes, 1)


    def test_retryability_is_strict_bool_and_never_grants_terminal_replay(self):
        for state in ('accepted', 'not_sent', 'refused', 'delivery_unknown', 'in_progress'):
            invalid_values = [('none', None), ('zero', 0), ('one', 1), ('text', 'false'), ('missing', None)]
            if state != 'not_sent':
                invalid_values.append(('terminal_true', True))
            for label, invalid in invalid_values:
                with self.subTest(state=state, invalid=label):
                    result = self.valid_result(state)
                    if label == 'missing':
                        del result['retryable']
                    else:
                        result['retryable'] = invalid
                    rc, emitted, send = self.invoke(result)
                    self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
                    send.assert_called_once()
        for retryable in (True, False):
            with self.subTest(state='not_sent', valid_retryable=retryable):
                result = self.valid_result('not_sent', retryable=retryable)
                rc, emitted, send = self.invoke(result)
                self.assertEqual((rc, emitted), (1, result))
                send.assert_called_once()

    def test_metadata_recursion_and_other_encoding_failures_become_safe_unknown(self):
        real_dumps = json.dumps
        encoding_calls = []

        def recurse_once(value, *args, **kwargs):
            encoding_calls.append(value)
            if len(encoding_calls) == 1:
                raise RecursionError('fixture encoding recursion')
            return real_dumps(value, *args, **kwargs)

        with self.subTest(encoding='RecursionError'):
            with patch.object(mod.json, 'dumps', side_effect=recurse_once):
                rc, emitted, send = self.invoke(self.valid_result(detail={'ordinary': 'metadata'}))
            self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
            send.assert_called_once()
            self.assertEqual(len(encoding_calls), 2, 'one failing encode, then the safe fallback encode')

        class HostileMetadata(dict):
            def items(self):
                raise RuntimeError('private serialization detail')

        with self.subTest(encoding='other_Exception'):
            rc, emitted, send = self.invoke(self.valid_result(detail=HostileMetadata(value=1)))
            self.assert_safe_unknown(rc, emitted, 'cli_invalid_result')
            send.assert_called_once()
            self.assertNotIn('private serialization detail', json.dumps(emitted))

        class DeliberateAbort(BaseException):
            pass

        abort = DeliberateAbort('preserve process control')

        class AbortingMetadata(dict):
            def items(self):
                raise abort

        output = _HandoffCLIOutput()
        with patch.object(mod, 'send_codex_thread_message', return_value=self.valid_result(detail=AbortingMetadata(value=1))) as send:
            with patch('sys.stdout', output):
                with self.assertRaises(DeliberateAbort) as raised:
                    mod.codex_thread_handoff_main(self.argv())
        self.assertIs(raised.exception, abort)
        send.assert_called_once()
        self.assertEqual(output.write_calls, 0)

import io
import json
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.cli_integrations import openai_codex_cli_integration as mod


class _ReadSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class _ReadConnection:
    def __init__(self, replies=None, prefix_frames=()):
        self.socket = _ReadSocket()
        self.sent = []
        self.closed = False
        self.prefix_frames = list(prefix_frames)
        self.replies = {
            "initialize": {},
            "thread/read": {"thread": {
                "id": "thread-1", "canAcceptDirectInput": True,
                "status": {"type": "active"}, "preview": "never export this preview",
            }},
            "thread/turns/list": {
                "data": [{"id": "turn-1", "status": "inProgress", "items": ["do not export"]}],
                "nextCursor": "limited-history",
            },
        }
        if replies:
            self.replies.update(replies)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def send(self, wire):
        self.sent.append(json.loads(wire))

    def recv(self, *, timeout):
        if self.prefix_frames:
            return self.prefix_frames.pop(0)
        request = self.sent[-1]
        reply = self.replies[request["method"]]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (str, bytes)):
            return reply
        return json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": reply})


class ReadThreadFallbackTests(unittest.TestCase):
    def query(self, connection=None, **kwargs):
        connection = connection or _ReadConnection()
        with patch.object(mod, "_open_codex_app_connection", return_value=connection) as connect:
            result = mod.read_codex_thread_state(
                socket_path="/tmp/explicit-owned-codex.sock", thread_id="thread-1", **kwargs,
            )
        self.assertEqual(connect.call_count, 1)
        return result, connection

    def assert_no_authority(self, result):
        self.assertIs(result["read_only"], True)
        for key in ("snapshot_atomic", "history_complete", "delivery_assessed",
                    "processing_confirmed", "completion_eligible", "execution_granted", "claim_granted"):
            self.assertIs(result[key], False)

    def test_success_uses_only_existing_read_rpc_and_returns_bounded_metadata(self):
        result, connection = self.query()
        self.assertEqual(result["observation_state"], "observed")
        self.assertEqual(result["latest_turn"], {"id": "turn-1", "status": "inProgress"})
        self.assertEqual([row["method"] for row in connection.sent], [
            "initialize", "initialized", "thread/read", "thread/turns/list",
        ])
        self.assertEqual(connection.sent[2]["params"], {"threadId": "thread-1", "includeTurns": False})
        self.assertEqual(connection.sent[3]["params"], {"threadId": "thread-1", "limit": 1, "sortDirection": "desc"})
        self.assertNotIn("preview", json.dumps(result))
        self.assertNotIn("do not export", json.dumps(result))
        self.assertTrue(connection.closed)
        self.assertTrue(all(value > 0 for value in connection.socket.timeouts))
        self.assert_no_authority(result)

    def test_nonaccepting_and_terminal_targets_are_readable_without_steering(self):
        for status in ("completed", "failed", "interrupted"):
            with self.subTest(status=status):
                connection = _ReadConnection({
                    "thread/read": {"thread": {"id": "thread-1", "canAcceptDirectInput": False}},
                    "thread/turns/list": {"data": [{"turn": {"id": "turn-1", "status": status}}]},
                })
                result, _ = self.query(connection)
                self.assertEqual(result["observation_state"], "observed")
                self.assertIs(result["thread_snapshot"]["can_accept_direct_input"], False)
                self.assertEqual(result["latest_turn"]["status"], status)
                self.assert_no_authority(result)

    def test_optional_thread_fields_may_be_unknown(self):
        result, _ = self.query(_ReadConnection({"thread/read": {"thread": {"id": "thread-1"}}}))
        self.assertEqual(result["observation_state"], "observed")
        self.assertIsNone(result["thread_snapshot"]["status"])
        self.assertIsNone(result["thread_snapshot"]["can_accept_direct_input"])

    def test_empty_history_is_not_a_non_delivery_or_completion_proof(self):
        result, _ = self.query(_ReadConnection({"thread/turns/list": {"data": []}}))
        self.assertEqual(result["observation_state"], "observed")
        self.assertIsNone(result["latest_turn"])
        self.assert_no_authority(result)

    def test_wrong_thread_identity_refuses_before_listing_turns(self):
        result, connection = self.query(_ReadConnection({"thread/read": {"thread": {"id": "other-thread"}}}))
        self.assertEqual(result["observation_state"], "refused")
        self.assertEqual(result["reason"], "thread_identity_mismatch")
        self.assertNotIn("thread/turns/list", [row["method"] for row in connection.sent])

    def test_malformed_thread_fields_are_not_coerced(self):
        for fields in ({"canAcceptDirectInput": 1}, {"status": []}, {"status": "x" * 65}):
            with self.subTest(fields=fields):
                result, _ = self.query(_ReadConnection({"thread/read": {"thread": {"id": "thread-1", **fields}}}))
                self.assertEqual(result["observation_state"], "refused")
                self.assertIsNone(result["thread_snapshot"])

    def test_malformed_or_oversized_latest_listing_keeps_partial_thread_observation(self):
        bad_rows = [None, {}, {"data": None}, {"data": [{}, {}]},
                    {"data": [{"id": "bad id", "status": "completed"}]},
                    {"data": [{"id": "turn-1", "status": True}]}]
        for listing in bad_rows:
            with self.subTest(listing=listing):
                result, _ = self.query(_ReadConnection({"thread/turns/list": listing}))
                self.assertEqual(result["observation_state"], "partial")
                self.assertEqual(result["thread_snapshot"]["id"], "thread-1")
                self.assertIsNone(result["latest_turn"])
                self.assert_no_authority(result)

    def test_lost_latest_reply_retains_successful_thread_read(self):
        result, _ = self.query(_ReadConnection({"thread/turns/list": TimeoutError("private detail")}))
        self.assertEqual(result["observation_state"], "partial")
        self.assertNotIn("private detail", json.dumps(result))
        self.assertEqual(result["thread_snapshot"]["id"], "thread-1")

    def test_connection_failure_is_bounded_redacted_and_never_retried(self):
        class Hostile(Exception):
            def __str__(self):
                raise AssertionError("must not format hostile exception")
        with patch.object(mod, "_open_codex_app_connection", side_effect=Hostile()) as connect:
            result = mod.read_codex_thread_state(socket_path="/tmp/codex.sock", thread_id="thread-1")
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(result["observation_state"], "unavailable")
        self.assert_no_authority(result)

    def test_duplicate_boolean_wrong_id_and_nonfinite_replies_are_refused(self):
        frames = [
            '{"id":1,"id":1,"result":{}}',
            '{"id":true,"result":{}}',
            '{"id":999,"result":{}}',
            '{"id":1,"result":{"x":NaN}}',
            '[]',
            '{"id":1,"error":{"message":"secret"}}',
        ]
        for frame in frames:
            with self.subTest(frame=frame):
                result, _ = self.query(_ReadConnection(prefix_frames=[frame]))
                self.assertEqual(result["observation_state"], "unavailable")
                self.assert_no_authority(result)

    def test_total_utf8_response_bytes_are_bounded(self):
        frame = json.dumps({"method": "notice", "params": "é" * 2500}, ensure_ascii=False)
        result, connection = self.query(_ReadConnection(prefix_frames=[frame]), max_response_bytes=4096)
        self.assertEqual(result["observation_state"], "unavailable")
        self.assertEqual(len(connection.sent), 1)

    def test_notification_flood_is_bounded(self):
        frames = [json.dumps({"method": "notice"})] * 33
        result, connection = self.query(_ReadConnection(prefix_frames=frames))
        self.assertEqual(result["observation_state"], "unavailable")
        self.assertEqual(len(connection.sent), 1)

    def test_expired_deadline_prevents_even_connecting(self):
        with patch.object(mod.time, "monotonic", side_effect=[0.0, 61.0]):
            with patch.object(mod, "_open_codex_app_connection") as connect:
                result = mod.read_codex_thread_state(socket_path="/tmp/codex.sock", thread_id="thread-1")
        connect.assert_not_called()
        self.assertEqual(result["observation_state"], "unavailable")

    def test_invalid_inputs_never_connect_or_touch_receipts(self):
        cases = [
            {"thread_id": "bad id"}, {"thread_id": 1},
            {"socket_path": "relative.sock"}, {"socket_path": "/tmp/../x.sock"},
            {"socket_path": "/" + "x" * 108},
            {"timeout_seconds": True}, {"timeout_seconds": float("nan")},
            {"timeout_seconds": 0}, {"timeout_seconds": 61},
            {"max_response_bytes": True}, {"max_response_bytes": 4095},
            {"max_response_bytes": 4194305},
        ]
        for update in cases:
            with self.subTest(update=update):
                with patch.object(mod, "_open_codex_app_connection") as connect:
                    result = mod.read_codex_thread_state(**{
                        "socket_path": "/tmp/codex.sock", "thread_id": "thread-1", **update,
                    })
                connect.assert_not_called()
                self.assertEqual(result["observation_state"], "refused")

    def test_untrusted_server_authority_fields_and_previews_are_ignored(self):
        connection = _ReadConnection({"thread/read": {"thread": {
            "id": "thread-1", "execution_granted": True, "claim_granted": True,
            "preview": "secret source text", "status": "idle",
        }}})
        result, _ = self.query(connection)
        self.assertEqual(result["observation_state"], "observed")
        self.assertNotIn("secret source text", json.dumps(result))
        self.assert_no_authority(result)

    def test_read_uses_no_receipt_file_write_or_delivery_helper(self):
        connection = _ReadConnection()
        with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no send")):
            with patch.object(mod.os, "open", side_effect=AssertionError("no receipt")):
                result, _ = self.query(connection)
        self.assertEqual(result["observation_state"], "observed")


class ReadThreadCLIEntryTests(unittest.TestCase):
    def result(self, state="observed"):
        return {
            "schema": "codex-existing-thread-state/v1", "observation_state": state,
            "reason": "metadata_observed", "thread_id": "thread-1",
            "thread_snapshot": {"id": "thread-1", "status": "active", "can_accept_direct_input": True},
            "latest_turn": {"id": "turn-1", "status": "inProgress"},
            "read_only": True, "snapshot_atomic": False, "history_complete": False,
            "delivery_assessed": False, "processing_confirmed": False,
            "completion_eligible": False, "execution_granted": False, "claim_granted": False,
        }

    def args(self):
        return ["read-thread", "--socket-path", "/tmp/codex.sock", "--thread-id", "thread-1"]

    def invoke(self, result=None, output=None, args=None):
        output = output or io.StringIO()
        with patch.object(mod, "read_codex_thread_state", return_value=result or self.result()) as query:
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no delivery")):
                with patch.object(mod, "get_openai_codex_cli_integration", side_effect=AssertionError("no generator")):
                    with patch("sys.stdout", output):
                        code = mod.codex_thread_handoff_main(self.args() if args is None else args)
        return code, output, query

    def test_existing_entry_dispatches_read_without_send_arguments_or_generator(self):
        code, output, query = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), self.result())
        query.assert_called_once_with(socket_path="/tmp/codex.sock", thread_id="thread-1", timeout_seconds=15.0, max_response_bytes=262144)

    def test_read_states_have_distinct_truthful_exit_codes(self):
        for state, expected in (("observed", 0), ("partial", 1), ("unavailable", 1), ("refused", 2)):
            with self.subTest(state=state):
                code, output, query = self.invoke(self.result(state))
                self.assertEqual(code, expected)
                self.assertEqual(json.loads(output.getvalue())["observation_state"], state)
                self.assertEqual(query.call_count, 1)

    def test_required_read_arguments_fail_before_transport(self):
        for args in (["read-thread"], ["read-thread", "--socket-path", "/tmp/codex.sock"], self.args() + ["--receipt-path", "/tmp/unselected.json"]):
            with self.subTest(args=args):
                with patch.object(mod, "read_codex_thread_state") as query:
                    with patch("sys.stderr", io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            mod.codex_thread_handoff_main(args)
                self.assertEqual(raised.exception.code, 2)
                query.assert_not_called()

    def test_invalid_or_authoritative_helper_result_becomes_unavailable(self):
        for update in ({"execution_granted": True}, {"processing_confirmed": 0}, {"read_only": 1}, {"unexpected": "secret"}):
            with self.subTest(update=update):
                code, output, _ = self.invoke({**self.result(), **update})
                result = json.loads(output.getvalue())
                self.assertEqual(code, 1)
                self.assertEqual(result["observation_state"], "unavailable")
                self.assertIs(result["execution_granted"], False)
                self.assertNotIn("secret", output.getvalue())

    def test_output_failure_does_not_repeat_the_query_or_write(self):
        class Broken:
            def __init__(self):
                self.writes = 0
            def write(self, value):
                self.writes += 1
                raise OSError("lost output")
            def flush(self):
                raise AssertionError("must not flush")
        output = Broken()
        code, _, query = self.invoke(output=output)
        self.assertEqual(code, 1)
        self.assertEqual(output.writes, 1)
        self.assertEqual(query.call_count, 1)

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.cli_integrations import openai_codex_cli_integration as mod


class PrivateArtifactFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="owned-handoff-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "selected.json"
        self.body = 'fictional legal-work control payload: λ\x00bytes\n'.encode()
        self.path.write_bytes(self.body)
        self.sha = hashlib.sha256(self.body).hexdigest()

    def read(self, **updates):
        return mod.read_codex_handoff_artifact(**{
            "private_root": str(self.root), "artifact_path": str(self.path),
            "expected_bytes": len(self.body), "expected_sha256": self.sha, **updates,
        })

    def assert_no_authority(self, value):
        self.assertIs(value["read_only"], True)
        for key in ("delivery_assessed", "processing_confirmed", "completion_eligible", "execution_granted", "claim_granted"):
            self.assertIs(value[key], False)

    def test_real_private_file_is_verified_losslessly_with_exact_full9_strings(self):
        before = self.path.stat()
        result = self.read()
        self.assertEqual(result["artifact_state"], "verified")
        self.assertEqual(result["bytes"], len(self.body))
        self.assertEqual(result["sha256"], self.sha)
        self.assertEqual(base64.b64decode(result["payload_base64"], validate=True), self.body)
        keys = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        self.assertEqual(result["full9"], {key: str(getattr(before, key)) for key in keys})
        self.assert_no_authority(result)

    def test_wrong_size_is_refused_without_payload(self):
        result = self.read(expected_bytes=len(self.body) + 1)
        self.assertEqual(result["artifact_state"], "refused")
        self.assertIsNone(result["payload_base64"])

    def test_wrong_digest_is_refused_without_payload(self):
        result = self.read(expected_sha256="0" * 64)
        self.assertEqual(result["reason"], "artifact_digest_mismatch")
        self.assertIsNone(result["payload_base64"])

    def test_outside_scope_traversal_and_invalid_bounds_refuse_before_open(self):
        cases = [
            {"artifact_path": str(self.root.parent / "outside.json")},
            {"artifact_path": str(self.root / "nested" / ".." / "selected.json")},
            {"private_root": "relative"}, {"expected_bytes": True},
            {"expected_bytes": 0}, {"expected_bytes": 1048577},
            {"max_bytes": True}, {"max_bytes": 1048577},
            {"expected_sha256": self.sha.upper()}, {"expected_sha256": "x" * 64},
        ]
        for case in cases:
            with self.subTest(case=case):
                with patch.object(mod.os, "open", side_effect=AssertionError("invalid scope must not open")):
                    result = self.read(**case)
                self.assertEqual(result["artifact_state"], "refused")

    def test_nonprivate_root_is_refused(self):
        self.root.chmod(0o755)
        try:
            result = self.read()
        finally:
            self.root.chmod(0o700)
        self.assertEqual(result["reason"], "artifact_root_not_private")

    def test_symlink_artifact_is_never_followed(self):
        target = self.root / "target.json"
        self.path.rename(target)
        self.path.symlink_to(target)
        result = self.read()
        self.assertEqual(result["artifact_state"], "unavailable")
        self.assertEqual(target.read_bytes(), self.body)

    def test_hardlinked_artifact_is_refused(self):
        alias = self.root / "alias.json"
        os.link(self.path, alias)
        result = self.read()
        self.assertEqual(result["artifact_state"], "refused")
        self.assertEqual(alias.read_bytes(), self.body)

    def test_fifo_artifact_refuses_without_blocking(self):
        self.path.unlink()
        os.mkfifo(self.path)
        result = self.read()
        self.assertEqual(result["artifact_state"], "refused")
        self.assertIsNone(result["payload_base64"])

    def test_symlink_root_is_never_followed(self):
        alias = self.root.parent / (self.root.name + "-alias")
        alias.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(alias.unlink)
        result = self.read(private_root=str(alias), artifact_path=str(alias / self.path.name))
        self.assertEqual(result["artifact_state"], "unavailable")

    def test_same_bytes_at_replaced_path_fail_the_full9_guard(self):
        real_read = os.read
        changed = []
        def replace_after_read(fd, count):
            data = real_read(fd, count)
            if data and not changed:
                changed.append(True)
                self.path.rename(self.root / "original.json")
                self.path.write_bytes(self.body)
            return data
        with patch.object(mod.os, "read", side_effect=replace_after_read):
            result = self.read()
        self.assertEqual(result["reason"], "artifact_changed")
        self.assertIsNone(result["payload_base64"])

    def test_replaced_root_namespace_is_detected(self):
        real_read = os.read
        changed = []
        backup = self.root.parent / (self.root.name + "-retained-original")
        self.addCleanup(lambda: shutil.rmtree(backup, ignore_errors=True))
        def replace_root(fd, count):
            data = real_read(fd, count)
            if data and not changed:
                changed.append(True)
                self.root.rename(backup)
                self.root.mkdir(mode=0o700)
                self.path.write_bytes(self.body)
            return data
        with patch.object(mod.os, "read", side_effect=replace_root):
            result = self.read()
        self.assertEqual(result["artifact_state"], "unavailable")
        self.assertIsNone(result["payload_base64"])

    def test_existing_cli_reads_real_artifact_without_socket_message_or_generator(self):
        args = ["read-artifact", "--private-root", str(self.root), "--artifact-path", str(self.path),
                "--expected-bytes", str(len(self.body)), "--expected-sha256", self.sha]
        output = io.StringIO()
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("no socket")):
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no delivery")):
                with patch.object(mod, "get_openai_codex_cli_integration", side_effect=AssertionError("no model")):
                    with patch("sys.stdout", output):
                        code = mod.codex_thread_handoff_main(args)
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["artifact_state"], "verified")
        self.assertEqual(base64.b64decode(result["payload_base64"]), self.body)
        self.assert_no_authority(result)

    def test_cli_requires_exact_artifact_identity_before_reading(self):
        for args in (["read-artifact"], ["read-artifact", "--private-root", str(self.root)]):
            with self.subTest(args=args):
                with patch.object(mod, "read_codex_handoff_artifact") as read:
                    with patch("sys.stderr", io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            mod.codex_thread_handoff_main(args)
                self.assertEqual(raised.exception.code, 2)
                read.assert_not_called()

    def test_cli_lost_output_does_not_repeat_the_artifact_read(self):
        class Lost:
            def __init__(self):
                self.writes = 0
            def write(self, value):
                self.writes += 1
                raise OSError("lost output")
            def flush(self):
                raise AssertionError("no second write or flush")
        output = Lost()
        args = ["read-artifact", "--private-root", str(self.root), "--artifact-path", str(self.path),
                "--expected-bytes", str(len(self.body)), "--expected-sha256", self.sha]
        with patch.object(mod, "read_codex_handoff_artifact", wraps=mod.read_codex_handoff_artifact) as read:
            with patch("sys.stdout", output):
                code = mod.codex_thread_handoff_main(args)
        self.assertEqual(code, 1)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(output.writes, 1)


class DurableHandoffProducerTests(unittest.TestCase):
    """Fictional local payloads; no real socket, worker or project initialization."""
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.path = self.root / "handoff.bin"
        self.payload = b'{"task_id":"fictional-unit","generation":7,"authority":false}\n'

    def write(self, **overrides):
        args = dict(private_root=str(self.root), artifact_path=str(self.path), payload=self.payload)
        args.update(overrides)
        return mod.write_codex_handoff_artifact(**args)

    def cli(self, *extra):
        import io
        args = ["handoff", "--socket-path", str(self.root / "socket"),
                "--thread-id", "fictional-thread", "--expected-turn-id", "fictional-turn",
                "--message", self.payload.decode(), "--receipt-path", str(self.root / "delivery.json"),
                "--prior-delivery-state", "unknown", "--offline-private-root", str(self.root),
                "--offline-artifact-path", str(self.path), *extra]
        out = io.StringIO()
        with patch("sys.stdout", out):
            rc = mod.codex_thread_handoff_main(args)
        return rc, json.loads(out.getvalue())

    def test_new_raw_bytes_are_retained_and_existing_reader_verifies(self):
        result = self.write()
        self.assertEqual(result["artifact_state"], "retained")
        self.assertTrue(result["created"] and result["readback_verified"])
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o400)
        self.assertTrue(all(type(v) is str for v in result["full9"].values()))
        read = mod.read_codex_handoff_artifact(private_root=str(self.root), artifact_path=str(self.path),
            expected_bytes=len(self.payload), expected_sha256=hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(read["artifact_state"], "verified")
        self.assertEqual(base64.b64decode(read["payload_base64"]), self.payload)
        for key in ("delivery_assessed", "processing_confirmed", "completion_eligible", "execution_granted", "claim_granted", "retryable"):
            self.assertIs(result[key], False)

    def test_existing_path_is_never_overwritten_or_reused(self):
        self.path.write_bytes(b"old-exact-history")
        before = self.path.stat()
        result = self.write()
        self.assertEqual(result["reason"], "artifact_already_exists")
        self.assertFalse(result["created"])
        self.assertEqual(self.path.read_bytes(), b"old-exact-history")
        self.assertEqual((before.st_ino, before.st_mtime_ns), (self.path.stat().st_ino, self.path.stat().st_mtime_ns))

    def test_invalid_inputs_refuse_before_any_open(self):
        for change in ({"payload":b""}, {"payload":"text"}, {"payload":b"abc", "max_bytes":True},
                       {"payload":b"abc", "max_bytes":2}, {"max_bytes":1048577},
                       {"artifact_path":str(self.root.parent / "outside")}, {"private_root":"relative"}):
            with self.subTest(change=repr(change)), patch.object(mod.os, "open", side_effect=AssertionError("must not open")):
                self.assertEqual(self.write(**change)["artifact_state"], "refused")

    def test_nonprivate_and_symlink_roots_refuse_without_artifact(self):
        self.root.chmod(0o755)
        self.assertEqual(self.write()["reason"], "artifact_root_not_private")
        self.assertFalse(self.path.exists())
        self.root.chmod(0o700)
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        result = self.write(private_root=str(link), artifact_path=str(link / "other"))
        self.assertEqual(result["artifact_state"], "refused")
        self.assertFalse((self.root / "other").exists())

    def test_short_writes_loop_until_all_bytes_are_retained(self):
        original = mod.os.write
        with patch.object(mod.os, "write", side_effect=lambda fd, b: original(fd, b[:3])):
            self.assertEqual(self.write()["artifact_state"], "retained")
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_zero_write_preserves_incomplete_file_and_refuses(self):
        with patch.object(mod.os, "write", return_value=0):
            result = self.write()
        self.assertEqual(result["artifact_state"], "incomplete")
        self.assertTrue(result["created"] and self.path.exists())
        self.assertFalse(result["readback_verified"] or result["retryable"])

    def test_fsync_failure_preserves_created_bytes_and_never_sends(self):
        with patch.object(mod.os, "fsync", side_effect=OSError("fictional fsync failure")), \
             patch.object(mod, "send_codex_thread_message") as send:
            rc, outcome = self.cli()
        self.assertEqual(rc, 3)
        self.assertEqual(outcome["delivery_state"], "delivery_unknown")
        self.assertIs(outcome["current_call_transport_attempted"], False)
        self.assertEqual(outcome["offline_artifact"]["artifact_state"], "incomplete")
        self.assertTrue(self.path.exists())
        send.assert_not_called()

    def test_readback_failure_cannot_report_retained(self):
        with patch.object(mod, "read_codex_handoff_artifact", return_value={"artifact_state":"refused"}):
            result = self.write()
        self.assertEqual(result["artifact_state"], "incomplete")
        self.assertTrue(self.path.exists())
        self.assertFalse(result["readback_verified"])

    def test_namespace_replacement_during_write_is_not_retained(self):
        original = mod.os.write
        moved = self.root.parent / (self.root.name + "-moved")
        self.addCleanup(lambda: __import__("shutil").rmtree(moved, ignore_errors=True))
        changed = []
        def replace(fd, block):
            count = original(fd, block)
            if not changed:
                self.root.rename(moved); self.root.mkdir(mode=0o700); changed.append(True)
            return count
        with patch.object(mod.os, "write", side_effect=replace):
            result = self.write()
        self.assertEqual(result["artifact_state"], "incomplete")
        self.assertTrue((moved / "handoff.bin").exists())
        self.assertFalse(self.path.exists())

    def test_artifact_is_readback_verified_before_single_transport_call(self):
        def unavailable(**kwargs):
            self.assertEqual(self.path.read_bytes(), self.payload)
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o400)
            return {"delivery_state":"not_sent", "reason":"fictional_transport_unavailable", "enqueue_acknowledged":False,
                    "processing_confirmed":False,"completion_eligible":False,"execution_granted":False,"claim_granted":False,"retryable":True}
        with patch.object(mod, "send_codex_thread_message", side_effect=unavailable) as send:
            rc, outcome = self.cli()
        self.assertEqual(rc, 1)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(outcome["offline_artifact"]["artifact_state"], "retained")
        self.assertFalse(outcome["execution_granted"])

    def test_unknown_delivery_remains_unknown_and_second_call_never_sends(self):
        unknown = {"delivery_state":"delivery_unknown", "reason":"fictional_timeout_after_send", "enqueue_acknowledged":False,
                   "processing_confirmed":False,"completion_eligible":False,"execution_granted":False,"claim_granted":False,"retryable":False}
        with patch.object(mod, "send_codex_thread_message", return_value=unknown) as send:
            rc, first = self.cli()
            second_rc, second = self.cli()
        self.assertEqual((rc, second_rc), (3, 3))
        self.assertEqual(send.call_count, 1)
        self.assertEqual(first["delivery_state"], "delivery_unknown")
        self.assertEqual(first["offline_artifact"]["artifact_state"], "retained")
        self.assertEqual(second["delivery_state"], "delivery_unknown")
        self.assertFalse(second["retryable"])

    def test_optional_pair_is_required_before_writing_or_sending(self):
        import io
        args=["handoff","--socket-path","unused","--thread-id","t","--expected-turn-id","u", "--message","x",
              "--receipt-path",str(self.root/"r"),"--prior-delivery-state","unknown","--offline-private-root",str(self.root)]
        with patch("sys.stderr",io.StringIO()), patch.object(mod,"send_codex_thread_message") as send:
            with self.assertRaises(SystemExit) as caught:mod.codex_thread_handoff_main(args)
        self.assertEqual(caught.exception.code,2)
        self.assertFalse(self.path.exists())
        send.assert_not_called()

    def test_unknown_prior_delivery_is_not_replayed_by_real_sender(self):
        with patch.object(mod,"_open_codex_app_connection",side_effect=AssertionError("no socket")) as connect:
            rc,outcome=self.cli()
        self.assertEqual(rc,3)
        self.assertEqual(outcome["delivery_state"],"delivery_unknown")
        self.assertEqual(outcome["offline_artifact"]["artifact_state"],"retained")
        connect.assert_not_called()

    def test_all_opened_writer_descriptors_close_on_success_and_failure(self):
        original=mod.os.open;opened=[]
        def track(*args,**kwargs):
            fd=original(*args,**kwargs);opened.append(fd);return fd
        with patch.object(mod.os,"open",side_effect=track):
            self.assertEqual(self.write()["artifact_state"],"retained")
        for fd in opened:
            with self.assertRaises(OSError):mod.os.fstat(fd)


    def test_persistence_failure_preserves_unknown_without_inspecting_prior_delivery(self):
        self.path.write_bytes(b"retained earlier payload")
        before = self.path.read_bytes()
        with patch.object(mod, "send_codex_thread_message") as send:
            rc, outcome = self.cli()
        self.assertEqual((rc, outcome["delivery_state"]), (3, "delivery_unknown"))
        self.assertIs(outcome["current_call_transport_attempted"], False)
        self.assertFalse(outcome["retryable"])
        self.assertEqual(self.path.read_bytes(), before)
        send.assert_not_called()

    def test_prior_not_attempted_does_not_override_uninspected_existing_receipt(self):
        self.path.write_bytes(b"existing artifact")
        receipt = self.root / "delivery.json"
        retained = b'{"opaque_prior_history":"must not be reclassified"}\n'
        receipt.write_bytes(retained)
        with patch.object(mod, "send_codex_thread_message") as send:
            rc, outcome = self.cli("--prior-delivery-state", "not_attempted")
        self.assertEqual((rc, outcome["delivery_state"]), (3, "delivery_unknown"))
        self.assertIs(outcome["current_call_transport_attempted"], False)
        self.assertFalse(outcome["retryable"])
        self.assertEqual(receipt.read_bytes(), retained)
        send.assert_not_called()


class ManifestHandoffRecoveryTests(unittest.TestCase):
    """Fictional handoffs; real bounded private-file verification, no delivery."""

    def setUp(self):
        import hashlib
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.body = b'{"task":"fictional-legal-software-unit","legal_admission":false}'
        self.target = self.root / "unit.json"
        self.target.write_bytes(self.body)
        self.target.chmod(0o400)
        self.manifest = self.root / "handoff.json"
        self.index = {"artifacts": {"unit": {
            "file": self.target.name, "bytes": len(self.body),
            "sha256": hashlib.sha256(self.body).hexdigest(),
        }}}
        self._retain_manifest()

    def _retain_manifest(self):
        import hashlib
        self.manifest.chmod(0o600) if self.manifest.exists() else None
        raw = json.dumps(self.index, sort_keys=True).encode("utf-8")
        self.manifest.write_bytes(raw)
        self.manifest.chmod(0o400)
        self.options = dict(
            private_root=self.root, manifest_path=self.manifest,
            manifest_bytes=len(raw), manifest_sha256=hashlib.sha256(raw).hexdigest(),
            artifact_name="unit", max_manifest_bytes=4096,
            max_artifact_bytes=4096, max_total_bytes=8192,
        )

    def _assert_no_authority(self, result):
        for key in ("delivery_assessed", "processing_confirmed", "completion_eligible",
                    "execution_granted", "claim_granted", "snapshot_atomic"):
            self.assertIs(result[key], False, key)
        self.assertIs(result["read_only"], True)

    def test_pinned_manifest_recovers_exact_missing_leaf_reference(self):
        import base64
        from unittest.mock import patch
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("no RPC")):
            result = mod.read_codex_handoff_manifest_artifact(**self.options)
        self.assertEqual(result["artifact_state"], "verified")
        self.assertIs(result["manifest_verified"], True)
        self.assertEqual(base64.b64decode(result["payload_base64"], validate=True), self.body)
        self.assertEqual(result["sha256"], self.index["artifacts"]["unit"]["sha256"])
        self.assertEqual(result["manifest"]["sha256"], self.options["manifest_sha256"])
        self._assert_no_authority(result)

    def test_manifest_pin_and_target_pin_changes_refuse(self):
        for change in ("manifest_hash", "manifest_bytes", "target_hash", "target_bytes"):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.index))
                options = dict(self.options)
                if change == "manifest_hash":
                    options["manifest_sha256"] = "0" * 64
                elif change == "manifest_bytes":
                    options["manifest_bytes"] += 1
                else:
                    row = self.index["artifacts"]["unit"]
                    row["sha256" if change == "target_hash" else "bytes"] = (
                        "0" * 64 if change == "target_hash" else len(self.body) + 1)
                    self._retain_manifest()
                    options = dict(self.options)
                result = mod.read_codex_handoff_manifest_artifact(**options)
                self.assertNotEqual(result["artifact_state"], "verified")
                self.assertIsNone(result["payload_base64"])
                self._assert_no_authority(result)
                self.index = original
                self._retain_manifest()

    def test_payload_claims_do_not_become_reader_authority(self):
        import hashlib
        self.body = b'{"claim_granted":true,"legal_admission":true,"completion_eligible":true}'
        self.target.chmod(0o600)
        self.target.write_bytes(self.body)
        self.target.chmod(0o400)
        self.index["artifacts"]["unit"].update(
            bytes=len(self.body), sha256=hashlib.sha256(self.body).hexdigest())
        self._retain_manifest()
        result = mod.read_codex_handoff_manifest_artifact(**self.options)
        self.assertEqual(result["artifact_state"], "verified")
        self._assert_no_authority(result)

    def test_manifest_shape_utf8_nonfinite_and_row_contract_refuse(self):
        import hashlib
        raws = [b"[]", b"{}", b"{\"artifacts\": []}", b"\xff",
                b'{"artifacts":{"unit":NaN}}']
        for mutation in ("authority", "bool_size", "extra_row", "too_many"):
            index = json.loads(json.dumps(self.index))
            if mutation == "authority": index["execution_granted"] = True
            elif mutation == "bool_size": index["artifacts"]["unit"]["bytes"] = True
            elif mutation == "extra_row": index["artifacts"]["unit"]["claim_granted"] = True
            else: index["artifacts"] = {str(i): index["artifacts"]["unit"] for i in range(65)}
            raws.append(json.dumps(index).encode())
        for raw in raws:
            with self.subTest(raw=raw[:100]):
                self.manifest.chmod(0o600)
                self.manifest.write_bytes(raw)
                self.manifest.chmod(0o400)
                options = {**self.options, "manifest_bytes": len(raw),
                           "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                           "max_manifest_bytes": 16384, "max_total_bytes": 32768}
                result = mod.read_codex_handoff_manifest_artifact(**options)
                self.assertNotEqual(result["artifact_state"], "verified")
                self.assertIsNone(result["payload_base64"])
                self._assert_no_authority(result)

    def test_request_types_and_budgets_refuse_before_any_file_read(self):
        from unittest.mock import patch
        changes = [dict(artifact_name=""), dict(artifact_name="../unit"),
                   dict(manifest_bytes=True), dict(max_manifest_bytes=True),
                   dict(max_artifact_bytes=0), dict(max_total_bytes=True),
                   dict(max_total_bytes=self.options["manifest_bytes"] - 1),
                   dict(manifest_sha256="F" * 64)]
        for change in changes:
            with self.subTest(change=change):
                with patch.object(mod, "read_codex_handoff_artifact", side_effect=AssertionError("no read")) as reader:
                    result = mod.read_codex_handoff_manifest_artifact(**{**self.options, **change})
                reader.assert_not_called()
                self.assertEqual(result["artifact_state"], "refused")
                self.assertIsNone(result["payload_base64"])

    def test_symlink_hardlink_and_nonprivate_root_refuse(self):
        for kind in ("symlink", "hardlink", "nonprivate_root"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o700)
                target = root / "unit.json"
                target.write_bytes(self.body)
                if kind == "symlink":
                    source = root / "source.json"
                    target.rename(source)
                    target.symlink_to(source)
                elif kind == "hardlink": os.link(target, root / "other.json")
                else: root.chmod(0o755)
                manifest = root / "handoff.json"
                raw = json.dumps(self.index).encode()
                manifest.write_bytes(raw)
                import hashlib
                result = mod.read_codex_handoff_manifest_artifact(**{
                    **self.options, "private_root": root, "manifest_path": manifest,
                    "manifest_bytes": len(raw), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                })
                self.assertNotEqual(result["artifact_state"], "verified")
                self.assertIsNone(result["payload_base64"])
                self._assert_no_authority(result)

    def test_cli_routes_to_manifest_reader_without_transport_or_wrapper(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        args = ["read-manifest-artifact", "--private-root", str(self.root),
                "--manifest-path", str(self.manifest),
                "--manifest-bytes", str(self.options["manifest_bytes"]),
                "--manifest-sha256", self.options["manifest_sha256"],
                "--artifact-name", "unit", "--max-manifest-bytes", "4096",
                "--max-artifact-bytes", "4096", "--max-total-bytes", "8192"]
        stdout = io.StringIO()
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("no RPC")), redirect_stdout(stdout):
            code = mod.codex_thread_handoff_main(args)
        self.assertEqual(code, 0)
        outcome = json.loads(stdout.getvalue())
        self.assertEqual(outcome["artifact_state"], "verified")
        self._assert_no_authority(outcome)

    def test_failed_alias_leaves_namespace_and_receipts_unchanged(self):
        receipt = self.root / "receipt.json"
        receipt.write_bytes(b'{"delivery_state":"delivery_unknown"}')
        before = {p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_ino) for p in self.root.iterdir()}
        result = mod.read_codex_handoff_manifest_artifact(**{**self.options, "artifact_name": "missing"})
        self.assertNotEqual(result["artifact_state"], "verified")
        after = {p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_ino) for p in self.root.iterdir()}
        self.assertEqual(after, before)
        self._assert_no_authority(result)

    def test_cli_failure_and_stdout_loss_never_send(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        args = ["read-manifest-artifact", "--private-root", str(self.root),
                "--manifest-path", str(self.manifest),
                "--manifest-bytes", str(self.options["manifest_bytes"]),
                "--manifest-sha256", self.options["manifest_sha256"],
                "--artifact-name", "missing"]
        stdout = io.StringIO()
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("no RPC")) as rpc, redirect_stdout(stdout):
            self.assertEqual(mod.codex_thread_handoff_main(args), 2)
        rpc.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["artifact_state"], "refused")
        class LostOutput:
            def write(self, value): raise OSError("fictional output loss")
            def flush(self): raise AssertionError("must not retry output")
        with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("no RPC")) as rpc, patch("sys.stdout", LostOutput()):
            self.assertEqual(mod.codex_thread_handoff_main(args), 1)
        rpc.assert_not_called()


class ManifestAdversarialRecoveryTests(unittest.TestCase):
    """Read real pinned files; malformed selection never probes another artifact."""

    def _manifest_private_root(self):
        temporary = tempfile.TemporaryDirectory(prefix="manifest-adversarial-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        os.chmod(root, 0o700)
        return root

    def _manifest_file(self, root, name, body):
        path = root / name
        path.write_bytes(body)
        os.chmod(path, 0o600)
        return path

    def _manifest_row(self, filename, body):
        import hashlib
        return {"file": filename, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest()}

    def _manifest_request(self, root, body, *, artifact_name="unit", **bounds):
        import hashlib
        manifest = self._manifest_file(root, "manifest.json", body)
        request = {
            "private_root": str(root), "manifest_path": str(manifest),
            "manifest_bytes": len(body),
            "manifest_sha256": hashlib.sha256(body).hexdigest(),
            "artifact_name": artifact_name,
            "max_manifest_bytes": 131072, "max_artifact_bytes": 131072,
            "max_total_bytes": 262144,
        }
        request.update(bounds)
        return manifest, request

    def _manifest_assert_refused(self, result):
        self.assertNotEqual(result.get("artifact_state"), "verified")
        self.assertIsNone(result.get("payload_base64"))

    def _manifest_assert_only_manifest_read(self, reader, manifest):
        self.assertLessEqual(reader.call_count, 1)
        for call in reader.call_args_list:
            self.assertEqual(Path(call.kwargs["artifact_path"]), manifest)

    def test_duplicate_json_keys_at_each_manifest_level_are_refused(self):
        payload = b'{"authored":"unit"}\n'
        row = json.dumps(self._manifest_row("unit.json", payload), separators=(",", ":"))
        entry_duplicate = row[:-1] + ',"bytes":' + str(len(payload)) + "}"
        documents = {
            "top": '{"artifacts":{"unit":' + row + '},"artifacts":{"unit":' + row + '}}',
            "alias": '{"artifacts":{"unit":' + row + ',"unit":' + row + '}}',
            "entry": '{"artifacts":{"unit":' + entry_duplicate + '}}',
        }
        real_reader = mod.read_codex_handoff_artifact
        for level, document in documents.items():
            with self.subTest(level=level):
                root = self._manifest_private_root()
                self._manifest_file(root, "unit.json", payload)
                manifest, request = self._manifest_request(root, document.encode("utf-8"))
                with patch.object(mod, "read_codex_handoff_artifact", wraps=real_reader) as reader:
                    result = mod.read_codex_handoff_manifest_artifact(**request)
                self._manifest_assert_refused(result)
                self._manifest_assert_only_manifest_read(reader, manifest)

    def test_traversal_absolute_and_self_target_never_open_an_outside_artifact(self):
        payload = b'{"authored":"outside-sentinel"}\n'
        outside_root = self._manifest_private_root()
        outside = self._manifest_file(outside_root, "must-not-open-outside.json", payload)
        real_open = os.open
        real_reader = mod.read_codex_handoff_artifact

        def observed_open(path, *args, **kwargs):
            try:
                text = os.fsdecode(os.fspath(path))
            except TypeError:
                return real_open(path, *args, **kwargs)
            if ".." in Path(text).parts or Path(text).name == outside.name:
                raise AssertionError("manifest selection tried to open outside its private root")
            return real_open(path, *args, **kwargs)

        for label, filename in (
            ("traversal", "../" + outside.name),
            ("absolute", str(outside)),
            ("self_target", "manifest.json"),
        ):
            with self.subTest(selection=label):
                root = self._manifest_private_root()
                document = json.dumps({"artifacts": {"unit": self._manifest_row(filename, payload)}}).encode()
                manifest, request = self._manifest_request(root, document)
                with patch.object(mod, "read_codex_handoff_artifact", wraps=real_reader) as reader:
                    with patch.object(mod.os, "open", side_effect=observed_open):
                        result = mod.read_codex_handoff_manifest_artifact(**request)
                self._manifest_assert_refused(result)
                self._manifest_assert_only_manifest_read(reader, manifest)

    def test_combined_byte_budget_refuses_before_touching_target(self):
        root = self._manifest_private_root()
        payload = b'{"authored":"budget-target"}\n'
        target = self._manifest_file(root, "unit.json", payload)
        document = json.dumps({"artifacts": {"unit": self._manifest_row(target.name, payload)}}).encode()
        manifest, request = self._manifest_request(
            root, document, max_manifest_bytes=len(document),
            max_artifact_bytes=len(payload), max_total_bytes=len(document) + len(payload) - 1,
        )
        real_reader = mod.read_codex_handoff_artifact
        real_open = os.open

        def observed_open(path, *args, **kwargs):
            try:
                name = Path(os.fsdecode(os.fspath(path))).name
            except TypeError:
                return real_open(path, *args, **kwargs)
            if name == target.name:
                raise AssertionError("total-budget refusal touched the target")
            return real_open(path, *args, **kwargs)

        with patch.object(mod, "read_codex_handoff_artifact", wraps=real_reader) as reader:
            with patch.object(mod.os, "open", side_effect=observed_open):
                result = mod.read_codex_handoff_manifest_artifact(**request)
        self._manifest_assert_refused(result)
        self._manifest_assert_only_manifest_read(reader, manifest)

    def test_missing_or_invalid_selected_alias_never_falls_back_to_other_artifacts(self):
        payloads = {"unit.json": b'{"authored":"unit"}\n',
                    "backup.json": b'{"authored":"backup"}\n',
                    "second.json": b'{"authored":"second"}\n'}
        valid_rows = {name.removesuffix(".json"): self._manifest_row(name, body)
                      for name, body in payloads.items()}
        invalid_row = dict(valid_rows["unit"], unexpected=True)
        cases = (
            ("missing", "absent", valid_rows),
            ("invalid_row", "unit", dict(valid_rows, unit=invalid_row)),
            ("nonobject_row", "unit", dict(valid_rows, unit=None)),
        )
        real_reader = mod.read_codex_handoff_artifact
        real_open = os.open

        def observed_open(path, *args, **kwargs):
            try:
                name = Path(os.fsdecode(os.fspath(path))).name
            except TypeError:
                return real_open(path, *args, **kwargs)
            if name in payloads:
                raise AssertionError("invalid selection probed a manifest artifact")
            return real_open(path, *args, **kwargs)

        for label, selected, rows in cases:
            with self.subTest(selection=label):
                root = self._manifest_private_root()
                for name, body in payloads.items():
                    self._manifest_file(root, name, body)
                document = json.dumps({"artifacts": rows}).encode()
                manifest, request = self._manifest_request(root, document, artifact_name=selected)
                with patch.object(mod, "read_codex_handoff_artifact", wraps=real_reader) as reader:
                    with patch.object(mod.os, "open", side_effect=observed_open):
                        result = mod.read_codex_handoff_manifest_artifact(**request)
                self._manifest_assert_refused(result)
                self._manifest_assert_only_manifest_read(reader, manifest)


class OwnerDependencyReadTests(unittest.TestCase):
    """One observed coordinator turn; no transport mutation or custody inference."""

    def _connection(self, status="inProgress", *, turn="owner-turn", thread="owner-thread", fail=None):
        class Client:
            def __init__(client):
                client.messages, client.closed = [], False
                client.socket = client
            def settimeout(client, seconds):
                if not 0 < seconds <= 60: raise AssertionError("unbounded read")
            def __enter__(client): return client
            def __exit__(client, *args): client.closed = True
            def send(client, raw): client.messages.append(json.loads(raw))
            def recv(client, timeout):
                request = client.messages[-1]
                method = request["method"]
                if method == fail: raise TimeoutError("fictional metadata observation timeout")
                results = {
                    "initialize": {},
                    "thread/read": {"thread": {"id": thread, "status": "active", "canAcceptDirectInput": True}},
                    "thread/turns/list": {"data": [{"id": turn, "status": status}]},
                }
                return json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": results[method]})
        return Client()

    def _read(self, client, **changes):
        options = dict(socket_path="/fixture/owner.sock", thread_id="owner-thread", expected_turn_id="owner-turn")
        options.update(changes)
        with patch.object(mod, "_open_codex_app_connection", return_value=client):
            result = mod.read_codex_owner_dependency(**options)
        self.assertTrue(client.closed)
        self.assertEqual([r["method"] for r in client.messages],
                         ["initialize", "initialized", "thread/read", "thread/turns/list"])
        self._no_grants(result)
        return result

    def _no_grants(self, result):
        self.assertIs(result["read_only"], True)
        for key in ("work_complete", "completion_authority", "completion_eligible",
                    "execution_granted", "claim_granted", "goal_status_changed",
                    "owner_custody_verified", "native_or_child_liveness_verified", "budget_reset"):
            self.assertIs(result[key], False, key)

    def test_exact_observed_inprogress_owner_turn_is_a_local_wait(self):
        client = self._connection()
        result = self._read(client)
        self.assertEqual(result["dependency_state"], "owner_turn_in_progress")
        self.assertEqual(result["action"], "await_owner_result")
        self.assertEqual(result["owner_handle"], {"kind": "codex_turn", "thread_id": "owner-thread", "turn_id": "owner-turn"})
        self.assertIs(result["observation"]["snapshot_atomic"], False)

    def test_terminal_turns_require_result_reconciliation_not_completion(self):
        for status in ("completed", "failed", "interrupted"):
            with self.subTest(status=status):
                result = self._read(self._connection(status))
                self.assertEqual(result["dependency_state"], "owner_turn_terminal")
                self.assertEqual(result["action"], "reconcile_owner_result")
                self.assertFalse(result["work_complete"])

    def test_changed_turn_never_adopts_the_new_owner_handle(self):
        result = self._read(self._connection(turn="replacement-turn"))
        self.assertEqual(result["dependency_state"], "owner_turn_changed")
        self.assertEqual(result["action"], "reconcile_owner_binding")
        self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")
        self.assertEqual(result["observation"]["latest_turn"]["id"], "replacement-turn")

    def test_invalid_owner_bindings_refuse_before_any_probe(self):
        for changes in ({"expected_turn_id": None}, {"expected_turn_id": True},
                        {"expected_turn_id": "../owner"}, {"expected_turn_id": ""},
                        {"thread_id": "other/thread"}, {"thread_id": False}):
            with self.subTest(changes=changes):
                with patch.object(mod, "read_codex_thread_state", side_effect=AssertionError("no probe")) as reader:
                    result = mod.read_codex_owner_dependency(**{
                        "socket_path": "/fixture/owner.sock", "thread_id": "owner-thread",
                        "expected_turn_id": "owner-turn", **changes})
                reader.assert_not_called()
                self.assertEqual(result["dependency_state"], "refused")
                self._no_grants(result)

    def test_observation_timeouts_never_prove_owner_stopped_or_retry_safe(self):
        for stage in ("initialize", "thread/turns/list"):
            with self.subTest(stage=stage):
                client = self._connection(fail=stage)
                with patch.object(mod, "_open_codex_app_connection", return_value=client):
                    result = mod.read_codex_owner_dependency(
                        socket_path="/fixture/owner.sock", thread_id="owner-thread", expected_turn_id="owner-turn")
                self.assertTrue(client.closed)
                self.assertEqual(result["dependency_state"], "unknown")
                self.assertEqual(result["action"], "reobserve_same_owner_handle")
                self._no_grants(result)
                self.assertTrue(all(r["method"] in {"initialize", "initialized", "thread/read", "thread/turns/list"} for r in client.messages))

    def test_cli_exit_dispositions_and_lost_output_never_mutate_or_replay(self):
        import io
        from contextlib import redirect_stdout
        args = ["read-owner", "--socket-path", "/fixture/owner.sock", "--thread-id", "owner-thread", "--expected-turn-id", "owner-turn"]
        for status, turn, expected in (("inProgress", "owner-turn", 4), ("completed", "owner-turn", 0),
                                       ("inProgress", "replacement-turn", 3), ("unrecognized", "owner-turn", 1)):
            with self.subTest(status=status, turn=turn):
                client = self._connection(status, turn=turn)
                output = io.StringIO()
                with patch.object(mod, "_open_codex_app_connection", return_value=client) as opening, redirect_stdout(output):
                    code = mod.codex_thread_handoff_main(args)
                self.assertEqual(code, expected)
                self.assertEqual(opening.call_count, 1)
                self._no_grants(json.loads(output.getvalue()))
                self.assertTrue(all(r["method"] in {"initialize", "initialized", "thread/read", "thread/turns/list"} for r in client.messages))
        class LostOutput:
            def write(self, value): raise OSError("fictional stdout loss")
            def flush(self): raise AssertionError("no output retry")
        client = self._connection()
        with patch.object(mod, "_open_codex_app_connection", return_value=client) as opening, patch("sys.stdout", LostOutput()):
            self.assertEqual(mod.codex_thread_handoff_main(args), 1)
        self.assertEqual(opening.call_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_bytes(b'{"delivery_state":"delivery_unknown"}')
            before = (receipt.read_bytes(), receipt.stat().st_ino, receipt.stat().st_mode)
            self._read(self._connection())
            self.assertEqual((receipt.read_bytes(), receipt.stat().st_ino, receipt.stat().st_mode), before)

    def test_unhashable_or_nonstr_observation_state_stays_unknown(self):
        helper = OwnerDependencyAdversarialTests()
        for state in ([], {}, None, False):
            with self.subTest(state=state):
                observation = helper._dependency_observation()
                observation["observation_state"] = state
                result = helper._dependency_call(observation)
                helper._dependency_assert_unknown(result)


class OwnerDependencyAdversarialTests(unittest.TestCase):
    """Fictional reader metadata cannot establish owner custody or task authority."""

    def _dependency_observation(self):
        return {
            "schema": "codex-existing-thread-state/v1",
            "observation_state": "observed", "reason": "metadata_observed",
            "thread_id": "owner-thread",
            "thread_snapshot": {"id": "owner-thread", "status": "active",
                                "can_accept_direct_input": True},
            "latest_turn": {"id": "owner-turn", "status": "inProgress"},
            "read_only": True, "snapshot_atomic": False, "history_complete": False,
            "delivery_assessed": False, "processing_confirmed": False,
            "completion_eligible": False, "execution_granted": False, "claim_granted": False,
        }

    def _dependency_call(self, observation):
        with patch.object(mod, "read_codex_thread_state", return_value=observation) as reader:
            with patch.object(mod, "_open_codex_app_connection",
                              side_effect=AssertionError("schema controls must not open a socket")) as connect:
                with patch.object(mod, "send_codex_thread_message",
                                  side_effect=AssertionError("dependency reads must never send")) as send:
                    result = mod.read_codex_owner_dependency(
                        socket_path="/tmp/owner-dependency-do-not-connect.sock",
                        thread_id="owner-thread", expected_turn_id="owner-turn",
                    )
        reader.assert_called_once_with(
            socket_path="/tmp/owner-dependency-do-not-connect.sock",
            thread_id="owner-thread", timeout_seconds=15.0, max_response_bytes=262144,
        )
        connect.assert_not_called()
        send.assert_not_called()
        self.assertIs(result["read_only"], True)
        for field in ("work_complete", "completion_authority", "completion_eligible",
                      "execution_granted", "claim_granted", "goal_status_changed",
                      "owner_custody_verified", "native_or_child_liveness_verified", "budget_reset"):
            self.assertIs(result[field], False, field)
        self.assertEqual(result["owner_handle"], {
            "kind": "codex_turn", "thread_id": "owner-thread", "turn_id": "owner-turn",
        })
        return result

    def _dependency_assert_not_decisive(self, result):
        self.assertIn(result["dependency_state"], {"refused", "unknown"})
        self.assertIn(result["action"], {"fix_request", "reobserve_same_owner_handle"})

    def _dependency_assert_unknown(self, result):
        self.assertEqual(result["dependency_state"], "unknown")
        self.assertEqual(result["action"], "reobserve_same_owner_handle")

    def test_malformed_or_elevated_reader_observation_cannot_become_owner_state(self):
        cases = [("not_an_object", None), ("list", []),
                 ("wrong_schema", {**self._dependency_observation(), "schema": "other/v1"}),
                 ("missing_schema", {key: value for key, value in self._dependency_observation().items()
                                     if key != "schema"}),
                 ("extra_authority", {**self._dependency_observation(), "work_complete": True}),
                 ("invalid_observation_state", {**self._dependency_observation(),
                                                 "observation_state": "completed"})]
        for value in (False, 1, None):
            cases.append(("read_only_" + repr(value),
                          {**self._dependency_observation(), "read_only": value}))
        for field in ("snapshot_atomic", "history_complete", "delivery_assessed",
                      "processing_confirmed", "completion_eligible", "execution_granted", "claim_granted"):
            for value in (True, 0, None):
                cases.append((field + "_" + repr(value),
                              {**self._dependency_observation(), field: value}))
        for label, observation in cases:
            with self.subTest(observation=label):
                self._dependency_assert_not_decisive(self._dependency_call(observation))

    def test_thread_and_snapshot_identity_mismatches_never_rebind_the_owner(self):
        cases = []
        for value in ("other-thread", None, 3):
            cases.append(("outer_" + repr(value), {**self._dependency_observation(), "thread_id": value}))
        for value in ("other-thread", "", None, 3):
            observation = self._dependency_observation()
            observation["thread_snapshot"]["id"] = value
            cases.append(("snapshot_" + repr(value), observation))
        observation = self._dependency_observation()
        del observation["thread_snapshot"]["id"]
        cases.append(("snapshot_id_missing", observation))
        for label, observation in cases:
            with self.subTest(identity=label):
                self._dependency_assert_not_decisive(self._dependency_call(observation))

    def test_partial_or_ambiguous_metadata_cannot_prove_an_owner_turn_live_or_stopped(self):
        cases = []
        for state in ("partial", "unavailable"):
            observation = self._dependency_observation()
            observation.update(observation_state=state, reason="latest_turn_unavailable", latest_turn=None)
            cases.append((state, observation))
        observation = self._dependency_observation()
        observation["latest_turn"] = None
        cases.append(("active_thread_without_a_turn", observation))
        observation = self._dependency_observation()
        observation["observation_state"] = "partial"
        cases.append(("partial_with_retained_turn_metadata", observation))
        for status in ("queued", "futureStatus", "unknown"):
            observation = self._dependency_observation()
            observation["latest_turn"]["status"] = status
            cases.append(("unknown_turn_status_" + status, observation))
        observation = self._dependency_observation()
        observation["thread_snapshot"] = None
        cases.append(("turn_without_thread_snapshot", observation))
        for label, observation in cases:
            with self.subTest(metadata=label):
                self._dependency_assert_unknown(self._dependency_call(observation))

    def test_inprogress_turn_with_nonactive_thread_snapshot_remains_unknown(self):
        for status in ("idle", "notLoaded", "systemError"):
            with self.subTest(thread_status=status):
                observation = self._dependency_observation()
                observation["thread_snapshot"]["status"] = status
                self._dependency_assert_unknown(self._dependency_call(observation))


class OwnerTransportFallbackTests(unittest.TestCase):
    """An explicit failed read may observe the same owner once through Unix RPC."""

    _false_flags = (
        "work_complete", "completion_authority", "completion_eligible",
        "execution_granted", "claim_granted", "goal_status_changed",
        "owner_custody_verified", "native_or_child_liveness_verified", "budget_reset",
    )

    def _error(self, method="read_thread", host="127.0.0.1", port=31277):
        return (
            "tool call error: tool call failed for `codex_tui/" + method + "`\n"
            "Caused by:\nTransport send error: error sending request for url "
            "(http://" + host + ":" + str(port) + "/mcp)"
        )

    def _request(self, **changes):
        return {
            "socket_path": "/fixture/owner.sock", "thread_id": "owner-thread",
            "expected_turn_id": "owner-turn", "mcp_method": "read_thread",
            "transport_error": self._error(), **changes,
        }

    def _owner(self, state="owner_turn_in_progress"):
        disposition = {
            "owner_turn_in_progress": ("await_owner_result", "exact_owner_turn_observed_in_progress"),
            "owner_turn_terminal": ("reconcile_owner_result", "owner_turn_terminal_result_not_assessed"),
            "owner_turn_changed": ("reconcile_owner_binding", "expected_owner_turn_differs"),
            "unknown": ("reobserve_same_owner_handle", "owner_observation_incomplete"),
            "refused": ("fix_request", "owner_probe_refused"),
        }
        action, reason = disposition[state]
        observation = OwnerDependencyAdversarialTests()._dependency_observation()
        if state == "owner_turn_terminal":
            observation["latest_turn"]["status"] = "completed"
        elif state == "owner_turn_changed":
            observation["latest_turn"]["id"] = "replacement-turn"
        elif state == "unknown":
            observation.update(observation_state="partial", latest_turn=None)
        return {
            "schema": "codex-owner-dependency-observation/v1",
            "dependency_state": state, "action": action, "reason": reason,
            "owner_handle": {"kind": "codex_turn", "thread_id": "owner-thread", "turn_id": "owner-turn"},
            "observation": observation, "read_only": True,
            **{key: False for key in self._false_flags},
        }

    def _assert_inert(self, result):
        self.assertEqual(result["schema"], "codex-owner-transport-fallback/v1")
        self.assertEqual(set(result), {
            "schema", "fallback_state", "reason", "action", "fallback_attempted",
            "mcp_failure", "owner_handle", "owner_result", "read_only", *self._false_flags,
        })
        self.assertIs(result["read_only"], True)
        for field in self._false_flags:
            self.assertIs(result[field], False, field)

    def _call(self, owner=None, **changes):
        with patch.object(mod, "read_codex_owner_dependency", return_value=self._owner() if owner is None else owner) as reader:
            with patch.object(mod, "_open_codex_app_connection", side_effect=AssertionError("delegation must not open another connection")) as connect:
                with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("status fallback must never send")) as send:
                    with patch.object(mod, "OpenAICodexCLIIntegration", side_effect=AssertionError("status fallback must never construct generation")) as generator:
                        result = mod.read_codex_owner_after_mcp_failure(**self._request(**changes))
        connect.assert_not_called()
        send.assert_not_called()
        generator.assert_not_called()
        self._assert_inert(result)
        return result, reader

    def _assert_refused_before_read(self, **changes):
        result, reader = self._call(**changes)
        self.assertEqual(result["fallback_state"], "refused")
        self.assertIs(result["fallback_attempted"], False)
        reader.assert_not_called()
        return result

    def test_allowed_read_failures_delegate_once_and_preserve_original_evidence(self):
        import hashlib
        for method in ("read_thread", "wait_threads"):
            for host in ("127.0.0.1", "localhost", "[::1]"):
                with self.subTest(method=method, host=host):
                    error = self._error(method, host) + "\nfictional detail: é"
                    result, reader = self._call(mcp_method=method, transport_error=error,
                                                timeout_seconds=2.75, max_response_bytes=8192)
                    reader.assert_called_once_with(
                        socket_path="/fixture/owner.sock", thread_id="owner-thread",
                        expected_turn_id="owner-turn", timeout_seconds=2.75, max_response_bytes=8192,
                    )
                    self.assertEqual(result["fallback_state"], "observed")
                    self.assertIs(result["fallback_attempted"], True)
                    self.assertEqual(result["mcp_failure"], {
                        "method": method, "error": error,
                        "sha256": hashlib.sha256(error.encode("utf-8")).hexdigest(),
                    })
                    self.assertEqual(result["owner_handle"], self._owner()["owner_handle"])

    def test_terminal_and_changed_owners_preserve_disposition_without_rebinding(self):
        for state in ("owner_turn_in_progress", "owner_turn_terminal", "owner_turn_changed"):
            with self.subTest(state=state):
                owner = self._owner(state)
                result, reader = self._call(owner)
                reader.assert_called_once()
                self.assertEqual(result["fallback_state"], "observed")
                self.assertEqual(result["owner_result"], owner)
                self.assertEqual(result["action"], owner["action"])
                self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")

    def test_nonread_and_mismatched_methods_refuse_before_read(self):
        for method in ("send_message", "start_thread", "resume_thread", "read-owner", "read_thread ", None, True):
            with self.subTest(method=method):
                self._assert_refused_before_read(mcp_method=method)
        self._assert_refused_before_read(mcp_method="wait_threads", transport_error=self._error("read_thread"))
        self._assert_refused_before_read(transport_error=self._error("wait_threads"))

    def test_nontransport_successful_or_nonloopback_evidence_refuses_before_read(self):
        error = self._error()
        cases = [
            "connection refused", "Transport send error: disconnected",
            error.replace("tool call error: ", ""),
            " " + error,
            error.replace("Transport send error:", "MCP application error:"),
            error.replace("http://127.0.0.1:", "https://127.0.0.1:"),
            error.replace("http://127.0.0.1:", "http://example.com:"),
            error.replace("/mcp)", "/other)"),
            json.dumps({"result": {"status": "completed"}, "diagnostic": error}),
            {"error": error}, None, False,
        ]
        for value in cases:
            with self.subTest(evidence=repr(value)):
                self._assert_refused_before_read(transport_error=value)
        for port in (0, 65536, -1, "not-a-port"):
            with self.subTest(port=port):
                self._assert_refused_before_read(transport_error=self._error(port=port))

    def test_policy_safety_and_refusal_text_never_authorizes_fallback(self):
        for suffix in ("\nPolicy refusal: do not retry this request.",
                       "\nSafety refusal: this operation is disallowed.",
                       "\nRequest refused by the policy checker."):
            with self.subTest(suffix=suffix):
                self._assert_refused_before_read(transport_error=self._error() + suffix)

    def test_invalid_owner_inputs_and_bounds_refuse_before_read(self):
        cases = [
            {"socket_path": "relative.sock"}, {"socket_path": "/tmp/../owner.sock"},
            {"socket_path": "/" + "x" * 108}, {"socket_path": "/tmp/owner\0.sock"},
            {"socket_path": None}, {"thread_id": "bad/thread"}, {"thread_id": True},
            {"expected_turn_id": ""}, {"expected_turn_id": "../owner"}, {"expected_turn_id": None},
            {"timeout_seconds": True}, {"timeout_seconds": 0}, {"timeout_seconds": 61},
            {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
            {"max_response_bytes": True}, {"max_response_bytes": 4095},
            {"max_response_bytes": 4194305},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                self._assert_refused_before_read(**changes)

    def test_error_utf8_budget_is_checked_without_truncation_or_coercion(self):
        error = self._error()
        exact = error + "\n" + "x" * (8192 - len(error.encode("utf-8")) - 1)
        result, reader = self._call(transport_error=exact)
        reader.assert_called_once()
        self.assertEqual(result["mcp_failure"]["error"], exact)
        for value in (exact + "x", error + "é" * 4096, error + "\ud800"):
            with self.subTest(length=len(value)):
                self._assert_refused_before_read(transport_error=value)

    def test_unknown_owner_observation_remains_unavailable_without_retry(self):
        owner = self._owner("unknown")
        result, reader = self._call(owner)
        reader.assert_called_once()
        self.assertEqual(result["fallback_state"], "unavailable")
        self.assertEqual(result["action"], "reobserve_same_owner_handle")
        self.assertEqual(result["owner_result"], owner)
        self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")

    def test_refused_owner_probe_is_never_retried_or_treated_as_observed(self):
        result, reader = self._call(self._owner("refused"))
        reader.assert_called_once()
        self.assertIn(result["fallback_state"], {"refused", "unavailable"})
        self.assertIs(result["fallback_attempted"], True)
        self.assertEqual(result["action"], "fix_request")
        self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")

    def test_malformed_or_authoritative_owner_result_stays_unavailable(self):
        for owner in (None, [], {**self._owner(), "work_complete": True},
                      {**self._owner(), "read_only": 1}):
            with self.subTest(owner=owner):
                with patch.object(mod, "read_codex_owner_dependency", return_value=owner) as reader:
                    result = mod.read_codex_owner_after_mcp_failure(**self._request())
                reader.assert_called_once()
                self._assert_inert(result)
                self.assertEqual(result["fallback_state"], "unavailable")
                self.assertEqual(result["action"], "reobserve_same_owner_handle")
                self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")

    def test_owner_probe_exception_is_unknown_redacted_and_never_replayed(self):
        class Hostile(Exception):
            def __str__(self):
                raise AssertionError("must not format fallback exception")
        with patch.object(mod, "read_codex_owner_dependency", side_effect=Hostile()) as reader:
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no replay")) as send:
                result = mod.read_codex_owner_after_mcp_failure(**self._request())
        reader.assert_called_once()
        send.assert_not_called()
        self._assert_inert(result)
        self.assertEqual(result["fallback_state"], "unavailable")
        self.assertIs(result["fallback_attempted"], True)
        self.assertEqual(result["action"], "reobserve_same_owner_handle")
        self.assertEqual(result["owner_handle"]["turn_id"], "owner-turn")

    def test_connection_timeout_stays_unknown_with_one_connection(self):
        with patch.object(mod, "_open_codex_app_connection", side_effect=TimeoutError("fictional connect timeout")) as connect:
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no send")) as send:
                result = mod.read_codex_owner_after_mcp_failure(**self._request())
        connect.assert_called_once()
        send.assert_not_called()
        self._assert_inert(result)
        self.assertEqual(result["fallback_state"], "unavailable")
        self.assertEqual(result["owner_result"]["dependency_state"], "unknown")
        self.assertEqual(result["action"], "reobserve_same_owner_handle")
        self.assertNotIn("fictional connect timeout", json.dumps(result))

    def test_thread_read_timeout_stays_unknown_without_steering_or_retry(self):
        client = OwnerDependencyReadTests()._connection(fail="thread/read")
        with patch.object(mod, "_open_codex_app_connection", return_value=client) as connect:
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no send")) as send:
                result = mod.read_codex_owner_after_mcp_failure(**self._request())
        connect.assert_called_once()
        send.assert_not_called()
        self.assertTrue(client.closed)
        self.assertEqual([item["method"] for item in client.messages], ["initialize", "initialized", "thread/read"])
        self._assert_inert(result)
        self.assertEqual(result["fallback_state"], "unavailable")
        self.assertEqual(result["owner_result"]["dependency_state"], "unknown")
        self.assertEqual(result["action"], "reobserve_same_owner_handle")

    def test_missing_websocket_dependency_is_reported_without_install_or_owner_inference(self):
        import builtins
        from types import SimpleNamespace
        original_import = builtins.__import__
        def missing_websocket(name, *args, **kwargs):
            if name == "websockets.sync.client":
                raise ModuleNotFoundError("fictional absent websocket package", name="websockets")
            return original_import(name, *args, **kwargs)
        socket_info = SimpleNamespace(st_mode=mod.stat.S_IFSOCK | 0o600, st_uid=os.geteuid())
        with patch.object(mod.os, "stat", return_value=socket_info):
            with patch("builtins.__import__", side_effect=missing_websocket):
                with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no send")) as send:
                    with patch.object(mod, "OpenAICodexCLIIntegration", side_effect=AssertionError("no generation or install")) as generator:
                        with patch("subprocess.run", side_effect=AssertionError("no dependency install")) as run:
                            with patch("subprocess.Popen", side_effect=AssertionError("no process start")) as popen:
                                thread = mod.read_codex_thread_state(socket_path="/fixture/owner.sock", thread_id="owner-thread")
                                owner = mod.read_codex_owner_dependency(
                                    socket_path="/fixture/owner.sock", thread_id="owner-thread", expected_turn_id="owner-turn")
                                fallback = mod.read_codex_owner_after_mcp_failure(**self._request())
        send.assert_not_called()
        generator.assert_not_called()
        run.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(thread["observation_state"], "unavailable")
        self.assertEqual(thread["reason"], "read_client_dependency_unavailable")
        self.assertEqual(owner["dependency_state"], "unknown")
        self.assertEqual(owner["reason"], "owner_client_dependency_unavailable")
        self.assertEqual(owner["action"], "reobserve_same_owner_handle")
        OwnerDependencyReadTests()._no_grants(owner)
        self._assert_inert(fallback)
        self.assertEqual(fallback["fallback_state"], "unavailable")
        self.assertEqual(fallback["owner_result"]["reason"], "owner_client_dependency_unavailable")

    def _cli_args(self, **changes):
        request = self._request(**changes)
        return [
            "read-owner-fallback", "--socket-path", request["socket_path"],
            "--thread-id", request["thread_id"], "--expected-turn-id", request["expected_turn_id"],
            "--mcp-method", request["mcp_method"], "--transport-error", request["transport_error"],
        ]

    def _cli_result(self):
        import hashlib
        error = self._error()
        owner = self._owner()
        return {
            "schema": "codex-owner-transport-fallback/v1", "fallback_state": "observed",
            "reason": "owner_observed_after_mcp_transport_failure", "action": owner["action"],
            "fallback_attempted": True,
            "mcp_failure": {"method": "read_thread", "error": error,
                            "sha256": hashlib.sha256(error.encode("utf-8")).hexdigest()},
            "owner_handle": owner["owner_handle"], "owner_result": owner, "read_only": True,
            **{key: False for key in self._false_flags},
        }

    def test_cli_forwards_default_bounds_and_original_evidence_once(self):
        from contextlib import redirect_stdout
        output = io.StringIO()
        with patch.object(mod, "read_codex_owner_after_mcp_failure", return_value=self._cli_result()) as fallback:
            with patch.object(mod, "OpenAICodexCLIIntegration", side_effect=AssertionError("no generation")):
                with redirect_stdout(output):
                    code = mod.codex_thread_handoff_main(self._cli_args())
        self.assertEqual(code, 4)
        fallback.assert_called_once_with(**self._request(), timeout_seconds=15.0, max_response_bytes=262144)
        self._assert_inert(json.loads(output.getvalue()))

    def test_cli_forwards_explicit_bounded_flags_once(self):
        from contextlib import redirect_stdout
        with patch.object(mod, "read_codex_owner_after_mcp_failure", return_value=self._cli_result()) as fallback:
            with redirect_stdout(io.StringIO()):
                code = mod.codex_thread_handoff_main(
                    self._cli_args() + ["--timeout-seconds", "2.75", "--max-response-bytes", "8192"])
        self.assertEqual(code, 4)
        fallback.assert_called_once_with(**self._request(), timeout_seconds=2.75, max_response_bytes=8192)

    def test_cli_refusal_never_observes_or_retries_the_owner(self):
        from contextlib import redirect_stdout
        output = io.StringIO()
        with patch.object(mod, "read_codex_owner_dependency", side_effect=AssertionError("no refused-work retry")) as reader:
            with patch.object(mod, "send_codex_thread_message", side_effect=AssertionError("no send")) as send:
                with redirect_stdout(output):
                    code = mod.codex_thread_handoff_main(self._cli_args(transport_error="policy refusal: request disallowed"))
        self.assertEqual(code, 2)
        reader.assert_not_called()
        send.assert_not_called()
        result = json.loads(output.getvalue())
        self._assert_inert(result)
        self.assertEqual(result["fallback_state"], "refused")
        self.assertIs(result["fallback_attempted"], False)

    def test_cli_invalid_bounds_refuse_before_any_owner_read(self):
        from contextlib import redirect_stdout
        for flags in (["--timeout-seconds", "nan"], ["--timeout-seconds", "61"],
                      ["--max-response-bytes", "4095"], ["--max-response-bytes", "4194305"]):
            with self.subTest(flags=flags):
                output = io.StringIO()
                with patch.object(mod, "read_codex_owner_dependency", side_effect=AssertionError("no invalid probe")) as reader:
                    with redirect_stdout(output):
                        code = mod.codex_thread_handoff_main(self._cli_args() + flags)
                self.assertEqual(code, 2)
                reader.assert_not_called()
                self.assertEqual(json.loads(output.getvalue())["fallback_state"], "refused")

    def test_cli_output_loss_never_repeats_probe_or_output(self):
        class LostOutput:
            def __init__(self): self.writes = 0
            def write(self, value):
                self.writes += 1
                raise OSError("fictional output loss")
            def flush(self): raise AssertionError("no output retry")
        output = LostOutput()
        with patch.object(mod, "read_codex_owner_after_mcp_failure", return_value=self._cli_result()) as fallback:
            with patch("sys.stdout", output):
                code = mod.codex_thread_handoff_main(self._cli_args())
        self.assertEqual(code, 1)
        self.assertEqual(output.writes, 1)
        fallback.assert_called_once()

    def test_cli_requires_transport_and_owner_flags_before_helper(self):
        for flag in ("--socket-path", "--thread-id", "--expected-turn-id", "--mcp-method", "--transport-error"):
            args = self._cli_args()
            index = args.index(flag)
            del args[index:index + 2]
            with self.subTest(flag=flag):
                with patch.object(mod, "read_codex_owner_after_mcp_failure") as fallback:
                    with patch("sys.stderr", io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            mod.codex_thread_handoff_main(args)
                self.assertEqual(raised.exception.code, 2)
                fallback.assert_not_called()
