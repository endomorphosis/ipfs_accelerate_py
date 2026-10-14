"""Run an explicitly captured two-round coding launch under one owner lease."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time


def run_candidate_coding_rounds(*, launch, command):
    """Wait and retire each child before considering the single continuation.

    Provider receipts remain the router's per-invocation observations. This
    record reports owner lifecycle only; a zero worker exit is not task success.
    """
    from .owner_candidate_dispatch import (
        CapturedCodingLaunch, prepare_second_coding_worker, retire_coding_worker,
    )
    if type(launch) is not CapturedCodingLaunch:
        raise ValueError("captured owner coding launch required")
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in signals}
    started = time.monotonic()
    rows = []
    failure = None
    code = 1

    def interrupted(signum, frame):
        raise InterruptedError("owner coding rounds interrupted")

    def handlers(value):
        for number in signals:
            signal.signal(number, value)

    def retire(process, broker, round_number, *, waited, suppress=False):
        handlers(signal.SIG_IGN)
        retired = False
        try:
            try:
                if process is not None and process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait()
            finally:
                retirement = retire_coding_worker(launch=launch, process=process, source_broker=broker)
            retired = True
            return retirement
        except BaseException:
            if not suppress:
                raise
        finally:
            rows.append(dict(round_number=round_number,
                returncode=None if process is None else process.returncode,
                wait_completed=waited, retirement_observed=retired))
            handlers(interrupted)

    def settle(process, broker, round_number):
        nonlocal code
        try:
            code = process.wait(timeout=launch.remaining_seconds)
        except BaseException:
            retire(process, broker, round_number, waited=False, suppress=True)
            raise
        return retire(process, broker, round_number, waited=True)

    try:
        handlers(interrupted)
        process = None
        try:
            launch.revalidate()
            launch.validate_first_command(command)
            process = subprocess.Popen(command)
            launch.source_broker.note_spawned(process)
        except BaseException:
            # Retirement runs its fixed isolated cleanup even if registration
            # fails. It must not mint a continuation from that failure.
            retire(process, launch.source_broker, 1, waited=False, suppress=True)
            raise
        retired = settle(process, launch.source_broker, 1)
        if type(code) is int and 1 <= code <= 255:
            prepared = prepare_second_coding_worker(launch=launch, retired_worker=retired)
            try:
                running = prepared.start()
                settle(running.process, running.source_broker, 2)
            finally:
                # A signal can arrive after start registers the child but
                # before its return value is assigned or settle is entered.
                # Recover custody from the prepared broker in that window.
                handlers(signal.SIG_IGN)
                pending_error = sys.exc_info()[0] is not None
                try:
                    child = prepared.source_broker._process
                    if child is not None and not any(row["round_number"] == 2 for row in rows):
                        retire(child, prepared.source_broker, 2, waited=False, suppress=pending_error)
                    prepared.close()
                except BaseException:
                    if not pending_error:
                        raise
                finally:
                    # Revoking the handoff does not certify descendant cleanup.
                    # A cleanup/custody failure remains the primary outcome.
                    closing_error = sys.exc_info()[0] is not None
                    try:
                        prepared.handoff.close()
                    except BaseException:
                        if not pending_error and not closing_error:
                            raise
        return code
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        # Never export the prompt, raw provider output or exception text here.
        observation = dict(schema="owner-candidate-rounds-observation@1", rounds=rows,
            max_rounds=2, elapsed_seconds=max(0.0, time.monotonic()-started),
            outcome="failed" if failure is not None else "workers_returned",
            failure_type=failure, completion_authority=False, proof_authority=False,
            source_obligations_discharged=0, accepted_index_published=False)
        try:
            print("[owner-candidate-rounds] "+json.dumps(observation, sort_keys=True, allow_nan=False), flush=True)
        except Exception:
            pass
