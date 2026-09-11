"""Retry only an owned, pure task projection across a Quack replica withdrawal.

The deadline is cooperative: interrupt can cancel DuckDB work, but cannot promise
termination of arbitrary kernel I/O. No mutation or borrowed connection enters
this retry loop, and a different owner generation is never silently adopted.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

READ_RETRY_SECONDS = 3.0
_BINDING_FIELDS = frozenset(
    {
        "server_id",
        "store_id",
        "database_uuid",
        "schema_revision",
        "generation",
        "process_birth_id",
        "listen_uri",
        "extension_fingerprint",
        "schema_fingerprint",
    }
)


class QuackReadUnavailable(RuntimeError):
    """The exact read could not finish within its original custody/deadline."""


@contextmanager
def owned_connection_deadline(connection, deadline):
    """Interrupt only this fresh connection; retire the timer before closing it."""
    if deadline is None:
        try:
            yield
        except Exception as error:
            from .duckdb_interrupts import reraise_process_interrupt

            reraise_process_interrupt(error)
            raise
        return
    if time.monotonic() >= deadline:
        raise QuackReadUnavailable("quack task read deadline exhausted")
    done = threading.Event()

    def expire():
        if done.wait(max(0.0, deadline - time.monotonic())):
            return
        # Repeated interrupt closes gaps between separate SELECT statements.
        while not done.is_set():
            try:
                connection.interrupt()
            except Exception:  # noqa: BLE001 - deadline still refuses a late result
                return
            done.wait(0.01)

    timer = threading.Thread(
        target=expire, name="quack-task-read-deadline", daemon=True
    )
    timer.start()
    try:
        yield
        if time.monotonic() >= deadline:
            raise QuackReadUnavailable("quack task read deadline exhausted")
    except Exception as error:
        from .duckdb_interrupts import reraise_process_interrupt

        reraise_process_interrupt(error)
        raise
    finally:
        done.set()
        timer.join()


def _replica_connection_lost(error):
    import duckdb

    if not isinstance(error, duckdb.IOException) or len(error.args) != 1:
        return False
    message = error.args[0]
    return (
        type(message) is str
        and len(message) <= 4096
        and message.startswith(
            "IO Error: Failed to send message: IO Error: Could not connect to server"
        )
    )


def read_task_projection(*, open_connection, read_projection, store_id, endpoint):
    """Replay a complete SELECT-only projection, never a yielded transaction."""
    deadline = time.monotonic() + READ_RETRY_SECONDS
    admitted = None
    while True:
        if time.monotonic() >= deadline:
            raise QuackReadUnavailable("quack task read deadline exhausted")
        connection = None
        try:
            connection = open_connection(deadline)
            binding = getattr(connection, "_quack_mutation_binding", None)
            if (
                type(binding) is not dict
                or set(binding) != _BINDING_FIELDS
                or binding.get("store_id") != store_id
                or binding.get("listen_uri") != endpoint
                or connection.in_transaction
                or getattr(connection, "_quack_pending_mutations", None) != []
            ):
                raise QuackReadUnavailable("quack task read binding unavailable")
            observed = tuple(
                (key, type(binding[key]), binding[key]) for key in sorted(binding)
            )
            if admitted is None:
                admitted = observed
            elif observed != admitted:
                raise QuackReadUnavailable("quack task read owner binding changed")
            with owned_connection_deadline(connection._connection, deadline):
                return read_projection(connection)
        except Exception as error:
            from .duckdb_interrupts import reraise_process_interrupt

            reraise_process_interrupt(error)
            if time.monotonic() >= deadline:
                raise QuackReadUnavailable(
                    "quack task read deadline exhausted"
                ) from None
            if not _replica_connection_lost(error):
                raise
        finally:
            if connection is not None:
                connection.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QuackReadUnavailable("quack task read deadline exhausted")
        time.sleep(min(0.02, remaining))
