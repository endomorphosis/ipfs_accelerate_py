"""Cooperative process shutdown at explicit database admission boundaries.

Python signal handlers may run while a native database call is active.  They
must not raise into that call or start cleanup recursively.  A completed,
already-admitted callback still records its outcome before the next boundary.
This latch never settles retained work or grants a new claim.
"""

import signal


class DatabaseDaemonShutdown:
    def __init__(self) -> None:
        self.signum: int | None = None

    def request(self, signum: int, _frame: object = None) -> None:
        # The handler does only a bounded memory update, including if a second
        # signal arrives while normal connection cleanup is in progress.
        if self.signum is None and signum in (signal.SIGTERM, signal.SIGINT):
            self.signum = int(signum)

    def checkpoint(self) -> None:
        # Called from ordinary Python control flow after native calls return.
        if self.signum is not None:
            raise SystemExit(128 + self.signum)
