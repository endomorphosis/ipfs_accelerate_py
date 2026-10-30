"""Bounded GIL-held startup diagnostics; no native timer or authority changes.

Sampling needs the GIL: native code holding it cannot be diagnosed by this
thread. External lifecycle health deadlines remain responsible for liveness.
"""
from __future__ import annotations

import json
import math
import sys
import threading


DEFAULT_INTERVAL_SECONDS = 10.0
MAX_THREADS = 8
MAX_FRAMES_PER_THREAD = 16
MAX_METADATA_BYTES = 16 * 1024


def startup_stack_metadata() -> dict:
    """Read code metadata only; never read locals, source text or linecache."""
    current = sys._current_frames()
    frame = None
    try:
        rows = []
        main_identifier = threading.main_thread().ident
        identifiers = ([main_identifier] if main_identifier in current else [])
        identifiers.extend(identifier for identifier in sorted(current)
                           if identifier != main_identifier)
        for identifier in identifiers[:MAX_THREADS]:
            frames = []
            frame = current[identifier]
            while frame is not None and len(frames) < MAX_FRAMES_PER_THREAD:
                code = frame.f_code
                frames.append(dict(filename=code.co_filename[:192],
                                   function=code.co_name[:96], line=frame.f_lineno))
                frame = frame.f_back
            rows.append(dict(thread_id=identifier, frames=frames,
                             frames_truncated=frame is not None))
        value = dict(schema='native-startup-stack-metadata/v1', threads=rows,
                     threads_truncated=len(current) > MAX_THREADS,
                     metadata_truncated=False)
        # Keep valid JSON within the byte ceiling, including escaped filenames.
        while len(json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii')) > MAX_METADATA_BYTES:
            value['metadata_truncated'] = True
            candidate = next((row for row in reversed(rows) if row['frames']), None)
            if candidate is None:
                raise RuntimeError('startup metadata bound cannot be met')
            candidate['frames'].pop()
            candidate['frames_truncated'] = True
        return value
    finally:
        # Do not retain another thread's frame or its object graph between ticks.
        frame = None
        current.clear()


class NativeOwnerStartupWatchdog:
    """Retain one real diagnostic thread and prove it stops on close.

    Logger I/O can be nonpreemptible. A failed join raises instead of reporting
    closure; the caller retains the actual object. This is no hard I/O deadline.
    """
    def __init__(self, *, logger, interval_seconds=DEFAULT_INTERVAL_SECONDS):
        if (type(interval_seconds) not in (int, float) or
                not math.isfinite(interval_seconds) or interval_seconds <= 0):
            raise ValueError('finite positive startup diagnostic interval required')
        self.interval_seconds = float(interval_seconds)
        self._logger = logger
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run,
            name='native-startup-metadata', daemon=True)
        self._start_attempted = False

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def start(self):
        if self._start_attempted or self._stop.is_set():
            raise RuntimeError('startup diagnostic thread is one-shot')
        self._start_attempted = True
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                value = startup_stack_metadata()
                encoded = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
                self._logger.info('Native owner startup stack metadata: %s', encoded)
            except BaseException as error:
                # Type only: exception messages can contain paths or data.
                self._logger.warning('Native owner startup metadata sampling failed: %s',
                                     type(error).__name__)

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2.0)
        elif self._start_attempted:
            raise RuntimeError('startup diagnostic start disposition unknown')
        if self._thread.is_alive():
            raise RuntimeError('startup diagnostic thread closure unproven')

    def __enter__(self):
        return self.start()

    def __exit__(self, *ignored):
        self.close()
