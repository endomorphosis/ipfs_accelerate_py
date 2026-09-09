"""Bound native extension images without sharing database connections.

DuckDB retains mappings for loaded extensions after a connection closes. A new
sealed memfd image per connection therefore grows resident memory indefinitely.
Keep a bounded set of exact images until process exit, and lend their verified
load paths to otherwise independent clients. Every borrow rechecks source
custody; every LOAD retains the existing immutable-image race guard.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .configured_board_extension_projection import (
    ConfiguredBoardExtensionProjectionError,
    ConfiguredBoardExtensionSetPin,
    ConfiguredBoardSealedExtensionSet,
    parse_configured_board_extension_set_pin,
    seal_configured_board_extension_set_home,
    verify_configured_board_extension_set_home,
)

MAX_PROCESS_EXTENSION_SETS = 4


class ConfiguredBoardExtensionImageLease:
    def __init__(self, cache: ConfiguredBoardExtensionImageCache,
                 image: ConfiguredBoardSealedExtensionSet) -> None:
        self._cache = cache
        self._image = image
        self._closed = False
        self._loading = 0

    def _require_open(self) -> None:
        self._cache._require_process()
        if self._closed or self._cache._closed:
            raise ConfiguredBoardExtensionProjectionError("extension image lease is closed")

    @property
    def pin(self) -> ConfiguredBoardExtensionSetPin:
        self._require_open()
        return self._image.pin

    @property
    def extension_directory(self) -> Path:
        self._require_open()
        return self._image.extension_directory

    @property
    def install_paths(self) -> dict[str, Path]:
        self._require_open()
        return self._image.install_paths

    @contextmanager
    def load_guard(self) -> Iterator[None]:
        self._require_open()
        with self._cache._lock:
            self._require_open()
            self._loading += 1
            try:
                with self._image.load_guard():
                    yield
            finally:
                self._loading -= 1

    def close(self) -> None:
        self._cache._require_process()
        with self._cache._lock:
            if self._closed:
                return
            if self._loading:
                raise ConfiguredBoardExtensionProjectionError("extension image LOAD is active")
            self._closed = True
            self._cache._borrowers -= 1


class ConfiguredBoardExtensionImageCache:
    def __init__(self, *, maximum: int = MAX_PROCESS_EXTENSION_SETS) -> None:
        if type(maximum) is not int or not 1 <= maximum <= MAX_PROCESS_EXTENSION_SETS:
            raise ValueError("extension image cache bound is invalid")
        self._pid = os.getpid()
        self._maximum = maximum
        self._lock = threading.RLock()
        self._images: dict[str, ConfiguredBoardSealedExtensionSet] = {}
        self._borrowers = 0
        self._closed = False

    def _require_process(self) -> None:
        # Check before taking the lock: a fork can inherit a peer-owned lock.
        # An exec gets its own cache and native loader. Never remove a parent's
        # private image paths from a forked child.
        if os.getpid() != self._pid:
            raise ConfiguredBoardExtensionProjectionError(
                "extension image cache requires its original process; exec after fork"
            )

    def borrow(self, pin: ConfiguredBoardExtensionSetPin,
               source_home: Path | str) -> ConfiguredBoardExtensionImageLease:
        self._require_process()
        parsed = parse_configured_board_extension_set_pin(pin.as_dict())
        with self._lock:
            if self._closed:
                raise ConfiguredBoardExtensionProjectionError("extension image cache is closed")
            image = self._images.get(parsed.set_id)
            if image is None:
                if len(self._images) >= self._maximum:
                    raise ConfiguredBoardExtensionProjectionError(
                        "process extension image bound exceeded; native process restart required"
                    )
                image = seal_configured_board_extension_set_home(parsed, source_home)
                self._images[parsed.set_id] = image
            else:
                verify_configured_board_extension_set_home(parsed.pins, source_home)
                image.verify()
            self._borrowers += 1
            return ConfiguredBoardExtensionImageLease(self, image)

    def close(self) -> None:
        """Release process images at teardown, after every client has closed."""
        self._require_process()
        with self._lock:
            if self._borrowers:
                raise ConfiguredBoardExtensionProjectionError("extension image clients remain open")
            self._close_images()

    def _close_images(self) -> None:
        self._closed = True
        for image in self._images.values():
            image.close()
        self._images.clear()

    def _process_exit(self) -> None:
        if os.getpid() == self._pid:
            # Process exit also releases abandoned leases; no later LOAD is
            # admitted. A forked child must leave the parent's paths intact.
            self._close_images()


_TRANSPORT_EXTENSION_IMAGES = ConfiguredBoardExtensionImageCache()
atexit.register(_TRANSPORT_EXTENSION_IMAGES._process_exit)


def borrow_configured_board_extension_set_home(
    pin: ConfiguredBoardExtensionSetPin, source_home: Path | str,
) -> ConfiguredBoardExtensionImageLease:
    return _TRANSPORT_EXTENSION_IMAGES.borrow(pin, source_home)
