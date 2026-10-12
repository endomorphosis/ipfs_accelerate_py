"""One host-issued, process-bound exact-source lookup through an existing owner.

The host launches this child, registers its actual process birth and issues a
read-only grant before writing credentials to the inherited pipe. Public stdin
contains only the lookup request; credentials never belong in argv or the
environment. Existing coding containers do not yet receive this channel. This
CLI is not a registered coding-model tool, and required facts remain inline.
No owner, database, source fallback, model or provider is constructed here.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import select
import stat
import sys
import time

MAX_REQUEST_BYTES = 8192
MAX_CREDENTIAL_BYTES = 16384
CREDENTIAL_TIMEOUT_SECONDS = 10.0
CREDENTIAL_SCHEMA = "supervisor-source-lookup-cli-credential@1"
_CREDENTIAL_FIELDS = {"schema", "socket_path", "token", "client_id",
                      "process_birth_id", "store_id"}
_ERROR = b"source_lookup_cli_refused\n"


class SourceLookupCLIError(ValueError):
    """The closed child transport refused without exporting private details."""


def _need(condition):
    if not condition:
        raise SourceLookupCLIError("source_lookup_cli_refused")


def _object(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value)
            value[key] = item
        return value

    def nonfinite(_value):
        raise SourceLookupCLIError("source_lookup_cli_refused")

    value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                       parse_constant=nonfinite)
    _need(type(value) is dict)
    return value


def _credential(descriptor):
    """Read a bounded inherited FIFO and close it on every outcome."""
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISFIFO(info.st_mode) and info.st_uid == os.getuid()
              and os.getuid() == os.geteuid())
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        _need(flags & os.O_ACCMODE == os.O_RDONLY)
        deadline = time.monotonic() + CREDENTIAL_TIMEOUT_SECONDS
        raw = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            _need(remaining > 0)
            readable, _, _ = select.select([descriptor], [], [], remaining)
            _need(readable)
            data = os.read(descriptor, MAX_CREDENTIAL_BYTES + 1 - len(raw))
            if not data:
                break
            raw.extend(data)
            _need(len(raw) <= MAX_CREDENTIAL_BYTES)
        current = os.fstat(descriptor)
        _need((current.st_dev, current.st_ino, current.st_uid, current.st_mode)
              == (info.st_dev, info.st_ino, info.st_uid, info.st_mode))
        value = _object(bytes(raw))
        _need(set(value) == _CREDENTIAL_FIELDS and value["schema"] == CREDENTIAL_SCHEMA)
        for field in _CREDENTIAL_FIELDS - {"schema"}:
            item = value[field]
            _need(type(item) is str and item and item == item.strip()
                  and "\0" not in item and len(item.encode("utf-8")) <= 4096)
        _need(Path(value["socket_path"]).is_absolute() and len(value["token"]) >= 16)
        return value
    finally:
        os.close(descriptor)


def main(argv=None):
    """Handshake, perform one bounded source_get, close, then emit its full reply."""
    connection = None
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        _need(len(args) == 2 and args[0] == "--credential-fd"
              and type(args[1]) is str and 1 <= len(args[1]) <= 10
              and args[1].isascii() and args[1].isdecimal())
        descriptor = int(args[1])
        _need(descriptor >= 3)
        credential = _credential(descriptor)
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        _need(len(raw) <= MAX_REQUEST_BYTES)
        payload = _object(raw)
        from ..task_sources.typed_state_owner import TypedStateOwnerConnection
        from ..task_sources.control_plane_contracts import canonical_json_bytes

        connection = TypedStateOwnerConnection(
            socket_path=Path(credential["socket_path"]), token=credential["token"],
            client_id=credential["client_id"], process_birth_id=credential["process_birth_id"],
            store_id=credential["store_id"], timeout_seconds=5.0)
        response = connection.source_get(payload)
        connection.close()
        connection = None
        output = canonical_json_bytes(response) + b"\n"
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        sys.stderr.buffer.write(_ERROR)
        sys.stderr.buffer.flush()
        return 1
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
