"""Small CONNECT-only proxy for an EAAEF worker's internal Docker network."""

from __future__ import annotations

import argparse
import ipaddress
import json
import selectors
import socket
import socketserver
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ipfs_accelerate_py.agent_implementation_route import (
    resolve_agent_implementation_route_binding,
)
from ipfs_accelerate_py.agent_supervisor.runtime.worker_network import (
    load_worker_network_authorization,
)

MAX_CONNECT_HEADER_BYTES = 16 * 1024
MAX_ROUTE_BINDING_BYTES = 64 * 1024
CONNECT_HEADER_TIMEOUT_SECONDS = 10.0
RELAY_IDLE_TIMEOUT_SECONDS = 30.0
RELAY_MAX_LIFETIME_SECONDS = 15 * 60.0
MAX_CONCURRENT_CONNECTIONS = 32
_RESPONSE_FORBIDDEN = b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n"
_RESPONSE_BAD_REQUEST = b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
_RESPONSE_METHOD = b"HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n"
_RESPONSE_UPSTREAM = b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"
_RESPONSE_CONNECTED = b"HTTP/1.1 200 Connection Established\r\n\r\n"
_RESPONSE_BUSY = b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n"


def parse_connect_authority(
    authority: str,
    *,
    allowed_hostnames: Iterable[str],
) -> tuple[str, int]:
    """Return an exact approved hostname and the fixed TLS port."""

    if authority.count(":") != 1 or "@" in authority:
        raise ValueError("CONNECT authority is malformed")
    hostname, port_text = authority.rsplit(":", 1)
    if hostname != hostname.lower() or hostname.endswith(".") or not hostname:
        raise ValueError("CONNECT hostname is not canonical")
    try:
        ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        pass
    else:
        raise ValueError("CONNECT IP literals are forbidden")
    if port_text != "443":
        raise ValueError("CONNECT destination port is forbidden")
    allowed = tuple(allowed_hostnames)
    if hostname not in allowed:
        raise ValueError("CONNECT hostname is not approved")
    return hostname, 443


def _connect_global_upstream(hostname: str, port: int) -> socket.socket:
    """Resolve at the proxy and connect only to public unicast addresses."""

    candidates = socket.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    last_error: OSError | None = None
    for family, socktype, proto, _canonical, sockaddr in candidates:
        address = ipaddress.ip_address(sockaddr[0])
        if (
            not address.is_global
            or address.is_multicast
            or address.is_reserved
            or address.is_loopback
            or address.is_link_local
            or address.is_private
            or address.is_unspecified
        ):
            continue
        upstream = socket.socket(family, socktype, proto)
        upstream.settimeout(15.0)
        try:
            upstream.connect(sockaddr)
        except OSError as exc:
            last_error = exc
            upstream.close()
            continue
        upstream.settimeout(None)
        return upstream
    if last_error is not None:
        raise last_error
    raise OSError("approved hostname did not resolve to a public address")


def _read_connect_header(client: socket.socket) -> bytes:
    previous_timeout = client.gettimeout()
    deadline = time.monotonic() + CONNECT_HEADER_TIMEOUT_SECONDS
    header = bytearray()
    try:
        while b"\r\n\r\n" not in header:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CONNECT header deadline expired")
            client.settimeout(remaining)
            chunk = client.recv(min(4096, MAX_CONNECT_HEADER_BYTES + 1 - len(header)))
            if not chunk:
                raise ValueError("CONNECT request ended before its headers")
            header.extend(chunk)
            if len(header) > MAX_CONNECT_HEADER_BYTES:
                raise ValueError("CONNECT headers exceed the bounded limit")
    finally:
        client.settimeout(previous_timeout)
    boundary = header.find(b"\r\n\r\n") + 4
    if boundary != len(header):
        raise ValueError("CONNECT request cannot pipeline tunnel bytes")
    return bytes(header)


def _relay(left: socket.socket, right: socket.socket) -> None:
    started = time.monotonic()
    last_activity = started
    selector = selectors.DefaultSelector()
    left_timeout = left.gettimeout()
    right_timeout = right.gettimeout()
    left.settimeout(RELAY_IDLE_TIMEOUT_SECONDS)
    right.settimeout(RELAY_IDLE_TIMEOUT_SECONDS)
    selector.register(left, selectors.EVENT_READ, right)
    selector.register(right, selectors.EVENT_READ, left)
    try:
        while selector.get_map():
            now = time.monotonic()
            if now - started >= RELAY_MAX_LIFETIME_SECONDS:
                return
            events = selector.select(
                timeout=min(
                    RELAY_IDLE_TIMEOUT_SECONDS,
                    RELAY_MAX_LIFETIME_SECONDS - (now - started),
                )
            )
            if not events:
                return
            for key, _events in events:
                source = key.fileobj
                destination = key.data
                data = source.recv(64 * 1024)
                if not data:
                    selector.unregister(source)
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    continue
                destination.sendall(data)
                last_activity = time.monotonic()
            if time.monotonic() - last_activity >= RELAY_IDLE_TIMEOUT_SECONDS:
                return
    finally:
        selector.close()
        left.settimeout(left_timeout)
        right.settimeout(right_timeout)


def serve_connect_client(
    client: socket.socket,
    *,
    allowed_hostnames: Iterable[str],
    connector: Callable[[str, int], socket.socket] = _connect_global_upstream,
) -> None:
    """Serve one bounded CONNECT request and close both sides."""

    upstream: socket.socket | None = None
    try:
        try:
            header = _read_connect_header(client)
            first_line = header.split(b"\r\n", 1)[0].decode("ascii", errors="strict")
            fields = first_line.split(" ")
        except (OSError, UnicodeError, ValueError):
            client.sendall(_RESPONSE_BAD_REQUEST)
            return
        if len(fields) != 3 or fields[0] != "CONNECT" or fields[2] != "HTTP/1.1":
            client.sendall(_RESPONSE_METHOD)
            return
        try:
            hostname, port = parse_connect_authority(
                fields[1],
                allowed_hostnames=allowed_hostnames,
            )
        except ValueError:
            client.sendall(_RESPONSE_FORBIDDEN)
            return
        try:
            upstream = connector(hostname, port)
        except OSError:
            client.sendall(_RESPONSE_UPSTREAM)
            return
        client.sendall(_RESPONSE_CONNECTED)
        _relay(client, upstream)
    finally:
        if upstream is not None:
            upstream.close()
        client.close()


class _ConnectServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True
    request_queue_size = MAX_CONCURRENT_CONNECTIONS

    def __init__(
        self,
        server_address: tuple[str, int],
        allowed_hostnames: tuple[str, ...],
    ) -> None:
        self.allowed_hostnames = allowed_hostnames
        self._connection_slots = threading.BoundedSemaphore(
            MAX_CONCURRENT_CONNECTIONS
        )
        super().__init__(server_address, _ConnectHandler)

    def process_request(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
    ) -> None:
        if not self._connection_slots.acquire(blocking=False):
            try:
                request.settimeout(1.0)
                request.sendall(_RESPONSE_BUSY)
            except OSError:
                pass
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class _ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        serve_connect_client(
            self.request,
            allowed_hostnames=self.server.allowed_hostnames,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EAAEF CONNECT allowlist proxy")
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--agent-implementation-route-json")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--authorization-artifact-cid")
    parser.add_argument("--provider", choices=("grok", "codex"))
    parser.add_argument("--expected-worker-principal-did")
    parser.add_argument("--expected-provider-principal-did")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--diagnostic-allow-host", action="append")
    args = parser.parse_args(argv)
    if args.diagnostic_only:
        if (
            args.agent_implementation_route_json
            or args.workspace is not None
            or args.authorization_artifact_cid
            or args.provider
            or args.expected_worker_principal_did
            or args.expected_provider_principal_did
            or not args.diagnostic_allow_host
        ):
            parser.error("diagnostic proxy arguments are invalid")
        try:
            if not ipaddress.ip_address(args.listen_host).is_loopback:
                parser.error("diagnostic proxy must listen only on loopback")
        except ValueError:
            parser.error("diagnostic proxy listener must be a loopback literal")
        allowed = tuple(args.diagnostic_allow_host)
    else:
        route_raw = str(args.agent_implementation_route_json or "")
        if (
            args.workspace is None
            or not args.authorization_artifact_cid
            or not args.provider
            or not args.expected_worker_principal_did
            or not args.expected_provider_principal_did
            or not route_raw
            or len(route_raw.encode("utf-8")) > MAX_ROUTE_BINDING_BYTES
            or args.diagnostic_allow_host
        ):
            parser.error("production proxy requires exact signed authority")
        try:
            def unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
                decoded: dict[str, object] = {}
                for key, value in pairs:
                    if key in decoded:
                        raise ValueError("duplicate route binding key")
                    decoded[key] = value
                return decoded

            route_value = json.loads(route_raw, object_pairs_hook=unique_object)
            route = resolve_agent_implementation_route_binding(
                route_value,
                repo_root=args.workspace,
                now_ms=int(time.time() * 1000),
                max_age_ms=15 * 60 * 1000,
            )
            invocation = route.invocation_binding
            if invocation is None:
                raise ValueError("signed invocation is absent")
            authorization = load_worker_network_authorization(
                invocation_binding=invocation,
                provider=args.provider,
                workspace=args.workspace,
                expected_artifact_cid=args.authorization_artifact_cid,
                expected_worker_principal_did=(
                    args.expected_worker_principal_did
                ),
                expected_provider_principal_did=(
                    args.expected_provider_principal_did
                ),
            )
            endpoint = authorization.proxy_endpoint.removeprefix("http://")
            expected_host, expected_port = endpoint.rsplit(":", 1)
            if args.listen_host != expected_host or args.listen_port != int(expected_port):
                raise ValueError("proxy listener does not match signed authority")
            allowed = authorization.allowed_hostnames
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"production proxy authority is invalid: {exc}")
    for hostname in allowed:
        parse_connect_authority(
            f"{hostname}:443",
            allowed_hostnames=allowed,
        )
    with _ConnectServer((args.listen_host, args.listen_port), allowed) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
