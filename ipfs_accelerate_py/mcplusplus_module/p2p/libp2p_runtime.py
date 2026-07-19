"""MCP++ canonical libp2p runtime and compatibility boundary.

All production code that creates py-libp2p hosts should go through this module.
It owns dependency compatibility patching, host construction, multiaddr parsing,
peer-id normalization, and Trio service lifecycle helpers for MCP++ Profile E,
TaskQueue P2P, cache sharing, and datasets MCP P2P integration.

The main issue is that libp2p expects multihash.Func attribute which
doesn't exist in newer versions of the multihash library. This module
patches the multihash module to provide the missing Func class.
"""

from __future__ import annotations

import inspect
import logging
import subprocess
import sys
import importlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

PY_LIBP2P_MAIN_SPEC = "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main"
PY_LIBP2P_PROTOBUF_SPEC = "protobuf>=5.27.0"
PY_LIBP2P_EXTRA_PACKAGES = (
    PY_LIBP2P_PROTOBUF_SPEC,
    "pymultihash>=0.8.2",
    "dnspython>=2.2.1",
    PY_LIBP2P_MAIN_SPEC,
)
LIBP2P_COMPAT_ERROR = (
    "libp2p is installed but MCP++ dependency compatibility patches could not be applied. "
    "Install the MCP++ P2P extra from py-libp2p main: "
    f"pip install {PY_LIBP2P_PROTOBUF_SPEC!r} {PY_LIBP2P_MAIN_SPEC!r} "
    "'pymultihash>=0.8.2' 'dnspython>=2.2.1'"
)
LIBP2P_INSTALL_HINT = (
    "pip install "
    f"{PY_LIBP2P_PROTOBUF_SPEC!r} 'pymultihash>=0.8.2' 'dnspython>=2.2.1' "
    f"{PY_LIBP2P_MAIN_SPEC!r}"
)

_OPTIONAL_LIBP2P_ERROR_MARKERS = (
    "runtime version cannot be older than the linked gencode version",
    "detected incompatible protobuf gencode/runtime versions",
    "no module named",
    "cannot import name",
)


def _is_optional_libp2p_runtime_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _OPTIONAL_LIBP2P_ERROR_MARKERS)


def patch_libp2p_compatibility():
    """
    Patch libp2p compatibility issues with multihash module.

    The libp2p library expects pymultihash package to be installed.
    If not available, this tries to create a compatibility layer using
    the multiformats package.

    Returns:
        bool: True if compatibility is ensured, False otherwise
    """
    try:
        # Note: the `pymultihash` PyPI distribution provides the `multihash` import.
        # Some environments may also inject a different `multihash` module into
        # sys.modules (e.g., via multiformats), so we validate what we imported.

        def _is_valid_multihash(mod) -> bool:
            try:
                # Modern pymultihash (e.g. 0.8.x) exposes `Func`/`FuncReg` and
                # does not provide `multihash.constants.HASH_CODES`.
                if hasattr(mod, "Func") and hasattr(mod, "digest") and callable(getattr(mod, "digest")):
                    return True

                # Older/alternate multihash implementations expose HASH_CODES.
                return bool(
                    hasattr(mod, "constants")
                    and hasattr(mod.constants, "HASH_CODES")
                    and isinstance(getattr(mod.constants, "HASH_CODES"), dict)
                )
            except Exception:
                return False

        try:
            import multihash  # type: ignore
        except ImportError:
            logger.error("multihash module not available")
            logger.error("Install with: pip install pymultihash>=0.8.2")
            return False

        if not _is_valid_multihash(multihash):
            # Best-effort: if something shadowed `multihash`, drop it and retry.
            try:
                sys.modules.pop("multihash.constants", None)
                sys.modules.pop("multihash", None)
                importlib.invalidate_caches()
                import multihash  # type: ignore
            except Exception:
                pass

        if not _is_valid_multihash(multihash):
            logger.warning("multihash.constants.HASH_CODES not found, cannot patch")
            return False

        def _ensure_multiformats_func(func_cls) -> None:
            try:
                from multiformats import multihash as mf_multihash  # type: ignore

                if not hasattr(mf_multihash, "Func"):
                    mf_multihash.Func = func_cls
            except Exception:
                pass

        def _ensure_digest_encodeable(target_mod) -> None:
            """Ensure target_mod.digest returns an encode()-capable value.

            Some environments (notably when `multihash` is provided by
            `multiformats.multihash`) return raw bytes. libp2p expects the
            digest return value to provide `.encode()`.

            We wrap raw bytes in a bytes subclass with an encode() method,
            preserving bytes-compatibility for other call sites.
            """

            orig_digest = getattr(target_mod, "digest", None)
            if not callable(orig_digest):
                return

            # Idempotency: patching multiple times can stack wrappers and hit
            # Python's recursion limit (KadDHT calls digest heavily).
            if getattr(orig_digest, "_ipfs_accelerate_patched_digest_encodeable", False):
                return

            # If we already captured the true original digest for this module,
            # wrap that instead of wrapping a wrapper.
            captured_orig = getattr(target_mod, "_ipfs_accelerate_orig_digest", None)
            if callable(captured_orig):
                orig_digest = captured_orig
            else:
                try:
                    setattr(target_mod, "_ipfs_accelerate_orig_digest", orig_digest)
                except Exception:
                    pass

            class _BytesWithEncode(bytes):
                def encode(self) -> bytes:  # type: ignore[override]
                    return bytes(self)

                @property
                def digest(self) -> bytes:  # type: ignore[override]
                    """Return the raw digest bytes for multihash-encoded values.

                    Some libp2p components (e.g. kad-dht distance metrics) expect
                    `multihash.digest(...).digest` to be the *raw* hash digest.
                    When using `multiformats.multihash`, `digest()` can return a
                    multihash-encoded bytestring: `<code><len><digest...>`.

                    This property best-effort parses the multihash prefix and
                    returns the trailing digest bytes. If parsing fails, it
                    falls back to returning the underlying bytes.
                    """

                    data = bytes(self)

                    def _read_uvarint(buf: bytes, start: int) -> tuple[int, int] | None:
                        value = 0
                        shift = 0
                        idx = start
                        while idx < len(buf):
                            b = buf[idx]
                            value |= (b & 0x7F) << shift
                            idx += 1
                            if (b & 0x80) == 0:
                                return value, idx
                            shift += 7
                            if shift > 63:
                                return None
                        return None

                    try:
                        # multihash format: varint(code) + varint(length) + digest bytes
                        code_res = _read_uvarint(data, 0)
                        if code_res is None:
                            return data
                        _, idx = code_res
                        len_res = _read_uvarint(data, idx)
                        if len_res is None:
                            return data
                        digest_len, idx2 = len_res
                        if digest_len < 0:
                            return data
                        end = idx2 + digest_len
                        if end > len(data):
                            return data
                        return data[idx2:end]
                    except Exception:
                        return data

            def digest(data, hash_func_name):
                # Some libp2p builds pass an Enum (e.g. multihash.Func.sha2_256)
                # where multiformats expects an int code or a string name.
                try:
                    import enum

                    if isinstance(hash_func_name, enum.Enum):
                        hash_func_name = hash_func_name.value
                except Exception:
                    pass

                out = orig_digest(data, hash_func_name)
                if hasattr(out, "encode") and callable(getattr(out, "encode")):
                    return out
                if isinstance(out, (bytes, bytearray)):
                    return _BytesWithEncode(out)
                return out

            setattr(digest, "_ipfs_accelerate_patched_digest_encodeable", True)
            target_mod.digest = digest

        def _ensure_multihash_encode(target_mod) -> None:
            """Provide multihash.encode(raw_digest, func) for py-libp2p builds.

            Recent py-libp2p code paths call ``multihash.encode`` with a raw
            digest and a hash code/name. ``pymultihash`` exposes ``digest`` and
            ``decode`` but not this wrapper API, so host creation can fail before
            any network work starts.
            """

            encode = getattr(target_mod, "encode", None)
            if callable(encode):
                return

            def _coerce_hash_func(hash_func_name):
                try:
                    import enum

                    if isinstance(hash_func_name, enum.Enum):
                        return hash_func_name.value
                except Exception:
                    pass
                return hash_func_name

            def encode(raw_digest, hash_func_name):  # type: ignore[no-redef]
                hash_func = _coerce_hash_func(hash_func_name)
                try:
                    from multiformats import multihash as mf_multihash  # type: ignore

                    return mf_multihash.wrap(bytes(raw_digest or b""), hash_func)
                except Exception:
                    code = int(hash_func)
                    data = bytes(raw_digest or b"")

                    def _uvarint(value: int) -> bytes:
                        out = bytearray()
                        value = int(value)
                        while True:
                            to_write = value & 0x7F
                            value >>= 7
                            if value:
                                out.append(to_write | 0x80)
                            else:
                                out.append(to_write)
                                break
                        return bytes(out)

                    return _uvarint(code) + _uvarint(len(data)) + data

            setattr(encode, "_ipfs_accelerate_patched_encode", True)
            target_mod.encode = encode

        # Check if Func already exists
        if hasattr(multihash, 'Func'):
            logger.debug("multihash.Func already exists, skipping patch")
            _ensure_multiformats_func(getattr(multihash, 'Func'))
            _ensure_digest_encodeable(multihash)
            _ensure_multihash_encode(multihash)
            try:
                from multiformats import multihash as mf_multihash  # type: ignore

                _ensure_digest_encodeable(mf_multihash)
            except Exception:
                pass

            # Patch known libp2p KadDHT ProviderStore bug where `_send_add_provider`
            # unconditionally closes `stream` in a `finally` block, causing:
            #   UnboundLocalError: cannot access local variable 'stream'...
            # when opening the stream fails.
            try:
                from libp2p.kad_dht.provider_store import ProviderStore  # type: ignore

                orig_send = getattr(ProviderStore, "_send_add_provider", None)
                if callable(orig_send) and not getattr(orig_send, "_ipfs_accelerate_patched", False):

                    async def _send_add_provider_safe(self, peer_id, key):  # type: ignore[no-redef]
                        try:
                            return await orig_send(self, peer_id, key)
                        except UnboundLocalError as exc:
                            # Treat as a failed send; the root cause is already
                            # logged (unable to connect / open stream).
                            if "stream" in str(exc):
                                return False
                            raise

                    setattr(_send_add_provider_safe, "_ipfs_accelerate_patched", True)
                    ProviderStore._send_add_provider = _send_add_provider_safe  # type: ignore[assignment]
                    logger.debug("✓ Patched libp2p ProviderStore._send_add_provider stream-close bug")
            except Exception:
                pass
            return True

        # Get hash codes from multihash.constants
        if not hasattr(multihash, 'constants') or not hasattr(multihash.constants, 'HASH_CODES'):
            logger.warning("multihash.constants.HASH_CODES not found, cannot patch")
            return False

        # Create Func class with hash algorithm codes
        class Func:
            """Hash function identifiers for multihash."""
            # Most commonly used hash functions
            identity = multihash.constants.HASH_CODES.get('id', 0)
            sha1 = multihash.constants.HASH_CODES.get('sha1', 17)
            sha2_256 = multihash.constants.HASH_CODES.get('sha2-256', 18)
            sha2_512 = multihash.constants.HASH_CODES.get('sha2-512', 19)
            sha3_512 = multihash.constants.HASH_CODES.get('sha3-512', 20)
            sha3_384 = multihash.constants.HASH_CODES.get('sha3-384', 21)
            sha3_256 = multihash.constants.HASH_CODES.get('sha3-256', 22)
            sha3_224 = multihash.constants.HASH_CODES.get('sha3-224', 23)
            shake_128 = multihash.constants.HASH_CODES.get('shake-128', 24)
            shake_256 = multihash.constants.HASH_CODES.get('shake-256', 25)
            keccak_224 = multihash.constants.HASH_CODES.get('keccak-224', 26)
            keccak_256 = multihash.constants.HASH_CODES.get('keccak-256', 27)
            keccak_384 = multihash.constants.HASH_CODES.get('keccak-384', 28)
            keccak_512 = multihash.constants.HASH_CODES.get('keccak-512', 29)
            blake2b_512 = multihash.constants.HASH_CODES.get('blake2b-512', 45632)
            blake2s_256 = multihash.constants.HASH_CODES.get('blake2s-256', 45664)

        # Patch the multihash module
        multihash.Func = Func
        _ensure_multiformats_func(Func)

        _ensure_digest_encodeable(multihash)
        _ensure_multihash_encode(multihash)
        try:
            from multiformats import multihash as mf_multihash  # type: ignore

            _ensure_digest_encodeable(mf_multihash)
        except Exception:
            pass

        logger.debug("  Ensured multihash.digest returns encode()-capable values")

        logger.info("✓ Successfully patched multihash.Func for libp2p compatibility")
        logger.debug(f"  multihash.Func.sha2_256 = {multihash.Func.sha2_256}")

        return True

    except Exception as e:
        logger.error(f"Failed to patch multihash for libp2p compatibility: {e}")
        return False


def ensure_libp2p_compatible():
    """
    Ensure libp2p is compatible with current dependencies.

    This function should be called before importing or using libp2p
    to apply any necessary compatibility patches.

    Returns:
        bool: True if libp2p is ready to use, False otherwise
    """
    try:
        # Apply compatibility patches
        if not patch_libp2p_compatibility():
            logger.warning("Could not apply libp2p compatibility patches")
            logger.info("To enable P2P features, install via MCP++ runtime: %s", LIBP2P_INSTALL_HINT)
            return False

        # Try to import libp2p to verify it works
        try:
            from libp2p import new_host
            logger.debug("✓ libp2p import successful")
            return True
        except ImportError as e:
            logger.warning(f"libp2p package not installed: {e}")
            logger.info("To enable P2P features, install via MCP++ runtime: %s", LIBP2P_INSTALL_HINT)
            return False

    except Exception as e:
        if _is_optional_libp2p_runtime_error(e):
            logger.warning(f"libp2p runtime unavailable, disabling P2P features: {e}")
        else:
            logger.error(f"Error ensuring libp2p compatibility: {e}")
        logger.info("To enable P2P features, install via MCP++ runtime: %s", LIBP2P_INSTALL_HINT)
        return False


def ensure_libp2p_runtime() -> bool:
    """Return True when py-libp2p can be imported after MCP++ patches."""

    try:
        if not ensure_libp2p_compatible():
            return False
        from libp2p import new_host  # noqa: F401

        return True
    except Exception:
        return False


def have_libp2p_runtime() -> bool:
    """Cheap availability check used by optional P2P features."""

    return ensure_libp2p_runtime()


def require_libp2p_runtime() -> None:
    """Raise a deterministic error when py-libp2p is unavailable/incompatible."""

    if not ensure_libp2p_runtime():
        raise RuntimeError(LIBP2P_COMPAT_ERROR)


def install_libp2p_runtime(
    *,
    quiet: bool = True,
    timeout: float = 120,
    upgrade: bool = True,
    force_reinstall: bool = False,
) -> bool:
    """Install the audited MCP++ libp2p runtime package set.

    Transport modules delegate here instead of carrying their own subprocess
    recipes. The package list is intentionally fixed to py-libp2p GitHub main
    plus known runtime companions; callers that need custom dependency
    resolution should use the explicit installer subsystem, not transport
    startup.
    """

    if ensure_libp2p_runtime():
        return True

    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("--quiet")
    if upgrade:
        cmd.append("--upgrade")
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.extend(PY_LIBP2P_EXTRA_PACKAGES)

    try:
        subprocess.check_call(
            cmd,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.PIPE if quiet else None,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.error("Failed to install MCP++ libp2p runtime: %s", exc)
        return False

    if not ensure_libp2p_runtime():
        logger.error("MCP++ libp2p runtime compatibility check failed after install")
        return False
    return True


async def install_libp2p_runtime_async(
    *,
    quiet: bool = True,
    timeout: float = 120,
    upgrade: bool = True,
    force_reinstall: bool = False,
) -> bool:
    """Trio-friendly wrapper around :func:`install_libp2p_runtime`."""

    import trio

    return await trio.to_thread.run_sync(
        lambda: install_libp2p_runtime(
            quiet=quiet,
            timeout=timeout,
            upgrade=upgrade,
            force_reinstall=force_reinstall,
        )
    )


async def new_libp2p_host(**kwargs: Any) -> Any:
    """Create a py-libp2p host through the MCP++ compatibility boundary."""

    require_libp2p_runtime()
    from libp2p import new_host

    host_obj = new_host(**kwargs)
    return await host_obj if inspect.isawaitable(host_obj) else host_obj


def create_libp2p_key_pair() -> Any:
    """Create a py-libp2p secp256k1 key pair through the MCP++ boundary."""

    require_libp2p_runtime()
    from libp2p.crypto.secp256k1 import create_new_key_pair

    return create_new_key_pair()


def peer_id_from_base58(value: str) -> Any:
    """Parse a base58 peer id through the MCP++ runtime boundary."""

    require_libp2p_runtime()
    from libp2p.peer.id import ID as PeerID

    return PeerID.from_base58(str(value))


def get_libp2p_protocol_type(default: Any = str) -> Any:
    """Return py-libp2p's protocol type constructor, or a safe fallback."""

    require_libp2p_runtime()
    try:
        from libp2p.custom_types import TProtocol  # type: ignore

        return TProtocol
    except Exception:
        try:
            from libp2p.typing import TProtocol  # type: ignore

            return TProtocol
        except Exception:
            return default


def get_libp2p_stream_interface(default: Any = Any) -> Any:
    """Return py-libp2p's stream interface type, or a safe fallback."""

    require_libp2p_runtime()
    for module_name in (
        "libp2p.network.stream.net_stream",
        "libp2p.network.stream.net_stream_interface",
    ):
        try:
            mod = __import__(module_name, fromlist=["INetStream"])
            return getattr(mod, "INetStream")
        except Exception:
            continue
    return default


def get_stream_eof_exceptions() -> tuple[type[BaseException], ...]:
    """Return stream EOF exception classes available in this py-libp2p build."""

    require_libp2p_runtime()
    exceptions: list[type[BaseException]] = []
    for module_name, symbol in (
        ("libp2p.network.stream.exceptions", "StreamEOF"),
        ("libp2p.stream_muxer.exceptions", "MuxedStreamEOF"),
    ):
        try:
            mod = __import__(module_name, fromlist=[symbol])
            exc = getattr(mod, symbol)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                exceptions.append(exc)
        except Exception:
            continue
    return tuple(exceptions) or (EOFError,)


def make_libp2p_resource_manager(
    *,
    max_connections: int = 100_000,
    max_streams: int = 100_000,
    max_memory_mb: int = 1024,
    enable_metrics: bool = False,
    enable_connection_pooling: bool = False,
    enable_memory_pooling: bool = False,
    enable_circuit_breaker: bool = False,
    enable_graceful_degradation: bool = False,
) -> Any | None:
    """Construct a py-libp2p ResourceManager when this build exposes it."""

    require_libp2p_runtime()
    try:
        from libp2p.rcmgr import ResourceLimits, new_resource_manager

        return new_resource_manager(
            limits=ResourceLimits(
                max_connections=max(64, int(max_connections)),
                max_memory_mb=max(128, int(max_memory_mb)),
                max_streams=max(1024, int(max_streams)),
            ),
            enable_metrics=bool(enable_metrics),
            enable_connection_pooling=bool(enable_connection_pooling),
            enable_memory_pooling=bool(enable_memory_pooling),
            enable_circuit_breaker=bool(enable_circuit_breaker),
            enable_graceful_degradation=bool(enable_graceful_degradation),
        )
    except Exception:
        return None


def make_kad_dht(host: Any, *, mode: str | None = None) -> Any | None:
    """Construct KadDHT across py-libp2p API variants."""

    require_libp2p_runtime()
    for module_name in ("libp2p.kad_dht.kad_dht", "libp2p.kad_dht"):
        try:
            mod = __import__(module_name, fromlist=["KadDHT"])
            cls = getattr(mod, "KadDHT")
            dht_mode = None
            if mode:
                try:
                    dht_mode_mod = __import__("libp2p.kad_dht.kad_dht", fromlist=["DHTMode"])
                    dht_mode_cls = getattr(dht_mode_mod, "DHTMode")
                    dht_mode = getattr(dht_mode_cls, str(mode).upper())
                except Exception:
                    dht_mode = None
            return cls(host, dht_mode) if dht_mode is not None else cls(host)
        except Exception:
            continue
    return None


def make_mdns_discovery(network: Any, *, port: int) -> Any:
    """Construct py-libp2p MDNSDiscovery through the MCP++ boundary."""

    require_libp2p_runtime()
    from libp2p.discovery.mdns.mdns import MDNSDiscovery

    return MDNSDiscovery(network, port=int(port))


def make_circuit_relay_v2(host: Any, *, allow_hop: bool = False) -> Any:
    """Construct Circuit Relay v2 through the MCP++ boundary."""

    require_libp2p_runtime()
    from libp2p.relay.circuit_v2.protocol import CircuitV2Protocol

    return CircuitV2Protocol(host, allow_hop=bool(allow_hop))


def make_dcutr_protocol(host: Any) -> Any:
    """Construct DCUtR through the MCP++ boundary."""

    require_libp2p_runtime()
    from libp2p.relay.circuit_v2.dcutr import DCUtRProtocol

    return DCUtRProtocol(host)


def _make_first_available(module_symbols: tuple[tuple[str, str], ...], *args: Any, **kwargs: Any) -> Any | None:
    require_libp2p_runtime()
    for module_name, symbol in module_symbols:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            return cls(*args, **kwargs)
        except Exception:
            continue
    return None


def make_rendezvous_service(host: Any) -> Any | None:
    """Construct a py-libp2p RendezvousService across API variants."""

    return _make_first_available(
        (
            ("libp2p.discovery.rendezvous.rendezvous", "RendezvousService"),
            ("libp2p.discovery.rendezvous", "RendezvousService"),
            ("libp2p.rendezvous", "RendezvousService"),
        ),
        host,
    )


def make_rendezvous_client(host: Any) -> Any | None:
    """Construct a py-libp2p RendezvousClient across API variants."""

    return _make_first_available(
        (
            ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
            ("libp2p.discovery.rendezvous", "RendezvousClient"),
            ("libp2p.rendezvous", "RendezvousClient"),
        ),
        host,
    )


def get_background_trio_service():
    """Return py-libp2p's Trio service context manager constructor."""

    require_libp2p_runtime()
    try:
        from libp2p.tools.anyio_service.context import background_trio_service

        return background_trio_service
    except Exception:
        pass
    try:
        from libp2p.tools.async_service import background_trio_service

        return background_trio_service
    except Exception:
        from libp2p.tools.async_service.trio_service import background_trio_service

        return background_trio_service


def make_multiaddr(value: str) -> Any:
    """Construct a Multiaddr through the MCP++ runtime boundary."""

    require_libp2p_runtime()
    from multiaddr import Multiaddr

    return Multiaddr(str(value))


def peerinfo_from_multiaddr(value: str) -> Any:
    """Parse a peer multiaddr into py-libp2p PeerInfo."""

    require_libp2p_runtime()
    from libp2p.peer.peerinfo import info_from_p2p_addr

    return info_from_p2p_addr(make_multiaddr(str(value)))


def peer_id_text(peer_id: Any) -> str:
    """Return a stable string representation for py-libp2p peer ids."""

    try:
        pretty = getattr(peer_id, "pretty", None)
        if callable(pretty):
            return str(pretty() or "").strip()
    except Exception:
        pass
    return str(peer_id or "").strip()


@asynccontextmanager
async def running_libp2p_host(
    *,
    listen_multiaddr: str = "/ip4/127.0.0.1/tcp/0",
    **host_kwargs: Any,
) -> AsyncIterator[Any]:
    """Create, start, listen, and close a py-libp2p host.

    This is the common context manager for MCP++ p2p tests, small scripts, and
    client-side one-shot RPC hosts.
    """

    host = await new_libp2p_host(**host_kwargs)
    background_trio_service = get_background_trio_service()
    try:
        async with background_trio_service(host.get_network()):
            await host.get_network().listen(make_multiaddr(listen_multiaddr))
            yield host
    finally:
        try:
            await host.close()
        except Exception:
            pass


__all__ = [
    "LIBP2P_COMPAT_ERROR",
    "LIBP2P_INSTALL_HINT",
    "PY_LIBP2P_EXTRA_PACKAGES",
    "PY_LIBP2P_MAIN_SPEC",
    "PY_LIBP2P_PROTOBUF_SPEC",
    "create_libp2p_key_pair",
    "ensure_libp2p_compatible",
    "ensure_libp2p_runtime",
    "get_background_trio_service",
    "get_libp2p_protocol_type",
    "get_libp2p_stream_interface",
    "get_stream_eof_exceptions",
    "have_libp2p_runtime",
    "install_libp2p_runtime",
    "install_libp2p_runtime_async",
    "make_circuit_relay_v2",
    "make_dcutr_protocol",
    "make_kad_dht",
    "make_libp2p_resource_manager",
    "make_mdns_discovery",
    "make_multiaddr",
    "make_rendezvous_client",
    "make_rendezvous_service",
    "new_libp2p_host",
    "patch_libp2p_compatibility",
    "peer_id_from_base58",
    "peer_id_text",
    "peerinfo_from_multiaddr",
    "require_libp2p_runtime",
    "running_libp2p_host",
]
