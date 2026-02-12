"""
libp2p Compatibility Module

This module provides compatibility fixes for libp2p Python library
to work with newer versions of dependencies.

The main issue is that libp2p expects multihash.Func attribute which
doesn't exist in newer versions of the multihash library. This module
patches the multihash module to provide the missing Func class.
"""

import logging
import sys
import importlib

logger = logging.getLogger(__name__)


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

            class _BytesWithEncode(bytes):
                def encode(self) -> bytes:  # type: ignore[override]
                    return bytes(self)

            def digest(data, hash_func_name):
                out = orig_digest(data, hash_func_name)
                if hasattr(out, "encode") and callable(getattr(out, "encode")):
                    return out
                if isinstance(out, (bytes, bytearray)):
                    return _BytesWithEncode(out)
                return out

            target_mod.digest = digest

        # Check if Func already exists
        if hasattr(multihash, 'Func'):
            logger.debug("multihash.Func already exists, skipping patch")
            _ensure_multiformats_func(getattr(multihash, 'Func'))
            _ensure_digest_encodeable(multihash)
            try:
                from multiformats import multihash as mf_multihash  # type: ignore

                _ensure_digest_encodeable(mf_multihash)
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
            logger.info("To enable P2P features, install: pip install 'libp2p @ git+https://github.com/libp2p/py-libp2p@main' pymultihash>=0.8.2")
            return False
        
        # Try to import libp2p to verify it works
        try:
            from libp2p import new_host
            logger.debug("✓ libp2p import successful")
            return True
        except ImportError as e:
            logger.warning(f"libp2p package not installed: {e}")
            logger.info("To enable P2P features, install: pip install 'libp2p @ git+https://github.com/libp2p/py-libp2p@main' pymultihash>=0.8.2")
            return False
        
    except Exception as e:
        logger.error(f"Error ensuring libp2p compatibility: {e}")
        logger.info("To enable P2P features, install: pip install 'libp2p @ git+https://github.com/libp2p/py-libp2p@main' pymultihash>=0.8.2")
        return False
