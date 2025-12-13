"""
libp2p Compatibility Module

This module provides compatibility fixes for libp2p Python library
to work with newer versions of dependencies.

The main issue is that libp2p expects multihash.Func attribute which
doesn't exist in newer versions of the multihash library. This module
patches the multihash module to provide the missing Func class.
"""

import logging

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
        # First try to import pymultihash (the expected package for libp2p)
        try:
            import pymultihash
            logger.debug("✓ pymultihash package available (native libp2p support)")
            return True
        except ImportError:
            logger.debug("pymultihash not installed, trying compatibility layer")
        
        # Fall back to multiformats-based compatibility
        try:
            import multihash
        except ImportError:
            logger.error("Neither pymultihash nor multihash available")
            logger.error("Install with: pip install pymultihash>=0.8.2")
            return False
        
        # Check if Func already exists
        if hasattr(multihash, 'Func'):
            logger.debug("multihash.Func already exists, skipping patch")
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
        
        # Add digest function if it doesn't exist
        if not hasattr(multihash, 'digest'):
            class MultihashWrapper:
                """Wrapper for Multihash that adds encode() method for libp2p compatibility."""
                def __init__(self, mh_obj, mh_bytes):
                    self._mh_obj = mh_obj
                    self._mh_bytes = mh_bytes
                    # Forward all attributes from the original Multihash
                    self.code = mh_obj.code
                    self.name = mh_obj.name
                    self.length = mh_obj.length
                    self.digest = mh_obj.digest
                
                def encode(self):
                    """Return the encoded multihash bytes."""
                    return self._mh_bytes
                
                def __repr__(self):
                    return f"MultihashWrapper({self._mh_obj})"
            
            def digest(data, hash_func_name):
                """
                Create a multihash from data using the specified hash function.
                
                Args:
                    data: bytes to hash
                    hash_func_name: name of hash function (e.g., 'sha2-256') or hash code (int)
                    
                Returns:
                    MultihashWrapper object with digest and encode() method
                """
                # Handle hash function name or code
                if isinstance(hash_func_name, int):
                    # It's a hash code, need to find the name
                    hash_code = hash_func_name
                    # Find the name from HASH_CODES
                    hash_name = None
                    for name, code in multihash.constants.HASH_CODES.items():
                        if code == hash_code:
                            hash_name = name
                            break
                    if hash_name is None:
                        raise ValueError(f"Unknown hash code: {hash_code}")
                    hash_func_name = hash_name
                
                # Encode the data with the hash function
                mh_bytes = multihash.encode(data, hash_func_name)
                # Decode to get Multihash object
                mh_obj = multihash.decode(mh_bytes)
                # Return wrapper with encode() method
                return MultihashWrapper(mh_obj, mh_bytes)
            
            multihash.digest = digest
            logger.debug("  Added multihash.digest function with encode() support")
        
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
            logger.info("To enable P2P features, install: pip install libp2p>=0.4.0 pymultihash>=0.8.2")
            return False
        
        # Try to import libp2p to verify it works
        try:
            from libp2p import new_host
            logger.debug("✓ libp2p import successful")
            return True
        except ImportError as e:
            logger.warning(f"libp2p package not installed: {e}")
            logger.info("To enable P2P features, install: pip install libp2p>=0.4.0 pymultihash>=0.8.2")
            return False
        
    except Exception as e:
        logger.error(f"Error ensuring libp2p compatibility: {e}")
        logger.info("To enable P2P features, install: pip install libp2p>=0.4.0 pymultihash>=0.8.2")
        return False
