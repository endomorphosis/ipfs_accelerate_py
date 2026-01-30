import hashlib
from multiformats import CID, multihash
import tempfile
import os
import sys

# Try to import storage wrapper with comprehensive fallback
try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

class ipfs_multiformats_py:
    def __init__(self, resources, metadata): 
        self.multihash = multihash
        # Initialize storage wrapper
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        return None
    
    # Step 1: Hash the file content with SHA-256
    def get_file_sha256(self, file_path):
        hasher = hashlib.sha256()
        
        # Try distributed storage first
        if self._storage:
            try:
                content = self._storage.read_file(file_path)
                if content:
                    hasher.update(content if isinstance(content, bytes) else content.encode())
                    return hasher.digest()
            except Exception:
                pass
        
        # Fallback to local file
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.digest()

    # Step 2: Wrap the hash in Multihash format
    def get_multihash_sha256(self, file_content_hash):
        mh = self.multihash.wrap(file_content_hash, 'sha2-256')
        return mh

    # Step 3: Generate CID from Multihash (CIDv1)
    def get_cid(self, file_data):
        if os.path.isfile(file_data) == True:
            absolute_path = os.path.abspath(file_data)
            file_content_hash = self.get_file_sha256(file_data)
            mh = self.get_multihash_sha256(file_content_hash)
            cid = CID('base32', 'raw', mh)
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                filename = f.name
                with open(filename, 'w') as f_new:
                    f_new.write(file_data)
                file_content_hash = self.get_file_sha256(filename)
                mh = self.get_multihash_sha256(file_content_hash)
                cid = CID('base32', 1, 'raw', mh)
                os.remove(filename)
        return str(cid)