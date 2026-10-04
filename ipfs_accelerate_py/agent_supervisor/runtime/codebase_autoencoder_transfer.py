"""Compatibility alias; shared implementation belongs to ipfs_datasets_py."""
import sys
from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder_transfer as _implementation

sys.modules[__name__] = _implementation
