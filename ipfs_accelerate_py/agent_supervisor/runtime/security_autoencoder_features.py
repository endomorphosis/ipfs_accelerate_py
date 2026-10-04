"""Compatibility alias; implementation is owned by ipfs_datasets_py."""
import sys
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_autoencoder_features as _implementation
sys.modules[__name__] = _implementation
