"""Compatibility alias for the datasets-owned reusable implementation."""
import sys as _sys
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_cve_canonical_export as _canonical

_sys.modules[__name__] = _canonical
