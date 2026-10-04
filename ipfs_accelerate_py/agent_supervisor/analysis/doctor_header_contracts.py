"""Compatibility alias for canonical datasets header-contract analysis."""
import sys
from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as _canonical
sys.modules[__name__] = _canonical
