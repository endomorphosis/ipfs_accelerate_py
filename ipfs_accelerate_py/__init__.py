import backends
from .backends import backends
import config
from .config import config
import endpoints
from .endpoints import endpoints_py
import install_depends
from .install_depends import install_depends
import test_ipfs_accelerate
from .test_ipfs_accelerate import test_ipfs_accelerate
import ipfs_accelerate
from .ipfs_accelerate import ipfs_accelerate_py
export = {
    "backends": backends,
    "config": config,
    "test_ipfs_accelerate": test_ipfs_accelerate,
    "install_depends": install_depends,
    "endpoints": endpoints,
    "ipfs_accelerate_py": ipfs_accelerate_py
}