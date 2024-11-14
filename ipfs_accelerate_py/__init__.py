from .worker import worker
from .config import config
from .endpoints import endpoints_py
from .install_depends import install_depends
from .test_ipfs_accelerate import test_ipfs_accelerate
from .ipfs_accelerate import ipfs_accelerate_py
export = {
    "config": config,
    "test_ipfs_accelerate": test_ipfs_accelerate,
    "install_depends": install_depends,
    "endpoints": endpoints,
    "ipfs_accelerate_py": ipfs_accelerate_py,
    "worker": worker
}