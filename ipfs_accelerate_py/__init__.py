from .container_backends import backends
from .install_depends import install_depends
from .ipfs_accelerate import ipfs_accelerate_py
from .ipfs_multiformats import ipfs_multiformats_py
from .worker import worker
from .config import config
from .test import test_ipfs_accelerate
from .test import test_hardware_backend
from .test import test_api_backend

export = {
    "backends": backends,
    "config": config,
    "test_ipfs_accelerate": test_ipfs_accelerate,
    "test_api_backend": test_api_backend,
    "test_hardware_backend": test_hardware_backend,
    "install_depends": install_depends,
    "ipfs_accelerate_py": ipfs_accelerate_py,
    "worker": worker,
    "ipfs_multiformats_py": ipfs_multiformats_py
}