import backends
from .backends import backends
import config
from .config import config
from .ipfs_accelerate import load_checkpoint_and_dispatch
import ipfs_accelerate_py
import test_ipfs_accelerate
from .test_ipfs_accelerate import test_ipfs_accelerate
export = {
    "backends": backends,
    "config": config,
    "load_checkpoint_and_dispatch": load_checkpoint_and_dispatch,
    "test_ipfs_accelerate": test_ipfs_accelerate
}