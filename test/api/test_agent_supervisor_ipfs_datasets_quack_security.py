"""LGSWF-072 adapter: datasets quack_security remains the server-side authority."""

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_quack_security import (
    AUTHORITY,
    scoped_authority,
)


def test_datasets_quack_security_module_is_importable() -> None:
    assert AUTHORITY.startswith("ipfs_datasets_py")
    assert scoped_authority() is not None
