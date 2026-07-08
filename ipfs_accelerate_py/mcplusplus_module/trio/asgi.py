"""ASGI entrypoint for Hypercorn.

Hypercorn's CLI does not consistently support a "factory" flag across versions.
Expose a module-level `app` so systemd can run:

  python -m hypercorn ipfs_accelerate_py.mcplusplus_module.trio.asgi:app

The underlying app is still created by `create_app()`.
"""

from __future__ import annotations

from .server import create_app

app = create_app()
