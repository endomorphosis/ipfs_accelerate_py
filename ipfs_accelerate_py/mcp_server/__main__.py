"""Canonical module entrypoint facade for unified MCP server package."""

from __future__ import annotations


def main() -> int:
    """Delegate CLI module entrypoint to compatibility MCP CLI."""
    from ipfs_accelerate_py.mcp.__main__ import main as legacy_main

    return int(legacy_main())


if __name__ == "__main__":
    raise SystemExit(main())
