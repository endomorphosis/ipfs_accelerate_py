"""Canonical module entrypoint facade for unified MCP server package."""

from __future__ import annotations


def main() -> int:
    """Run the canonical standalone CLI module entrypoint."""
    from ipfs_accelerate_py.mcp_server.standalone_server import main as standalone_main

    standalone_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
