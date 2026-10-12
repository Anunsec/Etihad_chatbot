"""Standalone MCP entry point exposing the configured flight tools."""

from __future__ import annotations

from .config import get_settings
from .flight_mcp import create_flight_mcp
from .flights import AviationStackProvider

mcp = create_flight_mcp(AviationStackProvider(get_settings()))


def run() -> None:
    """Serve the same MCP tool definitions used by the assistant graph."""
    mcp.run()


if __name__ == "__main__":
    run()
