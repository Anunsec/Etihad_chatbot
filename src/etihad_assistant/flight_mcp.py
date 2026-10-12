"""MCP tools and LangChain adapter for the read-only AviationStack provider."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

from fastmcp import FastMCP
from langchain.mcp import MCPAdapter

from .flights import FlightProvider


class MCPToolManager:
    """Dispatch the supplied tool categories to the configured live provider.

    Unlike the pasted example, provider failures are returned as unavailable; this
    manager never substitutes mock flights or a hard-coded access key.
    """

    def __init__(self, provider: FlightProvider):
        self.provider = provider

    def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Map public MCP tool names to the provider's typed operations."""
        if tool_name == "flight_status":
            return self.provider.get_flight_status(
                kwargs["flight_number"],
                date.fromisoformat(kwargs["date"]),
                kwargs.get("origin_iata"),
            )
        if tool_name == "flight_schedule":
            return self.provider.search_flight_schedule(
                kwargs["origin"], kwargs["destination"],
                date.fromisoformat(kwargs["date"]),
            )
        if tool_name == "routes":
            return self.provider.list_routes(
                kwargs.get("origin"), kwargs.get("destination"),
                kwargs.get("airline_iata", "EY"),
            )
        if tool_name == "first_flight":
            return self.provider.get_first_flight(
                kwargs["origin"], kwargs["destination"],
                date.fromisoformat(kwargs["date"]),
            )
        if tool_name == "airport_arrivals":
            return self.provider.list_airport_arrivals(kwargs["airport_iata"])
        raise ValueError(f"Unknown AviationStack MCP tool: {tool_name}")


def create_flight_mcp(provider: FlightProvider) -> FastMCP:
    """Create the AviationStack MCP server used by both the app and CLI server."""
    server = FastMCP("AviationStack Flight Data")
    manager = MCPToolManager(provider)

    @server.tool()
    def get_flight_status(
        flight_number: str, departure_date: str, origin_iata: str | None = None
    ) -> dict[str, Any]:
        """Return normalized AviationStack flight status; accepts no passenger data."""
        return manager.call_tool(
            "flight_status",
            flight_number=flight_number,
            date=departure_date,
            origin_iata=origin_iata,
        )

    @server.tool()
    def list_airport_arrivals(airport_iata: str) -> dict[str, Any]:
        """Return the current AviationStack airport arrivals board; no synthetic flights."""
        return manager.call_tool("airport_arrivals", airport_iata=airport_iata)

    @server.tool()
    def search_flight_schedule(
        origin_iata: str, destination_iata: str, travel_date: str
    ) -> dict[str, Any]:
        """Return normalized AviationStack schedule data; accepts no passenger data."""
        return manager.call_tool(
            "flight_schedule", origin=origin_iata, destination=destination_iata,
            date=travel_date,
        )

    @server.tool()
    def get_first_flight(
        origin_iata: str, destination_iata: str, travel_date: str
    ) -> dict[str, Any]:
        """Find the earliest scheduled departure for a route using AviationStack only."""
        return manager.call_tool(
            "first_flight", origin=origin_iata, destination=destination_iata,
            date=travel_date,
        )

    @server.tool()
    def list_routes(
        origin_iata: str | None = None,
        destination_iata: str | None = None,
        airline_iata: str = "EY",
    ) -> dict[str, Any]:
        """Return normalized AviationStack route data; accepts no passenger data."""
        return manager.call_tool(
            "routes", origin=origin_iata, destination=destination_iata,
            airline_iata=airline_iata,
        )

    return server


def _normalise_tool_result(value: Any) -> dict[str, Any]:
    """Unwrap MCP text/content variants into the provider's response dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"status": "error", "notice": value}
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                text_parts.append(item.text)
        if text_parts:
            return _normalise_tool_result("\n".join(text_parts))
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return _normalise_tool_result(content)
    return {"status": "error", "notice": f"Unexpected MCP result: {type(value).__name__}"}


async def invoke_flight_mcp(
    provider: FlightProvider, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Invoke a tool through the in-process MCP server and LangChain adapter."""
    server = create_flight_mcp(provider)
    async with MCPAdapter(server) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}
        if tool_name not in tools:
            return {"status": "error", "notice": f"MCP tool {tool_name} is unavailable"}
        return _normalise_tool_result(await tools[tool_name].ainvoke(arguments))


def invoke_flight_mcp_sync(
    provider: FlightProvider, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Expose the async MCP invocation to synchronous LangGraph nodes."""
    return asyncio.run(invoke_flight_mcp(provider, tool_name, arguments))
