"""AviationStack and MCP tests that reject fabricated or mismatched flight data."""

from __future__ import annotations

import ssl
from datetime import date

import httpx

from etihad_assistant.config import Settings
from etihad_assistant.flight_mcp import invoke_flight_mcp_sync
from etihad_assistant.flights import AVIATIONSTACK_BASE_URL, AviationStackProvider


def test_aviationstack_status_is_normalized_without_exposing_key() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "flight_date": "2026-10-06",
                        "flight_status": "active",
                        "airline": {"name": "Etihad Airways", "iata": "EY"},
                        "flight": {"number": "20", "iata": "EY20", "icao": "ETD20"},
                        "departure": {
                            "airport": "Abu Dhabi International",
                            "iata": "AUH",
                            "scheduled": "2026-10-06T10:00:00+00:00",
                            "delay": 5,
                        },
                        "arrival": {
                            "airport": "London Heathrow",
                            "iata": "LHR",
                            "scheduled": "2026-10-06T15:00:00+00:00",
                        },
                        "live": {"updated": "2026-10-06T10:05:00+00:00"},
                    }
                ]
            },
        )

    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    settings = Settings(
        aviationstack_api_key="test-secret-key",
        enable_live_flight_data=True,
    )
    provider = AviationStackProvider(settings, client=client)
    result = provider.get_flight_status("EY20", date(2026, 10, 6), "AUH")
    client.close()

    assert result["provider"] == "aviationstack"
    assert result["fixture"] is False
    assert result["flight_status"] == "active"
    assert result["flight"]["departure"]["delay_minutes"] == 5
    assert "test-secret-key" not in str(result)
    assert seen_request is not None
    assert seen_request.url.host == "api.aviationstack.com"
    assert seen_request.url.params["flight_iata"] == "EY20"
    assert seen_request.url.params["access_key"] == "test-secret-key"


def test_live_http_client_uses_system_trust_store(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubClient:
        def get(self, _endpoint: str, params: dict) -> httpx.Response:
            assert params["access_key"] == "test-key"
            return httpx.Response(
                200, json={"data": []},
                request=httpx.Request("GET", "https://api.aviationstack.com/v1/flights"),
            )

        def close(self) -> None:
            captured["closed"] = True

    def make_client(**kwargs) -> StubClient:
        captured.update(kwargs)
        return StubClient()

    monkeypatch.setattr("etihad_assistant.flights.httpx.Client", make_client)
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True)
    )
    result = provider.get_first_flight("DEL", "AUH", date(2026, 10, 1))
    assert result["status"] == "not_found"
    assert isinstance(captured["verify"], ssl.SSLContext)
    assert captured["verify"].verify_mode == ssl.CERT_REQUIRED
    assert captured["closed"] is True


def test_aviationstack_requires_matching_flight_date_and_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "flight_date": "2026-09-30",
                        "flight_status": "landed",
                        "flight": {"iata": "EY219"},
                    },
                    {
                        "flight_date": "2026-10-01",
                        "flight_status": "unknown",
                        "flight": {"iata": "EY219"},
                    },
                ]
            },
        )

    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler)
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = provider.get_flight_status("EY219", date(2026, 10, 1))
    client.close()
    assert result["status"] == "not_found"
    assert result["flight_status"] is None


def test_aviationstack_empty_result_has_no_synthetic_status() -> None:
    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = provider.get_flight_status("EY219", date(2026, 10, 1))
    client.close()
    assert result["status"] == "not_found"
    assert result["flight_status"] is None


def test_future_flight_uses_confirmed_schedule_not_live_feed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/routes"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "airline_iata": "EY",
                            "flight_number": "219",
                            "departure_airport_iata": "AUH",
                        }
                    ]
                },
            )
        assert request.url.path.endswith("/flightsFuture")
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "flight": {"number": "219", "iataNumber": "EY219"},
                        "departure": {
                            "iataCode": "AUH",
                            "scheduledTime": "2099-10-10 06:00:00",
                        },
                        "arrival": {"iataCode": "DEL"},
                        "airline": {"iataCode": "EY"},
                    }
                ]
            },
        )

    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler)
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = provider.get_flight_status("EY219", date(2099, 10, 10))
    client.close()

    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "routes", "flightsFuture"
    ]
    assert requests[1].url.params["iataCode"] == "AUH"
    assert requests[1].url.params["date"] == "2099-10-10"
    assert result["flight_status"] == "scheduled"
    assert result["status_source"] == "confirmed_future_schedule_not_live_status"


def test_future_flight_without_route_asks_for_airport() -> None:
    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = provider.get_flight_status("EY219", date(2099, 10, 10))
    client.close()
    assert result["status"] == "clarify"
    assert "departure airport" in result["question"]


def test_arrivals_timetable_is_retrieved_through_mcp_without_synthetic_rows() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "type": "arrival",
                        "status": "landed",
                        "flight": {"iataNumber": "EY218"},
                        "airline": {"name": "Etihad Airways"},
                        "departure": {"iataCode": "AUH"},
                        "arrival": {
                            "iataCode": "DEL",
                            "scheduledTime": "2026-10-07T18:30:00",
                        },
                    },
                    {
                        "type": "arrival",
                        "status": "scheduled",
                        "flight": {"iataNumber": "EY999"},
                        "arrival": {"iataCode": "BOM"},
                    },
                ]
            },
        )

    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler)
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = invoke_flight_mcp_sync(provider, "list_airport_arrivals", {"airport_iata": "DEL"})
    client.close()

    assert len(requests) == 1
    assert requests[0].url.path.endswith("/timetable")
    assert requests[0].url.params["iataCode"] == "DEL"
    assert requests[0].url.params["type"] == "arrival"
    assert result["status"] == "ok"
    assert len(result["arrivals"]) == 1
    assert result["arrivals"][0]["flight_iata"] == "EY218"
    assert result["arrivals"][0]["flight_status"] == "landed"


def test_first_flight_mcp_sorts_provider_rows_and_has_no_airline_filter() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [
            {"flight_date": "2026-10-01", "flight": {"iata": "EY100"},
             "departure": {"iata": "DEL", "scheduled": "2026-10-01T08:00:00+05:30"},
             "arrival": {"iata": "LHR"}},
            {"flight_date": "2026-10-01", "flight": {"iata": "BA200"},
             "departure": {"iata": "DEL", "scheduled": "2026-10-01T06:00:00+05:30"},
             "arrival": {"iata": "LHR"}},
            {"flight_date": "2026-10-01", "flight": {"iata": "BA999"},
             "departure": {"iata": "DEL", "scheduled": "2026-10-01T05:00:00+05:30"},
             "arrival": {"iata": "LGW"}},
        ]})

    client = httpx.Client(base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler))
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = invoke_flight_mcp_sync(provider, "get_first_flight", {
        "origin_iata": "DEL", "destination_iata": "LHR", "travel_date": "2026-10-01"
    })
    client.close()
    assert result["status"] == "ok"
    assert result["first_flight"]["flight"]["iata"] == "BA200"
    assert result["matching_flights"] == 2
    assert requests[0].url.path.endswith("/flights")
    assert requests[0].url.host == "api.aviationstack.com"
    assert requests[0].url.params["dep_iata"] == "DEL"
    assert requests[0].url.params["arr_iata"] == "LHR"
    assert "airline_iata" not in requests[0].url.params


def test_schedule_mcp_uses_pasted_route_query_without_mock_fallback() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [
            {"flight_date": "2026-10-01", "flight_status": "landed",
             "airline": {"name": "Etihad Airways", "iata": "EY"},
             "flight": {"iata": "EY213"},
             "departure": {"iata": "DEL", "scheduled": "2026-10-01T06:00:00+05:30"},
             "arrival": {"iata": "AUH", "scheduled": "2026-10-01T08:00:00+04:00"}},
            {"flight_date": "2026-10-01", "flight": {"iata": "EY999"},
             "departure": {"iata": "DEL", "scheduled": "2026-10-01T05:00:00+05:30"},
             "arrival": {"iata": "LHR"}},
        ]})

    client = httpx.Client(base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler))
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = invoke_flight_mcp_sync(provider, "search_flight_schedule", {
        "origin_iata": "DEL", "destination_iata": "AUH", "travel_date": "2026-10-01"
    })
    client.close()
    assert result["status"] == "ok"
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["flight"]["iata"] == "EY213"
    assert requests[0].url.path == "/v1/flights"
    assert requests[0].url.params["dep_iata"] == "DEL"
    assert requests[0].url.params["arr_iata"] == "AUH"
    assert requests[0].url.params["flight_date"] == "2026-10-01"
    assert "airline_iata" not in requests[0].url.params


def test_first_flight_empty_provider_result_has_no_synthetic_flight() -> None:
    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = provider.get_first_flight("DEL", "LHR", date(2026, 10, 1))
    client.close()
    assert result["status"] == "not_found"
    assert "first_flight" not in result

def test_first_flight_connection_error_is_actionable_and_has_no_flight() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection blocked", request=request)

    client = httpx.Client(
        base_url=AVIATIONSTACK_BASE_URL, transport=httpx.MockTransport(handler)
    )
    provider = AviationStackProvider(
        Settings(aviationstack_api_key="test-key", enable_live_flight_data=True), client=client
    )
    result = invoke_flight_mcp_sync(provider, "get_first_flight", {
        "origin_iata": "DEL", "destination_iata": "AUH", "travel_date": "2026-10-01"
    })
    client.close()
    assert result["status"] == "unavailable"
    assert result["error_type"] == "ConnectError"
    assert "HTTPS" in result["notice"]
    assert "first_flight" not in result
