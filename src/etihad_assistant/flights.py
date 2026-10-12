"""Validate and normalize real AviationStack status, arrivals, and schedule data."""

from __future__ import annotations

import re
import ssl
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from typing import Any

import httpx

from .config import Settings

AVIATIONSTACK_BASE_URL = "https://api.aviationstack.com/v1"
IATA_RE = re.compile(r"^[A-Z]{3}$")
FLIGHT_RE = re.compile(r"^[A-Z0-9]{2,3}\d{1,4}[A-Z]?$")
VALID_FLIGHT_STATUSES = frozenset(
    {"scheduled", "active", "landed", "cancelled", "incident", "diverted"}
)


class FlightProvider(ABC):
    """Operations exposed to the assistant through the MCP tool layer."""
    @abstractmethod
    def get_flight_status(
        self, flight_number: str, departure_date: date, origin_iata: str | None = None
    ) -> dict[str, Any]: ...

    @abstractmethod
    def list_airport_arrivals(self, airport_iata: str) -> dict[str, Any]: ...

    @abstractmethod
    def search_flight_schedule(
        self, origin_iata: str, destination_iata: str, travel_date: date
    ) -> dict[str, Any]: ...

    @abstractmethod
    def get_first_flight(
        self, origin_iata: str, destination_iata: str, travel_date: date
    ) -> dict[str, Any]: ...

    @abstractmethod
    def list_routes(
        self,
        origin_iata: str | None = None,
        destination_iata: str | None = None,
        airline_iata: str = "EY",
    ) -> dict[str, Any]: ...


class AviationStackProvider(FlightProvider):
    """Read-only AviationStack adapter with a fixed host and normalized responses."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.api_key = settings.aviationstack_api_key
        self.enabled = settings.enable_live_flight_data
        self.timeout = settings.aviationstack_timeout_seconds
        self.client = client

    @staticmethod
    def _retrieved_at() -> str:
        return datetime.now(UTC).isoformat()

    def _unavailable(self, notice: str, error_type: str) -> dict[str, Any]:
        """Return an explicit failure rather than fabricated flight details."""
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "unavailable",
            "error_type": error_type,
            "notice": notice,
        }

    @staticmethod
    def _iata(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not IATA_RE.fullmatch(normalized):
            raise ValueError(f"{field} must be a three-letter IATA code")
        return normalized

    @staticmethod
    def _flight_number(value: str) -> str:
        normalized = value.replace(" ", "").strip().upper()
        if not FLIGHT_RE.fullmatch(normalized):
            raise ValueError("flight_number is invalid")
        return normalized

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a fixed AviationStack endpoint and sanitize transport failures."""
        if not self.enabled:
            return self._unavailable(
                "Live flight data is disabled in configuration.", "provider_disabled"
            )
        if not self.api_key:
            return self._unavailable("AVIATIONSTACK_API_KEY is not configured.", "missing_api_key")

        request_params = {
            # The API key is never included in returned payloads or error notices.
            "access_key": self.api_key,
            **{key: value for key, value in params.items() if value is not None},
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(
            base_url=AVIATIONSTACK_BASE_URL,
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=False,
            verify=ssl.create_default_context(),
        )
        try:
            response = client.get(endpoint, params=request_params)
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError:
            return self._unavailable(
                "Cannot connect to AviationStack. Check outbound HTTPS, DNS, and proxy "
                "settings; no flight data was returned.",
                "ConnectError",
            )
        except (httpx.HTTPError, ValueError) as exc:
            return self._unavailable(
                "AviationStack request failed; no flight data was returned.",
                type(exc).__name__,
            )
        finally:
            if owns_client:
                client.close()

        if payload.get("success") is False or payload.get("error"):
            error = payload.get("error") or {}
            error_type = str(error.get("type") or error.get("code") or "provider_error")
            return self._unavailable(
                str(error.get("info") or "AviationStack returned an error."), error_type
            )
        return payload

    @staticmethod
    def _compact_flight(row: dict[str, Any]) -> dict[str, Any]:
        """Keep only relevant fields from a current/past flight record."""
        departure = row.get("departure") or {}
        arrival = row.get("arrival") or {}
        flight = row.get("flight") or {}
        airline = row.get("airline") or {}
        live = row.get("live") or {}
        return {
            "flight_date": row.get("flight_date"),
            "flight_status": str(row.get("flight_status") or "").strip().lower(),
            "airline": {"name": airline.get("name"), "iata": airline.get("iata")},
            "flight": {
                "number": flight.get("number"),
                "iata": flight.get("iata"),
                "icao": flight.get("icao"),
            },
            "departure": {
                "airport": departure.get("airport"),
                "iata": departure.get("iata"),
                "scheduled": departure.get("scheduled"),
                "estimated": departure.get("estimated"),
                "actual": departure.get("actual"),
                "delay_minutes": departure.get("delay"),
                "terminal": departure.get("terminal"),
                "gate": departure.get("gate"),
            },
            "arrival": {
                "airport": arrival.get("airport"),
                "iata": arrival.get("iata"),
                "scheduled": arrival.get("scheduled"),
                "estimated": arrival.get("estimated"),
                "actual": arrival.get("actual"),
                "delay_minutes": arrival.get("delay"),
                "terminal": arrival.get("terminal"),
                "gate": arrival.get("gate"),
            },
            "live": {
                "updated": live.get("updated"),
                "latitude": live.get("latitude"),
                "longitude": live.get("longitude"),
                "altitude": live.get("altitude"),
            }
            if live
            else None,
        }

    @staticmethod
    def _compact_future_flight(row: dict[str, Any], departure_date: date) -> dict[str, Any]:
        """Normalize future-timetable fields without implying live telemetry."""
        departure = row.get("departure") or {}
        arrival = row.get("arrival") or {}
        flight = row.get("flight") or {}
        airline = row.get("airline") or {}
        return {
            "flight_date": departure_date.isoformat(),
            "flight_status": "scheduled",
            "airline": {"name": airline.get("name"), "iata": airline.get("iataCode")},
            "flight": {
                "number": flight.get("number"),
                "iata": flight.get("iataNumber"),
                "icao": flight.get("icaoNumber"),
            },
            "departure": {
                "iata": departure.get("iataCode"),
                "scheduled": departure.get("scheduledTime"),
                "terminal": departure.get("terminal"),
                "gate": departure.get("gate"),
            },
            "arrival": {
                "iata": arrival.get("iataCode"),
                "scheduled": arrival.get("scheduledTime"),
                "terminal": arrival.get("terminal"),
                "gate": arrival.get("gate"),
            },
            "live": None,
        }

    def _future_flight_status(
        self, flight_number: str, departure_date: date, origin_iata: str | None
    ) -> dict[str, Any]:
        """Look for a confirmed future schedule, resolving origin if possible."""
        origin = origin_iata
        flight_digits = re.search(r"\d{1,4}", flight_number)
        airline_iata = flight_number[: flight_digits.start()] if flight_digits else ""
        number = flight_digits.group() if flight_digits else ""
        if origin is None:
            routes = self._get(
                "/routes",
                {"airline_iata": airline_iata, "flight_number": number, "limit": 100},
            )
            if routes.get("status") == "unavailable":
                return routes
            for route in routes.get("data", []):
                route_flight = route.get("flight") or {}
                route_airline = route.get("airline") or {}
                route_number = str(route.get("flight_number") or route_flight.get("number") or "")
                route_airline_code = str(
                    route.get("airline_iata") or route_airline.get("iata") or ""
                ).upper()
                if route_number != number or route_airline_code != airline_iata:
                    continue
                candidate = route.get("departure_airport_iata") or (
                    (route.get("departure") or {}).get("iata")
                )
                if candidate and IATA_RE.fullmatch(str(candidate).upper()):
                    origin = str(candidate).upper()
                    break
        if origin is None:
            return {
                "provider": "aviationstack",
                "retrieved_at": self._retrieved_at(),
                "fixture": False,
                "status": "clarify",
                "flight_status": None,
                "question": (
                    f"What is the departure airport IATA code for {flight_number} "
                    f"on {departure_date.isoformat()}?"
                ),
            }
        payload = self._get(
            "/flightsFuture",
            {
                "iataCode": origin,
                "type": "departure",
                "date": departure_date.isoformat(),
                "airline_iata": airline_iata,
                "flight_number": number,
                "limit": 100,
            },
        )
        if payload.get("status") == "unavailable":
            return payload
        matches = [
            self._compact_future_flight(row, departure_date)
            for row in payload.get("data", [])
            if str((row.get("flight") or {}).get("iataNumber") or "").upper()
            == flight_number
            and str((row.get("departure") or {}).get("iataCode") or "").upper()
            == origin
            and str((row.get("departure") or {}).get("scheduledTime") or "").startswith(
                departure_date.isoformat()
            )
        ]
        if not matches:
            return {
                "provider": "aviationstack",
                "retrieved_at": self._retrieved_at(),
                "fixture": False,
                "status": "not_found",
                "query": {"flight_number": flight_number, "departure_date": departure_date.isoformat()},
                "flight_status": None,
                "notice": (
                    f"AviationStack has no confirmed future schedule for {flight_number} "
                    f"on {departure_date.isoformat()}."
                ),
            }
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "ok",
            "query": {"flight_number": flight_number, "departure_date": departure_date.isoformat()},
            "flight_status": "scheduled",
            "status_source": "confirmed_future_schedule_not_live_status",
            "flight": matches[0],
            "notice": "Future schedule only; live operational status is not yet available.",
        }

    def get_flight_status(
        self, flight_number: str, departure_date: date, origin_iata: str | None = None
    ) -> dict[str, Any]:
        """Return a recognized status for the exact flight and departure date."""
        normalized_flight = self._flight_number(flight_number)
        normalized_origin = self._iata(origin_iata, "origin_iata")
        if departure_date > datetime.now(UTC).date():
            return self._future_flight_status(
                normalized_flight, departure_date, normalized_origin
            )
        params: dict[str, Any] = {
            "flight_iata": normalized_flight,
            "dep_iata": normalized_origin,
            "limit": 20,
        }
        if departure_date != datetime.now(UTC).date():
            params["flight_date"] = departure_date.isoformat()
        payload = self._get("/flights", params)
        if payload.get("status") == "unavailable":
            return payload

        candidates = [self._compact_flight(row) for row in payload.get("data", [])]
        matching = [
            # Provider search may include other dates or reused flight numbers.
            row
            for row in candidates
            if row.get("flight_date") == departure_date.isoformat()
            and str((row.get("flight") or {}).get("iata") or "").replace(" ", "").upper()
            == normalized_flight
            and (
                not normalized_origin
                or (row.get("departure") or {}).get("iata") == normalized_origin
            )
        ]
        selected = matching[0] if matching else None
        if not selected:
            return {
                "provider": "aviationstack",
                "retrieved_at": self._retrieved_at(),
                "fixture": False,
                "status": "not_found",
                "query": {
                    "flight_number": normalized_flight,
                    "departure_date": departure_date.isoformat(),
                    "origin_iata": normalized_origin,
                },
                "flight_status": None,
                "flight": None,
                "notice": (
                    f"AviationStack has no matching status for {normalized_flight} on "
                    f"{departure_date.isoformat()}."
                ),
            }
        if selected["flight_status"] not in VALID_FLIGHT_STATUSES:
            return {
                "provider": "aviationstack",
                "retrieved_at": self._retrieved_at(),
                "fixture": False,
                "status": "not_found",
                "query": {
                    "flight_number": normalized_flight,
                    "departure_date": departure_date.isoformat(),
                    "origin_iata": normalized_origin,
                },
                "flight_status": None,
                "flight": selected,
                "notice": (
                    f"AviationStack has no recognized flight status for {normalized_flight} "
                    f"on {departure_date.isoformat()}."
                ),
            }
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "ok",
            "query": {
                "flight_number": normalized_flight,
                "departure_date": departure_date.isoformat(),
                "origin_iata": normalized_origin,
            },
            "flight_status": selected["flight_status"],
            "flight": selected,
            "notice": None,
        }

    def list_airport_arrivals(self, airport_iata: str) -> dict[str, Any]:
        """Return up to 20 current matching arrivals, not a complete airport board."""
        airport = self._iata(airport_iata, "airport_iata")
        payload = self._get(
            "/timetable", {"iataCode": airport, "type": "arrival", "limit": 100}
        )
        if payload.get("status") == "unavailable":
            return payload

        arrivals: list[dict[str, Any]] = []
        for row in payload.get("data", []):
            arrival = row.get("arrival") or {}
            if str(arrival.get("iataCode") or "").upper() != airport:
                continue
            if row.get("type") not in (None, "arrival"):
                continue
            flight = row.get("flight") or {}
            flight_iata = str(flight.get("iataNumber") or "").upper()
            if not flight_iata:
                continue
            departure = row.get("departure") or {}
            raw_status = str(row.get("status") or "").strip().lower()
            arrivals.append(
                {
                    "flight_iata": flight_iata,
                    "airline": (row.get("airline") or {}).get("name"),
                    "from_iata": departure.get("iataCode"),
                    "scheduled_arrival": arrival.get("scheduledTime"),
                    "estimated_arrival": arrival.get("estimatedTime"),
                    "actual_arrival": arrival.get("actualTime"),
                    "terminal": arrival.get("terminal"),
                    "gate": arrival.get("gate"),
                    "flight_status": raw_status if raw_status in VALID_FLIGHT_STATUSES else None,
                }
            )
        if not arrivals:
            return {
                "provider": "aviationstack",
                "retrieved_at": self._retrieved_at(),
                "fixture": False,
                "status": "not_found",
                "notice": f"AviationStack returned no current arrivals for {airport}.",
                "airport_iata": airport,
            }
        pagination = payload.get("pagination") or {}
        total = pagination.get("total")
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "ok",
            "airport_iata": airport,
            "board_scope": "current_airport_arrivals",
            "arrivals": arrivals[:20],
            "matching_records_in_response": len(arrivals),
            "provider_total": total if isinstance(total, int) else None,
            "notice": "Showing up to 20 matching records from the current AviationStack board.",
        }

    def search_flight_schedule(
        self, origin_iata: str, destination_iata: str, travel_date: date
    ) -> dict[str, Any]:
        """Return all confirmed route matches found for the requested date."""
        origin = self._iata(origin_iata, "origin_iata")
        destination = self._iata(destination_iata, "destination_iata")
        route_result = self._route_flights(origin, destination, travel_date)
        if route_result["status"] != "ok":
            return route_result
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "ok",
            "origin_iata": origin,
            "destination_iata": destination,
            "date": travel_date.isoformat(),
            "schedules": route_result["flights"],
            "schedule_scope": route_result["schedule_scope"],
        }

    def _route_flights(
        self, origin_iata: str, destination_iata: str, travel_date: date
    ) -> dict[str, Any]:
        """Page through provider results before sorting route matches by departure."""
        today = datetime.now(UTC).date()
        future = travel_date > today
        endpoint = "/flightsFuture" if future else "/flights"
        params: dict[str, Any] = (
            {"iataCode": origin_iata, "type": "departure", "date": travel_date.isoformat()}
            if future
            else {"dep_iata": origin_iata, "arr_iata": destination_iata}
        )
        if not future and travel_date != today:
            params["flight_date"] = travel_date.isoformat()

        rows: list[dict[str, Any]] = []
        offset = 0
        # A partial board cannot establish which flight was first.
        for _page in range(10):
            payload = self._get(endpoint, {**params, "limit": 100, "offset": offset})
            if payload.get("status") == "unavailable":
                return payload
            page_rows = payload.get("data") or []
            if not isinstance(page_rows, list):
                return self._unavailable("AviationStack returned an invalid flight list.", "invalid_data")
            rows.extend(page_rows)
            pagination = payload.get("pagination") or {}
            total = pagination.get("total")
            if not isinstance(total, int) and len(page_rows) == 100:
                return self._unavailable("AviationStack omitted pagination for a full flight page.", "incomplete_data")
            if not isinstance(total, int) or offset + len(page_rows) >= total:
                break
            if not page_rows:
                return self._unavailable("AviationStack returned an incomplete flight list.", "incomplete_data")
            offset += len(page_rows)
        else:
            return self._unavailable("AviationStack flight list is too large to determine the first flight reliably.", "incomplete_data")

        compact = [
            self._compact_future_flight(row, travel_date) if future else self._compact_flight(row)
            for row in rows
            if isinstance(row, dict)
        ]
        matches = [
            item for item in compact
            if item.get("flight_date") == travel_date.isoformat()
            and str((item.get("departure") or {}).get("iata") or "").upper() == origin_iata
            and str((item.get("arrival") or {}).get("iata") or "").upper() == destination_iata
            and (item.get("departure") or {}).get("scheduled")
        ]
        matches.sort(key=lambda item: str(item["departure"]["scheduled"]))
        if not matches:
            return {
                "provider": "aviationstack", "retrieved_at": self._retrieved_at(),
                "fixture": False, "status": "not_found",
                "notice": f"AviationStack returned no {origin_iata}–{destination_iata} flights for {travel_date.isoformat()}.",
            }
        return {
            "provider": "aviationstack", "retrieved_at": self._retrieved_at(),
            "fixture": False, "status": "ok",
            "flights": matches,
            "schedule_scope": "future_schedule" if future else "flight_data",
        }

    def get_first_flight(
        self, origin_iata: str, destination_iata: str, travel_date: date
    ) -> dict[str, Any]:
        """Choose the earliest departure only after the route search completes."""
        origin = self._iata(origin_iata, "origin_iata")
        destination = self._iata(destination_iata, "destination_iata")
        route_result = self._route_flights(origin, destination, travel_date)
        if route_result["status"] != "ok":
            return route_result
        flights = route_result["flights"]
        return {
            "provider": "aviationstack", "retrieved_at": self._retrieved_at(),
            "fixture": False, "status": "ok",
            "query": {"origin_iata": origin, "destination_iata": destination,
                      "travel_date": travel_date.isoformat()},
            "first_flight": flights[0], "matching_flights": len(flights),
            "schedule_scope": route_result["schedule_scope"],
        }

    def list_routes(
        self,
        origin_iata: str | None = None,
        destination_iata: str | None = None,
        airline_iata: str = "EY",
    ) -> dict[str, Any]:
        """Return AviationStack route records filtered by optional airports and airline."""
        origin = self._iata(origin_iata, "origin_iata")
        destination = self._iata(destination_iata, "destination_iata")
        normalized_airline = airline_iata.strip().upper()
        if not re.fullmatch(r"^[A-Z0-9]{2,3}$", normalized_airline):
            raise ValueError("airline_iata is invalid")
        payload = self._get(
            "/routes",
            {
                "dep_iata": origin,
                "arr_iata": destination,
                "airline_iata": normalized_airline,
                "limit": 100,
            },
        )
        if payload.get("status") == "unavailable":
            return payload
        return {
            "provider": "aviationstack",
            "retrieved_at": self._retrieved_at(),
            "fixture": False,
            "status": "ok",
            "airline_iata": normalized_airline,
            "origin_iata": origin,
            "destination_iata": destination,
            "routes": payload.get("data", []),
        }
