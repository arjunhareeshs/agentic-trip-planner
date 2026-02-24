"""AviationStack API connector.

Provides flight search between cities.
"""

from __future__ import annotations

import httpx
import os
from utils.http_client import http_client

_BASE = "https://api.aviationstack.com/v1"


def search_flights(
    departure_iata: str,
    arrival_iata: str,
    date: str = "",
) -> list[dict]:
    """Search for flights between two airports.

    Args:
        departure_iata: IATA code of departure airport, e.g. "DEL".
        arrival_iata: IATA code of arrival airport, e.g. "MAA".
        date: optional date string YYYY-MM-DD.

    Returns:
        List of flight dicts with airline, flight_number, departure, arrival,
        status, and schedule information.
    """
    api_key = os.getenv("AVIATIONSTACK_API_KEY", "")
    if not api_key:
        print("⚠️ [AviationStack] AVIATIONSTACK_API_KEY not configured in .env")
        return [{"error": "AVIATIONSTACK_API_KEY not configured in .env"}]

    try:
        params: dict = {
            "access_key": api_key,
            "dep_iata": departure_iata,
            "arr_iata": arrival_iata,
            "limit": 10,
        }
        if date:
            params["flight_date"] = date

        resp = http_client.get(f"{_BASE}/flights", params=params)
        resp.raise_for_status()
        data = resp.json()

        flights = []
        for f in data.get("data", [])[:10]:
            flights.append(
                {
                    "airline": f.get("airline", {}).get("name"),
                    "flight_number": f.get("flight", {}).get("iata"),
                    "departure_airport": f.get("departure", {}).get("airport"),
                    "departure_time": f.get("departure", {}).get("scheduled"),
                    "arrival_airport": f.get("arrival", {}).get("airport"),
                    "arrival_time": f.get("arrival", {}).get("scheduled"),
                    "status": f.get("flight_status"),
                }
            )
        return flights if flights else [{"info": "No flights found for this route/date."}]
    except httpx.HTTPStatusError as exc:
        print(f"❌ [AviationStack] HTTP error {exc.response.status_code}: {exc.response.text}")
        return [{"error": f"AviationStack API returned {exc.response.status_code}"}]
    except httpx.RequestError as exc:
        print(f"❌ [AviationStack] Request error: {exc}")
        return [{"error": "Failed to connect to AviationStack API"}]
    except Exception as exc:
        print(f"❌ [AviationStack] Unexpected error: {exc}")
        return [{"error": f"AviationStack search failed: {exc}"}]
