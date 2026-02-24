"""OpenWeather API connector.

Provides weather forecast for trip planning.
"""

from __future__ import annotations

import httpx
import os
from utils.http_client import http_client

_BASE = "https://api.openweathermap.org/data/2.5"


def get_weather_forecast(lat: float, lng: float, days: int = 5) -> list[dict]:
    """Get weather forecast for a location.

    Args:
        lat: latitude of the location.
        lng: longitude of the location.
        days: number of days to forecast (max 5 for free tier).

    Returns:
        List of daily forecast dicts with date, temp, weather, humidity, wind.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        print("⚠️ [OpenWeather] OPENWEATHER_API_KEY not configured in .env")
        return [{"error": "OPENWEATHER_API_KEY not configured in .env"}]

    try:
        resp = http_client.get(
            f"{_BASE}/forecast",
            params={"lat": lat, "lon": lng, "appid": api_key, "units": "metric", "cnt": days * 8}
        )
        resp.raise_for_status()
        data = resp.json()

        # Group by day (take noon reading)
        daily: dict[str, dict] = {}
        for item in data.get("list", []):
            date = item["dt_txt"].split(" ")[0]
            hour = item["dt_txt"].split(" ")[1]
            if date not in daily or hour.startswith("12"):
                daily[date] = {
                    "date": date,
                    "temp_c": item["main"]["temp"],
                    "feels_like_c": item["main"]["feels_like"],
                    "humidity_pct": item["main"]["humidity"],
                    "weather": item["weather"][0]["description"],
                    "wind_speed_mps": item["wind"]["speed"],
                    "rain_mm": item.get("rain", {}).get("3h", 0),
                }

        forecasts = list(daily.values())[:days]
        if not forecasts:
            return [{"error": "No forecast data available for this location."}]
        return forecasts
    except httpx.HTTPStatusError as exc:
        print(f"❌ [OpenWeather] HTTP error {exc.response.status_code}: {exc.response.text}")
        return [{"error": f"OpenWeather API returned {exc.response.status_code}"}]
    except httpx.RequestError as exc:
        print(f"❌ [OpenWeather] Request error: {exc}")
        return [{"error": "Failed to connect to OpenWeather API"}]
    except Exception as exc:
        print(f"❌ [OpenWeather] Unexpected error: {exc}")
        return [{"error": f"OpenWeather forecast failed: {exc}"}]
