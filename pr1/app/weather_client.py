import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 5


class WeatherError(Exception):
    pass


def geocode(city: str) -> tuple[float, float, str]:
    response = requests.get(GEOCODING_URL, params={"name": city, "count": 1}, timeout=TIMEOUT)
    response.raise_for_status()
    results = response.json().get("results")
    if not results:
        raise WeatherError(f"city not found: {city}")
    match = results[0]
    return match["latitude"], match["longitude"], match["name"]


def get_weather(city: str) -> dict:
    latitude, longitude, resolved_name = geocode(city)

    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    current = response.json().get("current")
    if current is None or "temperature_2m" not in current or "wind_speed_10m" not in current:
        raise WeatherError("unexpected response from forecast API")

    return {
        "city": resolved_name,
        "temperature_c": current["temperature_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
    }
