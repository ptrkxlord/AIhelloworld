import sys

import fastapi
import requests

print("Python:", sys.version)
print("FastAPI:", fastapi.__version__)
print("Requests:", requests.__version__)

r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": "Kyiv", "count": 1}, timeout=5)
r.raise_for_status()
print("Open-Meteo geocoding reachable:", r.json()["results"][0]["name"])
