from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.weather_client import WeatherError, get_weather

app = FastAPI(title="PR1 Weather")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).parent.joinpath("static/index.html").read_text(encoding="utf-8")


@app.get("/weather")
def weather(city: str = Query(..., min_length=1)):
    try:
        return get_weather(city)
    except WeatherError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="weather service timed out")
    except requests.HTTPError as e:
        status = e.response.status_code
        raise HTTPException(status_code=400 if status < 500 else 502, detail="weather service error")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="weather service unavailable")
