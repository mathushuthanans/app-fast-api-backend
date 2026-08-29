import os

if os.getenv("ENV") != "production":
    from dotenv import load_dotenv

    load_dotenv()


API_KEY_GEMINI: str = os.getenv("API_KEY_GEMINI", "")
MODEL: str = os.getenv("MODEL", "")
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_CURRENT_AIR_URL: str = os.getenv("OPENWEATHER_CURRENT_AIR_URL", "")
OPENWEATHER_FORECAST_AIR_URL: str = os.getenv("OPENWEATHER_FORECAST_AIR_URL", "")
OPENWEATHER_CURRENT_WEATHER_URL = os.getenv(
    "OPENWEATHER_CURRENT_WEATHER_URL", ""
)

OPENWEATHER_REVERSE_GEO_URL: str = os.getenv("OPENWEATHER_REVERSE_GEO_URL", "")


if not API_KEY_GEMINI or not MODEL or not OPENWEATHER_API_KEY or not OPENWEATHER_CURRENT_AIR_URL or not OPENWEATHER_FORECAST_AIR_URL or not OPENWEATHER_CURRENT_WEATHER_URL or not OPENWEATHER_REVERSE_GEO_URL:
    raise EnvironmentError("Missing required environment variables")
