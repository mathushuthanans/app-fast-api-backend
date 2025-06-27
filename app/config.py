import os

if os.getenv("ENV") != "production":
    from dotenv import load_dotenv
    load_dotenv()

API_URL_WAQI = os.getenv("API_URL_WAQI")
API_KEY_GROQ = os.getenv("API_KEY_GROQ")
BASE_URL_GROQ = os.getenv("BASE_URL_GROQ")

if not all([API_URL_WAQI, API_KEY_GROQ, BASE_URL_GROQ]):
    raise EnvironmentError("Missing one or more required environment variables.")
