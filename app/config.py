import os
from dotenv import load_dotenv

load_dotenv()

API_URL_WAQI = os.getenv("API_URL_WAQI")
API_KEY_GROQ = os.getenv("API_KEY_GROQ")
BASE_URL_GROQ = os.getenv("BASE_URL_GROQ")
