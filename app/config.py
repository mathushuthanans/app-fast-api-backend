import os
from dotenv import load_dotenv

load_dotenv()
API_URL_WAQI = "https://api.waqi.info/feed/here/?token=95436f0dbd6f640af4ac23fc61c4352e6af45f8e"
api_key = "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"
base_url="https://api.groq.com/openai/v1"

print(API_URL_WAQI, api_key, base_url)