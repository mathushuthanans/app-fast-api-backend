import json
from typing import Any

from google import genai
from google.genai import types

from app.config import API_KEY_GEMINI, MODEL


client = genai.Client(api_key=API_KEY_GEMINI)


SYSTEM_INSTRUCTION = (
    "You are Clarity, an AI that returns pollution-related explanations "
    "in strict JSON format with keys: object, causes, effects. "
    "Always use less words, and use simpler and understandable words. "
    "Always assume pollutant units as:\n"
    "- CO: ppb\n"
    "- NO2: ppb\n"
    "- O3: µg/m³\n"
    "- PM10: µg/m³\n"
    "- PM2.5: µg/m³"
)


def ask_gemini(prompt: str) -> dict[str, Any]:
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
        ),
    )

    content = response.text

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse AI response as JSON",
            "raw_output": content,
        }
