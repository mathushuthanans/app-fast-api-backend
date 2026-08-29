from openai import OpenAI
import json
from app.config import api_key, base_url


client = OpenAI(api_key=api_key, base_url=base_url)

def ask_gemini(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Clarity, an AI that returns pollution-related explanations "
                    "in strict JSON format with keys: object, causes, effects. "
                    "Always use less words, and use simpler and understandable words"
                    "Always assume pollutant units as:\n"
                    "- CO: ppb\n"
                    "- NO2: ppb\n"
                    "- O3: µg/m³\n"
                    "- PM10: µg/m³\n"
                    "- PM2.5: µg/m³"
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON", "raw_output": content}
