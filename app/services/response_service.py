from fastapi import APIRouter
import requests
import joblib
import numpy as np
import os
from app.models.datamodel1 import HealthRiskRequest, RegionClassRequest
from app.services import extract_locations  # Fixed import path
from app.config import API_URL_WAQI
from ml_models.ai_logic import ask_groq
from ml_models.utils import calculate_overall_aqi as calc


router = APIRouter()

MODEL_DIR = extract_locations.extract()
health_model = joblib.load(os.path.join(MODEL_DIR, "health_risk_model.pkl"))
region_model = joblib.load(os.path.join(MODEL_DIR, "region_model.pkl"))



class ResponseService:
    def __init__(self):
        self.router = router
        self.router.add_api_route("/get_location", self.get_location, methods=["GET"])
        self.router.add_api_route("/predict_health_risk", self.predict_health_risk, methods=["POST"])
        self.router.add_api_route("/predict_region_class", self.predict_region_class, methods=["POST"])
        self.router.add_api_route("/bridge_predict", self.bridge_predict, methods=["GET"])

    def get_location(self):
        response = requests.get(API_URL_WAQI)

        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}"}
        
        data = response.json()
        if data.get("status") != "ok":
            return {"error": "API returned non-ok status."}
        
        city = data["data"]["city"]["name"]
        coordinates = data["data"]["city"]["geo"]
        iaqi = data["data"]["iaqi"]

        pm25 = iaqi.get("pm25", {}).get("v")
        pm10 = iaqi.get("pm10", {}).get("v")
        no2 = iaqi.get("no2", {}).get("v")
        o3 = iaqi.get("o3", {}).get("v")
        co = iaqi.get("co", {}).get("v")

        return {
            "city": city,
            "coordinates": coordinates,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "o3": o3,
            "co": co
        }

    def predict_health_risk(self, data: HealthRiskRequest):
        features = np.array([[data.pm25, data.pm10, data.no2, data.o3, data.co, data.asthma, data.heart_disease]])
        danger_scale = health_model.predict(features)[0]
        return {"danger_scale": danger_scale}
            
    def predict_region_class(self, data: RegionClassRequest):
        features = np.array([[data.pm25, data.pm10, data.no2, data.o3, data.co]])
        region_class = region_model.predict(features)[0]
        return {"region_class": region_class}

    def bridge_predict(self, asthma: int = 0, heart_disease: int = 0):
        response = requests.get(API_URL_WAQI)
        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}"}

        data = response.json()
        if data.get("status") != "ok":
            return {"error": "API returned non-ok status."}
        
        iaqi = data["data"]["iaqi"]
        pm25 = iaqi.get("pm25", {}).get("v", None)
        pm10 = iaqi.get("pm10", {}).get("v", None)
        no2 = iaqi.get("no2", {}).get("v", None)
        o3 = iaqi.get("o3", {}).get("v", None)
        co = iaqi.get("co", {}).get("v", None)

        if None in [pm25, pm10, no2, o3, co]:
            return {"error": "Incomplete AQI data received"}

        # Predict
        health_features = np.array([[pm25, pm10, no2, o3, co, asthma, heart_disease]])
        danger_scale = int(health_model.predict(health_features)[0])

        region_features = np.array([[pm25, pm10, no2, o3, co]])
        region_class = int(region_model.predict(region_features)[0])

        return {
            "location": {
                "city": data["data"]["city"]["name"],
                "coordinates": data["data"]["city"]["geo"]
            },
            "aqi_values": {
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "o3": o3,
                "co": co
            },
            "predictions": {
                "danger_scale": danger_scale,
                "region_class": region_class
            }
        }



    def help_info(self):
        message = """
    Clarity is an AI assistant that explains pollution data and health risks to citizens, policymakers, and students. It suggests actions, predicts policy impacts, creates quizzes, and simulates urban planning scenarios to promote pollution awareness and healthier communities.
    """
        return {"help": message}


    def explain_pollution(self, pollutant: str):
        prompt = (
            f"Explain the pollutant '{pollutant}'. "
            "Respond strictly in this JSON format:\n\n"
            "{\n"
            f'  "object": "What is {pollutant}?",\n'
            '  "causes": ["Cause 1", "Cause 2", "Cause 3"],\n'
            '  "effects": ["Effect 1", "Effect 2", "Effect 3"]\n'
            "}\n"
            "Do not include any extra text or explanation outside the JSON."
        )
        result = ask_groq(prompt)
        return result


    def predict_policy(self, policy: str, location: str, pm25: float, pm10: float, no2: float, co: float, o3: float):
        aqi = calc(pm25, pm10, no2, co, o3)
        prompt = f"""
    You are Clarity, an AI model that predicts the outcome of pollution-control policies.

    Inputs:
    - Policy: {policy}
    - Location: {location}
    - Current AQI: {aqi}
    - Pollutant measures: PM2.5={pm25}, PM10={pm10}, NO2={no2}, CO={co}, O3={o3}

    Return ONLY the following JSON object:
    {{
    "effects_of_policy": ["Effect 1", "Effect 2", "Effect 3"],
    "efficiency_ratio": 0.0,
    "old_aqi": {aqi},
    "new_aqi": 0.0
    }}
    Do not include any other explanation.
    """
        result = ask_groq(prompt)
        return result


    def compare_locations(self, location1: str, location2: str, pm25_1: float, pm10_1: float, pm25_2: float, pm10_2: float):
        prompt = f"""
    You are Clarity.
    Compare pollution health impact between:
    - {location1}: PM2.5={pm25_1}, PM10={pm10_1}
    - {location2}: PM2.5={pm25_2}, PM10={pm10_2}
    Give 2-line difference summary.
    Return as {{"comparison": "..."}}
    """
        return ask_groq(prompt)


    def health_risks(self, pm25: float, pm10: float, no2: float, co: float, o3: float):
        aqi = calc(pm25, pm10, no2, co, o3)
        prompt = f"""
    You are Clarity.
    AQI: {aqi}
    Pollutants: PM2.5={pm25}, PM10={pm10}, NO2={no2}, CO={co}, O3={o3}
    Give health risk summary per group in this JSON:
    {{
    "children": "...",
    "adults": "...",
    "elderly": "..."
    }}
    """
        return ask_groq(prompt)



    def suggest_policies(self, location: str, aqi: int, pm25: float, pm10: float, no2: float, co: float, o3: float):

        debug_policy_suggestions = False
        prompt = f"""
    You are Clarity, a narrow AI assistant for pollution policy.
    Suggest 3 effective policies based on the following context:
    Location: {location}
    AQI: {aqi}
    Pollutant levels:
    - PM2.5: {pm25}
    - PM10: {pm10}
    - NO2: {no2}
    - CO: {co}
    - O3: {o3}

    Respond only with a JSON array:
    ["Policy 1", "Policy 2", "Policy 3"]
    """
        return ask_groq(prompt) if not debug_policy_suggestions else ["Plant trees", "Ban diesel", "Promote cycling"]


    def citizen_actions(self, pm25: float, pm10: float, no2: float, co: float, o3: float):
        prompt = f"""
    You are Clarity, an AI assistant for pollution awareness.

    Given current pollutant levels (units: CO and NO2 in ppb; O3, PM10, PM2.5 in µg/m³):
    - PM2.5: {pm25}
    - PM10: {pm10}
    - NO2: {no2}
    - CO: {co}
    - O3: {o3}

    Suggest 5 simple, practical actions citizens can take right now to reduce their exposure and protect their health.

    Return a JSON array of action strings only, like:
    [
    \"Action 1\",
    \"Action 2\",
    \"Action 3\",
    \"Action 4\",
    \"Action 5\"
    ]
    """
        return ask_groq(prompt)


    def daily_tip(self):
        prompt = """
    You are Clarity, an AI assistant that gives short, actionable tips or facts about pollution.

    Give one daily pollution-related tip or fact that is:
    - Easy to understand
    - Practical or educational
    - No more than 2 sentences

    Respond as plain text.
    """
        result = ask_groq(prompt)
        return {"tip": result if isinstance(result, str) else str(result)}


    def myth_buster(self, claim: str):
        prompt = f"""
    You are Clarity, an AI myth-buster for pollution.

    Analyze the following claim and respond whether it's True or False, followed by a short explanation.

    Claim: "{claim}"

    Respond in JSON format:
    {{
    "verdict": "True" or "False",
    "explanation": "..."
    }}
    """
        return ask_groq(prompt)


    def reduce_pollution_plan(self, goal: str, location: str):
        prompt = f"""
    You are Clarity, an AI assistant helping design pollution-reduction strategies.

    Goal: {goal}
    Location: {location}

    Give a 3-point actionable plan that can help achieve this goal. Be realistic and location-aware.

    Respond in JSON format:
    {{
    "goal": "{goal}",
    "location": "{location}",
    "plan": ["Step 1", "Step 2", "Step 3"]
    }}
    """
        return ask_groq(prompt)


response_service = ResponseService()
