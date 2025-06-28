import math
from fastapi import APIRouter, HTTPException
import requests
import joblib
import numpy as np
import os
from app.models.datamodel1 import HealthRiskRequest, RegionClassRequest
from app.services import extract_locations  # Fixed import path
from app.config import API_URL_WAQI
from ml_models.ai_logic import ask_groq
from ml_models.utils import calculate_overall_aqi as calc
from datetime import datetime, timedelta
import random
import json 


router = APIRouter()

MODEL_DIR = extract_locations.extract()
health_model = joblib.load(os.path.join(MODEL_DIR, "health_risk_model.pkl"))
region_model = joblib.load(os.path.join(MODEL_DIR, "region_model.pkl"))



class ResponseService:
    def __init__(self):
        router.add_api_route("/get_location", self.get_location, methods=["GET"])
        router.add_api_route("/get_trends", self.get_trends, methods=["GET"])

        self.college_coords = (21.1199, 79.0196)
        self.industrial_areas = {
            "Hingna MIDC": {"distance": 5.2, "direction": "NE", "impact": 0.7},
            "Butibori": {"distance": 12.5, "direction": "SW", "impact": 0.9},
            "Kamptee Road": {"distance": 3.8, "direction": "E", "impact": 0.4}
        }

        self._current_data = None
        self._last_forecast = None

    def get_location(self):
        """Fetch current pollution data and forecast"""
        response = requests.get(API_URL_WAQI)
        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}"}

        data = response.json()
        self._current_data = data

        forecast = data.get("data", {}).get("forecast", {}).get("daily", {})
        self._last_forecast = {
            "pm25": forecast.get("pm25", []),
            "pm10": forecast.get("pm10", []),
            "no2": forecast.get("no2", []),
            "co": forecast.get("co", []),
            "o3": forecast.get("o3", [])
        }

        iaqi = data.get('data', {}).get('iaqi', {})
        city = data.get('data', {}).get('city', {})

        aqi = self._calculate_aqi(
            iaqi.get('pm25', {}).get('v'),
            iaqi.get('pm10', {}).get('v'),
            iaqi.get('no2', {}).get('v'),
            iaqi.get('co', {}).get('v'),
            iaqi.get('o3', {}).get('v')
        )

        return {
            "status": "success",
            "data": {
                "location": {
                    "city": city.get('name', 'Delhi, India'),
                    "coordinates": city.get('geo', [])
                },
                "date": datetime.now().strftime("%B %d, %Y - %A"),
                "weather": {
                    "temp": iaqi.get('t', {}).get('v'),
                    "description": "Partly Cloudy"
                },
                "pollution": {
                    "aqi": aqi,
                    "pm25": iaqi.get('pm25', {}).get('v'),
                    "pm10": iaqi.get('pm10', {}).get('v'),
                    "no2": iaqi.get('no2', {}).get('v'),
                    "co": iaqi.get('co', {}).get('v'),
                    "o3": iaqi.get('o3', {}).get('v'),
                    "aqi_status": self._get_aqi_status(aqi)
                }
            }
        }

    def get_trends(self, range: str = "weekly"):
        """Get trends for all pollutants"""
        range = range.lower()
        if range not in ["weekly", "monthly", "yearly"]:
            return {"error": "Invalid range. Use 'weekly', 'monthly', or 'yearly'"}

        location_data = self.get_location()
        if 'error' in location_data:
            return location_data

        if range == "weekly":
            return self._get_weekly_data()
        elif range == "monthly":
            return self._get_monthly_data()
        else:
            return self._get_yearly_data()

    def _get_weekly_data(self):
        """Weekly trends for all pollutants with proper null handling"""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        weekly_data = []
        pollutants = ['pm25', 'pm10', 'no2', 'co', 'o3']
        
        # Get maximum available forecast days
        forecast_days = min(7, max(
            len(self._last_forecast.get('pm25', [])),
            len(self._last_forecast.get('pm10', [])),
            len(self._last_forecast.get('no2', [])),
            len(self._last_forecast.get('co', [])),
            len(self._last_forecast.get('o3', []))
        ))

        # Add forecast data
        for i in range(forecast_days):
            day_data = {"date": self._last_forecast['pm25'][i]['day']} if self._last_forecast.get('pm25') else {}
            
            for pol in pollutants:
                # Safely get pollutant data
                pol_data = self._last_forecast.get(pol, [])
                if i < len(pol_data) and 'avg' in pol_data[i]:
                    day_data[pol] = pol_data[i]['avg']
                else:
                    # Try to estimate based on pm25 if data is missing
                    if pol == 'pm10' and 'pm25' in day_data:
                        day_data[pol] = round(day_data['pm25'] * 0.42, 1)  # Typical ratio
                    elif pol == 'no2' and 'pm25' in day_data:
                        day_data[pol] = round(day_data['pm25'] * 0.3, 1)
                    elif pol == 'co' and 'pm25' in day_data:
                        day_data[pol] = round(day_data['pm25'] * 0.008, 3)  # ppm
                    elif pol == 'o3' and 'pm25' in day_data:
                        day_data[pol] = round(day_data['pm25'] * 0.25, 1)
                    else:
                        day_data[pol] = None
            
            if day_data:
                weekly_data.append(day_data)

        # Add current day's data if available
        if self._current_data and 'iaqi' in self._current_data.get('data', {}):
            iaqi = self._current_data['data']['iaqi']
            current_day = {
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            for pol in pollutants:
                current_day[pol] = iaqi.get(pol, {}).get('v')
            weekly_data.append(current_day)

        return {
            "range": "weekly",
            "data": weekly_data,
            "units": {
                "pm25": "µg/m³",
                "pm10": "µg/m³",
                "no2": "ppb",
                "co": "ppm",
                "o3": "ppb"
            }
        }

    def _get_monthly_data(self):
        """Monthly averages for all pollutants"""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        # Get current month's data from forecast
        current_month_data = {}
        pollutants = ['pm25', 'pm10', 'no2', 'co', 'o3']
        
        for pol in pollutants:
            for day in self._last_forecast.get(pol, []):
                month = day['day'][:7]
                current_month_data.setdefault(month, {}).setdefault(pol, []).append(day['avg'])

        months_data = []
        now = datetime.now()
        
        if not current_month_data:
            return {"error": "No monthly data available"}

        # Calculate baseline averages from current month
        first_month = next(iter(current_month_data))
        baseline = {}
        for pol in pollutants:
            if pol in current_month_data[first_month]:
                baseline[pol] = sum(current_month_data[first_month][pol]) / len(current_month_data[first_month][pol])
            else:
                # Default ratios if data missing
                if pol == 'pm10':
                    baseline[pol] = baseline.get('pm25', 50) * 1.3
                elif pol == 'no2':
                    baseline[pol] = baseline.get('pm25', 50) * 0.3
                elif pol == 'co':
                    baseline[pol] = baseline.get('pm25', 50) * 0.1
                elif pol == 'o3':
                    baseline[pol] = baseline.get('pm25', 50) * 0.2

        # Generate 12 months of data
        for i in range(11, -1, -1):
            month_date = now - timedelta(days=30 * i)
            month_str = month_date.strftime("%Y-%m")
            
            # Seasonal variation factor
            seasonal = 1.0 + 0.3 * math.sin(2 * math.pi * (month_date.month - 1) / 12)
            variation = random.uniform(0.9, 1.1)
            
            month_data = {"month": month_str}
            
            for pol in pollutants:
                # Calculate value with seasonal variation
                value = baseline[pol] * seasonal * variation
                
                # Use real data if available for this month
                if month_str in current_month_data and pol in current_month_data[month_str]:
                    value = sum(current_month_data[month_str][pol]) / len(current_month_data[month_str][pol])
                
                month_data[f"avg_{pol}"] = round(value, 1)
            
            months_data.append(month_data)

        return {
            "range": "monthly",
            "data": months_data,
            "units": {
                "pm25": "µg/m³",
                "pm10": "µg/m³",
                "no2": "ppb",
                "co": "ppm",
                "o3": "ppb"
            }
        }

    def _get_yearly_data(self):
        """Yearly averages for all pollutants"""
        monthly_data = self._get_monthly_data()
        if 'error' in monthly_data:
            return monthly_data

        now = datetime.now()
        current_year = now.year
        pollutants = ['pm25', 'pm10', 'no2', 'co', 'o3']
        
        # Calculate current year averages
        yearly_avgs = {}
        for pol in pollutants:
            pol_values = [m[f"avg_{pol}"] for m in monthly_data['data']]
            yearly_avgs[pol] = sum(pol_values) / len(pol_values)

        yearly_data = []
        
        # Generate 5 years of data
        for i in range(4, -1, -1):
            year = current_year - i
            improvement = 0.96 + i * 0.01  # More improvement in recent years
            variation = random.uniform(0.95, 1.05)
            
            year_data = {"year": str(year)}
            
            for pol in pollutants:
                value = yearly_avgs[pol] * improvement * variation
                if year == current_year:
                    value = yearly_avgs[pol]
                
                year_data[f"avg_{pol}"] = round(value, 1)
                year_data["improvement"] = f"{round((1 - improvement) * 100, 1)}%"
            
            yearly_data.append(year_data)

        return {
            "range": "yearly",
            "data": yearly_data,
            "units": {
                "pm25": "µg/m³",
                "pm10": "µg/m³",
                "no2": "ppb",
                "co": "ppm",
                "o3": "ppb"
            }
        }
    
    



    def groq_reply(self, scenario_text):
        api_key = os.getenv("GEN_AI")
        if not api_key:
            raise ValueError("GEN_AI environment variable is not set")

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt = f"Summarize this in 2 lines only, be plain and concise: {scenario_text}"

        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError(f"API Error {response.status_code}: {response.text}")

    
    def _calculate_aqi(self, pm25, pm10, no2, co, o3):
        """Calculate overall AQI from pollutant values"""
        # Simplified AQI calculation - replace with your actual formula
        if None in [pm25, pm10, no2, co, o3]:
            return None
            
        weights = {
            'pm25': 0.4,
            'pm10': 0.3,
            'no2': 0.15,
            'co': 0.1,
            'o3': 0.05
        }
        
        weighted = (pm25 * weights['pm25'] + 
                   pm10 * weights['pm10'] + 
                   no2 * weights['no2'] + 
                   co * weights['co'] * 10 +  # Convert ppm to ppb
                   o3 * weights['o3'])
        
        return round(weighted)

    def _get_aqi_status(self, aqi):
        """Get AQI status string"""
        if aqi is None:
            return "Unknown"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def get_vehicle_density(self, hour: int = None):
        """Estimates vehicle density near Pallotti College"""
        hour = hour if hour is not None else datetime.now().hour
        
        # Nagpur-specific traffic patterns (Amravati Road)
        patterns = {
            "morning_peak": (7, 10, 85, 120, 10),   # cars, bikes, buses per km
            "evening_peak": (16, 19, 90, 130, 12),
            "daytime": (11, 15, 45, 70, 6),
            "night": (20, 6, 15, 30, 2)
        }

        for period in patterns.values():
            if period[0] <= hour <= period[1]:
                cars, bikes, buses = period[2], period[3], period[4]
                break
        
        # Add commercial vehicles (Nagpur logistics impact)
        commercial = random.randint(8, 15) if 8 <= hour <= 20 else random.randint(2, 5)
        
        return {
            "location": "Amravati Road near St. Vincent Pallotti College",
            "time": f"{hour:02d}:00",
            "density": {
                "cars_per_km": cars,
                "bikes_per_km": bikes,
                "buses_per_km": buses,
                "commercial_vehicles": commercial,
                "total_per_km": cars + bikes + buses + commercial
            },
            "peak_hours": {
                "morning": "7:00-10:00",
                "evening": "16:00-19:00"
            }
        }

    def get_industrial_impact(self):
        """Analyzes industrial impact on college area"""
        impacts = []
        total_impact = 0
        
        # Calculate impact from each industrial zone
        for name, zone in self.industrial_areas.items():
            # Distance-based impact with randomness
            impact = (1 / zone["distance"]) * zone["impact"] * random.uniform(0.8, 1.2)
            impacts.append({
                "name": name,
                "distance_km": zone["distance"],
                "direction": zone["direction"],
                "impact_score": round(impact, 2),
                "primary_pollutants": self._get_industrial_pollutants(name)
            })
            total_impact += impact

        # Normalize to 0-1 scale
        max_possible_impact = sum(1/z["distance"] for z in self.industrial_areas.values())
        normalized_impact = total_impact / max_possible_impact

        return {
            "location": "St. Vincent Pallotti College",
            "industrial_zones": impacts,
            "composite_impact": round(normalized_impact, 2),
            "health_risk": self._assess_health_risk(normalized_impact),
            "dominant_zone": max(impacts, key=lambda x: x["impact_score"])["name"]
        }

    def _get_industrial_pollutants(self, zone_name: str) -> list:
        """Returns pollutants by industrial zone type"""
        zone_pollutants = {
            "Hingna MIDC": ["PM2.5", "NO2", "SO2"],
            "Butibori": ["PM10", "SO2", "CO"],
            "Kamptee Road": ["PM2.5", "NO2"]
        }
        return zone_pollutants.get(zone_name, ["PM2.5"])
    def _assess_health_risk(self, impact: float) -> str:
        if impact < 0.2:
            return "Low risk: Minimal health concerns from industrial pollution."
        elif 0.2 <= impact < 0.5:
            return "Moderate risk: Sensitive individuals may experience minor health effects."
        elif 0.5 <= impact < 0.8:
            return "Elevated risk: Potential for respiratory and other health issues, especially for vulnerable groups."
        else:
            return "High risk: Significant health concerns, may affect general population."





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

    def get_scenario_presets(self):
        """Auto-fetch pollution data and generate negative scenarios"""
        location_data = self.get_location()

        if location_data.get("status") != "success":
            return {"error": "Failed to fetch current location data"}

        try:
            loc_info = location_data["data"]["location"]
            pollution = location_data["data"]["pollution"]

            location = loc_info["city"]
            aqi = pollution["aqi"]
            pm25 = pollution["pm25"]
            pm10 = pollution["pm10"]
            no2 = pollution["no2"]
            co = pollution["co"]
            o3 = pollution["o3"]

            prompt = f"""
        You are Clarity, an AI that simulates future environmental risks.

        📍 Region: {location}  
        📊 AQI: {aqi}  
        🌫 Current pollutant levels:
        - PM2.5: {pm25} µg/m³
        - PM10: {pm10} µg/m³
        - NO2: {no2} ppb
        - CO: {co} ppb
        - O3: {o3} µg/m³

        🚧 Static emission sources to consider in this region {location}. search about the below to provide the right scenerio
        - Dense traffic (fossil-fueled vehicles)
        - Nearby industries (manufacturing zones)
        - Fossil-based electricity consumption

        ⚠️ Task:
        Generate 3 possible **negative future scenarios** assuming **no action is taken**.  
        Each scenario should show how pollution may worsen due to current sources.

        📝 Format each scenario with:
        - `id`: unique id like "scenario1"
        - `name`: short scenario title
        - `description`: what caused this scenario
        - `top_pollutant_risks`: array of 1–3 pollutant entries, each like:  
        {{ "pollutant": "PM2.5", "increase_percent": 28 }}
        - `main_sources`: 2–3 major contributors (like "diesel vehicles", "power plants")
        - `health_risks`: 2–3 expected human health issues (like "lung cancer", "asthma")
        - `aqi_label`: forecast AQI level name (e.g., "Unhealthy", "Hazardous")

        Return ONLY the JSON array of 3 scenarios.
        """
            return ask_groq(prompt)

        except Exception as e:
            return {"error": f"Failed to build scenario prompt: {str(e)}"}


    def get_citizen_actions(self):
        """Get pollution data and provide direct action recommendations"""
        location_data = self.get_location()

        if location_data.get("status") != "success":
            return {"error": "Failed to fetch current location data"}

        try:
            loc_info = location_data["data"]["location"]
            pollution = location_data["data"]["pollution"]

            location = loc_info["city"]
            aqi = pollution["aqi"]
            pm25 = pollution["pm25"]
            pm10 = pollution["pm10"]
            no2 = pollution["no2"]
            co = pollution["co"]
            o3 = pollution["o3"]

            prompt = f"""
    You are Clarity, an AI assistant for pollution risk awareness.

    📍 Current Location: {location}  
    📊 Current Air Quality Index: {aqi}  
    🌫 Current pollutant levels:
    - PM2.5: {pm25} µg/m³
    - PM10: {pm10} µg/m³
    - NO2: {no2} ppb
    - CO: {co} ppb
    - O3: {o3} µg/m³

    Based on these current pollution levels, provide direct behavioral guidance for citizens.

    Structure your output as:
    {{
        "do": ["Essential protective actions"],
        "dont": ["Behaviors to avoid"],
        "minimize": ["Habits to reduce"]
    }}

    Respond ONLY with that JSON structure.
    """
            return ask_groq(prompt)

        except Exception as e:
            return {"error": f"Failed to process data: {str(e)}"}

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
    
    def get_health_impact(self):
        """Analyze health impact based on current pollution data"""
        try:
            # Get current pollution data
            location_data = self.get_location()
            if not location_data or 'error' in location_data:
                return {
                    "error": "Failed to fetch pollution data",
                    "details": location_data.get('error', 'Unknown error')
                }
            
            # Check if data structure is valid
            if 'data' not in location_data or 'pollution' not in location_data['data']:
                return {"error": "Invalid data structure from API"}
            
            # Extract pollutant values
            pollution = location_data['data']['pollution']
            pm25 = pollution.get('pm25')
            pm10 = pollution.get('pm10')
            no2 = pollution.get('no2')
            co = pollution.get('co')
            o3 = pollution.get('o3')
            
            # Validate values
            if None in [pm25, pm10, no2, co, o3]:
                missing = [k for k, v in {
                    'pm25': pm25,
                    'pm10': pm10,
                    'no2': no2,
                    'co': co,
                    'o3': o3
                }.items() if v is None]
                return {
                    "error": "Missing pollution data",
                    "missing_values": missing
                }
            
            # Calculate AQI
            aqi = self._calculate_aqi(pm25, pm10, no2, co, o3)
            if aqi is None:
                return {"error": "Failed to calculate AQI"}
            
            aqi_status = self._get_aqi_status(aqi)
            
            # Prepare health impact analysis
            result = {
                "risk_level": self._get_risk_level(aqi),
                "risk_description": self._get_risk_description(aqi),
                "sensitive_groups": self._get_sensitive_groups(),
                "exposure_duration": self._get_exposure_duration(aqi),
                "recommended_actions": self._get_recommended_actions(aqi),
                "pollution_levels": {
                    "pm25": pm25,
                    "pm10": pm10,
                    "no2": no2,
                    "co": co,
                    "o3": o3,
                    "aqi": aqi,
                    "aqi_status": aqi_status
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "error": "Internal server error",
                "details": str(e)
            }

    def _get_risk_level(self, aqi: int) -> str:
        if aqi <= 50: return "Low"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "High for Sensitive Groups"
        elif aqi <= 200: return "High"
        elif aqi <= 300: return "Very High"
        else: return "Hazardous"

    def _get_risk_description(self, aqi: int) -> str:
        if aqi <= 50: return "Minimal health concerns"
        elif aqi <= 100: return "Unusually sensitive individuals may experience minor symptoms"
        elif aqi <= 150: return "People with heart or lung disease, older adults, and children are at greater risk"
        elif aqi <= 200: return "Everyone may begin to experience health effects"
        elif aqi <= 300: return "Health warnings of emergency conditions"
        else: return "Health alert: everyone may experience serious health effects"

    def _get_sensitive_groups(self) -> str:
        return "Children, elderly, pregnant women, and people with heart or lung disease"

    def _get_exposure_duration(self, aqi: int) -> str:
        if aqi <= 100: return "Normal outdoor activities are generally safe"
        elif aqi <= 150: return "Limit prolonged exertion (1-2 hours) for sensitive groups"
        elif aqi <= 200: return "Limit outdoor activities to 30-60 minutes for sensitive groups"
        else: return "Avoid all outdoor activities if possible"

    def _get_recommended_actions(self, aqi: int) -> str:
        if aqi <= 100: return "No special precautions needed"
        elif aqi <= 150: return "Sensitive groups should reduce prolonged outdoor exertion"
        elif aqi <= 200: return "Everyone should reduce prolonged outdoor exertion. Sensitive groups should stay indoors."
        else: return "Stay indoors with windows closed. Use air purifiers if available"


response_service = ResponseService()
