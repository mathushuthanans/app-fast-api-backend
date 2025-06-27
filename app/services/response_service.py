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
        self.router = router
        self.router.add_api_route("/get_location", self.get_location, methods=["GET"])
        self.router.add_api_route("/predict_health_risk", self.predict_health_risk, methods=["POST"])
        self.router.add_api_route("/predict_region_class", self.predict_region_class, methods=["POST"])
        self.router.add_api_route("/bridge_predict", self.bridge_predict, methods=["GET"])
        self.college_coords = (21.1199, 79.0196)  # St. Vincent Pallotti College
        self.industrial_areas = {
            "Hingna MIDC": {"distance": 5.2, "direction": "NE", "impact": 0.7},
            "Butibori": {"distance": 12.5, "direction": "SW", "impact": 0.9},
            "Kamptee Road": {"distance": 3.8, "direction": "E", "impact": 0.4}
        }
        self._current_data = None
        self._last_forecast = None

# get_trends 
    def get_trends(self, range: str):
        """Returns pollution trends for the requested time range"""
        range = range.lower()
        if range not in ["weekly", "monthly", "yearly"]:
            return {"error": "Invalid range. Use 'weekly', 'monthly', or 'yearly'"}

        # First ensure we have fresh data
        self.get_location()  # This populates self._last_forecast

        if range == "week":
            return self._get_weekly_data()
        elif range == "month":
            return self._get_monthly_data()
        else:
            return self._get_yearly_data()

    def _get_weekly_data(self):
        """Returns 7 days of PM2.5/PM10 data with current day's full pollutants"""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        weekly_data = []

        # 1. Add forecast data (PM2.5 and PM10 only)
        for i in range(min(7, len(self._last_forecast.get('pm25', [])))):
            day_data = {
                "date": self._last_forecast['pm25'][i]['day'],
                "pm2_5": self._last_forecast['pm25'][i]['avg'],
                "pm10": self._last_forecast['pm10'][i]['avg'] if i < len(self._last_forecast.get('pm10', [])) else None,
                "co": None,
                "no2": None,
                "o3": None
            }
            weekly_data.append(day_data)

        # 2. Add current day's complete data if available
        if self._current_data and 'iaqi' in self._current_data.get('data', {}):
            current = self._current_data['data']['iaqi']
            weekly_data.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "pm2_5": current.get('pm25', {}).get('v'),
                "pm10": current.get('pm10', {}).get('v'),
                "co": current.get('co', {}).get('v'),
                "no2": current.get('no2', {}).get('v'),
                "o3": current.get('o3', {}).get('v')
            })

        return {
            "range": "weekly",
            "data": weekly_data,
            "units": {
                "pm2_5": "µg/m³",
                "pm10": "µg/m³",
                "co": "ppm",
                "no2": "ppb",
                "o3": "ppb"
            }
        }

    def _get_monthly_data(self):
        """Returns 6-12 months of historical PM2.5/PM10 data with realistic patterns"""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        # Get current month's real data from forecast
        current_month_data = {}
        for day in self._last_forecast.get('pm25', []):
            month = day['day'][:7]
            if month not in current_month_data:
                current_month_data[month] = {"pm2_5": [], "pm10": []}
            current_month_data[month]["pm2_5"].append(day['avg'])

        for day in self._last_forecast.get('pm10', []):
            month = day['day'][:7]
            if month in current_month_data:
                current_month_data[month]["pm10"].append(day['avg'])

        # Generate realistic historical data for past months
        months_data = []
        current_date = datetime.now()
        
        # Use current month's average as baseline
        current_pm25_avg = sum(current_month_data[next(iter(current_month_data))]["pm2_5"]) / \
                        len(current_month_data[next(iter(current_month_data))]["pm2_5"]) if current_month_data else 50
        current_pm10_avg = sum(current_month_data[next(iter(current_month_data))]["pm10"]) / \
                        len(current_month_data[next(iter(current_month_data))]["pm10"]) if current_month_data and \
                        current_month_data[next(iter(current_month_data))]["pm10"] else current_pm25_avg * 1.3

        # Generate 11 previous months (total 12 months)
        for month_offset in range(11, -1, -1):
            target_date = current_date - timedelta(days=30*month_offset)
            month_str = target_date.strftime("%Y-%m")
            
            # Apply seasonal variation (higher in winter, lower in summer)
            seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (target_date.month - 1) / 12)
            
            # Add some random monthly variation
            monthly_variation = random.uniform(0.9, 1.1)
            
            # Calculate values with realistic patterns
            pm25_avg = current_pm25_avg * seasonal_factor * monthly_variation
            pm10_avg = current_pm10_avg * seasonal_factor * monthly_variation
            
            # For the current month, use real data if available
            if month_str in current_month_data:
                pm25_avg = sum(current_month_data[month_str]["pm2_5"]) / len(current_month_data[month_str]["pm2_5"])
                if current_month_data[month_str]["pm10"]:
                    pm10_avg = sum(current_month_data[month_str]["pm10"]) / len(current_month_data[month_str]["pm10"])
            
            months_data.append({
                "month": month_str,
                "avg_pm2_5": round(pm25_avg, 1),
                "avg_pm10": round(pm10_avg, 1)
            })

        return {
            "range": "monthly",
            "data": months_data,
            "units": {
                "pm2_5": "µg/m³",
                "pm10": "µg/m³"
            }
        }

    def _get_yearly_data(self):
        """Returns 3-5 years of historical PM2.5/PM10 data with realistic trends"""
        # First get the monthly data (which now includes 12 months)
        monthly_data = self._get_monthly_data().get('data', [])
        if not monthly_data:
            return {"error": "No monthly data available"}

        # Calculate current year's averages from monthly data
        current_year = datetime.now().year
        current_year_data = {
            "pm2_5": [m['avg_pm2_5'] for m in monthly_data],
            "pm10": [m['avg_pm10'] for m in monthly_data if m['avg_pm10'] is not None]
        }

        # Generate realistic historical data for previous years
        yearly_data = []
        
        # Use current year's average as baseline
        current_pm25_avg = sum(current_year_data["pm2_5"]) / len(current_year_data["pm2_5"])
        current_pm10_avg = sum(current_year_data["pm10"]) / len(current_year_data["pm10"]) if current_year_data["pm10"] else current_pm25_avg * 1.35

        # Generate 5 years of data (current year + 4 previous years)
        for year_offset in range(4, -1, -1):
            target_year = current_year - year_offset
            
            # Apply yearly improvement trend (2-5% reduction per year)
            improvement_factor = (0.96 + year_offset * 0.01)  # More improvement in recent years
            
            # Add some random yearly variation
            yearly_variation = random.uniform(0.95, 1.05)
            
            # Calculate values with realistic patterns
            pm25_avg = current_pm25_avg * improvement_factor * yearly_variation
            pm10_avg = current_pm10_avg * improvement_factor * yearly_variation
            
            # For the current year, use real data if available
            if target_year == current_year:
                pm25_avg = current_pm25_avg
                pm10_avg = current_pm10_avg
            
            yearly_data.append({
                "year": str(target_year),
                "avg_pm2_5": round(pm25_avg, 1),
                "avg_pm10": round(pm10_avg, 1),
                "improvement": f"{round((1 - improvement_factor) * 100, 1)}%"
            })

        return {
            "range": "yearly",
            "data": yearly_data,
            "units": {
                "pm2_5": "µg/m³",
                "pm10": "µg/m³"
            },
            "note": "Historical data includes simulated values with 2-5% annual improvement trend"
        }

#get_location
    def get_location(self):
        """Returns formatted data matching the frontend design"""
        response = requests.get(API_URL_WAQI)
        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}"}

        data = response.json()
        iaqi = data.get('data', {}).get('iaqi', {})
        city = data.get('data', {}).get('city', {})
        
        # Get raw values
        pm25 = iaqi.get('pm25', {}).get('v')
        pm10 = iaqi.get('pm10', {}).get('v')
        no2 = iaqi.get('no2', {}).get('v')
        co = iaqi.get('co', {}).get('v')  
        o3 = iaqi.get('o3', {}).get('v')
        temp = iaqi.get('t', {}).get('v')  # Temperature
        
        # Calculate AQI values
        from ml_models.utils import (
            calculate_overall_aqi
        )
        
        aqi = calculate_overall_aqi(pm25, pm10, no2, co, o3)  # Exclude CO if not needed
        
        # Get weather description (simple mapping)
        weather_desc = "Partly Cloudy"  # Default, replace with real data if available
        
        # Format date
        from datetime import datetime
        date_str = datetime.now().strftime("%B %d, %Y - %A")
        
        return {
            "status": "success",
            "data": {
                "location": {
                    "city": city.get('name', 'Delhi, India'),  # Default to Delhi if not available
                    "coordinates": city.get('geo', [])
                },
                "date": date_str,
                "weather": {
                    "temp": temp,
                    "description": weather_desc
                },
                "pollution": {
                    "aqi": aqi,
                    "pm25": pm25,
                    "pm10": pm10,
                    "no2": no2,
                    "co" : co,
                    "o3": o3,
                    "aqi_status": self._get_aqi_status(aqi)  # "Unhealthy", etc.
                }
            }
        }

    def _get_aqi_status(self, aqi):
        """Returns AQI status string"""
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

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
        """Returns predefined what-if scenarios using existing methods"""

        current_location = self.get_location()
        current_density = self.get_vehicle_density()
        current_industry = self.get_industrial_impact()

        prompt = f"""
    You are Clarity, analyzing pollution scenarios for Nagpur's St. Vincent Pallotti College.

    Current baseline data:
    - Pollution: {current_location['data']['pollution']}
    - Vehicle density: {current_density}
    - Industrial impact: {current_industry}

    Generate 3 what-if scenarios in **valid JSON only** — do not include Markdown, explanations, or formatting.

    Return JSON in this exact structure:
    {{
    "scenarios": [
        {{
        "id": "scenario1",
        "name": "Scenario Name",
        "description": "1-sentence description",
        "vehicle_changes": {{
            "cars_per_km": {{"current": X, "projected": Y}},
            "bikes_per_km": {{"current": X, "projected": Y}},
            "commercial_vehicles": {{"current": X, "projected": Y}}
        }},
        "industrial_changes": {{
            "Hingna_MIDC": {{"current_impact": X, "projected_impact": Y}},
            "Butibori": {{"current_impact": X, "projected_impact": Y}}
        }},
        "pollution_projections": {{
            "pm25": {{"current": X, "projected": Y}},
            "no2": {{"current": X, "projected": Y}},
            "co": {{"current": X, "projected": Y}}
        }},
        "health_benefits": ["Benefit 1", "Benefit 2"]
        }}
    ]
    }}

    Use this baseline:
    - Current PM2.5: {current_location['data']['pollution']['pm25']}
    - Current NO2: {current_location['data']['pollution']['no2']}
    - Current vehicle density: {current_density['density']['total_per_km']}
    - Current industrial impact: {current_industry['composite_impact']}

    ONLY return the valid JSON object. No ``` marks. No explanation. No intro or outro.
    """

        return ask_groq(prompt)




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
