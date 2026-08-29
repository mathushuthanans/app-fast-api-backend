import math
from fastapi import APIRouter, HTTPException
import requests
import joblib
import numpy as np
import os
from app.models.datamodel1 import HealthRiskRequest, RegionClassRequest, IndustrialZone
from app.services import extract_locations  # Fixed import path
from app.config import API_KEY_GEMINI, OPENWEATHER_API_KEY, OPENWEATHER_CURRENT_AIR_URL, OPENWEATHER_CURRENT_WEATHER_URL, OPENWEATHER_FORECAST_AIR_URL, OPENWEATHER_REVERSE_GEO_URL
from ml_models.ai_logic import ask_gemini
from ml_models.utils import calculate_overall_aqi as calc
from datetime import datetime, timedelta
import random
import json
from typing import Any, TypedDict, cast, Literal, TypeAlias




router = APIRouter()

MODEL_DIR = extract_locations.extract()
health_model = joblib.load(os.path.join(MODEL_DIR, "health_risk_model.pkl"))
region_model = joblib.load(os.path.join(MODEL_DIR, "region_model.pkl"))



class ResponseService:
    def __init__(self):
        router.add_api_route("/get_location", self.get_location, methods=["GET"])
        router.add_api_route("/get_trends", self.get_trends, methods=["GET"])

        self._current_data = None
        self._last_forecast = None

    def get_location(self, lat: float, lon: float) -> dict[str, Any]:
        """Fetch current air pollution and forecast for supplied coordinates."""
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
        }

        try:
            current_response = requests.get(
                OPENWEATHER_CURRENT_AIR_URL,
                params=params,
                timeout=10,
            )
            forecast_response = requests.get(
                OPENWEATHER_FORECAST_AIR_URL,
                params=params,
                timeout=10,
            )
            weather_response = requests.get(
                OPENWEATHER_CURRENT_WEATHER_URL,
                params={**params, "units": "metric"},
                timeout=10,
            )
            geocode_response = requests.get(
                OPENWEATHER_REVERSE_GEO_URL,
                params={**params, "limit": 1},
                timeout=10,
            )

            if current_response.status_code != 200:
                return {
                    "error": (
                        f"OpenWeather current-air request failed: "
                        f"{current_response.status_code}"
                    )
                }

            if forecast_response.status_code != 200:
                return {
                    "error": (
                        f"OpenWeather forecast request failed: "
                        f"{forecast_response.status_code}"
                    )
                }

            current_data = current_response.json()
            forecast_data = forecast_response.json()

            current_list = current_data.get("list", [])
            forecast_list = forecast_data.get("list", [])

            if not current_list:
                return {"error": "OpenWeather returned no current air-pollution data"}

            current_entry = current_list[0]
            components = current_entry.get("components", {})
            openweather_aqi = current_entry.get("main", {}).get("aqi")

            pm25 = float(components.get("pm2_5", 0))
            pm10 = float(components.get("pm10", 0))
            no2 = float(components.get("no2", 0))
            co = float(components.get("co", 0))
            o3 = float(components.get("o3", 0))

            # Keep your existing app AQI calculation instead of OpenWeather's 1–5 AQI.
            aqi = self._calculate_aqi(pm25, pm10, no2, co, o3)

            # Convert OpenWeather's hourly forecast into your existing
            # [{"day": "YYYY-MM-DD", "avg": number}] trend format.
            daily_values: dict[str, dict[str, list[float]]] = {}

            pollutant_map = {
                "pm25": "pm2_5",
                "pm10": "pm10",
                "no2": "no2",
                "co": "co",
                "o3": "o3",
            }

            for entry in forecast_list:
                date = datetime.fromtimestamp(
                    entry["dt"]
                ).strftime("%Y-%m-%d")

                daily_values.setdefault(
                    date,
                    {pollutant: [] for pollutant in pollutant_map},
                )

                forecast_components = entry.get("components", {})

                for pollutant, openweather_name in pollutant_map.items():
                    value = forecast_components.get(openweather_name)

                    if value is not None:
                        daily_values[date][pollutant].append(float(value))

            self._last_forecast = {
                pollutant: [
                    {
                        "day": date,
                        "avg": round(
                            sum(values[pollutant]) / len(values[pollutant]),
                            2,
                        ),
                    }
                    for date, values in sorted(daily_values.items())
                    if values[pollutant]
                ]
                for pollutant in pollutant_map
            }

            self._current_data = current_data

            city = f"{lat:.4f}, {lon:.4f}"

            if geocode_response.status_code == 200:
                geocode_data = geocode_response.json()

                if geocode_data:
                    place = geocode_data[0]
                    city = ", ".join(
                        part
                        for part in [
                            place.get("name"),
                            place.get("state"),
                            place.get("country"),
                        ]
                        if part
                    )

            temperature = None
            weather_description = "Unavailable"

            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                temperature = weather_data.get("main", {}).get("temp")

                weather_items = weather_data.get("weather", [])
                if weather_items:
                    weather_description = weather_items[0].get(
                        "description",
                        "Unavailable",
                    )

            return {
                "status": "success",
                "data": {
                    "location": {
                        "city": city,
                        "coordinates": [lat, lon],
                    },
                    "date": datetime.now().strftime("%B %d, %Y - %A"),
                    "weather": {
                        "temp": temperature,
                        "description": weather_description,
                    },
                    "pollution": {
                        "aqi": aqi,
                        "pm25": pm25,
                        "pm10": pm10,
                        "no2": no2,
                        "co": co,
                        "o3": o3,
                        "aqi_status": self._get_aqi_status(aqi),
                        "openweather_aqi": openweather_aqi,
                    },
                },
            }

        except requests.RequestException as error:
            return {"error": f"OpenWeather request failed: {error}"}
        except (KeyError, TypeError, ValueError) as error:
            return {"error": f"Invalid OpenWeather response: {error}"}

    def get_trends(self, lat: float, lon: float, range: str = "weekly"):
        """Get trends for all pollutants"""
        range = range.lower()
        if range not in ["weekly", "monthly", "yearly"]:
            return {"error": "Invalid range. Use 'weekly', 'monthly', or 'yearly'"}

        location_data = self.get_location(lat, lon)
        if 'error' in location_data:
            return location_data

        if range == "weekly":
            return self._get_weekly_data()
        elif range == "monthly":
            return self._get_monthly_data()
        else:
            return self._get_yearly_data()

    def _get_weekly_data(self) -> dict[str, Any]:
        """Return available daily averages from the OpenWeather forecast."""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        pollutants = ["pm25", "pm10", "no2", "co", "o3"]
        data_by_date: dict[str, dict[str, str | float | None]] = {}

        for pollutant in pollutants:
            for forecast_day in self._last_forecast.get(pollutant, []):
                date = str(forecast_day["day"])

                data_by_date.setdefault(date, {"date": date})
                data_by_date[date][pollutant] = float(forecast_day["avg"])

        weekly_data = [
            data_by_date[date]
            for date in sorted(data_by_date)
        ]

        return {
            "range": "weekly",
            "data": weekly_data,
            "units": {
                "pm25": "µg/m³",
                "pm10": "µg/m³",
                "no2": "µg/m³",
                "co": "µg/m³",
                "o3": "µg/m³",
            },
            "note": (
                "OpenWeather provides an hourly air-pollution forecast "
                "for up to four days."
            ),
        }
    def _get_monthly_data(self):
        """Monthly averages for all pollutants."""
        if not self._last_forecast:
            return {"error": "No forecast data available"}

        current_month_data: dict[str, dict[str, list[float]]] = {}
        pollutants = ["pm25", "pm10", "no2", "co", "o3"]

        for pol in pollutants:
            for day in self._last_forecast.get(pol, []):
                month = str(day["day"])[:7]
                current_month_data.setdefault(month, {}).setdefault(pol, []).append(
                    float(day["avg"])
                )

        if not current_month_data:
            return {"error": "No monthly data available"}

        now = datetime.now()

        # Calculate baseline averages from the first available forecast month.
        first_month = next(iter(current_month_data))
        baseline: dict[str, float] = {}

        for pol in pollutants:
            if pol in current_month_data[first_month]:
                values = current_month_data[first_month][pol]
                baseline[pol] = sum(values) / len(values)
            elif pol == "pm10":
                baseline[pol] = baseline.get("pm25", 50.0) * 1.3
            elif pol == "no2":
                baseline[pol] = baseline.get("pm25", 50.0) * 0.3
            elif pol == "co":
                baseline[pol] = baseline.get("pm25", 50.0) * 0.1
            elif pol == "o3":
                baseline[pol] = baseline.get("pm25", 50.0) * 0.2
            else:
                baseline[pol] = 50.0

        months_data: list[dict[str, str | float]] = []

        for i in range(11, -1, -1):
            month_date = now - timedelta(days=30 * i)
            month_str = month_date.strftime("%Y-%m")

            seasonal = 1.0 + 0.3 * math.sin(
                2 * math.pi * (month_date.month - 1) / 12
            )
            variation = random.uniform(0.9, 1.1)

            month_data: dict[str, str | float] = {"month": month_str}

            for pol in pollutants:
                value = baseline[pol] * seasonal * variation

                if (
                    month_str in current_month_data
                    and pol in current_month_data[month_str]
                ):
                    values = current_month_data[month_str][pol]
                    value = sum(values) / len(values)

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
                "o3": "ppb",
            },
        }
        from typing import Any


    def _get_yearly_data(self) -> dict[str, Any]:
        """Yearly averages for all pollutants."""
        monthly_data = self._get_monthly_data()

        if "error" in monthly_data:
            return monthly_data

        now = datetime.now()
        current_year = now.year
        pollutants = ["pm25", "pm10", "no2", "co", "o3"]

        # Calculate current-year averages
        monthly_records = cast(
            list[dict[str, str | float]],
            monthly_data["data"],
        )

        yearly_avgs: dict[str, float] = {}

        for pol in pollutants:
            pol_values = [
                float(month[f"avg_{pol}"])
                for month in monthly_records
            ]
            yearly_avgs[pol] = sum(pol_values) / len(pol_values)

        yearly_data: list[dict[str, str | float]] = []

        # Generate 5 years of data
        for i in range(4, -1, -1):
            year = current_year - i

            improvement = 0.96 + i * 0.01
            variation = random.uniform(0.95, 1.05)

            year_data: dict[str, str | float] = {"year": str(year)}

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
                "o3": "ppb",
            },
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


    def _calculate_aqi(
        self,
        pm25: float | None,
        pm10: float | None,
        no2: float | None,
        co: float | None,
        o3: float | None,
    ) -> int | None:
        """
        Calculate an approximate 0–500 AQI from OpenWeather concentrations.

        OpenWeather returns all pollutant components in µg/m³.
        AQI breakpoints for CO, NO2, and O3 require converted units.
        """

        # Make sure all pollutant values are available.
        if (
            pm25 is None
            or pm10 is None
            or no2 is None
            or co is None
            or o3 is None
        ):
            return None

        # Explicitly convert to float so Pylance knows these are not None.
        pm25_value = float(pm25)
        pm10_value = float(pm10)
        no2_value = float(no2)
        co_value = float(co)
        o3_value = float(o3)

        def calculate_sub_index(
            concentration: float,
            breakpoints: list[tuple[float, float, int, int]],
        ) -> float:
            for low_c, high_c, low_i, high_i in breakpoints:
                if low_c <= concentration <= high_c:
                    return (
                        (high_i - low_i)
                        / (high_c - low_c)
                        * (concentration - low_c)
                        + low_i
                    )

            return 500.0 if concentration > breakpoints[-1][1] else 0.0

        # PM2.5 and PM10 are already in µg/m³.
        pm25_aqi = calculate_sub_index(
            pm25_value,
            [
                (0.0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500),
            ],
        )

        pm10_aqi = calculate_sub_index(
            pm10_value,
            [
                (0.0, 54.0, 0, 50),
                (55.0, 154.0, 51, 100),
                (155.0, 254.0, 101, 150),
                (255.0, 354.0, 151, 200),
                (355.0, 424.0, 201, 300),
                (425.0, 504.0, 301, 400),
                (505.0, 604.0, 401, 500),
            ],
        )

        # OpenWeather µg/m³ → AQI breakpoint units.
        no2_ppb = no2_value / 1.88
        o3_ppb = o3_value / 1.96
        co_ppm = co_value / 1145.0

        no2_aqi = calculate_sub_index(
            no2_ppb,
            [
                (0.0, 53.0, 0, 50),
                (54.0, 100.0, 51, 100),
                (101.0, 360.0, 101, 150),
                (361.0, 649.0, 151, 200),
                (650.0, 1249.0, 201, 300),
                (1250.0, 1649.0, 301, 400),
                (1650.0, 2049.0, 401, 500),
            ],
        )

        o3_aqi = calculate_sub_index(
            o3_ppb,
            [
                (0.0, 54.0, 0, 50),
                (55.0, 70.0, 51, 100),
                (71.0, 85.0, 101, 150),
                (86.0, 105.0, 151, 200),
                (106.0, 200.0, 201, 300),
            ],
        )

        co_aqi = calculate_sub_index(
            co_ppm,
            [
                (0.0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500),
            ],
        )

        return round(
            max(
                pm25_aqi,
                pm10_aqi,
                no2_aqi,
                o3_aqi,
                co_aqi,
            )
        )

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

    def get_vehicle_density(self, hour: int | None = None):
        """Estimate vehicle density near Pallotti College."""
        if hour is None:
            hour = datetime.now().hour

        if not 0 <= hour <= 23:
            raise ValueError("hour must be between 0 and 23")

        if 7 <= hour < 10:
            cars, bikes, buses = 85, 120, 10
            period_name = "morning_peak"
        elif 16 <= hour < 19:
            cars, bikes, buses = 90, 130, 12
            period_name = "evening_peak"
        elif 10 <= hour < 16:
            cars, bikes, buses = 45, 70, 6
            period_name = "daytime"
        else:
            cars, bikes, buses = 15, 30, 2
            period_name = "night_or_off_peak"

        commercial = (
            random.randint(8, 15)
            if 8 <= hour <= 20
            else random.randint(2, 5)
        )

        return {
            "location": "Amravati Road near St. Vincent Pallotti College",
            "time": f"{hour:02d}:00",
            "traffic_period": period_name,
            "density": {
                "cars_per_km": cars,
                "bikes_per_km": bikes,
                "buses_per_km": buses,
                "commercial_vehicles": commercial,
                "total_per_km": cars + bikes + buses + commercial,
            },
            "peak_hours": {
                "morning": "7:00-10:00",
                "evening": "16:00-19:00",
            },
        }

    def get_industrial_impact(self, lat: float, lon: float) -> dict[str, Any]:
        """Estimate emission-source impact for the current WAQI station."""
        location_data = cast(dict[str, Any], self.get_location(lat, lon))

        if location_data.get("status") != "success":
            return {"error": "Failed to fetch current location data"}

        location = location_data["data"]["location"]
        pollution = location_data["data"]["pollution"]

        aqi = float(pollution.get("aqi") or 0)
        pollution_factor = min(max(aqi / 300, 0.0), 1.0)

        source_profiles: list[tuple[str, float]] = [
            ("Industrial and manufacturing activity", 0.45),
            ("Freight and commercial transport", 0.35),
            ("Construction and road dust", 0.20),
        ]

        impacts: list[dict[str, Any]] = []

        for source_name, base_weight in source_profiles:
            impact_score = base_weight * (0.5 + pollution_factor)

            impacts.append({
                "name": source_name,
                "impact_score": round(impact_score, 2),
                "primary_pollutants": self._get_industrial_pollutants(source_name),
                "source": "AQI-based estimate",
            })

        composite_impact = sum(
            float(item["impact_score"])
            for item in impacts
        )

        dominant_source = max(
            impacts,
            key=lambda item: float(item["impact_score"]),
        )["name"]

        return {
            "location": location.get("city", "Unknown location"),
            "coordinates": location.get("coordinates", []),
            "industrial_sources": impacts,
            "composite_impact": round(composite_impact, 2),
            "health_risk": self._assess_health_risk(composite_impact),
            "dominant_source": dominant_source,
            "note": (
                "This is an AQI-based estimate for the configured WAQI station. "
                "It does not identify verified nearby industrial facilities."
            ),
        }

    def _assess_health_risk(self, impact: float) -> str:
        """Convert an estimated impact score into a simple health-risk label."""
        if impact < 0.2:
            return "Low"
        if impact < 0.5:
            return "Moderate"
        if impact < 0.8:
            return "Elevated"
        return "High"


    def _get_industrial_pollutants(self, source_name: str) -> list[str]:
        """Return expected pollutants for each general emission source."""
        source_pollutants = {
            "Industrial and manufacturing activity": ["PM2.5", "NO2", "SO2"],
            "Freight and commercial transport": ["PM2.5", "NO2", "CO"],
            "Construction and road dust": ["PM10", "PM2.5"],
        }

        return source_pollutants.get(source_name, ["PM2.5"])




    def predict_health_risk(self, data: HealthRiskRequest):
        features = np.array([[data.pm25, data.pm10, data.no2, data.o3, data.co, data.asthma, data.heart_disease]])
        danger_scale = health_model.predict(features)[0]
        return {"danger_scale": danger_scale}

    def predict_region_class(self, data: RegionClassRequest):
        features = np.array([[data.pm25, data.pm10, data.no2, data.o3, data.co]])
        region_class = region_model.predict(features)[0]
        return {"region_class": region_class}

    def bridge_predict(
        self,
        lat: float,
        lon: float,
        asthma: int = 0,
        heart_disease: int = 0,
    ) -> dict[str, Any]:
        """Predict health risk and region class from current OpenWeather data."""
        location_data = self.get_location(lat, lon)

        if "error" in location_data:
            return location_data

        location = location_data["data"]["location"]
        pollution = location_data["data"]["pollution"]

        pm25 = pollution.get("pm25")
        pm10 = pollution.get("pm10")
        no2 = pollution.get("no2")
        o3 = pollution.get("o3")
        co = pollution.get("co")

        if None in [pm25, pm10, no2, o3, co]:
            return {"error": "Incomplete OpenWeather pollution data"}

        health_features = np.array([
            [pm25, pm10, no2, o3, co, asthma, heart_disease]
        ])
        danger_scale = int(health_model.predict(health_features)[0])

        region_features = np.array([
            [pm25, pm10, no2, o3, co]
        ])
        region_class = int(region_model.predict(region_features)[0])

        return {
            "location": location,
            "aqi_values": {
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "o3": o3,
                "co": co,
            },
            "predictions": {
                "danger_scale": danger_scale,
                "region_class": region_class,
            },
        }

    def get_scenario_presets(self, lat: float, lon: float):
        """Generate estimated no-action pollution scenarios for the current WAQI station."""
        location_data = cast(dict[str, Any], self.get_location(lat, lon))

        if location_data.get("status") != "success":
            return {"error": "Failed to fetch current location data"}

        try:
            data = cast(dict[str, Any], location_data["data"])
            location = str(data["location"]["city"])
            pollution = cast(dict[str, Any], data["pollution"])

            aqi = float(pollution["aqi"])
            pm25 = float(pollution["pm25"])
            pm10 = float(pollution["pm10"])
            no2 = float(pollution["no2"])
            co = float(pollution["co"])
            o3 = float(pollution["o3"])

            prompt = f""" You are Clarity, an AI that simulates future environmental risks.
                    Region: {location}
                    Current AQI: {aqi}

                    Current pollutant levels:
                    - PM2.5: {pm25} µg/m³
                    - PM10: {pm10} µg/m³
                    - NO2: {no2} ppb
                    - CO: {co} ppb
                    - O3: {o3} µg/m³

                    Generate exactly 3 possible negative future scenarios, assuming no action is taken.

                    For each scenario return:
                    - id: unique ID such as "scenario1"
                    - name: short title
                    - description: what caused it
                    - top_pollutant_risks: array containing 1 to 3 items:
                    {{"pollutant": "PM2.5", "increase_percent": 28}}
                    - main_sources: array of 2 to 3 sources
                    - health_risks: array of 2 to 3 health risks
                    - aqi_label: for example "Unhealthy" or "Hazardous"

                    Return only a valid JSON array.
                    """

            return ask_gemini(prompt)
        except (KeyError, TypeError, ValueError) as error:
            return {"error": f"Invalid pollution data: {error}"}
        except Exception as error:
            return {"error": f"Failed to generate scenarios: {error}"}




    def get_citizen_actions(self, lat: float, lon: float):
        location_data = self.get_location(lat, lon)

        if location_data.get("status") != "success":
            return {"error": "Failed to fetch current location data"}

        try:
            response_data = cast(dict[str, Any], location_data["data"])
            loc_info = cast(dict[str, Any], response_data["location"])
            pollution = cast(dict[str, Any], response_data["pollution"])

            location = str(loc_info.get("city", "Unknown location"))
            aqi = pollution.get("aqi")
            pm25 = pollution.get("pm25")
            pm10 = pollution.get("pm10")
            no2 = pollution.get("no2")
            co = pollution.get("co")
            o3 = pollution.get("o3")

            if None in [aqi, pm25, pm10, no2, co, o3]:
                return {
                    "error": "Incomplete pollution data received",
                    "location": location,
                }

            prompt = f"""
    You are Clarity, an AI assistant for pollution-risk awareness.

    Current location: {location}
    Current AQI: {aqi}

    Current pollutant levels:
    - PM2.5: {pm25} µg/m³
    - PM10: {pm10} µg/m³
    - NO2: {no2} ppb
    - CO: {co} ppb
    - O3: {o3} µg/m³

    Give practical guidance for citizens based on these current levels.

    Return ONLY valid JSON in this format:
    {{
      "do": ["Essential protective actions"],
      "dont": ["Behaviors to avoid"],
      "minimize": ["Habits to reduce"]
    }}
    """

            return ask_gemini(prompt)

        except (KeyError, TypeError, ValueError) as error:
            return {"error": f"Invalid pollution-data structure: {error}"}
        except Exception as error:
            return {"error": f"Failed to generate citizen actions: {error}"}


    def compare_locations(self, location1: str, location2: str, pm25_1: float, pm10_1: float, pm25_2: float, pm10_2: float):
        prompt = f"""
    You are Clarity.
    Compare pollution health impact between:
    - {location1}: PM2.5={pm25_1}, PM10={pm10_1}
    - {location2}: PM2.5={pm25_2}, PM10={pm10_2}
    Give 2-line difference summary.
    Return as {{"comparison": "..."}}
    """
        return ask_gemini(prompt)


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
        return ask_gemini(prompt)



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
        return ask_gemini(prompt) if not debug_policy_suggestions else ["Plant trees", "Ban diesel", "Promote cycling"]


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
        return ask_gemini(prompt)


    def daily_tip(self):
        prompt = """
    You are Clarity, an AI assistant that gives short, actionable tips or facts about pollution.

    Give one daily pollution-related tip or fact that is:
    - Easy to understand
    - Practical or educational
    - No more than 2 sentences

    Respond as plain text.
    """
        result = ask_gemini(prompt)
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
        return ask_gemini(prompt)


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
        return ask_gemini(prompt)

    def get_health_impact(self, lat: float, lon: float):

        """Analyze health impact based on current pollution data"""
        try:
            # Get current pollution data
            location_data = self.get_location(lat, lon)
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
    def help_info(self) -> dict[str, str]:
            return {
                "help": (
                    "Clarity explains pollution data and health risks, suggests "
                    "actions, predicts policy impacts, and creates scenarios."
                )
            }


    def explain_pollution(self, pollutant: str) -> Any:
            prompt = f"""
        Explain the pollutant "{pollutant}" in simple words.

        Return only valid JSON:
        {{
          "object": "What is {pollutant}?",
          "causes": ["Cause 1", "Cause 2", "Cause 3"],
          "effects": ["Effect 1", "Effect 2", "Effect 3"]
        }}
        """

            return ask_gemini(prompt)


    def predict_policy(
            self,
            policy: str,
            location: str,
            pm25: float,
            pm10: float,
            no2: float,
            co: float,
            o3: float,
        ) -> Any:
            current_aqi = calc(pm25, pm10, no2, co, o3)

            prompt = f"""
        You are Clarity, an AI assistant for pollution-policy analysis.

        Policy: {policy}
        Location: {location}
        Current AQI: {current_aqi}

        Current pollutant levels:
        - PM2.5: {pm25}
        - PM10: {pm10}
        - NO2: {no2}
        - CO: {co}
        - O3: {o3}

        Estimate the likely impact of the policy.

        Return only valid JSON:
        {{
          "effects_of_policy": ["Effect 1", "Effect 2", "Effect 3"],
          "old_aqi": {current_aqi},
          "new_aqi": 0,
          "aqi_improvement_percent": 0,
          "health_benefits": "low or high",
          "timeline_months": 0
        }}
        """

            return ask_gemini(prompt)

response_service = ResponseService()
