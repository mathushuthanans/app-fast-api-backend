from fastapi import FastAPI, Query
from ai_logic import ask_groq
from app.services import response_service
rs = response_service.response_service


app = FastAPI()


@app.get("/help")
async def help_info():
    message = """
Clarity is an AI assistant that explains pollution data and health risks to citizens, policymakers, and students. It suggests actions, predicts policy impacts, creates quizzes, and simulates urban planning scenarios to promote pollution awareness and healthier communities.
"""
    return {"help": message}


@app.get("/explain")
async def explain_pollution(pollutant: str):
    prompt = (
        f"Explain the pollutant '{pollutant}'. "
        "Respond strictly in this JSON format:\n\n"
        "{\n"
        '  "object": "What is {pollutant}?",\n'
        '  "causes": ["Cause 1", "Cause 2", "Cause 3"],\n'
        '  "effects": ["Effect 1", "Effect 2", "Effect 3"]\n'
        "}\n"
        "Do not include any extra text or explanation outside the JSON."
    ).replace("{pollutant}", pollutant)

    result = ask_groq(prompt)
    return result



@app.get("/predict_policy")
async def predict_policy(
    policy: str,
    location: str,
    pm25: float,
    pm10: float,
    no2: float,
    co: float,
    o3: float
):
    aqi = [pm25, pm10, no2, co, o3]

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



@app.get("/compare_locations")
async def compare_locations(location1: str, location2: str,
                            pm25_1: float, pm10_1: float,
                            pm25_2: float, pm10_2: float):
    prompt = f"""
You are Clarity.
Compare pollution health impact between:
- {location1}: PM2.5={pm25_1}, PM10={pm10_1}
- {location2}: PM2.5={pm25_2}, PM10={pm10_2}
Give 2-line difference summary.
Return as {{"comparison": "..."}}
"""
    return ask_groq(prompt)



@app.get("/health_risks")
async def health_risks(pm25: float, pm10: float, no2: float, co: float, o3: float):
    aqi = [pm25, pm10, no2, co, o3]
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

@app.get("/get_scenario_presets")
async def get_scenario_presets():
    """Returns predefined what-if scenarios using existing methods"""
    # Get current baseline data using your methods
    current_location = rs.get_location()
    current_density = get_vehicle_density()
    current_industry = get_industrial_impact()
    
    prompt = f"""
You are Clarity, analyzing pollution scenarios for Nagpur's St. Vincent Pallotti College.
Current baseline data:
- Pollution: {current_location['data']['pollution']}
- Vehicle density: {current_density}
- Industrial impact: {current_industry}

Generate 3 what-if scenarios in this exact JSON format:
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

Use these current values from the baseline data:
- Current PM2.5: {current_location['data']['pollution']['pm25']}
- Current NO2: {current_location['data']['pollution']['no2']}
- Current vehicle density: {current_density['density']['total_per_km']}/km
- Current industrial impact: {current_industry['composite_impact']}
"""
    return ask_groq(prompt)



debug_policy_suggestions = False
@app.get("/suggest_policies")
async def suggest_policies(location: str, aqi: int,
                           pm25: float, pm10: float, no2: float, co: float, o3: float):
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

@app.get("/citizen_actions")
async def citizen_actions(pm25: float, pm10: float, no2: float, co: float, o3: float):
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



@app.get("/daily_tip")
async def daily_tip():
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



@app.get("/myth_buster")
async def myth_buster(claim: str):
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



@app.get("/reduce_pollution_plan")
async def reduce_pollution_plan(goal: str, location: str):
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