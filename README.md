````markdown
# PollutionViz — Backend

<p align="center">
  <strong>FastAPI backend for PollutionViz</strong><br>
  Location-aware air-quality data, health analysis, machine-learning predictions, and AI-powered pollution insights.
</p>

---

## Overview

PollutionViz is an air-pollution awareness and decision-support platform designed to help users understand local air quality, explore pollution scenarios, assess health impacts, and receive actionable recommendations.

This repository contains the **backend API** of PollutionViz.

The backend is built with **FastAPI** and integrates external pollution/weather services, machine-learning models, and AI-powered analysis to provide data consumed by the PollutionViz Flutter frontend.

---

## Features

### 🌍 Location & Air Quality

- Location-aware pollution data using latitude and longitude
- Current air-pollution measurements
- Weather information
- Reverse geocoding for readable location names
- AQI calculation from pollutant concentrations
- Pollution forecast data

### 📈 Pollution Trends

- Weekly pollution trends
- Monthly pollution summaries
- Yearly pollution summaries
- PM2.5 analysis
- PM10 analysis
- NO₂ analysis
- CO analysis
- O₃ analysis

### 🧠 Machine Learning

- Health-risk prediction
- Region classification
- Health-related pollution analysis
- Model-backed pollution intelligence

### 🤖 Clarity AI

- Pollution explanations
- Pollution myth buster
- Policy prediction
- Policy recommendations
- Citizen action recommendations
- Pollution-reduction planning

### 🏛️ Policy & Scenario Analysis

- Pollution scenario presets
- Location comparison
- Policy impact prediction
- Recommended pollution-control policies
- Location-based citizen actions

---

## Architecture

```text
                    ┌──────────────────────┐
                    │   Flutter Frontend   │
                    └──────────┬───────────┘
                               │
                               │ HTTP / REST
                               ▼
                    ┌──────────────────────┐
                    │     FastAPI API      │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │  OpenWeather │  │ ML Models    │  │  Clarity AI  │
     │     APIs     │  │              │  │ / Gen AI     │
     └──────────────┘  └──────────────┘  └──────────────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   JSON API Response  │
                    └──────────────────────┘
````

---

## Project Structure

```text
PollutionViz-Backend/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── main_routes.py
│   │
│   ├── models/
│   │   └── datamodel1.py
│   │
│   ├── services/
│   │   ├── response_service.py
│   │   └── ...
│   │
│   ├── config.py
│   └── main.py
│
├── ml_models/
│   ├── ai_logic.py
│   ├── utils.py
│   └── ...
│
├── requirements.txt
├── render.yaml
├── setup.py
└── .gitignore
```

---

## API Routes

The main API routes are defined in:

```text
app/api/routes/main_routes.py
```

### Dashboard Routes

```text
GET  /get_location
GET  /get_trends
GET  /get_scenario_presets
POST /citizen_actions
GET  /health_impact
```

Location-aware routes accept:

```text
lat
lon
```

Trend requests additionally accept:

```text
range
```

Supported values:

```text
weekly
monthly
yearly
```

### Machine Learning Routes

```text
POST /predict_health_risk
POST /predict_region_class
GET  /bridge_predict
```

### Clarity AI Routes

```text
GET  /help
GET  /explain
POST /predict_policy
GET  /predict_policy
POST /compare_locations
POST /health_risks
POST /suggest_policies
POST /citizen_actions
GET  /myth_buster
POST /reduce_pollution_plan
```

---

## External Data Sources

The backend uses external services to obtain and process pollution and weather information.

### OpenWeather

The location service uses latitude and longitude to request:

* Current air pollution
* Air-pollution forecast
* Current weather
* Reverse-geocoded location information

The backend converts the returned pollutant concentrations into the application's AQI representation and response format.

---

## Machine Learning

The backend loads trained machine-learning models from:

```text
ml_models/
```

The current service includes model-backed functionality for:

```text
Health Risk Prediction
Region Classification
```

Models are loaded by the response service and used by the corresponding API routes.

---

## Environment Variables

The backend expects sensitive credentials and service configuration to be provided through environment variables.

Examples include:

```text
OPENWEATHER_API_KEY
API_KEY_GEMINI
GEN_AI
API_KEY_GROQ
BASE_URL_GROQ
API_URL_WAQI
```

Do not commit real API keys or secrets to the repository.

For local development, use a `.env` file or your system/environment configuration according to your deployment setup.

---

## Requirements

Make sure the following are installed:

* Python 3.x
* pip
* Virtual environment support

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/mathushuthanans/PollutionViz-Backend.git
cd PollutionViz-Backend
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Set the required API keys and service configuration in your environment.

---

## Running the Backend

From the repository root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative OpenAPI documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## Android Emulator Integration

When the Flutter frontend runs on an Android Emulator and the FastAPI server runs on the development machine, the frontend can use:

```text
http://10.0.2.2:8000
```

instead of:

```text
http://127.0.0.1:8000
```

The intended request flow is:

```text
Android Emulator
       │
       │ lat + lon
       ▼
Flutter Frontend
       │
       │ HTTP
       ▼
FastAPI
       │
       ▼
OpenWeather / ML / AI
```

---

## Location-Based Processing

Location-dependent endpoints are designed around:

```text
latitude
longitude
```

For example:

```text
GET /get_location?lat=<LAT>&lon=<LON>
```

The backend uses these coordinates to request pollution, forecast, weather, and reverse-geocoding information.

---

## Response Service

The main application logic is centralized in:

```text
app/services/response_service.py
```

This service handles:

* OpenWeather API requests
* Pollution data processing
* AQI calculation
* Forecast aggregation
* Location resolution
* Health-risk logic
* Region classification
* Policy analysis
* AI-assisted responses

---

## Deployment

The backend includes a Render deployment configuration:

```text
render.yaml
```

The configured application entry point is:

```text
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

This means the deployed service uses the root:

```text
app/
```

package as the application entry point.

---

## Frontend

The corresponding Flutter frontend is available at:

[https://github.com/baala-codes/Flutter_ps8---frontend](https://github.com/baala-codes/Flutter_ps8---frontend)

The Flutter application consumes the REST APIs provided by this backend.

---

## Development Workflow

```text
1. Flutter requests location / pollution data
                ↓
2. FastAPI receives the request
                ↓
3. Request parameters are validated
                ↓
4. ResponseService processes the request
                ↓
5. External APIs / ML / AI services are called
                ↓
6. Backend formats the result
                ↓
7. JSON response returned to Flutter
```

---

## Current Development Focus

* Connecting frontend API calls to the current FastAPI route contract
* Passing dynamic latitude and longitude from the client
* Validating OpenWeather-backed responses
* Integrating ML prediction endpoints
* Integrating Clarity AI functionality
* Improving production deployment configuration

---

## Technology Stack

* **Python**
* **FastAPI**
* **Uvicorn**
* **Requests**
* **NumPy**
* **Joblib**
* **Machine Learning**
* **OpenWeather APIs**
* **Google Gemini / Generative AI**
* **Groq**
* **Render**

---

## Related Repository

Frontend:

[https://github.com/baala-codes/Flutter_ps8---frontend](https://github.com/baala-codes/Flutter_ps8---frontend)

Backend:

[https://github.com/mathushuthanans/PollutionViz-Backend](https://github.com/mathushuthanans/PollutionViz-Backend)

---

## Project Status

PollutionViz Backend is under active development.

The current backend provides REST endpoints for:

* Pollution and weather data
* Location information
* Pollution trends
* Health analysis
* Machine-learning predictions
* Policy analysis
* Citizen recommendations
* AI-powered pollution insights

````

One thing I would **not** put in the README is the actual API keys currently visible in your `render.yaml`; that configuration contains credentials, so those should be rotated and kept in deployment secrets rather than documented or committed. Your Render configuration confirms the backend is deployed from `app.main:app`. 

Also, your current backend repository is genuinely under **your GitHub account**, `mathushuthanans/PollutionViz-Backend`, and your account has admin/maintain/push access to it. 

For the Git commit, I'd use:

```text
docs: add comprehensive backend README
````
