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

```
## Project Structure
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

