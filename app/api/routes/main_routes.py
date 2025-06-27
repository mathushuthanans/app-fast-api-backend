from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.models.datamodel1 import HealthRiskRequest, RegionClassRequest
from app.models.datamodel1 import (
    PredictPolicyRequest, CompareLocationsRequest, HealthRisksRequest, 
    SuggestPoliciesRequest, CitizenActionsRequest, MythBusterRequest, 
    ReducePollutionPlanRequest
)
from app.services.response_service import response_service



app = FastAPI()
# Explicitly allow the origin where your HTML is served
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()



# === Dashboard Routes ===

# Done
@router.get('/get_trends')
def get_trends(range: str = "weekly"):  # Default to weekly if not specified
    return response_service.get_trends(range)


@router.get("/get_location")
def get_location():
    return response_service.get_location()

@router.get("/get_scenario_presets")
def get_scenario_presets():
    return response_service.get_scenario_presets()



# === ML Service Routes ===

@router.post("/predict_health_risk")
def predict_health_risk(data: HealthRiskRequest):
    return response_service.predict_health_risk(data)

@router.post("/predict_region_class")
def predict_region_class(data: RegionClassRequest):
    return response_service.predict_region_class(data)

@router.get("/bridge_predict")
def bridge_predict(asthma: int = 0, heart_disease: int = 0):
    return response_service.bridge_predict(asthma, heart_disease)



# === Clarity AI Routes ===

@router.get("/help")
def help_info():
    return response_service.help_info()

@router.get("/explain")
def explain_pollution(pollutant: str):
    return response_service.explain_pollution(pollutant)

@router.post("/predict_policy")
def predict_policy_post(request: PredictPolicyRequest):
    return response_service.predict_policy(
        request.policy, request.location, request.pm25, request.pm10,
        request.no2, request.co, request.o3
    )

@router.get("/predict_policy")
def predict_policy_get(
    policy: str, location: str,
    pm25: float, pm10: float,
    no2: float, co: float, o3: float
):
    return response_service.predict_policy(
        policy, location, pm25, pm10, no2, co, o3
    )


@router.post("/compare_locations")
def compare_locations(request: CompareLocationsRequest):
    return response_service.compare_locations(
        request.location1, request.location2,
        request.pm25_1, request.pm10_1, request.pm25_2, request.pm10_2
    )

@router.post("/health_risks")
def health_risks(request: HealthRisksRequest):
    return response_service.health_risks(
        request.pm25, request.pm10, request.no2, request.co, request.o3
    )

@router.post("/suggest_policies")
def suggest_policies(request: SuggestPoliciesRequest):
    return response_service.suggest_policies(
        request.location, request.aqi,
        request.pm25, request.pm10, request.no2, request.co, request.o3
    )

@router.post("/citizen_actions")
def citizen_actions(request: CitizenActionsRequest):
    return response_service.citizen_actions(
        request.pm25, request.pm10, request.no2, request.co, request.o3
    )

@router.get("/myth_buster")
def myth_buster(claim: str):
    return response_service.myth_buster(claim)


@router.post("/reduce_pollution_plan")
def reduce_pollution_plan(request: ReducePollutionPlanRequest):
    return response_service.reduce_pollution_plan(request.goal, request.location)

