from pydantic import BaseModel

# ML Model for health risk prediction
class HealthRiskRequest(BaseModel):
    pm25: float
    pm10: float
    no2: float
    o3: float
    co: float
    asthma: int
    heart_disease: int

# ML Model for region class prediction
class RegionClassRequest(BaseModel):
    pm25: float
    pm10: float
    no2: float
    o3: float
    co: float

# Common base for pollutant values
class PollutionBaseModel(BaseModel):
    pm25: float
    pm10: float
    no2: float
    co: float
    o3: float

class PredictPolicyRequest(PollutionBaseModel):
    policy: str
    location: str

class CompareLocationsRequest(BaseModel):
    location1: str
    location2: str
    pm25_1: float
    pm10_1: float
    pm25_2: float
    pm10_2: float

class HealthRisksRequest(PollutionBaseModel):
    pass

class SuggestPoliciesRequest(PollutionBaseModel):
    location: str
    aqi: int

class CitizenActionsRequest(PollutionBaseModel):
    pass

class MythBusterRequest(BaseModel):
    claim: str

class ReducePollutionPlanRequest(BaseModel):
    goal: str
    location: str

class HealthImpactResponse(BaseModel):
    risk_level: str
    risk_description: str
    sensitive_groups: str
    exposure_duration: str
    recommended_actions: str
