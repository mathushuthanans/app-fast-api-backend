import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Function 1: Train Health Risk Model
def train_health_risk_model():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "health_risk.csv")
    df = pd.read_csv(csv_path)

    features = ['pm25', 'pm10', 'no2', 'o3', 'co', 'asthma', 'heart_disease']
    X = df[features]
    y = df['danger_scale']

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=50)
    model.fit(xtrain, ytrain)

    joblib.dump(model, os.path.join(current_dir, "health_risk_model.pkl"))
    print("Health risk model saved.")
    return model

# Function 2: Train Region Class Model
def train_region_model():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "regionCategory.csv")
    df = pd.read_csv(csv_path)

    features = ['pm25', 'pm10', 'no2', 'o3', 'co']
    X = df[features]
    y = df['region_class']

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=50)
    model.fit(xtrain, ytrain)

    joblib.dump(model, os.path.join(current_dir, "region_model.pkl"))
    print("Region model saved.")
    return model

# Function 3: Predict using both models
def predict_health_and_region(pm25, pm10, no2, o3, co, asthma, heart_disease):
    current_dir = os.path.dirname(__file__)
    health_model_path = os.path.join(current_dir, "health_risk_model.pkl")
    region_model_path = os.path.join(current_dir, "region_model.pkl")

    # Load models
    health_model = joblib.load(health_model_path)
    region_model = joblib.load(region_model_path)

    # Prepare input for both models
    health_features = np.array([[pm25, pm10, no2, o3, co, asthma, heart_disease]])
    region_features = np.array([[pm25, pm10, no2, o3, co]])

    danger_scale = health_model.predict(health_features)[0]
    region_class = region_model.predict(region_features)[0]

    return {
        "danger_scale": danger_scale,
        "region_class": region_class
    }

# Example Usage (after training)
if __name__ == "__main__":
    # Train both models once
    train_health_risk_model()
    train_region_model()

    # Simulate prediction
    result = predict_health_and_region(pm25=55, pm10=80, no2=25, o3=40, co=0.5, asthma=1, heart_disease=0)
    print(result)
