# Import FastAPI framework
from fastapi import FastAPI

# Import joblib to load saved ML model
import joblib

# Import numpy for numerical processing
import numpy as np
import pandas as pd

# -----------------------------------
# STEP 1: Create FastAPI App
# -----------------------------------
# This initializes the API server

app = FastAPI()


# -----------------------------------
# STEP 2: Load Trained Model
# -----------------------------------
# Load model saved during training step

model = joblib.load("models/model.pkl")


# -----------------------------------
# STEP 3: Home Route
# -----------------------------------
# Simple API route to verify API is running
#
# URL:
# http://127.0.0.1:8000/

@app.get("/")
def home():

    return {
        "message": "Heart Disease Prediction API is Running"
    }


# -----------------------------------
# STEP 4: Prediction Route
# -----------------------------------
# This endpoint accepts patient data
# and returns prediction result
#
# URL:
# http://127.0.0.1:8000/predict
#
# Method:
# POST

@app.post("/predict")
def predict(data: dict):


    # -----------------------------------
    # STEP 5: Extract Input Values
    # -----------------------------------
    # Convert JSON values into list

    #values = list(data.values())


    # -----------------------------------
    # STEP 6: Convert into NumPy Array
    # -----------------------------------
    # ML models expect 2D array format

    #input_data = np.array(values).reshape(1, -1)
    input_data = pd.DataFrame([data])


    # -----------------------------------
    # STEP 7: Generate Prediction
    # -----------------------------------
    # 0 = No heart disease
    # 1 = Heart disease present

    prediction = model.predict(input_data)[0]


    # -----------------------------------
    # STEP 8: Generate Confidence Score
    # -----------------------------------
    # predict_proba returns probabilities
    #
    # [0] = probability of class 0
    # [1] = probability of class 1

    probability = model.predict_proba(input_data)[0][1]


    # -----------------------------------
    # STEP 9: Return API Response
    # -----------------------------------

    return {

        "prediction": int(prediction),

        "confidence": float(probability)
    }