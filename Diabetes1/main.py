from fastapi import FastAPI
import joblib
import uvicorn
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel



app = FastAPI()

model = joblib.load("log_model.pkl")
scaler = joblib.load("skaler.pkl")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]).reshape(1, -1)


    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return {
            "diabetes": bool(prediction),
            "probability": round(float(probability), 2)
        }


@app.get("/")
def home():
    return {"message": "Diabetes predict"}





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8010)
