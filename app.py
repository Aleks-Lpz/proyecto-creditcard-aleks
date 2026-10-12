from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pathlib
import numpy as np

app = FastAPI(title="Credit Card Fraud Prediction API", version="0.1.0")


model_path = pathlib.Path('model/creditcard-v1.joblib')
model = load(model_path)

class InputData(BaseModel):
    features: list[float]

class OutputData(BaseModel):
    score: float = 0.0

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return OutputData(score=float(prediction))
