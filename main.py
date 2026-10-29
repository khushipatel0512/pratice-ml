from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

BASE_DIR = Path(__file__).parent

app = FastAPI()
model = joblib.load(BASE_DIR / "linearmodel.pkl")


class datascience(BaseModel):
    AdvertisingBudget: float


# Website (index.html sits in the same folder)
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


# Old test route, kept for health checks
@app.get("/health")
def test():
    return {"msg": "uvicorn is working"}


@app.post("/predict")
def datainsert(ds: datascience):
    Sales = model.predict([[ds.AdvertisingBudget]])
    return {"predictsales": round(float(Sales[0]), 2)}