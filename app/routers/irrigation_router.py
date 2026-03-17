from fastapi import APIRouter
from pydantic import BaseModel
from app.services.irrigation_service import predict_irrigation

router = APIRouter()


class IrrigationRequest(BaseModel):
    temperature: float
    pressure: float
    altitude: float
    soil_moisture: float
    status: str
    hour: int
    day: int
    month: int


@router.post("/irrigation")
def irrigation(data: IrrigationRequest):
    return predict_irrigation(data)