from fastapi import APIRouter
from pydantic import BaseModel
from app.services.fertilizer_service import predict_fertilizer

router = APIRouter()


class FertilizerRequest(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    potassium: float
    phosphorous: float


@router.post("/fertilizer")
def fertilizer_prediction(data: FertilizerRequest):
    return predict_fertilizer(data)
