from fastapi import APIRouter
from pydantic import BaseModel
from app.services.crop_service import predict_crop

router = APIRouter()


class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@router.post("/crop-suggest")
def crop_prediction(data: CropRequest):
    return predict_crop(data)