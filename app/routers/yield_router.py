from fastapi import APIRouter
from pydantic import BaseModel
from app.services.yield_service import predict_yield

router = APIRouter()

class YieldRequest(BaseModel):
    state: str
    season: str
    crop: str
    year: int
    area: float


@router.post("/yield-predict")
def yield_prediction(data: YieldRequest):
    return predict_yield(data)
