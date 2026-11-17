from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.stock import PredictionResponse
from app.services.prediction_service import prediction_service

# router dedicated to prediction endpoints
router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.get("/{symbol}", response_model= PredictionResponse)
def get_stock_prediction( symbol: str, db: Session = Depends(get_db)):
    #generates a 5-day forecast for a given stock
    return prediction_service.predict_for_stock(symbol.upper(), db )

@router.get("/metrics/summary")
def get_model_evaluation_metrics():
    #Reports how well the prediction model is performing against baselines

    return {
        "evaluation_strategy" : (
            "Walk-foward validation"
            "(Train 2007-2019, Val 2020-2022, Test 2023-2024)"
        ),
        "baselines": {
            "always_up_accuracy": "31.0%",
            "tomorrow_equals_today_mae": "0.42 KEs"
        },
        "target_accuracy_goal": "Beat simple baselines on out-of-sample data",
        "status": "Baseline mode — awaiting trained ML model"
    }