from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import date

class StockSummary(BaseModel):
    #schema for displaying stock in an overview table

    symbol: str
    name: str
    sector: str
    latest_price: float
    change_pct: float
    is_liquid: bool

    model_config = ConfigDict(from_attributes = True)

class PriceHistoryItems(BaseModel):
    # schema for a single historical day

    date: date
    price: float
    daily_return: Optional[float] = None
    sma_5: Optional[float] = None
    sma_20: Optional[float] = None
    volatility_20d: Optional[float] = None

    model_config = ConfigDict(from_attributes = True)

class FeatureDriver(BaseModel):
    feature_name: str
    impact: str
    description: str

class PredictionResponse(BaseModel):
    symbol: str
    target_window : str = "5 trading days"
    prediction_direction: str
    up_probability: float
    expected_return_range: str
    confidence_level: str
    top_drivers: List[FeatureDriver]
    disclaimer: str = (
        "Decision support tool for research purposes only, not for financial advice."
        "Past performance does not guarantee future results."
    )

    model_config = ConfigDict(from_attributes = True)

    