from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.stock import Stock, DailyPrice
from app.schemas.stock import StockSummary, PriceHistoryItem

#router dedicated to stock endpoints
router = APIRouter(prefix="/stocks",tags=["stocks"])

@router.get("", response_model = List[StockSummary])
def get_stocks(db: Session = Depends(get_db)):
    #called by frontend to populate the main dashboard
    stocks = db.query(Stock).filter(Stock.is_liquid == True).all()
    results = []

    for stock in stocks:
        #find the most recent trading day for each stock
        latest = (
            db.query(DailyPrice)
            .filter(DailyPrice.symbol == stock.symbol)
            .order_by(DailyPrice.date.desc())
            .first()
        )

        if latest:
            results.append(
                StockSummary(
                    symbol = stock.symbol,
                    name = stock.name,
                    sector = stock.sector,
                    latest_price = latest.price,
                    change_pct = latest.daily_return if latest.daily_return is not None else 0.0,
                    is_liquid = stock.is_liquid
                )
            )

    return results

@router.get("/{symbol}", response_model = StockSummary )
def get_single_stock(symbol: str, db: Session = Depends(get_db)):
    #returns details for a single stock by its ticker symbol
    symbol_upper = symbol.upper()

    stock = db.query(Stock).filter(Stock.symbol == symbol_upper).first()
    if not stock:
        raise HTTPException( status_code= 404, detail= f"Stock '{symbol_upper}' not found")

    latest = (
        db.query(DailyPrice)
        .filter(DailyPrice.symbol == stock.symbol)
        .order_by(DailyPrice.date.desc())
        .first()
    )

    if not latest:
        raise HTTPException( status_code = 404, detail = f"No price data for '{symbol_upper}'")

    return StockSummary(
        symbol = stock.symbol,
        name = stock.name,
        sector = stock.sector,
        latest_price = latest.price,
        change_pct = latest.daily_return if latest.daily_return is not None else 0.0,
        is_liquid = stock.is_liquid
    )

@router.get("/{symbol}/history", response_model = List[PriceHistoryItem])
def get_stock_history( symbol: str, limit: int = 30, db: Session = Depends(get_db)):
    #returns historical price series with technical indicators for charting
    symbol_upper = symbol.upper()

    history = (
        db.query(DailyPrice)
        .filter(DailyPrice.symbol == symbol_upper)
        .order_by(DailyPrice.date.desc())
        .limit(limit)
        .all()
    )

    if not history:
        raise HTTPException( status_code = 404, detail = f"No history found for '{symbol_upper}'")

    #return in chronological order
    return sorted(history, key=lambda x: x.date)
