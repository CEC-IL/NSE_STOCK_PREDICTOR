from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Index
from app.database import Base

class Stock(Base):
    #representing a listed company in NSE

    __tablename__ = "stocks"

    id = Column(Integer, primary_key = True, index = True )
    symbol = Column( String(20), unique = True , index = True, nullable = False )
    name = Column (String(100), nullable = False )
    sector = Column( String(100), default = "General" )
    is_liquid = Column( Boolean, default = True )

class DailyPrice(Base):
    #storing historical daly price

    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key = True, index = True)
    symbol = Column(String(20), index = True, nullable = False)
    date = Column( Date, index = True, nullable = False )
    price = Column(Float, nullable = False )

    #Return features from ML
    daily_return = Column( Float, nullable = True )
    return_5d = Column ( Float, nullable = True )
    return_10d = Column ( Float, nullable = True )

    #Simple Moving Average
    sma_5d = Column( Float, nullable = True )
    sma_10d = Column( Float, nullable = True )
    sma_20d = Column( Float, nullable = True )

    #Volatility indicators
    volatility_5d = Column( Float, nullable = True )
    volatility_10d = Column( Float, nullable = True )
    volatility_20d = Column( Float, nullable = True )

    #Database constraint so that a stock can only have one record per day
    __table_args__ = (
        Index("idx_stock_date","symbol", "date", unique = True ),
    )
