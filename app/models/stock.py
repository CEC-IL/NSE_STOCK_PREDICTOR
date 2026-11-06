from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Index
from app.database import Base

class Stock(Base):
    #representing a listed company in NSE

    __tablename__ = "stocks"
    
