import os
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app.models.stock import Stock, DailyPrice

LIQUID_STOCKS = {
    "SCOM": {"name": "Safaricom PLC", "sector": "Telecommunications"},
    "EQTY": {"name": "Equity Group Holdings", "sector": "Banking"},
    "KCB": {"name": "KCB Group PLC", "sector": "Banking"},
    "EABL": {"name": "East African Breweries PLC", "sector": "Manufacturing"},
    "ABSA": {"name": "Absa Bank Kenya PLC", "sector": "Banking"},
    "COOP": {"name": "Co-operative Bank of Kenya", "sector": "Banking"},
    "BAT": {"name": "British American Tobacco Kenya", "sector": "Manufacturing"},
    "BAMB": {"name": "Bamburi Cement PLC", "sector": "Manufacturing"},
    "NCBA": {"name": "NCBA Group PLC", "sector": "Banking"},
    "SCBK": {"name": "Standard Chartered Bank Kenya", "sector": "Banking"},
    "DTK": {"name": "Diamond Trust Bank Kenya", "sector": "Banking"},
    "KEGN": {"name": "KenGen Co. Plc", "sector": "Energy"},
}

def load_ml_database_to_db(limit_to_liquid: bool = True):
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        #locate the csv file
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]
        csv_path = project_root / "ML" / "HISTORICAL_DATA" / "data" / "processed" / "nse_prediction_targets.csv"

        if not csv_path.exists():
            print(f"Error: Could not find ML dataset at: {csv_path}")
            return

        print(f"loading ML dataset from: {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory = False)

        

    finally:
        db.close()