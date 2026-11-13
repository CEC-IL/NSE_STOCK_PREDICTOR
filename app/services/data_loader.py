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
        project_root = current_file.parents[3]
        csv_path = project_root / "ML" / "HISTORICAL_DATA" / "data" / "processed" / "nse_prediction_targets.csv"

        if not csv_path.exists():
            print(f"Error: Could not find ML dataset at: {csv_path}")
            return

        print(f"loading ML dataset from: {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory = False)

        # standardize date formart
        df['date'] = pd.to_datetime(df['date']).dt.date

        # seed the stock company profiles
        print('Registering stock company profiles')
        for symbol, info in LIQUID_STOCKS.items():
            existing = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not existing:
                stock = Stock(
                    symbol=symbol,
                    name=info['name'],
                    sector = info['sector'],
                    is_liquid=True
                )
                db.add(stock)

        db.commit()

        # filter for liquid stocks
        if limit_to_liquid:
            df = df[df['code'].isin(LIQUID_STOCKS.keys())]

        total_rows = len(df)
        print(f"Preparing to insert {total_rows:,} daily records of liquid stocks ")

        #find existing records to avoid duplicates
        existing_records = set(
            db.query(DailyPrice.symbol, DailyPrice.date).all()
        )

        records_to_insert = []
        batch_size = 2000

        for _, row in df.iterrows():
            key =(row['code'], row['date'])
            if key in existing_records:
                continue

            records_to_insert.append(
                DailyPrice(
                    symbol=row["code"],
                    date=row["date"],
                    price=float(row["daily_sequence_price"]) if pd.notna(row["daily_sequence_price"]) else 0.0,
                    daily_return=float(row["daily_return"]) if pd.notna(row.get("daily_return")) else None,
                    return_5d=float(row["return_5d"]) if pd.notna(row.get("return_5d")) else None,
                    return_10d=float(row["return_10d"]) if pd.notna(row.get("return_10d")) else None,
                    sma_5d=float(row["sma_5"]) if pd.notna(row.get("sma_5")) else None,
                    sma_10d=float(row["sma_10"]) if pd.notna(row.get("sma_10")) else None,
                    sma_20d=float(row["sma_20"]) if pd.notna(row.get("sma_20")) else None,
                    volatility_5d=float(row["volatility_5d"]) if pd.notna(row.get("volatility_5d")) else None,
                    volatility_10d=float(row["volatility_10d"]) if pd.notna(row.get("volatility_10d")) else None,
                    volatility_20d=float(row["volatility_20d"]) if pd.notna(row.get("volatility_20d")) else None,
                )
            )

            #insert in batches for high speed
            if len(records_to_insert) >= batch_size:
                db.bulk_save_objects(records_to_insert)
                db.commit()
                records_to_insert = []

        #insert any remaining records
        if records_to_insert:
            db.bulk_save_objects(records_to_insert)
            db.commit()

        print('Database successfully populated with historical prices and features')
        
    finally:
        db.close()

if __name__== '__main__':
    load_ml_database_to_db()