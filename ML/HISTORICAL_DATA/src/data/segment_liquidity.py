
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_all_stocks_cleaned.csv"
REPORT_DIR = ROOT_DIR / "reports"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def segment_stock_universe():
    print(f"\n[LIQUIDITY] Loading cleaned dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["day_price"] = pd.to_numeric(df["day_price"], errors="coerce")
    df["change"] = pd.to_numeric(df["change"], errors="coerce").fillna(0)
    
    # Calculate liquidity metrics per stock
    grouped = df.groupby("code")
    
    records = []
    for code, g in grouped:
        total_days = len(g)
        active_vol_days = int((g["volume"] > 0).sum())
        price_change_days = int((g["change"] != 0).sum())
        zero_trade_days = total_days - active_vol_days
        
        activity_rate = (active_vol_days / total_days) * 100 if total_days > 0 else 0
        zero_trade_pct = (zero_trade_days / total_days) * 100 if total_days > 0 else 0
        mean_vol = float(g["volume"].mean())
        median_vol = float(g["volume"].median())
        mean_price = float(g["day_price"].dropna().mean()) if g["day_price"].notna().any() else 0.0
        
        # Determine classification
        if code.startswith("^"):
            category = "INDEX"
        elif code.endswith("-R") and total_days < 40:
            category = "RIGHTS_ISSUE"
        elif activity_rate >= 65.0 and total_days >= 200:
            category = "LIQUID"
        else:
            category = "ILLIQUID"
            
        records.append({
            "code": code,
            "category": category,
            "total_days": total_days,
            "active_trading_days": active_vol_days,
            "zero_trade_days": zero_trade_days,
            "activity_rate_pct": round(activity_rate, 2),
            "zero_trade_pct": round(zero_trade_pct, 2),
            "mean_daily_volume": round(mean_vol, 1),
            "median_daily_volume": round(median_vol, 1),
            "mean_price_kes": round(mean_price, 2),
            "first_date": str(g["date"].min().date()),
            "last_date": str(g["date"].max().date()),
        })
        
    summary_df = pd.DataFrame(records).sort_values(
        by=["category", "activity_rate_pct"], 
        ascending=[True, False]
    ).reset_index(drop=True)
    
    # Save segmentation table
    output_csv = REPORT_DIR / "stock_liquidity_classification.csv"
    summary_df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    cat_counts = summary_df["category"].value_counts()
    print("\n" + "=" * 65)
    print("NSE STOCK LIQUIDITY SEGMENTATION SUMMARY")
    print("=" * 65)
    for cat, count in cat_counts.items():
        print(f"  • {cat}: {count} instruments")
        
    liquid_stocks = summary_df[summary_df["category"] == "LIQUID"]
    illiquid_stocks = summary_df[summary_df["category"] == "ILLIQUID"]
    
    print(f"\nLiquid Cohort ({len(liquid_stocks)} stocks): Average Activity Rate = {liquid_stocks['activity_rate_pct'].mean():.1f}%")
    print(f"Illiquid Cohort ({len(illiquid_stocks)} stocks): Average Activity Rate = {illiquid_stocks['activity_rate_pct'].mean():.1f}%")
    print(f"\nSaved full classification table to: {output_csv}")
    
    return summary_df

if __name__ == "__main__":
    segment_stock_universe()
