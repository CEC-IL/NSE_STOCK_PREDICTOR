
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. PATHS & CONFIG
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_all_stocks_cleaned.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_liquid_features.csv"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

LIQUID_STOCKS = [
    "SCOM", "EQTY", "KCB", "EABL", "ABSA", 
    "COOP", "BAT", "BAMB", "SCBK", "NCBA", 
    "KEGN", "CTUM"
]

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    print(f"\n[FEATURES] Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Filter to liquid universe
    df = df[df["code"].isin(LIQUID_STOCKS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_price"] = pd.to_numeric(df["day_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    
    # Sort strictly by code and date to prevent temporal leakage
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    print(f"[FEATURES] Filtered to {len(df):,} observations across {df['code'].nunique()} liquid stocks.")
    
    feature_rows = []
    
    for code, group in df.groupby("code", sort=False):
        g = group.copy().reset_index(drop=True)
        prices = g["day_price"]
        volumes = g["volume"]
        
        # Targets (strictly forward looking)
        g["target_next_price_5d"] = prices.shift(-5)
        g["target_return_5d"] = ((g["target_next_price_5d"] / prices) - 1.0) * 100.0
        
        # Direction: UP (+1) if return > 0, DOWN (0) if return <= 0
        g["target_direction_5d"] = np.where(
            g["target_return_5d"].isna(),
            np.nan,
            np.where(g["target_return_5d"] > 0, 1, 0)
        )
        
        # Lags (strictly backward looking)
        g["lag_1"] = prices.shift(1)
        g["lag_5"] = prices.shift(5)
        g["lag_10"] = prices.shift(10)
        
        # Returns
        g["daily_return"] = ((prices / g["lag_1"]) - 1.0) * 100.0
        g["return_5d"] = ((prices / g["lag_5"]) - 1.0) * 100.0
        g["return_10d"] = ((prices / g["lag_10"]) - 1.0) * 100.0
        
        # Simple Moving Averages
        g["sma_5"] = prices.rolling(5, min_periods=5).mean()
        g["sma_20"] = prices.rolling(20, min_periods=20).mean()
        g["sma_50"] = prices.rolling(50, min_periods=50).mean()
        
        # RSI (14)
        g["rsi_14"] = calculate_rsi(prices, period=14)
        
        # MACD (12 EMA - 26 EMA)
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        g["macd"] = ema_12 - ema_26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        
        # Rolling Volatilities (annualized standard deviation of daily returns approx)
        g["volatility_5d"] = g["daily_return"].rolling(5, min_periods=5).std()
        g["volatility_10d"] = g["daily_return"].rolling(10, min_periods=10).std()
        g["volatility_20d"] = g["daily_return"].rolling(20, min_periods=20).std()
        
        # 52-Week (approx 250 trading days) High / Low Distance
        roll_high_250 = prices.rolling(250, min_periods=20).max()
        roll_low_250 = prices.rolling(250, min_periods=20).min()
        g["dist_52w_high"] = ((prices / roll_high_250) - 1.0) * 100.0
        g["dist_52w_low"] = ((prices / roll_low_250) - 1.0) * 100.0
        
        # Volume features
        vol_sma_5 = volumes.rolling(5, min_periods=5).mean()
        g["volume_ratio_5d"] = volumes / vol_sma_5.replace(0, np.nan)
        
        feature_rows.append(g)
        
    final_df = pd.concat(feature_rows, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[FEATURES] Saved liquid feature set ({len(final_df):,} rows) to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
