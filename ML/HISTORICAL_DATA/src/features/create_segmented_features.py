
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_all_stocks_cleaned.csv"
SEGMENT_FILE = ROOT_DIR / "reports" / "stock_liquidity_classification.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_days_since_trade(volume_series: pd.Series) -> pd.Series:
    """
    Computes consecutive trading sessions where volume == 0 (liquidity flag from Blueprint).
    """
    is_zero = (volume_series == 0).astype(int)
    # Cumulative count that resets whenever volume > 0
    zero_blocks = (is_zero == 0).cumsum()
    return is_zero.groupby(zero_blocks).cumsum()

def main():
    print(f"\n[SEGMENTED FEATURES] Loading cleaned data and liquidity classification...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    seg_df = pd.read_csv(SEGMENT_FILE)
    
    # Map liquidity status
    liquid_map = dict(zip(seg_df["code"], seg_df["category"]))
    df["category"] = df["code"].map(liquid_map).fillna("UNKNOWN")
    
    # Filter to actual equities (Liquid + Illiquid, excluding market indices and rights issues)
    equities_df = df[df["category"].isin(["LIQUID", "ILLIQUID"])].copy()
    equities_df["date"] = pd.to_datetime(equities_df["date"])
    equities_df["day_price"] = pd.to_numeric(equities_df["day_price"], errors="coerce")
    equities_df["volume"] = pd.to_numeric(equities_df["volume"], errors="coerce").fillna(0)
    
    equities_df = equities_df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    print(f"[SEGMENTED FEATURES] Total equity records: {len(equities_df):,} across {equities_df['code'].nunique()} stocks.")
    
    processed_groups = []
    
    for code, group in equities_df.groupby("code", sort=False):
        g = group.copy().reset_index(drop=True)
        prices = g["day_price"]
        volumes = g["volume"]
        cat = g["category"].iloc[0]
        
        # Targets (forward looking)
        g["target_next_price_5d"] = prices.shift(-5)
        g["target_return_5d"] = ((g["target_next_price_5d"] / prices) - 1.0) * 100.0
        g["target_direction_5d"] = np.where(
            g["target_return_5d"].isna(),
            np.nan,
            np.where(g["target_return_5d"] > 0, 1, 0)
        )
        
        # Lags & Returns (backward looking)
        g["lag_1"] = prices.shift(1)
        g["lag_5"] = prices.shift(5)
        g["daily_return"] = ((prices / g["lag_1"]) - 1.0) * 100.0
        g["return_5d"] = ((prices / g["lag_5"]) - 1.0) * 100.0
        
        # Moving Averages
        g["sma_5"] = prices.rolling(5, min_periods=5).mean()
        g["sma_20"] = prices.rolling(20, min_periods=20).mean()
        g["sma_50"] = prices.rolling(50, min_periods=50).mean()
        
        # Technicals
        g["rsi_14"] = calculate_rsi(prices, period=14)
        g["volatility_20d"] = g["daily_return"].rolling(20, min_periods=20).std().fillna(0)
        
        # Liquidity-Specific Features
        g["is_liquid"] = 1 if cat == "LIQUID" else 0
        g["days_since_trade"] = compute_days_since_trade(volumes)
        g["is_flatline_day"] = ((prices == g["lag_1"]) & (volumes == 0)).astype(int)
        
        # 52-Week distance
        roll_high_250 = prices.rolling(250, min_periods=20).max()
        roll_low_250 = prices.rolling(250, min_periods=20).min()
        g["dist_52w_high"] = ((prices / roll_high_250) - 1.0) * 100.0
        g["dist_52w_low"] = ((prices / roll_low_250) - 1.0) * 100.0
        
        # Volume ratio
        vol_sma_5 = volumes.rolling(5, min_periods=5).mean()
        g["volume_ratio_5d"] = volumes / vol_sma_5.replace(0, np.nan)
        
        processed_groups.append(g)
        
    full_features = pd.concat(processed_groups, ignore_index=True)
    
    # Save cohorts
    liquid_file = OUTPUT_DIR / "features_liquid_cohort.csv"
    illiquid_file = OUTPUT_DIR / "features_illiquid_cohort.csv"
    all_equities_file = OUTPUT_DIR / "features_all_equities.csv"
    
    liquid_df = full_features[full_features["is_liquid"] == 1]
    illiquid_df = full_features[full_features["is_liquid"] == 0]
    
    liquid_df.to_csv(liquid_file, index=False)
    illiquid_df.to_csv(illiquid_file, index=False)
    full_features.to_csv(all_equities_file, index=False)
    
    print(f"\n[SEGMENTED FEATURES] Saved Liquid Cohort ({len(liquid_df):,} rows, {liquid_df['code'].nunique()} stocks) -> {liquid_file.name}")
    print(f"[SEGMENTED FEATURES] Saved Illiquid Cohort ({len(illiquid_df):,} rows, {illiquid_df['code'].nunique()} stocks) -> {illiquid_file.name}")
    print(f"[SEGMENTED FEATURES] Saved All Equities ({len(full_features):,} rows) -> {all_equities_file.name}")

if __name__ == "__main__":
    main()
