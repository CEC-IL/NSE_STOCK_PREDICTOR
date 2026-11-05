
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nse_price_features.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD PRICE FEATURES
# --------------------------------------------------

print("\nLoading price features...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["daily_sequence_price"] = pd.to_numeric(
    df["daily_sequence_price"],
    errors="coerce"
)

df["daily_return"] = pd.to_numeric(
    df["daily_return"],
    errors="coerce"
)

df = df.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 3. CREATE MOVING AVERAGES
# --------------------------------------------------

print("Creating moving averages...")

grouped_prices = df.groupby("code", sort=False)[
    "daily_sequence_price"
]

for window in [5, 10, 20]:

    column_name = f"sma_{window}"

    df[column_name] = grouped_prices.transform(
        lambda prices: prices.rolling(
            window=window,
            min_periods=window
        ).mean()
    )


# --------------------------------------------------
# 4. CREATE VOLATILITY FEATURES
# --------------------------------------------------

print("Creating volatility features...")

grouped_returns = df.groupby("code", sort=False)[
    "daily_return"
]

for window in [5, 10, 20]:

    column_name = f"volatility_{window}d"

    df[column_name] = grouped_returns.transform(
        lambda returns: returns.rolling(
            window=window,
            min_periods=window
        ).std()
    )


# --------------------------------------------------
# 5. SAVE FEATURE DATASET
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR / "nse_technical_features.csv"
)

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 6. FEATURE AVAILABILITY REPORT
# --------------------------------------------------

technical_columns = [
    "sma_5",
    "sma_10",
    "sma_20",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d"
]

summary = []

for column in technical_columns:

    summary.append({
        "feature": column,
        "available_values": df[column].notna().sum(),
        "missing_values": df[column].isna().sum(),
        "missing_percentage": round(
            df[column].isna().mean() * 100,
            2
        )
    })

summary_df = pd.DataFrame(summary)

summary_file = (
    REPORT_DIR / "technical_feature_summary.csv"
)

summary_df.to_csv(summary_file, index=False)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 75)
print("TECHNICAL FEATURE ENGINEERING SUMMARY")
print("=" * 75)

print(f"Total records: {len(df):,}")
print(f"Stock codes: {df['code'].nunique()}")

print("\nFEATURE AVAILABILITY")
print("-" * 75)

print(summary_df.to_string(index=False))


# --------------------------------------------------
# 8. DISPLAY SCOM SAMPLE
# --------------------------------------------------

print("\nSAMPLE TECHNICAL FEATURES: SCOM")
print("-" * 75)

sample = df[df["code"] == "SCOM"]

print(
    sample[
        [
            "date",
            "daily_sequence_price",
            "sma_5",
            "sma_10",
            "sma_20",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d"
        ]
    ].tail(15).to_string(index=False)
)


# --------------------------------------------------
# 9. COMPLETION
# --------------------------------------------------

print("\nSAVED FILES")
print("-" * 75)

print(OUTPUT_FILE)
print(summary_file)

print("\nTechnical feature engineering completed successfully.")
