
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
    / "nse_daily_sequence.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DAILY SEQUENCE
# --------------------------------------------------

print("\nLoading daily sequence...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["daily_sequence_price"] = pd.to_numeric(
    df["daily_sequence_price"],
    errors="coerce"
)

df = df.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 3. CREATE PRICE FEATURES PER STOCK CODE
# --------------------------------------------------

print("Creating historical price features...")

print("Creating historical price features...")

# Group the data by stock code.
# This ensures that one stock's prices are never mixed with another stock's prices.
grouped_prices = df.groupby("code", sort=False)["daily_sequence_price"]

# Previous price observations
df["lag_1"] = grouped_prices.shift(1)
df["lag_5"] = grouped_prices.shift(5)
df["lag_10"] = grouped_prices.shift(10)

# Percentage returns
df["daily_return"] = (
    (df["daily_sequence_price"] / df["lag_1"]) - 1
) * 100

df["return_5d"] = (
    (df["daily_sequence_price"] / df["lag_5"]) - 1
) * 100

df["return_10d"] = (
    (df["daily_sequence_price"] / df["lag_10"]) - 1
) * 100

# --------------------------------------------------
# 4. SAVE FEATURE DATASET
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR / "nse_price_features.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 5. FEATURE QUALITY SUMMARY
# --------------------------------------------------

feature_columns = [
    "lag_1",
    "lag_5",
    "lag_10",
    "daily_return",
    "return_5d",
    "return_10d"
]

summary = []

for column in feature_columns:

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

summary_df.to_csv(
    REPORT_DIR / "price_feature_summary.csv",
    index=False
)


# --------------------------------------------------
# 6. PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 75)
print("PRICE FEATURE ENGINEERING SUMMARY")
print("=" * 75)

print(f"Total records: {len(df):,}")
print(f"Stock codes: {df['code'].nunique()}")

print("\nFEATURE AVAILABILITY")
print("-" * 75)

print(summary_df.to_string(index=False))

print("\nSAMPLE FEATURES: SCOM")
print("-" * 75)

sample = df[df["code"] == "SCOM"]

print(
    sample[
        [
            "date",
            "daily_sequence_price",
            "lag_1",
            "lag_5",
            "lag_10",
            "daily_return",
            "return_5d",
            "return_10d"
        ]
    ].tail(15).to_string(index=False)
)

print("\nSAVED FILES")
print("-" * 75)

print(OUTPUT_FILE)
print(REPORT_DIR / "price_feature_summary.csv")

print("\nPrice feature engineering completed successfully.")
