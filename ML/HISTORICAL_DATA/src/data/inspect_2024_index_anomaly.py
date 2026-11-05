
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "NSE_data_all_stocks_2024.csv"
)

raw = pd.read_csv(RAW_FILE)

# Standardize column names for easier access
raw.columns = [str(col).strip().lower() for col in raw.columns]

# Parse dates
raw["date"] = pd.to_datetime(
    raw["date"],
    format="mixed",
    errors="coerce",
    dayfirst=False
)

# Convert relevant numeric columns
numeric_columns = [
    "day price",
    "previous",
    "change",
    "change%",
    "day low",
    "day high"
]

for column in numeric_columns:
    if column in raw.columns:
        raw[column] = pd.to_numeric(
            raw[column]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

target_date = pd.Timestamp("2024-11-26")

# Inspect NSE 20-Share Index around the suspicious date
nearby = raw[
    (raw["code"] == "^N20I")
    & (raw["date"] >= target_date - pd.Timedelta(days=7))
    & (raw["date"] <= target_date + pd.Timedelta(days=7))
].sort_values("date")

print("\n" + "=" * 75)
print("ORIGINAL 2024 CSV: NSE 20-SHARE INDEX")
print("=" * 75)

print("\nAvailable columns:")
print(raw.columns.tolist())

print("\nRecords around 26 November 2024:")

print(nearby.to_string(index=False))

# Inspect the exact suspicious record
print("\n" + "=" * 75)
print("EXACT RECORD ON 26 NOVEMBER 2024")
print("=" * 75)

exact = nearby[nearby["date"] == target_date]

if exact.empty:
    print("Record not found in original CSV.")
else:
    print(exact.to_string(index=False))

print("\nInvestigation completed. No data was modified.")
