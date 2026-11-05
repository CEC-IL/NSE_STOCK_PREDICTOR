
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "processed" / "nse_all_stocks_cleaned.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned historical data
df = pd.read_csv(DATA_FILE)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sort records chronologically
df = df.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)

print("\n" + "=" * 75)
print("FEATURE ENGINEERING: DATA INSPECTION")
print("=" * 75)

# --------------------------------------------------
# 1. GENERAL INFORMATION
# --------------------------------------------------

print("\n1. GENERAL INFORMATION")
print("-" * 75)

print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Unique codes: {df['code'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# --------------------------------------------------
# 2. STOCK CODE INVENTORY
# --------------------------------------------------

print("\n2. STOCK CODE INVENTORY")
print("-" * 75)

code_summary = (
    df.groupby("code")
    .agg(
        name=("name", "first"),
        records=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        repeated_dates=("date", lambda x: x.duplicated(keep=False).sum())
    )
    .reset_index()
    .sort_values("code")
)

print(code_summary.to_string(index=False))

code_summary.to_csv(
    REPORT_DIR / "feature_engineering_code_inventory.csv",
    index=False
)

# --------------------------------------------------
# 3. POSSIBLE MARKET INDICES
# --------------------------------------------------

print("\n3. CODES THAT MAY REPRESENT MARKET INDICES")
print("-" * 75)

possible_indices = code_summary[
    code_summary["code"].astype(str).str.startswith("^")
]

if len(possible_indices) > 0:
    print(possible_indices[["code", "name", "records"]].to_string(index=False))
else:
    print("No codes beginning with '^' were found.")

# --------------------------------------------------
# 4. REPEATED DATE AND CODE RECORDS
# --------------------------------------------------

print("\n4. REPEATED DATE/CODE RECORDS")
print("-" * 75)

duplicate_mask = df.duplicated(
    subset=["code", "date"],
    keep=False
)

repeated_records = df[duplicate_mask].copy()

print(f"Rows involved in repeated date/code groups: {len(repeated_records)}")

print(
    "Number of repeated date/code groups:",
    repeated_records.groupby(["code", "date"]).ngroups
)

if not repeated_records.empty:
    print("\nRepeated records grouped by year:")

    repeated_records["year"] = repeated_records["date"].dt.year

    print(
        repeated_records.groupby("year")
        .size()
        .to_string()
    )

    print("\nRepeated records grouped by code:")

    print(
        repeated_records.groupby("code")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    # Save all repeated records for inspection
    repeated_records.to_csv(
        REPORT_DIR / "repeated_date_code_before_features.csv",
        index=False
    )

# --------------------------------------------------
# 5. CHECK PRICE AVAILABILITY
# --------------------------------------------------

print("\n5. DAY PRICE AVAILABILITY")
print("-" * 75)

print(f"Missing day_price values: {df['day_price'].isna().sum():,}")
print(f"Available day_price values: {df['day_price'].notna().sum():,}")

print("\nMissing day_price records by code:")

missing_prices = (
    df[df["day_price"].isna()]
    .groupby("code")
    .size()
    .sort_values(ascending=False)
)

print(missing_prices.to_string() if not missing_prices.empty else "None")

# --------------------------------------------------
# 6. SAVE REPORT
# --------------------------------------------------

print("\n6. REPORTS SAVED")
print("-" * 75)

print(REPORT_DIR / "feature_engineering_code_inventory.csv")
print(REPORT_DIR / "repeated_date_code_before_features.csv")

print("\nInspection completed successfully.")
