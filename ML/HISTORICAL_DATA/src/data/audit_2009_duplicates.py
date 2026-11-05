from pathlib import Path
import pandas as pd

# Locate the project folders
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "NSE_data_all_stocks_2009.csv"
REPORT_DIR = BASE_DIR / "reports"

# Read the original 2009 dataset
df = pd.read_csv(RAW_FILE)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Parse dates
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce",
    dayfirst=False
)

# Dates where repeated records were found
dates_to_check = [
    "2009-01-29",
    "2009-04-02",
    "2009-10-08"
]

dates_to_check = pd.to_datetime(dates_to_check)

# Select all records from those dates
audit = df[df["date"].isin(dates_to_check)].copy()

# Count records per date
print("\nRECORDS PER DATE")
print("=" * 50)

print(
    audit.groupby("date")
    .size()
    .to_string()
)

# Count repeated stock codes per date
print("\nREPEATED STOCK CODES")
print("=" * 50)

repeated = audit[
    audit.duplicated(
        subset=["date", "code"],
        keep=False
    )
].copy()

print(
    repeated.groupby(["date", "code"])
    .size()
    .to_string()
)

# Inspect BAT and BBK specifically
print("\nBAT AND BBK RECORDS")
print("=" * 50)

print(
    audit[audit["code"].isin(["BAT", "BBK"])]
    .to_string(index=False)
)

# Save all records from the three dates
output_file = REPORT_DIR / "2009_duplicate_date_audit.csv"

audit.to_csv(output_file, index=False)

print("\nAudit saved to:", output_file)