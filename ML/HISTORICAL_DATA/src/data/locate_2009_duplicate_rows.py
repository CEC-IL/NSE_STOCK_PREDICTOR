
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = (
    BASE_DIR / "data" / "raw"
    / "NSE_data_all_stocks_2009.csv"
)

REPORT_DIR = BASE_DIR / "reports"

df = pd.read_csv(RAW_FILE)

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("%", "pct", regex=False)
)

# Preserve original CSV row positions
df.insert(0, "source_row", range(1, len(df) + 1))

# Parse dates
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce",
    dayfirst=False
)

# Find all repeated date/code combinations
duplicate_mask = df.duplicated(
    subset=["date", "code"],
    keep=False
)

duplicates = df[duplicate_mask].copy()

# Keep the relevant columns
columns = [
    "source_row",
    "date",
    "code",
    "name",
    "day_price",
    "previous",
    "volume"
]

duplicates = duplicates[columns].sort_values(
    ["date", "source_row"]
)

# Save the results
output_file = REPORT_DIR / "2009_duplicate_rows_with_positions.csv"

duplicates.to_csv(output_file, index=False)

# Display every repeated record
print("\nALL REPEATED RECORDS IN ORIGINAL FILE")
print("=" * 80)

print(duplicates.to_string(index=False))

# Show the original row-number range for each date
print("\nROW NUMBER RANGES")
print("=" * 80)

print(
    duplicates.groupby("date")["source_row"]
    .agg(["count", "min", "max"])
    .to_string()
)

print("\nReport saved to:", output_file)
