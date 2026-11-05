
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "nse_all_stocks_cleaned.csv"
)

REPORT_DIR = BASE_DIR / "reports"

# Load the cleaned dataset
df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Find repeated date and stock-code combinations
duplicate_mask = df.duplicated(
    subset=["date", "code"],
    keep=False
)

duplicates = df[duplicate_mask].copy()

# Count how many distinct date/code groups are repeated
duplicate_groups = (
    duplicates.groupby(["date", "code"])
    .size()
    .reset_index(name="record_count")
)

# Sort for easier inspection
duplicates = duplicates.sort_values(
    ["code", "date"]
)

duplicate_groups = duplicate_groups.sort_values(
    ["code", "date"]
)

# Save detailed records and group summary
duplicates.to_csv(
    REPORT_DIR / "duplicate_records_detailed.csv",
    index=False
)

duplicate_groups.to_csv(
    REPORT_DIR / "duplicate_groups_summary.csv",
    index=False
)

# Display results
print("DUPLICATE INVESTIGATION")
print("=" * 50)

print(f"Rows involved: {len(duplicates)}")
print(f"Repeated date/code groups: {len(duplicate_groups)}")

print("\nLargest repeated groups:")
print(
    duplicate_groups
    .sort_values("record_count", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nFirst 30 repeated records:")
print(
    duplicates[
        ["date", "code", "name", "day_price", "previous", "volume"]
    ].head(30).to_string(index=False)
)

print("\nReports saved in:", REPORT_DIR)
