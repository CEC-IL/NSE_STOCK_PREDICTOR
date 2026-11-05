
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "processed" / "nse_all_stocks_cleaned.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def show_unique_values(series):
    """Display distinct values from a repeated record group."""
    values = series.drop_duplicates()

    return " | ".join(
        "MISSING" if pd.isna(value) else str(value)
        for value in values
    )


# Load the cleaned dataset
df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Record the row position in the cleaned CSV
df["cleaned_row_number"] = range(len(df))

# Identify every record belonging to a repeated date/code group
duplicate_mask = df.duplicated(
    subset=["code", "date"],
    keep=False
)

duplicate_rows = df[duplicate_mask].copy()

# Group repeated records
duplicate_groups = (
    duplicate_rows
    .groupby(["code", "date"], sort=True)
    .agg(
        records=("code", "size"),

        distinct_day_prices=(
            "day_price",
            lambda x: x.nunique(dropna=False)
        ),

        day_price_values=("day_price", show_unique_values),
        previous_values=("previous", show_unique_values),
        change_values=("change", show_unique_values),
        change_pct_values=("change_pct", show_unique_values),
        volume_values=("volume", show_unique_values),

        cleaned_row_numbers=(
            "cleaned_row_number",
            show_unique_values
        )
    )
    .reset_index()
)

# --------------------------------------------------
# 1. REPEATED DATES
# --------------------------------------------------

print("\n" + "=" * 75)
print("REPEATED RECORD DETAILS")
print("=" * 75)

print(f"\nTotal repeated groups: {len(duplicate_groups)}")
print(f"Total repeated records: {len(duplicate_rows)}")

print("\n1. REPEATED DATES")
print("-" * 75)

date_summary = (
    duplicate_groups
    .groupby("date")
    .agg(
        repeated_groups=("code", "size"),
        affected_records=("records", "sum"),
        affected_codes=("code", show_unique_values)
    )
    .reset_index()
)

print(date_summary.to_string(index=False))

# --------------------------------------------------
# 2. PRICE CONFLICTS
# --------------------------------------------------

print("\n2. DAY_PRICE CONFLICTS")
print("-" * 75)

conflicts = duplicate_groups[
    duplicate_groups["distinct_day_prices"] > 1
]

same_prices = duplicate_groups[
    duplicate_groups["distinct_day_prices"] == 1
]

print(f"Groups with different day_price values: {len(conflicts)}")
print(f"Groups with identical day_price values: {len(same_prices)}")

if not conflicts.empty:
    print("\nExamples of groups with different prices:")

    print(
        conflicts[
            [
                "code",
                "date",
                "day_price_values",
                "previous_values",
                "change_pct_values",
                "volume_values"
            ]
        ].head(20).to_string(index=False)
    )

# --------------------------------------------------
# 3. ALL DUPLICATE GROUP DETAILS
# --------------------------------------------------

print("\n3. ALL REPEATED GROUPS")
print("-" * 75)

print(
    duplicate_groups[
        [
            "code",
            "date",
            "records",
            "distinct_day_prices",
            "day_price_values",
            "previous_values",
            "volume_values"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 4. SAVE REPORTS
# --------------------------------------------------

duplicate_groups.to_csv(
    REPORT_DIR / "repeated_date_code_details.csv",
    index=False
)

duplicate_rows.to_csv(
    REPORT_DIR / "repeated_date_code_full_rows.csv",
    index=False
)

date_summary.to_csv(
    REPORT_DIR / "repeated_date_code_date_summary.csv",
    index=False
)

print("\n4. REPORTS SAVED")
print("-" * 75)

print("reports/repeated_date_code_details.csv")
print("reports/repeated_date_code_full_rows.csv")
print("reports/repeated_date_code_date_summary.csv")

print("\nInspection completed successfully.")
