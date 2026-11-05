
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
    / "nse_all_stocks_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def get_consistent_price(values):
    """
    Return a daily price only when every record
    for that code/date has the same non-missing price.

    Conflicting prices are not averaged or guessed.
    """
    if values.isna().any():
        return None

    unique_prices = values.unique()

    if len(unique_prices) == 1:
        return unique_prices[0]

    return None


def format_price_values(values):
    """Show all prices found within a date group."""
    formatted = []

    for value in values:
        if pd.isna(value):
            formatted.append("MISSING")
        else:
            formatted.append(str(value))

    return " | ".join(formatted)


# --------------------------------------------------
# 3. LOAD CLEANED DATA
# --------------------------------------------------

print("\nLoading cleaned NSE data...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["day_price"] = pd.to_numeric(
    df["day_price"],
    errors="coerce"
)

# Preserve the position of every row in the input CSV.
# This is an identifier, not a new market timestamp.
df["source_row_number"] = range(1, len(df) + 1)

# Sort in chronological order, retaining original order
# within each code/date group.
df = df.sort_values(
    ["code", "date", "source_row_number"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 4. IDENTIFY REPEATED RECORDS
# --------------------------------------------------

df["duplicate_occurrence"] = (
    df.groupby(["code", "date"], sort=False)
    .cumcount() + 1
)


# --------------------------------------------------
# 5. CREATE ONE ROW PER CODE AND DATE
# --------------------------------------------------

print("Building the daily sequence...")

daily = (
    df.groupby(
        ["code", "date"],
        sort=True,
        as_index=False
    )
    .agg(
        name=("name", "first"),

        records_on_date=("code", "size"),

        unique_day_prices=(
            "day_price",
            lambda values: values.nunique(dropna=True)
        ),

        has_missing_day_price=(
            "day_price",
            lambda values: values.isna().any()
        ),

        daily_sequence_price=(
            "day_price",
            get_consistent_price
        ),

        day_price_values=(
            "day_price",
            format_price_values
        )
    )
)


# --------------------------------------------------
# 6. FLAG AMBIGUOUS PRICES
# --------------------------------------------------

daily["is_repeated_date"] = (
    daily["records_on_date"] > 1
)

daily["has_conflicting_prices"] = (
    daily["unique_day_prices"] > 1
)

daily["sequence_price_ambiguous"] = (
    daily["is_repeated_date"]
    & (
        daily["has_conflicting_prices"]
        | daily["has_missing_day_price"]
        | (daily["unique_day_prices"] != 1)
    )
)


# --------------------------------------------------
# 7. CREATE CHRONOLOGICAL INDEX
# --------------------------------------------------

# Every unique code/date receives one sequence number.
# Two records on the same date share the same index.

daily = daily.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)

daily["trading_day_index"] = (
    daily.groupby("code")
    .cumcount() + 1
)


# --------------------------------------------------
# 8. CREATE DATE SAMPLE WEIGHTS
# --------------------------------------------------

# Unique dates receive weight 1.
# Two records on the same date each receive weight 0.5.
# Their combined weight remains 1.

daily["date_sample_weight"] = (
    1 / daily["records_on_date"]
)


# --------------------------------------------------
# 9. ADD DAILY INFORMATION TO EVERY ORIGINAL ROW
# --------------------------------------------------

sequence_columns = [
    "code",
    "date",
    "records_on_date",
    "unique_day_prices",
    "trading_day_index",
    "daily_sequence_price",
    "is_repeated_date",
    "has_conflicting_prices",
    "sequence_price_ambiguous",
    "date_sample_weight"
]

feature_base = df.merge(
    daily[sequence_columns],
    on=["code", "date"],
    how="left",
    validate="many_to_one"
)

# Restore the original row order.
feature_base = feature_base.sort_values(
    "source_row_number",
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 10. SAVE DATASETS
# --------------------------------------------------

feature_base_file = (
    OUTPUT_DIR / "nse_feature_base.csv"
)

daily_sequence_file = (
    OUTPUT_DIR / "nse_daily_sequence.csv"
)

ambiguous_dates_file = (
    REPORT_DIR / "feature_engineering_ambiguous_dates.csv"
)

feature_base.to_csv(
    feature_base_file,
    index=False
)

daily.to_csv(
    daily_sequence_file,
    index=False
)

daily[
    daily["sequence_price_ambiguous"]
].to_csv(
    ambiguous_dates_file,
    index=False
)


# --------------------------------------------------
# 11. PRINT SUMMARY
# --------------------------------------------------

repeated_groups = daily[
    daily["is_repeated_date"]
]

conflicting_groups = daily[
    daily["has_conflicting_prices"]
]

same_price_groups = repeated_groups[
    (repeated_groups["unique_day_prices"] == 1)
    & (~repeated_groups["has_missing_day_price"])
]

print("\n" + "=" * 75)
print("FEATURE BASE PREPARATION SUMMARY")
print("=" * 75)

print(f"Original records: {len(df):,}")
print(f"Feature base records: {len(feature_base):,}")
print(f"Unique stock codes: {feature_base['code'].nunique()}")
print(f"Unique code/date observations: {len(daily):,}")

print("\nREPEATED DATE SUMMARY")
print("-" * 75)

print(f"Repeated date groups: {len(repeated_groups)}")
print(
    "Records belonging to repeated groups:",
    repeated_groups["records_on_date"].sum()
)

print(f"Groups with conflicting prices: {len(conflicting_groups)}")
print(f"Groups with the same available price: {len(same_price_groups)}")

print("\nDAILY SEQUENCE SUMMARY")
print("-" * 75)

print(
    "Ambiguous repeated dates:",
    daily["sequence_price_ambiguous"].sum()
)

print(
    "Missing daily sequence prices:",
    daily["daily_sequence_price"].isna().sum()
)

print("\nSAVED FILES")
print("-" * 75)

print(feature_base_file)
print(daily_sequence_file)
print(ambiguous_dates_file)

print("\nFeature base preparation completed successfully.")
