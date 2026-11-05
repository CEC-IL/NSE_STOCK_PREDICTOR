
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nse_prediction_targets.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading prediction dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_columns = [
    "daily_sequence_price",
    "target_next_price",
    "target_direction",
    "lag_1",
    "lag_5",
    "lag_10",
    "daily_return",
    "return_5d",
    "return_10d",
    "sma_5",
    "sma_10",
    "sma_20",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

df = df.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 3. BASIC DATASET CHECKS
# --------------------------------------------------

print("Checking dataset structure...")

total_records = len(df)
total_codes = df["code"].nunique()

duplicate_code_dates = df.duplicated(
    subset=["code", "date"]
).sum()

missing_codes = df["code"].isna().sum()
missing_dates = df["date"].isna().sum()

date_order_checks = (
    df.groupby("code")["date"]
    .apply(lambda dates: dates.is_monotonic_increasing)
)

incorrect_date_order = (~date_order_checks).sum()


# --------------------------------------------------
# 4. VALIDATE NEXT-PRICE TARGETS
# --------------------------------------------------

print("Validating next-price targets...")

expected_next_price = (
    df.groupby("code", sort=False)["daily_sequence_price"]
    .shift(-1)
)

actual_next_price = df["target_next_price"]

price_target_matches = (
    np.isclose(
        actual_next_price,
        expected_next_price,
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True
    )
)

price_target_mismatches = (
    (~price_target_matches).sum()
)


# --------------------------------------------------
# 5. VALIDATE DIRECTION TARGETS
# --------------------------------------------------

print("Validating direction targets...")

current_price = df["daily_sequence_price"]
next_price = df["target_next_price"]

expected_direction = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64"
)

valid_prices = (
    current_price.notna()
    & next_price.notna()
)

expected_direction.loc[
    valid_prices & (next_price > current_price)
] = 1

expected_direction.loc[
    valid_prices & (next_price < current_price)
] = -1

expected_direction.loc[
    valid_prices & (next_price == current_price)
] = 0

actual_direction = df["target_direction"].astype("Int64")

direction_target_matches = (
    actual_direction.eq(expected_direction)
    | (
        actual_direction.isna()
        & expected_direction.isna()
    )
)

direction_target_mismatches = (
    (~direction_target_matches).sum()
)


# --------------------------------------------------
# 6. CHECK DIRECTION LABELS
# --------------------------------------------------

expected_labels = {
    1: "UP",
    -1: "DOWN",
    0: "UNCHANGED"
}

expected_label = (
    actual_direction.map(expected_labels)
)

actual_label = df["target_direction_label"]

label_matches = (
    expected_label.eq(actual_label)
    | (
        expected_label.isna()
        & actual_label.isna()
    )
)

label_mismatches = (~label_matches).sum()


# --------------------------------------------------
# 7. FEATURE MISSING-VALUE REPORT
# --------------------------------------------------

print("Checking feature availability...")

feature_columns = [
    "lag_1",
    "lag_5",
    "lag_10",
    "daily_return",
    "return_5d",
    "return_10d",
    "sma_5",
    "sma_10",
    "sma_20",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d"
]

missing_summary = []

for column in feature_columns:

    if column not in df.columns:
        continue

    missing_count = df[column].isna().sum()

    missing_summary.append({
        "feature": column,
        "available_values": df[column].notna().sum(),
        "missing_values": missing_count,
        "missing_percentage": round(
            df[column].isna().mean() * 100,
            2
        )
    })

missing_df = pd.DataFrame(missing_summary)

missing_df.to_csv(
    REPORT_DIR / "modeling_feature_missing_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. TARGET DISTRIBUTION
# --------------------------------------------------

print("Checking target distribution...")

direction_distribution = (
    df["target_direction_label"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 9. SAVE VALIDATION REPORT
# --------------------------------------------------

report_lines = [
    "=" * 75,
    "NSE MODELLING DATASET VALIDATION",
    "=" * 75,
    "",
    f"Total records: {total_records:,}",
    f"Unique stock codes: {total_codes}",
    f"Duplicate code/date records: {duplicate_code_dates}",
    f"Missing stock codes: {missing_codes}",
    f"Missing dates: {missing_dates}",
    f"Stock codes with incorrect date order: {incorrect_date_order}",
    "",
    "NEXT-PRICE TARGET VALIDATION",
    "-" * 75,
    f"Mismatched next-price targets: {price_target_mismatches}",
    "",
    "DIRECTION TARGET VALIDATION",
    "-" * 75,
    f"Mismatched direction targets: {direction_target_mismatches}",
    f"Mismatched direction labels: {label_mismatches}",
    "",
    "DIRECTION DISTRIBUTION",
    "-" * 75
]

for label, count in direction_distribution.items():
    report_lines.append(f"{label}: {count:,}")

report_lines.extend([
    "",
    "FEATURE AVAILABILITY",
    "-" * 75,
    missing_df.to_string(index=False),
    "",
    "=" * 75
])

validation_report = "\n".join(report_lines)

report_file = (
    REPORT_DIR / "modeling_dataset_validation.txt"
)

with open(report_file, "w", encoding="utf-8") as file:
    file.write(validation_report)


# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + validation_report)

print("\nSAVED REPORTS")
print("-" * 75)

print(report_file)

print(
    REPORT_DIR / "modeling_feature_missing_summary.csv"
)

print("\nDataset validation completed successfully.")
