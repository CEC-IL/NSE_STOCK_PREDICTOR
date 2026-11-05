
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
    / "nse_technical_features.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading technical features...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["daily_sequence_price"] = pd.to_numeric(
    df["daily_sequence_price"],
    errors="coerce"
)

df = df.sort_values(
    ["code", "date"],
    kind="stable"
).reset_index(drop=True)

print(f"Loaded observations: {len(df):,}")
print(f"Stock codes: {df['code'].nunique()}")


# --------------------------------------------------
# 3. CREATE NEXT-OBSERVATION PRICE TARGET
# --------------------------------------------------

print("\nCreating next-observation price targets...")

grouped_prices = df.groupby("code", sort=False)[
    "daily_sequence_price"
]

df["target_next_price"] = grouped_prices.shift(-1)


# --------------------------------------------------
# 4. CREATE DIRECTION TARGET
# --------------------------------------------------

print("Creating direction targets...")

current_price = df["daily_sequence_price"]
next_price = df["target_next_price"]

# Start with missing values.
df["target_direction"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64"
)

# UP = 1
df.loc[
    current_price.notna()
    & next_price.notna()
    & (next_price > current_price),
    "target_direction"
] = 1

# DOWN = -1
df.loc[
    current_price.notna()
    & next_price.notna()
    & (next_price < current_price),
    "target_direction"
] = -1

# UNCHANGED = 0
df.loc[
    current_price.notna()
    & next_price.notna()
    & (next_price == current_price),
    "target_direction"
] = 0


# --------------------------------------------------
# 5. CREATE DIRECTION LABELS
# --------------------------------------------------

direction_names = {
    1: "UP",
    -1: "DOWN",
    0: "UNCHANGED"
}

df["target_direction_label"] = (
    df["target_direction"]
    .map(direction_names)
)


# --------------------------------------------------
# 6. CHECK TARGET AVAILABILITY
# --------------------------------------------------

print("\nChecking target availability...")

total_records = len(df)

available_price_targets = (
    df["target_next_price"].notna().sum()
)

missing_price_targets = (
    df["target_next_price"].isna().sum()
)

available_direction_targets = (
    df["target_direction"].notna().sum()
)

missing_direction_targets = (
    df["target_direction"].isna().sum()
)


# --------------------------------------------------
# 7. DIRECTION DISTRIBUTION
# --------------------------------------------------

direction_counts = (
    df["target_direction_label"]
    .value_counts(dropna=False)
    .rename_axis("direction")
    .reset_index(name="records")
)

direction_counts["percentage"] = (
    direction_counts["records"] / total_records
) * 100

direction_counts["percentage"] = (
    direction_counts["percentage"].round(2)
)

direction_counts.to_csv(
    REPORT_DIR / "direction_target_distribution.csv",
    index=False
)


# --------------------------------------------------
# 8. SAVE TARGET DATASET
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR / "nse_prediction_targets.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 9. DISPLAY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 75)
print("PREDICTION TARGET ENGINEERING SUMMARY")
print("=" * 75)

print(f"Total observations: {total_records:,}")

print("\nNEXT-OBSERVATION PRICE TARGET")
print("-" * 75)

print(f"Available targets: {available_price_targets:,}")
print(f"Missing targets:   {missing_price_targets:,}")

print("\nDIRECTION TARGET")
print("-" * 75)

print(f"Available targets: {available_direction_targets:,}")
print(f"Missing targets:   {missing_direction_targets:,}")

print("\nDIRECTION DISTRIBUTION")
print("-" * 75)

print(direction_counts.to_string(index=False))


# --------------------------------------------------
# 10. DISPLAY SCOM SAMPLE
# --------------------------------------------------

print("\nSAMPLE PREDICTION TARGETS: SCOM")
print("-" * 75)

sample = df[df["code"] == "SCOM"]

print(
    sample[
        [
            "date",
            "daily_sequence_price",
            "target_next_price",
            "target_direction",
            "target_direction_label"
        ]
    ].tail(15).to_string(index=False)
)


# --------------------------------------------------
# 11. SAVE COMPLETION
# --------------------------------------------------

print("\nSAVED FILES")
print("-" * 75)

print(OUTPUT_FILE)
print(REPORT_DIR / "direction_target_distribution.csv")

print("\nPrediction target engineering completed successfully.")
