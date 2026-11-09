
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "nse_prediction_targets.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "modeling"
REPORT_DIR = ROOT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading prediction dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "code"]
).reset_index(drop=True)

print(f"Total records: {len(df):,}")
print(f"Unique stock codes: {df['code'].nunique()}")


# --------------------------------------------------
# 3. DEFINE MODEL FEATURES
# --------------------------------------------------

feature_columns = [
    "daily_sequence_price",

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
    "volatility_20d",

    "code"
]

required_columns = (
    feature_columns
    + [
        "date",
        "target_next_price",
        "target_direction",
        "target_direction_label"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# 4. CALCULATE NEXT OBSERVATION DATE
# --------------------------------------------------

print("\nCalculating next observation dates...")

df["target_next_date"] = (
    df.groupby("code", sort=False)["date"]
    .shift(-1)
)


# --------------------------------------------------
# 5. CALCULATE CHRONOLOGICAL SPLIT BOUNDARIES
# --------------------------------------------------

print("\nCalculating 70/15/15 chronological split...")

unique_dates = sorted(df["date"].dropna().unique())

total_dates = len(unique_dates)

train_end_index = int(total_dates * 0.70) - 1
validation_end_index = int(total_dates * 0.85) - 1

train_end_date = pd.Timestamp(
    unique_dates[train_end_index]
)

validation_end_date = pd.Timestamp(
    unique_dates[validation_end_index]
)

first_date = pd.Timestamp(unique_dates[0])
last_date = pd.Timestamp(unique_dates[-1])

print("\nDATE BOUNDARIES")
print("=" * 65)

print(f"First historical date:       {first_date.date()}")
print(f"Training ends:               {train_end_date.date()}")
print(f"Validation ends:             {validation_end_date.date()}")
print(f"Last historical date:        {last_date.date()}")


# --------------------------------------------------
# 6. CREATE SPLIT MASKS
# --------------------------------------------------

has_next_date = df["target_next_date"].notna()

train_mask = (
    (df["date"] <= train_end_date)
    & has_next_date
    & (df["target_next_date"] <= train_end_date)
)

validation_mask = (
    (df["date"] > train_end_date)
    & (df["date"] <= validation_end_date)
    & has_next_date
    & (df["target_next_date"] <= validation_end_date)
)

test_mask = (
    (df["date"] > validation_end_date)
    & has_next_date
    & (df["target_next_date"] > validation_end_date)
)


# --------------------------------------------------
# 7. PREPARE THE THREE DATASETS
# --------------------------------------------------

train_df = df.loc[train_mask].copy()

validation_df = df.loc[validation_mask].copy()

test_df = df.loc[test_mask].copy()


# --------------------------------------------------
# 8. DEFINE PRICE AND DIRECTION DATASETS
# --------------------------------------------------

price_columns = (
    ["date", "target_next_date"]
    + feature_columns
    + ["target_next_price"]
)

direction_columns = (
    ["date", "target_next_date"]
    + feature_columns
    + [
        "target_direction",
        "target_direction_label"
    ]
)


# --------------------------------------------------
# 9. REMOVE ROWS WITH MISSING TARGETS
# --------------------------------------------------

price_train = train_df.dropna(
    subset=["target_next_price"]
)[price_columns].copy()

price_validation = validation_df.dropna(
    subset=["target_next_price"]
)[price_columns].copy()

price_test = test_df.dropna(
    subset=["target_next_price"]
)[price_columns].copy()


direction_train = train_df.dropna(
    subset=["target_direction"]
)[direction_columns].copy()

direction_validation = validation_df.dropna(
    subset=["target_direction"]
)[direction_columns].copy()

direction_test = test_df.dropna(
    subset=["target_direction"]
)[direction_columns].copy()


# --------------------------------------------------
# 10. SAVE DATASETS
# --------------------------------------------------

datasets = {
    "price_train.csv": price_train,
    "price_validation.csv": price_validation,
    "price_test.csv": price_test,

    "direction_train.csv": direction_train,
    "direction_validation.csv": direction_validation,
    "direction_test.csv": direction_test
}

print("\nSaving datasets...")

for filename, dataset in datasets.items():

    output_path = OUTPUT_DIR / filename

    dataset.to_csv(output_path, index=False)

    print(
        f"{filename}: {len(dataset):,} records"
    )


# --------------------------------------------------
# 11. GENERATE SPLIT REPORT
# --------------------------------------------------

report_lines = []

report_lines.append("NSE MODELING DATASET SPLIT REPORT")
report_lines.append("=" * 65)

report_lines.append(
    f"Total historical dates: {total_dates:,}"
)

report_lines.append(
    f"Total original observations: {len(df):,}"
)

report_lines.append(
    f"Total stock codes: {df['code'].nunique()}"
)

report_lines.append("")

report_lines.append(
    f"Training period: {first_date.date()} to {train_end_date.date()}"
)

report_lines.append(
    f"Validation period: "
    f"after {train_end_date.date()} "
    f"to {validation_end_date.date()}"
)

report_lines.append(
    f"Testing period: after {validation_end_date.date()} "
    f"to {last_date.date()}"
)

report_lines.append("")

for filename, dataset in datasets.items():

    report_lines.append(
        f"{filename}: {len(dataset):,} records"
    )

report_lines.append("")

report_lines.append(
    f"Training feature dates: "
    f"{price_train['date'].min()} to "
    f"{price_train['date'].max()}"
)

report_lines.append(
    f"Validation feature dates: "
    f"{price_validation['date'].min()} to "
    f"{price_validation['date'].max()}"
)

report_lines.append(
    f"Testing feature dates: "
    f"{price_test['date'].min()} to "
    f"{price_test['date'].max()}"
)

report_lines.append("")

report_lines.append(
    "Missing feature values have been preserved."
)

report_lines.append(
    "No random splitting or feature imputation was performed."
)

report_path = REPORT_DIR / "modeling_split_summary.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# --------------------------------------------------
# 12. CHECK FOR DATE LEAKAGE
# --------------------------------------------------

assert (
    price_train["target_next_date"] <= train_end_date
).all()

assert (
    price_validation["date"] > train_end_date
).all()

assert (
    price_validation["target_next_date"] <= validation_end_date
).all()

assert (
    price_test["date"] > validation_end_date
).all()

assert (
    price_test["target_next_date"] > validation_end_date
).all()


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\nSPLIT SUMMARY")
print("=" * 65)

print(f"Training price records:       {len(price_train):,}")
print(f"Validation price records:     {len(price_validation):,}")
print(f"Testing price records:        {len(price_test):,}")

print()

print(f"Training direction records:   {len(direction_train):,}")
print(f"Validation direction records: {len(direction_validation):,}")
print(f"Testing direction records:    {len(direction_test):,}")

print("\nDate leakage checks: PASSED")

print("\nSplit preparation completed successfully.")

print(f"\nReport saved to: {report_path}")
print(f"Datasets saved to: {OUTPUT_DIR}")
