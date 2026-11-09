
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed" / "modeling"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. DATASET FILES
# --------------------------------------------------

files = {
    "price_train": "price_train.csv",
    "price_validation": "price_validation.csv",
    "price_test": "price_test.csv",

    "direction_train": "direction_train.csv",
    "direction_validation": "direction_validation.csv",
    "direction_test": "direction_test.csv",
}


datasets = {}

for name, filename in files.items():

    path = DATA_DIR / filename

    datasets[name] = pd.read_csv(
        path,
        low_memory=False
    )

    datasets[name]["date"] = pd.to_datetime(
        datasets[name]["date"]
    )

    datasets[name]["target_next_date"] = pd.to_datetime(
        datasets[name]["target_next_date"]
    )


# --------------------------------------------------
# 3. FEATURE COLUMNS
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
]


# --------------------------------------------------
# 4. BASIC DATASET SUMMARY
# --------------------------------------------------

print("\nMODELLING DATASET AUDIT")
print("=" * 75)

summary_rows = []

for name, data in datasets.items():

    summary_rows.append({
        "dataset": name,
        "records": len(data),
        "unique_codes": data["code"].nunique(),
        "first_date": data["date"].min().date(),
        "last_date": data["date"].max().date(),
        "missing_features": int(
            data[feature_columns].isna().sum().sum()
        ),
    })


summary = pd.DataFrame(summary_rows)

print("\nDATASET SUMMARY")
print(summary.to_string(index=False))


# --------------------------------------------------
# 5. STOCK CODE COVERAGE
# --------------------------------------------------

print("\n\nSTOCK CODE COVERAGE")
print("=" * 75)

all_codes = set()

for data in datasets.values():
    all_codes.update(data["code"].dropna().unique())

coverage_rows = []

for name, data in datasets.items():

    codes = set(data["code"].dropna().unique())

    coverage_rows.append({
        "dataset": name,
        "codes_present": len(codes),
        "codes_missing": len(all_codes - codes),
    })


coverage = pd.DataFrame(coverage_rows)

print(coverage.to_string(index=False))


# --------------------------------------------------
# 6. DIRECTION CLASS DISTRIBUTION
# --------------------------------------------------

print("\n\nDIRECTION CLASS DISTRIBUTION")
print("=" * 75)

direction_rows = []

for name in [
    "direction_train",
    "direction_validation",
    "direction_test"
]:

    data = datasets[name]

    counts = data["target_direction_label"].value_counts()

    percentages = (
        data["target_direction_label"]
        .value_counts(normalize=True)
        * 100
    )

    for label in ["UP", "DOWN", "UNCHANGED"]:

        direction_rows.append({
            "dataset": name,
            "class": label,
            "records": int(counts.get(label, 0)),
            "percentage": round(
                percentages.get(label, 0), 2
            ),
        })


direction_summary = pd.DataFrame(direction_rows)

print(direction_summary.to_string(index=False))


# --------------------------------------------------
# 7. MISSING FEATURE VALUES
# --------------------------------------------------

print("\n\nMISSING FEATURE VALUES")
print("=" * 75)

missing_rows = []

for name, data in datasets.items():

    for column in feature_columns:

        missing_count = int(data[column].isna().sum())

        missing_percentage = (
            missing_count / len(data) * 100
            if len(data) > 0 else 0
        )

        missing_rows.append({
            "dataset": name,
            "feature": column,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage, 2
            ),
        })


missing_summary = pd.DataFrame(missing_rows)

print(
    missing_summary.to_string(index=False)
)



# --------------------------------------------------
# 8. CHECK DATE OVERLAP AND TARGET LEAKAGE
# --------------------------------------------------

print("\n\nCHRONOLOGICAL DATE CHECK")
print("=" * 75)

price_train = datasets["price_train"]
price_validation = datasets["price_validation"]
price_test = datasets["price_test"]

checks = {

    "Training features end before validation features": (
        price_train["date"].max()
        < price_validation["date"].min()
    ),

    "Validation features end before testing features": (
        price_validation["date"].max()
        < price_test["date"].min()
    ),

    "Training targets remain before validation": (
        price_train["target_next_date"].max()
        < price_validation["date"].min()
    ),

    "Validation targets remain before testing": (
        price_validation["target_next_date"].max()
        < price_test["date"].min()
    ),

    "Testing targets remain after validation": (
        price_test["target_next_date"].min()
        > price_validation["date"].max()
    ),

    "All targets occur after their feature dates": (
        (
            price_train["target_next_date"]
            > price_train["date"]
        ).all()
        and
        (
            price_validation["target_next_date"]
            > price_validation["date"]
        ).all()
        and
        (
            price_test["target_next_date"]
            > price_test["date"]
        ).all()
    )
}


for description, passed in checks.items():

    status = "PASSED" if passed else "FAILED"

    print(f"{description}: {status}")


# --------------------------------------------------
# 9. SAVE AUDIT REPORTS
# --------------------------------------------------

summary.to_csv(
    REPORT_DIR / "modeling_dataset_audit.csv",
    index=False
)

coverage.to_csv(
    REPORT_DIR / "stock_code_coverage.csv",
    index=False
)

direction_summary.to_csv(
    REPORT_DIR / "direction_class_audit.csv",
    index=False
)

missing_summary.to_csv(
    REPORT_DIR / "modeling_missing_features_audit.csv",
    index=False
)


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

print("\n\nAUDIT COMPLETED")
print("=" * 75)

if all(checks.values()):
    print("All chronological checks passed.")
else:
    print("WARNING: One or more chronological checks failed.")

print("\nReports saved in:")
print(REPORT_DIR)
