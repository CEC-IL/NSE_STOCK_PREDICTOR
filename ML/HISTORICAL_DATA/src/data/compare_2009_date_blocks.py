from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

FILE_PATH = BASE_DIR / "data" / "raw" / "NSE_data_all_stocks_2009.csv"

df = pd.read_csv(FILE_PATH)

# Standardize column names for this investigation
df.columns = [col.strip().lower() for col in df.columns]

# Convert dates
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    dayfirst=False
)

# Dates that we are investigating
problem_dates = [
    "2009-01-29",
    "2009-04-02",
    "2009-10-08"
]

# Columns used to compare records
comparison_columns = [
    "code",
    "day price",
    "previous",
    "volume",
    "change",
    "change%",
]

# Keep only columns that exist
comparison_columns = [
    col for col in comparison_columns
    if col in df.columns
]

print("\nCOMPARING PROBLEMATIC 2009 DATE BLOCKS")
print("=" * 80)

for date_text in problem_dates:

    target_date = pd.Timestamp(date_text)

    matching = df[df["date"] == target_date].copy()

    print(f"\n\nDATE: {date_text}")
    print("-" * 80)

    print("Total records:", len(matching))

    if matching.empty:
        print("No records found.")
        continue

    # Display the records for this date
    print(
        matching[comparison_columns].to_string(index=True)
    )

    # Check whether individual company codes appear more than once
    duplicate_codes = matching[
        matching["code"].duplicated(keep=False)
    ]

    print("\nRepeated company codes:")

    if duplicate_codes.empty:
        print("None")
    else:
        print(
            duplicate_codes[comparison_columns].to_string(index=True)
        )

print("\nComparison complete.")