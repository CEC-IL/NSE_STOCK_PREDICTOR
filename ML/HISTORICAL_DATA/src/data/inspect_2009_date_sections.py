from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

FILE_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "NSE_data_all_stocks_2009.csv"
)

df = pd.read_csv(FILE_PATH)

# Find the date column regardless of capitalization
date_column = next(
    col for col in df.columns
    if col.strip().lower() == "date"
)

# Find the stock-code column
code_column = next(
    col for col in df.columns
    if col.strip().lower() == "code"
)

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=False
)

dates_to_check = [
    "2009-01-29",
    "2009-04-02",
    "2009-10-08"
]

print("\n2009 DATE SECTION INVESTIGATION")
print("=" * 70)

for date in dates_to_check:

    target_date = pd.Timestamp(date)

    positions = df.index[
        df[date_column] == target_date
    ].tolist()

    print(f"\nDATE: {date}")
    print("-" * 70)

    for position in positions:

        start = max(0, position - 3)
        end = min(len(df), position + 4)

        section = df.iloc[start:end].copy()

        print(
            f"\nOriginal rows {start} to {end - 1}:"
        )

        print(
            section[
                [date_column, code_column]
            ].to_string(index=True)
        )

print("\nInvestigation complete.")