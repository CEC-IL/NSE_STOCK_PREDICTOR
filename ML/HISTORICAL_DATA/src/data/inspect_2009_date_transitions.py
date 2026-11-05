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

date_column = next(
    col for col in df.columns
    if col.strip().lower() == "date"
)

code_column = next(
    col for col in df.columns
    if col.strip().lower() == "code"
)

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=False
)

# Find every point where the date changes
date_changes = (
    df[date_column] != df[date_column].shift()
)

transition_positions = df.index[date_changes].tolist()

print("\nDATE TRANSITIONS AROUND THE PROBLEMATIC RECORDS")
print("=" * 75)

for position in transition_positions:

    if 500 <= position <= 1200 or 8400 <= position <= 9000 or 10800 <= position <= 11300:

        start = max(0, position - 2)
        end = min(len(df), position + 3)

        print(f"\nTransition near original row {position}")

        print(
            df.iloc[start:end][
                [date_column, code_column]
            ].to_string(index=True)
        )

print("\nInvestigation complete.")