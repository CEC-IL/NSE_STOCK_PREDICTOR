
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"

CLEANED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nse_all_stocks_cleaned.csv"
)

df = pd.read_csv(CLEANED_FILE)

# --------------------------------------------------
# 1. INVESTIGATE ORIGINAL COLUMN NAMES
# --------------------------------------------------

print("\n" + "=" * 70)
print("ORIGINAL CHANGE AND ADJUSTED PRICE COLUMN NAMES")
print("=" * 70)

for file_path in sorted(RAW_DIR.glob("NSE_data_all_stocks_*.csv")):

    headers = pd.read_csv(file_path, nrows=0).columns

    relevant_headers = [
        column for column in headers
        if "change" in column.lower()
        or "adjust" in column.lower()
    ]

    print(f"\n{file_path.name}")
    print(relevant_headers)


# --------------------------------------------------
# 2. INVESTIGATE CHANGE VALUES
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHANGE COLUMN INVESTIGATION")
print("=" * 70)

print("Missing change values:", df["change"].isna().sum())
print("Available change values:", df["change"].notna().sum())

print("\nSample records with available change values:")

print(
    df.loc[
        df["change"].notna(),
        ["date", "code", "day_price", "previous", "change", "change_pct"]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# 3. INVESTIGATE HIGHEST DAY PRICE
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 HIGHEST DAY PRICE RECORDS")
print("=" * 70)

print(
    df.nlargest(10, "day_price")[
        ["date", "code", "name", "day_low", "day_high",
         "day_price", "previous", "volume"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 4. INVESTIGATE MISSING DAY PRICE
# --------------------------------------------------

print("\n" + "=" * 70)
print("RECORDS WITH MISSING DAY PRICE")
print("=" * 70)

missing_price = df[df["day_price"].isna()]

if missing_price.empty:
    print("No missing day_price records.")
else:
    print(missing_price.to_string(index=False))


# --------------------------------------------------
# 5. INVESTIGATE ZERO PREVIOUS PRICE
# --------------------------------------------------

print("\n" + "=" * 70)
print("RECORDS WITH ZERO PREVIOUS PRICE")
print("=" * 70)

zero_previous = df[df["previous"] == 0]

if zero_previous.empty:
    print("No zero previous values.")
else:
    print(zero_previous.to_string(index=False))


print("\nInvestigation completed.")
