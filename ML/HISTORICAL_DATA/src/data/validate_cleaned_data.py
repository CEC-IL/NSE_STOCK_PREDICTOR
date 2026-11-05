
from pathlib import Path
import pandas as pd

# Get the main HISTORICAL_DATA folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Location of the cleaned dataset
FILE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "nse_all_stocks_cleaned.csv"
)

# Location for the validation report
REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "data_validation_report.txt"
)

# Load the cleaned dataset
df = pd.read_csv(FILE_PATH)

report = []

def add_line(message=""):
    report.append(str(message))


# --------------------------------------------------
# 1. GENERAL DATASET INFORMATION
# --------------------------------------------------

add_line("=" * 65)
add_line("NSE DATA QUALITY VALIDATION REPORT")
add_line("=" * 65)

add_line(f"Total records: {len(df):,}")
add_line(f"Total columns: {len(df.columns)}")

add_line("\nCOLUMNS:")
for column in df.columns:
    add_line(f"- {column}")


# --------------------------------------------------
# 2. DATE RANGE
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

add_line("\n" + "=" * 65)
add_line("DATE VALIDATION")
add_line("=" * 65)

add_line(f"Earliest date: {df['date'].min()}")
add_line(f"Latest date: {df['date'].max()}")
add_line(f"Missing or invalid dates: {df['date'].isna().sum():,}")


# --------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("MISSING VALUES PER COLUMN")
add_line("=" * 65)

missing = df.isna().sum()

for column, count in missing.items():
    percentage = (count / len(df)) * 100
    add_line(
        f"{column}: {count:,} missing "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# 4. EXACT DUPLICATE ROWS
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("EXACT DUPLICATE ROWS")
add_line("=" * 65)

exact_duplicates = df.duplicated().sum()

add_line(f"Exact duplicate rows remaining: {exact_duplicates:,}")


# --------------------------------------------------
# 5. REPEATED DATE AND STOCK CODE COMBINATIONS
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("REPEATED DATE AND STOCK CODE RECORDS")
add_line("=" * 65)

repeated = df[
    df.duplicated(
        subset=["date", "code"],
        keep=False
    )
]

repeated_groups = (
    repeated.groupby(["date", "code"])
    .size()
    .reset_index(name="record_count")
)

add_line(
    f"Records belonging to repeated date/code groups: "
    f"{len(repeated):,}"
)

add_line(
    f"Number of repeated date/code groups: "
    f"{len(repeated_groups):,}"
)

add_line("\nFirst 20 repeated groups:")

if not repeated_groups.empty:
    add_line(
        repeated_groups.head(20).to_string(index=False)
    )
else:
    add_line("No repeated date/code groups found.")


# --------------------------------------------------
# 6. RECORDS PER YEAR
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("RECORDS PER YEAR")
add_line("=" * 65)

records_per_year = (
    df.dropna(subset=["date"])
    .groupby(df["date"].dt.year)
    .size()
)

for year, count in records_per_year.items():
    add_line(f"{year}: {count:,} records")


# --------------------------------------------------
# 7. STOCK CODE INFORMATION
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("STOCK CODE VALIDATION")
add_line("=" * 65)

add_line(
    f"Unique stock codes: {df['code'].nunique(dropna=True):,}"
)

add_line(
    f"Missing stock codes: {df['code'].isna().sum():,}"
)


# --------------------------------------------------
# 8. PRICE VALIDATION
# --------------------------------------------------

add_line("\n" + "=" * 65)
add_line("PRICE VALIDATION")
add_line("=" * 65)

price_columns = [
    "day_price",
    "previous",
    "12m_low",
    "12m_high",
    "adjusted_price"
]

for column in price_columns:
    if column not in df.columns:
        add_line(f"\n{column}: Column not found")
        continue

    values = pd.to_numeric(df[column], errors="coerce")

    add_line(f"\n{column}")
    add_line(f"Missing values: {values.isna().sum():,}")
    add_line(f"Minimum: {values.min()}")
    add_line(f"Maximum: {values.max()}")
    add_line(f"Zero values: {(values == 0).sum():,}")
    add_line(f"Negative values: {(values < 0).sum():,}")


# --------------------------------------------------
# 9. SAVE REPORT
# --------------------------------------------------

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

REPORT_PATH.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\nValidation completed successfully.")

print(f"Total records: {len(df):,}")
print(f"Exact duplicates remaining: {exact_duplicates:,}")
print(f"Repeated date/code records: {len(repeated):,}")

print(f"\nReport saved to:\n{REPORT_PATH}")
