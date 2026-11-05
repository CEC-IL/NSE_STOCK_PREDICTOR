
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = (
    BASE_DIR / "data" / "raw"
    / "NSE_data_all_stocks_2009.csv"
)

# Read the original CSV
df = pd.read_csv(RAW_FILE)

# Standardize column names:
# "Day Price" becomes "day_price"
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("%", "pct", regex=False)
)

# Display available columns to help identify naming differences
print("AVAILABLE COLUMNS:")
print(df.columns.tolist())

# Preserve original row positions
df.insert(0, "source_row", range(1, len(df) + 1))

# Parse dates
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce",
    dayfirst=False
)

dates = pd.to_datetime([
    "2009-01-29",
    "2009-04-02",
    "2009-10-08"
])

columns = [
    "source_row",
    "date",
    "code",
    "name",
    "day_low",
    "day_high",
    "day_price",
    "previous",
    "volume"
]

# Inspect original row order
for date in dates:
    print("\n" + "=" * 80)
    print(f"ORIGINAL ROW ORDER: {date.date()}")
    print("=" * 80)

    selected = df[df["date"] == date]

    print(
        selected[columns]
        .head(30)
        .to_string(index=False)
    )

    print(f"\nTotal records on this date: {len(selected)}")
