
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

def standardize_columns(df):
    """
    Standardize different column names used across years.
    """

    rename_map = {
        "date": "date",
        "code": "code",
        "name": "name",

        "12m low": "12m_low",
        "12m high": "12m_high",

        "day low": "day_low",
        "day high": "day_high",
        "day price": "day_price",

        "previous": "previous",
        "change": "change",

        "change%": "change_pct",
        "change %": "change_pct",

        "volume": "volume",

        "adjust": "adjusted_price",
        "adjusted": "adjusted_price",
        "adjusted price": "adjusted_price",
        "adjustedprice": "adjusted_price",
    }

    # Standardize capitalization and spaces.
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # Rename columns using the mapping above.
    df = df.rename(columns=rename_map)

    return df


# --------------------------------------------------
# 3. CONVERT TEXT NUMBERS INTO REAL NUMBERS
# --------------------------------------------------

def convert_to_number(series):
    """
    Convert text values into numeric values.

    Examples:
    '3,396.83' -> 3396.83
    '1.50%'    -> 1.50
    '-8.24%'   -> -8.24
    '-'        -> NaN

    Percentages remain in percentage points.
    """

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace(
            ["-", "", "nan", "None"],
            pd.NA
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# --------------------------------------------------
# 4. READ AND CLEAN EACH CSV FILE
# --------------------------------------------------

all_datasets = []
summary = []

csv_files = sorted(
    RAW_DIR.glob("NSE_data_all_stocks_*.csv")
)

print(f"Found {len(csv_files)} CSV files.")

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    try:

        # Read CSV with encoding fallback.
        try:
            df = pd.read_csv(
                file,
                encoding="utf-8-sig",
                low_memory=False
            )

        except UnicodeDecodeError:
            df = pd.read_csv(
                file,
                encoding="cp1252",
                low_memory=False
            )

        original_rows = len(df)

        # Standardize column names.
        df = standardize_columns(df)

        # Required columns.
        required_columns = [
            "date",
            "code",
            "name",
            "12m_low",
            "12m_high",
            "day_low",
            "day_high",
            "day_price",
            "previous",
            "change",
            "change_pct",
            "volume",
            "adjusted_price",
        ]

        # Add missing columns if necessary.
        for column in required_columns:

            if column not in df.columns:
                df[column] = pd.NA

        # Keep a consistent column order.
        df = df[required_columns].copy()

        # Remove completely empty rows.
        df = df.dropna(how="all")

        rows_after_empty_removal = len(df)

        # Clean text columns.
        for column in ["code", "name"]:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

        # Convert dates.
        df["date"] = pd.to_datetime(
            df["date"],
            format="mixed",
            errors="coerce",
            dayfirst=False
        )

        # Convert numerical columns.
        numeric_columns = [
            "12m_low",
            "12m_high",
            "day_low",
            "day_high",
            "day_price",
            "previous",
            "change",
            "change_pct",
            "volume",
            "adjusted_price",
        ]

        for column in numeric_columns:

            df[column] = convert_to_number(
                df[column]
            )

        # Count available percentage values.
        available_change_pct = (
            df["change_pct"].notna().sum()
        )

        missing_change_pct = (
            df["change_pct"].isna().sum()
        )

        # Remove exact duplicate rows only.
        before_duplicates = len(df)

        df = df.drop_duplicates()

        exact_duplicates_removed = (
            before_duplicates - len(df)
        )

        # Count missing essential information.
        missing_dates = df["date"].isna().sum()
        missing_codes = df["code"].isna().sum()
        missing_prices = df["day_price"].isna().sum()

        # Save dataset in memory.
        all_datasets.append(df)

        # Create per-file report.
        summary.append(
            f"FILE: {file.name}\n"
            f"Original rows: {original_rows}\n"
            f"Rows after empty-row removal: "
            f"{rows_after_empty_removal}\n"
            f"Exact duplicates removed: "
            f"{exact_duplicates_removed}\n"
            f"Rows retained: {len(df)}\n"
            f"Available change_pct values: "
            f"{available_change_pct}\n"
            f"Missing change_pct values: "
            f"{missing_change_pct}\n"
            f"Missing dates: {missing_dates}\n"
            f"Missing stock codes: {missing_codes}\n"
            f"Missing day prices: {missing_prices}\n"
            f"{'-' * 50}\n"
        )

        print(f"Rows retained: {len(df)}")

        print(
            "Available change_pct values: "
            f"{available_change_pct}"
        )

        print(
            "Exact duplicates removed: "
            f"{exact_duplicates_removed}"
        )

    except Exception as error:

        print(f"ERROR processing {file.name}: {error}")

        summary.append(
            f"ERROR processing {file.name}: {error}\n"
        )


# --------------------------------------------------
# 5. COMBINE ALL YEARLY DATASETS
# --------------------------------------------------

if not all_datasets:

    raise ValueError(
        "No datasets were successfully loaded. "
        "Check the data/raw folder."
    )

combined_df = pd.concat(
    all_datasets,
    ignore_index=True
)

# Sort by stock code and date.
# Stable sorting preserves the original order
# of records with the same stock code and date.
combined_df = combined_df.sort_values(
    by=["code", "date"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# 6. CHECK REPEATED DATE/CODE RECORDS
# --------------------------------------------------

# Keep these records. Do not delete them.
duplicate_keys = combined_df.duplicated(
    subset=["date", "code"],
    keep=False
)

duplicate_key_rows = combined_df[
    duplicate_keys
]

duplicate_key_rows.to_csv(
    REPORT_DIR / "duplicate_date_code_records.csv",
    index=False
)


# --------------------------------------------------
# 7. SAVE THE CLEANED DATASET
# --------------------------------------------------

output_file = (
    PROCESSED_DIR
    / "nse_all_stocks_cleaned.csv"
)

combined_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 8. SAVE THE CLEANING REPORT
# --------------------------------------------------

total_change_pct = (
    combined_df["change_pct"].notna().sum()
)

missing_change_pct = (
    combined_df["change_pct"].isna().sum()
)

summary.append(
    "\nFINAL COMBINED DATASET\n"
    f"Total rows: {len(combined_df)}\n"
    f"Total columns: {len(combined_df.columns)}\n"
    f"Date range: {combined_df['date'].min()} "
    f"to {combined_df['date'].max()}\n"
    f"Available change_pct values: "
    f"{total_change_pct}\n"
    f"Missing change_pct values: "
    f"{missing_change_pct}\n"
    f"Records sharing the same date and stock code: "
    f"{duplicate_key_rows.shape[0]}\n"
)

report_file = (
    REPORT_DIR / "cleaning_summary.txt"
)

report_file.write_text(
    "\n".join(summary),
    encoding="utf-8"
)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("DATA CLEANING COMPLETED")
print("=" * 65)

print(f"Total rows: {len(combined_df):,}")
print(f"Total columns: {len(combined_df.columns)}")

print(
    f"Date range: {combined_df['date'].min()} "
    f"to {combined_df['date'].max()}"
)

print(
    f"Available change_pct values: "
    f"{total_change_pct:,}"
)

print(
    f"Missing change_pct values: "
    f"{missing_change_pct:,}"
)

print(
    f"Repeated date/code records retained: "
    f"{len(duplicate_key_rows):,}"
)

print(f"\nCleaned dataset saved to:\n{output_file}")

print(f"\nCleaning report saved to:\n{report_file}")

print("\nFirst five cleaned records:")

print(
    combined_df.head().to_string(index=False)
)
