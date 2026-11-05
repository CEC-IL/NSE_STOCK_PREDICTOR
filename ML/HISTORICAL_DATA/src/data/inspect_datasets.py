from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Find all CSV files
csv_files = sorted(RAW_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files.")

report = []

# Inspect each dataset
for file in csv_files:

    print("\n" + "=" * 60)
    print(f"INSPECTING: {file.name}")
    print("=" * 60)

    try:
        # Read CSV
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

        # Basic information
        rows, columns = df.shape

        print(f"\nRows: {rows}")
        print(f"Columns: {columns}")

        print("\nCOLUMN NAMES:")
        print(df.columns.tolist())

        print("\nFIRST 5 ROWS:")
        print(df.head().to_string())

        print("\nDATA TYPES:")
        print(df.dtypes.to_string())

        print("\nMISSING VALUES:")
        print(df.isnull().sum().to_string())

        print("\nDUPLICATE ROWS:")
        print(df.duplicated().sum())

        # Add results to report
        report.append("\n" + "=" * 60)
        report.append(f"FILE: {file.name}")
        report.append("=" * 60)

        report.append(f"Rows: {rows}")
        report.append(f"Columns: {columns}")

        report.append(
            f"\nColumn names:\n{df.columns.tolist()}"
        )

        report.append(
            f"\nFirst 5 rows:\n{df.head().to_string()}"
        )

        report.append(
            f"\nData types:\n{df.dtypes.to_string()}"
        )

        report.append(
            f"\nMissing values:\n{df.isnull().sum().to_string()}"
        )

        report.append(
            f"\nDuplicate rows: {df.duplicated().sum()}"
        )

    except Exception as error:

        print(f"ERROR: {error}")

        report.append(
            f"\nERROR reading {file.name}: {error}"
        )


# Save inspection report
report_path = REPORT_DIR / "dataset_inventory.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)

print(f"Files inspected: {len(csv_files)}")
print(f"Report saved at: {report_path}")
