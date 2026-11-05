
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"

print("\nORIGINAL CHANGE% DATA BY YEAR")
print("=" * 70)

for file_path in sorted(RAW_DIR.glob("NSE_data_all_stocks_*.csv")):

    df = pd.read_csv(file_path)

    # Find the original percentage column
    column = next(
        (c for c in df.columns if c.strip().lower() == "change%"),
        None
    )

    print(f"\n{file_path.name}")

    if column is None:
        print("Change% column not found")
        continue

    values = df[column].astype(str).str.strip()

    valid = values[
        ~values.isin(["", "-", "nan", "None"])
    ]

    print("Total records:", len(values))
    print("Available percentage values:", len(valid))
    print("Missing or placeholder values:", len(values) - len(valid))

    print("Sample values:")
    print(valid.head(5).tolist())

print("\nInvestigation completed.")
