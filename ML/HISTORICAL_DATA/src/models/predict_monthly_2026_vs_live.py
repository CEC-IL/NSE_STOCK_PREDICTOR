# Predict Monthly (Jan–Nov 2026) Live Prices vs. ML & ARIMA Forecasts
# --------------------------------------------------------------
# This script extends `predict_2026_vs_live.py` to produce a month‑by‑month
# comparison for the period Jan 2026 → Nov 2026.  It:
#   1. Loads the daily live price CSV that the user uploaded (`media_1794228915208_cf2c45c7.csv`).
#   2. Aggregates each ticker to the **last‑trading‑day close** for every month.
#   3. Trains the XGBoost direction & return models once on historical data up to 31‑Oct‑2025.
#   4. Fits an ARIMA(1,1,1) model per ticker on the same historical price series.
#   5. For each month, forecasts the price (and 95 % CI) from the ARIMA model using the exact
#      number of days between the cutoff (31‑Oct‑2025) and the month‑end date.
#   6. Saves a detailed CSV report and a short markdown summary.
#
# Requirements:
#   - pandas, numpy, statsmodels, xgboost (already dependencies of the project).
#   - The uploaded live data file remains at the path stored in the brain artifact directory.
#
# Usage: from the repository root run
#   `python -m ML.HISTORICAL_DATA.src.models.predict_monthly_2026_vs_live`
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Configuration / paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# The daily live CSV uploaded by the user (brain artifact location).  We keep the
# absolute path to avoid copying the huge file into the project tree.
LIVE_DAILY_CSV = Path(
    r"C:\Users\Cecil.Kipchirchir\.gemini\antigravity\brain\eb1363ff-2c50-4361-bb65-5fcdfcc8d4b3\.user_uploaded\media_1794228915208_cf2c45c7.csv"
)

# ---------------------------------------------------------------------------
# Helper: aggregate daily live data to month‑end close per ticker
# ---------------------------------------------------------------------------
def load_monthly_live_prices() -> pd.DataFrame:
    """Return a DataFrame with columns: ticker, month (YYYY‑MM), live_close.

    The source CSV has columns:
        Date,Code,Name,12m Low,12m High,Day Low,Day High,Day Price,Previous,Change,Change%,Volume,Adjusted Price
    We keep only `Date`, `Code`, and `Day Price` (the intraday close).
    ``Date`` is in d‑Mon‑yy format (e.g., "2-Jan-26").
    """
    df = pd.read_csv(LIVE_DAILY_CSV, low_memory=False)
    # Standardise column names for clarity
    df = df.rename(columns={"Code": "ticker", "Day Price": "close", "Date": "date"})
    # Parse date (day‑month‑year with two‑digit year)
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")
    # Keep only needed columns
    df = df[["date", "ticker", "close"]]
    # Group by ticker and month, picking the last observed close in that month
    df.set_index("date", inplace=True)
    monthly = (
        df.groupby("ticker")
        .resample("M")
        .last()
        .reset_index()
        .rename(columns={"date": "month_end", "close": "live_close"})
    )
    # Keep months Jan‑Nov 2026 only
    monthly = monthly[(monthly["month_end"] >= pd.Timestamp("2026-01-01")) & (monthly["month_end"] <= pd.Timestamp("2026-11-30"))]
    monthly["month"] = monthly["month_end"].dt.to_period("M").astype(str)
    return monthly[["ticker", "month", "live_close", "month_end"]]

# ---------------------------------------------------------------------------
# Load historical features (same as the original script)
# ---------------------------------------------------------------------------
def load_historical_features():
    features_file = PROCESSED_DIR / "features_all_equities.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Missing {features_file}")
    df = pd.read_csv(features_file, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ---------------------------------------------------------------------------
# Columns used for the XGBoost models (identical to original script)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "daily_return",
    "return_5d",
    "sma_5",
    "sma_20",
    "sma_50",
    "rsi_14",
    "volatility_20d",
    "dist_52w_high",
    "dist_52w_low",
    "days_since_trade",
    "is_flatline_day",
]

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def generate_monthly_predictions():
    print("\n" + "=" * 70)
    print("NSE MONTHLY (Jan‑Nov 2026) PREDICTION VS LIVE DATA")
    print("=" * 70 + "\n")

    # Load data
    live_monthly = load_monthly_live_prices()
    hist_df = load_historical_features()

    # Train ML models on data up to 31‑Oct‑2025 (same cut‑off as original script)
    train_mask = hist_df["date"] <= pd.Timestamp("2025-10-31")
    train_df = hist_df[train_mask].dropna(subset=FEATURE_COLS + ["target_direction_5d"])
    X_train = train_df[FEATURE_COLS]
    y_dir_train = train_df["target_direction_5d"].astype(int)
    y_ret_train = train_df["target_return_5d"]
    print(f"[ML] Training XGBoost models on {len(X_train):,} observations...")
    clf = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_dir_train)
    reg = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_ret_train)

    # Prepare results container
    rows = []

    # Iterate over each ticker that appears both in historic data and live monthly data
    tickers = set(live_monthly["ticker"]).intersection(set(hist_df["code"]))
    for ticker in sorted(tickers):
        # Historical price series up to cut‑off
        ticker_hist = hist_df[hist_df["code"] == ticker]
        ticker_hist_cut = ticker_hist[ticker_hist["date"] <= pd.Timestamp("2025-10-31")]
        if ticker_hist_cut.empty:
            continue
        # Last known price (31‑Oct‑2025)
        last_obs = ticker_hist_cut.sort_values("date").iloc[-1]
        last_price = float(last_obs["day_price"])
        last_date = last_obs["date"]

        # Fit ARIMA on the historical price series (up to cut‑off)
        prices = ticker_hist_cut["day_price"].values
        try:
            arima_model = ARIMA(prices, order=(1, 1, 1)).fit()
        except Exception as e:
            print(f"[WARN] ARIMA failed for {ticker}: {e}")
            arima_model = None

        # Get all live month rows for this ticker
        ticker_live_months = live_monthly[live_monthly["ticker"] == ticker]
        for _, month_row in ticker_live_months.iterrows():
            month_end = month_row["month_end"]
            live_price = float(month_row["live_close"])
            month_label = month_row["month"]

            # ----------- XGBoost prediction (same for every month) -----------
            x_last = pd.DataFrame([last_obs[FEATURE_COLS]])
            prob_up = float(clf.predict_proba(x_last)[0, 1])
            ml_direction = "UP" if prob_up >= 0.52 else ("DOWN" if prob_up <= 0.48 else "FLAT")
            ml_return_pred = float(reg.predict(x_last)[0])
            ml_price = last_price * (1.0 + ml_return_pred / 100.0)

            # ----------- ARIMA forecast for this month ---------------------
            if arima_model is not None:
                # Number of days from cut‑off to month end (inclusive)
                days_ahead = (month_end - pd.Timestamp("2025-10-31")).days
                # statsmodels expects integer steps; we forecast that many days
                try:
                    arima_res = arima_model.get_forecast(steps=days_ahead)
                    arima_pred_price = float(arima_res.predicted_mean[-1])
                    conf_int = arima_res.conf_int(alpha=0.05)
                    lower_95 = max(0.01, float(conf_int[-1, 0]))
                    upper_95 = float(conf_int[-1, 1])
                except Exception as e:
                    # Fallback to last price with a generous band
                    arima_pred_price = last_price
                    lower_95 = last_price * 0.8
                    upper_95 = last_price * 1.2
            else:
                arima_pred_price = last_price
                lower_95 = last_price * 0.8
                upper_95 = last_price * 1.2

            inside_arima = "YES" if lower_95 <= live_price <= upper_95 else "NO"

            rows.append({
                "ticker": ticker,
                "month": month_label,
                "last_2025_price": round(last_price, 2),
                "live_2026_price": round(live_price, 2),
                "ml_predicted_direction": ml_direction,
                "ml_up_probability": round(prob_up, 3),
                "ml_predicted_price": round(ml_price, 2),
                "arima_forecast_price": round(arima_pred_price, 2),
                "arima_lower_95": round(lower_95, 2),
                "arima_upper_95": round(upper_95, 2),
                "inside_arima_cone": inside_arima,
            })

    # Build result DataFrame
    res_df = pd.DataFrame(rows)
    # Save CSV
    output_csv = REPORT_DIR / "jan_to_nov_2026_prediction_comparison.csv"
    res_df.to_csv(output_csv, index=False)
    print(f"\nSaved monthly comparison to: {output_csv}\n")

    # Summary statistics (overall, not per month)
    # Direction accuracy across all ticker‑month combos
    # For simplicity we compute accuracy against actual direction derived from live vs. last price
    def actual_dir(row):
        pct = ((row["live_2026_price"] / row["last_2025_price"]) - 1) * 100
        if pct > 1:
            return "UP"
        if pct < -1:
            return "DOWN"
        return "FLAT"
    res_df["actual_direction"] = res_df.apply(actual_dir, axis=1)
    dir_acc = (res_df["ml_predicted_direction"] == res_df["actual_direction"]).mean() * 100
    cone_cov = (res_df["inside_arima_cone"] == "YES").mean() * 100
    print(f"Overall directional accuracy (ML): {dir_acc:.1f}%")
    print(f"ARIMA cone coverage (monthly): {cone_cov:.1f}%")

    # Write a concise markdown summary
    summary_md = REPORT_DIR / "monthly_2026_prediction_analysis.md"
    with open(summary_md, "w") as f:
        f.write("# Monthly (Jan‑Nov 2026) Prediction vs Live NSE Data\n\n")
        f.write(f"**Directional accuracy (ML):** {dir_acc:.1f}%  \n")
        f.write(f"**ARIMA 95 % cone coverage:** {cone_cov:.1f}%  \n\n")
        f.write("## Sample rows (first 10)\n\n")
        f.write(res_df.head(10).to_markdown(index=False))
        f.write("\n\n")
        f.write("*All ticker‑month rows are saved in `jan_to_nov_2026_prediction_comparison.csv`.*\n")
    print(f"Saved markdown summary to: {summary_md}\n")

if __name__ == "__main__":
    generate_monthly_predictions()
