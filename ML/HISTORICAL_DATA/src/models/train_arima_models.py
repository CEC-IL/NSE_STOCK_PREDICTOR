
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_liquid_features.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "arima_predictions"
REPORT_DIR = ROOT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LIQUID_STOCKS = [
    "SCOM", "EQTY", "KCB", "EABL", "ABSA", 
    "COOP", "BAT", "BAMB", "SCBK", "NCBA", 
    "KEGN", "CTUM"
]

def evaluate_stock_arima(code: str, df_stock: pd.DataFrame):
    """
    Fits an ARIMA model on the training series and evaluates 5-day out-of-sample forecasts 
    and prediction intervals against actual prices and the persistence baseline.
    """
    df_stock = df_stock.sort_values("date").dropna(subset=["day_price"]).reset_index(drop=True)
    
    # Split by date (Strictly no random split)
    train_mask = df_stock["date"] <= pd.Timestamp("2020-03-02")
    val_mask = (df_stock["date"] > pd.Timestamp("2020-03-02")) & (df_stock["date"] <= pd.Timestamp("2022-12-22"))
    
    train_series = df_stock.loc[train_mask, "day_price"].values
    val_df = df_stock.loc[val_mask].copy().reset_index(drop=True)
    
    if len(train_series) < 100 or len(val_df) < 20:
        return None, None
        
    print(f"[{code}] Training ARIMA on {len(train_series):,} samples. Validating on {len(val_df):,} samples...")
    
    # Fit ARIMA(1, 1, 1) on historical series
    try:
        model = ARIMA(train_series, order=(1, 1, 1))
        fitted_model = model.fit()
    except Exception:
        # Fallback to ARIMA(0, 1, 1) if numerical convergence fails
        model = ARIMA(train_series, order=(0, 1, 1))
        fitted_model = model.fit()

    # Out-of-sample evaluation with rolling 5-day forecast windows
    step = 5
    actual_5d = []
    predicted_5d = []
    baseline_5d = []
    ci_lower = []
    ci_upper = []
    
    for i in range(0, len(val_df) - step, step):
        current_p = val_df.loc[i, "day_price"]
        actual_p = val_df.loc[i + step, "day_price"]
        
        # We forecast 5 steps forward
        # Using fitted model residuals to simulate interval scale
        res = fitted_model.forecast(steps=step)
        pred_p = max(0.01, float(res[-1]))
        
        # Estimate prediction interval from historical residual std
        sigma = np.std(fitted_model.resid[-200:]) if len(fitted_model.resid) > 200 else 1.0
        margin_95 = 1.96 * sigma * np.sqrt(step)
        
        lower_p = max(0.01, pred_p - margin_95)
        upper_p = pred_p + margin_95
        
        actual_5d.append(actual_p)
        predicted_5d.append(pred_p)
        baseline_5d.append(current_p)
        ci_lower.append(lower_p)
        ci_upper.append(upper_p)
        
    actual_5d = np.array(actual_5d)
    predicted_5d = np.array(predicted_5d)
    baseline_5d = np.array(baseline_5d)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Metrics
    arima_mae = mean_absolute_error(actual_5d, predicted_5d)
    arima_rmse = np.sqrt(mean_squared_error(actual_5d, predicted_5d))
    base_mae = mean_absolute_error(actual_5d, baseline_5d)
    base_rmse = np.sqrt(mean_squared_error(actual_5d, baseline_5d))
    
    # Direction: did predicted move direction match actual move direction?
    actual_dir = np.sign(actual_5d - baseline_5d)
    predicted_dir = np.sign(predicted_5d - baseline_5d)
    dir_acc = np.mean(actual_dir == predicted_dir)
    
    # Coverage: did actual fall within 95% interval?
    covered = (actual_5d >= ci_lower) & (actual_5d <= ci_upper)
    coverage = np.mean(covered)
    
    metrics = {
        "code": code,
        "sample_size": len(actual_5d),
        "arima_mae": arima_mae,
        "baseline_mae": base_mae,
        "mae_diff": arima_mae - base_mae,
        "arima_rmse": arima_rmse,
        "baseline_rmse": base_rmse,
        "direction_accuracy": dir_acc,
        "interval_coverage_95": coverage,
    }
    
    # -------------------------------------------------------------
    # Now fit on all historical data up to the latest date to produce
    # the active forward-looking 5-day forecast for the API / UI
    # -------------------------------------------------------------
    all_prices = df_stock["day_price"].values
    latest_model = ARIMA(all_prices, order=(1, 1, 1)).fit()
    forecast_res = latest_model.get_forecast(steps=5)
    
    last_known_price = float(all_prices[-1])
    latest_predicted_5d = float(forecast_res.predicted_mean[-1])
    conf_int = forecast_res.conf_int(alpha=0.05)
    latest_lower_95 = max(0.01, float(conf_int[-1, 0]))
    latest_upper_95 = float(conf_int[-1, 1])
    
    expected_return_pct = ((latest_predicted_5d / last_known_price) - 1.0) * 100.0
    direction = "UP" if expected_return_pct > 0.3 else ("DOWN" if expected_return_pct < -0.3 else "FLAT")
    
    latest_forecast = {
        "code": code,
        "latest_date": str(df_stock["date"].max().date()),
        "current_price": round(last_known_price, 2),
        "predicted_5d_price": round(latest_predicted_5d, 2),
        "expected_return_5d_pct": round(expected_return_pct, 2),
        "lower_bound_95": round(latest_lower_95, 2),
        "upper_bound_95": round(latest_upper_95, 2),
        "direction": direction,
        "order": "ARIMA(1,1,1)",
    }
    
    return metrics, latest_forecast

def main():
    print(f"\n[ARIMA] Loading liquid stock dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    
    all_metrics = []
    all_latest = []
    
    for code in LIQUID_STOCKS:
        df_stock = df[df["code"] == code]
        if df_stock.empty:
            continue
        m, f = evaluate_stock_arima(code, df_stock)
        if m:
            all_metrics.append(m)
            all_latest.append(f)
            
    metrics_df = pd.DataFrame(all_metrics)
    latest_df = pd.DataFrame(all_latest)
    
    # Save outputs
    metrics_path = REPORT_DIR / "arima_model_metrics.csv"
    latest_path = OUTPUT_DIR / "arima_latest_forecasts.csv"
    
    metrics_df.to_csv(metrics_path, index=False)
    latest_df.to_csv(latest_path, index=False)
    
    print("\n" + "=" * 65)
    print("ARIMA MODEL EVALUATION SUMMARY (OUT-OF-SAMPLE 5-DAY HORIZON)")
    print("=" * 65)
    print(metrics_df[["code", "arima_mae", "baseline_mae", "direction_accuracy", "interval_coverage_95"]].to_string(index=False))
    
    print(f"\nOverall Average Direction Accuracy: {metrics_df['direction_accuracy'].mean():.2%}")
    print(f"Overall Average 95% Interval Coverage: {metrics_df['interval_coverage_95'].mean():.2%}")
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved latest forecasts to: {latest_path}")

if __name__ == "__main__":
    main()
