
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "nse_liquid_features.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "ml_predictions"
REPORT_DIR = ROOT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "daily_return",
    "return_5d",
    "return_10d",
    "sma_5",
    "sma_20",
    "sma_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "dist_52w_high",
    "dist_52w_low",
    "volume_ratio_5d",
]

def main():
    print(f"\n[ML] Loading liquid feature dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    
    # Drop rows without complete feature histories
    df_clean = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    
    # Train / Val / Test time splits
    train_mask = df_clean["date"] <= pd.Timestamp("2020-03-02")
    val_mask = (df_clean["date"] > pd.Timestamp("2020-03-02")) & (df_clean["date"] <= pd.Timestamp("2022-12-22"))
    
    # For training, drop rows where target is NaN (at the tail of series)
    train_df = df_clean[train_mask & df_clean["target_direction_5d"].notna()]
    val_df = df_clean[val_mask & df_clean["target_direction_5d"].notna()]
    
    X_train = train_df[FEATURE_COLS]
    y_dir_train = train_df["target_direction_5d"].astype(int)
    y_ret_train = train_df["target_return_5d"]
    
    X_val = val_df[FEATURE_COLS]
    y_dir_val = val_df["target_direction_5d"].astype(int)
    y_ret_val = val_df["target_return_5d"]
    
    print(f"[ML] Training on {len(X_train):,} samples. Validating on {len(X_val):,} samples...")
    
    # 1. 5-Day Direction Classifier
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(X_train, y_dir_train)
    
    val_dir_preds = clf.predict(X_val)
    val_dir_probs = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_dir_val, val_dir_preds)
    bal_acc = balanced_accuracy_score(y_dir_val, val_dir_preds)
    f1 = f1_score(y_dir_val, val_dir_preds, average="macro")
    
    # Baseline comparison (Always Predict Majority Class)
    maj_class = y_dir_train.mode()[0]
    base_preds = np.full(len(y_dir_val), maj_class)
    base_acc = accuracy_score(y_dir_val, base_preds)
    
    # 2. 5-Day Return Regressor
    reg = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_ret_train)
    
    val_ret_preds = reg.predict(X_val)
    mae = mean_absolute_error(y_ret_val, val_ret_preds)
    rmse = np.sqrt(mean_squared_error(y_ret_val, val_ret_preds))
    
    # Return Persistence Baseline: return = 0.0
    base_ret_mae = mean_absolute_error(y_ret_val, np.zeros(len(y_ret_val)))
    
    # Feature importances
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n[ML] Top 5 Predictive Drivers for Direction:")
    for feat, imp in top_features:
        print(f"  - {feat}: {imp:.4f}")
        
    metrics_summary = [
        {
            "model": "XGBoost Classifier (5-day Direction)",
            "task": "5-day Direction",
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1,
            "baseline_accuracy": base_acc,
            "accuracy_gain_vs_baseline": acc - base_acc,
            "sample_size": len(y_dir_val),
        },
        {
            "model": "XGBoost Regressor (5-day Return)",
            "task": "5-day Return %",
            "mae": mae,
            "rmse": rmse,
            "baseline_mae": base_ret_mae,
            "mae_reduction_vs_baseline": base_ret_mae - mae,
            "sample_size": len(y_ret_val),
        }
    ]
    pd.DataFrame(metrics_summary).to_csv(REPORT_DIR / "liquid_ml_metrics.csv", index=False)
    
    # -------------------------------------------------------------
    # Active latest predictions per stock for API & UI
    # -------------------------------------------------------------
    latest_rows = []
    for code, group in df_clean.groupby("code", sort=False):
        last_obs = group.sort_values("date").iloc[-1]
        x_last = pd.DataFrame([last_obs[FEATURE_COLS]])
        
        prob_up = float(clf.predict_proba(x_last)[0, 1])
        pred_return = float(reg.predict(x_last)[0])
        
        # Determine signal & drivers
        dir_label = "UP" if prob_up >= 0.52 else ("DOWN" if prob_up <= 0.48 else "NEUTRAL")
        confidence = prob_up if prob_up >= 0.5 else (1.0 - prob_up)
        
        # Drivers based on current values
        rsi_val = float(last_obs["rsi_14"])
        rsi_status = "Oversold (Bullish)" if rsi_val < 35 else ("Overbought (Bearish)" if rsi_val > 70 else "Neutral RSI")
        mom_5d = float(last_obs["return_5d"])
        mom_status = f"5d Mom: {mom_5d:+.1f}%"
        vol_status = "Low Volatility" if float(last_obs["volatility_20d"]) < 2.0 else "High Volatility"
        
        drivers_str = f"{rsi_status}, {mom_status}, {vol_status}"
        
        latest_rows.append({
            "code": code,
            "latest_date": str(last_obs["date"].date()),
            "current_price": round(float(last_obs["day_price"]), 2),
            "ml_predicted_direction": dir_label,
            "ml_probability_up": round(prob_up, 3),
            "ml_confidence": round(confidence, 3),
            "ml_expected_return_5d": round(pred_return, 2),
            "expected_return_range": f"{round(pred_return - 2.5, 1)}% to {round(pred_return + 2.5, 1)}%",
            "top_drivers": drivers_str,
        })
        
    latest_df = pd.DataFrame(latest_rows)
    latest_df.to_csv(OUTPUT_DIR / "liquid_5d_predictions.csv", index=False)
    print(f"\n[ML] Saved active liquid predictions to {OUTPUT_DIR / 'liquid_5d_predictions.csv'}")
    print(latest_df[["code", "current_price", "ml_predicted_direction", "ml_probability_up", "ml_expected_return_5d", "top_drivers"]].to_string(index=False))

if __name__ == "__main__":
    main()
