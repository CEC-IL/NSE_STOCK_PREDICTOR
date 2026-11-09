
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed" / "modeling"
REPORT_DIR = ROOT_DIR / "reports"
PREDICTION_DIR = ROOT_DIR / "data" / "processed" / "baseline_predictions"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

print("\nLoading training and validation datasets...")

price_train = pd.read_csv(
    DATA_DIR / "price_train.csv",
    low_memory=False
)

price_validation = pd.read_csv(
    DATA_DIR / "price_validation.csv",
    low_memory=False
)

direction_train = pd.read_csv(
    DATA_DIR / "direction_train.csv",
    low_memory=False
)

direction_validation = pd.read_csv(
    DATA_DIR / "direction_validation.csv",
    low_memory=False
)

print(f"Price training records: {len(price_train):,}")
print(f"Price validation records: {len(price_validation):,}")
print(f"Direction training records: {len(direction_train):,}")
print(f"Direction validation records: {len(direction_validation):,}")


# --------------------------------------------------
# 3. PRICE BASELINE
# --------------------------------------------------

print("\nPRICE BASELINE")
print("=" * 65)

# The simplest price prediction:
# Next price = current price.

price_train = price_train.dropna(
    subset=["daily_sequence_price", "target_next_price"]
).copy()

price_validation = price_validation.dropna(
    subset=["daily_sequence_price", "target_next_price"]
).copy()

y_price_train = price_train["target_next_price"]

y_price_validation = price_validation["target_next_price"]

price_predictions = price_validation["daily_sequence_price"]

mae = mean_absolute_error(
    y_price_validation,
    price_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_price_validation,
        price_predictions
    )
)

r2 = r2_score(
    y_price_validation,
    price_predictions
)

print(f"Training records: {len(price_train):,}")
print(f"Validation records: {len(price_validation):,}")

print(f"\nMAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# Save price predictions

price_prediction_output = price_validation[
    ["date", "target_next_date", "code"]
].copy()

price_prediction_output["actual_next_price"] = (
    y_price_validation.to_numpy()
)

price_prediction_output["predicted_next_price"] = (
    price_predictions.to_numpy()
)

price_prediction_output["absolute_error"] = (
    np.abs(
        price_prediction_output["actual_next_price"]
        - price_prediction_output["predicted_next_price"]
    )
)

price_prediction_output.to_csv(
    PREDICTION_DIR / "price_baseline_validation_predictions.csv",
    index=False
)


# --------------------------------------------------
# 4. DIRECTION BASELINE
# --------------------------------------------------

print("\nDIRECTION BASELINE")
print("=" * 65)

X_train = direction_train[["code"]].copy()

y_train = direction_train["target_direction"]

X_validation = direction_validation[["code"]].copy()

y_validation = direction_validation["target_direction"]

# The classifier always predicts the most frequent
# direction class found in the training data.

direction_model = DummyClassifier(
    strategy="most_frequent"
)

direction_model.fit(
    X_train,
    y_train
)

direction_predictions = direction_model.predict(
    X_validation
)

accuracy = accuracy_score(
    y_validation,
    direction_predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_validation,
    direction_predictions
)

macro_f1 = f1_score(
    y_validation,
    direction_predictions,
    average="macro",
    zero_division=0
)

majority_class = direction_model.classes_[
    np.argmax(direction_model.class_prior_)
]

print(f"Training records: {len(X_train):,}")
print(f"Validation records: {len(X_validation):,}")

print(f"\nMajority class: {majority_class}")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")


# --------------------------------------------------
# 5. CONFUSION MATRIX
# --------------------------------------------------

labels = [-1, 0, 1]

matrix = confusion_matrix(
    y_validation,
    direction_predictions,
    labels=labels
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual DOWN", "Actual UNCHANGED", "Actual UP"],
    columns=["Predicted DOWN", "Predicted UNCHANGED", "Predicted UP"]
)

print("\nCONFUSION MATRIX")
print(matrix_df.to_string())


# --------------------------------------------------
# 6. SAVE DIRECTION PREDICTIONS
# --------------------------------------------------

direction_prediction_output = direction_validation[
    ["date", "target_next_date", "code"]
].copy()

direction_prediction_output["actual_direction"] = (
    y_validation.to_numpy()
)

direction_prediction_output["predicted_direction"] = (
    direction_predictions
)

direction_prediction_output.to_csv(
    PREDICTION_DIR / "direction_baseline_validation_predictions.csv",
    index=False
)


# --------------------------------------------------
# 7. SAVE BASELINE METRICS
# --------------------------------------------------

metrics = pd.DataFrame([
    {
        "task": "Price",
        "model": "Persistence baseline",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
    },
    {
        "task": "Direction",
        "model": "Most frequent class baseline",
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
    }
])

metrics.to_csv(
    REPORT_DIR / "baseline_model_metrics.csv",
    index=False
)

matrix_df.to_csv(
    REPORT_DIR / "direction_baseline_confusion_matrix.csv"
)


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\nBASELINE EVALUATION COMPLETED")
print("=" * 65)

print("\nMetrics saved to:")
print(REPORT_DIR / "baseline_model_metrics.csv")

print("\nPredictions saved to:")
print(PREDICTION_DIR)

print("\nThe testing datasets were not used.")
