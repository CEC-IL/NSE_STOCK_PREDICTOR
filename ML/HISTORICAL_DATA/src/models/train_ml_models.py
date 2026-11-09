
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBRegressor, XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "modeling"
OUTPUT_DIR = ROOT / "data" / "processed" / "ml_predictions"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. FEATURE DEFINITIONS
# --------------------------------------------------

NUMERIC_FEATURES = [
    "daily_sequence_price",
    "lag_1",
    "lag_5",
    "lag_10",
    "daily_return",
    "return_5d",
    "return_10d",
    "sma_5",
    "sma_10",
    "sma_20",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
]

CATEGORICAL_FEATURES = ["code"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PRICE_TARGET = "target_next_price"
DIRECTION_TARGET = "target_direction"

DIRECTION_LABELS = {
    -1: "DOWN",
    0: "UNCHANGED",
    1: "UP",
}


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_datasets():

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

    print("Price training rows:", len(price_train))
    print("Price validation rows:", len(price_validation))
    print("Direction training rows:", len(direction_train))
    print("Direction validation rows:", len(direction_validation))

    return (
        price_train,
        price_validation,
        direction_train,
        direction_validation,
    )


# --------------------------------------------------
# 4. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def create_model_pipeline(model):

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model),
    ])


# --------------------------------------------------
# 5. PREPARE FEATURES AND TARGETS
# --------------------------------------------------

def prepare_xy(data, target):

    required_columns = FEATURES + [target]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    clean_data = data.dropna(subset=[target]).copy()

    X = clean_data[FEATURES].copy()
    y = clean_data[target].copy()

    return X, y, clean_data


# --------------------------------------------------
# 6. PRICE MODEL EVALUATION
# --------------------------------------------------

def evaluate_price_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    validation_data,
):

    print(f"\nTraining {model_name}...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    print(f"{model_name} training completed.")

    prediction_start = time.time()

    predictions = model.predict(X_validation)

    prediction_seconds = time.time() - prediction_start

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_validation, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_validation, predictions)
    )

    r2 = r2_score(y_validation, predictions)

    actual = np.asarray(y_validation)
    errors = np.abs(actual - predictions)

    prediction_results = validation_data[
        ["date", "code", "target_next_date"]
    ].copy()

    prediction_results["actual_price"] = actual
    prediction_results["predicted_price"] = predictions
    prediction_results["absolute_error"] = errors

    prediction_results["actual_direction"] = np.sign(
        prediction_results["actual_price"]
        - validation_data["daily_sequence_price"].to_numpy()
    ).astype(int)

    prediction_results["predicted_direction"] = np.sign(
        prediction_results["predicted_price"]
        - validation_data["daily_sequence_price"].to_numpy()
    ).astype(int)

    prediction_results["actual_direction_label"] = (
        prediction_results["actual_direction"]
        .map(DIRECTION_LABELS)
    )

    prediction_results["predicted_direction_label"] = (
        prediction_results["predicted_direction"]
        .map(DIRECTION_LABELS)
    )

    prediction_results.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv",
        index=False
    )

    # Per-stock evaluation
    prediction_results["percentage_error"] = (
        prediction_results["absolute_error"]
        / prediction_results["actual_price"].abs().replace(0, np.nan)
        * 100
    )

    per_stock = (
        prediction_results
        .groupby("code")
        .agg(
            observations=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            mean_actual_price=("actual_price", "mean"),
            normalized_mae_percent=("percentage_error", "mean"),
        )
        .reset_index()
    )

    per_stock.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_per_stock.csv",
        index=False
    )

    direction_accuracy = accuracy_score(
        prediction_results["actual_direction"],
        prediction_results["predicted_direction"]
    )

    direction_balanced_accuracy = balanced_accuracy_score(
        prediction_results["actual_direction"],
        prediction_results["predicted_direction"]
    )

    direction_macro_f1 = f1_score(
        prediction_results["actual_direction"],
        prediction_results["predicted_direction"],
        labels=[-1, 0, 1],
        average="macro",
        zero_division=0,
    )

    metrics = {
        "model": model_name,
        "task": "price",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "direction_balanced_accuracy": direction_balanced_accuracy,
        "direction_macro_f1": direction_macro_f1,
        "macro_stock_mae": per_stock["mae"].mean(),
        "macro_stock_normalized_mae_percent": (
            per_stock["normalized_mae_percent"].mean()
        ),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "validation_rows": len(predictions),
    }

    print(f"{model_name} results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Direction accuracy: {direction_accuracy:.4f}")
    print(f"  Training time: {training_seconds:.2f} seconds")

    return metrics, model


# --------------------------------------------------
# 7. DIRECTION MODEL EVALUATION
# --------------------------------------------------

def evaluate_direction_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    validation_data,
):

    print(f"\nTraining {model_name}...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    print(f"{model_name} training completed.")

    prediction_start = time.time()

    predictions = model.predict(X_validation)

    prediction_seconds = time.time() - prediction_start

    accuracy = accuracy_score(y_validation, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_validation, predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        labels=[-1, 0, 1],
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[-1, 0, 1],
    )

    prediction_results = validation_data[
        ["date", "code", "target_next_date"]
    ].copy()

    prediction_results["actual_direction"] = y_validation.to_numpy()
    prediction_results["predicted_direction"] = predictions

    prediction_results["actual_label"] = (
        prediction_results["actual_direction"]
        .map(DIRECTION_LABELS)
    )

    prediction_results["predicted_label"] = (
        prediction_results["predicted_direction"]
        .map(DIRECTION_LABELS)
    )

    prediction_results.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv",
        index=False
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual_DOWN", "Actual_UNCHANGED", "Actual_UP"],
        columns=["Predicted_DOWN", "Predicted_UNCHANGED", "Predicted_UP"],
    )

    matrix_df.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    metrics = {
        "model": model_name,
        "task": "direction",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "validation_rows": len(predictions),
    }

    print(f"{model_name} results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print("  Confusion matrix:")
    print(matrix_df)

    return metrics, model


# --------------------------------------------------
# 8. MAIN TRAINING PROCESS
# --------------------------------------------------

def main():

    (
        price_train,
        price_validation,
        direction_train,
        direction_validation,
    ) = load_datasets()

    # Price datasets
    X_price_train, y_price_train, price_train_clean = prepare_xy(
        price_train, PRICE_TARGET
    )

    X_price_validation, y_price_validation, price_validation_clean = (
        prepare_xy(price_validation, PRICE_TARGET)
    )

    # Direction datasets
    X_direction_train, y_direction_train, direction_train_clean = (
        prepare_xy(direction_train, DIRECTION_TARGET)
    )

    X_direction_validation, y_direction_validation, direction_validation_clean = (
        prepare_xy(direction_validation, DIRECTION_TARGET)
    )

    print("\n================================================")
    print("PRICE MODEL TRAINING")
    print("================================================")

    price_models = {
        "Linear Regression": LinearRegression(),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            max_features=0.7,
            min_samples_leaf=5,
            max_samples=0.7,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost Regressor": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    all_metrics = []

    for name, estimator in price_models.items():

        pipeline = create_model_pipeline(estimator)

        metrics, trained_model = evaluate_price_model(
            name,
            pipeline,
            X_price_train,
            y_price_train,
            X_price_validation,
            y_price_validation,
            price_validation_clean,
        )

        all_metrics.append(metrics)

    print("\n================================================")
    print("DIRECTION MODEL TRAINING")
    print("================================================")

    direction_models = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            max_features=0.7,
            min_samples_leaf=5,
            max_samples=0.7,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "XGBoost Classifier": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    for name, estimator in direction_models.items():

        pipeline = create_model_pipeline(estimator)

        metrics, trained_model = evaluate_direction_model(
            name,
            pipeline,
            X_direction_train,
            y_direction_train,
            X_direction_validation,
            y_direction_validation,
            direction_validation_clean,
        )

        all_metrics.append(metrics)

    # Save combined results
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = REPORT_DIR / "ml_model_validation_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)

    print("\n================================================")
    print("MODEL TRAINING COMPLETED")
    print("================================================")

    print(metrics_df.to_string(index=False))

    print("\nMetrics saved to:")
    print(metrics_path)

    print("\nPredictions saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
