"""
Bridge Structural Health Monitoring — Model Training Pipeline
"""

from pathlib import Path
from datetime import datetime
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "bridge_processed.csv"

ARTIFACTS_DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
METADATA_DIR = ROOT / "artifacts" / "metadata"

REPORTS_DIR = ROOT / "reports"
MONITORING_LOGS_DIR = ROOT / "monitoring" / "logs"

BEST_MODEL_PATH = MODEL_DIR / "bridge_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
MLP_MODEL_PATH = MODEL_DIR / "mlp_model.pkl"

TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.json"

MODEL_VERSION_PATH = METADATA_DIR / "model_version.txt"
TRAINING_LOG_PATH = MONITORING_LOGS_DIR / "training.log"

X_TRAIN_PATH = ARTIFACTS_DATA_DIR / "X_train.npy"
X_TEST_PATH = ARTIFACTS_DATA_DIR / "X_test.npy"
Y_TRAIN_PATH = ARTIFACTS_DATA_DIR / "y_train.npy"
Y_TEST_PATH = ARTIFACTS_DATA_DIR / "y_test.npy"

COMPARISON_PLOT_PATH = REPORTS_DIR / "model_comparison.png"
BEST_MODEL_PLOT_PATH = REPORTS_DIR / "best_model_actual_vs_predicted.png"

TARGET = "Structural_Health_Index_SHI"


MONITORING_LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=TRAINING_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message):
    print(message)
    logging.info(message)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}\n"
            "Run: python src/data_preprocessing.py"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    drop_cols = [col for col in ["Timestamp"] if col in df.columns]

    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]

    return X, y


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": round(mse, 4),
        "RMSE": round(mse ** 0.5, 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R2_Score": round(r2_score(y_true, y_pred), 4),
    }


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def save_numpy_artifacts(X_train, X_test, y_train, y_test):
    ARTIFACTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(X_TRAIN_PATH, X_train.to_numpy())
    np.save(X_TEST_PATH, X_test.to_numpy())
    np.save(Y_TRAIN_PATH, y_train.to_numpy())
    np.save(Y_TEST_PATH, y_test.to_numpy())

    log(f"Saved X_train to: {X_TRAIN_PATH}")
    log(f"Saved X_test to: {X_TEST_PATH}")
    log(f"Saved y_train to: {Y_TRAIN_PATH}")
    log(f"Saved y_test to: {Y_TEST_PATH}")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_model_version(best_model_name):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_text = f"{best_model_name}_{timestamp}"

    with open(MODEL_VERSION_PATH, "w") as file:
        file.write(version_text)

    log(f"Model version saved to: {MODEL_VERSION_PATH}")


def save_comparison_plot(metrics):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(metrics.keys())
    rmse_values = [metrics[m]["RMSE"] for m in models]
    mae_values = [metrics[m]["MAE"] for m in models]
    x = range(len(models))
    bar_width = 0.4
    plt.figure(figsize=(8, 6))
    rmse_bars = plt.bar(x, rmse_values, width=bar_width, label="RMSE")
    mae_bars = plt.bar(
        [i + bar_width for i in x],
        mae_values,
        width=bar_width,
        label="MAE"
    )
    # Add value labels above bars as chen requested
    for bar in rmse_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    for bar in mae_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    plt.xticks([i + bar_width / 2 for i in x], models, rotation=10)
    plt.ylabel("Error")
    plt.title("Model Comparison: Random Forest vs Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH)
    plt.close()
    log(f"Model comparison plot saved to: {COMPARISON_PLOT_PATH}")


def save_actual_vs_predicted_plot(y_test, y_pred, model_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual SHI")
    plt.ylabel("Predicted SHI")
    plt.title(f"Best Model Actual vs Predicted SHI ({model_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BEST_MODEL_PLOT_PATH)
    plt.close()

    log(f"Actual vs predicted plot saved to: {BEST_MODEL_PLOT_PATH}")


def main():
    log("=" * 60)
    log("BRIDGE SHI MODEL TRAINING")
    log("=" * 60)

    X, y = load_data()

    log(f"Rows: {X.shape[0]}")
    log(f"Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    save_numpy_artifacts(X_train, X_test, y_train, y_test)

    log("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = calculate_metrics(y_test, rf_pred)
    log(f"Random Forest metrics: {rf_metrics}")

    log("Training Neural Network...")
    mlp_model = train_neural_network(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_metrics = calculate_metrics(y_test, mlp_pred)
    log(f"Neural Network metrics: {mlp_metrics}")

    all_metrics = {
        "RandomForestRegressor": rf_metrics,
        "MLPRegressor_NeuralNetwork": mlp_metrics
    }

    if rf_metrics["RMSE"] <= mlp_metrics["RMSE"]:
        best_model = rf_model
        best_model_name = "RandomForestRegressor"
        best_predictions = rf_pred
    else:
        best_model = mlp_model
        best_model_name = "MLPRegressor_NeuralNetwork"
        best_predictions = mlp_pred

    final_report = {
        "target": TARGET,
        "rows_used": X.shape[0],
        "features_used": X.shape[1],
        "models_compared": all_metrics,
        "best_model": best_model_name,
        "selection_reason": "Best model selected using lowest RMSE on the test set."
    }

    save_model(rf_model, RF_MODEL_PATH)
    save_model(mlp_model, MLP_MODEL_PATH)
    save_model(best_model, BEST_MODEL_PATH)

    save_json(final_report, TRAINING_HISTORY_PATH)
    save_model_version(best_model_name)

    save_comparison_plot(all_metrics)
    save_actual_vs_predicted_plot(y_test, best_predictions, best_model_name)

    log(f"Best model: {best_model_name}")
    log(f"Best model saved to: {BEST_MODEL_PATH}")
    log(f"Training history saved to: {TRAINING_HISTORY_PATH}")
    log("Model training complete.")


if __name__ == "__main__":
    main()