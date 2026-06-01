# Bridge Structural Health Monitoring — Model Evaluation Script
from pathlib import Path
import json
import logging
import joblib
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts" / "models" / "bridge_model.pkl"

X_TEST_PATH = ROOT / "artifacts" / "data" / "X_test.npy"
Y_TEST_PATH = ROOT / "artifacts" / "data" / "y_test.npy"

METRICS_PATH = ROOT / "artifacts" / "metrics" / "evaluation_metrics.json"

LOGS_DIR = ROOT / "monitoring" / "logs"
EVALUATION_LOG_PATH = LOGS_DIR / "evaluation.log"


LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=EVALUATION_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message):
    print(message)
    logging.info(message)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"X_test not found: {X_TEST_PATH}")

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"y_test not found: {Y_TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    return model, X_test, y_test


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": round(mse, 4),
        "RMSE": round(mse ** 0.5, 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R2_Score": round(r2_score(y_true, y_pred), 4),
    }


def save_metrics(metrics):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    log(f"Evaluation metrics saved to: {METRICS_PATH}")


def main():
    log("=" * 60)
    log("MODEL EVALUATION")
    log("=" * 60)

    model, X_test, y_test = load_artifacts()

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    log(f"Evaluation metrics: {metrics}")

    save_metrics(metrics)

    log("Evaluation complete.")


if __name__ == "__main__":
    main()