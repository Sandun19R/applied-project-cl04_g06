"""
Receive High Risk Alerts
 
Runs immediately after model prediction.
Classifies the predicted SHI into four risk bands and prints/logs an alert.
 
SHI bands (0-100 scale):
    SHI >= 80         Healthy
    60 <= SHI < 80    Moderate Risk
    40 <= SHI < 60    High Risk
    SHI < 40          Critical Alert

"""
 
from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
 
ROOT       = Path(__file__).resolve().parents[1]
LOG_PATH   = ROOT / "monitoring" / "logs" / "shi_alerts.log"
ALERT_PATH = ROOT / "monitoring" / "alerts" / "shi_alerts.json"
 
for _p in [LOG_PATH.parent, ALERT_PATH.parent]:
    _p.mkdir(parents=True, exist_ok=True)
 
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shi_alerts")


@dataclass
class SHIAlert:
    timestamp:  str
    shi_score:  float
    band_label: str
    level:      str
    message:    str
    action:     str


# Classify SHI value into risk band and return SHIAlert
def classify_shi(shi_score: float, timestamp: str = None) -> SHIAlert:

    ts        = timestamp or datetime.now().isoformat()
    shi_score = float(np.clip(shi_score, 0.0, 100.0))

    if shi_score >= 80:
        alert = SHIAlert(
            timestamp  = ts,
            shi_score  = round(shi_score, 2),
            band_label = "Healthy",
            level      = "OK",
            message    = "Bridge health is within normal range.",
            action     = "No action required. Continue routine monitoring.",
        )
    elif shi_score >= 60:
        alert = SHIAlert(
            timestamp  = ts,
            shi_score  = round(shi_score, 2),
            band_label = "Moderate Risk",
            level      = "WARNING",
            message    = "Moderate Bridge Health Risk Detected",
            action     = "Schedule inspection within 7 days. Increase monitoring frequency.",
        )
    elif shi_score >= 40:
        alert = SHIAlert(
            timestamp  = ts,
            shi_score  = round(shi_score, 2),
            band_label = "High Risk",
            level      = "HIGH",
            message    = "High Bridge Health Risk Detected",
            action     = "Inspection required within 24 hours. Consider load restrictions.",
        )
    else:
        alert = SHIAlert(
            timestamp  = ts,
            shi_score  = round(shi_score, 2),
            band_label = "Critical",
            level      = "CRITICAL",
            message    = "Critical Bridge Health Detected",
            action     = "Immediate inspection recommended. Notify structural engineer on-call.",
        )

    if alert.level != "OK":
        log_fn = logger.critical if alert.level == "CRITICAL" else logger.warning
        log_fn(f"[{alert.level}] SHI={shi_score:.1f} — {alert.message}")

    return alert

# Run SHI alert classification over entire prediction dataframe
def run_alerts(
    predictions_df: pd.DataFrame,
    shi_col:        str = "Structural_Health_Index_SHI",
    timestamp_col:  str = "Timestamp",
) -> list[SHIAlert]:
    if shi_col not in predictions_df.columns:
        raise ValueError(f"Column '{shi_col}' not found in predictions DataFrame.")

    shi_values = predictions_df[shi_col].values

    # Convert 0-1 model output to 0-100 scale if needed
    if shi_values.max() <= 1.0:
        shi_values = shi_values * 100

    timestamps = (
        predictions_df[timestamp_col].astype(str).tolist()
        if timestamp_col in predictions_df.columns
        else [None] * len(predictions_df)
    )

    alerts: list[SHIAlert] = []
    for shi, ts in zip(shi_values, timestamps):
        alert = classify_shi(float(shi), timestamp=ts)
        alerts.append(alert)

    non_ok = [a for a in alerts if a.level != "OK"]

    # Append non-OK alerts to persistent JSON log
    existing: list[dict] = []
    if ALERT_PATH.exists():
        try:
            with open(ALERT_PATH) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
    with open(ALERT_PATH, "w") as f:
        json.dump(existing + [asdict(a) for a in non_ok], f, indent=2)

    # Log summary
    logger.info(
        f"Processed {len(alerts)} predictions. "
        f"Saved {len(non_ok)} alerts to {ALERT_PATH}"
    )

    return alerts


# Main
def main():
    import joblib

    MODEL_PATH = ROOT / "artifacts" / "models" / "bridge_model.pkl"
    DATA_PATH  = ROOT / "data" / "processed" / "bridge_processed.csv"

    # Live: run on model predictions if model exists
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return

    model  = joblib.load(MODEL_PATH)
    df     = pd.read_csv(DATA_PATH).dropna()
    split  = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)

    TARGET   = "Structural_Health_Index_SHI"
    drop_cols = [c for c in [TARGET, "Timestamp"] if c in df_test.columns]
    X_test   = df_test.drop(columns=drop_cols)

    shi_preds = model.predict(X_test)
    predictions_df = df_test.copy()
    predictions_df[TARGET] = shi_preds

    run_alerts(predictions_df)

if __name__ == "__main__":
    main()
