"""
Detects anomalies 
    - Z-score detection  (|z| > 3)
    - IQR detection      (outside Q1 ± 1.5*IQR)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]

BASELINE_CSV = ROOT / "data" / "processed" / "bridge_processed.csv"
UPDATED_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset_updated.csv"

# Fallback datasets if updated dataset is missing
if not UPDATED_DATA_PATH.exists():
    for fallback in [ROOT / "data" / "new_data.csv", ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"]:
        if fallback.exists():
            UPDATED_DATA_PATH = fallback
            break

STATS_PATH = ROOT / "monitoring" / "alerts" / "sensor_stats.json"
RESULTS_PATH = ROOT / "monitoring" / "alerts" / "anomaly_results.csv"
LOG_PATH = ROOT / "monitoring" / "logs" / "anomaly_detection.log"

for path in [STATS_PATH.parent, LOG_PATH.parent]:
    path.mkdir(parents=True, exist_ok=True)


# Logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("anomaly_detection")

# Configuration
MONITORED_SENSORS = [
    "Strain_microstrain",
    "Deflection_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Displacement_mm",
    "Crack_Propagation_mm",
    "Corrosion_Level_percent",
    "Cable_Member_Tension_kN",
    "Bearing_Joint_Forces_kN",
    "Fatigue_Accumulation_au",
    "Modal_Frequency_Hz",
    "Temperature_C",
    "Humidity_percent",
    "Wind_Speed_ms",
    "Vehicle_Load_tons",
    "Acoustic_Emissions_levels",
]

Z_THRESHOLD = 3.0
IQR_FACTOR = 1.5

SHI_COL = "Structural_Health_Index_SHI"
HEALTHY_SHI_THRESHOLD = 0.75


# Data Class
@dataclass
class SensorAnomaly:
    sensor: str
    value: float
    method: str
    detail: str


# Detector
class AnomalyDetector:

    def __init__(
        self,
        z_threshold=Z_THRESHOLD,
        iqr_factor=IQR_FACTOR,
        sensors=None,
    ):
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor
        self.sensors = sensors or MONITORED_SENSORS
        self._stats = {}

    # Learn baseline ranges
    def fit(self, baseline_df=None):

        if baseline_df is None:

            if not BASELINE_CSV.exists():
                raise FileNotFoundError(
                    f"Baseline dataset not found at {BASELINE_CSV}.\n"
                    "Please run 'python src/data_preprocessing.py' first to generate the processed baseline data."
                )

            baseline_df = pd.read_csv(BASELINE_CSV)

        baseline_df = baseline_df.dropna()

        if SHI_COL in baseline_df.columns:
            baseline_df = baseline_df[
                baseline_df[SHI_COL] >= HEALTHY_SHI_THRESHOLD
            ]

        available = [
            s for s in self.sensors
            if s in baseline_df.columns
        ]

        for sensor in available:

            col = baseline_df[sensor]

            q1 = float(col.quantile(0.25))
            q3 = float(col.quantile(0.75))

            iqr = q3 - q1

            mean = float(col.mean())
            std = float(col.std())

            self._stats[sensor] = {
                "mean": mean,
                "std": std,
                "iqr_lower": q1 - self.iqr_factor * iqr,
                "iqr_upper": q3 + self.iqr_factor * iqr,
                "z_lower": mean - self.z_threshold * std,
                "z_upper": mean + self.z_threshold * std,
            }

        with open(STATS_PATH, "w") as f:
            json.dump(self._stats, f, indent=2)

        logger.info(
            f"Baseline fitted using {len(baseline_df)} rows"
        )

        return self

    # Load saved ranges
    def load(self):

        if not STATS_PATH.exists():
            raise FileNotFoundError(
                f"Stats file not found: {STATS_PATH}"
            )

        with open(STATS_PATH) as f:
            self._stats = json.load(f)

        return self

    # Check one row
    def _check_row(self, reading):

        anomalies = []

        for sensor, stats in self._stats.items():
            value = reading.get(sensor)
            
            if pd.isna(value):
                continue
            
            value = float(value)
            z = (
                value - stats["mean"]
            ) / (stats["std"] + 1e-9)

            if abs(z) > self.z_threshold:
                anomalies.append(
                    SensorAnomaly(
                        sensor=sensor,
                        value=value,
                        method="z-score",
                        detail=f"z={z:.2f}",
                    )
                )

                continue

            if (
                value < stats["iqr_lower"]
                or value > stats["iqr_upper"]
            ):
                anomalies.append(
                    SensorAnomaly(
                        sensor=sensor,
                        value=value,
                        method="IQR",
                        detail="outside IQR range",
                    )
                )

        return anomalies

    # Check entire dataset
    def detect(self, df):

        if not self._stats:
            raise RuntimeError(
                "Call fit() or load() first."
            )

        records = []

        for idx, row in df.iterrows():

            anomalies = self._check_row(row.to_dict())

            records.append({
                "row_index": idx,
                "is_anomaly": len(anomalies) > 0,
                "n_anomalies": len(anomalies),
                "flagged_sensors": ",".join(
                    a.sensor for a in anomalies
                ),
                "methods": ",".join(
                    sorted({a.method for a in anomalies})
                ),
            })

        result_df = pd.DataFrame(records)

        result_df.to_csv(
            RESULTS_PATH,
            index=False
        )

        logger.info(
            f"Checked {len(result_df)} rows. "
            f"Anomalous rows={int(result_df['is_anomaly'].sum())}"
        )

        return result_df


def main():

    detector = AnomalyDetector()

    if STATS_PATH.exists():
        detector.load()
    else:
        detector.fit()

    new_df = pd.read_csv(UPDATED_DATA_PATH)
    detector.detect(new_df)


if __name__ == "__main__":
    main()