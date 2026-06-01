"""
Bridge Structural Health Monitoring — Data Preprocessing Pipeline
"""

from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw_dataset" / "bridge_digital_twin_dataset.csv"

CLEANED_OUTPUT_PATH = ROOT / "data" / "cleaned" / "bridge_cleaned.csv"
PROCESSED_OUTPUT_PATH = ROOT / "data" / "processed" / "bridge_processed.csv"

ARTIFACTS_DATA_DIR = ROOT / "artifacts" / "data"
PREPROCESSING_DIR = ROOT / "artifacts" / "preprocessing"
METADATA_DIR = ROOT / "artifacts" / "metadata"

SCALER_OUTPUT_PATH = PREPROCESSING_DIR / "scaler.pkl"
ENCODER_OUTPUT_PATH = PREPROCESSING_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH = PREPROCESSING_DIR / "feature_columns.json"
X_NEW_PATH = ARTIFACTS_DATA_DIR / "X_new.npy"
DATA_VERSION_PATH = METADATA_DIR / "data_version.txt"

MONITORING_LOGS_DIR = ROOT / "monitoring" / "logs"
PREPROCESSING_LOG_PATH = MONITORING_LOGS_DIR / "preprocessing.log"

TARGET = "Structural_Health_Index_SHI"

COLS_DROP_MISSING = [
    "Vibration_Anomaly_Location"
]

COLS_DROP_LEAKAGE = [
    "SHI_Predicted_24h_Ahead",
    "SHI_Predicted_7d_Ahead",
    "SHI_Predicted_30d_Ahead",
    "Bridge_Mood_Meter",
    "Probability_of_Failure_PoF",
    "Maintenance_Alert",
    "Estimated_Repair_Cost_USD_incremental",
    "Carbon_Footprint_tCO2e_incremental",
    "Energy_Harvesting_Potential_W",
    "Simulated_Water_Flow_m3s",
    "Axle_Counts_pmin",
    "Air_Quality_Index_AQI",
]

Z_THRESHOLD = 3.0
IQR_FACTOR = 1.5


MONITORING_LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=PREPROCESSING_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message):
    print(message)
    logging.info(message)


def load_dataset(path: Path) -> pd.DataFrame:
    log("=" * 60)
    log("LOAD DATASET")
    log("=" * 60)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        log(f"Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

    log(f"Loaded dataset: {path}")
    log(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    log(f"Duplicates removed: {before - len(df)}")
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = COLS_DROP_MISSING + COLS_DROP_LEAKAGE
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns)

    log(f"Columns dropped: {len(existing_columns)}")
    for col in existing_columns:
        log(f"- {col}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    missing_before = int(df.isnull().sum().sum())
    log(f"Missing values before: {missing_before}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "Timestamp"]

    if numeric_cols:
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = int(df.isnull().sum().sum())
    log(f"Missing values after: {missing_after}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    skip_cols = [
        TARGET,
        "Flood_Event_Flag",
        "Anomaly_Detection_Score",
        "Localized_Strain_Hotspot",
        "High_Winds_Storms",
        "Abnormal_Traffic_Load_Surges",
        "Landslide_Ground_Movement",
        "Impact_Events_g",
        "Seismic_Activity_ms2",
    ]

    treat_cols = [col for col in numeric_cols if col not in skip_cols]
    total_fixed = 0

    for col in treat_cols:
        if df[col].dropna().shape[0] < 30:
            continue

        z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        z_mask = z_scores > Z_THRESHOLD

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        iqr_mask = (df[col] < q1 - IQR_FACTOR * iqr) | (
            df[col] > q3 + IQR_FACTOR * iqr
        )

        combined_mask = z_mask & iqr_mask
        flagged_count = int(combined_mask.sum())

        if flagged_count > 0:
            df.loc[combined_mask, col] = np.nan
            total_fixed += flagged_count

    if treat_cols:
        df[treat_cols] = (
            df[treat_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    log(f"Total outlier values corrected: {total_fixed}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "Timestamp"]

    encoders = {}

    if not categorical_cols:
        log("No categorical columns to encode.")
        PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, ENCODER_OUTPUT_PATH)
        return df

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col + "_enc"] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        df = df.drop(columns=[col])
        log(f"Encoded: {col} -> {col}_enc")

    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, ENCODER_OUTPUT_PATH)
    log(f"Encoder saved to: {ENCODER_OUTPUT_PATH}")
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    df_unscaled = df.copy()
    df_scaled = df.copy()

    no_scale_cols = [
        "Timestamp",
        TARGET,
        "Flood_Event_Flag",
        "High_Winds_Storms",
        "Abnormal_Traffic_Load_Surges",
        "Landslide_Ground_Movement",
    ]

    no_scale_cols = [col for col in no_scale_cols if col in df_scaled.columns]

    scale_cols = [
        col for col in df_scaled.select_dtypes(include="number").columns
        if col not in no_scale_cols
    ]

    scaler = StandardScaler()

    if scale_cols:
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_OUTPUT_PATH)

    log(f"Scaled columns: {len(scale_cols)}")
    log(f"Scaler saved to: {SCALER_OUTPUT_PATH}")
    return df_scaled, df_unscaled, scale_cols


def save_feature_columns(df_scaled: pd.DataFrame):
    feature_df = df_scaled.drop(columns=[TARGET], errors="ignore")
    feature_df = feature_df.drop(columns=["Timestamp"], errors="ignore")

    feature_columns = feature_df.columns.tolist()

    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_COLUMNS_PATH, "w") as file:
        json.dump(feature_columns, file, indent=4)

    log(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    return feature_columns


def save_x_new(df_scaled: pd.DataFrame, feature_columns: list):
    ARTIFACTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_new = df_scaled[feature_columns].to_numpy()
    np.save(X_NEW_PATH, X_new)

    log(f"X_new saved to: {X_NEW_PATH}")


def save_data_version(df: pd.DataFrame):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    version_text = f"rows={df.shape[0]}, columns={df.shape[1]}"

    with open(DATA_VERSION_PATH, "w") as file:
        file.write(version_text)

    log(f"Data version saved to: {DATA_VERSION_PATH}")


def validate_output(df_clean: pd.DataFrame, raw_shape: tuple) -> None:
    log("=" * 60)
    log("VALIDATION REPORT")
    log("=" * 60)

    log(f"Original shape: {raw_shape}")
    log(f"Cleaned shape: {df_clean.shape}")
    log(f"Missing values: {df_clean.isnull().sum().sum()}")
    log(f"Duplicate rows: {df_clean.duplicated().sum()}")

    if TARGET in df_clean.columns:
        log("Target summary:")
        logging.info(df_clean[TARGET].describe().to_string())


def save_outputs(df_scaled: pd.DataFrame, df_unscaled: pd.DataFrame) -> None:
    CLEANED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_unscaled.to_csv(CLEANED_OUTPUT_PATH, index=False)
    df_scaled.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    log(f"Cleaned dataset saved to: {CLEANED_OUTPUT_PATH}")
    log(f"Processed dataset saved to: {PROCESSED_OUTPUT_PATH}")


def run_pipeline(input_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\n" + "=" * 60)
    log("BRIDGE DATA PREPROCESSING PIPELINE")
    log("=" * 60)

    df = load_dataset(input_path)
    raw_shape = df.shape

    df = remove_duplicates(df)
    df = drop_unusable_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)

    df_scaled, df_unscaled, _ = scale_features(df)

    feature_columns = save_feature_columns(df_scaled)
    save_x_new(df_scaled, feature_columns)
    save_data_version(df_scaled)

    validate_output(df_unscaled, raw_shape)
    save_outputs(df_scaled, df_unscaled)

    log("Preprocessing pipeline complete.")
    return df_scaled, df_unscaled


if __name__ == "__main__":
    import sys

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DATA_PATH
    run_pipeline(input_path)