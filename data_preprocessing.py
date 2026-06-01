"""
Bridge Structural Health Monitoring — Data Preprocessing Pipeline

Steps covered:
1.  Load & initial inspection
2.  Remove duplicates
3.  Drop unrecoverable columns (>50% missing)
4.  Identify & handle missing values (interpolation / mode-fill)
5.  Detect & correct outliers (Z-score ∩ IQR double-filter)
6.  Drop leakage / output-only features
7.  Encode categorical variables
8.  Scale numeric features
9.  Post-cleaning validation report
10. Save cleaned dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_PATH  = "bridge_digital_twin_dataset.csv"
OUTPUT_PATH = "bridge_digital_twin_dataset_cleaned.csv"

# Columns to drop outright (>50% missing or data-leakage / output-only)
COLS_DROP_MISSING  = ["Vibration_Anomaly_Location"]   # 79% missing, categorical
COLS_DROP_LEAKAGE  = [
    "Bridge_Mood_Meter",                      # engineered label — leakage risk
    "Energy_Harvesting_Potential_W",          # derived output, not a predictor
    "Carbon_Footprint_tCO2e_incremental",     # output metric
    "Estimated_Repair_Cost_USD_incremental",  # target-derived output
    "SHI_Predicted_7d_Ahead",                 # collinear with 24h
    "SHI_Predicted_30d_Ahead",                # collinear with 24h
    "Simulated_Water_Flow_m3s",               # collinear with Flood_Event_Flag
    "Axle_Counts_pmin",                       # collinear with Traffic_Volume_vph
    "Air_Quality_Index_AQI",                  # very weak correlation, unrelated
]

# Categorical columns remaining after drops
CATEGORICAL_COLS = []  # Vibration_Anomaly_Location already dropped above

# Target column
TARGET = "Structural_Health_Index_SHI"

# Outlier thresholds
Z_THRESHOLD   = 3.0   # Z-score threshold
IQR_FACTOR    = 1.5   # IQR fence multiplier


# ─────────────────────────────────────────────
# LOAD & INITIAL INSPECTION
# ─────────────────────────────────────────────
def load_and_inspect(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("LOAD & INITIAL INSPECTION")
    print("=" * 60)

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    print(f"  Shape          : {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Date range     : {df['Timestamp'].min()} ->  {df['Timestamp'].max()}")
    print(f"  Numeric cols   : {df.select_dtypes(include='number').shape[1]}")
    print(f"  Non-numeric    : {df.select_dtypes(exclude='number').shape[1]}")
    print()
    return df


# ─────────────────────────────────────────────
# REMOVE DUPLICATES
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("DUPLICATE REMOVAL")
    print("=" * 60)

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"  Duplicates removed : {dropped}")
    print(f"  Shape after        : {df.shape}")
    print()
    return df


# ─────────────────────────────────────────────
# DROP UNRECOVERABLE / LEAKAGE COLUMNS
# ─────────────────────────────────────────────
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("DROP UNRECOVERABLE / LEAKAGE COLUMNS")
    print("=" * 60)

    all_drop = COLS_DROP_MISSING + COLS_DROP_LEAKAGE
    existing  = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=existing)

    print(f"  Columns dropped ({len(existing)}):")
    for c in existing:
        reason = "missing >50%" if c in COLS_DROP_MISSING else "leakage/redundant"
        print(f"   - {c}  [{reason}]")
    print(f"  Shape after : {df.shape}")
    print()
    return df


# ─────────────────────────────────────────────
# HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("MISSING VALUE IMPUTATION")
    print("=" * 60)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"  Columns with missing values: {len(missing)}")
    print(f"  Total missing cells        : {missing.sum():,}")
    print()

    # Sort by Timestamp to preserve time-series order
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    category_cols = df.select_dtypes(exclude="number").columns.tolist()
    category_cols = [c for c in category_cols if c != "Timestamp"]

    # Numeric: linear interpolation, then forward/back fill for edge NaNs
    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    # Categorical: mode fill
    for col in category_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Mode-filled  : {col}  (mode='{mode_val}')")

    remaining = df.isnull().sum().sum()
    print(f"\n  Remaining missing after imputation: {remaining}")
    print(f"  Shape after : {df.shape}")
    print()
    return df


# ─────────────────────────────────────────────
# OUTLIER DETECTION & CORRECTION
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("OUTLIER DETECTION & CORRECTION  (Z-score ∩ IQR)")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Exclude binary/flag columns and the target from outlier treatment
    skip_cols = [
        TARGET,
        "Flood_Event_Flag", "Maintenance_Alert", "Anomaly_Detection_Score",
        "Localized_Strain_Hotspot", "High_Winds_Storms",
        "Abnormal_Traffic_Load_Surges", "Landslide_Ground_Movement",
        "Impact_Events_g", "Seismic_Activity_ms2",
    ]
    treat_cols = [c for c in numeric_cols if c not in skip_cols]

    total_replaced = 0
    report_rows = []

    for col in treat_cols:
        series = df[col].dropna()
        if len(series) < 30:
            continue

        # Z-score mask
        z_scores  = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        z_mask    = z_scores > Z_THRESHOLD

        # IQR mask
        Q1, Q3    = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR       = Q3 - Q1
        iqr_mask  = (df[col] < Q1 - IQR_FACTOR * IQR) | (df[col] > Q3 + IQR_FACTOR * IQR)

        # Intersection: flagged by both
        combined  = z_mask & iqr_mask
        n_flagged = combined.sum()

        if n_flagged > 0:
            df.loc[combined, col] = np.nan
            total_replaced += n_flagged
            report_rows.append((col, n_flagged, round(n_flagged / len(df) * 100, 2)))

    # Re-interpolate the newly created NaNs
    treat_df = df[treat_cols]
    df[treat_cols] = (
        treat_df
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    print(f"  Columns treated     : {len(report_rows)}")
    print(f"  Total values fixed  : {total_replaced:,}")
    print()
    return df


# ─────────────────────────────────────────────
# ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("ENCODE CATEGORICAL VARIABLES")
    print("=" * 60)

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Timestamp"]

    if not cat_cols:
        print("  No categorical columns remaining — skipping.")
        print()
        return df

    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        df = df.drop(columns=[col])
        print(f"  Label-encoded: {col}  ->  {col}_enc  ({df[col+'_enc'].nunique()} classes)")

    print()
    return df


# ─────────────────────────────────────────────
# FEATURE SCALING
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_scaled   — standardised numeric features + target
        df_unscaled — original cleaned values (useful for interpretation)
    """
    print("=" * 60)
    print("FEATURE SCALING  (StandardScaler)")
    print("=" * 60)

    df_unscaled = df.copy()

    # Columns not to scale: Timestamp, binary flags, the target itself
    no_scale = ["Timestamp", TARGET,
                 "Flood_Event_Flag", "Maintenance_Alert",
                 "High_Winds_Storms", "Abnormal_Traffic_Load_Surges",
                 "Landslide_Ground_Movement"]
    no_scale = [c for c in no_scale if c in df.columns]

    scale_cols = [c for c in df.select_dtypes(include="number").columns
                  if c not in no_scale]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    print(f"  Columns scaled  : {len(scale_cols)}")
    print(f"  Columns kept as-is: {no_scale}")
    print()
    return df, df_unscaled


# ─────────────────────────────────────────────
# POST-CLEANING VALIDATION REPORT
# ─────────────────────────────────────────────
def validate(df_clean: pd.DataFrame, df_raw_shape: tuple) -> None:
    print("=" * 60)
    print("POST-CLEANING VALIDATION")
    print("=" * 60)

    print(f"  Original shape  : {df_raw_shape[0]:,} rows, {df_raw_shape[1]} cols")
    print(f"  Cleaned shape   : {df_clean.shape[0]:,} rows, {df_clean.shape[1]} cols")
    print(f"  Rows retained   : {df_clean.shape[0] / df_raw_shape[0] * 100:.1f}%")
    print(f"  Missing values  : {df_clean.isnull().sum().sum()}")
    print(f"  Duplicate rows  : {df_clean.duplicated().sum()}")
    print()
    print("  Target variable summary:")
    target_stats = df_clean[TARGET].describe()
    for stat, val in target_stats.items():
        print(f"    {stat:10s} : {val:.4f}")
    print()

# ─────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────
def save(df: pd.DataFrame, path: str) -> None:
    print("=" * 60)
    print("SAVE DATASET")
    print("=" * 60)
    df.to_csv(path, index=False)
    print(f"  Saved cleaned dataset → {path}")
    print(f"  Final shape          : {df.shape[0]:,} rows, {df.shape[1]} cols")
    print()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(input_path: str = INPUT_PATH,
                 output_path: str = OUTPUT_PATH) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    df = load_and_inspect(input_path)
    raw_shape = df.shape

    df = remove_duplicates(df)
    df = drop_columns(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    df_scaled, df_unscaled = scale_features(df)

    validate(df_unscaled, raw_shape)   # validate on unscaled for interpretability
    save(df_unscaled, output_path)     # save unscaled by default

    print("  Pipeline complete.\n")
    return df_scaled, df_unscaled


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_scaled, df_clean = run_pipeline()