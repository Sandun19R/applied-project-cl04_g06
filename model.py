import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("bridge_digital_twin_dataset(small).csv")

# Target column
target = "Structural_Health_Index_SHI"

# Features
features = [
    "Strain_microstrain",
    "Deflection_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Temperature_C",
    "Humidity_percent",
    "Wind_Speed_ms",
    "Vehicle_Load_tons"
]
# Clean dataset
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SHI")
plt.ylabel("Predicted SHI")
plt.title("Bridge SHI Prediction")
plt.grid(True)
plt.savefig("model_results.png")

# Save metrics
with open("metrics.txt", "w") as f:
    f.write("Bridge Regression Model Results\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

print("Model completed successfully")
