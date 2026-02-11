import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    Prophet,
    RandomForest,
    XGBModel,
    LightGBMModel,
    RNNModel,
    NBEATSModel
)

from darts.metrics import mae, rmse, mape
from darts.dataprocessing.transformers import Scaler

# ---------------------------
# 1. Load Data
# ---------------------------
df = pd.read_csv("inflation.csv")
df["date"] = pd.to_datetime(df["date"])

series = TimeSeries.from_dataframe(df, "date", "inflation")

# ---------------------------
# 2. Train/Test Split
# ---------------------------
train, test = series.split_before(0.8)

# Scale data (important for ML & DL)
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# ---------------------------
# 3. Initialize Models
# ---------------------------
models = {
    "ARIMA": ARIMA(),
    "Prophet": Prophet(),
    "RandomForest": RandomForest(lags=12),
    "XGBoost": XGBModel(lags=12),
    "LightGBM": LightGBMModel(lags=12),
    "LSTM": RNNModel(
        model="LSTM",
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=100,
        random_state=42
    ),
    "NBEATS": NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=100,
        random_state=42
    )
}

results = {}

# ---------------------------
# 4. Train & Predict
# ---------------------------
for name, model in models.items():
    print(f"Training {name}...")
    
    if name in ["ARIMA", "Prophet"]:
        model.fit(train)
        forecast = model.predict(len(test))
    else:
        model.fit(train_scaled)
        forecast_scaled = model.predict(len(test))
        forecast = scaler.inverse_transform(forecast_scaled)
    
    results[name] = {
        "forecast": forecast,
        "MAE": mae(test, forecast),
        "RMSE": rmse(test, forecast),
        "MAPE": mape(test, forecast)
    }

# ---------------------------
# 5. Print Metrics
# ---------------------------
metrics_df = pd.DataFrame({
    model: {
        "MAE": results[model]["MAE"],
        "RMSE": results[model]["RMSE"],
        "MAPE": results[model]["MAPE"]
    }
    for model in results
}).T

print("\nModel Comparison:")
print(metrics_df.sort_values("RMSE"))

# ---------------------------
# 6. Plot Results
# ---------------------------
plt.figure(figsize=(12, 6))
train.plot(label="Train")
test.plot(label="Actual", linewidth=2)

for name in results:
    results[name]["forecast"].plot(label=name)

plt.legend()
plt.title("Inflation Forecast Comparison")
plt.show()
