import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import (
    ARIMA,
    Prophet,
    RandomForest,
    XGBModel,
    LightGBMModel,
    RNNModel,
    NBEATSModel
)
from darts.metrics import mae, rmse, mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import backtest_forecasting

# --------------------------
# 1. Load Data
# --------------------------
df = pd.read_csv("inflation.csv")
df["date"] = pd.to_datetime(df["date"])
series = TimeSeries.from_dataframe(df, "date", "inflation")

# Scale (important for ML/DL)
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# --------------------------
# 2. Define Forecast Settings
# --------------------------
forecast_horizon = 6     # 6 months ahead
start_point = 0.6        # start backtesting after 60% of data

# --------------------------
# 3. Define Models
# --------------------------
models = {
    "ARIMA": ARIMA(),
    "Prophet": Prophet(),
    "RandomForest": RandomForest(lags=12),
    "XGBoost": XGBModel(lags=12),
    "LightGBM": LightGBMModel(lags=12),
    "LSTM": RNNModel(
        model="LSTM",
        input_chunk_length=12,
        output_chunk_length=forecast_horizon,
        n_epochs=100,
        random_state=42
    ),
    "NBEATS": NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=forecast_horizon,
        n_epochs=100,
        random_state=42
    )
}

results = {}

# --------------------------
# 4. Rolling Backtesting
# --------------------------
for name, model in models.items():
    print(f"Backtesting {name}...")

    if name in ["ARIMA", "Prophet"]:
        forecast = backtest_forecasting(
            model,
            series,
            start=start_point,
            forecast_horizon=forecast_horizon,
            retrain=True,
            verbose=True
        )
        actual_series = series.slice_intersect(forecast)
    else:
        forecast_scaled = backtest_forecasting(
            model,
            series_scaled,
            start=start_point,
            forecast_horizon=forecast_horizon,
            retrain=True,
            verbose=True
        )
        forecast = scaler.inverse_transform(forecast_scaled)
        actual_series = series.slice_intersect(forecast)

    results[name] = {
        "MAE": mae(actual_series, forecast),
        "RMSE": rmse(actual_series, forecast),
        "MAPE": mape(actual_series, forecast),
        "sMAPE": smape(actual_series, forecast),
        "forecast": forecast
    }

# --------------------------
# 5. Compare Metrics
# --------------------------
metrics_df = pd.DataFrame({
    model: {
        "MAE": results[model]["MAE"],
        "RMSE": results[model]["RMSE"],
        "MAPE": results[model]["MAPE"],
        "sMAPE": results[model]["sMAPE"]
    }
    for model in results
}).T

metrics_df = metrics_df.sort_values("RMSE")

print("\nProfessional Model Comparison:")
print(metrics_df)

# --------------------------
# 6. Plot Best Model
# --------------------------
best_model = metrics_df.index[0]
print(f"\nBest Model: {best_model}")

plt.figure(figsize=(12,6))
series.plot(label="Actual")
results[best_model]["forecast"].plot(label=f"{best_model} Forecast")
plt.legend()
plt.title("Best Model Rolling Forecast")
plt.show()
