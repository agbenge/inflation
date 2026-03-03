import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import torch 
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)


start_time = datetime.now()
results_folder="results"
print("Start Time:", start_time)
from darts import TimeSeries
from darts.models import SKLearnModel
from sklearn.linear_model import Lasso
from darts.models import (
    ARIMA,
    Prophet,
    RandomForestModel,
    XGBModel,
    LightGBMModel,
    RNNModel,
    NBEATSModel
)
from darts.metrics import mae, rmse, mape, smape
from darts.dataprocessing.transformers import Scaler

# --------------------------
# 1. Load Data
# --------------------------
df = pd.read_excel("data/inner_join.xlsx")
# Create proper date column
df['date'] = pd.to_datetime({
    'year': df['tyear'],
    'month': df['tmonth'],
    'day': 1
})
df_unique = df.drop_duplicates(subset=['date'])
# Convert to TimeSeries, fill missing months
series = TimeSeries.from_dataframe(
   df_unique,
    time_col='date',
    value_cols='foodAverage',
    fill_missing_dates=True,
    freq='MS'
)

# Scale (important for ML/DL models)
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# --------------------------
# 2. Forecast Settings
# --------------------------
forecast_horizon = 6
start_point = 0.6  # start after 60% of data

# --------------------------
# 3. Define Models
# --------------------------
models = {
    # "ARIMA": ARIMA(),
#     "LASSO":  SKLearnModel(
#     lags=12,
#     model=Lasso(alpha=0.1)
# ),
    # "Prophet": Prophet(changepoint_prior_scale=0.15, yearly_seasonality=4, seasonality_mode="additive"),
     
    #  "RandomForest": RandomForestModel(lags=12,
    # n_estimators=300,
    # max_depth=8,
    # min_samples_split=3,
    # min_samples_leaf=2,
    # max_features=0.8,
    # output_chunk_length=1,
    # bootstrap=True,
    # random_state=42,
    # n_jobs=-1),
    # "XGBoost": XGBModel( lags=12,
    # output_chunk_length=1,
    # n_estimators=300,
    # max_depth=5,
    # learning_rate=0.05,
    # subsample=0.8,
    # colsample_bytree=0.8,
    # gamma=0.1,
    # min_child_weight=3,
    # reg_alpha=0.1,
    # reg_lambda=1.0,
    # objective='reg:squarederror',
    # random_state=42,
    # n_jobs=-1),
   "LightGBM": LightGBMModel(
    lags=12,
    output_chunk_length=1,
    n_estimators=300,
    max_depth=6,
    num_leaves=25,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)
    # "LSTM": RNNModel(
    #     model="LSTM",
    #     input_chunk_length=12,
    #     output_chunk_length=forecast_horizon,
    #     n_epochs=100,
    #     random_state=42
    # ),
    # "NBEATS": NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=forecast_horizon,
    #     n_epochs=100,
    #     random_state=42
    # )
}

results = {}

# --------------------------
# 4. Rolling Backtesting (FIXED)
# --------------------------
for name, model in models.items(): 
    if name in ["ARIMA", "Prophet"]:
        forecast = model.historical_forecasts(
            series,
            start=start_point,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=True,
            verbose=True
        )
        actual = series.slice_intersect(forecast)

    else:
        forecast_scaled = model.historical_forecasts(
            series_scaled,
            start=start_point,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=True,
            verbose=True
        )
        forecast = scaler.inverse_transform(forecast_scaled)
        actual = series.slice_intersect(forecast)

    results[name] = {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "MAPE": mape(actual, forecast),
        "sMAPE": smape(actual, forecast),
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
}).T.sort_values("RMSE")

print("\nProfessional Model Comparison:")
print(metrics_df)

# --------------------------
# 6. Plot Best Model
# --------------------------
best_model = metrics_df.index[0]
print(f"\nBest Model: {best_model}")

# plt.figure(figsize=(12, 6))
# series.plot(label="Actual")
# results[best_model]["forecast"].plot(label=f"{best_model} Forecast")
# plt.legend()
# plt.title("Best Model Rolling Forecast")
# plt.show()
