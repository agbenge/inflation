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
series=series.astype(np.float32)

# Scale (important for ML/DL models)
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# --------------------------
# 2. Forecast Settings
# --------------------------
forecast_horizon = 6
start_point = 0.6   # start after 60% of data

# --------------------------
# 3. Define Models
# --------------------------
models = {
    # Statistical & Traditional ML (Fast)
    "ARIMA": ARIMA(),
    "RandomForest": RandomForest(lags=12),
    "XGBoost": XGBModel(lags=12, torch_dtype=torch.float32),
    "LightGBM": LightGBMModel(lags=12),

    # --- Slower Models (Uncomment to use) ---
    
    # "Prophet": Prophet(), # Can be slow due to Bayesian optimization
    
    # "LSTM": RNNModel(
    #     model="LSTM",
    #     input_chunk_length=12,
    #     output_chunk_length=forecast_horizon,
    #     n_epochs=100,  # 100 epochs takes significant time without a GPU
    #     random_state=42, 
    # ),
    
    # "NBEATS": NBEATSModel(
    #     input_chunk_length=12,
    #     output_chunk_length=forecast_horizon,
    #     n_epochs=100, # Very computationally intensive
    #     random_state=42, 
   #)

}

results = {}

# --------------------------
# 4. Rolling Backtesting (FIXED)
# --------------------------
for name, model in models.items():
    print(f"Backtesting {name}...")

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

print("\nModel Comparison:")
print(metrics_df)
for model_name in results:
    plt.figure(figsize=(12, 6))
    
    # Plot data
    series.plot(label="Actual", color="black", lw=1.5)
    results[model_name]["forecast"].plot(label=f"{model_name} Forecast", alpha=0.8)
    
    # Extract metrics for this specific model
    m_mae = metrics_df.loc[model_name, "MAE"]
    m_rmse = metrics_df.loc[model_name, "RMSE"]
    
    # Create a text string for the metrics box
    stats_text = f"MAE: {m_mae:.4f}\nRMSE: {m_rmse:.4f}"
    
    # Add the text box to the plot
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f"Forecast: {model_name}")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save each model's individual graph
    plt.savefig(f"{results_folder}/{model_name}_forecast.png", dpi=300, bbox_inches='tight')
    plt.close() 

# --------------------------
# 4. Best Model Specific Plot
# --------------------------
best_model = metrics_df.index[0]
plt.figure(figsize=(12, 6))

series.plot(label="Actual", color="gray", alpha=0.5)
results[best_model]["forecast"].plot(label=f"BEST: {best_model}", color="red", lw=2)

# Add metrics for the best model
best_stats = (f"BEST MODEL: {best_model}\n"
              f"RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f}\n"
              f"MAPE: {metrics_df.loc[best_model, 'MAPE']:.2f}%")

plt.gca().text(0.02, 0.95, best_stats, transform=plt.gca().transAxes, 
               fontsize=12, fontweight='bold', verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.title("Winner: Best Model Performance")
plt.savefig(f"{results_folder}/best_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()