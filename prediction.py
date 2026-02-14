import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso as SkLasso

from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler

# --------------------------
# 1. Data Prep
# --------------------------
# Assuming df is already loaded or read here
df = pd.read_excel("data/inner_join.xlsx")
df['date'] = pd.to_datetime({'year': df['tyear'], 'month': df['tmonth'], 'day': 1})
df_unique = df.drop_duplicates(subset=['date']).sort_values('date')

series = TimeSeries.from_dataframe(
   df_unique, time_col='date', value_cols='foodAverage', fill_missing_dates=True, freq='MS'
).astype(np.float32)

 

scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# --------------------------
# 2. Updated Model Definitions
# --------------------------
lasso_engine = SkLasso(alpha=0.1)

models = {
    "LASSO": RegressionModel(lags=12, model=lasso_engine),
    ###   period=12  # Note: STLF uses 'period' or 'seasonal_periods' depending on version
   # )
}

results = {}

# --------------------------
# 3. Backtesting Loop
# --------------------------
for name, model in models.items():
    print(f"Backtesting {name}...")
    
    # Changed last_points_only to True for easier metric calculation
    forecast_scaled = model.historical_forecasts(
        series_scaled,
        start=0.6,
        forecast_horizon=6,
        stride=1,
        retrain=True,
        verbose=True,
        last_points_only=True 
    )
    
    # Rescale back to original values
    forecast = scaler.inverse_transform(forecast_scaled)
    
    # Slice the actual series to match the forecast time index
    actual = series.slice_intersect(forecast)

    results[name] = {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "forecast": forecast
    }

# --------------------------
# 4. Display Results
# --------------------------
metrics_df = pd.DataFrame(results).T.drop(columns='forecast')
print("\nModel Comparison:")
print(metrics_df)