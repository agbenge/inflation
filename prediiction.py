import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import Lasso
import torch
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    ARIMA,
    Prophet,
    RandomForestModel,
    SKLearnModel,
    XGBModel,
    LightGBMModel,
    RNNModel,
    NBEATSModel
)


RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)

def process_case(file_path):
    
    
    print(f"\n--- Processing: {file_path.name} ---")
    
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime({'year': df['tyear'], 'month': df['tmonth'], 'day': 1})
    df.drop(columns=['tyear', 'tmonth'], inplace=True)
    df = df.sort_values('date').drop_duplicates(subset=['date'])

    target_col = 'allItemsYearOn'
    exclude_cols = ['date', target_col, 'tyear', 'tmonth', 'tday', 'index']
    feature_cols = [col for col in df.columns if col not in exclude_cols] 

    target_series = TimeSeries.from_dataframe(df, 'date', target_col, fill_missing_dates=True, freq='MS')
    
    cov_series = None
    if feature_cols:
        cov_series = TimeSeries.from_dataframe(df, 'date', feature_cols, fill_missing_dates=True, freq='MS')

    scaler_target = Scaler()
    target_scaled = scaler_target.fit_transform(target_series)
    
    cov_scaled = None
    if cov_series:
        scaler_cov = Scaler()
        cov_scaled = scaler_cov.fit_transform(cov_series)

    forecast_horizon = 6
    start_point = 0.7 

    models = {
        "LASSO":  SKLearnModel(
    lags=12,
    lags_past_covariates=12 if cov_scaled else None,
    model=Lasso(alpha=0.1)
),
        # "ARIMA": ARIMA(),
        # "Prophet": Prophet(),
        "RandomForest": RandomForestModel(lags=12, lags_past_covariates=12 if cov_scaled else None, output_chunk_length=1, n_estimators=100),
        # "XGBoost": XGBModel(lags=12, lags_past_covariates=12 if cov_scaled else None, output_chunk_length=1),
#   "LightGBM": LightGBMModel(lags=12, lags_past_covariates=12 if cov_scaled else None, output_chunk_length=1),
        # "LSTM": RNNModel(model="LSTM", input_chunk_length=12, output_chunk_length=forecast_horizon, n_epochs=50, random_state=42),
        # "NBEATS": NBEATSModel(input_chunk_length=12, output_chunk_length=forecast_horizon, n_epochs=50, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Logic for models that use Covariates vs Univariate
        kwargs = {}
        if name not in ["ARIMA", "Prophet"] and cov_scaled:
            kwargs['past_covariates'] = cov_scaled

        try:
            forecast_scaled = model.historical_forecasts(
                target_scaled,
                start=start_point,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=True,
                verbose=False,
                show_warnings=False,
                **kwargs
            )
            
            forecast = scaler_target.inverse_transform(forecast_scaled)
            actual = target_series.slice_intersect(forecast)

            results[name] = {
                "MAE": mae(actual, forecast),
                "RMSE": rmse(actual, forecast),
                "MAPE": mape(actual, forecast),
                "forecast": forecast
            }
        except Exception as e:
            print(f"      Error with {name}: {e}")

    # Metrics Summary
    metrics_df = pd.DataFrame({m: {k: v for k, v in results[m].items() if k != 'forecast'} for m in results}).T.sort_values("RMSE")
   

    # Plotting loop
    for model_name, data in results.items():
        plt.figure(figsize=(12, 6))
        target_series.plot(label="Actual", color="black")
        data["forecast"].plot(label=f"{model_name} Forecast")
        plt.title(f"{file_path.stem} - {model_name}")
        plt.savefig(RESULTS_FOLDER / f"{file_path.stem}_{model_name}.png")
        plt.close()

     
 
    # --------------------------
    # 5. Compare Metrics
    # --------------------------
    metrics_df = pd.DataFrame({
        model: { 
            "RMSE": results[model]["RMSE"],
            # "MAE": results[model]["MAE"],
            # "MAPE": results[model]["MAPE"], 
        }
        for model in results
    }).T.sort_values("RMSE")

    print("\nModel Comparison:")
    print(metrics_df)
    return metrics_df


       
