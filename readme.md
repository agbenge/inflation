source .venv/bin/activate

 
pip install 'u8darts[all]' pandas scikit-learn statsmodels prophet pmdarima openpyxl



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
