# Economic-Time-Series-Forecasting-on-AWS-SageMakerwith ARIMA, Prophet, and LSTM

This project forecasts U.S. Consumer Price Index (CPI) values using classical statistical models, trend-based forecasting, and deep learning. The goal is to compare model performance under a proper time-series validation framework and demonstrate deployment readiness using AWS SageMaker.

## Data
- Source: Federal Reserve Economic Data (FRED)
- Series: CPIAUCSL
- Frequency: Monthly

## Methods
- ARIMA (baseline statistical model)
- Prophet (trend + seasonality model)
- LSTM (univariate deep learning model)

A 24-month holdout period is used for evaluation.

## Results
Prophet achieved the lowest MAE and RMSE on the holdout set, outperforming both ARIMA and the univariate LSTM. This aligns with the trend-dominated structure of CPI data.

## Deployment Readiness
The selected Prophet model is serialized and packaged into a SageMaker-compatible `model.tar.gz`.  
A custom inference handler and optional deployment script are provided. Deployment is intentionally excluded from the notebook to avoid unnecessary cloud costs.
