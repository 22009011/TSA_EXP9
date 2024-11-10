# EX.NO.09        A project on Time Series Analysis on EV Data Using the ARIMA Model 

### Date: 
### Developed by: THANJIYAPPAN K
### Register Number: 212222240108

### AIM:
To create a project on time series analysis of EV data using the ARIMA model in Python and compare it with other models.

### ALGORITHM:
1. Explore the Dataset of EV Data 
   - Load the EV dataset and perform initial exploration, focusing on the `year` and `value` columns. Plot the time series to visualize trends.

2. Check for Stationarity of the Time Series 
   - Plot the time series data and use the following methods to assess stationarity:
     - Time Series Plot: Visualize the data for seasonality or trends.
     - ACF and PACF Plots**: Inspect autocorrelation and partial autocorrelation plots to understand the lag structure.
     - ADF Test**: Apply the Augmented Dickey-Fuller (ADF) test to check if the series is stationary.

3. Transform to Stationary (if needed)  
   - If the series is not stationary (as indicated by the ADF test), apply differencing to remove trends and make the series stationary.

4. Determine ARIMA Model Parameters (p, d, q) 
   - Use insights from the ACF and PACF plots to select the AR and MA terms (`p` and `q` values).
   - Choose `d` based on the differencing applied to achieve stationarity.

5. Fit the ARIMA Model 
   - Fit an ARIMA model with the selected `(p, d, q)` parameters on the historical EV data values.

6. Make Time Series Predictions  
   - Forecast future values for a specified time period (e.g., 12 years) using the fitted ARIMA model.

7. Auto-Fit the ARIMA Model (if applicable) 
   - Use auto-fitting methods (such as grid search or auto_arima from `pmdarima`) to automatically determine the best parameters for the model if needed.

8. Evaluate Model Predictions  
   - Compare the predicted values with actual values using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to assess the model's accuracy.
   - Plot the historical data and forecasted values to visualize the model's performance.

### PROGRAM:
```PY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
data = pd.read_csv('ev.csv')

# Convert 'year' column to datetime and set it as the index
data['year'] = pd.to_datetime(data['year'], format='%Y')
data.set_index('year', inplace=True)

# Extract the 'value' column for analysis
series = data['value'].dropna()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(series)
plt.title("EV Data Over Time")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# Augmented Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # Returns True if series is stationary

# Check stationarity and difference the series if needed
is_stationary = adf_test(series)
if not is_stationary:
    series_diff = series.diff().dropna()
    plt.figure(figsize=(10, 5))
    plt.plot(series_diff)
    plt.title("Differenced EV Data")
    plt.show()
else:
    series_diff = series

# Plot ACF and PACF
plot_acf(series_diff, lags=20)
plt.title("Autocorrelation (ACF) of Differenced Series")
plt.show()

plot_pacf(series_diff, lags=20)
plt.title("Partial Autocorrelation (PACF) of Differenced Series")
plt.show()

# ARIMA model parameters (p, d, q) - adjust based on ACF/PACF plots
p, d, q = 1, 1, 1  # These are example values; modify based on ACF/PACF insights

# Fit the ARIMA model
model = ARIMA(series, order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecasting
forecast_steps = 12  # Number of periods to forecast (e.g., years if yearly data)
forecast = fitted_model.forecast(steps=forecast_steps)

# Set up the forecast index
last_date = series.index[-1]
forecast_index = pd.date_range(last_date, periods=forecast_steps + 1, freq='Y')[1:]  # Shift to match forecast start

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(series, label="Historical Data")
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("EV Data Forecast")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

```

### OUTPUT:
EV SALES over time:
![image](https://github.com/user-attachments/assets/116c9ee5-2fd6-407f-aade-7965d9043ce4)


Autocorrelation:
![image](https://github.com/user-attachments/assets/ea718b7d-bbd4-420d-8731-c78e73b4f706)


Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/396d999a-b48b-4265-9cbb-a0c60dc5c62e)


Model summary:
![image](https://github.com/user-attachments/assets/a5c81acc-9755-4f75-883f-4b933c941bb0)


Power Consumption Forecast:
![image](https://github.com/user-attachments/assets/36189558-28e4-4d09-99c4-6be4c21c4e77)





### RESULT:
Thus the project on Time series analysis on Ev Sales based on the ARIMA model using python is executed successfully.
