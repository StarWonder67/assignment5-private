from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class ClassicalARMA:
    def __init__(self, order=(2, 0, 2)):
        # Order (p,d,q)
        # p: window size of AR model (how many past steps to consider)
        # d: degree of differencing (0 for stationary data)
        # q: window size of MA model (how many past error terms to consider)
        self.order = order
        self.model = None
        self.fit_result = None

    def fit(self, train_data):
        # ARMA assumes stationarity, so we use the incremental data
        self.model = ARIMA(train_data, order=self.order)
        self.fit_result = self.model.fit()
        print(self.fit_result.summary())

    def predict_next_step(self, history):
        if self.fit_result is None:
            raise ValueError("Model must be fit before predicting.")
        
        new_results = self.fit_result.apply(history)
        forecast = new_results.forecast(steps=1)
        
        # Handle output format (statsmodels returns a Series or array)
        if isinstance(forecast, pd.Series):
            return forecast.iloc[0]
        elif isinstance(forecast, np.ndarray):
            return forecast[0]
        else:
            return forecast

    def forecast(self, steps):
        return self.fit_result.forecast(steps=steps)