import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch

USERNAME = "muskan.sharma" 

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def autoregressive_forecast_dl(model, history, horizon, window_size, device='cpu'):
    """
    Generates multi-step forecast by feeding predictions back into input.
    """
    model.eval()
    current_seq = torch.FloatTensor(history[-window_size:]).unsqueeze(0).unsqueeze(-1).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(horizon):
            pred = model(current_seq)
            pred_val = pred.item()
            predictions.append(pred_val)
            
            # Update sequence: remove oldest, add new prediction
            new_input = torch.FloatTensor([[[pred_val]]]).to(device)
            current_seq = torch.cat((current_seq[:, 1:, :], new_input), dim=1)
            
    return predictions

def plot_forecasts(train, test, forecasts, labels, title):
    plt.figure(figsize=(12, 6))
    
    # Plot training tail
    plt.plot(train.index[-50:], train.values[-50:], label='Train (Last 50)', color='gray', alpha=0.5)
    
    # Plot Test Ground Truth
    plt.plot(test.index, test.values, label='Actual Test', color='black', linewidth=2)
    
    # Plot Forecasts
    for forecast, label in zip(forecasts, labels):
        forecast_index = test.index[:len(forecast)]
        plt.plot(forecast_index, forecast, label=label, linestyle='--')

    plt.title(f"{title}")
    plt.xlabel("Date")
    plt.ylabel("Stars (Incremental)")
    plt.legend()
    plt.text(0.95, 0.95, USERNAME, ha='right', va='top', 
             transform=plt.gca().transAxes, fontsize=10, color='gray', alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_error_vs_horizon(metrics_dict):
    plt.figure(figsize=(10, 6))
    
    for model_name, data in metrics_dict.items():
        horizons = sorted(data.keys())
        errors = [data[h] for h in horizons]
        plt.plot(horizons, errors, marker='o', label=model_name)
        
    plt.title("Prediction Error vs Forecast Horizon")
    plt.xlabel("Horizon (Days)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.text(0.95, 0.95, USERNAME, ha='right', va='top', 
             transform=plt.gca().transAxes, fontsize=10, color='gray', alpha=0.7)
    
    plt.grid(True)
    plt.show()