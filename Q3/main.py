import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from prep_stars import StarPreprocessor
from split_repos import chronological_split
from classical import ClassicalARMA
from dl_models import SimpleRNN, SimpleCNN
from train_models import train_dl_model
from evaluate import calculate_metrics, autoregressive_forecast_dl, plot_forecasts, plot_error_vs_horizon

# --- CONFIGURATION ---
CSV_PATH = "Q3/stars_data.csv"
REPO_NAMES = ["facebook/react", "pallets/flask"]  # List of repos to analyze
WINDOW_SIZE = 30
HORIZONS = [1, 3, 7, 14]
EPOCHS = 50  # Reduced for quick testing, increase for better results

# Initialize Preprocessor once
print("--- Loading and Cleaning Data ---")
preprocessor = StarPreprocessor(CSV_PATH)
preprocessor.load_data()

for repo_name in REPO_NAMES:
    print(f"\n{'='*40}")
    print(f"Processing Repository: {repo_name}")
    print(f"{'='*40}")

    # 1. DATA PREPARATION
    try:
        daily, incremental = preprocessor.get_repo_series(repo_name)
    except ValueError as e:
        print(e)
        continue

    train_raw, test_raw = chronological_split(incremental, split_ratio=0.8)
    
    train_scaled, test_scaled, min_val, max_val = preprocessor.normalize_series(train_raw.values, test_raw.values)
    
    print(f"Train Size: {len(train_scaled)} days")
    print(f"Test Size:  {len(test_scaled)} days")

    # 2. MODEL TRAINING

    # --- Classical ARMA ---
    print("\n> Training ARMA...")
    arma_model = ClassicalARMA(order=(2,0,2))
    # Suppress massive summary printout for cleaner logs
    try:
        arma_model.fit(train_scaled)
    except Exception as e:
        print(f"ARMA fitting failed: {e}")

    # --- RNN ---
    print("> Training RNN...")
    rnn_model = SimpleRNN(input_dim=1, hidden_dim=32, output_dim=1)
    # Re-initialize model for each repo to avoid carrying over weights
    rnn_model = train_dl_model(rnn_model, train_scaled, window_size=WINDOW_SIZE, epochs=EPOCHS)

    # --- CNN ---
    print("> Training CNN...")
    cnn_model = SimpleCNN(input_dim=1, output_dim=1, seq_len=WINDOW_SIZE)
    cnn_model = train_dl_model(cnn_model, train_scaled, window_size=WINDOW_SIZE, epochs=EPOCHS)

    # 3. EVALUATION (Multi-step Horizon Analysis)
    results = {'ARMA': {}, 'RNN': {}, 'CNN': {}}
    
    # Seed history for DL models (last window of training data)
    history_seed = train_scaled[-WINDOW_SIZE:]

    print("\n> Evaluating multi step Forecast Horizons...")
    for h in HORIZONS:
        # Get ground truth for this horizon
        if len(test_scaled) < h:
            print(f"Warning: Not enough test data for horizon {h}")
            break
            
        y_true_h = test_scaled[:h]

        # ARMA Evaluation
        try:
            arma_pred = arma_model.forecast(steps=h)
            _, rmse_arma = calculate_metrics(y_true_h, arma_pred)
            results['ARMA'][h] = rmse_arma
        except:
            results['ARMA'][h] = np.nan

        # RNN Evaluation
        rnn_preds = autoregressive_forecast_dl(rnn_model, history_seed, h, WINDOW_SIZE)
        _, rmse_rnn = calculate_metrics(y_true_h, rnn_preds)
        results['RNN'][h] = rmse_rnn

        # CNN Evaluation
        cnn_preds = autoregressive_forecast_dl(cnn_model, history_seed, h, WINDOW_SIZE)
        _, rmse_cnn = calculate_metrics(y_true_h, cnn_preds)
        results['CNN'][h] = rmse_cnn
        
        print(f"  Horizon {h}: ARMA={results['ARMA'][h]:.4f}, RNN={results['RNN'][h]:.4f}, CNN={results['CNN'][h]:.4f}")

    # 4. VISUALIZATION
    
    # Plot 1: RMSE vs Horizon
    plot_error_vs_horizon(results)
    
    # Plot 2: Visual Trajectory (Long Horizon Forecast)
    long_horizon = 30
    if len(test_scaled) >= long_horizon:
        # Generate Forecasts
        rnn_long_pred = autoregressive_forecast_dl(rnn_model, history_seed, long_horizon, WINDOW_SIZE)
        cnn_long_pred = autoregressive_forecast_dl(cnn_model, history_seed, long_horizon, WINDOW_SIZE)
        
        # Inverse Transform to Real Star Counts
        rnn_real = preprocessor.inverse_transform(np.array(rnn_long_pred), min_val, max_val)
        cnn_real = preprocessor.inverse_transform(np.array(cnn_long_pred), min_val, max_val)
        
        # Prepare Ground Truth
        test_real_values = preprocessor.inverse_transform(test_scaled[:long_horizon], min_val, max_val)
        test_dates = test_raw.index[:long_horizon]
        test_series = pd.Series(test_real_values, index=test_dates)
        
        # Plot
        plot_forecasts(
            train=train_raw,
            test=test_series,
            forecasts=[rnn_real, cnn_real],
            labels=['RNN Forecast', 'CNN Forecast'],
            title=f"{repo_name}: 30-Day Forecast (Incremental)"
        )
    else:
        print("Skipping trajectory plot (insufficient test data).")

print("\nDone! All repositories processed.")