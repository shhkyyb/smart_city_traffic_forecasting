import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import os

def run_training():
    print("--- 1. Loading and Preparing Data ---")
    dataset_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    df = pd.read_csv(dataset_path)
    
    # Convert DateTime to a real date object
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Sort data by time (crucial for time-series chronological splitting)
    df = df.sort_values('DateTime')
    
    print("\n--- 2. Feature Engineering ---")
    # We extract numerical "clues" (features) from the DateTime column
    # because XGBoost cannot read date strings directly.
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # 0 is Monday, 6 is Sunday
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    
    print("Created features: Hour, DayOfWeek, DayOfMonth, Month, Year")
    
    # 3. Train/Validation Chronological Split
    # In time-series, we cannot split randomly. We must train on the past 
    # and validate on the future. Let's use the last 20% of the data for testing.
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    
    # Define our inputs (Features, X) and outputs (Target, y)
    features = ['Junction', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Year']
    target = 'Vehicles'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    
    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    
    print("\n--- 3. Training the XGBoost Model ---")
    # XGBoost is a powerful Decision-Tree based algorithm.
    # It builds multiple trees step-by-step to correct previous errors.
    model = xgb.XGBRegressor(
        n_estimators=100,      # How many decision trees to build
        max_depth=6,           # How deep/complex each tree can be
        learning_rate=0.1,     # How fast the model learns
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    print("\n--- 4. Evaluation on Validation Set ---")
    # Let's make predictions on our future/validation set
    predictions = model.predict(X_val)
    
    # Calculate RMSE (Root Mean Squared Error)
    # RMSE represents the average deviation (error) between predictions and actual values.
    # E.g., if RMSE is 5, it means our predictions are off by ~5 cars on average.
    rmse = root_mean_squared_error(y_val, predictions)
    print(f"Validation RMSE: {rmse:.2f} vehicles")
    
    # Let's check a baseline metric: what if we just guessed the average traffic?
    baseline_predictions = np.full(shape=y_val.shape, fill_value=y_train.mean())
    baseline_rmse = root_mean_squared_error(y_val, baseline_predictions)
    print(f"Baseline (Average Guess) RMSE: {baseline_rmse:.2f} vehicles")
    print(f"Improvement: {((baseline_rmse - rmse) / baseline_rmse) * 100:.2f}% improvement over simple guessing!")
    
    # Save a comparison plot of Actual vs Predicted for a sample window (first 5 days of validation)
    plt.figure(figsize=(12, 6))
    
    # Filter validation set to first 120 hours of Junction 1
    val_j1 = val_df[val_df['Junction'] == 1].head(120)
    indices = val_j1.index
    predictions_j1 = predictions[val_df['Junction'] == 1][:120]
    
    plt.plot(val_j1['DateTime'], val_j1['Vehicles'], label="Actual Traffic", color="blue", linewidth=2)
    plt.plot(val_j1['DateTime'], predictions_j1, label="Predicted Traffic (XGBoost)", color="orange", linestyle="--", linewidth=2)
    
    plt.title("Actual vs Predicted Traffic (Junction 1 - 5 Days Forecast)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date & Time", fontsize=12)
    plt.ylabel("Number of Vehicles", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_image = "actual_vs_predicted_traffic.png"
    plt.savefig(output_image)
    print(f"\n--- Plot successfully saved as: {output_image} ---")

if __name__ == '__main__':
    run_training()
