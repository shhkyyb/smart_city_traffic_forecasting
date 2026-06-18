import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

def prepare_features(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    return df

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_val, y_val, y_train_mean):
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    
    baseline_preds = np.full_like(y_val, y_train_mean)
    baseline_rmse = root_mean_squared_error(y_val, baseline_preds)
    
    improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Improvement: {improvement:.2f}%")
    return preds

def plot_predictions(val_df, predictions, output_path):
    val_df = val_df.copy()
    val_df['Predictions'] = predictions
    sample = val_df[val_df['Junction'] == 1].head(120)
    
    plt.figure(figsize=(12, 6))
    plt.plot(sample['DateTime'], sample['Vehicles'], label="Actual", color="#1f77b4")
    plt.plot(sample['DateTime'], sample['Predictions'], label="Predicted (XGBoost)", color="#ff7f0e", linestyle="--")
    plt.title("Model Predictions vs Ground Truth (Junction 1)")
    plt.xlabel("Date")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    df = pd.read_csv(data_path)
    df = prepare_features(df)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    features = ['Junction', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Year']
    target = 'Vehicles'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    
    model = train_model(X_train, y_train)
    preds = evaluate(model, X_val, y_val, y_train.mean())
    plot_predictions(val_df, preds, "actual_vs_predicted_traffic.png")
