import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_weekly_traffic(df, output_path):
    df_j1 = df[df['Junction'] == 1].sort_values('DateTime')
    weekly = df_j1.head(168)
    
    plt.figure(figsize=(12, 6))
    plt.plot(weekly['DateTime'], weekly['Vehicles'], color='#1f77b4', linewidth=1.5)
    plt.title("Junction 1 Traffic - Hourly Counts (1 Week Sample)")
    plt.xlabel("Date & Time")
    plt.ylabel("Vehicles")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_hourly_averages(df, output_path):
    df_j1 = df[df['Junction'] == 1].copy()
    df_j1['Hour'] = df_j1['DateTime'].dt.hour
    hourly_avg = df_j1.groupby('Hour')['Vehicles'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.bar(hourly_avg.index, hourly_avg.values, color='#2ca02c', alpha=0.8)
    plt.title("Average Hourly Traffic (Junction 1)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Vehicle Count")
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    plot_weekly_traffic(df, "traffic_1week_junction1.png")
    plot_hourly_averages(df, "traffic_hourly_average.png")
