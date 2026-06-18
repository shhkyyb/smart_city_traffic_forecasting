import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    dataset_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    df = pd.read_csv(dataset_path)
    
    # 1. Convert 'DateTime' column from string to a real Python datetime object
    # This allows us to easily extract the day, hour, month, etc.
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 2. Let's filter our data to look at just Junction 1
    # This keeps our first graph simple and easy to understand
    df_j1 = df[df['Junction'] == 1].copy()
    
    # 3. Sort by time to make sure it's in order
    df_j1 = df_j1.sort_values('DateTime')
    
    # 4. Let's look at the first 7 days (7 days * 24 hours/day = 168 hours)
    weekly_data = df_j1.head(168)
    
    # 5. Create a plot of traffic volume over time
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['DateTime'], weekly_data['Vehicles'], marker='o', color='#1f77b4', linewidth=2)
    
    # Label our axes and title
    plt.title("Junction 1 - Traffic Patterns Over 1 Week (Hourly Count)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date & Time", fontsize=12)
    plt.ylabel("Number of Vehicles", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and save the image in the current directory
    plt.tight_layout()
    output_image = "traffic_1week_junction1.png"
    plt.savefig(output_image)
    print(f"--- Plot successfully saved as: {output_image} ---")
    
    # Let's write some code to see average traffic by hour of the day
    df_j1['Hour'] = df_j1['DateTime'].dt.hour
    hourly_avg = df_j1.groupby('Hour')['Vehicles'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.bar(hourly_avg.index, hourly_avg.values, color='#2ca02c', alpha=0.8)
    plt.title("Average Traffic Volume by Hour of the Day (Junction 1)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Hour of the Day (0-23)", fontsize=12)
    plt.ylabel("Average Vehicle Count", fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    output_bar = "traffic_hourly_average.png"
    plt.savefig(output_bar)
    print(f"--- Plot successfully saved as: {output_bar} ---")

if __name__ == "__main__":
    generate_plots()
