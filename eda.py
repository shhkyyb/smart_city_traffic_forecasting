import os
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def analyze_dataset(df):
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Junction Counts ---")
    print(df['Junction'].value_counts())
    print("\n--- Traffic Summary ---")
    print(df['Vehicles'].describe())

if __name__ == "__main__":
    data_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    df = load_data(data_path)
    analyze_dataset(df)
