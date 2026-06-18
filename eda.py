import pandas as pd
import os

def run_eda():
    # Define the path to our training dataset
    dataset_path = os.path.join("data", "smart-city-traffic-patterns", "train_aWnotuB.csv")
    
    # 1. Load the data
    print("--- Loading Dataset ---")
    df = pd.read_csv(dataset_path)
    
    # 2. View basic info (number of rows, columns, data types)
    print("\n--- Basic Information ---")
    print(df.info())
    
    # 3. View the first few rows
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    # 4. Check for missing values (crucial step in data prep!)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # 5. Check what junctions we have and how many records per junction
    print("\n--- Records per Junction ---")
    print(df['Junction'].value_counts())
    
    # 6. Basic statistics of vehicles
    print("\n--- Traffic Volume Statistics ---")
    print(df['Vehicles'].describe())

if __name__ == "__main__":
    run_eda()
