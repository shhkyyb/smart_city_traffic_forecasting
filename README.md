# Smart City Traffic Forecasting

This repository contains a machine learning pipeline to forecast traffic volume across city junctions. The project focuses on converting raw time-series data into a supervised regression framework using XGBoost.

## Project Architecture
The project is structured into modular scripts:
*   `eda.py`: Preliminary dataset inspection and summary statistics.
*   `visualize.py`: Extracts temporal seasonality and saves exploratory plots.
*   `train_predict.py`: Handles feature engineering, chronological data splitting, model training, evaluation, and prediction visualization.

## Feature Engineering
Temporal components were engineered from the raw `DateTime` field:
*   `Hour` (0-23) to capture intra-day cycles.
*   `DayOfWeek` (0-6) to capture weekly commute behavior.
*   `DayOfMonth` and `Month` to capture monthly seasonality.

## Modeling & Evaluation
To evaluate model performance under realistic forecasting constraints, we split the dataset chronologically (80% training, 20% validation).

An XGBoost Regressor was trained on the engineered features:
*   **Validation RMSE**: 8.21 vehicles
*   **Baseline (Historical Mean) RMSE**: 28.41 vehicles
*   **Performance Gain**: 71.08% improvement over the baseline guess.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Exploratory Data Analysis:
   ```bash
   python eda.py
   ```
3. Generate traffic visualization plots:
   ```bash
   python visualize.py
   ```
4. Train the model and plot forecasts:
   ```bash
   python train_predict.py
   ```
