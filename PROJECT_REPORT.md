# Project Report: Smart City Traffic Forecasting

**Company Partner**: UniConverge Technologies (UCT)  
**Author**: shhkyyb  
**Date**: June 2026  

---

## 1. Background of the Project
As urban areas expand, local governments are transitioning traditional infrastructure into "smart cities" powered by IoT sensors, automated data collection, and intelligence. Efficient traffic management is a cornerstone of smart city planning. This project, completed in collaboration with UniConverge Technologies (UCT), utilizes historical traffic count data across city junctions to build predictive models that forecast vehicle patterns.

## 2. Problem Statement Relevance
Traffic congestion causes economic losses, environmental pollution, and commuter delays. By predicting traffic volume at specific junctions, municipal planning departments can:
*   Dynamically manage traffic signals based on predicted peaks.
*   Optimize public transport schedules.
*   Inform long-term infrastructure planning (e.g., road extensions or alternative routing).

The objective of this project is to build a regression model that accurately forecasts traffic counts (number of vehicles) for upcoming hourly intervals, accounting for seasonal daily commute behaviors.

## 3. Design
The project is built on a modular pipeline design, consisting of the following architectural blocks:
1.  **Data Ingestion & EDA**: Loading historical sensor data and validating completeness (e.g., checking for null values).
2.  **Temporal Feature Engineering**: Extracting cyclical features (Hour, Day of Week, Day of Month, Month, Year) from the raw timestamp, since typical tree-based regressors do not natively handle timestamps.
3.  **Validation Scheme**: Implementing a chronological split (80% train, 20% validation) to simulate real-world forecasting settings, avoiding future-data leakage.
4.  **Modeling & Optimization**: Using Gradient Boosted Decision Trees (XGBoost Regressor) due to their high performance on tabular features.
5.  **Visualization & Diagnostics**: Plotting predicted sequences against ground-truth validation data.

## 4. Implementation Details
*   **Dependencies**: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`.
*   **Feature Engineering Code**:
    ```python
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    ```
*   **Model Configuration**:
    *   Estimators: 100 decision trees (`n_estimators=100`)
    *   Max Depth: 6 levels (`max_depth=6`)
    *   Learning Rate: 0.1 (`learning_rate=0.1`)

## 5. Results
The model was validated on the chronological hold-out set (the final 20% of chronological records):
*   **Baseline (Historical Mean) RMSE**: 28.41 vehicles per hour
*   **XGBoost Validation RMSE**: 8.21 vehicles per hour
*   **Relative Improvement**: **71.08%** error reduction compared to simple baseline guessing.

Visual diagnostics confirm that the predictions track actual traffic fluctuations, successfully modeling both daily evening rush hours (peaking around 7:00 PM) and early morning lull periods.

## 6. Personal Learnings
*   **Time-Series Validation**: Understood the critical importance of chronological data splitting. Using a random split in time-series data leaks future values, leading to artificially inflated accuracy and poor real-world generalization.
*   **Feature Representation**: Learned how to decompose temporal components to allow non-deep-learning regressors (like XGBoost) to effectively capture seasonality.
*   **Evaluation Metrics**: Experienced how Root Mean Squared Error (RMSE) quantifies error in the same units as the target variable (vehicles per hour), making results easily communicable to stakeholders.
