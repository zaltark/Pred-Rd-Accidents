import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

def train_and_evaluate_model(model_name, model, X_train, y_train, X_val, y_val, log_file):
    """Trains and evaluates a given model, logging the results."""
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    end_time = time.time()
    training_time = end_time - start_time

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    results = f"Model: {model_name}\n" \
              f"MAE: {mae:.4f}\n" \
              f"RMSE: {rmse:.4f}\n" \
              f"R2: {r2:.4f}\n" \
              f"Training Time: {training_time:.2f} seconds\n" \
              f"-----------------------------------\n"

    print(results)
    with open(log_file, 'a') as f:
        f.write(results)
    return mae, rmse, r2

if __name__ == "__main__":
    processed_data_path = os.path.join('data', 'processed', 'processed_train.csv')
    log_file = os.path.join('logs', 'model_training_log.txt')

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Clear previous log content
    with open(log_file, 'w') as f:
        f.write("--- Model Training Log ---\n")

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)

    # Separate target variable
    X = df.drop('accident_risk', axis=1)
    y = df['accident_risk']

    # Identify feature types
    numerical_features = ['high_risk_curvature_interaction', 'Visibility_Score', 'low_speed_x_Visibility_Score']
    boolean_features = ['public_road', 'is_unsigned_urban_road', 'is_accident_hotspot', 'is_urban_evening_clear_weather', 'holiday_x_time_of_day_x_bright', 'low_speed_x_road_type_urban', 'low_speed_x_lane_category_two_lanes', 'is_moderate_visibility_overprediction_zone']
    categorical_features = ['road_type', 'weather', 'time_of_day', 'school_season_x_time_of_day', 'speed_zone', 'lane_category', 'accident_hotspot_x_weather']

    # Ensure boolean features are treated as numerical (0/1)
    for col in boolean_features:
        X[col] = X[col].astype(int)

    # Preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features + boolean_features) # OHE booleans too for consistency
        ],
        remainder='passthrough' # Keep other columns (if any, though there shouldn't be)
    )

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training (X_train: {X_train.shape}, y_train: {y_train.shape}) and validation (X_val: {X_val.shape}, y_val: {y_val.shape}) sets.")

    # Define models
    models = {
        "XGBoostRegressor": xgb.XGBRegressor(random_state=42),
        "LightGBMRegressor": lgb.LGBMRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "CatBoostRegressor": CatBoostRegressor(random_state=42, verbose=0), # verbose=0 to suppress extensive output
        "RidgeRegressor": Ridge(random_state=42)
    }

    # Train and evaluate each model
    for name, model in models.items():
        # Create a pipeline with preprocessing and the model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', model)])
        train_and_evaluate_model(name, full_pipeline, X_train, y_train, X_val, y_val, log_file)

    print("\nAll baseline models trained and evaluated. Check logs/model_training_log.txt for details.")
