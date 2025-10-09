import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import time
import sys

# Add the project root to the Python path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import build_features function
from src.features.build_features import build_features

def create_preprocessor(X_train):
    """Creates and fits the ColumnTransformer for preprocessing."""
    numerical_features = ['high_risk_curvature_interaction', 'Visibility_Score']
    boolean_features = ['public_road', 'holiday', 'is_unsigned_urban_road', 'is_accident_hotspot', 'is_urban_evening_clear_weather']
    categorical_features = ['road_type', 'weather', 'time_of_day', 'school_season_x_time_of_day', 'speed_zone', 'lane_category', 'accident_hotspot_x_weather']

    for col in boolean_features:
        X_train[col] = X_train[col].astype(int) # Apply to X_train for fitting

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features + boolean_features)
        ],
        remainder='passthrough'
    )
    return preprocessor.fit(X_train) # Fit preprocessor once on training data

if __name__ == "__main__":
    # --- Paths ---
    raw_train_path = os.path.join('data', 'raw', 'train.csv')
    raw_test_path = os.path.join('data', 'raw', 'test.csv')
    submission_path = os.path.join('data', 'processed', 'submission.csv')
    final_log_file = os.path.join('logs', 'final_submission_log.txt') # Revert log file name

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_log_file), exist_ok=True)

    print("--- Preparing for Final Submission ---")

    # --- Load Data ---
    print(f"Loading raw training data from {raw_train_path}...")
    train_df = pd.read_csv(raw_train_path)
    print(f"Loading raw test data from {raw_test_path}...")
    test_df = pd.read_csv(raw_test_path)

    # Store test IDs for submission
    test_ids = test_df['id']

    # --- Apply Feature Engineering to Train and Test Data ---
    print("Applying feature engineering to training data...")
    train_df_fe = build_features(train_df.copy()) # Use a copy to avoid modifying raw_df
    print("Applying feature engineering to test data...")
    test_df_fe = build_features(test_df.copy()) # Use a copy to avoid modifying raw_df

    # Ensure consistent columns after FE (important for OHE)
    # Drop 'id' from training data as it's not a feature
    train_df_fe = train_df_fe.drop('id', axis=1, errors='ignore')
    test_df_fe = test_df_fe.drop('id', axis=1, errors='ignore') # Drop id from test features too

    # Separate target variable from training data
    X = train_df_fe.drop('accident_risk', axis=1)
    y = train_df_fe['accident_risk']

    # --- Data Split (for preprocessor fitting and base model training) ---
    # Use the full training data for preprocessor fitting and base model training
    X_train_full = X
    y_train_full = y # Use original target for training
    X_test_submission = test_df_fe # This is the actual test set for submission

    # --- Preprocessor ---
    preprocessor = create_preprocessor(X_train_full)
    print("Preprocessor fitted on full training data.")

    # --- Best XGBoost Base Model Parameters (from tuning) ---
    best_xgboost_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 550,
        'learning_rate': 0.012370036011546435,
        'max_depth': 8,
        'subsample': 0.892769159931097,
        'colsample_bytree': 0.763236692630663,
        'gamma': 0.0009958537492133792,
        'reg_alpha': 5.595257681864575e-07,
        'reg_lambda': 2.3721727576947476e-07,
        'random_state': 42,
        'n_jobs': -1
    }

    # --- Best Residual XGBoost Model Parameters (from tuning) ---
    best_residual_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 250,
        'learning_rate': 0.04137912638367983,
        'max_depth': 3,
        'subsample': 0.999185234245023,
        'colsample_bytree': 0.7901547283056515,
        'gamma': 1.0845170900304183e-08,
        'reg_alpha': 0.0016587705109178757,
        'reg_lambda': 0.00012874468299642057,
        'random_state': 42,
        'n_jobs': -1
    }

    # --- Step 1: Train the Base Model (XGBoost) on FULL Training Data ---
    print("\n--- Training Base XGBoost Model on full training data ---")
    base_model = xgb.XGBRegressor(**best_xgboost_params)
    base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', base_model)])
    start_time_base = time.time()
    base_pipeline.fit(X_train_full, y_train_full) # Train on original target
    y_pred_base_train_full = base_pipeline.predict(X_train_full) # Predictions on full training data
    end_time_base = time.time()
    training_time_base = end_time_base - start_time_base
    print(f"Base XGBoost Training Time: {training_time_base:.2f} seconds")

    # --- Step 2: Calculate Residuals from Base Model on FULL Training Data ---
    residuals_train_full = y_train_full - y_pred_base_train_full
    print(f"Calculated residuals for full training data. Mean: {np.mean(residuals_train_full):.4f}, Std: {np.std(residuals_train_full):.4f}")

    # --- Step 3: Train Residual Model on FULL Training Data ---
    print("\n--- Training Residual XGBoost Model on full training data ---")
    residual_model = xgb.XGBRegressor(**best_residual_params)
    residual_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', residual_model)])
    start_time_residual = time.time()
    residual_pipeline.fit(X_train_full, residuals_train_full) # Train on original X_train, target is residuals
    end_time_residual = time.time()
    training_time_residual = end_time_residual - start_time_residual
    print(f"Residual XGBoost Training Time: {training_time_residual:.2f} seconds")

    # --- Make Predictions on Test Data ---
    print("\n--- Making predictions on test data ---")
    y_pred_base_test = base_pipeline.predict(X_test_submission)
    residuals_pred_test = residual_pipeline.predict(X_test_submission)
    final_y_pred_test = y_pred_base_test + residuals_pred_test

    # Ensure predictions are within [0, 1] range
    final_y_pred_test = np.clip(final_y_pred_test, 0, 1)

    # --- Create Submission File ---
    submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': final_y_pred_test})
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at {submission_path}")

    # --- Log Submission Details ---
    total_training_time = training_time_base + training_time_residual
    with open(final_log_file, 'a') as f:
        f.write(f"\n--- Final Submission Details ---\n")
        f.write(f"Model: Stacking (XGBoost Base + Tuned XGBoost Residual)\n")
        f.write(f"Base Model Hyperparameters: {best_xgboost_params}\n")
        f.write(f"Residual Model Hyperparameters: {best_residual_params}\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
        f.write(f"Submission File: {submission_path}\n")
        f.write(f"Note: RMSE on validation set for this model was {0.076668:.4f}\n") # Hardcode best RMSE for logging
    print(f"Final submission details logged to {final_log_file}")
