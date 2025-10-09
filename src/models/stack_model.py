import pandas as pd
import numpy as np
import os
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Global variables for preprocessor and data split
preprocessor = None
X_train_global, X_val_global, y_train_global, y_val_global = None, None, None, None
y_pred_base_val_global = None # Predictions of base model on validation set
residuals_train_global = None # Residuals of base model on training set

def create_preprocessor(X):
    """Creates and fits the ColumnTransformer for preprocessing."""
    numerical_features = ['high_risk_curvature_interaction', 'Visibility_Score']
    boolean_features = ['public_road', 'holiday', 'is_unsigned_urban_road', 'is_accident_hotspot', 'is_urban_evening_clear_weather']
    categorical_features = ['road_type', 'weather', 'time_of_day', 'school_season_x_time_of_day', 'speed_zone', 'lane_category', 'accident_hotspot_x_weather']

    for col in boolean_features:
        X[col] = X[col].astype(int)

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features + boolean_features)
        ],
        remainder='passthrough'
    )
    return preprocessor.fit(X)

def objective(trial):
    """Optuna objective function for tuning the Residual XGBoost model."""
    global preprocessor, X_train_global, X_val_global, y_train_global, y_val_global, y_pred_base_val_global, residuals_train_global

    # Define hyperparameters to tune for the Residual Model
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50), # Smaller range for residual model
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    residual_model = xgb.XGBRegressor(**params)

    # Create a pipeline with preprocessing and the residual model
    residual_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', residual_model)])

    # Train the residual model on the original X_train, targeting the residuals of the base model
    residual_pipeline.fit(X_train_global, residuals_train_global)
    
    # Predict residuals on the validation set
    residuals_pred_val = residual_pipeline.predict(X_val_global)
    
    # Combine predictions
    final_y_pred_val = y_pred_base_val_global + residuals_pred_val
    
    rmse = np.sqrt(mean_squared_error(y_val_global, final_y_pred_val))

    return rmse

if __name__ == "__main__":
    processed_data_path = os.path.join('data', 'processed', 'processed_train.csv')
    tuning_log_file = os.path.join('logs', 'residual_xgboost_tuning_log.txt')

    os.makedirs(os.path.dirname(tuning_log_file), exist_ok=True)
    with open(tuning_log_file, 'w') as f:
        f.write("--- Residual XGBoost Model Tuning Log ---" + "\n")

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)

    X = df.drop('accident_risk', axis=1)
    y = df['accident_risk']

    X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training (X_train: {X_train_global.shape}, y_train: {y_train_global.shape}) and validation (X_val: {X_val_global.shape}, y_val: {y_val_global.shape}) sets.")

    preprocessor = create_preprocessor(X_train_global)
    print("Preprocessor fitted.")

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

    # --- Train the Base Model Once and Calculate Residuals ---
    print("\n--- Training Base XGBoost Model (once) ---")
    base_model = xgb.XGBRegressor(**best_xgboost_params)
    base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', base_model)])
    base_pipeline.fit(X_train_global, y_train_global)
    y_pred_base_train_global = base_pipeline.predict(X_train_global)
    y_pred_base_val_global = base_pipeline.predict(X_val_global)
    
    residuals_train_global = y_train_global - y_pred_base_train_global
    print(f"Base XGBoost RMSE on validation set: {np.sqrt(mean_squared_error(y_val_global, y_pred_base_val_global)):.4f}")
    print(f"Calculated residuals for training data. Mean: {np.mean(residuals_train_global):.4f}, Std: {np.std(residuals_train_global):.4f}")

    print("\nStarting Optuna study for Residual XGBoost model...")
    study = optuna.create_study(direction="minimize", study_name="residual_xgboost_tuning")
    study.optimize(objective, n_trials=50)

    print("\nOptuna study finished.")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    with open(tuning_log_file, 'a') as f:
        f.write(f"\nBest trial number: {study.best_trial.number}\n")
        f.write(f"Best RMSE: {study.best_value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Tuning results logged to {tuning_log_file}")