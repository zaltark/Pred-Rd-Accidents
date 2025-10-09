import pandas as pd
import numpy as np
import os
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import argparse

# Define global variables for preprocessor and data split to avoid re-creating them in each trial
preprocessor = None
X_train_global, X_val_global, y_train_global, y_val_global = None, None, None, None

def create_preprocessor(X):
    """Creates and fits the ColumnTransformer for preprocessing."""
    numerical_features = ['high_risk_curvature_interaction', 'Visibility_Score', 'low_speed_x_Visibility_Score']
    boolean_features = ['public_road', 'is_unsigned_urban_road', 'is_accident_hotspot', 'is_urban_evening_clear_weather', 'holiday_x_time_of_day_x_bright', 'low_speed_x_road_type_urban', 'low_speed_x_lane_category_two_lanes', 'is_moderate_visibility_overprediction_zone']
    categorical_features = ['road_type', 'weather', 'time_of_day', 'school_season_x_time_of_day', 'speed_zone', 'lane_category', 'accident_hotspot_x_weather']

    # Ensure boolean features are treated as numerical (0/1)
    for col in boolean_features:
        if col in X.columns:
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

def objective_xgb(trial):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 400, 600, step=50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 6, 8),
        'subsample': trial.suggest_uniform('subsample', 0.75, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.65, 0.80),
        'gamma': trial.suggest_loguniform('gamma', 1e-6, 1e-3),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-7, 1e-5),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-5),
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    return run_trial(model)

def objective_lgbm(trial):
    """Optuna objective function for LightGBM hyperparameter tuning."""
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 800, 1200, step=100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 40, 60),
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'subsample': trial.suggest_uniform('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 0.9),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-1),
        'random_state': 42,
        'n_jobs': -1
    }
    model = lgb.LGBMRegressor(**params)
    return run_trial(model)

def objective_catboost(trial):
    """Optuna objective function for CatBoost hyperparameter tuning."""
    params = {
        'objective': 'RMSE',
        'iterations': trial.suggest_int('iterations', 800, 1200, step=100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 100.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    return run_trial(model)

def run_trial(model):
    """Helper function to run a single trial."""
    global preprocessor, X_train_global, X_val_global, y_train_global, y_val_global
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    full_pipeline.fit(X_train_global, y_train_global)
    y_pred = full_pipeline.predict(X_val_global)
    rmse = np.sqrt(mean_squared_error(y_val_global, y_pred))
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--n_trials', type=int, default=50)
    args = parser.parse_args()

    model_name = args.model
    n_trials = args.n_trials

    processed_data_path = os.path.join('data', 'processed', 'processed_train.csv')
    tuning_log_file = os.path.join('logs', f'{model_name}_tuning_log.txt')

    os.makedirs(os.path.dirname(tuning_log_file), exist_ok=True)
    with open(tuning_log_file, 'w') as f:
        f.write(f"--- {model_name.upper()} Hyperparameter Tuning Log ---" + os.linesep)

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)

    X = df.drop('accident_risk', axis=1)
    y = df['accident_risk']

    X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training and validation sets.")

    preprocessor = create_preprocessor(X_train_global.copy())
    print("Preprocessor fitted.")

    objectives = {
        'xgboost': objective_xgb,
        'lightgbm': objective_lgbm,
        'catboost': objective_catboost
    }
    objective = objectives[model_name]

    print(f"Starting Optuna study for {model_name} with {n_trials} trials...")
    study = optuna.create_study(direction="minimize", study_name=f"{model_name}_tuning")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptuna study finished.")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    with open(tuning_log_file, 'a') as f:
        f.write(f"{os.linesep}Best trial number: {study.best_trial.number}{os.linesep}")
        f.write(f"Best RMSE: {study.best_value:.4f}{os.linesep}")
        f.write(f"Best hyperparameters:{os.linesep}")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}{os.linesep}")
    print(f"Tuning results logged to {tuning_log_file}")
