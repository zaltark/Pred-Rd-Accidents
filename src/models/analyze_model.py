import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor # Assuming CatBoost is the chosen best model

def generate_model_analysis_html(plots_info):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Performance Analysis</title>
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1, h2, h3 { color: #333; }
            img { border: 1px solid #ddd; border-radius: 4px; padding: 5px; max-width: 100%; height: auto; }
            .container { max-width: 800px; margin: auto; }
            .plot { margin-bottom: 2em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Performance Analysis</h1>
            <p>This report visualizes the performance of the best-tuned model (CatBoost Regressor) to understand where it succeeds and fails.</p>
    """

    for plot in plots_info:
        html_content += f"""
            <div class="plot">
                <h2>{plot['title']}</h2>
                <img src="{plot['path']}" alt="{plot['title']}">
                <p>{plot['explanation']}</p>
            </div>
        """
    html_content += """
        </div>
    </body>
    </html>
    """
    with open('reports/model_analysis.html', 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    processed_data_path = os.path.join('data', 'processed', 'processed_train.csv')
    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    plots_info = []

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)

    X = df.drop('accident_risk', axis=1)
    y = df['accident_risk']

    # Identify feature types (same as in train_model.py)
    numerical_features = ['high_risk_curvature_interaction', 'Visibility_Score']
    boolean_features = ['public_road', 'holiday', 'is_unsigned_urban_road', 'is_accident_hotspot', 'is_urban_evening_clear_weather']
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
            ('cat', categorical_transformer, categorical_features + boolean_features)
        ],
        remainder='passthrough'
    )

    # Split data (same as in train_model.py)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Re-train the best XGBoost model with optimal hyperparameters ---
    # Best hyperparameters from logs/xgboost_tuning_log.txt (Trial 46)
    best_xgboost_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': 0.02092818251110864,
        'max_depth': 7,
        'subsample': 0.8334193820470637,
        'colsample_bytree': 0.7136119355067383,
        'gamma': 0.00021931557846934974,
        'reg_alpha': 4.313649801339041e-06,
        'reg_lambda': 5.981642474583213e-07,
        'random_state': 42,
        'n_jobs': -1
    }
    best_model = xgb.XGBRegressor(**best_xgboost_params)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', best_model)])
    print("Training best XGBoost model...")
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Retrained XGBoost RMSE on validation set: {rmse:.4f}")

    # --- 1. Predicted vs. Actual Plot ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # y=x line
    plt.xlabel("Actual Accident Risk")
    plt.ylabel("Predicted Accident Risk")
    plt.title("Predicted vs. Actual Accident Risk")
    plot_path = os.path.join(output_dir, 'predicted_vs_actual.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Predicted vs. Actual Accident Risk',
        'path': os.path.join('figures', 'predicted_vs_actual.png'),
        'explanation': 'This scatter plot shows the model\'s predictions against the true accident risk values. Points close to the red diagonal line indicate accurate predictions. Deviations from the line show where the model is over- or under-predicting.'
    })

    # --- 1.1 Predicted vs. Actual Plot (colored by Visibility Score) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_val, y=y_pred, hue=X_val['Visibility_Score'].astype('category'), alpha=0.3, palette='viridis')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # y=x line
    plt.xlabel("Actual Accident Risk")
    plt.ylabel("Predicted Accident Risk")
    plt.title("Predicted vs. Actual Accident Risk (by Visibility Score)")
    plt.legend(title='Visibility Score')
    plot_path = os.path.join(output_dir, 'predicted_vs_actual_visibility_score.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Predicted vs. Actual Accident Risk (by Visibility Score)',
        'path': os.path.join('figures', 'predicted_vs_actual_visibility_score.png'),
        'explanation': 'This plot shows predictions vs. actuals, with points colored by the `Visibility_Score`. This helps identify if the model\'s performance varies systematically across different visibility conditions.'
    })

    # --- 1.2 Predicted vs. Actual Plot (colored by Weather) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_val, y=y_pred, hue=X_val['weather'], alpha=0.3, palette='tab10')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # y=x line
    plt.xlabel("Actual Accident Risk")
    plt.ylabel("Predicted Accident Risk")
    plt.title("Predicted vs. Actual Accident Risk (by Weather)")
    plt.legend(title='Weather')
    plot_path = os.path.join(output_dir, 'predicted_vs_actual_weather.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Predicted vs. Actual Accident Risk (by Weather)',
        'path': os.path.join('figures', 'predicted_vs_actual_weather.png'),
        'explanation': 'This plot shows predictions vs. actuals, with points colored by the `weather` condition. This helps identify if the model\'s performance varies systematically across different weather conditions.'
    })

    # --- 2. Residual Plot ---
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Accident Risk")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plot_path = os.path.join(output_dir, 'residuals_plot.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Residual Plot',
        'path': os.path.join('figures', 'residuals_plot.png'),
        'explanation': 'This plot shows the distribution of errors (residuals) against the predicted values. Ideally, residuals should be randomly scattered around zero, with no discernible patterns. Any patterns suggest systematic errors by the model.'
    })

    # --- 2.1 Residual Plot (colored by Visibility Score) ---
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_pred, y=residuals, hue=X_val['Visibility_Score'].astype('category'), alpha=0.3, palette='viridis')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Accident Risk")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot (by Visibility Score)")
    plt.legend(title='Visibility Score')
    plot_path = os.path.join(output_dir, 'residuals_plot_visibility_score.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Residual Plot (by Visibility Score)',
        'path': os.path.join('figures', 'residuals_plot_visibility_score.png'),
        'explanation': 'This plot shows residuals vs. predictions, with points colored by the `Visibility_Score`. This helps identify if the model\'s systematic errors are concentrated within specific visibility conditions.'
    })

    # --- 2.2 Residual Plot (colored by Weather) ---
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_pred, y=residuals, hue=X_val['weather'], alpha=0.3, palette='tab10')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Accident Risk")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot (by Weather)")
    plt.legend(title='Weather')
    plot_path = os.path.join(output_dir, 'residuals_plot_weather.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Residual Plot (by Weather)',
        'path': os.path.join('figures', 'residuals_plot_weather.png'),
        'explanation': 'This plot shows residuals vs. predictions, with points colored by the `weather` condition. This helps identify if the model\'s systematic errors are concentrated within specific weather conditions.'
    })

    # --- 3. Error Distribution (KDE Plot of Residuals) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.title("Distribution of Residuals")
    plot_path = os.path.join(output_dir, 'residuals_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Distribution of Residuals',
        'path': os.path.join('figures', 'residuals_distribution.png'),
        'explanation': 'This histogram and KDE plot show the distribution of the model\'s errors. Ideally, it should be centered around zero and resemble a normal distribution. Skewness or multiple peaks indicate potential issues.'
    })

    # --- 4. Feature Importance Plot ---
    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features + boolean_features)
    all_feature_names = numerical_features + ohe_feature_names.tolist()

    feature_importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20) # Top 20 features

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    plots_info.append({
        'title': 'Top 20 Feature Importances',
        'path': os.path.join('figures', 'feature_importance.png'),
        'explanation': 'This plot shows the relative importance of the top 20 features as determined by the XGBoost model. Features with higher importance contribute more to the model\'s predictions.'
    })

    # --- Analysis of Moderate Risk Predictions (Predicted Risk 0.2 - 0.6) ---
    moderate_risk_threshold_low = 0.2
    moderate_risk_threshold_high = 0.6

    # Filter validation data for moderate risk predictions
    moderate_risk_indices = (y_pred >= moderate_risk_threshold_low) & (y_pred <= moderate_risk_threshold_high)
    X_val_moderate = X_val[moderate_risk_indices]
    y_val_moderate = y_val[moderate_risk_indices]
    y_pred_moderate = y_pred[moderate_risk_indices]
    residuals_moderate = y_val_moderate - y_pred_moderate

    if not X_val_moderate.empty:
        plots_info.append({
            'title': 'Analysis of Moderate Risk Predictions (Predicted Risk 0.2 - 0.6)',
            'path': '', # No specific image for this title, just a section header
            'explanation': f'This section provides a targeted analysis of data points where the model predicted accident risk between {moderate_risk_threshold_low} and {moderate_risk_threshold_high}. This is the range where the model showed a funnel shape in its residuals.'
        })

        # Key features to analyze in this range
        key_features = ['road_type', 'speed_zone', 'Visibility_Score', 'lane_category', 'accident_hotspot_x_weather']

        for feature in key_features:
            # Distribution of Feature in Moderate Risk Range
            plt.figure(figsize=(10, 6))
            if X_val_moderate[feature].dtype == 'object' or X_val_moderate[feature].dtype == 'category':
                sns.countplot(y=X_val_moderate[feature], order=X_val_moderate[feature].value_counts().index)
            else: # Numerical
                sns.histplot(X_val_moderate[feature], kde=True)
            plt.title(f'Distribution of {feature} in Moderate Risk Predictions')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'moderate_risk_dist_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            plots_info.append({
                'title': f'Distribution of {feature} in Moderate Risk Predictions',
                'path': os.path.join('figures', f'moderate_risk_dist_{feature}.png'),
                'explanation': f'This plot shows the distribution of the `{feature}` feature specifically for data points where the predicted risk is between {moderate_risk_threshold_low} and {moderate_risk_threshold_high}.'
            })

            # Residuals vs. Feature in Moderate Risk Range
            plt.figure(figsize=(10, 6))
            if X_val_moderate[feature].dtype == 'object' or X_val_moderate[feature].dtype == 'category':
                sns.boxplot(x=X_val_moderate[feature], y=residuals_moderate)
            else: # Numerical
                sns.scatterplot(x=X_val_moderate[feature], y=residuals_moderate, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.title(f'Residuals vs. {feature} in Moderate Risk Predictions')
            plt.xlabel(feature)
            plt.ylabel('Residuals')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'moderate_risk_residuals_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            plots_info.append({
                'title': f'Residuals vs. {feature} in Moderate Risk Predictions',
                'path': os.path.join('figures', f'moderate_risk_residuals_{feature}.png'),
                'explanation': f'This plot shows the residuals against the `{feature}` feature, specifically for data points where the predicted risk is between {moderate_risk_threshold_low} and {moderate_risk_threshold_high}. Patterns here indicate features contributing to systematic errors in this range.'
            })
    else:
        plots_info.append({
            'title': 'Analysis of Moderate Risk Predictions (Predicted Risk 0.2 - 0.6)',
            'path': '',
            'explanation': 'No data points found with predicted risk between 0.2 and 0.6 for targeted analysis.'
        })

    # --- Analysis of Model Failures (Absolute Residual > 0.1) ---
    failure_threshold = 0.1

    # Filter validation data for failures
    failure_indices = (np.abs(residuals) > failure_threshold)
    X_val_failures = X_val[failure_indices]
    y_val_failures = y_val[failure_indices]
    y_pred_failures = y_pred[failure_indices]
    residuals_failures = residuals[failure_indices]

    if not X_val_failures.empty:
        plots_info.append({
            'title': f'Analysis of Model Failures (Absolute Residual > {failure_threshold})',
            'path': '', # Section header
            'explanation': f'This section provides a targeted analysis of data points where the model made significant errors (absolute residual greater than {failure_threshold}).'
        })

        # Key features to analyze in this range
        key_features = ['road_type', 'speed_zone', 'Visibility_Score', 'lane_category', 'accident_hotspot_x_weather']

        for feature in key_features:
            # Distribution of Feature in Failure Cases
            plt.figure(figsize=(10, 6))
            if X_val_failures[feature].dtype == 'object' or X_val_failures[feature].dtype == 'category':
                sns.countplot(y=X_val_failures[feature], order=X_val_failures[feature].value_counts().index)
            else: # Numerical
                sns.histplot(X_val_failures[feature], kde=True)
            plt.title(f'Distribution of {feature} in Failure Cases')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'failure_dist_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            plots_info.append({
                'title': f'Distribution of {feature} in Failure Cases',
                'path': os.path.join('figures', f'failure_dist_{feature}.png'),
                'explanation': f'This plot shows the distribution of the `{feature}` feature specifically for data points where the model made significant errors (absolute residual > {failure_threshold}).'
            })

            # Residuals vs. Feature in Failure Cases
            plt.figure(figsize=(10, 6))
            if X_val_failures[feature].dtype == 'object' or X_val_failures[feature].dtype == 'category':
                sns.boxplot(x=X_val_failures[feature], y=residuals_failures)
            else: # Numerical
                sns.scatterplot(x=X_val_failures[feature], y=residuals_failures, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.title(f'Residuals vs. {feature} in Failure Cases')
            plt.xlabel(feature)
            plt.ylabel('Residuals')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'failure_residuals_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            plots_info.append({
                'title': f'Residuals vs. {feature} in Failure Cases',
                'path': os.path.join('figures', f'failure_residuals_{feature}.png'),
                'explanation': f'This plot shows the residuals against the `{feature}` feature, specifically for data points where the model made significant errors (absolute residual > {failure_threshold}). Patterns here indicate features contributing to systematic errors in these failure cases.'
            })
    else:
        plots_info.append({
            'title': f'Analysis of Model Failures (Absolute Residual > {failure_threshold})',
            'path': '',
            'explanation': f'No data points found with absolute residual greater than {failure_threshold} for targeted analysis.'
        })

    # Generate HTML report
    generate_model_analysis_html(plots_info)
    print(f"Model analysis report generated at reports/model_analysis.html")
