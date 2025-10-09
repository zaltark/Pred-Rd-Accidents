# Model Tuning and Testing Guidelines

These guidelines outline the primary evaluation metrics and considerations for tuning and testing machine learning models within this project.

## 1. Primary Evaluation Metric

**Root Mean Squared Error (RMSE)** is the primary metric for evaluating model performance. All tuning and final model selection decisions should prioritize minimizing RMSE, as this directly aligns with the competition's grading criteria.

## 2. Secondary Evaluation Metrics

While RMSE is primary, the following metrics should also be considered for a comprehensive understanding of model performance:

*   **Mean Absolute Error (MAE)**: Provides a measure of the average magnitude of errors, without considering their direction. It is less sensitive to outliers than RMSE.
*   **R-squared (R2)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides a measure of how well future samples are likely to be predicted by the model.

## 3. Tuning Strategy

*   **Baseline First**: Always establish a baseline performance with default model parameters before initiating any tuning efforts.
*   **Iterative Tuning**: Tuning should be an iterative process, focusing on one set of hyperparameters or a small group at a time.
*   **Cross-Validation**: Utilize cross-validation techniques to ensure robust evaluation and prevent overfitting during tuning.

## 4. Handling Class Imbalance

*   **Oversampling**: To address class imbalance, particularly for the `is_accident_hotspot` feature, we will experiment with oversampling the minority class (hotspots). A starting point will be to oversample the minority class to approximately 33% of the majority class size. This will be integrated into our cross-validation strategy.

## 5. Testing and Validation

*   **Hold-out Set**: Maintain a separate validation set (as established in `train_model.py`) for unbiased evaluation of the final model.
*   **Reproducibility**: Ensure all tuning and testing steps are reproducible (e.g., by setting random seeds).
*   **Robustness Testing**: For final model evaluation, consider running the training/evaluation process with multiple different random seeds or using repeated cross-validation. This assesses the model's stability and provides a more reliable estimate of its performance across various data splits.
*   **Performance Metrics**: Always log key performance metrics, including training and inference times, alongside evaluation scores (RMSE, MAE, R2). This provides a holistic view of model efficiency and effectiveness.
*   **Addressing Residual Patterns**: Systematically analyze residual plots for patterns (e.g., funnel shapes, systematic biases). Strategies to address these include:
    *   **Target Transformation**: Applying transformations (e.g., logit for [0,1] bounded targets) to stabilize error variance.
    *   **Advanced Feature Engineering**: Creating features that specifically target the problematic regions identified in residual analysis.
    *   **Ensemble/Stacking Methods**: Combining multiple models to smooth out systematic errors.
